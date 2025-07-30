//===-- NetworkSymbolManager.cpp ----------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "lldb/Symbol/NetworkSymbolManager.h"
#include "lldb/API/SBCommandInterpreter.h"
#include "lldb/API/SBCommandReturnObject.h"
#include "lldb/API/SBDebugger.h"
#include "lldb/Core/Debugger.h"
#include "lldb/Interpreter/CommandInterpreter.h"
#include "lldb/Interpreter/CommandReturnObject.h"
#include "lldb/Utility/LLDBLog.h"
#include "lldb/Utility/Log.h"
#include "llvm/Support/FormatVariadic.h"

#include <algorithm>
#include <sstream>
#include <string>

using namespace lldb_private;
using namespace llvm;

NetworkSymbolManager::NetworkSymbolManager() = default;
NetworkSymbolManager::~NetworkSymbolManager() = default;

double NetworkSymbolManager::ServerAvailability::GetReliabilityScore() const {
  if (success_count + failure_count == 0)
    return 0.0;

  return static_cast<double>(success_count) / (success_count + failure_count);
}

bool NetworkSymbolManager::ServerAvailability::IsValid(std::chrono::minutes ttl) const {
  auto now = std::chrono::steady_clock::now();
  auto age = std::chrono::duration_cast<std::chrono::minutes>(now - last_checked);
  return age < ttl;
}

Status NetworkSymbolManager::Configure(const Configuration &config, bool respect_user_settings) {
  // Validate configuration first
  auto validation_error = ValidateConfiguration(config);
  if (!validation_error.Success())
    return validation_error;

  std::lock_guard<std::mutex> lock(settings_mutex_);
  config_ = config;

  Log *log = GetLog(LLDBLog::Symbols);
  LLDB_LOG(log,
           "NetworkSymbolManager configured: debuginfod_timeout={0}ms, "
           "symbol_server_timeout={1}ms, disable_network={2}, "
           "respect_user_settings={3}",
           config_.debuginfod_timeout_ms, config_.symbol_server_timeout_ms,
           config_.disable_network_symbols, respect_user_settings);

  return Status();
}

bool NetworkSymbolManager::IsServerResponsive(StringRef server_url,
                                              std::chrono::milliseconds test_timeout) {
  std::lock_guard<std::mutex> lock(server_cache_mutex_);

  std::string url_key = server_url.str();
  auto it = server_availability_cache_.find(url_key);

  // Check if we have valid cached information
  if (it != server_availability_cache_.end()) {
    const ServerAvailability &availability = it->second;

    // If server is temporarily blacklisted, don't even try
    if (availability.IsTemporarilyBlacklisted()) {
      Log *log = GetLog(LLDBLog::Symbols);
      LLDB_LOG(log, "Server {0} is temporarily blacklisted ({1} consecutive failures)",
               server_url, availability.consecutive_failures);
      return false;
    }

    // Use cached result if still valid
    if (availability.IsValid(std::chrono::minutes(config_.cache_ttl_minutes))) {
      return availability.is_responsive;
    }
  }

  // Release lock during network test to avoid blocking other operations
  server_cache_mutex_.unlock();

  auto start_time = std::chrono::steady_clock::now();
  bool responsive = TestServerConnectivity(server_url, test_timeout);
  auto end_time = std::chrono::steady_clock::now();
  auto response_time = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);

  // Re-acquire lock to update cache
  server_cache_mutex_.lock();

  ServerAvailability &availability = server_availability_cache_[url_key];
  availability.is_responsive = responsive;
  availability.last_checked = std::chrono::steady_clock::now();

  if (responsive) {
    availability.success_count++;
    // Update average response time using exponential moving average
    if (availability.average_response_time.count() == 0) {
      availability.average_response_time = response_time;
    } else {
      auto new_avg = (availability.average_response_time * 7 + response_time * 3) / 10;
      availability.average_response_time = new_avg;
    }
  } else {
    availability.failure_count++;
  }

  return responsive;
}

std::chrono::milliseconds NetworkSymbolManager::GetAdaptiveTimeout(StringRef server_url) const {
  if (!config_.enable_adaptive_timeouts) {
    return std::chrono::milliseconds(config_.debuginfod_timeout_ms);
  }

  std::lock_guard<std::mutex> lock(server_cache_mutex_);

  auto it = server_availability_cache_.find(server_url.str());
  if (it == server_availability_cache_.end()) {
    // No history, use default timeout
    return std::chrono::milliseconds(config_.debuginfod_timeout_ms);
  }

  const ServerAvailability &availability = it->second;

  // Base timeout on server reliability and average response time
  double reliability = availability.GetReliabilityScore();
  auto base_timeout = std::chrono::milliseconds(config_.debuginfod_timeout_ms);

  if (reliability > 0.8 && availability.average_response_time.count() > 0) {
    // Reliable server: use 2x average response time, but cap at configured timeout
    auto adaptive_timeout = availability.average_response_time * 2;
    return std::min(adaptive_timeout, base_timeout);
  } else if (reliability < 0.3) {
    // Unreliable server: use shorter timeout to fail fast
    return base_timeout / 2;
  }

  return base_timeout;
}

void NetworkSymbolManager::RecordServerResponse(StringRef server_url,
                                               std::chrono::milliseconds response_time,
                                               bool success) {
  std::lock_guard<std::mutex> lock(server_cache_mutex_);

  ServerAvailability &availability = server_availability_cache_[server_url.str()];
  availability.last_checked = std::chrono::steady_clock::now();

  if (success) {
    availability.success_count++;
    availability.is_responsive = true;
    availability.consecutive_failures = 0; // Reset failure streak

    // Update average response time using exponential moving average
    if (availability.average_response_time.count() == 0) {
      availability.average_response_time = response_time;
    } else {
      auto new_avg = (availability.average_response_time * 7 + response_time * 3) / 10;
      availability.average_response_time = new_avg;
    }
  } else {
    availability.failure_count++;
    availability.consecutive_failures++;
    availability.is_responsive = false;

    // Record first failure time for backoff calculation
    if (availability.consecutive_failures == 1) {
      availability.first_failure_time = std::chrono::steady_clock::now();
    }
  }
}

void NetworkSymbolManager::ClearServerCache() {
  std::lock_guard<std::mutex> lock(server_cache_mutex_);
  server_availability_cache_.clear();
}

Status NetworkSymbolManager::ApplyOptimizations(Debugger &debugger) {
  std::lock_guard<std::mutex> lock(settings_mutex_);

  if (settings_applied_) {
    return Status("Network symbol optimizations already applied");
  }

  Log *log = GetLog(LLDBLog::Symbols);

  // Query and backup existing settings before making changes
  std::vector<std::pair<std::string, std::string>> settings_to_apply;

  if (config_.disable_network_symbols) {
    settings_to_apply.emplace_back("symbols.enable-external-lookup", "false");
  } else {
    // Apply timeout optimizations
    settings_to_apply.emplace_back(
        "plugin.symbol-locator.debuginfod.timeout",
        std::to_string(config_.debuginfod_timeout_ms / 1000));
  }

  // Apply each setting with backup
  for (const auto &[setting_name, setting_value] : settings_to_apply) {
    auto error = ApplySetting(debugger, setting_name, setting_value);
    if (!error.Success()) {
      // Restore any settings we've already applied
      RestoreOriginalSettings(debugger);
      return error;
    }
  }

  settings_applied_ = true;
  LLDB_LOG(log, "NetworkSymbolManager optimizations applied successfully");

  return Status();
}

Status NetworkSymbolManager::ApplyOptimizations(lldb::SBDebugger &debugger) {
  // Bridge method: Use SBCommandInterpreter to apply settings
  std::lock_guard<std::mutex> lock(settings_mutex_);

  if (settings_applied_) {
    return Status("Network symbol optimizations already applied");
  }

  Log *log = GetLog(LLDBLog::Symbols);

  // Use SBCommandInterpreter to apply settings
  lldb::SBCommandInterpreter interpreter = debugger.GetCommandInterpreter();
  lldb::SBCommandReturnObject result;

  std::vector<std::pair<std::string, std::string>> settings_to_apply;

  if (config_.disable_network_symbols) {
    settings_to_apply.emplace_back("symbols.enable-external-lookup", "false");
  } else {
    // Apply timeout optimizations
    settings_to_apply.emplace_back(
        "plugin.symbol-locator.debuginfod.timeout",
        std::to_string(config_.debuginfod_timeout_ms / 1000));
  }

  // Apply each setting with backup
  for (const auto &[setting_name, setting_value] : settings_to_apply) {
    // First backup the existing setting
    std::string show_command = llvm::formatv("settings show {0}", setting_name).str();
    interpreter.HandleCommand(show_command.c_str(), result);

    if (result.Succeeded()) {
      original_settings_[setting_name] = result.GetOutput();
    }

    // Apply the new setting
    result.Clear();
    std::string set_command = llvm::formatv("settings set {0} {1}", setting_name, setting_value).str();
    interpreter.HandleCommand(set_command.c_str(), result);

    if (!result.Succeeded()) {
      // Restore any settings we've already applied
      RestoreOriginalSettings(debugger);
      return Status(llvm::formatv("Failed to apply setting {0}: {1}", setting_name, result.GetError()).str());
    }
  }

  settings_applied_ = true;
  LLDB_LOG(log, "NetworkSymbolManager optimizations applied successfully");

  return Status();
}

Status NetworkSymbolManager::RestoreOriginalSettings(Debugger &debugger) {
  std::lock_guard<std::mutex> lock(settings_mutex_);

  if (!settings_applied_) {
    return Status(); // Nothing to restore
  }

  CommandInterpreter &interpreter = debugger.GetCommandInterpreter();
  CommandReturnObject result(false);

  for (const auto &[setting_name, original_value] : original_settings_) {
    std::string command = formatv("settings set {0} {1}", setting_name, original_value);
    interpreter.HandleCommand(command.c_str(), eLazyBoolNo, result);

    if (!result.Succeeded()) {
      Log *log = GetLog(LLDBLog::Symbols);
      LLDB_LOG(log, "Failed to restore setting {0}: {1}", setting_name, result.GetErrorString());
    }
  }

  original_settings_.clear();
  settings_applied_ = false;

  return Status();
}

Status NetworkSymbolManager::RestoreOriginalSettings(lldb::SBDebugger &debugger) {
  // Bridge method: Use SBCommandInterpreter to restore settings
  std::lock_guard<std::mutex> lock(settings_mutex_);

  if (!settings_applied_) {
    return Status(); // Nothing to restore
  }

  lldb::SBCommandInterpreter interpreter = debugger.GetCommandInterpreter();
  lldb::SBCommandReturnObject result;

  for (const auto &[setting_name, original_value] : original_settings_) {
    std::string command = llvm::formatv("settings set {0} {1}", setting_name, original_value).str();
    interpreter.HandleCommand(command.c_str(), result);

    if (!result.Succeeded()) {
      Log *log = GetLog(LLDBLog::Symbols);
      LLDB_LOG(log, "Failed to restore setting {0}: {1}", setting_name, result.GetError());
    }
    result.Clear();
  }

  original_settings_.clear();
  settings_applied_ = false;

  return Status();
}

Status NetworkSymbolManager::ValidateConfiguration(const Configuration &config) {
  // Validate timeout ranges (0-60 seconds)
  if (config.debuginfod_timeout_ms > 60000) {
    return Status("debuginfod_timeout_ms must be <= 60000");
  }

  if (config.symbol_server_timeout_ms > 60000) {
    return Status("symbol_server_timeout_ms must be <= 60000");
  }

  if (config.cache_ttl_minutes == 0 || config.cache_ttl_minutes > 60) {
    return Status("cache_ttl_minutes must be between 1 and 60");
  }

  return Status();
}

bool NetworkSymbolManager::TestServerConnectivity(StringRef server_url,
                                                  std::chrono::milliseconds timeout) {
  Log *log = GetLog(LLDBLog::Symbols);

  // Simple connectivity test - for now we'll use a basic approach
  // since LLVM doesn't have HTTPClient available in all builds
  LLDB_LOG(log, "Testing connectivity to {0} with timeout {1}ms",
           server_url, timeout.count());

  // TODO: Implement actual network connectivity test using platform APIs
  // For now, assume connectivity exists and let LLDB's existing mechanisms
  // handle network timeouts and failures
  return true;

}

Status NetworkSymbolManager::QueryExistingSetting(Debugger &debugger,
                                                  StringRef setting_name,
                                                  std::string &value) {
  CommandInterpreter &interpreter = debugger.GetCommandInterpreter();
  CommandReturnObject result(false);

  std::string command = llvm::formatv("settings show {0}", setting_name).str();
  interpreter.HandleCommand(command.c_str(), eLazyBoolNo, result);

  if (result.Succeeded()) {
    value = std::string(result.GetOutputString());
    return Status();
  }

  return Status(llvm::formatv("Failed to query setting: {0}", setting_name).str());
}

Status NetworkSymbolManager::ApplySetting(Debugger &debugger,
                                          StringRef setting_name,
                                          StringRef value) {
  // First, backup the existing setting
  std::string original_value;
  auto error = QueryExistingSetting(debugger, setting_name, original_value);
  if (!error.Success()) {
    return error;
  }

  original_settings_[setting_name.str()] = original_value;

  // Apply the new setting
  CommandInterpreter &interpreter = debugger.GetCommandInterpreter();
  CommandReturnObject result(false);

  std::string command = llvm::formatv("settings set {0} {1}", setting_name, value).str();
  interpreter.HandleCommand(command.c_str(), eLazyBoolNo, result);

  if (!result.Succeeded()) {
    return Status(llvm::formatv("Failed to apply setting {0}: {1}", setting_name, result.GetErrorString()).str());
  }

  return Status();
}

bool NetworkSymbolManager::ShouldDisableNetworkSymbols() const {
  return config_.disable_network_symbols;
}

std::chrono::milliseconds NetworkSymbolManager::GetRecommendedDebuginfodTimeout() const {
  if (config_.disable_network_symbols) {
    return std::chrono::milliseconds(0);
  }

  return std::chrono::milliseconds(config_.debuginfod_timeout_ms);
}

std::chrono::milliseconds NetworkSymbolManager::GetRecommendedSymbolServerTimeout() const {
  if (config_.disable_network_symbols) {
    return std::chrono::milliseconds(0);
  }

  return std::chrono::milliseconds(config_.symbol_server_timeout_ms);
}

// ServerAvailability method implementations

bool NetworkSymbolManager::ServerAvailability::IsTemporarilyBlacklisted() const {
  // Blacklist servers with 3+ consecutive failures
  if (consecutive_failures < 3) {
    return false;
  }

  // Check if enough time has passed for backoff
  auto now = std::chrono::steady_clock::now();
  auto time_since_first_failure = std::chrono::duration_cast<std::chrono::minutes>(
      now - first_failure_time);

  return time_since_first_failure < GetBackoffTime();
}

std::chrono::minutes NetworkSymbolManager::ServerAvailability::GetBackoffTime() const {
  // Exponential backoff: 1, 2, 4, 8, 16 minutes (capped at 16)
  uint32_t backoff_minutes = std::min(16u, 1u << (consecutive_failures - 1));
  return std::chrono::minutes(backoff_minutes);
}



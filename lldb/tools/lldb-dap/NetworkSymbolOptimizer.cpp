//===-- NetworkSymbolOptimizer.cpp --------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "NetworkSymbolOptimizer.h"
#include "lldb/API/SBCommandInterpreter.h"
#include "lldb/API/SBCommandReturnObject.h"
#include "lldb/API/SBDebugger.h"
#include "lldb/Core/Debugger.h"
#include "lldb/Utility/LLDBLog.h"
#include "lldb/Utility/Log.h"

using namespace lldb_dap;
using namespace lldb_private;
using namespace lldb;

NetworkSymbolOptimizer::NetworkSymbolOptimizer()
    : manager_(std::make_unique<NetworkSymbolManager>()) {}

NetworkSymbolOptimizer::~NetworkSymbolOptimizer() = default;

Status NetworkSymbolOptimizer::Configure(const DAPConfiguration &dap_config,
                                         SBDebugger &debugger) {
  // Store the force optimizations flag
  force_optimizations_ = dap_config.force_optimizations;

  // Convert DAP configuration to NetworkSymbolManager configuration
  auto manager_config = ConvertDAPConfiguration(dap_config);

  // Configure the underlying manager with user settings respect enabled
  return manager_->Configure(manager_config, /*respect_user_settings=*/true);
}

Status NetworkSymbolOptimizer::ApplyOptimizations(SBDebugger &debugger) {
  if (optimizations_applied_) {
    return Status("Optimizations already applied");
  }

  // Get the current configuration to check if optimizations are enabled
  auto config = manager_->GetConfiguration();

  // Only proceed if optimizations are explicitly enabled (opt-in model)
  if (!config.enable_server_caching && !config.enable_adaptive_timeouts) {
    Log *log = GetLog(LLDBLog::Symbols);
    LLDB_LOG(log, "NetworkSymbolOptimizer: Optimizations not enabled, skipping");
    return Status(); // Success, but no action taken
  }

  // Check if user has configured relevant settings (unless force is enabled)
  if (!force_optimizations_) {
    std::vector<std::string> settings_to_check = {
      "symbols.enable-external-lookup",
      "symbols.enable-background-lookup",
      "symbols.auto-download",
      "symbols.load-on-demand",
      "plugin.symbol-locator.debuginfod.timeout",
      "plugin.symbol-locator.debuginfod.server-urls"
    };

    for (const auto &setting : settings_to_check) {
      if (IsUserConfigured(debugger, setting)) {
        Log *log = GetLog(LLDBLog::Symbols);
        LLDB_LOG(log, "NetworkSymbolOptimizer: User has configured symbol setting '{0}', "
                      "skipping automatic optimizations to respect user preferences", setting);
        return Status(); // Success, but no action taken
      }
    }
  }

  // Apply optimizations using the SBDebugger interface
  // Note: We'll need to implement a bridge method in NetworkSymbolManager
  // that accepts SBDebugger instead of internal Debugger*
  auto status = manager_->ApplyOptimizations(debugger);
  if (status.Success()) {
    optimizations_applied_ = true;
  }

  return status;
}

Status NetworkSymbolOptimizer::RestoreSettings(SBDebugger &debugger) {
  if (!optimizations_applied_) {
    return Status(); // Nothing to restore
  }

  auto status = manager_->RestoreOriginalSettings(debugger);
  if (status.Success()) {
    optimizations_applied_ = false;
  }

  return status;
}

bool NetworkSymbolOptimizer::ShouldDisableNetworkSymbols() const {
  return manager_->ShouldDisableNetworkSymbols();
}

uint32_t NetworkSymbolOptimizer::GetRecommendedDebuginfodTimeoutMs() const {
  auto timeout = manager_->GetRecommendedDebuginfodTimeout();
  return static_cast<uint32_t>(timeout.count());
}

uint32_t NetworkSymbolOptimizer::GetRecommendedSymbolServerTimeoutMs() const {
  auto timeout = manager_->GetRecommendedSymbolServerTimeout();
  return static_cast<uint32_t>(timeout.count());
}

bool NetworkSymbolOptimizer::IsServerResponsive(llvm::StringRef server_url) const {
  return manager_->IsServerResponsive(server_url);
}

void NetworkSymbolOptimizer::ClearServerCache() {
  manager_->ClearServerCache();
}

NetworkSymbolManager::Configuration
NetworkSymbolOptimizer::ConvertDAPConfiguration(const DAPConfiguration &dap_config) const {
  NetworkSymbolManager::Configuration config;

  // Only override defaults if explicitly specified in DAP configuration
  if (dap_config.debuginfod_timeout_ms > 0) {
    config.debuginfod_timeout_ms = dap_config.debuginfod_timeout_ms;
  }

  if (dap_config.symbol_server_timeout_ms > 0) {
    config.symbol_server_timeout_ms = dap_config.symbol_server_timeout_ms;
  }

  config.disable_network_symbols = dap_config.disable_network_symbols;

  // Opt-in model: only enable optimizations if explicitly requested
  config.enable_server_caching = dap_config.enable_optimizations;
  config.enable_adaptive_timeouts = dap_config.enable_optimizations;

  return config;
}

bool NetworkSymbolOptimizer::IsUserConfigured(SBDebugger &debugger,
                                             llvm::StringRef setting_name) const {
  SBCommandInterpreter interpreter = debugger.GetCommandInterpreter();
  SBCommandReturnObject result;

  // First try to get the setting value to see if it exists
  std::string show_command = "settings show " + setting_name.str();
  interpreter.HandleCommand(show_command.c_str(), result);

  if (!result.Succeeded()) {
    return false; // Setting doesn't exist or error occurred
  }

  std::string output = result.GetOutput();

  // Enhanced detection: Check multiple indicators of user configuration
  // 1. Setting explicitly shows as user-defined (not default)
  if (output.find("(default)") == std::string::npos) {
    return true;
  }

  // 2. Check if setting has been modified from its default value
  // Use "settings list" to get more detailed information
  result.Clear();
  std::string list_command = "settings list " + setting_name.str();
  interpreter.HandleCommand(list_command.c_str(), result);

  if (result.Succeeded()) {
    std::string list_output = result.GetOutput();
    // If the setting appears in the list output, it may have been explicitly set
    if (!list_output.empty() && list_output.find(setting_name.str()) != std::string::npos) {
      // Additional heuristic: check if the output contains value information
      // indicating the setting has been explicitly configured
      if (list_output.find("=") != std::string::npos) {
        return true;
      }
    }
  }

  return false; // Default to not user-configured if we can't determine
}

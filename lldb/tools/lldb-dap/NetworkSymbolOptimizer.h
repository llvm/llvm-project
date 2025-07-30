//===-- NetworkSymbolOptimizer.h ------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLDB_TOOLS_LLDB_DAP_NETWORKSYMBOLOPTIMIZER_H
#define LLDB_TOOLS_LLDB_DAP_NETWORKSYMBOLOPTIMIZER_H

#include "lldb/Symbol/NetworkSymbolManager.h"
#include "lldb/API/SBDebugger.h"
#include "lldb/lldb-forward.h"
#include "llvm/ADT/StringRef.h"

namespace lldb_dap {

/// NetworkSymbolOptimizer provides a DAP-layer interface to LLDB's
/// NetworkSymbolManager, implementing proper architectural separation
/// between protocol handling and symbol optimization logic.
///
/// This class addresses reviewer concerns about layer violations by:
/// 1. Delegating to appropriate LLDB subsystems via proper APIs
/// 2. Respecting user settings and providing opt-in behavior
/// 3. Separating network optimization from DAP protocol concerns
class NetworkSymbolOptimizer {
public:
  /// Configuration options that can be specified via DAP launch parameters
  struct DAPConfiguration {
    /// Optional debuginfod timeout in milliseconds (0 = use LLDB default)
    uint32_t debuginfod_timeout_ms = 0;

    /// Optional symbol server timeout in milliseconds (0 = use LLDB default)
    uint32_t symbol_server_timeout_ms = 0;

    /// Disable network symbol loading entirely
    bool disable_network_symbols = false;

    /// Enable intelligent optimizations (server caching, adaptive timeouts)
    /// This is now opt-in to respect LLDB's user control philosophy
    bool enable_optimizations = false;

    /// Force application of optimizations even if user has configured settings
    /// This should only be used when explicitly requested by the user
    bool force_optimizations = false;
  };

  NetworkSymbolOptimizer();
  ~NetworkSymbolOptimizer();

  /// Configure network symbol optimizations based on DAP launch parameters.
  /// This method respects existing user LLDB settings and provides opt-in
  /// behavior.
  lldb_private::Status Configure(const DAPConfiguration &dap_config,
                                 lldb::SBDebugger &debugger);

  /// Apply optimizations to the debugger instance.
  /// Only applies optimizations if user hasn't explicitly configured settings.
  lldb_private::Status ApplyOptimizations(lldb::SBDebugger &debugger);

  /// Restore original LLDB settings (called during cleanup)
  lldb_private::Status RestoreSettings(lldb::SBDebugger &debugger);

  /// Check if network symbol loading should be disabled
  bool ShouldDisableNetworkSymbols() const;

  /// Get recommended timeout for debuginfod operations
  uint32_t GetRecommendedDebuginfodTimeoutMs() const;

  /// Get recommended timeout for symbol server operations
  uint32_t GetRecommendedSymbolServerTimeoutMs() const;

  /// Test if a server is responsive (with caching)
  bool IsServerResponsive(llvm::StringRef server_url) const;

  /// Clear server availability cache (useful for network changes)
  void ClearServerCache();

private:
  /// The underlying network symbol manager
  std::unique_ptr<lldb_private::NetworkSymbolManager> manager_;

  /// Whether optimizations have been applied
  bool optimizations_applied_ = false;

  /// Whether to force optimizations even if user has configured settings
  bool force_optimizations_ = false;

  /// Convert DAP configuration to NetworkSymbolManager configuration.
  lldb_private::NetworkSymbolManager::Configuration
  ConvertDAPConfiguration(const DAPConfiguration &dap_config) const;

  /// Check if user has explicitly configured a setting.
  bool IsUserConfigured(lldb::SBDebugger &debugger,
                        llvm::StringRef setting_name) const;
};

} // namespace lldb_dap

#endif // LLDB_TOOLS_LLDB_DAP_NETWORKSYMBOLOPTIMIZER_H

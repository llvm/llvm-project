//===- CLOptionsSetup.h - Helpers to setup debug CL options -----*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DEBUG_CLOPTIONSSETUP_H
#define MLIR_DEBUG_CLOPTIONSSETUP_H

#include "mlir/Debug/BreakpointManagers/FileLineColLocBreakpointManager.h"
#include "llvm/ADT/StringRef.h"

#include <memory>

namespace mlir {
class MLIRContext;
namespace tracing {
class BreakpointManager;

class DebugConfig {
public:
  /// Register the options as global LLVM command line options.
  static void registerCLOptions();

  /// Create a new config with the default set from the CL options.
  static DebugConfig createFromCLOptions();

  ///
  /// Options.
  ///

  /// Enable the Debugger action hook: it makes a debugger (like gdb or lldb)
  /// able to intercept MLIR Actions.
  void enableDebuggerActionHook(bool enabled = true) {
    enableDebuggerActionHookFlag = enabled;
  }

  /// Return true if the debugger action hook is enabled.
  bool isDebuggerActionHookEnabled() const {
    return enableDebuggerActionHookFlag;
  }

  /// Set the filename to use for logging actions, use "-" for stdout.
  DebugConfig &logActionsTo(StringRef filename) {
    logActionsToFlag = filename;
    return *this;
  }
  /// Get the filename to use for logging actions.
  StringRef getLogActionsTo() const { return logActionsToFlag; }

  /// Get the filename to use for profiling actions.
  StringRef getProfileActionsTo() const { return profileActionsToFlag; }

  /// Set a location breakpoint manager to filter out action logging based on
  /// the attached IR location in the Action context. Ownership stays with the
  /// caller.
  void addLogActionLocFilter(tracing::BreakpointManager *breakpointManager) {
    logActionLocationFilter.push_back(breakpointManager);
  }

  /// Get the location breakpoint managers to use to filter out action logging.
  ArrayRef<tracing::BreakpointManager *> getLogActionsLocFilters() const {
    return logActionLocationFilter;
  }

protected:
  /// Enable the Debugger action hook: a debugger (like gdb or lldb) can
  /// intercept MLIR Actions.
  bool enableDebuggerActionHookFlag = false;

  /// Log action execution to the given file (or "-" for stdout)
  std::string logActionsToFlag;

  /// Profile action execution to the given file (or "-" for stdout)
  std::string profileActionsToFlag;

  /// Location Breakpoints to filter the action logging.
  std::vector<tracing::BreakpointManager *> logActionLocationFilter;
};

/// This is a RAII class that installs the debug handlers on the context
/// based on the provided configuration.
class InstallDebugHandler {
public:
  InstallDebugHandler(MLIRContext &context, const DebugConfig &config);
  ~InstallDebugHandler();

private:
  class Impl;
  std::unique_ptr<Impl> impl;
};

} // namespace tracing
} // namespace mlir

#endif // MLIR_DEBUG_CLOPTIONSSETUP_H

//===- ActionLogging.h -  Logging Actions *- C++ -*-==========================//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_TRACING_OBSERVERS_ACTIONLOGGING_H
#define MLIR_TRACING_OBSERVERS_ACTIONLOGGING_H

#include "mlir/Debug/BreakpointManager.h"
#include "mlir/Debug/ExecutionContext.h"
#include "mlir/IR/Action.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/raw_ostream.h"

namespace mlir {
namespace tracing {

/// This class defines an observer that print Actions before and after execution
/// on the provided stream.
struct ActionLogger : public ExecutionContext::Observer {
  ActionLogger(raw_ostream &os, bool printActions = true,
               bool printBreakpoints = true, bool printIRUnits = true)
      : os(os), printActions(printActions), printBreakpoints(printBreakpoints),
        printIRUnits(printIRUnits) {}

  void beforeExecute(const ActionActiveStack *action, Breakpoint *breakpoint,
                     bool willExecute) override;
  void afterExecute(const ActionActiveStack *action) override;

  /// If one of multiple breakpoint managers are set, only actions that are
  /// matching a breakpoint will be logged.
  void addBreakpointManager(const BreakpointManager *manager) {
    breakpointManagers.push_back(manager);
  }

private:
  /// Check if we should log this action or not.
  bool shouldLog(const ActionActiveStack *action);

  raw_ostream &os;
  bool printActions;
  bool printBreakpoints;
  bool printIRUnits;
  std::vector<const BreakpointManager *> breakpointManagers;
};

} // namespace tracing
} // namespace mlir

#endif // MLIR_TRACING_OBSERVERS_ACTIONLOGGING_H

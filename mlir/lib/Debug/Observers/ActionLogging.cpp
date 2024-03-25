//===- ActionLogging.cpp -  Logging Actions *- C++ -*-========================//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Debug/Observers/ActionLogging.h"
#include "mlir/Debug/BreakpointManager.h"
#include "mlir/IR/Action.h"
#include "llvm/Support/Threading.h"
#include "llvm/Support/raw_ostream.h"

using namespace mlir;
using namespace mlir::tracing;

//===----------------------------------------------------------------------===//
// ActionLogger
//===----------------------------------------------------------------------===//

bool ActionLogger::shouldLog(const ActionActiveStack *action) {
  // If some condition was set, we ensured it is met before logging.
  if (breakpointManagers.empty())
    return true;
  return llvm::any_of(breakpointManagers,
                      [&](const BreakpointManager *manager) {
                        return manager->match(action->getAction());
                      });
}

void ActionLogger::beforeExecute(const ActionActiveStack *action,
                                 Breakpoint *breakpoint, bool willExecute) {
  if (!shouldLog(action))
    return;
  SmallVector<char> name;
  llvm::get_thread_name(name);
  if (name.empty()) {
    llvm::raw_svector_ostream os(name);
    os << llvm::get_threadid();
  }
  os << "[thread " << name << "] ";
  if (willExecute)
    os << "begins ";
  else
    os << "skipping ";
  if (printBreakpoints) {
    if (breakpoint)
      os << "(on breakpoint: " << *breakpoint << ") ";
    else
      os << "(no breakpoint) ";
  }
  os << "Action ";
  if (printActions)
    action->getAction().print(os);
  else
    os << action->getAction().getTag();
  if (printIRUnits) {
    os << " (";
    interleaveComma(action->getAction().getContextIRUnits(), os);
    os << ")";
  }
  os << "`\n";
}

void ActionLogger::afterExecute(const ActionActiveStack *action) {
  if (!shouldLog(action))
    return;
  SmallVector<char> name;
  llvm::get_thread_name(name);
  if (name.empty()) {
    llvm::raw_svector_ostream os(name);
    os << llvm::get_threadid();
  }
  os << "[thread " << name << "] completed `" << action->getAction().getTag()
     << "`\n";
}

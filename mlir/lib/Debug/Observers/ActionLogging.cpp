//===- ActionLogging.cpp -  Logging Actions *- C++ -*-========================//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Debug/Observers/ActionLogging.h"
#include "llvm/Support/Threading.h"
#include <sstream>
#include <thread>

using namespace mlir;
using namespace mlir::tracing;

//===----------------------------------------------------------------------===//
// ActionLogger
//===----------------------------------------------------------------------===//

void ActionLogger::beforeExecute(const ActionActiveStack *action,
                                 Breakpoint *breakpoint, bool willExecute) {
  SmallVector<char> name;
  llvm::get_thread_name(name);
  os << "[thread " << name << "] ";
  if (willExecute)
    os << "begins ";
  else
    os << "skipping ";
  if (printBreakpoints) {
    if (breakpoint)
      os << " (on breakpoint: " << *breakpoint << ") ";
    else
      os << " (no breakpoint) ";
  }
  os << "Action ";
  if (printActions)
    action->getAction().print(os);
  else
    os << action->getAction().getTag();
  os << "`\n";
}

void ActionLogger::afterExecute(const ActionActiveStack *action) {
  SmallVector<char> name;
  llvm::get_thread_name(name);
  os << "[thread " << name << "] completed `" << action->getAction().getTag()
     << "`\n";
}

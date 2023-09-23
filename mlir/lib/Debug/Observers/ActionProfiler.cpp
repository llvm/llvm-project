//===- ActionProfiler.cpp -  Profiling Actions *- C++ -*-=====================//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Debug/Observers/ActionProfiler.h"
#include "mlir/Debug/BreakpointManager.h"
#include "mlir/IR/Action.h"
#include "mlir/Rewrite/PatternApplicator.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/Threading.h"
#include "llvm/Support/raw_ostream.h"
#include <chrono>

using namespace mlir;
using namespace mlir::tracing;

//===----------------------------------------------------------------------===//
// ActionProfiler
//===----------------------------------------------------------------------===//
void ActionProfiler::beforeExecute(const ActionActiveStack *action,
                                   Breakpoint *breakpoint, bool willExecute) {
  print(action, "B"); // begin event.
}

void ActionProfiler::afterExecute(const ActionActiveStack *action) {
  print(action, "E"); // end event.
}

// Print an event in JSON format.
void ActionProfiler::print(const ActionActiveStack *action,
                           llvm::StringRef phase) {
  if (printComma)
    os << ",\n";
  printComma = true;
  os << "{";
  os << R"("name": ")" << action->getAction().getTag() << "\", ";
  os << R"("cat": "PERF", )";
  os << R"("ph": ")" << phase << "\", ";
  os << R"("pid": 0, )";
  os << R"("tid": )" << llvm::get_threadid() << ", ";
  auto ts = std::chrono::steady_clock::now() - startTime;
  os << R"("ts": )"
     << std::chrono::duration_cast<std::chrono::microseconds>(ts).count();
  if (phase == "B") {
    os << R"(, "args": {)";
    os << R"("desc": ")";
    action->getAction().print(os);
    os << "\"}";
  }
  os << "}";
}

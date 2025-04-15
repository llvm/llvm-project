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
  // Create the event.
  std::string str;
  llvm::raw_string_ostream event(str);
  event << "{";
  event << R"("name": ")" << action->getAction().getTag() << "\", ";
  event << R"("cat": "PERF", )";
  event << R"("ph": ")" << phase << "\", ";
  event << R"("pid": 0, )";
  event << R"("tid": )" << llvm::get_threadid() << ", ";
  auto ts = std::chrono::steady_clock::now() - startTime;
  event << R"("ts": )"
        << std::chrono::duration_cast<std::chrono::microseconds>(ts).count();
  if (phase == "B") {
    event << R"(, "args": {)";
    event << R"("desc": ")";
    action->getAction().print(event);
    event << "\"}";
  }
  event << "}";

  // Print the event.
  std::lock_guard<std::mutex> guard(mutex);
  if (printComma)
    os << ",\n";
  printComma = true;
  os << str;
  os.flush();
}

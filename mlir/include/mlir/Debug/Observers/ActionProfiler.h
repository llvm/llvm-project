//===- ActionProfiler.h -  Profiling Actions *- C++ -*-=======================//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_TRACING_OBSERVERS_ACTIONPROFILER_H
#define MLIR_TRACING_OBSERVERS_ACTIONPROFILER_H

#include "mlir/Debug/ExecutionContext.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/raw_ostream.h"

#include <chrono>
#include <mutex>

namespace mlir {
namespace tracing {

/// This class defines an observer that profiles events before and after
/// execution on the provided stream. The events are stored in the Chrome trace
/// event format.
struct ActionProfiler : public ExecutionContext::Observer {
  ActionProfiler(raw_ostream &os)
      : os(os), startTime(std::chrono::steady_clock::now()) {
    os << "[";
  }

  ~ActionProfiler() override { os << "]"; }

  void beforeExecute(const ActionActiveStack *action, Breakpoint *breakpoint,
                     bool willExecute) override;
  void afterExecute(const ActionActiveStack *action) override;

private:
  void print(const ActionActiveStack *action, llvm::StringRef phase);

  raw_ostream &os;
  std::chrono::time_point<std::chrono::steady_clock> startTime;
  bool printComma = false;

  /// A mutex used to guard printing from multiple threads.
  std::mutex mutex;
};

} // namespace tracing
} // namespace mlir

#endif // MLIR_TRACING_OBSERVERS_ACTIONPROFILER_H

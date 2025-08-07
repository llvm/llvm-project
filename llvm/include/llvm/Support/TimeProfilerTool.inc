//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines command line options and helper class for tools that want
// to use time trace profiling. It is intended for inclusion in a tool's .cpp
// file.
//===----------------------------------------------------------------------===//

#ifndef LLVM_SUPPORT_TIMEROFILER_TOOL_INC
#define LLVM_SUPPORT_TIMEROFILER_TOOL_INC

#include "llvm/Support/CommandLine.h"
#include "llvm/Support/TimeProfiler.h"

static llvm::cl::opt<bool> TimeTrace("time-trace",
                                     llvm::cl::desc("Record time trace"));

static llvm::cl::opt<unsigned> TimeTraceGranularity(
    "time-trace-granularity",
    llvm::cl::desc(
        "Minimum time granularity (in microseconds) traced by time profiler"),
    llvm::cl::init(500), llvm::cl::Hidden);

static llvm::cl::opt<std::string>
    TimeTraceFile("time-trace-file",
                  llvm::cl::desc("Specify time trace file destination"),
                  llvm::cl::value_desc("filename"));

namespace {

/// The TimeTraceProfilerRAII is a helper class to initialize and write the time
/// trace profile, intended for use in various tools.
class TimeTraceProfilerRAII {
private:
  llvm::StringRef OutputFilename;

public:
  TimeTraceProfilerRAII(llvm::StringRef ProgramName,
                        llvm::StringRef OutputFilename)
      : OutputFilename(OutputFilename) {
    if (TimeTrace)
      llvm::timeTraceProfilerInitialize(TimeTraceGranularity, ProgramName);
  }
  ~TimeTraceProfilerRAII() {
    if (!TimeTrace)
      return;
    if (auto E = llvm::timeTraceProfilerWrite(TimeTraceFile, OutputFilename)) {
      llvm::logAllUnhandledErrors(std::move(E), llvm::errs());
      return;
    }
    llvm::timeTraceProfilerCleanup();
  }
};

} // namespace

#endif // LLVM_SUPPORT_TIMEROFILER_TOOL_INC

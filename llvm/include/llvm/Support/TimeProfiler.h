//===- llvm/Support/TimeProfiler.h - Hierarchical Time Profiler -*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_SUPPORT_TIMEPROFILER_H
#define LLVM_SUPPORT_TIMEPROFILER_H

#include "llvm/Support/Error.h"
#include "llvm/ADT/STLFunctionalExtras.h"

namespace llvm {

class raw_pwrite_stream;

struct TimeTraceProfiler;
TimeTraceProfiler *getTimeTraceProfilerInstance();

/// Initialize the time trace profiler.
/// This sets up the global \p TimeTraceProfilerInstance
/// variable to be the profiler instance.
void timeTraceProfilerInitialize(unsigned TimeTraceGranularity,
                                 StringRef ProcName);

/// Cleanup the time trace profiler, if it was initialized.
void timeTraceProfilerCleanup();

/// Finish a time trace profiler running on a worker thread.
void timeTraceProfilerFinishThread();

/// Is the time trace profiler enabled, i.e. initialized?
inline bool timeTraceProfilerEnabled() {
  return getTimeTraceProfilerInstance() != nullptr;
}

/// Write profiling data to output stream.
/// Data produced is JSON, in Chrome "Trace Event" format, see
/// https://docs.google.com/document/d/1CvAClvFfyA5R-PhYUmn5OOQtYMH4h6I0nSsKchNAySU/preview
void timeTraceProfilerWrite(raw_pwrite_stream &OS);

/// Write profiling data to a file.
/// The function will write to \p PreferredFileName if provided, if not
/// then will write to \p FallbackFileName appending .time-trace.
/// Returns a StringError indicating a failure if the function is
/// unable to open the file for writing.
Error timeTraceProfilerWrite(StringRef PreferredFileName,
                             StringRef FallbackFileName);

/// Manually begin a time section, with the given \p Name and \p Detail.
/// Profiler copies the string data, so the pointers can be given into
/// temporaries. Time sections can be hierarchical; every Begin must have a
/// matching End pair but they can nest.
void timeTraceProfilerBegin(StringRef Name, StringRef Detail);
void timeTraceProfilerBegin(StringRef Name,
                            llvm::function_ref<std::string()> Detail);

/// Manually end the last time section.
void timeTraceProfilerEnd();

/// The TimeTraceScope is a helper class to call the begin and end functions
/// of the time trace profiler.  When the object is constructed, it begins
/// the section; and when it is destroyed, it stops it. If the time profiler
/// is not initialized, the overhead is a single branch.
struct TimeTraceScope {

  TimeTraceScope() = delete;
  TimeTraceScope(const TimeTraceScope &) = delete;
  TimeTraceScope &operator=(const TimeTraceScope &) = delete;
  TimeTraceScope(TimeTraceScope &&) = delete;
  TimeTraceScope &operator=(TimeTraceScope &&) = delete;

  TimeTraceScope(StringRef Name) {
    if (getTimeTraceProfilerInstance() != nullptr)
      timeTraceProfilerBegin(Name, StringRef(""));
  }
  TimeTraceScope(StringRef Name, StringRef Detail) {
    if (getTimeTraceProfilerInstance() != nullptr)
      timeTraceProfilerBegin(Name, Detail);
  }
  TimeTraceScope(StringRef Name, llvm::function_ref<std::string()> Detail) {
    if (getTimeTraceProfilerInstance() != nullptr)
      timeTraceProfilerBegin(Name, Detail);
  }
  ~TimeTraceScope() {
    if (getTimeTraceProfilerInstance() != nullptr)
      timeTraceProfilerEnd();
  }
};

} // end namespace llvm

#endif

//===- llvm/Support/TimeProfiler.h - Hierarchical Time Profiler -*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_SUPPORT_TIME_PROFILER_H
#define LLVM_SUPPORT_TIME_PROFILER_H

#include "llvm/Support/raw_ostream.h"

namespace llvm {

struct TimeTraceProfiler;
extern TimeTraceProfiler *TimeTraceProfilerInstance;

/// Initialize the time trace profiler.
/// This sets up the global \p TimeTraceProfilerInstance
/// variable to be the profiler instance.
void timeTraceProfilerInitialize();

/// Cleanup the time trace profiler, if it was initialized.
void timeTraceProfilerCleanup();

/// Is the time trace profiler enabled, i.e. initialized?
inline bool timeTraceProfilerEnabled() {
  return TimeTraceProfilerInstance != nullptr;
}

/// Write profiling data to output file.
/// Data produced is JSON, in Chrome "Trace Event" format, see
/// https://docs.google.com/document/d/1CvAClvFfyA5R-PhYUmn5OOQtYMH4h6I0nSsKchNAySU/preview
void timeTraceProfilerWrite(std::unique_ptr<raw_pwrite_stream> &OS);

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
  TimeTraceScope(StringRef Name, StringRef Detail) {
    if (TimeTraceProfilerInstance != nullptr)
      timeTraceProfilerBegin(Name, Detail);
  }
  TimeTraceScope(StringRef Name, llvm::function_ref<std::string()> Detail) {
    if (TimeTraceProfilerInstance != nullptr)
      timeTraceProfilerBegin(Name, Detail);
  }
  ~TimeTraceScope() {
    if (TimeTraceProfilerInstance != nullptr)
      timeTraceProfilerEnd();
  }
};

} // end namespace llvm

#endif

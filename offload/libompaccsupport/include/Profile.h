//===-- Profile.h - Target independent OpenMP target RTL --------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Macros to provide profile support via LLVM's time profiler.
//
//===----------------------------------------------------------------------===//

#ifndef OMPTARGET_SHARED_PROFILE_H
#define OMPTARGET_SHARED_PROFILE_H

#include "Shared/Debug.h"
#include "Shared/EnvironmentVar.h"

#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/TimeProfiler.h"

/// Class that holds the singleton profiler and allows to start/end events.
class Profiler {

  Profiler() {
    if (!ProfileTraceFile.isPresent())
      return;

    // TODO: Add an alias without LIBOMPTARGET
    // Flag to modify the profile granularity (in us).
    Int32Envar ProfileGranularity =
        Int32Envar("LIBOMPTARGET_PROFILE_GRANULARITY", 500);

    llvm::timeTraceProfilerInitialize(ProfileGranularity /*us=*/,
                                      "libomptarget");
  }

  ~Profiler() {
    if (!ProfileTraceFile.isPresent())
      return;

    if (auto Err = llvm::timeTraceProfilerWrite(ProfileTraceFile.get(), "-"))
      REPORT() << "Error writing out the time trace: "
               << llvm::toString(std::move(Err)).c_str() << "\n";

    llvm::timeTraceProfilerCleanup();
  }

  // TODO: Add an alias without LIBOMPTARGET
  /// Flag to enable profiling which also specifies the file profile information
  /// is stored in.
  StringEnvar ProfileTraceFile = StringEnvar("LIBOMPTARGET_PROFILE");

public:
  static Profiler &get() {
    static Profiler P;
    return P;
  }

  /// Manually begin a time section, with the given \p Name and \p Detail.
  /// Profiler copies the string data, so the pointers can be given into
  /// temporaries. Time sections can be hierarchical; every Begin must have a
  /// matching End pair but they can nest.
  void beginSection(llvm::StringRef Name, llvm::StringRef Detail) {
    llvm::timeTraceProfilerBegin(Name, Detail);
  }
  void beginSection(llvm::StringRef Name,
                    llvm::function_ref<std::string()> Detail) {
    llvm::timeTraceProfilerBegin(Name, Detail);
  }

  /// Manually end the last time section.
  void endSection() { llvm::timeTraceProfilerEnd(); }
};

/// Time spend in the current scope, assigned to the function name.
#define TIMESCOPE() llvm::TimeTraceScope TimeScope(__PRETTY_FUNCTION__)

// The macros below that take an IDENT build the detail string lazily: parsing
// the source location and concatenating strings costs several allocations,
// which is significant for short runtime entry points (e.g., a 'target data'
// region with only 'use_device_addr'). llvm::TimeTraceScope invokes the detail
// callback only when the time-trace profiler is enabled.

/// Time spend in the current scope, assigned to the function name and source
/// info.
#define TIMESCOPE_WITH_IDENT(IDENT)                                            \
  llvm::TimeTraceScope TimeScope(__FUNCTION__, [&]() -> std::string {          \
    return SourceInfo(IDENT).getProfileLocation();                             \
  })

/// Time spend in the current scope, assigned to the given name and source
/// info.
#define TIMESCOPE_WITH_NAME_AND_IDENT(NAME, IDENT)                             \
  llvm::TimeTraceScope TimeScope(NAME, [&]() -> std::string {                  \
    return SourceInfo(IDENT).getProfileLocation();                             \
  })

/// Time spend in the current scope, assigned to the function name and source
/// info and RegionTypeMsg.
#define TIMESCOPE_WITH_RTM_AND_IDENT(RegionTypeMsg, IDENT)                     \
  llvm::TimeTraceScope TimeScope(__FUNCTION__, [&]() -> std::string {          \
    return std::string(SourceInfo(IDENT).getProfileLocation()) +               \
           (RegionTypeMsg);                                                    \
  })

/// Time spend in the current scope, assigned to the regionType
/// with details from runtime
#define TIMESCOPE_WITH_DETAILS_AND_IDENT(RegionTypeMsg, Details, IDENT)        \
  llvm::TimeTraceScope TimeScope(RegionTypeMsg, [&]() -> std::string {         \
    return std::string(SourceInfo(IDENT).getProfileLocation()) + (Details);    \
  })

/// Time spend in the current scope, assigned to the function name and source
/// with details
#define TIMESCOPE_WITH_DETAILS(Details)                                        \
  llvm::TimeTraceScope TimeScope(__FUNCTION__, Details)

#endif // OMPTARGET_SHARED_PROFILE_H

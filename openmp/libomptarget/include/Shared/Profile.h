//===-- Shared/Profile.h - Target independent OpenMP target RTL -*- C++ -*-===//
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

#include "llvm/Support/TimeProfiler.h"

/// Time spend in the current scope, assigned to the function name.
#define TIMESCOPE() llvm::TimeTraceScope TimeScope(__FUNCTION__)

/// Time spend in the current scope, assigned to the function name and source
/// info.
#define TIMESCOPE_WITH_IDENT(IDENT)                                            \
  SourceInfo SI(IDENT);                                                        \
  llvm::TimeTraceScope TimeScope(__FUNCTION__, SI.getProfileLocation())

/// Time spend in the current scope, assigned to the given name and source
/// info.
#define TIMESCOPE_WITH_NAME_AND_IDENT(NAME, IDENT)                             \
  SourceInfo SI(IDENT);                                                        \
  llvm::TimeTraceScope TimeScope(NAME, SI.getProfileLocation())

/// Time spend in the current scope, assigned to the function name and source
/// info and RegionTypeMsg.
#define TIMESCOPE_WITH_RTM_AND_IDENT(RegionTypeMsg, IDENT)                     \
  SourceInfo SI(IDENT);                                                        \
  std::string ProfileLocation = SI.getProfileLocation();                       \
  std::string RTM = RegionTypeMsg;                                             \
  llvm::TimeTraceScope TimeScope(__FUNCTION__, ProfileLocation + RTM)

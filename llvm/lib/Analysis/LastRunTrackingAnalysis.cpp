//===- LastRunTrackingAnalysis.cpp - Avoid running redundant pass -*- C++ -*-=//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This is an analysis pass to track a set of passes that have been run, so that
// we can avoid running a pass again if there is no change since the last run of
// the pass.
//
//===----------------------------------------------------------------------===//

#include "llvm/Analysis/LastRunTrackingAnalysis.h"
#include "llvm/ADT/Statistic.h"
#include "llvm/Support/CommandLine.h"

using namespace llvm;

#define DEBUG_TYPE "last-run-tracking"
STATISTIC(NumSkippedPasses, "Number of skipped passes");
STATISTIC(NumLRTQueries, "Number of LastRunTracking queries");

static cl::opt<bool>
    DisableLastRunTracking("disable-last-run-tracking", cl::Hidden,
                           cl::desc("Disable last run tracking"),
                           cl::init(false));

bool LastRunTrackingInfo::shouldSkipImpl(PassID ID, OptionPtr Ptr) const {
  if (DisableLastRunTracking)
    return false;
  ++NumLRTQueries;
  auto Iter = TrackedPasses.find(ID);
  if (Iter == TrackedPasses.end())
    return false;
  if (!Iter->second || Iter->second(Ptr)) {
    ++NumSkippedPasses;
    return true;
  }
  return false;
}

void LastRunTrackingInfo::updateImpl(PassID ID, bool Changed,
                                     CompatibilityCheckFn CheckFn) {
  if (Changed)
    TrackedPasses.clear();
  TrackedPasses[ID] = std::move(CheckFn);
}

AnalysisKey LastRunTrackingAnalysis::Key;

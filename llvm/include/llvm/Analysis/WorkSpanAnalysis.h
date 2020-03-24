//===- WorkSpanAnalysis.h - Analysis to estimate work and span --*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements an analysis pass to estimate the work and span of the
// program.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_ANALYSIS_WORKSPANANALYSIS_H_
#define LLVM_ANALYSIS_WORKSPANANALYSIS_H_

#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/Analysis/CodeMetrics.h"

// TODO: Build a CGSCC pass based on these analyses to efficiently estimate the
// work and span of all the functions in a module.

// TODO: Use BlockFrequencyInfo to improve how this analysis evaluates code with
// control flow.  Specifically, the analysis should weight the work and span of
// a block based on the probabilities of its incoming edges, with special care
// given to detach, reattach, and continue edges.

// TODO: Connect these analyses with a scalability profiler to implement PGO for
// Tapir.

namespace llvm {
class Loop;
class LoopInfo;
class ScalarEvolution;
class TargetLibraryInfo;
class TargetTransformInfo;

struct WSCost {
  int64_t Work = 0;
  int64_t Span = 0;

  bool UnknownCost = false;

  CodeMetrics Metrics;
};

// Get a constant trip count for the given loop.
unsigned getConstTripCount(const Loop *L, ScalarEvolution &SE);

void estimateLoopCost(WSCost &LoopCost, const Loop *L, LoopInfo *LI,
                      ScalarEvolution *SE, const TargetTransformInfo &TTI,
                      TargetLibraryInfo *TLI,
                      const SmallPtrSetImpl<const Value *> &EphValues);
}

#endif // LLVM_ANALYSIS_WORKSPANANALYSIS_H_

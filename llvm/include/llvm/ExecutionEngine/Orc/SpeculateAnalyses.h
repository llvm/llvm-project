//===-- SpeculateAnalyses.h  --*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// \file
/// Contains the Analyses and Result Interpretation to select likely functions
/// to Speculatively compile before they are called. [Experimentation]
//===----------------------------------------------------------------------===//

#ifndef LLVM_EXECUTIONENGINE_ORC_SPECULATEANALYSES_H
#define LLVM_EXECUTIONENGINE_ORC_SPECULATEANALYSES_H

#include "llvm/ExecutionEngine/Orc/Core.h"
#include "llvm/ExecutionEngine/Orc/Speculation.h"

#include <vector>

namespace llvm {

namespace orc {

// Direct calls in high frequency basic blocks are extracted.
class BlockFreqQuery {
private:
  void findCalles(const BasicBlock *, DenseSet<StringRef> &);
  size_t numBBToGet(size_t);

public:
  using ResultTy = Optional<DenseMap<StringRef, DenseSet<StringRef>>>;

  // Find likely next executables based on IR Block Frequency
  ResultTy operator()(Function &F, FunctionAnalysisManager &FAM);
};

// Walk the CFG by exploting BranchProbabilityInfo
class CFGWalkQuery {
public:
  using ResultTy = Optional<DenseMap<StringRef, DenseSet<StringRef>>>;
  ResultTy operator()(Function &F, FunctionAnalysisManager &FAM);
};

} // namespace orc
} // namespace llvm

#endif // LLVM_EXECUTIONENGINE_ORC_SPECULATEANALYSES_H

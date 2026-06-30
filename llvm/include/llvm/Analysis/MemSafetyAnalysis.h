//===- MemSafetyAnalysis.h - Memory access safety for a loop ----*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Originally developed by Advanced Micro Devices, Inc. (2015).
//
// MemSafetyAnalysis answers one question about a loop's memory accesses:
//
//   "Is this memory access guaranteed to execute on every iteration of the
//    loop?" -- isGuaranteedMemoryAccess(const SCEV *)
//
//   Used to legalise rewriting a masked store as load-blend-store: if the
//   same SCEV-keyed address is touched on every iteration anyway, the
//   sequence does not introduce any new fault.
//
// The analysis is built once per loop and consulted from the loop vectoriser
// when -enable-masked-memory-optimization is in effect.
//
// Phase-2 (stack-alloca padding) is layered on top of this analysis in a
// follow-up patch; the ctor here keeps the TLI/TTI parameters that Phase-2
// will use, so the analysis-construction call site does not have to change.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_ANALYSIS_MEMSAFETYANALYSIS_H
#define LLVM_ANALYSIS_MEMSAFETYANALYSIS_H

#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/IR/Value.h"
#include <map>

namespace llvm {

class BasicBlock;
class DominatorTree;
class Loop;
class LoopInfo;
class ScalarEvolution;
class SCEV;
class TargetLibraryInfo;
class TargetTransformInfo;

/// Classification of a per-block memory access. \c LocalSafe means the access
/// has been observed but not yet proven to execute on every iteration of the
/// enclosing loop; \c Safe means the access is proven to execute on every
/// iteration; \c Unsafe is reserved for future use.
enum class MemProperty { Unsafe, Safe, LocalSafe };

/// Per-block summary of memory accesses, indexed by the SCEV of the pointer.
class BlockMemInfo {
public:
  BasicBlock *BB;
  bool GuaranteedToExecute;
  std::map<const SCEV *, MemProperty> BlockMemAccess;

  BlockMemInfo(BasicBlock *BB, bool GuaranteedToExecute)
      : BB(BB), GuaranteedToExecute(GuaranteedToExecute) {}

  void addMemoryAccess(const SCEV *S, MemProperty AccessTy);
  bool isBlockGuaranteedToExecute() const { return GuaranteedToExecute; }
  void copyBlockMemInfo(BlockMemInfo *BMI);
};

/// MemSafetyAnalysis -- per-loop analysis that builds a SCEV-keyed map of
/// memory accesses guaranteed to execute on every iteration.
///
/// A memory access is guaranteed to execute when either:
///   * it lives in a block that dominates the latch (top-level block), or
///   * the same SCEV-keyed address is touched on every path of the
///     subtree rooted at the current block (propagated bottom-up).
///
/// See `examples` block in MemSafetyAnalysis.cpp for canonical cases.
class MemSafetyAnalysis {
public:
  MemSafetyAnalysis(Loop *L, LoopInfo *LI, ScalarEvolution *SE,
                    DominatorTree *DT, const TargetLibraryInfo *TLI,
                    const TargetTransformInfo *TTI);
  ~MemSafetyAnalysis();

  MemSafetyAnalysis(const MemSafetyAnalysis &) = delete;
  MemSafetyAnalysis &operator=(const MemSafetyAnalysis &) = delete;

  /// True if the analysis successfully ran on the loop. Clients must check
  /// this before consulting the other queries.
  bool isLegalAnalysis() const { return IsAnalysisValid; }

  /// True if the access at SCEV \p S is guaranteed to execute on every
  /// iteration of the loop body.
  bool isGuaranteedMemoryAccess(const SCEV *S) const;

  /// Debug: dump the analysis state to dbgs().
  void printAnalysis() const;

private:
  Loop *L;
  LoopInfo *LI;
  ScalarEvolution *SE;
  DominatorTree *DT;
  const TargetLibraryInfo *TLI;
  const TargetTransformInfo *TTI;
  bool IsAnalysisValid = false;

  std::map<BasicBlock *, BlockMemInfo *> BlockMemAccessMap;
  std::map<const SCEV *, MemProperty> SafeMemoryAccesses;

  BlockMemInfo *getBlockMemInfo(BasicBlock *BB);
  bool blockGuaranteedToExecute(BasicBlock *BB);
  void addGuaranteedMemoryAccess(const SCEV *S);
  bool isLegalLoopStructure();
  bool processBlock(BasicBlock *BB);
  bool analyzeMemoryAccessesInLoop();
  void clearLocalMemory();
};

} // namespace llvm

#endif // LLVM_ANALYSIS_MEMSAFETYANALYSIS_H

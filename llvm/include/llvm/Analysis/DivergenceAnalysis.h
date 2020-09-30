//===- llvm/Analysis/DivergenceAnalysis.h - Divergence Analysis -*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// \file
// The divergence analysis determines which instructions and branches are
// divergent given a set of divergent source instructions.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_ANALYSIS_DIVERGENCE_ANALYSIS_H
#define LLVM_ANALYSIS_DIVERGENCE_ANALYSIS_H

#include "llvm/ADT/DenseSet.h"
#include "llvm/Analysis/SyncDependenceAnalysis.h"
#include "llvm/IR/Function.h"
#include "llvm/Pass.h"
#include <vector>

namespace llvm {
class Module;
class Value;
class Instruction;
class Loop;
class raw_ostream;
class TargetTransformInfo;

/// \brief Generic divergence analysis for reducible CFGs.
///
/// This analysis propagates divergence in a data-parallel context from sources
/// of divergence to all users. It requires reducible CFGs. All assignments
/// should be in SSA form.
class DivergenceAnalysis {
public:
  /// \brief This instance will analyze the whole function \p F or the loop \p
  /// RegionLoop.
  ///
  /// \param RegionLoop if non-null the analysis is restricted to \p RegionLoop.
  /// Otherwise the whole function is analyzed.
  /// \param IsLCSSAForm whether the analysis may assume that the IR in the
  /// region in in LCSSA form.
  DivergenceAnalysis(const Function &F, const Loop *RegionLoop,
                     const DominatorTree &DT, const LoopInfo &LI,
                     SyncDependenceAnalysis &SDA, bool IsLCSSAForm);

  /// \brief The loop that defines the analyzed region (if any).
  const Loop *getRegionLoop() const { return RegionLoop; }
  const Function &getFunction() const { return F; }

  /// \brief Whether \p BB is part of the region.
  bool inRegion(const BasicBlock &BB) const;
  /// \brief Whether \p I is part of the region.
  bool inRegion(const Instruction &I) const;

  /// \brief Mark \p UniVal as a value that is always uniform.
  void addUniformOverride(const Value &UniVal);

  /// \brief Mark \p DivVal as a value that is always divergent. Will not do so
  /// if `isAlwaysUniform(DivVal)`.
  /// \returns Whether the tracked divergence state of \p DivVal changed.
  bool markDivergent(const Value &DivVal);

  /// \brief Propagate divergence to all instructions in the region.
  /// Divergence is seeded by calls to \p markDivergent.
  void compute();

  /// \brief Whether any value was marked or analyzed to be divergent.
  bool hasDetectedDivergence() const { return !DivergentValues.empty(); }

  /// \brief Whether \p Val will always return a uniform value regardless of its
  /// operands
  bool isAlwaysUniform(const Value &Val) const;

  /// \brief Whether \p Val is divergent at its definition.
  bool isDivergent(const Value &Val) const;

  /// \brief Whether \p U is divergent. Uses of a uniform value can be
  /// divergent.
  bool isDivergentUse(const Use &U) const;

  void print(raw_ostream &OS, const Module *) const;

private:
  /// \brief Mark \p Term as divergent and push all Instructions that become
  /// divergent as a result on the worklist.
  void analyzeControlDivergence(const Instruction &Term);
  /// \brief Mark all phi nodes in \p JoinBlock as divergent and push them on
  /// the worklist.
  void taintAndPushPhiNodes(const BasicBlock &JoinBlock);

  /// \brief Identify all Instructions that become divergent because \p DivExit
  /// is a divergent loop exit of \p DivLoop. Mark those instructions as
  /// divergent and push them on the worklist.
  void propagateLoopExitDivergence(const BasicBlock &DivExit,
                                   const Loop &DivLoop);

  /// \brief Internal implementation function for propagateLoopExitDivergence.
  void analyzeLoopExitDivergence(const BasicBlock &DivExit,
                                 const Loop &OuterDivLoop);

  /// \brief Mark all instruction as divergent that use a value defined in \p
  /// OuterDivLoop. Push their users on the worklist.
  void analyzeTemporalDivergence(const Instruction &I,
                                 const Loop &OuterDivLoop);

  /// \brief Push all users of \p Val (in the region) to the worklist.
  void pushUsers(const Value &I);

  /// \brief Whether \p Val is divergent when read in \p ObservingBlock.
  bool isTemporalDivergent(const BasicBlock &ObservingBlock,
                           const Value &Val) const;

  /// \brief Whether \p Block is join divergent
  ///
  /// (see markBlockJoinDivergent).
  bool isJoinDivergent(const BasicBlock &Block) const {
    return DivergentJoinBlocks.find(&Block) != DivergentJoinBlocks.end();
  }

private:
  const Function &F;
  // If regionLoop != nullptr, analysis is only performed within \p RegionLoop.
  // Otherwise, analyze the whole function
  const Loop *RegionLoop;

  const DominatorTree &DT;
  const LoopInfo &LI;

  // Recognized divergent loops
  DenseSet<const Loop *> DivergentLoops;

  // The SDA links divergent branches to divergent control-flow joins.
  SyncDependenceAnalysis &SDA;

  // Use simplified code path for LCSSA form.
  bool IsLCSSAForm;

  // Set of known-uniform values.
  DenseSet<const Value *> UniformOverrides;

  // Blocks with joining divergent control from different predecessors.
  DenseSet<const BasicBlock *> DivergentJoinBlocks; // FIXME Deprecated

  // Detected/marked divergent values.
  DenseSet<const Value *> DivergentValues;

  // Internal worklist for divergence propagation.
  std::vector<const Instruction *> Worklist;
};

/// \brief Divergence analysis frontend for GPU kernels.
class GPUDivergenceAnalysis {
  SyncDependenceAnalysis SDA;
  DivergenceAnalysis DA;

public:
  /// Runs the divergence analysis on @F, a GPU kernel
  GPUDivergenceAnalysis(Function &F, const DominatorTree &DT,
                        const PostDominatorTree &PDT, const LoopInfo &LI,
                        const TargetTransformInfo &TTI);

  /// Whether any divergence was detected.
  bool hasDivergence() const { return DA.hasDetectedDivergence(); }

  /// The GPU kernel this analysis result is for
  const Function &getFunction() const { return DA.getFunction(); }

  /// Whether \p V is divergent at its definition.
  bool isDivergent(const Value &V) const;

  /// Whether \p U is divergent. Uses of a uniform value can be divergent.
  bool isDivergentUse(const Use &U) const;

  /// Whether \p V is uniform/non-divergent.
  bool isUniform(const Value &V) const { return !isDivergent(V); }

  /// Whether \p U is uniform/non-divergent. Uses of a uniform value can be
  /// divergent.
  bool isUniformUse(const Use &U) const { return !isDivergentUse(U); }

  /// Print all divergent values in the kernel.
  void print(raw_ostream &OS, const Module *) const;
};

} // namespace llvm

#endif // LLVM_ANALYSIS_DIVERGENCE_ANALYSIS_H

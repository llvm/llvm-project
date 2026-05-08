//===- BranchProbabilityInfo.h - Branch Probability Analysis ----*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This pass is used to evaluate branch probabilties.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_ANALYSIS_BRANCHPROBABILITYINFO_H
#define LLVM_ANALYSIS_BRANCHPROBABILITYINFO_H

#include "llvm/IR/BasicBlock.h"
#include "llvm/IR/CFG.h"
#include "llvm/IR/PassManager.h"
#include "llvm/Pass.h"
#include "llvm/Support/BranchProbability.h"
#include "llvm/Support/Compiler.h"
#include <cassert>
#include <cstdint>
#include <memory>
#include <utility>

namespace llvm {

class Function;
class Loop;
class LoopInfo;
class raw_ostream;
class DominatorTree;
class PostDominatorTree;
class TargetLibraryInfo;
class Value;

/// Analysis providing branch probability information.
///
/// This is a function analysis which provides information on the relative
/// probabilities of each "edge" in the function's CFG where such an edge is
/// defined by a pair (PredBlock and an index in the successors). The
/// probability of an edge from one block is always relative to the
/// probabilities of other edges from the block. The probabilites of all edges
/// from a block sum to exactly one (100%).
/// We use a pair (PredBlock and an index in the successors) to uniquely
/// identify an edge, since we can have multiple edges from Src to Dst.
/// As an example, we can have a switch which jumps to Dst with value 0 and
/// value 10.
///
/// Process of computing branch probabilities can be logically viewed as three
/// step process:
///
///   First, if there is a profile information associated with the branch then
/// it is trivially translated to branch probabilities. There is one exception
/// from this rule though. Probabilities for edges leading to "unreachable"
/// blocks (blocks with the estimated weight not greater than
/// UNREACHABLE_WEIGHT) are evaluated according to static estimation and
/// override profile information. If no branch probabilities were calculated
/// on this step then take the next one.
///
///   Second, estimate absolute execution weights for each block based on
/// statically known information. Roots of such information are "cold",
/// "unreachable", "noreturn" and "unwind" blocks. Those blocks get their
/// weights set to BlockExecWeight::COLD, BlockExecWeight::UNREACHABLE,
/// BlockExecWeight::NORETURN and BlockExecWeight::UNWIND respectively. Then the
/// weights are propagated to the other blocks up the domination line. In
/// addition, if all successors have estimated weights set then maximum of these
/// weights assigned to the block itself (while this is not ideal heuristic in
/// theory it's simple and works reasonably well in most cases) and the process
/// repeats. Once the process of weights propagation converges branch
/// probabilities are set for all such branches that have at least one successor
/// with the weight set. Default execution weight (BlockExecWeight::DEFAULT) is
/// used for any successors which doesn't have its weight set. For loop back
/// branches we use their weights scaled by loop trip count equal to
/// 'LBH_TAKEN_WEIGHT/LBH_NOTTAKEN_WEIGHT'.
///
/// Here is a simple example demonstrating how the described algorithm works.
///
///          BB1
///         /   \
///        v     v
///      BB2     BB3
///     /   \
///    v     v
///  ColdBB  UnreachBB
///
/// Initially, ColdBB is associated with COLD_WEIGHT and UnreachBB with
/// UNREACHABLE_WEIGHT. COLD_WEIGHT is set to BB2 as maximum between its
/// successors. BB1 and BB3 has no explicit estimated weights and assumed to
/// have DEFAULT_WEIGHT. Based on assigned weights branches will have the
/// following probabilities:
/// P(BB1->BB2) = COLD_WEIGHT/(COLD_WEIGHT + DEFAULT_WEIGHT) =
///   0xffff / (0xffff + 0xfffff) = 0.0588(5.9%)
/// P(BB1->BB3) = DEFAULT_WEIGHT_WEIGHT/(COLD_WEIGHT + DEFAULT_WEIGHT) =
///          0xfffff / (0xffff + 0xfffff) = 0.941(94.1%)
/// P(BB2->ColdBB) = COLD_WEIGHT/(COLD_WEIGHT + UNREACHABLE_WEIGHT) = 1(100%)
/// P(BB2->UnreachBB) =
///   UNREACHABLE_WEIGHT/(COLD_WEIGHT+UNREACHABLE_WEIGHT) = 0(0%)
///
/// If no branch probabilities were calculated on this step then take the next
/// one.
///
///   Third, apply different kinds of local heuristics for each individual
/// branch until first match. For example probability of a pointer to be null is
/// estimated as PH_TAKEN_WEIGHT/(PH_TAKEN_WEIGHT + PH_NONTAKEN_WEIGHT). If
/// no local heuristic has been matched then branch is left with no explicit
/// probability set and assumed to have default probability.
class BranchProbabilityInfo {
public:
  BranchProbabilityInfo() = default;

  BranchProbabilityInfo(const Function &F, const LoopInfo &LI,
                        const TargetLibraryInfo *TLI = nullptr,
                        DominatorTree *DT = nullptr,
                        PostDominatorTree *PDT = nullptr) {
    calculate(F, LI, TLI, DT, PDT);
  }

  LLVM_ABI bool invalidate(Function &, const PreservedAnalyses &PA,
                           FunctionAnalysisManager::Invalidator &);

  LLVM_ABI void print(raw_ostream &OS) const;

  /// Get an edge's probability, relative to other out-edges of the Src.
  ///
  /// This routine provides access to the fractional probability between zero
  /// (0%) and one (100%) of this edge executing, relative to other edges
  /// leaving the 'Src' block. The returned probability is never zero, and can
  /// only be one if the source block has only one successor.
  LLVM_ABI BranchProbability
  getEdgeProbability(const BasicBlock *Src, unsigned IndexInSuccessors) const;

  /// Get the probability of going from Src to Dst.
  ///
  /// It returns the sum of all probabilities for edges from Src to Dst.
  LLVM_ABI BranchProbability getEdgeProbability(const BasicBlock *Src,
                                                const BasicBlock *Dst) const;

  /// Test if an edge is hot relative to other out-edges of the Src.
  ///
  /// Check whether this edge out of the source block is 'hot'. We define hot
  /// as having a relative probability > 80%.
  LLVM_ABI bool isEdgeHot(const BasicBlock *Src, const BasicBlock *Dst) const;

  /// Print an edge's probability.
  ///
  /// Retrieves an edge's probability similarly to \see getEdgeProbability, but
  /// then prints that probability to the provided stream. That stream is then
  /// returned.
  LLVM_ABI raw_ostream &printEdgeProbability(raw_ostream &OS,
                                             const BasicBlock *Src,
                                             const BasicBlock *Dst) const;

  /// Set the raw probabilities for all edges from the given block.
  ///
  /// This allows a pass to explicitly set edge probabilities for a block. It
  /// can be used when updating the CFG to update the branch probability
  /// information.
  LLVM_ABI void
  setEdgeProbability(const BasicBlock *Src,
                     const SmallVectorImpl<BranchProbability> &Probs);

  /// Copy outgoing edge probabilities from \p Src to \p Dst.
  ///
  /// This allows to keep probabilities unset for the destination if they were
  /// unset for source.
  LLVM_ABI void copyEdgeProbabilities(BasicBlock *Src, BasicBlock *Dst);

  /// Swap outgoing edges probabilities for \p Src with branch terminator
  LLVM_ABI void swapSuccEdgesProbabilities(const BasicBlock *Src);

  static BranchProbability getBranchProbStackProtector(bool IsLikely) {
    static const BranchProbability LikelyProb((1u << 20) - 1, 1u << 20);
    return IsLikely ? LikelyProb : LikelyProb.getCompl();
  }

  LLVM_ABI void calculate(const Function &F, const LoopInfo &LI,
                          const TargetLibraryInfo *TLI, DominatorTree *DT,
                          PostDominatorTree *PDT);

  /// Forget analysis results for the given basic block.
  LLVM_ABI void eraseBlock(const BasicBlock *BB);

private:
  MutableArrayRef<BranchProbability> allocEdges(const BasicBlock *BB);
  ArrayRef<BranchProbability> getEdges(const BasicBlock *BB) const;

  // Storage for branch probabilities.
  SmallVector<BranchProbability> Probs;
  // Map from block number to first edge.
  SmallVector<unsigned> EdgeStarts;

  /// Track the last function we run over for printing.
  const Function *LastF = nullptr;
  unsigned BlockNumberEpoch;
};

/// Analysis pass which computes \c BranchProbabilityInfo.
class BranchProbabilityAnalysis
    : public AnalysisInfoMixin<BranchProbabilityAnalysis> {
  friend AnalysisInfoMixin<BranchProbabilityAnalysis>;

  LLVM_ABI static AnalysisKey Key;

public:
  /// Provide the result type for this analysis pass.
  using Result = BranchProbabilityInfo;

  /// Run the analysis pass over a function and produce BPI.
  LLVM_ABI BranchProbabilityInfo run(Function &F, FunctionAnalysisManager &AM);
};

/// Printer pass for the \c BranchProbabilityAnalysis results.
class BranchProbabilityPrinterPass
    : public PassInfoMixin<BranchProbabilityPrinterPass> {
  raw_ostream &OS;

public:
  explicit BranchProbabilityPrinterPass(raw_ostream &OS) : OS(OS) {}

  LLVM_ABI PreservedAnalyses run(Function &F, FunctionAnalysisManager &AM);

  static bool isRequired() { return true; }
};

/// Legacy analysis pass which computes \c BranchProbabilityInfo.
class LLVM_ABI BranchProbabilityInfoWrapperPass : public FunctionPass {
  BranchProbabilityInfo BPI;

public:
  static char ID;

  BranchProbabilityInfoWrapperPass();

  BranchProbabilityInfo &getBPI() { return BPI; }
  const BranchProbabilityInfo &getBPI() const { return BPI; }

  void getAnalysisUsage(AnalysisUsage &AU) const override;
  bool runOnFunction(Function &F) override;
  void print(raw_ostream &OS, const Module *M = nullptr) const override;
};

} // end namespace llvm

#endif // LLVM_ANALYSIS_BRANCHPROBABILITYINFO_H

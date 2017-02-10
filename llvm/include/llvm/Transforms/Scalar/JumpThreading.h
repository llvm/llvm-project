//===- JumpThreading.h - thread control through conditional BBs -*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
/// \file
/// See the comments on JumpThreadingPass.
///
//===----------------------------------------------------------------------===//

#ifndef LLVM_TRANSFORMS_SCALAR_JUMPTHREADING_H
#define LLVM_TRANSFORMS_SCALAR_JUMPTHREADING_H

#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SmallSet.h"
#include "llvm/Analysis/BlockFrequencyInfo.h"
#include "llvm/Analysis/BlockFrequencyInfoImpl.h"
#include "llvm/Analysis/BranchProbabilityInfo.h"
#include "llvm/Analysis/LazyValueInfo.h"
#include "llvm/Analysis/TargetLibraryInfo.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/IntrinsicInst.h"
#include "llvm/IR/ValueHandle.h"

namespace llvm {

/// A private "module" namespace for types and utilities used by
/// JumpThreading.
/// These are implementation details and should not be used by clients.
namespace jumpthreading {
// These are at global scope so static functions can use them too.
typedef SmallVectorImpl<std::pair<Constant *, BasicBlock *>> PredValueInfo;
typedef SmallVector<std::pair<Constant *, BasicBlock *>, 8> PredValueInfoTy;

// This is used to keep track of what kind of constant we're currently hoping
// to find.
enum ConstantPreference { WantInteger, WantBlockAddress };
}

/// This pass performs 'jump threading', which looks at blocks that have
/// multiple predecessors and multiple successors.  If one or more of the
/// predecessors of the block can be proven to always jump to one of the
/// successors, we forward the edge from the predecessor to the successor by
/// duplicating the contents of this block.
///
/// An example of when this can occur is code like this:
///
///   if () { ...
///     X = 4;
///   }
///   if (X < 3) {
///
/// In this case, the unconditional branch at the end of the first if can be
/// revectored to the false side of the second if.
///
class JumpThreadingPass : public PassInfoMixin<JumpThreadingPass> {
  TargetLibraryInfo *TLI;
  LazyValueInfo *LVI;
  std::unique_ptr<BlockFrequencyInfo> BFI;
  std::unique_ptr<BranchProbabilityInfo> BPI;
  bool HasProfileData = false;
  bool HasGuards = false;
#ifdef NDEBUG
  SmallPtrSet<const BasicBlock *, 16> LoopHeaders;
#else
  SmallSet<AssertingVH<const BasicBlock>, 16> LoopHeaders;
#endif
  DenseSet<std::pair<Value *, BasicBlock *>> RecursionSet;

  unsigned BBDupThreshold;

  // RAII helper for updating the recursion stack.
  struct RecursionSetRemover {
    DenseSet<std::pair<Value *, BasicBlock *>> &TheSet;
    std::pair<Value *, BasicBlock *> ThePair;

    RecursionSetRemover(DenseSet<std::pair<Value *, BasicBlock *>> &S,
                        std::pair<Value *, BasicBlock *> P)
        : TheSet(S), ThePair(P) {}

    ~RecursionSetRemover() { TheSet.erase(ThePair); }
  };

public:
  JumpThreadingPass(int T = -1);

  // Glue for old PM.
  bool runImpl(Function &F, TargetLibraryInfo *TLI_, LazyValueInfo *LVI_,
               bool HasProfileData_, std::unique_ptr<BlockFrequencyInfo> BFI_,
               std::unique_ptr<BranchProbabilityInfo> BPI_);

  PreservedAnalyses run(Function &F, FunctionAnalysisManager &AM);

  void releaseMemory() {
    BFI.reset();
    BPI.reset();
  }

  void FindLoopHeaders(Function &F);
  bool ProcessBlock(BasicBlock *BB);
  bool ThreadEdge(BasicBlock *BB, const SmallVectorImpl<BasicBlock *> &PredBBs,
                  BasicBlock *SuccBB);
  bool DuplicateCondBranchOnPHIIntoPred(
      BasicBlock *BB, const SmallVectorImpl<BasicBlock *> &PredBBs);

  bool
  ComputeValueKnownInPredecessors(Value *V, BasicBlock *BB,
                                  jumpthreading::PredValueInfo &Result,
                                  jumpthreading::ConstantPreference Preference,
                                  Instruction *CxtI = nullptr);
  bool ProcessThreadableEdges(Value *Cond, BasicBlock *BB,
                              jumpthreading::ConstantPreference Preference,
                              Instruction *CxtI = nullptr);

  bool ProcessBranchOnPHI(PHINode *PN);
  bool ProcessBranchOnXOR(BinaryOperator *BO);
  bool ProcessImpliedCondition(BasicBlock *BB);

  bool SimplifyPartiallyRedundantLoad(LoadInst *LI);
  bool TryToUnfoldSelect(CmpInst *CondCmp, BasicBlock *BB);
  bool TryToUnfoldSelectInCurrBB(BasicBlock *BB);

  bool ProcessGuards(BasicBlock *BB);
  bool ThreadGuard(BasicBlock *BB, IntrinsicInst *Guard, BranchInst *BI);

private:
  BasicBlock *SplitBlockPreds(BasicBlock *BB, ArrayRef<BasicBlock *> Preds,
                              const char *Suffix);
  void UpdateBlockFreqAndEdgeWeight(BasicBlock *PredBB, BasicBlock *BB,
                                    BasicBlock *NewBB, BasicBlock *SuccBB);
  /// Check if the block has profile metadata for its outgoing edges.
  bool doesBlockHaveProfileData(BasicBlock *BB);
};

} // end namespace llvm

#endif

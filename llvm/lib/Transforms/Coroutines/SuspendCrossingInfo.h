//===- SuspendCrossingInfo.cpp - Utility for suspend crossing values ------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// The SuspendCrossingInfo maintains data that allows to answer a question
// whether given two BasicBlocks A and B there is a path from A to B that
// passes through a suspend point. Note, SuspendCrossingInfo is invalidated
// by changes to the CFG including adding/removing BBs due to its use of BB
// ptrs in the BlockToIndexMapping.
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIB_TRANSFORMS_COROUTINES_SUSPENDCROSSINGINFO_H
#define LLVM_LIB_TRANSFORMS_COROUTINES_SUSPENDCROSSINGINFO_H

#include "CoroInstr.h"
#include "llvm/ADT/BitVector.h"
#include "llvm/ADT/PostOrderIterator.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/IR/BasicBlock.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/Instruction.h"

namespace llvm {

// Provides two way mapping between the blocks and numbers.
class BlockToIndexMapping {
  SmallVector<BasicBlock *, 32> V;

public:
  size_t size() const { return V.size(); }

  BlockToIndexMapping(Function &F) {
    for (BasicBlock &BB : F)
      V.push_back(&BB);
    llvm::sort(V);
  }

  size_t blockToIndex(BasicBlock const *BB) const {
    auto *I = llvm::lower_bound(V, BB);
    assert(I != V.end() && *I == BB && "BasicBlockNumberng: Unknown block");
    return I - V.begin();
  }

  BasicBlock *indexToBlock(unsigned Index) const { return V[Index]; }
};

// The SuspendCrossingInfo maintains data that allows to answer a question
// whether given two BasicBlocks A and B there is a path from A to B that
// passes through a suspend point.
//
// For every basic block 'i' it maintains a BlockData that consists of:
//   Consumes:  a bit vector which contains a set of indices of blocks that can
//              reach block 'i'. A block can trivially reach itself.
//   Kills: a bit vector which contains a set of indices of blocks that can
//          reach block 'i' but there is a path crossing a suspend point
//          not repeating 'i' (path to 'i' without cycles containing 'i').
//   Suspend: a boolean indicating whether block 'i' contains a suspend point.
//   End: a boolean indicating whether block 'i' contains a coro.end intrinsic.
//   KillLoop: There is a path from 'i' to 'i' not otherwise repeating 'i' that
//             crosses a suspend point.
//
class SuspendCrossingInfo {
  BlockToIndexMapping Mapping;

  struct BlockData {
    BitVector Consumes;
    BitVector Kills;
    bool Suspend = false;
    bool End = false;
    bool KillLoop = false;
    bool Changed = false;
  };
  SmallVector<BlockData, 32> Block;

  iterator_range<pred_iterator> predecessors(BlockData const &BD) const {
    BasicBlock *BB = Mapping.indexToBlock(&BD - &Block[0]);
    return llvm::predecessors(BB);
  }

  BlockData &getBlockData(BasicBlock *BB) {
    return Block[Mapping.blockToIndex(BB)];
  }

  /// Compute the BlockData for the current function in one iteration.
  /// Initialize - Whether this is the first iteration, we can optimize
  /// the initial case a little bit by manual loop switch.
  /// Returns whether the BlockData changes in this iteration.
  template <bool Initialize = false>
  bool computeBlockData(const ReversePostOrderTraversal<Function *> &RPOT);

public:
#if !defined(NDEBUG) || defined(LLVM_ENABLE_DUMP)
  // Print order is in RPO
  void dump() const;
  void dump(StringRef Label, BitVector const &BV,
            const ReversePostOrderTraversal<Function *> &RPOT) const;
#endif

  SuspendCrossingInfo(Function &F,
                      const SmallVectorImpl<AnyCoroSuspendInst *> &CoroSuspends,
                      const SmallVectorImpl<AnyCoroEndInst *> &CoroEnds);

  /// Returns true if there is a path from \p From to \p To crossing a suspend
  /// point without crossing \p From a 2nd time.
  bool hasPathCrossingSuspendPoint(BasicBlock *From, BasicBlock *To) const;

  /// Returns true if there is a path from \p From to \p To crossing a suspend
  /// point without crossing \p From a 2nd time. If \p From is the same as \p To
  /// this will also check if there is a looping path crossing a suspend point.
  bool hasPathOrLoopCrossingSuspendPoint(BasicBlock *From,
                                         BasicBlock *To) const;

  bool isDefinitionAcrossSuspend(BasicBlock *DefBB, User *U) const {
    auto *I = cast<Instruction>(U);

    // We rewrote PHINodes, so that only the ones with exactly one incoming
    // value need to be analyzed.
    if (auto *PN = dyn_cast<PHINode>(I))
      if (PN->getNumIncomingValues() > 1)
        return false;

    BasicBlock *UseBB = I->getParent();

    // As a special case, treat uses by an llvm.coro.suspend.retcon or an
    // llvm.coro.suspend.async as if they were uses in the suspend's single
    // predecessor: the uses conceptually occur before the suspend.
    if (isa<CoroSuspendRetconInst>(I) || isa<CoroSuspendAsyncInst>(I)) {
      UseBB = UseBB->getSinglePredecessor();
      assert(UseBB && "should have split coro.suspend into its own block");
    }

    return hasPathCrossingSuspendPoint(DefBB, UseBB);
  }

  bool isDefinitionAcrossSuspend(Argument &A, User *U) const {
    return isDefinitionAcrossSuspend(&A.getParent()->getEntryBlock(), U);
  }

  bool isDefinitionAcrossSuspend(Instruction &I, User *U) const {
    auto *DefBB = I.getParent();

    // As a special case, treat values produced by an llvm.coro.suspend.*
    // as if they were defined in the single successor: the uses
    // conceptually occur after the suspend.
    if (isa<AnyCoroSuspendInst>(I)) {
      DefBB = DefBB->getSingleSuccessor();
      assert(DefBB && "should have split coro.suspend into its own block");
    }

    return isDefinitionAcrossSuspend(DefBB, U);
  }

  bool isDefinitionAcrossSuspend(Value &V, User *U) const {
    if (auto *Arg = dyn_cast<Argument>(&V))
      return isDefinitionAcrossSuspend(*Arg, U);
    if (auto *Inst = dyn_cast<Instruction>(&V))
      return isDefinitionAcrossSuspend(*Inst, U);

    llvm_unreachable(
        "Coroutine could only collect Argument and Instruction now.");
  }

  bool isDefinitionAcrossSuspend(Value &V) const {
    if (auto *Arg = dyn_cast<Argument>(&V)) {
      for (User *U : Arg->users())
        if (isDefinitionAcrossSuspend(*Arg, U))
          return true;
    } else if (auto *Inst = dyn_cast<Instruction>(&V)) {
      for (User *U : Inst->users())
        if (isDefinitionAcrossSuspend(*Inst, U))
          return true;
    }

    llvm_unreachable(
        "Coroutine could only collect Argument and Instruction now.");
  }
};

} // namespace llvm

#endif // LLVM_TRANSFORMS_COROUTINES_SUSPENDCROSSINGINFO_H

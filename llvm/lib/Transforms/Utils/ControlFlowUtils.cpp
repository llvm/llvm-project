//===- ControlFlowUtils.cpp - Control Flow Utilities -----------------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Utilities to manipulate the CFG and restore SSA for the new control flow.
//
//===----------------------------------------------------------------------===//

#include "llvm/Transforms/Utils/ControlFlowUtils.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/SmallSet.h"
#include "llvm/Analysis/DomTreeUpdater.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/ValueHandle.h"
#include "llvm/Transforms/Utils/Local.h"

#define DEBUG_TYPE "control-flow-hub"

using namespace llvm;

using BBPredicates = DenseMap<BasicBlock *, Instruction *>;
using EdgeDescriptor = ControlFlowHub::BranchDescriptor;

// Redirects the terminator of the incoming block to the first guard block in
// the hub. Returns the branch condition from `BB` if it exits.
// - If only one of Succ0 or Succ1 is not null, the corresponding branch
//   successor is redirected to the FirstGuardBlock.
// - Else both are not null, and branch is replaced with an unconditional
//   branch to the FirstGuardBlock.
static Value *redirectToHub(BasicBlock *BB, BasicBlock *Succ0,
                            BasicBlock *Succ1, BasicBlock *FirstGuardBlock) {
  assert(isa<BranchInst>(BB->getTerminator()) &&
         "Only support branch terminator.");
  auto *Branch = cast<BranchInst>(BB->getTerminator());
  auto *Condition = Branch->isConditional() ? Branch->getCondition() : nullptr;

  assert(Succ0 || Succ1);

  if (Branch->isUnconditional()) {
    assert(Succ0 == Branch->getSuccessor(0));
    assert(!Succ1);
    Branch->setSuccessor(0, FirstGuardBlock);
  } else {
    assert(!Succ1 || Succ1 == Branch->getSuccessor(1));
    if (Succ0 && !Succ1) {
      Branch->setSuccessor(0, FirstGuardBlock);
    } else if (Succ1 && !Succ0) {
      Branch->setSuccessor(1, FirstGuardBlock);
    } else {
      Branch->eraseFromParent();
      BranchInst::Create(FirstGuardBlock, BB);
    }
  }

  return Condition;
}

// Setup the branch instructions for guard blocks.
//
// Each guard block terminates in a conditional branch that transfers
// control to the corresponding outgoing block or the next guard
// block. The last guard block has two outgoing blocks as successors.
static void setupBranchForGuard(ArrayRef<BasicBlock *> GuardBlocks,
                                ArrayRef<BasicBlock *> Outgoing,
                                BBPredicates &GuardPredicates) {
  assert(Outgoing.size() > 1);
  assert(GuardBlocks.size() == Outgoing.size() - 1);
  int I = 0;
  for (int E = GuardBlocks.size() - 1; I != E; ++I) {
    BasicBlock *Out = Outgoing[I];
    BranchInst::Create(Out, GuardBlocks[I + 1], GuardPredicates[Out],
                       GuardBlocks[I]);
  }
  BasicBlock *Out = Outgoing[I];
  BranchInst::Create(Out, Outgoing[I + 1], GuardPredicates[Out],
                     GuardBlocks[I]);
}

// Assign an index to each outgoing block. At the corresponding guard
// block, compute the branch condition by comparing this index.
static void calcPredicateUsingInteger(ArrayRef<EdgeDescriptor> Branches,
                                      ArrayRef<BasicBlock *> Outgoing,
                                      ArrayRef<BasicBlock *> GuardBlocks,
                                      BBPredicates &GuardPredicates) {
  LLVMContext &Context = GuardBlocks.front()->getContext();
  BasicBlock *FirstGuardBlock = GuardBlocks.front();
  Type *Int32Ty = Type::getInt32Ty(Context);

  auto *Phi = PHINode::Create(Int32Ty, Branches.size(), "merged.bb.idx",
                              FirstGuardBlock);

  for (auto [BB, Succ0, Succ1] : Branches) {
    Value *Condition = redirectToHub(BB, Succ0, Succ1, FirstGuardBlock);
    Value *IncomingId = nullptr;
    if (Succ0 && Succ1) {
      auto Succ0Iter = find(Outgoing, Succ0);
      auto Succ1Iter = find(Outgoing, Succ1);
      Value *Id0 =
          ConstantInt::get(Int32Ty, std::distance(Outgoing.begin(), Succ0Iter));
      Value *Id1 =
          ConstantInt::get(Int32Ty, std::distance(Outgoing.begin(), Succ1Iter));
      IncomingId = SelectInst::Create(Condition, Id0, Id1, "target.bb.idx",
                                      BB->getTerminator()->getIterator());
    } else {
      // Get the index of the non-null successor.
      auto SuccIter = Succ0 ? find(Outgoing, Succ0) : find(Outgoing, Succ1);
      IncomingId =
          ConstantInt::get(Int32Ty, std::distance(Outgoing.begin(), SuccIter));
    }
    Phi->addIncoming(IncomingId, BB);
  }

  for (int I = 0, E = Outgoing.size() - 1; I != E; ++I) {
    BasicBlock *Out = Outgoing[I];
    LLVM_DEBUG(dbgs() << "Creating integer guard for " << Out->getName()
                      << "\n");
    auto *Cmp = ICmpInst::Create(Instruction::ICmp, ICmpInst::ICMP_EQ, Phi,
                                 ConstantInt::get(Int32Ty, I),
                                 Out->getName() + ".predicate", GuardBlocks[I]);
    GuardPredicates[Out] = Cmp;
  }
}

// Determine the branch condition to be used at each guard block from the
// original boolean values.
static void calcPredicateUsingBooleans(
    ArrayRef<EdgeDescriptor> Branches, ArrayRef<BasicBlock *> Outgoing,
    SmallVectorImpl<BasicBlock *> &GuardBlocks, BBPredicates &GuardPredicates,
    SmallVectorImpl<WeakVH> &DeletionCandidates) {
  LLVMContext &Context = GuardBlocks.front()->getContext();
  auto *BoolTrue = ConstantInt::getTrue(Context);
  auto *BoolFalse = ConstantInt::getFalse(Context);
  BasicBlock *FirstGuardBlock = GuardBlocks.front();

  // The predicate for the last outgoing is trivially true, and so we
  // process only the first N-1 successors.
  for (int I = 0, E = Outgoing.size() - 1; I != E; ++I) {
    BasicBlock *Out = Outgoing[I];
    LLVM_DEBUG(dbgs() << "Creating boolean guard for " << Out->getName()
                      << "\n");

    auto *Phi =
        PHINode::Create(Type::getInt1Ty(Context), Branches.size(),
                        StringRef("Guard.") + Out->getName(), FirstGuardBlock);
    GuardPredicates[Out] = Phi;
  }

  for (auto [BB, Succ0, Succ1] : Branches) {
    Value *Condition = redirectToHub(BB, Succ0, Succ1, FirstGuardBlock);

    // Optimization: Consider an incoming block A with both successors
    // Succ0 and Succ1 in the set of outgoing blocks. The predicates
    // for Succ0 and Succ1 complement each other. If Succ0 is visited
    // first in the loop below, control will branch to Succ0 using the
    // corresponding predicate. But if that branch is not taken, then
    // control must reach Succ1, which means that the incoming value of
    // the predicate from `BB` is true for Succ1.
    bool OneSuccessorDone = false;
    for (int I = 0, E = Outgoing.size() - 1; I != E; ++I) {
      BasicBlock *Out = Outgoing[I];
      PHINode *Phi = cast<PHINode>(GuardPredicates[Out]);
      if (Out != Succ0 && Out != Succ1) {
        Phi->addIncoming(BoolFalse, BB);
      } else if (!Succ0 || !Succ1 || OneSuccessorDone) {
        // Optimization: When only one successor is an outgoing block,
        // the incoming predicate from `BB` is always true.
        Phi->addIncoming(BoolTrue, BB);
      } else {
        assert(Succ0 && Succ1);
        if (Out == Succ0) {
          Phi->addIncoming(Condition, BB);
        } else {
          Value *Inverted = invertCondition(Condition);
          DeletionCandidates.push_back(Condition);
          Phi->addIncoming(Inverted, BB);
        }
        OneSuccessorDone = true;
      }
    }
  }
}

// Capture the existing control flow as guard predicates, and redirect
// control flow from \p Incoming block through the \p GuardBlocks to the
// \p Outgoing blocks.
//
// There is one guard predicate for each outgoing block OutBB. The
// predicate represents whether the hub should transfer control flow
// to OutBB. These predicates are NOT ORTHOGONAL. The Hub evaluates
// them in the same order as the Outgoing set-vector, and control
// branches to the first outgoing block whose predicate evaluates to true.
//
// The last guard block has two outgoing blocks as successors since the
// condition for the final outgoing block is trivially true. So we create one
// less block (including the first guard block) than the number of outgoing
// blocks.
static void convertToGuardPredicates(
    ArrayRef<EdgeDescriptor> Branches, ArrayRef<BasicBlock *> Outgoing,
    SmallVectorImpl<BasicBlock *> &GuardBlocks,
    SmallVectorImpl<WeakVH> &DeletionCandidates, const StringRef Prefix,
    std::optional<unsigned> MaxControlFlowBooleans) {
  BBPredicates GuardPredicates;
  Function *F = Outgoing.front()->getParent();

  for (int I = 0, E = Outgoing.size() - 1; I != E; ++I)
    GuardBlocks.push_back(
        BasicBlock::Create(F->getContext(), Prefix + ".guard", F));

  // When we are using an integer to record which target block to jump to, we
  // are creating less live values, actually we are using one single integer to
  // store the index of the target block. When we are using booleans to store
  // the branching information, we need (N-1) boolean values, where N is the
  // number of outgoing block.
  if (!MaxControlFlowBooleans || Outgoing.size() <= *MaxControlFlowBooleans)
    calcPredicateUsingBooleans(Branches, Outgoing, GuardBlocks, GuardPredicates,
                               DeletionCandidates);
  else
    calcPredicateUsingInteger(Branches, Outgoing, GuardBlocks, GuardPredicates);

  setupBranchForGuard(GuardBlocks, Outgoing, GuardPredicates);
}

// After creating a control flow hub, the operands of PHINodes in an outgoing
// block Out no longer match the predecessors of that block. Predecessors of Out
// that are incoming blocks to the hub are now replaced by just one edge from
// the hub. To match this new control flow, the corresponding values from each
// PHINode must now be moved a new PHINode in the first guard block of the hub.
//
// This operation cannot be performed with SSAUpdater, because it involves one
// new use: If the block Out is in the list of Incoming blocks, then the newly
// created PHI in the Hub will use itself along that edge from Out to Hub.
static void reconnectPhis(BasicBlock *Out, BasicBlock *GuardBlock,
                          ArrayRef<EdgeDescriptor> Incoming,
                          BasicBlock *FirstGuardBlock) {
  auto I = Out->begin();
  while (I != Out->end() && isa<PHINode>(I)) {
    auto *Phi = cast<PHINode>(I);
    auto *NewPhi =
        PHINode::Create(Phi->getType(), Incoming.size(),
                        Phi->getName() + ".moved", FirstGuardBlock->begin());
    bool AllUndef = true;
    for (auto [BB, Succ0, Succ1] : Incoming) {
      Value *V = PoisonValue::get(Phi->getType());
      if (BB == Out) {
        V = NewPhi;
      } else if (Phi->getBasicBlockIndex(BB) != -1) {
        V = Phi->removeIncomingValue(BB, false);
        AllUndef &= isa<UndefValue>(V);
      }
      NewPhi->addIncoming(V, BB);
    }
    assert(NewPhi->getNumIncomingValues() == Incoming.size());
    Value *NewV = NewPhi;
    if (AllUndef) {
      NewPhi->eraseFromParent();
      NewV = PoisonValue::get(Phi->getType());
    }
    if (Phi->getNumOperands() == 0) {
      Phi->replaceAllUsesWith(NewV);
      I = Phi->eraseFromParent();
      continue;
    }
    Phi->addIncoming(NewV, GuardBlock);
    ++I;
  }
}

BasicBlock *ControlFlowHub::finalize(
    DomTreeUpdater *DTU, SmallVectorImpl<BasicBlock *> &GuardBlocks,
    const StringRef Prefix, std::optional<unsigned> MaxControlFlowBooleans) {
#ifndef NDEBUG
  SmallSet<BasicBlock *, 8> Incoming;
#endif
  SetVector<BasicBlock *> Outgoing;

  for (auto [BB, Succ0, Succ1] : Branches) {
#ifndef NDEBUG
    assert(Incoming.insert(BB).second && "Duplicate entry for incoming block.");
#endif
    if (Succ0)
      Outgoing.insert(Succ0);
    if (Succ1)
      Outgoing.insert(Succ1);
  }

  if (Outgoing.size() < 2)
    return Outgoing.front();

  SmallVector<DominatorTree::UpdateType, 16> Updates;
  if (DTU) {
    for (auto [BB, Succ0, Succ1] : Branches) {
      if (Succ0)
        Updates.push_back({DominatorTree::Delete, BB, Succ0});
      if (Succ1)
        Updates.push_back({DominatorTree::Delete, BB, Succ1});
    }
  }

  SmallVector<WeakVH, 8> DeletionCandidates;
  convertToGuardPredicates(Branches, Outgoing.getArrayRef(), GuardBlocks,
                           DeletionCandidates, Prefix, MaxControlFlowBooleans);
  BasicBlock *FirstGuardBlock = GuardBlocks.front();

  // Update the PHINodes in each outgoing block to match the new control flow.
  for (int I = 0, E = GuardBlocks.size(); I != E; ++I)
    reconnectPhis(Outgoing[I], GuardBlocks[I], Branches, FirstGuardBlock);
  // Process the Nth (last) outgoing block with the (N-1)th (last) guard block.
  reconnectPhis(Outgoing.back(), GuardBlocks.back(), Branches, FirstGuardBlock);

  if (DTU) {
    int NumGuards = GuardBlocks.size();

    for (auto [BB, Succ0, Succ1] : Branches)
      Updates.push_back({DominatorTree::Insert, BB, FirstGuardBlock});

    for (int I = 0; I != NumGuards - 1; ++I) {
      Updates.push_back({DominatorTree::Insert, GuardBlocks[I], Outgoing[I]});
      Updates.push_back(
          {DominatorTree::Insert, GuardBlocks[I], GuardBlocks[I + 1]});
    }
    // The second successor of the last guard block is an outgoing block instead
    // of having a "next" guard block.
    Updates.push_back({DominatorTree::Insert, GuardBlocks[NumGuards - 1],
                       Outgoing[NumGuards - 1]});
    Updates.push_back({DominatorTree::Insert, GuardBlocks[NumGuards - 1],
                       Outgoing[NumGuards]});
    DTU->applyUpdates(Updates);
  }

  for (auto I : DeletionCandidates) {
    if (I->use_empty())
      if (auto *Inst = dyn_cast_or_null<Instruction>(I))
        Inst->eraseFromParent();
  }

  return FirstGuardBlock;
}

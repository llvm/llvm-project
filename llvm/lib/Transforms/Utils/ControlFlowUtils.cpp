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
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/Analysis/BlockFrequencyInfo.h"
#include "llvm/Analysis/DomTreeUpdater.h"
#include "llvm/Analysis/LoopInfo.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/ProfDataUtils.h"
#include "llvm/IR/ValueHandle.h"
#include "llvm/Support/BranchProbability.h"
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
  if (auto *Branch = dyn_cast<UncondBrInst>(BB->getTerminator())) {
    assert(Succ0 == Branch->getSuccessor(0));
    assert(!Succ1);
    Branch->setSuccessor(FirstGuardBlock);
    return nullptr;
  }

  auto *Branch = cast<CondBrInst>(BB->getTerminator());
  auto *Condition = Branch->getCondition();

  assert(Succ0 || Succ1);
  assert(!Succ1 || Succ1 == Branch->getSuccessor(1));
  if (Succ0 && !Succ1) {
    Branch->setSuccessor(0, FirstGuardBlock);
  } else if (Succ1 && !Succ0) {
    Branch->setSuccessor(1, FirstGuardBlock);
  } else {
    Branch->eraseFromParent();
    UncondBrInst::Create(FirstGuardBlock, BB);
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
    CondBrInst::Create(GuardPredicates[Out], Out, GuardBlocks[I + 1],
                       GuardBlocks[I]);
  }
  BasicBlock *Out = Outgoing[I];
  CondBrInst::Create(GuardPredicates[Out], Out, Outgoing[I + 1],
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
    if (Succ0 && Succ1 && Succ0 != Succ1) {
      auto Succ0Iter = find(Outgoing, Succ0);
      auto Succ1Iter = find(Outgoing, Succ1);
      Value *Id0 =
          ConstantInt::get(Int32Ty, std::distance(Outgoing.begin(), Succ0Iter));
      Value *Id1 =
          ConstantInt::get(Int32Ty, std::distance(Outgoing.begin(), Succ1Iter));
      IncomingId = SelectInst::Create(Condition, Id0, Id1, "target.bb.idx",
                                      BB->getTerminator()->getIterator());
    } else {
      // Get the index of the non-null successor, or when both successors
      // are the same block, use that block's index directly.
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
      } else if (!Succ0 || !Succ1 || Succ0 == Succ1 || OneSuccessorDone) {
        // Optimization: When only one successor is an outgoing block,
        // or both successors are the same block, the incoming predicate
        // from `BB` is always true.
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
      if (Phi->getBasicBlockIndex(BB) != -1) {
        V = Phi->removeIncomingValue(BB, false);
        // When both successors are the same (Succ0 == Succ1), there are two
        // edges from BB to Out, so we need to remove the second PHI entry too.
        if (Succ0 && Succ1 && Succ0 == Succ1 &&
            Phi->getBasicBlockIndex(BB) != -1)
          Phi->removeIncomingValue(BB, false);
        if (BB == Out) {
          V = NewPhi;
        }
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

bool llvm::functionHasScalableBranchProfile(const Function &F) {
  if (F.getEntryCount().value_or(0) > 0)
    return true;
  for (const BasicBlock &BB : F) {
    if (const Instruction *Term = BB.getTerminator())
      if (hasBranchWeightMD(*Term))
        return true;
  }
  return false;
}

bool llvm::functionHasBranchWeightsWiderThan32Bits(const Function &F) {
  for (const BasicBlock &BB : F) {
    const Instruction *Term = BB.getTerminator();
    if (!Term)
      continue;
    const MDNode *MD = getBranchWeightMDNode(*Term);
    if (!MD)
      continue;
    for (unsigned I = getBranchWeightOffset(MD), E = MD->getNumOperands();
         I != E; ++I) {
      const auto *Weight = mdconst::dyn_extract<ConstantInt>(MD->getOperand(I));
      if (Weight && Weight->getValue().getActiveBits() > 32)
        return true;
    }
  }
  return false;
}

void llvm::recordBlocksBeforeTransform(
    const Function &F, SmallPtrSetImpl<const BasicBlock *> &BlocksSeenBefore) {
  for (const BasicBlock &BB : F)
    BlocksSeenBefore.insert(&BB);
}

// Returns false if adding Amount to Total would overflow.
static bool addWeight(uint64_t &Total, uint64_t Amount) {
  if (Total > UINT64_MAX - Amount)
    return false;
  Total += Amount;
  return true;
}

// Return the execution count for a block that existed when BFI was computed.
// Prefer profile counts when available; otherwise fall back to the raw block
// frequency.
static std::optional<uint64_t>
getBlockCount(BlockFrequencyInfo &BFI,
              const SmallPtrSetImpl<const BasicBlock *> &KnownBlocks,
              const BasicBlock *Block) {
  if (!KnownBlocks.contains(Block))
    return std::nullopt;
  if (std::optional<uint64_t> Count = BFI.getBlockProfileCount(Block))
    return Count;
  return BFI.getBlockFreq(Block).getFrequency();
}

// This edge's share of the block count: EdgeWeight / TotalWeight * block count.
// Empty when the total wrapped, or when a taken edge scales down to 0.
static std::optional<uint64_t>
scaleToBlockCount(BlockFrequencyInfo &BFI,
                  const SmallPtrSetImpl<const BasicBlock *> &KnownBlocks,
                  const BasicBlock *FromBlock, uint64_t EdgeWeight,
                  uint64_t TotalWeight) {
  if (!TotalWeight || EdgeWeight > TotalWeight)
    return std::nullopt;
  std::optional<uint64_t> Count = getBlockCount(BFI, KnownBlocks, FromBlock);
  if (!Count)
    return std::nullopt;
  if (!*Count || !EdgeWeight)
    return 0;
  uint64_t Scaled =
      BranchProbability::getBranchProbability(EdgeWeight, TotalWeight)
          .scale(*Count);
  if (!Scaled)
    return std::nullopt;
  return Scaled;
}

// Profile information for an edge redirected through the guard hub.
struct RedirectedEdge {
  BasicBlock *Target = nullptr;
  // Estimated count reaching Target.
  std::optional<uint64_t> Count;
  // Source terminator had branch_weights.
  bool HadBranchWeights = false;
};

// False when a weight is not an integer or needs more than 64 bits.
static bool readBranchWeights64(const MDNode &MD,
                                SmallVectorImpl<uint64_t> &Weights) {
  Weights.clear();
  unsigned Offset = getBranchWeightOffset(&MD);
  if (Offset >= MD.getNumOperands())
    return false;
  for (unsigned I = Offset, E = MD.getNumOperands(); I != E; ++I) {
    const auto *Weight = mdconst::dyn_extract<ConstantInt>(MD.getOperand(I));
    if (!Weight || Weight->getValue().getActiveBits() > 64)
      return false;
    Weights.push_back(Weight->getValue().getZExtValue());
  }
  return true;
}

// Recover the count for a newly-created split block from its predecessor's
// count and branch weights.
static RedirectedEdge
countFromPredecessor(BlockFrequencyInfo &BFI,
                     const SmallPtrSetImpl<const BasicBlock *> &KnownBlocks,
                     const BasicBlock *SplitBlock) {
  RedirectedEdge Result;
  const BasicBlock *Pred = SplitBlock->getUniquePredecessor();
  if (!Pred)
    return Result;
  const Instruction *Term = Pred->getTerminator();
  if (!Term)
    return Result;
  unsigned NumSuccessors = Term->getNumSuccessors();
  const MDNode *MD = getBranchWeightMDNode(*Term);
  // Need per-successor weights to reconstruct the flow into SplitBlock.
  if (!MD || NumSuccessors < 2) {
    Result.HadBranchWeights = hasBranchWeightMD(*Term);
    return Result;
  }
  SmallVector<uint64_t, 8> Weights;
  if (!readBranchWeights64(*MD, Weights) || Weights.size() != NumSuccessors) {
    Result.HadBranchWeights = true;
    return Result;
  }
  Result.HadBranchWeights = true;
  uint64_t WeightToSplitBlock = 0;
  uint64_t TotalWeight = 0;
  bool SawSplitBlockEdge = false;
  for (unsigned I = 0; I != NumSuccessors; ++I) {
    if (!addWeight(TotalWeight, Weights[I]))
      return Result;
    if (Term->getSuccessor(I) != SplitBlock)
      continue;
    if (!addWeight(WeightToSplitBlock, Weights[I]))
      return Result;
    SawSplitBlockEdge = true;
  }
  if (!SawSplitBlockEdge)
    return Result;
  Result.Count = scaleToBlockCount(BFI, KnownBlocks, Pred, WeightToSplitBlock,
                                   TotalWeight);
  return Result;
}

// Edges redirected through the hub, plus whether the function already has
// profile data those edges can be annotated from.
struct CollectedRedirectedFlows {
  SmallVector<RedirectedEdge, 8> Flows;
  // Positive function_entry_count.
  bool HasPositiveEntryCount = false;
  // A redirected edge had branch_weights.
  bool SawBranchWeights = false;
};

// Counts for both edges of a conditional branch. The 32-bit weight helper
// drops the top bits, so the 64-bit metadata is read here.
static void addConditionalBranchFlows(
    const CondBrInst &Br, BasicBlock *FromBlock, BasicBlock *FirstTarget,
    BasicBlock *SecondTarget, BlockFrequencyInfo *BFI,
    const SmallPtrSetImpl<const BasicBlock *> &BlocksSeenBefore,
    function_ref<void(BasicBlock *, bool, std::optional<uint64_t>)> AddFlow) {
  const MDNode *MD = getBranchWeightMDNode(Br);
  // Both arms to one block carry that block's whole count.
  if (FirstTarget == SecondTarget) {
    std::optional<uint64_t> Count;
    if (BFI)
      Count = getBlockCount(*BFI, BlocksSeenBefore, FromBlock);
    AddFlow(FirstTarget, MD != nullptr, Count);
    return;
  }
  if (!MD) {
    AddFlow(FirstTarget, /*HadBranchWeights=*/false, std::nullopt);
    AddFlow(SecondTarget, /*HadBranchWeights=*/false, std::nullopt);
    return;
  }
  // Counts are unused without BFI.
  if (!BFI) {
    AddFlow(FirstTarget, /*HadBranchWeights=*/true, std::nullopt);
    AddFlow(SecondTarget, /*HadBranchWeights=*/true, std::nullopt);
    return;
  }
  SmallVector<uint64_t, 2> Weights;
  // A weight past 64 bits leaves both targets unknown.
  if (!readBranchWeights64(*MD, Weights) || Weights.size() != 2) {
    AddFlow(FirstTarget, /*HadBranchWeights=*/true, std::nullopt);
    AddFlow(SecondTarget, /*HadBranchWeights=*/true, std::nullopt);
    return;
  }
  uint64_t TrueWeight = Weights[0], FalseWeight = Weights[1];
  uint64_t TotalWeight = 0;
  bool CanScale =
      addWeight(TotalWeight, TrueWeight) && addWeight(TotalWeight, FalseWeight);
  std::optional<uint64_t> TrueCount, FalseCount;
  if (CanScale) {
    TrueCount = scaleToBlockCount(*BFI, BlocksSeenBefore, FromBlock, TrueWeight,
                                  TotalWeight);
    FalseCount = scaleToBlockCount(*BFI, BlocksSeenBefore, FromBlock,
                                   FalseWeight, TotalWeight);
  }
  AddFlow(FirstTarget, /*HadBranchWeights=*/true, TrueCount);
  AddFlow(SecondTarget, /*HadBranchWeights=*/true, FalseCount);
}

// Record each redirected edge.
// BFI only covers blocks that existed when it was computed. Newly-created
// hub/split blocks are therefore excluded from count recovery.
static CollectedRedirectedFlows
collectRedirectedFlows(ArrayRef<EdgeDescriptor> Branches,
                       const SmallPtrSetImpl<BasicBlock *> &SplitTargets,
                       BlockFrequencyInfo *BFI,
                       const SmallPtrSetImpl<const BasicBlock *> *KnownBlocks) {
  CollectedRedirectedFlows Result;
  SmallPtrSet<const BasicBlock *, 1> Empty;
  const SmallPtrSetImpl<const BasicBlock *> &BlocksSeenBefore =
      KnownBlocks ? *KnownBlocks : Empty;
  if (!Branches.empty()) {
    const Function *Fn = Branches.front().BB->getParent();
    Result.HasPositiveEntryCount = Fn->getEntryCount().value_or(0) > 0;
  }
  auto addFlow = [&](BasicBlock *Target, bool HadBranchWeights,
                     std::optional<uint64_t> Count = std::nullopt) {
    if (!Target)
      return;
    RedirectedEdge Edge;
    Edge.Target = Target;
    Edge.HadBranchWeights = HadBranchWeights;
    Edge.Count = Count;
    Result.Flows.push_back(Edge);
  };
  for (auto [FromBlock, FirstTarget, SecondTarget] : Branches) {
    // For blocks introduced while splitting edges/guards, recover the flow
    // from the original predecessor rather than from the synthetic block.
    if (SplitTargets.contains(FromBlock)) {
      BasicBlock *OriginalTarget = FirstTarget ? FirstTarget : SecondTarget;
      if (!BFI) {
        bool HadBranchWeights = false;
        if (const BasicBlock *Pred = FromBlock->getUniquePredecessor())
          if (const Instruction *Term = Pred->getTerminator())
            HadBranchWeights = hasBranchWeightMD(*Term);
        addFlow(OriginalTarget, HadBranchWeights);
      } else {
        RedirectedEdge Edge =
            countFromPredecessor(*BFI, BlocksSeenBefore, FromBlock);
        addFlow(OriginalTarget, Edge.HadBranchWeights, Edge.Count);
      }
      continue;
    }
    if (const auto *Br = dyn_cast<CondBrInst>(FromBlock->getTerminator())) {
      addConditionalBranchFlows(*Br, FromBlock, FirstTarget, SecondTarget, BFI,
                                BlocksSeenBefore, addFlow);
      continue;
    }
    if (!FirstTarget && !SecondTarget)
      continue;
    // Two different successors that are not a conditional branch. Neither
    // edge has a weight of its own.
    if (FirstTarget && SecondTarget && FirstTarget != SecondTarget) {
      addFlow(FirstTarget, /*HadBranchWeights=*/false);
      addFlow(SecondTarget, /*HadBranchWeights=*/false);
      continue;
    }
    BasicBlock *OnlyTarget = FirstTarget ? FirstTarget : SecondTarget;
    std::optional<uint64_t> Count;
    if (BFI)
      Count = getBlockCount(*BFI, BlocksSeenBefore, FromBlock);
    addFlow(OnlyTarget, /*HadBranchWeights=*/false, Count);
  }
  Result.SawBranchWeights =
      any_of(Result.Flows,
             [](const RedirectedEdge &Edge) { return Edge.HadBranchWeights; });
  return Result;
}

std::pair<BasicBlock *, bool> ControlFlowHub::finalize(
    DomTreeUpdater *DTU, SmallVectorImpl<BasicBlock *> &GuardBlocks,
    const StringRef Prefix, std::optional<unsigned> MaxControlFlowBooleans,
    ProfileInfo Profile) {
  BlockFrequencyInfo *BFI = Profile.BFI;
  const SmallPtrSetImpl<const BasicBlock *> *KnownBlocks = Profile.KnownBlocks;
#ifndef NDEBUG
  SmallPtrSet<BasicBlock *, 8> Incoming;
#endif
  SetVector<BasicBlock *> Outgoing;

  for (auto [BB, Succ0, Succ1] : Branches) {
#ifndef NDEBUG
    assert(
        (Incoming.insert(BB).second || isa<CallBrInst>(BB->getTerminator())) &&
        "Duplicate entry for incoming block.");
#endif
    if (Succ0)
      Outgoing.insert(Succ0);
    if (Succ1)
      Outgoing.insert(Succ1);
  }

  // Weights for the edges about to be redirected through the guards.
  CollectedRedirectedFlows Edges =
      collectRedirectedFlows(Branches, SplitTargets, BFI, KnownBlocks);
  // Already profiled: a guard with no recovered count is marked unknown.
  const bool MayAnnotate =
      Edges.SawBranchWeights || Edges.HasPositiveEntryCount;
  auto markUnknown = [&](Instruction &I) {
    if (Edges.HasPositiveEntryCount)
      setExplicitlyUnknownBranchWeightsIfProfiled(I, DEBUG_TYPE);
    else if (Edges.SawBranchWeights)
      setExplicitlyUnknownBranchWeights(I, DEBUG_TYPE);
  };

  assert(Outgoing.size() && "No outgoing edges");

  if (Outgoing.size() < 2)
    return {Outgoing.front(), false};

  SmallVector<DominatorTree::UpdateType, 16> Updates;
  if (DTU) {
    for (auto [BB, Succ0, Succ1] : Branches) {
      if (Succ0)
        Updates.push_back({DominatorTree::Delete, BB, Succ0});
      // Only add Succ1 if it's different from Succ0 to avoid duplicate updates
      if (Succ1 && Succ1 != Succ0)
        Updates.push_back({DominatorTree::Delete, BB, Succ1});
    }
  }

  SmallVector<WeakVH, 8> DeletionCandidates;
  convertToGuardPredicates(Branches, Outgoing.getArrayRef(), GuardBlocks,
                           DeletionCandidates, Prefix, MaxControlFlowBooleans);

  if (MayAnnotate) {
    // [src] -> [guard0] -> [Out0]
    //              |
    //              +-> [guard1] -> [Out1]
    //                       |
    //                       +-> [Out2]
    // guard0 compares Out0 with the rest of the chain. A target that could
    // not be scaled makes that guard unknown. guard1 still gets a weight
    // when Out1 and Out2 could be scaled. The examples below have one guard
    // each, so both outgoing counts are written on that branch.
    //
    // FixIrreducible. Entry count 100. right branches back to left.
    // Both exits are the same block.
    //
    //                     +----------------------------+
    //                     v                            |
    // [entry] --25--> [left] --80--+                   |
    //    |              |          v                   |
    //    |              v          |                   |
    //    |           [exit]        |                   |
    //    +-------------75-----> [right]----------------+
    //                              |
    //                              v
    //                           [exit]
    //
    // entry->left  = 100 * 10/40 = 25
    // entry->right = 100 * 30/40 = 75
    // left->right scales to 80
    //
    // After. One [left]. entry's two edges and left->right are redirected
    // through [guard]. right->left still targets [left].
    // In: 100 + 80 = 180. Out: 155 + 25 = 180.
    // 155 = 75 + 80 to right. 25 is entry->left. 80 is left->guard.
    //
    //                                   +-----------------+
    //                                   v                 |
    // [entry] --100--> [guard] --25--> [left]             |
    //                     | ^           |  |              |
    //                     | |           |  |              |
    //                     | +----80-----+  |              |
    //                    155             [exit]           |
    //                     |                               |
    //                     v                               |
    //                  [right]----------------------------+
    //                     |
    //                     v
    //                  [exit]
    //
    // UnifyLoopExits. Entry count 100. [latch] branches back to [header].
    //
    //                      +------------------------+
    //                      v                        |
    // [entry] --100--> [header] --120--> [latch]----+
    //                     |                 |
    //                    40                60
    //                     v                 v
    //                 [exit.a]          [exit.b]
    //
    // header = 100 + 60 = 160
    // header->latch = 160 * 3/4 = 120
    // header->exit.a = 160 * 1/4 = 40
    // latch->header = latch->exit.b = 120 * 1/2 = 60
    //
    // After. Both exits go through [guard]. The back edge stays.
    // In: 40 + 60 = 100. Out: 40 + 60 = 100.
    //
    //                      +------------------------+
    //                      v                        |
    // [entry] --100--> [header] --120--> [latch]----+
    //                     |                 |
    //                    40                60
    //                     +--------+--------+
    //                              v
    //                          [guard] --40--> [exit.a]
    //                             |
    //                            60
    //                             v
    //                         [exit.b]
    DenseMap<BasicBlock *, uint64_t> TargetWeight;
    DenseSet<BasicBlock *> Unresolved;
    // If a redirected edge has no target, the corresponding flow cannot be
    // recovered, so every synthesized guard is marked unknown.
    bool MissingTarget = any_of(Branches, [&](const EdgeDescriptor &Branch) {
      auto [FromBlock, FirstTarget, SecondTarget] = Branch;
      return !FirstTarget && !SecondTarget &&
             !SplitTargets.contains(FromBlock) &&
             !isa<CondBrInst>(FromBlock->getTerminator());
    });
    if (!BFI || MissingTarget) {
      for (BasicBlock *Out : Outgoing)
        Unresolved.insert(Out);
    } else {
      for (const RedirectedEdge &Edge : Edges.Flows) {
        // Overflow while accumulating counts makes the target unusable.
        if (!Edge.Count || !addWeight(TargetWeight[Edge.Target], *Edge.Count))
          Unresolved.insert(Edge.Target);
      }
    }
    for (unsigned I = 0, E = GuardBlocks.size(); I != E; ++I) {
      auto *Br = cast<CondBrInst>(GuardBlocks[I]->getTerminator());
      if (Unresolved.contains(Outgoing[I])) {
        markUnknown(*Br);
        continue;
      }
      uint64_t ThisTargetCount = TargetWeight.lookup(Outgoing[I]);
      uint64_t RestCount = 0;
      bool CanAnnotate = true;
      for (unsigned J = I + 1; J != Outgoing.size(); ++J) {
        if (Unresolved.contains(Outgoing[J]) ||
            !addWeight(RestCount, TargetWeight.lookup(Outgoing[J]))) {
          CanAnnotate = false;
          break;
        }
      }
      if (!CanAnnotate) {
        markUnknown(*Br);
        continue;
      }
      // {0, 0} would look like the block never runs.
      if (ThisTargetCount || RestCount)
        setFittedBranchWeights(*Br, {ThisTargetCount, RestCount},
                               /*IsExpected=*/false);
      else
        markUnknown(*Br);
    }
  }
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

  return {FirstGuardBlock, true};
}

//===- LifetimeMove.cpp - Narrowing lifetimes -----------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// The LifetimeMovePass identifies the precise lifetime range of allocas and
// repositions lifetime markers to stricter positions.
//
//===----------------------------------------------------------------------===//

#include "llvm/Transforms/Utils/LifetimeMove.h"
#include "llvm/Analysis/CFG.h"
#include "llvm/Analysis/CaptureTracking.h"
#include "llvm/Analysis/LoopInfo.h"
#include "llvm/Analysis/PostDominators.h"
#include "llvm/Analysis/PtrUseVisitor.h"
#include "llvm/IR/Dominators.h"
#include "llvm/IR/InstIterator.h"
#include "llvm/Transforms/Coroutines/CoroInstr.h"

#define DEBUG_TYPE "lifetime-move"

namespace llvm {
namespace {
class LifetimeMover : public PtrUseVisitor<LifetimeMover> {
  using This = LifetimeMover;
  using Base = PtrUseVisitor<LifetimeMover>;

  const DominatorTree &DT;
  const LoopInfo &LI;

  SmallVector<AllocaInst *, 4> Allocas;
  // Critical points are instructions where the crossing of a variable's
  // lifetime makes a difference. We attempt to rise lifetime.end
  // before critical points and sink lifetime.start after them.
  SmallVector<Instruction *, 4> CriticalPoints;

  SmallVector<Instruction *, 2> LifetimeStarts;
  SmallVector<Instruction *, 2> LifetimeEnds;
  SmallVector<Instruction *, 8> OtherUsers;
  SmallPtrSet<BasicBlock *, 2> LifetimeStartBBs;
  SmallPtrSet<BasicBlock *, 2> UserBBs;

public:
  LifetimeMover(Function &F, const DominatorTree &DT, const LoopInfo &LI);

  bool run();

  void visitInstruction(Instruction &I);
  void visitPHINode(PHINode &I);
  void visitSelectInst(SelectInst &I);
  void visitStoreInst(StoreInst &SI);
  void visitIntrinsicInst(IntrinsicInst &II);
  void visitMemIntrinsic(MemIntrinsic &I);
  void visitCallBase(CallBase &CB);

private:
  bool sinkLifetimeStartMarkers(AllocaInst *AI);
  bool riseLifetimeEndMarkers();
  void reset();
};
} // namespace

LifetimeMover::LifetimeMover(Function &F, const DominatorTree &DT,
                             const LoopInfo &LI)
    : Base(F.getDataLayout()), DT(DT), LI(LI) {
  for (Instruction &I : instructions(F)) {
    if (auto *AI = dyn_cast<AllocaInst>(&I))
      Allocas.push_back(AI);
    else if (isa<LifetimeIntrinsic>(I))
      continue;
    else if (isa<AnyCoroSuspendInst>(I))
      CriticalPoints.push_back(&I);
    else if (isa<CallInst>(I))
      CriticalPoints.push_back(&I);
    else if (isa<InvokeInst>(I))
      CriticalPoints.push_back(&I);
  }
}

bool LifetimeMover::run() {
  bool Changed = false;
  for (auto *AI : Allocas) {
    reset();
    Base::visitPtr(*AI);

    if (!LifetimeStarts.empty())
      Changed |= sinkLifetimeStartMarkers(AI);

    // Do not move lifetime.end if alloca escapes
    if (!LifetimeEnds.empty() && !PI.isEscaped())
      Changed |= riseLifetimeEndMarkers();
  }
  return Changed;
}

void LifetimeMover::visitInstruction(Instruction &I) {
  OtherUsers.push_back(&I);
  UserBBs.insert(I.getParent());
}

void LifetimeMover::visitPHINode(PHINode &I) { enqueueUsers(I); }

void LifetimeMover::visitSelectInst(SelectInst &I) { enqueueUsers(I); }

void LifetimeMover::visitStoreInst(StoreInst &SI) {
  if (SI.getPointerOperand() == U->get())
    return InstVisitor<This>::visitStoreInst(SI);

  // We are storing the pointer into a memory location, potentially escaping.
  // As an optimization, we try to detect simple cases where it doesn't
  // actually escape, for example:
  //   %ptr = alloca ..
  //   %addr = alloca ..
  //   store %ptr, %addr
  //   %x = load %addr
  //   ..
  // If %addr is only used by loading from it, we could simply treat %x as
  // another alias of %ptr, and not considering %ptr being escaped.
  auto IsSimpleStoreThenLoad = [&]() {
    auto *AI = dyn_cast<AllocaInst>(SI.getPointerOperand());
    // If the memory location we are storing to is not an alloca, it
    // could be an alias of some other memory locations, which is difficult
    // to analyze.
    if (!AI)
      return false;
    // StoreAliases contains aliases of the memory location stored into.
    SmallVector<Instruction *, 4> StoreAliases = {AI};
    while (!StoreAliases.empty()) {
      Instruction *I = StoreAliases.pop_back_val();
      for (User *U : I->users()) {
        // If we are loading from the memory location, we are creating an
        // alias of the original pointer.
        if (auto *LI = dyn_cast<LoadInst>(U)) {
          enqueueUsers(*LI);
          continue;
        }
        // If we are overriding the memory location, the pointer certainly
        // won't escape.
        if (auto *S = dyn_cast<StoreInst>(U))
          if (S->getPointerOperand() == I)
            continue;
        if (isa<LifetimeIntrinsic>(U))
          continue;
        // BitCastInst creats aliases of the memory location being stored
        // into.
        if (auto *BI = dyn_cast<BitCastInst>(U)) {
          StoreAliases.push_back(BI);
          continue;
        }
        return false;
      }
    }

    return true;
  };

  if (!IsSimpleStoreThenLoad())
    PI.setEscaped(&SI);
  InstVisitor<This>::visitStoreInst(SI);
}

void LifetimeMover::visitIntrinsicInst(IntrinsicInst &II) {
  // lifetime markers are not actual uses
  switch (II.getIntrinsicID()) {
  case Intrinsic::lifetime_start:
    LifetimeStarts.push_back(&II);
    LifetimeStartBBs.insert(II.getParent());
    break;
  case Intrinsic::lifetime_end:
    LifetimeEnds.push_back(&II);
    break;
  default:
    Base::visitIntrinsicInst(II);
  }
}

void LifetimeMover::visitMemIntrinsic(MemIntrinsic &I) { visitInstruction(I); }

void LifetimeMover::visitCallBase(CallBase &CB) {
  for (unsigned Op = 0, OpCount = CB.arg_size(); Op < OpCount; ++Op)
    if (U->get() == CB.getArgOperand(Op) && !CB.doesNotCapture(Op))
      PI.setEscaped(&CB);
  InstVisitor<This>::visitCallBase(CB);
}
/// For each local variable that all of its user are dominated by one of the
/// critical point, we sink their lifetime.start markers to the place where
/// after the critical point. Doing so minimizes the lifetime of each variable.
bool LifetimeMover::sinkLifetimeStartMarkers(AllocaInst *AI) {
  auto Update = [this](Instruction *Old, Instruction *New) {
    // Reject the new proposal if it lengthens lifetime
    if (DT.dominates(New, Old))
      return Old;

    if (LI.getLoopFor(New->getParent()))
      return Old;

    bool DomAll = llvm::all_of(UserBBs, [this, New](BasicBlock *UserBB) {
      // Instruction level analysis if lifetime and users share a common BB
      BasicBlock *NewBB = New->getParent();
      if (UserBB == NewBB) {
        return llvm::all_of(OtherUsers, [New, UserBB](Instruction *I) {
          return UserBB != I->getParent() || New->comesBefore(I);
        });
      }
      // Otherwise, BB level analysis is enough
      return DT.dominates(New, UserBB);
    });
    return DomAll ? New : Old;
  };

  // AllocaInst is a trivial critical point
  Instruction *DomPoint = AI;
  for (auto *P : CriticalPoints)
    DomPoint = Update(DomPoint, P);

  // Sink lifetime.start markers to dominate block when they are
  // only used outside the region.
  if (DomPoint != AI) {
    // If existing position is better, do nothing
    for (auto *P : LifetimeStarts) {
      if (P == Update(DomPoint, P))
        return false;
    }

    auto *NewStart = LifetimeStarts[0]->clone();
    NewStart->replaceUsesOfWith(NewStart->getOperand(0), AI);
    if (DomPoint->isTerminator())
      NewStart->insertBefore(
          cast<InvokeInst>(DomPoint)->getNormalDest()->getFirstNonPHIIt());
    else
      NewStart->insertAfter(DomPoint->getIterator());

    // All the outsided lifetime.start markers are no longer necessary.
    for (auto *I : LifetimeStarts) {
      if (LI.getLoopFor(I->getParent()))
        continue;

      bool Restart = llvm::any_of(LifetimeEnds, [this, I](Instruction *End) {
        return isPotentiallyReachable(End, I, &LifetimeStartBBs, &DT, &LI);
      });

      if (!Restart) {
        LifetimeStartBBs.erase(I->getParent());
        I->eraseFromParent();
      }
    }
    return true;
  }
  return false;
}
// Find the critical point that is dominated by all users of alloca,
// we will rise lifetime.end markers before the critical point.
bool LifetimeMover::riseLifetimeEndMarkers() {
  auto Update = [this](Instruction *Old, Instruction *New) {
    if (Old != nullptr && DT.dominates(Old, New))
      return Old;

    if (LI.getLoopFor(New->getParent()))
      return Old;

    bool DomAll = llvm::all_of(UserBBs, [this, New](BasicBlock *UserBB) {
      BasicBlock *NewBB = New->getParent();
      if (UserBB == NewBB) {
        return llvm::all_of(OtherUsers, [New, UserBB](Instruction *I) {
          return UserBB != I->getParent() || I->comesBefore(New);
        });
      }

      if (auto *L = LI.getLoopFor(UserBB)) {
        SmallVector<BasicBlock *, 2> EBs;
        L->getOutermostLoop()->getExitingBlocks(EBs);
        return llvm::all_of(EBs, [this, NewBB](BasicBlock *EB) {
          return DT.dominates(EB, NewBB);
        });
      }

      return DT.dominates(UserBB, NewBB);
    });
    return DomAll ? New : Old;
  };

  Instruction *DomPoint = nullptr;
  for (auto *P : CriticalPoints)
    DomPoint = Update(DomPoint, P);

  if (DomPoint != nullptr) {
    for (auto *P : LifetimeEnds) {
      if (P == Update(DomPoint, P))
        return false;
    }

    auto *NewEnd = LifetimeEnds[0]->clone();
    NewEnd->insertBefore(DomPoint->getIterator());

    for (auto *I : LifetimeEnds)
      if (!LI.getLoopFor(I->getParent()))
        I->eraseFromParent();
    return true;
  }
  return false;
}

void LifetimeMover::reset() {
  PI.reset();
  Worklist.clear();
  VisitedUses.clear();

  LifetimeStarts.clear();
  LifetimeEnds.clear();
  OtherUsers.clear();
  LifetimeStartBBs.clear();
  UserBBs.clear();
}

PreservedAnalyses LifetimeMovePass::run(Function &F,
                                        FunctionAnalysisManager &AM) {
  // FIXME: Only enable by default for coroutines for now
  if (!F.isPresplitCoroutine())
    return PreservedAnalyses::all();

  const DominatorTree &DT = AM.getResult<DominatorTreeAnalysis>(F);
  const LoopInfo &LI = AM.getResult<LoopAnalysis>(F);
  LifetimeMover Mover(F, DT, LI);
  if (!Mover.run())
    return PreservedAnalyses::all();

  PreservedAnalyses PA;
  PA.preserveSet<CFGAnalyses>();
  return PA;
}
} // namespace llvm

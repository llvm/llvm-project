//===- LifetimeMove.cpp - Narrowing lifetimes -----------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// The LifetimeMovePass identifies the precise lifetime range of allocas
//
//===----------------------------------------------------------------------===//

#include "llvm/Transforms/Scalar/LifetimeMove.h"
#include "llvm/Analysis/CFG.h"
#include "llvm/Analysis/CaptureTracking.h"
#include "llvm/IR/Dominators.h"
#include "llvm/IR/InstIterator.h"
#include "llvm/Transforms/Coroutines/CoroInstr.h"

#define DEBUG_TYPE "lifetime-move"

namespace llvm {
static bool mayEscape(Value *V, User *U) {
  if (V == U->stripInBoundsOffsets() || isa<PHINode>(U))
    return true;

  if (auto *SI = dyn_cast<StoreInst>(U))
    return SI->getValueOperand() == V;

  if (auto *CB = dyn_cast<CallBase>(U)) {
    unsigned OpCount = CB->arg_size();
    for (unsigned Op = 0; Op < OpCount; ++Op)
      if (V == CB->getArgOperand(Op) && !CB->doesNotCapture(Op))
        return true;
  }
  return false;
}

namespace {
class LifetimeMover {
  const DominatorTree &DT;

  SmallVector<AllocaInst *, 4> Allocas;
  // Critical points are instructions where the crossing of a variable's
  // lifetime makes a difference. We attempt to move lifetime.end
  // before critical points and lifetime.start after them.
  SmallVector<Instruction *, 4> CriticalPoints;

  SmallVector<Instruction *, 2> LifetimeStarts;
  SmallVector<Instruction *, 2> LifetimeEnds;
  SmallVector<Instruction *, 8> OtherUsers;
  SmallPtrSet<BasicBlock *, 2> UserBBs;

public:
  LifetimeMover(Function &F, const DominatorTree &DT);

  void run();

private:
  void sinkLifetimeStartMarkers(AllocaInst *AI);
  void riseLifetimeEndMarkers();
  bool collectLifetime(Instruction *I);
  void reset();
};
} // namespace

LifetimeMover::LifetimeMover(Function &F, const DominatorTree &DT) : DT(DT) {
  for (Instruction &I : instructions(F)) {
    if (auto *AI = dyn_cast<AllocaInst>(&I))
      Allocas.push_back(AI);
    else if (isa<AnyCoroSuspendInst>(I))
      CriticalPoints.push_back(&I);
  }
}

void LifetimeMover::run() {
  for (auto *AI : Allocas) {
    reset();

    bool Escape = false;
    for (User *U : AI->users()) {
      auto *I = cast<Instruction>(U);
      // lifetime markers are not actual uses
      if (collectLifetime(I))
        continue;

      // GEP and bitcast used by lifetime markers
      if (U->hasOneUse() && U->stripPointerCasts() == AI) {
        auto *U1 = cast<Instruction>(U->user_back());
        if (collectLifetime(U1))
          continue;
      }

      Escape |= mayEscape(AI, U);
      OtherUsers.push_back(I);
      UserBBs.insert(I->getParent());
    }

    if (!LifetimeStarts.empty())
      sinkLifetimeStartMarkers(AI);

    // Do not move lifetime.end if alloca escapes
    if (!LifetimeEnds.empty() && !Escape)
      riseLifetimeEndMarkers();
  }
}
/// For each local variable that all of its user are dominated by one of the
/// critical point, we sink their lifetime.start markers to the place where
/// after the critical point. Doing so minimizes the lifetime of each variable.
void LifetimeMover::sinkLifetimeStartMarkers(AllocaInst *AI) {
  auto Update = [this](Instruction *Old, Instruction *New) {
    // Reject if the new proposal lengthens the lifetime
    if (DT.dominates(New, Old))
      return Old;

    bool DomAll = llvm::all_of(UserBBs, [this, New](BasicBlock *UserBB) {
      // Instruction level analysis if lifetime and users share a common BB
      if (UserBB == New->getParent()) {
        return llvm::all_of(OtherUsers, [this, New, UserBB](Instruction *I) {
          return UserBB != I->getParent() || DT.dominates(New, I);
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
    for (auto *P : LifetimeStarts)
      if (isPotentiallyReachable(DomPoint, P) && (P == Update(DomPoint, P)))
        return;

    auto *NewStart = LifetimeStarts[0]->clone();
    NewStart->replaceUsesOfWith(NewStart->getOperand(1), AI);
    NewStart->insertAfter(DomPoint->getIterator());

    // All the outsided lifetime.start markers are no longer necessary.
    for (auto *I : LifetimeStarts)
      if (DT.dominates(I, DomPoint))
        I->eraseFromParent();
  }
}
// Find the critical point that is dominated by all users of alloca,
// we will rise lifetime.end markers before the critical point.
void LifetimeMover::riseLifetimeEndMarkers() {
  auto Update = [this](Instruction *Old, Instruction *New) {
    if (Old != nullptr && DT.dominates(Old, New))
      return Old;

    bool DomAll = llvm::all_of(UserBBs, [this, New](BasicBlock *UserBB) {
      if (UserBB == New->getParent()) {
        return llvm::all_of(OtherUsers, [this, New, UserBB](Instruction *I) {
          return UserBB != I->getParent() || DT.dominates(I, New);
        });
      }
      return DT.dominates(UserBB, New->getParent());
    });
    return DomAll ? New : Old;
  };

  Instruction *DomPoint = nullptr;
  for (auto *P : CriticalPoints)
    DomPoint = Update(DomPoint, P);

  if (DomPoint != nullptr) {
    for (auto *P : LifetimeEnds)
      if (isPotentiallyReachable(P, DomPoint) && (P == Update(DomPoint, P)))
        return;

    auto *NewEnd = LifetimeEnds[0]->clone();
    NewEnd->insertBefore(DomPoint->getIterator());

    for (auto *I : LifetimeEnds)
      if (DT.dominates(DomPoint, I))
        I->eraseFromParent();
  }
}

bool LifetimeMover::collectLifetime(Instruction *I) {
  if (auto *II = dyn_cast<IntrinsicInst>(I)) {
    auto ID = II->getIntrinsicID();
    if (ID == Intrinsic::lifetime_start) {
      LifetimeStarts.push_back(I);
      return true;
    }

    if (ID == Intrinsic::lifetime_end) {
      LifetimeEnds.push_back(I);
      return true;
    }
  }
  return false;
}

void LifetimeMover::reset() {
  LifetimeStarts.clear();
  LifetimeEnds.clear();
  OtherUsers.clear();
  UserBBs.clear();
}

PreservedAnalyses LifetimeMovePass::run(Function &F,
                                        FunctionAnalysisManager &AM) {
  // Works for coroutine for now
  if (!F.isPresplitCoroutine())
    PreservedAnalyses::all();

  const DominatorTree &DT = AM.getResult<DominatorTreeAnalysis>(F);
  LifetimeMover Mover(F, DT);
  Mover.run();
  return PreservedAnalyses::all();
}
} // namespace llvm

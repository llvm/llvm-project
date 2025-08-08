//===- SpillUtils.cpp - Utilities for checking for spills ---------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/Transforms/Coroutines/SpillUtils.h"
#include "CoroInternal.h"
#include "llvm/Analysis/CFG.h"
#include "llvm/Analysis/PtrUseVisitor.h"
#include "llvm/IR/CFG.h"
#include "llvm/IR/DebugInfo.h"
#include "llvm/IR/Dominators.h"
#include "llvm/IR/InstIterator.h"
#include "llvm/Transforms/Utils/BasicBlockUtils.h"

namespace llvm {

namespace coro {

namespace {

typedef SmallPtrSet<BasicBlock *, 8> VisitedBlocksSet;

static bool isNonSpilledIntrinsic(Instruction &I) {
  // Structural coroutine intrinsics that should not be spilled into the
  // coroutine frame.
  return isa<CoroIdInst>(&I) || isa<CoroSaveInst>(&I);
}

/// Does control flow starting at the given block ever reach a suspend
/// instruction before reaching a block in VisitedOrFreeBBs?
static bool isSuspendReachableFrom(BasicBlock *From,
                                   VisitedBlocksSet &VisitedOrFreeBBs) {
  // Eagerly try to add this block to the visited set. If it's already
  // there, stop recursing; this path doesn't reach a suspend before
  // either looping or reaching a freeing block.
  if (!VisitedOrFreeBBs.insert(From).second)
    return false;

  // We assume that we'll already have split suspends into their own blocks.
  if (coro::isSuspendBlock(From))
    return true;

  // Recurse on the successors.
  for (auto *Succ : successors(From)) {
    if (isSuspendReachableFrom(Succ, VisitedOrFreeBBs))
      return true;
  }

  return false;
}

/// Is the given alloca "local", i.e. bounded in lifetime to not cross a
/// suspend point?
static bool isLocalAlloca(CoroAllocaAllocInst *AI) {
  // Seed the visited set with all the basic blocks containing a free
  // so that we won't pass them up.
  VisitedBlocksSet VisitedOrFreeBBs;
  for (auto *User : AI->users()) {
    if (auto FI = dyn_cast<CoroAllocaFreeInst>(User))
      VisitedOrFreeBBs.insert(FI->getParent());
  }

  return !isSuspendReachableFrom(AI->getParent(), VisitedOrFreeBBs);
}

/// Turn the given coro.alloca.alloc call into a dynamic allocation.
/// This happens during the all-instructions iteration, so it must not
/// delete the call.
static Instruction *
lowerNonLocalAlloca(CoroAllocaAllocInst *AI, const coro::Shape &Shape,
                    SmallVectorImpl<Instruction *> &DeadInsts) {
  IRBuilder<> Builder(AI);
  auto Alloc = Shape.emitAlloc(Builder, AI->getSize(), nullptr);

  for (User *U : AI->users()) {
    if (isa<CoroAllocaGetInst>(U)) {
      U->replaceAllUsesWith(Alloc);
    } else {
      auto FI = cast<CoroAllocaFreeInst>(U);
      Builder.SetInsertPoint(FI);
      Shape.emitDealloc(Builder, Alloc, nullptr);
    }
    DeadInsts.push_back(cast<Instruction>(U));
  }

  // Push this on last so that it gets deleted after all the others.
  DeadInsts.push_back(AI);

  // Return the new allocation value so that we can check for needed spills.
  return cast<Instruction>(Alloc);
}

// We need to make room to insert a spill after initial PHIs, but before
// catchswitch instruction. Placing it before violates the requirement that
// catchswitch, like all other EHPads must be the first nonPHI in a block.
//
// Split away catchswitch into a separate block and insert in its place:
//
//   cleanuppad <InsertPt> cleanupret.
//
// cleanupret instruction will act as an insert point for the spill.
static Instruction *splitBeforeCatchSwitch(CatchSwitchInst *CatchSwitch) {
  BasicBlock *CurrentBlock = CatchSwitch->getParent();
  BasicBlock *NewBlock = CurrentBlock->splitBasicBlock(CatchSwitch);
  CurrentBlock->getTerminator()->eraseFromParent();

  auto *CleanupPad =
      CleanupPadInst::Create(CatchSwitch->getParentPad(), {}, "", CurrentBlock);
  auto *CleanupRet =
      CleanupReturnInst::Create(CleanupPad, NewBlock, CurrentBlock);
  return CleanupRet;
}

// We use a pointer use visitor to track how an alloca is being used.
// The goal is to be able to answer the following three questions:
//   1. Should this alloca be allocated on the frame instead.
//   2. Could the content of the alloca be modified prior to CoroBegin, which
//      would require copying the data from the alloca to the frame after
//      CoroBegin.
//   3. Are there any aliases created for this alloca prior to CoroBegin, but
//      used after CoroBegin. In that case, we will need to recreate the alias
//      after CoroBegin based off the frame.
//
// To answer question 1, we track two things:
//   A. List of all BasicBlocks that use this alloca or any of the aliases of
//   the alloca. In the end, we check if there exists any two basic blocks that
//   cross suspension points. If so, this alloca must be put on the frame.
//   B. Whether the alloca or any alias of the alloca is escaped at some point,
//   either by storing the address somewhere, or the address is used in a
//   function call that might capture. If it's ever escaped, this alloca must be
//   put on the frame conservatively.
//
// To answer quetion 2, we track through the variable MayWriteBeforeCoroBegin.
// Whenever a potential write happens, either through a store instruction, a
// function call or any of the memory intrinsics, we check whether this
// instruction is prior to CoroBegin.
//
// To answer question 3, we track the offsets of all aliases created for the
// alloca prior to CoroBegin but used after CoroBegin. std::optional is used to
// be able to represent the case when the offset is unknown (e.g. when you have
// a PHINode that takes in different offset values). We cannot handle unknown
// offsets and will assert. This is the potential issue left out. An ideal
// solution would likely require a significant redesign.

namespace {
struct AllocaUseVisitor : PtrUseVisitor<AllocaUseVisitor> {
  using Base = PtrUseVisitor<AllocaUseVisitor>;
  AllocaUseVisitor(const DataLayout &DL, const DominatorTree &DT,
                   const coro::Shape &CoroShape,
                   const SuspendCrossingInfo &Checker,
                   bool ShouldUseLifetimeStartInfo)
      : PtrUseVisitor(DL), DT(DT), CoroShape(CoroShape), Checker(Checker),
        ShouldUseLifetimeStartInfo(ShouldUseLifetimeStartInfo) {
    for (AnyCoroSuspendInst *SuspendInst : CoroShape.CoroSuspends)
      CoroSuspendBBs.insert(SuspendInst->getParent());
  }

  void visit(Instruction &I) {
    Users.insert(&I);
    Base::visit(I);
    // If the pointer is escaped prior to CoroBegin, we have to assume it would
    // be written into before CoroBegin as well.
    if (PI.isEscaped() &&
        !DT.dominates(CoroShape.CoroBegin, PI.getEscapingInst())) {
      MayWriteBeforeCoroBegin = true;
    }
  }
  // We need to provide this overload as PtrUseVisitor uses a pointer based
  // visiting function.
  void visit(Instruction *I) { return visit(*I); }

  void visitPHINode(PHINode &I) {
    enqueueUsers(I);
    handleAlias(I);
  }

  void visitSelectInst(SelectInst &I) {
    enqueueUsers(I);
    handleAlias(I);
  }

  void visitStoreInst(StoreInst &SI) {
    // Regardless whether the alias of the alloca is the value operand or the
    // pointer operand, we need to assume the alloca is been written.
    handleMayWrite(SI);

    if (SI.getValueOperand() != U->get())
      return;

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
            handleAlias(*LI);
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
  }

  // All mem intrinsics modify the data.
  void visitMemIntrinsic(MemIntrinsic &MI) { handleMayWrite(MI); }

  void visitBitCastInst(BitCastInst &BC) {
    Base::visitBitCastInst(BC);
    handleAlias(BC);
  }

  void visitAddrSpaceCastInst(AddrSpaceCastInst &ASC) {
    Base::visitAddrSpaceCastInst(ASC);
    handleAlias(ASC);
  }

  void visitGetElementPtrInst(GetElementPtrInst &GEPI) {
    // The base visitor will adjust Offset accordingly.
    Base::visitGetElementPtrInst(GEPI);
    handleAlias(GEPI);
  }

  void visitIntrinsicInst(IntrinsicInst &II) {
    // When we found the lifetime markers refers to a
    // subrange of the original alloca, ignore the lifetime
    // markers to avoid misleading the analysis.
    if (!IsOffsetKnown || !Offset.isZero())
      return Base::visitIntrinsicInst(II);
    switch (II.getIntrinsicID()) {
    default:
      return Base::visitIntrinsicInst(II);
    case Intrinsic::lifetime_start:
      LifetimeStarts.insert(&II);
      LifetimeStartBBs.push_back(II.getParent());
      break;
    case Intrinsic::lifetime_end:
      LifetimeEndBBs.insert(II.getParent());
      break;
    }
  }

  void visitCallBase(CallBase &CB) {
    for (unsigned Op = 0, OpCount = CB.arg_size(); Op < OpCount; ++Op)
      if (U->get() == CB.getArgOperand(Op) && !CB.doesNotCapture(Op))
        PI.setEscaped(&CB);
    handleMayWrite(CB);
  }

  bool getShouldLiveOnFrame() const {
    if (!ShouldLiveOnFrame)
      ShouldLiveOnFrame = computeShouldLiveOnFrame();
    return *ShouldLiveOnFrame;
  }

  bool getMayWriteBeforeCoroBegin() const { return MayWriteBeforeCoroBegin; }

  DenseMap<Instruction *, std::optional<APInt>> getAliasesCopy() const {
    assert(getShouldLiveOnFrame() && "This method should only be called if the "
                                     "alloca needs to live on the frame.");
    for (const auto &P : AliasOffetMap)
      if (!P.second)
        report_fatal_error("Unable to handle an alias with unknown offset "
                           "created before CoroBegin.");
    return AliasOffetMap;
  }

private:
  const DominatorTree &DT;
  const coro::Shape &CoroShape;
  const SuspendCrossingInfo &Checker;
  // All alias to the original AllocaInst, created before CoroBegin and used
  // after CoroBegin. Each entry contains the instruction and the offset in the
  // original Alloca. They need to be recreated after CoroBegin off the frame.
  DenseMap<Instruction *, std::optional<APInt>> AliasOffetMap{};
  SmallPtrSet<Instruction *, 4> Users{};
  SmallPtrSet<IntrinsicInst *, 2> LifetimeStarts{};
  SmallVector<BasicBlock *> LifetimeStartBBs{};
  SmallPtrSet<BasicBlock *, 2> LifetimeEndBBs{};
  SmallPtrSet<const BasicBlock *, 2> CoroSuspendBBs{};
  bool MayWriteBeforeCoroBegin{false};
  bool ShouldUseLifetimeStartInfo{true};

  mutable std::optional<bool> ShouldLiveOnFrame{};

  bool computeShouldLiveOnFrame() const {
    // If lifetime information is available, we check it first since it's
    // more precise. We look at every pair of lifetime.start intrinsic and
    // every basic block that uses the pointer to see if they cross suspension
    // points. The uses cover both direct uses as well as indirect uses.
    if (ShouldUseLifetimeStartInfo && !LifetimeStarts.empty()) {
      // If there is no explicit lifetime.end, then assume the address can
      // cross suspension points.
      if (LifetimeEndBBs.empty())
        return true;

      // If there is a path from a lifetime.start to a suspend without a
      // corresponding lifetime.end, then the alloca's lifetime persists
      // beyond that suspension point and the alloca must go on the frame.
      llvm::SmallVector<BasicBlock *> Worklist(LifetimeStartBBs);
      if (isManyPotentiallyReachableFromMany(Worklist, CoroSuspendBBs,
                                             &LifetimeEndBBs, &DT))
        return true;

      // Addresses are guaranteed to be identical after every lifetime.start so
      // we cannot use the local stack if the address escaped and there is a
      // suspend point between lifetime markers. This should also cover the
      // case of a single lifetime.start intrinsic in a loop with suspend point.
      if (PI.isEscaped()) {
        for (auto *A : LifetimeStarts) {
          for (auto *B : LifetimeStarts) {
            if (Checker.hasPathOrLoopCrossingSuspendPoint(A->getParent(),
                                                          B->getParent()))
              return true;
          }
        }
      }
      return false;
    }
    // FIXME: Ideally the isEscaped check should come at the beginning.
    // However there are a few loose ends that need to be fixed first before
    // we can do that. We need to make sure we are not over-conservative, so
    // that the data accessed in-between await_suspend and symmetric transfer
    // is always put on the stack, and also data accessed after coro.end is
    // always put on the stack (esp the return object). To fix that, we need
    // to:
    //  1) Potentially treat sret as nocapture in calls
    //  2) Special handle the return object and put it on the stack
    //  3) Utilize lifetime.end intrinsic
    if (PI.isEscaped())
      return true;

    for (auto *U1 : Users)
      for (auto *U2 : Users)
        if (Checker.isDefinitionAcrossSuspend(*U1, U2))
          return true;

    return false;
  }

  void handleMayWrite(const Instruction &I) {
    if (!DT.dominates(CoroShape.CoroBegin, &I))
      MayWriteBeforeCoroBegin = true;
  }

  bool usedAfterCoroBegin(Instruction &I) {
    for (auto &U : I.uses())
      if (DT.dominates(CoroShape.CoroBegin, U))
        return true;
    return false;
  }

  void handleAlias(Instruction &I) {
    // We track all aliases created prior to CoroBegin but used after.
    // These aliases may need to be recreated after CoroBegin if the alloca
    // need to live on the frame.
    if (DT.dominates(CoroShape.CoroBegin, &I) || !usedAfterCoroBegin(I))
      return;

    if (!IsOffsetKnown) {
      AliasOffetMap[&I].reset();
    } else {
      auto [Itr, Inserted] = AliasOffetMap.try_emplace(&I, Offset);
      if (!Inserted && Itr->second && *Itr->second != Offset) {
        // If we have seen two different possible values for this alias, we set
        // it to empty.
        Itr->second.reset();
      }
    }
  }
};
} // namespace

static void collectFrameAlloca(AllocaInst *AI, const coro::Shape &Shape,
                               const SuspendCrossingInfo &Checker,
                               SmallVectorImpl<AllocaInfo> &Allocas,
                               const DominatorTree &DT) {
  if (Shape.CoroSuspends.empty())
    return;

  // The PromiseAlloca will be specially handled since it needs to be in a
  // fixed position in the frame.
  if (AI == Shape.SwitchLowering.PromiseAlloca)
    return;

  // The __coro_gro alloca should outlive the promise, make sure we
  // keep it outside the frame.
  if (AI->hasMetadata(LLVMContext::MD_coro_outside_frame))
    return;

  // The code that uses lifetime.start intrinsic does not work for functions
  // with loops without exit. Disable it on ABIs we know to generate such
  // code.
  bool ShouldUseLifetimeStartInfo =
      (Shape.ABI != coro::ABI::Async && Shape.ABI != coro::ABI::Retcon &&
       Shape.ABI != coro::ABI::RetconOnce);
  AllocaUseVisitor Visitor{AI->getDataLayout(), DT, Shape, Checker,
                           ShouldUseLifetimeStartInfo};
  Visitor.visitPtr(*AI);
  if (!Visitor.getShouldLiveOnFrame())
    return;
  Allocas.emplace_back(AI, Visitor.getAliasesCopy(),
                       Visitor.getMayWriteBeforeCoroBegin());
}

} // namespace

void collectSpillsFromArgs(SpillInfo &Spills, Function &F,
                           const SuspendCrossingInfo &Checker) {
  // Collect the spills for arguments and other not-materializable values.
  for (Argument &A : F.args())
    for (User *U : A.users())
      if (Checker.isDefinitionAcrossSuspend(A, U))
        Spills[&A].push_back(cast<Instruction>(U));
}

void collectSpillsAndAllocasFromInsts(
    SpillInfo &Spills, SmallVector<AllocaInfo, 8> &Allocas,
    SmallVector<Instruction *, 4> &DeadInstructions,
    SmallVector<CoroAllocaAllocInst *, 4> &LocalAllocas, Function &F,
    const SuspendCrossingInfo &Checker, const DominatorTree &DT,
    const coro::Shape &Shape) {

  for (Instruction &I : instructions(F)) {
    // Values returned from coroutine structure intrinsics should not be part
    // of the Coroutine Frame.
    if (isNonSpilledIntrinsic(I) || &I == Shape.CoroBegin)
      continue;

    // Handle alloca.alloc specially here.
    if (auto AI = dyn_cast<CoroAllocaAllocInst>(&I)) {
      // Check whether the alloca's lifetime is bounded by suspend points.
      if (isLocalAlloca(AI)) {
        LocalAllocas.push_back(AI);
        continue;
      }

      // If not, do a quick rewrite of the alloca and then add spills of
      // the rewritten value. The rewrite doesn't invalidate anything in
      // Spills because the other alloca intrinsics have no other operands
      // besides AI, and it doesn't invalidate the iteration because we delay
      // erasing AI.
      auto Alloc = lowerNonLocalAlloca(AI, Shape, DeadInstructions);

      for (User *U : Alloc->users()) {
        if (Checker.isDefinitionAcrossSuspend(*Alloc, U))
          Spills[Alloc].push_back(cast<Instruction>(U));
      }
      continue;
    }

    // Ignore alloca.get; we process this as part of coro.alloca.alloc.
    if (isa<CoroAllocaGetInst>(I))
      continue;

    if (auto *AI = dyn_cast<AllocaInst>(&I)) {
      collectFrameAlloca(AI, Shape, Checker, Allocas, DT);
      continue;
    }

    for (User *U : I.users())
      if (Checker.isDefinitionAcrossSuspend(I, U)) {
        // We cannot spill a token.
        if (I.getType()->isTokenTy())
          report_fatal_error(
              "token definition is separated from the use by a suspend point");
        Spills[&I].push_back(cast<Instruction>(U));
      }
  }
}

void collectSpillsFromDbgInfo(SpillInfo &Spills, Function &F,
                              const SuspendCrossingInfo &Checker) {
  // We don't want the layout of coroutine frame to be affected
  // by debug information. So we only choose to salvage dbg.values for
  // whose value is already in the frame.
  // We would handle the dbg.values for allocas specially
  for (auto &Iter : Spills) {
    auto *V = Iter.first;
    SmallVector<DbgVariableRecord *, 16> DVRs;
    findDbgValues(V, DVRs);
    // Add the instructions which carry debug info that is in the frame.
    for (DbgVariableRecord *DVR : DVRs)
      if (Checker.isDefinitionAcrossSuspend(*V, DVR->Marker->MarkedInstr))
        Spills[V].push_back(DVR->Marker->MarkedInstr);
  }
}

/// Async and Retcon{Once} conventions assume that all spill uses can be sunk
/// after the coro.begin intrinsic.
void sinkSpillUsesAfterCoroBegin(const DominatorTree &Dom,
                                 CoroBeginInst *CoroBegin,
                                 coro::SpillInfo &Spills,
                                 SmallVectorImpl<coro::AllocaInfo> &Allocas) {
  SmallSetVector<Instruction *, 32> ToMove;
  SmallVector<Instruction *, 32> Worklist;

  // Collect all users that precede coro.begin.
  auto collectUsers = [&](Value *Def) {
    for (User *U : Def->users()) {
      auto Inst = cast<Instruction>(U);
      if (Inst->getParent() != CoroBegin->getParent() ||
          Dom.dominates(CoroBegin, Inst))
        continue;
      if (ToMove.insert(Inst))
        Worklist.push_back(Inst);
    }
  };
  for (auto &I : Spills)
    collectUsers(I.first);
  for (auto &I : Allocas)
    collectUsers(I.Alloca);

  // Recursively collect users before coro.begin.
  while (!Worklist.empty()) {
    auto *Def = Worklist.pop_back_val();
    for (User *U : Def->users()) {
      auto Inst = cast<Instruction>(U);
      if (Dom.dominates(CoroBegin, Inst))
        continue;
      if (ToMove.insert(Inst))
        Worklist.push_back(Inst);
    }
  }

  // Sort by dominance.
  SmallVector<Instruction *, 64> InsertionList(ToMove.begin(), ToMove.end());
  llvm::sort(InsertionList, [&Dom](Instruction *A, Instruction *B) -> bool {
    // If a dominates b it should precede (<) b.
    return Dom.dominates(A, B);
  });

  Instruction *InsertPt = CoroBegin->getNextNode();
  for (Instruction *Inst : InsertionList)
    Inst->moveBefore(InsertPt->getIterator());
}

BasicBlock::iterator getSpillInsertionPt(const coro::Shape &Shape, Value *Def,
                                         const DominatorTree &DT) {
  BasicBlock::iterator InsertPt;
  if (auto *Arg = dyn_cast<Argument>(Def)) {
    // For arguments, we will place the store instruction right after
    // the coroutine frame pointer instruction, i.e. coro.begin.
    InsertPt = Shape.getInsertPtAfterFramePtr();

    // If we're spilling an Argument, make sure we clear 'captures'
    // from the coroutine function.
    Arg->getParent()->removeParamAttr(Arg->getArgNo(), Attribute::Captures);
  } else if (auto *CSI = dyn_cast<AnyCoroSuspendInst>(Def)) {
    // Don't spill immediately after a suspend; splitting assumes
    // that the suspend will be followed by a branch.
    InsertPt = CSI->getParent()->getSingleSuccessor()->getFirstNonPHIIt();
  } else {
    auto *I = cast<Instruction>(Def);
    if (!DT.dominates(Shape.CoroBegin, I)) {
      // If it is not dominated by CoroBegin, then spill should be
      // inserted immediately after CoroFrame is computed.
      InsertPt = Shape.getInsertPtAfterFramePtr();
    } else if (auto *II = dyn_cast<InvokeInst>(I)) {
      // If we are spilling the result of the invoke instruction, split
      // the normal edge and insert the spill in the new block.
      auto *NewBB = SplitEdge(II->getParent(), II->getNormalDest());
      InsertPt = NewBB->getTerminator()->getIterator();
    } else if (isa<PHINode>(I)) {
      // Skip the PHINodes and EH pads instructions.
      BasicBlock *DefBlock = I->getParent();
      if (auto *CSI = dyn_cast<CatchSwitchInst>(DefBlock->getTerminator()))
        InsertPt = splitBeforeCatchSwitch(CSI)->getIterator();
      else
        InsertPt = DefBlock->getFirstInsertionPt();
    } else {
      assert(!I->isTerminator() && "unexpected terminator");
      // For all other values, the spill is placed immediately after
      // the definition.
      InsertPt = I->getNextNode()->getIterator();
    }
  }

  return InsertPt;
}

} // End namespace coro.

} // End namespace llvm.

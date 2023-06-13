//===- InferCallsiteAttrs.cpp - Propagate attributes to callsites ---------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements the InferCallsiteAttrs class.
//
//===----------------------------------------------------------------------===//


#include "llvm/Transforms/Utils/InferCallsiteAttrs.h"

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Analysis/ValueTracking.h"
#include "llvm/IR/BasicBlock.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/Instruction.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/Type.h"
#include "llvm/IR/User.h"
#include "llvm/IR/Value.h"

using namespace llvm;

#define DEBUG_TYPE "infer-callsite-attrs"

// Helper to add parameter attributes to callsite CB.
static bool addCallsiteParamAttributes(CallBase *CB,
                                       const SmallVector<unsigned> &ArgNos,
                                       Attribute::AttrKind Attr) {
  if (ArgNos.empty())
    return false;

  AttributeList AS = CB->getAttributes();
  LLVMContext &Ctx = CB->getContext();
  AS = AS.addParamAttribute(Ctx, ArgNos, Attribute::get(Ctx, Attr));

  CB->setAttributes(AS);
  return true;
}

// Helper to determine if a callsite is malloc like and doesn't provably
// escape.
static bool isCallUnkMallocLike(const Value *V) {
  auto *MCB = dyn_cast<CallBase>(V);
  if (MCB == nullptr)
    return false;

  return MCB->returnDoesNotAlias();
}

static bool isCallLocalMallocLike(const Value *V, const CallBase *CB) {
  if (!isCallUnkMallocLike(V))
    return false;
  auto *RI = dyn_cast<ReturnInst>(CB->getParent()->getTerminator());
  return RI == nullptr || RI->getReturnValue() != V;
}

// Check all instructions between callbase and end of basicblock (if that
// basic block ends in a return). This will cache the analysis information.
// Will break early if condition is met based on arguments.
bool InferCallsiteAttrs::checkBetweenCallsiteAndReturn(
    const CallBase *CB, bool BailOnStore, bool BailOnLoad,
    bool BailOnNonDirectTransfer, bool BailOnNotReturned) {
  const BasicBlock *BB = CB->getParent();
  auto *RI = dyn_cast<ReturnInst>(BB->getTerminator());

  if (RI == nullptr)
    return false;

  if (RI->getReturnValue() == CB)
    CurCBInfo.CallerReturnBasedOnCallsite = kYes;
  else {
    CurCBInfo.CallerReturnBasedOnCallsite = kNo;
    if (BailOnNotReturned)
      return false;
  }

  if (RI == CB->getNextNode()) {
    CurCBInfo.IsLastInsBeforeReturn = kYes;
    CurCBInfo.StoresBetweenReturn = kNo;
    CurCBInfo.LoadsBetweenReturn = kNo;
    CurCBInfo.NonDirectTransferBetweenReturn = kNo;
    return true;
  }
  CurCBInfo.IsLastInsBeforeReturn = kNo;

  if (BailOnStore && CurCBInfo.StoresBetweenReturn == kYes)
    return false;
  if (BailOnLoad && CurCBInfo.LoadsBetweenReturn == kYes)
    return false;
  if (BailOnNonDirectTransfer &&
      CurCBInfo.NonDirectTransferBetweenReturn == kYes)
    return false;

  if (CurCBInfo.StoresBetweenReturn != kMaybe &&
      CurCBInfo.LoadsBetweenReturn != kMaybe &&
      CurCBInfo.NonDirectTransferBetweenReturn != kMaybe)
    return true;

  unsigned Cnt = 0;
  for (const Instruction *Ins = CB->getNextNode(); Ins && Ins != RI;
       Ins = Ins->getNextNode()) {

    if (Cnt++ >= kMaxChecks)
      return false;

    if (Ins->mayWriteToMemory()) {
      CurCBInfo.StoresBetweenReturn = kYes;
      if (BailOnStore)
        return false;
    }

    if (Ins->mayReadFromMemory()) {
      CurCBInfo.LoadsBetweenReturn = kYes;
      if (BailOnLoad)
        return false;
    }

    if (!isGuaranteedToTransferExecutionToSuccessor(Ins)) {
      CurCBInfo.NonDirectTransferBetweenReturn = kYes;
      if (BailOnNonDirectTransfer)
        return false;
    }
  }

  if (CurCBInfo.StoresBetweenReturn == kMaybe)
    CurCBInfo.StoresBetweenReturn = kNo;
  if (CurCBInfo.LoadsBetweenReturn == kMaybe)
    CurCBInfo.LoadsBetweenReturn = kNo;
  if (CurCBInfo.NonDirectTransferBetweenReturn == kMaybe)
    CurCBInfo.NonDirectTransferBetweenReturn = kNo;

  return true;
}

// Check all instruction instructions preceding basic blocked (any instruction
// that may reach the callsite CB). If conditions are met, can set early
// return using BailOn* arguments.
bool InferCallsiteAttrs::checkPrecedingBBIns(const CallBase *CB,
                                             bool BailOnAlloca,
                                             bool BailOnLocalMalloc) {

  if (BailOnAlloca && CurCBInfo.PrecedingAlloca == kYes)
    return false;
  if (BailOnLocalMalloc && CurCBInfo.PrecedingLocalMalloc == kYes)
    return false;

  if (CurCBInfo.PrecedingAlloca != kMaybe &&
      CurCBInfo.PrecedingLocalMalloc != kMaybe)
    return true;

  SmallPtrSet<const BasicBlock *, 16> AllPreds;
  SmallVector<const BasicBlock *> Preds;
  unsigned Cnt = 0;
  AllPreds.insert(CB->getParent());
  Preds.push_back(CB->getParent());

  auto WorklistCont = [&](const BasicBlock *CurBB) {
    for (const BasicBlock *Pred : predecessors(CurBB))
      if (AllPreds.insert(Pred).second)
        Preds.push_back(Pred);
  };

  while (!Preds.empty()) {
    const BasicBlock *CurBB = Preds.pop_back_val();
    BasicBlockInfos &BBInfo = BBInfos[CurBB];

    if (BBInfo.Alloca == kNo && BBInfo.UnkMalloc == kNo) {
      WorklistCont(CurBB);
      continue;
    }

    auto ProcessHasAlloca = [this, BBInfo, BailOnAlloca]() {
      if (BBInfo.Alloca == kYes) {
        CurCBInfo.PrecedingAlloca = kYes;
        if (BailOnAlloca)
          return false;
      }
      return true;
    };

    auto ProcessHasUnkMalloc = [this, BBInfo, BailOnLocalMalloc,
                                CB](const BasicBlock *BB,
                                    const Value *V = nullptr) {
      // We check beyond just if there is a malloc. We only set local malloc if
      // that malloc is not guranteed to be made visible outside of the caller.
      // We don't exhaustively check (see if the malloc was stored to a ptr or
      // global variable), just check if its returned from our callsites basic
      // block.
      // TODO: We could also check if the malloc escapes in other ways than
      // return (like stored to a pointer or global), especially if we are
      // already iterating through all the instructions.
      if (BBInfo.UnkMalloc == kYes) {
        if (V == nullptr) {
          // We only know there is a malloc instruction, not where so iterate
          // and find.
          for (const Value &Val : *BB) {
            if (isCallLocalMallocLike(&Val, CB)) {
              CurCBInfo.PrecedingLocalMalloc = kYes;
              break;
            }
          }
        } else if (isCallLocalMallocLike(V, CB)) {
          CurCBInfo.PrecedingLocalMalloc = kYes;
        }

        if (BailOnLocalMalloc && CurCBInfo.PrecedingLocalMalloc == kYes)
          return false;
      }
      return true;
    };

    if (!ProcessHasAlloca() || !ProcessHasUnkMalloc(CurBB))
      return false;

    bool EarlyOut = false;
    // Check all instructions in current BB for an alloca/leaked malloc.
    for (const Value &V : *CurBB) {
      if (&V == CB) {
        EarlyOut = CurCBInfo.IsLastInsBeforeReturn != kYes;
        break;
      }

      // If we reach max checks and can't rule out alloca/leaked malloc case
      // fail.
      if (Cnt++ >= kMaxChecks)
        return false;

      if (isa<AllocaInst>(&V))
        BBInfo.Alloca = kYes;

      if (isCallUnkMallocLike(&V))
        BBInfo.UnkMalloc = kYes;

      if (!ProcessHasAlloca() || !ProcessHasUnkMalloc(CurBB, &V))
        return false;
    }

    if (!EarlyOut) {
      if (BBInfo.Alloca == kMaybe)
        BBInfo.Alloca = kNo;

      if (BBInfo.UnkMalloc == kMaybe)
        BBInfo.UnkMalloc = kNo;
    }

    WorklistCont(CurBB);
  }

  if (CurCBInfo.PrecedingAlloca == kMaybe)
    CurCBInfo.PrecedingAlloca = kNo;
  if (CurCBInfo.PrecedingLocalMalloc == kMaybe)
    CurCBInfo.PrecedingLocalMalloc = kNo;
  return true;
}

// Check all basic blocks for conditions. At the moment only condition is if
// landing/EH pad so will store result and break immediately if one is found.
// In the future may be extended to check other conditions.
bool InferCallsiteAttrs::checkAllBBs(bool BailOnPad) {
  if (BailOnPad && CurFnInfo.LandingOrEHPad == kYes)
    return false;

  if (CurFnInfo.LandingOrEHPad != kMaybe)
    return false;

  for (const BasicBlock &CurBB : *Caller) {
    if (CurBB.isEHPad() || CurBB.isLandingPad()) {
      CurFnInfo.LandingOrEHPad = kYes;
      if (BailOnPad)
        return false;
      // Nothing else to set/check
      break;
    }
  }

  if (CurFnInfo.LandingOrEHPad == kMaybe)
    CurFnInfo.LandingOrEHPad = kNo;
  return true;
}
// Try to propagate nocapture attribute from caller argument to callsite
// arguments.
bool InferCallsiteAttrs::tryPropagateNoCapture(CallBase *CB) {

  if (!isa<CallInst>(CB))
    return false;

  SmallVector<unsigned> NoCaptureArgs;

  // If this callsite is to a readonly function that doesn't throw then the
  // only way to the pointer to be captured is through the return value. If
  // the return type is void or the return value of this callsite is unused,
  // then all the pointer parameters at this callsite must be nocapture. NB:
  // This is a slight strengthening of the case done in the FunctionAttrs pass
  // which has the same logic but only for void function. At specific
  // callsites we can do non-void function if the return value is unused.
  bool IsAlwaysNoCapture = CB->onlyReadsMemory() && CB->doesNotThrow() &&
                           (CB->getType()->isVoidTy() || CB->use_empty());
  if (IsAlwaysNoCapture) {
    unsigned ArgN = 0;
    for (Value *V : CB->args()) {
      if (V->getType()->isPointerTy() &&
          !CB->paramHasAttr(ArgN, Attribute::NoCapture))
        NoCaptureArgs.push_back(ArgN);
      ++ArgN;
    }

    return addCallsiteParamAttributes(CB, NoCaptureArgs, Attribute::NoCapture);
  }

  // If this is not trivially nocapture, then we propagate a nocapture
  // argument if the callsite meets the following requirements:
  //
  //    1) The callsite is in a basic block that ends with a return
  //       statement.
  //       the callsite and return statement.
  //    2) Between the callsite the end of its basic block there are no
  //       may-write instructions.
  //    3) The return value of the callsite is not used (directly or
  //       indirectly) as the address of a may-read instruction.
  //    4) There are allocas or leaked (not freed or returned) mallocs
  //       reachable from the callsite.
  //    5) The callsite/caller are nothrow OR there is no landing padd in the
  //       caller.
  //
  // These requirements are intentionally over conservative. We are only
  // trying to catch relatively trivial cases.
  //
  // Requirements 1 & 2 are there to ensure that after the callsite has
  // returned, the state of any captured in memory pointers cannot change.
  // This implies that if the caller has any nocapture in memory gurantees,
  // that state has been reached by the end of the callsite.
  //
  // Requirements 3 & 4 are to cover cases where pointers could escape the
  // callsite (but not the caller) through non-dead code. Any return value
  // thats loaded from (or used to create a pointer that is loaded from) could
  // have derived from an argument. Finally, allocas/leaked mallocs in general
  // are difficult (so we avoid them entirely). Callsites can arbitrarily
  // store pointers in allocas for use later without violating a nocapture
  // gurantee by the caller, as the allocas are torn down at caller return.
  // Likewise a leaked malloc would not be accessible outside of the caller,
  // but could still be accessible after the callsite. There are a variety of
  // complex cases involving allocas/leaked mallocs. For simplicity, if we see
  // either we simply fail.
  //
  // Requirement 5 is to cover the last way to escape to occur. If the
  // callsite/caller is nothrow its a non-issue. If the callsite may throw,
  // then a method of capture is through an exception. If the caller has no
  // landing padd to catch this exception, then the exception state will be
  // visible outside of the caller so any gurantees about nocapture made by
  // the caller will apply to the callsites throw. If the caller has a landing
  // padd, its possible for the callsite to capture a pointer in a throw that
  // is later cleared by the caller.

  // Check easy O(1) stuff that can quickly rule out this callsite.
  const BasicBlock *BB = CB->getParent();
  // Make sure this BB ends in a return (Requirement 1).
  auto *RI = dyn_cast<ReturnInst>(BB->getTerminator());
  if (RI == nullptr)
    return false;

  // Req 2 fails.
  if (CurCBInfo.StoresBetweenReturn == kYes)
    return false;

  // Req 4 fails.
  if (CurCBInfo.PrecedingAlloca == kYes ||
      CurCBInfo.PrecedingLocalMalloc == kYes)
    return false;

  bool MayThrow = !(CB->doesNotThrow() || checkCallerDoesNotThrow());
  // Req 5 fails.
  if (MayThrow && CurFnInfo.LandingOrEHPad == kYes)
    return false;

  SmallPtrSet<Value *, 8> NoCaptureParentArguments;

  // See if caller has any nocapture arguments we may be able to propagate
  // attributes from.
  for (unsigned I = 0; I < Caller->arg_size(); ++I)
    if (checkCallerHasParamAttr(I, Attribute::NoCapture))
      NoCaptureParentArguments.insert(Caller->getArg(I));

  unsigned ArgN = 0;
  for (Value *V : CB->args()) {
    // See if this callsite argument is missing nocapture and its propagatable
    // (nocapture in the caller).
    if (!CB->paramHasAttr(ArgN, Attribute::NoCapture) &&
        NoCaptureParentArguments.contains(V))
      NoCaptureArgs.push_back(ArgN);
    ++ArgN;
  }

  // No point to do more expensive analysis if we won't be able to do anything
  // with it.
  if (NoCaptureArgs.empty())
    return false;

  // Check that between callsite and return we can't changed capture state
  // (BailOnStore). NB: we can accept non-directtransfer instructions as long as
  // the caller does not having any landing pads. If the caller has no landing
  // pad, then any exception/interrupt/etc... will leave the callers scope and
  // thus any caller nocapture gurantee will apply.
  if (!checkBetweenCallsiteAndReturn(CB, /*BailOnStore*/ true,
                                     /*BailOnLoad*/ false,
                                     /*BailOnNonDirectTransfer*/ false,
                                     /*BailOnNotReturned*/ false))
    return false;

  // We need to check if the load is from the return of this callsite. If so
  // then a pointer may have been return captured (Req 3).
  if (CurCBInfo.LoadsBetweenReturn == kYes) {

    // If the callsite return is used as the callers return, then the caller's
    // no-capture gurantee includes the callsites return so we don't need to
    // check the actual loads (NB: We fail on NonDirectTransfer between callsite
    // and return, so the callsite's return MUST reach the return
    // instruction).
    if (CurCBInfo.CallerReturnBasedOnCallsite != kYes) {

      // Figure out of the load is derived from the return of the callsite. If
      // so we assume its a captured pointer.
      SmallPtrSet<const Value *, 16> DerivedFromReturn;
      for (const Value *U : CB->uses())
        DerivedFromReturn.insert(U);

      unsigned Cnt = 0;
      for (const Instruction *Ins = CB; Ins && Ins != RI;
           Ins = Ins->getNextNode()) {
        if (Cnt++ >= kMaxChecks)
          return false;

        for (const Value *U : Ins->operands()) {
          if (DerivedFromReturn.contains(U)) {
            DerivedFromReturn.insert(Ins);
            break;
          }
        }

        if (Ins->mayReadFromMemory()) {

          // TODO: We could do a bit more analysis and check if Ins is used to
          // derived the Caller return value at all, rather than just checking
          // if they are equal.
          if ((!isa<LoadInst>(Ins) ||
               cast<LoadInst>(Ins)->getPointerOperand() != RI) &&
              DerivedFromReturn.contains(Ins))
            return false;
        }
      }
    }
  }

  // Check all predecessors (basic blocks from which an alloca or leaked
  // malloc may be able to reach this callsite). We are being incredibly//
  // conservative here. We could likely skip the alloca/leaked malloc search
  // in a few cases. 1) If the callsite is the last instruction before the
  // return or if there are no may-read instructions between the callsite and
  // the return.  2) If there are possible stores to the alloca/leaked malloc
  // that may reach the callsite its probably also safe. And/Or 3) If the
  // callsite is readonly it could never capture in memory so these
  // are non factor concerns. For now stay conservative, but over
  // time these optimizations can be added.
  if (!checkPrecedingBBIns(CB, /*BailOnAlloca*/ true,
                           /*BailOnLocalMalloc*/ true))
    return false;

  if (MayThrow && !checkAllBBs(/*BailOnPad*/ true))
    return false;

  return addCallsiteParamAttributes(CB, NoCaptureArgs, Attribute::NoCapture);
}

// Try trivial propagations (one where if true for the caller, are
// automatically true for the callsite without further analysis).
bool InferCallsiteAttrs::tryTrivialPropagations(CallBase *CB) {
  bool Changed = false;
  const std::array CallerFnAttrPropagations = {
      Attribute::MustProgress, Attribute::WillReturn, Attribute::NoSync};
  for (const Attribute::AttrKind Attr : CallerFnAttrPropagations) {
    if (checkCallerHasFnAttr(Attr) && !CB->hasFnAttr(Attr)) {
      Changed = true;
      CB->addFnAttr(Attr);
    }
  }

  const std::array CallerParamAttrPropagations = {
      Attribute::NoUndef,  Attribute::NonNull,  Attribute::NoFree,
      Attribute::ReadNone, Attribute::ReadOnly, Attribute::WriteOnly};

  for (const Attribute::AttrKind Attr : CallerParamAttrPropagations) {
    SmallPtrSet<Value *, 8> CallerArgs;
    SmallVector<unsigned> ArgNosAttr;
    for (unsigned I = 0; I < Caller->arg_size(); ++I)
      if (checkCallerHasParamAttr(I, Attr))
        CallerArgs.insert(Caller->getArg(I));

    unsigned ArgN = 0;
    // TODO: For the readnone, readonly, and writeonly attributes, we may be
    // able to inherit from the callsite params underlying object if that
    // underlying object is an argument.
    for (Value *V : CB->args()) {
      if (!CB->paramHasAttr(ArgN, Attr) && CallerArgs.contains(V))
        ArgNosAttr.push_back(ArgN);
      ArgN++;
    }

    Changed |= addCallsiteParamAttributes(CB, ArgNosAttr, Attr);
  }

  return Changed;
}

// Try propagations of return attributes (nonnull, noundef, etc...)
bool InferCallsiteAttrs::tryReturnPropagations(CallBase *CB) {
  std::optional<bool> CallsiteReturnMustBeCallerReturnCached;
  auto CallsiteReturnMustBeCallerReturn = [&]() {
    if (CallsiteReturnMustBeCallerReturnCached)
      return *CallsiteReturnMustBeCallerReturnCached;
    // We can only propagate return attribute if we are certain this
    // callsites return is used as the caller return (in it's basic
    // block).
    CallsiteReturnMustBeCallerReturnCached = checkBetweenCallsiteAndReturn(
        CB, /*BailOnStore*/ false, /*BailOnLoad*/ false,
        /*BailOnNonDirectTransfer*/ true, /*BailOnNotReturned*/ true);
    return *CallsiteReturnMustBeCallerReturnCached;
  };

  bool Changed = false;
  const std::array CallerReturnAttrPropagations = {Attribute::NoUndef,
                                                   Attribute::NonNull};
  for (const Attribute::AttrKind Attr : CallerReturnAttrPropagations) {
    if (checkCallerHasReturnAttr(Attr) && !CB->hasRetAttr(Attr)) {
      // Wait until we know we actually need it to do potentially expensive
      // analysis.
      if (!CallsiteReturnMustBeCallerReturn())
        return Changed;
      CB->addRetAttr(Attr);
      Changed = true;
    }
  }
  return Changed;
}

// Try propagations of memory access attribute (readnone, readonly, etc...).
bool InferCallsiteAttrs::tryMemoryPropagations(CallBase *CB) {
  std::optional<bool> MayHaveLocalMemoryArgsCached;
  std::optional<bool> MayHavePrecedingLocalMemoryCached;

  auto MayHavePrecedingLocalMemory = [&]() {
    if (MayHavePrecedingLocalMemoryCached)
      return *MayHavePrecedingLocalMemoryCached;
    MayHavePrecedingLocalMemoryCached =
        checkPrecedingBBIns(CB, /*BailOnAlloca*/ true,
                            /*BailOnLocalMalloc*/ true);
    return *MayHavePrecedingLocalMemoryCached;
  };

  auto MayHaveLocalMemoryArgs = [&]() {
    if (MayHaveLocalMemoryArgsCached)
      return *MayHaveLocalMemoryArgsCached;

    // If there are local memory regions that can reach this callsite,
    // then check all arguments. If we can't trace them back to some
    // value that is also visible outside the caller fail.
    for (Value *V : CB->args()) {
      Value *UnderlyingObj = getUnderlyingObject(V);
      // TODO: We probably don't need to bail entirely here. We could still
      // set parameter attributes for the callsite for the arguments that do
      // meet these conditions.
      if (!isa<Argument>(UnderlyingObj) && !isa<GlobalValue>(UnderlyingObj)) {
        // Don't do potentially very expensive preceding BB check unless
        // cheaper getUnderlyingObject check fails to prove what we need.
        // TODO: Does local malloc inherit parent memory access constraints
        // or is it like alloca? If the former set BailOnLocalMalloc to
        // false.
        MayHaveLocalMemoryArgsCached = MayHavePrecedingLocalMemory();
        return *MayHaveLocalMemoryArgsCached;
      }
    }
    MayHaveLocalMemoryArgsCached = true;
    return true;
  };

  bool Changed = false;
  // If the callsite has no local memory visible to it, then it shares
  // constraints with the caller as any pointer is has access too is shared
  // with the caller. For readnone, readonly, and writeonly simple not alloca
  // args is enough to propagate. For the ArgMemory attributes, we need
  // absolutely no local memory as otherwise we could nest caller local memory
  // in argument pointers then use that derefed caller local memory in the
  // callsite violating the constraint.
  if (checkCallerDoesNotAccessMemory() && !CB->doesNotAccessMemory()) {
    // Wait until we know we actually need it to do potentially expensive
    // analysis.
    if (!MayHaveLocalMemoryArgs())
      return Changed;
    CB->setDoesNotAccessMemory();
    Changed = true;
  }
  if (checkCallerOnlyReadsMemory() && !CB->onlyReadsMemory()) {
    if (!MayHaveLocalMemoryArgs())
      return Changed;
    CB->setOnlyReadsMemory();
    Changed = true;
  }
  if (checkCallerOnlyWritesMemory() && !CB->onlyWritesMemory()) {
    if (!MayHaveLocalMemoryArgs())
      return Changed;
    CB->setOnlyWritesMemory();
    Changed = true;
  }

  if (checkCallerOnlyAccessesArgMemory() && !CB->onlyAccessesArgMemory()) {
    // Switch to heavier check here.
    // TODO: We may be able to do some analysis of if any of the allocas are
    // ever stored anywhere (if thats not the case, then just argument memory
    // is enough again).
    if (!MayHavePrecedingLocalMemory())
      return Changed;
    CB->setOnlyAccessesArgMemory();
    Changed = true;
  }
  if (checkCallerOnlyAccessesInaccessibleMemory() &&
      !CB->onlyAccessesInaccessibleMemory()) {
    if (!MayHavePrecedingLocalMemory())
      return Changed;
    CB->setOnlyAccessesInaccessibleMemory();
    Changed = true;
  }
  if (checkCallerOnlyAccessesInaccessibleMemOrArgMem() &&
      !CB->onlyAccessesInaccessibleMemOrArgMem()) {
    if (!MayHavePrecedingLocalMemory())
      return Changed;
    CB->setOnlyAccessesInaccessibleMemOrArgMem();
    Changed = true;
  }

  return Changed;
}

// Add attributes to callsite, assumes Caller and CxtCB are setup already
bool InferCallsiteAttrs::inferCallsiteAttributesImpl(CallBase *CB) {

  memset(&CurCBInfo, kMaybe, sizeof(CurCBInfo));
  bool Changed = tryPropagateNoCapture(CB);
  Changed |= tryTrivialPropagations(CB);
  Changed |= tryReturnPropagations(CB);
  Changed |= tryMemoryPropagations(CB);
  return Changed;
}

void InferCallsiteAttrs::resetCache() {
  BBInfos.clear();
  FunctionInfos.clear();
}

// Add attributes to callsites based on the function is called in (or by
// setting CxtCallsite the exact callsite of the callsite).
bool InferCallsiteAttrs::inferCallsiteAttributes(CallBase *CB,
                                                 const CallBase *CxtCallsite) {
  const BasicBlock *BB = CB->getParent();
  assert(BB && "BasicBlock should never be null in this context");
  const Function *PF = BB->getParent();
  assert(PF && "Function should never but null in this context");

  // Setup caching state

  CurFnInfo = FunctionInfos[PF];
  Caller = PF;
  CxtCB = CxtCallsite;

  bool Changed = inferCallsiteAttributesImpl(CB);

  return Changed;
}

// Process all callsites in Function ParentFunc. This is more efficient that
// calling inferCallsiteAttributes in a loop as it 1) avoids some unnecessary
// cache lookups and 2) does some analysis while searching for callsites.
bool InferCallsiteAttrs::processFunction(Function *ParentFunc,
                                         const CallBase *ParentCallsite) {
  bool Changed = false;
  if (PreserveCache)
    CurFnInfo = FunctionInfos[ParentFunc];
  else
    memset(&CurFnInfo, kMaybe, sizeof(CurFnInfo));
  Caller = ParentFunc;
  CxtCB = ParentCallsite;
  for (BasicBlock &BB : *ParentFunc) {
    if (BB.isEHPad() || BB.isLandingPad())
      CurFnInfo.LandingOrEHPad = kYes;
    BasicBlockInfos &BBInfo = BBInfos[&BB];
    for (Value &V : BB) {
      if (isa<AllocaInst>(&V))
        BBInfo.Alloca = kYes;

      if (isCallUnkMallocLike(&V))
        BBInfo.UnkMalloc = kYes;

      if (auto *CB = dyn_cast<CallBase>(&V))
        Changed |= inferCallsiteAttributesImpl(CB);
    }

    if (BBInfo.Alloca == kMaybe)
      BBInfo.Alloca = kNo;

    if (BBInfo.UnkMalloc == kMaybe)
      BBInfo.UnkMalloc = kNo;
  }
  if (CurFnInfo.LandingOrEHPad == kMaybe)
    CurFnInfo.LandingOrEHPad = kNo;

  if (PreserveCache)
    FunctionInfos[ParentFunc] = CurFnInfo;
  else
    resetCache();

  return Changed;
}

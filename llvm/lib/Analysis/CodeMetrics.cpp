//===- CodeMetrics.cpp - Code cost measurements ---------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements code cost measurement utilities.
//
//===----------------------------------------------------------------------===//

#include "llvm/Analysis/CodeMetrics.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/Analysis/AssumptionCache.h"
#include "llvm/Analysis/LoopInfo.h"
#include "llvm/Analysis/TargetTransformInfo.h"
#include "llvm/Analysis/ValueTracking.h"
#include "llvm/IR/Function.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/InstructionCost.h"

#define DEBUG_TYPE "code-metrics"

using namespace llvm;

static void
appendSpeculatableOperands(const Value *V,
                           SmallPtrSetImpl<const Value *> &Visited,
                           SmallVectorImpl<const Value *> &Worklist) {
  const User *U = dyn_cast<User>(V);
  if (!U)
    return;

  for (const Value *Operand : U->operands())
    if (Visited.insert(Operand).second)
      if (isSafeToSpeculativelyExecute(Operand))
        Worklist.push_back(Operand);
}

static void completeEphemeralValues(SmallPtrSetImpl<const Value *> &Visited,
                                    SmallVectorImpl<const Value *> &Worklist,
                                    SmallPtrSetImpl<const Value *> &EphValues) {
  // Note: We don't speculate PHIs here, so we'll miss instruction chains kept
  // alive only by ephemeral values.

  // Walk the worklist using an index but without caching the size so we can
  // append more entries as we process the worklist. This forms a queue without
  // quadratic behavior by just leaving processed nodes at the head of the
  // worklist forever.
  for (int i = 0; i < (int)Worklist.size(); ++i) {
    const Value *V = Worklist[i];

    assert(Visited.count(V) &&
           "Failed to add a worklist entry to our visited set!");

    // If all uses of this value are ephemeral, then so is this value.
    if (!all_of(V->users(), [&](const User *U) { return EphValues.count(U); }))
      continue;

    EphValues.insert(V);
    LLVM_DEBUG(dbgs() << "Ephemeral Value: " << *V << "\n");

    // Append any more operands to consider.
    appendSpeculatableOperands(V, Visited, Worklist);
  }
}

// Find all ephemeral values.
void CodeMetrics::collectEphemeralValues(
    const Loop *L, AssumptionCache *AC,
    SmallPtrSetImpl<const Value *> &EphValues) {
  SmallPtrSet<const Value *, 32> Visited;
  SmallVector<const Value *, 16> Worklist;

  for (auto &AssumeVH : AC->assumptions()) {
    if (!AssumeVH)
      continue;
    Instruction *I = cast<Instruction>(AssumeVH);

    // Filter out call sites outside of the loop so we don't do a function's
    // worth of work for each of its loops (and, in the common case, ephemeral
    // values in the loop are likely due to @llvm.assume calls in the loop).
    if (!L->contains(I->getParent()))
      continue;

    if (EphValues.insert(I).second)
      appendSpeculatableOperands(I, Visited, Worklist);
  }

  completeEphemeralValues(Visited, Worklist, EphValues);
}

void CodeMetrics::collectEphemeralValues(
    const Function *F, AssumptionCache *AC,
    SmallPtrSetImpl<const Value *> &EphValues) {
  SmallPtrSet<const Value *, 32> Visited;
  SmallVector<const Value *, 16> Worklist;

  for (auto &AssumeVH : AC->assumptions()) {
    if (!AssumeVH)
      continue;
    Instruction *I = cast<Instruction>(AssumeVH);
    assert(I->getParent()->getParent() == F &&
           "Found assumption for the wrong function!");

    if (EphValues.insert(I).second)
      appendSpeculatableOperands(I, Visited, Worklist);
  }

  completeEphemeralValues(Visited, Worklist, EphValues);
}

/// Fill in the current structure with information gleaned from the specified
/// block.
void CodeMetrics::analyzeBasicBlock(
    const BasicBlock *BB, const TargetTransformInfo &TTI,
    const SmallPtrSetImpl<const Value *> &EphValues, bool PrepareForLTO) {
  ++NumBlocks;
  // Use a proxy variable for NumInsts of type InstructionCost, so that it can
  // use InstructionCost's arithmetic properties such as saturation when this
  // feature is added to InstructionCost.
  // When storing the value back to NumInsts, we can assume all costs are Valid
  // because the IR should not contain any nodes that cannot be costed. If that
  // happens the cost-model is broken.
  InstructionCost NumInstsProxy = NumInsts;
  InstructionCost NumInstsBeforeThisBB = NumInsts;
  for (const Instruction &I : *BB) {
    // Skip ephemeral values.
    if (EphValues.count(&I))
      continue;

    // Special handling for calls.
    if (const auto *Call = dyn_cast<CallBase>(&I)) {
      if (const Function *F = Call->getCalledFunction()) {
        bool IsLoweredToCall = TTI.isLoweredToCall(F);
        // If a function is both internal and has a single use, then it is
        // extremely likely to get inlined in the future (it was probably
        // exposed by an interleaved devirtualization pass).
        // When preparing for LTO, liberally consider calls as inline
        // candidates.
        if (!Call->isNoInline() && IsLoweredToCall &&
            ((F->hasInternalLinkage() && F->hasOneUse()) || PrepareForLTO)) {
          ++NumInlineCandidates;
        }

        // If this call is to function itself, then the function is recursive.
        // Inlining it into other functions is a bad idea, because this is
        // basically just a form of loop peeling, and our metrics aren't useful
        // for that case.
        if (F == BB->getParent())
          isRecursive = true;

        if (IsLoweredToCall)
          ++NumCalls;
      } else {
        // We don't want inline asm to count as a call - that would prevent loop
        // unrolling. The argument setup cost is still real, though.
        if (!Call->isInlineAsm())
          ++NumCalls;
      }
    }

    if (const AllocaInst *AI = dyn_cast<AllocaInst>(&I)) {
      if (!AI->isStaticAlloca())
        this->usesDynamicAlloca = true;
    }

    if (isa<ExtractElementInst>(I) || I.getType()->isVectorTy())
      ++NumVectorInsts;

    if (I.getType()->isTokenTy() && I.isUsedOutsideOfBlock(BB))
      notDuplicatable = true;

    if (const CallInst *CI = dyn_cast<CallInst>(&I)) {
      if (CI->cannotDuplicate())
        notDuplicatable = true;
      if (CI->isConvergent())
        convergent = true;
    }

    if (const InvokeInst *InvI = dyn_cast<InvokeInst>(&I))
      if (InvI->cannotDuplicate())
        notDuplicatable = true;

    NumInstsProxy += TTI.getUserCost(&I, TargetTransformInfo::TCK_CodeSize);
    NumInsts = *NumInstsProxy.getValue();
  }

  if (isa<ReturnInst>(BB->getTerminator()))
    ++NumRets;

  // We never want to inline functions that contain an indirectbr.  This is
  // incorrect because all the blockaddress's (in static global initializers
  // for example) would be referring to the original function, and this indirect
  // jump would jump from the inlined copy of the function into the original
  // function which is extremely undefined behavior.
  // FIXME: This logic isn't really right; we can safely inline functions
  // with indirectbr's as long as no other function or global references the
  // blockaddress of a block within the current function.  And as a QOI issue,
  // if someone is using a blockaddress without an indirectbr, and that
  // reference somehow ends up in another function or global, we probably
  // don't want to inline this function.
  notDuplicatable |= isa<IndirectBrInst>(BB->getTerminator());

  // Remember NumInsts for this BB.
  InstructionCost NumInstsThisBB = NumInstsProxy - NumInstsBeforeThisBB;
  NumBBInsts[BB] = *NumInstsThisBB.getValue();
}

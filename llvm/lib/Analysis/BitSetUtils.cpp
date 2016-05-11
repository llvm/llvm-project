//===- BitSetUtils.cpp - Utilities related to pointer bitsets -------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file contains functions that make it easier to manipulate bitsets for
// devirtualization.
//
//===----------------------------------------------------------------------===//

#include "llvm/Analysis/BitSetUtils.h"
#include "llvm/IR/Intrinsics.h"
#include "llvm/IR/Module.h"

using namespace llvm;

// Search for virtual calls that call FPtr and add them to DevirtCalls.
static void
findCallsAtConstantOffset(SmallVectorImpl<DevirtCallSite> &DevirtCalls,
                          Value *FPtr, uint64_t Offset) {
  for (const Use &U : FPtr->uses()) {
    Value *User = U.getUser();
    if (isa<BitCastInst>(User)) {
      findCallsAtConstantOffset(DevirtCalls, User, Offset);
    } else if (auto CI = dyn_cast<CallInst>(User)) {
      DevirtCalls.push_back({Offset, CI});
    } else if (auto II = dyn_cast<InvokeInst>(User)) {
      DevirtCalls.push_back({Offset, II});
    }
  }
}

// Search for virtual calls that load from VPtr and add them to DevirtCalls.
static void
findLoadCallsAtConstantOffset(Module *M,
                              SmallVectorImpl<DevirtCallSite> &DevirtCalls,
                              Value *VPtr, uint64_t Offset) {
  for (const Use &U : VPtr->uses()) {
    Value *User = U.getUser();
    if (isa<BitCastInst>(User)) {
      findLoadCallsAtConstantOffset(M, DevirtCalls, User, Offset);
    } else if (isa<LoadInst>(User)) {
      findCallsAtConstantOffset(DevirtCalls, User, Offset);
    } else if (auto GEP = dyn_cast<GetElementPtrInst>(User)) {
      // Take into account the GEP offset.
      if (VPtr == GEP->getPointerOperand() && GEP->hasAllConstantIndices()) {
        SmallVector<Value *, 8> Indices(GEP->op_begin() + 1, GEP->op_end());
        uint64_t GEPOffset = M->getDataLayout().getIndexedOffsetInType(
            GEP->getSourceElementType(), Indices);
        findLoadCallsAtConstantOffset(M, DevirtCalls, User, Offset + GEPOffset);
      }
    }
  }
}

void llvm::findDevirtualizableCalls(
    SmallVectorImpl<DevirtCallSite> &DevirtCalls,
    SmallVectorImpl<CallInst *> &Assumes, CallInst *CI) {
  assert(CI->getCalledFunction()->getIntrinsicID() == Intrinsic::bitset_test);

  Module *M = CI->getParent()->getParent()->getParent();

  // Find llvm.assume intrinsics for this llvm.bitset.test call.
  for (const Use &CIU : CI->uses()) {
    auto AssumeCI = dyn_cast<CallInst>(CIU.getUser());
    if (AssumeCI) {
      Function *F = AssumeCI->getCalledFunction();
      if (F && F->getIntrinsicID() == Intrinsic::assume)
        Assumes.push_back(AssumeCI);
    }
  }

  // If we found any, search for virtual calls based on %p and add them to
  // DevirtCalls.
  if (!Assumes.empty())
    findLoadCallsAtConstantOffset(M, DevirtCalls,
                                  CI->getArgOperand(0)->stripPointerCasts(), 0);
}

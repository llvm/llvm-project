//===-- IR/Statepoint.cpp -- gc.statepoint utilities ---  -----------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file contains some utility functions to help recognize gc.statepoint
// intrinsics.
//
//===----------------------------------------------------------------------===//

#include "llvm/IR/Statepoint.h"

#include "llvm/IR/Function.h"

using namespace llvm;

bool llvm::isStatepoint(const CallBase *Call) {
  if (auto *F = Call->getCalledFunction())
    return F->getIntrinsicID() == Intrinsic::experimental_gc_statepoint;
  return false;
}

bool llvm::isStatepoint(const Value *V) {
  if (auto *Call = dyn_cast<CallBase>(V))
    return isStatepoint(Call);
  return false;
}

bool llvm::isStatepoint(const Value &V) {
  return isStatepoint(&V);
}

bool llvm::isGCRelocate(const CallBase *Call) {
  return isa<GCRelocateInst>(Call);
}

bool llvm::isGCRelocate(const Value *V) {
  if (auto *Call = dyn_cast<CallBase>(V))
    return isGCRelocate(Call);
  return false;
}

bool llvm::isGCResult(const CallBase *Call) { return isa<GCResultInst>(Call); }

bool llvm::isGCResult(const Value *V) {
  if (auto *Call = dyn_cast<CallBase>(V))
    return isGCResult(Call);
  return false;
}

bool llvm::isStatepointDirectiveAttr(Attribute Attr) {
  return Attr.hasAttribute("statepoint-id") ||
         Attr.hasAttribute("statepoint-num-patch-bytes");
}

StatepointDirectives
llvm::parseStatepointDirectivesFromAttrs(AttributeList AS) {
  StatepointDirectives Result;

  Attribute AttrID =
      AS.getAttribute(AttributeList::FunctionIndex, "statepoint-id");
  uint64_t StatepointID;
  if (AttrID.isStringAttribute())
    if (!AttrID.getValueAsString().getAsInteger(10, StatepointID))
      Result.StatepointID = StatepointID;

  uint32_t NumPatchBytes;
  Attribute AttrNumPatchBytes = AS.getAttribute(AttributeList::FunctionIndex,
                                                "statepoint-num-patch-bytes");
  if (AttrNumPatchBytes.isStringAttribute())
    if (!AttrNumPatchBytes.getValueAsString().getAsInteger(10, NumPatchBytes))
      Result.NumPatchBytes = NumPatchBytes;

  return Result;
}

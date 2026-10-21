//===-- AGXAliasAnalysis.cpp - AGX address space alias analysis ----------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "AGXAliasAnalysis.h"
#include "Utils/AGXBaseInfo.h"
#include "llvm/Analysis/ValueTracking.h"
#include "llvm/IR/Instructions.h"

using namespace llvm;

#define DEBUG_TYPE "agx-aa"

AnalysisKey AGXAA::Key;
char AGXAAWrapperPass::ID = 0;
char AGXExternalAAWrapper::ID = 0;

INITIALIZE_PASS(AGXAAWrapperPass, "agx-aa",
                "AGX address space alias analysis", false, true)
INITIALIZE_PASS(AGXExternalAAWrapper, "agx-aa-wrapper",
                "AGX address space alias analysis wrapper", false, true)

ImmutablePass *llvm::createAGXAAWrapperPass() { return new AGXAAWrapperPass(); }
ImmutablePass *llvm::createAGXExternalAAWrapperPass() {
  return new AGXExternalAAWrapper();
}

AGXAAWrapperPass::AGXAAWrapperPass() : ImmutablePass(ID) {}

void AGXAAWrapperPass::getAnalysisUsage(AnalysisUsage &AU) const {
  AU.setPreservesAll();
}

/// The address space of a pointer, looking through a few generic casts to the
/// object.
static unsigned addressSpace(const Value *V) {
  auto AS = [](const Value *V) {
    if (auto *T = dyn_cast<PointerType>(V->getType()))
      return T->getAddressSpace();
    return unsigned(AGX::GENERIC_ADDRESS);
  };
  for (unsigned I = 0; I < 6 && AS(V) == AGX::GENERIC_ADDRESS; ++I) {
    const Value *U = getUnderlyingObject(V, 1);
    if (U == V)
      break;
    V = U;
  }
  return AS(V);
}

AliasResult AGXAAResult::alias(const MemoryLocation &LocA,
                               const MemoryLocation &LocB, AAQueryInfo &,
                               const Instruction *) {
  unsigned A = addressSpace(LocA.Ptr), B = addressSpace(LocB.Ptr);
  if (A == AGX::GENERIC_ADDRESS || B == AGX::GENERIC_ADDRESS || A == B)
    return AliasResult::MayAlias;
  return AliasResult::NoAlias;
}

ModRefInfo AGXAAResult::getModRefInfoMask(const MemoryLocation &Loc,
                                          AAQueryInfo &, bool) {
  if (addressSpace(Loc.Ptr) == AGX::CONSTANT_ADDRESS)
    return ModRefInfo::NoModRef;
  return ModRefInfo::ModRef;
}

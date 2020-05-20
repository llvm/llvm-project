//===- AMDGPUAliasAnalysis ------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
/// \file
/// This is the AMGPU address space based alias analysis pass.
//===----------------------------------------------------------------------===//

#include "AMDGPUAliasAnalysis.h"
#include "AMDGPU.h"
#include "llvm/ADT/Triple.h"
#include "llvm/Analysis/AliasAnalysis.h"
#include "llvm/Analysis/MemoryLocation.h"
#include "llvm/Analysis/ValueTracking.h"
#include "llvm/IR/Argument.h"
#include "llvm/IR/Attributes.h"
#include "llvm/IR/CallingConv.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/GlobalVariable.h"
#include "llvm/IR/Type.h"
#include "llvm/IR/Value.h"
#include "llvm/Pass.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/ErrorHandling.h"
#include <cassert>

using namespace llvm;

#define DEBUG_TYPE "amdgpu-aa"

// Register this pass...
char AMDGPUAAWrapperPass::ID = 0;
char AMDGPUExternalAAWrapper::ID = 0;

INITIALIZE_PASS(AMDGPUAAWrapperPass, "amdgpu-aa",
                "AMDGPU Address space based Alias Analysis", false, true)

INITIALIZE_PASS(AMDGPUExternalAAWrapper, "amdgpu-aa-wrapper",
                "AMDGPU Address space based Alias Analysis Wrapper", false, true)

ImmutablePass *llvm::createAMDGPUAAWrapperPass() {
  return new AMDGPUAAWrapperPass();
}

ImmutablePass *llvm::createAMDGPUExternalAAWrapperPass() {
  return new AMDGPUExternalAAWrapper();
}

void AMDGPUAAWrapperPass::getAnalysisUsage(AnalysisUsage &AU) const {
  AU.setPreservesAll();
}

// These arrays are indexed by address space value enum elements 0 ... to 7
static const AliasResult ASAliasRules[8][8] = {
  /*                    Flat       Global    Region    Group     Constant  Private   Constant 32-bit  Buffer Fat Ptr */
  /* Flat     */        {MayAlias, MayAlias, NoAlias,  MayAlias, MayAlias, MayAlias, MayAlias,        MayAlias},
  /* Global   */        {MayAlias, MayAlias, NoAlias , NoAlias , MayAlias, NoAlias , MayAlias,        MayAlias},
  /* Region   */        {NoAlias,  NoAlias , MayAlias, NoAlias , NoAlias,  NoAlias , NoAlias,         NoAlias},
  /* Group    */        {MayAlias, NoAlias , NoAlias , MayAlias, NoAlias , NoAlias , NoAlias ,        NoAlias},
  /* Constant */        {MayAlias, MayAlias, NoAlias,  NoAlias , NoAlias , NoAlias , MayAlias,        MayAlias},
  /* Private  */        {MayAlias, NoAlias , NoAlias , NoAlias , NoAlias , MayAlias, NoAlias ,        NoAlias},
  /* Constant 32-bit */ {MayAlias, MayAlias, NoAlias,  NoAlias , MayAlias, NoAlias , NoAlias ,        MayAlias},
  /* Buffer Fat Ptr  */ {MayAlias, MayAlias, NoAlias , NoAlias , MayAlias, NoAlias , MayAlias,        MayAlias}
};

static AliasResult getAliasResult(unsigned AS1, unsigned AS2) {
  static_assert(AMDGPUAS::MAX_AMDGPU_ADDRESS <= 7, "Addr space out of range");

  if (AS1 > AMDGPUAS::MAX_AMDGPU_ADDRESS || AS2 > AMDGPUAS::MAX_AMDGPU_ADDRESS)
    return MayAlias;

  return ASAliasRules[AS1][AS2];
}

AliasResult AMDGPUAAResult::alias(const MemoryLocation &LocA,
                                  const MemoryLocation &LocB,
                                  AAQueryInfo &AAQI) {
  unsigned asA = LocA.Ptr->getType()->getPointerAddressSpace();
  unsigned asB = LocB.Ptr->getType()->getPointerAddressSpace();

  AliasResult Result = getAliasResult(asA, asB);
  if (Result == NoAlias)
    return Result;

  // Forward the query to the next alias analysis.
  return AAResultBase::alias(LocA, LocB, AAQI);
}

bool AMDGPUAAResult::pointsToConstantMemory(const MemoryLocation &Loc,
                                            AAQueryInfo &AAQI, bool OrLocal) {
  unsigned AS = Loc.Ptr->getType()->getPointerAddressSpace();
  if (AS == AMDGPUAS::CONSTANT_ADDRESS ||
      AS == AMDGPUAS::CONSTANT_ADDRESS_32BIT)
    return true;

  const Value *Base = GetUnderlyingObject(Loc.Ptr, DL);
  AS = Base->getType()->getPointerAddressSpace();
  if (AS == AMDGPUAS::CONSTANT_ADDRESS ||
      AS == AMDGPUAS::CONSTANT_ADDRESS_32BIT)
    return true;

  if (const GlobalVariable *GV = dyn_cast<GlobalVariable>(Base)) {
    if (GV->isConstant())
      return true;
  } else if (const Argument *Arg = dyn_cast<Argument>(Base)) {
    const Function *F = Arg->getParent();

    // Only assume constant memory for arguments on kernels.
    switch (F->getCallingConv()) {
    default:
      return AAResultBase::pointsToConstantMemory(Loc, AAQI, OrLocal);
    case CallingConv::AMDGPU_LS:
    case CallingConv::AMDGPU_HS:
    case CallingConv::AMDGPU_ES:
    case CallingConv::AMDGPU_GS:
    case CallingConv::AMDGPU_VS:
    case CallingConv::AMDGPU_PS:
    case CallingConv::AMDGPU_CS:
    case CallingConv::AMDGPU_KERNEL:
    case CallingConv::SPIR_KERNEL:
      break;
    }

    unsigned ArgNo = Arg->getArgNo();
    /* On an argument, ReadOnly attribute indicates that the function does
       not write through this pointer argument, even though it may write
       to the memory that the pointer points to.
       On an argument, ReadNone attribute indicates that the function does
       not dereference that pointer argument, even though it may read or write
       the memory that the pointer points to if accessed through other pointers.
     */
    if (F->hasParamAttribute(ArgNo, Attribute::NoAlias) &&
        (F->hasParamAttribute(ArgNo, Attribute::ReadNone) ||
         F->hasParamAttribute(ArgNo, Attribute::ReadOnly))) {
      return true;
    }
  }
  return AAResultBase::pointsToConstantMemory(Loc, AAQI, OrLocal);
}

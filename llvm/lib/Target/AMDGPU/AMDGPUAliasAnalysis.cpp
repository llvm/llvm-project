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
#include "llvm/Analysis/ValueTracking.h"
#include "llvm/IR/Instructions.h"

using namespace llvm;

#define DEBUG_TYPE "amdgpu-aa"

AnalysisKey AMDGPUAA::Key;

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

AMDGPUAAWrapperPass::AMDGPUAAWrapperPass() : ImmutablePass(ID) {
  initializeAMDGPUAAWrapperPassPass(*PassRegistry::getPassRegistry());
}

void AMDGPUAAWrapperPass::getAnalysisUsage(AnalysisUsage &AU) const {
  AU.setPreservesAll();
}

static AliasResult getAliasResult(unsigned AS1, unsigned AS2) {
  static_assert(AMDGPUAS::MAX_AMDGPU_ADDRESS <= 7, "Addr space out of range");

  if (AS1 > AMDGPUAS::MAX_AMDGPU_ADDRESS || AS2 > AMDGPUAS::MAX_AMDGPU_ADDRESS)
    return AliasResult::MayAlias;

#define ASMay AliasResult::MayAlias
#define ASNo AliasResult::NoAlias
  // This array is indexed by address space value enum elements 0 ... to 7
  static const AliasResult ASAliasRules[8][8] = {
    /*                    Flat    Global Region Group  Constant Private Const32 Buf Fat Ptr */
    /* Flat     */        {ASMay, ASMay, ASNo,  ASMay, ASMay,   ASMay,  ASMay,  ASMay},
    /* Global   */        {ASMay, ASMay, ASNo,  ASNo,  ASMay,   ASNo,   ASMay,  ASMay},
    /* Region   */        {ASNo,  ASNo,  ASMay, ASNo,  ASNo,    ASNo,   ASNo,   ASNo},
    /* Group    */        {ASMay, ASNo,  ASNo,  ASMay, ASNo,    ASNo,   ASNo,   ASNo},
    /* Constant */        {ASMay, ASMay, ASNo,  ASNo,  ASNo,    ASNo,   ASMay,  ASMay},
    /* Private  */        {ASMay, ASNo,  ASNo,  ASNo,  ASNo,    ASMay,  ASNo,   ASNo},
    /* Constant 32-bit */ {ASMay, ASMay, ASNo,  ASNo,  ASMay,   ASNo,   ASNo,   ASMay},
    /* Buffer Fat Ptr  */ {ASMay, ASMay, ASNo,  ASNo,  ASMay,   ASNo,   ASMay,  ASMay}
  };
#undef ASMay
#undef ASNo

  return ASAliasRules[AS1][AS2];
}

AliasResult AMDGPUAAResult::alias(const MemoryLocation &LocA,
                                  const MemoryLocation &LocB, AAQueryInfo &AAQI,
                                  const Instruction *) {
  unsigned asA = LocA.Ptr->getType()->getPointerAddressSpace();
  unsigned asB = LocB.Ptr->getType()->getPointerAddressSpace();

  AliasResult Result = getAliasResult(asA, asB);
  if (Result == AliasResult::NoAlias)
    return Result;

  // In general, FLAT (generic) pointers could be aliased to LOCAL or PRIVATE
  // pointers. However, as LOCAL or PRIVATE pointers point to local objects, in
  // certain cases, it's still viable to check whether a FLAT pointer won't
  // alias to a LOCAL or PRIVATE pointer.
  MemoryLocation A = LocA;
  MemoryLocation B = LocB;
  // Canonicalize the location order to simplify the following alias check.
  if (asA != AMDGPUAS::FLAT_ADDRESS) {
    std::swap(asA, asB);
    std::swap(A, B);
  }
  if (asA == AMDGPUAS::FLAT_ADDRESS &&
      (asB == AMDGPUAS::LOCAL_ADDRESS || asB == AMDGPUAS::PRIVATE_ADDRESS)) {
    const auto *ObjA =
        getUnderlyingObject(A.Ptr->stripPointerCastsForAliasAnalysis());
    if (const LoadInst *LI = dyn_cast<LoadInst>(ObjA)) {
      // If a generic pointer is loaded from the constant address space, it
      // could only be a GLOBAL or CONSTANT one as that address space is solely
      // prepared on the host side, where only GLOBAL or CONSTANT variables are
      // visible. Note that this even holds for regular functions.
      if (LI->getPointerAddressSpace() == AMDGPUAS::CONSTANT_ADDRESS)
        return AliasResult::NoAlias;
    } else if (const Argument *Arg = dyn_cast<Argument>(ObjA)) {
      const Function *F = Arg->getParent();
      switch (F->getCallingConv()) {
      case CallingConv::AMDGPU_KERNEL:
        // In the kernel function, kernel arguments won't alias to (local)
        // variables in shared or private address space.
        return AliasResult::NoAlias;
      default:
        // TODO: In the regular function, if that local variable in the
        // location B is not captured, that argument pointer won't alias to it
        // as well.
        break;
      }
    }
  }

  // Forward the query to the next alias analysis.
  return AAResultBase::alias(LocA, LocB, AAQI, nullptr);
}

ModRefInfo AMDGPUAAResult::getModRefInfoMask(const MemoryLocation &Loc,
                                             AAQueryInfo &AAQI,
                                             bool IgnoreLocals) {
  unsigned AS = Loc.Ptr->getType()->getPointerAddressSpace();
  if (AS == AMDGPUAS::CONSTANT_ADDRESS ||
      AS == AMDGPUAS::CONSTANT_ADDRESS_32BIT)
    return ModRefInfo::NoModRef;

  const Value *Base = getUnderlyingObject(Loc.Ptr);
  AS = Base->getType()->getPointerAddressSpace();
  if (AS == AMDGPUAS::CONSTANT_ADDRESS ||
      AS == AMDGPUAS::CONSTANT_ADDRESS_32BIT)
    return ModRefInfo::NoModRef;

  return AAResultBase::getModRefInfoMask(Loc, AAQI, IgnoreLocals);
}

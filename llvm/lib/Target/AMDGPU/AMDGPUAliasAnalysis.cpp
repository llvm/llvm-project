//===- AMDGPUAliasAnalysis ------------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
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

INITIALIZE_PASS(AMDGPUAAWrapperPass, "amdgpu-aa",
                "AMDGPU Address space based Alias Analysis", false, true)

ImmutablePass *llvm::createAMDGPUAAWrapperPass() {
  return new AMDGPUAAWrapperPass();
}

void AMDGPUAAWrapperPass::getAnalysisUsage(AnalysisUsage &AU) const {
  AU.setPreservesAll();
}

// Must match the table in getAliasResult.
AMDGPUAAResult::ASAliasRulesTy::ASAliasRulesTy(AMDGPUAS AS_, Triple::ArchType Arch_)
  : Arch(Arch_), AS(AS_) {
  // These arrarys are indexed by address space value
  // enum elements 0 ... to 6
  static const AliasResult ASAliasRulesPrivIsZero[7][7] = {
  /*                    Private    Global    Constant  Group     Flat      Region    Constant 32-bit */
  /* Private  */        {MayAlias, NoAlias , NoAlias , NoAlias , MayAlias, NoAlias , NoAlias},
  /* Global   */        {NoAlias , MayAlias, MayAlias, NoAlias , MayAlias, NoAlias , MayAlias},
  /* Constant */        {NoAlias , MayAlias, MayAlias, NoAlias , MayAlias, NoAlias , MayAlias},
  /* Group    */        {NoAlias , NoAlias , NoAlias , MayAlias, MayAlias, NoAlias , NoAlias},
  /* Flat     */        {MayAlias, MayAlias, MayAlias, MayAlias, MayAlias, MayAlias, MayAlias},
  /* Region   */        {NoAlias , NoAlias , NoAlias , NoAlias , MayAlias, MayAlias, NoAlias},
  /* Constant 32-bit */ {NoAlias , MayAlias, MayAlias, NoAlias , MayAlias, NoAlias , MayAlias}
  };
  static const AliasResult ASAliasRulesGenIsZero[7][7] = {
  /*                    Flat       Global    Region    Group     Constant  Private   Constant 32-bit */
  /* Flat     */        {MayAlias, MayAlias, MayAlias, MayAlias, MayAlias, MayAlias, MayAlias},
  /* Global   */        {MayAlias, MayAlias, NoAlias , NoAlias , MayAlias, NoAlias , MayAlias},
  /* Region   */        {MayAlias, NoAlias , NoAlias , NoAlias,  MayAlias, NoAlias , MayAlias},
  /* Group    */        {MayAlias, NoAlias , NoAlias , MayAlias, NoAlias , NoAlias , NoAlias},
  /* Constant */        {MayAlias, MayAlias, MayAlias, NoAlias , NoAlias,  NoAlias , MayAlias},
  /* Private  */        {MayAlias, NoAlias , NoAlias , NoAlias , NoAlias , MayAlias, NoAlias},
  /* Constant 32-bit */ {MayAlias, MayAlias, MayAlias, NoAlias , MayAlias, NoAlias , NoAlias}
  };
  static_assert(AMDGPUAS::MAX_AMDGPU_ADDRESS <= 6, "Addr space out of range");
  if (AS.FLAT_ADDRESS == 0) {
    assert(AS.GLOBAL_ADDRESS         == 1 &&
           AS.REGION_ADDRESS         == 2 &&
           AS.LOCAL_ADDRESS          == 3 &&
           AS.CONSTANT_ADDRESS       == 4 &&
           AS.PRIVATE_ADDRESS        == 5 &&
           AS.CONSTANT_ADDRESS_32BIT == 6);
    ASAliasRules = &ASAliasRulesGenIsZero;
  } else {
    assert(AS.PRIVATE_ADDRESS        == 0 &&
           AS.GLOBAL_ADDRESS         == 1 &&
           AS.CONSTANT_ADDRESS       == 2 &&
           AS.LOCAL_ADDRESS          == 3 &&
           AS.FLAT_ADDRESS           == 4 &&
           AS.REGION_ADDRESS         == 5 &&
           AS.CONSTANT_ADDRESS_32BIT == 6);
    ASAliasRules = &ASAliasRulesPrivIsZero;
  }
}

AliasResult AMDGPUAAResult::ASAliasRulesTy::getAliasResult(unsigned AS1,
    unsigned AS2) const {
  if (AS1 > AS.MAX_AMDGPU_ADDRESS || AS2 > AS.MAX_AMDGPU_ADDRESS) {
    if (Arch == Triple::amdgcn)
      report_fatal_error("Pointer address space out of range");
    return AS1 == AS2 ? MayAlias : NoAlias;
  }

  return (*ASAliasRules)[AS1][AS2];
}

AliasResult AMDGPUAAResult::alias(const MemoryLocation &LocA,
                                  const MemoryLocation &LocB) {
  unsigned asA = LocA.Ptr->getType()->getPointerAddressSpace();
  unsigned asB = LocB.Ptr->getType()->getPointerAddressSpace();

  AliasResult Result = ASAliasRules.getAliasResult(asA, asB);
  if (Result == NoAlias) return Result;

  // Forward the query to the next alias analysis.
  return AAResultBase::alias(LocA, LocB);
}

bool AMDGPUAAResult::pointsToConstantMemory(const MemoryLocation &Loc,
                                            bool OrLocal) {
  const Value *Base = GetUnderlyingObject(Loc.Ptr, DL);

  if (Base->getType()->getPointerAddressSpace() == AS.CONSTANT_ADDRESS ||
      Base->getType()->getPointerAddressSpace() == AS.CONSTANT_ADDRESS_32BIT) {
    return true;
  }

  if (const GlobalVariable *GV = dyn_cast<GlobalVariable>(Base)) {
    if (GV->isConstant())
      return true;
  } else if (const Argument *Arg = dyn_cast<Argument>(Base)) {
    const Function *F = Arg->getParent();

    // Only assume constant memory for arguments on kernels.
    switch (F->getCallingConv()) {
    default:
      return AAResultBase::pointsToConstantMemory(Loc, OrLocal);
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
  return AAResultBase::pointsToConstantMemory(Loc, OrLocal);
}

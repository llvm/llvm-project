//===- AMDGPUAliasAnalysis ---------------------------------------*- C++ -*-==//
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

#include "AMDGPU.h"
#include "AMDGPUAliasAnalysis.h"
#include "llvm/Analysis/AliasAnalysis.h"
#include "llvm/Analysis/ValueTracking.h"
#include "llvm/Analysis/Passes.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/Module.h"
#include "llvm/Pass.h"

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

AliasResult AMDGPUAAResult::alias(const MemoryLocation &LocA,
                                  const MemoryLocation &LocB) {
  // This array is indexed by the AMDGPUAS::AddressSpaces
  // enum elements PRIVATE_ADDRESS ... to FLAT_ADDRESS
  // see "llvm/Transforms/AMDSPIRUtils.h"
  static const AliasResult ASAliasRules[5][5] = {
 /*             Private    Global    Constant  Group     Flat */
 /* Private  */ {MayAlias, NoAlias , NoAlias , NoAlias , MayAlias},
 /* Global   */ {NoAlias , MayAlias, NoAlias , NoAlias , MayAlias},
 /* Constant */ {NoAlias , NoAlias , MayAlias, NoAlias , MayAlias},
 /* Group    */ {NoAlias , NoAlias , NoAlias , MayAlias, MayAlias},
 /* Flat     */ {MayAlias, MayAlias, MayAlias, MayAlias, MayAlias}
  };
  unsigned asA = LocA.Ptr->getType()->getPointerAddressSpace();
  unsigned asB = LocB.Ptr->getType()->getPointerAddressSpace();
  if (asA > AMDGPUAS::AddressSpaces::FLAT_ADDRESS ||
      asB > AMDGPUAS::AddressSpaces::FLAT_ADDRESS)
    report_fatal_error("Pointer address space out of range");

  AliasResult Result = ASAliasRules[asA][asB];
  if (Result == NoAlias) return Result;

  if (isa<Argument>(LocA.Ptr) && isa<Argument>(LocB.Ptr)) {
    Type *T1 = cast<PointerType>(LocA.Ptr->getType())->getElementType();
    Type *T2 = cast<PointerType>(LocB.Ptr->getType())->getElementType();

    if ((T1->isVectorTy() && !T2->isVectorTy()) ||
        (T2->isVectorTy() && !T1->isVectorTy()))
      return NoAlias;
  }
  // Forward the query to the next alias analysis.
  return AAResultBase::alias(LocA, LocB);
}

bool AMDGPUAAResult::pointsToConstantMemory(const MemoryLocation &Loc,
                                            bool OrLocal) {
  const Value *Base = GetUnderlyingObject(Loc.Ptr, DL);

  if (Base->getType()->getPointerAddressSpace() ==
     AMDGPUAS::AddressSpaces::CONSTANT_ADDRESS) {
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
    case CallingConv::AMDGPU_VS:
    case CallingConv::AMDGPU_GS:
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
    if (F->getAttributes().hasAttribute(ArgNo + 1, Attribute::NoAlias) &&
          (F->getAttributes().hasAttribute(ArgNo + 1, Attribute::ReadNone) ||
           F->getAttributes().hasAttribute(ArgNo + 1, Attribute::ReadOnly))) {
      return true;
    }
  }
  return AAResultBase::pointsToConstantMemory(Loc, OrLocal);
}

//===- DXILRefineTypes.cpp ----------------------------------------------===////
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "DXILRefineTypes.h"
#include "llvm/IR/DerivedTypes.h"
#include "llvm/IR/GlobalVariable.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/Intrinsics.h"
#include "llvm/IR/Module.h"

using namespace llvm;

#define DEBUG_TYPE "dxil-refine-types"

static Type *inferType(Value *Operand) {
  if (auto *CI = dyn_cast<CallInst>(Operand))
    if (CI->getCalledFunction()->getName().starts_with(
            "llvm.dx.resource.getpointer"))
      if (auto *ExtType = dyn_cast<TargetExtType>(CI->getOperand(0)->getType()))
        return ExtType->getTypeParameter(0); // Valid for all dx.Types

  if (auto *AI = dyn_cast<AllocaInst>(Operand))
    return AI->getAllocatedType();

  // TODO: Extend to other useful/applicable cases
  return nullptr;
}

// Attempt to merge two inferred types.
//
// Returns nullptr, if an inferred can't be concluded
static Type *mergeInferredTypes(Type *A, Type *B) {
  if (!A)
    return B;

  if (!B)
    return A;

  if (A == B)
    return A;

  // Otherwise, neither was inferred, or inferred differently
  return nullptr;
}

bool DXILRefineTypesPass::runImpl(Function &F) {
  // First detect the pattern: (generated from SimplifyAnyMemTransfer)
  //   %temp = load Type, ptr %src, ...
  //   store Type %temp, ptr %dest, ...
  // where,
  //   store is the only user of %temp
  //   Type is either an i32 or i64
  //
  // We are currently only concerned with i32 and i64 as these can incidently
  // promote 16/32 bit types to 32/64 bit arthimetic.
  SmallVector<std::pair<LoadInst *, StoreInst *>, 4> ToVisit;
  for (BasicBlock &BB : F)
    for (Instruction &I : BB)
      if (auto *LI = dyn_cast<LoadInst>(&I))
        if (LI->hasOneUse())
          if (auto *SI = dyn_cast<StoreInst>(LI->user_back()))
            if (LI->getAccessType() == SI->getAccessType())
              if (LI->getAccessType()->isIntegerTy(32) ||
                  LI->getAccessType()->isIntegerTy(64))
                ToVisit.push_back({LI, SI});

  bool Modified = false;
  for (auto [LI, SI] : ToVisit) {
    Type *LoadTy = inferType(LI->getPointerOperand());
    Type *StoreTy = inferType(SI->getPointerOperand());

    Type *const InferredTy = mergeInferredTypes(LoadTy, StoreTy);
    if (!InferredTy || InferredTy == LI->getType())
      continue; // Nothing to be done. Skip.

    // Replace the type of the load/store
    IRBuilder<> LoadBuilder(SI);
    LoadInst *TypedLoad =
        LoadBuilder.CreateLoad(InferredTy, LI->getPointerOperand());

    TypedLoad->setAlignment(LI->getAlign());
    TypedLoad->setVolatile(LI->isVolatile());
    TypedLoad->setOrdering(LI->getOrdering());
    TypedLoad->setAAMetadata(LI->getAAMetadata());
    TypedLoad->copyMetadata(*LI, LLVMContext::MD_mem_parallel_loop_access);
    TypedLoad->copyMetadata(*LI, LLVMContext::MD_access_group);

    IRBuilder<> StoreBuilder(SI);
    StoreInst *TypedStore = StoreBuilder.CreateStore(
        TypedLoad, SI->getPointerOperand(), SI->isVolatile());

    TypedStore->setAlignment(SI->getAlign());
    TypedStore->setVolatile(SI->isVolatile());
    TypedStore->setOrdering(SI->getOrdering());
    TypedStore->setAAMetadata(SI->getAAMetadata());
    TypedStore->copyMetadata(*SI, LLVMContext::MD_mem_parallel_loop_access);
    TypedStore->copyMetadata(*SI, LLVMContext::MD_access_group);
    TypedStore->copyMetadata(*SI, LLVMContext::MD_DIAssignID);

    SI->eraseFromParent();
    LI->eraseFromParent();

    Modified = true;
  }

  return Modified;
}

PreservedAnalyses DXILRefineTypesPass::run(Function &F,
                                           FunctionAnalysisManager &AM) {
  if (!runImpl(F))
    return PreservedAnalyses::all();

  // TODO: Can probably preserve some CFG analyses
  return PreservedAnalyses::none();
}

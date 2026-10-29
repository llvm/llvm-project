//===- AIRScalarArgsToConstantBuffers.cpp - Scalars as constant buffers --===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "AIRScalarArgsToConstantBuffers.h"
#include "AIR.h"
#include "AIRAddressSpaces.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/IR/Attributes.h"
#include "llvm/IR/DerivedTypes.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/Module.h"
#include "llvm/InitializePasses.h"

using namespace llvm;

#define DEBUG_TYPE "air-scalar-args-to-constant-buffers"

// The type a by-value argument of type T is held in, or nullptr when the
// argument is not a scalar (pointers and aggregates keep their form).
static Type *storageType(Type *T) {
  Type *Elt = T->getScalarType();
  if (!Elt->isIntegerTy() && !Elt->isFloatingPointTy())
    return nullptr;
  if (!Elt->isIntegerTy(1))
    return T;
  Type *I8 = Type::getInt8Ty(T->getContext());
  if (auto *VT = dyn_cast<VectorType>(T))
    return VectorType::get(I8, VT->getElementCount());
  return I8;
}

static bool isCalledDirectly(const Function &F) {
  return any_of(F.users(), [&](const User *U) {
    auto *CB = dyn_cast<CallBase>(U);
    return CB && CB->getCalledFunction() == &F;
  });
}

static bool rewriteKernel(Function &F) {
  // Other uses (an address taken, a table) would see the new signature.
  if (any_of(F.users(), [](const User *U) { return !isa<CallBase>(U); }))
    return false;
  if (none_of(F.args(), [](Argument &A) { return storageType(A.getType()); }))
    return false;

  Module &M = *F.getParent();
  LLVMContext &Ctx = M.getContext();
  const DataLayout &DL = M.getDataLayout();
  PointerType *ConstPtr = PointerType::get(Ctx, metal::AS::Constant);

  FunctionType *OldFTy = F.getFunctionType();
  SmallVector<Type *, 8> Params;
  SmallVector<AttributeSet, 8> ParamAttrs;
  AttributeList OldAttrs = F.getAttributes();
  for (Argument &A : F.args()) {
    Type *Storage = storageType(A.getType());
    if (!Storage) {
      Params.push_back(A.getType());
      ParamAttrs.push_back(OldAttrs.getParamAttrs(A.getArgNo()));
      continue;
    }
    Params.push_back(ConstPtr);
    AttrBuilder AB(Ctx);
    // the parameter's string attributes (its role for the host) stay with it
    for (Attribute Att : OldAttrs.getParamAttrs(A.getArgNo()))
      if (Att.isStringAttribute())
        AB.addAttribute(Att);
    AB.addAttribute(Attribute::NoAlias);
    AB.addAttribute(Attribute::NonNull);
    AB.addAttribute(Attribute::NoUndef);
    AB.addAttribute(Attribute::ReadOnly);
    AB.addAlignmentAttr(DL.getABITypeAlign(Storage));
    AB.addDereferenceableAttr(DL.getTypeStoreSize(Storage));
    ParamAttrs.push_back(AttributeSet::get(Ctx, AB));
  }

  auto *NewFTy =
      FunctionType::get(OldFTy->getReturnType(), Params, OldFTy->isVarArg());
  Function *NewF =
      Function::Create(NewFTy, F.getLinkage(), F.getAddressSpace(), "", &M);
  NewF->copyAttributesFrom(&F);
  NewF->setAttributes(AttributeList::get(Ctx, OldAttrs.getFnAttrs(),
                                         OldAttrs.getRetAttrs(), ParamAttrs));
  NewF->copyMetadata(&F, 0);
  NewF->splice(NewF->begin(), &F);

  IRBuilder<> B(&*NewF->getEntryBlock().getFirstInsertionPt());
  for (auto [Old, New] : zip(F.args(), NewF->args())) {
    New.setName(Old.getName());
    Type *Storage = storageType(Old.getType());
    if (!Storage) {
      Old.replaceAllUsesWith(&New);
      continue;
    }
    Value *V = B.CreateAlignedLoad(Storage, &New, DL.getABITypeAlign(Storage),
                                   Old.getName() + ".val");
    if (Storage != Old.getType())
      V = B.CreateTrunc(V, Old.getType());
    Old.replaceAllUsesWith(V);
  }

  // Keep metadata that names the kernel pointing at the new function.
  ValueAsMetadata::handleRAUW(&F, NewF);
  std::string Name = F.getName().str();
  F.eraseFromParent();
  NewF->setName(Name);
  return true;
}

static bool scalarArgsToConstantBuffers(Module &M) {
  // A pre-baked !air.kernel describes the signature as it is.
  if (M.getNamedMetadata("air.kernel"))
    return false;
  // Kernels are the functions marked "air-kernel" when the module marks any,
  // otherwise those nothing calls (as air-system-values decides).
  bool HaveKernelAttr = any_of(M, [](const Function &F) {
    return !F.isDeclaration() && F.hasFnAttribute("air-kernel");
  });
  SmallVector<Function *, 4> Kernels;
  for (Function &F : M) {
    if (F.isDeclaration())
      continue;
    if (HaveKernelAttr ? F.hasFnAttribute("air-kernel")
                       : !isCalledDirectly(F))
      Kernels.push_back(&F);
  }
  bool Changed = false;
  for (Function *F : Kernels)
    Changed |= rewriteKernel(*F);
  return Changed;
}

PreservedAnalyses
AIRScalarArgsToConstantBuffersPass::run(Module &M, ModuleAnalysisManager &AM) {
  return scalarArgsToConstantBuffers(M) ? PreservedAnalyses::none()
                                        : PreservedAnalyses::all();
}

bool AIRScalarArgsToConstantBuffersLegacy::runOnModule(Module &M) {
  return scalarArgsToConstantBuffers(M);
}

char AIRScalarArgsToConstantBuffersLegacy::ID = 0;

INITIALIZE_PASS(AIRScalarArgsToConstantBuffersLegacy, DEBUG_TYPE,
                "AIR Scalar Arguments To Constant Buffers", false, false)

ModulePass *llvm::createAIRScalarArgsToConstantBuffersLegacyPass() {
  return new AIRScalarArgsToConstantBuffersLegacy();
}

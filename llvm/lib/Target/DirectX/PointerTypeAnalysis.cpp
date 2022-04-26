//===- Target/DirectX/PointerTypeAnalisis.cpp - PointerType analysis ------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Analysis pass to assign types to opaque pointers.
//
//===----------------------------------------------------------------------===//

#include "PointerTypeAnalysis.h"
#include "llvm/IR/Instructions.h"

using namespace llvm;
using namespace llvm::dxil;

namespace {

// Classifies the type of the value passed in by walking the value's users to
// find a typed instruction to materialize a type from.
TypedPointerType *classifyPointerType(const Value *V) {
  assert(V->getType()->isOpaquePointerTy() &&
         "classifyPointerType called with non-opaque pointer");
  Type *PointeeTy = nullptr;
  if (auto *Inst = dyn_cast<GetElementPtrInst>(V)) {
    if (!Inst->getResultElementType()->isOpaquePointerTy())
      PointeeTy = Inst->getResultElementType();
  } else if (auto *Inst = dyn_cast<AllocaInst>(V)) {
    PointeeTy = Inst->getAllocatedType();
  }
  for (const auto *User : V->users()) {
    Type *NewPointeeTy = nullptr;
    if (const auto *Inst = dyn_cast<LoadInst>(User)) {
      NewPointeeTy = Inst->getType();
    } else if (const auto *Inst = dyn_cast<StoreInst>(User)) {
      NewPointeeTy = Inst->getValueOperand()->getType();
    } else if (const auto *Inst = dyn_cast<GetElementPtrInst>(User)) {
      NewPointeeTy = Inst->getSourceElementType();
    }
    if (NewPointeeTy) {
      // HLSL doesn't support pointers, so it is unlikely to get more than one
      // or two levels of indirection in the IR. Because of this, recursion is
      // pretty safe.
      if (NewPointeeTy->isOpaquePointerTy())
        return TypedPointerType::get(classifyPointerType(User),
                                     V->getType()->getPointerAddressSpace());
      if (!PointeeTy)
        PointeeTy = NewPointeeTy;
      else if (PointeeTy != NewPointeeTy)
        PointeeTy = Type::getInt8Ty(V->getContext());
    }
  }
  // If we were unable to determine the pointee type, set to i8
  if (!PointeeTy)
    PointeeTy = Type::getInt8Ty(V->getContext());
  return TypedPointerType::get(PointeeTy,
                               V->getType()->getPointerAddressSpace());
}

// This function constructs a function type accepting typed pointers. It only
// handles function arguments and return types, and assigns the function type to
// the function's value in the type map.
void classifyFunctionType(const Function &F, PointerTypeMap &Map) {
  SmallVector<Type *, 8> NewArgs;
  bool HasOpaqueTy = false;
  Type *RetTy = F.getReturnType();
  if (RetTy->isOpaquePointerTy()) {
    RetTy = nullptr;
    for (const auto &B : F) {
      for (const auto &I : B) {
        if (const auto *RetInst = dyn_cast_or_null<ReturnInst>(&I)) {
          Type *NewRetTy = classifyPointerType(RetInst->getReturnValue());
          if (!RetTy)
            RetTy = NewRetTy;
          else if (RetTy != NewRetTy)
            RetTy = TypedPointerType::get(
                Type::getInt8Ty(I.getContext()),
                F.getReturnType()->getPointerAddressSpace());
        }
      }
    }
  }
  for (auto &A : F.args()) {
    Type *ArgTy = A.getType();
    if (ArgTy->isOpaquePointerTy()) {
      TypedPointerType *NewTy = classifyPointerType(&A);
      Map[&A] = NewTy;
      ArgTy = NewTy;
      HasOpaqueTy = true;
    }
    NewArgs.push_back(ArgTy);
  }
  if (!HasOpaqueTy)
    return;
  Map[&F] = FunctionType::get(RetTy, NewArgs, false);
}
} // anonymous namespace

PointerTypeMap PointerTypeAnalysis::run(const Module &M) {
  PointerTypeMap Map;
  for (auto &G : M.globals()) {
    if (G.getType()->isOpaquePointerTy())
      Map[&G] = classifyPointerType(&G);
  }
  for (auto &F : M) {
    classifyFunctionType(F, Map);

    for (const auto &B : F) {
      for (const auto &I : B) {
        if (I.getType()->isOpaquePointerTy())
          Map[&I] = classifyPointerType(&I);
      }
    }
  }

  return Map;
}

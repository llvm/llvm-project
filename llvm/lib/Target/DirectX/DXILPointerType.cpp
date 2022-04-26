//===- Target/DirectX/DXILTypedPointerType.cpp - DXIL Typed Pointer Type
//-------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
//
//===----------------------------------------------------------------------===//

#include "DXILPointerType.h"
#include "llvm/ADT/Any.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/IR/LLVMContext.h"

using namespace llvm;
using namespace llvm::dxil;

class TypedPointerTracking {
public:
  TypedPointerTracking() {}
  DenseMap<Type *, std::unique_ptr<TypedPointerType>> PointerTypes;
  DenseMap<std::pair<Type *, unsigned>, std::unique_ptr<TypedPointerType>>
      ASPointerTypes;
};

TypedPointerType *TypedPointerType::get(Type *EltTy, unsigned AddressSpace) {
  assert(EltTy && "Can't get a pointer to <null> type!");
  assert(isValidElementType(EltTy) && "Invalid type for pointer element!");

  llvm::Any &TargetData = EltTy->getContext().getTargetData();
  if (!TargetData.hasValue())
    TargetData = Any{std::make_shared<TypedPointerTracking>()};

  assert(any_isa<std::shared_ptr<TypedPointerTracking>>(TargetData) &&
         "Unexpected target data type");

  std::shared_ptr<TypedPointerTracking> Tracking =
      any_cast<std::shared_ptr<TypedPointerTracking>>(TargetData);

  // Since AddressSpace #0 is the common case, we special case it.
  std::unique_ptr<TypedPointerType> &Entry =
      AddressSpace == 0
          ? Tracking->PointerTypes[EltTy]
          : Tracking->ASPointerTypes[std::make_pair(EltTy, AddressSpace)];

  if (!Entry)
    Entry = std::unique_ptr<TypedPointerType>(
        new TypedPointerType(EltTy, AddressSpace));
  return Entry.get();
}

TypedPointerType::TypedPointerType(Type *E, unsigned AddrSpace)
    : Type(E->getContext(), DXILPointerTyID), PointeeTy(E) {
  ContainedTys = &PointeeTy;
  NumContainedTys = 1;
  setSubclassData(AddrSpace);
}

bool TypedPointerType::isValidElementType(Type *ElemTy) {
  return !ElemTy->isVoidTy() && !ElemTy->isLabelTy() &&
         !ElemTy->isMetadataTy() && !ElemTy->isTokenTy() &&
         !ElemTy->isX86_AMXTy();
}

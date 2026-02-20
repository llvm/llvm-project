//===- Value.cpp - Value Representation for llubi -------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements utility functions for the value representation.
//
//===----------------------------------------------------------------------===//

#include "Value.h"
#include "Context.h"
#include "llvm/ADT/SmallString.h"

namespace llvm::ubi {

void Pointer::print(raw_ostream &OS) const {
  SmallString<32> AddrStr;
  Address.toStringUnsigned(AddrStr, 16);
  OS << "ptr 0x" << AddrStr << " [";
  if (Obj && Obj->getState() != MemoryObjectState::Freed) {
    OS << Obj->getName();
    // TODO: print " (dead)" if the stack object is out of lifetime.
    if (Address != Obj->getAddress())
      OS << " + " << (Address - Obj->getAddress());
  } else {
    OS << "dangling";
  }
  OS << "]";
}

AnyValue Pointer::null(unsigned BitWidth) {
  return AnyValue(Pointer(nullptr, APInt::getZero(BitWidth)));
}

void AnyValue::print(raw_ostream &OS) const {
  switch (Kind) {
  case StorageKind::Integer:
    if (IntVal.getBitWidth() == 1) {
      OS << (IntVal.getBoolValue() ? "T" : "F");
      break;
    }
    OS << "i" << IntVal.getBitWidth() << ' ' << IntVal;
    break;
  case StorageKind::Float:
    OS << FloatVal;
    break;
  case StorageKind::Pointer:
    PtrVal.print(OS);
    break;
  case StorageKind::Poison:
    OS << "poison";
    break;
  case StorageKind::None:
    OS << "none";
    break;
  case StorageKind::Aggregate:
    OS << "{ ";
    for (size_t I = 0, E = AggVal.size(); I != E; ++I) {
      if (I != 0)
        OS << ", ";
      AggVal[I].print(OS);
    }
    OS << " }";
    break;
  }
}

void AnyValue::destroy() {
  switch (Kind) {
  case StorageKind::Integer:
    IntVal.~APInt();
    break;
  case StorageKind::Float:
    FloatVal.~APFloat();
    break;
  case StorageKind::Pointer:
    PtrVal.~Pointer();
    break;
  case StorageKind::Poison:
  case StorageKind::None:
    break;
  case StorageKind::Aggregate:
    AggVal.~vector();
    break;
  }
}

AnyValue::AnyValue(const AnyValue &Other) : Kind(Other.Kind) {
  switch (Other.Kind) {
  case StorageKind::Integer:
    new (&IntVal) APInt(Other.IntVal);
    break;
  case StorageKind::Float:
    new (&FloatVal) APFloat(Other.FloatVal);
    break;
  case StorageKind::Pointer:
    new (&PtrVal) Pointer(Other.PtrVal);
    break;
  case StorageKind::Poison:
  case StorageKind::None:
    break;
  case StorageKind::Aggregate:
    new (&AggVal) std::vector<AnyValue>(Other.AggVal);
    break;
  }
}
AnyValue::AnyValue(AnyValue &&Other) : Kind(Other.Kind) {
  switch (Other.Kind) {
  case StorageKind::Integer:
    new (&IntVal) APInt(std::move(Other.IntVal));
    break;
  case StorageKind::Float:
    new (&FloatVal) APFloat(std::move(Other.FloatVal));
    break;
  case StorageKind::Pointer:
    new (&PtrVal) Pointer(std::move(Other.PtrVal));
    break;
  case StorageKind::Poison:
  case StorageKind::None:
    break;
  case StorageKind::Aggregate:
    new (&AggVal) std::vector<AnyValue>(std::move(Other.AggVal));
    break;
  }
}

AnyValue &AnyValue::operator=(const AnyValue &Other) {
  if (&Other == this)
    return *this;

  destroy();
  Kind = Other.Kind;
  switch (Other.Kind) {
  case StorageKind::Integer:
    new (&IntVal) APInt(Other.IntVal);
    break;
  case StorageKind::Float:
    new (&FloatVal) APFloat(Other.FloatVal);
    break;
  case StorageKind::Pointer:
    new (&PtrVal) Pointer(Other.PtrVal);
    break;
  case StorageKind::Poison:
  case StorageKind::None:
    break;
  case StorageKind::Aggregate:
    new (&AggVal) std::vector<AnyValue>(Other.AggVal);
    break;
  }

  return *this;
}
AnyValue &AnyValue::operator=(AnyValue &&Other) {
  if (&Other == this)
    return *this;
  destroy();
  Kind = Other.Kind;
  switch (Other.Kind) {
  case StorageKind::Integer:
    new (&IntVal) APInt(std::move(Other.IntVal));
    break;
  case StorageKind::Float:
    new (&FloatVal) APFloat(std::move(Other.FloatVal));
    break;
  case StorageKind::Pointer:
    new (&PtrVal) Pointer(std::move(Other.PtrVal));
    break;
  case StorageKind::Poison:
  case StorageKind::None:
    break;
  case StorageKind::Aggregate:
    new (&AggVal) std::vector<AnyValue>(std::move(Other.AggVal));
    break;
  }

  return *this;
}

AnyValue AnyValue::getPoisonValue(Context &Ctx, Type *Ty) {
  if (Ty->isFloatingPointTy() || Ty->isIntegerTy() || Ty->isPointerTy())
    return AnyValue::poison();
  if (auto *VecTy = dyn_cast<VectorType>(Ty)) {
    uint32_t NumElements = Ctx.getEVL(VecTy->getElementCount());
    return AnyValue(std::vector<AnyValue>(NumElements, AnyValue::poison()));
  }
  if (auto *ArrTy = dyn_cast<ArrayType>(Ty)) {
    uint64_t NumElements = ArrTy->getNumElements();
    return AnyValue(std::vector<AnyValue>(
        NumElements, getPoisonValue(Ctx, ArrTy->getElementType())));
  }
  if (auto *StructTy = dyn_cast<StructType>(Ty)) {
    std::vector<AnyValue> Elements;
    Elements.reserve(StructTy->getNumElements());
    for (uint32_t I = 0, E = StructTy->getNumElements(); I != E; ++I)
      Elements.push_back(getPoisonValue(Ctx, StructTy->getElementType(I)));
    return AnyValue(std::move(Elements));
  }
  llvm_unreachable("Unsupported type");
}
AnyValue AnyValue::getNullValue(Context &Ctx, Type *Ty) {
  if (Ty->isIntegerTy())
    return AnyValue(APInt::getZero(Ty->getIntegerBitWidth()));
  if (Ty->isFloatingPointTy())
    return AnyValue(APFloat::getZero(Ty->getFltSemantics()));
  if (Ty->isPointerTy())
    return Pointer::null(
        Ctx.getDataLayout().getPointerSizeInBits(Ty->getPointerAddressSpace()));
  if (auto *VecTy = dyn_cast<VectorType>(Ty)) {
    uint32_t NumElements = Ctx.getEVL(VecTy->getElementCount());
    return AnyValue(std::vector<AnyValue>(
        NumElements, getNullValue(Ctx, VecTy->getElementType())));
  }
  if (auto *ArrTy = dyn_cast<ArrayType>(Ty)) {
    uint64_t NumElements = ArrTy->getNumElements();
    return AnyValue(std::vector<AnyValue>(
        NumElements, getNullValue(Ctx, ArrTy->getElementType())));
  }
  if (auto *StructTy = dyn_cast<StructType>(Ty)) {
    std::vector<AnyValue> Elements;
    Elements.reserve(StructTy->getNumElements());
    for (uint32_t I = 0, E = StructTy->getNumElements(); I != E; ++I)
      Elements.push_back(getNullValue(Ctx, StructTy->getElementType(I)));
    return AnyValue(std::move(Elements));
  }
  llvm_unreachable("Unsupported type");
}

AnyValue AnyValue::getVectorSplat(const AnyValue &Scalar, size_t NumElements) {
  assert(!Scalar.isAggregate() && !Scalar.isNone() && "Expect a scalar value");
  return AnyValue(std::vector<AnyValue>(NumElements, Scalar));
}

} // namespace llvm::ubi

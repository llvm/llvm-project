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

IntrusiveRefCntPtr<Provenance> Provenance::nullary() {
  static IntrusiveRefCntPtr<Provenance> Instance =
      makeIntrusiveRefCnt<Provenance>(nullptr);
  return Instance;
}

IntrusiveRefCntPtr<Provenance>
Provenance::getWithKnownMemoryObject(MemoryObject &KnownObj) {
  assert(!Obj && Wildcard && "The memory object has been determined.");
  auto Res = makeIntrusiveRefCnt<Provenance>(*this);
  Res->Obj = &KnownObj;
  Res->Tag = APInt();
  return Res;
}

void Pointer::print(raw_ostream &OS) const {
  SmallString<32> AddrStr;
  Address.toStringUnsigned(AddrStr, 16);
  OS << "ptr 0x" << AddrStr << " [";
  if (MemoryObject *Obj = Prov->getMemoryObject()) {
    if (Obj->isIRGlobalValue())
      OS << "@";
    OS << Obj->getName();
    if (Address != Obj->getAddress())
      OS << " + " << (Address - Obj->getAddress());
    MemoryObjectState State = Obj->getState();
    if (State != MemoryObjectState::Alive)
      OS << (State == MemoryObjectState::Dead ? " (dead)" : " (dangling)");
  } else {
    OS << (Prov->isWildcard() ? "wildcard" : "nullary");
  }
  // TODO: print provenance
  OS << "]";
}

AnyValue Pointer::null(unsigned AS, const DataLayout &DL) {
  return AnyValue(Pointer(Provenance::nullary(), DL.getNullPtrValue(AS)));
}

bool Pointer::isNullPtr(unsigned AS, const DataLayout &DL) const {
  return Address == DL.getNullPtrValue(AS);
}

void AnyValue::print(Context &Ctx, raw_ostream &OS) const {
  switch (Kind) {
  case StorageKind::Integer:
    if (IntVal.getBitWidth() == 1) {
      OS << (IntVal.getBoolValue() ? "T" : "F");
      break;
    }
    OS << "i" << IntVal.getBitWidth() << ' ' << IntVal;
    break;
  case StorageKind::Float: {
    switch (APFloat::SemanticsToEnum(FloatVal.getSemantics())) {
    default:
      llvm_unreachable("invalid fltSemantics");
    case APFloatBase::S_IEEEhalf:
      OS << "half ";
      break;
    case APFloatBase::S_BFloat:
      OS << "bfloat ";
      break;
    case APFloatBase::S_IEEEsingle:
      OS << "float ";
      break;
    case APFloatBase::S_IEEEdouble:
      OS << "double ";
      break;
    case APFloatBase::S_x87DoubleExtended:
      OS << "x86_fp80 ";
      break;
    case APFloatBase::S_IEEEquad:
      OS << "fp128 ";
      break;
    case APFloatBase::S_PPCDoubleDouble:
      OS << "ppc_fp128 ";
      break;
    }
    // We cannot reuse Value::print due to lack of LLVMContext here.
    // Similar to writeAPFloatInternal, output the FP constant value in
    // exponential notation if it is lossless, otherwise output it in
    // hexadecimal notation.
    SmallString<16> StrVal;
    if (FloatVal.toStringRoundTrip(StrVal, /*FormatPrecision=*/6,
                                   /*FormatMaxPadding=*/0,
                                   /*TruncateZero=*/false)) {
      OS << StrVal;
    } else {
      StrVal.clear();
      APInt Bits = FloatVal.bitcastToAPInt();
      Bits.toStringUnsigned(StrVal, 16);
      size_t MaxDigits = divideCeil(Bits.getBitWidth(), 4);
      OS << "0x";
      for (size_t Digits = StrVal.size(); Digits != MaxDigits; ++Digits)
        OS << '0';
      OS << StrVal;
    }
    break;
  }
  case StorageKind::Pointer:
    PtrVal.print(OS);
    break;
  case StorageKind::Byte:
    ByteVal.print(Ctx, OS);
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
      AggVal[I].print(Ctx, OS);
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
  case StorageKind::Byte:
    ByteVal.~ByteValue();
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
  case StorageKind::Byte:
    new (&ByteVal) ByteValue(Other.ByteVal);
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
  case StorageKind::Byte:
    new (&ByteVal) ByteValue(std::move(Other.ByteVal));
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
  case StorageKind::Byte:
    new (&ByteVal) ByteValue(Other.ByteVal);
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
  case StorageKind::Byte:
    new (&ByteVal) ByteValue(std::move(Other.ByteVal));
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
  if (Ty->isByteTy())
    return ByteValue::poison(Ty->getByteBitWidth(),
                             Ctx.getDataLayout().isLittleEndian());
  if (auto *VecTy = dyn_cast<VectorType>(Ty)) {
    uint32_t NumElements = Ctx.getEVL(VecTy->getElementCount());
    return AnyValue(std::vector<AnyValue>(
        NumElements, getPoisonValue(Ctx, VecTy->getScalarType())));
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
    return Pointer::null(Ty->getPointerAddressSpace(), Ctx.getDataLayout());
  if (Ty->isByteTy())
    return ByteValue::zero(Ty->getByteBitWidth(),
                           Ctx.getDataLayout().isLittleEndian());
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

ByteValue::ByteValue(const APInt &V, bool IsLittleEndian)
    : BitWidth(V.getBitWidth()), IsLittleEndian(IsLittleEndian) {
  Val.resize(divideCeil(BitWidth, 8));
  MutableBytesView View(Val, IsLittleEndian);
  for (uint32_t I = 0; I < BitWidth; I += 8)
    View[I / 8] = Byte::concrete(static_cast<uint8_t>(
        V.extractBitsAsZExtValue(std::min(BitWidth - I, 8U), I)));
}
ByteValue ByteValue::zero(uint32_t BitWidth, bool IsLittleEndian) {
  return ByteValue(
      BitWidth, std::vector<Byte>(divideCeil(BitWidth, 8), Byte::concrete(0)),
      IsLittleEndian);
}

ByteValue ByteValue::poison(uint32_t BitWidth, bool IsLittleEndian) {
  return ByteValue(BitWidth,
                   std::vector<Byte>(divideCeil(BitWidth, 8), Byte::poison()),
                   IsLittleEndian, /*ImplicitClearHighBits=*/true);
}

void ByteValue::print(Context &Ctx, raw_ostream &OS) const {
  OS << 'b' << BitWidth << ' ';

  auto PrintByte = [&](const Byte &V) {
    bool IsFullByte = (BitWidth & 7) == 0 ||
                      (IsLittleEndian ? &Val.back() : &Val.front()) != &V;
    // Try to print a byte in short form
    if (IsFullByte && V.ConcreteMask == 255 && V.TagMask == 0) {
      // Concrete value without provenance.
      OS << "0x" << hexdigit(V.Value >> 4) << hexdigit(V.Value & 15);
    } else if (IsFullByte && V.ConcreteMask == 0) {
      assert(V.Value == 0 && "Byte values don't contain undef bits.");
      // Poison bytes.
      OS << "0x!!";
    } else {
      uint32_t BitEnd = IsFullByte ? 8 : BitWidth & 7;
      for (uint32_t I = 0; I != BitEnd; ++I) {
        uint32_t Mask = 1U << (BitEnd - 1 - I);
        if (V.ConcreteMask & Mask)
          OS << (V.Value & Mask ? '1' : '0');
        else {
          assert((V.Value & Mask) == 0 &&
                 "Byte values don't contain undef bits.");
          OS << '!';
        }
      }
      assert((V.ConcreteMask & V.TagMask) == V.TagMask);
      if (V.TagMask) {
        // Print tags if available.
        OS << '(';
        for (uint32_t I = 0; I != BitEnd; ++I) {
          uint32_t Mask = 1U << (BitEnd - 1 - I);
          if (V.TagMask & Mask)
            OS << (V.TagValue & Mask ? '1' : '0');
          else
            OS << '!';
        }
        OS << ')';
      }
    }
    OS << ' ';
  };

  auto &DL = Ctx.getDataLayout();
  unsigned PtrWidthForAS0 = DL.getPointerSizeInBits(0);
  Type *PtrTy = PointerType::getUnqual(Ctx.getContext());

  if (PtrWidthForAS0 % 8 == 0 && BitWidth % PtrWidthForAS0 == 0) {
    // Try to treat the bytes value as an array of pointers in address space 0.
    unsigned PtrSize = PtrWidthForAS0 / 8;
    for (size_t I = 0, E = Val.size(); I != E; I += PtrSize) {
      ArrayRef<Byte> Slice = ArrayRef(Val).slice(I, PtrSize);
      if (all_of(Slice, [](const Byte &V) {
            assert((V.ConcreteMask & V.TagMask) == V.TagMask);
            return V.TagMask == 255;
          })) {
        AnyValue Res = Ctx.fromBytes(Slice, PtrTy);
        if (Res.isPointer()) {
          Res.asPointer().print(OS);
          OS << ' ';
          continue;
        }
      }

      // Otherwise, fallback into bytes array
      for (size_t J = 0; J != PtrSize; ++J)
        PrintByte(Val[I + J]);
    }
  } else {
    for (const Byte &V : Val)
      PrintByte(V);
  }
}

} // namespace llvm::ubi

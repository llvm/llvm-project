//===- SystemZ.cpp - SystemZ ABI Implementation ---------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/ABI/FunctionInfo.h"
#include "llvm/ABI/TargetInfo.h"
#include "llvm/ABI/Types.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/MathExtras.h"
#include <optional>

namespace llvm::abi {

class SystemZTargetInfo : public TargetInfo {
public:
  SystemZTargetInfo(TypeBuilder &TB, const SystemZABIOptions &Opts)
      : TargetInfo(TB), Opts(Opts) {}

  const ABICompatInfo &getABICompatInfo() const override {
    return Opts.CompatInfo;
  }

  void computeInfo(FunctionInfo &FI) const override {
    if (!maybeCommonClassifyReturnType(FI))
      FI.getReturnInfo() = classifyReturnType(FI.getReturnType());
    for (auto &I : FI.arguments())
      I.Info = classifyArgumentType(I.ABIType);
  }

  Align getMaxIntegerAlign() const override { return Align(8); }

private:
  SystemZABIOptions Opts;

  bool isPromotableIntegerTypeForABI(const Type *Ty) const;
  bool isCompoundType(const Type *Ty) const;
  bool isVectorArgumentType(const Type *Ty) const;
  const Type *getFPArgumentType(const Type *Ty, uint64_t Size) const;
  const Type *getSingleElementType(const Type *Ty) const;
  ArgInfo classifyReturnType(const Type *RetTy) const;
  ArgInfo classifyArgumentType(const Type *Ty) const;
};

bool SystemZTargetInfo::isPromotableIntegerTypeForABI(const Type *Ty) const {
  // Frontends represent enums using their underlying integer type.
  const auto *IT = dyn_cast<IntegerType>(Ty);
  if (!IT)
    return false;
  uint64_t Width = IT->getSizeInBits().getFixedValue();
  return isPromotableInteger(IT) || (IT->isBitInt() ? Width < 64 : Width == 32);
}

bool SystemZTargetInfo::isCompoundType(const Type *Ty) const {
  // Clang's ABI aggregate predicate distinguishes member-function pointers
  // from scalar data-member pointers.
  const Type *ValueTy = Ty;
  while (const auto *AT = dyn_cast<AtomicType>(ValueTy))
    ValueTy = AT->getValueType();
  if (const auto *MPT = dyn_cast<MemberPointerType>(ValueTy))
    return ValueTy == Ty && MPT->isFunctionPointer();
  return Ty->isComplex() || Ty->isVector() || isAggregateTypeForABI(Ty);
}

bool SystemZTargetInfo::isVectorArgumentType(const Type *Ty) const {
  const auto *VT = dyn_cast<VectorType>(Ty);
  return Opts.HasVector && VT && VT->isFixedLength() &&
         VT->getABISizeInBits() <= 128;
}

const Type *SystemZTargetInfo::getFPArgumentType(const Type *Ty,
                                                 uint64_t Size) const {
  if (Opts.IsSoftFloatABI)
    return nullptr;

  const auto *FT = dyn_cast<FloatType>(Ty);
  if (!FT)
    return nullptr;

  // An over-aligned single-element struct uses the FP type matching its size.
  if (FT->getSemantics() == &APFloat::IEEEhalf()) {
    if (Size == 16)
      return TB.getFloatType(APFloat::IEEEhalf(), Align(2));
    if (Size == 32)
      return TB.getFloatType(APFloat::IEEEsingle(), Align(4));
    return TB.getFloatType(APFloat::IEEEdouble(), Align(8));
  }
  if (FT->getSemantics() == &APFloat::IEEEsingle()) {
    if (Size == 32)
      return TB.getFloatType(APFloat::IEEEsingle(), Align(4));
    return TB.getFloatType(APFloat::IEEEdouble(), Align(8));
  }
  if (FT->getSemantics() == &APFloat::IEEEdouble())
    return TB.getFloatType(APFloat::IEEEdouble(), Align(8));
  return nullptr;
}

const Type *SystemZTargetInfo::getSingleElementType(const Type *Ty) const {
  const auto *RT = dyn_cast<RecordType>(Ty);
  if (!RT || RT->isUnion())
    return Ty;

  // The shared isSingleElementStruct helper differs: it unwraps arrays,
  // ignores empty fields, and rejects trailing padding, which SystemZ allows.
  const Type *Found = nullptr;

  // Check the bases first. Empty bases don't affect things either way.
  for (ArrayRef<FieldInfo> Bases :
       {RT->getBaseClasses(), RT->getVirtualBaseClasses()}) {
    for (const FieldInfo &Base : Bases) {
      if (Base.FieldType->isEmptyRecord())
        continue;
      if (Found)
        return Ty;
      Found = getSingleElementType(Base.FieldType);
    }
  }

  for (const FieldInfo &Field : RT->getFields()) {
    // Like isSingleElementRecord(), ignore C++20 empty data members.
    if (Field.HasNoUniqueAddress && Field.FieldType->isEmptyRecord())
      continue;
    // Empty fields, array fields, and anonymous bitfields count as fields.
    if (Found)
      return Ty;
    Found = getSingleElementType(Field.FieldType);
  }
  return Found ? Found : Ty;
}

ArgInfo SystemZTargetInfo::classifyReturnType(const Type *RetTy) const {
  if (RetTy->isVoid())
    return ArgInfo::getIgnore();
  if (isVectorArgumentType(RetTy))
    return ArgInfo::getDirect();
  if (isCompoundType(RetTy) || RetTy->getABISizeInBits() > 64)
    return getNaturalAlignIndirect(RetTy, getAllocaAddrSpace());
  return isPromotableIntegerTypeForABI(RetTy) ? ArgInfo::getExtend(RetTy)
                                              : ArgInfo::getDirect();
}

ArgInfo SystemZTargetInfo::classifyArgumentType(const Type *Ty) const {
  Ty = useFirstFieldIfTransparentUnion(Ty);

  if (RecordArgABI RAA = getRecordArgABI(Ty))
    return getNaturalAlignIndirect(Ty, getAllocaAddrSpace(),
                                   RAA == RAA_DirectInMemory);

  if (isPromotableIntegerTypeForABI(Ty))
    return ArgInfo::getExtend(Ty);

  uint64_t Size = Ty->getABISizeInBits();
  const Type *SingleElementTy = getSingleElementType(Ty);
  // Vector-like records cannot have padding.
  if (isVectorArgumentType(SingleElementTy) &&
      SingleElementTy->getABISizeInBits() == Size)
    return ArgInfo::getDirect(SingleElementTy);

  if (Size != 8 && Size != 16 && Size != 32 && Size != 64)
    return getNaturalAlignIndirect(Ty, getAllocaAddrSpace(), /*ByVal=*/false);

  if (const auto *RT = dyn_cast<RecordType>(Ty)) {
    if (RT->hasFlexibleArrayMember())
      return getNaturalAlignIndirect(Ty, getAllocaAddrSpace(), /*ByVal=*/false);

    if (const Type *FPArgTy = getFPArgumentType(SingleElementTy, Size))
      return ArgInfo::getDirect(FPArgTy);

    const Type *PassTy =
        TB.getIntegerType(Size, Align(Size / 8), /*Signed=*/false);
    if (Size <= 32) {
      // Clang's getNoExtend: Extend with neither signext nor zeroext set.
      return ArgInfo::getExtend(PassTy).setZeroExt(false);
    }
    return ArgInfo::getDirect(PassTy);
  }

  if (isCompoundType(Ty))
    return getNaturalAlignIndirect(Ty, getAllocaAddrSpace(), /*ByVal=*/false);
  return ArgInfo::getDirect();
}

class SystemZXPLinkTargetInfo : public TargetInfo {
public:
  SystemZXPLinkTargetInfo(TypeBuilder &TB, const SystemZABIOptions &Opts)
      : TargetInfo(TB), Opts(Opts) {}

  const ABICompatInfo &getABICompatInfo() const override {
    return Opts.CompatInfo;
  }

  void computeInfo(FunctionInfo &FI) const override {
    if (!maybeCommonClassifyReturnType(FI))
      FI.getReturnInfo() =
          classifyReturnType(FI.getReturnType(), FI.getCallingConvention());

    unsigned NumRequiredArgs = FI.getNumRequiredArgs();
    unsigned ArgNo = 0;
    for (auto &I : FI.arguments()) {
      bool IsNamedArg = ArgNo < NumRequiredArgs;
      I.Info = classifyArgumentType(I.ABIType, IsNamedArg,
                                    FI.getCallingConvention());
      ++ArgNo;
    }
  }

  Align getMaxIntegerAlign() const override { return Align(8); }

private:
  static constexpr unsigned GPRBits = 64;
  SystemZABIOptions Opts;

  bool isPromotableIntegerTypeForABI(const Type *Ty) const;
  bool isCompoundType(const Type *Ty) const;
  bool isVectorArgumentType(const Type *Ty) const;
  const Type *getSingleElementType(const Type *Ty) const;
  std::optional<const Type *> getFPTypeOfComplexLikeType(const Type *Ty) const;
  ArgInfo classifyReturnType(const Type *RetTy, CallingConv::ID CallConv) const;
  ArgInfo classifyArgumentType(const Type *Ty, bool IsNamedArg,
                               CallingConv::ID CallConv) const;
};

bool SystemZXPLinkTargetInfo::isPromotableIntegerTypeForABI(
    const Type *Ty) const {
  const auto *IT = dyn_cast<IntegerType>(Ty);
  if (!IT)
    return false;
  uint64_t Width = IT->getSizeInBits().getFixedValue();
  // Besides everything narrower than 64 bits, XPLINK also extends long and
  // unsigned long, but not long long.
  return isPromotableInteger(IT) || Width < 64 || IT->isLong();
}

bool SystemZXPLinkTargetInfo::isCompoundType(const Type *Ty) const {
  return Ty->isComplex() || isAggregateTypeForABI(Ty);
}

bool SystemZXPLinkTargetInfo::isVectorArgumentType(const Type *Ty) const {
  const auto *VT = dyn_cast<VectorType>(Ty);
  return Opts.HasVector && VT && VT->isFixedLength() &&
         VT->getABISizeInBits() <= 128;
}

// Types are not uniqued, so two equal types can be distinct objects. This
// plays the role of the canonical type comparison Clang uses. Records are
// compared by identity.
static bool isSameType(const Type *A, const Type *B) {
  if (A == B)
    return true;
  if (A->getKind() != B->getKind())
    return false;

  switch (A->getKind()) {
  case TypeKind::Void:
    return true;
  case TypeKind::Atomic:
    return isSameType(cast<AtomicType>(A)->getValueType(),
                      cast<AtomicType>(B)->getValueType());
  case TypeKind::Complex:
    return isSameType(cast<ComplexType>(A)->getElementType(),
                      cast<ComplexType>(B)->getElementType());
  case TypeKind::Integer: {
    const auto *IA = cast<IntegerType>(A);
    const auto *IB = cast<IntegerType>(B);
    return IA->getSizeInBits() == IB->getSizeInBits() &&
           IA->isSigned() == IB->isSigned() &&
           IA->isBitInt() == IB->isBitInt() && IA->isLong() == IB->isLong();
  }
  case TypeKind::Float:
    return cast<FloatType>(A)->getSemantics() ==
           cast<FloatType>(B)->getSemantics();
  case TypeKind::Pointer:
    return cast<PointerType>(A)->getAddrSpace() ==
               cast<PointerType>(B)->getAddrSpace() &&
           A->getSizeInBits() == B->getSizeInBits();
  case TypeKind::MemberPointer: {
    const auto *MA = cast<MemberPointerType>(A);
    const auto *MB = cast<MemberPointerType>(B);
    return MA->isFunctionPointer() == MB->isFunctionPointer() &&
           MA->getAddrSpace() == MB->getAddrSpace() &&
           A->getSizeInBits() == B->getSizeInBits();
  }
  case TypeKind::Array: {
    const auto *AA = cast<ArrayType>(A);
    const auto *AB = cast<ArrayType>(B);
    return AA->getNumElements() == AB->getNumElements() &&
           AA->isMatrixType() == AB->isMatrixType() &&
           isSameType(AA->getElementType(), AB->getElementType());
  }
  case TypeKind::Vector: {
    const auto *VA = cast<VectorType>(A);
    const auto *VB = cast<VectorType>(B);
    return VA->getNumElements() == VB->getNumElements() &&
           VA->getVectorKind() == VB->getVectorKind() &&
           isSameType(VA->getElementType(), VB->getElementType());
  }
  case TypeKind::Tuple: {
    const auto *TA = cast<TupleType>(A);
    const auto *TB = cast<TupleType>(B);
    return TA->getNumVectors() == TB->getNumVectors() &&
           isSameType(TA->getVectorType(), TB->getVectorType());
  }
  case TypeKind::Record:
    return false;
  }
  llvm_unreachable("unknown type kind");
}

const Type *
SystemZXPLinkTargetInfo::getSingleElementType(const Type *Ty) const {
  const auto *RT = dyn_cast<RecordType>(Ty);
  if (!RT)
    return Ty;

  // Check if the union only consists of a single type, and return that type.
  if (RT->isUnion()) {
    const Type *Found = nullptr;
    for (const FieldInfo &Field : RT->getFields()) {
      const Type *FieldTy = getSingleElementType(Field.FieldType);
      if (!Found)
        Found = FieldTy;
      else if (!isSameType(Found, FieldTy))
        return Ty;
    }
    return Found ? Found : Ty;
  }

  const Type *Found = nullptr;

  // Check the bases first. Empty bases don't affect things either way.
  for (ArrayRef<FieldInfo> Bases :
       {RT->getBaseClasses(), RT->getVirtualBaseClasses()}) {
    for (const FieldInfo &Base : Bases) {
      if (Base.FieldType->isEmptyRecord())
        continue;
      if (Found)
        return Ty;
      Found = getSingleElementType(Base.FieldType);
    }
  }

  // Count every field, including empty [[no_unique_address]] members.
  for (const FieldInfo &Field : RT->getFields()) {
    if (Found)
      return Ty;
    Found = getSingleElementType(Field.FieldType);
  }
  return Found ? Found : Ty;
}

std::optional<const Type *>
SystemZXPLinkTargetInfo::getFPTypeOfComplexLikeType(const Type *Ty) const {
  const auto *RT = dyn_cast<RecordType>(Ty);
  if (!RT || RT->isUnion())
    return std::nullopt;

  // Check for non-empty base classes.
  for (ArrayRef<FieldInfo> Bases :
       {RT->getBaseClasses(), RT->getVirtualBaseClasses()})
    for (const FieldInfo &Base : Bases)
      if (!Base.FieldType->isEmptyRecord())
        return std::nullopt;

  int Count = 0;
  const FloatType *ElemFT = nullptr;
  const Type *RetTy = nullptr;
  for (const FieldInfo &Field : RT->getFields()) {
    if (Count >= 2)
      return std::nullopt;

    const Type *SingleTy = getSingleElementType(Field.FieldType);
    const auto *FT = dyn_cast<FloatType>(SingleTy);
    if (!FT)
      return std::nullopt;

    // Only float, double and long double qualify, not half or bfloat.
    const fltSemantics *Sem = FT->getSemantics();
    if (Sem != &APFloat::IEEEsingle() && Sem != &APFloat::IEEEdouble() &&
        Sem != &APFloat::IEEEquad())
      return std::nullopt;

    if (Count == 0) {
      ElemFT = FT;
      RetTy = SingleTy;
    } else if (ElemFT->getSemantics() != FT->getSemantics()) {
      return std::nullopt;
    }
    Count++;
  }

  if (Count == 2 && RetTy) {
    if (Ty->getABISizeInBits() == 2 * RetTy->getABISizeInBits())
      return RetTy;
  }
  return std::nullopt;
}

ArgInfo
SystemZXPLinkTargetInfo::classifyReturnType(const Type *RetTy,
                                            CallingConv::ID CallConv) const {
  if (RetTy->isVoid())
    return ArgInfo::getIgnore();

  RetTy = useFirstFieldIfTransparentUnion(RetTy);

  if (CallConv != CallingConv::C &&
      (isAggregateTypeForABI(RetTy) || RetTy->isComplex()))
    return getNaturalAlignIndirect(RetTy, getAllocaAddrSpace());

  if (isVectorArgumentType(RetTy))
    return ArgInfo::getDirect();

  if (RetTy->isComplex())
    return ArgInfo::getDirect();

  if (isa<RecordType>(RetTy)) {
    if (getFPTypeOfComplexLikeType(RetTy))
      return ArgInfo::getDirect();
  }

  if (isAggregateTypeForABI(RetTy)) {
    uint64_t AggregateTypeSize = RetTy->getABISizeInBits();
    if (AggregateTypeSize <= 3 * GPRBits) {
      uint64_t NumElements =
          AggregateTypeSize / GPRBits + (AggregateTypeSize % GPRBits != 0);
      const Type *CoerceTy =
          TB.getIntegerType(GPRBits, Align(8), /*Signed=*/false);
      CoerceTy = TB.getArrayType(CoerceTy, NumElements, NumElements * GPRBits);
      return ArgInfo::getDirect(CoerceTy).setInReg();
    }
    return getNaturalAlignIndirect(RetTy, getAllocaAddrSpace());
  }

  return isPromotableIntegerTypeForABI(RetTy) ? ArgInfo::getExtend(RetTy)
                                              : ArgInfo::getDirect();
}

ArgInfo
SystemZXPLinkTargetInfo::classifyArgumentType(const Type *Ty, bool IsNamedArg,
                                              CallingConv::ID CallConv) const {
  Ty = useFirstFieldIfTransparentUnion(Ty);

  if (RecordArgABI RAA = getRecordArgABI(Ty))
    return getNaturalAlignIndirect(Ty, getAllocaAddrSpace(),
                                   RAA == RAA_DirectInMemory);

  if (isPromotableIntegerTypeForABI(Ty))
    return ArgInfo::getExtend(Ty);

  if (CallConv != CallingConv::C && isCompoundType(Ty))
    return getNaturalAlignIndirect(Ty, getAllocaAddrSpace(), /*ByVal=*/false);

  auto CompTy = getFPTypeOfComplexLikeType(Ty);
  if (IsNamedArg) {
    if (Ty->isComplex()) {
      auto AI = ArgInfo::getDirect();
      AI.setCanBeFlattened(false);
      return AI;
    }

    if (CompTy.has_value()) {
      const Type *CoerceTy =
          getStructOfTypes({*CompTy, *CompTy}, /*Packed=*/false);
      auto AI = ArgInfo::getDirect(CoerceTy);
      AI.setCanBeFlattened(false);
      return AI;
    }
  }

  if (isVectorArgumentType(Ty))
    return ArgInfo::getDirect();

  if (isa<RecordType>(Ty) || Ty->isComplex() || CompTy.has_value()) {
    uint64_t Bits = Ty->getABISizeInBits();
    const Type *CoerceTy;
    if (Bits <= GPRBits) {
      CoerceTy = TB.getIntegerType(GPRBits, Align(8), /*Signed=*/false);
    } else {
      uint64_t NumRegs = llvm::alignTo(Bits, GPRBits) / GPRBits;
      const Type *RegTy =
          TB.getIntegerType(GPRBits, Align(8), /*Signed=*/false);
      CoerceTy = TB.getArrayType(RegTy, NumRegs, NumRegs * GPRBits);
    }
    return ArgInfo::getDirect(CoerceTy);
  }

  if (isCompoundType(Ty))
    return getNaturalAlignIndirect(Ty, getAllocaAddrSpace(), /*ByVal=*/false);

  return ArgInfo::getDirect();
}

std::unique_ptr<TargetInfo>
createSystemZTargetInfo(TypeBuilder &TB, const SystemZABIOptions &Opts) {
  if (Opts.Kind == SystemZABIKind::XPLINK)
    return std::make_unique<SystemZXPLinkTargetInfo>(TB, Opts);
  return std::make_unique<SystemZTargetInfo>(TB, Opts);
}

} // namespace llvm::abi

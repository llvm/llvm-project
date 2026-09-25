//===- TargetInfo.cpp - Target ABI information ----------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/ABI/TargetInfo.h"
#include "llvm/Support/Casting.h"
#include <algorithm>
#include <cstdint>

using namespace llvm::abi;
using llvm::dyn_cast;

bool TargetInfo::isAggregateTypeForABI(const Type *Ty) const {
  // Atomic values use the evaluation kind of their underlying value type.
  if (const auto *AT = dyn_cast<AtomicType>(Ty))
    return isAggregateTypeForABI(AT->getValueType());

  // Check for fundamental scalar types.
  if (Ty->isInteger() || Ty->isFloat() || Ty->isPointer() || Ty->isVector())
    return false;

  // A matrix type is modeled as an array but lowers to a single flattened
  // vector and has scalar evaluation kind in classic CodeGen, so it is not an
  // aggregate for ABI purposes.
  if (const auto *AT = dyn_cast<ArrayType>(Ty))
    if (AT->isMatrixType())
      return false;

  // Everything else is treated as aggregate.
  return true;
}

bool TargetInfo::isPromotableInteger(const IntegerType *IT) const {
  // TODO: The threshold should be the target's int size rather than a
  // hardcoded 32.
  unsigned BitWidth = IT->getSizeInBits().getFixedValue();
  return BitWidth < 32;
}

ArgInfo TargetInfo::getNaturalAlignIndirect(const Type *Ty, unsigned AddrSpace,
                                            bool ByVal) const {
  return ArgInfo::getIndirect(Ty->getAlignment(), ByVal, AddrSpace);
}

RecordArgABI TargetInfo::getRecordArgABI(const RecordType *RT) const {
  if (RT && !RT->canPassInRegisters())
    return RAA_Indirect;
  return RAA_Default;
}

RecordArgABI TargetInfo::getRecordArgABI(const Type *Ty) const {
  // TODO: When Microsoft ABI is supported, CXX records may need different
  // handling here (see MicrosoftCXXABI::getRecordArgABI in Clang).
  const RecordType *RT = dyn_cast<RecordType>(Ty);
  if (!RT)
    return RAA_Default;
  return getRecordArgABI(RT);
}

const Type *TargetInfo::useFirstFieldIfTransparentUnion(const Type *Ty) const {
  if (const auto *RT = dyn_cast<RecordType>(Ty)) {
    if (RT->isUnion() && RT->isTransparentUnion()) {
      auto Fields = RT->getFields();
      assert(!Fields.empty() && "transparent union cannot be empty");
      return Fields.front().FieldType;
    }
  }
  return Ty;
}

const Type *TargetInfo::isSingleElementStruct(const Type *Ty) const {
  const auto *RT = dyn_cast<RecordType>(Ty);
  if (!RT)
    return nullptr;

  if (RT->hasFlexibleArrayMember())
    return nullptr;

  const Type *Found = nullptr;

  for (const auto &Base : RT->getBaseClasses()) {
    const Type *BaseTy = Base.FieldType;
    const auto *BaseRT = dyn_cast<RecordType>(BaseTy);

    if (!BaseRT || BaseRT->isEmpty())
      continue;

    const Type *Elem = isSingleElementStruct(BaseTy);
    if (!Elem || Found)
      return nullptr;
    Found = Elem;
  }

  for (const auto &FI : RT->getFields()) {
    if (FI.isEmpty())
      continue;

    const Type *FTy = FI.FieldType;

    // Treat single element arrays as the element.
    while (const auto *AT = dyn_cast<ArrayType>(FTy)) {
      if (AT->getNumElements() != 1)
        break;
      FTy = AT->getElementType();
    }

    const Type *Elem;
    if (!isAggregateTypeForABI(FTy))
      Elem = FTy;
    else
      Elem = isSingleElementStruct(FTy);
    if (!Elem || Found)
      return nullptr;
    Found = Elem;
  }

  if (!Found)
    return nullptr;

  // We don't consider a struct a single-element struct if it has padding
  // beyond the element type.
  if (Found->getSizeInBits() != Ty->getSizeInBits())
    return nullptr;

  return Found;
}

bool TargetInfo::maybeCommonClassifyReturnType(FunctionInfo &FI) const {
  const abi::Type *Ty = FI.getReturnType();

  // TODO: When Microsoft ABI is supported, CXX records may need different
  // handling here (see MicrosoftCXXABI::classifyReturnType in Clang).
  if (const auto *RT = llvm::dyn_cast<abi::RecordType>(Ty)) {
    if (!RT->canPassInRegisters()) {
      // A record that cannot pass in registers (e.g. a non-trivial copy/dtor)
      // is returned indirectly with ByVal=false. This is the RAA path and is
      // distinct from getIndirectReturnResult (plain aggregates), which uses
      // ByVal=true.
      FI.getReturnInfo() =
          ArgInfo::getIndirect(RT->getAlignment(), /*ByVal=*/false);
      return true;
    }
  }

  return false;
}

bool TargetInfo::isHomogeneousAggregate(const Type *Ty, const Type *&Base,
                                        uint64_t &Members) const {
  bool isMatrixHA = getABICompatInfo().IsMatrixHA;
  if (const auto *AT = dyn_cast<ArrayType>(Ty)) {
    if (!isMatrixHA && AT->isMatrixType())
      return false;
    uint64_t NElements = AT->getNumElements();
    if (NElements == 0)
      return false;
    if (!isHomogeneousAggregate(AT->getElementType(), Base, Members))
      return false;
    Members *= NElements;
  } else if (const auto *RT = dyn_cast<RecordType>(Ty)) {
    if (RT->hasFlexibleArrayMember())
      return false;

    Members = 0;

    // If this is a C++ record, check bases and ABI-specific restrictions.
    if (RT->isCXXRecord()) {
      if (!isPermittedToBeHomogeneousAggregate(RT))
        return false;

      for (const FieldInfo &BaseField : RT->getBaseClasses()) {
        if (BaseField.FieldType->isEmptyRecord())
          continue;

        uint64_t FldMembers = 0;
        if (!isHomogeneousAggregate(BaseField.FieldType, Base, FldMembers))
          return false;

        Members += FldMembers;
      }
    }

    for (const FieldInfo &FD : RT->getFields()) {
      // Ignore (non-zero arrays of) empty records.
      const Type *FT = FD.FieldType;
      while (const auto *AT = dyn_cast<ArrayType>(FT)) {
        // Don't drill down to the element type of a matrix type here.
        // That should fall through to the element isHomogeneousAggregate check.
        if (AT->isMatrixType())
          break;
        if (AT->getNumElements() == 0)
          return false;
        FT = AT->getElementType();
      }
      if (FT->isEmptyRecord())
        continue;

      if (isZeroLengthBitfieldPermittedInHomogeneousAggregate() &&
          FD.IsBitField && FD.BitFieldWidth == 0)
        continue;

      uint64_t FldMembers = 0;
      if (!isHomogeneousAggregate(FD.FieldType, Base, FldMembers))
        return false;

      Members =
          RT->isUnion() ? std::max(Members, FldMembers) : Members + FldMembers;
    }

    if (!Base)
      return false;

    // Ensure there is no padding.
    if (Base->getTypeAllocSize() * Members != Ty->getTypeAllocSize())
      return false;
  } else {
    Members = 1;
    const Type *ElemTy = Ty;
    if (const auto *CT = dyn_cast<ComplexType>(Ty)) {
      Members = 2;
      ElemTy = CT->getElementType();
    }

    // Most ABIs only support float, double, and some vector type widths.
    if (!isHomogeneousAggregateBaseType(ElemTy))
      return false;

    // The base type must be the same for all members. Types that agree in both
    // total size and mode (float vs. vector) are treated as equivalent here.
    if (!Base) {
      Base = ElemTy;
      // If it's a non-power-of-2 vector, its ABI size is already a power-of-2,
      // so widen it explicitly to match Clang.
      if (const auto *VT = dyn_cast<VectorType>(Base)) {
        assert(VT->isFixedLength() &&
               "scalable vectors are never homogeneous aggregates");
        uint64_t EltSize =
            VT->getElementType()->getSizeInBits().getFixedValue();
        unsigned NumElements =
            VT->getTypeAllocSize().getFixedValue() * 8 / EltSize;
        if (NumElements != VT->getNumElements().getKnownMinValue())
          Base = TB.getVectorType(VT->getElementType(),
                                  ElementCount::getFixed(NumElements),
                                  VT->getAlignment());
      }
    }

    if (Base->isVector() != ElemTy->isVector() ||
        Base->getTypeAllocSize() != ElemTy->getTypeAllocSize())
      return false;
  }
  return Members > 0 && isHomogeneousAggregateSmallEnough(Base, Members);
}

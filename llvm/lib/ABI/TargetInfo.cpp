//===- TargetInfo.cpp - Target ABI information ----------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/ABI/TargetInfo.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/MathExtras.h"
#include <algorithm>
#include <cstdint>

using namespace llvm::abi;
using llvm::dyn_cast;

bool TargetInfo::isAggregateTypeForABI(const Type *Ty) const {
  // Atomic values use the evaluation kind of their underlying value type.
  if (const auto *AT = dyn_cast<AtomicType>(Ty))
    return isAggregateTypeForABI(AT->getValueType());

  // Check for fundamental scalar types.
  if (Ty->isInteger() || Ty->isFloat() || Ty->isPointer() || Ty->isVector() ||
      Ty->isTuple())
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

const Type *TargetInfo::getI8Array(uint64_t NumBytes) const {
  assert(NumBytes != 0 && "empty padding");
  const Type *I8 = TB.getIntegerType(8, llvm::Align(1), /*Signed=*/false);
  return TB.getArrayType(I8, NumBytes, NumBytes * 8);
}

const Type *TargetInfo::getStructOfTypes(llvm::ArrayRef<const Type *> Elems,
                                         bool Packed) const {
  assert(!Elems.empty() && "empty coerce sequence");
  llvm::SmallVector<FieldInfo, 8> Fields;
  Fields.reserve(Elems.size());
  for (const Type *Elt : Elems)
    Fields.emplace_back(Elt, /*OffsetInBits=*/0);

  StructPacking Pack = Packed ? StructPacking::Packed : StructPacking::Default;
  return TB.getRecordType(Fields, llvm::TypeSize::getFixed(0), llvm::Align(1),
                          /*UnadjustedAlign=*/llvm::Align(1), Pack);
}

// Returns the alignment of Ty, a type returned by convertTypeForMem, as a
// member of the struct built there. A packed record has alignment 1. A record
// that is not packed has the alignment of its most-aligned member. An array has
// the alignment of its element type.
static llvm::Align getConvertedAlign(const Type *Ty) {
  if (const auto *AT = dyn_cast<ArrayType>(Ty))
    return getConvertedAlign(AT->getElementType());

  const auto *RT = dyn_cast<RecordType>(Ty);
  if (!RT)
    return Ty->getAlignment();
  if (RT->getPacking() == StructPacking::Packed)
    return llvm::Align(1);

  llvm::Align MaxAlign(1);
  for (llvm::ArrayRef<FieldInfo> Members :
       {RT->getFields(), RT->getBaseClasses()}) {
    for (const FieldInfo &Member : Members) {
      if (!Member.isEmpty())
        MaxAlign = std::max(MaxAlign, getConvertedAlign(Member.FieldType));
    }
  }
  return MaxAlign;
}

const Type *TargetInfo::convertTypeForMem(const Type *Ty) const {
  if (const auto *AT = dyn_cast<ArrayType>(Ty)) {
    if (AT->isMatrixType())
      return Ty;
    const Type *Elt = convertTypeForMem(AT->getElementType());
    if (Elt == AT->getElementType())
      return Ty;
    assert(AT->getSizeInBits().isFixed() &&
           "converted array element changes a scalable size");
    return TB.getArrayType(Elt, AT->getNumElements(),
                           AT->getSizeInBits().getFixedValue());
  }

  const auto *RT = dyn_cast<RecordType>(Ty);
  if (!RT || RT->isUnion())
    return Ty;

  // Current callers can't get here with virtual bases. If we need to handle
  // virtual bases in the future, we'll need explicit handling for that below.
  assert(RT->getNumVirtualBaseClasses() == 0 && "record has a virtual base");

  struct ConvertedMember {
    const Type *Ty;
    uint64_t Offset;
    llvm::Align Alignment;
  };
  llvm::SmallVector<ConvertedMember, 8> Members;
  // The record is packed when a member offset or the record size is not a
  // multiple of the converted alignment.
  bool Packed = false;
  llvm::Align MaxAlign(1);
  auto addMember = [&](const Type *MemberTy, uint64_t Offset) {
    const Type *ConvertedTy = convertTypeForMem(MemberTy);
    assert(!ConvertedTy->getSizeInBits().isScalable() &&
           "scalable member has no fixed offset");
    llvm::Align Alignment = getConvertedAlign(ConvertedTy);
    if (Offset % (Alignment.value() * 8) != 0)
      Packed = true;
    MaxAlign = std::max(MaxAlign, Alignment);
    Members.push_back({ConvertedTy, Offset, Alignment});
  };
  for (const FieldInfo &Base : RT->getBaseClasses()) {
    if (!Base.FieldType->isEmptyRecord())
      addMember(Base.FieldType, Base.OffsetInBits);
  }
  for (const FieldInfo &Field : RT->getFields()) {
    if (!Field.isEmpty())
      addMember(Field.FieldType, Field.OffsetInBits);
  }
  llvm::stable_sort(Members,
                    [](const ConvertedMember &A, const ConvertedMember &B) {
                      return A.Offset < B.Offset;
                    });

  llvm::TypeSize RecordSize = RT->getSizeInBits();
  if (RecordSize.isFixed() &&
      RecordSize.getFixedValue() % (MaxAlign.value() * 8) != 0)
    Packed = true;

  llvm::SmallVector<FieldInfo, 8> Fields;
  uint64_t Current = 0;
  // Padding in a packed record is explicit for every gap. Padding in any
  // other record is explicit only where the converted alignment does not place
  // the next member.
  auto needsPadding = [&](uint64_t Offset, llvm::Align MemberAlign) {
    uint64_t AlignBits = Packed ? 8 : MemberAlign.value() * 8;
    return Offset != llvm::alignTo(Current, AlignBits);
  };
  for (const ConvertedMember &Member : Members) {
    if (Member.Offset > Current &&
        needsPadding(Member.Offset, Member.Alignment)) {
      uint64_t PadBits = Member.Offset - Current;
      assert(PadBits % 8 == 0 && "padding is not a whole number of bytes");
      Fields.emplace_back(getI8Array(PadBits / 8), Current);
    }
    Fields.emplace_back(Member.Ty, Member.Offset);
    Current = std::max(Current, Member.Offset +
                                    Member.Ty->getSizeInBits().getFixedValue());
  }

  if (RecordSize.isFixed()) {
    uint64_t Size = RecordSize.getFixedValue();
    if (Size > Current && needsPadding(Size, MaxAlign)) {
      uint64_t PadBits = Size - Current;
      assert(PadBits % 8 == 0 && "tail padding is not a whole number of bytes");
      Fields.emplace_back(getI8Array(PadBits / 8), Current);
    }
  }

  StructPacking Pack = Packed ? StructPacking::Packed : StructPacking::Default;
  return TB.getRecordType(Fields, RecordSize, RT->getAlignment(),
                          RT->getUnadjustedAlignment(), Pack);
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

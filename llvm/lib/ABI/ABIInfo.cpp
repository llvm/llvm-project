//===----- ABIInfo.cpp ------------------------------------------- C++ ---===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#include "llvm/ABI/ABIInfo.h"

using namespace llvm::abi;
bool ABIInfo::isAggregateTypeForABI(const Type *Ty) const {
  // Check for fundamental scalar types
  if (Ty->isInteger() || Ty->isFloat() || Ty->isPointer() || Ty->isVector())
    return false;

  // Everything else is treated as aggregate
  return true;
}

bool ABIInfo::isPromotableInteger(const IntegerType *IT) const {
  unsigned BitWidth = IT->getSizeInBits().getFixedValue();
  return BitWidth < 32;
}

// Create indirect return with natural alignment
ABIArgInfo ABIInfo::getNaturalAlignIndirect(const Type *Ty, bool ByVal) const {
  return ABIArgInfo::getIndirect(Ty->getAlignment().value(), ByVal);
}
RecordArgABI ABIInfo::getRecordArgABI(const RecordType *RT) const {
  if (RT && !RT->canPassInRegisters())
    return RAA_Indirect;
  return RAA_Default;
}

RecordArgABI ABIInfo::getRecordArgABI(const RecordType *RT,
                                      bool IsCxxRecord) const {
  if (!IsCxxRecord) {
    if (!RT->canPassInRegisters())
      return RAA_Indirect;
    return RAA_Default;
  }
  return getRecordArgABI(RT);
}

RecordArgABI ABIInfo::getRecordArgABI(const Type *Ty) const {
  const RecordType *RT = dyn_cast<RecordType>(Ty);
  if (!RT)
    return RAA_Default;
  return getRecordArgABI(RT, RT->isCXXRecord());
}

bool ABIInfo::isZeroSizedType(const Type *Ty) const {
  return Ty->getSizeInBits().getFixedValue() == 0;
}

bool ABIInfo::isEmptyRecord(const RecordType *RT) const {
  if (RT->hasFlexibleArrayMember() || RT->isPolymorphic() ||
      RT->getNumVirtualBaseClasses() != 0)
    return false;

  for (unsigned I = 0; I < RT->getNumBaseClasses(); ++I) {
    const Type *BaseTy = RT->getBaseClasses()[I].FieldType;
    auto *BaseRT = dyn_cast<RecordType>(BaseTy);
    if (!BaseRT || !isEmptyRecord(BaseRT))
      return false;
  }

  for (unsigned I = 0; I < RT->getNumFields(); ++I) {
    const FieldInfo &FI = RT->getFields()[I];

    if (FI.IsBitField && FI.BitFieldWidth == 0)
      continue;
    if (FI.IsUnnamedBitfield)
      continue;

    if (!isZeroSizedType(FI.FieldType))
      return false;
  }
  return true;
}

bool ABIInfo::isEmptyField(const FieldInfo &FI) const {
  if (FI.IsUnnamedBitfield)
    return true;
  if (FI.IsBitField && FI.BitFieldWidth == 0)
    return true;

  const Type *Ty = FI.FieldType;
  while (auto *AT = dyn_cast<ArrayType>(Ty)) {
    if (AT->getNumElements() != 1)
      break;
    Ty = AT->getElementType();
  }

  if (auto *RT = dyn_cast<RecordType>(Ty))
    return isEmptyRecord(RT);

  return isZeroSizedType(Ty);
}

//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/ABI/Types.h"
#include "llvm/Support/Casting.h"

using namespace llvm;
using namespace llvm::abi;

bool RecordType::isEmpty() const {
  if (hasFlexibleArrayMember() || isPolymorphic() ||
      getNumVirtualBaseClasses() != 0)
    return false;

  for (const FieldInfo &Base : getBaseClasses()) {
    const auto *BaseRT = dyn_cast<RecordType>(Base.FieldType);
    if (!BaseRT || !BaseRT->isEmpty())
      return false;
  }

  for (const FieldInfo &FI : getFields()) {
    if (!FI.isEmpty())
      return false;
  }
  return true;
}

const FieldInfo *
RecordType::getElementContainingOffset(unsigned OffsetInBits) const {
  auto Contains = [&](const FieldInfo &Element) {
    unsigned Start = Element.OffsetInBits;
    unsigned Size = Element.FieldType->getSizeInBits().getFixedValue();
    return OffsetInBits >= Start && OffsetInBits < Start + Size;
  };

  for (const FieldInfo &Base : getBaseClasses()) {
    const auto *BaseRT = dyn_cast<RecordType>(Base.FieldType);
    if ((!BaseRT || !BaseRT->isEmpty()) && Contains(Base))
      return &Base;
  }

  for (const FieldInfo &VBase : getVirtualBaseClasses()) {
    const auto *VBaseRT = dyn_cast<RecordType>(VBase.FieldType);
    if ((!VBaseRT || !VBaseRT->isEmpty()) && Contains(VBase))
      return &VBase;
  }

  for (const FieldInfo &Field : getFields()) {
    if (Field.IsUnnamedBitfield)
      continue;
    if (Contains(Field))
      return &Field;
  }

  return nullptr;
}

bool FieldInfo::isEmpty() const {
  if (IsUnnamedBitfield)
    return true;
  if (IsBitField && BitFieldWidth == 0)
    return true;

  const Type *Ty = FieldType;
  while (const auto *AT = dyn_cast<ArrayType>(Ty)) {
    if (AT->getNumElements() != 1)
      break;
    Ty = AT->getElementType();
  }

  if (const auto *RT = dyn_cast<RecordType>(Ty))
    return RT->isEmpty();

  return Ty->isZeroSize();
}

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
  if (hasFlexibleArrayMember())
    return false;

  auto BaseIsEmpty = [](const FieldInfo &Base) {
    const auto *BaseRT = dyn_cast<RecordType>(Base.FieldType);
    return BaseRT && BaseRT->isEmpty();
  };

  // Direct virtual bases are not included in the base class list.
  for (const FieldInfo &Base : getBaseClasses())
    if (!BaseIsEmpty(Base))
      return false;
  for (const FieldInfo &Base : getDirectVirtualBaseClasses())
    if (!BaseIsEmpty(Base))
      return false;

  for (const FieldInfo &FI : getFields()) {
    // Don't treat vtable pointers as a source field for emptiness.
    if (FI.IsVTablePointer)
      continue;
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

  const Type *Ty = FieldType;
  bool WasArray = false;
  while (const auto *AT = dyn_cast<ArrayType>(Ty)) {
    // Constant arrays of zero length always count as empty.
    if (AT->getNumElements() == 0)
      return true;
    Ty = AT->getElementType();
    WasArray = true;
  }

  const auto *RT = dyn_cast<RecordType>(Ty);
  if (!RT)
    return false;

  // C++ record fields are never empty unless [[no_unique_address]] applies.
  // That exception does not apply to arrays of C++ empty records.
  if (RT->isCXXRecord() && (WasArray || !HasNoUniqueAddress))
    return false;

  return RT->isEmpty();
}

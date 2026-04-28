//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/ABI/Types.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
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
  SmallVector<std::pair<unsigned, const FieldInfo *>, 16> AllElements;

  for (const FieldInfo &Base : getBaseClasses()) {
    const auto *BaseRT = dyn_cast<RecordType>(Base.FieldType);
    if (!BaseRT || !BaseRT->isEmpty())
      AllElements.emplace_back(Base.OffsetInBits, &Base);
  }

  for (const FieldInfo &VBase : getVirtualBaseClasses()) {
    const auto *VBaseRT = dyn_cast<RecordType>(VBase.FieldType);
    if (!VBaseRT || !VBaseRT->isEmpty())
      AllElements.emplace_back(VBase.OffsetInBits, &VBase);
  }

  for (const FieldInfo &Field : getFields()) {
    if (Field.IsUnnamedBitfield)
      continue;
    AllElements.emplace_back(Field.OffsetInBits, &Field);
  }

  llvm::stable_sort(AllElements, [](const auto &A, const auto &B) {
    return A.first < B.first;
  });

  auto *It = llvm::upper_bound(AllElements, OffsetInBits,
                               [](unsigned Offset, const auto &Element) {
                                 return Offset < Element.first;
                               });

  if (It == AllElements.begin())
    return nullptr;

  --It;

  const FieldInfo *Candidate = It->second;
  unsigned ElementStart = It->first;
  unsigned ElementSize = Candidate->FieldType->getSizeInBits().getFixedValue();

  if (OffsetInBits >= ElementStart && OffsetInBits < ElementStart + ElementSize)
    return Candidate;

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

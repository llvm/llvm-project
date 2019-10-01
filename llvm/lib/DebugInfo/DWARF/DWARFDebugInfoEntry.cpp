//===- DWARFDebugInfoEntry.cpp --------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/DebugInfo/DWARF/DWARFDebugInfoEntry.h"
#include "llvm/ADT/Optional.h"
#include "llvm/DebugInfo/DWARF/DWARFDebugAbbrev.h"
#include "llvm/DebugInfo/DWARF/DWARFFormValue.h"
#include "llvm/DebugInfo/DWARF/DWARFUnit.h"
#include "llvm/Support/DataExtractor.h"
#include <cstddef>
#include <cstdint>

using namespace llvm;
using namespace dwarf;

bool DWARFDebugInfoEntry::extractFast(const DWARFUnit &U,
                                             uint64_t *OffsetPtr) {
  DWARFDataExtractor DebugInfoData = U.getDebugInfoExtractor();
  const uint64_t UEndOffset = U.getNextUnitOffset();
  return extractFast(U, OffsetPtr, DebugInfoData, UEndOffset, 0);
}

bool DWARFDebugInfoEntry::extractFast(const DWARFUnit &U, uint64_t *OffsetPtr,
                                      const DWARFDataExtractor &DebugInfoData,
                                      uint64_t UEndOffset, uint32_t D) {
  Offset = *OffsetPtr;
  Depth = D;
  if (Offset >= UEndOffset || !DebugInfoData.isValidOffset(Offset))
    return false;
  uint64_t AbbrCode = DebugInfoData.getULEB128(OffsetPtr);
  if (0 == AbbrCode) {
    // NULL debug tag entry.
    AbbrevDecl = nullptr;
    return true;
  }
  AbbrevDecl = U.getAbbreviations()->getAbbreviationDeclaration(AbbrCode);
  if (nullptr == AbbrevDecl) {
    // Restore the original offset.
    *OffsetPtr = Offset;
    return false;
  }
  // See if all attributes in this DIE have fixed byte sizes. If so, we can
  // just add this size to the offset to skip to the next DIE.
  if (Optional<size_t> FixedSize = AbbrevDecl->getFixedAttributesByteSize(U)) {
    *OffsetPtr += *FixedSize;
    return true;
  }

  // Skip all data in the .debug_info for the attributes
  for (const auto &AttrSpec : AbbrevDecl->attributes()) {
    // Check if this attribute has a fixed byte size.
    if (auto FixedSize = AttrSpec.getByteSize(U)) {
      // Attribute byte size if fixed, just add the size to the offset.
      *OffsetPtr += *FixedSize;
    } else if (!DWARFFormValue::skipValue(AttrSpec.Form, DebugInfoData,
                                          OffsetPtr, U.getFormParams())) {
      // We failed to skip this attribute's value, restore the original offset
      // and return the failure status.
      *OffsetPtr = Offset;
      return false;
    }
  }
  return true;
}

//===-- DWARFDebugInfoEntry.h -----------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIB_DEBUGINFO_DWARFDEBUGINFOENTRY_H
#define LLVM_LIB_DEBUGINFO_DWARFDEBUGINFOENTRY_H

#include "llvm/ADT/SmallVector.h"
#include "llvm/DebugInfo/DIContext.h"
#include "llvm/DebugInfo/DWARF/DWARFAbbreviationDeclaration.h"
#include "llvm/DebugInfo/DWARF/DWARFDebugRangeList.h"
#include "llvm/Support/DataTypes.h"
#include "llvm/Support/Dwarf.h"

namespace llvm {

class DWARFDebugAranges;
class DWARFCompileUnit;
class DWARFUnit;
class DWARFContext;
class DWARFFormValue;
struct DWARFDebugInfoEntryInlinedChain;

/// DWARFDebugInfoEntry - A DIE with only the minimum required data.
class DWARFDebugInfoEntry {
  /// Offset within the .debug_info of the start of this entry.
  uint32_t Offset;

  /// How many to add to "this" to get the sibling.
  uint32_t SiblingIdx;

  const DWARFAbbreviationDeclaration *AbbrevDecl;
public:
  DWARFDebugInfoEntry()
    : Offset(0), SiblingIdx(0), AbbrevDecl(nullptr) {}

  /// Extracts a debug info entry, which is a child of a given unit,
  /// starting at a given offset. If DIE can't be extracted, returns false and
  /// doesn't change OffsetPtr.
  bool extractFast(const DWARFUnit &U, uint32_t *OffsetPtr);
  /// High performance extraction should use this call.
  bool extractFast(const DWARFUnit &U, uint32_t *OffsetPtr,
                   const DataExtractor &DebugInfoData, uint32_t UEndOffset);

  uint32_t getOffset() const { return Offset; }
  bool hasChildren() const { return AbbrevDecl && AbbrevDecl->hasChildren(); }

  // We know we are kept in a vector of contiguous entries, so we know
  // our sibling will be some index after "this".
  const DWARFDebugInfoEntry *getSibling() const {
    return SiblingIdx > 0 ? this + SiblingIdx : nullptr;
  }

  // We know we are kept in a vector of contiguous entries, so we know
  // we don't need to store our child pointer, if we have a child it will
  // be the next entry in the list...
  const DWARFDebugInfoEntry *getFirstChild() const {
    return hasChildren() ? this + 1 : nullptr;
  }

  void setSibling(const DWARFDebugInfoEntry *Sibling) {
    if (Sibling) {
      // We know we are kept in a vector of contiguous entries, so we know
      // our sibling will be some index after "this".
      SiblingIdx = Sibling - this;
    } else
      SiblingIdx = 0;
  }

  const DWARFAbbreviationDeclaration *getAbbreviationDeclarationPtr() const {
    return AbbrevDecl;
  }
};

}

#endif

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

  /// The integer depth of this DIE within the compile unit DIEs where the
  /// compile/type unit DIE has a depth of zero.
  uint32_t Depth;

  const DWARFAbbreviationDeclaration *AbbrevDecl;
public:
  DWARFDebugInfoEntry()
    : Offset(0), Depth(0), AbbrevDecl(nullptr) {}

  /// Extracts a debug info entry, which is a child of a given unit,
  /// starting at a given offset. If DIE can't be extracted, returns false and
  /// doesn't change OffsetPtr.
  bool extractFast(const DWARFUnit &U, uint32_t *OffsetPtr);
  /// High performance extraction should use this call.
  bool extractFast(const DWARFUnit &U, uint32_t *OffsetPtr,
                   const DataExtractor &DebugInfoData,
                   uint32_t UEndOffset,
                   uint32_t Depth);

  uint32_t getOffset() const { return Offset; }
  uint32_t getDepth() const { return Depth; }
  dwarf::Tag getTag() const {
    return AbbrevDecl ? AbbrevDecl->getTag() : dwarf::DW_TAG_null;
  }
  bool hasChildren() const { return AbbrevDecl && AbbrevDecl->hasChildren(); }
  const DWARFAbbreviationDeclaration *getAbbreviationDeclarationPtr() const {
    return AbbrevDecl;
  }
};

}

#endif

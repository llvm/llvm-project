//===-- DWARFDebugInfoEntry.h -----------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_DEBUGINFO_DWARFDEBUGINFOENTRY_H
#define LLVM_DEBUGINFO_DWARFDEBUGINFOENTRY_H

#include "DWARFAbbreviationDeclaration.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/DataTypes.h"

namespace llvm {

class DWARFDebugAranges;
class DWARFCompileUnit;
class DWARFContext;
class DWARFFormValue;
class DWARFInlinedSubroutineChain;

/// DWARFDebugInfoEntryMinimal - A DIE with only the minimum required data.
class DWARFDebugInfoEntryMinimal {
  /// Offset within the .debug_info of the start of this entry.
  uint32_t Offset;

  /// How many to subtract from "this" to get the parent.
  /// If zero this die has no parent.
  uint32_t ParentIdx;

  /// How many to add to "this" to get the sibling.
  uint32_t SiblingIdx;

  const DWARFAbbreviationDeclaration *AbbrevDecl;
public:
  DWARFDebugInfoEntryMinimal()
    : Offset(0), ParentIdx(0), SiblingIdx(0), AbbrevDecl(0) {}

  void dump(raw_ostream &OS, const DWARFCompileUnit *cu,
            unsigned recurseDepth, unsigned indent = 0) const;
  void dumpAttribute(raw_ostream &OS, const DWARFCompileUnit *cu,
                     uint32_t *offset_ptr, uint16_t attr, uint16_t form,
                     unsigned indent = 0) const;

  bool extractFast(const DWARFCompileUnit *cu, const uint8_t *fixed_form_sizes,
                   uint32_t *offset_ptr);

  /// Extract a debug info entry for a given compile unit from the
  /// .debug_info and .debug_abbrev data starting at the given offset.
  bool extract(const DWARFCompileUnit *cu, uint32_t *offset_ptr);

  uint32_t getTag() const { return AbbrevDecl ? AbbrevDecl->getTag() : 0; }
  bool isNULL() const { return AbbrevDecl == 0; }

  /// Returns true if DIE represents a subprogram (not inlined).
  bool isSubprogramDIE() const;
  /// Returns true if DIE represents a subprogram or an inlined
  /// subroutine.
  bool isSubroutineDIE() const;

  uint32_t getOffset() const { return Offset; }
  uint32_t getNumAttributes() const {
    return !isNULL() ? AbbrevDecl->getNumAttributes() : 0;
  }
  bool hasChildren() const { return !isNULL() && AbbrevDecl->hasChildren(); }

  // We know we are kept in a vector of contiguous entries, so we know
  // our parent will be some index behind "this".
  DWARFDebugInfoEntryMinimal *getParent() {
    return ParentIdx > 0 ? this - ParentIdx : 0;
  }
  const DWARFDebugInfoEntryMinimal *getParent() const {
    return ParentIdx > 0 ? this - ParentIdx : 0;
  }
  // We know we are kept in a vector of contiguous entries, so we know
  // our sibling will be some index after "this".
  DWARFDebugInfoEntryMinimal *getSibling() {
    return SiblingIdx > 0 ? this + SiblingIdx : 0;
  }
  const DWARFDebugInfoEntryMinimal *getSibling() const {
    return SiblingIdx > 0 ? this + SiblingIdx : 0;
  }
  // We know we are kept in a vector of contiguous entries, so we know
  // we don't need to store our child pointer, if we have a child it will
  // be the next entry in the list...
  DWARFDebugInfoEntryMinimal *getFirstChild() {
    return hasChildren() ? this + 1 : 0;
  }
  const DWARFDebugInfoEntryMinimal *getFirstChild() const {
    return hasChildren() ? this + 1 : 0;
  }

  void setParent(DWARFDebugInfoEntryMinimal *parent) {
    if (parent) {
      // We know we are kept in a vector of contiguous entries, so we know
      // our parent will be some index behind "this".
      ParentIdx = this - parent;
    } else
      ParentIdx = 0;
  }
  void setSibling(DWARFDebugInfoEntryMinimal *sibling) {
    if (sibling) {
      // We know we are kept in a vector of contiguous entries, so we know
      // our sibling will be some index after "this".
      SiblingIdx = sibling - this;
      sibling->setParent(getParent());
    } else
      SiblingIdx = 0;
  }

  const DWARFAbbreviationDeclaration *getAbbreviationDeclarationPtr() const {
    return AbbrevDecl;
  }

  uint32_t getAttributeValue(const DWARFCompileUnit *cu,
                             const uint16_t attr, DWARFFormValue &formValue,
                             uint32_t *end_attr_offset_ptr = 0) const;

  const char* getAttributeValueAsString(const DWARFCompileUnit* cu,
                                        const uint16_t attr,
                                        const char *fail_value) const;

  uint64_t getAttributeValueAsUnsigned(const DWARFCompileUnit *cu,
                                       const uint16_t attr,
                                       uint64_t fail_value) const;

  uint64_t getAttributeValueAsReference(const DWARFCompileUnit *cu,
                                        const uint16_t attr,
                                        uint64_t fail_value) const;

  int64_t getAttributeValueAsSigned(const DWARFCompileUnit* cu,
                                    const uint16_t attr,
                                    int64_t fail_value) const;

  /// Retrieves DW_AT_low_pc and DW_AT_high_pc from CU.
  /// Returns true if both attributes are present.
  bool getLowAndHighPC(const DWARFCompileUnit *CU,
                       uint64_t &LowPC, uint64_t &HighPC) const;

  void buildAddressRangeTable(const DWARFCompileUnit *CU,
                              DWARFDebugAranges *DebugAranges) const;

  bool addressRangeContainsAddress(const DWARFCompileUnit *CU,
                                   const uint64_t Address) const;

  /// If a DIE represents a subprogram (or inlined subroutine),
  /// returns its mangled name (or short name, if mangled is missing).
  /// This name may be fetched from specification or abstract origin
  /// for this subprogram. Returns null if no name is found.
  const char* getSubroutineName(const DWARFCompileUnit *CU) const;

  /// Retrieves values of DW_AT_call_file, DW_AT_call_line and
  /// DW_AT_call_column from DIE (or zeroes if they are missing).
  void getCallerFrame(const DWARFCompileUnit *CU, uint32_t &CallFile,
                      uint32_t &CallLine, uint32_t &CallColumn) const;

  /// InlinedChain - represents a chain of inlined_subroutine
  /// DIEs, (possibly ending with subprogram DIE), all of which are contained
  /// in some concrete inlined instance tree. Address range for each DIE
  /// (except the last DIE) in this chain is contained in address
  /// range for next DIE in the chain.
  typedef SmallVector<DWARFDebugInfoEntryMinimal, 4> InlinedChain;

  /// Get inlined chain for a given address, rooted at the current DIE.
  /// Returns empty chain if address is not contained in address range
  /// of current DIE.
  InlinedChain getInlinedChainForAddress(const DWARFCompileUnit *CU,
                                         const uint64_t Address) const;
};

}

#endif

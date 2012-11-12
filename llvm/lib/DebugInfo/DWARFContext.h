//===-- DWARFContext.h ------------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===/

#ifndef LLVM_DEBUGINFO_DWARFCONTEXT_H
#define LLVM_DEBUGINFO_DWARFCONTEXT_H

#include "DWARFCompileUnit.h"
#include "DWARFDebugAranges.h"
#include "DWARFDebugLine.h"
#include "DWARFDebugRangeList.h"
#include "llvm/DebugInfo/DIContext.h"
#include "llvm/ADT/OwningPtr.h"
#include "llvm/ADT/SmallVector.h"

namespace llvm {

/// DWARFContext
/// This data structure is the top level entity that deals with dwarf debug
/// information parsing. The actual data is supplied through pure virtual
/// methods that a concrete implementation provides.
class DWARFContext : public DIContext {
  bool IsLittleEndian;
  const RelocAddrMap &RelocMap;

  SmallVector<DWARFCompileUnit, 1> CUs;
  OwningPtr<DWARFDebugAbbrev> Abbrev;
  OwningPtr<DWARFDebugAranges> Aranges;
  OwningPtr<DWARFDebugLine> Line;

  DWARFContext(DWARFContext &) LLVM_DELETED_FUNCTION;
  DWARFContext &operator=(DWARFContext &) LLVM_DELETED_FUNCTION;

  /// Read compile units from the debug_info section and store them in CUs.
  void parseCompileUnits();
protected:
  DWARFContext(bool isLittleEndian, const RelocAddrMap &Map) :
    IsLittleEndian(isLittleEndian), RelocMap(Map) {}
public:
  virtual void dump(raw_ostream &OS);

  /// Get the number of compile units in this context.
  unsigned getNumCompileUnits() {
    if (CUs.empty())
      parseCompileUnits();
    return CUs.size();
  }
  /// Get the compile unit at the specified index for this compile unit.
  DWARFCompileUnit *getCompileUnitAtIndex(unsigned index) {
    if (CUs.empty())
      parseCompileUnits();
    return &CUs[index];
  }

  /// Get a pointer to the parsed DebugAbbrev object.
  const DWARFDebugAbbrev *getDebugAbbrev();

  /// Get a pointer to the parsed DebugAranges object.
  const DWARFDebugAranges *getDebugAranges();

  /// Get a pointer to a parsed line table corresponding to a compile unit.
  const DWARFDebugLine::LineTable *
  getLineTableForCompileUnit(DWARFCompileUnit *cu);

  virtual DILineInfo getLineInfoForAddress(uint64_t Address,
      DILineInfoSpecifier Specifier = DILineInfoSpecifier());
  virtual DIInliningInfo getInliningInfoForAddress(uint64_t Address,
      DILineInfoSpecifier Specifier = DILineInfoSpecifier());

  bool isLittleEndian() const { return IsLittleEndian; }
  const RelocAddrMap &relocMap() const { return RelocMap; }

  virtual StringRef getInfoSection() = 0;
  virtual StringRef getAbbrevSection() = 0;
  virtual StringRef getARangeSection() = 0;
  virtual StringRef getLineSection() = 0;
  virtual StringRef getStringSection() = 0;
  virtual StringRef getRangeSection() = 0;

  static bool isSupportedVersion(unsigned version) {
    return version == 2 || version == 3;
  }
private:
  /// Return the compile unit that includes an offset (relative to .debug_info).
  DWARFCompileUnit *getCompileUnitForOffset(uint32_t Offset);

  /// Return the compile unit which contains instruction with provided
  /// address.
  DWARFCompileUnit *getCompileUnitForAddress(uint64_t Address);
};

/// DWARFContextInMemory is the simplest possible implementation of a
/// DWARFContext. It assumes all content is available in memory and stores
/// pointers to it.
class DWARFContextInMemory : public DWARFContext {
  virtual void anchor();
  StringRef InfoSection;
  StringRef AbbrevSection;
  StringRef ARangeSection;
  StringRef LineSection;
  StringRef StringSection;
  StringRef RangeSection;
public:
  DWARFContextInMemory(bool isLittleEndian,
                       StringRef infoSection,
                       StringRef abbrevSection,
                       StringRef aRangeSection,
                       StringRef lineSection,
                       StringRef stringSection,
                       StringRef rangeSection,
                       const RelocAddrMap &Map = RelocAddrMap())
    : DWARFContext(isLittleEndian, Map),
      InfoSection(infoSection),
      AbbrevSection(abbrevSection),
      ARangeSection(aRangeSection),
      LineSection(lineSection),
      StringSection(stringSection),
      RangeSection(rangeSection)
    {}

  virtual StringRef getInfoSection() { return InfoSection; }
  virtual StringRef getAbbrevSection() { return AbbrevSection; }
  virtual StringRef getARangeSection() { return ARangeSection; }
  virtual StringRef getLineSection() { return LineSection; }
  virtual StringRef getStringSection() { return StringSection; }
  virtual StringRef getRangeSection() { return RangeSection; }
};

}

#endif

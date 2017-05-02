//===- DWARFContext.h -------------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===/

#ifndef LLVM_DEBUGINFO_DWARF_DWARFCONTEXT_H
#define LLVM_DEBUGINFO_DWARF_DWARFCONTEXT_H

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/iterator_range.h"
#include "llvm/ADT/MapVector.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/DebugInfo/DIContext.h"
#include "llvm/DebugInfo/DWARF/DWARFCompileUnit.h"
#include "llvm/DebugInfo/DWARF/DWARFDebugAbbrev.h"
#include "llvm/DebugInfo/DWARF/DWARFDebugAranges.h"
#include "llvm/DebugInfo/DWARF/DWARFDebugFrame.h"
#include "llvm/DebugInfo/DWARF/DWARFDebugLine.h"
#include "llvm/DebugInfo/DWARF/DWARFDebugLoc.h"
#include "llvm/DebugInfo/DWARF/DWARFDebugMacro.h"
#include "llvm/DebugInfo/DWARF/DWARFGdbIndex.h"
#include "llvm/DebugInfo/DWARF/DWARFSection.h"
#include "llvm/DebugInfo/DWARF/DWARFTypeUnit.h"
#include "llvm/DebugInfo/DWARF/DWARFUnit.h"
#include "llvm/DebugInfo/DWARF/DWARFUnitIndex.h"
#include "llvm/Object/ObjectFile.h"
#include "llvm/Support/Host.h"
#include <cstdint>
#include <deque>
#include <map>
#include <memory>
#include <utility>

namespace llvm {

class MemoryBuffer;
class raw_ostream;

// In place of applying the relocations to the data we've read from disk we use
// a separate mapping table to the side and checking that at locations in the
// dwarf where we expect relocated values. This adds a bit of complexity to the
// dwarf parsing/extraction at the benefit of not allocating memory for the
// entire size of the debug info sections.
typedef DenseMap<uint64_t, std::pair<uint8_t, int64_t>> RelocAddrMap;

/// Reads a value from data extractor and applies a relocation to the result if
/// one exists for the given offset.
uint64_t getRelocatedValue(const DataExtractor &Data, uint32_t Size,
                           uint32_t *Off, const RelocAddrMap *Relocs);

/// DWARFContext
/// This data structure is the top level entity that deals with dwarf debug
/// information parsing. The actual data is supplied through pure virtual
/// methods that a concrete implementation provides.
class DWARFContext : public DIContext {
  DWARFUnitSection<DWARFCompileUnit> CUs;
  std::deque<DWARFUnitSection<DWARFTypeUnit>> TUs;
  std::unique_ptr<DWARFUnitIndex> CUIndex;
  std::unique_ptr<DWARFGdbIndex> GdbIndex;
  std::unique_ptr<DWARFUnitIndex> TUIndex;
  std::unique_ptr<DWARFDebugAbbrev> Abbrev;
  std::unique_ptr<DWARFDebugLoc> Loc;
  std::unique_ptr<DWARFDebugAranges> Aranges;
  std::unique_ptr<DWARFDebugLine> Line;
  std::unique_ptr<DWARFDebugFrame> DebugFrame;
  std::unique_ptr<DWARFDebugFrame> EHFrame;
  std::unique_ptr<DWARFDebugMacro> Macro;

  DWARFUnitSection<DWARFCompileUnit> DWOCUs;
  std::deque<DWARFUnitSection<DWARFTypeUnit>> DWOTUs;
  std::unique_ptr<DWARFDebugAbbrev> AbbrevDWO;
  std::unique_ptr<DWARFDebugLocDWO> LocDWO;

  /// Read compile units from the debug_info section (if necessary)
  /// and store them in CUs.
  void parseCompileUnits();

  /// Read type units from the debug_types sections (if necessary)
  /// and store them in TUs.
  void parseTypeUnits();

  /// Read compile units from the debug_info.dwo section (if necessary)
  /// and store them in DWOCUs.
  void parseDWOCompileUnits();

  /// Read type units from the debug_types.dwo section (if necessary)
  /// and store them in DWOTUs.
  void parseDWOTypeUnits();

public:
  DWARFContext() : DIContext(CK_DWARF) {}
  DWARFContext(DWARFContext &) = delete;
  DWARFContext &operator=(DWARFContext &) = delete;

  static bool classof(const DIContext *DICtx) {
    return DICtx->getKind() == CK_DWARF;
  }

  void dump(raw_ostream &OS, DIDumpType DumpType = DIDT_All,
            bool DumpEH = false, bool SummarizeTypes = false) override;

  bool verify(raw_ostream &OS, DIDumpType DumpType = DIDT_All) override;

  typedef DWARFUnitSection<DWARFCompileUnit>::iterator_range cu_iterator_range;
  typedef DWARFUnitSection<DWARFTypeUnit>::iterator_range tu_iterator_range;
  typedef iterator_range<decltype(TUs)::iterator> tu_section_iterator_range;

  /// Get compile units in this context.
  cu_iterator_range compile_units() {
    parseCompileUnits();
    return cu_iterator_range(CUs.begin(), CUs.end());
  }

  /// Get type units in this context.
  tu_section_iterator_range type_unit_sections() {
    parseTypeUnits();
    return tu_section_iterator_range(TUs.begin(), TUs.end());
  }

  /// Get compile units in the DWO context.
  cu_iterator_range dwo_compile_units() {
    parseDWOCompileUnits();
    return cu_iterator_range(DWOCUs.begin(), DWOCUs.end());
  }

  /// Get type units in the DWO context.
  tu_section_iterator_range dwo_type_unit_sections() {
    parseDWOTypeUnits();
    return tu_section_iterator_range(DWOTUs.begin(), DWOTUs.end());
  }

  /// Get the number of compile units in this context.
  unsigned getNumCompileUnits() {
    parseCompileUnits();
    return CUs.size();
  }

  /// Get the number of compile units in this context.
  unsigned getNumTypeUnits() {
    parseTypeUnits();
    return TUs.size();
  }

  /// Get the number of compile units in the DWO context.
  unsigned getNumDWOCompileUnits() {
    parseDWOCompileUnits();
    return DWOCUs.size();
  }

  /// Get the number of compile units in the DWO context.
  unsigned getNumDWOTypeUnits() {
    parseDWOTypeUnits();
    return DWOTUs.size();
  }

  /// Get the compile unit at the specified index for this compile unit.
  DWARFCompileUnit *getCompileUnitAtIndex(unsigned index) {
    parseCompileUnits();
    return CUs[index].get();
  }

  /// Get the compile unit at the specified index for the DWO compile units.
  DWARFCompileUnit *getDWOCompileUnitAtIndex(unsigned index) {
    parseDWOCompileUnits();
    return DWOCUs[index].get();
  }

  /// Get a DIE given an exact offset.
  DWARFDie getDIEForOffset(uint32_t Offset);

  const DWARFUnitIndex &getCUIndex();
  DWARFGdbIndex &getGdbIndex();
  const DWARFUnitIndex &getTUIndex();

  /// Get a pointer to the parsed DebugAbbrev object.
  const DWARFDebugAbbrev *getDebugAbbrev();

  /// Get a pointer to the parsed DebugLoc object.
  const DWARFDebugLoc *getDebugLoc();

  /// Get a pointer to the parsed dwo abbreviations object.
  const DWARFDebugAbbrev *getDebugAbbrevDWO();

  /// Get a pointer to the parsed DebugLoc object.
  const DWARFDebugLocDWO *getDebugLocDWO();

  /// Get a pointer to the parsed DebugAranges object.
  const DWARFDebugAranges *getDebugAranges();

  /// Get a pointer to the parsed frame information object.
  const DWARFDebugFrame *getDebugFrame();

  /// Get a pointer to the parsed eh frame information object.
  const DWARFDebugFrame *getEHFrame();

  /// Get a pointer to the parsed DebugMacro object.
  const DWARFDebugMacro *getDebugMacro();

  /// Get a pointer to a parsed line table corresponding to a compile unit.
  const DWARFDebugLine::LineTable *getLineTableForUnit(DWARFUnit *cu);

  DILineInfo getLineInfoForAddress(uint64_t Address,
      DILineInfoSpecifier Specifier = DILineInfoSpecifier()) override;
  DILineInfoTable getLineInfoForAddressRange(uint64_t Address, uint64_t Size,
      DILineInfoSpecifier Specifier = DILineInfoSpecifier()) override;
  DIInliningInfo getInliningInfoForAddress(uint64_t Address,
      DILineInfoSpecifier Specifier = DILineInfoSpecifier()) override;

  virtual bool isLittleEndian() const = 0;
  virtual uint8_t getAddressSize() const = 0;
  virtual const DWARFSection &getInfoSection() = 0;
  typedef MapVector<object::SectionRef, DWARFSection,
                    std::map<object::SectionRef, unsigned>> TypeSectionMap;
  virtual const TypeSectionMap &getTypesSections() = 0;
  virtual StringRef getAbbrevSection() = 0;
  virtual const DWARFSection &getLocSection() = 0;
  virtual StringRef getARangeSection() = 0;
  virtual StringRef getDebugFrameSection() = 0;
  virtual StringRef getEHFrameSection() = 0;
  virtual const DWARFSection &getLineSection() = 0;
  virtual StringRef getStringSection() = 0;
  virtual const DWARFSection& getRangeSection() = 0;
  virtual StringRef getMacinfoSection() = 0;
  virtual StringRef getPubNamesSection() = 0;
  virtual StringRef getPubTypesSection() = 0;
  virtual StringRef getGnuPubNamesSection() = 0;
  virtual StringRef getGnuPubTypesSection() = 0;

  // Sections for DWARF5 split dwarf proposal.
  virtual const DWARFSection &getInfoDWOSection() = 0;
  virtual const TypeSectionMap &getTypesDWOSections() = 0;
  virtual StringRef getAbbrevDWOSection() = 0;
  virtual const DWARFSection &getLineDWOSection() = 0;
  virtual const DWARFSection &getLocDWOSection() = 0;
  virtual StringRef getStringDWOSection() = 0;
  virtual StringRef getStringOffsetDWOSection() = 0;
  virtual const DWARFSection &getRangeDWOSection() = 0;
  virtual StringRef getAddrSection() = 0;
  virtual const DWARFSection& getAppleNamesSection() = 0;
  virtual const DWARFSection& getAppleTypesSection() = 0;
  virtual const DWARFSection& getAppleNamespacesSection() = 0;
  virtual const DWARFSection& getAppleObjCSection() = 0;
  virtual StringRef getCUIndexSection() = 0;
  virtual StringRef getGdbIndexSection() = 0;
  virtual StringRef getTUIndexSection() = 0;

  static bool isSupportedVersion(unsigned version) {
    return version == 2 || version == 3 || version == 4 || version == 5;
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

  bool IsLittleEndian;
  uint8_t AddressSize;
  DWARFSection InfoSection;
  TypeSectionMap TypesSections;
  StringRef AbbrevSection;
  DWARFSection LocSection;
  StringRef ARangeSection;
  StringRef DebugFrameSection;
  StringRef EHFrameSection;
  DWARFSection LineSection;
  StringRef StringSection;
  DWARFSection RangeSection;
  StringRef MacinfoSection;
  StringRef PubNamesSection;
  StringRef PubTypesSection;
  StringRef GnuPubNamesSection;
  StringRef GnuPubTypesSection;

  // Sections for DWARF5 split dwarf proposal.
  DWARFSection InfoDWOSection;
  TypeSectionMap TypesDWOSections;
  StringRef AbbrevDWOSection;
  DWARFSection LineDWOSection;
  DWARFSection LocDWOSection;
  StringRef StringDWOSection;
  StringRef StringOffsetDWOSection;
  DWARFSection RangeDWOSection;
  StringRef AddrSection;
  DWARFSection AppleNamesSection;
  DWARFSection AppleTypesSection;
  DWARFSection AppleNamespacesSection;
  DWARFSection AppleObjCSection;
  StringRef CUIndexSection;
  StringRef GdbIndexSection;
  StringRef TUIndexSection;

  SmallVector<SmallString<32>, 4> UncompressedSections;

  StringRef *MapSectionToMember(StringRef Name);

public:
  DWARFContextInMemory(const object::ObjectFile &Obj,
    const LoadedObjectInfo *L = nullptr);

  DWARFContextInMemory(const StringMap<std::unique_ptr<MemoryBuffer>> &Sections,
                       uint8_t AddrSize,
                       bool isLittleEndian = sys::IsLittleEndianHost);

  bool isLittleEndian() const override { return IsLittleEndian; }
  uint8_t getAddressSize() const override { return AddressSize; }
  const DWARFSection &getInfoSection() override { return InfoSection; }
  const TypeSectionMap &getTypesSections() override { return TypesSections; }
  StringRef getAbbrevSection() override { return AbbrevSection; }
  const DWARFSection &getLocSection() override { return LocSection; }
  StringRef getARangeSection() override { return ARangeSection; }
  StringRef getDebugFrameSection() override { return DebugFrameSection; }
  StringRef getEHFrameSection() override { return EHFrameSection; }
  const DWARFSection &getLineSection() override { return LineSection; }
  StringRef getStringSection() override { return StringSection; }
  const DWARFSection &getRangeSection() override { return RangeSection; }
  StringRef getMacinfoSection() override { return MacinfoSection; }
  StringRef getPubNamesSection() override { return PubNamesSection; }
  StringRef getPubTypesSection() override { return PubTypesSection; }
  StringRef getGnuPubNamesSection() override { return GnuPubNamesSection; }
  StringRef getGnuPubTypesSection() override { return GnuPubTypesSection; }
  const DWARFSection& getAppleNamesSection() override { return AppleNamesSection; }
  const DWARFSection& getAppleTypesSection() override { return AppleTypesSection; }
  const DWARFSection& getAppleNamespacesSection() override { return AppleNamespacesSection; }
  const DWARFSection& getAppleObjCSection() override { return AppleObjCSection; }

  // Sections for DWARF5 split dwarf proposal.
  const DWARFSection &getInfoDWOSection() override { return InfoDWOSection; }

  const TypeSectionMap &getTypesDWOSections() override {
    return TypesDWOSections;
  }

  StringRef getAbbrevDWOSection() override { return AbbrevDWOSection; }
  const DWARFSection &getLineDWOSection() override { return LineDWOSection; }
  const DWARFSection &getLocDWOSection() override { return LocDWOSection; }
  StringRef getStringDWOSection() override { return StringDWOSection; }

  StringRef getStringOffsetDWOSection() override {
    return StringOffsetDWOSection;
  }

  const DWARFSection &getRangeDWOSection() override { return RangeDWOSection; }

  StringRef getAddrSection() override {
    return AddrSection;
  }

  StringRef getCUIndexSection() override { return CUIndexSection; }
  StringRef getGdbIndexSection() override { return GdbIndexSection; }
  StringRef getTUIndexSection() override { return TUIndexSection; }
};

} // end namespace llvm

#endif // LLVM_DEBUGINFO_DWARF_DWARFCONTEXT_H

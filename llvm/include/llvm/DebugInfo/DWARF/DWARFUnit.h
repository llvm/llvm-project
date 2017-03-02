//===-- DWARFUnit.h ---------------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIB_DEBUGINFO_DWARFUNIT_H
#define LLVM_LIB_DEBUGINFO_DWARFUNIT_H

#include "llvm/ADT/Optional.h"
#include "llvm/ADT/iterator_range.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/DebugInfo/DWARF/DWARFDebugAbbrev.h"
#include "llvm/DebugInfo/DWARF/DWARFDebugInfoEntry.h"
#include "llvm/DebugInfo/DWARF/DWARFDebugRangeList.h"
#include "llvm/DebugInfo/DWARF/DWARFDie.h"
#include "llvm/DebugInfo/DWARF/DWARFRelocMap.h"
#include "llvm/DebugInfo/DWARF/DWARFSection.h"
#include "llvm/DebugInfo/DWARF/DWARFUnitIndex.h"
#include <vector>

namespace llvm {

namespace object {
class ObjectFile;
}

class DWARFContext;
class DWARFDebugAbbrev;
class DWARFUnit;
class StringRef;
class raw_ostream;

/// Base class for all DWARFUnitSection classes. This provides the
/// functionality common to all unit types.
class DWARFUnitSectionBase {
public:
  /// Returns the Unit that contains the given section offset in the
  /// same section this Unit originated from.
  virtual DWARFUnit *getUnitForOffset(uint32_t Offset) const = 0;

  void parse(DWARFContext &C, const DWARFSection &Section);
  void parseDWO(DWARFContext &C, const DWARFSection &DWOSection,
                DWARFUnitIndex *Index = nullptr);

protected:
  virtual void parseImpl(DWARFContext &Context, const DWARFSection &Section,
                         const DWARFDebugAbbrev *DA, StringRef RS, StringRef SS,
                         StringRef SOS, StringRef AOS, StringRef LS,
                         bool isLittleEndian, bool isDWO) = 0;

  ~DWARFUnitSectionBase() = default;
};

const DWARFUnitIndex &getDWARFUnitIndex(DWARFContext &Context,
                                        DWARFSectionKind Kind);

/// Concrete instance of DWARFUnitSection, specialized for one Unit type.
template<typename UnitType>
class DWARFUnitSection final : public SmallVector<std::unique_ptr<UnitType>, 1>,
                               public DWARFUnitSectionBase {
  bool Parsed = false;

public:
  typedef llvm::SmallVectorImpl<std::unique_ptr<UnitType>> UnitVector;
  typedef typename UnitVector::iterator iterator;
  typedef llvm::iterator_range<typename UnitVector::iterator> iterator_range;

  UnitType *getUnitForOffset(uint32_t Offset) const override {
    auto *CU = std::upper_bound(
        this->begin(), this->end(), Offset,
        [](uint32_t LHS, const std::unique_ptr<UnitType> &RHS) {
          return LHS < RHS->getNextUnitOffset();
        });
    if (CU != this->end())
      return CU->get();
    return nullptr;
  }

private:
  void parseImpl(DWARFContext &Context, const DWARFSection &Section,
                 const DWARFDebugAbbrev *DA, StringRef RS, StringRef SS,
                 StringRef SOS, StringRef AOS, StringRef LS, bool LE,
                 bool IsDWO) override {
    if (Parsed)
      return;
    const auto &Index = getDWARFUnitIndex(Context, UnitType::Section);
    DataExtractor Data(Section.Data, LE, 0);
    uint32_t Offset = 0;
    while (Data.isValidOffset(Offset)) {
      auto U = llvm::make_unique<UnitType>(Context, Section, DA, RS, SS, SOS,
                                           AOS, LS, LE, IsDWO, *this,
                                           Index.getFromOffset(Offset));
      if (!U->extract(Data, &Offset))
        break;
      this->push_back(std::move(U));
      Offset = this->back()->getNextUnitOffset();
    }
    Parsed = true;
  }
};

class DWARFUnit {
  DWARFContext &Context;
  // Section containing this DWARFUnit.
  const DWARFSection &InfoSection;

  const DWARFDebugAbbrev *Abbrev;
  StringRef RangeSection;
  uint32_t RangeSectionBase;
  StringRef LineSection;
  StringRef StringSection;
  StringRef StringOffsetSection;
  StringRef AddrOffsetSection;
  uint32_t AddrOffsetSectionBase;
  bool isLittleEndian;
  bool isDWO;
  const DWARFUnitSectionBase &UnitSection;

  uint32_t Offset;
  uint32_t Length;
  uint16_t Version;
  const DWARFAbbreviationDeclarationSet *Abbrevs;
  uint8_t AddrSize;
  uint64_t BaseAddr;
  // The compile unit debug information entry items.
  std::vector<DWARFDebugInfoEntry> DieArray;
  typedef iterator_range<std::vector<DWARFDebugInfoEntry>::iterator>
      die_iterator_range;

  class DWOHolder {
    object::OwningBinary<object::ObjectFile> DWOFile;
    std::unique_ptr<DWARFContext> DWOContext;
    DWARFUnit *DWOU;
  public:
    DWOHolder(StringRef DWOPath);
    DWARFUnit *getUnit() const { return DWOU; }
  };
  std::unique_ptr<DWOHolder> DWO;

  const DWARFUnitIndex::Entry *IndexEntry;

  uint32_t getDIEIndex(const DWARFDebugInfoEntry *Die) {
    auto First = DieArray.data();
    assert(Die >= First && Die < First + DieArray.size());
    return Die - First;
  }

protected:
  virtual bool extractImpl(DataExtractor debug_info, uint32_t *offset_ptr);
  /// Size in bytes of the unit header.
  virtual uint32_t getHeaderSize() const { return 11; }

public:
  DWARFUnit(DWARFContext &Context, const DWARFSection &Section,
            const DWARFDebugAbbrev *DA, StringRef RS, StringRef SS,
            StringRef SOS, StringRef AOS, StringRef LS, bool LE, bool IsDWO,
            const DWARFUnitSectionBase &UnitSection,
            const DWARFUnitIndex::Entry *IndexEntry = nullptr);

  virtual ~DWARFUnit();

  DWARFContext& getContext() const { return Context; }

  StringRef getLineSection() const { return LineSection; }
  StringRef getStringSection() const { return StringSection; }
  StringRef getStringOffsetSection() const { return StringOffsetSection; }
  void setAddrOffsetSection(StringRef AOS, uint32_t Base) {
    AddrOffsetSection = AOS;
    AddrOffsetSectionBase = Base;
  }
  void setRangesSection(StringRef RS, uint32_t Base) {
    RangeSection = RS;
    RangeSectionBase = Base;
  }

  bool getAddrOffsetSectionItem(uint32_t Index, uint64_t &Result) const;
  // FIXME: Result should be uint64_t in DWARF64.
  bool getStringOffsetSectionItem(uint32_t Index, uint32_t &Result) const;

  DataExtractor getDebugInfoExtractor() const {
    return DataExtractor(InfoSection.Data, isLittleEndian, AddrSize);
  }
  DataExtractor getStringExtractor() const {
    return DataExtractor(StringSection, false, 0);
  }

  const RelocAddrMap *getRelocMap() const { return &InfoSection.Relocs; }

  bool extract(DataExtractor debug_info, uint32_t* offset_ptr);

  /// extractRangeList - extracts the range list referenced by this compile
  /// unit from .debug_ranges section. Returns true on success.
  /// Requires that compile unit is already extracted.
  bool extractRangeList(uint32_t RangeListOffset,
                        DWARFDebugRangeList &RangeList) const;
  void clear();
  uint32_t getOffset() const { return Offset; }
  uint32_t getNextUnitOffset() const { return Offset + Length + 4; }
  uint32_t getLength() const { return Length; }
  uint16_t getVersion() const { return Version; }
  dwarf::DwarfFormat getFormat() const {
    return dwarf::DwarfFormat::DWARF32; // FIXME: Support DWARF64.
  }
  const DWARFAbbreviationDeclarationSet *getAbbreviations() const {
    return Abbrevs;
  }
  uint8_t getAddressByteSize() const { return AddrSize; }
  uint8_t getRefAddrByteSize() const {
    if (Version == 2)
      return AddrSize;
    return getDwarfOffsetByteSize();
  }
  uint8_t getDwarfOffsetByteSize() const {
    if (getFormat() == dwarf::DwarfFormat::DWARF64)
      return 8;
    return 4;
  }
  uint64_t getBaseAddress() const { return BaseAddr; }

  void setBaseAddress(uint64_t base_addr) {
    BaseAddr = base_addr;
  }

  DWARFDie getUnitDIE(bool ExtractUnitDIEOnly = true) {
    extractDIEsIfNeeded(ExtractUnitDIEOnly);
    if (DieArray.empty())
      return DWARFDie();
    return DWARFDie(this, &DieArray[0]);
  }

  const char *getCompilationDir();
  Optional<uint64_t> getDWOId();

  void collectAddressRanges(DWARFAddressRangesVector &CURanges);

  /// getInlinedChainForAddress - fetches inlined chain for a given address.
  /// Returns empty chain if there is no subprogram containing address. The
  /// chain is valid as long as parsed compile unit DIEs are not cleared.
  void getInlinedChainForAddress(uint64_t Address,
                                 SmallVectorImpl<DWARFDie> &InlinedChain);

  /// getUnitSection - Return the DWARFUnitSection containing this unit.
  const DWARFUnitSectionBase &getUnitSection() const { return UnitSection; }

  /// \brief Returns the number of DIEs in the unit. Parses the unit
  /// if necessary.
  unsigned getNumDIEs() {
    extractDIEsIfNeeded(false);
    return DieArray.size();
  }

  /// \brief Return the index of a DIE inside the unit's DIE vector.
  ///
  /// It is illegal to call this method with a DIE that hasn't be
  /// created by this unit. In other word, it's illegal to call this
  /// method on a DIE that isn't accessible by following
  /// children/sibling links starting from this unit's getUnitDIE().
  uint32_t getDIEIndex(const DWARFDie &D) {
    return getDIEIndex(D.getDebugInfoEntry());
  }

  /// \brief Return the DIE object at the given index.
  DWARFDie getDIEAtIndex(unsigned Index) {
    assert(Index < DieArray.size());
    return DWARFDie(this, &DieArray[Index]);
  }

  DWARFDie getParent(const DWARFDebugInfoEntry *Die);
  DWARFDie getSibling(const DWARFDebugInfoEntry *Die);

  /// \brief Return the DIE object for a given offset inside the
  /// unit's DIE vector.
  ///
  /// The unit needs to have its DIEs extracted for this method to work.
  DWARFDie getDIEForOffset(uint32_t Offset) {
    extractDIEsIfNeeded(false);
    assert(!DieArray.empty());
    auto it = std::lower_bound(
        DieArray.begin(), DieArray.end(), Offset,
        [](const DWARFDebugInfoEntry &LHS, uint32_t Offset) {
          return LHS.getOffset() < Offset;
        });
    if (it == DieArray.end())
      return DWARFDie();
    return DWARFDie(this, &*it);
  }

  uint32_t getLineTableOffset() const {
    if (IndexEntry)
      if (const auto *Contrib = IndexEntry->getOffset(DW_SECT_LINE))
        return Contrib->Offset;
    return 0;
  }

  die_iterator_range dies() {
    extractDIEsIfNeeded(false);
    return die_iterator_range(DieArray.begin(), DieArray.end());
  }

private:
  /// Size in bytes of the .debug_info data associated with this compile unit.
  size_t getDebugInfoSize() const { return Length + 4 - getHeaderSize(); }

  /// extractDIEsIfNeeded - Parses a compile unit and indexes its DIEs if it
  /// hasn't already been done. Returns the number of DIEs parsed at this call.
  size_t extractDIEsIfNeeded(bool CUDieOnly);
  /// extractDIEsToVector - Appends all parsed DIEs to a vector.
  void extractDIEsToVector(bool AppendCUDie, bool AppendNonCUDIEs,
                           std::vector<DWARFDebugInfoEntry> &DIEs) const;
  /// clearDIEs - Clear parsed DIEs to keep memory usage low.
  void clearDIEs(bool KeepCUDie);

  /// parseDWO - Parses .dwo file for current compile unit. Returns true if
  /// it was actually constructed.
  bool parseDWO();

  /// getSubprogramForAddress - Returns subprogram DIE with address range
  /// encompassing the provided address. The pointer is alive as long as parsed
  /// compile unit DIEs are not cleared.
  DWARFDie getSubprogramForAddress(uint64_t Address);
};

}

#endif

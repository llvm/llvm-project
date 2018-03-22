//===- DWARFUnit.h ----------------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_DEBUGINFO_DWARF_DWARFUNIT_H
#define LLVM_DEBUGINFO_DWARF_DWARFUNIT_H

#include "llvm/ADT/Optional.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/iterator_range.h"
#include "llvm/BinaryFormat/Dwarf.h"
#include "llvm/DebugInfo/DWARF/DWARFDebugInfoEntry.h"
#include "llvm/DebugInfo/DWARF/DWARFDebugRangeList.h"
#include "llvm/DebugInfo/DWARF/DWARFDie.h"
#include "llvm/DebugInfo/DWARF/DWARFFormValue.h"
#include "llvm/DebugInfo/DWARF/DWARFRelocMap.h"
#include "llvm/DebugInfo/DWARF/DWARFSection.h"
#include "llvm/DebugInfo/DWARF/DWARFUnitIndex.h"
#include "llvm/Support/DataExtractor.h"
#include <algorithm>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <map>
#include <memory>
#include <utility>
#include <vector>

namespace llvm {

class DWARFAbbreviationDeclarationSet;
class DWARFContext;
class DWARFDebugAbbrev;
class DWARFUnit;

/// Base class for all DWARFUnitSection classes. This provides the
/// functionality common to all unit types.
class DWARFUnitSectionBase {
public:
  /// Returns the Unit that contains the given section offset in the
  /// same section this Unit originated from.
  virtual DWARFUnit *getUnitForOffset(uint32_t Offset) const = 0;
  virtual DWARFUnit *getUnitForIndexEntry(const DWARFUnitIndex::Entry &E) = 0;

  void parse(DWARFContext &C, const DWARFSection &Section);
  void parseDWO(DWARFContext &C, const DWARFSection &DWOSection,
                bool Lazy = false);

protected:
  ~DWARFUnitSectionBase() = default;

  virtual void parseImpl(DWARFContext &Context, const DWARFObject &Obj,
                         const DWARFSection &Section,
                         const DWARFDebugAbbrev *DA, const DWARFSection *RS,
                         StringRef SS, const DWARFSection &SOS,
                         const DWARFSection *AOS, const DWARFSection &LS,
                         bool isLittleEndian, bool isDWO, bool Lazy) = 0;
};

const DWARFUnitIndex &getDWARFUnitIndex(DWARFContext &Context,
                                        DWARFSectionKind Kind);

/// Concrete instance of DWARFUnitSection, specialized for one Unit type.
template<typename UnitType>
class DWARFUnitSection final : public SmallVector<std::unique_ptr<UnitType>, 1>,
                               public DWARFUnitSectionBase {
  bool Parsed = false;
  std::function<std::unique_ptr<UnitType>(uint32_t)> Parser;

public:
  using UnitVector = SmallVectorImpl<std::unique_ptr<UnitType>>;
  using iterator = typename UnitVector::iterator;
  using iterator_range = llvm::iterator_range<typename UnitVector::iterator>;

  UnitType *getUnitForOffset(uint32_t Offset) const override {
    auto *CU = std::upper_bound(
        this->begin(), this->end(), Offset,
        [](uint32_t LHS, const std::unique_ptr<UnitType> &RHS) {
          return LHS < RHS->getNextUnitOffset();
        });
    if (CU != this->end() && (*CU)->getOffset() <= Offset)
      return CU->get();
    return nullptr;
  }
  UnitType *getUnitForIndexEntry(const DWARFUnitIndex::Entry &E) override {
    const auto *CUOff = E.getOffset(DW_SECT_INFO);
    if (!CUOff)
      return nullptr;

    auto Offset = CUOff->Offset;

    auto *CU = std::upper_bound(
        this->begin(), this->end(), CUOff->Offset,
        [](uint32_t LHS, const std::unique_ptr<UnitType> &RHS) {
          return LHS < RHS->getNextUnitOffset();
        });
    if (CU != this->end() && (*CU)->getOffset() <= Offset)
      return CU->get();

    if (!Parser)
      return nullptr;

    auto U = Parser(Offset);
    if (!U)
      U = nullptr;

    auto *NewCU = U.get();
    this->insert(CU, std::move(U));
    return NewCU;
  }

private:
  void parseImpl(DWARFContext &Context, const DWARFObject &Obj,
                 const DWARFSection &Section, const DWARFDebugAbbrev *DA,
                 const DWARFSection *RS, StringRef SS, const DWARFSection &SOS,
                 const DWARFSection *AOS, const DWARFSection &LS, bool LE,
                 bool IsDWO, bool Lazy) override {
    if (Parsed)
      return;
    DWARFDataExtractor Data(Obj, Section, LE, 0);
    if (!Parser) {
      const DWARFUnitIndex *Index = nullptr;
      if (IsDWO)
        Index = &getDWARFUnitIndex(Context, UnitType::Section);
      Parser = [=, &Context, &Section, &SOS,
                &LS](uint32_t Offset) -> std::unique_ptr<UnitType> {
        if (!Data.isValidOffset(Offset))
          return nullptr;
        auto U = llvm::make_unique<UnitType>(
            Context, Section, DA, RS, SS, SOS, AOS, LS, LE, IsDWO, *this,
            Index ? Index->getFromOffset(Offset) : nullptr);
        if (!U->extract(Data, &Offset))
          return nullptr;
        return U;
      };
    }
    if (Lazy)
      return;
    auto I = this->begin();
    uint32_t Offset = 0;
    while (Data.isValidOffset(Offset)) {
      if (I != this->end() && (*I)->getOffset() == Offset) {
        ++I;
        continue;
      }
      auto U = Parser(Offset);
      if (!U)
        break;
      Offset = U->getNextUnitOffset();
      I = std::next(this->insert(I, std::move(U)));
    }
    Parsed = true;
  }
};

/// Represents base address of the CU.
struct BaseAddress {
  uint64_t Address;
  uint64_t SectionIndex;
};

/// Represents a unit's contribution to the string offsets table.
struct StrOffsetsContributionDescriptor {
  uint64_t Base = 0;
  uint64_t Size = 0;
  /// Format and version.
  dwarf::FormParams FormParams = {0, 0, dwarf::DwarfFormat::DWARF32};

  StrOffsetsContributionDescriptor(uint64_t Base, uint64_t Size,
                                   uint8_t Version, dwarf::DwarfFormat Format)
      : Base(Base), Size(Size), FormParams({Version, 0, Format}) {}

  uint8_t getVersion() const { return FormParams.Version; }
  dwarf::DwarfFormat getFormat() const { return FormParams.Format; }
  uint8_t getDwarfOffsetByteSize() const {
    return FormParams.getDwarfOffsetByteSize();
  }
  /// Determine whether a contribution to the string offsets table is
  /// consistent with the relevant section size and that its length is
  /// a multiple of the size of one of its entries.
  Optional<StrOffsetsContributionDescriptor>
  validateContributionSize(DWARFDataExtractor &DA);
};

class DWARFUnit {
  DWARFContext &Context;
  /// Section containing this DWARFUnit.
  const DWARFSection &InfoSection;

  const DWARFDebugAbbrev *Abbrev;
  const DWARFSection *RangeSection;
  uint32_t RangeSectionBase;
  const DWARFSection &LineSection;
  StringRef StringSection;
  const DWARFSection &StringOffsetSection;
  const DWARFSection *AddrOffsetSection;
  uint32_t AddrOffsetSectionBase = 0;
  bool isLittleEndian;
  bool isDWO;
  const DWARFUnitSectionBase &UnitSection;

  // Version, address size, and DWARF format.
  dwarf::FormParams FormParams;
  /// Start, length, and DWARF format of the unit's contribution to the string
  /// offsets table (DWARF v5).
  Optional<StrOffsetsContributionDescriptor> StringOffsetsTableContribution;

  uint32_t Offset;
  uint32_t Length;
  mutable const DWARFAbbreviationDeclarationSet *Abbrevs;
  uint64_t AbbrOffset;
  uint8_t UnitType;
  llvm::Optional<BaseAddress> BaseAddr;
  /// The compile unit debug information entry items.
  std::vector<DWARFDebugInfoEntry> DieArray;

  /// Map from range's start address to end address and corresponding DIE.
  /// IntervalMap does not support range removal, as a result, we use the
  /// std::map::upper_bound for address range lookup.
  std::map<uint64_t, std::pair<uint64_t, DWARFDie>> AddrDieMap;

  using die_iterator_range =
      iterator_range<std::vector<DWARFDebugInfoEntry>::iterator>;

  std::shared_ptr<DWARFUnit> DWO;

  const DWARFUnitIndex::Entry *IndexEntry;

  uint32_t getDIEIndex(const DWARFDebugInfoEntry *Die) {
    auto First = DieArray.data();
    assert(Die >= First && Die < First + DieArray.size());
    return Die - First;
  }

protected:
  virtual bool extractImpl(const DWARFDataExtractor &debug_info,
                           uint32_t *offset_ptr);

  /// Size in bytes of the unit header.
  virtual uint32_t getHeaderSize() const { return getVersion() <= 4 ? 11 : 12; }

  /// Find the unit's contribution to the string offsets table and determine its
  /// length and form. The given offset is expected to be derived from the unit
  /// DIE's DW_AT_str_offsets_base attribute.
  Optional<StrOffsetsContributionDescriptor>
  determineStringOffsetsTableContribution(DWARFDataExtractor &DA,
                                          uint64_t Offset);

  /// Find the unit's contribution to the string offsets table and determine its
  /// length and form. The given offset is expected to be 0 in a dwo file or,
  /// in a dwp file, the start of the unit's contribution to the string offsets
  /// table section (as determined by the index table).
  Optional<StrOffsetsContributionDescriptor>
  determineStringOffsetsTableContributionDWO(DWARFDataExtractor &DA,
                                             uint64_t Offset);

public:
  DWARFUnit(DWARFContext &Context, const DWARFSection &Section,
            const DWARFDebugAbbrev *DA, const DWARFSection *RS, StringRef SS,
            const DWARFSection &SOS, const DWARFSection *AOS,
            const DWARFSection &LS, bool LE, bool IsDWO,
            const DWARFUnitSectionBase &UnitSection,
            const DWARFUnitIndex::Entry *IndexEntry = nullptr);

  virtual ~DWARFUnit();

  DWARFContext& getContext() const { return Context; }

  const DWARFSection &getLineSection() const { return LineSection; }
  StringRef getStringSection() const { return StringSection; }
  const DWARFSection &getStringOffsetSection() const {
    return StringOffsetSection;
  }

  void setAddrOffsetSection(const DWARFSection *AOS, uint32_t Base) {
    AddrOffsetSection = AOS;
    AddrOffsetSectionBase = Base;
  }

  /// Recursively update address to Die map.
  void updateAddressDieMap(DWARFDie Die);

  void setRangesSection(const DWARFSection *RS, uint32_t Base) {
    RangeSection = RS;
    RangeSectionBase = Base;
  }

  bool getAddrOffsetSectionItem(uint32_t Index, uint64_t &Result) const;
  bool getStringOffsetSectionItem(uint32_t Index, uint64_t &Result) const;

  DWARFDataExtractor getDebugInfoExtractor() const;

  DataExtractor getStringExtractor() const {
    return DataExtractor(StringSection, false, 0);
  }

  bool extract(const DWARFDataExtractor &debug_info, uint32_t *offset_ptr);

  /// extractRangeList - extracts the range list referenced by this compile
  /// unit from .debug_ranges section. Returns true on success.
  /// Requires that compile unit is already extracted.
  bool extractRangeList(uint32_t RangeListOffset,
                        DWARFDebugRangeList &RangeList) const;
  void clear();
  uint32_t getOffset() const { return Offset; }
  uint32_t getNextUnitOffset() const { return Offset + Length + 4; }
  uint32_t getLength() const { return Length; }

  const Optional<StrOffsetsContributionDescriptor> &
  getStringOffsetsTableContribution() const {
    return StringOffsetsTableContribution;
  }
  const dwarf::FormParams &getFormParams() const { return FormParams; }
  uint16_t getVersion() const { return FormParams.Version; }
  dwarf::DwarfFormat getFormat() const { return FormParams.Format; }
  uint8_t getAddressByteSize() const { return FormParams.AddrSize; }
  uint8_t getRefAddrByteSize() const { return FormParams.getRefAddrByteSize(); }
  uint8_t getDwarfOffsetByteSize() const {
    return FormParams.getDwarfOffsetByteSize();
  }

  uint8_t getDwarfStringOffsetsByteSize() const {
    assert(StringOffsetsTableContribution);
    return StringOffsetsTableContribution->getDwarfOffsetByteSize();
  }

  uint64_t getStringOffsetsBase() const {
    assert(StringOffsetsTableContribution);
    return StringOffsetsTableContribution->Base;
  }

  const DWARFAbbreviationDeclarationSet *getAbbreviations() const;

  uint8_t getUnitType() const { return UnitType; }

  static bool isMatchingUnitTypeAndTag(uint8_t UnitType, dwarf::Tag Tag) {
    switch (UnitType) {
    case dwarf::DW_UT_compile:
      return Tag == dwarf::DW_TAG_compile_unit;
    case dwarf::DW_UT_type:
      return Tag == dwarf::DW_TAG_type_unit;
    case dwarf::DW_UT_partial:
      return Tag == dwarf::DW_TAG_partial_unit;
    case dwarf::DW_UT_skeleton:
      return Tag == dwarf::DW_TAG_skeleton_unit;
    case dwarf::DW_UT_split_compile:
    case dwarf::DW_UT_split_type:
      return dwarf::isUnitType(Tag);
    }
    return false;
  }

  /// \brief Return the number of bytes for the header of a unit of
  /// UnitType type.
  ///
  /// This function must be called with a valid unit type which in
  /// DWARF5 is defined as one of the following six types.
  static uint32_t getDWARF5HeaderSize(uint8_t UnitType) {
    switch (UnitType) {
    case dwarf::DW_UT_compile:
    case dwarf::DW_UT_partial:
      return 12;
    case dwarf::DW_UT_skeleton:
    case dwarf::DW_UT_split_compile:
      return 20;
    case dwarf::DW_UT_type:
    case dwarf::DW_UT_split_type:
      return 24;
    }
    llvm_unreachable("Invalid UnitType.");
  }

  llvm::Optional<BaseAddress> getBaseAddress() const { return BaseAddr; }

  void setBaseAddress(BaseAddress BaseAddr) { this->BaseAddr = BaseAddr; }

  DWARFDie getUnitDIE(bool ExtractUnitDIEOnly = true) {
    extractDIEsIfNeeded(ExtractUnitDIEOnly);
    if (DieArray.empty())
      return DWARFDie();
    return DWARFDie(this, &DieArray[0]);
  }

  const char *getCompilationDir();
  Optional<uint64_t> getDWOId();

  void collectAddressRanges(DWARFAddressRangesVector &CURanges);

  /// Returns subprogram DIE with address range encompassing the provided
  /// address. The pointer is alive as long as parsed compile unit DIEs are not
  /// cleared.
  DWARFDie getSubroutineForAddress(uint64_t Address);

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
  DWARFDie getFirstChild(const DWARFDebugInfoEntry *Die);

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
    if (it != DieArray.end() && it->getOffset() == Offset)
      return DWARFDie(this, &*it);
    return DWARFDie();
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
};

} // end namespace llvm

#endif // LLVM_DEBUGINFO_DWARF_DWARFUNIT_H

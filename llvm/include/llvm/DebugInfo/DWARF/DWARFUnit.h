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
#include "llvm/DebugInfo/DWARF/DWARFDebugRnglists.h"
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

/// Base class describing the header of any kind of "unit."  Some information
/// is specific to certain unit types.  We separate this class out so we can
/// parse the header before deciding what specific kind of unit to construct.
class DWARFUnitHeader {
  // Offset within section.
  uint32_t Offset = 0;
  // Version, address size, and DWARF format.
  dwarf::FormParams FormParams;
  uint32_t Length = 0;
  uint64_t AbbrOffset = 0;

  // For DWO units only.
  const DWARFUnitIndex::Entry *IndexEntry = nullptr;

  // For type units only.
  uint64_t TypeHash = 0;
  uint32_t TypeOffset = 0;

  // For v5 split or skeleton compile units only.
  Optional<uint64_t> DWOId;

  // Unit type as parsed, or derived from the section kind.
  uint8_t UnitType = 0;

  // Size as parsed. uint8_t for compactness.
  uint8_t Size = 0;

public:
  /// Parse a unit header from \p debug_info starting at \p offset_ptr.
  bool extract(DWARFContext &Context, const DWARFDataExtractor &debug_info,
               uint32_t *offset_ptr, DWARFSectionKind Kind = DW_SECT_INFO,
               const DWARFUnitIndex *Index = nullptr);
  uint32_t getOffset() const { return Offset; }
  const dwarf::FormParams &getFormParams() const { return FormParams; }
  uint16_t getVersion() const { return FormParams.Version; }
  dwarf::DwarfFormat getFormat() const { return FormParams.Format; }
  uint8_t getAddressByteSize() const { return FormParams.AddrSize; }
  uint8_t getRefAddrByteSize() const { return FormParams.getRefAddrByteSize(); }
  uint8_t getDwarfOffsetByteSize() const {
    return FormParams.getDwarfOffsetByteSize();
  }
  uint32_t getLength() const { return Length; }
  uint64_t getAbbrOffset() const { return AbbrOffset; }
  Optional<uint64_t> getDWOId() const { return DWOId; }
  void setDWOId(uint64_t Id) {
    assert((!DWOId || *DWOId == Id) && "setting DWOId to a different value");
    DWOId = Id;
  }
  const DWARFUnitIndex::Entry *getIndexEntry() const { return IndexEntry; }
  uint64_t getTypeHash() const { return TypeHash; }
  uint32_t getTypeOffset() const { return TypeOffset; }
  uint8_t getUnitType() const { return UnitType; }
  bool isTypeUnit() const {
    return UnitType == dwarf::DW_UT_type || UnitType == dwarf::DW_UT_split_type;
  }
  uint8_t getSize() const { return Size; }
  // FIXME: Support DWARF64.
  uint32_t getNextUnitOffset() const { return Offset + Length + 4; }
};

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
        DWARFUnitHeader Header;
        if (!Header.extract(Context, Data, &Offset, UnitType::Section, Index))
          return nullptr;
        auto U = llvm::make_unique<UnitType>(
            Context, Section, Header, DA, RS, SS, SOS, AOS, LS, LE, IsDWO,
            *this);
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
  /// The contribution size not including the header.
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

  DWARFUnitHeader Header;
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

  /// Start, length, and DWARF format of the unit's contribution to the string
  /// offsets table (DWARF v5).
  Optional<StrOffsetsContributionDescriptor> StringOffsetsTableContribution;

  /// A table of range lists (DWARF v5 and later).
  Optional<DWARFDebugRnglistTable> RngListTable;

  mutable const DWARFAbbreviationDeclarationSet *Abbrevs;
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

  uint32_t getDIEIndex(const DWARFDebugInfoEntry *Die) {
    auto First = DieArray.data();
    assert(Die >= First && Die < First + DieArray.size());
    return Die - First;
  }

protected:
  const DWARFUnitHeader &getHeader() const { return Header; }

  /// Size in bytes of the parsed unit header.
  uint32_t getHeaderSize() const { return Header.getSize(); }

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
            const DWARFUnitHeader &Header,
            const DWARFDebugAbbrev *DA, const DWARFSection *RS, StringRef SS,
            const DWARFSection &SOS, const DWARFSection *AOS,
            const DWARFSection &LS, bool LE, bool IsDWO,
            const DWARFUnitSectionBase &UnitSection);

  virtual ~DWARFUnit();

  DWARFContext& getContext() const { return Context; }
  uint32_t getOffset() const { return Header.getOffset(); }
  const dwarf::FormParams &getFormParams() const {
    return Header.getFormParams();
  }
  uint16_t getVersion() const { return Header.getVersion(); }
  uint8_t getAddressByteSize() const { return Header.getAddressByteSize(); }
  uint8_t getRefAddrByteSize() const { return Header.getRefAddrByteSize(); }
  uint8_t getDwarfOffsetByteSize() const {
    return Header.getDwarfOffsetByteSize();
  }
  uint32_t getLength() const { return Header.getLength(); }
  uint8_t getUnitType() const { return Header.getUnitType(); }
  uint32_t getNextUnitOffset() const { return Header.getNextUnitOffset(); }
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

  /// Extract the range list referenced by this compile unit from the
  /// .debug_ranges section. If the extraction is unsuccessful, an error
  /// is returned. Successful extraction requires that the compile unit
  /// has already been extracted.
  Error extractRangeList(uint32_t RangeListOffset,
                         DWARFDebugRangeList &RangeList) const;
  void clear();

  const Optional<StrOffsetsContributionDescriptor> &
  getStringOffsetsTableContribution() const {
    return StringOffsetsTableContribution;
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

  /// Return the number of bytes for the header of a unit of
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

  llvm::Optional<BaseAddress> getBaseAddress();

  DWARFDie getUnitDIE(bool ExtractUnitDIEOnly = true) {
    extractDIEsIfNeeded(ExtractUnitDIEOnly);
    if (DieArray.empty())
      return DWARFDie();
    return DWARFDie(this, &DieArray[0]);
  }

  const char *getCompilationDir();
  Optional<uint64_t> getDWOId() {
    extractDIEsIfNeeded(/*CUDieOnly*/ true);
    return getHeader().getDWOId();
  }
  void setDWOId(uint64_t NewID) { Header.setDWOId(NewID); }

  /// Return a vector of address ranges resulting from a (possibly encoded)
  /// range list starting at a given offset in the appropriate ranges section.
  Expected<DWARFAddressRangesVector> findRnglistFromOffset(uint32_t Offset);

  /// Return a vector of address ranges retrieved from an encoded range
  /// list whose offset is found via a table lookup given an index (DWARF v5
  /// and later).
  Expected<DWARFAddressRangesVector> findRnglistFromIndex(uint32_t Index);

  /// Return a rangelist's offset based on an index. The index designates
  /// an entry in the rangelist table's offset array and is supplied by
  /// DW_FORM_rnglistx.
  Optional<uint32_t> getRnglistOffset(uint32_t Index) {
    if (RngListTable)
      return RngListTable->getOffsetEntry(Index);
    return None;
  }

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

  /// Returns the number of DIEs in the unit. Parses the unit
  /// if necessary.
  unsigned getNumDIEs() {
    extractDIEsIfNeeded(false);
    return DieArray.size();
  }

  /// Return the index of a DIE inside the unit's DIE vector.
  ///
  /// It is illegal to call this method with a DIE that hasn't be
  /// created by this unit. In other word, it's illegal to call this
  /// method on a DIE that isn't accessible by following
  /// children/sibling links starting from this unit's getUnitDIE().
  uint32_t getDIEIndex(const DWARFDie &D) {
    return getDIEIndex(D.getDebugInfoEntry());
  }

  /// Return the DIE object at the given index.
  DWARFDie getDIEAtIndex(unsigned Index) {
    assert(Index < DieArray.size());
    return DWARFDie(this, &DieArray[Index]);
  }

  DWARFDie getParent(const DWARFDebugInfoEntry *Die);
  DWARFDie getSibling(const DWARFDebugInfoEntry *Die);
  DWARFDie getFirstChild(const DWARFDebugInfoEntry *Die);

  /// Return the DIE object for a given offset inside the
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
    if (auto IndexEntry = Header.getIndexEntry())
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
  size_t getDebugInfoSize() const {
    return Header.getLength() + 4 - getHeaderSize();
  }

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

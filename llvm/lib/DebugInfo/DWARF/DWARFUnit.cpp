//===- DWARFUnit.cpp ------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/DebugInfo/DWARF/DWARFUnit.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/DebugInfo/DWARF/DWARFAbbreviationDeclaration.h"
#include "llvm/DebugInfo/DWARF/DWARFCompileUnit.h"
#include "llvm/DebugInfo/DWARF/DWARFContext.h"
#include "llvm/DebugInfo/DWARF/DWARFDebugAbbrev.h"
#include "llvm/DebugInfo/DWARF/DWARFDebugInfoEntry.h"
#include "llvm/DebugInfo/DWARF/DWARFDebugRnglists.h"
#include "llvm/DebugInfo/DWARF/DWARFDie.h"
#include "llvm/DebugInfo/DWARF/DWARFFormValue.h"
#include "llvm/DebugInfo/DWARF/DWARFTypeUnit.h"
#include "llvm/Support/DataExtractor.h"
#include "llvm/Support/Errc.h"
#include "llvm/Support/Path.h"
#include <algorithm>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <utility>
#include <vector>

using namespace llvm;
using namespace dwarf;

void DWARFUnitVector::addUnitsForSection(DWARFContext &C,
                                         const DWARFSection &Section,
                                         DWARFSectionKind SectionKind) {
  const DWARFObject &D = C.getDWARFObj();
  addUnitsImpl(C, D, Section, C.getDebugAbbrev(), &D.getRangesSection(),
               &D.getLocSection(), D.getStrSection(),
               D.getStrOffsetsSection(), &D.getAddrSection(),
               D.getLineSection(), D.isLittleEndian(), false, false,
               SectionKind);
}

void DWARFUnitVector::addUnitsForDWOSection(DWARFContext &C,
                                            const DWARFSection &DWOSection,
                                            DWARFSectionKind SectionKind,
                                            bool Lazy) {
  const DWARFObject &D = C.getDWARFObj();
  addUnitsImpl(C, D, DWOSection, C.getDebugAbbrevDWO(), &D.getRangesDWOSection(),
               &D.getLocDWOSection(), D.getStrDWOSection(),
               D.getStrOffsetsDWOSection(), &D.getAddrSection(),
               D.getLineDWOSection(), C.isLittleEndian(), true, Lazy,
               SectionKind);
}

void DWARFUnitVector::addUnitsImpl(
    DWARFContext &Context, const DWARFObject &Obj, const DWARFSection &Section,
    const DWARFDebugAbbrev *DA, const DWARFSection *RS,
    const DWARFSection *LocSection, StringRef SS, const DWARFSection &SOS,
    const DWARFSection *AOS, const DWARFSection &LS, bool LE, bool IsDWO,
    bool Lazy, DWARFSectionKind SectionKind) {
  DWARFDataExtractor Data(Obj, Section, LE, 0);
  // Lazy initialization of Parser, now that we have all section info.
  if (!Parser) {
    Parser = [=, &Context, &Obj, &Section, &SOS,
              &LS](uint64_t Offset, DWARFSectionKind SectionKind,
                   const DWARFSection *CurSection,
                   const DWARFUnitIndex::Entry *IndexEntry)
        -> std::unique_ptr<DWARFUnit> {
      const DWARFSection &InfoSection = CurSection ? *CurSection : Section;
      DWARFDataExtractor Data(Obj, InfoSection, LE, 0);
      if (!Data.isValidOffset(Offset))
        return nullptr;
      const DWARFUnitIndex *Index = nullptr;
      if (IsDWO)
        Index = &getDWARFUnitIndex(Context, SectionKind);
      DWARFUnitHeader Header;
      if (!Header.extract(Context, Data, &Offset, SectionKind, Index,
                          IndexEntry))
        return nullptr;
      std::unique_ptr<DWARFUnit> U;
      if (Header.isTypeUnit())
        U = std::make_unique<DWARFTypeUnit>(Context, InfoSection, Header, DA,
                                             RS, LocSection, SS, SOS, AOS, LS,
                                             LE, IsDWO, *this);
      else
        U = std::make_unique<DWARFCompileUnit>(Context, InfoSection, Header,
                                                DA, RS, LocSection, SS, SOS,
                                                AOS, LS, LE, IsDWO, *this);
      return U;
    };
  }
  if (Lazy)
    return;
  // Find a reasonable insertion point within the vector.  We skip over
  // (a) units from a different section, (b) units from the same section
  // but with lower offset-within-section.  This keeps units in order
  // within a section, although not necessarily within the object file,
  // even if we do lazy parsing.
  auto I = this->begin();
  uint64_t Offset = 0;
  while (Data.isValidOffset(Offset)) {
    if (I != this->end() &&
        (&(*I)->getInfoSection() != &Section || (*I)->getOffset() == Offset)) {
      ++I;
      continue;
    }
    auto U = Parser(Offset, SectionKind, &Section, nullptr);
    // If parsing failed, we're done with this section.
    if (!U)
      break;
    Offset = U->getNextUnitOffset();
    I = std::next(this->insert(I, std::move(U)));
  }
}

DWARFUnit *DWARFUnitVector::addUnit(std::unique_ptr<DWARFUnit> Unit) {
  auto I = std::upper_bound(begin(), end(), Unit,
                            [](const std::unique_ptr<DWARFUnit> &LHS,
                               const std::unique_ptr<DWARFUnit> &RHS) {
                              return LHS->getOffset() < RHS->getOffset();
                            });
  return this->insert(I, std::move(Unit))->get();
}

DWARFUnit *DWARFUnitVector::getUnitForOffset(uint64_t Offset) const {
  auto end = begin() + getNumInfoUnits();
  auto *CU =
      std::upper_bound(begin(), end, Offset,
                       [](uint64_t LHS, const std::unique_ptr<DWARFUnit> &RHS) {
                         return LHS < RHS->getNextUnitOffset();
                       });
  if (CU != end && (*CU)->getOffset() <= Offset)
    return CU->get();
  return nullptr;
}

DWARFUnit *
DWARFUnitVector::getUnitForIndexEntry(const DWARFUnitIndex::Entry &E) {
  const auto *CUOff = E.getOffset(DW_SECT_INFO);
  if (!CUOff)
    return nullptr;

  auto Offset = CUOff->Offset;
  auto end = begin() + getNumInfoUnits();

  auto *CU =
      std::upper_bound(begin(), end, CUOff->Offset,
                       [](uint64_t LHS, const std::unique_ptr<DWARFUnit> &RHS) {
                         return LHS < RHS->getNextUnitOffset();
                       });
  if (CU != end && (*CU)->getOffset() <= Offset)
    return CU->get();

  if (!Parser)
    return nullptr;

  auto U = Parser(Offset, DW_SECT_INFO, nullptr, &E);
  if (!U)
    U = nullptr;

  auto *NewCU = U.get();
  this->insert(CU, std::move(U));
  ++NumInfoUnits;
  return NewCU;
}

DWARFUnit::DWARFUnit(DWARFContext &DC, const DWARFSection &Section,
                     const DWARFUnitHeader &Header, const DWARFDebugAbbrev *DA,
                     const DWARFSection *RS, const DWARFSection *LocSection,
                     StringRef SS, const DWARFSection &SOS,
                     const DWARFSection *AOS, const DWARFSection &LS, bool LE,
                     bool IsDWO, const DWARFUnitVector &UnitVector)
    : Context(DC), InfoSection(Section), Header(Header), Abbrev(DA),
      RangeSection(RS), LineSection(LS), StringSection(SS),
      StringOffsetSection(SOS), AddrOffsetSection(AOS), isLittleEndian(LE),
      IsDWO(IsDWO), UnitVector(UnitVector) {
  clear();
  if (IsDWO) {
    // If we are reading a package file, we need to adjust the location list
    // data based on the index entries.
    StringRef Data = LocSection->Data;
    if (auto *IndexEntry = Header.getIndexEntry())
      if (const auto *C = IndexEntry->getOffset(DW_SECT_LOC))
        Data = Data.substr(C->Offset, C->Length);

    DWARFDataExtractor DWARFData =
        Header.getVersion() >= 5
            ? DWARFDataExtractor(Context.getDWARFObj(),
                                 Context.getDWARFObj().getLoclistsDWOSection(),
                                 isLittleEndian, getAddressByteSize())
            : DWARFDataExtractor(Data, isLittleEndian, getAddressByteSize());
    LocTable =
        std::make_unique<DWARFDebugLoclists>(DWARFData, Header.getVersion());

  } else if (Header.getVersion() >= 5) {
    LocTable = std::make_unique<DWARFDebugLoclists>(
        DWARFDataExtractor(Context.getDWARFObj(),
                           Context.getDWARFObj().getLoclistsSection(),
                           isLittleEndian, getAddressByteSize()),
        Header.getVersion());
  } else {
    LocTable = std::make_unique<DWARFDebugLoc>(
        DWARFDataExtractor(Context.getDWARFObj(), *LocSection, isLittleEndian,
                           getAddressByteSize()));
  }
}

DWARFUnit::~DWARFUnit() = default;

DWARFDataExtractor DWARFUnit::getDebugInfoExtractor() const {
  return DWARFDataExtractor(Context.getDWARFObj(), InfoSection, isLittleEndian,
                            getAddressByteSize());
}

Optional<object::SectionedAddress>
DWARFUnit::getAddrOffsetSectionItem(uint32_t Index) const {
  if (IsDWO) {
    auto R = Context.info_section_units();
    auto I = R.begin();
    // Surprising if a DWO file has more than one skeleton unit in it - this
    // probably shouldn't be valid, but if a use case is found, here's where to
    // support it (probably have to linearly search for the matching skeleton CU
    // here)
    if (I != R.end() && std::next(I) == R.end())
      return (*I)->getAddrOffsetSectionItem(Index);
  }
  if (!AddrOffsetSectionBase)
    return None;
  uint64_t Offset = *AddrOffsetSectionBase + Index * getAddressByteSize();
  if (AddrOffsetSection->Data.size() < Offset + getAddressByteSize())
    return None;
  DWARFDataExtractor DA(Context.getDWARFObj(), *AddrOffsetSection,
                        isLittleEndian, getAddressByteSize());
  uint64_t Section;
  uint64_t Address = DA.getRelocatedAddress(&Offset, &Section);
  return {{Address, Section}};
}

Optional<uint64_t> DWARFUnit::getStringOffsetSectionItem(uint32_t Index) const {
  if (!StringOffsetsTableContribution)
    return None;
  unsigned ItemSize = getDwarfStringOffsetsByteSize();
  uint64_t Offset = getStringOffsetsBase() + Index * ItemSize;
  if (StringOffsetSection.Data.size() < Offset + ItemSize)
    return None;
  DWARFDataExtractor DA(Context.getDWARFObj(), StringOffsetSection,
                        isLittleEndian, 0);
  return DA.getRelocatedValue(ItemSize, &Offset);
}

bool DWARFUnitHeader::extract(DWARFContext &Context,
                              const DWARFDataExtractor &debug_info,
                              uint64_t *offset_ptr,
                              DWARFSectionKind SectionKind,
                              const DWARFUnitIndex *Index,
                              const DWARFUnitIndex::Entry *Entry) {
  Offset = *offset_ptr;
  Error Err = Error::success();
  IndexEntry = Entry;
  if (!IndexEntry && Index)
    IndexEntry = Index->getFromOffset(*offset_ptr);
  Length = debug_info.getRelocatedValue(4, offset_ptr, nullptr, &Err);
  FormParams.Format = DWARF32;
  if (Length == dwarf::DW_LENGTH_DWARF64) {
    Length = debug_info.getU64(offset_ptr, &Err);
    FormParams.Format = DWARF64;
  }
  FormParams.Version = debug_info.getU16(offset_ptr, &Err);
  if (FormParams.Version >= 5) {
    UnitType = debug_info.getU8(offset_ptr, &Err);
    FormParams.AddrSize = debug_info.getU8(offset_ptr, &Err);
    AbbrOffset = debug_info.getRelocatedValue(
        FormParams.getDwarfOffsetByteSize(), offset_ptr, nullptr, &Err);
  } else {
    AbbrOffset = debug_info.getRelocatedValue(
        FormParams.getDwarfOffsetByteSize(), offset_ptr, nullptr, &Err);
    FormParams.AddrSize = debug_info.getU8(offset_ptr, &Err);
    // Fake a unit type based on the section type.  This isn't perfect,
    // but distinguishing compile and type units is generally enough.
    if (SectionKind == DW_SECT_TYPES)
      UnitType = DW_UT_type;
    else
      UnitType = DW_UT_compile;
  }
  if (isTypeUnit()) {
    TypeHash = debug_info.getU64(offset_ptr, &Err);
    TypeOffset = debug_info.getUnsigned(
        offset_ptr, FormParams.getDwarfOffsetByteSize(), &Err);
  } else if (UnitType == DW_UT_split_compile || UnitType == DW_UT_skeleton)
    DWOId = debug_info.getU64(offset_ptr, &Err);

  if (errorToBool(std::move(Err)))
    return false;

  if (IndexEntry) {
    if (AbbrOffset)
      return false;
    auto *UnitContrib = IndexEntry->getOffset();
    if (!UnitContrib || UnitContrib->Length != (Length + 4))
      return false;
    auto *AbbrEntry = IndexEntry->getOffset(DW_SECT_ABBREV);
    if (!AbbrEntry)
      return false;
    AbbrOffset = AbbrEntry->Offset;
  }

  // Header fields all parsed, capture the size of this unit header.
  assert(*offset_ptr - Offset <= 255 && "unexpected header size");
  Size = uint8_t(*offset_ptr - Offset);

  // Type offset is unit-relative; should be after the header and before
  // the end of the current unit.
  bool TypeOffsetOK =
      !isTypeUnit()
          ? true
          : TypeOffset >= Size &&
                TypeOffset < getLength() + getUnitLengthFieldByteSize();
  bool LengthOK = debug_info.isValidOffset(getNextUnitOffset() - 1);
  bool VersionOK = DWARFContext::isSupportedVersion(getVersion());
  bool AddrSizeOK = getAddressByteSize() == 4 || getAddressByteSize() == 8;

  if (!LengthOK || !VersionOK || !AddrSizeOK || !TypeOffsetOK)
    return false;

  // Keep track of the highest DWARF version we encounter across all units.
  Context.setMaxVersionIfGreater(getVersion());
  return true;
}

// Parse the rangelist table header, including the optional array of offsets
// following it (DWARF v5 and later).
template<typename ListTableType>
static Expected<ListTableType>
parseListTableHeader(DWARFDataExtractor &DA, uint64_t Offset,
                        DwarfFormat Format) {
  // We are expected to be called with Offset 0 or pointing just past the table
  // header. Correct Offset in the latter case so that it points to the start
  // of the header.
  if (Offset > 0) {
    uint64_t HeaderSize = DWARFListTableHeader::getHeaderSize(Format);
    if (Offset < HeaderSize)
      return createStringError(errc::invalid_argument, "did not detect a valid"
                               " list table with base = 0x%" PRIx64 "\n",
                               Offset);
    Offset -= HeaderSize;
  }
  ListTableType Table;
  if (Error E = Table.extractHeaderAndOffsets(DA, &Offset))
    return std::move(E);
  return Table;
}

Error DWARFUnit::extractRangeList(uint64_t RangeListOffset,
                                  DWARFDebugRangeList &RangeList) const {
  // Require that compile unit is extracted.
  assert(!DieArray.empty());
  DWARFDataExtractor RangesData(Context.getDWARFObj(), *RangeSection,
                                isLittleEndian, getAddressByteSize());
  uint64_t ActualRangeListOffset = RangeSectionBase + RangeListOffset;
  return RangeList.extract(RangesData, &ActualRangeListOffset);
}

void DWARFUnit::clear() {
  Abbrevs = nullptr;
  BaseAddr.reset();
  RangeSectionBase = 0;
  LocSectionBase = 0;
  AddrOffsetSectionBase = None;
  clearDIEs(false);
  DWO.reset();
}

const char *DWARFUnit::getCompilationDir() {
  return dwarf::toString(getUnitDIE().find(DW_AT_comp_dir), nullptr);
}

void DWARFUnit::extractDIEsToVector(
    bool AppendCUDie, bool AppendNonCUDies,
    std::vector<DWARFDebugInfoEntry> &Dies) const {
  if (!AppendCUDie && !AppendNonCUDies)
    return;

  // Set the offset to that of the first DIE and calculate the start of the
  // next compilation unit header.
  uint64_t DIEOffset = getOffset() + getHeaderSize();
  uint64_t NextCUOffset = getNextUnitOffset();
  DWARFDebugInfoEntry DIE;
  DWARFDataExtractor DebugInfoData = getDebugInfoExtractor();
  uint32_t Depth = 0;
  bool IsCUDie = true;

  while (DIE.extractFast(*this, &DIEOffset, DebugInfoData, NextCUOffset,
                         Depth)) {
    if (IsCUDie) {
      if (AppendCUDie)
        Dies.push_back(DIE);
      if (!AppendNonCUDies)
        break;
      // The average bytes per DIE entry has been seen to be
      // around 14-20 so let's pre-reserve the needed memory for
      // our DIE entries accordingly.
      Dies.reserve(Dies.size() + getDebugInfoSize() / 14);
      IsCUDie = false;
    } else {
      Dies.push_back(DIE);
    }

    if (const DWARFAbbreviationDeclaration *AbbrDecl =
            DIE.getAbbreviationDeclarationPtr()) {
      // Normal DIE
      if (AbbrDecl->hasChildren())
        ++Depth;
    } else {
      // NULL DIE.
      if (Depth > 0)
        --Depth;
      if (Depth == 0)
        break;  // We are done with this compile unit!
    }
  }

  // Give a little bit of info if we encounter corrupt DWARF (our offset
  // should always terminate at or before the start of the next compilation
  // unit header).
  if (DIEOffset > NextCUOffset)
    Context.getWarningHandler()(
        createStringError(errc::invalid_argument,
                          "DWARF compile unit extends beyond its "
                          "bounds cu 0x%8.8" PRIx64 " "
                          "at 0x%8.8" PRIx64 "\n",
                          getOffset(), DIEOffset));
}

void DWARFUnit::extractDIEsIfNeeded(bool CUDieOnly) {
  if (Error e = tryExtractDIEsIfNeeded(CUDieOnly))
    Context.getRecoverableErrorHandler()(std::move(e));
}

Error DWARFUnit::tryExtractDIEsIfNeeded(bool CUDieOnly) {
  if ((CUDieOnly && !DieArray.empty()) ||
      DieArray.size() > 1)
    return Error::success(); // Already parsed.

  bool HasCUDie = !DieArray.empty();
  extractDIEsToVector(!HasCUDie, !CUDieOnly, DieArray);

  if (DieArray.empty())
    return Error::success();

  // If CU DIE was just parsed, copy several attribute values from it.
  if (HasCUDie)
    return Error::success();

  DWARFDie UnitDie(this, &DieArray[0]);
  if (Optional<uint64_t> DWOId = toUnsigned(UnitDie.find(DW_AT_GNU_dwo_id)))
    Header.setDWOId(*DWOId);
  if (!IsDWO) {
    assert(AddrOffsetSectionBase == None);
    assert(RangeSectionBase == 0);
    assert(LocSectionBase == 0);
    AddrOffsetSectionBase = toSectionOffset(UnitDie.find(DW_AT_addr_base));
    if (!AddrOffsetSectionBase)
      AddrOffsetSectionBase =
          toSectionOffset(UnitDie.find(DW_AT_GNU_addr_base));
    RangeSectionBase = toSectionOffset(UnitDie.find(DW_AT_rnglists_base), 0);
    LocSectionBase = toSectionOffset(UnitDie.find(DW_AT_loclists_base), 0);
  }

  // In general, in DWARF v5 and beyond we derive the start of the unit's
  // contribution to the string offsets table from the unit DIE's
  // DW_AT_str_offsets_base attribute. Split DWARF units do not use this
  // attribute, so we assume that there is a contribution to the string
  // offsets table starting at offset 0 of the debug_str_offsets.dwo section.
  // In both cases we need to determine the format of the contribution,
  // which may differ from the unit's format.
  DWARFDataExtractor DA(Context.getDWARFObj(), StringOffsetSection,
                        isLittleEndian, 0);
  if (IsDWO || getVersion() >= 5) {
    auto StringOffsetOrError =
        IsDWO ? determineStringOffsetsTableContributionDWO(DA)
              : determineStringOffsetsTableContribution(DA);
    if (!StringOffsetOrError)
      return createStringError(errc::invalid_argument,
                               "invalid reference to or invalid content in "
                               ".debug_str_offsets[.dwo]: " +
                                   toString(StringOffsetOrError.takeError()));

    StringOffsetsTableContribution = *StringOffsetOrError;
  }

  // DWARF v5 uses the .debug_rnglists and .debug_rnglists.dwo sections to
  // describe address ranges.
  if (getVersion() >= 5) {
    if (IsDWO)
      setRangesSection(&Context.getDWARFObj().getRnglistsDWOSection(), 0);
    else
      setRangesSection(&Context.getDWARFObj().getRnglistsSection(),
                       toSectionOffset(UnitDie.find(DW_AT_rnglists_base), 0));
    if (RangeSection->Data.size()) {
      // Parse the range list table header. Individual range lists are
      // extracted lazily.
      DWARFDataExtractor RangesDA(Context.getDWARFObj(), *RangeSection,
                                  isLittleEndian, 0);
      auto TableOrError = parseListTableHeader<DWARFDebugRnglistTable>(
          RangesDA, RangeSectionBase, Header.getFormat());
      if (!TableOrError)
        return createStringError(errc::invalid_argument,
                                 "parsing a range list table: " +
                                     toString(TableOrError.takeError()));

      RngListTable = TableOrError.get();

      // In a split dwarf unit, there is no DW_AT_rnglists_base attribute.
      // Adjust RangeSectionBase to point past the table header.
      if (IsDWO && RngListTable)
        RangeSectionBase = RngListTable->getHeaderSize();
    }

    // In a split dwarf unit, there is no DW_AT_loclists_base attribute.
    // Setting LocSectionBase to point past the table header.
    if (IsDWO)
      setLocSection(&Context.getDWARFObj().getLoclistsDWOSection(),
                    DWARFListTableHeader::getHeaderSize(Header.getFormat()));
    else
      setLocSection(&Context.getDWARFObj().getLoclistsSection(),
                    toSectionOffset(UnitDie.find(DW_AT_loclists_base), 0));

    if (LocSection->Data.size()) {
      if (IsDWO)
        LoclistTableHeader.emplace(".debug_loclists.dwo", "locations");
      else
        LoclistTableHeader.emplace(".debug_loclists", "locations");

      uint64_t HeaderSize = DWARFListTableHeader::getHeaderSize(Header.getFormat());
      uint64_t Offset = getLocSectionBase();
      DWARFDataExtractor Data(Context.getDWARFObj(), *LocSection,
                              isLittleEndian, getAddressByteSize());
      if (Offset < HeaderSize)
        return createStringError(errc::invalid_argument,
                                 "did not detect a valid"
                                 " list table with base = 0x%" PRIx64 "\n",
                                 Offset);
      Offset -= HeaderSize;
      if (Error E = LoclistTableHeader->extract(Data, &Offset))
        return createStringError(errc::invalid_argument,
                                 "parsing a loclist table: " +
                                     toString(std::move(E)));
    }
  }

  // Don't fall back to DW_AT_GNU_ranges_base: it should be ignored for
  // skeleton CU DIE, so that DWARF users not aware of it are not broken.
  return Error::success();
}

bool DWARFUnit::parseDWO() {
  if (IsDWO)
    return false;
  if (DWO.get())
    return false;
  DWARFDie UnitDie = getUnitDIE();
  if (!UnitDie)
    return false;
  auto DWOFileName = getVersion() >= 5
                         ? dwarf::toString(UnitDie.find(DW_AT_dwo_name))
                         : dwarf::toString(UnitDie.find(DW_AT_GNU_dwo_name));
  if (!DWOFileName)
    return false;
  auto CompilationDir = dwarf::toString(UnitDie.find(DW_AT_comp_dir));
  SmallString<16> AbsolutePath;
  if (sys::path::is_relative(*DWOFileName) && CompilationDir &&
      *CompilationDir) {
    sys::path::append(AbsolutePath, *CompilationDir);
  }
  sys::path::append(AbsolutePath, *DWOFileName);
  auto DWOId = getDWOId();
  if (!DWOId)
    return false;
  auto DWOContext = Context.getDWOContext(AbsolutePath);
  if (!DWOContext)
    return false;

  DWARFCompileUnit *DWOCU = DWOContext->getDWOCompileUnitForHash(*DWOId);
  if (!DWOCU)
    return false;
  DWO = std::shared_ptr<DWARFCompileUnit>(std::move(DWOContext), DWOCU);
  // Share .debug_addr and .debug_ranges section with compile unit in .dwo
  if (AddrOffsetSectionBase)
    DWO->setAddrOffsetSection(AddrOffsetSection, *AddrOffsetSectionBase);
  if (getVersion() >= 5) {
    DWO->setRangesSection(&Context.getDWARFObj().getRnglistsDWOSection(), 0);
    DWARFDataExtractor RangesDA(Context.getDWARFObj(), *RangeSection,
                                isLittleEndian, 0);
    if (auto TableOrError = parseListTableHeader<DWARFDebugRnglistTable>(
            RangesDA, RangeSectionBase, Header.getFormat()))
      DWO->RngListTable = TableOrError.get();
    else
      Context.getRecoverableErrorHandler()(createStringError(
          errc::invalid_argument, "parsing a range list table: %s",
          toString(TableOrError.takeError()).c_str()));

    if (DWO->RngListTable)
      DWO->RangeSectionBase = DWO->RngListTable->getHeaderSize();
  } else {
    auto DWORangesBase = UnitDie.getRangesBaseAttribute();
    DWO->setRangesSection(RangeSection, DWORangesBase ? *DWORangesBase : 0);
  }

  return true;
}

void DWARFUnit::clearDIEs(bool KeepCUDie) {
  if (DieArray.size() > (unsigned)KeepCUDie) {
    DieArray.resize((unsigned)KeepCUDie);
    DieArray.shrink_to_fit();
  }
}

Expected<DWARFAddressRangesVector>
DWARFUnit::findRnglistFromOffset(uint64_t Offset) {
  if (getVersion() <= 4) {
    DWARFDebugRangeList RangeList;
    if (Error E = extractRangeList(Offset, RangeList))
      return std::move(E);
    return RangeList.getAbsoluteRanges(getBaseAddress());
  }
  if (RngListTable) {
    DWARFDataExtractor RangesData(Context.getDWARFObj(), *RangeSection,
                                  isLittleEndian, RngListTable->getAddrSize());
    auto RangeListOrError = RngListTable->findList(RangesData, Offset);
    if (RangeListOrError)
      return RangeListOrError.get().getAbsoluteRanges(getBaseAddress(), *this);
    return RangeListOrError.takeError();
  }

  return createStringError(errc::invalid_argument,
                           "missing or invalid range list table");
}

Expected<DWARFAddressRangesVector>
DWARFUnit::findRnglistFromIndex(uint32_t Index) {
  if (auto Offset = getRnglistOffset(Index))
    return findRnglistFromOffset(*Offset);

  if (RngListTable)
    return createStringError(errc::invalid_argument,
                             "invalid range list table index %d", Index);

  return createStringError(errc::invalid_argument,
                           "missing or invalid range list table");
}

Expected<DWARFAddressRangesVector> DWARFUnit::collectAddressRanges() {
  DWARFDie UnitDie = getUnitDIE();
  if (!UnitDie)
    return createStringError(errc::invalid_argument, "No unit DIE");

  // First, check if unit DIE describes address ranges for the whole unit.
  auto CUDIERangesOrError = UnitDie.getAddressRanges();
  if (!CUDIERangesOrError)
    return createStringError(errc::invalid_argument,
                             "decoding address ranges: %s",
                             toString(CUDIERangesOrError.takeError()).c_str());
  return *CUDIERangesOrError;
}

Expected<DWARFLocationExpressionsVector>
DWARFUnit::findLoclistFromOffset(uint64_t Offset) {
  DWARFLocationExpressionsVector Result;

  Error InterpretationError = Error::success();

  Error ParseError = getLocationTable().visitAbsoluteLocationList(
      Offset, getBaseAddress(),
      [this](uint32_t Index) { return getAddrOffsetSectionItem(Index); },
      [&](Expected<DWARFLocationExpression> L) {
        if (L)
          Result.push_back(std::move(*L));
        else
          InterpretationError =
              joinErrors(L.takeError(), std::move(InterpretationError));
        return !InterpretationError;
      });

  if (ParseError || InterpretationError)
    return joinErrors(std::move(ParseError), std::move(InterpretationError));

  return Result;
}

void DWARFUnit::updateAddressDieMap(DWARFDie Die) {
  if (Die.isSubroutineDIE()) {
    auto DIERangesOrError = Die.getAddressRanges();
    if (DIERangesOrError) {
      for (const auto &R : DIERangesOrError.get()) {
        // Ignore 0-sized ranges.
        if (R.LowPC == R.HighPC)
          continue;
        auto B = AddrDieMap.upper_bound(R.LowPC);
        if (B != AddrDieMap.begin() && R.LowPC < (--B)->second.first) {
          // The range is a sub-range of existing ranges, we need to split the
          // existing range.
          if (R.HighPC < B->second.first)
            AddrDieMap[R.HighPC] = B->second;
          if (R.LowPC > B->first)
            AddrDieMap[B->first].first = R.LowPC;
        }
        AddrDieMap[R.LowPC] = std::make_pair(R.HighPC, Die);
      }
    } else
      llvm::consumeError(DIERangesOrError.takeError());
  }
  // Parent DIEs are added to the AddrDieMap prior to the Children DIEs to
  // simplify the logic to update AddrDieMap. The child's range will always
  // be equal or smaller than the parent's range. With this assumption, when
  // adding one range into the map, it will at most split a range into 3
  // sub-ranges.
  for (DWARFDie Child = Die.getFirstChild(); Child; Child = Child.getSibling())
    updateAddressDieMap(Child);
}

DWARFDie DWARFUnit::getSubroutineForAddress(uint64_t Address) {
  extractDIEsIfNeeded(false);
  if (AddrDieMap.empty())
    updateAddressDieMap(getUnitDIE());
  auto R = AddrDieMap.upper_bound(Address);
  if (R == AddrDieMap.begin())
    return DWARFDie();
  // upper_bound's previous item contains Address.
  --R;
  if (Address >= R->second.first)
    return DWARFDie();
  return R->second.second;
}

void
DWARFUnit::getInlinedChainForAddress(uint64_t Address,
                                     SmallVectorImpl<DWARFDie> &InlinedChain) {
  assert(InlinedChain.empty());
  // Try to look for subprogram DIEs in the DWO file.
  parseDWO();
  // First, find the subroutine that contains the given address (the leaf
  // of inlined chain).
  DWARFDie SubroutineDIE =
      (DWO ? *DWO : *this).getSubroutineForAddress(Address);

  if (!SubroutineDIE)
    return;

  while (!SubroutineDIE.isSubprogramDIE()) {
    if (SubroutineDIE.getTag() == DW_TAG_inlined_subroutine)
      InlinedChain.push_back(SubroutineDIE);
    SubroutineDIE  = SubroutineDIE.getParent();
  }
  InlinedChain.push_back(SubroutineDIE);
}

const DWARFUnitIndex &llvm::getDWARFUnitIndex(DWARFContext &Context,
                                              DWARFSectionKind Kind) {
  if (Kind == DW_SECT_INFO)
    return Context.getCUIndex();
  assert(Kind == DW_SECT_TYPES);
  return Context.getTUIndex();
}

DWARFDie DWARFUnit::getParent(const DWARFDebugInfoEntry *Die) {
  if (!Die)
    return DWARFDie();
  const uint32_t Depth = Die->getDepth();
  // Unit DIEs always have a depth of zero and never have parents.
  if (Depth == 0)
    return DWARFDie();
  // Depth of 1 always means parent is the compile/type unit.
  if (Depth == 1)
    return getUnitDIE();
  // Look for previous DIE with a depth that is one less than the Die's depth.
  const uint32_t ParentDepth = Depth - 1;
  for (uint32_t I = getDIEIndex(Die) - 1; I > 0; --I) {
    if (DieArray[I].getDepth() == ParentDepth)
      return DWARFDie(this, &DieArray[I]);
  }
  return DWARFDie();
}

DWARFDie DWARFUnit::getSibling(const DWARFDebugInfoEntry *Die) {
  if (!Die)
    return DWARFDie();
  uint32_t Depth = Die->getDepth();
  // Unit DIEs always have a depth of zero and never have siblings.
  if (Depth == 0)
    return DWARFDie();
  // NULL DIEs don't have siblings.
  if (Die->getAbbreviationDeclarationPtr() == nullptr)
    return DWARFDie();

  // Find the next DIE whose depth is the same as the Die's depth.
  for (size_t I = getDIEIndex(Die) + 1, EndIdx = DieArray.size(); I < EndIdx;
       ++I) {
    if (DieArray[I].getDepth() == Depth)
      return DWARFDie(this, &DieArray[I]);
  }
  return DWARFDie();
}

DWARFDie DWARFUnit::getPreviousSibling(const DWARFDebugInfoEntry *Die) {
  if (!Die)
    return DWARFDie();
  uint32_t Depth = Die->getDepth();
  // Unit DIEs always have a depth of zero and never have siblings.
  if (Depth == 0)
    return DWARFDie();

  // Find the previous DIE whose depth is the same as the Die's depth.
  for (size_t I = getDIEIndex(Die); I > 0;) {
    --I;
    if (DieArray[I].getDepth() == Depth - 1)
      return DWARFDie();
    if (DieArray[I].getDepth() == Depth)
      return DWARFDie(this, &DieArray[I]);
  }
  return DWARFDie();
}

DWARFDie DWARFUnit::getFirstChild(const DWARFDebugInfoEntry *Die) {
  if (!Die->hasChildren())
    return DWARFDie();

  // We do not want access out of bounds when parsing corrupted debug data.
  size_t I = getDIEIndex(Die) + 1;
  if (I >= DieArray.size())
    return DWARFDie();
  return DWARFDie(this, &DieArray[I]);
}

DWARFDie DWARFUnit::getLastChild(const DWARFDebugInfoEntry *Die) {
  if (!Die->hasChildren())
    return DWARFDie();

  uint32_t Depth = Die->getDepth();
  for (size_t I = getDIEIndex(Die) + 1, EndIdx = DieArray.size(); I < EndIdx;
       ++I) {
    if (DieArray[I].getDepth() == Depth + 1 &&
        DieArray[I].getTag() == dwarf::DW_TAG_null)
      return DWARFDie(this, &DieArray[I]);
    assert(DieArray[I].getDepth() > Depth && "Not processing children?");
  }
  return DWARFDie();
}

const DWARFAbbreviationDeclarationSet *DWARFUnit::getAbbreviations() const {
  if (!Abbrevs)
    Abbrevs = Abbrev->getAbbreviationDeclarationSet(Header.getAbbrOffset());
  return Abbrevs;
}

llvm::Optional<object::SectionedAddress> DWARFUnit::getBaseAddress() {
  if (BaseAddr)
    return BaseAddr;

  DWARFDie UnitDie = getUnitDIE();
  Optional<DWARFFormValue> PC = UnitDie.find({DW_AT_low_pc, DW_AT_entry_pc});
  BaseAddr = toSectionedAddress(PC);
  return BaseAddr;
}

Expected<StrOffsetsContributionDescriptor>
StrOffsetsContributionDescriptor::validateContributionSize(
    DWARFDataExtractor &DA) {
  uint8_t EntrySize = getDwarfOffsetByteSize();
  // In order to ensure that we don't read a partial record at the end of
  // the section we validate for a multiple of the entry size.
  uint64_t ValidationSize = alignTo(Size, EntrySize);
  // Guard against overflow.
  if (ValidationSize >= Size)
    if (DA.isValidOffsetForDataOfSize((uint32_t)Base, ValidationSize))
      return *this;
  return createStringError(errc::invalid_argument, "length exceeds section size");
}

// Look for a DWARF64-formatted contribution to the string offsets table
// starting at a given offset and record it in a descriptor.
static Expected<StrOffsetsContributionDescriptor>
parseDWARF64StringOffsetsTableHeader(DWARFDataExtractor &DA, uint64_t Offset) {
  if (!DA.isValidOffsetForDataOfSize(Offset, 16))
    return createStringError(errc::invalid_argument, "section offset exceeds section size");

  if (DA.getU32(&Offset) != dwarf::DW_LENGTH_DWARF64)
    return createStringError(errc::invalid_argument, "32 bit contribution referenced from a 64 bit unit");

  uint64_t Size = DA.getU64(&Offset);
  uint8_t Version = DA.getU16(&Offset);
  (void)DA.getU16(&Offset); // padding
  // The encoded length includes the 2-byte version field and the 2-byte
  // padding, so we need to subtract them out when we populate the descriptor.
  return StrOffsetsContributionDescriptor(Offset, Size - 4, Version, DWARF64);
}

// Look for a DWARF32-formatted contribution to the string offsets table
// starting at a given offset and record it in a descriptor.
static Expected<StrOffsetsContributionDescriptor>
parseDWARF32StringOffsetsTableHeader(DWARFDataExtractor &DA, uint64_t Offset) {
  if (!DA.isValidOffsetForDataOfSize(Offset, 8))
    return createStringError(errc::invalid_argument, "section offset exceeds section size");

  uint32_t ContributionSize = DA.getU32(&Offset);
  if (ContributionSize >= dwarf::DW_LENGTH_lo_reserved)
    return createStringError(errc::invalid_argument, "invalid length");

  uint8_t Version = DA.getU16(&Offset);
  (void)DA.getU16(&Offset); // padding
  // The encoded length includes the 2-byte version field and the 2-byte
  // padding, so we need to subtract them out when we populate the descriptor.
  return StrOffsetsContributionDescriptor(Offset, ContributionSize - 4, Version,
                                          DWARF32);
}

static Expected<StrOffsetsContributionDescriptor>
parseDWARFStringOffsetsTableHeader(DWARFDataExtractor &DA,
                                   llvm::dwarf::DwarfFormat Format,
                                   uint64_t Offset) {
  StrOffsetsContributionDescriptor Desc;
  switch (Format) {
  case dwarf::DwarfFormat::DWARF64: {
    if (Offset < 16)
      return createStringError(errc::invalid_argument, "insufficient space for 64 bit header prefix");
    auto DescOrError = parseDWARF64StringOffsetsTableHeader(DA, Offset - 16);
    if (!DescOrError)
      return DescOrError.takeError();
    Desc = *DescOrError;
    break;
  }
  case dwarf::DwarfFormat::DWARF32: {
    if (Offset < 8)
      return createStringError(errc::invalid_argument, "insufficient space for 32 bit header prefix");
    auto DescOrError = parseDWARF32StringOffsetsTableHeader(DA, Offset - 8);
    if (!DescOrError)
      return DescOrError.takeError();
    Desc = *DescOrError;
    break;
  }
  }
  return Desc.validateContributionSize(DA);
}

Expected<Optional<StrOffsetsContributionDescriptor>>
DWARFUnit::determineStringOffsetsTableContribution(DWARFDataExtractor &DA) {
  uint64_t Offset;
  if (IsDWO) {
    Offset = 0;
    if (DA.getData().data() == nullptr)
      return None;
  } else {
    auto OptOffset = toSectionOffset(getUnitDIE().find(DW_AT_str_offsets_base));
    if (!OptOffset)
      return None;
    Offset = *OptOffset;
  }
  auto DescOrError = parseDWARFStringOffsetsTableHeader(DA, Header.getFormat(), Offset);
  if (!DescOrError)
    return DescOrError.takeError();
  return *DescOrError;
}

Expected<Optional<StrOffsetsContributionDescriptor>>
DWARFUnit::determineStringOffsetsTableContributionDWO(DWARFDataExtractor & DA) {
  uint64_t Offset = 0;
  auto IndexEntry = Header.getIndexEntry();
  const auto *C =
      IndexEntry ? IndexEntry->getOffset(DW_SECT_STR_OFFSETS) : nullptr;
  if (C)
    Offset = C->Offset;
  if (getVersion() >= 5) {
    if (DA.getData().data() == nullptr)
      return None;
    Offset += Header.getFormat() == dwarf::DwarfFormat::DWARF32 ? 8 : 16;
    // Look for a valid contribution at the given offset.
    auto DescOrError = parseDWARFStringOffsetsTableHeader(DA, Header.getFormat(), Offset);
    if (!DescOrError)
      return DescOrError.takeError();
    return *DescOrError;
  }
  // Prior to DWARF v5, we derive the contribution size from the
  // index table (in a package file). In a .dwo file it is simply
  // the length of the string offsets section.
  if (!IndexEntry)
    return {
        Optional<StrOffsetsContributionDescriptor>(
            {0, StringOffsetSection.Data.size(), 4, DWARF32})};
  if (C)
    return {Optional<StrOffsetsContributionDescriptor>(
        {C->Offset, C->Length, 4, DWARF32})};
  return None;
}

//===- DWARFUnit.cpp ------------------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "llvm/DebugInfo/DWARF/DWARFUnit.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/DebugInfo/DWARF/DWARFAbbreviationDeclaration.h"
#include "llvm/DebugInfo/DWARF/DWARFContext.h"
#include "llvm/DebugInfo/DWARF/DWARFDebugAbbrev.h"
#include "llvm/DebugInfo/DWARF/DWARFDebugInfoEntry.h"
#include "llvm/DebugInfo/DWARF/DWARFDebugRnglists.h"
#include "llvm/DebugInfo/DWARF/DWARFDie.h"
#include "llvm/DebugInfo/DWARF/DWARFFormValue.h"
#include "llvm/Support/DataExtractor.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/WithColor.h"
#include <algorithm>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <utility>
#include <vector>

using namespace llvm;
using namespace dwarf;

void DWARFUnitSectionBase::parse(DWARFContext &C, const DWARFSection &Section) {
  const DWARFObject &D = C.getDWARFObj();
  parseImpl(C, D, Section, C.getDebugAbbrev(), &D.getRangeSection(),
            D.getStringSection(), D.getStringOffsetSection(),
            &D.getAddrSection(), D.getLineSection(), D.isLittleEndian(), false,
            false);
}

void DWARFUnitSectionBase::parseDWO(DWARFContext &C,
                                    const DWARFSection &DWOSection, bool Lazy) {
  const DWARFObject &D = C.getDWARFObj();
  parseImpl(C, D, DWOSection, C.getDebugAbbrevDWO(), &D.getRangeDWOSection(),
            D.getStringDWOSection(), D.getStringOffsetDWOSection(),
            &D.getAddrSection(), D.getLineDWOSection(), C.isLittleEndian(),
            true, Lazy);
}

DWARFUnit::DWARFUnit(DWARFContext &DC, const DWARFSection &Section,
                     const DWARFUnitHeader &Header,
                     const DWARFDebugAbbrev *DA, const DWARFSection *RS,
                     StringRef SS, const DWARFSection &SOS,
                     const DWARFSection *AOS, const DWARFSection &LS, bool LE,
                     bool IsDWO, const DWARFUnitSectionBase &UnitSection)
    : Context(DC), InfoSection(Section), Header(Header), Abbrev(DA),
      RangeSection(RS), LineSection(LS), StringSection(SS),
      StringOffsetSection(SOS),  AddrOffsetSection(AOS), isLittleEndian(LE),
      isDWO(IsDWO), UnitSection(UnitSection) {
  clear();
}

DWARFUnit::~DWARFUnit() = default;

DWARFDataExtractor DWARFUnit::getDebugInfoExtractor() const {
  return DWARFDataExtractor(Context.getDWARFObj(), InfoSection, isLittleEndian,
                            getAddressByteSize());
}

bool DWARFUnit::getAddrOffsetSectionItem(uint32_t Index,
                                                uint64_t &Result) const {
  uint32_t Offset = AddrOffsetSectionBase + Index * getAddressByteSize();
  if (AddrOffsetSection->Data.size() < Offset + getAddressByteSize())
    return false;
  DWARFDataExtractor DA(Context.getDWARFObj(), *AddrOffsetSection,
                        isLittleEndian, getAddressByteSize());
  Result = DA.getRelocatedAddress(&Offset);
  return true;
}

bool DWARFUnit::getStringOffsetSectionItem(uint32_t Index,
                                           uint64_t &Result) const {
  if (!StringOffsetsTableContribution)
    return false;
  unsigned ItemSize = getDwarfStringOffsetsByteSize();
  uint32_t Offset = getStringOffsetsBase() + Index * ItemSize;
  if (StringOffsetSection.Data.size() < Offset + ItemSize)
    return false;
  DWARFDataExtractor DA(Context.getDWARFObj(), StringOffsetSection,
                        isLittleEndian, 0);
  Result = DA.getRelocatedValue(ItemSize, &Offset);
  return true;
}

bool DWARFUnitHeader::extract(DWARFContext &Context,
                              const DWARFDataExtractor &debug_info,
                              uint32_t *offset_ptr,
                              DWARFSectionKind SectionKind,
                              const DWARFUnitIndex *Index) {
  Offset = *offset_ptr;
  IndexEntry = Index ? Index->getFromOffset(*offset_ptr) : nullptr;
  Length = debug_info.getU32(offset_ptr);
  // FIXME: Support DWARF64.
  unsigned SizeOfLength = 4;
  FormParams.Format = DWARF32;
  FormParams.Version = debug_info.getU16(offset_ptr);
  if (FormParams.Version >= 5) {
    UnitType = debug_info.getU8(offset_ptr);
    FormParams.AddrSize = debug_info.getU8(offset_ptr);
    AbbrOffset = debug_info.getU32(offset_ptr);
  } else {
    AbbrOffset = debug_info.getRelocatedValue(4, offset_ptr);
    FormParams.AddrSize = debug_info.getU8(offset_ptr);
    // Fake a unit type based on the section type.  This isn't perfect,
    // but distinguishing compile and type units is generally enough.
    if (SectionKind == DW_SECT_TYPES)
      UnitType = DW_UT_type;
    else
      UnitType = DW_UT_compile;
  }
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
  if (isTypeUnit()) {
    TypeHash = debug_info.getU64(offset_ptr);
    TypeOffset = debug_info.getU32(offset_ptr);
  } else if (UnitType == DW_UT_split_compile || UnitType == DW_UT_skeleton)
    DWOId = debug_info.getU64(offset_ptr);

  // Header fields all parsed, capture the size of this unit header.
  assert(*offset_ptr - Offset <= 255 && "unexpected header size");
  Size = uint8_t(*offset_ptr - Offset);

  // Type offset is unit-relative; should be after the header and before
  // the end of the current unit.
  bool TypeOffsetOK =
      !isTypeUnit()
          ? true
          : TypeOffset >= Size && TypeOffset < getLength() + SizeOfLength;
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
static Expected<DWARFDebugRnglistTable>
parseRngListTableHeader(DWARFDataExtractor &DA, uint32_t Offset) {
  // TODO: Support DWARF64
  // We are expected to be called with Offset 0 or pointing just past the table
  // header, which is 12 bytes long for DWARF32.
  if (Offset > 0) {
    if (Offset < 12U) {
      std::string Buffer;
      raw_string_ostream Stream(Buffer);
      Stream << format(
          "Did not detect a valid range list table with base = 0x%x", Offset);
      return make_error<StringError>(Stream.str(), inconvertibleErrorCode());
    }
    Offset -= 12U;
  }
  llvm::DWARFDebugRnglistTable Table;
  if (Error E = Table.extractHeaderAndOffsets(DA, &Offset))
    return std::move(E);
  return Table;
}

Error DWARFUnit::extractRangeList(uint32_t RangeListOffset,
                                  DWARFDebugRangeList &RangeList) const {
  // Require that compile unit is extracted.
  assert(!DieArray.empty());
  DWARFDataExtractor RangesData(Context.getDWARFObj(), *RangeSection,
                                isLittleEndian, getAddressByteSize());
  uint32_t ActualRangeListOffset = RangeSectionBase + RangeListOffset;
  return RangeList.extract(RangesData, &ActualRangeListOffset);
}

void DWARFUnit::clear() {
  Abbrevs = nullptr;
  BaseAddr.reset();
  RangeSectionBase = 0;
  AddrOffsetSectionBase = 0;
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
  uint32_t DIEOffset = getOffset() + getHeaderSize();
  uint32_t NextCUOffset = getNextUnitOffset();
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
    WithColor::warning() << format("DWARF compile unit extends beyond its "
                                   "bounds cu 0x%8.8x at 0x%8.8x\n",
                                   getOffset(), DIEOffset);
}

size_t DWARFUnit::extractDIEsIfNeeded(bool CUDieOnly) {
  if ((CUDieOnly && !DieArray.empty()) ||
      DieArray.size() > 1)
    return 0; // Already parsed.

  bool HasCUDie = !DieArray.empty();
  extractDIEsToVector(!HasCUDie, !CUDieOnly, DieArray);

  if (DieArray.empty())
    return 0;

  // If CU DIE was just parsed, copy several attribute values from it.
  if (!HasCUDie) {
    DWARFDie UnitDie = getUnitDIE();
    if (Optional<uint64_t> DWOId = toUnsigned(UnitDie.find(DW_AT_GNU_dwo_id)))
      Header.setDWOId(*DWOId);
    if (!isDWO) {
      assert(AddrOffsetSectionBase == 0);
      assert(RangeSectionBase == 0);
      AddrOffsetSectionBase =
          toSectionOffset(UnitDie.find(DW_AT_GNU_addr_base), 0);
      RangeSectionBase = toSectionOffset(UnitDie.find(DW_AT_rnglists_base), 0);
    }

    // In general, in DWARF v5 and beyond we derive the start of the unit's
    // contribution to the string offsets table from the unit DIE's
    // DW_AT_str_offsets_base attribute. Split DWARF units do not use this
    // attribute, so we assume that there is a contribution to the string
    // offsets table starting at offset 0 of the debug_str_offsets.dwo section.
    // In both cases we need to determine the format of the contribution,
    // which may differ from the unit's format.
    uint64_t StringOffsetsContributionBase =
        isDWO ? 0 : toSectionOffset(UnitDie.find(DW_AT_str_offsets_base), 0);
    auto IndexEntry = Header.getIndexEntry();
    if (IndexEntry)
      if (const auto *C = IndexEntry->getOffset(DW_SECT_STR_OFFSETS))
        StringOffsetsContributionBase += C->Offset;

    DWARFDataExtractor DA(Context.getDWARFObj(), StringOffsetSection,
                          isLittleEndian, 0);
    if (isDWO)
      StringOffsetsTableContribution =
          determineStringOffsetsTableContributionDWO(
              DA, StringOffsetsContributionBase);
    else if (getVersion() >= 5)
      StringOffsetsTableContribution = determineStringOffsetsTableContribution(
          DA, StringOffsetsContributionBase);

    // DWARF v5 uses the .debug_rnglists and .debug_rnglists.dwo sections to
    // describe address ranges.
    if (getVersion() >= 5) {
      if (isDWO)
        setRangesSection(&Context.getDWARFObj().getRnglistsDWOSection(), 0);
      else
        setRangesSection(&Context.getDWARFObj().getRnglistsSection(),
                         toSectionOffset(UnitDie.find(DW_AT_rnglists_base), 0));
      // Parse the range list table header. Individual range lists are
      // extracted lazily.
      DWARFDataExtractor RangesDA(Context.getDWARFObj(), *RangeSection,
                                  isLittleEndian, 0);
      if (auto TableOrError =
              parseRngListTableHeader(RangesDA, RangeSectionBase))
        RngListTable = TableOrError.get();
      else
        WithColor::error() << "parsing a range list table: "
                           << toString(TableOrError.takeError())
                           << '\n';

      // In a split dwarf unit, there is no DW_AT_rnglists_base attribute.
      // Adjust RangeSectionBase to point past the table header.
      if (isDWO && RngListTable)
        RangeSectionBase = RngListTable->getHeaderSize();
    }

    // Don't fall back to DW_AT_GNU_ranges_base: it should be ignored for
    // skeleton CU DIE, so that DWARF users not aware of it are not broken.
  }

  return DieArray.size();
}

bool DWARFUnit::parseDWO() {
  if (isDWO)
    return false;
  if (DWO.get())
    return false;
  DWARFDie UnitDie = getUnitDIE();
  if (!UnitDie)
    return false;
  auto DWOFileName = dwarf::toString(UnitDie.find(DW_AT_GNU_dwo_name));
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
  DWO->setAddrOffsetSection(AddrOffsetSection, AddrOffsetSectionBase);
  if (getVersion() >= 5) {
    DWO->setRangesSection(&Context.getDWARFObj().getRnglistsDWOSection(), 0);
    DWARFDataExtractor RangesDA(Context.getDWARFObj(), *RangeSection,
                                isLittleEndian, 0);
    if (auto TableOrError = parseRngListTableHeader(RangesDA, RangeSectionBase))
      DWO->RngListTable = TableOrError.get();
    else
      WithColor::error() << "parsing a range list table: "
                         << toString(TableOrError.takeError())
                         << '\n';
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
DWARFUnit::findRnglistFromOffset(uint32_t Offset) {
  if (getVersion() <= 4) {
    DWARFDebugRangeList RangeList;
    if (Error E = extractRangeList(Offset, RangeList))
      return std::move(E);
    return RangeList.getAbsoluteRanges(getBaseAddress());
  }
  if (RngListTable) {
    DWARFDataExtractor RangesData(Context.getDWARFObj(), *RangeSection,
                                  isLittleEndian, RngListTable->getAddrSize());
    auto RangeListOrError = RngListTable->findRangeList(RangesData, Offset);
    if (RangeListOrError)
      return RangeListOrError.get().getAbsoluteRanges(getBaseAddress());
    return RangeListOrError.takeError();
  }

  return make_error<StringError>("missing or invalid range list table",
                                 inconvertibleErrorCode());
}

Expected<DWARFAddressRangesVector>
DWARFUnit::findRnglistFromIndex(uint32_t Index) {
  if (auto Offset = getRnglistOffset(Index))
    return findRnglistFromOffset(*Offset + RangeSectionBase);

  std::string Buffer;
  raw_string_ostream Stream(Buffer);
  if (RngListTable)
    Stream << format("invalid range list table index %d", Index);
  else
    Stream << "missing or invalid range list table";
  return make_error<StringError>(Stream.str(), inconvertibleErrorCode());
}

void DWARFUnit::collectAddressRanges(DWARFAddressRangesVector &CURanges) {
  DWARFDie UnitDie = getUnitDIE();
  if (!UnitDie)
    return;
  // First, check if unit DIE describes address ranges for the whole unit.
  auto CUDIERangesOrError = UnitDie.getAddressRanges();
  if (CUDIERangesOrError) {
    if (!CUDIERangesOrError.get().empty()) {
      CURanges.insert(CURanges.end(), CUDIERangesOrError.get().begin(),
                      CUDIERangesOrError.get().end());
      return;
    }
  } else
    WithColor::error() << "decoding address ranges: "
                       << toString(CUDIERangesOrError.takeError()) << '\n';

  // This function is usually called if there in no .debug_aranges section
  // in order to produce a compile unit level set of address ranges that
  // is accurate. If the DIEs weren't parsed, then we don't want all dies for
  // all compile units to stay loaded when they weren't needed. So we can end
  // up parsing the DWARF and then throwing them all away to keep memory usage
  // down.
  const bool ClearDIEs = extractDIEsIfNeeded(false) > 1;
  getUnitDIE().collectChildrenAddressRanges(CURanges);

  // Collect address ranges from DIEs in .dwo if necessary.
  bool DWOCreated = parseDWO();
  if (DWO)
    DWO->collectAddressRanges(CURanges);
  if (DWOCreated)
    DWO.reset();

  // Keep memory down by clearing DIEs if this generate function
  // caused them to be parsed.
  if (ClearDIEs)
    clearDIEs(true);
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
      (DWO ? DWO.get() : this)->getSubroutineForAddress(Address);

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

DWARFDie DWARFUnit::getFirstChild(const DWARFDebugInfoEntry *Die) {
  if (!Die->hasChildren())
    return DWARFDie();

  // We do not want access out of bounds when parsing corrupted debug data.
  size_t I = getDIEIndex(Die) + 1;
  if (I >= DieArray.size())
    return DWARFDie();
  return DWARFDie(this, &DieArray[I]);
}

const DWARFAbbreviationDeclarationSet *DWARFUnit::getAbbreviations() const {
  if (!Abbrevs)
    Abbrevs = Abbrev->getAbbreviationDeclarationSet(Header.getAbbrOffset());
  return Abbrevs;
}

llvm::Optional<BaseAddress> DWARFUnit::getBaseAddress() {
  if (BaseAddr)
    return BaseAddr;

  DWARFDie UnitDie = getUnitDIE();
  Optional<DWARFFormValue> PC = UnitDie.find({DW_AT_low_pc, DW_AT_entry_pc});
  if (Optional<uint64_t> Addr = toAddress(PC))
    BaseAddr = {*Addr, PC->getSectionIndex()};

  return BaseAddr;
}

Optional<StrOffsetsContributionDescriptor>
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
  return Optional<StrOffsetsContributionDescriptor>();
}

// Look for a DWARF64-formatted contribution to the string offsets table
// starting at a given offset and record it in a descriptor.
static Optional<StrOffsetsContributionDescriptor>
parseDWARF64StringOffsetsTableHeader(DWARFDataExtractor &DA, uint32_t Offset) {
  if (!DA.isValidOffsetForDataOfSize(Offset, 16))
    return Optional<StrOffsetsContributionDescriptor>();

  if (DA.getU32(&Offset) != 0xffffffff)
    return Optional<StrOffsetsContributionDescriptor>();

  uint64_t Size = DA.getU64(&Offset);
  uint8_t Version = DA.getU16(&Offset);
  (void)DA.getU16(&Offset); // padding
  // The encoded length includes the 2-byte version field and the 2-byte
  // padding, so we need to subtract them out when we populate the descriptor.
  return StrOffsetsContributionDescriptor(Offset, Size - 4, Version, DWARF64);
  //return Optional<StrOffsetsContributionDescriptor>(Descriptor);
}

// Look for a DWARF32-formatted contribution to the string offsets table
// starting at a given offset and record it in a descriptor.
static Optional<StrOffsetsContributionDescriptor>
parseDWARF32StringOffsetsTableHeader(DWARFDataExtractor &DA, uint32_t Offset) {
  if (!DA.isValidOffsetForDataOfSize(Offset, 8))
    return Optional<StrOffsetsContributionDescriptor>();
  uint32_t ContributionSize = DA.getU32(&Offset);
  if (ContributionSize >= 0xfffffff0)
    return Optional<StrOffsetsContributionDescriptor>();
  uint8_t Version = DA.getU16(&Offset);
  (void)DA.getU16(&Offset); // padding
  // The encoded length includes the 2-byte version field and the 2-byte
  // padding, so we need to subtract them out when we populate the descriptor.
  return StrOffsetsContributionDescriptor(Offset, ContributionSize - 4, Version,
                                          DWARF32);
  //return Optional<StrOffsetsContributionDescriptor>(Descriptor);
}

Optional<StrOffsetsContributionDescriptor>
DWARFUnit::determineStringOffsetsTableContribution(DWARFDataExtractor &DA,
                                                   uint64_t Offset) {
  Optional<StrOffsetsContributionDescriptor> Descriptor;
  // Attempt to find a DWARF64 contribution 16 bytes before the base.
  if (Offset >= 16)
    Descriptor =
        parseDWARF64StringOffsetsTableHeader(DA, (uint32_t)Offset - 16);
  // Try to find a DWARF32 contribution 8 bytes before the base.
  if (!Descriptor && Offset >= 8)
    Descriptor = parseDWARF32StringOffsetsTableHeader(DA, (uint32_t)Offset - 8);
  return Descriptor ? Descriptor->validateContributionSize(DA) : Descriptor;
}

Optional<StrOffsetsContributionDescriptor>
DWARFUnit::determineStringOffsetsTableContributionDWO(DWARFDataExtractor &DA,
                                                      uint64_t Offset) {
  if (getVersion() >= 5) {
    // Look for a valid contribution at the given offset.
    auto Descriptor =
        parseDWARF64StringOffsetsTableHeader(DA, (uint32_t)Offset);
    if (!Descriptor)
      Descriptor = parseDWARF32StringOffsetsTableHeader(DA, (uint32_t)Offset);
    return Descriptor ? Descriptor->validateContributionSize(DA) : Descriptor;
  }
  // Prior to DWARF v5, we derive the contribution size from the
  // index table (in a package file). In a .dwo file it is simply
  // the length of the string offsets section.
  uint64_t Size = 0;
  auto IndexEntry = Header.getIndexEntry();
  if (!IndexEntry)
    Size = StringOffsetSection.Data.size();
  else if (const auto *C = IndexEntry->getOffset(DW_SECT_STR_OFFSETS))
    Size = C->Length;
  // Return a descriptor with the given offset as base, version 4 and
  // DWARF32 format.
  //return Optional<StrOffsetsContributionDescriptor>(
      //StrOffsetsContributionDescriptor(Offset, Size, 4, DWARF32));
  return StrOffsetsContributionDescriptor(Offset, Size, 4, DWARF32);
}

//===- DWARFUnit.cpp ------------------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "llvm/DebugInfo/DWARF/DWARFUnit.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/DebugInfo/DWARF/DWARFAbbreviationDeclaration.h"
#include "llvm/DebugInfo/DWARF/DWARFContext.h"
#include "llvm/DebugInfo/DWARF/DWARFDebugAbbrev.h"
#include "llvm/DebugInfo/DWARF/DWARFDebugInfoEntry.h"
#include "llvm/DebugInfo/DWARF/DWARFDie.h"
#include "llvm/DebugInfo/DWARF/DWARFFormValue.h"
#include "llvm/Support/DataExtractor.h"
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

void DWARFUnitSectionBase::parse(DWARFContext &C, const DWARFSection &Section) {
  const DWARFObject &D = C.getDWARFObj();
  parseImpl(C, Section, C.getDebugAbbrev(), &D.getRangeSection(),
            D.getStringSection(), D.getStringOffsetSection(),
            &D.getAddrSection(), D.getLineSection(), D.isLittleEndian(), false,
            false);
}

void DWARFUnitSectionBase::parseDWO(DWARFContext &C,
                                    const DWARFSection &DWOSection, bool Lazy) {
  const DWARFObject &D = C.getDWARFObj();
  parseImpl(C, DWOSection, C.getDebugAbbrevDWO(), &D.getRangeDWOSection(),
            D.getStringDWOSection(), D.getStringOffsetDWOSection(),
            &D.getAddrSection(), D.getLineDWOSection(), C.isLittleEndian(),
            true, Lazy);
}

DWARFUnit::DWARFUnit(DWARFContext &DC, const DWARFSection &Section,
                     const DWARFDebugAbbrev *DA, const DWARFSection *RS,
                     StringRef SS, const DWARFSection &SOS,
                     const DWARFSection *AOS, const DWARFSection &LS, bool LE,
                     bool IsDWO, const DWARFUnitSectionBase &UnitSection,
                     const DWARFUnitIndex::Entry *IndexEntry)
    : Context(DC), InfoSection(Section), Abbrev(DA), RangeSection(RS),
      LineSection(LS), StringSection(SS), StringOffsetSection(SOS),
      AddrOffsetSection(AOS), isLittleEndian(LE), isDWO(IsDWO),
      UnitSection(UnitSection), IndexEntry(IndexEntry) {
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

bool DWARFUnit::extractImpl(DataExtractor debug_info, uint32_t *offset_ptr) {
  Length = debug_info.getU32(offset_ptr);
  // FIXME: Support DWARF64.
  FormParams.Format = DWARF32;
  FormParams.Version = debug_info.getU16(offset_ptr);
  if (FormParams.Version >= 5) {
    UnitType = debug_info.getU8(offset_ptr);
    FormParams.AddrSize = debug_info.getU8(offset_ptr);
    AbbrOffset = debug_info.getU32(offset_ptr);
  } else {
    AbbrOffset = debug_info.getU32(offset_ptr);
    FormParams.AddrSize = debug_info.getU8(offset_ptr);
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

  bool LengthOK = debug_info.isValidOffset(getNextUnitOffset() - 1);
  bool VersionOK = DWARFContext::isSupportedVersion(getVersion());
  bool AddrSizeOK = getAddressByteSize() == 4 || getAddressByteSize() == 8;

  if (!LengthOK || !VersionOK || !AddrSizeOK)
    return false;

  // Keep track of the highest DWARF version we encounter across all units.
  Context.setMaxVersionIfGreater(getVersion());
  return true;
}

bool DWARFUnit::extract(DataExtractor debug_info, uint32_t *offset_ptr) {
  clear();

  Offset = *offset_ptr;

  if (debug_info.isValidOffset(*offset_ptr)) {
    if (extractImpl(debug_info, offset_ptr))
      return true;

    // reset the offset to where we tried to parse from if anything went wrong
    *offset_ptr = Offset;
  }

  return false;
}

bool DWARFUnit::extractRangeList(uint32_t RangeListOffset,
                                 DWARFDebugRangeList &RangeList) const {
  // Require that compile unit is extracted.
  assert(!DieArray.empty());
  DWARFDataExtractor RangesData(Context.getDWARFObj(), *RangeSection,
                                isLittleEndian, getAddressByteSize());
  uint32_t ActualRangeListOffset = RangeSectionBase + RangeListOffset;
  return RangeList.extract(RangesData, &ActualRangeListOffset);
}

void DWARFUnit::clear() {
  Offset = 0;
  Length = 0;
  Abbrevs = nullptr;
  FormParams = DWARFFormParams({0, 0, DWARF32});
  BaseAddr.reset();
  RangeSectionBase = 0;
  AddrOffsetSectionBase = 0;
  clearDIEs(false);
  DWO.reset();
}

const char *DWARFUnit::getCompilationDir() {
  return dwarf::toString(getUnitDIE().find(DW_AT_comp_dir), nullptr);
}

Optional<uint64_t> DWARFUnit::getDWOId() {
  return toUnsigned(getUnitDIE().find(DW_AT_GNU_dwo_id));
}

void DWARFUnit::extractDIEsToVector(
    bool AppendCUDie, bool AppendNonCUDies,
    std::vector<DWARFDebugInfoEntry> &Dies) const {
  if (!AppendCUDie && !AppendNonCUDies)
    return;

  // Set the offset to that of the first DIE and calculate the start of the
  // next compilation unit header.
  uint32_t DIEOffset = Offset + getHeaderSize();
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
    fprintf(stderr, "warning: DWARF compile unit extends beyond its "
                    "bounds cu 0x%8.8x at 0x%8.8x'\n", getOffset(), DIEOffset);
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
    Optional<DWARFFormValue> PC = UnitDie.find({DW_AT_low_pc, DW_AT_entry_pc});
    if (Optional<uint64_t> Addr = toAddress(PC))
        setBaseAddress({*Addr, PC->getSectionIndex()});

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
  auto DWORangesBase = UnitDie.getRangesBaseAttribute();
  DWO->setRangesSection(RangeSection, DWORangesBase ? *DWORangesBase : 0);
  return true;
}

void DWARFUnit::clearDIEs(bool KeepCUDie) {
  if (DieArray.size() > (unsigned)KeepCUDie) {
    DieArray.resize((unsigned)KeepCUDie);
    DieArray.shrink_to_fit();
  }
}

void DWARFUnit::collectAddressRanges(DWARFAddressRangesVector &CURanges) {
  DWARFDie UnitDie = getUnitDIE();
  if (!UnitDie)
    return;
  // First, check if unit DIE describes address ranges for the whole unit.
  const auto &CUDIERanges = UnitDie.getAddressRanges();
  if (!CUDIERanges.empty()) {
    CURanges.insert(CURanges.end(), CUDIERanges.begin(), CUDIERanges.end());
    return;
  }

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

// Populates a map from PC addresses to subprogram DIEs.
//
// This routine tries to look at the smallest amount of the debug info it can
// to locate the DIEs. This is because many subprograms will never end up being
// read or needed at all. We want to be as lazy as possible.
void DWARFUnit::buildSubprogramDIEAddrMap() {
  assert(SubprogramDIEAddrMap.empty() && "Must only build this map once!");
  SmallVector<DWARFDie, 16> Worklist;
  Worklist.push_back(getUnitDIE());
  do {
    DWARFDie Die = Worklist.pop_back_val();

    // Queue up child DIEs to recurse through.
    // FIXME: This causes us to read a lot more debug info than we really need.
    // We should look at pruning out DIEs which cannot transitively hold
    // separate subprograms.
    for (DWARFDie Child : Die.children())
      Worklist.push_back(Child);

    // If handling a non-subprogram DIE, nothing else to do.
    if (!Die.isSubprogramDIE())
      continue;

    // For subprogram DIEs, store them, and insert relevant markers into the
    // address map. We don't care about overlap at all here as DWARF doesn't
    // meaningfully support that, so we simply will insert a range with no DIE
    // starting from the high PC. In the event there are overlaps, sorting
    // these may truncate things in surprising ways but still will allow
    // lookups to proceed.
    int DIEIndex = SubprogramDIEAddrInfos.size();
    SubprogramDIEAddrInfos.push_back({Die, (uint64_t)-1, {}});
    for (const auto &R : Die.getAddressRanges()) {
      // Ignore 0-sized ranges.
      if (R.LowPC == R.HighPC)
        continue;

      SubprogramDIEAddrMap.push_back({R.LowPC, DIEIndex});
      SubprogramDIEAddrMap.push_back({R.HighPC, -1});

      if (R.LowPC < SubprogramDIEAddrInfos.back().SubprogramBasePC)
        SubprogramDIEAddrInfos.back().SubprogramBasePC = R.LowPC;
    }
  } while (!Worklist.empty());

  if (SubprogramDIEAddrMap.empty()) {
    // If we found no ranges, create a no-op map so that lookups remain simple
    // but never find anything.
    SubprogramDIEAddrMap.push_back({0, -1});
    return;
  }

  // Next, sort the ranges and remove both exact duplicates and runs with the
  // same DIE index. We order the ranges so that non-empty ranges are
  // preferred. Because there may be ties, we also need to use stable sort.
  std::stable_sort(SubprogramDIEAddrMap.begin(), SubprogramDIEAddrMap.end(),
                   [](const std::pair<uint64_t, int64_t> &LHS,
                      const std::pair<uint64_t, int64_t> &RHS) {
                     if (LHS.first < RHS.first)
                       return true;
                     if (LHS.first > RHS.first)
                       return false;

                     // For ranges that start at the same address, keep the one
                     // with a DIE.
                     if (LHS.second != -1 && RHS.second == -1)
                       return true;

                     return false;
                   });
  SubprogramDIEAddrMap.erase(
      std::unique(SubprogramDIEAddrMap.begin(), SubprogramDIEAddrMap.end(),
                  [](const std::pair<uint64_t, int64_t> &LHS,
                     const std::pair<uint64_t, int64_t> &RHS) {
                    // If the start addresses are exactly the same, we can
                    // remove all but the first one as it is the only one that
                    // will be found and used.
                    //
                    // If the DIE indices are the same, we can "merge" the
                    // ranges by eliminating the second.
                    return LHS.first == RHS.first || LHS.second == RHS.second;
                  }),
      SubprogramDIEAddrMap.end());

  assert(SubprogramDIEAddrMap.back().second == -1 &&
         "The last interval must not have a DIE as each DIE's address range is "
         "bounded.");
}

// Build the second level of mapping from PC to DIE, specifically one that maps
// a PC *within* a particular DWARF subprogram into a precise, maximally nested
// inlined subroutine DIE (if any exists). We build a separate map for each
// subprogram because many subprograms will never get queried for an address
// and this allows us to be significantly lazier in reading the DWARF itself.
void DWARFUnit::buildInlinedSubroutineDIEAddrMap(
    SubprogramDIEAddrInfo &SPInfo) {
  auto &AddrMap = SPInfo.InlinedSubroutineDIEAddrMap;
  uint64_t BasePC = SPInfo.SubprogramBasePC;

  auto SubroutineAddrMapSorter = [](const std::pair<int, int> &LHS,
                                    const std::pair<int, int> &RHS) {
    if (LHS.first < RHS.first)
      return true;
    if (LHS.first > RHS.first)
      return false;

    // For ranges that start at the same address, keep the
    // non-empty one.
    if (LHS.second != -1 && RHS.second == -1)
      return true;

    return false;
  };
  auto SubroutineAddrMapUniquer = [](const std::pair<int, int> &LHS,
                                     const std::pair<int, int> &RHS) {
    // If the start addresses are exactly the same, we can
    // remove all but the first one as it is the only one that
    // will be found and used.
    //
    // If the DIE indices are the same, we can "merge" the
    // ranges by eliminating the second.
    return LHS.first == RHS.first || LHS.second == RHS.second;
  };

  struct DieAndParentIntervalRange {
    DWARFDie Die;
    int ParentIntervalsBeginIdx, ParentIntervalsEndIdx;
  };

  SmallVector<DieAndParentIntervalRange, 16> Worklist;
  auto EnqueueChildDIEs = [&](const DWARFDie &Die, int ParentIntervalsBeginIdx,
                              int ParentIntervalsEndIdx) {
    for (DWARFDie Child : Die.children())
      Worklist.push_back(
          {Child, ParentIntervalsBeginIdx, ParentIntervalsEndIdx});
  };
  EnqueueChildDIEs(SPInfo.SubprogramDIE, 0, 0);
  while (!Worklist.empty()) {
    DWARFDie Die = Worklist.back().Die;
    int ParentIntervalsBeginIdx = Worklist.back().ParentIntervalsBeginIdx;
    int ParentIntervalsEndIdx = Worklist.back().ParentIntervalsEndIdx;
    Worklist.pop_back();

    // If we encounter a nested subprogram, simply ignore it. We map to
    // (disjoint) subprograms before arriving here and we don't want to examine
    // any inlined subroutines of an unrelated subpragram.
    if (Die.getTag() == DW_TAG_subprogram)
      continue;

    // For non-subroutines, just recurse to keep searching for inlined
    // subroutines.
    if (Die.getTag() != DW_TAG_inlined_subroutine) {
      EnqueueChildDIEs(Die, ParentIntervalsBeginIdx, ParentIntervalsEndIdx);
      continue;
    }

    // Capture the inlined subroutine DIE that we will reference from the map.
    int DIEIndex = InlinedSubroutineDIEs.size();
    InlinedSubroutineDIEs.push_back(Die);

    int DieIntervalsBeginIdx = AddrMap.size();
    // First collect the PC ranges for this DIE into our subroutine interval
    // map.
    for (auto R : Die.getAddressRanges()) {
      // Clamp the PCs to be above the base.
      R.LowPC = std::max(R.LowPC, BasePC);
      R.HighPC = std::max(R.HighPC, BasePC);
      // Compute relative PCs from the subprogram base and drop down to an
      // unsigned 32-bit int to represent them within the data structure. This
      // lets us cover a 4gb single subprogram. Because subprograms may be
      // partitioned into distant parts of a binary (think hot/cold
      // partitioning) we want to preserve as much as we can here without
      // burning extra memory. Past that, we will simply truncate and lose the
      // ability to map those PCs to a DIE more precise than the subprogram.
      const uint32_t MaxRelativePC = std::numeric_limits<uint32_t>::max();
      uint32_t RelativeLowPC = (R.LowPC - BasePC) > (uint64_t)MaxRelativePC
                                   ? MaxRelativePC
                                   : (uint32_t)(R.LowPC - BasePC);
      uint32_t RelativeHighPC = (R.HighPC - BasePC) > (uint64_t)MaxRelativePC
                                    ? MaxRelativePC
                                    : (uint32_t)(R.HighPC - BasePC);
      // Ignore empty or bogus ranges.
      if (RelativeLowPC >= RelativeHighPC)
        continue;
      AddrMap.push_back({RelativeLowPC, DIEIndex});
      AddrMap.push_back({RelativeHighPC, -1});
    }

    // If there are no address ranges, there is nothing to do to map into them
    // and there cannot be any child subroutine DIEs with address ranges of
    // interest as those would all be required to nest within this DIE's
    // non-existent ranges, so we can immediately continue to the next DIE in
    // the worklist.
    if (DieIntervalsBeginIdx == (int)AddrMap.size())
      continue;

    // The PCs from this DIE should never overlap, so we can easily sort them
    // here.
    std::sort(AddrMap.begin() + DieIntervalsBeginIdx, AddrMap.end(),
              SubroutineAddrMapSorter);
    // Remove any dead ranges. These should only come from "empty" ranges that
    // were clobbered by some other range.
    AddrMap.erase(std::unique(AddrMap.begin() + DieIntervalsBeginIdx,
                              AddrMap.end(), SubroutineAddrMapUniquer),
                  AddrMap.end());

    // Compute the end index of this DIE's addr map intervals.
    int DieIntervalsEndIdx = AddrMap.size();

    assert(DieIntervalsBeginIdx != DieIntervalsEndIdx &&
           "Must not have an empty map for this layer!");
    assert(AddrMap.back().second == -1 && "Must end with an empty range!");
    assert(std::is_sorted(AddrMap.begin() + DieIntervalsBeginIdx, AddrMap.end(),
                          less_first()) &&
           "Failed to sort this DIE's interals!");

    // If we have any parent intervals, walk the newly added ranges and find
    // the parent ranges they were inserted into. Both of these are sorted and
    // neither has any overlaps. We need to append new ranges to split up any
    // parent ranges these new ranges would overlap when we merge them.
    if (ParentIntervalsBeginIdx != ParentIntervalsEndIdx) {
      int ParentIntervalIdx = ParentIntervalsBeginIdx;
      for (int i = DieIntervalsBeginIdx, e = DieIntervalsEndIdx - 1; i < e;
           ++i) {
        const uint32_t IntervalStart = AddrMap[i].first;
        const uint32_t IntervalEnd = AddrMap[i + 1].first;
        const int IntervalDieIdx = AddrMap[i].second;
        if (IntervalDieIdx == -1) {
          // For empty intervals, nothing is required. This is a bit surprising
          // however. If the prior interval overlaps a parent interval and this
          // would be necessary to mark the end, we will synthesize a new end
          // that switches back to the parent DIE below. And this interval will
          // get dropped in favor of one with a DIE attached. However, we'll
          // still include this and so worst-case, it will still end the prior
          // interval.
          continue;
        }

        // We are walking the new ranges in order, so search forward from the
        // last point for a parent range that might overlap.
        auto ParentIntervalsRange =
            make_range(AddrMap.begin() + ParentIntervalIdx,
                       AddrMap.begin() + ParentIntervalsEndIdx);
        assert(std::is_sorted(ParentIntervalsRange.begin(),
                              ParentIntervalsRange.end(), less_first()) &&
               "Unsorted parent intervals can't be searched!");
        auto PI = std::upper_bound(
            ParentIntervalsRange.begin(), ParentIntervalsRange.end(),
            IntervalStart,
            [](uint32_t LHS, const std::pair<uint32_t, int32_t> &RHS) {
              return LHS < RHS.first;
            });
        if (PI == ParentIntervalsRange.begin() ||
            PI == ParentIntervalsRange.end())
          continue;

        ParentIntervalIdx = PI - AddrMap.begin();
        int32_t &ParentIntervalDieIdx = std::prev(PI)->second;
        uint32_t &ParentIntervalStart = std::prev(PI)->first;
        const uint32_t ParentIntervalEnd = PI->first;

        // If the new range starts exactly at the position of the parent range,
        // we need to adjust the parent range. Note that these collisions can
        // only happen with the original parent range because we will merge any
        // adjacent ranges in the child.
        if (IntervalStart == ParentIntervalStart) {
          // If there will be a tail, just shift the start of the parent
          // forward. Note that this cannot change the parent ordering.
          if (IntervalEnd < ParentIntervalEnd) {
            ParentIntervalStart = IntervalEnd;
            continue;
          }
          // Otherwise, mark this as becoming empty so we'll remove it and
          // prefer the child range.
          ParentIntervalDieIdx = -1;
          continue;
        }

        // Finally, if the parent interval will need to remain as a prefix to
        // this one, insert a new interval to cover any tail.
        if (IntervalEnd < ParentIntervalEnd)
          AddrMap.push_back({IntervalEnd, ParentIntervalDieIdx});
      }
    }

    // Note that we don't need to re-sort even this DIE's address map intervals
    // after this. All of the newly added intervals actually fill in *gaps* in
    // this DIE's address map, and we know that children won't need to lookup
    // into those gaps.

    // Recurse through its children, giving them the interval map range of this
    // DIE to use as their parent intervals.
    EnqueueChildDIEs(Die, DieIntervalsBeginIdx, DieIntervalsEndIdx);
  }

  if (AddrMap.empty()) {
    AddrMap.push_back({0, -1});
    return;
  }

  // Now that we've added all of the intervals needed, we need to resort and
  // unique them. Most notably, this will remove all the empty ranges that had
  // a parent range covering, etc. We only expect a single non-empty interval
  // at any given start point, so we just use std::sort. This could potentially
  // produce non-deterministic maps for invalid DWARF.
  std::sort(AddrMap.begin(), AddrMap.end(), SubroutineAddrMapSorter);
  AddrMap.erase(
      std::unique(AddrMap.begin(), AddrMap.end(), SubroutineAddrMapUniquer),
      AddrMap.end());
}

DWARFDie DWARFUnit::getSubroutineForAddress(uint64_t Address) {
  extractDIEsIfNeeded(false);

  // We use a two-level mapping structure to locate subroutines for a given PC
  // address.
  //
  // First, we map the address to a subprogram. This can be done more cheaply
  // because subprograms cannot nest within each other. It also allows us to
  // avoid detailed examination of many subprograms, instead only focusing on
  // the ones which we end up actively querying.
  if (SubprogramDIEAddrMap.empty())
    buildSubprogramDIEAddrMap();

  assert(!SubprogramDIEAddrMap.empty() &&
         "We must always end up with a non-empty map!");

  auto I = std::upper_bound(
      SubprogramDIEAddrMap.begin(), SubprogramDIEAddrMap.end(), Address,
      [](uint64_t LHS, const std::pair<uint64_t, int64_t> &RHS) {
        return LHS < RHS.first;
      });
  // If we find the beginning, then the address is before the first subprogram.
  if (I == SubprogramDIEAddrMap.begin())
    return DWARFDie();
  // Back up to the interval containing the address and see if it
  // has a DIE associated with it.
  --I;
  if (I->second == -1)
    return DWARFDie();

  auto &SPInfo = SubprogramDIEAddrInfos[I->second];

  // Now that we have the subprogram for this address, we do the second level
  // mapping by building a map within a subprogram's PC range to any specific
  // inlined subroutine.
  if (SPInfo.InlinedSubroutineDIEAddrMap.empty())
    buildInlinedSubroutineDIEAddrMap(SPInfo);

  // We lookup within the inlined subroutine using a subprogram-relative
  // address.
  assert(Address >= SPInfo.SubprogramBasePC &&
         "Address isn't above the start of the subprogram!");
  uint32_t RelativeAddr = ((Address - SPInfo.SubprogramBasePC) >
                           (uint64_t)std::numeric_limits<uint32_t>::max())
                              ? std::numeric_limits<uint32_t>::max()
                              : (uint32_t)(Address - SPInfo.SubprogramBasePC);

  auto J =
      std::upper_bound(SPInfo.InlinedSubroutineDIEAddrMap.begin(),
                       SPInfo.InlinedSubroutineDIEAddrMap.end(), RelativeAddr,
                       [](uint32_t LHS, const std::pair<uint32_t, int32_t> &RHS) {
                         return LHS < RHS.first;
                       });
  // If we find the beginning, the address is before any inlined subroutine so
  // return the subprogram DIE.
  if (J == SPInfo.InlinedSubroutineDIEAddrMap.begin())
    return SPInfo.SubprogramDIE;
  // Back up `J` and return the inlined subroutine if we have one or the
  // subprogram if we don't.
  --J;
  return J->second == -1 ? SPInfo.SubprogramDIE
                         : InlinedSubroutineDIEs[J->second];
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

  while (SubroutineDIE) {
    if (SubroutineDIE.isSubroutineDIE())
      InlinedChain.push_back(SubroutineDIE);
    SubroutineDIE  = SubroutineDIE.getParent();
  }
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
    Abbrevs = Abbrev->getAbbreviationDeclarationSet(AbbrOffset);
  return Abbrevs;
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
  return StrOffsetsContributionDescriptor(Offset, Size, Version, DWARF64);
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
  return StrOffsetsContributionDescriptor(Offset, ContributionSize, Version, DWARF32);
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

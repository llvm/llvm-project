//===-- DWARFCompileUnit.cpp ----------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "DWARFCompileUnit.h"
#include "DWARFContext.h"
#include "DWARFFormValue.h"
#include "llvm/Support/Dwarf.h"
#include "llvm/Support/Format.h"
#include "llvm/Support/raw_ostream.h"
using namespace llvm;
using namespace dwarf;

DataExtractor DWARFCompileUnit::getDebugInfoExtractor() const {
  return DataExtractor(Context.getInfoSection(),
                       Context.isLittleEndian(), getAddressByteSize());
}

bool DWARFCompileUnit::extract(DataExtractor debug_info, uint32_t *offset_ptr) {
  clear();

  Offset = *offset_ptr;

  if (debug_info.isValidOffset(*offset_ptr)) {
    uint64_t abbrOffset;
    const DWARFDebugAbbrev *abbr = Context.getDebugAbbrev();
    Length = debug_info.getU32(offset_ptr);
    Version = debug_info.getU16(offset_ptr);
    abbrOffset = debug_info.getU32(offset_ptr);
    AddrSize = debug_info.getU8(offset_ptr);

    bool lengthOK = debug_info.isValidOffset(getNextCompileUnitOffset()-1);
    bool versionOK = DWARFContext::isSupportedVersion(Version);
    bool abbrOffsetOK = Context.getAbbrevSection().size() > abbrOffset;
    bool addrSizeOK = AddrSize == 4 || AddrSize == 8;

    if (lengthOK && versionOK && addrSizeOK && abbrOffsetOK && abbr != NULL) {
      Abbrevs = abbr->getAbbreviationDeclarationSet(abbrOffset);
      return true;
    }

    // reset the offset to where we tried to parse from if anything went wrong
    *offset_ptr = Offset;
  }

  return false;
}

uint32_t
DWARFCompileUnit::extract(uint32_t offset, DataExtractor debug_info_data,
                          const DWARFAbbreviationDeclarationSet *abbrevs) {
  clear();

  Offset = offset;

  if (debug_info_data.isValidOffset(offset)) {
    Length = debug_info_data.getU32(&offset);
    Version = debug_info_data.getU16(&offset);
    bool abbrevsOK = debug_info_data.getU32(&offset) == abbrevs->getOffset();
    Abbrevs = abbrevs;
    AddrSize = debug_info_data.getU8(&offset);

    bool versionOK = DWARFContext::isSupportedVersion(Version);
    bool addrSizeOK = AddrSize == 4 || AddrSize == 8;

    if (versionOK && addrSizeOK && abbrevsOK &&
        debug_info_data.isValidOffset(offset))
      return offset;
  }
  return 0;
}

bool DWARFCompileUnit::extractRangeList(uint32_t RangeListOffset,
                                        DWARFDebugRangeList &RangeList) const {
  // Require that compile unit is extracted.
  assert(DieArray.size() > 0);
  DataExtractor RangesData(Context.getRangeSection(),
                           Context.isLittleEndian(), AddrSize);
  return RangeList.extract(RangesData, &RangeListOffset);
}

void DWARFCompileUnit::clear() {
  Offset = 0;
  Length = 0;
  Version = 0;
  Abbrevs = 0;
  AddrSize = 0;
  BaseAddr = 0;
  clearDIEs(false);
}

void DWARFCompileUnit::dump(raw_ostream &OS) {
  OS << format("0x%08x", Offset) << ": Compile Unit:"
     << " length = " << format("0x%08x", Length)
     << " version = " << format("0x%04x", Version)
     << " abbr_offset = " << format("0x%04x", Abbrevs->getOffset())
     << " addr_size = " << format("0x%02x", AddrSize)
     << " (next CU at " << format("0x%08x", getNextCompileUnitOffset())
     << ")\n";

  const DWARFDebugInfoEntryMinimal *CU = getCompileUnitDIE(false);
  assert(CU && "Null Compile Unit?");
  CU->dump(OS, this, -1U);
}

const char *DWARFCompileUnit::getCompilationDir() {
  extractDIEsIfNeeded(true);
  if (DieArray.empty())
    return 0;
  return DieArray[0].getAttributeValueAsString(this, DW_AT_comp_dir, 0);
}

void DWARFCompileUnit::setDIERelations() {
  if (DieArray.empty())
    return;
  DWARFDebugInfoEntryMinimal *die_array_begin = &DieArray.front();
  DWARFDebugInfoEntryMinimal *die_array_end = &DieArray.back();
  DWARFDebugInfoEntryMinimal *curr_die;
  // We purposely are skipping the last element in the array in the loop below
  // so that we can always have a valid next item
  for (curr_die = die_array_begin; curr_die < die_array_end; ++curr_die) {
    // Since our loop doesn't include the last element, we can always
    // safely access the next die in the array.
    DWARFDebugInfoEntryMinimal *next_die = curr_die + 1;

    const DWARFAbbreviationDeclaration *curr_die_abbrev =
      curr_die->getAbbreviationDeclarationPtr();

    if (curr_die_abbrev) {
      // Normal DIE
      if (curr_die_abbrev->hasChildren())
        next_die->setParent(curr_die);
      else
        curr_die->setSibling(next_die);
    } else {
      // NULL DIE that terminates a sibling chain
      DWARFDebugInfoEntryMinimal *parent = curr_die->getParent();
      if (parent)
        parent->setSibling(next_die);
    }
  }

  // Since we skipped the last element, we need to fix it up!
  if (die_array_begin < die_array_end)
    curr_die->setParent(die_array_begin);
}

size_t DWARFCompileUnit::extractDIEsIfNeeded(bool cu_die_only) {
  const size_t initial_die_array_size = DieArray.size();
  if ((cu_die_only && initial_die_array_size > 0) ||
      initial_die_array_size > 1)
    return 0; // Already parsed

  // Set the offset to that of the first DIE and calculate the start of the
  // next compilation unit header.
  uint32_t offset = getFirstDIEOffset();
  uint32_t next_cu_offset = getNextCompileUnitOffset();

  DWARFDebugInfoEntryMinimal die;
  // Keep a flat array of the DIE for binary lookup by DIE offset
  uint32_t depth = 0;
  // We are in our compile unit, parse starting at the offset
  // we were told to parse

  const uint8_t *fixed_form_sizes =
    DWARFFormValue::getFixedFormSizesForAddressSize(getAddressByteSize());

  while (offset < next_cu_offset &&
         die.extractFast(this, fixed_form_sizes, &offset)) {

    if (depth == 0) {
      uint64_t base_addr =
        die.getAttributeValueAsUnsigned(this, DW_AT_low_pc, -1U);
      if (base_addr == -1U)
        base_addr = die.getAttributeValueAsUnsigned(this, DW_AT_entry_pc, 0);
      setBaseAddress(base_addr);
    }

    if (cu_die_only) {
      addDIE(die);
      return 1;
    }
    else if (depth == 0 && initial_die_array_size == 1)
      // Don't append the CU die as we already did that
      ;
    else
      addDIE(die);

    const DWARFAbbreviationDeclaration *abbrDecl =
      die.getAbbreviationDeclarationPtr();
    if (abbrDecl) {
      // Normal DIE
      if (abbrDecl->hasChildren())
        ++depth;
    } else {
      // NULL DIE.
      if (depth > 0)
        --depth;
      if (depth == 0)
        break;  // We are done with this compile unit!
    }

  }

  // Give a little bit of info if we encounter corrupt DWARF (our offset
  // should always terminate at or before the start of the next compilation
  // unit header).
  if (offset > next_cu_offset)
    fprintf(stderr, "warning: DWARF compile unit extends beyond its"
                    "bounds cu 0x%8.8x at 0x%8.8x'\n", getOffset(), offset);

  setDIERelations();
  return DieArray.size();
}

void DWARFCompileUnit::clearDIEs(bool keep_compile_unit_die) {
  if (DieArray.size() > (unsigned)keep_compile_unit_die) {
    // std::vectors never get any smaller when resized to a smaller size,
    // or when clear() or erase() are called, the size will report that it
    // is smaller, but the memory allocated remains intact (call capacity()
    // to see this). So we need to create a temporary vector and swap the
    // contents which will cause just the internal pointers to be swapped
    // so that when "tmp_array" goes out of scope, it will destroy the
    // contents.

    // Save at least the compile unit DIE
    std::vector<DWARFDebugInfoEntryMinimal> tmpArray;
    DieArray.swap(tmpArray);
    if (keep_compile_unit_die)
      DieArray.push_back(tmpArray.front());
  }
}

void
DWARFCompileUnit::buildAddressRangeTable(DWARFDebugAranges *debug_aranges,
                                         bool clear_dies_if_already_not_parsed){
  // This function is usually called if there in no .debug_aranges section
  // in order to produce a compile unit level set of address ranges that
  // is accurate. If the DIEs weren't parsed, then we don't want all dies for
  // all compile units to stay loaded when they weren't needed. So we can end
  // up parsing the DWARF and then throwing them all away to keep memory usage
  // down.
  const bool clear_dies = extractDIEsIfNeeded(false) > 1 &&
                          clear_dies_if_already_not_parsed;
  DieArray[0].buildAddressRangeTable(this, debug_aranges);

  // Keep memory down by clearing DIEs if this generate function
  // caused them to be parsed.
  if (clear_dies)
    clearDIEs(true);
}

DWARFDebugInfoEntryMinimal::InlinedChain
DWARFCompileUnit::getInlinedChainForAddress(uint64_t Address) {
  // First, find a subprogram that contains the given address (the root
  // of inlined chain).
  extractDIEsIfNeeded(false);
  const DWARFDebugInfoEntryMinimal *SubprogramDIE = 0;
  for (size_t i = 0, n = DieArray.size(); i != n; i++) {
    if (DieArray[i].isSubprogramDIE() &&
        DieArray[i].addressRangeContainsAddress(this, Address)) {
      SubprogramDIE = &DieArray[i];
      break;
    }
  }
  // Get inlined chain rooted at this subprogram DIE.
  if (!SubprogramDIE)
    return DWARFDebugInfoEntryMinimal::InlinedChain();
  return SubprogramDIE->getInlinedChainForAddress(this, Address);
}

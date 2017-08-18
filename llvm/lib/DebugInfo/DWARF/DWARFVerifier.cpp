//===- DWARFVerifier.cpp --------------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "llvm/DebugInfo/DWARF/DWARFVerifier.h"
#include "llvm/DebugInfo/DWARF/DWARFCompileUnit.h"
#include "llvm/DebugInfo/DWARF/DWARFContext.h"
#include "llvm/DebugInfo/DWARF/DWARFDebugLine.h"
#include "llvm/DebugInfo/DWARF/DWARFDie.h"
#include "llvm/DebugInfo/DWARF/DWARFFormValue.h"
#include "llvm/DebugInfo/DWARF/DWARFSection.h"
#include "llvm/DebugInfo/DWARF/DWARFAcceleratorTable.h"
#include "llvm/Support/raw_ostream.h"
#include <map>
#include <set>
#include <vector>

using namespace llvm;
using namespace dwarf;
using namespace object;

bool DWARFVerifier::verifyUnitHeader(const DWARFDataExtractor DebugInfoData,
                                     uint32_t *Offset, unsigned UnitIndex,
                                     uint8_t &UnitType, bool &isUnitDWARF64) {
  uint32_t AbbrOffset, Length;
  uint8_t AddrSize = 0;
  uint16_t Version;
  bool Success = true;

  bool ValidLength = false;
  bool ValidVersion = false;
  bool ValidAddrSize = false;
  bool ValidType = true;
  bool ValidAbbrevOffset = true;

  uint32_t OffsetStart = *Offset;
  Length = DebugInfoData.getU32(Offset);
  if (Length == UINT32_MAX) {
    isUnitDWARF64 = true;
    OS << format(
        "Unit[%d] is in 64-bit DWARF format; cannot verify from this point.\n",
        UnitIndex);
    return false;
  }
  Version = DebugInfoData.getU16(Offset);

  if (Version >= 5) {
    UnitType = DebugInfoData.getU8(Offset);
    AddrSize = DebugInfoData.getU8(Offset);
    AbbrOffset = DebugInfoData.getU32(Offset);
    ValidType = DWARFUnit::isValidUnitType(UnitType);
  } else {
    UnitType = 0;
    AbbrOffset = DebugInfoData.getU32(Offset);
    AddrSize = DebugInfoData.getU8(Offset);
  }

  if (!DCtx.getDebugAbbrev()->getAbbreviationDeclarationSet(AbbrOffset))
    ValidAbbrevOffset = false;

  ValidLength = DebugInfoData.isValidOffset(OffsetStart + Length + 3);
  ValidVersion = DWARFContext::isSupportedVersion(Version);
  ValidAddrSize = AddrSize == 4 || AddrSize == 8;
  if (!ValidLength || !ValidVersion || !ValidAddrSize || !ValidAbbrevOffset ||
      !ValidType) {
    Success = false;
    OS << format("Units[%d] - start offset: 0x%08x \n", UnitIndex, OffsetStart);
    if (!ValidLength)
      OS << "\tError: The length for this unit is too "
            "large for the .debug_info provided.\n";
    if (!ValidVersion)
      OS << "\tError: The 16 bit unit header version is not valid.\n";
    if (!ValidType)
      OS << "\tError: The unit type encoding is not valid.\n";
    if (!ValidAbbrevOffset)
      OS << "\tError: The offset into the .debug_abbrev section is "
            "not valid.\n";
    if (!ValidAddrSize)
      OS << "\tError: The address size is unsupported.\n";
  }
  *Offset = OffsetStart + Length + 4;
  return Success;
}

bool DWARFVerifier::verifyUnitContents(DWARFUnit Unit) {
  uint32_t NumUnitErrors = 0;
  unsigned NumDies = Unit.getNumDIEs();
  for (unsigned I = 0; I < NumDies; ++I) {
    auto Die = Unit.getDIEAtIndex(I);
    if (Die.getTag() == DW_TAG_null)
      continue;
    for (auto AttrValue : Die.attributes()) {
      NumUnitErrors += verifyDebugInfoAttribute(Die, AttrValue);
      NumUnitErrors += verifyDebugInfoForm(Die, AttrValue);
    }
  }
  return NumUnitErrors == 0;
}

bool DWARFVerifier::handleDebugInfo() {
  OS << "Verifying .debug_info Unit Header Chain...\n";

  DWARFDataExtractor DebugInfoData(DCtx.getInfoSection(), DCtx.isLittleEndian(),
                                   0);
  uint32_t NumDebugInfoErrors = 0;
  uint32_t OffsetStart = 0, Offset = 0, UnitIdx = 0;
  uint8_t UnitType = 0;
  bool isUnitDWARF64 = false;
  bool isHeaderChainValid = true;
  bool hasDIE = DebugInfoData.isValidOffset(Offset);
  while (hasDIE) {
    OffsetStart = Offset;
    if (!verifyUnitHeader(DebugInfoData, &Offset, UnitIdx, UnitType,
                          isUnitDWARF64)) {
      isHeaderChainValid = false;
      if (isUnitDWARF64)
        break;
    } else {
      std::unique_ptr<DWARFUnit> Unit;
      switch (UnitType) {
      case dwarf::DW_UT_type:
      case dwarf::DW_UT_split_type: {
        DWARFUnitSection<DWARFTypeUnit> TUSection{};
        Unit.reset(new DWARFTypeUnit(
            DCtx, DCtx.getInfoSection(), DCtx.getDebugAbbrev(),
            &DCtx.getRangeSection(), DCtx.getStringSection(),
            DCtx.getStringOffsetSection(), &DCtx.getAppleObjCSection(),
            DCtx.getLineSection(), DCtx.isLittleEndian(), false, TUSection,
            nullptr));
        break;
      }
      case dwarf::DW_UT_skeleton:
      case dwarf::DW_UT_split_compile:
      case dwarf::DW_UT_compile:
      case dwarf::DW_UT_partial:
      // UnitType = 0 means that we are
      // verifying a compile unit in DWARF v4.
      case 0: {
        DWARFUnitSection<DWARFCompileUnit> CUSection{};
        Unit.reset(new DWARFCompileUnit(
            DCtx, DCtx.getInfoSection(), DCtx.getDebugAbbrev(),
            &DCtx.getRangeSection(), DCtx.getStringSection(),
            DCtx.getStringOffsetSection(), &DCtx.getAppleObjCSection(),
            DCtx.getLineSection(), DCtx.isLittleEndian(), false, CUSection,
            nullptr));
        break;
      }
      default: { llvm_unreachable("Invalid UnitType."); }
      }
      Unit->extract(DebugInfoData, &OffsetStart);
      if (!verifyUnitContents(*Unit))
        ++NumDebugInfoErrors;
    }
    hasDIE = DebugInfoData.isValidOffset(Offset);
    ++UnitIdx;
  }
  if (UnitIdx == 0 && !hasDIE) {
    OS << "Warning: .debug_info is empty.\n";
    isHeaderChainValid = true;
  }
  NumDebugInfoErrors += verifyDebugInfoReferences();
  return (isHeaderChainValid && NumDebugInfoErrors == 0);
}

unsigned DWARFVerifier::verifyDebugInfoAttribute(const DWARFDie &Die,
                                                 DWARFAttribute &AttrValue) {
  unsigned NumErrors = 0;
  const auto Attr = AttrValue.Attr;
  switch (Attr) {
  case DW_AT_ranges:
    // Make sure the offset in the DW_AT_ranges attribute is valid.
    if (auto SectionOffset = AttrValue.Value.getAsSectionOffset()) {
      if (*SectionOffset >= DCtx.getRangeSection().Data.size()) {
        ++NumErrors;
        OS << "error: DW_AT_ranges offset is beyond .debug_ranges "
              "bounds:\n";
        Die.dump(OS, 0);
        OS << "\n";
      }
    } else {
      ++NumErrors;
      OS << "error: DIE has invalid DW_AT_ranges encoding:\n";
      Die.dump(OS, 0);
      OS << "\n";
    }
    break;
  case DW_AT_stmt_list:
    // Make sure the offset in the DW_AT_stmt_list attribute is valid.
    if (auto SectionOffset = AttrValue.Value.getAsSectionOffset()) {
      if (*SectionOffset >= DCtx.getLineSection().Data.size()) {
        ++NumErrors;
        OS << "error: DW_AT_stmt_list offset is beyond .debug_line "
              "bounds: "
           << format("0x%08" PRIx64, *SectionOffset) << "\n";
        Die.dump(OS, 0);
        OS << "\n";
      }
    } else {
      ++NumErrors;
      OS << "error: DIE has invalid DW_AT_stmt_list encoding:\n";
      Die.dump(OS, 0);
      OS << "\n";
    }
    break;

  default:
    break;
  }
  return NumErrors;
}

unsigned DWARFVerifier::verifyDebugInfoForm(const DWARFDie &Die,
                                            DWARFAttribute &AttrValue) {
  unsigned NumErrors = 0;
  const auto Form = AttrValue.Value.getForm();
  switch (Form) {
  case DW_FORM_ref1:
  case DW_FORM_ref2:
  case DW_FORM_ref4:
  case DW_FORM_ref8:
  case DW_FORM_ref_udata: {
    // Verify all CU relative references are valid CU offsets.
    Optional<uint64_t> RefVal = AttrValue.Value.getAsReference();
    assert(RefVal);
    if (RefVal) {
      auto DieCU = Die.getDwarfUnit();
      auto CUSize = DieCU->getNextUnitOffset() - DieCU->getOffset();
      auto CUOffset = AttrValue.Value.getRawUValue();
      if (CUOffset >= CUSize) {
        ++NumErrors;
        OS << "error: " << FormEncodingString(Form) << " CU offset "
           << format("0x%08" PRIx64, CUOffset)
           << " is invalid (must be less than CU size of "
           << format("0x%08" PRIx32, CUSize) << "):\n";
        Die.dump(OS, 0);
        OS << "\n";
      } else {
        // Valid reference, but we will verify it points to an actual
        // DIE later.
        ReferenceToDIEOffsets[*RefVal].insert(Die.getOffset());
      }
    }
    break;
  }
  case DW_FORM_ref_addr: {
    // Verify all absolute DIE references have valid offsets in the
    // .debug_info section.
    Optional<uint64_t> RefVal = AttrValue.Value.getAsReference();
    assert(RefVal);
    if (RefVal) {
      if (*RefVal >= DCtx.getInfoSection().Data.size()) {
        ++NumErrors;
        OS << "error: DW_FORM_ref_addr offset beyond .debug_info "
              "bounds:\n";
        Die.dump(OS, 0);
        OS << "\n";
      } else {
        // Valid reference, but we will verify it points to an actual
        // DIE later.
        ReferenceToDIEOffsets[*RefVal].insert(Die.getOffset());
      }
    }
    break;
  }
  case DW_FORM_strp: {
    auto SecOffset = AttrValue.Value.getAsSectionOffset();
    assert(SecOffset); // DW_FORM_strp is a section offset.
    if (SecOffset && *SecOffset >= DCtx.getStringSection().size()) {
      ++NumErrors;
      OS << "error: DW_FORM_strp offset beyond .debug_str bounds:\n";
      Die.dump(OS, 0);
      OS << "\n";
    }
    break;
  }
  default:
    break;
  }
  return NumErrors;
}

unsigned DWARFVerifier::verifyDebugInfoReferences() {
  // Take all references and make sure they point to an actual DIE by
  // getting the DIE by offset and emitting an error
  OS << "Verifying .debug_info references...\n";
  unsigned NumErrors = 0;
  for (auto Pair : ReferenceToDIEOffsets) {
    auto Die = DCtx.getDIEForOffset(Pair.first);
    if (Die)
      continue;
    ++NumErrors;
    OS << "error: invalid DIE reference " << format("0x%08" PRIx64, Pair.first)
       << ". Offset is in between DIEs:\n";
    for (auto Offset : Pair.second) {
      auto ReferencingDie = DCtx.getDIEForOffset(Offset);
      ReferencingDie.dump(OS, 0);
      OS << "\n";
    }
    OS << "\n";
  }
  return NumErrors;
}

void DWARFVerifier::verifyDebugLineStmtOffsets() {
  std::map<uint64_t, DWARFDie> StmtListToDie;
  for (const auto &CU : DCtx.compile_units()) {
    auto Die = CU->getUnitDIE();
    // Get the attribute value as a section offset. No need to produce an
    // error here if the encoding isn't correct because we validate this in
    // the .debug_info verifier.
    auto StmtSectionOffset = toSectionOffset(Die.find(DW_AT_stmt_list));
    if (!StmtSectionOffset)
      continue;
    const uint32_t LineTableOffset = *StmtSectionOffset;
    auto LineTable = DCtx.getLineTableForUnit(CU.get());
    if (LineTableOffset < DCtx.getLineSection().Data.size()) {
      if (!LineTable) {
        ++NumDebugLineErrors;
        OS << "error: .debug_line[" << format("0x%08" PRIx32, LineTableOffset)
           << "] was not able to be parsed for CU:\n";
        Die.dump(OS, 0);
        OS << '\n';
        continue;
      }
    } else {
      // Make sure we don't get a valid line table back if the offset is wrong.
      assert(LineTable == nullptr);
      // Skip this line table as it isn't valid. No need to create an error
      // here because we validate this in the .debug_info verifier.
      continue;
    }
    auto Iter = StmtListToDie.find(LineTableOffset);
    if (Iter != StmtListToDie.end()) {
      ++NumDebugLineErrors;
      OS << "error: two compile unit DIEs, "
         << format("0x%08" PRIx32, Iter->second.getOffset()) << " and "
         << format("0x%08" PRIx32, Die.getOffset())
         << ", have the same DW_AT_stmt_list section offset:\n";
      Iter->second.dump(OS, 0);
      Die.dump(OS, 0);
      OS << '\n';
      // Already verified this line table before, no need to do it again.
      continue;
    }
    StmtListToDie[LineTableOffset] = Die;
  }
}

void DWARFVerifier::verifyDebugLineRows() {
  for (const auto &CU : DCtx.compile_units()) {
    auto Die = CU->getUnitDIE();
    auto LineTable = DCtx.getLineTableForUnit(CU.get());
    // If there is no line table we will have created an error in the
    // .debug_info verifier or in verifyDebugLineStmtOffsets().
    if (!LineTable)
      continue;
    uint32_t MaxFileIndex = LineTable->Prologue.FileNames.size();
    uint64_t PrevAddress = 0;
    uint32_t RowIndex = 0;
    for (const auto &Row : LineTable->Rows) {
      if (Row.Address < PrevAddress) {
        ++NumDebugLineErrors;
        OS << "error: .debug_line["
           << format("0x%08" PRIx64,
                     *toSectionOffset(Die.find(DW_AT_stmt_list)))
           << "] row[" << RowIndex
           << "] decreases in address from previous row:\n";

        DWARFDebugLine::Row::dumpTableHeader(OS);
        if (RowIndex > 0)
          LineTable->Rows[RowIndex - 1].dump(OS);
        Row.dump(OS);
        OS << '\n';
      }

      if (Row.File > MaxFileIndex) {
        ++NumDebugLineErrors;
        OS << "error: .debug_line["
           << format("0x%08" PRIx64,
                     *toSectionOffset(Die.find(DW_AT_stmt_list)))
           << "][" << RowIndex << "] has invalid file index " << Row.File
           << " (valid values are [1," << MaxFileIndex << "]):\n";
        DWARFDebugLine::Row::dumpTableHeader(OS);
        Row.dump(OS);
        OS << '\n';
      }
      if (Row.EndSequence)
        PrevAddress = 0;
      else
        PrevAddress = Row.Address;
      ++RowIndex;
    }
  }
}

bool DWARFVerifier::handleDebugLine() {
  NumDebugLineErrors = 0;
  OS << "Verifying .debug_line...\n";
  verifyDebugLineStmtOffsets();
  verifyDebugLineRows();
  return NumDebugLineErrors == 0;
}

bool DWARFVerifier::handleAppleNames() {
  NumAppleNamesErrors = 0;

  DWARFDataExtractor AppleNamesSection(DCtx.getAppleNamesSection(),
                                       DCtx.isLittleEndian(), 0);
  DataExtractor StrData(DCtx.getStringSection(), DCtx.isLittleEndian(), 0);
  DWARFAcceleratorTable AppleNames(AppleNamesSection, StrData);

  if (!AppleNames.extract()) {
    return true;
  }

  OS << "Verifying .apple_names...\n";

  // Verify that all buckets have a valid hash index or are empty.
  uint32_t NumBuckets = AppleNames.getNumBuckets();
  uint32_t NumHashes = AppleNames.getNumHashes();

  uint32_t BucketsOffset =
      AppleNames.getSizeHdr() + AppleNames.getHeaderDataLength();
  uint32_t HashesBase = BucketsOffset + NumBuckets * 4;
  uint32_t OffsetsBase = HashesBase + NumHashes * 4;

  for (uint32_t BucketIdx = 0; BucketIdx < NumBuckets; ++BucketIdx) {
    uint32_t HashIdx = AppleNamesSection.getU32(&BucketsOffset);
    if (HashIdx >= NumHashes && HashIdx != UINT32_MAX) {
      OS << format("error: Bucket[%d] has invalid hash index: %u\n", BucketIdx,
                   HashIdx);
      ++NumAppleNamesErrors;
    }
  }

  uint32_t NumAtoms = AppleNames.getAtomsDesc().size();
  if (NumAtoms == 0) {
    OS << "error: no atoms; failed to read HashData\n";
    ++NumAppleNamesErrors;
    return false;
  }

  if (!AppleNames.validateForms()) {
    OS << "error: unsupported form; failed to read HashData\n";
    ++NumAppleNamesErrors;
    return false;
  }

  for (uint32_t HashIdx = 0; HashIdx < NumHashes; ++HashIdx) {
    uint32_t HashOffset = HashesBase + 4 * HashIdx;
    uint32_t DataOffset = OffsetsBase + 4 * HashIdx;
    uint32_t Hash = AppleNamesSection.getU32(&HashOffset);
    uint32_t HashDataOffset = AppleNamesSection.getU32(&DataOffset);
    if (!AppleNamesSection.isValidOffsetForDataOfSize(HashDataOffset,
                                                      sizeof(uint64_t))) {
      OS << format("error: Hash[%d] has invalid HashData offset: 0x%08x\n",
                   HashIdx, HashDataOffset);
      ++NumAppleNamesErrors;
    }

    uint32_t StrpOffset;
    uint32_t StringOffset;
    uint32_t StringCount = 0;
    uint32_t DieOffset = dwarf::DW_INVALID_OFFSET;

    while ((StrpOffset = AppleNamesSection.getU32(&HashDataOffset)) != 0) {
      const uint32_t NumHashDataObjects =
          AppleNamesSection.getU32(&HashDataOffset);
      for (uint32_t HashDataIdx = 0; HashDataIdx < NumHashDataObjects;
           ++HashDataIdx) {
        DieOffset = AppleNames.readAtoms(HashDataOffset);
        if (!DCtx.getDIEForOffset(DieOffset)) {
          const uint32_t BucketIdx =
              NumBuckets ? (Hash % NumBuckets) : UINT32_MAX;
          StringOffset = StrpOffset;
          const char *Name = StrData.getCStr(&StringOffset);
          if (!Name)
            Name = "<NULL>";

          OS << format(
              "error: .apple_names Bucket[%d] Hash[%d] = 0x%08x "
              "Str[%u] = 0x%08x "
              "DIE[%d] = 0x%08x is not a valid DIE offset for \"%s\".\n",
              BucketIdx, HashIdx, Hash, StringCount, StrpOffset, HashDataIdx,
              DieOffset, Name);

          ++NumAppleNamesErrors;
        }
      }
      ++StringCount;
    }
  }
  return NumAppleNamesErrors == 0;
}

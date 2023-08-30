//=== OutputSections.cpp --------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "OutputSections.h"
#include "DWARFLinkerCompileUnit.h"
#include "llvm/ADT/StringSwitch.h"

namespace llvm {
namespace dwarflinker_parallel {

static constexpr StringLiteral SectionNames[SectionKindsNum] = {
    "debug_info",     "debug_line",     "debug_frame",      "debug_ranges",
    "debug_rnglists", "debug_loc",      "debug_loclists",   "debug_aranges",
    "debug_abbrev",   "debug_macinfo",  "debug_macro",      "debug_addr",
    "debug_str",      "debug_line_str", "debug_str_offsets"};

const StringLiteral &getSectionName(DebugSectionKind SectionKind) {
  return SectionNames[static_cast<uint8_t>(SectionKind)];
}

std::optional<DebugSectionKind> parseDebugTableName(llvm::StringRef SecName) {
  return llvm::StringSwitch<std::optional<DebugSectionKind>>(
             SecName.substr(SecName.find_first_not_of("._")))
      .Case(getSectionName(DebugSectionKind::DebugInfo),
            DebugSectionKind::DebugInfo)
      .Case(getSectionName(DebugSectionKind::DebugLine),
            DebugSectionKind::DebugLine)
      .Case(getSectionName(DebugSectionKind::DebugFrame),
            DebugSectionKind::DebugFrame)
      .Case(getSectionName(DebugSectionKind::DebugRange),
            DebugSectionKind::DebugRange)
      .Case(getSectionName(DebugSectionKind::DebugRngLists),
            DebugSectionKind::DebugRngLists)
      .Case(getSectionName(DebugSectionKind::DebugLoc),
            DebugSectionKind::DebugLoc)
      .Case(getSectionName(DebugSectionKind::DebugLocLists),
            DebugSectionKind::DebugLocLists)
      .Case(getSectionName(DebugSectionKind::DebugARanges),
            DebugSectionKind::DebugARanges)
      .Case(getSectionName(DebugSectionKind::DebugAbbrev),
            DebugSectionKind::DebugAbbrev)
      .Case(getSectionName(DebugSectionKind::DebugMacinfo),
            DebugSectionKind::DebugMacinfo)
      .Case(getSectionName(DebugSectionKind::DebugMacro),
            DebugSectionKind::DebugMacro)
      .Case(getSectionName(DebugSectionKind::DebugAddr),
            DebugSectionKind::DebugAddr)
      .Case(getSectionName(DebugSectionKind::DebugStr),
            DebugSectionKind::DebugStr)
      .Case(getSectionName(DebugSectionKind::DebugLineStr),
            DebugSectionKind::DebugLineStr)
      .Case(getSectionName(DebugSectionKind::DebugStrOffsets),
            DebugSectionKind::DebugStrOffsets)
      .Default(std::nullopt);

  return std::nullopt;
}

DebugDieRefPatch::DebugDieRefPatch(uint64_t PatchOffset, CompileUnit *SrcCU,
                                   CompileUnit *RefCU, uint32_t RefIdx)
    : SectionPatch({PatchOffset}),
      RefCU(RefCU, (SrcCU != nullptr) &&
                       (SrcCU->getUniqueID() == RefCU->getUniqueID())),
      RefDieIdxOrClonedOffset(RefIdx) {}

DebugULEB128DieRefPatch::DebugULEB128DieRefPatch(uint64_t PatchOffset,
                                                 CompileUnit *SrcCU,
                                                 CompileUnit *RefCU,
                                                 uint32_t RefIdx)
    : SectionPatch({PatchOffset}),
      RefCU(RefCU, SrcCU->getUniqueID() == RefCU->getUniqueID()),
      RefDieIdxOrClonedOffset(RefIdx) {}

void SectionDescriptor::erase() {
  StartOffset = 0;
  Contents = OutSectionDataTy();
  ListDebugStrPatch.erase();
  ListDebugLineStrPatch.erase();
  ListDebugRangePatch.erase();
  ListDebugLocPatch.erase();
  ListDebugDieRefPatch.erase();
  ListDebugULEB128DieRefPatch.erase();
  ListDebugOffsetPatch.erase();
}

void SectionDescriptor::setSizesForSectionCreatedByAsmPrinter() {
  if (Contents.empty())
    return;

  MemoryBufferRef Mem(Contents, "obj");
  Expected<std::unique_ptr<object::ObjectFile>> Obj =
      object::ObjectFile::createObjectFile(Mem);
  if (!Obj) {
    consumeError(Obj.takeError());
    Contents.clear();
    return;
  }

  for (const object::SectionRef &Sect : (*Obj).get()->sections()) {
    Expected<StringRef> SectNameOrErr = Sect.getName();
    if (!SectNameOrErr) {
      consumeError(SectNameOrErr.takeError());
      continue;
    }

    if (std::optional<DebugSectionKind> SectKind =
            parseDebugTableName(*SectNameOrErr)) {
      if (*SectKind == SectionKind) {
        Expected<StringRef> Data = Sect.getContents();
        if (!Data) {
          consumeError(SectNameOrErr.takeError());
          Contents.clear();
          return;
        }

        SectionOffsetInsideAsmPrinterOutputStart =
            Data->data() - Contents.data();
        SectionOffsetInsideAsmPrinterOutputEnd =
            SectionOffsetInsideAsmPrinterOutputStart + Data->size();
      }
    }
  }
}

void SectionDescriptor::emitIntVal(uint64_t Val, unsigned Size) {
  switch (Size) {
  case 1: {
    OS.write(static_cast<uint8_t>(Val));
  } break;
  case 2: {
    uint16_t ShortVal = static_cast<uint16_t>(Val);
    if ((Endianess == support::endianness::little) != sys::IsLittleEndianHost)
      sys::swapByteOrder(ShortVal);
    OS.write(reinterpret_cast<const char *>(&ShortVal), Size);
  } break;
  case 4: {
    uint32_t ShortVal = static_cast<uint32_t>(Val);
    if ((Endianess == support::endianness::little) != sys::IsLittleEndianHost)
      sys::swapByteOrder(ShortVal);
    OS.write(reinterpret_cast<const char *>(&ShortVal), Size);
  } break;
  case 8: {
    if ((Endianess == support::endianness::little) != sys::IsLittleEndianHost)
      sys::swapByteOrder(Val);
    OS.write(reinterpret_cast<const char *>(&Val), Size);
  } break;
  default:
    llvm_unreachable("Unsupported integer type size");
  }
}

void SectionDescriptor::emitString(dwarf::Form StringForm,
                                   const char *StringVal) {
  assert(StringVal != nullptr);

  switch (StringForm) {
  case dwarf::DW_FORM_string: {
    emitInplaceString(GlobalData.translateString(StringVal));
  } break;
  case dwarf::DW_FORM_strp: {
    notePatch(DebugStrPatch{
        {OS.tell()}, GlobalData.getStringPool().insert(StringVal).first});
    emitStringPlaceholder();
  } break;
  case dwarf::DW_FORM_line_strp: {
    notePatch(DebugLineStrPatch{
        {OS.tell()}, GlobalData.getStringPool().insert(StringVal).first});
    emitStringPlaceholder();
  } break;
  default:
    llvm_unreachable("Unsupported string form");
    break;
  };
}

void SectionDescriptor::apply(uint64_t PatchOffset, dwarf::Form AttrForm,
                              uint64_t Val) {
  switch (AttrForm) {
  case dwarf::DW_FORM_strp:
  case dwarf::DW_FORM_line_strp: {
    applyIntVal(PatchOffset, Val, Format.getDwarfOffsetByteSize());
  } break;

  case dwarf::DW_FORM_ref_addr: {
    applyIntVal(PatchOffset, Val, Format.getRefAddrByteSize());
  } break;
  case dwarf::DW_FORM_ref1: {
    applyIntVal(PatchOffset, Val, 1);
  } break;
  case dwarf::DW_FORM_ref2: {
    applyIntVal(PatchOffset, Val, 2);
  } break;
  case dwarf::DW_FORM_ref4: {
    applyIntVal(PatchOffset, Val, 4);
  } break;
  case dwarf::DW_FORM_ref8: {
    applyIntVal(PatchOffset, Val, 8);
  } break;

  case dwarf::DW_FORM_data1: {
    applyIntVal(PatchOffset, Val, 1);
  } break;
  case dwarf::DW_FORM_data2: {
    applyIntVal(PatchOffset, Val, 2);
  } break;
  case dwarf::DW_FORM_data4: {
    applyIntVal(PatchOffset, Val, 4);
  } break;
  case dwarf::DW_FORM_data8: {
    applyIntVal(PatchOffset, Val, 8);
  } break;
  case dwarf::DW_FORM_udata: {
    applyULEB128(PatchOffset, Val);
  } break;
  case dwarf::DW_FORM_sdata: {
    applySLEB128(PatchOffset, Val);
  } break;
  case dwarf::DW_FORM_sec_offset: {
    applyIntVal(PatchOffset, Val, Format.getDwarfOffsetByteSize());
  } break;
  case dwarf::DW_FORM_flag: {
    applyIntVal(PatchOffset, Val, 1);
  } break;

  default:
    llvm_unreachable("Unsupported attribute form");
    break;
  }
}

uint64_t SectionDescriptor::getIntVal(uint64_t PatchOffset, unsigned Size) {
  assert(PatchOffset < getContents().size());
  switch (Size) {
  case 1: {
    return *reinterpret_cast<const uint8_t *>(
        (getContents().data() + PatchOffset));
  }
  case 2: {
    return support::endian::read16(getContents().data() + PatchOffset,
                                   Endianess);
  }
  case 4: {
    return support::endian::read32(getContents().data() + PatchOffset,
                                   Endianess);
  }
  case 8: {
    return support::endian::read64(getContents().data() + PatchOffset,
                                   Endianess);
  }
  }
  llvm_unreachable("Unsupported integer type size");
  return 0;
}

void SectionDescriptor::applyIntVal(uint64_t PatchOffset, uint64_t Val,
                                    unsigned Size) {
  assert(PatchOffset < getContents().size());

  switch (Size) {
  case 1: {
    support::endian::write(
        const_cast<char *>(getContents().data() + PatchOffset),
        static_cast<uint8_t>(Val), Endianess);
  } break;
  case 2: {
    support::endian::write(
        const_cast<char *>(getContents().data() + PatchOffset),
        static_cast<uint16_t>(Val), Endianess);
  } break;
  case 4: {
    support::endian::write(
        const_cast<char *>(getContents().data() + PatchOffset),
        static_cast<uint32_t>(Val), Endianess);
  } break;
  case 8: {
    support::endian::write(
        const_cast<char *>(getContents().data() + PatchOffset),
        static_cast<uint64_t>(Val), Endianess);
  } break;
  default:
    llvm_unreachable("Unsupported integer type size");
  }
}

void SectionDescriptor::applyULEB128(uint64_t PatchOffset, uint64_t Val) {
  assert(PatchOffset < getContents().size());

  uint8_t ULEB[16];
  uint8_t DestSize = Format.getDwarfOffsetByteSize() + 1;
  uint8_t RealSize = encodeULEB128(Val, ULEB, DestSize);

  memcpy(const_cast<char *>(getContents().data() + PatchOffset), ULEB,
         RealSize);
}

/// Writes integer value \p Val of SLEB128 format by specified \p PatchOffset.
void SectionDescriptor::applySLEB128(uint64_t PatchOffset, uint64_t Val) {
  assert(PatchOffset < getContents().size());

  uint8_t SLEB[16];
  uint8_t DestSize = Format.getDwarfOffsetByteSize() + 1;
  uint8_t RealSize = encodeSLEB128(Val, SLEB, DestSize);

  memcpy(const_cast<char *>(getContents().data() + PatchOffset), SLEB,
         RealSize);
}

void OutputSections::applyPatches(
    SectionDescriptor &Section,
    StringEntryToDwarfStringPoolEntryMap &DebugStrStrings,
    StringEntryToDwarfStringPoolEntryMap &DebugLineStrStrings) {

  Section.ListDebugStrPatch.forEach([&](DebugStrPatch &Patch) {
    DwarfStringPoolEntryWithExtString *Entry =
        DebugStrStrings.getExistingEntry(Patch.String);
    assert(Entry != nullptr);

    Section.apply(Patch.PatchOffset, dwarf::DW_FORM_strp, Entry->Offset);
  });

  Section.ListDebugLineStrPatch.forEach([&](DebugLineStrPatch &Patch) {
    DwarfStringPoolEntryWithExtString *Entry =
        DebugLineStrStrings.getExistingEntry(Patch.String);
    assert(Entry != nullptr);

    Section.apply(Patch.PatchOffset, dwarf::DW_FORM_line_strp, Entry->Offset);
  });

  std::optional<SectionDescriptor *> RangeSection;
  if (Format.Version >= 5)
    RangeSection = getSectionDescriptor(DebugSectionKind::DebugRngLists);
  else
    RangeSection = getSectionDescriptor(DebugSectionKind::DebugRange);

  if (RangeSection) {
    Section.ListDebugRangePatch.forEach([&](DebugRangePatch &Patch) {
      uint64_t FinalValue =
          Section.getIntVal(Patch.PatchOffset, Format.getDwarfOffsetByteSize());
      FinalValue += (*RangeSection)->StartOffset;

      Section.apply(Patch.PatchOffset, dwarf::DW_FORM_sec_offset, FinalValue);
    });
  }

  std::optional<SectionDescriptor *> LocationSection;
  if (Format.Version >= 5)
    LocationSection = getSectionDescriptor(DebugSectionKind::DebugLocLists);
  else
    LocationSection = getSectionDescriptor(DebugSectionKind::DebugLoc);

  if (LocationSection) {
    Section.ListDebugLocPatch.forEach([&](DebugLocPatch &Patch) {
      uint64_t FinalValue =
          Section.getIntVal(Patch.PatchOffset, Format.getDwarfOffsetByteSize());
      FinalValue += (*LocationSection)->StartOffset;

      Section.apply(Patch.PatchOffset, dwarf::DW_FORM_sec_offset, FinalValue);
    });
  }

  Section.ListDebugDieRefPatch.forEach([&](DebugDieRefPatch &Patch) {
    uint64_t FinalOffset = Patch.RefDieIdxOrClonedOffset;
    dwarf::Form FinalForm = dwarf::DW_FORM_ref4;

    if (!Patch.RefCU.getInt()) {
      std::optional<SectionDescriptor *> ReferencedSectionDescriptor =
          Patch.RefCU.getPointer()->getSectionDescriptor(
              DebugSectionKind::DebugInfo);
      if (!ReferencedSectionDescriptor) {
        // Referenced section should be already created at this point.
        llvm_unreachable("Referenced section does not exist");
      }

      FinalForm = dwarf::DW_FORM_ref_addr;
      FinalOffset += (*ReferencedSectionDescriptor)->StartOffset;
    }

    Section.apply(Patch.PatchOffset, FinalForm, FinalOffset);
  });

  Section.ListDebugULEB128DieRefPatch.forEach(
      [&](DebugULEB128DieRefPatch &Patch) {
        assert(Patch.RefCU.getInt());
        Section.apply(Patch.PatchOffset, dwarf::DW_FORM_udata,
                      Patch.RefDieIdxOrClonedOffset);
      });

  Section.ListDebugOffsetPatch.forEach([&](DebugOffsetPatch &Patch) {
    uint64_t FinalValue = Patch.SectionPtr.getPointer()->StartOffset;
    if (Patch.SectionPtr.getInt())
      FinalValue +=
          Section.getIntVal(Patch.PatchOffset, Format.getDwarfOffsetByteSize());

    Section.apply(Patch.PatchOffset, dwarf::DW_FORM_sec_offset, FinalValue);
  });
}

} // end of namespace dwarflinker_parallel
} // end of namespace llvm

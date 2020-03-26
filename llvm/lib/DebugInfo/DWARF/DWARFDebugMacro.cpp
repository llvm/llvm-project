//===- DWARFDebugMacro.cpp ------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/DebugInfo/DWARF/DWARFDebugMacro.h"
#include "llvm/BinaryFormat/Dwarf.h"
#include "llvm/DebugInfo/DWARF/DWARFContext.h"
#include "llvm/DebugInfo/DWARF/DWARFDataExtractor.h"
#include "llvm/Support/WithColor.h"
#include "llvm/Support/raw_ostream.h"
#include <cstdint>

using namespace llvm;
using namespace dwarf;

void DWARFDebugMacro::MacroHeader::dumpMacroHeader(raw_ostream &OS) const {
  // FIXME: Add support for dumping opcode_operands_table
  OS << format("macro header: version = 0x%04" PRIx16 ", flags = 0x%02" PRIx8,
               Version, Flags);
  if (Flags & MACRO_DEBUG_LINE_OFFSET)
    OS << format(", debug_line_offset = 0x%04" PRIx64 "\n", DebugLineOffset);
  else
    OS << "\n";
}

void DWARFDebugMacro::dump(raw_ostream &OS) const {
  unsigned IndLevel = 0;
  for (const auto &Macros : MacroLists) {
    OS << format("0x%08" PRIx64 ":\n", Macros.Offset);
    if (Macros.Header.Version >= 5)
      Macros.Header.dumpMacroHeader(OS);
    for (const Entry &E : Macros.Macros) {
      // There should not be DW_MACINFO_end_file when IndLevel is Zero. However,
      // this check handles the case of corrupted ".debug_macinfo" section.
      if (IndLevel > 0)
        IndLevel -= (E.Type == DW_MACINFO_end_file);
      // Print indentation.
      for (unsigned I = 0; I < IndLevel; I++)
        OS << "  ";
      IndLevel += (E.Type == DW_MACINFO_start_file);
      // Based on which version we are handling choose appropriate macro forms.
      if (Macros.Header.Version >= 5)
        WithColor(OS, HighlightColor::Macro).get() << MacroString(E.Type);
      else
        WithColor(OS, HighlightColor::Macro).get() << MacinfoString(E.Type);
      switch (E.Type) {
      default:
        // Got a corrupted ".debug_macinfo/.debug_macro" section (invalid
        // macinfo type).
        break;
        // debug_macro and debug_macinfo share some common encodings.
        // DW_MACRO_define     == DW_MACINFO_define
        // DW_MACRO_undef      == DW_MACINFO_undef
        // DW_MACRO_start_file == DW_MACINFO_start_file
        // DW_MACRO_end_file   == DW_MACINFO_end_file
        // For readability/uniformity we are using DW_MACRO_*.
      case DW_MACRO_define:
      case DW_MACRO_undef:
      case DW_MACRO_define_strp:
      case DW_MACRO_undef_strp:
        OS << " - lineno: " << E.Line;
        OS << " macro: " << E.MacroStr;
        break;
      case DW_MACRO_start_file:
        OS << " - lineno: " << E.Line;
        OS << " filenum: " << E.File;
        break;
      case DW_MACRO_end_file:
        break;
      case DW_MACINFO_vendor_ext:
        OS << " - constant: " << E.ExtConstant;
        OS << " string: " << E.ExtStr;
        break;
      }
      OS << "\n";
    }
  }
}

Error DWARFDebugMacro::parse(DataExtractor StringExtractor,
                             DWARFDataExtractor Data, bool IsMacro) {
  uint64_t Offset = 0;
  MacroList *M = nullptr;
  while (Data.isValidOffset(Offset)) {
    if (!M) {
      MacroLists.emplace_back();
      M = &MacroLists.back();
      M->Offset = Offset;
      if (IsMacro) {
        auto Err = M->Header.parseMacroHeader(Data, &Offset);
        if (Err)
          return std::move(Err);
      }
    }
    // A macro list entry consists of:
    M->Macros.emplace_back();
    Entry &E = M->Macros.back();
    // 1. Macinfo type
    E.Type = Data.getULEB128(&Offset);

    if (E.Type == 0) {
      // Reached end of a ".debug_macinfo/debug_macro" section contribution.
      M = nullptr;
      continue;
    }

    switch (E.Type) {
    default:
      // Got a corrupted ".debug_macinfo" section (invalid macinfo type).
      // Push the corrupted entry to the list and halt parsing.
      E.Type = DW_MACINFO_invalid;
      return Error::success();
    // debug_macro and debug_macinfo share some common encodings.
    // DW_MACRO_define     == DW_MACINFO_define
    // DW_MACRO_undef      == DW_MACINFO_undef
    // DW_MACRO_start_file == DW_MACINFO_start_file
    // DW_MACRO_end_file   == DW_MACINFO_end_file
    // For readibility/uniformity we are using DW_MACRO_*.
    case DW_MACRO_define:
    case DW_MACRO_undef:
      // 2. Source line
      E.Line = Data.getULEB128(&Offset);
      // 3. Macro string
      E.MacroStr = Data.getCStr(&Offset);
      break;
    case DW_MACRO_define_strp:
    case DW_MACRO_undef_strp: {
      uint64_t StrOffset = 0;
      // 2. Source line
      E.Line = Data.getULEB128(&Offset);
      // 3. Macro string
      // FIXME: Add support for DWARF64
      StrOffset = Data.getRelocatedValue(/*OffsetSize=*/4, &Offset);
      E.MacroStr = StringExtractor.getCStr(&StrOffset);
      break;
    }
    case DW_MACRO_start_file:
      // 2. Source line
      E.Line = Data.getULEB128(&Offset);
      // 3. Source file id
      E.File = Data.getULEB128(&Offset);
      break;
    case DW_MACRO_end_file:
      break;
    case DW_MACINFO_vendor_ext:
      // 2. Vendor extension constant
      E.ExtConstant = Data.getULEB128(&Offset);
      // 3. Vendor extension string
      E.ExtStr = Data.getCStr(&Offset);
      break;
    }
  }
  return Error::success();
}

Error DWARFDebugMacro::MacroHeader::parseMacroHeader(DWARFDataExtractor Data,
                                                     uint64_t *Offset) {
  Version = Data.getU16(Offset);
  uint8_t FlagData = Data.getU8(Offset);
  // FIXME: Add support for DWARF64
  if (FlagData & MACRO_OFFSET_SIZE)
    return createStringError(errc::not_supported, "DWARF64 is not supported");

  // FIXME: Add support for parsing opcode_operands_table
  if (FlagData & MACRO_OPCODE_OPERANDS_TABLE)
    return createStringError(errc::not_supported,
                             "opcode_operands_table is not supported");
  Flags = FlagData;
  if (Flags & MACRO_DEBUG_LINE_OFFSET)
    DebugLineOffset = Data.getU32(Offset);
  return Error::success();
}

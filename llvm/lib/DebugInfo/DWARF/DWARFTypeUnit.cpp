//===- DWARFTypeUnit.cpp --------------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "llvm/DebugInfo/DIContext.h"
#include "llvm/DebugInfo/DWARF/DWARFDebugAbbrev.h"
#include "llvm/DebugInfo/DWARF/DWARFDie.h"
#include "llvm/DebugInfo/DWARF/DWARFTypeUnit.h"
#include "llvm/DebugInfo/DWARF/DWARFUnit.h"
#include "llvm/Support/Format.h"
#include "llvm/Support/raw_ostream.h"
#include <cinttypes>

using namespace llvm;

bool DWARFTypeUnit::extractImpl(DataExtractor debug_info,
                                uint32_t *offset_ptr) {
  if (!DWARFUnit::extractImpl(debug_info, offset_ptr))
    return false;
  TypeHash = debug_info.getU64(offset_ptr);
  TypeOffset = debug_info.getU32(offset_ptr);
  // TypeOffset is relative to the beginning of the header,
  // so we have to account for the leading length field.
  // FIXME: The size of the length field is 12 in DWARF64.
  unsigned SizeOfLength = 4;
  return TypeOffset < getLength() + SizeOfLength;
}

void DWARFTypeUnit::dump(raw_ostream &OS, bool SummarizeTypes) {
  DWARFDie TD = getDIEForOffset(TypeOffset + getOffset());
  const char *Name = TD.getName(DINameKind::ShortName);

  if (SummarizeTypes) {
    OS << "name = '" << Name << "'"
       << " type_signature = " << format("0x%016" PRIx64, TypeHash)
       << " length = " << format("0x%08x", getLength()) << '\n';
    return;
  }

  OS << format("0x%08x", getOffset()) << ": Type Unit:"
     << " length = " << format("0x%08x", getLength())
     << " version = " << format("0x%04x", getVersion());
  if (getVersion() >= 5)
    OS << " unit_type = " << dwarf::UnitTypeString(getUnitType());
  OS << " abbr_offset = " << format("0x%04x", getAbbreviations()->getOffset())
     << " addr_size = " << format("0x%02x", getAddressByteSize())
     << " name = '" << Name << "'"
     << " type_signature = " << format("0x%016" PRIx64, TypeHash)
     << " type_offset = " << format("0x%04x", TypeOffset)
     << " (next unit at " << format("0x%08x", getNextUnitOffset()) << ")\n";

  if (DWARFDie TU = getUnitDIE(false))
    TU.dump(OS, -1U);
  else
    OS << "<type unit can't be parsed!>\n\n";
}

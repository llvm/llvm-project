//===- DWARFTypeUnit.cpp --------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/DebugInfo/DWARF/DWARFTypeUnit.h"
#include "llvm/DebugInfo/DIContext.h"
#include "llvm/DebugInfo/DWARF/DWARFDie.h"
#include "llvm/Support/Format.h"
#include "llvm/Support/FormatAdapters.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/raw_ostream.h"
#include <cinttypes>

using namespace llvm;

void DWARFTypeUnit::dump(raw_ostream &OS, DIDumpOptions DumpOpts) {
  DWARFDie TD = getDIEForOffset(getTypeOffset() + getOffset());
  const char *Name = TD.getName(DINameKind::ShortName);
  int OffsetDumpWidth = 2 * dwarf::getDwarfOffsetByteSize(getFormat());

  if (DumpOpts.SummarizeTypes) {
    OS << "name = '" << Name << "'"
       << ", type_signature = " << formatv("{0:x16}", getTypeHash())
       << ", length = "
       << formatv("0x{0:x-}", fmt_align(getLength(), AlignStyle::Right,
                                        OffsetDumpWidth, '0'))
       << '\n';
    return;
  }

  OS << formatv("{0:x8}", getOffset()) << ": Type Unit:"
     << " length = "
     << formatv("0x{0:x-}",
                fmt_align(getLength(), AlignStyle::Right, OffsetDumpWidth, '0'))
     << ", format = " << dwarf::FormatString(getFormat())
     << ", version = " << formatv("{0:x4}", getVersion());
  if (getVersion() >= 5)
    OS << ", unit_type = " << dwarf::UnitTypeString(getUnitType());
  OS << ", abbr_offset = " << formatv("{0:x4}", getAbbrOffset());
  if (!getAbbreviations())
    OS << " (invalid)";
  OS << ", addr_size = " << formatv("{0:x2}", getAddressByteSize())
     << ", name = '" << Name << "'"
     << ", type_signature = " << formatv("{0:x16}", getTypeHash())
     << ", type_offset = " << formatv("{0:x4}", getTypeOffset())
     << " (next unit at " << formatv("{0:x8}", getNextUnitOffset()) << ")\n";

  if (DWARFDie TU = getUnitDIE(false))
    TU.dump(OS, 0, DumpOpts);
  else
    OS << "<type unit can't be parsed!>\n\n";
}

//===-- DWARFCompileUnit.cpp ----------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/DebugInfo/DWARF/DWARFCompileUnit.h"
#include "llvm/DebugInfo/DIContext.h"
#include "llvm/DebugInfo/DWARF/DWARFDie.h"

#include "llvm/Support/Format.h"
#include "llvm/Support/FormatAdapters.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/raw_ostream.h"

using namespace llvm;

void DWARFCompileUnit::dump(raw_ostream &OS, DIDumpOptions DumpOpts) {
  if (DumpOpts.SummarizeTypes)
    return;
  int OffsetDumpWidth = 2 * dwarf::getDwarfOffsetByteSize(getFormat());
  OS << formatv("{0:x+8}", getOffset()) << ": Compile Unit:"
     << " length = "
     << formatv("0x{0:x-}",
                fmt_align(getLength(), AlignStyle::Right, OffsetDumpWidth, '0'))
     << ", format = " << dwarf::FormatString(getFormat())
     << ", version = " << formatv("{0:x+4}", getVersion());

  if (getVersion() >= 5)
    OS << ", unit_type = " << dwarf::UnitTypeString(getUnitType());
  OS << ", abbr_offset = " << formatv("{0:x+4}", getAbbrOffset());
  if (!getAbbreviations())
    OS << " (invalid)";
  OS << ", addr_size = " << formatv("{0:x+2}", getAddressByteSize());
  if (getVersion() >= 5 && (getUnitType() == dwarf::DW_UT_skeleton ||
                            getUnitType() == dwarf::DW_UT_split_compile))
    OS << ", DWO_id = " << formatv("{0:x+16}", *getDWOId());
  OS << " (next unit at " << formatv("{0:x+8}", getNextUnitOffset()) << ")\n";

  if (DWARFDie CUDie = getUnitDIE(false)) {
    CUDie.dump(OS, 0, DumpOpts);
    if (DumpOpts.DumpNonSkeleton) {
      DWARFDie NonSkeletonCUDie = getNonSkeletonUnitDIE(false);
      if (NonSkeletonCUDie && CUDie != NonSkeletonCUDie)
        NonSkeletonCUDie.dump(OS, 0, DumpOpts);
    }
  } else {
    OS << "<compile unit can't be parsed!>\n\n";
  }
}

// VTable anchor.
DWARFCompileUnit::~DWARFCompileUnit() = default;

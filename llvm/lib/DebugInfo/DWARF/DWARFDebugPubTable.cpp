//===- DWARFDebugPubTable.cpp ---------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/DebugInfo/DWARF/DWARFDebugPubTable.h"
#include "llvm/DebugInfo/DWARF/DWARFDataExtractor.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/BinaryFormat/Dwarf.h"
#include "llvm/Support/DataExtractor.h"
#include "llvm/Support/Format.h"
#include "llvm/Support/raw_ostream.h"
#include <cstdint>

using namespace llvm;
using namespace dwarf;

Error DWARFDebugPubTable::extract(DWARFDataExtractor Data, bool GnuStyle) {
  this->GnuStyle = GnuStyle;
  Sets.clear();
  DataExtractor::Cursor C(0);
  while (C && Data.isValidOffset(C.tell())) {
    Sets.push_back({});
    Set &SetData = Sets.back();

    std::tie(SetData.Length, SetData.Format) = Data.getInitialLength(C);
    const unsigned OffsetSize = dwarf::getDwarfOffsetByteSize(SetData.Format);

    SetData.Version = Data.getU16(C);
    SetData.Offset = Data.getRelocatedValue(C, OffsetSize);
    SetData.Size = Data.getUnsigned(C, OffsetSize);

    while (C) {
      uint64_t DieRef = Data.getUnsigned(C, OffsetSize);
      if (DieRef == 0)
        break;
      uint8_t IndexEntryValue = GnuStyle ? Data.getU8(C) : 0;
      StringRef Name = Data.getCStrRef(C);
      SetData.Entries.push_back(
          {DieRef, PubIndexEntryDescriptor(IndexEntryValue), Name});
    }
  }
  return C.takeError();
}

void DWARFDebugPubTable::dump(raw_ostream &OS) const {
  for (const Set &S : Sets) {
    int OffsetDumpWidth = 2 * dwarf::getDwarfOffsetByteSize(S.Format);
    OS << "length = " << format("0x%0*" PRIx64, OffsetDumpWidth, S.Length);
    OS << ", format = " << dwarf::FormatString(S.Format);
    OS << ", version = " << format("0x%04x", S.Version);
    OS << ", unit_offset = "
       << format("0x%0*" PRIx64, OffsetDumpWidth, S.Offset);
    OS << ", unit_size = " << format("0x%0*" PRIx64, OffsetDumpWidth, S.Size)
       << '\n';
    OS << (GnuStyle ? "Offset     Linkage  Kind     Name\n"
                    : "Offset     Name\n");

    for (const Entry &E : S.Entries) {
      OS << format("0x%0*" PRIx64 " ", OffsetDumpWidth, E.SecOffset);
      if (GnuStyle) {
        StringRef EntryLinkage =
            GDBIndexEntryLinkageString(E.Descriptor.Linkage);
        StringRef EntryKind = dwarf::GDBIndexEntryKindString(E.Descriptor.Kind);
        OS << format("%-8s", EntryLinkage.data()) << ' '
           << format("%-8s", EntryKind.data()) << ' ';
      }
      OS << '\"' << E.Name << "\"\n";
    }
  }
}

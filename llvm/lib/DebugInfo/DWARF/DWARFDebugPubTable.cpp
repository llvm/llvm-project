//===- DWARFDebugPubTable.cpp ---------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/DebugInfo/DWARF/DWARFDebugPubTable.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/BinaryFormat/Dwarf.h"
#include "llvm/DebugInfo/DWARF/DWARFDataExtractor.h"
#include "llvm/Support/DataExtractor.h"
#include "llvm/Support/Errc.h"
#include "llvm/Support/Format.h"
#include "llvm/Support/FormatAdapters.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/raw_ostream.h"
#include <cstdint>

using namespace llvm;
using namespace dwarf;

void DWARFDebugPubTable::extract(
    DWARFDataExtractor Data, bool GnuStyle,
    function_ref<void(Error)> RecoverableErrorHandler) {
  this->GnuStyle = GnuStyle;
  Sets.clear();
  uint64_t Offset = 0;
  while (Data.isValidOffset(Offset)) {
    uint64_t SetOffset = Offset;
    Sets.push_back({});
    Set &NewSet = Sets.back();

    DataExtractor::Cursor C(Offset);
    std::tie(NewSet.Length, NewSet.Format) = Data.getInitialLength(C);
    if (!C) {
      // Drop the newly added set because it does not contain anything useful
      // to dump.
      Sets.pop_back();
      RecoverableErrorHandler(createStringError(
          errc::invalid_argument,
          "name lookup table at offset 0x%" PRIx64 " parsing failed: %s",
          SetOffset, toString(C.takeError()).c_str()));
      return;
    }

    Offset = C.tell() + NewSet.Length;
    DWARFDataExtractor SetData(Data, Offset);
    const unsigned OffsetSize = dwarf::getDwarfOffsetByteSize(NewSet.Format);

    NewSet.Version = SetData.getU16(C);
    NewSet.Offset = SetData.getRelocatedValue(C, OffsetSize);
    NewSet.Size = SetData.getUnsigned(C, OffsetSize);

    if (!C) {
      // Preserve the newly added set because at least some fields of the header
      // are read and can be dumped.
      RecoverableErrorHandler(
          createStringError(errc::invalid_argument,
                            "name lookup table at offset 0x%" PRIx64
                            " does not have a complete header: %s",
                            SetOffset, toString(C.takeError()).c_str()));
      continue;
    }

    while (C) {
      uint64_t DieRef = SetData.getUnsigned(C, OffsetSize);
      if (DieRef == 0)
        break;
      uint8_t IndexEntryValue = GnuStyle ? SetData.getU8(C) : 0;
      StringRef Name = SetData.getCStrRef(C);
      if (C)
        NewSet.Entries.push_back(
            {DieRef, PubIndexEntryDescriptor(IndexEntryValue), Name});
    }

    if (!C) {
      RecoverableErrorHandler(createStringError(
          errc::invalid_argument,
          "name lookup table at offset 0x%" PRIx64 " parsing failed: %s",
          SetOffset, toString(C.takeError()).c_str()));
      continue;
    }
    if (C.tell() != Offset)
      RecoverableErrorHandler(createStringError(
          errc::invalid_argument,
          "name lookup table at offset 0x%" PRIx64
          " has a terminator at offset 0x%" PRIx64
          " before the expected end at 0x%" PRIx64,
          SetOffset, C.tell() - OffsetSize, Offset - OffsetSize));
  }
}

void DWARFDebugPubTable::dump(raw_ostream &OS) const {
  for (const Set &S : Sets) {
    int OffsetDumpWidth = 2 * dwarf::getDwarfOffsetByteSize(S.Format);
    OS << "length = "
       << formatv("0x{0:x-}",
                  fmt_align(S.Length, AlignStyle::Right, OffsetDumpWidth, '0'));
    OS << ", format = " << dwarf::FormatString(S.Format);
    OS << ", version = " << formatv("{0:x4}", S.Version);
    OS << ", unit_offset = "
       << formatv("0x{0:x-}",
                  fmt_align(S.Offset, AlignStyle::Right, OffsetDumpWidth, '0'));
    OS << ", unit_size = "
       << formatv("0x{0:x-}",
                  fmt_align(S.Size, AlignStyle::Right, OffsetDumpWidth, '0'))
       << '\n';
    OS << (GnuStyle ? "Offset     Linkage  Kind     Name\n"
                    : "Offset     Name\n");

    for (const Entry &E : S.Entries) {
      OS << formatv("0x{0:x-} ", fmt_align(E.SecOffset, AlignStyle::Right,
                                           OffsetDumpWidth, '0'));
      if (GnuStyle) {
        StringRef EntryLinkage =
            GDBIndexEntryLinkageString(E.Descriptor.Linkage);
        StringRef EntryKind = dwarf::GDBIndexEntryKindString(E.Descriptor.Kind);
        OS << formatv("{0,-8}", EntryLinkage.data()) << ' '
           << formatv("{0,-8}", EntryKind.data()) << ' ';
      }
      OS << '\"' << E.Name << "\"\n";
    }
  }
}

//===- DWARFDebugArangeSet.cpp --------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/DebugInfo/DWARF/DWARFDebugArangeSet.h"
#include "llvm/BinaryFormat/Dwarf.h"
#include "llvm/Support/Errc.h"
#include "llvm/Support/Format.h"
#include "llvm/Support/raw_ostream.h"
#include <cassert>
#include <cinttypes>
#include <cstdint>
#include <cstring>

using namespace llvm;

void DWARFDebugArangeSet::Descriptor::dump(raw_ostream &OS,
                                           uint32_t AddressSize) const {
  OS << format("[0x%*.*" PRIx64 ", ", AddressSize * 2, AddressSize * 2, Address)
     << format(" 0x%*.*" PRIx64 ")", AddressSize * 2, AddressSize * 2,
               getEndAddress());
}

void DWARFDebugArangeSet::clear() {
  Offset = -1ULL;
  std::memset(&HeaderData, 0, sizeof(Header));
  ArangeDescriptors.clear();
}

Error DWARFDebugArangeSet::extract(DWARFDataExtractor data,
                                   uint64_t *offset_ptr) {
  assert(data.isValidOffset(*offset_ptr));
  ArangeDescriptors.clear();
  Offset = *offset_ptr;

  // 7.21 Address Range Table (extract)
  // Each set of entries in the table of address ranges contained in
  // the .debug_aranges section begins with a header containing:
  // 1. unit_length (initial length)
  //    A 4-byte (32-bit DWARF) or 12-byte (64-bit DWARF) length containing
  //    the length of the set of entries for this compilation unit,
  //    not including the length field itself.
  // 2. version (uhalf)
  //    The value in this field is 2.
  // 3. debug_info_offset (section offset)
  //    A 4-byte (32-bit DWARF) or 8-byte (64-bit DWARF) offset into the
  //    .debug_info section of the compilation unit header.
  // 4. address_size (ubyte)
  // 5. segment_selector_size (ubyte)
  // This header is followed by a series of tuples. Each tuple consists of
  // a segment, an address and a length. The segment selector size is given by
  // the segment_selector_size field of the header; the address and length
  // size are each given by the address_size field of the header. Each set of
  // tuples is terminated by a 0 for the segment, a 0 for the address and 0
  // for the length. If the segment_selector_size field in the header is zero,
  // the segment selectors are omitted from all tuples, including
  // the terminating tuple.

  Error Err = Error::success();
  std::tie(HeaderData.Length, HeaderData.Format) =
      data.getInitialLength(offset_ptr, &Err);
  HeaderData.Version = data.getU16(offset_ptr, &Err);
  HeaderData.CuOffset = data.getUnsigned(
      offset_ptr, dwarf::getDwarfOffsetByteSize(HeaderData.Format), &Err);
  HeaderData.AddrSize = data.getU8(offset_ptr, &Err);
  HeaderData.SegSize = data.getU8(offset_ptr, &Err);
  if (Err) {
    return createStringError(errc::invalid_argument,
                             "parsing address ranges table at offset 0x%" PRIx64
                             ": %s",
                             Offset, toString(std::move(Err)).c_str());
  }

  // Perform basic validation of the header fields.
  uint64_t full_length =
      dwarf::getUnitLengthFieldByteSize(HeaderData.Format) + HeaderData.Length;
  if (!data.isValidOffsetForDataOfSize(Offset, full_length))
    return createStringError(errc::invalid_argument,
                             "the length of address range table at offset "
                             "0x%" PRIx64 " exceeds section size",
                             Offset);
  if (HeaderData.AddrSize != 4 && HeaderData.AddrSize != 8)
    return createStringError(errc::invalid_argument,
                             "address range table at offset 0x%" PRIx64
                             " has unsupported address size: %d "
                             "(4 and 8 supported)",
                             Offset, HeaderData.AddrSize);
  if (HeaderData.SegSize != 0)
    return createStringError(errc::not_supported,
                             "non-zero segment selector size in address range "
                             "table at offset 0x%" PRIx64 " is not supported",
                             Offset);

  // The first tuple following the header in each set begins at an offset that
  // is a multiple of the size of a single tuple (that is, twice the size of
  // an address because we do not support non-zero segment selector sizes).
  // Therefore, the full length should also be a multiple of the tuple size.
  const uint32_t tuple_size = HeaderData.AddrSize * 2;
  if (full_length % tuple_size != 0)
    return createStringError(
        errc::invalid_argument,
        "address range table at offset 0x%" PRIx64
        " has length that is not a multiple of the tuple size",
        Offset);

  // The header is padded, if necessary, to the appropriate boundary.
  const uint32_t header_size = *offset_ptr - Offset;
  uint32_t first_tuple_offset = 0;
  while (first_tuple_offset < header_size)
    first_tuple_offset += tuple_size;

  // There should be space for at least one tuple.
  if (full_length <= first_tuple_offset)
    return createStringError(
        errc::invalid_argument,
        "address range table at offset 0x%" PRIx64
        " has an insufficient length to contain any entries",
        Offset);

  *offset_ptr = Offset + first_tuple_offset;

  Descriptor arangeDescriptor;

  static_assert(sizeof(arangeDescriptor.Address) ==
                    sizeof(arangeDescriptor.Length),
                "Different datatypes for addresses and sizes!");
  assert(sizeof(arangeDescriptor.Address) >= HeaderData.AddrSize);

  uint64_t end_offset = Offset + full_length;
  while (*offset_ptr < end_offset) {
    arangeDescriptor.Address = data.getUnsigned(offset_ptr, HeaderData.AddrSize);
    arangeDescriptor.Length = data.getUnsigned(offset_ptr, HeaderData.AddrSize);

    if (arangeDescriptor.Length == 0) {
      // Each set of tuples is terminated by a 0 for the address and 0
      // for the length.
      if (arangeDescriptor.Address == 0 && *offset_ptr == end_offset)
        return ErrorSuccess();
      return createStringError(
          errc::invalid_argument,
          "address range table at offset 0x%" PRIx64
          " has an invalid tuple (length = 0) at offset 0x%" PRIx64,
          Offset, *offset_ptr - tuple_size);
    }

    ArangeDescriptors.push_back(arangeDescriptor);
  }

  return createStringError(errc::invalid_argument,
                           "address range table at offset 0x%" PRIx64
                           " is not terminated by null entry",
                           Offset);
}

void DWARFDebugArangeSet::dump(raw_ostream &OS) const {
  int OffsetDumpWidth = 2 * dwarf::getDwarfOffsetByteSize(HeaderData.Format);
  OS << "Address Range Header: "
     << format("length = 0x%0*" PRIx64 ", ", OffsetDumpWidth, HeaderData.Length)
     << format("version = 0x%4.4x, ", HeaderData.Version)
     << format("cu_offset = 0x%0*" PRIx64 ", ", OffsetDumpWidth,
               HeaderData.CuOffset)
     << format("addr_size = 0x%2.2x, ", HeaderData.AddrSize)
     << format("seg_size = 0x%2.2x\n", HeaderData.SegSize);

  for (const auto &Desc : ArangeDescriptors) {
    Desc.dump(OS, HeaderData.AddrSize);
    OS << '\n';
  }
}

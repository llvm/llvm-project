//===- DWARFDebugArangeSet.cpp --------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/DebugInfo/DWARF/DWARFDebugArangeSet.h"
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

Error DWARFDebugArangeSet::extract(DataExtractor data, uint64_t *offset_ptr) {
  assert(data.isValidOffset(*offset_ptr));
  ArangeDescriptors.clear();
  Offset = *offset_ptr;

  // 7.20 Address Range Table
  //
  // Each set of entries in the table of address ranges contained in
  // the .debug_aranges section begins with a header consisting of: a
  // 4-byte length containing the length of the set of entries for this
  // compilation unit, not including the length field itself; a 2-byte
  // version identifier containing the value 2 for DWARF Version 2; a
  // 4-byte offset into the.debug_infosection; a 1-byte unsigned integer
  // containing the size in bytes of an address (or the offset portion of
  // an address for segmented addressing) on the target system; and a
  // 1-byte unsigned integer containing the size in bytes of a segment
  // descriptor on the target system. This header is followed by a series
  // of tuples. Each tuple consists of an address and a length, each in
  // the size appropriate for an address on the target architecture.
  HeaderData.Length = data.getU32(offset_ptr);
  HeaderData.Version = data.getU16(offset_ptr);
  HeaderData.CuOffset = data.getU32(offset_ptr);
  HeaderData.AddrSize = data.getU8(offset_ptr);
  HeaderData.SegSize = data.getU8(offset_ptr);

  // Perform basic validation of the header fields.
  if (!data.isValidOffsetForDataOfSize(Offset, HeaderData.Length + 4))
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

  // The first tuple following the header in each set begins at an offset
  // that is a multiple of the size of a single tuple (that is, twice the
  // size of an address). The header is padded, if necessary, to the
  // appropriate boundary.
  const uint32_t header_size = *offset_ptr - Offset;
  const uint32_t tuple_size = HeaderData.AddrSize * 2;
  uint32_t first_tuple_offset = 0;
  while (first_tuple_offset < header_size)
    first_tuple_offset += tuple_size;

  *offset_ptr = Offset + first_tuple_offset;

  Descriptor arangeDescriptor;

  static_assert(sizeof(arangeDescriptor.Address) ==
                    sizeof(arangeDescriptor.Length),
                "Different datatypes for addresses and sizes!");
  assert(sizeof(arangeDescriptor.Address) >= HeaderData.AddrSize);

  while (data.isValidOffset(*offset_ptr)) {
    arangeDescriptor.Address = data.getUnsigned(offset_ptr, HeaderData.AddrSize);
    arangeDescriptor.Length = data.getUnsigned(offset_ptr, HeaderData.AddrSize);

    // Each set of tuples is terminated by a 0 for the address and 0
    // for the length.
    if (arangeDescriptor.Address == 0 && arangeDescriptor.Length == 0)
      return ErrorSuccess();
    ArangeDescriptors.push_back(arangeDescriptor);
  }

  return createStringError(errc::invalid_argument,
                           "address range table at offset 0x%" PRIx64
                           " is not terminated by null entry",
                           Offset);
}

void DWARFDebugArangeSet::dump(raw_ostream &OS) const {
  OS << format("Address Range Header: length = 0x%8.8x, version = 0x%4.4x, ",
               HeaderData.Length, HeaderData.Version)
     << format("cu_offset = 0x%8.8x, addr_size = 0x%2.2x, seg_size = 0x%2.2x\n",
               HeaderData.CuOffset, HeaderData.AddrSize, HeaderData.SegSize);

  for (const auto &Desc : ArangeDescriptors) {
    Desc.dump(OS, HeaderData.AddrSize);
    OS << '\n';
  }
}

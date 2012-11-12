//===-- DWARFDebugArangeSet.cpp -------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "DWARFDebugArangeSet.h"
#include "llvm/Support/Format.h"
#include "llvm/Support/raw_ostream.h"
#include <algorithm>
#include <cassert>
using namespace llvm;

void DWARFDebugArangeSet::clear() {
  Offset = -1U;
  std::memset(&Header, 0, sizeof(Header));
  ArangeDescriptors.clear();
}

void DWARFDebugArangeSet::compact() {
  if (ArangeDescriptors.empty())
    return;

  // Iterate through all arange descriptors and combine any ranges that
  // overlap or have matching boundaries. The ArangeDescriptors are assumed
  // to be in ascending order.
  uint32_t i = 0;
  while (i + 1 < ArangeDescriptors.size()) {
    if (ArangeDescriptors[i].getEndAddress() >= ArangeDescriptors[i+1].Address){
      // The current range ends at or exceeds the start of the next address
      // range. Compute the max end address between the two and use that to
      // make the new length.
      const uint64_t max_end_addr =
        std::max(ArangeDescriptors[i].getEndAddress(),
                 ArangeDescriptors[i+1].getEndAddress());
      ArangeDescriptors[i].Length = max_end_addr - ArangeDescriptors[i].Address;
      // Now remove the next entry as it was just combined with the previous one
      ArangeDescriptors.erase(ArangeDescriptors.begin()+i+1);
    } else {
      // Discontiguous address range, just proceed to the next one.
      ++i;
    }
  }
}

bool
DWARFDebugArangeSet::extract(DataExtractor data, uint32_t *offset_ptr) {
  if (data.isValidOffset(*offset_ptr)) {
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
    Header.Length = data.getU32(offset_ptr);
    Header.Version = data.getU16(offset_ptr);
    Header.CuOffset = data.getU32(offset_ptr);
    Header.AddrSize = data.getU8(offset_ptr);
    Header.SegSize = data.getU8(offset_ptr);

    // Perform basic validation of the header fields.
    if (!data.isValidOffsetForDataOfSize(Offset, Header.Length) ||
        (Header.AddrSize != 4 && Header.AddrSize != 8)) {
      clear();
      return false;
    }

    // The first tuple following the header in each set begins at an offset
    // that is a multiple of the size of a single tuple (that is, twice the
    // size of an address). The header is padded, if necessary, to the
    // appropriate boundary.
    const uint32_t header_size = *offset_ptr - Offset;
    const uint32_t tuple_size = Header.AddrSize * 2;
    uint32_t first_tuple_offset = 0;
    while (first_tuple_offset < header_size)
      first_tuple_offset += tuple_size;

    *offset_ptr = Offset + first_tuple_offset;

    Descriptor arangeDescriptor;

    assert(sizeof(arangeDescriptor.Address) == sizeof(arangeDescriptor.Length));
    assert(sizeof(arangeDescriptor.Address) >= Header.AddrSize);

    while (data.isValidOffset(*offset_ptr)) {
      arangeDescriptor.Address = data.getUnsigned(offset_ptr, Header.AddrSize);
      arangeDescriptor.Length = data.getUnsigned(offset_ptr, Header.AddrSize);

      // Each set of tuples is terminated by a 0 for the address and 0
      // for the length.
      if (arangeDescriptor.Address || arangeDescriptor.Length)
        ArangeDescriptors.push_back(arangeDescriptor);
      else
        break; // We are done if we get a zero address and length
    }

    return !ArangeDescriptors.empty();
  }
  return false;
}

void DWARFDebugArangeSet::dump(raw_ostream &OS) const {
  OS << format("Address Range Header: length = 0x%8.8x, version = 0x%4.4x, ",
               Header.Length, Header.Version)
     << format("cu_offset = 0x%8.8x, addr_size = 0x%2.2x, seg_size = 0x%2.2x\n",
               Header.CuOffset, Header.AddrSize, Header.SegSize);

  const uint32_t hex_width = Header.AddrSize * 2;
  for (DescriptorConstIter pos = ArangeDescriptors.begin(),
       end = ArangeDescriptors.end(); pos != end; ++pos)
    OS << format("[0x%*.*" PRIx64 " -", hex_width, hex_width, pos->Address)
       << format(" 0x%*.*" PRIx64 ")\n",
                 hex_width, hex_width, pos->getEndAddress());
}


namespace {
  class DescriptorContainsAddress {
    const uint64_t Address;
  public:
    DescriptorContainsAddress(uint64_t address) : Address(address) {}
    bool operator()(const DWARFDebugArangeSet::Descriptor &desc) const {
      return Address >= desc.Address && Address < (desc.Address + desc.Length);
    }
  };
}

uint32_t DWARFDebugArangeSet::findAddress(uint64_t address) const {
  DescriptorConstIter end = ArangeDescriptors.end();
  DescriptorConstIter pos =
    std::find_if(ArangeDescriptors.begin(), end, // Range
                 DescriptorContainsAddress(address)); // Predicate
  if (pos != end)
    return Header.CuOffset;

  return -1U;
}

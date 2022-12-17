//===-- DIERef.cpp --------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "DIERef.h"
#include "lldb/Utility/DataEncoder.h"
#include "lldb/Utility/DataExtractor.h"
#include "llvm/Support/Format.h"

using namespace lldb;
using namespace lldb_private;

void llvm::format_provider<DIERef>::format(const DIERef &ref, raw_ostream &OS,
                                           StringRef Style) {
  if (ref.dwo_num())
    OS << format_hex_no_prefix(*ref.dwo_num(), 8) << "/";
  OS << (ref.section() == DIERef::DebugInfo ? "INFO" : "TYPE");
  OS << "/" << format_hex_no_prefix(ref.die_offset(), 8);
}

constexpr uint32_t k_dwo_num_mask = 0x3FFFFFFF;
constexpr uint32_t k_dwo_num_valid_bitmask = (1u << 30);
constexpr uint32_t k_section_bitmask = (1u << 31);

llvm::Optional<DIERef> DIERef::Decode(const DataExtractor &data,
                                      lldb::offset_t *offset_ptr) {
  const uint32_t bitfield_storage = data.GetU32(offset_ptr);
  uint32_t dwo_num = bitfield_storage & k_dwo_num_mask;
  bool dwo_num_valid = (bitfield_storage & (k_dwo_num_valid_bitmask)) != 0;
  Section section = (Section)((bitfield_storage & (k_section_bitmask)) != 0);
  // DIE offsets can't be zero and if we fail to decode something from data,
  // it will return 0
  dw_offset_t die_offset = data.GetU32(offset_ptr);
  if (die_offset == 0)
    return std::nullopt;
  if (dwo_num_valid)
    return DIERef(dwo_num, section, die_offset);
  else
    return DIERef(std::nullopt, section, die_offset);
}

void DIERef::Encode(DataEncoder &encoder) const {
  uint32_t bitfield_storage = m_dwo_num;
  if (m_dwo_num_valid)
    bitfield_storage |= k_dwo_num_valid_bitmask;
  if (m_section)
    bitfield_storage |= k_section_bitmask;
  encoder.AppendU32(bitfield_storage);
  static_assert(sizeof(m_die_offset) == 4, "m_die_offset must be 4 bytes");
  encoder.AppendU32(m_die_offset);
}

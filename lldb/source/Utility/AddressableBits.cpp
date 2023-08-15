//===-- AddressableBits.cpp -----------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "lldb/Utility/AddressableBits.h"
#include "lldb/Target/Process.h"
#include "lldb/lldb-types.h"

using namespace lldb;
using namespace lldb_private;

void AddressableBits::SetAddressableBits(uint32_t addressing_bits) {
  m_low_memory_addr_bits = m_high_memory_addr_bits = addressing_bits;
}

void AddressableBits::SetAddressableBits(uint32_t lowmem_addressing_bits,
                                         uint32_t highmem_addressing_bits) {
  m_low_memory_addr_bits = lowmem_addressing_bits;
  m_high_memory_addr_bits = highmem_addressing_bits;
}

void AddressableBits::Clear() {
  m_low_memory_addr_bits = m_high_memory_addr_bits = 0;
}

void AddressableBits::SetProcessMasks(Process &process) {
  // In case either value is set to 0, indicating it was not set, use the
  // other value.
  if (m_low_memory_addr_bits == 0)
    m_low_memory_addr_bits = m_high_memory_addr_bits;
  if (m_high_memory_addr_bits == 0)
    m_high_memory_addr_bits = m_low_memory_addr_bits;

  if (m_low_memory_addr_bits == 0)
    return;

  addr_t address_mask = ~((1ULL << m_low_memory_addr_bits) - 1);
  process.SetCodeAddressMask(address_mask);
  process.SetDataAddressMask(address_mask);

  if (m_low_memory_addr_bits != m_high_memory_addr_bits) {
    lldb::addr_t hi_address_mask = ~((1ULL << m_high_memory_addr_bits) - 1);
    process.SetHighmemCodeAddressMask(hi_address_mask);
    process.SetHighmemDataAddressMask(hi_address_mask);
  }
}

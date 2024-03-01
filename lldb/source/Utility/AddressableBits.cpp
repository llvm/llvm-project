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

void AddressableBits::SetLowmemAddressableBits(
    uint32_t lowmem_addressing_bits) {
  m_low_memory_addr_bits = lowmem_addressing_bits;
}

void AddressableBits::SetHighmemAddressableBits(
    uint32_t highmem_addressing_bits) {
  m_high_memory_addr_bits = highmem_addressing_bits;
}

void AddressableBits::SetProcessMasks(Process &process) {
  if (m_low_memory_addr_bits == 0 && m_high_memory_addr_bits == 0)
    return;

  if (m_low_memory_addr_bits != 0) {
    addr_t low_addr_mask = ~((1ULL << m_low_memory_addr_bits) - 1);
    process.SetCodeAddressMask(low_addr_mask);
    process.SetDataAddressMask(low_addr_mask);
  }

  if (m_high_memory_addr_bits != 0) {
    addr_t hi_addr_mask = ~((1ULL << m_high_memory_addr_bits) - 1);
    process.SetHighmemCodeAddressMask(hi_addr_mask);
    process.SetHighmemDataAddressMask(hi_addr_mask);
  }
}

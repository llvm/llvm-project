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

  // If we don't have an addressable bits value for low memory,
  // see if we have a Code/Data mask already, and use that.
  // Or use the high memory addressable bits value as a last
  // resort.
  addr_t low_addr_mask;
  if (m_low_memory_addr_bits == 0) {
    if (process.GetCodeAddressMask() != UINT64_MAX)
      low_addr_mask = process.GetCodeAddressMask();
    else if (process.GetDataAddressMask() != UINT64_MAX)
      low_addr_mask = process.GetDataAddressMask();
    else
      low_addr_mask = ~((1ULL << m_high_memory_addr_bits) - 1);
  } else {
    low_addr_mask = ~((1ULL << m_low_memory_addr_bits) - 1);
  }

  // If we don't have an addressable bits value for high memory,
  // see if we have a Code/Data mask already, and use that.
  // Or use the low memory addressable bits value as a last
  // resort.
  addr_t hi_addr_mask;
  if (m_high_memory_addr_bits == 0) {
    if (process.GetHighmemCodeAddressMask() != UINT64_MAX)
      hi_addr_mask = process.GetHighmemCodeAddressMask();
    else if (process.GetHighmemDataAddressMask() != UINT64_MAX)
      hi_addr_mask = process.GetHighmemDataAddressMask();
    else
      hi_addr_mask = ~((1ULL << m_low_memory_addr_bits) - 1);
  } else {
    hi_addr_mask = ~((1ULL << m_high_memory_addr_bits) - 1);
  }

  process.SetCodeAddressMask(low_addr_mask);
  process.SetDataAddressMask(low_addr_mask);

  if (low_addr_mask != hi_addr_mask) {
    process.SetHighmemCodeAddressMask(hi_addr_mask);
    process.SetHighmemDataAddressMask(hi_addr_mask);
  }
}

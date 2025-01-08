//===-- NativeRegisterContextDBReg_arm64.cpp ------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "NativeRegisterContextDBReg_arm64.h"

#include "lldb/Utility/LLDBLog.h"
#include "lldb/Utility/Log.h"
#include "lldb/Utility/RegisterValue.h"

using namespace lldb_private;

uint32_t
NativeRegisterContextDBReg_arm64::GetWatchpointSize(uint32_t wp_index) {
  Log *log = GetLog(LLDBLog::Watchpoints);
  LLDB_LOG(log, "wp_index: {0}", wp_index);

  switch ((m_hwp_regs[wp_index].control >> 5) & 0xff) {
  case 0x01:
    return 1;
  case 0x03:
    return 2;
  case 0x0f:
    return 4;
  case 0xff:
    return 8;
  default:
    return 0;
  }
}

std::optional<NativeRegisterContextDBReg::WatchpointDetails>
NativeRegisterContextDBReg_arm64::AdjustWatchpoint(
    const WatchpointDetails &details) {
  size_t size = details.size;
  lldb::addr_t addr = details.addr;
  // Check if size has a valid hardware watchpoint length.
  if (size != 1 && size != 2 && size != 4 && size != 8)
    return std::nullopt;

  // Check 8-byte alignment for hardware watchpoint target address. Below is a
  // hack to recalculate address and size in order to make sure we can watch
  // non 8-byte aligned addresses as well.
  if (addr & 0x07) {
    uint8_t watch_mask = (addr & 0x07) + size;

    if (watch_mask > 0x08)
      return std::nullopt;

    if (watch_mask <= 0x02)
      size = 2;
    else if (watch_mask <= 0x04)
      size = 4;
    else
      size = 8;

    addr = addr & (~0x07);
  }
  return WatchpointDetails{size, addr};
}

uint32_t NativeRegisterContextDBReg_arm64::MakeBreakControlValue(size_t size) {
  // PAC (bits 2:1): 0b10
  const uint32_t pac_bits = 2 << 1;

  // BAS (bits 12:5) hold a bit-mask of addresses to watch
  // e.g. 0b00000001 means 1 byte at address
  //      0b00000011 means 2 bytes (addr..addr+1)
  //      ...
  //      0b11111111 means 8 bytes (addr..addr+7)
  size_t encoded_size = ((1 << size) - 1) << 5;

  // Return encoded hardware breakpoint control value.
  return m_hw_dbg_enable_bit | pac_bits | encoded_size;
}

uint32_t
NativeRegisterContextDBReg_arm64::MakeWatchControlValue(size_t size,
                                                        uint32_t watch_flags) {
  // PAC (bits 2:1): 0b10
  const uint32_t pac_bits = 2 << 1;

  // BAS (bits 12:5) hold a bit-mask of addresses to watch
  // e.g. 0b00000001 means 1 byte at address
  //      0b00000011 means 2 bytes (addr..addr+1)
  //      ...
  //      0b11111111 means 8 bytes (addr..addr+7)
  size_t encoded_size = ((1 << size) - 1) << 5;

  // Return encoded hardware watchpoint control value.
  return m_hw_dbg_enable_bit | pac_bits | encoded_size | (watch_flags << 3);
}

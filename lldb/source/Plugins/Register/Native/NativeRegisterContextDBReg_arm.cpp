//===-- NativeRegisterContextDBReg_arm.cpp --------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "NativeRegisterContextDBReg_arm.h"

#include "lldb/Utility/LLDBLog.h"
#include "lldb/Utility/Log.h"
#include "lldb/Utility/RegisterValue.h"

using namespace lldb_private;

uint32_t NativeRegisterContextDBReg_arm::GetWatchpointSize(uint32_t wp_index) {
  Log *log = GetLog(LLDBLog::Watchpoints);
  LLDB_LOG(log, "wp_index: {0}", wp_index);

  switch ((m_hwp_regs[wp_index].control >> 5) & 0x0f) {
  case 0x01:
    return 1;
  case 0x03:
    return 2;
  case 0x07:
    return 3;
  case 0x0f:
    return 4;
  default:
    return 0;
  }
}

std::optional<NativeRegisterContextDBReg::WatchpointDetails>
NativeRegisterContextDBReg_arm::AdjustWatchpoint(
    const WatchpointDetails &details) {
  auto [size, addr] = details;

  if (size == 0 || size > 4)
    return {};

  // Check 4-byte alignment for hardware watchpoint target address. Below is a
  // hack to recalculate address and size in order to make sure we can watch
  // non 4-byte aligned addresses as well.
  if (addr & 0x03) {
    uint8_t watch_mask = (addr & 0x03) + size;
    if (watch_mask > 0x04)
      return {};
    else if (watch_mask <= 0x02)
      size = 2;
    else
      size = 4;

    addr = addr & (~0x03);
  }

  return WatchpointDetails{size, addr};
}

NativeRegisterContextDBReg::BreakpointDetails
NativeRegisterContextDBReg_arm::AdjustBreakpoint(
    const BreakpointDetails &details) {
  BreakpointDetails bd = details;
  // Use size to get a hint of arm vs thumb modes.
  // LLDB usually aligns this client side, but other clients may not.
  switch (bd.size) {
  case 2:
    bd.addr &= ~1;
    break;
  case 4:
    bd.addr &= ~3;
    break;
  default:
    // We assume that ValidateBreakpoint would have caught this earlier.
    llvm_unreachable("Invalid breakpoint size!");
  }

  return bd;
}

uint32_t NativeRegisterContextDBReg_arm::MakeBreakControlValue(size_t size) {
  switch (size) {
  case 2:
    return (0x3 << 5) | 7;
  case 4:
    return (0xfu << 5) | 7;
  default:
    // ValidateBreakpoint would have rejected this earlier.
    llvm_unreachable("Invalid breakpoint size.");
  }
}

uint32_t
NativeRegisterContextDBReg_arm::MakeWatchControlValue(size_t size,
                                                      uint32_t watch_flags) {
  // We can only watch up to four bytes that follow a 4 byte aligned address
  // per watchpoint register pair, so make sure we can properly encode this.
  // We assume that the address was 4 byte aligned by AdjustWatchpoint.
  uint32_t byte_mask = (1u << size) - 1u;

  // Check if we need multiple watchpoint register
  if (byte_mask > 0xfu)
    return LLDB_INVALID_INDEX32;

  // Setup control value
  // Make the byte_mask into a valid Byte Address Select mask
  uint32_t control_value = byte_mask << 5;

  // Turn on appropriate watchpoint flags read or write
  control_value |= (watch_flags << 3);

  // Enable this watchpoint and make it stop in privileged or user mode;
  control_value |= 7;

  return control_value;
}

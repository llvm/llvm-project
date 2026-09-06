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

#include "llvm/Support/MathExtras.h"

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
  // For the way we are using BAS watchpoints, their size must be a power of 2
  // that is less than 8. Note that this is a subset of what hardware allows.
  const size_t max_size = 8;
  size_t size = details.size;
  if (!llvm::isPowerOf2_64(size) || size > max_size)
    return std::nullopt;

  // The start address must be aligned to 8 bytes.
  lldb::addr_t addr = details.addr;
  const size_t misalignment = addr & (max_size - 1);
  if (misalignment == 0)
    return details;

  // The start address is not aligned, but we might be able to expand the
  // watched range backwards to the previous 8 byte aligned address while
  // keeping the size <= 8 bytes.
  //
  //  Aligned Address
  //  |     /--------- Start Address
  // [ ][ ][ ][ ][ ][ ][ ][ ][ ][ ][ ]
  //  |     |<-------- Size ------->|
  //  |<----- Aligned Size -------->|
  //
  size_t aligned_size = misalignment + size;
  if (aligned_size > max_size) {
    // The range does not fit within a single BAS watchpoint, reject it.
    return std::nullopt;
  }

  // Size must still be a power of 2. After correcting this, sometimes we will
  // watch bytes before and after the intended range. Stops in these extra bytes
  // will be filtered out.
  //
  // For example, A is an aligned address and S is the start address. You want
  // to watch the 1 byte at address S.
  // [A][ ][S][ ]
  //       { }
  // This is misaligned by 2, so our aligned size is 3.
  // [A][ ][S][ ]
  // {       }
  // We cannot watch 3 bytes but we can watch 4.
  // [A][ ][S][ ]
  // {          }
  // The watched range is expanded before and after S.

  // Round up to a power of 2 size.
  size = llvm::PowerOf2Ceil(aligned_size);
  // Align the address down.
  addr -= misalignment;

  return WatchpointDetails{size, addr};
}

uint32_t NativeRegisterContextDBReg_arm64::MakeBreakControlValue(size_t size) {
  // PAC (bits 2:1): 0b10
  const uint32_t pac_bits = 2 << 1;

  // BAS (bits 12:5) hold a bit-mask of addresses to watch.
  // Hardware does allow gaps but we only support unbroken ranges at this time
  //
  // e.g. 0b00000001 means 1 byte at address
  //      0b00000011 means 2 bytes (addr..addr+1)
  //      ...
  //      0b11111111 means 8 bytes (addr..addr+7)
  size_t encoded_size = ((1 << size) - 1) << 5;

  return m_hw_dbg_enable_bit | pac_bits | encoded_size;
}

uint32_t
NativeRegisterContextDBReg_arm64::MakeWatchControlValue(size_t size,
                                                        uint32_t watch_flags) {
  // PAC (bits 2:1): 0b10
  const uint32_t pac_bits = 2 << 1;

  // BAS (bits 12:5) hold a bit-mask of addresses to watch.
  // Hardware does allow gaps but we only support unbroken ranges at this time.
  //
  // e.g. 0b00000001 means 1 byte at address
  //      0b00000011 means 2 bytes (addr..addr+1)
  //      ...
  //      0b11111111 means 8 bytes (addr..addr+7)
  size_t encoded_size = ((1 << size) - 1) << 5;

  return m_hw_dbg_enable_bit | pac_bits | encoded_size | (watch_flags << 3);
}

//===-- RegisterContextDarwin_arm64.cpp -----------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "RegisterContextDarwin_arm64.h"

#include "lldb/Register/Arch/lldb-arm64-dwarf-register-enums.h"
#include "lldb/Register/Arch/lldb-arm64-register-enums.h"
#include "lldb/lldb-defines.h"
#include "lldb/lldb-enumerations.h"
#include "lldb/lldb-private-types.h"
#include "lldb/lldb-private.h"
#include <cstddef>
#include <cstdint>

#include "RegisterContextDarwinConstants.h"

#include "lldb/Target/Process.h"
#include "lldb/Target/Thread.h"
#include "lldb/Utility/DataBufferHeap.h"
#include "lldb/Utility/DataExtractor.h"
#include "lldb/Utility/Endian.h"
#include "lldb/Utility/Log.h"
#include "lldb/Register/RegisterValue.h"
#include "lldb/Utility/Scalar.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/Compiler.h"

#include "Plugins/Process/Utility/InstructionUtils.h"

#include <memory>

#if defined(__APPLE__) && (defined(__arm64__) || defined(__aarch64__))
#include <sys/types.h>
#include <sys/sysctl.h>
#endif

#include "lldb/Register/Arch/lldb-arm64-dwarf-register-enums.h"

namespace reg_ids = lldb_private::registers::arm64::id;
using namespace lldb;
using namespace lldb_private;

#define REG_CONTEXT_SIZE                                                       \
  (sizeof(RegisterContextDarwin_arm64::GPR) +                                  \
   sizeof(RegisterContextDarwin_arm64::FPU) +                                  \
   sizeof(RegisterContextDarwin_arm64::EXC))

RegisterContextDarwin_arm64::RegisterContextDarwin_arm64(
    Thread &thread, uint32_t concrete_frame_idx)
    : ThreadRegisterContext(thread, concrete_frame_idx), gpr(), fpu(), exc(),
      dbg() {
  SetRegisterLayout(RegisterLayoutDarwin_arm64::GetShared());

  uint32_t i;
  for (i = 0; i < kNumErrors; i++) {
    gpr_errs[i] = -1;
    fpu_errs[i] = -1;
    exc_errs[i] = -1;
  }
}

RegisterContextDarwin_arm64::~RegisterContextDarwin_arm64() = default;

void RegisterContextDarwin_arm64::InvalidateAllRegisters() {
  InvalidateAllRegisterStates();
}

int RegisterContextDarwin_arm64::ReadGPR(bool force) {
  int set = GPRRegSet;
  if (force || !RegisterSetIsCached(set)) {
    SetError(set, Read, DoReadGPR(GetThreadID(), set, gpr));
  }
  return GetError(GPRRegSet, Read);
}

int RegisterContextDarwin_arm64::ReadFPU(bool force) {
  int set = FPURegSet;
  if (force || !RegisterSetIsCached(set)) {
    SetError(set, Read, DoReadFPU(GetThreadID(), set, fpu));
  }
  return GetError(FPURegSet, Read);
}

int RegisterContextDarwin_arm64::ReadEXC(bool force) {
  int set = EXCRegSet;
  if (force || !RegisterSetIsCached(set)) {
    SetError(set, Read, DoReadEXC(GetThreadID(), set, exc));
  }
  return GetError(EXCRegSet, Read);
}

int RegisterContextDarwin_arm64::ReadDBG(bool force) {
  int set = DBGRegSet;
  if (force || !RegisterSetIsCached(set)) {
    SetError(set, Read, DoReadDBG(GetThreadID(), set, dbg));
  }
  return GetError(DBGRegSet, Read);
}

int RegisterContextDarwin_arm64::WriteGPR() {
  int set = GPRRegSet;
  if (!RegisterSetIsCached(set)) {
    SetError(set, Write, -1);
    return KERN_INVALID_ARGUMENT;
  }
  SetError(set, Write, DoWriteGPR(GetThreadID(), set, gpr));
  SetError(set, Read, -1);
  return GetError(GPRRegSet, Write);
}

int RegisterContextDarwin_arm64::WriteFPU() {
  int set = FPURegSet;
  if (!RegisterSetIsCached(set)) {
    SetError(set, Write, -1);
    return KERN_INVALID_ARGUMENT;
  }
  SetError(set, Write, DoWriteFPU(GetThreadID(), set, fpu));
  SetError(set, Read, -1);
  return GetError(FPURegSet, Write);
}

int RegisterContextDarwin_arm64::WriteEXC() {
  int set = EXCRegSet;
  if (!RegisterSetIsCached(set)) {
    SetError(set, Write, -1);
    return KERN_INVALID_ARGUMENT;
  }
  SetError(set, Write, DoWriteEXC(GetThreadID(), set, exc));
  SetError(set, Read, -1);
  return GetError(EXCRegSet, Write);
}

int RegisterContextDarwin_arm64::WriteDBG() {
  int set = DBGRegSet;
  if (!RegisterSetIsCached(set)) {
    SetError(set, Write, -1);
    return KERN_INVALID_ARGUMENT;
  }
  SetError(set, Write, DoWriteDBG(GetThreadID(), set, dbg));
  SetError(set, Read, -1);
  return GetError(DBGRegSet, Write);
}

int RegisterContextDarwin_arm64::ReadRegisterSet(uint32_t set, bool force) {
  switch (set) {
  case GPRRegSet:
    return ReadGPR(force);
  case FPURegSet:
    return ReadFPU(force);
  case EXCRegSet:
    return ReadEXC(force);
  case DBGRegSet:
    return ReadDBG(force);
  default:
    break;
  }
  return KERN_INVALID_ARGUMENT;
}

int RegisterContextDarwin_arm64::WriteRegisterSet(uint32_t set) {
  // Make sure we have a valid context to set.
  if (RegisterSetIsCached(set)) {
    switch (set) {
    case GPRRegSet:
      return WriteGPR();
    case FPURegSet:
      return WriteFPU();
    case EXCRegSet:
      return WriteEXC();
    case DBGRegSet:
      return WriteDBG();
    default:
      break;
    }
  }
  return KERN_INVALID_ARGUMENT;
}

void RegisterContextDarwin_arm64::LogDBGRegisters(Log *log, const DBG &dbg) {
  if (log) {
    for (uint32_t i = 0; i < 16; i++)
      LLDB_LOGF(log,
                "BVR%-2u/BCR%-2u = { 0x%8.8" PRIu64 ", 0x%8.8" PRIu64
                " } WVR%-2u/WCR%-2u "
                "= { 0x%8.8" PRIu64 ", 0x%8.8" PRIu64 " }",
                i, i, dbg.bvr[i], dbg.bcr[i], i, i, dbg.wvr[i], dbg.wcr[i]);
  }
}

llvm::Expected<RegisterValue>
RegisterContextDarwin_arm64::ReadRegister(RegisterID id) {
  if (!GetDescription()->Find(id))
    return llvm::createStringError("unknown register ID");
  // Preserve the register views supported by the Darwin context.
  if (id >= reg_ids::fpu_s13 && id <= reg_ids::fpu_s15)
    return llvm::createStringError("register value is unavailable");

  auto required = GetLayout().GetRequiredBanks(id);
  if (!required)
    return required.takeError();
  for (RegisterBankID bank : *required)
    if (bank == DBGRegSet || ReadRegisterSet(bank, false) != 0)
      return llvm::createStringError("register value is unavailable");

  const RegisterData banks[] = {
      {GPRRegSet, {reinterpret_cast<const uint8_t *>(&gpr), sizeof(gpr)}},
      {FPURegSet, {reinterpret_cast<const uint8_t *>(&fpu), sizeof(fpu)}},
      {EXCRegSet, {reinterpret_cast<const uint8_t *>(&exc), sizeof(exc)}},
      {DBGRegSet, {reinterpret_cast<const uint8_t *>(&dbg), sizeof(dbg)}},
  };
  return GetLayout().Read(id, banks);
}

llvm::Error
RegisterContextDarwin_arm64::WriteRegister(RegisterID id,
                                           const RegisterValue &value) {
  if (!GetDescription()->Find(id))
    return llvm::createStringError("unknown register ID");
  // Preserve the existing restriction on writes through subregister views.
  if ((id >= reg_ids::gpr_w0 && id <= reg_ids::gpr_w28) ||
      (id >= reg_ids::fpu_s0 && id <= reg_ids::fpu_d31))
    return llvm::createStringError("register write is unavailable");

  auto required = GetLayout().GetRequiredBanks(id, true);
  if (!required)
    return required.takeError();
  for (RegisterBankID bank : *required)
    if (bank == DBGRegSet || ReadRegisterSet(bank, false) != 0)
      return llvm::createStringError("register write is unavailable");

  const MutableRegisterData banks[] = {
      {GPRRegSet, {reinterpret_cast<uint8_t *>(&gpr), sizeof(gpr)}},
      {FPURegSet, {reinterpret_cast<uint8_t *>(&fpu), sizeof(fpu)}},
      {EXCRegSet, {reinterpret_cast<uint8_t *>(&exc), sizeof(exc)}},
      {DBGRegSet, {reinterpret_cast<uint8_t *>(&dbg), sizeof(dbg)}},
  };
  if (auto error = GetLayout().Write(id, value, banks))
    return error;

  for (RegisterBankID bank : *required)
    if (WriteRegisterSet(bank) != 0)
      return llvm::createStringError("register write failed");

  return llvm::Error::success();
}

bool RegisterContextDarwin_arm64::ReadAllRegisterValues(
    lldb::WritableDataBufferSP &data_sp) {
  data_sp = std::make_shared<DataBufferHeap>(REG_CONTEXT_SIZE, 0);
  if (ReadGPR(false) == KERN_SUCCESS && ReadFPU(false) == KERN_SUCCESS &&
      ReadEXC(false) == KERN_SUCCESS) {
    uint8_t *dst = data_sp->GetBytes();
    ::memcpy(dst, &gpr, sizeof(gpr));
    dst += sizeof(gpr);

    ::memcpy(dst, &fpu, sizeof(fpu));
    dst += sizeof(gpr);

    ::memcpy(dst, &exc, sizeof(exc));
    return true;
  }
  return false;
}

bool RegisterContextDarwin_arm64::WriteAllRegisterValues(
    const lldb::DataBufferSP &data_sp) {
  if (data_sp && data_sp->GetByteSize() == REG_CONTEXT_SIZE) {
    const uint8_t *src = data_sp->GetBytes();
    ::memcpy(&gpr, src, sizeof(gpr));
    src += sizeof(gpr);

    ::memcpy(&fpu, src, sizeof(fpu));
    src += sizeof(gpr);

    ::memcpy(&exc, src, sizeof(exc));
    uint32_t success_count = 0;
    if (WriteGPR() == KERN_SUCCESS)
      ++success_count;
    if (WriteFPU() == KERN_SUCCESS)
      ++success_count;
    if (WriteEXC() == KERN_SUCCESS)
      ++success_count;
    return success_count == 3;
  }
  return false;
}

uint32_t RegisterContextDarwin_arm64::NumSupportedHardwareWatchpoints() {
#if defined(__APPLE__) && (defined(__arm64__) || defined(__aarch64__))
  // autodetect how many watchpoints are supported dynamically...
  static uint32_t g_num_supported_hw_watchpoints = UINT32_MAX;
  if (g_num_supported_hw_watchpoints == UINT32_MAX) {
    size_t len;
    uint32_t n = 0;
    len = sizeof(n);
    if (::sysctlbyname("hw.optional.watchpoint", &n, &len, NULL, 0) == 0) {
      g_num_supported_hw_watchpoints = n;
    }
  }
  return g_num_supported_hw_watchpoints;
#else
  // TODO: figure out remote case here!
  return 2;
#endif
}

uint32_t RegisterContextDarwin_arm64::SetHardwareWatchpoint(lldb::addr_t addr,
                                                            size_t size,
                                                            bool read,
                                                            bool write) {
  //    if (log) log->Printf
  //    ("RegisterContextDarwin_arm64::EnableHardwareWatchpoint(addr = %8.8p,
  //    size = %u, read = %u, write = %u)", addr, size, read, write);

  const uint32_t num_hw_watchpoints = NumSupportedHardwareWatchpoints();

  // Can't watch zero bytes
  if (size == 0)
    return LLDB_INVALID_INDEX32;

  // We must watch for either read or write
  if (!read && !write)
    return LLDB_INVALID_INDEX32;

  // Can't watch more than 4 bytes per WVR/WCR pair
  if (size > 4)
    return LLDB_INVALID_INDEX32;

  // We can only watch up to four bytes that follow a 4 byte aligned address
  // per watchpoint register pair. Since we have at most so we can only watch
  // until the next 4 byte boundary and we need to make sure we can properly
  // encode this.
  uint32_t addr_word_offset = addr % 4;
  //    if (log) log->Printf
  //    ("RegisterContextDarwin_arm64::EnableHardwareWatchpoint() -
  //    addr_word_offset = 0x%8.8x", addr_word_offset);

  uint32_t byte_mask = ((1u << size) - 1u) << addr_word_offset;
  //    if (log) log->Printf
  //    ("RegisterContextDarwin_arm64::EnableHardwareWatchpoint() - byte_mask =
  //    0x%8.8x", byte_mask);
  if (byte_mask > 0xfu)
    return LLDB_INVALID_INDEX32;

  // Read the debug state
  int kret = ReadDBG(false);

  if (kret == KERN_SUCCESS) {
    // Check to make sure we have the needed hardware support
    uint32_t i = 0;

    for (i = 0; i < num_hw_watchpoints; ++i) {
      if ((dbg.wcr[i] & WCR_ENABLE) == 0)
        break; // We found an available hw breakpoint slot (in i)
    }

    // See if we found an available hw breakpoint slot above
    if (i < num_hw_watchpoints) {
      // Make the byte_mask into a valid Byte Address Select mask
      uint32_t byte_address_select = byte_mask << 5;
      // Make sure bits 1:0 are clear in our address
      dbg.wvr[i] = addr & ~((lldb::addr_t)3);
      dbg.wcr[i] = byte_address_select |     // Which bytes that follow the IMVA
                                             // that we will watch
                   S_USER |                  // Stop only in user mode
                   (read ? WCR_LOAD : 0) |   // Stop on read access?
                   (write ? WCR_STORE : 0) | // Stop on write access?
                   WCR_ENABLE;               // Enable this watchpoint;

      kret = WriteDBG();
      //            if (log) log->Printf
      //            ("RegisterContextDarwin_arm64::EnableHardwareWatchpoint()
      //            WriteDBG() => 0x%8.8x.", kret);

      if (kret == KERN_SUCCESS)
        return i;
    } else {
      //            if (log) log->Printf
      //            ("RegisterContextDarwin_arm64::EnableHardwareWatchpoint():
      //            All hardware resources (%u) are in use.",
      //            num_hw_watchpoints);
    }
  }
  return LLDB_INVALID_INDEX32;
}

bool RegisterContextDarwin_arm64::ClearHardwareWatchpoint(uint32_t hw_index) {
  int kret = ReadDBG(false);

  const uint32_t num_hw_points = NumSupportedHardwareWatchpoints();
  if (kret == KERN_SUCCESS) {
    if (hw_index < num_hw_points) {
      dbg.wcr[hw_index] = 0;
      //            if (log) log->Printf
      //            ("RegisterContextDarwin_arm64::ClearHardwareWatchpoint( %u )
      //            - WVR%u = 0x%8.8x  WCR%u = 0x%8.8x",
      //                    hw_index,
      //                    hw_index,
      //                    dbg.wvr[hw_index],
      //                    hw_index,
      //                    dbg.wcr[hw_index]);

      kret = WriteDBG();

      if (kret == KERN_SUCCESS)
        return true;
    }
  }
  return false;
}

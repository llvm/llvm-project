//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "RegisterContextDarwin_arm32.h"
#include "RegisterContextDarwinConstants.h"

#include "lldb/Register/RegisterValue.h"
#include "lldb/Utility/DataBufferHeap.h"
#include "lldb/Utility/DataExtractor.h"
#include "lldb/Utility/Endian.h"
#include "lldb/Utility/Log.h"
#include "lldb/Utility/Scalar.h"
#include "llvm/Support/Compiler.h"

#include "Plugins/Process/Utility/InstructionUtils.h"

#include <memory>

#include "llvm/ADT/STLExtras.h"

using namespace lldb;
using namespace lldb_private;

#define REG_CONTEXT_SIZE                                                       \
  (sizeof(RegisterContextDarwin_arm32::GPR) +                                  \
   sizeof(RegisterContextDarwin_arm32::FPU) +                                  \
   sizeof(RegisterContextDarwin_arm32::EXC))

// General purpose registers
RegisterContextDarwin_arm32::RegisterContextDarwin_arm32(
    Thread &thread, uint32_t concrete_frame_idx)
    : ThreadRegisterContext(thread, concrete_frame_idx), gpr(), fpu(), exc() {
  SetRegisterLayout(RegisterLayoutDarwin_arm32::GetShared());

  uint32_t i;
  for (i = 0; i < kNumErrors; i++) {
    gpr_errs[i] = -1;
    fpu_errs[i] = -1;
    exc_errs[i] = -1;
  }
}

RegisterContextDarwin_arm32::~RegisterContextDarwin_arm32() = default;

void RegisterContextDarwin_arm32::InvalidateAllRegisters() {
  InvalidateAllRegisterStates();
}

int RegisterContextDarwin_arm32::ReadGPR(bool force) {
  int set = GPRRegSet;
  if (force || !RegisterSetIsCached(set)) {
    SetError(set, Read, DoReadGPR(GetThreadID(), set, gpr));
  }
  return GetError(GPRRegSet, Read);
}

int RegisterContextDarwin_arm32::ReadFPU(bool force) {
  int set = FPURegSet;
  if (force || !RegisterSetIsCached(set)) {
    SetError(set, Read, DoReadFPU(GetThreadID(), set, fpu));
  }
  return GetError(FPURegSet, Read);
}

int RegisterContextDarwin_arm32::ReadEXC(bool force) {
  int set = EXCRegSet;
  if (force || !RegisterSetIsCached(set)) {
    SetError(set, Read, DoReadEXC(GetThreadID(), set, exc));
  }
  return GetError(EXCRegSet, Read);
}

int RegisterContextDarwin_arm32::ReadDBG(bool force) {
  int set = DBGRegSet;
  if (force || !RegisterSetIsCached(set)) {
    SetError(set, Read, DoReadDBG(GetThreadID(), set, dbg));
  }
  return GetError(DBGRegSet, Read);
}

int RegisterContextDarwin_arm32::WriteGPR() {
  int set = GPRRegSet;
  if (!RegisterSetIsCached(set)) {
    SetError(set, Write, -1);
    return KERN_INVALID_ARGUMENT;
  }
  SetError(set, Write, DoWriteGPR(GetThreadID(), set, gpr));
  SetError(set, Read, -1);
  return GetError(GPRRegSet, Write);
}

int RegisterContextDarwin_arm32::WriteFPU() {
  int set = FPURegSet;
  if (!RegisterSetIsCached(set)) {
    SetError(set, Write, -1);
    return KERN_INVALID_ARGUMENT;
  }
  SetError(set, Write, DoWriteFPU(GetThreadID(), set, fpu));
  SetError(set, Read, -1);
  return GetError(FPURegSet, Write);
}

int RegisterContextDarwin_arm32::WriteEXC() {
  int set = EXCRegSet;
  if (!RegisterSetIsCached(set)) {
    SetError(set, Write, -1);
    return KERN_INVALID_ARGUMENT;
  }
  SetError(set, Write, DoWriteEXC(GetThreadID(), set, exc));
  SetError(set, Read, -1);
  return GetError(EXCRegSet, Write);
}

int RegisterContextDarwin_arm32::WriteDBG() {
  int set = DBGRegSet;
  if (!RegisterSetIsCached(set)) {
    SetError(set, Write, -1);
    return KERN_INVALID_ARGUMENT;
  }
  SetError(set, Write, DoWriteDBG(GetThreadID(), set, dbg));
  SetError(set, Read, -1);
  return GetError(DBGRegSet, Write);
}

int RegisterContextDarwin_arm32::ReadRegisterSet(uint32_t set, bool force) {
  switch (set) {
  case GPRRegSet:
    return ReadGPR(force);
  case GPRAltRegSet:
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

int RegisterContextDarwin_arm32::WriteRegisterSet(uint32_t set) {
  // Make sure we have a valid context to set.
  if (RegisterSetIsCached(set)) {
    switch (set) {
    case GPRRegSet:
      return WriteGPR();
    case GPRAltRegSet:
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

void RegisterContextDarwin_arm32::LogDBGRegisters(Log *log, const DBG &dbg) {
  if (log) {
    for (uint32_t i = 0; i < 16; i++)
      LLDB_LOGF(log,
                "BVR%-2u/BCR%-2u = { 0x%8.8x, 0x%8.8x } WVR%-2u/WCR%-2u = { "
                "0x%8.8x, 0x%8.8x }",
                i, i, dbg.bvr[i], dbg.bcr[i], i, i, dbg.wvr[i], dbg.wcr[i]);
  }
}

llvm::Expected<RegisterValue>
RegisterContextDarwin_arm32::ReadRegister(RegisterID id) {
  if (!GetDescription()->Find(id))
    return llvm::createStringError("unknown register ID");

  auto required = GetLayout().GetRequiredBanks(id);
  if (!required)
    return required.takeError();
  for (RegisterBankID bank : *required)
    if (ReadRegisterSet(bank, false) != 0)
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
RegisterContextDarwin_arm32::WriteRegister(RegisterID id,
                                           const RegisterValue &value) {
  if (!GetDescription()->Find(id))
    return llvm::createStringError("unknown register ID");

  auto required = GetLayout().GetRequiredBanks(id, true);
  if (!required)
    return required.takeError();
  for (RegisterBankID bank : *required)
    if (ReadRegisterSet(bank, false) != 0)
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

bool RegisterContextDarwin_arm32::ReadAllRegisterValues(
    lldb::WritableDataBufferSP &data_sp) {
  data_sp = std::make_shared<DataBufferHeap>(REG_CONTEXT_SIZE, 0);
  if (data_sp && ReadGPR(false) == KERN_SUCCESS &&
      ReadFPU(false) == KERN_SUCCESS && ReadEXC(false) == KERN_SUCCESS) {
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

bool RegisterContextDarwin_arm32::WriteAllRegisterValues(
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

uint32_t RegisterContextDarwin_arm32::NumSupportedHardwareBreakpoints() {
#if defined(__APPLE__) && defined(__arm__)
  // Set the init value to something that will let us know that we need to
  // autodetect how many breakpoints are supported dynamically...
  static uint32_t g_num_supported_hw_breakpoints = UINT32_MAX;
  if (g_num_supported_hw_breakpoints == UINT32_MAX) {
    // Set this to zero in case we can't tell if there are any HW breakpoints
    g_num_supported_hw_breakpoints = 0;

    uint32_t register_DBGDIDR;

    asm("mrc p14, 0, %0, c0, c0, 0" : "=r"(register_DBGDIDR));
    g_num_supported_hw_breakpoints = Bits32(register_DBGDIDR, 27, 24);
    // Zero is reserved for the BRP count, so don't increment it if it is zero
    if (g_num_supported_hw_breakpoints > 0)
      g_num_supported_hw_breakpoints++;
  }
  return g_num_supported_hw_breakpoints;
#else
  // TODO: figure out remote case here!
  return 6;
#endif
}

uint32_t RegisterContextDarwin_arm32::SetHardwareBreakpoint(lldb::addr_t addr,
                                                            size_t size) {
  // Make sure our address isn't bogus
  if (addr & 1)
    return LLDB_INVALID_INDEX32;

  int kret = ReadDBG(false);

  if (kret == KERN_SUCCESS) {
    const uint32_t num_hw_breakpoints = NumSupportedHardwareBreakpoints();
    uint32_t i;
    for (i = 0; i < num_hw_breakpoints; ++i) {
      if ((dbg.bcr[i] & BCR_ENABLE) == 0)
        break; // We found an available hw breakpoint slot (in i)
    }

    // See if we found an available hw breakpoint slot above
    if (i < num_hw_breakpoints) {
      // Make sure bits 1:0 are clear in our address
      dbg.bvr[i] = addr & ~((lldb::addr_t)3);

      if (size == 2 || addr & 2) {
        uint32_t byte_addr_select = (addr & 2) ? BAS_IMVA_2_3 : BAS_IMVA_0_1;

        // We have a thumb breakpoint
        // We have an ARM breakpoint
        dbg.bcr[i] = BCR_M_IMVA_MATCH | // Stop on address match
                     byte_addr_select | // Set the correct byte address select
                                        // so we only trigger on the correct
                                        // opcode
                     S_USER |    // Which modes should this breakpoint stop in?
                     BCR_ENABLE; // Enable this hardware breakpoint
                                 //                if (log) log->Printf
        //                ("RegisterContextDarwin_arm32::EnableHardwareBreakpoint(
        //                addr = %8.8p, size = %u ) - BVR%u/BCR%u = 0x%8.8x /
        //                0x%8.8x (Thumb)",
        //                        addr,
        //                        size,
        //                        i,
        //                        i,
        //                        dbg.bvr[i],
        //                        dbg.bcr[i]);
      } else if (size == 4) {
        // We have an ARM breakpoint
        dbg.bcr[i] =
            BCR_M_IMVA_MATCH | // Stop on address match
            BAS_IMVA_ALL | // Stop on any of the four bytes following the IMVA
            S_USER |       // Which modes should this breakpoint stop in?
            BCR_ENABLE;    // Enable this hardware breakpoint
                           //                if (log) log->Printf
        //                ("RegisterContextDarwin_arm32::EnableHardwareBreakpoint(
        //                addr = %8.8p, size = %u ) - BVR%u/BCR%u = 0x%8.8x /
        //                0x%8.8x (ARM)",
        //                        addr,
        //                        size,
        //                        i,
        //                        i,
        //                        dbg.bvr[i],
        //                        dbg.bcr[i]);
      }

      kret = WriteDBG();
      //            if (log) log->Printf
      //            ("RegisterContextDarwin_arm32::EnableHardwareBreakpoint()
      //            WriteDBG() => 0x%8.8x.", kret);

      if (kret == KERN_SUCCESS)
        return i;
    }
    //        else
    //        {
    //            if (log) log->Printf
    //            ("RegisterContextDarwin_arm32::EnableHardwareBreakpoint(addr =
    //            %8.8p, size = %u) => all hardware breakpoint resources are
    //            being used.", addr, size);
    //        }
  }

  return LLDB_INVALID_INDEX32;
}

bool RegisterContextDarwin_arm32::ClearHardwareBreakpoint(uint32_t hw_index) {
  int kret = ReadDBG(false);

  const uint32_t num_hw_points = NumSupportedHardwareBreakpoints();
  if (kret == KERN_SUCCESS) {
    if (hw_index < num_hw_points) {
      dbg.bcr[hw_index] = 0;
      //            if (log) log->Printf
      //            ("RegisterContextDarwin_arm32::SetHardwareBreakpoint( %u ) -
      //            BVR%u = 0x%8.8x  BCR%u = 0x%8.8x",
      //                    hw_index,
      //                    hw_index,
      //                    dbg.bvr[hw_index],
      //                    hw_index,
      //                    dbg.bcr[hw_index]);

      kret = WriteDBG();

      if (kret == KERN_SUCCESS)
        return true;
    }
  }
  return false;
}

uint32_t RegisterContextDarwin_arm32::NumSupportedHardwareWatchpoints() {
#if defined(__APPLE__) && defined(__arm__)
  // Set the init value to something that will let us know that we need to
  // autodetect how many watchpoints are supported dynamically...
  static uint32_t g_num_supported_hw_watchpoints = UINT32_MAX;
  if (g_num_supported_hw_watchpoints == UINT32_MAX) {
    // Set this to zero in case we can't tell if there are any HW breakpoints
    g_num_supported_hw_watchpoints = 0;

    uint32_t register_DBGDIDR;
    asm("mrc p14, 0, %0, c0, c0, 0" : "=r"(register_DBGDIDR));
    g_num_supported_hw_watchpoints = Bits32(register_DBGDIDR, 31, 28) + 1;
  }
  return g_num_supported_hw_watchpoints;
#else
  // TODO: figure out remote case here!
  return 2;
#endif
}

uint32_t RegisterContextDarwin_arm32::SetHardwareWatchpoint(lldb::addr_t addr,
                                                            size_t size,
                                                            bool read,
                                                            bool write) {
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
  //    ("RegisterContextDarwin_arm32::EnableHardwareWatchpoint() -
  //    addr_word_offset = 0x%8.8x", addr_word_offset);

  uint32_t byte_mask = ((1u << size) - 1u) << addr_word_offset;
  //    if (log) log->Printf
  //    ("RegisterContextDarwin_arm32::EnableHardwareWatchpoint() - byte_mask =
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
      //            ("RegisterContextDarwin_arm32::EnableHardwareWatchpoint()
      //            WriteDBG() => 0x%8.8x.", kret);

      if (kret == KERN_SUCCESS)
        return i;
    } else {
      //            if (log) log->Printf
      //            ("RegisterContextDarwin_arm32::EnableHardwareWatchpoint():
      //            All hardware resources (%u) are in use.",
      //            num_hw_watchpoints);
    }
  }
  return LLDB_INVALID_INDEX32;
}

bool RegisterContextDarwin_arm32::ClearHardwareWatchpoint(uint32_t hw_index) {
  int kret = ReadDBG(false);

  const uint32_t num_hw_points = NumSupportedHardwareWatchpoints();
  if (kret == KERN_SUCCESS) {
    if (hw_index < num_hw_points) {
      dbg.wcr[hw_index] = 0;
      //            if (log) log->Printf
      //            ("RegisterContextDarwin_arm32::ClearHardwareWatchpoint( %u )
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

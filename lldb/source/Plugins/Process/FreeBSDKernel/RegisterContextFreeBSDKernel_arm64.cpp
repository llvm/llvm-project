//===-- RegisterContextFreeBSDKernel_arm64.cpp ----------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "RegisterContextFreeBSDKernel_arm64.h"
#include "Plugins/Process/Utility/lldb-arm64-register-enums.h"

#include "lldb/Symbol/Symbol.h"
#include "lldb/Target/Process.h"
#include "lldb/Target/Target.h"
#include "lldb/Target/Thread.h"
#include "lldb/Utility/LLDBLog.h"
#include "lldb/Utility/Log.h"
#include "lldb/Utility/RegisterValue.h"
#include "llvm/Support/Endian.h"

using namespace lldb;
using namespace lldb_private;

RegisterContextFreeBSDKernel_arm64::RegisterContextFreeBSDKernel_arm64(
    Thread &thread, std::unique_ptr<RegisterInfoPOSIX_arm64> register_info_up,
    lldb::addr_t pcb_addr)
    : RegisterContextPOSIX_arm64(thread, std::move(register_info_up)),
      m_pcb_addr(pcb_addr) {}

bool RegisterContextFreeBSDKernel_arm64::ReadGPR() { return true; }

bool RegisterContextFreeBSDKernel_arm64::ReadFPR() { return true; }

bool RegisterContextFreeBSDKernel_arm64::WriteGPR() {
  assert(0);
  return false;
}

bool RegisterContextFreeBSDKernel_arm64::WriteFPR() {
  assert(0);
  return false;
}

bool RegisterContextFreeBSDKernel_arm64::ReadRegister(
    const RegisterInfo *reg_info, RegisterValue &value) {
  if (m_pcb_addr == LLDB_INVALID_ADDRESS)
    return false;

  // https://cgit.freebsd.org/src/tree/sys/arm64/include/pcb.h
  struct {
    llvm::support::ulittle64_t x[12];
    llvm::support::ulittle64_t sp;
  } pcb;

  // https://cgit.freebsd.org/src/tree/sys/arm64/include/pcb.h?h=stable%2F13
  struct {
    llvm::support::ulittle64_t x[30];
    llvm::support::ulittle64_t lr;
    llvm::support::ulittle64_t _reserved;
    llvm::support::ulittle64_t sp;
  } pcb13;

  Status error;
  constexpr int FBSD14 = 1400084;
  int osreldate = FBSD14;

  if (auto osreldate_or_null = GetOsreldate())
    osreldate = *osreldate_or_null;
  else
    LLDB_LOGF(GetLog(LLDBLog::Object),
              "Cannot find osreldate. Defaulting to %d.", FBSD14);

  // TODO: LLVM 24: Remove FreeBSD 13 support
  if (osreldate >= FBSD14) {
    constexpr uint32_t PCB_FP = 10;
    constexpr uint32_t PCB_LR = 11;
    size_t rd =
        m_thread.GetProcess()->ReadMemory(m_pcb_addr, &pcb, sizeof(pcb), error);
    if (rd != sizeof(pcb))
      return false;

    uint32_t reg = reg_info->kinds[lldb::eRegisterKindLLDB];
    switch (reg) {
    case gpr_x19_arm64:
    case gpr_x20_arm64:
    case gpr_x21_arm64:
    case gpr_x22_arm64:
    case gpr_x23_arm64:
    case gpr_x24_arm64:
    case gpr_x25_arm64:
    case gpr_x26_arm64:
    case gpr_x27_arm64:
    case gpr_x28_arm64:
    case gpr_fp_arm64:
      static_assert(gpr_fp_arm64 - gpr_x19_arm64 == PCB_FP,
                    "nonconsecutive arm64 register numbers");
      value = pcb.x[reg - gpr_x19_arm64];
      break;
    case gpr_lr_arm64:
    case gpr_pc_arm64:
      // The pc of crashing thread is stored in lr.
      static_assert(gpr_lr_arm64 - gpr_x19_arm64 == PCB_LR,
                    "nonconsecutive arm64 register numbers");
      value = pcb.x[reg - gpr_x19_arm64];
      break;
    case gpr_sp_arm64:
      value = pcb.sp;
      break;
    default:
      return false;
    }
  } else {
    size_t rd = m_thread.GetProcess()->ReadMemory(m_pcb_addr, &pcb13,
                                                  sizeof(pcb13), error);
    if (rd != sizeof(pcb13))
      return false;

    uint32_t reg = reg_info->kinds[lldb::eRegisterKindLLDB];
    switch (reg) {
    case gpr_x0_arm64:
    case gpr_x1_arm64:
    case gpr_x2_arm64:
    case gpr_x3_arm64:
    case gpr_x4_arm64:
    case gpr_x5_arm64:
    case gpr_x6_arm64:
    case gpr_x7_arm64:
    case gpr_x8_arm64:
    case gpr_x9_arm64:
    case gpr_x10_arm64:
    case gpr_x11_arm64:
    case gpr_x12_arm64:
    case gpr_x13_arm64:
    case gpr_x14_arm64:
    case gpr_x15_arm64:
    case gpr_x16_arm64:
    case gpr_x17_arm64:
    case gpr_x18_arm64:
    case gpr_x19_arm64:
    case gpr_x20_arm64:
    case gpr_x21_arm64:
    case gpr_x22_arm64:
    case gpr_x23_arm64:
    case gpr_x24_arm64:
    case gpr_x25_arm64:
    case gpr_x26_arm64:
    case gpr_x27_arm64:
    case gpr_x28_arm64:
    case gpr_fp_arm64:
      static_assert(gpr_fp_arm64 - gpr_x0_arm64 == 29,
                    "nonconsecutive arm64 register numbers");
      value = pcb13.x[reg - gpr_x0_arm64];
      break;
    case gpr_sp_arm64:
      value = pcb13.sp;
      break;
    case gpr_pc_arm64:
      // The pc of crashing thread is stored in lr.
      value = pcb13.lr;
      break;
    default:
      return false;
    }
  }
  return true;
}

bool RegisterContextFreeBSDKernel_arm64::WriteRegister(
    const RegisterInfo *reg_info, const RegisterValue &value) {
  return false;
}

std::optional<int> RegisterContextFreeBSDKernel_arm64::GetOsreldate() {
  ProcessSP process_sp = m_thread.GetProcess();
  if (!process_sp)
    return std::nullopt;

  Target &target = process_sp->GetTarget();

  SymbolContextList sc_list;
  target.GetImages().FindSymbolsWithNameAndType(ConstString("osreldate"),
                                                lldb::eSymbolTypeData, sc_list);
  if (sc_list.GetSize() == 0)
    return std::nullopt;

  SymbolContext sc;
  sc_list.GetContextAtIndex(0, sc);
  if (!sc.symbol)
    return std::nullopt;

  lldb::addr_t addr = sc.symbol->GetLoadAddress(&target);
  if (addr == LLDB_INVALID_ADDRESS)
    return std::nullopt;

  Status error;
  int osreldate = 0;
  size_t bytes_read =
      process_sp->ReadMemory(addr, &osreldate, sizeof(osreldate), error);
  if (bytes_read == sizeof(osreldate) && error.Success())
    return osreldate;

  return std::nullopt;
}

//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This file contains the definition of the
/// RegisterContextFreeBSDKernel_riscv64 class, which is used for reading
/// registers from PCB in riscv64 kernel dump.
///
//===----------------------------------------------------------------------===//

#include "RegisterContextFreeBSDKernel_riscv64.h"

#include "lldb/Target/Process.h"
#include "lldb/Target/Thread.h"
#include "lldb/Utility/RegisterValue.h"
#include "llvm/Support/Endian.h"

using namespace lldb;
using namespace lldb_private;

RegisterContextFreeBSDKernel_riscv64::RegisterContextFreeBSDKernel_riscv64(
    Thread &thread, std::unique_ptr<RegisterInfoPOSIX_riscv64> register_info_up,
    lldb::addr_t pcb_addr)
    : RegisterContextPOSIX_riscv64(thread, std::move(register_info_up)),
      m_pcb_addr(pcb_addr) {}

bool RegisterContextFreeBSDKernel_riscv64::ReadGPR() { return true; }

bool RegisterContextFreeBSDKernel_riscv64::ReadFPR() { return true; }

bool RegisterContextFreeBSDKernel_riscv64::WriteGPR() {
  assert(0);
  return false;
}

bool RegisterContextFreeBSDKernel_riscv64::WriteFPR() {
  assert(0);
  return false;
}

bool RegisterContextFreeBSDKernel_riscv64::ReadRegister(
    const RegisterInfo *reg_info, RegisterValue &value) {
  if (m_pcb_addr == LLDB_INVALID_ADDRESS)
    return false;

  // https://cgit.freebsd.org/src/tree/sys/riscv/include/pcb.h
  struct {
    llvm::support::ulittle64_t ra;
    llvm::support::ulittle64_t sp;
    llvm::support::ulittle64_t gp;
    llvm::support::ulittle64_t tp;
    llvm::support::ulittle64_t s[12];
  } pcb;

  Status error;
  size_t rd =
      m_thread.GetProcess()->ReadMemory(m_pcb_addr, &pcb, sizeof(pcb), error);
  if (rd != sizeof(pcb))
    return false;

  uint32_t reg = reg_info->kinds[lldb::eRegisterKindLLDB];
  switch (reg) {
  case gpr_pc_riscv:
  case gpr_ra_riscv:
    // Supply the RA as PC as well to simulate the PC as if the thread had just
    // returned.
    value = pcb.ra;
    break;
  case gpr_sp_riscv:
    value = pcb.sp;
    break;
  case gpr_gp_riscv:
    value = pcb.gp;
    break;
  case gpr_tp_riscv:
    value = pcb.tp;
    break;
  case gpr_fp_riscv:
    value = pcb.s[0];
    break;
  case gpr_s1_riscv:
    value = pcb.s[1];
    break;
  case gpr_s2_riscv:
  case gpr_s3_riscv:
  case gpr_s4_riscv:
  case gpr_s5_riscv:
  case gpr_s6_riscv:
  case gpr_s7_riscv:
  case gpr_s8_riscv:
  case gpr_s9_riscv:
  case gpr_s10_riscv:
  case gpr_s11_riscv:
    value = pcb.s[reg - gpr_s2_riscv];
    break;
  default:
    return false;
  }
  return true;
}

bool RegisterContextFreeBSDKernel_riscv64::WriteRegister(
    const RegisterInfo *reg_info, const RegisterValue &value) {
  return false;
}

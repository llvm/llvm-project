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
/// RegisterContextFreeBSDKernel_ppc64le class, which is used for reading
/// registers from PCB in ppc64le kernel dump.
///
//===----------------------------------------------------------------------===//

#include "RegisterContextFreeBSDKernel_ppc64le.h"

#include "lldb/Target/Process.h"
#include "lldb/Target/Thread.h"
#include "lldb/Utility/RegisterValue.h"
#include "llvm/Support/Endian.h"

using namespace lldb;
using namespace lldb_private;

RegisterContextFreeBSDKernel_ppc64le::RegisterContextFreeBSDKernel_ppc64le(
    Thread &thread, lldb_private::RegisterInfoInterface *register_info,
    lldb::addr_t pcb_addr)
    : RegisterContextPOSIX_ppc64le(thread, 0, register_info),
      m_pcb_addr(pcb_addr) {}

bool RegisterContextFreeBSDKernel_ppc64le::ReadRegister(
    const RegisterInfo *reg_info, RegisterValue &value) {
  if (m_pcb_addr == LLDB_INVALID_ADDRESS)
    return false;

  // https://cgit.freebsd.org/src/tree/sys/powerpc/include/pcb.h
  struct {
    llvm::support::ulittle64_t context[20];
    llvm::support::ulittle64_t cr;
    llvm::support::ulittle64_t sp;
    llvm::support::ulittle64_t toc;
    llvm::support::ulittle64_t lr;
  } pcb;

  Status error;
  size_t rd =
      m_thread.GetProcess()->ReadMemory(m_pcb_addr, &pcb, sizeof(pcb), error);
  if (rd != sizeof(pcb))
    return false;

  uint32_t reg = reg_info->kinds[lldb::eRegisterKindLLDB];
  switch (reg) {
  case gpr_r1_ppc64le:
    // r1 is saved in the sp field
    value = pcb.sp;
    break;
  case gpr_r2_ppc64le:
    // r2 is saved in the toc field
    value = pcb.toc;
    break;
  case gpr_r12_ppc64le:
  case gpr_r13_ppc64le:
  case gpr_r14_ppc64le:
  case gpr_r15_ppc64le:
  case gpr_r16_ppc64le:
  case gpr_r17_ppc64le:
  case gpr_r18_ppc64le:
  case gpr_r19_ppc64le:
  case gpr_r20_ppc64le:
  case gpr_r21_ppc64le:
  case gpr_r22_ppc64le:
  case gpr_r23_ppc64le:
  case gpr_r24_ppc64le:
  case gpr_r25_ppc64le:
  case gpr_r26_ppc64le:
  case gpr_r27_ppc64le:
  case gpr_r28_ppc64le:
  case gpr_r29_ppc64le:
  case gpr_r30_ppc64le:
  case gpr_r31_ppc64le:
    value = pcb.context[reg - gpr_r12_ppc64le];
    break;
  case gpr_pc_ppc64le:
  case gpr_lr_ppc64le:
    // The pc of crashing thread is stored in lr.
    value = pcb.lr;
    break;
  case gpr_cr_ppc64le:
    value = pcb.cr;
    break;
  default:
    return false;
  }
  return true;
}

bool RegisterContextFreeBSDKernel_ppc64le::WriteRegister(
    const RegisterInfo *reg_info, const RegisterValue &value) {
  return false;
}

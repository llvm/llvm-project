//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This file contains the definition of the RegisterContextFreeBSDKernel_arm
/// class, which is used for reading registers from PCB in arm kernel dump.
///
//===----------------------------------------------------------------------===//

#include "RegisterContextFreeBSDKernel_arm.h"
#include "Plugins/Process/Utility/lldb-arm-register-enums.h"

#include "lldb/Target/Process.h"
#include "lldb/Target/Thread.h"
#include "lldb/Utility/RegisterValue.h"
#include "llvm/Support/Endian.h"

using namespace lldb;
using namespace lldb_private;

RegisterContextFreeBSDKernel_arm::RegisterContextFreeBSDKernel_arm(
    Thread &thread, std::unique_ptr<RegisterInfoPOSIX_arm> register_info_up,
    lldb::addr_t pcb_addr)
    : RegisterContextPOSIX_arm(thread, std::move(register_info_up)),
      m_pcb_addr(pcb_addr) {}

bool RegisterContextFreeBSDKernel_arm::ReadGPR() { return true; }

bool RegisterContextFreeBSDKernel_arm::ReadFPR() { return true; }

bool RegisterContextFreeBSDKernel_arm::WriteGPR() {
  assert(0);
  return false;
}

bool RegisterContextFreeBSDKernel_arm::WriteFPR() {
  assert(0);
  return false;
}

bool RegisterContextFreeBSDKernel_arm::ReadRegister(
    const RegisterInfo *reg_info, RegisterValue &value) {
  if (m_pcb_addr == LLDB_INVALID_ADDRESS)
    return false;

  // https://cgit.freebsd.org/src/tree/sys/arm/include/frame.h
  // struct pcb's first field is struct switchframe which is the only field used
  // by debugger and should be aligned by 8 bytes.
  struct {
    // Aka switchframe.sf_r4 to switchframe.sf_pc.
    llvm::support::ulittle32_t r4;
    llvm::support::ulittle32_t r5;
    llvm::support::ulittle32_t r6;
    llvm::support::ulittle32_t r7;
    llvm::support::ulittle32_t r8;
    llvm::support::ulittle32_t r9;
    llvm::support::ulittle32_t r10;
    llvm::support::ulittle32_t r11;
    llvm::support::ulittle32_t r12;
    llvm::support::ulittle32_t sp;
    llvm::support::ulittle32_t lr;
    llvm::support::ulittle32_t pc;
  } pcb;

  Status error;
  size_t rd =
      m_thread.GetProcess()->ReadMemory(m_pcb_addr, &pcb, sizeof(pcb), error);
  if (rd != sizeof(pcb))
    return false;

  uint32_t reg = reg_info->kinds[lldb::eRegisterKindLLDB];
  switch (reg) {

#define REG(x)                                                                 \
  case gpr_##x##_arm:                                                          \
    value = pcb.x;                                                             \
    break;

    REG(r4);
    REG(r5);
    REG(r6);
    REG(r7);
    REG(r8);
    REG(r9);
    REG(r10);
    REG(r11);
    REG(r12);
    REG(sp);
    REG(lr);
    REG(pc);

#undef REG

  default:
    return false;
  }
  return true;
}

bool RegisterContextFreeBSDKernel_arm::WriteRegister(
    const RegisterInfo *reg_info, const RegisterValue &value) {
  return false;
}

//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "RegisterContextFreeBSDKernelCore_riscv64.h"
#include "lldb/Register/Arch/lldb-riscv-register-enums.h"

#include "lldb/Target/Process.h"
#include "lldb/Target/Thread.h"
#include "lldb/Register/RegisterValue.h"
#include "llvm/Support/Endian.h"

#if defined(__FreeBSD__) && defined(__riscv) && __riscv_xlen == 64
#include <machine/pcb.h>
#endif


namespace reg_ids = lldb_private::registers::riscv::id;
using namespace lldb;
using namespace lldb_private;

RegisterContextFreeBSDKernelCore_riscv64::
    RegisterContextFreeBSDKernelCore_riscv64(Thread &thread,
                                             RegisterLayout layout,
                                             lldb::addr_t pcb_addr)
    : ThreadRegisterContext(thread, 0, std::move(layout)),
      m_pcb_addr(pcb_addr) {}

llvm::Expected<RegisterValue>
RegisterContextFreeBSDKernelCore_riscv64::ReadRegister(RegisterID id) {
  auto description = GetDescription();
  const auto *reg_info = description ? description->Find(id) : nullptr;
  if (!reg_info)
    return llvm::createStringError("unknown register ID");
  RegisterValue value;
  if (m_pcb_addr == LLDB_INVALID_ADDRESS)
    return llvm::createStringError("register value is unavailable");

  // https://cgit.freebsd.org/src/tree/sys/riscv/include/pcb.h
  struct {
    llvm::support::ulittle64_t ra;
    llvm::support::ulittle64_t sp;
    llvm::support::ulittle64_t gp;
    llvm::support::ulittle64_t tp;
    llvm::support::ulittle64_t s[12];
  } pcb;

#if defined(__FreeBSD__) && defined(__riscv) && __riscv_xlen == 64
  static_assert(offsetof(struct pcb, pcb_ra) == offsetof(decltype(pcb), ra));
  static_assert(offsetof(struct pcb, pcb_sp) == offsetof(decltype(pcb), sp));
  static_assert(offsetof(struct pcb, pcb_gp) == offsetof(decltype(pcb), gp));
  static_assert(offsetof(struct pcb, pcb_tp) == offsetof(decltype(pcb), tp));
  static_assert(offsetof(struct pcb, pcb_s) == offsetof(decltype(pcb), s));
#endif

  Status error;
  size_t rd =
      m_thread.GetProcess()->ReadMemory(m_pcb_addr, &pcb, sizeof(pcb), error);
  if (rd != sizeof(pcb))
    return llvm::createStringError("register value is unavailable");

  uint32_t reg = reg_info->numbers[lldb::eRegisterKindLLDB];
  switch (reg) {
  case reg_ids::gpr_pc:
  case reg_ids::gpr_ra:
    // Supply the RA as PC as well to simulate the PC as if the thread had just
    // returned.
    value = pcb.ra;
    break;
  case reg_ids::gpr_sp:
    value = pcb.sp;
    break;
  case reg_ids::gpr_gp:
    value = pcb.gp;
    break;
  case reg_ids::gpr_tp:
    value = pcb.tp;
    break;
  case reg_ids::gpr_fp:
    value = pcb.s[0];
    break;
  case reg_ids::gpr_s1:
    value = pcb.s[1];
    break;
  case reg_ids::gpr_s2:
  case reg_ids::gpr_s3:
  case reg_ids::gpr_s4:
  case reg_ids::gpr_s5:
  case reg_ids::gpr_s6:
  case reg_ids::gpr_s7:
  case reg_ids::gpr_s8:
  case reg_ids::gpr_s9:
  case reg_ids::gpr_s10:
  case reg_ids::gpr_s11:
    value = pcb.s[reg - reg_ids::gpr_s2 + 2];
    break;
  default:
    return llvm::createStringError("register value is unavailable");
  }
  return value;
}

llvm::Error RegisterContextFreeBSDKernelCore_riscv64::WriteRegister(
    RegisterID id, const RegisterValue &value) {
  return llvm::createStringError("register write is unavailable");
}

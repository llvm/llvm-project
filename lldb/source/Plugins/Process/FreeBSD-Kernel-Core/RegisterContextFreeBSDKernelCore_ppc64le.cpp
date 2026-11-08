//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "RegisterContextFreeBSDKernelCore_ppc64le.h"
#include "lldb/Register/Arch/lldb-ppc64le-register-enums.h"

#include "lldb/Target/Process.h"
#include "lldb/Target/Thread.h"
#include "lldb/Register/RegisterValue.h"
#include "llvm/Support/Endian.h"

#if defined(__FreeBSD__) && defined(__powerpc64__) && defined(__LITTLE_ENDIAN__)
#include <machine/pcb.h>
#endif


namespace reg_ids = lldb_private::registers::ppc64le::id;
using namespace lldb;
using namespace lldb_private;

RegisterContextFreeBSDKernelCore_ppc64le::
    RegisterContextFreeBSDKernelCore_ppc64le(Thread &thread,
                                             RegisterLayout layout,
                                             lldb::addr_t pcb_addr)
    : ThreadRegisterContext(thread, 0, std::move(layout)),
      m_pcb_addr(pcb_addr) {}

llvm::Expected<RegisterValue>
RegisterContextFreeBSDKernelCore_ppc64le::ReadRegister(RegisterID id) {
  auto description = GetDescription();
  const auto *reg_info = description ? description->Find(id) : nullptr;
  if (!reg_info)
    return llvm::createStringError("unknown register ID");
  RegisterValue value;
  if (m_pcb_addr == LLDB_INVALID_ADDRESS)
    return llvm::createStringError("register value is unavailable");

  // https://cgit.freebsd.org/src/tree/sys/powerpc/include/pcb.h
  struct {
    llvm::support::ulittle64_t context[20];
    llvm::support::ulittle64_t cr;
    llvm::support::ulittle64_t sp;
    llvm::support::ulittle64_t toc;
    llvm::support::ulittle64_t lr;
  } pcb;

#if defined(__FreeBSD__) && defined(__powerpc64__) && defined(__LITTLE_ENDIAN__)
  static_assert(offsetof(struct pcb, pcb_context) ==
                offsetof(decltype(pcb), context));
  static_assert(offsetof(struct pcb, pcb_cr) == offsetof(decltype(pcb), cr));
  static_assert(offsetof(struct pcb, pcb_sp) == offsetof(decltype(pcb), sp));
  static_assert(offsetof(struct pcb, pcb_toc) == offsetof(decltype(pcb), toc));
  static_assert(offsetof(struct pcb, pcb_lr) == offsetof(decltype(pcb), lr));
#endif

  Status error;
  size_t rd =
      m_thread.GetProcess()->ReadMemory(m_pcb_addr, &pcb, sizeof(pcb), error);
  if (rd != sizeof(pcb))
    return llvm::createStringError("register value is unavailable");

  uint32_t reg = reg_info->numbers[lldb::eRegisterKindLLDB];
  switch (reg) {
  case reg_ids::gpr_r1:
    // r1 is saved in the sp field
    value = pcb.sp;
    break;
  case reg_ids::gpr_r2:
    // r2 is saved in the toc field
    value = pcb.toc;
    break;
  case reg_ids::gpr_r12:
  case reg_ids::gpr_r13:
  case reg_ids::gpr_r14:
  case reg_ids::gpr_r15:
  case reg_ids::gpr_r16:
  case reg_ids::gpr_r17:
  case reg_ids::gpr_r18:
  case reg_ids::gpr_r19:
  case reg_ids::gpr_r20:
  case reg_ids::gpr_r21:
  case reg_ids::gpr_r22:
  case reg_ids::gpr_r23:
  case reg_ids::gpr_r24:
  case reg_ids::gpr_r25:
  case reg_ids::gpr_r26:
  case reg_ids::gpr_r27:
  case reg_ids::gpr_r28:
  case reg_ids::gpr_r29:
  case reg_ids::gpr_r30:
  case reg_ids::gpr_r31:
    value = pcb.context[reg - reg_ids::gpr_r12];
    break;
  case reg_ids::gpr_pc:
  case reg_ids::gpr_lr:
    // The pc of crashing thread is stored in lr.
    value = pcb.lr;
    break;
  case reg_ids::gpr_cr:
    value = pcb.cr;
    break;
  default:
    return llvm::createStringError("register value is unavailable");
  }
  return value;
}

llvm::Error RegisterContextFreeBSDKernelCore_ppc64le::WriteRegister(
    RegisterID id, const RegisterValue &value) {
  return llvm::createStringError("register write is unavailable");
}

//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "RegisterContextFreeBSDKernelCore_arm.h"
#include "lldb/Register/Arch/lldb-arm-register-enums.h"

#include "lldb/Target/Process.h"
#include "lldb/Target/Thread.h"
#include "lldb/Register/RegisterValue.h"
#include "llvm/Support/Endian.h"

#if defined(__FreeBSD__) && defined(__arm__)
#include <machine/frame.h>
#endif


namespace reg_ids = lldb_private::registers::arm::id;
using namespace lldb;
using namespace lldb_private;

RegisterContextFreeBSDKernelCore_arm::RegisterContextFreeBSDKernelCore_arm(
    Thread &thread, RegisterLayout layout, lldb::addr_t pcb_addr)
    : ThreadRegisterContext(thread, 0, std::move(layout)),
      m_pcb_addr(pcb_addr) {}

llvm::Expected<RegisterValue>
RegisterContextFreeBSDKernelCore_arm::ReadRegister(RegisterID id) {
  auto description = GetDescription();
  const auto *reg_info = description ? description->Find(id) : nullptr;
  if (!reg_info)
    return llvm::createStringError("unknown register ID");
  RegisterValue value;
  if (m_pcb_addr == LLDB_INVALID_ADDRESS)
    return llvm::createStringError("register value is unavailable");

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

#if defined(__FreeBSD__) && defined(__arm__)
  static_assert(offsetof(struct switchframe, sf_r4) ==
                offsetof(decltype(pcb), r4));
  static_assert(offsetof(struct switchframe, sf_r5) ==
                offsetof(decltype(pcb), r5));
  static_assert(offsetof(struct switchframe, sf_r6) ==
                offsetof(decltype(pcb), r6));
  static_assert(offsetof(struct switchframe, sf_r7) ==
                offsetof(decltype(pcb), r7));
  static_assert(offsetof(struct switchframe, sf_r8) ==
                offsetof(decltype(pcb), r8));
  static_assert(offsetof(struct switchframe, sf_r9) ==
                offsetof(decltype(pcb), r9));
  static_assert(offsetof(struct switchframe, sf_r10) ==
                offsetof(decltype(pcb), r10));
  static_assert(offsetof(struct switchframe, sf_r11) ==
                offsetof(decltype(pcb), r11));
  static_assert(offsetof(struct switchframe, sf_r12) ==
                offsetof(decltype(pcb), r12));
  static_assert(offsetof(struct switchframe, sf_sp) ==
                offsetof(decltype(pcb), sp));
  static_assert(offsetof(struct switchframe, sf_lr) ==
                offsetof(decltype(pcb), lr));
  static_assert(offsetof(struct switchframe, sf_pc) ==
                offsetof(decltype(pcb), pc));
#endif

  Status error;
  size_t rd =
      m_thread.GetProcess()->ReadMemory(m_pcb_addr, &pcb, sizeof(pcb), error);
  if (rd != sizeof(pcb))
    return llvm::createStringError("register value is unavailable");

  uint32_t reg = reg_info->numbers[lldb::eRegisterKindLLDB];
  switch (reg) {

#define REG(x)                                                                 \
  case reg_ids::gpr_##x:                                                       \
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
    return llvm::createStringError("register value is unavailable");
  }
  return value;
}

llvm::Error RegisterContextFreeBSDKernelCore_arm::WriteRegister(
    RegisterID id, const RegisterValue &value) {
  return llvm::createStringError("register write is unavailable");
}

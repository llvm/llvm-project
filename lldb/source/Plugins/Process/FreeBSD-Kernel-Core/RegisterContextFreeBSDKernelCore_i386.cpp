//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <set>
#include "lldb/Register/Arch/lldb-x86-register-enums.h"

#include "RegisterContextFreeBSDKernelCore_i386.h"

#include "lldb/Target/Process.h"
#include "lldb/Target/Thread.h"
#include "lldb/Register/RegisterValue.h"
#include "llvm/Support/Endian.h"

#if defined(__FreeBSD__) && defined(__i386__)
#include <machine/pcb.h>
#endif


namespace reg_ids = lldb_private::registers::x86::id;
using namespace lldb;
using namespace lldb_private;

RegisterContextFreeBSDKernelCore_i386::RegisterContextFreeBSDKernelCore_i386(
    Thread &thread, RegisterLayout layout, lldb::addr_t pcb_addr)
    : ThreadRegisterContext(thread, 0, std::move(layout)),
      m_pcb_addr(pcb_addr) {

  const size_t group_count = [&]() -> size_t { return 2; }();
  auto previous = GetDescription();
  auto selected = previous->GetGroups().take_front(group_count);
  std::vector<RegisterGroup> groups(selected.begin(), selected.end());
  std::set<RegisterID> visible;
  for (const auto &group : groups)
    visible.insert(group.registers.begin(), group.registers.end());
  std::vector<RegisterID> order;
  for (RegisterID id : previous->GetUserRegisters())
    if (visible.count(id))
      order.push_back(id);
  auto description = llvm::cantFail(RegisterDescription::Create(
      {previous->GetRegisters().begin(), previous->GetRegisters().end()},
      std::move(groups),
      {previous->GetOwnedTypes().begin(), previous->GetOwnedTypes().end()}));
  auto presentation = llvm::cantFail(description.WithUserRegisters(order));
  SetRegisterLayout(llvm::cantFail(RegisterLayout::Create(
      std::make_shared<RegisterDescription>(std::move(presentation)),
      GetLayout())));
}

llvm::Expected<RegisterValue>
RegisterContextFreeBSDKernelCore_i386::ReadRegister(RegisterID id) {
  auto description = GetDescription();
  const auto *reg_info = description ? description->Find(id) : nullptr;
  if (!reg_info)
    return llvm::createStringError("unknown register ID");
  RegisterValue value;
  if (m_pcb_addr == LLDB_INVALID_ADDRESS)
    return llvm::createStringError("register value is unavailable");

  // https://cgit.freebsd.org/src/tree/sys/i386/include/pcb.h
  struct {
    llvm::support::ulittle32_t edi;
    llvm::support::ulittle32_t esi;
    llvm::support::ulittle32_t ebp;
    llvm::support::ulittle32_t esp;
    llvm::support::ulittle32_t ebx;
    llvm::support::ulittle32_t eip;
  } pcb;

#if defined(__FreeBSD__) && defined(__i386__)
  static_assert(offsetof(struct pcb, pcb_edi) == offsetof(decltype(pcb), edi));
  static_assert(offsetof(struct pcb, pcb_esi) == offsetof(decltype(pcb), esi));
  static_assert(offsetof(struct pcb, pcb_ebp) == offsetof(decltype(pcb), ebp));
  static_assert(offsetof(struct pcb, pcb_esp) == offsetof(decltype(pcb), esp));
  static_assert(offsetof(struct pcb, pcb_ebx) == offsetof(decltype(pcb), ebx));
  static_assert(offsetof(struct pcb, pcb_eip) == offsetof(decltype(pcb), eip));
#endif

  Status error;
  size_t rd =
      m_thread.GetProcess()->ReadMemory(m_pcb_addr, &pcb, sizeof(pcb), error);
  if (rd != sizeof(pcb))
    return llvm::createStringError("register value is unavailable");

  uint32_t reg = reg_info->numbers[lldb::eRegisterKindLLDB];
  switch (reg) {
#define REG(x)                                                                 \
  case reg_ids::x:                                                             \
    value = pcb.x;                                                             \
    break;

    REG(edi);
    REG(esi);
    REG(ebp);
    REG(esp);
    REG(eip);

#undef REG

  default:
    return llvm::createStringError("register value is unavailable");
  }

  return value;
}

llvm::Error RegisterContextFreeBSDKernelCore_i386::WriteRegister(
    RegisterID id, const RegisterValue &value) {
  return llvm::createStringError("register write is unavailable");
}

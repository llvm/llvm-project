//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <set>
#include "lldb/Register/Arch/lldb-x86_64-register-enums.h"

#include "RegisterContextFreeBSDKernelCore_x86_64.h"

#include "lldb/Target/Process.h"
#include "lldb/Target/Thread.h"
#include "lldb/Register/RegisterValue.h"
#include "llvm/Support/Endian.h"

#if defined(__FreeBSD__) && defined(__amd64__)
#include <machine/pcb.h>
#endif


namespace reg_ids = lldb_private::registers::x86_64::id;
using namespace lldb;
using namespace lldb_private;

RegisterContextFreeBSDKernelCore_x86_64::
    RegisterContextFreeBSDKernelCore_x86_64(Thread &thread,
                                            RegisterLayout layout,
                                            lldb::addr_t pcb_addr)
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
RegisterContextFreeBSDKernelCore_x86_64::ReadRegister(RegisterID id) {
  auto description = GetDescription();
  const auto *reg_info = description ? description->Find(id) : nullptr;
  if (!reg_info)
    return llvm::createStringError("unknown register ID");
  RegisterValue value;
  if (m_pcb_addr == LLDB_INVALID_ADDRESS)
    return llvm::createStringError("register value is unavailable");

  // https://cgit.freebsd.org/src/tree/sys/amd64/include/pcb.h
  struct {
    llvm::support::ulittle64_t r15;
    llvm::support::ulittle64_t r14;
    llvm::support::ulittle64_t r13;
    llvm::support::ulittle64_t r12;
    llvm::support::ulittle64_t rbp;
    llvm::support::ulittle64_t rsp;
    llvm::support::ulittle64_t rbx;
    llvm::support::ulittle64_t rip;
  } pcb;

  constexpr unsigned short CODE_SEL = 4 << 3;
  constexpr unsigned short DATA_SEL = 5 << 3;

#if defined(__FreeBSD__) && defined(__amd64__)
  static_assert(offsetof(struct pcb, pcb_r15) == offsetof(decltype(pcb), r15));
  static_assert(offsetof(struct pcb, pcb_r14) == offsetof(decltype(pcb), r14));
  static_assert(offsetof(struct pcb, pcb_r13) == offsetof(decltype(pcb), r13));
  static_assert(offsetof(struct pcb, pcb_r12) == offsetof(decltype(pcb), r12));
  static_assert(offsetof(struct pcb, pcb_rbp) == offsetof(decltype(pcb), rbp));
  static_assert(offsetof(struct pcb, pcb_rsp) == offsetof(decltype(pcb), rsp));
  static_assert(offsetof(struct pcb, pcb_rbx) == offsetof(decltype(pcb), rbx));
  static_assert(offsetof(struct pcb, pcb_rip) == offsetof(decltype(pcb), rip));
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

    REG(r15);
    REG(r14);
    REG(r13);
    REG(r12);
    REG(rbp);
    REG(rsp);
    REG(rbx);
    REG(rip);
#undef REG
  case reg_ids::cs:
    value.SetUInt(CODE_SEL, reg_info->byte_size);
    break;
  case reg_ids::ss:
    value.SetUInt(DATA_SEL, reg_info->byte_size);
    break;
  default:
    return llvm::createStringError("register value is unavailable");
  }

  return value;
}

llvm::Error RegisterContextFreeBSDKernelCore_x86_64::WriteRegister(
    RegisterID id, const RegisterValue &value) {
  return llvm::createStringError("register write is unavailable");
}

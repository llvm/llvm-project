//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "RegisterContextFreeBSDKernelCore_arm64.h"
#include "lldb/Register/Arch/lldb-arm64-register-enums.h"

#include "lldb/Symbol/Symbol.h"
#include "lldb/Target/Process.h"
#include "lldb/Target/Target.h"
#include "lldb/Target/Thread.h"
#include "lldb/Utility/LLDBLog.h"
#include "lldb/Utility/Log.h"
#include "lldb/Register/RegisterValue.h"
#include "llvm/Support/Endian.h"

#if defined(__FreeBSD__) && defined(__aarch64__)
#include <machine/pcb.h>
#include <sys/param.h>
#endif


namespace reg_ids = lldb_private::registers::arm64::id;
using namespace lldb;
using namespace lldb_private;

RegisterContextFreeBSDKernelCore_arm64::RegisterContextFreeBSDKernelCore_arm64(
    Thread &thread, RegisterLayout layout, lldb::addr_t pcb_addr)
    : ThreadRegisterContext(thread, 0, std::move(layout)),
      m_pcb_addr(pcb_addr) {}

llvm::Expected<RegisterValue>
RegisterContextFreeBSDKernelCore_arm64::ReadRegister(RegisterID id) {
  auto description = GetDescription();
  const auto *reg_info = description ? description->Find(id) : nullptr;
  if (!reg_info)
    return llvm::createStringError("unknown register ID");
  RegisterValue value;
  if (m_pcb_addr == LLDB_INVALID_ADDRESS)
    return llvm::createStringError("register value is unavailable");

  // https://cgit.freebsd.org/src/tree/sys/arm64/include/pcb.h
  struct {
    llvm::support::ulittle64_t x[12];
    llvm::support::ulittle64_t sp;
  } pcb;

#if defined(__FreeBSD__) && defined(__aarch64__) && __FreeBSD_version >= 1400084
  static_assert(offsetof(struct pcb, pcb_x) == offsetof(decltype(pcb), x));
  static_assert(offsetof(struct pcb, pcb_sp) == offsetof(decltype(pcb), sp));
#endif

  // https://cgit.freebsd.org/src/tree/sys/arm64/include/pcb.h?h=stable%2F13
  struct {
    llvm::support::ulittle64_t x[30];
    llvm::support::ulittle64_t lr;
    llvm::support::ulittle64_t _reserved;
    llvm::support::ulittle64_t sp;
  } pcb13;

#if defined(__FreeBSD__) && defined(__aarch64__) && __FreeBSD_version < 1400084
  static_assert(offsetof(struct pcb, pcb_x) == offsetof(decltype(pcb13), x));
  static_assert(offsetof(struct pcb, pcb_lr) == offsetof(decltype(pcb13), lr));
  static_assert(offsetof(struct pcb, pcb_sp) == offsetof(decltype(pcb13), sp));
#endif

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
    constexpr uint32_t pcb_fp = 10;
    constexpr uint32_t pcb_lr = 11;
    size_t rd =
        m_thread.GetProcess()->ReadMemory(m_pcb_addr, &pcb, sizeof(pcb), error);
    if (rd != sizeof(pcb))
      return llvm::createStringError("register value is unavailable");

    uint32_t reg = reg_info->numbers[lldb::eRegisterKindLLDB];
    switch (reg) {
    case reg_ids::gpr_x19:
    case reg_ids::gpr_x20:
    case reg_ids::gpr_x21:
    case reg_ids::gpr_x22:
    case reg_ids::gpr_x23:
    case reg_ids::gpr_x24:
    case reg_ids::gpr_x25:
    case reg_ids::gpr_x26:
    case reg_ids::gpr_x27:
    case reg_ids::gpr_x28:
    case reg_ids::gpr_fp:
      static_assert(reg_ids::gpr_fp - reg_ids::gpr_x19 == pcb_fp,
                    "nonconsecutive arm64 register numbers");
      value = pcb.x[reg - reg_ids::gpr_x19];
      break;
    case reg_ids::gpr_sp:
      value = pcb.sp;
      break;
    case reg_ids::gpr_pc:
      // The pc of crashing thread is stored in lr.
      static_assert(reg_ids::gpr_lr - reg_ids::gpr_x19 == pcb_lr,
                    "nonconsecutive arm64 register numbers");
      value = pcb.x[reg_ids::gpr_lr - reg_ids::gpr_x19];
      break;
    default:
      return llvm::createStringError("register value is unavailable");
    }
  } else {
    size_t rd = m_thread.GetProcess()->ReadMemory(m_pcb_addr, &pcb13,
                                                  sizeof(pcb13), error);
    if (rd != sizeof(pcb13))
      return llvm::createStringError("register value is unavailable");

    uint32_t reg = reg_info->numbers[lldb::eRegisterKindLLDB];
    switch (reg) {
    case reg_ids::gpr_x0:
    case reg_ids::gpr_x1:
    case reg_ids::gpr_x2:
    case reg_ids::gpr_x3:
    case reg_ids::gpr_x4:
    case reg_ids::gpr_x5:
    case reg_ids::gpr_x6:
    case reg_ids::gpr_x7:
    case reg_ids::gpr_x8:
    case reg_ids::gpr_x9:
    case reg_ids::gpr_x10:
    case reg_ids::gpr_x11:
    case reg_ids::gpr_x12:
    case reg_ids::gpr_x13:
    case reg_ids::gpr_x14:
    case reg_ids::gpr_x15:
    case reg_ids::gpr_x16:
    case reg_ids::gpr_x17:
    case reg_ids::gpr_x18:
    case reg_ids::gpr_x19:
    case reg_ids::gpr_x20:
    case reg_ids::gpr_x21:
    case reg_ids::gpr_x22:
    case reg_ids::gpr_x23:
    case reg_ids::gpr_x24:
    case reg_ids::gpr_x25:
    case reg_ids::gpr_x26:
    case reg_ids::gpr_x27:
    case reg_ids::gpr_x28:
    case reg_ids::gpr_fp:
      static_assert(reg_ids::gpr_fp - reg_ids::gpr_x0 == 29,
                    "nonconsecutive arm64 register numbers");
      value = pcb13.x[reg - reg_ids::gpr_x0];
      break;
    case reg_ids::gpr_sp:
      value = pcb13.sp;
      break;
    case reg_ids::gpr_pc:
      // The pc of crashing thread is stored in lr.
      value = pcb13.lr;
      break;
    default:
      return llvm::createStringError("register value is unavailable");
    }
  }
  return value;
}

llvm::Error RegisterContextFreeBSDKernelCore_arm64::WriteRegister(
    RegisterID id, const RegisterValue &value) {
  return llvm::createStringError("register write is unavailable");
}

std::optional<int> RegisterContextFreeBSDKernelCore_arm64::GetOsreldate() {
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

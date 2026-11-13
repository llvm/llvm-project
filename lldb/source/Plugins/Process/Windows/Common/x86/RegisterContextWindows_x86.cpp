//===-- RegisterContextWindows_x86.cpp ------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#if defined(__i386__) || defined(_M_IX86)

#include "lldb/Host/windows/HostThreadWindows.h"
#include "lldb/Host/windows/windows.h"
#include "lldb/Register/RegisterValue.h"
#include "lldb/Utility/Status.h"
#include "lldb/lldb-private-types.h"

#include "ProcessWindowsLog.h"
#include "RegisterContextWindows_x86.h"
#include "RegisterLayoutWindows_x86.h"
#include "TargetThreadWindows.h"
#include "lldb/Register/Arch/lldb-x86-register-enums.h"
#include "lldb/Register/RegisterLayout.h"


namespace reg_ids = lldb_private::registers::x86::id;
using namespace lldb;
using namespace lldb_private;

namespace {
const RegisterLayout &RegisterMetadata() {
  static const ArchSpec arch("i686-pc-windows");
  static const auto info = CreateWindowsX86RegisterLayout(
      arch, X86RegisterMode::I386, WindowsX86ContextKind::Frame);
  return info;
}
} // namespace

// Constructors and Destructors
RegisterContextWindows_x86::RegisterContextWindows_x86(
    Thread &thread, uint32_t concrete_frame_idx)
    : RegisterContextWindows(thread, concrete_frame_idx) {
  SetRegisterLayout(RegisterMetadata());
}

RegisterContextWindows_x86::~RegisterContextWindows_x86() {}

llvm::Expected<RegisterValue>
RegisterContextWindows_x86::ReadRegister(RegisterID id) {
  auto description = GetDescription();
  const auto *reg_info = description ? description->Find(id) : nullptr;
  if (!reg_info)
    return llvm::createStringError("unknown register ID");
  RegisterValue reg_value;
  if (!CacheAllRegisterValues())
    return llvm::createStringError("register value is unavailable");

  if (reg_info == nullptr)
    return llvm::createStringError("register value is unavailable");

  uint32_t reg = reg_info->numbers[eRegisterKindLLDB];
  switch (reg) {
  case reg_ids::eax: {
    if (!(ReadRegisterHelper(CONTEXT_INTEGER, "EAX", m_context.Eax, reg_value)))
      return llvm::createStringError("register read failed");
    return reg_value;
  }
  case reg_ids::ebx: {
    if (!(ReadRegisterHelper(CONTEXT_INTEGER, "EBX", m_context.Ebx, reg_value)))
      return llvm::createStringError("register read failed");
    return reg_value;
  }
  case reg_ids::ecx: {
    if (!(ReadRegisterHelper(CONTEXT_INTEGER, "ECX", m_context.Ecx, reg_value)))
      return llvm::createStringError("register read failed");
    return reg_value;
  }
  case reg_ids::edx: {
    if (!(ReadRegisterHelper(CONTEXT_INTEGER, "EDX", m_context.Edx, reg_value)))
      return llvm::createStringError("register read failed");
    return reg_value;
  }
  case reg_ids::edi: {
    if (!(ReadRegisterHelper(CONTEXT_INTEGER, "EDI", m_context.Edi, reg_value)))
      return llvm::createStringError("register read failed");
    return reg_value;
  }
  case reg_ids::esi: {
    if (!(ReadRegisterHelper(CONTEXT_INTEGER, "ESI", m_context.Esi, reg_value)))
      return llvm::createStringError("register read failed");
    return reg_value;
  }
  case reg_ids::ebp: {
    if (!(ReadRegisterHelper(CONTEXT_CONTROL, "EBP", m_context.Ebp, reg_value)))
      return llvm::createStringError("register read failed");
    return reg_value;
  }
  case reg_ids::esp: {
    if (!(ReadRegisterHelper(CONTEXT_CONTROL, "ESP", m_context.Esp, reg_value)))
      return llvm::createStringError("register read failed");
    return reg_value;
  }
  case reg_ids::eip: {
    if (!(ReadRegisterHelper(CONTEXT_CONTROL, "EIP", m_context.Eip, reg_value)))
      return llvm::createStringError("register read failed");
    return reg_value;
  }
  case reg_ids::eflags: {
    if (!(ReadRegisterHelper(CONTEXT_CONTROL, "EFLAGS", m_context.EFlags,
                             reg_value)))
      return llvm::createStringError("register read failed");
    return reg_value;
  }
  default:
    Log *log = GetLog(WindowsLog::Registers);
    LLDB_LOG(log, "Requested unknown register {0}", reg);
    break;
  }
  return llvm::createStringError("register value is unavailable");
}

llvm::Error
RegisterContextWindows_x86::WriteRegister(RegisterID id,
                                               const RegisterValue &reg_value) {
  auto description = GetDescription();
  const auto *reg_info = description ? description->Find(id) : nullptr;
  if (!reg_info)
    return llvm::createStringError("unknown register ID");
  // Since we cannot only write a single register value to the inferior, we
  // need to make sure our cached copy of the register values are fresh.
  // Otherwise when writing EAX, for example, we may also overwrite some other
  // register with a stale value.
  if (!CacheAllRegisterValues())
    return llvm::createStringError("register write is unavailable");

  Log *log = GetLog(WindowsLog::Registers);
  uint32_t reg = reg_info->numbers[eRegisterKindLLDB];
  switch (reg) {
  case reg_ids::eax:
    LLDB_LOG(log, "Write value {0:x} to EAX", reg_value.GetAsUInt32());
    m_context.Eax = reg_value.GetAsUInt32();
    break;
  case reg_ids::ebx:
    LLDB_LOG(log, "Write value {0:x} to EBX", reg_value.GetAsUInt32());
    m_context.Ebx = reg_value.GetAsUInt32();
    break;
  case reg_ids::ecx:
    LLDB_LOG(log, "Write value {0:x} to ECX", reg_value.GetAsUInt32());
    m_context.Ecx = reg_value.GetAsUInt32();
    break;
  case reg_ids::edx:
    LLDB_LOG(log, "Write value {0:x} to EDX", reg_value.GetAsUInt32());
    m_context.Edx = reg_value.GetAsUInt32();
    break;
  case reg_ids::edi:
    LLDB_LOG(log, "Write value {0:x} to EDI", reg_value.GetAsUInt32());
    m_context.Edi = reg_value.GetAsUInt32();
    break;
  case reg_ids::esi:
    LLDB_LOG(log, "Write value {0:x} to ESI", reg_value.GetAsUInt32());
    m_context.Esi = reg_value.GetAsUInt32();
    break;
  case reg_ids::ebp:
    LLDB_LOG(log, "Write value {0:x} to EBP", reg_value.GetAsUInt32());
    m_context.Ebp = reg_value.GetAsUInt32();
    break;
  case reg_ids::esp:
    LLDB_LOG(log, "Write value {0:x} to ESP", reg_value.GetAsUInt32());
    m_context.Esp = reg_value.GetAsUInt32();
    break;
  case reg_ids::eip:
    LLDB_LOG(log, "Write value {0:x} to EIP", reg_value.GetAsUInt32());
    m_context.Eip = reg_value.GetAsUInt32();
    break;
  case reg_ids::eflags:
    LLDB_LOG(log, "Write value {0:x} to EFLAGS", reg_value.GetAsUInt32());
    m_context.EFlags = reg_value.GetAsUInt32();
    break;
  default:
    LLDB_LOG(log, "Write value {0:x} to unknown register {1}",
             reg_value.GetAsUInt32(), reg);
    return llvm::createStringError("register write is unavailable");
  }

  // Physically update the registers in the target process.
  {
    if (!(ApplyAllRegisterValues()))
      return llvm::createStringError("register write failed");
    return llvm::Error::success();
  }
}

bool RegisterContextWindows_x86::ReadRegisterHelper(
    DWORD flags_required, const char *reg_name, DWORD value,
    RegisterValue &reg_value) const {
  Log *log = GetLog(WindowsLog::Registers);
  if ((m_context.ContextFlags & flags_required) != flags_required) {
    LLDB_LOG(log, "Thread context doesn't have {0}", reg_name);
    return false;
  }
  LLDB_LOG(log, "Read value {0:x} from {1}", value, reg_name);
  reg_value.SetUInt32(value);
  return true;
}

#endif // defined(__i386__) || defined(_M_IX86)

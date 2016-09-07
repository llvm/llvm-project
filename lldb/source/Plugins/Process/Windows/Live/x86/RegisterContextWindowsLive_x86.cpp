//===-- RegisterContextWindowsLive_x86.cpp ------------------------*- C++
//-*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "lldb/Core/Error.h"
#include "lldb/Core/RegisterValue.h"
#include "lldb/Host/windows/HostThreadWindows.h"
#include "lldb/Host/windows/windows.h"
#include "lldb/lldb-private-types.h"

#include "ProcessWindowsLog.h"
#include "RegisterContextWindowsLive_x86.h"
#include "TargetThreadWindows.h"
#include "lldb-x86-register-enums.h"

#include "llvm/ADT/STLExtras.h"

using namespace lldb;

namespace lldb_private {

RegisterContextWindowsLive_x86::RegisterContextWindowsLive_x86(
    Thread &thread, uint32_t concrete_frame_idx)
    : RegisterContextWindows_x86(thread, concrete_frame_idx) {}

RegisterContextWindowsLive_x86::~RegisterContextWindowsLive_x86() {}

bool RegisterContextWindowsLive_x86::WriteRegister(
    const RegisterInfo *reg_info, const RegisterValue &reg_value) {
  // Since we cannot only write a single register value to the inferior, we need
  // to make sure
  // our cached copy of the register values are fresh.  Otherwise when writing
  // EAX, for example,
  // we may also overwrite some other register with a stale value.
  if (!CacheAllRegisterValues())
    return false;

  uint32_t reg = reg_info->kinds[eRegisterKindLLDB];
  switch (reg) {
  case lldb_eax_i386:
    WINLOG_IFALL(WINDOWS_LOG_REGISTERS, "Write value 0x%x to EAX",
                 reg_value.GetAsUInt32());
    m_context.Eax = reg_value.GetAsUInt32();
    break;
  case lldb_ebx_i386:
    WINLOG_IFALL(WINDOWS_LOG_REGISTERS, "Write value 0x%x to EBX",
                 reg_value.GetAsUInt32());
    m_context.Ebx = reg_value.GetAsUInt32();
    break;
  case lldb_ecx_i386:
    WINLOG_IFALL(WINDOWS_LOG_REGISTERS, "Write value 0x%x to ECX",
                 reg_value.GetAsUInt32());
    m_context.Ecx = reg_value.GetAsUInt32();
    break;
  case lldb_edx_i386:
    WINLOG_IFALL(WINDOWS_LOG_REGISTERS, "Write value 0x%x to EDX",
                 reg_value.GetAsUInt32());
    m_context.Edx = reg_value.GetAsUInt32();
    break;
  case lldb_edi_i386:
    WINLOG_IFALL(WINDOWS_LOG_REGISTERS, "Write value 0x%x to EDI",
                 reg_value.GetAsUInt32());
    m_context.Edi = reg_value.GetAsUInt32();
    break;
  case lldb_esi_i386:
    WINLOG_IFALL(WINDOWS_LOG_REGISTERS, "Write value 0x%x to ESI",
                 reg_value.GetAsUInt32());
    m_context.Esi = reg_value.GetAsUInt32();
    break;
  case lldb_ebp_i386:
    WINLOG_IFALL(WINDOWS_LOG_REGISTERS, "Write value 0x%x to EBP",
                 reg_value.GetAsUInt32());
    m_context.Ebp = reg_value.GetAsUInt32();
    break;
  case lldb_esp_i386:
    WINLOG_IFALL(WINDOWS_LOG_REGISTERS, "Write value 0x%x to ESP",
                 reg_value.GetAsUInt32());
    m_context.Esp = reg_value.GetAsUInt32();
    break;
  case lldb_eip_i386:
    WINLOG_IFALL(WINDOWS_LOG_REGISTERS, "Write value 0x%x to EIP",
                 reg_value.GetAsUInt32());
    m_context.Eip = reg_value.GetAsUInt32();
    break;
  case lldb_eflags_i386:
    WINLOG_IFALL(WINDOWS_LOG_REGISTERS, "Write value 0x%x to EFLAGS",
                 reg_value.GetAsUInt32());
    m_context.EFlags = reg_value.GetAsUInt32();
    break;
  default:
    WINWARN_IFALL(WINDOWS_LOG_REGISTERS,
                  "Write value 0x%x to unknown register %u",
                  reg_value.GetAsUInt32(), reg);
  }

  // Physically update the registers in the target process.
  TargetThreadWindows &wthread = static_cast<TargetThreadWindows &>(m_thread);
  return ::SetThreadContext(
      wthread.GetHostThread().GetNativeThread().GetSystemHandle(), &m_context);
}

} // namespace lldb_private

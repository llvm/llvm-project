//===-- RegisterContextWindowsLive_x64.cpp ----------------------*- C++ -*-===//
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

#include "RegisterContextWindowsLive_x64.h"
#include "TargetThreadWindows.h"
#include "lldb-x86-register-enums.h"

#include "llvm/ADT/STLExtras.h"

using namespace lldb;
using namespace lldb_private;

RegisterContextWindowsLive_x64::RegisterContextWindowsLive_x64(
    Thread &thread, uint32_t concrete_frame_idx)
    : RegisterContextWindows_x64(thread, concrete_frame_idx) {}

RegisterContextWindowsLive_x64::~RegisterContextWindowsLive_x64() {}

bool RegisterContextWindowsLive_x64::ReadRegister(const RegisterInfo *reg_info,
                                                  RegisterValue &reg_value) {
  if (!CacheAllRegisterValues())
    return false;

  switch (reg_info->kinds[eRegisterKindLLDB]) {
  case lldb_rax_x86_64:
    reg_value.SetUInt64(m_context.Rax);
    break;
  case lldb_rbx_x86_64:
    reg_value.SetUInt64(m_context.Rbx);
    break;
  case lldb_rcx_x86_64:
    reg_value.SetUInt64(m_context.Rcx);
    break;
  case lldb_rdx_x86_64:
    reg_value.SetUInt64(m_context.Rdx);
    break;
  case lldb_rdi_x86_64:
    reg_value.SetUInt64(m_context.Rdi);
    break;
  case lldb_rsi_x86_64:
    reg_value.SetUInt64(m_context.Rsi);
    break;
  case lldb_r8_x86_64:
    reg_value.SetUInt64(m_context.R8);
    break;
  case lldb_r9_x86_64:
    reg_value.SetUInt64(m_context.R9);
    break;
  case lldb_r10_x86_64:
    reg_value.SetUInt64(m_context.R10);
    break;
  case lldb_r11_x86_64:
    reg_value.SetUInt64(m_context.R11);
    break;
  case lldb_r12_x86_64:
    reg_value.SetUInt64(m_context.R12);
    break;
  case lldb_r13_x86_64:
    reg_value.SetUInt64(m_context.R13);
    break;
  case lldb_r14_x86_64:
    reg_value.SetUInt64(m_context.R14);
    break;
  case lldb_r15_x86_64:
    reg_value.SetUInt64(m_context.R15);
    break;
  case lldb_rbp_x86_64:
    reg_value.SetUInt64(m_context.Rbp);
    break;
  case lldb_rsp_x86_64:
    reg_value.SetUInt64(m_context.Rsp);
    break;
  case lldb_rip_x86_64:
    reg_value.SetUInt64(m_context.Rip);
    break;
  case lldb_rflags_x86_64:
    reg_value.SetUInt64(m_context.EFlags);
    break;
  }
  return true;
}

bool RegisterContextWindowsLive_x64::WriteRegister(
    const RegisterInfo *reg_info, const RegisterValue &reg_value) {
  // Since we cannot only write a single register value to the inferior, we need
  // to make sure
  // our cached copy of the register values are fresh.  Otherwise when writing
  // EAX, for example,
  // we may also overwrite some other register with a stale value.
  if (!CacheAllRegisterValues())
    return false;

  switch (reg_info->kinds[eRegisterKindLLDB]) {
  case lldb_rax_x86_64:
    m_context.Rax = reg_value.GetAsUInt64();
    break;
  case lldb_rbx_x86_64:
    m_context.Rbx = reg_value.GetAsUInt64();
    break;
  case lldb_rcx_x86_64:
    m_context.Rcx = reg_value.GetAsUInt64();
    break;
  case lldb_rdx_x86_64:
    m_context.Rdx = reg_value.GetAsUInt64();
    break;
  case lldb_rdi_x86_64:
    m_context.Rdi = reg_value.GetAsUInt64();
    break;
  case lldb_rsi_x86_64:
    m_context.Rsi = reg_value.GetAsUInt64();
    break;
  case lldb_r8_x86_64:
    m_context.R8 = reg_value.GetAsUInt64();
    break;
  case lldb_r9_x86_64:
    m_context.R9 = reg_value.GetAsUInt64();
    break;
  case lldb_r10_x86_64:
    m_context.R10 = reg_value.GetAsUInt64();
    break;
  case lldb_r11_x86_64:
    m_context.R11 = reg_value.GetAsUInt64();
    break;
  case lldb_r12_x86_64:
    m_context.R12 = reg_value.GetAsUInt64();
    break;
  case lldb_r13_x86_64:
    m_context.R13 = reg_value.GetAsUInt64();
    break;
  case lldb_r14_x86_64:
    m_context.R14 = reg_value.GetAsUInt64();
    break;
  case lldb_r15_x86_64:
    m_context.R15 = reg_value.GetAsUInt64();
    break;
  case lldb_rbp_x86_64:
    m_context.Rbp = reg_value.GetAsUInt64();
    break;
  case lldb_rsp_x86_64:
    m_context.Rsp = reg_value.GetAsUInt64();
    break;
  case lldb_rip_x86_64:
    m_context.Rip = reg_value.GetAsUInt64();
    break;
  case lldb_rflags_x86_64:
    m_context.EFlags = reg_value.GetAsUInt64();
    break;
  }

  // Physically update the registers in the target process.
  TargetThreadWindows &wthread = static_cast<TargetThreadWindows &>(m_thread);
  return ::SetThreadContext(
      wthread.GetHostThread().GetNativeThread().GetSystemHandle(), &m_context);
}

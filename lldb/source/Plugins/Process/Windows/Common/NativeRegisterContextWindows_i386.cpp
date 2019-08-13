//===-- NativeRegisterContextWindows_i386.cpp -------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#if defined(_WIN32) && !defined(_WIN64)

#include "NativeRegisterContextWindows_i386.h"

#include "NativeThreadWindows.h"
#include "Plugins/Process/Utility/RegisterContextWindows_i386.h"
#include "ProcessWindowsLog.h"
#include "lldb/Host/HostInfo.h"
#include "lldb/Host/HostThread.h"
#include "lldb/Host/Windows/HostThreadWindows.h"
#include "lldb/Host/windows/windows.h"

#include "lldb/Utility/Log.h"
#include "lldb/Utility/RegisterValue.h"
#include "llvm/ADT/STLExtras.h"

using namespace lldb;
using namespace lldb_private;

#define REG_CONTEXT_SIZE sizeof(::CONTEXT)

namespace {
static const uint32_t g_gpr_regnums_i386[] = {
    lldb_eax_i386,      lldb_ebx_i386,    lldb_ecx_i386, lldb_edx_i386,
    lldb_edi_i386,      lldb_esi_i386,    lldb_ebp_i386, lldb_esp_i386,
    lldb_eip_i386,      lldb_eflags_i386, lldb_cs_i386,  lldb_fs_i386,
    lldb_gs_i386,       lldb_ss_i386,     lldb_ds_i386,  lldb_es_i386,
    LLDB_INVALID_REGNUM // Register sets must be terminated with this flag.
};

static const RegisterSet g_reg_sets_i386[] = {
    {"General Purpose Registers", "gpr",
     llvm::array_lengthof(g_gpr_regnums_i386) - 1, g_gpr_regnums_i386},
};

enum { k_num_register_sets = 1 };

} // namespace

static RegisterInfoInterface *
CreateRegisterInfoInterface(const ArchSpec &target_arch) {
  assert((HostInfo::GetArchitecture().GetAddressByteSize() == 4) &&
         "Register setting path assumes this is a 32-bit host");
  return new RegisterContextWindows_i386(target_arch);
}

static Status GetThreadContextHelper(lldb::thread_t thread_handle,
                                     PCONTEXT context_ptr,
                                     const DWORD control_flag) {
  Log *log = ProcessWindowsLog::GetLogIfAny(WINDOWS_LOG_REGISTERS);
  Status error;

  memset(context_ptr, 0, sizeof(::CONTEXT));
  context_ptr->ContextFlags = control_flag;
  if (!::GetThreadContext(thread_handle, context_ptr)) {
    error.SetError(GetLastError(), eErrorTypeWin32);
    LLDB_LOG(log, "{0} GetThreadContext failed with error {1}", __FUNCTION__,
             error);
    return error;
  }
  return Status();
}

static Status SetThreadContextHelper(lldb::thread_t thread_handle,
                                     PCONTEXT context_ptr) {
  Log *log = ProcessWindowsLog::GetLogIfAny(WINDOWS_LOG_REGISTERS);
  Status error;

  if (!::SetThreadContext(thread_handle, context_ptr)) {
    error.SetError(GetLastError(), eErrorTypeWin32);
    LLDB_LOG(log, "{0} SetThreadContext failed with error {1}", __FUNCTION__,
             error);
    return error;
  }
  return Status();
}

std::unique_ptr<NativeRegisterContextWindows>
NativeRegisterContextWindows::CreateHostNativeRegisterContextWindows(
    const ArchSpec &target_arch, NativeThreadProtocol &native_thread) {
  return llvm::make_unique<NativeRegisterContextWindows_i386>(target_arch,
                                                              native_thread);
}

NativeRegisterContextWindows_i386::NativeRegisterContextWindows_i386(
    const ArchSpec &target_arch, NativeThreadProtocol &native_thread)
    : NativeRegisterContextWindows(native_thread,
                                   CreateRegisterInfoInterface(target_arch)) {}

bool NativeRegisterContextWindows_i386::IsGPR(uint32_t reg_index) const {
  return (reg_index < k_first_alias_i386);
}

uint32_t NativeRegisterContextWindows_i386::GetRegisterSetCount() const {
  return k_num_register_sets;
}

const RegisterSet *
NativeRegisterContextWindows_i386::GetRegisterSet(uint32_t set_index) const {
  if (set_index >= k_num_register_sets)
    return nullptr;
  return &g_reg_sets_i386[set_index];
}

Status NativeRegisterContextWindows_i386::GPRRead(const uint32_t reg,
                                                  RegisterValue &reg_value) {
  ::CONTEXT tls_context;
  DWORD context_flag = CONTEXT_CONTROL | CONTEXT_INTEGER | CONTEXT_SEGMENTS;
  Status error =
      GetThreadContextHelper(GetThreadHandle(), &tls_context, context_flag);
  if (error.Fail())
    return error;

  switch (reg) {
  case lldb_eax_i386:
    reg_value.SetUInt32(tls_context.Eax);
    break;
  case lldb_ebx_i386:
    reg_value.SetUInt32(tls_context.Ebx);
    break;
  case lldb_ecx_i386:
    reg_value.SetUInt32(tls_context.Ecx);
    break;
  case lldb_edx_i386:
    reg_value.SetUInt32(tls_context.Edx);
    break;
  case lldb_edi_i386:
    reg_value.SetUInt32(tls_context.Edi);
    break;
  case lldb_esi_i386:
    reg_value.SetUInt32(tls_context.Esi);
    break;
  case lldb_ebp_i386:
    reg_value.SetUInt32(tls_context.Ebp);
    break;
  case lldb_esp_i386:
    reg_value.SetUInt32(tls_context.Esp);
    break;
  case lldb_eip_i386:
    reg_value.SetUInt32(tls_context.Eip);
    break;
  case lldb_eflags_i386:
    reg_value.SetUInt32(tls_context.EFlags);
    break;
  case lldb_cs_i386:
    reg_value.SetUInt32(tls_context.SegCs);
    break;
  case lldb_fs_i386:
    reg_value.SetUInt32(tls_context.SegFs);
    break;
  case lldb_gs_i386:
    reg_value.SetUInt32(tls_context.SegGs);
    break;
  case lldb_ss_i386:
    reg_value.SetUInt32(tls_context.SegSs);
    break;
  case lldb_ds_i386:
    reg_value.SetUInt32(tls_context.SegDs);
    break;
  case lldb_es_i386:
    reg_value.SetUInt32(tls_context.SegEs);
    break;
  }

  return error;
}

Status
NativeRegisterContextWindows_i386::GPRWrite(const uint32_t reg,
                                            const RegisterValue &reg_value) {
  ::CONTEXT tls_context;
  DWORD context_flag = CONTEXT_CONTROL | CONTEXT_INTEGER | CONTEXT_SEGMENTS;
  auto thread_handle = GetThreadHandle();
  Status error =
      GetThreadContextHelper(thread_handle, &tls_context, context_flag);
  if (error.Fail())
    return error;

  switch (reg) {
  case lldb_eax_i386:
    tls_context.Eax = reg_value.GetAsUInt32();
    break;
  case lldb_ebx_i386:
    tls_context.Ebx = reg_value.GetAsUInt32();
    break;
  case lldb_ecx_i386:
    tls_context.Ecx = reg_value.GetAsUInt32();
    break;
  case lldb_edx_i386:
    tls_context.Edx = reg_value.GetAsUInt32();
    break;
  case lldb_edi_i386:
    tls_context.Edi = reg_value.GetAsUInt32();
    break;
  case lldb_esi_i386:
    tls_context.Esi = reg_value.GetAsUInt32();
    break;
  case lldb_ebp_i386:
    tls_context.Ebp = reg_value.GetAsUInt32();
    break;
  case lldb_esp_i386:
    tls_context.Esp = reg_value.GetAsUInt32();
    break;
  case lldb_eip_i386:
    tls_context.Eip = reg_value.GetAsUInt32();
    break;
  case lldb_eflags_i386:
    tls_context.EFlags = reg_value.GetAsUInt32();
    break;
  case lldb_cs_i386:
    tls_context.SegCs = reg_value.GetAsUInt32();
    break;
  case lldb_fs_i386:
    tls_context.SegFs = reg_value.GetAsUInt32();
    break;
  case lldb_gs_i386:
    tls_context.SegGs = reg_value.GetAsUInt32();
    break;
  case lldb_ss_i386:
    tls_context.SegSs = reg_value.GetAsUInt32();
    break;
  case lldb_ds_i386:
    tls_context.SegDs = reg_value.GetAsUInt32();
    break;
  case lldb_es_i386:
    tls_context.SegEs = reg_value.GetAsUInt32();
    break;
  }

  return SetThreadContextHelper(thread_handle, &tls_context);
}

Status
NativeRegisterContextWindows_i386::ReadRegister(const RegisterInfo *reg_info,
                                                RegisterValue &reg_value) {
  Status error;
  Log *log = ProcessWindowsLog::GetLogIfAny(WINDOWS_LOG_REGISTERS);

  if (!reg_info) {
    error.SetErrorString("reg_info NULL");
    return error;
  }

  const uint32_t reg = reg_info->kinds[lldb::eRegisterKindLLDB];
  if (reg == LLDB_INVALID_REGNUM) {
    // This is likely an internal register for lldb use only and should not be
    // directly queried.
    error.SetErrorStringWithFormat("register \"%s\" is an internal-only lldb "
                                   "register, cannot read directly",
                                   reg_info->name);
    return error;
  }

  if (IsGPR(reg))
    return GPRRead(reg, reg_value);

  return Status("unimplemented");
}

Status NativeRegisterContextWindows_i386::WriteRegister(
    const RegisterInfo *reg_info, const RegisterValue &reg_value) {
  Log *log = ProcessWindowsLog::GetLogIfAny(WINDOWS_LOG_REGISTERS);
  Status error;

  if (!reg_info) {
    error.SetErrorString("reg_info NULL");
    return error;
  }

  const uint32_t reg = reg_info->kinds[lldb::eRegisterKindLLDB];
  if (reg == LLDB_INVALID_REGNUM) {
    // This is likely an internal register for lldb use only and should not be
    // directly queried.
    error.SetErrorStringWithFormat("register \"%s\" is an internal-only lldb "
                                   "register, cannot read directly",
                                   reg_info->name);
    return error;
  }

  if (IsGPR(reg))
    return GPRRead(reg, reg_value);

  return Status("unimplemented");
}

Status NativeRegisterContextWindows_x86_64::ReadAllRegisterValues(
    lldb::DataBufferSP &data_sp) {
  const size_t data_size = REG_CONTEXT_SIZE;
  data_sp = std::make_shared<DataBufferHeap>(data_size, 0);
  ::CONTEXT tls_context;
  Status error =
      GetThreadContextHelper(GetThreadHandle(), &tls_context, CONTEXT_ALL);
  if (error.Fail())
    return error;

  uint8_t *dst = data_sp->GetBytes();
  ::memcpy(dst, &tls_context, data_size);
  return error;
}

Status NativeRegisterContextWindows_i386::WriteAllRegisterValues(
    const lldb::DataBufferSP &data_sp) {
  Status error;
  const size_t data_size = REG_CONTEXT_SIZE;
  if (!data_sp) {
    error.SetErrorStringWithFormat(
        "NativeRegisterContextWindows_i386::%s invalid data_sp provided",
        __FUNCTION__);
    return error;
  }

  if (data_sp->GetByteSize() != data_size) {
    error.SetErrorStringWithFormatv(
        "data_sp contained mismatched data size, expected {0}, actual {1}",
        data_size, data_sp->GetByteSize());
    return error;
  }

  ::CONTEXT tls_context;
  memcpy(&tls_context, data_sp->GetBytes(), data_size);
  return SetThreadContextHelper(GetThreadHandle(), &tls_context);
}

Status NativeRegisterContextWindows_i386::IsWatchpointHit(uint32_t wp_index,
                                                          bool &is_hit) {
  return Status("unimplemented");
}

Status NativeRegisterContextWindows_i386::GetWatchpointHitIndex(
    uint32_t &wp_index, lldb::addr_t trap_addr) {
  return Status("unimplemented");
}

Status NativeRegisterContextWindows_i386::IsWatchpointVacant(uint32_t wp_index,
                                                             bool &is_vacant) {
  return Status("unimplemented");
}

Status NativeRegisterContextWindows_i386::SetHardwareWatchpointWithIndex(
    lldb::addr_t addr, size_t size, uint32_t watch_flags, uint32_t wp_index) {
  return Status("unimplemented");
}

bool NativeRegisterContextWindows_i386::ClearHardwareWatchpoint(
    uint32_t wp_index) {
  return false;
}

Status NativeRegisterContextWindows_i386::ClearAllHardwareWatchpoints() {
  return Status("unimplemented");
}

uint32_t NativeRegisterContextWindows_i386::SetHardwareWatchpoint(
    lldb::addr_t addr, size_t size, uint32_t watch_flags) {
  return LLDB_INVALID_INDEX32;
}

lldb::addr_t
NativeRegisterContextWindows_i386::GetWatchpointAddress(uint32_t wp_index) {
  return LLDB_INVALID_ADDRESS;
}

uint32_t NativeRegisterContextWindows_i386::NumSupportedHardwareWatchpoints() {
  // Not implemented
  return 0;
}

#endif

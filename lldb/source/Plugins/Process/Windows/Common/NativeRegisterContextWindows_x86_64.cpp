//===-- NativeRegisterContextWindows_x86_64.cpp -----------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#if defined(_WIN64)

#include "NativeRegisterContextWindows_x86_64.h"
#include "NativeRegisterContextWindows_WoW64.h"
#include "NativeThreadWindows.h"
#include "Plugins/Process/Utility/RegisterContextWindows_i386.h"
#include "Plugins/Process/Utility/RegisterContextWindows_x86_64.h"
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
static const uint32_t g_gpr_regnums_x86_64[] = {
    lldb_rax_x86_64,    lldb_rbx_x86_64,    lldb_rcx_x86_64, lldb_rdx_x86_64,
    lldb_rdi_x86_64,    lldb_rsi_x86_64,    lldb_rbp_x86_64, lldb_rsp_x86_64,
    lldb_r8_x86_64,     lldb_r9_x86_64,     lldb_r10_x86_64, lldb_r11_x86_64,
    lldb_r12_x86_64,    lldb_r13_x86_64,    lldb_r14_x86_64, lldb_r15_x86_64,
    lldb_rip_x86_64,    lldb_rflags_x86_64, lldb_cs_x86_64,  lldb_fs_x86_64,
    lldb_gs_x86_64,     lldb_ss_x86_64,     lldb_ds_x86_64,  lldb_es_x86_64,
    LLDB_INVALID_REGNUM // Register set must be terminated with this flag
};

static const uint32_t g_fpr_regnums_x86_64[] = {
    lldb_xmm0_x86_64,   lldb_xmm1_x86_64,  lldb_xmm2_x86_64,  lldb_xmm3_x86_64,
    lldb_xmm4_x86_64,   lldb_xmm5_x86_64,  lldb_xmm6_x86_64,  lldb_xmm7_x86_64,
    lldb_xmm8_x86_64,   lldb_xmm9_x86_64,  lldb_xmm10_x86_64, lldb_xmm11_x86_64,
    lldb_xmm12_x86_64,  lldb_xmm13_x86_64, lldb_xmm14_x86_64, lldb_xmm15_x86_64,
    LLDB_INVALID_REGNUM // Register set must be terminated with this flag
};

static const RegisterSet g_reg_sets_x86_64[] = {
    {"General Purpose Registers", "gpr",
     llvm::array_lengthof(g_gpr_regnums_x86_64) - 1, g_gpr_regnums_x86_64},
    {"Floating Point Registers", "fpr",
     llvm::array_lengthof(g_fpr_regnums_x86_64) - 1, g_fpr_regnums_x86_64}};

enum { k_num_register_sets = 2 };

} // namespace

static RegisterInfoInterface *
CreateRegisterInfoInterface(const ArchSpec &target_arch) {
  assert((HostInfo::GetArchitecture().GetAddressByteSize() == 8) &&
         "Register setting path assumes this is a 64-bit host");
  return new RegisterContextWindows_x86_64(target_arch);
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
  // It's assumed that the thread has stopped.
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
  // Register context for a WoW64 application.
  if (target_arch.GetAddressByteSize() == 4)
    return std::make_unique<NativeRegisterContextWindows_WoW64>(target_arch,
                                                                 native_thread);

  // Register context for a native 64-bit application.
  return std::make_unique<NativeRegisterContextWindows_x86_64>(target_arch,
                                                                native_thread);
}

NativeRegisterContextWindows_x86_64::NativeRegisterContextWindows_x86_64(
    const ArchSpec &target_arch, NativeThreadProtocol &native_thread)
    : NativeRegisterContextWindows(native_thread,
                                   CreateRegisterInfoInterface(target_arch)) {}

bool NativeRegisterContextWindows_x86_64::IsGPR(uint32_t reg_index) const {
  return (reg_index >= k_first_gpr_x86_64 && reg_index < k_first_alias_x86_64);
}

bool NativeRegisterContextWindows_x86_64::IsFPR(uint32_t reg_index) const {
  return (reg_index >= lldb_xmm0_x86_64 && reg_index <= k_last_fpr_x86_64);
}

uint32_t NativeRegisterContextWindows_x86_64::GetRegisterSetCount() const {
  return k_num_register_sets;
}

const RegisterSet *
NativeRegisterContextWindows_x86_64::GetRegisterSet(uint32_t set_index) const {
  if (set_index >= k_num_register_sets)
    return nullptr;
  return &g_reg_sets_x86_64[set_index];
}

Status NativeRegisterContextWindows_x86_64::GPRRead(const uint32_t reg,
                                                    RegisterValue &reg_value) {
  ::CONTEXT tls_context;
  DWORD context_flag = CONTEXT_CONTROL | CONTEXT_INTEGER | CONTEXT_SEGMENTS;
  Status error =
      GetThreadContextHelper(GetThreadHandle(), &tls_context, context_flag);
  if (error.Fail())
    return error;

  switch (reg) {
  case lldb_rax_x86_64:
    reg_value.SetUInt64(tls_context.Rax);
    break;
  case lldb_rbx_x86_64:
    reg_value.SetUInt64(tls_context.Rbx);
    break;
  case lldb_rcx_x86_64:
    reg_value.SetUInt64(tls_context.Rcx);
    break;
  case lldb_rdx_x86_64:
    reg_value.SetUInt64(tls_context.Rdx);
    break;
  case lldb_rdi_x86_64:
    reg_value.SetUInt64(tls_context.Rdi);
    break;
  case lldb_rsi_x86_64:
    reg_value.SetUInt64(tls_context.Rsi);
    break;
  case lldb_rbp_x86_64:
    reg_value.SetUInt64(tls_context.Rbp);
    break;
  case lldb_rsp_x86_64:
    reg_value.SetUInt64(tls_context.Rsp);
    break;
  case lldb_r8_x86_64:
    reg_value.SetUInt64(tls_context.R8);
    break;
  case lldb_r9_x86_64:
    reg_value.SetUInt64(tls_context.R9);
    break;
  case lldb_r10_x86_64:
    reg_value.SetUInt64(tls_context.R10);
    break;
  case lldb_r11_x86_64:
    reg_value.SetUInt64(tls_context.R11);
    break;
  case lldb_r12_x86_64:
    reg_value.SetUInt64(tls_context.R12);
    break;
  case lldb_r13_x86_64:
    reg_value.SetUInt64(tls_context.R13);
    break;
  case lldb_r14_x86_64:
    reg_value.SetUInt64(tls_context.R14);
    break;
  case lldb_r15_x86_64:
    reg_value.SetUInt64(tls_context.R15);
    break;
  case lldb_rip_x86_64:
    reg_value.SetUInt64(tls_context.Rip);
    break;
  case lldb_rflags_x86_64:
    reg_value.SetUInt64(tls_context.EFlags | 0x2); // Bit #1 always 1
    break;
  case lldb_cs_x86_64:
    reg_value.SetUInt16(tls_context.SegCs);
    break;
  case lldb_fs_x86_64:
    reg_value.SetUInt16(tls_context.SegFs);
    break;
  case lldb_gs_x86_64:
    reg_value.SetUInt16(tls_context.SegGs);
    break;
  case lldb_ss_x86_64:
    reg_value.SetUInt16(tls_context.SegSs);
    break;
  case lldb_ds_x86_64:
    reg_value.SetUInt16(tls_context.SegDs);
    break;
  case lldb_es_x86_64:
    reg_value.SetUInt16(tls_context.SegEs);
    break;
  }

  return error;
}

Status
NativeRegisterContextWindows_x86_64::GPRWrite(const uint32_t reg,
                                              const RegisterValue &reg_value) {
  ::CONTEXT tls_context;
  DWORD context_flag = CONTEXT_CONTROL | CONTEXT_INTEGER | CONTEXT_SEGMENTS;
  auto thread_handle = GetThreadHandle();
  Status error =
      GetThreadContextHelper(thread_handle, &tls_context, context_flag);
  if (error.Fail())
    return error;

  switch (reg) {
  case lldb_rax_x86_64:
    tls_context.Rax = reg_value.GetAsUInt64();
    break;
  case lldb_rbx_x86_64:
    tls_context.Rbx = reg_value.GetAsUInt64();
    break;
  case lldb_rcx_x86_64:
    tls_context.Rcx = reg_value.GetAsUInt64();
    break;
  case lldb_rdx_x86_64:
    tls_context.Rdx = reg_value.GetAsUInt64();
    break;
  case lldb_rdi_x86_64:
    tls_context.Rdi = reg_value.GetAsUInt64();
    break;
  case lldb_rsi_x86_64:
    tls_context.Rsi = reg_value.GetAsUInt64();
    break;
  case lldb_rbp_x86_64:
    tls_context.Rbp = reg_value.GetAsUInt64();
    break;
  case lldb_rsp_x86_64:
    tls_context.Rsp = reg_value.GetAsUInt64();
    break;
  case lldb_r8_x86_64:
    tls_context.R8 = reg_value.GetAsUInt64();
    break;
  case lldb_r9_x86_64:
    tls_context.R9 = reg_value.GetAsUInt64();
    break;
  case lldb_r10_x86_64:
    tls_context.R10 = reg_value.GetAsUInt64();
    break;
  case lldb_r11_x86_64:
    tls_context.R11 = reg_value.GetAsUInt64();
    break;
  case lldb_r12_x86_64:
    tls_context.R12 = reg_value.GetAsUInt64();
    break;
  case lldb_r13_x86_64:
    tls_context.R13 = reg_value.GetAsUInt64();
    break;
  case lldb_r14_x86_64:
    tls_context.R14 = reg_value.GetAsUInt64();
    break;
  case lldb_r15_x86_64:
    tls_context.R15 = reg_value.GetAsUInt64();
    break;
  case lldb_rip_x86_64:
    tls_context.Rip = reg_value.GetAsUInt64();
    break;
  case lldb_rflags_x86_64:
    tls_context.EFlags = reg_value.GetAsUInt64();
    break;
  case lldb_cs_x86_64:
    tls_context.SegCs = reg_value.GetAsUInt16();
    break;
  case lldb_fs_x86_64:
    tls_context.SegFs = reg_value.GetAsUInt16();
    break;
  case lldb_gs_x86_64:
    tls_context.SegGs = reg_value.GetAsUInt16();
    break;
  case lldb_ss_x86_64:
    tls_context.SegSs = reg_value.GetAsUInt16();
    break;
  case lldb_ds_x86_64:
    tls_context.SegDs = reg_value.GetAsUInt16();
    break;
  case lldb_es_x86_64:
    tls_context.SegEs = reg_value.GetAsUInt16();
    break;
  }

  return SetThreadContextHelper(thread_handle, &tls_context);
}

Status NativeRegisterContextWindows_x86_64::FPRRead(const uint32_t reg,
                                                    RegisterValue &reg_value) {
  ::CONTEXT tls_context;
  DWORD context_flag = CONTEXT_CONTROL | CONTEXT_FLOATING_POINT;
  Status error =
      GetThreadContextHelper(GetThreadHandle(), &tls_context, context_flag);
  if (error.Fail())
    return error;

  switch (reg) {
  case lldb_xmm0_x86_64:
    reg_value.SetBytes(&tls_context.Xmm0, 16, endian::InlHostByteOrder());
    break;
  case lldb_xmm1_x86_64:
    reg_value.SetBytes(&tls_context.Xmm1, 16, endian::InlHostByteOrder());
    break;
  case lldb_xmm2_x86_64:
    reg_value.SetBytes(&tls_context.Xmm2, 16, endian::InlHostByteOrder());
    break;
  case lldb_xmm3_x86_64:
    reg_value.SetBytes(&tls_context.Xmm3, 16, endian::InlHostByteOrder());
    break;
  case lldb_xmm4_x86_64:
    reg_value.SetBytes(&tls_context.Xmm4, 16, endian::InlHostByteOrder());
    break;
  case lldb_xmm5_x86_64:
    reg_value.SetBytes(&tls_context.Xmm5, 16, endian::InlHostByteOrder());
    break;
  case lldb_xmm6_x86_64:
    reg_value.SetBytes(&tls_context.Xmm6, 16, endian::InlHostByteOrder());
    break;
  case lldb_xmm7_x86_64:
    reg_value.SetBytes(&tls_context.Xmm7, 16, endian::InlHostByteOrder());
    break;
  case lldb_xmm8_x86_64:
    reg_value.SetBytes(&tls_context.Xmm8, 16, endian::InlHostByteOrder());
    break;
  case lldb_xmm9_x86_64:
    reg_value.SetBytes(&tls_context.Xmm9, 16, endian::InlHostByteOrder());
    break;
  case lldb_xmm10_x86_64:
    reg_value.SetBytes(&tls_context.Xmm10, 16, endian::InlHostByteOrder());
    break;
  case lldb_xmm11_x86_64:
    reg_value.SetBytes(&tls_context.Xmm11, 16, endian::InlHostByteOrder());
    break;
  case lldb_xmm12_x86_64:
    reg_value.SetBytes(&tls_context.Xmm12, 16, endian::InlHostByteOrder());
    break;
  case lldb_xmm13_x86_64:
    reg_value.SetBytes(&tls_context.Xmm13, 16, endian::InlHostByteOrder());
    break;
  case lldb_xmm14_x86_64:
    reg_value.SetBytes(&tls_context.Xmm14, 16, endian::InlHostByteOrder());
    break;
  case lldb_xmm15_x86_64:
    reg_value.SetBytes(&tls_context.Xmm15, 16, endian::InlHostByteOrder());
    break;
  }

  return error;
}

Status
NativeRegisterContextWindows_x86_64::FPRWrite(const uint32_t reg,
                                              const RegisterValue &reg_value) {
  ::CONTEXT tls_context;
  DWORD context_flag = CONTEXT_CONTROL | CONTEXT_FLOATING_POINT;
  auto thread_handle = GetThreadHandle();
  Status error =
      GetThreadContextHelper(thread_handle, &tls_context, context_flag);
  if (error.Fail())
    return error;

  switch (reg) {
  case lldb_xmm0_x86_64:
    memcpy(&tls_context.Xmm0, reg_value.GetBytes(), 16);
    break;
  case lldb_xmm1_x86_64:
    memcpy(&tls_context.Xmm1, reg_value.GetBytes(), 16);
    break;
  case lldb_xmm2_x86_64:
    memcpy(&tls_context.Xmm2, reg_value.GetBytes(), 16);
    break;
  case lldb_xmm3_x86_64:
    memcpy(&tls_context.Xmm3, reg_value.GetBytes(), 16);
    break;
  case lldb_xmm4_x86_64:
    memcpy(&tls_context.Xmm4, reg_value.GetBytes(), 16);
    break;
  case lldb_xmm5_x86_64:
    memcpy(&tls_context.Xmm5, reg_value.GetBytes(), 16);
    break;
  case lldb_xmm6_x86_64:
    memcpy(&tls_context.Xmm6, reg_value.GetBytes(), 16);
    break;
  case lldb_xmm7_x86_64:
    memcpy(&tls_context.Xmm7, reg_value.GetBytes(), 16);
    break;
  case lldb_xmm8_x86_64:
    memcpy(&tls_context.Xmm8, reg_value.GetBytes(), 16);
    break;
  case lldb_xmm9_x86_64:
    memcpy(&tls_context.Xmm9, reg_value.GetBytes(), 16);
    break;
  case lldb_xmm10_x86_64:
    memcpy(&tls_context.Xmm10, reg_value.GetBytes(), 16);
    break;
  case lldb_xmm11_x86_64:
    memcpy(&tls_context.Xmm11, reg_value.GetBytes(), 16);
    break;
  case lldb_xmm12_x86_64:
    memcpy(&tls_context.Xmm12, reg_value.GetBytes(), 16);
    break;
  case lldb_xmm13_x86_64:
    memcpy(&tls_context.Xmm13, reg_value.GetBytes(), 16);
    break;
  case lldb_xmm14_x86_64:
    memcpy(&tls_context.Xmm14, reg_value.GetBytes(), 16);
    break;
  case lldb_xmm15_x86_64:
    memcpy(&tls_context.Xmm15, reg_value.GetBytes(), 16);
    break;
  }

  return SetThreadContextHelper(thread_handle, &tls_context);
}

Status
NativeRegisterContextWindows_x86_64::ReadRegister(const RegisterInfo *reg_info,
                                                  RegisterValue &reg_value) {
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

  if (IsFPR(reg))
    return FPRRead(reg, reg_value);

  return Status("unimplemented");
}

Status NativeRegisterContextWindows_x86_64::WriteRegister(
    const RegisterInfo *reg_info, const RegisterValue &reg_value) {
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
    return GPRWrite(reg, reg_value);

  if (IsFPR(reg))
    return FPRWrite(reg, reg_value);

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

Status NativeRegisterContextWindows_x86_64::WriteAllRegisterValues(
    const lldb::DataBufferSP &data_sp) {
  Status error;
  const size_t data_size = REG_CONTEXT_SIZE;
  if (!data_sp) {
    error.SetErrorStringWithFormat(
        "NativeRegisterContextWindows_x86_64::%s invalid data_sp provided",
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

Status NativeRegisterContextWindows_x86_64::IsWatchpointHit(uint32_t wp_index,
                                                            bool &is_hit) {
  return Status("unimplemented");
}

Status NativeRegisterContextWindows_x86_64::GetWatchpointHitIndex(
    uint32_t &wp_index, lldb::addr_t trap_addr) {
  return Status("unimplemented");
}

Status
NativeRegisterContextWindows_x86_64::IsWatchpointVacant(uint32_t wp_index,
                                                        bool &is_vacant) {
  return Status("unimplemented");
}

Status NativeRegisterContextWindows_x86_64::SetHardwareWatchpointWithIndex(
    lldb::addr_t addr, size_t size, uint32_t watch_flags, uint32_t wp_index) {
  return Status("unimplemented");
}

bool NativeRegisterContextWindows_x86_64::ClearHardwareWatchpoint(
    uint32_t wp_index) {
  return false;
}

Status NativeRegisterContextWindows_x86_64::ClearAllHardwareWatchpoints() {
  return Status("unimplemented");
}

uint32_t NativeRegisterContextWindows_x86_64::SetHardwareWatchpoint(
    lldb::addr_t addr, size_t size, uint32_t watch_flags) {
  return LLDB_INVALID_INDEX32;
}

lldb::addr_t
NativeRegisterContextWindows_x86_64::GetWatchpointAddress(uint32_t wp_index) {
  return LLDB_INVALID_ADDRESS;
}

uint32_t
NativeRegisterContextWindows_x86_64::NumSupportedHardwareWatchpoints() {
  // Not implemented
  return 0;
}

#endif // defined(_WIN64)

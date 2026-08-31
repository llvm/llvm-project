//===-- RegisterContextWindows_arm64.cpp ----------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#if defined(__aarch64__) || defined(_M_ARM64)

#include "RegisterContextWindows_arm64.h"
#include "ProcessWindowsLog.h"
#include "TargetThreadWindows.h"

#include "lldb/Host/windows/HostThreadWindows.h"
#include "lldb/Host/windows/windows.h"
#include "lldb/Utility/RegisterValue.h"
#include "lldb/Utility/Status.h"
#include "lldb/lldb-private-types.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/ScopeExit.h"

using namespace lldb;
using namespace lldb_private;

#define GPR_OFFSET(idx) 0
#define GPR_OFFSET_NAME(reg) 0

#define FPU_OFFSET(idx) 0
#define FPU_OFFSET_NAME(reg) 0

#define EXC_OFFSET_NAME(reg) 0
#define DBG_OFFSET_NAME(reg) 0

#define DEFINE_DBG(reg, i)                                                     \
  #reg, nullptr, 0, DBG_OFFSET_NAME(reg[i]), eEncodingUint, eFormatHex,        \
      {LLDB_INVALID_REGNUM, LLDB_INVALID_REGNUM, LLDB_INVALID_REGNUM,          \
       LLDB_INVALID_REGNUM, LLDB_INVALID_REGNUM},                              \
      nullptr, nullptr, nullptr

// Include RegisterInfos_arm64 to declare our g_register_infos_arm64 structure.
#define DECLARE_REGISTER_INFOS_ARM64_STRUCT
#include "Plugins/Process/Utility/RegisterInfos_arm64.h"
#undef DECLARE_REGISTER_INFOS_ARM64_STRUCT

static size_t k_num_register_infos = std::size(g_register_infos_arm64_le);

// Array of lldb register numbers used to define the set of all General Purpose
// Registers
uint32_t g_gpr_reg_indices[] = {
    gpr_x0,  gpr_x1,   gpr_x2,  gpr_x3,  gpr_x4,  gpr_x5,  gpr_x6,  gpr_x7,
    gpr_x8,  gpr_x9,   gpr_x10, gpr_x11, gpr_x12, gpr_x13, gpr_x14, gpr_x15,
    gpr_x16, gpr_x17,  gpr_x18, gpr_x19, gpr_x20, gpr_x21, gpr_x22, gpr_x23,
    gpr_x24, gpr_x25,  gpr_x26, gpr_x27, gpr_x28, gpr_fp,  gpr_lr,  gpr_sp,
    gpr_pc,  gpr_cpsr,

    gpr_w0,  gpr_w1,   gpr_w2,  gpr_w3,  gpr_w4,  gpr_w5,  gpr_w6,  gpr_w7,
    gpr_w8,  gpr_w9,   gpr_w10, gpr_w11, gpr_w12, gpr_w13, gpr_w14, gpr_w15,
    gpr_w16, gpr_w17,  gpr_w18, gpr_w19, gpr_w20, gpr_w21, gpr_w22, gpr_w23,
    gpr_w24, gpr_w25,  gpr_w26, gpr_w27, gpr_w28,
};

uint32_t g_fpu_reg_indices[] = {
    fpu_v0,   fpu_v1,   fpu_v2,  fpu_v3,  fpu_v4,  fpu_v5,  fpu_v6,  fpu_v7,
    fpu_v8,   fpu_v9,   fpu_v10, fpu_v11, fpu_v12, fpu_v13, fpu_v14, fpu_v15,
    fpu_v16,  fpu_v17,  fpu_v18, fpu_v19, fpu_v20, fpu_v21, fpu_v22, fpu_v23,
    fpu_v24,  fpu_v25,  fpu_v26, fpu_v27, fpu_v28, fpu_v29, fpu_v30, fpu_v31,

    fpu_s0,   fpu_s1,   fpu_s2,  fpu_s3,  fpu_s4,  fpu_s5,  fpu_s6,  fpu_s7,
    fpu_s8,   fpu_s9,   fpu_s10, fpu_s11, fpu_s12, fpu_s13, fpu_s14, fpu_s15,
    fpu_s16,  fpu_s17,  fpu_s18, fpu_s19, fpu_s20, fpu_s21, fpu_s22, fpu_s23,
    fpu_s24,  fpu_s25,  fpu_s26, fpu_s27, fpu_s28, fpu_s29, fpu_s30, fpu_s31,

    fpu_d0,   fpu_d1,   fpu_d2,  fpu_d3,  fpu_d4,  fpu_d5,  fpu_d6,  fpu_d7,
    fpu_d8,   fpu_d9,   fpu_d10, fpu_d11, fpu_d12, fpu_d13, fpu_d14, fpu_d15,
    fpu_d16,  fpu_d17,  fpu_d18, fpu_d19, fpu_d20, fpu_d21, fpu_d22, fpu_d23,
    fpu_d24,  fpu_d25,  fpu_d26, fpu_d27, fpu_d28, fpu_d29, fpu_d30, fpu_d31,

    fpu_fpsr, fpu_fpcr,
};

RegisterSet g_register_sets[] = {
    {"General Purpose Registers", "gpr", std::size(g_gpr_reg_indices),
     g_gpr_reg_indices},
    {"Floating Point Registers", "fpu", std::size(g_fpu_reg_indices),
     g_fpu_reg_indices},
};

static Status GetThreadContextLength(DWORD context_flags,
                                     DWORD &context_length) {
  Log *log = GetLog(WindowsLog::Registers);
  Status error;

  if (InitializeContext(nullptr, context_flags, nullptr, &context_length)) {
    error = Status::FromErrorString("InitializeContext succeeded unexpectedly");
    LLDB_LOG(log, "{0}", error);
    return error;
  }

  if (GetLastError() != ERROR_INSUFFICIENT_BUFFER) {
    error = Status(GetLastError(), eErrorTypeWin32);
    LLDB_LOG(log,
             "InitializeContext failed with unexpected error {0}, expected "
             "ERROR_INSUFFICIENT_BUFFER",
             error);
    return error;
  }

  return error;
}

static Status GetThreadContextHelper(lldb::thread_t thread_handle,
                                     DWORD context_flags, PCONTEXT &context,
                                     DataBufferHeap *context_buffer) {
  Log *log = GetLog(WindowsLog::Registers);
  Status error;
  DWORD context_length = 0;

  if (!context_buffer) {
    error = Status::FromErrorString("context buffer not allocated");
    LLDB_LOG(log, "{0}", error);
    return error;
  }

  error = GetThreadContextLength(context_flags, context_length);
  if (error.Fail())
    return error;

  if (context_buffer->SetByteSize(context_length) != context_length) {
    error = Status::FromErrorString("failed to resize context buffer");
    LLDB_LOG(log, "{0}", error);
    return error;
  }

  if (!InitializeContext(context_buffer->GetBytes(), context_flags, &context,
                         &context_length)) {
    error = Status(GetLastError(), eErrorTypeWin32);
    LLDB_LOG(log, "InitializeContext failed with error {0}", error);
    return error;
  }

  if (!::GetThreadContext(thread_handle, context)) {
    error = Status(GetLastError(), eErrorTypeWin32);
    LLDB_LOG(log, "GetThreadContext failed with error {0}", error);
    return error;
  }

  return error;
}

static bool SetThreadContextHelper(lldb::thread_t thread_handle,
                                   PCONTEXT context) {
  Log *log = GetLog(WindowsLog::Registers);

  if (!::SetThreadContext(thread_handle, context)) {
    LLDB_LOG(log, "SetThreadContext failed with error {0}", GetLastError());
    return false;
  }

  return true;
}

// Constructors and Destructors
RegisterContextWindows_arm64::RegisterContextWindows_arm64(
    Thread &thread, uint32_t concrete_frame_idx)
    : RegisterContextWindows(thread, concrete_frame_idx),
      m_context_arm64(nullptr), m_context_arm64_buffer(nullptr) {}

RegisterContextWindows_arm64::~RegisterContextWindows_arm64() {}

size_t RegisterContextWindows_arm64::GetRegisterCount() {
  return std::size(g_register_infos_arm64_le);
}

const RegisterInfo *
RegisterContextWindows_arm64::GetRegisterInfoAtIndex(size_t reg) {
  if (reg < k_num_register_infos)
    return &g_register_infos_arm64_le[reg];
  return nullptr;
}

size_t RegisterContextWindows_arm64::GetRegisterSetCount() {
  return std::size(g_register_sets);
}

const RegisterSet *
RegisterContextWindows_arm64::GetRegisterSet(size_t reg_set) {
  return &g_register_sets[reg_set];
}

bool RegisterContextWindows_arm64::ReadRegister(const RegisterInfo *reg_info,
                                                RegisterValue &reg_value) {
  Log *log = GetLog(WindowsLog::Registers);

  if (!reg_info) {
    LLDB_LOG(log, "reg_info NULL");
    return false;
  }

  const uint32_t reg = reg_info->kinds[lldb::eRegisterKindLLDB];
  if (reg == LLDB_INVALID_REGNUM) {
    // This is likely an internal register for lldb use only and should not be
    // directly queried.
    LLDB_LOG(log,
             "register is an internal-only lldb register, cannot read "
             "directly {1}",
             reg_info->name);
    return false;
  }

  if (IsGPR(reg))
    return GPRRead(reg, reg_value);

  if (IsFPR(reg))
    return FPRRead(reg, reg_value);

  LLDB_LOG(log, "unsupported register");
  return false;
}

bool RegisterContextWindows_arm64::WriteRegister(
    const RegisterInfo *reg_info, const RegisterValue &reg_value) {
  Log *log = GetLog(WindowsLog::Registers);

  if (!reg_info) {
    LLDB_LOG(log, "reg_info NULL");
    return false;
  }

  const uint32_t reg = reg_info->kinds[lldb::eRegisterKindLLDB];
  if (reg == LLDB_INVALID_REGNUM) {
    // This is likely an internal register for lldb use only and should not be
    // directly queried.
    LLDB_LOG(log,
             "register is an internal-only lldb register, cannot read "
             "directly {1}",
             reg_info->name);
    return false;
  }

  if (IsGPR(reg))
    return GPRWrite(reg, reg_value);

  if (IsFPR(reg))
    return FPRWrite(reg, reg_value);

  LLDB_LOG(log, "unsupported register");
  return false;
}

void RegisterContextWindows_arm64::InvalidateAllRegisters() {
  m_context_arm64 = nullptr;
  m_context_arm64_buffer.reset();
}

bool RegisterContextWindows_arm64::ReadAllRegisterValues(
    lldb::WritableDataBufferSP &data_sp) {
  Log *log = GetLog(WindowsLog::Registers);

  if (!CacheAllRegisterValues())
    return false;

  if (!m_context_arm64_buffer) {
    LLDB_LOG(log, "register context buffer is not available");
    return false;
  }

  data_sp =
      std::make_shared<DataBufferHeap>(m_context_arm64_buffer->GetBytes(),
                                       m_context_arm64_buffer->GetByteSize());

  return true;
}

bool RegisterContextWindows_arm64::WriteAllRegisterValues(
    const lldb::DataBufferSP &data_sp) {
  Log *log = GetLog(WindowsLog::Registers);
  Status error;

  auto cleanup = llvm::make_scope_exit([&]() { m_context_arm64 = nullptr; });

  if (!data_sp) {
    LLDB_LOG(log, "invalid data_sp", error);
    return false;
  }

  DWORD context_flags = CONTEXT_ALL;
  DWORD context_length = 0;

  error = GetThreadContextLength(context_flags, context_length);
  if (error.Fail())
    return false;

  if (data_sp->GetByteSize() != context_length) {
    LLDB_LOG(log,
             "data_sp contained mismatched data size, expected {0}, actual {1}",
             context_length, data_sp->GetByteSize());
    return false;
  }

  PCONTEXT context = nullptr;
  DataBufferHeap context_buffer;
  error = GetThreadContextHelper(GetThreadHandle(), context_flags, context,
                                 &context_buffer);
  if (error.Fail())
    return false;

  ::memcpy(context_buffer.GetBytes(), data_sp->GetBytes(), context_length);

  return SetThreadContextHelper(GetThreadHandle(), context);
}

bool RegisterContextWindows_arm64::CacheAllRegisterValues() {
  DWORD context_flags = CONTEXT_ALL;

  if (m_context_arm64 &&
      (m_context_arm64->ContextFlags & context_flags) == context_flags)
    return true;

  m_context_arm64 = nullptr;

  if (!m_context_arm64_buffer)
    m_context_arm64_buffer = std::make_shared<DataBufferHeap>();

  if (GetThreadContextHelper(GetThreadHandle(), context_flags, m_context_arm64,
                             m_context_arm64_buffer.get())
          .Fail()) {
    m_context_arm64 = nullptr;
    return false;
  }

  return true;
}

bool RegisterContextWindows_arm64::GPRRead(const uint32_t reg,
                                           RegisterValue &reg_value) {
  if (!CacheAllRegisterValues())
    return false;

  switch (reg) {
  case gpr_x0_arm64:
  case gpr_x1_arm64:
  case gpr_x2_arm64:
  case gpr_x3_arm64:
  case gpr_x4_arm64:
  case gpr_x5_arm64:
  case gpr_x6_arm64:
  case gpr_x7_arm64:
  case gpr_x8_arm64:
  case gpr_x9_arm64:
  case gpr_x10_arm64:
  case gpr_x11_arm64:
  case gpr_x12_arm64:
  case gpr_x13_arm64:
  case gpr_x14_arm64:
  case gpr_x15_arm64:
  case gpr_x16_arm64:
  case gpr_x17_arm64:
  case gpr_x18_arm64:
  case gpr_x19_arm64:
  case gpr_x20_arm64:
  case gpr_x21_arm64:
  case gpr_x22_arm64:
  case gpr_x23_arm64:
  case gpr_x24_arm64:
  case gpr_x25_arm64:
  case gpr_x26_arm64:
  case gpr_x27_arm64:
  case gpr_x28_arm64:
    reg_value.SetUInt64(m_context_arm64->X[reg - gpr_x0_arm64]);
    break;

  case gpr_fp_arm64:
    reg_value.SetUInt64(m_context_arm64->Fp);
    break;
  case gpr_sp_arm64:
    reg_value.SetUInt64(m_context_arm64->Sp);
    break;
  case gpr_lr_arm64:
    reg_value.SetUInt64(m_context_arm64->Lr);
    break;
  case gpr_pc_arm64:
    reg_value.SetUInt64(m_context_arm64->Pc);
    break;
  case gpr_cpsr_arm64:
    reg_value.SetUInt32(m_context_arm64->Cpsr);
    break;

  case gpr_w0_arm64:
  case gpr_w1_arm64:
  case gpr_w2_arm64:
  case gpr_w3_arm64:
  case gpr_w4_arm64:
  case gpr_w5_arm64:
  case gpr_w6_arm64:
  case gpr_w7_arm64:
  case gpr_w8_arm64:
  case gpr_w9_arm64:
  case gpr_w10_arm64:
  case gpr_w11_arm64:
  case gpr_w12_arm64:
  case gpr_w13_arm64:
  case gpr_w14_arm64:
  case gpr_w15_arm64:
  case gpr_w16_arm64:
  case gpr_w17_arm64:
  case gpr_w18_arm64:
  case gpr_w19_arm64:
  case gpr_w20_arm64:
  case gpr_w21_arm64:
  case gpr_w22_arm64:
  case gpr_w23_arm64:
  case gpr_w24_arm64:
  case gpr_w25_arm64:
  case gpr_w26_arm64:
  case gpr_w27_arm64:
  case gpr_w28_arm64:
    reg_value.SetUInt32(static_cast<uint32_t>(
        m_context_arm64->X[reg - gpr_w0_arm64] & 0xffffffff));
    break;
  }

  return true;
}

bool RegisterContextWindows_arm64::GPRWrite(const uint32_t reg,
                                            const RegisterValue &reg_value) {
  auto cleanup = llvm::make_scope_exit([&]() { m_context_arm64 = nullptr; });

  PCONTEXT context = nullptr;
  DataBufferHeap context_buffer;
  DWORD context_flags = CONTEXT_CONTROL | CONTEXT_INTEGER;
  auto thread_handle = GetThreadHandle();

  if (GetThreadContextHelper(thread_handle, context_flags, context,
                             &context_buffer)
          .Fail())
    return false;

  switch (reg) {
  case gpr_x0_arm64:
  case gpr_x1_arm64:
  case gpr_x2_arm64:
  case gpr_x3_arm64:
  case gpr_x4_arm64:
  case gpr_x5_arm64:
  case gpr_x6_arm64:
  case gpr_x7_arm64:
  case gpr_x8_arm64:
  case gpr_x9_arm64:
  case gpr_x10_arm64:
  case gpr_x11_arm64:
  case gpr_x12_arm64:
  case gpr_x13_arm64:
  case gpr_x14_arm64:
  case gpr_x15_arm64:
  case gpr_x16_arm64:
  case gpr_x17_arm64:
  case gpr_x18_arm64:
  case gpr_x19_arm64:
  case gpr_x20_arm64:
  case gpr_x21_arm64:
  case gpr_x22_arm64:
  case gpr_x23_arm64:
  case gpr_x24_arm64:
  case gpr_x25_arm64:
  case gpr_x26_arm64:
  case gpr_x27_arm64:
  case gpr_x28_arm64:
    context->X[reg - gpr_x0_arm64] = reg_value.GetAsUInt64();
    break;

  case gpr_fp_arm64:
    context->Fp = reg_value.GetAsUInt64();
    break;
  case gpr_sp_arm64:
    context->Sp = reg_value.GetAsUInt64();
    break;
  case gpr_lr_arm64:
    context->Lr = reg_value.GetAsUInt64();
    break;
  case gpr_pc_arm64:
    context->Pc = reg_value.GetAsUInt64();
    break;
  case gpr_cpsr_arm64:
    context->Cpsr = reg_value.GetAsUInt32();
    break;

  case gpr_w0_arm64:
  case gpr_w1_arm64:
  case gpr_w2_arm64:
  case gpr_w3_arm64:
  case gpr_w4_arm64:
  case gpr_w5_arm64:
  case gpr_w6_arm64:
  case gpr_w7_arm64:
  case gpr_w8_arm64:
  case gpr_w9_arm64:
  case gpr_w10_arm64:
  case gpr_w11_arm64:
  case gpr_w12_arm64:
  case gpr_w13_arm64:
  case gpr_w14_arm64:
  case gpr_w15_arm64:
  case gpr_w16_arm64:
  case gpr_w17_arm64:
  case gpr_w18_arm64:
  case gpr_w19_arm64:
  case gpr_w20_arm64:
  case gpr_w21_arm64:
  case gpr_w22_arm64:
  case gpr_w23_arm64:
  case gpr_w24_arm64:
  case gpr_w25_arm64:
  case gpr_w26_arm64:
  case gpr_w27_arm64:
  case gpr_w28_arm64:
    context->X[reg - gpr_w0_arm64] = reg_value.GetAsUInt32();
    break;
  }

  return SetThreadContextHelper(thread_handle, context);
}

bool RegisterContextWindows_arm64::FPRRead(const uint32_t reg,
                                           RegisterValue &reg_value) {
  if (!CacheAllRegisterValues())
    return false;

  switch (reg) {
  case fpu_v0_arm64:
  case fpu_v1_arm64:
  case fpu_v2_arm64:
  case fpu_v3_arm64:
  case fpu_v4_arm64:
  case fpu_v5_arm64:
  case fpu_v6_arm64:
  case fpu_v7_arm64:
  case fpu_v8_arm64:
  case fpu_v9_arm64:
  case fpu_v10_arm64:
  case fpu_v11_arm64:
  case fpu_v12_arm64:
  case fpu_v13_arm64:
  case fpu_v14_arm64:
  case fpu_v15_arm64:
  case fpu_v16_arm64:
  case fpu_v17_arm64:
  case fpu_v18_arm64:
  case fpu_v19_arm64:
  case fpu_v20_arm64:
  case fpu_v21_arm64:
  case fpu_v22_arm64:
  case fpu_v23_arm64:
  case fpu_v24_arm64:
  case fpu_v25_arm64:
  case fpu_v26_arm64:
  case fpu_v27_arm64:
  case fpu_v28_arm64:
  case fpu_v29_arm64:
  case fpu_v30_arm64:
  case fpu_v31_arm64: {
    reg_value.SetBytes(m_context_arm64->V[reg - fpu_v0_arm64].B, 16,
                       endian::InlHostByteOrder());
    break;
  }

  case fpu_s0_arm64:
  case fpu_s1_arm64:
  case fpu_s2_arm64:
  case fpu_s3_arm64:
  case fpu_s4_arm64:
  case fpu_s5_arm64:
  case fpu_s6_arm64:
  case fpu_s7_arm64:
  case fpu_s8_arm64:
  case fpu_s9_arm64:
  case fpu_s10_arm64:
  case fpu_s11_arm64:
  case fpu_s12_arm64:
  case fpu_s13_arm64:
  case fpu_s14_arm64:
  case fpu_s15_arm64:
  case fpu_s16_arm64:
  case fpu_s17_arm64:
  case fpu_s18_arm64:
  case fpu_s19_arm64:
  case fpu_s20_arm64:
  case fpu_s21_arm64:
  case fpu_s22_arm64:
  case fpu_s23_arm64:
  case fpu_s24_arm64:
  case fpu_s25_arm64:
  case fpu_s26_arm64:
  case fpu_s27_arm64:
  case fpu_s28_arm64:
  case fpu_s29_arm64:
  case fpu_s30_arm64:
  case fpu_s31_arm64:
    reg_value.SetFloat(m_context_arm64->V[reg - fpu_s0_arm64].S[0]);
    break;

  case fpu_d0_arm64:
  case fpu_d1_arm64:
  case fpu_d2_arm64:
  case fpu_d3_arm64:
  case fpu_d4_arm64:
  case fpu_d5_arm64:
  case fpu_d6_arm64:
  case fpu_d7_arm64:
  case fpu_d8_arm64:
  case fpu_d9_arm64:
  case fpu_d10_arm64:
  case fpu_d11_arm64:
  case fpu_d12_arm64:
  case fpu_d13_arm64:
  case fpu_d14_arm64:
  case fpu_d15_arm64:
  case fpu_d16_arm64:
  case fpu_d17_arm64:
  case fpu_d18_arm64:
  case fpu_d19_arm64:
  case fpu_d20_arm64:
  case fpu_d21_arm64:
  case fpu_d22_arm64:
  case fpu_d23_arm64:
  case fpu_d24_arm64:
  case fpu_d25_arm64:
  case fpu_d26_arm64:
  case fpu_d27_arm64:
  case fpu_d28_arm64:
  case fpu_d29_arm64:
  case fpu_d30_arm64:
  case fpu_d31_arm64:
    reg_value.SetDouble(m_context_arm64->V[reg - fpu_d0_arm64].D[0]);
    break;

  case fpu_fpsr_arm64:
    reg_value.SetUInt32(m_context_arm64->Fpsr);
    break;

  case fpu_fpcr_arm64:
    reg_value.SetUInt32(m_context_arm64->Fpcr);
    break;
  }

  return true;
}

bool RegisterContextWindows_arm64::FPRWrite(const uint32_t reg,
                                            const RegisterValue &reg_value) {
  auto cleanup = llvm::make_scope_exit([&]() { m_context_arm64 = nullptr; });

  PCONTEXT context = nullptr;
  DataBufferHeap context_buffer;
  DWORD context_flags = CONTEXT_CONTROL | CONTEXT_FLOATING_POINT;
  auto thread_handle = GetThreadHandle();

  if (GetThreadContextHelper(thread_handle, context_flags, context,
                             &context_buffer)
          .Fail())
    return false;

  switch (reg) {
  case fpu_v0_arm64:
  case fpu_v1_arm64:
  case fpu_v2_arm64:
  case fpu_v3_arm64:
  case fpu_v4_arm64:
  case fpu_v5_arm64:
  case fpu_v6_arm64:
  case fpu_v7_arm64:
  case fpu_v8_arm64:
  case fpu_v9_arm64:
  case fpu_v10_arm64:
  case fpu_v11_arm64:
  case fpu_v12_arm64:
  case fpu_v13_arm64:
  case fpu_v14_arm64:
  case fpu_v15_arm64:
  case fpu_v16_arm64:
  case fpu_v17_arm64:
  case fpu_v18_arm64:
  case fpu_v19_arm64:
  case fpu_v20_arm64:
  case fpu_v21_arm64:
  case fpu_v22_arm64:
  case fpu_v23_arm64:
  case fpu_v24_arm64:
  case fpu_v25_arm64:
  case fpu_v26_arm64:
  case fpu_v27_arm64:
  case fpu_v28_arm64:
  case fpu_v29_arm64:
  case fpu_v30_arm64:
  case fpu_v31_arm64:
    memcpy(context->V[reg - fpu_v0_arm64].B, reg_value.GetBytes(), 16);
    break;

  case fpu_s0_arm64:
  case fpu_s1_arm64:
  case fpu_s2_arm64:
  case fpu_s3_arm64:
  case fpu_s4_arm64:
  case fpu_s5_arm64:
  case fpu_s6_arm64:
  case fpu_s7_arm64:
  case fpu_s8_arm64:
  case fpu_s9_arm64:
  case fpu_s10_arm64:
  case fpu_s11_arm64:
  case fpu_s12_arm64:
  case fpu_s13_arm64:
  case fpu_s14_arm64:
  case fpu_s15_arm64:
  case fpu_s16_arm64:
  case fpu_s17_arm64:
  case fpu_s18_arm64:
  case fpu_s19_arm64:
  case fpu_s20_arm64:
  case fpu_s21_arm64:
  case fpu_s22_arm64:
  case fpu_s23_arm64:
  case fpu_s24_arm64:
  case fpu_s25_arm64:
  case fpu_s26_arm64:
  case fpu_s27_arm64:
  case fpu_s28_arm64:
  case fpu_s29_arm64:
  case fpu_s30_arm64:
  case fpu_s31_arm64:
    context->V[reg - fpu_s0_arm64].S[0] = reg_value.GetAsFloat();
    break;

  case fpu_d0_arm64:
  case fpu_d1_arm64:
  case fpu_d2_arm64:
  case fpu_d3_arm64:
  case fpu_d4_arm64:
  case fpu_d5_arm64:
  case fpu_d6_arm64:
  case fpu_d7_arm64:
  case fpu_d8_arm64:
  case fpu_d9_arm64:
  case fpu_d10_arm64:
  case fpu_d11_arm64:
  case fpu_d12_arm64:
  case fpu_d13_arm64:
  case fpu_d14_arm64:
  case fpu_d15_arm64:
  case fpu_d16_arm64:
  case fpu_d17_arm64:
  case fpu_d18_arm64:
  case fpu_d19_arm64:
  case fpu_d20_arm64:
  case fpu_d21_arm64:
  case fpu_d22_arm64:
  case fpu_d23_arm64:
  case fpu_d24_arm64:
  case fpu_d25_arm64:
  case fpu_d26_arm64:
  case fpu_d27_arm64:
  case fpu_d28_arm64:
  case fpu_d29_arm64:
  case fpu_d30_arm64:
  case fpu_d31_arm64:
    context->V[reg - fpu_d0_arm64].D[0] = reg_value.GetAsDouble();
    break;

  case fpu_fpsr_arm64:
    context->Fpsr = reg_value.GetAsUInt32();
    break;

  case fpu_fpcr_arm64:
    context->Fpcr = reg_value.GetAsUInt32();
    break;
  }

  return SetThreadContextHelper(thread_handle, context);
}

bool RegisterContextWindows_arm64::IsGPR(uint32_t reg) const {
  return (reg >= k_first_gpr_arm64 && reg <= k_last_gpr_arm64);
}

bool RegisterContextWindows_arm64::IsFPR(uint32_t reg) const {
  return (reg >= k_first_fpr_arm64 && reg <= k_last_fpr_arm64);
}

#endif // defined(__aarch64__) || defined(_M_ARM64)

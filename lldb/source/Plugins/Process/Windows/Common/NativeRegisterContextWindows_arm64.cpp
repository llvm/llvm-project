//===-- NativeRegisterContextWindows_arm64.cpp ----------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#if defined(__aarch64__) || defined(_M_ARM64)

#include <set>
#include "lldb/Register/Arch/lldb-arm64-register-enums.h"

#include "NativeRegisterContextWindows_arm64.h"
#include "NativeThreadWindows.h"
#include "ProcessWindowsLog.h"
#include "lldb/Host/HostInfo.h"
#include "lldb/Host/HostThread.h"
#include "lldb/Host/windows/HostThreadWindows.h"

#include "lldb/Utility/Log.h"
#include "lldb/Register/RegisterValue.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/ScopeExit.h"

namespace reg_ids = lldb_private::registers::arm64::id;
using namespace lldb;
using namespace lldb_private;

#define REG_CONTEXT_SIZE sizeof(::CONTEXT)

namespace {
static const uint32_t g_gpr_regnums_arm64[] = {
    reg_ids::gpr_x0,    reg_ids::gpr_x1,   reg_ids::gpr_x2,  reg_ids::gpr_x3,
    reg_ids::gpr_x4,    reg_ids::gpr_x5,   reg_ids::gpr_x6,  reg_ids::gpr_x7,
    reg_ids::gpr_x8,    reg_ids::gpr_x9,   reg_ids::gpr_x10, reg_ids::gpr_x11,
    reg_ids::gpr_x12,   reg_ids::gpr_x13,  reg_ids::gpr_x14, reg_ids::gpr_x15,
    reg_ids::gpr_x16,   reg_ids::gpr_x17,  reg_ids::gpr_x18, reg_ids::gpr_x19,
    reg_ids::gpr_x20,   reg_ids::gpr_x21,  reg_ids::gpr_x22, reg_ids::gpr_x23,
    reg_ids::gpr_x24,   reg_ids::gpr_x25,  reg_ids::gpr_x26, reg_ids::gpr_x27,
    reg_ids::gpr_x28,   reg_ids::gpr_fp,   reg_ids::gpr_lr,  reg_ids::gpr_sp,
    reg_ids::gpr_pc,    reg_ids::gpr_cpsr, reg_ids::gpr_w0,  reg_ids::gpr_w1,
    reg_ids::gpr_w2,    reg_ids::gpr_w3,   reg_ids::gpr_w4,  reg_ids::gpr_w5,
    reg_ids::gpr_w6,    reg_ids::gpr_w7,   reg_ids::gpr_w8,  reg_ids::gpr_w9,
    reg_ids::gpr_w10,   reg_ids::gpr_w11,  reg_ids::gpr_w12, reg_ids::gpr_w13,
    reg_ids::gpr_w14,   reg_ids::gpr_w15,  reg_ids::gpr_w16, reg_ids::gpr_w17,
    reg_ids::gpr_w18,   reg_ids::gpr_w19,  reg_ids::gpr_w20, reg_ids::gpr_w21,
    reg_ids::gpr_w22,   reg_ids::gpr_w23,  reg_ids::gpr_w24, reg_ids::gpr_w25,
    reg_ids::gpr_w26,   reg_ids::gpr_w27,  reg_ids::gpr_w28,
    LLDB_INVALID_REGNUM // Register set must be terminated with this flag
};
static_assert(((sizeof g_gpr_regnums_arm64 / sizeof g_gpr_regnums_arm64[0]) -
               1) == reg_ids::k_num_gpr_registers,
              "g_gpr_regnums_arm64 has wrong number of register infos");

static const uint32_t g_fpr_regnums_arm64[] = {
    reg_ids::fpu_v0,    reg_ids::fpu_v1,   reg_ids::fpu_v2,  reg_ids::fpu_v3,
    reg_ids::fpu_v4,    reg_ids::fpu_v5,   reg_ids::fpu_v6,  reg_ids::fpu_v7,
    reg_ids::fpu_v8,    reg_ids::fpu_v9,   reg_ids::fpu_v10, reg_ids::fpu_v11,
    reg_ids::fpu_v12,   reg_ids::fpu_v13,  reg_ids::fpu_v14, reg_ids::fpu_v15,
    reg_ids::fpu_v16,   reg_ids::fpu_v17,  reg_ids::fpu_v18, reg_ids::fpu_v19,
    reg_ids::fpu_v20,   reg_ids::fpu_v21,  reg_ids::fpu_v22, reg_ids::fpu_v23,
    reg_ids::fpu_v24,   reg_ids::fpu_v25,  reg_ids::fpu_v26, reg_ids::fpu_v27,
    reg_ids::fpu_v28,   reg_ids::fpu_v29,  reg_ids::fpu_v30, reg_ids::fpu_v31,
    reg_ids::fpu_s0,    reg_ids::fpu_s1,   reg_ids::fpu_s2,  reg_ids::fpu_s3,
    reg_ids::fpu_s4,    reg_ids::fpu_s5,   reg_ids::fpu_s6,  reg_ids::fpu_s7,
    reg_ids::fpu_s8,    reg_ids::fpu_s9,   reg_ids::fpu_s10, reg_ids::fpu_s11,
    reg_ids::fpu_s12,   reg_ids::fpu_s13,  reg_ids::fpu_s14, reg_ids::fpu_s15,
    reg_ids::fpu_s16,   reg_ids::fpu_s17,  reg_ids::fpu_s18, reg_ids::fpu_s19,
    reg_ids::fpu_s20,   reg_ids::fpu_s21,  reg_ids::fpu_s22, reg_ids::fpu_s23,
    reg_ids::fpu_s24,   reg_ids::fpu_s25,  reg_ids::fpu_s26, reg_ids::fpu_s27,
    reg_ids::fpu_s28,   reg_ids::fpu_s29,  reg_ids::fpu_s30, reg_ids::fpu_s31,

    reg_ids::fpu_d0,    reg_ids::fpu_d1,   reg_ids::fpu_d2,  reg_ids::fpu_d3,
    reg_ids::fpu_d4,    reg_ids::fpu_d5,   reg_ids::fpu_d6,  reg_ids::fpu_d7,
    reg_ids::fpu_d8,    reg_ids::fpu_d9,   reg_ids::fpu_d10, reg_ids::fpu_d11,
    reg_ids::fpu_d12,   reg_ids::fpu_d13,  reg_ids::fpu_d14, reg_ids::fpu_d15,
    reg_ids::fpu_d16,   reg_ids::fpu_d17,  reg_ids::fpu_d18, reg_ids::fpu_d19,
    reg_ids::fpu_d20,   reg_ids::fpu_d21,  reg_ids::fpu_d22, reg_ids::fpu_d23,
    reg_ids::fpu_d24,   reg_ids::fpu_d25,  reg_ids::fpu_d26, reg_ids::fpu_d27,
    reg_ids::fpu_d28,   reg_ids::fpu_d29,  reg_ids::fpu_d30, reg_ids::fpu_d31,
    reg_ids::fpu_fpsr,  reg_ids::fpu_fpcr,
    LLDB_INVALID_REGNUM // Register set must be terminated with this flag
};
static_assert(((sizeof g_fpr_regnums_arm64 / sizeof g_fpr_regnums_arm64[0]) -
               1) == reg_ids::k_num_fpr_registers,
              "g_fpu_regnums_arm64 has wrong number of register infos");

struct RegisterGroupNumbers {
  const char *name;
  const char *short_name;
  std::vector<uint32_t> registers;
};

static const RegisterGroupNumbers g_reg_sets_arm64[] = {
    {"General Purpose Registers", "gpr",
     std::vector<RegisterID>(g_gpr_regnums_arm64,
                             g_gpr_regnums_arm64 +
                                 std::size(g_gpr_regnums_arm64) - 1)},
    {"Floating Point Registers", "fpr",
     std::vector<RegisterID>(g_fpr_regnums_arm64,
                             g_fpr_regnums_arm64 +
                                 std::size(g_fpr_regnums_arm64) - 1)}};

enum { k_num_register_sets = 2 };

} // namespace

static RegisterLayout CreateRegisterLayout(const ArchSpec &target_arch) {
  assert((HostInfo::GetArchitecture().GetAddressByteSize() == 8) &&
         "Register setting path assumes this is a 64-bit host");
  return llvm::cantFail(
      RegisterLayout::Create(RegisterInfo_arm64::CreateDescription(
                                 RegisterInfo_arm64::eRegsetMaskDefault),
                             RegisterLayout(target_arch.GetByteOrder())));
}

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

  return Status();
}

static Status SetThreadContextHelper(lldb::thread_t thread_handle,
                                     PCONTEXT context) {
  Log *log = GetLog(WindowsLog::Registers);
  Status error;
  if (!::SetThreadContext(thread_handle, context)) {
    error = Status(GetLastError(), eErrorTypeWin32);
    LLDB_LOG(log, "SetThreadContext failed with error {0}", error);
    return error;
  }
  return error;
}

std::unique_ptr<NativeRegisterContextWindows>
NativeRegisterContextWindows::CreateHostNativeRegisterContextWindows(
    const ArchSpec &target_arch, NativeThreadProtocol &native_thread) {
  // Register context for a native 64-bit application.
  return std::make_unique<NativeRegisterContextWindows_arm64>(target_arch,
                                                              native_thread);
}

NativeRegisterContextWindows_arm64::NativeRegisterContextWindows_arm64(
    const ArchSpec &target_arch, NativeThreadProtocol &native_thread)
    : NativeRegisterContext(native_thread, target_arch,
                            CreateRegisterLayout(target_arch)),
      m_context(nullptr), m_context_buffer(nullptr) {
  // Currently, there is no API to query the maximum supported hardware
  // breakpoints and watchpoints on Windows. The values set below are based
  // on tests conducted on Windows 11 with Snapdragon Elite X hardware.
  m_max_hwp_supported = 1;

  // Bind native presentation policy to this schema generation once. The
  // private definitions above use the backend's external numbering.
  auto select_group = [&](uint32_t set_index) -> const RegisterGroupNumbers * {
    if (set_index >= k_num_register_sets)
      return nullptr;
    return &g_reg_sets_arm64[set_index];
  };
  auto previous = GetDescription();
  std::vector<RegisterGroup> groups;
  std::set<RegisterID> visible;
  for (uint32_t index = 0; index < k_num_register_sets; ++index) {
    const auto *input = select_group(index);
    if (!input)
      continue;
    RegisterGroup group{input->name, input->short_name, {}};
    for (uint32_t number : input->registers) {
      const auto *reg = previous->FindByNumber(lldb::eRegisterKindLLDB, number);
      assert(reg && "native presentation names an unknown register");
      if (reg) {
        group.registers.push_back(reg->id);
        visible.insert(reg->id);
      }
    }
    groups.push_back(std::move(group));
  }
  auto description = llvm::cantFail(RegisterDescription::Create(
      {previous->GetRegisters().begin(), previous->GetRegisters().end()},
      std::move(groups),
      {previous->GetOwnedTypes().begin(), previous->GetOwnedTypes().end()}));
  std::vector<RegisterID> order;
  for (RegisterID id : previous->GetUserRegisters())
    if (visible.count(id))
      order.push_back(id);
  auto presentation = llvm::cantFail(description.WithUserRegisters(order));
  SetRegisterLayout(llvm::cantFail(RegisterLayout::Create(
      std::make_shared<RegisterDescription>(std::move(presentation)),
      GetLayout())));
}

bool NativeRegisterContextWindows_arm64::IsGPR(uint32_t reg_index) const {
  return (reg_index >= reg_ids::k_first_gpr &&
          reg_index <= reg_ids::k_last_gpr);
}

bool NativeRegisterContextWindows_arm64::IsFPR(uint32_t reg_index) const {
  return (reg_index >= reg_ids::k_first_fpr &&
          reg_index <= reg_ids::k_last_fpr);
}

Status NativeRegisterContextWindows_arm64::GPRRead(const uint32_t reg,
                                                   RegisterValue &reg_value) {
  Status error = CacheAllRegisterValues();
  if (error.Fail())
    return error;

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
    reg_value.SetUInt64(m_context->X[reg - reg_ids::gpr_x0]);
    break;

  case reg_ids::gpr_fp:
    reg_value.SetUInt64(m_context->Fp);
    break;
  case reg_ids::gpr_sp:
    reg_value.SetUInt64(m_context->Sp);
    break;
  case reg_ids::gpr_lr:
    reg_value.SetUInt64(m_context->Lr);
    break;
  case reg_ids::gpr_pc:
    reg_value.SetUInt64(m_context->Pc);
    break;
  case reg_ids::gpr_cpsr:
    reg_value.SetUInt32(m_context->Cpsr);
    break;

  case reg_ids::gpr_w0:
  case reg_ids::gpr_w1:
  case reg_ids::gpr_w2:
  case reg_ids::gpr_w3:
  case reg_ids::gpr_w4:
  case reg_ids::gpr_w5:
  case reg_ids::gpr_w6:
  case reg_ids::gpr_w7:
  case reg_ids::gpr_w8:
  case reg_ids::gpr_w9:
  case reg_ids::gpr_w10:
  case reg_ids::gpr_w11:
  case reg_ids::gpr_w12:
  case reg_ids::gpr_w13:
  case reg_ids::gpr_w14:
  case reg_ids::gpr_w15:
  case reg_ids::gpr_w16:
  case reg_ids::gpr_w17:
  case reg_ids::gpr_w18:
  case reg_ids::gpr_w19:
  case reg_ids::gpr_w20:
  case reg_ids::gpr_w21:
  case reg_ids::gpr_w22:
  case reg_ids::gpr_w23:
  case reg_ids::gpr_w24:
  case reg_ids::gpr_w25:
  case reg_ids::gpr_w26:
  case reg_ids::gpr_w27:
  case reg_ids::gpr_w28:
    reg_value.SetUInt32(
        static_cast<uint32_t>(m_context->X[reg - reg_ids::gpr_w0] & 0xffffffff));
    break;
  }

  return error;
}

Status
NativeRegisterContextWindows_arm64::GPRWrite(const uint32_t reg,
                                             const RegisterValue &reg_value) {
  auto cleanup = llvm::scope_exit([&]() { m_context = nullptr; });

  PCONTEXT context = nullptr;
  DataBufferHeap context_buffer;
  DWORD context_flags = CONTEXT_CONTROL | CONTEXT_INTEGER;
  auto thread_handle = GetThreadHandle();

  Status error = GetThreadContextHelper(thread_handle, context_flags, context,
                                        &context_buffer);
  if (error.Fail())
    return error;

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
    context->X[reg - reg_ids::gpr_x0] = reg_value.GetAsUInt64();
    break;

  case reg_ids::gpr_fp:
    context->Fp = reg_value.GetAsUInt64();
    break;
  case reg_ids::gpr_sp:
    context->Sp = reg_value.GetAsUInt64();
    break;
  case reg_ids::gpr_lr:
    context->Lr = reg_value.GetAsUInt64();
    break;
  case reg_ids::gpr_pc:
    context->Pc = reg_value.GetAsUInt64();
    break;
  case reg_ids::gpr_cpsr:
    context->Cpsr = reg_value.GetAsUInt32();
    break;

  case reg_ids::gpr_w0:
  case reg_ids::gpr_w1:
  case reg_ids::gpr_w2:
  case reg_ids::gpr_w3:
  case reg_ids::gpr_w4:
  case reg_ids::gpr_w5:
  case reg_ids::gpr_w6:
  case reg_ids::gpr_w7:
  case reg_ids::gpr_w8:
  case reg_ids::gpr_w9:
  case reg_ids::gpr_w10:
  case reg_ids::gpr_w11:
  case reg_ids::gpr_w12:
  case reg_ids::gpr_w13:
  case reg_ids::gpr_w14:
  case reg_ids::gpr_w15:
  case reg_ids::gpr_w16:
  case reg_ids::gpr_w17:
  case reg_ids::gpr_w18:
  case reg_ids::gpr_w19:
  case reg_ids::gpr_w20:
  case reg_ids::gpr_w21:
  case reg_ids::gpr_w22:
  case reg_ids::gpr_w23:
  case reg_ids::gpr_w24:
  case reg_ids::gpr_w25:
  case reg_ids::gpr_w26:
  case reg_ids::gpr_w27:
  case reg_ids::gpr_w28:
    context->X[reg - reg_ids::gpr_w0] = reg_value.GetAsUInt32();
    break;
  }

  return SetThreadContextHelper(thread_handle, context);
}

Status NativeRegisterContextWindows_arm64::FPRRead(const uint32_t reg,
                                                   RegisterValue &reg_value) {
  Status error = CacheAllRegisterValues();
  if (error.Fail())
    return error;

  switch (reg) {
  case reg_ids::fpu_v0:
  case reg_ids::fpu_v1:
  case reg_ids::fpu_v2:
  case reg_ids::fpu_v3:
  case reg_ids::fpu_v4:
  case reg_ids::fpu_v5:
  case reg_ids::fpu_v6:
  case reg_ids::fpu_v7:
  case reg_ids::fpu_v8:
  case reg_ids::fpu_v9:
  case reg_ids::fpu_v10:
  case reg_ids::fpu_v11:
  case reg_ids::fpu_v12:
  case reg_ids::fpu_v13:
  case reg_ids::fpu_v14:
  case reg_ids::fpu_v15:
  case reg_ids::fpu_v16:
  case reg_ids::fpu_v17:
  case reg_ids::fpu_v18:
  case reg_ids::fpu_v19:
  case reg_ids::fpu_v20:
  case reg_ids::fpu_v21:
  case reg_ids::fpu_v22:
  case reg_ids::fpu_v23:
  case reg_ids::fpu_v24:
  case reg_ids::fpu_v25:
  case reg_ids::fpu_v26:
  case reg_ids::fpu_v27:
  case reg_ids::fpu_v28:
  case reg_ids::fpu_v29:
  case reg_ids::fpu_v30:
  case reg_ids::fpu_v31:
    reg_value.SetBytes(m_context->V[reg - reg_ids::fpu_v0].B, 16,
                       endian::InlHostByteOrder());
    break;

  case reg_ids::fpu_s0:
  case reg_ids::fpu_s1:
  case reg_ids::fpu_s2:
  case reg_ids::fpu_s3:
  case reg_ids::fpu_s4:
  case reg_ids::fpu_s5:
  case reg_ids::fpu_s6:
  case reg_ids::fpu_s7:
  case reg_ids::fpu_s8:
  case reg_ids::fpu_s9:
  case reg_ids::fpu_s10:
  case reg_ids::fpu_s11:
  case reg_ids::fpu_s12:
  case reg_ids::fpu_s13:
  case reg_ids::fpu_s14:
  case reg_ids::fpu_s15:
  case reg_ids::fpu_s16:
  case reg_ids::fpu_s17:
  case reg_ids::fpu_s18:
  case reg_ids::fpu_s19:
  case reg_ids::fpu_s20:
  case reg_ids::fpu_s21:
  case reg_ids::fpu_s22:
  case reg_ids::fpu_s23:
  case reg_ids::fpu_s24:
  case reg_ids::fpu_s25:
  case reg_ids::fpu_s26:
  case reg_ids::fpu_s27:
  case reg_ids::fpu_s28:
  case reg_ids::fpu_s29:
  case reg_ids::fpu_s30:
  case reg_ids::fpu_s31:
    reg_value.SetFloat(m_context->V[reg - reg_ids::fpu_s0].S[0]);
    break;

  case reg_ids::fpu_d0:
  case reg_ids::fpu_d1:
  case reg_ids::fpu_d2:
  case reg_ids::fpu_d3:
  case reg_ids::fpu_d4:
  case reg_ids::fpu_d5:
  case reg_ids::fpu_d6:
  case reg_ids::fpu_d7:
  case reg_ids::fpu_d8:
  case reg_ids::fpu_d9:
  case reg_ids::fpu_d10:
  case reg_ids::fpu_d11:
  case reg_ids::fpu_d12:
  case reg_ids::fpu_d13:
  case reg_ids::fpu_d14:
  case reg_ids::fpu_d15:
  case reg_ids::fpu_d16:
  case reg_ids::fpu_d17:
  case reg_ids::fpu_d18:
  case reg_ids::fpu_d19:
  case reg_ids::fpu_d20:
  case reg_ids::fpu_d21:
  case reg_ids::fpu_d22:
  case reg_ids::fpu_d23:
  case reg_ids::fpu_d24:
  case reg_ids::fpu_d25:
  case reg_ids::fpu_d26:
  case reg_ids::fpu_d27:
  case reg_ids::fpu_d28:
  case reg_ids::fpu_d29:
  case reg_ids::fpu_d30:
  case reg_ids::fpu_d31:
    reg_value.SetDouble(m_context->V[reg - reg_ids::fpu_d0].D[0]);
    break;

  case reg_ids::fpu_fpsr:
    reg_value.SetUInt32(m_context->Fpsr);
    break;

  case reg_ids::fpu_fpcr:
    reg_value.SetUInt32(m_context->Fpcr);
    break;
  }

  return error;
}

Status
NativeRegisterContextWindows_arm64::FPRWrite(const uint32_t reg,
                                             const RegisterValue &reg_value) {
  auto cleanup = llvm::scope_exit([&]() { m_context = nullptr; });

  PCONTEXT context = nullptr;
  DataBufferHeap context_buffer;
  DWORD context_flags = CONTEXT_CONTROL | CONTEXT_FLOATING_POINT;
  auto thread_handle = GetThreadHandle();

  Status error = GetThreadContextHelper(thread_handle, context_flags, context,
                                        &context_buffer);
  if (error.Fail())
    return error;

  switch (reg) {
  case reg_ids::fpu_v0:
  case reg_ids::fpu_v1:
  case reg_ids::fpu_v2:
  case reg_ids::fpu_v3:
  case reg_ids::fpu_v4:
  case reg_ids::fpu_v5:
  case reg_ids::fpu_v6:
  case reg_ids::fpu_v7:
  case reg_ids::fpu_v8:
  case reg_ids::fpu_v9:
  case reg_ids::fpu_v10:
  case reg_ids::fpu_v11:
  case reg_ids::fpu_v12:
  case reg_ids::fpu_v13:
  case reg_ids::fpu_v14:
  case reg_ids::fpu_v15:
  case reg_ids::fpu_v16:
  case reg_ids::fpu_v17:
  case reg_ids::fpu_v18:
  case reg_ids::fpu_v19:
  case reg_ids::fpu_v20:
  case reg_ids::fpu_v21:
  case reg_ids::fpu_v22:
  case reg_ids::fpu_v23:
  case reg_ids::fpu_v24:
  case reg_ids::fpu_v25:
  case reg_ids::fpu_v26:
  case reg_ids::fpu_v27:
  case reg_ids::fpu_v28:
  case reg_ids::fpu_v29:
  case reg_ids::fpu_v30:
  case reg_ids::fpu_v31:
    memcpy(context->V[reg - reg_ids::fpu_v0].B, reg_value.GetBytes(), 16);
    break;

  case reg_ids::fpu_s0:
  case reg_ids::fpu_s1:
  case reg_ids::fpu_s2:
  case reg_ids::fpu_s3:
  case reg_ids::fpu_s4:
  case reg_ids::fpu_s5:
  case reg_ids::fpu_s6:
  case reg_ids::fpu_s7:
  case reg_ids::fpu_s8:
  case reg_ids::fpu_s9:
  case reg_ids::fpu_s10:
  case reg_ids::fpu_s11:
  case reg_ids::fpu_s12:
  case reg_ids::fpu_s13:
  case reg_ids::fpu_s14:
  case reg_ids::fpu_s15:
  case reg_ids::fpu_s16:
  case reg_ids::fpu_s17:
  case reg_ids::fpu_s18:
  case reg_ids::fpu_s19:
  case reg_ids::fpu_s20:
  case reg_ids::fpu_s21:
  case reg_ids::fpu_s22:
  case reg_ids::fpu_s23:
  case reg_ids::fpu_s24:
  case reg_ids::fpu_s25:
  case reg_ids::fpu_s26:
  case reg_ids::fpu_s27:
  case reg_ids::fpu_s28:
  case reg_ids::fpu_s29:
  case reg_ids::fpu_s30:
  case reg_ids::fpu_s31:
    context->V[reg - reg_ids::fpu_s0].S[0] = reg_value.GetAsFloat();
    break;

  case reg_ids::fpu_d0:
  case reg_ids::fpu_d1:
  case reg_ids::fpu_d2:
  case reg_ids::fpu_d3:
  case reg_ids::fpu_d4:
  case reg_ids::fpu_d5:
  case reg_ids::fpu_d6:
  case reg_ids::fpu_d7:
  case reg_ids::fpu_d8:
  case reg_ids::fpu_d9:
  case reg_ids::fpu_d10:
  case reg_ids::fpu_d11:
  case reg_ids::fpu_d12:
  case reg_ids::fpu_d13:
  case reg_ids::fpu_d14:
  case reg_ids::fpu_d15:
  case reg_ids::fpu_d16:
  case reg_ids::fpu_d17:
  case reg_ids::fpu_d18:
  case reg_ids::fpu_d19:
  case reg_ids::fpu_d20:
  case reg_ids::fpu_d21:
  case reg_ids::fpu_d22:
  case reg_ids::fpu_d23:
  case reg_ids::fpu_d24:
  case reg_ids::fpu_d25:
  case reg_ids::fpu_d26:
  case reg_ids::fpu_d27:
  case reg_ids::fpu_d28:
  case reg_ids::fpu_d29:
  case reg_ids::fpu_d30:
  case reg_ids::fpu_d31:
    context->V[reg - reg_ids::fpu_d0].D[0] = reg_value.GetAsDouble();
    break;

  case reg_ids::fpu_fpsr:
    context->Fpsr = reg_value.GetAsUInt32();
    break;

  case reg_ids::fpu_fpcr:
    context->Fpcr = reg_value.GetAsUInt32();
    break;
  }

  return SetThreadContextHelper(thread_handle, context);
}

llvm::Expected<RegisterValue>
NativeRegisterContextWindows_arm64::ReadRegister(RegisterID id) {
  RegisterValue reg_value;
  auto description = GetDescription();
  const auto *descriptor = description ? description->Find(id) : nullptr;
  Status error;
  if (!descriptor) {
    error = Status::FromErrorString("unknown register");
    return error.ToError();
  }

  const uint32_t reg = descriptor->numbers[lldb::eRegisterKindLLDB];
  if (reg == LLDB_INVALID_REGNUM) {
    // This is likely an internal register for lldb use only and should not be
    // directly queried.
    error = Status::FromErrorStringWithFormat(
        "register \"%s\" is an internal-only lldb "
        "register, cannot read directly",
        descriptor->name.c_str());
    return error.ToError();
  }

  if (IsGPR(reg)) {
    if (Status status = GPRRead(reg, reg_value); status.Fail())
      return status.ToError();
    return reg_value;
  }

  if (IsFPR(reg)) {
    if (Status status = FPRRead(reg, reg_value); status.Fail())
      return status.ToError();
    return reg_value;
  }

  return Status::FromErrorString("unimplemented").ToError();
}

llvm::Error NativeRegisterContextWindows_arm64::WriteRegister(
    RegisterID id, const RegisterValue &reg_value) {
  auto description = GetDescription();
  const auto *descriptor = description ? description->Find(id) : nullptr;
  Status error;

  if (!descriptor) {
    error = Status::FromErrorString("unknown register");
    return error.ToError();
  }

  const uint32_t reg = descriptor->numbers[lldb::eRegisterKindLLDB];
  if (reg == LLDB_INVALID_REGNUM) {
    // This is likely an internal register for lldb use only and should not be
    // directly written.
    error = Status::FromErrorStringWithFormat(
        "register \"%s\" is an internal-only lldb "
        "register, cannot write directly",
        descriptor->name.c_str());
    return error.ToError();
  }

  if (IsGPR(reg))
    return GPRWrite(reg, reg_value).ToError();

  if (IsFPR(reg))
    return FPRWrite(reg, reg_value).ToError();

  return Status::FromErrorString("unimplemented").ToError();
}

Status NativeRegisterContextWindows_arm64::ReadAllRegisterValues(
    lldb::WritableDataBufferSP &data_sp) {
  Log *log = GetLog(WindowsLog::Registers);

  Status error = CacheAllRegisterValues();
  if (error.Fail())
    return error;

  if (!m_context_buffer) {
    error = Status::FromErrorString("register context buffer is not available");
    LLDB_LOG(log, "{0}", error);
    return error;
  }

  data_sp = std::make_shared<DataBufferHeap>(m_context_buffer->GetBytes(),
                                             m_context_buffer->GetByteSize());

  return error;
}

Status NativeRegisterContextWindows_arm64::WriteAllRegisterValues(
    const lldb::DataBufferSP &data_sp) {
  auto cleanup = llvm::scope_exit([&]() { m_context = nullptr; });

  Log *log = GetLog(WindowsLog::Registers);
  Status error;

  if (!data_sp) {
    error = Status::FromErrorString("invalid data_sp");
    LLDB_LOG(log, "{0}", error);
    return error;
  }

  DWORD context_flags = CONTEXT_ALL;
  DWORD context_length = 0;

  error = GetThreadContextLength(context_flags, context_length);
  if (error.Fail())
    return error;

  if (data_sp->GetByteSize() != context_length) {
    error = Status::FromErrorStringWithFormatv(
        "data_sp contained mismatched data size, expected {0}, actual {1}",
        context_length, data_sp->GetByteSize());
    LLDB_LOG(log, "{0}", error);
    return error;
  }

  PCONTEXT context = nullptr;
  DataBufferHeap context_buffer;
  error = GetThreadContextHelper(GetThreadHandle(), context_flags, context,
                                 &context_buffer);
  if (error.Fail())
    return error;

  ::memcpy(context_buffer.GetBytes(), data_sp->GetBytes(), context_length);

  return SetThreadContextHelper(GetThreadHandle(), context);
}

llvm::Error NativeRegisterContextWindows_arm64::ReadHardwareDebugInfo() {
  Status error = CacheAllRegisterValues();
  if (error.Fail())
    return error.ToError();

  for (uint32_t i = 0; i < m_max_hwp_supported; i++) {
    m_hwp_regs[i].address = m_context->Wvr[i];
    m_hwp_regs[i].control = m_context->Wcr[i];
  }

  return llvm::Error::success();
}

llvm::Error
NativeRegisterContextWindows_arm64::WriteHardwareDebugRegs(DREGType hwbType) {
  auto cleanup = llvm::scope_exit([&]() { m_context = nullptr; });

  PCONTEXT context = nullptr;
  DataBufferHeap context_buffer;
  DWORD context_flags = CONTEXT_DEBUG_REGISTERS;
  auto thread_handle = GetThreadHandle();

  Status error = GetThreadContextHelper(thread_handle, context_flags, context,
                                        &context_buffer);
  if (error.Fail())
    return error.ToError();

  if (hwbType == eDREGTypeWATCH) {
    for (uint32_t i = 0; i < m_max_hwp_supported; i++) {
      context->Wvr[i] = m_hwp_regs[i].address;
      context->Wcr[i] = m_hwp_regs[i].control;
    }
  }

  return SetThreadContextHelper(GetThreadHandle(), context).ToError();
}

void NativeRegisterContextWindows_arm64::InvalidateAllRegisters() {
  m_context = nullptr;
  m_context_buffer.reset();
}

Status NativeRegisterContextWindows_arm64::CacheAllRegisterValues() {
  Status error;
  DWORD context_flags = CONTEXT_ALL;

  if (m_context && (m_context->ContextFlags & context_flags) == context_flags)
    return error;

  m_context = nullptr;

  if (!m_context_buffer)
    m_context_buffer = std::make_shared<DataBufferHeap>();

  error = GetThreadContextHelper(GetThreadHandle(), context_flags, m_context,
                                 m_context_buffer.get());

  if (error.Fail())
    m_context = nullptr;

  return error;
}

#endif // defined(__aarch64__) || defined(_M_ARM64)

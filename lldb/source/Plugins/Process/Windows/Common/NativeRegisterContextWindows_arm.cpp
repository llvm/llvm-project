//===-- NativeRegisterContextWindows_arm.cpp ------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#if defined(__arm__) || defined(_M_ARM)

#include <set>
#include "lldb/Register/Arch/lldb-arm-register-enums.h"

#include "NativeRegisterContextWindows_arm.h"
#include "NativeThreadWindows.h"
#include "ProcessWindowsLog.h"
#include "lldb/Host/HostInfo.h"
#include "lldb/Host/HostThread.h"
#include "lldb/Host/windows/HostThreadWindows.h"
#include "lldb/Host/windows/windows.h"
#include "lldb/Register/Common/RegisterInfo_arm.h"

#include "lldb/Utility/Log.h"
#include "lldb/Register/RegisterValue.h"
#include "llvm/ADT/STLExtras.h"

namespace reg_ids = lldb_private::registers::arm::id;
using namespace lldb;
using namespace lldb_private;

#define REG_CONTEXT_SIZE sizeof(::CONTEXT)

namespace {
static const uint32_t g_gpr_regnums_arm[] = {
    reg_ids::gpr_r0,    reg_ids::gpr_r1, reg_ids::gpr_r2,  reg_ids::gpr_r3,
    reg_ids::gpr_r4,    reg_ids::gpr_r5, reg_ids::gpr_r6,  reg_ids::gpr_r7,
    reg_ids::gpr_r8,    reg_ids::gpr_r9, reg_ids::gpr_r10, reg_ids::gpr_r11,
    reg_ids::gpr_r12,   reg_ids::gpr_sp, reg_ids::gpr_lr,  reg_ids::gpr_pc,
    reg_ids::gpr_cpsr,
    LLDB_INVALID_REGNUM // Register set must be terminated with this flag
};
static_assert(((sizeof g_gpr_regnums_arm / sizeof g_gpr_regnums_arm[0]) - 1) ==
                  reg_ids::k_num_gpr_registers,
              "g_gpr_regnums_arm has wrong number of register infos");

static const uint32_t g_fpr_regnums_arm[] = {
    reg_ids::fpu_s0,    reg_ids::fpu_s1,  reg_ids::fpu_s2,  reg_ids::fpu_s3,
    reg_ids::fpu_s4,    reg_ids::fpu_s5,  reg_ids::fpu_s6,  reg_ids::fpu_s7,
    reg_ids::fpu_s8,    reg_ids::fpu_s9,  reg_ids::fpu_s10, reg_ids::fpu_s11,
    reg_ids::fpu_s12,   reg_ids::fpu_s13, reg_ids::fpu_s14, reg_ids::fpu_s15,
    reg_ids::fpu_s16,   reg_ids::fpu_s17, reg_ids::fpu_s18, reg_ids::fpu_s19,
    reg_ids::fpu_s20,   reg_ids::fpu_s21, reg_ids::fpu_s22, reg_ids::fpu_s23,
    reg_ids::fpu_s24,   reg_ids::fpu_s25, reg_ids::fpu_s26, reg_ids::fpu_s27,
    reg_ids::fpu_s28,   reg_ids::fpu_s29, reg_ids::fpu_s30, reg_ids::fpu_s31,

    reg_ids::fpu_d0,    reg_ids::fpu_d1,  reg_ids::fpu_d2,  reg_ids::fpu_d3,
    reg_ids::fpu_d4,    reg_ids::fpu_d5,  reg_ids::fpu_d6,  reg_ids::fpu_d7,
    reg_ids::fpu_d8,    reg_ids::fpu_d9,  reg_ids::fpu_d10, reg_ids::fpu_d11,
    reg_ids::fpu_d12,   reg_ids::fpu_d13, reg_ids::fpu_d14, reg_ids::fpu_d15,
    reg_ids::fpu_d16,   reg_ids::fpu_d17, reg_ids::fpu_d18, reg_ids::fpu_d19,
    reg_ids::fpu_d20,   reg_ids::fpu_d21, reg_ids::fpu_d22, reg_ids::fpu_d23,
    reg_ids::fpu_d24,   reg_ids::fpu_d25, reg_ids::fpu_d26, reg_ids::fpu_d27,
    reg_ids::fpu_d28,   reg_ids::fpu_d29, reg_ids::fpu_d30, reg_ids::fpu_d31,

    reg_ids::fpu_q0,    reg_ids::fpu_q1,  reg_ids::fpu_q2,  reg_ids::fpu_q3,
    reg_ids::fpu_q4,    reg_ids::fpu_q5,  reg_ids::fpu_q6,  reg_ids::fpu_q7,
    reg_ids::fpu_q8,    reg_ids::fpu_q9,  reg_ids::fpu_q10, reg_ids::fpu_q11,
    reg_ids::fpu_q12,   reg_ids::fpu_q13, reg_ids::fpu_q14, reg_ids::fpu_q15,

    reg_ids::fpu_fpscr,
    LLDB_INVALID_REGNUM // Register set must be terminated with this flag
};
static_assert(((sizeof g_fpr_regnums_arm / sizeof g_fpr_regnums_arm[0]) - 1) ==
                  reg_ids::k_num_fpr_registers,
              "g_fpu_regnums_arm has wrong number of register infos");

struct RegisterGroupNumbers {
  const char *name;
  const char *short_name;
  std::vector<uint32_t> registers;
};

static const RegisterGroupNumbers g_reg_sets_arm[] = {
    {"General Purpose Registers", "gpr",
     std::vector<RegisterID>(g_gpr_regnums_arm,
                             g_gpr_regnums_arm + std::size(g_gpr_regnums_arm) -
                                 1)},
    {"Floating Point Registers", "fpr",
     std::vector<RegisterID>(g_fpr_regnums_arm,
                             g_fpr_regnums_arm + std::size(g_fpr_regnums_arm) -
                                 1)}};

enum { k_num_register_sets = 2 };

} // namespace

static RegisterLayout CreateRegisterLayout(const ArchSpec &target_arch) {
  assert((HostInfo::GetArchitecture().GetAddressByteSize() == 4) &&
         "Register setting path assumes this is a 32-bit host");
  return llvm::cantFail(
      RegisterLayout::Create(RegisterInfo_arm::CreateDescription(),
                             RegisterLayout(target_arch.GetByteOrder())));
}

static Status GetThreadContextHelper(lldb::thread_t thread_handle,
                                     PCONTEXT context_ptr,
                                     const DWORD control_flag) {
  Log *log = GetLog(WindowsLog::Registers);
  Status error;

  memset(context_ptr, 0, sizeof(::CONTEXT));
  context_ptr->ContextFlags = control_flag;
  if (!::GetThreadContext(thread_handle, context_ptr)) {
    error = Status(GetLastError(), eErrorTypeWin32);
    LLDB_LOG(log, "{0} GetThreadContext failed with error {1}", __FUNCTION__,
             error);
    return error;
  }
  return Status();
}

static Status SetThreadContextHelper(lldb::thread_t thread_handle,
                                     PCONTEXT context_ptr) {
  Log *log = GetLog(WindowsLog::Registers);
  Status error;
  // It's assumed that the thread has stopped.
  if (!::SetThreadContext(thread_handle, context_ptr)) {
    error = Status(GetLastError(), eErrorTypeWin32);
    LLDB_LOG(log, "{0} SetThreadContext failed with error {1}", __FUNCTION__,
             error);
    return error;
  }
  return Status();
}

std::unique_ptr<NativeRegisterContextWindows>
NativeRegisterContextWindows::CreateHostNativeRegisterContextWindows(
    const ArchSpec &target_arch, NativeThreadProtocol &native_thread) {
  // TODO: Register context for a WoW64 application?

  // Register context for a native 64-bit application.
  return std::make_unique<NativeRegisterContextWindows_arm>(target_arch,
                                                            native_thread);
}

NativeRegisterContextWindows_arm::NativeRegisterContextWindows_arm(
    const ArchSpec &target_arch, NativeThreadProtocol &native_thread)
    : NativeRegisterContext(native_thread, target_arch,
                            CreateRegisterLayout(target_arch)) {
  // Bind native presentation policy to this schema generation once. The
  // private definitions above use the backend's external numbering.
  auto select_group = [&](uint32_t set_index) -> const RegisterGroupNumbers * {
    if (set_index >= k_num_register_sets)
      return nullptr;
    return &g_reg_sets_arm[set_index];
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

bool NativeRegisterContextWindows_arm::IsGPR(uint32_t reg_index) const {
  return (reg_index >= reg_ids::k_first_gpr &&
          reg_index <= reg_ids::k_last_gpr);
}

bool NativeRegisterContextWindows_arm::IsFPR(uint32_t reg_index) const {
  return (reg_index >= reg_ids::k_first_fpr &&
          reg_index <= reg_ids::k_last_fpr);
}

Status NativeRegisterContextWindows_arm::GPRRead(const uint32_t reg,
                                                 RegisterValue &reg_value) {
  ::CONTEXT tls_context;
  DWORD context_flag = CONTEXT_CONTROL | CONTEXT_INTEGER;
  Status error =
      GetThreadContextHelper(GetThreadHandle(), &tls_context, context_flag);
  if (error.Fail())
    return error;

  switch (reg) {
  case reg_ids::gpr_r0:
    reg_value.SetUInt32(tls_context.R0);
    break;
  case reg_ids::gpr_r1:
    reg_value.SetUInt32(tls_context.R1);
    break;
  case reg_ids::gpr_r2:
    reg_value.SetUInt32(tls_context.R2);
    break;
  case reg_ids::gpr_r3:
    reg_value.SetUInt32(tls_context.R3);
    break;
  case reg_ids::gpr_r4:
    reg_value.SetUInt32(tls_context.R4);
    break;
  case reg_ids::gpr_r5:
    reg_value.SetUInt32(tls_context.R5);
    break;
  case reg_ids::gpr_r6:
    reg_value.SetUInt32(tls_context.R6);
    break;
  case reg_ids::gpr_r7:
    reg_value.SetUInt32(tls_context.R7);
    break;
  case reg_ids::gpr_r8:
    reg_value.SetUInt32(tls_context.R8);
    break;
  case reg_ids::gpr_r9:
    reg_value.SetUInt32(tls_context.R9);
    break;
  case reg_ids::gpr_r10:
    reg_value.SetUInt32(tls_context.R10);
    break;
  case reg_ids::gpr_r11:
    reg_value.SetUInt32(tls_context.R11);
    break;
  case reg_ids::gpr_r12:
    reg_value.SetUInt32(tls_context.R12);
    break;
  case reg_ids::gpr_sp:
    reg_value.SetUInt32(tls_context.Sp);
    break;
  case reg_ids::gpr_lr:
    reg_value.SetUInt32(tls_context.Lr);
    break;
  case reg_ids::gpr_pc:
    reg_value.SetUInt32(tls_context.Pc);
    break;
  case reg_ids::gpr_cpsr:
    reg_value.SetUInt32(tls_context.Cpsr);
    break;
  }

  return error;
}

Status
NativeRegisterContextWindows_arm::GPRWrite(const uint32_t reg,
                                           const RegisterValue &reg_value) {
  ::CONTEXT tls_context;
  DWORD context_flag = CONTEXT_CONTROL | CONTEXT_INTEGER;
  auto thread_handle = GetThreadHandle();
  Status error =
      GetThreadContextHelper(thread_handle, &tls_context, context_flag);
  if (error.Fail())
    return error;

  switch (reg) {
  case reg_ids::gpr_r0:
    tls_context.R0 = reg_value.GetAsUInt32();
    break;
  case reg_ids::gpr_r1:
    tls_context.R1 = reg_value.GetAsUInt32();
    break;
  case reg_ids::gpr_r2:
    tls_context.R2 = reg_value.GetAsUInt32();
    break;
  case reg_ids::gpr_r3:
    tls_context.R3 = reg_value.GetAsUInt32();
    break;
  case reg_ids::gpr_r4:
    tls_context.R4 = reg_value.GetAsUInt32();
    break;
  case reg_ids::gpr_r5:
    tls_context.R5 = reg_value.GetAsUInt32();
    break;
  case reg_ids::gpr_r6:
    tls_context.R6 = reg_value.GetAsUInt32();
    break;
  case reg_ids::gpr_r7:
    tls_context.R7 = reg_value.GetAsUInt32();
    break;
  case reg_ids::gpr_r8:
    tls_context.R8 = reg_value.GetAsUInt32();
    break;
  case reg_ids::gpr_r9:
    tls_context.R9 = reg_value.GetAsUInt32();
    break;
  case reg_ids::gpr_r10:
    tls_context.R10 = reg_value.GetAsUInt32();
    break;
  case reg_ids::gpr_r11:
    tls_context.R11 = reg_value.GetAsUInt32();
    break;
  case reg_ids::gpr_r12:
    tls_context.R12 = reg_value.GetAsUInt32();
    break;
  case reg_ids::gpr_sp:
    tls_context.Sp = reg_value.GetAsUInt32();
    break;
  case reg_ids::gpr_lr:
    tls_context.Lr = reg_value.GetAsUInt32();
    break;
  case reg_ids::gpr_pc:
    tls_context.Pc = reg_value.GetAsUInt32();
    break;
  case reg_ids::gpr_cpsr:
    tls_context.Cpsr = reg_value.GetAsUInt32();
    break;
  }

  return SetThreadContextHelper(thread_handle, &tls_context);
}

Status NativeRegisterContextWindows_arm::FPRRead(const uint32_t reg,
                                                 RegisterValue &reg_value) {
  ::CONTEXT tls_context;
  DWORD context_flag = CONTEXT_CONTROL | CONTEXT_FLOATING_POINT;
  Status error =
      GetThreadContextHelper(GetThreadHandle(), &tls_context, context_flag);
  if (error.Fail())
    return error;

  switch (reg) {
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
    reg_value.SetUInt32(tls_context.S[reg - reg_ids::fpu_s0],
                        RegisterValue::eTypeFloat);
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
    reg_value.SetUInt64(tls_context.D[reg - reg_ids::fpu_d0],
                        RegisterValue::eTypeDouble);
    break;

  case reg_ids::fpu_q0:
  case reg_ids::fpu_q1:
  case reg_ids::fpu_q2:
  case reg_ids::fpu_q3:
  case reg_ids::fpu_q4:
  case reg_ids::fpu_q5:
  case reg_ids::fpu_q6:
  case reg_ids::fpu_q7:
  case reg_ids::fpu_q8:
  case reg_ids::fpu_q9:
  case reg_ids::fpu_q10:
  case reg_ids::fpu_q11:
  case reg_ids::fpu_q12:
  case reg_ids::fpu_q13:
  case reg_ids::fpu_q14:
  case reg_ids::fpu_q15:
    reg_value.SetBytes(&tls_context.Q[reg - reg_ids::fpu_q0], 16,
                       endian::InlHostByteOrder());
    break;

  case reg_ids::fpu_fpscr:
    reg_value.SetUInt32(tls_context.Fpscr);
    break;
  }

  return error;
}

Status
NativeRegisterContextWindows_arm::FPRWrite(const uint32_t reg,
                                           const RegisterValue &reg_value) {
  ::CONTEXT tls_context;
  DWORD context_flag = CONTEXT_CONTROL | CONTEXT_FLOATING_POINT;
  auto thread_handle = GetThreadHandle();
  Status error =
      GetThreadContextHelper(thread_handle, &tls_context, context_flag);
  if (error.Fail())
    return error;

  switch (reg) {
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
    tls_context.S[reg - reg_ids::fpu_s0] = reg_value.GetAsUInt32();
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
    tls_context.D[reg - reg_ids::fpu_d0] = reg_value.GetAsUInt64();
    break;

  case reg_ids::fpu_q0:
  case reg_ids::fpu_q1:
  case reg_ids::fpu_q2:
  case reg_ids::fpu_q3:
  case reg_ids::fpu_q4:
  case reg_ids::fpu_q5:
  case reg_ids::fpu_q6:
  case reg_ids::fpu_q7:
  case reg_ids::fpu_q8:
  case reg_ids::fpu_q9:
  case reg_ids::fpu_q10:
  case reg_ids::fpu_q11:
  case reg_ids::fpu_q12:
  case reg_ids::fpu_q13:
  case reg_ids::fpu_q14:
  case reg_ids::fpu_q15:
    memcpy(&tls_context.Q[reg - reg_ids::fpu_q0], reg_value.GetBytes(), 16);
    break;

  case reg_ids::fpu_fpscr:
    tls_context.Fpscr = reg_value.GetAsUInt32();
    break;
  }

  return SetThreadContextHelper(thread_handle, &tls_context);
}

llvm::Expected<RegisterValue>
NativeRegisterContextWindows_arm::ReadRegister(RegisterID id) {
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

llvm::Error NativeRegisterContextWindows_arm::WriteRegister(
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

Status NativeRegisterContextWindows_arm::ReadAllRegisterValues(
    lldb::WritableDataBufferSP &data_sp) {
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

Status NativeRegisterContextWindows_arm::WriteAllRegisterValues(
    const lldb::DataBufferSP &data_sp) {
  Status error;
  const size_t data_size = REG_CONTEXT_SIZE;
  if (!data_sp) {
    error = Status::FromErrorStringWithFormat(
        "NativeRegisterContextWindows_arm::%s invalid data_sp provided",
        __FUNCTION__);
    return error;
  }

  if (data_sp->GetByteSize() != data_size) {
    error = Status::FromErrorStringWithFormatv(
        "data_sp contained mismatched data size, expected {0}, actual {1}",
        data_size, data_sp->GetByteSize());
    return error;
  }

  ::CONTEXT tls_context;
  memcpy(&tls_context, data_sp->GetBytes(), data_size);
  return SetThreadContextHelper(GetThreadHandle(), &tls_context);
}

Status NativeRegisterContextWindows_arm::IsWatchpointHit(uint32_t wp_index,
                                                         bool &is_hit) {
  return Status::FromErrorString("unimplemented");
}

Status NativeRegisterContextWindows_arm::GetWatchpointHitIndex(
    uint32_t &wp_index, lldb::addr_t trap_addr) {
  return Status::FromErrorString("unimplemented");
}

Status NativeRegisterContextWindows_arm::IsWatchpointVacant(uint32_t wp_index,
                                                            bool &is_vacant) {
  return Status::FromErrorString("unimplemented");
}

Status NativeRegisterContextWindows_arm::SetHardwareWatchpointWithIndex(
    lldb::addr_t addr, size_t size, uint32_t watch_flags, uint32_t wp_index) {
  return Status::FromErrorString("unimplemented");
}

bool NativeRegisterContextWindows_arm::ClearHardwareWatchpoint(
    uint32_t wp_index) {
  return false;
}

Status NativeRegisterContextWindows_arm::ClearAllHardwareWatchpoints() {
  return Status::FromErrorString("unimplemented");
}

uint32_t NativeRegisterContextWindows_arm::SetHardwareWatchpoint(
    lldb::addr_t addr, size_t size, uint32_t watch_flags) {
  return LLDB_INVALID_INDEX32;
}

lldb::addr_t
NativeRegisterContextWindows_arm::GetWatchpointAddress(uint32_t wp_index) {
  return LLDB_INVALID_ADDRESS;
}

uint32_t NativeRegisterContextWindows_arm::NumSupportedHardwareWatchpoints() {
  // Not implemented
  return 0;
}

#endif // defined(__arm__) || defined(_M_ARM)

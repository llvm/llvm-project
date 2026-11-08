//===-- RegisterContextWindows_arm64.cpp ----------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#if defined(__aarch64__) || defined(_M_ARM64)

#include "lldb/Host/windows/HostThreadWindows.h"

#include "lldb/Register/Arch/lldb-arm64-register-enums.h"
#include "lldb/Register/Arch/lldb-arm64-dwarf-register-enums.h"
#include "lldb/Register/Common/RegisterInfo_arm64.h"
#include "lldb/lldb-defines.h"
#include "lldb/lldb-enumerations.h"
#include "lldb/lldb-private.h"
#include "llvm/Support/Compiler.h"
#include <cstddef>
#include <cstdint>

#include "lldb/Host/windows/windows.h"
#include "lldb/Register/RegisterValue.h"
#include "lldb/Utility/Status.h"
#include "lldb/lldb-private-types.h"

#include "RegisterContextWindows_arm64.h"
#include "TargetThreadWindows.h"
#include "lldb/Register/RegisterDescription.h"

#include "llvm/ADT/STLExtras.h"

namespace reg_ids = lldb_private::registers::arm64::id;
using namespace lldb;
using namespace lldb_private;

// Array of lldb register numbers used to define the set of all General Purpose
// Registers
uint32_t g_gpr_reg_indices[] = {
    reg_ids::gpr_x0,  reg_ids::gpr_x1,   reg_ids::gpr_x2,  reg_ids::gpr_x3,
    reg_ids::gpr_x4,  reg_ids::gpr_x5,   reg_ids::gpr_x6,  reg_ids::gpr_x7,
    reg_ids::gpr_x8,  reg_ids::gpr_x9,   reg_ids::gpr_x10, reg_ids::gpr_x11,
    reg_ids::gpr_x12, reg_ids::gpr_x13,  reg_ids::gpr_x14, reg_ids::gpr_x15,
    reg_ids::gpr_x16, reg_ids::gpr_x17,  reg_ids::gpr_x18, reg_ids::gpr_x19,
    reg_ids::gpr_x20, reg_ids::gpr_x21,  reg_ids::gpr_x22, reg_ids::gpr_x23,
    reg_ids::gpr_x24, reg_ids::gpr_x25,  reg_ids::gpr_x26, reg_ids::gpr_x27,
    reg_ids::gpr_x28, reg_ids::gpr_fp,   reg_ids::gpr_lr,  reg_ids::gpr_sp,
    reg_ids::gpr_pc,  reg_ids::gpr_cpsr,

    reg_ids::gpr_w0,  reg_ids::gpr_w1,   reg_ids::gpr_w2,  reg_ids::gpr_w3,
    reg_ids::gpr_w4,  reg_ids::gpr_w5,   reg_ids::gpr_w6,  reg_ids::gpr_w7,
    reg_ids::gpr_w8,  reg_ids::gpr_w9,   reg_ids::gpr_w10, reg_ids::gpr_w11,
    reg_ids::gpr_w12, reg_ids::gpr_w13,  reg_ids::gpr_w14, reg_ids::gpr_w15,
    reg_ids::gpr_w16, reg_ids::gpr_w17,  reg_ids::gpr_w18, reg_ids::gpr_w19,
    reg_ids::gpr_w20, reg_ids::gpr_w21,  reg_ids::gpr_w22, reg_ids::gpr_w23,
    reg_ids::gpr_w24, reg_ids::gpr_w25,  reg_ids::gpr_w26, reg_ids::gpr_w27,
    reg_ids::gpr_w28,
};

uint32_t g_fpu_reg_indices[] = {
    reg_ids::fpu_v0,   reg_ids::fpu_v1,   reg_ids::fpu_v2,  reg_ids::fpu_v3,
    reg_ids::fpu_v4,   reg_ids::fpu_v5,   reg_ids::fpu_v6,  reg_ids::fpu_v7,
    reg_ids::fpu_v8,   reg_ids::fpu_v9,   reg_ids::fpu_v10, reg_ids::fpu_v11,
    reg_ids::fpu_v12,  reg_ids::fpu_v13,  reg_ids::fpu_v14, reg_ids::fpu_v15,
    reg_ids::fpu_v16,  reg_ids::fpu_v17,  reg_ids::fpu_v18, reg_ids::fpu_v19,
    reg_ids::fpu_v20,  reg_ids::fpu_v21,  reg_ids::fpu_v22, reg_ids::fpu_v23,
    reg_ids::fpu_v24,  reg_ids::fpu_v25,  reg_ids::fpu_v26, reg_ids::fpu_v27,
    reg_ids::fpu_v28,  reg_ids::fpu_v29,  reg_ids::fpu_v30, reg_ids::fpu_v31,

    reg_ids::fpu_s0,   reg_ids::fpu_s1,   reg_ids::fpu_s2,  reg_ids::fpu_s3,
    reg_ids::fpu_s4,   reg_ids::fpu_s5,   reg_ids::fpu_s6,  reg_ids::fpu_s7,
    reg_ids::fpu_s8,   reg_ids::fpu_s9,   reg_ids::fpu_s10, reg_ids::fpu_s11,
    reg_ids::fpu_s12,  reg_ids::fpu_s13,  reg_ids::fpu_s14, reg_ids::fpu_s15,
    reg_ids::fpu_s16,  reg_ids::fpu_s17,  reg_ids::fpu_s18, reg_ids::fpu_s19,
    reg_ids::fpu_s20,  reg_ids::fpu_s21,  reg_ids::fpu_s22, reg_ids::fpu_s23,
    reg_ids::fpu_s24,  reg_ids::fpu_s25,  reg_ids::fpu_s26, reg_ids::fpu_s27,
    reg_ids::fpu_s28,  reg_ids::fpu_s29,  reg_ids::fpu_s30, reg_ids::fpu_s31,

    reg_ids::fpu_d0,   reg_ids::fpu_d1,   reg_ids::fpu_d2,  reg_ids::fpu_d3,
    reg_ids::fpu_d4,   reg_ids::fpu_d5,   reg_ids::fpu_d6,  reg_ids::fpu_d7,
    reg_ids::fpu_d8,   reg_ids::fpu_d9,   reg_ids::fpu_d10, reg_ids::fpu_d11,
    reg_ids::fpu_d12,  reg_ids::fpu_d13,  reg_ids::fpu_d14, reg_ids::fpu_d15,
    reg_ids::fpu_d16,  reg_ids::fpu_d17,  reg_ids::fpu_d18, reg_ids::fpu_d19,
    reg_ids::fpu_d20,  reg_ids::fpu_d21,  reg_ids::fpu_d22, reg_ids::fpu_d23,
    reg_ids::fpu_d24,  reg_ids::fpu_d25,  reg_ids::fpu_d26, reg_ids::fpu_d27,
    reg_ids::fpu_d28,  reg_ids::fpu_d29,  reg_ids::fpu_d30, reg_ids::fpu_d31,

    reg_ids::fpu_fpsr, reg_ids::fpu_fpcr,
};

RegisterGroup g_register_sets[] = {
    {"General Purpose Registers", "gpr",
     std::vector<RegisterID>(g_gpr_reg_indices,
                             g_gpr_reg_indices + std::size(g_gpr_reg_indices))},
    {"Floating Point Registers", "fpu",
     std::vector<RegisterID>(
         g_fpu_reg_indices, g_fpu_reg_indices + std::size(g_fpu_reg_indices))}};

// Constructors and Destructors
static std::shared_ptr<const RegisterDescription> RegisterMetadata() {
  static const auto description = [] {
    auto cpu = RegisterInfo_arm64::CreateDescription(0);
    return std::make_shared<RegisterDescription>(
        llvm::cantFail(RegisterDescription::Create(
            {cpu->GetRegisters().begin(), cpu->GetRegisters().end()},
            {std::begin(g_register_sets), std::end(g_register_sets)},
            {cpu->GetOwnedTypes().begin(), cpu->GetOwnedTypes().end()})));
  }();
  return description;
}

RegisterContextWindows_arm64::RegisterContextWindows_arm64(
    Thread &thread, uint32_t concrete_frame_idx)
    : RegisterContextWindows(thread, concrete_frame_idx) {
  SetRegisterDescription(RegisterMetadata());
}

RegisterContextWindows_arm64::~RegisterContextWindows_arm64() {}

llvm::Expected<RegisterValue>
RegisterContextWindows_arm64::ReadRegister(RegisterID id) {
  auto description = GetDescription();
  const auto *reg_info = description ? description->Find(id) : nullptr;
  if (!reg_info)
    return llvm::createStringError("unknown register ID");
  RegisterValue reg_value;
  if (!CacheAllRegisterValues())
    return llvm::createStringError("register value is unavailable");

  if (reg_info == nullptr)
    return llvm::createStringError("register value is unavailable");

  const uint32_t reg = reg_info->numbers[eRegisterKindLLDB];

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
    reg_value.SetUInt64(m_context.X[reg - reg_ids::gpr_x0]);
    break;

  case reg_ids::gpr_fp:
    reg_value.SetUInt64(m_context.Fp);
    break;
  case reg_ids::gpr_sp:
    reg_value.SetUInt64(m_context.Sp);
    break;
  case reg_ids::gpr_lr:
    reg_value.SetUInt64(m_context.Lr);
    break;
  case reg_ids::gpr_pc:
    reg_value.SetUInt64(m_context.Pc);
    break;
  case reg_ids::gpr_cpsr:
    reg_value.SetUInt32(m_context.Cpsr);
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
        static_cast<uint32_t>(m_context.X[reg - reg_ids::gpr_w0] & 0xffffffff));
    break;

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
    reg_value.SetBytes(m_context.V[reg - reg_ids::fpu_v0].B, reg_info->byte_size,
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
    reg_value.SetFloat(m_context.V[reg - reg_ids::fpu_s0].S[0]);
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
    reg_value.SetDouble(m_context.V[reg - reg_ids::fpu_d0].D[0]);
    break;

  case reg_ids::fpu_fpsr:
    reg_value.SetUInt32(m_context.Fpsr);
    break;

  case reg_ids::fpu_fpcr:
    reg_value.SetUInt32(m_context.Fpcr);
    break;

  default:
    reg_value.SetValueToInvalid();
    return llvm::createStringError("register value is unavailable");
  }
  return reg_value;
}

llvm::Error
RegisterContextWindows_arm64::WriteRegister(RegisterID id,
                                            const RegisterValue &reg_value) {
  auto description = GetDescription();
  const auto *reg_info = description ? description->Find(id) : nullptr;
  if (!reg_info)
    return llvm::createStringError("unknown register ID");
  // Since we cannot only write a single register value to the inferior, we
  // need to make sure our cached copy of the register values are fresh.
  // Otherwise when writing one register, we may also overwrite some other
  // register with a stale value.
  if (!CacheAllRegisterValues())
    return llvm::createStringError("register write is unavailable");

  const uint32_t reg = reg_info->numbers[eRegisterKindLLDB];

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
    m_context.X[reg - reg_ids::gpr_x0] = reg_value.GetAsUInt64();
    break;

  case reg_ids::gpr_fp:
    m_context.Fp = reg_value.GetAsUInt64();
    break;
  case reg_ids::gpr_sp:
    m_context.Sp = reg_value.GetAsUInt64();
    break;
  case reg_ids::gpr_lr:
    m_context.Lr = reg_value.GetAsUInt64();
    break;
  case reg_ids::gpr_pc:
    m_context.Pc = reg_value.GetAsUInt64();
    break;
  case reg_ids::gpr_cpsr:
    m_context.Cpsr = reg_value.GetAsUInt32();
    break;

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
    memcpy(m_context.V[reg - reg_ids::fpu_v0].B, reg_value.GetBytes(), 16);
    break;

  case reg_ids::fpu_fpsr:
    m_context.Fpsr = reg_value.GetAsUInt32();
    break;

  case reg_ids::fpu_fpcr:
    m_context.Fpcr = reg_value.GetAsUInt32();
    break;

  default:
    return llvm::createStringError("register write is unavailable");
  }

  // Physically update the registers in the target process.
  {
    if (!(ApplyAllRegisterValues()))
      return llvm::createStringError("register write failed");
    return llvm::Error::success();
  }
}

#endif // defined(__aarch64__) || defined(_M_ARM64)

//===-- RegisterContextWindows_arm.cpp ------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#if defined(__arm__) || defined(_M_ARM)

#include "lldb/Host/windows/HostThreadWindows.h"

#include "lldb/Register/Arch/lldb-arm-register-enums.h"
#include "lldb/Register/Arch/lldb-arm-dwarf-register-enums.h"
#include "lldb/Register/Common/RegisterInfo_arm.h"
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

#include "ProcessWindowsLog.h"
#include "RegisterContextWindows_arm.h"
#include "TargetThreadWindows.h"
#include "lldb/Register/RegisterDescription.h"

#include "llvm/ADT/STLExtras.h"

namespace reg_ids = lldb_private::registers::arm::id;
using namespace lldb;
using namespace lldb_private;

// Array of lldb register numbers used to define the set of all General Purpose
// Registers
uint32_t g_gpr_reg_indices[] = {
    reg_ids::gpr_r0,   reg_ids::gpr_r1, reg_ids::gpr_r2,  reg_ids::gpr_r3,
    reg_ids::gpr_r4,   reg_ids::gpr_r5, reg_ids::gpr_r6,  reg_ids::gpr_r7,
    reg_ids::gpr_r8,   reg_ids::gpr_r9, reg_ids::gpr_r10, reg_ids::gpr_r11,
    reg_ids::gpr_r12,  reg_ids::gpr_sp, reg_ids::gpr_lr,  reg_ids::gpr_pc,
    reg_ids::gpr_cpsr,
};

uint32_t g_fpu_reg_indices[] = {
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
    auto cpu = RegisterInfo_arm::CreateDescription();
    return std::make_shared<RegisterDescription>(
        llvm::cantFail(RegisterDescription::Create(
            {cpu->GetRegisters().begin(), cpu->GetRegisters().end()},
            {std::begin(g_register_sets), std::end(g_register_sets)},
            {cpu->GetOwnedTypes().begin(), cpu->GetOwnedTypes().end()})));
  }();
  return description;
}

RegisterContextWindows_arm::RegisterContextWindows_arm(
    Thread &thread, uint32_t concrete_frame_idx)
    : RegisterContextWindows(thread, concrete_frame_idx) {
  SetRegisterDescription(RegisterMetadata());
}

RegisterContextWindows_arm::~RegisterContextWindows_arm() = default;

llvm::Expected<RegisterValue>
RegisterContextWindows_arm::ReadRegister(RegisterID id) {
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
  case reg_ids::gpr_r0:
    reg_value.SetUInt32(m_context.R0);
    break;
  case reg_ids::gpr_r1:
    reg_value.SetUInt32(m_context.R1);
    break;
  case reg_ids::gpr_r2:
    reg_value.SetUInt32(m_context.R2);
    break;
  case reg_ids::gpr_r3:
    reg_value.SetUInt32(m_context.R3);
    break;
  case reg_ids::gpr_r4:
    reg_value.SetUInt32(m_context.R4);
    break;
  case reg_ids::gpr_r5:
    reg_value.SetUInt32(m_context.R5);
    break;
  case reg_ids::gpr_r6:
    reg_value.SetUInt32(m_context.R6);
    break;
  case reg_ids::gpr_r7:
    reg_value.SetUInt32(m_context.R7);
    break;
  case reg_ids::gpr_r8:
    reg_value.SetUInt32(m_context.R8);
    break;
  case reg_ids::gpr_r9:
    reg_value.SetUInt32(m_context.R9);
    break;
  case reg_ids::gpr_r10:
    reg_value.SetUInt32(m_context.R10);
    break;
  case reg_ids::gpr_r11:
    reg_value.SetUInt32(m_context.R11);
    break;
  case reg_ids::gpr_r12:
    reg_value.SetUInt32(m_context.R12);
    break;
  case reg_ids::gpr_sp:
    reg_value.SetUInt32(m_context.Sp);
    break;
  case reg_ids::gpr_lr:
    reg_value.SetUInt32(m_context.Lr);
    break;
  case reg_ids::gpr_pc:
    reg_value.SetUInt32(m_context.Pc);
    break;
  case reg_ids::gpr_cpsr:
    reg_value.SetUInt32(m_context.Cpsr);
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
    reg_value.SetUInt32(m_context.S[reg - reg_ids::fpu_s0],
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
    reg_value.SetUInt64(m_context.D[reg - reg_ids::fpu_d0],
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
    reg_value.SetBytes(&m_context.Q[reg - reg_ids::fpu_q0], reg_info->byte_size,
                       endian::InlHostByteOrder());
    break;

  case reg_ids::fpu_fpscr:
    reg_value.SetUInt32(m_context.Fpscr);
    break;

  default:
    reg_value.SetValueToInvalid();
    return llvm::createStringError("register value is unavailable");
  }
  return reg_value;
}

llvm::Error
RegisterContextWindows_arm::WriteRegister(RegisterID id,
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

  const uint32_t reg = reg_info->numbers[eRegisterKindLLDB];

  Log *log = GetLog(WindowsLog::Registers);
  switch (reg) {
  case reg_ids::gpr_r0:
    m_context.R0 = reg_value.GetAsUInt32();
    break;
  case reg_ids::gpr_r1:
    m_context.R1 = reg_value.GetAsUInt32();
    break;
  case reg_ids::gpr_r2:
    m_context.R2 = reg_value.GetAsUInt32();
    break;
  case reg_ids::gpr_r3:
    m_context.R3 = reg_value.GetAsUInt32();
    break;
  case reg_ids::gpr_r4:
    m_context.R4 = reg_value.GetAsUInt32();
    break;
  case reg_ids::gpr_r5:
    m_context.R5 = reg_value.GetAsUInt32();
    break;
  case reg_ids::gpr_r6:
    m_context.R6 = reg_value.GetAsUInt32();
    break;
  case reg_ids::gpr_r7:
    m_context.R7 = reg_value.GetAsUInt32();
    break;
  case reg_ids::gpr_r8:
    m_context.R8 = reg_value.GetAsUInt32();
    break;
  case reg_ids::gpr_r9:
    m_context.R9 = reg_value.GetAsUInt32();
    break;
  case reg_ids::gpr_r10:
    m_context.R10 = reg_value.GetAsUInt32();
    break;
  case reg_ids::gpr_r11:
    m_context.R11 = reg_value.GetAsUInt32();
    break;
  case reg_ids::gpr_r12:
    m_context.R12 = reg_value.GetAsUInt32();
    break;
  case reg_ids::gpr_sp:
    m_context.Sp = reg_value.GetAsUInt32();
    break;
  case reg_ids::gpr_lr:
    m_context.Lr = reg_value.GetAsUInt32();
    break;
  case reg_ids::gpr_pc:
    m_context.Pc = reg_value.GetAsUInt32();
    break;
  case reg_ids::gpr_cpsr:
    m_context.Cpsr = reg_value.GetAsUInt32();
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
    m_context.S[reg - reg_ids::fpu_s0] = reg_value.GetAsUInt32();
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
    m_context.D[reg - reg_ids::fpu_d0] = reg_value.GetAsUInt64();
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
    memcpy(&m_context.Q[reg - reg_ids::fpu_q0], reg_value.GetBytes(), 16);
    break;

  case reg_ids::fpu_fpscr:
    m_context.Fpscr = reg_value.GetAsUInt32();
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

#endif // defined(__arm__) || defined(_M_ARM)

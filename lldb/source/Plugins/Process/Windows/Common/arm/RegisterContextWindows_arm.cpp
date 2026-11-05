//===-- RegisterContextWindows_arm.cpp ------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#if defined(__arm__) || defined(_M_ARM)

#include "lldb/Host/windows/HostThreadWindows.h"

#include "Register/Arch/lldb-arm-register-enums.h"
#include "lldb/Host/windows/windows.h"
#include "lldb/Utility/RegisterValue.h"
#include "lldb/Utility/Status.h"
#include "lldb/lldb-private-types.h"

#include "ProcessWindowsLog.h"
#include "RegisterContextWindows_arm.h"
#include "TargetThreadWindows.h"

#include "llvm/ADT/STLExtras.h"

using namespace lldb;
using namespace lldb_private;

#define GPR_OFFSET(idx) 0
#define FPU_OFFSET(idx) 0
#define FPSCR_OFFSET 0
#define TLS_OFFSET 0
#define EXC_OFFSET(reg) 0
#define DBG_OFFSET_NAME(reg) 0

#define DEFINE_DBG(reg, i)                                                     \
  #reg, nullptr, 0, DBG_OFFSET_NAME(reg[i]), eEncodingUint, eFormatHex,        \
      {LLDB_INVALID_REGNUM, LLDB_INVALID_REGNUM, LLDB_INVALID_REGNUM,          \
       LLDB_INVALID_REGNUM, LLDB_INVALID_REGNUM},                              \
      nullptr, nullptr, nullptr

// Include RegisterInfos_arm to declare our g_register_infos_arm structure.
#define DECLARE_REGISTER_INFOS_ARM_STRUCT
#include "Register/Arch/RegisterInfos_arm.h"
#undef DECLARE_REGISTER_INFOS_ARM_STRUCT

static size_t k_num_register_infos = std::size(g_register_infos_arm);

// Array of lldb register numbers used to define the set of all General Purpose
// Registers
uint32_t g_gpr_reg_indices[] = {
    gpr_r0_arm, gpr_r1_arm,  gpr_r2_arm,  gpr_r3_arm,  gpr_r4_arm, gpr_r5_arm, gpr_r6_arm, gpr_r7_arm,   gpr_r8_arm,
    gpr_r9_arm, gpr_r10_arm, gpr_r11_arm, gpr_r12_arm, gpr_sp_arm, gpr_lr_arm, gpr_pc_arm, gpr_cpsr_arm,
};

uint32_t g_fpu_reg_indices[] = {
    fpu_s0_arm,    fpu_s1_arm,  fpu_s2_arm,  fpu_s3_arm,  fpu_s4_arm,  fpu_s5_arm,  fpu_s6_arm,  fpu_s7_arm,
    fpu_s8_arm,    fpu_s9_arm,  fpu_s10_arm, fpu_s11_arm, fpu_s12_arm, fpu_s13_arm, fpu_s14_arm, fpu_s15_arm,
    fpu_s16_arm,   fpu_s17_arm, fpu_s18_arm, fpu_s19_arm, fpu_s20_arm, fpu_s21_arm, fpu_s22_arm, fpu_s23_arm,
    fpu_s24_arm,   fpu_s25_arm, fpu_s26_arm, fpu_s27_arm, fpu_s28_arm, fpu_s29_arm, fpu_s30_arm, fpu_s31_arm,

    fpu_d0_arm,    fpu_d1_arm,  fpu_d2_arm,  fpu_d3_arm,  fpu_d4_arm,  fpu_d5_arm,  fpu_d6_arm,  fpu_d7_arm,
    fpu_d8_arm,    fpu_d9_arm,  fpu_d10_arm, fpu_d11_arm, fpu_d12_arm, fpu_d13_arm, fpu_d14_arm, fpu_d15_arm,
    fpu_d16_arm,   fpu_d17_arm, fpu_d18_arm, fpu_d19_arm, fpu_d20_arm, fpu_d21_arm, fpu_d22_arm, fpu_d23_arm,
    fpu_d24_arm,   fpu_d25_arm, fpu_d26_arm, fpu_d27_arm, fpu_d28_arm, fpu_d29_arm, fpu_d30_arm, fpu_d31_arm,

    fpu_q0_arm,    fpu_q1_arm,  fpu_q2_arm,  fpu_q3_arm,  fpu_q4_arm,  fpu_q5_arm,  fpu_q6_arm,  fpu_q7_arm,
    fpu_q8_arm,    fpu_q9_arm,  fpu_q10_arm, fpu_q11_arm, fpu_q12_arm, fpu_q13_arm, fpu_q14_arm, fpu_q15_arm,

    fpu_fpscr_arm,
};

RegisterSet g_register_sets[] = {
    {"General Purpose Registers", "gpr", std::size(g_gpr_reg_indices),
     g_gpr_reg_indices},
    {"Floating Point Registers", "fpu", std::size(g_fpu_reg_indices),
     g_fpu_reg_indices},
};

// Constructors and Destructors
RegisterContextWindows_arm::RegisterContextWindows_arm(
    Thread &thread, uint32_t concrete_frame_idx)
    : RegisterContextWindows(thread, concrete_frame_idx) {}

RegisterContextWindows_arm::~RegisterContextWindows_arm() = default;

size_t RegisterContextWindows_arm::GetRegisterCount() {
  return std::size(g_register_infos_arm);
}

const RegisterInfo *
RegisterContextWindows_arm::GetRegisterInfoAtIndex(size_t reg) {
  if (reg < k_num_register_infos)
    return &g_register_infos_arm[reg];
  return nullptr;
}

size_t RegisterContextWindows_arm::GetRegisterSetCount() {
  return std::size(g_register_sets);
}

const RegisterSet *RegisterContextWindows_arm::GetRegisterSet(size_t reg_set) {
  return &g_register_sets[reg_set];
}

bool RegisterContextWindows_arm::ReadRegister(const RegisterInfo *reg_info,
                                              RegisterValue &reg_value) {
  if (!CacheAllRegisterValues())
    return false;

  if (reg_info == nullptr)
    return false;

  const uint32_t reg = reg_info->kinds[eRegisterKindLLDB];

  switch (reg) {
  case gpr_r0_arm:
    reg_value.SetUInt32(m_context.R0);
    break;
  case gpr_r1_arm:
    reg_value.SetUInt32(m_context.R1);
    break;
  case gpr_r2_arm:
    reg_value.SetUInt32(m_context.R2);
    break;
  case gpr_r3_arm:
    reg_value.SetUInt32(m_context.R3);
    break;
  case gpr_r4_arm:
    reg_value.SetUInt32(m_context.R4);
    break;
  case gpr_r5_arm:
    reg_value.SetUInt32(m_context.R5);
    break;
  case gpr_r6_arm:
    reg_value.SetUInt32(m_context.R6);
    break;
  case gpr_r7_arm:
    reg_value.SetUInt32(m_context.R7);
    break;
  case gpr_r8_arm:
    reg_value.SetUInt32(m_context.R8);
    break;
  case gpr_r9_arm:
    reg_value.SetUInt32(m_context.R9);
    break;
  case gpr_r10_arm:
    reg_value.SetUInt32(m_context.R10);
    break;
  case gpr_r11_arm:
    reg_value.SetUInt32(m_context.R11);
    break;
  case gpr_r12_arm:
    reg_value.SetUInt32(m_context.R12);
    break;
  case gpr_sp_arm:
    reg_value.SetUInt32(m_context.Sp);
    break;
  case gpr_lr_arm:
    reg_value.SetUInt32(m_context.Lr);
    break;
  case gpr_pc_arm:
    reg_value.SetUInt32(m_context.Pc);
    break;
  case gpr_cpsr_arm:
    reg_value.SetUInt32(m_context.Cpsr);
    break;

  case fpu_s0_arm:
  case fpu_s1_arm:
  case fpu_s2_arm:
  case fpu_s3_arm:
  case fpu_s4_arm:
  case fpu_s5_arm:
  case fpu_s6_arm:
  case fpu_s7_arm:
  case fpu_s8_arm:
  case fpu_s9_arm:
  case fpu_s10_arm:
  case fpu_s11_arm:
  case fpu_s12_arm:
  case fpu_s13_arm:
  case fpu_s14_arm:
  case fpu_s15_arm:
  case fpu_s16_arm:
  case fpu_s17_arm:
  case fpu_s18_arm:
  case fpu_s19_arm:
  case fpu_s20_arm:
  case fpu_s21_arm:
  case fpu_s22_arm:
  case fpu_s23_arm:
  case fpu_s24_arm:
  case fpu_s25_arm:
  case fpu_s26_arm:
  case fpu_s27_arm:
  case fpu_s28_arm:
  case fpu_s29_arm:
  case fpu_s30_arm:
  case fpu_s31_arm:
    reg_value.SetUInt32(m_context.S[reg - fpu_s0_arm], RegisterValue::eTypeFloat);
    break;

  case fpu_d0_arm:
  case fpu_d1_arm:
  case fpu_d2_arm:
  case fpu_d3_arm:
  case fpu_d4_arm:
  case fpu_d5_arm:
  case fpu_d6_arm:
  case fpu_d7_arm:
  case fpu_d8_arm:
  case fpu_d9_arm:
  case fpu_d10_arm:
  case fpu_d11_arm:
  case fpu_d12_arm:
  case fpu_d13_arm:
  case fpu_d14_arm:
  case fpu_d15_arm:
  case fpu_d16_arm:
  case fpu_d17_arm:
  case fpu_d18_arm:
  case fpu_d19_arm:
  case fpu_d20_arm:
  case fpu_d21_arm:
  case fpu_d22_arm:
  case fpu_d23_arm:
  case fpu_d24_arm:
  case fpu_d25_arm:
  case fpu_d26_arm:
  case fpu_d27_arm:
  case fpu_d28_arm:
  case fpu_d29_arm:
  case fpu_d30_arm:
  case fpu_d31_arm:
    reg_value.SetUInt64(m_context.D[reg - fpu_d0_arm], RegisterValue::eTypeDouble);
    break;

  case fpu_q0_arm:
  case fpu_q1_arm:
  case fpu_q2_arm:
  case fpu_q3_arm:
  case fpu_q4_arm:
  case fpu_q5_arm:
  case fpu_q6_arm:
  case fpu_q7_arm:
  case fpu_q8_arm:
  case fpu_q9_arm:
  case fpu_q10_arm:
  case fpu_q11_arm:
  case fpu_q12_arm:
  case fpu_q13_arm:
  case fpu_q14_arm:
  case fpu_q15_arm:
    reg_value.SetBytes(&m_context.Q[reg - fpu_q0_arm], reg_info->byte_size,
                       endian::InlHostByteOrder());
    break;

  case fpu_fpscr_arm:
    reg_value.SetUInt32(m_context.Fpscr);
    break;

  default:
    reg_value.SetValueToInvalid();
    return false;
  }
  return true;
}

bool RegisterContextWindows_arm::WriteRegister(const RegisterInfo *reg_info,
                                               const RegisterValue &reg_value) {
  // Since we cannot only write a single register value to the inferior, we
  // need to make sure our cached copy of the register values are fresh.
  // Otherwise when writing EAX, for example, we may also overwrite some other
  // register with a stale value.
  if (!CacheAllRegisterValues())
    return false;

  const uint32_t reg = reg_info->kinds[eRegisterKindLLDB];

  Log *log = GetLog(WindowsLog::Registers);
  switch (reg) {
  case gpr_r0_arm:
    m_context.R0 = reg_value.GetAsUInt32();
    break;
  case gpr_r1_arm:
    m_context.R1 = reg_value.GetAsUInt32();
    break;
  case gpr_r2_arm:
    m_context.R2 = reg_value.GetAsUInt32();
    break;
  case gpr_r3_arm:
    m_context.R3 = reg_value.GetAsUInt32();
    break;
  case gpr_r4_arm:
    m_context.R4 = reg_value.GetAsUInt32();
    break;
  case gpr_r5_arm:
    m_context.R5 = reg_value.GetAsUInt32();
    break;
  case gpr_r6_arm:
    m_context.R6 = reg_value.GetAsUInt32();
    break;
  case gpr_r7_arm:
    m_context.R7 = reg_value.GetAsUInt32();
    break;
  case gpr_r8_arm:
    m_context.R8 = reg_value.GetAsUInt32();
    break;
  case gpr_r9_arm:
    m_context.R9 = reg_value.GetAsUInt32();
    break;
  case gpr_r10_arm:
    m_context.R10 = reg_value.GetAsUInt32();
    break;
  case gpr_r11_arm:
    m_context.R11 = reg_value.GetAsUInt32();
    break;
  case gpr_r12_arm:
    m_context.R12 = reg_value.GetAsUInt32();
    break;
  case gpr_sp_arm:
    m_context.Sp = reg_value.GetAsUInt32();
    break;
  case gpr_lr_arm:
    m_context.Lr = reg_value.GetAsUInt32();
    break;
  case gpr_pc_arm:
    m_context.Pc = reg_value.GetAsUInt32();
    break;
  case gpr_cpsr_arm:
    m_context.Cpsr = reg_value.GetAsUInt32();
    break;

  case fpu_s0_arm:
  case fpu_s1_arm:
  case fpu_s2_arm:
  case fpu_s3_arm:
  case fpu_s4_arm:
  case fpu_s5_arm:
  case fpu_s6_arm:
  case fpu_s7_arm:
  case fpu_s8_arm:
  case fpu_s9_arm:
  case fpu_s10_arm:
  case fpu_s11_arm:
  case fpu_s12_arm:
  case fpu_s13_arm:
  case fpu_s14_arm:
  case fpu_s15_arm:
  case fpu_s16_arm:
  case fpu_s17_arm:
  case fpu_s18_arm:
  case fpu_s19_arm:
  case fpu_s20_arm:
  case fpu_s21_arm:
  case fpu_s22_arm:
  case fpu_s23_arm:
  case fpu_s24_arm:
  case fpu_s25_arm:
  case fpu_s26_arm:
  case fpu_s27_arm:
  case fpu_s28_arm:
  case fpu_s29_arm:
  case fpu_s30_arm:
  case fpu_s31_arm:
    m_context.S[reg - fpu_s0_arm] = reg_value.GetAsUInt32();
    break;

  case fpu_d0_arm:
  case fpu_d1_arm:
  case fpu_d2_arm:
  case fpu_d3_arm:
  case fpu_d4_arm:
  case fpu_d5_arm:
  case fpu_d6_arm:
  case fpu_d7_arm:
  case fpu_d8_arm:
  case fpu_d9_arm:
  case fpu_d10_arm:
  case fpu_d11_arm:
  case fpu_d12_arm:
  case fpu_d13_arm:
  case fpu_d14_arm:
  case fpu_d15_arm:
  case fpu_d16_arm:
  case fpu_d17_arm:
  case fpu_d18_arm:
  case fpu_d19_arm:
  case fpu_d20_arm:
  case fpu_d21_arm:
  case fpu_d22_arm:
  case fpu_d23_arm:
  case fpu_d24_arm:
  case fpu_d25_arm:
  case fpu_d26_arm:
  case fpu_d27_arm:
  case fpu_d28_arm:
  case fpu_d29_arm:
  case fpu_d30_arm:
  case fpu_d31_arm:
    m_context.D[reg - fpu_d0_arm] = reg_value.GetAsUInt64();
    break;

  case fpu_q0_arm:
  case fpu_q1_arm:
  case fpu_q2_arm:
  case fpu_q3_arm:
  case fpu_q4_arm:
  case fpu_q5_arm:
  case fpu_q6_arm:
  case fpu_q7_arm:
  case fpu_q8_arm:
  case fpu_q9_arm:
  case fpu_q10_arm:
  case fpu_q11_arm:
  case fpu_q12_arm:
  case fpu_q13_arm:
  case fpu_q14_arm:
  case fpu_q15_arm:
    memcpy(&m_context.Q[reg - fpu_q0_arm], reg_value.GetBytes(), 16);
    break;

  case fpu_fpscr_arm:
    m_context.Fpscr = reg_value.GetAsUInt32();
    break;

  default:
    LLDB_LOG(log, "Write value {0:x} to unknown register {1}",
             reg_value.GetAsUInt32(), reg);
    return false;
  }

  // Physically update the registers in the target process.
  return ApplyAllRegisterValues();
}

#endif // defined(__arm__) || defined(_M_ARM)

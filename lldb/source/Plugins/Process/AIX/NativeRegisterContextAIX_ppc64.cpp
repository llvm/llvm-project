//===------ NativeRegisterContextAIX_ppc64.cpp ----------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#if defined(__powerpc64__)

#include "NativeRegisterContextAIX_ppc64.h"

using namespace lldb;
using namespace lldb_private;
using namespace lldb_private::process_aix;

static const uint32_t g_gpr_regnums_ppc64[] = {
    gpr_r0_ppc64,       gpr_r1_ppc64,  gpr_r2_ppc64,  gpr_r3_ppc64,
    gpr_r4_ppc64,       gpr_r5_ppc64,  gpr_r6_ppc64,  gpr_r7_ppc64,
    gpr_r8_ppc64,       gpr_r9_ppc64,  gpr_r10_ppc64, gpr_r11_ppc64,
    gpr_r12_ppc64,      gpr_r13_ppc64, gpr_r14_ppc64, gpr_r15_ppc64,
    gpr_r16_ppc64,      gpr_r17_ppc64, gpr_r18_ppc64, gpr_r19_ppc64,
    gpr_r20_ppc64,      gpr_r21_ppc64, gpr_r22_ppc64, gpr_r23_ppc64,
    gpr_r24_ppc64,      gpr_r25_ppc64, gpr_r26_ppc64, gpr_r27_ppc64,
    gpr_r28_ppc64,      gpr_r29_ppc64, gpr_r30_ppc64, gpr_r31_ppc64,
    gpr_cr_ppc64,       gpr_msr_ppc64, gpr_xer_ppc64, gpr_lr_ppc64,
    gpr_ctr_ppc64,      gpr_pc_ppc64,
    LLDB_INVALID_REGNUM // register sets need to end with this flag
};

static const uint32_t g_fpr_regnums_ppc64[] = {
    fpr_f0_ppc64,       fpr_f1_ppc64,  fpr_f2_ppc64,  fpr_f3_ppc64,
    fpr_f4_ppc64,       fpr_f5_ppc64,  fpr_f6_ppc64,  fpr_f7_ppc64,
    fpr_f8_ppc64,       fpr_f9_ppc64,  fpr_f10_ppc64, fpr_f11_ppc64,
    fpr_f12_ppc64,      fpr_f13_ppc64, fpr_f14_ppc64, fpr_f15_ppc64,
    fpr_f16_ppc64,      fpr_f17_ppc64, fpr_f18_ppc64, fpr_f19_ppc64,
    fpr_f20_ppc64,      fpr_f21_ppc64, fpr_f22_ppc64, fpr_f23_ppc64,
    fpr_f24_ppc64,      fpr_f25_ppc64, fpr_f26_ppc64, fpr_f27_ppc64,
    fpr_f28_ppc64,      fpr_f29_ppc64, fpr_f30_ppc64, fpr_f31_ppc64,
    fpr_fpscr_ppc64,
    LLDB_INVALID_REGNUM // register sets need to end with this flag
};

static const uint32_t g_vmx_regnums_ppc64[] = {
    vmx_vr0_ppc64,      vmx_vr1_ppc64,    vmx_vr2_ppc64,  vmx_vr3_ppc64,
    vmx_vr4_ppc64,      vmx_vr5_ppc64,    vmx_vr6_ppc64,  vmx_vr7_ppc64,
    vmx_vr8_ppc64,      vmx_vr9_ppc64,    vmx_vr10_ppc64, vmx_vr11_ppc64,
    vmx_vr12_ppc64,     vmx_vr13_ppc64,   vmx_vr14_ppc64, vmx_vr15_ppc64,
    vmx_vr16_ppc64,     vmx_vr17_ppc64,   vmx_vr18_ppc64, vmx_vr19_ppc64,
    vmx_vr20_ppc64,     vmx_vr21_ppc64,   vmx_vr22_ppc64, vmx_vr23_ppc64,
    vmx_vr24_ppc64,     vmx_vr25_ppc64,   vmx_vr26_ppc64, vmx_vr27_ppc64,
    vmx_vr28_ppc64,     vmx_vr29_ppc64,   vmx_vr30_ppc64, vmx_vr31_ppc64,
    vmx_vscr_ppc64,     vmx_vrsave_ppc64,
    LLDB_INVALID_REGNUM // register sets need to end with this flag
};

static const uint32_t g_vsx_regnums_ppc64[] = {
    vsx_vs0_ppc64,      vsx_vs1_ppc64,  vsx_vs2_ppc64,  vsx_vs3_ppc64,
    vsx_vs4_ppc64,      vsx_vs5_ppc64,  vsx_vs6_ppc64,  vsx_vs7_ppc64,
    vsx_vs8_ppc64,      vsx_vs9_ppc64,  vsx_vs10_ppc64, vsx_vs11_ppc64,
    vsx_vs12_ppc64,     vsx_vs13_ppc64, vsx_vs14_ppc64, vsx_vs15_ppc64,
    vsx_vs16_ppc64,     vsx_vs17_ppc64, vsx_vs18_ppc64, vsx_vs19_ppc64,
    vsx_vs20_ppc64,     vsx_vs21_ppc64, vsx_vs22_ppc64, vsx_vs23_ppc64,
    vsx_vs24_ppc64,     vsx_vs25_ppc64, vsx_vs26_ppc64, vsx_vs27_ppc64,
    vsx_vs28_ppc64,     vsx_vs29_ppc64, vsx_vs30_ppc64, vsx_vs31_ppc64,
    vsx_vs32_ppc64,     vsx_vs33_ppc64, vsx_vs34_ppc64, vsx_vs35_ppc64,
    vsx_vs36_ppc64,     vsx_vs37_ppc64, vsx_vs38_ppc64, vsx_vs39_ppc64,
    vsx_vs40_ppc64,     vsx_vs41_ppc64, vsx_vs42_ppc64, vsx_vs43_ppc64,
    vsx_vs44_ppc64,     vsx_vs45_ppc64, vsx_vs46_ppc64, vsx_vs47_ppc64,
    vsx_vs48_ppc64,     vsx_vs49_ppc64, vsx_vs50_ppc64, vsx_vs51_ppc64,
    vsx_vs52_ppc64,     vsx_vs53_ppc64, vsx_vs54_ppc64, vsx_vs55_ppc64,
    vsx_vs56_ppc64,     vsx_vs57_ppc64, vsx_vs58_ppc64, vsx_vs59_ppc64,
    vsx_vs60_ppc64,     vsx_vs61_ppc64, vsx_vs62_ppc64, vsx_vs63_ppc64,
    LLDB_INVALID_REGNUM // register sets need to end with this flag
};

// Number of register sets provided by this context.
static constexpr int k_num_register_sets = 4;

static const RegisterSet g_reg_sets_ppc64[k_num_register_sets] = {
    {"General Purpose Registers", "gpr", k_num_gpr_registers_ppc64,
     g_gpr_regnums_ppc64},
    {"Floating Point Registers", "fpr", k_num_fpr_registers_ppc64,
     g_fpr_regnums_ppc64},
    {"AltiVec/VMX Registers", "vmx", k_num_vmx_registers_ppc64,
     g_vmx_regnums_ppc64},
    {"VSX Registers", "vsx", k_num_vsx_registers_ppc64, g_vsx_regnums_ppc64},
};

NativeRegisterContextAIX_ppc64::NativeRegisterContextAIX_ppc64(
    const ArchSpec &target_arch, NativeThreadProtocol &native_thread)
    : NativeRegisterContextAIX(native_thread) {
  if (target_arch.GetMachine() != llvm::Triple::ppc64)
    llvm_unreachable("Unhandled target architecture.");
}

uint32_t NativeRegisterContextAIX_ppc64::GetRegisterSetCount() const {
  return k_num_register_sets;
}

const RegisterSet *
NativeRegisterContextAIX_ppc64::GetRegisterSet(uint32_t set_index) const {
  if (set_index < k_num_register_sets)
    return &g_reg_sets_ppc64[set_index];

  return nullptr;
}

uint32_t NativeRegisterContextAIX_ppc64::GetUserRegisterCount() const {
  uint32_t count = 0;
  for (uint32_t set_index = 0; set_index < k_num_register_sets; ++set_index)
    count += g_reg_sets_ppc64[set_index].num_registers;
  return count;
}

Status
NativeRegisterContextAIX_ppc64::ReadRegister(const RegisterInfo *reg_info,
                                             RegisterValue &reg_value) {
  return Status("unimplemented");
}

Status
NativeRegisterContextAIX_ppc64::WriteRegister(const RegisterInfo *reg_info,
                                              const RegisterValue &reg_value) {
  return Status("unimplemented");
}

Status NativeRegisterContextAIX_ppc64::ReadAllRegisterValues(
    lldb::WritableDataBufferSP &data_sp) {
  return Status("unimplemented");
}

Status NativeRegisterContextAIX_ppc64::WriteAllRegisterValues(
    const lldb::DataBufferSP &data_sp) {
  return Status("unimplemented");
}

bool NativeRegisterContextAIX_ppc64::IsGPR(unsigned reg) const {
  return reg <= k_last_gpr_ppc64;
}

bool NativeRegisterContextAIX_ppc64::IsFPR(unsigned reg) const {
  return (k_first_fpr_ppc64 <= reg && reg <= k_last_fpr_ppc64);
}

bool NativeRegisterContextAIX_ppc64::IsVMX(unsigned reg) const {
  return (reg >= k_first_vmx_ppc64) && (reg <= k_last_vmx_ppc64);
}

bool NativeRegisterContextAIX_ppc64::IsVSX(unsigned reg) const {
  return (reg >= k_first_vsx_ppc64) && (reg <= k_last_vsx_ppc64);
}

uint32_t NativeRegisterContextAIX_ppc64::CalculateFprOffset(
    const RegisterInfo *reg_info) const {
  return 0;
}

uint32_t NativeRegisterContextAIX_ppc64::CalculateVmxOffset(
    const RegisterInfo *reg_info) const {
  return 0;
}

uint32_t NativeRegisterContextAIX_ppc64::CalculateVsxOffset(
    const RegisterInfo *reg_info) const {
  return 0;
}

#endif // defined(__powerpc64__)

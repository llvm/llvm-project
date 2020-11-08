//===-- NativeRegisterContextFreeBSD_x86_64.cpp ---------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#if defined(__i386__) || defined(__x86_64__)

#include "NativeRegisterContextFreeBSD_x86_64.h"

// clang-format off
#include <x86/fpu.h>
#include <x86/specialreg.h>
#include <cpuid.h>
// clang-format on

#include "lldb/Host/HostInfo.h"
#include "lldb/Utility/DataBufferHeap.h"
#include "lldb/Utility/Log.h"
#include "lldb/Utility/RegisterValue.h"
#include "lldb/Utility/Status.h"

#include "NativeProcessFreeBSD.h"
#include "Plugins/Process/Utility/RegisterContextFreeBSD_i386.h"
#include "Plugins/Process/Utility/RegisterContextFreeBSD_x86_64.h"

using namespace lldb_private;
using namespace lldb_private::process_freebsd;

// Private namespace.

namespace {
// x86 64-bit general purpose registers.
static const uint32_t g_gpr_regnums_x86_64[] = {
    lldb_rax_x86_64,    lldb_rbx_x86_64,    lldb_rcx_x86_64, lldb_rdx_x86_64,
    lldb_rdi_x86_64,    lldb_rsi_x86_64,    lldb_rbp_x86_64, lldb_rsp_x86_64,
    lldb_r8_x86_64,     lldb_r9_x86_64,     lldb_r10_x86_64, lldb_r11_x86_64,
    lldb_r12_x86_64,    lldb_r13_x86_64,    lldb_r14_x86_64, lldb_r15_x86_64,
    lldb_rip_x86_64,    lldb_rflags_x86_64, lldb_cs_x86_64,  lldb_fs_x86_64,
    lldb_gs_x86_64,     lldb_ss_x86_64,     lldb_ds_x86_64,  lldb_es_x86_64,
    lldb_eax_x86_64,    lldb_ebx_x86_64,    lldb_ecx_x86_64, lldb_edx_x86_64,
    lldb_edi_x86_64,    lldb_esi_x86_64,    lldb_ebp_x86_64, lldb_esp_x86_64,
    lldb_r8d_x86_64,  // Low 32 bits or r8
    lldb_r9d_x86_64,  // Low 32 bits or r9
    lldb_r10d_x86_64, // Low 32 bits or r10
    lldb_r11d_x86_64, // Low 32 bits or r11
    lldb_r12d_x86_64, // Low 32 bits or r12
    lldb_r13d_x86_64, // Low 32 bits or r13
    lldb_r14d_x86_64, // Low 32 bits or r14
    lldb_r15d_x86_64, // Low 32 bits or r15
    lldb_ax_x86_64,     lldb_bx_x86_64,     lldb_cx_x86_64,  lldb_dx_x86_64,
    lldb_di_x86_64,     lldb_si_x86_64,     lldb_bp_x86_64,  lldb_sp_x86_64,
    lldb_r8w_x86_64,  // Low 16 bits or r8
    lldb_r9w_x86_64,  // Low 16 bits or r9
    lldb_r10w_x86_64, // Low 16 bits or r10
    lldb_r11w_x86_64, // Low 16 bits or r11
    lldb_r12w_x86_64, // Low 16 bits or r12
    lldb_r13w_x86_64, // Low 16 bits or r13
    lldb_r14w_x86_64, // Low 16 bits or r14
    lldb_r15w_x86_64, // Low 16 bits or r15
    lldb_ah_x86_64,     lldb_bh_x86_64,     lldb_ch_x86_64,  lldb_dh_x86_64,
    lldb_al_x86_64,     lldb_bl_x86_64,     lldb_cl_x86_64,  lldb_dl_x86_64,
    lldb_dil_x86_64,    lldb_sil_x86_64,    lldb_bpl_x86_64, lldb_spl_x86_64,
    lldb_r8l_x86_64,    // Low 8 bits or r8
    lldb_r9l_x86_64,    // Low 8 bits or r9
    lldb_r10l_x86_64,   // Low 8 bits or r10
    lldb_r11l_x86_64,   // Low 8 bits or r11
    lldb_r12l_x86_64,   // Low 8 bits or r12
    lldb_r13l_x86_64,   // Low 8 bits or r13
    lldb_r14l_x86_64,   // Low 8 bits or r14
    lldb_r15l_x86_64,   // Low 8 bits or r15
    LLDB_INVALID_REGNUM // register sets need to end with this flag
};
static_assert((sizeof(g_gpr_regnums_x86_64) / sizeof(g_gpr_regnums_x86_64[0])) -
                      1 ==
                  k_num_gpr_registers_x86_64,
              "g_gpr_regnums_x86_64 has wrong number of register infos");

// x86 64-bit floating point registers.
static const uint32_t g_fpu_regnums_x86_64[] = {
    lldb_fctrl_x86_64,     lldb_fstat_x86_64, lldb_ftag_x86_64,
    lldb_fop_x86_64,       lldb_fiseg_x86_64, lldb_fioff_x86_64,
    lldb_foseg_x86_64,     lldb_fooff_x86_64, lldb_mxcsr_x86_64,
    lldb_mxcsrmask_x86_64, lldb_st0_x86_64,   lldb_st1_x86_64,
    lldb_st2_x86_64,       lldb_st3_x86_64,   lldb_st4_x86_64,
    lldb_st5_x86_64,       lldb_st6_x86_64,   lldb_st7_x86_64,
    lldb_mm0_x86_64,       lldb_mm1_x86_64,   lldb_mm2_x86_64,
    lldb_mm3_x86_64,       lldb_mm4_x86_64,   lldb_mm5_x86_64,
    lldb_mm6_x86_64,       lldb_mm7_x86_64,   lldb_xmm0_x86_64,
    lldb_xmm1_x86_64,      lldb_xmm2_x86_64,  lldb_xmm3_x86_64,
    lldb_xmm4_x86_64,      lldb_xmm5_x86_64,  lldb_xmm6_x86_64,
    lldb_xmm7_x86_64,      lldb_xmm8_x86_64,  lldb_xmm9_x86_64,
    lldb_xmm10_x86_64,     lldb_xmm11_x86_64, lldb_xmm12_x86_64,
    lldb_xmm13_x86_64,     lldb_xmm14_x86_64, lldb_xmm15_x86_64,
    LLDB_INVALID_REGNUM // register sets need to end with this flag
};
static_assert((sizeof(g_fpu_regnums_x86_64) / sizeof(g_fpu_regnums_x86_64[0])) -
                      1 ==
                  k_num_fpr_registers_x86_64,
              "g_fpu_regnums_x86_64 has wrong number of register infos");

// x86 64-bit registers available via XState.
static const uint32_t g_xstate_regnums_x86_64[] = {
    lldb_ymm0_x86_64, lldb_ymm1_x86_64, lldb_ymm2_x86_64, lldb_ymm3_x86_64,
    lldb_ymm4_x86_64, lldb_ymm5_x86_64, lldb_ymm6_x86_64, lldb_ymm7_x86_64,
    lldb_ymm8_x86_64, lldb_ymm9_x86_64, lldb_ymm10_x86_64, lldb_ymm11_x86_64,
    lldb_ymm12_x86_64, lldb_ymm13_x86_64, lldb_ymm14_x86_64, lldb_ymm15_x86_64,
    // Note: we currently do not provide them but this is needed to avoid
    // unnamed groups in SBFrame::GetRegisterContext().
    lldb_bnd0_x86_64, lldb_bnd1_x86_64, lldb_bnd2_x86_64, lldb_bnd3_x86_64,
    lldb_bndcfgu_x86_64, lldb_bndstatus_x86_64,
    LLDB_INVALID_REGNUM // register sets need to end with this flag
};
static_assert((sizeof(g_xstate_regnums_x86_64) /
               sizeof(g_xstate_regnums_x86_64[0])) -
                      1 ==
                  k_num_avx_registers_x86_64 + k_num_mpx_registers_x86_64,
              "g_xstate_regnums_x86_64 has wrong number of register infos");

// x86 debug registers.
static const uint32_t g_dbr_regnums_x86_64[] = {
    lldb_dr0_x86_64,    lldb_dr1_x86_64, lldb_dr2_x86_64, lldb_dr3_x86_64,
    lldb_dr4_x86_64,    lldb_dr5_x86_64, lldb_dr6_x86_64, lldb_dr7_x86_64,
    LLDB_INVALID_REGNUM // register sets need to end with this flag
};
static_assert((sizeof(g_dbr_regnums_x86_64) / sizeof(g_dbr_regnums_x86_64[0])) -
                      1 ==
                  k_num_dbr_registers_x86_64,
              "g_dbr_regnums_x86_64 has wrong number of register infos");

// x86 32-bit general purpose registers.
const uint32_t g_gpr_regnums_i386[] = {
    lldb_eax_i386,      lldb_ebx_i386,    lldb_ecx_i386, lldb_edx_i386,
    lldb_edi_i386,      lldb_esi_i386,    lldb_ebp_i386, lldb_esp_i386,
    lldb_eip_i386,      lldb_eflags_i386, lldb_cs_i386,  lldb_fs_i386,
    lldb_gs_i386,       lldb_ss_i386,     lldb_ds_i386,  lldb_es_i386,
    lldb_ax_i386,       lldb_bx_i386,     lldb_cx_i386,  lldb_dx_i386,
    lldb_di_i386,       lldb_si_i386,     lldb_bp_i386,  lldb_sp_i386,
    lldb_ah_i386,       lldb_bh_i386,     lldb_ch_i386,  lldb_dh_i386,
    lldb_al_i386,       lldb_bl_i386,     lldb_cl_i386,  lldb_dl_i386,
    LLDB_INVALID_REGNUM // register sets need to end with this flag
};
static_assert((sizeof(g_gpr_regnums_i386) / sizeof(g_gpr_regnums_i386[0])) -
                      1 ==
                  k_num_gpr_registers_i386,
              "g_gpr_regnums_i386 has wrong number of register infos");

// x86 32-bit floating point registers.
const uint32_t g_fpu_regnums_i386[] = {
    lldb_fctrl_i386,    lldb_fstat_i386,     lldb_ftag_i386,  lldb_fop_i386,
    lldb_fiseg_i386,    lldb_fioff_i386,     lldb_foseg_i386, lldb_fooff_i386,
    lldb_mxcsr_i386,    lldb_mxcsrmask_i386, lldb_st0_i386,   lldb_st1_i386,
    lldb_st2_i386,      lldb_st3_i386,       lldb_st4_i386,   lldb_st5_i386,
    lldb_st6_i386,      lldb_st7_i386,       lldb_mm0_i386,   lldb_mm1_i386,
    lldb_mm2_i386,      lldb_mm3_i386,       lldb_mm4_i386,   lldb_mm5_i386,
    lldb_mm6_i386,      lldb_mm7_i386,       lldb_xmm0_i386,  lldb_xmm1_i386,
    lldb_xmm2_i386,     lldb_xmm3_i386,      lldb_xmm4_i386,  lldb_xmm5_i386,
    lldb_xmm6_i386,     lldb_xmm7_i386,
    LLDB_INVALID_REGNUM // register sets need to end with this flag
};
static_assert((sizeof(g_fpu_regnums_i386) / sizeof(g_fpu_regnums_i386[0])) -
                      1 ==
                  k_num_fpr_registers_i386,
              "g_fpu_regnums_i386 has wrong number of register infos");

// x86 64-bit registers available via XState.
static const uint32_t g_xstate_regnums_i386[] = {
    lldb_ymm0_i386, lldb_ymm1_i386, lldb_ymm2_i386, lldb_ymm3_i386,
    lldb_ymm4_i386, lldb_ymm5_i386, lldb_ymm6_i386, lldb_ymm7_i386,
    // Note: we currently do not provide them but this is needed to avoid
    // unnamed groups in SBFrame::GetRegisterContext().
    lldb_bnd0_i386, lldb_bnd1_i386, lldb_bnd2_i386, lldb_bnd3_i386,
    lldb_bndcfgu_i386, lldb_bndstatus_i386,
    LLDB_INVALID_REGNUM // register sets need to end with this flag
};
static_assert(
    (sizeof(g_xstate_regnums_i386) / sizeof(g_xstate_regnums_i386[0])) - 1 ==
        k_num_avx_registers_i386 + k_num_mpx_registers_i386,
    "g_xstate_regnums_i386 has wrong number of register infos");

// x86 debug registers.
static const uint32_t g_dbr_regnums_i386[] = {
    lldb_dr0_i386,      lldb_dr1_i386, lldb_dr2_i386, lldb_dr3_i386,
    lldb_dr4_i386,      lldb_dr5_i386, lldb_dr6_i386, lldb_dr7_i386,
    LLDB_INVALID_REGNUM // register sets need to end with this flag
};
static_assert((sizeof(g_dbr_regnums_i386) / sizeof(g_dbr_regnums_i386[0])) -
                      1 ==
                  k_num_dbr_registers_i386,
              "g_dbr_regnums_i386 has wrong number of register infos");

// Number of register sets provided by this context.
enum { k_num_register_sets = 4 };

// Register sets for x86 32-bit.
static const RegisterSet g_reg_sets_i386[k_num_register_sets] = {
    {"General Purpose Registers", "gpr", k_num_gpr_registers_i386,
     g_gpr_regnums_i386},
    {"Floating Point Registers", "fpu", k_num_fpr_registers_i386,
     g_fpu_regnums_i386},
    {"Extended State Registers", "xstate",
     k_num_avx_registers_i386 + k_num_mpx_registers_i386,
     g_xstate_regnums_i386},
    {"Debug Registers", "dbr", k_num_dbr_registers_i386, g_dbr_regnums_i386},
};

// Register sets for x86 64-bit.
static const RegisterSet g_reg_sets_x86_64[k_num_register_sets] = {
    {"General Purpose Registers", "gpr", k_num_gpr_registers_x86_64,
     g_gpr_regnums_x86_64},
    {"Floating Point Registers", "fpu", k_num_fpr_registers_x86_64,
     g_fpu_regnums_x86_64},
    {"Extended State Registers", "xstate",
     k_num_avx_registers_x86_64 + k_num_mpx_registers_x86_64,
     g_xstate_regnums_x86_64},
    {"Debug Registers", "dbr", k_num_dbr_registers_x86_64,
     g_dbr_regnums_x86_64},
};

#define REG_CONTEXT_SIZE (GetRegisterInfoInterface().GetGPRSize())
} // namespace

NativeRegisterContextFreeBSD *
NativeRegisterContextFreeBSD::CreateHostNativeRegisterContextFreeBSD(
    const ArchSpec &target_arch, NativeThreadProtocol &native_thread) {
  return new NativeRegisterContextFreeBSD_x86_64(target_arch, native_thread);
}

// NativeRegisterContextFreeBSD_x86_64 members.

static RegisterInfoInterface *
CreateRegisterInfoInterface(const ArchSpec &target_arch) {
  if (HostInfo::GetArchitecture().GetAddressByteSize() == 4) {
    // 32-bit hosts run with a RegisterContextFreeBSD_i386 context.
    return new RegisterContextFreeBSD_i386(target_arch);
  } else {
    assert((HostInfo::GetArchitecture().GetAddressByteSize() == 8) &&
           "Register setting path assumes this is a 64-bit host");
    // X86_64 hosts know how to work with 64-bit and 32-bit EXEs using the
    // x86_64 register context.
    return new RegisterContextFreeBSD_x86_64(target_arch);
  }
}

NativeRegisterContextFreeBSD_x86_64::NativeRegisterContextFreeBSD_x86_64(
    const ArchSpec &target_arch, NativeThreadProtocol &native_thread)
    : NativeRegisterContextRegisterInfo(
          native_thread, CreateRegisterInfoInterface(target_arch)),
      m_gpr(), m_fpr(), m_dbr() {}

// CONSIDER after local and llgs debugging are merged, register set support can
// be moved into a base x86-64 class with IsRegisterSetAvailable made virtual.
uint32_t NativeRegisterContextFreeBSD_x86_64::GetRegisterSetCount() const {
  uint32_t sets = 0;
  for (uint32_t set_index = 0; set_index < k_num_register_sets; ++set_index) {
    if (GetSetForNativeRegNum(set_index) != -1)
      ++sets;
  }

  return sets;
}

const RegisterSet *
NativeRegisterContextFreeBSD_x86_64::GetRegisterSet(uint32_t set_index) const {
  switch (GetRegisterInfoInterface().GetTargetArchitecture().GetMachine()) {
  case llvm::Triple::x86:
    return &g_reg_sets_i386[set_index];
  case llvm::Triple::x86_64:
    return &g_reg_sets_x86_64[set_index];
  default:
    llvm_unreachable("Unhandled target architecture.");
  }
}

static constexpr int RegNumX86ToX86_64(int regnum) {
  switch (regnum) {
  case lldb_eax_i386:
    return lldb_rax_x86_64;
  case lldb_ebx_i386:
    return lldb_rbx_x86_64;
  case lldb_ecx_i386:
    return lldb_rcx_x86_64;
  case lldb_edx_i386:
    return lldb_rdx_x86_64;
  case lldb_edi_i386:
    return lldb_rdi_x86_64;
  case lldb_esi_i386:
    return lldb_rsi_x86_64;
  case lldb_ebp_i386:
    return lldb_rbp_x86_64;
  case lldb_esp_i386:
    return lldb_rsp_x86_64;
  case lldb_eip_i386:
    return lldb_rip_x86_64;
  case lldb_eflags_i386:
    return lldb_rflags_x86_64;
  case lldb_cs_i386:
    return lldb_cs_x86_64;
  case lldb_fs_i386:
    return lldb_fs_x86_64;
  case lldb_gs_i386:
    return lldb_gs_x86_64;
  case lldb_ss_i386:
    return lldb_ss_x86_64;
  case lldb_ds_i386:
    return lldb_ds_x86_64;
  case lldb_es_i386:
    return lldb_es_x86_64;
  case lldb_fctrl_i386:
    return lldb_fctrl_x86_64;
  case lldb_fstat_i386:
    return lldb_fstat_x86_64;
  case lldb_ftag_i386:
    return lldb_ftag_x86_64;
  case lldb_fop_i386:
    return lldb_fop_x86_64;
  case lldb_fiseg_i386:
    return lldb_fiseg_x86_64;
  case lldb_fioff_i386:
    return lldb_fioff_x86_64;
  case lldb_foseg_i386:
    return lldb_foseg_x86_64;
  case lldb_fooff_i386:
    return lldb_fooff_x86_64;
  case lldb_mxcsr_i386:
    return lldb_mxcsr_x86_64;
  case lldb_mxcsrmask_i386:
    return lldb_mxcsrmask_x86_64;
  case lldb_st0_i386:
  case lldb_st1_i386:
  case lldb_st2_i386:
  case lldb_st3_i386:
  case lldb_st4_i386:
  case lldb_st5_i386:
  case lldb_st6_i386:
  case lldb_st7_i386:
    return lldb_st0_x86_64 + regnum - lldb_st0_i386;
  case lldb_mm0_i386:
  case lldb_mm1_i386:
  case lldb_mm2_i386:
  case lldb_mm3_i386:
  case lldb_mm4_i386:
  case lldb_mm5_i386:
  case lldb_mm6_i386:
  case lldb_mm7_i386:
    return lldb_mm0_x86_64 + regnum - lldb_mm0_i386;
  case lldb_xmm0_i386:
  case lldb_xmm1_i386:
  case lldb_xmm2_i386:
  case lldb_xmm3_i386:
  case lldb_xmm4_i386:
  case lldb_xmm5_i386:
  case lldb_xmm6_i386:
  case lldb_xmm7_i386:
    return lldb_xmm0_x86_64 + regnum - lldb_xmm0_i386;
  case lldb_ymm0_i386:
  case lldb_ymm1_i386:
  case lldb_ymm2_i386:
  case lldb_ymm3_i386:
  case lldb_ymm4_i386:
  case lldb_ymm5_i386:
  case lldb_ymm6_i386:
  case lldb_ymm7_i386:
    return lldb_ymm0_x86_64 + regnum - lldb_ymm0_i386;
  case lldb_bnd0_i386:
  case lldb_bnd1_i386:
  case lldb_bnd2_i386:
  case lldb_bnd3_i386:
    return lldb_bnd0_x86_64 + regnum - lldb_bnd0_i386;
  case lldb_bndcfgu_i386:
    return lldb_bndcfgu_x86_64;
  case lldb_bndstatus_i386:
    return lldb_bndstatus_x86_64;
  case lldb_dr0_i386:
  case lldb_dr1_i386:
  case lldb_dr2_i386:
  case lldb_dr3_i386:
  case lldb_dr4_i386:
  case lldb_dr5_i386:
  case lldb_dr6_i386:
  case lldb_dr7_i386:
    return lldb_dr0_x86_64 + regnum - lldb_dr0_i386;
  default:
    llvm_unreachable("Unhandled i386 register.");
  }
}

int NativeRegisterContextFreeBSD_x86_64::GetSetForNativeRegNum(
    int reg_num) const {

  switch (GetRegisterInfoInterface().GetTargetArchitecture().GetMachine()) {
  case llvm::Triple::x86:
    if (reg_num >= k_first_gpr_i386 && reg_num <= k_last_gpr_i386)
      return GPRegSet;
    if (reg_num >= k_first_fpr_i386 && reg_num <= k_last_fpr_i386)
      return FPRegSet;
    if (reg_num >= k_first_avx_i386 && reg_num <= k_last_avx_i386)
      return XSaveRegSet; // AVX
    if (reg_num >= k_first_mpxr_i386 && reg_num <= k_last_mpxr_i386)
      return -1; // MPXR
    if (reg_num >= k_first_mpxc_i386 && reg_num <= k_last_mpxc_i386)
      return -1; // MPXC
    if (reg_num >= k_first_dbr_i386 && reg_num <= k_last_dbr_i386)
      return DBRegSet; // DBR
    break;
  case llvm::Triple::x86_64:
    if (reg_num >= k_first_gpr_x86_64 && reg_num <= k_last_gpr_x86_64)
      return GPRegSet;
    if (reg_num >= k_first_fpr_x86_64 && reg_num <= k_last_fpr_x86_64)
      return FPRegSet;
    if (reg_num >= k_first_avx_x86_64 && reg_num <= k_last_avx_x86_64)
      return XSaveRegSet; // AVX
    if (reg_num >= k_first_mpxr_x86_64 && reg_num <= k_last_mpxr_x86_64)
      return -1; // MPXR
    if (reg_num >= k_first_mpxc_x86_64 && reg_num <= k_last_mpxc_x86_64)
      return -1; // MPXC
    if (reg_num >= k_first_dbr_x86_64 && reg_num <= k_last_dbr_x86_64)
      return DBRegSet; // DBR
    break;
  default:
    llvm_unreachable("Unhandled target architecture.");
  }

  llvm_unreachable("Register does not belong to any register set");
}

Status NativeRegisterContextFreeBSD_x86_64::ReadRegisterSet(uint32_t set) {
  switch (set) {
  case GPRegSet:
    return NativeProcessFreeBSD::PtraceWrapper(PT_GETREGS, m_thread.GetID(),
                                               &m_gpr);
  case FPRegSet:
#if defined(__x86_64__)
    return NativeProcessFreeBSD::PtraceWrapper(PT_GETFPREGS, m_thread.GetID(),
                                               &m_fpr);
#else
    return NativeProcessFreeBSD::PtraceWrapper(PT_GETXMMREGS, m_thread.GetID(),
                                               &m_fpr);
#endif
  case DBRegSet:
    return NativeProcessFreeBSD::PtraceWrapper(PT_GETDBREGS, m_thread.GetID(),
                                               &m_dbr);
  case XSaveRegSet: {
    struct ptrace_xstate_info info;
    Status ret = NativeProcessFreeBSD::PtraceWrapper(
        PT_GETXSTATE_INFO, GetProcessPid(), &info, sizeof(info));
    if (!ret.Success())
      return ret;

    assert(info.xsave_mask & XFEATURE_ENABLED_X87);
    assert(info.xsave_mask & XFEATURE_ENABLED_SSE);

    m_xsave_offsets[YMMXSaveSet] = LLDB_INVALID_XSAVE_OFFSET;
    if (info.xsave_mask & XFEATURE_ENABLED_YMM_HI128) {
      uint32_t eax, ecx, edx;
      __get_cpuid_count(0x0D, 2, &eax, &m_xsave_offsets[YMMXSaveSet], &ecx,
                        &edx);
    }

    m_xsave.resize(info.xsave_len);
    return NativeProcessFreeBSD::PtraceWrapper(PT_GETXSTATE, GetProcessPid(),
                                               m_xsave.data(), m_xsave.size());
  }
  }
  llvm_unreachable("NativeRegisterContextFreeBSD_x86_64::ReadRegisterSet");
}

Status NativeRegisterContextFreeBSD_x86_64::WriteRegisterSet(uint32_t set) {
  switch (set) {
  case GPRegSet:
    return NativeProcessFreeBSD::PtraceWrapper(PT_SETREGS, m_thread.GetID(),
                                               &m_gpr);
  case FPRegSet:
#if defined(__x86_64__)
    return NativeProcessFreeBSD::PtraceWrapper(PT_SETFPREGS, m_thread.GetID(),
                                               &m_fpr);
#else
    return NativeProcessFreeBSD::PtraceWrapper(PT_SETXMMREGS, m_thread.GetID(),
                                               &m_fpr);
#endif
  case DBRegSet:
    return NativeProcessFreeBSD::PtraceWrapper(PT_SETDBREGS, m_thread.GetID(),
                                               &m_dbr);
  case XSaveRegSet:
    // ReadRegisterSet() must always be called before WriteRegisterSet().
    assert(m_xsave.size() > 0);
    return NativeProcessFreeBSD::PtraceWrapper(PT_SETXSTATE, GetProcessPid(),
                                               m_xsave.data(), m_xsave.size());
  }
  llvm_unreachable("NativeRegisterContextFreeBSD_x86_64::WriteRegisterSet");
}

Status
NativeRegisterContextFreeBSD_x86_64::ReadRegister(const RegisterInfo *reg_info,
                                                  RegisterValue &reg_value) {
  Status error;

  if (!reg_info) {
    error.SetErrorString("reg_info NULL");
    return error;
  }

  uint32_t reg = reg_info->kinds[lldb::eRegisterKindLLDB];
  if (reg == LLDB_INVALID_REGNUM) {
    // This is likely an internal register for lldb use only and should not be
    // directly queried.
    error.SetErrorStringWithFormat("register \"%s\" is an internal-only lldb "
                                   "register, cannot read directly",
                                   reg_info->name);
    return error;
  }

  int set = GetSetForNativeRegNum(reg);
  if (set == -1) {
    // This is likely an internal register for lldb use only and should not be
    // directly queried.
    error.SetErrorStringWithFormat("register \"%s\" is in unrecognized set",
                                   reg_info->name);
    return error;
  }

  switch (GetRegisterInfoInterface().GetTargetArchitecture().GetMachine()) {
  case llvm::Triple::x86_64:
    break;
  case llvm::Triple::x86:
    reg = RegNumX86ToX86_64(reg);
    break;
  default:
    llvm_unreachable("Unhandled target architecture.");
  }

  error = ReadRegisterSet(set);
  if (error.Fail())
    return error;

  switch (reg) {
#if defined(__x86_64__)
  case lldb_rax_x86_64:
    reg_value = (uint64_t)m_gpr.r_rax;
    break;
  case lldb_rbx_x86_64:
    reg_value = (uint64_t)m_gpr.r_rbx;
    break;
  case lldb_rcx_x86_64:
    reg_value = (uint64_t)m_gpr.r_rcx;
    break;
  case lldb_rdx_x86_64:
    reg_value = (uint64_t)m_gpr.r_rdx;
    break;
  case lldb_rdi_x86_64:
    reg_value = (uint64_t)m_gpr.r_rdi;
    break;
  case lldb_rsi_x86_64:
    reg_value = (uint64_t)m_gpr.r_rsi;
    break;
  case lldb_rbp_x86_64:
    reg_value = (uint64_t)m_gpr.r_rbp;
    break;
  case lldb_rsp_x86_64:
    reg_value = (uint64_t)m_gpr.r_rsp;
    break;
  case lldb_r8_x86_64:
    reg_value = (uint64_t)m_gpr.r_r8;
    break;
  case lldb_r9_x86_64:
    reg_value = (uint64_t)m_gpr.r_r9;
    break;
  case lldb_r10_x86_64:
    reg_value = (uint64_t)m_gpr.r_r10;
    break;
  case lldb_r11_x86_64:
    reg_value = (uint64_t)m_gpr.r_r11;
    break;
  case lldb_r12_x86_64:
    reg_value = (uint64_t)m_gpr.r_r12;
    break;
  case lldb_r13_x86_64:
    reg_value = (uint64_t)m_gpr.r_r13;
    break;
  case lldb_r14_x86_64:
    reg_value = (uint64_t)m_gpr.r_r14;
    break;
  case lldb_r15_x86_64:
    reg_value = (uint64_t)m_gpr.r_r15;
    break;
  case lldb_rip_x86_64:
    reg_value = (uint64_t)m_gpr.r_rip;
    break;
  case lldb_rflags_x86_64:
    reg_value = (uint64_t)m_gpr.r_rflags;
    break;
  case lldb_cs_x86_64:
    reg_value = (uint64_t)m_gpr.r_cs;
    break;
  case lldb_fs_x86_64:
    reg_value = (uint16_t)m_gpr.r_fs;
    break;
  case lldb_gs_x86_64:
    reg_value = (uint16_t)m_gpr.r_gs;
    break;
  case lldb_ss_x86_64:
    reg_value = (uint64_t)m_gpr.r_ss;
    break;
  case lldb_ds_x86_64:
    reg_value = (uint16_t)m_gpr.r_ds;
    break;
  case lldb_es_x86_64:
    reg_value = (uint16_t)m_gpr.r_es;
    break;
#else
  case lldb_rax_x86_64:
    reg_value = (uint32_t)m_gpr.r_eax;
    break;
  case lldb_rbx_x86_64:
    reg_value = (uint32_t)m_gpr.r_ebx;
    break;
  case lldb_rcx_x86_64:
    reg_value = (uint32_t)m_gpr.r_ecx;
    break;
  case lldb_rdx_x86_64:
    reg_value = (uint32_t)m_gpr.r_edx;
    break;
  case lldb_rdi_x86_64:
    reg_value = (uint32_t)m_gpr.r_edi;
    break;
  case lldb_rsi_x86_64:
    reg_value = (uint32_t)m_gpr.r_esi;
    break;
  case lldb_rsp_x86_64:
    reg_value = (uint32_t)m_gpr.r_esp;
    break;
  case lldb_rbp_x86_64:
    reg_value = (uint32_t)m_gpr.r_ebp;
    break;
  case lldb_rip_x86_64:
    reg_value = (uint32_t)m_gpr.r_eip;
    break;
  case lldb_rflags_x86_64:
    reg_value = (uint32_t)m_gpr.r_eflags;
    break;
  case lldb_cs_x86_64:
    reg_value = (uint32_t)m_gpr.r_cs;
    break;
  case lldb_fs_x86_64:
    reg_value = (uint16_t)m_gpr.r_fs;
    break;
  case lldb_gs_x86_64:
    reg_value = (uint16_t)m_gpr.r_gs;
    break;
  case lldb_ss_x86_64:
    reg_value = (uint32_t)m_gpr.r_ss;
    break;
  case lldb_ds_x86_64:
    reg_value = (uint16_t)m_gpr.r_ds;
    break;
  case lldb_es_x86_64:
    reg_value = (uint16_t)m_gpr.r_es;
    break;
#endif
#if defined(__x86_64__)
// the 32-bit field carries more detail, so we don't have to reinvent
// the wheel
#define FPR_ENV(x) ((struct envxmm32 *)m_fpr.fpr_env)->x
#else
#define FPR_ENV(x) ((struct envxmm *)m_fpr.xmm_env)->x
#endif
  case lldb_fctrl_x86_64:
    reg_value = (uint16_t)FPR_ENV(en_cw);
    break;
  case lldb_fstat_x86_64:
    reg_value = (uint16_t)FPR_ENV(en_sw);
    break;
  case lldb_ftag_x86_64:
    reg_value = (uint16_t)FPR_ENV(en_tw);
    break;
  case lldb_fop_x86_64:
    reg_value = (uint16_t)FPR_ENV(en_opcode);
    break;
  case lldb_fiseg_x86_64:
    reg_value = (uint32_t)FPR_ENV(en_fcs);
    break;
  case lldb_fioff_x86_64:
    reg_value = (uint32_t)FPR_ENV(en_fip);
    break;
  case lldb_foseg_x86_64:
    reg_value = (uint32_t)FPR_ENV(en_fos);
    break;
  case lldb_fooff_x86_64:
    reg_value = (uint32_t)FPR_ENV(en_foo);
    break;
  case lldb_mxcsr_x86_64:
    reg_value = (uint32_t)FPR_ENV(en_mxcsr);
    break;
  case lldb_mxcsrmask_x86_64:
    reg_value = (uint32_t)FPR_ENV(en_mxcsr_mask);
    break;
  case lldb_st0_x86_64:
  case lldb_st1_x86_64:
  case lldb_st2_x86_64:
  case lldb_st3_x86_64:
  case lldb_st4_x86_64:
  case lldb_st5_x86_64:
  case lldb_st6_x86_64:
  case lldb_st7_x86_64:
#if defined(__x86_64__)
    reg_value.SetBytes(&m_fpr.fpr_acc[reg - lldb_st0_x86_64],
                       reg_info->byte_size, endian::InlHostByteOrder());
#else
    reg_value.SetBytes(&m_fpr.xmm_acc[reg - lldb_st0_x86_64],
                       reg_info->byte_size, endian::InlHostByteOrder());
#endif
    break;
  case lldb_mm0_x86_64:
  case lldb_mm1_x86_64:
  case lldb_mm2_x86_64:
  case lldb_mm3_x86_64:
  case lldb_mm4_x86_64:
  case lldb_mm5_x86_64:
  case lldb_mm6_x86_64:
  case lldb_mm7_x86_64:
#if defined(__x86_64__)
    reg_value.SetBytes(&m_fpr.fpr_acc[reg - lldb_mm0_x86_64],
                       reg_info->byte_size, endian::InlHostByteOrder());
#else
    reg_value.SetBytes(&m_fpr.xmm_acc[reg - lldb_mm0_x86_64],
                       reg_info->byte_size, endian::InlHostByteOrder());
#endif
    break;
  case lldb_xmm0_x86_64:
  case lldb_xmm1_x86_64:
  case lldb_xmm2_x86_64:
  case lldb_xmm3_x86_64:
  case lldb_xmm4_x86_64:
  case lldb_xmm5_x86_64:
  case lldb_xmm6_x86_64:
  case lldb_xmm7_x86_64:
  case lldb_xmm8_x86_64:
  case lldb_xmm9_x86_64:
  case lldb_xmm10_x86_64:
  case lldb_xmm11_x86_64:
  case lldb_xmm12_x86_64:
  case lldb_xmm13_x86_64:
  case lldb_xmm14_x86_64:
  case lldb_xmm15_x86_64:
#if defined(__x86_64__)
    reg_value.SetBytes(&m_fpr.fpr_xacc[reg - lldb_xmm0_x86_64],
                       reg_info->byte_size, endian::InlHostByteOrder());
#else
    reg_value.SetBytes(&m_fpr.xmm_reg[reg - lldb_xmm0_x86_64],
                       reg_info->byte_size, endian::InlHostByteOrder());
#endif
    break;
  case lldb_ymm0_x86_64:
  case lldb_ymm1_x86_64:
  case lldb_ymm2_x86_64:
  case lldb_ymm3_x86_64:
  case lldb_ymm4_x86_64:
  case lldb_ymm5_x86_64:
  case lldb_ymm6_x86_64:
  case lldb_ymm7_x86_64:
  case lldb_ymm8_x86_64:
  case lldb_ymm9_x86_64:
  case lldb_ymm10_x86_64:
  case lldb_ymm11_x86_64:
  case lldb_ymm12_x86_64:
  case lldb_ymm13_x86_64:
  case lldb_ymm14_x86_64:
  case lldb_ymm15_x86_64: {
    uint32_t offset = m_xsave_offsets[YMMXSaveSet];
    if (offset == LLDB_INVALID_XSAVE_OFFSET) {
      error.SetErrorStringWithFormat(
          "register \"%s\" not supported by CPU/kernel", reg_info->name);
    } else {
      uint32_t reg_index = reg - lldb_ymm0_x86_64;
      auto *fpreg = reinterpret_cast<struct savexmm_ymm *>(m_xsave.data());
      auto *ymmreg = reinterpret_cast<struct ymmacc *>(m_xsave.data() + offset);

      YMMReg ymm =
          XStateToYMM(reinterpret_cast<void *>(&fpreg->sv_xmm[reg_index]),
                      reinterpret_cast<void *>(&ymmreg[reg_index]));
      reg_value.SetBytes(ymm.bytes, reg_info->byte_size,
                         endian::InlHostByteOrder());
    }
    break;
  }
  case lldb_dr0_x86_64:
  case lldb_dr1_x86_64:
  case lldb_dr2_x86_64:
  case lldb_dr3_x86_64:
  case lldb_dr4_x86_64:
  case lldb_dr5_x86_64:
  case lldb_dr6_x86_64:
  case lldb_dr7_x86_64:
    reg_value = (uint64_t)m_dbr.dr[reg - lldb_dr0_x86_64];
    break;
  default:
    llvm_unreachable("Reading unknown/unsupported register");
  }

  return error;
}

Status NativeRegisterContextFreeBSD_x86_64::WriteRegister(
    const RegisterInfo *reg_info, const RegisterValue &reg_value) {

  Status error;

  if (!reg_info) {
    error.SetErrorString("reg_info NULL");
    return error;
  }

  uint32_t reg = reg_info->kinds[lldb::eRegisterKindLLDB];
  if (reg == LLDB_INVALID_REGNUM) {
    // This is likely an internal register for lldb use only and should not be
    // directly queried.
    error.SetErrorStringWithFormat("register \"%s\" is an internal-only lldb "
                                   "register, cannot read directly",
                                   reg_info->name);
    return error;
  }

  int set = GetSetForNativeRegNum(reg);
  if (set == -1) {
    // This is likely an internal register for lldb use only and should not be
    // directly queried.
    error.SetErrorStringWithFormat("register \"%s\" is in unrecognized set",
                                   reg_info->name);
    return error;
  }

  switch (GetRegisterInfoInterface().GetTargetArchitecture().GetMachine()) {
  case llvm::Triple::x86_64:
    break;
  case llvm::Triple::x86:
    reg = RegNumX86ToX86_64(reg);
    break;
  default:
    llvm_unreachable("Unhandled target architecture.");
  }

  error = ReadRegisterSet(set);
  if (error.Fail())
    return error;

  switch (reg) {
#if defined(__x86_64__)
  case lldb_rax_x86_64:
    m_gpr.r_rax = reg_value.GetAsUInt64();
    break;
  case lldb_rbx_x86_64:
    m_gpr.r_rbx = reg_value.GetAsUInt64();
    break;
  case lldb_rcx_x86_64:
    m_gpr.r_rcx = reg_value.GetAsUInt64();
    break;
  case lldb_rdx_x86_64:
    m_gpr.r_rdx = reg_value.GetAsUInt64();
    break;
  case lldb_rdi_x86_64:
    m_gpr.r_rdi = reg_value.GetAsUInt64();
    break;
  case lldb_rsi_x86_64:
    m_gpr.r_rsi = reg_value.GetAsUInt64();
    break;
  case lldb_rbp_x86_64:
    m_gpr.r_rbp = reg_value.GetAsUInt64();
    break;
  case lldb_rsp_x86_64:
    m_gpr.r_rsp = reg_value.GetAsUInt64();
    break;
  case lldb_r8_x86_64:
    m_gpr.r_r8 = reg_value.GetAsUInt64();
    break;
  case lldb_r9_x86_64:
    m_gpr.r_r9 = reg_value.GetAsUInt64();
    break;
  case lldb_r10_x86_64:
    m_gpr.r_r10 = reg_value.GetAsUInt64();
    break;
  case lldb_r11_x86_64:
    m_gpr.r_r11 = reg_value.GetAsUInt64();
    break;
  case lldb_r12_x86_64:
    m_gpr.r_r12 = reg_value.GetAsUInt64();
    break;
  case lldb_r13_x86_64:
    m_gpr.r_r13 = reg_value.GetAsUInt64();
    break;
  case lldb_r14_x86_64:
    m_gpr.r_r14 = reg_value.GetAsUInt64();
    break;
  case lldb_r15_x86_64:
    m_gpr.r_r15 = reg_value.GetAsUInt64();
    break;
  case lldb_rip_x86_64:
    m_gpr.r_rip = reg_value.GetAsUInt64();
    break;
  case lldb_rflags_x86_64:
    m_gpr.r_rflags = reg_value.GetAsUInt64();
    break;
  case lldb_cs_x86_64:
    m_gpr.r_cs = reg_value.GetAsUInt64();
    break;
  case lldb_fs_x86_64:
    m_gpr.r_fs = reg_value.GetAsUInt16();
    break;
  case lldb_gs_x86_64:
    m_gpr.r_gs = reg_value.GetAsUInt16();
    break;
  case lldb_ss_x86_64:
    m_gpr.r_ss = reg_value.GetAsUInt64();
    break;
  case lldb_ds_x86_64:
    m_gpr.r_ds = reg_value.GetAsUInt16();
    break;
  case lldb_es_x86_64:
    m_gpr.r_es = reg_value.GetAsUInt16();
    break;
#else
  case lldb_rax_x86_64:
    m_gpr.r_eax = reg_value.GetAsUInt32();
    break;
  case lldb_rbx_x86_64:
    m_gpr.r_ebx = reg_value.GetAsUInt32();
    break;
  case lldb_rcx_x86_64:
    m_gpr.r_ecx = reg_value.GetAsUInt32();
    break;
  case lldb_rdx_x86_64:
    m_gpr.r_edx = reg_value.GetAsUInt32();
    break;
  case lldb_rdi_x86_64:
    m_gpr.r_edi = reg_value.GetAsUInt32();
    break;
  case lldb_rsi_x86_64:
    m_gpr.r_esi = reg_value.GetAsUInt32();
    break;
  case lldb_rsp_x86_64:
    m_gpr.r_esp = reg_value.GetAsUInt32();
    break;
  case lldb_rbp_x86_64:
    m_gpr.r_ebp = reg_value.GetAsUInt32();
    break;
  case lldb_rip_x86_64:
    m_gpr.r_eip = reg_value.GetAsUInt32();
    break;
  case lldb_rflags_x86_64:
    m_gpr.r_eflags = reg_value.GetAsUInt32();
    break;
  case lldb_cs_x86_64:
    m_gpr.r_cs = reg_value.GetAsUInt32();
    break;
  case lldb_fs_x86_64:
    m_gpr.r_fs = reg_value.GetAsUInt16();
    break;
  case lldb_gs_x86_64:
    m_gpr.r_gs = reg_value.GetAsUInt16();
    break;
  case lldb_ss_x86_64:
    m_gpr.r_ss = reg_value.GetAsUInt32();
    break;
  case lldb_ds_x86_64:
    m_gpr.r_ds = reg_value.GetAsUInt16();
    break;
  case lldb_es_x86_64:
    m_gpr.r_es = reg_value.GetAsUInt16();
    break;
#endif
  case lldb_fctrl_x86_64:
    FPR_ENV(en_cw) = reg_value.GetAsUInt16();
    break;
  case lldb_fstat_x86_64:
    FPR_ENV(en_sw) = reg_value.GetAsUInt16();
    break;
  case lldb_ftag_x86_64:
    FPR_ENV(en_tw) = reg_value.GetAsUInt16();
    break;
  case lldb_fop_x86_64:
    FPR_ENV(en_opcode) = reg_value.GetAsUInt16();
    break;
  case lldb_fiseg_x86_64:
    FPR_ENV(en_fcs) = reg_value.GetAsUInt32();
    break;
  case lldb_fioff_x86_64:
    FPR_ENV(en_fip) = reg_value.GetAsUInt32();
    break;
  case lldb_foseg_x86_64:
    FPR_ENV(en_fos) = reg_value.GetAsUInt32();
    break;
  case lldb_fooff_x86_64:
    FPR_ENV(en_foo) = reg_value.GetAsUInt32();
    break;
  case lldb_mxcsr_x86_64:
    FPR_ENV(en_mxcsr) = reg_value.GetAsUInt32();
    break;
  case lldb_mxcsrmask_x86_64:
    FPR_ENV(en_mxcsr_mask) = reg_value.GetAsUInt32();
    break;
  case lldb_st0_x86_64:
  case lldb_st1_x86_64:
  case lldb_st2_x86_64:
  case lldb_st3_x86_64:
  case lldb_st4_x86_64:
  case lldb_st5_x86_64:
  case lldb_st6_x86_64:
  case lldb_st7_x86_64:
#if defined(__x86_64__)
    ::memcpy(&m_fpr.fpr_acc[reg - lldb_st0_x86_64], reg_value.GetBytes(),
             reg_value.GetByteSize());
#else
    ::memcpy(&m_fpr.xmm_acc[reg - lldb_st0_x86_64], reg_value.GetBytes(),
             reg_value.GetByteSize());
#endif
    break;
  case lldb_mm0_x86_64:
  case lldb_mm1_x86_64:
  case lldb_mm2_x86_64:
  case lldb_mm3_x86_64:
  case lldb_mm4_x86_64:
  case lldb_mm5_x86_64:
  case lldb_mm6_x86_64:
  case lldb_mm7_x86_64:
#if defined(__x86_64__)
    ::memcpy(&m_fpr.fpr_acc[reg - lldb_mm0_x86_64], reg_value.GetBytes(),
             reg_value.GetByteSize());
#else
    ::memcpy(&m_fpr.xmm_acc[reg - lldb_mm0_x86_64], reg_value.GetBytes(),
             reg_value.GetByteSize());
#endif
    break;
  case lldb_xmm0_x86_64:
  case lldb_xmm1_x86_64:
  case lldb_xmm2_x86_64:
  case lldb_xmm3_x86_64:
  case lldb_xmm4_x86_64:
  case lldb_xmm5_x86_64:
  case lldb_xmm6_x86_64:
  case lldb_xmm7_x86_64:
  case lldb_xmm8_x86_64:
  case lldb_xmm9_x86_64:
  case lldb_xmm10_x86_64:
  case lldb_xmm11_x86_64:
  case lldb_xmm12_x86_64:
  case lldb_xmm13_x86_64:
  case lldb_xmm14_x86_64:
  case lldb_xmm15_x86_64:
#if defined(__x86_64__)
    ::memcpy(&m_fpr.fpr_xacc[reg - lldb_xmm0_x86_64], reg_value.GetBytes(),
             reg_value.GetByteSize());
#else
    ::memcpy(&m_fpr.xmm_reg[reg - lldb_xmm0_x86_64], reg_value.GetBytes(),
             reg_value.GetByteSize());
#endif
    break;
  case lldb_ymm0_x86_64:
  case lldb_ymm1_x86_64:
  case lldb_ymm2_x86_64:
  case lldb_ymm3_x86_64:
  case lldb_ymm4_x86_64:
  case lldb_ymm5_x86_64:
  case lldb_ymm6_x86_64:
  case lldb_ymm7_x86_64:
  case lldb_ymm8_x86_64:
  case lldb_ymm9_x86_64:
  case lldb_ymm10_x86_64:
  case lldb_ymm11_x86_64:
  case lldb_ymm12_x86_64:
  case lldb_ymm13_x86_64:
  case lldb_ymm14_x86_64:
  case lldb_ymm15_x86_64: {
    uint32_t offset = m_xsave_offsets[YMMXSaveSet];
    if (offset == LLDB_INVALID_XSAVE_OFFSET) {
      error.SetErrorStringWithFormat(
          "register \"%s\" not supported by CPU/kernel", reg_info->name);
    } else {
      uint32_t reg_index = reg - lldb_ymm0_x86_64;
      auto *fpreg = reinterpret_cast<struct savexmm_ymm *>(m_xsave.data());
      auto *ymmreg = reinterpret_cast<struct ymmacc *>(m_xsave.data() + offset);

      YMMReg ymm;
      ::memcpy(ymm.bytes, reg_value.GetBytes(), reg_value.GetByteSize());
      YMMToXState(ymm, reinterpret_cast<void *>(&fpreg->sv_xmm[reg_index]),
                  reinterpret_cast<void *>(&ymmreg[reg_index]));
    }
    break;
  }
  case lldb_dr0_x86_64:
  case lldb_dr1_x86_64:
  case lldb_dr2_x86_64:
  case lldb_dr3_x86_64:
  case lldb_dr4_x86_64:
  case lldb_dr5_x86_64:
  case lldb_dr6_x86_64:
  case lldb_dr7_x86_64:
    m_dbr.dr[reg - lldb_dr0_x86_64] = reg_value.GetAsUInt64();
    break;
  default:
    llvm_unreachable("Reading unknown/unsupported register");
  }

  return WriteRegisterSet(set);
}

Status NativeRegisterContextFreeBSD_x86_64::ReadAllRegisterValues(
    lldb::DataBufferSP &data_sp) {
  Status error;

  data_sp.reset(new DataBufferHeap(REG_CONTEXT_SIZE, 0));
  error = ReadRegisterSet(GPRegSet);
  if (error.Fail())
    return error;

  uint8_t *dst = data_sp->GetBytes();
  ::memcpy(dst, &m_gpr, GetRegisterInfoInterface().GetGPRSize());
  dst += GetRegisterInfoInterface().GetGPRSize();

  return error;
}

Status NativeRegisterContextFreeBSD_x86_64::WriteAllRegisterValues(
    const lldb::DataBufferSP &data_sp) {
  Status error;

  if (!data_sp) {
    error.SetErrorStringWithFormat(
        "NativeRegisterContextFreeBSD_x86_64::%s invalid data_sp provided",
        __FUNCTION__);
    return error;
  }

  if (data_sp->GetByteSize() != REG_CONTEXT_SIZE) {
    error.SetErrorStringWithFormat(
        "NativeRegisterContextFreeBSD_x86_64::%s data_sp contained mismatched "
        "data size, expected %zu, actual %" PRIu64,
        __FUNCTION__, REG_CONTEXT_SIZE, data_sp->GetByteSize());
    return error;
  }

  uint8_t *src = data_sp->GetBytes();
  if (src == nullptr) {
    error.SetErrorStringWithFormat("NativeRegisterContextFreeBSD_x86_64::%s "
                                   "DataBuffer::GetBytes() returned a null "
                                   "pointer",
                                   __FUNCTION__);
    return error;
  }
  ::memcpy(&m_gpr, src, GetRegisterInfoInterface().GetGPRSize());

  error = WriteRegisterSet(GPRegSet);
  if (error.Fail())
    return error;
  src += GetRegisterInfoInterface().GetGPRSize();

  return error;
}

int NativeRegisterContextFreeBSD_x86_64::GetDR(int num) const {
  assert(num >= 0 && num <= 7);
  switch (GetRegisterInfoInterface().GetTargetArchitecture().GetMachine()) {
  case llvm::Triple::x86:
    return lldb_dr0_i386 + num;
  case llvm::Triple::x86_64:
    return lldb_dr0_x86_64 + num;
  default:
    llvm_unreachable("Unhandled target architecture.");
  }
}

llvm::Error NativeRegisterContextFreeBSD_x86_64::CopyHardwareWatchpointsFrom(
    NativeRegisterContextFreeBSD &source) {
  auto &r_source = static_cast<NativeRegisterContextFreeBSD_x86_64 &>(source);
  Status res = r_source.ReadRegisterSet(DBRegSet);
  if (!res.Fail()) {
    // copy dbregs only if any watchpoints were set
    if ((r_source.m_dbr.dr[7] & 0xFF) == 0)
      return llvm::Error::success();

    m_dbr = r_source.m_dbr;
    res = WriteRegisterSet(DBRegSet);
  }
  return res.ToError();
}

#endif // defined(__x86_64__)

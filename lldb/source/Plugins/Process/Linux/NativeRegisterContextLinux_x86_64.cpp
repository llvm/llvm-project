//===-- NativeRegisterContextLinux_x86_64.cpp -----------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#if defined(__i386__) || defined(__x86_64__)

#include "NativeRegisterContextLinux_x86_64.h"

#include "lldb/Host/HostInfo.h"
#include "lldb/Utility/DataBufferHeap.h"
#include "lldb/Utility/Log.h"
#include "lldb/Utility/RegisterValue.h"
#include "lldb/Utility/Status.h"

#include "Plugins/Process/Utility/RegisterContextLinux_i386.h"
#include "Plugins/Process/Utility/RegisterContextLinux_x86_64.h"
#include <cpuid.h>
#include <linux/elf.h>

// Newer toolchains define __get_cpuid_count in cpuid.h, but some
// older-but-still-supported ones (e.g. gcc 5.4.0) don't, so we
// define it locally here, following the definition in clang/lib/Headers.
static inline int get_cpuid_count(unsigned int __leaf,
                                  unsigned int __subleaf,
                                  unsigned int *__eax, unsigned int *__ebx,
                                  unsigned int *__ecx, unsigned int *__edx)
{
  unsigned int __max_leaf = __get_cpuid_max(__leaf & 0x80000000, nullptr);

  if (__max_leaf == 0 || __max_leaf < __leaf)
    return 0;

  __cpuid_count(__leaf, __subleaf, *__eax, *__ebx, *__ecx, *__edx);
  return 1;
}

using namespace lldb_private;
using namespace lldb_private::process_linux;

// x86 32-bit general purpose registers.
static const uint32_t g_gpr_regnums_i386[] = {
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
static const uint32_t g_fpu_regnums_i386[] = {
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

// x86 32-bit AVX registers.
static const uint32_t g_avx_regnums_i386[] = {
    lldb_ymm0_i386,     lldb_ymm1_i386, lldb_ymm2_i386, lldb_ymm3_i386,
    lldb_ymm4_i386,     lldb_ymm5_i386, lldb_ymm6_i386, lldb_ymm7_i386,
    LLDB_INVALID_REGNUM // register sets need to end with this flag
};
static_assert((sizeof(g_avx_regnums_i386) / sizeof(g_avx_regnums_i386[0])) -
                      1 ==
                  k_num_avx_registers_i386,
              " g_avx_regnums_i386 has wrong number of register infos");

// x64 32-bit MPX registers.
static const uint32_t g_mpx_regnums_i386[] = {
    lldb_bnd0_i386,     lldb_bnd1_i386, lldb_bnd2_i386, lldb_bnd3_i386,
    lldb_bndcfgu_i386,  lldb_bndstatus_i386,
    LLDB_INVALID_REGNUM // register sets need to end with this flag
};
static_assert((sizeof(g_mpx_regnums_i386) / sizeof(g_mpx_regnums_i386[0])) -
                      1 ==
                  k_num_mpx_registers_i386,
              "g_mpx_regnums_x86_64 has wrong number of register infos");

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
    lldb_fctrl_x86_64,  lldb_fstat_x86_64, lldb_ftag_x86_64,
    lldb_fop_x86_64,    lldb_fiseg_x86_64, lldb_fioff_x86_64,
    lldb_fip_x86_64,    lldb_foseg_x86_64, lldb_fooff_x86_64,
    lldb_fdp_x86_64,    lldb_mxcsr_x86_64, lldb_mxcsrmask_x86_64,
    lldb_st0_x86_64,    lldb_st1_x86_64,   lldb_st2_x86_64,
    lldb_st3_x86_64,    lldb_st4_x86_64,   lldb_st5_x86_64,
    lldb_st6_x86_64,    lldb_st7_x86_64,   lldb_mm0_x86_64,
    lldb_mm1_x86_64,    lldb_mm2_x86_64,   lldb_mm3_x86_64,
    lldb_mm4_x86_64,    lldb_mm5_x86_64,   lldb_mm6_x86_64,
    lldb_mm7_x86_64,    lldb_xmm0_x86_64,  lldb_xmm1_x86_64,
    lldb_xmm2_x86_64,   lldb_xmm3_x86_64,  lldb_xmm4_x86_64,
    lldb_xmm5_x86_64,   lldb_xmm6_x86_64,  lldb_xmm7_x86_64,
    lldb_xmm8_x86_64,   lldb_xmm9_x86_64,  lldb_xmm10_x86_64,
    lldb_xmm11_x86_64,  lldb_xmm12_x86_64, lldb_xmm13_x86_64,
    lldb_xmm14_x86_64,  lldb_xmm15_x86_64,
    LLDB_INVALID_REGNUM // register sets need to end with this flag
};
static_assert((sizeof(g_fpu_regnums_x86_64) / sizeof(g_fpu_regnums_x86_64[0])) -
                      1 ==
                  k_num_fpr_registers_x86_64,
              "g_fpu_regnums_x86_64 has wrong number of register infos");

// x86 64-bit AVX registers.
static const uint32_t g_avx_regnums_x86_64[] = {
    lldb_ymm0_x86_64,   lldb_ymm1_x86_64,  lldb_ymm2_x86_64,  lldb_ymm3_x86_64,
    lldb_ymm4_x86_64,   lldb_ymm5_x86_64,  lldb_ymm6_x86_64,  lldb_ymm7_x86_64,
    lldb_ymm8_x86_64,   lldb_ymm9_x86_64,  lldb_ymm10_x86_64, lldb_ymm11_x86_64,
    lldb_ymm12_x86_64,  lldb_ymm13_x86_64, lldb_ymm14_x86_64, lldb_ymm15_x86_64,
    LLDB_INVALID_REGNUM // register sets need to end with this flag
};
static_assert((sizeof(g_avx_regnums_x86_64) / sizeof(g_avx_regnums_x86_64[0])) -
                      1 ==
                  k_num_avx_registers_x86_64,
              "g_avx_regnums_x86_64 has wrong number of register infos");

// x86 64-bit MPX registers.
static const uint32_t g_mpx_regnums_x86_64[] = {
    lldb_bnd0_x86_64,    lldb_bnd1_x86_64,    lldb_bnd2_x86_64,
    lldb_bnd3_x86_64,    lldb_bndcfgu_x86_64, lldb_bndstatus_x86_64,
    LLDB_INVALID_REGNUM // register sets need to end with this flag
};
static_assert((sizeof(g_mpx_regnums_x86_64) / sizeof(g_mpx_regnums_x86_64[0])) -
                      1 ==
                  k_num_mpx_registers_x86_64,
              "g_mpx_regnums_x86_64 has wrong number of register infos");

// Number of register sets provided by this context.
constexpr unsigned k_num_extended_register_sets = 2, k_num_register_sets = 4;

// Register sets for x86 32-bit.
static const RegisterSet g_reg_sets_i386[k_num_register_sets] = {
    {"General Purpose Registers", "gpr", k_num_gpr_registers_i386,
     g_gpr_regnums_i386},
    {"Floating Point Registers", "fpu", k_num_fpr_registers_i386,
     g_fpu_regnums_i386},
    {"Advanced Vector Extensions", "avx", k_num_avx_registers_i386,
     g_avx_regnums_i386},
    { "Memory Protection Extensions", "mpx", k_num_mpx_registers_i386,
     g_mpx_regnums_i386}};

// Register sets for x86 64-bit.
static const RegisterSet g_reg_sets_x86_64[k_num_register_sets] = {
    {"General Purpose Registers", "gpr", k_num_gpr_registers_x86_64,
     g_gpr_regnums_x86_64},
    {"Floating Point Registers", "fpu", k_num_fpr_registers_x86_64,
     g_fpu_regnums_x86_64},
    {"Advanced Vector Extensions", "avx", k_num_avx_registers_x86_64,
     g_avx_regnums_x86_64},
    { "Memory Protection Extensions", "mpx", k_num_mpx_registers_x86_64,
     g_mpx_regnums_x86_64}};

#define REG_CONTEXT_SIZE (GetRegisterInfoInterface().GetGPRSize() + sizeof(FPR))

// Required ptrace defines.

// Support ptrace extensions even when compiled without required kernel support
#ifndef NT_X86_XSTATE
#define NT_X86_XSTATE 0x202
#endif
#ifndef NT_PRXFPREG
#define NT_PRXFPREG 0x46e62b7f
#endif

// On x86_64 NT_PRFPREG is used to access the FXSAVE area. On i386, we need to
// use NT_PRXFPREG.
static inline unsigned int fxsr_regset(const ArchSpec &arch) {
  return arch.GetAddressByteSize() == 8 ? NT_PRFPREG : NT_PRXFPREG;
}

// Required MPX define.

// Support MPX extensions also if compiled with compiler without MPX support.
#ifndef bit_MPX
#define bit_MPX 0x4000
#endif

// XCR0 extended register sets masks.
#define mask_XSTATE_AVX (1ULL << 2)
#define mask_XSTATE_BNDREGS (1ULL << 3)
#define mask_XSTATE_BNDCFG (1ULL << 4)
#define mask_XSTATE_MPX (mask_XSTATE_BNDREGS | mask_XSTATE_BNDCFG)

std::unique_ptr<NativeRegisterContextLinux>
NativeRegisterContextLinux::CreateHostNativeRegisterContextLinux(
    const ArchSpec &target_arch, NativeThreadProtocol &native_thread) {
  return std::unique_ptr<NativeRegisterContextLinux>(
      new NativeRegisterContextLinux_x86_64(target_arch, native_thread));
}

// NativeRegisterContextLinux_x86_64 members.

static RegisterInfoInterface *
CreateRegisterInfoInterface(const ArchSpec &target_arch) {
  if (HostInfo::GetArchitecture().GetAddressByteSize() == 4) {
    // 32-bit hosts run with a RegisterContextLinux_i386 context.
    return new RegisterContextLinux_i386(target_arch);
  } else {
    assert((HostInfo::GetArchitecture().GetAddressByteSize() == 8) &&
           "Register setting path assumes this is a 64-bit host");
    // X86_64 hosts know how to work with 64-bit and 32-bit EXEs using the
    // x86_64 register context.
    return new RegisterContextLinux_x86_64(target_arch);
  }
}

// Return the size of the XSTATE area supported on this cpu. It is necessary to
// allocate the full size of the area even if we do not use/recognise all of it
// because ptrace(PTRACE_SETREGSET, NT_X86_XSTATE) will refuse to write to it if
// we do not pass it a buffer of sufficient size. The size is always at least
// sizeof(FPR) so that the allocated buffer can be safely cast to FPR*.
static std::size_t GetXSTATESize() {
  unsigned int eax, ebx, ecx, edx;
  // First check whether the XSTATE are is supported at all.
  if (!__get_cpuid(1, &eax, &ebx, &ecx, &edx) || !(ecx & bit_XSAVE))
    return sizeof(FPR);

  // Then fetch the maximum size of the area.
  if (!get_cpuid_count(0x0d, 0, &eax, &ebx, &ecx, &edx))
    return sizeof(FPR);
  return std::max<std::size_t>(ecx, sizeof(FPR));
}

NativeRegisterContextLinux_x86_64::NativeRegisterContextLinux_x86_64(
    const ArchSpec &target_arch, NativeThreadProtocol &native_thread)
    : NativeRegisterContextRegisterInfo(
          native_thread, CreateRegisterInfoInterface(target_arch)),
      m_xstate_type(XStateType::Invalid), m_ymm_set(), m_mpx_set(),
      m_reg_info(), m_gpr_x86_64() {
  // Set up data about ranges of valid registers.
  switch (target_arch.GetMachine()) {
  case llvm::Triple::x86:
    m_reg_info.num_registers = k_num_registers_i386;
    m_reg_info.num_gpr_registers = k_num_gpr_registers_i386;
    m_reg_info.num_fpr_registers = k_num_fpr_registers_i386;
    m_reg_info.num_avx_registers = k_num_avx_registers_i386;
    m_reg_info.num_mpx_registers = k_num_mpx_registers_i386;
    m_reg_info.last_gpr = k_last_gpr_i386;
    m_reg_info.first_fpr = k_first_fpr_i386;
    m_reg_info.last_fpr = k_last_fpr_i386;
    m_reg_info.first_st = lldb_st0_i386;
    m_reg_info.last_st = lldb_st7_i386;
    m_reg_info.first_mm = lldb_mm0_i386;
    m_reg_info.last_mm = lldb_mm7_i386;
    m_reg_info.first_xmm = lldb_xmm0_i386;
    m_reg_info.last_xmm = lldb_xmm7_i386;
    m_reg_info.first_ymm = lldb_ymm0_i386;
    m_reg_info.last_ymm = lldb_ymm7_i386;
    m_reg_info.first_mpxr = lldb_bnd0_i386;
    m_reg_info.last_mpxr = lldb_bnd3_i386;
    m_reg_info.first_mpxc = lldb_bndcfgu_i386;
    m_reg_info.last_mpxc = lldb_bndstatus_i386;
    m_reg_info.first_dr = lldb_dr0_i386;
    m_reg_info.last_dr = lldb_dr7_i386;
    m_reg_info.gpr_flags = lldb_eflags_i386;
    break;
  case llvm::Triple::x86_64:
    m_reg_info.num_registers = k_num_registers_x86_64;
    m_reg_info.num_gpr_registers = k_num_gpr_registers_x86_64;
    m_reg_info.num_fpr_registers = k_num_fpr_registers_x86_64;
    m_reg_info.num_avx_registers = k_num_avx_registers_x86_64;
    m_reg_info.num_mpx_registers = k_num_mpx_registers_x86_64;
    m_reg_info.last_gpr = k_last_gpr_x86_64;
    m_reg_info.first_fpr = k_first_fpr_x86_64;
    m_reg_info.last_fpr = k_last_fpr_x86_64;
    m_reg_info.first_st = lldb_st0_x86_64;
    m_reg_info.last_st = lldb_st7_x86_64;
    m_reg_info.first_mm = lldb_mm0_x86_64;
    m_reg_info.last_mm = lldb_mm7_x86_64;
    m_reg_info.first_xmm = lldb_xmm0_x86_64;
    m_reg_info.last_xmm = lldb_xmm15_x86_64;
    m_reg_info.first_ymm = lldb_ymm0_x86_64;
    m_reg_info.last_ymm = lldb_ymm15_x86_64;
    m_reg_info.first_mpxr = lldb_bnd0_x86_64;
    m_reg_info.last_mpxr = lldb_bnd3_x86_64;
    m_reg_info.first_mpxc = lldb_bndcfgu_x86_64;
    m_reg_info.last_mpxc = lldb_bndstatus_x86_64;
    m_reg_info.first_dr = lldb_dr0_x86_64;
    m_reg_info.last_dr = lldb_dr7_x86_64;
    m_reg_info.gpr_flags = lldb_rflags_x86_64;
    break;
  default:
    assert(false && "Unhandled target architecture.");
    break;
  }

  std::size_t xstate_size = GetXSTATESize();
  m_xstate.reset(static_cast<FPR *>(std::malloc(xstate_size)));
  m_iovec.iov_base = m_xstate.get();
  m_iovec.iov_len = xstate_size;

  // Clear out the FPR state.
  ::memset(m_xstate.get(), 0, xstate_size);

  // Store byte offset of fctrl (i.e. first register of FPR)
  const RegisterInfo *reg_info_fctrl = GetRegisterInfoByName("fctrl");
  m_fctrl_offset_in_userarea = reg_info_fctrl->byte_offset;
}

// CONSIDER after local and llgs debugging are merged, register set support can
// be moved into a base x86-64 class with IsRegisterSetAvailable made virtual.
uint32_t NativeRegisterContextLinux_x86_64::GetRegisterSetCount() const {
  uint32_t sets = 0;
  for (uint32_t set_index = 0; set_index < k_num_register_sets; ++set_index) {
    if (IsRegisterSetAvailable(set_index))
      ++sets;
  }

  return sets;
}

uint32_t NativeRegisterContextLinux_x86_64::GetUserRegisterCount() const {
  uint32_t count = 0;
  for (uint32_t set_index = 0; set_index < k_num_register_sets; ++set_index) {
    const RegisterSet *set = GetRegisterSet(set_index);
    if (set)
      count += set->num_registers;
  }
  return count;
}

const RegisterSet *
NativeRegisterContextLinux_x86_64::GetRegisterSet(uint32_t set_index) const {
  if (!IsRegisterSetAvailable(set_index))
    return nullptr;

  switch (GetRegisterInfoInterface().GetTargetArchitecture().GetMachine()) {
  case llvm::Triple::x86:
    return &g_reg_sets_i386[set_index];
  case llvm::Triple::x86_64:
    return &g_reg_sets_x86_64[set_index];
  default:
    assert(false && "Unhandled target architecture.");
    return nullptr;
  }

  return nullptr;
}

Status
NativeRegisterContextLinux_x86_64::ReadRegister(const RegisterInfo *reg_info,
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

  if (IsFPR(reg) || IsAVX(reg) || IsMPX(reg)) {
    error = ReadFPR();
    if (error.Fail())
      return error;
  } else {
    uint32_t full_reg = reg;
    bool is_subreg = reg_info->invalidate_regs &&
                     (reg_info->invalidate_regs[0] != LLDB_INVALID_REGNUM);

    if (is_subreg) {
      // Read the full aligned 64-bit register.
      full_reg = reg_info->invalidate_regs[0];
    }

    error = ReadRegisterRaw(full_reg, reg_value);

    if (error.Success()) {
      // If our read was not aligned (for ah,bh,ch,dh), shift our returned
      // value one byte to the right.
      if (is_subreg && (reg_info->byte_offset & 0x1))
        reg_value.SetUInt64(reg_value.GetAsUInt64() >> 8);

      // If our return byte size was greater than the return value reg size,
      // then use the type specified by reg_info rather than the uint64_t
      // default
      if (reg_value.GetByteSize() > reg_info->byte_size)
        reg_value.SetType(reg_info);
    }
    return error;
  }

  if (reg_info->encoding == lldb::eEncodingVector) {
    lldb::ByteOrder byte_order = GetByteOrder();

    if (byte_order != lldb::eByteOrderInvalid) {
      if (reg >= m_reg_info.first_st && reg <= m_reg_info.last_st)
        reg_value.SetBytes(
            m_xstate->fxsave.stmm[reg - m_reg_info.first_st].bytes,
            reg_info->byte_size, byte_order);
      if (reg >= m_reg_info.first_mm && reg <= m_reg_info.last_mm)
        reg_value.SetBytes(
            m_xstate->fxsave.stmm[reg - m_reg_info.first_mm].bytes,
            reg_info->byte_size, byte_order);
      if (reg >= m_reg_info.first_xmm && reg <= m_reg_info.last_xmm)
        reg_value.SetBytes(
            m_xstate->fxsave.xmm[reg - m_reg_info.first_xmm].bytes,
            reg_info->byte_size, byte_order);
      if (reg >= m_reg_info.first_ymm && reg <= m_reg_info.last_ymm) {
        // Concatenate ymm using the register halves in xmm.bytes and
        // ymmh.bytes
        if (CopyXSTATEtoYMM(reg, byte_order))
          reg_value.SetBytes(m_ymm_set.ymm[reg - m_reg_info.first_ymm].bytes,
                             reg_info->byte_size, byte_order);
        else {
          error.SetErrorString("failed to copy ymm register value");
          return error;
        }
      }
      if (reg >= m_reg_info.first_mpxr && reg <= m_reg_info.last_mpxr) {
        if (CopyXSTATEtoMPX(reg))
          reg_value.SetBytes(m_mpx_set.mpxr[reg - m_reg_info.first_mpxr].bytes,
                             reg_info->byte_size, byte_order);
        else {
          error.SetErrorString("failed to copy mpx register value");
          return error;
        }
      }
      if (reg >= m_reg_info.first_mpxc && reg <= m_reg_info.last_mpxc) {
        if (CopyXSTATEtoMPX(reg))
          reg_value.SetBytes(m_mpx_set.mpxc[reg - m_reg_info.first_mpxc].bytes,
                             reg_info->byte_size, byte_order);
        else {
          error.SetErrorString("failed to copy mpx register value");
          return error;
        }
      }

      if (reg_value.GetType() != RegisterValue::eTypeBytes)
        error.SetErrorString(
            "write failed - type was expected to be RegisterValue::eTypeBytes");

      return error;
    }

    error.SetErrorString("byte order is invalid");
    return error;
  }

  // Get pointer to m_xstate->fxsave variable and set the data from it.

  // Byte offsets of all registers are calculated wrt 'UserArea' structure.
  // However, ReadFPR() reads fpu registers {using ptrace(PTRACE_GETFPREGS,..)}
  // and stores them in 'm_fpr' (of type FPR structure). To extract values of
  // fpu registers, m_fpr should be read at byte offsets calculated wrt to FPR
  // structure.

  // Since, FPR structure is also one of the member of UserArea structure.
  // byte_offset(fpu wrt FPR) = byte_offset(fpu wrt UserArea) -
  // byte_offset(fctrl wrt UserArea)
  assert((reg_info->byte_offset - m_fctrl_offset_in_userarea) < sizeof(FPR));
  uint8_t *src = (uint8_t *)m_xstate.get() + reg_info->byte_offset -
                 m_fctrl_offset_in_userarea;

  if (src == reinterpret_cast<uint8_t *>(&m_xstate->fxsave.ftag)) {
    reg_value.SetUInt16(AbridgedToFullTagWord(
        m_xstate->fxsave.ftag, m_xstate->fxsave.fstat, m_xstate->fxsave.stmm));
    return error;
  }

  switch (reg_info->byte_size) {
  case 1:
    reg_value.SetUInt8(*(uint8_t *)src);
    break;
  case 2:
    reg_value.SetUInt16(*(uint16_t *)src);
    break;
  case 4:
    reg_value.SetUInt32(*(uint32_t *)src);
    break;
  case 8:
    reg_value.SetUInt64(*(uint64_t *)src);
    break;
  default:
    assert(false && "Unhandled data size.");
    error.SetErrorStringWithFormat("unhandled byte size: %" PRIu32,
                                   reg_info->byte_size);
    break;
  }

  return error;
}

void NativeRegisterContextLinux_x86_64::UpdateXSTATEforWrite(
    uint32_t reg_index) {
  XSAVE_HDR::XFeature &xstate_bv = m_xstate->xsave.header.xstate_bv;
  if (IsFPR(reg_index)) {
    // IsFPR considers both %st and %xmm registers as floating point, but these
    // map to two features. Set both flags, just in case.
    xstate_bv |= XSAVE_HDR::XFeature::FP | XSAVE_HDR::XFeature::SSE;
  } else if (IsAVX(reg_index)) {
    // Lower bytes of some %ymm registers are shared with %xmm registers.
    xstate_bv |= XSAVE_HDR::XFeature::YMM | XSAVE_HDR::XFeature::SSE;
  } else if (IsMPX(reg_index)) {
    // MPX registers map to two XSAVE features.
    xstate_bv |= XSAVE_HDR::XFeature::BNDREGS | XSAVE_HDR::XFeature::BNDCSR;
  }
}

Status NativeRegisterContextLinux_x86_64::WriteRegister(
    const RegisterInfo *reg_info, const RegisterValue &reg_value) {
  assert(reg_info && "reg_info is null");

  const uint32_t reg_index = reg_info->kinds[lldb::eRegisterKindLLDB];
  if (reg_index == LLDB_INVALID_REGNUM)
    return Status("no lldb regnum for %s", reg_info && reg_info->name
                                               ? reg_info->name
                                               : "<unknown register>");

  UpdateXSTATEforWrite(reg_index);

  if (IsGPR(reg_index) || IsDR(reg_index))
    return WriteRegisterRaw(reg_index, reg_value);

  if (IsFPR(reg_index) || IsAVX(reg_index) || IsMPX(reg_index)) {
    if (reg_info->encoding == lldb::eEncodingVector) {
      if (reg_index >= m_reg_info.first_st && reg_index <= m_reg_info.last_st)
        ::memcpy(m_xstate->fxsave.stmm[reg_index - m_reg_info.first_st].bytes,
                 reg_value.GetBytes(), reg_value.GetByteSize());

      if (reg_index >= m_reg_info.first_mm && reg_index <= m_reg_info.last_mm)
        ::memcpy(m_xstate->fxsave.stmm[reg_index - m_reg_info.first_mm].bytes,
                 reg_value.GetBytes(), reg_value.GetByteSize());

      if (reg_index >= m_reg_info.first_xmm && reg_index <= m_reg_info.last_xmm)
        ::memcpy(m_xstate->fxsave.xmm[reg_index - m_reg_info.first_xmm].bytes,
                 reg_value.GetBytes(), reg_value.GetByteSize());

      if (reg_index >= m_reg_info.first_ymm &&
          reg_index <= m_reg_info.last_ymm) {
        // Store ymm register content, and split into the register halves in
        // xmm.bytes and ymmh.bytes
        ::memcpy(m_ymm_set.ymm[reg_index - m_reg_info.first_ymm].bytes,
                 reg_value.GetBytes(), reg_value.GetByteSize());
        if (!CopyYMMtoXSTATE(reg_index, GetByteOrder()))
          return Status("CopyYMMtoXSTATE() failed");
      }

      if (reg_index >= m_reg_info.first_mpxr &&
          reg_index <= m_reg_info.last_mpxr) {
        ::memcpy(m_mpx_set.mpxr[reg_index - m_reg_info.first_mpxr].bytes,
                 reg_value.GetBytes(), reg_value.GetByteSize());
        if (!CopyMPXtoXSTATE(reg_index))
          return Status("CopyMPXtoXSTATE() failed");
      }

      if (reg_index >= m_reg_info.first_mpxc &&
          reg_index <= m_reg_info.last_mpxc) {
        ::memcpy(m_mpx_set.mpxc[reg_index - m_reg_info.first_mpxc].bytes,
                 reg_value.GetBytes(), reg_value.GetByteSize());
        if (!CopyMPXtoXSTATE(reg_index))
          return Status("CopyMPXtoXSTATE() failed");
      }
    } else {
      // Get pointer to m_xstate->fxsave variable and set the data to it.

      // Byte offsets of all registers are calculated wrt 'UserArea' structure.
      // However, WriteFPR() takes m_fpr (of type FPR structure) and writes
      // only fpu registers using ptrace(PTRACE_SETFPREGS,..) API. Hence fpu
      // registers should be written in m_fpr at byte offsets calculated wrt
      // FPR structure.

      // Since, FPR structure is also one of the member of UserArea structure.
      // byte_offset(fpu wrt FPR) = byte_offset(fpu wrt UserArea) -
      // byte_offset(fctrl wrt UserArea)
      assert((reg_info->byte_offset - m_fctrl_offset_in_userarea) <
             sizeof(FPR));
      uint8_t *dst = (uint8_t *)m_xstate.get() + reg_info->byte_offset -
                     m_fctrl_offset_in_userarea;

      if (dst == reinterpret_cast<uint8_t *>(&m_xstate->fxsave.ftag))
        m_xstate->fxsave.ftag = FullToAbridgedTagWord(reg_value.GetAsUInt16());
      else {
        switch (reg_info->byte_size) {
        case 1:
          *(uint8_t *)dst = reg_value.GetAsUInt8();
          break;
        case 2:
          *(uint16_t *)dst = reg_value.GetAsUInt16();
          break;
        case 4:
          *(uint32_t *)dst = reg_value.GetAsUInt32();
          break;
        case 8:
          *(uint64_t *)dst = reg_value.GetAsUInt64();
          break;
        default:
          assert(false && "Unhandled data size.");
          return Status("unhandled register data size %" PRIu32,
                        reg_info->byte_size);
        }
      }
    }

    Status error = WriteFPR();
    if (error.Fail())
      return error;

    if (IsAVX(reg_index)) {
      if (!CopyYMMtoXSTATE(reg_index, GetByteOrder()))
        return Status("CopyYMMtoXSTATE() failed");
    }

    if (IsMPX(reg_index)) {
      if (!CopyMPXtoXSTATE(reg_index))
        return Status("CopyMPXtoXSTATE() failed");
    }
    return Status();
  }
  return Status("failed - register wasn't recognized to be a GPR or an FPR, "
                "write strategy unknown");
}

Status NativeRegisterContextLinux_x86_64::ReadAllRegisterValues(
    lldb::DataBufferSP &data_sp) {
  Status error;

  data_sp.reset(new DataBufferHeap(REG_CONTEXT_SIZE, 0));
  error = ReadGPR();
  if (error.Fail())
    return error;

  error = ReadFPR();
  if (error.Fail())
    return error;

  uint8_t *dst = data_sp->GetBytes();
  ::memcpy(dst, &m_gpr_x86_64, GetRegisterInfoInterface().GetGPRSize());
  dst += GetRegisterInfoInterface().GetGPRSize();
  if (m_xstate_type == XStateType::FXSAVE)
    ::memcpy(dst, &m_xstate->fxsave, sizeof(m_xstate->fxsave));
  else if (m_xstate_type == XStateType::XSAVE) {
    lldb::ByteOrder byte_order = GetByteOrder();

    if (IsCPUFeatureAvailable(RegSet::avx)) {
      // Assemble the YMM register content from the register halves.
      for (uint32_t reg = m_reg_info.first_ymm; reg <= m_reg_info.last_ymm;
           ++reg) {
        if (!CopyXSTATEtoYMM(reg, byte_order)) {
          error.SetErrorStringWithFormat(
              "NativeRegisterContextLinux_x86_64::%s "
              "CopyXSTATEtoYMM() failed for reg num "
              "%" PRIu32,
              __FUNCTION__, reg);
          return error;
        }
      }
    }

    if (IsCPUFeatureAvailable(RegSet::mpx)) {
      for (uint32_t reg = m_reg_info.first_mpxr; reg <= m_reg_info.last_mpxc;
           ++reg) {
        if (!CopyXSTATEtoMPX(reg)) {
          error.SetErrorStringWithFormat(
              "NativeRegisterContextLinux_x86_64::%s "
              "CopyXSTATEtoMPX() failed for reg num "
              "%" PRIu32,
              __FUNCTION__, reg);
          return error;
        }
      }
    }
    // Copy the extended register state including the assembled ymm registers.
    ::memcpy(dst, m_xstate.get(), sizeof(FPR));
  } else {
    assert(false && "how do we save the floating point registers?");
    error.SetErrorString("unsure how to save the floating point registers");
  }
  /** The following code is specific to Linux x86 based architectures,
   *  where the register orig_eax (32 bit)/orig_rax (64 bit) is set to
   *  -1 to solve the bug 23659, such a setting prevents the automatic
   *  decrement of the instruction pointer which was causing the SIGILL
   *  exception.
   * **/

  RegisterValue value((uint64_t)-1);
  const RegisterInfo *reg_info =
      GetRegisterInfoInterface().GetDynamicRegisterInfo("orig_eax");
  if (reg_info == nullptr)
    reg_info = GetRegisterInfoInterface().GetDynamicRegisterInfo("orig_rax");

  if (reg_info != nullptr)
    return DoWriteRegisterValue(reg_info->byte_offset, reg_info->name, value);

  return error;
}

Status NativeRegisterContextLinux_x86_64::WriteAllRegisterValues(
    const lldb::DataBufferSP &data_sp) {
  Status error;

  if (!data_sp) {
    error.SetErrorStringWithFormat(
        "NativeRegisterContextLinux_x86_64::%s invalid data_sp provided",
        __FUNCTION__);
    return error;
  }

  if (data_sp->GetByteSize() != REG_CONTEXT_SIZE) {
    error.SetErrorStringWithFormatv(
        "data_sp contained mismatched data size, expected {0}, actual {1}",
        REG_CONTEXT_SIZE, data_sp->GetByteSize());
    return error;
  }

  uint8_t *src = data_sp->GetBytes();
  if (src == nullptr) {
    error.SetErrorStringWithFormat("NativeRegisterContextLinux_x86_64::%s "
                                   "DataBuffer::GetBytes() returned a null "
                                   "pointer",
                                   __FUNCTION__);
    return error;
  }
  ::memcpy(&m_gpr_x86_64, src, GetRegisterInfoInterface().GetGPRSize());

  error = WriteGPR();
  if (error.Fail())
    return error;

  src += GetRegisterInfoInterface().GetGPRSize();
  if (m_xstate_type == XStateType::FXSAVE)
    ::memcpy(&m_xstate->fxsave, src, sizeof(m_xstate->fxsave));
  else if (m_xstate_type == XStateType::XSAVE)
    ::memcpy(&m_xstate->xsave, src, sizeof(m_xstate->xsave));

  error = WriteFPR();
  if (error.Fail())
    return error;

  if (m_xstate_type == XStateType::XSAVE) {
    lldb::ByteOrder byte_order = GetByteOrder();

    if (IsCPUFeatureAvailable(RegSet::avx)) {
      // Parse the YMM register content from the register halves.
      for (uint32_t reg = m_reg_info.first_ymm; reg <= m_reg_info.last_ymm;
           ++reg) {
        if (!CopyYMMtoXSTATE(reg, byte_order)) {
          error.SetErrorStringWithFormat(
              "NativeRegisterContextLinux_x86_64::%s "
              "CopyYMMtoXSTATE() failed for reg num "
              "%" PRIu32,
              __FUNCTION__, reg);
          return error;
        }
      }
    }

    if (IsCPUFeatureAvailable(RegSet::mpx)) {
      for (uint32_t reg = m_reg_info.first_mpxr; reg <= m_reg_info.last_mpxc;
           ++reg) {
        if (!CopyMPXtoXSTATE(reg)) {
          error.SetErrorStringWithFormat(
              "NativeRegisterContextLinux_x86_64::%s "
              "CopyMPXtoXSTATE() failed for reg num "
              "%" PRIu32,
              __FUNCTION__, reg);
          return error;
        }
      }
    }
  }

  return error;
}

bool NativeRegisterContextLinux_x86_64::IsCPUFeatureAvailable(
    RegSet feature_code) const {
  if (m_xstate_type == XStateType::Invalid) {
    if (const_cast<NativeRegisterContextLinux_x86_64 *>(this)->ReadFPR().Fail())
      return false;
  }
  switch (feature_code) {
  case RegSet::gpr:
  case RegSet::fpu:
    return true;
  case RegSet::avx: // Check if CPU has AVX and if there is kernel support, by
                    // reading in the XCR0 area of XSAVE.
    if ((m_xstate->xsave.i387.xcr0 & mask_XSTATE_AVX) == mask_XSTATE_AVX)
      return true;
     break;
  case RegSet::mpx: // Check if CPU has MPX and if there is kernel support, by
                    // reading in the XCR0 area of XSAVE.
    if ((m_xstate->xsave.i387.xcr0 & mask_XSTATE_MPX) == mask_XSTATE_MPX)
      return true;
    break;
  }
  return false;
}

bool NativeRegisterContextLinux_x86_64::IsRegisterSetAvailable(
    uint32_t set_index) const {
  uint32_t num_sets = k_num_register_sets - k_num_extended_register_sets;

  switch (static_cast<RegSet>(set_index)) {
  case RegSet::gpr:
  case RegSet::fpu:
    return (set_index < num_sets);
  case RegSet::avx:
    return IsCPUFeatureAvailable(RegSet::avx);
  case RegSet::mpx:
    return IsCPUFeatureAvailable(RegSet::mpx);
  }
  return false;
}

bool NativeRegisterContextLinux_x86_64::IsGPR(uint32_t reg_index) const {
  // GPRs come first.
  return reg_index <= m_reg_info.last_gpr;
}

bool NativeRegisterContextLinux_x86_64::IsFPR(uint32_t reg_index) const {
  return (m_reg_info.first_fpr <= reg_index &&
          reg_index <= m_reg_info.last_fpr);
}

bool NativeRegisterContextLinux_x86_64::IsDR(uint32_t reg_index) const {
  return (m_reg_info.first_dr <= reg_index &&
          reg_index <= m_reg_info.last_dr);
}

Status NativeRegisterContextLinux_x86_64::WriteFPR() {
  switch (m_xstate_type) {
  case XStateType::FXSAVE:
    return WriteRegisterSet(
        &m_iovec, sizeof(m_xstate->fxsave),
        fxsr_regset(GetRegisterInfoInterface().GetTargetArchitecture()));
  case XStateType::XSAVE:
    return WriteRegisterSet(&m_iovec, sizeof(m_xstate->xsave), NT_X86_XSTATE);
  default:
    return Status("Unrecognized FPR type.");
  }
}

bool NativeRegisterContextLinux_x86_64::IsAVX(uint32_t reg_index) const {
  if (!IsCPUFeatureAvailable(RegSet::avx))
    return false;
  return (m_reg_info.first_ymm <= reg_index &&
          reg_index <= m_reg_info.last_ymm);
}

bool NativeRegisterContextLinux_x86_64::CopyXSTATEtoYMM(
    uint32_t reg_index, lldb::ByteOrder byte_order) {
  if (!IsAVX(reg_index))
    return false;

  if (byte_order == lldb::eByteOrderLittle) {
    uint32_t reg_no = reg_index - m_reg_info.first_ymm;
    m_ymm_set.ymm[reg_no] = XStateToYMM(
        m_xstate->fxsave.xmm[reg_no].bytes,
        m_xstate->xsave.ymmh[reg_no].bytes);
    return true;
  }

  return false; // unsupported or invalid byte order
}

bool NativeRegisterContextLinux_x86_64::CopyYMMtoXSTATE(
    uint32_t reg, lldb::ByteOrder byte_order) {
  if (!IsAVX(reg))
    return false;

  if (byte_order == lldb::eByteOrderLittle) {
    uint32_t reg_no = reg - m_reg_info.first_ymm;
    YMMToXState(m_ymm_set.ymm[reg_no],
        m_xstate->fxsave.xmm[reg_no].bytes,
        m_xstate->xsave.ymmh[reg_no].bytes);
    return true;
  }

  return false; // unsupported or invalid byte order
}

void *NativeRegisterContextLinux_x86_64::GetFPRBuffer() {
  switch (m_xstate_type) {
  case XStateType::FXSAVE:
    return &m_xstate->fxsave;
  case XStateType::XSAVE:
    return &m_iovec;
  default:
    return nullptr;
  }
}

size_t NativeRegisterContextLinux_x86_64::GetFPRSize() {
  switch (m_xstate_type) {
  case XStateType::FXSAVE:
    return sizeof(m_xstate->fxsave);
  case XStateType::XSAVE:
    return sizeof(m_iovec);
  default:
    return 0;
  }
}

Status NativeRegisterContextLinux_x86_64::ReadFPR() {
  Status error;

  // Probe XSAVE and if it is not supported fall back to FXSAVE.
  if (m_xstate_type != XStateType::FXSAVE) {
    error = ReadRegisterSet(&m_iovec, sizeof(m_xstate->xsave), NT_X86_XSTATE);
    if (!error.Fail()) {
      m_xstate_type = XStateType::XSAVE;
      return error;
    }
  }
  error = ReadRegisterSet(
      &m_iovec, sizeof(m_xstate->xsave),
      fxsr_regset(GetRegisterInfoInterface().GetTargetArchitecture()));
  if (!error.Fail()) {
    m_xstate_type = XStateType::FXSAVE;
    return error;
  }
  return Status("Unrecognized FPR type.");
}

bool NativeRegisterContextLinux_x86_64::IsMPX(uint32_t reg_index) const {
  if (!IsCPUFeatureAvailable(RegSet::mpx))
    return false;
  return (m_reg_info.first_mpxr <= reg_index &&
          reg_index <= m_reg_info.last_mpxc);
}

bool NativeRegisterContextLinux_x86_64::CopyXSTATEtoMPX(uint32_t reg) {
  if (!IsMPX(reg))
    return false;

  if (reg >= m_reg_info.first_mpxr && reg <= m_reg_info.last_mpxr) {
    ::memcpy(m_mpx_set.mpxr[reg - m_reg_info.first_mpxr].bytes,
             m_xstate->xsave.mpxr[reg - m_reg_info.first_mpxr].bytes,
             sizeof(MPXReg));
  } else {
    ::memcpy(m_mpx_set.mpxc[reg - m_reg_info.first_mpxc].bytes,
             m_xstate->xsave.mpxc[reg - m_reg_info.first_mpxc].bytes,
             sizeof(MPXCsr));
  }
  return true;
}

bool NativeRegisterContextLinux_x86_64::CopyMPXtoXSTATE(uint32_t reg) {
  if (!IsMPX(reg))
    return false;

  if (reg >= m_reg_info.first_mpxr && reg <= m_reg_info.last_mpxr) {
    ::memcpy(m_xstate->xsave.mpxr[reg - m_reg_info.first_mpxr].bytes,
             m_mpx_set.mpxr[reg - m_reg_info.first_mpxr].bytes, sizeof(MPXReg));
  } else {
    ::memcpy(m_xstate->xsave.mpxc[reg - m_reg_info.first_mpxc].bytes,
             m_mpx_set.mpxc[reg - m_reg_info.first_mpxc].bytes, sizeof(MPXCsr));
  }
  return true;
}

uint32_t
NativeRegisterContextLinux_x86_64::GetPtraceOffset(uint32_t reg_index) {
  // If register is MPX, remove extra factor from gdb offset
  return GetRegisterInfoAtIndex(reg_index)->byte_offset -
         (IsMPX(reg_index) ? 128 : 0);
}

llvm::Optional<NativeRegisterContextLinux::SyscallData>
NativeRegisterContextLinux_x86_64::GetSyscallData() {
  switch (GetRegisterInfoInterface().GetTargetArchitecture().GetMachine()) {
  case llvm::Triple::x86: {
    static const uint8_t Int80[] = {0xcd, 0x80};
    static const uint32_t Args[] = {lldb_eax_i386, lldb_ebx_i386, lldb_ecx_i386,
                                    lldb_edx_i386, lldb_esi_i386, lldb_edi_i386,
                                    lldb_ebp_i386};
    return SyscallData{Int80, Args, lldb_eax_i386};
  }
  case llvm::Triple::x86_64: {
    static const uint8_t Syscall[] = {0x0f, 0x05};
    static const uint32_t Args[] = {
        lldb_rax_x86_64, lldb_rdi_x86_64, lldb_rsi_x86_64, lldb_rdx_x86_64,
        lldb_r10_x86_64, lldb_r8_x86_64,  lldb_r9_x86_64};
    return SyscallData{Syscall, Args, lldb_rax_x86_64};
  }
  default:
    llvm_unreachable("Unhandled architecture!");
  }
}

llvm::Optional<NativeRegisterContextLinux::MmapData>
NativeRegisterContextLinux_x86_64::GetMmapData() {
  switch (GetRegisterInfoInterface().GetTargetArchitecture().GetMachine()) {
  case llvm::Triple::x86:
    return MmapData{192, 91};
  case llvm::Triple::x86_64:
    return MmapData{9, 11};
  default:
    llvm_unreachable("Unhandled architecture!");
  }
}

#endif // defined(__i386__) || defined(__x86_64__)

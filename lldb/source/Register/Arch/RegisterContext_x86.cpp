//===-- RegisterContext_x86.cpp ---------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "RegisterContext_x86.h"
#include "lldb-x86-register-enums.h"
#include "lldb/lldb-defines.h"

using namespace lldb_private;

uint32_t x86_register_info::g_contained_eax[] = {lldb_eax_i386,
                                                 LLDB_INVALID_REGNUM};
uint32_t x86_register_info::g_contained_ebx[] = {lldb_ebx_i386,
                                                 LLDB_INVALID_REGNUM};
uint32_t x86_register_info::g_contained_ecx[] = {lldb_ecx_i386,
                                                 LLDB_INVALID_REGNUM};
uint32_t x86_register_info::g_contained_edx[] = {lldb_edx_i386,
                                                 LLDB_INVALID_REGNUM};
uint32_t x86_register_info::g_contained_edi[] = {lldb_edi_i386,
                                                 LLDB_INVALID_REGNUM};
uint32_t x86_register_info::g_contained_esi[] = {lldb_esi_i386,
                                                 LLDB_INVALID_REGNUM};
uint32_t x86_register_info::g_contained_ebp[] = {lldb_ebp_i386,
                                                 LLDB_INVALID_REGNUM};
uint32_t x86_register_info::g_contained_esp[] = {lldb_esp_i386,
                                                 LLDB_INVALID_REGNUM};

uint32_t x86_register_info::g_invalidate_eax[] = {lldb_eax_i386, lldb_ax_i386,
                                                  lldb_ah_i386, lldb_al_i386,
                                                  LLDB_INVALID_REGNUM};
uint32_t x86_register_info::g_invalidate_ebx[] = {lldb_ebx_i386, lldb_bx_i386,
                                                  lldb_bh_i386, lldb_bl_i386,
                                                  LLDB_INVALID_REGNUM};
uint32_t x86_register_info::g_invalidate_ecx[] = {lldb_ecx_i386, lldb_cx_i386,
                                                  lldb_ch_i386, lldb_cl_i386,
                                                  LLDB_INVALID_REGNUM};
uint32_t x86_register_info::g_invalidate_edx[] = {lldb_edx_i386, lldb_dx_i386,
                                                  lldb_dh_i386, lldb_dl_i386,
                                                  LLDB_INVALID_REGNUM};
uint32_t x86_register_info::g_invalidate_edi[] = {lldb_edi_i386, lldb_di_i386,
                                                  LLDB_INVALID_REGNUM};
uint32_t x86_register_info::g_invalidate_esi[] = {lldb_esi_i386, lldb_si_i386,
                                                  LLDB_INVALID_REGNUM};
uint32_t x86_register_info::g_invalidate_ebp[] = {lldb_ebp_i386, lldb_bp_i386,
                                                  LLDB_INVALID_REGNUM};
uint32_t x86_register_info::g_invalidate_esp[] = {lldb_esp_i386, lldb_sp_i386,
                                                  LLDB_INVALID_REGNUM};

uint32_t x86_register_info::g_contained_rax[] = {lldb_rax_x86_64,
                                                 LLDB_INVALID_REGNUM};
uint32_t x86_register_info::g_contained_rbx[] = {lldb_rbx_x86_64,
                                                 LLDB_INVALID_REGNUM};
uint32_t x86_register_info::g_contained_rcx[] = {lldb_rcx_x86_64,
                                                 LLDB_INVALID_REGNUM};
uint32_t x86_register_info::g_contained_rdx[] = {lldb_rdx_x86_64,
                                                 LLDB_INVALID_REGNUM};
uint32_t x86_register_info::g_contained_rdi[] = {lldb_rdi_x86_64,
                                                 LLDB_INVALID_REGNUM};
uint32_t x86_register_info::g_contained_rsi[] = {lldb_rsi_x86_64,
                                                 LLDB_INVALID_REGNUM};
uint32_t x86_register_info::g_contained_rbp[] = {lldb_rbp_x86_64,
                                                 LLDB_INVALID_REGNUM};
uint32_t x86_register_info::g_contained_rsp[] = {lldb_rsp_x86_64,
                                                 LLDB_INVALID_REGNUM};
uint32_t x86_register_info::g_contained_r8[] = {lldb_r8_x86_64,
                                                LLDB_INVALID_REGNUM};
uint32_t x86_register_info::g_contained_r9[] = {lldb_r9_x86_64,
                                                LLDB_INVALID_REGNUM};
uint32_t x86_register_info::g_contained_r10[] = {lldb_r10_x86_64,
                                                 LLDB_INVALID_REGNUM};
uint32_t x86_register_info::g_contained_r11[] = {lldb_r11_x86_64,
                                                 LLDB_INVALID_REGNUM};
uint32_t x86_register_info::g_contained_r12[] = {lldb_r12_x86_64,
                                                 LLDB_INVALID_REGNUM};
uint32_t x86_register_info::g_contained_r13[] = {lldb_r13_x86_64,
                                                 LLDB_INVALID_REGNUM};
uint32_t x86_register_info::g_contained_r14[] = {lldb_r14_x86_64,
                                                 LLDB_INVALID_REGNUM};
uint32_t x86_register_info::g_contained_r15[] = {lldb_r15_x86_64,
                                                 LLDB_INVALID_REGNUM};

uint32_t x86_register_info::g_invalidate_rax[] = {
    lldb_rax_x86_64, lldb_eax_x86_64, lldb_ax_x86_64,
    lldb_ah_x86_64,  lldb_al_x86_64,  LLDB_INVALID_REGNUM};
uint32_t x86_register_info::g_invalidate_rbx[] = {
    lldb_rbx_x86_64, lldb_ebx_x86_64, lldb_bx_x86_64,
    lldb_bh_x86_64,  lldb_bl_x86_64,  LLDB_INVALID_REGNUM};
uint32_t x86_register_info::g_invalidate_rcx[] = {
    lldb_rcx_x86_64, lldb_ecx_x86_64, lldb_cx_x86_64,
    lldb_ch_x86_64,  lldb_cl_x86_64,  LLDB_INVALID_REGNUM};
uint32_t x86_register_info::g_invalidate_rdx[] = {
    lldb_rdx_x86_64, lldb_edx_x86_64, lldb_dx_x86_64,
    lldb_dh_x86_64,  lldb_dl_x86_64,  LLDB_INVALID_REGNUM};
uint32_t x86_register_info::g_invalidate_rdi[] = {
    lldb_rdi_x86_64, lldb_edi_x86_64, lldb_di_x86_64, lldb_dil_x86_64,
    LLDB_INVALID_REGNUM};
uint32_t x86_register_info::g_invalidate_rsi[] = {
    lldb_rsi_x86_64, lldb_esi_x86_64, lldb_si_x86_64, lldb_sil_x86_64,
    LLDB_INVALID_REGNUM};
uint32_t x86_register_info::g_invalidate_rbp[] = {
    lldb_rbp_x86_64, lldb_ebp_x86_64, lldb_bp_x86_64, lldb_bpl_x86_64,
    LLDB_INVALID_REGNUM};
uint32_t x86_register_info::g_invalidate_rsp[] = {
    lldb_rsp_x86_64, lldb_esp_x86_64, lldb_sp_x86_64, lldb_spl_x86_64,
    LLDB_INVALID_REGNUM};
uint32_t x86_register_info::g_invalidate_r8[] = {
    lldb_r8_x86_64, lldb_r8d_x86_64, lldb_r8w_x86_64, lldb_r8l_x86_64,
    LLDB_INVALID_REGNUM};
uint32_t x86_register_info::g_invalidate_r9[] = {
    lldb_r9_x86_64, lldb_r9d_x86_64, lldb_r9w_x86_64, lldb_r9l_x86_64,
    LLDB_INVALID_REGNUM};
uint32_t x86_register_info::g_invalidate_r10[] = {
    lldb_r10_x86_64, lldb_r10d_x86_64, lldb_r10w_x86_64, lldb_r10l_x86_64,
    LLDB_INVALID_REGNUM};
uint32_t x86_register_info::g_invalidate_r11[] = {
    lldb_r11_x86_64, lldb_r11d_x86_64, lldb_r11w_x86_64, lldb_r11l_x86_64,
    LLDB_INVALID_REGNUM};
uint32_t x86_register_info::g_invalidate_r12[] = {
    lldb_r12_x86_64, lldb_r12d_x86_64, lldb_r12w_x86_64, lldb_r12l_x86_64,
    LLDB_INVALID_REGNUM};
uint32_t x86_register_info::g_invalidate_r13[] = {
    lldb_r13_x86_64, lldb_r13d_x86_64, lldb_r13w_x86_64, lldb_r13l_x86_64,
    LLDB_INVALID_REGNUM};
uint32_t x86_register_info::g_invalidate_r14[] = {
    lldb_r14_x86_64, lldb_r14d_x86_64, lldb_r14w_x86_64, lldb_r14l_x86_64,
    LLDB_INVALID_REGNUM};
uint32_t x86_register_info::g_invalidate_r15[] = {
    lldb_r15_x86_64, lldb_r15d_x86_64, lldb_r15w_x86_64, lldb_r15l_x86_64,
    LLDB_INVALID_REGNUM};

uint32_t x86_register_info::g_contained_fip[] = {lldb_fip_x86_64,
                                                 LLDB_INVALID_REGNUM};
uint32_t x86_register_info::g_contained_fdp[] = {lldb_fdp_x86_64,
                                                 LLDB_INVALID_REGNUM};

uint32_t x86_register_info::g_invalidate_fip[] = {
    lldb_fip_x86_64, lldb_fioff_x86_64, lldb_fiseg_x86_64, LLDB_INVALID_REGNUM};
uint32_t x86_register_info::g_invalidate_fdp[] = {
    lldb_fdp_x86_64, lldb_fooff_x86_64, lldb_foseg_x86_64, LLDB_INVALID_REGNUM};

uint32_t x86_register_info::g_contained_st0_32[] = {lldb_st0_i386,
                                                    LLDB_INVALID_REGNUM};
uint32_t x86_register_info::g_contained_st1_32[] = {lldb_st1_i386,
                                                    LLDB_INVALID_REGNUM};
uint32_t x86_register_info::g_contained_st2_32[] = {lldb_st2_i386,
                                                    LLDB_INVALID_REGNUM};
uint32_t x86_register_info::g_contained_st3_32[] = {lldb_st3_i386,
                                                    LLDB_INVALID_REGNUM};
uint32_t x86_register_info::g_contained_st4_32[] = {lldb_st4_i386,
                                                    LLDB_INVALID_REGNUM};
uint32_t x86_register_info::g_contained_st5_32[] = {lldb_st5_i386,
                                                    LLDB_INVALID_REGNUM};
uint32_t x86_register_info::g_contained_st6_32[] = {lldb_st6_i386,
                                                    LLDB_INVALID_REGNUM};
uint32_t x86_register_info::g_contained_st7_32[] = {lldb_st7_i386,
                                                    LLDB_INVALID_REGNUM};

uint32_t x86_register_info::g_invalidate_st0_32[] = {
    lldb_st0_i386, lldb_mm0_i386, LLDB_INVALID_REGNUM};
uint32_t x86_register_info::g_invalidate_st1_32[] = {
    lldb_st1_i386, lldb_mm1_i386, LLDB_INVALID_REGNUM};
uint32_t x86_register_info::g_invalidate_st2_32[] = {
    lldb_st2_i386, lldb_mm2_i386, LLDB_INVALID_REGNUM};
uint32_t x86_register_info::g_invalidate_st3_32[] = {
    lldb_st3_i386, lldb_mm3_i386, LLDB_INVALID_REGNUM};
uint32_t x86_register_info::g_invalidate_st4_32[] = {
    lldb_st4_i386, lldb_mm4_i386, LLDB_INVALID_REGNUM};
uint32_t x86_register_info::g_invalidate_st5_32[] = {
    lldb_st5_i386, lldb_mm5_i386, LLDB_INVALID_REGNUM};
uint32_t x86_register_info::g_invalidate_st6_32[] = {
    lldb_st6_i386, lldb_mm6_i386, LLDB_INVALID_REGNUM};
uint32_t x86_register_info::g_invalidate_st7_32[] = {
    lldb_st7_i386, lldb_mm7_i386, LLDB_INVALID_REGNUM};

uint32_t x86_register_info::g_contained_st0_64[] = {lldb_st0_x86_64,
                                                    LLDB_INVALID_REGNUM};
uint32_t x86_register_info::g_contained_st1_64[] = {lldb_st1_x86_64,
                                                    LLDB_INVALID_REGNUM};
uint32_t x86_register_info::g_contained_st2_64[] = {lldb_st2_x86_64,
                                                    LLDB_INVALID_REGNUM};
uint32_t x86_register_info::g_contained_st3_64[] = {lldb_st3_x86_64,
                                                    LLDB_INVALID_REGNUM};
uint32_t x86_register_info::g_contained_st4_64[] = {lldb_st4_x86_64,
                                                    LLDB_INVALID_REGNUM};
uint32_t x86_register_info::g_contained_st5_64[] = {lldb_st5_x86_64,
                                                    LLDB_INVALID_REGNUM};
uint32_t x86_register_info::g_contained_st6_64[] = {lldb_st6_x86_64,
                                                    LLDB_INVALID_REGNUM};
uint32_t x86_register_info::g_contained_st7_64[] = {lldb_st7_x86_64,
                                                    LLDB_INVALID_REGNUM};

uint32_t x86_register_info::g_invalidate_st0_64[] = {
    lldb_st0_x86_64, lldb_mm0_x86_64, LLDB_INVALID_REGNUM};
uint32_t x86_register_info::g_invalidate_st1_64[] = {
    lldb_st1_x86_64, lldb_mm1_x86_64, LLDB_INVALID_REGNUM};
uint32_t x86_register_info::g_invalidate_st2_64[] = {
    lldb_st2_x86_64, lldb_mm2_x86_64, LLDB_INVALID_REGNUM};
uint32_t x86_register_info::g_invalidate_st3_64[] = {
    lldb_st3_x86_64, lldb_mm3_x86_64, LLDB_INVALID_REGNUM};
uint32_t x86_register_info::g_invalidate_st4_64[] = {
    lldb_st4_x86_64, lldb_mm4_x86_64, LLDB_INVALID_REGNUM};
uint32_t x86_register_info::g_invalidate_st5_64[] = {
    lldb_st5_x86_64, lldb_mm5_x86_64, LLDB_INVALID_REGNUM};
uint32_t x86_register_info::g_invalidate_st6_64[] = {
    lldb_st6_x86_64, lldb_mm6_x86_64, LLDB_INVALID_REGNUM};
uint32_t x86_register_info::g_invalidate_st7_64[] = {
    lldb_st7_x86_64, lldb_mm7_x86_64, LLDB_INVALID_REGNUM};

// Convert the 8-bit abridged FPU Tag Word (as found in FXSAVE) to the full
// 16-bit FPU Tag Word (as found in FSAVE, and used by gdb protocol).  This
// requires knowing the values of the ST(i) registers and the FPU Status Word.
uint16_t lldb_private::AbridgedToFullTagWord(uint8_t abridged_tw, uint16_t sw,
                                             llvm::ArrayRef<MMSReg> st_regs) {
  // Tag word is using internal FPU register numbering rather than ST(i).
  // Mapping to ST(i): i = FPU regno - TOP (Status Word, bits 11:13).
  // Here we start with FPU reg 7 and go down.
  int st = 7 - ((sw >> 11) & 7);
  uint16_t tw = 0;
  for (uint8_t mask = 0x80; mask != 0; mask >>= 1) {
    tw <<= 2;
    if (abridged_tw & mask) {
      // The register is non-empty, so we need to check the value of ST(i).
      uint16_t exp =
          st_regs[st].comp.sign_exp & 0x7fff; // Discard the sign bit.
      if (exp == 0) {
        if (st_regs[st].comp.mantissa == 0)
          tw |= 1; // Zero
        else
          tw |= 2; // Denormal
      } else if (exp == 0x7fff)
        tw |= 2; // Infinity or NaN
      // 0 if normal number
    } else
      tw |= 3; // Empty register

    // Rotate ST down.
    st = (st - 1) & 7;
  }

  return tw;
}

// Convert the 16-bit FPU Tag Word to the abridged 8-bit value, to be written
// into FXSAVE.
uint8_t lldb_private::FullToAbridgedTagWord(uint16_t tw) {
  uint8_t abridged_tw = 0;
  for (uint16_t mask = 0xc000; mask != 0; mask >>= 2) {
    abridged_tw <<= 1;
    // full TW uses 11 for empty registers, aTW uses 0
    if ((tw & mask) != mask)
      abridged_tw |= 1;
  }
  return abridged_tw;
}

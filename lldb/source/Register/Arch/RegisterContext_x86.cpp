//===-- RegisterContext_x86.cpp ---------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "RegisterContext_x86.h"
#include "lldb/Register/Arch/lldb-x86-register-enums.h"
#include "lldb/Register/Arch/lldb-x86_64-register-enums.h"
#include "lldb/lldb-defines.h"

using namespace lldb_private;

uint32_t x86_register_info::g_contained_eax[] = {registers::x86::id::eax,
                                                 LLDB_INVALID_REGNUM};
uint32_t x86_register_info::g_contained_ebx[] = {registers::x86::id::ebx,
                                                 LLDB_INVALID_REGNUM};
uint32_t x86_register_info::g_contained_ecx[] = {registers::x86::id::ecx,
                                                 LLDB_INVALID_REGNUM};
uint32_t x86_register_info::g_contained_edx[] = {registers::x86::id::edx,
                                                 LLDB_INVALID_REGNUM};
uint32_t x86_register_info::g_contained_edi[] = {registers::x86::id::edi,
                                                 LLDB_INVALID_REGNUM};
uint32_t x86_register_info::g_contained_esi[] = {registers::x86::id::esi,
                                                 LLDB_INVALID_REGNUM};
uint32_t x86_register_info::g_contained_ebp[] = {registers::x86::id::ebp,
                                                 LLDB_INVALID_REGNUM};
uint32_t x86_register_info::g_contained_esp[] = {registers::x86::id::esp,
                                                 LLDB_INVALID_REGNUM};

uint32_t x86_register_info::g_invalidate_eax[] = {registers::x86::id::eax, registers::x86::id::ax,
                                                  registers::x86::id::ah, registers::x86::id::al,
                                                  LLDB_INVALID_REGNUM};
uint32_t x86_register_info::g_invalidate_ebx[] = {registers::x86::id::ebx, registers::x86::id::bx,
                                                  registers::x86::id::bh, registers::x86::id::bl,
                                                  LLDB_INVALID_REGNUM};
uint32_t x86_register_info::g_invalidate_ecx[] = {registers::x86::id::ecx, registers::x86::id::cx,
                                                  registers::x86::id::ch, registers::x86::id::cl,
                                                  LLDB_INVALID_REGNUM};
uint32_t x86_register_info::g_invalidate_edx[] = {registers::x86::id::edx, registers::x86::id::dx,
                                                  registers::x86::id::dh, registers::x86::id::dl,
                                                  LLDB_INVALID_REGNUM};
uint32_t x86_register_info::g_invalidate_edi[] = {registers::x86::id::edi, registers::x86::id::di,
                                                  LLDB_INVALID_REGNUM};
uint32_t x86_register_info::g_invalidate_esi[] = {registers::x86::id::esi, registers::x86::id::si,
                                                  LLDB_INVALID_REGNUM};
uint32_t x86_register_info::g_invalidate_ebp[] = {registers::x86::id::ebp, registers::x86::id::bp,
                                                  LLDB_INVALID_REGNUM};
uint32_t x86_register_info::g_invalidate_esp[] = {registers::x86::id::esp, registers::x86::id::sp,
                                                  LLDB_INVALID_REGNUM};

uint32_t x86_register_info::g_contained_rax[] = {registers::x86_64::id::rax,
                                                 LLDB_INVALID_REGNUM};
uint32_t x86_register_info::g_contained_rbx[] = {registers::x86_64::id::rbx,
                                                 LLDB_INVALID_REGNUM};
uint32_t x86_register_info::g_contained_rcx[] = {registers::x86_64::id::rcx,
                                                 LLDB_INVALID_REGNUM};
uint32_t x86_register_info::g_contained_rdx[] = {registers::x86_64::id::rdx,
                                                 LLDB_INVALID_REGNUM};
uint32_t x86_register_info::g_contained_rdi[] = {registers::x86_64::id::rdi,
                                                 LLDB_INVALID_REGNUM};
uint32_t x86_register_info::g_contained_rsi[] = {registers::x86_64::id::rsi,
                                                 LLDB_INVALID_REGNUM};
uint32_t x86_register_info::g_contained_rbp[] = {registers::x86_64::id::rbp,
                                                 LLDB_INVALID_REGNUM};
uint32_t x86_register_info::g_contained_rsp[] = {registers::x86_64::id::rsp,
                                                 LLDB_INVALID_REGNUM};
uint32_t x86_register_info::g_contained_r8[] = {registers::x86_64::id::r8,
                                                LLDB_INVALID_REGNUM};
uint32_t x86_register_info::g_contained_r9[] = {registers::x86_64::id::r9,
                                                LLDB_INVALID_REGNUM};
uint32_t x86_register_info::g_contained_r10[] = {registers::x86_64::id::r10,
                                                 LLDB_INVALID_REGNUM};
uint32_t x86_register_info::g_contained_r11[] = {registers::x86_64::id::r11,
                                                 LLDB_INVALID_REGNUM};
uint32_t x86_register_info::g_contained_r12[] = {registers::x86_64::id::r12,
                                                 LLDB_INVALID_REGNUM};
uint32_t x86_register_info::g_contained_r13[] = {registers::x86_64::id::r13,
                                                 LLDB_INVALID_REGNUM};
uint32_t x86_register_info::g_contained_r14[] = {registers::x86_64::id::r14,
                                                 LLDB_INVALID_REGNUM};
uint32_t x86_register_info::g_contained_r15[] = {registers::x86_64::id::r15,
                                                 LLDB_INVALID_REGNUM};

uint32_t x86_register_info::g_invalidate_rax[] = {
    registers::x86_64::id::rax, registers::x86_64::id::eax,
    registers::x86_64::id::ax,  registers::x86_64::id::ah,
    registers::x86_64::id::al,  LLDB_INVALID_REGNUM};
uint32_t x86_register_info::g_invalidate_rbx[] = {
    registers::x86_64::id::rbx, registers::x86_64::id::ebx,
    registers::x86_64::id::bx,  registers::x86_64::id::bh,
    registers::x86_64::id::bl,  LLDB_INVALID_REGNUM};
uint32_t x86_register_info::g_invalidate_rcx[] = {
    registers::x86_64::id::rcx, registers::x86_64::id::ecx,
    registers::x86_64::id::cx,  registers::x86_64::id::ch,
    registers::x86_64::id::cl,  LLDB_INVALID_REGNUM};
uint32_t x86_register_info::g_invalidate_rdx[] = {
    registers::x86_64::id::rdx, registers::x86_64::id::edx,
    registers::x86_64::id::dx,  registers::x86_64::id::dh,
    registers::x86_64::id::dl,  LLDB_INVALID_REGNUM};
uint32_t x86_register_info::g_invalidate_rdi[] = {
    registers::x86_64::id::rdi, registers::x86_64::id::edi, registers::x86_64::id::di, registers::x86_64::id::dil,
    LLDB_INVALID_REGNUM};
uint32_t x86_register_info::g_invalidate_rsi[] = {
    registers::x86_64::id::rsi, registers::x86_64::id::esi, registers::x86_64::id::si, registers::x86_64::id::sil,
    LLDB_INVALID_REGNUM};
uint32_t x86_register_info::g_invalidate_rbp[] = {
    registers::x86_64::id::rbp, registers::x86_64::id::ebp, registers::x86_64::id::bp, registers::x86_64::id::bpl,
    LLDB_INVALID_REGNUM};
uint32_t x86_register_info::g_invalidate_rsp[] = {
    registers::x86_64::id::rsp, registers::x86_64::id::esp, registers::x86_64::id::sp, registers::x86_64::id::spl,
    LLDB_INVALID_REGNUM};
uint32_t x86_register_info::g_invalidate_r8[] = {
    registers::x86_64::id::r8, registers::x86_64::id::r8d,
    registers::x86_64::id::r8w, registers::x86_64::id::r8l,
    LLDB_INVALID_REGNUM};
uint32_t x86_register_info::g_invalidate_r9[] = {
    registers::x86_64::id::r9, registers::x86_64::id::r9d,
    registers::x86_64::id::r9w, registers::x86_64::id::r9l,
    LLDB_INVALID_REGNUM};
uint32_t x86_register_info::g_invalidate_r10[] = {
    registers::x86_64::id::r10, registers::x86_64::id::r10d,
    registers::x86_64::id::r10w, registers::x86_64::id::r10l,
    LLDB_INVALID_REGNUM};
uint32_t x86_register_info::g_invalidate_r11[] = {
    registers::x86_64::id::r11, registers::x86_64::id::r11d,
    registers::x86_64::id::r11w, registers::x86_64::id::r11l,
    LLDB_INVALID_REGNUM};
uint32_t x86_register_info::g_invalidate_r12[] = {
    registers::x86_64::id::r12, registers::x86_64::id::r12d,
    registers::x86_64::id::r12w, registers::x86_64::id::r12l,
    LLDB_INVALID_REGNUM};
uint32_t x86_register_info::g_invalidate_r13[] = {
    registers::x86_64::id::r13, registers::x86_64::id::r13d,
    registers::x86_64::id::r13w, registers::x86_64::id::r13l,
    LLDB_INVALID_REGNUM};
uint32_t x86_register_info::g_invalidate_r14[] = {
    registers::x86_64::id::r14, registers::x86_64::id::r14d,
    registers::x86_64::id::r14w, registers::x86_64::id::r14l,
    LLDB_INVALID_REGNUM};
uint32_t x86_register_info::g_invalidate_r15[] = {
    registers::x86_64::id::r15, registers::x86_64::id::r15d,
    registers::x86_64::id::r15w, registers::x86_64::id::r15l,
    LLDB_INVALID_REGNUM};

uint32_t x86_register_info::g_contained_fip[] = {registers::x86_64::id::fip,
                                                 LLDB_INVALID_REGNUM};
uint32_t x86_register_info::g_contained_fdp[] = {registers::x86_64::id::fdp,
                                                 LLDB_INVALID_REGNUM};

uint32_t x86_register_info::g_invalidate_fip[] = {
    registers::x86_64::id::fip, registers::x86_64::id::fioff,
    registers::x86_64::id::fiseg, LLDB_INVALID_REGNUM};
uint32_t x86_register_info::g_invalidate_fdp[] = {
    registers::x86_64::id::fdp, registers::x86_64::id::fooff,
    registers::x86_64::id::foseg, LLDB_INVALID_REGNUM};

uint32_t x86_register_info::g_contained_st0_32[] = {registers::x86::id::st0,
                                                    LLDB_INVALID_REGNUM};
uint32_t x86_register_info::g_contained_st1_32[] = {registers::x86::id::st1,
                                                    LLDB_INVALID_REGNUM};
uint32_t x86_register_info::g_contained_st2_32[] = {registers::x86::id::st2,
                                                    LLDB_INVALID_REGNUM};
uint32_t x86_register_info::g_contained_st3_32[] = {registers::x86::id::st3,
                                                    LLDB_INVALID_REGNUM};
uint32_t x86_register_info::g_contained_st4_32[] = {registers::x86::id::st4,
                                                    LLDB_INVALID_REGNUM};
uint32_t x86_register_info::g_contained_st5_32[] = {registers::x86::id::st5,
                                                    LLDB_INVALID_REGNUM};
uint32_t x86_register_info::g_contained_st6_32[] = {registers::x86::id::st6,
                                                    LLDB_INVALID_REGNUM};
uint32_t x86_register_info::g_contained_st7_32[] = {registers::x86::id::st7,
                                                    LLDB_INVALID_REGNUM};

uint32_t x86_register_info::g_invalidate_st0_32[] = {
    registers::x86::id::st0, registers::x86::id::mm0, LLDB_INVALID_REGNUM};
uint32_t x86_register_info::g_invalidate_st1_32[] = {
    registers::x86::id::st1, registers::x86::id::mm1, LLDB_INVALID_REGNUM};
uint32_t x86_register_info::g_invalidate_st2_32[] = {
    registers::x86::id::st2, registers::x86::id::mm2, LLDB_INVALID_REGNUM};
uint32_t x86_register_info::g_invalidate_st3_32[] = {
    registers::x86::id::st3, registers::x86::id::mm3, LLDB_INVALID_REGNUM};
uint32_t x86_register_info::g_invalidate_st4_32[] = {
    registers::x86::id::st4, registers::x86::id::mm4, LLDB_INVALID_REGNUM};
uint32_t x86_register_info::g_invalidate_st5_32[] = {
    registers::x86::id::st5, registers::x86::id::mm5, LLDB_INVALID_REGNUM};
uint32_t x86_register_info::g_invalidate_st6_32[] = {
    registers::x86::id::st6, registers::x86::id::mm6, LLDB_INVALID_REGNUM};
uint32_t x86_register_info::g_invalidate_st7_32[] = {
    registers::x86::id::st7, registers::x86::id::mm7, LLDB_INVALID_REGNUM};

uint32_t x86_register_info::g_contained_st0_64[] = {registers::x86_64::id::st0,
                                                    LLDB_INVALID_REGNUM};
uint32_t x86_register_info::g_contained_st1_64[] = {registers::x86_64::id::st1,
                                                    LLDB_INVALID_REGNUM};
uint32_t x86_register_info::g_contained_st2_64[] = {registers::x86_64::id::st2,
                                                    LLDB_INVALID_REGNUM};
uint32_t x86_register_info::g_contained_st3_64[] = {registers::x86_64::id::st3,
                                                    LLDB_INVALID_REGNUM};
uint32_t x86_register_info::g_contained_st4_64[] = {registers::x86_64::id::st4,
                                                    LLDB_INVALID_REGNUM};
uint32_t x86_register_info::g_contained_st5_64[] = {registers::x86_64::id::st5,
                                                    LLDB_INVALID_REGNUM};
uint32_t x86_register_info::g_contained_st6_64[] = {registers::x86_64::id::st6,
                                                    LLDB_INVALID_REGNUM};
uint32_t x86_register_info::g_contained_st7_64[] = {registers::x86_64::id::st7,
                                                    LLDB_INVALID_REGNUM};

uint32_t x86_register_info::g_invalidate_st0_64[] = {
    registers::x86_64::id::st0, registers::x86_64::id::mm0, LLDB_INVALID_REGNUM};
uint32_t x86_register_info::g_invalidate_st1_64[] = {
    registers::x86_64::id::st1, registers::x86_64::id::mm1, LLDB_INVALID_REGNUM};
uint32_t x86_register_info::g_invalidate_st2_64[] = {
    registers::x86_64::id::st2, registers::x86_64::id::mm2, LLDB_INVALID_REGNUM};
uint32_t x86_register_info::g_invalidate_st3_64[] = {
    registers::x86_64::id::st3, registers::x86_64::id::mm3, LLDB_INVALID_REGNUM};
uint32_t x86_register_info::g_invalidate_st4_64[] = {
    registers::x86_64::id::st4, registers::x86_64::id::mm4, LLDB_INVALID_REGNUM};
uint32_t x86_register_info::g_invalidate_st5_64[] = {
    registers::x86_64::id::st5, registers::x86_64::id::mm5, LLDB_INVALID_REGNUM};
uint32_t x86_register_info::g_invalidate_st6_64[] = {
    registers::x86_64::id::st6, registers::x86_64::id::mm6, LLDB_INVALID_REGNUM};
uint32_t x86_register_info::g_invalidate_st7_64[] = {
    registers::x86_64::id::st7, registers::x86_64::id::mm7, LLDB_INVALID_REGNUM};

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

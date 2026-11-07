//===-- RegisterInfos_x86_64_with_base_shared.cpp--------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "RegisterInfos_x86_64_with_base_shared.h"
#include "lldb/Register/Arch/lldb-x86_64-register-enums.h"
#include "lldb/Register/Arch/lldb-x86-register-enums.h"

#include "lldb/lldb-defines.h"
#include <cassert>
#include <mutex>

using namespace lldb;

namespace lldb_private {

uint32_t RegisterInfos_x86_64_with_base_shared::g_contained_eax[] = {
    registers::x86::id::eax, LLDB_INVALID_REGNUM};
uint32_t RegisterInfos_x86_64_with_base_shared::g_contained_ebx[] = {
    registers::x86::id::ebx, LLDB_INVALID_REGNUM};
uint32_t RegisterInfos_x86_64_with_base_shared::g_contained_ecx[] = {
    registers::x86::id::ecx, LLDB_INVALID_REGNUM};
uint32_t RegisterInfos_x86_64_with_base_shared::g_contained_edx[] = {
    registers::x86::id::edx, LLDB_INVALID_REGNUM};
uint32_t RegisterInfos_x86_64_with_base_shared::g_contained_edi[] = {
    registers::x86::id::edi, LLDB_INVALID_REGNUM};
uint32_t RegisterInfos_x86_64_with_base_shared::g_contained_esi[] = {
    registers::x86::id::esi, LLDB_INVALID_REGNUM};
uint32_t RegisterInfos_x86_64_with_base_shared::g_contained_ebp[] = {
    registers::x86::id::ebp, LLDB_INVALID_REGNUM};
uint32_t RegisterInfos_x86_64_with_base_shared::g_contained_esp[] = {
    registers::x86::id::esp, LLDB_INVALID_REGNUM};

uint32_t RegisterInfos_x86_64_with_base_shared::g_invalidate_eax[] = {
    registers::x86::id::eax, registers::x86::id::ax, registers::x86::id::ah, registers::x86::id::al,
    LLDB_INVALID_REGNUM};
uint32_t RegisterInfos_x86_64_with_base_shared::g_invalidate_ebx[] = {
    registers::x86::id::ebx, registers::x86::id::bx, registers::x86::id::bh, registers::x86::id::bl,
    LLDB_INVALID_REGNUM};
uint32_t RegisterInfos_x86_64_with_base_shared::g_invalidate_ecx[] = {
    registers::x86::id::ecx, registers::x86::id::cx, registers::x86::id::ch, registers::x86::id::cl,
    LLDB_INVALID_REGNUM};
uint32_t RegisterInfos_x86_64_with_base_shared::g_invalidate_edx[] = {
    registers::x86::id::edx, registers::x86::id::dx, registers::x86::id::dh, registers::x86::id::dl,
    LLDB_INVALID_REGNUM};
uint32_t RegisterInfos_x86_64_with_base_shared::g_invalidate_edi[] = {
    registers::x86::id::edi, registers::x86::id::di, LLDB_INVALID_REGNUM};
uint32_t RegisterInfos_x86_64_with_base_shared::g_invalidate_esi[] = {
    registers::x86::id::esi, registers::x86::id::si, LLDB_INVALID_REGNUM};
uint32_t RegisterInfos_x86_64_with_base_shared::g_invalidate_ebp[] = {
    registers::x86::id::ebp, registers::x86::id::bp, LLDB_INVALID_REGNUM};
uint32_t RegisterInfos_x86_64_with_base_shared::g_invalidate_esp[] = {
    registers::x86::id::esp, registers::x86::id::sp, LLDB_INVALID_REGNUM};

uint32_t RegisterInfos_x86_64_with_base_shared::g_contained_rax[] = {
    registers::x86_64::id::with_base::rax, LLDB_INVALID_REGNUM};
uint32_t RegisterInfos_x86_64_with_base_shared::g_contained_rbx[] = {
    registers::x86_64::id::with_base::rbx, LLDB_INVALID_REGNUM};
uint32_t RegisterInfos_x86_64_with_base_shared::g_contained_rcx[] = {
    registers::x86_64::id::with_base::rcx, LLDB_INVALID_REGNUM};
uint32_t RegisterInfos_x86_64_with_base_shared::g_contained_rdx[] = {
    registers::x86_64::id::with_base::rdx, LLDB_INVALID_REGNUM};
uint32_t RegisterInfos_x86_64_with_base_shared::g_contained_rdi[] = {
    registers::x86_64::id::with_base::rdi, LLDB_INVALID_REGNUM};
uint32_t RegisterInfos_x86_64_with_base_shared::g_contained_rsi[] = {
    registers::x86_64::id::with_base::rsi, LLDB_INVALID_REGNUM};
uint32_t RegisterInfos_x86_64_with_base_shared::g_contained_rbp[] = {
    registers::x86_64::id::with_base::rbp, LLDB_INVALID_REGNUM};
uint32_t RegisterInfos_x86_64_with_base_shared::g_contained_rsp[] = {
    registers::x86_64::id::with_base::rsp, LLDB_INVALID_REGNUM};
uint32_t RegisterInfos_x86_64_with_base_shared::g_contained_r8[] = {
    registers::x86_64::id::with_base::r8, LLDB_INVALID_REGNUM};
uint32_t RegisterInfos_x86_64_with_base_shared::g_contained_r9[] = {
    registers::x86_64::id::with_base::r9, LLDB_INVALID_REGNUM};
uint32_t RegisterInfos_x86_64_with_base_shared::g_contained_r10[] = {
    registers::x86_64::id::with_base::r10, LLDB_INVALID_REGNUM};
uint32_t RegisterInfos_x86_64_with_base_shared::g_contained_r11[] = {
    registers::x86_64::id::with_base::r11, LLDB_INVALID_REGNUM};
uint32_t RegisterInfos_x86_64_with_base_shared::g_contained_r12[] = {
    registers::x86_64::id::with_base::r12, LLDB_INVALID_REGNUM};
uint32_t RegisterInfos_x86_64_with_base_shared::g_contained_r13[] = {
    registers::x86_64::id::with_base::r13, LLDB_INVALID_REGNUM};
uint32_t RegisterInfos_x86_64_with_base_shared::g_contained_r14[] = {
    registers::x86_64::id::with_base::r14, LLDB_INVALID_REGNUM};
uint32_t RegisterInfos_x86_64_with_base_shared::g_contained_r15[] = {
    registers::x86_64::id::with_base::r15, LLDB_INVALID_REGNUM};

uint32_t RegisterInfos_x86_64_with_base_shared::g_invalidate_rax[] = {
    registers::x86_64::id::with_base::rax,
    registers::x86_64::id::with_base::eax,
    registers::x86_64::id::with_base::ax,
    registers::x86_64::id::with_base::ah,
    registers::x86_64::id::with_base::al,
    LLDB_INVALID_REGNUM};
uint32_t RegisterInfos_x86_64_with_base_shared::g_invalidate_rbx[] = {
    registers::x86_64::id::with_base::rbx,
    registers::x86_64::id::with_base::ebx,
    registers::x86_64::id::with_base::bx,
    registers::x86_64::id::with_base::bh,
    registers::x86_64::id::with_base::bl,
    LLDB_INVALID_REGNUM};
uint32_t RegisterInfos_x86_64_with_base_shared::g_invalidate_rcx[] = {
    registers::x86_64::id::with_base::rcx,
    registers::x86_64::id::with_base::ecx,
    registers::x86_64::id::with_base::cx,
    registers::x86_64::id::with_base::ch,
    registers::x86_64::id::with_base::cl,
    LLDB_INVALID_REGNUM};
uint32_t RegisterInfos_x86_64_with_base_shared::g_invalidate_rdx[] = {
    registers::x86_64::id::with_base::rdx,
    registers::x86_64::id::with_base::edx,
    registers::x86_64::id::with_base::dx,
    registers::x86_64::id::with_base::dh,
    registers::x86_64::id::with_base::dl,
    LLDB_INVALID_REGNUM};
uint32_t RegisterInfos_x86_64_with_base_shared::g_invalidate_rdi[] = {
    registers::x86_64::id::with_base::rdi,
    registers::x86_64::id::with_base::edi, registers::x86_64::id::with_base::di,
    registers::x86_64::id::with_base::dil, LLDB_INVALID_REGNUM};
uint32_t RegisterInfos_x86_64_with_base_shared::g_invalidate_rsi[] = {
    registers::x86_64::id::with_base::rsi,
    registers::x86_64::id::with_base::esi, registers::x86_64::id::with_base::si,
    registers::x86_64::id::with_base::sil, LLDB_INVALID_REGNUM};
uint32_t RegisterInfos_x86_64_with_base_shared::g_invalidate_rbp[] = {
    registers::x86_64::id::with_base::rbp,
    registers::x86_64::id::with_base::ebp, registers::x86_64::id::with_base::bp,
    registers::x86_64::id::with_base::bpl, LLDB_INVALID_REGNUM};
uint32_t RegisterInfos_x86_64_with_base_shared::g_invalidate_rsp[] = {
    registers::x86_64::id::with_base::rsp,
    registers::x86_64::id::with_base::esp, registers::x86_64::id::with_base::sp,
    registers::x86_64::id::with_base::spl, LLDB_INVALID_REGNUM};
uint32_t RegisterInfos_x86_64_with_base_shared::g_invalidate_r8[] = {
    registers::x86_64::id::with_base::r8, registers::x86_64::id::with_base::r8d,
    registers::x86_64::id::with_base::r8w, registers::x86_64::id::with_base::r8l,
    LLDB_INVALID_REGNUM};
uint32_t RegisterInfos_x86_64_with_base_shared::g_invalidate_r9[] = {
    registers::x86_64::id::with_base::r9, registers::x86_64::id::with_base::r9d,
    registers::x86_64::id::with_base::r9w, registers::x86_64::id::with_base::r9l,
    LLDB_INVALID_REGNUM};
uint32_t RegisterInfos_x86_64_with_base_shared::g_invalidate_r10[] = {
    registers::x86_64::id::with_base::r10, registers::x86_64::id::with_base::r10d,
    registers::x86_64::id::with_base::r10w, registers::x86_64::id::with_base::r10l,
    LLDB_INVALID_REGNUM};
uint32_t RegisterInfos_x86_64_with_base_shared::g_invalidate_r11[] = {
    registers::x86_64::id::with_base::r11, registers::x86_64::id::with_base::r11d,
    registers::x86_64::id::with_base::r11w, registers::x86_64::id::with_base::r11l,
    LLDB_INVALID_REGNUM};
uint32_t RegisterInfos_x86_64_with_base_shared::g_invalidate_r12[] = {
    registers::x86_64::id::with_base::r12, registers::x86_64::id::with_base::r12d,
    registers::x86_64::id::with_base::r12w, registers::x86_64::id::with_base::r12l,
    LLDB_INVALID_REGNUM};
uint32_t RegisterInfos_x86_64_with_base_shared::g_invalidate_r13[] = {
    registers::x86_64::id::with_base::r13, registers::x86_64::id::with_base::r13d,
    registers::x86_64::id::with_base::r13w, registers::x86_64::id::with_base::r13l,
    LLDB_INVALID_REGNUM};
uint32_t RegisterInfos_x86_64_with_base_shared::g_invalidate_r14[] = {
    registers::x86_64::id::with_base::r14, registers::x86_64::id::with_base::r14d,
    registers::x86_64::id::with_base::r14w, registers::x86_64::id::with_base::r14l,
    LLDB_INVALID_REGNUM};
uint32_t RegisterInfos_x86_64_with_base_shared::g_invalidate_r15[] = {
    registers::x86_64::id::with_base::r15, registers::x86_64::id::with_base::r15d,
    registers::x86_64::id::with_base::r15w, registers::x86_64::id::with_base::r15l,
    LLDB_INVALID_REGNUM};

uint32_t RegisterInfos_x86_64_with_base_shared::g_contained_fip[] = {
    registers::x86_64::id::with_base::fip, LLDB_INVALID_REGNUM};
uint32_t RegisterInfos_x86_64_with_base_shared::g_contained_fdp[] = {
    registers::x86_64::id::with_base::fdp, LLDB_INVALID_REGNUM};

uint32_t RegisterInfos_x86_64_with_base_shared::g_invalidate_fip[] = {
    registers::x86_64::id::with_base::fip,
    registers::x86_64::id::with_base::fioff,
    registers::x86_64::id::with_base::fiseg, LLDB_INVALID_REGNUM};
uint32_t RegisterInfos_x86_64_with_base_shared::g_invalidate_fdp[] = {
    registers::x86_64::id::with_base::fdp,
    registers::x86_64::id::with_base::fooff,
    registers::x86_64::id::with_base::foseg, LLDB_INVALID_REGNUM};

uint32_t RegisterInfos_x86_64_with_base_shared::g_contained_st0_32[] = {
    registers::x86::id::st0, LLDB_INVALID_REGNUM};
uint32_t RegisterInfos_x86_64_with_base_shared::g_contained_st1_32[] = {
    registers::x86::id::st1, LLDB_INVALID_REGNUM};
uint32_t RegisterInfos_x86_64_with_base_shared::g_contained_st2_32[] = {
    registers::x86::id::st2, LLDB_INVALID_REGNUM};
uint32_t RegisterInfos_x86_64_with_base_shared::g_contained_st3_32[] = {
    registers::x86::id::st3, LLDB_INVALID_REGNUM};
uint32_t RegisterInfos_x86_64_with_base_shared::g_contained_st4_32[] = {
    registers::x86::id::st4, LLDB_INVALID_REGNUM};
uint32_t RegisterInfos_x86_64_with_base_shared::g_contained_st5_32[] = {
    registers::x86::id::st5, LLDB_INVALID_REGNUM};
uint32_t RegisterInfos_x86_64_with_base_shared::g_contained_st6_32[] = {
    registers::x86::id::st6, LLDB_INVALID_REGNUM};
uint32_t RegisterInfos_x86_64_with_base_shared::g_contained_st7_32[] = {
    registers::x86::id::st7, LLDB_INVALID_REGNUM};

uint32_t RegisterInfos_x86_64_with_base_shared::g_invalidate_st0_32[] = {
    registers::x86::id::st0, registers::x86::id::mm0, LLDB_INVALID_REGNUM};
uint32_t RegisterInfos_x86_64_with_base_shared::g_invalidate_st1_32[] = {
    registers::x86::id::st1, registers::x86::id::mm1, LLDB_INVALID_REGNUM};
uint32_t RegisterInfos_x86_64_with_base_shared::g_invalidate_st2_32[] = {
    registers::x86::id::st2, registers::x86::id::mm2, LLDB_INVALID_REGNUM};
uint32_t RegisterInfos_x86_64_with_base_shared::g_invalidate_st3_32[] = {
    registers::x86::id::st3, registers::x86::id::mm3, LLDB_INVALID_REGNUM};
uint32_t RegisterInfos_x86_64_with_base_shared::g_invalidate_st4_32[] = {
    registers::x86::id::st4, registers::x86::id::mm4, LLDB_INVALID_REGNUM};
uint32_t RegisterInfos_x86_64_with_base_shared::g_invalidate_st5_32[] = {
    registers::x86::id::st5, registers::x86::id::mm5, LLDB_INVALID_REGNUM};
uint32_t RegisterInfos_x86_64_with_base_shared::g_invalidate_st6_32[] = {
    registers::x86::id::st6, registers::x86::id::mm6, LLDB_INVALID_REGNUM};
uint32_t RegisterInfos_x86_64_with_base_shared::g_invalidate_st7_32[] = {
    registers::x86::id::st7, registers::x86::id::mm7, LLDB_INVALID_REGNUM};

uint32_t RegisterInfos_x86_64_with_base_shared::g_contained_st0_64[] = {
    registers::x86_64::id::with_base::st0, LLDB_INVALID_REGNUM};
uint32_t RegisterInfos_x86_64_with_base_shared::g_contained_st1_64[] = {
    registers::x86_64::id::with_base::st1, LLDB_INVALID_REGNUM};
uint32_t RegisterInfos_x86_64_with_base_shared::g_contained_st2_64[] = {
    registers::x86_64::id::with_base::st2, LLDB_INVALID_REGNUM};
uint32_t RegisterInfos_x86_64_with_base_shared::g_contained_st3_64[] = {
    registers::x86_64::id::with_base::st3, LLDB_INVALID_REGNUM};
uint32_t RegisterInfos_x86_64_with_base_shared::g_contained_st4_64[] = {
    registers::x86_64::id::with_base::st4, LLDB_INVALID_REGNUM};
uint32_t RegisterInfos_x86_64_with_base_shared::g_contained_st5_64[] = {
    registers::x86_64::id::with_base::st5, LLDB_INVALID_REGNUM};
uint32_t RegisterInfos_x86_64_with_base_shared::g_contained_st6_64[] = {
    registers::x86_64::id::with_base::st6, LLDB_INVALID_REGNUM};
uint32_t RegisterInfos_x86_64_with_base_shared::g_contained_st7_64[] = {
    registers::x86_64::id::with_base::st7, LLDB_INVALID_REGNUM};

uint32_t RegisterInfos_x86_64_with_base_shared::g_invalidate_st0_64[] = {
    registers::x86_64::id::with_base::st0, registers::x86_64::id::with_base::mm0,
    LLDB_INVALID_REGNUM};
uint32_t RegisterInfos_x86_64_with_base_shared::g_invalidate_st1_64[] = {
    registers::x86_64::id::with_base::st1, registers::x86_64::id::with_base::mm1,
    LLDB_INVALID_REGNUM};
uint32_t RegisterInfos_x86_64_with_base_shared::g_invalidate_st2_64[] = {
    registers::x86_64::id::with_base::st2, registers::x86_64::id::with_base::mm2,
    LLDB_INVALID_REGNUM};
uint32_t RegisterInfos_x86_64_with_base_shared::g_invalidate_st3_64[] = {
    registers::x86_64::id::with_base::st3, registers::x86_64::id::with_base::mm3,
    LLDB_INVALID_REGNUM};
uint32_t RegisterInfos_x86_64_with_base_shared::g_invalidate_st4_64[] = {
    registers::x86_64::id::with_base::st4, registers::x86_64::id::with_base::mm4,
    LLDB_INVALID_REGNUM};
uint32_t RegisterInfos_x86_64_with_base_shared::g_invalidate_st5_64[] = {
    registers::x86_64::id::with_base::st5, registers::x86_64::id::with_base::mm5,
    LLDB_INVALID_REGNUM};
uint32_t RegisterInfos_x86_64_with_base_shared::g_invalidate_st6_64[] = {
    registers::x86_64::id::with_base::st6, registers::x86_64::id::with_base::mm6,
    LLDB_INVALID_REGNUM};
uint32_t RegisterInfos_x86_64_with_base_shared::g_invalidate_st7_64[] = {
    registers::x86_64::id::with_base::st7, registers::x86_64::id::with_base::mm7,
    LLDB_INVALID_REGNUM};

RegInfo &GetRegInfoShared(llvm::Triple::ArchType arch_type, bool with_base) {
  static std::once_flag once_flag_x86, once_flag_x86_64,
      once_flag_x86_64_with_base;
  static RegInfo reg_info_x86, reg_info_x86_64, reg_info_x86_64_with_base, reg_info_invalid;

  switch (arch_type) {
  case llvm::Triple::x86:
    std::call_once(once_flag_x86, []() {
      reg_info_x86.num_registers = registers::x86::id::k_num_registers;
      reg_info_x86.num_gpr_registers = registers::x86::id::k_num_gpr_registers;
      reg_info_x86.num_fpr_registers = registers::x86::id::k_num_fpr_registers;
      reg_info_x86.num_avx_registers = registers::x86::id::k_num_avx_registers;
      reg_info_x86.last_gpr = registers::x86::id::k_last_gpr;
      reg_info_x86.first_fpr = registers::x86::id::k_first_fpr;
      reg_info_x86.last_fpr = registers::x86::id::k_last_fpr;
      reg_info_x86.first_st = registers::x86::id::st0;
      reg_info_x86.last_st = registers::x86::id::st7;
      reg_info_x86.first_mm = registers::x86::id::mm0;
      reg_info_x86.last_mm = registers::x86::id::mm7;
      reg_info_x86.first_xmm = registers::x86::id::xmm0;
      reg_info_x86.last_xmm = registers::x86::id::xmm7;
      reg_info_x86.first_ymm = registers::x86::id::ymm0;
      reg_info_x86.last_ymm = registers::x86::id::ymm7;
      reg_info_x86.first_dr = registers::x86::id::dr0;
      reg_info_x86.gpr_flags = registers::x86::id::eflags;
    });

    return reg_info_x86;
  case llvm::Triple::x86_64:
    if (with_base) {
      std::call_once(once_flag_x86_64_with_base, []() {
        reg_info_x86_64_with_base.num_registers =
            registers::x86_64::id::with_base::k_num_registers;
        reg_info_x86_64_with_base.num_gpr_registers =
            registers::x86_64::id::with_base::k_num_gpr_registers;
        reg_info_x86_64_with_base.num_fpr_registers =
            registers::x86_64::id::with_base::k_num_fpr_registers;
        reg_info_x86_64_with_base.num_avx_registers =
            registers::x86_64::id::with_base::k_num_avx_registers;
        reg_info_x86_64_with_base.last_gpr =
            registers::x86_64::id::with_base::k_last_gpr;
        reg_info_x86_64_with_base.first_fpr =
            registers::x86_64::id::with_base::k_first_fpr;
        reg_info_x86_64_with_base.last_fpr =
            registers::x86_64::id::with_base::k_last_fpr;
        reg_info_x86_64_with_base.first_st =
            registers::x86_64::id::with_base::st0;
        reg_info_x86_64_with_base.last_st =
            registers::x86_64::id::with_base::st7;
        reg_info_x86_64_with_base.first_mm =
            registers::x86_64::id::with_base::mm0;
        reg_info_x86_64_with_base.last_mm =
            registers::x86_64::id::with_base::mm7;
        reg_info_x86_64_with_base.first_xmm =
            registers::x86_64::id::with_base::xmm0;
        reg_info_x86_64_with_base.last_xmm =
            registers::x86_64::id::with_base::xmm15;
        reg_info_x86_64_with_base.first_ymm =
            registers::x86_64::id::with_base::ymm0;
        reg_info_x86_64_with_base.last_ymm =
            registers::x86_64::id::with_base::ymm15;
        reg_info_x86_64_with_base.first_dr =
            registers::x86_64::id::with_base::dr0;
        reg_info_x86_64_with_base.gpr_flags =
            registers::x86_64::id::with_base::rflags;
      });

      return reg_info_x86_64_with_base;
    } else {
      std::call_once(once_flag_x86_64, []() {
        reg_info_x86_64.num_registers = registers::x86_64::id::k_num_registers;
        reg_info_x86_64.num_gpr_registers =
            registers::x86_64::id::k_num_gpr_registers;
        reg_info_x86_64.num_fpr_registers =
            registers::x86_64::id::k_num_fpr_registers;
        reg_info_x86_64.num_avx_registers =
            registers::x86_64::id::k_num_avx_registers;
        reg_info_x86_64.last_gpr = registers::x86_64::id::k_last_gpr;
        reg_info_x86_64.first_fpr = registers::x86_64::id::k_first_fpr;
        reg_info_x86_64.last_fpr = registers::x86_64::id::k_last_fpr;
        reg_info_x86_64.first_st = registers::x86_64::id::st0;
        reg_info_x86_64.last_st = registers::x86_64::id::st7;
        reg_info_x86_64.first_mm = registers::x86_64::id::mm0;
        reg_info_x86_64.last_mm = registers::x86_64::id::mm7;
        reg_info_x86_64.first_xmm = registers::x86_64::id::xmm0;
        reg_info_x86_64.last_xmm = registers::x86_64::id::xmm15;
        reg_info_x86_64.first_ymm = registers::x86_64::id::ymm0;
        reg_info_x86_64.last_ymm = registers::x86_64::id::ymm15;
        reg_info_x86_64.first_dr = registers::x86_64::id::dr0;
        reg_info_x86_64.gpr_flags = registers::x86_64::id::rflags;
      });
      return reg_info_x86_64;
    }
  default:
    assert(false && "Unhandled target architecture.");
    return reg_info_invalid;
  }
}

} // namespace lldb_private

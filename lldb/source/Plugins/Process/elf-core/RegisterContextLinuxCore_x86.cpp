//===-- RegisterContextLinuxCore_x86.cpp ----------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "RegisterContextLinuxCore_x86.h"
#include "lldb/Register/Arch/lldb-x86_64-register-enums.h"
#include "lldb/Register/Arch/lldb-x86-register-enums.h"
#include "lldb/Target/Thread.h"
#include "lldb/Utility/DataExtractor.h"
#include "lldb/Register/RegisterValue.h"

using namespace lldb_private;

const uint32_t g_gpr_regnums_i386[] = {
    registers::x86::id::eax, registers::x86::id::ebx,
    registers::x86::id::ecx, registers::x86::id::edx,
    registers::x86::id::edi, registers::x86::id::esi,
    registers::x86::id::ebp, registers::x86::id::esp,
    registers::x86::id::eip, registers::x86::id::eflags,
    registers::x86::id::cs,  registers::x86::id::fs,
    registers::x86::id::gs,  registers::x86::id::ss,
    registers::x86::id::ds,  registers::x86::id::es,
    registers::x86::id::ax,  registers::x86::id::bx,
    registers::x86::id::cx,  registers::x86::id::dx,
    registers::x86::id::di,  registers::x86::id::si,
    registers::x86::id::bp,  registers::x86::id::sp,
    registers::x86::id::ah,  registers::x86::id::bh,
    registers::x86::id::ch,  registers::x86::id::dh,
    registers::x86::id::al,  registers::x86::id::bl,
    registers::x86::id::cl,  registers::x86::id::dl,
    LLDB_INVALID_REGNUM, // Register sets must be terminated with
                         // LLDB_INVALID_REGNUM.
};
static_assert((sizeof(g_gpr_regnums_i386) / sizeof(g_gpr_regnums_i386[0])) -
                      1 ==
                  registers::x86::id::k_num_gpr_registers,
              "g_gpr_regnums_i386 has wrong number of register infos");

const uint32_t g_lldb_regnums_i386[] = {
    registers::x86::id::fctrl, registers::x86::id::fstat,
    registers::x86::id::ftag,  registers::x86::id::fop,
    registers::x86::id::fiseg, registers::x86::id::fioff,
    registers::x86::id::foseg, registers::x86::id::fooff,
    registers::x86::id::mxcsr, registers::x86::id::mxcsrmask,
    registers::x86::id::st0,   registers::x86::id::st1,
    registers::x86::id::st2,   registers::x86::id::st3,
    registers::x86::id::st4,   registers::x86::id::st5,
    registers::x86::id::st6,   registers::x86::id::st7,
    registers::x86::id::mm0,   registers::x86::id::mm1,
    registers::x86::id::mm2,   registers::x86::id::mm3,
    registers::x86::id::mm4,   registers::x86::id::mm5,
    registers::x86::id::mm6,   registers::x86::id::mm7,
    registers::x86::id::xmm0,  registers::x86::id::xmm1,
    registers::x86::id::xmm2,  registers::x86::id::xmm3,
    registers::x86::id::xmm4,  registers::x86::id::xmm5,
    registers::x86::id::xmm6,  registers::x86::id::xmm7,
    LLDB_INVALID_REGNUM // Register sets must be terminated with
                        // LLDB_INVALID_REGNUM.
};
static_assert((sizeof(g_lldb_regnums_i386) / sizeof(g_lldb_regnums_i386[0])) -
                      1 ==
                  registers::x86::id::k_num_fpr_registers,
              "g_lldb_regnums_i386 has wrong number of register infos");

const uint32_t g_avx_regnums_i386[] = {
    registers::x86::id::ymm0, registers::x86::id::ymm1,
    registers::x86::id::ymm2, registers::x86::id::ymm3,
    registers::x86::id::ymm4, registers::x86::id::ymm5,
    registers::x86::id::ymm6, registers::x86::id::ymm7,
    LLDB_INVALID_REGNUM // Register sets must be terminated with
                        // LLDB_INVALID_REGNUM.
};
static_assert((sizeof(g_avx_regnums_i386) / sizeof(g_avx_regnums_i386[0])) -
                      1 ==
                  registers::x86::id::k_num_avx_registers,
              " g_avx_regnums_i386 has wrong number of register infos");

static const uint32_t g_gpr_regnums_x86_64[] = {
    registers::x86_64::id::with_base::rax,
    registers::x86_64::id::with_base::rbx,
    registers::x86_64::id::with_base::rcx,
    registers::x86_64::id::with_base::rdx,
    registers::x86_64::id::with_base::rdi,
    registers::x86_64::id::with_base::rsi,
    registers::x86_64::id::with_base::rbp,
    registers::x86_64::id::with_base::rsp,
    registers::x86_64::id::with_base::r8,
    registers::x86_64::id::with_base::r9,
    registers::x86_64::id::with_base::r10,
    registers::x86_64::id::with_base::r11,
    registers::x86_64::id::with_base::r12,
    registers::x86_64::id::with_base::r13,
    registers::x86_64::id::with_base::r14,
    registers::x86_64::id::with_base::r15,
    registers::x86_64::id::with_base::rip,
    registers::x86_64::id::with_base::rflags,
    registers::x86_64::id::with_base::cs,
    registers::x86_64::id::with_base::fs,
    registers::x86_64::id::with_base::gs,
    registers::x86_64::id::with_base::ss,
    registers::x86_64::id::with_base::fs_base,
    registers::x86_64::id::with_base::gs_base,
    registers::x86_64::id::with_base::ds,
    registers::x86_64::id::with_base::es,
    registers::x86_64::id::with_base::eax,
    registers::x86_64::id::with_base::ebx,
    registers::x86_64::id::with_base::ecx,
    registers::x86_64::id::with_base::edx,
    registers::x86_64::id::with_base::edi,
    registers::x86_64::id::with_base::esi,
    registers::x86_64::id::with_base::ebp,
    registers::x86_64::id::with_base::esp,
    registers::x86_64::id::with_base::r8d,  // Low 32 bits or r8
    registers::x86_64::id::with_base::r9d,  // Low 32 bits or r9
    registers::x86_64::id::with_base::r10d, // Low 32 bits or r10
    registers::x86_64::id::with_base::r11d, // Low 32 bits or r11
    registers::x86_64::id::with_base::r12d, // Low 32 bits or r12
    registers::x86_64::id::with_base::r13d, // Low 32 bits or r13
    registers::x86_64::id::with_base::r14d, // Low 32 bits or r14
    registers::x86_64::id::with_base::r15d, // Low 32 bits or r15
    registers::x86_64::id::with_base::ax,
    registers::x86_64::id::with_base::bx,
    registers::x86_64::id::with_base::cx,
    registers::x86_64::id::with_base::dx,
    registers::x86_64::id::with_base::di,
    registers::x86_64::id::with_base::si,
    registers::x86_64::id::with_base::bp,
    registers::x86_64::id::with_base::sp,
    registers::x86_64::id::with_base::r8w,  // Low 16 bits or r8
    registers::x86_64::id::with_base::r9w,  // Low 16 bits or r9
    registers::x86_64::id::with_base::r10w, // Low 16 bits or r10
    registers::x86_64::id::with_base::r11w, // Low 16 bits or r11
    registers::x86_64::id::with_base::r12w, // Low 16 bits or r12
    registers::x86_64::id::with_base::r13w, // Low 16 bits or r13
    registers::x86_64::id::with_base::r14w, // Low 16 bits or r14
    registers::x86_64::id::with_base::r15w, // Low 16 bits or r15
    registers::x86_64::id::with_base::ah,
    registers::x86_64::id::with_base::bh,
    registers::x86_64::id::with_base::ch,
    registers::x86_64::id::with_base::dh,
    registers::x86_64::id::with_base::al,
    registers::x86_64::id::with_base::bl,
    registers::x86_64::id::with_base::cl,
    registers::x86_64::id::with_base::dl,
    registers::x86_64::id::with_base::dil,
    registers::x86_64::id::with_base::sil,
    registers::x86_64::id::with_base::bpl,
    registers::x86_64::id::with_base::spl,
    registers::x86_64::id::with_base::r8l,  // Low 8 bits or r8
    registers::x86_64::id::with_base::r9l,  // Low 8 bits or r9
    registers::x86_64::id::with_base::r10l, // Low 8 bits or r10
    registers::x86_64::id::with_base::r11l, // Low 8 bits or r11
    registers::x86_64::id::with_base::r12l, // Low 8 bits or r12
    registers::x86_64::id::with_base::r13l, // Low 8 bits or r13
    registers::x86_64::id::with_base::r14l, // Low 8 bits or r14
    registers::x86_64::id::with_base::r15l, // Low 8 bits or r15
    LLDB_INVALID_REGNUM          // Register sets must be terminated with
                                 // LLDB_INVALID_REGNUM.
};
static_assert((sizeof(g_gpr_regnums_x86_64) / sizeof(g_gpr_regnums_x86_64[0])) -
                      1 ==
                  registers::x86_64::id::with_base::k_num_gpr_registers,
              "g_gpr_regnums_x86_64 has wrong number of register infos");

static const uint32_t g_lldb_regnums_x86_64[] = {
    registers::x86_64::id::with_base::fctrl,
    registers::x86_64::id::with_base::fstat,
    registers::x86_64::id::with_base::ftag,
    registers::x86_64::id::with_base::fop,
    registers::x86_64::id::with_base::fiseg,
    registers::x86_64::id::with_base::fioff,
    registers::x86_64::id::with_base::fip,
    registers::x86_64::id::with_base::foseg,
    registers::x86_64::id::with_base::fooff,
    registers::x86_64::id::with_base::fdp,
    registers::x86_64::id::with_base::mxcsr,
    registers::x86_64::id::with_base::mxcsrmask,
    registers::x86_64::id::with_base::st0,
    registers::x86_64::id::with_base::st1,
    registers::x86_64::id::with_base::st2,
    registers::x86_64::id::with_base::st3,
    registers::x86_64::id::with_base::st4,
    registers::x86_64::id::with_base::st5,
    registers::x86_64::id::with_base::st6,
    registers::x86_64::id::with_base::st7,
    registers::x86_64::id::with_base::mm0,
    registers::x86_64::id::with_base::mm1,
    registers::x86_64::id::with_base::mm2,
    registers::x86_64::id::with_base::mm3,
    registers::x86_64::id::with_base::mm4,
    registers::x86_64::id::with_base::mm5,
    registers::x86_64::id::with_base::mm6,
    registers::x86_64::id::with_base::mm7,
    registers::x86_64::id::with_base::xmm0,
    registers::x86_64::id::with_base::xmm1,
    registers::x86_64::id::with_base::xmm2,
    registers::x86_64::id::with_base::xmm3,
    registers::x86_64::id::with_base::xmm4,
    registers::x86_64::id::with_base::xmm5,
    registers::x86_64::id::with_base::xmm6,
    registers::x86_64::id::with_base::xmm7,
    registers::x86_64::id::with_base::xmm8,
    registers::x86_64::id::with_base::xmm9,
    registers::x86_64::id::with_base::xmm10,
    registers::x86_64::id::with_base::xmm11,
    registers::x86_64::id::with_base::xmm12,
    registers::x86_64::id::with_base::xmm13,
    registers::x86_64::id::with_base::xmm14,
    registers::x86_64::id::with_base::xmm15,
    LLDB_INVALID_REGNUM // Register sets must be terminated with
                        // LLDB_INVALID_REGNUM.
};
static_assert(
    (sizeof(g_lldb_regnums_x86_64) / sizeof(g_lldb_regnums_x86_64[0])) - 1 ==
        registers::x86_64::id::with_base::k_num_fpr_registers,
    "g_lldb_regnums_x86_64 has wrong number of register infos");

static const uint32_t g_avx_regnums_x86_64[] = {
    registers::x86_64::id::with_base::ymm0,
    registers::x86_64::id::with_base::ymm1,
    registers::x86_64::id::with_base::ymm2,
    registers::x86_64::id::with_base::ymm3,
    registers::x86_64::id::with_base::ymm4,
    registers::x86_64::id::with_base::ymm5,
    registers::x86_64::id::with_base::ymm6,
    registers::x86_64::id::with_base::ymm7,
    registers::x86_64::id::with_base::ymm8,
    registers::x86_64::id::with_base::ymm9,
    registers::x86_64::id::with_base::ymm10,
    registers::x86_64::id::with_base::ymm11,
    registers::x86_64::id::with_base::ymm12,
    registers::x86_64::id::with_base::ymm13,
    registers::x86_64::id::with_base::ymm14,
    registers::x86_64::id::with_base::ymm15,
    LLDB_INVALID_REGNUM // Register sets must be terminated with
                        // LLDB_INVALID_REGNUM.
};
static_assert((sizeof(g_avx_regnums_x86_64) / sizeof(g_avx_regnums_x86_64[0])) -
                      1 ==
                  registers::x86_64::id::with_base::k_num_avx_registers,
              "g_avx_regnums_x86_64 has wrong number of register infos");

static const RegisterSet g_reg_sets_i386[] = {
    {"General Purpose Registers", "gpr", registers::x86::id::k_num_gpr_registers,
     g_gpr_regnums_i386},
    {"Floating Point Registers", "fpu", registers::x86::id::k_num_fpr_registers,
     g_lldb_regnums_i386},
    {"Advanced Vector Extensions", "avx", registers::x86::id::k_num_avx_registers,
     g_avx_regnums_i386}};

static const RegisterSet g_reg_sets_x86_64[] = {
    {"General Purpose Registers", "gpr",
     registers::x86_64::id::with_base::k_num_gpr_registers,
     g_gpr_regnums_x86_64},
    {"Floating Point Registers", "fpu",
     registers::x86_64::id::with_base::k_num_fpr_registers,
     g_lldb_regnums_x86_64},
    {"Advanced Vector Extensions", "avx",
     registers::x86_64::id::with_base::k_num_avx_registers,
     g_avx_regnums_x86_64}};

RegisterContextLinuxCore_x86::RegisterContextLinuxCore_x86(
    Thread &thread, RegisterInfoInterface *register_info,
    const DataExtractor &gpregset, llvm::ArrayRef<CoreNote> notes)
    : RegisterContextELFCore_x86(thread, register_info, gpregset, notes) {}

const RegisterSet *RegisterContextLinuxCore_x86::GetRegisterSet(size_t set) {
  if (IsRegisterSetAvailable(set)) {
    switch (m_register_info_up->GetTargetArchitecture().GetMachine()) {
    case llvm::Triple::x86:
      return &g_reg_sets_i386[set];
    case llvm::Triple::x86_64:
      return &g_reg_sets_x86_64[set];
    default:
      assert(false && "Unhandled target architecture.");
      return nullptr;
    }
  }
  return nullptr;
}

RegInfo &RegisterContextLinuxCore_x86::GetRegInfo() {
  return GetRegInfoShared(
      m_register_info_up->GetTargetArchitecture().GetMachine(),
      /*with_base=*/true);
}

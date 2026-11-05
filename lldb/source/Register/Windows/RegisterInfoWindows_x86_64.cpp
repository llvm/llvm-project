//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "RegisterInfoWindows_x86_64.h"
#include "lldb/Register/Arch/lldb-x86_64-ehframe-register-enums.h"
#include "Register/Arch/RegisterContext_x86.h"
#include "lldb/Register/Arch/lldb-x86_64-dwarf-register-enums.h"
#include "lldb/Register/Arch/lldb-x86_64-register-enums.h"
#include "lldb/lldb-defines.h"

#include <vector>


namespace dwarf_regs = lldb_private::registers::x86_64::dwarf;
namespace ehframe_regs = lldb_private::registers::x86_64::ehframe;
namespace reg_ids = lldb_private::registers::x86_64::id;
using namespace lldb_private;
using namespace lldb;

namespace {
typedef struct _GPR {
  uint64_t rax;
  uint64_t rcx;
  uint64_t rdx;
  uint64_t rbx;
  uint64_t rsp;
  uint64_t rbp;
  uint64_t rsi;
  uint64_t rdi;
  uint64_t r8;
  uint64_t r9;
  uint64_t r10;
  uint64_t r11;
  uint64_t r12;
  uint64_t r13;
  uint64_t r14;
  uint64_t r15;
  uint64_t rip;
  uint64_t rflags;
  uint16_t cs;
  uint16_t fs;
  uint16_t gs;
  uint16_t ss;
  uint16_t ds;
  uint16_t es;
} GPR;

#define GPR_OFFSET(regname) (LLVM_EXTENSION offsetof(GPR, regname))
#define DEFINE_GPR(reg, alt, kind1, kind2, kind3, kind4)                       \
  {                                                                            \
#reg, alt, sizeof(((GPR *)nullptr)->reg), GPR_OFFSET(reg), eEncodingUint,  \
        eFormatHex,                                                            \
        {kind1, kind2, kind3, kind4, reg_ids::reg }, nullptr, nullptr,  \
        nullptr,                                                               \
  }

typedef struct _FPReg {
  XMMReg xmm0;
  XMMReg xmm1;
  XMMReg xmm2;
  XMMReg xmm3;
  XMMReg xmm4;
  XMMReg xmm5;
  XMMReg xmm6;
  XMMReg xmm7;
  XMMReg xmm8;
  XMMReg xmm9;
  XMMReg xmm10;
  XMMReg xmm11;
  XMMReg xmm12;
  XMMReg xmm13;
  XMMReg xmm14;
  XMMReg xmm15;
} FPReg;

#define FPR_OFFSET(regname)                                                    \
  (sizeof(GPR) + LLVM_EXTENSION offsetof(FPReg, regname))

#define DEFINE_XMM(reg)                                                        \
  {                                                                            \
#reg, NULL, sizeof(((FPReg *)nullptr)->reg), FPR_OFFSET(reg),              \
        eEncodingUint, eFormatVectorOfUInt64,                                  \
        {ehframe_regs::reg, dwarf_regs::reg, LLDB_INVALID_REGNUM,      \
         LLDB_INVALID_REGNUM, reg_ids::reg },                           \
         nullptr, nullptr, nullptr,                                            \
  }

// clang-format off
static RegisterInfo g_register_infos_x86_64[] = {
// General purpose registers     EH_Frame              DWARF                 Generic                     Process Plugin
//  ===========================  ==================    ================      =========================   ====================
    DEFINE_GPR(rax,    nullptr,  ehframe_regs::rax,     dwarf_regs::rax,     LLDB_INVALID_REGNUM,        LLDB_INVALID_REGNUM),
    DEFINE_GPR(rbx,    nullptr,  ehframe_regs::rbx,     dwarf_regs::rbx,     LLDB_INVALID_REGNUM,        LLDB_INVALID_REGNUM),
    DEFINE_GPR(rcx,    nullptr,  ehframe_regs::rcx,     dwarf_regs::rcx,     LLDB_REGNUM_GENERIC_ARG1,   LLDB_INVALID_REGNUM),
    DEFINE_GPR(rdx,    nullptr,  ehframe_regs::rdx,     dwarf_regs::rdx,     LLDB_REGNUM_GENERIC_ARG2,   LLDB_INVALID_REGNUM),
    DEFINE_GPR(rdi,    nullptr,  ehframe_regs::rdi,     dwarf_regs::rdi,     LLDB_INVALID_REGNUM,        LLDB_INVALID_REGNUM),
    DEFINE_GPR(rsi,    nullptr,  ehframe_regs::rsi,     dwarf_regs::rsi,     LLDB_INVALID_REGNUM,        LLDB_INVALID_REGNUM),
    DEFINE_GPR(rbp,    nullptr,  ehframe_regs::rbp,     dwarf_regs::rbp,     LLDB_REGNUM_GENERIC_FP,     LLDB_INVALID_REGNUM),
    DEFINE_GPR(rsp,    nullptr,  ehframe_regs::rsp,     dwarf_regs::rsp,     LLDB_REGNUM_GENERIC_SP,     LLDB_INVALID_REGNUM),
    DEFINE_GPR(r8,     nullptr,  ehframe_regs::r8,      dwarf_regs::r8,      LLDB_REGNUM_GENERIC_ARG3,   LLDB_INVALID_REGNUM),
    DEFINE_GPR(r9,     nullptr,  ehframe_regs::r9,      dwarf_regs::r9,      LLDB_REGNUM_GENERIC_ARG4,   LLDB_INVALID_REGNUM),
    DEFINE_GPR(r10,    nullptr,  ehframe_regs::r10,     dwarf_regs::r10,     LLDB_INVALID_REGNUM,        LLDB_INVALID_REGNUM),
    DEFINE_GPR(r11,    nullptr,  ehframe_regs::r11,     dwarf_regs::r11,     LLDB_INVALID_REGNUM,        LLDB_INVALID_REGNUM),
    DEFINE_GPR(r12,    nullptr,  ehframe_regs::r12,     dwarf_regs::r12,     LLDB_INVALID_REGNUM,        LLDB_INVALID_REGNUM),
    DEFINE_GPR(r13,    nullptr,  ehframe_regs::r13,     dwarf_regs::r13,     LLDB_INVALID_REGNUM,        LLDB_INVALID_REGNUM),
    DEFINE_GPR(r14,    nullptr,  ehframe_regs::r14,     dwarf_regs::r14,     LLDB_INVALID_REGNUM,        LLDB_INVALID_REGNUM),
    DEFINE_GPR(r15,    nullptr,  ehframe_regs::r15,     dwarf_regs::r15,     LLDB_INVALID_REGNUM,        LLDB_INVALID_REGNUM),
    DEFINE_GPR(rip,    nullptr,  ehframe_regs::rip,     dwarf_regs::rip,     LLDB_REGNUM_GENERIC_PC,     LLDB_INVALID_REGNUM),
    DEFINE_GPR(rflags, nullptr,  ehframe_regs::rflags,  dwarf_regs::rflags,  LLDB_REGNUM_GENERIC_FLAGS,  LLDB_INVALID_REGNUM),
    DEFINE_GPR(cs,     nullptr,  ehframe_regs::cs,      dwarf_regs::cs,      LLDB_INVALID_REGNUM,        LLDB_INVALID_REGNUM),
    DEFINE_GPR(fs,     nullptr,  ehframe_regs::fs,      dwarf_regs::fs,      LLDB_INVALID_REGNUM,        LLDB_INVALID_REGNUM),
    DEFINE_GPR(gs,     nullptr,  ehframe_regs::gs,      dwarf_regs::gs,      LLDB_INVALID_REGNUM,        LLDB_INVALID_REGNUM),
    DEFINE_GPR(ss,     nullptr,  ehframe_regs::ss,      dwarf_regs::ss,      LLDB_INVALID_REGNUM,        LLDB_INVALID_REGNUM),
    DEFINE_GPR(ds,     nullptr,  ehframe_regs::ds,      dwarf_regs::ds,      LLDB_INVALID_REGNUM,        LLDB_INVALID_REGNUM),
    DEFINE_GPR(es,     nullptr,  ehframe_regs::es,      dwarf_regs::es,      LLDB_INVALID_REGNUM,        LLDB_INVALID_REGNUM),
    DEFINE_XMM(xmm0),
    DEFINE_XMM(xmm1),
    DEFINE_XMM(xmm2),
    DEFINE_XMM(xmm3),
    DEFINE_XMM(xmm4),
    DEFINE_XMM(xmm5),
    DEFINE_XMM(xmm6),
    DEFINE_XMM(xmm7),
    DEFINE_XMM(xmm8),
    DEFINE_XMM(xmm9),
    DEFINE_XMM(xmm10),
    DEFINE_XMM(xmm11),
    DEFINE_XMM(xmm12),
    DEFINE_XMM(xmm13),
    DEFINE_XMM(xmm14),
    DEFINE_XMM(xmm15)
};
// clang-format on
} // namespace

RegisterInfoWindows_x86_64::RegisterInfoWindows_x86_64(
    const ArchSpec &target_arch)
    : lldb_private::RegisterInfoInterface(target_arch) {
  assert(target_arch.GetMachine() == llvm::Triple::x86_64);
}

const RegisterInfo *RegisterInfoWindows_x86_64::GetRegisterInfo() const {
  return g_register_infos_x86_64;
}

uint32_t RegisterInfoWindows_x86_64::GetRegisterCount() const {
  return std::size(g_register_infos_x86_64);
}

uint32_t RegisterInfoWindows_x86_64::GetUserRegisterCount() const {
  return std::size(g_register_infos_x86_64);
}

size_t RegisterInfoWindows_x86_64::GetGPRSize() const { return sizeof(GPR); }

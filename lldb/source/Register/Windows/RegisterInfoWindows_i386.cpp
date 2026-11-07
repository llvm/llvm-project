//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "RegisterInfoWindows_i386.h"
#include "lldb/Register/Arch/lldb-x86-dwarf-register-enums.h"
#include "lldb/Register/Arch/lldb-x86-ehframe-register-enums.h"
#include "lldb/Register/Arch/lldb-x86-register-enums.h"
#include "lldb/lldb-defines.h"


namespace dwarf_regs = lldb_private::registers::x86::dwarf;
namespace ehframe_regs = lldb_private::registers::x86::ehframe;
namespace reg_ids = lldb_private::registers::x86::id;
using namespace lldb_private;
using namespace lldb;

namespace {
// Declare our g_register_infos structure.
typedef struct _GPR {
  uint32_t eax;
  uint32_t ebx;
  uint32_t ecx;
  uint32_t edx;
  uint32_t edi;
  uint32_t esi;
  uint32_t ebp;
  uint32_t esp;
  uint32_t eip;
  uint32_t eflags;
  uint32_t cs;
  uint32_t fs;
  uint32_t gs;
  uint32_t ss;
  uint32_t ds;
  uint32_t es;
} GPR;

#define GPR_OFFSET(regname) (LLVM_EXTENSION offsetof(GPR, regname))

#define DEFINE_GPR(reg, alt, kind1, kind2, kind3, kind4)                       \
  {                                                                            \
#reg, alt, sizeof(((GPR *)nullptr)->reg), GPR_OFFSET(reg), eEncodingUint,  \
        eFormatHex,                                                            \
        {kind1, kind2, kind3, kind4, reg_ids::reg }, nullptr, nullptr,    \
        nullptr,                                                               \
  }

// clang-format off
static RegisterInfo g_register_infos_i386[] = {
// General purpose registers     EH_Frame              DWARF                 Generic                     Process Plugin
//  ===========================  ==================    ================      =========================   ====================
    DEFINE_GPR(eax,   nullptr,   ehframe_regs::eax,     dwarf_regs::eax,       LLDB_INVALID_REGNUM,        LLDB_INVALID_REGNUM),
    DEFINE_GPR(ebx,   nullptr,   ehframe_regs::ebx,     dwarf_regs::ebx,       LLDB_INVALID_REGNUM,        LLDB_INVALID_REGNUM),
    DEFINE_GPR(ecx,   nullptr,   ehframe_regs::ecx,     dwarf_regs::ecx,       LLDB_INVALID_REGNUM,        LLDB_INVALID_REGNUM),
    DEFINE_GPR(edx,   nullptr,   ehframe_regs::edx,     dwarf_regs::edx,       LLDB_INVALID_REGNUM,        LLDB_INVALID_REGNUM),
    DEFINE_GPR(edi,   nullptr,   ehframe_regs::edi,     dwarf_regs::edi,       LLDB_INVALID_REGNUM,        LLDB_INVALID_REGNUM),
    DEFINE_GPR(esi,   nullptr,   ehframe_regs::esi,     dwarf_regs::esi,       LLDB_INVALID_REGNUM,        LLDB_INVALID_REGNUM),
    DEFINE_GPR(ebp,   "fp",      ehframe_regs::ebp,     dwarf_regs::ebp,       LLDB_REGNUM_GENERIC_FP,     LLDB_INVALID_REGNUM),
    DEFINE_GPR(esp,   "sp",      ehframe_regs::esp,     dwarf_regs::esp,       LLDB_REGNUM_GENERIC_SP,     LLDB_INVALID_REGNUM),
    DEFINE_GPR(eip,   "pc",      ehframe_regs::eip,     dwarf_regs::eip,       LLDB_REGNUM_GENERIC_PC,     LLDB_INVALID_REGNUM),
    DEFINE_GPR(eflags, "flags",  ehframe_regs::eflags,  dwarf_regs::eflags,    LLDB_REGNUM_GENERIC_FLAGS,  LLDB_INVALID_REGNUM),
    DEFINE_GPR(cs,     nullptr,  LLDB_INVALID_REGNUM,  dwarf_regs::cs,        LLDB_INVALID_REGNUM,        LLDB_INVALID_REGNUM),
    DEFINE_GPR(fs,     nullptr,  LLDB_INVALID_REGNUM,  dwarf_regs::fs,        LLDB_INVALID_REGNUM,        LLDB_INVALID_REGNUM),
    DEFINE_GPR(gs,     nullptr,  LLDB_INVALID_REGNUM,  dwarf_regs::gs,        LLDB_INVALID_REGNUM,        LLDB_INVALID_REGNUM),
    DEFINE_GPR(ss,     nullptr,  LLDB_INVALID_REGNUM,  dwarf_regs::ss,        LLDB_INVALID_REGNUM,        LLDB_INVALID_REGNUM),
    DEFINE_GPR(ds,     nullptr,  LLDB_INVALID_REGNUM,  dwarf_regs::ds,        LLDB_INVALID_REGNUM,        LLDB_INVALID_REGNUM),
    DEFINE_GPR(es,     nullptr,  LLDB_INVALID_REGNUM,  dwarf_regs::es,        LLDB_INVALID_REGNUM,        LLDB_INVALID_REGNUM),
};
// clang-format on
} // namespace

RegisterInfoWindows_i386::RegisterInfoWindows_i386(
    const ArchSpec &target_arch)
    : lldb_private::RegisterInfoInterface(target_arch) {
  assert(target_arch.GetMachine() == llvm::Triple::x86);
}

const RegisterInfo *RegisterInfoWindows_i386::GetRegisterInfo() const {
  return g_register_infos_i386;
}

uint32_t RegisterInfoWindows_i386::GetRegisterCount() const {
  return std::size(g_register_infos_i386);
}

uint32_t RegisterInfoWindows_i386::GetUserRegisterCount() const {
  return std::size(g_register_infos_i386);
}

size_t RegisterInfoWindows_i386::GetGPRSize() const { return sizeof(GPR); }

//===-- RegisterContextWindows_x86.cpp --------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "lldb/Core/Error.h"
#include "lldb/Core/RegisterValue.h"
#include "lldb/Host/windows/HostThreadWindows.h"
#include "lldb/Host/windows/windows.h"
#include "lldb/lldb-private-types.h"

#include "ProcessWindowsLog.h"
#include "RegisterContextWindows_x86.h"
#include "RegisterContext_x86.h"
#include "lldb-x86-register-enums.h"

#include "llvm/ADT/STLExtras.h"

using namespace lldb;
using namespace lldb_private;

#define DEFINE_GPR(reg, alt) #reg, alt, 4, 0, eEncodingUint, eFormatHexUppercase
#define DEFINE_GPR_BIN(reg, alt) #reg, alt, 4, 0, eEncodingUint, eFormatBinary

namespace {

// This enum defines the layout of the global RegisterInfo array.  This is
// necessary because
// lldb register sets are defined in terms of indices into the register array.
// As such, the
// order of RegisterInfos defined in global registers array must match the order
// defined here.
// When defining the register set layouts, these values can appear in an
// arbitrary order, and that
// determines the order that register values are displayed in a dump.
enum RegisterIndex {
  eRegisterIndexEax,
  eRegisterIndexEbx,
  eRegisterIndexEcx,
  eRegisterIndexEdx,
  eRegisterIndexEdi,
  eRegisterIndexEsi,
  eRegisterIndexEbp,
  eRegisterIndexEsp,
  eRegisterIndexEip,
  eRegisterIndexEflags
};

// Array of all register information supported by Windows x86
RegisterInfo g_register_infos[] = {
    //  Macro auto defines most stuff   eh_frame                DWARF
    //  GENERIC                    GDB                   LLDB
    //  VALUE REGS    INVALIDATE REGS
    //  ==============================  =======================
    //  ===================  =========================  ===================
    //  =================  ==========    ===============
    {DEFINE_GPR(eax, nullptr),
     {ehframe_eax_i386, dwarf_eax_i386, LLDB_INVALID_REGNUM,
      LLDB_INVALID_REGNUM, lldb_eax_i386},
     nullptr,
     nullptr},
    {DEFINE_GPR(ebx, nullptr),
     {ehframe_ebx_i386, dwarf_ebx_i386, LLDB_INVALID_REGNUM,
      LLDB_INVALID_REGNUM, lldb_ebx_i386},
     nullptr,
     nullptr},
    {DEFINE_GPR(ecx, nullptr),
     {ehframe_ecx_i386, dwarf_ecx_i386, LLDB_INVALID_REGNUM,
      LLDB_INVALID_REGNUM, lldb_ecx_i386},
     nullptr,
     nullptr},
    {DEFINE_GPR(edx, nullptr),
     {ehframe_edx_i386, dwarf_edx_i386, LLDB_INVALID_REGNUM,
      LLDB_INVALID_REGNUM, lldb_edx_i386},
     nullptr,
     nullptr},
    {DEFINE_GPR(edi, nullptr),
     {ehframe_edi_i386, dwarf_edi_i386, LLDB_INVALID_REGNUM,
      LLDB_INVALID_REGNUM, lldb_edi_i386},
     nullptr,
     nullptr},
    {DEFINE_GPR(esi, nullptr),
     {ehframe_esi_i386, dwarf_esi_i386, LLDB_INVALID_REGNUM,
      LLDB_INVALID_REGNUM, lldb_esi_i386},
     nullptr,
     nullptr},
    {DEFINE_GPR(ebp, "fp"),
     {ehframe_ebp_i386, dwarf_ebp_i386, LLDB_REGNUM_GENERIC_FP,
      LLDB_INVALID_REGNUM, lldb_ebp_i386},
     nullptr,
     nullptr},
    {DEFINE_GPR(esp, "sp"),
     {ehframe_esp_i386, dwarf_esp_i386, LLDB_REGNUM_GENERIC_SP,
      LLDB_INVALID_REGNUM, lldb_esp_i386},
     nullptr,
     nullptr},
    {DEFINE_GPR(eip, "pc"),
     {ehframe_eip_i386, dwarf_eip_i386, LLDB_REGNUM_GENERIC_PC,
      LLDB_INVALID_REGNUM, lldb_eip_i386},
     nullptr,
     nullptr},
    {DEFINE_GPR_BIN(eflags, "flags"),
     {ehframe_eflags_i386, dwarf_eflags_i386, LLDB_REGNUM_GENERIC_FLAGS,
      LLDB_INVALID_REGNUM, lldb_eflags_i386},
     nullptr,
     nullptr},
};

// Array of lldb register numbers used to define the set of all General Purpose
// Registers
uint32_t g_gpr_reg_indices[] = {eRegisterIndexEax, eRegisterIndexEbx,
                                eRegisterIndexEcx, eRegisterIndexEdx,
                                eRegisterIndexEdi, eRegisterIndexEsi,
                                eRegisterIndexEbp, eRegisterIndexEsp,
                                eRegisterIndexEip, eRegisterIndexEflags};

RegisterSet g_register_sets[] = {
    {"General Purpose Registers", "gpr",
     llvm::array_lengthof(g_gpr_reg_indices), g_gpr_reg_indices},
};
}

//------------------------------------------------------------------
// Constructors and Destructors
//------------------------------------------------------------------
RegisterContextWindows_x86::RegisterContextWindows_x86(
    Thread &thread, uint32_t concrete_frame_idx)
    : RegisterContextWindows(thread, concrete_frame_idx) {}

RegisterContextWindows_x86::~RegisterContextWindows_x86() {}

size_t RegisterContextWindows_x86::GetRegisterCount() {
  return llvm::array_lengthof(g_register_infos);
}

const RegisterInfo *
RegisterContextWindows_x86::GetRegisterInfoAtIndex(size_t reg) {
  return &g_register_infos[reg];
}

size_t RegisterContextWindows_x86::GetRegisterSetCount() {
  return llvm::array_lengthof(g_register_sets);
}

const RegisterSet *RegisterContextWindows_x86::GetRegisterSet(size_t reg_set) {
  return &g_register_sets[reg_set];
}

bool RegisterContextWindows_x86::ReadRegister(const RegisterInfo *reg_info,
                                              RegisterValue &reg_value) {
  if (!CacheAllRegisterValues())
    return false;

  uint32_t reg = reg_info->kinds[eRegisterKindLLDB];
  switch (reg) {
  case lldb_eax_i386:
    return ReadRegisterHelper(CONTEXT_INTEGER, "EAX", m_context.Eax, reg_value);
  case lldb_ebx_i386:
    return ReadRegisterHelper(CONTEXT_INTEGER, "EBX", m_context.Ebx, reg_value);
  case lldb_ecx_i386:
    return ReadRegisterHelper(CONTEXT_INTEGER, "ECX", m_context.Ecx, reg_value);
  case lldb_edx_i386:
    return ReadRegisterHelper(CONTEXT_INTEGER, "EDX", m_context.Edx, reg_value);
  case lldb_edi_i386:
    return ReadRegisterHelper(CONTEXT_INTEGER, "EDI", m_context.Edi, reg_value);
  case lldb_esi_i386:
    return ReadRegisterHelper(CONTEXT_INTEGER, "ESI", m_context.Esi, reg_value);
  case lldb_ebp_i386:
    return ReadRegisterHelper(CONTEXT_CONTROL, "EBP", m_context.Ebp, reg_value);
  case lldb_esp_i386:
    return ReadRegisterHelper(CONTEXT_CONTROL, "ESP", m_context.Esp, reg_value);
  case lldb_eip_i386:
    return ReadRegisterHelper(CONTEXT_CONTROL, "EIP", m_context.Eip, reg_value);
  case lldb_eflags_i386:
    return ReadRegisterHelper(CONTEXT_CONTROL, "EFLAGS", m_context.EFlags,
                              reg_value);
  default:
    WINWARN_IFALL(WINDOWS_LOG_REGISTERS, "Requested unknown register %u", reg);
    break;
  }
  return false;
}

bool RegisterContextWindows_x86::ReadRegisterHelper(
    DWORD flags_required, const char *reg_name, DWORD value,
    RegisterValue &reg_value) const {
  if ((m_context.ContextFlags & flags_required) != flags_required) {
    WINLOG_IFALL(WINDOWS_LOG_REGISTERS, "Thread context doesn't have %s",
                 reg_name);
    return false;
  }
  WINLOG_IFALL(WINDOWS_LOG_REGISTERS, "Read value 0x%x from %s", value,
               reg_name);
  reg_value.SetUInt32(value);
  return true;
}

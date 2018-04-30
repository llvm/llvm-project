//===-- RegisterContextWindows_x64.cpp --------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "lldb/Core/RegisterValue.h"
#include "lldb/Host/windows/HostThreadWindows.h"
#include "lldb/Host/windows/windows.h"
#include "lldb/Utility/Status.h"
#include "lldb/lldb-private-types.h"

#include "RegisterContextWindows_x64.h"
#include "RegisterContext_x86.h"
#include "TargetThreadWindows.h"
#include "lldb-x86-register-enums.h"

#include "llvm/ADT/STLExtras.h"

using namespace lldb;
using namespace lldb_private;

#define DEFINE_GPR(reg, alt) #reg, alt, 8, 0, eEncodingUint, eFormatHexUppercase
#define DEFINE_GPR_BIN(reg, alt) #reg, alt, 8, 0, eEncodingUint, eFormatBinary

namespace {

// This enum defines the layout of the global RegisterInfo array.  This is
// necessary because lldb register sets are defined in terms of indices into
// the register array. As such, the order of RegisterInfos defined in global
// registers array must match the order defined here. When defining the
// register set layouts, these values can appear in an arbitrary order, and
// that determines the order that register values are displayed in a dump.
enum RegisterIndex {
  eRegisterIndexRax,
  eRegisterIndexRbx,
  eRegisterIndexRcx,
  eRegisterIndexRdx,
  eRegisterIndexRdi,
  eRegisterIndexRsi,
  eRegisterIndexR8,
  eRegisterIndexR9,
  eRegisterIndexR10,
  eRegisterIndexR11,
  eRegisterIndexR12,
  eRegisterIndexR13,
  eRegisterIndexR14,
  eRegisterIndexR15,
  eRegisterIndexRbp,
  eRegisterIndexRsp,
  eRegisterIndexRip,
  eRegisterIndexRflags
};

// Array of all register information supported by Windows x86
RegisterInfo g_register_infos[] = {
    //  Macro auto defines most stuff     eh_frame                  DWARF
    //  GENERIC
    //  GDB                  LLDB                  VALUE REGS    INVALIDATE REGS
    //  ================================  =========================
    //  ======================  =========================
    //  ===================  =================     ==========    ===============
    {DEFINE_GPR(rax, nullptr),
     {dwarf_rax_x86_64, dwarf_rax_x86_64, LLDB_INVALID_REGNUM,
      LLDB_INVALID_REGNUM, lldb_rax_x86_64},
     nullptr,
     nullptr},
    {DEFINE_GPR(rbx, nullptr),
     {dwarf_rbx_x86_64, dwarf_rbx_x86_64, LLDB_INVALID_REGNUM,
      LLDB_INVALID_REGNUM, lldb_rbx_x86_64},
     nullptr,
     nullptr},
    {DEFINE_GPR(rcx, nullptr),
     {dwarf_rcx_x86_64, dwarf_rcx_x86_64, LLDB_INVALID_REGNUM,
      LLDB_INVALID_REGNUM, lldb_rcx_x86_64},
     nullptr,
     nullptr},
    {DEFINE_GPR(rdx, nullptr),
     {dwarf_rdx_x86_64, dwarf_rdx_x86_64, LLDB_INVALID_REGNUM,
      LLDB_INVALID_REGNUM, lldb_rdx_x86_64},
     nullptr,
     nullptr},
    {DEFINE_GPR(rdi, nullptr),
     {dwarf_rdi_x86_64, dwarf_rdi_x86_64, LLDB_INVALID_REGNUM,
      LLDB_INVALID_REGNUM, lldb_rdi_x86_64},
     nullptr,
     nullptr},
    {DEFINE_GPR(rsi, nullptr),
     {dwarf_rsi_x86_64, dwarf_rsi_x86_64, LLDB_INVALID_REGNUM,
      LLDB_INVALID_REGNUM, lldb_rsi_x86_64},
     nullptr,
     nullptr},
    {DEFINE_GPR(r8, nullptr),
     {dwarf_r8_x86_64, dwarf_r8_x86_64, LLDB_INVALID_REGNUM,
      LLDB_INVALID_REGNUM, lldb_r8_x86_64},
     nullptr,
     nullptr},
    {DEFINE_GPR(r9, nullptr),
     {dwarf_r9_x86_64, dwarf_r9_x86_64, LLDB_INVALID_REGNUM,
      LLDB_INVALID_REGNUM, lldb_r9_x86_64},
     nullptr,
     nullptr},
    {DEFINE_GPR(r10, nullptr),
     {dwarf_r10_x86_64, dwarf_r10_x86_64, LLDB_INVALID_REGNUM,
      LLDB_INVALID_REGNUM, lldb_r10_x86_64},
     nullptr,
     nullptr},
    {DEFINE_GPR(r11, nullptr),
     {dwarf_r11_x86_64, dwarf_r11_x86_64, LLDB_INVALID_REGNUM,
      LLDB_INVALID_REGNUM, lldb_r11_x86_64},
     nullptr,
     nullptr},
    {DEFINE_GPR(r12, nullptr),
     {dwarf_r12_x86_64, dwarf_r12_x86_64, LLDB_INVALID_REGNUM,
      LLDB_INVALID_REGNUM, lldb_r12_x86_64},
     nullptr,
     nullptr},
    {DEFINE_GPR(r13, nullptr),
     {dwarf_r13_x86_64, dwarf_r13_x86_64, LLDB_INVALID_REGNUM,
      LLDB_INVALID_REGNUM, lldb_r13_x86_64},
     nullptr,
     nullptr},
    {DEFINE_GPR(r14, nullptr),
     {dwarf_r14_x86_64, dwarf_r14_x86_64, LLDB_INVALID_REGNUM,
      LLDB_INVALID_REGNUM, lldb_r14_x86_64},
     nullptr,
     nullptr},
    {DEFINE_GPR(r15, nullptr),
     {dwarf_r15_x86_64, dwarf_r15_x86_64, LLDB_INVALID_REGNUM,
      LLDB_INVALID_REGNUM, lldb_r15_x86_64},
     nullptr,
     nullptr},
    {DEFINE_GPR(rbp, "fp"),
     {dwarf_rbp_x86_64, dwarf_rbp_x86_64, LLDB_REGNUM_GENERIC_FP,
      LLDB_INVALID_REGNUM, lldb_rbp_x86_64},
     nullptr,
     nullptr},
    {DEFINE_GPR(rsp, "sp"),
     {dwarf_rsp_x86_64, dwarf_rsp_x86_64, LLDB_REGNUM_GENERIC_SP,
      LLDB_INVALID_REGNUM, lldb_rsp_x86_64},
     nullptr,
     nullptr},
    {DEFINE_GPR(rip, "pc"),
     {dwarf_rip_x86_64, dwarf_rip_x86_64, LLDB_REGNUM_GENERIC_PC,
      LLDB_INVALID_REGNUM, lldb_rip_x86_64},
     nullptr,
     nullptr},
    {DEFINE_GPR_BIN(eflags, "flags"),
     {LLDB_INVALID_REGNUM, LLDB_INVALID_REGNUM, LLDB_REGNUM_GENERIC_FLAGS,
      LLDB_INVALID_REGNUM, lldb_rflags_x86_64},
     nullptr,
     nullptr},
};

static size_t k_num_register_infos = llvm::array_lengthof(g_register_infos);

// Array of lldb register numbers used to define the set of all General Purpose
// Registers
uint32_t g_gpr_reg_indices[] = {
    eRegisterIndexRax, eRegisterIndexRbx, eRegisterIndexRcx,
    eRegisterIndexRdx, eRegisterIndexRdi, eRegisterIndexRsi,
    eRegisterIndexR8,  eRegisterIndexR9,  eRegisterIndexR10,
    eRegisterIndexR11, eRegisterIndexR12, eRegisterIndexR13,
    eRegisterIndexR14, eRegisterIndexR15, eRegisterIndexRbp,
    eRegisterIndexRsp, eRegisterIndexRip, eRegisterIndexRflags};

RegisterSet g_register_sets[] = {
    {"General Purpose Registers", "gpr",
     llvm::array_lengthof(g_gpr_reg_indices), g_gpr_reg_indices},
};
}

//------------------------------------------------------------------
// Constructors and Destructors
//------------------------------------------------------------------
RegisterContextWindows_x64::RegisterContextWindows_x64(
    Thread &thread, uint32_t concrete_frame_idx)
    : RegisterContextWindows(thread, concrete_frame_idx) {}

RegisterContextWindows_x64::~RegisterContextWindows_x64() {}

size_t RegisterContextWindows_x64::GetRegisterCount() {
  return llvm::array_lengthof(g_register_infos);
}

const RegisterInfo *
RegisterContextWindows_x64::GetRegisterInfoAtIndex(size_t reg) {
  if (reg < k_num_register_infos)
    return &g_register_infos[reg];
  return NULL;
}

size_t RegisterContextWindows_x64::GetRegisterSetCount() {
  return llvm::array_lengthof(g_register_sets);
}

const RegisterSet *RegisterContextWindows_x64::GetRegisterSet(size_t reg_set) {
  return &g_register_sets[reg_set];
}

bool RegisterContextWindows_x64::ReadRegister(const RegisterInfo *reg_info,
                                              RegisterValue &reg_value) {
  if (!CacheAllRegisterValues())
    return false;

  switch (reg_info->kinds[eRegisterKindLLDB]) {
  case lldb_rax_x86_64:
    reg_value.SetUInt64(m_context.Rax);
    break;
  case lldb_rbx_x86_64:
    reg_value.SetUInt64(m_context.Rbx);
    break;
  case lldb_rcx_x86_64:
    reg_value.SetUInt64(m_context.Rcx);
    break;
  case lldb_rdx_x86_64:
    reg_value.SetUInt64(m_context.Rdx);
    break;
  case lldb_rdi_x86_64:
    reg_value.SetUInt64(m_context.Rdi);
    break;
  case lldb_rsi_x86_64:
    reg_value.SetUInt64(m_context.Rsi);
    break;
  case lldb_r8_x86_64:
    reg_value.SetUInt64(m_context.R8);
    break;
  case lldb_r9_x86_64:
    reg_value.SetUInt64(m_context.R9);
    break;
  case lldb_r10_x86_64:
    reg_value.SetUInt64(m_context.R10);
    break;
  case lldb_r11_x86_64:
    reg_value.SetUInt64(m_context.R11);
    break;
  case lldb_r12_x86_64:
    reg_value.SetUInt64(m_context.R12);
    break;
  case lldb_r13_x86_64:
    reg_value.SetUInt64(m_context.R13);
    break;
  case lldb_r14_x86_64:
    reg_value.SetUInt64(m_context.R14);
    break;
  case lldb_r15_x86_64:
    reg_value.SetUInt64(m_context.R15);
    break;
  case lldb_rbp_x86_64:
    reg_value.SetUInt64(m_context.Rbp);
    break;
  case lldb_rsp_x86_64:
    reg_value.SetUInt64(m_context.Rsp);
    break;
  case lldb_rip_x86_64:
    reg_value.SetUInt64(m_context.Rip);
    break;
  case lldb_rflags_x86_64:
    reg_value.SetUInt64(m_context.EFlags);
    break;
  }
  return true;
}

bool RegisterContextWindows_x64::WriteRegister(const RegisterInfo *reg_info,
                                               const RegisterValue &reg_value) {
  // Since we cannot only write a single register value to the inferior, we
  // need to make sure our cached copy of the register values are fresh.
  // Otherwise when writing EAX, for example, we may also overwrite some other
  // register with a stale value.
  if (!CacheAllRegisterValues())
    return false;

  switch (reg_info->kinds[eRegisterKindLLDB]) {
  case lldb_rax_x86_64:
    m_context.Rax = reg_value.GetAsUInt64();
    break;
  case lldb_rbx_x86_64:
    m_context.Rbx = reg_value.GetAsUInt64();
    break;
  case lldb_rcx_x86_64:
    m_context.Rcx = reg_value.GetAsUInt64();
    break;
  case lldb_rdx_x86_64:
    m_context.Rdx = reg_value.GetAsUInt64();
    break;
  case lldb_rdi_x86_64:
    m_context.Rdi = reg_value.GetAsUInt64();
    break;
  case lldb_rsi_x86_64:
    m_context.Rsi = reg_value.GetAsUInt64();
    break;
  case lldb_r8_x86_64:
    m_context.R8 = reg_value.GetAsUInt64();
    break;
  case lldb_r9_x86_64:
    m_context.R9 = reg_value.GetAsUInt64();
    break;
  case lldb_r10_x86_64:
    m_context.R10 = reg_value.GetAsUInt64();
    break;
  case lldb_r11_x86_64:
    m_context.R11 = reg_value.GetAsUInt64();
    break;
  case lldb_r12_x86_64:
    m_context.R12 = reg_value.GetAsUInt64();
    break;
  case lldb_r13_x86_64:
    m_context.R13 = reg_value.GetAsUInt64();
    break;
  case lldb_r14_x86_64:
    m_context.R14 = reg_value.GetAsUInt64();
    break;
  case lldb_r15_x86_64:
    m_context.R15 = reg_value.GetAsUInt64();
    break;
  case lldb_rbp_x86_64:
    m_context.Rbp = reg_value.GetAsUInt64();
    break;
  case lldb_rsp_x86_64:
    m_context.Rsp = reg_value.GetAsUInt64();
    break;
  case lldb_rip_x86_64:
    m_context.Rip = reg_value.GetAsUInt64();
    break;
  case lldb_rflags_x86_64:
    m_context.EFlags = reg_value.GetAsUInt64();
    break;
  }

  // Physically update the registers in the target process.
  TargetThreadWindows &wthread = static_cast<TargetThreadWindows &>(m_thread);
  return ::SetThreadContext(
      wthread.GetHostThread().GetNativeThread().GetSystemHandle(), &m_context);
}

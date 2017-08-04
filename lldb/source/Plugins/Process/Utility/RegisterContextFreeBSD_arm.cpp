//===-- RegisterContextFreeBSD_arm.cpp -------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===---------------------------------------------------------------------===//

#include <cassert>
#include <stddef.h>
#include <vector>

#include "lldb/lldb-defines.h"
#include "llvm/Support/Compiler.h"

#include "RegisterContextFreeBSD_arm.h"

using namespace lldb;
using namespace lldb_private;

// Based on RegisterContextLinux_arm.cpp and
// http://svnweb.freebsd.org/base/head/sys/arm/include/reg.h
#define GPR_OFFSET(idx) ((idx)*4)
#define FPU_OFFSET(idx) ((idx)*4 + sizeof(RegisterContextFreeBSD_arm::GPR))
#define FPSCR_OFFSET                                                           \
  (LLVM_EXTENSION offsetof(RegisterContextFreeBSD_arm::FPU, fpscr) +           \
   sizeof(RegisterContextFreeBSD_arm::GPR))
#define EXC_OFFSET(idx)                                                        \
  ((idx)*4 + sizeof(RegisterContextFreeBSD_arm::GPR) +                         \
   sizeof(RegisterContextFreeBSD_arm::FPU))
#define DBG_OFFSET(reg)                                                        \
  ((LLVM_EXTENSION offsetof(RegisterContextFreeBSD_arm::DBG, reg) +            \
    sizeof(RegisterContextFreeBSD_arm::GPR) +                                  \
    sizeof(RegisterContextFreeBSD_arm::FPU) +                                  \
    sizeof(RegisterContextFreeBSD_arm::EXC)))

#define DEFINE_DBG(reg, i)                                                     \
  #reg, NULL, sizeof(((RegisterContextFreeBSD_arm::DBG *) NULL)->reg[i]),      \
                      DBG_OFFSET(reg[i]), eEncodingUint, eFormatHex,           \
                                 {LLDB_INVALID_REGNUM, LLDB_INVALID_REGNUM,    \
                                  LLDB_INVALID_REGNUM, LLDB_INVALID_REGNUM,    \
                                  dbg_##reg##i },                              \
                                  NULL, NULL, NULL, 0
#define REG_CONTEXT_SIZE                                                       \
  (sizeof(RegisterContextFreeBSD_arm::GPR) +                                   \
   sizeof(RegisterContextFreeBSD_arm::FPU) +                                   \
   sizeof(RegisterContextFreeBSD_arm::EXC))

//-----------------------------------------------------------------------------
// Include RegisterInfos_arm to declare our g_register_infos_arm structure.
//-----------------------------------------------------------------------------
#define DECLARE_REGISTER_INFOS_ARM_STRUCT
#include "RegisterInfos_arm.h"
#undef DECLARE_REGISTER_INFOS_ARM_STRUCT

static const lldb_private::RegisterInfo *
GetRegisterInfoPtr(const lldb_private::ArchSpec &target_arch) {
  switch (target_arch.GetMachine()) {
  case llvm::Triple::arm:
    return g_register_infos_arm;
  default:
    assert(false && "Unhandled target architecture.");
    return NULL;
  }
}

static uint32_t
GetRegisterInfoCount(const lldb_private::ArchSpec &target_arch) {
  switch (target_arch.GetMachine()) {
  case llvm::Triple::arm:
    return static_cast<uint32_t>(sizeof(g_register_infos_arm) /
                                 sizeof(g_register_infos_arm[0]));
  default:
    assert(false && "Unhandled target architecture.");
    return 0;
  }
}

RegisterContextFreeBSD_arm::RegisterContextFreeBSD_arm(
    const lldb_private::ArchSpec &target_arch)
    : lldb_private::RegisterInfoInterface(target_arch),
      m_register_info_p(GetRegisterInfoPtr(target_arch)),
      m_register_info_count(GetRegisterInfoCount(target_arch)) {}

size_t RegisterContextFreeBSD_arm::GetGPRSize() const {
  return sizeof(struct RegisterContextFreeBSD_arm::GPR);
}

const lldb_private::RegisterInfo *
RegisterContextFreeBSD_arm::GetRegisterInfo() const {
  return m_register_info_p;
}

uint32_t RegisterContextFreeBSD_arm::GetRegisterCount() const {
  return m_register_info_count;
}

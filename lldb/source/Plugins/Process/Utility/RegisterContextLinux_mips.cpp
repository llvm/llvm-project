//===-- RegisterContextLinux_mips.cpp ------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===---------------------------------------------------------------------===//

#include <stddef.h>
#include <vector>

// For eh_frame and DWARF Register numbers
#include "RegisterContextLinux_mips.h"

// Internal codes for mips registers
#include "lldb-mips-linux-register-enums.h"

// For GP and FP buffers
#include "RegisterContext_mips.h"

using namespace lldb_private;
using namespace lldb;

//---------------------------------------------------------------------------
// Include RegisterInfos_mips to declare our g_register_infos_mips structure.
//---------------------------------------------------------------------------
#define DECLARE_REGISTER_INFOS_MIPS_STRUCT
#include "RegisterInfos_mips.h"
#undef DECLARE_REGISTER_INFOS_MIPS_STRUCT

uint32_t GetUserRegisterInfoCount(bool msa_present) {
  if (msa_present)
    return static_cast<uint32_t>(k_num_user_registers_mips);
  return static_cast<uint32_t>(k_num_user_registers_mips -
                               k_num_msa_registers_mips);
}

RegisterContextLinux_mips::RegisterContextLinux_mips(
    const ArchSpec &target_arch, bool msa_present)
    : RegisterInfoInterface(target_arch),
      m_user_register_count(GetUserRegisterInfoCount(msa_present)) {}

size_t RegisterContextLinux_mips::GetGPRSize() const {
  return sizeof(GPR_linux_mips);
}

const RegisterInfo *RegisterContextLinux_mips::GetRegisterInfo() const {
  switch (m_target_arch.GetMachine()) {
  case llvm::Triple::mips:
  case llvm::Triple::mipsel:
    return g_register_infos_mips;
  default:
    assert(false && "Unhandled target architecture.");
    return NULL;
  }
}

uint32_t RegisterContextLinux_mips::GetRegisterCount() const {
  return static_cast<uint32_t>(sizeof(g_register_infos_mips) /
                               sizeof(g_register_infos_mips[0]));
}

uint32_t RegisterContextLinux_mips::GetUserRegisterCount() const {
  return static_cast<uint32_t>(m_user_register_count);
}

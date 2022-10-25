//===-- RegisterInfoPOSIX_loongarch64.cpp --------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===---------------------------------------------------------------------===//

#include <cassert>
#include <lldb/Utility/Flags.h>
#include <stddef.h>

#include "lldb/lldb-defines.h"
#include "llvm/Support/Compiler.h"

#include "RegisterInfoPOSIX_loongarch64.h"

const lldb_private::RegisterInfo *
RegisterInfoPOSIX_loongarch64::GetRegisterInfoPtr(
    const lldb_private::ArchSpec &target_arch) {
  switch (target_arch.GetMachine()) {
  default:
    assert(false && "Unhandled target architecture.");
    return nullptr;
  }
}

uint32_t RegisterInfoPOSIX_loongarch64::GetRegisterInfoCount(
    const lldb_private::ArchSpec &target_arch) {
  switch (target_arch.GetMachine()) {
  default:
    assert(false && "Unhandled target architecture.");
    return 0;
  }
}

RegisterInfoPOSIX_loongarch64::RegisterInfoPOSIX_loongarch64(
    const lldb_private::ArchSpec &target_arch, lldb_private::Flags flags)
    : lldb_private::RegisterInfoAndSetInterface(target_arch),
      m_register_info_p(GetRegisterInfoPtr(target_arch)),
      m_register_info_count(GetRegisterInfoCount(target_arch)) {}

uint32_t RegisterInfoPOSIX_loongarch64::GetRegisterCount() const { return 0; }

size_t RegisterInfoPOSIX_loongarch64::GetGPRSize() const {
  return sizeof(struct RegisterInfoPOSIX_loongarch64::GPR);
}

size_t RegisterInfoPOSIX_loongarch64::GetFPRSize() const {
  return sizeof(struct RegisterInfoPOSIX_loongarch64::FPR);
}

const lldb_private::RegisterInfo *
RegisterInfoPOSIX_loongarch64::GetRegisterInfo() const {
  return m_register_info_p;
}

size_t RegisterInfoPOSIX_loongarch64::GetRegisterSetCount() const { return 0; }

size_t RegisterInfoPOSIX_loongarch64::GetRegisterSetFromRegisterIndex(
    uint32_t reg_index) const {
  return LLDB_INVALID_REGNUM;
}

const lldb_private::RegisterSet *
RegisterInfoPOSIX_loongarch64::GetRegisterSet(size_t set_index) const {
  return nullptr;
}

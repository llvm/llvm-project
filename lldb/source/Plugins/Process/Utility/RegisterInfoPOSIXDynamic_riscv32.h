//===-- RegisterInfoPOSIXDynamic_riscv32.h ----------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLDB_SOURCE_PLUGINS_PROCESS_UTILITY_REGISTERINFOPOSIXDYNAMIC_RISCV32_H
#define LLDB_SOURCE_PLUGINS_PROCESS_UTILITY_REGISTERINFOPOSIXDYNAMIC_RISCV32_H

#include "RegisterInfoAndSetInterface.h"
#include "lldb/Target/DynamicRegisterInfo.h"
#include "lldb/Target/RegisterContext.h"
#include "lldb/Utility/Flags.h"
#include "lldb/lldb-private.h"
#include <map>

class RegisterInfoPOSIXDynamic_riscv32
    : public lldb_private::RegisterInfoAndSetInterface {
public:
  RegisterInfoPOSIXDynamic_riscv32(const lldb_private::ArchSpec &target_arch);

  size_t GetGPRSize() const override;

  size_t GetFPRSize() const override;

  const lldb_private::RegisterInfo *GetRegisterInfo() const override;

  uint32_t GetRegisterCount() const override;

  const lldb_private::RegisterSet *
  GetRegisterSet(size_t reg_set) const override;

  size_t GetRegisterSetCount() const override;

  size_t GetRegisterSetFromRegisterIndex(uint32_t reg_index) const override;

  size_t SetRegisterInfo(
      std::vector<lldb_private::DynamicRegisterInfo::Register> regs);

  const lldb_private::RegisterInfo *
  GetRegisterInfo(llvm::StringRef reg_name) const;

private:
  lldb_private::DynamicRegisterInfo m_dyn_reg_infos;
  const lldb_private::ArchSpec m_target_arch;
};

#endif // LLDB_SOURCE_PLUGINS_PROCESS_UTILITY_REGISTERINFOPOSIXDYNAMIC_RISCV32_H

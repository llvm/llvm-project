//===----------------------------------------------------------------------===//
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
#include "lldb/lldb-private.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"

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

  /// Builds CS register information entries for 32-bit RISC-V debug targets on
  /// the basis of the enabled ISA extensions.
  ///
  /// Construct a baseline CSR container, \p cs_reg_infos , and apply extension
  /// patches in a deterministic order so that the final CSR metadata depends
  /// only on \p features and conflict resolution is predictable.
  static void BuildCSRegInfos(
      llvm::ArrayRef<std::string> features,
      llvm::SmallVectorImpl<lldb_private::RegisterInfo> &cs_reg_infos);

private:
  lldb_private::DynamicRegisterInfo m_dyn_reg_infos;
  const lldb_private::ArchSpec m_target_arch;

  /// Applies the CS register information patch set for a given feature.
  ///
  /// Override a baseline CSR metadata container, \p cs_reg_infos , with
  /// feature-specific definitions by looking up the patch list for \p feature
  /// and updating only the affected CSR entries in-place.
  static void ConfigureCSRegInfos(
      llvm::StringRef feature,
      llvm::SmallVectorImpl<lldb_private::RegisterInfo> &cs_reg_infos);
};

#endif // LLDB_SOURCE_PLUGINS_PROCESS_UTILITY_REGISTERINFOPOSIXDYNAMIC_RISCV32_H

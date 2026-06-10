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

  /// @brief Builds CS register information entries for 32-bit RISC-V debug
  ///        targets on the basis of the enabled ISA extensions.
  ///
  /// Custom and vendor RISC-V extensions can define CSRs that overlap
  /// in address space. This routine constructs a baseline CSR container and
  /// applies extension patches in a deterministic order so that the final CSR
  /// metadata depends only on the feature set and conflict resolution is
  /// predictable.
  ///
  /// @param[in] features ISA extension feature names.
  ///
  /// @return Vector of CS register information entries for the 32-bit RISC-V
  ///         debug target configuration.
  std::vector<lldb_private::RegisterInfo>
  GetCSRegisterInfos(const std::vector<std::string> &features);

private:
  lldb_private::DynamicRegisterInfo m_dyn_reg_infos;
  const lldb_private::ArchSpec m_target_arch;

  /// @brief Applies the CS register information patch set for a given feature.
  ///
  /// CSR metadata is constructed from a baseline container and then selectively
  /// overridden by feature-specific definitions. This helper performs the
  /// override by looking up the patch list for the feature and updating only
  /// the affected CSR entries in-place.
  ///
  /// @param[in,out] cs_reg_infos CS register information vector to update
  ///                             in-place.
  /// @param[in]     feature      Feature name used to select a patch set
  ///                             (e.g., "default").
  void
  ConfigureCSRegInfos(std::vector<lldb_private::RegisterInfo> &cs_reg_infos,
                      llvm::StringRef feature);
};

#endif // LLDB_SOURCE_PLUGINS_PROCESS_UTILITY_REGISTERINFOPOSIXDYNAMIC_RISCV32_H

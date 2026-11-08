//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLDB_SOURCE_PLUGINS_PROCESS_ELF_CORE_REGISTERCONTEXTELFCORE_ARM64_H
#define LLDB_SOURCE_PLUGINS_PROCESS_ELF_CORE_REGISTERCONTEXTELFCORE_ARM64_H

#include "lldb/Register/Common/RegisterInfo_arm64.h"
#include "lldb/Register/Layouts/Linux/RegisterLayoutLinux_arm64.h"
#include "lldb/Target/ThreadRegisterContext.h"

#include "Plugins/Process/elf-core/RegisterUtilities.h"
#include "lldb/Utility/DataBufferHeap.h"
#include "lldb/Utility/DataExtractor.h"
#include <optional>

class RegisterContextELFCore_arm64
    : public lldb_private::ThreadRegisterContext {
public:
  static llvm::Expected<std::unique_ptr<RegisterContextELFCore_arm64>>
  Create(lldb_private::Thread &thread, const lldb_private::ArchSpec &arch,
         const lldb_private::DataExtractor &gpregset,
         llvm::ArrayRef<lldb_private::CoreNote> notes);

  ~RegisterContextELFCore_arm64() override;

  llvm::Expected<lldb_private::RegisterValue>
  ReadRegister(lldb_private::RegisterID id) override;

  llvm::Error WriteRegister(lldb_private::RegisterID id,
                     const lldb_private::RegisterValue &value) override;

  bool ReadAllRegisterValues(lldb::WritableDataBufferSP &data_sp) override;

  bool WriteAllRegisterValues(const lldb::DataBufferSP &data_sp) override;

  bool HardwareSingleStep(bool enable) override;

protected:
  RegisterContextELFCore_arm64(
      lldb_private::Thread &thread, lldb_private::RegisterLayout layout,
      const lldb_private::DataExtractor &gpregset,
      llvm::ArrayRef<lldb_private::CoreNote> notes,
      std::optional<lldb_private::linux_arm64::RegisterConfig> config);

private:
  std::optional<lldb_private::linux_arm64::RegisterConfig> m_register_config;
  lldb_private::DataExtractor m_gpr_data;
  lldb_private::DataExtractor m_fpr_data;
  lldb_private::DataExtractor m_sve_data;
  lldb_private::DataExtractor m_pac_data;
  lldb_private::DataExtractor m_tls_data;
  lldb_private::DataExtractor m_za_data;
  lldb_private::DataExtractor m_mte_data;
  lldb_private::DataExtractor m_zt_data;
  lldb_private::DataExtractor m_fpmr_data;
  lldb_private::DataExtractor m_gcs_data;
  lldb_private::DataExtractor m_poe_data;

  lldb_private::SVEState m_sve_state = lldb_private::SVEState::Unknown;
  uint16_t m_sve_vector_length = 0;

  // These are pseudo registers derived from the values in SSVE and ZA data.
  lldb_private::linux_arm64::SMERegisters m_sme_pseudo_regs;

  void ConfigureRegisterContext();

  uint64_t GetSVERegVG() { return m_sve_vector_length / 8; }
};

#endif // LLDB_SOURCE_PLUGINS_PROCESS_ELF_CORE_REGISTERCONTEXTELFCORE_ARM64_H

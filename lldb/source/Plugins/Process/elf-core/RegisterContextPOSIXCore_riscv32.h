//===-- RegisterContextPOSIXCore_riscv32.h ----------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLDB_SOURCE_PLUGINS_PROCESS_ELF_CORE_REGISTERCONTEXTPOSIXCORE_RISCV32_H
#define LLDB_SOURCE_PLUGINS_PROCESS_ELF_CORE_REGISTERCONTEXTPOSIXCORE_RISCV32_H

// #include "Plugins/Process/Utility/RegisterContextPOSIX_riscv32.h"
#include "Plugins/Process/Utility/RegisterInfoPOSIXDynamic_riscv32.h"

#include "Plugins/Process/elf-core/RegisterUtilities.h"
#include "lldb/Target/Thread.h"
#include "lldb/Utility/DataExtractor.h"
#include "lldb/Utility/RegisterValue.h"

#include <memory>

class RegisterContextCorePOSIX_riscv32 : public lldb_private::RegisterContext {
public:
  struct csr_kv_t {
    uint32_t addr; // CSR address/index
    uint32_t val;  // CSR value
  };

  static std::unique_ptr<RegisterContextCorePOSIX_riscv32>
  Create(lldb_private::Thread &thread, const lldb_private::ArchSpec &arch,
         const lldb_private::DataExtractor &gpregset,
         llvm::ArrayRef<lldb_private::CoreNote> notes);

  ~RegisterContextCorePOSIX_riscv32() override;

  void InvalidateAllRegisters() override;

  size_t GetRegisterCount() override;

  const lldb_private::RegisterInfo *GetRegisterInfoAtIndex(size_t reg) override;

  size_t GetRegisterSetCount() override;

  const lldb_private::RegisterSet *GetRegisterSet(size_t set) override;

  bool ReadRegister(const lldb_private::RegisterInfo *reg_info,
                    lldb_private::RegisterValue &value) override;

  bool WriteRegister(const lldb_private::RegisterInfo *reg_info,
                     const lldb_private::RegisterValue &value) override;

protected:
  RegisterContextCorePOSIX_riscv32(
      lldb_private::Thread &thread,
      std::unique_ptr<RegisterInfoPOSIXDynamic_riscv32> register_info,
      const lldb_private::DataExtractor &gpregset,
      llvm::ArrayRef<lldb_private::CoreNote> notes);

  bool IsGPR(unsigned reg);

  bool IsFPR(unsigned reg);

  bool IsCSR(unsigned reg);

private:
  std::unique_ptr<RegisterInfoPOSIXDynamic_riscv32> m_reg_infos_up;
  lldb_private::DataExtractor m_gpregset;
  lldb_private::DataExtractor m_fpregset;
  lldb_private::DataExtractor m_csregset;
  std::vector<uint32_t> m_csregset_regnums;
};

#endif // LLDB_SOURCE_PLUGINS_PROCESS_ELF_CORE_REGISTERCONTEXTPOSIXCORE_RISCV32_H

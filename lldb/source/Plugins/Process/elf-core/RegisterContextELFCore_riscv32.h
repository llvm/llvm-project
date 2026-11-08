//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLDB_SOURCE_PLUGINS_PROCESS_ELF_CORE_REGISTERCONTEXTELFCORE_RISCV32_H
#define LLDB_SOURCE_PLUGINS_PROCESS_ELF_CORE_REGISTERCONTEXTELFCORE_RISCV32_H

#include "Plugins/Process/elf-core/RegisterUtilities.h"
#include "lldb/Register/Common/RegisterInfo_riscv32.h"
#include "lldb/Register/RegisterDescription.h"
#include "lldb/Target/Thread.h"
#include "lldb/Utility/DataExtractor.h"
#include "lldb/Target/ThreadRegisterContext.h"
#include "lldb/Register/RegisterValue.h"
#include "llvm/ADT/SmallVector.h"

#include <memory>

class RegisterContextELFCore_riscv32 : public lldb_private::ThreadRegisterContext {
public:
  struct csr_kv_t {
    uint32_t addr; // CSR address/index
    uint32_t val;  // CSR value
  };

  static llvm::Expected<std::unique_ptr<RegisterContextELFCore_riscv32>>
  Create(lldb_private::Thread &thread, const lldb_private::ArchSpec &arch,
         const lldb_private::DataExtractor &gpregset,
         llvm::ArrayRef<lldb_private::CoreNote> notes);

  ~RegisterContextELFCore_riscv32() override;

  void InvalidateAllRegisters() override;

  llvm::Expected<lldb_private::RegisterValue>
  ReadRegister(lldb_private::RegisterID id) override;

  llvm::Error WriteRegister(lldb_private::RegisterID id,
                     const lldb_private::RegisterValue &value) override;

protected:
  RegisterContextELFCore_riscv32(
      lldb_private::Thread &thread, const lldb_private::ArchSpec &arch,
      const lldb_private::DataExtractor &gpregset,
      llvm::ArrayRef<lldb_private::CoreNote> notes);

  /// Retrieves enabled RISC-V ISA extension feature names from the module's
  /// ArchSpec.
  void GetFeatures(llvm::SmallVectorImpl<std::string> &features) const;

private:
  lldb_private::DataExtractor m_gpregset;
  lldb_private::DataExtractor m_fpregset;
  lldb_private::DataExtractor m_csregset;
};

#endif // LLDB_SOURCE_PLUGINS_PROCESS_ELF_CORE_REGISTERCONTEXTELFCORE_RISCV32_H

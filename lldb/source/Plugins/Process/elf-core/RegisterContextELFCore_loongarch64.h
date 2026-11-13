//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLDB_SOURCE_PLUGINS_PROCESS_ELF_CORE_REGISTERCONTEXTELFCORE_LOONGARCH64_H
#define LLDB_SOURCE_PLUGINS_PROCESS_ELF_CORE_REGISTERCONTEXTELFCORE_LOONGARCH64_H

#include "lldb/Register/Arch/lldb-loongarch-register-enums.h"
#include "lldb/Register/Common/RegisterInfo_loongarch64.h"
#include "lldb/Target/ThreadRegisterContext.h"

#include "Plugins/Process/elf-core/RegisterUtilities.h"
#include "lldb/Target/Thread.h"
#include "lldb/Utility/DataExtractor.h"
#include "lldb/Register/RegisterValue.h"

#include <memory>

class RegisterContextELFCore_loongarch64
    : public lldb_private::ThreadRegisterContext {
public:
  static llvm::Expected<std::unique_ptr<RegisterContextELFCore_loongarch64>>
  Create(lldb_private::Thread &thread, const lldb_private::ArchSpec &arch,
         const lldb_private::DataExtractor &gpregset,
         llvm::ArrayRef<lldb_private::CoreNote> notes);

  ~RegisterContextELFCore_loongarch64() override;

  llvm::Expected<lldb_private::RegisterValue>
  ReadRegister(lldb_private::RegisterID id) override;

  llvm::Error WriteRegister(lldb_private::RegisterID id,
                     const lldb_private::RegisterValue &value) override;

protected:
  RegisterContextELFCore_loongarch64(
      lldb_private::Thread &thread, lldb_private::RegisterLayout layout,
      const lldb_private::DataExtractor &gpregset,
      llvm::ArrayRef<lldb_private::CoreNote> notes);

private:
  lldb_private::DataExtractor m_gpr;
  lldb_private::DataExtractor m_fpr;
};

#endif // LLDB_SOURCE_PLUGINS_PROCESS_ELF_CORE_REGISTERCONTEXTELFCORE_LOONGARCH64_H

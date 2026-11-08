//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLDB_SOURCE_PLUGINS_PROCESS_ELF_CORE_REGISTERCONTEXTELFCORE_X86_H
#define LLDB_SOURCE_PLUGINS_PROCESS_ELF_CORE_REGISTERCONTEXTELFCORE_X86_H

#include "Plugins/Process/elf-core/RegisterUtilities.h"
#include "lldb/Register/Arch/lldb-x86-register-enums.h"
#include "lldb/Register/Common/RegisterInfo_x86.h"
#include "lldb/Target/ThreadRegisterContext.h"

class RegisterContextELFCore_x86
    : public lldb_private::ThreadRegisterContext {
public:
  RegisterContextELFCore_x86(lldb_private::Thread &thread,
                             lldb_private::RegisterLayout layout,
                             const lldb_private::DataExtractor &gpregset,
                             llvm::ArrayRef<lldb_private::CoreNote> notes);

  llvm::Expected<lldb_private::RegisterValue>
  ReadRegister(lldb_private::RegisterID id) override;

  llvm::Error WriteRegister(lldb_private::RegisterID id,
                     const lldb_private::RegisterValue &value) override;

  bool ReadAllRegisterValues(lldb::WritableDataBufferSP &data_sp) override;

  bool WriteAllRegisterValues(const lldb::DataBufferSP &data_sp) override;

  bool HardwareSingleStep(bool enable) override;

private:
  std::unique_ptr<uint8_t[]> m_gpregset;
  std::unique_ptr<uint8_t[]> m_fpregset;
};

#endif // LLDB_SOURCE_PLUGINS_PROCESS_ELF_CORE_REGISTERCONTEXTELFCORE_X86_H

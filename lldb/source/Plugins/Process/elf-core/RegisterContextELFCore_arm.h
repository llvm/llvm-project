//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLDB_SOURCE_PLUGINS_PROCESS_ELF_CORE_REGISTERCONTEXTELFCORE_ARM_H
#define LLDB_SOURCE_PLUGINS_PROCESS_ELF_CORE_REGISTERCONTEXTELFCORE_ARM_H

#include "Plugins/Process/elf-core/RegisterUtilities.h"
#include "lldb/Register/Common/RegisterInfo_arm.h"
#include "lldb/Target/ThreadRegisterContext.h"
#include "lldb/Utility/DataBufferHeap.h"
#include "lldb/Utility/DataExtractor.h"

class RegisterContextELFCore_arm
    : public lldb_private::ThreadRegisterContext {
public:
  RegisterContextELFCore_arm(lldb_private::Thread &thread,
                             lldb_private::RegisterLayout layout,
                             const lldb_private::DataExtractor &gpregset,
                             llvm::ArrayRef<lldb_private::CoreNote> notes);

  ~RegisterContextELFCore_arm() override;

  llvm::Expected<lldb_private::RegisterValue>
  ReadRegister(lldb_private::RegisterID id) override;

  llvm::Error WriteRegister(lldb_private::RegisterID id,
                     const lldb_private::RegisterValue &value) override;

  bool ReadAllRegisterValues(lldb::WritableDataBufferSP &data_sp) override;

  bool WriteAllRegisterValues(const lldb::DataBufferSP &data_sp) override;

  bool HardwareSingleStep(bool enable) override;

private:
  lldb::DataBufferSP m_gpr_buffer;
  lldb_private::DataExtractor m_gpr;
  lldb_private::DataExtractor m_fpr;
};

#endif // LLDB_SOURCE_PLUGINS_PROCESS_ELF_CORE_REGISTERCONTEXTELFCORE_ARM_H

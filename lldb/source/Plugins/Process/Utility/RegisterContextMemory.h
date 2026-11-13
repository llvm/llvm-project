//===-- RegisterContextMemory.h ---------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLDB_SOURCE_PLUGINS_PROCESS_UTILITY_REGISTERCONTEXTMEMORY_H
#define LLDB_SOURCE_PLUGINS_PROCESS_UTILITY_REGISTERCONTEXTMEMORY_H

#include <vector>

#include "lldb/Register/RegisterDefinition.h"
#include "lldb/Target/ThreadRegisterContext.h"
#include "lldb/Utility/DataExtractor.h"
#include "lldb/lldb-private.h"

class RegisterContextMemory : public lldb_private::ThreadRegisterContext {
public:
  RegisterContextMemory(lldb_private::Thread &thread,
                        uint32_t concrete_frame_idx,
                        std::shared_ptr<const lldb_private::RegisterLayout> reg_info_sp,
                        lldb::addr_t reg_data_addr);

  ~RegisterContextMemory() override;

  void InvalidateAllRegisters() override;

  // If all of the thread register are in a contiguous buffer in
  // memory, then the default ReadRegister/WriteRegister and
  // ReadAllRegisterValues/WriteAllRegisterValues will work. If thread
  // registers are not contiguous, clients will want to subclass this
  // class and modify the read/write functions as needed.

  llvm::Expected<lldb_private::RegisterValue>
  ReadRegister(lldb_private::RegisterID id) override;

  llvm::Error
  WriteRegister(lldb_private::RegisterID id,
                     const lldb_private::RegisterValue &reg_value) override;

  llvm::Expected<lldb_private::RegisterCheckpoint> SaveCheckpoint() override;

  bool ReadAllRegisterValues(lldb::WritableDataBufferSP &data_sp) override;

  bool WriteAllRegisterValues(const lldb::DataBufferSP &data_sp) override;

  void SetAllRegisterData(const lldb::DataBufferSP &data_sp);

protected:
  void SetAllRegisterValid(bool b);

  std::shared_ptr<const lldb_private::RegisterLayout> m_reg_infos_sp;
  std::vector<bool> m_reg_valid;
  lldb::WritableDataBufferSP m_data;
  lldb_private::DataExtractor m_reg_data;
  lldb::addr_t m_reg_data_addr; // If this is valid, then we have a register
                                // context that is stored in memmory

private:
  RegisterContextMemory(const RegisterContextMemory &) = delete;
  const RegisterContextMemory &
  operator=(const RegisterContextMemory &) = delete;
};

#endif // LLDB_SOURCE_PLUGINS_PROCESS_UTILITY_REGISTERCONTEXTMEMORY_H

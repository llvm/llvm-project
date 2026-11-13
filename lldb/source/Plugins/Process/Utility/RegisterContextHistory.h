//===-- RegisterContextHistory.h ----------------------------------------*- C++
//-*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLDB_SOURCE_PLUGINS_PROCESS_UTILITY_REGISTERCONTEXTHISTORY_H
#define LLDB_SOURCE_PLUGINS_PROCESS_UTILITY_REGISTERCONTEXTHISTORY_H

#include "lldb/Target/ThreadRegisterContext.h"
#include "lldb/lldb-private.h"

namespace lldb_private {

class RegisterContextHistory : public lldb_private::ThreadRegisterContext {
public:
  typedef std::shared_ptr<RegisterContextHistory> SharedPtr;

  RegisterContextHistory(Thread &thread, uint32_t concrete_frame_idx,
                         uint32_t address_byte_size, lldb::addr_t pc_value);

  ~RegisterContextHistory() override;

  llvm::Expected<lldb_private::RegisterValue>
  ReadRegister(lldb_private::RegisterID id) override;

  llvm::Error WriteRegister(lldb_private::RegisterID id,
                     const lldb_private::RegisterValue &value) override;

  bool ReadAllRegisterValues(lldb::WritableDataBufferSP &data_sp) override;

  bool WriteAllRegisterValues(const lldb::DataBufferSP &data_sp) override;

private:
  lldb::addr_t m_pc_value;

  RegisterContextHistory(const RegisterContextHistory &) = delete;
  const RegisterContextHistory &
  operator=(const RegisterContextHistory &) = delete;
};
} // namespace lldb_private

#endif // LLDB_SOURCE_PLUGINS_PROCESS_UTILITY_REGISTERCONTEXTHISTORY_H

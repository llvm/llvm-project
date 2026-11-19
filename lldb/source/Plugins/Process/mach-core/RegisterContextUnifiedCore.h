//===-- RegisterContextUnifiedCore.h --------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLDB_SOURCE_PLUGINS_PROCESS_MACH_CORE_REGISTERCONTEXTUNIFIEDCORE_H
#define LLDB_SOURCE_PLUGINS_PROCESS_MACH_CORE_REGISTERCONTEXTUNIFIEDCORE_H

#include <string>
#include <vector>

#include "lldb/Target/ThreadRegisterContext.h"
#include "lldb/Utility/ConstString.h"
#include "lldb/Utility/StructuredData.h"
#include "lldb/lldb-enumerations.h"
#include "lldb/lldb-private.h"

namespace lldb_private {

class RegisterContextUnifiedCore : public ThreadRegisterContext {
public:
  RegisterContextUnifiedCore(
      Thread &thread, uint32_t concrete_frame_idx,
      lldb::ThreadRegisterContextSP core_thread_regctx_sp,
      lldb_private::StructuredData::ObjectSP metadata_thread_registers);

  void InvalidateAllRegisters() override {};

  llvm::Expected<lldb_private::RegisterValue>
  ReadRegister(lldb_private::RegisterID id) override;

  llvm::Error WriteRegister(lldb_private::RegisterID id,
                     const lldb_private::RegisterValue &value) override;

private:
  std::map<RegisterID, RegisterValue> m_values;
};

} // namespace lldb_private

#endif // LLDB_SOURCE_PLUGINS_PROCESS_MACH_CORE_REGISTERCONTEXTUNIFIEDCORE_H

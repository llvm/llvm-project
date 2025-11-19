//===-- RegisterContextUnifiedCore.h --------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLDB_SOURCE_PLUGINS_PROCESS_REGISTERCONTEXT_UNIFIED_CORE_H
#define LLDB_SOURCE_PLUGINS_PROCESS_REGISTERCONTEXT_UNIFIED_CORE_H

#include <string>
#include <vector>

#include "lldb/Target/RegisterContext.h"
#include "lldb/Utility/ConstString.h"
#include "lldb/Utility/StructuredData.h"
#include "lldb/lldb-enumerations.h"
#include "lldb/lldb-private.h"

namespace lldb_private {

class RegisterContextUnifiedCore : public RegisterContext {
public:
  RegisterContextUnifiedCore(
      Thread &thread, uint32_t concrete_frame_idx,
      lldb::RegisterContextSP core_thread_regctx_sp,
      lldb_private::StructuredData::ObjectSP metadata_thread_registers);

  void InvalidateAllRegisters() override {};

  size_t GetRegisterCount() override;

  const lldb_private::RegisterInfo *GetRegisterInfoAtIndex(size_t reg) override;

  size_t GetRegisterSetCount() override;

  const lldb_private::RegisterSet *GetRegisterSet(size_t set) override;

  bool ReadRegister(const lldb_private::RegisterInfo *reg_info,
                    lldb_private::RegisterValue &value) override;

  bool WriteRegister(const lldb_private::RegisterInfo *reg_info,
                     const lldb_private::RegisterValue &value) override;

private:
  std::vector<lldb_private::RegisterSet> m_register_sets;
  std::vector<lldb_private::RegisterInfo> m_register_infos;
  /// For each register set, an array of register numbers included.
  std::map<size_t, std::vector<uint32_t>> m_regset_regnum_collection;
  /// Bytes of the register contents.
  std::vector<uint8_t> m_register_data;
};

} // namespace lldb_private

#endif // LLDB_SOURCE_PLUGINS_PROCESS_REGISTERCONTEXT_UNIFIED_CORE_H

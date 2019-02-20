//===-- MemoryHistoryASan.h -------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef liblldb_MemoryHistoryASan_h_
#define liblldb_MemoryHistoryASan_h_

#include "lldb/Target/ABI.h"
#include "lldb/Target/MemoryHistory.h"
#include "lldb/Target/Process.h"
#include "lldb/lldb-private.h"

namespace lldb_private {

class MemoryHistoryASan : public lldb_private::MemoryHistory {
public:
  ~MemoryHistoryASan() override = default;

  static lldb::MemoryHistorySP
  CreateInstance(const lldb::ProcessSP &process_sp);

  static void Initialize();

  static void Terminate();

  static lldb_private::ConstString GetPluginNameStatic();

  lldb_private::ConstString GetPluginName() override {
    return GetPluginNameStatic();
  }

  uint32_t GetPluginVersion() override { return 1; }

  lldb_private::HistoryThreads GetHistoryThreads(lldb::addr_t address) override;

private:
  MemoryHistoryASan(const lldb::ProcessSP &process_sp);

  lldb::ProcessWP m_process_wp;
};

} // namespace lldb_private

#endif // liblldb_MemoryHistoryASan_h_

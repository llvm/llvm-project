//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLDB_SOURCE_PLUGINS_PROCESS_WASM_REGISTERCONTEXTWASM_H
#define LLDB_SOURCE_PLUGINS_PROCESS_WASM_REGISTERCONTEXTWASM_H

#include "Plugins/Process/gdb-remote/GDBRemoteRegisterContext.h"
#include "Plugins/Process/gdb-remote/ThreadGDBRemote.h"
#include "ThreadWasm.h"
#include "Utility/WasmVirtualRegisters.h"
#include "lldb/lldb-private-types.h"
#include <unordered_map>

namespace lldb_private {
namespace wasm {

class RegisterContextWasm;

typedef std::shared_ptr<RegisterContextWasm> RegisterContextWasmSP;

struct WasmVirtualRegisterInfo : public RegisterInfo {
  WasmVirtualRegisterKinds kind;
  uint32_t index;

  WasmVirtualRegisterInfo(WasmVirtualRegisterKinds kind, uint32_t index)
      : RegisterInfo(), kind(kind), index(index) {}
};

class RegisterContextWasm
    : public process_gdb_remote::GDBRemoteRegisterContext {
public:
  RegisterContextWasm(
      process_gdb_remote::ThreadGDBRemote &thread, uint32_t concrete_frame_idx,
      process_gdb_remote::GDBRemoteDynamicRegisterInfoSP reg_info_sp);

  ~RegisterContextWasm() override;

  uint32_t ConvertRegisterKindToRegisterNumber(lldb::RegisterKind kind,
                                               uint32_t num) override;

  void InvalidateAllRegisters() override;

  size_t GetRegisterCount() override;

  const RegisterInfo *GetRegisterInfoAtIndex(size_t reg) override;

  size_t GetRegisterSetCount() override;

  const RegisterSet *GetRegisterSet(size_t reg_set) override;

  bool ReadRegister(const RegisterInfo *reg_info,
                    RegisterValue &value) override;

  bool WriteRegister(const RegisterInfo *reg_info,
                     const RegisterValue &value) override;

private:
  std::unordered_map<size_t, std::unique_ptr<WasmVirtualRegisterInfo>>
      m_register_map;
};

} // namespace wasm
} // namespace lldb_private

#endif

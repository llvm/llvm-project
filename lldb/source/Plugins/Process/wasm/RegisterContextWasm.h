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
#include "lldb/Register/Arch/lldb-wasm-register-enums.h"
#include "lldb/lldb-private-types.h"
#include <algorithm>
#include <unordered_map>

namespace lldb_private {
namespace wasm {

class RegisterContextWasm;

typedef std::shared_ptr<RegisterContextWasm> RegisterContextWasmSP;

class RegisterContextWasm
    : public process_gdb_remote::GDBRemoteRegisterContext {
public:
  RegisterContextWasm(process_gdb_remote::ThreadGDBRemote &thread,
                      uint32_t concrete_frame_idx,
                      std::shared_ptr<const RegisterLayout> layout_sp);

  ~RegisterContextWasm() override;

  llvm::Expected<RegisterID>
  ResolveRegisterNumber(lldb::RegisterKind kind,
                        uint32_t number) const override;

  void InvalidateAllRegisters() override;

  llvm::Expected<lldb_private::RegisterValue>
  ReadRegister(lldb_private::RegisterID id) override;

  llvm::Error WriteRegister(lldb_private::RegisterID id,
                            const lldb_private::RegisterValue &value) override;

private:
  /// The module whose code this context's frame is executing. Resolved on each
  /// use rather than cached: the innermost frame's register context is the
  /// thread's own and outlives a stop, so a cached id would go on naming the
  /// module of a previous stop.
  uint32_t GetModuleID();
};

} // namespace wasm
} // namespace lldb_private

#endif

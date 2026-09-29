//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLDB_SOURCE_PLUGINS_PROCESS_WASM_THREADWASM_H
#define LLDB_SOURCE_PLUGINS_PROCESS_WASM_THREADWASM_H

#include "Plugins/Process/gdb-remote/ThreadGDBRemote.h"

namespace lldb_private {
namespace wasm {

/// ProcessWasm provides the access to the Wasm program state
/// retrieved from the Wasm engine.
class ThreadWasm : public process_gdb_remote::ThreadGDBRemote {
public:
  ThreadWasm(Process &process, lldb::tid_t tid)
      : process_gdb_remote::ThreadGDBRemote(process, tid) {}
  ~ThreadWasm() override = default;

  /// Retrieve the current call stack from the WebAssembly remote process.
  llvm::Expected<std::vector<lldb::addr_t>> GetWasmCallStack();

  /// Return the program counter the Wasm unwinder recorded for the given
  /// concrete frame index, or LLDB_INVALID_ADDRESS if it is unavailable.
  lldb::addr_t GetConcreteFramePC(uint32_t concrete_frame_idx);

  lldb::RegisterContextSP
  CreateRegisterContextForFrame(StackFrame *frame) override;

protected:
  Unwind &GetUnwinder() override;

  ThreadWasm(const ThreadWasm &);
  const ThreadWasm &operator=(const ThreadWasm &) = delete;
};

} // namespace wasm
} // namespace lldb_private

#endif // LLDB_SOURCE_PLUGINS_PROCESS_WASM_THREADWASM_H

//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLDB_SOURCE_PLUGINS_PROCESS_WASM_PROCESSWASM_H
#define LLDB_SOURCE_PLUGINS_PROCESS_WASM_PROCESSWASM_H

#include "Plugins/ObjectFile/wasm/WasmAddress.h"
#include "Plugins/Process/gdb-remote/ProcessGDBRemote.h"
#include "Utility/WasmVirtualRegisters.h"

namespace lldb_private {
namespace wasm {

/// ProcessWasm provides the access to the Wasm program state
/// retrieved from the Wasm engine.
class ProcessWasm : public process_gdb_remote::ProcessGDBRemote {
public:
  ProcessWasm(lldb::TargetSP target_sp, lldb::ListenerSP listener_sp);
  ~ProcessWasm() override = default;

  static lldb::ProcessSP CreateInstance(lldb::TargetSP target_sp,
                                        lldb::ListenerSP listener_sp,
                                        const FileSpec *crash_file_path,
                                        bool can_connect);

  static void Initialize();
  static void DebuggerInitialize(Debugger &debugger);
  static void Terminate();

  static llvm::StringRef GetPluginNameStatic();
  static llvm::StringRef GetPluginDescriptionStatic();

  llvm::StringRef GetPluginName() override;

  bool ShouldUseMemoryCache(const ProcessAddress &process_addr) override;

  size_t DoReadMemory(const ProcessAddress &vm_addr, void *buf, size_t size,
                      Status &error) override;

  bool CanDebug(lldb::TargetSP target_sp,
                bool plugin_specified_by_name) override;

  /// Retrieve the current call stack from the WebAssembly remote process.
  llvm::Expected<std::vector<lldb::addr_t>> GetWasmCallStack(lldb::tid_t tid);

  /// Query the value of a WebAssembly variable from the WebAssembly
  /// remote process.
  llvm::Expected<lldb::DataBufferSP>
  GetWasmVariable(WasmVirtualRegisterKinds kind, int frame_index, int index);

protected:
  std::shared_ptr<process_gdb_remote::ThreadGDBRemote>
  CreateThread(lldb::tid_t tid) override;

private:
  friend class UnwindWasm;
  friend class ThreadWasm;

  /// Read a WebAssembly global by its index in the global index space of the
  /// module it belongs to. The index space is per module, so an index only
  /// names a global together with the module it is an index into.
  size_t ReadGlobal(uint32_t module_id, uint32_t index, void *buf, size_t size,
                    Status &error);

  lldb::DynamicRegisterInfoSP &GetRegisterInfo() { return m_register_info_sp; }

  ProcessWasm(const ProcessWasm &);
  const ProcessWasm &operator=(const ProcessWasm &) = delete;
};

} // namespace wasm
} // namespace lldb_private

#endif

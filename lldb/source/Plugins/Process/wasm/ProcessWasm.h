//===-- ProcessWasm.h -------------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLDB_SOURCE_PLUGINS_PROCESS_WASM_PROCESSWASM_H
#define LLDB_SOURCE_PLUGINS_PROCESS_WASM_PROCESSWASM_H

#include "Plugins/Process/gdb-remote/ProcessGDBRemote.h"
#include "lldb/Target/RegisterContext.h"

namespace lldb_private {
namespace wasm {

/// Each WebAssembly module has separated address spaces for Code and Memory.
/// A WebAssembly module also has a Data section which, when the module is
/// loaded, gets mapped into a region in the module Memory.
/// For the purpose of debugging, we can represent all these separated 32-bit
/// address spaces with a single virtual 64-bit address space.
///
/// Struct wasm_addr_t provides this encoding using bitfields
///
enum WasmAddressType { Memory = 0x00, Code = 0x01, Invalid = 0x03 };

struct wasm_addr_t {
  uint64_t offset : 32;
  uint64_t module_id : 30;
  uint64_t type : 2;

  wasm_addr_t(lldb::addr_t addr)
      : type(addr >> 62), module_id((addr & 0x00ffffff00000000) >> 32),
        offset(addr & 0x00000000ffffffff) {}

  wasm_addr_t(WasmAddressType type_, uint32_t module_id_, uint32_t offset_)
      : type(type_), module_id(module_id_), offset(offset_) {}

  WasmAddressType GetType() { return static_cast<WasmAddressType>(type); }
  operator lldb::addr_t() { return *(uint64_t *)this; }
};

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

  /// PluginInterface protocol.
  /// \{
  llvm::StringRef GetPluginName() override;
  /// \}

  size_t ReadMemory(lldb::addr_t vm_addr, void *buf, size_t size,
                    Status &error) override;

  lldb::ModuleSP ReadModuleFromMemory(const FileSpec &file_spec,
                                      lldb::addr_t header_addr,
                                      size_t size_to_read = 512) override;

  lldb::addr_t FixMemoryAddress(lldb::addr_t address,
                                StackFrame *stack_frame) const override;

  /// Query the value of a WebAssembly local variable from the WebAssembly
  /// remote process.
  bool GetWasmLocal(int frame_index, int index, void *buf, size_t buffer_size,
                    size_t &size);

  /// Query the value of a WebAssembly global variable from the WebAssembly
  /// remote process.
  bool GetWasmGlobal(int frame_index, int index, void *buf, size_t buffer_size,
                     size_t &size);

  /// Query the value of an item in the WebAssembly operand stack from the
  /// WebAssembly remote process.
  bool GetWasmStackValue(int frame_index, int index, void *buf,
                         size_t buffer_size, size_t &size);

  /// Read from the WebAssembly Memory space.
  size_t WasmReadMemory(uint32_t wasm_module_id, lldb::addr_t addr, void *buf,
                        size_t buffer_size);

  /// Read from the WebAssembly Data space.
  size_t WasmReadData(uint32_t wasm_module_id, lldb::addr_t addr, void *buf,
                      size_t buffer_size);

  /// Retrieve the current call stack from the WebAssembly remote process.
  bool GetWasmCallStack(lldb::tid_t tid,
                        std::vector<lldb::addr_t> &call_stack_pcs);

  // Check if a given Process
  bool CanDebug(lldb::TargetSP target_sp,
                bool plugin_specified_by_name) override;

protected:
  /// ProcessGDBRemote protocol.
  /// \{
  std::shared_ptr<process_gdb_remote::ThreadGDBRemote>
  CreateThread(lldb::tid_t tid);
  /// \}

private:
  friend class UnwindWasm;
  friend class ThreadWasm;

  process_gdb_remote::GDBRemoteDynamicRegisterInfoSP &GetRegisterInfo() {
    return m_register_info_sp;
  }

  ProcessWasm(const ProcessWasm &);
  const ProcessWasm &operator=(const ProcessWasm &) = delete;
};

} // namespace wasm
} // namespace lldb_private

#endif // LLDB_SOURCE_PLUGINS_PROCESS_WASM_PROCESSWASM_H

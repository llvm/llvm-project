//===----- wasmRegisterContext.h --------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLDB_SOURCE_PLUGINS_PROCESS_WASM_WASMREGISTERCONTEXT_H
#define LLDB_SOURCE_PLUGINS_PROCESS_WASM_WASMREGISTERCONTEXT_H

#include "Plugins/Process/gdb-remote/GDBRemoteRegisterContext.h"
#include "Plugins/SymbolFile/DWARF/DWARFWasm.h"
#include "ThreadWasm.h"
#include "lldb/lldb-private-types.h"
#include <unordered_map>

namespace lldb_private {
namespace wasm {

class WasmRegisterContext;

typedef std::shared_ptr<WasmRegisterContext> WasmRegisterContextSP;

/*
 * WebAssembly locals, globals and operand stacks are encoded as virtual
 * registers with the format:
 *   | WasmVirtualRegisterKinds: 2 bits | index: 30 bits |
 * where tag is:
 *   0: Not a WebAssembly location
 *   1: Local
 *   2: Global
 *   3: Operand stack value
 */

enum WasmVirtualRegisterKinds {
  eNotAWasmLocation = 0,
  eLocal,        ///< wasm local
  eGlobal,       ///< wasm global
  eOperandStack, ///< wasm operand stack
};

struct WasmVirtualRegisterInfo : public RegisterInfo {
  WasmVirtualRegisterKinds kind;
  uint32_t index;

  WasmVirtualRegisterInfo(WasmVirtualRegisterKinds kind, uint32_t index)
      : RegisterInfo(), kind(kind), index(index) {}

  static WasmVirtualRegisterKinds VirtualRegisterKindFromDWARFLocation(
      plugin::dwarf::DWARFWasmLocation dwarf_location) {
    switch (dwarf_location) {
    case plugin::dwarf::DWARFWasmLocation::eLocal:
      return WasmVirtualRegisterKinds::eLocal;
    case plugin::dwarf::DWARFWasmLocation::eGlobal:
    case plugin::dwarf::DWARFWasmLocation::eGlobalU32:
      return WasmVirtualRegisterKinds::eLocal;
    case plugin::dwarf::DWARFWasmLocation::eOperandStack:
      return WasmVirtualRegisterKinds::eOperandStack;
    default:
      llvm_unreachable("Invalid DWARF Wasm location");
    }
  }
};

class WasmRegisterContext
    : public process_gdb_remote::GDBRemoteRegisterContext {
public:
  static const uint32_t kTagMask = 0x03;
  static const uint32_t kIndexMask = 0x3fffffff;
  static const uint32_t kTagShift = 30;

  WasmRegisterContext(
      wasm::ThreadWasm &thread, uint32_t concrete_frame_idx,
      process_gdb_remote::GDBRemoteDynamicRegisterInfoSP reg_info_sp);

  ~WasmRegisterContext() override;

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

#endif // LLDB_SOURCE_PLUGINS_PROCESS_WASM_WASMREGISTERCONTEXT_H

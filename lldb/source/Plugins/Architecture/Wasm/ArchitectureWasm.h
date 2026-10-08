//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLDB_SOURCE_PLUGINS_ARCHITECTURE_WASM_ARCHITECTUREWASM_H
#define LLDB_SOURCE_PLUGINS_ARCHITECTURE_WASM_ARCHITECTUREWASM_H

#include "lldb/Core/Architecture.h"

namespace lldb_private {

class ArchitectureWasm : public Architecture {
public:
  static llvm::StringRef GetPluginNameStatic() { return "wasm"; }
  static void Initialize();
  static void Terminate();

  llvm::StringRef GetPluginName() override { return GetPluginNameStatic(); }

  void OverrideStopInfo(Thread &thread) const override {}

  /// A WebAssembly function begins with a local variable declaration header
  /// that is part of the function but is not an executable instruction. Skip it
  /// so the address lands on the first instruction.
  Address SkipFunctionHeader(Address addr) const override;

private:
  static std::unique_ptr<Architecture> Create(const ArchSpec &arch);
  ArchitectureWasm() = default;
};

} // namespace lldb_private

#endif // LLDB_SOURCE_PLUGINS_ARCHITECTURE_WASM_ARCHITECTUREWASM_H

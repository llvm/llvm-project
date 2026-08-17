//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "Plugins/Architecture/Wasm/ArchitectureWasm.h"
#include "lldb/Core/PluginManager.h"
#include "lldb/Symbol/Symbol.h"
#include "lldb/Symbol/SymbolContext.h"
#include "lldb/Utility/ArchSpec.h"

using namespace lldb_private;
using namespace lldb;

LLDB_PLUGIN_DEFINE(ArchitectureWasm)

void ArchitectureWasm::Initialize() {
  PluginManager::RegisterPlugin(GetPluginNameStatic(),
                                "WebAssembly-specific algorithms",
                                &ArchitectureWasm::Create);
}

void ArchitectureWasm::Terminate() {
  PluginManager::UnregisterPlugin(&ArchitectureWasm::Create);
}

std::unique_ptr<Architecture> ArchitectureWasm::Create(const ArchSpec &arch) {
  llvm::Triple::ArchType machine = arch.GetMachine();
  if (machine != llvm::Triple::wasm32 && machine != llvm::Triple::wasm64)
    return nullptr;
  return std::unique_ptr<Architecture>(new ArchitectureWasm());
}

Address ArchitectureWasm::SkipFunctionHeader(Address addr) const {
  // The header is recorded as the symbol's prologue.
  SymbolContext sc;
  addr.CalculateSymbolContext(&sc, eSymbolContextSymbol);
  if (!sc.symbol)
    return addr;

  const uint32_t header_size = sc.symbol->GetPrologueByteSize();
  if (!header_size)
    return addr;

  const addr_t symbol_addr = sc.symbol->GetAddress().GetFileAddress();
  const addr_t this_addr = addr.GetFileAddress();
  if (this_addr < symbol_addr || this_addr >= symbol_addr + header_size)
    return addr;

  addr.Slide(symbol_addr + header_size - this_addr);
  return addr;
}

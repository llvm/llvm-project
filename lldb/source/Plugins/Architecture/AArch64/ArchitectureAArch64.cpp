//===-- ArchitectureAArch64.cpp -------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "Plugins/Architecture/AArch64/ArchitectureAArch64.h"
#include "lldb/Core/PluginManager.h"
#include "lldb/Utility/ArchSpec.h"

using namespace lldb_private;
using namespace lldb;

LLDB_PLUGIN_DEFINE(ArchitectureAArch64)

ConstString ArchitectureAArch64::GetPluginNameStatic() {
  return ConstString("aarch64");
}

void ArchitectureAArch64::Initialize() {
  PluginManager::RegisterPlugin(GetPluginNameStatic(),
                                "AArch64-specific algorithms",
                                &ArchitectureAArch64::Create);
}

void ArchitectureAArch64::Terminate() {
  PluginManager::UnregisterPlugin(&ArchitectureAArch64::Create);
}

std::unique_ptr<Architecture>
ArchitectureAArch64::Create(const ArchSpec &arch) {
  auto machine = arch.GetMachine();
  if (machine != llvm::Triple::aarch64 && machine != llvm::Triple::aarch64_be &&
      machine != llvm::Triple::aarch64_32) {
    return nullptr;
  }
  return std::unique_ptr<Architecture>(new ArchitectureAArch64());
}

ConstString ArchitectureAArch64::GetPluginName() {
  return GetPluginNameStatic();
}
uint32_t ArchitectureAArch64::GetPluginVersion() { return 1; }

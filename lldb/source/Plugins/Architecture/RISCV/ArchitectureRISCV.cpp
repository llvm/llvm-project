//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "Plugins/Architecture/RISCV/ArchitectureRISCV.h"
#include "lldb/Core/PluginManager.h"
#include "lldb/Utility/ArchSpec.h"

using namespace lldb_private;

LLDB_PLUGIN_DEFINE(ArchitectureRISCV)

void ArchitectureRISCV::Initialize() {
  PluginManager::RegisterPlugin(GetPluginNameStatic(),
                                "RISC-V-specific algorithms",
                                &ArchitectureRISCV::Create);
}

void ArchitectureRISCV::Terminate() {
  PluginManager::UnregisterPlugin(&ArchitectureRISCV::Create);
}

std::unique_ptr<Architecture> ArchitectureRISCV::Create(const ArchSpec &arch) {
  auto machine = arch.GetMachine();
  if (machine != llvm::Triple::riscv32 && machine != llvm::Triple::riscv64)
    return nullptr;
  return std::unique_ptr<Architecture>(new ArchitectureRISCV());
}

bool ArchitectureRISCV::IsValidTrapInstruction(
    llvm::ArrayRef<uint8_t> reference, llvm::ArrayRef<uint8_t> observed) const {
  // RISC-V has only two trap encodings here: 16-bit C.EBREAK or 32-bit EBREAK.
  // These instructions don't have any operands so check that the reference and
  // observed bytes match.
  if ((reference.size() != 2 && reference.size() != 4) ||
      reference.size() > observed.size())
    return false;

  return reference == observed.take_front(reference.size());
}

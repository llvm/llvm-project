//===-- ArchitectureRISCV.cpp----------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "Plugins/Architecture/RISCV/ArchitectureRISCV.h"
#include "lldb/Core/PluginManager.h"
#include "lldb/Target/RegisterContext.h"
#include "lldb/Target/Thread.h"
#include "lldb/Utility/ArchSpec.h"

#include "llvm/IR/LegacyPassManager.h"

#include "DirectToIndirectFCR.h"

using namespace lldb_private;
using namespace lldb;

LLDB_PLUGIN_DEFINE(ArchitectureRISCV)

void ArchitectureRISCV::Initialize() {
  PluginManager::RegisterPlugin(GetPluginNameStatic(),
                                "RISCV-specific algorithms",
                                &ArchitectureRISCV::Create);
}

void ArchitectureRISCV::Terminate() {
  PluginManager::UnregisterPlugin(&ArchitectureRISCV::Create);
}

std::unique_ptr<Architecture> ArchitectureRISCV::Create(const ArchSpec &arch) {
  if (!arch.GetTriple().isRISCV())
    return nullptr;
  return std::unique_ptr<Architecture>(new ArchitectureRISCV());
}

void ArchitectureRISCV::OverrideStopInfo(Thread &thread) const {}

std::unique_ptr<llvm::legacy::PassManager>
ArchitectureRISCV::GetArchitectureCustomPasses(
    const ExecutionContext &exe_ctx, const llvm::StringRef expr) const {
  // LLDB generates additional support functions like
  // '_$__lldb_valid_pointer_check', that do not require custom passes
  if (expr != "$__lldb_expr")
    return nullptr;

  std::unique_ptr<llvm::legacy::PassManager> custom_passes =
      std::make_unique<llvm::legacy::PassManager>();
  auto *P = createDirectToIndirectFCR(exe_ctx);
  custom_passes->add(P);
  return custom_passes;
}

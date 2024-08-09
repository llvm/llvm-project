//===-- ArchitectureRISCV.h -------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#pragma once

#include "lldb/Core/Architecture.h"

namespace lldb_private {

class ArchitectureRISCV : public Architecture {
public:
  static llvm::StringRef GetPluginNameStatic() { return "riscv"; }
  static void Initialize();
  static void Terminate();

  llvm::StringRef GetPluginName() override { return GetPluginNameStatic(); }

  void OverrideStopInfo(Thread &thread) const override;

  std::unique_ptr<llvm::legacy::PassManager>
  GetArchitectureCustomPasses(const ExecutionContext &exe_ctx,
                              const llvm::StringRef expr) const override;

private:
  static std::unique_ptr<Architecture> Create(const ArchSpec &arch);
  ArchitectureRISCV() = default;
};

} // namespace lldb_private

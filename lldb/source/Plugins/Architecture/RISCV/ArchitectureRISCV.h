//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLDB_SOURCE_PLUGINS_ARCHITECTURE_RISCV_ARCHITECTURERISCV_H
#define LLDB_SOURCE_PLUGINS_ARCHITECTURE_RISCV_ARCHITECTURERISCV_H

#include "lldb/Core/Architecture.h"

namespace lldb_private {

class ArchitectureRISCV : public Architecture {
public:
  static llvm::StringRef GetPluginNameStatic() { return "riscv"; }
  static void Initialize();
  static void Terminate();

  llvm::StringRef GetPluginName() override { return GetPluginNameStatic(); }

  void OverrideStopInfo(Thread &thread) const override {}

  bool IsValidTrapInstruction(llvm::ArrayRef<uint8_t> reference,
                              llvm::ArrayRef<uint8_t> observed) const override;

private:
  static std::unique_ptr<Architecture> Create(const ArchSpec &arch);
  ArchitectureRISCV() = default;
};

} // namespace lldb_private

#endif // LLDB_SOURCE_PLUGINS_ARCHITECTURE_RISCV_ARCHITECTURERISCV_H

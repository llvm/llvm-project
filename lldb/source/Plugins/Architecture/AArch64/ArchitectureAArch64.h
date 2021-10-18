//===-- ArchitectureAArch64.h -----------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLDB_SOURCE_PLUGINS_ARCHITECTURE_AARCH64_ARCHITECTUREAARCH64_H
#define LLDB_SOURCE_PLUGINS_ARCHITECTURE_AARCH64_ARCHITECTUREAARCH64_H

#include "Plugins/Process/Utility/MemoryTagManagerAArch64MTE.h"
#include "lldb/Core/Architecture.h"

namespace lldb_private {

class ArchitectureAArch64 : public Architecture {
public:
  static ConstString GetPluginNameStatic();
  static void Initialize();
  static void Terminate();

  llvm::StringRef GetPluginName() override {
    return GetPluginNameStatic().GetStringRef();
  }

  void OverrideStopInfo(Thread &thread) const override{};

  const MemoryTagManager *GetMemoryTagManager() const override {
    return &m_memory_tag_manager;
  }

private:
  static std::unique_ptr<Architecture> Create(const ArchSpec &arch);
  ArchitectureAArch64() = default;
  MemoryTagManagerAArch64MTE m_memory_tag_manager;
};

} // namespace lldb_private

#endif // LLDB_SOURCE_PLUGINS_ARCHITECTURE_AARCH64_ARCHITECTUREAARCH64_H

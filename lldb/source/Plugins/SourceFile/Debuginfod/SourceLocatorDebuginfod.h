//===-- SourceLocatorDebuginfod.h -------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLDB_SOURCE_PLUGINS_SOURCELOCATOR_DEBUGINFOD_SOURCELOCATORDEBUGINFOD_H
#define LLDB_SOURCE_PLUGINS_SOURCELOCATOR_DEBUGINFOD_SOURCELOCATORDEBUGINFOD_H

#include "lldb/Core/SourceLocator.h"
#include <lldb/Core/Debugger.h>
namespace lldb_private {

class SourceLocatorDebuginfod : public SourceLocator {
public:
  explicit SourceLocatorDebuginfod() = default;
  static void Initialize();
  static void Terminate();

  static llvm::StringRef GetPluginNameStatic() { return "debuginfod-source"; };
  static llvm::StringRef GetPluginDescriptionStatic() {
    return "Debuginfod source locator";
  };

  static SourceLocator *CreateInstance();
  // static SourceLoc
  llvm::StringRef GetPluginName() override { return GetPluginNameStatic(); }

  static std::optional<FileSpec> LocateSourceFile(const ModuleSpec &module_spec,
                                                  const FileSpec &file_spec);
};
} // namespace lldb_private

#endif // SOURCELOCATORDEBUGINFOD_H

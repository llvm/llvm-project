//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLDB_SOURCE_PLUGINS_SYMBOLLOCATOR_SCRIPTED_SYMBOLLOCATORSCRIPTED_H
#define LLDB_SOURCE_PLUGINS_SYMBOLLOCATOR_SCRIPTED_SYMBOLLOCATORSCRIPTED_H

#include "lldb/Symbol/SymbolLocator.h"
#include "lldb/lldb-private.h"

namespace lldb_private {

class SymbolLocatorScripted : public SymbolLocator {
public:
  SymbolLocatorScripted();

  static void Initialize();
  static void Terminate();

  static llvm::StringRef GetPluginNameStatic() { return "scripted"; }
  static llvm::StringRef GetPluginDescriptionStatic();

  static lldb_private::SymbolLocator *CreateInstance();

  /// PluginInterface protocol.
  llvm::StringRef GetPluginName() override { return GetPluginNameStatic(); }

  static std::optional<FileSpec>
  LocateSourceFile(const lldb::TargetSP &target_sp,
                   const lldb::ModuleSP &module_sp,
                   const FileSpec &original_source_file);
};

} // namespace lldb_private

#endif // LLDB_SOURCE_PLUGINS_SYMBOLLOCATOR_SCRIPTED_SYMBOLLOCATORSCRIPTED_H

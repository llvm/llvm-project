//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLDB_SOURCE_PLUGINS_SYMBOLLOCATOR_SYMSTORE_SYMBOLLOCATORSYMSTORE_H
#define LLDB_SOURCE_PLUGINS_SYMBOLLOCATOR_SYMSTORE_SYMBOLLOCATORSYMSTORE_H

#include "lldb/Core/Debugger.h"
#include "lldb/Symbol/SymbolLocator.h"
#include "lldb/lldb-private.h"

namespace lldb_private {

/// This plugin implements lookup in Microsoft SymStore instances. This can work
/// cross-platform and for arbitrary debug info formats, but the focus is on PDB
/// with PE/COFF binaries on Windows.
class SymbolLocatorSymStore : public SymbolLocator {
public:
  SymbolLocatorSymStore();

  static void Initialize();
  static void Terminate();
  static void DebuggerInitialize(Debugger &debugger);

  static llvm::StringRef GetPluginNameStatic() { return "symstore"; }
  static llvm::StringRef GetPluginDescriptionStatic();

  static lldb_private::SymbolLocator *CreateInstance();

  /// PluginInterface protocol.
  /// \{
  llvm::StringRef GetPluginName() override { return GetPluginNameStatic(); }
  /// \}

  // Locate the symbol file given a module specification.
  //
  // Locating the file should happen only on the local computer or using the
  // current computers global settings.
  static std::optional<FileSpec>
  LocateExecutableSymbolFile(const ModuleSpec &module_spec,
                             const FileSpecList &default_search_paths);

  struct LookupEntry {
    std::string source;
    std::optional<std::string> cache;
  };

  static std::vector<LookupEntry> ParseEnvSymbolPaths(llvm::StringRef val);
  static std::string GetSystemDefaultCachePath();
};

} // namespace lldb_private

#endif // LLDB_SOURCE_PLUGINS_SYMBOLLOCATOR_SYMSTORE_SYMBOLLOCATORSYMSTORE_H

//===-- SymbolLocatorDebugSymbols.h -----------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLDB_SOURCE_PLUGINS_SYMBOLLOCATOR_DEBUGSYMBOLS_SYMBOLLOCATORDEBUGSYMBOLS_H
#define LLDB_SOURCE_PLUGINS_SYMBOLLOCATOR_DEBUGSYMBOLS_SYMBOLLOCATORDEBUGSYMBOLS_H

#include "lldb/Symbol/SymbolLocator.h"
#include "lldb/lldb-private.h"

namespace lldb_private {

class SymbolLocatorDebugSymbols : public SymbolLocator {
public:
  SymbolLocatorDebugSymbols();

  static void Initialize();
  static void Terminate();

  static llvm::StringRef GetPluginNameStatic() { return "DebugSymbols"; }
  static llvm::StringRef GetPluginDescriptionStatic();

  static lldb_private::SymbolLocator *CreateInstance();

  /// PluginInterface protocol.
  /// \{
  llvm::StringRef GetPluginName() override { return GetPluginNameStatic(); }
  /// \}

  // Locate the executable file given a module specification.
  //
  // Locating the file should happen only on the local computer or using the
  // current computers global settings.
  static std::optional<ModuleSpec>
  LocateExecutableObjectFile(const ModuleSpec &module_spec);
};

} // namespace lldb_private

#endif // LLDB_SOURCE_PLUGINS_SYMBOLLOCATOR_DEBUGSYMBOLS_SYMBOLLOCATORDEBUGSYMBOLS_H

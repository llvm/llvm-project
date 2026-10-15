//===-- RustLanguage.h ----------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLDB_SOURCE_PLUGINS_LANGUAGE_RUST_RUSTLANGUAGE_H
#define LLDB_SOURCE_PLUGINS_LANGUAGE_RUST_RUSTLANGUAGE_H

#include "lldb/Target/Language.h"
#include "lldb/lldb-private.h"

namespace lldb_private {

class RustLanguage : public Language {
public:
  lldb::LanguageType GetLanguageType() const override {
    return lldb::eLanguageTypeRust;
  }

  bool IsSourceFile(llvm::StringRef file_path) const override;

  DemangledName DemangleSymbol(ConstString mangled,
                               Mangled::ManglingScheme scheme,
                               const DemangleOptions &options) const override;

  // Static Functions
  static void Initialize();

  static void Terminate();

  static Language *CreateInstance(lldb::LanguageType language);

  static llvm::StringRef GetPluginNameStatic() { return "rust"; }

  // PluginInterface protocol
  llvm::StringRef GetPluginName() override { return GetPluginNameStatic(); }
};

} // namespace lldb_private

#endif // LLDB_SOURCE_PLUGINS_LANGUAGE_RUST_RUSTLANGUAGE_H

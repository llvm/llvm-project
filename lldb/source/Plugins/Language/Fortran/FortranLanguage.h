//===-- FortranLanguage.h -------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLDB_SOURCE_PLUGINS_LANGUAGE_FORTRAN_FORTRANLANGUAGE_H
#define LLDB_SOURCE_PLUGINS_LANGUAGE_FORTRAN_FORTRANLANGUAGE_H
#include "lldb/Target/Language.h"

#include "llvm/ADT/StringRef.h"

#include "lldb/Target/Language.h"
#include "lldb/Utility/ConstString.h"
#include "lldb/lldb-private.h"

namespace lldb_private {

class FortranLanguage : public Language {
public:
  FortranLanguage() = default;

  ~FortranLanguage() override = default;

  lldb::LanguageType GetLanguageType() const override {
    return lldb::eLanguageTypeFortran90;
  }
  //------------------------------------------------------------------
  // Static Functions
  //------------------------------------------------------------------
  static void Initialize();

  static void Terminate();

  static lldb_private::Language *CreateInstance(lldb::LanguageType language);

  static llvm::StringRef GetPluginNameStatic();

  //------------------------------------------------------------------
  // PluginInterface protocol
  //------------------------------------------------------------------
  llvm::StringRef GetPluginName() override;

  uint32_t GetPluginVersion();

  bool IsSourceFile(llvm::StringRef file_path) const override;
};

}; // namespace lldb_private

#endif // LLDB_SOURCE_PLUGINS_LANGUAGE_FORTRAN_FORTRANLANGUAGE_H
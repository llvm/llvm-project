//===-- GoLanguage.h --------------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLDB_SOURCE_PLUGINS_LANGUAGE_GO_GOLANGUAGE_H
#define LLDB_SOURCE_PLUGINS_LANGUAGE_GO_GOLANGUAGE_H

#include "lldb/Target/Language.h"
#include "lldb/lldb-private.h"

namespace lldb_private {

class GoLanguage : public Language {
public:
  GoLanguage() = default;
  ~GoLanguage() override = default;

  lldb::LanguageType GetLanguageType() const override {
    return lldb::eLanguageTypeGo;
  }

  llvm::StringRef GetUserEntryPointName() const override { return "main.main"; }

  HardcodedFormatters::HardcodedSummaryFinder GetHardcodedSummaries() override;

  bool IsSourceFile(llvm::StringRef file_path) const override;

  static void Initialize();
  static void Terminate();
  static Language *CreateInstance(lldb::LanguageType language);
  static llvm::StringRef GetPluginNameStatic() { return "go"; }

  llvm::StringRef GetPluginName() override { return GetPluginNameStatic(); }
};

} // namespace lldb_private

#endif // LLDB_SOURCE_PLUGINS_LANGUAGE_GO_GOLANGUAGE_H

//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This file implements the Fortran language Plugin.
///
//===----------------------------------------------------------------------===//

#include "llvm/ADT/StringRef.h"

#include "FortranLanguage.h"

#include "lldb/Core/PluginManager.h"

#include "Plugins/TypeSystem/Fortran/TypeSystemFortran.h"

using namespace llvm;
using namespace lldb;
using namespace lldb_private;
using namespace lldb_private::formatters;

LLDB_PLUGIN_DEFINE(FortranLanguage)

void FortranLanguage::Initialize() {
  PluginManager::RegisterPlugin(GetPluginNameStatic(), "Fortran Language",
                                CreateInstance);
}

void FortranLanguage::Terminate() {
  PluginManager::UnregisterPlugin(CreateInstance);
}

StringRef FortranLanguage::GetPluginNameStatic() {
  static llvm::StringRef g_name("fortran");
  return g_name;
}

//------------------------------------------------------------------
// PluginInterface protocol
//------------------------------------------------------------------
StringRef FortranLanguage::GetPluginName() { return GetPluginNameStatic(); }

uint32_t FortranLanguage::GetPluginVersion() { return 1; }

Language *FortranLanguage::CreateInstance(LanguageType language) {
  if (Language::LanguageIsFortran(language)) {
    return new FortranLanguage();
  }
  return nullptr;
}

bool FortranLanguage::IsSourceFile(StringRef file_path) const {
  const auto suffixes = {".f90", ".f", ".f95", ".f03", ".f08", ".f18"};
  for (auto suffix : suffixes) {
    if (file_path.ends_with_insensitive(suffix))
      return true;
  }
  return false;
}

//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "TypeSystemFortran.h"

#include "lldb/Core/PluginManager.h"
#include "lldb/Symbol/SymbolFile.h"
#include "lldb/Target/Language.h"
#include "lldb/Target/Target.h"

using namespace lldb;
using namespace lldb_private;
using namespace llvm;
using namespace lldb_private::plugin::dwarf;

LLDB_PLUGIN_DEFINE(TypeSystemFortran)

char TypeSystemFortran::ID;

TypeSystemFortran::~TypeSystemFortran() = default;
TypeSystemFortran::TypeSystemFortran() = default;

void TypeSystemFortran::Initialize() {
  PluginManager::RegisterPlugin(GetPluginNameStatic(), "fortran plug-in",
                                CreateInstance, GetSupportedLanguagesForTypes(),
                                GetSupportedLanguagesForExpressions());
}

void TypeSystemFortran::Terminate() {
  PluginManager::UnregisterPlugin(CreateInstance);
}

TypeSystemSP TypeSystemFortran::CreateInstance(LanguageType language,
                                               Module *module, Target *target) {
  if (Language::LanguageIsFortran(language))
    return std::make_shared<TypeSystemFortran>();

  return TypeSystemSP();
}

LanguageSet TypeSystemFortran::GetSupportedLanguagesForTypes() {
  LanguageSet languages;
  languages.Insert(eLanguageTypeFortran77);
  languages.Insert(eLanguageTypeFortran90);
  languages.Insert(eLanguageTypeFortran95);
  languages.Insert(eLanguageTypeFortran03);
  languages.Insert(eLanguageTypeFortran08);
  languages.Insert(eLanguageTypeFortran18);
  return languages;
}

LanguageSet TypeSystemFortran::GetSupportedLanguagesForExpressions() {
  return GetSupportedLanguagesForTypes();
}

bool TypeSystemFortran::SupportsLanguage(lldb::LanguageType language) {
  return Language::LanguageIsFortran(language);
}

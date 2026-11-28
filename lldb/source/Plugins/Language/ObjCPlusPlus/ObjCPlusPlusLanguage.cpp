//===-- ObjCPlusPlusLanguage.cpp ------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "ObjCPlusPlusLanguage.h"

#include "lldb/Core/PluginManager.h"
#include "lldb/Utility/ConstString.h"

using namespace lldb;
using namespace lldb_private;

LLDB_PLUGIN_DEFINE(ObjCPlusPlusLanguage)

bool ObjCPlusPlusLanguage::IsSourceFile(llvm::StringRef file_path) const {
  const auto suffixes = {".h", ".mm"};
  for (auto suffix : suffixes) {
    if (file_path.ends_with_insensitive(suffix))
      return true;
  }
  return false;
}

bool ObjCPlusPlusLanguage::IsCategoryApplicable(
    lldb::LanguageType category_language,
    lldb::LanguageType valobj_language) const {
  if (category_language != lldb::eLanguageTypeObjC_plus_plus)
    return Language::IsCategoryApplicable(category_language, valobj_language);
  return Language::LanguageIsC(valobj_language) ||
         valobj_language == lldb::eLanguageTypeC_plus_plus ||
         valobj_language == lldb::eLanguageTypeObjC ||
         valobj_language == lldb::eLanguageTypeObjC_plus_plus;
}

void ObjCPlusPlusLanguage::Initialize() {
  PluginManager::RegisterPlugin(GetPluginNameStatic(), "Objective-C++ Language",
                                CreateInstance);
}

void ObjCPlusPlusLanguage::Terminate() {
  PluginManager::UnregisterPlugin(CreateInstance);
}

// Static Functions
Language *ObjCPlusPlusLanguage::CreateInstance(lldb::LanguageType language) {
  switch (language) {
  case lldb::eLanguageTypeObjC_plus_plus:
    return new ObjCPlusPlusLanguage();
  default:
    return nullptr;
  }
}

std::optional<bool>
ObjCPlusPlusLanguage::GetBooleanFromString(llvm::StringRef str) const {
  return llvm::StringSwitch<std::optional<bool>>(str)
      .Cases({"true", "YES"}, {true})
      .Cases({"false", "NO"}, {false})
      .Default({});
}

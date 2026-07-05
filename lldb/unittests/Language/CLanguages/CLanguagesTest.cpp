//===-- CLanguagesTest.cpp ------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "Plugins/Language/CPlusPlus/CPlusPlusLanguage.h"
#include "Plugins/Language/ObjC/ObjCLanguage.h"
#include "Plugins/Language/ObjCPlusPlus/ObjCPlusPlusLanguage.h"
#include "TestingSupport/SubsystemRAII.h"
#include "lldb/lldb-enumerations.h"

#include "gmock/gmock.h"
#include "gtest/gtest.h"

using namespace lldb_private;

/// Returns the name of the LLDB plugin for the given language or an empty
/// string if there is no fitting plugin.
static llvm::StringRef GetPluginName(lldb::LanguageType language) {
  Language *language_plugin = Language::FindPlugin(language);
  if (language_plugin)
    return language_plugin->GetPluginName();
  return "";
}

TEST(CLanguages, LookupCLanguagesByLanguageType) {
  SubsystemRAII<CPlusPlusLanguage, ObjCPlusPlusLanguage, ObjCLanguage> langs;

  // There is no plugin to find for C.
  EXPECT_EQ(Language::FindPlugin(lldb::eLanguageTypeC), nullptr);
  EXPECT_EQ(Language::FindPlugin(lldb::eLanguageTypeC89), nullptr);
  EXPECT_EQ(Language::FindPlugin(lldb::eLanguageTypeC99), nullptr);
  EXPECT_EQ(Language::FindPlugin(lldb::eLanguageTypeC11), nullptr);

  EXPECT_EQ(GetPluginName(lldb::eLanguageTypeC_plus_plus), "cplusplus");
  EXPECT_EQ(GetPluginName(lldb::eLanguageTypeC_plus_plus_03), "cplusplus");
  EXPECT_EQ(GetPluginName(lldb::eLanguageTypeC_plus_plus_11), "cplusplus");
  EXPECT_EQ(GetPluginName(lldb::eLanguageTypeC_plus_plus_14), "cplusplus");

  EXPECT_EQ(GetPluginName(lldb::eLanguageTypeObjC), "objc");

  EXPECT_EQ(GetPluginName(lldb::eLanguageTypeObjC_plus_plus), "objcplusplus");
}

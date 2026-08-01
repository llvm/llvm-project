//===-- FortranLanguagesTest.cpp ------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "Plugins/Language/Fortran/FortranLanguage.h"
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

TEST(FortranLanguage, LookupFortranLanguageByLanguageType) {
  SubsystemRAII<FortranLanguage> langs;

  EXPECT_EQ(GetPluginName(lldb::eLanguageTypeFortran77), "fortran");
  EXPECT_EQ(GetPluginName(lldb::eLanguageTypeFortran90), "fortran");
  EXPECT_EQ(GetPluginName(lldb::eLanguageTypeFortran95), "fortran");
  EXPECT_EQ(GetPluginName(lldb::eLanguageTypeFortran03), "fortran");
  EXPECT_EQ(GetPluginName(lldb::eLanguageTypeFortran08), "fortran");
  EXPECT_EQ(GetPluginName(lldb::eLanguageTypeFortran18), "fortran");
}

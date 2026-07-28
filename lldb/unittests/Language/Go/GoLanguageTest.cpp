//===-- GoLanguageTest.cpp ------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "Plugins/Language/Go/GoLanguage.h"
#include "TestingSupport/SubsystemRAII.h"
#include "lldb/lldb-enumerations.h"

#include "gtest/gtest.h"

using namespace lldb_private;

TEST(GoLanguage, LookupByLanguageType) {
  SubsystemRAII<GoLanguage> language;

  Language *plugin = Language::FindPlugin(lldb::eLanguageTypeGo);
  ASSERT_NE(plugin, nullptr);
  EXPECT_EQ(plugin->GetPluginName(), "go");
  EXPECT_EQ(plugin->GetLanguageType(), lldb::eLanguageTypeGo);
  EXPECT_EQ(plugin->GetUserEntryPointName(), "main.main");
  EXPECT_EQ(plugin->GetHardcodedSummaries().size(), 1u);

  EXPECT_EQ(GoLanguage::CreateInstance(lldb::eLanguageTypeC), nullptr);
}

TEST(GoLanguage, RecognizesGoSourceFiles) {
  GoLanguage language;

  EXPECT_TRUE(language.IsSourceFile("main.go"));
  EXPECT_TRUE(language.IsSourceFile("/tmp/UPPER.GO"));
  EXPECT_FALSE(language.IsSourceFile("main.c"));
  EXPECT_FALSE(language.IsSourceFile("go.mod"));
}

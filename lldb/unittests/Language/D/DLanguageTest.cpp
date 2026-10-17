//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "Plugins/Language/D/DLanguage.h"
#include "TestingSupport/SubsystemRAII.h"
#include "lldb/Core/Mangled.h"
#include "lldb/lldb-enumerations.h"

#include "gtest/gtest.h"

using namespace lldb_private;

TEST(DLanguage, LookupDLanguageByLanguageType) {
  SubsystemRAII<DLanguage> langs;

  Language *lang = Language::FindPlugin(lldb::eLanguageTypeD);
  ASSERT_NE(lang, nullptr);
  EXPECT_EQ(lang->GetPluginName(), "d");
  EXPECT_TRUE(lang->IsSourceFile("main.d"));
  EXPECT_TRUE(lang->IsSourceFile("module.di"));
  EXPECT_FALSE(lang->IsSourceFile("main.cpp"));
}

TEST(DLanguage, RegistersDManglingScheme) {
  SubsystemRAII<DLanguage> langs;

  EXPECT_EQ(Mangled::GetLanguageForManglingScheme(
                Mangled::GetManglingScheme("_D3foo3barFZv")),
            lldb::eLanguageTypeD);
  EXPECT_EQ(Mangled::GetLanguageForManglingScheme(
                Mangled::GetManglingScheme("_Dmain")),
            lldb::eLanguageTypeD);
  EXPECT_EQ(Mangled::GetManglingScheme("_DDD"), Mangled::eManglingSchemeNone);
  EXPECT_EQ(Mangled::GetManglingScheme("_Dmainx"),
            Mangled::eManglingSchemeNone);
  EXPECT_EQ(Mangled::GetManglingScheme("_D"), Mangled::eManglingSchemeNone);
  EXPECT_EQ(Mangled("_Dmain").GetDemangledName(), ConstString("D main"));
}

//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "Plugins/Language/Rust/RustLanguage.h"
#include "TestingSupport/SubsystemRAII.h"
#include "lldb/Core/Mangled.h"
#include "lldb/lldb-enumerations.h"

#include "gtest/gtest.h"

using namespace lldb_private;

TEST(RustLanguage, LookupRustLanguageByLanguageType) {
  SubsystemRAII<RustLanguage> langs;

  Language *lang = Language::FindPlugin(lldb::eLanguageTypeRust);
  ASSERT_NE(lang, nullptr);
  EXPECT_EQ(lang->GetPluginName(), "rust");
  EXPECT_TRUE(lang->IsSourceFile("main.rs"));
  EXPECT_FALSE(lang->IsSourceFile("main.cpp"));
}

TEST(RustLanguage, RegistersRustV0ManglingScheme) {
  SubsystemRAII<RustLanguage> langs;

  EXPECT_EQ(Mangled::GetLanguageForManglingScheme(
                Mangled::GetManglingScheme("_RNvC1a4main")),
            lldb::eLanguageTypeRust);
  EXPECT_EQ(Mangled("_RNvC1a4main").GetDemangledName(), ConstString("a::main"));
}

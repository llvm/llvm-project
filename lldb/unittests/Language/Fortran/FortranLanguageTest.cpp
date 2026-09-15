//===----------------------------------------------------------------------===//
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

TEST(FortranLanguage, LookupFortranLanguageByLanguageType) {
  SubsystemRAII<FortranLanguage> langs;

  const auto types = {
      lldb::eLanguageTypeFortran77, lldb::eLanguageTypeFortran90,
      lldb::eLanguageTypeFortran95, lldb::eLanguageTypeFortran03,
      lldb::eLanguageTypeFortran08, lldb::eLanguageTypeFortran18};

  for (lldb::LanguageType lang_type : types) {
    Language *lang = Language::FindPlugin(lang_type);
    ASSERT_NE(lang, nullptr);
    EXPECT_EQ(lang->GetPluginName(), "fortran");
  }
}

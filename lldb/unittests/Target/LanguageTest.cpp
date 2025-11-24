//===-- LanguageTest.cpp --------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "lldb/Target/Language.h"
#include "lldb/lldb-enumerations.h"
#include "gtest/gtest.h"

using namespace lldb_private;
using namespace lldb;

namespace {
class LanguageTest : public ::testing::Test {};
} // namespace

TEST_F(LanguageTest, SourceLanguage_GetDescription) {
  for (uint32_t i = 1; i < lldb::eNumLanguageTypes; ++i) {
    // 0x29 is unassigned
    if (i == 0x29)
      continue;

    auto lang_type = static_cast<lldb::LanguageType>(i);
    SourceLanguage lang(lang_type);

    // eLanguageTypeHIP is not implemented as a DW_LNAME because of a conflict.
    if (lang_type == lldb::eLanguageTypeHIP)
      EXPECT_FALSE(lang);
    else
      EXPECT_TRUE(lang);
  }

  EXPECT_EQ(SourceLanguage(eLanguageTypeC_plus_plus).GetDescription(),
            "ISO C++");
  EXPECT_EQ(SourceLanguage(eLanguageTypeC_plus_plus_17).GetDescription(),
            "C++17");
  EXPECT_EQ(SourceLanguage(eLanguageTypeC_plus_plus_20).GetDescription(),
            "C++20");

  EXPECT_EQ(SourceLanguage(eLanguageTypeC).GetDescription(), "C (K&R and ISO)");
  EXPECT_EQ(SourceLanguage(eLanguageTypeC89).GetDescription(), "C89");

  EXPECT_EQ(SourceLanguage(eLanguageTypeObjC).GetDescription(), "Objective C");
  EXPECT_EQ(SourceLanguage(eLanguageTypeMipsAssembler).GetDescription(),
            "Assembly");

  auto next_vendor_language =
      static_cast<lldb::LanguageType>(eLanguageTypeMipsAssembler + 1);
  if (next_vendor_language < eNumLanguageTypes)
    EXPECT_NE(SourceLanguage(next_vendor_language).GetDescription(), "Unknown");

  EXPECT_EQ(SourceLanguage(eLanguageTypeUnknown).GetDescription(), "Unknown");
}

TEST_F(LanguageTest, SourceLanguage_AsLanguageType) {
  EXPECT_EQ(SourceLanguage(eLanguageTypeC_plus_plus).AsLanguageType(),
            eLanguageTypeC_plus_plus);
  EXPECT_EQ(SourceLanguage(eLanguageTypeC_plus_plus_03).AsLanguageType(),
            eLanguageTypeC_plus_plus_03);

  // Vendor-specific language code.
  EXPECT_EQ(SourceLanguage(eLanguageTypeMipsAssembler).AsLanguageType(),
            eLanguageTypeAssembly);
  EXPECT_EQ(SourceLanguage(eLanguageTypeUnknown).AsLanguageType(),
            eLanguageTypeUnknown);
}

TEST_F(LanguageTest, SourceLanguage_LastStandardLanguage) {
  // eLanguageTypeLastStandardLanguage should be treated as a standard DWARF
  // language.
  SourceLanguage lang(eLanguageTypeLastStandardLanguage);
  EXPECT_TRUE(lang);

  // It should have a valid description (not "Unknown").
  EXPECT_NE(lang.GetDescription(), "Unknown");

  // It should convert to the correct language type.
  EXPECT_EQ(lang.AsLanguageType(), eLanguageTypeLastStandardLanguage);
}

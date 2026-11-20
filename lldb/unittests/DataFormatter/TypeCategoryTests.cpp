//===-- TypeCategoryTests.cpp ---------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "lldb/Core/PluginManager.h"
#include "lldb/DataFormatters/FormatManager.h"
#include "lldb/DataFormatters/TypeSummary.h"
#include "lldb/Target/Language.h"
#include "llvm/ADT/ScopeExit.h"

#include "gtest/gtest.h"

using namespace lldb;
using namespace lldb_private;

namespace {
/// A language plugin for Ada 83 whose categories also apply to Ada 95 values.
class MockLanguage : public Language {
public:
  static Language *CreateInstance(LanguageType language) {
    if (language == eLanguageTypeAda83)
      return new MockLanguage();
    return nullptr;
  }

  LanguageType GetLanguageType() const override { return eLanguageTypeAda83; }
  bool IsSourceFile(llvm::StringRef file_path) const override { return false; }
  llvm::StringRef GetPluginName() override { return "mock-language"; }

  bool IsCategoryApplicable(LanguageType category_language,
                            LanguageType valobj_language) const override {
    return valobj_language == eLanguageTypeAda95 ||
           Language::IsCategoryApplicable(category_language, valobj_language);
  }
};
} // namespace

/// Returns whether \p category provides its summary for a type named "T" to a
/// value of \p language.
static bool AppliesTo(TypeCategoryImpl &category, LanguageType language) {
  FormattersMatchVector candidates = {
      {"T", nullptr, TypeImpl(), FormattersMatchCandidate::Flags()}};
  TypeSummaryImplSP summary;
  return category.Get(language, candidates, summary);
}

TEST(TypeCategoryTests, IsApplicable) {
  ASSERT_TRUE(PluginManager::RegisterPlugin("mock-language", "",
                                            MockLanguage::CreateInstance));
  llvm::scope_exit cleanup{
      [] { PluginManager::UnregisterPlugin(MockLanguage::CreateInstance); }};

  FormatManager format_manager;
  auto make_category = [&](llvm::StringRef name, LanguageType language) {
    TypeCategoryImplSP category = format_manager.GetCategory(name);
    category->AddTypeSummary("T", eFormatterMatchExact,
                             std::make_shared<StringSummaryFormat>(
                                 TypeSummaryImpl::Flags(), "summary"));
    format_manager.EnableCategory(name, TypeCategoryMap::Default, language);
    return category;
  };

  // A category without a language applies to values of all languages.
  TypeCategoryImplSP any = make_category("any", eLanguageTypeUnknown);
  EXPECT_TRUE(AppliesTo(*any, eLanguageTypeAda95));

  // The plugin of the category's language decides.
  TypeCategoryImplSP ada = make_category("ada", eLanguageTypeAda83);
  EXPECT_TRUE(AppliesTo(*ada, eLanguageTypeAda83));
  EXPECT_TRUE(AppliesTo(*ada, eLanguageTypeAda95));
  EXPECT_FALSE(AppliesTo(*ada, eLanguageTypeCobol74));

  // No plugin claims C, whose versions count as one language.
  TypeCategoryImplSP c = make_category("c", eLanguageTypeC);
  EXPECT_TRUE(AppliesTo(*c, eLanguageTypeC99));
  EXPECT_FALSE(AppliesTo(*c, eLanguageTypeC_plus_plus));

  // Without a plugin, a category only applies to values of its language.
  TypeCategoryImplSP fortran = make_category("fortran", eLanguageTypeFortran77);
  EXPECT_TRUE(AppliesTo(*fortran, eLanguageTypeFortran77));
  EXPECT_FALSE(AppliesTo(*fortran, eLanguageTypeFortran90));
}

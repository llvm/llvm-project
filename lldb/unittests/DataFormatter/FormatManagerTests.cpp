//===-- FormatManagerTests.cpp --------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "lldb/Core/PluginManager.h"
#include "lldb/DataFormatters/FormatManager.h"
#include "lldb/Target/Language.h"
#include "llvm/ADT/ScopeExit.h"

#include "gtest/gtest.h"

using namespace lldb;
using namespace lldb_private;

namespace {
/// A language plugin for Ada 83 whose formatters also apply to Ada 95 values.
class MockAda83Language : public Language {
public:
  static Language *CreateInstance(LanguageType language) {
    if (language == eLanguageTypeAda83)
      return new MockAda83Language();
    return nullptr;
  }

  LanguageType GetLanguageType() const override { return eLanguageTypeAda83; }
  bool IsSourceFile(llvm::StringRef file_path) const override { return false; }
  llvm::StringRef GetPluginName() override { return "mock-ada83"; }

  std::vector<LanguageType>
  GetFormatterLanguages(LanguageType language) const override {
    if (language == eLanguageTypeAda95)
      return {eLanguageTypeAda83};
    return Language::GetFormatterLanguages(language);
  }
};

/// A language plugin for COBOL 74 that lets its own formatters and those of
/// Ada 83 apply to Ada 95 values.
class MockCobol74Language : public Language {
public:
  static Language *CreateInstance(LanguageType language) {
    if (language == eLanguageTypeCobol74)
      return new MockCobol74Language();
    return nullptr;
  }

  LanguageType GetLanguageType() const override { return eLanguageTypeCobol74; }
  bool IsSourceFile(llvm::StringRef file_path) const override { return false; }
  llvm::StringRef GetPluginName() override { return "mock-cobol74"; }

  std::vector<LanguageType>
  GetFormatterLanguages(LanguageType language) const override {
    if (language == eLanguageTypeAda95)
      return {eLanguageTypeCobol74, eLanguageTypeAda83};
    return Language::GetFormatterLanguages(language);
  }
};
} // namespace

TEST(FormatManagerTests, CandidateLanguages) {
  ASSERT_TRUE(PluginManager::RegisterPlugin("mock-ada83", "",
                                            MockAda83Language::CreateInstance));
  ASSERT_TRUE(PluginManager::RegisterPlugin(
      "mock-cobol74", "", MockCobol74Language::CreateInstance));
  llvm::scope_exit cleanup{[] {
    PluginManager::UnregisterPlugin(MockAda83Language::CreateInstance);
    PluginManager::UnregisterPlugin(MockCobol74Language::CreateInstance);
  }};

  // By default, a language's formatters apply to its own values.
  std::vector<LanguageType> candidates = {eLanguageTypeAda83};
  EXPECT_EQ(FormatManager::GetCandidateLanguages(eLanguageTypeAda83),
            candidates);

  // The languages come in the order in which Language::ForEach() visits the
  // plugins, and each one only once.
  candidates = {eLanguageTypeAda83, eLanguageTypeCobol74};
  EXPECT_EQ(FormatManager::GetCandidateLanguages(eLanguageTypeAda95),
            candidates);

  // No formatters apply to values of a language that no plugin claims.
  EXPECT_TRUE(
      FormatManager::GetCandidateLanguages(eLanguageTypeFortran77).empty());
}

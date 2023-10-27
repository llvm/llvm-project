//===-- ObjCLanguageTest.cpp ----------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#include "Plugins/Language/ObjC/ObjCLanguage.h"
#include "lldb/lldb-enumerations.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include <optional>

#include "llvm/ADT/StringRef.h"

using namespace lldb_private;

TEST(ObjCLanguage, MethodNameParsing) {
  struct TestCase {
    llvm::StringRef input;
    llvm::StringRef full_name_sans_category;
    llvm::StringRef class_name;
    llvm::StringRef class_name_with_category;
    llvm::StringRef category;
    llvm::StringRef selector;
  };

  TestCase strict_cases[] = {
      {"-[MyClass mySelector:]", "", "MyClass", "MyClass", "", "mySelector:"},
      {"+[MyClass mySelector:]", "", "MyClass", "MyClass", "", "mySelector:"},
      {"-[MyClass(my_category) mySelector:]", "-[MyClass mySelector:]",
       "MyClass", "MyClass(my_category)", "my_category", "mySelector:"},
      {"+[MyClass(my_category) mySelector:]", "+[MyClass mySelector:]",
       "MyClass", "MyClass(my_category)", "my_category", "mySelector:"},
  };

  TestCase lax_cases[] = {
      {"[MyClass mySelector:]", "", "MyClass", "MyClass", "", "mySelector:"},
      {"[MyClass(my_category) mySelector:]", "[MyClass mySelector:]", "MyClass",
       "MyClass(my_category)", "my_category", "mySelector:"},
  };

  // First, be strict
  for (const auto &test : strict_cases) {
    std::optional<const ObjCLanguage::MethodName> method =
        ObjCLanguage::MethodName::Create(test.input, /*strict = */ true);
    EXPECT_TRUE(method.has_value());
    EXPECT_EQ(test.full_name_sans_category,
              method->GetFullNameWithoutCategory());
    EXPECT_EQ(test.class_name, method->GetClassName());
    EXPECT_EQ(test.class_name_with_category,
              method->GetClassNameWithCategory());
    EXPECT_EQ(test.category, method->GetCategory());
    EXPECT_EQ(test.selector, method->GetSelector());
  }

  // We should make sure strict parsing does not accept lax cases
  for (const auto &test : lax_cases) {
    std::optional<const ObjCLanguage::MethodName> method =
        ObjCLanguage::MethodName::Create(test.input, /*strict = */ true);
    EXPECT_FALSE(method.has_value());
  }

  // All strict cases should work when not lax
  for (const auto &test : strict_cases) {
    std::optional<const ObjCLanguage::MethodName> method =
        ObjCLanguage::MethodName::Create(test.input, /*strict = */ false);
    EXPECT_TRUE(method.has_value());
    EXPECT_EQ(test.full_name_sans_category,
              method->GetFullNameWithoutCategory());
    EXPECT_EQ(test.class_name, method->GetClassName());
    EXPECT_EQ(test.class_name_with_category,
              method->GetClassNameWithCategory());
    EXPECT_EQ(test.category, method->GetCategory());
    EXPECT_EQ(test.selector, method->GetSelector());
  }

  // Make sure non-strict parsing works
  for (const auto &test : lax_cases) {
    std::optional<const ObjCLanguage::MethodName> method =
        ObjCLanguage::MethodName::Create(test.input, /*strict = */ false);
    EXPECT_TRUE(method.has_value());
    EXPECT_EQ(test.full_name_sans_category,
              method->GetFullNameWithoutCategory());
    EXPECT_EQ(test.class_name, method->GetClassName());
    EXPECT_EQ(test.class_name_with_category,
              method->GetClassNameWithCategory());
    EXPECT_EQ(test.category, method->GetCategory());
    EXPECT_EQ(test.selector, method->GetSelector());
  }
}

TEST(ObjCLanguage, InvalidMethodNameParsing) {
  // Tests that we correctly reject malformed function names

  llvm::StringRef test_cases[] = {"+[Uh oh!",
                                  "-[Definitely not...",
                                  "[Nice try ] :)",
                                  "+MaybeIfYouSquintYourEyes]",
                                  "?[Tricky]",
                                  "+[]",
                                  "-[]",
                                  "[]"};

  for (const auto &name : test_cases) {
    std::optional<const ObjCLanguage::MethodName> strict_method =
        ObjCLanguage::MethodName::Create(name, /*strict = */ false);
    EXPECT_FALSE(strict_method.has_value());

    std::optional<const ObjCLanguage::MethodName> lax_method =
        ObjCLanguage::MethodName::Create(name, /*strict = */ false);
    EXPECT_FALSE(lax_method.has_value());
  }
}

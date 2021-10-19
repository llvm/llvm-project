//===- llvm/unittest/StringViewTest.cpp - StringView unit tests -----------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/Demangle/StringView.h"
#include "gtest/gtest.h"

using namespace llvm;
using llvm::itanium_demangle::StringView;

namespace llvm {
namespace itanium_demangle {

std::ostream &operator<<(std::ostream &OS, const StringView &S) {
  return OS.write(S.begin(), S.size());
}

} // namespace itanium_demangle
} // namespace llvm

TEST(StringViewTest, EmptyInitializerList) {
  StringView S = {};
  EXPECT_TRUE(S.empty());

  S = {};
  EXPECT_TRUE(S.empty());
}

TEST(StringViewTest, Substr) {
  StringView S("abcdef");

  EXPECT_EQ("abcdef", S.substr(0));
  EXPECT_EQ("f", S.substr(5));
  EXPECT_EQ("", S.substr(6));

  EXPECT_EQ("", S.substr(0, 0));
  EXPECT_EQ("a", S.substr(0, 1));
  EXPECT_EQ("abcde", S.substr(0, 5));
  EXPECT_EQ("abcdef", S.substr(0, 6));
  EXPECT_EQ("abcdef", S.substr(0, 7));

  EXPECT_EQ("f", S.substr(5, 100));
  EXPECT_EQ("", S.substr(6, 100));
}

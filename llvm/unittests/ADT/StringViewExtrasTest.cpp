//===- StringExtrasTest.cpp - Unit tests for String extras ----------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/ADT/StringViewExtras.h"
#include "llvm/Support/raw_ostream.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"

#include <string_view>

using namespace llvm;

TEST(StringViewExtrasTest, starts_with) {
  std::string haystack = "hello world";
  EXPECT_TRUE(llvm::starts_with(haystack, 'h'));
  EXPECT_FALSE(llvm::starts_with(haystack, '\0'));
  EXPECT_TRUE(llvm::starts_with(haystack, "hello"));
  // TODO: should this differ from \0?
  EXPECT_TRUE(llvm::starts_with(haystack, ""));

  std::string empty;
  EXPECT_FALSE(llvm::starts_with(empty, 'h'));
  EXPECT_FALSE(llvm::starts_with(empty, '\0'));
  EXPECT_FALSE(llvm::starts_with(empty, "hello"));
  // TODO: should this differ from \0?
  EXPECT_TRUE(llvm::starts_with(empty, ""));
}

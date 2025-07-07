//===- llvm/unittest/ADT/StringTableTest.cpp - StringTable tests ----------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/ADT/StringTable.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include <cstdlib>

using namespace llvm;

namespace {

using ::testing::Eq;
using ::testing::StrEq;

TEST(StringTableTest, Basic) {
  static constexpr char InputTable[] = "\0test\0";
  constexpr StringTable T = InputTable;

  // We support some limited constexpr operations, check those first.
  static_assert(T.size() == sizeof(InputTable));
  static_assert(T[0].empty());
  static_assert(T[StringTable::Offset()].empty());
  static_assert(T[1].size() == 4);

  // And use normal Google Test runtime assertions to check the contents and
  // give more complete error messages.
  EXPECT_THAT(T[0], Eq(""));
  EXPECT_THAT(T[StringTable::Offset()], Eq(""));
  EXPECT_THAT(T[1], Eq("test"));

  // Also check that this is a valid C-string.
  EXPECT_THAT(T[1].data(), StrEq("test"));
}

} // anonymous namespace

//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "lldb/DataFormatters/FormattersHelpers.h"

#include "gtest/gtest.h"

using namespace lldb;
using namespace lldb_private;

using namespace lldb_private::formatters;

TEST(FormattersHelpersTests, ExtractIndexFromString) {
  EXPECT_EQ(ExtractIndexFromString("[0]"), std::optional<size_t>(0));
  EXPECT_EQ(ExtractIndexFromString("[1]"), std::optional<size_t>(1));
  EXPECT_EQ(ExtractIndexFromString("[42]"), std::optional<size_t>(42));
  EXPECT_EQ(ExtractIndexFromString("[1234567]"), std::optional<size_t>(1234567));

  // The base is auto-detected, so hex and octal are accepted.
  EXPECT_EQ(ExtractIndexFromString("[0x10]"), std::optional<size_t>(16));
  EXPECT_EQ(ExtractIndexFromString("[010]"), std::optional<size_t>(8));

  EXPECT_EQ(ExtractIndexFromString(llvm::StringRef()), std::nullopt);
  EXPECT_EQ(ExtractIndexFromString(""), std::nullopt);
  EXPECT_EQ(ExtractIndexFromString("[]"), std::nullopt);
  EXPECT_EQ(ExtractIndexFromString("["), std::nullopt);
  EXPECT_EQ(ExtractIndexFromString("[abc]"), std::nullopt);
  EXPECT_EQ(ExtractIndexFromString("42"), std::nullopt);
  EXPECT_EQ(ExtractIndexFromString("42]"), std::nullopt);
  EXPECT_EQ(ExtractIndexFromString("a[1]"), std::nullopt);
  EXPECT_EQ(ExtractIndexFromString("[42"), std::nullopt);
  EXPECT_EQ(ExtractIndexFromString("[99999999999999999999999999]"),
            std::nullopt);
  EXPECT_EQ(ExtractIndexFromString("[-1]"), std::nullopt);
  EXPECT_EQ(ExtractIndexFromString("[-2]"), std::nullopt);
}

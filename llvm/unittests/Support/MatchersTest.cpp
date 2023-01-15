//===----- unittests/MatchersTest.cpp -------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/Testing/Support/SupportHelpers.h"
#include "gmock/gmock-matchers.h"

using ::testing::_;
using ::testing::AllOf;
using ::testing::Gt;
using ::testing::Lt;
using ::testing::Not;

namespace {
TEST(MatchersTest, Optional) {
  EXPECT_THAT(std::optional<int>(std::nullopt), Not(llvm::ValueIs(_)));
  EXPECT_THAT(std::optional<int>(10), llvm::ValueIs(10));
  EXPECT_THAT(std::optional<int>(10), llvm::ValueIs(AllOf(Lt(11), Gt(9))));
}
} // namespace

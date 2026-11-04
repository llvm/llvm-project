//===-- DAPErrorTest.cpp---------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "DAPError.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include <string>
#include <system_error>

using namespace lldb_dap;
using namespace llvm;

TEST(DAPErrorTest, DefaultConstructor) {
  DAPError error("Invalid thread");

  EXPECT_EQ(error.getMessage(), "Invalid thread");
  EXPECT_EQ(error.convertToErrorCode(), llvm::inconvertibleErrorCode());
  EXPECT_TRUE(error.getShowUser());
  EXPECT_EQ(error.getURL(), std::nullopt);
  EXPECT_EQ(error.getURLLabel(), std::nullopt);
}

TEST(DAPErrorTest, FullConstructor) {
  auto timed_out = std::make_error_code(std::errc::timed_out);
  DAPError error("Timed out", timed_out, false, "URL", "URLLabel");

  EXPECT_EQ(error.getMessage(), "Timed out");
  EXPECT_EQ(error.convertToErrorCode(), timed_out);
  EXPECT_FALSE(error.getShowUser());
  EXPECT_THAT(error.getURL(), testing::Optional<std::string>("URL"));
  EXPECT_THAT(error.getURLLabel(), testing::Optional<std::string>("URLLabel"));
}

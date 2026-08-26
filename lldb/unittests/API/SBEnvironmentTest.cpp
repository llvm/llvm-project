//===-- SBEnvironment.cpp -------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===/

#include "lldb/API/SBEnvironment.h"
#include "gtest/gtest.h"

TEST(SBEnvironmentTest, SetAndGetEnv) {

  lldb::SBEnvironment env{};

  // Setting an env var without a value does not crash.
  env.Set("FOO", nullptr, false);
  const char *foo_val = env.Get("FOO");
  EXPECT_STREQ(foo_val, "");

  env.Set("BAR", "BAR_VALUE", true);
  env.Set("BAR", nullptr, true);
  const char *bar_val = env.Get("BAR");
  EXPECT_STREQ(bar_val, "") << "'BAR' should return the most recent value";
}

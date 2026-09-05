//===-- SBEnvironment.cpp -------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===/

// Use the umbrella header for -Wdocumentation.
#include "lldb/API/LLDB.h"

#include "lldb/API/SBEnvironment.h"
#include "gtest/gtest.h"

TEST(SBEnvironmentTest, SetAndGetEnv) {

  lldb::SBEnvironment env{};

  // Setting an env var without a value does not crash.
  EXPECT_TRUE(env.Set("FOO", nullptr, false));
  const char *foo_val = env.Get("FOO");
  EXPECT_STREQ(foo_val, "");

  EXPECT_TRUE(env.Set("BAR", "BAR_VALUE", true));
  EXPECT_TRUE(env.Set("BAR", nullptr, true));
  const char *bar_val = env.Get("BAR");
  EXPECT_STREQ(bar_val, "") << "'BAR' should return the most recent value";

  EXPECT_FALSE(env.Set(nullptr, "VALUE", true));
  EXPECT_FALSE(env.Set("", "VALUE", true));
  EXPECT_FALSE(env.Set(" ", "VALUE", true));

  EXPECT_FALSE(env.Get(nullptr));
  EXPECT_FALSE(env.Get(""));
  EXPECT_FALSE(env.Get(" "));
}

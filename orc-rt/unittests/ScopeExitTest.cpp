//===- ScopeExitTest.cpp --------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Tests for orc-rt's ScopeExit.h APIs.
//
//===----------------------------------------------------------------------===//

#include "orc-rt/ScopeExit.h"
#include "gtest/gtest.h"

using namespace orc_rt;

TEST(ScopeExitTest, Noop) {
  auto _ = make_scope_exit([]() {});
}

TEST(ScopeExitTest, OnScopeExit) {
  bool ScopeExitRun = false;
  {
    auto _ = make_scope_exit([&]() { ScopeExitRun = true; });
    EXPECT_FALSE(ScopeExitRun);
  }
  EXPECT_TRUE(ScopeExitRun);
}

TEST(ScopeExitTest, Release) {
  bool ScopeExitRun = false;
  {
    auto OnExit = make_scope_exit([&]() { ScopeExitRun = true; });
    EXPECT_FALSE(ScopeExitRun);
    OnExit.release();
  }
  EXPECT_FALSE(ScopeExitRun);
}

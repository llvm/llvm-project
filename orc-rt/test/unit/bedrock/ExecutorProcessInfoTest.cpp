//===- ExecutorProcessInfoTest.cpp ----------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Tests for ExecutorProcessInfo APIs.
//
//===----------------------------------------------------------------------===//

#include "orc-rt/bedrock/ExecutorProcessInfo.h"
#include "orc-rt/support/bit.h"
#include "gtest/gtest.h"

#include <unistd.h>

using namespace orc_rt;

TEST(ExecutorProcessInfoTest, DetectSucceeds) {
  auto EPI = ExecutorProcessInfo::Detect();
  EXPECT_TRUE(!!EPI);
}

TEST(ExecutorProcessInfoTest, DetectPageSizeIsPowerOfTwo) {
  auto EPI = cantFail(ExecutorProcessInfo::Detect());
  EXPECT_GT(EPI.pageSize(), 0U);
  EXPECT_TRUE(has_single_bit(EPI.pageSize()));
}

TEST(ExecutorProcessInfoTest, DetectPageSizeAtLeast4096) {
  auto EPI = cantFail(ExecutorProcessInfo::Detect());
  EXPECT_GE(EPI.pageSize(), 4096U);
}

TEST(ExecutorProcessInfoTest, DetectPageSizeMatchesSysconf) {
  auto EPI = cantFail(ExecutorProcessInfo::Detect());
  EXPECT_EQ(EPI.pageSize(), static_cast<size_t>(sysconf(_SC_PAGESIZE)));
}

TEST(ExecutorProcessInfoTest, ConstructWithExplicitValues) {
  ExecutorProcessInfo EPI("x86_64-unknown-linux-gnu", 4096, "+x,+a,+b");
  EXPECT_EQ(EPI.targetTriple(), "x86_64-unknown-linux-gnu");
  EXPECT_EQ(EPI.pageSize(), 4096U);
  EXPECT_EQ(EPI.targetCPUFeatures(), "+x,+a,+b");
}

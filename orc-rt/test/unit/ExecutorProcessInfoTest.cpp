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

#include "orc-rt/ExecutorProcessInfo.h"
#include "orc-rt/Math.h"
#include "gtest/gtest.h"

#include <algorithm>
#include <unistd.h>

using namespace orc_rt;

TEST(ExecutorProcessInfoTest, DetectSucceeds) {
  auto EPI = ExecutorProcessInfo::Detect();
  EXPECT_TRUE(!!EPI);
}

TEST(ExecutorProcessInfoTest, DetectPageSizeIsPowerOfTwo) {
  auto EPI = cantFail(ExecutorProcessInfo::Detect());
  EXPECT_GT(EPI.pageSize(), 0U);
  EXPECT_TRUE(isPowerOf2(EPI.pageSize()));
}

TEST(ExecutorProcessInfoTest, DetectPageSizeAtLeast4096) {
  auto EPI = cantFail(ExecutorProcessInfo::Detect());
  EXPECT_GE(EPI.pageSize(), 4096U);
}

TEST(ExecutorProcessInfoTest, DetectPageSizeMatchesSysconf) {
  auto EPI = cantFail(ExecutorProcessInfo::Detect());
  EXPECT_EQ(EPI.pageSize(), static_cast<size_t>(sysconf(_SC_PAGESIZE)));
}

TEST(ExecutorProcessInfoTest, DetectTargetTripleNotEmpty) {
  auto EPI = cantFail(ExecutorProcessInfo::Detect());
  EXPECT_FALSE(EPI.targetTriple().empty());
}

TEST(ExecutorProcessInfoTest, DetectTargetTripleHasValidStructure) {
  auto EPI = cantFail(ExecutorProcessInfo::Detect());
  // A valid triple has 2 hyphens (arch-vendor-os) or 3 (arch-vendor-os-env).
  auto NumHyphens =
      std::count(EPI.targetTriple().begin(), EPI.targetTriple().end(), '-');
  EXPECT_GE(NumHyphens, 2);
  EXPECT_LE(NumHyphens, 3);
}

TEST(ExecutorProcessInfoTest, DetectTargetTripleArchMatchesCompileTarget) {
  auto EPI = cantFail(ExecutorProcessInfo::Detect());
#if defined(__x86_64__) || defined(_M_X64)
  EXPECT_EQ(EPI.targetTriple().substr(0, 6), "x86_64");
#elif defined(__aarch64__) || defined(_M_ARM64)
  EXPECT_EQ(EPI.targetTriple().substr(0, 7), "aarch64");
#endif
}

TEST(ExecutorProcessInfoTest, DetectTargetTripleOSMatchesCompileTarget) {
  auto EPI = cantFail(ExecutorProcessInfo::Detect());
#if defined(__APPLE__)
  EXPECT_NE(EPI.targetTriple().find("darwin"), std::string::npos);
#elif defined(__linux__)
  EXPECT_NE(EPI.targetTriple().find("linux"), std::string::npos);
#endif
}

TEST(ExecutorProcessInfoTest, ConstructWithExplicitValues) {
  ExecutorProcessInfo EPI("x86_64-unknown-linux-gnu", 4096);
  EXPECT_EQ(EPI.targetTriple(), "x86_64-unknown-linux-gnu");
  EXPECT_EQ(EPI.pageSize(), 4096U);
}

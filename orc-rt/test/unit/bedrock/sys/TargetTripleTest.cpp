//===- TargetTripleTest.cpp -----------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Tests for orc-rt's bedrock/sys/TargetTriple.h APIs.
//
//===----------------------------------------------------------------------===//

#include "orc-rt-internal/bedrock/sys/TargetTriple.h"
#include "gtest/gtest.h"

#include <algorithm>

using namespace orc_rt;

TEST(TargetTripleTest, NotEmpty) {
  EXPECT_FALSE(sys::detectTargetTriple().empty());
}

TEST(TargetTripleTest, HasValidStructure) {
  auto Triple = sys::detectTargetTriple();
  // A valid triple has 2 hyphens (arch-vendor-os) or 3 (arch-vendor-os-env).
  auto NumHyphens = std::count(Triple.begin(), Triple.end(), '-');
  EXPECT_GE(NumHyphens, 2);
  EXPECT_LE(NumHyphens, 3);
}

TEST(TargetTripleTest, ArchMatchesCompileTarget) {
  auto Triple = sys::detectTargetTriple();
#if defined(__arm64e__)
  EXPECT_EQ(Triple.substr(0, 7), "arm64e-");
#elif defined(__APPLE__) && (defined(__arm64__) || defined(__aarch64__))
  EXPECT_EQ(Triple.substr(0, 6), "arm64-");
#elif defined(__aarch64__) || defined(_M_ARM64)
  EXPECT_EQ(Triple.substr(0, 8), "aarch64-");
#elif defined(__x86_64h__)
  EXPECT_EQ(Triple.substr(0, 8), "x86_64h-");
#elif defined(__x86_64__) || defined(_M_X64)
  EXPECT_EQ(Triple.substr(0, 7), "x86_64-");
#endif
}

TEST(TargetTripleTest, OSMatchesCompileTarget) {
  auto Triple = sys::detectTargetTriple();
#if defined(__APPLE__)
  EXPECT_NE(Triple.find("-apple-"), std::string::npos);
#elif defined(__linux__)
  EXPECT_NE(Triple.find("-linux-"), std::string::npos);
#endif
}

TEST(TargetTripleTest, CachedResultIsIdempotent) {
  EXPECT_EQ(sys::detectTargetTriple(), sys::detectTargetTriple());
}

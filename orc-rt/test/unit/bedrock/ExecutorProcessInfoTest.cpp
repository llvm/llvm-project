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
#include "orc-rt-internal/bedrock/TargetDetails.h"
#include "orc-rt/support/Math.h"
#include "gtest/gtest.h"

#include <algorithm>
#include <unistd.h>

using namespace orc_rt;
using namespace orc_rt::target_detail;

namespace orc_rt {
/// Friend of ExecutorProcessInfo; forwards to the private helpers so the
/// tests below can exercise them without widening the public API.
struct ExecutorProcessInfoTestAccess {
  static std::vector<std::string_view> detectTargetCPUFeatures() {
    return ExecutorProcessInfo::detectTargetCPUFeatures();
  }
  static std::string
  formatCPUFeatures(const std::vector<std::string_view> &Features) {
    return ExecutorProcessInfo::formatCPUFeatures(Features);
  }
  static std::string
  makeTargetTriple(std::initializer_list<std::string_view> Components) {
    return ExecutorProcessInfo::makeTargetTriple(Components);
  }
};
} // namespace orc_rt

using TestAccess = orc_rt::ExecutorProcessInfoTestAccess;

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
#if defined(__arm64e__)
  EXPECT_EQ(EPI.targetTriple().substr(0, 7), "arm64e-");
#elif defined(__APPLE__) && (defined(__arm64__) || defined(__aarch64__))
  EXPECT_EQ(EPI.targetTriple().substr(0, 6), "arm64-");
#elif defined(__aarch64__) || defined(_M_ARM64)
  EXPECT_EQ(EPI.targetTriple().substr(0, 8), "aarch64-");
#elif defined(__x86_64h__)
  EXPECT_EQ(EPI.targetTriple().substr(0, 8), "x86_64h-");
#elif defined(__x86_64__) || defined(_M_X64)
  EXPECT_EQ(EPI.targetTriple().substr(0, 7), "x86_64-");
#endif
}

TEST(ExecutorProcessInfoTest, DetectTargetTripleOSMatchesCompileTarget) {
  auto EPI = cantFail(ExecutorProcessInfo::Detect());
#if defined(__APPLE__)
  EXPECT_NE(EPI.targetTriple().find("-apple-"), std::string::npos);
#elif defined(__linux__)
  EXPECT_NE(EPI.targetTriple().find("-linux-"), std::string::npos);
#endif
}

TEST(ExecutorProcessInfoTest, ConstructWithExplicitValues) {
  ExecutorProcessInfo EPI("x86_64-unknown-linux-gnu", 4096, "+x,+a,+b");
  EXPECT_EQ(EPI.targetTriple(), "x86_64-unknown-linux-gnu");
  EXPECT_EQ(EPI.pageSize(), 4096U);
  EXPECT_EQ(EPI.targetCPUFeatures(), "+x,+a,+b");
}

TEST(ExecutorProcessInfoTest, FormatEmptyFeatures) {
  std::vector<std::string_view> V;

  EXPECT_EQ(TestAccess::formatCPUFeatures(V), "");
}

TEST(ExecutorProcessInfoTest, FormatSingleFeature) {
  std::vector<std::string_view> V = {feature::x86::avx2};

  EXPECT_EQ(TestAccess::formatCPUFeatures(V), "+avx2");
}

TEST(ExecutorProcessInfoTest, FormatMultipleFeatures) {
  std::vector<std::string_view> V = {feature::aarch64::neon,
                                     feature::aarch64::dotprod,
                                     feature::aarch64::fullfp16};

  EXPECT_EQ(TestAccess::formatCPUFeatures(V), "+neon,+dotprod,+fullfp16");
}

TEST(ExecutorProcessInfoTest, DetectDoesNotCrash) {
  [[maybe_unused]] auto _ = TestAccess::detectTargetCPUFeatures();
  SUCCEED();
}

// this test should catch any issues that arise from out of order tests
// hopefully.
TEST(ExecutorProcessInfoTest, CachedCPUFeaturesResultIsIdempotent) {
  EXPECT_EQ(ExecutorProcessInfo::detectCPUFeatures(),
            ExecutorProcessInfo::detectCPUFeatures());
}

TEST(ExecutorProcessInfoTest, CachedStringMatchesFormatted) {
  std::string F =
      TestAccess::formatCPUFeatures(TestAccess::detectTargetCPUFeatures());
  std::string C = ExecutorProcessInfo::detectCPUFeatures();

  EXPECT_EQ(F, C);
}

TEST(ExecutorProcessInfoTest, MakeTargetTripleAllComponents) {
  EXPECT_EQ(
      TestAccess::makeTargetTriple({"arm64", "apple", "ios26.0", "macabi"}),
      "arm64-apple-ios26.0-macabi");
}

TEST(ExecutorProcessInfoTest, MakeTargetTripleDropsTrailingEmpty) {
  EXPECT_EQ(TestAccess::makeTargetTriple({"arm64", "apple", "macosx26.1", ""}),
            "arm64-apple-macosx26.1");
}

TEST(ExecutorProcessInfoTest, MakeTargetTripleDoubleDashMiddleEmpty) {
  EXPECT_EQ(TestAccess::makeTargetTriple({"x86_64", "", "linux", "gnu"}),
            "x86_64--linux-gnu");
  EXPECT_EQ(TestAccess::makeTargetTriple({"x86_64", "", "linux", ""}),
            "x86_64--linux");
}

TEST(ExecutorProcessInfoTest, MakeTargetTripleEmpty) {
  EXPECT_EQ(TestAccess::makeTargetTriple({}), "");
}

TEST(ExecutorProcessInfoTest, MakeTargetTripleExtraComponents) {
  EXPECT_EQ(TestAccess::makeTargetTriple({"a", "b", "c", "d", "e"}),
            "a-b-c-d-e");
}

TEST(ExecutorProcessInfoTest, CachedTargetTripleResultIsIdempotent) {
  EXPECT_EQ(ExecutorProcessInfo::detectTargetTriple(),
            ExecutorProcessInfo::detectTargetTriple());
}

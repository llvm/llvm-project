//===- CPUFeaturesTest.cpp ------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Tests for orc-rt's bedrock/host/CPUFeatures.h APIs.
//
//===----------------------------------------------------------------------===//

#include "orc-rt-internal/bedrock/host/CPUFeatures.h"
#include "gtest/gtest.h"

using namespace orc_rt;

TEST(CPUFeaturesTest, DetectDoesNotCrash) {
  [[maybe_unused]] auto _ = host::detectTargetCPUFeatures();
  SUCCEED();
}

// this test should catch any issues that arise from out of order tests
// hopefully.
TEST(CPUFeaturesTest, CachedResultIsIdempotent) {
  EXPECT_EQ(host::detectTargetCPUFeatures(), host::detectTargetCPUFeatures());
}

// A temporary test to tide us over until the next branch lands.
// We we enable regression testing for EPI.
#if defined(_WIN32) && defined(_M_X64)
TEST(CPUFeaturesTest, DetectsBaselineX86Features) {
  auto Features = host::detectTargetCPUFeatures();

  EXPECT_NE(std::find(Features.begin(), Features.end(), "sse"), Features.end());
  EXPECT_NE(std::find(Features.begin(), Features.end(), "sse2"),
            Features.end());
}
#endif

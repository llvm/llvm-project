//===- unittest/ProfileData/InstProfDataTest.cpp ----------------------------=//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "gtest/gtest.h"

#include <cstdint>

#define INSTR_PROF_VALUE_PROF_MEMOP_API
#include "llvm/ProfileData/InstrProfData.inc"

namespace {

TEST(InstrProfDataTest, MapValueToRangeRepValue) {
  EXPECT_EQ(0ULL, InstrProfGetRangeRepValue(0));
  EXPECT_EQ(1ULL, InstrProfGetRangeRepValue(1));
  EXPECT_EQ(2ULL, InstrProfGetRangeRepValue(2));
  EXPECT_EQ(3ULL, InstrProfGetRangeRepValue(3));
  EXPECT_EQ(4ULL, InstrProfGetRangeRepValue(4));
  EXPECT_EQ(5ULL, InstrProfGetRangeRepValue(5));
  EXPECT_EQ(6ULL, InstrProfGetRangeRepValue(6));
  EXPECT_EQ(7ULL, InstrProfGetRangeRepValue(7));
  EXPECT_EQ(8ULL, InstrProfGetRangeRepValue(8));
  EXPECT_EQ(9ULL, InstrProfGetRangeRepValue(9));
  EXPECT_EQ(16ULL, InstrProfGetRangeRepValue(16));
  EXPECT_EQ(17ULL, InstrProfGetRangeRepValue(30));
  EXPECT_EQ(32ULL, InstrProfGetRangeRepValue(32));
  EXPECT_EQ(33ULL, InstrProfGetRangeRepValue(54));
  EXPECT_EQ(64ULL, InstrProfGetRangeRepValue(64));
  EXPECT_EQ(65ULL, InstrProfGetRangeRepValue(127));
  EXPECT_EQ(128ULL, InstrProfGetRangeRepValue(128));
  EXPECT_EQ(129ULL, InstrProfGetRangeRepValue(200));
  EXPECT_EQ(256ULL, InstrProfGetRangeRepValue(256));
  EXPECT_EQ(257ULL, InstrProfGetRangeRepValue(397));
  EXPECT_EQ(512ULL, InstrProfGetRangeRepValue(512));
  EXPECT_EQ(513ULL, InstrProfGetRangeRepValue(2832048023));
}

TEST(InstrProfDataTest, IsInOneValueRange) {
  EXPECT_EQ(true, InstrProfIsSingleValRange(0));
  EXPECT_EQ(true, InstrProfIsSingleValRange(1));
  EXPECT_EQ(true, InstrProfIsSingleValRange(2));
  EXPECT_EQ(true, InstrProfIsSingleValRange(3));
  EXPECT_EQ(true, InstrProfIsSingleValRange(4));
  EXPECT_EQ(true, InstrProfIsSingleValRange(5));
  EXPECT_EQ(true, InstrProfIsSingleValRange(6));
  EXPECT_EQ(true, InstrProfIsSingleValRange(7));
  EXPECT_EQ(true, InstrProfIsSingleValRange(8));
  EXPECT_EQ(false, InstrProfIsSingleValRange(9));
  EXPECT_EQ(true, InstrProfIsSingleValRange(16));
  EXPECT_EQ(false, InstrProfIsSingleValRange(30));
  EXPECT_EQ(true, InstrProfIsSingleValRange(32));
  EXPECT_EQ(false, InstrProfIsSingleValRange(54));
  EXPECT_EQ(true, InstrProfIsSingleValRange(64));
  EXPECT_EQ(false, InstrProfIsSingleValRange(127));
  EXPECT_EQ(true, InstrProfIsSingleValRange(128));
  EXPECT_EQ(false, InstrProfIsSingleValRange(200));
  EXPECT_EQ(true, InstrProfIsSingleValRange(256));
  EXPECT_EQ(false, InstrProfIsSingleValRange(397));
  EXPECT_EQ(true, InstrProfIsSingleValRange(512));
  EXPECT_EQ(false, InstrProfIsSingleValRange(2832048023344));
}

} // end anonymous namespace

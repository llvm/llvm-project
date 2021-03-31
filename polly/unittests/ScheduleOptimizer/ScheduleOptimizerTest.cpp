//===- ScheduleOptimizerTest.cpp ------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "polly/ScheduleOptimizer.h"
#include "gtest/gtest.h"
#include "isl/stream.h"
#include "isl/val.h"

using namespace isl;
using namespace polly;

namespace {

TEST(ScheduleOptimizer, getPartialTilePrefixes) {

  isl_ctx *ctx = isl_ctx_alloc();

  {
    // Verify that for a loop with 3 iterations starting at 0 that is
    // pre-vectorized (strip-mined with a factor of 2), we correctly identify
    // that only the first two iterations are full vector iterations.
    isl::map Schedule(
        ctx, "{[i] -> [floor(i/2), i - 2 * floor(i/2)] : 0 <= i < 3 }");
    isl::set ScheduleRange = Schedule.range();
    isl::set Result = getPartialTilePrefixes(ScheduleRange, 2);

    EXPECT_TRUE(Result.is_equal(isl::set(ctx, "{[0]}")));
  }

  {
    // Verify that for a loop with 3 iterations starting at 1 that is
    // pre-vectorized (strip-mined with a factor of 2), we correctly identify
    // that only the last two iterations are full vector iterations.
    isl::map Schedule(
        ctx, "{[i] -> [floor(i/2), i - 2 * floor(i/2)] : 1 <= i < 4 }");
    isl::set ScheduleRange = Schedule.range();
    isl::set Result = getPartialTilePrefixes(ScheduleRange, 2);

    EXPECT_TRUE(Result.is_equal(isl::set(ctx, "{[1]}")));
  }

  {
    // Verify that for a loop with 6 iterations starting at 1 that is
    // pre-vectorized (strip-mined with a factor of 2), we correctly identify
    // that all but the first and the last iteration are full vector iterations.
    isl::map Schedule(
        ctx, "{[i] -> [floor(i/2), i - 2 * floor(i/2)] : 1 <= i < 6 }");
    isl::set ScheduleRange = Schedule.range();
    isl::set Result = getPartialTilePrefixes(ScheduleRange, 2);

    EXPECT_TRUE(Result.is_equal(isl::set(ctx, "{[1]; [2]}")));
  }

  isl_ctx_free(ctx);
}
} // anonymous namespace

//===- FlattenTest.cpp ----------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "polly/FlattenAlgo.h"
#include "polly/Support/GICHelper.h"
#include "gtest/gtest.h"
#include "isl/union_map.h"

using namespace llvm;
using namespace polly;

namespace {

/// Flatten a schedule and compare to the expected result.
///
/// @param ScheduleStr The schedule to flatten as string.
/// @param ExpectedStr The expected result as string.
///
/// @result Whether the flattened schedule is the same as the expected schedule.
bool checkFlatten(const char *ScheduleStr, const char *ExpectedStr) {
  auto *Ctx = isl_ctx_alloc();
  bool Success;

  {
    auto Schedule = isl::union_map(Ctx, ScheduleStr);
    auto Expected = isl::union_map(Ctx, ExpectedStr);

    auto Result = flattenSchedule(std::move(Schedule));
    Success = Result.is_equal(Expected);
  }

  isl_ctx_free(Ctx);
  return Success;
}

TEST(Flatten, FlattenTrivial) {
  EXPECT_TRUE(checkFlatten("{ A[] -> [0] }", "{ A[] -> [0] }"));
  EXPECT_TRUE(checkFlatten("{ A[i] -> [i, 0] : 0 <= i < 10 }",
                           "{ A[i] -> [i] : 0 <= i < 10 }"));
  EXPECT_TRUE(checkFlatten("{ A[i] -> [0, i] : 0 <= i < 10 }",
                           "{ A[i] -> [i] : 0 <= i < 10 }"));
}

TEST(Flatten, FlattenSequence) {
  EXPECT_TRUE(checkFlatten(
      "[n] -> { A[i] -> [0, i] : 0 <= i < n; B[i] -> [1, i] : 0 <= i < n }",
      "[n] -> { A[i] -> [i] : 0 <= i < n; B[i] -> [n + i] : 0 <= i < n }"));

  EXPECT_TRUE(checkFlatten(
      "{ A[i] -> [0, i] : 0 <= i < 10; B[i] -> [1, i] : 0 <= i < 10 }",
      "{ A[i] -> [i] : 0 <= i < 10; B[i] -> [10 + i] : 0 <= i < 10 }"));
}

TEST(Flatten, FlattenLoop) {
  EXPECT_TRUE(checkFlatten(
      "[n] -> { A[i] -> [i, 0] : 0 <= i < n; B[i] -> [i, 1] : 0 <= i < n }",
      "[n] -> { A[i] -> [2i] : 0 <= i < n; B[i] -> [2i + 1] : 0 <= i < n }"));

  EXPECT_TRUE(checkFlatten(
      "{ A[i] -> [i, 0] : 0 <= i < 10; B[i] -> [i, 1] : 0 <= i < 10 }",
      "{ A[i] -> [2i] : 0 <= i < 10; B[i] -> [2i + 1] : 0 <= i < 10 }"));
}
} // anonymous namespace

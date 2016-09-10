//===- FlattenTest.cpp ----------------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
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
  isl_bool Success;

  {
    auto Schedule = give(isl_union_map_read_from_str(Ctx, ScheduleStr));
    auto Expected = give(isl_union_map_read_from_str(Ctx, ExpectedStr));

    auto Result = flattenSchedule(std::move(Schedule));
    Success = isl_union_map_is_equal(Result.keep(), Expected.keep());
  }

  isl_ctx_free(Ctx);
  return Success == isl_bool_true;
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

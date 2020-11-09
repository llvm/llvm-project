//===- llvm/unittest/Support/ParallelTest.cpp -----------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Parallel.h unit tests.
///
//===----------------------------------------------------------------------===//

#include "llvm/Support/Parallel.h"
#include "gtest/gtest.h"
#include <array>
#include <random>

uint32_t array[1024 * 1024];

using namespace llvm;

// Tests below are hanging up on mingw. Investigating.
#if !defined(__MINGW32__)

TEST(Parallel, sort) {
  std::mt19937 randEngine;
  std::uniform_int_distribution<uint32_t> dist;

  for (auto &i : array)
    i = dist(randEngine);

  parallelSort(std::begin(array), std::end(array));
  ASSERT_TRUE(llvm::is_sorted(array));
}

TEST(Parallel, parallel_for) {
  // We need to test the case with a TaskSize > 1. We are white-box testing
  // here. The TaskSize is calculated as (End - Begin) / 1024 at the time of
  // writing.
  uint32_t range[2050];
  std::fill(range, range + 2050, 1);
  parallelForEachN(0, 2049, [&range](size_t I) { ++range[I]; });

  uint32_t expected[2049];
  std::fill(expected, expected + 2049, 2);
  ASSERT_TRUE(std::equal(range, range + 2049, expected));
  // Check that we don't write past the end of the requested range.
  ASSERT_EQ(range[2049], 1u);
}

TEST(Parallel, TransformReduce) {
  // Sum an empty list, check that it works.
  auto identity = [](uint32_t v) { return v; };
  uint32_t sum = parallelTransformReduce(ArrayRef<uint32_t>(), 0U,
                                         std::plus<uint32_t>(), identity);
  EXPECT_EQ(sum, 0U);

  // Sum the lengths of these strings in parallel.
  const char *strs[] = {"a", "ab", "abc", "abcd", "abcde", "abcdef"};
  size_t lenSum =
      parallelTransformReduce(strs, static_cast<size_t>(0), std::plus<size_t>(),
                              [](const char *s) { return strlen(s); });
  EXPECT_EQ(lenSum, static_cast<size_t>(21));

  // Check that we handle non-divisible task sizes as above.
  uint32_t range[2050];
  std::fill(std::begin(range), std::end(range), 1);
  sum = parallelTransformReduce(range, 0U, std::plus<uint32_t>(), identity);
  EXPECT_EQ(sum, 2050U);

  std::fill(std::begin(range), std::end(range), 2);
  sum = parallelTransformReduce(range, 0U, std::plus<uint32_t>(), identity);
  EXPECT_EQ(sum, 4100U);

  // Avoid one large task.
  uint32_t range2[3060];
  std::fill(std::begin(range2), std::end(range2), 1);
  sum = parallelTransformReduce(range2, 0U, std::plus<uint32_t>(), identity);
  EXPECT_EQ(sum, 3060U);
}

TEST(Parallel, ForEachError) {
  int nums[] = {1, 2, 3, 4, 5, 6};
  Error e = parallelForEachError(nums, [](int v) -> Error {
    if ((v & 1) == 0)
      return createStringError(std::errc::invalid_argument, "asdf");
    return Error::success();
  });
  EXPECT_TRUE(e.isA<ErrorList>());
  std::string errText = toString(std::move(e));
  EXPECT_EQ(errText, std::string("asdf\nasdf\nasdf"));
}

#endif

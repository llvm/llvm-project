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

  sort(parallel::par, std::begin(array), std::end(array));
  ASSERT_TRUE(std::is_sorted(std::begin(array), std::end(array)));
}

TEST(Parallel, parallel_for) {
  // We need to test the case with a TaskSize > 1. We are white-box testing
  // here. The TaskSize is calculated as (End - Begin) / 1024 at the time of
  // writing.
  uint32_t range[2050];
  std::fill(range, range + 2050, 1);
  for_each_n(parallel::par, 0, 2049, [&range](size_t I) { ++range[I]; });

  uint32_t expected[2049];
  std::fill(expected, expected + 2049, 2);
  ASSERT_TRUE(std::equal(range, range + 2049, expected));
  // Check that we don't write past the end of the requested range.
  ASSERT_EQ(range[2049], 1u);
}

#endif

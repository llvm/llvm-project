//===- unittests/ExponentialBackoffTest.cpp -------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/Support/ExponentialBackoff.h"
#include "gtest/gtest.h"
#include <chrono>

using namespace llvm;
using namespace std::chrono_literals;

namespace {

TEST(ExponentialBackoffTest, Timeout) {
  auto Start = std::chrono::steady_clock::now();
  // Use short enough times that this test runs quickly.
  ExponentialBackoff Backoff(100ms, 1ms, 10ms);
  do {
  } while (Backoff.waitForNextAttempt());
  auto Duration = std::chrono::steady_clock::now() - Start;
  EXPECT_GE(Duration, 100ms);
}

// Testing individual wait duration is omitted as those tests would be
// non-deterministic.

} // end anonymous namespace

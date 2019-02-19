//===--- status_test.cpp - Tests for the Status and Expected classes ------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "status.h"

#include "gtest/gtest.h"

#include <memory>

namespace {

struct RefCounter {
  static int Count;

  RefCounter() { ++Count; }
  ~RefCounter() { --Count; }
  RefCounter(const RefCounter &) = delete;
  RefCounter &operator=(const RefCounter &) = delete;
};

int RefCounter::Count;

TEST(Expected, RefCounter) {
  RefCounter::Count = 0;
  using uptr = std::unique_ptr<RefCounter>;

  acxxel::Expected<uptr> E0(uptr(new RefCounter));
  EXPECT_FALSE(E0.isError());
  EXPECT_EQ(1, RefCounter::Count);

  acxxel::Expected<uptr> E1(std::move(E0));
  EXPECT_FALSE(E1.isError());
  EXPECT_EQ(1, RefCounter::Count);

  acxxel::Expected<uptr> E2(acxxel::Status("nothing in here yet"));
  EXPECT_TRUE(E2.isError());
  EXPECT_EQ(1, RefCounter::Count);
  E2 = std::move(E1);
  EXPECT_FALSE(E2.isError());
  EXPECT_EQ(1, RefCounter::Count);

  EXPECT_EQ(1, E2.getValue()->Count);
  EXPECT_FALSE(E2.isError());
  EXPECT_EQ(1, RefCounter::Count);

  EXPECT_EQ(1, E2.takeValue()->Count);
  EXPECT_EQ(0, RefCounter::Count);
}

} // namespace

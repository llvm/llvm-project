//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "lldb/Utility/Iterable.h"

#include "gtest/gtest.h"

#include <utility>

using namespace lldb_private;

namespace {
struct FlagMutex {
  bool locked = false;
  void lock() { locked = true; }
  void unlock() { locked = false; }
};

// Records whether the mutex was held when the iterable asked for begin() and
// end().
struct Container {
  using const_iterator = const int *;

  explicit Container(FlagMutex &mutex) : mutex(mutex) {}

  const int *begin() const {
    begin_locked = mutex.locked;
    return values;
  }
  const int *end() const {
    end_locked = mutex.locked;
    return values + 3;
  }

  FlagMutex &mutex;
  int values[3] = {1, 2, 3};
  mutable bool begin_locked = false;
  mutable bool end_locked = false;
};

using Iterable = LockingAdaptedIterable<FlagMutex, Container>;
} // namespace

TEST(LockingAdaptedIterableTest, LocksBeforeReadingTheRange) {
  FlagMutex mutex;
  Container container(mutex);
  {
    Iterable iterable(container, mutex);
    EXPECT_TRUE(container.begin_locked);
    EXPECT_TRUE(container.end_locked);
    EXPECT_TRUE(mutex.locked);

    int sum = 0;
    for (int value : iterable)
      sum += value;
    EXPECT_EQ(sum, 6);
  }
  EXPECT_FALSE(mutex.locked);
}

TEST(LockingAdaptedIterableTest, MoveKeepsTheLock) {
  FlagMutex mutex;
  Container container(mutex);
  {
    Iterable iterable(container, mutex);
    {
      Iterable moved(std::move(iterable));
      EXPECT_TRUE(mutex.locked);
      EXPECT_EQ(moved.begin(), container.values);
    }
    EXPECT_FALSE(mutex.locked);
  }
  EXPECT_FALSE(mutex.locked);
}

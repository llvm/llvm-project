//===- LockedAccessTest.cpp -----------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Tests for orc-rt's LockedAccess.h APIs.
//
//===----------------------------------------------------------------------===//

#include "orc-rt/LockedAccess.h"
#include "gtest/gtest.h"

#include <mutex>
#include <thread>
#include <type_traits>

using namespace orc_rt;

namespace {

template <typename T> struct Foo {
  std::mutex M;
  T Val{};

  Foo() = default;
  Foo(T Val) : Val(std::move(Val)) {}

  auto get() { return LockedAccess(Val, M); }

  auto getConst() { return LockedAccess(std::as_const(Val), M); }
};

} // anonymous namespace

TEST(LockedAccessTest, ArrowRead) {
  Foo<std::pair<int, int>> F({42, 7});
  EXPECT_EQ(F.get()->first, 42);
  EXPECT_EQ(F.get()->second, 7);
}

TEST(LockedAccessTest, ArrowWrite) {
  Foo<std::pair<int, int>> F;
  F.get()->first = 10;
  F.get()->second = 20;
  EXPECT_EQ(F.Val.first, 10);
  EXPECT_EQ(F.Val.second, 20);
}

TEST(LockedAccessTest, ConstArrowReturnsConstPtr) {
  Foo<int> F(42);
  static_assert(std::is_const_v<
                    std::remove_pointer_t<decltype(F.getConst().operator->())>>,
                "const overload should return const pointer");
}

TEST(LockedAccessTest, DerefRead) {
  Foo<int> F(42);
  EXPECT_EQ(*F.get(), 42);
}

TEST(LockedAccessTest, DerefWrite) {
  Foo<int> F;
  *F.get() = 7;
  EXPECT_EQ(F.Val, 7);
}

TEST(LockedAccessTest, DerefPassToFunction) {
  Foo<int> F(42);
  auto timesTwo = [](int &X) { return X * 2; };
  EXPECT_EQ(timesTwo(*F.get()), 84);
}

TEST(LockedAccessTest, ConstDerefReturnsConstRef) {
  Foo<int> F(42);
  static_assert(
      std::is_const_v<std::remove_reference_t<decltype(*F.getConst())>>,
      "const overload should return const reference");
}

TEST(LockedAccessTest, WithRefRead) {
  Foo<int> F(42);
  F.get().with_ref([](int &Y) { EXPECT_EQ(Y, 42); });
}

TEST(LockedAccessTest, WithRefWrite) {
  Foo<int> F;
  F.get().with_ref([](int &Y) { Y = 7; });
  EXPECT_EQ(F.Val, 7);
}

TEST(LockedAccessTest, WithRefReturnValue) {
  Foo<int> F(42);
  int Result = F.get().with_ref([](int &Y) { return Y + 1; });
  EXPECT_EQ(Result, 43);
}

TEST(LockedAccessTest, WithRefReturnReference) {
  Foo<std::pair<int, int>> F({1, 2});
  int &Ref = F.get().with_ref(
      [](std::pair<int, int> &P) -> int & { return P.second; });
  EXPECT_EQ(&Ref, &F.Val.second);
}

TEST(LockedAccessTest, ConstWithRefGetsConstReference) {
  Foo<int> F(42);
  F.getConst().with_ref([](const int &Y) { EXPECT_EQ(Y, 42); });
}

TEST(LockedAccessTest, WithRefMultiStatement) {
  Foo<std::pair<int, int>> F;
  F.get().with_ref([](std::pair<int, int> &P) {
    P.first = 10;
    P.second = P.first + 5;
  });
  EXPECT_EQ(F.Val.first, 10);
  EXPECT_EQ(F.Val.second, 15);
}

TEST(LockedAccessTest, HoldsLockDuringArrow) {
  struct Checker {
    Foo<Checker> *F;
    bool isLocked() { return !F->M.try_lock(); }
  };
  Foo<Checker> F({&F});
  EXPECT_TRUE(F.get()->isLocked());
}

TEST(LockedAccessTest, HoldsLockDuringWithRef) {
  Foo<int> F;
  EXPECT_TRUE(F.get().with_ref([&](int &) { return !F.M.try_lock(); }));
}

TEST(LockedAccessTest, ProtectsAcrossThreads) {
  Foo<int> F;
  const int Iters = 10000;

  auto Increment = [&]() {
    for (int I = 0; I < Iters; ++I)
      F.get().with_ref([](int &C) { ++C; });
  };

  std::thread T1(Increment);
  std::thread T2(Increment);
  T1.join();
  T2.join();

  EXPECT_EQ(F.Val, 2 * Iters);
}

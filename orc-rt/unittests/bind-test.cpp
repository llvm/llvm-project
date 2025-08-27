//===- bind-test.cpp ------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Tests for orc-rt's bind-test.h APIs.
//
//===----------------------------------------------------------------------===//

#include "CommonTestUtils.h"
#include "orc-rt/bind.h"
#include "orc-rt/move_only_function.h"
#include "gtest/gtest.h"

using namespace orc_rt;

static void voidVoid(void) {}

TEST(BindTest, VoidVoid) {
  auto B = bind_front(voidVoid);
  B();
}

static int addInts(int X, int Y) { return X + Y; }

TEST(BindTest, SimpleBind) {
  auto Add1 = bind_front(addInts, 1);
  EXPECT_EQ(Add1(2), 3);
}

TEST(BindTest, NoBoundArguments) {
  auto Add = bind_front(addInts);
  EXPECT_EQ(Add(1, 2), 3);
}

TEST(BindTest, NoFreeArguments) {
  auto Add1And2 = bind_front(addInts, 1, 2);
  EXPECT_EQ(Add1And2(), 3);
}

TEST(BindTest, LambdaCapture) {
  auto Add1 = bind_front([](int X, int Y) { return X + Y; }, 1);
  EXPECT_EQ(Add1(2), 3);
}

TEST(BindTest, MinimalMoves) {
  OpCounter::reset();
  {
    auto B = bind_front([](OpCounter &O, int) {}, OpCounter());
    B(0);
  }
  EXPECT_EQ(OpCounter::defaultConstructions(), 1U);
  EXPECT_EQ(OpCounter::copies(), 0U);
  EXPECT_EQ(OpCounter::moves(), 1U);
  EXPECT_EQ(OpCounter::destructions(), 2U);
}

TEST(BindTest, MinimalCopies) {
  OpCounter::reset();
  {
    OpCounter O;
    auto B = bind_front([](OpCounter &O, int) {}, O);
    B(0);
  }
  EXPECT_EQ(OpCounter::defaultConstructions(), 1U);
  EXPECT_EQ(OpCounter::copies(), 1U);
  EXPECT_EQ(OpCounter::moves(), 0U);
  EXPECT_EQ(OpCounter::destructions(), 2U);
}

static int increment(int N) { return N + 1; }

TEST(BindTest, BindFunction) {
  auto Op = bind_front([](int op(int), int arg) { return op(arg); }, increment);
  EXPECT_EQ(Op(1), 2);
}

TEST(BindTest, BindTo_move_only_function) {
  move_only_function<int(int, int)> Add = [](int X, int Y) { return X + Y; };
  auto Add1 = bind_front(std::move(Add), 1);
  EXPECT_EQ(Add1(2), 3);
}

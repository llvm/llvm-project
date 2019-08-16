//===- llvm/unittest/ADT/MakeUniqueTest.cpp - make_unique unit tests ------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/ADT/STLExtras.h"
#include "gtest/gtest.h"
#include <tuple>
using namespace llvm;

namespace {

TEST(MakeUniqueTest, SingleObject) {
  auto p0 = make_unique<int>();
  EXPECT_TRUE((bool)p0);
  EXPECT_EQ(0, *p0);

  auto p1 = make_unique<int>(5);
  EXPECT_TRUE((bool)p1);
  EXPECT_EQ(5, *p1);

  auto p2 = make_unique<std::tuple<int, int>>(0, 1);
  EXPECT_TRUE((bool)p2);
  EXPECT_EQ(std::make_tuple(0, 1), *p2);

  auto p3 = make_unique<std::tuple<int, int, int>>(0, 1, 2);
  EXPECT_TRUE((bool)p3);
  EXPECT_EQ(std::make_tuple(0, 1, 2), *p3);

  auto p4 = make_unique<std::tuple<int, int, int, int>>(0, 1, 2, 3);
  EXPECT_TRUE((bool)p4);
  EXPECT_EQ(std::make_tuple(0, 1, 2, 3), *p4);

  auto p5 = make_unique<std::tuple<int, int, int, int, int>>(0, 1, 2, 3, 4);
  EXPECT_TRUE((bool)p5);
  EXPECT_EQ(std::make_tuple(0, 1, 2, 3, 4), *p5);

  auto p6 =
      make_unique<std::tuple<int, int, int, int, int, int>>(0, 1, 2, 3, 4, 5);
  EXPECT_TRUE((bool)p6);
  EXPECT_EQ(std::make_tuple(0, 1, 2, 3, 4, 5), *p6);

  auto p7 = make_unique<std::tuple<int, int, int, int, int, int, int>>(
      0, 1, 2, 3, 4, 5, 6);
  EXPECT_TRUE((bool)p7);
  EXPECT_EQ(std::make_tuple(0, 1, 2, 3, 4, 5, 6), *p7);

  auto p8 = make_unique<std::tuple<int, int, int, int, int, int, int, int>>(
      0, 1, 2, 3, 4, 5, 6, 7);
  EXPECT_TRUE((bool)p8);
  EXPECT_EQ(std::make_tuple(0, 1, 2, 3, 4, 5, 6, 7), *p8);

  auto p9 =
      make_unique<std::tuple<int, int, int, int, int, int, int, int, int>>(
          0, 1, 2, 3, 4, 5, 6, 7, 8);
  EXPECT_TRUE((bool)p9);
  EXPECT_EQ(std::make_tuple(0, 1, 2, 3, 4, 5, 6, 7, 8), *p9);

  auto p10 =
      make_unique<std::tuple<int, int, int, int, int, int, int, int, int, int>>(
          0, 1, 2, 3, 4, 5, 6, 7, 8, 9);
  EXPECT_TRUE((bool)p10);
  EXPECT_EQ(std::make_tuple(0, 1, 2, 3, 4, 5, 6, 7, 8, 9), *p10);
}

TEST(MakeUniqueTest, Array) {
  auto p1 = make_unique<int[]>(2);
  EXPECT_TRUE((bool)p1);
  EXPECT_EQ(0, p1[0]);
  EXPECT_EQ(0, p1[1]);
}
}

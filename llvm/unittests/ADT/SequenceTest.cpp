//===- SequenceTest.cpp - Unit tests for a sequence abstraciton -----------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/ADT/Sequence.h"
#include "gtest/gtest.h"

#include <list>

using namespace llvm;

namespace {

TEST(SequenceTest, Basic) {
  int x = 0;
  for (int i : seq(0, 10)) {
    EXPECT_EQ(x, i);
    x++;
  }
  EXPECT_EQ(10, x);

  auto my_seq = seq(0, 4);
  EXPECT_EQ(4, my_seq.end() - my_seq.begin());
  for (int i : {0, 1, 2, 3})
    EXPECT_EQ(i, (int)my_seq.begin()[i]);

  EXPECT_TRUE(my_seq.begin() < my_seq.end());

  auto adjusted_begin = my_seq.begin() + 2;
  auto adjusted_end = my_seq.end() - 2;
  EXPECT_TRUE(adjusted_begin == adjusted_end);
  EXPECT_EQ(2, *adjusted_begin);
  EXPECT_EQ(2, *adjusted_end);
}

} // anonymous namespace

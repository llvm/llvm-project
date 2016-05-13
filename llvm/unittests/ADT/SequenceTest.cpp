//===- SequenceTest.cpp - Unit tests for a sequence abstraciton -----------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "llvm/ADT/Sequence.h"
#include "gtest/gtest.h"

#include <list>

using namespace llvm;

namespace {

TEST(SequenceTest, Basic) {
  int x = 0;
  for (int i : seq(0, 10))
    EXPECT_EQ(x++, i);
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

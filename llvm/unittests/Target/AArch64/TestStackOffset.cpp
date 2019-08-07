//===- TestStackOffset.cpp - StackOffset unit tests------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "AArch64StackOffset.h"
#include "gtest/gtest.h"

using namespace llvm;

TEST(StackOffset, MixedSize) {
  StackOffset A(1, MVT::i8);
  EXPECT_EQ(1, A.getBytes());

  StackOffset B(2, MVT::i32);
  EXPECT_EQ(8, B.getBytes());

  StackOffset C(2, MVT::v4i64);
  EXPECT_EQ(64, C.getBytes());
}

TEST(StackOffset, Add) {
  StackOffset A(1, MVT::i64);
  StackOffset B(1, MVT::i32);
  StackOffset C = A + B;
  EXPECT_EQ(12, C.getBytes());

  StackOffset D(1, MVT::i32);
  D += A;
  EXPECT_EQ(12, D.getBytes());
}

TEST(StackOffset, Sub) {
  StackOffset A(1, MVT::i64);
  StackOffset B(1, MVT::i32);
  StackOffset C = A - B;
  EXPECT_EQ(4, C.getBytes());

  StackOffset D(1, MVT::i64);
  D -= A;
  EXPECT_EQ(0, D.getBytes());
}

TEST(StackOffset, isZero) {
  StackOffset A(0, MVT::i64);
  StackOffset B(0, MVT::i32);
  EXPECT_TRUE(!A);
  EXPECT_TRUE(!(A + B));
}

TEST(StackOffset, getForFrameOffset) {
  StackOffset A(1, MVT::i64);
  StackOffset B(1, MVT::i32);
  int64_t ByteSized;
  (A + B).getForFrameOffset(ByteSized);
  EXPECT_EQ(12, ByteSized);
}

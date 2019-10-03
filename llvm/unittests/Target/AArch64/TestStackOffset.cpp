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

  StackOffset D(2, MVT::nxv4i64);
  EXPECT_EQ(64, D.getScalableBytes());

  StackOffset E(2, MVT::v4i64);
  EXPECT_EQ(0, E.getScalableBytes());

  StackOffset F(2, MVT::nxv4i64);
  EXPECT_EQ(0, F.getBytes());
}

TEST(StackOffset, Add) {
  StackOffset A(1, MVT::i64);
  StackOffset B(1, MVT::i32);
  StackOffset C = A + B;
  EXPECT_EQ(12, C.getBytes());

  StackOffset D(1, MVT::i32);
  D += A;
  EXPECT_EQ(12, D.getBytes());

  StackOffset E(1, MVT::nxv1i32);
  StackOffset F = C + E;
  EXPECT_EQ(12, F.getBytes());
  EXPECT_EQ(4, F.getScalableBytes());
}

TEST(StackOffset, Sub) {
  StackOffset A(1, MVT::i64);
  StackOffset B(1, MVT::i32);
  StackOffset C = A - B;
  EXPECT_EQ(4, C.getBytes());

  StackOffset D(1, MVT::i64);
  D -= A;
  EXPECT_EQ(0, D.getBytes());

  C += StackOffset(2, MVT::nxv1i32);
  StackOffset E = StackOffset(1, MVT::nxv1i32);
  StackOffset F = C - E;
  EXPECT_EQ(4, F.getBytes());
  EXPECT_EQ(4, F.getScalableBytes());
}

TEST(StackOffset, isZero) {
  StackOffset A(0, MVT::i64);
  StackOffset B(0, MVT::i32);
  EXPECT_TRUE(!A);
  EXPECT_TRUE(!(A + B));

  StackOffset C(0, MVT::nxv1i32);
  EXPECT_TRUE(!(A + C));

  StackOffset D(1, MVT::nxv1i32);
  EXPECT_FALSE(!(A + D));
}

TEST(StackOffset, isValid) {
  EXPECT_FALSE(StackOffset(1, MVT::nxv8i1).isValid());
  EXPECT_TRUE(StackOffset(2, MVT::nxv8i1).isValid());

#ifndef NDEBUG
#ifdef GTEST_HAS_DEATH_TEST
  EXPECT_DEATH(StackOffset(1, MVT::i1),
               "Offset type is not a multiple of bytes");
  EXPECT_DEATH(StackOffset(1, MVT::nxv1i1),
               "Offset type is not a multiple of bytes");
#endif // defined GTEST_HAS_DEATH_TEST
#endif // not defined NDEBUG
}

TEST(StackOffset, getForFrameOffset) {
  StackOffset A(1, MVT::i64);
  StackOffset B(1, MVT::i32);
  StackOffset C(1, MVT::nxv4i32);

  // If all offsets can be materialized with only ADDVL,
  // make sure PLSized is 0.
  int64_t ByteSized, VLSized, PLSized;
  (A + B + C).getForFrameOffset(ByteSized, PLSized, VLSized);
  EXPECT_EQ(12, ByteSized);
  EXPECT_EQ(1, VLSized);
  EXPECT_EQ(0, PLSized);

  // If we need an ADDPL to materialize the offset, and the number of scalable
  // bytes fits the ADDPL immediate, fold the scalable bytes to fit in PLSized.
  StackOffset D(1, MVT::nxv16i1);
  (C + D).getForFrameOffset(ByteSized, PLSized, VLSized);
  EXPECT_EQ(0, ByteSized);
  EXPECT_EQ(0, VLSized);
  EXPECT_EQ(9, PLSized);

  StackOffset E(4, MVT::nxv4i32);
  StackOffset F(1, MVT::nxv16i1);
  (E + F).getForFrameOffset(ByteSized, PLSized, VLSized);
  EXPECT_EQ(0, ByteSized);
  EXPECT_EQ(0, VLSized);
  EXPECT_EQ(33, PLSized);

  // If the offset requires an ADDPL instruction to materialize, and would
  // require more than two instructions, decompose it into both
  // ADDVL (n x 16 bytes) and ADDPL (n x 2 bytes) instructions.
  StackOffset G(8, MVT::nxv4i32);
  StackOffset H(1, MVT::nxv16i1);
  (G + H).getForFrameOffset(ByteSized, PLSized, VLSized);
  EXPECT_EQ(0, ByteSized);
  EXPECT_EQ(8, VLSized);
  EXPECT_EQ(1, PLSized);
}

//===------ HLSLRootSignatureRangeTest.cpp - RootSignature Range tests ----===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/Frontend/HLSL/RootSignatureValidations.h"
#include "gtest/gtest.h"

using namespace llvm::hlsl::rootsig;

namespace {

TEST(HLSLRootSignatureTest, NoOverlappingInsertTests) {
  // Ensures that there is never a reported overlap
  ResourceRange::MapT::Allocator Allocator;
  ResourceRange Range(Allocator);

  RangeInfo A;
  A.LowerBound = 0;
  A.UpperBound = 3;
  EXPECT_EQ(Range.insert(A), std::nullopt);

  RangeInfo B;
  B.LowerBound = 4;
  B.UpperBound = 7;
  EXPECT_EQ(Range.insert(B), std::nullopt);

  RangeInfo C;
  C.LowerBound = 10;
  C.UpperBound = RangeInfo::Unbounded;
  EXPECT_EQ(Range.insert(C), std::nullopt);

  // A = [0;3]
  EXPECT_EQ(Range.lookup(0), &A);
  EXPECT_EQ(Range.lookup(2), &A);
  EXPECT_EQ(Range.lookup(3), &A);

  // B = [4;7]
  EXPECT_EQ(Range.lookup(4), &B);
  EXPECT_EQ(Range.lookup(5), &B);
  EXPECT_EQ(Range.lookup(7), &B);

  EXPECT_EQ(Range.lookup(8), nullptr);
  EXPECT_EQ(Range.lookup(9), nullptr);

  // C = [10;unbounded]
  EXPECT_EQ(Range.lookup(10), &C);
  EXPECT_EQ(Range.lookup(42), &C);
  EXPECT_EQ(Range.lookup(98237423), &C);
  EXPECT_EQ(Range.lookup(RangeInfo::Unbounded), &C);
}

TEST(HLSLRootSignatureTest, SingleOverlappingInsertTests) {
  // Ensures that we correctly report an overlap when we insert a range that
  // overlaps with one other range but does not cover (replace) it
  ResourceRange::MapT::Allocator Allocator;
  ResourceRange Range(Allocator);

  RangeInfo A;
  A.LowerBound = 1;
  A.UpperBound = 5;
  EXPECT_EQ(Range.insert(A), std::nullopt);

  RangeInfo B;
  B.LowerBound = 0;
  B.UpperBound = 2;
  EXPECT_EQ(Range.insert(B).value(), &A);

  RangeInfo C;
  C.LowerBound = 4;
  C.UpperBound = RangeInfo::Unbounded;
  EXPECT_EQ(Range.insert(C).value(), &A);

  // A = [1;5]
  EXPECT_EQ(Range.lookup(1), &A);
  EXPECT_EQ(Range.lookup(2), &A);
  EXPECT_EQ(Range.lookup(3), &A);
  EXPECT_EQ(Range.lookup(4), &A);
  EXPECT_EQ(Range.lookup(5), &A);

  // B = [0;0]
  EXPECT_EQ(Range.lookup(0), &B);

  // C = [6; unbounded]
  EXPECT_EQ(Range.lookup(6), &C);
  EXPECT_EQ(Range.lookup(RangeInfo::Unbounded), &C);
}

TEST(HLSLRootSignatureTest, MultipleOverlappingInsertTests) {
  // Ensures that we correctly report an overlap when inserted range
  // overlaps more than one range and it does not cover (replace) either
  // range. In this case it will just fill in the interval between the two
  ResourceRange::MapT::Allocator Allocator;
  ResourceRange Range(Allocator);

  RangeInfo A;
  A.LowerBound = 0;
  A.UpperBound = 2;
  EXPECT_EQ(Range.insert(A), std::nullopt);

  RangeInfo B;
  B.LowerBound = 4;
  B.UpperBound = 6;
  EXPECT_EQ(Range.insert(B), std::nullopt);

  RangeInfo C;
  C.LowerBound = 1;
  C.UpperBound = 5;
  EXPECT_EQ(Range.insert(C).value(), &A);

  // A = [0;2]
  EXPECT_EQ(Range.lookup(0), &A);
  EXPECT_EQ(Range.lookup(1), &A);
  EXPECT_EQ(Range.lookup(2), &A);

  // B = [4;6]
  EXPECT_EQ(Range.lookup(4), &B);
  EXPECT_EQ(Range.lookup(5), &B);
  EXPECT_EQ(Range.lookup(6), &B);

  // C = [3;3]
  EXPECT_EQ(Range.lookup(3), &C);
}

TEST(HLSLRootSignatureTest, CoverInsertTests) {
  // Ensures that we correctly report an overlap when inserted range
  // covers one or more ranges
  ResourceRange::MapT::Allocator Allocator;
  ResourceRange Range(Allocator);

  RangeInfo A;
  A.LowerBound = 0;
  A.UpperBound = 2;
  EXPECT_EQ(Range.insert(A), std::nullopt);

  RangeInfo B;
  B.LowerBound = 4;
  B.UpperBound = 5;
  EXPECT_EQ(Range.insert(B), std::nullopt);

  // Covers B
  RangeInfo C;
  C.LowerBound = 4;
  C.UpperBound = 6;
  EXPECT_EQ(Range.insert(C).value(), &B);

  // A = [0;2]
  // C = [4;6] <- covers reference to B
  EXPECT_EQ(Range.lookup(0), &A);
  EXPECT_EQ(Range.lookup(1), &A);
  EXPECT_EQ(Range.lookup(2), &A);
  EXPECT_EQ(Range.lookup(3), nullptr);
  EXPECT_EQ(Range.lookup(4), &C);
  EXPECT_EQ(Range.lookup(5), &C);
  EXPECT_EQ(Range.lookup(6), &C);

  // Covers all other ranges
  RangeInfo D;
  D.LowerBound = 0;
  D.UpperBound = 7;
  EXPECT_EQ(Range.insert(D).value(), &A);

  // D = [0;7] <- Covers reference to A and C
  EXPECT_EQ(Range.lookup(0), &D);
  EXPECT_EQ(Range.lookup(1), &D);
  EXPECT_EQ(Range.lookup(2), &D);
  EXPECT_EQ(Range.lookup(3), &D);
  EXPECT_EQ(Range.lookup(4), &D);
  EXPECT_EQ(Range.lookup(5), &D);
  EXPECT_EQ(Range.lookup(6), &D);
  EXPECT_EQ(Range.lookup(7), &D);
}

} // namespace

//===- RegAllocBasicTest.cpp - Tests for basic register allocator --------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "../../lib/CodeGen/RegAllocBasic.h"
#include "llvm/CodeGen/LiveInterval.h"
#include "gtest/gtest.h"
#include <queue>

using namespace llvm;

namespace {

TEST(RegAllocBasicTest, CompSpillWeightDeterminism) {
  // Test that CompSpillWeight provides deterministic ordering by testing
  // all valid combinations of weight and register number comparisons.
  //
  // This is a regression test for non-deterministic behavior that occurred
  // when ASAN changed memory layout, causing different heap configurations
  // for LiveIntervals with identical spill weights.
  //
  // Comparison has two components: (weight, register number)
  // Each can be <, =, or > resulting in 9 total combinations, but the case
  // where register is equal and weight is different is impossible because each
  // virtual register has exactly one LiveInterval with one weight.

  // Create 3 LiveIntervals to test all 7 valid comparison combinations.
  // Layout: W{weight}R{register}
  LiveInterval W1R10(Register::index2VirtReg(10), 1.0f);
  LiveInterval W1R30(Register::index2VirtReg(30), 1.0f);
  LiveInterval W2R20(Register::index2VirtReg(20), 2.0f);

  CompSpillWeight IsLessThan;

  // Test all meaningful combinations of (weight comparison, register
  // comparison). IsLessThan(A, B) returns true if A has lower priority than B
  // (A < B in max-heap)

  // Case 1: weight <, reg < : A has lower priority (weight dominates)
  EXPECT_TRUE(IsLessThan(&W1R10, &W2R20));
  EXPECT_FALSE(IsLessThan(&W2R20, &W1R10));

  // Case 2: weight <, reg > : A has lower priority (weight dominates)
  EXPECT_TRUE(IsLessThan(&W1R30, &W2R20));
  EXPECT_FALSE(IsLessThan(&W2R20, &W1R30));

  // Case 3: weight =, reg < : A has lower priority (register breaks tie)
  EXPECT_TRUE(IsLessThan(&W1R10, &W1R30));
  EXPECT_FALSE(IsLessThan(&W1R30, &W1R10));

  // Case 4: weight =, reg = : equal (both comparisons return false)
  EXPECT_FALSE(IsLessThan(&W1R10, &W1R10));

  // Case 5: weight =, reg > : A has higher priority (register breaks tie)
  EXPECT_FALSE(IsLessThan(&W1R30, &W1R10));
  EXPECT_TRUE(IsLessThan(&W1R10, &W1R30));

  // Case 6: weight >, reg < : A has higher priority (weight dominates)
  EXPECT_FALSE(IsLessThan(&W2R20, &W1R30));
  EXPECT_TRUE(IsLessThan(&W1R30, &W2R20));

  // Case 7: weight >, reg > : A has higher priority (weight dominates)
  EXPECT_FALSE(IsLessThan(&W2R20, &W1R10));
  EXPECT_TRUE(IsLessThan(&W1R10, &W2R20));

  // Test priority_queue ordering with all intervals
  std::priority_queue<const LiveInterval *, std::vector<const LiveInterval *>,
                      CompSpillWeight>
      PQ;

  // Insert all intervals in arbitrary order
  PQ.push(&W1R30);
  PQ.push(&W2R20);
  PQ.push(&W1R10);

  // Expected order: highest weight first, then descending register within ties
  // Weight 2.0: R20
  EXPECT_EQ(PQ.top()->reg(), Register::index2VirtReg(20));
  EXPECT_EQ(PQ.top()->weight(), 2.0f);
  PQ.pop();

  // Weight 1.0: R30, R10 (descending register order)
  EXPECT_EQ(PQ.top()->reg(), Register::index2VirtReg(30));
  EXPECT_EQ(PQ.top()->weight(), 1.0f);
  PQ.pop();
  EXPECT_EQ(PQ.top()->reg(), Register::index2VirtReg(10));
  EXPECT_EQ(PQ.top()->weight(), 1.0f);
}

} // anonymous namespace

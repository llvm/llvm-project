//===- SliceWalkTest.cpp - Tests for slice walk continuations
//---------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Analysis/SliceWalk.h"
#include "mlir/IR/Block.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/MLIRContext.h"
#include "gtest/gtest.h"

using namespace mlir;

TEST(WalkContinuationTest, CopyAndMoveEmptyContinuations) {
  for (auto continuation :
       {WalkContinuation::interrupt(), WalkContinuation::skip(),
        WalkContinuation::advanceTo({})}) {
    auto copy = continuation;
    auto moved = std::move(copy);
    EXPECT_EQ(moved.wasInterrupted(), continuation.wasInterrupted());
    EXPECT_EQ(moved.wasSkipped(), continuation.wasSkipped());
    EXPECT_EQ(moved.wasAdvancedTo(), continuation.wasAdvancedTo());
    EXPECT_TRUE(moved.getNextValues().empty());

    auto assigned = WalkContinuation::interrupt();
    assigned = continuation;
    EXPECT_EQ(assigned.wasSkipped(), continuation.wasSkipped());
    EXPECT_EQ(assigned.wasAdvancedTo(), continuation.wasAdvancedTo());
    assigned = std::move(moved);
    EXPECT_EQ(assigned.wasInterrupted(), continuation.wasInterrupted());
    EXPECT_EQ(assigned.wasSkipped(), continuation.wasSkipped());
    EXPECT_EQ(assigned.wasAdvancedTo(), continuation.wasAdvancedTo());
  }
}

TEST(WalkContinuationTest, CopyAndMoveHeapContinuations) {
  MLIRContext context;
  Block block;
  for (unsigned i = 0; i < 32; ++i)
    block.addArgument(IntegerType::get(&context, 32),
                      UnknownLoc::get(&context));
  SmallVector<Value> values(block.getArguments());

  auto continuation = WalkContinuation::advanceTo(values);
  auto copy = continuation;
  auto moved = std::move(copy);
  EXPECT_TRUE(moved.wasAdvancedTo());
  EXPECT_EQ(moved.getNextValues(), ArrayRef<Value>(values));

  auto assigned = WalkContinuation::skip();
  assigned = continuation;
  EXPECT_TRUE(assigned.wasAdvancedTo());
  EXPECT_EQ(assigned.getNextValues(), ArrayRef<Value>(values));
  assigned = std::move(moved);
  EXPECT_TRUE(assigned.wasAdvancedTo());
  EXPECT_EQ(assigned.getNextValues(), ArrayRef<Value>(values));

  assigned = WalkContinuation::skip();
  EXPECT_TRUE(assigned.wasSkipped());
  EXPECT_TRUE(assigned.getNextValues().empty());
}

TEST(WalkContinuationTest, DiagnosticInterrupts) {
  MLIRContext context;
  WalkContinuation continuation(
      Diagnostic(UnknownLoc::get(&context), DiagnosticSeverity::Error));
  EXPECT_TRUE(continuation.wasInterrupted());
  EXPECT_TRUE(continuation.getNextValues().empty());
}

//===- OperationSupportTest.cpp - Operation support unit tests ------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/IR/OperationSupport.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/StandardTypes.h"
#include "gtest/gtest.h"

using namespace mlir;
using namespace mlir::detail;

static Operation *createOp(MLIRContext *context,
                           ArrayRef<Value> operands = llvm::None,
                           ArrayRef<Type> resultTypes = llvm::None,
                           unsigned int numRegions = 0) {
  context->allowUnregisteredDialects();
  return Operation::create(UnknownLoc::get(context),
                           OperationName("foo.bar", context), resultTypes,
                           operands, llvm::None, llvm::None, numRegions);
}

namespace {
TEST(OperandStorageTest, NonResizable) {
  MLIRContext context(false);
  Builder builder(&context);

  Operation *useOp =
      createOp(&context, /*operands=*/llvm::None, builder.getIntegerType(16));
  Value operand = useOp->getResult(0);

  // Create a non-resizable operation with one operand.
  Operation *user = createOp(&context, operand);

  // The same number of operands is okay.
  user->setOperands(operand);
  EXPECT_EQ(user->getNumOperands(), 1u);

  // Removing is okay.
  user->setOperands(llvm::None);
  EXPECT_EQ(user->getNumOperands(), 0u);

  // Destroy the operations.
  user->destroy();
  useOp->destroy();
}

TEST(OperandStorageTest, Resizable) {
  MLIRContext context(false);
  Builder builder(&context);

  Operation *useOp =
      createOp(&context, /*operands=*/llvm::None, builder.getIntegerType(16));
  Value operand = useOp->getResult(0);

  // Create a resizable operation with one operand.
  Operation *user = createOp(&context, operand);

  // The same number of operands is okay.
  user->setOperands(operand);
  EXPECT_EQ(user->getNumOperands(), 1u);

  // Removing is okay.
  user->setOperands(llvm::None);
  EXPECT_EQ(user->getNumOperands(), 0u);

  // Adding more operands is okay.
  user->setOperands({operand, operand, operand});
  EXPECT_EQ(user->getNumOperands(), 3u);

  // Destroy the operations.
  user->destroy();
  useOp->destroy();
}

TEST(OperandStorageTest, RangeReplace) {
  MLIRContext context(false);
  Builder builder(&context);

  Operation *useOp =
      createOp(&context, /*operands=*/llvm::None, builder.getIntegerType(16));
  Value operand = useOp->getResult(0);

  // Create a resizable operation with one operand.
  Operation *user = createOp(&context, operand);

  // Check setting with the same number of operands.
  user->setOperands(/*start=*/0, /*length=*/1, operand);
  EXPECT_EQ(user->getNumOperands(), 1u);

  // Check setting with more operands.
  user->setOperands(/*start=*/0, /*length=*/1, {operand, operand, operand});
  EXPECT_EQ(user->getNumOperands(), 3u);

  // Check setting with less operands.
  user->setOperands(/*start=*/1, /*length=*/2, {operand});
  EXPECT_EQ(user->getNumOperands(), 2u);

  // Check inserting without replacing operands.
  user->setOperands(/*start=*/2, /*length=*/0, {operand});
  EXPECT_EQ(user->getNumOperands(), 3u);

  // Check erasing operands.
  user->setOperands(/*start=*/0, /*length=*/3, {});
  EXPECT_EQ(user->getNumOperands(), 0u);

  // Destroy the operations.
  user->destroy();
  useOp->destroy();
}

TEST(OperandStorageTest, MutableRange) {
  MLIRContext context(false);
  Builder builder(&context);

  Operation *useOp =
      createOp(&context, /*operands=*/llvm::None, builder.getIntegerType(16));
  Value operand = useOp->getResult(0);

  // Create a resizable operation with one operand.
  Operation *user = createOp(&context, operand);

  // Check setting with the same number of operands.
  MutableOperandRange mutableOperands(user);
  mutableOperands.assign(operand);
  EXPECT_EQ(mutableOperands.size(), 1u);
  EXPECT_EQ(user->getNumOperands(), 1u);

  // Check setting with more operands.
  mutableOperands.assign({operand, operand, operand});
  EXPECT_EQ(mutableOperands.size(), 3u);
  EXPECT_EQ(user->getNumOperands(), 3u);

  // Check with inserting a new operand.
  mutableOperands.append({operand, operand});
  EXPECT_EQ(mutableOperands.size(), 5u);
  EXPECT_EQ(user->getNumOperands(), 5u);

  // Check erasing operands.
  mutableOperands.clear();
  EXPECT_EQ(mutableOperands.size(), 0u);
  EXPECT_EQ(user->getNumOperands(), 0u);

  // Destroy the operations.
  user->destroy();
  useOp->destroy();
}

TEST(OperationOrderTest, OrderIsAlwaysValid) {
  MLIRContext context(false);
  Builder builder(&context);

  Operation *containerOp =
      createOp(&context, /*operands=*/llvm::None, /*resultTypes=*/llvm::None,
               /*numRegions=*/1);
  Region &region = containerOp->getRegion(0);
  Block *block = new Block();
  region.push_back(block);

  // Insert two operations, then iteratively add more operations in the middle
  // of them. Eventually we will insert more than kOrderStride operations and
  // the block order will need to be recomputed.
  Operation *frontOp = createOp(&context);
  Operation *backOp = createOp(&context);
  block->push_back(frontOp);
  block->push_back(backOp);

  // Chosen to be larger than Operation::kOrderStride.
  int kNumOpsToInsert = 10;
  for (int i = 0; i < kNumOpsToInsert; ++i) {
    Operation *op = createOp(&context);
    block->getOperations().insert(backOp->getIterator(), op);
    ASSERT_TRUE(op->isBeforeInBlock(backOp));
    // Note verifyOpOrder() returns false if the order is valid.
    ASSERT_FALSE(block->verifyOpOrder());
  }

  containerOp->destroy();
}

} // end namespace

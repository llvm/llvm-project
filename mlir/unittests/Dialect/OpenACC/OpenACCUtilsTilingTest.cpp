//===- OpenACCUtilsTilingTest.cpp - Unit tests for loop tiling utilities --===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/OpenACC/OpenACCUtilsTiling.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/OpenACC/OpenACC.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OwningOpRef.h"
#include "gtest/gtest.h"

using namespace mlir;
using namespace mlir::acc;

//===----------------------------------------------------------------------===//
// Test Fixture
//===----------------------------------------------------------------------===//

class OpenACCUtilsTilingTest : public ::testing::Test {
protected:
  OpenACCUtilsTilingTest() : b(&context), loc(UnknownLoc::get(&context)) {
    context.loadDialect<acc::OpenACCDialect, arith::ArithDialect,
                        memref::MemRefDialect, func::FuncDialect>();
  }

  // Create a simple LoopOp with specified bounds using the simple builder
  acc::LoopOp createLoopOp(OpBuilder &builder, ValueRange lbs, ValueRange ubs,
                           ValueRange steps) {
    auto loopOp = acc::LoopOp::create(builder, loc, lbs, ubs, steps,
                                      acc::LoopParMode::loop_independent);

    // Add body block with IV arguments and yield
    Region &region = loopOp.getRegion();
    Block *block = builder.createBlock(&region, region.begin());
    for (Value lb : lbs)
      block->addArgument(lb.getType(), loc);
    builder.setInsertionPointToEnd(block);
    acc::YieldOp::create(builder, loc);

    return loopOp;
  }

  // Helper to count nested acc.loop ops within a loop
  unsigned countNestedLoops(acc::LoopOp loop) {
    unsigned count = 0;
    loop.getBody().walk([&](acc::LoopOp) { ++count; });
    return count;
  }

  // Helper to collect all nested acc.loop ops in order
  SmallVector<acc::LoopOp> collectNestedLoops(acc::LoopOp loop) {
    SmallVector<acc::LoopOp> loops;
    loop.getBody().walk(
        [&](acc::LoopOp nestedLoop) { loops.push_back(nestedLoop); });
    return loops;
  }

  MLIRContext context;
  OpBuilder b;
  Location loc;
};

//===----------------------------------------------------------------------===//
// tileACCLoops Tests
//===----------------------------------------------------------------------===//

TEST_F(OpenACCUtilsTilingTest, tileACCLoopsSingleLoop) {
  // Create a module to hold the function
  OwningOpRef<ModuleOp> module = ModuleOp::create(loc);
  Block *moduleBlock = module->getBody();

  OpBuilder::InsertionGuard guard(b);
  b.setInsertionPointToStart(moduleBlock);

  // Create a function
  auto funcType = b.getFunctionType({}, {});
  OwningOpRef<func::FuncOp> funcOp =
      func::FuncOp::create(b, loc, "test_func", funcType);
  Block *funcBlock = funcOp->addEntryBlock();

  b.setInsertionPointToStart(funcBlock);

  // Create loop bounds
  Value lb =
      arith::ConstantOp::create(b, loc, b.getIndexType(), b.getIndexAttr(0));
  Value ub =
      arith::ConstantOp::create(b, loc, b.getIndexType(), b.getIndexAttr(100));
  Value step =
      arith::ConstantOp::create(b, loc, b.getIndexType(), b.getIndexAttr(1));
  Value tileSize =
      arith::ConstantOp::create(b, loc, b.getIndexType(), b.getIndexAttr(4));

  // Create the loop (single IV)
  acc::LoopOp loopOp = createLoopOp(b, {lb}, {ub}, {step});

  // Tile the loop using IRRewriter
  IRRewriter rewriter(&context);
  rewriter.setInsertionPoint(loopOp);

  SmallVector<Value> tileSizes = {tileSize};

  acc::LoopOp tileGroup =
      tileACCLoops(loopOp, tileSizes, /*defaultTileSize=*/128, rewriter);

  // Verify the tile-group loop was created
  EXPECT_TRUE(tileGroup != nullptr);
  EXPECT_FALSE(tileGroup.getBody().empty());

  // A single-IV tile(4) produces one tile-group loop wrapping one element-group
  // loop: exactly one nested loop.
  EXPECT_EQ(countNestedLoops(tileGroup), 1u);

  // The tile-group loop carries its single IV.
  EXPECT_EQ(tileGroup.getBody().getNumArguments(), 1u);

  auto nestedLoops = collectNestedLoops(tileGroup);
  ASSERT_EQ(nestedLoops.size(), 1u);
  // The element-group loop should carry the single IV as well.
  EXPECT_EQ(nestedLoops[0].getBody().getNumArguments(), 1u);
}

TEST_F(OpenACCUtilsTilingTest, tileACCLoopsFusedTwoDim) {
  // Create a module to hold the function
  OwningOpRef<ModuleOp> module = ModuleOp::create(loc);
  Block *moduleBlock = module->getBody();

  OpBuilder::InsertionGuard guard(b);
  b.setInsertionPointToStart(moduleBlock);

  // Create a function
  auto funcType = b.getFunctionType({}, {});
  OwningOpRef<func::FuncOp> funcOp =
      func::FuncOp::create(b, loc, "test_func", funcType);
  Block *funcBlock = funcOp->addEntryBlock();

  b.setInsertionPointToStart(funcBlock);

  // Create bounds for a single fused loop carrying two IVs.
  Value lb1 =
      arith::ConstantOp::create(b, loc, b.getIndexType(), b.getIndexAttr(0));
  Value ub1 =
      arith::ConstantOp::create(b, loc, b.getIndexType(), b.getIndexAttr(100));
  Value step1 =
      arith::ConstantOp::create(b, loc, b.getIndexType(), b.getIndexAttr(1));
  Value lb2 =
      arith::ConstantOp::create(b, loc, b.getIndexType(), b.getIndexAttr(0));
  Value ub2 =
      arith::ConstantOp::create(b, loc, b.getIndexType(), b.getIndexAttr(50));
  Value step2 =
      arith::ConstantOp::create(b, loc, b.getIndexType(), b.getIndexAttr(1));

  // Tile sizes
  Value tileSize1 =
      arith::ConstantOp::create(b, loc, b.getIndexType(), b.getIndexAttr(4));
  Value tileSize2 =
      arith::ConstantOp::create(b, loc, b.getIndexType(), b.getIndexAttr(8));

  // Create a single fused loop with two IVs.
  acc::LoopOp fusedLoop =
      createLoopOp(b, {lb1, lb2}, {ub1, ub2}, {step1, step2});

  IRRewriter rewriter(&context);
  rewriter.setInsertionPoint(fusedLoop);

  SmallVector<Value> tileSizes = {tileSize1, tileSize2};

  acc::LoopOp tileGroup =
      tileACCLoops(fusedLoop, tileSizes, /*defaultTileSize=*/128, rewriter);

  // Verify the tile-group loop nest was created.
  EXPECT_TRUE(tileGroup != nullptr);
  EXPECT_FALSE(tileGroup.getBody().empty());

  // tile(4,8) on a fused 2-IV loop produces exactly two multi-IV loops: a
  // tile-group loop wrapping a single element-group loop (one nested loop),
  // rather than a 4-deep nest of single-IV loops.
  EXPECT_EQ(countNestedLoops(tileGroup), 1u);

  // Both groups carry all (two) IVs.
  EXPECT_EQ(tileGroup.getBody().getNumArguments(), 2u);

  auto nestedLoops = collectNestedLoops(tileGroup);
  ASSERT_EQ(nestedLoops.size(), 1u);
  EXPECT_EQ(nestedLoops[0].getBody().getNumArguments(), 2u);
}

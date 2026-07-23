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

  // Create the loop
  acc::LoopOp loopOp = createLoopOp(b, {lb}, {ub}, {step});

  // Tile the loop using IRRewriter
  IRRewriter rewriter(&context);
  rewriter.setInsertionPoint(loopOp);

  SmallVector<acc::LoopOp> loopsToTile = {loopOp};
  SmallVector<Value> tileSizes = {tileSize};

  acc::LoopOp tiledLoop =
      tileACCLoops(loopsToTile, tileSizes, /*defaultTileSize=*/128, rewriter);

  // Verify the tiled loop was created
  EXPECT_TRUE(tiledLoop != nullptr);
  EXPECT_FALSE(tiledLoop.getBody().empty());

  // After tiling a single loop with tile(4), we should have:
  // - 1 tile loop (the outer loop)
  // - 1 element loop nested inside
  // Total: 1 nested loop inside the tile loop
  EXPECT_EQ(countNestedLoops(tiledLoop), 1u);

  // The tile loop (outer) should have 1 IV
  EXPECT_EQ(tiledLoop.getBody().getNumArguments(), 1u);

  // Collect nested loops and verify
  auto nestedLoops = collectNestedLoops(tiledLoop);
  EXPECT_EQ(nestedLoops.size(), 1u);
  // The element loop should have 1 IV
  if (!nestedLoops.empty())
    EXPECT_EQ(nestedLoops[0].getBody().getNumArguments(), 1u);
}

TEST_F(OpenACCUtilsTilingTest, tileACCLoopsNestedLoops) {
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

  // Create loop bounds for outer loop
  Value lb1 =
      arith::ConstantOp::create(b, loc, b.getIndexType(), b.getIndexAttr(0));
  Value ub1 =
      arith::ConstantOp::create(b, loc, b.getIndexType(), b.getIndexAttr(100));
  Value step1 =
      arith::ConstantOp::create(b, loc, b.getIndexType(), b.getIndexAttr(1));

  // Create loop bounds for inner loop
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

  // Create outer loop
  acc::LoopOp outerLoop = createLoopOp(b, {lb1}, {ub1}, {step1});

  // Create inner loop inside outer loop
  b.setInsertionPoint(outerLoop.getBody().getTerminator());
  acc::LoopOp innerLoop = createLoopOp(b, {lb2}, {ub2}, {step2});

  // Tile the loops
  IRRewriter rewriter(&context);
  rewriter.setInsertionPoint(outerLoop);

  SmallVector<acc::LoopOp> loopsToTile = {outerLoop, innerLoop};
  SmallVector<Value> tileSizes = {tileSize1, tileSize2};

  acc::LoopOp tiledLoop =
      tileACCLoops(loopsToTile, tileSizes, /*defaultTileSize=*/128, rewriter);

  // Verify the tiled loop nest was created
  EXPECT_TRUE(tiledLoop != nullptr);
  EXPECT_FALSE(tiledLoop.getBody().empty());

  // After tiling a 2-level nested loop with tile(4,8), we should have:
  // tile_loop_1 -> tile_loop_2 -> element_loop_1 -> element_loop_2
  // Total: 3 nested loops inside the outermost tile loop
  unsigned nestedCount = countNestedLoops(tiledLoop);
  EXPECT_EQ(nestedCount, 3u);

  // The outermost tile loop should have 1 IV
  EXPECT_EQ(tiledLoop.getBody().getNumArguments(), 1u);

  // Collect all nested loops and verify each has 1 IV
  auto nestedLoops = collectNestedLoops(tiledLoop);
  EXPECT_EQ(nestedLoops.size(), 3u);
  for (auto loop : nestedLoops)
    EXPECT_EQ(loop.getBody().getNumArguments(), 1u);
}

//===----------------------------------------------------------------------===//
// uncollapseLoops Tests
//===----------------------------------------------------------------------===//

TEST_F(OpenACCUtilsTilingTest, uncollapseLoopsBasic) {
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

  // Create loop bounds for a collapsed 2-level loop
  Value lb1 =
      arith::ConstantOp::create(b, loc, b.getIndexType(), b.getIndexAttr(0));
  Value ub1 =
      arith::ConstantOp::create(b, loc, b.getIndexType(), b.getIndexAttr(10));
  Value step1 =
      arith::ConstantOp::create(b, loc, b.getIndexType(), b.getIndexAttr(1));
  Value lb2 =
      arith::ConstantOp::create(b, loc, b.getIndexType(), b.getIndexAttr(0));
  Value ub2 =
      arith::ConstantOp::create(b, loc, b.getIndexType(), b.getIndexAttr(20));
  Value step2 =
      arith::ConstantOp::create(b, loc, b.getIndexType(), b.getIndexAttr(1));

  // Create a collapsed loop with 2 IVs
  acc::LoopOp collapsedLoop =
      createLoopOp(b, {lb1, lb2}, {ub1, ub2}, {step1, step2});

  // Set the collapse attribute
  collapsedLoop.setCollapseForDeviceTypes(&context, {acc::DeviceType::None},
                                          llvm::APInt(64, 1));

  // Uncollapse the loop: tileCount=2, collapseCount=1
  IRRewriter rewriter(&context);
  rewriter.setInsertionPoint(collapsedLoop);

  SmallVector<acc::LoopOp> uncollapsedLoops = uncollapseLoops(
      collapsedLoop, /*tileCount=*/2, /*collapseCount=*/1, rewriter);

  // Should produce 2 loops (one outer with collapse=1, one inner)
  EXPECT_EQ(uncollapsedLoops.size(), 2u);

  if (uncollapsedLoops.size() >= 2) {
    // Verify the outer loop has 1 IV (collapseCount=1)
    acc::LoopOp outerLoop = uncollapsedLoops[0];
    EXPECT_EQ(outerLoop.getBody().getNumArguments(), 1u);
    EXPECT_EQ(outerLoop.getLowerbound().size(), 1u);
    EXPECT_EQ(outerLoop.getUpperbound().size(), 1u);
    EXPECT_EQ(outerLoop.getStep().size(), 1u);

    // Verify the inner loop has 1 IV
    acc::LoopOp innerLoop = uncollapsedLoops[1];
    EXPECT_EQ(innerLoop.getBody().getNumArguments(), 1u);
    EXPECT_EQ(innerLoop.getLowerbound().size(), 1u);
    EXPECT_EQ(innerLoop.getUpperbound().size(), 1u);
    EXPECT_EQ(innerLoop.getStep().size(), 1u);

    // Verify nesting: inner loop should be inside outer loop
    unsigned nestedCount = countNestedLoops(outerLoop);
    EXPECT_EQ(nestedCount, 1u);
  }
}

TEST_F(OpenACCUtilsTilingTest, uncollapseLoopsThreeLevels) {
  // Test uncollapsing with 3 levels: collapse(2) with tile(3)
  OwningOpRef<ModuleOp> module = ModuleOp::create(loc);
  Block *moduleBlock = module->getBody();

  OpBuilder::InsertionGuard guard(b);
  b.setInsertionPointToStart(moduleBlock);

  auto funcType = b.getFunctionType({}, {});
  OwningOpRef<func::FuncOp> funcOp =
      func::FuncOp::create(b, loc, "test_func", funcType);
  Block *funcBlock = funcOp->addEntryBlock();

  b.setInsertionPointToStart(funcBlock);

  // Create 3 sets of bounds
  Value lb1 =
      arith::ConstantOp::create(b, loc, b.getIndexType(), b.getIndexAttr(0));
  Value ub1 =
      arith::ConstantOp::create(b, loc, b.getIndexType(), b.getIndexAttr(10));
  Value step1 =
      arith::ConstantOp::create(b, loc, b.getIndexType(), b.getIndexAttr(1));
  Value lb2 =
      arith::ConstantOp::create(b, loc, b.getIndexType(), b.getIndexAttr(0));
  Value ub2 =
      arith::ConstantOp::create(b, loc, b.getIndexType(), b.getIndexAttr(20));
  Value step2 =
      arith::ConstantOp::create(b, loc, b.getIndexType(), b.getIndexAttr(1));
  Value lb3 =
      arith::ConstantOp::create(b, loc, b.getIndexType(), b.getIndexAttr(0));
  Value ub3 =
      arith::ConstantOp::create(b, loc, b.getIndexType(), b.getIndexAttr(30));
  Value step3 =
      arith::ConstantOp::create(b, loc, b.getIndexType(), b.getIndexAttr(1));

  // Create a collapsed loop with 3 IVs
  acc::LoopOp collapsedLoop =
      createLoopOp(b, {lb1, lb2, lb3}, {ub1, ub2, ub3}, {step1, step2, step3});

  // Set collapse(2)
  collapsedLoop.setCollapseForDeviceTypes(&context, {acc::DeviceType::None},
                                          llvm::APInt(64, 2));

  // Uncollapse: tileCount=3, collapseCount=2
  // This should create: outer loop with 2 IVs, then 1 inner loop
  IRRewriter rewriter(&context);
  rewriter.setInsertionPoint(collapsedLoop);

  SmallVector<acc::LoopOp> uncollapsedLoops = uncollapseLoops(
      collapsedLoop, /*tileCount=*/3, /*collapseCount=*/2, rewriter);

  // Should produce 2 loops
  EXPECT_EQ(uncollapsedLoops.size(), 2u);

  if (uncollapsedLoops.size() >= 2) {
    // Outer loop should have 2 IVs (from collapse=2)
    acc::LoopOp outerLoop = uncollapsedLoops[0];
    EXPECT_EQ(outerLoop.getBody().getNumArguments(), 2u);
    EXPECT_EQ(outerLoop.getLowerbound().size(), 2u);

    // Inner loop should have 1 IV (the 3rd dimension)
    acc::LoopOp innerLoop = uncollapsedLoops[1];
    EXPECT_EQ(innerLoop.getBody().getNumArguments(), 1u);
    EXPECT_EQ(innerLoop.getLowerbound().size(), 1u);
  }
}

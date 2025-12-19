//===- OpenACCUtilsLoopTest.cpp - Unit tests for OpenACC loop utilities --===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/OpenACC/OpenACCUtilsLoop.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/OpenACC/OpenACC.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OwningOpRef.h"
#include "mlir/IR/Value.h"
#include "gtest/gtest.h"

using namespace mlir;
using namespace mlir::acc;

//===----------------------------------------------------------------------===//
// Test Fixture
//===----------------------------------------------------------------------===//

class OpenACCUtilsLoopTest : public ::testing::Test {
protected:
  OpenACCUtilsLoopTest() : b(&context), loc(UnknownLoc::get(&context)) {
    context.loadDialect<acc::OpenACCDialect, affine::AffineDialect,
                        arith::ArithDialect, memref::MemRefDialect,
                        func::FuncDialect, scf::SCFDialect,
                        cf::ControlFlowDialect>();
  }

  /// Helper to create an index constant
  Value createIndexConstant(int64_t value) {
    return arith::ConstantOp::create(b, loc, b.getIndexType(),
                                     b.getIndexAttr(value));
  }

  /// Helper to create an i32 constant
  Value createI32Constant(int32_t value) {
    return arith::ConstantOp::create(b, loc, b.getI32Type(),
                                     b.getI32IntegerAttr(value));
  }

  /// Helper to create a simple acc.loop with the given bounds.
  /// Preserves the builder's insertion point.
  acc::LoopOp createLoopOp(ValueRange lbs, ValueRange ubs, ValueRange steps,
                           bool inclusiveUpperbound = true) {
    OpBuilder::InsertionGuard guard(b);

    auto loopOp = acc::LoopOp::create(b, loc, lbs, ubs, steps,
                                      acc::LoopParMode::loop_independent);

    // Set inclusive upper bound attribute
    SmallVector<bool> inclusiveFlags(lbs.size(), inclusiveUpperbound);
    loopOp.setInclusiveUpperboundAttr(b.getDenseBoolArrayAttr(inclusiveFlags));

    // Add body block with IV arguments and yield
    Region &region = loopOp.getRegion();
    Block *block = b.createBlock(&region, region.begin());
    for (Value lb : lbs)
      block->addArgument(lb.getType(), loc);
    b.setInsertionPointToEnd(block);
    acc::YieldOp::create(b, loc);

    return loopOp;
  }

  /// Helper to create an unstructured acc.loop with multiple blocks and ops.
  /// Preserves the builder's insertion point.
  acc::LoopOp createUnstructuredLoopOp(ValueRange lbs, ValueRange ubs,
                                       ValueRange steps) {
    OpBuilder::InsertionGuard guard(b);

    auto loopOp = acc::LoopOp::create(b, loc, lbs, ubs, steps,
                                      acc::LoopParMode::loop_independent);
    loopOp.setInclusiveUpperboundAttr(
        b.getDenseBoolArrayAttr(SmallVector<bool>(lbs.size(), true)));
    loopOp.setUnstructuredAttr(b.getUnitAttr());

    // Create 4 blocks with control flow to test proper replication
    Region &region = loopOp.getRegion();
    Block *entry = b.createBlock(&region, region.begin());
    Block *thenBlock = b.createBlock(&region, region.end());
    Block *elseBlock = b.createBlock(&region, region.end());
    Block *exitBlock = b.createBlock(&region, region.end());

    // Entry block: create a condition and conditional branch
    b.setInsertionPointToEnd(entry);
    Value cond =
        arith::ConstantOp::create(b, loc, b.getI1Type(), b.getBoolAttr(true));
    cf::CondBranchOp::create(b, loc, cond, thenBlock, elseBlock);

    // Then block: create an arith op and branch to exit
    b.setInsertionPointToEnd(thenBlock);
    Value c1 =
        arith::ConstantOp::create(b, loc, b.getIndexType(), b.getIndexAttr(1));
    Value c2 =
        arith::ConstantOp::create(b, loc, b.getIndexType(), b.getIndexAttr(2));
    arith::AddIOp::create(b, loc, c1, c2);
    cf::BranchOp::create(b, loc, exitBlock);

    // Else block: create a different arith op and branch to exit
    b.setInsertionPointToEnd(elseBlock);
    Value c3 =
        arith::ConstantOp::create(b, loc, b.getIndexType(), b.getIndexAttr(3));
    Value c4 =
        arith::ConstantOp::create(b, loc, b.getIndexType(), b.getIndexAttr(4));
    arith::MulIOp::create(b, loc, c3, c4);
    cf::BranchOp::create(b, loc, exitBlock);

    // Exit block: yield
    b.setInsertionPointToEnd(exitBlock);
    acc::YieldOp::create(b, loc);

    return loopOp;
  }

  /// Create a module with a function and set the insertion point in it
  std::pair<OwningOpRef<ModuleOp>, func::FuncOp> createModuleWithFunc() {
    OwningOpRef<ModuleOp> module = ModuleOp::create(loc);
    b.setInsertionPointToStart(module->getBody());

    auto funcType = b.getFunctionType({}, {});
    auto funcOp = func::FuncOp::create(b, loc, "test_func", funcType);
    Block *entryBlock = funcOp.addEntryBlock();
    b.setInsertionPointToStart(entryBlock);

    return {std::move(module), funcOp};
  }

  /// Create a module with a function that has arguments
  std::pair<OwningOpRef<ModuleOp>, func::FuncOp>
  createModuleWithFuncArgs(TypeRange argTypes) {
    OwningOpRef<ModuleOp> module = ModuleOp::create(loc);
    b.setInsertionPointToStart(module->getBody());

    auto funcType = b.getFunctionType(argTypes, {});
    auto funcOp = func::FuncOp::create(b, loc, "test_func", funcType);
    Block *entryBlock = funcOp.addEntryBlock();
    b.setInsertionPointToStart(entryBlock);

    return {std::move(module), funcOp};
  }

  /// Helper to extract constant index value from a Value
  std::optional<int64_t> getConstantIndex(Value v) {
    if (auto constOp = v.getDefiningOp<arith::ConstantIndexOp>())
      return constOp.value();
    if (auto constOp = v.getDefiningOp<arith::ConstantOp>()) {
      if (auto intAttr = dyn_cast<IntegerAttr>(constOp.getValue()))
        return intAttr.getInt();
    }
    return std::nullopt;
  }

  MLIRContext context;
  OpBuilder b;
  Location loc;
};

//===----------------------------------------------------------------------===//
// convertACCLoopToSCFFor Tests
//===----------------------------------------------------------------------===//

TEST_F(OpenACCUtilsLoopTest, ConvertSimpleLoopToSCFFor) {
  auto [module, funcOp] = createModuleWithFunc();

  Value c0 = createIndexConstant(0);
  Value c10 = createIndexConstant(10);
  Value c1 = createIndexConstant(1);

  acc::LoopOp loopOp = createLoopOp({c0}, {c10}, {c1});
  scf::ForOp forOp = convertACCLoopToSCFFor(loopOp, /*enableCollapse=*/false);

  ASSERT_TRUE(forOp);

  // Verify IV type is index
  EXPECT_TRUE(forOp.getInductionVar().getType().isIndex());

  // Verify bounds: lb=0, ub=11 (folded from 10+1), step=1
  auto lbConst = getConstantIndex(forOp.getLowerBound());
  ASSERT_TRUE(lbConst.has_value());
  EXPECT_EQ(*lbConst, 0);

  auto ubConst = getConstantIndex(forOp.getUpperBound());
  ASSERT_TRUE(ubConst.has_value());
  EXPECT_EQ(*ubConst, 11); // inclusive 10 becomes exclusive 11

  auto stepConst = getConstantIndex(forOp.getStep());
  ASSERT_TRUE(stepConst.has_value());
  EXPECT_EQ(*stepConst, 1);

  // Verify the body has a yield terminator
  EXPECT_TRUE(isa<scf::YieldOp>(forOp.getBody()->getTerminator()));
}

TEST_F(OpenACCUtilsLoopTest, ConvertLoopWithI32Bounds) {
  auto [module, funcOp] = createModuleWithFunc();

  Value lb = createI32Constant(0);
  Value ub = createI32Constant(100);
  Value step = createI32Constant(1);

  acc::LoopOp loopOp = createLoopOp({lb}, {ub}, {step});
  scf::ForOp forOp = convertACCLoopToSCFFor(loopOp, /*enableCollapse=*/false);

  ASSERT_TRUE(forOp);

  // IV type should be converted to index
  EXPECT_TRUE(forOp.getInductionVar().getType().isIndex());

  // Bounds should be cast to index type
  EXPECT_TRUE(forOp.getLowerBound().getType().isIndex());
  EXPECT_TRUE(forOp.getUpperBound().getType().isIndex());
  EXPECT_TRUE(forOp.getStep().getType().isIndex());

  // Verify the body has a yield terminator
  EXPECT_TRUE(isa<scf::YieldOp>(forOp.getBody()->getTerminator()));
}

TEST_F(OpenACCUtilsLoopTest, ConvertLoopWithNonConstantBounds) {
  auto [module, funcOp] =
      createModuleWithFuncArgs({b.getIndexType(), b.getIndexType()});
  Block &entryBlock = funcOp.getBody().front();

  Value lb = entryBlock.getArgument(0);
  Value ub = entryBlock.getArgument(1);
  Value step = createIndexConstant(1);

  acc::LoopOp loopOp = createLoopOp({lb}, {ub}, {step});
  scf::ForOp forOp = convertACCLoopToSCFFor(loopOp, /*enableCollapse=*/false);

  ASSERT_TRUE(forOp);

  // Lower bound should be the function argument (no cast needed for index)
  EXPECT_EQ(forOp.getLowerBound(), lb);

  // Upper bound should be ub + 1 (for inclusive -> exclusive conversion)
  // Check it's an addi of ub and 1
  auto ubAddOp = forOp.getUpperBound().getDefiningOp<arith::AddIOp>();
  ASSERT_TRUE(ubAddOp);
  EXPECT_EQ(ubAddOp.getLhs(), ub);
  auto oneConst = getConstantIndex(ubAddOp.getRhs());
  ASSERT_TRUE(oneConst.has_value());
  EXPECT_EQ(*oneConst, 1);

  // Step should be the constant 1
  EXPECT_EQ(forOp.getStep(), step);
}

TEST_F(OpenACCUtilsLoopTest, ConvertLoopToSCFForWithCollapse) {
  auto [module, funcOp] = createModuleWithFunc();

  Value c0 = createIndexConstant(0);
  Value c10 = createIndexConstant(10);
  Value c1 = createIndexConstant(1);

  acc::LoopOp loopOp = createLoopOp({c0, c0}, {c10, c10}, {c1, c1});
  scf::ForOp forOp = convertACCLoopToSCFFor(loopOp, /*enableCollapse=*/true);

  ASSERT_TRUE(forOp);

  // With collapse, there should be NO nested for loops
  bool hasNestedFor = false;
  forOp.getBody()->walk([&](scf::ForOp) { hasNestedFor = true; });
  EXPECT_FALSE(hasNestedFor);

  // The collapsed loop should iterate over the product of dimensions
  // lb=0, step=1 (after collapsing two 0..10 inclusive loops)
  auto lbConst = getConstantIndex(forOp.getLowerBound());
  ASSERT_TRUE(lbConst.has_value());
  EXPECT_EQ(*lbConst, 0);

  auto stepConst = getConstantIndex(forOp.getStep());
  ASSERT_TRUE(stepConst.has_value());
  EXPECT_EQ(*stepConst, 1);

  // Upper bound should be 11*11=121 (product of trip counts)
  // coalesceLoops normalizes the loops, so ub = totalTripCount
  EXPECT_TRUE(forOp.getUpperBound().getType().isIndex());
}

TEST_F(OpenACCUtilsLoopTest, ConvertLoopToSCFForNoCollapse) {
  auto [module, funcOp] = createModuleWithFunc();

  Value c0 = createIndexConstant(0);
  Value c10 = createIndexConstant(10);
  Value c1 = createIndexConstant(1);

  acc::LoopOp loopOp = createLoopOp({c0, c0}, {c10, c10}, {c1, c1});
  scf::ForOp forOp = convertACCLoopToSCFFor(loopOp, /*enableCollapse=*/false);

  ASSERT_TRUE(forOp);

  bool hasNestedFor = false;
  forOp.getBody()->walk([&](scf::ForOp) { hasNestedFor = true; });
  EXPECT_TRUE(hasNestedFor);
}

TEST_F(OpenACCUtilsLoopTest, ConvertLoopToSCFForExclusiveUpperBound) {
  auto [module, funcOp] = createModuleWithFunc();

  Value c0 = createIndexConstant(0);
  Value c10 = createIndexConstant(10);
  Value c1 = createIndexConstant(1);

  acc::LoopOp loopOp =
      createLoopOp({c0}, {c10}, {c1}, /*inclusiveUpperbound=*/false);
  scf::ForOp forOp = convertACCLoopToSCFFor(loopOp, /*enableCollapse=*/false);

  ASSERT_TRUE(forOp);

  // With exclusive upper bound, ub should remain 10 (no +1 adjustment)
  EXPECT_EQ(forOp.getLowerBound(), c0);
  EXPECT_EQ(forOp.getUpperBound(), c10);
  EXPECT_EQ(forOp.getStep(), c1);
}

//===----------------------------------------------------------------------===//
// convertACCLoopToSCFParallel Tests
//===----------------------------------------------------------------------===//

TEST_F(OpenACCUtilsLoopTest, ConvertSimpleLoopToSCFParallel) {
  auto [module, funcOp] = createModuleWithFunc();

  Value c0 = createIndexConstant(0);
  Value c10 = createIndexConstant(10);
  Value c1 = createIndexConstant(1);

  acc::LoopOp loopOp = createLoopOp({c0}, {c10}, {c1});
  scf::ParallelOp parallelOp = convertACCLoopToSCFParallel(loopOp, b);

  ASSERT_TRUE(parallelOp);
  EXPECT_EQ(parallelOp.getNumLoops(), 1u);

  // scf.parallel uses normalized bounds: lb=0, step=1, ub=tripCount
  auto lb = getConstantIndex(parallelOp.getLowerBound()[0]);
  auto step = getConstantIndex(parallelOp.getStep()[0]);
  auto ub = getConstantIndex(parallelOp.getUpperBound()[0]);
  ASSERT_TRUE(lb.has_value());
  ASSERT_TRUE(step.has_value());
  ASSERT_TRUE(ub.has_value());
  EXPECT_EQ(*lb, 0);
  EXPECT_EQ(*step, 1);
  EXPECT_EQ(*ub, 11); // trip count for 0..10 inclusive with step 1

  // Verify IVs are index type
  EXPECT_EQ(parallelOp.getInductionVars().size(), 1u);
  EXPECT_TRUE(parallelOp.getInductionVars()[0].getType().isIndex());
}

TEST_F(OpenACCUtilsLoopTest, ConvertLoopWithI32BoundsToSCFParallel) {
  auto [module, funcOp] = createModuleWithFunc();

  Value lb = createI32Constant(5);
  Value ub = createI32Constant(15);
  Value step = createI32Constant(2);

  acc::LoopOp loopOp = createLoopOp({lb}, {ub}, {step});
  scf::ParallelOp parallelOp = convertACCLoopToSCFParallel(loopOp, b);

  ASSERT_TRUE(parallelOp);
  EXPECT_EQ(parallelOp.getNumLoops(), 1u);

  // All bounds should be index type (converted from i32)
  EXPECT_TRUE(parallelOp.getLowerBound()[0].getType().isIndex());
  EXPECT_TRUE(parallelOp.getUpperBound()[0].getType().isIndex());
  EXPECT_TRUE(parallelOp.getStep()[0].getType().isIndex());

  // Normalized: lb=0, step=1
  // Note: ub is trip count but not folded because index_cast prevents folding
  auto lbConst = getConstantIndex(parallelOp.getLowerBound()[0]);
  auto stepConst = getConstantIndex(parallelOp.getStep()[0]);
  ASSERT_TRUE(lbConst.has_value());
  ASSERT_TRUE(stepConst.has_value());
  EXPECT_EQ(*lbConst, 0);
  EXPECT_EQ(*stepConst, 1);

  // Verify IVs are index type
  EXPECT_TRUE(parallelOp.getInductionVars()[0].getType().isIndex());
}

TEST_F(OpenACCUtilsLoopTest, ConvertLoopWithNonConstantBoundsToSCFParallel) {
  auto [module, funcOp] = createModuleWithFuncArgs(
      {b.getIndexType(), b.getIndexType(), b.getIndexType()});
  Block &entryBlock = funcOp.getBody().front();

  Value lb = entryBlock.getArgument(0);
  Value ub = entryBlock.getArgument(1);
  Value step = entryBlock.getArgument(2);

  acc::LoopOp loopOp = createLoopOp({lb}, {ub}, {step});
  scf::ParallelOp parallelOp = convertACCLoopToSCFParallel(loopOp, b);

  ASSERT_TRUE(parallelOp);
  EXPECT_EQ(parallelOp.getNumLoops(), 1u);

  // Normalized: lb=0, step=1
  auto lbConst = getConstantIndex(parallelOp.getLowerBound()[0]);
  auto stepConst = getConstantIndex(parallelOp.getStep()[0]);
  ASSERT_TRUE(lbConst.has_value());
  ASSERT_TRUE(stepConst.has_value());
  EXPECT_EQ(*lbConst, 0);
  EXPECT_EQ(*stepConst, 1);

  // Upper bound should be computed trip count (not a constant)
  // Verify it's not a simple constant (since bounds are dynamic)
  EXPECT_FALSE(getConstantIndex(parallelOp.getUpperBound()[0]).has_value());
}

TEST_F(OpenACCUtilsLoopTest, ConvertMultiDimLoopToSCFParallel) {
  auto [module, funcOp] = createModuleWithFunc();

  Value c0 = createIndexConstant(0);
  Value c10 = createIndexConstant(10);
  Value c1 = createIndexConstant(1);

  acc::LoopOp loopOp = createLoopOp({c0, c0}, {c10, c10}, {c1, c1});
  scf::ParallelOp parallelOp = convertACCLoopToSCFParallel(loopOp, b);

  ASSERT_TRUE(parallelOp);
  EXPECT_EQ(parallelOp.getNumLoops(), 2u);

  // Both dimensions should have normalized lb=0, step=1, ub=11
  for (unsigned i = 0; i < 2; ++i) {
    auto lb = getConstantIndex(parallelOp.getLowerBound()[i]);
    auto step = getConstantIndex(parallelOp.getStep()[i]);
    auto ub = getConstantIndex(parallelOp.getUpperBound()[i]);

    ASSERT_TRUE(lb.has_value());
    ASSERT_TRUE(step.has_value());
    ASSERT_TRUE(ub.has_value());

    EXPECT_EQ(*lb, 0);
    EXPECT_EQ(*step, 1);
    EXPECT_EQ(*ub, 11); // 0..10 inclusive = 11 iterations
  }

  // Should have 2 induction variables
  EXPECT_EQ(parallelOp.getInductionVars().size(), 2u);
  EXPECT_TRUE(parallelOp.getInductionVars()[0].getType().isIndex());
  EXPECT_TRUE(parallelOp.getInductionVars()[1].getType().isIndex());
}

TEST_F(OpenACCUtilsLoopTest, ConvertLoopWithLargeStepToSCFParallel) {
  auto [module, funcOp] = createModuleWithFunc();

  Value lb = createIndexConstant(0);
  Value ub = createIndexConstant(100);
  Value step = createIndexConstant(10);

  acc::LoopOp loopOp = createLoopOp({lb}, {ub}, {step});
  scf::ParallelOp parallelOp = convertACCLoopToSCFParallel(loopOp, b);

  ASSERT_TRUE(parallelOp);
  EXPECT_EQ(parallelOp.getNumLoops(), 1u);

  // Normalized: lb=0, step=1, ub=tripCount
  auto lbConst = getConstantIndex(parallelOp.getLowerBound()[0]);
  auto stepConst = getConstantIndex(parallelOp.getStep()[0]);
  auto ubConst = getConstantIndex(parallelOp.getUpperBound()[0]);
  ASSERT_TRUE(lbConst.has_value());
  ASSERT_TRUE(stepConst.has_value());
  ASSERT_TRUE(ubConst.has_value());
  EXPECT_EQ(*lbConst, 0);
  EXPECT_EQ(*stepConst, 1);
  EXPECT_EQ(*ubConst, 11); // trip count for 0..100 inclusive with step 10

  // Verify IV is index type
  EXPECT_TRUE(parallelOp.getInductionVars()[0].getType().isIndex());
}

//===----------------------------------------------------------------------===//
// convertUnstructuredACCLoopToSCFExecuteRegion Tests
//===----------------------------------------------------------------------===//

TEST_F(OpenACCUtilsLoopTest, ConvertUnstructuredLoopToExecuteRegion) {
  auto [module, funcOp] = createModuleWithFunc();

  Value c0 = createIndexConstant(0);
  Value c10 = createIndexConstant(10);
  Value c1 = createIndexConstant(1);

  acc::LoopOp loopOp = createUnstructuredLoopOp({c0}, {c10}, {c1});

  // Verify the source loop has 4 blocks
  EXPECT_EQ(loopOp.getRegion().getBlocks().size(), 4u);

  scf::ExecuteRegionOp exeRegionOp =
      convertUnstructuredACCLoopToSCFExecuteRegion(loopOp, b);

  ASSERT_TRUE(exeRegionOp);

  // The execute_region should have 4 blocks replicated from the source
  EXPECT_EQ(exeRegionOp.getRegion().getBlocks().size(), 4u);

  // Verify that the control flow structure is preserved:
  Block &entryBlock = exeRegionOp.getRegion().front();
  EXPECT_TRUE(isa<cf::CondBranchOp>(entryBlock.getTerminator()));

  Block &exitBlock = exeRegionOp.getRegion().back();
  EXPECT_TRUE(isa<scf::YieldOp>(exitBlock.getTerminator()));

  // Count arith operations to verify body was cloned correctly
  unsigned addCount = 0;
  unsigned mulCount = 0;
  exeRegionOp.getRegion().walk([&](arith::AddIOp) { ++addCount; });
  exeRegionOp.getRegion().walk([&](arith::MulIOp) { ++mulCount; });
  EXPECT_EQ(addCount, 1u);
  EXPECT_EQ(mulCount, 1u);
}

TEST_F(OpenACCUtilsLoopTest, ConvertUnstructuredLoopPreservesSuccessors) {
  auto [module, funcOp] = createModuleWithFunc();

  Value c0 = createIndexConstant(0);
  Value c10 = createIndexConstant(10);
  Value c1 = createIndexConstant(1);

  acc::LoopOp loopOp = createUnstructuredLoopOp({c0}, {c10}, {c1});
  scf::ExecuteRegionOp exeRegionOp =
      convertUnstructuredACCLoopToSCFExecuteRegion(loopOp, b);

  ASSERT_TRUE(exeRegionOp);

  Block &entryBlock = exeRegionOp.getRegion().front();
  auto condBranch = dyn_cast<cf::CondBranchOp>(entryBlock.getTerminator());
  ASSERT_TRUE(condBranch);

  // Both successors should exist in the region
  Block *trueDest = condBranch.getTrueDest();
  Block *falseDest = condBranch.getFalseDest();
  EXPECT_TRUE(trueDest->getParent() == &exeRegionOp.getRegion());
  EXPECT_TRUE(falseDest->getParent() == &exeRegionOp.getRegion());
}

//===----------------------------------------------------------------------===//
// Error Case Tests
//===----------------------------------------------------------------------===//

TEST_F(OpenACCUtilsLoopTest, UnstructuredLoopWithYieldOperandsReturnsNullptr) {
  auto [module, funcOp] = createModuleWithFunc();

  Value c0 = createIndexConstant(0);
  Value c10 = createIndexConstant(10);
  Value c1 = createIndexConstant(1);

  // Create an unstructured loop where the yield has operands (simulating
  // a loop with results, which is not yet supported)
  auto loopOp = acc::LoopOp::create(b, loc, {c0}, {c10}, {c1},
                                    acc::LoopParMode::loop_independent);
  loopOp.setInclusiveUpperboundAttr(b.getDenseBoolArrayAttr({true}));
  loopOp.setUnstructuredAttr(b.getUnitAttr());

  // Create multi-block body with yield that has operands
  {
    OpBuilder::InsertionGuard guard(b);
    Region &region = loopOp.getRegion();
    Block *entry = b.createBlock(&region, region.begin());
    Block *exitBlock = b.createBlock(&region, region.end());

    b.setInsertionPointToEnd(entry);
    cf::BranchOp::create(b, loc, exitBlock);

    b.setInsertionPointToEnd(exitBlock);
    // Create a yield with operands - this triggers the error
    Value result = createI32Constant(42);
    acc::YieldOp::create(b, loc, ValueRange{result});
  }
  // InsertionGuard restores insertion point to after loopOp

  // Use a diagnostic handler to capture the error
  std::string errorMsg;
  ScopedDiagnosticHandler handler(&context, [&](Diagnostic &diag) {
    if (diag.getSeverity() == DiagnosticSeverity::Error) {
      llvm::raw_string_ostream os(errorMsg);
      os << diag;
    }
    return success();
  });

  scf::ExecuteRegionOp exeRegionOp =
      convertUnstructuredACCLoopToSCFExecuteRegion(loopOp, b);

  // Should return nullptr due to unsupported loop with results
  EXPECT_FALSE(exeRegionOp);
  EXPECT_TRUE(errorMsg.find("not yet supported") != std::string::npos);
}

//===- OpenACCCGOpsTest.cpp - Unit tests for OpenACC codegen ops ----------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/OpenACC/OpenACC.h"
#include "mlir/Dialect/OpenACC/OpenACCUtilsCG.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OwningOpRef.h"
#include "mlir/IR/PatternMatch.h"
#include "gtest/gtest.h"
#include <optional>

using namespace mlir;
using namespace mlir::acc;

//===----------------------------------------------------------------------===//
// Test Fixture
//===----------------------------------------------------------------------===//

class OpenACCCGOpsTest : public ::testing::Test {
protected:
  OpenACCCGOpsTest() : b(&context), loc(UnknownLoc::get(&context)) {
    context.loadDialect<OpenACCDialect, arith::ArithDialect, func::FuncDialect,
                        gpu::GPUDialect>();
  }

  /// Module with a single no-arg `func.func`, insertion point at the start of
  /// its entry block (ready for host values and `buildComputeRegion` IR).
  struct HostContext {
    OwningOpRef<ModuleOp> module;
    IRRewriter rewriter;
    func::FuncOp func;
    Block *entry = nullptr;

    HostContext(MLIRContext &ctx, Location loc, OpBuilder &builder,
                StringRef funcName = "f")
        : module(ModuleOp::create(builder, loc)), rewriter(&ctx) {
      rewriter.setInsertionPointToEnd(module->getBody());
      func = func::FuncOp::create(rewriter, loc, funcName,
                                  builder.getFunctionType({}, {}));
      entry = func.addEntryBlock();
      rewriter.setInsertionPointToEnd(entry);
      func::ReturnOp::create(rewriter, loc);
      rewriter.setInsertionPointToStart(entry);
    }
  };

  /// Build a single-block region for `buildComputeRegion`: optional one
  /// argument (mapped to `ins`), optional `arith.addi` of that argument with
  /// itself, then `acc.yield`. `regionOut` must be empty.
  static void populateSourceRegionSingleBlock(Region &regionOut,
                                              MLIRContext &ctx, Location loc,
                                              std::optional<Type> mapArgType,
                                              bool addSelfAddi) {
    assert(regionOut.empty() && "expected an empty region");
    Block *block = new Block();
    regionOut.push_back(block);
    OpBuilder regionBuilder(&ctx);
    regionBuilder.setInsertionPointToStart(block);
    if (mapArgType) {
      BlockArgument arg = block->addArgument(*mapArgType, loc);
      if (addSelfAddi)
        arith::AddIOp::create(regionBuilder, loc, arg, arg);
    }
    YieldOp::create(regionBuilder, loc);
  }

  /// Single-block region with an `i32` producer (`arith.constant`) and a user
  /// (`arith.addi`) both inside the region — valid clone source with no
  /// external `ins` captures.
  static void populateSourceRegionWithInternalI32Constant(Region &regionOut,
                                                          MLIRContext &ctx,
                                                          Location loc,
                                                          int64_t cst) {
    assert(regionOut.empty() && "expected an empty region");
    Block *block = new Block();
    regionOut.push_back(block);
    OpBuilder regionBuilder(&ctx);
    regionBuilder.setInsertionPointToStart(block);
    Value k = arith::ConstantIntOp::create(regionBuilder, loc,
                                           IntegerType::get(&ctx, 32), cst);
    arith::AddIOp::create(regionBuilder, loc, k, k);
    YieldOp::create(regionBuilder, loc);
  }

  MLIRContext context;
  OpBuilder b;
  Location loc;
};

//===----------------------------------------------------------------------===//
// ComputeRegionOp::wireHoistedValueThroughIns
//===----------------------------------------------------------------------===//

TEST_F(OpenACCCGOpsTest, WireHoistedValueThroughInsAfterHoisting) {
  HostContext host(context, loc, b);

  Region sourceRegion;
  populateSourceRegionWithInternalI32Constant(sourceRegion, context, loc, 7);
  IRMapping mapping;
  auto cr = buildComputeRegion(loc, /*launchArgs=*/{}, /*inputArgs=*/{},
                               SerialOp::getOperationName(), sourceRegion,
                               host.rewriter, mapping);
  ASSERT_TRUE(cr);

  arith::ConstantIntOp producer;
  arith::AddIOp addOp;
  for (Operation &op : cr.getRegion().front().getOperations()) {
    if (auto c = dyn_cast<arith::ConstantIntOp>(op))
      producer = c;
    else if (auto a = dyn_cast<arith::AddIOp>(op))
      addOp = a;
  }
  ASSERT_TRUE(producer);
  ASSERT_TRUE(addOp);
  Value produced = producer.getResult();
  ASSERT_EQ(addOp.getLhs(), produced);
  ASSERT_EQ(addOp.getRhs(), produced);

  // Hoist the producer out of the region (same idea as ACCImplicitDeclare).
  producer->moveBefore(cr.getOperation());
  ASSERT_TRUE(cr.wireHoistedValueThroughIns(produced).has_value());

  EXPECT_EQ(addOp.getLhs(), addOp.getRhs());
  EXPECT_TRUE(isa<BlockArgument>(addOp.getLhs()));
  EXPECT_TRUE(isa<BlockArgument>(addOp.getRhs()));
  EXPECT_TRUE(succeeded(host.module->verify()));
}

TEST_F(OpenACCCGOpsTest, WireHoistedValueThroughInsNoUseInside) {
  HostContext host(context, loc, b);
  Value v = arith::ConstantIntOp::create(host.rewriter, loc, b.getI32Type(), 1);
  Value w = arith::ConstantIntOp::create(host.rewriter, loc, b.getI32Type(), 2);

  Region sourceRegion;
  populateSourceRegionSingleBlock(sourceRegion, context, loc,
                                  /*mapArgType=*/std::nullopt,
                                  /*addSelfAddi=*/false);
  IRMapping mapping;
  auto cr = buildComputeRegion(loc, /*launchArgs=*/{}, ValueRange(v),
                               SerialOp::getOperationName(), sourceRegion,
                               host.rewriter, mapping);
  ASSERT_TRUE(cr);

  EXPECT_FALSE(cr.wireHoistedValueThroughIns(w).has_value());
  EXPECT_TRUE(succeeded(host.module->verify()));
}

TEST_F(OpenACCCGOpsTest, WireHoistedValueThroughInsDefinedInside) {
  HostContext host(context, loc, b);
  auto c128 = arith::ConstantIndexOp::create(host.rewriter, loc, 128);
  auto threadXDim = GPUParallelDimAttr::threadXDim(&context);
  auto pw = ParWidthOp::create(host.rewriter, loc, c128, threadXDim);

  Region sourceRegion;
  populateSourceRegionSingleBlock(sourceRegion, context, loc,
                                  /*mapArgType=*/std::nullopt,
                                  /*addSelfAddi=*/false);
  IRMapping mapping;
  auto cr = buildComputeRegion(loc, ValueRange(pw.getResult()),
                               /*inputArgs=*/{}, ParallelOp::getOperationName(),
                               sourceRegion, host.rewriter, mapping);
  ASSERT_TRUE(cr);

  BlockArgument launchArg = cr.getRegion().front().getArgument(0);
  EXPECT_FALSE(cr.wireHoistedValueThroughIns(launchArg).has_value());
  EXPECT_TRUE(succeeded(host.module->verify()));
}

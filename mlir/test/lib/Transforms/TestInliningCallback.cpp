//===- TestInliningCallback.cpp - Pass to inline calls in the test dialect
//--------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// This file implements a pass to test inlining callbacks including
// canHandleMultipleBlocks and doClone.
//===----------------------------------------------------------------------===//

#include "TestDialect.h"
#include "TestOps.h"
#include "mlir/Analysis/CallGraph.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/Inliner.h"
#include "mlir/Transforms/InliningUtils.h"
#include "llvm/ADT/StringSet.h"

using namespace mlir;
using namespace test;

namespace {
struct InlinerCallback
    : public PassWrapper<InlinerCallback, OperationPass<func::FuncOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(InlinerCallback)

  StringRef getArgument() const final { return "test-inline-callback"; }
  StringRef getDescription() const final {
    return "Test inlining region calls with call back functions";
  }

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<scf::SCFDialect>();
  }

  static LogicalResult runPipelineHelper(Pass &pass, OpPassManager &pipeline,
                                         Operation *op) {
    return mlir::cast<InlinerCallback>(pass).runPipeline(pipeline, op);
  }

  // Customize the implementation of Inliner::doClone
  // Wrap the callee into scf.execute_region operation
  static void testDoClone(OpBuilder &builder, Region *src, Block *inlineBlock,
                          Block *postInsertBlock, IRMapping &mapper,
                          bool shouldCloneInlinedRegion) {
    // Create a new scf.execute_region operation
    mlir::Operation &call = inlineBlock->back();
    builder.setInsertionPointAfter(&call);

    auto executeRegionOp = mlir::scf::ExecuteRegionOp::create(
        builder, call.getLoc(), call.getResultTypes());
    mlir::Region &region = executeRegionOp.getRegion();

    // Move the inlined blocks into the region
    src->cloneInto(&region, mapper);

    // Split block before scf operation.
    inlineBlock->splitBlock(executeRegionOp.getOperation());

    // Replace all test.return with scf.yield
    for (mlir::Block &block : region) {

      for (mlir::Operation &op : llvm::make_early_inc_range(block)) {
        if (test::TestReturnOp returnOp =
                llvm::dyn_cast<test::TestReturnOp>(&op)) {
          mlir::OpBuilder returnBuilder(returnOp);
          mlir::scf::YieldOp::create(returnBuilder, returnOp.getLoc(),
                                     returnOp.getOperands());
          returnOp.erase();
        }
      }
    }

    // Add test.return after scf.execute_region
    builder.setInsertionPointAfter(executeRegionOp);
    test::TestReturnOp::create(builder, executeRegionOp.getLoc(),
                               executeRegionOp.getResults());
  }

  void runOnOperation() override {
    InlinerConfig config;
    CallGraph &cg = getAnalysis<CallGraph>();

    func::FuncOp function = getOperation();

    // By default, assume that any inlining is profitable.
    auto profitabilityCb = [&](const mlir::Inliner::ResolvedCall &call) {
      return true;
    };

    // Set the clone callback in the config
    config.setCloneCallback([](OpBuilder &builder, Region *src,
                               Block *inlineBlock, Block *postInsertBlock,
                               IRMapping &mapper,
                               bool shouldCloneInlinedRegion) {
      return testDoClone(builder, src, inlineBlock, postInsertBlock, mapper,
                         shouldCloneInlinedRegion);
    });

    // Set canHandleMultipleBlocks to true in the config
    config.setCanHandleMultipleBlocks();

    // Get an instance of the inliner.
    Inliner inliner(function, cg, *this, getAnalysisManager(),
                    runPipelineHelper, config, profitabilityCb);

    // Collect each of the direct function calls within the module.
    SmallVector<func::CallIndirectOp> callers;
    function.walk(
        [&](func::CallIndirectOp caller) { callers.push_back(caller); });

    // Build the inliner interface.
    InlinerInterface interface(&getContext());

    // Try to inline each of the call operations.
    for (auto caller : callers) {
      auto callee = dyn_cast_or_null<FunctionalRegionOp>(
          caller.getCallee().getDefiningOp());
      if (!callee)
        continue;

      // Inline the functional region operation, but only clone the internal
      // region if there is more than one use.
      if (failed(inlineRegion(
              interface, config.getCloneCallback(), &callee.getBody(), caller,
              caller.getArgOperands(), caller.getResults(), caller.getLoc(),
              /*shouldCloneInlinedRegion=*/!callee.getResult().hasOneUse())))
        continue;

      // If the inlining was successful then erase the call and callee if
      // possible.
      caller.erase();
      if (callee.use_empty())
        callee.erase();
    }
  }
};
} // namespace

namespace mlir {
namespace test {
void registerInlinerCallback() { PassRegistration<InlinerCallback>(); }
} // namespace test
} // namespace mlir

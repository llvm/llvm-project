//===- TestLinalgDropUnitDims.cpp - Test Linalg drop unit dims -----------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements a pass for testing the transformation to drop unit
// extent dimensions from `linalg.generic` operations.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

using namespace mlir;

namespace {

LogicalResult dropOutermostUnitDims(RewriterBase &rewriter,
                                    linalg::GenericOp genericOp) {
  linalg::ControlDropUnitDims options;
  options.controlFn = [](Operation *op) { return SmallVector<unsigned>{0}; };
  FailureOr<linalg::DropUnitDimsResult> result =
      linalg::dropUnitDims(rewriter, genericOp, options);
  if (failed(result)) {
    return failure();
  }
  rewriter.replaceOp(genericOp, result->replacements);
  return success();
}

LogicalResult dropOutermostUnitDimsWithEncoding(RewriterBase &rewriter,
                                                linalg::GenericOp genericOp) {
  linalg::ControlDropUnitDims options;
  linalg::ControlDropUnitDims::CollapseFnTy defaultCollapseFn =
      options.collapseFn;
  linalg::ControlDropUnitDims::ExpandFnTy defaultExpandFn = options.expandFn;
  options.controlFn = [](Operation *op) { return SmallVector<unsigned>{0}; };
  options.collapseFn =
      [=](RewriterBase &rewriter, Location loc, Value operand,
          ArrayRef<int64_t> targetShape,
          ArrayRef<ReassociationIndices> reassociation,
          const linalg::ControlDropUnitDims &control) -> FailureOr<Value> {
    if (auto tensorType = dyn_cast<RankedTensorType>(operand.getType())) {
      if (tensorType.getEncoding()) {
        assert(control.rankReductionStrategy ==
                   linalg::ControlDropUnitDims::RankReductionStrategy::
                       ReassociativeReshape &&
               "unexpected rank reduction strategy");
        auto targetType = RankedTensorType::get(
            targetShape, tensorType.getElementType(), tensorType.getEncoding());
        return tensor::CollapseShapeOp::create(rewriter, loc, targetType,
                                               operand, reassociation)
            .getResult();
      }
    }
    return defaultCollapseFn(rewriter, loc, operand, targetShape, reassociation,
                             control);
  };
  options.expandFn =
      [=](RewriterBase &rewriter, Location loc, Value result, Value origDest,
          ArrayRef<ReassociationIndices> reassociation,
          const linalg::ControlDropUnitDims &control) -> FailureOr<Value> {
    if (auto tensorType = dyn_cast<RankedTensorType>(origDest.getType())) {
      if (tensorType.getEncoding()) {
        assert(control.rankReductionStrategy ==
                   linalg::ControlDropUnitDims::RankReductionStrategy::
                       ReassociativeReshape &&
               "unexpected rank reduction strategy");
        return tensor::ExpandShapeOp::create(rewriter, loc, tensorType, result,
                                             reassociation)
            .getResult();
      }
    }
    return defaultExpandFn(rewriter, loc, result, origDest, reassociation,
                           control);
  };

  FailureOr<linalg::DropUnitDimsResult> result =
      linalg::dropUnitDims(rewriter, genericOp, options);
  if (failed(result)) {
    return failure();
  }
  rewriter.replaceOp(genericOp, result->replacements);
  return success();
}

struct TestLinalgDropUnitDims
    : public PassWrapper<TestLinalgDropUnitDims, OperationPass<func::FuncOp>> {

  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(TestLinalgDropUnitDims)

  TestLinalgDropUnitDims() = default;
  TestLinalgDropUnitDims(const TestLinalgDropUnitDims &pass)
      : PassWrapper(pass) {}

  Option<bool> collapseEncoded{
      *this, "collapse-encoded",
      llvm::cl::desc("Collapse and expand tensors with encodings"),
      llvm::cl::init(false)};

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<linalg::LinalgDialect>();
  }

  StringRef getArgument() const final { return "test-linalg-drop-unit-dims"; }

  StringRef getDescriptions() const {
    return "Test transformation to drop unit-extent dims from Linalg "
           "operations";
  }

  void runOnOperation() override {
    MLIRContext *context = &this->getContext();
    func::FuncOp funcOp = this->getOperation();
    IRRewriter rewriter(context);
    SmallVector<linalg::GenericOp> genericOps;
    funcOp.walk(
        [&](linalg::GenericOp genericOp) { genericOps.push_back(genericOp); });

    for (auto genericOp : genericOps) {
      rewriter.setInsertionPoint(genericOp);
      if (collapseEncoded) {
        (void)dropOutermostUnitDimsWithEncoding(rewriter, genericOp);
        continue;
      }
      (void)dropOutermostUnitDims(rewriter, genericOp);
    }
  }
};
} // namespace

namespace mlir {
namespace test {
void registerTestLinalgDropUnitDims() {
  PassRegistration<TestLinalgDropUnitDims>();
}
} // namespace test
} // namespace mlir

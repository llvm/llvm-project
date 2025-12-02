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

  options.controlFn = [](Operation *op) { return SmallVector<unsigned>{0}; };
  options.computeOperandShapeAndMapFn =
      [](const linalg::ControlDropUnitDims &control, MLIRContext *context,
         IndexingMapOpInterface op, OpOperand *opOperand,
         linalg::ControlDropUnitDims::DimensionMapping &oldDimsToNewDimsMap,
         ArrayRef<AffineExpr> dimReplacements)
      -> linalg::ControlDropUnitDims::UnitExtentReplacementInfo {
    auto isCollapsible = [](Type ty) { return isa<RankedTensorType>(ty); };
    auto indexingMap = op.getMatchingIndexingMap(opOperand);
    SmallVector<int64_t> shape = op.getStaticOperandShape(opOperand);
    if (!isCollapsible(opOperand->get().getType())) {
      AffineMap newIndexingMap = indexingMap.replaceDimsAndSymbols(
          dimReplacements, ArrayRef<AffineExpr>{}, oldDimsToNewDimsMap.size(),
          0);
      linalg::ControlDropUnitDims::UnitExtentReplacementInfo info;
      info.indexMap = newIndexingMap;
      info.targetShape = llvm::to_vector(shape);
      return info;
    }
    return control.dropUnitExtentFromOperandMetadata(
        context, op, opOperand, oldDimsToNewDimsMap, dimReplacements);
  };

  // Preserve encoding when collapsing
  options.collapseValueFn =
      [](const linalg::ControlDropUnitDims &control, RewriterBase &rewriter,
         Location loc, Value operand, ArrayRef<int64_t> targetShape,
         ArrayRef<ReassociationIndices> reassociation) -> Value {
    auto tensorType = cast<RankedTensorType>(operand.getType());
    assert(control.rankReductionStrategy ==
               linalg::ControlDropUnitDims::RankReductionStrategy::
                   ReassociativeReshape &&
           "unexpected rank reduction strategy");
    auto targetType = RankedTensorType::get(
        targetShape, tensorType.getElementType(), tensorType.getEncoding());
    return tensor::CollapseShapeOp::create(rewriter, loc, targetType, operand,
                                           reassociation);
  };

  // Attach test attribute to expand operations
  options.expandValueFn =
      [](const linalg::ControlDropUnitDims &control, RewriterBase &rewriter,
         Location loc, Value result, Value origDest,
         ArrayRef<ReassociationIndices> reassociation) -> Value {
    auto origResultType = cast<RankedTensorType>(origDest.getType());
    auto expandOp = tensor::ExpandShapeOp::create(rewriter, loc, origResultType,
                                                  result, reassociation);
    expandOp->setDiscardableAttr("test.unit_dims_expanded",
                                 rewriter.getUnitAttr());
    return expandOp.getResult();
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

  Option<bool> preserveEncoding{
      *this, "collapse-encoded",
      llvm::cl::desc(
          "Collapse tensors with encodings and unit extend dimensions"),
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
      if (preserveEncoding) {
        (void)dropOutermostUnitDimsWithEncoding(rewriter, genericOp);
      } else {
        (void)dropOutermostUnitDims(rewriter, genericOp);
      }
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

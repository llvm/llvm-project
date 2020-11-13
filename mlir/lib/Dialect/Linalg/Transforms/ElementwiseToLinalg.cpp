//===- ElementwiseToLinalg.cpp - conversion of elementwise to linalg ------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Linalg/Passes.h"

#include "PassDetail.h"
#include "mlir/Dialect/Linalg/IR/LinalgOps.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/Transforms/DialectConversion.h"

using namespace mlir;

static bool isElementwiseMappableOpOnRankedTensors(Operation *op) {
  if (!op->hasTrait<OpTrait::ElementwiseMappable>())
    return false;

  // TODO: The conversion pattern can be made to work for `any_of` here, but
  // it's more complex as it requires tracking which operands are scalars.
  return llvm::all_of(op->getOperandTypes(),
                      [](Type type) { return type.isa<RankedTensorType>(); });
}

namespace {
struct ConvertStdElementwiseOpOnRankedTensors : public RewritePattern {
  ConvertStdElementwiseOpOnRankedTensors()
      : RewritePattern(/*benefit=*/1, MatchAnyOpTypeTag()) {}
  LogicalResult matchAndRewrite(Operation *op,
                                PatternRewriter &rewriter) const final {
    if (!isElementwiseMappableOpOnRankedTensors(op))
      return rewriter.notifyMatchFailure(
          op, "requires elementwise op on ranked tensors");

    auto rank = op->getResult(0).getType().cast<RankedTensorType>().getRank();
    SmallVector<AffineMap, 3> indexingMaps(
        op->getNumResults() + op->getNumOperands(),
        rewriter.getMultiDimIdentityMap(rank));
    SmallVector<StringRef, 6> iteratorTypes(rank,
                                            getParallelIteratorTypeName());
    rewriter.replaceOpWithNewOp<linalg::GenericOp>(
        op, /*resultTensorTypes=*/op->getResultTypes(),
        /*inputs=*/op->getOperands(),
        /*outputBuffers=*/ValueRange(),
        /*initTensors=*/ValueRange(),
        /*indexingMaps=*/indexingMaps,
        /*iteratorTypes=*/iteratorTypes,
        /*bodyBuilder=*/
        [&](OpBuilder &builder, Location loc, ValueRange regionArgs) {
          OperationState state(loc, op->getName());
          state.addAttributes(op->getAttrs());
          state.addOperands(regionArgs);
          auto resultTypes = llvm::to_vector<6>(
              llvm::map_range(op->getResultTypes(), [](Type type) {
                return type.cast<TensorType>().getElementType();
              }));
          state.addTypes(resultTypes);
          auto *scalarOp = builder.createOperation(state);
          builder.create<linalg::YieldOp>(loc, scalarOp->getResults());
        });
    return success();
  }
};
} // namespace

void mlir::populateElementwiseToLinalgConversionPatterns(
    OwningRewritePatternList &patterns, MLIRContext *) {
  patterns.insert<ConvertStdElementwiseOpOnRankedTensors>();
}

namespace {
class ConvertElementwiseToLinalgPass
    : public ConvertElementwiseToLinalgBase<ConvertElementwiseToLinalgPass> {

  void runOnFunction() final {
    auto func = getOperation();
    auto *context = &getContext();
    ConversionTarget target(*context);
    OwningRewritePatternList patterns;

    populateElementwiseToLinalgConversionPatterns(patterns, context);
    target.markUnknownOpDynamicallyLegal([](Operation *op) {
      return !isElementwiseMappableOpOnRankedTensors(op);
    });

    if (failed(applyPartialConversion(func, target, std::move(patterns))))
      signalPassFailure();
  }
};
} // namespace

std::unique_ptr<OperationPass<FuncOp>>
mlir::createConvertElementwiseToLinalgPass() {
  return std::make_unique<ConvertElementwiseToLinalgPass>();
}

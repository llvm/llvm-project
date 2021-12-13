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
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Dialect/Linalg/Utils/Utils.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/Dialect/StandardOps/Utils/Utils.h"
#include "mlir/Transforms/DialectConversion.h"

using namespace mlir;

static bool isElementwiseMappableOpOnRankedTensors(Operation *op) {
  if (!OpTrait::hasElementwiseMappableTraits(op))
    return false;

  // TODO: The conversion pattern can be made to work for `any_of` here, but
  // it's more complex as it requires tracking which operands are scalars.
  return llvm::all_of(op->getOperandTypes(),
                      [](Type type) { return type.isa<RankedTensorType>(); });
}

/// Given `op` assumed `isElementwiseMappableOpOnRankedTensors`, iterate over
/// the result types and return a list of values such that, for each result type
/// `t` and value `v` at the same index `idx`:
///   1. `v.getType() == t`
///   2. If an operand of `op` has type `t`, let `operand_first` be the first
///      such operand. Then`v == operand_first`.
///   3. Otherwise, v is a newly created `linalg::InitTensorOp` with:
///        a. Static and dynamic dims extracted from the first operand of `op`.
///        b. Elemental type equal to the elemental type of `t`.
///
/// This is sufficient because ElementwiseMappable guarantees that "The static
/// types of all vector (resp. tensor) operands and results must have the same
/// shape".
static SmallVector<Value, 4>
getOrCreateOperandsMatchingResultTypes(OpBuilder &b, Operation *op) {
  assert(isElementwiseMappableOpOnRankedTensors(op));
  Location loc = op->getLoc();
  ValueRange operands = op->getOperands();
  TypeRange rankedTensorTypes = op->getResultTypes();
  SmallVector<Value, 4> res;
  res.reserve(rankedTensorTypes.size());
  for (Type t : rankedTensorTypes) {
    // Try to find an operand with type matching the result tensor.
    bool found = false;
    for (Value v : operands) {
      if (v.getType() == t) {
        found = true;
        res.push_back(v);
        break;
      }
    }
    if (found)
      continue;

    // Extract static / dynamic shape mix from the first operand.
    Value firstOperand = operands.front();
    auto rankedTensorType = t.cast<RankedTensorType>();
    auto staticShape = llvm::to_vector<4>(rankedTensorType.getShape());
    auto dynamicShape = linalg::getDynOperands(loc, firstOperand, b);

    res.push_back(b.create<linalg::InitTensorOp>(
        loc, dynamicShape, staticShape, rankedTensorType.getElementType()));
  }
  return res;
}

namespace {
struct ConvertAnyElementwiseMappableOpOnRankedTensors : public RewritePattern {
  ConvertAnyElementwiseMappableOpOnRankedTensors(MLIRContext *context)
      : RewritePattern(MatchAnyOpTypeTag(), /*benefit=*/1, context) {}
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
    auto outputs = getOrCreateOperandsMatchingResultTypes(rewriter, op);
    rewriter.replaceOpWithNewOp<linalg::GenericOp>(
        op, /*resultTensorTypes=*/op->getResultTypes(),
        /*inputs=*/op->getOperands(),
        /*outputs=*/outputs,
        /*indexingMaps=*/indexingMaps,
        /*iteratorTypes=*/iteratorTypes,
        /*bodyBuilder=*/
        [&](OpBuilder &builder, Location loc, ValueRange regionArgs) {
          OperationState state(loc, op->getName());
          state.addAttributes(op->getAttrs());
          // Only take the input operands in the cloned elementwise op.
          state.addOperands(regionArgs.take_front(op->getNumOperands()));
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

void mlir::linalg::populateElementwiseToLinalgConversionPatterns(
    RewritePatternSet &patterns) {
  patterns.add<ConvertAnyElementwiseMappableOpOnRankedTensors>(
      patterns.getContext());
}

namespace {
class ConvertElementwiseToLinalgPass
    : public ConvertElementwiseToLinalgBase<ConvertElementwiseToLinalgPass> {

  void runOnFunction() final {
    auto func = getOperation();
    auto *context = &getContext();
    ConversionTarget target(*context);
    RewritePatternSet patterns(context);

    mlir::linalg::populateElementwiseToLinalgConversionPatterns(patterns);
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

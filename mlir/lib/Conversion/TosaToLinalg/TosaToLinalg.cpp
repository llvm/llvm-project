//===- TosaToLinalg.cpp - Lowering Tosa to Linalg Dialect -----------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// These rewriters lower from the Tosa to the Linalg dialect.
//
//===----------------------------------------------------------------------===//

#include "mlir/Conversion/TosaToLinalg/TosaToLinalg.h"
#include "mlir/Dialect/Linalg/IR/LinalgOps.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/Dialect/Tosa/IR/TosaOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

using namespace mlir;

static SmallVector<StringRef> getNParallelLoopsAttrs(unsigned nParallelLoops) {
  return SmallVector<StringRef>(nParallelLoops, getParallelIteratorTypeName());
}

static Value
createLinalgBodyCalculationForElementwiseOp(Operation *op, ValueRange args,
                                            ArrayRef<Type> resultTypes,
                                            PatternRewriter &rewriter) {
  Location loc = op->getLoc();
  auto elementTy =
      op->getOperand(0).getType().cast<ShapedType>().getElementType();

  // tosa::AbsOp
  if (isa<tosa::AbsOp>(op) && elementTy.isa<FloatType>())
    return rewriter.create<mlir::AbsFOp>(loc, resultTypes, args);

  // tosa::AddOp
  if (isa<tosa::AddOp>(op) && elementTy.isa<FloatType>())
    return rewriter.create<mlir::AddFOp>(loc, resultTypes, args);

  if (isa<tosa::AddOp>(op) && elementTy.isa<IntegerType>())
    return rewriter.create<mlir::AddIOp>(loc, resultTypes, args);

  // tosa::BitwiseAndOp
  if (isa<tosa::BitwiseAndOp>(op) && elementTy.isa<IntegerType>())
    return rewriter.create<mlir::AndOp>(loc, resultTypes, args);

  // tosa::BitwiseOrOp
  if (isa<tosa::BitwiseOrOp>(op) && elementTy.isa<IntegerType>())
    return rewriter.create<mlir::OrOp>(loc, resultTypes, args);

  // tosa::BitwiseXOrOp
  if (isa<tosa::BitwiseXorOp>(op) && elementTy.isa<IntegerType>())
    return rewriter.create<mlir::XOrOp>(loc, resultTypes, args);

  // tosa::LogicalLeftShiftOp
  if (isa<tosa::LogicalLeftShiftOp>(op) && elementTy.isa<IntegerType>())
    return rewriter.create<mlir::ShiftLeftOp>(loc, resultTypes, args);

  // tosa::LogicalrightShiftOp
  if (isa<tosa::LogicalRightShiftOp>(op) && elementTy.isa<IntegerType>())
    return rewriter.create<mlir::UnsignedShiftRightOp>(loc, resultTypes, args);

  // tosa::PowOp
  if (isa<tosa::PowOp>(op) && elementTy.isa<FloatType>())
    return rewriter.create<mlir::PowFOp>(loc, resultTypes, args);

  // tosa::LogOp
  if (isa<tosa::LogOp>(op) && elementTy.isa<FloatType>())
    return rewriter.create<mlir::LogOp>(loc, resultTypes, args);

  // tosa::ExpOp
  if (isa<tosa::ExpOp>(op) && elementTy.isa<FloatType>())
    return rewriter.create<mlir::ExpOp>(loc, resultTypes, args);

  // tosa::SubOp
  if (isa<tosa::SubOp>(op) && elementTy.isa<FloatType>())
    return rewriter.create<mlir::SubFOp>(loc, resultTypes, args);

  if (isa<tosa::SubOp>(op) && elementTy.isa<IntegerType>())
    return rewriter.create<mlir::SubIOp>(loc, resultTypes, args);

  // tosa::TanhOp
  if (isa<tosa::TanhOp>(op) && elementTy.isa<FloatType>())
    return rewriter.create<mlir::TanhOp>(loc, resultTypes, args);

  // tosa::GreaterOp
  if (isa<tosa::GreaterOp>(op) && elementTy.isa<FloatType>())
    return rewriter.create<mlir::CmpFOp>(loc, CmpFPredicate::OGT, args[0],
                                         args[1]);

  if (isa<tosa::GreaterOp>(op) && elementTy.isSignlessInteger())
    return rewriter.create<mlir::CmpIOp>(loc, CmpIPredicate::sgt, args[0],
                                         args[1]);

  // tosa::GreaterEqualOp
  if (isa<tosa::GreaterEqualOp>(op) && elementTy.isa<FloatType>())
    return rewriter.create<mlir::CmpFOp>(loc, CmpFPredicate::OGE, args[0],
                                         args[1]);

  if (isa<tosa::GreaterEqualOp>(op) && elementTy.isSignlessInteger())
    return rewriter.create<mlir::CmpIOp>(loc, CmpIPredicate::sge, args[0],
                                         args[1]);

  // tosa::MaximumOp
  if (isa<tosa::MaximumOp>(op) && elementTy.isa<FloatType>()) {
    auto predicate = rewriter.create<mlir::CmpFOp>(loc, CmpFPredicate::OGT,
                                                   args[0], args[1]);
    return rewriter.create<mlir::SelectOp>(loc, predicate, args[0], args[1]);
  }

  if (isa<tosa::MaximumOp>(op) && elementTy.isSignlessInteger()) {
    auto predicate = rewriter.create<mlir::CmpIOp>(loc, CmpIPredicate::sgt,
                                                   args[0], args[1]);
    return rewriter.create<mlir::SelectOp>(loc, predicate, args[0], args[1]);
  }

  // tosa::MinimumOp
  if (isa<tosa::MinimumOp>(op) && elementTy.isa<FloatType>()) {
    auto predicate = rewriter.create<mlir::CmpFOp>(loc, CmpFPredicate::OLT,
                                                   args[0], args[1]);
    return rewriter.create<mlir::SelectOp>(loc, predicate, args[0], args[1]);
  }

  if (isa<tosa::MinimumOp>(op) && elementTy.isSignlessInteger()) {
    auto predicate = rewriter.create<mlir::CmpIOp>(loc, CmpIPredicate::slt,
                                                   args[0], args[1]);
    return rewriter.create<mlir::SelectOp>(loc, predicate, args[0], args[1]);
  }

  // tosa::CeilOp
  if (isa<tosa::CeilOp>(op) && elementTy.isa<FloatType>())
    return rewriter.create<mlir::CeilFOp>(loc, resultTypes, args);

  // tosa::FloorOp
  if (isa<tosa::FloorOp>(op) && elementTy.isa<FloatType>())
    return rewriter.create<mlir::FloorFOp>(loc, resultTypes, args);

  (void)rewriter.notifyMatchFailure(
      op, "unhandled op for linalg body calculation for elementwise op");
  return nullptr;
}

static LogicalResult
elementwiseMatchAndRewriteHelper(Operation *operation,
                                 PatternRewriter &rewriter) {
  auto loc = operation->getLoc();
  auto results = operation->getResults();
  auto t0 = operation->getOperand(0).getType().template dyn_cast<ShapedType>();
  if (!t0)
    return rewriter.notifyMatchFailure(operation,
                                       "All results must be a shaped type");

  // For now require no broadcasting. Consider making it support broadcasting
  // operations.
  Type uniqueInTy = operation->getOperand(0).getType();
  bool allInputTypesEqual =
      llvm::all_of(operation->getOperandTypes(),
                   [&](Type operandTy) { return operandTy == uniqueInTy; });
  if (!allInputTypesEqual)
    return rewriter.notifyMatchFailure(operation,
                                       "All operands must have the same type");
  bool resultAndInputShapeEqual =
      llvm::all_of(operation->getResultTypes(), [&](Type resultTy) {
        return resultTy.cast<ShapedType>().getShape() == t0.getShape();
      });

  if (!resultAndInputShapeEqual)
    return rewriter.notifyMatchFailure(
        operation, "All results must have the same shape as the input");

  // Construct the indexing maps needed for linalg.generic ops.
  SmallVector<Type> bodyArgTypes;

  for (Value in : operation->getOperands())
    bodyArgTypes.emplace_back(getElementTypeOrSelf(in.getType()));

  SmallVector<Type> opResultTypes;
  SmallVector<Value> initTensors;
  for (auto result : results) {
    auto resultType = result.getType().template cast<ShapedType>();
    if (!resultType.hasStaticShape())
      return rewriter.notifyMatchFailure(
          operation,
          "tosa to linalg conversion expects statically shaped tensors");

    initTensors.push_back(rewriter.create<linalg::InitTensorOp>(
        loc, ArrayRef<Value>({}), resultType.getShape(),
        resultType.getElementType()));
    opResultTypes.push_back(result.getType());
  }

  auto bodyResultTypes = llvm::to_vector<4>(llvm::map_range(
      initTensors, [](Value v) { return getElementTypeOrSelf(v); }));

  // Supports only non-broadcasted operation. Shoudl consider update indexing
  // map to be multidimensional.
  unsigned nloops = t0.getRank();
  AffineMap commonIndexingMap = rewriter.getMultiDimIdentityMap(nloops);
  SmallVector<AffineMap, 2> indexingMaps(
      operation->getNumOperands() + bodyResultTypes.size(), commonIndexingMap);

  bool didEncounterError = false;
  auto linalgOp = rewriter.create<linalg::GenericOp>(
      loc, opResultTypes, operation->getOperands(), initTensors, indexingMaps,
      getNParallelLoopsAttrs(nloops),
      [&](OpBuilder &nestedBuilder, Location nestedLoc, ValueRange blockArgs) {
        Value opResult = createLinalgBodyCalculationForElementwiseOp(
            operation, blockArgs.take_front(operation->getNumOperands()),
            bodyResultTypes, rewriter);
        if (opResult) {
          didEncounterError = true;
        }
        nestedBuilder.create<linalg::YieldOp>(loc, opResult);
      });

  if (!didEncounterError)
    return failure();

  rewriter.replaceOp(operation, linalgOp->getResults());
  return success();
}

namespace {

template <typename SrcOp>
class PointwiseConverter : public OpRewritePattern<SrcOp> {
public:
  using OpRewritePattern<SrcOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(SrcOp op,
                                PatternRewriter &rewriter) const final {
    return elementwiseMatchAndRewriteHelper(op, rewriter);
  }
};

} // namespace

void mlir::tosa::populateTosaToLinalgOnTensorsConversionPatterns(
    MLIRContext *context, OwningRewritePatternList *patterns) {
  patterns->insert<
      PointwiseConverter<tosa::AddOp>, PointwiseConverter<tosa::SubOp>,
      PointwiseConverter<tosa::PowOp>, PointwiseConverter<tosa::LogOp>,
      PointwiseConverter<tosa::ExpOp>, PointwiseConverter<tosa::AbsOp>,
      PointwiseConverter<tosa::TanhOp>, PointwiseConverter<tosa::BitwiseAndOp>,
      PointwiseConverter<tosa::BitwiseOrOp>,
      PointwiseConverter<tosa::BitwiseXorOp>,
      PointwiseConverter<tosa::LogicalLeftShiftOp>,
      PointwiseConverter<tosa::LogicalRightShiftOp>,
      PointwiseConverter<tosa::GreaterOp>,
      PointwiseConverter<tosa::GreaterEqualOp>,
      PointwiseConverter<tosa::MaximumOp>, PointwiseConverter<tosa::MinimumOp>,
      PointwiseConverter<tosa::CeilOp>, PointwiseConverter<tosa::FloorOp>>(
      context);
}

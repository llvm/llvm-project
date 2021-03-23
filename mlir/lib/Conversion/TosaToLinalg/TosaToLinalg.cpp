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
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/Dialect/Tosa/IR/TosaOps.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#include <numeric>

using namespace mlir;

static SmallVector<StringRef> getNParallelLoopsAttrs(unsigned nParallelLoops) {
  return SmallVector<StringRef>(nParallelLoops, getParallelIteratorTypeName());
}

template <typename T>
static mlir::ConstantOp
createConstFromIntAttribute(Operation *op, std::string attrName,
                            Type requiredAttrType, OpBuilder &rewriter) {
  auto castedN = static_cast<T>(
      op->getAttr(attrName).cast<IntegerAttr>().getValue().getSExtValue());
  return rewriter.create<mlir::ConstantOp>(
      op->getLoc(), IntegerAttr::get(requiredAttrType, castedN));
}

template <typename T>
static void getValuesFromIntArrayAttribute(ArrayAttr attr,
                                           SmallVector<T> &arrayValues) {
  for (Attribute val : attr.getValue()) {
    arrayValues.push_back(val.cast<IntegerAttr>().getValue().getSExtValue());
  }
}

// Generates an affine map for parallel operations on a given type. This
// performs implicit broadcasting across any dimension of size-1.
static AffineMap createAffineMapForType(ShapedType type,
                                        PatternRewriter &rewriter) {
  unsigned rank = type.getRank();
  auto shape = type.getShape();
  SmallVector<AffineExpr, 4> dimExprs;
  dimExprs.reserve(rank);
  for (unsigned i = 0; i < rank; ++i) {
    // If the dimension is one we can broadcast the input with a constant
    // affine expression.
    if (shape[i] == 1)
      dimExprs.push_back(rewriter.getAffineConstantExpr(0));
    else
      dimExprs.push_back(rewriter.getAffineDimExpr(i));
  }
  return AffineMap::get(/*dimCount=*/rank, /*symbolCount=*/0, dimExprs,
                        rewriter.getContext());
}

template <typename T, typename P>
static mlir::SelectOp clampHelper(Location loc, Value arg, mlir::ConstantOp min,
                                  mlir::ConstantOp max, P pred,
                                  OpBuilder &rewriter) {
  auto smallerThanMin = rewriter.create<T>(loc, pred, arg, min);
  auto minOrArg =
      rewriter.create<mlir::SelectOp>(loc, smallerThanMin, min, arg);
  auto largerThanMax = rewriter.create<T>(loc, pred, max, arg);
  return rewriter.create<mlir::SelectOp>(loc, largerThanMax, max, minOrArg);
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

  // tosa::SubOp
  if (isa<tosa::SubOp>(op) && elementTy.isa<FloatType>())
    return rewriter.create<mlir::SubFOp>(loc, resultTypes, args);

  if (isa<tosa::SubOp>(op) && elementTy.isa<IntegerType>())
    return rewriter.create<mlir::SubIOp>(loc, resultTypes, args);

  // tosa::MulOp
  if (isa<tosa::MulOp>(op) && elementTy.isa<FloatType>()) {
    if (dyn_cast<tosa::MulOp>(op).shift() != 0) {
      (void)rewriter.notifyMatchFailure(op,
                                        "Cannot have shift value for float");
      return nullptr;
    }
    return rewriter.create<mlir::MulFOp>(loc, resultTypes, args);
  }

  if (isa<tosa::MulOp>(op) && elementTy.isa<IntegerType>()) {
    Value a = args[0];
    Value b = args[1];
    auto shift =
        op->getAttr("shift").cast<IntegerAttr>().getValue().getSExtValue();
    if (shift > 0) {
      auto shiftConst =
          rewriter.create<ConstantIntOp>(loc, shift, /*bitwidth=*/8);
      if (!a.getType().isInteger(32))
        a = rewriter.create<SignExtendIOp>(loc, rewriter.getI32Type(), a);

      if (!b.getType().isInteger(32))
        b = rewriter.create<SignExtendIOp>(loc, rewriter.getI32Type(), b);

      auto result = rewriter.create<tosa::ApplyScaleOp>(
          loc, rewriter.getI32Type(), a, b, shiftConst,
          rewriter.getBoolAttr(false));

      if (elementTy.isInteger(32))
        return result;

      return rewriter.create<TruncateIOp>(loc, elementTy, result);
    }

    int aWidth = a.getType().getIntOrFloatBitWidth();
    int bWidth = b.getType().getIntOrFloatBitWidth();
    int cWidth = resultTypes[0].getIntOrFloatBitWidth();

    if (aWidth < cWidth)
      a = rewriter.create<SignExtendIOp>(loc, resultTypes[0], a);
    if (bWidth < cWidth)
      b = rewriter.create<SignExtendIOp>(loc, resultTypes[0], b);

    return rewriter.create<mlir::MulIOp>(loc, resultTypes, a, b);
  }

  // tosa::NegateOp
  if (isa<tosa::NegateOp>(op) && elementTy.isa<IntegerType>()) {
    auto constant =
        rewriter.create<mlir::ConstantOp>(loc, IntegerAttr::get(elementTy, -1));
    return rewriter.create<mlir::MulIOp>(loc, resultTypes, args[0], constant);
  }

  if (isa<tosa::NegateOp>(op) && elementTy.isa<FloatType>())
    return rewriter.create<mlir::NegFOp>(loc, resultTypes, args);

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

  // tosa::LogicalRightShiftOp
  if (isa<tosa::LogicalRightShiftOp>(op) && elementTy.isa<IntegerType>())
    return rewriter.create<mlir::UnsignedShiftRightOp>(loc, resultTypes, args);

  // tosa::LogicalAnd
  if (isa<tosa::LogicalAndOp>(op) && elementTy.isInteger(1))
    return rewriter.create<mlir::AndOp>(loc, resultTypes, args);

  // tosa::LogicalNot
  if (isa<tosa::LogicalNotOp>(op) && elementTy.isInteger(1)) {
    auto one = rewriter.create<mlir::ConstantOp>(
        loc, rewriter.getIntegerAttr(elementTy, 1));
    return rewriter.create<mlir::XOrOp>(loc, resultTypes, args[0], one);
  }

  // tosa::LogicalOr
  if (isa<tosa::LogicalOrOp>(op) && elementTy.isInteger(1))
    return rewriter.create<mlir::OrOp>(loc, resultTypes, args);

  // tosa::LogicalXor
  if (isa<tosa::LogicalXorOp>(op) && elementTy.isInteger(1))
    return rewriter.create<mlir::XOrOp>(loc, resultTypes, args);

  // tosa::PowOp
  if (isa<tosa::PowOp>(op) && elementTy.isa<FloatType>())
    return rewriter.create<mlir::math::PowFOp>(loc, resultTypes, args);

  // tosa::RsqrtOp
  if (isa<tosa::RsqrtOp>(op) && elementTy.isa<FloatType>())
    return rewriter.create<mlir::math::RsqrtOp>(loc, resultTypes, args);

  // tosa::LogOp
  if (isa<tosa::LogOp>(op) && elementTy.isa<FloatType>())
    return rewriter.create<mlir::math::LogOp>(loc, resultTypes, args);

  // tosa::ExpOp
  if (isa<tosa::ExpOp>(op) && elementTy.isa<FloatType>())
    return rewriter.create<mlir::math::ExpOp>(loc, resultTypes, args);

  // tosa::TanhOp
  if (isa<tosa::TanhOp>(op) && elementTy.isa<FloatType>())
    return rewriter.create<mlir::math::TanhOp>(loc, resultTypes, args);

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

  // tosa::SelectOp
  if (isa<tosa::SelectOp>(op)) {
    elementTy = op->getOperand(1).getType().cast<ShapedType>().getElementType();
    if (elementTy.isa<FloatType>() || elementTy.isa<IntegerType>())
      return rewriter.create<mlir::SelectOp>(loc, args[0], args[1], args[2]);
  }

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

  // tosa::ClampOp
  if (isa<tosa::ClampOp>(op) && elementTy.isa<FloatType>()) {
    auto min = rewriter.create<mlir::ConstantOp>(loc, elementTy,
                                                 op->getAttr("min_fp"));
    auto max = rewriter.create<mlir::ConstantOp>(loc, elementTy,
                                                 op->getAttr("max_fp"));
    return clampHelper<mlir::CmpFOp>(loc, args[0], min, max, CmpFPredicate::OLT,
                                     rewriter);
  }

  if (isa<tosa::ClampOp>(op) && elementTy.isa<IntegerType>()) {
    auto min = createConstFromIntAttribute<int32_t>(op, "min_int", elementTy,
                                                    rewriter);
    auto max = createConstFromIntAttribute<int32_t>(op, "max_int", elementTy,
                                                    rewriter);
    return clampHelper<mlir::CmpIOp>(loc, args[0], min, max, CmpIPredicate::slt,
                                     rewriter);
  }

  // tosa::ReluNOp
  if (isa<tosa::ReluNOp>(op) && elementTy.isa<FloatType>()) {
    auto zero =
        rewriter.create<mlir::ConstantOp>(loc, FloatAttr::get(elementTy, 0));
    auto n = rewriter.create<mlir::ConstantOp>(loc, elementTy,
                                               op->getAttr("max_fp"));
    return clampHelper<mlir::CmpFOp>(loc, args[0], zero, n, CmpFPredicate::OLT,
                                     rewriter);
  }

  if (isa<tosa::ReluNOp>(op) && elementTy.isa<IntegerType>()) {
    auto zero =
        rewriter.create<mlir::ConstantOp>(loc, IntegerAttr::get(elementTy, 0));
    auto n = createConstFromIntAttribute<int32_t>(op, "max_int", elementTy,
                                                  rewriter);
    return clampHelper<mlir::CmpIOp>(loc, args[0], zero, n, CmpIPredicate::slt,
                                     rewriter);
  }

  // tosa::CastOp
  if (isa<tosa::CastOp>(op)) {
    Type srcTy = elementTy;
    Type dstTy = resultTypes.front();
    bool bitExtend =
        srcTy.getIntOrFloatBitWidth() < dstTy.getIntOrFloatBitWidth();

    if (srcTy == dstTy)
      return args.front();

    if (srcTy.isa<FloatType>() && dstTy.isa<FloatType>() && bitExtend)
      return rewriter.create<mlir::FPExtOp>(loc, resultTypes, args, mlir::None);

    if (srcTy.isa<FloatType>() && dstTy.isa<FloatType>() && !bitExtend)
      return rewriter.create<mlir::FPTruncOp>(loc, resultTypes, args,
                                              mlir::None);

    // 1-bit integers need to be treated as signless.
    if (srcTy.isInteger(1) && mlir::UIToFPOp::areCastCompatible(srcTy, dstTy))
      return rewriter.create<mlir::UIToFPOp>(loc, resultTypes, args,
                                             mlir::None);

    if (srcTy.isInteger(1) && dstTy.isa<IntegerType>() && bitExtend)
      return rewriter.create<mlir::ZeroExtendIOp>(loc, resultTypes, args,
                                                  mlir::None);

    // All other si-to-fp conversions should be handled by SIToFP.
    if (mlir::SIToFPOp::areCastCompatible(srcTy, dstTy))
      return rewriter.create<mlir::SIToFPOp>(loc, resultTypes, args,
                                             mlir::None);

    // Casting to boolean, floats need to only be checked as not-equal to zero.
    if (srcTy.isa<FloatType>() && dstTy.isInteger(1)) {
      Value zero =
          rewriter.create<ConstantOp>(loc, rewriter.getFloatAttr(srcTy, 0.0));
      return rewriter.create<mlir::CmpFOp>(loc, CmpFPredicate::UNE,
                                           args.front(), zero);
    }

    if (mlir::FPToSIOp::areCastCompatible(srcTy, dstTy))
      return rewriter.create<mlir::FPToSIOp>(loc, resultTypes, args,
                                             mlir::None);

    // Casting to boolean, integers need to only be checked as not-equal to
    // zero.
    if (srcTy.isa<IntegerType>() && dstTy.isInteger(1)) {
      Value zero =
          rewriter.create<ConstantIntOp>(loc, 0, srcTy.getIntOrFloatBitWidth());
      return rewriter.create<mlir::CmpIOp>(loc, CmpIPredicate::ne, args.front(),
                                           zero);
    }

    if (srcTy.isa<IntegerType>() && dstTy.isa<IntegerType>() && bitExtend)
      return rewriter.create<mlir::SignExtendIOp>(loc, resultTypes, args,
                                                  mlir::None);

    if (srcTy.isa<IntegerType>() && dstTy.isa<IntegerType>() && !bitExtend)
      return rewriter.create<mlir::TruncateIOp>(loc, resultTypes, args,
                                                mlir::None);
  }

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

  assert(operation->getNumResults() == 1 &&
         "All TOSA elementwise ops should only return a single result.");

  // Construct the indexing maps needed for linalg.generic ops.
  SmallVector<Type> bodyArgTypes;

  for (Value in : operation->getOperands())
    bodyArgTypes.emplace_back(getElementTypeOrSelf(in.getType()));

  SmallVector<Type> opResultTypes;
  SmallVector<Value> initTensors;
  for (auto result : results) {
    auto resultTy = result.getType().template cast<ShapedType>();
    if (!resultTy.hasStaticShape())
      return rewriter.notifyMatchFailure(
          operation,
          "tosa to linalg conversion expects statically shaped tensors");

    initTensors.push_back(rewriter.create<linalg::InitTensorOp>(
        loc, ArrayRef<Value>({}), resultTy.getShape(),
        resultTy.getElementType()));
    opResultTypes.push_back(result.getType());
  }

  auto bodyResultTypes = llvm::to_vector<4>(llvm::map_range(
      initTensors, [](Value v) { return getElementTypeOrSelf(v); }));

  unsigned nloops = t0.getRank();
  SmallVector<AffineMap, 2> indexingMaps;
  indexingMaps.reserve(operation->getNumOperands() + bodyResultTypes.size());

  // Input indexing maps may be broadcasted.
  for (Type type : operation->getOperandTypes()) {
    indexingMaps.push_back(
        createAffineMapForType(type.cast<ShapedType>(), rewriter));
  }

  indexingMaps.append(operation->getNumResults(),
                      rewriter.getMultiDimIdentityMap(nloops));

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

// Returns the constant initial value for a given reduction operation. The
// attribute type varies depending on the element type required.
static Attribute createInitialValueForReduceOp(Operation *op, Type elementTy,
                                               PatternRewriter &rewriter) {
  if (isa<tosa::ReduceSumOp>(op) && elementTy.isa<FloatType>())
    return rewriter.getFloatAttr(elementTy, 0.0);

  if (isa<tosa::ReduceSumOp>(op) && elementTy.isa<IntegerType>())
    return rewriter.getIntegerAttr(elementTy, 0);

  if (isa<tosa::ReduceProdOp>(op) && elementTy.isa<FloatType>())
    return rewriter.getFloatAttr(elementTy, 1.0);

  if (isa<tosa::ReduceProdOp>(op) && elementTy.isa<IntegerType>())
    return rewriter.getIntegerAttr(elementTy, 1);

  if (isa<tosa::ReduceMinOp>(op) && elementTy.isa<FloatType>())
    return rewriter.getFloatAttr(
        elementTy, APFloat::getLargest(
                       elementTy.cast<FloatType>().getFloatSemantics(), false));

  if (isa<tosa::ReduceMinOp>(op) && elementTy.isa<IntegerType>())
    return rewriter.getIntegerAttr(
        elementTy, APInt::getSignedMaxValue(elementTy.getIntOrFloatBitWidth()));

  if (isa<tosa::ReduceMaxOp>(op) && elementTy.isa<FloatType>())
    return rewriter.getFloatAttr(
        elementTy, APFloat::getLargest(
                       elementTy.cast<FloatType>().getFloatSemantics(), true));

  if (isa<tosa::ReduceMaxOp>(op) && elementTy.isa<IntegerType>())
    return rewriter.getIntegerAttr(
        elementTy, APInt::getSignedMinValue(elementTy.getIntOrFloatBitWidth()));

  return {};
}

// Creates the body calculation for a reduction. The operations vary depending
// on the input type.
static Value createLinalgBodyCalculationForReduceOp(Operation *op,
                                                    ValueRange args,
                                                    Type elementTy,
                                                    PatternRewriter &rewriter) {
  Location loc = op->getLoc();
  if (isa<tosa::ReduceSumOp>(op) && elementTy.isa<FloatType>()) {
    return rewriter.create<AddFOp>(loc, args);
  }

  if (isa<tosa::ReduceSumOp>(op) && elementTy.isa<IntegerType>()) {
    return rewriter.create<AddIOp>(loc, args);
  }

  if (isa<tosa::ReduceProdOp>(op) && elementTy.isa<FloatType>()) {
    return rewriter.create<MulFOp>(loc, args);
  }

  if (isa<tosa::ReduceProdOp>(op) && elementTy.isa<IntegerType>()) {
    return rewriter.create<MulIOp>(loc, args);
  }

  if (isa<tosa::ReduceMinOp>(op) && elementTy.isa<FloatType>()) {
    auto predicate = rewriter.create<mlir::CmpFOp>(loc, CmpFPredicate::OLT,
                                                   args[0], args[1]);
    return rewriter.create<mlir::SelectOp>(loc, predicate, args[0], args[1]);
  }

  if (isa<tosa::ReduceMinOp>(op) && elementTy.isa<IntegerType>()) {
    auto predicate = rewriter.create<mlir::CmpIOp>(loc, CmpIPredicate::slt,
                                                   args[0], args[1]);
    return rewriter.create<mlir::SelectOp>(loc, predicate, args[0], args[1]);
  }

  if (isa<tosa::ReduceMaxOp>(op) && elementTy.isa<FloatType>()) {
    auto predicate = rewriter.create<mlir::CmpFOp>(loc, CmpFPredicate::OGT,
                                                   args[0], args[1]);
    return rewriter.create<mlir::SelectOp>(loc, predicate, args[0], args[1]);
  }

  if (isa<tosa::ReduceMaxOp>(op) && elementTy.isa<IntegerType>()) {
    auto predicate = rewriter.create<mlir::CmpIOp>(loc, CmpIPredicate::sgt,
                                                   args[0], args[1]);
    return rewriter.create<mlir::SelectOp>(loc, predicate, args[0], args[1]);
  }

  return {};
}

// Performs the match and rewrite for reduction operations. This includes
// declaring a correctly sized initial value, and the linalg.generic operation
// that reduces across the specified axis.
static LogicalResult reduceMatchAndRewriteHelper(Operation *op, uint64_t axis,
                                                 PatternRewriter &rewriter) {
  auto loc = op->getLoc();
  auto inputTy = op->getOperand(0).getType().template cast<ShapedType>();
  auto resultTy = op->getResult(0).getType().template cast<ShapedType>();
  auto elementTy = resultTy.getElementType();
  Value input = op->getOperand(0);

  // First fill the output buffer with the init value.
  auto initTensor = rewriter
                        .create<linalg::InitTensorOp>(loc, ArrayRef<Value>({}),
                                                      resultTy.getShape(),
                                                      resultTy.getElementType())
                        .result();

  auto fillValueAttr = createInitialValueForReduceOp(op, elementTy, rewriter);
  if (!fillValueAttr)
    return rewriter.notifyMatchFailure(
        op, "No initial value found for reduction operation");

  auto fillValue = rewriter.create<ConstantOp>(loc, fillValueAttr);
  auto filledTensor =
      rewriter.create<linalg::FillOp>(loc, initTensor, fillValue).result();

  SmallVector<AffineExpr, 2> srcExprs;
  SmallVector<AffineExpr, 2> dstExprs;
  SmallVector<StringRef, 4> iteratorTypes;
  for (unsigned int i = 0, rank = inputTy.getRank(); i != rank; ++i) {
    srcExprs.push_back(mlir::getAffineDimExpr(i, rewriter.getContext()));

    iteratorTypes.push_back(axis == i ? getReductionIteratorTypeName()
                                      : getParallelIteratorTypeName());
    if (axis != i)
      dstExprs.push_back(mlir::getAffineDimExpr(i, rewriter.getContext()));
  }

  bool didEncounterError = false;
  auto maps = AffineMap::inferFromExprList({srcExprs, dstExprs});
  auto linalgOp = rewriter.create<linalg::GenericOp>(
      loc, resultTy, input, filledTensor, maps, iteratorTypes,
      [&](OpBuilder &nestedBuilder, Location nestedLoc, ValueRange blockArgs) {
        auto result = createLinalgBodyCalculationForReduceOp(
            op, blockArgs, elementTy, rewriter);
        if (result)
          didEncounterError = true;

        nestedBuilder.create<linalg::YieldOp>(loc, result);
      });

  if (!didEncounterError)
    return failure();

  rewriter.replaceOp(op, linalgOp.getOperation()->getResults());
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

class ReshapeConverter : public OpConversionPattern<tosa::ReshapeOp> {
public:
  using OpConversionPattern<tosa::ReshapeOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(tosa::ReshapeOp reshape, ArrayRef<Value> args,
                  ConversionPatternRewriter &rewriter) const final {
    typename tosa::ReshapeOp::Adaptor operands(args);

    ShapedType operandTy = operands.input1().getType().cast<ShapedType>();
    ShapedType resultTy = reshape.getType().template cast<ShapedType>();

    if (!operandTy.hasStaticShape() || !resultTy.hasStaticShape())
      return failure();

    // Compute the reassociation maps for the linalg operation.
    ArrayRef<int64_t> expandedShape =
        (operandTy.getRank() > resultTy.getRank() ? operandTy.getShape()
                                                  : resultTy.getShape());
    ArrayRef<int64_t> collapsedShape =
        (operandTy.getRank() > resultTy.getRank() ? resultTy.getShape()
                                                  : operandTy.getShape());
    unsigned currSrcDim = 0, currDstDim = 0;
    SmallVector<linalg::ReassociationExprs, 4> reassociationMap(
        collapsedShape.size());

    // First scan all dimensions in the source shapes to see whether we have a
    // perfect case where consecutive dimensions in source are collapsed. For
    // such case we can just generate one single linalg.reshape.
    bool isCollapsingSource = true;
    while (currSrcDim < expandedShape.size() &&
           currDstDim < collapsedShape.size()) {
      int64_t dstSize = collapsedShape[currDstDim];
      int64_t srcSize = expandedShape[currSrcDim];
      while (srcSize < dstSize && currSrcDim < expandedShape.size()) {
        reassociationMap[currDstDim].push_back(
            rewriter.getAffineDimExpr(currSrcDim++));
        srcSize *= expandedShape[currSrcDim];
      }
      if (srcSize == dstSize) {
        reassociationMap[currDstDim].push_back(
            rewriter.getAffineDimExpr(currSrcDim++));
        // If the next dim in collapsedShape is not 1, treat subsequent dims in
        // expandedShape which are 1 to be collapsed.
        if (currDstDim == collapsedShape.size() - 1 ||
            collapsedShape[currDstDim + 1] != 1) {
          while (currSrcDim < expandedShape.size() &&
                 expandedShape[currSrcDim] == 1) {
            reassociationMap[currDstDim].push_back(
                rewriter.getAffineDimExpr(currSrcDim++));
          }
        }
      } else {
        isCollapsingSource = false;
        break;
      }
      currDstDim++;
    }
    if (currSrcDim != expandedShape.size() ||
        currDstDim != collapsedShape.size())
      isCollapsingSource = false;

    // Otherwise, we need to first reduce all source dimensions into one and
    // then expand to the destination dimensions.
    if (!isCollapsingSource) {
      auto getIdentityExprs = [&rewriter](int n) {
        SmallVector<AffineExpr, 4> exprs;
        for (int i = 0; i < n; ++i)
          exprs.push_back(rewriter.getAffineDimExpr(i));
        return exprs;
      };
      Location loc = reshape.getLoc();
      int64_t totalElems =
          std::accumulate(expandedShape.begin(), expandedShape.end(), 1,
                          std::multiplies<int64_t>());
      auto elemTy = operandTy.getElementType();
      SmallVector<linalg::ReassociationExprs, 4> collapsingMap = {
          // Use operandTy here because we need to collapse all operands
          // dimensions.
          getIdentityExprs(operandTy.getShape().size())};
      SmallVector<linalg::ReassociationExprs, 4> expandingMap = {
          // Use resultTy here because we need to expand to all result
          // dimensions.
          getIdentityExprs(resultTy.getShape().size())};

      auto collapsedTy = RankedTensorType::get({totalElems}, elemTy);
      Value collapsedOp = rewriter.create<linalg::TensorReshapeOp>(
          loc, collapsedTy, args[0], collapsingMap);
      rewriter.replaceOpWithNewOp<linalg::TensorReshapeOp>(
          reshape, resultTy, collapsedOp, expandingMap);

      return success();
    }

    rewriter.replaceOpWithNewOp<linalg::TensorReshapeOp>(
        reshape, resultTy, args[0], reassociationMap);

    return success();
  }
};

class TransposeConverter : public OpRewritePattern<tosa::TransposeOp> {
public:
  using OpRewritePattern<tosa::TransposeOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(tosa::TransposeOp op,
                                PatternRewriter &rewriter) const final {
    DenseIntElementsAttr perms;
    if (!matchPattern(op.perms(), m_Constant(&perms))) {
      return failure();
    }

    auto resultTy = op.getType().cast<ShapedType>();
    if (!resultTy.hasStaticShape())
      return failure();

    SmallVector<AffineExpr, 2> inputExprs;
    inputExprs.resize(resultTy.getRank());
    for (auto permutation : llvm::enumerate(perms.getIntValues())) {
      inputExprs[permutation.value().getZExtValue()] =
          rewriter.getAffineDimExpr(permutation.index());
    }

    auto initTensor = rewriter.create<linalg::InitTensorOp>(
        op.getLoc(), ArrayRef<Value>({}), resultTy.getShape(),
        resultTy.getElementType());

    SmallVector<AffineMap, 2> affineMaps = {
        AffineMap::get(resultTy.getRank(), /*symbolCount=*/0, inputExprs,
                       rewriter.getContext()),
        rewriter.getMultiDimIdentityMap(resultTy.getRank())};

    rewriter.replaceOpWithNewOp<linalg::GenericOp>(
        op, resultTy, op.input1(), ValueRange{initTensor}, affineMaps,
        getNParallelLoopsAttrs(resultTy.getRank()),
        [&](OpBuilder &nestedBuilder, Location nestedLoc, ValueRange args) {
          nestedBuilder.create<linalg::YieldOp>(op.getLoc(), *args.begin());
        });
    return success();
  }
};

class RescaleConverter : public OpRewritePattern<tosa::RescaleOp> {
public:
  using OpRewritePattern<tosa::RescaleOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(tosa::RescaleOp op,
                                PatternRewriter &rewriter) const final {
    auto loc = op.getLoc();
    auto input = op.input();
    auto inputTy = op.input().getType().cast<ShapedType>();
    auto outputTy = op.output().getType().cast<ShapedType>();
    unsigned rank = inputTy.getRank();

    if (!outputTy.hasStaticShape())
      return rewriter.notifyMatchFailure(
          op, "tosa to linalg conversion expects statically shaped tensors");

    // The shift and multiplier values.
    SmallVector<int32_t> multiplierValues;
    getValuesFromIntArrayAttribute(op.multiplier(), multiplierValues);

    SmallVector<int8_t> shiftValues;
    getValuesFromIntArrayAttribute(op.shift(), shiftValues);

    // Double round only occurs if shift is greater than 31, check that this
    // is ever true.
    bool doubleRound =
        op.double_round() &&
        llvm::any_of(shiftValues, [](int32_t v) { return v > 31; });

    // We need to broadcast along the last dimension, so make all dims 1.
    SmallVector<int64_t> multiplierShape;
    multiplierShape.resize(rank, 1);

    SmallVector<int64_t> shiftShape;
    shiftShape.resize(rank, 1);

    // Set the channel dimension to match the number of shift/broadcast
    // channels.
    if (!multiplierShape.empty())
      multiplierShape.back() = multiplierValues.size();
    if (!shiftShape.empty())
      shiftShape.back() = shiftValues.size();

    // Create the tensor types.
    auto multiplierType =
        RankedTensorType::get(multiplierShape, rewriter.getI32Type());
    auto shiftType =
        RankedTensorType::get(shiftShape, rewriter.getIntegerType(8));

    auto multiplierConst = rewriter.create<ConstantOp>(
        loc, DenseIntElementsAttr::get(multiplierType, multiplierValues));

    auto shiftConst = rewriter.create<ConstantOp>(
        loc, DenseIntElementsAttr::get(shiftType, shiftValues));

    // Construct the indexing maps needed for linalg.generic ops.
    SmallVector<Type> bodyArgTypes = {getElementTypeOrSelf(inputTy),
                                      rewriter.getI32Type(),
                                      rewriter.getI32Type()};
    Value initTensor = rewriter.create<linalg::InitTensorOp>(
        loc, ArrayRef<Value>({}), outputTy.getShape(),
        outputTy.getElementType());

    SmallVector<AffineMap, 4> indexingMaps;

    // Indexing map for input values.
    indexingMaps.push_back(rewriter.getMultiDimIdentityMap(rank));

    // Shift and multiplier will need to broadcast across their non channel
    // values.
    indexingMaps.push_back(createAffineMapForType(multiplierType, rewriter));
    indexingMaps.push_back(createAffineMapForType(shiftType, rewriter));

    // Indexing maps for output values.
    indexingMaps.push_back(rewriter.getMultiDimIdentityMap(rank));

    auto linalgOp = rewriter.create<linalg::GenericOp>(
        loc, outputTy, ValueRange{input, multiplierConst, shiftConst},
        ValueRange{initTensor}, indexingMaps, getNParallelLoopsAttrs(rank),
        [&](OpBuilder &nestedBuilder, Location nestedLoc,
            ValueRange blockArgs) {
          // For now we do all of our math in 64-bit. This is not optimal but
          // should be correct for now, consider computing correct bit depth
          // later.
          auto inputZp = createConstFromIntAttribute<int32_t>(
              op, "input_zp", nestedBuilder.getI32Type(), nestedBuilder);
          auto outputZp = createConstFromIntAttribute<int32_t>(
              op, "output_zp", nestedBuilder.getI32Type(), nestedBuilder);

          Value value = blockArgs[0];
          Value multiplier = blockArgs[1];
          Value shift = blockArgs[2];

          if (value.getType().getIntOrFloatBitWidth() < 32) {
            value = nestedBuilder.create<SignExtendIOp>(
                nestedLoc, nestedBuilder.getI32Type(), value);
          }

          value = nestedBuilder.create<SubIOp>(nestedLoc, value, inputZp);

          value = nestedBuilder.create<tosa::ApplyScaleOp>(
              loc, nestedBuilder.getI32Type(), value, multiplier, shift,
              nestedBuilder.getBoolAttr(doubleRound));

          // Move to the new zero-point.
          value = nestedBuilder.create<AddIOp>(nestedLoc, value, outputZp);

          // Saturate to the output size.
          IntegerType outIntType =
              blockArgs.back().getType().cast<IntegerType>();
          unsigned outBitWidth = outIntType.getWidth();
          auto intMin = nestedBuilder.create<ConstantOp>(
              loc, nestedBuilder.getIntegerAttr(
                       nestedBuilder.getI32Type(),
                       APInt::getSignedMinValue(outBitWidth).getSExtValue()));
          auto intMax = nestedBuilder.create<ConstantOp>(
              loc, nestedBuilder.getIntegerAttr(
                       nestedBuilder.getI32Type(),
                       APInt::getSignedMaxValue(outBitWidth).getSExtValue()));

          value = clampHelper<mlir::CmpIOp>(nestedLoc, value, intMin, intMax,
                                            CmpIPredicate::slt, nestedBuilder);

          if (outIntType.getWidth() < 32) {
            value =
                nestedBuilder.create<TruncateIOp>(nestedLoc, outIntType, value);
          }

          nestedBuilder.create<linalg::YieldOp>(loc, value);
        });

    rewriter.replaceOp(op, linalgOp->getResults());
    return success();
  }
};

// At the codegen level any identity operations should be removed. Any cases
// where identity is load-bearing (e.g. cross device computation) should be
// handled before lowering to codegen.
template <typename SrcOp>
class IdentityNConverter : public OpRewritePattern<SrcOp> {
public:
  using OpRewritePattern<SrcOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(SrcOp op,
                                PatternRewriter &rewriter) const final {
    rewriter.replaceOp(op, op.getOperation()->getOperands());
    return success();
  }
};

template <typename SrcOp>
class ReduceConverter : public OpRewritePattern<SrcOp> {
public:
  using OpRewritePattern<SrcOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(SrcOp reduceOp,
                                PatternRewriter &rewriter) const final {
    return reduceMatchAndRewriteHelper(reduceOp, reduceOp.axis(), rewriter);
  }
};

struct ConcatConverter : public OpConversionPattern<tosa::ConcatOp> {
  using OpConversionPattern<tosa::ConcatOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(tosa::ConcatOp op, ArrayRef<Value> args,
                  ConversionPatternRewriter &rewriter) const override {
    auto resultType = op.getType().dyn_cast<RankedTensorType>();
    if (!resultType || !resultType.hasStaticShape()) {
      return rewriter.notifyMatchFailure(op,
                                         "expected static shaped tensor type");
    }

    Location loc = op.getLoc();
    int axis = op.axis();
    Value axisValue =
        rewriter.create<ConstantOp>(loc, rewriter.getIndexAttr(axis));
    int rank = resultType.getRank();
    SmallVector<Value, 3> offsets, sizes, strides;
    sizes.reserve(rank);
    strides.resize(rank, rewriter.create<ConstantIndexOp>(loc, 1));
    offsets.resize(rank, rewriter.create<ConstantIndexOp>(loc, 0));

    for (int i = 0; i < rank; ++i) {
      sizes.push_back(rewriter.create<memref::DimOp>(loc, args[0], i));
    }

    Value resultDimSize = sizes[axis];
    for (auto arg : args.drop_front()) {
      auto size = rewriter.create<memref::DimOp>(loc, arg, axisValue);
      resultDimSize = rewriter.create<AddIOp>(loc, resultDimSize, size);
    }
    sizes[axis] = resultDimSize;

    Value result = rewriter.create<linalg::InitTensorOp>(
        loc, resultType.getShape(), resultType.getElementType());

    for (auto arg : args) {
      sizes[axis] = rewriter.create<memref::DimOp>(loc, arg, axisValue);
      result = rewriter.create<SubTensorInsertOp>(loc, arg, result, offsets,
                                                  sizes, strides);
      offsets[axis] = rewriter.create<AddIOp>(loc, offsets[axis], sizes[axis]);
    }
    rewriter.replaceOp(op, result);
    return success();
  }
};

class ReverseConverter : public OpRewritePattern<tosa::ReverseOp> {
public:
  using OpRewritePattern<tosa::ReverseOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(tosa::ReverseOp op,
                                PatternRewriter &rewriter) const final {
    auto loc = op.getLoc();
    Value input = op.input();
    auto inputTy = input.getType().template cast<ShapedType>();
    auto resultTy = op.getType().template cast<ShapedType>();
    auto rank = resultTy.getRank();
    auto axis = op.axis();

    if (!inputTy.hasStaticShape())
      return rewriter.notifyMatchFailure(
          op, "No initial value found for reduction operation");

    // First fill the output buffer with the init value.
    auto initTensor = rewriter
                          .create<linalg::InitTensorOp>(
                              loc, ArrayRef<Value>({}), inputTy.getShape(),
                              inputTy.getElementType())
                          .result();

    SmallVector<AffineExpr, 2> inputExprs;
    inputExprs.resize(resultTy.getRank());

    for (int i = 0; i < rank; i++)
      inputExprs[i] = rewriter.getAffineDimExpr(i);

    inputExprs[axis] =
        rewriter.getAffineConstantExpr(inputTy.getDimSize(axis) - 1) -
        inputExprs[axis];

    SmallVector<AffineMap, 2> affineMaps = {
        AffineMap::get(resultTy.getRank(), /*symbolCount=*/0, inputExprs,
                       rewriter.getContext()),
        rewriter.getMultiDimIdentityMap(resultTy.getRank())};

    rewriter.replaceOpWithNewOp<linalg::GenericOp>(
        op, resultTy, op.input(), ValueRange{initTensor}, affineMaps,
        getNParallelLoopsAttrs(resultTy.getRank()),
        [&](OpBuilder &nestedBuilder, Location nestedLoc, ValueRange args) {
          nestedBuilder.create<linalg::YieldOp>(op.getLoc(), *args.begin());
        });

    return success();
  }
};

} // namespace

void mlir::tosa::populateTosaToLinalgOnTensorsConversionPatterns(
    OwningRewritePatternList *patterns) {
  patterns->insert<
      PointwiseConverter<tosa::AddOp>, PointwiseConverter<tosa::SubOp>,
      PointwiseConverter<tosa::MulOp>, PointwiseConverter<tosa::NegateOp>,
      PointwiseConverter<tosa::PowOp>, PointwiseConverter<tosa::RsqrtOp>,
      PointwiseConverter<tosa::LogOp>, PointwiseConverter<tosa::ExpOp>,
      PointwiseConverter<tosa::AbsOp>, PointwiseConverter<tosa::TanhOp>,
      PointwiseConverter<tosa::BitwiseAndOp>,
      PointwiseConverter<tosa::BitwiseOrOp>,
      PointwiseConverter<tosa::BitwiseXorOp>,
      PointwiseConverter<tosa::LogicalAndOp>,
      PointwiseConverter<tosa::LogicalNotOp>,
      PointwiseConverter<tosa::LogicalOrOp>,
      PointwiseConverter<tosa::LogicalXorOp>, PointwiseConverter<tosa::CastOp>,
      PointwiseConverter<tosa::LogicalLeftShiftOp>,
      PointwiseConverter<tosa::LogicalRightShiftOp>,
      PointwiseConverter<tosa::SelectOp>, PointwiseConverter<tosa::GreaterOp>,
      PointwiseConverter<tosa::GreaterEqualOp>,
      PointwiseConverter<tosa::MaximumOp>, PointwiseConverter<tosa::MinimumOp>,
      PointwiseConverter<tosa::CeilOp>, PointwiseConverter<tosa::FloorOp>,
      PointwiseConverter<tosa::ClampOp>, PointwiseConverter<tosa::ReluNOp>,
      IdentityNConverter<tosa::IdentityOp>,
      IdentityNConverter<tosa::IdentityNOp>, ReduceConverter<tosa::ReduceMinOp>,
      ReduceConverter<tosa::ReduceMaxOp>, ReduceConverter<tosa::ReduceSumOp>,
      ReduceConverter<tosa::ReduceProdOp>, ConcatConverter, ReshapeConverter,
      RescaleConverter, ReverseConverter, TransposeConverter>(
      patterns->getContext());
}

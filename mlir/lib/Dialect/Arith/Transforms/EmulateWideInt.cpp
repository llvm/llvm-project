//===- EmulateWideInt.cpp - Wide integer operation emulation ----*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Arith/Transforms/Passes.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Arith/Transforms/WideIntEmulationConverter.h"
#include "mlir/Dialect/Arith/Utils/Utils.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Func/Transforms/FuncConversions.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/TypeUtilities.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/DialectConversion.h"
#include "llvm/ADT/APInt.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/MathExtras.h"
#include <cassert>

namespace mlir::arith {
#define GEN_PASS_DEF_ARITHEMULATEWIDEINT
#include "mlir/Dialect/Arith/Transforms/Passes.h.inc"
} // namespace mlir::arith

using namespace mlir;

//===----------------------------------------------------------------------===//
// Common Helper Functions
//===----------------------------------------------------------------------===//

/// Returns N bottom and N top bits from `value`, where N = `newBitWidth`.
/// Treats `value` as a 2*N bits-wide integer.
/// The bottom bits are returned in the first pair element, while the top bits
/// in the second one.
static std::pair<APInt, APInt> getHalves(const APInt &value,
                                         unsigned newBitWidth) {
  APInt low = value.extractBits(newBitWidth, 0);
  APInt high = value.extractBits(newBitWidth, newBitWidth);
  return {std::move(low), std::move(high)};
}

/// Returns the type with the last (innermost) dimension reduced to x1.
/// Scalarizes 1D vector inputs to match how we extract/insert vector values,
/// e.g.:
///   - vector<3x2xi16> --> vector<3x1xi16>
///   - vector<2xi16>   --> i16
static Type reduceInnermostDim(VectorType type) {
  if (type.getShape().size() == 1)
    return type.getElementType();

  auto newShape = to_vector(type.getShape());
  newShape.back() = 1;
  return VectorType::get(newShape, type.getElementType());
}

/// Extracts the `input` vector slice with elements at the last dimension offset
/// by `lastOffset`. Returns a value of vector type with the last dimension
/// reduced to x1 or fully scalarized, e.g.:
///   - vector<3x2xi16> --> vector<3x1xi16>
///   - vector<2xi16>   --> i16
static Value extractLastDimSlice(ConversionPatternRewriter &rewriter,
                                 Location loc, Value input,
                                 int64_t lastOffset) {
  ArrayRef<int64_t> shape = cast<VectorType>(input.getType()).getShape();
  assert(lastOffset < shape.back() && "Offset out of bounds");

  // Scalarize the result in case of 1D vectors.
  if (shape.size() == 1)
    return rewriter.create<vector::ExtractOp>(loc, input, lastOffset);

  SmallVector<int64_t> offsets(shape.size(), 0);
  offsets.back() = lastOffset;
  auto sizes = llvm::to_vector(shape);
  sizes.back() = 1;
  SmallVector<int64_t> strides(shape.size(), 1);

  return rewriter.create<vector::ExtractStridedSliceOp>(loc, input, offsets,
                                                        sizes, strides);
}

/// Extracts two vector slices from the `input` whose type is `vector<...x2T>`,
/// with the first element at offset 0 and the second element at offset 1.
static std::pair<Value, Value>
extractLastDimHalves(ConversionPatternRewriter &rewriter, Location loc,
                     Value input) {
  return {extractLastDimSlice(rewriter, loc, input, 0),
          extractLastDimSlice(rewriter, loc, input, 1)};
}

// Performs a vector shape cast to drop the trailing x1 dimension. If the
// `input` is a scalar, this is a noop.
static Value dropTrailingX1Dim(ConversionPatternRewriter &rewriter,
                               Location loc, Value input) {
  auto vecTy = dyn_cast<VectorType>(input.getType());
  if (!vecTy)
    return input;

  // Shape cast to drop the last x1 dimension.
  ArrayRef<int64_t> shape = vecTy.getShape();
  assert(shape.size() >= 2 && "Expected vector with at list two dims");
  assert(shape.back() == 1 && "Expected the last vector dim to be x1");

  auto newVecTy = VectorType::get(shape.drop_back(), vecTy.getElementType());
  return rewriter.create<vector::ShapeCastOp>(loc, newVecTy, input);
}

/// Performs a vector shape cast to append an x1 dimension. If the
/// `input` is a scalar, this is a noop.
static Value appendX1Dim(ConversionPatternRewriter &rewriter, Location loc,
                         Value input) {
  auto vecTy = dyn_cast<VectorType>(input.getType());
  if (!vecTy)
    return input;

  // Add a trailing x1 dim.
  auto newShape = llvm::to_vector(vecTy.getShape());
  newShape.push_back(1);
  auto newTy = VectorType::get(newShape, vecTy.getElementType());
  return rewriter.create<vector::ShapeCastOp>(loc, newTy, input);
}

/// Inserts the `source` vector slice into the `dest` vector at offset
/// `lastOffset` in the last dimension. `source` can be a scalar when `dest` is
/// a 1D vector.
static Value insertLastDimSlice(ConversionPatternRewriter &rewriter,
                                Location loc, Value source, Value dest,
                                int64_t lastOffset) {
  ArrayRef<int64_t> shape = cast<VectorType>(dest.getType()).getShape();
  assert(lastOffset < shape.back() && "Offset out of bounds");

  // Handle scalar source.
  if (isa<IntegerType>(source.getType()))
    return rewriter.create<vector::InsertOp>(loc, source, dest, lastOffset);

  SmallVector<int64_t> offsets(shape.size(), 0);
  offsets.back() = lastOffset;
  SmallVector<int64_t> strides(shape.size(), 1);
  return rewriter.create<vector::InsertStridedSliceOp>(loc, source, dest,
                                                       offsets, strides);
}

/// Constructs a new vector of type `resultType` by creating a series of
/// insertions of `resultComponents`, each at the next offset of the last vector
/// dimension.
/// When all `resultComponents` are scalars, the result type is `vector<NxT>`;
/// when `resultComponents` are `vector<...x1xT>`s, the result type is
/// `vector<...xNxT>`, where `N` is the number of `resultComponents`.
static Value constructResultVector(ConversionPatternRewriter &rewriter,
                                   Location loc, VectorType resultType,
                                   ValueRange resultComponents) {
  llvm::ArrayRef<int64_t> resultShape = resultType.getShape();
  (void)resultShape;
  assert(!resultShape.empty() && "Result expected to have dimensions");
  assert(resultShape.back() == static_cast<int64_t>(resultComponents.size()) &&
         "Wrong number of result components");

  Value resultVec = createScalarOrSplatConstant(rewriter, loc, resultType, 0);
  for (auto [i, component] : llvm::enumerate(resultComponents))
    resultVec = insertLastDimSlice(rewriter, loc, component, resultVec, i);

  return resultVec;
}

namespace {
//===----------------------------------------------------------------------===//
// ConvertConstant
//===----------------------------------------------------------------------===//

struct ConvertConstant final : OpConversionPattern<arith::ConstantOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(arith::ConstantOp op, OpAdaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Type oldType = op.getType();
    auto newType = getTypeConverter()->convertType<VectorType>(oldType);
    if (!newType)
      return rewriter.notifyMatchFailure(
          op, llvm::formatv("unsupported type: {0}", op.getType()));

    unsigned newBitWidth = newType.getElementTypeBitWidth();
    Attribute oldValue = op.getValueAttr();

    if (auto intAttr = dyn_cast<IntegerAttr>(oldValue)) {
      auto [low, high] = getHalves(intAttr.getValue(), newBitWidth);
      auto newAttr = DenseElementsAttr::get(newType, {low, high});
      rewriter.replaceOpWithNewOp<arith::ConstantOp>(op, newAttr);
      return success();
    }

    if (auto splatAttr = dyn_cast<SplatElementsAttr>(oldValue)) {
      auto [low, high] =
          getHalves(splatAttr.getSplatValue<APInt>(), newBitWidth);
      int64_t numSplatElems = splatAttr.getNumElements();
      SmallVector<APInt> values;
      values.reserve(numSplatElems * 2);
      for (int64_t i = 0; i < numSplatElems; ++i) {
        values.push_back(low);
        values.push_back(high);
      }

      auto attr = DenseElementsAttr::get(newType, values);
      rewriter.replaceOpWithNewOp<arith::ConstantOp>(op, attr);
      return success();
    }

    if (auto elemsAttr = dyn_cast<DenseElementsAttr>(oldValue)) {
      int64_t numElems = elemsAttr.getNumElements();
      SmallVector<APInt> values;
      values.reserve(numElems * 2);
      for (const APInt &origVal : elemsAttr.getValues<APInt>()) {
        auto [low, high] = getHalves(origVal, newBitWidth);
        values.push_back(std::move(low));
        values.push_back(std::move(high));
      }

      auto attr = DenseElementsAttr::get(newType, values);
      rewriter.replaceOpWithNewOp<arith::ConstantOp>(op, attr);
      return success();
    }

    return rewriter.notifyMatchFailure(op.getLoc(),
                                       "unhandled constant attribute");
  }
};

//===----------------------------------------------------------------------===//
// ConvertAddI
//===----------------------------------------------------------------------===//

struct ConvertAddI final : OpConversionPattern<arith::AddIOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(arith::AddIOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op->getLoc();
    auto newTy = getTypeConverter()->convertType<VectorType>(op.getType());
    if (!newTy)
      return rewriter.notifyMatchFailure(
          loc, llvm::formatv("unsupported type: {0}", op.getType()));

    Type newElemTy = reduceInnermostDim(newTy);

    auto [lhsElem0, lhsElem1] =
        extractLastDimHalves(rewriter, loc, adaptor.getLhs());
    auto [rhsElem0, rhsElem1] =
        extractLastDimHalves(rewriter, loc, adaptor.getRhs());

    auto lowSum =
        rewriter.create<arith::AddUIExtendedOp>(loc, lhsElem0, rhsElem0);
    Value overflowVal =
        rewriter.create<arith::ExtUIOp>(loc, newElemTy, lowSum.getOverflow());

    Value high0 = rewriter.create<arith::AddIOp>(loc, overflowVal, lhsElem1);
    Value high = rewriter.create<arith::AddIOp>(loc, high0, rhsElem1);

    Value resultVec =
        constructResultVector(rewriter, loc, newTy, {lowSum.getSum(), high});
    rewriter.replaceOp(op, resultVec);
    return success();
  }
};

//===----------------------------------------------------------------------===//
// ConvertBitwiseBinary
//===----------------------------------------------------------------------===//

/// Conversion pattern template for bitwise binary ops, e.g., `arith.andi`.
template <typename BinaryOp>
struct ConvertBitwiseBinary final : OpConversionPattern<BinaryOp> {
  using OpConversionPattern<BinaryOp>::OpConversionPattern;
  using OpAdaptor = typename OpConversionPattern<BinaryOp>::OpAdaptor;

  LogicalResult
  matchAndRewrite(BinaryOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op->getLoc();
    auto newTy = this->getTypeConverter()->template convertType<VectorType>(
        op.getType());
    if (!newTy)
      return rewriter.notifyMatchFailure(
          loc, llvm::formatv("unsupported type: {0}", op.getType()));

    auto [lhsElem0, lhsElem1] =
        extractLastDimHalves(rewriter, loc, adaptor.getLhs());
    auto [rhsElem0, rhsElem1] =
        extractLastDimHalves(rewriter, loc, adaptor.getRhs());

    Value resElem0 = rewriter.create<BinaryOp>(loc, lhsElem0, rhsElem0);
    Value resElem1 = rewriter.create<BinaryOp>(loc, lhsElem1, rhsElem1);
    Value resultVec =
        constructResultVector(rewriter, loc, newTy, {resElem0, resElem1});
    rewriter.replaceOp(op, resultVec);
    return success();
  }
};

//===----------------------------------------------------------------------===//
// ConvertCmpI
//===----------------------------------------------------------------------===//

/// Returns the matching unsigned version of the given predicate `pred`, or the
/// same predicate if `pred` is not a signed.
static arith::CmpIPredicate toUnsignedPredicate(arith::CmpIPredicate pred) {
  using P = arith::CmpIPredicate;
  switch (pred) {
  case P::sge:
    return P::uge;
  case P::sgt:
    return P::ugt;
  case P::sle:
    return P::ule;
  case P::slt:
    return P::ult;
  default:
    return pred;
  }
}

struct ConvertCmpI final : OpConversionPattern<arith::CmpIOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(arith::CmpIOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op->getLoc();
    auto inputTy =
        getTypeConverter()->convertType<VectorType>(op.getLhs().getType());
    if (!inputTy)
      return rewriter.notifyMatchFailure(
          loc, llvm::formatv("unsupported type: {0}", op.getType()));

    arith::CmpIPredicate highPred = adaptor.getPredicate();
    arith::CmpIPredicate lowPred = toUnsignedPredicate(highPred);

    auto [lhsElem0, lhsElem1] =
        extractLastDimHalves(rewriter, loc, adaptor.getLhs());
    auto [rhsElem0, rhsElem1] =
        extractLastDimHalves(rewriter, loc, adaptor.getRhs());

    Value lowCmp =
        rewriter.create<arith::CmpIOp>(loc, lowPred, lhsElem0, rhsElem0);
    Value highCmp =
        rewriter.create<arith::CmpIOp>(loc, highPred, lhsElem1, rhsElem1);

    Value cmpResult{};
    switch (highPred) {
    case arith::CmpIPredicate::eq: {
      cmpResult = rewriter.create<arith::AndIOp>(loc, lowCmp, highCmp);
      break;
    }
    case arith::CmpIPredicate::ne: {
      cmpResult = rewriter.create<arith::OrIOp>(loc, lowCmp, highCmp);
      break;
    }
    default: {
      // Handle inequality checks.
      Value highEq = rewriter.create<arith::CmpIOp>(
          loc, arith::CmpIPredicate::eq, lhsElem1, rhsElem1);
      cmpResult =
          rewriter.create<arith::SelectOp>(loc, highEq, lowCmp, highCmp);
      break;
    }
    }

    assert(cmpResult && "Unhandled case");
    rewriter.replaceOp(op, dropTrailingX1Dim(rewriter, loc, cmpResult));
    return success();
  }
};

//===----------------------------------------------------------------------===//
// ConvertMulI
//===----------------------------------------------------------------------===//

struct ConvertMulI final : OpConversionPattern<arith::MulIOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(arith::MulIOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op->getLoc();
    auto newTy = getTypeConverter()->convertType<VectorType>(op.getType());
    if (!newTy)
      return rewriter.notifyMatchFailure(
          loc, llvm::formatv("unsupported type: {0}", op.getType()));

    auto [lhsElem0, lhsElem1] =
        extractLastDimHalves(rewriter, loc, adaptor.getLhs());
    auto [rhsElem0, rhsElem1] =
        extractLastDimHalves(rewriter, loc, adaptor.getRhs());

    // The multiplication algorithm used is the standard (long) multiplication.
    // Multiplying two i2N integers produces (at most) an i4N result, but
    // because the calculation of top i2N is not necessary, we omit it.
    auto mulLowLow =
        rewriter.create<arith::MulUIExtendedOp>(loc, lhsElem0, rhsElem0);
    Value mulLowHi = rewriter.create<arith::MulIOp>(loc, lhsElem0, rhsElem1);
    Value mulHiLow = rewriter.create<arith::MulIOp>(loc, lhsElem1, rhsElem0);

    Value resLow = mulLowLow.getLow();
    Value resHi =
        rewriter.create<arith::AddIOp>(loc, mulLowLow.getHigh(), mulLowHi);
    resHi = rewriter.create<arith::AddIOp>(loc, resHi, mulHiLow);

    Value resultVec =
        constructResultVector(rewriter, loc, newTy, {resLow, resHi});
    rewriter.replaceOp(op, resultVec);
    return success();
  }
};

//===----------------------------------------------------------------------===//
// ConvertExtSI
//===----------------------------------------------------------------------===//

struct ConvertExtSI final : OpConversionPattern<arith::ExtSIOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(arith::ExtSIOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op->getLoc();
    auto newTy = getTypeConverter()->convertType<VectorType>(op.getType());
    if (!newTy)
      return rewriter.notifyMatchFailure(
          loc, llvm::formatv("unsupported type: {0}", op.getType()));

    Type newResultComponentTy = reduceInnermostDim(newTy);

    // Sign-extend the input value to determine the low half of the result.
    // Then, check if the low half is negative, and sign-extend the comparison
    // result to get the high half.
    Value newOperand = appendX1Dim(rewriter, loc, adaptor.getIn());
    Value extended = rewriter.createOrFold<arith::ExtSIOp>(
        loc, newResultComponentTy, newOperand);
    Value operandZeroCst =
        createScalarOrSplatConstant(rewriter, loc, newResultComponentTy, 0);
    Value signBit = rewriter.create<arith::CmpIOp>(
        loc, arith::CmpIPredicate::slt, extended, operandZeroCst);
    Value signValue =
        rewriter.create<arith::ExtSIOp>(loc, newResultComponentTy, signBit);

    Value resultVec =
        constructResultVector(rewriter, loc, newTy, {extended, signValue});
    rewriter.replaceOp(op, resultVec);
    return success();
  }
};

//===----------------------------------------------------------------------===//
// ConvertExtUI
//===----------------------------------------------------------------------===//

struct ConvertExtUI final : OpConversionPattern<arith::ExtUIOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(arith::ExtUIOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op->getLoc();
    auto newTy = getTypeConverter()->convertType<VectorType>(op.getType());
    if (!newTy)
      return rewriter.notifyMatchFailure(
          loc, llvm::formatv("unsupported type: {0}", op.getType()));

    Type newResultComponentTy = reduceInnermostDim(newTy);

    // Zero-extend the input value to determine the low half of the result.
    // The high half is always zero.
    Value newOperand = appendX1Dim(rewriter, loc, adaptor.getIn());
    Value extended = rewriter.createOrFold<arith::ExtUIOp>(
        loc, newResultComponentTy, newOperand);
    Value zeroCst = createScalarOrSplatConstant(rewriter, loc, newTy, 0);
    Value newRes = insertLastDimSlice(rewriter, loc, extended, zeroCst, 0);
    rewriter.replaceOp(op, newRes);
    return success();
  }
};

//===----------------------------------------------------------------------===//
// ConvertMaxMin
//===----------------------------------------------------------------------===//

template <typename SourceOp, arith::CmpIPredicate CmpPred>
struct ConvertMaxMin final : OpConversionPattern<SourceOp> {
  using OpConversionPattern<SourceOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(SourceOp op, typename SourceOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op->getLoc();

    Type oldTy = op.getType();
    auto newTy = dyn_cast_or_null<VectorType>(
        this->getTypeConverter()->convertType(oldTy));
    if (!newTy)
      return rewriter.notifyMatchFailure(
          loc, llvm::formatv("unsupported type: {0}", op.getType()));

    // Rewrite Max*I/Min*I as compare and select over original operands. Let
    // the CmpI and Select emulation patterns handle the final legalization.
    Value cmp =
        rewriter.create<arith::CmpIOp>(loc, CmpPred, op.getLhs(), op.getRhs());
    rewriter.replaceOpWithNewOp<arith::SelectOp>(op, cmp, op.getLhs(),
                                                 op.getRhs());
    return success();
  }
};

// Convert IndexCast ops
//===----------------------------------------------------------------------===//

/// Returns true iff the type is `index` or `vector<...index>`.
static bool isIndexOrIndexVector(Type type) {
  if (isa<IndexType>(type))
    return true;

  if (auto vectorTy = dyn_cast<VectorType>(type))
    if (isa<IndexType>(vectorTy.getElementType()))
      return true;

  return false;
}

template <typename CastOp>
struct ConvertIndexCastIntToIndex final : OpConversionPattern<CastOp> {
  using OpConversionPattern<CastOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(CastOp op, typename CastOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Type resultType = op.getType();
    if (!isIndexOrIndexVector(resultType))
      return failure();

    Location loc = op.getLoc();
    Type inType = op.getIn().getType();
    auto newInTy =
        this->getTypeConverter()->template convertType<VectorType>(inType);
    if (!newInTy)
      return rewriter.notifyMatchFailure(
          loc, llvm::formatv("unsupported type: {0}", inType));

    // Discard the high half of the input truncating the original value.
    Value extracted = extractLastDimSlice(rewriter, loc, adaptor.getIn(), 0);
    extracted = dropTrailingX1Dim(rewriter, loc, extracted);
    rewriter.replaceOpWithNewOp<CastOp>(op, resultType, extracted);
    return success();
  }
};

template <typename CastOp, typename ExtensionOp>
struct ConvertIndexCastIndexToInt final : OpConversionPattern<CastOp> {
  using OpConversionPattern<CastOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(CastOp op, typename CastOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Type inType = op.getIn().getType();
    if (!isIndexOrIndexVector(inType))
      return failure();

    Location loc = op.getLoc();
    auto *typeConverter =
        this->template getTypeConverter<arith::WideIntEmulationConverter>();

    Type resultType = op.getType();
    auto newTy = typeConverter->template convertType<VectorType>(resultType);
    if (!newTy)
      return rewriter.notifyMatchFailure(
          loc, llvm::formatv("unsupported type: {0}", resultType));

    // Emit an index cast over the matching narrow type.
    Type narrowTy =
        rewriter.getIntegerType(typeConverter->getMaxTargetIntBitWidth());
    if (auto vecTy = dyn_cast<VectorType>(resultType))
      narrowTy = VectorType::get(vecTy.getShape(), narrowTy);

    // Sign or zero-extend the result. Let the matching conversion pattern
    // legalize the extension op.
    Value underlyingVal =
        rewriter.create<CastOp>(loc, narrowTy, adaptor.getIn());
    rewriter.replaceOpWithNewOp<ExtensionOp>(op, resultType, underlyingVal);
    return success();
  }
};

//===----------------------------------------------------------------------===//
// ConvertSelect
//===----------------------------------------------------------------------===//

struct ConvertSelect final : OpConversionPattern<arith::SelectOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(arith::SelectOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op->getLoc();
    auto newTy = getTypeConverter()->convertType<VectorType>(op.getType());
    if (!newTy)
      return rewriter.notifyMatchFailure(
          loc, llvm::formatv("unsupported type: {0}", op.getType()));

    auto [trueElem0, trueElem1] =
        extractLastDimHalves(rewriter, loc, adaptor.getTrueValue());
    auto [falseElem0, falseElem1] =
        extractLastDimHalves(rewriter, loc, adaptor.getFalseValue());
    Value cond = appendX1Dim(rewriter, loc, adaptor.getCondition());

    Value resElem0 =
        rewriter.create<arith::SelectOp>(loc, cond, trueElem0, falseElem0);
    Value resElem1 =
        rewriter.create<arith::SelectOp>(loc, cond, trueElem1, falseElem1);
    Value resultVec =
        constructResultVector(rewriter, loc, newTy, {resElem0, resElem1});
    rewriter.replaceOp(op, resultVec);
    return success();
  }
};

//===----------------------------------------------------------------------===//
// ConvertShLI
//===----------------------------------------------------------------------===//

struct ConvertShLI final : OpConversionPattern<arith::ShLIOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(arith::ShLIOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op->getLoc();

    Type oldTy = op.getType();
    auto newTy = getTypeConverter()->convertType<VectorType>(oldTy);
    if (!newTy)
      return rewriter.notifyMatchFailure(
          loc, llvm::formatv("unsupported type: {0}", op.getType()));

    Type newOperandTy = reduceInnermostDim(newTy);
    // `oldBitWidth` == `2 * newBitWidth`
    unsigned newBitWidth = newTy.getElementTypeBitWidth();

    auto [lhsElem0, lhsElem1] =
        extractLastDimHalves(rewriter, loc, adaptor.getLhs());
    Value rhsElem0 = extractLastDimSlice(rewriter, loc, adaptor.getRhs(), 0);

    // Assume that the shift amount is < 2 * newBitWidth. Calculate the low and
    // high halves of the results separately:
    //   1. low := LHS.low shli RHS
    //
    //   2. high := a or b or c, where:
    //     a) Bits from LHS.high, shifted by the RHS.
    //     b) Bits from LHS.low, shifted right. These come into play when
    //        RHS < newBitWidth, e.g.:
    //         [0000][llll] shli 3 --> [0lll][l000]
    //                                    ^
    //                                    |
    //                           [llll] shrui (4 - 3)
    //     c) Bits from LHS.low, shifted left. These matter when
    //        RHS > newBitWidth, e.g.:
    //         [0000][llll] shli 7 --> [l000][0000]
    //                                   ^
    //                                   |
    //                          [llll] shli (7 - 4)
    //
    // Because shifts by values >= newBitWidth are undefined, we ignore the high
    // half of RHS, and introduce 'bounds checks' to account for
    // RHS.low > newBitWidth.
    //
    // TODO: Explore possible optimizations.
    Value zeroCst = createScalarOrSplatConstant(rewriter, loc, newOperandTy, 0);
    Value elemBitWidth =
        createScalarOrSplatConstant(rewriter, loc, newOperandTy, newBitWidth);

    Value illegalElemShift = rewriter.create<arith::CmpIOp>(
        loc, arith::CmpIPredicate::uge, rhsElem0, elemBitWidth);

    Value shiftedElem0 =
        rewriter.create<arith::ShLIOp>(loc, lhsElem0, rhsElem0);
    Value resElem0 = rewriter.create<arith::SelectOp>(loc, illegalElemShift,
                                                      zeroCst, shiftedElem0);

    Value cappedShiftAmount = rewriter.create<arith::SelectOp>(
        loc, illegalElemShift, elemBitWidth, rhsElem0);
    Value rightShiftAmount =
        rewriter.create<arith::SubIOp>(loc, elemBitWidth, cappedShiftAmount);
    Value shiftedRight =
        rewriter.create<arith::ShRUIOp>(loc, lhsElem0, rightShiftAmount);
    Value overshotShiftAmount =
        rewriter.create<arith::SubIOp>(loc, rhsElem0, elemBitWidth);
    Value shiftedLeft =
        rewriter.create<arith::ShLIOp>(loc, lhsElem0, overshotShiftAmount);

    Value shiftedElem1 =
        rewriter.create<arith::ShLIOp>(loc, lhsElem1, rhsElem0);
    Value resElem1High = rewriter.create<arith::SelectOp>(
        loc, illegalElemShift, zeroCst, shiftedElem1);
    Value resElem1Low = rewriter.create<arith::SelectOp>(
        loc, illegalElemShift, shiftedLeft, shiftedRight);
    Value resElem1 =
        rewriter.create<arith::OrIOp>(loc, resElem1Low, resElem1High);

    Value resultVec =
        constructResultVector(rewriter, loc, newTy, {resElem0, resElem1});
    rewriter.replaceOp(op, resultVec);
    return success();
  }
};

//===----------------------------------------------------------------------===//
// ConvertShRUI
//===----------------------------------------------------------------------===//

struct ConvertShRUI final : OpConversionPattern<arith::ShRUIOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(arith::ShRUIOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op->getLoc();

    Type oldTy = op.getType();
    auto newTy = getTypeConverter()->convertType<VectorType>(oldTy);
    if (!newTy)
      return rewriter.notifyMatchFailure(
          loc, llvm::formatv("unsupported type: {0}", op.getType()));

    Type newOperandTy = reduceInnermostDim(newTy);
    // `oldBitWidth` == `2 * newBitWidth`
    unsigned newBitWidth = newTy.getElementTypeBitWidth();

    auto [lhsElem0, lhsElem1] =
        extractLastDimHalves(rewriter, loc, adaptor.getLhs());
    Value rhsElem0 = extractLastDimSlice(rewriter, loc, adaptor.getRhs(), 0);

    // Assume that the shift amount is < 2 * newBitWidth. Calculate the low and
    // high halves of the results separately:
    //   1. low := a or b or c, where:
    //     a) Bits from LHS.low, shifted by the RHS.
    //     b) Bits from LHS.high, shifted left. These matter when
    //        RHS < newBitWidth, e.g.:
    //         [hhhh][0000] shrui 3 --> [000h][hhh0]
    //                                          ^
    //                                          |
    //                                 [hhhh] shli (4 - 1)
    //     c) Bits from LHS.high, shifted right. These come into play when
    //        RHS > newBitWidth, e.g.:
    //         [hhhh][0000] shrui 7 --> [0000][000h]
    //                                          ^
    //                                          |
    //                                 [hhhh] shrui (7 - 4)
    //
    //   2. high := LHS.high shrui RHS
    //
    // Because shifts by values >= newBitWidth are undefined, we ignore the high
    // half of RHS, and introduce 'bounds checks' to account for
    // RHS.low > newBitWidth.
    //
    // TODO: Explore possible optimizations.
    Value zeroCst = createScalarOrSplatConstant(rewriter, loc, newOperandTy, 0);
    Value elemBitWidth =
        createScalarOrSplatConstant(rewriter, loc, newOperandTy, newBitWidth);

    Value illegalElemShift = rewriter.create<arith::CmpIOp>(
        loc, arith::CmpIPredicate::uge, rhsElem0, elemBitWidth);

    Value shiftedElem0 =
        rewriter.create<arith::ShRUIOp>(loc, lhsElem0, rhsElem0);
    Value resElem0Low = rewriter.create<arith::SelectOp>(loc, illegalElemShift,
                                                         zeroCst, shiftedElem0);
    Value shiftedElem1 =
        rewriter.create<arith::ShRUIOp>(loc, lhsElem1, rhsElem0);
    Value resElem1 = rewriter.create<arith::SelectOp>(loc, illegalElemShift,
                                                      zeroCst, shiftedElem1);

    Value cappedShiftAmount = rewriter.create<arith::SelectOp>(
        loc, illegalElemShift, elemBitWidth, rhsElem0);
    Value leftShiftAmount =
        rewriter.create<arith::SubIOp>(loc, elemBitWidth, cappedShiftAmount);
    Value shiftedLeft =
        rewriter.create<arith::ShLIOp>(loc, lhsElem1, leftShiftAmount);
    Value overshotShiftAmount =
        rewriter.create<arith::SubIOp>(loc, rhsElem0, elemBitWidth);
    Value shiftedRight =
        rewriter.create<arith::ShRUIOp>(loc, lhsElem1, overshotShiftAmount);

    Value resElem0High = rewriter.create<arith::SelectOp>(
        loc, illegalElemShift, shiftedRight, shiftedLeft);
    Value resElem0 =
        rewriter.create<arith::OrIOp>(loc, resElem0Low, resElem0High);

    Value resultVec =
        constructResultVector(rewriter, loc, newTy, {resElem0, resElem1});
    rewriter.replaceOp(op, resultVec);
    return success();
  }
};

//===----------------------------------------------------------------------===//
// ConvertShRSI
//===----------------------------------------------------------------------===//

struct ConvertShRSI final : OpConversionPattern<arith::ShRSIOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(arith::ShRSIOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op->getLoc();

    Type oldTy = op.getType();
    auto newTy = getTypeConverter()->convertType<VectorType>(oldTy);
    if (!newTy)
      return rewriter.notifyMatchFailure(
          loc, llvm::formatv("unsupported type: {0}", op.getType()));

    Value lhsElem1 = extractLastDimSlice(rewriter, loc, adaptor.getLhs(), 1);
    Value rhsElem0 = extractLastDimSlice(rewriter, loc, adaptor.getRhs(), 0);

    Type narrowTy = rhsElem0.getType();
    int64_t origBitwidth = newTy.getElementTypeBitWidth() * 2;

    // Rewrite this as an bitwise or of `arith.shrui` and sign extension bits.
    // Perform as many ops over the narrow integer type as possible and let the
    // other emulation patterns convert the rest.
    Value elemZero = createScalarOrSplatConstant(rewriter, loc, narrowTy, 0);
    Value signBit = rewriter.create<arith::CmpIOp>(
        loc, arith::CmpIPredicate::slt, lhsElem1, elemZero);
    signBit = dropTrailingX1Dim(rewriter, loc, signBit);

    // Create a bit pattern of either all ones or all zeros. Then shift it left
    // to calculate the sign extension bits created by shifting the original
    // sign bit right.
    Value allSign = rewriter.create<arith::ExtSIOp>(loc, oldTy, signBit);
    Value maxShift =
        createScalarOrSplatConstant(rewriter, loc, narrowTy, origBitwidth);
    Value numNonSignExtBits =
        rewriter.create<arith::SubIOp>(loc, maxShift, rhsElem0);
    numNonSignExtBits = dropTrailingX1Dim(rewriter, loc, numNonSignExtBits);
    numNonSignExtBits =
        rewriter.create<arith::ExtUIOp>(loc, oldTy, numNonSignExtBits);
    Value signBits =
        rewriter.create<arith::ShLIOp>(loc, allSign, numNonSignExtBits);

    // Use original arguments to create the right shift.
    Value shrui =
        rewriter.create<arith::ShRUIOp>(loc, op.getLhs(), op.getRhs());
    Value shrsi = rewriter.create<arith::OrIOp>(loc, shrui, signBits);

    // Handle shifting by zero. This is necessary when the `signBits` shift is
    // invalid.
    Value isNoop = rewriter.create<arith::CmpIOp>(loc, arith::CmpIPredicate::eq,
                                                  rhsElem0, elemZero);
    isNoop = dropTrailingX1Dim(rewriter, loc, isNoop);
    rewriter.replaceOpWithNewOp<arith::SelectOp>(op, isNoop, op.getLhs(),
                                                 shrsi);

    return success();
  }
};

//===----------------------------------------------------------------------===//
// ConvertSIToFP
//===----------------------------------------------------------------------===//

struct ConvertSIToFP final : OpConversionPattern<arith::SIToFPOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(arith::SIToFPOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();

    Value in = op.getIn();
    Type oldTy = in.getType();
    auto newTy = getTypeConverter()->convertType<VectorType>(oldTy);
    if (!newTy)
      return rewriter.notifyMatchFailure(
          loc, llvm::formatv("unsupported type: {0}", oldTy));

    unsigned oldBitWidth = getElementTypeOrSelf(oldTy).getIntOrFloatBitWidth();
    Value zeroCst = createScalarOrSplatConstant(rewriter, loc, oldTy, 0);
    Value oneCst = createScalarOrSplatConstant(rewriter, loc, oldTy, 1);
    Value allOnesCst = createScalarOrSplatConstant(
        rewriter, loc, oldTy, APInt::getAllOnes(oldBitWidth));

    // To avoid operating on very large unsigned numbers, perform the
    // conversion on the absolute value. Then, decide whether to negate the
    // result or not based on that sign bit. We assume two's complement and
    // implement negation by flipping all bits and adding 1.
    // Note that this relies on the the other conversion patterns to legalize
    // created ops and narrow the bit widths.
    Value isNeg = rewriter.create<arith::CmpIOp>(loc, arith::CmpIPredicate::slt,
                                                 in, zeroCst);
    Value bitwiseNeg = rewriter.create<arith::XOrIOp>(loc, in, allOnesCst);
    Value neg = rewriter.create<arith::AddIOp>(loc, bitwiseNeg, oneCst);
    Value abs = rewriter.create<arith::SelectOp>(loc, isNeg, neg, in);

    Value absResult = rewriter.create<arith::UIToFPOp>(loc, op.getType(), abs);
    Value negResult = rewriter.create<arith::NegFOp>(loc, absResult);
    rewriter.replaceOpWithNewOp<arith::SelectOp>(op, isNeg, negResult,
                                                 absResult);
    return success();
  }
};

//===----------------------------------------------------------------------===//
// ConvertUIToFP
//===----------------------------------------------------------------------===//

struct ConvertUIToFP final : OpConversionPattern<arith::UIToFPOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(arith::UIToFPOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();

    Type oldTy = op.getIn().getType();
    auto newTy = getTypeConverter()->convertType<VectorType>(oldTy);
    if (!newTy)
      return rewriter.notifyMatchFailure(
          loc, llvm::formatv("unsupported type: {0}", oldTy));
    unsigned newBitWidth = newTy.getElementTypeBitWidth();

    auto [low, hi] = extractLastDimHalves(rewriter, loc, adaptor.getIn());
    Value lowInt = dropTrailingX1Dim(rewriter, loc, low);
    Value hiInt = dropTrailingX1Dim(rewriter, loc, hi);
    Value zeroCst =
        createScalarOrSplatConstant(rewriter, loc, hiInt.getType(), 0);

    // The final result has the following form:
    //   if (hi == 0) return uitofp(low)
    //   else         return uitofp(low) + uitofp(hi) * 2^BW
    //
    // where `BW` is the bitwidth of the narrowed integer type. We emit a
    // select to make it easier to fold-away the `hi` part calculation when it
    // is known to be zero.
    //
    // Note 1: The emulation is precise only for input values that have exact
    // integer representation in the result floating point type, and may lead
    // loss of precision otherwise.
    //
    // Note 2: We do not strictly need the `hi == 0`, case, but it makes
    // constant folding easier.
    Value hiEqZero = rewriter.create<arith::CmpIOp>(
        loc, arith::CmpIPredicate::eq, hiInt, zeroCst);

    Type resultTy = op.getType();
    Type resultElemTy = getElementTypeOrSelf(resultTy);
    Value lowFp = rewriter.create<arith::UIToFPOp>(loc, resultTy, lowInt);
    Value hiFp = rewriter.create<arith::UIToFPOp>(loc, resultTy, hiInt);

    int64_t pow2Int = int64_t(1) << newBitWidth;
    TypedAttr pow2Attr =
        rewriter.getFloatAttr(resultElemTy, static_cast<double>(pow2Int));
    if (auto vecTy = dyn_cast<VectorType>(resultTy))
      pow2Attr = SplatElementsAttr::get(vecTy, pow2Attr);

    Value pow2Val = rewriter.create<arith::ConstantOp>(loc, resultTy, pow2Attr);

    Value hiVal = rewriter.create<arith::MulFOp>(loc, hiFp, pow2Val);
    Value result = rewriter.create<arith::AddFOp>(loc, lowFp, hiVal);

    rewriter.replaceOpWithNewOp<arith::SelectOp>(op, hiEqZero, lowFp, result);
    return success();
  }
};

//===----------------------------------------------------------------------===//
// ConvertTruncI
//===----------------------------------------------------------------------===//

struct ConvertTruncI final : OpConversionPattern<arith::TruncIOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(arith::TruncIOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    // Check if the result type is legal for this target. Currently, we do not
    // support truncation to types wider than supported by the target.
    if (!getTypeConverter()->isLegal(op.getType()))
      return rewriter.notifyMatchFailure(
          loc, llvm::formatv("unsupported truncation result type: {0}",
                             op.getType()));

    // Discard the high half of the input. Truncate the low half, if
    // necessary.
    Value extracted = extractLastDimSlice(rewriter, loc, adaptor.getIn(), 0);
    extracted = dropTrailingX1Dim(rewriter, loc, extracted);
    Value truncated =
        rewriter.createOrFold<arith::TruncIOp>(loc, op.getType(), extracted);
    rewriter.replaceOp(op, truncated);
    return success();
  }
};

//===----------------------------------------------------------------------===//
// ConvertVectorPrint
//===----------------------------------------------------------------------===//

struct ConvertVectorPrint final : OpConversionPattern<vector::PrintOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(vector::PrintOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<vector::PrintOp>(op, adaptor.getSource());
    return success();
  }
};

//===----------------------------------------------------------------------===//
// Pass Definition
//===----------------------------------------------------------------------===//

struct EmulateWideIntPass final
    : arith::impl::ArithEmulateWideIntBase<EmulateWideIntPass> {
  using ArithEmulateWideIntBase::ArithEmulateWideIntBase;

  void runOnOperation() override {
    if (!llvm::isPowerOf2_32(widestIntSupported) || widestIntSupported < 2) {
      signalPassFailure();
      return;
    }

    Operation *op = getOperation();
    MLIRContext *ctx = op->getContext();

    arith::WideIntEmulationConverter typeConverter(widestIntSupported);
    ConversionTarget target(*ctx);
    target.addDynamicallyLegalOp<func::FuncOp>([&typeConverter](Operation *op) {
      return typeConverter.isLegal(cast<func::FuncOp>(op).getFunctionType());
    });
    auto opLegalCallback = [&typeConverter](Operation *op) {
      return typeConverter.isLegal(op);
    };
    target.addDynamicallyLegalOp<func::CallOp, func::ReturnOp>(opLegalCallback);
    target
        .addDynamicallyLegalDialect<arith::ArithDialect, vector::VectorDialect>(
            opLegalCallback);

    RewritePatternSet patterns(ctx);
    arith::populateArithWideIntEmulationPatterns(typeConverter, patterns);

    if (failed(applyPartialConversion(op, target, std::move(patterns))))
      signalPassFailure();
  }
};
} // end anonymous namespace

//===----------------------------------------------------------------------===//
// Public Interface Definition
//===----------------------------------------------------------------------===//

arith::WideIntEmulationConverter::WideIntEmulationConverter(
    unsigned widestIntSupportedByTarget)
    : maxIntWidth(widestIntSupportedByTarget) {
  assert(llvm::isPowerOf2_32(widestIntSupportedByTarget) &&
         "Only power-of-two integers with are supported");
  assert(widestIntSupportedByTarget >= 2 && "Integer type too narrow");

  // Allow unknown types.
  addConversion([](Type ty) -> std::optional<Type> { return ty; });

  // Scalar case.
  addConversion([this](IntegerType ty) -> std::optional<Type> {
    unsigned width = ty.getWidth();
    if (width <= maxIntWidth)
      return ty;

    // i2N --> vector<2xiN>
    if (width == 2 * maxIntWidth)
      return VectorType::get(2, IntegerType::get(ty.getContext(), maxIntWidth));

    return std::nullopt;
  });

  // Vector case.
  addConversion([this](VectorType ty) -> std::optional<Type> {
    auto intTy = dyn_cast<IntegerType>(ty.getElementType());
    if (!intTy)
      return ty;

    unsigned width = intTy.getWidth();
    if (width <= maxIntWidth)
      return ty;

    // vector<...xi2N> --> vector<...x2xiN>
    if (width == 2 * maxIntWidth) {
      auto newShape = to_vector(ty.getShape());
      newShape.push_back(2);
      return VectorType::get(newShape,
                             IntegerType::get(ty.getContext(), maxIntWidth));
    }

    return std::nullopt;
  });

  // Function case.
  addConversion([this](FunctionType ty) -> std::optional<Type> {
    // Convert inputs and results, e.g.:
    //   (i2N, i2N) -> i2N --> (vector<2xiN>, vector<2xiN>) -> vector<2xiN>
    SmallVector<Type> inputs;
    if (failed(convertTypes(ty.getInputs(), inputs)))
      return std::nullopt;

    SmallVector<Type> results;
    if (failed(convertTypes(ty.getResults(), results)))
      return std::nullopt;

    return FunctionType::get(ty.getContext(), inputs, results);
  });
}

void arith::populateArithWideIntEmulationPatterns(
    WideIntEmulationConverter &typeConverter, RewritePatternSet &patterns) {
  // Populate `func.*` conversion patterns.
  populateFunctionOpInterfaceTypeConversionPattern<func::FuncOp>(patterns,
                                                                 typeConverter);
  populateCallOpTypeConversionPattern(patterns, typeConverter);
  populateReturnOpTypeConversionPattern(patterns, typeConverter);

  // Populate `arith.*` conversion patterns.
  patterns.add<
      // Misc ops.
      ConvertConstant, ConvertCmpI, ConvertSelect, ConvertVectorPrint,
      // Binary ops.
      ConvertAddI, ConvertMulI, ConvertShLI, ConvertShRSI, ConvertShRUI,
      ConvertMaxMin<arith::MaxUIOp, arith::CmpIPredicate::ugt>,
      ConvertMaxMin<arith::MaxSIOp, arith::CmpIPredicate::sgt>,
      ConvertMaxMin<arith::MinUIOp, arith::CmpIPredicate::ult>,
      ConvertMaxMin<arith::MinSIOp, arith::CmpIPredicate::slt>,
      // Bitwise binary ops.
      ConvertBitwiseBinary<arith::AndIOp>, ConvertBitwiseBinary<arith::OrIOp>,
      ConvertBitwiseBinary<arith::XOrIOp>,
      // Extension and truncation ops.
      ConvertExtSI, ConvertExtUI, ConvertTruncI,
      // Cast ops.
      ConvertIndexCastIntToIndex<arith::IndexCastOp>,
      ConvertIndexCastIntToIndex<arith::IndexCastUIOp>,
      ConvertIndexCastIndexToInt<arith::IndexCastOp, arith::ExtSIOp>,
      ConvertIndexCastIndexToInt<arith::IndexCastUIOp, arith::ExtUIOp>,
      ConvertSIToFP, ConvertUIToFP>(typeConverter, patterns.getContext());
}

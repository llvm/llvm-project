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
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Func/Transforms/FuncConversions.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/Transforms/DialectConversion.h"
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

/// Returns the type with the last (innermost) dimention reduced to x1.
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

/// Returns a constant of integer of vector type filled with (repeated) `value`.
static Value createScalarOrSplatConstant(ConversionPatternRewriter &rewriter,
                                         Location loc, Type type,
                                         const APInt &value) {
  Attribute attr;
  if (auto intTy = type.dyn_cast<IntegerType>()) {
    attr = rewriter.getIntegerAttr(type, value);
  } else {
    auto vecTy = type.cast<VectorType>();
    attr = SplatElementsAttr::get(vecTy, value);
  }

  return rewriter.create<arith::ConstantOp>(loc, attr);
}

/// Returns a constant of integer of vector type filled with (repeated) `value`.
static Value createScalarOrSplatConstant(ConversionPatternRewriter &rewriter,
                                         Location loc, Type type,
                                         int64_t value) {
  unsigned elementBitWidth = 0;
  if (auto intTy = type.dyn_cast<IntegerType>())
    elementBitWidth = intTy.getWidth();
  else
    elementBitWidth = type.cast<VectorType>().getElementTypeBitWidth();

  return createScalarOrSplatConstant(rewriter, loc, type,
                                     APInt(elementBitWidth, value));
}

/// Extracts the `input` vector slice with elements at the last dimension offset
/// by `lastOffset`. Returns a value of vector type with the last dimension
/// reduced to x1 or fully scalarized, e.g.:
///   - vector<3x2xi16> --> vector<3x1xi16>
///   - vector<2xi16>   --> i16
static Value extractLastDimSlice(ConversionPatternRewriter &rewriter,
                                 Location loc, Value input,
                                 int64_t lastOffset) {
  ArrayRef<int64_t> shape = input.getType().cast<VectorType>().getShape();
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
  auto vecTy = input.getType().dyn_cast<VectorType>();
  if (!vecTy)
    return input;

  // Shape cast to drop the last x1 dimention.
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
  auto vecTy = input.getType().dyn_cast<VectorType>();
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
  ArrayRef<int64_t> shape = dest.getType().cast<VectorType>().getShape();
  assert(lastOffset < shape.back() && "Offset out of bounds");

  // Handle scalar source.
  if (source.getType().isa<IntegerType>())
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
/// `vector<...xNxT>`, where `N` is the number of `resultComponenets`.
static Value constructResultVector(ConversionPatternRewriter &rewriter,
                                   Location loc, VectorType resultType,
                                   ValueRange resultComponents) {
  llvm::ArrayRef<int64_t> resultShape = resultType.getShape();
  (void)resultShape;
  assert(!resultShape.empty() && "Result expected to have dimentions");
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
    auto newType = getTypeConverter()->convertType(oldType).cast<VectorType>();
    unsigned newBitWidth = newType.getElementTypeBitWidth();
    Attribute oldValue = op.getValueAttr();

    if (auto intAttr = oldValue.dyn_cast<IntegerAttr>()) {
      auto [low, high] = getHalves(intAttr.getValue(), newBitWidth);
      auto newAttr = DenseElementsAttr::get(newType, {low, high});
      rewriter.replaceOpWithNewOp<arith::ConstantOp>(op, newAttr);
      return success();
    }

    if (auto splatAttr = oldValue.dyn_cast<SplatElementsAttr>()) {
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

    if (auto elemsAttr = oldValue.dyn_cast<DenseElementsAttr>()) {
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
    auto newTy = getTypeConverter()
                     ->convertType(op.getType())
                     .dyn_cast_or_null<VectorType>();
    if (!newTy)
      return rewriter.notifyMatchFailure(loc, "expected scalar or vector type");

    Type newElemTy = reduceInnermostDim(newTy);

    auto [lhsElem0, lhsElem1] =
        extractLastDimHalves(rewriter, loc, adaptor.getLhs());
    auto [rhsElem0, rhsElem1] =
        extractLastDimHalves(rewriter, loc, adaptor.getRhs());

    auto lowSum = rewriter.create<arith::AddUICarryOp>(loc, lhsElem0, rhsElem0);
    Value carryVal =
        rewriter.create<arith::ExtUIOp>(loc, newElemTy, lowSum.getCarry());

    Value high0 = rewriter.create<arith::AddIOp>(loc, carryVal, lhsElem1);
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
    auto newTy = this->getTypeConverter()
                     ->convertType(op.getType())
                     .template dyn_cast_or_null<VectorType>();
    if (!newTy)
      return rewriter.notifyMatchFailure(loc, "unsupported type");

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
// ConvertMulI
//===----------------------------------------------------------------------===//

struct ConvertMulI final : OpConversionPattern<arith::MulIOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(arith::MulIOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op->getLoc();
    auto newTy = getTypeConverter()
                     ->convertType(op.getType())
                     .dyn_cast_or_null<VectorType>();
    if (!newTy)
      return rewriter.notifyMatchFailure(loc, "expected scalar or vector type");

    Type newElemTy = reduceInnermostDim(newTy);
    unsigned newBitWidth = newTy.getElementTypeBitWidth();
    unsigned digitBitWidth = newBitWidth / 2;

    auto [lhsElem0, lhsElem1] =
        extractLastDimHalves(rewriter, loc, adaptor.getLhs());
    auto [rhsElem0, rhsElem1] =
        extractLastDimHalves(rewriter, loc, adaptor.getRhs());

    // Emulate multiplication by splitting each input element of type i2N into 4
    // digits of type iN and bit width i(N/2). This is so that the intermediate
    // multiplications and additions do not overflow. We extract these i(N/2)
    // digits from iN vector elements by masking (low digit) and shifting right
    // (high digit).
    //
    // The multiplication algorithm used is the standard (long) multiplication.
    // Multiplying two i2N integers produces (at most) a i4N result, but because
    // the calculation of top i2N is not necessary, we omit it.
    // In total, this implementations performs 10 intermediate multiplications
    // and 16 additions. The number of multiplications could be decreased by
    // switching to a more efficient algorithm like Karatsuba. This would,
    // however, require being able to perform (intermediate) wide additions and
    // subtractions, so it is not clear that such implementation would be more
    // efficient.

    APInt lowMaskVal(newBitWidth, 1);
    lowMaskVal = lowMaskVal.shl(digitBitWidth) - 1;
    Value lowMask =
        createScalarOrSplatConstant(rewriter, loc, newElemTy, lowMaskVal);
    auto getLowDigit = [lowMask, newElemTy, loc, &rewriter](Value v) {
      return rewriter.create<arith::AndIOp>(loc, newElemTy, v, lowMask);
    };

    Value shiftVal =
        createScalarOrSplatConstant(rewriter, loc, newElemTy, digitBitWidth);
    auto getHighDigit = [shiftVal, loc, &rewriter](Value v) {
      return rewriter.create<arith::ShRUIOp>(loc, v, shiftVal);
    };

    Value zeroDigit = createScalarOrSplatConstant(rewriter, loc, newElemTy, 0);
    std::array<Value, 4> resultDigits = {zeroDigit, zeroDigit, zeroDigit,
                                         zeroDigit};
    std::array<Value, 4> lhsDigits = {
        getLowDigit(lhsElem0), getHighDigit(lhsElem0), getLowDigit(lhsElem1),
        getHighDigit(lhsElem1)};
    std::array<Value, 4> rhsDigits = {
        getLowDigit(rhsElem0), getHighDigit(rhsElem0), getLowDigit(rhsElem1),
        getHighDigit(rhsElem1)};

    for (unsigned i = 0, e = lhsDigits.size(); i != e; ++i) {
      for (unsigned j = 0; i + j != e; ++j) {
        Value mul =
            rewriter.create<arith::MulIOp>(loc, lhsDigits[i], rhsDigits[j]);
        Value current =
            rewriter.createOrFold<arith::AddIOp>(loc, resultDigits[i + j], mul);
        resultDigits[i + j] = getLowDigit(current);
        if (i + j + 1 != e) {
          Value carry = rewriter.createOrFold<arith::AddIOp>(
              loc, resultDigits[i + j + 1], getHighDigit(current));
          resultDigits[i + j + 1] = carry;
        }
      }
    }

    auto combineDigits = [shiftVal, loc, &rewriter](Value low, Value high) {
      Value highBits = rewriter.create<arith::ShLIOp>(loc, high, shiftVal);
      return rewriter.create<arith::OrIOp>(loc, low, highBits);
    };
    Value resultElem0 = combineDigits(resultDigits[0], resultDigits[1]);
    Value resultElem1 = combineDigits(resultDigits[2], resultDigits[3]);
    Value resultVec =
        constructResultVector(rewriter, loc, newTy, {resultElem0, resultElem1});
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
    auto newTy = getTypeConverter()
                     ->convertType(op.getType())
                     .dyn_cast_or_null<VectorType>();
    if (!newTy)
      return rewriter.notifyMatchFailure(loc, "unsupported type");

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
    auto newTy = getTypeConverter()
                     ->convertType(op.getType())
                     .dyn_cast_or_null<VectorType>();
    if (!newTy)
      return rewriter.notifyMatchFailure(loc, "unsupported type");

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
// ConvertShLI
//===----------------------------------------------------------------------===//

struct ConvertShLI final : OpConversionPattern<arith::ShLIOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(arith::ShLIOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op->getLoc();

    Type oldTy = op.getType();
    auto newTy =
        getTypeConverter()->convertType(oldTy).dyn_cast_or_null<VectorType>();
    if (!newTy)
      return rewriter.notifyMatchFailure(loc, "unsupported type");

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
    auto newTy =
        getTypeConverter()->convertType(oldTy).dyn_cast_or_null<VectorType>();
    if (!newTy)
      return rewriter.notifyMatchFailure(loc, "unsupported type");

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
      return rewriter.notifyMatchFailure(loc,
                                         "unsupported truncation result type");

    // Discard the high half of the input. Truncate the low half, if necessary.
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
    arith::populateWideIntEmulationPatterns(typeConverter, patterns);

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
  addConversion([](Type ty) -> Optional<Type> { return ty; });

  // Scalar case.
  addConversion([this](IntegerType ty) -> Optional<Type> {
    unsigned width = ty.getWidth();
    if (width <= maxIntWidth)
      return ty;

    // i2N --> vector<2xiN>
    if (width == 2 * maxIntWidth)
      return VectorType::get(2, IntegerType::get(ty.getContext(), maxIntWidth));

    return None;
  });

  // Vector case.
  addConversion([this](VectorType ty) -> Optional<Type> {
    auto intTy = ty.getElementType().dyn_cast<IntegerType>();
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

    return None;
  });

  // Function case.
  addConversion([this](FunctionType ty) -> Optional<Type> {
    // Convert inputs and results, e.g.:
    //   (i2N, i2N) -> i2N --> (vector<2xiN>, vector<2xiN>) -> vector<2xiN>
    SmallVector<Type> inputs;
    if (failed(convertTypes(ty.getInputs(), inputs)))
      return None;

    SmallVector<Type> results;
    if (failed(convertTypes(ty.getResults(), results)))
      return None;

    return FunctionType::get(ty.getContext(), inputs, results);
  });
}

void arith::populateWideIntEmulationPatterns(
    WideIntEmulationConverter &typeConverter, RewritePatternSet &patterns) {
  // Populate `func.*` conversion patterns.
  populateFunctionOpInterfaceTypeConversionPattern<func::FuncOp>(patterns,
                                                                 typeConverter);
  populateCallOpTypeConversionPattern(patterns, typeConverter);
  populateReturnOpTypeConversionPattern(patterns, typeConverter);

  // Populate `arith.*` conversion patterns.
  patterns.add<
      // Misc ops.
      ConvertConstant, ConvertVectorPrint,
      // Binary ops.
      ConvertAddI, ConvertMulI, ConvertShLI, ConvertShRUI,
      // Bitwise binary ops.
      ConvertBitwiseBinary<arith::AndIOp>, ConvertBitwiseBinary<arith::OrIOp>,
      ConvertBitwiseBinary<arith::XOrIOp>,
      // Extension and truncation ops.
      ConvertExtSI, ConvertExtUI, ConvertTruncI>(typeConverter,
                                                 patterns.getContext());
}

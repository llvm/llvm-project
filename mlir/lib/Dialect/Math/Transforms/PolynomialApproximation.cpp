//===- PolynomialApproximation.cpp - Approximate math operations ----------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements expansion of math operations to fast approximations
// that do not rely on any of the library functions.
//
//===----------------------------------------------------------------------===//

#include <climits>
#include <cstddef>

#include "mlir/Dialect/Arithmetic/IR/Arithmetic.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/Math/Transforms/Approximation.h"
#include "mlir/Dialect/Math/Transforms/Passes.h"
#include "mlir/Dialect/Utils/IndexingUtils.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/Dialect/Vector/Utils/VectorUtils.h"
#include "mlir/Dialect/X86Vector/X86VectorDialect.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/ImplicitLocOpBuilder.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/TypeUtilities.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/STLExtras.h"

using namespace mlir;
using namespace mlir::math;
using namespace mlir::vector;

// Returns vector shape if the type is a vector. Returns an empty shape if it is
// not a vector.
static ArrayRef<int64_t> vectorShape(Type type) {
  auto vectorType = type.dyn_cast<VectorType>();
  return vectorType ? vectorType.getShape() : ArrayRef<int64_t>();
}

static ArrayRef<int64_t> vectorShape(Value value) {
  return vectorShape(value.getType());
}

//----------------------------------------------------------------------------//
// Broadcast scalar types and values into vector types and values.
//----------------------------------------------------------------------------//

// Broadcasts scalar type into vector type (iff shape is non-scalar).
static Type broadcast(Type type, ArrayRef<int64_t> shape) {
  assert(!type.isa<VectorType>() && "must be scalar type");
  return !shape.empty() ? VectorType::get(shape, type) : type;
}

// Broadcasts scalar value into vector (iff shape is non-scalar).
static Value broadcast(ImplicitLocOpBuilder &builder, Value value,
                       ArrayRef<int64_t> shape) {
  assert(!value.getType().isa<VectorType>() && "must be scalar value");
  auto type = broadcast(value.getType(), shape);
  return !shape.empty() ? builder.create<BroadcastOp>(type, value) : value;
}

//----------------------------------------------------------------------------//
// Helper function to handle n-D vectors with 1-D operations.
//----------------------------------------------------------------------------//

// Expands and unrolls n-D vector operands into multiple fixed size 1-D vectors
// and calls the compute function with 1-D vector operands. Stitches back all
// results into the original n-D vector result.
//
// Examples: vectorWidth = 8
//   - vector<4x8xf32> unrolled 4 times
//   - vector<16xf32> expanded to vector<2x8xf32> and unrolled 2 times
//   - vector<4x16xf32> expanded to vector<4x2x8xf32> and unrolled 4*2 times
//
// Some math approximations rely on ISA-specific operations that only accept
// fixed size 1-D vectors (e.g. AVX expects vectors of width 8).
//
// It is the caller's responsibility to verify that the inner dimension is
// divisible by the vectorWidth, and that all operands have the same vector
// shape.
static Value
handleMultidimensionalVectors(ImplicitLocOpBuilder &builder,
                              ValueRange operands, int64_t vectorWidth,
                              llvm::function_ref<Value(ValueRange)> compute) {
  assert(!operands.empty() && "operands must be not empty");
  assert(vectorWidth > 0 && "vector width must be larger than 0");

  VectorType inputType = operands[0].getType().cast<VectorType>();
  ArrayRef<int64_t> inputShape = inputType.getShape();

  // If input shape matches target vector width, we can just call the
  // user-provided compute function with the operands.
  if (inputShape == llvm::makeArrayRef(vectorWidth))
    return compute(operands);

  // Check if the inner dimension has to be expanded, or we can directly iterate
  // over the outer dimensions of the vector.
  int64_t innerDim = inputShape.back();
  int64_t expansionDim = innerDim / vectorWidth;
  assert((innerDim % vectorWidth == 0) && "invalid inner dimension size");

  // Maybe expand operands to the higher rank vector shape that we'll use to
  // iterate over and extract one dimensional vectors.
  SmallVector<int64_t> expandedShape(inputShape.begin(), inputShape.end());
  SmallVector<Value> expandedOperands(operands);

  if (expansionDim > 1) {
    // Expand shape from [..., innerDim] to [..., expansionDim, vectorWidth].
    expandedShape.insert(expandedShape.end() - 1, expansionDim);
    expandedShape.back() = vectorWidth;

    for (unsigned i = 0; i < operands.size(); ++i) {
      auto operand = operands[i];
      auto eltType = operand.getType().cast<VectorType>().getElementType();
      auto expandedType = VectorType::get(expandedShape, eltType);
      expandedOperands[i] =
          builder.create<vector::ShapeCastOp>(expandedType, operand);
    }
  }

  // Iterate over all outer dimensions of the compute shape vector type.
  auto iterationDims = ArrayRef<int64_t>(expandedShape).drop_back();
  int64_t maxLinearIndex = computeMaxLinearIndex(iterationDims);

  SmallVector<int64_t> ones(iterationDims.size(), 1);
  auto strides = computeStrides(iterationDims, ones);

  // Compute results for each one dimensional vector.
  SmallVector<Value> results(maxLinearIndex);

  for (int64_t i = 0; i < maxLinearIndex; ++i) {
    auto offsets = delinearize(strides, i);

    SmallVector<Value> extracted(expandedOperands.size());
    for (const auto &tuple : llvm::enumerate(expandedOperands))
      extracted[tuple.index()] =
          builder.create<vector::ExtractOp>(tuple.value(), offsets);

    results[i] = compute(extracted);
  }

  // Stitch results together into one large vector.
  Type resultEltType = results[0].getType().cast<VectorType>().getElementType();
  Type resultExpandedType = VectorType::get(expandedShape, resultEltType);
  Value result = builder.create<arith::ConstantOp>(
      resultExpandedType, builder.getZeroAttr(resultExpandedType));

  for (int64_t i = 0; i < maxLinearIndex; ++i)
    result = builder.create<vector::InsertOp>(results[i], result,
                                              delinearize(strides, i));

  // Reshape back to the original vector shape.
  return builder.create<vector::ShapeCastOp>(
      VectorType::get(inputShape, resultEltType), result);
}

//----------------------------------------------------------------------------//
// Helper functions to create constants.
//----------------------------------------------------------------------------//

static Value f32Cst(ImplicitLocOpBuilder &builder, float value) {
  return builder.create<arith::ConstantOp>(builder.getF32FloatAttr(value));
}

static Value i32Cst(ImplicitLocOpBuilder &builder, int32_t value) {
  return builder.create<arith::ConstantOp>(builder.getI32IntegerAttr(value));
}

static Value f32FromBits(ImplicitLocOpBuilder &builder, uint32_t bits) {
  Value i32Value = i32Cst(builder, static_cast<int32_t>(bits));
  return builder.create<arith::BitcastOp>(builder.getF32Type(), i32Value);
}

//----------------------------------------------------------------------------//
// Helper functions to build math functions approximations.
//----------------------------------------------------------------------------//

static Value min(ImplicitLocOpBuilder &builder, Value a, Value b) {
  return builder.create<arith::SelectOp>(
      builder.create<arith::CmpFOp>(arith::CmpFPredicate::OLT, a, b), a, b);
}

static Value max(ImplicitLocOpBuilder &builder, Value a, Value b) {
  return builder.create<arith::SelectOp>(
      builder.create<arith::CmpFOp>(arith::CmpFPredicate::OGT, a, b), a, b);
}

static Value clamp(ImplicitLocOpBuilder &builder, Value value, Value lowerBound,
                   Value upperBound) {
  return max(builder, min(builder, value, upperBound), lowerBound);
}

// Decomposes given floating point value `arg` into a normalized fraction and
// an integral power of two (see std::frexp). Returned values have float type.
static std::pair<Value, Value> frexp(ImplicitLocOpBuilder &builder, Value arg,
                                     bool isPositive = false) {
  assert(getElementTypeOrSelf(arg).isF32() && "arg must be f32 type");
  ArrayRef<int64_t> shape = vectorShape(arg);

  auto bcast = [&](Value value) -> Value {
    return broadcast(builder, value, shape);
  };

  auto i32 = builder.getIntegerType(32);
  auto i32Vec = broadcast(i32, shape);
  auto f32Vec = broadcast(builder.getF32Type(), shape);

  Value cst126f = f32Cst(builder, 126.0f);
  Value cstHalf = f32Cst(builder, 0.5f);
  Value cstInvMantMask = f32FromBits(builder, ~0x7f800000u);

  // Bitcast to i32 for bitwise operations.
  Value i32Half = builder.create<arith::BitcastOp>(i32, cstHalf);
  Value i32InvMantMask = builder.create<arith::BitcastOp>(i32, cstInvMantMask);
  Value i32Arg = builder.create<arith::BitcastOp>(i32Vec, arg);

  // Compute normalized fraction.
  Value tmp0 = builder.create<arith::AndIOp>(i32Arg, bcast(i32InvMantMask));
  Value tmp1 = builder.create<arith::OrIOp>(tmp0, bcast(i32Half));
  Value normalizedFraction = builder.create<arith::BitcastOp>(f32Vec, tmp1);

  // Compute exponent.
  Value arg0 = isPositive ? arg : builder.create<math::AbsOp>(arg);
  Value biasedExponentBits = builder.create<arith::ShRUIOp>(
      builder.create<arith::BitcastOp>(i32Vec, arg0),
      bcast(i32Cst(builder, 23)));
  Value biasedExponent =
      builder.create<arith::SIToFPOp>(f32Vec, biasedExponentBits);
  Value exponent =
      builder.create<arith::SubFOp>(biasedExponent, bcast(cst126f));

  return {normalizedFraction, exponent};
}

// Computes exp2 for an i32 argument.
static Value exp2I32(ImplicitLocOpBuilder &builder, Value arg) {
  assert(getElementTypeOrSelf(arg).isInteger(32) && "arg must be i32 type");
  ArrayRef<int64_t> shape = vectorShape(arg);

  auto bcast = [&](Value value) -> Value {
    return broadcast(builder, value, shape);
  };

  auto f32Vec = broadcast(builder.getF32Type(), shape);
  // The exponent of f32 located at 23-bit.
  auto exponetBitLocation = bcast(i32Cst(builder, 23));
  // Set the exponent bias to zero.
  auto bias = bcast(i32Cst(builder, 127));

  Value biasedArg = builder.create<arith::AddIOp>(arg, bias);
  Value exp2ValueInt =
      builder.create<arith::ShLIOp>(biasedArg, exponetBitLocation);
  Value exp2ValueF32 = builder.create<arith::BitcastOp>(f32Vec, exp2ValueInt);

  return exp2ValueF32;
}

namespace {
Value makePolynomialCalculation(ImplicitLocOpBuilder &builder,
                                llvm::ArrayRef<Value> coeffs, Value x) {
  assert(getElementTypeOrSelf(x).isF32() && "x must be f32 type");
  ArrayRef<int64_t> shape = vectorShape(x);

  if (coeffs.empty())
    return broadcast(builder, f32Cst(builder, 0.0f), shape);

  if (coeffs.size() == 1)
    return coeffs[0];

  Value res = builder.create<math::FmaOp>(x, coeffs[coeffs.size() - 1],
                                          coeffs[coeffs.size() - 2]);
  for (auto i = ptrdiff_t(coeffs.size()) - 3; i >= 0; --i) {
    res = builder.create<math::FmaOp>(x, res, coeffs[i]);
  }
  return res;
}
} // namespace

//----------------------------------------------------------------------------//
// Helper function/pattern to insert casts for reusing F32 bit expansion.
//----------------------------------------------------------------------------//

template <typename T>
LogicalResult insertCasts(Operation *op, PatternRewriter &rewriter) {
  // Conservatively only allow where the operand and result types are exactly 1.
  Type origType = op->getResultTypes().front();
  for (Type t : llvm::drop_begin(op->getResultTypes()))
    if (origType != t)
      return rewriter.notifyMatchFailure(op, "required all types to match");
  for (Type t : op->getOperandTypes())
    if (origType != t)
      return rewriter.notifyMatchFailure(op, "required all types to match");

  // Skip if already F32  or larger than 32 bits.
  if (getElementTypeOrSelf(origType).isF32() ||
      getElementTypeOrSelf(origType).getIntOrFloatBitWidth() > 32)
    return failure();

  // Create F32 equivalent type.
  Type newType;
  if (auto shaped = origType.dyn_cast<ShapedType>()) {
    newType = shaped.clone(rewriter.getF32Type());
  } else if (origType.isa<FloatType>()) {
    newType = rewriter.getF32Type();
  } else {
    return rewriter.notifyMatchFailure(op,
                                       "unable to find F32 equivalent type");
  }

  Location loc = op->getLoc();
  SmallVector<Value> operands;
  for (auto operand : op->getOperands())
    operands.push_back(rewriter.create<arith::ExtFOp>(loc, newType, operand));
  auto result = rewriter.create<math::Atan2Op>(loc, newType, operands);
  rewriter.replaceOpWithNewOp<arith::TruncFOp>(op, origType, result);
  return success();
}

namespace {
// Pattern to cast to F32 to reuse F32 expansion as fallback for single-result
// op.
// TODO: Consider revising to avoid adding multiple casts for a subgraph that is
// all in lower precision. Currently this is only fallback support and performs
// simplistic casting.
template <typename T>
struct ReuseF32Expansion : public OpRewritePattern<T> {
public:
  using OpRewritePattern<T>::OpRewritePattern;
  LogicalResult matchAndRewrite(T op, PatternRewriter &rewriter) const final {
    static_assert(
        T::template hasTrait<mlir::OpTrait::SameOperandsAndResultType>(),
        "requires same operands and result types");
    return insertCasts<T>(op, rewriter);
  }
};
} // namespace

//----------------------------------------------------------------------------//
// AtanOp approximation.
//----------------------------------------------------------------------------//

namespace {
struct AtanApproximation : public OpRewritePattern<math::AtanOp> {
public:
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(math::AtanOp op,
                                PatternRewriter &rewriter) const final;
};
} // namespace

LogicalResult
AtanApproximation::matchAndRewrite(math::AtanOp op,
                                   PatternRewriter &rewriter) const {
  auto operand = op.getOperand();
  if (!getElementTypeOrSelf(operand).isF32())
    return rewriter.notifyMatchFailure(op, "unsupported operand type");

  ArrayRef<int64_t> shape = vectorShape(op.getOperand());

  ImplicitLocOpBuilder builder(op->getLoc(), rewriter);
  auto one = broadcast(builder, f32Cst(builder, 1.0f), shape);

  // Remap the problem over [0.0, 1.0] by looking at the absolute value and the
  // handling symmetry.
  Value abs = builder.create<math::AbsOp>(operand);
  Value reciprocal = builder.create<arith::DivFOp>(one, abs);
  Value compare =
      builder.create<arith::CmpFOp>(arith::CmpFPredicate::OLT, abs, reciprocal);
  Value x = builder.create<arith::SelectOp>(compare, abs, reciprocal);

  // Perform the Taylor series approximation for atan over the range
  // [-1.0, 1.0].
  auto n1 = broadcast(builder, f32Cst(builder, 0.14418283f), shape);
  auto n2 = broadcast(builder, f32Cst(builder, -0.34999234f), shape);
  auto n3 = broadcast(builder, f32Cst(builder, -0.01067831f), shape);
  auto n4 = broadcast(builder, f32Cst(builder, 1.00209986f), shape);

  Value p = builder.create<math::FmaOp>(x, n1, n2);
  p = builder.create<math::FmaOp>(x, p, n3);
  p = builder.create<math::FmaOp>(x, p, n4);
  p = builder.create<arith::MulFOp>(x, p);

  // Remap the solution for over [0.0, 1.0] to [0.0, inf]
  auto halfPi = broadcast(builder, f32Cst(builder, 1.57079632679f), shape);
  Value sub = builder.create<arith::SubFOp>(halfPi, p);
  Value select = builder.create<arith::SelectOp>(compare, p, sub);

  // Correct for signing of the input.
  rewriter.replaceOpWithNewOp<math::CopySignOp>(op, select, operand);
  return success();
}

//----------------------------------------------------------------------------//
// AtanOp approximation.
//----------------------------------------------------------------------------//

namespace {
struct Atan2Approximation : public OpRewritePattern<math::Atan2Op> {
public:
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(math::Atan2Op op,
                                PatternRewriter &rewriter) const final;
};
} // namespace

LogicalResult
Atan2Approximation::matchAndRewrite(math::Atan2Op op,
                                    PatternRewriter &rewriter) const {
  auto y = op.getOperand(0);
  auto x = op.getOperand(1);
  if (!getElementTypeOrSelf(x).isF32())
    return rewriter.notifyMatchFailure(op, "unsupported operand type");

  ImplicitLocOpBuilder builder(op->getLoc(), rewriter);
  ArrayRef<int64_t> shape = vectorShape(op.getResult());

  // Compute atan in the valid range.
  auto div = builder.create<arith::DivFOp>(y, x);
  auto atan = builder.create<math::AtanOp>(div);

  // Determine what the atan would be for a 180 degree rotation.
  auto zero = broadcast(builder, f32Cst(builder, 0.0f), shape);
  auto pi = broadcast(builder, f32Cst(builder, 3.14159265359f), shape);
  auto addPi = builder.create<arith::AddFOp>(atan, pi);
  auto subPi = builder.create<arith::SubFOp>(atan, pi);
  auto atanGt =
      builder.create<arith::CmpFOp>(arith::CmpFPredicate::OGT, atan, zero);
  auto flippedAtan = builder.create<arith::SelectOp>(atanGt, subPi, addPi);

  // Determine whether to directly use atan or use the 180 degree flip
  auto xGt = builder.create<arith::CmpFOp>(arith::CmpFPredicate::OGT, x, zero);
  Value result = builder.create<arith::SelectOp>(xGt, atan, flippedAtan);

  // Handle x = 0, y > 0
  Value xZero =
      builder.create<arith::CmpFOp>(arith::CmpFPredicate::OEQ, x, zero);
  Value yGt = builder.create<arith::CmpFOp>(arith::CmpFPredicate::OGT, y, zero);
  Value isHalfPi = builder.create<arith::AndIOp>(xZero, yGt);
  auto halfPi = broadcast(builder, f32Cst(builder, 1.57079632679f), shape);
  result = builder.create<arith::SelectOp>(isHalfPi, halfPi, result);

  // Handle x = 0, y < 0
  Value yLt = builder.create<arith::CmpFOp>(arith::CmpFPredicate::OLT, y, zero);
  Value isNegativeHalfPiPi = builder.create<arith::AndIOp>(xZero, yLt);
  auto negativeHalfPiPi =
      broadcast(builder, f32Cst(builder, -1.57079632679f), shape);
  result = builder.create<arith::SelectOp>(isNegativeHalfPiPi, negativeHalfPiPi,
                                           result);

  // Handle x = 0, y = 0;
  Value yZero =
      builder.create<arith::CmpFOp>(arith::CmpFPredicate::OEQ, y, zero);
  Value isNan = builder.create<arith::AndIOp>(xZero, yZero);
  Value cstNan = broadcast(builder, f32FromBits(builder, 0x7fc00000), shape);
  result = builder.create<arith::SelectOp>(isNan, cstNan, result);

  rewriter.replaceOp(op, result);
  return success();
}

//----------------------------------------------------------------------------//
// TanhOp approximation.
//----------------------------------------------------------------------------//

namespace {
struct TanhApproximation : public OpRewritePattern<math::TanhOp> {
public:
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(math::TanhOp op,
                                PatternRewriter &rewriter) const final;
};
} // namespace

LogicalResult
TanhApproximation::matchAndRewrite(math::TanhOp op,
                                   PatternRewriter &rewriter) const {
  if (!getElementTypeOrSelf(op.getOperand()).isF32())
    return rewriter.notifyMatchFailure(op, "unsupported operand type");

  ArrayRef<int64_t> shape = vectorShape(op.getOperand());

  ImplicitLocOpBuilder builder(op->getLoc(), rewriter);
  auto bcast = [&](Value value) -> Value {
    return broadcast(builder, value, shape);
  };

  // Clamp operand into [plusClamp, minusClamp] range.
  Value minusClamp = bcast(f32Cst(builder, -7.99881172180175781f));
  Value plusClamp = bcast(f32Cst(builder, 7.99881172180175781f));
  Value x = clamp(builder, op.getOperand(), minusClamp, plusClamp);

  // Mask for tiny values that are approximated with `operand`.
  Value tiny = bcast(f32Cst(builder, 0.0004f));
  Value tinyMask = builder.create<arith::CmpFOp>(
      arith::CmpFPredicate::OLT, builder.create<math::AbsOp>(op.getOperand()),
      tiny);

  // The monomial coefficients of the numerator polynomial (odd).
  Value alpha1 = bcast(f32Cst(builder, 4.89352455891786e-03f));
  Value alpha3 = bcast(f32Cst(builder, 6.37261928875436e-04f));
  Value alpha5 = bcast(f32Cst(builder, 1.48572235717979e-05f));
  Value alpha7 = bcast(f32Cst(builder, 5.12229709037114e-08f));
  Value alpha9 = bcast(f32Cst(builder, -8.60467152213735e-11f));
  Value alpha11 = bcast(f32Cst(builder, 2.00018790482477e-13f));
  Value alpha13 = bcast(f32Cst(builder, -2.76076847742355e-16f));

  // The monomial coefficients of the denominator polynomial (even).
  Value beta0 = bcast(f32Cst(builder, 4.89352518554385e-03f));
  Value beta2 = bcast(f32Cst(builder, 2.26843463243900e-03f));
  Value beta4 = bcast(f32Cst(builder, 1.18534705686654e-04f));
  Value beta6 = bcast(f32Cst(builder, 1.19825839466702e-06f));

  // Since the polynomials are odd/even, we need x^2.
  Value x2 = builder.create<arith::MulFOp>(x, x);

  // Evaluate the numerator polynomial p.
  Value p = builder.create<math::FmaOp>(x2, alpha13, alpha11);
  p = builder.create<math::FmaOp>(x2, p, alpha9);
  p = builder.create<math::FmaOp>(x2, p, alpha7);
  p = builder.create<math::FmaOp>(x2, p, alpha5);
  p = builder.create<math::FmaOp>(x2, p, alpha3);
  p = builder.create<math::FmaOp>(x2, p, alpha1);
  p = builder.create<arith::MulFOp>(x, p);

  // Evaluate the denominator polynomial q.
  Value q = builder.create<math::FmaOp>(x2, beta6, beta4);
  q = builder.create<math::FmaOp>(x2, q, beta2);
  q = builder.create<math::FmaOp>(x2, q, beta0);

  // Divide the numerator by the denominator.
  Value res = builder.create<arith::SelectOp>(
      tinyMask, x, builder.create<arith::DivFOp>(p, q));

  rewriter.replaceOp(op, res);

  return success();
}

#define LN2_VALUE                                                              \
  0.693147180559945309417232121458176568075500134360255254120680009493393621L
#define LOG2E_VALUE                                                            \
  1.442695040888963407359924681001892137426645954152985934135449406931109219L

//----------------------------------------------------------------------------//
// LogOp and Log2Op approximation.
//----------------------------------------------------------------------------//

namespace {
template <typename Op>
struct LogApproximationBase : public OpRewritePattern<Op> {
  using OpRewritePattern<Op>::OpRewritePattern;

  /// Base 2 if 'base2' is set; natural logarithm (base e) otherwise.
  LogicalResult logMatchAndRewrite(Op op, PatternRewriter &rewriter,
                                   bool base2) const;
};
} // namespace

// This approximation comes from Julien Pommier's SSE math library.
// Link: http://gruntthepeon.free.fr/ssemath
template <typename Op>
LogicalResult
LogApproximationBase<Op>::logMatchAndRewrite(Op op, PatternRewriter &rewriter,
                                             bool base2) const {
  if (!getElementTypeOrSelf(op.getOperand()).isF32())
    return rewriter.notifyMatchFailure(op, "unsupported operand type");

  ArrayRef<int64_t> shape = vectorShape(op.getOperand());

  ImplicitLocOpBuilder builder(op->getLoc(), rewriter);
  auto bcast = [&](Value value) -> Value {
    return broadcast(builder, value, shape);
  };

  Value cstZero = bcast(f32Cst(builder, 0.0f));
  Value cstOne = bcast(f32Cst(builder, 1.0f));
  Value cstNegHalf = bcast(f32Cst(builder, -0.5f));

  // The smallest non denormalized float number.
  Value cstMinNormPos = bcast(f32FromBits(builder, 0x00800000u));
  Value cstMinusInf = bcast(f32FromBits(builder, 0xff800000u));
  Value cstPosInf = bcast(f32FromBits(builder, 0x7f800000u));
  Value cstNan = bcast(f32FromBits(builder, 0x7fc00000));

  // Polynomial coefficients.
  Value cstCephesSQRTHF = bcast(f32Cst(builder, 0.707106781186547524f));
  Value cstCephesLogP0 = bcast(f32Cst(builder, 7.0376836292E-2f));
  Value cstCephesLogP1 = bcast(f32Cst(builder, -1.1514610310E-1f));
  Value cstCephesLogP2 = bcast(f32Cst(builder, 1.1676998740E-1f));
  Value cstCephesLogP3 = bcast(f32Cst(builder, -1.2420140846E-1f));
  Value cstCephesLogP4 = bcast(f32Cst(builder, +1.4249322787E-1f));
  Value cstCephesLogP5 = bcast(f32Cst(builder, -1.6668057665E-1f));
  Value cstCephesLogP6 = bcast(f32Cst(builder, +2.0000714765E-1f));
  Value cstCephesLogP7 = bcast(f32Cst(builder, -2.4999993993E-1f));
  Value cstCephesLogP8 = bcast(f32Cst(builder, +3.3333331174E-1f));

  Value x = op.getOperand();

  // Truncate input values to the minimum positive normal.
  x = max(builder, x, cstMinNormPos);

  // Extract significant in the range [0.5,1) and exponent.
  std::pair<Value, Value> pair = frexp(builder, x, /*isPositive=*/true);
  x = pair.first;
  Value e = pair.second;

  // Shift the inputs from the range [0.5,1) to [sqrt(1/2), sqrt(2)) and shift
  // by -1.0. The values are then centered around 0, which improves the
  // stability of the polynomial evaluation:
  //
  //   if( x < SQRTHF ) {
  //     e -= 1;
  //     x = x + x - 1.0;
  //   } else { x = x - 1.0; }
  Value mask = builder.create<arith::CmpFOp>(arith::CmpFPredicate::OLT, x,
                                             cstCephesSQRTHF);
  Value tmp = builder.create<arith::SelectOp>(mask, x, cstZero);

  x = builder.create<arith::SubFOp>(x, cstOne);
  e = builder.create<arith::SubFOp>(
      e, builder.create<arith::SelectOp>(mask, cstOne, cstZero));
  x = builder.create<arith::AddFOp>(x, tmp);

  Value x2 = builder.create<arith::MulFOp>(x, x);
  Value x3 = builder.create<arith::MulFOp>(x2, x);

  // Evaluate the polynomial approximant of degree 8 in three parts.
  Value y0, y1, y2;
  y0 = builder.create<math::FmaOp>(cstCephesLogP0, x, cstCephesLogP1);
  y1 = builder.create<math::FmaOp>(cstCephesLogP3, x, cstCephesLogP4);
  y2 = builder.create<math::FmaOp>(cstCephesLogP6, x, cstCephesLogP7);
  y0 = builder.create<math::FmaOp>(y0, x, cstCephesLogP2);
  y1 = builder.create<math::FmaOp>(y1, x, cstCephesLogP5);
  y2 = builder.create<math::FmaOp>(y2, x, cstCephesLogP8);
  y0 = builder.create<math::FmaOp>(y0, x3, y1);
  y0 = builder.create<math::FmaOp>(y0, x3, y2);
  y0 = builder.create<arith::MulFOp>(y0, x3);

  y0 = builder.create<math::FmaOp>(cstNegHalf, x2, y0);
  x = builder.create<arith::AddFOp>(x, y0);

  if (base2) {
    Value cstLog2e = bcast(f32Cst(builder, static_cast<float>(LOG2E_VALUE)));
    x = builder.create<math::FmaOp>(x, cstLog2e, e);
  } else {
    Value cstLn2 = bcast(f32Cst(builder, static_cast<float>(LN2_VALUE)));
    x = builder.create<math::FmaOp>(e, cstLn2, x);
  }

  Value invalidMask = builder.create<arith::CmpFOp>(arith::CmpFPredicate::ULT,
                                                    op.getOperand(), cstZero);
  Value zeroMask = builder.create<arith::CmpFOp>(arith::CmpFPredicate::OEQ,
                                                 op.getOperand(), cstZero);
  Value posInfMask = builder.create<arith::CmpFOp>(arith::CmpFPredicate::OEQ,
                                                   op.getOperand(), cstPosInf);

  // Filter out invalid values:
  //  • x == 0     -> -INF
  //  • x < 0      ->  NAN
  //  • x == +INF  -> +INF
  Value aproximation = builder.create<arith::SelectOp>(
      zeroMask, cstMinusInf,
      builder.create<arith::SelectOp>(
          invalidMask, cstNan,
          builder.create<arith::SelectOp>(posInfMask, cstPosInf, x)));

  rewriter.replaceOp(op, aproximation);

  return success();
}

namespace {
struct LogApproximation : public LogApproximationBase<math::LogOp> {
  using LogApproximationBase::LogApproximationBase;

  LogicalResult matchAndRewrite(math::LogOp op,
                                PatternRewriter &rewriter) const final {
    return logMatchAndRewrite(op, rewriter, /*base2=*/false);
  }
};
} // namespace

namespace {
struct Log2Approximation : public LogApproximationBase<math::Log2Op> {
  using LogApproximationBase::LogApproximationBase;

  LogicalResult matchAndRewrite(math::Log2Op op,
                                PatternRewriter &rewriter) const final {
    return logMatchAndRewrite(op, rewriter, /*base2=*/true);
  }
};
} // namespace

//----------------------------------------------------------------------------//
// Log1p approximation.
//----------------------------------------------------------------------------//

namespace {
struct Log1pApproximation : public OpRewritePattern<math::Log1pOp> {
public:
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(math::Log1pOp op,
                                PatternRewriter &rewriter) const final;
};
} // namespace

// Approximate log(1+x).
LogicalResult
Log1pApproximation::matchAndRewrite(math::Log1pOp op,
                                    PatternRewriter &rewriter) const {
  if (!getElementTypeOrSelf(op.getOperand()).isF32())
    return rewriter.notifyMatchFailure(op, "unsupported operand type");

  ArrayRef<int64_t> shape = vectorShape(op.getOperand());

  ImplicitLocOpBuilder builder(op->getLoc(), rewriter);
  auto bcast = [&](Value value) -> Value {
    return broadcast(builder, value, shape);
  };

  // Approximate log(1+x) using the following, due to W. Kahan:
  //   u = x + 1.0;
  //   if (u == 1.0 || u == inf) return x;
  //   return x * log(u) / (u - 1.0);
  //          ^^^^^^^^^^^^^^^^^^^^^^
  //             "logLarge" below.
  Value cstOne = bcast(f32Cst(builder, 1.0f));
  Value x = op.getOperand();
  Value u = builder.create<arith::AddFOp>(x, cstOne);
  Value uSmall =
      builder.create<arith::CmpFOp>(arith::CmpFPredicate::OEQ, u, cstOne);
  Value logU = builder.create<math::LogOp>(u);
  Value uInf =
      builder.create<arith::CmpFOp>(arith::CmpFPredicate::OEQ, u, logU);
  Value logLarge = builder.create<arith::MulFOp>(
      x, builder.create<arith::DivFOp>(
             logU, builder.create<arith::SubFOp>(u, cstOne)));
  Value approximation = builder.create<arith::SelectOp>(
      builder.create<arith::OrIOp>(uSmall, uInf), x, logLarge);
  rewriter.replaceOp(op, approximation);
  return success();
}

//----------------------------------------------------------------------------//
// Erf approximation.
//----------------------------------------------------------------------------//

// Approximates erf(x) with
// a - P(x)/Q(x)
// where P and Q are polynomials of degree 4.
// Different coefficients are chosen based on the value of x.
// The approximation error is ~2.5e-07.
// Boost's minimax tool that utilizes the Remez method was used to find the
// coefficients.
LogicalResult
ErfPolynomialApproximation::matchAndRewrite(math::ErfOp op,
                                            PatternRewriter &rewriter) const {
  if (!getElementTypeOrSelf(op.getOperand()).isF32())
    return rewriter.notifyMatchFailure(op, "unsupported operand type");

  ArrayRef<int64_t> shape = vectorShape(op.getOperand());

  ImplicitLocOpBuilder builder(op->getLoc(), rewriter);
  auto bcast = [&](Value value) -> Value {
    return broadcast(builder, value, shape);
  };

  const int intervalsCount = 3;
  const int polyDegree = 4;

  Value zero = bcast(f32Cst(builder, 0));
  Value one = bcast(f32Cst(builder, 1));
  Value pp[intervalsCount][polyDegree + 1];
  pp[0][0] = bcast(f32Cst(builder, +0.00000000000000000e+00f));
  pp[0][1] = bcast(f32Cst(builder, +1.12837916222975858e+00f));
  pp[0][2] = bcast(f32Cst(builder, -5.23018562988006470e-01f));
  pp[0][3] = bcast(f32Cst(builder, +2.09741709609267072e-01f));
  pp[0][4] = bcast(f32Cst(builder, +2.58146801602987875e-02f));
  pp[1][0] = bcast(f32Cst(builder, +0.00000000000000000e+00f));
  pp[1][1] = bcast(f32Cst(builder, +1.12750687816789140e+00f));
  pp[1][2] = bcast(f32Cst(builder, -3.64721408487825775e-01f));
  pp[1][3] = bcast(f32Cst(builder, +1.18407396425136952e-01f));
  pp[1][4] = bcast(f32Cst(builder, +3.70645533056476558e-02f));
  pp[2][0] = bcast(f32Cst(builder, -3.30093071049483172e-03f));
  pp[2][1] = bcast(f32Cst(builder, +3.51961938357697011e-03f));
  pp[2][2] = bcast(f32Cst(builder, -1.41373622814988039e-03f));
  pp[2][3] = bcast(f32Cst(builder, +2.53447094961941348e-04f));
  pp[2][4] = bcast(f32Cst(builder, -1.71048029455037401e-05f));

  Value qq[intervalsCount][polyDegree + 1];
  qq[0][0] = bcast(f32Cst(builder, +1.000000000000000000e+00f));
  qq[0][1] = bcast(f32Cst(builder, -4.635138185962547255e-01f));
  qq[0][2] = bcast(f32Cst(builder, +5.192301327279782447e-01f));
  qq[0][3] = bcast(f32Cst(builder, -1.318089722204810087e-01f));
  qq[0][4] = bcast(f32Cst(builder, +7.397964654672315005e-02f));
  qq[1][0] = bcast(f32Cst(builder, +1.00000000000000000e+00f));
  qq[1][1] = bcast(f32Cst(builder, -3.27607011824493086e-01f));
  qq[1][2] = bcast(f32Cst(builder, +4.48369090658821977e-01f));
  qq[1][3] = bcast(f32Cst(builder, -8.83462621207857930e-02f));
  qq[1][4] = bcast(f32Cst(builder, +5.72442770283176093e-02f));
  qq[2][0] = bcast(f32Cst(builder, +1.00000000000000000e+00f));
  qq[2][1] = bcast(f32Cst(builder, -2.06069165953913769e+00f));
  qq[2][2] = bcast(f32Cst(builder, +1.62705939945477759e+00f));
  qq[2][3] = bcast(f32Cst(builder, -5.83389859211130017e-01f));
  qq[2][4] = bcast(f32Cst(builder, +8.21908939856640930e-02f));

  Value offsets[intervalsCount];
  offsets[0] = bcast(f32Cst(builder, 0.0f));
  offsets[1] = bcast(f32Cst(builder, 0.0f));
  offsets[2] = bcast(f32Cst(builder, 1.0f));

  Value bounds[intervalsCount];
  bounds[0] = bcast(f32Cst(builder, 0.8f));
  bounds[1] = bcast(f32Cst(builder, 2.0f));
  bounds[2] = bcast(f32Cst(builder, 3.75f));

  Value isNegativeArg = builder.create<arith::CmpFOp>(arith::CmpFPredicate::OLT,
                                                      op.getOperand(), zero);
  Value negArg = builder.create<arith::NegFOp>(op.getOperand());
  Value x =
      builder.create<arith::SelectOp>(isNegativeArg, negArg, op.getOperand());

  Value offset = offsets[0];
  Value p[polyDegree + 1];
  Value q[polyDegree + 1];
  for (int i = 0; i <= polyDegree; ++i) {
    p[i] = pp[0][i];
    q[i] = qq[0][i];
  }

  // TODO: maybe use vector stacking to reduce the number of selects.
  Value isLessThanBound[intervalsCount];
  for (int j = 0; j < intervalsCount - 1; ++j) {
    isLessThanBound[j] =
        builder.create<arith::CmpFOp>(arith::CmpFPredicate::OLT, x, bounds[j]);
    for (int i = 0; i <= polyDegree; ++i) {
      p[i] = builder.create<arith::SelectOp>(isLessThanBound[j], p[i],
                                             pp[j + 1][i]);
      q[i] = builder.create<arith::SelectOp>(isLessThanBound[j], q[i],
                                             qq[j + 1][i]);
    }
    offset = builder.create<arith::SelectOp>(isLessThanBound[j], offset,
                                             offsets[j + 1]);
  }
  isLessThanBound[intervalsCount - 1] = builder.create<arith::CmpFOp>(
      arith::CmpFPredicate::ULT, x, bounds[intervalsCount - 1]);

  Value pPoly = makePolynomialCalculation(builder, p, x);
  Value qPoly = makePolynomialCalculation(builder, q, x);
  Value rationalPoly = builder.create<arith::DivFOp>(pPoly, qPoly);
  Value formula = builder.create<arith::AddFOp>(offset, rationalPoly);
  formula = builder.create<arith::SelectOp>(isLessThanBound[intervalsCount - 1],
                                            formula, one);

  // erf is odd function: erf(x) = -erf(-x).
  Value negFormula = builder.create<arith::NegFOp>(formula);
  Value res =
      builder.create<arith::SelectOp>(isNegativeArg, negFormula, formula);

  rewriter.replaceOp(op, res);

  return success();
}

//----------------------------------------------------------------------------//
// Exp approximation.
//----------------------------------------------------------------------------//

namespace {

struct ExpApproximation : public OpRewritePattern<math::ExpOp> {
public:
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(math::ExpOp op,
                                PatternRewriter &rewriter) const final;
};
} // namespace

// Approximate exp(x) using its reduced range exp(y) where y is in the range
// [0, ln(2)], let y = x - floor(x / ln(2)) * ln(2) = x - k * ln(2), exp(x)
// = exp(y) * 2^k. exp(y).
LogicalResult
ExpApproximation::matchAndRewrite(math::ExpOp op,
                                  PatternRewriter &rewriter) const {
  if (!getElementTypeOrSelf(op.getOperand()).isF32())
    return rewriter.notifyMatchFailure(op, "unsupported operand type");

  ArrayRef<int64_t> shape = vectorShape(op.getOperand());

  ImplicitLocOpBuilder builder(op->getLoc(), rewriter);

  // TODO: Consider a common pattern rewriter with all methods below to
  // write the approximations.
  auto bcast = [&](Value value) -> Value {
    return broadcast(builder, value, shape);
  };
  auto fmla = [&](Value a, Value b, Value c) {
    return builder.create<math::FmaOp>(a, b, c);
  };
  auto mul = [&](Value a, Value b) -> Value {
    return builder.create<arith::MulFOp>(a, b);
  };
  auto sub = [&](Value a, Value b) -> Value {
    return builder.create<arith::SubFOp>(a, b);
  };
  auto floor = [&](Value a) { return builder.create<math::FloorOp>(a); };

  Value cstLn2 = bcast(f32Cst(builder, static_cast<float>(LN2_VALUE)));
  Value cstLog2E = bcast(f32Cst(builder, static_cast<float>(LOG2E_VALUE)));

  // Polynomial coefficients.
  Value cstCephesExpP0 = bcast(f32Cst(builder, 1.0));
  Value cstCephesExpP1 = bcast(f32Cst(builder, 1.0));
  Value cstCephesExpP2 = bcast(f32Cst(builder, 0.49970514590562437052f));
  Value cstCephesExpP3 = bcast(f32Cst(builder, 0.16873890085469545053f));
  Value cstCephesExpP4 = bcast(f32Cst(builder, 0.03668965196652099192f));
  Value cstCephesExpP5 = bcast(f32Cst(builder, 0.01314350012789660196f));

  Value x = op.getOperand();

  // Reduced y = x - floor(x / ln(2)) * ln(2) = x - k * ln(2)
  Value xL2Inv = mul(x, cstLog2E);
  Value kF32 = floor(xL2Inv);
  Value kLn2 = mul(kF32, cstLn2);
  Value y = sub(x, kLn2);

  // Use Estrin's evaluation scheme with 3 independent parts:
  // P(y)^y : (c0 + c1 y) + (c2 + c3 y) y^2 + (c4 + c5 y) y^4
  Value y2 = mul(y, y);
  Value y4 = mul(y2, y2);

  Value q0 = fmla(cstCephesExpP1, y, cstCephesExpP0);
  Value q1 = fmla(cstCephesExpP3, y, cstCephesExpP2);
  Value q2 = fmla(cstCephesExpP5, y, cstCephesExpP4);
  Value expY = fmla(q1, y2, q0);
  expY = fmla(q2, y4, expY);

  auto i32Vec = broadcast(builder.getI32Type(), shape);

  // exp2(k)
  Value k = builder.create<arith::FPToSIOp>(i32Vec, kF32);
  Value exp2KValue = exp2I32(builder, k);

  // exp(x) = exp(y) * exp2(k)
  expY = mul(expY, exp2KValue);

  // Handle overflow, inf and underflow of exp(x). exp(x) range is [0, inf], its
  // partitioned as the following:
  // exp(x) = 0, x <= -inf
  // exp(x) = underflow (min_float), x <= -88
  // exp(x) = inf (min_float), x >= 88
  // Note: |k| = 127 is the value where the 8-bits exponent saturates.
  Value zerof32Const = bcast(f32Cst(builder, 0));
  auto constPosInfinity =
      bcast(f32Cst(builder, std::numeric_limits<float>::infinity()));
  auto constNegIfinity =
      bcast(f32Cst(builder, -std::numeric_limits<float>::infinity()));
  auto underflow = bcast(f32Cst(builder, std::numeric_limits<float>::min()));

  Value kMaxConst = bcast(i32Cst(builder, 127));
  Value kMaxNegConst = bcast(i32Cst(builder, -127));
  Value rightBound =
      builder.create<arith::CmpIOp>(arith::CmpIPredicate::sle, k, kMaxConst);
  Value leftBound =
      builder.create<arith::CmpIOp>(arith::CmpIPredicate::sge, k, kMaxNegConst);

  Value isNegInfinityX = builder.create<arith::CmpFOp>(
      arith::CmpFPredicate::OEQ, x, constNegIfinity);
  Value isPosInfinityX = builder.create<arith::CmpFOp>(
      arith::CmpFPredicate::OEQ, x, constPosInfinity);
  Value isPostiveX =
      builder.create<arith::CmpFOp>(arith::CmpFPredicate::OGT, x, zerof32Const);
  Value isComputable = builder.create<arith::AndIOp>(rightBound, leftBound);

  expY = builder.create<arith::SelectOp>(
      isNegInfinityX, zerof32Const,
      builder.create<arith::SelectOp>(
          isPosInfinityX, constPosInfinity,
          builder.create<arith::SelectOp>(
              isComputable, expY,
              builder.create<arith::SelectOp>(isPostiveX, constPosInfinity,
                                              underflow))));

  rewriter.replaceOp(op, expY);

  return success();
}

//----------------------------------------------------------------------------//
// ExpM1 approximation.
//----------------------------------------------------------------------------//

namespace {

struct ExpM1Approximation : public OpRewritePattern<math::ExpM1Op> {
public:
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(math::ExpM1Op op,
                                PatternRewriter &rewriter) const final;
};
} // namespace

LogicalResult
ExpM1Approximation::matchAndRewrite(math::ExpM1Op op,
                                    PatternRewriter &rewriter) const {
  if (!getElementTypeOrSelf(op.getOperand()).isF32())
    return rewriter.notifyMatchFailure(op, "unsupported operand type");

  ArrayRef<int64_t> shape = vectorShape(op.getOperand());

  ImplicitLocOpBuilder builder(op->getLoc(), rewriter);
  auto bcast = [&](Value value) -> Value {
    return broadcast(builder, value, shape);
  };

  // expm1(x) = exp(x) - 1 = u - 1.
  // We have to handle it carefully when x is near 0, i.e. u ~= 1,
  // and when the input is ~= -inf, i.e. u - 1 ~= -1.
  Value cstOne = bcast(f32Cst(builder, 1.0f));
  Value cstNegOne = bcast(f32Cst(builder, -1.0f));
  Value x = op.getOperand();
  Value u = builder.create<math::ExpOp>(x);
  Value uEqOne =
      builder.create<arith::CmpFOp>(arith::CmpFPredicate::OEQ, u, cstOne);
  Value uMinusOne = builder.create<arith::SubFOp>(u, cstOne);
  Value uMinusOneEqNegOne = builder.create<arith::CmpFOp>(
      arith::CmpFPredicate::OEQ, uMinusOne, cstNegOne);
  // logU = log(u) ~= x
  Value logU = builder.create<math::LogOp>(u);

  // Detect exp(x) = +inf; written this way to avoid having to form +inf.
  Value isInf =
      builder.create<arith::CmpFOp>(arith::CmpFPredicate::OEQ, logU, u);

  // (u - 1) * (x / ~x)
  Value expm1 = builder.create<arith::MulFOp>(
      uMinusOne, builder.create<arith::DivFOp>(x, logU));
  expm1 = builder.create<arith::SelectOp>(isInf, u, expm1);
  Value approximation = builder.create<arith::SelectOp>(
      uEqOne, x,
      builder.create<arith::SelectOp>(uMinusOneEqNegOne, cstNegOne, expm1));
  rewriter.replaceOp(op, approximation);
  return success();
}

//----------------------------------------------------------------------------//
// Sin and Cos approximation.
//----------------------------------------------------------------------------//

namespace {

template <bool isSine, typename OpTy>
struct SinAndCosApproximation : public OpRewritePattern<OpTy> {
public:
  using OpRewritePattern<OpTy>::OpRewritePattern;

  LogicalResult matchAndRewrite(OpTy op, PatternRewriter &rewriter) const final;
};
} // namespace

#define TWO_OVER_PI                                                            \
  0.6366197723675813430755350534900574481378385829618257949906693762L
#define PI_OVER_2                                                              \
  1.5707963267948966192313216916397514420985846996875529104874722961L

// Approximates sin(x) or cos(x) by finding the best approximation polynomial in
// the reduced range [0, pi/2] for both sin(x) and cos(x). Then given y in the
// reduced range sin(x) will be computed as sin(y), -sin(y), cos(y) or -cos(y).
template <bool isSine, typename OpTy>
LogicalResult SinAndCosApproximation<isSine, OpTy>::matchAndRewrite(
    OpTy op, PatternRewriter &rewriter) const {
  static_assert(
      llvm::is_one_of<OpTy, math::SinOp, math::CosOp>::value,
      "SinAndCosApproximation pattern expects math::SinOp or math::CosOp");

  if (!getElementTypeOrSelf(op.getOperand()).isF32())
    return rewriter.notifyMatchFailure(op, "unsupported operand type");

  ArrayRef<int64_t> shape = vectorShape(op.getOperand());

  ImplicitLocOpBuilder builder(op->getLoc(), rewriter);
  auto bcast = [&](Value value) -> Value {
    return broadcast(builder, value, shape);
  };
  auto mul = [&](Value a, Value b) -> Value {
    return builder.create<arith::MulFOp>(a, b);
  };
  auto sub = [&](Value a, Value b) -> Value {
    return builder.create<arith::SubFOp>(a, b);
  };
  auto floor = [&](Value a) { return builder.create<math::FloorOp>(a); };

  auto i32Vec = broadcast(builder.getI32Type(), shape);
  auto fPToSingedInteger = [&](Value a) -> Value {
    return builder.create<arith::FPToSIOp>(i32Vec, a);
  };

  auto modulo4 = [&](Value a) -> Value {
    return builder.create<arith::AndIOp>(a, bcast(i32Cst(builder, 3)));
  };

  auto isEqualTo = [&](Value a, Value b) -> Value {
    return builder.create<arith::CmpIOp>(arith::CmpIPredicate::eq, a, b);
  };

  auto isGreaterThan = [&](Value a, Value b) -> Value {
    return builder.create<arith::CmpIOp>(arith::CmpIPredicate::sgt, a, b);
  };

  auto select = [&](Value cond, Value t, Value f) -> Value {
    return builder.create<arith::SelectOp>(cond, t, f);
  };

  auto fmla = [&](Value a, Value b, Value c) {
    return builder.create<math::FmaOp>(a, b, c);
  };

  auto bitwiseOr = [&](Value a, Value b) {
    return builder.create<arith::OrIOp>(a, b);
  };

  Value twoOverPi = bcast(f32Cst(builder, (float)TWO_OVER_PI));
  Value piOverTwo = bcast(f32Cst(builder, (float)PI_OVER_2));

  Value x = op.getOperand();

  Value k = floor(mul(x, twoOverPi));

  Value y = sub(x, mul(k, piOverTwo));

  Value cstOne = bcast(f32Cst(builder, 1.0));
  Value cstNegativeOne = bcast(f32Cst(builder, -1.0));

  Value cstSC2 = bcast(f32Cst(builder, -0.16666667163372039794921875f));
  Value cstSC4 = bcast(f32Cst(builder, 8.333347737789154052734375e-3f));
  Value cstSC6 = bcast(f32Cst(builder, -1.9842604524455964565277099609375e-4f));
  Value cstSC8 =
      bcast(f32Cst(builder, 2.760012648650445044040679931640625e-6f));
  Value cstSC10 =
      bcast(f32Cst(builder, -2.50293279435709337121807038784027099609375e-8f));

  Value cstCC2 = bcast(f32Cst(builder, -0.5f));
  Value cstCC4 = bcast(f32Cst(builder, 4.166664183139801025390625e-2f));
  Value cstCC6 = bcast(f32Cst(builder, -1.388833043165504932403564453125e-3f));
  Value cstCC8 = bcast(f32Cst(builder, 2.47562347794882953166961669921875e-5f));
  Value cstCC10 =
      bcast(f32Cst(builder, -2.59630184018533327616751194000244140625e-7f));

  Value kMod4 = modulo4(fPToSingedInteger(k));

  Value kR0 = isEqualTo(kMod4, bcast(i32Cst(builder, 0)));
  Value kR1 = isEqualTo(kMod4, bcast(i32Cst(builder, 1)));
  Value kR2 = isEqualTo(kMod4, bcast(i32Cst(builder, 2)));
  Value kR3 = isEqualTo(kMod4, bcast(i32Cst(builder, 3)));

  Value sinuseCos = isSine ? bitwiseOr(kR1, kR3) : bitwiseOr(kR0, kR2);
  Value negativeRange = isSine ? isGreaterThan(kMod4, bcast(i32Cst(builder, 1)))
                               : bitwiseOr(kR1, kR2);

  Value y2 = mul(y, y);

  Value base = select(sinuseCos, cstOne, y);
  Value cstC2 = select(sinuseCos, cstCC2, cstSC2);
  Value cstC4 = select(sinuseCos, cstCC4, cstSC4);
  Value cstC6 = select(sinuseCos, cstCC6, cstSC6);
  Value cstC8 = select(sinuseCos, cstCC8, cstSC8);
  Value cstC10 = select(sinuseCos, cstCC10, cstSC10);

  Value v1 = fmla(y2, cstC10, cstC8);
  Value v2 = fmla(y2, v1, cstC6);
  Value v3 = fmla(y2, v2, cstC4);
  Value v4 = fmla(y2, v3, cstC2);
  Value v5 = fmla(y2, v4, cstOne);
  Value v6 = mul(base, v5);

  Value approximation = select(negativeRange, mul(cstNegativeOne, v6), v6);

  rewriter.replaceOp(op, approximation);

  return success();
}

//----------------------------------------------------------------------------//
// Rsqrt approximation.
//----------------------------------------------------------------------------//

namespace {
struct RsqrtApproximation : public OpRewritePattern<math::RsqrtOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(math::RsqrtOp op,
                                PatternRewriter &rewriter) const final;
};
} // namespace

LogicalResult
RsqrtApproximation::matchAndRewrite(math::RsqrtOp op,
                                    PatternRewriter &rewriter) const {
  if (!getElementTypeOrSelf(op.getOperand()).isF32())
    return rewriter.notifyMatchFailure(op, "unsupported operand type");

  ArrayRef<int64_t> shape = vectorShape(op.getOperand());

  // Only support already-vectorized rsqrt's.
  if (shape.empty() || shape.back() % 8 != 0)
    return rewriter.notifyMatchFailure(op, "unsupported operand type");

  ImplicitLocOpBuilder builder(op->getLoc(), rewriter);
  auto bcast = [&](Value value) -> Value {
    return broadcast(builder, value, shape);
  };

  Value cstPosInf = bcast(f32FromBits(builder, 0x7f800000u));
  Value cstOnePointFive = bcast(f32Cst(builder, 1.5f));
  Value cstNegHalf = bcast(f32Cst(builder, -0.5f));
  Value cstMinNormPos = bcast(f32FromBits(builder, 0x00800000u));

  Value negHalf = builder.create<arith::MulFOp>(op.getOperand(), cstNegHalf);

  // Select only the inverse sqrt of positive normals (denormals are
  // flushed to zero).
  Value ltMinMask = builder.create<arith::CmpFOp>(
      arith::CmpFPredicate::OLT, op.getOperand(), cstMinNormPos);
  Value infMask = builder.create<arith::CmpFOp>(arith::CmpFPredicate::OEQ,
                                                op.getOperand(), cstPosInf);
  Value notNormalFiniteMask = builder.create<arith::OrIOp>(ltMinMask, infMask);

  // Compute an approximate result.
  Value yApprox = handleMultidimensionalVectors(
      builder, op->getOperands(), 8, [&builder](ValueRange operands) -> Value {
        return builder.create<x86vector::RsqrtOp>(operands);
      });

  // Do a single step of Newton-Raphson iteration to improve the approximation.
  // This uses the formula y_{n+1} = y_n * (1.5 - y_n * (0.5 * x) * y_n).
  // It is essential to evaluate the inner term like this because forming
  // y_n^2 may over- or underflow.
  Value inner = builder.create<arith::MulFOp>(negHalf, yApprox);
  Value fma = builder.create<math::FmaOp>(yApprox, inner, cstOnePointFive);
  Value yNewton = builder.create<arith::MulFOp>(yApprox, fma);

  // Select the result of the Newton-Raphson step for positive normal arguments.
  // For other arguments, choose the output of the intrinsic. This will
  // return rsqrt(+inf) = 0, rsqrt(x) = NaN if x < 0, and rsqrt(x) = +inf if
  // x is zero or a positive denormalized float (equivalent to flushing positive
  // denormalized inputs to zero).
  Value res =
      builder.create<arith::SelectOp>(notNormalFiniteMask, yApprox, yNewton);
  rewriter.replaceOp(op, res);

  return success();
}

//----------------------------------------------------------------------------//

void mlir::populateMathPolynomialApproximationPatterns(
    RewritePatternSet &patterns,
    const MathPolynomialApproximationOptions &options) {
  patterns.add<AtanApproximation, Atan2Approximation, TanhApproximation,
               LogApproximation, Log2Approximation, Log1pApproximation,
               ErfPolynomialApproximation, ExpApproximation, ExpM1Approximation,
               ReuseF32Expansion<math::Atan2Op>,
               SinAndCosApproximation<true, math::SinOp>,
               SinAndCosApproximation<false, math::CosOp>>(
      patterns.getContext());
  if (options.enableAvx2)
    patterns.add<RsqrtApproximation>(patterns.getContext());
}

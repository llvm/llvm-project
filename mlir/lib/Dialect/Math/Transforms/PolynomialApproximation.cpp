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
#include <cmath>
#include <cstddef>

#include "mlir/Dialect/Arith/IR/Arith.h"
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
#include "llvm/Support/MathExtras.h"

using namespace mlir;
using namespace mlir::math;
using namespace mlir::vector;

// Helper to encapsulate a vector's shape (including scalable dims).
struct VectorShape {
  ArrayRef<int64_t> sizes;
  ArrayRef<bool> scalableFlags;

  bool empty() const { return sizes.empty(); }
};

// Returns vector shape if the type is a vector. Returns an empty shape if it is
// not a vector.
static VectorShape vectorShape(Type type) {
  auto vectorType = dyn_cast<VectorType>(type);
  return vectorType
             ? VectorShape{vectorType.getShape(), vectorType.getScalableDims()}
             : VectorShape{};
}

static VectorShape vectorShape(Value value) {
  return vectorShape(value.getType());
}

//----------------------------------------------------------------------------//
// Broadcast scalar types and values into vector types and values.
//----------------------------------------------------------------------------//

// Broadcasts scalar type into vector type (iff shape is non-scalar).
static Type broadcast(Type type, VectorShape shape) {
  assert(!isa<VectorType>(type) && "must be scalar type");
  return !shape.empty()
             ? VectorType::get(shape.sizes, type, shape.scalableFlags)
             : type;
}

// Broadcasts scalar value into vector (iff shape is non-scalar).
static Value broadcast(ImplicitLocOpBuilder &builder, Value value,
                       VectorShape shape) {
  assert(!isa<VectorType>(value.getType()) && "must be scalar value");
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

  VectorType inputType = cast<VectorType>(operands[0].getType());
  ArrayRef<int64_t> inputShape = inputType.getShape();

  // If input shape matches target vector width, we can just call the
  // user-provided compute function with the operands.
  if (inputShape == llvm::ArrayRef(vectorWidth))
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
      auto eltType = cast<VectorType>(operand.getType()).getElementType();
      auto expandedType = VectorType::get(expandedShape, eltType);
      expandedOperands[i] =
          builder.create<vector::ShapeCastOp>(expandedType, operand);
    }
  }

  // Iterate over all outer dimensions of the compute shape vector type.
  auto iterationDims = ArrayRef<int64_t>(expandedShape).drop_back();
  int64_t maxIndex = computeMaxLinearIndex(iterationDims);
  auto strides = computeStrides(iterationDims);

  // Compute results for each one dimensional vector.
  SmallVector<Value> results(maxIndex);

  for (int64_t i = 0; i < maxIndex; ++i) {
    auto offsets = delinearize(i, strides);

    SmallVector<Value> extracted(expandedOperands.size());
    for (const auto &tuple : llvm::enumerate(expandedOperands))
      extracted[tuple.index()] =
          builder.create<vector::ExtractOp>(tuple.value(), offsets);

    results[i] = compute(extracted);
  }

  // Stitch results together into one large vector.
  Type resultEltType = cast<VectorType>(results[0].getType()).getElementType();
  Type resultExpandedType = VectorType::get(expandedShape, resultEltType);
  Value result = builder.create<arith::ConstantOp>(
      resultExpandedType, builder.getZeroAttr(resultExpandedType));

  for (int64_t i = 0; i < maxIndex; ++i)
    result = builder.create<vector::InsertOp>(results[i], result,
                                              delinearize(i, strides));

  // Reshape back to the original vector shape.
  return builder.create<vector::ShapeCastOp>(
      VectorType::get(inputShape, resultEltType), result);
}

//----------------------------------------------------------------------------//
// Helper functions to create constants.
//----------------------------------------------------------------------------//

static Value floatCst(ImplicitLocOpBuilder &builder, float value,
                      Type elementType) {
  assert((elementType.isF16() || elementType.isF32()) &&
         "x must be f16 or f32 type.");
  return builder.create<arith::ConstantOp>(
      builder.getFloatAttr(elementType, value));
}

static Value f32Cst(ImplicitLocOpBuilder &builder, double value) {
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

// Return the minimum of the two values or NaN if value is NaN
static Value min(ImplicitLocOpBuilder &builder, Value value, Value bound) {
  return builder.create<arith::SelectOp>(
      builder.create<arith::CmpFOp>(arith::CmpFPredicate::ULT, value, bound),
      value, bound);
}

// Return the maximum of the two values or NaN if value is NaN
static Value max(ImplicitLocOpBuilder &builder, Value value, Value bound) {
  return builder.create<arith::SelectOp>(
      builder.create<arith::CmpFOp>(arith::CmpFPredicate::UGT, value, bound),
      value, bound);
}

// Return the clamped value or NaN if value is NaN
static Value clamp(ImplicitLocOpBuilder &builder, Value value, Value lowerBound,
                   Value upperBound) {
  return max(builder, min(builder, value, upperBound), lowerBound);
}

// Decomposes given floating point value `arg` into a normalized fraction and
// an integral power of two (see std::frexp). Returned values have float type.
static std::pair<Value, Value> frexp(ImplicitLocOpBuilder &builder, Value arg,
                                     bool isPositive = false) {
  assert(getElementTypeOrSelf(arg).isF32() && "arg must be f32 type");
  VectorShape shape = vectorShape(arg);

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
  Value arg0 = isPositive ? arg : builder.create<math::AbsFOp>(arg);
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
  VectorShape shape = vectorShape(arg);

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
  Type elementType = getElementTypeOrSelf(x);
  assert((elementType.isF32() || elementType.isF16()) &&
         "x must be f32 or f16 type");
  VectorShape shape = vectorShape(x);

  if (coeffs.empty())
    return broadcast(builder, floatCst(builder, 0.0f, elementType), shape);

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
  if (auto shaped = dyn_cast<ShapedType>(origType)) {
    newType = shaped.clone(rewriter.getF32Type());
  } else if (isa<FloatType>(origType)) {
    newType = rewriter.getF32Type();
  } else {
    return rewriter.notifyMatchFailure(op,
                                       "unable to find F32 equivalent type");
  }

  Location loc = op->getLoc();
  SmallVector<Value> operands;
  for (auto operand : op->getOperands())
    operands.push_back(rewriter.create<arith::ExtFOp>(loc, newType, operand));
  auto result =
      rewriter.create<T>(loc, TypeRange{newType}, operands, op->getAttrs());
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

  VectorShape shape = vectorShape(op.getOperand());

  ImplicitLocOpBuilder builder(op->getLoc(), rewriter);
  Value abs = builder.create<math::AbsFOp>(operand);

  auto one = broadcast(builder, f32Cst(builder, 1.0), shape);

  // When 0.66 < x <= 2.41 we do (x-1) / (x+1):
  auto twoThirds = broadcast(builder, f32Cst(builder, 0.66), shape);
  Value cmp2 =
      builder.create<arith::CmpFOp>(arith::CmpFPredicate::OGT, abs, twoThirds);
  Value addone = builder.create<arith::AddFOp>(abs, one);
  Value subone = builder.create<arith::SubFOp>(abs, one);
  Value xnum = builder.create<arith::SelectOp>(cmp2, subone, abs);
  Value xden = builder.create<arith::SelectOp>(cmp2, addone, one);

  auto bcast = [&](Value value) -> Value {
    return broadcast(builder, value, shape);
  };

  // Break into the <= 0.66 or > 2.41 we do x or 1/x:
  auto tan3pio8 = bcast(f32Cst(builder, 2.41421356237309504880));
  Value cmp1 =
      builder.create<arith::CmpFOp>(arith::CmpFPredicate::OGT, abs, tan3pio8);
  xnum = builder.create<arith::SelectOp>(cmp1, one, xnum);
  xden = builder.create<arith::SelectOp>(cmp1, abs, xden);

  Value x = builder.create<arith::DivFOp>(xnum, xden);
  Value xx = builder.create<arith::MulFOp>(x, x);

  // Perform the Taylor series approximation for atan over the range
  // [0.0, 0.66].
  auto p0 = bcast(f32Cst(builder, -8.750608600031904122785e-01));
  auto p1 = bcast(f32Cst(builder, -1.615753718733365076637e+01));
  auto p2 = bcast(f32Cst(builder, -7.500855792314704667340e+01));
  auto p3 = bcast(f32Cst(builder, -1.228866684490136173410e+02));
  auto p4 = bcast(f32Cst(builder, -6.485021904942025371773e+01));
  auto q0 = bcast(f32Cst(builder, +2.485846490142306297962e+01));
  auto q1 = bcast(f32Cst(builder, +1.650270098316988542046e+02));
  auto q2 = bcast(f32Cst(builder, +4.328810604912902668951e+02));
  auto q3 = bcast(f32Cst(builder, +4.853903996359136964868e+02));
  auto q4 = bcast(f32Cst(builder, +1.945506571482613964425e+02));

  // Apply the polynomial approximation for the numerator:
  Value n = p0;
  n = builder.create<math::FmaOp>(xx, n, p1);
  n = builder.create<math::FmaOp>(xx, n, p2);
  n = builder.create<math::FmaOp>(xx, n, p3);
  n = builder.create<math::FmaOp>(xx, n, p4);
  n = builder.create<arith::MulFOp>(n, xx);

  // Apply the polynomial approximation for the denominator:
  Value d = q0;
  d = builder.create<math::FmaOp>(xx, d, q1);
  d = builder.create<math::FmaOp>(xx, d, q2);
  d = builder.create<math::FmaOp>(xx, d, q3);
  d = builder.create<math::FmaOp>(xx, d, q4);

  // Compute approximation of theta:
  Value ans0 = builder.create<arith::DivFOp>(n, d);
  ans0 = builder.create<math::FmaOp>(ans0, x, x);

  // Correct for the input mapping's angles:
  Value mpi4 = bcast(f32Cst(builder, llvm::numbers::pi / 4));
  Value ans2 = builder.create<arith::AddFOp>(mpi4, ans0);
  Value ans = builder.create<arith::SelectOp>(cmp2, ans2, ans0);

  Value mpi2 = bcast(f32Cst(builder, llvm::numbers::pi / 2));
  Value ans1 = builder.create<arith::SubFOp>(mpi2, ans0);
  ans = builder.create<arith::SelectOp>(cmp1, ans1, ans);

  // Correct for signing of the input.
  rewriter.replaceOpWithNewOp<math::CopySignOp>(op, ans, operand);
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
  VectorShape shape = vectorShape(op.getResult());

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

  VectorShape shape = vectorShape(op.getOperand());

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
      arith::CmpFPredicate::OLT, builder.create<math::AbsFOp>(op.getOperand()),
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

  VectorShape shape = vectorShape(op.getOperand());

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

  VectorShape shape = vectorShape(op.getOperand());

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
  Value operand = op.getOperand();
  Type elementType = getElementTypeOrSelf(operand);

  if (!(elementType.isF32() || elementType.isF16()))
    return rewriter.notifyMatchFailure(op,
                                       "only f32 and f16 type is supported.");
  VectorShape shape = vectorShape(operand);

  ImplicitLocOpBuilder builder(op->getLoc(), rewriter);
  auto bcast = [&](Value value) -> Value {
    return broadcast(builder, value, shape);
  };

  const int intervalsCount = 3;
  const int polyDegree = 4;

  Value zero = bcast(floatCst(builder, 0, elementType));
  Value one = bcast(floatCst(builder, 1, elementType));
  Value pp[intervalsCount][polyDegree + 1];
  pp[0][0] = bcast(floatCst(builder, +0.00000000000000000e+00f, elementType));
  pp[0][1] = bcast(floatCst(builder, +1.12837916222975858e+00f, elementType));
  pp[0][2] = bcast(floatCst(builder, -5.23018562988006470e-01f, elementType));
  pp[0][3] = bcast(floatCst(builder, +2.09741709609267072e-01f, elementType));
  pp[0][4] = bcast(floatCst(builder, +2.58146801602987875e-02f, elementType));
  pp[1][0] = bcast(floatCst(builder, +0.00000000000000000e+00f, elementType));
  pp[1][1] = bcast(floatCst(builder, +1.12750687816789140e+00f, elementType));
  pp[1][2] = bcast(floatCst(builder, -3.64721408487825775e-01f, elementType));
  pp[1][3] = bcast(floatCst(builder, +1.18407396425136952e-01f, elementType));
  pp[1][4] = bcast(floatCst(builder, +3.70645533056476558e-02f, elementType));
  pp[2][0] = bcast(floatCst(builder, -3.30093071049483172e-03f, elementType));
  pp[2][1] = bcast(floatCst(builder, +3.51961938357697011e-03f, elementType));
  pp[2][2] = bcast(floatCst(builder, -1.41373622814988039e-03f, elementType));
  pp[2][3] = bcast(floatCst(builder, +2.53447094961941348e-04f, elementType));
  pp[2][4] = bcast(floatCst(builder, -1.71048029455037401e-05f, elementType));

  Value qq[intervalsCount][polyDegree + 1];
  qq[0][0] = bcast(floatCst(builder, +1.000000000000000000e+00f, elementType));
  qq[0][1] = bcast(floatCst(builder, -4.635138185962547255e-01f, elementType));
  qq[0][2] = bcast(floatCst(builder, +5.192301327279782447e-01f, elementType));
  qq[0][3] = bcast(floatCst(builder, -1.318089722204810087e-01f, elementType));
  qq[0][4] = bcast(floatCst(builder, +7.397964654672315005e-02f, elementType));
  qq[1][0] = bcast(floatCst(builder, +1.00000000000000000e+00f, elementType));
  qq[1][1] = bcast(floatCst(builder, -3.27607011824493086e-01f, elementType));
  qq[1][2] = bcast(floatCst(builder, +4.48369090658821977e-01f, elementType));
  qq[1][3] = bcast(floatCst(builder, -8.83462621207857930e-02f, elementType));
  qq[1][4] = bcast(floatCst(builder, +5.72442770283176093e-02f, elementType));
  qq[2][0] = bcast(floatCst(builder, +1.00000000000000000e+00f, elementType));
  qq[2][1] = bcast(floatCst(builder, -2.06069165953913769e+00f, elementType));
  qq[2][2] = bcast(floatCst(builder, +1.62705939945477759e+00f, elementType));
  qq[2][3] = bcast(floatCst(builder, -5.83389859211130017e-01f, elementType));
  qq[2][4] = bcast(floatCst(builder, +8.21908939856640930e-02f, elementType));

  Value offsets[intervalsCount];
  offsets[0] = bcast(floatCst(builder, 0.0f, elementType));
  offsets[1] = bcast(floatCst(builder, 0.0f, elementType));
  offsets[2] = bcast(floatCst(builder, 1.0f, elementType));

  Value bounds[intervalsCount];
  bounds[0] = bcast(floatCst(builder, 0.8f, elementType));
  bounds[1] = bcast(floatCst(builder, 2.0f, elementType));
  bounds[2] = bcast(floatCst(builder, 3.75f, elementType));

  Value isNegativeArg =
      builder.create<arith::CmpFOp>(arith::CmpFPredicate::OLT, operand, zero);
  Value negArg = builder.create<arith::NegFOp>(operand);
  Value x = builder.create<arith::SelectOp>(isNegativeArg, negArg, operand);

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

Value clampWithNormals(ImplicitLocOpBuilder &builder, const VectorShape shape,
                       Value value, float lowerBound, float upperBound) {
  assert(!std::isnan(lowerBound));
  assert(!std::isnan(upperBound));

  auto bcast = [&](Value value) -> Value {
    return broadcast(builder, value, shape);
  };

  auto selectCmp = [&builder](auto pred, Value value, Value bound) {
    return builder.create<arith::SelectOp>(
        builder.create<arith::CmpFOp>(pred, value, bound), value, bound);
  };

  // Note: prefer UGE/ULE vs. UGT/ULT, since they generate vmaxps/vminps vs.
  // vcmpleps+vmovaps on x86_64. The latter outcome is also obtained with
  // arith::{Max,Min}FOp.
  value = selectCmp(arith::CmpFPredicate::UGE, value,
                    bcast(f32Cst(builder, lowerBound)));
  value = selectCmp(arith::CmpFPredicate::ULE, value,
                    bcast(f32Cst(builder, upperBound)));
  return value;
}

struct ExpApproximation : public OpRewritePattern<math::ExpOp> {
public:
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(math::ExpOp op,
                                PatternRewriter &rewriter) const final;
};

LogicalResult
ExpApproximation::matchAndRewrite(math::ExpOp op,
                                  PatternRewriter &rewriter) const {
  auto shape = vectorShape(op.getOperand().getType());
  auto elementTy = getElementTypeOrSelf(op.getType());
  if (!elementTy.isF32())
    return rewriter.notifyMatchFailure(op, "unsupported operand type");

  ImplicitLocOpBuilder builder(op->getLoc(), rewriter);

  auto add = [&](Value a, Value b) -> Value {
    return builder.create<arith::AddFOp>(a, b);
  };
  auto bcast = [&](Value value) -> Value {
    return broadcast(builder, value, shape);
  };
  auto floor = [&](Value a) { return builder.create<math::FloorOp>(a); };
  auto fmla = [&](Value a, Value b, Value c) {
    return builder.create<math::FmaOp>(a, b, c);
  };
  auto mul = [&](Value a, Value b) -> Value {
    return builder.create<arith::MulFOp>(a, b);
  };

  // Polynomial approximation from Cephes.
  //
  // To compute e^x, we re-express it as
  //
  //   e^x = e^(a + b)
  //       = e^(a + n log(2))
  //       = e^a * 2^n.
  //
  // We choose n = round(x / log(2)), restricting the value of `a` to
  // (-log(2)/2, log(2)/2).  We then use a polynomial to compute e^a. The
  // relative error between our approximation and the true value of e^a is less
  // than 2^-22.5 for all values of `a` within this range.

  // Restrict input to a small range, including some values that evaluate to
  // +/- inf.  Note that for our lower bound, we choose log(2^-126) instead of
  // log(F32_EPSILON). We do so because this routine always flushes denormal
  // floating points to 0. Therefore, we only need to worry about exponentiating
  // up to the smallest representable non-denormal floating point, which is
  // 2^-126.

  // Constants.
  Value cstHalf = bcast(f32Cst(builder, 0.5f));
  Value cstOne = bcast(f32Cst(builder, 1.0f));

  // 1/log(2)
  Value cstLog2ef = bcast(f32Cst(builder, 1.44269504088896341f));

  Value cstExpC1 = bcast(f32Cst(builder, -0.693359375f));
  Value cstExpC2 = bcast(f32Cst(builder, 2.12194440e-4f));
  Value cstExpP0 = bcast(f32Cst(builder, 1.9875691500E-4f));
  Value cstExpP1 = bcast(f32Cst(builder, 1.3981999507E-3f));
  Value cstExpP2 = bcast(f32Cst(builder, 8.3334519073E-3f));
  Value cstExpP3 = bcast(f32Cst(builder, 4.1665795894E-2f));
  Value cstExpP4 = bcast(f32Cst(builder, 1.6666665459E-1f));
  Value cstExpP5 = bcast(f32Cst(builder, 5.0000001201E-1f));

  // Our computations below aren't particularly sensitive to the exact choices
  // here, so we choose values a bit larger/smaller than
  //
  //   log(F32_MAX) = 88.723...
  //   log(2^-126) = -87.337...
  Value x = op.getOperand();
  x = clampWithNormals(builder, shape, x, -87.8f, 88.8f);
  Value n = floor(fmla(x, cstLog2ef, cstHalf));

  // When we eventually do the multiplication in e^a * 2^n, we need to handle
  // the case when n > 127, the max fp32 exponent (so 2^n == inf) but e^a < 1
  // (so e^a * 2^n != inf).  There's a similar problem for n < -126, the
  // smallest fp32 exponent.
  //
  // A straightforward solution would be to detect n out of range and split it
  // up, doing
  //
  //   e^a * 2^n = e^a * 2^(n1 + n2)
  //             = (2^n1 * e^a) * 2^n2.
  //
  // But it turns out this approach is quite slow, probably because it
  // manipulates subnormal values.
  //
  // The approach we use instead is to clamp n to [-127, 127]. Let n' be the
  // value of n clamped to [-127, 127]. In the case where n' = 127, `a` can grow
  // up to as large as 88.8 - 127 * log(2) which is about 0.7703. Even though
  // this value of `a` is outside our previously specified range, e^a will still
  // only have a relative error of approximately 2^-16 at worse. In practice
  // this seems to work well enough; it passes our exhaustive tests, breaking
  // only one result, and by one ulp (we return exp(88.7228394) = max-float but
  // we should return inf).
  //
  // In the case where n' = -127, the original input value of x is so small that
  // e^x, our final answer, is less than 2^-126. Since 2^-126 is the smallest
  // normal floating point, and since we flush denormals, we simply return 0. We
  // do this in a branchless way by observing that our code for constructing 2^n
  // produces 0 if n = -127.
  //
  // The proof that n' = -127 implies e^x < 2^-126 is as follows:
  //
  //    n' = -127 implies n <= -127
  //              implies round(x / log(2)) <= -127
  //              implies x/log(2) < -126.5
  //              implies x < -126.5 * log(2)
  //              implies e^x < e^(-126.5 * log(2))
  //              implies e^x < 2^-126.5 < 2^-126
  //
  //    This proves that n' = -127 implies e^x < 2^-126.
  n = clampWithNormals(builder, shape, n, -127.0f, 127.0f);

  // Computes x = x - n' * log(2), the value for `a`
  x = fmla(cstExpC1, n, x);
  x = fmla(cstExpC2, n, x);

  // Polynomial to compute z = e^a, accurate for a in (-0.5, 0.5).
  Value z = fmla(x, cstExpP0, cstExpP1);
  z = fmla(z, x, cstExpP2);
  z = fmla(z, x, cstExpP3);
  z = fmla(z, x, cstExpP4);
  z = fmla(z, x, cstExpP5);
  z = fmla(z, mul(x, x), x);
  z = add(cstOne, z);

  // Convert n' to an i32.  This is safe because we clamped it above.
  auto i32Vec = broadcast(builder.getI32Type(), shape);
  Value nI32 = builder.create<arith::FPToSIOp>(i32Vec, n);

  // Creates the value 2^n' if -126 <= n' <= 127 and 0 if n' = -127.
  Value pow2 = exp2I32(builder, nI32);

  // Return z * 2^n' if -126 <= n' <= 127 and 0 if n = -127.
  Value ret = mul(z, pow2);

  rewriter.replaceOp(op, ret);
  return mlir::success();
}

} // namespace

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

  VectorShape shape = vectorShape(op.getOperand());

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
  Value uEqOneOrNaN =
      builder.create<arith::CmpFOp>(arith::CmpFPredicate::UEQ, u, cstOne);
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
      uEqOneOrNaN, x,
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

  VectorShape shape = vectorShape(op.getOperand());

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
// Cbrt approximation.
//----------------------------------------------------------------------------//

namespace {
struct CbrtApproximation : public OpRewritePattern<math::CbrtOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(math::CbrtOp op,
                                PatternRewriter &rewriter) const final;
};
} // namespace

// Estimation of cube-root using an algorithm defined in
// Hacker's Delight 2nd Edition.
LogicalResult
CbrtApproximation::matchAndRewrite(math::CbrtOp op,
                                   PatternRewriter &rewriter) const {
  auto operand = op.getOperand();
  if (!getElementTypeOrSelf(operand).isF32())
    return rewriter.notifyMatchFailure(op, "unsupported operand type");

  ImplicitLocOpBuilder b(op->getLoc(), rewriter);
  VectorShape shape = vectorShape(operand);

  Type floatTy = getElementTypeOrSelf(operand.getType());
  Type intTy = b.getIntegerType(floatTy.getIntOrFloatBitWidth());

  // Convert to vector types if necessary.
  floatTy = broadcast(floatTy, shape);
  intTy = broadcast(intTy, shape);

  auto bconst = [&](TypedAttr attr) -> Value {
    Value value = b.create<arith::ConstantOp>(attr);
    return broadcast(b, value, shape);
  };

  // Declare the initial values:
  Value intTwo = bconst(b.getI32IntegerAttr(2));
  Value intFour = bconst(b.getI32IntegerAttr(4));
  Value intEight = bconst(b.getI32IntegerAttr(8));
  Value intMagic = bconst(b.getI32IntegerAttr(0x2a5137a0));
  Value fpThird = bconst(b.getF32FloatAttr(0.33333333f));
  Value fpTwo = bconst(b.getF32FloatAttr(2.0f));
  Value fpZero = bconst(b.getF32FloatAttr(0.0f));

  // Compute an approximation of one third:
  // union {int ix; float x;};
  // x = x0;
  // ix = ix/4 + ix/16;
  Value absValue = b.create<math::AbsFOp>(operand);
  Value intValue = b.create<arith::BitcastOp>(intTy, absValue);
  Value divideBy4 = b.create<arith::ShRSIOp>(intValue, intTwo);
  Value divideBy16 = b.create<arith::ShRSIOp>(intValue, intFour);
  intValue = b.create<arith::AddIOp>(divideBy4, divideBy16);

  // ix = ix + ix/16;
  divideBy16 = b.create<arith::ShRSIOp>(intValue, intFour);
  intValue = b.create<arith::AddIOp>(intValue, divideBy16);

  // ix = ix + ix/256;
  Value divideBy256 = b.create<arith::ShRSIOp>(intValue, intEight);
  intValue = b.create<arith::AddIOp>(intValue, divideBy256);

  // ix = 0x2a5137a0 + ix;
  intValue = b.create<arith::AddIOp>(intValue, intMagic);

  // Perform one newtons step:
  // x = 0.33333333f*(2.0f*x + x0/(x*x));
  Value floatValue = b.create<arith::BitcastOp>(floatTy, intValue);
  Value squared = b.create<arith::MulFOp>(floatValue, floatValue);
  Value mulTwo = b.create<arith::MulFOp>(floatValue, fpTwo);
  Value divSquared = b.create<arith::DivFOp>(absValue, squared);
  floatValue = b.create<arith::AddFOp>(mulTwo, divSquared);
  floatValue = b.create<arith::MulFOp>(floatValue, fpThird);

  // x = 0.33333333f*(2.0f*x + x0/(x*x));
  squared = b.create<arith::MulFOp>(floatValue, floatValue);
  mulTwo = b.create<arith::MulFOp>(floatValue, fpTwo);
  divSquared = b.create<arith::DivFOp>(absValue, squared);
  floatValue = b.create<arith::AddFOp>(mulTwo, divSquared);
  floatValue = b.create<arith::MulFOp>(floatValue, fpThird);

  // Check for zero and restore sign.
  Value isZero =
      b.create<arith::CmpFOp>(arith::CmpFPredicate::OEQ, absValue, fpZero);
  floatValue = b.create<arith::SelectOp>(isZero, fpZero, floatValue);
  floatValue = b.create<math::CopySignOp>(floatValue, operand);

  rewriter.replaceOp(op, floatValue);
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

  VectorShape shape = vectorShape(op.getOperand());

  // Only support already-vectorized rsqrt's.
  if (shape.empty() || shape.sizes.back() % 8 != 0)
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

void mlir::populatePolynomialApproximateTanhPattern(
    RewritePatternSet &patterns) {
  patterns.add<TanhApproximation>(patterns.getContext());
}

void mlir::populatePolynomialApproximateErfPattern(
    RewritePatternSet &patterns) {
  patterns.add<ErfPolynomialApproximation>(patterns.getContext());
}

void mlir::populateMathPolynomialApproximationPatterns(
    RewritePatternSet &patterns,
    const MathPolynomialApproximationOptions &options) {
  // Patterns for leveraging existing f32 lowerings on other data types.
  patterns
      .add<ReuseF32Expansion<math::AtanOp>, ReuseF32Expansion<math::Atan2Op>,
           ReuseF32Expansion<math::TanhOp>, ReuseF32Expansion<math::LogOp>,
           ReuseF32Expansion<math::Log2Op>, ReuseF32Expansion<math::Log1pOp>,
           ReuseF32Expansion<math::ErfOp>, ReuseF32Expansion<math::ExpOp>,
           ReuseF32Expansion<math::ExpM1Op>, ReuseF32Expansion<math::CbrtOp>,
           ReuseF32Expansion<math::SinOp>, ReuseF32Expansion<math::CosOp>>(
          patterns.getContext());

  patterns.add<AtanApproximation, Atan2Approximation, TanhApproximation,
               LogApproximation, Log2Approximation, Log1pApproximation,
               ErfPolynomialApproximation, ExpApproximation, ExpM1Approximation,
               CbrtApproximation, SinAndCosApproximation<true, math::SinOp>,
               SinAndCosApproximation<false, math::CosOp>>(
      patterns.getContext());
  if (options.enableAvx2) {
    patterns.add<RsqrtApproximation, ReuseF32Expansion<math::RsqrtOp>>(
        patterns.getContext());
  }
}

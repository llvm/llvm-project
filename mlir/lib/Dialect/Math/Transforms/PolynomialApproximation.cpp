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

#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMTypes.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/Math/Transforms/Passes.h"
#include "mlir/Dialect/Vector/VectorOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/ImplicitLocOpBuilder.h"
#include "mlir/Transforms/Bufferize.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include <climits>

using namespace mlir;
using namespace mlir::vector;

using TypePredicate = llvm::function_ref<bool(Type)>;

// Returns vector width if the element type is matching the predicate (scalars
// that do match the predicate have width equal to `1`).
static Optional<int> vectorWidth(Type type, TypePredicate pred) {
  // If the type matches the predicate then its width is `1`.
  if (pred(type))
    return 1;

  // Otherwise check if the type is a vector type.
  auto vectorType = type.dyn_cast<VectorType>();
  if (vectorType && pred(vectorType.getElementType())) {
    assert(vectorType.getRank() == 1 && "only 1d vectors are supported");
    return vectorType.getDimSize(0);
  }

  return llvm::None;
}

// Returns vector width of the type. If the type is a scalar returns `1`.
static int vectorWidth(Type type) {
  auto vectorType = type.dyn_cast<VectorType>();
  return vectorType ? vectorType.getDimSize(0) : 1;
}

// Returns vector element type. If the type is a scalar returns the argument.
LLVM_ATTRIBUTE_UNUSED static Type elementType(Type type) {
  auto vectorType = type.dyn_cast<VectorType>();
  return vectorType ? vectorType.getElementType() : type;
}

LLVM_ATTRIBUTE_UNUSED static bool isF32(Type type) { return type.isF32(); }

LLVM_ATTRIBUTE_UNUSED static bool isI32(Type type) {
  return type.isInteger(32);
}

//----------------------------------------------------------------------------//
// Broadcast scalar types and values into vector types and values.
//----------------------------------------------------------------------------//

// Broadcasts scalar type into vector type (iff width is greater then 1).
static Type broadcast(Type type, int width) {
  assert(!type.isa<VectorType>() && "must be scalar type");
  return width > 1 ? VectorType::get({width}, type) : type;
}

// Broadcasts scalar value into vector (iff width is greater then 1).
static Value broadcast(ImplicitLocOpBuilder &builder, Value value, int width) {
  assert(!value.getType().isa<VectorType>() && "must be scalar value");
  auto type = broadcast(value.getType(), width);
  return width > 1 ? builder.create<BroadcastOp>(type, value) : value;
}

//----------------------------------------------------------------------------//
// Helper functions to create constants.
//----------------------------------------------------------------------------//

static Value f32Cst(ImplicitLocOpBuilder &builder, float value) {
  return builder.create<ConstantOp>(builder.getF32Type(),
                                    builder.getF32FloatAttr(value));
}

static Value i32Cst(ImplicitLocOpBuilder &builder, int32_t value) {
  return builder.create<ConstantOp>(builder.getI32Type(),
                                    builder.getI32IntegerAttr(value));
}

static Value f32FromBits(ImplicitLocOpBuilder &builder, uint32_t bits) {
  Value i32Value = i32Cst(builder, static_cast<int32_t>(bits));
  return builder.create<LLVM::BitcastOp>(builder.getF32Type(), i32Value);
}

//----------------------------------------------------------------------------//
// Helper functions to build math functions approximations.
//----------------------------------------------------------------------------//

static Value min(ImplicitLocOpBuilder &builder, Value a, Value b) {
  return builder.create<SelectOp>(
      builder.create<CmpFOp>(CmpFPredicate::OLT, a, b), a, b);
}

static Value max(ImplicitLocOpBuilder &builder, Value a, Value b) {
  return builder.create<SelectOp>(
      builder.create<CmpFOp>(CmpFPredicate::OGT, a, b), a, b);
}

static Value clamp(ImplicitLocOpBuilder &builder, Value value, Value lowerBound,
                   Value upperBound) {
  return max(builder, min(builder, value, upperBound), lowerBound);
}

// Decomposes given floating point value `arg` into a normalized fraction and
// an integral power of two (see std::frexp). Returned values have float type.
static std::pair<Value, Value> frexp(ImplicitLocOpBuilder &builder, Value arg,
                                     bool is_positive = false) {
  assert(isF32(elementType(arg.getType())) && "argument must be f32 type");

  int width = vectorWidth(arg.getType());

  auto bcast = [&](Value value) -> Value {
    return broadcast(builder, value, width);
  };

  auto i32 = builder.getIntegerType(32);
  auto i32Vec = broadcast(i32, width);
  auto f32Vec = broadcast(builder.getF32Type(), width);

  Value cst126f = f32Cst(builder, 126.0f);
  Value cstHalf = f32Cst(builder, 0.5f);
  Value cstInvMantMask = f32FromBits(builder, ~0x7f800000u);

  // Bitcast to i32 for bitwise operations.
  Value i32Half = builder.create<LLVM::BitcastOp>(i32, cstHalf);
  Value i32InvMantMask = builder.create<LLVM::BitcastOp>(i32, cstInvMantMask);
  Value i32Arg = builder.create<LLVM::BitcastOp>(i32Vec, arg);

  // Compute normalized fraction.
  Value tmp0 = builder.create<LLVM::AndOp>(i32Arg, bcast(i32InvMantMask));
  Value tmp1 = builder.create<LLVM::OrOp>(tmp0, bcast(i32Half));
  Value normalizedFraction = builder.create<LLVM::BitcastOp>(f32Vec, tmp1);

  // Compute exponent.
  Value arg0 = is_positive ? arg : builder.create<AbsFOp>(arg);
  Value biasedExponentBits = builder.create<UnsignedShiftRightOp>(
      builder.create<LLVM::BitcastOp>(i32Vec, arg0),
      bcast(i32Cst(builder, 23)));
  Value biasedExponent = builder.create<SIToFPOp>(f32Vec, biasedExponentBits);
  Value exponent = builder.create<SubFOp>(biasedExponent, bcast(cst126f));

  return {normalizedFraction, exponent};
}

// Computes exp2 for an i32 argument.
static Value exp2I32(ImplicitLocOpBuilder &builder, Value arg) {
  assert(isI32(elementType(arg.getType())) && "argument must be i32 type");

  int width = vectorWidth(arg.getType());

  auto bcast = [&](Value value) -> Value {
    return broadcast(builder, value, width);
  };

  auto f32Vec = broadcast(builder.getF32Type(), width);
  // The exponent of f32 located at 23-bit.
  auto exponetBitLocation = bcast(i32Cst(builder, 23));
  // Set the exponent bias to zero.
  auto bias = bcast(i32Cst(builder, 127));

  Value biasedArg = builder.create<AddIOp>(arg, bias);
  Value exp2ValueInt =
      builder.create<ShiftLeftOp>(biasedArg, exponetBitLocation);
  Value exp2ValueF32 = builder.create<LLVM::BitcastOp>(f32Vec, exp2ValueInt);

  return exp2ValueF32;
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
  auto width = vectorWidth(op.operand().getType(), isF32);
  if (!width.hasValue())
    return rewriter.notifyMatchFailure(op, "unsupported operand type");

  ImplicitLocOpBuilder builder(op->getLoc(), rewriter);
  auto bcast = [&](Value value) -> Value {
    return broadcast(builder, value, *width);
  };

  // Clamp operand into [plusClamp, minusClamp] range.
  Value minusClamp = bcast(f32Cst(builder, -7.9053111076354980f));
  Value plusClamp = bcast(f32Cst(builder, 7.90531110763549805f));
  Value x = clamp(builder, op.operand(), minusClamp, plusClamp);

  // Mask for tiny values that are approximated with `operand`.
  Value tiny = bcast(f32Cst(builder, 0.0004f));
  Value tinyMask = builder.create<CmpFOp>(
      CmpFPredicate::OLT, builder.create<AbsFOp>(op.operand()), tiny);

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
  Value x2 = builder.create<MulFOp>(x, x);

  // Evaluate the numerator polynomial p.
  Value p = builder.create<FmaFOp>(x2, alpha13, alpha11);
  p = builder.create<FmaFOp>(x2, p, alpha9);
  p = builder.create<FmaFOp>(x2, p, alpha7);
  p = builder.create<FmaFOp>(x2, p, alpha5);
  p = builder.create<FmaFOp>(x2, p, alpha3);
  p = builder.create<FmaFOp>(x2, p, alpha1);
  p = builder.create<MulFOp>(x, p);

  // Evaluate the denominator polynomial q.
  Value q = builder.create<FmaFOp>(x2, beta6, beta4);
  q = builder.create<FmaFOp>(x2, q, beta2);
  q = builder.create<FmaFOp>(x2, q, beta0);

  // Divide the numerator by the denominator.
  Value res =
      builder.create<SelectOp>(tinyMask, x, builder.create<DivFOp>(p, q));

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
  auto width = vectorWidth(op.operand().getType(), isF32);
  if (!width.hasValue())
    return rewriter.notifyMatchFailure(op, "unsupported operand type");

  ImplicitLocOpBuilder builder(op->getLoc(), rewriter);
  auto bcast = [&](Value value) -> Value {
    return broadcast(builder, value, *width);
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

  Value x = op.operand();

  // Truncate input values to the minimum positive normal.
  x = max(builder, x, cstMinNormPos);

  // Extract significant in the range [0.5,1) and exponent.
  std::pair<Value, Value> pair = frexp(builder, x, /*is_positive=*/true);
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
  Value mask = builder.create<CmpFOp>(CmpFPredicate::OLT, x, cstCephesSQRTHF);
  Value tmp = builder.create<SelectOp>(mask, x, cstZero);

  x = builder.create<SubFOp>(x, cstOne);
  e = builder.create<SubFOp>(e,
                             builder.create<SelectOp>(mask, cstOne, cstZero));
  x = builder.create<AddFOp>(x, tmp);

  Value x2 = builder.create<MulFOp>(x, x);
  Value x3 = builder.create<MulFOp>(x2, x);

  // Evaluate the polynomial approximant of degree 8 in three parts.
  Value y0, y1, y2;
  y0 = builder.create<FmaFOp>(cstCephesLogP0, x, cstCephesLogP1);
  y1 = builder.create<FmaFOp>(cstCephesLogP3, x, cstCephesLogP4);
  y2 = builder.create<FmaFOp>(cstCephesLogP6, x, cstCephesLogP7);
  y0 = builder.create<FmaFOp>(y0, x, cstCephesLogP2);
  y1 = builder.create<FmaFOp>(y1, x, cstCephesLogP5);
  y2 = builder.create<FmaFOp>(y2, x, cstCephesLogP8);
  y0 = builder.create<FmaFOp>(y0, x3, y1);
  y0 = builder.create<FmaFOp>(y0, x3, y2);
  y0 = builder.create<MulFOp>(y0, x3);

  y0 = builder.create<FmaFOp>(cstNegHalf, x2, y0);
  x = builder.create<AddFOp>(x, y0);

  if (base2) {
    Value cstLog2e = bcast(f32Cst(builder, static_cast<float>(LOG2E_VALUE)));
    x = builder.create<FmaFOp>(x, cstLog2e, e);
  } else {
    Value cstLn2 = bcast(f32Cst(builder, static_cast<float>(LN2_VALUE)));
    x = builder.create<FmaFOp>(e, cstLn2, x);
  }

  Value invalidMask =
      builder.create<CmpFOp>(CmpFPredicate::ULT, op.operand(), cstZero);
  Value zeroMask =
      builder.create<CmpFOp>(CmpFPredicate::OEQ, op.operand(), cstZero);
  Value posInfMask =
      builder.create<CmpFOp>(CmpFPredicate::OEQ, op.operand(), cstPosInf);

  // Filter out invalid values:
  //  • x == 0     -> -INF
  //  • x < 0      ->  NAN
  //  • x == +INF  -> +INF
  Value aproximation = builder.create<SelectOp>(
      zeroMask, cstMinusInf,
      builder.create<SelectOp>(
          invalidMask, cstNan,
          builder.create<SelectOp>(posInfMask, cstPosInf, x)));

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
  auto width = vectorWidth(op.operand().getType(), isF32);
  if (!width.hasValue())
    return rewriter.notifyMatchFailure(op, "unsupported operand type");

  ImplicitLocOpBuilder builder(op->getLoc(), rewriter);
  auto bcast = [&](Value value) -> Value {
    return broadcast(builder, value, *width);
  };

  // Approximate log(1+x) using the following, due to W. Kahan:
  //   u = x + 1.0;
  //   if (u == 1.0 || u == inf) return x;
  //   return x * log(u) / (u - 1.0);
  //          ^^^^^^^^^^^^^^^^^^^^^^
  //             "logLarge" below.
  Value cstOne = bcast(f32Cst(builder, 1.0f));
  Value x = op.operand();
  Value u = builder.create<AddFOp>(x, cstOne);
  Value uSmall = builder.create<CmpFOp>(CmpFPredicate::OEQ, u, cstOne);
  Value logU = builder.create<math::LogOp>(u);
  Value uInf = builder.create<CmpFOp>(CmpFPredicate::OEQ, u, logU);
  Value logLarge = builder.create<MulFOp>(
      x, builder.create<DivFOp>(logU, builder.create<SubFOp>(u, cstOne)));
  Value approximation = builder.create<SelectOp>(
      builder.create<LLVM::OrOp>(uSmall, uInf), x, logLarge);
  rewriter.replaceOp(op, approximation);
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
  auto width = vectorWidth(op.operand().getType(), isF32);
  if (!width.hasValue())
    return rewriter.notifyMatchFailure(op, "unsupported operand type");
  ImplicitLocOpBuilder builder(op->getLoc(), rewriter);

  // TODO: Consider a common pattern rewriter with all methods below to
  // write the approximations.
  auto bcast = [&](Value value) -> Value {
    return broadcast(builder, value, *width);
  };
  auto fmla = [&](Value a, Value b, Value c) {
    return builder.create<FmaFOp>(a, b, c);
  };
  auto mul = [&](Value a, Value b) -> Value {
    return builder.create<MulFOp>(a, b);
  };
  auto sub = [&](Value a, Value b) -> Value {
    return builder.create<SubFOp>(a, b);
  };
  auto floor = [&](Value a) { return builder.create<FloorFOp>(a); };

  Value cstLn2 = bcast(f32Cst(builder, static_cast<float>(LN2_VALUE)));
  Value cstLog2E = bcast(f32Cst(builder, static_cast<float>(LOG2E_VALUE)));

  // Polynomial coefficients.
  Value cstCephesExpP0 = bcast(f32Cst(builder, 1.0));
  Value cstCephesExpP1 = bcast(f32Cst(builder, 1.0));
  Value cstCephesExpP2 = bcast(f32Cst(builder, 0.49970514590562437052f));
  Value cstCephesExpP3 = bcast(f32Cst(builder, 0.16873890085469545053f));
  Value cstCephesExpP4 = bcast(f32Cst(builder, 0.03668965196652099192f));
  Value cstCephesExpP5 = bcast(f32Cst(builder, 0.01314350012789660196f));

  Value x = op.operand();

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

  auto i32Vec = broadcast(builder.getI32Type(), *width);

  // exp2(k)
  Value k = builder.create<FPToSIOp>(kF32, i32Vec);
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
  Value rightBound = builder.create<CmpIOp>(CmpIPredicate::sle, k, kMaxConst);
  Value leftBound = builder.create<CmpIOp>(CmpIPredicate::sge, k, kMaxNegConst);

  Value isNegInfinityX =
      builder.create<CmpFOp>(CmpFPredicate::OEQ, x, constNegIfinity);
  Value isPostiveX =
      builder.create<CmpFOp>(CmpFPredicate::OGT, x, zerof32Const);
  Value isComputable = builder.create<AndOp>(rightBound, leftBound);

  expY = builder.create<SelectOp>(
      isComputable, expY,
      builder.create<SelectOp>(
          isPostiveX, constPosInfinity,
          builder.create<SelectOp>(isNegInfinityX, zerof32Const, underflow)));

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
  auto width = vectorWidth(op.operand().getType(), isF32);
  if (!width.hasValue())
    return rewriter.notifyMatchFailure(op, "unsupported operand type");

  ImplicitLocOpBuilder builder(op->getLoc(), rewriter);
  auto bcast = [&](Value value) -> Value {
    return broadcast(builder, value, *width);
  };

  // expm1(x) = exp(x) - 1 = u - 1.
  // We have to handle it carefully when x is near 0, i.e. u ~= 1,
  // and when the input is ~= -inf, i.e. u - 1 ~= -1.
  Value cstOne = bcast(f32Cst(builder, 1.0f));
  Value cstNegOne = bcast(f32Cst(builder, -1.0f));
  Value x = op.operand();
  Value u = builder.create<math::ExpOp>(x);
  Value uEqOne = builder.create<CmpFOp>(CmpFPredicate::OEQ, u, cstOne);
  Value uMinusOne = builder.create<SubFOp>(u, cstOne);
  Value uMinusOneEqNegOne =
      builder.create<CmpFOp>(CmpFPredicate::OEQ, uMinusOne, cstNegOne);
  // logU = log(u) ~= x
  Value logU = builder.create<math::LogOp>(u);

  // Detect exp(x) = +inf; written this way to avoid having to form +inf.
  Value isInf = builder.create<CmpFOp>(CmpFPredicate::OEQ, logU, u);

  // (u - 1) * (x / ~x)
  Value expm1 =
      builder.create<MulFOp>(uMinusOne, builder.create<DivFOp>(x, logU));
  expm1 = builder.create<SelectOp>(isInf, u, expm1);
  Value approximation = builder.create<SelectOp>(
      uEqOne, x, builder.create<SelectOp>(uMinusOneEqNegOne, cstNegOne, expm1));
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
  auto width = vectorWidth(op.operand().getType(), isF32);
  if (!width.hasValue())
    return rewriter.notifyMatchFailure(op, "unsupported operand type");

  ImplicitLocOpBuilder builder(op->getLoc(), rewriter);
  auto bcast = [&](Value value) -> Value {
    return broadcast(builder, value, *width);
  };
  auto mul = [&](Value a, Value b) -> Value {
    return builder.create<MulFOp>(a, b);
  };
  auto sub = [&](Value a, Value b) -> Value {
    return builder.create<SubFOp>(a, b);
  };
  auto floor = [&](Value a) { return builder.create<FloorFOp>(a); };

  auto i32Vec = broadcast(builder.getI32Type(), *width);
  auto fPToSingedInteger = [&](Value a) -> Value {
    return builder.create<FPToSIOp>(a, i32Vec);
  };

  auto modulo4 = [&](Value a) -> Value {
    return builder.create<AndOp>(a, bcast(i32Cst(builder, 3)));
  };

  auto isEqualTo = [&](Value a, Value b) -> Value {
    return builder.create<CmpIOp>(CmpIPredicate::eq, a, b);
  };

  auto isGreaterThan = [&](Value a, Value b) -> Value {
    return builder.create<CmpIOp>(CmpIPredicate::sgt, a, b);
  };

  auto select = [&](Value cond, Value t, Value f) -> Value {
    return builder.create<SelectOp>(cond, t, f);
  };

  auto fmla = [&](Value a, Value b, Value c) {
    return builder.create<FmaFOp>(a, b, c);
  };

  auto bitwiseOr = [&](Value a, Value b) { return builder.create<OrOp>(a, b); };

  Value twoOverPi = bcast(f32Cst(builder, TWO_OVER_PI));
  Value piOverTwo = bcast(f32Cst(builder, PI_OVER_2));

  Value x = op.operand();

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

void mlir::populateMathPolynomialApproximationPatterns(
    RewritePatternSet &patterns) {
  patterns.add<TanhApproximation, LogApproximation, Log2Approximation,
               Log1pApproximation, ExpApproximation, ExpM1Approximation,
               SinAndCosApproximation<true, math::SinOp>,
               SinAndCosApproximation<false, math::CosOp>>(
      patterns.getContext());
}

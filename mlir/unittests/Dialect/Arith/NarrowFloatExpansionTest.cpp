//===- NarrowFloatExpansionTest.cpp - Narrow float expansion unit tests ---===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Checks the FloatFormatSpecs against APFloat, and the folded expansion (greedy
// driver, public populate functions) against APFloat::convert, from bf16, tf32,
// f32, f64, f80 and f128.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Arith/Utils/NarrowFloatExpansion.h"
#include "mlir/AsmParser/AsmParser.h"
#include "mlir/Dialect/Arith/Transforms/Passes.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Matchers.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/ADT/APFloat.h"
#include "llvm/ADT/StringExtras.h"
#include "gtest/gtest.h"
#include <optional>

using namespace mlir;

namespace {
/// How a format encodes NaN, if at all.
enum NaNFamily {
  NoNaN,      ///< FiniteOnly: no NaN and no Inf.
  IEEENaN,    ///< IEEE 754: an Inf and a NaN range above it.
  AllOnesNaN, ///< NanOnly: one NaN per sign, exponent and mantissa all-ones.
  NegZeroNaN, ///< NanOnly (FNUZ): one NaN, in the -0 slot.
};

struct FormatRow {
  const char *name; ///< The MLIR type, as parsed.
  unsigned sizeInBits, mantBits, expBits;
  int bias;
  bool isSigned;
  NaNFamily nanFamily;
  uint64_t maxFiniteBits; ///< The largest finite value, sign clear.
  uint64_t quietNaNBits;  ///< APFloat::getNaN; max-finite without a NaN.
  bool hasZero = true;
};

// E.g. f8E4M3FN (precision 4, minExponent -6): 3 mantissa and 4 exponent bits,
// bias 7, max-finite 0x7E = 448 (0x7F is the NaN). APFloat gives FNUZ formats a
// minimum exponent one lower, so f8E4M3FNUZ has bias 8 and f8E5M2FNUZ 16;
// f8E8M0FNU (no zero, minimum exponent -127) has bias 128, code c being
// 2^(c - 127).
const FormatRow kFormats[] = {
    {"f8E5M2", 8, 2, 5, 15, true, IEEENaN, 0x7B, 0x7E},
    {"f8E4M3", 8, 3, 4, 7, true, IEEENaN, 0x77, 0x7C},
    {"f8E3M4", 8, 4, 3, 3, true, IEEENaN, 0x6F, 0x78},
    {"f8E4M3FN", 8, 3, 4, 7, true, AllOnesNaN, 0x7E, 0x7F},
    {"f8E5M3FNU", 8, 3, 5, 15, false, AllOnesNaN, 0xFE, 0xFF},
    {"f8E8M0FNU", 8, 0, 8, 128, false, AllOnesNaN, 0xFE, 0xFF, false},
    {"f8E4M3FNUZ", 8, 3, 4, 8, true, NegZeroNaN, 0x7F, 0x80},
    {"f8E5M2FNUZ", 8, 2, 5, 16, true, NegZeroNaN, 0x7F, 0x80},
    {"f8E4M3B11FNUZ", 8, 3, 4, 11, true, NegZeroNaN, 0x7F, 0x80},
    {"f6E3M2FN", 6, 2, 3, 3, true, NoNaN, 0x1F, 0x1F},
    {"f6E2M3FN", 6, 3, 2, 1, true, NoNaN, 0x1F, 0x1F},
    {"f4E2M1FN", 4, 1, 2, 1, true, NoNaN, 0x7, 0x7},
    {"f16", 16, 10, 5, 15, true, IEEENaN, 0x7BFF, 0x7E00},
    {"bf16", 16, 7, 8, 127, true, IEEENaN, 0x7F7F, 0x7FC0},
    {"tf32", 19, 10, 8, 127, true, IEEENaN, 0x3FBFF, 0x3FE00},
    {"f32", 32, 23, 8, 127, true, IEEENaN, 0x7F7FFFFF, 0x7FC00000},
    {"f64", 64, 52, 11, 1023, true, IEEENaN, 0x7FEFFFFFFFFFFFFF,
     0x7FF8000000000000},
};

void expectSpecMatches(const arith::FloatFormatSpec &spec,
                       const FormatRow &row) {
  EXPECT_EQ(spec.sizeInBits, row.sizeInBits);
  EXPECT_EQ(spec.mantBits, row.mantBits);
  EXPECT_EQ(spec.expBits, row.expBits);
  EXPECT_EQ(spec.maxStoredExp, (1u << row.expBits) - 1);
  EXPECT_EQ(spec.bias, row.bias);
  EXPECT_EQ(spec.isSigned, row.isSigned);
  if (row.isSigned) {
    EXPECT_EQ(spec.signBitPos, row.sizeInBits - 1);
  }
  EXPECT_EQ(spec.hasInf, row.nanFamily == IEEENaN);
  EXPECT_EQ(spec.hasAllOnesNaN, row.nanFamily == AllOnesNaN);
  EXPECT_EQ(spec.isFNUZ, row.nanFamily == NegZeroNaN);
  EXPECT_EQ(spec.hasNaN(), row.nanFamily != NoNaN);
  EXPECT_EQ(spec.maxFiniteBitsAbs, row.maxFiniteBits);
  EXPECT_EQ(spec.quietNaNBits, row.quietNaNBits);
  EXPECT_EQ(spec.hasZero, row.hasZero);
}

TEST(FloatFormatSpecTest, BuiltinFloatTypes) {
  MLIRContext ctx;
  for (const FormatRow &row : kFormats) {
    SCOPED_TRACE(row.name);
    auto type = dyn_cast_if_present<FloatType>(parseType(row.name, &ctx));
    ASSERT_TRUE(type);
    expectSpecMatches(arith::floatFormatSpecFromFloatType(type), row);
    expectSpecMatches(
        arith::floatFormatSpecFromSemantics(type.getFloatSemantics()), row);
  }
}

// Apart from each signed IEEE format with itself, only f8E5M2 / f16 and any
// two of bf16 / tf32 / f32 share an exponent layout; f8E5M2FNUZ has f16's
// exponent width but bias 16 and no Inf, and f8E5M3FNU has no sign.
TEST(FloatFormatSpecTest, SameExponentLayout) {
  MLIRContext ctx;
  auto specOf = [&](StringRef name) {
    return arith::floatFormatSpecFromFloatType(
        cast<FloatType>(parseType(name, &ctx)));
  };
  auto isPair = [](StringRef a, StringRef b, StringRef x, StringRef y) {
    return (a == x && b == y) || (a == y && b == x);
  };
  for (const FormatRow &a : kFormats) {
    for (const FormatRow &b : kFormats) {
      SCOPED_TRACE(testing::Message() << a.name << " and " << b.name);
      bool expected = (StringRef(a.name) == b.name && a.nanFamily == IEEENaN) ||
                      isPair(a.name, b.name, "f8E5M2", "f16") ||
                      isPair(a.name, b.name, "bf16", "tf32") ||
                      isPair(a.name, b.name, "bf16", "f32") ||
                      isPair(a.name, b.name, "tf32", "f32");
      EXPECT_EQ(arith::haveSameExponentLayout(specOf(a.name), specOf(b.name)),
                expected);
    }
  }

  // Among the builtin formats only Inf separates a pair on its own (f8E4M3 and
  // f8E4M3FN); no two differ only in bias, say. So each condition is also
  // checked alone, on a copy of f16's spec with that property changed.
  const arith::FloatFormatSpec f16 = specOf("f16");
  arith::FloatFormatSpec otherBias = f16, otherExpBits = f16, noInf = f16,
                         noSign = f16;
  otherBias.bias += 1;
  otherExpBits.expBits += 1;
  noInf.hasInf = false;
  noSign.isSigned = false;
  for (const arith::FloatFormatSpec &other :
       {otherBias, otherExpBits, noInf, noSign}) {
    EXPECT_FALSE(arith::haveSameExponentLayout(f16, other));
    EXPECT_FALSE(arith::haveSameExponentLayout(other, f16));
  }
}

// Every APFloat format the expansion can describe (a zero and subnormals, or
// neither if unsigned with precision 1; at most 64 bits), so one added to
// APFloat later is covered too: each code a spec records decodes, through
// APFloat, to the value the expansion relies on.
TEST(FloatFormatSpecTest, EncodingsMatchAPFloat) {
  for (unsigned i = 0; i != APFloat::S_MaxSemantics + 1; ++i) {
    const llvm::fltSemantics &sem =
        APFloat::EnumToSemantics(static_cast<APFloat::Semantics>(i));
    if (sem.hasZero != sem.hasDenormals ||
        (!sem.hasZero && (APFloat::semanticsHasSignedRepr(sem) ||
                          APFloat::semanticsPrecision(sem) != 1)) ||
        APFloat::semanticsSizeInBits(sem) > 64)
      continue;
    SCOPED_TRACE(testing::Message() << "APFloat::Semantics " << i);
    arith::FloatFormatSpec spec = arith::floatFormatSpecFromSemantics(sem);
    unsigned size = spec.sizeInBits;
    auto decode = [&](uint64_t code) {
      return APFloat(sem, APInt(size, code));
    };

    EXPECT_EQ(size, APFloat::semanticsSizeInBits(sem));
    EXPECT_EQ(spec.mantBits, APFloat::semanticsPrecision(sem) - 1);
    EXPECT_EQ(spec.isSigned, APFloat::semanticsHasSignedRepr(sem));
    EXPECT_EQ(spec.hasInf, APFloat::semanticsHasInf(sem));
    EXPECT_EQ(spec.hasNaN(), APFloat::semanticsHasNaN(sem));
    EXPECT_EQ(spec.expBits + spec.mantBits + (spec.isSigned ? 1 : 0), size);
    EXPECT_EQ(spec.maxStoredExp, (1u << spec.expBits) - 1);
    EXPECT_EQ(spec.hasZero, sem.hasZero);

    // Exponent field 0 is the zero / subnormal binade, and field 1 holds the
    // smallest normal, 2^(1 - bias); without a zero, code 0 holds it.
    if (spec.hasZero) {
      EXPECT_TRUE(decode(0).isPosZero());
      EXPECT_TRUE(decode(1).bitwiseIsEqual(APFloat::getSmallest(sem)));
    }
    APFloat minNormal = decode(spec.hasZero ? uint64_t(1) << spec.mantBits : 0);
    EXPECT_TRUE(minNormal.bitwiseIsEqual(APFloat::getSmallestNormalized(sem)));
    EXPECT_EQ(ilogb(minNormal), 1 - spec.bias);
    EXPECT_EQ(spec.maxFiniteBitsAbs,
              APFloat::getLargest(sem).bitcastToAPInt().getZExtValue());

    uint64_t mantMask = (uint64_t(1) << spec.mantBits) - 1;
    uint64_t topExpCode = uint64_t(spec.maxStoredExp) << spec.mantBits;
    if (spec.isSigned) {
      uint64_t signBit = uint64_t(1) << spec.signBitPos;
      EXPECT_EQ(spec.signBitPos, size - 1);
      // The sign bit alone is -0, or the NaN of an FNUZ format.
      APFloat signOnly = decode(signBit);
      EXPECT_TRUE(spec.isFNUZ ? signOnly.isNaN() : signOnly.isNegZero());
      EXPECT_EQ(spec.isFNUZ, spec.quietNaNBits == signBit);
    }
    if (spec.hasInf) {
      EXPECT_TRUE(decode(topExpCode).isPosInfinity());
      // IEEE quiet NaN: the top mantissa bit set.
      EXPECT_EQ(spec.quietNaNBits,
                topExpCode | (uint64_t(1) << (spec.mantBits - 1)));
    }
    if (spec.hasAllOnesNaN) {
      EXPECT_EQ(spec.quietNaNBits, topExpCode | mantMask);
    }
    if (!spec.hasNaN()) {
      // No NaN to emit: a NaN source saturates to +max-finite.
      EXPECT_EQ(spec.quietNaNBits, spec.maxFiniteBitsAbs);
      continue;
    }
    APFloat quietNaN = decode(spec.quietNaNBits);
    EXPECT_TRUE(quietNaN.isNaN() && !quietNaN.isSignaling());
    EXPECT_EQ(spec.quietNaNBits,
              APFloat::getNaN(sem).bitcastToAPInt().getZExtValue());
  }
}

/// Folds a constant extf, or truncf without a rounding mode, into f64 or f128,
/// rounding to nearest even like the runtime conversion. TruncFOp::fold folds
/// only exact conversions, but the f80 and f128 expansion rounds through f64.
template <typename OpTy>
struct FoldWideConversion : public OpRewritePattern<OpTy> {
  using OpRewritePattern<OpTy>::OpRewritePattern;
  LogicalResult matchAndRewrite(OpTy op,
                                PatternRewriter &rewriter) const override {
    auto trunc = dyn_cast<arith::TruncFOp>(op.getOperation());
    if (trunc && trunc.getRoundingmodeAttr())
      return failure();
    auto type = dyn_cast<FloatType>(op.getType());
    FloatAttr in;
    if (!type || !(type.isF64() || type.isF128()) ||
        !matchPattern(op.getIn(), m_Constant(&in)))
      return failure();
    APFloat value = in.getValue();
    bool losesInfo = false;
    (void)value.convert(type.getFloatSemantics(), APFloat::rmNearestTiesToEven,
                        &losesInfo);
    rewriter.replaceOpWithNewOp<arith::ConstantOp>(op,
                                                   FloatAttr::get(type, value));
    return success();
  }
};

/// Expands `arith.truncf` of the `srcTy` constant `srcBits` to `dstTy` in `rnd`
/// (if set) with `populate`'s patterns (greedy driver, no folding), then folds.
/// Returns the folded bits, or std::nullopt if the truncf survived or did not
/// fold.
std::optional<uint64_t>
expandThenFold(MLIRContext &ctx, FloatType srcTy, const APInt &srcBits,
               FloatType dstTy, std::optional<arith::RoundingMode> rnd,
               function_ref<void(RewritePatternSet &)> populate) {
  Location loc = UnknownLoc::get(&ctx);
  OwningOpRef<ModuleOp> module = ModuleOp::create(loc);
  OpBuilder b(module->getBodyRegion());
  auto func = func::FuncOp::create(b, loc, "f", b.getFunctionType({}, dstTy));
  b.setInsertionPointToStart(func.addEntryBlock());
  APFloat src(srcTy.getFloatSemantics(), srcBits);
  Value cst = arith::ConstantOp::create(b, loc, FloatAttr::get(srcTy, src));
  auto rndAttr = rnd ? arith::RoundingModeAttr::get(&ctx, *rnd)
                     : arith::RoundingModeAttr();
  Value trunc = arith::TruncFOp::create(b, loc, dstTy, cst, rndAttr,
                                        arith::FastMathFlagsAttr{});
  func::ReturnOp::create(b, loc, trunc);

  RewritePatternSet patterns(&ctx);
  populate(patterns);
  if (failed(applyPatternsGreedily(func, std::move(patterns),
                                   GreedyRewriteConfig().enableFolding(false))))
    return std::nullopt;
  // TruncFOp::fold alone folds an exact conversion: the patterns must have
  // replaced the truncf (a truncf to f64 left is part of the expansion).
  if (llvm::any_of(func.getBody().getOps<arith::TruncFOp>(),
                   [&](arith::TruncFOp op) { return op.getType() == dstTy; }))
    return std::nullopt;
  RewritePatternSet folds(&ctx);
  folds.add<FoldWideConversion<arith::TruncFOp>,
            FoldWideConversion<arith::ExtFOp>>(&ctx);
  if (failed(applyPatternsGreedily(func, std::move(folds))))
    return std::nullopt;
  Value result = func.getBody().front().getTerminator()->getOperand(0);
  auto folded = result.getDefiningOp<arith::ConstantOp>();
  if (!folded)
    return std::nullopt;
  return cast<FloatAttr>(folded.getValue())
      .getValue()
      .bitcastToAPInt()
      .getZExtValue();
}

const std::pair<arith::RoundingMode, llvm::RoundingMode> kModes[] = {
    {arith::RoundingMode::to_nearest_even,
     llvm::RoundingMode::NearestTiesToEven},
    {arith::RoundingMode::to_nearest_away,
     llvm::RoundingMode::NearestTiesToAway},
    {arith::RoundingMode::toward_zero, llvm::RoundingMode::TowardZero},
    {arith::RoundingMode::upward, llvm::RoundingMode::TowardPositive},
    {arith::RoundingMode::downward, llvm::RoundingMode::TowardNegative},
};

/// Whether APFloat::convert of `x` into `sem` in `rnd` would make a NaN of x's
/// sign in a format without a sign (it asserts): whether |x| gets a NaN in the
/// mirrored mode.
bool needsNegativeNaN(const APFloat &x, const llvm::fltSemantics &sem,
                      llvm::RoundingMode rnd) {
  if (!x.isNegative() || APFloat::semanticsHasSignedRepr(sem))
    return false;
  if (rnd == llvm::RoundingMode::TowardPositive)
    rnd = llvm::RoundingMode::TowardNegative;
  else if (rnd == llvm::RoundingMode::TowardNegative)
    rnd = llvm::RoundingMode::TowardPositive;
  APFloat magnitude = abs(x);
  bool losesInfo = false;
  (void)magnitude.convert(sem, rnd, &losesInfo);
  return magnitude.isNaN();
}

/// Checks the folded expansion of every input in every rounding mode against
/// APFloat::convert. APFloat has no result for a NaN or an Inf into a format
/// without them; the expansion gives +max-finite and +-max-finite there. Nor
/// has it one where needsNegativeNaN holds; those inputs are skipped.
void expectFoldedMatchesAPFloat(
    MLIRContext &ctx, FloatType srcTy, ArrayRef<APInt> inputs, FloatType dstTy,
    function_ref<void(RewritePatternSet &)> populate) {
  const llvm::fltSemantics &dstSem = dstTy.getFloatSemantics();
  std::string types;
  llvm::raw_string_ostream(types) << srcTy << " to " << dstTy;
  for (const APInt &bits : inputs) {
    for (auto [rnd, llvmRnd] : kModes) {
      SCOPED_TRACE(testing::Message()
                   << types << " 0x"
                   << llvm::toString(bits, 16, /*Signed=*/false) << " "
                   << arith::stringifyRoundingMode(rnd));
      APFloat expected(srcTy.getFloatSemantics(), bits);
      if (!APFloat::semanticsHasNaN(dstSem) && !expected.isFinite()) {
        bool negativeInf = expected.isInfinity() && expected.isNegative();
        expected = APFloat::getLargest(dstSem, negativeInf);
      } else {
        if (needsNegativeNaN(expected, dstSem, llvmRnd))
          continue;
        bool losesInfo = false;
        (void)expected.convert(dstSem, llvmRnd, &losesInfo);
      }
      std::optional<uint64_t> got =
          expandThenFold(ctx, srcTy, bits, dstTy, rnd, populate);
      ASSERT_TRUE(got.has_value());
      EXPECT_EQ(*got, expected.bitcastToAPInt().getZExtValue());
    }
  }
}

void expectFoldedMatchesAPFloat(
    MLIRContext &ctx, FloatType dstTy, ArrayRef<uint32_t> inputs,
    function_ref<void(RewritePatternSet &)> populate) {
  SmallVector<APInt> bits;
  for (uint32_t input : inputs)
    bits.push_back(APInt(32, input));
  expectFoldedMatchesAPFloat(ctx, Float32Type::get(&ctx), bits, dstTy,
                             populate);
}

// f8E5M3FNU (no sign, no pass option) via the generic populate: zero, subnormal
// steps and ties, ties at 1.0, max-finite 114688 and the tie above it, an
// overflow, +Inf, a quiet and a signaling NaN; and negative values, which
// round with their sign (upward toward zero): ties at -1.0625 and -2^-18,
// -1.5*2^-17, -2^-19, -114688, -118784 (the tie above it), -1e6 and -0. Also
// bf16, rounded in a carrier wider than itself: -1.0625, -2^-18, 1.0625, -0.
TEST(NarrowFloatExpansionTest, UnsignedFoldedValuesMatchAPFloat) {
  MLIRContext ctx;
  ctx.loadDialect<arith::ArithDialect, func::FuncDialect>();
  const uint32_t inputs[] = {
      0x00000000, 0x00000001, 0x36000000, 0x36800000, 0x37000000,
      0x37400000, 0x3F800000, 0x3F880000, 0x3F880008, 0x3F980000,
      0x47E00000, 0x47E80000, 0x49742400, 0x7F800000, 0x7FC00000,
      0x7F800001, 0xBF880000, 0xBF880008, 0xB6800000, 0xB7400000,
      0xB6000000, 0xC7E00000, 0xC7E80000, 0xC9742400, 0x80000000};
  const uint16_t bf16Inputs[] = {0xBF88, 0xB680, 0x3F88, 0x8000};
  FloatType dstTy = Float8E5M3FNUType::get(&ctx);
  auto populate = [](RewritePatternSet &p) {
    arith::populateExpandNarrowFloatPatterns(p,
                                             TypeID::get<Float8E5M3FNUType>());
  };
  expectFoldedMatchesAPFloat(ctx, dstTy, inputs, populate);
  SmallVector<APInt> bf16Bits;
  for (uint16_t input : bf16Inputs)
    bf16Bits.push_back(APInt(16, input));
  expectFoldedMatchesAPFloat(ctx, BFloat16Type::get(&ctx), bf16Bits, dstTy,
                             populate);
}

// f8E8M0FNU (no zero, no sign; code c is 2^(c - 127)) through its populate
// function, from f32 (extended to f64) and f64: ties at 1.5 and 3 (they round
// up: the kept bit is the implicit one), 2^-127 (code 0), the tie 1.5*2^-127,
// 2^-128 and +-0 (code 0 too, without a zero), 2^127 (max-finite), the overflow
// tie 1.5*2^127, max-finite of f32 and f64, -3, -1.25, a negative overflow,
// +Inf and a NaN. Also tf32 (extended to f64 too): 1.5, -3, 2^-127 and
// 1.5*2^-127 (subnormals), -1.25, 2^127, max-finite and +Inf.
TEST(NarrowFloatExpansionTest, NoZeroFoldedValuesMatchAPFloat) {
  MLIRContext ctx;
  ctx.loadDialect<arith::ArithDialect, func::FuncDialect>();
  const uint32_t f32Inputs[] = {0x3FC00000, 0x40400000, 0x00400000, 0x00600000,
                                0x00200000, 0x00000000, 0x80000000, 0x7F000000,
                                0x7F400000, 0x7F7FFFFF, 0xC0400000, 0xBFA00000,
                                0xFF7FFFFF, 0x7F800000, 0x7FC00000};
  const uint64_t f64Inputs[] = {
      0x3FF8000000000000, 0x4008000000000000, 0x3800000000000000,
      0x3808000000000000, 0x37F0000000000000, 0x0000000000000001,
      0x8000000000000000, 0x47E0000000000000, 0x47E8000000000000,
      0x7FEFFFFFFFFFFFFF, 0xC008000000000000, 0xBFF4000000000000,
      0xC7F0000000000000, 0x7FF0000000000000, 0x7FF8000000000000};
  const uint32_t tf32Inputs[] = {0x1FE00, 0x60200, 0x00200, 0x00300,
                                 0x5FD00, 0x3F800, 0x3FBFF, 0x3FC00};
  FloatType dstTy = Float8E8M0FNUType::get(&ctx);
  expectFoldedMatchesAPFloat(ctx, dstTy, f32Inputs,
                             arith::populateExpandF8E8M0Patterns);
  SmallVector<APInt> f64Bits;
  for (uint64_t input : f64Inputs)
    f64Bits.push_back(APInt(64, input));
  expectFoldedMatchesAPFloat(ctx, Float64Type::get(&ctx), f64Bits, dstTy,
                             arith::populateExpandF8E8M0Patterns);
  SmallVector<APInt> tf32Bits;
  for (uint32_t input : tf32Inputs)
    tf32Bits.push_back(APInt(19, input));
  expectFoldedMatchesAPFloat(ctx, FloatTF32Type::get(&ctx), tf32Bits, dstTy,
                             arith::populateExpandF8E8M0Patterns);
}

// Without a rounding mode, populateExpandF8E8M0Patterns expands a truncf into
// f8E8M0FNU by its exponent, rounding toward zero (1.5 gives 1, code 0x7F),
// from f32 and f64 alike; the generic patterns leave it.
TEST(NarrowFloatExpansionTest, NoZeroWithoutRoundingMode) {
  MLIRContext ctx;
  ctx.loadDialect<arith::ArithDialect, func::FuncDialect>();
  FloatType dstTy = Float8E8M0FNUType::get(&ctx);
  auto generic = [](RewritePatternSet &p) {
    arith::populateExpandNarrowFloatPatterns(p,
                                             TypeID::get<Float8E8M0FNUType>());
  };
  const std::pair<FloatType, APInt> inputs[] = {
      {Float32Type::get(&ctx), APInt(32, 0x3FC00000)},
      {Float64Type::get(&ctx), APInt(64, 0x3FF8000000000000)}};
  for (const auto &[srcTy, bits] : inputs) {
    EXPECT_EQ(expandThenFold(ctx, srcTy, bits, dstTy, std::nullopt,
                             arith::populateExpandF8E8M0Patterns),
              std::optional<uint64_t>(0x7F));
    EXPECT_FALSE(
        expandThenFold(ctx, srcTy, bits, dstTy, std::nullopt, generic));
  }
}

// A signed format through its per-format populate function, negative inputs
// included: ties at 1.0625 and at -2^-10 (half the smallest subnormal), -448
// (max-finite) and -464 (the tie above it), a negative overflow, -Inf and -0.
TEST(NarrowFloatExpansionTest, SignedFoldedValuesMatchAPFloat) {
  MLIRContext ctx;
  ctx.loadDialect<arith::ArithDialect, func::FuncDialect>();
  const uint32_t inputs[] = {0x3F880000, 0xBF880008, 0x3B000000, 0xBA800000,
                             0xC3E00000, 0xC3E80000, 0xC9742400, 0xFF800000,
                             0x80000000, 0xFFC00000};
  expectFoldedMatchesAPFloat(ctx, Float8E4M3FNType::get(&ctx), inputs,
                             arith::populateExpandF8E4M3FNPatterns);
}

// A truncf from f128 or f80 rounds to odd into f64 first. The inputs are ones
// that rounding to nearest through f64 would round twice: a tie or a value of a
// destination plus or minus less than half an f64 ulp, a sticky bit past f64's
// precision, values above f64's max-finite and below its subnormals; also +-0,
// +-Inf and NaNs of each sign, quiet and signaling. Pairs are {high, low}.
TEST(NarrowFloatExpansionTest, WideSourceFoldedValuesMatchAPFloat) {
  MLIRContext ctx;
  ctx.loadDialect<arith::ArithDialect, func::FuncDialect>();
  const std::pair<uint64_t, uint64_t> f128Inputs[] = {
      {0x3FFF100000000000, 0x0000040000000000}, // 1.0625 + 2^-70
      {0x3FFF0FFFFFFFFFFF, 0xFFFFFC0000000000}, // 1.0625 - 2^-70
      {0xBFFF100000000000, 0x0000040000000000}, // -(1.0625 + 2^-70)
      {0x3FFF200000000000, 0x0000040000000000}, // 1.125 + 2^-70
      {0xBFFF1FFFFFFFFFFF, 0xFFFFFC0000000000}, // -(1.125 - 2^-70)
      {0x3FFF400000000000, 0x0000040000000000}, // 1.25 + 2^-70
      {0x3FFF3FFFFFFFFFFF, 0xFFFFFC0000000000}, // 1.25 - 2^-70
      {0x3FFF000000000000, 0x0000040000000000}, // 1 + 2^-70
      {0xBFFF000000000000, 0x0000040000000000}, // -(1 + 2^-70)
      {0x4001400000000000, 0x0000040000000000}, // 5 + 2^-68
      {0xC0013FFFFFFFFFFF, 0xFFFFFC0000000000}, // -(5 - 2^-68)
      {0x3FFF200000000000, 0x0000000000000001}, // 1.125 + 2^-112
      {0x3FFD000000000000, 0x0000000400000000}, // 0.25 + 2^-80
      {0x3FEE000000000000, 0x0000008000000000}, // 2^-17 + 2^-90
      {0xBFF4FFFFFFFFFFFF, 0xFFFFFFFE00000000}, // -(2^-10 - 2^-90)
      {0x4007D00000000000, 0x0000100000000000}, // 464 + 2^-60
      {0xC00EDFFFFFFFFFFF, 0xFFFFFF8000000000}, // -(61440 - 2^-58)
      {0x444B000000000000, 0x0000000000000000}, // 2^1100
      {0xC44B000000000000, 0x0000000000000000}, // -2^1100
      {0x43FEFFFFFFFFFFFF, 0xF400000000000000}, // f64 max + 2^969
      {0x43FEFFFFFFFFFFFF, 0xF800000000000200}, // f64 max + 2^970 + 2^920
      {0x3BB4000000000000, 0x0000000000000000}, // 2^-1099
      {0xBBB4000000000000, 0x0000000000000000}, // -2^-1099
      {0x0000000000000000, 0x0000000000000000}, // +0
      {0x8000000000000000, 0x0000000000000000}, // -0
      {0x7FFF000000000000, 0x0000000000000000}, // +Inf
      {0xFFFF000000000000, 0x0000000000000000}, // -Inf
      {0x7FFF800000000000, 0x0000000000000123}, // quiet NaN
      {0xFFFFE00000000000, 0x0000000000000000}, // -quiet NaN
      {0x7FFF400000000000, 0x0000000000000000}, // signaling NaN
      {0xFFFF000000000000, 0x0000000000000001}, // -signaling NaN
  };
  // The high 16 bits are sign and exponent; the low 64 the significand.
  const std::pair<uint64_t, uint64_t> f80Inputs[] = {
      {0x3FFF, 0x8800000000000001}, // 1.0625 + 2^-63
      {0xBFFF, 0x9000000000000001}, // -(1.125 + 2^-63)
      {0x3FFF, 0x9FFFFFFFFFFFFFFF}, // 1.25 - 2^-63
      {0x4007, 0xE800000000000001}, // 464 + 2^-55
      {0x7E7F, 0x8000000000000000}, // 2^16000
      {0x817F, 0x8000000000000000}, // -2^-16000
      {0x0000, 0x0000000000000001}, // 2^-16445
      {0x0000, 0x0000000000000000}, // +0
      {0x8000, 0x0000000000000000}, // -0
      {0x7FFF, 0x8000000000000000}, // +Inf
      {0x7FFF, 0xC000000000000123}, // quiet NaN
      {0xFFFF, 0xA000000000000000}, // -signaling NaN
  };
  SmallVector<APInt> f128Bits, f80Bits;
  for (auto [high, low] : f128Inputs)
    f128Bits.push_back(APInt(128, {low, high}));
  for (auto [high, low] : f80Inputs)
    f80Bits.push_back(APInt(80, {low, high}));
  const TypeID types[] = {
      TypeID::get<Float8E5M2Type>(), TypeID::get<Float8E4M3FNType>(),
      TypeID::get<Float4E2M1FNType>(), TypeID::get<Float8E5M2FNUZType>(),
      TypeID::get<Float8E8M0FNUType>()};
  auto populate = [&](RewritePatternSet &p) {
    arith::populateExpandNarrowFloatPatterns(p, types);
  };
  for (FloatType dstTy : {FloatType(Float8E5M2Type::get(&ctx)),
                          FloatType(Float8E4M3FNType::get(&ctx)),
                          FloatType(Float4E2M1FNType::get(&ctx)),
                          FloatType(Float8E5M2FNUZType::get(&ctx)),
                          FloatType(Float8E8M0FNUType::get(&ctx))}) {
    expectFoldedMatchesAPFloat(ctx, Float128Type::get(&ctx), f128Bits, dstTy,
                               populate);
    expectFoldedMatchesAPFloat(ctx, Float80Type::get(&ctx), f80Bits, dstTy,
                               populate);
  }
}

#if GTEST_HAS_DEATH_TEST && !defined(NDEBUG)
// x87's format stores its integer bit.
TEST(FloatFormatSpecDeathTest, RejectsExplicitIntegerBit) {
  EXPECT_DEATH(
      arith::floatFormatSpecFromSemantics(APFloat::x87DoubleExtended()),
      "the integer bit must be implicit");
}
#endif
} // namespace

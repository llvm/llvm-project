//===- NarrowFloatExpansionTest.cpp - Narrow float expansion unit tests ---===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Checks the FloatFormatSpecs against APFloat, and the folded expansion (greedy
// driver, public populate functions) against APFloat::convert, from bf16 and
// f32.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Arith/Utils/NarrowFloatExpansion.h"
#include "mlir/AsmParser/AsmParser.h"
#include "mlir/Dialect/Arith/Transforms/Passes.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/MLIRContext.h"
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
};

// E.g. f8E4M3FN (precision 4, minExponent -6): 3 mantissa and 4 exponent bits,
// bias 7, max-finite 0x7E = 448 (0x7F is the NaN). APFloat gives FNUZ formats a
// minimum exponent one lower, so f8E4M3FNUZ has bias 8 and f8E5M2FNUZ 16.
const FormatRow kFormats[] = {
    {"f8E5M2", 8, 2, 5, 15, true, IEEENaN, 0x7B, 0x7E},
    {"f8E4M3", 8, 3, 4, 7, true, IEEENaN, 0x77, 0x7C},
    {"f8E3M4", 8, 4, 3, 3, true, IEEENaN, 0x6F, 0x78},
    {"f8E4M3FN", 8, 3, 4, 7, true, AllOnesNaN, 0x7E, 0x7F},
    {"f8E5M3FNU", 8, 3, 5, 15, false, AllOnesNaN, 0xFE, 0xFF},
    {"f8E4M3FNUZ", 8, 3, 4, 8, true, NegZeroNaN, 0x7F, 0x80},
    {"f8E5M2FNUZ", 8, 2, 5, 16, true, NegZeroNaN, 0x7F, 0x80},
    {"f8E4M3B11FNUZ", 8, 3, 4, 11, true, NegZeroNaN, 0x7F, 0x80},
    {"f6E3M2FN", 6, 2, 3, 3, true, NoNaN, 0x1F, 0x1F},
    {"f6E2M3FN", 6, 3, 2, 1, true, NoNaN, 0x1F, 0x1F},
    {"f4E2M1FN", 4, 1, 2, 1, true, NoNaN, 0x7, 0x7},
    {"f16", 16, 10, 5, 15, true, IEEENaN, 0x7BFF, 0x7E00},
    {"bf16", 16, 7, 8, 127, true, IEEENaN, 0x7F7F, 0x7FC0},
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

// Every APFloat format the expansion can describe (a zero, subnormals, at most
// 64 bits), so one added to APFloat later is covered too: each code a spec
// records decodes, through APFloat, to the value the expansion relies on.
TEST(FloatFormatSpecTest, EncodingsMatchAPFloat) {
  for (unsigned i = 0; i != APFloat::S_MaxSemantics + 1; ++i) {
    const llvm::fltSemantics &sem =
        APFloat::EnumToSemantics(static_cast<APFloat::Semantics>(i));
    if (!sem.hasZero || !sem.hasDenormals ||
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

    // Exponent field 0 is the zero / subnormal binade, and field 1 holds the
    // smallest normal, 2^(1 - bias).
    EXPECT_TRUE(decode(0).isPosZero());
    EXPECT_TRUE(decode(1).bitwiseIsEqual(APFloat::getSmallest(sem)));
    APFloat minNormal = decode(uint64_t(1) << spec.mantBits);
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
  if (failed(applyPatternsGreedily(func, RewritePatternSet(&ctx))))
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

// A signed format, negative inputs included: ties at 1.0625 and at -2^-10
// (half the smallest subnormal), -448 (max-finite) and -464 (the tie above
// it), a negative overflow, -Inf and -0.
TEST(NarrowFloatExpansionTest, SignedFoldedValuesMatchAPFloat) {
  MLIRContext ctx;
  ctx.loadDialect<arith::ArithDialect, func::FuncDialect>();
  const uint32_t inputs[] = {0x3F880000, 0xBF880008, 0x3B000000, 0xBA800000,
                             0xC3E00000, 0xC3E80000, 0xC9742400, 0xFF800000,
                             0x80000000, 0xFFC00000};
  auto populate = [](RewritePatternSet &p) {
    arith::populateExpandNarrowFloatPatterns(p,
                                             TypeID::get<Float8E4M3FNType>());
  };
  expectFoldedMatchesAPFloat(ctx, Float8E4M3FNType::get(&ctx), inputs,
                             populate);
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

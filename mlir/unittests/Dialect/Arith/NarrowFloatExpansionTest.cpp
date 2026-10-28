//===- NarrowFloatExpansionTest.cpp - Narrow float expansion unit tests ---===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Checks the FloatFormatSpecs against APFloat.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Arith/Utils/NarrowFloatExpansion.h"
#include "mlir/AsmParser/AsmParser.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/MLIRContext.h"
#include "llvm/ADT/APFloat.h"
#include "gtest/gtest.h"

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

#if GTEST_HAS_DEATH_TEST && !defined(NDEBUG)
// x87's format stores its integer bit.
TEST(FloatFormatSpecDeathTest, RejectsExplicitIntegerBit) {
  EXPECT_DEATH(
      arith::floatFormatSpecFromSemantics(APFloat::x87DoubleExtended()),
      "the integer bit must be implicit");
}
#endif
} // namespace

//===-- APFloat.cpp - Implement APFloat class -----------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements a class to represent arbitrary precision floating
// point values and provide a variety of arithmetic operations on them.
//
//===----------------------------------------------------------------------===//

#include "llvm/ADT/APFloat.h"
#include "llvm/ADT/APSInt.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/FloatingPointMode.h"
#include "llvm/ADT/FoldingSet.h"
#include "llvm/ADT/Hashing.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Config/llvm-config.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/MathExtras.h"
#include "llvm/Support/raw_ostream.h"
#include <cstring>
#include <limits.h>

#define APFLOAT_DISPATCH_ON_SEMANTICS(METHOD_CALL)                             \
  do {                                                                         \
    if (usesLayout<IEEEFloat>(getSemantics()))                                 \
      return U.IEEE.METHOD_CALL;                                               \
    if (usesLayout<DoubleAPFloat>(getSemantics()))                             \
      return U.Double.METHOD_CALL;                                             \
    if (usesLayout<HexFloat>(getSemantics()))                                  \
      return U.Hex.METHOD_CALL;                                                \
    llvm_unreachable("Unexpected semantics");                                  \
  } while (false)

using namespace llvm;

/// A macro used to combine two fcCategory enums into one key which can be used
/// in a switch statement to classify how the interaction of two APFloat's
/// categories affects an operation.
///
/// TODO: If clang source code is ever allowed to use constexpr in its own
/// codebase, change this into a static inline function.
#define PackCategoriesIntoKey(_lhs, _rhs) ((_lhs) * 4 + (_rhs))

/* Assumed in hexadecimal significand parsing, and conversion to
   hexadecimal strings.  */
static_assert(APFloatBase::integerPartWidth % 4 == 0, "Part width must be divisible by 4!");

namespace llvm {

constexpr fltSemantics APFloatBase::semIEEEhalf = {"IEEEhalf", 15, -14, 11, 16};
constexpr fltSemantics APFloatBase::semBFloat = {"BFloat", 127, -126, 8, 16};
constexpr fltSemantics APFloatBase::semIEEEsingle = {"IEEESingle", 127, -126,
                                                     24, 32};
constexpr fltSemantics APFloatBase::semIEEEdouble = {"IEEEdouble", 1023, -1022,
                                                     53, 64};
constexpr fltSemantics APFloatBase::semIEEEquad = {"IEEEquad", 16383, -16382,
                                                   113, 128};
constexpr fltSemantics APFloatBase::semFloat8E5M2 = {"Float8E5M2", 15, -14, 3,
                                                     8};
constexpr fltSemantics APFloatBase::semFloat8E5M2FNUZ = {
    "Float8E5M2FNUZ",
    15,
    -15,
    3,
    8,
    2,
    fltNonfiniteBehavior::NanOnly,
    fltNanEncoding::NegativeZero};
constexpr fltSemantics APFloatBase::semFloat8E4M3 = {"Float8E4M3", 7, -6, 4, 8};
constexpr fltSemantics APFloatBase::semFloat8E4M3FN = {
    "Float8E4M3FN",         8, -6, 4, 8, 2, fltNonfiniteBehavior::NanOnly,
    fltNanEncoding::AllOnes};
constexpr fltSemantics APFloatBase::semFloat8E4M3FNUZ = {
    "Float8E4M3FNUZ",
    7,
    -7,
    4,
    8,
    2,
    fltNonfiniteBehavior::NanOnly,
    fltNanEncoding::NegativeZero};
constexpr fltSemantics APFloatBase::semFloat8E4M3B11FNUZ = {
    "Float8E4M3B11FNUZ",         4, -10, 4, 8, 2, fltNonfiniteBehavior::NanOnly,
    fltNanEncoding::NegativeZero};
constexpr fltSemantics APFloatBase::semFloat8E3M4 = {"Float8E3M4", 3, -2, 5, 8};
constexpr fltSemantics APFloatBase::semFloatTF32 = {"FloatTF32", 127, -126, 11,
                                                    19};
constexpr fltSemantics APFloatBase::semFloat8E8M0FNU = {
    "Float8E8M0FNU",
    127,
    -127,
    1,
    8,
    2,
    fltNonfiniteBehavior::NanOnly,
    fltNanEncoding::AllOnes,
    false,
    false,
    false};

constexpr fltSemantics APFloatBase::semFloat6E3M2FN = {
    "Float6E3M2FN", 4, -2, 3, 6, 2, fltNonfiniteBehavior::FiniteOnly};
constexpr fltSemantics APFloatBase::semFloat6E2M3FN = {
    "Float6E2M3FN", 2, 0, 4, 6, 2, fltNonfiniteBehavior::FiniteOnly};
constexpr fltSemantics APFloatBase::semFloat4E2M1FN = {
    "Float4E2M1FN", 2, 0, 2, 4, 2, fltNonfiniteBehavior::FiniteOnly};
constexpr fltSemantics APFloatBase::semX87DoubleExtended = {
    "X87DoubleExtended", 16383, -16382, 64, 80};
constexpr fltSemantics APFloatBase::semBogus = {"Bogus", 0, 0, 0, 0};
constexpr fltSemantics APFloatBase::semPPCDoubleDouble = {"PPCDoubleDouble", -1,
                                                          0, 0, 128};
constexpr fltSemantics APFloatBase::semPPCDoubleDoubleLegacy = {
    "PPCDoubleDoubleLegacy", 1023, -1022 + 53, 53 + 53, 128};

/* Hex float
   All three sizes share the same sign and exponent fields
   1 bit for sign
   7 bits for unsigned CHARACTERISTIC, which is the exponent + a bias of 64.
     Therefore the range of exponent is -64 .. +63.
     The exponent is for base 16.
     (Therefore, the minimum exponent for base 2 is 16^-64 == 2^-256
     The maximum exponent for base 2 is 16^63 == 2^252)
   For Hex_FP32, the next 24 bits are the SIGNIFICAND.
   For Hex_FP64, the next 56 bits are the SIGNIFICAND.
   For Hex_FP128, two 64-bit values are used.  The first is as Hex_FP64,
   In the second, the lower 56 bits form the lower significant part
   of the SIGNIFICAND.
   Precision is given in terms of the radix, e.g., Hex_FP32 has
   6 hexits of precision.
*/
constexpr fltSemantics APFloatBase::semHex_FP32 = {
    "Hex_FP32", 63, -64, 6, 32, 16, fltNonfiniteBehavior::FiniteOnly};
constexpr fltSemantics APFloatBase::semHex_FP64 = {
    "Hex_FP64", 63, -64, 14, 64, 16, fltNonfiniteBehavior::FiniteOnly};
constexpr fltSemantics APFloatBase::semHex_FP128 = {
    "Hex_FP128", 63, -64, 28, 128, 16, fltNonfiniteBehavior::FiniteOnly};

const llvm::fltSemantics &APFloatBase::EnumToSemantics(Semantics S) {
  switch (S) {
  case S_IEEEhalf:
    return IEEEhalf();
  case S_BFloat:
    return BFloat();
  case S_IEEEsingle:
    return IEEEsingle();
  case S_IEEEdouble:
    return IEEEdouble();
  case S_IEEEquad:
    return IEEEquad();
  case S_PPCDoubleDouble:
    return PPCDoubleDouble();
  case S_PPCDoubleDoubleLegacy:
    return PPCDoubleDoubleLegacy();
  case S_Float8E5M2:
    return Float8E5M2();
  case S_Float8E5M2FNUZ:
    return Float8E5M2FNUZ();
  case S_Float8E4M3:
    return Float8E4M3();
  case S_Float8E4M3FN:
    return Float8E4M3FN();
  case S_Float8E4M3FNUZ:
    return Float8E4M3FNUZ();
  case S_Float8E4M3B11FNUZ:
    return Float8E4M3B11FNUZ();
  case S_Float8E3M4:
    return Float8E3M4();
  case S_FloatTF32:
    return FloatTF32();
  case S_Float8E8M0FNU:
    return Float8E8M0FNU();
  case S_Float6E3M2FN:
    return Float6E3M2FN();
  case S_Float6E2M3FN:
    return Float6E2M3FN();
  case S_Float4E2M1FN:
    return Float4E2M1FN();
  case S_x87DoubleExtended:
    return x87DoubleExtended();
  case S_HexFP32:
    return HexFP32();
  case S_HexFP64:
    return HexFP64();
  case S_HexFP128:
    return HexFP128();
  }
  llvm_unreachable("Unrecognised floating semantics");
}

APFloatBase::Semantics
APFloatBase::SemanticsToEnum(const llvm::fltSemantics &Sem) {
  if (&Sem == &llvm::APFloat::IEEEhalf())
    return S_IEEEhalf;
  else if (&Sem == &llvm::APFloat::BFloat())
    return S_BFloat;
  else if (&Sem == &llvm::APFloat::IEEEsingle())
    return S_IEEEsingle;
  else if (&Sem == &llvm::APFloat::IEEEdouble())
    return S_IEEEdouble;
  else if (&Sem == &llvm::APFloat::IEEEquad())
    return S_IEEEquad;
  else if (&Sem == &llvm::APFloat::PPCDoubleDouble())
    return S_PPCDoubleDouble;
  else if (&Sem == &llvm::APFloat::PPCDoubleDoubleLegacy())
    return S_PPCDoubleDoubleLegacy;
  else if (&Sem == &llvm::APFloat::Float8E5M2())
    return S_Float8E5M2;
  else if (&Sem == &llvm::APFloat::Float8E5M2FNUZ())
    return S_Float8E5M2FNUZ;
  else if (&Sem == &llvm::APFloat::Float8E4M3())
    return S_Float8E4M3;
  else if (&Sem == &llvm::APFloat::Float8E4M3FN())
    return S_Float8E4M3FN;
  else if (&Sem == &llvm::APFloat::Float8E4M3FNUZ())
    return S_Float8E4M3FNUZ;
  else if (&Sem == &llvm::APFloat::Float8E4M3B11FNUZ())
    return S_Float8E4M3B11FNUZ;
  else if (&Sem == &llvm::APFloat::Float8E3M4())
    return S_Float8E3M4;
  else if (&Sem == &llvm::APFloat::FloatTF32())
    return S_FloatTF32;
  else if (&Sem == &llvm::APFloat::Float8E8M0FNU())
    return S_Float8E8M0FNU;
  else if (&Sem == &llvm::APFloat::Float6E3M2FN())
    return S_Float6E3M2FN;
  else if (&Sem == &llvm::APFloat::Float6E2M3FN())
    return S_Float6E2M3FN;
  else if (&Sem == &llvm::APFloat::Float4E2M1FN())
    return S_Float4E2M1FN;
  else if (&Sem == &llvm::APFloat::x87DoubleExtended())
    return S_x87DoubleExtended;
  else if (&Sem == &llvm::APFloat::HexFP32())
    return S_HexFP32;
  else if (&Sem == &llvm::APFloat::HexFP64())
    return S_HexFP64;
  else if (&Sem == &llvm::APFloat::HexFP128())
    return S_HexFP128;
  else
    llvm_unreachable("Unknown floating semantics");
}

bool APFloatBase::isRepresentableBy(const fltSemantics &A,
                                    const fltSemantics &B) {
  return A.maxExponent <= B.maxExponent && A.minExponent >= B.minExponent &&
         A.precision <= B.precision;
}

/* A tight upper bound on number of parts required to hold the value
   pow(5, power) is

     power * 815 / (351 * integerPartWidth) + 1

   However, whilst the result may require only this many parts,
   because we are multiplying two values to get it, the
   multiplication may require an extra part with the excess part
   being zero (consider the trivial case of 1 * 1, tcFullMultiply
   requires two parts to hold the single-part result).  So we add an
   extra one to guarantee enough space whilst multiplying.  */
const unsigned int maxExponent = 16383;
const unsigned int maxPrecision = 113;
const unsigned int maxPowerOfFiveExponent = maxExponent + maxPrecision - 1;
const unsigned int maxPowerOfFiveParts =
    2 +
    ((maxPowerOfFiveExponent * 815) / (351 * APFloatBase::integerPartWidth));

const char *APFloatBase::semanticsName(const fltSemantics &semantics) {
  return semantics.name;
}
unsigned int APFloatBase::semanticsPrecision(const fltSemantics &semantics) {
  return semantics.precision;
}
APFloatBase::ExponentType
APFloatBase::semanticsMaxExponent(const fltSemantics &semantics) {
  return semantics.maxExponent;
}
APFloatBase::ExponentType
APFloatBase::semanticsMinExponent(const fltSemantics &semantics) {
  return semantics.minExponent;
}
unsigned int APFloatBase::semanticsSizeInBits(const fltSemantics &semantics) {
  return semantics.sizeInBits;
}
unsigned int APFloatBase::semanticsIntSizeInBits(const fltSemantics &semantics,
                                                 bool isSigned) {
  // The max FP value is pow(2, MaxExponent) * (1 + MaxFraction), so we need
  // at least one more bit than the MaxExponent to hold the max FP value.
  unsigned int MinBitWidth = semanticsMaxExponent(semantics) + 1;
  // Extra sign bit needed.
  if (isSigned)
    ++MinBitWidth;
  return MinBitWidth;
}

bool APFloatBase::semanticsHasZero(const fltSemantics &semantics) {
  return semantics.hasZero;
}

bool APFloatBase::semanticsHasSignedRepr(const fltSemantics &semantics) {
  return semantics.hasSignedRepr;
}

unsigned int APFloatBase::semanticsRadix(const fltSemantics &semantics) {
  return semantics.radix;
}

bool APFloatBase::semanticsHasInf(const fltSemantics &semantics) {
  return semantics.nonFiniteBehavior == fltNonfiniteBehavior::IEEE754;
}

bool APFloatBase::semanticsHasNaN(const fltSemantics &semantics) {
  return semantics.nonFiniteBehavior != fltNonfiniteBehavior::FiniteOnly;
}

bool APFloatBase::isIEEELikeFP(const fltSemantics &semantics) {
  // Keep in sync with Type::isIEEELikeFPTy
  return SemanticsToEnum(semantics) <= S_IEEEquad;
}

bool APFloatBase::hasSignBitInMSB(const fltSemantics &semantics) {
  return semantics.hasSignBitInMSB;
}

bool APFloatBase::isRepresentableAsNormalIn(const fltSemantics &Src,
                                            const fltSemantics &Dst) {
  // Exponent range must be larger.
  if (Src.maxExponent >= Dst.maxExponent || Src.minExponent <= Dst.minExponent)
    return false;

  // If the mantissa is long enough, the result value could still be denormal
  // with a larger exponent range.
  //
  // FIXME: This condition is probably not accurate but also shouldn't be a
  // practical concern with existing types.
  return Dst.precision >= Src.precision;
}

unsigned APFloatBase::getSizeInBits(const fltSemantics &Sem) {
  return Sem.sizeInBits;
}

static constexpr APFloatBase::ExponentType
exponentZero(const fltSemantics &semantics) {
  return semantics.minExponent - 1;
}

static constexpr APFloatBase::ExponentType
exponentInf(const fltSemantics &semantics) {
  return semantics.maxExponent + 1;
}

static constexpr APFloatBase::ExponentType
exponentNaN(const fltSemantics &semantics) {
  if (semantics.nonFiniteBehavior == fltNonfiniteBehavior::NanOnly) {
    if (semantics.nanEncoding == fltNanEncoding::NegativeZero)
      return exponentZero(semantics);
    if (semantics.hasSignedRepr)
      return semantics.maxExponent;
  }
  return semantics.maxExponent + 1;
}

/* A bunch of private, handy routines.  */

static inline Error createError(const Twine &Err) {
  return make_error<StringError>(Err, inconvertibleErrorCode());
}

static constexpr inline unsigned int partCountForBits(unsigned int bits) {
  return std::max(1u, (bits + APFloatBase::integerPartWidth - 1) /
                          APFloatBase::integerPartWidth);
}

/* Returns 0U-9U.  Return values >= 10U are not digits.  */
static inline unsigned int
decDigitValue(unsigned int c)
{
  return c - '0';
}

/* Return the value of a decimal exponent of the form
   [+-]ddddddd.

   If the exponent overflows, returns a large exponent with the
   appropriate sign.  */
static Expected<int> readExponent(StringRef::iterator begin,
                                  StringRef::iterator end) {
  bool isNegative;
  unsigned int absExponent;
  const unsigned int overlargeExponent = 24000;  /* FIXME.  */
  StringRef::iterator p = begin;

  // Treat no exponent as 0 to match binutils
  if (p == end || ((*p == '-' || *p == '+') && (p + 1) == end)) {
    return 0;
  }

  isNegative = (*p == '-');
  if (*p == '-' || *p == '+') {
    p++;
    if (p == end)
      return createError("Exponent has no digits");
  }

  absExponent = decDigitValue(*p++);
  if (absExponent >= 10U)
    return createError("Invalid character in exponent");

  for (; p != end; ++p) {
    unsigned int value;

    value = decDigitValue(*p);
    if (value >= 10U)
      return createError("Invalid character in exponent");

    absExponent = absExponent * 10U + value;
    if (absExponent >= overlargeExponent) {
      absExponent = overlargeExponent;
      break;
    }
  }

  if (isNegative)
    return -(int) absExponent;
  else
    return (int) absExponent;
}

/* This is ugly and needs cleaning up, but I don't immediately see
   how whilst remaining safe.  */
static Expected<int> totalExponent(StringRef::iterator p,
                                   StringRef::iterator end,
                                   int exponentAdjustment) {
  int unsignedExponent;
  bool negative, overflow;
  int exponent = 0;

  if (p == end)
    return createError("Exponent has no digits");

  negative = *p == '-';
  if (*p == '-' || *p == '+') {
    p++;
    if (p == end)
      return createError("Exponent has no digits");
  }

  unsignedExponent = 0;
  overflow = false;
  for (; p != end; ++p) {
    unsigned int value;

    value = decDigitValue(*p);
    if (value >= 10U)
      return createError("Invalid character in exponent");

    unsignedExponent = unsignedExponent * 10 + value;
    if (unsignedExponent > 32767) {
      overflow = true;
      break;
    }
  }

  if (exponentAdjustment > 32767 || exponentAdjustment < -32768)
    overflow = true;

  if (!overflow) {
    exponent = unsignedExponent;
    if (negative)
      exponent = -exponent;
    exponent += exponentAdjustment;
    if (exponent > 32767 || exponent < -32768)
      overflow = true;
  }

  if (overflow)
    exponent = negative ? -32768: 32767;

  return exponent;
}

static Expected<StringRef::iterator>
skipLeadingZeroesAndAnyDot(StringRef::iterator begin, StringRef::iterator end,
                           StringRef::iterator *dot) {
  StringRef::iterator p = begin;
  *dot = end;
  while (p != end && *p == '0')
    p++;

  if (p != end && *p == '.') {
    *dot = p++;

    if (end - begin == 1)
      return createError("Significand has no digits");

    while (p != end && *p == '0')
      p++;
  }

  return p;
}

/* Given a normal decimal floating point number of the form

     dddd.dddd[eE][+-]ddd

   where the decimal point and exponent are optional, fill out the
   structure D.  Exponent is appropriate if the significand is
   treated as an integer, and normalizedExponent if the significand
   is taken to have the decimal point after a single leading
   non-zero digit.

   If the value is zero, V->firstSigDigit points to a non-digit, and
   the return exponent is zero.
*/
struct decimalInfo {
  const char *firstSigDigit;
  const char *lastSigDigit;
  int exponent;
  int normalizedExponent;
};

static Error interpretDecimal(StringRef::iterator begin,
                              StringRef::iterator end, decimalInfo *D) {
  StringRef::iterator dot = end;

  auto PtrOrErr = skipLeadingZeroesAndAnyDot(begin, end, &dot);
  if (!PtrOrErr)
    return PtrOrErr.takeError();
  StringRef::iterator p = *PtrOrErr;

  D->firstSigDigit = p;
  D->exponent = 0;
  D->normalizedExponent = 0;

  for (; p != end; ++p) {
    if (*p == '.') {
      if (dot != end)
        return createError("String contains multiple dots");
      dot = p++;
      if (p == end)
        break;
    }
    if (decDigitValue(*p) >= 10U)
      break;
  }

  if (p != end) {
    if (*p != 'e' && *p != 'E')
      return createError("Invalid character in significand");
    if (p == begin)
      return createError("Significand has no digits");
    if (dot != end && p - begin == 1)
      return createError("Significand has no digits");

    /* p points to the first non-digit in the string */
    auto ExpOrErr = readExponent(p + 1, end);
    if (!ExpOrErr)
      return ExpOrErr.takeError();
    D->exponent = *ExpOrErr;

    /* Implied decimal point?  */
    if (dot == end)
      dot = p;
  }

  /* If number is all zeroes accept any exponent.  */
  if (p != D->firstSigDigit) {
    /* Drop insignificant trailing zeroes.  */
    if (p != begin) {
      do
        do
          p--;
        while (p != begin && *p == '0');
      while (p != begin && *p == '.');
    }

    /* Adjust the exponents for any decimal point.  */
    D->exponent += static_cast<APFloat::ExponentType>((dot - p) - (dot > p));
    D->normalizedExponent = (D->exponent +
              static_cast<APFloat::ExponentType>((p - D->firstSigDigit)
                                      - (dot > D->firstSigDigit && dot < p)));
  }

  D->lastSigDigit = p;
  return Error::success();
}

/* Return the trailing fraction of a hexadecimal number.
   DIGITVALUE is the first hex digit of the fraction, P points to
   the next digit.  */
static Expected<lostFraction>
trailingHexadecimalFraction(StringRef::iterator p, StringRef::iterator end,
                            unsigned int digitValue) {
  unsigned int hexDigit;

  /* If the first trailing digit isn't 0 or 8 we can work out the
     fraction immediately.  */
  if (digitValue > 8)
    return lfMoreThanHalf;
  else if (digitValue < 8 && digitValue > 0)
    return lfLessThanHalf;

  // Otherwise we need to find the first non-zero digit.
  while (p != end && (*p == '0' || *p == '.'))
    p++;

  if (p == end)
    return createError("Invalid trailing hexadecimal fraction!");

  hexDigit = hexDigitValue(*p);

  /* If we ran off the end it is exactly zero or one-half, otherwise
     a little more.  */
  if (hexDigit == UINT_MAX)
    return digitValue == 0 ? lfExactlyZero: lfExactlyHalf;
  else
    return digitValue == 0 ? lfLessThanHalf: lfMoreThanHalf;
}

/* Return the fraction lost were a bignum truncated losing the least
   significant BITS bits.  */
static lostFraction
lostFractionThroughTruncation(const APFloatBase::integerPart *parts,
                              unsigned int partCount,
                              unsigned int bits)
{
  unsigned int lsb;

  lsb = APInt::tcLSB(parts, partCount);

  /* Note this is guaranteed true if bits == 0, or LSB == UINT_MAX.  */
  if (bits <= lsb)
    return lfExactlyZero;
  if (bits == lsb + 1)
    return lfExactlyHalf;
  if (bits <= partCount * APFloatBase::integerPartWidth &&
      APInt::tcExtractBit(parts, bits - 1))
    return lfMoreThanHalf;

  return lfLessThanHalf;
}

/* Shift DST right BITS bits noting lost fraction.  */
static lostFraction
shiftRight(APFloatBase::integerPart *dst, unsigned int parts, unsigned int bits)
{
  lostFraction lost_fraction;

  lost_fraction = lostFractionThroughTruncation(dst, parts, bits);

  APInt::tcShiftRight(dst, parts, bits);

  return lost_fraction;
}

/* Combine the effect of two lost fractions.  */
static lostFraction
combineLostFractions(lostFraction moreSignificant,
                     lostFraction lessSignificant)
{
  if (lessSignificant != lfExactlyZero) {
    if (moreSignificant == lfExactlyZero)
      moreSignificant = lfLessThanHalf;
    else if (moreSignificant == lfExactlyHalf)
      moreSignificant = lfMoreThanHalf;
  }

  return moreSignificant;
}

/* The error from the true value, in half-ulps, on multiplying two
   floating point numbers, which differ from the value they
   approximate by at most HUE1 and HUE2 half-ulps, is strictly less
   than the returned value.

   See "How to Read Floating Point Numbers Accurately" by William D
   Clinger.  */
static unsigned int
HUerrBound(bool inexactMultiply, unsigned int HUerr1, unsigned int HUerr2)
{
  assert(HUerr1 < 2 || HUerr2 < 2 || (HUerr1 + HUerr2 < 8));

  if (HUerr1 + HUerr2 == 0)
    return inexactMultiply * 2;  /* <= inexactMultiply half-ulps.  */
  else
    return inexactMultiply + 2 * (HUerr1 + HUerr2);
}

/* The number of ulps from the boundary (zero, or half if ISNEAREST)
   when the least significant BITS are truncated.  BITS cannot be
   zero.  */
static APFloatBase::integerPart
ulpsFromBoundary(const APFloatBase::integerPart *parts, unsigned int bits,
                 bool isNearest) {
  unsigned int count, partBits;
  APFloatBase::integerPart part, boundary;

  assert(bits != 0);

  bits--;
  count = bits / APFloatBase::integerPartWidth;
  partBits = bits % APFloatBase::integerPartWidth + 1;

  part = parts[count] & (~(APFloatBase::integerPart) 0 >> (APFloatBase::integerPartWidth - partBits));

  if (isNearest)
    boundary = (APFloatBase::integerPart) 1 << (partBits - 1);
  else
    boundary = 0;

  if (count == 0) {
    if (part - boundary <= boundary - part)
      return part - boundary;
    else
      return boundary - part;
  }

  if (part == boundary) {
    while (--count)
      if (parts[count])
        return ~(APFloatBase::integerPart) 0; /* A lot.  */

    return parts[0];
  } else if (part == boundary - 1) {
    while (--count)
      if (~parts[count])
        return ~(APFloatBase::integerPart) 0; /* A lot.  */

    return -parts[0];
  }

  return ~(APFloatBase::integerPart) 0; /* A lot.  */
}

/* Place pow(5, power) in DST, and return the number of parts used.
   DST must be at least one part larger than size of the answer.  */
static unsigned int
powerOf5(APFloatBase::integerPart *dst, unsigned int power) {
  static const APFloatBase::integerPart firstEightPowers[] = { 1, 5, 25, 125, 625, 3125, 15625, 78125 };
  APFloatBase::integerPart pow5s[maxPowerOfFiveParts * 2 + 5];
  pow5s[0] = 78125 * 5;

  unsigned int partsCount = 1;
  APFloatBase::integerPart scratch[maxPowerOfFiveParts], *p1, *p2, *pow5;
  unsigned int result;
  assert(power <= maxExponent);

  p1 = dst;
  p2 = scratch;

  *p1 = firstEightPowers[power & 7];
  power >>= 3;

  result = 1;
  pow5 = pow5s;

  for (unsigned int n = 0; power; power >>= 1, n++) {
    /* Calculate pow(5,pow(2,n+3)) if we haven't yet.  */
    if (n != 0) {
      APInt::tcFullMultiply(pow5, pow5 - partsCount, pow5 - partsCount,
                            partsCount, partsCount);
      partsCount *= 2;
      if (pow5[partsCount - 1] == 0)
        partsCount--;
    }

    if (power & 1) {
      APFloatBase::integerPart *tmp;

      APInt::tcFullMultiply(p2, p1, pow5, result, partsCount);
      result += partsCount;
      if (p2[result - 1] == 0)
        result--;

      /* Now result is in p1 with partsCount parts and p2 is scratch
         space.  */
      tmp = p1;
      p1 = p2;
      p2 = tmp;
    }

    pow5 += partsCount;
  }

  if (p1 != dst)
    APInt::tcAssign(dst, p1, result);

  return result;
}

/* Zero at the end to avoid modular arithmetic when adding one; used
   when rounding up during hexadecimal output.  */
static const char hexDigitsLower[] = "0123456789abcdef0";
static const char hexDigitsUpper[] = "0123456789ABCDEF0";
static const char infinityL[] = "infinity";
static const char infinityU[] = "INFINITY";
static const char NaNL[] = "nan";
static const char NaNU[] = "NAN";

/* Write out an integerPart in hexadecimal, starting with the most
   significant nibble.  Write out exactly COUNT hexdigits, return
   COUNT.  */
static unsigned int
partAsHex (char *dst, APFloatBase::integerPart part, unsigned int count,
           const char *hexDigitChars)
{
  unsigned int result = count;

  assert(count != 0 && count <= APFloatBase::integerPartWidth / 4);

  part >>= (APFloatBase::integerPartWidth - 4 * count);
  while (count--) {
    dst[count] = hexDigitChars[part & 0xf];
    part >>= 4;
  }

  return result;
}

/* Write out an unsigned decimal integer.  */
static char *
writeUnsignedDecimal (char *dst, unsigned int n)
{
  char buff[40], *p;

  p = buff;
  do
    *p++ = '0' + n % 10;
  while (n /= 10);

  do
    *dst++ = *--p;
  while (p != buff);

  return dst;
}

/* Write out a signed decimal integer.  */
static char *
writeSignedDecimal (char *dst, int value)
{
  if (value < 0) {
    *dst++ = '-';
    dst = writeUnsignedDecimal(dst, -(unsigned) value);
  } else {
    dst = writeUnsignedDecimal(dst, value);
  }

  return dst;
}

// Compute the ULP of the input using a definition from:
// Jean-Michel Muller. On the definition of ulp(x). [Research Report] RR-5504,
// LIP RR-2005-09, INRIA, LIP. 2005, pp.16. inria-00070503
static APFloat harrisonUlp(const APFloat &X) {
  const fltSemantics &Sem = X.getSemantics();
  switch (X.getCategory()) {
  case APFloat::fcNaN:
    return APFloat::getQNaN(Sem);
  case APFloat::fcInfinity:
    return APFloat::getInf(Sem);
  case APFloat::fcZero:
    return APFloat::getSmallest(Sem);
  case APFloat::fcNormal:
    break;
  }
  if (X.isDenormal() || X.isSmallestNormalized())
    return APFloat::getSmallest(Sem);
  int Exp = ilogb(X);
  if (X.getExactLog2() != INT_MIN)
    Exp -= 1;
  return scalbn(APFloat::getOne(Sem), Exp - (Sem.precision - 1),
                APFloat::rmNearestTiesToEven);
}

namespace detail {
/* Constructors.  */
void IEEEFloat::initialize(const fltSemantics *ourSemantics) {
  unsigned int count;

  semantics = ourSemantics;
  count = partCount();
  if (count > 1)
    significand.parts = new integerPart[count];
}

void IEEEFloat::freeSignificand() {
  if (needsCleanup())
    delete [] significand.parts;
}

void IEEEFloat::assign(const IEEEFloat &rhs) {
  assert(semantics == rhs.semantics);

  sign = rhs.sign;
  category = rhs.category;
  exponent = rhs.exponent;
  if (isFiniteNonZero() || category == fcNaN)
    copySignificand(rhs);
}

void IEEEFloat::copySignificand(const IEEEFloat &rhs) {
  assert(isFiniteNonZero() || category == fcNaN);
  assert(rhs.partCount() >= partCount());

  APInt::tcAssign(significandParts(), rhs.significandParts(),
                  partCount());
}

/* Make this number a NaN, with an arbitrary but deterministic value
   for the significand.  If double or longer, this is a signalling NaN,
   which may not be ideal.  If float, this is QNaN(0).  */
void IEEEFloat::makeNaN(bool SNaN, bool Negative, const APInt *fill) {
  if (semantics->nonFiniteBehavior == fltNonfiniteBehavior::FiniteOnly)
    llvm_unreachable("This floating point format does not support NaN");

  if (Negative && !semantics->hasSignedRepr)
    llvm_unreachable(
        "This floating point format does not support signed values");

  category = fcNaN;
  sign = Negative;
  exponent = exponentNaN();

  integerPart *significand = significandParts();
  unsigned numParts = partCount();

  APInt fill_storage;
  if (semantics->nonFiniteBehavior == fltNonfiniteBehavior::NanOnly) {
    // Finite-only types do not distinguish signalling and quiet NaN, so
    // make them all signalling.
    SNaN = false;
    if (semantics->nanEncoding == fltNanEncoding::NegativeZero) {
      sign = true;
      fill_storage = APInt::getZero(semantics->precision - 1);
    } else {
      fill_storage = APInt::getAllOnes(semantics->precision - 1);
    }
    fill = &fill_storage;
  }

  // Set the significand bits to the fill.
  if (!fill || fill->getNumWords() < numParts)
    APInt::tcSet(significand, 0, numParts);
  if (fill) {
    APInt::tcAssign(significand, fill->getRawData(),
                    std::min(fill->getNumWords(), numParts));

    // Zero out the excess bits of the significand.
    unsigned bitsToPreserve = semantics->precision - 1;
    unsigned part = bitsToPreserve / 64;
    bitsToPreserve %= 64;
    significand[part] &= ((1ULL << bitsToPreserve) - 1);
    for (part++; part != numParts; ++part)
      significand[part] = 0;
  }

  unsigned QNaNBit =
      (semantics->precision >= 2) ? (semantics->precision - 2) : 0;

  if (SNaN) {
    // We always have to clear the QNaN bit to make it an SNaN.
    APInt::tcClearBit(significand, QNaNBit);

    // If there are no bits set in the payload, we have to set
    // *something* to make it a NaN instead of an infinity;
    // conventionally, this is the next bit down from the QNaN bit.
    if (APInt::tcIsZero(significand, numParts))
      APInt::tcSetBit(significand, QNaNBit - 1);
  } else if (semantics->nanEncoding == fltNanEncoding::NegativeZero) {
    // The only NaN is a quiet NaN, and it has no bits sets in the significand.
    // Do nothing.
  } else {
    // We always have to set the QNaN bit to make it a QNaN.
    APInt::tcSetBit(significand, QNaNBit);
  }

  // For x87 extended precision, we want to make a NaN, not a
  // pseudo-NaN.  Maybe we should expose the ability to make
  // pseudo-NaNs?
  if (semantics == &APFloatBase::semX87DoubleExtended)
    APInt::tcSetBit(significand, QNaNBit + 1);
}

IEEEFloat &IEEEFloat::operator=(const IEEEFloat &rhs) {
  if (this != &rhs) {
    if (semantics != rhs.semantics) {
      freeSignificand();
      initialize(rhs.semantics);
    }
    assign(rhs);
  }

  return *this;
}

IEEEFloat &IEEEFloat::operator=(IEEEFloat &&rhs) {
  freeSignificand();

  semantics = rhs.semantics;
  significand = rhs.significand;
  exponent = rhs.exponent;
  category = rhs.category;
  sign = rhs.sign;

  rhs.semantics = &APFloatBase::semBogus;
  return *this;
}

bool IEEEFloat::isDenormal() const {
  return isFiniteNonZero() && (exponent == semantics->minExponent) &&
         (APInt::tcExtractBit(significandParts(),
                              semantics->precision - 1) == 0);
}

bool IEEEFloat::isSmallest() const {
  // The smallest number by magnitude in our format will be the smallest
  // denormal, i.e. the floating point number with exponent being minimum
  // exponent and significand bitwise equal to 1 (i.e. with MSB equal to 0).
  return isFiniteNonZero() && exponent == semantics->minExponent &&
    significandMSB() == 0;
}

bool IEEEFloat::isSmallestNormalized() const {
  return getCategory() == fcNormal && exponent == semantics->minExponent &&
         isSignificandAllZerosExceptMSB();
}

unsigned int IEEEFloat::getNumHighBits() const {
  const unsigned int PartCount = partCountForBits(semantics->precision);
  const unsigned int Bits = PartCount * integerPartWidth;

  // Compute how many bits are used in the final word.
  // When precision is just 1, it represents the 'Pth'
  // Precision bit and not the actual significand bit.
  const unsigned int NumHighBits = (semantics->precision > 1)
                                       ? (Bits - semantics->precision + 1)
                                       : (Bits - semantics->precision);
  return NumHighBits;
}

bool IEEEFloat::isSignificandAllOnes() const {
  // Test if the significand excluding the integral bit is all ones. This allows
  // us to test for binade boundaries.
  const integerPart *Parts = significandParts();
  const unsigned PartCount = partCountForBits(semantics->precision);
  for (unsigned i = 0; i < PartCount - 1; i++)
    if (~Parts[i])
      return false;

  // Set the unused high bits to all ones when we compare.
  const unsigned NumHighBits = getNumHighBits();
  assert(NumHighBits <= integerPartWidth && NumHighBits > 0 &&
         "Can not have more high bits to fill than integerPartWidth");
  const integerPart HighBitFill =
    ~integerPart(0) << (integerPartWidth - NumHighBits);
  if ((semantics->precision <= 1) || (~(Parts[PartCount - 1] | HighBitFill)))
    return false;

  return true;
}

bool IEEEFloat::isSignificandAllOnesExceptLSB() const {
  // Test if the significand excluding the integral bit is all ones except for
  // the least significant bit.
  const integerPart *Parts = significandParts();

  if (Parts[0] & 1)
    return false;

  const unsigned PartCount = partCountForBits(semantics->precision);
  for (unsigned i = 0; i < PartCount - 1; i++) {
    if (~Parts[i] & ~unsigned{!i})
      return false;
  }

  // Set the unused high bits to all ones when we compare.
  const unsigned NumHighBits = getNumHighBits();
  assert(NumHighBits <= integerPartWidth && NumHighBits > 0 &&
         "Can not have more high bits to fill than integerPartWidth");
  const integerPart HighBitFill = ~integerPart(0)
                                  << (integerPartWidth - NumHighBits);
  if (~(Parts[PartCount - 1] | HighBitFill | 0x1))
    return false;

  return true;
}

bool IEEEFloat::isSignificandAllZeros() const {
  // Test if the significand excluding the integral bit is all zeros. This
  // allows us to test for binade boundaries.
  const integerPart *Parts = significandParts();
  const unsigned PartCount = partCountForBits(semantics->precision);

  for (unsigned i = 0; i < PartCount - 1; i++)
    if (Parts[i])
      return false;

  // Compute how many bits are used in the final word.
  const unsigned NumHighBits = getNumHighBits();
  assert(NumHighBits < integerPartWidth && "Can not have more high bits to "
         "clear than integerPartWidth");
  const integerPart HighBitMask = ~integerPart(0) >> NumHighBits;

  if ((semantics->precision > 1) && (Parts[PartCount - 1] & HighBitMask))
    return false;

  return true;
}

bool IEEEFloat::isSignificandAllZerosExceptMSB() const {
  const integerPart *Parts = significandParts();
  const unsigned PartCount = partCountForBits(semantics->precision);

  for (unsigned i = 0; i < PartCount - 1; i++) {
    if (Parts[i])
      return false;
  }

  const unsigned NumHighBits = getNumHighBits();
  const integerPart MSBMask = integerPart(1)
                              << (integerPartWidth - NumHighBits);
  return ((semantics->precision <= 1) || (Parts[PartCount - 1] == MSBMask));
}

bool IEEEFloat::isLargest() const {
  bool IsMaxExp = isFiniteNonZero() && exponent == semantics->maxExponent;
  if (semantics->nonFiniteBehavior == fltNonfiniteBehavior::NanOnly &&
      semantics->nanEncoding == fltNanEncoding::AllOnes) {
    // The largest number by magnitude in our format will be the floating point
    // number with maximum exponent and with significand that is all ones except
    // the LSB.
    return (IsMaxExp && APFloat::hasSignificand(*semantics))
               ? isSignificandAllOnesExceptLSB()
               : IsMaxExp;
  } else {
    // The largest number by magnitude in our format will be the floating point
    // number with maximum exponent and with significand that is all ones.
    return IsMaxExp && isSignificandAllOnes();
  }
}

bool IEEEFloat::isInteger() const {
  // This could be made more efficient; I'm going for obviously correct.
  if (!isFinite()) return false;
  IEEEFloat truncated = *this;
  truncated.roundToIntegral(rmTowardZero);
  return compare(truncated) == cmpEqual;
}

bool IEEEFloat::bitwiseIsEqual(const IEEEFloat &rhs) const {
  if (this == &rhs)
    return true;
  if (semantics != rhs.semantics ||
      category != rhs.category ||
      sign != rhs.sign)
    return false;
  if (category==fcZero || category==fcInfinity)
    return true;

  if (isFiniteNonZero() && exponent != rhs.exponent)
    return false;

  return std::equal(significandParts(), significandParts() + partCount(),
                    rhs.significandParts());
}

IEEEFloat::IEEEFloat(const fltSemantics &ourSemantics, integerPart value) {
  initialize(&ourSemantics);
  sign = 0;
  category = fcNormal;
  zeroSignificand();
  exponent = ourSemantics.precision - 1;
  significandParts()[0] = value;
  normalize(rmNearestTiesToEven, lfExactlyZero);
}

IEEEFloat::IEEEFloat(const fltSemantics &ourSemantics) {
  initialize(&ourSemantics);
  // The Float8E8MOFNU format does not have a representation
  // for zero. So, use the closest representation instead.
  // Moreover, the all-zero encoding represents a valid
  // normal value (which is the smallestNormalized here).
  // Hence, we call makeSmallestNormalized (where category is
  // 'fcNormal') instead of makeZero (where category is 'fcZero').
  ourSemantics.hasZero ? makeZero(false) : makeSmallestNormalized(false);
}

// Delegate to the previous constructor, because later copy constructor may
// actually inspects category, which can't be garbage.
IEEEFloat::IEEEFloat(const fltSemantics &ourSemantics, uninitializedTag tag)
    : IEEEFloat(ourSemantics) {}

IEEEFloat::IEEEFloat(const IEEEFloat &rhs) {
  initialize(rhs.semantics);
  assign(rhs);
}

IEEEFloat::IEEEFloat(IEEEFloat &&rhs) : semantics(&APFloatBase::semBogus) {
  *this = std::move(rhs);
}

IEEEFloat::~IEEEFloat() { freeSignificand(); }

unsigned int IEEEFloat::partCount() const {
  return partCountForBits(semantics->precision + 1);
}

const APFloat::integerPart *IEEEFloat::significandParts() const {
  return const_cast<IEEEFloat *>(this)->significandParts();
}

APFloat::integerPart *IEEEFloat::significandParts() {
  if (partCount() > 1)
    return significand.parts;
  else
    return &significand.part;
}

void IEEEFloat::zeroSignificand() {
  APInt::tcSet(significandParts(), 0, partCount());
}

/* Increment an fcNormal floating point number's significand.  */
void IEEEFloat::incrementSignificand() {
  integerPart carry;

  carry = APInt::tcIncrement(significandParts(), partCount());

  /* Our callers should never cause us to overflow.  */
  assert(carry == 0);
  (void)carry;
}

/* Add the significand of the RHS.  Returns the carry flag.  */
APFloat::integerPart IEEEFloat::addSignificand(const IEEEFloat &rhs) {
  integerPart *parts;

  parts = significandParts();

  assert(semantics == rhs.semantics);
  assert(exponent == rhs.exponent);

  return APInt::tcAdd(parts, rhs.significandParts(), 0, partCount());
}

/* Subtract the significand of the RHS with a borrow flag.  Returns
   the borrow flag.  */
APFloat::integerPart IEEEFloat::subtractSignificand(const IEEEFloat &rhs,
                                                    integerPart borrow) {
  integerPart *parts;

  parts = significandParts();

  assert(semantics == rhs.semantics);
  assert(exponent == rhs.exponent);

  return APInt::tcSubtract(parts, rhs.significandParts(), borrow,
                           partCount());
}

/* Multiply the significand of the RHS.  If ADDEND is non-NULL, add it
   on to the full-precision result of the multiplication.  Returns the
   lost fraction.  */
lostFraction IEEEFloat::multiplySignificand(const IEEEFloat &rhs,
                                            IEEEFloat addend,
                                            bool ignoreAddend) {
  unsigned int omsb;        // One, not zero, based MSB.
  unsigned int partsCount, newPartsCount, precision;
  integerPart *lhsSignificand;
  integerPart scratch[4];
  integerPart *fullSignificand;
  lostFraction lost_fraction;
  bool ignored;

  assert(semantics == rhs.semantics);

  precision = semantics->precision;

  // Allocate space for twice as many bits as the original significand, plus one
  // extra bit for the addition to overflow into.
  newPartsCount = partCountForBits(precision * 2 + 1);

  if (newPartsCount > 4)
    fullSignificand = new integerPart[newPartsCount];
  else
    fullSignificand = scratch;

  lhsSignificand = significandParts();
  partsCount = partCount();

  APInt::tcFullMultiply(fullSignificand, lhsSignificand,
                        rhs.significandParts(), partsCount, partsCount);

  lost_fraction = lfExactlyZero;
  omsb = APInt::tcMSB(fullSignificand, newPartsCount) + 1;
  exponent += rhs.exponent;

  // Assume the operands involved in the multiplication are single-precision
  // FP, and the two multiplicants are:
  //   *this = a23 . a22 ... a0 * 2^e1
  //     rhs = b23 . b22 ... b0 * 2^e2
  // the result of multiplication is:
  //   *this = c48 c47 c46 . c45 ... c0 * 2^(e1+e2)
  // Note that there are three significant bits at the left-hand side of the
  // radix point: two for the multiplication, and an overflow bit for the
  // addition (that will always be zero at this point). Move the radix point
  // toward left by two bits, and adjust exponent accordingly.
  exponent += 2;

  if (!ignoreAddend && addend.isNonZero()) {
    // The intermediate result of the multiplication has "2 * precision"
    // signicant bit; adjust the addend to be consistent with mul result.
    //
    Significand savedSignificand = significand;
    const fltSemantics *savedSemantics = semantics;
    fltSemantics extendedSemantics;
    opStatus status;
    unsigned int extendedPrecision;

    // Normalize our MSB to one below the top bit to allow for overflow.
    extendedPrecision = 2 * precision + 1;
    if (omsb != extendedPrecision - 1) {
      assert(extendedPrecision > omsb);
      APInt::tcShiftLeft(fullSignificand, newPartsCount,
                         (extendedPrecision - 1) - omsb);
      exponent -= (extendedPrecision - 1) - omsb;
    }

    /* Create new semantics.  */
    extendedSemantics = *semantics;
    extendedSemantics.precision = extendedPrecision;

    if (newPartsCount == 1)
      significand.part = fullSignificand[0];
    else
      significand.parts = fullSignificand;
    semantics = &extendedSemantics;

    // Make a copy so we can convert it to the extended semantics.
    // Note that we cannot convert the addend directly, as the extendedSemantics
    // is a local variable (which we take a reference to).
    IEEEFloat extendedAddend(addend);
    status = extendedAddend.convert(extendedSemantics, APFloat::rmTowardZero,
                                    &ignored);
    assert(status == APFloat::opOK);
    (void)status;

    // Shift the significand of the addend right by one bit. This guarantees
    // that the high bit of the significand is zero (same as fullSignificand),
    // so the addition will overflow (if it does overflow at all) into the top bit.
    lost_fraction = extendedAddend.shiftSignificandRight(1);
    assert(lost_fraction == lfExactlyZero &&
           "Lost precision while shifting addend for fused-multiply-add.");

    lost_fraction = addOrSubtractSignificand(extendedAddend, false);

    /* Restore our state.  */
    if (newPartsCount == 1)
      fullSignificand[0] = significand.part;
    significand = savedSignificand;
    semantics = savedSemantics;

    omsb = APInt::tcMSB(fullSignificand, newPartsCount) + 1;
  }

  // Convert the result having "2 * precision" significant-bits back to the one
  // having "precision" significant-bits. First, move the radix point from
  // poision "2*precision - 1" to "precision - 1". The exponent need to be
  // adjusted by "2*precision - 1" - "precision - 1" = "precision".
  exponent -= precision + 1;

  // In case MSB resides at the left-hand side of radix point, shift the
  // mantissa right by some amount to make sure the MSB reside right before
  // the radix point (i.e. "MSB . rest-significant-bits").
  //
  // Note that the result is not normalized when "omsb < precision". So, the
  // caller needs to call IEEEFloat::normalize() if normalized value is
  // expected.
  if (omsb > precision) {
    unsigned int bits, significantParts;
    lostFraction lf;

    bits = omsb - precision;
    significantParts = partCountForBits(omsb);
    lf = shiftRight(fullSignificand, significantParts, bits);
    lost_fraction = combineLostFractions(lf, lost_fraction);
    exponent += bits;
  }

  APInt::tcAssign(lhsSignificand, fullSignificand, partsCount);

  if (newPartsCount > 4)
    delete [] fullSignificand;

  return lost_fraction;
}

lostFraction IEEEFloat::multiplySignificand(const IEEEFloat &rhs) {
  // When the given semantics has zero, the addend here is a zero.
  // i.e . it belongs to the 'fcZero' category.
  // But when the semantics does not support zero, we need to
  // explicitly convey that this addend should be ignored
  // for multiplication.
  return multiplySignificand(rhs, IEEEFloat(*semantics), !semantics->hasZero);
}

/* Multiply the significands of LHS and RHS to DST.  */
lostFraction IEEEFloat::divideSignificand(const IEEEFloat &rhs) {
  unsigned int bit, i, partsCount;
  const integerPart *rhsSignificand;
  integerPart *lhsSignificand, *dividend, *divisor;
  integerPart scratch[4];
  lostFraction lost_fraction;

  assert(semantics == rhs.semantics);

  lhsSignificand = significandParts();
  rhsSignificand = rhs.significandParts();
  partsCount = partCount();

  if (partsCount > 2)
    dividend = new integerPart[partsCount * 2];
  else
    dividend = scratch;

  divisor = dividend + partsCount;

  /* Copy the dividend and divisor as they will be modified in-place.  */
  for (i = 0; i < partsCount; i++) {
    dividend[i] = lhsSignificand[i];
    divisor[i] = rhsSignificand[i];
    lhsSignificand[i] = 0;
  }

  exponent -= rhs.exponent;

  unsigned int precision = semantics->precision;

  /* Normalize the divisor.  */
  bit = precision - APInt::tcMSB(divisor, partsCount) - 1;
  if (bit) {
    exponent += bit;
    APInt::tcShiftLeft(divisor, partsCount, bit);
  }

  /* Normalize the dividend.  */
  bit = precision - APInt::tcMSB(dividend, partsCount) - 1;
  if (bit) {
    exponent -= bit;
    APInt::tcShiftLeft(dividend, partsCount, bit);
  }

  /* Ensure the dividend >= divisor initially for the loop below.
     Incidentally, this means that the division loop below is
     guaranteed to set the integer bit to one.  */
  if (APInt::tcCompare(dividend, divisor, partsCount) < 0) {
    exponent--;
    APInt::tcShiftLeft(dividend, partsCount, 1);
    assert(APInt::tcCompare(dividend, divisor, partsCount) >= 0);
  }

  /* Long division.  */
  for (bit = precision; bit; bit -= 1) {
    if (APInt::tcCompare(dividend, divisor, partsCount) >= 0) {
      APInt::tcSubtract(dividend, divisor, 0, partsCount);
      APInt::tcSetBit(lhsSignificand, bit - 1);
    }

    APInt::tcShiftLeft(dividend, partsCount, 1);
  }

  /* Figure out the lost fraction.  */
  int cmp = APInt::tcCompare(dividend, divisor, partsCount);

  if (cmp > 0)
    lost_fraction = lfMoreThanHalf;
  else if (cmp == 0)
    lost_fraction = lfExactlyHalf;
  else if (APInt::tcIsZero(dividend, partsCount))
    lost_fraction = lfExactlyZero;
  else
    lost_fraction = lfLessThanHalf;

  if (partsCount > 2)
    delete [] dividend;

  return lost_fraction;
}

unsigned int IEEEFloat::significandMSB() const {
  return APInt::tcMSB(significandParts(), partCount());
}

unsigned int IEEEFloat::significandLSB() const {
  return APInt::tcLSB(significandParts(), partCount());
}

/* Note that a zero result is NOT normalized to fcZero.  */
lostFraction IEEEFloat::shiftSignificandRight(unsigned int bits) {
  /* Our exponent should not overflow.  */
  assert((ExponentType) (exponent + bits) >= exponent);

  exponent += bits;

  return shiftRight(significandParts(), partCount(), bits);
}

/* Shift the significand left BITS bits, subtract BITS from its exponent.  */
void IEEEFloat::shiftSignificandLeft(unsigned int bits) {
  assert(bits < semantics->precision ||
         (semantics->precision == 1 && bits <= 1));

  if (bits) {
    unsigned int partsCount = partCount();

    APInt::tcShiftLeft(significandParts(), partsCount, bits);
    exponent -= bits;

    assert(!APInt::tcIsZero(significandParts(), partsCount));
  }
}

APFloat::cmpResult IEEEFloat::compareAbsoluteValue(const IEEEFloat &rhs) const {
  int compare;

  assert(semantics == rhs.semantics);
  assert(isFiniteNonZero());
  assert(rhs.isFiniteNonZero());

  compare = exponent - rhs.exponent;

  /* If exponents are equal, do an unsigned bignum comparison of the
     significands.  */
  if (compare == 0)
    compare = APInt::tcCompare(significandParts(), rhs.significandParts(),
                               partCount());

  if (compare > 0)
    return cmpGreaterThan;
  else if (compare < 0)
    return cmpLessThan;
  else
    return cmpEqual;
}

/* Set the least significant BITS bits of a bignum, clear the
   rest.  */
static void tcSetLeastSignificantBits(APInt::WordType *dst, unsigned parts,
                                      unsigned bits) {
  unsigned i = 0;
  while (bits > APInt::APINT_BITS_PER_WORD) {
    dst[i++] = ~(APInt::WordType)0;
    bits -= APInt::APINT_BITS_PER_WORD;
  }

  if (bits)
    dst[i++] = ~(APInt::WordType)0 >> (APInt::APINT_BITS_PER_WORD - bits);

  while (i < parts)
    dst[i++] = 0;
}

/* Handle overflow.  Sign is preserved.  We either become infinity or
   the largest finite number.  */
APFloat::opStatus IEEEFloat::handleOverflow(roundingMode rounding_mode) {
  if (semantics->nonFiniteBehavior != fltNonfiniteBehavior::FiniteOnly) {
    /* Infinity?  */
    if (rounding_mode == rmNearestTiesToEven ||
        rounding_mode == rmNearestTiesToAway ||
        (rounding_mode == rmTowardPositive && !sign) ||
        (rounding_mode == rmTowardNegative && sign)) {
      if (semantics->nonFiniteBehavior == fltNonfiniteBehavior::NanOnly)
        makeNaN(false, sign);
      else
        category = fcInfinity;
      return static_cast<opStatus>(opOverflow | opInexact);
    }
  }

  /* Otherwise we become the largest finite number.  */
  category = fcNormal;
  exponent = semantics->maxExponent;
  tcSetLeastSignificantBits(significandParts(), partCount(),
                            semantics->precision);
  if (semantics->nonFiniteBehavior == fltNonfiniteBehavior::NanOnly &&
      semantics->nanEncoding == fltNanEncoding::AllOnes)
    APInt::tcClearBit(significandParts(), 0);

  return opInexact;
}

/* Returns TRUE if, when truncating the current number, with BIT the
   new LSB, with the given lost fraction and rounding mode, the result
   would need to be rounded away from zero (i.e., by increasing the
   signficand).  This routine must work for fcZero of both signs, and
   fcNormal numbers.  */
bool IEEEFloat::roundAwayFromZero(roundingMode rounding_mode,
                                  lostFraction lost_fraction,
                                  unsigned int bit) const {
  /* NaNs and infinities should not have lost fractions.  */
  assert(isFiniteNonZero() || category == fcZero);

  /* Current callers never pass this so we don't handle it.  */
  assert(lost_fraction != lfExactlyZero);

  switch (rounding_mode) {
  case rmNearestTiesToAway:
    return lost_fraction == lfExactlyHalf || lost_fraction == lfMoreThanHalf;

  case rmNearestTiesToEven:
    if (lost_fraction == lfMoreThanHalf)
      return true;

    /* Our zeroes don't have a significand to test.  */
    if (lost_fraction == lfExactlyHalf && category != fcZero)
      return APInt::tcExtractBit(significandParts(), bit);

    return false;

  case rmTowardZero:
    return false;

  case rmTowardPositive:
    return !sign;

  case rmTowardNegative:
    return sign;

  default:
    break;
  }
  llvm_unreachable("Invalid rounding mode found");
}

APFloat::opStatus IEEEFloat::normalize(roundingMode rounding_mode,
                                       lostFraction lost_fraction) {
  unsigned int omsb;                /* One, not zero, based MSB.  */
  int exponentChange;

  if (!isFiniteNonZero())
    return opOK;

  /* Before rounding normalize the exponent of fcNormal numbers.  */
  omsb = significandMSB() + 1;

  // Only skip this `if` if the value is exactly zero.
  if (omsb || lost_fraction != lfExactlyZero) {
    /* OMSB is numbered from 1.  We want to place it in the integer
       bit numbered PRECISION if possible, with a compensating change in
       the exponent.  */
    exponentChange = omsb - semantics->precision;

    /* If the resulting exponent is too high, overflow according to
       the rounding mode.  */
    if (exponent + exponentChange > semantics->maxExponent)
      return handleOverflow(rounding_mode);

    /* Subnormal numbers have exponent minExponent, and their MSB
       is forced based on that.  */
    if (exponent + exponentChange < semantics->minExponent)
      exponentChange = semantics->minExponent - exponent;

    /* Shifting left is easy as we don't lose precision.  */
    if (exponentChange < 0) {
      assert(lost_fraction == lfExactlyZero);

      shiftSignificandLeft(-exponentChange);

      return opOK;
    }

    if (exponentChange > 0) {
      lostFraction lf;

      /* Shift right and capture any new lost fraction.  */
      lf = shiftSignificandRight(exponentChange);

      lost_fraction = combineLostFractions(lf, lost_fraction);

      /* Keep OMSB up-to-date.  */
      if (omsb > (unsigned) exponentChange)
        omsb -= exponentChange;
      else
        omsb = 0;
    }
  }

  // The all-ones values is an overflow if NaN is all ones. If NaN is
  // represented by negative zero, then it is a valid finite value.
  if (semantics->nonFiniteBehavior == fltNonfiniteBehavior::NanOnly &&
      semantics->nanEncoding == fltNanEncoding::AllOnes &&
      exponent == semantics->maxExponent && isSignificandAllOnes())
    return handleOverflow(rounding_mode);

  /* Now round the number according to rounding_mode given the lost
     fraction.  */

  /* As specified in IEEE 754, since we do not trap we do not report
     underflow for exact results.  */
  if (lost_fraction == lfExactlyZero) {
    /* Canonicalize zeroes.  */
    if (omsb == 0) {
      category = fcZero;
      if (semantics->nanEncoding == fltNanEncoding::NegativeZero)
        sign = false;
      if (!semantics->hasZero)
        makeSmallestNormalized(false);
    }

    return opOK;
  }

  /* Increment the significand if we're rounding away from zero.  */
  if (roundAwayFromZero(rounding_mode, lost_fraction, 0)) {
    if (omsb == 0)
      exponent = semantics->minExponent;

    incrementSignificand();
    omsb = significandMSB() + 1;

    /* Did the significand increment overflow?  */
    if (omsb == (unsigned) semantics->precision + 1) {
      /* Renormalize by incrementing the exponent and shifting our
         significand right one.  However if we already have the
         maximum exponent we overflow to infinity.  */
      if (exponent == semantics->maxExponent)
        // Invoke overflow handling with a rounding mode that will guarantee
        // that the result gets turned into the correct infinity representation.
        // This is needed instead of just setting the category to infinity to
        // account for 8-bit floating point types that have no inf, only NaN.
        return handleOverflow(sign ? rmTowardNegative : rmTowardPositive);

      shiftSignificandRight(1);

      return opInexact;
    }

    // The all-ones values is an overflow if NaN is all ones. If NaN is
    // represented by negative zero, then it is a valid finite value.
    if (semantics->nonFiniteBehavior == fltNonfiniteBehavior::NanOnly &&
        semantics->nanEncoding == fltNanEncoding::AllOnes &&
        exponent == semantics->maxExponent && isSignificandAllOnes())
      return handleOverflow(rounding_mode);
  }

  /* The normal case - we were and are not denormal, and any
     significand increment above didn't overflow.  */
  if (omsb == semantics->precision)
    return opInexact;

  /* We have a non-zero denormal.  */
  assert(omsb < semantics->precision);

  /* Canonicalize zeroes.  */
  if (omsb == 0) {
    category = fcZero;
    if (semantics->nanEncoding == fltNanEncoding::NegativeZero)
      sign = false;
    // This condition handles the case where the semantics
    // does not have zero but uses the all-zero encoding
    // to represent the smallest normal value.
    if (!semantics->hasZero)
      makeSmallestNormalized(false);
  }

  /* The fcZero case is a denormal that underflowed to zero.  */
  return (opStatus) (opUnderflow | opInexact);
}

APFloat::opStatus IEEEFloat::addOrSubtractSpecials(const IEEEFloat &rhs,
                                                   bool subtract) {
  switch (PackCategoriesIntoKey(category, rhs.category)) {
  default:
    llvm_unreachable(nullptr);

  case PackCategoriesIntoKey(fcZero, fcNaN):
  case PackCategoriesIntoKey(fcNormal, fcNaN):
  case PackCategoriesIntoKey(fcInfinity, fcNaN):
    assign(rhs);
    [[fallthrough]];
  case PackCategoriesIntoKey(fcNaN, fcZero):
  case PackCategoriesIntoKey(fcNaN, fcNormal):
  case PackCategoriesIntoKey(fcNaN, fcInfinity):
  case PackCategoriesIntoKey(fcNaN, fcNaN):
    if (isSignaling()) {
      makeQuiet();
      return opInvalidOp;
    }
    return rhs.isSignaling() ? opInvalidOp : opOK;

  case PackCategoriesIntoKey(fcNormal, fcZero):
  case PackCategoriesIntoKey(fcInfinity, fcNormal):
  case PackCategoriesIntoKey(fcInfinity, fcZero):
    return opOK;

  case PackCategoriesIntoKey(fcNormal, fcInfinity):
  case PackCategoriesIntoKey(fcZero, fcInfinity):
    category = fcInfinity;
    sign = rhs.sign ^ subtract;
    return opOK;

  case PackCategoriesIntoKey(fcZero, fcNormal):
    assign(rhs);
    sign = rhs.sign ^ subtract;
    return opOK;

  case PackCategoriesIntoKey(fcZero, fcZero):
    /* Sign depends on rounding mode; handled by caller.  */
    return opOK;

  case PackCategoriesIntoKey(fcInfinity, fcInfinity):
    /* Differently signed infinities can only be validly
       subtracted.  */
    if (((sign ^ rhs.sign)!=0) != subtract) {
      makeNaN();
      return opInvalidOp;
    }

    return opOK;

  case PackCategoriesIntoKey(fcNormal, fcNormal):
    return opDivByZero;
  }
}

/* Add or subtract two normal numbers.  */
lostFraction IEEEFloat::addOrSubtractSignificand(const IEEEFloat &rhs,
                                                 bool subtract) {
  integerPart carry = 0;
  lostFraction lost_fraction;
  int bits;

  /* Determine if the operation on the absolute values is effectively
     an addition or subtraction.  */
  subtract ^= static_cast<bool>(sign ^ rhs.sign);

  /* Are we bigger exponent-wise than the RHS?  */
  bits = exponent - rhs.exponent;

  /* Subtraction is more subtle than one might naively expect.  */
  if (subtract) {
    if ((bits < 0) && !semantics->hasSignedRepr)
      llvm_unreachable(
          "This floating point format does not support signed values");

    IEEEFloat temp_rhs(rhs);
    bool lost_fraction_is_from_rhs = false;

    if (bits == 0)
      lost_fraction = lfExactlyZero;
    else if (bits > 0) {
      lost_fraction = temp_rhs.shiftSignificandRight(bits - 1);
      lost_fraction_is_from_rhs = true;
      shiftSignificandLeft(1);
    } else {
      lost_fraction = shiftSignificandRight(-bits - 1);
      temp_rhs.shiftSignificandLeft(1);
    }

    // Should we reverse the subtraction.
    cmpResult cmp_result = compareAbsoluteValue(temp_rhs);
    if (cmp_result == cmpLessThan) {
      bool borrow =
          lost_fraction != lfExactlyZero && !lost_fraction_is_from_rhs;
      if (borrow) {
        // The lost fraction is being subtracted, borrow from the significand
        // and invert `lost_fraction`.
        if (lost_fraction == lfLessThanHalf)
          lost_fraction = lfMoreThanHalf;
        else if (lost_fraction == lfMoreThanHalf)
          lost_fraction = lfLessThanHalf;
      }
      carry = temp_rhs.subtractSignificand(*this, borrow);
      copySignificand(temp_rhs);
      sign = !sign;
    } else if (cmp_result == cmpGreaterThan) {
      bool borrow = lost_fraction != lfExactlyZero && lost_fraction_is_from_rhs;
      if (borrow) {
        // The lost fraction is being subtracted, borrow from the significand
        // and invert `lost_fraction`.
        if (lost_fraction == lfLessThanHalf)
          lost_fraction = lfMoreThanHalf;
        else if (lost_fraction == lfMoreThanHalf)
          lost_fraction = lfLessThanHalf;
      }
      carry = subtractSignificand(temp_rhs, borrow);
    } else { // cmpEqual
      zeroSignificand();
      if (lost_fraction != lfExactlyZero && lost_fraction_is_from_rhs) {
        // rhs is slightly larger due to the lost fraction, flip the sign.
        sign = !sign;
      }
    }

    /* The code above is intended to ensure that no borrow is
       necessary.  */
    assert(!carry);
    (void)carry;
  } else {
    if (bits > 0) {
      IEEEFloat temp_rhs(rhs);

      lost_fraction = temp_rhs.shiftSignificandRight(bits);
      carry = addSignificand(temp_rhs);
    } else {
      lost_fraction = shiftSignificandRight(-bits);
      carry = addSignificand(rhs);
    }

    /* We have a guard bit; generating a carry cannot happen.  */
    assert(!carry);
    (void)carry;
  }

  return lost_fraction;
}

APFloat::opStatus IEEEFloat::multiplySpecials(const IEEEFloat &rhs) {
  switch (PackCategoriesIntoKey(category, rhs.category)) {
  default:
    llvm_unreachable(nullptr);

  case PackCategoriesIntoKey(fcZero, fcNaN):
  case PackCategoriesIntoKey(fcNormal, fcNaN):
  case PackCategoriesIntoKey(fcInfinity, fcNaN):
    assign(rhs);
    sign = false;
    [[fallthrough]];
  case PackCategoriesIntoKey(fcNaN, fcZero):
  case PackCategoriesIntoKey(fcNaN, fcNormal):
  case PackCategoriesIntoKey(fcNaN, fcInfinity):
  case PackCategoriesIntoKey(fcNaN, fcNaN):
    sign ^= rhs.sign; // restore the original sign
    if (isSignaling()) {
      makeQuiet();
      return opInvalidOp;
    }
    return rhs.isSignaling() ? opInvalidOp : opOK;

  case PackCategoriesIntoKey(fcNormal, fcInfinity):
  case PackCategoriesIntoKey(fcInfinity, fcNormal):
  case PackCategoriesIntoKey(fcInfinity, fcInfinity):
    category = fcInfinity;
    return opOK;

  case PackCategoriesIntoKey(fcZero, fcNormal):
  case PackCategoriesIntoKey(fcNormal, fcZero):
  case PackCategoriesIntoKey(fcZero, fcZero):
    category = fcZero;
    return opOK;

  case PackCategoriesIntoKey(fcZero, fcInfinity):
  case PackCategoriesIntoKey(fcInfinity, fcZero):
    makeNaN();
    return opInvalidOp;

  case PackCategoriesIntoKey(fcNormal, fcNormal):
    return opOK;
  }
}

APFloat::opStatus IEEEFloat::divideSpecials(const IEEEFloat &rhs) {
  switch (PackCategoriesIntoKey(category, rhs.category)) {
  default:
    llvm_unreachable(nullptr);

  case PackCategoriesIntoKey(fcZero, fcNaN):
  case PackCategoriesIntoKey(fcNormal, fcNaN):
  case PackCategoriesIntoKey(fcInfinity, fcNaN):
    assign(rhs);
    sign = false;
    [[fallthrough]];
  case PackCategoriesIntoKey(fcNaN, fcZero):
  case PackCategoriesIntoKey(fcNaN, fcNormal):
  case PackCategoriesIntoKey(fcNaN, fcInfinity):
  case PackCategoriesIntoKey(fcNaN, fcNaN):
    sign ^= rhs.sign; // restore the original sign
    if (isSignaling()) {
      makeQuiet();
      return opInvalidOp;
    }
    return rhs.isSignaling() ? opInvalidOp : opOK;

  case PackCategoriesIntoKey(fcInfinity, fcZero):
  case PackCategoriesIntoKey(fcInfinity, fcNormal):
  case PackCategoriesIntoKey(fcZero, fcInfinity):
  case PackCategoriesIntoKey(fcZero, fcNormal):
    return opOK;

  case PackCategoriesIntoKey(fcNormal, fcInfinity):
    category = fcZero;
    return opOK;

  case PackCategoriesIntoKey(fcNormal, fcZero):
    if (semantics->nonFiniteBehavior == fltNonfiniteBehavior::NanOnly)
      makeNaN(false, sign);
    else
      category = fcInfinity;
    return opDivByZero;

  case PackCategoriesIntoKey(fcInfinity, fcInfinity):
  case PackCategoriesIntoKey(fcZero, fcZero):
    makeNaN();
    return opInvalidOp;

  case PackCategoriesIntoKey(fcNormal, fcNormal):
    return opOK;
  }
}

APFloat::opStatus IEEEFloat::modSpecials(const IEEEFloat &rhs) {
  switch (PackCategoriesIntoKey(category, rhs.category)) {
  default:
    llvm_unreachable(nullptr);

  case PackCategoriesIntoKey(fcZero, fcNaN):
  case PackCategoriesIntoKey(fcNormal, fcNaN):
  case PackCategoriesIntoKey(fcInfinity, fcNaN):
    assign(rhs);
    [[fallthrough]];
  case PackCategoriesIntoKey(fcNaN, fcZero):
  case PackCategoriesIntoKey(fcNaN, fcNormal):
  case PackCategoriesIntoKey(fcNaN, fcInfinity):
  case PackCategoriesIntoKey(fcNaN, fcNaN):
    if (isSignaling()) {
      makeQuiet();
      return opInvalidOp;
    }
    return rhs.isSignaling() ? opInvalidOp : opOK;

  case PackCategoriesIntoKey(fcZero, fcInfinity):
  case PackCategoriesIntoKey(fcZero, fcNormal):
  case PackCategoriesIntoKey(fcNormal, fcInfinity):
    return opOK;

  case PackCategoriesIntoKey(fcNormal, fcZero):
  case PackCategoriesIntoKey(fcInfinity, fcZero):
  case PackCategoriesIntoKey(fcInfinity, fcNormal):
  case PackCategoriesIntoKey(fcInfinity, fcInfinity):
  case PackCategoriesIntoKey(fcZero, fcZero):
    makeNaN();
    return opInvalidOp;

  case PackCategoriesIntoKey(fcNormal, fcNormal):
    return opOK;
  }
}

APFloat::opStatus IEEEFloat::remainderSpecials(const IEEEFloat &rhs) {
  switch (PackCategoriesIntoKey(category, rhs.category)) {
  default:
    llvm_unreachable(nullptr);

  case PackCategoriesIntoKey(fcZero, fcNaN):
  case PackCategoriesIntoKey(fcNormal, fcNaN):
  case PackCategoriesIntoKey(fcInfinity, fcNaN):
    assign(rhs);
    [[fallthrough]];
  case PackCategoriesIntoKey(fcNaN, fcZero):
  case PackCategoriesIntoKey(fcNaN, fcNormal):
  case PackCategoriesIntoKey(fcNaN, fcInfinity):
  case PackCategoriesIntoKey(fcNaN, fcNaN):
    if (isSignaling()) {
      makeQuiet();
      return opInvalidOp;
    }
    return rhs.isSignaling() ? opInvalidOp : opOK;

  case PackCategoriesIntoKey(fcZero, fcInfinity):
  case PackCategoriesIntoKey(fcZero, fcNormal):
  case PackCategoriesIntoKey(fcNormal, fcInfinity):
    return opOK;

  case PackCategoriesIntoKey(fcNormal, fcZero):
  case PackCategoriesIntoKey(fcInfinity, fcZero):
  case PackCategoriesIntoKey(fcInfinity, fcNormal):
  case PackCategoriesIntoKey(fcInfinity, fcInfinity):
  case PackCategoriesIntoKey(fcZero, fcZero):
    makeNaN();
    return opInvalidOp;

  case PackCategoriesIntoKey(fcNormal, fcNormal):
    return opDivByZero; // fake status, indicating this is not a special case
  }
}

/* Change sign.  */
void IEEEFloat::changeSign() {
  // With NaN-as-negative-zero, neither NaN or negative zero can change
  // their signs.
  if (semantics->nanEncoding == fltNanEncoding::NegativeZero &&
      (isZero() || isNaN()))
    return;
  /* Look mummy, this one's easy.  */
  sign = !sign;
}

/* Normalized addition or subtraction.  */
APFloat::opStatus IEEEFloat::addOrSubtract(const IEEEFloat &rhs,
                                           roundingMode rounding_mode,
                                           bool subtract) {
  opStatus fs;

  fs = addOrSubtractSpecials(rhs, subtract);

  /* This return code means it was not a simple case.  */
  if (fs == opDivByZero) {
    lostFraction lost_fraction;

    lost_fraction = addOrSubtractSignificand(rhs, subtract);
    fs = normalize(rounding_mode, lost_fraction);

    /* Can only be zero if we lost no fraction.  */
    assert(category != fcZero || lost_fraction == lfExactlyZero);
  }

  /* If two numbers add (exactly) to zero, IEEE 754 decrees it is a
     positive zero unless rounding to minus infinity, except that
     adding two like-signed zeroes gives that zero.  */
  if (category == fcZero) {
    if (rhs.category != fcZero || (sign == rhs.sign) == subtract)
      sign = (rounding_mode == rmTowardNegative);
    // NaN-in-negative-zero means zeros need to be normalized to +0.
    if (semantics->nanEncoding == fltNanEncoding::NegativeZero)
      sign = false;
  }

  return fs;
}

/* Normalized addition.  */
APFloat::opStatus IEEEFloat::add(const IEEEFloat &rhs,
                                 roundingMode rounding_mode) {
  return addOrSubtract(rhs, rounding_mode, false);
}

/* Normalized subtraction.  */
APFloat::opStatus IEEEFloat::subtract(const IEEEFloat &rhs,
                                      roundingMode rounding_mode) {
  return addOrSubtract(rhs, rounding_mode, true);
}

/* Normalized multiply.  */
APFloat::opStatus IEEEFloat::multiply(const IEEEFloat &rhs,
                                      roundingMode rounding_mode) {
  opStatus fs;

  sign ^= rhs.sign;
  fs = multiplySpecials(rhs);

  if (isZero() && semantics->nanEncoding == fltNanEncoding::NegativeZero)
    sign = false;
  if (isFiniteNonZero()) {
    lostFraction lost_fraction = multiplySignificand(rhs);
    fs = normalize(rounding_mode, lost_fraction);
    if (lost_fraction != lfExactlyZero)
      fs = (opStatus) (fs | opInexact);
  }

  return fs;
}

/* Normalized divide.  */
APFloat::opStatus IEEEFloat::divide(const IEEEFloat &rhs,
                                    roundingMode rounding_mode) {
  opStatus fs;

  sign ^= rhs.sign;
  fs = divideSpecials(rhs);

  if (isZero() && semantics->nanEncoding == fltNanEncoding::NegativeZero)
    sign = false;
  if (isFiniteNonZero()) {
    lostFraction lost_fraction = divideSignificand(rhs);
    fs = normalize(rounding_mode, lost_fraction);
    if (lost_fraction != lfExactlyZero)
      fs = (opStatus) (fs | opInexact);
  }

  return fs;
}

/* Normalized remainder.  */
APFloat::opStatus IEEEFloat::remainder(const IEEEFloat &rhs) {
  opStatus fs;
  unsigned int origSign = sign;

  // First handle the special cases.
  fs = remainderSpecials(rhs);
  if (fs != opDivByZero)
    return fs;

  fs = opOK;

  // Make sure the current value is less than twice the denom. If the addition
  // did not succeed (an overflow has happened), which means that the finite
  // value we currently posses must be less than twice the denom (as we are
  // using the same semantics).
  IEEEFloat P2 = rhs;
  if (P2.add(rhs, rmNearestTiesToEven) == opOK) {
    fs = mod(P2);
    assert(fs == opOK);
  }

  // Lets work with absolute numbers.
  IEEEFloat P = rhs;
  P.sign = false;
  sign = false;

  //
  // To calculate the remainder we use the following scheme.
  //
  // The remainder is defained as follows:
  //
  // remainder = numer - rquot * denom = x - r * p
  //
  // Where r is the result of: x/p, rounded toward the nearest integral value
  // (with halfway cases rounded toward the even number).
  //
  // Currently, (after x mod 2p):
  // r is the number of 2p's present inside x, which is inherently, an even
  // number of p's.
  //
  // We may split the remaining calculation into 4 options:
  // - if x < 0.5p then we round to the nearest number with is 0, and are done.
  // - if x == 0.5p then we round to the nearest even number which is 0, and we
  //   are done as well.
  // - if 0.5p < x < p then we round to nearest number which is 1, and we have
  //   to subtract 1p at least once.
  // - if x >= p then we must subtract p at least once, as x must be a
  //   remainder.
  //
  // By now, we were done, or we added 1 to r, which in turn, now an odd number.
  //
  // We can now split the remaining calculation to the following 3 options:
  // - if x < 0.5p then we round to the nearest number with is 0, and are done.
  // - if x == 0.5p then we round to the nearest even number. As r is odd, we
  //   must round up to the next even number. so we must subtract p once more.
  // - if x > 0.5p (and inherently x < p) then we must round r up to the next
  //   integral, and subtract p once more.
  //

  // Extend the semantics to prevent an overflow/underflow or inexact result.
  bool losesInfo;
  fltSemantics extendedSemantics = *semantics;
  extendedSemantics.maxExponent++;
  extendedSemantics.minExponent--;
  extendedSemantics.precision += 2;

  IEEEFloat VEx = *this;
  fs = VEx.convert(extendedSemantics, rmNearestTiesToEven, &losesInfo);
  assert(fs == opOK && !losesInfo);
  IEEEFloat PEx = P;
  fs = PEx.convert(extendedSemantics, rmNearestTiesToEven, &losesInfo);
  assert(fs == opOK && !losesInfo);

  // It is simpler to work with 2x instead of 0.5p, and we do not need to lose
  // any fraction.
  fs = VEx.add(VEx, rmNearestTiesToEven);
  assert(fs == opOK);

  if (VEx.compare(PEx) == cmpGreaterThan) {
    fs = subtract(P, rmNearestTiesToEven);
    assert(fs == opOK);

    // Make VEx = this.add(this), but because we have different semantics, we do
    // not want to `convert` again, so we just subtract PEx twice (which equals
    // to the desired value).
    fs = VEx.subtract(PEx, rmNearestTiesToEven);
    assert(fs == opOK);
    fs = VEx.subtract(PEx, rmNearestTiesToEven);
    assert(fs == opOK);

    cmpResult result = VEx.compare(PEx);
    if (result == cmpGreaterThan || result == cmpEqual) {
      fs = subtract(P, rmNearestTiesToEven);
      assert(fs == opOK);
    }
  }

  if (isZero()) {
    sign = origSign;    // IEEE754 requires this
    if (semantics->nanEncoding == fltNanEncoding::NegativeZero)
      // But some 8-bit floats only have positive 0.
      sign = false;
  }

  else
    sign ^= origSign;
  return fs;
}

/* Normalized llvm frem (C fmod). */
APFloat::opStatus IEEEFloat::mod(const IEEEFloat &rhs) {
  opStatus fs;
  fs = modSpecials(rhs);
  unsigned int origSign = sign;

  while (isFiniteNonZero() && rhs.isFiniteNonZero() &&
         compareAbsoluteValue(rhs) != cmpLessThan) {
    int Exp = ilogb(*this) - ilogb(rhs);
    IEEEFloat V = scalbn(rhs, Exp, rmNearestTiesToEven);
    // V can overflow to NaN with fltNonfiniteBehavior::NanOnly, so explicitly
    // check for it.
    if (V.isNaN() || compareAbsoluteValue(V) == cmpLessThan)
      V = scalbn(rhs, Exp - 1, rmNearestTiesToEven);
    V.sign = sign;

    fs = subtract(V, rmNearestTiesToEven);

    // When the semantics supports zero, this loop's
    // exit-condition is handled by the 'isFiniteNonZero'
    // category check above. However, when the semantics
    // does not have 'fcZero' and we have reached the
    // minimum possible value, (and any further subtract
    // will underflow to the same value) explicitly
    // provide an exit-path here.
    if (!semantics->hasZero && this->isSmallest())
      break;

    assert(fs==opOK);
  }
  if (isZero()) {
    sign = origSign; // fmod requires this
    if (semantics->nanEncoding == fltNanEncoding::NegativeZero)
      sign = false;
  }
  return fs;
}

/* Normalized fused-multiply-add.  */
APFloat::opStatus IEEEFloat::fusedMultiplyAdd(const IEEEFloat &multiplicand,
                                              const IEEEFloat &addend,
                                              roundingMode rounding_mode) {
  opStatus fs;

  /* Post-multiplication sign, before addition.  */
  sign ^= multiplicand.sign;

  /* If and only if all arguments are normal do we need to do an
     extended-precision calculation.  */
  if (isFiniteNonZero() &&
      multiplicand.isFiniteNonZero() &&
      addend.isFinite()) {
    lostFraction lost_fraction;

    lost_fraction = multiplySignificand(multiplicand, addend);
    fs = normalize(rounding_mode, lost_fraction);
    if (lost_fraction != lfExactlyZero)
      fs = (opStatus) (fs | opInexact);

    /* If two numbers add (exactly) to zero, IEEE 754 decrees it is a
       positive zero unless rounding to minus infinity, except that
       adding two like-signed zeroes gives that zero.  */
    if (category == fcZero && !(fs & opUnderflow) && sign != addend.sign) {
      sign = (rounding_mode == rmTowardNegative);
      if (semantics->nanEncoding == fltNanEncoding::NegativeZero)
        sign = false;
    }
  } else {
    fs = multiplySpecials(multiplicand);

    /* FS can only be opOK or opInvalidOp.  There is no more work
       to do in the latter case.  The IEEE-754R standard says it is
       implementation-defined in this case whether, if ADDEND is a
       quiet NaN, we raise invalid op; this implementation does so.

       If we need to do the addition we can do so with normal
       precision.  */
    if (fs == opOK)
      fs = addOrSubtract(addend, rounding_mode, false);
  }

  return fs;
}

/* Rounding-mode correct round to integral value.  */
APFloat::opStatus IEEEFloat::roundToIntegral(roundingMode rounding_mode) {
  opStatus fs;

  if (isInfinity())
    // [IEEE Std 754-2008 6.1]:
    // The behavior of infinity in floating-point arithmetic is derived from the
    // limiting cases of real arithmetic with operands of arbitrarily
    // large magnitude, when such a limit exists.
    // ...
    // Operations on infinite operands are usually exact and therefore signal no
    // exceptions ...
    return opOK;

  if (isNaN()) {
    if (isSignaling()) {
      // [IEEE Std 754-2008 6.2]:
      // Under default exception handling, any operation signaling an invalid
      // operation exception and for which a floating-point result is to be
      // delivered shall deliver a quiet NaN.
      makeQuiet();
      // [IEEE Std 754-2008 6.2]:
      // Signaling NaNs shall be reserved operands that, under default exception
      // handling, signal the invalid operation exception(see 7.2) for every
      // general-computational and signaling-computational operation except for
      // the conversions described in 5.12.
      return opInvalidOp;
    } else {
      // [IEEE Std 754-2008 6.2]:
      // For an operation with quiet NaN inputs, other than maximum and minimum
      // operations, if a floating-point result is to be delivered the result
      // shall be a quiet NaN which should be one of the input NaNs.
      // ...
      // Every general-computational and quiet-computational operation involving
      // one or more input NaNs, none of them signaling, shall signal no
      // exception, except fusedMultiplyAdd might signal the invalid operation
      // exception(see 7.2).
      return opOK;
    }
  }

  if (isZero()) {
    // [IEEE Std 754-2008 6.3]:
    // ... the sign of the result of conversions, the quantize operation, the
    // roundToIntegral operations, and the roundToIntegralExact(see 5.3.1) is
    // the sign of the first or only operand.
    return opOK;
  }

  // If the exponent is large enough, we know that this value is already
  // integral, and the arithmetic below would potentially cause it to saturate
  // to +/-Inf.  Bail out early instead.
  if (exponent + 1 >= (int)APFloat::semanticsPrecision(*semantics))
    return opOK;

  // The algorithm here is quite simple: we add 2^(p-1), where p is the
  // precision of our format, and then subtract it back off again.  The choice
  // of rounding modes for the addition/subtraction determines the rounding mode
  // for our integral rounding as well.
  // NOTE: When the input value is negative, we do subtraction followed by
  // addition instead.
  APInt IntegerConstant(NextPowerOf2(APFloat::semanticsPrecision(*semantics)),
                        1);
  IntegerConstant <<= APFloat::semanticsPrecision(*semantics) - 1;
  IEEEFloat MagicConstant(*semantics);
  fs = MagicConstant.convertFromAPInt(IntegerConstant, false,
                                      rmNearestTiesToEven);
  assert(fs == opOK);
  MagicConstant.sign = sign;

  // Preserve the input sign so that we can handle the case of zero result
  // correctly.
  bool inputSign = isNegative();

  fs = add(MagicConstant, rounding_mode);

  // Current value and 'MagicConstant' are both integers, so the result of the
  // subtraction is always exact according to Sterbenz' lemma.
  subtract(MagicConstant, rounding_mode);

  // Restore the input sign.
  if (inputSign != isNegative())
    changeSign();

  return fs;
}

/* Comparison requires normalized numbers.  */
APFloat::cmpResult IEEEFloat::compare(const IEEEFloat &rhs) const {
  cmpResult result;

  assert(semantics == rhs.semantics);

  switch (PackCategoriesIntoKey(category, rhs.category)) {
  default:
    llvm_unreachable(nullptr);

  case PackCategoriesIntoKey(fcNaN, fcZero):
  case PackCategoriesIntoKey(fcNaN, fcNormal):
  case PackCategoriesIntoKey(fcNaN, fcInfinity):
  case PackCategoriesIntoKey(fcNaN, fcNaN):
  case PackCategoriesIntoKey(fcZero, fcNaN):
  case PackCategoriesIntoKey(fcNormal, fcNaN):
  case PackCategoriesIntoKey(fcInfinity, fcNaN):
    return cmpUnordered;

  case PackCategoriesIntoKey(fcInfinity, fcNormal):
  case PackCategoriesIntoKey(fcInfinity, fcZero):
  case PackCategoriesIntoKey(fcNormal, fcZero):
    if (sign)
      return cmpLessThan;
    else
      return cmpGreaterThan;

  case PackCategoriesIntoKey(fcNormal, fcInfinity):
  case PackCategoriesIntoKey(fcZero, fcInfinity):
  case PackCategoriesIntoKey(fcZero, fcNormal):
    if (rhs.sign)
      return cmpGreaterThan;
    else
      return cmpLessThan;

  case PackCategoriesIntoKey(fcInfinity, fcInfinity):
    if (sign == rhs.sign)
      return cmpEqual;
    else if (sign)
      return cmpLessThan;
    else
      return cmpGreaterThan;

  case PackCategoriesIntoKey(fcZero, fcZero):
    return cmpEqual;

  case PackCategoriesIntoKey(fcNormal, fcNormal):
    break;
  }

  /* Two normal numbers.  Do they have the same sign?  */
  if (sign != rhs.sign) {
    if (sign)
      result = cmpLessThan;
    else
      result = cmpGreaterThan;
  } else {
    /* Compare absolute values; invert result if negative.  */
    result = compareAbsoluteValue(rhs);

    if (sign) {
      if (result == cmpLessThan)
        result = cmpGreaterThan;
      else if (result == cmpGreaterThan)
        result = cmpLessThan;
    }
  }

  return result;
}

/// IEEEFloat::convert - convert a value of one floating point type to another.
/// The return value corresponds to the IEEE754 exceptions.  *losesInfo
/// records whether the transformation lost information, i.e. whether
/// converting the result back to the original type will produce the
/// original value (this is almost the same as return value==fsOK, but there
/// are edge cases where this is not so).

APFloat::opStatus IEEEFloat::convert(const fltSemantics &toSemantics,
                                     roundingMode rounding_mode,
                                     bool *losesInfo) {
  lostFraction lostFraction;
  unsigned int newPartCount, oldPartCount;
  opStatus fs;
  int shift;
  const fltSemantics &fromSemantics = *semantics;
  bool is_signaling = isSignaling();

  lostFraction = lfExactlyZero;
  newPartCount = partCountForBits(toSemantics.precision + 1);
  oldPartCount = partCount();
  shift = toSemantics.precision - fromSemantics.precision;

  bool X86SpecialNan = false;
  if (&fromSemantics == &APFloatBase::semX87DoubleExtended &&
      &toSemantics != &APFloatBase::semX87DoubleExtended && category == fcNaN &&
      (!(*significandParts() & 0x8000000000000000ULL) ||
       !(*significandParts() & 0x4000000000000000ULL))) {
    // x86 has some unusual NaNs which cannot be represented in any other
    // format; note them here.
    X86SpecialNan = true;
  }

  // If this is a truncation of a denormal number, and the target semantics
  // has larger exponent range than the source semantics (this can happen
  // when truncating from PowerPC double-double to double format), the
  // right shift could lose result mantissa bits.  Adjust exponent instead
  // of performing excessive shift.
  // Also do a similar trick in case shifting denormal would produce zero
  // significand as this case isn't handled correctly by normalize.
  if (shift < 0 && isFiniteNonZero()) {
    int omsb = significandMSB() + 1;
    int exponentChange = omsb - fromSemantics.precision;
    if (exponent + exponentChange < toSemantics.minExponent)
      exponentChange = toSemantics.minExponent - exponent;
    exponentChange = std::max(exponentChange, shift);
    if (exponentChange < 0) {
      shift -= exponentChange;
      exponent += exponentChange;
    } else if (omsb <= -shift) {
      exponentChange = omsb + shift - 1; // leave at least one bit set
      shift -= exponentChange;
      exponent += exponentChange;
    }
  }

  // If this is a truncation, perform the shift before we narrow the storage.
  if (shift < 0 && (isFiniteNonZero() ||
                    (category == fcNaN && semantics->nonFiniteBehavior !=
                                              fltNonfiniteBehavior::NanOnly)))
    lostFraction = shiftRight(significandParts(), oldPartCount, -shift);

  // Fix the storage so it can hold to new value.
  if (newPartCount > oldPartCount) {
    // The new type requires more storage; make it available.
    integerPart *newParts;
    newParts = new integerPart[newPartCount];
    APInt::tcSet(newParts, 0, newPartCount);
    if (isFiniteNonZero() || category==fcNaN)
      APInt::tcAssign(newParts, significandParts(), oldPartCount);
    freeSignificand();
    significand.parts = newParts;
  } else if (newPartCount == 1 && oldPartCount != 1) {
    // Switch to built-in storage for a single part.
    integerPart newPart = 0;
    if (isFiniteNonZero() || category==fcNaN)
      newPart = significandParts()[0];
    freeSignificand();
    significand.part = newPart;
  }

  // Now that we have the right storage, switch the semantics.
  semantics = &toSemantics;

  // If this is an extension, perform the shift now that the storage is
  // available.
  if (shift > 0 && (isFiniteNonZero() || category==fcNaN))
    APInt::tcShiftLeft(significandParts(), newPartCount, shift);

  if (isFiniteNonZero()) {
    fs = normalize(rounding_mode, lostFraction);
    *losesInfo = (fs != opOK);
  } else if (category == fcNaN) {
    if (semantics->nonFiniteBehavior == fltNonfiniteBehavior::NanOnly) {
      *losesInfo =
          fromSemantics.nonFiniteBehavior != fltNonfiniteBehavior::NanOnly;
      makeNaN(false, sign);
      return is_signaling ? opInvalidOp : opOK;
    }

    // If NaN is negative zero, we need to create a new NaN to avoid converting
    // NaN to -Inf.
    if (fromSemantics.nanEncoding == fltNanEncoding::NegativeZero &&
        semantics->nanEncoding != fltNanEncoding::NegativeZero)
      makeNaN(false, false);

    *losesInfo = lostFraction != lfExactlyZero || X86SpecialNan;

    // For x87 extended precision, we want to make a NaN, not a special NaN if
    // the input wasn't special either.
    if (!X86SpecialNan && semantics == &APFloatBase::semX87DoubleExtended)
      APInt::tcSetBit(significandParts(), semantics->precision - 1);

    // Convert of sNaN creates qNaN and raises an exception (invalid op).
    // This also guarantees that a sNaN does not become Inf on a truncation
    // that loses all payload bits.
    if (is_signaling) {
      makeQuiet();
      fs = opInvalidOp;
    } else {
      fs = opOK;
    }
  } else if (category == fcInfinity &&
             semantics->nonFiniteBehavior == fltNonfiniteBehavior::NanOnly) {
    makeNaN(false, sign);
    *losesInfo = true;
    fs = opInexact;
  } else if (category == fcZero &&
             semantics->nanEncoding == fltNanEncoding::NegativeZero) {
    // Negative zero loses info, but positive zero doesn't.
    *losesInfo =
        fromSemantics.nanEncoding != fltNanEncoding::NegativeZero && sign;
    fs = *losesInfo ? opInexact : opOK;
    // NaN is negative zero means -0 -> +0, which can lose information
    sign = false;
  } else {
    *losesInfo = false;
    fs = opOK;
  }

  if (category == fcZero && !semantics->hasZero)
    makeSmallestNormalized(false);
  return fs;
}

/* Convert a floating point number to an integer according to the
   rounding mode.  If the rounded integer value is out of range this
   returns an invalid operation exception and the contents of the
   destination parts are unspecified.  If the rounded value is in
   range but the floating point number is not the exact integer, the C
   standard doesn't require an inexact exception to be raised.  IEEE
   854 does require it so we do that.

   Note that for conversions to integer type the C standard requires
   round-to-zero to always be used.  */
APFloat::opStatus IEEEFloat::convertToSignExtendedInteger(
    MutableArrayRef<integerPart> parts, unsigned int width, bool isSigned,
    roundingMode rounding_mode, bool *isExact) const {
  lostFraction lost_fraction;
  const integerPart *src;
  unsigned int dstPartsCount, truncatedBits;

  *isExact = false;

  /* Handle the three special cases first.  */
  if (category == fcInfinity || category == fcNaN)
    return opInvalidOp;

  dstPartsCount = partCountForBits(width);
  assert(dstPartsCount <= parts.size() && "Integer too big");

  if (category == fcZero) {
    APInt::tcSet(parts.data(), 0, dstPartsCount);
    // Negative zero can't be represented as an int.
    *isExact = !sign;
    return opOK;
  }

  src = significandParts();

  /* Step 1: place our absolute value, with any fraction truncated, in
     the destination.  */
  if (exponent < 0) {
    /* Our absolute value is less than one; truncate everything.  */
    APInt::tcSet(parts.data(), 0, dstPartsCount);
    /* For exponent -1 the integer bit represents .5, look at that.
       For smaller exponents leftmost truncated bit is 0. */
    truncatedBits = semantics->precision -1U - exponent;
  } else {
    /* We want the most significant (exponent + 1) bits; the rest are
       truncated.  */
    unsigned int bits = exponent + 1U;

    /* Hopelessly large in magnitude?  */
    if (bits > width)
      return opInvalidOp;

    if (bits < semantics->precision) {
      /* We truncate (semantics->precision - bits) bits.  */
      truncatedBits = semantics->precision - bits;
      APInt::tcExtract(parts.data(), dstPartsCount, src, bits, truncatedBits);
    } else {
      /* We want at least as many bits as are available.  */
      APInt::tcExtract(parts.data(), dstPartsCount, src, semantics->precision,
                       0);
      APInt::tcShiftLeft(parts.data(), dstPartsCount,
                         bits - semantics->precision);
      truncatedBits = 0;
    }
  }

  /* Step 2: work out any lost fraction, and increment the absolute
     value if we would round away from zero.  */
  if (truncatedBits) {
    lost_fraction = lostFractionThroughTruncation(src, partCount(),
                                                  truncatedBits);
    if (lost_fraction != lfExactlyZero &&
        roundAwayFromZero(rounding_mode, lost_fraction, truncatedBits)) {
      if (APInt::tcIncrement(parts.data(), dstPartsCount))
        return opInvalidOp;     /* Overflow.  */
    }
  } else {
    lost_fraction = lfExactlyZero;
  }

  /* Step 3: check if we fit in the destination.  */
  unsigned int omsb = APInt::tcMSB(parts.data(), dstPartsCount) + 1;

  if (sign) {
    if (!isSigned) {
      /* Negative numbers cannot be represented as unsigned.  */
      if (omsb != 0)
        return opInvalidOp;
    } else {
      /* It takes omsb bits to represent the unsigned integer value.
         We lose a bit for the sign, but care is needed as the
         maximally negative integer is a special case.  */
      if (omsb == width &&
          APInt::tcLSB(parts.data(), dstPartsCount) + 1 != omsb)
        return opInvalidOp;

      /* This case can happen because of rounding.  */
      if (omsb > width)
        return opInvalidOp;
    }

    APInt::tcNegate (parts.data(), dstPartsCount);
  } else {
    if (omsb >= width + !isSigned)
      return opInvalidOp;
  }

  if (lost_fraction == lfExactlyZero) {
    *isExact = true;
    return opOK;
  }
  return opInexact;
}

/* Same as convertToSignExtendedInteger, except we provide
   deterministic values in case of an invalid operation exception,
   namely zero for NaNs and the minimal or maximal value respectively
   for underflow or overflow.
   The *isExact output tells whether the result is exact, in the sense
   that converting it back to the original floating point type produces
   the original value.  This is almost equivalent to result==opOK,
   except for negative zeroes.
*/
APFloat::opStatus
IEEEFloat::convertToInteger(MutableArrayRef<integerPart> parts,
                            unsigned int width, bool isSigned,
                            roundingMode rounding_mode, bool *isExact) const {
  opStatus fs;

  fs = convertToSignExtendedInteger(parts, width, isSigned, rounding_mode,
                                    isExact);

  if (fs == opInvalidOp) {
    unsigned int bits, dstPartsCount;

    dstPartsCount = partCountForBits(width);
    assert(dstPartsCount <= parts.size() && "Integer too big");

    if (category == fcNaN)
      bits = 0;
    else if (sign)
      bits = isSigned;
    else
      bits = width - isSigned;

    tcSetLeastSignificantBits(parts.data(), dstPartsCount, bits);
    if (sign && isSigned)
      APInt::tcShiftLeft(parts.data(), dstPartsCount, width - 1);
  }

  return fs;
}

/* Convert an unsigned integer SRC to a floating point number,
   rounding according to ROUNDING_MODE.  The sign of the floating
   point number is not modified.  */
APFloat::opStatus IEEEFloat::convertFromUnsignedParts(
    const integerPart *src, unsigned int srcCount, roundingMode rounding_mode) {
  unsigned int omsb, precision, dstCount;
  integerPart *dst;
  lostFraction lost_fraction;

  category = fcNormal;
  omsb = APInt::tcMSB(src, srcCount) + 1;
  dst = significandParts();
  dstCount = partCount();
  precision = semantics->precision;

  /* We want the most significant PRECISION bits of SRC.  There may not
     be that many; extract what we can.  */
  if (precision <= omsb) {
    exponent = omsb - 1;
    lost_fraction = lostFractionThroughTruncation(src, srcCount,
                                                  omsb - precision);
    APInt::tcExtract(dst, dstCount, src, precision, omsb - precision);
  } else {
    exponent = precision - 1;
    lost_fraction = lfExactlyZero;
    APInt::tcExtract(dst, dstCount, src, omsb, 0);
  }

  return normalize(rounding_mode, lost_fraction);
}

APFloat::opStatus IEEEFloat::convertFromAPInt(const APInt &Val, bool isSigned,
                                              roundingMode rounding_mode) {
  unsigned int partCount = Val.getNumWords();
  APInt api = Val;

  sign = false;
  if (isSigned && api.isNegative()) {
    sign = true;
    api = -api;
  }

  return convertFromUnsignedParts(api.getRawData(), partCount, rounding_mode);
}

Expected<APFloat::opStatus>
IEEEFloat::convertFromHexadecimalString(StringRef s,
                                        roundingMode rounding_mode) {
  lostFraction lost_fraction = lfExactlyZero;

  category = fcNormal;
  zeroSignificand();
  exponent = 0;

  integerPart *significand = significandParts();
  unsigned partsCount = partCount();
  unsigned bitPos = partsCount * integerPartWidth;
  bool computedTrailingFraction = false;

  // Skip leading zeroes and any (hexa)decimal point.
  StringRef::iterator begin = s.begin();
  StringRef::iterator end = s.end();
  StringRef::iterator dot;
  auto PtrOrErr = skipLeadingZeroesAndAnyDot(begin, end, &dot);
  if (!PtrOrErr)
    return PtrOrErr.takeError();
  StringRef::iterator p = *PtrOrErr;
  StringRef::iterator firstSignificantDigit = p;

  while (p != end) {
    integerPart hex_value;

    if (*p == '.') {
      if (dot != end)
        return createError("String contains multiple dots");
      dot = p++;
      continue;
    }

    hex_value = hexDigitValue(*p);
    if (hex_value == UINT_MAX)
      break;

    p++;

    // Store the number while we have space.
    if (bitPos) {
      bitPos -= 4;
      hex_value <<= bitPos % integerPartWidth;
      significand[bitPos / integerPartWidth] |= hex_value;
    } else if (!computedTrailingFraction) {
      auto FractOrErr = trailingHexadecimalFraction(p, end, hex_value);
      if (!FractOrErr)
        return FractOrErr.takeError();
      lost_fraction = *FractOrErr;
      computedTrailingFraction = true;
    }
  }

  /* Hex floats require an exponent but not a hexadecimal point.  */
  if (p == end)
    return createError("Hex strings require an exponent");
  if (*p != 'p' && *p != 'P')
    return createError("Invalid character in significand");
  if (p == begin)
    return createError("Significand has no digits");
  if (dot != end && p - begin == 1)
    return createError("Significand has no digits");

  /* Ignore the exponent if we are zero.  */
  if (p != firstSignificantDigit) {
    int expAdjustment;

    /* Implicit hexadecimal point?  */
    if (dot == end)
      dot = p;

    /* Calculate the exponent adjustment implicit in the number of
       significant digits.  */
    expAdjustment = static_cast<int>(dot - firstSignificantDigit);
    if (expAdjustment < 0)
      expAdjustment++;
    expAdjustment = expAdjustment * 4 - 1;

    /* Adjust for writing the significand starting at the most
       significant nibble.  */
    expAdjustment += semantics->precision;
    expAdjustment -= partsCount * integerPartWidth;

    /* Adjust for the given exponent.  */
    auto ExpOrErr = totalExponent(p + 1, end, expAdjustment);
    if (!ExpOrErr)
      return ExpOrErr.takeError();
    exponent = *ExpOrErr;
  }

  return normalize(rounding_mode, lost_fraction);
}

APFloat::opStatus
IEEEFloat::roundSignificandWithExponent(const integerPart *decSigParts,
                                        unsigned sigPartCount, int exp,
                                        roundingMode rounding_mode) {
  unsigned int parts, pow5PartCount;
  fltSemantics calcSemantics = {"calc", 32767, -32767, 0, 0};
  integerPart pow5Parts[maxPowerOfFiveParts];
  bool isNearest;

  isNearest = (rounding_mode == rmNearestTiesToEven ||
               rounding_mode == rmNearestTiesToAway);

  parts = partCountForBits(semantics->precision + 11);

  /* Calculate pow(5, abs(exp)).  */
  pow5PartCount = powerOf5(pow5Parts, exp >= 0 ? exp: -exp);

  for (;; parts *= 2) {
    opStatus sigStatus, powStatus;
    unsigned int excessPrecision, truncatedBits;

    calcSemantics.precision = parts * integerPartWidth - 1;
    excessPrecision = calcSemantics.precision - semantics->precision;
    truncatedBits = excessPrecision;

    IEEEFloat decSig(calcSemantics, uninitialized);
    decSig.makeZero(sign);
    IEEEFloat pow5(calcSemantics);

    sigStatus = decSig.convertFromUnsignedParts(decSigParts, sigPartCount,
                                                rmNearestTiesToEven);
    powStatus = pow5.convertFromUnsignedParts(pow5Parts, pow5PartCount,
                                              rmNearestTiesToEven);
    /* Add exp, as 10^n = 5^n * 2^n.  */
    decSig.exponent += exp;

    lostFraction calcLostFraction;
    integerPart HUerr, HUdistance;
    unsigned int powHUerr;

    if (exp >= 0) {
      /* multiplySignificand leaves the precision-th bit set to 1.  */
      calcLostFraction = decSig.multiplySignificand(pow5);
      powHUerr = powStatus != opOK;
    } else {
      calcLostFraction = decSig.divideSignificand(pow5);
      /* Denormal numbers have less precision.  */
      if (decSig.exponent < semantics->minExponent) {
        excessPrecision += (semantics->minExponent - decSig.exponent);
        truncatedBits = excessPrecision;
        excessPrecision = std::min(excessPrecision, calcSemantics.precision);
      }
      /* Extra half-ulp lost in reciprocal of exponent.  */
      powHUerr = (powStatus == opOK && calcLostFraction == lfExactlyZero) ? 0:2;
    }

    /* Both multiplySignificand and divideSignificand return the
       result with the integer bit set.  */
    assert(APInt::tcExtractBit
           (decSig.significandParts(), calcSemantics.precision - 1) == 1);

    HUerr = HUerrBound(calcLostFraction != lfExactlyZero, sigStatus != opOK,
                       powHUerr);
    HUdistance = 2 * ulpsFromBoundary(decSig.significandParts(),
                                      excessPrecision, isNearest);

    /* Are we guaranteed to round correctly if we truncate?  */
    if (HUdistance >= HUerr) {
      APInt::tcExtract(significandParts(), partCount(), decSig.significandParts(),
                       calcSemantics.precision - excessPrecision,
                       excessPrecision);
      /* Take the exponent of decSig.  If we tcExtract-ed less bits
         above we must adjust our exponent to compensate for the
         implicit right shift.  */
      exponent = (decSig.exponent + semantics->precision
                  - (calcSemantics.precision - excessPrecision));
      calcLostFraction = lostFractionThroughTruncation(decSig.significandParts(),
                                                       decSig.partCount(),
                                                       truncatedBits);
      return normalize(rounding_mode, calcLostFraction);
    }
  }
}

Expected<APFloat::opStatus>
IEEEFloat::convertFromDecimalString(StringRef str, roundingMode rounding_mode) {
  decimalInfo D;
  opStatus fs;

  /* Scan the text.  */
  StringRef::iterator p = str.begin();
  if (Error Err = interpretDecimal(p, str.end(), &D))
    return std::move(Err);

  /* Handle the quick cases.  First the case of no significant digits,
     i.e. zero, and then exponents that are obviously too large or too
     small.  Writing L for log 10 / log 2, a number d.ddddd*10^exp
     definitely overflows if

           (exp - 1) * L >= maxExponent

     and definitely underflows to zero where

           (exp + 1) * L <= minExponent - precision

     With integer arithmetic the tightest bounds for L are

           93/28 < L < 196/59            [ numerator <= 256 ]
           42039/12655 < L < 28738/8651  [ numerator <= 65536 ]
  */

  // Test if we have a zero number allowing for strings with no null terminators
  // and zero decimals with non-zero exponents.
  //
  // We computed firstSigDigit by ignoring all zeros and dots. Thus if
  // D->firstSigDigit equals str.end(), every digit must be a zero and there can
  // be at most one dot. On the other hand, if we have a zero with a non-zero
  // exponent, then we know that D.firstSigDigit will be non-numeric.
  if (D.firstSigDigit == str.end() || decDigitValue(*D.firstSigDigit) >= 10U) {
    category = fcZero;
    fs = opOK;
    if (semantics->nanEncoding == fltNanEncoding::NegativeZero)
      sign = false;
    if (!semantics->hasZero)
      makeSmallestNormalized(false);

    /* Check whether the normalized exponent is high enough to overflow
       max during the log-rebasing in the max-exponent check below. */
  } else if (D.normalizedExponent - 1 > INT_MAX / 42039) {
    fs = handleOverflow(rounding_mode);

  /* If it wasn't, then it also wasn't high enough to overflow max
     during the log-rebasing in the min-exponent check.  Check that it
     won't overflow min in either check, then perform the min-exponent
     check. */
  } else if (D.normalizedExponent - 1 < INT_MIN / 42039 ||
             (D.normalizedExponent + 1) * 28738 <=
               8651 * (semantics->minExponent - (int) semantics->precision)) {
    /* Underflow to zero and round.  */
    category = fcNormal;
    zeroSignificand();
    fs = normalize(rounding_mode, lfLessThanHalf);

  /* We can finally safely perform the max-exponent check. */
  } else if ((D.normalizedExponent - 1) * 42039
             >= 12655 * semantics->maxExponent) {
    /* Overflow and round.  */
    fs = handleOverflow(rounding_mode);
  } else {
    integerPart *decSignificand;
    unsigned int partCount;

    /* A tight upper bound on number of bits required to hold an
       N-digit decimal integer is N * 196 / 59.  Allocate enough space
       to hold the full significand, and an extra part required by
       tcMultiplyPart.  */
    partCount = static_cast<unsigned int>(D.lastSigDigit - D.firstSigDigit) + 1;
    partCount = partCountForBits(1 + 196 * partCount / 59);
    decSignificand = new integerPart[partCount + 1];
    partCount = 0;

    /* Convert to binary efficiently - we do almost all multiplication
       in an integerPart.  When this would overflow do we do a single
       bignum multiplication, and then revert again to multiplication
       in an integerPart.  */
    do {
      integerPart decValue, val, multiplier;

      val = 0;
      multiplier = 1;

      do {
        if (*p == '.') {
          p++;
          if (p == str.end()) {
            break;
          }
        }
        decValue = decDigitValue(*p++);
        if (decValue >= 10U) {
          delete[] decSignificand;
          return createError("Invalid character in significand");
        }
        multiplier *= 10;
        val = val * 10 + decValue;
        /* The maximum number that can be multiplied by ten with any
           digit added without overflowing an integerPart.  */
      } while (p <= D.lastSigDigit && multiplier <= (~ (integerPart) 0 - 9) / 10);

      /* Multiply out the current part.  */
      APInt::tcMultiplyPart(decSignificand, decSignificand, multiplier, val,
                            partCount, partCount + 1, false);

      /* If we used another part (likely but not guaranteed), increase
         the count.  */
      if (decSignificand[partCount])
        partCount++;
    } while (p <= D.lastSigDigit);

    category = fcNormal;
    fs = roundSignificandWithExponent(decSignificand, partCount,
                                      D.exponent, rounding_mode);

    delete [] decSignificand;
  }

  return fs;
}

bool IEEEFloat::convertFromStringSpecials(StringRef str) {
  const size_t MIN_NAME_SIZE = 3;

  if (str.size() < MIN_NAME_SIZE)
    return false;

  if (str == "inf" || str == "INFINITY" || str == "+Inf") {
    makeInf(false);
    return true;
  }

  bool IsNegative = str.consume_front("-");
  if (IsNegative) {
    if (str.size() < MIN_NAME_SIZE)
      return false;

    if (str == "inf" || str == "INFINITY" || str == "Inf") {
      makeInf(true);
      return true;
    }
  }

  // If we have a 's' (or 'S') prefix, then this is a Signaling NaN.
  bool IsSignaling = str.consume_front_insensitive("s");
  if (IsSignaling) {
    if (str.size() < MIN_NAME_SIZE)
      return false;
  }

  if (str.consume_front("nan") || str.consume_front("NaN")) {
    // A NaN without payload.
    if (str.empty()) {
      makeNaN(IsSignaling, IsNegative);
      return true;
    }

    // Allow the payload to be inside parentheses.
    if (str.front() == '(') {
      // Parentheses should be balanced (and not empty).
      if (str.size() <= 2 || str.back() != ')')
        return false;

      str = str.slice(1, str.size() - 1);
    }

    // Determine the payload number's radix.
    unsigned Radix = 10;
    if (str[0] == '0') {
      if (str.size() > 1 && tolower(str[1]) == 'x') {
        str = str.drop_front(2);
        Radix = 16;
      } else {
        Radix = 8;
      }
    }

    // Parse the payload and make the NaN.
    APInt Payload;
    if (!str.getAsInteger(Radix, Payload)) {
      makeNaN(IsSignaling, IsNegative, &Payload);
      return true;
    }
  }

  return false;
}

Expected<APFloat::opStatus>
IEEEFloat::convertFromString(StringRef str, roundingMode rounding_mode) {
  if (str.empty())
    return createError("Invalid string length");

  // Handle special cases.
  if (convertFromStringSpecials(str))
    return opOK;

  /* Handle a leading minus sign.  */
  StringRef::iterator p = str.begin();
  size_t slen = str.size();
  sign = *p == '-' ? 1 : 0;
  if (sign && !semantics->hasSignedRepr)
    llvm_unreachable(
        "This floating point format does not support signed values");

  if (*p == '-' || *p == '+') {
    p++;
    slen--;
    if (!slen)
      return createError("String has no digits");
  }

  if (slen >= 2 && p[0] == '0' && (p[1] == 'x' || p[1] == 'X')) {
    if (slen == 2)
      return createError("Invalid string");
    return convertFromHexadecimalString(StringRef(p + 2, slen - 2),
                                        rounding_mode);
  }

  return convertFromDecimalString(StringRef(p, slen), rounding_mode);
}

/* Write out a hexadecimal representation of the floating point value
   to DST, which must be of sufficient size, in the C99 form
   [-]0xh.hhhhp[+-]d.  Return the number of characters written,
   excluding the terminating NUL.

   If UPPERCASE, the output is in upper case, otherwise in lower case.

   HEXDIGITS digits appear altogether, rounding the value if
   necessary.  If HEXDIGITS is 0, the minimal precision to display the
   number precisely is used instead.  If nothing would appear after
   the decimal point it is suppressed.

   The decimal exponent is always printed and has at least one digit.
   Zero values display an exponent of zero.  Infinities and NaNs
   appear as "infinity" or "nan" respectively.

   The above rules are as specified by C99.  There is ambiguity about
   what the leading hexadecimal digit should be.  This implementation
   uses whatever is necessary so that the exponent is displayed as
   stored.  This implies the exponent will fall within the IEEE format
   range, and the leading hexadecimal digit will be 0 (for denormals),
   1 (normal numbers) or 2 (normal numbers rounded-away-from-zero with
   any other digits zero).
*/
unsigned int IEEEFloat::convertToHexString(char *dst, unsigned int hexDigits,
                                           bool upperCase,
                                           roundingMode rounding_mode) const {
  char *p;

  p = dst;
  if (sign)
    *dst++ = '-';

  switch (category) {
  case fcInfinity:
    memcpy (dst, upperCase ? infinityU: infinityL, sizeof infinityU - 1);
    dst += sizeof infinityL - 1;
    break;

  case fcNaN:
    memcpy (dst, upperCase ? NaNU: NaNL, sizeof NaNU - 1);
    dst += sizeof NaNU - 1;
    break;

  case fcZero:
    *dst++ = '0';
    *dst++ = upperCase ? 'X': 'x';
    *dst++ = '0';
    if (hexDigits > 1) {
      *dst++ = '.';
      memset (dst, '0', hexDigits - 1);
      dst += hexDigits - 1;
    }
    *dst++ = upperCase ? 'P': 'p';
    *dst++ = '0';
    break;

  case fcNormal:
    dst = convertNormalToHexString (dst, hexDigits, upperCase, rounding_mode);
    break;
  }

  *dst = 0;

  return static_cast<unsigned int>(dst - p);
}

/* Does the hard work of outputting the correctly rounded hexadecimal
   form of a normal floating point number with the specified number of
   hexadecimal digits.  If HEXDIGITS is zero the minimum number of
   digits necessary to print the value precisely is output.  */
char *IEEEFloat::convertNormalToHexString(char *dst, unsigned int hexDigits,
                                          bool upperCase,
                                          roundingMode rounding_mode) const {
  unsigned int count, valueBits, shift, partsCount, outputDigits;
  const char *hexDigitChars;
  const integerPart *significand;
  char *p;
  bool roundUp;

  *dst++ = '0';
  *dst++ = upperCase ? 'X': 'x';

  roundUp = false;
  hexDigitChars = upperCase ? hexDigitsUpper: hexDigitsLower;

  significand = significandParts();
  partsCount = partCount();

  /* +3 because the first digit only uses the single integer bit, so
     we have 3 virtual zero most-significant-bits.  */
  valueBits = semantics->precision + 3;
  shift = integerPartWidth - valueBits % integerPartWidth;

  /* The natural number of digits required ignoring trailing
     insignificant zeroes.  */
  outputDigits = (valueBits - significandLSB () + 3) / 4;

  /* hexDigits of zero means use the required number for the
     precision.  Otherwise, see if we are truncating.  If we are,
     find out if we need to round away from zero.  */
  if (hexDigits) {
    if (hexDigits < outputDigits) {
      /* We are dropping non-zero bits, so need to check how to round.
         "bits" is the number of dropped bits.  */
      unsigned int bits;
      lostFraction fraction;

      bits = valueBits - hexDigits * 4;
      fraction = lostFractionThroughTruncation (significand, partsCount, bits);
      roundUp = roundAwayFromZero(rounding_mode, fraction, bits);
    }
    outputDigits = hexDigits;
  }

  /* Write the digits consecutively, and start writing in the location
     of the hexadecimal point.  We move the most significant digit
     left and add the hexadecimal point later.  */
  p = ++dst;

  count = (valueBits + integerPartWidth - 1) / integerPartWidth;

  while (outputDigits && count) {
    integerPart part;

    /* Put the most significant integerPartWidth bits in "part".  */
    if (--count == partsCount)
      part = 0;  /* An imaginary higher zero part.  */
    else
      part = significand[count] << shift;

    if (count && shift)
      part |= significand[count - 1] >> (integerPartWidth - shift);

    /* Convert as much of "part" to hexdigits as we can.  */
    unsigned int curDigits = integerPartWidth / 4;

    curDigits = std::min(curDigits, outputDigits);
    dst += partAsHex (dst, part, curDigits, hexDigitChars);
    outputDigits -= curDigits;
  }

  if (roundUp) {
    char *q = dst;

    /* Note that hexDigitChars has a trailing '0'.  */
    do {
      q--;
      *q = hexDigitChars[hexDigitValue (*q) + 1];
    } while (*q == '0');
    assert(q >= p);
  } else {
    /* Add trailing zeroes.  */
    memset (dst, '0', outputDigits);
    dst += outputDigits;
  }

  /* Move the most significant digit to before the point, and if there
     is something after the decimal point add it.  This must come
     after rounding above.  */
  p[-1] = p[0];
  if (dst -1 == p)
    dst--;
  else
    p[0] = '.';

  /* Finally output the exponent.  */
  *dst++ = upperCase ? 'P': 'p';

  return writeSignedDecimal (dst, exponent);
}

hash_code hash_value(const IEEEFloat &Arg) {
  if (!Arg.isFiniteNonZero())
    return hash_combine((uint8_t)Arg.category,
                        // NaN has no sign, fix it at zero.
                        Arg.isNaN() ? (uint8_t)0 : (uint8_t)Arg.sign,
                        Arg.semantics->precision);

  // Normal floats need their exponent and significand hashed.
  return hash_combine((uint8_t)Arg.category, (uint8_t)Arg.sign,
                      Arg.semantics->precision, Arg.exponent,
                      hash_combine_range(
                        Arg.significandParts(),
                        Arg.significandParts() + Arg.partCount()));
}

// Conversion from APFloat to/from host float/double.  It may eventually be
// possible to eliminate these and have everybody deal with APFloats, but that
// will take a while.  This approach will not easily extend to long double.
// Current implementation requires integerPartWidth==64, which is correct at
// the moment but could be made more general.

// Denormals have exponent minExponent in APFloat, but minExponent-1 in
// the actual IEEE respresentations.  We compensate for that here.

APInt IEEEFloat::convertF80LongDoubleAPFloatToAPInt() const {
  assert(semantics ==
         (const llvm::fltSemantics *)&APFloatBase::semX87DoubleExtended);
  assert(partCount()==2);

  uint64_t myexponent, mysignificand;

  if (isFiniteNonZero()) {
    myexponent = exponent+16383; //bias
    mysignificand = significandParts()[0];
    if (myexponent==1 && !(mysignificand & 0x8000000000000000ULL))
      myexponent = 0;   // denormal
  } else if (category==fcZero) {
    myexponent = 0;
    mysignificand = 0;
  } else if (category==fcInfinity) {
    myexponent = 0x7fff;
    mysignificand = 0x8000000000000000ULL;
  } else {
    assert(category == fcNaN && "Unknown category");
    myexponent = 0x7fff;
    mysignificand = significandParts()[0];
  }

  uint64_t words[2];
  words[0] = mysignificand;
  words[1] =  ((uint64_t)(sign & 1) << 15) |
              (myexponent & 0x7fffLL);
  return APInt(80, words);
}

APInt IEEEFloat::convertPPCDoubleDoubleLegacyAPFloatToAPInt() const {
  assert(semantics ==
         (const llvm::fltSemantics *)&APFloatBase::semPPCDoubleDoubleLegacy);
  assert(partCount()==2);

  uint64_t words[2];
  opStatus fs;
  bool losesInfo;

  // Convert number to double.  To avoid spurious underflows, we re-
  // normalize against the "double" minExponent first, and only *then*
  // truncate the mantissa.  The result of that second conversion
  // may be inexact, but should never underflow.
  // Declare fltSemantics before APFloat that uses it (and
  // saves pointer to it) to ensure correct destruction order.
  fltSemantics extendedSemantics = *semantics;
  extendedSemantics.minExponent = APFloatBase::semIEEEdouble.minExponent;
  IEEEFloat extended(*this);
  fs = extended.convert(extendedSemantics, rmNearestTiesToEven, &losesInfo);
  assert(fs == opOK && !losesInfo);
  (void)fs;

  IEEEFloat u(extended);
  fs = u.convert(APFloatBase::semIEEEdouble, rmNearestTiesToEven, &losesInfo);
  assert(fs == opOK || fs == opInexact);
  (void)fs;
  words[0] = *u.convertDoubleAPFloatToAPInt().getRawData();

  // If conversion was exact or resulted in a special case, we're done;
  // just set the second double to zero.  Otherwise, re-convert back to
  // the extended format and compute the difference.  This now should
  // convert exactly to double.
  if (u.isFiniteNonZero() && losesInfo) {
    fs = u.convert(extendedSemantics, rmNearestTiesToEven, &losesInfo);
    assert(fs == opOK && !losesInfo);
    (void)fs;

    IEEEFloat v(extended);
    v.subtract(u, rmNearestTiesToEven);
    fs = v.convert(APFloatBase::semIEEEdouble, rmNearestTiesToEven, &losesInfo);
    assert(fs == opOK && !losesInfo);
    (void)fs;
    words[1] = *v.convertDoubleAPFloatToAPInt().getRawData();
  } else {
    words[1] = 0;
  }

  return APInt(128, words);
}

template <const fltSemantics &S>
APInt IEEEFloat::convertIEEEFloatToAPInt() const {
  assert(semantics == &S);
  const int bias = (semantics == &APFloatBase::semFloat8E8M0FNU)
                       ? -S.minExponent
                       : -(S.minExponent - 1);
  constexpr unsigned int trailing_significand_bits = S.precision - 1;
  constexpr int integer_bit_part = trailing_significand_bits / integerPartWidth;
  constexpr integerPart integer_bit =
      integerPart{1} << (trailing_significand_bits % integerPartWidth);
  constexpr uint64_t significand_mask = integer_bit - 1;
  constexpr unsigned int exponent_bits =
      trailing_significand_bits ? (S.sizeInBits - 1 - trailing_significand_bits)
                                : S.sizeInBits;
  static_assert(exponent_bits < 64);
  constexpr uint64_t exponent_mask = (uint64_t{1} << exponent_bits) - 1;

  uint64_t myexponent;
  std::array<integerPart, partCountForBits(trailing_significand_bits)>
      mysignificand;

  if (isFiniteNonZero()) {
    myexponent = exponent + bias;
    std::copy_n(significandParts(), mysignificand.size(),
                mysignificand.begin());
    if (myexponent == 1 &&
        !(significandParts()[integer_bit_part] & integer_bit))
      myexponent = 0; // denormal
  } else if (category == fcZero) {
    if (!S.hasZero)
      llvm_unreachable("semantics does not support zero!");
    myexponent = ::exponentZero(S) + bias;
    mysignificand.fill(0);
  } else if (category == fcInfinity) {
    if (S.nonFiniteBehavior == fltNonfiniteBehavior::NanOnly ||
        S.nonFiniteBehavior == fltNonfiniteBehavior::FiniteOnly)
      llvm_unreachable("semantics don't support inf!");
    myexponent = ::exponentInf(S) + bias;
    mysignificand.fill(0);
  } else {
    assert(category == fcNaN && "Unknown category!");
    if (S.nonFiniteBehavior == fltNonfiniteBehavior::FiniteOnly)
      llvm_unreachable("semantics don't support NaN!");
    myexponent = ::exponentNaN(S) + bias;
    std::copy_n(significandParts(), mysignificand.size(),
                mysignificand.begin());
  }
  std::array<uint64_t, (S.sizeInBits + 63) / 64> words;
  auto words_iter =
      std::copy_n(mysignificand.begin(), mysignificand.size(), words.begin());
  if constexpr (significand_mask != 0 || trailing_significand_bits == 0) {
    // Clear the integer bit.
    words[mysignificand.size() - 1] &= significand_mask;
  }
  std::fill(words_iter, words.end(), uint64_t{0});
  constexpr size_t last_word = words.size() - 1;
  uint64_t shifted_sign = static_cast<uint64_t>(sign & 1)
                          << ((S.sizeInBits - 1) % 64);
  words[last_word] |= shifted_sign;
  uint64_t shifted_exponent = (myexponent & exponent_mask)
                              << (trailing_significand_bits % 64);
  words[last_word] |= shifted_exponent;
  if constexpr (last_word == 0) {
    return APInt(S.sizeInBits, words[0]);
  }
  return APInt(S.sizeInBits, words);
}

APInt IEEEFloat::convertQuadrupleAPFloatToAPInt() const {
  assert(partCount() == 2);
  return convertIEEEFloatToAPInt<APFloatBase::semIEEEquad>();
}

APInt IEEEFloat::convertDoubleAPFloatToAPInt() const {
  assert(partCount()==1);
  return convertIEEEFloatToAPInt<APFloatBase::semIEEEdouble>();
}

APInt IEEEFloat::convertFloatAPFloatToAPInt() const {
  assert(partCount()==1);
  return convertIEEEFloatToAPInt<APFloatBase::semIEEEsingle>();
}

APInt IEEEFloat::convertBFloatAPFloatToAPInt() const {
  assert(partCount() == 1);
  return convertIEEEFloatToAPInt<APFloatBase::semBFloat>();
}

APInt IEEEFloat::convertHalfAPFloatToAPInt() const {
  assert(partCount()==1);
  return convertIEEEFloatToAPInt<APFloatBase::APFloatBase::semIEEEhalf>();
}

APInt IEEEFloat::convertFloat8E5M2APFloatToAPInt() const {
  assert(partCount() == 1);
  return convertIEEEFloatToAPInt<APFloatBase::semFloat8E5M2>();
}

APInt IEEEFloat::convertFloat8E5M2FNUZAPFloatToAPInt() const {
  assert(partCount() == 1);
  return convertIEEEFloatToAPInt<APFloatBase::semFloat8E5M2FNUZ>();
}

APInt IEEEFloat::convertFloat8E4M3APFloatToAPInt() const {
  assert(partCount() == 1);
  return convertIEEEFloatToAPInt<APFloatBase::semFloat8E4M3>();
}

APInt IEEEFloat::convertFloat8E4M3FNAPFloatToAPInt() const {
  assert(partCount() == 1);
  return convertIEEEFloatToAPInt<APFloatBase::semFloat8E4M3FN>();
}

APInt IEEEFloat::convertFloat8E4M3FNUZAPFloatToAPInt() const {
  assert(partCount() == 1);
  return convertIEEEFloatToAPInt<APFloatBase::semFloat8E4M3FNUZ>();
}

APInt IEEEFloat::convertFloat8E4M3B11FNUZAPFloatToAPInt() const {
  assert(partCount() == 1);
  return convertIEEEFloatToAPInt<APFloatBase::semFloat8E4M3B11FNUZ>();
}

APInt IEEEFloat::convertFloat8E3M4APFloatToAPInt() const {
  assert(partCount() == 1);
  return convertIEEEFloatToAPInt<APFloatBase::semFloat8E3M4>();
}

APInt IEEEFloat::convertFloatTF32APFloatToAPInt() const {
  assert(partCount() == 1);
  return convertIEEEFloatToAPInt<APFloatBase::semFloatTF32>();
}

APInt IEEEFloat::convertFloat8E8M0FNUAPFloatToAPInt() const {
  assert(partCount() == 1);
  return convertIEEEFloatToAPInt<APFloatBase::semFloat8E8M0FNU>();
}

APInt IEEEFloat::convertFloat6E3M2FNAPFloatToAPInt() const {
  assert(partCount() == 1);
  return convertIEEEFloatToAPInt<APFloatBase::semFloat6E3M2FN>();
}

APInt IEEEFloat::convertFloat6E2M3FNAPFloatToAPInt() const {
  assert(partCount() == 1);
  return convertIEEEFloatToAPInt<APFloatBase::semFloat6E2M3FN>();
}

APInt IEEEFloat::convertFloat4E2M1FNAPFloatToAPInt() const {
  assert(partCount() == 1);
  return convertIEEEFloatToAPInt<APFloatBase::semFloat4E2M1FN>();
}

// This function creates an APInt that is just a bit map of the floating
// point constant as it would appear in memory.  It is not a conversion,
// and treating the result as a normal integer is unlikely to be useful.

APInt IEEEFloat::bitcastToAPInt() const {
  if (semantics == (const llvm::fltSemantics *)&APFloatBase::semIEEEhalf)
    return convertHalfAPFloatToAPInt();

  if (semantics == (const llvm::fltSemantics *)&APFloatBase::semBFloat)
    return convertBFloatAPFloatToAPInt();

  if (semantics == (const llvm::fltSemantics *)&APFloatBase::semIEEEsingle)
    return convertFloatAPFloatToAPInt();

  if (semantics == (const llvm::fltSemantics *)&APFloatBase::semIEEEdouble)
    return convertDoubleAPFloatToAPInt();

  if (semantics == (const llvm::fltSemantics *)&APFloatBase::semIEEEquad)
    return convertQuadrupleAPFloatToAPInt();

  if (semantics ==
      (const llvm::fltSemantics *)&APFloatBase::semPPCDoubleDoubleLegacy)
    return convertPPCDoubleDoubleLegacyAPFloatToAPInt();

  if (semantics == (const llvm::fltSemantics *)&APFloatBase::semFloat8E5M2)
    return convertFloat8E5M2APFloatToAPInt();

  if (semantics == (const llvm::fltSemantics *)&APFloatBase::semFloat8E5M2FNUZ)
    return convertFloat8E5M2FNUZAPFloatToAPInt();

  if (semantics == (const llvm::fltSemantics *)&APFloatBase::semFloat8E4M3)
    return convertFloat8E4M3APFloatToAPInt();

  if (semantics == (const llvm::fltSemantics *)&APFloatBase::semFloat8E4M3FN)
    return convertFloat8E4M3FNAPFloatToAPInt();

  if (semantics == (const llvm::fltSemantics *)&APFloatBase::semFloat8E4M3FNUZ)
    return convertFloat8E4M3FNUZAPFloatToAPInt();

  if (semantics ==
      (const llvm::fltSemantics *)&APFloatBase::semFloat8E4M3B11FNUZ)
    return convertFloat8E4M3B11FNUZAPFloatToAPInt();

  if (semantics == (const llvm::fltSemantics *)&APFloatBase::semFloat8E3M4)
    return convertFloat8E3M4APFloatToAPInt();

  if (semantics == (const llvm::fltSemantics *)&APFloatBase::semFloatTF32)
    return convertFloatTF32APFloatToAPInt();

  if (semantics == (const llvm::fltSemantics *)&APFloatBase::semFloat8E8M0FNU)
    return convertFloat8E8M0FNUAPFloatToAPInt();

  if (semantics == (const llvm::fltSemantics *)&APFloatBase::semFloat6E3M2FN)
    return convertFloat6E3M2FNAPFloatToAPInt();

  if (semantics == (const llvm::fltSemantics *)&APFloatBase::semFloat6E2M3FN)
    return convertFloat6E2M3FNAPFloatToAPInt();

  if (semantics == (const llvm::fltSemantics *)&APFloatBase::semFloat4E2M1FN)
    return convertFloat4E2M1FNAPFloatToAPInt();

  assert(semantics ==
             (const llvm::fltSemantics *)&APFloatBase::semX87DoubleExtended &&
         "unknown format!");
  return convertF80LongDoubleAPFloatToAPInt();
}

float IEEEFloat::convertToFloat() const {
  assert(semantics == (const llvm::fltSemantics *)&APFloatBase::semIEEEsingle &&
         "Float semantics are not IEEEsingle");
  APInt api = bitcastToAPInt();
  return api.bitsToFloat();
}

double IEEEFloat::convertToDouble() const {
  assert(semantics == (const llvm::fltSemantics *)&APFloatBase::semIEEEdouble &&
         "Float semantics are not IEEEdouble");
  APInt api = bitcastToAPInt();
  return api.bitsToDouble();
}

#ifdef HAS_IEE754_FLOAT128
float128 IEEEFloat::convertToQuad() const {
  assert(semantics == (const llvm::fltSemantics *)&APFloatBase::semIEEEquad &&
         "Float semantics are not IEEEquads");
  APInt api = bitcastToAPInt();
  return api.bitsToQuad();
}
#endif

/// Integer bit is explicit in this format.  Intel hardware (387 and later)
/// does not support these bit patterns:
///  exponent = all 1's, integer bit 0, significand 0 ("pseudoinfinity")
///  exponent = all 1's, integer bit 0, significand nonzero ("pseudoNaN")
///  exponent!=0 nor all 1's, integer bit 0 ("unnormal")
///  exponent = 0, integer bit 1 ("pseudodenormal")
/// At the moment, the first three are treated as NaNs, the last one as Normal.
void IEEEFloat::initFromF80LongDoubleAPInt(const APInt &api) {
  uint64_t i1 = api.getRawData()[0];
  uint64_t i2 = api.getRawData()[1];
  uint64_t myexponent = (i2 & 0x7fff);
  uint64_t mysignificand = i1;
  uint8_t myintegerbit = mysignificand >> 63;

  initialize(&APFloatBase::semX87DoubleExtended);
  assert(partCount()==2);

  sign = static_cast<unsigned int>(i2>>15);
  if (myexponent == 0 && mysignificand == 0) {
    makeZero(sign);
  } else if (myexponent==0x7fff && mysignificand==0x8000000000000000ULL) {
    makeInf(sign);
  } else if ((myexponent == 0x7fff && mysignificand != 0x8000000000000000ULL) ||
             (myexponent != 0x7fff && myexponent != 0 && myintegerbit == 0)) {
    category = fcNaN;
    exponent = exponentNaN();
    significandParts()[0] = mysignificand;
    significandParts()[1] = 0;
  } else {
    category = fcNormal;
    exponent = myexponent - 16383;
    significandParts()[0] = mysignificand;
    significandParts()[1] = 0;
    if (myexponent==0)          // denormal
      exponent = -16382;
  }
}

void IEEEFloat::initFromPPCDoubleDoubleLegacyAPInt(const APInt &api) {
  uint64_t i1 = api.getRawData()[0];
  uint64_t i2 = api.getRawData()[1];
  opStatus fs;
  bool losesInfo;

  // Get the first double and convert to our format.
  initFromDoubleAPInt(APInt(64, i1));
  fs = convert(APFloatBase::semPPCDoubleDoubleLegacy, rmNearestTiesToEven,
               &losesInfo);
  assert(fs == opOK && !losesInfo);
  (void)fs;

  // Unless we have a special case, add in second double.
  if (isFiniteNonZero()) {
    IEEEFloat v(APFloatBase::semIEEEdouble, APInt(64, i2));
    fs = v.convert(APFloatBase::semPPCDoubleDoubleLegacy, rmNearestTiesToEven,
                   &losesInfo);
    assert(fs == opOK && !losesInfo);
    (void)fs;

    add(v, rmNearestTiesToEven);
  }
}

// The E8M0 format has the following characteristics:
// It is an 8-bit unsigned format with only exponents (no actual significand).
// No encodings for {zero, infinities or denorms}.
// NaN is represented by all 1's.
// Bias is 127.
void IEEEFloat::initFromFloat8E8M0FNUAPInt(const APInt &api) {
  const uint64_t exponent_mask = 0xff;
  uint64_t val = api.getRawData()[0];
  uint64_t myexponent = (val & exponent_mask);

  initialize(&APFloatBase::semFloat8E8M0FNU);
  assert(partCount() == 1);

  // This format has unsigned representation only
  sign = 0;

  // Set the significand
  // This format does not have any significand but the 'Pth' precision bit is
  // always set to 1 for consistency in APFloat's internal representation.
  uint64_t mysignificand = 1;
  significandParts()[0] = mysignificand;

  // This format can either have a NaN or fcNormal
  // All 1's i.e. 255 is a NaN
  if (val == exponent_mask) {
    category = fcNaN;
    exponent = exponentNaN();
    return;
  }
  // Handle fcNormal...
  category = fcNormal;
  exponent = myexponent - 127; // 127 is bias
}
template <const fltSemantics &S>
void IEEEFloat::initFromIEEEAPInt(const APInt &api) {
  assert(api.getBitWidth() == S.sizeInBits);
  constexpr integerPart integer_bit = integerPart{1}
                                      << ((S.precision - 1) % integerPartWidth);
  constexpr uint64_t significand_mask = integer_bit - 1;
  constexpr unsigned int trailing_significand_bits = S.precision - 1;
  constexpr unsigned int stored_significand_parts =
      partCountForBits(trailing_significand_bits);
  constexpr unsigned int exponent_bits =
      S.sizeInBits - 1 - trailing_significand_bits;
  static_assert(exponent_bits < 64);
  constexpr uint64_t exponent_mask = (uint64_t{1} << exponent_bits) - 1;
  constexpr int bias = -(S.minExponent - 1);

  // Copy the bits of the significand. We need to clear out the exponent and
  // sign bit in the last word.
  std::array<integerPart, stored_significand_parts> mysignificand;
  std::copy_n(api.getRawData(), mysignificand.size(), mysignificand.begin());
  if constexpr (significand_mask != 0) {
    mysignificand[mysignificand.size() - 1] &= significand_mask;
  }

  // We assume the last word holds the sign bit, the exponent, and potentially
  // some of the trailing significand field.
  uint64_t last_word = api.getRawData()[api.getNumWords() - 1];
  uint64_t myexponent =
      (last_word >> (trailing_significand_bits % 64)) & exponent_mask;

  initialize(&S);
  assert(partCount() == mysignificand.size());

  sign = static_cast<unsigned int>(last_word >> ((S.sizeInBits - 1) % 64));

  bool all_zero_significand = llvm::all_of(mysignificand, equal_to(0));

  bool is_zero = myexponent == 0 && all_zero_significand;

  if constexpr (S.nonFiniteBehavior == fltNonfiniteBehavior::IEEE754) {
    if (myexponent - bias == ::exponentInf(S) && all_zero_significand) {
      makeInf(sign);
      return;
    }
  }

  bool is_nan = false;

  if constexpr (S.nanEncoding == fltNanEncoding::IEEE) {
    is_nan = myexponent - bias == ::exponentNaN(S) && !all_zero_significand;
  } else if constexpr (S.nanEncoding == fltNanEncoding::AllOnes) {
    bool all_ones_significand =
        std::all_of(mysignificand.begin(), mysignificand.end() - 1,
                    [](integerPart bits) { return bits == ~integerPart{0}; }) &&
        (!significand_mask ||
         mysignificand[mysignificand.size() - 1] == significand_mask);
    is_nan = myexponent - bias == ::exponentNaN(S) && all_ones_significand;
  } else if constexpr (S.nanEncoding == fltNanEncoding::NegativeZero) {
    is_nan = is_zero && sign;
  }

  if (is_nan) {
    category = fcNaN;
    exponent = ::exponentNaN(S);
    std::copy_n(mysignificand.begin(), mysignificand.size(),
                significandParts());
    return;
  }

  if (is_zero) {
    makeZero(sign);
    return;
  }

  category = fcNormal;
  exponent = myexponent - bias;
  std::copy_n(mysignificand.begin(), mysignificand.size(), significandParts());
  if (myexponent == 0) // denormal
    exponent = S.minExponent;
  else
    significandParts()[mysignificand.size()-1] |= integer_bit; // integer bit
}

void IEEEFloat::initFromQuadrupleAPInt(const APInt &api) {
  initFromIEEEAPInt<APFloatBase::semIEEEquad>(api);
}

void IEEEFloat::initFromDoubleAPInt(const APInt &api) {
  initFromIEEEAPInt<APFloatBase::semIEEEdouble>(api);
}

void IEEEFloat::initFromFloatAPInt(const APInt &api) {
  initFromIEEEAPInt<APFloatBase::semIEEEsingle>(api);
}

void IEEEFloat::initFromBFloatAPInt(const APInt &api) {
  initFromIEEEAPInt<APFloatBase::semBFloat>(api);
}

void IEEEFloat::initFromHalfAPInt(const APInt &api) {
  initFromIEEEAPInt<APFloatBase::semIEEEhalf>(api);
}

void IEEEFloat::initFromFloat8E5M2APInt(const APInt &api) {
  initFromIEEEAPInt<APFloatBase::semFloat8E5M2>(api);
}

void IEEEFloat::initFromFloat8E5M2FNUZAPInt(const APInt &api) {
  initFromIEEEAPInt<APFloatBase::semFloat8E5M2FNUZ>(api);
}

void IEEEFloat::initFromFloat8E4M3APInt(const APInt &api) {
  initFromIEEEAPInt<APFloatBase::semFloat8E4M3>(api);
}

void IEEEFloat::initFromFloat8E4M3FNAPInt(const APInt &api) {
  initFromIEEEAPInt<APFloatBase::semFloat8E4M3FN>(api);
}

void IEEEFloat::initFromFloat8E4M3FNUZAPInt(const APInt &api) {
  initFromIEEEAPInt<APFloatBase::semFloat8E4M3FNUZ>(api);
}

void IEEEFloat::initFromFloat8E4M3B11FNUZAPInt(const APInt &api) {
  initFromIEEEAPInt<APFloatBase::semFloat8E4M3B11FNUZ>(api);
}

void IEEEFloat::initFromFloat8E3M4APInt(const APInt &api) {
  initFromIEEEAPInt<APFloatBase::semFloat8E3M4>(api);
}

void IEEEFloat::initFromFloatTF32APInt(const APInt &api) {
  initFromIEEEAPInt<APFloatBase::semFloatTF32>(api);
}

void IEEEFloat::initFromFloat6E3M2FNAPInt(const APInt &api) {
  initFromIEEEAPInt<APFloatBase::semFloat6E3M2FN>(api);
}

void IEEEFloat::initFromFloat6E2M3FNAPInt(const APInt &api) {
  initFromIEEEAPInt<APFloatBase::semFloat6E2M3FN>(api);
}

void IEEEFloat::initFromFloat4E2M1FNAPInt(const APInt &api) {
  initFromIEEEAPInt<APFloatBase::semFloat4E2M1FN>(api);
}

/// Treat api as containing the bits of a floating point number.
void IEEEFloat::initFromAPInt(const fltSemantics *Sem, const APInt &api) {
  assert(api.getBitWidth() == Sem->sizeInBits);
  if (Sem == &APFloatBase::semIEEEhalf)
    return initFromHalfAPInt(api);
  if (Sem == &APFloatBase::semBFloat)
    return initFromBFloatAPInt(api);
  if (Sem == &APFloatBase::semIEEEsingle)
    return initFromFloatAPInt(api);
  if (Sem == &APFloatBase::semIEEEdouble)
    return initFromDoubleAPInt(api);
  if (Sem == &APFloatBase::semX87DoubleExtended)
    return initFromF80LongDoubleAPInt(api);
  if (Sem == &APFloatBase::semIEEEquad)
    return initFromQuadrupleAPInt(api);
  if (Sem == &APFloatBase::semPPCDoubleDoubleLegacy)
    return initFromPPCDoubleDoubleLegacyAPInt(api);
  if (Sem == &APFloatBase::semFloat8E5M2)
    return initFromFloat8E5M2APInt(api);
  if (Sem == &APFloatBase::semFloat8E5M2FNUZ)
    return initFromFloat8E5M2FNUZAPInt(api);
  if (Sem == &APFloatBase::semFloat8E4M3)
    return initFromFloat8E4M3APInt(api);
  if (Sem == &APFloatBase::semFloat8E4M3FN)
    return initFromFloat8E4M3FNAPInt(api);
  if (Sem == &APFloatBase::semFloat8E4M3FNUZ)
    return initFromFloat8E4M3FNUZAPInt(api);
  if (Sem == &APFloatBase::semFloat8E4M3B11FNUZ)
    return initFromFloat8E4M3B11FNUZAPInt(api);
  if (Sem == &APFloatBase::semFloat8E3M4)
    return initFromFloat8E3M4APInt(api);
  if (Sem == &APFloatBase::semFloatTF32)
    return initFromFloatTF32APInt(api);
  if (Sem == &APFloatBase::semFloat8E8M0FNU)
    return initFromFloat8E8M0FNUAPInt(api);
  if (Sem == &APFloatBase::semFloat6E3M2FN)
    return initFromFloat6E3M2FNAPInt(api);
  if (Sem == &APFloatBase::semFloat6E2M3FN)
    return initFromFloat6E2M3FNAPInt(api);
  if (Sem == &APFloatBase::semFloat4E2M1FN)
    return initFromFloat4E2M1FNAPInt(api);

  llvm_unreachable("unsupported semantics");
}

/// Make this number the largest magnitude normal number in the given
/// semantics.
void IEEEFloat::makeLargest(bool Negative) {
  if (Negative && !semantics->hasSignedRepr)
    llvm_unreachable(
        "This floating point format does not support signed values");
  // We want (in interchange format):
  //   sign = {Negative}
  //   exponent = 1..10
  //   significand = 1..1
  category = fcNormal;
  sign = Negative;
  exponent = semantics->maxExponent;

  // Use memset to set all but the highest integerPart to all ones.
  integerPart *significand = significandParts();
  unsigned PartCount = partCount();
  memset(significand, 0xFF, sizeof(integerPart)*(PartCount - 1));

  // Set the high integerPart especially setting all unused top bits for
  // internal consistency.
  const unsigned NumUnusedHighBits =
    PartCount*integerPartWidth - semantics->precision;
  significand[PartCount - 1] = (NumUnusedHighBits < integerPartWidth)
                                   ? (~integerPart(0) >> NumUnusedHighBits)
                                   : 0;
  if (semantics->nonFiniteBehavior == fltNonfiniteBehavior::NanOnly &&
      semantics->nanEncoding == fltNanEncoding::AllOnes &&
      (semantics->precision > 1))
    significand[0] &= ~integerPart(1);
}

/// Make this number the smallest magnitude denormal number in the given
/// semantics.
void IEEEFloat::makeSmallest(bool Negative) {
  if (Negative && !semantics->hasSignedRepr)
    llvm_unreachable(
        "This floating point format does not support signed values");
  // We want (in interchange format):
  //   sign = {Negative}
  //   exponent = 0..0
  //   significand = 0..01
  category = fcNormal;
  sign = Negative;
  exponent = semantics->minExponent;
  APInt::tcSet(significandParts(), 1, partCount());
}

void IEEEFloat::makeSmallestNormalized(bool Negative) {
  if (Negative && !semantics->hasSignedRepr)
    llvm_unreachable(
        "This floating point format does not support signed values");
  // We want (in interchange format):
  //   sign = {Negative}
  //   exponent = 0..0
  //   significand = 10..0

  category = fcNormal;
  zeroSignificand();
  sign = Negative;
  exponent = semantics->minExponent;
  APInt::tcSetBit(significandParts(), semantics->precision - 1);
}

IEEEFloat::IEEEFloat(const fltSemantics &Sem, const APInt &API) {
  initFromAPInt(&Sem, API);
}

IEEEFloat::IEEEFloat(float f) {
  initFromAPInt(&APFloatBase::semIEEEsingle, APInt::floatToBits(f));
}

IEEEFloat::IEEEFloat(double d) {
  initFromAPInt(&APFloatBase::semIEEEdouble, APInt::doubleToBits(d));
}

namespace {
  void append(SmallVectorImpl<char> &Buffer, StringRef Str) {
    Buffer.append(Str.begin(), Str.end());
  }

  /// Removes data from the given significand until it is no more
  /// precise than is required for the desired precision.
  void AdjustToPrecision(APInt &significand,
                         int &exp, unsigned FormatPrecision) {
    unsigned bits = significand.getActiveBits();

    // 196/59 is a very slight overestimate of lg_2(10).
    unsigned bitsRequired = (FormatPrecision * 196 + 58) / 59;

    if (bits <= bitsRequired) return;

    unsigned tensRemovable = (bits - bitsRequired) * 59 / 196;
    if (!tensRemovable) return;

    exp += tensRemovable;

    APInt divisor(significand.getBitWidth(), 1);
    APInt powten(significand.getBitWidth(), 10);
    while (true) {
      if (tensRemovable & 1)
        divisor *= powten;
      tensRemovable >>= 1;
      if (!tensRemovable) break;
      powten *= powten;
    }

    significand = significand.udiv(divisor);

    // Truncate the significand down to its active bit count.
    significand = significand.trunc(significand.getActiveBits());
  }


  void AdjustToPrecision(SmallVectorImpl<char> &buffer,
                         int &exp, unsigned FormatPrecision) {
    unsigned N = buffer.size();
    if (N <= FormatPrecision) return;

    // The most significant figures are the last ones in the buffer.
    unsigned FirstSignificant = N - FormatPrecision;

    // Round.
    // FIXME: this probably shouldn't use 'round half up'.

    // Rounding down is just a truncation, except we also want to drop
    // trailing zeros from the new result.
    if (buffer[FirstSignificant - 1] < '5') {
      while (FirstSignificant < N && buffer[FirstSignificant] == '0')
        FirstSignificant++;

      exp += FirstSignificant;
      buffer.erase(&buffer[0], &buffer[FirstSignificant]);
      return;
    }

    // Rounding up requires a decimal add-with-carry.  If we continue
    // the carry, the newly-introduced zeros will just be truncated.
    for (unsigned I = FirstSignificant; I != N; ++I) {
      if (buffer[I] == '9') {
        FirstSignificant++;
      } else {
        buffer[I]++;
        break;
      }
    }

    // If we carried through, we have exactly one digit of precision.
    if (FirstSignificant == N) {
      exp += FirstSignificant;
      buffer.clear();
      buffer.push_back('1');
      return;
    }

    exp += FirstSignificant;
    buffer.erase(&buffer[0], &buffer[FirstSignificant]);
  }

  void toStringImpl(SmallVectorImpl<char> &Str, const bool isNeg, int exp,
                    APInt significand, unsigned FormatPrecision,
                    unsigned FormatMaxPadding, bool TruncateZero) {
    const int semanticsPrecision = significand.getBitWidth();

    if (isNeg)
      Str.push_back('-');

    // Set FormatPrecision if zero.  We want to do this before we
    // truncate trailing zeros, as those are part of the precision.
    if (!FormatPrecision) {
      // We use enough digits so the number can be round-tripped back to an
      // APFloat. The formula comes from "How to Print Floating-Point Numbers
      // Accurately" by Steele and White.
      // FIXME: Using a formula based purely on the precision is conservative;
      // we can print fewer digits depending on the actual value being printed.

      // FormatPrecision = 2 + floor(significandBits / lg_2(10))
      FormatPrecision = 2 + semanticsPrecision * 59 / 196;
    }

    // Ignore trailing binary zeros.
    int trailingZeros = significand.countr_zero();
    exp += trailingZeros;
    significand.lshrInPlace(trailingZeros);

    // Change the exponent from 2^e to 10^e.
    if (exp == 0) {
      // Nothing to do.
    } else if (exp > 0) {
      // Just shift left.
      significand = significand.zext(semanticsPrecision + exp);
      significand <<= exp;
      exp = 0;
    } else { /* exp < 0 */
      int texp = -exp;

      // We transform this using the identity:
      //   (N)(2^-e) == (N)(5^e)(10^-e)
      // This means we have to multiply N (the significand) by 5^e.
      // To avoid overflow, we have to operate on numbers large
      // enough to store N * 5^e:
      //   log2(N * 5^e) == log2(N) + e * log2(5)
      //                 <= semantics->precision + e * 137 / 59
      //   (log_2(5) ~ 2.321928 < 2.322034 ~ 137/59)

      unsigned precision = semanticsPrecision + (137 * texp + 136) / 59;

      // Multiply significand by 5^e.
      //   N * 5^0101 == N * 5^(1*1) * 5^(0*2) * 5^(1*4) * 5^(0*8)
      significand = significand.zext(precision);
      APInt five_to_the_i(precision, 5);
      while (true) {
        if (texp & 1)
          significand *= five_to_the_i;

        texp >>= 1;
        if (!texp)
          break;
        five_to_the_i *= five_to_the_i;
      }
    }

    AdjustToPrecision(significand, exp, FormatPrecision);

    SmallVector<char, 256> buffer;

    // Fill the buffer.
    unsigned precision = significand.getBitWidth();
    if (precision < 4) {
      // We need enough precision to store the value 10.
      precision = 4;
      significand = significand.zext(precision);
    }
    APInt ten(precision, 10);
    APInt digit(precision, 0);

    bool inTrail = true;
    while (significand != 0) {
      // digit <- significand % 10
      // significand <- significand / 10
      APInt::udivrem(significand, ten, significand, digit);

      unsigned d = digit.getZExtValue();

      // Drop trailing zeros.
      if (inTrail && !d)
        exp++;
      else {
        buffer.push_back((char) ('0' + d));
        inTrail = false;
      }
    }

    assert(!buffer.empty() && "no characters in buffer!");

    // Drop down to FormatPrecision.
    // TODO: don't do more precise calculations above than are required.
    AdjustToPrecision(buffer, exp, FormatPrecision);

    unsigned NDigits = buffer.size();

    // Check whether we should use scientific notation.
    bool FormatScientific;
    if (!FormatMaxPadding)
      FormatScientific = true;
    else {
      if (exp >= 0) {
        // 765e3 --> 765000
        //              ^^^
        // But we shouldn't make the number look more precise than it is.
        FormatScientific = ((unsigned) exp > FormatMaxPadding ||
                            NDigits + (unsigned) exp > FormatPrecision);
      } else {
        // Power of the most significant digit.
        int MSD = exp + (int) (NDigits - 1);
        if (MSD >= 0) {
          // 765e-2 == 7.65
          FormatScientific = false;
        } else {
          // 765e-5 == 0.00765
          //           ^ ^^
          FormatScientific = ((unsigned) -MSD) > FormatMaxPadding;
        }
      }
    }

    // Scientific formatting is pretty straightforward.
    if (FormatScientific) {
      exp += (NDigits - 1);

      Str.push_back(buffer[NDigits-1]);
      Str.push_back('.');
      if (NDigits == 1 && TruncateZero)
        Str.push_back('0');
      else
        for (unsigned I = 1; I != NDigits; ++I)
          Str.push_back(buffer[NDigits-1-I]);
      // Fill with zeros up to FormatPrecision.
      if (!TruncateZero && FormatPrecision > NDigits - 1)
        Str.append(FormatPrecision - NDigits + 1, '0');
      // For !TruncateZero we use lower 'e'.
      Str.push_back(TruncateZero ? 'E' : 'e');

      Str.push_back(exp >= 0 ? '+' : '-');
      if (exp < 0)
        exp = -exp;
      SmallVector<char, 6> expbuf;
      do {
        expbuf.push_back((char) ('0' + (exp % 10)));
        exp /= 10;
      } while (exp);
      // Exponent always at least two digits if we do not truncate zeros.
      if (!TruncateZero && expbuf.size() < 2)
        expbuf.push_back('0');
      for (unsigned I = 0, E = expbuf.size(); I != E; ++I)
        Str.push_back(expbuf[E-1-I]);
      return;
    }

    // Non-scientific, positive exponents.
    if (exp >= 0) {
      for (unsigned I = 0; I != NDigits; ++I)
        Str.push_back(buffer[NDigits-1-I]);
      for (unsigned I = 0; I != (unsigned) exp; ++I)
        Str.push_back('0');
      return;
    }

    // Non-scientific, negative exponents.

    // The number of digits to the left of the decimal point.
    int NWholeDigits = exp + (int) NDigits;

    unsigned I = 0;
    if (NWholeDigits > 0) {
      for (; I != (unsigned) NWholeDigits; ++I)
        Str.push_back(buffer[NDigits-I-1]);
      Str.push_back('.');
    } else {
      unsigned NZeros = 1 + (unsigned) -NWholeDigits;

      Str.push_back('0');
      Str.push_back('.');
      for (unsigned Z = 1; Z != NZeros; ++Z)
        Str.push_back('0');
    }

    for (; I != NDigits; ++I)
      Str.push_back(buffer[NDigits-I-1]);

  }
} // namespace

void IEEEFloat::toString(SmallVectorImpl<char> &Str, unsigned FormatPrecision,
                         unsigned FormatMaxPadding, bool TruncateZero) const {
  switch (category) {
  case fcInfinity:
    if (isNegative())
      return append(Str, "-Inf");
    else
      return append(Str, "+Inf");

  case fcNaN: return append(Str, "NaN");

  case fcZero:
    if (isNegative())
      Str.push_back('-');

    if (!FormatMaxPadding) {
      if (TruncateZero)
        append(Str, "0.0E+0");
      else {
        append(Str, "0.0");
        if (FormatPrecision > 1)
          Str.append(FormatPrecision - 1, '0');
        append(Str, "e+00");
      }
    } else {
      Str.push_back('0');
    }
    return;

  case fcNormal:
    break;
  }

  // Decompose the number into an APInt and an exponent.
  int exp = exponent - ((int) semantics->precision - 1);
  APInt significand(
      semantics->precision,
      ArrayRef(significandParts(), partCountForBits(semantics->precision)));

  toStringImpl(Str, isNegative(), exp, significand, FormatPrecision,
               FormatMaxPadding, TruncateZero);

}

int IEEEFloat::getExactLog2Abs() const {
  if (!isFinite() || isZero())
    return INT_MIN;

  const integerPart *Parts = significandParts();
  const int PartCount = partCountForBits(semantics->precision);

  int PopCount = 0;
  for (int i = 0; i < PartCount; ++i) {
    PopCount += llvm::popcount(Parts[i]);
    if (PopCount > 1)
      return INT_MIN;
  }

  if (exponent != semantics->minExponent)
    return exponent;

  int CountrParts = 0;
  for (int i = 0; i < PartCount;
       ++i, CountrParts += APInt::APINT_BITS_PER_WORD) {
    if (Parts[i] != 0) {
      return exponent - semantics->precision + CountrParts +
             llvm::countr_zero(Parts[i]) + 1;
    }
  }

  llvm_unreachable("didn't find the set bit");
}

bool IEEEFloat::isSignaling() const {
  if (!isNaN())
    return false;
  if (semantics->nonFiniteBehavior == fltNonfiniteBehavior::NanOnly ||
      semantics->nonFiniteBehavior == fltNonfiniteBehavior::FiniteOnly)
    return false;

  // IEEE-754R 2008 6.2.1: A signaling NaN bit string should be encoded with the
  // first bit of the trailing significand being 0.
  return !APInt::tcExtractBit(significandParts(), semantics->precision - 2);
}

/// IEEE-754R 2008 5.3.1: nextUp/nextDown.
///
/// *NOTE* since nextDown(x) = -nextUp(-x), we only implement nextUp with
/// appropriate sign switching before/after the computation.
APFloat::opStatus IEEEFloat::next(bool nextDown) {
  // If we are performing nextDown, swap sign so we have -x.
  if (nextDown)
    changeSign();

  // Compute nextUp(x)
  opStatus result = opOK;

  // Handle each float category separately.
  switch (category) {
  case fcInfinity:
    // nextUp(+inf) = +inf
    if (!isNegative())
      break;
    // nextUp(-inf) = -getLargest()
    makeLargest(true);
    break;
  case fcNaN:
    // IEEE-754R 2008 6.2 Par 2: nextUp(sNaN) = qNaN. Set Invalid flag.
    // IEEE-754R 2008 6.2: nextUp(qNaN) = qNaN. Must be identity so we do not
    //                     change the payload.
    if (isSignaling()) {
      result = opInvalidOp;
      // For consistency, propagate the sign of the sNaN to the qNaN.
      makeNaN(false, isNegative(), nullptr);
    }
    break;
  case fcZero:
    // nextUp(pm 0) = +getSmallest()
    makeSmallest(false);
    break;
  case fcNormal:
    // nextUp(-getSmallest()) = -0
    if (isSmallest() && isNegative()) {
      APInt::tcSet(significandParts(), 0, partCount());
      category = fcZero;
      exponent = 0;
      if (semantics->nanEncoding == fltNanEncoding::NegativeZero)
        sign = false;
      if (!semantics->hasZero)
        makeSmallestNormalized(false);
      break;
    }

    if (isLargest() && !isNegative()) {
      if (semantics->nonFiniteBehavior == fltNonfiniteBehavior::NanOnly) {
        // nextUp(getLargest()) == NAN
        makeNaN();
        break;
      } else if (semantics->nonFiniteBehavior ==
                 fltNonfiniteBehavior::FiniteOnly) {
        // nextUp(getLargest()) == getLargest()
        break;
      } else {
        // nextUp(getLargest()) == INFINITY
        APInt::tcSet(significandParts(), 0, partCount());
        category = fcInfinity;
        exponent = semantics->maxExponent + 1;
        break;
      }
    }

    // nextUp(normal) == normal + inc.
    if (isNegative()) {
      // If we are negative, we need to decrement the significand.

      // We only cross a binade boundary that requires adjusting the exponent
      // if:
      //   1. exponent != semantics->minExponent. This implies we are not in the
      //   smallest binade or are dealing with denormals.
      //   2. Our significand excluding the integral bit is all zeros.
      bool WillCrossBinadeBoundary =
        exponent != semantics->minExponent && isSignificandAllZeros();

      // Decrement the significand.
      //
      // We always do this since:
      //   1. If we are dealing with a non-binade decrement, by definition we
      //   just decrement the significand.
      //   2. If we are dealing with a normal -> normal binade decrement, since
      //   we have an explicit integral bit the fact that all bits but the
      //   integral bit are zero implies that subtracting one will yield a
      //   significand with 0 integral bit and 1 in all other spots. Thus we
      //   must just adjust the exponent and set the integral bit to 1.
      //   3. If we are dealing with a normal -> denormal binade decrement,
      //   since we set the integral bit to 0 when we represent denormals, we
      //   just decrement the significand.
      integerPart *Parts = significandParts();
      APInt::tcDecrement(Parts, partCount());

      if (WillCrossBinadeBoundary) {
        // Our result is a normal number. Do the following:
        // 1. Set the integral bit to 1.
        // 2. Decrement the exponent.
        APInt::tcSetBit(Parts, semantics->precision - 1);
        exponent--;
      }
    } else {
      // If we are positive, we need to increment the significand.

      // We only cross a binade boundary that requires adjusting the exponent if
      // the input is not a denormal and all of said input's significand bits
      // are set. If all of said conditions are true: clear the significand, set
      // the integral bit to 1, and increment the exponent. If we have a
      // denormal always increment since moving denormals and the numbers in the
      // smallest normal binade have the same exponent in our representation.
      // If there are only exponents, any increment always crosses the
      // BinadeBoundary.
      bool WillCrossBinadeBoundary = !APFloat::hasSignificand(*semantics) ||
                                     (!isDenormal() && isSignificandAllOnes());

      if (WillCrossBinadeBoundary) {
        integerPart *Parts = significandParts();
        APInt::tcSet(Parts, 0, partCount());
        APInt::tcSetBit(Parts, semantics->precision - 1);
        assert(exponent != semantics->maxExponent &&
               "We can not increment an exponent beyond the maxExponent allowed"
               " by the given floating point semantics.");
        exponent++;
      } else {
        incrementSignificand();
      }
    }
    break;
  }

  // If we are performing nextDown, swap sign so we have -nextUp(-x)
  if (nextDown)
    changeSign();

  return result;
}

APFloatBase::ExponentType IEEEFloat::exponentNaN() const {
  return ::exponentNaN(*semantics);
}

APFloatBase::ExponentType IEEEFloat::exponentInf() const {
  return ::exponentInf(*semantics);
}

APFloatBase::ExponentType IEEEFloat::exponentZero() const {
  return ::exponentZero(*semantics);
}

void IEEEFloat::makeInf(bool Negative) {
  if (semantics->nonFiniteBehavior == fltNonfiniteBehavior::FiniteOnly)
    llvm_unreachable("This floating point format does not support Inf");

  if (semantics->nonFiniteBehavior == fltNonfiniteBehavior::NanOnly) {
    // There is no Inf, so make NaN instead.
    makeNaN(false, Negative);
    return;
  }
  category = fcInfinity;
  sign = Negative;
  exponent = exponentInf();
  APInt::tcSet(significandParts(), 0, partCount());
}

void IEEEFloat::makeZero(bool Negative) {
  if (!semantics->hasZero)
    llvm_unreachable("This floating point format does not support Zero");

  category = fcZero;
  sign = Negative;
  if (semantics->nanEncoding == fltNanEncoding::NegativeZero) {
    // Merge negative zero to positive because 0b10000...000 is used for NaN
    sign = false;
  }
  exponent = exponentZero();
  APInt::tcSet(significandParts(), 0, partCount());
}

void IEEEFloat::makeQuiet() {
  assert(isNaN());
  if (semantics->nonFiniteBehavior != fltNonfiniteBehavior::NanOnly)
    APInt::tcSetBit(significandParts(), semantics->precision - 2);
}

int ilogb(const IEEEFloat &Arg) {
  if (Arg.isNaN())
    return APFloat::IEK_NaN;
  if (Arg.isZero())
    return APFloat::IEK_Zero;
  if (Arg.isInfinity())
    return APFloat::IEK_Inf;
  if (!Arg.isDenormal())
    return Arg.exponent;

  IEEEFloat Normalized(Arg);
  int SignificandBits = Arg.getSemantics().precision - 1;

  Normalized.exponent += SignificandBits;
  Normalized.normalize(APFloat::rmNearestTiesToEven, lfExactlyZero);
  return Normalized.exponent - SignificandBits;
}

IEEEFloat scalbn(IEEEFloat X, int Exp, roundingMode RoundingMode) {
  auto MaxExp = X.getSemantics().maxExponent;
  auto MinExp = X.getSemantics().minExponent;

  // If Exp is wildly out-of-scale, simply adding it to X.exponent will
  // overflow; clamp it to a safe range before adding, but ensure that the range
  // is large enough that the clamp does not change the result. The range we
  // need to support is the difference between the largest possible exponent and
  // the normalized exponent of half the smallest denormal.

  int SignificandBits = X.getSemantics().precision - 1;
  int MaxIncrement = MaxExp - (MinExp - SignificandBits) + 1;

  // Clamp to one past the range ends to let normalize handle overlflow.
  X.exponent += std::clamp(Exp, -MaxIncrement - 1, MaxIncrement);
  X.normalize(RoundingMode, lfExactlyZero);
  if (X.isNaN())
    X.makeQuiet();
  return X;
}

IEEEFloat frexp(const IEEEFloat &Val, int &Exp, roundingMode RM) {
  Exp = ilogb(Val);

  // Quiet signalling nans.
  if (Exp == APFloat::IEK_NaN) {
    IEEEFloat Quiet(Val);
    Quiet.makeQuiet();
    return Quiet;
  }

  if (Exp == APFloat::IEK_Inf)
    return Val;

  // 1 is added because frexp is defined to return a normalized fraction in
  // +/-[0.5, 1.0), rather than the usual +/-[1.0, 2.0).
  Exp = Exp == APFloat::IEK_Zero ? 0 : Exp + 1;
  return scalbn(Val, -Exp, RM);
}

DoubleAPFloat::DoubleAPFloat(const fltSemantics &S)
    : Semantics(&S),
      Floats(new APFloat[2]{APFloat(APFloatBase::semIEEEdouble),
                            APFloat(APFloatBase::semIEEEdouble)}) {
  assert(Semantics == &APFloatBase::semPPCDoubleDouble);
}

DoubleAPFloat::DoubleAPFloat(const fltSemantics &S, uninitializedTag)
    : Semantics(&S), Floats(new APFloat[2]{
                         APFloat(APFloatBase::semIEEEdouble, uninitialized),
                         APFloat(APFloatBase::semIEEEdouble, uninitialized)}) {
  assert(Semantics == &APFloatBase::semPPCDoubleDouble);
}

DoubleAPFloat::DoubleAPFloat(const fltSemantics &S, integerPart I)
    : Semantics(&S),
      Floats(new APFloat[2]{APFloat(APFloatBase::semIEEEdouble, I),
                            APFloat(APFloatBase::semIEEEdouble)}) {
  assert(Semantics == &APFloatBase::semPPCDoubleDouble);
}

DoubleAPFloat::DoubleAPFloat(const fltSemantics &S, const APInt &I)
    : Semantics(&S),
      Floats(new APFloat[2]{
          APFloat(APFloatBase::semIEEEdouble, APInt(64, I.getRawData()[0])),
          APFloat(APFloatBase::semIEEEdouble, APInt(64, I.getRawData()[1]))}) {
  assert(Semantics == &APFloatBase::semPPCDoubleDouble);
}

DoubleAPFloat::DoubleAPFloat(const fltSemantics &S, APFloat &&First,
                             APFloat &&Second)
    : Semantics(&S),
      Floats(new APFloat[2]{std::move(First), std::move(Second)}) {
  assert(Semantics == &APFloatBase::semPPCDoubleDouble);
  assert(&Floats[0].getSemantics() == &APFloatBase::semIEEEdouble);
  assert(&Floats[1].getSemantics() == &APFloatBase::semIEEEdouble);
}

DoubleAPFloat::DoubleAPFloat(const DoubleAPFloat &RHS)
    : Semantics(RHS.Semantics),
      Floats(RHS.Floats ? new APFloat[2]{APFloat(RHS.Floats[0]),
                                         APFloat(RHS.Floats[1])}
                        : nullptr) {
  assert(Semantics == &APFloatBase::semPPCDoubleDouble);
}

DoubleAPFloat::DoubleAPFloat(DoubleAPFloat &&RHS)
    : Semantics(RHS.Semantics), Floats(RHS.Floats) {
  RHS.Semantics = &APFloatBase::semBogus;
  RHS.Floats = nullptr;
  assert(Semantics == &APFloatBase::semPPCDoubleDouble);
}

DoubleAPFloat &DoubleAPFloat::operator=(const DoubleAPFloat &RHS) {
  if (Semantics == RHS.Semantics && RHS.Floats) {
    Floats[0] = RHS.Floats[0];
    Floats[1] = RHS.Floats[1];
  } else if (this != &RHS) {
    this->~DoubleAPFloat();
    new (this) DoubleAPFloat(RHS);
  }
  return *this;
}

// Returns a result such that:
// 1. abs(Lo) <= ulp(Hi)/2
// 2. Hi == RTNE(Hi + Lo)
// 3. Hi + Lo == X + Y
//
// Requires that log2(X) >= log2(Y).
static std::pair<APFloat, APFloat> fastTwoSum(APFloat X, APFloat Y) {
  if (!X.isFinite())
    return {X, APFloat::getZero(X.getSemantics(), /*Negative=*/false)};
  APFloat Hi = X + Y;
  APFloat Delta = Hi - X;
  APFloat Lo = Y - Delta;
  return {Hi, Lo};
}

// Implement addition, subtraction, multiplication and division based on:
// "Software for Doubled-Precision Floating-Point Computations",
// by Seppo Linnainmaa, ACM TOMS vol 7 no 3, September 1981, pages 272-283.
APFloat::opStatus DoubleAPFloat::addImpl(const APFloat &a, const APFloat &aa,
                                         const APFloat &c, const APFloat &cc,
                                         roundingMode RM) {
  int Status = opOK;
  APFloat z = a;
  Status |= z.add(c, RM);
  if (!z.isFinite()) {
    if (!z.isInfinity()) {
      Floats[0] = std::move(z);
      Floats[1].makeZero(/* Neg = */ false);
      return (opStatus)Status;
    }
    Status = opOK;
    auto AComparedToC = a.compareAbsoluteValue(c);
    z = cc;
    Status |= z.add(aa, RM);
    if (AComparedToC == APFloat::cmpGreaterThan) {
      // z = cc + aa + c + a;
      Status |= z.add(c, RM);
      Status |= z.add(a, RM);
    } else {
      // z = cc + aa + a + c;
      Status |= z.add(a, RM);
      Status |= z.add(c, RM);
    }
    if (!z.isFinite()) {
      Floats[0] = std::move(z);
      Floats[1].makeZero(/* Neg = */ false);
      return (opStatus)Status;
    }
    Floats[0] = z;
    APFloat zz = aa;
    Status |= zz.add(cc, RM);
    if (AComparedToC == APFloat::cmpGreaterThan) {
      // Floats[1] = a - z + c + zz;
      Floats[1] = a;
      Status |= Floats[1].subtract(z, RM);
      Status |= Floats[1].add(c, RM);
      Status |= Floats[1].add(zz, RM);
    } else {
      // Floats[1] = c - z + a + zz;
      Floats[1] = c;
      Status |= Floats[1].subtract(z, RM);
      Status |= Floats[1].add(a, RM);
      Status |= Floats[1].add(zz, RM);
    }
  } else {
    // q = a - z;
    APFloat q = a;
    Status |= q.subtract(z, RM);

    // zz = q + c + (a - (q + z)) + aa + cc;
    // Compute a - (q + z) as -((q + z) - a) to avoid temporary copies.
    auto zz = q;
    Status |= zz.add(c, RM);
    Status |= q.add(z, RM);
    Status |= q.subtract(a, RM);
    q.changeSign();
    Status |= zz.add(q, RM);
    Status |= zz.add(aa, RM);
    Status |= zz.add(cc, RM);
    if (zz.isZero() && !zz.isNegative()) {
      Floats[0] = std::move(z);
      Floats[1].makeZero(/* Neg = */ false);
      return opOK;
    }
    Floats[0] = z;
    Status |= Floats[0].add(zz, RM);
    if (!Floats[0].isFinite()) {
      Floats[1].makeZero(/* Neg = */ false);
      return (opStatus)Status;
    }
    Floats[1] = std::move(z);
    Status |= Floats[1].subtract(Floats[0], RM);
    Status |= Floats[1].add(zz, RM);
  }
  return (opStatus)Status;
}

APFloat::opStatus DoubleAPFloat::addWithSpecial(const DoubleAPFloat &LHS,
                                                const DoubleAPFloat &RHS,
                                                DoubleAPFloat &Out,
                                                roundingMode RM) {
  if (LHS.getCategory() == fcNaN) {
    Out = LHS;
    return opOK;
  }
  if (RHS.getCategory() == fcNaN) {
    Out = RHS;
    return opOK;
  }
  if (LHS.getCategory() == fcZero) {
    Out = RHS;
    return opOK;
  }
  if (RHS.getCategory() == fcZero) {
    Out = LHS;
    return opOK;
  }
  if (LHS.getCategory() == fcInfinity && RHS.getCategory() == fcInfinity &&
      LHS.isNegative() != RHS.isNegative()) {
    Out.makeNaN(false, Out.isNegative(), nullptr);
    return opInvalidOp;
  }
  if (LHS.getCategory() == fcInfinity) {
    Out = LHS;
    return opOK;
  }
  if (RHS.getCategory() == fcInfinity) {
    Out = RHS;
    return opOK;
  }
  assert(LHS.getCategory() == fcNormal && RHS.getCategory() == fcNormal);

  APFloat A(LHS.Floats[0]), AA(LHS.Floats[1]), C(RHS.Floats[0]),
      CC(RHS.Floats[1]);
  assert(&A.getSemantics() == &APFloatBase::semIEEEdouble);
  assert(&AA.getSemantics() == &APFloatBase::semIEEEdouble);
  assert(&C.getSemantics() == &APFloatBase::semIEEEdouble);
  assert(&CC.getSemantics() == &APFloatBase::semIEEEdouble);
  assert(&Out.Floats[0].getSemantics() == &APFloatBase::semIEEEdouble);
  assert(&Out.Floats[1].getSemantics() == &APFloatBase::semIEEEdouble);
  return Out.addImpl(A, AA, C, CC, RM);
}

APFloat::opStatus DoubleAPFloat::add(const DoubleAPFloat &RHS,
                                     roundingMode RM) {
  return addWithSpecial(*this, RHS, *this, RM);
}

APFloat::opStatus DoubleAPFloat::subtract(const DoubleAPFloat &RHS,
                                          roundingMode RM) {
  changeSign();
  auto Ret = add(RHS, RM);
  changeSign();
  return Ret;
}

APFloat::opStatus DoubleAPFloat::multiply(const DoubleAPFloat &RHS,
                                          APFloat::roundingMode RM) {
  const auto &LHS = *this;
  auto &Out = *this;
  /* Interesting observation: For special categories, finding the lowest
     common ancestor of the following layered graph gives the correct
     return category:

        NaN
       /   \
     Zero  Inf
       \   /
       Normal

     e.g. NaN * NaN = NaN
          Zero * Inf = NaN
          Normal * Zero = Zero
          Normal * Inf = Inf
  */
  if (LHS.getCategory() == fcNaN) {
    Out = LHS;
    return opOK;
  }
  if (RHS.getCategory() == fcNaN) {
    Out = RHS;
    return opOK;
  }
  if ((LHS.getCategory() == fcZero && RHS.getCategory() == fcInfinity) ||
      (LHS.getCategory() == fcInfinity && RHS.getCategory() == fcZero)) {
    Out.makeNaN(false, false, nullptr);
    return opOK;
  }
  if (LHS.getCategory() == fcZero || LHS.getCategory() == fcInfinity) {
    Out = LHS;
    return opOK;
  }
  if (RHS.getCategory() == fcZero || RHS.getCategory() == fcInfinity) {
    Out = RHS;
    return opOK;
  }
  assert(LHS.getCategory() == fcNormal && RHS.getCategory() == fcNormal &&
         "Special cases not handled exhaustively");

  int Status = opOK;
  APFloat A = Floats[0], B = Floats[1], C = RHS.Floats[0], D = RHS.Floats[1];
  // t = a * c
  APFloat T = A;
  Status |= T.multiply(C, RM);
  if (!T.isFiniteNonZero()) {
    Floats[0] = T;
    Floats[1].makeZero(/* Neg = */ false);
    return (opStatus)Status;
  }

  // tau = fmsub(a, c, t), that is -fmadd(-a, c, t).
  APFloat Tau = A;
  T.changeSign();
  Status |= Tau.fusedMultiplyAdd(C, T, RM);
  T.changeSign();
  {
    // v = a * d
    APFloat V = A;
    Status |= V.multiply(D, RM);
    // w = b * c
    APFloat W = B;
    Status |= W.multiply(C, RM);
    Status |= V.add(W, RM);
    // tau += v + w
    Status |= Tau.add(V, RM);
  }
  // u = t + tau
  APFloat U = T;
  Status |= U.add(Tau, RM);

  Floats[0] = U;
  if (!U.isFinite()) {
    Floats[1].makeZero(/* Neg = */ false);
  } else {
    // Floats[1] = (t - u) + tau
    Status |= T.subtract(U, RM);
    Status |= T.add(Tau, RM);
    Floats[1] = T;
  }
  return (opStatus)Status;
}

APFloat::opStatus DoubleAPFloat::divide(const DoubleAPFloat &RHS,
                                        APFloat::roundingMode RM) {
  assert(Semantics == &APFloatBase::semPPCDoubleDouble &&
         "Unexpected Semantics");
  APFloat Tmp(APFloatBase::semPPCDoubleDoubleLegacy, bitcastToAPInt());
  auto Ret = Tmp.divide(
      APFloat(APFloatBase::semPPCDoubleDoubleLegacy, RHS.bitcastToAPInt()), RM);
  *this = DoubleAPFloat(APFloatBase::semPPCDoubleDouble, Tmp.bitcastToAPInt());
  return Ret;
}

APFloat::opStatus DoubleAPFloat::remainder(const DoubleAPFloat &RHS) {
  assert(Semantics == &APFloatBase::semPPCDoubleDouble &&
         "Unexpected Semantics");
  APFloat Tmp(APFloatBase::semPPCDoubleDoubleLegacy, bitcastToAPInt());
  auto Ret = Tmp.remainder(
      APFloat(APFloatBase::semPPCDoubleDoubleLegacy, RHS.bitcastToAPInt()));
  *this = DoubleAPFloat(APFloatBase::semPPCDoubleDouble, Tmp.bitcastToAPInt());
  return Ret;
}

APFloat::opStatus DoubleAPFloat::mod(const DoubleAPFloat &RHS) {
  assert(Semantics == &APFloatBase::semPPCDoubleDouble &&
         "Unexpected Semantics");
  APFloat Tmp(APFloatBase::semPPCDoubleDoubleLegacy, bitcastToAPInt());
  auto Ret = Tmp.mod(
      APFloat(APFloatBase::semPPCDoubleDoubleLegacy, RHS.bitcastToAPInt()));
  *this = DoubleAPFloat(APFloatBase::semPPCDoubleDouble, Tmp.bitcastToAPInt());
  return Ret;
}

APFloat::opStatus
DoubleAPFloat::fusedMultiplyAdd(const DoubleAPFloat &Multiplicand,
                                const DoubleAPFloat &Addend,
                                APFloat::roundingMode RM) {
  assert(Semantics == &APFloatBase::semPPCDoubleDouble &&
         "Unexpected Semantics");
  APFloat Tmp(APFloatBase::semPPCDoubleDoubleLegacy, bitcastToAPInt());
  auto Ret = Tmp.fusedMultiplyAdd(
      APFloat(APFloatBase::semPPCDoubleDoubleLegacy,
              Multiplicand.bitcastToAPInt()),
      APFloat(APFloatBase::semPPCDoubleDoubleLegacy, Addend.bitcastToAPInt()),
      RM);
  *this = DoubleAPFloat(APFloatBase::semPPCDoubleDouble, Tmp.bitcastToAPInt());
  return Ret;
}

APFloat::opStatus DoubleAPFloat::roundToIntegral(APFloat::roundingMode RM) {
  assert(Semantics == &APFloatBase::semPPCDoubleDouble &&
         "Unexpected Semantics");
  const APFloat &Hi = getFirst();
  const APFloat &Lo = getSecond();

  APFloat RoundedHi = Hi;
  const opStatus HiStatus = RoundedHi.roundToIntegral(RM);

  // We can reduce the problem to just the high part if the input:
  // 1. Represents a non-finite value.
  // 2. Has a component which is zero.
  if (!Hi.isFiniteNonZero() || Lo.isZero()) {
    Floats[0] = std::move(RoundedHi);
    Floats[1].makeZero(/*Neg=*/false);
    return HiStatus;
  }

  // Adjust `Rounded` in the direction of `TieBreaker` if `ToRound` was at a
  // halfway point.
  auto RoundToNearestHelper = [](APFloat ToRound, APFloat Rounded,
                                 APFloat TieBreaker) {
    // RoundingError tells us which direction we rounded:
    //   - RoundingError > 0: we rounded up.
    //   - RoundingError < 0: we rounded down.
    // Sterbenz' lemma ensures that RoundingError is exact.
    const APFloat RoundingError = Rounded - ToRound;
    if (TieBreaker.isNonZero() &&
        TieBreaker.isNegative() != RoundingError.isNegative() &&
        abs(RoundingError).isExactlyValue(0.5))
      Rounded.add(
          APFloat::getOne(Rounded.getSemantics(), TieBreaker.isNegative()),
          rmNearestTiesToEven);
    return Rounded;
  };

  // Case 1: Hi is not an integer.
  // Special cases are for rounding modes that are sensitive to ties.
  if (RoundedHi != Hi) {
    // We need to consider the case where Hi was between two integers and the
    // rounding mode broke the tie when, in fact, Lo may have had a different
    // sign than Hi.
    if (RM == rmNearestTiesToAway || RM == rmNearestTiesToEven)
      RoundedHi = RoundToNearestHelper(Hi, RoundedHi, Lo);

    Floats[0] = std::move(RoundedHi);
    Floats[1].makeZero(/*Neg=*/false);
    return HiStatus;
  }

  // Case 2: Hi is an integer.
  // Special cases are for rounding modes which are rounding towards or away from zero.
  RoundingMode LoRoundingMode;
  if (RM == rmTowardZero)
    // When our input is positive, we want the Lo component rounded toward
    // negative infinity to get the smallest result magnitude. Likewise,
    // negative inputs want the Lo component rounded toward positive infinity.
    LoRoundingMode = isNegative() ? rmTowardPositive : rmTowardNegative;
  else
    LoRoundingMode = RM;

  APFloat RoundedLo = Lo;
  const opStatus LoStatus = RoundedLo.roundToIntegral(LoRoundingMode);
  if (LoRoundingMode == rmNearestTiesToAway)
    // We need to consider the case where Lo was between two integers and the
    // rounding mode broke the tie when, in fact, Hi may have had a different
    // sign than Lo.
    RoundedLo = RoundToNearestHelper(Lo, RoundedLo, Hi);

  // We must ensure that the final result has no overlap between the two APFloat values.
  std::tie(RoundedHi, RoundedLo) = fastTwoSum(RoundedHi, RoundedLo);

  Floats[0] = std::move(RoundedHi);
  Floats[1] = std::move(RoundedLo);
  return LoStatus;
}

void DoubleAPFloat::changeSign() {
  Floats[0].changeSign();
  Floats[1].changeSign();
}

APFloat::cmpResult
DoubleAPFloat::compareAbsoluteValue(const DoubleAPFloat &RHS) const {
  // Compare absolute values of the high parts.
  const cmpResult HiPartCmp = Floats[0].compareAbsoluteValue(RHS.Floats[0]);
  if (HiPartCmp != cmpEqual)
    return HiPartCmp;

  // Zero, regardless of sign, is equal.
  if (Floats[1].isZero() && RHS.Floats[1].isZero())
    return cmpEqual;

  // At this point, |this->Hi| == |RHS.Hi|.
  // The magnitude is |Hi+Lo| which is Hi+|Lo| if signs of Hi and Lo are the
  // same, and Hi-|Lo| if signs are different.
  const bool ThisIsSubtractive =
      Floats[0].isNegative() != Floats[1].isNegative();
  const bool RHSIsSubtractive =
      RHS.Floats[0].isNegative() != RHS.Floats[1].isNegative();

  // Case 1: The low part of 'this' is zero.
  if (Floats[1].isZero())
    // We are comparing |Hi| vs. |Hi| ± |RHS.Lo|.
    // If RHS is subtractive, its magnitude is smaller.
    // If RHS is additive, its magnitude is larger.
    return RHSIsSubtractive ? cmpGreaterThan : cmpLessThan;

  // Case 2: The low part of 'RHS' is zero (and we know 'this' is not).
  if (RHS.Floats[1].isZero())
    // We are comparing |Hi| ± |This.Lo| vs. |Hi|.
    // If 'this' is subtractive, its magnitude is smaller.
    // If 'this' is additive, its magnitude is larger.
    return ThisIsSubtractive ? cmpLessThan : cmpGreaterThan;

  // If their natures differ, the additive one is larger.
  if (ThisIsSubtractive != RHSIsSubtractive)
    return ThisIsSubtractive ? cmpLessThan : cmpGreaterThan;

  // Case 3: Both are additive (Hi+|Lo|) or both are subtractive (Hi-|Lo|).
  // The comparison now depends on the magnitude of the low parts.
  const cmpResult LoPartCmp = Floats[1].compareAbsoluteValue(RHS.Floats[1]);

  if (ThisIsSubtractive) {
    // Both are subtractive (Hi-|Lo|), so the comparison of |Lo| is inverted.
    if (LoPartCmp == cmpLessThan)
      return cmpGreaterThan;
    if (LoPartCmp == cmpGreaterThan)
      return cmpLessThan;
  }

  // If additive, the comparison of |Lo| is direct.
  // If equal, they are equal.
  return LoPartCmp;
}

APFloat::fltCategory DoubleAPFloat::getCategory() const {
  return Floats[0].getCategory();
}

bool DoubleAPFloat::isNegative() const { return Floats[0].isNegative(); }

void DoubleAPFloat::makeInf(bool Neg) {
  Floats[0].makeInf(Neg);
  Floats[1].makeZero(/* Neg = */ false);
}

void DoubleAPFloat::makeZero(bool Neg) {
  Floats[0].makeZero(Neg);
  Floats[1].makeZero(/* Neg = */ false);
}

void DoubleAPFloat::makeLargest(bool Neg) {
  assert(Semantics == &APFloatBase::semPPCDoubleDouble &&
         "Unexpected Semantics");
  Floats[0] =
      APFloat(APFloatBase::semIEEEdouble, APInt(64, 0x7fefffffffffffffull));
  Floats[1] =
      APFloat(APFloatBase::semIEEEdouble, APInt(64, 0x7c8ffffffffffffeull));
  if (Neg)
    changeSign();
}

void DoubleAPFloat::makeSmallest(bool Neg) {
  assert(Semantics == &APFloatBase::semPPCDoubleDouble &&
         "Unexpected Semantics");
  Floats[0].makeSmallest(Neg);
  Floats[1].makeZero(/* Neg = */ false);
}

void DoubleAPFloat::makeSmallestNormalized(bool Neg) {
  assert(Semantics == &APFloatBase::semPPCDoubleDouble &&
         "Unexpected Semantics");
  Floats[0] =
      APFloat(APFloatBase::semIEEEdouble, APInt(64, 0x0360000000000000ull));
  if (Neg)
    Floats[0].changeSign();
  Floats[1].makeZero(/* Neg = */ false);
}

void DoubleAPFloat::makeNaN(bool SNaN, bool Neg, const APInt *fill) {
  Floats[0].makeNaN(SNaN, Neg, fill);
  Floats[1].makeZero(/* Neg = */ false);
}

APFloat::cmpResult DoubleAPFloat::compare(const DoubleAPFloat &RHS) const {
  auto Result = Floats[0].compare(RHS.Floats[0]);
  // |Float[0]| > |Float[1]|
  if (Result == APFloat::cmpEqual)
    return Floats[1].compare(RHS.Floats[1]);
  return Result;
}

bool DoubleAPFloat::bitwiseIsEqual(const DoubleAPFloat &RHS) const {
  return Floats[0].bitwiseIsEqual(RHS.Floats[0]) &&
         Floats[1].bitwiseIsEqual(RHS.Floats[1]);
}

hash_code hash_value(const DoubleAPFloat &Arg) {
  if (Arg.Floats)
    return hash_combine(hash_value(Arg.Floats[0]), hash_value(Arg.Floats[1]));
  return hash_combine(Arg.Semantics);
}

APInt DoubleAPFloat::bitcastToAPInt() const {
  assert(Semantics == &APFloatBase::semPPCDoubleDouble &&
         "Unexpected Semantics");
  uint64_t Data[] = {
      Floats[0].bitcastToAPInt().getRawData()[0],
      Floats[1].bitcastToAPInt().getRawData()[0],
  };
  return APInt(128, Data);
}

Expected<APFloat::opStatus> DoubleAPFloat::convertFromString(StringRef S,
                                                             roundingMode RM) {
  assert(Semantics == &APFloatBase::semPPCDoubleDouble &&
         "Unexpected Semantics");
  APFloat Tmp(APFloatBase::semPPCDoubleDoubleLegacy);
  auto Ret = Tmp.convertFromString(S, RM);
  *this = DoubleAPFloat(APFloatBase::semPPCDoubleDouble, Tmp.bitcastToAPInt());
  return Ret;
}

// The double-double lattice of values corresponds to numbers which obey:
// - abs(lo) <= 1/2 * ulp(hi)
// - roundTiesToEven(hi + lo) == hi
//
// nextUp must choose the smallest output > input that follows these rules.
// nexDown must choose the largest output < input that follows these rules.
APFloat::opStatus DoubleAPFloat::next(bool nextDown) {
  assert(Semantics == &APFloatBase::semPPCDoubleDouble &&
         "Unexpected Semantics");
  // nextDown(x) = -nextUp(-x)
  if (nextDown) {
    changeSign();
    APFloat::opStatus Result = next(/*nextDown=*/false);
    changeSign();
    return Result;
  }
  switch (getCategory()) {
  case fcInfinity:
    // nextUp(+inf) = +inf
    // nextUp(-inf) = -getLargest()
    if (isNegative())
      makeLargest(true);
    return opOK;

  case fcNaN:
    // IEEE-754R 2008 6.2 Par 2: nextUp(sNaN) = qNaN. Set Invalid flag.
    // IEEE-754R 2008 6.2: nextUp(qNaN) = qNaN. Must be identity so we do not
    //                     change the payload.
    if (getFirst().isSignaling()) {
      // For consistency, propagate the sign of the sNaN to the qNaN.
      makeNaN(false, isNegative(), nullptr);
      return opInvalidOp;
    }
    return opOK;

  case fcZero:
    // nextUp(pm 0) = +getSmallest()
    makeSmallest(false);
    return opOK;

  case fcNormal:
    break;
  }

  const APFloat &HiOld = getFirst();
  const APFloat &LoOld = getSecond();

  APFloat NextLo = LoOld;
  NextLo.next(/*nextDown=*/false);

  // We want to admit values where:
  // 1. abs(Lo) <= ulp(Hi)/2
  // 2. Hi == RTNE(Hi + lo)
  auto InLattice = [](const APFloat &Hi, const APFloat &Lo) {
    return Hi + Lo == Hi;
  };

  // Check if (HiOld, nextUp(LoOld) is in the lattice.
  if (InLattice(HiOld, NextLo)) {
    // Yes, the result is (HiOld, nextUp(LoOld)).
    Floats[1] = std::move(NextLo);

    // TODO: Because we currently rely on semPPCDoubleDoubleLegacy, our maximum
    // value is defined to have exactly 106 bits of precision. This limitation
    // results in semPPCDoubleDouble being unable to reach its maximum canonical
    // value.
    DoubleAPFloat Largest{*Semantics, uninitialized};
    Largest.makeLargest(/*Neg=*/false);
    if (compare(Largest) == cmpGreaterThan)
      makeInf(/*Neg=*/false);

    return opOK;
  }

  // Now we need to handle the cases where (HiOld, nextUp(LoOld)) is not the
  // correct result. We know the new hi component will be nextUp(HiOld) but our
  // lattice rules make it a little ambiguous what the correct NextLo must be.
  APFloat NextHi = HiOld;
  NextHi.next(/*nextDown=*/false);

  // nextUp(getLargest()) == INFINITY
  if (NextHi.isInfinity()) {
    makeInf(/*Neg=*/false);
    return opOK;
  }

  // IEEE 754-2019 5.3.1:
  // "If x is the negative number of least magnitude in x's format, nextUp(x) is
  // -0."
  if (NextHi.isZero()) {
    makeZero(/*Neg=*/true);
    return opOK;
  }

  // abs(NextLo) must be <= ulp(NextHi)/2. We want NextLo to be as close to
  // negative infinity as possible.
  NextLo = neg(scalbn(harrisonUlp(NextHi), -1, rmTowardZero));
  if (!InLattice(NextHi, NextLo))
    // RTNE may mean that Lo must be < ulp(NextHi) / 2 so we bump NextLo.
    NextLo.next(/*nextDown=*/false);

  Floats[0] = std::move(NextHi);
  Floats[1] = std::move(NextLo);

  return opOK;
}

APFloat::opStatus DoubleAPFloat::convertToSignExtendedInteger(
    MutableArrayRef<integerPart> Input, unsigned int Width, bool IsSigned,
    roundingMode RM, bool *IsExact) const {
  assert(Semantics == &APFloatBase::semPPCDoubleDouble &&
         "Unexpected Semantics");

  // If Hi is not finite, or Lo is zero, the value is entirely represented
  // by Hi. Delegate to the simpler single-APFloat conversion.
  if (!getFirst().isFiniteNonZero() || getSecond().isZero())
    return getFirst().convertToInteger(Input, Width, IsSigned, RM, IsExact);

  // First, round the full double-double value to an integral value. This
  // simplifies the rest of the function, as we no longer need to consider
  // fractional parts.
  *IsExact = false;
  DoubleAPFloat Integral = *this;
  const opStatus RoundStatus = Integral.roundToIntegral(RM);
  if (RoundStatus == opInvalidOp)
    return opInvalidOp;
  const APFloat &IntegralHi = Integral.getFirst();
  const APFloat &IntegralLo = Integral.getSecond();

  // If rounding results in either component being zero, the sum is trivial.
  // Delegate to the simpler single-APFloat conversion.
  bool HiIsExact;
  if (IntegralHi.isZero() || IntegralLo.isZero()) {
    const opStatus HiStatus =
        IntegralHi.convertToInteger(Input, Width, IsSigned, RM, &HiIsExact);
    // The conversion from an integer-valued float to an APInt may fail if the
    // result would be out of range.  Regardless, taking this path is only
    // possible if rounding occurred during the initial `roundToIntegral`.
    return HiStatus == opOK ? opInexact : HiStatus;
  }

  // A negative number cannot be represented by an unsigned integer.
  // Since a double-double is canonical, if Hi is negative, the sum is negative.
  if (!IsSigned && IntegralHi.isNegative())
    return opInvalidOp;

  // Handle the special boundary case where |Hi| is exactly the power of two
  // that marks the edge of the integer's range (e.g., 2^63 for int64_t). In
  // this situation, Hi itself won't fit, but the sum Hi + Lo might.
  // `PositiveOverflowWidth` is the bit number for this boundary (N-1 for
  // signed, N for unsigned).
  bool LoIsExact;
  const int HiExactLog2 = IntegralHi.getExactLog2Abs();
  const unsigned PositiveOverflowWidth = IsSigned ? Width - 1 : Width;
  if (HiExactLog2 >= 0 &&
      static_cast<unsigned>(HiExactLog2) == PositiveOverflowWidth) {
    // If Hi and Lo have the same sign, |Hi + Lo| > |Hi|, so the sum is
    // guaranteed to overflow. E.g., for uint128_t, (2^128, 1) overflows.
    if (IntegralHi.isNegative() == IntegralLo.isNegative())
      return opInvalidOp;

    // If the signs differ, the sum will fit. We can compute the result using
    // properties of two's complement arithmetic without a wide intermediate
    // integer. E.g., for uint128_t, (2^128, -1) should be 2^128 - 1.
    const opStatus LoStatus = IntegralLo.convertToInteger(
        Input, Width, /*IsSigned=*/true, RM, &LoIsExact);
    if (LoStatus == opInvalidOp)
      return opInvalidOp;

    // Adjust the bit pattern of Lo to account for Hi's value:
    //  - For unsigned (Hi=2^Width): `2^Width + Lo` in `Width`-bit
    //    arithmetic is equivalent to just `Lo`. The conversion of `Lo` above
    //    already produced the correct final bit pattern.
    //  - For signed (Hi=2^(Width-1)): The sum `2^(Width-1) + Lo` (where Lo<0)
    //    can be computed by taking the two's complement pattern for `Lo` and
    //    clearing the sign bit.
    if (IsSigned && !IntegralHi.isNegative())
      APInt::tcClearBit(Input.data(), PositiveOverflowWidth);
    *IsExact = RoundStatus == opOK;
    return RoundStatus;
  }

  // Convert Hi into an integer.  This may not fit but that is OK: we know that
  // Hi + Lo would not fit either in this situation.
  const opStatus HiStatus = IntegralHi.convertToInteger(
      Input, Width, IsSigned, rmTowardZero, &HiIsExact);
  if (HiStatus == opInvalidOp)
    return HiStatus;

  // Convert Lo into a temporary integer of the same width.
  APSInt LoResult{Width, /*isUnsigned=*/!IsSigned};
  const opStatus LoStatus =
      IntegralLo.convertToInteger(LoResult, rmTowardZero, &LoIsExact);
  if (LoStatus == opInvalidOp)
    return LoStatus;

  // Add Lo to Hi. This addition is guaranteed not to overflow because of the
  // double-double canonicalization rule (`|Lo| <= ulp(Hi)/2`). The only case
  // where the sum could cross the integer type's boundary is when Hi is a
  // power of two, which is handled by the special case block above.
  APInt::tcAdd(Input.data(), LoResult.getRawData(), /*carry=*/0, Input.size());

  *IsExact = RoundStatus == opOK;
  return RoundStatus;
}

APFloat::opStatus
DoubleAPFloat::convertToInteger(MutableArrayRef<integerPart> Input,
                                unsigned int Width, bool IsSigned,
                                roundingMode RM, bool *IsExact) const {
  opStatus FS =
      convertToSignExtendedInteger(Input, Width, IsSigned, RM, IsExact);

  if (FS == opInvalidOp) {
    const unsigned DstPartsCount = partCountForBits(Width);
    assert(DstPartsCount <= Input.size() && "Integer too big");

    unsigned Bits;
    if (getCategory() == fcNaN)
      Bits = 0;
    else if (isNegative())
      Bits = IsSigned;
    else
      Bits = Width - IsSigned;

    tcSetLeastSignificantBits(Input.data(), DstPartsCount, Bits);
    if (isNegative() && IsSigned)
      APInt::tcShiftLeft(Input.data(), DstPartsCount, Width - 1);
  }

  return FS;
}

APFloat::opStatus DoubleAPFloat::handleOverflow(roundingMode RM) {
  switch (RM) {
  case APFloat::rmTowardZero:
    makeLargest(/*Neg=*/isNegative());
    break;
  case APFloat::rmTowardNegative:
    if (isNegative())
      makeInf(/*Neg=*/true);
    else
      makeLargest(/*Neg=*/false);
    break;
  case APFloat::rmTowardPositive:
    if (isNegative())
      makeLargest(/*Neg=*/true);
    else
      makeInf(/*Neg=*/false);
    break;
  case APFloat::rmNearestTiesToAway:
  case APFloat::rmNearestTiesToEven:
    makeInf(/*Neg=*/isNegative());
    break;
  default:
    llvm_unreachable("Invalid rounding mode found");
  }
  opStatus S = opInexact;
  if (!getFirst().isFinite())
    S = static_cast<opStatus>(S | opOverflow);
  return S;
}

APFloat::opStatus DoubleAPFloat::convertFromUnsignedParts(
    const integerPart *Src, unsigned int SrcCount, roundingMode RM) {
  // Find the most significant bit of the source integer. APInt::tcMSB returns
  // UINT_MAX for a zero value.
  const unsigned SrcMSB = APInt::tcMSB(Src, SrcCount);
  if (SrcMSB == UINT_MAX) {
    // The source integer is 0.
    makeZero(/*Neg=*/false);
    return opOK;
  }

  // Create a minimally-sized APInt to represent the source value.
  const unsigned SrcBitWidth = SrcMSB + 1;
  APSInt SrcInt{APInt{/*numBits=*/SrcBitWidth, ArrayRef(Src, SrcCount)},
                /*isUnsigned=*/true};

  // Stage 1: Initial Approximation.
  // Convert the source integer SrcInt to the Hi part of the DoubleAPFloat.
  // We use round-to-nearest because it minimizes the initial error, which is
  // crucial for the subsequent steps.
  APFloat Hi{getFirst().getSemantics()};
  Hi.convertFromAPInt(SrcInt, /*IsSigned=*/false, rmNearestTiesToEven);

  // If the first approximation already overflows, the number is too large.
  // NOTE: The underlying semantics are *more* conservative when choosing to
  // overflow because their notion of ULP is much larger. As such, it is always
  // safe to overflow at the DoubleAPFloat level if the APFloat overflows.
  if (!Hi.isFinite())
    return handleOverflow(RM);

  // Stage 2: Exact Error Calculation.
  // Calculate the exact error of the first approximation: Error = SrcInt - Hi.
  // This is done by converting Hi back to an integer and subtracting it from
  // the original source.
  bool HiAsIntIsExact;
  // Create an integer representation of Hi. Its width is determined by the
  // exponent of Hi, ensuring it's just large enough. This width can exceed
  // SrcBitWidth if the conversion to Hi rounded up to a power of two.
  // accurately when converted back to an integer.
  APSInt HiAsInt{static_cast<uint32_t>(ilogb(Hi) + 1), /*isUnsigned=*/true};
  Hi.convertToInteger(HiAsInt, rmNearestTiesToEven, &HiAsIntIsExact);
  const APInt Error = SrcInt.zext(HiAsInt.getBitWidth()) - HiAsInt;

  // Stage 3: Error Approximation and Rounding.
  // Convert the integer error into the Lo part of the DoubleAPFloat. This step
  // captures the remainder of the original number. The rounding mode for this
  // conversion (LoRM) may need to be adjusted from the user-requested RM to
  // ensure the final sum (Hi + Lo) rounds correctly.
  roundingMode LoRM = RM;
  // Adjustments are only necessary when the initial approximation Hi was an
  // overestimate, making the Error negative.
  if (Error.isNegative()) {
    if (RM == rmNearestTiesToAway) {
      // For rmNearestTiesToAway, a tie should round away from zero. Since
      // SrcInt is positive, this means rounding toward +infinity.
      // A standard conversion of a negative Error would round ties toward
      // -infinity, causing the final sum Hi + Lo to be smaller. To
      // counteract this, we detect the tie case and override the rounding
      // mode for Lo to rmTowardPositive.
      const unsigned ErrorActiveBits = Error.getSignificantBits() - 1;
      const unsigned LoPrecision = getSecond().getSemantics().precision;
      if (ErrorActiveBits > LoPrecision) {
        const unsigned RoundingBoundary = ErrorActiveBits - LoPrecision;
        // A tie occurs when the bits to be truncated are of the form 100...0.
        // This is detected by checking if the number of trailing zeros is
        // exactly one less than the number of bits being truncated.
        if (Error.countTrailingZeros() == RoundingBoundary - 1)
          LoRM = rmTowardPositive;
      }
    } else if (RM == rmTowardZero) {
      // For rmTowardZero, the final positive result must be truncated (rounded
      // down). When Hi is an overestimate, Error is negative. A standard
      // rmTowardZero conversion of Error would make it *less* negative,
      // effectively rounding the final sum Hi + Lo *up*. To ensure the sum
      // rounds down correctly, we force Lo to round toward -infinity.
      LoRM = rmTowardNegative;
    }
  }

  APFloat Lo{getSecond().getSemantics()};
  opStatus Status = Lo.convertFromAPInt(Error, /*IsSigned=*/true, LoRM);

  // Renormalize the pair (Hi, Lo) into a canonical DoubleAPFloat form where the
  // components do not overlap. fastTwoSum performs this operation.
  std::tie(Hi, Lo) = fastTwoSum(Hi, Lo);
  Floats[0] = std::move(Hi);
  Floats[1] = std::move(Lo);

  // A final check for overflow is needed because fastTwoSum can cause a
  // carry-out from Lo that pushes Hi to infinity.
  if (!getFirst().isFinite())
    return handleOverflow(RM);

  // The largest DoubleAPFloat must be canonical. Values which are larger are
  // not canonical and are equivalent to overflow.
  if (getFirst().isFiniteNonZero() && Floats[0].isLargest()) {
    DoubleAPFloat Largest{*Semantics};
    Largest.makeLargest(/*Neg=*/false);
    if (compare(Largest) == APFloat::cmpGreaterThan)
      return handleOverflow(RM);
  }

  // The final status of the operation is determined by the conversion of the
  // error term. If Lo could represent Error exactly, the entire conversion
  // is exact. Otherwise, it's inexact.
  return Status;
}

APFloat::opStatus DoubleAPFloat::convertFromAPInt(const APInt &Input,
                                                  bool IsSigned,
                                                  roundingMode RM) {
  const bool NegateInput = IsSigned && Input.isNegative();
  APInt API = Input;
  if (NegateInput)
    API.negate();

  const APFloat::opStatus Status =
      convertFromUnsignedParts(API.getRawData(), API.getNumWords(), RM);
  if (NegateInput)
    changeSign();
  return Status;
}

unsigned int DoubleAPFloat::convertToHexString(char *DST,
                                               unsigned int HexDigits,
                                               bool UpperCase,
                                               roundingMode RM) const {
  assert(Semantics == &APFloatBase::semPPCDoubleDouble &&
         "Unexpected Semantics");
  return APFloat(APFloatBase::semPPCDoubleDoubleLegacy, bitcastToAPInt())
      .convertToHexString(DST, HexDigits, UpperCase, RM);
}

bool DoubleAPFloat::isDenormal() const {
  return getCategory() == fcNormal &&
         (Floats[0].isDenormal() || Floats[1].isDenormal() ||
          // (double)(Hi + Lo) == Hi defines a normal number.
          Floats[0] != Floats[0] + Floats[1]);
}

bool DoubleAPFloat::isSmallest() const {
  if (getCategory() != fcNormal)
    return false;
  DoubleAPFloat Tmp(*this);
  Tmp.makeSmallest(this->isNegative());
  return Tmp.compare(*this) == cmpEqual;
}

bool DoubleAPFloat::isSmallestNormalized() const {
  if (getCategory() != fcNormal)
    return false;

  DoubleAPFloat Tmp(*this);
  Tmp.makeSmallestNormalized(this->isNegative());
  return Tmp.compare(*this) == cmpEqual;
}

bool DoubleAPFloat::isLargest() const {
  if (getCategory() != fcNormal)
    return false;
  DoubleAPFloat Tmp(*this);
  Tmp.makeLargest(this->isNegative());
  return Tmp.compare(*this) == cmpEqual;
}

bool DoubleAPFloat::isInteger() const {
  assert(Semantics == &APFloatBase::semPPCDoubleDouble &&
         "Unexpected Semantics");
  return Floats[0].isInteger() && Floats[1].isInteger();
}

void DoubleAPFloat::toString(SmallVectorImpl<char> &Str,
                             unsigned FormatPrecision,
                             unsigned FormatMaxPadding,
                             bool TruncateZero) const {
  assert(Semantics == &APFloatBase::semPPCDoubleDouble &&
         "Unexpected Semantics");
  APFloat(APFloatBase::semPPCDoubleDoubleLegacy, bitcastToAPInt())
      .toString(Str, FormatPrecision, FormatMaxPadding, TruncateZero);
}

int DoubleAPFloat::getExactLog2Abs() const {
  // In order for Hi + Lo to be a power of two, the following must be true:
  // 1. Hi must be a power of two.
  // 2. Lo must be zero.
  if (getSecond().isNonZero())
    return INT_MIN;
  return getFirst().getExactLog2Abs();
}

int ilogb(const DoubleAPFloat &Arg) {
  const APFloat &Hi = Arg.getFirst();
  const APFloat &Lo = Arg.getSecond();
  int IlogbResult = ilogb(Hi);
  // Zero and non-finite values can delegate to ilogb(Hi).
  if (Arg.getCategory() != fcNormal)
    return IlogbResult;
  // If Lo can't change the binade, we can delegate to ilogb(Hi).
  if (Lo.isZero() || Hi.isNegative() == Lo.isNegative())
    return IlogbResult;
  if (Hi.getExactLog2Abs() == INT_MIN)
    return IlogbResult;
  // Numbers of the form 2^a - 2^b or -2^a + 2^b are almost powers of two but
  // get nudged out of the binade by the low component.
  return IlogbResult - 1;
}

DoubleAPFloat scalbn(const DoubleAPFloat &Arg, int Exp,
                     APFloat::roundingMode RM) {
  assert(Arg.Semantics == &APFloatBase::PPCDoubleDouble() &&
         "Unexpected Semantics");
  return DoubleAPFloat(APFloatBase::PPCDoubleDouble(),
                       scalbn(Arg.Floats[0], Exp, RM),
                       scalbn(Arg.Floats[1], Exp, RM));
}

DoubleAPFloat frexp(const DoubleAPFloat &Arg, int &Exp,
                    APFloat::roundingMode RM) {
  assert(Arg.Semantics == &APFloatBase::PPCDoubleDouble() &&
         "Unexpected Semantics");

  // Get the unbiased exponent e of the number, where |Arg| = m * 2^e for m in
  // [1.0, 2.0).
  Exp = ilogb(Arg);

  // For NaNs, quiet any signaling NaN and return the result, as per standard
  // practice.
  if (Exp == APFloat::IEK_NaN) {
    DoubleAPFloat Quiet{Arg};
    Quiet.getFirst() = Quiet.getFirst().makeQuiet();
    return Quiet;
  }

  // For infinity, return it unchanged. The exponent remains IEK_Inf.
  if (Exp == APFloat::IEK_Inf)
    return Arg;

  // For zero, the fraction is zero and the standard requires the exponent be 0.
  if (Exp == APFloat::IEK_Zero) {
    Exp = 0;
    return Arg;
  }

  const APFloat &Hi = Arg.getFirst();
  const APFloat &Lo = Arg.getSecond();

  // frexp requires the fraction's absolute value to be in [0.5, 1.0).
  // ilogb provides an exponent for an absolute value in [1.0, 2.0).
  // Increment the exponent to ensure the fraction is in the correct range.
  ++Exp;

  const bool SignsDisagree = Hi.isNegative() != Lo.isNegative();
  APFloat Second = Lo;
  if (Arg.getCategory() == APFloat::fcNormal && Lo.isFiniteNonZero()) {
    roundingMode LoRoundingMode;
    // The interpretation of rmTowardZero depends on the sign of the combined
    // Arg rather than the sign of the component.
    if (RM == rmTowardZero)
      LoRoundingMode = Arg.isNegative() ? rmTowardPositive : rmTowardNegative;
    // For rmNearestTiesToAway, we face a similar problem. If signs disagree,
    // Lo is a correction *toward* zero relative to Hi. Rounding Lo
    // "away from zero" based on its own sign would move the value in the
    // wrong direction. As a safe proxy, we use rmNearestTiesToEven, which is
    // direction-agnostic. We only need to bother with this if Lo is scaled
    // down.
    else if (RM == rmNearestTiesToAway && SignsDisagree && Exp > 0)
      LoRoundingMode = rmNearestTiesToEven;
    else
      LoRoundingMode = RM;
    Second = scalbn(Lo, -Exp, LoRoundingMode);
    // The rmNearestTiesToEven proxy is correct most of the time, but it
    // differs from rmNearestTiesToAway when the scaled value of Lo is an
    // exact midpoint.
    // NOTE: This is morally equivalent to roundTiesTowardZero.
    if (RM == rmNearestTiesToAway && LoRoundingMode == rmNearestTiesToEven) {
      // Re-scale the result back to check if rounding occurred.
      const APFloat RecomposedLo = scalbn(Second, Exp, rmNearestTiesToEven);
      if (RecomposedLo != Lo) {
        // RoundingError tells us which direction we rounded:
        //   - RoundingError > 0: we rounded up.
        //   - RoundingError < 0: we down up.
        const APFloat RoundingError = RecomposedLo - Lo;
        // Determine if scalbn(Lo, -Exp) landed exactly on a midpoint.
        // We do this by checking if the absolute rounding error is exactly
        // half a ULP of the result.
        const APFloat UlpOfSecond = harrisonUlp(Second);
        const APFloat ScaledUlpOfSecond =
            scalbn(UlpOfSecond, Exp - 1, rmNearestTiesToEven);
        const bool IsMidpoint = abs(RoundingError) == ScaledUlpOfSecond;
        const bool RoundedLoAway =
            Second.isNegative() == RoundingError.isNegative();
        // The sign of Hi and Lo disagree and we rounded Lo away: we must
        // decrease the magnitude of Second to increase the magnitude
        // First+Second.
        if (IsMidpoint && RoundedLoAway)
          Second.next(/*nextDown=*/!Second.isNegative());
      }
    }
    // Handle a tricky edge case where Arg is slightly less than a power of two
    // (e.g., Arg = 2^k - epsilon). In this situation:
    // 1. Hi is 2^k, and Lo is a small negative value -epsilon.
    // 2. ilogb(Arg) correctly returns k-1.
    // 3. Our initial Exp becomes (k-1) + 1 = k.
    // 4. Scaling Hi (2^k) by 2^-k would yield a magnitude of 1.0 and
    //    scaling Lo by 2^-k would yield zero. This would make the result 1.0
    //    which is an invalid fraction, as the required interval is [0.5, 1.0).
    // We detect this specific case by checking if Hi is a power of two and if
    // the scaled Lo underflowed to zero. The fix: Increment Exp to k+1. This
    // adjusts the scale factor, causing Hi to be scaled to 0.5, which is a
    // valid fraction.
    if (Second.isZero() && SignsDisagree && Hi.getExactLog2Abs() != INT_MIN)
      ++Exp;
  }

  APFloat First = scalbn(Hi, -Exp, RM);
  return DoubleAPFloat(APFloatBase::PPCDoubleDouble(), std::move(First),
                       std::move(Second));
}

// class HexFloatArith implements HFP arithmetic using the conventions
// and approaches of the arith library used by the IBM XL compiler,
// and matches the behaviour of the hardware.
class HexFloatArith {
public:
  struct value_t {
    int sign; // -1 for negative, +1 for positive
    int exponent;
    APInt fraction;
  };
  static void fetch(const HexFloat &in, value_t &out);
  static void align(value_t &, value_t &, bool sticky = false);
  static void add(value_t &, const value_t &);
  static void sub(value_t &, const value_t &);
  static void mult(value_t &, const value_t &);
  static void divide(value_t &, const value_t &);
  static void norm(value_t &);
  static int putres(const value_t &, HexFloat &);
};

unsigned int HexFloat::getNumPrecisionBits(const fltSemantics *semantics) {
  assert(APFloat::usesLayout<HexFloat>(*semantics) && "not a HexFloat");
  return 4 * semantics->precision;
}

void HexFloat::initialize(const fltSemantics *ourSemantics) {
  semantics = ourSemantics;
  significand = APInt(getNumPrecisionBits(semantics), 0);
  makeZero(/* IsNegative */ false);
}

void HexFloat::assign(const HexFloat &rhs) {
  assert(semantics == rhs.semantics);
  sign = rhs.sign;
  exponent = rhs.exponent;
  significand = rhs.significand;
  low_sign = rhs.low_sign;
  low_exponent = rhs.low_exponent;
}

HexFloat::HexFloat(const fltSemantics &ourSemantics)
    : semantics(&ourSemantics) {
  initialize(&ourSemantics);
}

HexFloat::HexFloat(const fltSemantics &ourSemantics, integerPart intValue)
    : semantics(&ourSemantics) {
  initialize(&ourSemantics);
  static_assert(sizeof(intValue) <= sizeof(uint64_t));
  if (!intValue) {
    // initialize will have set us to zero
    return;
  }

  // intValue is regarded as an integer.
  // Therefore its radix point is to the right of intValue.
  // HexFloat has the radix point to the left of the significand,
  // therefore we initialize the exponent to the number of hexits
  // intValue could contain.  We then normalize, decremeting exponent
  // each time we shift.

  APInt working_significand(sizeof(intValue) * 8, intValue);
  APInt mask(working_significand.getBitWidth(), 0);
  mask.setBit(mask.getBitWidth() - 4); // lowest bit of top hexit
  // the corresponding constructor for IEEE seems to assume that the
  // value cannot be negative.
  sign = 0;
  exponent = (sizeof(intValue) * 8) / 4;
  // normalize
  while (mask.ugt(working_significand)) {
    working_significand <<= 4;
    exponent--;
  }
  int delta_width =
      working_significand.getBitWidth() - significand.getBitWidth();
  if (delta_width > 0) {
    // HexFloat always rounds towards 0, so truncate is adequate
    // APInt:trunc truncates on left
    working_significand = working_significand.lshr(delta_width);
    working_significand = working_significand.trunc(significand.getBitWidth());
  } else if (delta_width < 0) {
    // APInt::zext extends on left
    working_significand = working_significand.zext(significand.getBitWidth());
    working_significand <<= -delta_width;
  }
  significand = working_significand;
}

HexFloat::HexFloat(const fltSemantics &ourSemantics, uninitializedTag)
    : semantics(&ourSemantics) {
  initialize(&ourSemantics);
}

HexFloat::HexFloat(const fltSemantics &ourSemantics,
                   const APInt &EncodedHexFloat)
    : semantics(&ourSemantics) {
  initialize(&ourSemantics);
  assert(semantics == &ourSemantics && "semantics not initialized as expected");

  assert(EncodedHexFloat.getBitWidth() == semantics->sizeInBits &&
         "EncodedHexFloat has unexpected size");

  auto get_sign_exponent = [](const APInt &i, int &s, int &e) {
    auto left_byte = i.getHiBits(8).trunc(8).getLimitedValue();
    s = (left_byte & 0x80) ? 1 : 0;
    e = (left_byte & 0x7f) - 64;
  };
  int s, e;
  get_sign_exponent(EncodedHexFloat, s, e);
  sign = s;
  exponent = e;
  if (semantics == &APFloatBase::HexFP128()) {
    // we need to remove the sign/exponent byte from the lower order 64 bit
    // value, and save them.
    // We then need to form the significand from the low 56 bits of each part.
    APInt low(EncodedHexFloat.trunc(64));
    APInt high(EncodedHexFloat.lshr(64).trunc(64));
    APInt low_significand(low.trunc(56).zext(112));
    APInt high_significand(high.trunc(56).zext(112));
    high_significand <<= 56;
    significand = high_significand | low_significand;
    get_sign_exponent(low, s, e);
    low_sign = s;
    low_exponent = e;
  } else {
    auto NumPrecisionBits = getNumPrecisionBits(semantics);
    significand =
        EncodedHexFloat.getLoBits(NumPrecisionBits).trunc(NumPrecisionBits);
  }
  assert(significand.getBitWidth() == getNumPrecisionBits(semantics));
}

HexFloat::HexFloat(double d) {
  semantics = &APFloatBase::HexFP64();
  initialize(semantics);
  llvm_unreachable("HexFloat constructor double: cannot create from double\n");
}

HexFloat::HexFloat(float f) {
  semantics = &APFloatBase::HexFP32();
  initialize(semantics);
  llvm_unreachable("HexFloat constructor float: cannot create from float\n");
}

HexFloat::HexFloat(const HexFloat &rhs) {
  initialize(rhs.semantics);
  assign(rhs);
}

HexFloat::HexFloat(HexFloat &&rhs) { *this = std::move(rhs); }

HexFloat::~HexFloat() {}

HexFloat &HexFloat::operator=(const HexFloat &rhs) {
  if (this != &rhs) {
    if (semantics != rhs.semantics) {
      initialize(rhs.semantics);
    }
    assign(rhs);
  }
  return *this;
}

HexFloat &HexFloat::operator=(HexFloat &&rhs) {
  if (this != &rhs) {
    if (semantics != rhs.semantics) {
      initialize(rhs.semantics);
    }
    assign(rhs);
  }
  return *this;
}

void HexFloatArith::fetch(const HexFloat &hf, value_t &v) {
  v.sign = hf.sign ? -1 : 1;
  v.exponent = hf.exponent;
  // 4 bits each for the guard hexit on the right,
  // and possible carry on the left.
  v.fraction = hf.significand.zext(hf.significand.getBitWidth() + 8);
  // zext adds on left.
  v.fraction <<= 4;
  if (v.fraction.isZero())
    v.sign = 0;
}

void HexFloatArith::align(value_t &left, value_t &right,
                          bool sticky /* = false */) {
  // align by increasing the lower exponent to the match the higher
  // sticky indicates that the least significant bit should be set if
  // if any 1s are shifted out on the right.
  assert((left.fraction.getBitWidth() == right.fraction.getBitWidth()) &&
         "left and right significands have different sizes");

  int nhexits = left.fraction.getBitWidth() / 4; // 4 bits per hexit
  int delta_exp = left.exponent - right.exponent;
  value_t *value_to_align, *other;

  if (delta_exp == 0)
    return;

  if (delta_exp > 0) {
    // the value to align is the right
    value_to_align = &right;
    other = &left;
  } else {
    value_to_align = &left;
    other = &right;
    delta_exp = -delta_exp;
  }
  value_to_align->exponent = other->exponent;

  if (delta_exp >= (nhexits - 1)) {
    value_to_align->fraction.clearAllBits();
    return;
  }

  int placesToShift = delta_exp * 4;

  if (sticky) {
    int numTrailingZeros = value_to_align->fraction.countTrailingZeros();
    sticky = (numTrailingZeros < placesToShift);
  }

  value_to_align->fraction = value_to_align->fraction.lshr(placesToShift);

  if (sticky)
    value_to_align->fraction.setBit(0);
}

void HexFloatArith::norm(value_t &v) {
  if (v.fraction.isZero()) {
    v.sign = 0;
    v.exponent = 0;
    return;
  }
  APInt t(v.fraction.getBitWidth(), 0); // t will be used to hold
                                        // a value that v will
                                        // be compared with

  // check for carry
  t.setBit(t.getBitWidth() - 4); // the low bit of the top hexit,
                                 // i.e., where the carry is
  if (v.fraction.uge(t)) {
    v.fraction = v.fraction.lshr(4);
    v.exponent++;
    return;
  }

  // general case --- normalize
  t.clearAllBits();
  t.setBit(t.getBitWidth() - 8); // lowest bit of the top hexit of the value
  while (v.fraction.ult(t)) {
    v.fraction <<= 4;
    v.exponent--;
  }
}

int HexFloatArith::putres(const value_t &v, HexFloat &result) {
  // This is the inverse of "fetch", and is responsible for converting
  // the (internal) arith representation to a HexFloat.

  int ret_val = v.sign;
  if (v.exponent < -64) {
    // underflow --- set to 0.0
    result.makeZero(v.sign < 0);
    return 0;
  } else if (v.exponent > 1024) {
    /* This indicates division by zero (see HexFloatArith::divide).  */
    /* In this case the arith putres takes the first input operand   */
    /* as the result.  The first operand is result (see the control  */
    /* flow in HexFloat::divide).  Thus there is  nothing to do      */
    /* here, as input object to the HexFloat routine (i.e., the      */
    /* HexFloat::divide's *this object) hasn't been modified.        */
    return ret_val;
  }

  // shift right to eliminate the guard hexit on the right.
  APInt fraction = v.fraction.lshr(4);
  fraction = fraction.trunc(fraction.getBitWidth() - 8);
  assert((fraction.getBitWidth() == result.significand.getBitWidth()) &&
         "fraction has unexpected width");
  result.significand = fraction;
  if (result.significand.isZero()) {
    result.makeZero(v.sign < 0);
    ret_val = 0;
  } else {
    assert(v.sign != 0 && "v.sign is unexpectedly zero");
    result.sign = v.sign < 0 ? 1 : 0;
    if (v.exponent > 63) {
      // silently wrapround the exponent
      result.exponent = v.exponent % 64;
    } else {
      // underflow case, i.e., exppnent < -64 was dealt with above
      result.exponent = v.exponent;
    }
    if (result.semantics == &APFloatBase::HexFP128()) {
      result.low_exponent = (result.exponent - 14) & 0x7f;
      result.low_sign = result.sign;
    }
  }
  return ret_val;
}

void HexFloatArith::add(value_t &left, const value_t &right) {
  assert(left.fraction.countLeadingZeros() >= 4 &&
         "expect top hexit of left to be zero");
  assert(right.fraction.countLeadingZeros() >= 4 &&
         "expect top hexit of right to be zero");
  assert((((1 == left.sign) && (-1 != right.sign)) ||
          ((-1 == left.sign) && (1 != right.sign)) || (0 == left.sign)) &&
         "add expects values to have same sign or be zero");
  assert(left.exponent == right.exponent &&
         "add expects values to have the same exponent");
  left.sign |= right.sign; // can't have +1 and -1, and | deals with 0
  left.fraction += right.fraction;
}

void HexFloatArith::sub(value_t &left, const value_t &right) {
  assert(left.exponent == right.exponent &&
         "sub expects values to have the same exponent");
  assert(left.fraction.countl_zero() >= 4 &&
         "top (carry/guard) hexit of left unexpectedly not zero");
  assert(right.fraction.countl_zero() >= 4 &&
         "top (carry/guard) hexit of right unexpectedly not zero");
  left.fraction -= right.fraction;
  APInt t(left.fraction.getBitWidth(), 0);
  t.setHighBits(1);
  if (left.fraction.uge(t)) {
    left.fraction.negate();
    // The following assignment is safe, in that we know that right.sign cannot
    // be zero.  We know this for the following reason.
    // The top hexit of the fraction is for carry/borrow.
    // On entry this hexit must have been zero (we asserted that above).
    // The only way the hexit could become set is if
    // right.fraction > left.fraction.
    // That means that right.fraction must be > 0,
    // which means that right.sign cannot be 0.
    assert(right.sign != 0 && "right.sign unexpectedly zero");
    left.sign = -right.sign;
  }
}

void HexFloatArith::mult(value_t &left, const value_t &right) {
  // This needs a little care.  The intermediate product needs to
  // be twice as large as the arguments.  However, recall that
  // in the representation here there is a guard hexit on the right,
  // and a hexit on the left for a possible carry.
  // APInt requires that operands and results have the same size.
  // We therefore need to extend the size of the operands before
  // working on them.
  // Result is placed in left.
  int extend_to = 2 * (left.fraction.getBitWidth() - 4) + 4;
  int extend_by = extend_to - left.fraction.getBitWidth();
  APInt L = left.fraction.zext(extend_to);
  APInt R = right.fraction.zext(extend_to);
  APInt prod(L * R);

  // The assignment below is safe.
  // If either left.sign or right.sign is zero, the result sign is zero.
  // That's ok. because the fraction will also be zero.
  assert((left.sign || left.fraction.isZero()) &&
         "left.sign is zero, but fraction is not");
  assert((right.sign || right.fraction.isZero()) &&
         "right.sign is zero, but fraction is not");
  left.sign = left.sign * right.sign;
  left.exponent = left.exponent + right.exponent;
  left.fraction = prod.lshr(extend_by).trunc(left.fraction.getBitWidth());
}

/// divide numerator by divisor, leaving the result in numerator.
/// numerator and divisor are assumed to be normalized.
void HexFloatArith::divide(value_t &numerator, const value_t &divisor_in) {
  value_t divisor = divisor_in;
  APInt t(divisor.fraction.getBitWidth(), 0);

  // check if dividing by 0
  if (0 == divisor.sign) {
    /* divisor is zero */
    numerator.exponent += 2048;
    // want to set all hexits in the fraction to f, with the
    // exception of the left most, which should be 0.
    // APInt has setHighBits, but no corresponding clearHighBits
    numerator.fraction.clearAllBits();
    numerator.fraction.setLowBits(numerator.fraction.getBitWidth() - 4);
    return;
  }

  // check if numerator is 0.  If it is, then the result is 0.
  if (0 == numerator.sign)
    return;

  // bin-normalize the divisor
  // The loop is safe, i.e., will terminate, because at this point
  // we know the diviosr is not zero.
  // We assert for sanity.
  // We also know that because the divisor is normalized the loop
  // will not iterate more than 3 times.
  // Numerator left-most bits are not lost, and simply shift into the carry
  // hexit.  We deal with that after the loop by renormalizing.
  assert(!divisor.fraction.isZero() && "divisor unexpectedly zero");
  t.clearAllBits();
  t.setBit(t.getBitWidth() - 4 - 1);
  while (divisor.fraction.ult(t)) {
    numerator.fraction <<= 1;
    divisor.fraction <<= 1;
  }

  // Check if need to renormalize the numerator
  t.clearAllBits();
  t.setBit(t.getBitWidth() - 4); // low bit of highest hexit, i.e., the carry
  if (numerator.fraction.uge(t)) {
    numerator.fraction = numerator.fraction.lshr(4);
    numerator.exponent++;
  }

  int extend_to = 2 * (numerator.fraction.getBitWidth() - 4) + 4;
  int extend_by = extend_to - numerator.fraction.getBitWidth();
  APInt top = numerator.fraction.zext(extend_to);
  APInt bottom = divisor.fraction.zext(extend_to);
  top <<= extend_by;
  APInt q = top.udiv(bottom);

  numerator.sign = numerator.sign * divisor.sign;
  numerator.exponent = numerator.exponent - divisor.exponent;
  numerator.fraction = q.trunc(numerator.fraction.getBitWidth());
}

opStatus HexFloat::add(const HexFloat &RHS, roundingMode RM) {
  HexFloatArith::value_t left, right;

  HexFloatArith::fetch(*this, left);
  HexFloatArith::fetch(RHS, right);
  HexFloatArith::align(left, right);
  if ((left.sign != 0) && (right.sign != 0) && (left.sign != right.sign)) {
    right.sign = -right.sign;
    HexFloatArith::sub(left, right);
  } else {
    HexFloatArith::add(left, right);
  }
  HexFloatArith::norm(left);
  HexFloatArith::putres(left, *this);

  return opOK;
}

opStatus HexFloat::subtract(const HexFloat &RHS, roundingMode RM) {
  HexFloatArith::value_t left, right;

  HexFloatArith::fetch(*this, left);
  HexFloatArith::fetch(RHS, right);
  HexFloatArith::align(left, right);
  if ((left.sign != 0) && (right.sign != 0) && (left.sign != right.sign)) {
    right.sign = left.sign;
    HexFloatArith::add(left, right);
  } else {
    HexFloatArith::sub(left, right);
  }
  HexFloatArith::norm(left);
  HexFloatArith::putres(left, *this);

  return opOK;
}

opStatus HexFloat::multiply(const HexFloat &RHS, roundingMode RM) {
  HexFloatArith::value_t left, right;

  HexFloatArith::fetch(*this, left);
  HexFloatArith::fetch(RHS, right);
  HexFloatArith::norm(left);
  HexFloatArith::norm(right);
  HexFloatArith::mult(left, right);
  HexFloatArith::norm(left);
  HexFloatArith::putres(left, *this);

  return opOK;
}

opStatus HexFloat::divide(const HexFloat &RHS, roundingMode RM) {
  HexFloatArith::value_t numerator, divisor;

  HexFloatArith::fetch(*this, numerator);
  HexFloatArith::fetch(RHS, divisor);
  HexFloatArith::norm(numerator);
  HexFloatArith::norm(divisor);
  HexFloatArith::divide(numerator, divisor);
  HexFloatArith::norm(numerator);
  HexFloatArith::putres(numerator, *this);

  return opOK;
}

opStatus HexFloat::remainder(const HexFloat &RHS) {
  llvm_unreachable("TODO: HexFloat::remainder not yet implemented");
}

opStatus HexFloat::mod(const HexFloat &RHS) {
  llvm_unreachable("TODO: HexFloat::mod not yet implemented");
}

opStatus HexFloat::fusedMultiplyAdd(const HexFloat &Multiplicand,
                                    const HexFloat &Addend, roundingMode RM) {
  auto widen = [](HexFloatArith::value_t &v) {
    int widthIn = v.fraction.getBitWidth();
    int widthOut = 2 * widthIn;
    v.fraction = v.fraction.zext(widthOut);
    v.fraction <<= (widthOut - widthIn);
  };

  auto narrow = [](HexFloatArith::value_t &v) {
    int widthIn = v.fraction.getBitWidth();
    int widthOut = widthIn / 2;
    v.fraction = v.fraction.getHiBits(widthOut).trunc(widthOut);
  };

  HexFloatArith::value_t m1, m2, a;

  // fetch operands
  HexFloatArith::fetch(*this, m1);
  HexFloatArith::fetch(Multiplicand, m2);
  HexFloatArith::fetch(Addend, a);

  // extend operands
  widen(m1);
  widen(m2);
  widen(a);

  // normalize
  HexFloatArith::norm(m1);
  HexFloatArith::norm(m2);
  HexFloatArith::norm(a);

  // Multiply and normalize.  (Need normalize to prevent precision loss
  // during forthcoming add.)
  HexFloatArith::mult(m1, m2);
  HexFloatArith::norm(m1);

  // align in preparation for the add
  HexFloatArith::align(m1, a, /*sticky=*/true);

  // now do the addition
  if ((m1.sign != 0) && (a.sign != 0) && (m1.sign != a.sign)) {
    a.sign = -a.sign;
    HexFloatArith::sub(m1, a);
  } else {
    HexFloatArith::add(m1, a);
  }
  // normalize
  HexFloatArith::norm(m1);

  // narrow the result
  narrow(m1);

  // unpack
  HexFloatArith::putres(m1, *this);

  return opOK;
}

APFloat::fltCategory HexFloat::getCategory() const {
  // HexFloat does not have NaN or Infinity
  if (isZero())
    return fcZero;
  return fcNormal;
}

void HexFloat::makeZero(bool Neg) {
  // we set the exponent to minExponent so that when the bias is added,
  // the biased field is 0; i.e., the encoded representation is all zeros
  // with the exception of the sign bit which may be one to represent -0.0.
  sign = Neg ? 1 : 0;
  exponent = -64;
  low_sign = 0;
  low_exponent = -64;
  significand.clearAllBits();
}

void HexFloat::makeInf(bool Neg) {
  // HexFloat has no special indicator for infinity
  makeLargest(Neg);
}

void HexFloat::makeNaN(bool SNaN, bool Neg, const APInt *fill) {
  // HexFloat doesn't have the concept of NaN, but we need to so something
  // if for no other reason than preventing the unittests from failing.
  // FIXME: ideally the APFloat API should have predicates that indicate
  //        whether a floating point class supports NaN, and other concepts.
  makeZero(false);
}

void HexFloat::makeLargest(bool Neg) {
  sign = Neg ? 1 : 0;
  exponent = semantics->maxExponent;
  significand.setAllBits();
}

void HexFloat::makeSmallest(bool Neg) {
  // this is a denormal
  sign = Neg ? 1 : 0;
  exponent = semantics->minExponent;
  significand.clearAllBits();
  significand.setBit(0);
}

void HexFloat::makeSmallestNormalized(bool Neg) {
  sign = Neg ? 1 : 0;
  exponent = semantics->minExponent;
  significand.clearAllBits();
  significand.setBit(getNumPrecisionBits() - 4);
}

bool HexFloat::needsCleanup() const {
  // All our members but significand are trivally destructable,
  // needsCleanup depends on significand
  return significand.needsCleanup();
}

opStatus HexFloat::roundToIntegral(roundingMode RM) {
  if (significand.isZero()) {
    return opOK;
  }

  // Check to see if the value is already an integer.
  // If the exponent is at least as large as the number of
  // hexits in the significand then the value is definitely
  // an integer.
  // If the exponent is less than the maximum hexit width,
  // the value could be an integer if it has trailing zeros.
  int hexitWidth = significand.getBitWidth() / 4; // max possible hexits
  int hexitTrailingZeros = significand.countr_zero() / 4;

  // we know that the significand is not zero, whch means that
  // the number of trailing zero hexits must be strictly less
  // than the maximum number of hexits.
  assert(hexitTrailingZeros < hexitWidth);

  if (exponent >= (hexitWidth - hexitTrailingZeros))
    return opOK;

  // Now the non-trivial cases.

  if (exponent < 1) {
    // then the absolute value must be strictly less than one.
    // THerefore the rounded value will be -1, 0, or 1,
    // depending on the mode and the value.
    // We also know that the significand is not zero
    auto makeOne = [&]() {
      // sign stays what it was
      // 1 has an exponent of 1, and the top hexit is 1,
      // which in binary is 0001.
      exponent = 1;
      significand.clearAllBits();
      significand.setBit(significand.getBitWidth() - 4);
    };

    // for an exponent < 1, half must be represented by
    // exponent = 0, and only the top bit of the significand set.
    // We will need this constant for resolving ties for some modes.
    APInt halfSignificand(significand);
    halfSignificand.clearAllBits();
    halfSignificand.setBit(halfSignificand.getBitWidth() - 1);

    switch (RM) {
    case rmTowardZero:
      makeZero(sign);
      break;
    case rmTowardPositive:
      if (sign) {
        // negative, so rounding up comes to zero
        makeZero(sign);
      } else {
        // make one
        makeOne();
      }
      break;
    case rmTowardNegative:
      if (sign) {
        // negative, so round to -1
        makeOne();
      } else {
        makeZero(sign);
      }
      break;
    case rmNearestTiesToEven:
      if ((exponent < 0) || significand.ule(halfSignificand)) {
        // if exponent < 0, then the absolute value must be less than a half,
        // so we round to zero.`.
        // if the exponent is not less than zero, then it must be 0 because
        // we know that it is less than 1.
        // If the exponent is 0, then we want to round to 0 if the value is a
        // half.
        makeZero(sign);
      } else {
        assert(exponent == 0);
        makeOne();
      }
      break;
    case rmNearestTiesToAway:
      if ((exponent < 0) || significand.ult(halfSignificand)) {
        makeZero(sign);
      } else {
        assert(exponent == 0);
        makeOne();
      }
      break;
    default:
      llvm_unreachable("unknwon rounding mode");
    }
    return opInexact;
  }

  // From here on the exponent must >= 1
  assert(exponent >= 1);

  // Because of the tests above, the exponent must be strictly
  // less than the number of hexits in the significand
  assert(exponent < hexitWidth);

  // We also know this:
  assert(exponent < (hexitWidth - hexitTrailingZeros));

  // find the position of the radix point
  // An exponent >= 1 means that there is at least one hexit
  // to the left of the radix point.
  // From the tests above, we also know that there must be
  // at least one hexit to the right of the radix point.
  int hexitsToRightOfPoint = hexitWidth - exponent;
  assert(hexitsToRightOfPoint >= 1);
  assert(hexitsToRightOfPoint < hexitWidth);

  APInt fractionMask(significand);
  fractionMask.clearAllBits();
  fractionMask.setLowBits(4 * hexitsToRightOfPoint);

  APInt integerMask(fractionMask);
  integerMask.flipAllBits();

  APInt fraction(significand);
  fraction &= fractionMask;

  APInt topBitOfFractionMask(fractionMask);
  topBitOfFractionMask.clearLowBits(4 * hexitsToRightOfPoint - 1);

  APInt bottomBitOfIntegerMask(topBitOfFractionMask);
  bottomBitOfIntegerMask <<= 1;

  APInt integer(significand);
  integer &= integerMask;

  if (!fraction.isZero()) {
    // only have anyting to do if the fraction part is not zero
    auto increment = [RM](HexFloat *lhs, const APInt &rhs) {
      HexFloat other(*lhs);
      other.significand = rhs;
      other.sign = 0;
      lhs->add(other, RM);
    };
    auto decrement = [RM](HexFloat *lhs, const APInt &rhs) {
      HexFloat other(*lhs);
      other.significand = rhs;
      other.sign = 0;
      lhs->subtract(other, RM);
    };
    switch (RM) {
    case rmTowardZero:
      // truncate away everything to right of radix point
      significand = integer;
      break;
    case rmTowardPositive:
      // negative:   -11.3 ==> -11, and -13.7 ==> -13.
      //    i.e., truncate away the fraction
      // positive:    11.3 ==> 12, and 13.7 ==> 14
      //    i.e., truncate away the fraction, and increment integer;
      // In either case, truncate away the fraction
      significand = integer;
      if (!sign) {
        // i.e., positive
        increment(this, bottomBitOfIntegerMask);
      }
      break;
    case rmTowardNegative:
      // negative:   -11.3 ==> -12, and -13.7 ==> -14.
      //   i.e., truncate away the fraction, and decrement integer
      // positive:    11.3 ==> 11, and 13.7 ==> 13
      //   i.e., truncate away the fraction
      // In either case, truncate away the fraction.
      significand = integer;
      if (sign) {
        // i.e., negative
        decrement(this, bottomBitOfIntegerMask);
      }
      break;
    case rmNearestTiesToEven:
      if (sign) {
        // -11.5 ==> -12  and -12.5 ==> -12
        significand = integer;
        if (fraction.ult(topBitOfFractionMask)) {
          // e.g., -11.3, which goes to -11
        } else if (fraction.ugt(topBitOfFractionMask)) {
          // e.g., -12.7, which should go to -13
          decrement(this, bottomBitOfIntegerMask);
        } else if (integer.intersects(bottomBitOfIntegerMask)) {
          // then we had a tie, and the interm result is odd.
          // Need to subtract one.
          decrement(this, bottomBitOfIntegerMask);
        }
      } else {
        // 11.5 ==> 12   abd   12.5 ==> 12
        significand = integer;
        if (fraction.ult(topBitOfFractionMask)) {
          // e.g., 12.3  ==>  12
        } else if (fraction.ugt(topBitOfFractionMask)) {
          // e.g.  12.7  ==>  13
          increment(this, bottomBitOfIntegerMask);
        } else if (integer.intersects(bottomBitOfIntegerMask)) {
          // have tie and interm integer is odd
          increment(this, bottomBitOfIntegerMask);
        }
      }
      break;
    case rmNearestTiesToAway:
      if (sign) {
        significand = integer;
        if (fraction.ult(topBitOfFractionMask)) {
          // e.g.,  -11.4  ==>  -11
        } else {
          // e.g.,  -11.5  ==>  -12   and  -11.6  ==>  -12
          decrement(this, bottomBitOfIntegerMask);
        }
      } else {
        significand = integer;
        if (fraction.ult(topBitOfFractionMask)) {
          // e.g.,  11.4  ==>  11
        } else {
          // e.g.,  11.5  ==>  12   and  11.6  ==>  12
          increment(this, bottomBitOfIntegerMask);
        }
      }
      break;
    default:
      llvm_unreachable("unknown rounding mode");
    }
  }
  // if the significand is zero, make us a canonical zero
  if (significand.isZero())
    makeZero(isNegative());

  return opInexact;
}

opStatus HexFloat::next(bool nextDown) {
  auto do_increment = [&]() {
    // incrementing the significand will cause a carry iff all bits are 1
    if (significand.isAllOnes()) {
      // then the increment would cause a carry
      significand.clearAllBits();
      // set the most significant hexit to 1.
      significand.setBit(significand.getBitWidth() - 4);
      exponent++;
      if (exponent > 64)
        exponent = exponent % 64;
    } else {
      // the easy case --- just increment the significand
      significand += 1;
    }
  };
  auto do_decrement = [&]() {
    // There are two cases where it would be unsafe to decrement:
    // + if the significand were sll zeros.  This happens only when
    //   the value is zero.  This is checked for and handled separately.
    // + the value is all zeros except for the least significant
    //   (i.e., the value is smallest denormal).  Decrementing would
    //    result in zero.  This case, too, is handled specially.
    // Thus it is safe to decrement.
    // We assume that the value is currently as normal as possible.
    significand -= 1;
    assert(!significand.isZero() && "significand is unexpectedly zero");
    // normalize if possible
    if (exponent > -64 && significand.countLeadingZeros() >= 4) {
      // the top hexit is now zero.
      // Shift up.  We never need to shift more than one hexit.
      significand <<= 4;
      exponent--;
    }
  };

  // Note that HexFloat does not have special representations for NaN
  // or infinity
  if (!nextDown) {
    // nextUp
    // deal with the special cases first
    if (isZero()) {
      makeSmallest(/* is neg */ false);
    } else if (isNegative() && isSmallest()) {
      // value is the negative number with the least magnitude.
      // IEEE says return -0
      makeZero(/* is neg */ true);
    } else {
      // general case
      if (isNegative())
        do_decrement();
      else
        do_increment();
    }
  } else {
    // nextDown
    if (isZero()) {
      makeSmallest(/* is neg */ true);
    } else if (!isNegative() && isSmallest()) {
      // +ve with smallest magnitude
      makeZero(/* is neg */ false);
    } else {
      if (isNegative())
        do_increment();
      else
        do_decrement();
    }
  }
  return opOK;
}

void HexFloat::changeSign() { sign = !sign; }

opStatus HexFloat::convert(const fltSemantics &toSemantics,
                           roundingMode rounding_mode, bool *losesInfo) {
  class OnExit {
    bool *losesInfo;

  public:
    bool LostInfo = false;
    OnExit(bool *losesInfo_) : losesInfo(losesInfo_) {}
    ~OnExit() {
      if (losesInfo)
        *losesInfo = LostInfo;
    }
  };
  OnExit OnExit(losesInfo);
  opStatus fs = opOK;

  const fltSemantics &fromSemantics = *semantics;
  if (APFloat::usesLayout<HexFloat>(toSemantics)) {
    // converting from one HexFloat precision to another
    const int from_bits_precision = getNumPrecisionBits(&fromSemantics);
    const int to_bits_precision = getNumPrecisionBits(&toSemantics);
    if (to_bits_precision == from_bits_precision) {
      // nothing to do, except note no loss of info
      OnExit.LostInfo = false;
    } else if (significand.isZero()) {
      OnExit.LostInfo = false;
      significand = APInt(to_bits_precision, 0);
      low_sign = 0;
      low_exponent = -64;
    } else if (to_bits_precision > from_bits_precision) {
      /* widening */
      significand = significand.zext(to_bits_precision);
      significand <<= (to_bits_precision - from_bits_precision);
      OnExit.LostInfo = false;
      low_sign = 0;
      low_exponent = -64;
    } else {
      // from_bits_precision > to_bits_precision i.e., narrowing
      // check for loss of precision
      int num_trailing_zeros = significand.countTrailingZeros();
      // countTrailingZeros returns the width if there are no set bits,
      // i.e., if the value is zero.  But we know the value isn't zero
      // because it was checked above.
      if (num_trailing_zeros >= (from_bits_precision - to_bits_precision)) {
        // all the bits we're going to truncate are zero
        OnExit.LostInfo = false;
      } else {
        OnExit.LostInfo = true;
      }
      // When narrowing, we truncate to the new width without rounding
      significand = significand.lshr(from_bits_precision - to_bits_precision);
      significand = significand.trunc(to_bits_precision);
    }
    // finally, update the semantics
    semantics = &toSemantics;
  } else {
    llvm_unreachable("attempting to comver HexFloat to something else");
  }
  return fs;
}

/* Comparison requires normalized numbers.  */
cmpResult HexFloat::compare(const HexFloat &rhs) const {
  cmpResult result;

  assert(semantics == rhs.semantics);

  if (isZero() && rhs.isZero())
    return cmpEqual;

  /* Two normal numbers.  Do they have the same sign?  */
  if (sign != rhs.sign) {
    if (sign)
      result = cmpLessThan;
    else
      result = cmpGreaterThan;
  } else {
    /* Compare absolute values; invert result if negative.  */
    result = compareAbsoluteValue(rhs);

    if (sign) {
      if (result == cmpLessThan)
        result = cmpGreaterThan;
      else if (result == cmpGreaterThan)
        result = cmpLessThan;
    }
  }

  return result;
}

cmpResult HexFloat::compareAbsoluteValue(const HexFloat &rhs) const {
  int compare = 0;
  assert(semantics == rhs.semantics);
  compare = exponent - rhs.exponent;

  /* If exponents are equal, do an unsigned bignum comparison of the
     significands.  */
  if (compare == 0) {
    if (significand.ugt(rhs.significand))
      compare = 1;
    else if (significand.ult(rhs.significand))
      compare = -1;
  }

  if (compare > 0)
    return cmpGreaterThan;
  else if (compare < 0)
    return cmpLessThan;
  return cmpEqual;
}

bool HexFloat::bitwiseIsEqual(const HexFloat &other) const {
  if (this == &other)
    return true;
  if (semantics != other.semantics)
    return false;
  if (sign != other.sign)
    return false;
  if (exponent != other.exponent)
    return false;
  if (semantics == &APFloatBase::HexFP128()) {
    if (low_sign != other.low_sign || low_exponent != other.low_exponent)
      return false;
  }
  return significand == other.significand;
}

bool HexFloat::roundAwayFromZero(int sign, const APInt &fraction,
                                 roundingMode RM, lostFraction lost_fraction,
                                 int bit) {
  /* Current callers never pass this so we don't handle it.  */
  assert(lost_fraction != lfExactlyZero);

  switch (RM) {
  case rmNearestTiesToAway:
    return lost_fraction == lfExactlyHalf || lost_fraction == lfMoreThanHalf;

  case rmNearestTiesToEven:
    if (lost_fraction == lfMoreThanHalf)
      return true;

    if (lost_fraction == lfExactlyHalf)
      return APInt::tcExtractBit(fraction.getRawData(), bit);

    return false;

  case rmTowardZero:
    return false;

  case rmTowardPositive:
    return !sign;

  case rmTowardNegative:
    return sign;

  default:
    break;
  }
  llvm_unreachable("Invalid rounding mode found");
}

opStatus
HexFloat::convertToSignExtendedInteger(MutableArrayRef<integerPart> output,
                                       unsigned int width, bool isSigned,
                                       roundingMode RM, bool *isExact) const {
  class OnExit {
    bool *IsExact;

  public:
    bool Exact = false;
    OnExit(bool *isExact) : IsExact(isExact) {}
    ~OnExit() {
      if (IsExact)
        *IsExact = Exact;
    }
  };
  OnExit OnExit(isExact);

  unsigned int dstPartsCount = partCountForBits(width);
  assert(dstPartsCount <= output.size() &&
         "Integer width too large for output");

  if (significand.isZero()) {
    // then we don't care about the exponent
    OnExit.Exact = !isNegative();
    APInt::tcSet(output.data(), 0, dstPartsCount);
    return opOK;
  }

  // we can treat the significand as a binary fraction,
  // and adjust the exponent
  APInt fraction(significand);
  int e = exponent * 4; // base 16 to base 2

  // normalize fraction to base 2
  // the loop must terminate because we know that fraction is not zero
  while (!fraction.isSignBitSet()) {
    fraction <<= 1;
    e--;
  }
  assert(fraction.isSignBitSet() &&
         "top bit unexpectedly not set after normalization");

  // Step 1: place the absolute value of result in output,
  //         truncating as required
  int truncatedBits;
  if (e < 1) {
    // then we have a value less than one
    APInt::tcSet(output.data(), 0, dstPartsCount);
    // the number of bits truncated away is nbits  plus abs(e)
    truncatedBits = fraction.getBitWidth() - e;
  } else {
    // The integer part of the value is the top e of bits of the fraction

    // check if too large
    const unsigned int ue =
        (unsigned int)e; // ease comparisions since we know e >= 0
    if (ue > width)
      return opInvalidOp;
    if (ue < fraction.getBitWidth()) {
      truncatedBits = fraction.getBitWidth() - e;
      APInt::tcExtract(output.data(), dstPartsCount, fraction.getRawData(), e,
                       truncatedBits);
    } else {
      APInt::tcExtract(output.data(), dstPartsCount, fraction.getRawData(),
                       fraction.getBitWidth(), 0);
      APInt::tcShiftLeft(output.data(), dstPartsCount,
                         e - fraction.getBitWidth());
      truncatedBits = 0;
    }
  }

  // Step 2: work out the lost fraction, and adjust the result as
  //         dictated by the rounding mode
  assert(truncatedBits >= 0 && "truncatedButs unexpectedly negative");
  lostFraction lost_fraction;
  if (truncatedBits) {
    lost_fraction = lostFractionThroughTruncation(
        fraction.getRawData(), fraction.getNumWords(), truncatedBits);
    if (lost_fraction != lfExactlyZero &&
        roundAwayFromZero(sign, fraction, RM, lost_fraction, truncatedBits)) {

      if (APInt::tcIncrement(output.data(), dstPartsCount))
        return opInvalidOp; /* Overflow */
    }
  } else {
    lost_fraction = lfExactlyZero;
  }

  // Step 3: check result fits in destination
  unsigned int omsb = APInt::tcMSB(output.data(), dstPartsCount) + 1;
  if (sign) {
    if (!isSigned) {
      // converting negative floating point to unsigned integer
      if (!APInt::tcIsZero(output.data(), dstPartsCount))
        return opInvalidOp;
    } else {
      // converting negative floating point to signed integer

      // It takes omsb bits to represent the unsigned integer value.
      // We lose a bit for the sign.  Thus if the top bit is set
      // we can't represent the negative value, unless the value
      // is -2^n (assuming 2s-complement).
      unsigned int lsb = APInt::tcLSB(output.data(), dstPartsCount) + 1;
      if (omsb == width && lsb != omsb)
        // top bit is set, and there is at least one other set bit.
        // Hence the negative of this value cannot be repreesnted in
        // width bits
        return opInvalidOp;

      if (omsb > width)
        // this case can happen because of rounding
        return opInvalidOp;
    }
    APInt::tcNegate(output.data(), dstPartsCount);
  } else {
    // positive floating point
    if ((isSigned && (omsb >= width)) || (!isSigned && (omsb >= (width + 1))))
      // value is too large
      return opInvalidOp;
  }

  if (lost_fraction == lfExactlyZero) {
    OnExit.Exact = true;
    return opOK;
  }
  OnExit.Exact = false;
  return opInexact;
}

opStatus HexFloat::convertToInteger(MutableArrayRef<integerPart> output,
                                    unsigned int width, bool isSigned,
                                    roundingMode RM, bool *isExact) const {
  opStatus fs;
  fs = convertToSignExtendedInteger(output, width, isSigned, RM, isExact);

  if (fs == opInvalidOp) {
    unsigned int bits, dstPartsCount = partCountForBits(width);
    assert(dstPartsCount <= output.size() && "Integer too big");
    if (isNegative())
      bits = isSigned;
    else
      bits = width - isSigned;
    tcSetLeastSignificantBits(output.data(), dstPartsCount, bits);
    if (isNegative() && isSigned)
      APInt::tcShiftLeft(output.data(), dstPartsCount, width - 1);
  }
  return fs;
}

opStatus HexFloat::convertFromAPInt(const APInt &input, bool isSigned,
                                    roundingMode RM) {
  APInt api(input); // working copy
  int s, e;         // working copies of sign and exponent

  if (api.isZero()) {
    makeZero(false);
    return opOK;
  }

  if (isSigned && api.isNegative()) {
    s = 1;
    api = -api;
  } else {
    s = 0;
  }

  // round up size of api to be multiple of 4,
  // i.e., a whole number of hexits
  if (auto r = api.getBitWidth() % 4) {
    api = api.zext(api.getBitWidth() + 4 - r);
  }

  APInt mask(api.getBitWidth(), 0);
  mask.setBit(mask.getBitWidth() - 4); // lowest bit of top hexit
  // the corresponding constructor for IEEE seems to assume that the
  // value cannot be negative.
  e = api.getBitWidth() / 4; // move radix point to left of api
  // normalize
  while (mask.ugt(api)) {
    api <<= 4;
    e--;
  }
  int delta_width = api.getBitWidth() - significand.getBitWidth();
  if (delta_width > 0) {
    // the integer has more bits then the HexFloat has precision
    // We will always need to truncate away the excess.
    // We do this at the end of this just before the closing
    // brace of this block.
    // Between here and where the truncation takes place
    // we make any adjustments to the part to be retained
    // required by the rounding mode.

    auto increment = [&]() {
      bool overFlow = false;
      APInt delta(api.getBitWidth(), 0);
      delta.setBit(delta_width);
      api = api.uadd_ov(delta, overFlow);
      if (overFlow) {
        // renormalize api
        api = api.lshr(4);
        api.setBit(api.getBitWidth() - 4);
        e++;
      }
    };

    int leastSigBitPos = api.countr_zero();
    if (leastSigBitPos >= delta_width) {
      // all zeros in the excess, so just truncate
    } else {
      // the chunk to be discarded is non-zero
      APInt msbDiscard(api.getBitWidth(), 0);
      msbDiscard.setBit(delta_width - 1);
      APInt discard(api);
      // want to clear the top bits, but APInt doesn't provide this
      discard.flipAllBits();
      discard.setHighBits(discard.getBitWidth() - delta_width);
      discard.flipAllBits();

      int cmpVal;
      if (discard.ult(msbDiscard))
        cmpVal = -1;
      else if (discard.ugt(msbDiscard))
        cmpVal = 1;
      else
        cmpVal = 0;

      switch (RM) {
      case rmNearestTiesToEven:
        if (cmpVal > 0)
          increment();
        else if (cmpVal == 0) {
          // tie: need to check LSB of bit to be retained
          APInt lsb(api.getBitWidth(), 0);
          lsb.setBit(delta_width);
          if (lsb.intersects(api)) {
            // then the retained part is "odd", so need increment
            increment();
          }
        }
        break;
      case rmNearestTiesToAway:
        if (cmpVal >= 0)
          increment();
        break;
      case rmTowardPositive:
        if (!s)
          increment();
        break;
      case rmTowardNegative:
        if (s)
          increment();
        break;
      case rmTowardZero:
        // just truncate
        break;
      default:
        break;
      }
    }
    // Now do the truncation
    api = api.lshr(delta_width);
    api = api.trunc(significand.getBitWidth());
  } else if (delta_width < 0) {
    // APInt::zext extends on left
    api = api.zext(significand.getBitWidth());
    api <<= -delta_width;
  }
  assert(api.getBitWidth() == significand.getBitWidth());
  if (e < -64)
    return opUnderflow;
  else if (e > 63)
    return opOverflow;

  sign = s;
  exponent = e;
  significand = api;
  return opOK;
}

opStatus HexFloat::convertFromSignExtendedInteger(const integerPart *input,
                                                  unsigned int inputSize,
                                                  bool isSigned,
                                                  roundingMode RM) {
  llvm_unreachable("HexFloat::convertFromSignExtendedInteger not implemented");
}

opStatus HexFloat::convertFromZeroExtendedInteger(const integerPart *input,
                                                  unsigned int inputSize,
                                                  bool isSigned,
                                                  roundingMode RM) {
  llvm_unreachable("HexFloat::convertFromZeroExtendedInteger not implemented");
}

Expected<opStatus> HexFloat::convertFromString(StringRef str,
                                               roundingMode rounding_mode) {
  assert(APFloat::usesLayout<HexFloat>(*semantics) && "Unexpected Semantics");

  if (str.empty())
    return createError("Invalid string length");

  /* Handle a leading minus sign.  */
  StringRef::iterator p = str.begin();
  size_t slen = str.size();
  sign = *p == '-' ? 1 : 0;
  if (*p == '-' || *p == '+') {
    p++;
    slen--;
    if (!slen)
      return createError("String has no digits");
  }

  if (slen >= 2 && p[0] == '0' && (p[1] == 'x' || p[1] == 'X'))
    return convertFromHexadecimalString(StringRef(p + 2, slen - 2),
                                        rounding_mode);
  return convertFromDecimalString(StringRef(p, slen), rounding_mode);
}

opStatus HexFloat::handleOverflow(bool isNeg) {
  makeLargest(isNeg);
  return (opStatus)(opOverflow | opInexact);
}

opStatus HexFloat::handleUnderflow(bool isNeg) {
  makeZero(isNeg);
  return (opStatus)(opUnderflow | opInexact);
}

opStatus HexFloat::handleDenorm(APInt &significand, int &exponent) {
  if (exponent > 63)
    return handleOverflow(sign);
  while (exponent < -64) {
    if (significand.getLoBits(8) == 0) {
      significand = significand.udiv(16);
      exponent = exponent + 1;
    } else
      break;
  }
  if (exponent < -64 || significand.getHiBits(getNumPrecisionBits()).isZero())
    return handleUnderflow(sign);
  return opOK;
}

Expected<opStatus>
HexFloat::convertFromHexadecimalString(StringRef str,
                                       RoundingMode rounding_mode) {
  // we are converting from values of the form  0x1234.567p89, where
  // the radix point and the exponent are optional.
  // Note that the exponent after the "p" is to base 2, ie, in this
  // case, 2^89.
  // The main complexity in this routine is dealing with the radix point.
  // The approach taken here is to read the significand as hex, initially
  // ignoring the radix point.
  // The next task is to make the base 2 exponent a multiple of 4 to ease
  // normalization later on.
  // We make the base 2 exponent a multiple of 4 by shifting the value left
  // (which multiples by 2 with each positioo shifted) and reducing the
  // base 2 exponent to a multiple of 4.
  // At this point we can now work in terms of powers of 16 since
  // 2^4 == 16.  From here on we assume the exponent is of powers of 16.
  // We then take into account the radix point, adjusting the exponent
  // so that it is on the right of the value.
  // We then normalize the value by removing leading 0 hexits, and
  // shifting the value to the left.
  // We then make the final adjustment to the exponent by shifting it
  // to the left end; this is the number of hexits in the value.
  // Shifting the radix point left divides by 16, so the exponent must
  // be increased by one for each hexit in the value.
  //
  // In dealing with the strings, some care is needed as in some cases
  // the iterators point to an actual value, and in others to one to the right.

  assert(APFloat::usesLayout<HexFloat>(*semantics) && "Unexpected Semantics");

  significand.clearAllBits();
  exponent = 0;
  // use one additional hexit than the precision to perform rounding
  unsigned size = std::max((unsigned)str.size() * 8, getNumPrecisionBits() + 4);
  APInt tmpSignificand = APInt(size, 0);
  int exp = 0;
  // Skip leading zeroes and any (hexa)decimal point.
  StringRef::iterator begin = str.begin();
  StringRef::iterator end = str.end();
  StringRef::iterator dot = str.end();

  // ensure that the size of the working significand is an exact number of
  // hexits
  assert((tmpSignificand.getBitWidth() % 4) == 0);

  auto PtrOrErr = skipLeadingZeroesAndAnyDot(begin, end, &dot);
  if (!PtrOrErr)
    return PtrOrErr.takeError();

  StringRef::iterator firstSigDigit = *PtrOrErr;
  StringRef::iterator p;
  StringRef::iterator lastSigDigit = end;

  size_t expPosition = str.find_insensitive("p");
  if (expPosition != StringRef::npos)
    lastSigDigit = begin + expPosition;

  // lastSigDigit is one to the right of the actual last digit

  // if no radix point, set dot to the last significant digit for calculation
  // purposes
  if (dot == str.end())
    dot = lastSigDigit;

  // read the significand, ignoring any radix point.
  // This, effectively, treats the significand as an integer,
  // with the (implied) radix point immediately to the right
  // of the value.
  // As we loop, we keep track of where the dot is so that
  // the exponent can be adjusted later.
  for (p = firstSigDigit; p != lastSigDigit; p++) {
    if (*p == '.') {
      if (dot != lastSigDigit)
        return createError("String contains multiple dots");
      dot = p;
      continue;
    }
    unsigned hex = hexDigitValue(*p);
    if (hex == (unsigned)-1)
      break;
    tmpSignificand <<= 4;
    tmpSignificand += hex;
  }

  // if the significand is zero, it doesn't matter what the exponent is.
  if (tmpSignificand.isZero()) {
    makeZero(isNegative());
    return opOK;
  }

  // Note: at this point dot points either to the radix point,
  //       in which case it points to the actual position.
  //       Otherwise it is lastSigDigit, which is one beyond the actual
  //       last digit, i.e., the value is an integer.

  if (p != end) {
    auto ExpOrErr = readExponent(p + 1, end);
    if (!ExpOrErr)
      return ExpOrErr.takeError();
    exp = *ExpOrErr;
  }

  // at this point we have the base 2 exponent, either 0 if none
  // was given after a 'p', or the value given by the p.
  // Before we deal with the position of the radix point
  // in the significand, we are going to shift the significand
  // left to make the base 2 exponent a multiple of 4.
  //
  // In the following, let e be the exponent, and R = e % 4
  // We have three cases to consider:
  //   e == 0
  //     no adjustment is needed
  //   e > 0
  //     if R != 0
  //       shifting left by R multiples by 2^R,
  //       so we must reduce e by R
  //   e < 0
  //     we are guaranteed by the C/C++ standards that a/b truncates to 0,
  //     so that -7/4 == -1.
  //     We are also guaranteed that (a/b) * b + a%b == a.
  //     Thus a%b must be negative if a is -ve, and b is +ve.
  //     In this case, we shift left by R + 4, and reduce the
  //     exponent by (R + 4).
  //
  // Examples:
  //   x * 2^23
  //     R == 3
  //     x * 2^23 == x * 2^3 * 2^20 == (x << 3) * 2^20
  //   x * 2*-23
  //     R = -3
  //     x * 2^-23 == x * 2^1 * 2^-24 == (x << 1) * 2^-24

  if (int R = (exp % 4)) {
    if (R < 0)
      R += 4;
    tmpSignificand <<= R;
    exp -= R;
  }

  // At this point the exponent should br a multiple of 4
  assert(0 == (exp % 4));

  // convert to base 16 --- we can do this simply by dividing the exponent by 4
  exp = exp / 4;

  // move the radix point to the right:
  // lastSigDigit points to one beyond the mantissa (recall, lastSigDigit
  // is either 'end', or where the "p" is.  In either case, it is one
  // place to the right of actual last digit.
  // dot is either lastSigDigit if there is no dot, or where the
  // dot actually is.
  // If there is no dot, there is no adjustment to make.
  if (lastSigDigit != dot) {
    // examples:
    //  string index: 01234567     lastSigDigit  dot    shift right
    //  ------------------------------------------------------------
    //                1234.5p1          6         4       1
    //                0.0012p4          6         1       4
    exp -= (lastSigDigit - dot - 1);
  }

  // get rid of leading zeroes, shift left, and adjust exponent
  unsigned leadingZeros = tmpSignificand.countl_zero() / 4;
  tmpSignificand <<= leadingZeros * 4;
  exp += (tmpSignificand.getBitWidth() / 4 - leadingZeros);

  if (exp > 63 || exp < -64) {
    opStatus fs = handleDenorm(tmpSignificand, exp);
    if (fs != opOK)
      return fs;
  }

  // check and perform rounding by checking the bit at precision + 1
  unsigned excessBits =
      tmpSignificand.getBitWidth() - getNumPrecisionBits() - 1;
  if (tmpSignificand[excessBits]) {
    APInt mask(tmpSignificand.getBitWidth(), 0);
    mask.setBit(excessBits);
    tmpSignificand += mask;
  }

  significand = tmpSignificand.getHiBits(getNumPrecisionBits())
                    .trunc(getNumPrecisionBits());
  exponent = exp;

  return opOK;
}

Expected<opStatus>
HexFloat::convertFromDecimalString(StringRef str, roundingMode rounding_mode) {
  assert(APFloat::usesLayout<HexFloat>(*semantics) && "Unexpected Semantics");
  decimalInfo D;
  significand.clearAllBits();
  exponent = 0;

  /* Scan the text.  */
  StringRef::iterator p = str.begin();
  if (Error Err = interpretDecimal(p, str.end(), &D))
    return std::move(Err);

  /* Handle the quick cases.  First the case of no significant digits,
     i.e. zero, and then exponents that are obviously too large or too
     small.  Writing L for log 10 / log 2, maxExponent2 is the binary
     exponent, and precision is in binary bits, a number d.ddddd*10^exp
     definitely overflows if

           (exp - 1) * L >= maxExponent

     and definitely underflows to zero where

           (exp + 1) * L <= minExponent - precision

     With integer arithmetic the tightest bounds for L are

           93/28 < L < 196/59            [ numerator <= 256 ]
           42039/12655 < L < 28738/8651  [ numerator <= 65536 ]
  */

  // Test if we have a zero number allowing for strings with no null terminators
  // and zero decimals with non-zero exponents.
  //
  // We computed firstSigDigit by ignoring all zeros and dots. Thus if
  // D->firstSigDigit equals str.end(), every digit must be a zero and there can
  // be at most one dot. On the other hand, if we have a zero with a non-zero
  // exponent, then we know that D.firstSigDigit will be non-numeric.
  //  + maximum HexFloat exponent is 63.  Maximum fraction is 1 - epsilon,
  //    so the maximum value that can be represented in HexFloat is < 16^63
  //    We want to find the exponent of the maximum decimal that can be
  //    represented
  //      16^63 = 10^x  ==>  63 * l(16) = x * l(10)
  //      x = 63 * l(16) / l(10  =  75.86
  //  + rpeating for the minimum exponent of -64, x = -77.06

  int max10Exponent = 75;
  int min10Exponent = -78;
  if (D.firstSigDigit == str.end() || decDigitValue(*D.firstSigDigit) >= 10U) {
    category = fcZero;
    makeZero(sign);
    return opOK;
  }
  /* Check whether the normalized exponent is high enough to overflow*/
  if (D.normalizedExponent > max10Exponent)
    return handleOverflow(sign);
  /* If it wasn't, then it also wasn't high enough to overflow max
     during the log-rebasing in the min-exponent check.  Check that it
     won't overflow min*/
  if (D.normalizedExponent <
      min10Exponent - (int)getNumPrecisionBits() * 59 / 196) {
    /* Underflow to zero and round.  */
    category = fcNormal;
    return handleUnderflow(sign);
  }

  /* A tight upper bound on number of bits required to hold an
     N-digit decimal integer is N * 196 / 59. The bits required for a
     multiplication of M * N bits is (M + N). Allocate enough space to hold
     the full significand.*/
  unsigned decDigits =
      static_cast<unsigned int>(D.lastSigDigit - D.firstSigDigit) + 1;
  int exp = D.exponent > 0 ? D.exponent : -D.exponent;
  unsigned size = (decDigits + exp + 1) * 196 / 59;
  size += 4 - size % 4; // make sure size is divisible by 4 for hexits
  // use one additional hexit than the precision to perform rounding
  size = std::max(size, getNumPrecisionBits() + 4);
  StringRef sig = StringRef(D.firstSigDigit, decDigits);

  /* Build the significand from the string */
  APInt tmpSignificand = APInt(size, 0);
  int tmpExponent = 0;
  StringRef::iterator e = sig.end();
  for (StringRef::iterator p = sig.begin(); p != e; p++) {
    if (*p != '.') {
      tmpSignificand *= 10;
      tmpSignificand += *p - '0';
    }
  }
  /* Build the multiplier from the exponent
   * 10^n = 5^n * 2^n */
  APInt multiplier = APInt(size, 1);
  for (int i = 0; i < exp; i++) {
    multiplier *= 5;
    multiplier <<= 1;
  }
  /* If the exponent is greater than 0, mulitply the significand and
   * multiplier. If the exponent is negative then divide the significand by
   * the multiplier. */
  if (D.exponent >= 0)
    tmpSignificand *= multiplier;
  else {
    /* Extend the values to 4x the bit width so when we perform division, there
     * is no loss of precision. This also ensures the width is divisible
     * by 4 so we can convert 2^n to 16^n */
    int extend_to = 4 * tmpSignificand.getBitWidth();
    int extend_by = extend_to - tmpSignificand.getBitWidth();
    APInt top = tmpSignificand.zext(extend_to);
    APInt bottom = multiplier.zext(extend_to);
    top <<= extend_by;
    tmpSignificand = top.udiv(bottom);
    tmpExponent = -extend_by / 4;
  }
  // get rid of leading zeroes
  unsigned leadingZeros = tmpSignificand.countl_zero() / 4;
  tmpSignificand <<= leadingZeros * 4;
  tmpExponent -= leadingZeros;

  tmpExponent += tmpSignificand.getBitWidth() / 4;

  if (tmpExponent > 63 || tmpExponent < -64) {
    opStatus fs = handleDenorm(tmpSignificand, tmpExponent);
    if (fs != opOK)
      return fs;
  }

  // check and perform rounding by checking the bit at precision + 1
  unsigned excessBits =
      tmpSignificand.getBitWidth() - getNumPrecisionBits() - 1;
  if (tmpSignificand[excessBits]) {
    APInt mask(tmpSignificand.getBitWidth(), 0);
    mask.setBit(excessBits);
    tmpSignificand += mask;
  }

  significand = tmpSignificand.getHiBits(getNumPrecisionBits())
                    .trunc(getNumPrecisionBits());
  exponent = tmpExponent;

  return opOK;
}

APInt HexFloat::bitcastToAPInt() const {
  auto get_sign_exponent_byte = [](int s, int e) {
    return s << 7 | ((e + 64) & 0x7f);
  };
  const bool is128 = semantics == &APFloatBase::HexFP128();
  const int NumPrecisionBits = getNumPrecisionBits();
  const int width = NumPrecisionBits + (is128 ? 16 : 8);
  APInt sign_and_exponent(width, get_sign_exponent_byte(sign, exponent));
  sign_and_exponent <<= (width - 8);
  APInt Ret(sign_and_exponent);

  if (is128) {
    APInt low(significand.extractBits(56, 0));
    APInt high(significand.extractBits(56, 56));
    APInt low_sign_exponent(width,
                            get_sign_exponent_byte(low_sign, low_exponent));
    low = low.zext(width);
    high = high.zext(width);
    high <<= 64;
    low_sign_exponent <<= 56;
    Ret |= high;
    Ret |= low_sign_exponent;
    Ret |= low;
  } else {
    Ret |= (significand.zextOrTrunc(width));
  }
  return Ret;
}

unsigned int HexFloat::convertToHexString(char *DST, unsigned int HexDigits,
                                          bool upperCase,
                                          roundingMode RM) const {
  llvm_unreachable("HexFloat::convertToHexString nor supported");
}

bool HexFloat::isZero() const {
  // exponent, and signficant are zero for zero value
  return exponent == -64 && significand.isZero();
}

bool HexFloat::isNegative() const {
  // like IEEE, zeros can be -ve, so we ignore the exponent/fraction
  return sign;
}

bool HexFloat::isDenormal() const {
  // It seems that HexFloat allows arbitrary denormals,
  // i.e., when the exponent is other than -64.
  // We need, therefore, to look at the most significant hexit.
  return significand.countLeadingZeros() >= 4;
}

bool HexFloat::isSmallest() const {
  return exponent == -64 && significand.isOne();
}

bool HexFloat::isLargest() const {
  return exponent == 63 && significand.isAllOnes();
}

bool HexFloat::isInteger() const {
  // as for the IEEE case
  HexFloat truncated = *this;
  truncated.roundToIntegral(rmTowardZero);
  return compare(truncated) == cmpEqual;
}

bool HexFloat::isSmallestNormalized() const {
  // we ignore the sign for this predicate --- see the corresponding IEEE method
  return exponent == -64 && significand.isOneBitSet(getNumPrecisionBits() - 4);
}

int HexFloat::getExactLog2Abs() const {
  if (isZero())
    return INT_MIN;
  if (1 != significand.popcount())
    return INT_MIN; // not exactly one bit set

  // exponent is to base 16.  The corresponding power of 2 is 4 * exponent.
  // The significand represents a fraction.  There is no implicit bit.
  // The most significant (left-most) bit represents 2^(-1).
  return 4 * exponent - (significand.countl_zero() + 1);
}

void HexFloat::toString(SmallVectorImpl<char> &str, unsigned precision,
                        unsigned maxPadding, bool truncateZero) const {

  // handle 0 as a special case
  if (isZero()) {
    if (isNegative())
      str.push_back('-');

    if (!maxPadding) {
      if (truncateZero)
        append(str, "0.0E+0");
      else {
        append(str, "0.0");
        if (precision > 1)
          str.append(precision - 1, '0');
        append(str, "e+00");
      }
    } else
      str.push_back('0');
    return;
  }

  // hand off to toStringImpl.
  // toStringImpl expects the significand to be in base 2,
  // so we need to adjust the expoent (recall, 16 == 2^4,
  // so 16^x == 2^4x).
  // Additionally, toStringImpl treats the significand
  // as an integer, so we need to adjust the exponent
  // so the radix point is on the right.
  int e = 4 * exponent - significand.getBitWidth();
  toStringImpl(str, isNegative(), e, significand, precision, maxPadding,
               truncateZero);
}

hash_code HexFloat::hash_value() const {
  if (isZero()) {
    return hash_combine(semantics);
  }
  return hash_combine(getNumPrecisionBits(), (uint8_t)sign, exponent,
                      ::hash_value(significand));
}

int ilogb(const HexFloat &Arg) {
  if (Arg.isZero())
    return APFloatBase::IEK_Zero;

  // need to be careful in case significand is not normalized
  int nLeadingZeroHexits = Arg.significand.countLeadingZeros() / 4;
  return Arg.exponent - 1 - nLeadingZeroHexits;
}

HexFloat scalbn(HexFloat X, int Exp, roundingMode RoundingMode) {
  if (Exp == 0 || X.isZero()) {
    // Exp == 0  ==> multiplying by 1
    // isZero() ==> value will be zero after multiplication
    return X;
  }

  // first normalize X, adjusting significand in place,
  // and assigning interim exponent to e
  int e = X.exponent;
  int nLeadingZeroHexits = X.significand.countLeadingZeros() / 4;
  if (nLeadingZeroHexits) {
    X.significand <<= (4 * nLeadingZeroHexits);
    e -= nLeadingZeroHexits;
  }

  // now multiply by 16^Exp
  e += Exp;
  if (e > 63) {
    // overflow --- clamp to maximum value
    X.exponent = 63;
    X.significand.setAllBits();
  } else {
    if (e < -64) {
      X.exponent = -64;
      // see if we can denormalize the significand
      int nHexits = X.significand.getBitWidth() / 4;
      int placesToShift = -64 - e;
      if (placesToShift <= nHexits) {
        // example: e is -66, so -64 - e == 2
        //          if 2 is less than the number of hexits
        //          in the significand, we can denormalize
        X.significand = X.significand.lshr(placesToShift * 4);
      } else {
        // too small to denormalize
        X.significand.clearAllBits();
      }
    } else {
      // ordinary case
      X.exponent = e;
    }
  }

  return X;
}

HexFloat frexp(HexFloat X, int &Exp, roundingMode RM) {
  if (X.isZero()) {
    Exp = 0;
    X.makeZero(X.isNegative());
    return X;
  }
  Exp = X.exponent;
  // normalize
  int NumLeadingZeroHexits = X.significand.countLeadingZeros() / 4;
  if (NumLeadingZeroHexits > 0) {
    X.significand <<= (NumLeadingZeroHexits * 4);
    Exp -= NumLeadingZeroHexits;
  }
  X.exponent = 0;
  if (X.semantics == &APFloat::HexFP128())
    X.low_exponent = -14;
  // convert from base 16 to baes 2
  Exp *= 4;
  //  ensure that 1 < |significand| <= 1/2
  // Note that we are now working with a exponent of base 2.
  while (!X.significand.isSignBitSet()) {
    X.significand <<= 1;
    Exp--;
  }
  return X;
}

void HexFloat::dump() const {
  SmallString<32> S;
  significand.toStringUnsigned(S, 16);
  dbgs() << "(" << (isNegative() ? "1" : "0") << "/" << low_sign << ", "
         << exponent << "/" << low_exponent << ", ("
         << significand.getBitWidth() << ", " << S << ")"
         << ")\n";
}

} // namespace detail

APFloat::Storage::Storage(IEEEFloat F, const fltSemantics &Semantics) {
  if (usesLayout<IEEEFloat>(Semantics)) {
    new (&IEEE) IEEEFloat(std::move(F));
    return;
  }
  if (usesLayout<DoubleAPFloat>(Semantics)) {
    const fltSemantics &S = F.getSemantics();
    new (&Double) DoubleAPFloat(Semantics, APFloat(std::move(F), S),
                                APFloat(APFloatBase::IEEEdouble()));
    return;
  }
  llvm_unreachable("Unexpected semantics");
}

Expected<APFloat::opStatus> APFloat::convertFromString(StringRef Str,
                                                       roundingMode RM) {
  APFLOAT_DISPATCH_ON_SEMANTICS(convertFromString(Str, RM));
}

hash_code hash_value(const APFloat &Arg) {
  if (APFloat::usesLayout<detail::IEEEFloat>(Arg.getSemantics()))
    return hash_value(Arg.U.IEEE);
  if (APFloat::usesLayout<detail::DoubleAPFloat>(Arg.getSemantics()))
    return hash_value(Arg.U.Double);
  if (APFloat::usesLayout<detail::HexFloat>(Arg.getSemantics()))
    return Arg.U.Hex.hash_value();
  llvm_unreachable("Unexpected semantics");
}

APFloat::APFloat(const fltSemantics &Semantics, StringRef S)
    : APFloat(Semantics) {
  auto StatusOrErr = convertFromString(S, rmNearestTiesToEven);
  assert(StatusOrErr && "Invalid floating point representation");
  consumeError(StatusOrErr.takeError());
}

FPClassTest APFloat::classify() const {
  if (isZero())
    return isNegative() ? fcNegZero : fcPosZero;
  if (isNormal())
    return isNegative() ? fcNegNormal : fcPosNormal;
  if (isDenormal())
    return isNegative() ? fcNegSubnormal : fcPosSubnormal;
  if (isInfinity())
    return isNegative() ? fcNegInf : fcPosInf;
  assert(isNaN() && "Other class of FP constant");
  return isSignaling() ? fcSNan : fcQNan;
}

bool APFloat::getExactInverse(APFloat *Inv) const {
  // Only finite, non-zero numbers can have a useful, representable inverse.
  // This check filters out +/- zero, +/- infinity, and NaN.
  if (!isFiniteNonZero())
    return false;

  // Historically, this function rejects subnormal inputs.  One reason why this
  // might be important is that subnormals may behave differently under FTZ/DAZ
  // runtime behavior.
  if (isDenormal())
    return false;

  // A number has an exact, representable inverse if and only if it is a power
  // of two.
  //
  // Mathematical Rationale:
  // 1. A binary floating-point number x is a dyadic rational, meaning it can
  //    be written as x = M / 2^k for integers M (the significand) and k.
  // 2. The inverse is 1/x = 2^k / M.
  // 3. For 1/x to also be a dyadic rational (and thus exactly representable
  //    in binary), its denominator M must also be a power of two.
  //    Let's say M = 2^m.
  // 4. Substituting this back into the formula for x, we get
  //    x = (2^m) / (2^k) = 2^(m-k).
  //
  // This proves that x must be a power of two.

  // The following will work iff the radix of the representation is a power of 2
  // A power of 2 has exactly one bit set.
  const unsigned int R = getSemantics().radix;
  assert(llvm::has_single_bit(R) && "radix is not a power of 2");

  // getExactLog2Abs() returns the integer exponent if the number is a power of
  // two or INT_MIN if it is not.
  const int Exp = getExactLog2Abs();
  if (Exp == INT_MIN)
    return false;

  // Let R be the radix, and let R be a power of 2, and let r = lg2(R)
  // then:
  //   2^Exp = R ^ floor(Exp/r) * 2 ^ (Exp mod r)
  // We can compute R ^ floor(Exp/r) using scalbn.
  // We can compute 2 ^ (exp mod r) using left shifts.
  // We need to take a little care when calculating floor and mod
  // to handle negative exponents correctly.

  APFloat Reciprocal(getSemantics());
  if (R == 2) {
    // The inverse of +/- 2^Exp is +/- 2^(-Exp). We can compute this by
    // scaling 1.0 by the negated exponent.
    Reciprocal =
        scalbn(APFloat::getOne(getSemantics(), /*Negative=*/isNegative()), -Exp,
               rmTowardZero);
  } else {
    // general case
    // The exact inverse of 2^Exp is 2^-Exp
    // Let Exp = Ar + b where  0 <\ !B| < r
    // then
    //   2^(Ar + B) == 2^(Ar) * 2^(B) == R^A * 2^B
    // and
    //   2 ^ -(Ar + B) == 2^-Ar * 2^-B == R^-A * 2^-B
    // We will use left shifts to exponentiate the 2.
    // Thus we need the power of 2 to be >= 0.
    // The code below sets up p2 to be <= 0 so that we shift left by -p2.

    const unsigned int r = llvm::countr_zero(R);
    int pR, p2;

    if (Exp >= 0) {
      pR = Exp / r;
      p2 = Exp % r;
      if (p2 > 0) {
        p2 -= r;
        pR++;
      }
    } else {
      pR = -(-Exp / r);
      p2 = -(-Exp % r);
    }

    APInt PowerOf2AsAPInt(32, 1 << -p2);
    APFloat PowerOf2AsAPFloat(getSemantics());
    auto convertStatus = PowerOf2AsAPFloat.convertFromAPInt(
        PowerOf2AsAPInt, /* isSigned */ false, rmTowardZero);
    assert(convertStatus == opOK &&
           "error converting PowerOf2AsAPint to an APFloat");
    if (isNegative() != PowerOf2AsAPFloat.isNegative())
      PowerOf2AsAPFloat.changeSign();
    assert(isNegative() == PowerOf2AsAPFloat.isNegative());
    Reciprocal = scalbn(PowerOf2AsAPFloat, -pR, rmTowardZero);
  }

  // scalbn might round if the resulting exponent -Exp is outside the
  // representable range, causing overflow (to infinity) or underflow. We
  // must verify that the result is still the exact power of two we expect.
  if (Reciprocal.getExactLog2Abs() != -Exp)
    return false;

  // Avoid multiplication with a subnormal, it is not safe on all platforms and
  // may be slower than a normal division.
  if (Reciprocal.isDenormal())
    return false;

  assert(Reciprocal.isFiniteNonZero());

  if (Inv)
    *Inv = std::move(Reciprocal);

  return true;
}

APFloat::opStatus APFloat::convert(const fltSemantics &ToSemantics,
                                   roundingMode RM, bool *losesInfo) {
  if (&getSemantics() == &ToSemantics) {
    *losesInfo = false;
    return opOK;
  }
  if (usesLayout<IEEEFloat>(getSemantics()) &&
      usesLayout<IEEEFloat>(ToSemantics))
    return U.IEEE.convert(ToSemantics, RM, losesInfo);
  if (usesLayout<IEEEFloat>(getSemantics()) &&
      usesLayout<DoubleAPFloat>(ToSemantics)) {
    assert(&ToSemantics == &APFloatBase::semPPCDoubleDouble);
    auto Ret =
        U.IEEE.convert(APFloatBase::semPPCDoubleDoubleLegacy, RM, losesInfo);
    *this = APFloat(ToSemantics, U.IEEE.bitcastToAPInt());
    return Ret;
  }
  if (usesLayout<DoubleAPFloat>(getSemantics()) &&
      usesLayout<IEEEFloat>(ToSemantics)) {
    auto Ret = getIEEE().convert(ToSemantics, RM, losesInfo);
    *this = APFloat(std::move(getIEEE()), ToSemantics);
    return Ret;
  }
  if (usesLayout<HexFloat>(getSemantics()) &&
      usesLayout<HexFloat>(ToSemantics)) {
    return U.Hex.convert(ToSemantics, RM, losesInfo);
  }
  llvm_unreachable("Unexpected semantics");
}

APFloat APFloat::getAllOnesValue(const fltSemantics &Semantics) {
  return APFloat(Semantics, APInt::getAllOnes(Semantics.sizeInBits));
}

void APFloat::print(raw_ostream &OS) const {
  SmallVector<char, 16> Buffer;
  toString(Buffer);
  OS << Buffer;
}

#if !defined(NDEBUG) || defined(LLVM_ENABLE_DUMP)
LLVM_DUMP_METHOD void APFloat::dump() const {
  print(dbgs());
  dbgs() << '\n';
}
#endif

void APFloat::Profile(FoldingSetNodeID &NID) const {
  NID.Add(bitcastToAPInt());
}

APFloat::opStatus APFloat::convertToInteger(APSInt &result,
                                            roundingMode rounding_mode,
                                            bool *isExact) const {
  unsigned bitWidth = result.getBitWidth();
  SmallVector<uint64_t, 4> parts(result.getNumWords());
  opStatus status = convertToInteger(parts, bitWidth, result.isSigned(),
                                     rounding_mode, isExact);
  // Keeps the original signed-ness.
  result = APInt(bitWidth, parts);
  return status;
}

double APFloat::convertToDouble() const {
  if (&getSemantics() ==
      (const llvm::fltSemantics *)&APFloatBase::semIEEEdouble)
    return getIEEE().convertToDouble();
  assert(isRepresentableBy(getSemantics(), semIEEEdouble) &&
         "Float semantics is not representable by IEEEdouble");
  APFloat Temp = *this;
  bool LosesInfo;
  opStatus St =
      Temp.convert(APFloatBase::semIEEEdouble, rmNearestTiesToEven, &LosesInfo);
  assert(!(St & opInexact) && !LosesInfo && "Unexpected imprecision");
  (void)St;
  return Temp.getIEEE().convertToDouble();
}

#ifdef HAS_IEE754_FLOAT128
float128 APFloat::convertToQuad() const {
  if (&getSemantics() == (const llvm::fltSemantics *)&APFloatBase::semIEEEquad)
    return getIEEE().convertToQuad();
  assert(isRepresentableBy(getSemantics(), semIEEEquad) &&
         "Float semantics is not representable by IEEEquad");
  APFloat Temp = *this;
  bool LosesInfo;
  opStatus St =
      Temp.convert(APFloatBase::semIEEEquad, rmNearestTiesToEven, &LosesInfo);
  assert(!(St & opInexact) && !LosesInfo && "Unexpected imprecision");
  (void)St;
  return Temp.getIEEE().convertToQuad();
}
#endif

float APFloat::convertToFloat() const {
  if (&getSemantics() ==
      (const llvm::fltSemantics *)&APFloatBase::semIEEEsingle)
    return getIEEE().convertToFloat();
  assert(isRepresentableBy(getSemantics(), semIEEEsingle) &&
         "Float semantics is not representable by IEEEsingle");
  APFloat Temp = *this;
  bool LosesInfo;
  opStatus St =
      Temp.convert(APFloatBase::semIEEEsingle, rmNearestTiesToEven, &LosesInfo);
  assert(!(St & opInexact) && !LosesInfo && "Unexpected imprecision");
  (void)St;
  return Temp.getIEEE().convertToFloat();
}

bool APFloatBase::isValidArbitraryFPFormat(StringRef Format) {
  static constexpr StringLiteral ValidFormats[] = {
      "Float8E5M2",     "Float8E5M2FNUZ",    "Float8E4M3",  "Float8E4M3FN",
      "Float8E4M3FNUZ", "Float8E4M3B11FNUZ", "Float8E3M4",  "Float8E8M0FNU",
      "Float6E3M2FN",   "Float6E2M3FN",      "Float4E2M1FN"};
  return llvm::is_contained(ValidFormats, Format);
}

APFloat::Storage::~Storage() {
  if (usesLayout<IEEEFloat>(*semantics)) {
    IEEE.~IEEEFloat();
    return;
  }
  if (usesLayout<DoubleAPFloat>(*semantics)) {
    Double.~DoubleAPFloat();
    return;
  }
  if (usesLayout<HexFloat>(*semantics)) {
    Hex.~HexFloat();
    return;
  }
  llvm_unreachable("Unexpected semantics");
}

APFloat::Storage::Storage(const APFloat::Storage &RHS) {
  if (usesLayout<IEEEFloat>(*RHS.semantics)) {
    new (this) IEEEFloat(RHS.IEEE);
    return;
  }
  if (usesLayout<DoubleAPFloat>(*RHS.semantics)) {
    new (this) DoubleAPFloat(RHS.Double);
    return;
  }
  if (usesLayout<HexFloat>(*RHS.semantics)) {
    new (this) HexFloat(RHS.Hex);
    return;
  }
  llvm_unreachable("Unexpected semantics");
}

APFloat::Storage::Storage(APFloat::Storage &&RHS) {
  if (usesLayout<IEEEFloat>(*RHS.semantics)) {
    new (this) IEEEFloat(std::move(RHS.IEEE));
    return;
  }
  if (usesLayout<DoubleAPFloat>(*RHS.semantics)) {
    new (this) DoubleAPFloat(std::move(RHS.Double));
    return;
  }
  if (usesLayout<HexFloat>(*RHS.semantics)) {
    new (this) HexFloat(std::move(RHS.Hex));
    return;
  }
  llvm_unreachable("Unexpected semantics");
}

APFloat::Storage &APFloat::Storage::operator=(const APFloat::Storage &RHS) {
  if (usesLayout<IEEEFloat>(*semantics) &&
      usesLayout<IEEEFloat>(*RHS.semantics)) {
    IEEE = RHS.IEEE;
  } else if (usesLayout<DoubleAPFloat>(*semantics) &&
             usesLayout<DoubleAPFloat>(*RHS.semantics)) {
    Double = RHS.Double;
  } else if (usesLayout<HexFloat>(*semantics) &&
             usesLayout<HexFloat>(*RHS.semantics)) {
    Hex = RHS.Hex;
  } else if (this != &RHS) {
    this->~Storage();
    new (this) Storage(RHS);
  }
  return *this;
}

APFloat::Storage &APFloat::Storage::operator=(APFloat::Storage &&RHS) {
  if (usesLayout<IEEEFloat>(*semantics) &&
      usesLayout<IEEEFloat>(*RHS.semantics)) {
    IEEE = std::move(RHS.IEEE);
  } else if (usesLayout<DoubleAPFloat>(*semantics) &&
             usesLayout<DoubleAPFloat>(*RHS.semantics)) {
    Double = std::move(RHS.Double);
  } else if (usesLayout<HexFloat>(*semantics) &&
             usesLayout<HexFloat>(*RHS.semantics)) {
    Hex = std::move(RHS.Hex);
  } else if (this != &RHS) {
    this->~Storage();
    new (this) Storage(std::move(RHS));
  }
  return *this;
}

} // namespace llvm

#undef APFLOAT_DISPATCH_ON_SEMANTICS

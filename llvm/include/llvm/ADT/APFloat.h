//===- llvm/ADT/APFloat.h - Arbitrary Precision Floating Point ---*- C++ -*-==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This file declares a class to represent arbitrary precision floating point
/// values and provide a variety of arithmetic operations on them.
///
//===----------------------------------------------------------------------===//

#ifndef LLVM_ADT_APFLOAT_H
#define LLVM_ADT_APFLOAT_H

#include "llvm/ADT/APInt.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/FloatingPointMode.h"
#include "llvm/Support/Compiler.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/float128.h"
#include <memory>

#define APFLOAT_DISPATCH_ON_SEMANTICS(METHOD_CALL)                             \
  do {                                                                         \
    if (usesLayout<IEEEFloat>(getSemantics()))                                 \
      return U.IEEE.METHOD_CALL;                                               \
    if (usesLayout<DoubleAPFloat>(getSemantics()))                             \
      return U.Double.METHOD_CALL;                                             \
    llvm_unreachable("Unexpected semantics");                                  \
  } while (false)

namespace llvm {

struct fltSemantics;
class APSInt;
class StringRef;
class APFloat;
class raw_ostream;

template <typename T> class Expected;
template <typename T> class SmallVectorImpl;

/// Enum that represents what fraction of the LSB truncated bits of an fp number
/// represent.
///
/// This essentially combines the roles of guard and sticky bits.
enum lostFraction { // Example of truncated bits:
  lfExactlyZero,    // 000000
  lfLessThanHalf,   // 0xxxxx  x's not all zero
  lfExactlyHalf,    // 100000
  lfMoreThanHalf    // 1xxxxx  x's not all zero
};

/// A self-contained host- and target-independent arbitrary-precision
/// floating-point software implementation.
///
/// APFloat uses bignum integer arithmetic as provided by static functions in
/// the APInt class.  The library will work with bignum integers whose parts are
/// any unsigned type at least 16 bits wide, but 64 bits is recommended.
///
/// Written for clarity rather than speed, in particular with a view to use in
/// the front-end of a cross compiler so that target arithmetic can be correctly
/// performed on the host.  Performance should nonetheless be reasonable,
/// particularly for its intended use.  It may be useful as a base
/// implementation for a run-time library during development of a faster
/// target-specific one.
///
/// All 5 rounding modes in the IEEE-754R draft are handled correctly for all
/// implemented operations.  Currently implemented operations are add, subtract,
/// multiply, divide, fused-multiply-add, conversion-to-float,
/// conversion-to-integer and conversion-from-integer.  New rounding modes
/// (e.g. away from zero) can be added with three or four lines of code.
///
/// Four formats are built-in: IEEE single precision, double precision,
/// quadruple precision, and x87 80-bit extended double (when operating with
/// full extended precision).  Adding a new format that obeys IEEE semantics
/// only requires adding two lines of code: a declaration and definition of the
/// format.
///
/// All operations return the status of that operation as an exception bit-mask,
/// so multiple operations can be done consecutively with their results or-ed
/// together.  The returned status can be useful for compiler diagnostics; e.g.,
/// inexact, underflow and overflow can be easily diagnosed on constant folding,
/// and compiler optimizers can determine what exceptions would be raised by
/// folding operations and optimize, or perhaps not optimize, accordingly.
///
/// At present, underflow tininess is detected after rounding; it should be
/// straight forward to add support for the before-rounding case too.
///
/// The library reads hexadecimal floating point numbers as per C99, and
/// correctly rounds if necessary according to the specified rounding mode.
/// Syntax is required to have been validated by the caller.  It also converts
/// floating point numbers to hexadecimal text as per the C99 %a and %A
/// conversions.  The output precision (or alternatively the natural minimal
/// precision) can be specified; if the requested precision is less than the
/// natural precision the output is correctly rounded for the specified rounding
/// mode.
///
/// It also reads decimal floating point numbers and correctly rounds according
/// to the specified rounding mode.
///
/// Conversion to decimal text is not currently implemented.
///
/// Non-zero finite numbers are represented internally as a sign bit, a 16-bit
/// signed exponent, and the significand as an array of integer parts.  After
/// normalization of a number of precision P the exponent is within the range of
/// the format, and if the number is not denormal the P-th bit of the
/// significand is set as an explicit integer bit.  For denormals the most
/// significant bit is shifted right so that the exponent is maintained at the
/// format's minimum, so that the smallest denormal has just the least
/// significant bit of the significand set.  The sign of zeroes and infinities
/// is significant; the exponent and significand of such numbers is not stored,
/// but has a known implicit (deterministic) value: 0 for the significands, 0
/// for zero exponent, all 1 bits for infinity exponent.  For NaNs the sign and
/// significand are deterministic, although not really meaningful, and preserved
/// in non-conversion operations.  The exponent is implicitly all 1 bits.
///
/// APFloat does not provide any exception handling beyond default exception
/// handling. We represent Signaling NaNs via IEEE-754R 2008 6.2.1 should clause
/// by encoding Signaling NaNs with the first bit of its trailing significand as
/// 0.
///
/// TODO
/// ====
///
/// Some features that may or may not be worth adding:
///
/// Binary to decimal conversion (hard).
///
/// Optional ability to detect underflow tininess before rounding.
///
/// New formats: x87 in single and double precision mode (IEEE apart from
/// extended exponent range) (hard).
///
/// New operations: sqrt, IEEE remainder, C90 fmod, nexttoward.
///

// This is the common type definitions shared by APFloat and its internal
// implementation classes. This struct should not define any non-static data
// members.
struct APFloatBase {
  typedef APInt::WordType integerPart;
  static constexpr unsigned integerPartWidth = APInt::APINT_BITS_PER_WORD;

  /// A signed type to represent a floating point numbers unbiased exponent.
  typedef int32_t ExponentType;

  /// \name Floating Point Semantics.
  /// @{
  enum Semantics {
    S_IEEEhalf,
    S_BFloat,
    S_IEEEsingle,
    S_IEEEdouble,
    S_IEEEquad,
    // The IBM double-double semantics. Such a number consists of a pair of
    // IEEE 64-bit doubles (Hi, Lo), where |Hi| > |Lo|, and if normal,
    // (double)(Hi + Lo) == Hi. The numeric value it's modeling is Hi + Lo.
    // Therefore it has two 53-bit mantissa parts that aren't necessarily
    // adjacent to each other, and two 11-bit exponents.
    //
    // Note: we need to make the value different from semBogus as otherwise
    // an unsafe optimization may collapse both values to a single address,
    // and we heavily rely on them having distinct addresses.
    S_PPCDoubleDouble,
    // These are legacy semantics for the fallback, inaccurate implementation
    // of IBM double-double, if the accurate semPPCDoubleDouble doesn't handle
    // the operation. It's equivalent to having an IEEE number with consecutive
    // 106 bits of mantissa and 11 bits of exponent.
    //
    // It's not equivalent to IBM double-double. For example, a legit IBM
    // double-double, 1 + epsilon:
    //
    // 1 + epsilon = 1 + (1 >> 1076)
    //
    // is not representable by a consecutive 106 bits of mantissa.
    //
    // Currently, these semantics are used in the following way:
    //
    //   semPPCDoubleDouble -> (IEEEdouble, IEEEdouble) ->
    //   (64-bit APInt, 64-bit APInt) -> (128-bit APInt) ->
    //   semPPCDoubleDoubleLegacy -> IEEE operations
    //
    // We use bitcastToAPInt() to get the bit representation (in APInt) of the
    // underlying IEEEdouble, then use the APInt constructor to construct the
    // legacy IEEE float.
    //
    // TODO: Implement all operations in semPPCDoubleDouble, and delete these
    // semantics.
    S_PPCDoubleDoubleLegacy,
    // 8-bit floating point number following IEEE-754 conventions with bit
    // layout S1E5M2 as described in https://arxiv.org/abs/2209.05433.
    S_Float8E5M2,
    // 8-bit floating point number mostly following IEEE-754 conventions
    // and bit layout S1E5M2 described in https://arxiv.org/abs/2206.02915,
    // with expanded range and with no infinity or signed zero.
    // NaN is represented as negative zero. (FN -> Finite, UZ -> unsigned zero).
    // This format's exponent bias is 16, instead of the 15 (2 ** (5 - 1) - 1)
    // that IEEE precedent would imply.
    S_Float8E5M2FNUZ,
    // 8-bit floating point number following IEEE-754 conventions with bit
    // layout S1E4M3.
    S_Float8E4M3,
    // 8-bit floating point number mostly following IEEE-754 conventions with
    // bit layout S1E4M3 as described in https://arxiv.org/abs/2209.05433.
    // Unlike IEEE-754 types, there are no infinity values, and NaN is
    // represented with the exponent and mantissa bits set to all 1s.
    S_Float8E4M3FN,
    // 8-bit floating point number mostly following IEEE-754 conventions
    // and bit layout S1E4M3 described in https://arxiv.org/abs/2206.02915,
    // with expanded range and with no infinity or signed zero.
    // NaN is represented as negative zero. (FN -> Finite, UZ -> unsigned zero).
    // This format's exponent bias is 8, instead of the 7 (2 ** (4 - 1) - 1)
    // that IEEE precedent would imply.
    S_Float8E4M3FNUZ,
    // 8-bit floating point number mostly following IEEE-754 conventions
    // and bit layout S1E4M3 with expanded range and with no infinity or signed
    // zero.
    // NaN is represented as negative zero. (FN -> Finite, UZ -> unsigned zero).
    // This format's exponent bias is 11, instead of the 7 (2 ** (4 - 1) - 1)
    // that IEEE precedent would imply.
    S_Float8E4M3B11FNUZ,
    // 8-bit floating point number following IEEE-754 conventions with bit
    // layout S1E3M4.
    S_Float8E3M4,
    // Floating point number that occupies 32 bits or less of storage, providing
    // improved range compared to half (16-bit) formats, at (potentially)
    // greater throughput than single precision (32-bit) formats.
    S_FloatTF32,
    // 8-bit floating point number with (all the) 8 bits for the exponent
    // like in FP32. There are no zeroes, no infinities, and no denormal values.
    // This format has unsigned representation only. (U -> Unsigned only).
    // NaN is represented with all bits set to 1. Bias is 127.
    // This format represents the scale data type in the MX specification from:
    // https://www.opencompute.org/documents/ocp-microscaling-formats-mx-v1-0-spec-final-pdf
    S_Float8E8M0FNU,
    // 6-bit floating point number with bit layout S1E3M2. Unlike IEEE-754
    // types, there are no infinity or NaN values. The format is detailed in
    // https://www.opencompute.org/documents/ocp-microscaling-formats-mx-v1-0-spec-final-pdf
    S_Float6E3M2FN,
    // 6-bit floating point number with bit layout S1E2M3. Unlike IEEE-754
    // types, there are no infinity or NaN values. The format is detailed in
    // https://www.opencompute.org/documents/ocp-microscaling-formats-mx-v1-0-spec-final-pdf
    S_Float6E2M3FN,
    // 4-bit floating point number with bit layout S1E2M1. Unlike IEEE-754
    // types, there are no infinity or NaN values. The format is detailed in
    // https://www.opencompute.org/documents/ocp-microscaling-formats-mx-v1-0-spec-final-pdf
    S_Float4E2M1FN,
    // TODO: Documentation is missing.
    S_x87DoubleExtended,
    S_MaxSemantics = S_x87DoubleExtended,
  };

  LLVM_ABI static const llvm::fltSemantics &EnumToSemantics(Semantics S);
  LLVM_ABI static Semantics SemanticsToEnum(const llvm::fltSemantics &Sem);

  LLVM_ABI static const fltSemantics &IEEEhalf() LLVM_READNONE;
  LLVM_ABI static const fltSemantics &BFloat() LLVM_READNONE;
  LLVM_ABI static const fltSemantics &IEEEsingle() LLVM_READNONE;
  LLVM_ABI static const fltSemantics &IEEEdouble() LLVM_READNONE;
  LLVM_ABI static const fltSemantics &IEEEquad() LLVM_READNONE;
  LLVM_ABI static const fltSemantics &PPCDoubleDouble() LLVM_READNONE;
  LLVM_ABI static const fltSemantics &PPCDoubleDoubleLegacy() LLVM_READNONE;
  LLVM_ABI static const fltSemantics &Float8E5M2() LLVM_READNONE;
  LLVM_ABI static const fltSemantics &Float8E5M2FNUZ() LLVM_READNONE;
  LLVM_ABI static const fltSemantics &Float8E4M3() LLVM_READNONE;
  LLVM_ABI static const fltSemantics &Float8E4M3FN() LLVM_READNONE;
  LLVM_ABI static const fltSemantics &Float8E4M3FNUZ() LLVM_READNONE;
  LLVM_ABI static const fltSemantics &Float8E4M3B11FNUZ() LLVM_READNONE;
  LLVM_ABI static const fltSemantics &Float8E3M4() LLVM_READNONE;
  LLVM_ABI static const fltSemantics &FloatTF32() LLVM_READNONE;
  LLVM_ABI static const fltSemantics &Float8E8M0FNU() LLVM_READNONE;
  LLVM_ABI static const fltSemantics &Float6E3M2FN() LLVM_READNONE;
  LLVM_ABI static const fltSemantics &Float6E2M3FN() LLVM_READNONE;
  LLVM_ABI static const fltSemantics &Float4E2M1FN() LLVM_READNONE;
  LLVM_ABI static const fltSemantics &x87DoubleExtended() LLVM_READNONE;

  /// A Pseudo fltsemantic used to construct APFloats that cannot conflict with
  /// anything real.
  LLVM_ABI static const fltSemantics &Bogus() LLVM_READNONE;

  // Returns true if any number described by this semantics can be precisely
  // represented by the specified semantics. Does not take into account
  // the value of fltNonfiniteBehavior, hasZero, hasSignedRepr.
  LLVM_ABI static bool isRepresentableBy(const fltSemantics &A,
                                         const fltSemantics &B);

  /// @}

  /// IEEE-754R 5.11: Floating Point Comparison Relations.
  enum cmpResult {
    cmpLessThan,
    cmpEqual,
    cmpGreaterThan,
    cmpUnordered
  };

  /// IEEE-754R 4.3: Rounding-direction attributes.
  using roundingMode = llvm::RoundingMode;

  static constexpr roundingMode rmNearestTiesToEven =
                                                RoundingMode::NearestTiesToEven;
  static constexpr roundingMode rmTowardPositive = RoundingMode::TowardPositive;
  static constexpr roundingMode rmTowardNegative = RoundingMode::TowardNegative;
  static constexpr roundingMode rmTowardZero     = RoundingMode::TowardZero;
  static constexpr roundingMode rmNearestTiesToAway =
                                                RoundingMode::NearestTiesToAway;

  /// IEEE-754R 7: Default exception handling.
  ///
  /// opUnderflow or opOverflow are always returned or-ed with opInexact.
  ///
  /// APFloat models this behavior specified by IEEE-754:
  ///   "For operations producing results in floating-point format, the default
  ///    result of an operation that signals the invalid operation exception
  ///    shall be a quiet NaN."
  enum opStatus {
    opOK = 0x00,
    opInvalidOp = 0x01,
    opDivByZero = 0x02,
    opOverflow = 0x04,
    opUnderflow = 0x08,
    opInexact = 0x10
  };

  /// Category of internally-represented number.
  enum fltCategory {
    fcInfinity,
    fcNaN,
    fcNormal,
    fcZero
  };

  /// Convenience enum used to construct an uninitialized APFloat.
  enum uninitializedTag {
    uninitialized
  };

  /// Enumeration of \c ilogb error results.
  enum IlogbErrorKinds {
    IEK_Zero = INT_MIN + 1,
    IEK_NaN = INT_MIN,
    IEK_Inf = INT_MAX
  };

  LLVM_ABI static unsigned int semanticsPrecision(const fltSemantics &);
  LLVM_ABI static ExponentType semanticsMinExponent(const fltSemantics &);
  LLVM_ABI static ExponentType semanticsMaxExponent(const fltSemantics &);
  LLVM_ABI static unsigned int semanticsSizeInBits(const fltSemantics &);
  LLVM_ABI static unsigned int semanticsIntSizeInBits(const fltSemantics &,
                                                      bool);
  LLVM_ABI static bool semanticsHasZero(const fltSemantics &);
  LLVM_ABI static bool semanticsHasSignedRepr(const fltSemantics &);
  LLVM_ABI static bool semanticsHasInf(const fltSemantics &);
  LLVM_ABI static bool semanticsHasNaN(const fltSemantics &);
  LLVM_ABI static bool isIEEELikeFP(const fltSemantics &);
  LLVM_ABI static bool hasSignBitInMSB(const fltSemantics &);

  // Returns true if any number described by \p Src can be precisely represented
  // by a normal (not subnormal) value in \p Dst.
  LLVM_ABI static bool isRepresentableAsNormalIn(const fltSemantics &Src,
                                                 const fltSemantics &Dst);

  /// Returns the size of the floating point number (in bits) in the given
  /// semantics.
  LLVM_ABI static unsigned getSizeInBits(const fltSemantics &Sem);
};

namespace detail {

using integerPart = APFloatBase::integerPart;
using uninitializedTag = APFloatBase::uninitializedTag;
using roundingMode = APFloatBase::roundingMode;
using opStatus = APFloatBase::opStatus;
using cmpResult = APFloatBase::cmpResult;
using fltCategory = APFloatBase::fltCategory;
using ExponentType = APFloatBase::ExponentType;
static constexpr uninitializedTag uninitialized = APFloatBase::uninitialized;
static constexpr roundingMode rmNearestTiesToEven =
    APFloatBase::rmNearestTiesToEven;
static constexpr roundingMode rmNearestTiesToAway =
    APFloatBase::rmNearestTiesToAway;
static constexpr roundingMode rmTowardNegative = APFloatBase::rmTowardNegative;
static constexpr roundingMode rmTowardPositive = APFloatBase::rmTowardPositive;
static constexpr roundingMode rmTowardZero = APFloatBase::rmTowardZero;
static constexpr unsigned integerPartWidth = APFloatBase::integerPartWidth;
static constexpr cmpResult cmpEqual = APFloatBase::cmpEqual;
static constexpr cmpResult cmpLessThan = APFloatBase::cmpLessThan;
static constexpr cmpResult cmpGreaterThan = APFloatBase::cmpGreaterThan;
static constexpr cmpResult cmpUnordered = APFloatBase::cmpUnordered;
static constexpr opStatus opOK = APFloatBase::opOK;
static constexpr opStatus opInvalidOp = APFloatBase::opInvalidOp;
static constexpr opStatus opDivByZero = APFloatBase::opDivByZero;
static constexpr opStatus opOverflow = APFloatBase::opOverflow;
static constexpr opStatus opUnderflow = APFloatBase::opUnderflow;
static constexpr opStatus opInexact = APFloatBase::opInexact;
static constexpr fltCategory fcInfinity = APFloatBase::fcInfinity;
static constexpr fltCategory fcNaN = APFloatBase::fcNaN;
static constexpr fltCategory fcNormal = APFloatBase::fcNormal;
static constexpr fltCategory fcZero = APFloatBase::fcZero;

class IEEEFloat final {
public:
  /// \name Constructors
  /// @{

  LLVM_ABI IEEEFloat(const fltSemantics &); // Default construct to +0.0
  LLVM_ABI IEEEFloat(const fltSemantics &, integerPart);
  LLVM_ABI IEEEFloat(const fltSemantics &, uninitializedTag);
  LLVM_ABI IEEEFloat(const fltSemantics &, const APInt &);
  LLVM_ABI explicit IEEEFloat(double d);
  LLVM_ABI explicit IEEEFloat(float f);
  LLVM_ABI IEEEFloat(const IEEEFloat &);
  LLVM_ABI IEEEFloat(IEEEFloat &&);
  LLVM_ABI ~IEEEFloat();

  /// @}

  /// Returns whether this instance allocated memory.
  bool needsCleanup() const { return partCount() > 1; }

  /// \name Convenience "constructors"
  /// @{

  /// @}

  /// \name Arithmetic
  /// @{

  LLVM_ABI opStatus add(const IEEEFloat &, roundingMode);
  LLVM_ABI opStatus subtract(const IEEEFloat &, roundingMode);
  LLVM_ABI opStatus multiply(const IEEEFloat &, roundingMode);
  LLVM_ABI opStatus divide(const IEEEFloat &, roundingMode);
  /// IEEE remainder.
  LLVM_ABI opStatus remainder(const IEEEFloat &);
  /// C fmod, or llvm frem.
  LLVM_ABI opStatus mod(const IEEEFloat &);
  LLVM_ABI opStatus fusedMultiplyAdd(const IEEEFloat &, const IEEEFloat &,
                                     roundingMode);
  LLVM_ABI opStatus roundToIntegral(roundingMode);
  /// IEEE-754R 5.3.1: nextUp/nextDown.
  LLVM_ABI opStatus next(bool nextDown);

  /// @}

  /// \name Sign operations.
  /// @{

  LLVM_ABI void changeSign();

  /// @}

  /// \name Conversions
  /// @{

  LLVM_ABI opStatus convert(const fltSemantics &, roundingMode, bool *);
  LLVM_ABI opStatus convertToInteger(MutableArrayRef<integerPart>, unsigned int,
                                     bool, roundingMode, bool *) const;
  LLVM_ABI opStatus convertFromAPInt(const APInt &, bool, roundingMode);
  LLVM_ABI opStatus convertFromSignExtendedInteger(const integerPart *,
                                                   unsigned int, bool,
                                                   roundingMode);
  LLVM_ABI opStatus convertFromZeroExtendedInteger(const integerPart *,
                                                   unsigned int, bool,
                                                   roundingMode);
  LLVM_ABI Expected<opStatus> convertFromString(StringRef, roundingMode);
  LLVM_ABI APInt bitcastToAPInt() const;
  LLVM_ABI double convertToDouble() const;
#ifdef HAS_IEE754_FLOAT128
  LLVM_ABI float128 convertToQuad() const;
#endif
  LLVM_ABI float convertToFloat() const;

  /// @}

  /// The definition of equality is not straightforward for floating point, so
  /// we won't use operator==.  Use one of the following, or write whatever it
  /// is you really mean.
  bool operator==(const IEEEFloat &) const = delete;

  /// IEEE comparison with another floating point number (NaNs compare
  /// unordered, 0==-0).
  LLVM_ABI cmpResult compare(const IEEEFloat &) const;

  /// Bitwise comparison for equality (QNaNs compare equal, 0!=-0).
  LLVM_ABI bool bitwiseIsEqual(const IEEEFloat &) const;

  /// Write out a hexadecimal representation of the floating point value to DST,
  /// which must be of sufficient size, in the C99 form [-]0xh.hhhhp[+-]d.
  /// Return the number of characters written, excluding the terminating NUL.
  LLVM_ABI unsigned int convertToHexString(char *dst, unsigned int hexDigits,
                                           bool upperCase, roundingMode) const;

  /// \name IEEE-754R 5.7.2 General operations.
  /// @{

  /// IEEE-754R isSignMinus: Returns true if and only if the current value is
  /// negative.
  ///
  /// This applies to zeros and NaNs as well.
  bool isNegative() const { return sign; }

  /// IEEE-754R isNormal: Returns true if and only if the current value is normal.
  ///
  /// This implies that the current value of the float is not zero, subnormal,
  /// infinite, or NaN following the definition of normality from IEEE-754R.
  bool isNormal() const { return !isDenormal() && isFiniteNonZero(); }

  /// Returns true if and only if the current value is zero, subnormal, or
  /// normal.
  ///
  /// This means that the value is not infinite or NaN.
  bool isFinite() const { return !isNaN() && !isInfinity(); }

  /// Returns true if and only if the float is plus or minus zero.
  bool isZero() const { return category == fltCategory::fcZero; }

  /// IEEE-754R isSubnormal(): Returns true if and only if the float is a
  /// denormal.
  LLVM_ABI bool isDenormal() const;

  /// IEEE-754R isInfinite(): Returns true if and only if the float is infinity.
  bool isInfinity() const { return category == fcInfinity; }

  /// Returns true if and only if the float is a quiet or signaling NaN.
  bool isNaN() const { return category == fcNaN; }

  /// Returns true if and only if the float is a signaling NaN.
  LLVM_ABI bool isSignaling() const;

  /// @}

  /// \name Simple Queries
  /// @{

  fltCategory getCategory() const { return category; }
  const fltSemantics &getSemantics() const { return *semantics; }
  bool isNonZero() const { return category != fltCategory::fcZero; }
  bool isFiniteNonZero() const { return isFinite() && !isZero(); }
  bool isPosZero() const { return isZero() && !isNegative(); }
  bool isNegZero() const { return isZero() && isNegative(); }

  /// Returns true if and only if the number has the smallest possible non-zero
  /// magnitude in the current semantics.
  LLVM_ABI bool isSmallest() const;

  /// Returns true if this is the smallest (by magnitude) normalized finite
  /// number in the given semantics.
  LLVM_ABI bool isSmallestNormalized() const;

  /// Returns true if and only if the number has the largest possible finite
  /// magnitude in the current semantics.
  LLVM_ABI bool isLargest() const;

  /// Returns true if and only if the number is an exact integer.
  LLVM_ABI bool isInteger() const;

  /// @}

  LLVM_ABI IEEEFloat &operator=(const IEEEFloat &);
  LLVM_ABI IEEEFloat &operator=(IEEEFloat &&);

  /// Overload to compute a hash code for an APFloat value.
  ///
  /// Note that the use of hash codes for floating point values is in general
  /// frought with peril. Equality is hard to define for these values. For
  /// example, should negative and positive zero hash to different codes? Are
  /// they equal or not? This hash value implementation specifically
  /// emphasizes producing different codes for different inputs in order to
  /// be used in canonicalization and memoization. As such, equality is
  /// bitwiseIsEqual, and 0 != -0.
  LLVM_ABI friend hash_code hash_value(const IEEEFloat &Arg);

  /// Converts this value into a decimal string.
  ///
  /// \param FormatPrecision The maximum number of digits of
  ///   precision to output.  If there are fewer digits available,
  ///   zero padding will not be used unless the value is
  ///   integral and small enough to be expressed in
  ///   FormatPrecision digits.  0 means to use the natural
  ///   precision of the number.
  /// \param FormatMaxPadding The maximum number of zeros to
  ///   consider inserting before falling back to scientific
  ///   notation.  0 means to always use scientific notation.
  ///
  /// \param TruncateZero Indicate whether to remove the trailing zero in
  ///   fraction part or not. Also setting this parameter to false forcing
  ///   producing of output more similar to default printf behavior.
  ///   Specifically the lower e is used as exponent delimiter and exponent
  ///   always contains no less than two digits.
  ///
  /// Number       Precision    MaxPadding      Result
  /// ------       ---------    ----------      ------
  /// 1.01E+4              5             2       10100
  /// 1.01E+4              4             2       1.01E+4
  /// 1.01E+4              5             1       1.01E+4
  /// 1.01E-2              5             2       0.0101
  /// 1.01E-2              4             2       0.0101
  /// 1.01E-2              4             1       1.01E-2
  LLVM_ABI void toString(SmallVectorImpl<char> &Str,
                         unsigned FormatPrecision = 0,
                         unsigned FormatMaxPadding = 3,
                         bool TruncateZero = true) const;

  LLVM_ABI LLVM_READONLY int getExactLog2Abs() const;

  LLVM_ABI friend int ilogb(const IEEEFloat &Arg);

  LLVM_ABI friend IEEEFloat scalbn(IEEEFloat X, int Exp, roundingMode);

  LLVM_ABI friend IEEEFloat frexp(const IEEEFloat &X, int &Exp, roundingMode);

  /// \name Special value setters.
  /// @{

  LLVM_ABI void makeLargest(bool Neg = false);
  LLVM_ABI void makeSmallest(bool Neg = false);
  LLVM_ABI void makeNaN(bool SNaN = false, bool Neg = false,
                        const APInt *fill = nullptr);
  LLVM_ABI void makeInf(bool Neg = false);
  LLVM_ABI void makeZero(bool Neg = false);
  LLVM_ABI void makeQuiet();

  /// Returns the smallest (by magnitude) normalized finite number in the given
  /// semantics.
  ///
  /// \param Negative - True iff the number should be negative
  LLVM_ABI void makeSmallestNormalized(bool Negative = false);

  /// @}

  LLVM_ABI cmpResult compareAbsoluteValue(const IEEEFloat &) const;

private:
  /// \name Simple Queries
  /// @{

  integerPart *significandParts();
  const integerPart *significandParts() const;
  LLVM_ABI unsigned int partCount() const;

  /// @}

  /// \name Significand operations.
  /// @{

  integerPart addSignificand(const IEEEFloat &);
  integerPart subtractSignificand(const IEEEFloat &, integerPart);
  // Exported for IEEEFloatUnitTestHelper.
  LLVM_ABI lostFraction addOrSubtractSignificand(const IEEEFloat &,
                                                 bool subtract);
  lostFraction multiplySignificand(const IEEEFloat &, IEEEFloat,
                                   bool ignoreAddend = false);
  lostFraction multiplySignificand(const IEEEFloat&);
  lostFraction divideSignificand(const IEEEFloat &);
  void incrementSignificand();
  void initialize(const fltSemantics *);
  void shiftSignificandLeft(unsigned int);
  lostFraction shiftSignificandRight(unsigned int);
  unsigned int significandLSB() const;
  unsigned int significandMSB() const;
  void zeroSignificand();
  unsigned int getNumHighBits() const;
  /// Return true if the significand excluding the integral bit is all ones.
  bool isSignificandAllOnes() const;
  bool isSignificandAllOnesExceptLSB() const;
  /// Return true if the significand excluding the integral bit is all zeros.
  bool isSignificandAllZeros() const;
  bool isSignificandAllZerosExceptMSB() const;

  /// @}

  /// \name Arithmetic on special values.
  /// @{

  opStatus addOrSubtractSpecials(const IEEEFloat &, bool subtract);
  opStatus divideSpecials(const IEEEFloat &);
  opStatus multiplySpecials(const IEEEFloat &);
  opStatus modSpecials(const IEEEFloat &);
  opStatus remainderSpecials(const IEEEFloat&);

  /// @}

  /// \name Miscellany
  /// @{

  bool convertFromStringSpecials(StringRef str);
  opStatus normalize(roundingMode, lostFraction);
  opStatus addOrSubtract(const IEEEFloat &, roundingMode, bool subtract);
  opStatus handleOverflow(roundingMode);
  bool roundAwayFromZero(roundingMode, lostFraction, unsigned int) const;
  opStatus convertToSignExtendedInteger(MutableArrayRef<integerPart>,
                                        unsigned int, bool, roundingMode,
                                        bool *) const;
  opStatus convertFromUnsignedParts(const integerPart *, unsigned int,
                                    roundingMode);
  Expected<opStatus> convertFromHexadecimalString(StringRef, roundingMode);
  Expected<opStatus> convertFromDecimalString(StringRef, roundingMode);
  char *convertNormalToHexString(char *, unsigned int, bool,
                                 roundingMode) const;
  opStatus roundSignificandWithExponent(const integerPart *, unsigned int, int,
                                        roundingMode);
  ExponentType exponentNaN() const;
  ExponentType exponentInf() const;
  ExponentType exponentZero() const;

  /// @}

  template <const fltSemantics &S> APInt convertIEEEFloatToAPInt() const;
  APInt convertHalfAPFloatToAPInt() const;
  APInt convertBFloatAPFloatToAPInt() const;
  APInt convertFloatAPFloatToAPInt() const;
  APInt convertDoubleAPFloatToAPInt() const;
  APInt convertQuadrupleAPFloatToAPInt() const;
  APInt convertF80LongDoubleAPFloatToAPInt() const;
  APInt convertPPCDoubleDoubleLegacyAPFloatToAPInt() const;
  APInt convertFloat8E5M2APFloatToAPInt() const;
  APInt convertFloat8E5M2FNUZAPFloatToAPInt() const;
  APInt convertFloat8E4M3APFloatToAPInt() const;
  APInt convertFloat8E4M3FNAPFloatToAPInt() const;
  APInt convertFloat8E4M3FNUZAPFloatToAPInt() const;
  APInt convertFloat8E4M3B11FNUZAPFloatToAPInt() const;
  APInt convertFloat8E3M4APFloatToAPInt() const;
  APInt convertFloatTF32APFloatToAPInt() const;
  APInt convertFloat8E8M0FNUAPFloatToAPInt() const;
  APInt convertFloat6E3M2FNAPFloatToAPInt() const;
  APInt convertFloat6E2M3FNAPFloatToAPInt() const;
  APInt convertFloat4E2M1FNAPFloatToAPInt() const;
  void initFromAPInt(const fltSemantics *Sem, const APInt &api);
  template <const fltSemantics &S> void initFromIEEEAPInt(const APInt &api);
  void initFromHalfAPInt(const APInt &api);
  void initFromBFloatAPInt(const APInt &api);
  void initFromFloatAPInt(const APInt &api);
  void initFromDoubleAPInt(const APInt &api);
  void initFromQuadrupleAPInt(const APInt &api);
  void initFromF80LongDoubleAPInt(const APInt &api);
  void initFromPPCDoubleDoubleLegacyAPInt(const APInt &api);
  void initFromFloat8E5M2APInt(const APInt &api);
  void initFromFloat8E5M2FNUZAPInt(const APInt &api);
  void initFromFloat8E4M3APInt(const APInt &api);
  void initFromFloat8E4M3FNAPInt(const APInt &api);
  void initFromFloat8E4M3FNUZAPInt(const APInt &api);
  void initFromFloat8E4M3B11FNUZAPInt(const APInt &api);
  void initFromFloat8E3M4APInt(const APInt &api);
  void initFromFloatTF32APInt(const APInt &api);
  void initFromFloat8E8M0FNUAPInt(const APInt &api);
  void initFromFloat6E3M2FNAPInt(const APInt &api);
  void initFromFloat6E2M3FNAPInt(const APInt &api);
  void initFromFloat4E2M1FNAPInt(const APInt &api);

  void assign(const IEEEFloat &);
  void copySignificand(const IEEEFloat &);
  void freeSignificand();

  /// Note: this must be the first data member.
  /// The semantics that this value obeys.
  const fltSemantics *semantics;

  /// A binary fraction with an explicit integer bit.
  ///
  /// The significand must be at least one bit wider than the target precision.
  union Significand {
    integerPart part;
    integerPart *parts;
  } significand;

  /// The signed unbiased exponent of the value.
  ExponentType exponent;

  /// What kind of floating point number this is.
  ///
  /// Only 2 bits are required, but VisualStudio incorrectly sign extends it.
  /// Using the extra bit keeps it from failing under VisualStudio.
  fltCategory category : 3;

  /// Sign bit of the number.
  unsigned int sign : 1;

  friend class IEEEFloatUnitTestHelper;
};

LLVM_ABI hash_code hash_value(const IEEEFloat &Arg);
LLVM_ABI int ilogb(const IEEEFloat &Arg);
LLVM_ABI IEEEFloat scalbn(IEEEFloat X, int Exp, roundingMode);
LLVM_ABI IEEEFloat frexp(const IEEEFloat &Val, int &Exp, roundingMode RM);

// This mode implements more precise float in terms of two APFloats.
// The interface and layout is designed for arbitrary underlying semantics,
// though currently only PPCDoubleDouble semantics are supported, whose
// corresponding underlying semantics are IEEEdouble.
class DoubleAPFloat final {
  // Note: this must be the first data member.
  const fltSemantics *Semantics;
  APFloat *Floats;

  opStatus addImpl(const APFloat &a, const APFloat &aa, const APFloat &c,
                   const APFloat &cc, roundingMode RM);

  opStatus addWithSpecial(const DoubleAPFloat &LHS, const DoubleAPFloat &RHS,
                          DoubleAPFloat &Out, roundingMode RM);
  opStatus convertToSignExtendedInteger(MutableArrayRef<integerPart> Input,
                                        unsigned int Width, bool IsSigned,
                                        roundingMode RM, bool *IsExact) const;

public:
  LLVM_ABI DoubleAPFloat(const fltSemantics &S);
  LLVM_ABI DoubleAPFloat(const fltSemantics &S, uninitializedTag);
  LLVM_ABI DoubleAPFloat(const fltSemantics &S, integerPart);
  LLVM_ABI DoubleAPFloat(const fltSemantics &S, const APInt &I);
  LLVM_ABI DoubleAPFloat(const fltSemantics &S, APFloat &&First,
                         APFloat &&Second);
  LLVM_ABI DoubleAPFloat(const DoubleAPFloat &RHS);
  LLVM_ABI DoubleAPFloat(DoubleAPFloat &&RHS);
  ~DoubleAPFloat();

  LLVM_ABI DoubleAPFloat &operator=(const DoubleAPFloat &RHS);
  inline DoubleAPFloat &operator=(DoubleAPFloat &&RHS);

  bool needsCleanup() const { return Floats != nullptr; }

  inline APFloat &getFirst();
  inline const APFloat &getFirst() const;
  inline APFloat &getSecond();
  inline const APFloat &getSecond() const;

  LLVM_ABI opStatus add(const DoubleAPFloat &RHS, roundingMode RM);
  LLVM_ABI opStatus subtract(const DoubleAPFloat &RHS, roundingMode RM);
  LLVM_ABI opStatus multiply(const DoubleAPFloat &RHS, roundingMode RM);
  LLVM_ABI opStatus divide(const DoubleAPFloat &RHS, roundingMode RM);
  LLVM_ABI opStatus remainder(const DoubleAPFloat &RHS);
  LLVM_ABI opStatus mod(const DoubleAPFloat &RHS);
  LLVM_ABI opStatus fusedMultiplyAdd(const DoubleAPFloat &Multiplicand,
                                     const DoubleAPFloat &Addend,
                                     roundingMode RM);
  LLVM_ABI opStatus roundToIntegral(roundingMode RM);
  LLVM_ABI void changeSign();
  LLVM_ABI cmpResult compareAbsoluteValue(const DoubleAPFloat &RHS) const;

  LLVM_ABI fltCategory getCategory() const;
  LLVM_ABI bool isNegative() const;

  LLVM_ABI void makeInf(bool Neg);
  LLVM_ABI void makeZero(bool Neg);
  LLVM_ABI void makeLargest(bool Neg);
  LLVM_ABI void makeSmallest(bool Neg);
  LLVM_ABI void makeSmallestNormalized(bool Neg);
  LLVM_ABI void makeNaN(bool SNaN, bool Neg, const APInt *fill);

  LLVM_ABI cmpResult compare(const DoubleAPFloat &RHS) const;
  LLVM_ABI bool bitwiseIsEqual(const DoubleAPFloat &RHS) const;
  LLVM_ABI APInt bitcastToAPInt() const;
  LLVM_ABI Expected<opStatus> convertFromString(StringRef, roundingMode);
  LLVM_ABI opStatus next(bool nextDown);

  LLVM_ABI opStatus convertToInteger(MutableArrayRef<integerPart> Input,
                                     unsigned int Width, bool IsSigned,
                                     roundingMode RM, bool *IsExact) const;
  LLVM_ABI opStatus convertFromAPInt(const APInt &Input, bool IsSigned,
                                     roundingMode RM);
  LLVM_ABI opStatus convertFromSignExtendedInteger(const integerPart *Input,
                                                   unsigned int InputSize,
                                                   bool IsSigned,
                                                   roundingMode RM);
  LLVM_ABI opStatus convertFromZeroExtendedInteger(const integerPart *Input,
                                                   unsigned int InputSize,
                                                   bool IsSigned,
                                                   roundingMode RM);
  LLVM_ABI unsigned int convertToHexString(char *DST, unsigned int HexDigits,
                                           bool UpperCase,
                                           roundingMode RM) const;

  LLVM_ABI bool isDenormal() const;
  LLVM_ABI bool isSmallest() const;
  LLVM_ABI bool isSmallestNormalized() const;
  LLVM_ABI bool isLargest() const;
  LLVM_ABI bool isInteger() const;

  LLVM_ABI void toString(SmallVectorImpl<char> &Str, unsigned FormatPrecision,
                         unsigned FormatMaxPadding,
                         bool TruncateZero = true) const;

  LLVM_ABI LLVM_READONLY int getExactLog2Abs() const;

  LLVM_ABI friend int ilogb(const DoubleAPFloat &X);
  LLVM_ABI friend DoubleAPFloat scalbn(const DoubleAPFloat &X, int Exp,
                                       roundingMode);
  LLVM_ABI friend DoubleAPFloat frexp(const DoubleAPFloat &X, int &Exp,
                                      roundingMode);
  LLVM_ABI friend hash_code hash_value(const DoubleAPFloat &Arg);
};

LLVM_ABI hash_code hash_value(const DoubleAPFloat &Arg);
LLVM_ABI DoubleAPFloat scalbn(const DoubleAPFloat &Arg, int Exp,
                              roundingMode RM);
LLVM_ABI DoubleAPFloat frexp(const DoubleAPFloat &X, int &Exp, roundingMode);

} // End detail namespace

// This is a interface class that is currently forwarding functionalities from
// detail::IEEEFloat.
class APFloat : public APFloatBase {
  typedef detail::IEEEFloat IEEEFloat;
  typedef detail::DoubleAPFloat DoubleAPFloat;

  static_assert(std::is_standard_layout<IEEEFloat>::value);

  union Storage {
    const fltSemantics *semantics;
    IEEEFloat IEEE;
    DoubleAPFloat Double;

    LLVM_ABI explicit Storage(IEEEFloat F, const fltSemantics &S);
    explicit Storage(DoubleAPFloat F, const fltSemantics &S)
        : Double(std::move(F)) {
      assert(&S == &PPCDoubleDouble());
    }

    template <typename... ArgTypes>
    Storage(const fltSemantics &Semantics, ArgTypes &&... Args) {
      if (usesLayout<IEEEFloat>(Semantics)) {
        new (&IEEE) IEEEFloat(Semantics, std::forward<ArgTypes>(Args)...);
        return;
      }
      if (usesLayout<DoubleAPFloat>(Semantics)) {
        new (&Double) DoubleAPFloat(Semantics, std::forward<ArgTypes>(Args)...);
        return;
      }
      llvm_unreachable("Unexpected semantics");
    }

    ~Storage() {
      if (usesLayout<IEEEFloat>(*semantics)) {
        IEEE.~IEEEFloat();
        return;
      }
      if (usesLayout<DoubleAPFloat>(*semantics)) {
        Double.~DoubleAPFloat();
        return;
      }
      llvm_unreachable("Unexpected semantics");
    }

    Storage(const Storage &RHS) {
      if (usesLayout<IEEEFloat>(*RHS.semantics)) {
        new (this) IEEEFloat(RHS.IEEE);
        return;
      }
      if (usesLayout<DoubleAPFloat>(*RHS.semantics)) {
        new (this) DoubleAPFloat(RHS.Double);
        return;
      }
      llvm_unreachable("Unexpected semantics");
    }

    Storage(Storage &&RHS) {
      if (usesLayout<IEEEFloat>(*RHS.semantics)) {
        new (this) IEEEFloat(std::move(RHS.IEEE));
        return;
      }
      if (usesLayout<DoubleAPFloat>(*RHS.semantics)) {
        new (this) DoubleAPFloat(std::move(RHS.Double));
        return;
      }
      llvm_unreachable("Unexpected semantics");
    }

    Storage &operator=(const Storage &RHS) {
      if (usesLayout<IEEEFloat>(*semantics) &&
          usesLayout<IEEEFloat>(*RHS.semantics)) {
        IEEE = RHS.IEEE;
      } else if (usesLayout<DoubleAPFloat>(*semantics) &&
                 usesLayout<DoubleAPFloat>(*RHS.semantics)) {
        Double = RHS.Double;
      } else if (this != &RHS) {
        this->~Storage();
        new (this) Storage(RHS);
      }
      return *this;
    }

    Storage &operator=(Storage &&RHS) {
      if (usesLayout<IEEEFloat>(*semantics) &&
          usesLayout<IEEEFloat>(*RHS.semantics)) {
        IEEE = std::move(RHS.IEEE);
      } else if (usesLayout<DoubleAPFloat>(*semantics) &&
                 usesLayout<DoubleAPFloat>(*RHS.semantics)) {
        Double = std::move(RHS.Double);
      } else if (this != &RHS) {
        this->~Storage();
        new (this) Storage(std::move(RHS));
      }
      return *this;
    }
  } U;

  template <typename T> static bool usesLayout(const fltSemantics &Semantics) {
    static_assert(std::is_same<T, IEEEFloat>::value ||
                  std::is_same<T, DoubleAPFloat>::value);
    if (std::is_same<T, DoubleAPFloat>::value) {
      return &Semantics == &PPCDoubleDouble();
    }
    return &Semantics != &PPCDoubleDouble();
  }

  IEEEFloat &getIEEE() {
    if (usesLayout<IEEEFloat>(*U.semantics))
      return U.IEEE;
    if (usesLayout<DoubleAPFloat>(*U.semantics))
      return U.Double.getFirst().U.IEEE;
    llvm_unreachable("Unexpected semantics");
  }

  const IEEEFloat &getIEEE() const {
    if (usesLayout<IEEEFloat>(*U.semantics))
      return U.IEEE;
    if (usesLayout<DoubleAPFloat>(*U.semantics))
      return U.Double.getFirst().U.IEEE;
    llvm_unreachable("Unexpected semantics");
  }

  void makeZero(bool Neg) { APFLOAT_DISPATCH_ON_SEMANTICS(makeZero(Neg)); }

  void makeInf(bool Neg) { APFLOAT_DISPATCH_ON_SEMANTICS(makeInf(Neg)); }

  void makeNaN(bool SNaN, bool Neg, const APInt *fill) {
    APFLOAT_DISPATCH_ON_SEMANTICS(makeNaN(SNaN, Neg, fill));
  }

  void makeLargest(bool Neg) {
    APFLOAT_DISPATCH_ON_SEMANTICS(makeLargest(Neg));
  }

  void makeSmallest(bool Neg) {
    APFLOAT_DISPATCH_ON_SEMANTICS(makeSmallest(Neg));
  }

  void makeSmallestNormalized(bool Neg) {
    APFLOAT_DISPATCH_ON_SEMANTICS(makeSmallestNormalized(Neg));
  }

  explicit APFloat(IEEEFloat F, const fltSemantics &S) : U(std::move(F), S) {}
  explicit APFloat(DoubleAPFloat F, const fltSemantics &S)
      : U(std::move(F), S) {}

  // Compares the absolute value of this APFloat with another.  Both operands
  // must be finite non-zero.
  cmpResult compareAbsoluteValue(const APFloat &RHS) const {
    assert(&getSemantics() == &RHS.getSemantics() &&
           "Should only compare APFloats with the same semantics");
    if (usesLayout<IEEEFloat>(getSemantics()))
      return U.IEEE.compareAbsoluteValue(RHS.U.IEEE);
    if (usesLayout<DoubleAPFloat>(getSemantics()))
      return U.Double.compareAbsoluteValue(RHS.U.Double);
    llvm_unreachable("Unexpected semantics");
  }

public:
  APFloat(const fltSemantics &Semantics) : U(Semantics) {}
  LLVM_ABI APFloat(const fltSemantics &Semantics, StringRef S);
  APFloat(const fltSemantics &Semantics, integerPart I) : U(Semantics, I) {}
  template <typename T,
            typename = std::enable_if_t<std::is_floating_point<T>::value>>
  APFloat(const fltSemantics &Semantics, T V) = delete;
  // TODO: Remove this constructor. This isn't faster than the first one.
  APFloat(const fltSemantics &Semantics, uninitializedTag)
      : U(Semantics, uninitialized) {}
  APFloat(const fltSemantics &Semantics, const APInt &I) : U(Semantics, I) {}
  explicit APFloat(double d) : U(IEEEFloat(d), IEEEdouble()) {}
  explicit APFloat(float f) : U(IEEEFloat(f), IEEEsingle()) {}
  APFloat(const APFloat &RHS) = default;
  APFloat(APFloat &&RHS) = default;

  ~APFloat() = default;

  bool needsCleanup() const { APFLOAT_DISPATCH_ON_SEMANTICS(needsCleanup()); }

  /// Factory for Positive and Negative Zero.
  ///
  /// \param Negative True iff the number should be negative.
  static APFloat getZero(const fltSemantics &Sem, bool Negative = false) {
    APFloat Val(Sem, uninitialized);
    Val.makeZero(Negative);
    return Val;
  }

  /// Factory for Positive and Negative One.
  ///
  /// \param Negative True iff the number should be negative.
  static APFloat getOne(const fltSemantics &Sem, bool Negative = false) {
    APFloat Val(Sem, 1U);
    if (Negative)
      Val.changeSign();
    return Val;
  }

  /// Factory for Positive and Negative Infinity.
  ///
  /// \param Negative True iff the number should be negative.
  static APFloat getInf(const fltSemantics &Sem, bool Negative = false) {
    APFloat Val(Sem, uninitialized);
    Val.makeInf(Negative);
    return Val;
  }

  /// Factory for NaN values.
  ///
  /// \param Negative - True iff the NaN generated should be negative.
  /// \param payload - The unspecified fill bits for creating the NaN, 0 by
  /// default.  The value is truncated as necessary.
  static APFloat getNaN(const fltSemantics &Sem, bool Negative = false,
                        uint64_t payload = 0) {
    if (payload) {
      APInt intPayload(64, payload);
      return getQNaN(Sem, Negative, &intPayload);
    } else {
      return getQNaN(Sem, Negative, nullptr);
    }
  }

  /// Factory for QNaN values.
  static APFloat getQNaN(const fltSemantics &Sem, bool Negative = false,
                         const APInt *payload = nullptr) {
    APFloat Val(Sem, uninitialized);
    Val.makeNaN(false, Negative, payload);
    return Val;
  }

  /// Factory for SNaN values.
  static APFloat getSNaN(const fltSemantics &Sem, bool Negative = false,
                         const APInt *payload = nullptr) {
    APFloat Val(Sem, uninitialized);
    Val.makeNaN(true, Negative, payload);
    return Val;
  }

  /// Returns the largest finite number in the given semantics.
  ///
  /// \param Negative - True iff the number should be negative
  static APFloat getLargest(const fltSemantics &Sem, bool Negative = false) {
    APFloat Val(Sem, uninitialized);
    Val.makeLargest(Negative);
    return Val;
  }

  /// Returns the smallest (by magnitude) finite number in the given semantics.
  /// Might be denormalized, which implies a relative loss of precision.
  ///
  /// \param Negative - True iff the number should be negative
  static APFloat getSmallest(const fltSemantics &Sem, bool Negative = false) {
    APFloat Val(Sem, uninitialized);
    Val.makeSmallest(Negative);
    return Val;
  }

  /// Returns the smallest (by magnitude) normalized finite number in the given
  /// semantics.
  ///
  /// \param Negative - True iff the number should be negative
  static APFloat getSmallestNormalized(const fltSemantics &Sem,
                                       bool Negative = false) {
    APFloat Val(Sem, uninitialized);
    Val.makeSmallestNormalized(Negative);
    return Val;
  }

  /// Returns a float which is bitcasted from an all one value int.
  ///
  /// \param Semantics - type float semantics
  LLVM_ABI static APFloat getAllOnesValue(const fltSemantics &Semantics);

  /// Returns true if the given semantics has actual significand.
  ///
  /// \param Sem - type float semantics
  static bool hasSignificand(const fltSemantics &Sem) {
    return &Sem != &Float8E8M0FNU();
  }

  /// Used to insert APFloat objects, or objects that contain APFloat objects,
  /// into FoldingSets.
  LLVM_ABI void Profile(FoldingSetNodeID &NID) const;

  opStatus add(const APFloat &RHS, roundingMode RM) {
    assert(&getSemantics() == &RHS.getSemantics() &&
           "Should only call on two APFloats with the same semantics");
    if (usesLayout<IEEEFloat>(getSemantics()))
      return U.IEEE.add(RHS.U.IEEE, RM);
    if (usesLayout<DoubleAPFloat>(getSemantics()))
      return U.Double.add(RHS.U.Double, RM);
    llvm_unreachable("Unexpected semantics");
  }
  opStatus subtract(const APFloat &RHS, roundingMode RM) {
    assert(&getSemantics() == &RHS.getSemantics() &&
           "Should only call on two APFloats with the same semantics");
    if (usesLayout<IEEEFloat>(getSemantics()))
      return U.IEEE.subtract(RHS.U.IEEE, RM);
    if (usesLayout<DoubleAPFloat>(getSemantics()))
      return U.Double.subtract(RHS.U.Double, RM);
    llvm_unreachable("Unexpected semantics");
  }
  opStatus multiply(const APFloat &RHS, roundingMode RM) {
    assert(&getSemantics() == &RHS.getSemantics() &&
           "Should only call on two APFloats with the same semantics");
    if (usesLayout<IEEEFloat>(getSemantics()))
      return U.IEEE.multiply(RHS.U.IEEE, RM);
    if (usesLayout<DoubleAPFloat>(getSemantics()))
      return U.Double.multiply(RHS.U.Double, RM);
    llvm_unreachable("Unexpected semantics");
  }
  opStatus divide(const APFloat &RHS, roundingMode RM) {
    assert(&getSemantics() == &RHS.getSemantics() &&
           "Should only call on two APFloats with the same semantics");
    if (usesLayout<IEEEFloat>(getSemantics()))
      return U.IEEE.divide(RHS.U.IEEE, RM);
    if (usesLayout<DoubleAPFloat>(getSemantics()))
      return U.Double.divide(RHS.U.Double, RM);
    llvm_unreachable("Unexpected semantics");
  }
  opStatus remainder(const APFloat &RHS) {
    assert(&getSemantics() == &RHS.getSemantics() &&
           "Should only call on two APFloats with the same semantics");
    if (usesLayout<IEEEFloat>(getSemantics()))
      return U.IEEE.remainder(RHS.U.IEEE);
    if (usesLayout<DoubleAPFloat>(getSemantics()))
      return U.Double.remainder(RHS.U.Double);
    llvm_unreachable("Unexpected semantics");
  }
  opStatus mod(const APFloat &RHS) {
    assert(&getSemantics() == &RHS.getSemantics() &&
           "Should only call on two APFloats with the same semantics");
    if (usesLayout<IEEEFloat>(getSemantics()))
      return U.IEEE.mod(RHS.U.IEEE);
    if (usesLayout<DoubleAPFloat>(getSemantics()))
      return U.Double.mod(RHS.U.Double);
    llvm_unreachable("Unexpected semantics");
  }
  opStatus fusedMultiplyAdd(const APFloat &Multiplicand, const APFloat &Addend,
                            roundingMode RM) {
    assert(&getSemantics() == &Multiplicand.getSemantics() &&
           "Should only call on APFloats with the same semantics");
    assert(&getSemantics() == &Addend.getSemantics() &&
           "Should only call on APFloats with the same semantics");
    if (usesLayout<IEEEFloat>(getSemantics()))
      return U.IEEE.fusedMultiplyAdd(Multiplicand.U.IEEE, Addend.U.IEEE, RM);
    if (usesLayout<DoubleAPFloat>(getSemantics()))
      return U.Double.fusedMultiplyAdd(Multiplicand.U.Double, Addend.U.Double,
                                       RM);
    llvm_unreachable("Unexpected semantics");
  }
  opStatus roundToIntegral(roundingMode RM) {
    APFLOAT_DISPATCH_ON_SEMANTICS(roundToIntegral(RM));
  }

  // TODO: bool parameters are not readable and a source of bugs.
  // Do something.
  opStatus next(bool nextDown) {
    APFLOAT_DISPATCH_ON_SEMANTICS(next(nextDown));
  }

  /// Negate an APFloat.
  APFloat operator-() const {
    APFloat Result(*this);
    Result.changeSign();
    return Result;
  }

  /// Add two APFloats, rounding ties to the nearest even.
  /// No error checking.
  APFloat operator+(const APFloat &RHS) const {
    APFloat Result(*this);
    (void)Result.add(RHS, rmNearestTiesToEven);
    return Result;
  }

  /// Subtract two APFloats, rounding ties to the nearest even.
  /// No error checking.
  APFloat operator-(const APFloat &RHS) const {
    APFloat Result(*this);
    (void)Result.subtract(RHS, rmNearestTiesToEven);
    return Result;
  }

  /// Multiply two APFloats, rounding ties to the nearest even.
  /// No error checking.
  APFloat operator*(const APFloat &RHS) const {
    APFloat Result(*this);
    (void)Result.multiply(RHS, rmNearestTiesToEven);
    return Result;
  }

  /// Divide the first APFloat by the second, rounding ties to the nearest even.
  /// No error checking.
  APFloat operator/(const APFloat &RHS) const {
    APFloat Result(*this);
    (void)Result.divide(RHS, rmNearestTiesToEven);
    return Result;
  }

  void changeSign() { APFLOAT_DISPATCH_ON_SEMANTICS(changeSign()); }
  void clearSign() {
    if (isNegative())
      changeSign();
  }
  void copySign(const APFloat &RHS) {
    if (isNegative() != RHS.isNegative())
      changeSign();
  }

  /// A static helper to produce a copy of an APFloat value with its sign
  /// copied from some other APFloat.
  static APFloat copySign(APFloat Value, const APFloat &Sign) {
    Value.copySign(Sign);
    return Value;
  }

  /// Assuming this is an IEEE-754 NaN value, quiet its signaling bit.
  /// This preserves the sign and payload bits.
  APFloat makeQuiet() const {
    APFloat Result(*this);
    Result.getIEEE().makeQuiet();
    return Result;
  }

  LLVM_ABI opStatus convert(const fltSemantics &ToSemantics, roundingMode RM,
                            bool *losesInfo);
  // Convert a floating point number to an integer according to the
  // rounding mode.  We provide deterministic values in case of an invalid
  // operation exception, namely zero for NaNs and the minimal or maximal value
  // respectively for underflow or overflow.
  // The *IsExact output tells whether the result is exact, in the sense that
  // converting it back to the original floating point type produces the
  // original value.  This is almost equivalent to result==opOK, except for
  // negative zeroes.
  opStatus convertToInteger(MutableArrayRef<integerPart> Input,
                            unsigned int Width, bool IsSigned, roundingMode RM,
                            bool *IsExact) const {
    APFLOAT_DISPATCH_ON_SEMANTICS(
        convertToInteger(Input, Width, IsSigned, RM, IsExact));
  }
  // Same as convertToInteger(integerPart*, ...), except the result is returned
  // in an APSInt, whose initial bit-width and signed-ness are used to determine
  // the precision of the conversion.
  LLVM_ABI opStatus convertToInteger(APSInt &Result, roundingMode RM,
                                     bool *IsExact) const;
  opStatus convertFromAPInt(const APInt &Input, bool IsSigned,
                            roundingMode RM) {
    APFLOAT_DISPATCH_ON_SEMANTICS(convertFromAPInt(Input, IsSigned, RM));
  }
  opStatus convertFromSignExtendedInteger(const integerPart *Input,
                                          unsigned int InputSize, bool IsSigned,
                                          roundingMode RM) {
    APFLOAT_DISPATCH_ON_SEMANTICS(
        convertFromSignExtendedInteger(Input, InputSize, IsSigned, RM));
  }
  opStatus convertFromZeroExtendedInteger(const integerPart *Input,
                                          unsigned int InputSize, bool IsSigned,
                                          roundingMode RM) {
    APFLOAT_DISPATCH_ON_SEMANTICS(
        convertFromZeroExtendedInteger(Input, InputSize, IsSigned, RM));
  }
  LLVM_ABI Expected<opStatus> convertFromString(StringRef, roundingMode);
  APInt bitcastToAPInt() const {
    APFLOAT_DISPATCH_ON_SEMANTICS(bitcastToAPInt());
  }

  /// Converts this APFloat to host double value.
  ///
  /// \pre The APFloat must be built using semantics, that can be represented by
  /// the host double type without loss of precision. It can be IEEEdouble and
  /// shorter semantics, like IEEEsingle and others.
  LLVM_ABI double convertToDouble() const;

  /// Converts this APFloat to host float value.
  ///
  /// \pre The APFloat must be built using semantics, that can be represented by
  /// the host float type without loss of precision. It can be IEEEquad and
  /// shorter semantics, like IEEEdouble and others.
#ifdef HAS_IEE754_FLOAT128
  LLVM_ABI float128 convertToQuad() const;
#endif

  /// Converts this APFloat to host float value.
  ///
  /// \pre The APFloat must be built using semantics, that can be represented by
  /// the host float type without loss of precision. It can be IEEEsingle and
  /// shorter semantics, like IEEEhalf.
  LLVM_ABI float convertToFloat() const;

  bool operator==(const APFloat &RHS) const { return compare(RHS) == cmpEqual; }

  bool operator!=(const APFloat &RHS) const { return compare(RHS) != cmpEqual; }

  bool operator<(const APFloat &RHS) const {
    return compare(RHS) == cmpLessThan;
  }

  bool operator>(const APFloat &RHS) const {
    return compare(RHS) == cmpGreaterThan;
  }

  bool operator<=(const APFloat &RHS) const {
    cmpResult Res = compare(RHS);
    return Res == cmpLessThan || Res == cmpEqual;
  }

  bool operator>=(const APFloat &RHS) const {
    cmpResult Res = compare(RHS);
    return Res == cmpGreaterThan || Res == cmpEqual;
  }

  // IEEE comparison with another floating point number (NaNs compare unordered,
  // 0==-0).
  cmpResult compare(const APFloat &RHS) const {
    assert(&getSemantics() == &RHS.getSemantics() &&
           "Should only compare APFloats with the same semantics");
    if (usesLayout<IEEEFloat>(getSemantics()))
      return U.IEEE.compare(RHS.U.IEEE);
    if (usesLayout<DoubleAPFloat>(getSemantics()))
      return U.Double.compare(RHS.U.Double);
    llvm_unreachable("Unexpected semantics");
  }

  bool bitwiseIsEqual(const APFloat &RHS) const {
    if (&getSemantics() != &RHS.getSemantics())
      return false;
    if (usesLayout<IEEEFloat>(getSemantics()))
      return U.IEEE.bitwiseIsEqual(RHS.U.IEEE);
    if (usesLayout<DoubleAPFloat>(getSemantics()))
      return U.Double.bitwiseIsEqual(RHS.U.Double);
    llvm_unreachable("Unexpected semantics");
  }

  /// We don't rely on operator== working on double values, as
  /// it returns true for things that are clearly not equal, like -0.0 and 0.0.
  /// As such, this method can be used to do an exact bit-for-bit comparison of
  /// two floating point values.
  ///
  /// We leave the version with the double argument here because it's just so
  /// convenient to write "2.0" and the like.  Without this function we'd
  /// have to duplicate its logic everywhere it's called.
  bool isExactlyValue(double V) const {
    bool ignored;
    APFloat Tmp(V);
    Tmp.convert(getSemantics(), APFloat::rmNearestTiesToEven, &ignored);
    return bitwiseIsEqual(Tmp);
  }

  unsigned int convertToHexString(char *DST, unsigned int HexDigits,
                                  bool UpperCase, roundingMode RM) const {
    APFLOAT_DISPATCH_ON_SEMANTICS(
        convertToHexString(DST, HexDigits, UpperCase, RM));
  }

  bool isZero() const { return getCategory() == fcZero; }
  bool isInfinity() const { return getCategory() == fcInfinity; }
  bool isNaN() const { return getCategory() == fcNaN; }

  bool isNegative() const { return getIEEE().isNegative(); }
  bool isDenormal() const { APFLOAT_DISPATCH_ON_SEMANTICS(isDenormal()); }
  bool isSignaling() const { return getIEEE().isSignaling(); }

  bool isNormal() const { return !isDenormal() && isFiniteNonZero(); }
  bool isFinite() const { return !isNaN() && !isInfinity(); }

  fltCategory getCategory() const { return getIEEE().getCategory(); }
  const fltSemantics &getSemantics() const { return *U.semantics; }
  bool isNonZero() const { return !isZero(); }
  bool isFiniteNonZero() const { return isFinite() && !isZero(); }
  bool isPosZero() const { return isZero() && !isNegative(); }
  bool isNegZero() const { return isZero() && isNegative(); }
  bool isPosInfinity() const { return isInfinity() && !isNegative(); }
  bool isNegInfinity() const { return isInfinity() && isNegative(); }
  bool isSmallest() const { APFLOAT_DISPATCH_ON_SEMANTICS(isSmallest()); }
  bool isLargest() const { APFLOAT_DISPATCH_ON_SEMANTICS(isLargest()); }
  bool isInteger() const { APFLOAT_DISPATCH_ON_SEMANTICS(isInteger()); }

  bool isSmallestNormalized() const {
    APFLOAT_DISPATCH_ON_SEMANTICS(isSmallestNormalized());
  }

  /// Return the FPClassTest which will return true for the value.
  LLVM_ABI FPClassTest classify() const;

  APFloat &operator=(const APFloat &RHS) = default;
  APFloat &operator=(APFloat &&RHS) = default;

  void toString(SmallVectorImpl<char> &Str, unsigned FormatPrecision = 0,
                unsigned FormatMaxPadding = 3, bool TruncateZero = true) const {
    APFLOAT_DISPATCH_ON_SEMANTICS(
        toString(Str, FormatPrecision, FormatMaxPadding, TruncateZero));
  }

  LLVM_ABI void print(raw_ostream &) const;

#if !defined(NDEBUG) || defined(LLVM_ENABLE_DUMP)
  LLVM_DUMP_METHOD void dump() const;
#endif

  /// If this value is normal and has an exact, normal, multiplicative inverse,
  /// store it in inv and return true.
  bool getExactInverse(APFloat *Inv) const;

  // If this is an exact power of two, return the exponent while ignoring the
  // sign bit. If it's not an exact power of 2, return INT_MIN
  LLVM_READONLY
  int getExactLog2Abs() const {
    APFLOAT_DISPATCH_ON_SEMANTICS(getExactLog2Abs());
  }

  // If this is an exact power of two, return the exponent. If it's not an exact
  // power of 2, return INT_MIN
  LLVM_READONLY
  int getExactLog2() const {
    return isNegative() ? INT_MIN : getExactLog2Abs();
  }

  LLVM_ABI friend hash_code hash_value(const APFloat &Arg);
  friend int ilogb(const APFloat &Arg);
  friend APFloat scalbn(APFloat X, int Exp, roundingMode RM);
  friend APFloat frexp(const APFloat &X, int &Exp, roundingMode RM);
  friend IEEEFloat;
  friend DoubleAPFloat;
};

static_assert(sizeof(APFloat) == sizeof(detail::IEEEFloat),
              "Empty base class optimization is not performed.");

/// See friend declarations above.
///
/// These additional declarations are required in order to compile LLVM with IBM
/// xlC compiler.
LLVM_ABI hash_code hash_value(const APFloat &Arg);

/// Returns the exponent of the internal representation of the APFloat.
///
/// Because the radix of APFloat is 2, this is equivalent to floor(log2(x)).
/// For special APFloat values, this returns special error codes:
///
///   NaN -> \c IEK_NaN
///   0   -> \c IEK_Zero
///   Inf -> \c IEK_Inf
///
inline int ilogb(const APFloat &Arg) {
  if (APFloat::usesLayout<detail::IEEEFloat>(Arg.getSemantics()))
    return ilogb(Arg.U.IEEE);
  if (APFloat::usesLayout<detail::DoubleAPFloat>(Arg.getSemantics()))
    return ilogb(Arg.U.Double);
  llvm_unreachable("Unexpected semantics");
}

/// Returns: X * 2^Exp for integral exponents.
inline APFloat scalbn(APFloat X, int Exp, APFloat::roundingMode RM) {
  if (APFloat::usesLayout<detail::IEEEFloat>(X.getSemantics()))
    return APFloat(scalbn(X.U.IEEE, Exp, RM), X.getSemantics());
  if (APFloat::usesLayout<detail::DoubleAPFloat>(X.getSemantics()))
    return APFloat(scalbn(X.U.Double, Exp, RM), X.getSemantics());
  llvm_unreachable("Unexpected semantics");
}

/// Equivalent of C standard library function.
///
/// While the C standard says Exp is an unspecified value for infinity and nan,
/// this returns INT_MAX for infinities, and INT_MIN for NaNs.
inline APFloat frexp(const APFloat &X, int &Exp, APFloat::roundingMode RM) {
  if (APFloat::usesLayout<detail::IEEEFloat>(X.getSemantics()))
    return APFloat(frexp(X.U.IEEE, Exp, RM), X.getSemantics());
  if (APFloat::usesLayout<detail::DoubleAPFloat>(X.getSemantics()))
    return APFloat(frexp(X.U.Double, Exp, RM), X.getSemantics());
  llvm_unreachable("Unexpected semantics");
}
/// Returns the absolute value of the argument.
inline APFloat abs(APFloat X) {
  X.clearSign();
  return X;
}

/// Returns the negated value of the argument.
inline APFloat neg(APFloat X) {
  X.changeSign();
  return X;
}

/// Implements IEEE-754 2008 minNum semantics. Returns the smaller of the
/// 2 arguments if both are not NaN. If either argument is a qNaN, returns the
/// other argument. If either argument is sNaN, return a qNaN.
/// -0 is treated as ordered less than +0.
LLVM_READONLY
inline APFloat minnum(const APFloat &A, const APFloat &B) {
  if (A.isSignaling())
    return A.makeQuiet();
  if (B.isSignaling())
    return B.makeQuiet();
  if (A.isNaN())
    return B;
  if (B.isNaN())
    return A;
  if (A.isZero() && B.isZero() && (A.isNegative() != B.isNegative()))
    return A.isNegative() ? A : B;
  return B < A ? B : A;
}

/// Implements IEEE-754 2008 maxNum semantics. Returns the larger of the
/// 2 arguments if both are not NaN. If either argument is a qNaN, returns the
/// other argument. If either argument is sNaN, return a qNaN.
/// +0 is treated as ordered greater than -0.
LLVM_READONLY
inline APFloat maxnum(const APFloat &A, const APFloat &B) {
  if (A.isSignaling())
    return A.makeQuiet();
  if (B.isSignaling())
    return B.makeQuiet();
  if (A.isNaN())
    return B;
  if (B.isNaN())
    return A;
  if (A.isZero() && B.isZero() && (A.isNegative() != B.isNegative()))
    return A.isNegative() ? B : A;
  return A < B ? B : A;
}

/// Implements IEEE 754-2019 minimum semantics. Returns the smaller of 2
/// arguments, returning a quiet NaN if an argument is a NaN and treating -0
/// as less than +0.
LLVM_READONLY
inline APFloat minimum(const APFloat &A, const APFloat &B) {
  if (A.isNaN())
    return A.makeQuiet();
  if (B.isNaN())
    return B.makeQuiet();
  if (A.isZero() && B.isZero() && (A.isNegative() != B.isNegative()))
    return A.isNegative() ? A : B;
  return B < A ? B : A;
}

/// Implements IEEE 754-2019 minimumNumber semantics. Returns the smaller
/// of 2 arguments, not propagating NaNs and treating -0 as less than +0.
LLVM_READONLY
inline APFloat minimumnum(const APFloat &A, const APFloat &B) {
  if (A.isNaN())
    return B.isNaN() ? B.makeQuiet() : B;
  if (B.isNaN())
    return A;
  if (A.isZero() && B.isZero() && (A.isNegative() != B.isNegative()))
    return A.isNegative() ? A : B;
  return B < A ? B : A;
}

/// Implements IEEE 754-2019 maximum semantics. Returns the larger of 2
/// arguments, returning a quiet NaN if an argument is a NaN and treating -0
/// as less than +0.
LLVM_READONLY
inline APFloat maximum(const APFloat &A, const APFloat &B) {
  if (A.isNaN())
    return A.makeQuiet();
  if (B.isNaN())
    return B.makeQuiet();
  if (A.isZero() && B.isZero() && (A.isNegative() != B.isNegative()))
    return A.isNegative() ? B : A;
  return A < B ? B : A;
}

/// Implements IEEE 754-2019 maximumNumber semantics. Returns the larger
/// of 2 arguments, not propagating NaNs and treating -0 as less than +0.
LLVM_READONLY
inline APFloat maximumnum(const APFloat &A, const APFloat &B) {
  if (A.isNaN())
    return B.isNaN() ? B.makeQuiet() : B;
  if (B.isNaN())
    return A;
  if (A.isZero() && B.isZero() && (A.isNegative() != B.isNegative()))
    return A.isNegative() ? B : A;
  return A < B ? B : A;
}

inline raw_ostream &operator<<(raw_ostream &OS, const APFloat &V) {
  V.print(OS);
  return OS;
}

// We want the following functions to be available in the header for inlining.
// We cannot define them inline in the class definition of `DoubleAPFloat`
// because doing so would instantiate `std::unique_ptr<APFloat[]>` before
// `APFloat` is defined, and that would be undefined behavior.
namespace detail {

DoubleAPFloat &DoubleAPFloat::operator=(DoubleAPFloat &&RHS) {
  if (this != &RHS) {
    this->~DoubleAPFloat();
    new (this) DoubleAPFloat(std::move(RHS));
  }
  return *this;
}

APFloat &DoubleAPFloat::getFirst() { return Floats[0]; }
const APFloat &DoubleAPFloat::getFirst() const { return Floats[0]; }
APFloat &DoubleAPFloat::getSecond() { return Floats[1]; }
const APFloat &DoubleAPFloat::getSecond() const { return Floats[1]; }

inline DoubleAPFloat::~DoubleAPFloat() { delete[] Floats; }

} // namespace detail

} // namespace llvm

#undef APFLOAT_DISPATCH_ON_SEMANTICS
#endif // LLVM_ADT_APFLOAT_H

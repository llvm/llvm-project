//== llvm/Support/APFloat.h - Arbitrary Precision Floating Point -*- C++ -*-==//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file declares a class to represent arbitrary precision floating
// point values and provide a variety of arithmetic operations on them.
//
//===----------------------------------------------------------------------===//

/*  A self-contained host- and target-independent arbitrary-precision
    floating-point software implementation.  It uses bignum integer
    arithmetic as provided by static functions in the APInt class.
    The library will work with bignum integers whose parts are any
    unsigned type at least 16 bits wide, but 64 bits is recommended.

    Written for clarity rather than speed, in particular with a view
    to use in the front-end of a cross compiler so that target
    arithmetic can be correctly performed on the host.  Performance
    should nonetheless be reasonable, particularly for its intended
    use.  It may be useful as a base implementation for a run-time
    library during development of a faster target-specific one.

    All 5 rounding modes in the IEEE-754R draft are handled correctly
    for all implemented operations.  Currently implemented operations
    are add, subtract, multiply, divide, fused-multiply-add,
    conversion-to-float, conversion-to-integer and
    conversion-from-integer.  New rounding modes (e.g. away from zero)
    can be added with three or four lines of code.

    Four formats are built-in: IEEE single precision, double
    precision, quadruple precision, and x87 80-bit extended double
    (when operating with full extended precision).  Adding a new
    format that obeys IEEE semantics only requires adding two lines of
    code: a declaration and definition of the format.

    All operations return the status of that operation as an exception
    bit-mask, so multiple operations can be done consecutively with
    their results or-ed together.  The returned status can be useful
    for compiler diagnostics; e.g., inexact, underflow and overflow
    can be easily diagnosed on constant folding, and compiler
    optimizers can determine what exceptions would be raised by
    folding operations and optimize, or perhaps not optimize,
    accordingly.

    At present, underflow tininess is detected after rounding; it
    should be straight forward to add support for the before-rounding
    case too.

    The library reads hexadecimal floating point numbers as per C99,
    and correctly rounds if necessary according to the specified
    rounding mode.  Syntax is required to have been validated by the
    caller.  It also converts floating point numbers to hexadecimal
    text as per the C99 %a and %A conversions.  The output precision
    (or alternatively the natural minimal precision) can be specified;
    if the requested precision is less than the natural precision the
    output is correctly rounded for the specified rounding mode.

    It also reads decimal floating point numbers and correctly rounds
    according to the specified rounding mode.

    Conversion to decimal text is not currently implemented.

    Non-zero finite numbers are represented internally as a sign bit,
    a 16-bit signed exponent, and the significand as an array of
    integer parts.  After normalization of a number of precision P the
    exponent is within the range of the format, and if the number is
    not denormal the P-th bit of the significand is set as an explicit
    integer bit.  For denormals the most significant bit is shifted
    right so that the exponent is maintained at the format's minimum,
    so that the smallest denormal has just the least significant bit
    of the significand set.  The sign of zeroes and infinities is
    significant; the exponent and significand of such numbers is not
    stored, but has a known implicit (deterministic) value: 0 for the
    significands, 0 for zero exponent, all 1 bits for infinity
    exponent.  For NaNs the sign and significand are deterministic,
    although not really meaningful, and preserved in non-conversion
    operations.  The exponent is implicitly all 1 bits.

    TODO
    ====

    Some features that may or may not be worth adding:

    Binary to decimal conversion (hard).

    Optional ability to detect underflow tininess before rounding.

    New formats: x87 in single and double precision mode (IEEE apart
    from extended exponent range) (hard).

    New operations: sqrt, IEEE remainder, C90 fmod, nextafter,
    nexttoward.
*/

#ifndef LLVM_FLOAT_H
#define LLVM_FLOAT_H

// APInt contains static functions implementing bignum arithmetic.
#include "llvm/ADT/APInt.h"

namespace llvm {

  /* Exponents are stored as signed numbers.  */
  typedef signed short exponent_t;

  struct fltSemantics;

  /* When bits of a floating point number are truncated, this enum is
     used to indicate what fraction of the LSB those bits represented.
     It essentially combines the roles of guard and sticky bits.  */
  enum lostFraction {           // Example of truncated bits:
    lfExactlyZero,              // 000000
    lfLessThanHalf,             // 0xxxxx  x's not all zero
    lfExactlyHalf,              // 100000
    lfMoreThanHalf              // 1xxxxx  x's not all zero
  };

  class APFloat {
  public:

    /* We support the following floating point semantics.  */
    static const fltSemantics IEEEsingle;
    static const fltSemantics IEEEdouble;
    static const fltSemantics IEEEquad;
    static const fltSemantics PPCDoubleDouble;
    static const fltSemantics x87DoubleExtended;
    /* And this pseudo, used to construct APFloats that cannot
       conflict with anything real. */
    static const fltSemantics Bogus;

    static unsigned int semanticsPrecision(const fltSemantics &);

    /* Floating point numbers have a four-state comparison relation.  */
    enum cmpResult {
      cmpLessThan,
      cmpEqual,
      cmpGreaterThan,
      cmpUnordered
    };

    /* IEEE-754R gives five rounding modes.  */
    enum roundingMode {
      rmNearestTiesToEven,
      rmTowardPositive,
      rmTowardNegative,
      rmTowardZero,
      rmNearestTiesToAway
    };

    // Operation status.  opUnderflow or opOverflow are always returned
    // or-ed with opInexact.
    enum opStatus {
      opOK          = 0x00,
      opInvalidOp   = 0x01,
      opDivByZero   = 0x02,
      opOverflow    = 0x04,
      opUnderflow   = 0x08,
      opInexact     = 0x10
    };

    // Category of internally-represented number.
    enum fltCategory {
      fcInfinity,
      fcNaN,
      fcNormal,
      fcZero
    };

    // Constructors.
    APFloat(const fltSemantics &, const char *);
    APFloat(const fltSemantics &, integerPart);
    APFloat(const fltSemantics &, fltCategory, bool negative);
    explicit APFloat(double d);
    explicit APFloat(float f);
    explicit APFloat(const APInt &, bool isIEEE = false);
    APFloat(const APFloat &);
    ~APFloat();

    // Convenience "constructors"
    static APFloat getZero(const fltSemantics &Sem, bool Negative = false) {
      return APFloat(Sem, fcZero, Negative);
    }
    static APFloat getInf(const fltSemantics &Sem, bool Negative = false) {
      return APFloat(Sem, fcInfinity, Negative);
    }
    static APFloat getNaN(const fltSemantics &Sem, bool Negative = false) {
      return APFloat(Sem, fcNaN, Negative);
    }

    /// Profile - Used to insert APFloat objects, or objects that contain
    ///  APFloat objects, into FoldingSets.
    void Profile(FoldingSetNodeID& NID) const;

    /// @brief Used by the Bitcode serializer to emit APInts to Bitcode.
    void Emit(Serializer& S) const;

    /// @brief Used by the Bitcode deserializer to deserialize APInts.
    static APFloat ReadVal(Deserializer& D);

    /* Arithmetic.  */
    opStatus add(const APFloat &, roundingMode);
    opStatus subtract(const APFloat &, roundingMode);
    opStatus multiply(const APFloat &, roundingMode);
    opStatus divide(const APFloat &, roundingMode);
    /* IEEE remainder. */
    opStatus remainder(const APFloat &);
    /* C fmod, or llvm frem. */
    opStatus mod(const APFloat &, roundingMode);
    opStatus fusedMultiplyAdd(const APFloat &, const APFloat &, roundingMode);

    /* Sign operations.  */
    void changeSign();
    void clearSign();
    void copySign(const APFloat &);

    /* Conversions.  */
    opStatus convert(const fltSemantics &, roundingMode, bool *);
    opStatus convertToInteger(integerPart *, unsigned int, bool,
                              roundingMode, bool *) const;
    opStatus convertFromAPInt(const APInt &,
                              bool, roundingMode);
    opStatus convertFromSignExtendedInteger(const integerPart *, unsigned int,
                                            bool, roundingMode);
    opStatus convertFromZeroExtendedInteger(const integerPart *, unsigned int,
                                            bool, roundingMode);
    opStatus convertFromString(const char *, roundingMode);
    APInt bitcastToAPInt() const;
    double convertToDouble() const;
    float convertToFloat() const;

    /* The definition of equality is not straightforward for floating point,
       so we won't use operator==.  Use one of the following, or write
       whatever it is you really mean. */
    // bool operator==(const APFloat &) const;     // DO NOT IMPLEMENT

    /* IEEE comparison with another floating point number (NaNs
       compare unordered, 0==-0). */
    cmpResult compare(const APFloat &) const;

    /* Bitwise comparison for equality (QNaNs compare equal, 0!=-0). */
    bool bitwiseIsEqual(const APFloat &) const;

    /* Write out a hexadecimal representation of the floating point
       value to DST, which must be of sufficient size, in the C99 form
       [-]0xh.hhhhp[+-]d.  Return the number of characters written,
       excluding the terminating NUL.  */
    unsigned int convertToHexString(char *dst, unsigned int hexDigits,
                                    bool upperCase, roundingMode) const;

    /* Simple queries.  */
    fltCategory getCategory() const { return category; }
    const fltSemantics &getSemantics() const { return *semantics; }
    bool isZero() const { return category == fcZero; }
    bool isNonZero() const { return category != fcZero; }
    bool isNaN() const { return category == fcNaN; }
    bool isInfinity() const { return category == fcInfinity; }
    bool isNegative() const { return sign; }
    bool isPosZero() const { return isZero() && !isNegative(); }
    bool isNegZero() const { return isZero() && isNegative(); }

    APFloat& operator=(const APFloat &);

    /* Return an arbitrary integer value usable for hashing. */
    uint32_t getHashValue() const;

  private:

    /* Trivial queries.  */
    integerPart *significandParts();
    const integerPart *significandParts() const;
    unsigned int partCount() const;

    /* Significand operations.  */
    integerPart addSignificand(const APFloat &);
    integerPart subtractSignificand(const APFloat &, integerPart);
    lostFraction addOrSubtractSignificand(const APFloat &, bool subtract);
    lostFraction multiplySignificand(const APFloat &, const APFloat *);
    lostFraction divideSignificand(const APFloat &);
    void incrementSignificand();
    void initialize(const fltSemantics *);
    void shiftSignificandLeft(unsigned int);
    lostFraction shiftSignificandRight(unsigned int);
    unsigned int significandLSB() const;
    unsigned int significandMSB() const;
    void zeroSignificand();

    /* Arithmetic on special values.  */
    opStatus addOrSubtractSpecials(const APFloat &, bool subtract);
    opStatus divideSpecials(const APFloat &);
    opStatus multiplySpecials(const APFloat &);
    opStatus modSpecials(const APFloat &);

    /* Miscellany.  */
    void makeNaN(void);
    opStatus normalize(roundingMode, lostFraction);
    opStatus addOrSubtract(const APFloat &, roundingMode, bool subtract);
    cmpResult compareAbsoluteValue(const APFloat &) const;
    opStatus handleOverflow(roundingMode);
    bool roundAwayFromZero(roundingMode, lostFraction, unsigned int) const;
    opStatus convertToSignExtendedInteger(integerPart *, unsigned int, bool,
                                          roundingMode, bool *) const;
    opStatus convertFromUnsignedParts(const integerPart *, unsigned int,
                                      roundingMode);
    opStatus convertFromHexadecimalString(const char *, roundingMode);
    opStatus convertFromDecimalString (const char *, roundingMode);
    char *convertNormalToHexString(char *, unsigned int, bool,
                                   roundingMode) const;
    opStatus roundSignificandWithExponent(const integerPart *, unsigned int,
                                          int, roundingMode);

    APInt convertFloatAPFloatToAPInt() const;
    APInt convertDoubleAPFloatToAPInt() const;
    APInt convertF80LongDoubleAPFloatToAPInt() const;
    APInt convertPPCDoubleDoubleAPFloatToAPInt() const;
    void initFromAPInt(const APInt& api, bool isIEEE = false);
    void initFromFloatAPInt(const APInt& api);
    void initFromDoubleAPInt(const APInt& api);
    void initFromF80LongDoubleAPInt(const APInt& api);
    void initFromPPCDoubleDoubleAPInt(const APInt& api);

    void assign(const APFloat &);
    void copySignificand(const APFloat &);
    void freeSignificand();

    /* What kind of semantics does this value obey?  */
    const fltSemantics *semantics;

    /* Significand - the fraction with an explicit integer bit.  Must be
       at least one bit wider than the target precision.  */
    union Significand
    {
      integerPart part;
      integerPart *parts;
    } significand;

    /* The exponent - a signed number.  */
    exponent_t exponent;

    /* What kind of floating point number this is.  */
    /* Only 2 bits are required, but VisualStudio incorrectly sign extends
       it.  Using the extra bit keeps it from failing under VisualStudio */
    fltCategory category: 3;

    /* The sign bit of this number.  */
    unsigned int sign: 1;

    /* For PPCDoubleDouble, we have a second exponent and sign (the second
       significand is appended to the first one, although it would be wrong to
       regard these as a single number for arithmetic purposes).  These fields
       are not meaningful for any other type. */
    exponent_t exponent2 : 11;
    unsigned int sign2: 1;
  };
} /* namespace llvm */

#endif /* LLVM_FLOAT_H */

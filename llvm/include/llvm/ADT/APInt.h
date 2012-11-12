//===-- llvm/ADT/APInt.h - For Arbitrary Precision Integer -----*- C++ -*--===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file implements a class to represent arbitrary precision integral
// constant values and operations on them.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_APINT_H
#define LLVM_APINT_H

#include "llvm/ADT/ArrayRef.h"
#include "llvm/Support/Compiler.h"
#include "llvm/Support/MathExtras.h"
#include <cassert>
#include <climits>
#include <cstring>
#include <string>

namespace llvm {
  class Deserializer;
  class FoldingSetNodeID;
  class Serializer;
  class StringRef;
  class hash_code;
  class raw_ostream;

  template<typename T>
  class SmallVectorImpl;

  // An unsigned host type used as a single part of a multi-part
  // bignum.
  typedef uint64_t integerPart;

  const unsigned int host_char_bit = 8;
  const unsigned int integerPartWidth = host_char_bit *
    static_cast<unsigned int>(sizeof(integerPart));

//===----------------------------------------------------------------------===//
//                              APInt Class
//===----------------------------------------------------------------------===//

/// APInt - This class represents arbitrary precision constant integral values.
/// It is a functional replacement for common case unsigned integer type like
/// "unsigned", "unsigned long" or "uint64_t", but also allows non-byte-width
/// integer sizes and large integer value types such as 3-bits, 15-bits, or more
/// than 64-bits of precision. APInt provides a variety of arithmetic operators
/// and methods to manipulate integer values of any bit-width. It supports both
/// the typical integer arithmetic and comparison operations as well as bitwise
/// manipulation.
///
/// The class has several invariants worth noting:
///   * All bit, byte, and word positions are zero-based.
///   * Once the bit width is set, it doesn't change except by the Truncate,
///     SignExtend, or ZeroExtend operations.
///   * All binary operators must be on APInt instances of the same bit width.
///     Attempting to use these operators on instances with different bit
///     widths will yield an assertion.
///   * The value is stored canonically as an unsigned value. For operations
///     where it makes a difference, there are both signed and unsigned variants
///     of the operation. For example, sdiv and udiv. However, because the bit
///     widths must be the same, operations such as Mul and Add produce the same
///     results regardless of whether the values are interpreted as signed or
///     not.
///   * In general, the class tries to follow the style of computation that LLVM
///     uses in its IR. This simplifies its use for LLVM.
///
/// @brief Class for arbitrary precision integers.
class APInt {
  unsigned BitWidth;      ///< The number of bits in this APInt.

  /// This union is used to store the integer value. When the
  /// integer bit-width <= 64, it uses VAL, otherwise it uses pVal.
  union {
    uint64_t VAL;    ///< Used to store the <= 64 bits integer value.
    uint64_t *pVal;  ///< Used to store the >64 bits integer value.
  };

  /// This enum is used to hold the constants we needed for APInt.
  enum {
    /// Bits in a word
    APINT_BITS_PER_WORD = static_cast<unsigned int>(sizeof(uint64_t)) *
                          CHAR_BIT,
    /// Byte size of a word
    APINT_WORD_SIZE = static_cast<unsigned int>(sizeof(uint64_t))
  };

  /// This constructor is used only internally for speed of construction of
  /// temporaries. It is unsafe for general use so it is not public.
  /// @brief Fast internal constructor
  APInt(uint64_t* val, unsigned bits) : BitWidth(bits), pVal(val) { }

  /// @returns true if the number of bits <= 64, false otherwise.
  /// @brief Determine if this APInt just has one word to store value.
  bool isSingleWord() const {
    return BitWidth <= APINT_BITS_PER_WORD;
  }

  /// @returns the word position for the specified bit position.
  /// @brief Determine which word a bit is in.
  static unsigned whichWord(unsigned bitPosition) {
    return bitPosition / APINT_BITS_PER_WORD;
  }

  /// @returns the bit position in a word for the specified bit position
  /// in the APInt.
  /// @brief Determine which bit in a word a bit is in.
  static unsigned whichBit(unsigned bitPosition) {
    return bitPosition % APINT_BITS_PER_WORD;
  }

  /// This method generates and returns a uint64_t (word) mask for a single
  /// bit at a specific bit position. This is used to mask the bit in the
  /// corresponding word.
  /// @returns a uint64_t with only bit at "whichBit(bitPosition)" set
  /// @brief Get a single bit mask.
  static uint64_t maskBit(unsigned bitPosition) {
    return 1ULL << whichBit(bitPosition);
  }

  /// This method is used internally to clear the to "N" bits in the high order
  /// word that are not used by the APInt. This is needed after the most
  /// significant word is assigned a value to ensure that those bits are
  /// zero'd out.
  /// @brief Clear unused high order bits
  APInt& clearUnusedBits() {
    // Compute how many bits are used in the final word
    unsigned wordBits = BitWidth % APINT_BITS_PER_WORD;
    if (wordBits == 0)
      // If all bits are used, we want to leave the value alone. This also
      // avoids the undefined behavior of >> when the shift is the same size as
      // the word size (64).
      return *this;

    // Mask out the high bits.
    uint64_t mask = ~uint64_t(0ULL) >> (APINT_BITS_PER_WORD - wordBits);
    if (isSingleWord())
      VAL &= mask;
    else
      pVal[getNumWords() - 1] &= mask;
    return *this;
  }

  /// @returns the corresponding word for the specified bit position.
  /// @brief Get the word corresponding to a bit position
  uint64_t getWord(unsigned bitPosition) const {
    return isSingleWord() ? VAL : pVal[whichWord(bitPosition)];
  }

  /// Converts a string into a number.  The string must be non-empty
  /// and well-formed as a number of the given base. The bit-width
  /// must be sufficient to hold the result.
  ///
  /// This is used by the constructors that take string arguments.
  ///
  /// StringRef::getAsInteger is superficially similar but (1) does
  /// not assume that the string is well-formed and (2) grows the
  /// result to hold the input.
  ///
  /// @param radix 2, 8, 10, 16, or 36
  /// @brief Convert a char array into an APInt
  void fromString(unsigned numBits, StringRef str, uint8_t radix);

  /// This is used by the toString method to divide by the radix. It simply
  /// provides a more convenient form of divide for internal use since KnuthDiv
  /// has specific constraints on its inputs. If those constraints are not met
  /// then it provides a simpler form of divide.
  /// @brief An internal division function for dividing APInts.
  static void divide(const APInt LHS, unsigned lhsWords,
                     const APInt &RHS, unsigned rhsWords,
                     APInt *Quotient, APInt *Remainder);

  /// out-of-line slow case for inline constructor
  void initSlowCase(unsigned numBits, uint64_t val, bool isSigned);

  /// shared code between two array constructors
  void initFromArray(ArrayRef<uint64_t> array);

  /// out-of-line slow case for inline copy constructor
  void initSlowCase(const APInt& that);

  /// out-of-line slow case for shl
  APInt shlSlowCase(unsigned shiftAmt) const;

  /// out-of-line slow case for operator&
  APInt AndSlowCase(const APInt& RHS) const;

  /// out-of-line slow case for operator|
  APInt OrSlowCase(const APInt& RHS) const;

  /// out-of-line slow case for operator^
  APInt XorSlowCase(const APInt& RHS) const;

  /// out-of-line slow case for operator=
  APInt& AssignSlowCase(const APInt& RHS);

  /// out-of-line slow case for operator==
  bool EqualSlowCase(const APInt& RHS) const;

  /// out-of-line slow case for operator==
  bool EqualSlowCase(uint64_t Val) const;

  /// out-of-line slow case for countLeadingZeros
  unsigned countLeadingZerosSlowCase() const;

  /// out-of-line slow case for countTrailingOnes
  unsigned countTrailingOnesSlowCase() const;

  /// out-of-line slow case for countPopulation
  unsigned countPopulationSlowCase() const;

public:
  /// @name Constructors
  /// @{
  /// If isSigned is true then val is treated as if it were a signed value
  /// (i.e. as an int64_t) and the appropriate sign extension to the bit width
  /// will be done. Otherwise, no sign extension occurs (high order bits beyond
  /// the range of val are zero filled).
  /// @param numBits the bit width of the constructed APInt
  /// @param val the initial value of the APInt
  /// @param isSigned how to treat signedness of val
  /// @brief Create a new APInt of numBits width, initialized as val.
  APInt(unsigned numBits, uint64_t val, bool isSigned = false)
    : BitWidth(numBits), VAL(0) {
    assert(BitWidth && "bitwidth too small");
    if (isSingleWord())
      VAL = val;
    else
      initSlowCase(numBits, val, isSigned);
    clearUnusedBits();
  }

  /// Note that bigVal.size() can be smaller or larger than the corresponding
  /// bit width but any extraneous bits will be dropped.
  /// @param numBits the bit width of the constructed APInt
  /// @param bigVal a sequence of words to form the initial value of the APInt
  /// @brief Construct an APInt of numBits width, initialized as bigVal[].
  APInt(unsigned numBits, ArrayRef<uint64_t> bigVal);
  /// Equivalent to APInt(numBits, ArrayRef<uint64_t>(bigVal, numWords)), but
  /// deprecated because this constructor is prone to ambiguity with the
  /// APInt(unsigned, uint64_t, bool) constructor.
  ///
  /// If this overload is ever deleted, care should be taken to prevent calls
  /// from being incorrectly captured by the APInt(unsigned, uint64_t, bool)
  /// constructor.
  APInt(unsigned numBits, unsigned numWords, const uint64_t bigVal[]);

  /// This constructor interprets the string \p str in the given radix. The
  /// interpretation stops when the first character that is not suitable for the
  /// radix is encountered, or the end of the string. Acceptable radix values
  /// are 2, 8, 10, 16, and 36. It is an error for the value implied by the 
  /// string to require more bits than numBits.
  ///
  /// @param numBits the bit width of the constructed APInt
  /// @param str the string to be interpreted
  /// @param radix the radix to use for the conversion 
  /// @brief Construct an APInt from a string representation.
  APInt(unsigned numBits, StringRef str, uint8_t radix);

  /// Simply makes *this a copy of that.
  /// @brief Copy Constructor.
  APInt(const APInt& that)
    : BitWidth(that.BitWidth), VAL(0) {
    assert(BitWidth && "bitwidth too small");
    if (isSingleWord())
      VAL = that.VAL;
    else
      initSlowCase(that);
  }

#if LLVM_USE_RVALUE_REFERENCES
  /// @brief Move Constructor.
  APInt(APInt&& that) : BitWidth(that.BitWidth), VAL(that.VAL) {
    that.BitWidth = 0;
  }
#endif

  /// @brief Destructor.
  ~APInt() {
    if (!isSingleWord())
      delete [] pVal;
  }

  /// Default constructor that creates an uninitialized APInt.  This is useful
  ///  for object deserialization (pair this with the static method Read).
  explicit APInt() : BitWidth(1) {}

  /// Profile - Used to insert APInt objects, or objects that contain APInt
  ///  objects, into FoldingSets.
  void Profile(FoldingSetNodeID& id) const;

  /// @}
  /// @name Value Tests
  /// @{
  /// This tests the high bit of this APInt to determine if it is set.
  /// @returns true if this APInt is negative, false otherwise
  /// @brief Determine sign of this APInt.
  bool isNegative() const {
    return (*this)[BitWidth - 1];
  }

  /// This tests the high bit of the APInt to determine if it is unset.
  /// @brief Determine if this APInt Value is non-negative (>= 0)
  bool isNonNegative() const {
    return !isNegative();
  }

  /// This tests if the value of this APInt is positive (> 0). Note
  /// that 0 is not a positive value.
  /// @returns true if this APInt is positive.
  /// @brief Determine if this APInt Value is positive.
  bool isStrictlyPositive() const {
    return isNonNegative() && !!*this;
  }

  /// This checks to see if the value has all bits of the APInt are set or not.
  /// @brief Determine if all bits are set
  bool isAllOnesValue() const {
    return countPopulation() == BitWidth;
  }

  /// This checks to see if the value of this APInt is the maximum unsigned
  /// value for the APInt's bit width.
  /// @brief Determine if this is the largest unsigned value.
  bool isMaxValue() const {
    return countPopulation() == BitWidth;
  }

  /// This checks to see if the value of this APInt is the maximum signed
  /// value for the APInt's bit width.
  /// @brief Determine if this is the largest signed value.
  bool isMaxSignedValue() const {
    return BitWidth == 1 ? VAL == 0 :
                          !isNegative() && countPopulation() == BitWidth - 1;
  }

  /// This checks to see if the value of this APInt is the minimum unsigned
  /// value for the APInt's bit width.
  /// @brief Determine if this is the smallest unsigned value.
  bool isMinValue() const {
    return !*this;
  }

  /// This checks to see if the value of this APInt is the minimum signed
  /// value for the APInt's bit width.
  /// @brief Determine if this is the smallest signed value.
  bool isMinSignedValue() const {
    return BitWidth == 1 ? VAL == 1 : isNegative() && isPowerOf2();
  }

  /// @brief Check if this APInt has an N-bits unsigned integer value.
  bool isIntN(unsigned N) const {
    assert(N && "N == 0 ???");
    return getActiveBits() <= N;
  }

  /// @brief Check if this APInt has an N-bits signed integer value.
  bool isSignedIntN(unsigned N) const {
    assert(N && "N == 0 ???");
    return getMinSignedBits() <= N;
  }

  /// @returns true if the argument APInt value is a power of two > 0.
  bool isPowerOf2() const {
    if (isSingleWord())
      return isPowerOf2_64(VAL);
    return countPopulationSlowCase() == 1;
  }

  /// isSignBit - Return true if this is the value returned by getSignBit.
  bool isSignBit() const { return isMinSignedValue(); }

  /// This converts the APInt to a boolean value as a test against zero.
  /// @brief Boolean conversion function.
  bool getBoolValue() const {
    return !!*this;
  }

  /// getLimitedValue - If this value is smaller than the specified limit,
  /// return it, otherwise return the limit value.  This causes the value
  /// to saturate to the limit.
  uint64_t getLimitedValue(uint64_t Limit = ~0ULL) const {
    return (getActiveBits() > 64 || getZExtValue() > Limit) ?
      Limit :  getZExtValue();
  }

  /// @}
  /// @name Value Generators
  /// @{
  /// @brief Gets maximum unsigned value of APInt for specific bit width.
  static APInt getMaxValue(unsigned numBits) {
    return getAllOnesValue(numBits);
  }

  /// @brief Gets maximum signed value of APInt for a specific bit width.
  static APInt getSignedMaxValue(unsigned numBits) {
    APInt API = getAllOnesValue(numBits);
    API.clearBit(numBits - 1);
    return API;
  }

  /// @brief Gets minimum unsigned value of APInt for a specific bit width.
  static APInt getMinValue(unsigned numBits) {
    return APInt(numBits, 0);
  }

  /// @brief Gets minimum signed value of APInt for a specific bit width.
  static APInt getSignedMinValue(unsigned numBits) {
    APInt API(numBits, 0);
    API.setBit(numBits - 1);
    return API;
  }

  /// getSignBit - This is just a wrapper function of getSignedMinValue(), and
  /// it helps code readability when we want to get a SignBit.
  /// @brief Get the SignBit for a specific bit width.
  static APInt getSignBit(unsigned BitWidth) {
    return getSignedMinValue(BitWidth);
  }

  /// @returns the all-ones value for an APInt of the specified bit-width.
  /// @brief Get the all-ones value.
  static APInt getAllOnesValue(unsigned numBits) {
    return APInt(numBits, -1ULL, true);
  }

  /// @returns the '0' value for an APInt of the specified bit-width.
  /// @brief Get the '0' value.
  static APInt getNullValue(unsigned numBits) {
    return APInt(numBits, 0);
  }

  /// Get an APInt with the same BitWidth as this APInt, just zero mask
  /// the low bits and right shift to the least significant bit.
  /// @returns the high "numBits" bits of this APInt.
  APInt getHiBits(unsigned numBits) const;

  /// Get an APInt with the same BitWidth as this APInt, just zero mask
  /// the high bits.
  /// @returns the low "numBits" bits of this APInt.
  APInt getLoBits(unsigned numBits) const;

  /// getOneBitSet - Return an APInt with exactly one bit set in the result.
  static APInt getOneBitSet(unsigned numBits, unsigned BitNo) {
    APInt Res(numBits, 0);
    Res.setBit(BitNo);
    return Res;
  }
  
  /// Constructs an APInt value that has a contiguous range of bits set. The
  /// bits from loBit (inclusive) to hiBit (exclusive) will be set. All other
  /// bits will be zero. For example, with parameters(32, 0, 16) you would get
  /// 0x0000FFFF. If hiBit is less than loBit then the set bits "wrap". For
  /// example, with parameters (32, 28, 4), you would get 0xF000000F.
  /// @param numBits the intended bit width of the result
  /// @param loBit the index of the lowest bit set.
  /// @param hiBit the index of the highest bit set.
  /// @returns An APInt value with the requested bits set.
  /// @brief Get a value with a block of bits set.
  static APInt getBitsSet(unsigned numBits, unsigned loBit, unsigned hiBit) {
    assert(hiBit <= numBits && "hiBit out of range");
    assert(loBit < numBits && "loBit out of range");
    if (hiBit < loBit)
      return getLowBitsSet(numBits, hiBit) |
             getHighBitsSet(numBits, numBits-loBit);
    return getLowBitsSet(numBits, hiBit-loBit).shl(loBit);
  }

  /// Constructs an APInt value that has the top hiBitsSet bits set.
  /// @param numBits the bitwidth of the result
  /// @param hiBitsSet the number of high-order bits set in the result.
  /// @brief Get a value with high bits set
  static APInt getHighBitsSet(unsigned numBits, unsigned hiBitsSet) {
    assert(hiBitsSet <= numBits && "Too many bits to set!");
    // Handle a degenerate case, to avoid shifting by word size
    if (hiBitsSet == 0)
      return APInt(numBits, 0);
    unsigned shiftAmt = numBits - hiBitsSet;
    // For small values, return quickly
    if (numBits <= APINT_BITS_PER_WORD)
      return APInt(numBits, ~0ULL << shiftAmt);
    return getAllOnesValue(numBits).shl(shiftAmt);
  }

  /// Constructs an APInt value that has the bottom loBitsSet bits set.
  /// @param numBits the bitwidth of the result
  /// @param loBitsSet the number of low-order bits set in the result.
  /// @brief Get a value with low bits set
  static APInt getLowBitsSet(unsigned numBits, unsigned loBitsSet) {
    assert(loBitsSet <= numBits && "Too many bits to set!");
    // Handle a degenerate case, to avoid shifting by word size
    if (loBitsSet == 0)
      return APInt(numBits, 0);
    if (loBitsSet == APINT_BITS_PER_WORD)
      return APInt(numBits, -1ULL);
    // For small values, return quickly.
    if (loBitsSet <= APINT_BITS_PER_WORD)
      return APInt(numBits, -1ULL >> (APINT_BITS_PER_WORD - loBitsSet));
    return getAllOnesValue(numBits).lshr(numBits - loBitsSet);
  }

  /// \brief Determine if two APInts have the same value, after zero-extending
  /// one of them (if needed!) to ensure that the bit-widths match.
  static bool isSameValue(const APInt &I1, const APInt &I2) {
    if (I1.getBitWidth() == I2.getBitWidth())
      return I1 == I2;

    if (I1.getBitWidth() > I2.getBitWidth())
      return I1 == I2.zext(I1.getBitWidth());

    return I1.zext(I2.getBitWidth()) == I2;
  }
  
  /// \brief Overload to compute a hash_code for an APInt value.
  friend hash_code hash_value(const APInt &Arg);

  /// This function returns a pointer to the internal storage of the APInt.
  /// This is useful for writing out the APInt in binary form without any
  /// conversions.
  const uint64_t* getRawData() const {
    if (isSingleWord())
      return &VAL;
    return &pVal[0];
  }

  /// @}
  /// @name Unary Operators
  /// @{
  /// @returns a new APInt value representing *this incremented by one
  /// @brief Postfix increment operator.
  const APInt operator++(int) {
    APInt API(*this);
    ++(*this);
    return API;
  }

  /// @returns *this incremented by one
  /// @brief Prefix increment operator.
  APInt& operator++();

  /// @returns a new APInt representing *this decremented by one.
  /// @brief Postfix decrement operator.
  const APInt operator--(int) {
    APInt API(*this);
    --(*this);
    return API;
  }

  /// @returns *this decremented by one.
  /// @brief Prefix decrement operator.
  APInt& operator--();

  /// Performs a bitwise complement operation on this APInt.
  /// @returns an APInt that is the bitwise complement of *this
  /// @brief Unary bitwise complement operator.
  APInt operator~() const {
    APInt Result(*this);
    Result.flipAllBits();
    return Result;
  }

  /// Negates *this using two's complement logic.
  /// @returns An APInt value representing the negation of *this.
  /// @brief Unary negation operator
  APInt operator-() const {
    return APInt(BitWidth, 0) - (*this);
  }

  /// Performs logical negation operation on this APInt.
  /// @returns true if *this is zero, false otherwise.
  /// @brief Logical negation operator.
  bool operator!() const {
    if (isSingleWord())
      return !VAL;

    for (unsigned i = 0; i != getNumWords(); ++i)
      if (pVal[i])
        return false;
    return true;
  }

  /// @}
  /// @name Assignment Operators
  /// @{
  /// @returns *this after assignment of RHS.
  /// @brief Copy assignment operator.
  APInt& operator=(const APInt& RHS) {
    // If the bitwidths are the same, we can avoid mucking with memory
    if (isSingleWord() && RHS.isSingleWord()) {
      VAL = RHS.VAL;
      BitWidth = RHS.BitWidth;
      return clearUnusedBits();
    }

    return AssignSlowCase(RHS);
  }

#if LLVM_USE_RVALUE_REFERENCES
  /// @brief Move assignment operator.
  APInt& operator=(APInt&& that) {
    if (!isSingleWord())
      delete [] pVal;

    BitWidth = that.BitWidth;
    VAL = that.VAL;

    that.BitWidth = 0;

    return *this;
  }
#endif

  /// The RHS value is assigned to *this. If the significant bits in RHS exceed
  /// the bit width, the excess bits are truncated. If the bit width is larger
  /// than 64, the value is zero filled in the unspecified high order bits.
  /// @returns *this after assignment of RHS value.
  /// @brief Assignment operator.
  APInt& operator=(uint64_t RHS);

  /// Performs a bitwise AND operation on this APInt and RHS. The result is
  /// assigned to *this.
  /// @returns *this after ANDing with RHS.
  /// @brief Bitwise AND assignment operator.
  APInt& operator&=(const APInt& RHS);

  /// Performs a bitwise OR operation on this APInt and RHS. The result is
  /// assigned *this;
  /// @returns *this after ORing with RHS.
  /// @brief Bitwise OR assignment operator.
  APInt& operator|=(const APInt& RHS);

  /// Performs a bitwise OR operation on this APInt and RHS. RHS is
  /// logically zero-extended or truncated to match the bit-width of
  /// the LHS.
  /// 
  /// @brief Bitwise OR assignment operator.
  APInt& operator|=(uint64_t RHS) {
    if (isSingleWord()) {
      VAL |= RHS;
      clearUnusedBits();
    } else {
      pVal[0] |= RHS;
    }
    return *this;
  }

  /// Performs a bitwise XOR operation on this APInt and RHS. The result is
  /// assigned to *this.
  /// @returns *this after XORing with RHS.
  /// @brief Bitwise XOR assignment operator.
  APInt& operator^=(const APInt& RHS);

  /// Multiplies this APInt by RHS and assigns the result to *this.
  /// @returns *this
  /// @brief Multiplication assignment operator.
  APInt& operator*=(const APInt& RHS);

  /// Adds RHS to *this and assigns the result to *this.
  /// @returns *this
  /// @brief Addition assignment operator.
  APInt& operator+=(const APInt& RHS);

  /// Subtracts RHS from *this and assigns the result to *this.
  /// @returns *this
  /// @brief Subtraction assignment operator.
  APInt& operator-=(const APInt& RHS);

  /// Shifts *this left by shiftAmt and assigns the result to *this.
  /// @returns *this after shifting left by shiftAmt
  /// @brief Left-shift assignment function.
  APInt& operator<<=(unsigned shiftAmt) {
    *this = shl(shiftAmt);
    return *this;
  }

  /// @}
  /// @name Binary Operators
  /// @{
  /// Performs a bitwise AND operation on *this and RHS.
  /// @returns An APInt value representing the bitwise AND of *this and RHS.
  /// @brief Bitwise AND operator.
  APInt operator&(const APInt& RHS) const {
    assert(BitWidth == RHS.BitWidth && "Bit widths must be the same");
    if (isSingleWord())
      return APInt(getBitWidth(), VAL & RHS.VAL);
    return AndSlowCase(RHS);
  }
  APInt And(const APInt& RHS) const {
    return this->operator&(RHS);
  }

  /// Performs a bitwise OR operation on *this and RHS.
  /// @returns An APInt value representing the bitwise OR of *this and RHS.
  /// @brief Bitwise OR operator.
  APInt operator|(const APInt& RHS) const {
    assert(BitWidth == RHS.BitWidth && "Bit widths must be the same");
    if (isSingleWord())
      return APInt(getBitWidth(), VAL | RHS.VAL);
    return OrSlowCase(RHS);
  }
  APInt Or(const APInt& RHS) const {
    return this->operator|(RHS);
  }

  /// Performs a bitwise XOR operation on *this and RHS.
  /// @returns An APInt value representing the bitwise XOR of *this and RHS.
  /// @brief Bitwise XOR operator.
  APInt operator^(const APInt& RHS) const {
    assert(BitWidth == RHS.BitWidth && "Bit widths must be the same");
    if (isSingleWord())
      return APInt(BitWidth, VAL ^ RHS.VAL);
    return XorSlowCase(RHS);
  }
  APInt Xor(const APInt& RHS) const {
    return this->operator^(RHS);
  }

  /// Multiplies this APInt by RHS and returns the result.
  /// @brief Multiplication operator.
  APInt operator*(const APInt& RHS) const;

  /// Adds RHS to this APInt and returns the result.
  /// @brief Addition operator.
  APInt operator+(const APInt& RHS) const;
  APInt operator+(uint64_t RHS) const {
    return (*this) + APInt(BitWidth, RHS);
  }

  /// Subtracts RHS from this APInt and returns the result.
  /// @brief Subtraction operator.
  APInt operator-(const APInt& RHS) const;
  APInt operator-(uint64_t RHS) const {
    return (*this) - APInt(BitWidth, RHS);
  }

  APInt operator<<(unsigned Bits) const {
    return shl(Bits);
  }

  APInt operator<<(const APInt &Bits) const {
    return shl(Bits);
  }

  /// Arithmetic right-shift this APInt by shiftAmt.
  /// @brief Arithmetic right-shift function.
  APInt ashr(unsigned shiftAmt) const;

  /// Logical right-shift this APInt by shiftAmt.
  /// @brief Logical right-shift function.
  APInt lshr(unsigned shiftAmt) const;

  /// Left-shift this APInt by shiftAmt.
  /// @brief Left-shift function.
  APInt shl(unsigned shiftAmt) const {
    assert(shiftAmt <= BitWidth && "Invalid shift amount");
    if (isSingleWord()) {
      if (shiftAmt >= BitWidth)
        return APInt(BitWidth, 0); // avoid undefined shift results
      return APInt(BitWidth, VAL << shiftAmt);
    }
    return shlSlowCase(shiftAmt);
  }

  /// @brief Rotate left by rotateAmt.
  APInt rotl(unsigned rotateAmt) const;

  /// @brief Rotate right by rotateAmt.
  APInt rotr(unsigned rotateAmt) const;

  /// Arithmetic right-shift this APInt by shiftAmt.
  /// @brief Arithmetic right-shift function.
  APInt ashr(const APInt &shiftAmt) const;

  /// Logical right-shift this APInt by shiftAmt.
  /// @brief Logical right-shift function.
  APInt lshr(const APInt &shiftAmt) const;

  /// Left-shift this APInt by shiftAmt.
  /// @brief Left-shift function.
  APInt shl(const APInt &shiftAmt) const;

  /// @brief Rotate left by rotateAmt.
  APInt rotl(const APInt &rotateAmt) const;

  /// @brief Rotate right by rotateAmt.
  APInt rotr(const APInt &rotateAmt) const;

  /// Perform an unsigned divide operation on this APInt by RHS. Both this and
  /// RHS are treated as unsigned quantities for purposes of this division.
  /// @returns a new APInt value containing the division result
  /// @brief Unsigned division operation.
  APInt udiv(const APInt &RHS) const;

  /// Signed divide this APInt by APInt RHS.
  /// @brief Signed division function for APInt.
  APInt sdiv(const APInt &RHS) const {
    if (isNegative())
      if (RHS.isNegative())
        return (-(*this)).udiv(-RHS);
      else
        return -((-(*this)).udiv(RHS));
    else if (RHS.isNegative())
      return -(this->udiv(-RHS));
    return this->udiv(RHS);
  }

  /// Perform an unsigned remainder operation on this APInt with RHS being the
  /// divisor. Both this and RHS are treated as unsigned quantities for purposes
  /// of this operation. Note that this is a true remainder operation and not
  /// a modulo operation because the sign follows the sign of the dividend
  /// which is *this.
  /// @returns a new APInt value containing the remainder result
  /// @brief Unsigned remainder operation.
  APInt urem(const APInt &RHS) const;

  /// Signed remainder operation on APInt.
  /// @brief Function for signed remainder operation.
  APInt srem(const APInt &RHS) const {
    if (isNegative())
      if (RHS.isNegative())
        return -((-(*this)).urem(-RHS));
      else
        return -((-(*this)).urem(RHS));
    else if (RHS.isNegative())
      return this->urem(-RHS);
    return this->urem(RHS);
  }

  /// Sometimes it is convenient to divide two APInt values and obtain both the
  /// quotient and remainder. This function does both operations in the same
  /// computation making it a little more efficient. The pair of input arguments
  /// may overlap with the pair of output arguments. It is safe to call
  /// udivrem(X, Y, X, Y), for example.
  /// @brief Dual division/remainder interface.
  static void udivrem(const APInt &LHS, const APInt &RHS,
                      APInt &Quotient, APInt &Remainder);

  static void sdivrem(const APInt &LHS, const APInt &RHS,
                      APInt &Quotient, APInt &Remainder) {
    if (LHS.isNegative()) {
      if (RHS.isNegative())
        APInt::udivrem(-LHS, -RHS, Quotient, Remainder);
      else {
        APInt::udivrem(-LHS, RHS, Quotient, Remainder);
        Quotient = -Quotient;
      }
      Remainder = -Remainder;
    } else if (RHS.isNegative()) {
      APInt::udivrem(LHS, -RHS, Quotient, Remainder);
      Quotient = -Quotient;
    } else {
      APInt::udivrem(LHS, RHS, Quotient, Remainder);
    }
  }
  
  
  // Operations that return overflow indicators.
  APInt sadd_ov(const APInt &RHS, bool &Overflow) const;
  APInt uadd_ov(const APInt &RHS, bool &Overflow) const;
  APInt ssub_ov(const APInt &RHS, bool &Overflow) const;
  APInt usub_ov(const APInt &RHS, bool &Overflow) const;
  APInt sdiv_ov(const APInt &RHS, bool &Overflow) const;
  APInt smul_ov(const APInt &RHS, bool &Overflow) const;
  APInt umul_ov(const APInt &RHS, bool &Overflow) const;
  APInt sshl_ov(unsigned Amt, bool &Overflow) const;

  /// @returns the bit value at bitPosition
  /// @brief Array-indexing support.
  bool operator[](unsigned bitPosition) const {
    assert(bitPosition < getBitWidth() && "Bit position out of bounds!");
    return (maskBit(bitPosition) &
            (isSingleWord() ? VAL : pVal[whichWord(bitPosition)])) != 0;
  }

  /// @}
  /// @name Comparison Operators
  /// @{
  /// Compares this APInt with RHS for the validity of the equality
  /// relationship.
  /// @brief Equality operator.
  bool operator==(const APInt& RHS) const {
    assert(BitWidth == RHS.BitWidth && "Comparison requires equal bit widths");
    if (isSingleWord())
      return VAL == RHS.VAL;
    return EqualSlowCase(RHS);
  }

  /// Compares this APInt with a uint64_t for the validity of the equality
  /// relationship.
  /// @returns true if *this == Val
  /// @brief Equality operator.
  bool operator==(uint64_t Val) const {
    if (isSingleWord())
      return VAL == Val;
    return EqualSlowCase(Val);
  }

  /// Compares this APInt with RHS for the validity of the equality
  /// relationship.
  /// @returns true if *this == Val
  /// @brief Equality comparison.
  bool eq(const APInt &RHS) const {
    return (*this) == RHS;
  }

  /// Compares this APInt with RHS for the validity of the inequality
  /// relationship.
  /// @returns true if *this != Val
  /// @brief Inequality operator.
  bool operator!=(const APInt& RHS) const {
    return !((*this) == RHS);
  }

  /// Compares this APInt with a uint64_t for the validity of the inequality
  /// relationship.
  /// @returns true if *this != Val
  /// @brief Inequality operator.
  bool operator!=(uint64_t Val) const {
    return !((*this) == Val);
  }

  /// Compares this APInt with RHS for the validity of the inequality
  /// relationship.
  /// @returns true if *this != Val
  /// @brief Inequality comparison
  bool ne(const APInt &RHS) const {
    return !((*this) == RHS);
  }

  /// Regards both *this and RHS as unsigned quantities and compares them for
  /// the validity of the less-than relationship.
  /// @returns true if *this < RHS when both are considered unsigned.
  /// @brief Unsigned less than comparison
  bool ult(const APInt &RHS) const;

  /// Regards both *this as an unsigned quantity and compares it with RHS for
  /// the validity of the less-than relationship.
  /// @returns true if *this < RHS when considered unsigned.
  /// @brief Unsigned less than comparison
  bool ult(uint64_t RHS) const {
    return ult(APInt(getBitWidth(), RHS));
  }

  /// Regards both *this and RHS as signed quantities and compares them for
  /// validity of the less-than relationship.
  /// @returns true if *this < RHS when both are considered signed.
  /// @brief Signed less than comparison
  bool slt(const APInt& RHS) const;

  /// Regards both *this as a signed quantity and compares it with RHS for
  /// the validity of the less-than relationship.
  /// @returns true if *this < RHS when considered signed.
  /// @brief Signed less than comparison
  bool slt(uint64_t RHS) const {
    return slt(APInt(getBitWidth(), RHS));
  }

  /// Regards both *this and RHS as unsigned quantities and compares them for
  /// validity of the less-or-equal relationship.
  /// @returns true if *this <= RHS when both are considered unsigned.
  /// @brief Unsigned less or equal comparison
  bool ule(const APInt& RHS) const {
    return ult(RHS) || eq(RHS);
  }

  /// Regards both *this as an unsigned quantity and compares it with RHS for
  /// the validity of the less-or-equal relationship.
  /// @returns true if *this <= RHS when considered unsigned.
  /// @brief Unsigned less or equal comparison
  bool ule(uint64_t RHS) const {
    return ule(APInt(getBitWidth(), RHS));
  }

  /// Regards both *this and RHS as signed quantities and compares them for
  /// validity of the less-or-equal relationship.
  /// @returns true if *this <= RHS when both are considered signed.
  /// @brief Signed less or equal comparison
  bool sle(const APInt& RHS) const {
    return slt(RHS) || eq(RHS);
  }

  /// Regards both *this as a signed quantity and compares it with RHS for
  /// the validity of the less-or-equal relationship.
  /// @returns true if *this <= RHS when considered signed.
  /// @brief Signed less or equal comparison
  bool sle(uint64_t RHS) const {
    return sle(APInt(getBitWidth(), RHS));
  }

  /// Regards both *this and RHS as unsigned quantities and compares them for
  /// the validity of the greater-than relationship.
  /// @returns true if *this > RHS when both are considered unsigned.
  /// @brief Unsigned greather than comparison
  bool ugt(const APInt& RHS) const {
    return !ult(RHS) && !eq(RHS);
  }

  /// Regards both *this as an unsigned quantity and compares it with RHS for
  /// the validity of the greater-than relationship.
  /// @returns true if *this > RHS when considered unsigned.
  /// @brief Unsigned greater than comparison
  bool ugt(uint64_t RHS) const {
    return ugt(APInt(getBitWidth(), RHS));
  }

  /// Regards both *this and RHS as signed quantities and compares them for
  /// the validity of the greater-than relationship.
  /// @returns true if *this > RHS when both are considered signed.
  /// @brief Signed greather than comparison
  bool sgt(const APInt& RHS) const {
    return !slt(RHS) && !eq(RHS);
  }

  /// Regards both *this as a signed quantity and compares it with RHS for
  /// the validity of the greater-than relationship.
  /// @returns true if *this > RHS when considered signed.
  /// @brief Signed greater than comparison
  bool sgt(uint64_t RHS) const {
    return sgt(APInt(getBitWidth(), RHS));
  }

  /// Regards both *this and RHS as unsigned quantities and compares them for
  /// validity of the greater-or-equal relationship.
  /// @returns true if *this >= RHS when both are considered unsigned.
  /// @brief Unsigned greater or equal comparison
  bool uge(const APInt& RHS) const {
    return !ult(RHS);
  }

  /// Regards both *this as an unsigned quantity and compares it with RHS for
  /// the validity of the greater-or-equal relationship.
  /// @returns true if *this >= RHS when considered unsigned.
  /// @brief Unsigned greater or equal comparison
  bool uge(uint64_t RHS) const {
    return uge(APInt(getBitWidth(), RHS));
  }

  /// Regards both *this and RHS as signed quantities and compares them for
  /// validity of the greater-or-equal relationship.
  /// @returns true if *this >= RHS when both are considered signed.
  /// @brief Signed greather or equal comparison
  bool sge(const APInt& RHS) const {
    return !slt(RHS);
  }

  /// Regards both *this as a signed quantity and compares it with RHS for
  /// the validity of the greater-or-equal relationship.
  /// @returns true if *this >= RHS when considered signed.
  /// @brief Signed greater or equal comparison
  bool sge(uint64_t RHS) const {
    return sge(APInt(getBitWidth(), RHS));
  }

  
  
  
  /// This operation tests if there are any pairs of corresponding bits
  /// between this APInt and RHS that are both set.
  bool intersects(const APInt &RHS) const {
    return (*this & RHS) != 0;
  }

  /// @}
  /// @name Resizing Operators
  /// @{
  /// Truncate the APInt to a specified width. It is an error to specify a width
  /// that is greater than or equal to the current width.
  /// @brief Truncate to new width.
  APInt trunc(unsigned width) const;

  /// This operation sign extends the APInt to a new width. If the high order
  /// bit is set, the fill on the left will be done with 1 bits, otherwise zero.
  /// It is an error to specify a width that is less than or equal to the
  /// current width.
  /// @brief Sign extend to a new width.
  APInt sext(unsigned width) const;

  /// This operation zero extends the APInt to a new width. The high order bits
  /// are filled with 0 bits.  It is an error to specify a width that is less
  /// than or equal to the current width.
  /// @brief Zero extend to a new width.
  APInt zext(unsigned width) const;

  /// Make this APInt have the bit width given by \p width. The value is sign
  /// extended, truncated, or left alone to make it that width.
  /// @brief Sign extend or truncate to width
  APInt sextOrTrunc(unsigned width) const;

  /// Make this APInt have the bit width given by \p width. The value is zero
  /// extended, truncated, or left alone to make it that width.
  /// @brief Zero extend or truncate to width
  APInt zextOrTrunc(unsigned width) const;

  /// Make this APInt have the bit width given by \p width. The value is sign
  /// extended, or left alone to make it that width.
  /// @brief Sign extend or truncate to width
  APInt sextOrSelf(unsigned width) const;

  /// Make this APInt have the bit width given by \p width. The value is zero
  /// extended, or left alone to make it that width.
  /// @brief Zero extend or truncate to width
  APInt zextOrSelf(unsigned width) const;

  /// @}
  /// @name Bit Manipulation Operators
  /// @{
  /// @brief Set every bit to 1.
  void setAllBits() {
    if (isSingleWord())
      VAL = -1ULL;
    else {
      // Set all the bits in all the words.
      for (unsigned i = 0; i < getNumWords(); ++i)
        pVal[i] = -1ULL;
    }
    // Clear the unused ones
    clearUnusedBits();
  }

  /// Set the given bit to 1 whose position is given as "bitPosition".
  /// @brief Set a given bit to 1.
  void setBit(unsigned bitPosition);

  /// @brief Set every bit to 0.
  void clearAllBits() {
    if (isSingleWord())
      VAL = 0;
    else
      memset(pVal, 0, getNumWords() * APINT_WORD_SIZE);
  }

  /// Set the given bit to 0 whose position is given as "bitPosition".
  /// @brief Set a given bit to 0.
  void clearBit(unsigned bitPosition);

  /// @brief Toggle every bit to its opposite value.
  void flipAllBits() {
    if (isSingleWord())
      VAL ^= -1ULL;
    else {
      for (unsigned i = 0; i < getNumWords(); ++i)
        pVal[i] ^= -1ULL;
    }
    clearUnusedBits();
  }

  /// Toggle a given bit to its opposite value whose position is given
  /// as "bitPosition".
  /// @brief Toggles a given bit to its opposite value.
  void flipBit(unsigned bitPosition);

  /// @}
  /// @name Value Characterization Functions
  /// @{

  /// @returns the total number of bits.
  unsigned getBitWidth() const {
    return BitWidth;
  }

  /// Here one word's bitwidth equals to that of uint64_t.
  /// @returns the number of words to hold the integer value of this APInt.
  /// @brief Get the number of words.
  unsigned getNumWords() const {
    return getNumWords(BitWidth);
  }

  /// Here one word's bitwidth equals to that of uint64_t.
  /// @returns the number of words to hold the integer value with a
  /// given bit width.
  /// @brief Get the number of words.
  static unsigned getNumWords(unsigned BitWidth) {
    return (BitWidth + APINT_BITS_PER_WORD - 1) / APINT_BITS_PER_WORD;
  }

  /// This function returns the number of active bits which is defined as the
  /// bit width minus the number of leading zeros. This is used in several
  /// computations to see how "wide" the value is.
  /// @brief Compute the number of active bits in the value
  unsigned getActiveBits() const {
    return BitWidth - countLeadingZeros();
  }

  /// This function returns the number of active words in the value of this
  /// APInt. This is used in conjunction with getActiveData to extract the raw
  /// value of the APInt.
  unsigned getActiveWords() const {
    return whichWord(getActiveBits()-1) + 1;
  }

  /// Computes the minimum bit width for this APInt while considering it to be
  /// a signed (and probably negative) value. If the value is not negative,
  /// this function returns the same value as getActiveBits()+1. Otherwise, it
  /// returns the smallest bit width that will retain the negative value. For
  /// example, -1 can be written as 0b1 or 0xFFFFFFFFFF. 0b1 is shorter and so
  /// for -1, this function will always return 1.
  /// @brief Get the minimum bit size for this signed APInt
  unsigned getMinSignedBits() const {
    if (isNegative())
      return BitWidth - countLeadingOnes() + 1;
    return getActiveBits()+1;
  }

  /// This method attempts to return the value of this APInt as a zero extended
  /// uint64_t. The bitwidth must be <= 64 or the value must fit within a
  /// uint64_t. Otherwise an assertion will result.
  /// @brief Get zero extended value
  uint64_t getZExtValue() const {
    if (isSingleWord())
      return VAL;
    assert(getActiveBits() <= 64 && "Too many bits for uint64_t");
    return pVal[0];
  }

  /// This method attempts to return the value of this APInt as a sign extended
  /// int64_t. The bit width must be <= 64 or the value must fit within an
  /// int64_t. Otherwise an assertion will result.
  /// @brief Get sign extended value
  int64_t getSExtValue() const {
    if (isSingleWord())
      return int64_t(VAL << (APINT_BITS_PER_WORD - BitWidth)) >>
                     (APINT_BITS_PER_WORD - BitWidth);
    assert(getMinSignedBits() <= 64 && "Too many bits for int64_t");
    return int64_t(pVal[0]);
  }

  /// This method determines how many bits are required to hold the APInt
  /// equivalent of the string given by \p str.
  /// @brief Get bits required for string value.
  static unsigned getBitsNeeded(StringRef str, uint8_t radix);

  /// countLeadingZeros - This function is an APInt version of the
  /// countLeadingZeros_{32,64} functions in MathExtras.h. It counts the number
  /// of zeros from the most significant bit to the first one bit.
  /// @returns BitWidth if the value is zero, otherwise
  /// returns the number of zeros from the most significant bit to the first
  /// one bits.
  unsigned countLeadingZeros() const {
    if (isSingleWord()) {
      unsigned unusedBits = APINT_BITS_PER_WORD - BitWidth;
      return CountLeadingZeros_64(VAL) - unusedBits;
    }
    return countLeadingZerosSlowCase();
  }

  /// countLeadingOnes - This function is an APInt version of the
  /// countLeadingOnes_{32,64} functions in MathExtras.h. It counts the number
  /// of ones from the most significant bit to the first zero bit.
  /// @returns 0 if the high order bit is not set, otherwise
  /// returns the number of 1 bits from the most significant to the least
  /// @brief Count the number of leading one bits.
  unsigned countLeadingOnes() const;

  /// Computes the number of leading bits of this APInt that are equal to its
  /// sign bit.
  unsigned getNumSignBits() const {
    return isNegative() ? countLeadingOnes() : countLeadingZeros();
  }

  /// countTrailingZeros - This function is an APInt version of the
  /// countTrailingZeros_{32,64} functions in MathExtras.h. It counts
  /// the number of zeros from the least significant bit to the first set bit.
  /// @returns BitWidth if the value is zero, otherwise
  /// returns the number of zeros from the least significant bit to the first
  /// one bit.
  /// @brief Count the number of trailing zero bits.
  unsigned countTrailingZeros() const;

  /// countTrailingOnes - This function is an APInt version of the
  /// countTrailingOnes_{32,64} functions in MathExtras.h. It counts
  /// the number of ones from the least significant bit to the first zero bit.
  /// @returns BitWidth if the value is all ones, otherwise
  /// returns the number of ones from the least significant bit to the first
  /// zero bit.
  /// @brief Count the number of trailing one bits.
  unsigned countTrailingOnes() const {
    if (isSingleWord())
      return CountTrailingOnes_64(VAL);
    return countTrailingOnesSlowCase();
  }

  /// countPopulation - This function is an APInt version of the
  /// countPopulation_{32,64} functions in MathExtras.h. It counts the number
  /// of 1 bits in the APInt value.
  /// @returns 0 if the value is zero, otherwise returns the number of set
  /// bits.
  /// @brief Count the number of bits set.
  unsigned countPopulation() const {
    if (isSingleWord())
      return CountPopulation_64(VAL);
    return countPopulationSlowCase();
  }

  /// @}
  /// @name Conversion Functions
  /// @{
  void print(raw_ostream &OS, bool isSigned) const;

  /// toString - Converts an APInt to a string and append it to Str.  Str is
  /// commonly a SmallString.
  void toString(SmallVectorImpl<char> &Str, unsigned Radix, bool Signed,
                bool formatAsCLiteral = false) const;

  /// Considers the APInt to be unsigned and converts it into a string in the
  /// radix given. The radix can be 2, 8, 10 16, or 36.
  void toStringUnsigned(SmallVectorImpl<char> &Str, unsigned Radix = 10) const {
    toString(Str, Radix, false, false);
  }

  /// Considers the APInt to be signed and converts it into a string in the
  /// radix given. The radix can be 2, 8, 10, 16, or 36.
  void toStringSigned(SmallVectorImpl<char> &Str, unsigned Radix = 10) const {
    toString(Str, Radix, true, false);
  }

  /// toString - This returns the APInt as a std::string.  Note that this is an
  /// inefficient method.  It is better to pass in a SmallVector/SmallString
  /// to the methods above to avoid thrashing the heap for the string.
  std::string toString(unsigned Radix, bool Signed) const;


  /// @returns a byte-swapped representation of this APInt Value.
  APInt byteSwap() const;

  /// @brief Converts this APInt to a double value.
  double roundToDouble(bool isSigned) const;

  /// @brief Converts this unsigned APInt to a double value.
  double roundToDouble() const {
    return roundToDouble(false);
  }

  /// @brief Converts this signed APInt to a double value.
  double signedRoundToDouble() const {
    return roundToDouble(true);
  }

  /// The conversion does not do a translation from integer to double, it just
  /// re-interprets the bits as a double. Note that it is valid to do this on
  /// any bit width. Exactly 64 bits will be translated.
  /// @brief Converts APInt bits to a double
  double bitsToDouble() const {
    union {
      uint64_t I;
      double D;
    } T;
    T.I = (isSingleWord() ? VAL : pVal[0]);
    return T.D;
  }

  /// The conversion does not do a translation from integer to float, it just
  /// re-interprets the bits as a float. Note that it is valid to do this on
  /// any bit width. Exactly 32 bits will be translated.
  /// @brief Converts APInt bits to a double
  float bitsToFloat() const {
    union {
      unsigned I;
      float F;
    } T;
    T.I = unsigned((isSingleWord() ? VAL : pVal[0]));
    return T.F;
  }

  /// The conversion does not do a translation from double to integer, it just
  /// re-interprets the bits of the double.
  /// @brief Converts a double to APInt bits.
  static APInt doubleToBits(double V) {
    union {
      uint64_t I;
      double D;
    } T;
    T.D = V;
    return APInt(sizeof T * CHAR_BIT, T.I);
  }

  /// The conversion does not do a translation from float to integer, it just
  /// re-interprets the bits of the float.
  /// @brief Converts a float to APInt bits.
  static APInt floatToBits(float V) {
    union {
      unsigned I;
      float F;
    } T;
    T.F = V;
    return APInt(sizeof T * CHAR_BIT, T.I);
  }

  /// @}
  /// @name Mathematics Operations
  /// @{

  /// @returns the floor log base 2 of this APInt.
  unsigned logBase2() const {
    return BitWidth - 1 - countLeadingZeros();
  }

  /// @returns the ceil log base 2 of this APInt.
  unsigned ceilLogBase2() const {
    return BitWidth - (*this - 1).countLeadingZeros();
  }

  /// @returns the log base 2 of this APInt if its an exact power of two, -1
  /// otherwise
  int32_t exactLogBase2() const {
    if (!isPowerOf2())
      return -1;
    return logBase2();
  }

  /// @brief Compute the square root
  APInt sqrt() const;

  /// If *this is < 0 then return -(*this), otherwise *this;
  /// @brief Get the absolute value;
  APInt abs() const {
    if (isNegative())
      return -(*this);
    return *this;
  }

  /// @returns the multiplicative inverse for a given modulo.
  APInt multiplicativeInverse(const APInt& modulo) const;

  /// @}
  /// @name Support for division by constant
  /// @{

  /// Calculate the magic number for signed division by a constant.
  struct ms;
  ms magic() const;

  /// Calculate the magic number for unsigned division by a constant.
  struct mu;
  mu magicu(unsigned LeadingZeros = 0) const;

  /// @}
  /// @name Building-block Operations for APInt and APFloat
  /// @{

  // These building block operations operate on a representation of
  // arbitrary precision, two's-complement, bignum integer values.
  // They should be sufficient to implement APInt and APFloat bignum
  // requirements.  Inputs are generally a pointer to the base of an
  // array of integer parts, representing an unsigned bignum, and a
  // count of how many parts there are.

  /// Sets the least significant part of a bignum to the input value,
  /// and zeroes out higher parts.  */
  static void tcSet(integerPart *, integerPart, unsigned int);

  /// Assign one bignum to another.
  static void tcAssign(integerPart *, const integerPart *, unsigned int);

  /// Returns true if a bignum is zero, false otherwise.
  static bool tcIsZero(const integerPart *, unsigned int);

  /// Extract the given bit of a bignum; returns 0 or 1.  Zero-based.
  static int tcExtractBit(const integerPart *, unsigned int bit);

  /// Copy the bit vector of width srcBITS from SRC, starting at bit
  /// srcLSB, to DST, of dstCOUNT parts, such that the bit srcLSB
  /// becomes the least significant bit of DST.  All high bits above
  /// srcBITS in DST are zero-filled.
  static void tcExtract(integerPart *, unsigned int dstCount,
                        const integerPart *,
                        unsigned int srcBits, unsigned int srcLSB);

  /// Set the given bit of a bignum.  Zero-based.
  static void tcSetBit(integerPart *, unsigned int bit);

  /// Clear the given bit of a bignum.  Zero-based.
  static void tcClearBit(integerPart *, unsigned int bit);

  /// Returns the bit number of the least or most significant set bit
  /// of a number.  If the input number has no bits set -1U is
  /// returned.
  static unsigned int tcLSB(const integerPart *, unsigned int);
  static unsigned int tcMSB(const integerPart *parts, unsigned int n);

  /// Negate a bignum in-place.
  static void tcNegate(integerPart *, unsigned int);

  /// DST += RHS + CARRY where CARRY is zero or one.  Returns the
  /// carry flag.
  static integerPart tcAdd(integerPart *, const integerPart *,
                           integerPart carry, unsigned);

  /// DST -= RHS + CARRY where CARRY is zero or one.  Returns the
  /// carry flag.
  static integerPart tcSubtract(integerPart *, const integerPart *,
                                integerPart carry, unsigned);

  ///  DST += SRC * MULTIPLIER + PART   if add is true
  ///  DST  = SRC * MULTIPLIER + PART   if add is false
  ///
  ///  Requires 0 <= DSTPARTS <= SRCPARTS + 1.  If DST overlaps SRC
  ///  they must start at the same point, i.e. DST == SRC.
  ///
  ///  If DSTPARTS == SRC_PARTS + 1 no overflow occurs and zero is
  ///  returned.  Otherwise DST is filled with the least significant
  ///  DSTPARTS parts of the result, and if all of the omitted higher
  ///  parts were zero return zero, otherwise overflow occurred and
  ///  return one.
  static int tcMultiplyPart(integerPart *dst, const integerPart *src,
                            integerPart multiplier, integerPart carry,
                            unsigned int srcParts, unsigned int dstParts,
                            bool add);

  /// DST = LHS * RHS, where DST has the same width as the operands
  /// and is filled with the least significant parts of the result.
  /// Returns one if overflow occurred, otherwise zero.  DST must be
  /// disjoint from both operands.
  static int tcMultiply(integerPart *, const integerPart *,
                        const integerPart *, unsigned);

  /// DST = LHS * RHS, where DST has width the sum of the widths of
  /// the operands.  No overflow occurs.  DST must be disjoint from
  /// both operands. Returns the number of parts required to hold the
  /// result.
  static unsigned int tcFullMultiply(integerPart *, const integerPart *,
                                     const integerPart *, unsigned, unsigned);

  /// If RHS is zero LHS and REMAINDER are left unchanged, return one.
  /// Otherwise set LHS to LHS / RHS with the fractional part
  /// discarded, set REMAINDER to the remainder, return zero.  i.e.
  ///
  ///  OLD_LHS = RHS * LHS + REMAINDER
  ///
  ///  SCRATCH is a bignum of the same size as the operands and result
  ///  for use by the routine; its contents need not be initialized
  ///  and are destroyed.  LHS, REMAINDER and SCRATCH must be
  ///  distinct.
  static int tcDivide(integerPart *lhs, const integerPart *rhs,
                      integerPart *remainder, integerPart *scratch,
                      unsigned int parts);

  /// Shift a bignum left COUNT bits.  Shifted in bits are zero.
  /// There are no restrictions on COUNT.
  static void tcShiftLeft(integerPart *, unsigned int parts,
                          unsigned int count);

  /// Shift a bignum right COUNT bits.  Shifted in bits are zero.
  /// There are no restrictions on COUNT.
  static void tcShiftRight(integerPart *, unsigned int parts,
                           unsigned int count);

  /// The obvious AND, OR and XOR and complement operations.
  static void tcAnd(integerPart *, const integerPart *, unsigned int);
  static void tcOr(integerPart *, const integerPart *, unsigned int);
  static void tcXor(integerPart *, const integerPart *, unsigned int);
  static void tcComplement(integerPart *, unsigned int);

  /// Comparison (unsigned) of two bignums.
  static int tcCompare(const integerPart *, const integerPart *,
                       unsigned int);

  /// Increment a bignum in-place.  Return the carry flag.
  static integerPart tcIncrement(integerPart *, unsigned int);

  /// Set the least significant BITS and clear the rest.
  static void tcSetLeastSignificantBits(integerPart *, unsigned int,
                                        unsigned int bits);

  /// @brief debug method
  void dump() const;

  /// @}
};

/// Magic data for optimising signed division by a constant.
struct APInt::ms {
  APInt m;  ///< magic number
  unsigned s;  ///< shift amount
};

/// Magic data for optimising unsigned division by a constant.
struct APInt::mu {
  APInt m;     ///< magic number
  bool a;      ///< add indicator
  unsigned s;  ///< shift amount
};

inline bool operator==(uint64_t V1, const APInt& V2) {
  return V2 == V1;
}

inline bool operator!=(uint64_t V1, const APInt& V2) {
  return V2 != V1;
}

inline raw_ostream &operator<<(raw_ostream &OS, const APInt &I) {
  I.print(OS, true);
  return OS;
}

namespace APIntOps {

/// @brief Determine the smaller of two APInts considered to be signed.
inline APInt smin(const APInt &A, const APInt &B) {
  return A.slt(B) ? A : B;
}

/// @brief Determine the larger of two APInts considered to be signed.
inline APInt smax(const APInt &A, const APInt &B) {
  return A.sgt(B) ? A : B;
}

/// @brief Determine the smaller of two APInts considered to be signed.
inline APInt umin(const APInt &A, const APInt &B) {
  return A.ult(B) ? A : B;
}

/// @brief Determine the larger of two APInts considered to be unsigned.
inline APInt umax(const APInt &A, const APInt &B) {
  return A.ugt(B) ? A : B;
}

/// @brief Check if the specified APInt has a N-bits unsigned integer value.
inline bool isIntN(unsigned N, const APInt& APIVal) {
  return APIVal.isIntN(N);
}

/// @brief Check if the specified APInt has a N-bits signed integer value.
inline bool isSignedIntN(unsigned N, const APInt& APIVal) {
  return APIVal.isSignedIntN(N);
}

/// @returns true if the argument APInt value is a sequence of ones
/// starting at the least significant bit with the remainder zero.
inline bool isMask(unsigned numBits, const APInt& APIVal) {
  return numBits <= APIVal.getBitWidth() &&
    APIVal == APInt::getLowBitsSet(APIVal.getBitWidth(), numBits);
}

/// @returns true if the argument APInt value contains a sequence of ones
/// with the remainder zero.
inline bool isShiftedMask(unsigned numBits, const APInt& APIVal) {
  return isMask(numBits, (APIVal - APInt(numBits,1)) | APIVal);
}

/// @returns a byte-swapped representation of the specified APInt Value.
inline APInt byteSwap(const APInt& APIVal) {
  return APIVal.byteSwap();
}

/// @returns the floor log base 2 of the specified APInt value.
inline unsigned logBase2(const APInt& APIVal) {
  return APIVal.logBase2();
}

/// GreatestCommonDivisor - This function returns the greatest common
/// divisor of the two APInt values using Euclid's algorithm.
/// @returns the greatest common divisor of Val1 and Val2
/// @brief Compute GCD of two APInt values.
APInt GreatestCommonDivisor(const APInt& Val1, const APInt& Val2);

/// Treats the APInt as an unsigned value for conversion purposes.
/// @brief Converts the given APInt to a double value.
inline double RoundAPIntToDouble(const APInt& APIVal) {
  return APIVal.roundToDouble();
}

/// Treats the APInt as a signed value for conversion purposes.
/// @brief Converts the given APInt to a double value.
inline double RoundSignedAPIntToDouble(const APInt& APIVal) {
  return APIVal.signedRoundToDouble();
}

/// @brief Converts the given APInt to a float vlalue.
inline float RoundAPIntToFloat(const APInt& APIVal) {
  return float(RoundAPIntToDouble(APIVal));
}

/// Treast the APInt as a signed value for conversion purposes.
/// @brief Converts the given APInt to a float value.
inline float RoundSignedAPIntToFloat(const APInt& APIVal) {
  return float(APIVal.signedRoundToDouble());
}

/// RoundDoubleToAPInt - This function convert a double value to an APInt value.
/// @brief Converts the given double value into a APInt.
APInt RoundDoubleToAPInt(double Double, unsigned width);

/// RoundFloatToAPInt - Converts a float value into an APInt value.
/// @brief Converts a float value into a APInt.
inline APInt RoundFloatToAPInt(float Float, unsigned width) {
  return RoundDoubleToAPInt(double(Float), width);
}

/// Arithmetic right-shift the APInt by shiftAmt.
/// @brief Arithmetic right-shift function.
inline APInt ashr(const APInt& LHS, unsigned shiftAmt) {
  return LHS.ashr(shiftAmt);
}

/// Logical right-shift the APInt by shiftAmt.
/// @brief Logical right-shift function.
inline APInt lshr(const APInt& LHS, unsigned shiftAmt) {
  return LHS.lshr(shiftAmt);
}

/// Left-shift the APInt by shiftAmt.
/// @brief Left-shift function.
inline APInt shl(const APInt& LHS, unsigned shiftAmt) {
  return LHS.shl(shiftAmt);
}

/// Signed divide APInt LHS by APInt RHS.
/// @brief Signed division function for APInt.
inline APInt sdiv(const APInt& LHS, const APInt& RHS) {
  return LHS.sdiv(RHS);
}

/// Unsigned divide APInt LHS by APInt RHS.
/// @brief Unsigned division function for APInt.
inline APInt udiv(const APInt& LHS, const APInt& RHS) {
  return LHS.udiv(RHS);
}

/// Signed remainder operation on APInt.
/// @brief Function for signed remainder operation.
inline APInt srem(const APInt& LHS, const APInt& RHS) {
  return LHS.srem(RHS);
}

/// Unsigned remainder operation on APInt.
/// @brief Function for unsigned remainder operation.
inline APInt urem(const APInt& LHS, const APInt& RHS) {
  return LHS.urem(RHS);
}

/// Performs multiplication on APInt values.
/// @brief Function for multiplication operation.
inline APInt mul(const APInt& LHS, const APInt& RHS) {
  return LHS * RHS;
}

/// Performs addition on APInt values.
/// @brief Function for addition operation.
inline APInt add(const APInt& LHS, const APInt& RHS) {
  return LHS + RHS;
}

/// Performs subtraction on APInt values.
/// @brief Function for subtraction operation.
inline APInt sub(const APInt& LHS, const APInt& RHS) {
  return LHS - RHS;
}

/// Performs bitwise AND operation on APInt LHS and
/// APInt RHS.
/// @brief Bitwise AND function for APInt.
inline APInt And(const APInt& LHS, const APInt& RHS) {
  return LHS & RHS;
}

/// Performs bitwise OR operation on APInt LHS and APInt RHS.
/// @brief Bitwise OR function for APInt.
inline APInt Or(const APInt& LHS, const APInt& RHS) {
  return LHS | RHS;
}

/// Performs bitwise XOR operation on APInt.
/// @brief Bitwise XOR function for APInt.
inline APInt Xor(const APInt& LHS, const APInt& RHS) {
  return LHS ^ RHS;
}

/// Performs a bitwise complement operation on APInt.
/// @brief Bitwise complement function.
inline APInt Not(const APInt& APIVal) {
  return ~APIVal;
}

} // End of APIntOps namespace

  // See friend declaration above. This additional declaration is required in
  // order to compile LLVM with IBM xlC compiler.
  hash_code hash_value(const APInt &Arg);
} // End of llvm namespace

#endif

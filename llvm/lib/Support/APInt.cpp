//===-- APInt.cpp - Implement APInt class ---------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file was developed by Sheng Zhou and is distributed under the 
// University of Illinois Open Source License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file implements a class to represent arbitrary precision integer
// constant values and provide a variety of arithmetic operations on them.
//
//===----------------------------------------------------------------------===//

#define DEBUG_TYPE "apint"
#include "llvm/ADT/APInt.h"
#include "llvm/DerivedTypes.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/MathExtras.h"
#include <math.h>
#include <limits>
#include <cstring>
#include <cstdlib>
#ifndef NDEBUG
#include <iomanip>
#endif

using namespace llvm;

/// A utility function for allocating memory, checking for allocation failures,
/// and ensuring the contents are zeroed.
inline static uint64_t* getClearedMemory(uint32_t numWords) {
  uint64_t * result = new uint64_t[numWords];
  assert(result && "APInt memory allocation fails!");
  memset(result, 0, numWords * sizeof(uint64_t));
  return result;
}

/// A utility function for allocating memory and checking for allocation 
/// failure.  The content is not zeroed.
inline static uint64_t* getMemory(uint32_t numWords) {
  uint64_t * result = new uint64_t[numWords];
  assert(result && "APInt memory allocation fails!");
  return result;
}

APInt::APInt(uint32_t numBits, uint64_t val, bool isSigned) 
  : BitWidth(numBits), VAL(0) {
  assert(BitWidth >= IntegerType::MIN_INT_BITS && "bitwidth too small");
  assert(BitWidth <= IntegerType::MAX_INT_BITS && "bitwidth too large");
  if (isSingleWord())
    VAL = val;
  else {
    pVal = getClearedMemory(getNumWords());
    pVal[0] = val;
    if (isSigned && int64_t(val) < 0) 
      for (unsigned i = 1; i < getNumWords(); ++i)
        pVal[i] = -1ULL;
  }
  clearUnusedBits();
}

APInt::APInt(uint32_t numBits, uint32_t numWords, uint64_t bigVal[])
  : BitWidth(numBits), VAL(0)  {
  assert(BitWidth >= IntegerType::MIN_INT_BITS && "bitwidth too small");
  assert(BitWidth <= IntegerType::MAX_INT_BITS && "bitwidth too large");
  assert(bigVal && "Null pointer detected!");
  if (isSingleWord())
    VAL = bigVal[0];
  else {
    // Get memory, cleared to 0
    pVal = getClearedMemory(getNumWords());
    // Calculate the number of words to copy
    uint32_t words = std::min<uint32_t>(numWords, getNumWords());
    // Copy the words from bigVal to pVal
    memcpy(pVal, bigVal, words * APINT_WORD_SIZE);
  }
  // Make sure unused high bits are cleared
  clearUnusedBits();
}

APInt::APInt(uint32_t numbits, const char StrStart[], uint32_t slen, 
             uint8_t radix) 
  : BitWidth(numbits), VAL(0) {
  fromString(numbits, StrStart, slen, radix);
}

APInt::APInt(uint32_t numbits, const std::string& Val, uint8_t radix)
  : BitWidth(numbits), VAL(0) {
  assert(!Val.empty() && "String empty?");
  fromString(numbits, Val.c_str(), Val.size(), radix);
}

APInt::APInt(const APInt& that)
  : BitWidth(that.BitWidth), VAL(0) {
  if (isSingleWord()) 
    VAL = that.VAL;
  else {
    pVal = getMemory(getNumWords());
    memcpy(pVal, that.pVal, getNumWords() * APINT_WORD_SIZE);
  }
}

APInt::~APInt() {
  if (!isSingleWord() && pVal) 
    delete [] pVal;
}

APInt& APInt::operator=(const APInt& RHS) {
  // Don't do anything for X = X
  if (this == &RHS)
    return *this;

  // If the bitwidths are the same, we can avoid mucking with memory
  if (BitWidth == RHS.getBitWidth()) {
    if (isSingleWord()) 
      VAL = RHS.VAL;
    else
      memcpy(pVal, RHS.pVal, getNumWords() * APINT_WORD_SIZE);
    return *this;
  }

  if (isSingleWord())
    if (RHS.isSingleWord())
      VAL = RHS.VAL;
    else {
      VAL = 0;
      pVal = getMemory(RHS.getNumWords());
      memcpy(pVal, RHS.pVal, RHS.getNumWords() * APINT_WORD_SIZE);
    }
  else if (getNumWords() == RHS.getNumWords()) 
    memcpy(pVal, RHS.pVal, RHS.getNumWords() * APINT_WORD_SIZE);
  else if (RHS.isSingleWord()) {
    delete [] pVal;
    VAL = RHS.VAL;
  } else {
    delete [] pVal;
    pVal = getMemory(RHS.getNumWords());
    memcpy(pVal, RHS.pVal, RHS.getNumWords() * APINT_WORD_SIZE);
  }
  BitWidth = RHS.BitWidth;
  return clearUnusedBits();
}

APInt& APInt::operator=(uint64_t RHS) {
  if (isSingleWord()) 
    VAL = RHS;
  else {
    pVal[0] = RHS;
    memset(pVal+1, 0, (getNumWords() - 1) * APINT_WORD_SIZE);
  }
  return clearUnusedBits();
}

/// add_1 - This function adds a single "digit" integer, y, to the multiple 
/// "digit" integer array,  x[]. x[] is modified to reflect the addition and
/// 1 is returned if there is a carry out, otherwise 0 is returned.
/// @returns the carry of the addition.
static bool add_1(uint64_t dest[], uint64_t x[], uint32_t len, uint64_t y) {
  for (uint32_t i = 0; i < len; ++i) {
    dest[i] = y + x[i];
    if (dest[i] < y)
      y = 1; // Carry one to next digit.
    else {
      y = 0; // No need to carry so exit early
      break;
    }
  }
  return y;
}

/// @brief Prefix increment operator. Increments the APInt by one.
APInt& APInt::operator++() {
  if (isSingleWord()) 
    ++VAL;
  else
    add_1(pVal, pVal, getNumWords(), 1);
  return clearUnusedBits();
}

/// sub_1 - This function subtracts a single "digit" (64-bit word), y, from 
/// the multi-digit integer array, x[], propagating the borrowed 1 value until 
/// no further borrowing is neeeded or it runs out of "digits" in x.  The result
/// is 1 if "borrowing" exhausted the digits in x, or 0 if x was not exhausted.
/// In other words, if y > x then this function returns 1, otherwise 0.
/// @returns the borrow out of the subtraction
static bool sub_1(uint64_t x[], uint32_t len, uint64_t y) {
  for (uint32_t i = 0; i < len; ++i) {
    uint64_t X = x[i];
    x[i] -= y;
    if (y > X) 
      y = 1;  // We have to "borrow 1" from next "digit"
    else {
      y = 0;  // No need to borrow
      break;  // Remaining digits are unchanged so exit early
    }
  }
  return bool(y);
}

/// @brief Prefix decrement operator. Decrements the APInt by one.
APInt& APInt::operator--() {
  if (isSingleWord()) 
    --VAL;
  else
    sub_1(pVal, getNumWords(), 1);
  return clearUnusedBits();
}

/// add - This function adds the integer array x to the integer array Y and
/// places the result in dest. 
/// @returns the carry out from the addition
/// @brief General addition of 64-bit integer arrays
static bool add(uint64_t *dest, const uint64_t *x, const uint64_t *y, 
                uint32_t len) {
  bool carry = false;
  for (uint32_t i = 0; i< len; ++i) {
    uint64_t limit = std::min(x[i],y[i]); // must come first in case dest == x
    dest[i] = x[i] + y[i] + carry;
    carry = dest[i] < limit || (carry && dest[i] == limit);
  }
  return carry;
}

/// Adds the RHS APint to this APInt.
/// @returns this, after addition of RHS.
/// @brief Addition assignment operator. 
APInt& APInt::operator+=(const APInt& RHS) {
  assert(BitWidth == RHS.BitWidth && "Bit widths must be the same");
  if (isSingleWord()) 
    VAL += RHS.VAL;
  else {
    add(pVal, pVal, RHS.pVal, getNumWords());
  }
  return clearUnusedBits();
}

/// Subtracts the integer array y from the integer array x 
/// @returns returns the borrow out.
/// @brief Generalized subtraction of 64-bit integer arrays.
static bool sub(uint64_t *dest, const uint64_t *x, const uint64_t *y, 
                uint32_t len) {
  bool borrow = false;
  for (uint32_t i = 0; i < len; ++i) {
    uint64_t x_tmp = borrow ? x[i] - 1 : x[i];
    borrow = y[i] > x_tmp || (borrow && x[i] == 0);
    dest[i] = x_tmp - y[i];
  }
  return borrow;
}

/// Subtracts the RHS APInt from this APInt
/// @returns this, after subtraction
/// @brief Subtraction assignment operator. 
APInt& APInt::operator-=(const APInt& RHS) {
  assert(BitWidth == RHS.BitWidth && "Bit widths must be the same");
  if (isSingleWord()) 
    VAL -= RHS.VAL;
  else
    sub(pVal, pVal, RHS.pVal, getNumWords());
  return clearUnusedBits();
}

/// Multiplies an integer array, x by a a uint64_t integer and places the result
/// into dest. 
/// @returns the carry out of the multiplication.
/// @brief Multiply a multi-digit APInt by a single digit (64-bit) integer.
static uint64_t mul_1(uint64_t dest[], uint64_t x[], uint32_t len, uint64_t y) {
  // Split y into high 32-bit part (hy)  and low 32-bit part (ly)
  uint64_t ly = y & 0xffffffffULL, hy = y >> 32;
  uint64_t carry = 0;

  // For each digit of x.
  for (uint32_t i = 0; i < len; ++i) {
    // Split x into high and low words
    uint64_t lx = x[i] & 0xffffffffULL;
    uint64_t hx = x[i] >> 32;
    // hasCarry - A flag to indicate if there is a carry to the next digit.
    // hasCarry == 0, no carry
    // hasCarry == 1, has carry
    // hasCarry == 2, no carry and the calculation result == 0.
    uint8_t hasCarry = 0;
    dest[i] = carry + lx * ly;
    // Determine if the add above introduces carry.
    hasCarry = (dest[i] < carry) ? 1 : 0;
    carry = hx * ly + (dest[i] >> 32) + (hasCarry ? (1ULL << 32) : 0);
    // The upper limit of carry can be (2^32 - 1)(2^32 - 1) + 
    // (2^32 - 1) + 2^32 = 2^64.
    hasCarry = (!carry && hasCarry) ? 1 : (!carry ? 2 : 0);

    carry += (lx * hy) & 0xffffffffULL;
    dest[i] = (carry << 32) | (dest[i] & 0xffffffffULL);
    carry = (((!carry && hasCarry != 2) || hasCarry == 1) ? (1ULL << 32) : 0) + 
            (carry >> 32) + ((lx * hy) >> 32) + hx * hy;
  }
  return carry;
}

/// Multiplies integer array x by integer array y and stores the result into 
/// the integer array dest. Note that dest's size must be >= xlen + ylen.
/// @brief Generalized multiplicate of integer arrays.
static void mul(uint64_t dest[], uint64_t x[], uint32_t xlen, uint64_t y[], 
                uint32_t ylen) {
  dest[xlen] = mul_1(dest, x, xlen, y[0]);
  for (uint32_t i = 1; i < ylen; ++i) {
    uint64_t ly = y[i] & 0xffffffffULL, hy = y[i] >> 32;
    uint64_t carry = 0, lx = 0, hx = 0;
    for (uint32_t j = 0; j < xlen; ++j) {
      lx = x[j] & 0xffffffffULL;
      hx = x[j] >> 32;
      // hasCarry - A flag to indicate if has carry.
      // hasCarry == 0, no carry
      // hasCarry == 1, has carry
      // hasCarry == 2, no carry and the calculation result == 0.
      uint8_t hasCarry = 0;
      uint64_t resul = carry + lx * ly;
      hasCarry = (resul < carry) ? 1 : 0;
      carry = (hasCarry ? (1ULL << 32) : 0) + hx * ly + (resul >> 32);
      hasCarry = (!carry && hasCarry) ? 1 : (!carry ? 2 : 0);

      carry += (lx * hy) & 0xffffffffULL;
      resul = (carry << 32) | (resul & 0xffffffffULL);
      dest[i+j] += resul;
      carry = (((!carry && hasCarry != 2) || hasCarry == 1) ? (1ULL << 32) : 0)+
              (carry >> 32) + (dest[i+j] < resul ? 1 : 0) + 
              ((lx * hy) >> 32) + hx * hy;
    }
    dest[i+xlen] = carry;
  }
}

APInt& APInt::operator*=(const APInt& RHS) {
  assert(BitWidth == RHS.BitWidth && "Bit widths must be the same");
  if (isSingleWord()) {
    VAL *= RHS.VAL;
    clearUnusedBits();
    return *this;
  }

  // Get some bit facts about LHS and check for zero
  uint32_t lhsBits = getActiveBits();
  uint32_t lhsWords = !lhsBits ? 0 : whichWord(lhsBits - 1) + 1;
  if (!lhsWords) 
    // 0 * X ===> 0
    return *this;

  // Get some bit facts about RHS and check for zero
  uint32_t rhsBits = RHS.getActiveBits();
  uint32_t rhsWords = !rhsBits ? 0 : whichWord(rhsBits - 1) + 1;
  if (!rhsWords) {
    // X * 0 ===> 0
    clear();
    return *this;
  }

  // Allocate space for the result
  uint32_t destWords = rhsWords + lhsWords;
  uint64_t *dest = getMemory(destWords);

  // Perform the long multiply
  mul(dest, pVal, lhsWords, RHS.pVal, rhsWords);

  // Copy result back into *this
  clear();
  uint32_t wordsToCopy = destWords >= getNumWords() ? getNumWords() : destWords;
  memcpy(pVal, dest, wordsToCopy * APINT_WORD_SIZE);

  // delete dest array and return
  delete[] dest;
  return *this;
}

APInt& APInt::operator&=(const APInt& RHS) {
  assert(BitWidth == RHS.BitWidth && "Bit widths must be the same");
  if (isSingleWord()) {
    VAL &= RHS.VAL;
    return *this;
  }
  uint32_t numWords = getNumWords();
  for (uint32_t i = 0; i < numWords; ++i)
    pVal[i] &= RHS.pVal[i];
  return *this;
}

APInt& APInt::operator|=(const APInt& RHS) {
  assert(BitWidth == RHS.BitWidth && "Bit widths must be the same");
  if (isSingleWord()) {
    VAL |= RHS.VAL;
    return *this;
  }
  uint32_t numWords = getNumWords();
  for (uint32_t i = 0; i < numWords; ++i)
    pVal[i] |= RHS.pVal[i];
  return *this;
}

APInt& APInt::operator^=(const APInt& RHS) {
  assert(BitWidth == RHS.BitWidth && "Bit widths must be the same");
  if (isSingleWord()) {
    VAL ^= RHS.VAL;
    this->clearUnusedBits();
    return *this;
  } 
  uint32_t numWords = getNumWords();
  for (uint32_t i = 0; i < numWords; ++i)
    pVal[i] ^= RHS.pVal[i];
  return clearUnusedBits();
}

APInt APInt::operator&(const APInt& RHS) const {
  assert(BitWidth == RHS.BitWidth && "Bit widths must be the same");
  if (isSingleWord())
    return APInt(getBitWidth(), VAL & RHS.VAL);

  uint32_t numWords = getNumWords();
  uint64_t* val = getMemory(numWords);
  for (uint32_t i = 0; i < numWords; ++i)
    val[i] = pVal[i] & RHS.pVal[i];
  return APInt(val, getBitWidth());
}

APInt APInt::operator|(const APInt& RHS) const {
  assert(BitWidth == RHS.BitWidth && "Bit widths must be the same");
  if (isSingleWord())
    return APInt(getBitWidth(), VAL | RHS.VAL);

  uint32_t numWords = getNumWords();
  uint64_t *val = getMemory(numWords);
  for (uint32_t i = 0; i < numWords; ++i)
    val[i] = pVal[i] | RHS.pVal[i];
  return APInt(val, getBitWidth());
}

APInt APInt::operator^(const APInt& RHS) const {
  assert(BitWidth == RHS.BitWidth && "Bit widths must be the same");
  if (isSingleWord())
    return APInt(BitWidth, VAL ^ RHS.VAL);

  uint32_t numWords = getNumWords();
  uint64_t *val = getMemory(numWords);
  for (uint32_t i = 0; i < numWords; ++i)
    val[i] = pVal[i] ^ RHS.pVal[i];

  // 0^0==1 so clear the high bits in case they got set.
  return APInt(val, getBitWidth()).clearUnusedBits();
}

bool APInt::operator !() const {
  if (isSingleWord())
    return !VAL;

  for (uint32_t i = 0; i < getNumWords(); ++i)
    if (pVal[i]) 
      return false;
  return true;
}

APInt APInt::operator*(const APInt& RHS) const {
  assert(BitWidth == RHS.BitWidth && "Bit widths must be the same");
  if (isSingleWord())
    return APInt(BitWidth, VAL * RHS.VAL);
  APInt Result(*this);
  Result *= RHS;
  return Result.clearUnusedBits();
}

APInt APInt::operator+(const APInt& RHS) const {
  assert(BitWidth == RHS.BitWidth && "Bit widths must be the same");
  if (isSingleWord())
    return APInt(BitWidth, VAL + RHS.VAL);
  APInt Result(BitWidth, 0);
  add(Result.pVal, this->pVal, RHS.pVal, getNumWords());
  return Result.clearUnusedBits();
}

APInt APInt::operator-(const APInt& RHS) const {
  assert(BitWidth == RHS.BitWidth && "Bit widths must be the same");
  if (isSingleWord())
    return APInt(BitWidth, VAL - RHS.VAL);
  APInt Result(BitWidth, 0);
  sub(Result.pVal, this->pVal, RHS.pVal, getNumWords());
  return Result.clearUnusedBits();
}

bool APInt::operator[](uint32_t bitPosition) const {
  return (maskBit(bitPosition) & 
          (isSingleWord() ?  VAL : pVal[whichWord(bitPosition)])) != 0;
}

bool APInt::operator==(const APInt& RHS) const {
  assert(BitWidth == RHS.BitWidth && "Comparison requires equal bit widths");
  if (isSingleWord())
    return VAL == RHS.VAL;

  // Get some facts about the number of bits used in the two operands.
  uint32_t n1 = getActiveBits();
  uint32_t n2 = RHS.getActiveBits();

  // If the number of bits isn't the same, they aren't equal
  if (n1 != n2) 
    return false;

  // If the number of bits fits in a word, we only need to compare the low word.
  if (n1 <= APINT_BITS_PER_WORD)
    return pVal[0] == RHS.pVal[0];

  // Otherwise, compare everything
  for (int i = whichWord(n1 - 1); i >= 0; --i)
    if (pVal[i] != RHS.pVal[i]) 
      return false;
  return true;
}

bool APInt::operator==(uint64_t Val) const {
  if (isSingleWord())
    return VAL == Val;

  uint32_t n = getActiveBits(); 
  if (n <= APINT_BITS_PER_WORD)
    return pVal[0] == Val;
  else
    return false;
}

bool APInt::ult(const APInt& RHS) const {
  assert(BitWidth == RHS.BitWidth && "Bit widths must be same for comparison");
  if (isSingleWord())
    return VAL < RHS.VAL;

  // Get active bit length of both operands
  uint32_t n1 = getActiveBits();
  uint32_t n2 = RHS.getActiveBits();

  // If magnitude of LHS is less than RHS, return true.
  if (n1 < n2)
    return true;

  // If magnitude of RHS is greather than LHS, return false.
  if (n2 < n1)
    return false;

  // If they bot fit in a word, just compare the low order word
  if (n1 <= APINT_BITS_PER_WORD && n2 <= APINT_BITS_PER_WORD)
    return pVal[0] < RHS.pVal[0];

  // Otherwise, compare all words
  uint32_t topWord = whichWord(std::max(n1,n2)-1);
  for (int i = topWord; i >= 0; --i) {
    if (pVal[i] > RHS.pVal[i]) 
      return false;
    if (pVal[i] < RHS.pVal[i]) 
      return true;
  }
  return false;
}

bool APInt::slt(const APInt& RHS) const {
  assert(BitWidth == RHS.BitWidth && "Bit widths must be same for comparison");
  if (isSingleWord()) {
    int64_t lhsSext = (int64_t(VAL) << (64-BitWidth)) >> (64-BitWidth);
    int64_t rhsSext = (int64_t(RHS.VAL) << (64-BitWidth)) >> (64-BitWidth);
    return lhsSext < rhsSext;
  }

  APInt lhs(*this);
  APInt rhs(RHS);
  bool lhsNeg = isNegative();
  bool rhsNeg = rhs.isNegative();
  if (lhsNeg) {
    // Sign bit is set so perform two's complement to make it positive
    lhs.flip();
    lhs++;
  }
  if (rhsNeg) {
    // Sign bit is set so perform two's complement to make it positive
    rhs.flip();
    rhs++;
  }

  // Now we have unsigned values to compare so do the comparison if necessary
  // based on the negativeness of the values.
  if (lhsNeg)
    if (rhsNeg)
      return lhs.ugt(rhs);
    else
      return true;
  else if (rhsNeg)
    return false;
  else 
    return lhs.ult(rhs);
}

APInt& APInt::set(uint32_t bitPosition) {
  if (isSingleWord()) 
    VAL |= maskBit(bitPosition);
  else 
    pVal[whichWord(bitPosition)] |= maskBit(bitPosition);
  return *this;
}

APInt& APInt::set() {
  if (isSingleWord()) {
    VAL = -1ULL;
    return clearUnusedBits();
  }

  // Set all the bits in all the words.
  for (uint32_t i = 0; i < getNumWords(); ++i)
    pVal[i] = -1ULL;
  // Clear the unused ones
  return clearUnusedBits();
}

/// Set the given bit to 0 whose position is given as "bitPosition".
/// @brief Set a given bit to 0.
APInt& APInt::clear(uint32_t bitPosition) {
  if (isSingleWord()) 
    VAL &= ~maskBit(bitPosition);
  else 
    pVal[whichWord(bitPosition)] &= ~maskBit(bitPosition);
  return *this;
}

/// @brief Set every bit to 0.
APInt& APInt::clear() {
  if (isSingleWord()) 
    VAL = 0;
  else 
    memset(pVal, 0, getNumWords() * APINT_WORD_SIZE);
  return *this;
}

/// @brief Bitwise NOT operator. Performs a bitwise logical NOT operation on
/// this APInt.
APInt APInt::operator~() const {
  APInt Result(*this);
  Result.flip();
  return Result;
}

/// @brief Toggle every bit to its opposite value.
APInt& APInt::flip() {
  if (isSingleWord()) {
    VAL ^= -1ULL;
    return clearUnusedBits();
  }
  for (uint32_t i = 0; i < getNumWords(); ++i)
    pVal[i] ^= -1ULL;
  return clearUnusedBits();
}

/// Toggle a given bit to its opposite value whose position is given 
/// as "bitPosition".
/// @brief Toggles a given bit to its opposite value.
APInt& APInt::flip(uint32_t bitPosition) {
  assert(bitPosition < BitWidth && "Out of the bit-width range!");
  if ((*this)[bitPosition]) clear(bitPosition);
  else set(bitPosition);
  return *this;
}

uint32_t APInt::getBitsNeeded(const char* str, uint32_t slen, uint8_t radix) {
  assert(str != 0 && "Invalid value string");
  assert(slen > 0 && "Invalid string length");

  // Each computation below needs to know if its negative
  uint32_t isNegative = str[0] == '-';
  if (isNegative) {
    slen--;
    str++;
  }
  // For radixes of power-of-two values, the bits required is accurately and
  // easily computed
  if (radix == 2)
    return slen + isNegative;
  if (radix == 8)
    return slen * 3 + isNegative;
  if (radix == 16)
    return slen * 4 + isNegative;

  // Otherwise it must be radix == 10, the hard case
  assert(radix == 10 && "Invalid radix");

  // This is grossly inefficient but accurate. We could probably do something
  // with a computation of roughly slen*64/20 and then adjust by the value of
  // the first few digits. But, I'm not sure how accurate that could be.

  // Compute a sufficient number of bits that is always large enough but might
  // be too large. This avoids the assertion in the constructor.
  uint32_t sufficient = slen*64/18;

  // Convert to the actual binary value.
  APInt tmp(sufficient, str, slen, radix);

  // Compute how many bits are required.
  return isNegative + tmp.logBase2() + 1;
}

uint64_t APInt::getHashValue() const {
  // Put the bit width into the low order bits.
  uint64_t hash = BitWidth;

  // Add the sum of the words to the hash.
  if (isSingleWord())
    hash += VAL << 6; // clear separation of up to 64 bits
  else
    for (uint32_t i = 0; i < getNumWords(); ++i)
      hash += pVal[i] << 6; // clear sepration of up to 64 bits
  return hash;
}

/// HiBits - This function returns the high "numBits" bits of this APInt.
APInt APInt::getHiBits(uint32_t numBits) const {
  return APIntOps::lshr(*this, BitWidth - numBits);
}

/// LoBits - This function returns the low "numBits" bits of this APInt.
APInt APInt::getLoBits(uint32_t numBits) const {
  return APIntOps::lshr(APIntOps::shl(*this, BitWidth - numBits), 
                        BitWidth - numBits);
}

bool APInt::isPowerOf2() const {
  return (!!*this) && !(*this & (*this - APInt(BitWidth,1)));
}

uint32_t APInt::countLeadingZeros() const {
  uint32_t Count = 0;
  if (isSingleWord())
    Count = CountLeadingZeros_64(VAL);
  else {
    for (uint32_t i = getNumWords(); i > 0u; --i) {
      if (pVal[i-1] == 0)
        Count += APINT_BITS_PER_WORD;
      else {
        Count += CountLeadingZeros_64(pVal[i-1]);
        break;
      }
    }
  }
  uint32_t remainder = BitWidth % APINT_BITS_PER_WORD;
  if (remainder)
    Count -= APINT_BITS_PER_WORD - remainder;
  return Count;
}

static uint32_t countLeadingOnes_64(uint64_t V, uint32_t skip) {
  uint32_t Count = 0;
  if (skip)
    V <<= skip;
  while (V && (V & (1ULL << 63))) {
    Count++;
    V <<= 1;
  }
  return Count;
}

uint32_t APInt::countLeadingOnes() const {
  if (isSingleWord())
    return countLeadingOnes_64(VAL, APINT_BITS_PER_WORD - BitWidth);

  uint32_t highWordBits = BitWidth % APINT_BITS_PER_WORD;
  uint32_t shift = (highWordBits == 0 ? 0 : APINT_BITS_PER_WORD - highWordBits);
  int i = getNumWords() - 1;
  uint32_t Count = countLeadingOnes_64(pVal[i], shift);
  if (Count == highWordBits) {
    for (i--; i >= 0; --i) {
      if (pVal[i] == -1ULL)
        Count += APINT_BITS_PER_WORD;
      else {
        Count += countLeadingOnes_64(pVal[i], 0);
        break;
      }
    }
  }
  return Count;
}

uint32_t APInt::countTrailingZeros() const {
  if (isSingleWord())
    return CountTrailingZeros_64(VAL);
  uint32_t Count = 0;
  uint32_t i = 0;
  for (; i < getNumWords() && pVal[i] == 0; ++i)
    Count += APINT_BITS_PER_WORD;
  if (i < getNumWords())
    Count += CountTrailingZeros_64(pVal[i]);
  return Count;
}

uint32_t APInt::countPopulation() const {
  if (isSingleWord())
    return CountPopulation_64(VAL);
  uint32_t Count = 0;
  for (uint32_t i = 0; i < getNumWords(); ++i)
    Count += CountPopulation_64(pVal[i]);
  return Count;
}

APInt APInt::byteSwap() const {
  assert(BitWidth >= 16 && BitWidth % 16 == 0 && "Cannot byteswap!");
  if (BitWidth == 16)
    return APInt(BitWidth, ByteSwap_16(uint16_t(VAL)));
  else if (BitWidth == 32)
    return APInt(BitWidth, ByteSwap_32(uint32_t(VAL)));
  else if (BitWidth == 48) {
    uint32_t Tmp1 = uint32_t(VAL >> 16);
    Tmp1 = ByteSwap_32(Tmp1);
    uint16_t Tmp2 = uint16_t(VAL);
    Tmp2 = ByteSwap_16(Tmp2);
    return APInt(BitWidth, (uint64_t(Tmp2) << 32) | Tmp1);
  } else if (BitWidth == 64)
    return APInt(BitWidth, ByteSwap_64(VAL));
  else {
    APInt Result(BitWidth, 0);
    char *pByte = (char*)Result.pVal;
    for (uint32_t i = 0; i < BitWidth / APINT_WORD_SIZE / 2; ++i) {
      char Tmp = pByte[i];
      pByte[i] = pByte[BitWidth / APINT_WORD_SIZE - 1 - i];
      pByte[BitWidth / APINT_WORD_SIZE - i - 1] = Tmp;
    }
    return Result;
  }
}

APInt llvm::APIntOps::GreatestCommonDivisor(const APInt& API1, 
                                            const APInt& API2) {
  APInt A = API1, B = API2;
  while (!!B) {
    APInt T = B;
    B = APIntOps::urem(A, B);
    A = T;
  }
  return A;
}

APInt llvm::APIntOps::RoundDoubleToAPInt(double Double, uint32_t width) {
  union {
    double D;
    uint64_t I;
  } T;
  T.D = Double;

  // Get the sign bit from the highest order bit
  bool isNeg = T.I >> 63;

  // Get the 11-bit exponent and adjust for the 1023 bit bias
  int64_t exp = ((T.I >> 52) & 0x7ff) - 1023;

  // If the exponent is negative, the value is < 0 so just return 0.
  if (exp < 0)
    return APInt(width, 0u);

  // Extract the mantissa by clearing the top 12 bits (sign + exponent).
  uint64_t mantissa = (T.I & (~0ULL >> 12)) | 1ULL << 52;

  // If the exponent doesn't shift all bits out of the mantissa
  if (exp < 52)
    return isNeg ? -APInt(width, mantissa >> (52 - exp)) : 
                    APInt(width, mantissa >> (52 - exp));

  // If the client didn't provide enough bits for us to shift the mantissa into
  // then the result is undefined, just return 0
  if (width <= exp - 52)
    return APInt(width, 0);

  // Otherwise, we have to shift the mantissa bits up to the right location
  APInt Tmp(width, mantissa);
  Tmp = Tmp.shl(exp - 52);
  return isNeg ? -Tmp : Tmp;
}

/// RoundToDouble - This function convert this APInt to a double.
/// The layout for double is as following (IEEE Standard 754):
///  --------------------------------------
/// |  Sign    Exponent    Fraction    Bias |
/// |-------------------------------------- |
/// |  1[63]   11[62-52]   52[51-00]   1023 |
///  -------------------------------------- 
double APInt::roundToDouble(bool isSigned) const {

  // Handle the simple case where the value is contained in one uint64_t.
  if (isSingleWord() || getActiveBits() <= APINT_BITS_PER_WORD) {
    if (isSigned) {
      int64_t sext = (int64_t(VAL) << (64-BitWidth)) >> (64-BitWidth);
      return double(sext);
    } else
      return double(VAL);
  }

  // Determine if the value is negative.
  bool isNeg = isSigned ? (*this)[BitWidth-1] : false;

  // Construct the absolute value if we're negative.
  APInt Tmp(isNeg ? -(*this) : (*this));

  // Figure out how many bits we're using.
  uint32_t n = Tmp.getActiveBits();

  // The exponent (without bias normalization) is just the number of bits
  // we are using. Note that the sign bit is gone since we constructed the
  // absolute value.
  uint64_t exp = n;

  // Return infinity for exponent overflow
  if (exp > 1023) {
    if (!isSigned || !isNeg)
      return std::numeric_limits<double>::infinity();
    else 
      return -std::numeric_limits<double>::infinity();
  }
  exp += 1023; // Increment for 1023 bias

  // Number of bits in mantissa is 52. To obtain the mantissa value, we must
  // extract the high 52 bits from the correct words in pVal.
  uint64_t mantissa;
  unsigned hiWord = whichWord(n-1);
  if (hiWord == 0) {
    mantissa = Tmp.pVal[0];
    if (n > 52)
      mantissa >>= n - 52; // shift down, we want the top 52 bits.
  } else {
    assert(hiWord > 0 && "huh?");
    uint64_t hibits = Tmp.pVal[hiWord] << (52 - n % APINT_BITS_PER_WORD);
    uint64_t lobits = Tmp.pVal[hiWord-1] >> (11 + n % APINT_BITS_PER_WORD);
    mantissa = hibits | lobits;
  }

  // The leading bit of mantissa is implicit, so get rid of it.
  uint64_t sign = isNeg ? (1ULL << (APINT_BITS_PER_WORD - 1)) : 0;
  union {
    double D;
    uint64_t I;
  } T;
  T.I = sign | (exp << 52) | mantissa;
  return T.D;
}

// Truncate to new width.
APInt &APInt::trunc(uint32_t width) {
  assert(width < BitWidth && "Invalid APInt Truncate request");
  assert(width >= IntegerType::MIN_INT_BITS && "Can't truncate to 0 bits");
  uint32_t wordsBefore = getNumWords();
  BitWidth = width;
  uint32_t wordsAfter = getNumWords();
  if (wordsBefore != wordsAfter) {
    if (wordsAfter == 1) {
      uint64_t *tmp = pVal;
      VAL = pVal[0];
      delete [] tmp;
    } else {
      uint64_t *newVal = getClearedMemory(wordsAfter);
      for (uint32_t i = 0; i < wordsAfter; ++i)
        newVal[i] = pVal[i];
      delete [] pVal;
      pVal = newVal;
    }
  }
  return clearUnusedBits();
}

// Sign extend to a new width.
APInt &APInt::sext(uint32_t width) {
  assert(width > BitWidth && "Invalid APInt SignExtend request");
  assert(width <= IntegerType::MAX_INT_BITS && "Too many bits");
  // If the sign bit isn't set, this is the same as zext.
  if (!isNegative()) {
    zext(width);
    return *this;
  }

  // The sign bit is set. First, get some facts
  uint32_t wordsBefore = getNumWords();
  uint32_t wordBits = BitWidth % APINT_BITS_PER_WORD;
  BitWidth = width;
  uint32_t wordsAfter = getNumWords();

  // Mask the high order word appropriately
  if (wordsBefore == wordsAfter) {
    uint32_t newWordBits = width % APINT_BITS_PER_WORD;
    // The extension is contained to the wordsBefore-1th word.
    uint64_t mask = ~0ULL;
    if (newWordBits)
      mask >>= APINT_BITS_PER_WORD - newWordBits;
    mask <<= wordBits;
    if (wordsBefore == 1)
      VAL |= mask;
    else
      pVal[wordsBefore-1] |= mask;
    return clearUnusedBits();
  }

  uint64_t mask = wordBits == 0 ? 0 : ~0ULL << wordBits;
  uint64_t *newVal = getMemory(wordsAfter);
  if (wordsBefore == 1)
    newVal[0] = VAL | mask;
  else {
    for (uint32_t i = 0; i < wordsBefore; ++i)
      newVal[i] = pVal[i];
    newVal[wordsBefore-1] |= mask;
  }
  for (uint32_t i = wordsBefore; i < wordsAfter; i++)
    newVal[i] = -1ULL;
  if (wordsBefore != 1)
    delete [] pVal;
  pVal = newVal;
  return clearUnusedBits();
}

//  Zero extend to a new width.
APInt &APInt::zext(uint32_t width) {
  assert(width > BitWidth && "Invalid APInt ZeroExtend request");
  assert(width <= IntegerType::MAX_INT_BITS && "Too many bits");
  uint32_t wordsBefore = getNumWords();
  BitWidth = width;
  uint32_t wordsAfter = getNumWords();
  if (wordsBefore != wordsAfter) {
    uint64_t *newVal = getClearedMemory(wordsAfter);
    if (wordsBefore == 1)
      newVal[0] = VAL;
    else 
      for (uint32_t i = 0; i < wordsBefore; ++i)
        newVal[i] = pVal[i];
    if (wordsBefore != 1)
      delete [] pVal;
    pVal = newVal;
  }
  return *this;
}

APInt &APInt::zextOrTrunc(uint32_t width) {
  if (BitWidth < width)
    return zext(width);
  if (BitWidth > width)
    return trunc(width);
  return *this;
}

APInt &APInt::sextOrTrunc(uint32_t width) {
  if (BitWidth < width)
    return sext(width);
  if (BitWidth > width)
    return trunc(width);
  return *this;
}

/// Arithmetic right-shift this APInt by shiftAmt.
/// @brief Arithmetic right-shift function.
APInt APInt::ashr(uint32_t shiftAmt) const {
  assert(shiftAmt <= BitWidth && "Invalid shift amount");
  // Handle a degenerate case
  if (shiftAmt == 0)
    return *this;

  // Handle single word shifts with built-in ashr
  if (isSingleWord()) {
    if (shiftAmt == BitWidth)
      return APInt(BitWidth, 0); // undefined
    else {
      uint32_t SignBit = APINT_BITS_PER_WORD - BitWidth;
      return APInt(BitWidth, 
        (((int64_t(VAL) << SignBit) >> SignBit) >> shiftAmt));
    }
  }

  // If all the bits were shifted out, the result is, technically, undefined.
  // We return -1 if it was negative, 0 otherwise. We check this early to avoid
  // issues in the algorithm below.
  if (shiftAmt == BitWidth) {
    if (isNegative())
      return APInt(BitWidth, -1ULL);
    else
      return APInt(BitWidth, 0);
  }

  // Create some space for the result.
  uint64_t * val = new uint64_t[getNumWords()];

  // Compute some values needed by the following shift algorithms
  uint32_t wordShift = shiftAmt % APINT_BITS_PER_WORD; // bits to shift per word
  uint32_t offset = shiftAmt / APINT_BITS_PER_WORD; // word offset for shift
  uint32_t breakWord = getNumWords() - 1 - offset; // last word affected
  uint32_t bitsInWord = whichBit(BitWidth); // how many bits in last word?
  if (bitsInWord == 0)
    bitsInWord = APINT_BITS_PER_WORD;

  // If we are shifting whole words, just move whole words
  if (wordShift == 0) {
    // Move the words containing significant bits
    for (uint32_t i = 0; i <= breakWord; ++i) 
      val[i] = pVal[i+offset]; // move whole word

    // Adjust the top significant word for sign bit fill, if negative
    if (isNegative())
      if (bitsInWord < APINT_BITS_PER_WORD)
        val[breakWord] |= ~0ULL << bitsInWord; // set high bits
  } else {
    // Shift the low order words 
    for (uint32_t i = 0; i < breakWord; ++i) {
      // This combines the shifted corresponding word with the low bits from
      // the next word (shifted into this word's high bits).
      val[i] = (pVal[i+offset] >> wordShift) | 
               (pVal[i+offset+1] << (APINT_BITS_PER_WORD - wordShift));
    }

    // Shift the break word. In this case there are no bits from the next word
    // to include in this word.
    val[breakWord] = pVal[breakWord+offset] >> wordShift;

    // Deal with sign extenstion in the break word, and possibly the word before
    // it.
    if (isNegative()) {
      if (wordShift > bitsInWord) {
        if (breakWord > 0)
          val[breakWord-1] |= 
            ~0ULL << (APINT_BITS_PER_WORD - (wordShift - bitsInWord));
        val[breakWord] |= ~0ULL;
      } else 
        val[breakWord] |= (~0ULL << (bitsInWord - wordShift));
    }
  }

  // Remaining words are 0 or -1, just assign them.
  uint64_t fillValue = (isNegative() ? -1ULL : 0);
  for (uint32_t i = breakWord+1; i < getNumWords(); ++i)
    val[i] = fillValue;
  return APInt(val, BitWidth).clearUnusedBits();
}

/// Logical right-shift this APInt by shiftAmt.
/// @brief Logical right-shift function.
APInt APInt::lshr(uint32_t shiftAmt) const {
  if (isSingleWord()) {
    if (shiftAmt == BitWidth)
      return APInt(BitWidth, 0);
    else 
      return APInt(BitWidth, this->VAL >> shiftAmt);
  }

  // If all the bits were shifted out, the result is 0. This avoids issues
  // with shifting by the size of the integer type, which produces undefined
  // results. We define these "undefined results" to always be 0.
  if (shiftAmt == BitWidth)
    return APInt(BitWidth, 0);

  // Create some space for the result.
  uint64_t * val = new uint64_t[getNumWords()];

  // If we are shifting less than a word, compute the shift with a simple carry
  if (shiftAmt < APINT_BITS_PER_WORD) {
    uint64_t carry = 0;
    for (int i = getNumWords()-1; i >= 0; --i) {
      val[i] = (pVal[i] >> shiftAmt) | carry;
      carry = pVal[i] << (APINT_BITS_PER_WORD - shiftAmt);
    }
    return APInt(val, BitWidth).clearUnusedBits();
  }

  // Compute some values needed by the remaining shift algorithms
  uint32_t wordShift = shiftAmt % APINT_BITS_PER_WORD;
  uint32_t offset = shiftAmt / APINT_BITS_PER_WORD;

  // If we are shifting whole words, just move whole words
  if (wordShift == 0) {
    for (uint32_t i = 0; i < getNumWords() - offset; ++i) 
      val[i] = pVal[i+offset];
    for (uint32_t i = getNumWords()-offset; i < getNumWords(); i++)
      val[i] = 0;
    return APInt(val,BitWidth).clearUnusedBits();
  }

  // Shift the low order words 
  uint32_t breakWord = getNumWords() - offset -1;
  for (uint32_t i = 0; i < breakWord; ++i)
    val[i] = (pVal[i+offset] >> wordShift) |
             (pVal[i+offset+1] << (APINT_BITS_PER_WORD - wordShift));
  // Shift the break word.
  val[breakWord] = pVal[breakWord+offset] >> wordShift;

  // Remaining words are 0
  for (uint32_t i = breakWord+1; i < getNumWords(); ++i)
    val[i] = 0;
  return APInt(val, BitWidth).clearUnusedBits();
}

/// Left-shift this APInt by shiftAmt.
/// @brief Left-shift function.
APInt APInt::shl(uint32_t shiftAmt) const {
  assert(shiftAmt <= BitWidth && "Invalid shift amount");
  if (isSingleWord()) {
    if (shiftAmt == BitWidth)
      return APInt(BitWidth, 0); // avoid undefined shift results
    return APInt(BitWidth, VAL << shiftAmt);
  }

  // If all the bits were shifted out, the result is 0. This avoids issues
  // with shifting by the size of the integer type, which produces undefined
  // results. We define these "undefined results" to always be 0.
  if (shiftAmt == BitWidth)
    return APInt(BitWidth, 0);

  // If none of the bits are shifted out, the result is *this. This avoids a
  // lshr by the words size in the loop below which can produce incorrect
  // results. It also avoids the expensive computation below for a common case.
  if (shiftAmt == 0)
    return *this;

  // Create some space for the result.
  uint64_t * val = new uint64_t[getNumWords()];

  // If we are shifting less than a word, do it the easy way
  if (shiftAmt < APINT_BITS_PER_WORD) {
    uint64_t carry = 0;
    for (uint32_t i = 0; i < getNumWords(); i++) {
      val[i] = pVal[i] << shiftAmt | carry;
      carry = pVal[i] >> (APINT_BITS_PER_WORD - shiftAmt);
    }
    return APInt(val, BitWidth).clearUnusedBits();
  }

  // Compute some values needed by the remaining shift algorithms
  uint32_t wordShift = shiftAmt % APINT_BITS_PER_WORD;
  uint32_t offset = shiftAmt / APINT_BITS_PER_WORD;

  // If we are shifting whole words, just move whole words
  if (wordShift == 0) {
    for (uint32_t i = 0; i < offset; i++) 
      val[i] = 0;
    for (uint32_t i = offset; i < getNumWords(); i++)
      val[i] = pVal[i-offset];
    return APInt(val,BitWidth).clearUnusedBits();
  }

  // Copy whole words from this to Result.
  uint32_t i = getNumWords() - 1;
  for (; i > offset; --i)
    val[i] = pVal[i-offset] << wordShift |
             pVal[i-offset-1] >> (APINT_BITS_PER_WORD - wordShift);
  val[offset] = pVal[0] << wordShift;
  for (i = 0; i < offset; ++i)
    val[i] = 0;
  return APInt(val, BitWidth).clearUnusedBits();
}


// Square Root - this method computes and returns the square root of "this".
// Three mechanisms are used for computation. For small values (<= 5 bits),
// a table lookup is done. This gets some performance for common cases. For
// values using less than 52 bits, the value is converted to double and then
// the libc sqrt function is called. The result is rounded and then converted
// back to a uint64_t which is then used to construct the result. Finally,
// the Babylonian method for computing square roots is used. 
APInt APInt::sqrt() const {

  // Determine the magnitude of the value.
  uint32_t magnitude = getActiveBits();

  // Use a fast table for some small values. This also gets rid of some
  // rounding errors in libc sqrt for small values.
  if (magnitude <= 5) {
    static const uint8_t results[32] = {
      /*     0 */ 0,
      /*  1- 2 */ 1, 1,
      /*  3- 6 */ 2, 2, 2, 2, 
      /*  7-12 */ 3, 3, 3, 3, 3, 3,
      /* 13-20 */ 4, 4, 4, 4, 4, 4, 4, 4,
      /* 21-30 */ 5, 5, 5, 5, 5, 5, 5, 5, 5, 5,
      /*    31 */ 6
    };
    return APInt(BitWidth, results[ (isSingleWord() ? VAL : pVal[0]) ]);
  }

  // If the magnitude of the value fits in less than 52 bits (the precision of
  // an IEEE double precision floating point value), then we can use the
  // libc sqrt function which will probably use a hardware sqrt computation.
  // This should be faster than the algorithm below.
  if (magnitude < 52) {
#ifdef _MSC_VER
    // Amazingly, VC++ doesn't have round().
    return APInt(BitWidth, 
                 uint64_t(::sqrt(double(isSingleWord()?VAL:pVal[0]))) + 0.5);
#else
    return APInt(BitWidth, 
                 uint64_t(::round(::sqrt(double(isSingleWord()?VAL:pVal[0])))));
#endif
  }

  // Okay, all the short cuts are exhausted. We must compute it. The following
  // is a classical Babylonian method for computing the square root. This code
  // was adapted to APINt from a wikipedia article on such computations.
  // See http://www.wikipedia.org/ and go to the page named
  // Calculate_an_integer_square_root. 
  uint32_t nbits = BitWidth, i = 4;
  APInt testy(BitWidth, 16);
  APInt x_old(BitWidth, 1);
  APInt x_new(BitWidth, 0);
  APInt two(BitWidth, 2);

  // Select a good starting value using binary logarithms.
  for (;; i += 2, testy = testy.shl(2)) 
    if (i >= nbits || this->ule(testy)) {
      x_old = x_old.shl(i / 2);
      break;
    }

  // Use the Babylonian method to arrive at the integer square root: 
  for (;;) {
    x_new = (this->udiv(x_old) + x_old).udiv(two);
    if (x_old.ule(x_new))
      break;
    x_old = x_new;
  }

  // Make sure we return the closest approximation
  // NOTE: The rounding calculation below is correct. It will produce an 
  // off-by-one discrepancy with results from pari/gp. That discrepancy has been
  // determined to be a rounding issue with pari/gp as it begins to use a 
  // floating point representation after 192 bits. There are no discrepancies
  // between this algorithm and pari/gp for bit widths < 192 bits.
  APInt square(x_old * x_old);
  APInt nextSquare((x_old + 1) * (x_old +1));
  if (this->ult(square))
    return x_old;
  else if (this->ule(nextSquare)) {
    APInt midpoint((nextSquare - square).udiv(two));
    APInt offset(*this - square);
    if (offset.ult(midpoint))
      return x_old;
    else
      return x_old + 1;
  } else
    assert(0 && "Error in APInt::sqrt computation");
  return x_old + 1;
}

/// Implementation of Knuth's Algorithm D (Division of nonnegative integers)
/// from "Art of Computer Programming, Volume 2", section 4.3.1, p. 272. The
/// variables here have the same names as in the algorithm. Comments explain
/// the algorithm and any deviation from it.
static void KnuthDiv(uint32_t *u, uint32_t *v, uint32_t *q, uint32_t* r, 
                     uint32_t m, uint32_t n) {
  assert(u && "Must provide dividend");
  assert(v && "Must provide divisor");
  assert(q && "Must provide quotient");
  assert(u != v && u != q && v != q && "Must us different memory");
  assert(n>1 && "n must be > 1");

  // Knuth uses the value b as the base of the number system. In our case b
  // is 2^31 so we just set it to -1u.
  uint64_t b = uint64_t(1) << 32;

  DEBUG(cerr << "KnuthDiv: m=" << m << " n=" << n << '\n');
  DEBUG(cerr << "KnuthDiv: original:");
  DEBUG(for (int i = m+n; i >=0; i--) cerr << " " << std::setbase(16) << u[i]);
  DEBUG(cerr << " by");
  DEBUG(for (int i = n; i >0; i--) cerr << " " << std::setbase(16) << v[i-1]);
  DEBUG(cerr << '\n');
  // D1. [Normalize.] Set d = b / (v[n-1] + 1) and multiply all the digits of 
  // u and v by d. Note that we have taken Knuth's advice here to use a power 
  // of 2 value for d such that d * v[n-1] >= b/2 (b is the base). A power of 
  // 2 allows us to shift instead of multiply and it is easy to determine the 
  // shift amount from the leading zeros.  We are basically normalizing the u
  // and v so that its high bits are shifted to the top of v's range without
  // overflow. Note that this can require an extra word in u so that u must
  // be of length m+n+1.
  uint32_t shift = CountLeadingZeros_32(v[n-1]);
  uint32_t v_carry = 0;
  uint32_t u_carry = 0;
  if (shift) {
    for (uint32_t i = 0; i < m+n; ++i) {
      uint32_t u_tmp = u[i] >> (32 - shift);
      u[i] = (u[i] << shift) | u_carry;
      u_carry = u_tmp;
    }
    for (uint32_t i = 0; i < n; ++i) {
      uint32_t v_tmp = v[i] >> (32 - shift);
      v[i] = (v[i] << shift) | v_carry;
      v_carry = v_tmp;
    }
  }
  u[m+n] = u_carry;
  DEBUG(cerr << "KnuthDiv:   normal:");
  DEBUG(for (int i = m+n; i >=0; i--) cerr << " " << std::setbase(16) << u[i]);
  DEBUG(cerr << " by");
  DEBUG(for (int i = n; i >0; i--) cerr << " " << std::setbase(16) << v[i-1]);
  DEBUG(cerr << '\n');

  // D2. [Initialize j.]  Set j to m. This is the loop counter over the places.
  int j = m;
  do {
    DEBUG(cerr << "KnuthDiv: quotient digit #" << j << '\n');
    // D3. [Calculate q'.]. 
    //     Set qp = (u[j+n]*b + u[j+n-1]) / v[n-1]. (qp=qprime=q')
    //     Set rp = (u[j+n]*b + u[j+n-1]) % v[n-1]. (rp=rprime=r')
    // Now test if qp == b or qp*v[n-2] > b*rp + u[j+n-2]; if so, decrease
    // qp by 1, inrease rp by v[n-1], and repeat this test if rp < b. The test
    // on v[n-2] determines at high speed most of the cases in which the trial
    // value qp is one too large, and it eliminates all cases where qp is two 
    // too large. 
    uint64_t dividend = ((uint64_t(u[j+n]) << 32) + u[j+n-1]);
    DEBUG(cerr << "KnuthDiv: dividend == " << dividend << '\n');
    uint64_t qp = dividend / v[n-1];
    uint64_t rp = dividend % v[n-1];
    if (qp == b || qp*v[n-2] > b*rp + u[j+n-2]) {
      qp--;
      rp += v[n-1];
      if (rp < b && (qp == b || qp*v[n-2] > b*rp + u[j+n-2]))
        qp--;
    }
    DEBUG(cerr << "KnuthDiv: qp == " << qp << ", rp == " << rp << '\n');

    // D4. [Multiply and subtract.] Replace (u[j+n]u[j+n-1]...u[j]) with
    // (u[j+n]u[j+n-1]..u[j]) - qp * (v[n-1]...v[1]v[0]). This computation
    // consists of a simple multiplication by a one-place number, combined with
    // a subtraction. 
    bool isNeg = false;
    for (uint32_t i = 0; i < n; ++i) {
      uint64_t u_tmp = uint64_t(u[j+i]) | (uint64_t(u[j+i+1]) << 32);
      uint64_t subtrahend = uint64_t(qp) * uint64_t(v[i]);
      bool borrow = subtrahend > u_tmp;
      DEBUG(cerr << "KnuthDiv: u_tmp == " << u_tmp 
                 << ", subtrahend == " << subtrahend
                 << ", borrow = " << borrow << '\n');

      uint64_t result = u_tmp - subtrahend;
      uint32_t k = j + i;
      u[k++] = result & (b-1); // subtract low word
      u[k++] = result >> 32;   // subtract high word
      while (borrow && k <= m+n) { // deal with borrow to the left
        borrow = u[k] == 0;
        u[k]--;
        k++;
      }
      isNeg |= borrow;
      DEBUG(cerr << "KnuthDiv: u[j+i] == " << u[j+i] << ",  u[j+i+1] == " << 
                    u[j+i+1] << '\n'); 
    }
    DEBUG(cerr << "KnuthDiv: after subtraction:");
    DEBUG(for (int i = m+n; i >=0; i--) cerr << " " << u[i]);
    DEBUG(cerr << '\n');
    // The digits (u[j+n]...u[j]) should be kept positive; if the result of 
    // this step is actually negative, (u[j+n]...u[j]) should be left as the 
    // true value plus b**(n+1), namely as the b's complement of
    // the true value, and a "borrow" to the left should be remembered.
    //
    if (isNeg) {
      bool carry = true;  // true because b's complement is "complement + 1"
      for (uint32_t i = 0; i <= m+n; ++i) {
        u[i] = ~u[i] + carry; // b's complement
        carry = carry && u[i] == 0;
      }
    }
    DEBUG(cerr << "KnuthDiv: after complement:");
    DEBUG(for (int i = m+n; i >=0; i--) cerr << " " << u[i]);
    DEBUG(cerr << '\n');

    // D5. [Test remainder.] Set q[j] = qp. If the result of step D4 was 
    // negative, go to step D6; otherwise go on to step D7.
    q[j] = qp;
    if (isNeg) {
      // D6. [Add back]. The probability that this step is necessary is very 
      // small, on the order of only 2/b. Make sure that test data accounts for
      // this possibility. Decrease q[j] by 1 
      q[j]--;
      // and add (0v[n-1]...v[1]v[0]) to (u[j+n]u[j+n-1]...u[j+1]u[j]). 
      // A carry will occur to the left of u[j+n], and it should be ignored 
      // since it cancels with the borrow that occurred in D4.
      bool carry = false;
      for (uint32_t i = 0; i < n; i++) {
        uint32_t limit = std::min(u[j+i],v[i]);
        u[j+i] += v[i] + carry;
        carry = u[j+i] < limit || (carry && u[j+i] == limit);
      }
      u[j+n] += carry;
    }
    DEBUG(cerr << "KnuthDiv: after correction:");
    DEBUG(for (int i = m+n; i >=0; i--) cerr <<" " << u[i]);
    DEBUG(cerr << "\nKnuthDiv: digit result = " << q[j] << '\n');

  // D7. [Loop on j.]  Decrease j by one. Now if j >= 0, go back to D3.
  } while (--j >= 0);

  DEBUG(cerr << "KnuthDiv: quotient:");
  DEBUG(for (int i = m; i >=0; i--) cerr <<" " << q[i]);
  DEBUG(cerr << '\n');

  // D8. [Unnormalize]. Now q[...] is the desired quotient, and the desired
  // remainder may be obtained by dividing u[...] by d. If r is non-null we
  // compute the remainder (urem uses this).
  if (r) {
    // The value d is expressed by the "shift" value above since we avoided
    // multiplication by d by using a shift left. So, all we have to do is
    // shift right here. In order to mak
    if (shift) {
      uint32_t carry = 0;
      DEBUG(cerr << "KnuthDiv: remainder:");
      for (int i = n-1; i >= 0; i--) {
        r[i] = (u[i] >> shift) | carry;
        carry = u[i] << (32 - shift);
        DEBUG(cerr << " " << r[i]);
      }
    } else {
      for (int i = n-1; i >= 0; i--) {
        r[i] = u[i];
        DEBUG(cerr << " " << r[i]);
      }
    }
    DEBUG(cerr << '\n');
  }
  DEBUG(cerr << std::setbase(10) << '\n');
}

void APInt::divide(const APInt LHS, uint32_t lhsWords, 
                   const APInt &RHS, uint32_t rhsWords,
                   APInt *Quotient, APInt *Remainder)
{
  assert(lhsWords >= rhsWords && "Fractional result");

  // First, compose the values into an array of 32-bit words instead of 
  // 64-bit words. This is a necessity of both the "short division" algorithm
  // and the the Knuth "classical algorithm" which requires there to be native 
  // operations for +, -, and * on an m bit value with an m*2 bit result. We 
  // can't use 64-bit operands here because we don't have native results of 
  // 128-bits. Furthremore, casting the 64-bit values to 32-bit values won't 
  // work on large-endian machines.
  uint64_t mask = ~0ull >> (sizeof(uint32_t)*8);
  uint32_t n = rhsWords * 2;
  uint32_t m = (lhsWords * 2) - n;

  // Allocate space for the temporary values we need either on the stack, if
  // it will fit, or on the heap if it won't.
  uint32_t SPACE[128];
  uint32_t *U = 0;
  uint32_t *V = 0;
  uint32_t *Q = 0;
  uint32_t *R = 0;
  if ((Remainder?4:3)*n+2*m+1 <= 128) {
    U = &SPACE[0];
    V = &SPACE[m+n+1];
    Q = &SPACE[(m+n+1) + n];
    if (Remainder)
      R = &SPACE[(m+n+1) + n + (m+n)];
  } else {
    U = new uint32_t[m + n + 1];
    V = new uint32_t[n];
    Q = new uint32_t[m+n];
    if (Remainder)
      R = new uint32_t[n];
  }

  // Initialize the dividend
  memset(U, 0, (m+n+1)*sizeof(uint32_t));
  for (unsigned i = 0; i < lhsWords; ++i) {
    uint64_t tmp = (LHS.getNumWords() == 1 ? LHS.VAL : LHS.pVal[i]);
    U[i * 2] = tmp & mask;
    U[i * 2 + 1] = tmp >> (sizeof(uint32_t)*8);
  }
  U[m+n] = 0; // this extra word is for "spill" in the Knuth algorithm.

  // Initialize the divisor
  memset(V, 0, (n)*sizeof(uint32_t));
  for (unsigned i = 0; i < rhsWords; ++i) {
    uint64_t tmp = (RHS.getNumWords() == 1 ? RHS.VAL : RHS.pVal[i]);
    V[i * 2] = tmp & mask;
    V[i * 2 + 1] = tmp >> (sizeof(uint32_t)*8);
  }

  // initialize the quotient and remainder
  memset(Q, 0, (m+n) * sizeof(uint32_t));
  if (Remainder)
    memset(R, 0, n * sizeof(uint32_t));

  // Now, adjust m and n for the Knuth division. n is the number of words in 
  // the divisor. m is the number of words by which the dividend exceeds the
  // divisor (i.e. m+n is the length of the dividend). These sizes must not 
  // contain any zero words or the Knuth algorithm fails.
  for (unsigned i = n; i > 0 && V[i-1] == 0; i--) {
    n--;
    m++;
  }
  for (unsigned i = m+n; i > 0 && U[i-1] == 0; i--)
    m--;

  // If we're left with only a single word for the divisor, Knuth doesn't work
  // so we implement the short division algorithm here. This is much simpler
  // and faster because we are certain that we can divide a 64-bit quantity
  // by a 32-bit quantity at hardware speed and short division is simply a
  // series of such operations. This is just like doing short division but we
  // are using base 2^32 instead of base 10.
  assert(n != 0 && "Divide by zero?");
  if (n == 1) {
    uint32_t divisor = V[0];
    uint32_t remainder = 0;
    for (int i = m+n-1; i >= 0; i--) {
      uint64_t partial_dividend = uint64_t(remainder) << 32 | U[i];
      if (partial_dividend == 0) {
        Q[i] = 0;
        remainder = 0;
      } else if (partial_dividend < divisor) {
        Q[i] = 0;
        remainder = partial_dividend;
      } else if (partial_dividend == divisor) {
        Q[i] = 1;
        remainder = 0;
      } else {
        Q[i] = partial_dividend / divisor;
        remainder = partial_dividend - (Q[i] * divisor);
      }
    }
    if (R)
      R[0] = remainder;
  } else {
    // Now we're ready to invoke the Knuth classical divide algorithm. In this
    // case n > 1.
    KnuthDiv(U, V, Q, R, m, n);
  }

  // If the caller wants the quotient
  if (Quotient) {
    // Set up the Quotient value's memory.
    if (Quotient->BitWidth != LHS.BitWidth) {
      if (Quotient->isSingleWord())
        Quotient->VAL = 0;
      else
        delete [] Quotient->pVal;
      Quotient->BitWidth = LHS.BitWidth;
      if (!Quotient->isSingleWord())
        Quotient->pVal = getClearedMemory(Quotient->getNumWords());
    } else
      Quotient->clear();

    // The quotient is in Q. Reconstitute the quotient into Quotient's low 
    // order words.
    if (lhsWords == 1) {
      uint64_t tmp = 
        uint64_t(Q[0]) | (uint64_t(Q[1]) << (APINT_BITS_PER_WORD / 2));
      if (Quotient->isSingleWord())
        Quotient->VAL = tmp;
      else
        Quotient->pVal[0] = tmp;
    } else {
      assert(!Quotient->isSingleWord() && "Quotient APInt not large enough");
      for (unsigned i = 0; i < lhsWords; ++i)
        Quotient->pVal[i] = 
          uint64_t(Q[i*2]) | (uint64_t(Q[i*2+1]) << (APINT_BITS_PER_WORD / 2));
    }
  }

  // If the caller wants the remainder
  if (Remainder) {
    // Set up the Remainder value's memory.
    if (Remainder->BitWidth != RHS.BitWidth) {
      if (Remainder->isSingleWord())
        Remainder->VAL = 0;
      else
        delete [] Remainder->pVal;
      Remainder->BitWidth = RHS.BitWidth;
      if (!Remainder->isSingleWord())
        Remainder->pVal = getClearedMemory(Remainder->getNumWords());
    } else
      Remainder->clear();

    // The remainder is in R. Reconstitute the remainder into Remainder's low
    // order words.
    if (rhsWords == 1) {
      uint64_t tmp = 
        uint64_t(R[0]) | (uint64_t(R[1]) << (APINT_BITS_PER_WORD / 2));
      if (Remainder->isSingleWord())
        Remainder->VAL = tmp;
      else
        Remainder->pVal[0] = tmp;
    } else {
      assert(!Remainder->isSingleWord() && "Remainder APInt not large enough");
      for (unsigned i = 0; i < rhsWords; ++i)
        Remainder->pVal[i] = 
          uint64_t(R[i*2]) | (uint64_t(R[i*2+1]) << (APINT_BITS_PER_WORD / 2));
    }
  }

  // Clean up the memory we allocated.
  if (U != &SPACE[0]) {
    delete [] U;
    delete [] V;
    delete [] Q;
    delete [] R;
  }
}

APInt APInt::udiv(const APInt& RHS) const {
  assert(BitWidth == RHS.BitWidth && "Bit widths must be the same");

  // First, deal with the easy case
  if (isSingleWord()) {
    assert(RHS.VAL != 0 && "Divide by zero?");
    return APInt(BitWidth, VAL / RHS.VAL);
  }

  // Get some facts about the LHS and RHS number of bits and words
  uint32_t rhsBits = RHS.getActiveBits();
  uint32_t rhsWords = !rhsBits ? 0 : (APInt::whichWord(rhsBits - 1) + 1);
  assert(rhsWords && "Divided by zero???");
  uint32_t lhsBits = this->getActiveBits();
  uint32_t lhsWords = !lhsBits ? 0 : (APInt::whichWord(lhsBits - 1) + 1);

  // Deal with some degenerate cases
  if (!lhsWords) 
    // 0 / X ===> 0
    return APInt(BitWidth, 0); 
  else if (lhsWords < rhsWords || this->ult(RHS)) {
    // X / Y ===> 0, iff X < Y
    return APInt(BitWidth, 0);
  } else if (*this == RHS) {
    // X / X ===> 1
    return APInt(BitWidth, 1);
  } else if (lhsWords == 1 && rhsWords == 1) {
    // All high words are zero, just use native divide
    return APInt(BitWidth, this->pVal[0] / RHS.pVal[0]);
  }

  // We have to compute it the hard way. Invoke the Knuth divide algorithm.
  APInt Quotient(1,0); // to hold result.
  divide(*this, lhsWords, RHS, rhsWords, &Quotient, 0);
  return Quotient;
}

APInt APInt::urem(const APInt& RHS) const {
  assert(BitWidth == RHS.BitWidth && "Bit widths must be the same");
  if (isSingleWord()) {
    assert(RHS.VAL != 0 && "Remainder by zero?");
    return APInt(BitWidth, VAL % RHS.VAL);
  }

  // Get some facts about the LHS
  uint32_t lhsBits = getActiveBits();
  uint32_t lhsWords = !lhsBits ? 0 : (whichWord(lhsBits - 1) + 1);

  // Get some facts about the RHS
  uint32_t rhsBits = RHS.getActiveBits();
  uint32_t rhsWords = !rhsBits ? 0 : (APInt::whichWord(rhsBits - 1) + 1);
  assert(rhsWords && "Performing remainder operation by zero ???");

  // Check the degenerate cases
  if (lhsWords == 0) {
    // 0 % Y ===> 0
    return APInt(BitWidth, 0);
  } else if (lhsWords < rhsWords || this->ult(RHS)) {
    // X % Y ===> X, iff X < Y
    return *this;
  } else if (*this == RHS) {
    // X % X == 0;
    return APInt(BitWidth, 0);
  } else if (lhsWords == 1) {
    // All high words are zero, just use native remainder
    return APInt(BitWidth, pVal[0] % RHS.pVal[0]);
  }

  // We have to compute it the hard way. Invoke the Knute divide algorithm.
  APInt Remainder(1,0);
  divide(*this, lhsWords, RHS, rhsWords, 0, &Remainder);
  return Remainder;
}

void APInt::fromString(uint32_t numbits, const char *str, uint32_t slen, 
                       uint8_t radix) {
  // Check our assumptions here
  assert((radix == 10 || radix == 8 || radix == 16 || radix == 2) &&
         "Radix should be 2, 8, 10, or 16!");
  assert(str && "String is null?");
  bool isNeg = str[0] == '-';
  if (isNeg)
    str++, slen--;
  assert((slen <= numbits || radix != 2) && "Insufficient bit width");
  assert((slen*3 <= numbits || radix != 8) && "Insufficient bit width");
  assert((slen*4 <= numbits || radix != 16) && "Insufficient bit width");
  assert(((slen*64)/22 <= numbits || radix != 10) && "Insufficient bit width");

  // Allocate memory
  if (!isSingleWord())
    pVal = getClearedMemory(getNumWords());

  // Figure out if we can shift instead of multiply
  uint32_t shift = (radix == 16 ? 4 : radix == 8 ? 3 : radix == 2 ? 1 : 0);

  // Set up an APInt for the digit to add outside the loop so we don't
  // constantly construct/destruct it.
  APInt apdigit(getBitWidth(), 0);
  APInt apradix(getBitWidth(), radix);

  // Enter digit traversal loop
  for (unsigned i = 0; i < slen; i++) {
    // Get a digit
    uint32_t digit = 0;
    char cdigit = str[i];
    if (isdigit(cdigit))
      digit = cdigit - '0';
    else if (isxdigit(cdigit))
      if (cdigit >= 'a')
        digit = cdigit - 'a' + 10;
      else if (cdigit >= 'A')
        digit = cdigit - 'A' + 10;
      else
        assert(0 && "huh?");
    else
      assert(0 && "Invalid character in digit string");

    // Shift or multiple the value by the radix
    if (shift)
      this->shl(shift);
    else
      *this *= apradix;

    // Add in the digit we just interpreted
    if (apdigit.isSingleWord())
      apdigit.VAL = digit;
    else
      apdigit.pVal[0] = digit;
    *this += apdigit;
  }
  // If its negative, put it in two's complement form
  if (isNeg) {
    (*this)--;
    this->flip();
  }
}

std::string APInt::toString(uint8_t radix, bool wantSigned) const {
  assert((radix == 10 || radix == 8 || radix == 16 || radix == 2) &&
         "Radix should be 2, 8, 10, or 16!");
  static const char *digits[] = { 
    "0","1","2","3","4","5","6","7","8","9","A","B","C","D","E","F" 
  };
  std::string result;
  uint32_t bits_used = getActiveBits();
  if (isSingleWord()) {
    char buf[65];
    const char *format = (radix == 10 ? (wantSigned ? "%lld" : "%llu") :
       (radix == 16 ? "%llX" : (radix == 8 ? "%llo" : 0)));
    if (format) {
      if (wantSigned) {
        int64_t sextVal = (int64_t(VAL) << (APINT_BITS_PER_WORD-BitWidth)) >> 
                           (APINT_BITS_PER_WORD-BitWidth);
        sprintf(buf, format, sextVal);
      } else 
        sprintf(buf, format, VAL);
    } else {
      memset(buf, 0, 65);
      uint64_t v = VAL;
      while (bits_used) {
        uint32_t bit = v & 1;
        bits_used--;
        buf[bits_used] = digits[bit][0];
        v >>=1;
      }
    }
    result = buf;
    return result;
  }

  if (radix != 10) {
    uint64_t mask = radix - 1;
    uint32_t shift = (radix == 16 ? 4 : radix  == 8 ? 3 : 1);
    uint32_t nibbles = APINT_BITS_PER_WORD / shift;
    for (uint32_t i = 0; i < getNumWords(); ++i) {
      uint64_t value = pVal[i];
      for (uint32_t j = 0; j < nibbles; ++j) {
        result.insert(0, digits[ value & mask ]);
        value >>= shift;
      }
    }
    return result;
  }

  APInt tmp(*this);
  APInt divisor(4, radix);
  APInt zero(tmp.getBitWidth(), 0);
  size_t insert_at = 0;
  if (wantSigned && tmp[BitWidth-1]) {
    // They want to print the signed version and it is a negative value
    // Flip the bits and add one to turn it into the equivalent positive
    // value and put a '-' in the result.
    tmp.flip();
    tmp++;
    result = "-";
    insert_at = 1;
  }
  if (tmp == APInt(tmp.getBitWidth(), 0))
    result = "0";
  else while (tmp.ne(zero)) {
    APInt APdigit(1,0);
    APInt tmp2(tmp.getBitWidth(), 0);
    divide(tmp, tmp.getNumWords(), divisor, divisor.getNumWords(), &tmp2, 
           &APdigit);
    uint32_t digit = APdigit.getZExtValue();
    assert(digit < radix && "divide failed");
    result.insert(insert_at,digits[digit]);
    tmp = tmp2;
  }

  return result;
}

#ifndef NDEBUG
void APInt::dump() const
{
  cerr << "APInt(" << BitWidth << ")=" << std::setbase(16);
  if (isSingleWord())
    cerr << VAL;
  else for (unsigned i = getNumWords(); i > 0; i--) {
    cerr << pVal[i-1] << " ";
  }
  cerr << " U(" << this->toString(10) << ") S(" << this->toStringSigned(10)
       << ")\n" << std::setbase(10);
}
#endif

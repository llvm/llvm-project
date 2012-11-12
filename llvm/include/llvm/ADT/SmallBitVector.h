//===- llvm/ADT/SmallBitVector.h - 'Normally small' bit vectors -*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file implements the SmallBitVector class.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_ADT_SMALLBITVECTOR_H
#define LLVM_ADT_SMALLBITVECTOR_H

#include "llvm/ADT/BitVector.h"
#include "llvm/Support/Compiler.h"
#include "llvm/Support/MathExtras.h"
#include <cassert>

namespace llvm {

/// SmallBitVector - This is a 'bitvector' (really, a variable-sized bit array),
/// optimized for the case when the array is small.  It contains one
/// pointer-sized field, which is directly used as a plain collection of bits
/// when possible, or as a pointer to a larger heap-allocated array when
/// necessary.  This allows normal "small" cases to be fast without losing
/// generality for large inputs.
///
class SmallBitVector {
  // TODO: In "large" mode, a pointer to a BitVector is used, leading to an
  // unnecessary level of indirection. It would be more efficient to use a
  // pointer to memory containing size, allocation size, and the array of bits.
  uintptr_t X;

  enum {
    // The number of bits in this class.
    NumBaseBits = sizeof(uintptr_t) * CHAR_BIT,

    // One bit is used to discriminate between small and large mode. The
    // remaining bits are used for the small-mode representation.
    SmallNumRawBits = NumBaseBits - 1,

    // A few more bits are used to store the size of the bit set in small mode.
    // Theoretically this is a ceil-log2. These bits are encoded in the most
    // significant bits of the raw bits.
    SmallNumSizeBits = (NumBaseBits == 32 ? 5 :
                        NumBaseBits == 64 ? 6 :
                        SmallNumRawBits),

    // The remaining bits are used to store the actual set in small mode.
    SmallNumDataBits = SmallNumRawBits - SmallNumSizeBits
  };

public:
  // Encapsulation of a single bit.
  class reference {
    SmallBitVector &TheVector;
    unsigned BitPos;

  public:
    reference(SmallBitVector &b, unsigned Idx) : TheVector(b), BitPos(Idx) {}

    reference& operator=(reference t) {
      *this = bool(t);
      return *this;
    }

    reference& operator=(bool t) {
      if (t)
        TheVector.set(BitPos);
      else
        TheVector.reset(BitPos);
      return *this;
    }

    operator bool() const {
      return const_cast<const SmallBitVector &>(TheVector).operator[](BitPos);
    }
  };

private:
  bool isSmall() const {
    return X & uintptr_t(1);
  }

  BitVector *getPointer() const {
    assert(!isSmall());
    return reinterpret_cast<BitVector *>(X);
  }

  void switchToSmall(uintptr_t NewSmallBits, size_t NewSize) {
    X = 1;
    setSmallSize(NewSize);
    setSmallBits(NewSmallBits);
  }

  void switchToLarge(BitVector *BV) {
    X = reinterpret_cast<uintptr_t>(BV);
    assert(!isSmall() && "Tried to use an unaligned pointer");
  }

  // Return all the bits used for the "small" representation; this includes
  // bits for the size as well as the element bits.
  uintptr_t getSmallRawBits() const {
    assert(isSmall());
    return X >> 1;
  }

  void setSmallRawBits(uintptr_t NewRawBits) {
    assert(isSmall());
    X = (NewRawBits << 1) | uintptr_t(1);
  }

  // Return the size.
  size_t getSmallSize() const {
    return getSmallRawBits() >> SmallNumDataBits;
  }

  void setSmallSize(size_t Size) {
    setSmallRawBits(getSmallBits() | (Size << SmallNumDataBits));
  }

  // Return the element bits.
  uintptr_t getSmallBits() const {
    return getSmallRawBits() & ~(~uintptr_t(0) << getSmallSize());
  }

  void setSmallBits(uintptr_t NewBits) {
    setSmallRawBits((NewBits & ~(~uintptr_t(0) << getSmallSize())) |
                    (getSmallSize() << SmallNumDataBits));
  }

public:
  /// SmallBitVector default ctor - Creates an empty bitvector.
  SmallBitVector() : X(1) {}

  /// SmallBitVector ctor - Creates a bitvector of specified number of bits. All
  /// bits are initialized to the specified value.
  explicit SmallBitVector(unsigned s, bool t = false) {
    if (s <= SmallNumDataBits)
      switchToSmall(t ? ~uintptr_t(0) : 0, s);
    else
      switchToLarge(new BitVector(s, t));
  }

  /// SmallBitVector copy ctor.
  SmallBitVector(const SmallBitVector &RHS) {
    if (RHS.isSmall())
      X = RHS.X;
    else
      switchToLarge(new BitVector(*RHS.getPointer()));
  }

#if LLVM_USE_RVALUE_REFERENCES
  SmallBitVector(SmallBitVector &&RHS) : X(RHS.X) {
    RHS.X = 1;
  }
#endif

  ~SmallBitVector() {
    if (!isSmall())
      delete getPointer();
  }

  /// empty - Tests whether there are no bits in this bitvector.
  bool empty() const {
    return isSmall() ? getSmallSize() == 0 : getPointer()->empty();
  }

  /// size - Returns the number of bits in this bitvector.
  size_t size() const {
    return isSmall() ? getSmallSize() : getPointer()->size();
  }

  /// count - Returns the number of bits which are set.
  unsigned count() const {
    if (isSmall()) {
      uintptr_t Bits = getSmallBits();
      if (sizeof(uintptr_t) * CHAR_BIT == 32)
        return CountPopulation_32(Bits);
      if (sizeof(uintptr_t) * CHAR_BIT == 64)
        return CountPopulation_64(Bits);
      llvm_unreachable("Unsupported!");
    }
    return getPointer()->count();
  }

  /// any - Returns true if any bit is set.
  bool any() const {
    if (isSmall())
      return getSmallBits() != 0;
    return getPointer()->any();
  }

  /// all - Returns true if all bits are set.
  bool all() const {
    if (isSmall())
      return getSmallBits() == (uintptr_t(1) << getSmallSize()) - 1;
    return getPointer()->all();
  }

  /// none - Returns true if none of the bits are set.
  bool none() const {
    if (isSmall())
      return getSmallBits() == 0;
    return getPointer()->none();
  }

  /// find_first - Returns the index of the first set bit, -1 if none
  /// of the bits are set.
  int find_first() const {
    if (isSmall()) {
      uintptr_t Bits = getSmallBits();
      if (Bits == 0)
        return -1;
      if (sizeof(uintptr_t) * CHAR_BIT == 32)
        return CountTrailingZeros_32(Bits);
      if (sizeof(uintptr_t) * CHAR_BIT == 64)
        return CountTrailingZeros_64(Bits);
      llvm_unreachable("Unsupported!");
    }
    return getPointer()->find_first();
  }

  /// find_next - Returns the index of the next set bit following the
  /// "Prev" bit. Returns -1 if the next set bit is not found.
  int find_next(unsigned Prev) const {
    if (isSmall()) {
      uintptr_t Bits = getSmallBits();
      // Mask off previous bits.
      Bits &= ~uintptr_t(0) << (Prev + 1);
      if (Bits == 0 || Prev + 1 >= getSmallSize())
        return -1;
      if (sizeof(uintptr_t) * CHAR_BIT == 32)
        return CountTrailingZeros_32(Bits);
      if (sizeof(uintptr_t) * CHAR_BIT == 64)
        return CountTrailingZeros_64(Bits);
      llvm_unreachable("Unsupported!");
    }
    return getPointer()->find_next(Prev);
  }

  /// clear - Clear all bits.
  void clear() {
    if (!isSmall())
      delete getPointer();
    switchToSmall(0, 0);
  }

  /// resize - Grow or shrink the bitvector.
  void resize(unsigned N, bool t = false) {
    if (!isSmall()) {
      getPointer()->resize(N, t);
    } else if (SmallNumDataBits >= N) {
      uintptr_t NewBits = t ? ~uintptr_t(0) << getSmallSize() : 0;
      setSmallSize(N);
      setSmallBits(NewBits | getSmallBits());
    } else {
      BitVector *BV = new BitVector(N, t);
      uintptr_t OldBits = getSmallBits();
      for (size_t i = 0, e = getSmallSize(); i != e; ++i)
        (*BV)[i] = (OldBits >> i) & 1;
      switchToLarge(BV);
    }
  }

  void reserve(unsigned N) {
    if (isSmall()) {
      if (N > SmallNumDataBits) {
        uintptr_t OldBits = getSmallRawBits();
        size_t SmallSize = getSmallSize();
        BitVector *BV = new BitVector(SmallSize);
        for (size_t i = 0; i < SmallSize; ++i)
          if ((OldBits >> i) & 1)
            BV->set(i);
        BV->reserve(N);
        switchToLarge(BV);
      }
    } else {
      getPointer()->reserve(N);
    }
  }

  // Set, reset, flip
  SmallBitVector &set() {
    if (isSmall())
      setSmallBits(~uintptr_t(0));
    else
      getPointer()->set();
    return *this;
  }

  SmallBitVector &set(unsigned Idx) {
    if (isSmall())
      setSmallBits(getSmallBits() | (uintptr_t(1) << Idx));
    else
      getPointer()->set(Idx);
    return *this;
  }

  /// set - Efficiently set a range of bits in [I, E)
  SmallBitVector &set(unsigned I, unsigned E) {
    assert(I <= E && "Attempted to set backwards range!");
    assert(E <= size() && "Attempted to set out-of-bounds range!");
    if (I == E) return *this;
    if (isSmall()) {
      uintptr_t EMask = ((uintptr_t)1) << E;
      uintptr_t IMask = ((uintptr_t)1) << I;
      uintptr_t Mask = EMask - IMask;
      setSmallBits(getSmallBits() | Mask);
    } else
      getPointer()->set(I, E);
    return *this;
  }

  SmallBitVector &reset() {
    if (isSmall())
      setSmallBits(0);
    else
      getPointer()->reset();
    return *this;
  }

  SmallBitVector &reset(unsigned Idx) {
    if (isSmall())
      setSmallBits(getSmallBits() & ~(uintptr_t(1) << Idx));
    else
      getPointer()->reset(Idx);
    return *this;
  }

  /// reset - Efficiently reset a range of bits in [I, E)
  SmallBitVector &reset(unsigned I, unsigned E) {
    assert(I <= E && "Attempted to reset backwards range!");
    assert(E <= size() && "Attempted to reset out-of-bounds range!");
    if (I == E) return *this;
    if (isSmall()) {
      uintptr_t EMask = ((uintptr_t)1) << E;
      uintptr_t IMask = ((uintptr_t)1) << I;
      uintptr_t Mask = EMask - IMask;
      setSmallBits(getSmallBits() & ~Mask);
    } else
      getPointer()->reset(I, E);
    return *this;
  }

  SmallBitVector &flip() {
    if (isSmall())
      setSmallBits(~getSmallBits());
    else
      getPointer()->flip();
    return *this;
  }

  SmallBitVector &flip(unsigned Idx) {
    if (isSmall())
      setSmallBits(getSmallBits() ^ (uintptr_t(1) << Idx));
    else
      getPointer()->flip(Idx);
    return *this;
  }

  // No argument flip.
  SmallBitVector operator~() const {
    return SmallBitVector(*this).flip();
  }

  // Indexing.
  reference operator[](unsigned Idx) {
    assert(Idx < size() && "Out-of-bounds Bit access.");
    return reference(*this, Idx);
  }

  bool operator[](unsigned Idx) const {
    assert(Idx < size() && "Out-of-bounds Bit access.");
    if (isSmall())
      return ((getSmallBits() >> Idx) & 1) != 0;
    return getPointer()->operator[](Idx);
  }

  bool test(unsigned Idx) const {
    return (*this)[Idx];
  }

  /// Test if any common bits are set.
  bool anyCommon(const SmallBitVector &RHS) const {
    if (isSmall() && RHS.isSmall())
      return (getSmallBits() & RHS.getSmallBits()) != 0;
    if (!isSmall() && !RHS.isSmall())
      return getPointer()->anyCommon(*RHS.getPointer());

    for (unsigned i = 0, e = std::min(size(), RHS.size()); i != e; ++i)
      if (test(i) && RHS.test(i))
        return true;
    return false;
  }

  // Comparison operators.
  bool operator==(const SmallBitVector &RHS) const {
    if (size() != RHS.size())
      return false;
    if (isSmall())
      return getSmallBits() == RHS.getSmallBits();
    else
      return *getPointer() == *RHS.getPointer();
  }

  bool operator!=(const SmallBitVector &RHS) const {
    return !(*this == RHS);
  }

  // Intersection, union, disjoint union.
  SmallBitVector &operator&=(const SmallBitVector &RHS) {
    resize(std::max(size(), RHS.size()));
    if (isSmall())
      setSmallBits(getSmallBits() & RHS.getSmallBits());
    else if (!RHS.isSmall())
      getPointer()->operator&=(*RHS.getPointer());
    else {
      SmallBitVector Copy = RHS;
      Copy.resize(size());
      getPointer()->operator&=(*Copy.getPointer());
    }
    return *this;
  }

  SmallBitVector &operator|=(const SmallBitVector &RHS) {
    resize(std::max(size(), RHS.size()));
    if (isSmall())
      setSmallBits(getSmallBits() | RHS.getSmallBits());
    else if (!RHS.isSmall())
      getPointer()->operator|=(*RHS.getPointer());
    else {
      SmallBitVector Copy = RHS;
      Copy.resize(size());
      getPointer()->operator|=(*Copy.getPointer());
    }
    return *this;
  }

  SmallBitVector &operator^=(const SmallBitVector &RHS) {
    resize(std::max(size(), RHS.size()));
    if (isSmall())
      setSmallBits(getSmallBits() ^ RHS.getSmallBits());
    else if (!RHS.isSmall())
      getPointer()->operator^=(*RHS.getPointer());
    else {
      SmallBitVector Copy = RHS;
      Copy.resize(size());
      getPointer()->operator^=(*Copy.getPointer());
    }
    return *this;
  }

  // Assignment operator.
  const SmallBitVector &operator=(const SmallBitVector &RHS) {
    if (isSmall()) {
      if (RHS.isSmall())
        X = RHS.X;
      else
        switchToLarge(new BitVector(*RHS.getPointer()));
    } else {
      if (!RHS.isSmall())
        *getPointer() = *RHS.getPointer();
      else {
        delete getPointer();
        X = RHS.X;
      }
    }
    return *this;
  }

#if LLVM_USE_RVALUE_REFERENCES
  const SmallBitVector &operator=(SmallBitVector &&RHS) {
    if (this != &RHS) {
      clear();
      swap(RHS);
    }
    return *this;
  }
#endif

  void swap(SmallBitVector &RHS) {
    std::swap(X, RHS.X);
  }

  /// setBitsInMask - Add '1' bits from Mask to this vector. Don't resize.
  /// This computes "*this |= Mask".
  void setBitsInMask(const uint32_t *Mask, unsigned MaskWords = ~0u) {
    if (isSmall())
      applyMask<true, false>(Mask, MaskWords);
    else
      getPointer()->setBitsInMask(Mask, MaskWords);
  }

  /// clearBitsInMask - Clear any bits in this vector that are set in Mask.
  /// Don't resize. This computes "*this &= ~Mask".
  void clearBitsInMask(const uint32_t *Mask, unsigned MaskWords = ~0u) {
    if (isSmall())
      applyMask<false, false>(Mask, MaskWords);
    else
      getPointer()->clearBitsInMask(Mask, MaskWords);
  }

  /// setBitsNotInMask - Add a bit to this vector for every '0' bit in Mask.
  /// Don't resize.  This computes "*this |= ~Mask".
  void setBitsNotInMask(const uint32_t *Mask, unsigned MaskWords = ~0u) {
    if (isSmall())
      applyMask<true, true>(Mask, MaskWords);
    else
      getPointer()->setBitsNotInMask(Mask, MaskWords);
  }

  /// clearBitsNotInMask - Clear a bit in this vector for every '0' bit in Mask.
  /// Don't resize.  This computes "*this &= Mask".
  void clearBitsNotInMask(const uint32_t *Mask, unsigned MaskWords = ~0u) {
    if (isSmall())
      applyMask<false, true>(Mask, MaskWords);
    else
      getPointer()->clearBitsNotInMask(Mask, MaskWords);
  }

private:
  template<bool AddBits, bool InvertMask>
  void applyMask(const uint32_t *Mask, unsigned MaskWords) {
    assert((NumBaseBits == 64 || NumBaseBits == 32) && "Unsupported word size");
    if (NumBaseBits == 64 && MaskWords >= 2) {
      uint64_t M = Mask[0] | (uint64_t(Mask[1]) << 32);
      if (InvertMask) M = ~M;
      if (AddBits) setSmallBits(getSmallBits() | M);
      else         setSmallBits(getSmallBits() & ~M);
    } else {
      uint32_t M = Mask[0];
      if (InvertMask) M = ~M;
      if (AddBits) setSmallBits(getSmallBits() | M);
      else         setSmallBits(getSmallBits() & ~M);
    }
  }
};

inline SmallBitVector
operator&(const SmallBitVector &LHS, const SmallBitVector &RHS) {
  SmallBitVector Result(LHS);
  Result &= RHS;
  return Result;
}

inline SmallBitVector
operator|(const SmallBitVector &LHS, const SmallBitVector &RHS) {
  SmallBitVector Result(LHS);
  Result |= RHS;
  return Result;
}

inline SmallBitVector
operator^(const SmallBitVector &LHS, const SmallBitVector &RHS) {
  SmallBitVector Result(LHS);
  Result ^= RHS;
  return Result;
}

} // End llvm namespace

namespace std {
  /// Implement std::swap in terms of BitVector swap.
  inline void
  swap(llvm::SmallBitVector &LHS, llvm::SmallBitVector &RHS) {
    LHS.swap(RHS);
  }
}

#endif

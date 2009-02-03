//===- llvm/ADT/BitVector.h - Bit vectors -----------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file implements the BitVector class.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_ADT_BITVECTOR_H
#define LLVM_ADT_BITVECTOR_H

#include "llvm/Support/MathExtras.h"
#include <algorithm>
#include <cassert>
#include <cstring>

namespace llvm {

class BitVector {
  typedef unsigned long BitWord;

  enum { BITWORD_SIZE = (unsigned)sizeof(BitWord) * 8 };

  BitWord  *Bits;        // Actual bits.
  unsigned Size;         // Size of bitvector in bits.
  unsigned Capacity;     // Size of allocated memory in BitWord.

public:
  // Encapsulation of a single bit.
  class reference {
    friend class BitVector;

    BitWord *WordRef;
    unsigned BitPos;

    reference();  // Undefined

  public:
    reference(BitVector &b, unsigned Idx) {
      WordRef = &b.Bits[Idx / BITWORD_SIZE];
      BitPos = Idx % BITWORD_SIZE;
    }

    ~reference() {}

    reference& operator=(bool t) {
      if (t)
        *WordRef |= 1L << BitPos;
      else
        *WordRef &= ~(1L << BitPos);
      return *this;
    }

    operator bool() const {
      return ((*WordRef) & (1L << BitPos)) ? true : false;
    }
  };


  /// BitVector default ctor - Creates an empty bitvector.
  BitVector() : Size(0), Capacity(0) {
    Bits = 0;
  }

  /// BitVector ctor - Creates a bitvector of specified number of bits. All
  /// bits are initialized to the specified value.
  explicit BitVector(unsigned s, bool t = false) : Size(s) {
    Capacity = NumBitWords(s);
    Bits = new BitWord[Capacity];
    init_words(Bits, Capacity, t);
    if (t)
      clear_unused_bits();
  }

  /// BitVector copy ctor.
  BitVector(const BitVector &RHS) : Size(RHS.size()) {
    if (Size == 0) {
      Bits = 0;
      Capacity = 0;
      return;
    }

    Capacity = NumBitWords(RHS.size());
    Bits = new BitWord[Capacity];
    std::copy(RHS.Bits, &RHS.Bits[Capacity], Bits);
  }

  ~BitVector() {
    delete[] Bits;
  }

  /// size - Returns the number of bits in this bitvector.
  unsigned size() const { return Size; }

  /// count - Returns the number of bits which are set.
  unsigned count() const {
    unsigned NumBits = 0;
    for (unsigned i = 0; i < NumBitWords(size()); ++i)
      if (sizeof(BitWord) == 4)
        NumBits += CountPopulation_32((uint32_t)Bits[i]);
      else if (sizeof(BitWord) == 8)
        NumBits += CountPopulation_64(Bits[i]);
      else
        assert(0 && "Unsupported!");
    return NumBits;
  }

  /// any - Returns true if any bit is set.
  bool any() const {
    for (unsigned i = 0; i < NumBitWords(size()); ++i)
      if (Bits[i] != 0)
        return true;
    return false;
  }

  /// none - Returns true if none of the bits are set.
  bool none() const {
    return !any();
  }

  /// find_first - Returns the index of the first set bit, -1 if none
  /// of the bits are set.
  int find_first() const {
    for (unsigned i = 0; i < NumBitWords(size()); ++i)
      if (Bits[i] != 0) {
        if (sizeof(BitWord) == 4)
          return i * BITWORD_SIZE + CountTrailingZeros_32((uint32_t)Bits[i]);
        else if (sizeof(BitWord) == 8)
          return i * BITWORD_SIZE + CountTrailingZeros_64(Bits[i]);
        else
          assert(0 && "Unsupported!");
      }
    return -1;
  }

  /// find_next - Returns the index of the next set bit following the
  /// "Prev" bit. Returns -1 if the next set bit is not found.
  int find_next(unsigned Prev) const {
    ++Prev;
    if (Prev >= Size)
      return -1;

    unsigned WordPos = Prev / BITWORD_SIZE;
    unsigned BitPos = Prev % BITWORD_SIZE;
    BitWord Copy = Bits[WordPos];
    // Mask off previous bits.
    Copy &= ~0L << BitPos;

    if (Copy != 0) {
      if (sizeof(BitWord) == 4)
        return WordPos * BITWORD_SIZE + CountTrailingZeros_32((uint32_t)Copy);
      else if (sizeof(BitWord) == 8)
        return WordPos * BITWORD_SIZE + CountTrailingZeros_64(Copy);
      else
        assert(0 && "Unsupported!");
    }

    // Check subsequent words.
    for (unsigned i = WordPos+1; i < NumBitWords(size()); ++i)
      if (Bits[i] != 0) {
        if (sizeof(BitWord) == 4)
          return i * BITWORD_SIZE + CountTrailingZeros_32((uint32_t)Bits[i]);
        else if (sizeof(BitWord) == 8)
          return i * BITWORD_SIZE + CountTrailingZeros_64(Bits[i]);
        else
          assert(0 && "Unsupported!");
      }
    return -1;
  }

  /// clear - Clear all bits.
  void clear() {
    Size = 0;
  }

  /// resize - Grow or shrink the bitvector.
  void resize(unsigned N, bool t = false) {
    if (N > Capacity * BITWORD_SIZE) {
      unsigned OldCapacity = Capacity;
      grow(N);
      init_words(&Bits[OldCapacity], (Capacity-OldCapacity), t);
    }

    // Set any old unused bits that are now included in the BitVector. This
    // may set bits that are not included in the new vector, but we will clear
    // them back out below.
    if (N > Size)
      set_unused_bits(t);

    // Update the size, and clear out any bits that are now unused
    unsigned OldSize = Size;
    Size = N;
    if (t || N < OldSize)
      clear_unused_bits();
  }

  void reserve(unsigned N) {
    if (N > Capacity * BITWORD_SIZE)
      grow(N);
  }

  // Set, reset, flip
  BitVector &set() {
    init_words(Bits, Capacity, true);
    clear_unused_bits();
    return *this;
  }

  BitVector &set(unsigned Idx) {
    Bits[Idx / BITWORD_SIZE] |= 1L << (Idx % BITWORD_SIZE);
    return *this;
  }

  BitVector &reset() {
    init_words(Bits, Capacity, false);
    return *this;
  }

  BitVector &reset(unsigned Idx) {
    Bits[Idx / BITWORD_SIZE] &= ~(1L << (Idx % BITWORD_SIZE));
    return *this;
  }

  BitVector &flip() {
    for (unsigned i = 0; i < NumBitWords(size()); ++i)
      Bits[i] = ~Bits[i];
    clear_unused_bits();
    return *this;
  }

  BitVector &flip(unsigned Idx) {
    Bits[Idx / BITWORD_SIZE] ^= 1L << (Idx % BITWORD_SIZE);
    return *this;
  }

  // No argument flip.
  BitVector operator~() const {
    return BitVector(*this).flip();
  }

  // Indexing.
  reference operator[](unsigned Idx) {
    assert (Idx < Size && "Out-of-bounds Bit access.");
    return reference(*this, Idx);
  }

  bool operator[](unsigned Idx) const {
    assert (Idx < Size && "Out-of-bounds Bit access.");
    BitWord Mask = 1L << (Idx % BITWORD_SIZE);
    return (Bits[Idx / BITWORD_SIZE] & Mask) != 0;
  }

  bool test(unsigned Idx) const {
    return (*this)[Idx];
  }

  // Comparison operators.
  bool operator==(const BitVector &RHS) const {
    unsigned ThisWords = NumBitWords(size());
    unsigned RHSWords  = NumBitWords(RHS.size());
    unsigned i;
    for (i = 0; i != std::min(ThisWords, RHSWords); ++i)
      if (Bits[i] != RHS.Bits[i])
        return false;

    // Verify that any extra words are all zeros.
    if (i != ThisWords) {
      for (; i != ThisWords; ++i)
        if (Bits[i])
          return false;
    } else if (i != RHSWords) {
      for (; i != RHSWords; ++i)
        if (RHS.Bits[i])
          return false;
    }
    return true;
  }

  bool operator!=(const BitVector &RHS) const {
    return !(*this == RHS);
  }

  // Intersection, union, disjoint union.
  BitVector &operator&=(const BitVector &RHS) {
    unsigned ThisWords = NumBitWords(size());
    unsigned RHSWords  = NumBitWords(RHS.size());
    unsigned i;
    for (i = 0; i != std::min(ThisWords, RHSWords); ++i)
      Bits[i] &= RHS.Bits[i];

    // Any bits that are just in this bitvector become zero, because they aren't
    // in the RHS bit vector.  Any words only in RHS are ignored because they
    // are already zero in the LHS.
    for (; i != ThisWords; ++i)
      Bits[i] = 0;

    return *this;
  }

  BitVector &operator|=(const BitVector &RHS) {
    assert(Size == RHS.Size && "Illegal operation!");
    for (unsigned i = 0; i < NumBitWords(size()); ++i)
      Bits[i] |= RHS.Bits[i];
    return *this;
  }

  BitVector &operator^=(const BitVector &RHS) {
    assert(Size == RHS.Size && "Illegal operation!");
    for (unsigned i = 0; i < NumBitWords(size()); ++i)
      Bits[i] ^= RHS.Bits[i];
    return *this;
  }

  // Assignment operator.
  const BitVector &operator=(const BitVector &RHS) {
    if (this == &RHS) return *this;

    Size = RHS.size();
    unsigned RHSWords = NumBitWords(Size);
    if (Size <= Capacity * BITWORD_SIZE) {
      std::copy(RHS.Bits, &RHS.Bits[RHSWords], Bits);
      clear_unused_bits();
      return *this;
    }

    // Grow the bitvector to have enough elements.
    Capacity = RHSWords;
    BitWord *NewBits = new BitWord[Capacity];
    std::copy(RHS.Bits, &RHS.Bits[RHSWords], NewBits);

    // Destroy the old bits.
    delete[] Bits;
    Bits = NewBits;

    return *this;
  }

private:
  unsigned NumBitWords(unsigned S) const {
    return (S + BITWORD_SIZE-1) / BITWORD_SIZE;
  }

  // Set the unused bits in the high words.
  void set_unused_bits(bool t = true) {
    //  Set high words first.
    unsigned UsedWords = NumBitWords(Size);
    if (Capacity > UsedWords)
      init_words(&Bits[UsedWords], (Capacity-UsedWords), t);

    //  Then set any stray high bits of the last used word.
    unsigned ExtraBits = Size % BITWORD_SIZE;
    if (ExtraBits) {
      Bits[UsedWords-1] &= ~(~0L << ExtraBits);
      Bits[UsedWords-1] |= (0 - (BitWord)t) << ExtraBits;
    }
  }

  // Clear the unused bits in the high words.
  void clear_unused_bits() {
    set_unused_bits(false);
  }

  void grow(unsigned NewSize) {
    unsigned OldCapacity = Capacity;
    Capacity = NumBitWords(NewSize);
    BitWord *NewBits = new BitWord[Capacity];

    // Copy the old bits over.
    if (OldCapacity != 0)
      std::copy(Bits, &Bits[OldCapacity], NewBits);

    // Destroy the old bits.
    delete[] Bits;
    Bits = NewBits;

    clear_unused_bits();
  }

  void init_words(BitWord *B, unsigned NumWords, bool t) {
    memset(B, 0 - (int)t, NumWords*sizeof(BitWord));
  }
};

inline BitVector operator&(const BitVector &LHS, const BitVector &RHS) {
  BitVector Result(LHS);
  Result &= RHS;
  return Result;
}

inline BitVector operator|(const BitVector &LHS, const BitVector &RHS) {
  BitVector Result(LHS);
  Result |= RHS;
  return Result;
}

inline BitVector operator^(const BitVector &LHS, const BitVector &RHS) {
  BitVector Result(LHS);
  Result ^= RHS;
  return Result;
}

} // End llvm namespace
#endif

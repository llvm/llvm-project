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

#include "llvm/ADT/ArrayRef.h"
#include "llvm/Support/MathExtras.h"
#include <algorithm>
#include <cassert>
#include <climits>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <utility>

namespace llvm {

class BitVector {
  typedef unsigned long BitWord;

  enum { BITWORD_SIZE = (unsigned)sizeof(BitWord) * CHAR_BIT };

  static_assert(BITWORD_SIZE == 64 || BITWORD_SIZE == 32,
                "Unsupported word size");

  MutableArrayRef<BitWord> Bits; // Actual bits.
  unsigned Size;                 // Size of bitvector in bits.

public:
  typedef unsigned size_type;
  // Encapsulation of a single bit.
  class reference {
    friend class BitVector;

    BitWord *WordRef;
    unsigned BitPos;

  public:
    reference(BitVector &b, unsigned Idx) {
      WordRef = &b.Bits[Idx / BITWORD_SIZE];
      BitPos = Idx % BITWORD_SIZE;
    }

    reference() = delete;
    reference(const reference&) = default;

    reference &operator=(reference t) {
      *this = bool(t);
      return *this;
    }

    reference& operator=(bool t) {
      if (t)
        *WordRef |= BitWord(1) << BitPos;
      else
        *WordRef &= ~(BitWord(1) << BitPos);
      return *this;
    }

    operator bool() const {
      return ((*WordRef) & (BitWord(1) << BitPos)) != 0;
    }
  };


  /// BitVector default ctor - Creates an empty bitvector.
  BitVector() : Size(0) {}

  /// BitVector ctor - Creates a bitvector of specified number of bits. All
  /// bits are initialized to the specified value.
  explicit BitVector(unsigned s, bool t = false) : Size(s) {
    size_t Capacity = NumBitWords(s);
    Bits = allocate(Capacity);
    init_words(Bits, t);
    if (t)
      clear_unused_bits();
  }

  /// BitVector copy ctor.
  BitVector(const BitVector &RHS) : Size(RHS.size()) {
    if (Size == 0) {
      Bits = MutableArrayRef<BitWord>();
      return;
    }

    size_t Capacity = NumBitWords(RHS.size());
    Bits = allocate(Capacity);
    std::memcpy(Bits.data(), RHS.Bits.data(), Capacity * sizeof(BitWord));
  }

  BitVector(BitVector &&RHS) : Bits(RHS.Bits), Size(RHS.Size) {
    RHS.Bits = MutableArrayRef<BitWord>();
    RHS.Size = 0;
  }

  ~BitVector() { std::free(Bits.data()); }

  /// empty - Tests whether there are no bits in this bitvector.
  bool empty() const { return Size == 0; }

  /// size - Returns the number of bits in this bitvector.
  size_type size() const { return Size; }

  /// count - Returns the number of bits which are set.
  size_type count() const {
    unsigned NumBits = 0;
    for (unsigned i = 0; i < NumBitWords(size()); ++i)
      NumBits += countPopulation(Bits[i]);
    return NumBits;
  }

  /// any - Returns true if any bit is set.
  bool any() const {
    for (unsigned i = 0; i < NumBitWords(size()); ++i)
      if (Bits[i] != 0)
        return true;
    return false;
  }

  /// all - Returns true if all bits are set.
  bool all() const {
    for (unsigned i = 0; i < Size / BITWORD_SIZE; ++i)
      if (Bits[i] != ~0UL)
        return false;

    // If bits remain check that they are ones. The unused bits are always zero.
    if (unsigned Remainder = Size % BITWORD_SIZE)
      return Bits[Size / BITWORD_SIZE] == (1UL << Remainder) - 1;

    return true;
  }

  /// none - Returns true if none of the bits are set.
  bool none() const {
    return !any();
  }

  /// find_first - Returns the index of the first set bit, -1 if none
  /// of the bits are set.
  int find_first() const {
    for (unsigned i = 0; i < NumBitWords(size()); ++i)
      if (Bits[i] != 0)
        return i * BITWORD_SIZE + countTrailingZeros(Bits[i]);
    return -1;
  }

  /// find_last - Returns the index of the last set bit, -1 if none of the bits
  /// are set.
  int find_last() const {
    if (Size == 0)
      return -1;

    unsigned N = NumBitWords(size());
    assert(N > 0);

    unsigned i = N - 1;
    while (i > 0 && Bits[i] == BitWord(0))
      --i;

    return int((i + 1) * BITWORD_SIZE - countLeadingZeros(Bits[i])) - 1;
  }

  /// find_first_unset - Returns the index of the first unset bit, -1 if all
  /// of the bits are set.
  int find_first_unset() const {
    for (unsigned i = 0; i < NumBitWords(size()); ++i)
      if (Bits[i] != ~0UL) {
        unsigned Result = i * BITWORD_SIZE + countTrailingOnes(Bits[i]);
        return Result < size() ? Result : -1;
      }
    return -1;
  }

  /// find_last_unset - Returns the index of the last unset bit, -1 if all of
  /// the bits are set.
  int find_last_unset() const {
    if (Size == 0)
      return -1;

    const unsigned N = NumBitWords(size());
    assert(N > 0);

    unsigned i = N - 1;
    BitWord W = Bits[i];

    // The last word in the BitVector has some unused bits, so we need to set
    // them all to 1 first.  Set them all to 1 so they don't get treated as
    // valid unset bits.
    unsigned UnusedCount = BITWORD_SIZE - size() % BITWORD_SIZE;
    W |= maskLeadingOnes<BitWord>(UnusedCount);

    while (W == ~BitWord(0) && --i > 0)
      W = Bits[i];

    return int((i + 1) * BITWORD_SIZE - countLeadingOnes(W)) - 1;
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
    Copy &= maskTrailingZeros<BitWord>(BitPos);

    if (Copy != 0)
      return WordPos * BITWORD_SIZE + countTrailingZeros(Copy);

    // Check subsequent words.
    for (unsigned i = WordPos+1; i < NumBitWords(size()); ++i)
      if (Bits[i] != 0)
        return i * BITWORD_SIZE + countTrailingZeros(Bits[i]);
    return -1;
  }

  /// find_next_unset - Returns the index of the next unset bit following the
  /// "Prev" bit.  Returns -1 if all remaining bits are set.
  int find_next_unset(unsigned Prev) const {
    ++Prev;
    if (Prev >= Size)
      return -1;

    unsigned WordPos = Prev / BITWORD_SIZE;
    unsigned BitPos = Prev % BITWORD_SIZE;
    BitWord Copy = Bits[WordPos];
    // Mask in previous bits.
    BitWord Mask = (1 << BitPos) - 1;
    Copy |= Mask;

    if (Copy != ~0UL)
      return next_unset_in_word(WordPos, Copy);

    // Check subsequent words.
    for (unsigned i = WordPos + 1; i < NumBitWords(size()); ++i)
      if (Bits[i] != ~0UL)
        return next_unset_in_word(i, Bits[i]);
    return -1;
  }

  /// find_prev - Returns the index of the first set bit that precedes the
  /// the bit at \p PriorTo.  Returns -1 if all previous bits are unset.
  int find_prev(unsigned PriorTo) {
    if (PriorTo == 0)
      return -1;

    --PriorTo;

    unsigned WordPos = PriorTo / BITWORD_SIZE;
    unsigned BitPos = PriorTo % BITWORD_SIZE;
    BitWord Copy = Bits[WordPos];
    // Mask off next bits.
    Copy &= maskTrailingOnes<BitWord>(BitPos + 1);

    if (Copy != 0)
      return (WordPos + 1) * BITWORD_SIZE - countLeadingZeros(Copy) - 1;

    // Check previous words.
    for (unsigned i = 1; i <= WordPos; ++i) {
      unsigned Index = WordPos - i;
      if (Bits[Index] == 0)
        continue;
      return (Index + 1) * BITWORD_SIZE - countLeadingZeros(Bits[Index]) - 1;
    }
    return -1;
  }

  /// clear - Removes all bits from the bitvector. Does not change capacity.
  void clear() {
    Size = 0;
  }

  /// resize - Grow or shrink the bitvector.
  void resize(unsigned N, bool t = false) {
    if (N > getBitCapacity()) {
      unsigned OldCapacity = Bits.size();
      grow(N);
      init_words(Bits.drop_front(OldCapacity), t);
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
    if (N > getBitCapacity())
      grow(N);
  }

  // Set, reset, flip
  BitVector &set() {
    init_words(Bits, true);
    clear_unused_bits();
    return *this;
  }

  BitVector &set(unsigned Idx) {
    assert(Bits.data() && "Bits never allocated");
    Bits[Idx / BITWORD_SIZE] |= BitWord(1) << (Idx % BITWORD_SIZE);
    return *this;
  }

  /// set - Efficiently set a range of bits in [I, E)
  BitVector &set(unsigned I, unsigned E) {
    assert(I <= E && "Attempted to set backwards range!");
    assert(E <= size() && "Attempted to set out-of-bounds range!");

    if (I == E) return *this;

    if (I / BITWORD_SIZE == E / BITWORD_SIZE) {
      BitWord EMask = 1UL << (E % BITWORD_SIZE);
      BitWord IMask = 1UL << (I % BITWORD_SIZE);
      BitWord Mask = EMask - IMask;
      Bits[I / BITWORD_SIZE] |= Mask;
      return *this;
    }

    BitWord PrefixMask = ~0UL << (I % BITWORD_SIZE);
    Bits[I / BITWORD_SIZE] |= PrefixMask;
    I = alignTo(I, BITWORD_SIZE);

    for (; I + BITWORD_SIZE <= E; I += BITWORD_SIZE)
      Bits[I / BITWORD_SIZE] = ~0UL;

    BitWord PostfixMask = (1UL << (E % BITWORD_SIZE)) - 1;
    if (I < E)
      Bits[I / BITWORD_SIZE] |= PostfixMask;

    return *this;
  }

  BitVector &reset() {
    init_words(Bits, false);
    return *this;
  }

  BitVector &reset(unsigned Idx) {
    Bits[Idx / BITWORD_SIZE] &= ~(BitWord(1) << (Idx % BITWORD_SIZE));
    return *this;
  }

  /// reset - Efficiently reset a range of bits in [I, E)
  BitVector &reset(unsigned I, unsigned E) {
    assert(I <= E && "Attempted to reset backwards range!");
    assert(E <= size() && "Attempted to reset out-of-bounds range!");

    if (I == E) return *this;

    if (I / BITWORD_SIZE == E / BITWORD_SIZE) {
      BitWord EMask = 1UL << (E % BITWORD_SIZE);
      BitWord IMask = 1UL << (I % BITWORD_SIZE);
      BitWord Mask = EMask - IMask;
      Bits[I / BITWORD_SIZE] &= ~Mask;
      return *this;
    }

    BitWord PrefixMask = ~0UL << (I % BITWORD_SIZE);
    Bits[I / BITWORD_SIZE] &= ~PrefixMask;
    I = alignTo(I, BITWORD_SIZE);

    for (; I + BITWORD_SIZE <= E; I += BITWORD_SIZE)
      Bits[I / BITWORD_SIZE] = 0UL;

    BitWord PostfixMask = (1UL << (E % BITWORD_SIZE)) - 1;
    if (I < E)
      Bits[I / BITWORD_SIZE] &= ~PostfixMask;

    return *this;
  }

  BitVector &flip() {
    for (unsigned i = 0; i < NumBitWords(size()); ++i)
      Bits[i] = ~Bits[i];
    clear_unused_bits();
    return *this;
  }

  BitVector &flip(unsigned Idx) {
    Bits[Idx / BITWORD_SIZE] ^= BitWord(1) << (Idx % BITWORD_SIZE);
    return *this;
  }

  // Indexing.
  reference operator[](unsigned Idx) {
    assert (Idx < Size && "Out-of-bounds Bit access.");
    return reference(*this, Idx);
  }

  bool operator[](unsigned Idx) const {
    assert (Idx < Size && "Out-of-bounds Bit access.");
    BitWord Mask = BitWord(1) << (Idx % BITWORD_SIZE);
    return (Bits[Idx / BITWORD_SIZE] & Mask) != 0;
  }

  bool test(unsigned Idx) const {
    return (*this)[Idx];
  }

  /// Test if any common bits are set.
  bool anyCommon(const BitVector &RHS) const {
    unsigned ThisWords = NumBitWords(size());
    unsigned RHSWords  = NumBitWords(RHS.size());
    for (unsigned i = 0, e = std::min(ThisWords, RHSWords); i != e; ++i)
      if (Bits[i] & RHS.Bits[i])
        return true;
    return false;
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

  /// Intersection, union, disjoint union.
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

  /// reset - Reset bits that are set in RHS. Same as *this &= ~RHS.
  BitVector &reset(const BitVector &RHS) {
    unsigned ThisWords = NumBitWords(size());
    unsigned RHSWords  = NumBitWords(RHS.size());
    unsigned i;
    for (i = 0; i != std::min(ThisWords, RHSWords); ++i)
      Bits[i] &= ~RHS.Bits[i];
    return *this;
  }

  /// test - Check if (This - RHS) is zero.
  /// This is the same as reset(RHS) and any().
  bool test(const BitVector &RHS) const {
    unsigned ThisWords = NumBitWords(size());
    unsigned RHSWords  = NumBitWords(RHS.size());
    unsigned i;
    for (i = 0; i != std::min(ThisWords, RHSWords); ++i)
      if ((Bits[i] & ~RHS.Bits[i]) != 0)
        return true;

    for (; i != ThisWords ; ++i)
      if (Bits[i] != 0)
        return true;

    return false;
  }

  BitVector &operator|=(const BitVector &RHS) {
    if (size() < RHS.size())
      resize(RHS.size());
    for (size_t i = 0, e = NumBitWords(RHS.size()); i != e; ++i)
      Bits[i] |= RHS.Bits[i];
    return *this;
  }

  BitVector &operator^=(const BitVector &RHS) {
    if (size() < RHS.size())
      resize(RHS.size());
    for (size_t i = 0, e = NumBitWords(RHS.size()); i != e; ++i)
      Bits[i] ^= RHS.Bits[i];
    return *this;
  }

  BitVector &operator>>=(unsigned N) {
    assert(N <= Size);
    if (LLVM_UNLIKELY(empty() || N == 0))
      return *this;

    unsigned NumWords = NumBitWords(Size);
    assert(NumWords >= 1);

    wordShr(N / BITWORD_SIZE);

    unsigned BitDistance = N % BITWORD_SIZE;
    if (BitDistance == 0)
      return *this;

    // When the shift size is not a multiple of the word size, then we have
    // a tricky situation where each word in succession needs to extract some
    // of the bits from the next word and or them into this word while
    // shifting this word to make room for the new bits.  This has to be done
    // for every word in the array.

    // Since we're shifting each word right, some bits will fall off the end
    // of each word to the right, and empty space will be created on the left.
    // The final word in the array will lose bits permanently, so starting at
    // the beginning, work forwards shifting each word to the right, and
    // OR'ing in the bits from the end of the next word to the beginning of
    // the current word.

    // Example:
    //   Starting with {0xAABBCCDD, 0xEEFF0011, 0x22334455} and shifting right
    //   by 4 bits.
    // Step 1: Word[0] >>= 4           ; 0x0ABBCCDD
    // Step 2: Word[0] |= 0x10000000   ; 0x1ABBCCDD
    // Step 3: Word[1] >>= 4           ; 0x0EEFF001
    // Step 4: Word[1] |= 0x50000000   ; 0x5EEFF001
    // Step 5: Word[2] >>= 4           ; 0x02334455
    // Result: { 0x1ABBCCDD, 0x5EEFF001, 0x02334455 }
    const BitWord Mask = maskTrailingOnes<BitWord>(BitDistance);
    const unsigned LSH = BITWORD_SIZE - BitDistance;

    for (unsigned I = 0; I < NumWords - 1; ++I) {
      Bits[I] >>= BitDistance;
      Bits[I] |= (Bits[I + 1] & Mask) << LSH;
    }

    Bits[NumWords - 1] >>= BitDistance;

    return *this;
  }

  BitVector &operator<<=(unsigned N) {
    assert(N <= Size);
    if (LLVM_UNLIKELY(empty() || N == 0))
      return *this;

    unsigned NumWords = NumBitWords(Size);
    assert(NumWords >= 1);

    wordShl(N / BITWORD_SIZE);

    unsigned BitDistance = N % BITWORD_SIZE;
    if (BitDistance == 0)
      return *this;

    // When the shift size is not a multiple of the word size, then we have
    // a tricky situation where each word in succession needs to extract some
    // of the bits from the previous word and or them into this word while
    // shifting this word to make room for the new bits.  This has to be done
    // for every word in the array.  This is similar to the algorithm outlined
    // in operator>>=, but backwards.

    // Since we're shifting each word left, some bits will fall off the end
    // of each word to the left, and empty space will be created on the right.
    // The first word in the array will lose bits permanently, so starting at
    // the end, work backwards shifting each word to the left, and OR'ing
    // in the bits from the end of the next word to the beginning of the
    // current word.

    // Example:
    //   Starting with {0xAABBCCDD, 0xEEFF0011, 0x22334455} and shifting left
    //   by 4 bits.
    // Step 1: Word[2] <<= 4           ; 0x23344550
    // Step 2: Word[2] |= 0x0000000E   ; 0x2334455E
    // Step 3: Word[1] <<= 4           ; 0xEFF00110
    // Step 4: Word[1] |= 0x0000000A   ; 0xEFF0011A
    // Step 5: Word[0] <<= 4           ; 0xABBCCDD0
    // Result: { 0xABBCCDD0, 0xEFF0011A, 0x2334455E }
    const BitWord Mask = maskLeadingOnes<BitWord>(BitDistance);
    const unsigned RSH = BITWORD_SIZE - BitDistance;

    for (int I = NumWords - 1; I > 0; --I) {
      Bits[I] <<= BitDistance;
      Bits[I] |= (Bits[I - 1] & Mask) >> RSH;
    }
    Bits[0] <<= BitDistance;
    clear_unused_bits();

    return *this;
  }

  // Assignment operator.
  const BitVector &operator=(const BitVector &RHS) {
    if (this == &RHS) return *this;

    Size = RHS.size();
    unsigned RHSWords = NumBitWords(Size);
    if (Size <= getBitCapacity()) {
      if (Size)
        std::memcpy(Bits.data(), RHS.Bits.data(), RHSWords * sizeof(BitWord));
      clear_unused_bits();
      return *this;
    }

    // Grow the bitvector to have enough elements.
    unsigned NewCapacity = RHSWords;
    assert(NewCapacity > 0 && "negative capacity?");
    auto NewBits = allocate(NewCapacity);
    std::memcpy(NewBits.data(), RHS.Bits.data(), NewCapacity * sizeof(BitWord));

    // Destroy the old bits.
    std::free(Bits.data());
    Bits = NewBits;

    return *this;
  }

  const BitVector &operator=(BitVector &&RHS) {
    if (this == &RHS) return *this;

    std::free(Bits.data());
    Bits = RHS.Bits;
    Size = RHS.Size;

    RHS.Bits = MutableArrayRef<BitWord>();
    RHS.Size = 0;

    return *this;
  }

  void swap(BitVector &RHS) {
    std::swap(Bits, RHS.Bits);
    std::swap(Size, RHS.Size);
  }

  //===--------------------------------------------------------------------===//
  // Portable bit mask operations.
  //===--------------------------------------------------------------------===//
  //
  // These methods all operate on arrays of uint32_t, each holding 32 bits. The
  // fixed word size makes it easier to work with literal bit vector constants
  // in portable code.
  //
  // The LSB in each word is the lowest numbered bit.  The size of a portable
  // bit mask is always a whole multiple of 32 bits.  If no bit mask size is
  // given, the bit mask is assumed to cover the entire BitVector.

  /// setBitsInMask - Add '1' bits from Mask to this vector. Don't resize.
  /// This computes "*this |= Mask".
  void setBitsInMask(const uint32_t *Mask, unsigned MaskWords = ~0u) {
    applyMask<true, false>(Mask, MaskWords);
  }

  /// clearBitsInMask - Clear any bits in this vector that are set in Mask.
  /// Don't resize. This computes "*this &= ~Mask".
  void clearBitsInMask(const uint32_t *Mask, unsigned MaskWords = ~0u) {
    applyMask<false, false>(Mask, MaskWords);
  }

  /// setBitsNotInMask - Add a bit to this vector for every '0' bit in Mask.
  /// Don't resize.  This computes "*this |= ~Mask".
  void setBitsNotInMask(const uint32_t *Mask, unsigned MaskWords = ~0u) {
    applyMask<true, true>(Mask, MaskWords);
  }

  /// clearBitsNotInMask - Clear a bit in this vector for every '0' bit in Mask.
  /// Don't resize.  This computes "*this &= Mask".
  void clearBitsNotInMask(const uint32_t *Mask, unsigned MaskWords = ~0u) {
    applyMask<false, true>(Mask, MaskWords);
  }

private:
  /// \brief Perform a logical left shift of \p Count words by moving everything
  /// \p Count words to the right in memory.
  ///
  /// While confusing, words are stored from least significant at Bits[0] to
  /// most significant at Bits[NumWords-1].  A logical shift left, however,
  /// moves the current least significant bit to a higher logical index, and
  /// fills the previous least significant bits with 0.  Thus, we actually
  /// need to move the bytes of the memory to the right, not to the left.
  /// Example:
  ///   Words = [0xBBBBAAAA, 0xDDDDFFFF, 0x00000000, 0xDDDD0000]
  /// represents a BitVector where 0xBBBBAAAA contain the least significant
  /// bits.  So if we want to shift the BitVector left by 2 words, we need to
  /// turn this into 0x00000000 0x00000000 0xBBBBAAAA 0xDDDDFFFF by using a
  /// memmove which moves right, not left.
  void wordShl(uint32_t Count) {
    if (Count == 0)
      return;

    uint32_t NumWords = NumBitWords(Size);

    auto Src = Bits.take_front(NumWords).drop_back(Count);
    auto Dest = Bits.take_front(NumWords).drop_front(Count);

    // Since we always move Word-sized chunks of data with src and dest both
    // aligned to a word-boundary, we don't need to worry about endianness
    // here.
    std::memmove(Dest.begin(), Src.begin(), Dest.size() * sizeof(BitWord));
    std::memset(Bits.data(), 0, Count * sizeof(BitWord));
    clear_unused_bits();
  }

  /// \brief Perform a logical right shift of \p Count words by moving those
  /// words to the left in memory.  See wordShl for more information.
  ///
  void wordShr(uint32_t Count) {
    if (Count == 0)
      return;

    uint32_t NumWords = NumBitWords(Size);

    auto Src = Bits.take_front(NumWords).drop_front(Count);
    auto Dest = Bits.take_front(NumWords).drop_back(Count);
    assert(Dest.size() == Src.size());

    std::memmove(Dest.begin(), Src.begin(), Dest.size() * sizeof(BitWord));
    std::memset(Dest.end(), 0, Count * sizeof(BitWord));
  }

  MutableArrayRef<BitWord> allocate(size_t NumWords) {
    BitWord *RawBits = (BitWord *)std::malloc(NumWords * sizeof(BitWord));
    return MutableArrayRef<BitWord>(RawBits, NumWords);
  }

  int next_unset_in_word(int WordIndex, BitWord Word) const {
    unsigned Result = WordIndex * BITWORD_SIZE + countTrailingOnes(Word);
    return Result < size() ? Result : -1;
  }

  unsigned NumBitWords(unsigned S) const {
    return (S + BITWORD_SIZE-1) / BITWORD_SIZE;
  }

  // Set the unused bits in the high words.
  void set_unused_bits(bool t = true) {
    //  Set high words first.
    unsigned UsedWords = NumBitWords(Size);
    if (Bits.size() > UsedWords)
      init_words(Bits.drop_front(UsedWords), t);

    //  Then set any stray high bits of the last used word.
    unsigned ExtraBits = Size % BITWORD_SIZE;
    if (ExtraBits) {
      BitWord ExtraBitMask = ~0UL << ExtraBits;
      if (t)
        Bits[UsedWords-1] |= ExtraBitMask;
      else
        Bits[UsedWords-1] &= ~ExtraBitMask;
    }
  }

  // Clear the unused bits in the high words.
  void clear_unused_bits() {
    set_unused_bits(false);
  }

  void grow(unsigned NewSize) {
    size_t NewCapacity = std::max<size_t>(NumBitWords(NewSize), Bits.size() * 2);
    assert(NewCapacity > 0 && "realloc-ing zero space");
    BitWord *NewBits =
        (BitWord *)std::realloc(Bits.data(), NewCapacity * sizeof(BitWord));
    Bits = MutableArrayRef<BitWord>(NewBits, NewCapacity);
    clear_unused_bits();
  }

  void init_words(MutableArrayRef<BitWord> B, bool t) {
    if (B.size() > 0)
      memset(B.data(), 0 - (int)t, B.size() * sizeof(BitWord));
  }

  template<bool AddBits, bool InvertMask>
  void applyMask(const uint32_t *Mask, unsigned MaskWords) {
    static_assert(BITWORD_SIZE % 32 == 0, "Unsupported BitWord size.");
    MaskWords = std::min(MaskWords, (size() + 31) / 32);
    const unsigned Scale = BITWORD_SIZE / 32;
    unsigned i;
    for (i = 0; MaskWords >= Scale; ++i, MaskWords -= Scale) {
      BitWord BW = Bits[i];
      // This inner loop should unroll completely when BITWORD_SIZE > 32.
      for (unsigned b = 0; b != BITWORD_SIZE; b += 32) {
        uint32_t M = *Mask++;
        if (InvertMask) M = ~M;
        if (AddBits) BW |=   BitWord(M) << b;
        else         BW &= ~(BitWord(M) << b);
      }
      Bits[i] = BW;
    }
    for (unsigned b = 0; MaskWords; b += 32, --MaskWords) {
      uint32_t M = *Mask++;
      if (InvertMask) M = ~M;
      if (AddBits) Bits[i] |=   BitWord(M) << b;
      else         Bits[i] &= ~(BitWord(M) << b);
    }
    if (AddBits)
      clear_unused_bits();
  }

public:
  /// Return the size (in bytes) of the bit vector.
  size_t getMemorySize() const { return Bits.size() * sizeof(BitWord); }
  size_t getBitCapacity() const { return Bits.size() * BITWORD_SIZE; }
};

static inline size_t capacity_in_bytes(const BitVector &X) {
  return X.getMemorySize();
}

} // end namespace llvm

namespace std {
  /// Implement std::swap in terms of BitVector swap.
  inline void
  swap(llvm::BitVector &LHS, llvm::BitVector &RHS) {
    LHS.swap(RHS);
  }
} // end namespace std

#endif // LLVM_ADT_BITVECTOR_H

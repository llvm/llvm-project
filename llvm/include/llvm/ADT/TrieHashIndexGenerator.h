//===- TrieHashIndexGenerator.h ---------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_ADT_TRIEHASHINDEXGENERATOR_H
#define LLVM_ADT_TRIEHASHINDEXGENERATOR_H

#include "llvm/ADT/ArrayRef.h"
#include <optional>

namespace llvm {

/// The utility class that helps computing the index of the object inside trie
/// from its hash. The generator can be configured with the number of bits
/// used for each level of trie structure with \c NumRootsBits and \c
/// NumSubtrieBits.
/// For example, try computing indexes for a 16-bit hash 0x1234 with 8-bit root
/// and 4-bit sub-trie:
///
///   IndexGenerator IndexGen{8, 4, Hash};
///   size_t index1 = IndexGen.next(); // index 18 in root node.
///   size_t index2 = IndexGen.next(); // index 3 in sub-trie level 1.
///   size_t index3 = IndexGen.next(); // index 4 in sub-tire level 2.
///
/// This is used by different trie implementation to figure out where to
/// insert/find the object in the data structure.
struct TrieHashIndexGenerator {
  size_t NumRootBits;
  size_t NumSubtrieBits;
  ArrayRef<uint8_t> Bytes;
  std::optional<size_t> StartBit = std::nullopt;

  // Get the number of bits used to generate current index.
  size_t getNumBits() const {
    assert(StartBit);
    size_t TotalNumBits = Bytes.size() * 8;
    assert(*StartBit <= TotalNumBits);
    return std::min(*StartBit ? NumSubtrieBits : NumRootBits,
                    TotalNumBits - *StartBit);
  }

  // Get the index of the object in the next level of trie.
  size_t next() {
    if (!StartBit) {
      // Compute index for root when StartBit is not set.
      StartBit = 0;
      return getIndex(Bytes, *StartBit, NumRootBits);
    }
    if (*StartBit < Bytes.size() * 8) {
      // Compute index for sub-trie.
      *StartBit += *StartBit ? NumSubtrieBits : NumRootBits;
      assert((*StartBit - NumRootBits) % NumSubtrieBits == 0);
      return getIndex(Bytes, *StartBit, NumSubtrieBits);
    }
    // All the bits are consumed.
    return end();
  }

  // Provide a hint to speed up the index generation by providing the
  // information of the hash in current level. For example, if the object is
  // known to have \c Index on a level that already consumes first n \c Bits of
  // the hash, it can start index generation from this level by calling \c hint
  // function.
  size_t hint(unsigned Index, unsigned Bit) {
    assert(Bit < Bytes.size() * 8);
    assert(Bit == 0 || (Bit - NumRootBits) % NumSubtrieBits == 0);
    StartBit = Bit;
    return Index;
  }

  // Utility function for looking up the index in the trie for an object that
  // has colliding hash bits in the front as the hash of the object that is
  // currently being computed.
  size_t getCollidingBits(ArrayRef<uint8_t> CollidingBits) const {
    assert(StartBit);
    return getIndex(CollidingBits, *StartBit, NumSubtrieBits);
  }

  size_t end() const { return SIZE_MAX; }

  // Compute the index for the object from its hash, current start bits, and
  // the number of bits used for current level.
  static size_t getIndex(ArrayRef<uint8_t> Bytes, size_t StartBit,
                         size_t NumBits) {
    assert(StartBit < Bytes.size() * 8);
    // Drop all the bits before StartBit.
    Bytes = Bytes.drop_front(StartBit / 8u);
    StartBit %= 8u;
    size_t Index = 0;
    // Compute the index using the bits in range [StartBit, StartBit + NumBits),
    // note the range can spread across few `uint8_t` in the array.
    for (uint8_t Byte : Bytes) {
      size_t ByteStart = 0, ByteEnd = 8;
      if (StartBit) {
        ByteStart = StartBit;
        Byte &= (1u << (8 - StartBit)) - 1u;
        StartBit = 0;
      }
      size_t CurrentNumBits = ByteEnd - ByteStart;
      if (CurrentNumBits > NumBits) {
        Byte >>= CurrentNumBits - NumBits;
        CurrentNumBits = NumBits;
      }
      Index <<= CurrentNumBits;
      Index |= Byte & ((1u << CurrentNumBits) - 1u);

      assert(NumBits >= CurrentNumBits);
      NumBits -= CurrentNumBits;
      if (!NumBits)
        break;
    }
    return Index;
  }
};

} // namespace llvm

#endif // LLVM_ADT_TRIEHASHINDEXGENERATOR_H

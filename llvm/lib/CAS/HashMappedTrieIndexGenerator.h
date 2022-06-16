//===- HashMappedTrieIndexGenerator.h ---------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIB_CAS_HASHMAPPEDTRIEINDEXGENERATOR_H
#define LLVM_LIB_CAS_HASHMAPPEDTRIEINDEXGENERATOR_H

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/Optional.h"

namespace llvm {
namespace cas {

struct IndexGenerator {
  size_t NumRootBits;
  size_t NumSubtrieBits;
  ArrayRef<uint8_t> Bytes;
  Optional<size_t> StartBit = None;

  size_t getNumBits() const {
    assert(StartBit);
    size_t TotalNumBits = Bytes.size() * 8;
    assert(*StartBit <= TotalNumBits);
    return std::min(*StartBit ? NumSubtrieBits : NumRootBits,
                    TotalNumBits - *StartBit);
  }
  size_t next() {
    size_t Index;
    if (!StartBit) {
      StartBit = 0;
      Index = getIndex(Bytes, *StartBit, NumRootBits);
    } else {
      *StartBit += *StartBit ? NumSubtrieBits : NumRootBits;
      assert((*StartBit - NumRootBits) % NumSubtrieBits == 0);
      Index = getIndex(Bytes, *StartBit, NumSubtrieBits);
    }
    return Index;
  }

  size_t hint(unsigned Index, unsigned Bit) {
    assert(Index >= 0);
    assert(Bit < Bytes.size() * 8);
    assert(Bit == 0 || (Bit - NumRootBits) % NumSubtrieBits == 0);
    StartBit = Bit;
    return Index;
  }

  size_t getCollidingBits(ArrayRef<uint8_t> CollidingBits) const {
    assert(StartBit);
    return getIndex(CollidingBits, *StartBit, NumSubtrieBits);
  }

  static size_t getIndex(ArrayRef<uint8_t> Bytes, size_t StartBit,
                         size_t NumBits) {
    assert(StartBit < Bytes.size() * 8);

    Bytes = Bytes.drop_front(StartBit / 8u);
    StartBit %= 8u;
    size_t Index = 0;
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

} // namespace cas
} // namespace llvm

#endif // LLVM_LIB_CAS_HASHMAPPEDTRIEINDEXGENERATOR_H

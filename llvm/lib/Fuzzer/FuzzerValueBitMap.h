//===- FuzzerValueBitMap.h - INTERNAL - Bit map -----------------*- C++ -* ===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
// ValueBitMap.
//===----------------------------------------------------------------------===//

#ifndef LLVM_FUZZER_VALUE_BIT_MAP_H
#define LLVM_FUZZER_VALUE_BIT_MAP_H

namespace fuzzer {

// A bit map containing kMapSizeInWords bits.
struct ValueBitMap {
  static const size_t kMapSizeInBits = 65371;        // Prime.
  static const size_t kMapSizeInBitsAligned = 65536; // 2^16
  static const size_t kBitsInWord = (sizeof(uintptr_t) * 8);
  static const size_t kMapSizeInWords = kMapSizeInBitsAligned / kBitsInWord;
 public:
  // Clears all bits.
  void Reset() { memset(Map, 0, sizeof(Map)); }

  // Computed a hash function of Value and sets the corresponding bit.
  inline void AddValue(uintptr_t Value) {
    uintptr_t Idx = Value < kMapSizeInBits ? Value : Value % kMapSizeInBits;
    uintptr_t WordIdx = Idx / kBitsInWord;
    uintptr_t BitIdx = Idx % kBitsInWord;
    Map[WordIdx] |= 1UL << BitIdx;
  }

  // Merges 'Other' into 'this', clears 'Other',
  // returns the number of set bits in 'this'.
  ATTRIBUTE_TARGET_POPCNT
  size_t MergeFrom(ValueBitMap &Other) {
    uintptr_t Res = 0;
    for (size_t i = 0; i < kMapSizeInWords; i++) {
      auto O = Other.Map[i];
      auto M = Map[i];
      if (O) {
        Map[i] = (M |= O);
        Other.Map[i] = 0;
      }
      if (M)
        Res += __builtin_popcountl(M);
    }
    return Res;
  }

 private:
  uintptr_t Map[kMapSizeInWords] __attribute__((aligned(512)));
};

}  // namespace fuzzer

#endif  // LLVM_FUZZER_VALUE_BIT_MAP_H

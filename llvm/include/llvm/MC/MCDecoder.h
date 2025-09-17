//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// Disassembler decoder helper functions.
//===----------------------------------------------------------------------===//
#ifndef LLVM_MC_MCDECODER_H
#define LLVM_MC_MCDECODER_H

#include "llvm/ADT/APInt.h"
#include "llvm/MC/MCDisassembler/MCDisassembler.h"
#include "llvm/Support/MathExtras.h"
#include <bitset>
#include <cassert>

namespace llvm::MCD {

// Helper to propagate SoftFail status. Returns false if the status is Fail;
// callers are expected to early-exit in that condition. (Note, the '&' operator
// is correct to propagate the values of this enum; see comment on 'enum
// DecodeStatus'.)
inline bool Check(MCDisassembler::DecodeStatus &Out,
                  MCDisassembler::DecodeStatus In) {
  Out = static_cast<MCDisassembler::DecodeStatus>(Out & In);
  return Out != MCDisassembler::Fail;
}

// Extracts a given span of bits from the instruction bits and return it as an
// integer.
template <typename IntType>
#if defined(_MSC_VER) && !defined(__clang__)
__declspec(noinline)
#endif
inline std::enable_if_t<std::is_integral_v<IntType>, IntType>
fieldFromInstruction(const IntType &Insn, unsigned StartBit, unsigned NumBits) {
  assert(StartBit + NumBits <= 64 && "Cannot support >64-bit extractions!");
  assert(StartBit + NumBits <= (sizeof(IntType) * 8) &&
         "Instruction field out of bounds!");
  const IntType Mask = maskTrailingOnes<IntType>(NumBits);
  return (Insn >> StartBit) & Mask;
}

template <typename InsnType>
inline std::enable_if_t<!std::is_integral_v<InsnType>, uint64_t>
fieldFromInstruction(const InsnType &Insn, unsigned StartBit,
                     unsigned NumBits) {
  return Insn.extractBitsAsZExtValue(NumBits, StartBit);
}

template <size_t N>
uint64_t fieldFromInstruction(const std::bitset<N> &Insn, unsigned StartBit,
                              unsigned NumBits) {
  assert(StartBit + NumBits <= N && "Instruction field out of bounds!");
  assert(NumBits <= 64 && "Cannot support >64-bit extractions!");
  const std::bitset<N> Mask(maskTrailingOnes<uint64_t>(NumBits));
  return ((Insn >> StartBit) & Mask).to_ullong();
}

template <unsigned StartBit, unsigned NumBits, typename T>
inline std::enable_if_t<std::is_unsigned_v<T>, T> extractBits(T Val) {
  static_assert(StartBit + NumBits <= std::numeric_limits<T>::digits);
  return (Val >> StartBit) & maskTrailingOnes<T>(NumBits);
}

template <unsigned StartBit, unsigned NumBits, size_t N>
uint64_t extractBits(const std::bitset<N> &Val) {
  static_assert(StartBit + NumBits <= N);
  std::bitset<N> Mask = maskTrailingOnes<uint64_t>(NumBits);
  return ((Val >> StartBit) & Mask).to_ullong();
}

template <unsigned StartBit, unsigned NumBits>
uint64_t extractBits(const APInt &Val) {
  return Val.extractBitsAsZExtValue(NumBits, StartBit);
}

} // namespace llvm::MCD

#endif // LLVM_MC_MCDECODER_H

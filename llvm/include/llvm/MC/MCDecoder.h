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

// Helper function for inserting bits extracted from an encoded instruction into
// an integer-typed field.
template <typename IntType>
static std::enable_if_t<std::is_integral_v<IntType>, void>
insertBits(IntType &field, IntType bits, unsigned startBit, unsigned numBits) {
  // Check that no bit beyond numBits is set, so that a simple bitwise |
  // is sufficient.
  assert((~(((IntType)1 << numBits) - 1) & bits) == 0 &&
         "bits has more than numBits bits set");
  assert(startBit + numBits <= sizeof(IntType) * 8);
  (void)numBits;
  field |= bits << startBit;
}

} // namespace llvm::MCD

#endif // LLVM_MC_MCDECODER_H

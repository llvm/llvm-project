//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// Disassembler decoder state machine helper functions.
//===----------------------------------------------------------------------===//
#ifndef LLVM_MC_MCDECODER_H
#define LLVM_MC_MCDECODER_H

#include "llvm/MC/MCDisassembler/MCDisassembler.h"
#include "llvm/Support/MathExtras.h"
#include <bitset>
#include <cassert>

namespace llvm {
class APInt;
}

namespace llvm::MCD {

inline bool Check(MCDisassembler::DecodeStatus &Out,
                  MCDisassembler::DecodeStatus In) {
  Out = static_cast<MCDisassembler::DecodeStatus>(Out & In);
  return Out != MCDisassembler::Fail;
}

// fieldFromInstruction - Extracts a given span of bits from the instruction
// bits and return it as an integer.
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

template <size_t N>
inline uint64_t fieldFromInstruction(const std::bitset<N> &Insn,
                                     unsigned StartBit, unsigned NumBits) {
  assert(StartBit + NumBits <= N && "Instruction field out of bounds!");
  assert(NumBits <= 64 && "Cannot support >64-bit extractions!");
  const std::bitset<N> Mask(maskTrailingOnes<uint64_t>(NumBits));
  return ((Insn >> StartBit) & Mask).to_ullong();
}

template <typename InsnType>
inline std::enable_if_t<!std::is_integral_v<InsnType>, uint64_t>
fieldFromInstruction(const InsnType &Insn, unsigned StartBit,
                     unsigned NumBits) {
  return Insn.extractBitsAsZExtValue(NumBits, StartBit);
}

// InsnTypeTraits - Used by the decoder emitter to query whether a given type
// is valid type for the template type parameter <InsnType> and if yes, the
// instruction bitwidth its used to represent.
struct InsnTypeTraits {
  bool IsValid;
  size_t BitWidth;
};

// By default, all types are invalid.
template <typename T> inline constexpr InsnTypeTraits InsnTraits{false, 0};

// Integer types are valid types for InsnType.
template <> inline constexpr InsnTypeTraits InsnTraits<uint8_t>{true, 8};
template <> inline constexpr InsnTypeTraits InsnTraits<uint16_t>{true, 16};
template <> inline constexpr InsnTypeTraits InsnTraits<uint32_t>{true, 32};
template <> inline constexpr InsnTypeTraits InsnTraits<uint64_t>{true, 64};

// APInt type is valid, but is used for variable length instructions, so set the
// size to be 0.
template <> inline constexpr InsnTypeTraits InsnTraits<APInt>{true, 0};

// std::bitset<> is valid.
template <size_t N>
inline constexpr InsnTypeTraits InsnTraits<std::bitset<N>>{true, N};

} // namespace llvm::MCD

#endif

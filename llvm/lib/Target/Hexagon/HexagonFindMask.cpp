//===- HexagonFindMask.cpp - Hexagon relocation-application helper ------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Hexagon instruction encodings scatter an operand's bits across
// non-contiguous positions in the 32-bit instruction word. This file
// provides the helpers needed to apply an ELF relocation to such an
// instruction: locate which bits belong to the operand (using the
// generated per-instruction encoding table), scatter the relocated value
// into exactly those bit positions, and verify the result against the
// range/alignment rules from the Hexagon ABI.
//
// The functions are exposed with C linkage so that external tools that do
// not link against LLVM's C++ ABI (e.g. other Hexagon toolchain components)
// can compute relocations identically to LLVM itself. Inside LLVM, they are
// used by RuntimeDyldELF to resolve Hexagon relocations for MCJIT.
//
//===----------------------------------------------------------------------===//

#include "HexagonRelocationInfo.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/MathExtras.h"
#include "llvm/Target/Hexagon/HexagonRelocation.h"

namespace {
struct HexagonInstruction {
  uint32_t insnMask;
  uint32_t insnCmpMask;
  uint32_t insnBitMask;
  bool isDuplex;
};
} // namespace

#include "HexagonDepMask.h"

// Finds the bit mask for Insn by scanning the generated instruction
// encoding table.
static uint32_t findBitMaskHelper(uint32_t Insn,
                                  const HexagonInstruction *Encodings,
                                  size_t NumInsns) {
  for (size_t I = 0; I < NumInsns; ++I) {
    if (((Insn & 0xc000) == 0) && !Encodings[I].isDuplex)
      continue;
    if (((Insn & 0xc000) != 0) && Encodings[I].isDuplex)
      continue;
    if (((Encodings[I].insnMask) & Insn) == Encodings[I].insnCmpMask) {
      return Encodings[I].insnBitMask;
    }
  }
  llvm_unreachable("Cannot find encoding for Instruction!");
}

static uint32_t findMask(uint32_t I) {
  return findBitMaskHelper((uint32_t)I, InstructionEncodings,
                           sizeof(InstructionEncodings) /
                               sizeof(HexagonInstruction));
}

// Scatters the bits of Data over the positions set in Mask.
static uint32_t applyMask(uint32_t Mask, uint32_t Data) {
  uint32_t Result = 0;
  size_t Off = 0;

  for (size_t Bit = 0; Bit != sizeof(uint32_t) * 8; ++Bit) {
    const bool ValBit = (Data >> Off) & 1;
    const bool MaskBit = (Mask >> Bit) & 1;
    if (MaskBit) {
      Result |= (ValBit << Bit);
      ++Off;
    }
  }
  return Result;
}

// The low relocation in an extended branch encodes only bits 5:0. The
// EffectiveBits table entry retains the ABI-specified branch width.
static uint32_t getAppliedBits(uint32_t RelocType) {
  switch (RelocType) {
  case llvm::ELF::R_HEX_B22_PCREL_X:
  case llvm::ELF::R_HEX_B15_PCREL_X:
  case llvm::ELF::R_HEX_B13_PCREL_X:
  case llvm::ELF::R_HEX_B9_PCREL_X:
  case llvm::ELF::R_HEX_B7_PCREL_X:
  case llvm::ELF::R_HEX_GD_PLT_B22_PCREL_X:
  case llvm::ELF::R_HEX_LD_PLT_B22_PCREL_X:
    return 6;
  default:
    return llvm::Hexagon::Relocs[RelocType].EffectiveBits;
  }
}

// Computes the instruction bits with the relocation applied: shift, then
// truncate to the relocation's effective bit width, then scatter into the
// instruction's bit mask.
template <typename T>
static uint32_t doRelocHelper(uint32_t RelocType, uint32_t Instruction,
                              T Value) {
  const auto &Reloc = llvm::Hexagon::Relocs[RelocType];
  const uint32_t AppliedBits = getAppliedBits(RelocType);
  if (AppliedBits == 0)
    return Instruction;

  T ValueAfterShift = Value >> Reloc.Shift;
  uint32_t EffectiveValue =
      ValueAfterShift & llvm::maskTrailingOnes<uint32_t>(AppliedBits);
  uint32_t BitMask = Reloc.BitMask;
  if (BitMask == 0)
    BitMask = findMask(Instruction);
  Instruction &= ~BitMask;
  EffectiveValue = applyMask(BitMask, EffectiveValue);
  return EffectiveValue | Instruction;
}

static const llvm::Hexagon::HexagonRelocationInfo *
getRelocInfo(uint32_t RelocType) {
  if (RelocType >= std::size(llvm::Hexagon::Relocs))
    return nullptr;
  return &llvm::Hexagon::Relocs[RelocType];
}

extern "C" {

LLVM_ABI uint32_t LLVMHexagonApplyRelocation(uint32_t RelocType,
                                             uint32_t Instruction,
                                             uint32_t Value) {
  const auto *R = getRelocInfo(RelocType);
  if (!R)
    return Instruction;
  if (R->IsSigned)
    return doRelocHelper<int32_t>(RelocType, Instruction, Value);
  return doRelocHelper<uint32_t>(RelocType, Instruction, Value);
}

LLVM_ABI bool LLVMHexagonGetRelocationInfo(uint32_t RelocType,
                                           LLVMHexagonRelocInfo *Info) {
  const auto *R = getRelocInfo(RelocType);
  if (!R)
    return false;
  Info->Name = R->Name;
  Info->Type = R->Type;
  Info->EffectiveBits = R->EffectiveBits;
  Info->BitMask = R->BitMask;
  Info->Alignment = R->Alignment;
  Info->Shift = R->Shift;
  Info->VerifyRange = R->VerifyRange;
  Info->VerifyAlignment = R->VerifyAlignment;
  Info->IsSigned = R->IsSigned;
  return true;
}

LLVM_ABI bool LLVMHexagonVerifyRange(uint32_t RelocType, uint32_t Value) {
  const auto *R = getRelocInfo(RelocType);
  if (!R)
    return false;
  if (R->IsSigned)
    return llvm::isIntN(R->EffectiveBits, (int32_t)Value);
  return llvm::isUIntN(R->EffectiveBits, Value);
}

LLVM_ABI bool LLVMHexagonVerifyAlignment(uint32_t RelocType, uint32_t Value) {
  const auto *R = getRelocInfo(RelocType);
  return R && (R->Alignment == 0 || Value % R->Alignment == 0);
}

LLVM_ABI bool LLVMHexagonIsTruncated(uint32_t RelocType, uint32_t Value) {
  const auto *R = getRelocInfo(RelocType);
  if (!R || R->IsSigned || R->EffectiveBits == 0)
    return false;
  return Value & ~llvm::maskTrailingOnes<uint32_t>(R->EffectiveBits);
}

} // extern "C"

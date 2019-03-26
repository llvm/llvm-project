//===- AArch64ExpandImm.h - AArch64 Immediate Expansion -------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements the AArch64ExpandImm stuff.
//
//===----------------------------------------------------------------------===//

#include "AArch64.h"
#include "AArch64ExpandImm.h"
#include "MCTargetDesc/AArch64AddressingModes.h"

namespace llvm {

namespace AArch64_IMM {

/// Helper function which extracts the specified 16-bit chunk from a
/// 64-bit value.
static uint64_t getChunk(uint64_t Imm, unsigned ChunkIdx) {
  assert(ChunkIdx < 4 && "Out of range chunk index specified!");

  return (Imm >> (ChunkIdx * 16)) & 0xFFFF;
}

/// Check whether the given 16-bit chunk replicated to full 64-bit width
/// can be materialized with an ORR instruction.
static bool canUseOrr(uint64_t Chunk, uint64_t &Encoding) {
  Chunk = (Chunk << 48) | (Chunk << 32) | (Chunk << 16) | Chunk;

  return AArch64_AM::processLogicalImmediate(Chunk, 64, Encoding);
}

/// Check for identical 16-bit chunks within the constant and if so
/// materialize them with a single ORR instruction. The remaining one or two
/// 16-bit chunks will be materialized with MOVK instructions.
///
/// This allows us to materialize constants like |A|B|A|A| or |A|B|C|A| (order
/// of the chunks doesn't matter), assuming |A|A|A|A| can be materialized with
/// an ORR instruction.
static bool tryToreplicateChunks(uint64_t UImm,
				 SmallVectorImpl<ImmInsnModel> &Insn) {
  using CountMap = DenseMap<uint64_t, unsigned>;

  CountMap Counts;

  // Scan the constant and count how often every chunk occurs.
  for (unsigned Idx = 0; Idx < 4; ++Idx)
    ++Counts[getChunk(UImm, Idx)];

  // Traverse the chunks to find one which occurs more than once.
  for (CountMap::const_iterator Chunk = Counts.begin(), End = Counts.end();
       Chunk != End; ++Chunk) {
    const uint64_t ChunkVal = Chunk->first;
    const unsigned Count = Chunk->second;

    uint64_t Encoding = 0;

    // We are looking for chunks which have two or three instances and can be
    // materialized with an ORR instruction.
    if ((Count != 2 && Count != 3) || !canUseOrr(ChunkVal, Encoding))
      continue;

    const bool CountThree = Count == 3;

    Insn.push_back({ AArch64::ORRXri, 0, Encoding });

    unsigned ShiftAmt = 0;
    uint64_t Imm16 = 0;
    // Find the first chunk not materialized with the ORR instruction.
    for (; ShiftAmt < 64; ShiftAmt += 16) {
      Imm16 = (UImm >> ShiftAmt) & 0xFFFF;

      if (Imm16 != ChunkVal)
        break;
    }

    // Create the first MOVK instruction.
    Insn.push_back({ AArch64::MOVKXi, Imm16,
		     AArch64_AM::getShifterImm(AArch64_AM::LSL, ShiftAmt) });

    // In case we have three instances the whole constant is now materialized
    // and we can exit.
    if (CountThree)
      return true;

    // Find the remaining chunk which needs to be materialized.
    for (ShiftAmt += 16; ShiftAmt < 64; ShiftAmt += 16) {
      Imm16 = (UImm >> ShiftAmt) & 0xFFFF;

      if (Imm16 != ChunkVal)
        break;
    }
    Insn.push_back({ AArch64::MOVKXi, Imm16,
                     AArch64_AM::getShifterImm(AArch64_AM::LSL, ShiftAmt) });
    return true;
  }

  return false;
}

/// Check whether this chunk matches the pattern '1...0...'. This pattern
/// starts a contiguous sequence of ones if we look at the bits from the LSB
/// towards the MSB.
static bool isStartChunk(uint64_t Chunk) {
  if (Chunk == 0 || Chunk == std::numeric_limits<uint64_t>::max())
    return false;

  return isMask_64(~Chunk);
}

/// Check whether this chunk matches the pattern '0...1...' This pattern
/// ends a contiguous sequence of ones if we look at the bits from the LSB
/// towards the MSB.
static bool isEndChunk(uint64_t Chunk) {
  if (Chunk == 0 || Chunk == std::numeric_limits<uint64_t>::max())
    return false;

  return isMask_64(Chunk);
}

/// Clear or set all bits in the chunk at the given index.
static uint64_t updateImm(uint64_t Imm, unsigned Idx, bool Clear) {
  const uint64_t Mask = 0xFFFF;

  if (Clear)
    // Clear chunk in the immediate.
    Imm &= ~(Mask << (Idx * 16));
  else
    // Set all bits in the immediate for the particular chunk.
    Imm |= Mask << (Idx * 16);

  return Imm;
}

/// Check whether the constant contains a sequence of contiguous ones,
/// which might be interrupted by one or two chunks. If so, materialize the
/// sequence of contiguous ones with an ORR instruction.
/// Materialize the chunks which are either interrupting the sequence or outside
/// of the sequence with a MOVK instruction.
///
/// Assuming S is a chunk which starts the sequence (1...0...), E is a chunk
/// which ends the sequence (0...1...). Then we are looking for constants which
/// contain at least one S and E chunk.
/// E.g. |E|A|B|S|, |A|E|B|S| or |A|B|E|S|.
///
/// We are also looking for constants like |S|A|B|E| where the contiguous
/// sequence of ones wraps around the MSB into the LSB.
static bool trySequenceOfOnes(uint64_t UImm,
                              SmallVectorImpl<ImmInsnModel> &Insn) {
  const int NotSet = -1;
  const uint64_t Mask = 0xFFFF;

  int StartIdx = NotSet;
  int EndIdx = NotSet;
  // Try to find the chunks which start/end a contiguous sequence of ones.
  for (int Idx = 0; Idx < 4; ++Idx) {
    int64_t Chunk = getChunk(UImm, Idx);
    // Sign extend the 16-bit chunk to 64-bit.
    Chunk = (Chunk << 48) >> 48;

    if (isStartChunk(Chunk))
      StartIdx = Idx;
    else if (isEndChunk(Chunk))
      EndIdx = Idx;
  }

  // Early exit in case we can't find a start/end chunk.
  if (StartIdx == NotSet || EndIdx == NotSet)
    return false;

  // Outside of the contiguous sequence of ones everything needs to be zero.
  uint64_t Outside = 0;
  // Chunks between the start and end chunk need to have all their bits set.
  uint64_t Inside = Mask;

  // If our contiguous sequence of ones wraps around from the MSB into the LSB,
  // just swap indices and pretend we are materializing a contiguous sequence
  // of zeros surrounded by a contiguous sequence of ones.
  if (StartIdx > EndIdx) {
    std::swap(StartIdx, EndIdx);
    std::swap(Outside, Inside);
  }

  uint64_t OrrImm = UImm;
  int FirstMovkIdx = NotSet;
  int SecondMovkIdx = NotSet;

  // Find out which chunks we need to patch up to obtain a contiguous sequence
  // of ones.
  for (int Idx = 0; Idx < 4; ++Idx) {
    const uint64_t Chunk = getChunk(UImm, Idx);

    // Check whether we are looking at a chunk which is not part of the
    // contiguous sequence of ones.
    if ((Idx < StartIdx || EndIdx < Idx) && Chunk != Outside) {
      OrrImm = updateImm(OrrImm, Idx, Outside == 0);

      // Remember the index we need to patch.
      if (FirstMovkIdx == NotSet)
        FirstMovkIdx = Idx;
      else
        SecondMovkIdx = Idx;

      // Check whether we are looking a chunk which is part of the contiguous
      // sequence of ones.
    } else if (Idx > StartIdx && Idx < EndIdx && Chunk != Inside) {
      OrrImm = updateImm(OrrImm, Idx, Inside != Mask);

      // Remember the index we need to patch.
      if (FirstMovkIdx == NotSet)
        FirstMovkIdx = Idx;
      else
        SecondMovkIdx = Idx;
    }
  }
  assert(FirstMovkIdx != NotSet && "Constant materializable with single ORR!");

  // Create the ORR-immediate instruction.
  uint64_t Encoding = 0;
  AArch64_AM::processLogicalImmediate(OrrImm, 64, Encoding);
  Insn.push_back({ AArch64::ORRXri, 0, Encoding });

  const bool SingleMovk = SecondMovkIdx == NotSet;
  Insn.push_back({ AArch64::MOVKXi, getChunk(UImm, FirstMovkIdx),
                   AArch64_AM::getShifterImm(AArch64_AM::LSL,
                                             FirstMovkIdx * 16) });

  // Early exit in case we only need to emit a single MOVK instruction.
  if (SingleMovk)
    return true;

  // Create the second MOVK instruction.
  Insn.push_back({ AArch64::MOVKXi, getChunk(UImm, SecondMovkIdx),
	           AArch64_AM::getShifterImm(AArch64_AM::LSL,
                                             SecondMovkIdx * 16) });

  return true;
}

/// \brief Expand a MOVi32imm or MOVi64imm pseudo instruction to a
/// MOVZ or MOVN of width BitSize followed by up to 3 MOVK instructions.
static inline void expandMOVImmSimple(uint64_t Imm, unsigned BitSize,
				      unsigned OneChunks, unsigned ZeroChunks,
				      SmallVectorImpl<ImmInsnModel> &Insn) {
  const unsigned Mask = 0xFFFF;

  // Use a MOVZ or MOVN instruction to set the high bits, followed by one or
  // more MOVK instructions to insert additional 16-bit portions into the
  // lower bits.
  bool isNeg = false;

  // Use MOVN to materialize the high bits if we have more all one chunks
  // than all zero chunks.
  if (OneChunks > ZeroChunks) {
    isNeg = true;
    Imm = ~Imm;
  }

  unsigned FirstOpc;
  if (BitSize == 32) {
    Imm &= (1LL << 32) - 1;
    FirstOpc = (isNeg ? AArch64::MOVNWi : AArch64::MOVZWi);
  } else {
    FirstOpc = (isNeg ? AArch64::MOVNXi : AArch64::MOVZXi);
  }
  unsigned Shift = 0;     // LSL amount for high bits with MOVZ/MOVN
  unsigned LastShift = 0; // LSL amount for last MOVK
  if (Imm != 0) {
    unsigned LZ = countLeadingZeros(Imm);
    unsigned TZ = countTrailingZeros(Imm);
    Shift = (TZ / 16) * 16;
    LastShift = ((63 - LZ) / 16) * 16;
  }
  unsigned Imm16 = (Imm >> Shift) & Mask;

  Insn.push_back({ FirstOpc, Imm16,
                   AArch64_AM::getShifterImm(AArch64_AM::LSL, Shift) });

  if (Shift == LastShift)
    return;

  // If a MOVN was used for the high bits of a negative value, flip the rest
  // of the bits back for use with MOVK.
  if (isNeg)
    Imm = ~Imm;

  unsigned Opc = (BitSize == 32 ? AArch64::MOVKWi : AArch64::MOVKXi);
  while (Shift < LastShift) {
    Shift += 16;
    Imm16 = (Imm >> Shift) & Mask;
    if (Imm16 == (isNeg ? Mask : 0))
      continue; // This 16-bit portion is already set correctly.

    Insn.push_back({ Opc, Imm16,
                     AArch64_AM::getShifterImm(AArch64_AM::LSL, Shift) });
  }
}

/// Expand a MOVi32imm or MOVi64imm pseudo instruction to one or more
/// real move-immediate instructions to synthesize the immediate.
void expandMOVImm(uint64_t Imm, unsigned BitSize,
		  SmallVectorImpl<ImmInsnModel> &Insn) {
  const unsigned Mask = 0xFFFF;

  // Scan the immediate and count the number of 16-bit chunks which are either
  // all ones or all zeros.
  unsigned OneChunks = 0;
  unsigned ZeroChunks = 0;
  for (unsigned Shift = 0; Shift < BitSize; Shift += 16) {
    const unsigned Chunk = (Imm >> Shift) & Mask;
    if (Chunk == Mask)
      OneChunks++;
    else if (Chunk == 0)
      ZeroChunks++;
  }

  // Prefer MOVZ/MOVN over ORR because of the rules for the "mov" alias.
  if ((BitSize / 16) - OneChunks <= 1 || (BitSize / 16) - ZeroChunks <= 1) {
    expandMOVImmSimple(Imm, BitSize, OneChunks, ZeroChunks, Insn);
    return;
  }

  // Try a single ORR.
  uint64_t UImm = Imm << (64 - BitSize) >> (64 - BitSize);
  uint64_t Encoding;
  if (AArch64_AM::processLogicalImmediate(UImm, BitSize, Encoding)) {
    unsigned Opc = (BitSize == 32 ? AArch64::ORRWri : AArch64::ORRXri);
    Insn.push_back({ Opc, 0, Encoding });
    return;
  }

  // One to up three instruction sequences.
  //
  // Prefer MOVZ/MOVN followed by MOVK; it's more readable, and possibly the
  // fastest sequence with fast literal generation.
  if (OneChunks >= (BitSize / 16) - 2 || ZeroChunks >= (BitSize / 16) - 2) {
    expandMOVImmSimple(Imm, BitSize, OneChunks, ZeroChunks, Insn);
    return;
  }

  assert(BitSize == 64 && "All 32-bit immediates can be expanded with a"
                          "MOVZ/MOVK pair");

  // Try other two-instruction sequences.

  // 64-bit ORR followed by MOVK.
  // We try to construct the ORR immediate in three different ways: either we
  // zero out the chunk which will be replaced, we fill the chunk which will
  // be replaced with ones, or we take the bit pattern from the other half of
  // the 64-bit immediate. This is comprehensive because of the way ORR
  // immediates are constructed.
  for (unsigned Shift = 0; Shift < BitSize; Shift += 16) {
    uint64_t ShiftedMask = (0xFFFFULL << Shift);
    uint64_t ZeroChunk = UImm & ~ShiftedMask;
    uint64_t OneChunk = UImm | ShiftedMask;
    uint64_t RotatedImm = (UImm << 32) | (UImm >> 32);
    uint64_t ReplicateChunk = ZeroChunk | (RotatedImm & ShiftedMask);
    if (AArch64_AM::processLogicalImmediate(ZeroChunk, BitSize, Encoding) ||
        AArch64_AM::processLogicalImmediate(OneChunk, BitSize, Encoding) ||
        AArch64_AM::processLogicalImmediate(ReplicateChunk, BitSize,
                                            Encoding)) {
      // Create the ORR-immediate instruction.
      Insn.push_back({ AArch64::ORRXri, 0, Encoding });

      // Create the MOVK instruction.
      const unsigned Imm16 = getChunk(UImm, Shift / 16);
      Insn.push_back({ AArch64::MOVKXi, Imm16,
		       AArch64_AM::getShifterImm(AArch64_AM::LSL, Shift) });
      return;
    }
  }

  // FIXME: Add more two-instruction sequences.

  // Three instruction sequences.
  //
  // Prefer MOVZ/MOVN followed by two MOVK; it's more readable, and possibly
  // the fastest sequence with fast literal generation. (If neither MOVK is
  // part of a fast literal generation pair, it could be slower than the
  // four-instruction sequence, but we won't worry about that for now.)
  if (OneChunks || ZeroChunks) {
    expandMOVImmSimple(Imm, BitSize, OneChunks, ZeroChunks, Insn);
    return;
  }

  // Check for identical 16-bit chunks within the constant and if so materialize
  // them with a single ORR instruction. The remaining one or two 16-bit chunks
  // will be materialized with MOVK instructions.
  if (BitSize == 64 && tryToreplicateChunks(UImm, Insn))
    return;

  // Check whether the constant contains a sequence of contiguous ones, which
  // might be interrupted by one or two chunks. If so, materialize the sequence
  // of contiguous ones with an ORR instruction. Materialize the chunks which
  // are either interrupting the sequence or outside of the sequence with a
  // MOVK instruction.
  if (BitSize == 64 && trySequenceOfOnes(UImm, Insn))
    return;

  // We found no possible two or three instruction sequence; use the general
  // four-instruction sequence.
  expandMOVImmSimple(Imm, BitSize, OneChunks, ZeroChunks, Insn);
}

} // end namespace AArch64_AM

} // end namespace llvm

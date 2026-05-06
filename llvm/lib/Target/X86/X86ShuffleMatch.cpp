//===-- X86ShuffleMatch.cpp - X86 Shuffle Pattern Matching ---------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements shared shuffle pattern matching functions that can be
// used by both SelectionDAG and GlobalISel lowering.
//
//===----------------------------------------------------------------------===//

#include "X86ShuffleMatch.h"
#include "llvm/ADT/SmallVector.h"

using namespace llvm;

/// Compute the SHUFPS/SHUFPD immediate encoding for a per-lane shuffle mask.
static unsigned getShufpImm(ArrayRef<int> LaneMask, unsigned EltsPerLane) {
  unsigned BitsPerIdx = (EltsPerLane == 4) ? 2 : 1;
  unsigned HalfLane = EltsPerLane / 2;
  unsigned Imm = 0;
  for (unsigned i = 0; i < EltsPerLane; ++i) {
    int M = LaneMask[i];
    unsigned Val;
    if (M < 0)
      Val = (i < HalfLane) ? i : (i - HalfLane);
    else
      Val = M;
    Imm |= (Val & ((1u << BitsPerIdx) - 1)) << (i * BitsPerIdx);
  }
  return Imm;
}

bool X86::matchShufpMask(ArrayRef<int> Mask, unsigned NumElts,
                         unsigned NumSrcElts, unsigned EltSize,
                         bool SingleSource, unsigned &Imm, bool &Swap) {
  unsigned EltsPerLane = 128 / EltSize; // 4 for 32-bit, 2 for 64-bit
  unsigned NumLanes = NumElts / EltsPerLane;
  unsigned HalfLane = EltsPerLane / 2;

  // SHUFPS/SHUFPD only work for 32-bit or 64-bit elements
  if (EltSize != 32 && EltSize != 64)
    return false;

  for (int Attempt = 0; Attempt < 2; ++Attempt) {
    bool TrySwap = (Attempt == 1);
    if (TrySwap && SingleSource)
      break;

    bool Valid = true;
    unsigned FirstLaneImm = 0;

    for (unsigned Lane = 0; Lane < NumLanes && Valid; ++Lane) {
      unsigned LaneStart = Lane * EltsPerLane;
      SmallVector<int, 4> LaneMask(EltsPerLane);

      for (unsigned i = 0; i < EltsPerLane && Valid; ++i) {
        int M = Mask[LaneStart + i];

        if (M < 0) {
          LaneMask[i] = -1;
          continue;
        }

        bool FromSrc2 = ((unsigned)M >= NumSrcElts);
        unsigned SrcIdx = FromSrc2 ? (M - NumSrcElts) : M;

        // Must reference the same lane in the source
        if (SrcIdx / EltsPerLane != Lane) {
          Valid = false;
          break;
        }
        unsigned SrcLaneOff = SrcIdx % EltsPerLane;

        if (i < HalfLane) {
          // Low half: should come from the first operand
          bool WantSrc2 = TrySwap;
          if (!SingleSource && FromSrc2 != WantSrc2) {
            Valid = false;
            break;
          }
        } else {
          // High half: should come from the second operand
          bool WantSrc2 = !TrySwap;
          if (!SingleSource && FromSrc2 != WantSrc2) {
            Valid = false;
            break;
          }
        }

        LaneMask[i] = SrcLaneOff;
      }

      if (!Valid)
        break;

      unsigned CurImm = getShufpImm(LaneMask, EltsPerLane);
      if (Lane == 0)
        FirstLaneImm = CurImm;
      else if (CurImm != FirstLaneImm)
        Valid = false;
    }

    if (Valid) {
      Imm = FirstLaneImm;
      Swap = TrySwap;
      return true;
    }
  }
  return false;
}

bool X86::isBroadcastMask(ArrayRef<int> Mask) {
  if (Mask.empty())
    return false;

  int BroadcastIdx = -1;
  for (int M : Mask) {
    if (M < 0) // undef
      continue;
    if (BroadcastIdx < 0)
      BroadcastIdx = M;
    else if (M != BroadcastIdx)
      return false;
  }
  return BroadcastIdx >= 0;
}

bool X86::matchBlendMask(ArrayRef<int> Mask, unsigned NumElts,
                         unsigned NumSrcElts) {
  for (unsigned i = 0; i < NumElts; ++i) {
    if (Mask[i] < 0)
      continue; // undef is fine
    if (Mask[i] != (int)i && Mask[i] != (int)(i + NumSrcElts))
      return false;
  }
  return true;
}

bool X86::matchUnpackLowMask(ArrayRef<int> Mask, unsigned NumElts,
                             unsigned NumSrcElts, unsigned EltSize,
                             bool &Swap) {
  unsigned EltsPerLane = 128 / EltSize;
  unsigned NumLanes = NumElts / EltsPerLane;
  unsigned HalfLane = EltsPerLane / 2;

  for (int Attempt = 0; Attempt < 2; ++Attempt) {
    bool TrySwap = (Attempt == 1);
    bool Valid = true;

    for (unsigned Lane = 0; Lane < NumLanes && Valid; ++Lane) {
      unsigned LaneStart = Lane * EltsPerLane;
      unsigned SrcLaneBase = Lane * EltsPerLane;

      for (unsigned i = 0; i < EltsPerLane && Valid; ++i) {
        int M = Mask[LaneStart + i];
        if (M < 0)
          continue;

        bool FromSrc2 = ((unsigned)M >= NumSrcElts);
        unsigned SrcIdx = FromSrc2 ? (M - NumSrcElts) : M;

        // Determine expected source and index for UNPCKL
        bool ExpectSrc2;
        unsigned ExpectIdx;
        if ((i % 2) == 0) {
          // Even positions: low half of first source
          ExpectSrc2 = TrySwap;
          ExpectIdx = SrcLaneBase + (i / 2);
        } else {
          // Odd positions: low half of second source
          ExpectSrc2 = !TrySwap;
          ExpectIdx = SrcLaneBase + (i / 2);
        }

        if (FromSrc2 != ExpectSrc2 || SrcIdx != ExpectIdx) {
          Valid = false;
          break;
        }
      }
    }

    if (Valid) {
      Swap = TrySwap;
      return true;
    }
  }
  return false;
}

bool X86::matchUnpackHighMask(ArrayRef<int> Mask, unsigned NumElts,
                              unsigned NumSrcElts, unsigned EltSize,
                              bool &Swap) {
  unsigned EltsPerLane = 128 / EltSize;
  unsigned NumLanes = NumElts / EltsPerLane;
  unsigned HalfLane = EltsPerLane / 2;

  for (int Attempt = 0; Attempt < 2; ++Attempt) {
    bool TrySwap = (Attempt == 1);
    bool Valid = true;

    for (unsigned Lane = 0; Lane < NumLanes && Valid; ++Lane) {
      unsigned LaneStart = Lane * EltsPerLane;
      unsigned SrcLaneBase = Lane * EltsPerLane;

      for (unsigned i = 0; i < EltsPerLane && Valid; ++i) {
        int M = Mask[LaneStart + i];
        if (M < 0)
          continue;

        bool FromSrc2 = ((unsigned)M >= NumSrcElts);
        unsigned SrcIdx = FromSrc2 ? (M - NumSrcElts) : M;

        // Determine expected source and index for UNPCKH
        bool ExpectSrc2;
        unsigned ExpectIdx;
        if ((i % 2) == 0) {
          // Even positions: high half of first source
          ExpectSrc2 = TrySwap;
          ExpectIdx = SrcLaneBase + HalfLane + (i / 2);
        } else {
          // Odd positions: high half of second source
          ExpectSrc2 = !TrySwap;
          ExpectIdx = SrcLaneBase + HalfLane + (i / 2);
        }

        if (FromSrc2 != ExpectSrc2 || SrcIdx != ExpectIdx) {
          Valid = false;
          break;
        }
      }
    }

    if (Valid) {
      Swap = TrySwap;
      return true;
    }
  }
  return false;
}

bool X86::matchPshufdMask(ArrayRef<int> Mask, unsigned NumElts, unsigned &Imm) {
  // PSHUFD works on 4 doublewords (128-bit lane)
  if (NumElts != 4 && NumElts != 8 && NumElts != 16)
    return false;

  unsigned LaneSize = 4; // 4 x i32 per 128-bit lane
  unsigned NumLanes = NumElts / LaneSize;

  unsigned FirstLaneImm = 0;
  for (unsigned Lane = 0; Lane < NumLanes; ++Lane) {
    unsigned LaneStart = Lane * LaneSize;
    unsigned SrcLaneBase = Lane * LaneSize;
    unsigned LaneImm = 0;

    for (unsigned i = 0; i < LaneSize; ++i) {
      int M = Mask[LaneStart + i];
      if (M < 0)
        M = i; // Treat undef as identity

      // Must reference same lane
      if ((unsigned)M < SrcLaneBase || (unsigned)M >= SrcLaneBase + LaneSize)
        return false;

      unsigned Idx = M - SrcLaneBase;
      LaneImm |= (Idx << (i * 2));
    }

    if (Lane == 0)
      FirstLaneImm = LaneImm;
    else if (LaneImm != FirstLaneImm)
      return false; // All lanes must have same pattern
  }

  Imm = FirstLaneImm;
  return true;
}

bool X86::matchPshufbMask(ArrayRef<int> Mask, unsigned NumElts,
                          unsigned EltSize) {
  // PSHUFB works at byte granularity
  if (EltSize != 8)
    return false;

  // Check that no shuffle crosses 128-bit lane boundaries
  for (unsigned i = 0; i < NumElts; ++i) {
    int M = Mask[i];
    if (M < 0)
      continue; // undef is OK

    unsigned DstLane = i / 16;
    unsigned SrcLane = M / 16;
    if (DstLane != SrcLane)
      return false;
  }

  return true;
}

bool X86::matchVPermilMask(ArrayRef<int> Mask, unsigned NumElts,
                           unsigned EltSize, int &Imm) {
  // VPERMILPS works on 4 elements per lane, VPERMILPD on 2 elements per lane
  if (EltSize != 32 && EltSize != 64)
    return false;

  unsigned EltsPerLane = 128 / EltSize;
  unsigned NumLanes = NumElts / EltsPerLane;

  // Check if this is an immediate form (same pattern in all lanes)
  SmallVector<int, 4> FirstLaneMask;
  for (unsigned i = 0; i < EltsPerLane; ++i) {
    FirstLaneMask.push_back(Mask[i] >= 0 ? (Mask[i] % EltsPerLane) : -1);
  }

  bool IsImmediate = true;
  for (unsigned Lane = 1; Lane < NumLanes; ++Lane) {
    unsigned LaneStart = Lane * EltsPerLane;
    for (unsigned i = 0; i < EltsPerLane; ++i) {
      int M = Mask[LaneStart + i];
      int ExpectedM = FirstLaneMask[i];

      if (M < 0 && ExpectedM < 0)
        continue;
      if (M >= 0 && ExpectedM >= 0) {
        unsigned MLane = M % EltsPerLane;
        if (MLane != (unsigned)ExpectedM) {
          IsImmediate = false;
          break;
        }
      } else {
        IsImmediate = false;
        break;
      }
    }
    if (!IsImmediate)
      break;
  }

  // Check all references stay in-lane
  for (unsigned i = 0; i < NumElts; ++i) {
    int M = Mask[i];
    if (M < 0)
      continue;

    unsigned DstLane = i / EltsPerLane;
    unsigned SrcLane = M / EltsPerLane;
    if (DstLane != SrcLane)
      return false;
  }

  if (IsImmediate && EltsPerLane == 4) {
    // Compute VPERMILPS immediate
    unsigned ImmVal = 0;
    for (unsigned i = 0; i < 4; ++i) {
      int M = FirstLaneMask[i];
      if (M < 0)
        M = i;
      ImmVal |= (M & 3) << (i * 2);
    }
    Imm = ImmVal;
  } else if (IsImmediate && EltsPerLane == 2) {
    // Compute VPERMILPD immediate
    unsigned ImmVal = 0;
    for (unsigned i = 0; i < 2; ++i) {
      int M = FirstLaneMask[i];
      if (M < 0)
        M = i;
      ImmVal |= (M & 1) << i;
    }
    Imm = ImmVal;
  } else {
    Imm = -1; // Variable mask form
  }

  return true;
}

bool X86::matchVPermiMask(ArrayRef<int> Mask, unsigned NumElts, unsigned &Imm) {
  // VPERMQ/VPERMPD - 256-bit cross-lane permute (AVX2)
  // Works on 4 x 64-bit elements
  if (NumElts != 4)
    return false;

  unsigned ImmVal = 0;
  for (unsigned i = 0; i < 4; ++i) {
    int M = Mask[i];
    if (M < 0)
      M = i; // Treat undef as identity

    if (M < 0 || M >= 4)
      return false;

    ImmVal |= (M & 3) << (i * 2);
  }

  Imm = ImmVal;
  return true;
}

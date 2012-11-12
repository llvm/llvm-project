//===-- X86ShuffleDecode.cpp - X86 shuffle decode logic -------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// Define several functions to decode x86 specific shuffle semantics into a
// generic vector mask.
//
//===----------------------------------------------------------------------===//

#include "X86ShuffleDecode.h"

//===----------------------------------------------------------------------===//
//  Vector Mask Decoding
//===----------------------------------------------------------------------===//

namespace llvm {

void DecodeINSERTPSMask(unsigned Imm, SmallVectorImpl<int> &ShuffleMask) {
  // Defaults the copying the dest value.
  ShuffleMask.push_back(0);
  ShuffleMask.push_back(1);
  ShuffleMask.push_back(2);
  ShuffleMask.push_back(3);

  // Decode the immediate.
  unsigned ZMask = Imm & 15;
  unsigned CountD = (Imm >> 4) & 3;
  unsigned CountS = (Imm >> 6) & 3;

  // CountS selects which input element to use.
  unsigned InVal = 4+CountS;
  // CountD specifies which element of destination to update.
  ShuffleMask[CountD] = InVal;
  // ZMask zaps values, potentially overriding the CountD elt.
  if (ZMask & 1) ShuffleMask[0] = SM_SentinelZero;
  if (ZMask & 2) ShuffleMask[1] = SM_SentinelZero;
  if (ZMask & 4) ShuffleMask[2] = SM_SentinelZero;
  if (ZMask & 8) ShuffleMask[3] = SM_SentinelZero;
}

// <3,1> or <6,7,2,3>
void DecodeMOVHLPSMask(unsigned NElts, SmallVectorImpl<int> &ShuffleMask) {
  for (unsigned i = NElts/2; i != NElts; ++i)
    ShuffleMask.push_back(NElts+i);

  for (unsigned i = NElts/2; i != NElts; ++i)
    ShuffleMask.push_back(i);
}

// <0,2> or <0,1,4,5>
void DecodeMOVLHPSMask(unsigned NElts, SmallVectorImpl<int> &ShuffleMask) {
  for (unsigned i = 0; i != NElts/2; ++i)
    ShuffleMask.push_back(i);

  for (unsigned i = 0; i != NElts/2; ++i)
    ShuffleMask.push_back(NElts+i);
}

/// DecodePSHUFMask - This decodes the shuffle masks for pshufd, and vpermilp*.
/// VT indicates the type of the vector allowing it to handle different
/// datatypes and vector widths.
void DecodePSHUFMask(MVT VT, unsigned Imm, SmallVectorImpl<int> &ShuffleMask) {
  unsigned NumElts = VT.getVectorNumElements();

  unsigned NumLanes = VT.getSizeInBits() / 128;
  unsigned NumLaneElts = NumElts / NumLanes;

  unsigned NewImm = Imm;
  for (unsigned l = 0; l != NumElts; l += NumLaneElts) {
    for (unsigned i = 0; i != NumLaneElts; ++i) {
      ShuffleMask.push_back(NewImm % NumLaneElts + l);
      NewImm /= NumLaneElts;
    }
    if (NumLaneElts == 4) NewImm = Imm; // reload imm
  }
}

void DecodePSHUFHWMask(MVT VT, unsigned Imm,
                       SmallVectorImpl<int> &ShuffleMask) {
  unsigned NumElts = VT.getVectorNumElements();

  for (unsigned l = 0; l != NumElts; l += 8) {
    unsigned NewImm = Imm;
    for (unsigned i = 0, e = 4; i != e; ++i) {
      ShuffleMask.push_back(l + i);
    }
    for (unsigned i = 4, e = 8; i != e; ++i) {
      ShuffleMask.push_back(l + 4 + (NewImm & 3));
      NewImm >>= 2;
    }
  }
}

void DecodePSHUFLWMask(MVT VT, unsigned Imm,
                       SmallVectorImpl<int> &ShuffleMask) {
  unsigned NumElts = VT.getVectorNumElements();

  for (unsigned l = 0; l != NumElts; l += 8) {
    unsigned NewImm = Imm;
    for (unsigned i = 0, e = 4; i != e; ++i) {
      ShuffleMask.push_back(l + (NewImm & 3));
      NewImm >>= 2;
    }
    for (unsigned i = 4, e = 8; i != e; ++i) {
      ShuffleMask.push_back(l + i);
    }
  }
}

/// DecodeSHUFPMask - This decodes the shuffle masks for shufp*. VT indicates
/// the type of the vector allowing it to handle different datatypes and vector
/// widths.
void DecodeSHUFPMask(MVT VT, unsigned Imm, SmallVectorImpl<int> &ShuffleMask) {
  unsigned NumElts = VT.getVectorNumElements();

  unsigned NumLanes = VT.getSizeInBits() / 128;
  unsigned NumLaneElts = NumElts / NumLanes;

  unsigned NewImm = Imm;
  for (unsigned l = 0; l != NumElts; l += NumLaneElts) {
    // each half of a lane comes from different source
    for (unsigned s = 0; s != NumElts*2; s += NumElts) {
      for (unsigned i = 0; i != NumLaneElts/2; ++i) {
        ShuffleMask.push_back(NewImm % NumLaneElts + s + l);
        NewImm /= NumLaneElts;
      }
    }
    if (NumLaneElts == 4) NewImm = Imm; // reload imm
  }
}

/// DecodeUNPCKHMask - This decodes the shuffle masks for unpckhps/unpckhpd
/// and punpckh*. VT indicates the type of the vector allowing it to handle
/// different datatypes and vector widths.
void DecodeUNPCKHMask(MVT VT, SmallVectorImpl<int> &ShuffleMask) {
  unsigned NumElts = VT.getVectorNumElements();

  // Handle 128 and 256-bit vector lengths. AVX defines UNPCK* to operate
  // independently on 128-bit lanes.
  unsigned NumLanes = VT.getSizeInBits() / 128;
  if (NumLanes == 0 ) NumLanes = 1;  // Handle MMX
  unsigned NumLaneElts = NumElts / NumLanes;

  for (unsigned l = 0; l != NumElts; l += NumLaneElts) {
    for (unsigned i = l + NumLaneElts/2, e = l + NumLaneElts; i != e; ++i) {
      ShuffleMask.push_back(i);          // Reads from dest/src1
      ShuffleMask.push_back(i+NumElts);  // Reads from src/src2
    }
  }
}

/// DecodeUNPCKLMask - This decodes the shuffle masks for unpcklps/unpcklpd
/// and punpckl*. VT indicates the type of the vector allowing it to handle
/// different datatypes and vector widths.
void DecodeUNPCKLMask(MVT VT, SmallVectorImpl<int> &ShuffleMask) {
  unsigned NumElts = VT.getVectorNumElements();

  // Handle 128 and 256-bit vector lengths. AVX defines UNPCK* to operate
  // independently on 128-bit lanes.
  unsigned NumLanes = VT.getSizeInBits() / 128;
  if (NumLanes == 0 ) NumLanes = 1;  // Handle MMX
  unsigned NumLaneElts = NumElts / NumLanes;

  for (unsigned l = 0; l != NumElts; l += NumLaneElts) {
    for (unsigned i = l, e = l + NumLaneElts/2; i != e; ++i) {
      ShuffleMask.push_back(i);          // Reads from dest/src1
      ShuffleMask.push_back(i+NumElts);  // Reads from src/src2
    }
  }
}

void DecodeVPERM2X128Mask(MVT VT, unsigned Imm,
                          SmallVectorImpl<int> &ShuffleMask) {
  if (Imm & 0x88)
    return; // Not a shuffle

  unsigned HalfSize = VT.getVectorNumElements()/2;

  for (unsigned l = 0; l != 2; ++l) {
    unsigned HalfBegin = ((Imm >> (l*4)) & 0x3) * HalfSize;
    for (unsigned i = HalfBegin, e = HalfBegin+HalfSize; i != e; ++i)
      ShuffleMask.push_back(i);
  }
}

/// DecodeVPERMMask - this decodes the shuffle masks for VPERMQ/VPERMPD.
/// No VT provided since it only works on 256-bit, 4 element vectors.
void DecodeVPERMMask(unsigned Imm, SmallVectorImpl<int> &ShuffleMask) {
  for (unsigned i = 0; i != 4; ++i) {
    ShuffleMask.push_back((Imm >> (2*i)) & 3);
  }
}

} // llvm namespace

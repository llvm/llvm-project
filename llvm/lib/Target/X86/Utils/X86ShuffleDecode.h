//===-- X86ShuffleDecode.h - X86 shuffle decode logic -----------*-C++-*---===//
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

#ifndef X86_SHUFFLE_DECODE_H
#define X86_SHUFFLE_DECODE_H

#include "llvm/ADT/SmallVector.h"
#include "llvm/CodeGen/ValueTypes.h"

//===----------------------------------------------------------------------===//
//  Vector Mask Decoding
//===----------------------------------------------------------------------===//

namespace llvm {
enum {
  SM_SentinelZero = -1
};

void DecodeINSERTPSMask(unsigned Imm, SmallVectorImpl<int> &ShuffleMask);

// <3,1> or <6,7,2,3>
void DecodeMOVHLPSMask(unsigned NElts, SmallVectorImpl<int> &ShuffleMask);

// <0,2> or <0,1,4,5>
void DecodeMOVLHPSMask(unsigned NElts, SmallVectorImpl<int> &ShuffleMask);

void DecodePSHUFMask(MVT VT, unsigned Imm, SmallVectorImpl<int> &ShuffleMask);

void DecodePSHUFHWMask(MVT VT, unsigned Imm, SmallVectorImpl<int> &ShuffleMask);

void DecodePSHUFLWMask(MVT, unsigned Imm, SmallVectorImpl<int> &ShuffleMask);

/// DecodeSHUFPMask - This decodes the shuffle masks for shufp*. VT indicates
/// the type of the vector allowing it to handle different datatypes and vector
/// widths.
void DecodeSHUFPMask(MVT VT, unsigned Imm, SmallVectorImpl<int> &ShuffleMask);

/// DecodeUNPCKHMask - This decodes the shuffle masks for unpckhps/unpckhpd
/// and punpckh*. VT indicates the type of the vector allowing it to handle
/// different datatypes and vector widths.
void DecodeUNPCKHMask(MVT VT, SmallVectorImpl<int> &ShuffleMask);

/// DecodeUNPCKLMask - This decodes the shuffle masks for unpcklps/unpcklpd
/// and punpckl*. VT indicates the type of the vector allowing it to handle
/// different datatypes and vector widths.
void DecodeUNPCKLMask(MVT VT, SmallVectorImpl<int> &ShuffleMask);


void DecodeVPERM2X128Mask(MVT VT, unsigned Imm,
                          SmallVectorImpl<int> &ShuffleMask);

/// DecodeVPERMMask - this decodes the shuffle masks for VPERMQ/VPERMPD.
/// No VT provided since it only works on 256-bit, 4 element vectors.
void DecodeVPERMMask(unsigned Imm, SmallVectorImpl<int> &ShuffleMask);

} // llvm namespace

#endif

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

#ifndef LLVM_LIB_TARGET_X86_UTILS_X86SHUFFLEDECODE_H
#define LLVM_LIB_TARGET_X86_UTILS_X86SHUFFLEDECODE_H

#include "llvm/ADT/SmallVector.h"

//===----------------------------------------------------------------------===//
//  Vector Mask Decoding
//===----------------------------------------------------------------------===//

namespace llvm {
template <typename T> class ArrayRef;
class MVT;

enum { SM_SentinelUndef = -1, SM_SentinelZero = -2 };

/// Decode a 128-bit INSERTPS instruction as a v4f32 shuffle mask.
void DecodeINSERTPSMask(unsigned Imm, SmallVectorImpl<int> &ShuffleMask);

// Insert the bottom Len elements from a second source into a vector starting at
// element Idx.
void DecodeInsertElementMask(MVT VT, unsigned Idx, unsigned Len,
                             SmallVectorImpl<int> &ShuffleMask);

/// Decode a MOVHLPS instruction as a v2f64/v4f32 shuffle mask.
/// i.e. <3,1> or <6,7,2,3>
void DecodeMOVHLPSMask(unsigned NElts, SmallVectorImpl<int> &ShuffleMask);

/// Decode a MOVLHPS instruction as a v2f64/v4f32 shuffle mask.
/// i.e. <0,2> or <0,1,4,5>
void DecodeMOVLHPSMask(unsigned NElts, SmallVectorImpl<int> &ShuffleMask);

void DecodeMOVSLDUPMask(MVT VT, SmallVectorImpl<int> &ShuffleMask);

void DecodeMOVSHDUPMask(MVT VT, SmallVectorImpl<int> &ShuffleMask);

void DecodeMOVDDUPMask(MVT VT, SmallVectorImpl<int> &ShuffleMask);

void DecodePSLLDQMask(MVT VT, unsigned Imm, SmallVectorImpl<int> &ShuffleMask);

void DecodePSRLDQMask(MVT VT, unsigned Imm, SmallVectorImpl<int> &ShuffleMask);

void DecodePALIGNRMask(MVT VT, unsigned Imm, SmallVectorImpl<int> &ShuffleMask);

void DecodeVALIGNMask(MVT VT, unsigned Imm, SmallVectorImpl<int> &ShuffleMask);

/// Decodes the shuffle masks for pshufd/pshufw/vpermilpd/vpermilps.
/// VT indicates the type of the vector allowing it to handle different
/// datatypes and vector widths.
void DecodePSHUFMask(MVT VT, unsigned Imm, SmallVectorImpl<int> &ShuffleMask);

/// Decodes the shuffle masks for pshufhw.
/// VT indicates the type of the vector allowing it to handle different
/// datatypes and vector widths.
void DecodePSHUFHWMask(MVT VT, unsigned Imm, SmallVectorImpl<int> &ShuffleMask);

/// Decodes the shuffle masks for pshuflw.
/// VT indicates the type of the vector allowing it to handle different
/// datatypes and vector widths.
void DecodePSHUFLWMask(MVT VT, unsigned Imm, SmallVectorImpl<int> &ShuffleMask);

/// Decodes a PSWAPD 3DNow! instruction.
void DecodePSWAPMask(MVT VT, SmallVectorImpl<int> &ShuffleMask);

/// Decodes the shuffle masks for shufp*.
/// VT indicates the type of the vector allowing it to handle different
/// datatypes and vector widths.
void DecodeSHUFPMask(MVT VT, unsigned Imm, SmallVectorImpl<int> &ShuffleMask);

/// Decodes the shuffle masks for unpckhps/unpckhpd and punpckh*.
/// VT indicates the type of the vector allowing it to handle different
/// datatypes and vector widths.
void DecodeUNPCKHMask(MVT VT, SmallVectorImpl<int> &ShuffleMask);

/// Decodes the shuffle masks for unpcklps/unpcklpd and punpckl*.
/// VT indicates the type of the vector allowing it to handle different
/// datatypes and vector widths.
void DecodeUNPCKLMask(MVT VT, SmallVectorImpl<int> &ShuffleMask);

/// Decodes a broadcast of the first element of a vector.
void DecodeVectorBroadcast(MVT DstVT, SmallVectorImpl<int> &ShuffleMask);

/// Decodes a broadcast of a subvector to a larger vector type.
void DecodeSubVectorBroadcast(MVT DstVT, MVT SrcVT,
                              SmallVectorImpl<int> &ShuffleMask);

/// Decode a PSHUFB mask from a raw array of constants such as from
/// BUILD_VECTOR.
void DecodePSHUFBMask(ArrayRef<uint64_t> RawMask,
                      SmallVectorImpl<int> &ShuffleMask);

/// Decode a BLEND immediate mask into a shuffle mask.
void DecodeBLENDMask(MVT VT, unsigned Imm, SmallVectorImpl<int> &ShuffleMask);

void DecodeVPERM2X128Mask(MVT VT, unsigned Imm,
                          SmallVectorImpl<int> &ShuffleMask);

/// Decode a shuffle packed values at 128-bit granularity
/// immediate mask into a shuffle mask.
void decodeVSHUF64x2FamilyMask(MVT VT, unsigned Imm,
                               SmallVectorImpl<int> &ShuffleMask);

/// Decodes the shuffle masks for VPERMQ/VPERMPD.
void DecodeVPERMMask(MVT VT, unsigned Imm, SmallVectorImpl<int> &ShuffleMask);

/// Decode a VPPERM mask from a raw array of constants such as from
/// BUILD_VECTOR.
/// This can only basic masks (permutes + zeros), not any of the other
/// operations that VPPERM can perform.
void DecodeVPPERMMask(ArrayRef<uint64_t> RawMask,
                      SmallVectorImpl<int> &ShuffleMask);

/// Decode a zero extension instruction as a shuffle mask.
void DecodeZeroExtendMask(MVT SrcScalarVT, MVT DstVT,
                          SmallVectorImpl<int> &ShuffleMask);

/// Decode a move lower and zero upper instruction as a shuffle mask.
void DecodeZeroMoveLowMask(MVT VT, SmallVectorImpl<int> &ShuffleMask);

/// Decode a scalar float move instruction as a shuffle mask.
void DecodeScalarMoveMask(MVT VT, bool IsLoad,
                          SmallVectorImpl<int> &ShuffleMask);

/// Decode a SSE4A EXTRQ instruction as a v16i8 shuffle mask.
void DecodeEXTRQIMask(int Len, int Idx,
                      SmallVectorImpl<int> &ShuffleMask);

/// Decode a SSE4A INSERTQ instruction as a v16i8 shuffle mask.
void DecodeINSERTQIMask(int Len, int Idx,
                        SmallVectorImpl<int> &ShuffleMask);

/// Decode a VPERMILPD/VPERMILPS variable mask from a raw array of constants.
void DecodeVPERMILPMask(MVT VT, ArrayRef<uint64_t> RawMask,
                        SmallVectorImpl<int> &ShuffleMask);

/// Decode a VPERMIL2PD/VPERMIL2PS variable mask from a raw array of constants.
void DecodeVPERMIL2PMask(MVT VT, unsigned M2Z, ArrayRef<uint64_t> RawMask,
                         SmallVectorImpl<int> &ShuffleMask);

/// Decode a VPERM W/D/Q/PS/PD mask from a raw array of constants.
void DecodeVPERMVMask(ArrayRef<uint64_t> RawMask,
                      SmallVectorImpl<int> &ShuffleMask);

/// Decode a VPERMT2 W/D/Q/PS/PD mask from a raw array of constants.
void DecodeVPERMV3Mask(ArrayRef<uint64_t> RawMask,
                      SmallVectorImpl<int> &ShuffleMask);
} // llvm namespace

#endif

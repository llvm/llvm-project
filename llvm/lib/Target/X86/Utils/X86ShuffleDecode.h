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
#include "llvm/ADT/ArrayRef.h"

//===----------------------------------------------------------------------===//
//  Vector Mask Decoding
//===----------------------------------------------------------------------===//

namespace llvm {
class Constant;
class MVT;

enum { SM_SentinelUndef = -1, SM_SentinelZero = -2 };

void DecodeINSERTPSMask(unsigned Imm, SmallVectorImpl<int> &ShuffleMask);

// <3,1> or <6,7,2,3>
void DecodeMOVHLPSMask(unsigned NElts, SmallVectorImpl<int> &ShuffleMask);

// <0,2> or <0,1,4,5>
void DecodeMOVLHPSMask(unsigned NElts, SmallVectorImpl<int> &ShuffleMask);

void DecodeMOVSLDUPMask(MVT VT, SmallVectorImpl<int> &ShuffleMask);

void DecodeMOVSHDUPMask(MVT VT, SmallVectorImpl<int> &ShuffleMask);

void DecodeMOVDDUPMask(MVT VT, SmallVectorImpl<int> &ShuffleMask);

void DecodePSLLDQMask(MVT VT, unsigned Imm, SmallVectorImpl<int> &ShuffleMask);

void DecodePSRLDQMask(MVT VT, unsigned Imm, SmallVectorImpl<int> &ShuffleMask);

void DecodePALIGNRMask(MVT VT, unsigned Imm, SmallVectorImpl<int> &ShuffleMask);

void DecodePSHUFMask(MVT VT, unsigned Imm, SmallVectorImpl<int> &ShuffleMask);

void DecodePSHUFHWMask(MVT VT, unsigned Imm, SmallVectorImpl<int> &ShuffleMask);

void DecodePSHUFLWMask(MVT, unsigned Imm, SmallVectorImpl<int> &ShuffleMask);

/// \brief Decodes a PSWAPD 3DNow! instruction.
void DecodePSWAPMask(MVT VT, SmallVectorImpl<int> &ShuffleMask);

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

/// \brief Decode a PSHUFB mask from an IR-level vector constant.
void DecodePSHUFBMask(const Constant *C, SmallVectorImpl<int> &ShuffleMask);

/// \brief Decode a PSHUFB mask from a raw array of constants such as from
/// BUILD_VECTOR.
void DecodePSHUFBMask(ArrayRef<uint64_t> RawMask,
                      SmallVectorImpl<int> &ShuffleMask);

/// \brief Decode a BLEND immediate mask into a shuffle mask.
void DecodeBLENDMask(MVT VT, unsigned Imm, SmallVectorImpl<int> &ShuffleMask);

void DecodeVPERM2X128Mask(MVT VT, unsigned Imm,
                          SmallVectorImpl<int> &ShuffleMask);

/// \brief Decode a shuffle packed values at 128-bit granularity
/// immediate mask into a shuffle mask.
void decodeVSHUF64x2FamilyMask(MVT VT, unsigned Imm,
                               SmallVectorImpl<int> &ShuffleMask);

/// DecodeVPERMMask - this decodes the shuffle masks for VPERMQ/VPERMPD.
/// No VT provided since it only works on 256-bit, 4 element vectors.
void DecodeVPERMMask(unsigned Imm, SmallVectorImpl<int> &ShuffleMask);

/// \brief Decode a VPERMILP variable mask from an IR-level vector constant.
void DecodeVPERMILPMask(const Constant *C, SmallVectorImpl<int> &ShuffleMask);

/// \brief Decode a zero extension instruction as a shuffle mask.
void DecodeZeroExtendMask(MVT SrcVT, MVT DstVT,
                          SmallVectorImpl<int> &ShuffleMask);

/// \brief Decode a move lower and zero upper instruction as a shuffle mask.
void DecodeZeroMoveLowMask(MVT VT, SmallVectorImpl<int> &ShuffleMask);

/// \brief Decode a scalar float move instruction as a shuffle mask.
void DecodeScalarMoveMask(MVT VT, bool IsLoad,
                          SmallVectorImpl<int> &ShuffleMask);

/// \brief Decode a SSE4A EXTRQ instruction as a v16i8 shuffle mask.
void DecodeEXTRQIMask(int Len, int Idx,
                      SmallVectorImpl<int> &ShuffleMask);

/// \brief Decode a SSE4A INSERTQ instruction as a v16i8 shuffle mask.
void DecodeINSERTQIMask(int Len, int Idx,
                        SmallVectorImpl<int> &ShuffleMask);

/// \brief Decode a VPERM W/D/Q/PS/PD mask from an IR-level vector constant.
void DecodeVPERMVMask(const Constant *C, MVT VT,
                      SmallVectorImpl<int> &ShuffleMask);

/// \brief Decode a VPERM W/D/Q/PS/PD mask from a raw array of constants.
void DecodeVPERMVMask(ArrayRef<uint64_t> RawMask,
                      SmallVectorImpl<int> &ShuffleMask);

/// \brief Decode a VPERMT2 W/D/Q/PS/PD mask from an IR-level vector constant.
void DecodeVPERMV3Mask(const Constant *C, MVT VT,
                      SmallVectorImpl<int> &ShuffleMask);

/// \brief Decode a VPERMT2 W/D/Q/PS/PD mask from a raw array of constants.
void DecodeVPERMV3Mask(ArrayRef<uint64_t> RawMask,
                      SmallVectorImpl<int> &ShuffleMask);
} // llvm namespace

#endif

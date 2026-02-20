//===-- X86InstrFragmentsSIMD.td - x86 SIMD ISA ------------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file provides pattern fragments useful for SIMD instructions.
//
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// MMX specific DAG Nodes.
//===----------------------------------------------------------------------===//

// Copies a 32-bit value from the low word of a MMX
// vector to a GPR.
def MMX_X86movd2w : SDNode<"X86ISD::MMX_MOVD2W", SDTypeProfile<1, 1,
                            [SDTCisVT<0, i32>, SDTCisVT<1, x86mmx>]>>;

// Copies a GPR into the low 32-bit word of a MMX vector
// and zero out the high word.
def MMX_X86movw2d : SDNode<"X86ISD::MMX_MOVW2D", SDTypeProfile<1, 1,
                            [SDTCisVT<0, x86mmx>, SDTCisVT<1, i32>]>>;

//===----------------------------------------------------------------------===//
// MMX Pattern Fragments
//===----------------------------------------------------------------------===//

def load_mmx : PatFrag<(ops node:$ptr), (x86mmx (load node:$ptr))>;

//===----------------------------------------------------------------------===//
// SSE specific DAG Nodes.
//===----------------------------------------------------------------------===//

def SDTX86VFCMP : SDTypeProfile<1, 3, [SDTCisFP<0>, SDTCisVec<0>,
                                       SDTCisSameAs<0, 1>, SDTCisSameAs<1, 2>,
                                       SDTCisVT<3, i8>]>;

/// Floating point max and min.
def X86fmin    : SDNode<"X86ISD::FMIN",      SDTFPBinOp>;
def X86fmax    : SDNode<"X86ISD::FMAX",      SDTFPBinOp>;

// Scalar intrinsic floating point max and min.
def X86fmins   : SDNode<"X86ISD::FMINS",     SDTFPBinOp>;
def X86fmaxs   : SDNode<"X86ISD::FMAXS",     SDTFPBinOp>;

// Commutative and Associative FMIN and FMAX.
def X86fminc    : SDNode<"X86ISD::FMINC", SDTFPBinOp,
    [SDNPCommutative, SDNPAssociative]>;
def X86fmaxc    : SDNode<"X86ISD::FMAXC", SDTFPBinOp,
    [SDNPCommutative, SDNPAssociative]>;

let IsStrictFP = true in {
  /// Floating point max and min.
  def X86strict_fmin : SDNode<"X86ISD::STRICT_FMIN", SDTFPBinOp,
                              [SDNPHasChain]>;
  def X86strict_fmax : SDNode<"X86ISD::STRICT_FMAX", SDTFPBinOp,
                              [SDNPHasChain]>;
}

def X86any_fmin : PatFrags<(ops node:$src1, node:$src2),
                           [(X86strict_fmin node:$src1, node:$src2),
                            (X86fmin node:$src1, node:$src2)]>;
def X86any_fmax : PatFrags<(ops node:$src1, node:$src2),
                           [(X86strict_fmax node:$src1, node:$src2),
                            (X86fmax node:$src1, node:$src2)]>;

// Bitwise logical AND of floating point values. This corresponds
// to X86::ANDPS or X86::ANDPD.
def X86fand    : SDNode<"X86ISD::FAND",      SDTFPBinOp,
                        [SDNPCommutative, SDNPAssociative]>;

// Bitwise logical OR of floating point values. This corresponds
// to X86::ORPS or X86::ORPD.
def X86for     : SDNode<"X86ISD::FOR",       SDTFPBinOp,
                        [SDNPCommutative, SDNPAssociative]>;

// Bitwise logical XOR of floating point values. This corresponds
// to X86::XORPS or X86::XORPD.
def X86fxor    : SDNode<"X86ISD::FXOR",      SDTFPBinOp,
                        [SDNPCommutative, SDNPAssociative]>;

//  Bitwise logical ANDNOT of floating point values. This
// corresponds to X86::ANDNPS or X86::ANDNPD.
def X86fandn   : SDNode<"X86ISD::FANDN",     SDTFPBinOp>;

// Floating point reciprocal-sqrt and reciprocal approximation.
// Note that these typically require refinement
// in order to obtain suitable precision.
def X86frsqrt  : SDNode<"X86ISD::FRSQRT",    SDTFPUnaryOp>;
def X86frcp    : SDNode<"X86ISD::FRCP",      SDTFPUnaryOp>;

// Floating point horizontal add/sub.
def X86fhadd   : SDNode<"X86ISD::FHADD",     SDTFPBinOp>;
def X86fhsub   : SDNode<"X86ISD::FHSUB",     SDTFPBinOp>;

// Integer horizontal add/sub.
def X86hadd    : SDNode<"X86ISD::HADD",      SDTIntBinOp>;
def X86hsub    : SDNode<"X86ISD::HSUB",      SDTIntBinOp>;
def X86hadds   : SDNode<"X86ISD::HADDS",     SDTIntBinOp>;
def X86hsubs   : SDNode<"X86ISD::HSUBS",     SDTIntBinOp>;

// X86 compare and logical compare instructions.
def X86comi    : SDNode<"X86ISD::COMI",      SDTX86FCmp>;
def X86ucomi   : SDNode<"X86ISD::UCOMI",     SDTX86FCmp>;

// X86 compare with Intrinsics similar to COMI.
def X86comi512       : SDNode<"X86ISD::COMX",      SDTX86FCmp>;
def X86ucomi512      : SDNode<"X86ISD::UCOMX",     SDTX86FCmp>;

def SDTX86Cmps : SDTypeProfile<1, 3, [SDTCisFP<0>, SDTCisSameAs<0, 1>,
                                      SDTCisSameAs<1, 2>, SDTCisVT<3, i8>]>;

// X86 FP SETCC, implemented with CMP{cc}SS/CMP{cc}SD.
// Operands are two FP values to compare; result is a mask of
// 0s or 1s.  Generally DTRT for C/C++ with NaNs.
def X86cmps    : SDNode<"X86ISD::FSETCC",    SDTX86Cmps>;

// Shuffle 16 8-bit values within a vector.
def X86pshufb  : SDNode<"X86ISD::PSHUFB",
                 SDTypeProfile<1, 2, [SDTCVecEltisVT<0, i8>, SDTCisSameAs<0,1>,
                                      SDTCisSameAs<0,2>]>>;

// Compute Sum of Absolute Differences.
def X86psadbw  : SDNode<"X86ISD::PSADBW",
                 SDTypeProfile<1, 2, [SDTCVecEltisVT<0, i64>,
                                      SDTCVecEltisVT<1, i8>,
                                      SDTCisSameSizeAs<0,1>,
                                      SDTCisSameAs<1,2>]>, [SDNPCommutative]>;

def SDTX86PSADBW : SDTypeProfile<1, 3, [SDTCVecEltisVT<0, i16>,
                                        SDTCVecEltisVT<1, i8>,
                                        SDTCisSameSizeAs<0,1>,
                                        SDTCisSameAs<1,2>, SDTCisVT<3, i8>]>;

// Compute Double Block Packed Sum-Absolute-Differences
def X86dbpsadbw : SDNode<"X86ISD::DBPSADBW", SDTX86PSADBW>;

// Bitwise Logical AND NOT of Packed FP values.
def X86andnp   : SDNode<"X86ISD::ANDNP",
                 SDTypeProfile<1, 2, [SDTCisVec<0>, SDTCisSameAs<0,1>,
                                      SDTCisSameAs<0,2>]>>;

def X86multishift   : SDNode<"X86ISD::MULTISHIFT",
                 SDTypeProfile<1, 2, [SDTCisVec<0>, SDTCisVec<1>,
                                      SDTCisSameAs<1,2>]>>;

// Extract an 8-bit value from a vector and zero extend it to
// i32, corresponds to X86::PEXTRB.
def X86pextrb  : SDNode<"X86ISD::PEXTRB",
                 SDTypeProfile<1, 2, [SDTCisVT<0, i32>, SDTCisVT<1, v16i8>,
                                      SDTCisVT<2, i8>]>>;

// Extract a 16-bit value from a vector and zero extend it to
// i32, corresponds to X86::PEXTRW.
def X86pextrw  : SDNode<"X86ISD::PEXTRW",
                 SDTypeProfile<1, 2, [SDTCisVT<0, i32>, SDTCisVT<1, v8i16>,
                                      SDTCisVT<2, i8>]>>;

// Insert the lower 8-bits of a 32-bit value to a vector,
// corresponds to X86::PINSRB.
def X86pinsrb  : SDNode<"X86ISD::PINSRB",
                 SDTypeProfile<1, 3, [SDTCisVT<0, v16i8>, SDTCisSameAs<0,1>,
                                      SDTCisVT<2, i32>, SDTCisVT<3, i8>]>>;

// Insert the lower 16-bits of a 32-bit value to a vector,
// corresponds to X86::PINSRW.
def X86pinsrw  : SDNode<"X86ISD::PINSRW",
                 SDTypeProfile<1, 3, [SDTCisVT<0, v8i16>, SDTCisSameAs<0,1>,
                                      SDTCisVT<2, i32>, SDTCisVT<3, i8>]>>;

// Insert any element of a 4 x float vector into any element
// of a destination 4 x floatvector.
def X86insertps : SDNode<"X86ISD::INSERTPS",
                 SDTypeProfile<1, 3, [SDTCisVT<0, v4f32>, SDTCisSameAs<0,1>,
                                      SDTCisVT<2, v4f32>, SDTCisVT<3, i8>]>>;

// Vector move to low scalar and zero higher vector elements.
def X86vzmovl  : SDNode<"X86ISD::VZEXT_MOVL",
                 SDTypeProfile<1, 1, [SDTCisSameAs<0,1>]>>;

// Load, scalar_to_vector, and zero extend.
def X86vzld  : SDNode<"X86ISD::VZEXT_LOAD", SDTLoad,
                      [SDNPHasChain, SDNPMayLoad, SDNPMemOperand]>;

// extract_vector_elt, store.
def X86vextractst  : SDNode<"X86ISD::VEXTRACT_STORE", SDTStore,
                     [SDNPHasChain, SDNPMayStore, SDNPMemOperand]>;

// scalar broadcast from memory.
def X86VBroadcastld  : SDNode<"X86ISD::VBROADCAST_LOAD", SDTLoad,
                      [SDNPHasChain, SDNPMayLoad, SDNPMemOperand]>;

// subvector broadcast from memory.
def X86SubVBroadcastld : SDNode<"X86ISD::SUBV_BROADCAST_LOAD", SDTLoad,
                         [SDNPHasChain, SDNPMayLoad, SDNPMemOperand]>;

def SDTVtrunc    : SDTypeProfile<1, 1, [SDTCisVec<0>, SDTCisVec<1>,
                                        SDTCisInt<0>, SDTCisInt<1>,
                                        SDTCisOpSmallerThanOp<0, 1>]>;
def SDTVmtrunc   : SDTypeProfile<1, 3, [SDTCisVec<0>, SDTCisVec<1>,
                                        SDTCisInt<0>, SDTCisInt<1>,
                                        SDTCisOpSmallerThanOp<0, 1>,
                                        SDTCisSameAs<0, 2>,
                                        SDTCVecEltisVT<3, i1>,
                                        SDTCisSameNumEltsAs<1, 3>]>;

// Vector integer truncate.
def X86vtrunc    : SDNode<"X86ISD::VTRUNC",   SDTVtrunc>;

// Vector integer truncate with unsigned/signed saturation.
def X86vtruncs   : SDNode<"X86ISD::VTRUNCS",  SDTVtrunc>;
def X86vtruncus  : SDNode<"X86ISD::VTRUNCUS", SDTVtrunc>;

// Masked version of the above. Used when less than a 128-bit result is
// produced since the mask only applies to the lower elements and can't
// be represented by a select.
// SRC, PASSTHRU, MASK
def X86vmtrunc   : SDNode<"X86ISD::VMTRUNC",   SDTVmtrunc>;
def X86vmtruncs  : SDNode<"X86ISD::VMTRUNCS",  SDTVmtrunc>;
def X86vmtruncus : SDNode<"X86ISD::VMTRUNCUS", SDTVmtrunc>;

// Vector FP extend.
def X86vfpext  : SDNode<"X86ISD::VFPEXT",
                        SDTypeProfile<1, 1, [SDTCisFP<0>, SDTCisVec<0>,
                                             SDTCisFP<1>, SDTCisVec<1>]>>;
def X86vfpextSAE  : SDNode<"X86ISD::VFPEXT_SAE",
                        SDTypeProfile<1, 1, [SDTCisFP<0>, SDTCisVec<0>,
                                             SDTCisFP<1>, SDTCisVec<1>,
                                             SDTCisOpSmallerThanOp<1, 0>]>>;

// Vector FP extend.
let IsStrictFP = true in
def X86strict_vfpext  : SDNode<"X86ISD::STRICT_VFPEXT",
                               SDTypeProfile<1, 1, [SDTCisFP<0>, SDTCisVec<0>,
                                                    SDTCisFP<1>, SDTCisVec<1>]>,
                                                    [SDNPHasChain]>;

def X86any_vfpext : PatFrags<(ops node:$src),
                              [(X86strict_vfpext node:$src),
                               (X86vfpext node:$src)]>;

// Vector FP round.
def X86vfpround: SDNode<"X86ISD::VFPROUND",
                        SDTypeProfile<1, 1, [SDTCisFP<0>, SDTCisVec<0>,
                                             SDTCisFP<1>, SDTCisVec<1>,
                                             SDTCisOpSmallerThanOp<0, 1>]>>;

// Convert TWO packed single data to one packed data
def X86vfpround2 : SDNode<"X86ISD::VFPROUND2",
                          SDTypeProfile<1, 2, [SDTCisFP<0>, SDTCisVec<0>,
                                               SDTCisFP<1>, SDTCisVec<1>,
                                               SDTCisSameAs<1, 2>,
                                               SDTCisOpSmallerThanOp<0, 1>]>>;

// Vector FP round.
let IsStrictFP = true in
def X86strict_vfpround: SDNode<"X86ISD::STRICT_VFPROUND",
                        SDTypeProfile<1, 1, [SDTCisFP<0>, SDTCisVec<0>,
                                             SDTCisFP<1>, SDTCisVec<1>,
                                             SDTCisOpSmallerThanOp<0, 1>]>,
                                             [SDNPHasChain]>;

def X86any_vfpround : PatFrags<(ops node:$src),
                              [(X86strict_vfpround node:$src),
                               (X86vfpround node:$src)]>;

def X86frounds   : SDNode<"X86ISD::VFPROUNDS",
                           SDTypeProfile<1, 2, [SDTCisFP<0>, SDTCisVec<0>,
                                                SDTCisSameAs<0, 1>,
                                                SDTCisFP<2>, SDTCisVec<2>,
                                                SDTCisSameSizeAs<0, 2>]>>;

def X86froundsRnd: SDNode<"X86ISD::VFPROUNDS_RND",
                        SDTypeProfile<1, 3, [SDTCisFP<0>, SDTCisVec<0>,
                                             SDTCisSameAs<0, 1>,
                                             SDTCisFP<2>, SDTCisVec<2>,
                                             SDTCisSameSizeAs<0, 2>,
                                             SDTCisVT<3, i32>]>>;

// Vector FP extend.
def X86fpexts     : SDNode<"X86ISD::VFPEXTS",
                        SDTypeProfile<1, 2, [SDTCisFP<0>, SDTCisVec<0>,
                                             SDTCisSameAs<0, 1>,
                                             SDTCisFP<2>, SDTCisVec<2>,
                                             SDTCisSameSizeAs<0, 2>]>>;
def X86fpextsSAE  : SDNode<"X86ISD::VFPEXTS_SAE",
                        SDTypeProfile<1, 2, [SDTCisFP<0>, SDTCisVec<0>,
                                             SDTCisSameAs<0, 1>,
                                             SDTCisFP<2>, SDTCisVec<2>,
                                             SDTCisSameSizeAs<0, 2>]>>;

// Masked version of VFPROUND. Used for v2f64->v4f32.
// SRC, PASSTHRU, MASK
def X86vmfpround: SDNode<"X86ISD::VMFPROUND",
                         SDTypeProfile<1, 3, [SDTCisFP<0>, SDTCisVec<0>,
                                              SDTCisFP<1>, SDTCisVec<1>,
                                              SDTCisSameAs<0, 2>,
                                              SDTCVecEltisVT<3, i1>,
                                              SDTCisSameNumEltsAs<1, 3>]>>;

def X86vshiftimm : SDTypeProfile<1, 2, [SDTCisVec<0>, SDTCisSameAs<0,1>,
                                        SDTCisVT<2, i8>, SDTCisInt<0>]>;

// 128-bit vector logical left / right shift
def X86vshldq  : SDNode<"X86ISD::VSHLDQ",    X86vshiftimm>;
def X86vshrdq  : SDNode<"X86ISD::VSRLDQ",    X86vshiftimm>;

// Vector integer comparisons.
def X86pcmpeq  : SDNode<"X86ISD::PCMPEQ", SDTIntBinOp, [SDNPCommutative]>;
def X86pcmpgt  : SDNode<"X86ISD::PCMPGT", SDTIntBinOp>;

// Vector packed double/float comparison.
def X86cmpp    : SDNode<"X86ISD::CMPP",      SDTX86VFCMP>;

// Vector packed double/float comparison.
let IsStrictFP = true in
def X86strict_cmpp : SDNode<"X86ISD::STRICT_CMPP", SDTX86VFCMP, [SDNPHasChain]>;

def X86any_cmpp    : PatFrags<(ops node:$src1, node:$src2, node:$src3),
                               [(X86strict_cmpp node:$src1, node:$src2, node:$src3),
                                (X86cmpp node:$src1, node:$src2, node:$src3)]>;

def X86CmpMaskCC :
      SDTypeProfile<1, 3, [SDTCisVec<0>, SDTCVecEltisVT<0, i1>,
                       SDTCisVec<1>, SDTCisSameAs<2, 1>,
                       SDTCisSameNumEltsAs<0, 1>, SDTCisVT<3, i8>]>;

def X86MaskCmpMaskCC :
      SDTypeProfile<1, 4, [SDTCisVec<0>, SDTCVecEltisVT<0, i1>,
                       SDTCisVec<1>, SDTCisSameAs<2, 1>,
                       SDTCisSameNumEltsAs<0, 1>, SDTCisVT<3, i8>, SDTCisSameAs<4, 0>]>;
def X86CmpMaskCCScalar :
      SDTypeProfile<1, 3, [SDTCisInt<0>, SDTCisFP<1>, SDTCisSameAs<1, 2>,
                           SDTCisVT<3, i8>]>;

// Vector comparison generating mask bits for fp and
// integer signed and unsigned data types.
def X86cmpm     : SDNode<"X86ISD::CMPM",     X86CmpMaskCC>;

// Vector mask comparison generating mask bits for FP values.
def X86cmpmm    : SDNode<"X86ISD::CMPMM",    X86MaskCmpMaskCC>;

// Vector comparison generating mask bits for fp and
// integer signed and unsigned data types.
let IsStrictFP = true in
def X86strict_cmpm : SDNode<"X86ISD::STRICT_CMPM", X86CmpMaskCC, [SDNPHasChain]>;

def X86any_cmpm    : PatFrags<(ops node:$src1, node:$src2, node:$src3),
                               [(X86strict_cmpm node:$src1, node:$src2, node:$src3),
                                (X86cmpm node:$src1, node:$src2, node:$src3)]>;

// Vector mask comparison with SAE for FP values.
def X86cmpmmSAE : SDNode<"X86ISD::CMPMM_SAE", X86MaskCmpMaskCC>;

// X86 FP SETCC, similar to FSETCC, but with output as an i1 mask and
// and a version with SAE.
def X86cmpms    : SDNode<"X86ISD::FSETCCM",   X86CmpMaskCCScalar>;
def X86cmpmsSAE : SDNode<"X86ISD::FSETCCM_SAE",   X86CmpMaskCCScalar>;

// v8i16 Horizontal minimum and position.
def X86phminpos: SDNode<"X86ISD::PHMINPOS",
                 SDTypeProfile<1, 1, [SDTCisVT<0, v8i16>, SDTCisVT<1, v8i16>]>>;

def X86vshiftuniform : SDTypeProfile<1, 2, [SDTCisVec<0>, SDTCisSameAs<0,1>,
                                            SDTCisVec<2>, SDTCisInt<0>,
                                            SDTCisInt<2>]>;

// Vector shift elements
def X86vshl    : SDNode<"X86ISD::VSHL", X86vshiftuniform>;
def X86vsrl    : SDNode<"X86ISD::VSRL", X86vshiftuniform>;
def X86vsra    : SDNode<"X86ISD::VSRA", X86vshiftuniform>;

def X86vshiftvariable : SDTypeProfile<1, 2, [SDTCisVec<0>, SDTCisSameAs<0,1>,
                                             SDTCisSameAs<0,2>, SDTCisInt<0>]>;

// Vector variable shift
def X86vshlv   : SDNode<"X86ISD::VSHLV", X86vshiftvariable>;
def X86vsrlv   : SDNode<"X86ISD::VSRLV", X86vshiftvariable>;
def X86vsrav   : SDNode<"X86ISD::VSRAV", X86vshiftvariable>;

// Vector shift elements by immediate
def X86vshli   : SDNode<"X86ISD::VSHLI", X86vshiftimm>;
def X86vsrli   : SDNode<"X86ISD::VSRLI", X86vshiftimm>;
def X86vsrai   : SDNode<"X86ISD::VSRAI", X86vshiftimm>;

// Shifts of mask registers.
def X86kshiftl : SDNode<"X86ISD::KSHIFTL",
                        SDTypeProfile<1, 2, [SDTCVecEltisVT<0, i1>,
                                             SDTCisSameAs<0, 1>,
                                             SDTCisVT<2, i8>]>>;
def X86kshiftr : SDNode<"X86ISD::KSHIFTR",
                        SDTypeProfile<1, 2, [SDTCVecEltisVT<0, i1>,
                                             SDTCisSameAs<0, 1>,
                                             SDTCisVT<2, i8>]>>;

// ADD for masks.
def X86kadd : SDNode<"X86ISD::KADD", SDTIntBinOp, [SDNPCommutative]>;

// Bit rotate by immediate
def X86vrotli  : SDNode<"X86ISD::VROTLI", X86vshiftimm>;
def X86vrotri  : SDNode<"X86ISD::VROTRI", X86vshiftimm>;

// XOP arithmetic/logical shifts.
def X86vpshl   : SDNode<"X86ISD::VPSHL", X86vshiftvariable>;
def X86vpsha   : SDNode<"X86ISD::VPSHA", X86vshiftvariable>;

// XOP signed/unsigned integer comparisons.
def X86vpcom   : SDNode<"X86ISD::VPCOM",
                        SDTypeProfile<1, 3, [SDTCisVec<0>, SDTCisSameAs<0,1>,
                                             SDTCisSameAs<0,2>,
                                             SDTCisVT<3, i8>, SDTCisInt<0>]>>;
def X86vpcomu  : SDNode<"X86ISD::VPCOMU",
                        SDTypeProfile<1, 3, [SDTCisVec<0>, SDTCisSameAs<0,1>,
                                             SDTCisSameAs<0,2>,
                                             SDTCisVT<3, i8>, SDTCisInt<0>]>>;

// XOP two source permutation.
def X86vpermil2 : SDNode<"X86ISD::VPERMIL2",
                        SDTypeProfile<1, 4, [SDTCisVec<0>, SDTCisSameAs<0,1>,
                                             SDTCisSameAs<0,2>,
                                             SDTCisFP<0>, SDTCisInt<3>,
                                             SDTCisSameNumEltsAs<0, 3>,
                                             SDTCisSameSizeAs<0,3>,
                                             SDTCisVT<4, i8>]>>;

// XOP packed permute bytes.
def X86vpperm : SDNode<"X86ISD::VPPERM",
                        SDTypeProfile<1, 3, [SDTCisVT<0, v16i8>, SDTCisSameAs<0,1>,
                                             SDTCisSameAs<0,2>, SDTCisSameAs<0, 3>]>>;

def SDTX86CmpPTest : SDTypeProfile<1, 2, [SDTCisVT<0, i32>,
                                          SDTCisVec<1>,
                                          SDTCisSameAs<2, 1>]>;

// Vector Multiply Packed UnsignedIntegers with Round and Scale.
def X86mulhrs  : SDNode<"X86ISD::MULHRS", SDTIntBinOp, [SDNPCommutative]>;

// Vector bitwise comparisons.
def X86ptest   : SDNode<"X86ISD::PTEST", SDTX86CmpPTest>;

// Vector packed fp sign bitwise comparisons.
def X86testp   : SDNode<"X86ISD::TESTP", SDTX86CmpPTest>;

// OR/AND test for masks.
def X86kortest : SDNode<"X86ISD::KORTEST", SDTX86CmpPTest>;
def X86ktest   : SDNode<"X86ISD::KTEST", SDTX86CmpPTest>;

// Vector sign bit extraction.
def X86movmsk : SDNode<"X86ISD::MOVMSK",
                        SDTypeProfile<1, 1, [SDTCisVT<0, i32>, SDTCisVec<1>]>>;

// X86 Select
def X86selects : SDNode<"X86ISD::SELECTS",
                        SDTypeProfile<1, 3, [SDTCisVT<1, v1i1>,
                                             SDTCisSameAs<0, 2>,
                                             SDTCisSameAs<2, 3>]>>;

// Vector multiply packed unsigned doubleword integers.
def X86pmuludq : SDNode<"X86ISD::PMULUDQ",
                        SDTypeProfile<1, 2, [SDTCVecEltisVT<0, i64>,
                                             SDTCisSameAs<0,1>,
                                             SDTCisSameAs<1,2>]>,
                                             [SDNPCommutative]>;

// Vector multiply packed signed doubleword integers.
def X86pmuldq  : SDNode<"X86ISD::PMULDQ",
                        SDTypeProfile<1, 2, [SDTCVecEltisVT<0, i64>,
                                             SDTCisSameAs<0,1>,
                                             SDTCisSameAs<1,2>]>,
                                             [SDNPCommutative]>;

/// SSE4A Extraction and Insertion.
def X86extrqi : SDNode<"X86ISD::EXTRQI",
                  SDTypeProfile<1, 3, [SDTCisVT<0, v2i64>, SDTCisSameAs<0,1>,
                                       SDTCisVT<2, i8>, SDTCisVT<3, i8>]>>;
def X86insertqi : SDNode<"X86ISD::INSERTQI",
                    SDTypeProfile<1, 4, [SDTCisVT<0, v2i64>, SDTCisSameAs<0,1>,
                                         SDTCisSameAs<1,2>, SDTCisVT<3, i8>,
                                         SDTCisVT<4, i8>]>>;

// Specific shuffle nodes - At some point ISD::VECTOR_SHUFFLE will always get
// translated into one of the target nodes below during lowering.
// Note: this is a work in progress...
def SDTShuff1Op : SDTypeProfile<1, 1, [SDTCisVec<0>, SDTCisSameAs<0,1>]>;
def SDTShuff2Op : SDTypeProfile<1, 2, [SDTCisVec<0>, SDTCisSameAs<0,1>,
                                SDTCisSameAs<0,2>]>;
def SDTShuff2OpFP : SDTypeProfile<1, 2, [SDTCisVec<0>, SDTCisFP<0>,
                                         SDTCisSameAs<0,1>, SDTCisSameAs<0,2>]>;

def SDTShuff2OpM : SDTypeProfile<1, 2, [SDTCisVec<0>, SDTCisSameAs<0,1>,
                                        SDTCisFP<0>, SDTCisInt<2>,
                                        SDTCisSameNumEltsAs<0,2>,
                                        SDTCisSameSizeAs<0,2>]>;
def SDTShuff2OpI : SDTypeProfile<1, 2, [SDTCisVec<0>,
                                 SDTCisSameAs<0,1>, SDTCisVT<2, i8>]>;
def SDTShuff3OpI : SDTypeProfile<1, 3, [SDTCisVec<0>, SDTCisSameAs<0,1>,
                                 SDTCisSameAs<0,2>, SDTCisVT<3, i8>]>;
def SDTFPBinOpImm: SDTypeProfile<1, 3, [SDTCisFP<0>, SDTCisVec<0>,
                                        SDTCisSameAs<0,1>,
                                        SDTCisSameAs<0,2>,
                                        SDTCisVT<3, i32>]>;
def SDTFPTernaryOpImm: SDTypeProfile<1, 4, [SDTCisFP<0>, SDTCisSameAs<0,1>,
                                            SDTCisSameAs<0,2>,
                                            SDTCisInt<3>,
                                            SDTCisSameSizeAs<0, 3>,
                                            SDTCisSameNumEltsAs<0, 3>,
                                            SDTCisVT<4, i32>]>;
def SDTFPUnaryOpImm: SDTypeProfile<1, 2, [SDTCisFP<0>,
                                          SDTCisSameAs<0,1>,
                                          SDTCisVT<2, i32>]>;

def SDTVBroadcast  : SDTypeProfile<1, 1, [SDTCisVec<0>]>;
def SDTVBroadcastm : SDTypeProfile<1, 1, [SDTCisVec<0>,
                                          SDTCisInt<0>, SDTCisInt<1>]>;

def SDTBlend : SDTypeProfile<1, 3, [SDTCisVec<0>, SDTCisSameAs<0,1>,
                             SDTCisSameAs<1,2>, SDTCisVT<3, i8>]>;

def SDTTernlog  : SDTypeProfile<1, 4, [SDTCisInt<0>, SDTCisVec<0>,
                                       SDTCisSameAs<0,1>, SDTCisSameAs<0,2>,
                                       SDTCisSameAs<0,3>, SDTCisVT<4, i8>]>;

def SDTFPBinOpRound : SDTypeProfile<1, 3, [      // fadd_round, fmul_round, etc.
  SDTCisSameAs<0, 1>, SDTCisSameAs<0, 2>, SDTCisFP<0>, SDTCisVT<3, i32>]>;

def SDTFPUnaryOpRound : SDTypeProfile<1, 2, [      // fsqrt_round, fgetexp_round, etc.
  SDTCisSameAs<0, 1>, SDTCisFP<0>, SDTCisVT<2, i32>]>;

def SDTFmaRound : SDTypeProfile<1, 4, [SDTCisSameAs<0,1>,
                           SDTCisSameAs<1,2>, SDTCisSameAs<1,3>,
                           SDTCisFP<0>, SDTCisVT<4, i32>]>;

def SDTFPToxIntSatOp
    : SDTypeProfile<1,
                    1, [SDTCisVec<0>, SDTCisVec<1>, SDTCisInt<0>, SDTCisFP<1>]>;

// Custom handling for FP_TO_xINT_SAT
def X86fp2sisat : SDNode<"X86ISD::FP_TO_SINT_SAT", SDTFPToxIntSatOp>;
def X86fp2uisat : SDNode<"X86ISD::FP_TO_UINT_SAT", SDTFPToxIntSatOp>;

// Intra-lane alignr.
def X86PAlignr : SDNode<"X86ISD::PALIGNR",
                        SDTypeProfile<1, 3, [SDTCVecEltisVT<0, i8>,
                                             SDTCisSameAs<0,1>,
                                             SDTCisSameAs<0,2>,
                                             SDTCisVT<3, i8>]>>;

// AVX512 inter-lane alignr.
def X86VAlign  : SDNode<"X86ISD::VALIGN", SDTShuff3OpI>;

// VBMI2 Concat & Shift.
def X86VShld   : SDNode<"X86ISD::VSHLD", SDTShuff3OpI>;
def X86VShrd   : SDNode<"X86ISD::VSHRD", SDTShuff3OpI>;

// Detect Conflicts Within a Vector
def X86Conflict : SDNode<"X86ISD::CONFLICT", SDTIntUnaryOp>;

def X86PShufd  : SDNode<"X86ISD::PSHUFD", SDTShuff2OpI>;
def X86PShufhw : SDNode<"X86ISD::PSHUFHW", SDTShuff2OpI>;
def X86PShuflw : SDNode<"X86ISD::PSHUFLW", SDTShuff2OpI>;

def X86Shufp   : SDNode<"X86ISD::SHUFP", SDTShuff3OpI>;

// Shuffle Packed Values at 128-bit granularity.
def X86Shuf128 : SDNode<"X86ISD::SHUF128", SDTShuff3OpI>;

def X86Movddup  : SDNode<"X86ISD::MOVDDUP", SDTShuff1Op>;
def X86Movshdup : SDNode<"X86ISD::MOVSHDUP", SDTShuff1Op>;
def X86Movsldup : SDNode<"X86ISD::MOVSLDUP", SDTShuff1Op>;

def X86Movsd : SDNode<"X86ISD::MOVSD",
                      SDTypeProfile<1, 2, [SDTCisVT<0, v2f64>,
                                           SDTCisVT<1, v2f64>,
                                           SDTCisVT<2, v2f64>]>>;
def X86Movss : SDNode<"X86ISD::MOVSS",
                      SDTypeProfile<1, 2, [SDTCisVT<0, v4f32>,
                                           SDTCisVT<1, v4f32>,
                                           SDTCisVT<2, v4f32>]>>;

def X86Movsh : SDNode<"X86ISD::MOVSH",
                      SDTypeProfile<1, 2, [SDTCisVT<0, v8f16>,
                                           SDTCisVT<1, v8f16>,
                                           SDTCisVT<2, v8f16>]>>;

def X86Movlhps : SDNode<"X86ISD::MOVLHPS",
                        SDTypeProfile<1, 2, [SDTCisVT<0, v4f32>,
                                             SDTCisVT<1, v4f32>,
                                             SDTCisVT<2, v4f32>]>>;
def X86Movhlps : SDNode<"X86ISD::MOVHLPS",
                        SDTypeProfile<1, 2, [SDTCisVT<0, v4f32>,
                                             SDTCisVT<1, v4f32>,
                                             SDTCisVT<2, v4f32>]>>;

def SDTPack : SDTypeProfile<1, 2, [SDTCisVec<0>, SDTCisInt<0>,
                                   SDTCisVec<1>, SDTCisInt<1>,
                                   SDTCisSameSizeAs<0,1>,
                                   SDTCisSameAs<1,2>,
                                   SDTCisOpSmallerThanOp<0, 1>]>;

// Several flavors of instructions with vector shuffle behaviors.
// Saturated signed/unnsigned packing.
def X86Packss : SDNode<"X86ISD::PACKSS", SDTPack>;
def X86Packus : SDNode<"X86ISD::PACKUS", SDTPack>;

def X86Unpckl : SDNode<"X86ISD::UNPCKL", SDTShuff2Op>;
def X86Unpckh : SDNode<"X86ISD::UNPCKH", SDTShuff2Op>;

// Multiply and Add Packed Integers.
def X86vpmaddubsw  : SDNode<"X86ISD::VPMADDUBSW",
                            SDTypeProfile<1, 2, [SDTCVecEltisVT<0, i16>,
                                                 SDTCVecEltisVT<1, i8>,
                                                 SDTCisSameSizeAs<0,1>,
                                                 SDTCisSameAs<1,2>]>>;
def X86vpmaddwd    : SDNode<"X86ISD::VPMADDWD",
                            SDTypeProfile<1, 2, [SDTCVecEltisVT<0, i32>,
                                                 SDTCVecEltisVT<1, i16>,
                                                 SDTCisSameSizeAs<0,1>,
                                                 SDTCisSameAs<1,2>]>,
                            [SDNPCommutative]>;

def X86VPermilpv  : SDNode<"X86ISD::VPERMILPV", SDTShuff2OpM>;
def X86VPermilpi  : SDNode<"X86ISD::VPERMILPI", SDTShuff2OpI>;

// Variable Permute (VPERM).
// Res = VPERMV MaskV, V0
def X86VPermv     : SDNode<"X86ISD::VPERMV",
                           SDTypeProfile<1, 2, [SDTCisVec<0>, SDTCisInt<1>,
                                                SDTCisSameNumEltsAs<0,1>,
                                                SDTCisSameSizeAs<0,1>,
                                                SDTCisSameAs<0,2>]>>;

def X86VPermi     : SDNode<"X86ISD::VPERMI",    SDTShuff2OpI>;

// 3-op Variable Permute (VPERMT2).
// Res = VPERMV3 V0, MaskV, V1
def X86VPermt2     : SDNode<"X86ISD::VPERMV3",
                    SDTypeProfile<1, 3, [SDTCisVec<0>,
                                         SDTCisSameAs<0,1>, SDTCisInt<2>,
                                         SDTCisVec<2>, SDTCisSameNumEltsAs<0, 2>,
                                         SDTCisSameSizeAs<0,2>,
                                         SDTCisSameAs<0,3>]>, []>;

// Bitwise ternary logic.
def X86vpternlog  : SDNode<"X86ISD::VPTERNLOG", SDTTernlog>;

def X86VPerm2x128 : SDNode<"X86ISD::VPERM2X128", SDTShuff3OpI>;

// Fix Up Special Packed Float32/64 values.
def X86VFixupimm     : SDNode<"X86ISD::VFIXUPIMM", SDTFPTernaryOpImm>;
def X86VFixupimmSAE  : SDNode<"X86ISD::VFIXUPIMM_SAE", SDTFPTernaryOpImm>;
def X86VFixupimms    : SDNode<"X86ISD::VFIXUPIMMS", SDTFPTernaryOpImm>;
def X86VFixupimmSAEs : SDNode<"X86ISD::VFIXUPIMMS_SAE", SDTFPTernaryOpImm>;

// Range Restriction Calculation For Packed Pairs of Float32/64 values.
def X86VRange      : SDNode<"X86ISD::VRANGE",        SDTFPBinOpImm>;
def X86VRangeSAE   : SDNode<"X86ISD::VRANGE_SAE",    SDTFPBinOpImm>;
def X86Ranges      : SDNode<"X86ISD::VRANGES",       SDTFPBinOpImm>;
def X86RangesSAE    : SDNode<"X86ISD::VRANGES_SAE",  SDTFPBinOpImm>;

// Reduce - Perform Reduction Transformation on scalar\packed FP.
def X86VReduce     : SDNode<"X86ISD::VREDUCE",       SDTFPUnaryOpImm>;
def X86VReduceSAE  : SDNode<"X86ISD::VREDUCE_SAE",   SDTFPUnaryOpImm>;
def X86Reduces     : SDNode<"X86ISD::VREDUCES",      SDTFPBinOpImm>;
def X86ReducesSAE  : SDNode<"X86ISD::VREDUCES_SAE",  SDTFPBinOpImm>;

// RndScale - Round FP Values To Include A Given Number Of Fraction Bits.
// Also used by the legacy (V)ROUND intrinsics where we mask out the
// scaling part of the immediate.
def X86VRndScale   : SDNode<"X86ISD::VRNDSCALE",     SDTFPUnaryOpImm>;
def X86VRndScaleSAE: SDNode<"X86ISD::VRNDSCALE_SAE", SDTFPUnaryOpImm>;
def X86RndScales   : SDNode<"X86ISD::VRNDSCALES",    SDTFPBinOpImm>;
def X86RndScalesSAE: SDNode<"X86ISD::VRNDSCALES_SAE",SDTFPBinOpImm>;

// RndScale - Round FP Values To Include A Given Number Of Fraction Bits.
// Also used by the legacy (V)ROUND intrinsics where we mask out the
// scaling part of the immediate.
let IsStrictFP = true in
def X86strict_VRndScale : SDNode<"X86ISD::STRICT_VRNDSCALE", SDTFPUnaryOpImm,
                                  [SDNPHasChain]>;

def X86any_VRndScale    : PatFrags<(ops node:$src1, node:$src2),
                                    [(X86strict_VRndScale node:$src1, node:$src2),
                                    (X86VRndScale node:$src1, node:$src2)]>;

// Extract Normalized Mantissas.
def X86VGetMant    : SDNode<"X86ISD::VGETMANT",      SDTFPUnaryOpImm>;
def X86VGetMantSAE : SDNode<"X86ISD::VGETMANT_SAE",  SDTFPUnaryOpImm>;
def X86GetMants    : SDNode<"X86ISD::VGETMANTS",     SDTFPBinOpImm>;
def X86GetMantsSAE : SDNode<"X86ISD::VGETMANTS_SAE", SDTFPBinOpImm>;

// Tests Types Of a FP Values for packed types.
def X86Vfpclass    : SDNode<"X86ISD::VFPCLASS",
                       SDTypeProfile<1, 2, [SDTCVecEltisVT<0, i1>,
                                            SDTCisFP<1>,
                                            SDTCisSameNumEltsAs<0,1>,
                                            SDTCisVT<2, i32>]>, []>;

// Tests Types Of a FP Values for scalar types.
def X86Vfpclasss   : SDNode<"X86ISD::VFPCLASSS",
                       SDTypeProfile<1, 2, [SDTCisVT<0, v1i1>,
                                            SDTCisFP<1>, SDTCisVT<2, i32>]>,[]>;

// Broadcast (splat) scalar or element 0 of a vector. If the operand is
// a vector, this node may change the vector length as part of the splat.
def X86VBroadcast : SDNode<"X86ISD::VBROADCAST", SDTVBroadcast>;

// Broadcast mask to vector.
def X86VBroadcastm : SDNode<"X86ISD::VBROADCASTM", SDTVBroadcastm>;

// Blend where the selector is an immediate.
def X86Blendi    : SDNode<"X86ISD::BLENDI",   SDTBlend>;

// Dynamic (non-constant condition) vector blend where only the sign bits
// of the condition elements are used. This is used to enforce that the
// condition mask is not valid for generic VSELECT optimizations. This
// is also used to implement the intrinsics.
// Operands are in VSELECT order: MASK, TRUE, FALSE
def X86Blendv    : SDNode<"X86ISD::BLENDV",
                     SDTypeProfile<1, 3, [SDTCisVec<0>, SDTCisInt<1>,
                                          SDTCisSameAs<0, 2>,
                                          SDTCisSameAs<2, 3>,
                                          SDTCisSameNumEltsAs<0, 1>,
                                          SDTCisSameSizeAs<0, 1>]>>;

// Combined add and sub on an FP vector.
def X86Addsub    : SDNode<"X86ISD::ADDSUB", SDTFPBinOp>;

//  FP vector ops with rounding mode.
def X86faddRnd   : SDNode<"X86ISD::FADD_RND",  SDTFPBinOpRound>;
def X86fadds     : SDNode<"X86ISD::FADDS",     SDTFPBinOp>;
def X86faddRnds  : SDNode<"X86ISD::FADDS_RND", SDTFPBinOpRound>;
def X86fsubRnd   : SDNode<"X86ISD::FSUB_RND",  SDTFPBinOpRound>;
def X86fsubs     : SDNode<"X86ISD::FSUBS",     SDTFPBinOp>;
def X86fsubRnds  : SDNode<"X86ISD::FSUBS_RND", SDTFPBinOpRound>;
def X86fmulRnd   : SDNode<"X86ISD::FMUL_RND",  SDTFPBinOpRound>;
def X86fmuls     : SDNode<"X86ISD::FMULS",     SDTFPBinOp>;
def X86fmulRnds  : SDNode<"X86ISD::FMULS_RND", SDTFPBinOpRound>;
def X86fdivRnd   : SDNode<"X86ISD::FDIV_RND",  SDTFPBinOpRound>;
def X86fdivs     : SDNode<"X86ISD::FDIVS",     SDTFPBinOp>;
def X86fdivRnds  : SDNode<"X86ISD::FDIVS_RND", SDTFPBinOpRound>;
def X86fmaxSAE   : SDNode<"X86ISD::FMAX_SAE",  SDTFPBinOp>;
def X86fmaxSAEs  : SDNode<"X86ISD::FMAXS_SAE", SDTFPBinOp>;
def X86fminSAE   : SDNode<"X86ISD::FMIN_SAE",  SDTFPBinOp>;
def X86fminSAEs  : SDNode<"X86ISD::FMINS_SAE", SDTFPBinOp>;

// FP Scale.
def X86scalef    : SDNode<"X86ISD::SCALEF",         SDTFPBinOp>;
def X86scalefRnd : SDNode<"X86ISD::SCALEF_RND",     SDTFPBinOpRound>;
def X86scalefs   : SDNode<"X86ISD::SCALEFS",        SDTFPBinOp>;
def X86scalefsRnd: SDNode<"X86ISD::SCALEFS_RND",    SDTFPBinOpRound>;

//  FP vector ops with rounding mode.
def X86fsqrtRnd     : SDNode<"X86ISD::FSQRT_RND",   SDTFPUnaryOpRound>;
def X86fsqrts       : SDNode<"X86ISD::FSQRTS", SDTFPBinOp>;
def X86fsqrtRnds    : SDNode<"X86ISD::FSQRTS_RND", SDTFPBinOpRound>;

// FP vector get exponent.
def X86fgetexp      : SDNode<"X86ISD::FGETEXP", SDTFPUnaryOp>;
def X86fgetexpSAE   : SDNode<"X86ISD::FGETEXP_SAE", SDTFPUnaryOp>;
def X86fgetexps     : SDNode<"X86ISD::FGETEXPS", SDTFPBinOp>;
def X86fgetexpSAEs  : SDNode<"X86ISD::FGETEXPS_SAE", SDTFPBinOp>;

// FMA nodes.
// We use the target independent ISD::FMA for the non-inverted case.
def X86Fnmadd   : SDNode<"X86ISD::FNMADD",   SDTFPTernaryOp, [SDNPCommutative]>;
def X86Fmsub    : SDNode<"X86ISD::FMSUB",    SDTFPTernaryOp, [SDNPCommutative]>;
def X86Fnmsub   : SDNode<"X86ISD::FNMSUB",   SDTFPTernaryOp, [SDNPCommutative]>;
def X86Fmaddsub : SDNode<"X86ISD::FMADDSUB", SDTFPTernaryOp, [SDNPCommutative]>;
def X86Fmsubadd : SDNode<"X86ISD::FMSUBADD", SDTFPTernaryOp, [SDNPCommutative]>;

let IsStrictFP = true in {
  // Strict FMA nodes.
  def X86strict_Fnmadd : SDNode<"X86ISD::STRICT_FNMADD", SDTFPTernaryOp,
                                [SDNPCommutative, SDNPHasChain]>;
  def X86strict_Fmsub  : SDNode<"X86ISD::STRICT_FMSUB", SDTFPTernaryOp,
                                [SDNPCommutative, SDNPHasChain]>;
  def X86strict_Fnmsub : SDNode<"X86ISD::STRICT_FNMSUB", SDTFPTernaryOp,
                                [SDNPCommutative, SDNPHasChain]>;
}

def X86any_Fnmadd : PatFrags<(ops node:$src1, node:$src2, node:$src3),
                             [(X86strict_Fnmadd node:$src1, node:$src2, node:$src3),
                              (X86Fnmadd node:$src1, node:$src2, node:$src3)]>;

def X86any_Fmsub : PatFrags<(ops node:$src1, node:$src2, node:$src3),
                            [(X86strict_Fmsub node:$src1, node:$src2, node:$src3),
                             (X86Fmsub node:$src1, node:$src2, node:$src3)]>;

def X86any_Fnmsub : PatFrags<(ops node:$src1, node:$src2, node:$src3),
                             [(X86strict_Fnmsub node:$src1, node:$src2, node:$src3),
                              (X86Fnmsub node:$src1, node:$src2, node:$src3)]>;

// FMA with rounding mode.
def X86FmaddRnd     : SDNode<"X86ISD::FMADD_RND",     SDTFmaRound, [SDNPCommutative]>;
def X86FnmaddRnd    : SDNode<"X86ISD::FNMADD_RND",    SDTFmaRound, [SDNPCommutative]>;
def X86FmsubRnd     : SDNode<"X86ISD::FMSUB_RND",     SDTFmaRound, [SDNPCommutative]>;
def X86FnmsubRnd    : SDNode<"X86ISD::FNMSUB_RND",    SDTFmaRound, [SDNPCommutative]>;
def X86FmaddsubRnd  : SDNode<"X86ISD::FMADDSUB_RND",  SDTFmaRound, [SDNPCommutative]>;
def X86FmsubaddRnd  : SDNode<"X86ISD::FMSUBADD_RND",  SDTFmaRound, [SDNPCommutative]>;

// For avx512-vp2intersect
def X86vp2intersect : SDNode<"X86ISD::VP2INTERSECT",
                              SDTypeProfile<1, 2, [SDTCisVT<0, untyped>,
                                                   SDTCisVec<1>, SDTCisSameAs<1, 2>]>>;

def SDTIFma : SDTypeProfile<1, 3, [SDTCisInt<0>, SDTCisSameAs<0,1>,
                           SDTCisSameAs<1,2>, SDTCisSameAs<1,3>]>;

// AVX512IFMA multiply and add.
// NOTE: These are different than the instruction and perform
// op0 x op1 + op2.
def x86vpmadd52l     : SDNode<"X86ISD::VPMADD52L",     SDTIFma, [SDNPCommutative]>;
def x86vpmadd52h     : SDNode<"X86ISD::VPMADD52H",     SDTIFma, [SDNPCommutative]>;

// AVX512-FP16 complex addition and multiplication.
def x86vfmaddc       : SDNode<"X86ISD::VFMADDC",       SDTFPTernaryOp,  [SDNPCommutative]>;
def x86vfmaddcRnd    : SDNode<"X86ISD::VFMADDC_RND",   SDTFmaRound,     [SDNPCommutative]>;
def x86vfcmaddc      : SDNode<"X86ISD::VFCMADDC",      SDTFPTernaryOp>;
def x86vfcmaddcRnd   : SDNode<"X86ISD::VFCMADDC_RND",  SDTFmaRound>;
def x86vfmulc        : SDNode<"X86ISD::VFMULC",        SDTFPBinOp,      [SDNPCommutative]>;
def x86vfmulcRnd     : SDNode<"X86ISD::VFMULC_RND",    SDTFPBinOpRound, [SDNPCommutative]>;
def x86vfcmulc       : SDNode<"X86ISD::VFCMULC",       SDTFPBinOp>;
def x86vfcmulcRnd    : SDNode<"X86ISD::VFCMULC_RND",   SDTFPBinOpRound>;

def x86vfmaddcSh     : SDNode<"X86ISD::VFMADDCSH",     SDTFPTernaryOp,  [SDNPCommutative]>;
def x86vfcmaddcSh    : SDNode<"X86ISD::VFCMADDCSH",    SDTFPTernaryOp>;
def x86vfmulcSh      : SDNode<"X86ISD::VFMULCSH",      SDTFPBinOp,      [SDNPCommutative]>;
def x86vfcmulcSh     : SDNode<"X86ISD::VFCMULCSH",     SDTFPBinOp>;
def x86vfmaddcShRnd  : SDNode<"X86ISD::VFMADDCSH_RND", SDTFmaRound,     [SDNPCommutative]>;
def x86vfcmaddcShRnd : SDNode<"X86ISD::VFCMADDCSH_RND",SDTFmaRound>;
def x86vfmulcShRnd   : SDNode<"X86ISD::VFMULCSH_RND",  SDTFPBinOpRound, [SDNPCommutative]>;
def x86vfcmulcShRnd  : SDNode<"X86ISD::VFCMULCSH_RND", SDTFPBinOpRound>;

// AVX-512 reciprocal approximations with a little more precision.
def X86rsqrt14   : SDNode<"X86ISD::RSQRT14",  SDTFPUnaryOp>;
def X86rsqrt14s  : SDNode<"X86ISD::RSQRT14S", SDTFPBinOp>;
def X86rcp14     : SDNode<"X86ISD::RCP14",    SDTFPUnaryOp>;
def X86rcp14s    : SDNode<"X86ISD::RCP14S",   SDTFPBinOp>;

// VNNI
def SDTVnni : SDTypeProfile<1, 3, [SDTCisVec<0>, SDTCisSameAs<0,1>,
                                   SDTCisSameAs<1,2>, SDTCisSameAs<1,3>]>;

// VNNI
def X86Vpdpbusd  : SDNode<"X86ISD::VPDPBUSD", SDTVnni>;
def X86Vpdpbusds : SDNode<"X86ISD::VPDPBUSDS", SDTVnni>;
def X86Vpdpwssd  : SDNode<"X86ISD::VPDPWSSD", SDTVnni>;
def X86Vpdpwssds : SDNode<"X86ISD::VPDPWSSDS", SDTVnni>;

// Compress and expand.
def X86compress: SDNode<"X86ISD::COMPRESS", SDTypeProfile<1, 3,
                              [SDTCisSameAs<0, 1>, SDTCisVec<1>,
                               SDTCisSameAs<0, 2>, SDTCVecEltisVT<3, i1>,
                               SDTCisSameNumEltsAs<0, 3>]>, []>;
def X86expand  : SDNode<"X86ISD::EXPAND", SDTypeProfile<1, 3,
                              [SDTCisSameAs<0, 1>, SDTCisVec<1>,
                               SDTCisSameAs<0, 2>, SDTCVecEltisVT<3, i1>,
                               SDTCisSameNumEltsAs<0, 3>]>, []>;

// Bits shuffle
// vpshufbitqmb
def X86Vpshufbitqmb : SDNode<"X86ISD::VPSHUFBITQMB",
                             SDTypeProfile<1, 2, [SDTCisVec<0>, SDTCisVec<1>,
                                                  SDTCisSameAs<1,2>,
                                                  SDTCVecEltisVT<0,i1>,
                                                  SDTCisSameNumEltsAs<0,1>]>>;

def SDTintToFP: SDTypeProfile<1, 2, [SDTCisVec<0>, SDTCisFP<0>,
                                     SDTCisSameAs<0,1>, SDTCisInt<2>]>;
def SDTintToFPRound: SDTypeProfile<1, 3, [SDTCisVec<0>, SDTCisFP<0>,
                                          SDTCisSameAs<0,1>, SDTCisInt<2>,
                                          SDTCisVT<3, i32>]>;

def SDTFloatToInt: SDTypeProfile<1, 1, [SDTCisVec<0>, SDTCisVec<1>,
                                        SDTCisInt<0>, SDTCisFP<1>]>;
def SDTFloatToIntRnd: SDTypeProfile<1, 2, [SDTCisVec<0>, SDTCisVec<1>,
                                           SDTCisInt<0>, SDTCisFP<1>,
                                           SDTCisVT<2, i32>]>;
def SDTSFloatToInt: SDTypeProfile<1, 1, [SDTCisInt<0>, SDTCisFP<1>,
                                         SDTCisVec<1>]>;
def SDTSFloatToIntRnd: SDTypeProfile<1, 2, [SDTCisInt<0>, SDTCisFP<1>,
                                            SDTCisVec<1>, SDTCisVT<2, i32>]>;

def SDTVintToFP: SDTypeProfile<1, 1, [SDTCisVec<0>, SDTCisVec<1>,
                                      SDTCisFP<0>, SDTCisInt<1>]>;
def SDTVintToFPRound: SDTypeProfile<1, 2, [SDTCisVec<0>, SDTCisVec<1>,
                                           SDTCisFP<0>, SDTCisInt<1>,
                                           SDTCisVT<2, i32>]>;

// Convert Unsigned/Integer to Floating-Point Value with rounding mode.
def X86SintToFp     : SDNode<"X86ISD::SCALAR_SINT_TO_FP",      SDTintToFP>;
def X86SintToFpRnd  : SDNode<"X86ISD::SCALAR_SINT_TO_FP_RND",  SDTintToFPRound>;
def X86UintToFp     : SDNode<"X86ISD::SCALAR_UINT_TO_FP",      SDTintToFP>;
def X86UintToFpRnd  : SDNode<"X86ISD::SCALAR_UINT_TO_FP_RND",  SDTintToFPRound>;

// Scalar float/double to signed/unsigned integer with truncation.
def X86cvtts2Int  : SDNode<"X86ISD::CVTTS2SI",  SDTSFloatToInt>;
def X86cvtts2UInt : SDNode<"X86ISD::CVTTS2UI",  SDTSFloatToInt>;
def X86cvtts2IntSAE  : SDNode<"X86ISD::CVTTS2SI_SAE",  SDTSFloatToInt>;
def X86cvtts2UIntSAE : SDNode<"X86ISD::CVTTS2UI_SAE",  SDTSFloatToInt>;

// Scalar float/double to signed/unsigned integer.
def X86cvts2si  : SDNode<"X86ISD::CVTS2SI", SDTSFloatToInt>;
def X86cvts2usi : SDNode<"X86ISD::CVTS2UI", SDTSFloatToInt>;
def X86cvts2siRnd  : SDNode<"X86ISD::CVTS2SI_RND", SDTSFloatToIntRnd>;
def X86cvts2usiRnd : SDNode<"X86ISD::CVTS2UI_RND", SDTSFloatToIntRnd>;

// Scalar float/double to signed/unsigned integer with saturation.
def X86cvttss2Int     : SDNode<"X86ISD::CVTTS2SIS",  SDTSFloatToInt>;
def X86cvttss2UInt    : SDNode<"X86ISD::CVTTS2UIS",  SDTSFloatToInt>;
def X86cvttss2IntSAE  : SDNode<"X86ISD::CVTTS2SIS_SAE",  SDTSFloatToInt>;
def X86cvttss2UIntSAE : SDNode<"X86ISD::CVTTS2UIS_SAE",  SDTSFloatToInt>;

// Vector with rounding mode

// Convert Unsigned/Integer to Floating-Point Value with rounding mode.
def X86VSintToFpRnd   : SDNode<"X86ISD::SINT_TO_FP_RND",  SDTVintToFPRound>;
def X86VUintToFpRnd   : SDNode<"X86ISD::UINT_TO_FP_RND",  SDTVintToFPRound>;

// Saturation enabled Vector float/double to signed/unsigned
// integer with truncation.
def X86cvttp2sisSAE    : SDNode<"X86ISD::CVTTP2SIS_SAE", SDTFloatToInt>;
def X86cvttp2uisSAE    : SDNode<"X86ISD::CVTTP2UIS_SAE", SDTFloatToInt>;
def X86cvttp2sis       : SDNode<"X86ISD::CVTTP2SIS",  SDTFloatToInt>;
def X86cvttp2uis       : SDNode<"X86ISD::CVTTP2UIS",  SDTFloatToInt>;

// Vector without rounding mode

// Vector float/double to signed/unsigned integer with truncation.
def X86cvttp2si      : SDNode<"X86ISD::CVTTP2SI",  SDTFloatToInt>;
def X86cvttp2ui      : SDNode<"X86ISD::CVTTP2UI",  SDTFloatToInt>;
def X86cvttp2siSAE   : SDNode<"X86ISD::CVTTP2SI_SAE", SDTFloatToInt>;
def X86cvttp2uiSAE   : SDNode<"X86ISD::CVTTP2UI_SAE", SDTFloatToInt>;

let IsStrictFP = true in {
  // Vector float/double to signed/unsigned integer with truncation.
  def X86strict_cvttp2si
      : SDNode<"X86ISD::STRICT_CVTTP2SI", SDTFloatToInt, [SDNPHasChain]>;
  def X86strict_cvttp2ui
      : SDNode<"X86ISD::STRICT_CVTTP2UI", SDTFloatToInt, [SDNPHasChain]>;
}

def X86any_cvttp2si : PatFrags<(ops node:$src),
                               [(X86strict_cvttp2si node:$src),
                                (X86cvttp2si node:$src)]>;
def X86any_cvttp2ui : PatFrags<(ops node:$src),
                               [(X86strict_cvttp2ui node:$src),
                                (X86cvttp2ui node:$src)]>;

// Vector signed/unsigned integer to float/double.
def X86VSintToFP      : SDNode<"X86ISD::CVTSI2P",  SDTVintToFP>;
def X86VUintToFP      : SDNode<"X86ISD::CVTUI2P",  SDTVintToFP>;

let IsStrictFP = true in {
  // Vector signed/unsigned integer to float/double.
  def X86strict_VSintToFP
      : SDNode<"X86ISD::STRICT_CVTSI2P", SDTVintToFP, [SDNPHasChain]>;
  def X86strict_VUintToFP
      : SDNode<"X86ISD::STRICT_CVTUI2P", SDTVintToFP, [SDNPHasChain]>;
}

def X86any_VSintToFP : PatFrags<(ops node:$src),
                                [(X86strict_VSintToFP node:$src),
                                 (X86VSintToFP node:$src)]>;
def X86any_VUintToFP : PatFrags<(ops node:$src),
                                [(X86strict_VUintToFP node:$src),
                                 (X86VUintToFP node:$src)]>;


// Vector float/double to signed/unsigned integer.
def X86cvtp2Int      : SDNode<"X86ISD::CVTP2SI",  SDTFloatToInt>;
def X86cvtp2UInt     : SDNode<"X86ISD::CVTP2UI",  SDTFloatToInt>;
def X86cvtp2IntRnd   : SDNode<"X86ISD::CVTP2SI_RND",  SDTFloatToIntRnd>;
def X86cvtp2UIntRnd  : SDNode<"X86ISD::CVTP2UI_RND",  SDTFloatToIntRnd>;

// Masked versions of above
def SDTMVintToFP: SDTypeProfile<1, 3, [SDTCisVec<0>, SDTCisVec<1>,
                                       SDTCisFP<0>, SDTCisInt<1>,
                                       SDTCisSameAs<0, 2>,
                                       SDTCVecEltisVT<3, i1>,
                                       SDTCisSameNumEltsAs<1, 3>]>;
def SDTMFloatToInt: SDTypeProfile<1, 3, [SDTCisVec<0>, SDTCisVec<1>,
                                         SDTCisInt<0>, SDTCisFP<1>,
                                         SDTCisSameSizeAs<0, 1>,
                                         SDTCisSameAs<0, 2>,
                                         SDTCVecEltisVT<3, i1>,
                                         SDTCisSameNumEltsAs<1, 3>]>;

def X86VMSintToFP    : SDNode<"X86ISD::MCVTSI2P",  SDTMVintToFP>;
def X86VMUintToFP    : SDNode<"X86ISD::MCVTUI2P",  SDTMVintToFP>;

// Masked versions of CVTTS2[US]IS. Used for v2f64->v4f32.
// SRC, PASSTHRU, MASK
def X86mcvtp2Int     : SDNode<"X86ISD::MCVTP2SI",  SDTMFloatToInt>;
def X86mcvtp2UInt    : SDNode<"X86ISD::MCVTP2UI",  SDTMFloatToInt>;
def X86mcvttp2si     : SDNode<"X86ISD::MCVTTP2SI", SDTMFloatToInt>;
def X86mcvttp2ui     : SDNode<"X86ISD::MCVTTP2UI", SDTMFloatToInt>;

// Masked versions of CVTP2[SU]IS. Used for v2f64 to v4i32.
// SRC, PASSTHRU, MASK
def X86mcvttp2sis     : SDNode<"X86ISD::MCVTTP2SIS", SDTMFloatToInt>;
def X86mcvttp2uis     : SDNode<"X86ISD::MCVTTP2UIS", SDTMFloatToInt>;

def SDTcvtph2ps : SDTypeProfile<1, 1, [SDTCVecEltisVT<0, f32>,
                                       SDTCVecEltisVT<1, i16>]>;

def SDTcvtps2ph : SDTypeProfile<1, 2, [SDTCVecEltisVT<0, i16>,
                                       SDTCVecEltisVT<1, f32>,
                                       SDTCisVT<2, i32>]>;

// Conversions between float and half-float.
def X86cvtps2ph        : SDNode<"X86ISD::CVTPS2PH", SDTcvtps2ph>;
def X86cvtps2phSAE     : SDNode<"X86ISD::CVTPS2PH_SAE", SDTcvtps2ph>;
def X86cvtph2ps        : SDNode<"X86ISD::CVTPH2PS", SDTcvtph2ps>;
def X86cvtph2psSAE     : SDNode<"X86ISD::CVTPH2PS_SAE", SDTcvtph2ps>;

let IsStrictFP = true in {
  // Conversions between float and half-float.
  def X86strict_cvtps2ph : SDNode<"X86ISD::STRICT_CVTPS2PH", SDTcvtps2ph,
                                  [SDNPHasChain]>;
  def X86strict_cvtph2ps : SDNode<"X86ISD::STRICT_CVTPH2PS", SDTcvtph2ps,
                                  [SDNPHasChain]>;
}

def X86any_cvtph2ps : PatFrags<(ops node:$src),
                               [(X86strict_cvtph2ps node:$src),
                                (X86cvtph2ps node:$src)]>;

def X86any_cvtps2ph : PatFrags<(ops node:$src1, node:$src2),
                               [(X86strict_cvtps2ph node:$src1, node:$src2),
                                (X86cvtps2ph node:$src1, node:$src2)]>;

def SDTmcvtps2ph : SDTypeProfile<1, 4, [SDTCVecEltisVT<0, i16>,
                                        SDTCVecEltisVT<1, f32>,
                                        SDTCisVT<2, i32>,
                                        SDTCisSameAs<0, 3>,
                                        SDTCVecEltisVT<4, i1>,
                                        SDTCisSameNumEltsAs<1, 4>]>;

// Masked version of CVTPS2PH.
// SRC, RND, PASSTHRU, MASK
def X86mcvtps2ph : SDNode<"X86ISD::MCVTPS2PH", SDTmcvtps2ph>;
def X86mcvtps2phSAE : SDNode<"X86ISD::MCVTPS2PH_SAE", SDTmcvtps2ph>;

def X86vfproundRnd: SDNode<"X86ISD::VFPROUND_RND",
                        SDTypeProfile<1, 2, [SDTCisFP<0>, SDTCisVec<0>,
                                             SDTCisFP<1>, SDTCisVec<1>,
                                             SDTCisOpSmallerThanOp<0, 1>,
                                             SDTCisVT<2, i32>]>>;

def X86vminmax : SDNode<"X86ISD::VMINMAX", SDTypeProfile<1, 3, [SDTCisSameAs<0,1>,
                                           SDTCisSameAs<0,2>, SDTCisInt<3>]>>;
def X86vminmaxSae : SDNode<"X86ISD::VMINMAX_SAE", SDTypeProfile<1, 3, [SDTCisSameAs<0,1>,
                                            SDTCisSameAs<0,2>, SDTCisInt<3>]>>;

def X86vminmaxs : SDNode<"X86ISD::VMINMAXS", SDTypeProfile<1, 3, [SDTCisSameAs<0,1>,
                                             SDTCisSameAs<0,2>, SDTCisInt<3>]>>;
def X86vminmaxsSae : SDNode<"X86ISD::VMINMAXS_SAE", SDTypeProfile<1, 3, [SDTCisSameAs<0,1>,
                                                    SDTCisSameAs<0,2>, SDTCisInt<3>]>>;

// Masked version of CVTNEPS2BF16.
// SRC, PASSTHRU, MASK
def X86mcvtneps2bf16 : SDNode<"X86ISD::MCVTNEPS2BF16",
                       SDTypeProfile<1, 3, [SDTCVecEltisVT<0, bf16>,
                                            SDTCVecEltisVT<1, f32>,
                                            SDTCisSameAs<0, 2>,
                                            SDTCVecEltisVT<3, i1>,
                                            SDTCisSameNumEltsAs<1, 3>]>>;

// Vector float to bfloat16.
// Convert packed single data to packed BF16 data
def X86cvtneps2bf16 :  SDNode<"X86ISD::CVTNEPS2BF16",
                       SDTypeProfile<1, 1, [SDTCVecEltisVT<0, bf16>,
                                            SDTCVecEltisVT<1, f32>]>>;

// Dot product of BF16/FP16 pairs to accumulated into
// packed single precision.
def X86dpbf16ps :      SDNode<"X86ISD::DPBF16PS",
                       SDTypeProfile<1, 3, [SDTCVecEltisVT<0, f32>,
                                            SDTCisSameAs<0,1>,
                                            SDTCVecEltisVT<2, bf16>,
                                            SDTCisSameAs<2,3>]>>;
def X86dpfp16ps :      SDNode<"X86ISD::DPFP16PS",
                       SDTypeProfile<1, 3, [SDTCVecEltisVT<0, f32>,
                                            SDTCisSameAs<0,1>,
                                            SDTCVecEltisVT<2, f16>,
                                            SDTCisSameAs<2,3>]>>;

// Galois Field Arithmetic Instructions
def X86GF2P8affineinvqb : SDNode<"X86ISD::GF2P8AFFINEINVQB", SDTBlend>;
def X86GF2P8affineqb    : SDNode<"X86ISD::GF2P8AFFINEQB", SDTBlend>;
def X86GF2P8mulb        : SDNode<"X86ISD::GF2P8MULB", SDTIntBinOp>;

// carry-less multiplication
def X86pclmulqdq : SDNode<"X86ISD::PCLMULQDQ", SDTBlend>;

def SDTX86MaskedStore: SDTypeProfile<0, 3, [       // masked store
  SDTCisVec<0>, SDTCisPtrTy<1>, SDTCisVec<2>, SDTCisSameNumEltsAs<0, 2>
]>;

def X86vpdpbssd  : SDNode<"X86ISD::VPDPBSSD",  SDTVnni>;
def X86vpdpbssds : SDNode<"X86ISD::VPDPBSSDS", SDTVnni>;
def X86vpdpbsud  : SDNode<"X86ISD::VPDPBSUD",  SDTVnni>;
def X86vpdpbsuds : SDNode<"X86ISD::VPDPBSUDS", SDTVnni>;
def X86vpdpbuud  : SDNode<"X86ISD::VPDPBUUD",  SDTVnni>;
def X86vpdpbuuds : SDNode<"X86ISD::VPDPBUUDS", SDTVnni>;

def X86vpdpwsud  : SDNode<"X86ISD::VPDPWSUD",  SDTVnni>;
def X86vpdpwsuds : SDNode<"X86ISD::VPDPWSUDS", SDTVnni>;
def X86vpdpwusd  : SDNode<"X86ISD::VPDPWUSD",  SDTVnni>;
def X86vpdpwusds : SDNode<"X86ISD::VPDPWUSDS", SDTVnni>;
def X86vpdpwuud  : SDNode<"X86ISD::VPDPWUUD",  SDTVnni>;
def X86vpdpwuuds : SDNode<"X86ISD::VPDPWUUDS", SDTVnni>;

def X86Vmpsadbw : SDNode<"X86ISD::MPSADBW", SDTX86PSADBW>;

// in place saturated cvt fp-to-int
def X86vcvtp2ibs : SDNode<"X86ISD::CVTP2IBS",  SDTFloatToInt>;
def X86vcvtp2iubs : SDNode<"X86ISD::CVTP2IUBS",  SDTFloatToInt>;

def X86vcvtp2ibsRnd : SDNode<"X86ISD::CVTP2IBS_RND",  SDTFloatToIntRnd>;
def X86vcvtp2iubsRnd : SDNode<"X86ISD::CVTP2IUBS_RND",  SDTFloatToIntRnd>;

// in place saturated cvtt fp-to-int staff
def X86vcvttp2ibs : SDNode<"X86ISD::CVTTP2IBS",  SDTFloatToInt>;
def X86vcvttp2iubs : SDNode<"X86ISD::CVTTP2IUBS",  SDTFloatToInt>;

def X86vcvttp2ibsSAE : SDNode<"X86ISD::CVTTP2IBS_SAE", SDTFloatToInt>;
def X86vcvttp2iubsSAE : SDNode<"X86ISD::CVTTP2IUBS_SAE", SDTFloatToInt>;

def SDTAVX10CONVERT_I82F16 : SDTypeProfile<1, 2, [
  SDTCVecEltisVT<0, i8>, SDTCVecEltisVT<1, f16>, SDTCisSameAs<1, 2>
]>;

def SDTAVX10CONVERT_F16I8 : SDTypeProfile<1, 1, [
  SDTCVecEltisVT<0, f16>, SDTCVecEltisVT<1, i8>
]>;

def SDTAVX10CONVERT_I8F16 : SDTypeProfile<1, 1, [
  SDTCVecEltisVT<0, i8>, SDTCVecEltisVT<1, f16>
]>;

def SDTAVX10CONVERT_I8F16_MASK : SDTypeProfile<1, 3, [
  SDTCVecEltisVT<0, i8>, SDTCVecEltisVT<1, f16>,
  SDTCisSameAs<0, 2>, SDTCVecEltisVT<3, i1>,
  SDTCisSameNumEltsAs<1, 3>
]>;

def SDTAVX10CONVERT_2I8F16 : SDTypeProfile<1, 2, [
  SDTCVecEltisVT<0, i8>, SDTCVecEltisVT<1, i8>, SDTCVecEltisVT<2, f16>
]>;

def SDTAVX10CONVERT_2I8F16_MASK : SDTypeProfile<1, 4, [
  SDTCVecEltisVT<0, i8>, SDTCisSameAs<0, 1>,
  SDTCVecEltisVT<2, f16>, SDTCisSameAs<0, 3>, SDTCVecEltisVT<4, i1>,
  SDTCisSameNumEltsAs<2, 4>
]>;

def X86vfpround2Rnd : SDNode<"X86ISD::VFPROUND2_RND",
                      SDTypeProfile<1, 3, [SDTCVecEltisVT<0, f16>,
                                           SDTCVecEltisVT<1, f32>,
                                           SDTCisSameAs<1, 2>,
                                           SDTCisVT<3, i32>]>>;
// 3op
def X86vcvt2ph2bf8 : SDNode<"X86ISD::VCVT2PH2BF8",
                              SDTAVX10CONVERT_I82F16>;
def X86vcvt2ph2bf8s : SDNode<"X86ISD::VCVT2PH2BF8S",
                               SDTAVX10CONVERT_I82F16>;
def X86vcvt2ph2hf8 : SDNode<"X86ISD::VCVT2PH2HF8",
                              SDTAVX10CONVERT_I82F16>;
def X86vcvt2ph2hf8s : SDNode<"X86ISD::VCVT2PH2HF8S",
                               SDTAVX10CONVERT_I82F16>;
// 2op no broadcast
def X86vcvthf82ph : SDNode<"X86ISD::VCVTHF82PH",
                           SDTAVX10CONVERT_F16I8>;
// 2op
def X86vcvtbiasph2bf8 : SDNode<"X86ISD::VCVTBIASPH2BF8",
                               SDTAVX10CONVERT_2I8F16>;
def X86vcvtbiasph2bf8s : SDNode<"X86ISD::VCVTBIASPH2BF8S",
                                SDTAVX10CONVERT_2I8F16>;
def X86vcvtbiasph2hf8 : SDNode<"X86ISD::VCVTBIASPH2HF8",
                               SDTAVX10CONVERT_2I8F16>;
def X86vcvtbiasph2hf8s : SDNode<"X86ISD::VCVTBIASPH2HF8S",
                                SDTAVX10CONVERT_2I8F16>;
def X86vcvtph2bf8 : SDNode<"X86ISD::VCVTPH2BF8",
                             SDTAVX10CONVERT_I8F16>;
def X86vcvtph2bf8s : SDNode<"X86ISD::VCVTPH2BF8S",
                              SDTAVX10CONVERT_I8F16>;
def X86vcvtph2hf8 : SDNode<"X86ISD::VCVTPH2HF8",
                             SDTAVX10CONVERT_I8F16>;
def X86vcvtph2hf8s : SDNode<"X86ISD::VCVTPH2HF8S",
                              SDTAVX10CONVERT_I8F16>;

def X86vmcvtbiasph2bf8 : SDNode<"X86ISD::VMCVTBIASPH2BF8",
                         SDTAVX10CONVERT_2I8F16_MASK>;
def X86vmcvtbiasph2bf8s : SDNode<"X86ISD::VMCVTBIASPH2BF8S",
                          SDTAVX10CONVERT_2I8F16_MASK>;
def X86vmcvtbiasph2hf8 : SDNode<"X86ISD::VMCVTBIASPH2HF8",
                         SDTAVX10CONVERT_2I8F16_MASK>;
def X86vmcvtbiasph2hf8s : SDNode<"X86ISD::VMCVTBIASPH2HF8S",
                          SDTAVX10CONVERT_2I8F16_MASK>;
def X86vmcvtph2bf8 : SDNode<"X86ISD::VMCVTPH2BF8",
                       SDTAVX10CONVERT_I8F16_MASK>;
def X86vmcvtph2bf8s : SDNode<"X86ISD::VMCVTPH2BF8S",
                        SDTAVX10CONVERT_I8F16_MASK>;
def X86vmcvtph2hf8 : SDNode<"X86ISD::VMCVTPH2HF8",
                       SDTAVX10CONVERT_I8F16_MASK>;
def X86vmcvtph2hf8s : SDNode<"X86ISD::VMCVTPH2HF8S",
                        SDTAVX10CONVERT_I8F16_MASK>;

//===----------------------------------------------------------------------===//
// SSE pattern fragments
//===----------------------------------------------------------------------===//

// 128-bit load pattern fragments
def loadv8f16    : PatFrag<(ops node:$ptr), (v8f16 (load node:$ptr))>;
def loadv8bf16   : PatFrag<(ops node:$ptr), (v8bf16 (load node:$ptr))>;
def loadv4f32    : PatFrag<(ops node:$ptr), (v4f32 (load node:$ptr))>;
def loadv2f64    : PatFrag<(ops node:$ptr), (v2f64 (load node:$ptr))>;
def loadv2i64    : PatFrag<(ops node:$ptr), (v2i64 (load node:$ptr))>;
def loadv4i32    : PatFrag<(ops node:$ptr), (v4i32 (load node:$ptr))>;
def loadv8i16    : PatFrag<(ops node:$ptr), (v8i16 (load node:$ptr))>;
def loadv16i8    : PatFrag<(ops node:$ptr), (v16i8 (load node:$ptr))>;

// 256-bit load pattern fragments
def loadv16f16   : PatFrag<(ops node:$ptr), (v16f16 (load node:$ptr))>;
def loadv16bf16  : PatFrag<(ops node:$ptr), (v16bf16 (load node:$ptr))>;
def loadv8f32    : PatFrag<(ops node:$ptr), (v8f32  (load node:$ptr))>;
def loadv4f64    : PatFrag<(ops node:$ptr), (v4f64  (load node:$ptr))>;
def loadv4i64    : PatFrag<(ops node:$ptr), (v4i64  (load node:$ptr))>;
def loadv8i32    : PatFrag<(ops node:$ptr), (v8i32  (load node:$ptr))>;
def loadv16i16   : PatFrag<(ops node:$ptr), (v16i16 (load node:$ptr))>;
def loadv32i8    : PatFrag<(ops node:$ptr), (v32i8  (load node:$ptr))>;

// 512-bit load pattern fragments
def loadv32f16   : PatFrag<(ops node:$ptr), (v32f16 (load node:$ptr))>;
def loadv32bf16  : PatFrag<(ops node:$ptr), (v32bf16 (load node:$ptr))>;
def loadv16f32   : PatFrag<(ops node:$ptr), (v16f32 (load node:$ptr))>;
def loadv8f64    : PatFrag<(ops node:$ptr), (v8f64  (load node:$ptr))>;
def loadv8i64    : PatFrag<(ops node:$ptr), (v8i64  (load node:$ptr))>;
def loadv16i32   : PatFrag<(ops node:$ptr), (v16i32 (load node:$ptr))>;
def loadv32i16   : PatFrag<(ops node:$ptr), (v32i16 (load node:$ptr))>;
def loadv64i8    : PatFrag<(ops node:$ptr), (v64i8  (load node:$ptr))>;

// 128-/256-/512-bit extload pattern fragments
def extloadv2f32 : PatFrag<(ops node:$ptr), (extloadvf32 node:$ptr)>;
def extloadv4f32 : PatFrag<(ops node:$ptr), (extloadvf32 node:$ptr)>;
def extloadv8f32 : PatFrag<(ops node:$ptr), (extloadvf32 node:$ptr)>;
def extloadv2f16 : PatFrag<(ops node:$ptr), (extloadvf16 node:$ptr)>;
def extloadv4f16 : PatFrag<(ops node:$ptr), (extloadvf16 node:$ptr)>;
def extloadv8f16 : PatFrag<(ops node:$ptr), (extloadvf16 node:$ptr)>;
def extloadv16f16 : PatFrag<(ops node:$ptr), (extloadvf16 node:$ptr)>;

// Like 'store', but always requires vector size alignment.
def alignedstore : PatFrag<(ops node:$val, node:$ptr),
                           (store node:$val, node:$ptr), [{
  auto *St = cast<StoreSDNode>(N);
  return St->getAlign() >= St->getMemoryVT().getStoreSize();
}]> {
  let GISelPredicateCode = [{
    auto &LdSt = cast<GLoadStore>(MI);
    return LdSt.getAlign() >= LdSt.getMemSize().getValue();
  }];
}

// Like 'load', but always requires vector size alignment.
def alignedload : PatFrag<(ops node:$ptr), (load node:$ptr), [{
  auto *Ld = cast<LoadSDNode>(N);
  return Ld->getAlign() >= Ld->getMemoryVT().getStoreSize();
}]> {
  let GISelPredicateCode = [{
    auto &LdSt = cast<GLoadStore>(MI);
    return LdSt.getAlign() >= LdSt.getMemSize().getValue();
  }];
}

// 128-bit aligned load pattern fragments
// NOTE: all 128-bit integer vector loads are promoted to v2i64
def alignedloadv8f16 : PatFrag<(ops node:$ptr),
                               (v8f16 (alignedload node:$ptr))>;
def alignedloadv8bf16 : PatFrag<(ops node:$ptr),
                                (v8bf16 (alignedload node:$ptr))>;
def alignedloadv4f32 : PatFrag<(ops node:$ptr),
                               (v4f32 (alignedload node:$ptr))>;
def alignedloadv2f64 : PatFrag<(ops node:$ptr),
                               (v2f64 (alignedload node:$ptr))>;
def alignedloadv2i64 : PatFrag<(ops node:$ptr),
                               (v2i64 (alignedload node:$ptr))>;
def alignedloadv4i32 : PatFrag<(ops node:$ptr),
                               (v4i32 (alignedload node:$ptr))>;
def alignedloadv8i16 : PatFrag<(ops node:$ptr),
                               (v8i16 (alignedload node:$ptr))>;
def alignedloadv16i8 : PatFrag<(ops node:$ptr),
                               (v16i8 (alignedload node:$ptr))>;

// 256-bit aligned load pattern fragments
// NOTE: all 256-bit integer vector loads are promoted to v4i64
def alignedloadv16f16 : PatFrag<(ops node:$ptr),
                                (v16f16 (alignedload node:$ptr))>;
def alignedloadv16bf16 : PatFrag<(ops node:$ptr),
                                 (v16bf16 (alignedload node:$ptr))>;
def alignedloadv8f32  : PatFrag<(ops node:$ptr),
                                (v8f32  (alignedload node:$ptr))>;
def alignedloadv4f64  : PatFrag<(ops node:$ptr),
                                (v4f64  (alignedload node:$ptr))>;
def alignedloadv4i64  : PatFrag<(ops node:$ptr),
                                (v4i64  (alignedload node:$ptr))>;
def alignedloadv8i32  : PatFrag<(ops node:$ptr),
                                (v8i32  (alignedload node:$ptr))>;
def alignedloadv16i16 : PatFrag<(ops node:$ptr),
                                (v16i16 (alignedload node:$ptr))>;
def alignedloadv32i8  : PatFrag<(ops node:$ptr),
                                (v32i8  (alignedload node:$ptr))>;

// 512-bit aligned load pattern fragments
def alignedloadv32f16 : PatFrag<(ops node:$ptr),
                                (v32f16 (alignedload node:$ptr))>;
def alignedloadv32bf16 : PatFrag<(ops node:$ptr),
                                 (v32bf16 (alignedload node:$ptr))>;
def alignedloadv16f32 : PatFrag<(ops node:$ptr),
                                (v16f32 (alignedload node:$ptr))>;
def alignedloadv8f64  : PatFrag<(ops node:$ptr),
                                (v8f64  (alignedload node:$ptr))>;
def alignedloadv8i64  : PatFrag<(ops node:$ptr),
                                (v8i64  (alignedload node:$ptr))>;
def alignedloadv16i32 : PatFrag<(ops node:$ptr),
                                (v16i32 (alignedload node:$ptr))>;
def alignedloadv32i16 : PatFrag<(ops node:$ptr),
                                (v32i16 (alignedload node:$ptr))>;
def alignedloadv64i8  : PatFrag<(ops node:$ptr),
                                (v64i8  (alignedload node:$ptr))>;

// Like 'load', but uses special alignment checks suitable for use in
// memory operands in most SSE instructions, which are required to
// be naturally aligned on some targets but not on others.  If the subtarget
// allows unaligned accesses, match any load, though this may require
// setting a feature bit in the processor (on startup, for example).
// Opteron 10h and later implement such a feature.
def memop : PatFrag<(ops node:$ptr), (load node:$ptr), [{
  auto *Ld = cast<LoadSDNode>(N);
  return Subtarget->hasSSEUnalignedMem() ||
         Ld->getAlign() >= Ld->getMemoryVT().getStoreSize();
}]>;

// 128-bit memop pattern fragments
// NOTE: all 128-bit integer vector loads are promoted to v2i64
def memopv4f32 : PatFrag<(ops node:$ptr), (v4f32 (memop node:$ptr))>;
def memopv2f64 : PatFrag<(ops node:$ptr), (v2f64 (memop node:$ptr))>;
def memopv2i64 : PatFrag<(ops node:$ptr), (v2i64 (memop node:$ptr))>;
def memopv4i32 : PatFrag<(ops node:$ptr), (v4i32 (memop node:$ptr))>;
def memopv8i16 : PatFrag<(ops node:$ptr), (v8i16 (memop node:$ptr))>;
def memopv16i8 : PatFrag<(ops node:$ptr), (v16i8 (memop node:$ptr))>;

// 128-bit bitconvert pattern fragments
def bc_v4f32 : PatFrag<(ops node:$in), (v4f32 (bitconvert node:$in))>;
def bc_v2f64 : PatFrag<(ops node:$in), (v2f64 (bitconvert node:$in))>;
def bc_v16i8 : PatFrag<(ops node:$in), (v16i8 (bitconvert node:$in))>;
def bc_v8i16 : PatFrag<(ops node:$in), (v8i16 (bitconvert node:$in))>;
def bc_v4i32 : PatFrag<(ops node:$in), (v4i32 (bitconvert node:$in))>;
def bc_v2i64 : PatFrag<(ops node:$in), (v2i64 (bitconvert node:$in))>;

// 256-bit bitconvert pattern fragments
def bc_v32i8 : PatFrag<(ops node:$in), (v32i8 (bitconvert node:$in))>;
def bc_v16i16 : PatFrag<(ops node:$in), (v16i16 (bitconvert node:$in))>;
def bc_v8i32 : PatFrag<(ops node:$in), (v8i32 (bitconvert node:$in))>;
def bc_v4i64 : PatFrag<(ops node:$in), (v4i64 (bitconvert node:$in))>;
def bc_v8f32 : PatFrag<(ops node:$in), (v8f32 (bitconvert node:$in))>;
def bc_v4f64 : PatFrag<(ops node:$in), (v4f64 (bitconvert node:$in))>;

// 512-bit bitconvert pattern fragments
def bc_v64i8 : PatFrag<(ops node:$in), (v64i8 (bitconvert node:$in))>;
def bc_v32i16 : PatFrag<(ops node:$in), (v32i16 (bitconvert node:$in))>;
def bc_v16i32 : PatFrag<(ops node:$in), (v16i32 (bitconvert node:$in))>;
def bc_v8i64 : PatFrag<(ops node:$in), (v8i64 (bitconvert node:$in))>;
def bc_v8f64 : PatFrag<(ops node:$in), (v8f64 (bitconvert node:$in))>;
def bc_v16f32 : PatFrag<(ops node:$in), (v16f32 (bitconvert node:$in))>;

def X86vzload16 : PatFrag<(ops node:$src),
                          (X86vzld node:$src), [{
  return cast<MemIntrinsicSDNode>(N)->getMemoryVT().getStoreSize() == 2;
}]>;

def X86vzload32 : PatFrag<(ops node:$src),
                          (X86vzld node:$src), [{
  return cast<MemIntrinsicSDNode>(N)->getMemoryVT().getStoreSize() == 4;
}]>;

def X86vzload64 : PatFrag<(ops node:$src),
                          (X86vzld node:$src), [{
  return cast<MemIntrinsicSDNode>(N)->getMemoryVT().getStoreSize() == 8;
}]>;

def X86vextractstore64 : PatFrag<(ops node:$val, node:$ptr),
                                 (X86vextractst node:$val, node:$ptr), [{
  return cast<MemIntrinsicSDNode>(N)->getMemoryVT().getStoreSize() == 8;
}]>;

def X86VBroadcastld8 : PatFrag<(ops node:$src),
                               (X86VBroadcastld node:$src), [{
  return cast<MemIntrinsicSDNode>(N)->getMemoryVT().getStoreSize() == 1;
}]>;

def X86VBroadcastld16 : PatFrag<(ops node:$src),
                                (X86VBroadcastld node:$src), [{
  return cast<MemIntrinsicSDNode>(N)->getMemoryVT().getStoreSize() == 2;
}]>;

def X86VBroadcastld32 : PatFrag<(ops node:$src),
                                (X86VBroadcastld node:$src), [{
  return cast<MemIntrinsicSDNode>(N)->getMemoryVT().getStoreSize() == 4;
}]>;

def X86VBroadcastld64 : PatFrag<(ops node:$src),
                                (X86VBroadcastld node:$src), [{
  return cast<MemIntrinsicSDNode>(N)->getMemoryVT().getStoreSize() == 8;
}]>;

def X86SubVBroadcastld128 : PatFrag<(ops node:$src),
                                    (X86SubVBroadcastld node:$src), [{
  return cast<MemIntrinsicSDNode>(N)->getMemoryVT().getStoreSize() == 16;
}]>;

def X86SubVBroadcastld256 : PatFrag<(ops node:$src),
                                    (X86SubVBroadcastld node:$src), [{
  return cast<MemIntrinsicSDNode>(N)->getMemoryVT().getStoreSize() == 32;
}]>;

// Scalar SSE intrinsic fragments to match several different types of loads.
// Used by scalar SSE intrinsic instructions which have 128 bit types, but
// only load a single element.
// FIXME: We should add more canolicalizing in DAGCombine. Particulary removing
// the simple_load case.
def sse_load_bf16 : PatFrags<(ops node:$ptr),
                            [(v8bf16 (simple_load node:$ptr)),
                             (v8bf16 (X86vzload16 node:$ptr)),
                             (v8bf16 (scalar_to_vector (loadf16 node:$ptr)))]>;
def sse_load_f16 : PatFrags<(ops node:$ptr),
                            [(v8f16 (simple_load node:$ptr)),
                             (v8f16 (X86vzload16 node:$ptr)),
                             (v8f16 (scalar_to_vector (loadf16 node:$ptr)))]>;
def sse_load_f32 : PatFrags<(ops node:$ptr),
                            [(v4f32 (simple_load node:$ptr)),
                             (v4f32 (X86vzload32 node:$ptr)),
                             (v4f32 (scalar_to_vector (loadf32 node:$ptr)))]>;
def sse_load_f64 : PatFrags<(ops node:$ptr),
                            [(v2f64 (simple_load node:$ptr)),
                             (v2f64 (X86vzload64 node:$ptr)),
                             (v2f64 (scalar_to_vector (loadf64 node:$ptr)))]>;

def fp16imm0 : PatLeaf<(f16 fpimm), [{
  return N->isExactlyValue(+0.0);
}]>;

def fp32imm0 : PatLeaf<(f32 fpimm), [{
  return N->isExactlyValue(+0.0);
}]>;

def fp64imm0 : PatLeaf<(f64 fpimm), [{
  return N->isExactlyValue(+0.0);
}]>;

def fp128imm0 : PatLeaf<(f128 fpimm), [{
  return N->isExactlyValue(+0.0);
}]>;

// EXTRACT_get_vextract128_imm xform function: convert extract_subvector index
// to VEXTRACTF128/VEXTRACTI128 imm.
def EXTRACT_get_vextract128_imm : SDNodeXForm<extract_subvector, [{
  return getExtractVEXTRACTImmediate(N, 128, SDLoc(N));
}]>;

// INSERT_get_vinsert128_imm xform function: convert insert_subvector index to
// VINSERTF128/VINSERTI128 imm.
def INSERT_get_vinsert128_imm : SDNodeXForm<insert_subvector, [{
  return getInsertVINSERTImmediate(N, 128, SDLoc(N));
}]>;

// INSERT_get_vperm2x128_imm xform function: convert insert_subvector index to
// commuted VPERM2F128/VPERM2I128 imm.
def INSERT_get_vperm2x128_commutedimm : SDNodeXForm<insert_subvector, [{
  return getPermuteVINSERTCommutedImmediate(N, 128, SDLoc(N));
}]>;

// EXTRACT_get_vextract256_imm xform function: convert extract_subvector index
// to VEXTRACTF64x4 imm.
def EXTRACT_get_vextract256_imm : SDNodeXForm<extract_subvector, [{
  return getExtractVEXTRACTImmediate(N, 256, SDLoc(N));
}]>;

// INSERT_get_vinsert256_imm xform function: convert insert_subvector index to
// VINSERTF64x4 imm.
def INSERT_get_vinsert256_imm : SDNodeXForm<insert_subvector, [{
  return getInsertVINSERTImmediate(N, 256, SDLoc(N));
}]>;

def vextract128_extract : PatFrag<(ops node:$bigvec, node:$index),
                                   (extract_subvector node:$bigvec,
                                                      node:$index), [{
  // Index 0 can be handled via extract_subreg.
  return !isNullConstant(N->getOperand(1));
}], EXTRACT_get_vextract128_imm>;

def vinsert128_insert : PatFrag<(ops node:$bigvec, node:$smallvec,
                                      node:$index),
                                 (insert_subvector node:$bigvec, node:$smallvec,
                                                   node:$index), [{}],
                                INSERT_get_vinsert128_imm>;

def vextract256_extract : PatFrag<(ops node:$bigvec, node:$index),
                                   (extract_subvector node:$bigvec,
                                                      node:$index), [{
  // Index 0 can be handled via extract_subreg.
  return !isNullConstant(N->getOperand(1));
}], EXTRACT_get_vextract256_imm>;

def vinsert256_insert : PatFrag<(ops node:$bigvec, node:$smallvec,
                                      node:$index),
                                 (insert_subvector node:$bigvec, node:$smallvec,
                                                   node:$index), [{}],
                                INSERT_get_vinsert256_imm>;

def masked_load : PatFrag<(ops node:$src1, node:$src2, node:$src3),
                          (masked_ld node:$src1, undef, node:$src2, node:$src3), [{
  return !cast<MaskedLoadSDNode>(N)->isExpandingLoad() &&
    cast<MaskedLoadSDNode>(N)->getExtensionType() == ISD::NON_EXTLOAD &&
    cast<MaskedLoadSDNode>(N)->isUnindexed();
}]>;

def masked_load_aligned : PatFrag<(ops node:$src1, node:$src2, node:$src3),
                         (masked_load node:$src1, node:$src2, node:$src3), [{
  // Use the node type to determine the size the alignment needs to match.
  // We can't use memory VT because type widening changes the node VT, but
  // not the memory VT.
  auto *Ld = cast<MaskedLoadSDNode>(N);
  return Ld->getAlign() >= Ld->getValueType(0).getStoreSize();
}]>;

def X86mExpandingLoad : PatFrag<(ops node:$src1, node:$src2, node:$src3),
                         (masked_ld node:$src1, undef, node:$src2, node:$src3), [{
  return cast<MaskedLoadSDNode>(N)->isExpandingLoad() &&
         cast<MaskedLoadSDNode>(N)->isUnindexed();
}]>;

// Masked store fragments.
// X86mstore can't be implemented in core DAG files because some targets
// do not support vector types (llvm-tblgen will fail).
def masked_store : PatFrag<(ops node:$src1, node:$src2, node:$src3),
                        (masked_st node:$src1, node:$src2, undef, node:$src3), [{
  return !cast<MaskedStoreSDNode>(N)->isTruncatingStore() &&
         !cast<MaskedStoreSDNode>(N)->isCompressingStore() &&
         cast<MaskedStoreSDNode>(N)->isUnindexed();
}]>;

def masked_store_aligned : PatFrag<(ops node:$src1, node:$src2, node:$src3),
                         (masked_store node:$src1, node:$src2, node:$src3), [{
  // Use the node type to determine the size the alignment needs to match.
  // We can't use memory VT because type widening changes the node VT, but
  // not the memory VT.
  auto *St = cast<MaskedStoreSDNode>(N);
  return St->getAlign() >= St->getOperand(1).getValueType().getStoreSize();
}]>;

def X86mCompressingStore : PatFrag<(ops node:$src1, node:$src2, node:$src3),
                             (masked_st node:$src1, node:$src2, undef, node:$src3), [{
    return cast<MaskedStoreSDNode>(N)->isCompressingStore() &&
           cast<MaskedStoreSDNode>(N)->isUnindexed();
}]>;

// masked truncstore fragments
// X86mtruncstore can't be implemented in core DAG files because some targets
// doesn't support vector type ( llvm-tblgen will fail)
def X86mtruncstore : PatFrag<(ops node:$src1, node:$src2, node:$src3),
                             (masked_st node:$src1, node:$src2, undef, node:$src3), [{
    return cast<MaskedStoreSDNode>(N)->isTruncatingStore() &&
           cast<MaskedStoreSDNode>(N)->isUnindexed();
}]>;
def masked_truncstorevi8 :
  PatFrag<(ops node:$src1, node:$src2, node:$src3),
          (X86mtruncstore node:$src1, node:$src2, node:$src3), [{
  return cast<MaskedStoreSDNode>(N)->getMemoryVT().getScalarType() == MVT::i8;
}]>;
def masked_truncstorevi16 :
  PatFrag<(ops node:$src1, node:$src2, node:$src3),
          (X86mtruncstore node:$src1, node:$src2, node:$src3), [{
  return cast<MaskedStoreSDNode>(N)->getMemoryVT().getScalarType() == MVT::i16;
}]>;
def masked_truncstorevi32 :
  PatFrag<(ops node:$src1, node:$src2, node:$src3),
          (X86mtruncstore node:$src1, node:$src2, node:$src3), [{
  return cast<MaskedStoreSDNode>(N)->getMemoryVT().getScalarType() == MVT::i32;
}]>;

// Vector truncating store with unsigned/signed saturation
def X86TruncSStore : SDNode<"X86ISD::VTRUNCSTORES",  SDTStore,
                       [SDNPHasChain, SDNPMayStore, SDNPMemOperand]>;
def X86TruncUSStore : SDNode<"X86ISD::VTRUNCSTOREUS",  SDTStore,
                       [SDNPHasChain, SDNPMayStore, SDNPMemOperand]>;

// Vector truncating masked store with unsigned/signed saturation
def X86MTruncSStore : SDNode<"X86ISD::VMTRUNCSTORES",  SDTX86MaskedStore,
                       [SDNPHasChain, SDNPMayStore, SDNPMemOperand]>;
def X86MTruncUSStore : SDNode<"X86ISD::VMTRUNCSTOREUS",  SDTX86MaskedStore,
                       [SDNPHasChain, SDNPMayStore, SDNPMemOperand]>;

def truncstore_s_vi8 : PatFrag<(ops node:$val, node:$ptr),
                               (X86TruncSStore node:$val, node:$ptr), [{
  return cast<MemIntrinsicSDNode>(N)->getMemoryVT().getScalarType() == MVT::i8;
}]>;

def truncstore_us_vi8 : PatFrag<(ops node:$val, node:$ptr),
                               (X86TruncUSStore node:$val, node:$ptr), [{
  return cast<MemIntrinsicSDNode>(N)->getMemoryVT().getScalarType() == MVT::i8;
}]>;

def truncstore_s_vi16 : PatFrag<(ops node:$val, node:$ptr),
                               (X86TruncSStore node:$val, node:$ptr), [{
  return cast<MemIntrinsicSDNode>(N)->getMemoryVT().getScalarType() == MVT::i16;
}]>;

def truncstore_us_vi16 : PatFrag<(ops node:$val, node:$ptr),
                               (X86TruncUSStore node:$val, node:$ptr), [{
  return cast<MemIntrinsicSDNode>(N)->getMemoryVT().getScalarType() == MVT::i16;
}]>;

def truncstore_s_vi32 : PatFrag<(ops node:$val, node:$ptr),
                               (X86TruncSStore node:$val, node:$ptr), [{
  return cast<MemIntrinsicSDNode>(N)->getMemoryVT().getScalarType() == MVT::i32;
}]>;

def truncstore_us_vi32 : PatFrag<(ops node:$val, node:$ptr),
                               (X86TruncUSStore node:$val, node:$ptr), [{
  return cast<MemIntrinsicSDNode>(N)->getMemoryVT().getScalarType() == MVT::i32;
}]>;

def masked_truncstore_s_vi8 : PatFrag<(ops node:$src1, node:$src2, node:$src3),
                     (X86MTruncSStore node:$src1, node:$src2, node:$src3), [{
  return cast<MemIntrinsicSDNode>(N)->getMemoryVT().getScalarType() == MVT::i8;
}]>;

def masked_truncstore_us_vi8 : PatFrag<(ops node:$src1, node:$src2, node:$src3),
                               (X86MTruncUSStore node:$src1, node:$src2, node:$src3), [{
  return cast<MemIntrinsicSDNode>(N)->getMemoryVT().getScalarType() == MVT::i8;
}]>;

def masked_truncstore_s_vi16 : PatFrag<(ops node:$src1, node:$src2, node:$src3),
                               (X86MTruncSStore node:$src1, node:$src2, node:$src3), [{
  return cast<MemIntrinsicSDNode>(N)->getMemoryVT().getScalarType() == MVT::i16;
}]>;

def masked_truncstore_us_vi16 : PatFrag<(ops node:$src1, node:$src2, node:$src3),
                               (X86MTruncUSStore node:$src1, node:$src2, node:$src3), [{
  return cast<MemIntrinsicSDNode>(N)->getMemoryVT().getScalarType() == MVT::i16;
}]>;

def masked_truncstore_s_vi32 : PatFrag<(ops node:$src1, node:$src2, node:$src3),
                               (X86MTruncSStore node:$src1, node:$src2, node:$src3), [{
  return cast<MemIntrinsicSDNode>(N)->getMemoryVT().getScalarType() == MVT::i32;
}]>;

def masked_truncstore_us_vi32 : PatFrag<(ops node:$src1, node:$src2, node:$src3),
                               (X86MTruncUSStore node:$src1, node:$src2, node:$src3), [{
  return cast<MemIntrinsicSDNode>(N)->getMemoryVT().getScalarType() == MVT::i32;
}]>;

def X86Vfpclasss_su : PatFrag<(ops node:$src1, node:$src2),
                              (X86Vfpclasss node:$src1, node:$src2), [{
  return N->hasOneUse();
}]>;

def X86Vfpclass_su : PatFrag<(ops node:$src1, node:$src2),
                             (X86Vfpclass node:$src1, node:$src2), [{
  return N->hasOneUse();
}]>;

// These nodes use 'vnot' instead of 'not' to support vectors.
def vandn : PatFrag<(ops node:$i0, node:$i1), (and (vnot node:$i0), node:$i1)>;
def vxnor : PatFrag<(ops node:$i0, node:$i1), (vnot (xor node:$i0, node:$i1))>;

// Used for matching masked operations. Ensures the operation part only has a
// single use.
def vselect_mask : PatFrag<(ops node:$mask, node:$src1, node:$src2),
                           (vselect node:$mask, node:$src1, node:$src2), [{
  return isProfitableToFormMaskedOp(N);
}]>;

def X86selects_mask : PatFrag<(ops node:$mask, node:$src1, node:$src2),
                              (X86selects node:$mask, node:$src1, node:$src2), [{
  return isProfitableToFormMaskedOp(N);
}]>;

def X86cmpms_su : PatFrag<(ops node:$src1, node:$src2, node:$cc),
                          (X86cmpms node:$src1, node:$src2, node:$cc), [{
  return N->hasOneUse();
}]>;
def X86cmpmsSAE_su : PatFrag<(ops node:$src1, node:$src2, node:$cc),
                          (X86cmpmsSAE node:$src1, node:$src2, node:$cc), [{
  return N->hasOneUse();
}]>;

// PatFrags that contain a select and a truncate op. The take operands in the
// same order as X86vmtrunc, X86vmtruncs, X86vmtruncus. This allows us to pass
// either to the multiclasses.
def select_trunc : PatFrag<(ops node:$src, node:$src0, node:$mask),
                           (vselect_mask node:$mask,
                                         (trunc node:$src), node:$src0)>;
def select_truncs : PatFrag<(ops node:$src, node:$src0, node:$mask),
                            (vselect_mask node:$mask,
                                          (X86vtruncs node:$src), node:$src0)>;
def select_truncus : PatFrag<(ops node:$src, node:$src0, node:$mask),
                             (vselect_mask node:$mask,
                                           (X86vtruncus node:$src), node:$src0)>;

def X86Vpshufbitqmb_su : PatFrag<(ops node:$src1, node:$src2),
                                 (X86Vpshufbitqmb node:$src1, node:$src2), [{
  return N->hasOneUse();
}]>;

def X86pcmpgtm : PatFrag<(ops node:$src1, node:$src2),
                         (setcc node:$src1, node:$src2, SETGT)>;

def X86pcmpm_imm : SDNodeXForm<setcc, [{
  ISD::CondCode CC = cast<CondCodeSDNode>(N->getOperand(2))->get();
  uint8_t SSECC = X86::getVPCMPImmForCond(CC);
  return getI8Imm(SSECC, SDLoc(N));
}]>;

// Swapped operand version of the above.
def X86pcmpm_imm_commute : SDNodeXForm<setcc, [{
  ISD::CondCode CC = cast<CondCodeSDNode>(N->getOperand(2))->get();
  uint8_t SSECC = X86::getVPCMPImmForCond(CC);
  SSECC = X86::getSwappedVPCMPImm(SSECC);
  return getI8Imm(SSECC, SDLoc(N));
}]>;

def X86pcmpm : PatFrag<(ops node:$src1, node:$src2, node:$cc),
                       (setcc node:$src1, node:$src2, node:$cc), [{
  ISD::CondCode CC = cast<CondCodeSDNode>(N->getOperand(2))->get();
  return !ISD::isUnsignedIntSetCC(CC);
}], X86pcmpm_imm>;

def X86pcmpm_su : PatFrag<(ops node:$src1, node:$src2, node:$cc),
                          (setcc node:$src1, node:$src2, node:$cc), [{
  ISD::CondCode CC = cast<CondCodeSDNode>(N->getOperand(2))->get();
  return N->hasOneUse() && !ISD::isUnsignedIntSetCC(CC);
}], X86pcmpm_imm>;

def X86pcmpum : PatFrag<(ops node:$src1, node:$src2, node:$cc),
                        (setcc node:$src1, node:$src2, node:$cc), [{
  ISD::CondCode CC = cast<CondCodeSDNode>(N->getOperand(2))->get();
  return ISD::isUnsignedIntSetCC(CC);
}], X86pcmpm_imm>;

def X86pcmpum_su : PatFrag<(ops node:$src1, node:$src2, node:$cc),
                           (setcc node:$src1, node:$src2, node:$cc), [{
  ISD::CondCode CC = cast<CondCodeSDNode>(N->getOperand(2))->get();
  return N->hasOneUse() && ISD::isUnsignedIntSetCC(CC);
}], X86pcmpm_imm>;

def X86cmpm_su : PatFrag<(ops node:$src1, node:$src2, node:$cc),
                         (X86cmpm node:$src1, node:$src2, node:$cc), [{
  return N->hasOneUse();
}]>;

def X86cmpm_imm_commute : SDNodeXForm<timm, [{
  uint8_t Imm = X86::getSwappedVCMPImm(N->getZExtValue() & 0x1f);
  return getI8Imm(Imm, SDLoc(N));
}]>;

def X86vpmaddwd_su : PatFrag<(ops node:$lhs, node:$rhs),
                             (X86vpmaddwd node:$lhs, node:$rhs), [{
  return N->hasOneUse();
}]>;

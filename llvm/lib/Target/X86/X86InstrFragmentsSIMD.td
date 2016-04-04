//===-- X86InstrFragmentsSIMD.td - x86 SIMD ISA ------------*- tablegen -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file provides pattern fragments useful for SIMD instructions.
//
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// MMX specific DAG Nodes.
//===----------------------------------------------------------------------===//

// Low word of MMX to GPR.
def MMX_X86movd2w : SDNode<"X86ISD::MMX_MOVD2W", SDTypeProfile<1, 1,
                            [SDTCisVT<0, i32>, SDTCisVT<1, x86mmx>]>>;
// GPR to low word of MMX.
def MMX_X86movw2d : SDNode<"X86ISD::MMX_MOVW2D", SDTypeProfile<1, 1,
                            [SDTCisVT<0, x86mmx>, SDTCisVT<1, i32>]>>;

//===----------------------------------------------------------------------===//
// MMX Pattern Fragments
//===----------------------------------------------------------------------===//

def load_mmx : PatFrag<(ops node:$ptr), (x86mmx (load node:$ptr))>;
def load_mvmmx : PatFrag<(ops node:$ptr),
                         (x86mmx (MMX_X86movw2d (load node:$ptr)))>;
def bc_mmx  : PatFrag<(ops node:$in), (x86mmx  (bitconvert node:$in))>;

//===----------------------------------------------------------------------===//
// SSE specific DAG Nodes.
//===----------------------------------------------------------------------===//

def SDTX86VFCMP : SDTypeProfile<1, 3, [SDTCisInt<0>, SDTCisSameAs<1, 2>,
                                       SDTCisFP<1>, SDTCisVT<3, i8>,
                                       SDTCisVec<1>]>;
def SDTX86CmpTestSae : SDTypeProfile<1, 3, [SDTCisVT<0, i32>, 
                                     SDTCisSameAs<1, 2>, SDTCisInt<3>]>;

def X86fmin    : SDNode<"X86ISD::FMIN",      SDTFPBinOp>;
def X86fmax    : SDNode<"X86ISD::FMAX",      SDTFPBinOp>;

// Commutative and Associative FMIN and FMAX.
def X86fminc    : SDNode<"X86ISD::FMINC", SDTFPBinOp,
    [SDNPCommutative, SDNPAssociative]>;
def X86fmaxc    : SDNode<"X86ISD::FMAXC", SDTFPBinOp,
    [SDNPCommutative, SDNPAssociative]>;

def X86fand    : SDNode<"X86ISD::FAND",      SDTFPBinOp,
                        [SDNPCommutative, SDNPAssociative]>;
def X86for     : SDNode<"X86ISD::FOR",       SDTFPBinOp,
                        [SDNPCommutative, SDNPAssociative]>;
def X86fxor    : SDNode<"X86ISD::FXOR",      SDTFPBinOp,
                        [SDNPCommutative, SDNPAssociative]>;
def X86fandn   : SDNode<"X86ISD::FANDN",     SDTFPBinOp,
                        [SDNPCommutative, SDNPAssociative]>;
def X86frsqrt  : SDNode<"X86ISD::FRSQRT",    SDTFPUnaryOp>;
def X86frcp    : SDNode<"X86ISD::FRCP",      SDTFPUnaryOp>;
def X86frsqrt14s: SDNode<"X86ISD::FRSQRT",  SDTFPBinOp>;
def X86frcp14s : SDNode<"X86ISD::FRCP",    SDTFPBinOp>;
def X86fgetsign: SDNode<"X86ISD::FGETSIGNx86",SDTFPToIntOp>;
def X86fhadd   : SDNode<"X86ISD::FHADD",     SDTFPBinOp>;
def X86fhsub   : SDNode<"X86ISD::FHSUB",     SDTFPBinOp>;
def X86hadd    : SDNode<"X86ISD::HADD",      SDTIntBinOp>;
def X86hsub    : SDNode<"X86ISD::HSUB",      SDTIntBinOp>;
def X86comi    : SDNode<"X86ISD::COMI",      SDTX86CmpTest>;
def X86comiSae : SDNode<"X86ISD::COMI",      SDTX86CmpTestSae>;
def X86ucomi   : SDNode<"X86ISD::UCOMI",     SDTX86CmpTest>;
def X86ucomiSae: SDNode<"X86ISD::UCOMI",     SDTX86CmpTestSae>;
def X86cmps    : SDNode<"X86ISD::FSETCC",     SDTX86Cmps>;
//def X86cmpsd   : SDNode<"X86ISD::FSETCCsd",    SDTX86Cmpsd>;
def X86cvtdq2pd: SDNode<"X86ISD::CVTDQ2PD",
                 SDTypeProfile<1, 1, [SDTCisVT<0, v2f64>,
                                      SDTCisVT<1, v4i32>]>>;
def X86cvtudq2pd: SDNode<"X86ISD::CVTUDQ2PD",
                 SDTypeProfile<1, 1, [SDTCisVT<0, v2f64>,
                                      SDTCisVT<1, v4i32>]>>;
def X86pshufb  : SDNode<"X86ISD::PSHUFB",
                 SDTypeProfile<1, 2, [SDTCVecEltisVT<0, i8>, SDTCisSameAs<0,1>,
                                      SDTCisSameAs<0,2>]>>;
def X86psadbw  : SDNode<"X86ISD::PSADBW",
                 SDTypeProfile<1, 2, [SDTCVecEltisVT<0, i64>,
                                      SDTCVecEltisVT<1, i8>,
                                      SDTCisSameSizeAs<0,1>,
                                      SDTCisSameAs<1,2>]>>;
def X86dbpsadbw : SDNode<"X86ISD::DBPSADBW",
                  SDTypeProfile<1, 3, [SDTCVecEltisVT<0, i16>,
                                       SDTCVecEltisVT<1, i8>,
                                       SDTCisSameSizeAs<0,1>,
                                       SDTCisSameAs<1,2>, SDTCisInt<3>]>>;
def X86andnp   : SDNode<"X86ISD::ANDNP",
                 SDTypeProfile<1, 2, [SDTCisVec<0>, SDTCisSameAs<0,1>,
                                      SDTCisSameAs<0,2>]>>;
def X86multishift   : SDNode<"X86ISD::MULTISHIFT",
                 SDTypeProfile<1, 2, [SDTCisVec<0>, SDTCisVec<1>,
                                      SDTCisSameAs<1,2>]>>;
def X86pextrb  : SDNode<"X86ISD::PEXTRB",
                 SDTypeProfile<1, 2, [SDTCisVT<0, i32>, SDTCisVT<1, v16i8>,
                                      SDTCisPtrTy<2>]>>;
def X86pextrw  : SDNode<"X86ISD::PEXTRW",
                 SDTypeProfile<1, 2, [SDTCisVT<0, i32>, SDTCisVT<1, v8i16>,
                                      SDTCisPtrTy<2>]>>;
def X86pinsrb  : SDNode<"X86ISD::PINSRB",
                 SDTypeProfile<1, 3, [SDTCisVT<0, v16i8>, SDTCisSameAs<0,1>,
                                      SDTCisVT<2, i32>, SDTCisPtrTy<3>]>>;
def X86pinsrw  : SDNode<"X86ISD::PINSRW",
                 SDTypeProfile<1, 3, [SDTCisVT<0, v8i16>, SDTCisSameAs<0,1>,
                                      SDTCisVT<2, i32>, SDTCisPtrTy<3>]>>;
def X86insertps : SDNode<"X86ISD::INSERTPS",
                 SDTypeProfile<1, 3, [SDTCisVT<0, v4f32>, SDTCisSameAs<0,1>,
                                      SDTCisVT<2, v4f32>, SDTCisVT<3, i8>]>>;
def X86vzmovl  : SDNode<"X86ISD::VZEXT_MOVL",
                 SDTypeProfile<1, 1, [SDTCisSameAs<0,1>]>>;

def X86vzload  : SDNode<"X86ISD::VZEXT_LOAD", SDTLoad,
                        [SDNPHasChain, SDNPMayLoad, SDNPMemOperand]>;

def X86vzext   : SDNode<"X86ISD::VZEXT",
                         SDTypeProfile<1, 1, [SDTCisVec<0>, SDTCisVec<1>,
                                              SDTCisInt<0>, SDTCisInt<1>,
                                              SDTCisOpSmallerThanOp<1, 0>]>>;

def X86vsext   : SDNode<"X86ISD::VSEXT",
                         SDTypeProfile<1, 1, [SDTCisVec<0>, SDTCisVec<1>,
                                              SDTCisInt<0>, SDTCisInt<1>,
                                              SDTCisOpSmallerThanOp<1, 0>]>>;

def SDTVtrunc    : SDTypeProfile<1, 1, [SDTCisVec<0>, SDTCisVec<1>,
                                       SDTCisInt<0>, SDTCisInt<1>,
                                       SDTCisOpSmallerThanOp<0, 1>]>;

def X86vtrunc    : SDNode<"X86ISD::VTRUNC",   SDTVtrunc>;
def X86vtruncs   : SDNode<"X86ISD::VTRUNCS",  SDTVtrunc>;
def X86vtruncus  : SDNode<"X86ISD::VTRUNCUS", SDTVtrunc>;

def X86trunc    : SDNode<"X86ISD::TRUNC",
                         SDTypeProfile<1, 1, [SDTCisInt<0>, SDTCisInt<1>,
                                              SDTCisOpSmallerThanOp<0, 1>]>>;
def X86vfpext  : SDNode<"X86ISD::VFPEXT",
                        SDTypeProfile<1, 1, [SDTCisVec<0>, SDTCisVec<1>,
                                             SDTCisFP<0>, SDTCisFP<1>,
                                             SDTCisOpSmallerThanOp<1, 0>]>>;
def X86vfpround: SDNode<"X86ISD::VFPROUND",
                        SDTypeProfile<1, 1, [SDTCisVec<0>, SDTCisVec<1>,
                                             SDTCisFP<0>, SDTCisFP<1>,
                                             SDTCisOpSmallerThanOp<0, 1>]>>;

def X86fround: SDNode<"X86ISD::VFPROUND",
                        SDTypeProfile<1, 2, [SDTCisFP<0>, SDTCisFP<1>,SDTCisFP<2>,
                                             SDTCVecEltisVT<0, f32>,
                                             SDTCVecEltisVT<1, f64>,
                                             SDTCVecEltisVT<2, f64>,
                                             SDTCisOpSmallerThanOp<0, 1>]>>;
def X86froundRnd: SDNode<"X86ISD::VFPROUND",
                        SDTypeProfile<1, 3, [SDTCisFP<0>, SDTCisFP<1>,SDTCisFP<2>,
                                             SDTCVecEltisVT<0, f32>,
                                             SDTCVecEltisVT<1, f64>,
                                             SDTCVecEltisVT<2, f64>,
                                             SDTCisOpSmallerThanOp<0, 1>,
                                             SDTCisInt<3>]>>;

def X86fpext  : SDNode<"X86ISD::VFPEXT",
                        SDTypeProfile<1, 2, [SDTCisFP<0>, SDTCisFP<1>,SDTCisFP<2>,
                                             SDTCVecEltisVT<0, f64>,
                                             SDTCVecEltisVT<1, f32>,
                                             SDTCVecEltisVT<2, f32>,
                                             SDTCisOpSmallerThanOp<1, 0>]>>;

def X86fpextRnd  : SDNode<"X86ISD::VFPEXT",
                        SDTypeProfile<1, 3, [SDTCisFP<0>, SDTCisFP<1>,SDTCisFP<2>,
                                             SDTCVecEltisVT<0, f64>,
                                             SDTCVecEltisVT<1, f32>,
                                             SDTCVecEltisVT<2, f32>,
                                             SDTCisOpSmallerThanOp<1, 0>,
                                             SDTCisInt<3>]>>;

def X86vshldq  : SDNode<"X86ISD::VSHLDQ",    SDTIntShiftOp>;
def X86vshrdq  : SDNode<"X86ISD::VSRLDQ",    SDTIntShiftOp>;
def X86cmpp    : SDNode<"X86ISD::CMPP",      SDTX86VFCMP>;
def X86pcmpeq  : SDNode<"X86ISD::PCMPEQ", SDTIntBinOp, [SDNPCommutative]>;
def X86pcmpgt  : SDNode<"X86ISD::PCMPGT", SDTIntBinOp>;

def X86IntCmpMask : SDTypeProfile<1, 2,
    [SDTCisVec<0>, SDTCisSameAs<1, 2>, SDTCisInt<1>]>;
def X86pcmpeqm  : SDNode<"X86ISD::PCMPEQM", X86IntCmpMask, [SDNPCommutative]>;
def X86pcmpgtm  : SDNode<"X86ISD::PCMPGTM", X86IntCmpMask>;

def X86CmpMaskCC :
      SDTypeProfile<1, 3, [SDTCisVec<0>, SDTCVecEltisVT<0, i1>,
                       SDTCisVec<1>, SDTCisSameAs<2, 1>,
                       SDTCisSameNumEltsAs<0, 1>, SDTCisVT<3, i8>]>;
def X86CmpMaskCCRound :
      SDTypeProfile<1, 4, [SDTCisVec<0>,SDTCVecEltisVT<0, i1>,
                       SDTCisVec<1>, SDTCisSameAs<2, 1>,
                       SDTCisSameNumEltsAs<0, 1>, SDTCisVT<3, i8>,
                       SDTCisInt<4>]>;
def X86CmpMaskCCScalar :
      SDTypeProfile<1, 3, [SDTCisInt<0>, SDTCisSameAs<1, 2>, SDTCisVT<3, i8>]>;

def X86CmpMaskCCScalarRound :
      SDTypeProfile<1, 4, [SDTCisInt<0>, SDTCisSameAs<1, 2>, SDTCisVT<3, i8>,
                           SDTCisInt<4>]>;

def X86cmpm     : SDNode<"X86ISD::CMPM",     X86CmpMaskCC>;
def X86cmpmRnd  : SDNode<"X86ISD::CMPM_RND", X86CmpMaskCCRound>;
def X86cmpmu    : SDNode<"X86ISD::CMPMU",    X86CmpMaskCC>;
def X86cmpms    : SDNode<"X86ISD::FSETCC",   X86CmpMaskCCScalar>;
def X86cmpmsRnd : SDNode<"X86ISD::FSETCC",   X86CmpMaskCCScalarRound>;

def X86vshl    : SDNode<"X86ISD::VSHL",
                        SDTypeProfile<1, 2, [SDTCisVec<0>, SDTCisSameAs<0,1>,
                                      SDTCisVec<2>]>>;
def X86vsrl    : SDNode<"X86ISD::VSRL",
                        SDTypeProfile<1, 2, [SDTCisVec<0>, SDTCisSameAs<0,1>,
                                      SDTCisVec<2>]>>;
def X86vsra    : SDNode<"X86ISD::VSRA",
                        SDTypeProfile<1, 2, [SDTCisVec<0>, SDTCisSameAs<0,1>,
                                      SDTCisVec<2>]>>;

def X86vshli   : SDNode<"X86ISD::VSHLI", SDTIntShiftOp>;
def X86vsrli   : SDNode<"X86ISD::VSRLI", SDTIntShiftOp>;
def X86vsrai   : SDNode<"X86ISD::VSRAI", SDTIntShiftOp>;

def X86vrotli  : SDNode<"X86ISD::VROTLI", SDTIntShiftOp>;
def X86vrotri  : SDNode<"X86ISD::VROTRI", SDTIntShiftOp>;

def X86vprot   : SDNode<"X86ISD::VPROT",
                        SDTypeProfile<1, 2, [SDTCisVec<0>, SDTCisSameAs<0,1>,
                                             SDTCisSameAs<0,2>]>>;
def X86vproti  : SDNode<"X86ISD::VPROTI",
                        SDTypeProfile<1, 2, [SDTCisVec<0>, SDTCisSameAs<0,1>,
                                             SDTCisVT<2, i8>]>>;

def X86vpshl   : SDNode<"X86ISD::VPSHL",
                        SDTypeProfile<1, 2, [SDTCisVec<0>, SDTCisSameAs<0,1>,
                                             SDTCisSameAs<0,2>]>>;
def X86vpsha   : SDNode<"X86ISD::VPSHA",
                        SDTypeProfile<1, 2, [SDTCisVec<0>, SDTCisSameAs<0,1>,
                                             SDTCisSameAs<0,2>]>>;

def X86vpcom   : SDNode<"X86ISD::VPCOM",
                        SDTypeProfile<1, 3, [SDTCisVec<0>, SDTCisSameAs<0,1>,
                                             SDTCisSameAs<0,2>,
                                             SDTCisVT<3, i8>]>>;
def X86vpcomu  : SDNode<"X86ISD::VPCOMU",
                        SDTypeProfile<1, 3, [SDTCisVec<0>, SDTCisSameAs<0,1>,
                                             SDTCisSameAs<0,2>,
                                             SDTCisVT<3, i8>]>>;

def X86vpperm : SDNode<"X86ISD::VPPERM",
                        SDTypeProfile<1, 3, [SDTCisVT<0, v16i8>, SDTCisSameAs<0,1>,
                                             SDTCisSameAs<0,2>]>>;

def SDTX86CmpPTest : SDTypeProfile<1, 2, [SDTCisVT<0, i32>,
                                          SDTCisVec<1>,
                                          SDTCisSameAs<2, 1>]>;

def SDTX86Testm : SDTypeProfile<1, 2, [SDTCisVec<0>, SDTCisVec<1>,
                                       SDTCisSameAs<2, 1>, SDTCVecEltisVT<0, i1>,
                                       SDTCisSameNumEltsAs<0, 1>]>;

def X86addus   : SDNode<"X86ISD::ADDUS", SDTIntBinOp>;
def X86subus   : SDNode<"X86ISD::SUBUS", SDTIntBinOp>;
def X86adds    : SDNode<"X86ISD::ADDS", SDTIntBinOp>;
def X86subs    : SDNode<"X86ISD::SUBS", SDTIntBinOp>;
def X86mulhrs  : SDNode<"X86ISD::MULHRS" , SDTIntBinOp>;
def X86avg     : SDNode<"X86ISD::AVG" , SDTIntBinOp>;
def X86ptest   : SDNode<"X86ISD::PTEST", SDTX86CmpPTest>;
def X86testp   : SDNode<"X86ISD::TESTP", SDTX86CmpPTest>;
def X86kortest : SDNode<"X86ISD::KORTEST", SDTX86CmpPTest>;
def X86ktest   : SDNode<"X86ISD::KTEST", SDTX86CmpPTest>;
def X86testm   : SDNode<"X86ISD::TESTM", SDTX86Testm, [SDNPCommutative]>;
def X86testnm  : SDNode<"X86ISD::TESTNM", SDTX86Testm, [SDNPCommutative]>;

def X86movmsk : SDNode<"X86ISD::MOVMSK",
                        SDTypeProfile<1, 1, [SDTCisVT<0, i32>, SDTCisVec<1>]>>;

def X86select  : SDNode<"X86ISD::SELECT"     , SDTSelect>;

def X86pmuludq : SDNode<"X86ISD::PMULUDQ",
                        SDTypeProfile<1, 2, [SDTCVecEltisVT<0, i64>,
                                             SDTCVecEltisVT<1, i32>,
                                             SDTCisSameSizeAs<0,1>,
                                             SDTCisSameAs<1,2>]>>;
def X86pmuldq  : SDNode<"X86ISD::PMULDQ",
                        SDTypeProfile<1, 2, [SDTCVecEltisVT<0, i64>,
                                             SDTCVecEltisVT<1, i32>,
                                             SDTCisSameSizeAs<0,1>,
                                             SDTCisSameAs<1,2>]>>;

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

def SDTShuff2OpM : SDTypeProfile<1, 2, [SDTCisVec<0>, SDTCisSameAs<0,1>,
                                        SDTCisSameSizeAs<0,2>,
                                        SDTCisSameNumEltsAs<0,2>]>;
def SDTShuff2OpI : SDTypeProfile<1, 2, [SDTCisVec<0>,
                                 SDTCisSameAs<0,1>, SDTCisVT<2, i8>]>;
def SDTShuff3OpI : SDTypeProfile<1, 3, [SDTCisVec<0>, SDTCisSameAs<0,1>,
                                 SDTCisSameAs<0,2>, SDTCisVT<3, i8>]>;
def SDTFPBinOpImmRound: SDTypeProfile<1, 4, [SDTCisVec<0>, SDTCisSameAs<0,1>,
                             SDTCisSameAs<0,2>, SDTCisInt<3>, SDTCisInt<4>]>;
def SDTFPTernaryOpImmRound: SDTypeProfile<1, 5, [SDTCisVec<0>, SDTCisSameAs<0,1>,
                             SDTCisSameAs<0,2>, SDTCisVec<3>, SDTCisInt<4>, SDTCisInt<5>]>;
def SDTFPTernaryOpImmRounds: SDTypeProfile<1, 5, [SDTCisSameAs<0,1>,
                             SDTCisSameAs<0,2>,SDTCisInt<3>, SDTCisInt<4>, SDTCisInt<5>]>;
def SDTFPUnaryOpImmRound: SDTypeProfile<1, 3, [SDTCisVec<0>, SDTCisSameAs<0,1>,
                              SDTCisInt<2>, SDTCisInt<3>]>;

def SDTVBroadcast  : SDTypeProfile<1, 1, [SDTCisVec<0>]>;
def SDTVBroadcastm : SDTypeProfile<1, 1, [SDTCisVec<0>,
                                          SDTCisInt<0>, SDTCisInt<1>]>;

def SDTBlend : SDTypeProfile<1, 3, [SDTCisVec<0>, SDTCisSameAs<0,1>,
                             SDTCisSameAs<1,2>, SDTCisVT<3, i8>]>;

def SDTTernlog  : SDTypeProfile<1, 4, [SDTCisVec<0>, SDTCisSameAs<0,1>,
                                SDTCisSameAs<0,2>, SDTCisSameAs<0,3>,
                                SDTCisVT<4, i8>]>;

def SDTFPBinOpRound : SDTypeProfile<1, 3, [      // fadd_round, fmul_round, etc.
  SDTCisSameAs<0, 1>, SDTCisSameAs<0, 2>, SDTCisFP<0>, SDTCisInt<3>]>;

def SDTFPUnaryOpRound : SDTypeProfile<1, 2, [      // fsqrt_round, fgetexp_round, etc.
  SDTCisSameAs<0, 1>, SDTCisFP<0>, SDTCisInt<2>]>;

def SDTFma : SDTypeProfile<1, 3, [SDTCisSameAs<0,1>,
                           SDTCisSameAs<1,2>, SDTCisSameAs<1,3>]>;
def SDTFmaRound : SDTypeProfile<1, 4, [SDTCisSameAs<0,1>,
                           SDTCisSameAs<1,2>, SDTCisSameAs<1,3>, SDTCisInt<4>]>;
def STDFp1SrcRm : SDTypeProfile<1, 2, [SDTCisSameAs<0,1>,
                           SDTCisVec<0>, SDTCisVT<2, i32>]>;
def STDFp2SrcRm : SDTypeProfile<1, 3, [SDTCisSameAs<0,1>,
                           SDTCisVec<0>, SDTCisVT<3, i32>]>;
def STDFp3SrcRm : SDTypeProfile<1, 4, [SDTCisSameAs<0,1>,
                           SDTCisVec<0>, SDTCisVT<3, i32>, SDTCisVT<4, i32>]>;

def X86PAlignr : SDNode<"X86ISD::PALIGNR", SDTShuff3OpI>;
def X86VAlign  : SDNode<"X86ISD::VALIGN", SDTShuff3OpI>;

def X86Abs      : SDNode<"X86ISD::ABS", SDTIntUnaryOp>;
def X86Conflict : SDNode<"X86ISD::CONFLICT", SDTIntUnaryOp>;

def X86PShufd  : SDNode<"X86ISD::PSHUFD", SDTShuff2OpI>;
def X86PShufhw : SDNode<"X86ISD::PSHUFHW", SDTShuff2OpI>;
def X86PShuflw : SDNode<"X86ISD::PSHUFLW", SDTShuff2OpI>;

def X86Shufp   : SDNode<"X86ISD::SHUFP", SDTShuff3OpI>;
def X86Shuf128 : SDNode<"X86ISD::SHUF128", SDTShuff3OpI>;

def X86Movddup  : SDNode<"X86ISD::MOVDDUP", SDTShuff1Op>;
def X86Movshdup : SDNode<"X86ISD::MOVSHDUP", SDTShuff1Op>;
def X86Movsldup : SDNode<"X86ISD::MOVSLDUP", SDTShuff1Op>;

def X86Movsd : SDNode<"X86ISD::MOVSD", SDTShuff2Op>;
def X86Movss : SDNode<"X86ISD::MOVSS", SDTShuff2Op>;

def X86Movlhps : SDNode<"X86ISD::MOVLHPS", SDTShuff2Op>;
def X86Movlhpd : SDNode<"X86ISD::MOVLHPD", SDTShuff2Op>;
def X86Movhlps : SDNode<"X86ISD::MOVHLPS", SDTShuff2Op>;

def X86Movlps : SDNode<"X86ISD::MOVLPS", SDTShuff2Op>;
def X86Movlpd : SDNode<"X86ISD::MOVLPD", SDTShuff2Op>;

def SDTPack : SDTypeProfile<1, 2, [SDTCisVec<0>, SDTCisVec<1>,
                                   SDTCisSameSizeAs<0,1>,
                                   SDTCisSameAs<1,2>]>;
def X86Packss : SDNode<"X86ISD::PACKSS", SDTPack>;
def X86Packus : SDNode<"X86ISD::PACKUS", SDTPack>;

def X86Unpckl : SDNode<"X86ISD::UNPCKL", SDTShuff2Op>;
def X86Unpckh : SDNode<"X86ISD::UNPCKH", SDTShuff2Op>;

def X86vpmaddubsw  : SDNode<"X86ISD::VPMADDUBSW" , SDTPack>;
def X86vpmaddwd    : SDNode<"X86ISD::VPMADDWD"   , SDTPack>;

def X86VPermilpv  : SDNode<"X86ISD::VPERMILPV", SDTShuff2OpM>;
def X86VPermilpi  : SDNode<"X86ISD::VPERMILPI", SDTShuff2OpI>;
def X86VPermv     : SDNode<"X86ISD::VPERMV",
                           SDTypeProfile<1, 2, [SDTCisVec<0>, SDTCisInt<1>,
                                                SDTCisSameNumEltsAs<0,1>,
                                                SDTCisSameSizeAs<0,1>,
                                                SDTCisSameAs<0,2>]>>;
def X86VPermi     : SDNode<"X86ISD::VPERMI",    SDTShuff2OpI>;
def X86VPermt2     : SDNode<"X86ISD::VPERMV3",
                    SDTypeProfile<1, 3, [SDTCisVec<0>,
                                         SDTCisSameAs<0,1>, SDTCisInt<2>,
                                         SDTCisVec<2>, SDTCisSameNumEltsAs<0, 2>,
                                         SDTCisSameSizeAs<0,2>,
                                         SDTCisSameAs<0,3>]>, []>;

def X86VPermi2X   : SDNode<"X86ISD::VPERMIV3",
                    SDTypeProfile<1, 3, [SDTCisVec<0>, SDTCisInt<1>,
                                         SDTCisVec<1>, SDTCisSameNumEltsAs<0, 1>,
                                         SDTCisSameSizeAs<0,1>,
                                         SDTCisSameAs<0,2>,
                                         SDTCisSameAs<0,3>]>, []>;

def X86vpternlog  : SDNode<"X86ISD::VPTERNLOG", SDTTernlog>;

def X86VPerm2x128 : SDNode<"X86ISD::VPERM2X128", SDTShuff3OpI>;

def X86VFixupimm   : SDNode<"X86ISD::VFIXUPIMM", SDTFPTernaryOpImmRound>;
def X86VFixupimmScalar   : SDNode<"X86ISD::VFIXUPIMMS", SDTFPTernaryOpImmRounds>;
def X86VRange      : SDNode<"X86ISD::VRANGE",    SDTFPBinOpImmRound>;
def X86VReduce     : SDNode<"X86ISD::VREDUCE",   SDTFPUnaryOpImmRound>;
def X86VRndScale   : SDNode<"X86ISD::VRNDSCALE", SDTFPUnaryOpImmRound>;
def X86VGetMant    : SDNode<"X86ISD::VGETMANT",  SDTFPUnaryOpImmRound>;
def X86Vfpclass    : SDNode<"X86ISD::VFPCLASS",
                       SDTypeProfile<1, 2, [SDTCisVec<0>, SDTCVecEltisVT<0, i1>,
                                            SDTCisVec<1>, SDTCisFP<1>,
                                            SDTCisSameNumEltsAs<0,1>,
                                            SDTCisVT<2, i32>]>, []>;
def X86Vfpclasss   : SDNode<"X86ISD::VFPCLASSS",
                       SDTypeProfile<1, 2, [SDTCisVT<0, i1>,
                                            SDTCisFP<1>, SDTCisVT<2, i32>]>,[]>;

def X86SubVBroadcast : SDNode<"X86ISD::SUBV_BROADCAST",
                    SDTypeProfile<1, 1, [SDTCisVec<0>, SDTCisVec<1>,
                                         SDTCisSubVecOfVec<1, 0>]>, []>;
// SDTCisSubVecOfVec restriction cannot be applied for 128 bit version of VBROADCASTI32x2.
def X86SubV32x2Broadcast : SDNode<"X86ISD::SUBV_BROADCAST",
                    SDTypeProfile<1, 1, [SDTCisVec<0>,
                                         SDTCisSameAs<0,1>]>, []>;

def X86VBroadcast : SDNode<"X86ISD::VBROADCAST", SDTVBroadcast>;
def X86VBroadcastm : SDNode<"X86ISD::VBROADCASTM", SDTVBroadcastm>;
def X86Vinsert   : SDNode<"X86ISD::VINSERT",  SDTypeProfile<1, 3,
                              [SDTCisSameAs<0, 1>, SDTCisEltOfVec<2, 1>,
                               SDTCisPtrTy<3>]>, []>;
def X86Vextract   : SDNode<"X86ISD::VEXTRACT",  SDTypeProfile<1, 2,
                              [SDTCisEltOfVec<0, 1>, SDTCisVec<1>,
                               SDTCisPtrTy<2>]>, []>;

def X86Blendi    : SDNode<"X86ISD::BLENDI",   SDTBlend>;

def X86Addsub    : SDNode<"X86ISD::ADDSUB", SDTFPBinOp>;

def X86faddRnd   : SDNode<"X86ISD::FADD_RND",  SDTFPBinOpRound>;
def X86fsubRnd   : SDNode<"X86ISD::FSUB_RND",  SDTFPBinOpRound>;
def X86fmulRnd   : SDNode<"X86ISD::FMUL_RND",  SDTFPBinOpRound>;
def X86fdivRnd   : SDNode<"X86ISD::FDIV_RND",  SDTFPBinOpRound>;
def X86fmaxRnd   : SDNode<"X86ISD::FMAX_RND",       SDTFPBinOpRound>;
def X86scalef    : SDNode<"X86ISD::SCALEF",         SDTFPBinOpRound>;
def X86fminRnd   : SDNode<"X86ISD::FMIN_RND",       SDTFPBinOpRound>;
def X86fsqrtRnd     : SDNode<"X86ISD::FSQRT_RND",   SDTFPUnaryOpRound>;
def X86fsqrtRnds    : SDNode<"X86ISD::FSQRT_RND",   STDFp2SrcRm>;
def X86fgetexpRnd   : SDNode<"X86ISD::FGETEXP_RND", SDTFPUnaryOpRound>;
def X86fgetexpRnds  : SDNode<"X86ISD::FGETEXP_RND", STDFp2SrcRm>;

def X86Fmadd     : SDNode<"X86ISD::FMADD",     SDTFma>;
def X86Fnmadd    : SDNode<"X86ISD::FNMADD",    SDTFma>;
def X86Fmsub     : SDNode<"X86ISD::FMSUB",     SDTFma>;
def X86Fnmsub    : SDNode<"X86ISD::FNMSUB",    SDTFma>;
def X86Fmaddsub  : SDNode<"X86ISD::FMADDSUB",  SDTFma>;
def X86Fmsubadd  : SDNode<"X86ISD::FMSUBADD",  SDTFma>;

def X86FmaddRnd     : SDNode<"X86ISD::FMADD_RND",     SDTFmaRound>;
def X86FnmaddRnd    : SDNode<"X86ISD::FNMADD_RND",    SDTFmaRound>;
def X86FmsubRnd     : SDNode<"X86ISD::FMSUB_RND",     SDTFmaRound>;
def X86FnmsubRnd    : SDNode<"X86ISD::FNMSUB_RND",    SDTFmaRound>;
def X86FmaddsubRnd  : SDNode<"X86ISD::FMADDSUB_RND",  SDTFmaRound>;
def X86FmsubaddRnd  : SDNode<"X86ISD::FMSUBADD_RND",  SDTFmaRound>;

def x86vpmadd52l     : SDNode<"X86ISD::VPMADD52L",     SDTFma>;
def x86vpmadd52h     : SDNode<"X86ISD::VPMADD52H",     SDTFma>;

def X86rsqrt28   : SDNode<"X86ISD::RSQRT28",  STDFp1SrcRm>;
def X86rcp28     : SDNode<"X86ISD::RCP28",    STDFp1SrcRm>;
def X86exp2      : SDNode<"X86ISD::EXP2",     STDFp1SrcRm>;

def X86rsqrt28s  : SDNode<"X86ISD::RSQRT28",   STDFp2SrcRm>;
def X86rcp28s    : SDNode<"X86ISD::RCP28",     STDFp2SrcRm>;
def X86RndScales : SDNode<"X86ISD::VRNDSCALE", STDFp3SrcRm>;
def X86Reduces   : SDNode<"X86ISD::VREDUCE",   STDFp3SrcRm>;
def X86GetMants  : SDNode<"X86ISD::VGETMANT",  STDFp3SrcRm>;

def SDT_PCMPISTRI : SDTypeProfile<2, 3, [SDTCisVT<0, i32>, SDTCisVT<1, i32>,
                                         SDTCisVT<2, v16i8>, SDTCisVT<3, v16i8>,
                                         SDTCisVT<4, i8>]>;
def SDT_PCMPESTRI : SDTypeProfile<2, 5, [SDTCisVT<0, i32>, SDTCisVT<1, i32>,
                                         SDTCisVT<2, v16i8>, SDTCisVT<3, i32>,
                                         SDTCisVT<4, v16i8>, SDTCisVT<5, i32>,
                                         SDTCisVT<6, i8>]>;

def X86pcmpistri : SDNode<"X86ISD::PCMPISTRI", SDT_PCMPISTRI>;
def X86pcmpestri : SDNode<"X86ISD::PCMPESTRI", SDT_PCMPESTRI>;

def X86compress: SDNode<"X86ISD::COMPRESS", SDTypeProfile<1, 1,
                              [SDTCisSameAs<0, 1>, SDTCisVec<1>]>, []>;
def X86expand  : SDNode<"X86ISD::EXPAND", SDTypeProfile<1, 1,
                              [SDTCisSameAs<0, 1>, SDTCisVec<1>]>, []>;

def SDTintToFPRound: SDTypeProfile<1, 3, [SDTCisVec<0>, SDTCisFP<0>,
                                          SDTCisSameAs<0,1>, SDTCisInt<2>,
                                          SDTCisVT<3, i32>]>;

def SDTDoubleToInt: SDTypeProfile<1, 1, [SDTCisVec<0>, SDTCisVec<1>,
                                         SDTCisInt<0>, SDTCVecEltisVT<1, f64>]>;
def SDTFloatToInt: SDTypeProfile<1, 1, [SDTCisVec<0>, SDTCisVec<1>,
                                         SDTCisInt<0>, SDTCVecEltisVT<1, f32>]>;

def SDTDoubleToIntRnd: SDTypeProfile<1, 2, [SDTCisVec<0>, SDTCisVec<1>,
                                         SDTCisInt<0>, SDTCVecEltisVT<1, f64>]>;
def SDTSDoubleToIntRnd: SDTypeProfile<1, 2, [SDTCisInt<0>,SDTCisFP<1>, 
                                             SDTCVecEltisVT<1, f64>, SDTCisInt<2>]>;
def SDTFloatToIntRnd: SDTypeProfile<1, 2, [SDTCisVec<0>, SDTCisVec<1>,
                                         SDTCisInt<0>, SDTCVecEltisVT<1, f32>]>;
def SDTSFloatToIntRnd: SDTypeProfile<1, 2, [SDTCisInt<0>, SDTCisFP<1>,
                                            SDTCVecEltisVT<1, f32>, SDTCisInt<2>]>;
def SDTVintToFPRound: SDTypeProfile<1, 2, [SDTCisVec<0>, SDTCisVec<1>,
                                           SDTCisFP<0>, SDTCVecEltisVT<1, i32>,
                                           SDTCisInt<2>]>;
def SDTVlongToFPRound: SDTypeProfile<1, 2, [SDTCisVec<0>, SDTCisVec<1>,
                                           SDTCisFP<0>, SDTCVecEltisVT<1, i64>,
                                           SDTCisInt<2>]>;

def SDTVFPToIntRound: SDTypeProfile<1, 2, [SDTCisVec<0>, SDTCisVec<1>,
                                           SDTCisFP<1>, SDTCVecEltisVT<0, i32>,
                                           SDTCisInt<2>]>;
def SDTVFPToLongRound: SDTypeProfile<1, 2, [SDTCisVec<0>, SDTCisVec<1>,
                                           SDTCisFP<1>, SDTCVecEltisVT<0, i64>,
                                           SDTCisInt<2>]>;

// Scalar
def X86SintToFpRnd  : SDNode<"X86ISD::SINT_TO_FP_RND",  SDTintToFPRound>;
def X86UintToFpRnd  : SDNode<"X86ISD::UINT_TO_FP_RND",  SDTintToFPRound>;

def X86cvttss2IntRnd      : SDNode<"X86ISD::FP_TO_SINT_RND",  SDTSFloatToIntRnd>;
def X86cvttss2UIntRnd     : SDNode<"X86ISD::FP_TO_UINT_RND",  SDTSFloatToIntRnd>;
def X86cvttsd2IntRnd      : SDNode<"X86ISD::FP_TO_SINT_RND",  SDTSDoubleToIntRnd>;
def X86cvttsd2UIntRnd     : SDNode<"X86ISD::FP_TO_UINT_RND",  SDTSDoubleToIntRnd>;

def  X86cvtsd2si  : SDNode<"X86ISD::SCALAR_FP_TO_SINT_RND", SDTSDoubleToIntRnd>;
def  X86cvtsd2usi : SDNode<"X86ISD::SCALAR_FP_TO_UINT_RND", SDTSDoubleToIntRnd>;
def  X86cvtss2si  : SDNode<"X86ISD::SCALAR_FP_TO_SINT_RND", SDTSFloatToIntRnd>;
def  X86cvtss2usi : SDNode<"X86ISD::SCALAR_FP_TO_UINT_RND", SDTSFloatToIntRnd>;

// Vector with rounding mode

// cvtt fp-to-int staff
def X86VFpToSintRnd   : SDNode<"ISD::FP_TO_SINT",  SDTVFPToIntRound>;
def X86VFpToUintRnd   : SDNode<"ISD::FP_TO_UINT",  SDTVFPToIntRound>;
def X86VFpToSlongRnd  : SDNode<"ISD::FP_TO_SINT",  SDTVFPToLongRound>;
def X86VFpToUlongRnd  : SDNode<"ISD::FP_TO_UINT",  SDTVFPToLongRound>;

def X86VSintToFpRnd   : SDNode<"ISD::SINT_TO_FP",  SDTVintToFPRound>;
def X86VUintToFpRnd   : SDNode<"ISD::UINT_TO_FP",  SDTVintToFPRound>;
def X86VSlongToFpRnd  : SDNode<"ISD::SINT_TO_FP",  SDTVlongToFPRound>;
def X86VUlongToFpRnd  : SDNode<"ISD::UINT_TO_FP",  SDTVlongToFPRound>;

// cvt fp-to-int staff
def X86cvtps2IntRnd      : SDNode<"X86ISD::FP_TO_SINT_RND",  SDTFloatToIntRnd>;
def X86cvtps2UIntRnd     : SDNode<"X86ISD::FP_TO_UINT_RND",  SDTFloatToIntRnd>;
def X86cvtpd2IntRnd      : SDNode<"X86ISD::FP_TO_SINT_RND",  SDTDoubleToIntRnd>;
def X86cvtpd2UIntRnd     : SDNode<"X86ISD::FP_TO_UINT_RND",  SDTDoubleToIntRnd>;

// Vector without rounding mode
def X86cvtps2Int      : SDNode<"X86ISD::FP_TO_SINT_RND",  SDTFloatToInt>;
def X86cvtps2UInt     : SDNode<"X86ISD::FP_TO_UINT_RND",  SDTFloatToInt>;
def X86cvtpd2Int      : SDNode<"X86ISD::FP_TO_SINT_RND",  SDTDoubleToInt>;
def X86cvtpd2UInt     : SDNode<"X86ISD::FP_TO_UINT_RND",  SDTDoubleToInt>;

def X86cvtph2ps     : SDNode<"ISD::FP16_TO_FP",
                              SDTypeProfile<1, 2, [SDTCisVec<0>, SDTCisVec<1>,
                                                   SDTCVecEltisVT<0, f32>,
                                                   SDTCVecEltisVT<1, i16>,
                                                   SDTCisFP<0>,
                                                   SDTCisVT<2, i32>]> >;

def X86cvtps2ph   : SDNode<"ISD::FP_TO_FP16",
                        SDTypeProfile<1, 3, [SDTCisVec<0>, SDTCisVec<1>,
                                             SDTCVecEltisVT<0, i16>,
                                             SDTCVecEltisVT<1, f32>,
                                             SDTCisFP<1>, SDTCisVT<2, i32>,
                                             SDTCisVT<3, i32>]> >;
def X86vfpextRnd  : SDNode<"X86ISD::VFPEXT",
                        SDTypeProfile<1, 2, [SDTCisVec<0>, SDTCisVec<1>,
                                             SDTCisFP<0>, SDTCisFP<1>,
                                             SDTCVecEltisVT<0, f64>,
                                             SDTCVecEltisVT<1, f32>,
                                             SDTCisOpSmallerThanOp<1, 0>,
                                             SDTCisVT<2, i32>]>>;
def X86vfproundRnd: SDNode<"X86ISD::VFPROUND",
                        SDTypeProfile<1, 2, [SDTCisVec<0>, SDTCisVec<1>,
                                             SDTCisFP<0>, SDTCisFP<1>,
                                             SDTCVecEltisVT<0, f32>,
                                             SDTCVecEltisVT<1, f64>,
                                             SDTCisOpSmallerThanOp<0, 1>,
                                             SDTCisVT<2, i32>]>>;

def X86cvt2mask   : SDNode<"X86ISD::CVT2MASK", SDTIntTruncOp>;

//===----------------------------------------------------------------------===//
// SSE Complex Patterns
//===----------------------------------------------------------------------===//

// These are 'extloads' from a scalar to the low element of a vector, zeroing
// the top elements.  These are used for the SSE 'ss' and 'sd' instruction
// forms.
def sse_load_f32 : ComplexPattern<v4f32, 5, "selectScalarSSELoad", [],
                                  [SDNPHasChain, SDNPMayLoad, SDNPMemOperand,
                                   SDNPWantRoot]>;
def sse_load_f64 : ComplexPattern<v2f64, 5, "selectScalarSSELoad", [],
                                  [SDNPHasChain, SDNPMayLoad, SDNPMemOperand,
                                   SDNPWantRoot]>;

def ssmem : Operand<v4f32> {
  let PrintMethod = "printf32mem";
  let MIOperandInfo = (ops ptr_rc, i8imm, ptr_rc_nosp, i32imm, i8imm);
  let ParserMatchClass = X86Mem32AsmOperand;
  let OperandType = "OPERAND_MEMORY";
}
def sdmem : Operand<v2f64> {
  let PrintMethod = "printf64mem";
  let MIOperandInfo = (ops ptr_rc, i8imm, ptr_rc_nosp, i32imm, i8imm);
  let ParserMatchClass = X86Mem64AsmOperand;
  let OperandType = "OPERAND_MEMORY";
}

//===----------------------------------------------------------------------===//
// SSE pattern fragments
//===----------------------------------------------------------------------===//

// 128-bit load pattern fragments
// NOTE: all 128-bit integer vector loads are promoted to v2i64
def loadv4f32    : PatFrag<(ops node:$ptr), (v4f32 (load node:$ptr))>;
def loadv2f64    : PatFrag<(ops node:$ptr), (v2f64 (load node:$ptr))>;
def loadv2i64    : PatFrag<(ops node:$ptr), (v2i64 (load node:$ptr))>;

// 256-bit load pattern fragments
// NOTE: all 256-bit integer vector loads are promoted to v4i64
def loadv8f32    : PatFrag<(ops node:$ptr), (v8f32 (load node:$ptr))>;
def loadv4f64    : PatFrag<(ops node:$ptr), (v4f64 (load node:$ptr))>;
def loadv4i64    : PatFrag<(ops node:$ptr), (v4i64 (load node:$ptr))>;

// 512-bit load pattern fragments
def loadv16f32   : PatFrag<(ops node:$ptr), (v16f32 (load node:$ptr))>;
def loadv8f64    : PatFrag<(ops node:$ptr), (v8f64 (load node:$ptr))>;
def loadv64i8    : PatFrag<(ops node:$ptr), (v64i8 (load node:$ptr))>;
def loadv32i16   : PatFrag<(ops node:$ptr), (v32i16 (load node:$ptr))>;
def loadv16i32   : PatFrag<(ops node:$ptr), (v16i32 (load node:$ptr))>;
def loadv8i64    : PatFrag<(ops node:$ptr), (v8i64 (load node:$ptr))>;

// 128-/256-/512-bit extload pattern fragments
def extloadv2f32 : PatFrag<(ops node:$ptr), (v2f64 (extloadvf32 node:$ptr))>;
def extloadv4f32 : PatFrag<(ops node:$ptr), (v4f64 (extloadvf32 node:$ptr))>;
def extloadv8f32 : PatFrag<(ops node:$ptr), (v8f64 (extloadvf32 node:$ptr))>;

// These are needed to match a scalar load that is used in a vector-only
// math instruction such as the FP logical ops: andps, andnps, orps, xorps.
// The memory operand is required to be a 128-bit load, so it must be converted
// from a vector to a scalar.
def loadf32_128 : PatFrag<(ops node:$ptr),
  (f32 (extractelt (loadv4f32 node:$ptr), (iPTR 0)))>;
def loadf64_128 : PatFrag<(ops node:$ptr),
  (f64 (extractelt (loadv2f64 node:$ptr), (iPTR 0)))>;

// Like 'store', but always requires 128-bit vector alignment.
def alignedstore : PatFrag<(ops node:$val, node:$ptr),
                           (store node:$val, node:$ptr), [{
  return cast<StoreSDNode>(N)->getAlignment() >= 16;
}]>;

// Like 'store', but always requires 256-bit vector alignment.
def alignedstore256 : PatFrag<(ops node:$val, node:$ptr),
                              (store node:$val, node:$ptr), [{
  return cast<StoreSDNode>(N)->getAlignment() >= 32;
}]>;

// Like 'store', but always requires 512-bit vector alignment.
def alignedstore512 : PatFrag<(ops node:$val, node:$ptr),
                              (store node:$val, node:$ptr), [{
  return cast<StoreSDNode>(N)->getAlignment() >= 64;
}]>;

// Like 'load', but always requires 128-bit vector alignment.
def alignedload : PatFrag<(ops node:$ptr), (load node:$ptr), [{
  return cast<LoadSDNode>(N)->getAlignment() >= 16;
}]>;

// Like 'load', but always requires 256-bit vector alignment.
def alignedload256 : PatFrag<(ops node:$ptr), (load node:$ptr), [{
  return cast<LoadSDNode>(N)->getAlignment() >= 32;
}]>;

// Like 'load', but always requires 512-bit vector alignment.
def alignedload512 : PatFrag<(ops node:$ptr), (load node:$ptr), [{
  return cast<LoadSDNode>(N)->getAlignment() >= 64;
}]>;

def alignedloadfsf32 : PatFrag<(ops node:$ptr),
                               (f32 (alignedload node:$ptr))>;
def alignedloadfsf64 : PatFrag<(ops node:$ptr),
                               (f64 (alignedload node:$ptr))>;

// 128-bit aligned load pattern fragments
// NOTE: all 128-bit integer vector loads are promoted to v2i64
def alignedloadv4f32 : PatFrag<(ops node:$ptr),
                               (v4f32 (alignedload node:$ptr))>;
def alignedloadv2f64 : PatFrag<(ops node:$ptr),
                               (v2f64 (alignedload node:$ptr))>;
def alignedloadv2i64 : PatFrag<(ops node:$ptr),
                               (v2i64 (alignedload node:$ptr))>;

// 256-bit aligned load pattern fragments
// NOTE: all 256-bit integer vector loads are promoted to v4i64
def alignedloadv8f32 : PatFrag<(ops node:$ptr),
                               (v8f32 (alignedload256 node:$ptr))>;
def alignedloadv4f64 : PatFrag<(ops node:$ptr),
                               (v4f64 (alignedload256 node:$ptr))>;
def alignedloadv4i64 : PatFrag<(ops node:$ptr),
                               (v4i64 (alignedload256 node:$ptr))>;

// 512-bit aligned load pattern fragments
def alignedloadv16f32 : PatFrag<(ops node:$ptr),
                                (v16f32 (alignedload512 node:$ptr))>;
def alignedloadv16i32 : PatFrag<(ops node:$ptr),
                                (v16i32 (alignedload512 node:$ptr))>;
def alignedloadv8f64  : PatFrag<(ops node:$ptr),
                                (v8f64  (alignedload512 node:$ptr))>;
def alignedloadv8i64  : PatFrag<(ops node:$ptr),
                                (v8i64  (alignedload512 node:$ptr))>;

// Like 'load', but uses special alignment checks suitable for use in
// memory operands in most SSE instructions, which are required to
// be naturally aligned on some targets but not on others.  If the subtarget
// allows unaligned accesses, match any load, though this may require
// setting a feature bit in the processor (on startup, for example).
// Opteron 10h and later implement such a feature.
def memop : PatFrag<(ops node:$ptr), (load node:$ptr), [{
  return    Subtarget->hasSSEUnalignedMem()
         || cast<LoadSDNode>(N)->getAlignment() >= 16;
}]>;

def memopfsf32 : PatFrag<(ops node:$ptr), (f32   (memop node:$ptr))>;
def memopfsf64 : PatFrag<(ops node:$ptr), (f64   (memop node:$ptr))>;

// 128-bit memop pattern fragments
// NOTE: all 128-bit integer vector loads are promoted to v2i64
def memopv4f32 : PatFrag<(ops node:$ptr), (v4f32 (memop node:$ptr))>;
def memopv2f64 : PatFrag<(ops node:$ptr), (v2f64 (memop node:$ptr))>;
def memopv2i64 : PatFrag<(ops node:$ptr), (v2i64 (memop node:$ptr))>;

// These are needed to match a scalar memop that is used in a vector-only
// math instruction such as the FP logical ops: andps, andnps, orps, xorps.
// The memory operand is required to be a 128-bit load, so it must be converted
// from a vector to a scalar.
def memopfsf32_128 : PatFrag<(ops node:$ptr),
  (f32 (extractelt (memopv4f32 node:$ptr), (iPTR 0)))>;
def memopfsf64_128 : PatFrag<(ops node:$ptr),
  (f64 (extractelt (memopv2f64 node:$ptr), (iPTR 0)))>;


// SSSE3 uses MMX registers for some instructions. They aren't aligned on a
// 16-byte boundary.
// FIXME: 8 byte alignment for mmx reads is not required
def memop64 : PatFrag<(ops node:$ptr), (load node:$ptr), [{
  return cast<LoadSDNode>(N)->getAlignment() >= 8;
}]>;

def memopmmx  : PatFrag<(ops node:$ptr), (x86mmx  (memop64 node:$ptr))>;

def mgatherv4i32 : PatFrag<(ops node:$src1, node:$src2, node:$src3),
  (masked_gather node:$src1, node:$src2, node:$src3) , [{
  if (MaskedGatherSDNode *Mgt = dyn_cast<MaskedGatherSDNode>(N))
    return (Mgt->getIndex().getValueType() == MVT::v4i32 ||
            Mgt->getBasePtr().getValueType() == MVT::v4i32);
  return false;
}]>;

def mgatherv8i32 : PatFrag<(ops node:$src1, node:$src2, node:$src3),
  (masked_gather node:$src1, node:$src2, node:$src3) , [{
  if (MaskedGatherSDNode *Mgt = dyn_cast<MaskedGatherSDNode>(N))
    return (Mgt->getIndex().getValueType() == MVT::v8i32 ||
            Mgt->getBasePtr().getValueType() == MVT::v8i32);
  return false;
}]>;

def mgatherv2i64 : PatFrag<(ops node:$src1, node:$src2, node:$src3),
  (masked_gather node:$src1, node:$src2, node:$src3) , [{
  if (MaskedGatherSDNode *Mgt = dyn_cast<MaskedGatherSDNode>(N))
    return (Mgt->getIndex().getValueType() == MVT::v2i64 ||
            Mgt->getBasePtr().getValueType() == MVT::v2i64);
  return false;
}]>;
def mgatherv4i64 : PatFrag<(ops node:$src1, node:$src2, node:$src3),
  (masked_gather node:$src1, node:$src2, node:$src3) , [{
  if (MaskedGatherSDNode *Mgt = dyn_cast<MaskedGatherSDNode>(N))
    return (Mgt->getIndex().getValueType() == MVT::v4i64 ||
            Mgt->getBasePtr().getValueType() == MVT::v4i64);
  return false;
}]>;
def mgatherv8i64 : PatFrag<(ops node:$src1, node:$src2, node:$src3),
  (masked_gather node:$src1, node:$src2, node:$src3) , [{
  if (MaskedGatherSDNode *Mgt = dyn_cast<MaskedGatherSDNode>(N))
    return (Mgt->getIndex().getValueType() == MVT::v8i64 ||
            Mgt->getBasePtr().getValueType() == MVT::v8i64);
  return false;
}]>;
def mgatherv16i32 : PatFrag<(ops node:$src1, node:$src2, node:$src3),
  (masked_gather node:$src1, node:$src2, node:$src3) , [{
  if (MaskedGatherSDNode *Mgt = dyn_cast<MaskedGatherSDNode>(N))
    return (Mgt->getIndex().getValueType() == MVT::v16i32 ||
            Mgt->getBasePtr().getValueType() == MVT::v16i32);
  return false;
}]>;

def mscatterv2i64 : PatFrag<(ops node:$src1, node:$src2, node:$src3),
  (masked_scatter node:$src1, node:$src2, node:$src3) , [{
  if (MaskedScatterSDNode *Sc = dyn_cast<MaskedScatterSDNode>(N))
    return (Sc->getIndex().getValueType() == MVT::v2i64 ||
            Sc->getBasePtr().getValueType() == MVT::v2i64);
  return false;
}]>;

def mscatterv4i32 : PatFrag<(ops node:$src1, node:$src2, node:$src3),
  (masked_scatter node:$src1, node:$src2, node:$src3) , [{
  if (MaskedScatterSDNode *Sc = dyn_cast<MaskedScatterSDNode>(N))
    return (Sc->getIndex().getValueType() == MVT::v4i32 ||
            Sc->getBasePtr().getValueType() == MVT::v4i32);
  return false;
}]>;

def mscatterv4i64 : PatFrag<(ops node:$src1, node:$src2, node:$src3),
  (masked_scatter node:$src1, node:$src2, node:$src3) , [{
  if (MaskedScatterSDNode *Sc = dyn_cast<MaskedScatterSDNode>(N))
    return (Sc->getIndex().getValueType() == MVT::v4i64 ||
            Sc->getBasePtr().getValueType() == MVT::v4i64);
  return false;
}]>;

def mscatterv8i32 : PatFrag<(ops node:$src1, node:$src2, node:$src3),
  (masked_scatter node:$src1, node:$src2, node:$src3) , [{
  if (MaskedScatterSDNode *Sc = dyn_cast<MaskedScatterSDNode>(N))
    return (Sc->getIndex().getValueType() == MVT::v8i32 ||
            Sc->getBasePtr().getValueType() == MVT::v8i32);
  return false;
}]>;

def mscatterv8i64 : PatFrag<(ops node:$src1, node:$src2, node:$src3),
  (masked_scatter node:$src1, node:$src2, node:$src3) , [{
  if (MaskedScatterSDNode *Sc = dyn_cast<MaskedScatterSDNode>(N))
    return (Sc->getIndex().getValueType() == MVT::v8i64 ||
            Sc->getBasePtr().getValueType() == MVT::v8i64);
  return false;
}]>;
def mscatterv16i32 : PatFrag<(ops node:$src1, node:$src2, node:$src3),
  (masked_scatter node:$src1, node:$src2, node:$src3) , [{
  if (MaskedScatterSDNode *Sc = dyn_cast<MaskedScatterSDNode>(N))
    return (Sc->getIndex().getValueType() == MVT::v16i32 ||
            Sc->getBasePtr().getValueType() == MVT::v16i32);
  return false;
}]>;

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

// 512-bit bitconvert pattern fragments
def bc_v16i32 : PatFrag<(ops node:$in), (v16i32 (bitconvert node:$in))>;
def bc_v8i64 : PatFrag<(ops node:$in), (v8i64 (bitconvert node:$in))>;
def bc_v8f64 : PatFrag<(ops node:$in), (v8f64 (bitconvert node:$in))>;
def bc_v16f32 : PatFrag<(ops node:$in), (v16f32 (bitconvert node:$in))>;

def vzmovl_v2i64 : PatFrag<(ops node:$src),
                           (bitconvert (v2i64 (X86vzmovl
                             (v2i64 (scalar_to_vector (loadi64 node:$src))))))>;
def vzmovl_v4i32 : PatFrag<(ops node:$src),
                           (bitconvert (v4i32 (X86vzmovl
                             (v4i32 (scalar_to_vector (loadi32 node:$src))))))>;

def vzload_v2i64 : PatFrag<(ops node:$src),
                           (bitconvert (v2i64 (X86vzload node:$src)))>;


def fp32imm0 : PatLeaf<(f32 fpimm), [{
  return N->isExactlyValue(+0.0);
}]>;

def I8Imm : SDNodeXForm<imm, [{
  // Transformation function: get the low 8 bits.
  return getI8Imm((uint8_t)N->getZExtValue(), SDLoc(N));
}]>;

def FROUND_NO_EXC : ImmLeaf<i32, [{ return Imm == 8; }]>;
def FROUND_CURRENT : ImmLeaf<i32, [{
  return Imm == X86::STATIC_ROUNDING::CUR_DIRECTION;
}]>;

// BYTE_imm - Transform bit immediates into byte immediates.
def BYTE_imm  : SDNodeXForm<imm, [{
  // Transformation function: imm >> 3
  return getI32Imm(N->getZExtValue() >> 3, SDLoc(N));
}]>;

// EXTRACT_get_vextract128_imm xform function: convert extract_subvector index
// to VEXTRACTF128/VEXTRACTI128 imm.
def EXTRACT_get_vextract128_imm : SDNodeXForm<extract_subvector, [{
  return getI8Imm(X86::getExtractVEXTRACT128Immediate(N), SDLoc(N));
}]>;

// INSERT_get_vinsert128_imm xform function: convert insert_subvector index to
// VINSERTF128/VINSERTI128 imm.
def INSERT_get_vinsert128_imm : SDNodeXForm<insert_subvector, [{
  return getI8Imm(X86::getInsertVINSERT128Immediate(N), SDLoc(N));
}]>;

// EXTRACT_get_vextract256_imm xform function: convert extract_subvector index
// to VEXTRACTF64x4 imm.
def EXTRACT_get_vextract256_imm : SDNodeXForm<extract_subvector, [{
  return getI8Imm(X86::getExtractVEXTRACT256Immediate(N), SDLoc(N));
}]>;

// INSERT_get_vinsert256_imm xform function: convert insert_subvector index to
// VINSERTF64x4 imm.
def INSERT_get_vinsert256_imm : SDNodeXForm<insert_subvector, [{
  return getI8Imm(X86::getInsertVINSERT256Immediate(N), SDLoc(N));
}]>;

def vextract128_extract : PatFrag<(ops node:$bigvec, node:$index),
                                   (extract_subvector node:$bigvec,
                                                      node:$index), [{
  return X86::isVEXTRACT128Index(N);
}], EXTRACT_get_vextract128_imm>;

def vinsert128_insert : PatFrag<(ops node:$bigvec, node:$smallvec,
                                      node:$index),
                                 (insert_subvector node:$bigvec, node:$smallvec,
                                                   node:$index), [{
  return X86::isVINSERT128Index(N);
}], INSERT_get_vinsert128_imm>;


def vextract256_extract : PatFrag<(ops node:$bigvec, node:$index),
                                   (extract_subvector node:$bigvec,
                                                      node:$index), [{
  return X86::isVEXTRACT256Index(N);
}], EXTRACT_get_vextract256_imm>;

def vinsert256_insert : PatFrag<(ops node:$bigvec, node:$smallvec,
                                      node:$index),
                                 (insert_subvector node:$bigvec, node:$smallvec,
                                                   node:$index), [{
  return X86::isVINSERT256Index(N);
}], INSERT_get_vinsert256_imm>;

def masked_load_aligned128 : PatFrag<(ops node:$src1, node:$src2, node:$src3),
                         (masked_load node:$src1, node:$src2, node:$src3), [{
  if (auto *Load = dyn_cast<MaskedLoadSDNode>(N))
    return Load->getAlignment() >= 16;
  return false;
}]>;

def masked_load_aligned256 : PatFrag<(ops node:$src1, node:$src2, node:$src3),
                         (masked_load node:$src1, node:$src2, node:$src3), [{
  if (auto *Load = dyn_cast<MaskedLoadSDNode>(N))
    return Load->getAlignment() >= 32;
  return false;
}]>;

def masked_load_aligned512 : PatFrag<(ops node:$src1, node:$src2, node:$src3),
                         (masked_load node:$src1, node:$src2, node:$src3), [{
  if (auto *Load = dyn_cast<MaskedLoadSDNode>(N))
    return Load->getAlignment() >= 64;
  return false;
}]>;

def masked_load_unaligned : PatFrag<(ops node:$src1, node:$src2, node:$src3),
                         (masked_load node:$src1, node:$src2, node:$src3), [{
  return isa<MaskedLoadSDNode>(N);
}]>;

// Masked store fragments.
// X86mstore can't be implemented in core DAG files because some targets
// do not support vector types (llvm-tblgen will fail).
def X86mstore : PatFrag<(ops node:$src1, node:$src2, node:$src3),
                        (masked_store node:$src1, node:$src2, node:$src3), [{
  return !cast<MaskedStoreSDNode>(N)->isTruncatingStore();
}]>;

def masked_store_aligned128 : PatFrag<(ops node:$src1, node:$src2, node:$src3),
                         (X86mstore node:$src1, node:$src2, node:$src3), [{
  if (auto *Store = dyn_cast<MaskedStoreSDNode>(N))
    return Store->getAlignment() >= 16;
  return false;
}]>;

def masked_store_aligned256 : PatFrag<(ops node:$src1, node:$src2, node:$src3),
                         (X86mstore node:$src1, node:$src2, node:$src3), [{
  if (auto *Store = dyn_cast<MaskedStoreSDNode>(N))
    return Store->getAlignment() >= 32;
  return false;
}]>;

def masked_store_aligned512 : PatFrag<(ops node:$src1, node:$src2, node:$src3),
                         (X86mstore node:$src1, node:$src2, node:$src3), [{
  if (auto *Store = dyn_cast<MaskedStoreSDNode>(N))
    return Store->getAlignment() >= 64;
  return false;
}]>;

def masked_store_unaligned : PatFrag<(ops node:$src1, node:$src2, node:$src3),
                         (X86mstore node:$src1, node:$src2, node:$src3), [{
  return isa<MaskedStoreSDNode>(N);
}]>;

// masked truncstore fragments
// X86mtruncstore can't be implemented in core DAG files because some targets
// doesn't support vector type ( llvm-tblgen will fail)
def X86mtruncstore : PatFrag<(ops node:$src1, node:$src2, node:$src3),
                             (masked_store node:$src1, node:$src2, node:$src3), [{
    return cast<MaskedStoreSDNode>(N)->isTruncatingStore();
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

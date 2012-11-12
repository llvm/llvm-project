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
// MMX Pattern Fragments
//===----------------------------------------------------------------------===//

def load_mmx : PatFrag<(ops node:$ptr), (x86mmx (load node:$ptr))>;
def bc_mmx  : PatFrag<(ops node:$in), (x86mmx  (bitconvert node:$in))>;

//===----------------------------------------------------------------------===//
// SSE specific DAG Nodes.
//===----------------------------------------------------------------------===//

def SDTX86FPShiftOp : SDTypeProfile<1, 2, [ SDTCisSameAs<0, 1>,
                                            SDTCisFP<0>, SDTCisInt<2> ]>;
def SDTX86VFCMP : SDTypeProfile<1, 3, [SDTCisInt<0>, SDTCisSameAs<1, 2>,
                                       SDTCisFP<1>, SDTCisVT<3, i8>]>;

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
def X86frsqrt  : SDNode<"X86ISD::FRSQRT",    SDTFPUnaryOp>;
def X86frcp    : SDNode<"X86ISD::FRCP",      SDTFPUnaryOp>;
def X86fsrl    : SDNode<"X86ISD::FSRL",      SDTX86FPShiftOp>;
def X86fgetsign: SDNode<"X86ISD::FGETSIGNx86",SDTFPToIntOp>;
def X86fhadd   : SDNode<"X86ISD::FHADD",     SDTFPBinOp>;
def X86fhsub   : SDNode<"X86ISD::FHSUB",     SDTFPBinOp>;
def X86hadd    : SDNode<"X86ISD::HADD",      SDTIntBinOp>;
def X86hsub    : SDNode<"X86ISD::HSUB",      SDTIntBinOp>;
def X86comi    : SDNode<"X86ISD::COMI",      SDTX86CmpTest>;
def X86ucomi   : SDNode<"X86ISD::UCOMI",     SDTX86CmpTest>;
def X86cmpss   : SDNode<"X86ISD::FSETCCss",    SDTX86Cmpss>;
def X86cmpsd   : SDNode<"X86ISD::FSETCCsd",    SDTX86Cmpsd>;
def X86pshufb  : SDNode<"X86ISD::PSHUFB",
                 SDTypeProfile<1, 2, [SDTCisVec<0>, SDTCisSameAs<0,1>,
                                      SDTCisSameAs<0,2>]>>;
def X86andnp   : SDNode<"X86ISD::ANDNP",
                 SDTypeProfile<1, 2, [SDTCisVec<0>, SDTCisSameAs<0,1>,
                                      SDTCisSameAs<0,2>]>>;
def X86psign   : SDNode<"X86ISD::PSIGN",
                 SDTypeProfile<1, 2, [SDTCisVec<0>, SDTCisSameAs<0,1>,
                                      SDTCisSameAs<0,2>]>>;
def X86pextrb  : SDNode<"X86ISD::PEXTRB",
                 SDTypeProfile<1, 2, [SDTCisVT<0, i32>, SDTCisPtrTy<2>]>>;
def X86pextrw  : SDNode<"X86ISD::PEXTRW",
                 SDTypeProfile<1, 2, [SDTCisVT<0, i32>, SDTCisPtrTy<2>]>>;
def X86pinsrb  : SDNode<"X86ISD::PINSRB",
                 SDTypeProfile<1, 3, [SDTCisVT<0, v16i8>, SDTCisSameAs<0,1>,
                                      SDTCisVT<2, i32>, SDTCisPtrTy<3>]>>;
def X86pinsrw  : SDNode<"X86ISD::PINSRW",
                 SDTypeProfile<1, 3, [SDTCisVT<0, v8i16>, SDTCisSameAs<0,1>,
                                      SDTCisVT<2, i32>, SDTCisPtrTy<3>]>>;
def X86insrtps : SDNode<"X86ISD::INSERTPS",
                 SDTypeProfile<1, 3, [SDTCisVT<0, v4f32>, SDTCisSameAs<0,1>,
                                      SDTCisVT<2, v4f32>, SDTCisPtrTy<3>]>>;
def X86vzmovl  : SDNode<"X86ISD::VZEXT_MOVL",
                 SDTypeProfile<1, 1, [SDTCisSameAs<0,1>]>>;

def X86vzmovly  : SDNode<"X86ISD::VZEXT_MOVL",
                 SDTypeProfile<1, 1, [SDTCisVec<0>, SDTCisVec<1>,
                                      SDTCisOpSmallerThanOp<1, 0> ]>>;

def X86vsmovl  : SDNode<"X86ISD::VSEXT_MOVL",
                 SDTypeProfile<1, 1,
                 [SDTCisVec<0>, SDTCisInt<1>, SDTCisInt<0>]>>;

def X86vzload  : SDNode<"X86ISD::VZEXT_LOAD", SDTLoad,
                        [SDNPHasChain, SDNPMayLoad, SDNPMemOperand]>;

def X86vzext   : SDNode<"X86ISD::VZEXT",
                         SDTypeProfile<1, 1, [SDTCisVec<0>, SDTCisVec<1>,
                                              SDTCisInt<0>, SDTCisInt<1>]>>;

def X86vsext   : SDNode<"X86ISD::VSEXT",
                         SDTypeProfile<1, 1, [SDTCisVec<0>, SDTCisVec<1>,
                                              SDTCisInt<0>, SDTCisInt<1>]>>;

def X86vfpext  : SDNode<"X86ISD::VFPEXT",
                        SDTypeProfile<1, 1, [SDTCisVec<0>, SDTCisVec<1>,
                                             SDTCisFP<0>, SDTCisFP<1>]>>;
def X86vfpround: SDNode<"X86ISD::VFPROUND",
                        SDTypeProfile<1, 1, [SDTCisVec<0>, SDTCisVec<1>,
                                             SDTCisFP<0>, SDTCisFP<1>]>>;

def X86vshldq  : SDNode<"X86ISD::VSHLDQ",    SDTIntShiftOp>;
def X86vshrdq  : SDNode<"X86ISD::VSRLDQ",    SDTIntShiftOp>;
def X86cmpp    : SDNode<"X86ISD::CMPP",      SDTX86VFCMP>;
def X86pcmpeq  : SDNode<"X86ISD::PCMPEQ", SDTIntBinOp, [SDNPCommutative]>;
def X86pcmpgt  : SDNode<"X86ISD::PCMPGT", SDTIntBinOp>;

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

def SDTX86CmpPTest : SDTypeProfile<1, 2, [SDTCisVT<0, i32>,
                                          SDTCisVec<1>,
                                          SDTCisSameAs<2, 1>]>;
def X86ptest   : SDNode<"X86ISD::PTEST", SDTX86CmpPTest>;
def X86testp   : SDNode<"X86ISD::TESTP", SDTX86CmpPTest>;

def X86pmuludq : SDNode<"X86ISD::PMULUDQ",
                        SDTypeProfile<1, 2, [SDTCisVec<0>, SDTCisVec<1>,
                                      SDTCisSameAs<1,2>]>>;

// Specific shuffle nodes - At some point ISD::VECTOR_SHUFFLE will always get
// translated into one of the target nodes below during lowering.
// Note: this is a work in progress...
def SDTShuff1Op : SDTypeProfile<1, 1, [SDTCisVec<0>, SDTCisSameAs<0,1>]>;
def SDTShuff2Op : SDTypeProfile<1, 2, [SDTCisVec<0>, SDTCisSameAs<0,1>,
                                SDTCisSameAs<0,2>]>;

def SDTShuff2OpI : SDTypeProfile<1, 2, [SDTCisVec<0>,
                                 SDTCisSameAs<0,1>, SDTCisInt<2>]>;
def SDTShuff3OpI : SDTypeProfile<1, 3, [SDTCisVec<0>, SDTCisSameAs<0,1>,
                                 SDTCisSameAs<0,2>, SDTCisInt<3>]>;

def SDTVBroadcast : SDTypeProfile<1, 1, [SDTCisVec<0>]>;
def SDTBlend : SDTypeProfile<1, 3, [SDTCisVec<0>, SDTCisSameAs<0,1>,
                             SDTCisSameAs<1,2>, SDTCisVT<3, i32>]>;

def SDTFma : SDTypeProfile<1, 3, [SDTCisSameAs<0,1>,
                           SDTCisSameAs<1,2>, SDTCisSameAs<1,3>]>;

def X86PAlign : SDNode<"X86ISD::PALIGN", SDTShuff3OpI>;

def X86PShufd  : SDNode<"X86ISD::PSHUFD", SDTShuff2OpI>;
def X86PShufhw : SDNode<"X86ISD::PSHUFHW", SDTShuff2OpI>;
def X86PShuflw : SDNode<"X86ISD::PSHUFLW", SDTShuff2OpI>;

def X86Shufp : SDNode<"X86ISD::SHUFP", SDTShuff3OpI>;

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

def X86Unpckl : SDNode<"X86ISD::UNPCKL", SDTShuff2Op>;
def X86Unpckh : SDNode<"X86ISD::UNPCKH", SDTShuff2Op>;

def X86VPermilp  : SDNode<"X86ISD::VPERMILP", SDTShuff2OpI>;
def X86VPermv    : SDNode<"X86ISD::VPERMV",   SDTShuff2Op>;
def X86VPermi    : SDNode<"X86ISD::VPERMI",   SDTShuff2OpI>;

def X86VPerm2x128 : SDNode<"X86ISD::VPERM2X128", SDTShuff3OpI>;

def X86VBroadcast : SDNode<"X86ISD::VBROADCAST", SDTVBroadcast>;

def X86Blendpw   : SDNode<"X86ISD::BLENDPW",   SDTBlend>;
def X86Blendps   : SDNode<"X86ISD::BLENDPS",   SDTBlend>;
def X86Blendpd   : SDNode<"X86ISD::BLENDPD",   SDTBlend>;
def X86Fmadd     : SDNode<"X86ISD::FMADD",     SDTFma>;
def X86Fnmadd    : SDNode<"X86ISD::FNMADD",    SDTFma>;
def X86Fmsub     : SDNode<"X86ISD::FMSUB",     SDTFma>;
def X86Fnmsub    : SDNode<"X86ISD::FNMSUB",    SDTFma>;
def X86Fmaddsub  : SDNode<"X86ISD::FMADDSUB",  SDTFma>;
def X86Fmsubadd  : SDNode<"X86ISD::FMSUBADD",  SDTFma>;

def SDT_PCMPISTRI : SDTypeProfile<2, 3, [SDTCisVT<0, i32>, SDTCisVT<1, i32>,
                                         SDTCisVT<2, v16i8>, SDTCisVT<3, v16i8>,
                                         SDTCisVT<4, i8>]>;
def SDT_PCMPESTRI : SDTypeProfile<2, 5, [SDTCisVT<0, i32>, SDTCisVT<1, i32>,
                                         SDTCisVT<2, v16i8>, SDTCisVT<3, i32>,
                                         SDTCisVT<4, v16i8>, SDTCisVT<5, i32>,
                                         SDTCisVT<6, i8>]>;

def X86pcmpistri : SDNode<"X86ISD::PCMPISTRI", SDT_PCMPISTRI>;
def X86pcmpestri : SDNode<"X86ISD::PCMPESTRI", SDT_PCMPESTRI>;

//===----------------------------------------------------------------------===//
// SSE Complex Patterns
//===----------------------------------------------------------------------===//

// These are 'extloads' from a scalar to the low element of a vector, zeroing
// the top elements.  These are used for the SSE 'ss' and 'sd' instruction
// forms.
def sse_load_f32 : ComplexPattern<v4f32, 5, "SelectScalarSSELoad", [],
                                  [SDNPHasChain, SDNPMayLoad, SDNPMemOperand,
                                   SDNPWantRoot]>;
def sse_load_f64 : ComplexPattern<v2f64, 5, "SelectScalarSSELoad", [],
                                  [SDNPHasChain, SDNPMayLoad, SDNPMemOperand,
                                   SDNPWantRoot]>;

def ssmem : Operand<v4f32> {
  let PrintMethod = "printf32mem";
  let MIOperandInfo = (ops ptr_rc, i8imm, ptr_rc_nosp, i32imm, i8imm);
  let ParserMatchClass = X86MemAsmOperand;
  let OperandType = "OPERAND_MEMORY";
}
def sdmem : Operand<v2f64> {
  let PrintMethod = "printf64mem";
  let MIOperandInfo = (ops ptr_rc, i8imm, ptr_rc_nosp, i32imm, i8imm);
  let ParserMatchClass = X86MemAsmOperand;
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

// 128-/256-bit extload pattern fragments
def extloadv2f32 : PatFrag<(ops node:$ptr), (v2f64 (extloadvf32 node:$ptr))>;
def extloadv4f32 : PatFrag<(ops node:$ptr), (v4f64 (extloadvf32 node:$ptr))>;

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

// Like 'load', but always requires 128-bit vector alignment.
def alignedload : PatFrag<(ops node:$ptr), (load node:$ptr), [{
  return cast<LoadSDNode>(N)->getAlignment() >= 16;
}]>;

// Like 'X86vzload', but always requires 128-bit vector alignment.
def alignedX86vzload : PatFrag<(ops node:$ptr), (X86vzload node:$ptr), [{
  return cast<MemSDNode>(N)->getAlignment() >= 16;
}]>;

// Like 'load', but always requires 256-bit vector alignment.
def alignedload256 : PatFrag<(ops node:$ptr), (load node:$ptr), [{
  return cast<LoadSDNode>(N)->getAlignment() >= 32;
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

// Like 'load', but uses special alignment checks suitable for use in
// memory operands in most SSE instructions, which are required to
// be naturally aligned on some targets but not on others.  If the subtarget
// allows unaligned accesses, match any load, though this may require
// setting a feature bit in the processor (on startup, for example).
// Opteron 10h and later implement such a feature.
def memop : PatFrag<(ops node:$ptr), (load node:$ptr), [{
  return    Subtarget->hasVectorUAMem()
         || cast<LoadSDNode>(N)->getAlignment() >= 16;
}]>;

def memopfsf32 : PatFrag<(ops node:$ptr), (f32   (memop node:$ptr))>;
def memopfsf64 : PatFrag<(ops node:$ptr), (f64   (memop node:$ptr))>;

// 128-bit memop pattern fragments
// NOTE: all 128-bit integer vector loads are promoted to v2i64
def memopv4f32 : PatFrag<(ops node:$ptr), (v4f32 (memop node:$ptr))>;
def memopv2f64 : PatFrag<(ops node:$ptr), (v2f64 (memop node:$ptr))>;
def memopv2i64 : PatFrag<(ops node:$ptr), (v2i64 (memop node:$ptr))>;

// 256-bit memop pattern fragments
// NOTE: all 256-bit integer vector loads are promoted to v4i64
def memopv8f32 : PatFrag<(ops node:$ptr), (v8f32 (memop node:$ptr))>;
def memopv4f64 : PatFrag<(ops node:$ptr), (v4f64 (memop node:$ptr))>;
def memopv4i64 : PatFrag<(ops node:$ptr), (v4i64 (memop node:$ptr))>;

// SSSE3 uses MMX registers for some instructions. They aren't aligned on a
// 16-byte boundary.
// FIXME: 8 byte alignment for mmx reads is not required
def memop64 : PatFrag<(ops node:$ptr), (load node:$ptr), [{
  return cast<LoadSDNode>(N)->getAlignment() >= 8;
}]>;

def memopmmx  : PatFrag<(ops node:$ptr), (x86mmx  (memop64 node:$ptr))>;

// MOVNT Support
// Like 'store', but requires the non-temporal bit to be set
def nontemporalstore : PatFrag<(ops node:$val, node:$ptr),
                           (st node:$val, node:$ptr), [{
  if (StoreSDNode *ST = dyn_cast<StoreSDNode>(N))
    return ST->isNonTemporal();
  return false;
}]>;

def alignednontemporalstore : PatFrag<(ops node:$val, node:$ptr),
                                    (st node:$val, node:$ptr), [{
  if (StoreSDNode *ST = dyn_cast<StoreSDNode>(N))
    return ST->isNonTemporal() && !ST->isTruncatingStore() &&
           ST->getAddressingMode() == ISD::UNINDEXED &&
           ST->getAlignment() >= 16;
  return false;
}]>;

def unalignednontemporalstore : PatFrag<(ops node:$val, node:$ptr),
                                      (st node:$val, node:$ptr), [{
  if (StoreSDNode *ST = dyn_cast<StoreSDNode>(N))
    return ST->isNonTemporal() &&
           ST->getAlignment() < 16;
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

// BYTE_imm - Transform bit immediates into byte immediates.
def BYTE_imm  : SDNodeXForm<imm, [{
  // Transformation function: imm >> 3
  return getI32Imm(N->getZExtValue() >> 3);
}]>;

// EXTRACT_get_vextractf128_imm xform function: convert extract_subvector index
// to VEXTRACTF128 imm.
def EXTRACT_get_vextractf128_imm : SDNodeXForm<extract_subvector, [{
  return getI8Imm(X86::getExtractVEXTRACTF128Immediate(N));
}]>;

// INSERT_get_vinsertf128_imm xform function: convert insert_subvector index to
// VINSERTF128 imm.
def INSERT_get_vinsertf128_imm : SDNodeXForm<insert_subvector, [{
  return getI8Imm(X86::getInsertVINSERTF128Immediate(N));
}]>;

def vextractf128_extract : PatFrag<(ops node:$bigvec, node:$index),
                                   (extract_subvector node:$bigvec,
                                                      node:$index), [{
  return X86::isVEXTRACTF128Index(N);
}], EXTRACT_get_vextractf128_imm>;

def vinsertf128_insert : PatFrag<(ops node:$bigvec, node:$smallvec,
                                      node:$index),
                                 (insert_subvector node:$bigvec, node:$smallvec,
                                                   node:$index), [{
  return X86::isVINSERTF128Index(N);
}], INSERT_get_vinsertf128_imm>;


//====- X86InstrSSE.td - Describe the X86 Instruction Set --*- tablegen -*-===//
// 
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
// 
//===----------------------------------------------------------------------===//
//
// This file describes the X86 SSE instruction set, defining the instructions,
// and properties of the instructions which are needed for code generation,
// machine code emission, and analysis.
//
//===----------------------------------------------------------------------===//


//===----------------------------------------------------------------------===//
// SSE specific DAG Nodes.
//===----------------------------------------------------------------------===//

def SDTX86FPShiftOp : SDTypeProfile<1, 2, [ SDTCisSameAs<0, 1>,
                                            SDTCisFP<0>, SDTCisInt<2> ]>;
def SDTX86VFCMP : SDTypeProfile<1, 3, [SDTCisInt<0>, SDTCisSameAs<1, 2>,
                                       SDTCisFP<1>, SDTCisVT<3, i8>]>;

def X86fmin    : SDNode<"X86ISD::FMIN",      SDTFPBinOp>;
def X86fmax    : SDNode<"X86ISD::FMAX",      SDTFPBinOp>;
def X86fand    : SDNode<"X86ISD::FAND",      SDTFPBinOp,
                        [SDNPCommutative, SDNPAssociative]>;
def X86for     : SDNode<"X86ISD::FOR",       SDTFPBinOp,
                        [SDNPCommutative, SDNPAssociative]>;
def X86fxor    : SDNode<"X86ISD::FXOR",      SDTFPBinOp,
                        [SDNPCommutative, SDNPAssociative]>;
def X86frsqrt  : SDNode<"X86ISD::FRSQRT",    SDTFPUnaryOp>;
def X86frcp    : SDNode<"X86ISD::FRCP",      SDTFPUnaryOp>;
def X86fsrl    : SDNode<"X86ISD::FSRL",      SDTX86FPShiftOp>;
def X86comi    : SDNode<"X86ISD::COMI",      SDTX86CmpTest>;
def X86ucomi   : SDNode<"X86ISD::UCOMI",     SDTX86CmpTest>;
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
                                      SDTCisVT<2, f32>, SDTCisPtrTy<3>]>>;
def X86vzmovl  : SDNode<"X86ISD::VZEXT_MOVL",
                 SDTypeProfile<1, 1, [SDTCisSameAs<0,1>]>>;
def X86vzload  : SDNode<"X86ISD::VZEXT_LOAD", SDTLoad,
                        [SDNPHasChain, SDNPMayLoad]>;
def X86vshl    : SDNode<"X86ISD::VSHL",      SDTIntShiftOp>;
def X86vshr    : SDNode<"X86ISD::VSRL",      SDTIntShiftOp>;
def X86cmpps   : SDNode<"X86ISD::CMPPS",     SDTX86VFCMP>;
def X86cmppd   : SDNode<"X86ISD::CMPPD",     SDTX86VFCMP>;
def X86pcmpeqb : SDNode<"X86ISD::PCMPEQB", SDTIntBinOp, [SDNPCommutative]>;
def X86pcmpeqw : SDNode<"X86ISD::PCMPEQW", SDTIntBinOp, [SDNPCommutative]>;
def X86pcmpeqd : SDNode<"X86ISD::PCMPEQD", SDTIntBinOp, [SDNPCommutative]>;
def X86pcmpeqq : SDNode<"X86ISD::PCMPEQQ", SDTIntBinOp, [SDNPCommutative]>;
def X86pcmpgtb : SDNode<"X86ISD::PCMPGTB", SDTIntBinOp>;
def X86pcmpgtw : SDNode<"X86ISD::PCMPGTW", SDTIntBinOp>;
def X86pcmpgtd : SDNode<"X86ISD::PCMPGTD", SDTIntBinOp>;
def X86pcmpgtq : SDNode<"X86ISD::PCMPGTQ", SDTIntBinOp>;

//===----------------------------------------------------------------------===//
// SSE Complex Patterns
//===----------------------------------------------------------------------===//

// These are 'extloads' from a scalar to the low element of a vector, zeroing
// the top elements.  These are used for the SSE 'ss' and 'sd' instruction
// forms.
def sse_load_f32 : ComplexPattern<v4f32, 4, "SelectScalarSSELoad", [],
                                  [SDNPHasChain, SDNPMayLoad]>;
def sse_load_f64 : ComplexPattern<v2f64, 4, "SelectScalarSSELoad", [],
                                  [SDNPHasChain, SDNPMayLoad]>;

def ssmem : Operand<v4f32> {
  let PrintMethod = "printf32mem";
  let MIOperandInfo = (ops ptr_rc, i8imm, ptr_rc, i32imm);
}
def sdmem : Operand<v2f64> {
  let PrintMethod = "printf64mem";
  let MIOperandInfo = (ops ptr_rc, i8imm, ptr_rc, i32imm);
}

//===----------------------------------------------------------------------===//
// SSE pattern fragments
//===----------------------------------------------------------------------===//

def loadv4f32    : PatFrag<(ops node:$ptr), (v4f32 (load node:$ptr))>;
def loadv2f64    : PatFrag<(ops node:$ptr), (v2f64 (load node:$ptr))>;
def loadv4i32    : PatFrag<(ops node:$ptr), (v4i32 (load node:$ptr))>;
def loadv2i64    : PatFrag<(ops node:$ptr), (v2i64 (load node:$ptr))>;

// Like 'store', but always requires vector alignment.
def alignedstore : PatFrag<(ops node:$val, node:$ptr),
                           (store node:$val, node:$ptr), [{
  return cast<StoreSDNode>(N)->getAlignment() >= 16;
}]>;

// Like 'load', but always requires vector alignment.
def alignedload : PatFrag<(ops node:$ptr), (load node:$ptr), [{
  return cast<LoadSDNode>(N)->getAlignment() >= 16;
}]>;

def alignedloadfsf32 : PatFrag<(ops node:$ptr), (f32   (alignedload node:$ptr))>;
def alignedloadfsf64 : PatFrag<(ops node:$ptr), (f64   (alignedload node:$ptr))>;
def alignedloadv4f32 : PatFrag<(ops node:$ptr), (v4f32 (alignedload node:$ptr))>;
def alignedloadv2f64 : PatFrag<(ops node:$ptr), (v2f64 (alignedload node:$ptr))>;
def alignedloadv4i32 : PatFrag<(ops node:$ptr), (v4i32 (alignedload node:$ptr))>;
def alignedloadv2i64 : PatFrag<(ops node:$ptr), (v2i64 (alignedload node:$ptr))>;

// Like 'load', but uses special alignment checks suitable for use in
// memory operands in most SSE instructions, which are required to
// be naturally aligned on some targets but not on others.
// FIXME: Actually implement support for targets that don't require the
//        alignment. This probably wants a subtarget predicate.
def memop : PatFrag<(ops node:$ptr), (load node:$ptr), [{
  return cast<LoadSDNode>(N)->getAlignment() >= 16;
}]>;

def memopfsf32 : PatFrag<(ops node:$ptr), (f32   (memop node:$ptr))>;
def memopfsf64 : PatFrag<(ops node:$ptr), (f64   (memop node:$ptr))>;
def memopv4f32 : PatFrag<(ops node:$ptr), (v4f32 (memop node:$ptr))>;
def memopv2f64 : PatFrag<(ops node:$ptr), (v2f64 (memop node:$ptr))>;
def memopv4i32 : PatFrag<(ops node:$ptr), (v4i32 (memop node:$ptr))>;
def memopv2i64 : PatFrag<(ops node:$ptr), (v2i64 (memop node:$ptr))>;
def memopv16i8 : PatFrag<(ops node:$ptr), (v16i8 (memop node:$ptr))>;

// SSSE3 uses MMX registers for some instructions. They aren't aligned on a
// 16-byte boundary.
// FIXME: 8 byte alignment for mmx reads is not required
def memop64 : PatFrag<(ops node:$ptr), (load node:$ptr), [{
  return cast<LoadSDNode>(N)->getAlignment() >= 8;
}]>;

def memopv8i8  : PatFrag<(ops node:$ptr), (v8i8  (memop64 node:$ptr))>;
def memopv4i16 : PatFrag<(ops node:$ptr), (v4i16 (memop64 node:$ptr))>;
def memopv8i16 : PatFrag<(ops node:$ptr), (v8i16 (memop64 node:$ptr))>;
def memopv2i32 : PatFrag<(ops node:$ptr), (v2i32 (memop64 node:$ptr))>;

def bc_v4f32 : PatFrag<(ops node:$in), (v4f32 (bitconvert node:$in))>;
def bc_v2f64 : PatFrag<(ops node:$in), (v2f64 (bitconvert node:$in))>;
def bc_v16i8 : PatFrag<(ops node:$in), (v16i8 (bitconvert node:$in))>;
def bc_v8i16 : PatFrag<(ops node:$in), (v8i16 (bitconvert node:$in))>;
def bc_v4i32 : PatFrag<(ops node:$in), (v4i32 (bitconvert node:$in))>;
def bc_v2i64 : PatFrag<(ops node:$in), (v2i64 (bitconvert node:$in))>;

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

def PSxLDQ_imm  : SDNodeXForm<imm, [{
  // Transformation function: imm >> 3
  return getI32Imm(N->getZExtValue() >> 3);
}]>;

// SHUFFLE_get_shuf_imm xform function: convert vector_shuffle mask to PSHUF*,
// SHUFP* etc. imm.
def SHUFFLE_get_shuf_imm : SDNodeXForm<build_vector, [{
  return getI8Imm(X86::getShuffleSHUFImmediate(N));
}]>;

// SHUFFLE_get_pshufhw_imm xform function: convert vector_shuffle mask to 
// PSHUFHW imm.
def SHUFFLE_get_pshufhw_imm : SDNodeXForm<build_vector, [{
  return getI8Imm(X86::getShufflePSHUFHWImmediate(N));
}]>;

// SHUFFLE_get_pshuflw_imm xform function: convert vector_shuffle mask to 
// PSHUFLW imm.
def SHUFFLE_get_pshuflw_imm : SDNodeXForm<build_vector, [{
  return getI8Imm(X86::getShufflePSHUFLWImmediate(N));
}]>;

def SSE_splat_mask : PatLeaf<(build_vector), [{
  return X86::isSplatMask(N);
}], SHUFFLE_get_shuf_imm>;

def SSE_splat_lo_mask : PatLeaf<(build_vector), [{
  return X86::isSplatLoMask(N);
}]>;

def MOVDDUP_shuffle_mask : PatLeaf<(build_vector), [{
  return X86::isMOVDDUPMask(N);
}]>;

def MOVHLPS_shuffle_mask : PatLeaf<(build_vector), [{
  return X86::isMOVHLPSMask(N);
}]>;

def MOVHLPS_v_undef_shuffle_mask : PatLeaf<(build_vector), [{
  return X86::isMOVHLPS_v_undef_Mask(N);
}]>;

def MOVHP_shuffle_mask : PatLeaf<(build_vector), [{
  return X86::isMOVHPMask(N);
}]>;

def MOVLP_shuffle_mask : PatLeaf<(build_vector), [{
  return X86::isMOVLPMask(N);
}]>;

def MOVL_shuffle_mask : PatLeaf<(build_vector), [{
  return X86::isMOVLMask(N);
}]>;

def MOVSHDUP_shuffle_mask : PatLeaf<(build_vector), [{
  return X86::isMOVSHDUPMask(N);
}]>;

def MOVSLDUP_shuffle_mask : PatLeaf<(build_vector), [{
  return X86::isMOVSLDUPMask(N);
}]>;

def UNPCKL_shuffle_mask : PatLeaf<(build_vector), [{
  return X86::isUNPCKLMask(N);
}]>;

def UNPCKH_shuffle_mask : PatLeaf<(build_vector), [{
  return X86::isUNPCKHMask(N);
}]>;

def UNPCKL_v_undef_shuffle_mask : PatLeaf<(build_vector), [{
  return X86::isUNPCKL_v_undef_Mask(N);
}]>;

def UNPCKH_v_undef_shuffle_mask : PatLeaf<(build_vector), [{
  return X86::isUNPCKH_v_undef_Mask(N);
}]>;

def PSHUFD_shuffle_mask : PatLeaf<(build_vector), [{
  return X86::isPSHUFDMask(N);
}], SHUFFLE_get_shuf_imm>;

def PSHUFHW_shuffle_mask : PatLeaf<(build_vector), [{
  return X86::isPSHUFHWMask(N);
}], SHUFFLE_get_pshufhw_imm>;

def PSHUFLW_shuffle_mask : PatLeaf<(build_vector), [{
  return X86::isPSHUFLWMask(N);
}], SHUFFLE_get_pshuflw_imm>;

def SHUFP_unary_shuffle_mask : PatLeaf<(build_vector), [{
  return X86::isPSHUFDMask(N);
}], SHUFFLE_get_shuf_imm>;

def SHUFP_shuffle_mask : PatLeaf<(build_vector), [{
  return X86::isSHUFPMask(N);
}], SHUFFLE_get_shuf_imm>;

def PSHUFD_binary_shuffle_mask : PatLeaf<(build_vector), [{
  return X86::isSHUFPMask(N);
}], SHUFFLE_get_shuf_imm>;


//===----------------------------------------------------------------------===//
// SSE scalar FP Instructions
//===----------------------------------------------------------------------===//

// CMOV* - Used to implement the SSE SELECT DAG operation.  Expanded by the
// scheduler into a branch sequence.
// These are expanded by the scheduler.
let Uses = [EFLAGS], usesCustomDAGSchedInserter = 1 in {
  def CMOV_FR32 : I<0, Pseudo,
                    (outs FR32:$dst), (ins FR32:$t, FR32:$f, i8imm:$cond),
                    "#CMOV_FR32 PSEUDO!",
                    [(set FR32:$dst, (X86cmov FR32:$t, FR32:$f, imm:$cond,
                                                  EFLAGS))]>;
  def CMOV_FR64 : I<0, Pseudo,
                    (outs FR64:$dst), (ins FR64:$t, FR64:$f, i8imm:$cond),
                    "#CMOV_FR64 PSEUDO!",
                    [(set FR64:$dst, (X86cmov FR64:$t, FR64:$f, imm:$cond,
                                                  EFLAGS))]>;
  def CMOV_V4F32 : I<0, Pseudo,
                    (outs VR128:$dst), (ins VR128:$t, VR128:$f, i8imm:$cond),
                    "#CMOV_V4F32 PSEUDO!",
                    [(set VR128:$dst,
                      (v4f32 (X86cmov VR128:$t, VR128:$f, imm:$cond,
                                          EFLAGS)))]>;
  def CMOV_V2F64 : I<0, Pseudo,
                    (outs VR128:$dst), (ins VR128:$t, VR128:$f, i8imm:$cond),
                    "#CMOV_V2F64 PSEUDO!",
                    [(set VR128:$dst,
                      (v2f64 (X86cmov VR128:$t, VR128:$f, imm:$cond,
                                          EFLAGS)))]>;
  def CMOV_V2I64 : I<0, Pseudo,
                    (outs VR128:$dst), (ins VR128:$t, VR128:$f, i8imm:$cond),
                    "#CMOV_V2I64 PSEUDO!",
                    [(set VR128:$dst,
                      (v2i64 (X86cmov VR128:$t, VR128:$f, imm:$cond,
                                          EFLAGS)))]>;
}

//===----------------------------------------------------------------------===//
// SSE1 Instructions
//===----------------------------------------------------------------------===//

// Move Instructions
let neverHasSideEffects = 1 in
def MOVSSrr : SSI<0x10, MRMSrcReg, (outs FR32:$dst), (ins FR32:$src),
                  "movss\t{$src, $dst|$dst, $src}", []>;
let canFoldAsLoad = 1, isReMaterializable = 1, mayHaveSideEffects = 1 in
def MOVSSrm : SSI<0x10, MRMSrcMem, (outs FR32:$dst), (ins f32mem:$src),
                  "movss\t{$src, $dst|$dst, $src}",
                  [(set FR32:$dst, (loadf32 addr:$src))]>;
def MOVSSmr : SSI<0x11, MRMDestMem, (outs), (ins f32mem:$dst, FR32:$src),
                  "movss\t{$src, $dst|$dst, $src}",
                  [(store FR32:$src, addr:$dst)]>;

// Conversion instructions
def CVTTSS2SIrr : SSI<0x2C, MRMSrcReg, (outs GR32:$dst), (ins FR32:$src),
                      "cvttss2si\t{$src, $dst|$dst, $src}",
                      [(set GR32:$dst, (fp_to_sint FR32:$src))]>;
def CVTTSS2SIrm : SSI<0x2C, MRMSrcMem, (outs GR32:$dst), (ins f32mem:$src),
                      "cvttss2si\t{$src, $dst|$dst, $src}",
                      [(set GR32:$dst, (fp_to_sint (loadf32 addr:$src)))]>;
def CVTSI2SSrr  : SSI<0x2A, MRMSrcReg, (outs FR32:$dst), (ins GR32:$src),
                      "cvtsi2ss\t{$src, $dst|$dst, $src}",
                      [(set FR32:$dst, (sint_to_fp GR32:$src))]>;
def CVTSI2SSrm  : SSI<0x2A, MRMSrcMem, (outs FR32:$dst), (ins i32mem:$src),
                      "cvtsi2ss\t{$src, $dst|$dst, $src}",
                      [(set FR32:$dst, (sint_to_fp (loadi32 addr:$src)))]>;

// Match intrinsics which expect XMM operand(s).
def Int_CVTSS2SIrr : SSI<0x2D, MRMSrcReg, (outs GR32:$dst), (ins VR128:$src),
                         "cvtss2si\t{$src, $dst|$dst, $src}",
                         [(set GR32:$dst, (int_x86_sse_cvtss2si VR128:$src))]>;
def Int_CVTSS2SIrm : SSI<0x2D, MRMSrcMem, (outs GR32:$dst), (ins f32mem:$src),
                         "cvtss2si\t{$src, $dst|$dst, $src}",
                         [(set GR32:$dst, (int_x86_sse_cvtss2si
                                           (load addr:$src)))]>;

// Match intrinisics which expect MM and XMM operand(s).
def Int_CVTPS2PIrr : PSI<0x2D, MRMSrcReg, (outs VR64:$dst), (ins VR128:$src),
                         "cvtps2pi\t{$src, $dst|$dst, $src}",
                         [(set VR64:$dst, (int_x86_sse_cvtps2pi VR128:$src))]>;
def Int_CVTPS2PIrm : PSI<0x2D, MRMSrcMem, (outs VR64:$dst), (ins f64mem:$src),
                         "cvtps2pi\t{$src, $dst|$dst, $src}",
                         [(set VR64:$dst, (int_x86_sse_cvtps2pi 
                                           (load addr:$src)))]>;
def Int_CVTTPS2PIrr: PSI<0x2C, MRMSrcReg, (outs VR64:$dst), (ins VR128:$src),
                         "cvttps2pi\t{$src, $dst|$dst, $src}",
                         [(set VR64:$dst, (int_x86_sse_cvttps2pi VR128:$src))]>;
def Int_CVTTPS2PIrm: PSI<0x2C, MRMSrcMem, (outs VR64:$dst), (ins f64mem:$src),
                         "cvttps2pi\t{$src, $dst|$dst, $src}",
                         [(set VR64:$dst, (int_x86_sse_cvttps2pi 
                                           (load addr:$src)))]>;
let Constraints = "$src1 = $dst" in {
  def Int_CVTPI2PSrr : PSI<0x2A, MRMSrcReg, 
                           (outs VR128:$dst), (ins VR128:$src1, VR64:$src2),
                        "cvtpi2ps\t{$src2, $dst|$dst, $src2}",
                        [(set VR128:$dst, (int_x86_sse_cvtpi2ps VR128:$src1,
                                           VR64:$src2))]>;
  def Int_CVTPI2PSrm : PSI<0x2A, MRMSrcMem, 
                           (outs VR128:$dst), (ins VR128:$src1, i64mem:$src2),
                        "cvtpi2ps\t{$src2, $dst|$dst, $src2}",
                        [(set VR128:$dst, (int_x86_sse_cvtpi2ps VR128:$src1, 
                                            (load addr:$src2)))]>;
}

// Aliases for intrinsics
def Int_CVTTSS2SIrr : SSI<0x2C, MRMSrcReg, (outs GR32:$dst), (ins VR128:$src),
                          "cvttss2si\t{$src, $dst|$dst, $src}",
                          [(set GR32:$dst,
                            (int_x86_sse_cvttss2si VR128:$src))]>;
def Int_CVTTSS2SIrm : SSI<0x2C, MRMSrcMem, (outs GR32:$dst), (ins f32mem:$src),
                          "cvttss2si\t{$src, $dst|$dst, $src}",
                          [(set GR32:$dst,
                            (int_x86_sse_cvttss2si(load addr:$src)))]>;

let Constraints = "$src1 = $dst" in {
  def Int_CVTSI2SSrr : SSI<0x2A, MRMSrcReg,
                           (outs VR128:$dst), (ins VR128:$src1, GR32:$src2),
                           "cvtsi2ss\t{$src2, $dst|$dst, $src2}",
                           [(set VR128:$dst, (int_x86_sse_cvtsi2ss VR128:$src1,
                                              GR32:$src2))]>;
  def Int_CVTSI2SSrm : SSI<0x2A, MRMSrcMem,
                           (outs VR128:$dst), (ins VR128:$src1, i32mem:$src2),
                           "cvtsi2ss\t{$src2, $dst|$dst, $src2}",
                           [(set VR128:$dst, (int_x86_sse_cvtsi2ss VR128:$src1,
                                              (loadi32 addr:$src2)))]>;
}

// Comparison instructions
let Constraints = "$src1 = $dst", neverHasSideEffects = 1 in {
  def CMPSSrr : SSIi8<0xC2, MRMSrcReg, 
                    (outs FR32:$dst), (ins FR32:$src1, FR32:$src, SSECC:$cc),
                    "cmp${cc}ss\t{$src, $dst|$dst, $src}", []>;
let mayLoad = 1 in
  def CMPSSrm : SSIi8<0xC2, MRMSrcMem, 
                    (outs FR32:$dst), (ins FR32:$src1, f32mem:$src, SSECC:$cc),
                    "cmp${cc}ss\t{$src, $dst|$dst, $src}", []>;
}

let Defs = [EFLAGS] in {
def UCOMISSrr: PSI<0x2E, MRMSrcReg, (outs), (ins FR32:$src1, FR32:$src2),
                   "ucomiss\t{$src2, $src1|$src1, $src2}",
                   [(X86cmp FR32:$src1, FR32:$src2), (implicit EFLAGS)]>;
def UCOMISSrm: PSI<0x2E, MRMSrcMem, (outs), (ins FR32:$src1, f32mem:$src2),
                   "ucomiss\t{$src2, $src1|$src1, $src2}",
                   [(X86cmp FR32:$src1, (loadf32 addr:$src2)),
                    (implicit EFLAGS)]>;
} // Defs = [EFLAGS]

// Aliases to match intrinsics which expect XMM operand(s).
let Constraints = "$src1 = $dst" in {
  def Int_CMPSSrr : SSIi8<0xC2, MRMSrcReg, 
                        (outs VR128:$dst), (ins VR128:$src1, VR128:$src, SSECC:$cc),
                        "cmp${cc}ss\t{$src, $dst|$dst, $src}",
                        [(set VR128:$dst, (int_x86_sse_cmp_ss VR128:$src1,
                                           VR128:$src, imm:$cc))]>;
  def Int_CMPSSrm : SSIi8<0xC2, MRMSrcMem, 
                        (outs VR128:$dst), (ins VR128:$src1, f32mem:$src, SSECC:$cc),
                        "cmp${cc}ss\t{$src, $dst|$dst, $src}",
                        [(set VR128:$dst, (int_x86_sse_cmp_ss VR128:$src1,
                                           (load addr:$src), imm:$cc))]>;
}

let Defs = [EFLAGS] in {
def Int_UCOMISSrr: PSI<0x2E, MRMSrcReg, (outs), (ins VR128:$src1, VR128:$src2),
                       "ucomiss\t{$src2, $src1|$src1, $src2}",
                       [(X86ucomi (v4f32 VR128:$src1), VR128:$src2),
                        (implicit EFLAGS)]>;
def Int_UCOMISSrm: PSI<0x2E, MRMSrcMem, (outs),(ins VR128:$src1, f128mem:$src2),
                       "ucomiss\t{$src2, $src1|$src1, $src2}",
                       [(X86ucomi (v4f32 VR128:$src1), (load addr:$src2)),
                        (implicit EFLAGS)]>;

def Int_COMISSrr: PSI<0x2F, MRMSrcReg, (outs), (ins VR128:$src1, VR128:$src2),
                      "comiss\t{$src2, $src1|$src1, $src2}",
                      [(X86comi (v4f32 VR128:$src1), VR128:$src2),
                       (implicit EFLAGS)]>;
def Int_COMISSrm: PSI<0x2F, MRMSrcMem, (outs), (ins VR128:$src1, f128mem:$src2),
                      "comiss\t{$src2, $src1|$src1, $src2}",
                      [(X86comi (v4f32 VR128:$src1), (load addr:$src2)),
                       (implicit EFLAGS)]>;
} // Defs = [EFLAGS]

// Aliases of packed SSE1 instructions for scalar use. These all have names that
// start with 'Fs'.

// Alias instructions that map fld0 to pxor for sse.
let isReMaterializable = 1, isAsCheapAsAMove = 1 in
def FsFLD0SS : I<0xEF, MRMInitReg, (outs FR32:$dst), (ins),
                 "pxor\t$dst, $dst", [(set FR32:$dst, fp32imm0)]>,
               Requires<[HasSSE1]>, TB, OpSize;

// Alias instruction to do FR32 reg-to-reg copy using movaps. Upper bits are
// disregarded.
let neverHasSideEffects = 1 in 
def FsMOVAPSrr : PSI<0x28, MRMSrcReg, (outs FR32:$dst), (ins FR32:$src),
                     "movaps\t{$src, $dst|$dst, $src}", []>;

// Alias instruction to load FR32 from f128mem using movaps. Upper bits are
// disregarded.
let canFoldAsLoad = 1 in
def FsMOVAPSrm : PSI<0x28, MRMSrcMem, (outs FR32:$dst), (ins f128mem:$src),
                     "movaps\t{$src, $dst|$dst, $src}",
                     [(set FR32:$dst, (alignedloadfsf32 addr:$src))]>;

// Alias bitwise logical operations using SSE logical ops on packed FP values.
let Constraints = "$src1 = $dst" in {
let isCommutable = 1 in {
  def FsANDPSrr : PSI<0x54, MRMSrcReg, (outs FR32:$dst),
                                       (ins FR32:$src1, FR32:$src2),
                      "andps\t{$src2, $dst|$dst, $src2}",
                      [(set FR32:$dst, (X86fand FR32:$src1, FR32:$src2))]>;
  def FsORPSrr  : PSI<0x56, MRMSrcReg, (outs FR32:$dst),
                                       (ins FR32:$src1, FR32:$src2),
                      "orps\t{$src2, $dst|$dst, $src2}",
                      [(set FR32:$dst, (X86for FR32:$src1, FR32:$src2))]>;
  def FsXORPSrr : PSI<0x57, MRMSrcReg, (outs FR32:$dst),
                                       (ins FR32:$src1, FR32:$src2),
                      "xorps\t{$src2, $dst|$dst, $src2}",
                      [(set FR32:$dst, (X86fxor FR32:$src1, FR32:$src2))]>;
}

def FsANDPSrm : PSI<0x54, MRMSrcMem, (outs FR32:$dst),
                                     (ins FR32:$src1, f128mem:$src2),
                    "andps\t{$src2, $dst|$dst, $src2}",
                    [(set FR32:$dst, (X86fand FR32:$src1,
                                      (memopfsf32 addr:$src2)))]>;
def FsORPSrm  : PSI<0x56, MRMSrcMem, (outs FR32:$dst),
                                     (ins FR32:$src1, f128mem:$src2),
                    "orps\t{$src2, $dst|$dst, $src2}",
                    [(set FR32:$dst, (X86for FR32:$src1,
                                      (memopfsf32 addr:$src2)))]>;
def FsXORPSrm : PSI<0x57, MRMSrcMem, (outs FR32:$dst),
                                     (ins FR32:$src1, f128mem:$src2),
                    "xorps\t{$src2, $dst|$dst, $src2}",
                    [(set FR32:$dst, (X86fxor FR32:$src1,
                                      (memopfsf32 addr:$src2)))]>;

let neverHasSideEffects = 1 in {
def FsANDNPSrr : PSI<0x55, MRMSrcReg,
                     (outs FR32:$dst), (ins FR32:$src1, FR32:$src2),
                     "andnps\t{$src2, $dst|$dst, $src2}", []>;
let mayLoad = 1 in
def FsANDNPSrm : PSI<0x55, MRMSrcMem,
                     (outs FR32:$dst), (ins FR32:$src1, f128mem:$src2),
                     "andnps\t{$src2, $dst|$dst, $src2}", []>;
}
}

/// basic_sse1_fp_binop_rm - SSE1 binops come in both scalar and vector forms.
///
/// In addition, we also have a special variant of the scalar form here to
/// represent the associated intrinsic operation.  This form is unlike the
/// plain scalar form, in that it takes an entire vector (instead of a scalar)
/// and leaves the top elements undefined.
///
/// These three forms can each be reg+reg or reg+mem, so there are a total of
/// six "instructions".
///
let Constraints = "$src1 = $dst" in {
multiclass basic_sse1_fp_binop_rm<bits<8> opc, string OpcodeStr,
                                  SDNode OpNode, Intrinsic F32Int,
                                  bit Commutable = 0> {
  // Scalar operation, reg+reg.
  def SSrr : SSI<opc, MRMSrcReg, (outs FR32:$dst), (ins FR32:$src1, FR32:$src2),
                 !strconcat(OpcodeStr, "ss\t{$src2, $dst|$dst, $src2}"),
                 [(set FR32:$dst, (OpNode FR32:$src1, FR32:$src2))]> {
    let isCommutable = Commutable;
  }

  // Scalar operation, reg+mem.
  def SSrm : SSI<opc, MRMSrcMem, (outs FR32:$dst),
                                 (ins FR32:$src1, f32mem:$src2),
                 !strconcat(OpcodeStr, "ss\t{$src2, $dst|$dst, $src2}"),
                 [(set FR32:$dst, (OpNode FR32:$src1, (load addr:$src2)))]>;
                 
  // Vector operation, reg+reg.
  def PSrr : PSI<opc, MRMSrcReg, (outs VR128:$dst),
                                 (ins VR128:$src1, VR128:$src2),
               !strconcat(OpcodeStr, "ps\t{$src2, $dst|$dst, $src2}"),
               [(set VR128:$dst, (v4f32 (OpNode VR128:$src1, VR128:$src2)))]> {
    let isCommutable = Commutable;
  }

  // Vector operation, reg+mem.
  def PSrm : PSI<opc, MRMSrcMem, (outs VR128:$dst),
                                 (ins VR128:$src1, f128mem:$src2),
                 !strconcat(OpcodeStr, "ps\t{$src2, $dst|$dst, $src2}"),
             [(set VR128:$dst, (OpNode VR128:$src1, (memopv4f32 addr:$src2)))]>;

  // Intrinsic operation, reg+reg.
  def SSrr_Int : SSI<opc, MRMSrcReg, (outs VR128:$dst),
                                     (ins VR128:$src1, VR128:$src2),
                     !strconcat(OpcodeStr, "ss\t{$src2, $dst|$dst, $src2}"),
                     [(set VR128:$dst, (F32Int VR128:$src1, VR128:$src2))]> {
    let isCommutable = Commutable;
  }

  // Intrinsic operation, reg+mem.
  def SSrm_Int : SSI<opc, MRMSrcMem, (outs VR128:$dst),
                                     (ins VR128:$src1, ssmem:$src2),
                     !strconcat(OpcodeStr, "ss\t{$src2, $dst|$dst, $src2}"),
                     [(set VR128:$dst, (F32Int VR128:$src1,
                                               sse_load_f32:$src2))]>;
}
}

// Arithmetic instructions
defm ADD : basic_sse1_fp_binop_rm<0x58, "add", fadd, int_x86_sse_add_ss, 1>;
defm MUL : basic_sse1_fp_binop_rm<0x59, "mul", fmul, int_x86_sse_mul_ss, 1>;
defm SUB : basic_sse1_fp_binop_rm<0x5C, "sub", fsub, int_x86_sse_sub_ss>;
defm DIV : basic_sse1_fp_binop_rm<0x5E, "div", fdiv, int_x86_sse_div_ss>;

/// sse1_fp_binop_rm - Other SSE1 binops
///
/// This multiclass is like basic_sse1_fp_binop_rm, with the addition of
/// instructions for a full-vector intrinsic form.  Operations that map
/// onto C operators don't use this form since they just use the plain
/// vector form instead of having a separate vector intrinsic form.
///
/// This provides a total of eight "instructions".
///
let Constraints = "$src1 = $dst" in {
multiclass sse1_fp_binop_rm<bits<8> opc, string OpcodeStr,
                            SDNode OpNode,
                            Intrinsic F32Int,
                            Intrinsic V4F32Int,
                            bit Commutable = 0> {

  // Scalar operation, reg+reg.
  def SSrr : SSI<opc, MRMSrcReg, (outs FR32:$dst), (ins FR32:$src1, FR32:$src2),
                 !strconcat(OpcodeStr, "ss\t{$src2, $dst|$dst, $src2}"),
                 [(set FR32:$dst, (OpNode FR32:$src1, FR32:$src2))]> {
    let isCommutable = Commutable;
  }

  // Scalar operation, reg+mem.
  def SSrm : SSI<opc, MRMSrcMem, (outs FR32:$dst),
                                 (ins FR32:$src1, f32mem:$src2),
                 !strconcat(OpcodeStr, "ss\t{$src2, $dst|$dst, $src2}"),
                 [(set FR32:$dst, (OpNode FR32:$src1, (load addr:$src2)))]>;
                 
  // Vector operation, reg+reg.
  def PSrr : PSI<opc, MRMSrcReg, (outs VR128:$dst),
                                 (ins VR128:$src1, VR128:$src2),
               !strconcat(OpcodeStr, "ps\t{$src2, $dst|$dst, $src2}"),
               [(set VR128:$dst, (v4f32 (OpNode VR128:$src1, VR128:$src2)))]> {
    let isCommutable = Commutable;
  }

  // Vector operation, reg+mem.
  def PSrm : PSI<opc, MRMSrcMem, (outs VR128:$dst),
                                 (ins VR128:$src1, f128mem:$src2),
                 !strconcat(OpcodeStr, "ps\t{$src2, $dst|$dst, $src2}"),
             [(set VR128:$dst, (OpNode VR128:$src1, (memopv4f32 addr:$src2)))]>;

  // Intrinsic operation, reg+reg.
  def SSrr_Int : SSI<opc, MRMSrcReg, (outs VR128:$dst),
                                     (ins VR128:$src1, VR128:$src2),
                     !strconcat(OpcodeStr, "ss\t{$src2, $dst|$dst, $src2}"),
                     [(set VR128:$dst, (F32Int VR128:$src1, VR128:$src2))]> {
    let isCommutable = Commutable;
  }

  // Intrinsic operation, reg+mem.
  def SSrm_Int : SSI<opc, MRMSrcMem, (outs VR128:$dst),
                                     (ins VR128:$src1, ssmem:$src2),
                     !strconcat(OpcodeStr, "ss\t{$src2, $dst|$dst, $src2}"),
                     [(set VR128:$dst, (F32Int VR128:$src1,
                                               sse_load_f32:$src2))]>;

  // Vector intrinsic operation, reg+reg.
  def PSrr_Int : PSI<opc, MRMSrcReg, (outs VR128:$dst),
                                     (ins VR128:$src1, VR128:$src2),
                     !strconcat(OpcodeStr, "ps\t{$src2, $dst|$dst, $src2}"),
                     [(set VR128:$dst, (V4F32Int VR128:$src1, VR128:$src2))]> {
    let isCommutable = Commutable;
  }

  // Vector intrinsic operation, reg+mem.
  def PSrm_Int : PSI<opc, MRMSrcMem, (outs VR128:$dst),
                                     (ins VR128:$src1, f128mem:$src2),
                     !strconcat(OpcodeStr, "ps\t{$src2, $dst|$dst, $src2}"),
           [(set VR128:$dst, (V4F32Int VR128:$src1, (memopv4f32 addr:$src2)))]>;
}
}

defm MAX : sse1_fp_binop_rm<0x5F, "max", X86fmax,
                            int_x86_sse_max_ss, int_x86_sse_max_ps>;
defm MIN : sse1_fp_binop_rm<0x5D, "min", X86fmin,
                            int_x86_sse_min_ss, int_x86_sse_min_ps>;

//===----------------------------------------------------------------------===//
// SSE packed FP Instructions

// Move Instructions
let neverHasSideEffects = 1 in 
def MOVAPSrr : PSI<0x28, MRMSrcReg, (outs VR128:$dst), (ins VR128:$src),
                   "movaps\t{$src, $dst|$dst, $src}", []>;
let canFoldAsLoad = 1, isReMaterializable = 1, mayHaveSideEffects = 1 in
def MOVAPSrm : PSI<0x28, MRMSrcMem, (outs VR128:$dst), (ins f128mem:$src),
                   "movaps\t{$src, $dst|$dst, $src}",
                   [(set VR128:$dst, (alignedloadv4f32 addr:$src))]>;

def MOVAPSmr : PSI<0x29, MRMDestMem, (outs), (ins f128mem:$dst, VR128:$src),
                   "movaps\t{$src, $dst|$dst, $src}",
                   [(alignedstore (v4f32 VR128:$src), addr:$dst)]>;

let neverHasSideEffects = 1 in
def MOVUPSrr : PSI<0x10, MRMSrcReg, (outs VR128:$dst), (ins VR128:$src),
                   "movups\t{$src, $dst|$dst, $src}", []>;
let canFoldAsLoad = 1 in
def MOVUPSrm : PSI<0x10, MRMSrcMem, (outs VR128:$dst), (ins f128mem:$src),
                   "movups\t{$src, $dst|$dst, $src}",
                   [(set VR128:$dst, (loadv4f32 addr:$src))]>;
def MOVUPSmr : PSI<0x11, MRMDestMem, (outs), (ins f128mem:$dst, VR128:$src),
                   "movups\t{$src, $dst|$dst, $src}",
                   [(store (v4f32 VR128:$src), addr:$dst)]>;

// Intrinsic forms of MOVUPS load and store
let canFoldAsLoad = 1 in
def MOVUPSrm_Int : PSI<0x10, MRMSrcMem, (outs VR128:$dst), (ins f128mem:$src),
                       "movups\t{$src, $dst|$dst, $src}",
                       [(set VR128:$dst, (int_x86_sse_loadu_ps addr:$src))]>;
def MOVUPSmr_Int : PSI<0x11, MRMDestMem, (outs), (ins f128mem:$dst, VR128:$src),
                       "movups\t{$src, $dst|$dst, $src}",
                       [(int_x86_sse_storeu_ps addr:$dst, VR128:$src)]>;

let Constraints = "$src1 = $dst" in {
  let AddedComplexity = 20 in {
    def MOVLPSrm : PSI<0x12, MRMSrcMem,
                       (outs VR128:$dst), (ins VR128:$src1, f64mem:$src2),
                       "movlps\t{$src2, $dst|$dst, $src2}",
       [(set VR128:$dst, 
             (v4f32 (vector_shuffle VR128:$src1,
                     (bc_v4f32 (v2f64 (scalar_to_vector (loadf64 addr:$src2)))),
                     MOVLP_shuffle_mask)))]>;
    def MOVHPSrm : PSI<0x16, MRMSrcMem,
                       (outs VR128:$dst), (ins VR128:$src1, f64mem:$src2),
                       "movhps\t{$src2, $dst|$dst, $src2}",
       [(set VR128:$dst, 
             (v4f32 (vector_shuffle VR128:$src1,
                     (bc_v4f32 (v2f64 (scalar_to_vector (loadf64 addr:$src2)))),
                     MOVHP_shuffle_mask)))]>;
  } // AddedComplexity
} // Constraints = "$src1 = $dst"


def MOVLPSmr : PSI<0x13, MRMDestMem, (outs), (ins f64mem:$dst, VR128:$src),
                   "movlps\t{$src, $dst|$dst, $src}",
                   [(store (f64 (vector_extract (bc_v2f64 (v4f32 VR128:$src)),
                                 (iPTR 0))), addr:$dst)]>;

// v2f64 extract element 1 is always custom lowered to unpack high to low
// and extract element 0 so the non-store version isn't too horrible.
def MOVHPSmr : PSI<0x17, MRMDestMem, (outs), (ins f64mem:$dst, VR128:$src),
                   "movhps\t{$src, $dst|$dst, $src}",
                   [(store (f64 (vector_extract
                                 (v2f64 (vector_shuffle
                                         (bc_v2f64 (v4f32 VR128:$src)), (undef),
                                         UNPCKH_shuffle_mask)), (iPTR 0))),
                     addr:$dst)]>;

let Constraints = "$src1 = $dst" in {
let AddedComplexity = 20 in {
def MOVLHPSrr : PSI<0x16, MRMSrcReg, (outs VR128:$dst), (ins VR128:$src1, VR128:$src2),
                    "movlhps\t{$src2, $dst|$dst, $src2}",
                    [(set VR128:$dst,
                      (v4f32 (vector_shuffle VR128:$src1, VR128:$src2,
                              MOVHP_shuffle_mask)))]>;

def MOVHLPSrr : PSI<0x12, MRMSrcReg, (outs VR128:$dst), (ins VR128:$src1, VR128:$src2),
                    "movhlps\t{$src2, $dst|$dst, $src2}",
                    [(set VR128:$dst,
                      (v4f32 (vector_shuffle VR128:$src1, VR128:$src2,
                              MOVHLPS_shuffle_mask)))]>;
} // AddedComplexity
} // Constraints = "$src1 = $dst"

let AddedComplexity = 20 in
def : Pat<(v4f32 (vector_shuffle VR128:$src, (undef), MOVDDUP_shuffle_mask)),
          (MOVLHPSrr VR128:$src, VR128:$src)>, Requires<[HasSSE1]>;




// Arithmetic

/// sse1_fp_unop_rm - SSE1 unops come in both scalar and vector forms.
///
/// In addition, we also have a special variant of the scalar form here to
/// represent the associated intrinsic operation.  This form is unlike the
/// plain scalar form, in that it takes an entire vector (instead of a
/// scalar) and leaves the top elements undefined.
///
/// And, we have a special variant form for a full-vector intrinsic form.
///
/// These four forms can each have a reg or a mem operand, so there are a
/// total of eight "instructions".
///
multiclass sse1_fp_unop_rm<bits<8> opc, string OpcodeStr,
                           SDNode OpNode,
                           Intrinsic F32Int,
                           Intrinsic V4F32Int,
                           bit Commutable = 0> {
  // Scalar operation, reg.
  def SSr : SSI<opc, MRMSrcReg, (outs FR32:$dst), (ins FR32:$src),
                !strconcat(OpcodeStr, "ss\t{$src, $dst|$dst, $src}"),
                [(set FR32:$dst, (OpNode FR32:$src))]> {
    let isCommutable = Commutable;
  }

  // Scalar operation, mem.
  def SSm : SSI<opc, MRMSrcMem, (outs FR32:$dst), (ins f32mem:$src),
                !strconcat(OpcodeStr, "ss\t{$src, $dst|$dst, $src}"),
                [(set FR32:$dst, (OpNode (load addr:$src)))]>;
                 
  // Vector operation, reg.
  def PSr : PSI<opc, MRMSrcReg, (outs VR128:$dst), (ins VR128:$src),
              !strconcat(OpcodeStr, "ps\t{$src, $dst|$dst, $src}"),
              [(set VR128:$dst, (v4f32 (OpNode VR128:$src)))]> {
    let isCommutable = Commutable;
  }

  // Vector operation, mem.
  def PSm : PSI<opc, MRMSrcMem, (outs VR128:$dst), (ins f128mem:$src),
                !strconcat(OpcodeStr, "ps\t{$src, $dst|$dst, $src}"),
                [(set VR128:$dst, (OpNode (memopv4f32 addr:$src)))]>;

  // Intrinsic operation, reg.
  def SSr_Int : SSI<opc, MRMSrcReg, (outs VR128:$dst), (ins VR128:$src),
                    !strconcat(OpcodeStr, "ss\t{$src, $dst|$dst, $src}"),
                    [(set VR128:$dst, (F32Int VR128:$src))]> {
    let isCommutable = Commutable;
  }

  // Intrinsic operation, mem.
  def SSm_Int : SSI<opc, MRMSrcMem, (outs VR128:$dst), (ins ssmem:$src),
                    !strconcat(OpcodeStr, "ss\t{$src, $dst|$dst, $src}"),
                    [(set VR128:$dst, (F32Int sse_load_f32:$src))]>;

  // Vector intrinsic operation, reg
  def PSr_Int : PSI<opc, MRMSrcReg, (outs VR128:$dst), (ins VR128:$src),
                    !strconcat(OpcodeStr, "ps\t{$src, $dst|$dst, $src}"),
                    [(set VR128:$dst, (V4F32Int VR128:$src))]> {
    let isCommutable = Commutable;
  }

  // Vector intrinsic operation, mem
  def PSm_Int : PSI<opc, MRMSrcMem, (outs VR128:$dst), (ins f128mem:$src),
                    !strconcat(OpcodeStr, "ps\t{$src, $dst|$dst, $src}"),
                    [(set VR128:$dst, (V4F32Int (memopv4f32 addr:$src)))]>;
}

// Square root.
defm SQRT  : sse1_fp_unop_rm<0x51, "sqrt",  fsqrt,
                             int_x86_sse_sqrt_ss, int_x86_sse_sqrt_ps>;

// Reciprocal approximations. Note that these typically require refinement
// in order to obtain suitable precision.
defm RSQRT : sse1_fp_unop_rm<0x52, "rsqrt", X86frsqrt,
                             int_x86_sse_rsqrt_ss, int_x86_sse_rsqrt_ps>;
defm RCP   : sse1_fp_unop_rm<0x53, "rcp",   X86frcp,
                             int_x86_sse_rcp_ss, int_x86_sse_rcp_ps>;

// Logical
let Constraints = "$src1 = $dst" in {
  let isCommutable = 1 in {
    def ANDPSrr : PSI<0x54, MRMSrcReg,
                      (outs VR128:$dst), (ins VR128:$src1, VR128:$src2),
                      "andps\t{$src2, $dst|$dst, $src2}",
                      [(set VR128:$dst, (v2i64
                                         (and VR128:$src1, VR128:$src2)))]>;
    def ORPSrr  : PSI<0x56, MRMSrcReg,
                      (outs VR128:$dst), (ins VR128:$src1, VR128:$src2),
                      "orps\t{$src2, $dst|$dst, $src2}",
                      [(set VR128:$dst, (v2i64
                                         (or VR128:$src1, VR128:$src2)))]>;
    def XORPSrr : PSI<0x57, MRMSrcReg,
                      (outs VR128:$dst), (ins VR128:$src1, VR128:$src2),
                      "xorps\t{$src2, $dst|$dst, $src2}",
                      [(set VR128:$dst, (v2i64
                                         (xor VR128:$src1, VR128:$src2)))]>;
  }

  def ANDPSrm : PSI<0x54, MRMSrcMem,
                    (outs VR128:$dst), (ins VR128:$src1, f128mem:$src2),
                    "andps\t{$src2, $dst|$dst, $src2}",
                    [(set VR128:$dst, (and (bc_v2i64 (v4f32 VR128:$src1)),
                                       (memopv2i64 addr:$src2)))]>;
  def ORPSrm  : PSI<0x56, MRMSrcMem,
                    (outs VR128:$dst), (ins VR128:$src1, f128mem:$src2),
                    "orps\t{$src2, $dst|$dst, $src2}",
                    [(set VR128:$dst, (or (bc_v2i64 (v4f32 VR128:$src1)),
                                       (memopv2i64 addr:$src2)))]>;
  def XORPSrm : PSI<0x57, MRMSrcMem,
                    (outs VR128:$dst), (ins VR128:$src1, f128mem:$src2),
                    "xorps\t{$src2, $dst|$dst, $src2}",
                    [(set VR128:$dst, (xor (bc_v2i64 (v4f32 VR128:$src1)),
                                       (memopv2i64 addr:$src2)))]>;
  def ANDNPSrr : PSI<0x55, MRMSrcReg,
                     (outs VR128:$dst), (ins VR128:$src1, VR128:$src2),
                     "andnps\t{$src2, $dst|$dst, $src2}",
                     [(set VR128:$dst,
                       (v2i64 (and (xor VR128:$src1,
                                    (bc_v2i64 (v4i32 immAllOnesV))),
                               VR128:$src2)))]>;
  def ANDNPSrm : PSI<0x55, MRMSrcMem,
                     (outs VR128:$dst), (ins VR128:$src1,f128mem:$src2),
                     "andnps\t{$src2, $dst|$dst, $src2}",
                     [(set VR128:$dst,
                       (v2i64 (and (xor (bc_v2i64 (v4f32 VR128:$src1)),
                                    (bc_v2i64 (v4i32 immAllOnesV))),
                               (memopv2i64 addr:$src2))))]>;
}

let Constraints = "$src1 = $dst" in {
  def CMPPSrri : PSIi8<0xC2, MRMSrcReg, 
                    (outs VR128:$dst), (ins VR128:$src1, VR128:$src, SSECC:$cc),
                    "cmp${cc}ps\t{$src, $dst|$dst, $src}",
                    [(set VR128:$dst, (int_x86_sse_cmp_ps VR128:$src1,
                                                        VR128:$src, imm:$cc))]>;
  def CMPPSrmi : PSIi8<0xC2, MRMSrcMem, 
                  (outs VR128:$dst), (ins VR128:$src1, f128mem:$src, SSECC:$cc),
                  "cmp${cc}ps\t{$src, $dst|$dst, $src}",
                  [(set VR128:$dst, (int_x86_sse_cmp_ps VR128:$src1,
                                            (memop addr:$src), imm:$cc))]>;
}
def : Pat<(v4i32 (X86cmpps (v4f32 VR128:$src1), VR128:$src2, imm:$cc)),
          (CMPPSrri VR128:$src1, VR128:$src2, imm:$cc)>;
def : Pat<(v4i32 (X86cmpps (v4f32 VR128:$src1), (memop addr:$src2), imm:$cc)),
          (CMPPSrmi VR128:$src1, addr:$src2, imm:$cc)>;

// Shuffle and unpack instructions
let Constraints = "$src1 = $dst" in {
  let isConvertibleToThreeAddress = 1 in // Convert to pshufd
    def SHUFPSrri : PSIi8<0xC6, MRMSrcReg, 
                          (outs VR128:$dst), (ins VR128:$src1,
                           VR128:$src2, i32i8imm:$src3),
                          "shufps\t{$src3, $src2, $dst|$dst, $src2, $src3}",
                          [(set VR128:$dst,
                            (v4f32 (vector_shuffle
                                    VR128:$src1, VR128:$src2,
                                    SHUFP_shuffle_mask:$src3)))]>;
  def SHUFPSrmi : PSIi8<0xC6, MRMSrcMem, 
                        (outs VR128:$dst), (ins VR128:$src1,
                         f128mem:$src2, i32i8imm:$src3),
                        "shufps\t{$src3, $src2, $dst|$dst, $src2, $src3}",
                        [(set VR128:$dst,
                          (v4f32 (vector_shuffle
                                  VR128:$src1, (memopv4f32 addr:$src2),
                                  SHUFP_shuffle_mask:$src3)))]>;

  let AddedComplexity = 10 in {
    def UNPCKHPSrr : PSI<0x15, MRMSrcReg, 
                         (outs VR128:$dst), (ins VR128:$src1, VR128:$src2),
                         "unpckhps\t{$src2, $dst|$dst, $src2}",
                         [(set VR128:$dst,
                           (v4f32 (vector_shuffle
                                   VR128:$src1, VR128:$src2,
                                   UNPCKH_shuffle_mask)))]>;
    def UNPCKHPSrm : PSI<0x15, MRMSrcMem, 
                         (outs VR128:$dst), (ins VR128:$src1, f128mem:$src2),
                         "unpckhps\t{$src2, $dst|$dst, $src2}",
                         [(set VR128:$dst,
                           (v4f32 (vector_shuffle
                                   VR128:$src1, (memopv4f32 addr:$src2),
                                   UNPCKH_shuffle_mask)))]>;

    def UNPCKLPSrr : PSI<0x14, MRMSrcReg, 
                         (outs VR128:$dst), (ins VR128:$src1, VR128:$src2),
                         "unpcklps\t{$src2, $dst|$dst, $src2}",
                         [(set VR128:$dst,
                           (v4f32 (vector_shuffle
                                   VR128:$src1, VR128:$src2,
                                   UNPCKL_shuffle_mask)))]>;
    def UNPCKLPSrm : PSI<0x14, MRMSrcMem, 
                         (outs VR128:$dst), (ins VR128:$src1, f128mem:$src2),
                         "unpcklps\t{$src2, $dst|$dst, $src2}",
                         [(set VR128:$dst,
                           (v4f32 (vector_shuffle
                                   VR128:$src1, (memopv4f32 addr:$src2),
                                   UNPCKL_shuffle_mask)))]>;
  } // AddedComplexity
} // Constraints = "$src1 = $dst"

// Mask creation
def MOVMSKPSrr : PSI<0x50, MRMSrcReg, (outs GR32:$dst), (ins VR128:$src),
                     "movmskps\t{$src, $dst|$dst, $src}",
                     [(set GR32:$dst, (int_x86_sse_movmsk_ps VR128:$src))]>;
def MOVMSKPDrr : PSI<0x50, MRMSrcReg, (outs GR32:$dst), (ins VR128:$src),
                     "movmskpd\t{$src, $dst|$dst, $src}",
                     [(set GR32:$dst, (int_x86_sse2_movmsk_pd VR128:$src))]>;

// Prefetch intrinsic.
def PREFETCHT0   : PSI<0x18, MRM1m, (outs), (ins i8mem:$src),
    "prefetcht0\t$src", [(prefetch addr:$src, imm, (i32 3))]>;
def PREFETCHT1   : PSI<0x18, MRM2m, (outs), (ins i8mem:$src),
    "prefetcht1\t$src", [(prefetch addr:$src, imm, (i32 2))]>;
def PREFETCHT2   : PSI<0x18, MRM3m, (outs), (ins i8mem:$src),
    "prefetcht2\t$src", [(prefetch addr:$src, imm, (i32 1))]>;
def PREFETCHNTA  : PSI<0x18, MRM0m, (outs), (ins i8mem:$src),
    "prefetchnta\t$src", [(prefetch addr:$src, imm, (i32 0))]>;

// Non-temporal stores
def MOVNTPSmr : PSI<0x2B, MRMDestMem, (outs), (ins i128mem:$dst, VR128:$src),
                    "movntps\t{$src, $dst|$dst, $src}",
                    [(int_x86_sse_movnt_ps addr:$dst, VR128:$src)]>;

// Load, store, and memory fence
def SFENCE : PSI<0xAE, MRM7m, (outs), (ins), "sfence", [(int_x86_sse_sfence)]>;

// MXCSR register
def LDMXCSR : PSI<0xAE, MRM2m, (outs), (ins i32mem:$src),
                  "ldmxcsr\t$src", [(int_x86_sse_ldmxcsr addr:$src)]>;
def STMXCSR : PSI<0xAE, MRM3m, (outs), (ins i32mem:$dst),
                  "stmxcsr\t$dst", [(int_x86_sse_stmxcsr addr:$dst)]>;

// Alias instructions that map zero vector to pxor / xorp* for sse.
// We set canFoldAsLoad because this can be converted to a constant-pool
// load of an all-zeros value if folding it would be beneficial.
let isReMaterializable = 1, isAsCheapAsAMove = 1, canFoldAsLoad = 1 in
def V_SET0 : PSI<0x57, MRMInitReg, (outs VR128:$dst), (ins),
                 "xorps\t$dst, $dst",
                 [(set VR128:$dst, (v4i32 immAllZerosV))]>;

let Predicates = [HasSSE1] in {
  def : Pat<(v2i64 immAllZerosV), (V_SET0)>;
  def : Pat<(v8i16 immAllZerosV), (V_SET0)>;
  def : Pat<(v16i8 immAllZerosV), (V_SET0)>;
  def : Pat<(v2f64 immAllZerosV), (V_SET0)>;
  def : Pat<(v4f32 immAllZerosV), (V_SET0)>;
}

// FR32 to 128-bit vector conversion.
def MOVSS2PSrr : SSI<0x10, MRMSrcReg, (outs VR128:$dst), (ins FR32:$src),
                      "movss\t{$src, $dst|$dst, $src}",
                      [(set VR128:$dst,
                        (v4f32 (scalar_to_vector FR32:$src)))]>;
def MOVSS2PSrm : SSI<0x10, MRMSrcMem, (outs VR128:$dst), (ins f32mem:$src),
                     "movss\t{$src, $dst|$dst, $src}",
                     [(set VR128:$dst,
                       (v4f32 (scalar_to_vector (loadf32 addr:$src))))]>;

// FIXME: may not be able to eliminate this movss with coalescing the src and
// dest register classes are different. We really want to write this pattern
// like this:
// def : Pat<(f32 (vector_extract (v4f32 VR128:$src), (iPTR 0))),
//           (f32 FR32:$src)>;
def MOVPS2SSrr : SSI<0x10, MRMSrcReg, (outs FR32:$dst), (ins VR128:$src),
                     "movss\t{$src, $dst|$dst, $src}",
                     [(set FR32:$dst, (vector_extract (v4f32 VR128:$src),
                                       (iPTR 0)))]>;
def MOVPS2SSmr : SSI<0x11, MRMDestMem, (outs), (ins f32mem:$dst, VR128:$src),
                     "movss\t{$src, $dst|$dst, $src}",
                     [(store (f32 (vector_extract (v4f32 VR128:$src),
                                   (iPTR 0))), addr:$dst)]>;


// Move to lower bits of a VR128, leaving upper bits alone.
// Three operand (but two address) aliases.
let Constraints = "$src1 = $dst" in {
let neverHasSideEffects = 1 in
  def MOVLSS2PSrr : SSI<0x10, MRMSrcReg,
                        (outs VR128:$dst), (ins VR128:$src1, FR32:$src2),
                        "movss\t{$src2, $dst|$dst, $src2}", []>;

  let AddedComplexity = 15 in
    def MOVLPSrr : SSI<0x10, MRMSrcReg,
                       (outs VR128:$dst), (ins VR128:$src1, VR128:$src2),
                       "movss\t{$src2, $dst|$dst, $src2}",
                       [(set VR128:$dst,
                         (v4f32 (vector_shuffle VR128:$src1, VR128:$src2,
                                 MOVL_shuffle_mask)))]>;
}

// Move to lower bits of a VR128 and zeroing upper bits.
// Loading from memory automatically zeroing upper bits.
let AddedComplexity = 20 in
def MOVZSS2PSrm : SSI<0x10, MRMSrcMem, (outs VR128:$dst), (ins f32mem:$src),
                      "movss\t{$src, $dst|$dst, $src}",
                   [(set VR128:$dst, (v4f32 (X86vzmovl (v4f32 (scalar_to_vector
                                                      (loadf32 addr:$src))))))]>;

def : Pat<(v4f32 (X86vzmovl (loadv4f32 addr:$src))),
          (MOVZSS2PSrm addr:$src)>;

//===----------------------------------------------------------------------===//
// SSE2 Instructions
//===----------------------------------------------------------------------===//

// Move Instructions
let neverHasSideEffects = 1 in
def MOVSDrr : SDI<0x10, MRMSrcReg, (outs FR64:$dst), (ins FR64:$src),
                  "movsd\t{$src, $dst|$dst, $src}", []>;
let canFoldAsLoad = 1, isReMaterializable = 1, mayHaveSideEffects = 1 in
def MOVSDrm : SDI<0x10, MRMSrcMem, (outs FR64:$dst), (ins f64mem:$src),
                  "movsd\t{$src, $dst|$dst, $src}",
                  [(set FR64:$dst, (loadf64 addr:$src))]>;
def MOVSDmr : SDI<0x11, MRMDestMem, (outs), (ins f64mem:$dst, FR64:$src),
                  "movsd\t{$src, $dst|$dst, $src}",
                  [(store FR64:$src, addr:$dst)]>;

// Conversion instructions
def CVTTSD2SIrr : SDI<0x2C, MRMSrcReg, (outs GR32:$dst), (ins FR64:$src),
                      "cvttsd2si\t{$src, $dst|$dst, $src}",
                      [(set GR32:$dst, (fp_to_sint FR64:$src))]>;
def CVTTSD2SIrm : SDI<0x2C, MRMSrcMem, (outs GR32:$dst), (ins f64mem:$src),
                      "cvttsd2si\t{$src, $dst|$dst, $src}",
                      [(set GR32:$dst, (fp_to_sint (loadf64 addr:$src)))]>;
def CVTSD2SSrr  : SDI<0x5A, MRMSrcReg, (outs FR32:$dst), (ins FR64:$src),
                      "cvtsd2ss\t{$src, $dst|$dst, $src}",
                      [(set FR32:$dst, (fround FR64:$src))]>;
def CVTSD2SSrm  : SDI<0x5A, MRMSrcMem, (outs FR32:$dst), (ins f64mem:$src), 
                      "cvtsd2ss\t{$src, $dst|$dst, $src}",
                      [(set FR32:$dst, (fround (loadf64 addr:$src)))]>;
def CVTSI2SDrr  : SDI<0x2A, MRMSrcReg, (outs FR64:$dst), (ins GR32:$src),
                      "cvtsi2sd\t{$src, $dst|$dst, $src}",
                      [(set FR64:$dst, (sint_to_fp GR32:$src))]>;
def CVTSI2SDrm  : SDI<0x2A, MRMSrcMem, (outs FR64:$dst), (ins i32mem:$src),
                      "cvtsi2sd\t{$src, $dst|$dst, $src}",
                      [(set FR64:$dst, (sint_to_fp (loadi32 addr:$src)))]>;

// SSE2 instructions with XS prefix
def CVTSS2SDrr : I<0x5A, MRMSrcReg, (outs FR64:$dst), (ins FR32:$src),
                   "cvtss2sd\t{$src, $dst|$dst, $src}",
                   [(set FR64:$dst, (fextend FR32:$src))]>, XS,
                 Requires<[HasSSE2]>;
def CVTSS2SDrm : I<0x5A, MRMSrcMem, (outs FR64:$dst), (ins f32mem:$src),
                   "cvtss2sd\t{$src, $dst|$dst, $src}",
                   [(set FR64:$dst, (extloadf32 addr:$src))]>, XS,
                 Requires<[HasSSE2]>;

// Match intrinsics which expect XMM operand(s).
def Int_CVTSD2SIrr : SDI<0x2D, MRMSrcReg, (outs GR32:$dst), (ins VR128:$src),
                         "cvtsd2si\t{$src, $dst|$dst, $src}",
                         [(set GR32:$dst, (int_x86_sse2_cvtsd2si VR128:$src))]>;
def Int_CVTSD2SIrm : SDI<0x2D, MRMSrcMem, (outs GR32:$dst), (ins f128mem:$src),
                         "cvtsd2si\t{$src, $dst|$dst, $src}",
                         [(set GR32:$dst, (int_x86_sse2_cvtsd2si
                                           (load addr:$src)))]>;

// Match intrinisics which expect MM and XMM operand(s).
def Int_CVTPD2PIrr : PDI<0x2D, MRMSrcReg, (outs VR64:$dst), (ins VR128:$src),
                         "cvtpd2pi\t{$src, $dst|$dst, $src}",
                         [(set VR64:$dst, (int_x86_sse_cvtpd2pi VR128:$src))]>;
def Int_CVTPD2PIrm : PDI<0x2D, MRMSrcMem, (outs VR64:$dst), (ins f128mem:$src),
                         "cvtpd2pi\t{$src, $dst|$dst, $src}",
                         [(set VR64:$dst, (int_x86_sse_cvtpd2pi 
                                           (memop addr:$src)))]>;
def Int_CVTTPD2PIrr: PDI<0x2C, MRMSrcReg, (outs VR64:$dst), (ins VR128:$src),
                         "cvttpd2pi\t{$src, $dst|$dst, $src}",
                         [(set VR64:$dst, (int_x86_sse_cvttpd2pi VR128:$src))]>;
def Int_CVTTPD2PIrm: PDI<0x2C, MRMSrcMem, (outs VR64:$dst), (ins f128mem:$src),
                         "cvttpd2pi\t{$src, $dst|$dst, $src}",
                         [(set VR64:$dst, (int_x86_sse_cvttpd2pi 
                                           (memop addr:$src)))]>;
def Int_CVTPI2PDrr : PDI<0x2A, MRMSrcReg, (outs VR128:$dst), (ins VR64:$src),
                         "cvtpi2pd\t{$src, $dst|$dst, $src}",
                         [(set VR128:$dst, (int_x86_sse_cvtpi2pd VR64:$src))]>;
def Int_CVTPI2PDrm : PDI<0x2A, MRMSrcMem, (outs VR128:$dst), (ins i64mem:$src),
                         "cvtpi2pd\t{$src, $dst|$dst, $src}",
                         [(set VR128:$dst, (int_x86_sse_cvtpi2pd 
                                            (load addr:$src)))]>;

// Aliases for intrinsics
def Int_CVTTSD2SIrr : SDI<0x2C, MRMSrcReg, (outs GR32:$dst), (ins VR128:$src),
                          "cvttsd2si\t{$src, $dst|$dst, $src}",
                          [(set GR32:$dst,
                            (int_x86_sse2_cvttsd2si VR128:$src))]>;
def Int_CVTTSD2SIrm : SDI<0x2C, MRMSrcMem, (outs GR32:$dst), (ins f128mem:$src),
                          "cvttsd2si\t{$src, $dst|$dst, $src}",
                          [(set GR32:$dst, (int_x86_sse2_cvttsd2si
                                            (load addr:$src)))]>;

// Comparison instructions
let Constraints = "$src1 = $dst", neverHasSideEffects = 1 in {
  def CMPSDrr : SDIi8<0xC2, MRMSrcReg, 
                    (outs FR64:$dst), (ins FR64:$src1, FR64:$src, SSECC:$cc),
                    "cmp${cc}sd\t{$src, $dst|$dst, $src}", []>;
let mayLoad = 1 in
  def CMPSDrm : SDIi8<0xC2, MRMSrcMem, 
                    (outs FR64:$dst), (ins FR64:$src1, f64mem:$src, SSECC:$cc),
                    "cmp${cc}sd\t{$src, $dst|$dst, $src}", []>;
}

let Defs = [EFLAGS] in {
def UCOMISDrr: PDI<0x2E, MRMSrcReg, (outs), (ins FR64:$src1, FR64:$src2),
                   "ucomisd\t{$src2, $src1|$src1, $src2}",
                   [(X86cmp FR64:$src1, FR64:$src2), (implicit EFLAGS)]>;
def UCOMISDrm: PDI<0x2E, MRMSrcMem, (outs), (ins FR64:$src1, f64mem:$src2),
                   "ucomisd\t{$src2, $src1|$src1, $src2}",
                   [(X86cmp FR64:$src1, (loadf64 addr:$src2)),
                    (implicit EFLAGS)]>;
} // Defs = [EFLAGS]

// Aliases to match intrinsics which expect XMM operand(s).
let Constraints = "$src1 = $dst" in {
  def Int_CMPSDrr : SDIi8<0xC2, MRMSrcReg, 
                        (outs VR128:$dst), (ins VR128:$src1, VR128:$src, SSECC:$cc),
                        "cmp${cc}sd\t{$src, $dst|$dst, $src}",
                        [(set VR128:$dst, (int_x86_sse2_cmp_sd VR128:$src1,
                                           VR128:$src, imm:$cc))]>;
  def Int_CMPSDrm : SDIi8<0xC2, MRMSrcMem, 
                        (outs VR128:$dst), (ins VR128:$src1, f64mem:$src, SSECC:$cc),
                        "cmp${cc}sd\t{$src, $dst|$dst, $src}",
                        [(set VR128:$dst, (int_x86_sse2_cmp_sd VR128:$src1,
                                           (load addr:$src), imm:$cc))]>;
}

let Defs = [EFLAGS] in {
def Int_UCOMISDrr: PDI<0x2E, MRMSrcReg, (outs), (ins VR128:$src1, VR128:$src2),
                       "ucomisd\t{$src2, $src1|$src1, $src2}",
                       [(X86ucomi (v2f64 VR128:$src1), (v2f64 VR128:$src2)),
                        (implicit EFLAGS)]>;
def Int_UCOMISDrm: PDI<0x2E, MRMSrcMem, (outs),(ins VR128:$src1, f128mem:$src2),
                       "ucomisd\t{$src2, $src1|$src1, $src2}",
                       [(X86ucomi (v2f64 VR128:$src1), (load addr:$src2)),
                        (implicit EFLAGS)]>;

def Int_COMISDrr: PDI<0x2F, MRMSrcReg, (outs), (ins VR128:$src1, VR128:$src2),
                      "comisd\t{$src2, $src1|$src1, $src2}",
                      [(X86comi (v2f64 VR128:$src1), (v2f64 VR128:$src2)),
                       (implicit EFLAGS)]>;
def Int_COMISDrm: PDI<0x2F, MRMSrcMem, (outs), (ins VR128:$src1, f128mem:$src2),
                      "comisd\t{$src2, $src1|$src1, $src2}",
                      [(X86comi (v2f64 VR128:$src1), (load addr:$src2)),
                       (implicit EFLAGS)]>;
} // Defs = [EFLAGS]

// Aliases of packed SSE2 instructions for scalar use. These all have names that
// start with 'Fs'.

// Alias instructions that map fld0 to pxor for sse.
let isReMaterializable = 1, isAsCheapAsAMove = 1 in
def FsFLD0SD : I<0xEF, MRMInitReg, (outs FR64:$dst), (ins),
                 "pxor\t$dst, $dst", [(set FR64:$dst, fpimm0)]>,
               Requires<[HasSSE2]>, TB, OpSize;

// Alias instruction to do FR64 reg-to-reg copy using movapd. Upper bits are
// disregarded.
let neverHasSideEffects = 1 in
def FsMOVAPDrr : PDI<0x28, MRMSrcReg, (outs FR64:$dst), (ins FR64:$src),
                     "movapd\t{$src, $dst|$dst, $src}", []>;

// Alias instruction to load FR64 from f128mem using movapd. Upper bits are
// disregarded.
let canFoldAsLoad = 1 in
def FsMOVAPDrm : PDI<0x28, MRMSrcMem, (outs FR64:$dst), (ins f128mem:$src),
                     "movapd\t{$src, $dst|$dst, $src}",
                     [(set FR64:$dst, (alignedloadfsf64 addr:$src))]>;

// Alias bitwise logical operations using SSE logical ops on packed FP values.
let Constraints = "$src1 = $dst" in {
let isCommutable = 1 in {
  def FsANDPDrr : PDI<0x54, MRMSrcReg, (outs FR64:$dst),
                                       (ins FR64:$src1, FR64:$src2),
                      "andpd\t{$src2, $dst|$dst, $src2}",
                      [(set FR64:$dst, (X86fand FR64:$src1, FR64:$src2))]>;
  def FsORPDrr  : PDI<0x56, MRMSrcReg, (outs FR64:$dst),
                                       (ins FR64:$src1, FR64:$src2),
                      "orpd\t{$src2, $dst|$dst, $src2}",
                      [(set FR64:$dst, (X86for FR64:$src1, FR64:$src2))]>;
  def FsXORPDrr : PDI<0x57, MRMSrcReg, (outs FR64:$dst),
                                       (ins FR64:$src1, FR64:$src2),
                      "xorpd\t{$src2, $dst|$dst, $src2}",
                      [(set FR64:$dst, (X86fxor FR64:$src1, FR64:$src2))]>;
}

def FsANDPDrm : PDI<0x54, MRMSrcMem, (outs FR64:$dst),
                                     (ins FR64:$src1, f128mem:$src2),
                    "andpd\t{$src2, $dst|$dst, $src2}",
                    [(set FR64:$dst, (X86fand FR64:$src1,
                                      (memopfsf64 addr:$src2)))]>;
def FsORPDrm  : PDI<0x56, MRMSrcMem, (outs FR64:$dst),
                                     (ins FR64:$src1, f128mem:$src2),
                    "orpd\t{$src2, $dst|$dst, $src2}",
                    [(set FR64:$dst, (X86for FR64:$src1,
                                      (memopfsf64 addr:$src2)))]>;
def FsXORPDrm : PDI<0x57, MRMSrcMem, (outs FR64:$dst),
                                     (ins FR64:$src1, f128mem:$src2),
                    "xorpd\t{$src2, $dst|$dst, $src2}",
                    [(set FR64:$dst, (X86fxor FR64:$src1,
                                      (memopfsf64 addr:$src2)))]>;

let neverHasSideEffects = 1 in {
def FsANDNPDrr : PDI<0x55, MRMSrcReg,
                     (outs FR64:$dst), (ins FR64:$src1, FR64:$src2),
                     "andnpd\t{$src2, $dst|$dst, $src2}", []>;
let mayLoad = 1 in
def FsANDNPDrm : PDI<0x55, MRMSrcMem,
                     (outs FR64:$dst), (ins FR64:$src1, f128mem:$src2),
                     "andnpd\t{$src2, $dst|$dst, $src2}", []>;
}
}

/// basic_sse2_fp_binop_rm - SSE2 binops come in both scalar and vector forms.
///
/// In addition, we also have a special variant of the scalar form here to
/// represent the associated intrinsic operation.  This form is unlike the
/// plain scalar form, in that it takes an entire vector (instead of a scalar)
/// and leaves the top elements undefined.
///
/// These three forms can each be reg+reg or reg+mem, so there are a total of
/// six "instructions".
///
let Constraints = "$src1 = $dst" in {
multiclass basic_sse2_fp_binop_rm<bits<8> opc, string OpcodeStr,
                                  SDNode OpNode, Intrinsic F64Int,
                                  bit Commutable = 0> {
  // Scalar operation, reg+reg.
  def SDrr : SDI<opc, MRMSrcReg, (outs FR64:$dst), (ins FR64:$src1, FR64:$src2),
                 !strconcat(OpcodeStr, "sd\t{$src2, $dst|$dst, $src2}"),
                 [(set FR64:$dst, (OpNode FR64:$src1, FR64:$src2))]> {
    let isCommutable = Commutable;
  }

  // Scalar operation, reg+mem.
  def SDrm : SDI<opc, MRMSrcMem, (outs FR64:$dst),
                                 (ins FR64:$src1, f64mem:$src2),
                 !strconcat(OpcodeStr, "sd\t{$src2, $dst|$dst, $src2}"),
                 [(set FR64:$dst, (OpNode FR64:$src1, (load addr:$src2)))]>;
                 
  // Vector operation, reg+reg.
  def PDrr : PDI<opc, MRMSrcReg, (outs VR128:$dst),
                                 (ins VR128:$src1, VR128:$src2),
               !strconcat(OpcodeStr, "pd\t{$src2, $dst|$dst, $src2}"),
               [(set VR128:$dst, (v2f64 (OpNode VR128:$src1, VR128:$src2)))]> {
    let isCommutable = Commutable;
  }

  // Vector operation, reg+mem.
  def PDrm : PDI<opc, MRMSrcMem, (outs VR128:$dst),
                                 (ins VR128:$src1, f128mem:$src2),
                 !strconcat(OpcodeStr, "pd\t{$src2, $dst|$dst, $src2}"),
             [(set VR128:$dst, (OpNode VR128:$src1, (memopv2f64 addr:$src2)))]>;

  // Intrinsic operation, reg+reg.
  def SDrr_Int : SDI<opc, MRMSrcReg, (outs VR128:$dst),
                                     (ins VR128:$src1, VR128:$src2),
                     !strconcat(OpcodeStr, "sd\t{$src2, $dst|$dst, $src2}"),
                     [(set VR128:$dst, (F64Int VR128:$src1, VR128:$src2))]> {
    let isCommutable = Commutable;
  }

  // Intrinsic operation, reg+mem.
  def SDrm_Int : SDI<opc, MRMSrcMem, (outs VR128:$dst),
                                     (ins VR128:$src1, sdmem:$src2),
                     !strconcat(OpcodeStr, "sd\t{$src2, $dst|$dst, $src2}"),
                     [(set VR128:$dst, (F64Int VR128:$src1,
                                               sse_load_f64:$src2))]>;
}
}

// Arithmetic instructions
defm ADD : basic_sse2_fp_binop_rm<0x58, "add", fadd, int_x86_sse2_add_sd, 1>;
defm MUL : basic_sse2_fp_binop_rm<0x59, "mul", fmul, int_x86_sse2_mul_sd, 1>;
defm SUB : basic_sse2_fp_binop_rm<0x5C, "sub", fsub, int_x86_sse2_sub_sd>;
defm DIV : basic_sse2_fp_binop_rm<0x5E, "div", fdiv, int_x86_sse2_div_sd>;

/// sse2_fp_binop_rm - Other SSE2 binops
///
/// This multiclass is like basic_sse2_fp_binop_rm, with the addition of
/// instructions for a full-vector intrinsic form.  Operations that map
/// onto C operators don't use this form since they just use the plain
/// vector form instead of having a separate vector intrinsic form.
///
/// This provides a total of eight "instructions".
///
let Constraints = "$src1 = $dst" in {
multiclass sse2_fp_binop_rm<bits<8> opc, string OpcodeStr,
                            SDNode OpNode,
                            Intrinsic F64Int,
                            Intrinsic V2F64Int,
                            bit Commutable = 0> {

  // Scalar operation, reg+reg.
  def SDrr : SDI<opc, MRMSrcReg, (outs FR64:$dst), (ins FR64:$src1, FR64:$src2),
                 !strconcat(OpcodeStr, "sd\t{$src2, $dst|$dst, $src2}"),
                 [(set FR64:$dst, (OpNode FR64:$src1, FR64:$src2))]> {
    let isCommutable = Commutable;
  }

  // Scalar operation, reg+mem.
  def SDrm : SDI<opc, MRMSrcMem, (outs FR64:$dst),
                                 (ins FR64:$src1, f64mem:$src2),
                 !strconcat(OpcodeStr, "sd\t{$src2, $dst|$dst, $src2}"),
                 [(set FR64:$dst, (OpNode FR64:$src1, (load addr:$src2)))]>;
                 
  // Vector operation, reg+reg.
  def PDrr : PDI<opc, MRMSrcReg, (outs VR128:$dst),
                                 (ins VR128:$src1, VR128:$src2),
               !strconcat(OpcodeStr, "pd\t{$src2, $dst|$dst, $src2}"),
               [(set VR128:$dst, (v2f64 (OpNode VR128:$src1, VR128:$src2)))]> {
    let isCommutable = Commutable;
  }

  // Vector operation, reg+mem.
  def PDrm : PDI<opc, MRMSrcMem, (outs VR128:$dst),
                                 (ins VR128:$src1, f128mem:$src2),
                 !strconcat(OpcodeStr, "pd\t{$src2, $dst|$dst, $src2}"),
             [(set VR128:$dst, (OpNode VR128:$src1, (memopv2f64 addr:$src2)))]>;

  // Intrinsic operation, reg+reg.
  def SDrr_Int : SDI<opc, MRMSrcReg, (outs VR128:$dst),
                                     (ins VR128:$src1, VR128:$src2),
                     !strconcat(OpcodeStr, "sd\t{$src2, $dst|$dst, $src2}"),
                     [(set VR128:$dst, (F64Int VR128:$src1, VR128:$src2))]> {
    let isCommutable = Commutable;
  }

  // Intrinsic operation, reg+mem.
  def SDrm_Int : SDI<opc, MRMSrcMem, (outs VR128:$dst),
                                     (ins VR128:$src1, sdmem:$src2),
                     !strconcat(OpcodeStr, "sd\t{$src2, $dst|$dst, $src2}"),
                     [(set VR128:$dst, (F64Int VR128:$src1,
                                               sse_load_f64:$src2))]>;

  // Vector intrinsic operation, reg+reg.
  def PDrr_Int : PDI<opc, MRMSrcReg, (outs VR128:$dst),
                                     (ins VR128:$src1, VR128:$src2),
                     !strconcat(OpcodeStr, "pd\t{$src2, $dst|$dst, $src2}"),
                     [(set VR128:$dst, (V2F64Int VR128:$src1, VR128:$src2))]> {
    let isCommutable = Commutable;
  }

  // Vector intrinsic operation, reg+mem.
  def PDrm_Int : PDI<opc, MRMSrcMem, (outs VR128:$dst),
                                     (ins VR128:$src1, f128mem:$src2),
                     !strconcat(OpcodeStr, "pd\t{$src2, $dst|$dst, $src2}"),
                     [(set VR128:$dst, (V2F64Int VR128:$src1,
                                                 (memopv2f64 addr:$src2)))]>;
}
}

defm MAX : sse2_fp_binop_rm<0x5F, "max", X86fmax,
                            int_x86_sse2_max_sd, int_x86_sse2_max_pd>;
defm MIN : sse2_fp_binop_rm<0x5D, "min", X86fmin,
                            int_x86_sse2_min_sd, int_x86_sse2_min_pd>;

//===----------------------------------------------------------------------===//
// SSE packed FP Instructions

// Move Instructions
let neverHasSideEffects = 1 in
def MOVAPDrr : PDI<0x28, MRMSrcReg, (outs VR128:$dst), (ins VR128:$src),
                   "movapd\t{$src, $dst|$dst, $src}", []>;
let canFoldAsLoad = 1, isReMaterializable = 1, mayHaveSideEffects = 1 in
def MOVAPDrm : PDI<0x28, MRMSrcMem, (outs VR128:$dst), (ins f128mem:$src),
                   "movapd\t{$src, $dst|$dst, $src}",
                   [(set VR128:$dst, (alignedloadv2f64 addr:$src))]>;

def MOVAPDmr : PDI<0x29, MRMDestMem, (outs), (ins f128mem:$dst, VR128:$src),
                   "movapd\t{$src, $dst|$dst, $src}",
                   [(alignedstore (v2f64 VR128:$src), addr:$dst)]>;

let neverHasSideEffects = 1 in
def MOVUPDrr : PDI<0x10, MRMSrcReg, (outs VR128:$dst), (ins VR128:$src),
                   "movupd\t{$src, $dst|$dst, $src}", []>;
let canFoldAsLoad = 1 in
def MOVUPDrm : PDI<0x10, MRMSrcMem, (outs VR128:$dst), (ins f128mem:$src),
                   "movupd\t{$src, $dst|$dst, $src}",
                   [(set VR128:$dst, (loadv2f64 addr:$src))]>;
def MOVUPDmr : PDI<0x11, MRMDestMem, (outs), (ins f128mem:$dst, VR128:$src),
                   "movupd\t{$src, $dst|$dst, $src}",
                   [(store (v2f64 VR128:$src), addr:$dst)]>;

// Intrinsic forms of MOVUPD load and store
def MOVUPDrm_Int : PDI<0x10, MRMSrcMem, (outs VR128:$dst), (ins f128mem:$src),
                       "movupd\t{$src, $dst|$dst, $src}",
                       [(set VR128:$dst, (int_x86_sse2_loadu_pd addr:$src))]>;
def MOVUPDmr_Int : PDI<0x11, MRMDestMem, (outs), (ins f128mem:$dst, VR128:$src),
                       "movupd\t{$src, $dst|$dst, $src}",
                       [(int_x86_sse2_storeu_pd addr:$dst, VR128:$src)]>;

let Constraints = "$src1 = $dst" in {
  let AddedComplexity = 20 in {
    def MOVLPDrm : PDI<0x12, MRMSrcMem,
                       (outs VR128:$dst), (ins VR128:$src1, f64mem:$src2),
                       "movlpd\t{$src2, $dst|$dst, $src2}",
                       [(set VR128:$dst, 
                         (v2f64 (vector_shuffle VR128:$src1,
                                 (scalar_to_vector (loadf64 addr:$src2)),
                                 MOVLP_shuffle_mask)))]>;
    def MOVHPDrm : PDI<0x16, MRMSrcMem,
                       (outs VR128:$dst), (ins VR128:$src1, f64mem:$src2),
                       "movhpd\t{$src2, $dst|$dst, $src2}",
                       [(set VR128:$dst, 
                         (v2f64 (vector_shuffle VR128:$src1,
                                 (scalar_to_vector (loadf64 addr:$src2)),
                                 MOVHP_shuffle_mask)))]>;
  } // AddedComplexity
} // Constraints = "$src1 = $dst"

def MOVLPDmr : PDI<0x13, MRMDestMem, (outs), (ins f64mem:$dst, VR128:$src),
                   "movlpd\t{$src, $dst|$dst, $src}",
                   [(store (f64 (vector_extract (v2f64 VR128:$src),
                                 (iPTR 0))), addr:$dst)]>;

// v2f64 extract element 1 is always custom lowered to unpack high to low
// and extract element 0 so the non-store version isn't too horrible.
def MOVHPDmr : PDI<0x17, MRMDestMem, (outs), (ins f64mem:$dst, VR128:$src),
                   "movhpd\t{$src, $dst|$dst, $src}",
                   [(store (f64 (vector_extract
                                 (v2f64 (vector_shuffle VR128:$src, (undef),
                                         UNPCKH_shuffle_mask)), (iPTR 0))),
                     addr:$dst)]>;

// SSE2 instructions without OpSize prefix
def Int_CVTDQ2PSrr : I<0x5B, MRMSrcReg, (outs VR128:$dst), (ins VR128:$src),
                       "cvtdq2ps\t{$src, $dst|$dst, $src}",
                       [(set VR128:$dst, (int_x86_sse2_cvtdq2ps VR128:$src))]>,
                     TB, Requires<[HasSSE2]>;
def Int_CVTDQ2PSrm : I<0x5B, MRMSrcMem, (outs VR128:$dst), (ins i128mem:$src),
                      "cvtdq2ps\t{$src, $dst|$dst, $src}",
                      [(set VR128:$dst, (int_x86_sse2_cvtdq2ps
                                        (bitconvert (memopv2i64 addr:$src))))]>,
                     TB, Requires<[HasSSE2]>;

// SSE2 instructions with XS prefix
def Int_CVTDQ2PDrr : I<0xE6, MRMSrcReg, (outs VR128:$dst), (ins VR128:$src),
                       "cvtdq2pd\t{$src, $dst|$dst, $src}",
                       [(set VR128:$dst, (int_x86_sse2_cvtdq2pd VR128:$src))]>,
                     XS, Requires<[HasSSE2]>;
def Int_CVTDQ2PDrm : I<0xE6, MRMSrcMem, (outs VR128:$dst), (ins i64mem:$src),
                     "cvtdq2pd\t{$src, $dst|$dst, $src}",
                     [(set VR128:$dst, (int_x86_sse2_cvtdq2pd
                                        (bitconvert (memopv2i64 addr:$src))))]>,
                     XS, Requires<[HasSSE2]>;

def Int_CVTPS2DQrr : PDI<0x5B, MRMSrcReg, (outs VR128:$dst), (ins VR128:$src),
                        "cvtps2dq\t{$src, $dst|$dst, $src}",
                        [(set VR128:$dst, (int_x86_sse2_cvtps2dq VR128:$src))]>;
def Int_CVTPS2DQrm : PDI<0x5B, MRMSrcMem, (outs VR128:$dst), (ins f128mem:$src),
                         "cvtps2dq\t{$src, $dst|$dst, $src}",
                         [(set VR128:$dst, (int_x86_sse2_cvtps2dq
                                            (memop addr:$src)))]>;
// SSE2 packed instructions with XS prefix
def Int_CVTTPS2DQrr : I<0x5B, MRMSrcReg, (outs VR128:$dst), (ins VR128:$src),
                        "cvttps2dq\t{$src, $dst|$dst, $src}",
                        [(set VR128:$dst, (int_x86_sse2_cvttps2dq VR128:$src))]>,
                      XS, Requires<[HasSSE2]>;
def Int_CVTTPS2DQrm : I<0x5B, MRMSrcMem, (outs VR128:$dst), (ins f128mem:$src),
                        "cvttps2dq\t{$src, $dst|$dst, $src}",
                        [(set VR128:$dst, (int_x86_sse2_cvttps2dq
                                           (memop addr:$src)))]>,
                      XS, Requires<[HasSSE2]>;

// SSE2 packed instructions with XD prefix
def Int_CVTPD2DQrr : I<0xE6, MRMSrcReg, (outs VR128:$dst), (ins VR128:$src),
                       "cvtpd2dq\t{$src, $dst|$dst, $src}",
                       [(set VR128:$dst, (int_x86_sse2_cvtpd2dq VR128:$src))]>,
                     XD, Requires<[HasSSE2]>;
def Int_CVTPD2DQrm : I<0xE6, MRMSrcMem, (outs VR128:$dst), (ins f128mem:$src),
                       "cvtpd2dq\t{$src, $dst|$dst, $src}",
                       [(set VR128:$dst, (int_x86_sse2_cvtpd2dq
                                          (memop addr:$src)))]>,
                     XD, Requires<[HasSSE2]>;

def Int_CVTTPD2DQrr : PDI<0xE6, MRMSrcReg, (outs VR128:$dst), (ins VR128:$src),
                          "cvttpd2dq\t{$src, $dst|$dst, $src}",
                       [(set VR128:$dst, (int_x86_sse2_cvttpd2dq VR128:$src))]>;
def Int_CVTTPD2DQrm : PDI<0xE6, MRMSrcMem, (outs VR128:$dst),(ins f128mem:$src),
                          "cvttpd2dq\t{$src, $dst|$dst, $src}",
                          [(set VR128:$dst, (int_x86_sse2_cvttpd2dq
                                             (memop addr:$src)))]>;

// SSE2 instructions without OpSize prefix
def Int_CVTPS2PDrr : I<0x5A, MRMSrcReg, (outs VR128:$dst), (ins VR128:$src),
                       "cvtps2pd\t{$src, $dst|$dst, $src}",
                       [(set VR128:$dst, (int_x86_sse2_cvtps2pd VR128:$src))]>,
                     TB, Requires<[HasSSE2]>;
def Int_CVTPS2PDrm : I<0x5A, MRMSrcMem, (outs VR128:$dst), (ins f64mem:$src),
                       "cvtps2pd\t{$src, $dst|$dst, $src}",
                       [(set VR128:$dst, (int_x86_sse2_cvtps2pd
                                          (load addr:$src)))]>,
                     TB, Requires<[HasSSE2]>;

def Int_CVTPD2PSrr : PDI<0x5A, MRMSrcReg, (outs VR128:$dst), (ins VR128:$src),
                         "cvtpd2ps\t{$src, $dst|$dst, $src}",
                        [(set VR128:$dst, (int_x86_sse2_cvtpd2ps VR128:$src))]>;
def Int_CVTPD2PSrm : PDI<0x5A, MRMSrcMem, (outs VR128:$dst), (ins f128mem:$src),
                         "cvtpd2ps\t{$src, $dst|$dst, $src}",
                         [(set VR128:$dst, (int_x86_sse2_cvtpd2ps
                                            (memop addr:$src)))]>;

// Match intrinsics which expect XMM operand(s).
// Aliases for intrinsics
let Constraints = "$src1 = $dst" in {
def Int_CVTSI2SDrr: SDI<0x2A, MRMSrcReg,
                        (outs VR128:$dst), (ins VR128:$src1, GR32:$src2),
                        "cvtsi2sd\t{$src2, $dst|$dst, $src2}",
                        [(set VR128:$dst, (int_x86_sse2_cvtsi2sd VR128:$src1,
                                           GR32:$src2))]>;
def Int_CVTSI2SDrm: SDI<0x2A, MRMSrcMem,
                        (outs VR128:$dst), (ins VR128:$src1, i32mem:$src2),
                        "cvtsi2sd\t{$src2, $dst|$dst, $src2}",
                        [(set VR128:$dst, (int_x86_sse2_cvtsi2sd VR128:$src1,
                                           (loadi32 addr:$src2)))]>;
def Int_CVTSD2SSrr: SDI<0x5A, MRMSrcReg,
                        (outs VR128:$dst), (ins VR128:$src1, VR128:$src2),
                   "cvtsd2ss\t{$src2, $dst|$dst, $src2}",
                   [(set VR128:$dst, (int_x86_sse2_cvtsd2ss VR128:$src1,
                                      VR128:$src2))]>;
def Int_CVTSD2SSrm: SDI<0x5A, MRMSrcMem,
                        (outs VR128:$dst), (ins VR128:$src1, f64mem:$src2), 
                   "cvtsd2ss\t{$src2, $dst|$dst, $src2}",
                   [(set VR128:$dst, (int_x86_sse2_cvtsd2ss VR128:$src1,
                                      (load addr:$src2)))]>;
def Int_CVTSS2SDrr: I<0x5A, MRMSrcReg,
                      (outs VR128:$dst), (ins VR128:$src1, VR128:$src2),
                    "cvtss2sd\t{$src2, $dst|$dst, $src2}",
                    [(set VR128:$dst, (int_x86_sse2_cvtss2sd VR128:$src1,
                                       VR128:$src2))]>, XS,
                    Requires<[HasSSE2]>;
def Int_CVTSS2SDrm: I<0x5A, MRMSrcMem,
                      (outs VR128:$dst), (ins VR128:$src1, f32mem:$src2),
                    "cvtss2sd\t{$src2, $dst|$dst, $src2}",
                    [(set VR128:$dst, (int_x86_sse2_cvtss2sd VR128:$src1,
                                       (load addr:$src2)))]>, XS,
                    Requires<[HasSSE2]>;
}

// Arithmetic

/// sse2_fp_unop_rm - SSE2 unops come in both scalar and vector forms.
///
/// In addition, we also have a special variant of the scalar form here to
/// represent the associated intrinsic operation.  This form is unlike the
/// plain scalar form, in that it takes an entire vector (instead of a
/// scalar) and leaves the top elements undefined.
///
/// And, we have a special variant form for a full-vector intrinsic form.
///
/// These four forms can each have a reg or a mem operand, so there are a
/// total of eight "instructions".
///
multiclass sse2_fp_unop_rm<bits<8> opc, string OpcodeStr,
                           SDNode OpNode,
                           Intrinsic F64Int,
                           Intrinsic V2F64Int,
                           bit Commutable = 0> {
  // Scalar operation, reg.
  def SDr : SDI<opc, MRMSrcReg, (outs FR64:$dst), (ins FR64:$src),
                !strconcat(OpcodeStr, "sd\t{$src, $dst|$dst, $src}"),
                [(set FR64:$dst, (OpNode FR64:$src))]> {
    let isCommutable = Commutable;
  }

  // Scalar operation, mem.
  def SDm : SDI<opc, MRMSrcMem, (outs FR64:$dst), (ins f64mem:$src),
                !strconcat(OpcodeStr, "sd\t{$src, $dst|$dst, $src}"),
                [(set FR64:$dst, (OpNode (load addr:$src)))]>;
                 
  // Vector operation, reg.
  def PDr : PDI<opc, MRMSrcReg, (outs VR128:$dst), (ins VR128:$src),
              !strconcat(OpcodeStr, "pd\t{$src, $dst|$dst, $src}"),
              [(set VR128:$dst, (v2f64 (OpNode VR128:$src)))]> {
    let isCommutable = Commutable;
  }

  // Vector operation, mem.
  def PDm : PDI<opc, MRMSrcMem, (outs VR128:$dst), (ins f128mem:$src),
                !strconcat(OpcodeStr, "pd\t{$src, $dst|$dst, $src}"),
                [(set VR128:$dst, (OpNode (memopv2f64 addr:$src)))]>;

  // Intrinsic operation, reg.
  def SDr_Int : SDI<opc, MRMSrcReg, (outs VR128:$dst), (ins VR128:$src),
                    !strconcat(OpcodeStr, "sd\t{$src, $dst|$dst, $src}"),
                    [(set VR128:$dst, (F64Int VR128:$src))]> {
    let isCommutable = Commutable;
  }

  // Intrinsic operation, mem.
  def SDm_Int : SDI<opc, MRMSrcMem, (outs VR128:$dst), (ins sdmem:$src),
                    !strconcat(OpcodeStr, "sd\t{$src, $dst|$dst, $src}"),
                    [(set VR128:$dst, (F64Int sse_load_f64:$src))]>;

  // Vector intrinsic operation, reg
  def PDr_Int : PDI<opc, MRMSrcReg, (outs VR128:$dst), (ins VR128:$src),
                    !strconcat(OpcodeStr, "pd\t{$src, $dst|$dst, $src}"),
                    [(set VR128:$dst, (V2F64Int VR128:$src))]> {
    let isCommutable = Commutable;
  }

  // Vector intrinsic operation, mem
  def PDm_Int : PDI<opc, MRMSrcMem, (outs VR128:$dst), (ins f128mem:$src),
                    !strconcat(OpcodeStr, "pd\t{$src, $dst|$dst, $src}"),
                    [(set VR128:$dst, (V2F64Int (memopv2f64 addr:$src)))]>;
}

// Square root.
defm SQRT  : sse2_fp_unop_rm<0x51, "sqrt",  fsqrt,
                             int_x86_sse2_sqrt_sd, int_x86_sse2_sqrt_pd>;

// There is no f64 version of the reciprocal approximation instructions.

// Logical
let Constraints = "$src1 = $dst" in {
  let isCommutable = 1 in {
    def ANDPDrr : PDI<0x54, MRMSrcReg,
                      (outs VR128:$dst), (ins VR128:$src1, VR128:$src2),
                      "andpd\t{$src2, $dst|$dst, $src2}",
                      [(set VR128:$dst,
                        (and (bc_v2i64 (v2f64 VR128:$src1)),
                         (bc_v2i64 (v2f64 VR128:$src2))))]>;
    def ORPDrr  : PDI<0x56, MRMSrcReg,
                      (outs VR128:$dst), (ins VR128:$src1, VR128:$src2),
                      "orpd\t{$src2, $dst|$dst, $src2}",
                      [(set VR128:$dst,
                        (or (bc_v2i64 (v2f64 VR128:$src1)),
                         (bc_v2i64 (v2f64 VR128:$src2))))]>;
    def XORPDrr : PDI<0x57, MRMSrcReg,
                      (outs VR128:$dst), (ins VR128:$src1, VR128:$src2),
                      "xorpd\t{$src2, $dst|$dst, $src2}",
                      [(set VR128:$dst,
                        (xor (bc_v2i64 (v2f64 VR128:$src1)),
                         (bc_v2i64 (v2f64 VR128:$src2))))]>;
  }

  def ANDPDrm : PDI<0x54, MRMSrcMem,
                    (outs VR128:$dst), (ins VR128:$src1, f128mem:$src2),
                    "andpd\t{$src2, $dst|$dst, $src2}",
                    [(set VR128:$dst,
                      (and (bc_v2i64 (v2f64 VR128:$src1)),
                       (memopv2i64 addr:$src2)))]>;
  def ORPDrm  : PDI<0x56, MRMSrcMem,
                    (outs VR128:$dst), (ins VR128:$src1, f128mem:$src2),
                    "orpd\t{$src2, $dst|$dst, $src2}",
                    [(set VR128:$dst,
                      (or (bc_v2i64 (v2f64 VR128:$src1)),
                       (memopv2i64 addr:$src2)))]>;
  def XORPDrm : PDI<0x57, MRMSrcMem,
                    (outs VR128:$dst), (ins VR128:$src1, f128mem:$src2),
                    "xorpd\t{$src2, $dst|$dst, $src2}",
                    [(set VR128:$dst,
                      (xor (bc_v2i64 (v2f64 VR128:$src1)),
                       (memopv2i64 addr:$src2)))]>;
  def ANDNPDrr : PDI<0x55, MRMSrcReg,
                     (outs VR128:$dst), (ins VR128:$src1, VR128:$src2),
                     "andnpd\t{$src2, $dst|$dst, $src2}",
                     [(set VR128:$dst,
                       (and (vnot (bc_v2i64 (v2f64 VR128:$src1))),
                        (bc_v2i64 (v2f64 VR128:$src2))))]>;
  def ANDNPDrm : PDI<0x55, MRMSrcMem,
                     (outs VR128:$dst), (ins VR128:$src1,f128mem:$src2),
                     "andnpd\t{$src2, $dst|$dst, $src2}",
                     [(set VR128:$dst,
                       (and (vnot (bc_v2i64 (v2f64 VR128:$src1))),
                        (memopv2i64 addr:$src2)))]>;
}

let Constraints = "$src1 = $dst" in {
  def CMPPDrri : PDIi8<0xC2, MRMSrcReg, 
                    (outs VR128:$dst), (ins VR128:$src1, VR128:$src, SSECC:$cc),
                    "cmp${cc}pd\t{$src, $dst|$dst, $src}",
                    [(set VR128:$dst, (int_x86_sse2_cmp_pd VR128:$src1,
                                                        VR128:$src, imm:$cc))]>;
  def CMPPDrmi : PDIi8<0xC2, MRMSrcMem, 
                  (outs VR128:$dst), (ins VR128:$src1, f128mem:$src, SSECC:$cc),
                  "cmp${cc}pd\t{$src, $dst|$dst, $src}",
                  [(set VR128:$dst, (int_x86_sse2_cmp_pd VR128:$src1,
                                                 (memop addr:$src), imm:$cc))]>;
}
def : Pat<(v2i64 (X86cmppd (v2f64 VR128:$src1), VR128:$src2, imm:$cc)),
          (CMPPDrri VR128:$src1, VR128:$src2, imm:$cc)>;
def : Pat<(v2i64 (X86cmppd (v2f64 VR128:$src1), (memop addr:$src2), imm:$cc)),
          (CMPPDrmi VR128:$src1, addr:$src2, imm:$cc)>;

// Shuffle and unpack instructions
let Constraints = "$src1 = $dst" in {
  def SHUFPDrri : PDIi8<0xC6, MRMSrcReg, 
                 (outs VR128:$dst), (ins VR128:$src1, VR128:$src2, i8imm:$src3),
                 "shufpd\t{$src3, $src2, $dst|$dst, $src2, $src3}",
                 [(set VR128:$dst, (v2f64 (vector_shuffle
                                           VR128:$src1, VR128:$src2,
                                           SHUFP_shuffle_mask:$src3)))]>;
  def SHUFPDrmi : PDIi8<0xC6, MRMSrcMem, 
                        (outs VR128:$dst), (ins VR128:$src1,
                         f128mem:$src2, i8imm:$src3),
                        "shufpd\t{$src3, $src2, $dst|$dst, $src2, $src3}",
                        [(set VR128:$dst,
                          (v2f64 (vector_shuffle
                                  VR128:$src1, (memopv2f64 addr:$src2),
                                  SHUFP_shuffle_mask:$src3)))]>;

  let AddedComplexity = 10 in {
    def UNPCKHPDrr : PDI<0x15, MRMSrcReg, 
                         (outs VR128:$dst), (ins VR128:$src1, VR128:$src2),
                         "unpckhpd\t{$src2, $dst|$dst, $src2}",
                         [(set VR128:$dst,
                           (v2f64 (vector_shuffle
                                   VR128:$src1, VR128:$src2,
                                   UNPCKH_shuffle_mask)))]>;
    def UNPCKHPDrm : PDI<0x15, MRMSrcMem, 
                         (outs VR128:$dst), (ins VR128:$src1, f128mem:$src2),
                         "unpckhpd\t{$src2, $dst|$dst, $src2}",
                         [(set VR128:$dst,
                           (v2f64 (vector_shuffle
                                   VR128:$src1, (memopv2f64 addr:$src2),
                                   UNPCKH_shuffle_mask)))]>;

    def UNPCKLPDrr : PDI<0x14, MRMSrcReg, 
                         (outs VR128:$dst), (ins VR128:$src1, VR128:$src2),
                         "unpcklpd\t{$src2, $dst|$dst, $src2}",
                         [(set VR128:$dst,
                           (v2f64 (vector_shuffle
                                   VR128:$src1, VR128:$src2,
                                   UNPCKL_shuffle_mask)))]>;
    def UNPCKLPDrm : PDI<0x14, MRMSrcMem, 
                         (outs VR128:$dst), (ins VR128:$src1, f128mem:$src2),
                         "unpcklpd\t{$src2, $dst|$dst, $src2}",
                         [(set VR128:$dst,
                           (v2f64 (vector_shuffle
                                   VR128:$src1, (memopv2f64 addr:$src2),
                                   UNPCKL_shuffle_mask)))]>;
  } // AddedComplexity
} // Constraints = "$src1 = $dst"


//===----------------------------------------------------------------------===//
// SSE integer instructions

// Move Instructions
let neverHasSideEffects = 1 in
def MOVDQArr : PDI<0x6F, MRMSrcReg, (outs VR128:$dst), (ins VR128:$src),
                   "movdqa\t{$src, $dst|$dst, $src}", []>;
let canFoldAsLoad = 1, mayLoad = 1 in
def MOVDQArm : PDI<0x6F, MRMSrcMem, (outs VR128:$dst), (ins i128mem:$src),
                   "movdqa\t{$src, $dst|$dst, $src}",
                   [/*(set VR128:$dst, (alignedloadv2i64 addr:$src))*/]>;
let mayStore = 1 in
def MOVDQAmr : PDI<0x7F, MRMDestMem, (outs), (ins i128mem:$dst, VR128:$src),
                   "movdqa\t{$src, $dst|$dst, $src}",
                   [/*(alignedstore (v2i64 VR128:$src), addr:$dst)*/]>;
let canFoldAsLoad = 1, mayLoad = 1 in
def MOVDQUrm :   I<0x6F, MRMSrcMem, (outs VR128:$dst), (ins i128mem:$src),
                   "movdqu\t{$src, $dst|$dst, $src}",
                   [/*(set VR128:$dst, (loadv2i64 addr:$src))*/]>,
                 XS, Requires<[HasSSE2]>;
let mayStore = 1 in
def MOVDQUmr :   I<0x7F, MRMDestMem, (outs), (ins i128mem:$dst, VR128:$src),
                   "movdqu\t{$src, $dst|$dst, $src}",
                   [/*(store (v2i64 VR128:$src), addr:$dst)*/]>,
                 XS, Requires<[HasSSE2]>;

// Intrinsic forms of MOVDQU load and store
let canFoldAsLoad = 1 in
def MOVDQUrm_Int :   I<0x6F, MRMSrcMem, (outs VR128:$dst), (ins i128mem:$src),
                       "movdqu\t{$src, $dst|$dst, $src}",
                       [(set VR128:$dst, (int_x86_sse2_loadu_dq addr:$src))]>,
                 XS, Requires<[HasSSE2]>;
def MOVDQUmr_Int :   I<0x7F, MRMDestMem, (outs), (ins i128mem:$dst, VR128:$src),
                       "movdqu\t{$src, $dst|$dst, $src}",
                       [(int_x86_sse2_storeu_dq addr:$dst, VR128:$src)]>,
                     XS, Requires<[HasSSE2]>;

let Constraints = "$src1 = $dst" in {

multiclass PDI_binop_rm_int<bits<8> opc, string OpcodeStr, Intrinsic IntId,
                            bit Commutable = 0> {
  def rr : PDI<opc, MRMSrcReg, (outs VR128:$dst), (ins VR128:$src1, VR128:$src2),
               !strconcat(OpcodeStr, "\t{$src2, $dst|$dst, $src2}"),
               [(set VR128:$dst, (IntId VR128:$src1, VR128:$src2))]> {
    let isCommutable = Commutable;
  }
  def rm : PDI<opc, MRMSrcMem, (outs VR128:$dst), (ins VR128:$src1, i128mem:$src2),
               !strconcat(OpcodeStr, "\t{$src2, $dst|$dst, $src2}"),
               [(set VR128:$dst, (IntId VR128:$src1,
                                        (bitconvert (memopv2i64 addr:$src2))))]>;
}

multiclass PDI_binop_rmi_int<bits<8> opc, bits<8> opc2, Format ImmForm,
                             string OpcodeStr,
                             Intrinsic IntId, Intrinsic IntId2> {
  def rr : PDI<opc, MRMSrcReg, (outs VR128:$dst), (ins VR128:$src1, VR128:$src2),
               !strconcat(OpcodeStr, "\t{$src2, $dst|$dst, $src2}"),
               [(set VR128:$dst, (IntId VR128:$src1, VR128:$src2))]>;
  def rm : PDI<opc, MRMSrcMem, (outs VR128:$dst), (ins VR128:$src1, i128mem:$src2),
               !strconcat(OpcodeStr, "\t{$src2, $dst|$dst, $src2}"),
               [(set VR128:$dst, (IntId VR128:$src1,
                                        (bitconvert (memopv2i64 addr:$src2))))]>;
  def ri : PDIi8<opc2, ImmForm, (outs VR128:$dst), (ins VR128:$src1, i32i8imm:$src2),
               !strconcat(OpcodeStr, "\t{$src2, $dst|$dst, $src2}"),
               [(set VR128:$dst, (IntId2 VR128:$src1, (i32 imm:$src2)))]>;
}

/// PDI_binop_rm - Simple SSE2 binary operator.
multiclass PDI_binop_rm<bits<8> opc, string OpcodeStr, SDNode OpNode,
                        ValueType OpVT, bit Commutable = 0> {
  def rr : PDI<opc, MRMSrcReg, (outs VR128:$dst), (ins VR128:$src1, VR128:$src2),
               !strconcat(OpcodeStr, "\t{$src2, $dst|$dst, $src2}"),
               [(set VR128:$dst, (OpVT (OpNode VR128:$src1, VR128:$src2)))]> {
    let isCommutable = Commutable;
  }
  def rm : PDI<opc, MRMSrcMem, (outs VR128:$dst), (ins VR128:$src1, i128mem:$src2),
               !strconcat(OpcodeStr, "\t{$src2, $dst|$dst, $src2}"),
               [(set VR128:$dst, (OpVT (OpNode VR128:$src1,
                                       (bitconvert (memopv2i64 addr:$src2)))))]>;
}

/// PDI_binop_rm_v2i64 - Simple SSE2 binary operator whose type is v2i64.
///
/// FIXME: we could eliminate this and use PDI_binop_rm instead if tblgen knew
/// to collapse (bitconvert VT to VT) into its operand.
///
multiclass PDI_binop_rm_v2i64<bits<8> opc, string OpcodeStr, SDNode OpNode,
                              bit Commutable = 0> {
  def rr : PDI<opc, MRMSrcReg, (outs VR128:$dst), (ins VR128:$src1, VR128:$src2),
               !strconcat(OpcodeStr, "\t{$src2, $dst|$dst, $src2}"),
               [(set VR128:$dst, (v2i64 (OpNode VR128:$src1, VR128:$src2)))]> {
    let isCommutable = Commutable;
  }
  def rm : PDI<opc, MRMSrcMem, (outs VR128:$dst), (ins VR128:$src1, i128mem:$src2),
               !strconcat(OpcodeStr, "\t{$src2, $dst|$dst, $src2}"),
               [(set VR128:$dst, (OpNode VR128:$src1,(memopv2i64 addr:$src2)))]>;
}

} // Constraints = "$src1 = $dst"

// 128-bit Integer Arithmetic

defm PADDB : PDI_binop_rm<0xFC, "paddb", add, v16i8, 1>;
defm PADDW : PDI_binop_rm<0xFD, "paddw", add, v8i16, 1>;
defm PADDD : PDI_binop_rm<0xFE, "paddd", add, v4i32, 1>;
defm PADDQ : PDI_binop_rm_v2i64<0xD4, "paddq", add, 1>;

defm PADDSB  : PDI_binop_rm_int<0xEC, "paddsb" , int_x86_sse2_padds_b, 1>;
defm PADDSW  : PDI_binop_rm_int<0xED, "paddsw" , int_x86_sse2_padds_w, 1>;
defm PADDUSB : PDI_binop_rm_int<0xDC, "paddusb", int_x86_sse2_paddus_b, 1>;
defm PADDUSW : PDI_binop_rm_int<0xDD, "paddusw", int_x86_sse2_paddus_w, 1>;

defm PSUBB : PDI_binop_rm<0xF8, "psubb", sub, v16i8>;
defm PSUBW : PDI_binop_rm<0xF9, "psubw", sub, v8i16>;
defm PSUBD : PDI_binop_rm<0xFA, "psubd", sub, v4i32>;
defm PSUBQ : PDI_binop_rm_v2i64<0xFB, "psubq", sub>;

defm PSUBSB  : PDI_binop_rm_int<0xE8, "psubsb" , int_x86_sse2_psubs_b>;
defm PSUBSW  : PDI_binop_rm_int<0xE9, "psubsw" , int_x86_sse2_psubs_w>;
defm PSUBUSB : PDI_binop_rm_int<0xD8, "psubusb", int_x86_sse2_psubus_b>;
defm PSUBUSW : PDI_binop_rm_int<0xD9, "psubusw", int_x86_sse2_psubus_w>;

defm PMULLW : PDI_binop_rm<0xD5, "pmullw", mul, v8i16, 1>;

defm PMULHUW : PDI_binop_rm_int<0xE4, "pmulhuw", int_x86_sse2_pmulhu_w, 1>;
defm PMULHW  : PDI_binop_rm_int<0xE5, "pmulhw" , int_x86_sse2_pmulh_w , 1>;
defm PMULUDQ : PDI_binop_rm_int<0xF4, "pmuludq", int_x86_sse2_pmulu_dq, 1>;

defm PMADDWD : PDI_binop_rm_int<0xF5, "pmaddwd", int_x86_sse2_pmadd_wd, 1>;

defm PAVGB  : PDI_binop_rm_int<0xE0, "pavgb", int_x86_sse2_pavg_b, 1>;
defm PAVGW  : PDI_binop_rm_int<0xE3, "pavgw", int_x86_sse2_pavg_w, 1>;


defm PMINUB : PDI_binop_rm_int<0xDA, "pminub", int_x86_sse2_pminu_b, 1>;
defm PMINSW : PDI_binop_rm_int<0xEA, "pminsw", int_x86_sse2_pmins_w, 1>;
defm PMAXUB : PDI_binop_rm_int<0xDE, "pmaxub", int_x86_sse2_pmaxu_b, 1>;
defm PMAXSW : PDI_binop_rm_int<0xEE, "pmaxsw", int_x86_sse2_pmaxs_w, 1>;
defm PSADBW : PDI_binop_rm_int<0xE0, "psadbw", int_x86_sse2_psad_bw, 1>;


defm PSLLW : PDI_binop_rmi_int<0xF1, 0x71, MRM6r, "psllw",
                               int_x86_sse2_psll_w, int_x86_sse2_pslli_w>;
defm PSLLD : PDI_binop_rmi_int<0xF2, 0x72, MRM6r, "pslld",
                               int_x86_sse2_psll_d, int_x86_sse2_pslli_d>;
defm PSLLQ : PDI_binop_rmi_int<0xF3, 0x73, MRM6r, "psllq",
                               int_x86_sse2_psll_q, int_x86_sse2_pslli_q>;

defm PSRLW : PDI_binop_rmi_int<0xD1, 0x71, MRM2r, "psrlw",
                               int_x86_sse2_psrl_w, int_x86_sse2_psrli_w>;
defm PSRLD : PDI_binop_rmi_int<0xD2, 0x72, MRM2r, "psrld",
                               int_x86_sse2_psrl_d, int_x86_sse2_psrli_d>;
defm PSRLQ : PDI_binop_rmi_int<0xD3, 0x73, MRM2r, "psrlq",
                               int_x86_sse2_psrl_q, int_x86_sse2_psrli_q>;

defm PSRAW : PDI_binop_rmi_int<0xE1, 0x71, MRM4r, "psraw",
                               int_x86_sse2_psra_w, int_x86_sse2_psrai_w>;
defm PSRAD : PDI_binop_rmi_int<0xE2, 0x72, MRM4r, "psrad",
                               int_x86_sse2_psra_d, int_x86_sse2_psrai_d>;

// 128-bit logical shifts.
let Constraints = "$src1 = $dst", neverHasSideEffects = 1 in {
  def PSLLDQri : PDIi8<0x73, MRM7r,
                       (outs VR128:$dst), (ins VR128:$src1, i32i8imm:$src2),
                       "pslldq\t{$src2, $dst|$dst, $src2}", []>;
  def PSRLDQri : PDIi8<0x73, MRM3r,
                       (outs VR128:$dst), (ins VR128:$src1, i32i8imm:$src2),
                       "psrldq\t{$src2, $dst|$dst, $src2}", []>;
  // PSRADQri doesn't exist in SSE[1-3].
}

let Predicates = [HasSSE2] in {
  def : Pat<(int_x86_sse2_psll_dq VR128:$src1, imm:$src2),
            (v2i64 (PSLLDQri VR128:$src1, (PSxLDQ_imm imm:$src2)))>;
  def : Pat<(int_x86_sse2_psrl_dq VR128:$src1, imm:$src2),
            (v2i64 (PSRLDQri VR128:$src1, (PSxLDQ_imm imm:$src2)))>;
  def : Pat<(int_x86_sse2_psll_dq_bs VR128:$src1, imm:$src2),
            (v2i64 (PSLLDQri VR128:$src1, imm:$src2))>;
  def : Pat<(int_x86_sse2_psrl_dq_bs VR128:$src1, imm:$src2),
            (v2i64 (PSRLDQri VR128:$src1, imm:$src2))>;
  def : Pat<(v2f64 (X86fsrl VR128:$src1, i32immSExt8:$src2)),
            (v2f64 (PSRLDQri VR128:$src1, (PSxLDQ_imm imm:$src2)))>;

  // Shift up / down and insert zero's.
  def : Pat<(v2i64 (X86vshl  VR128:$src, (i8 imm:$amt))),
            (v2i64 (PSLLDQri VR128:$src, (PSxLDQ_imm imm:$amt)))>;
  def : Pat<(v2i64 (X86vshr  VR128:$src, (i8 imm:$amt))),
            (v2i64 (PSRLDQri VR128:$src, (PSxLDQ_imm imm:$amt)))>;
}

// Logical
defm PAND : PDI_binop_rm_v2i64<0xDB, "pand", and, 1>;
defm POR  : PDI_binop_rm_v2i64<0xEB, "por" , or , 1>;
defm PXOR : PDI_binop_rm_v2i64<0xEF, "pxor", xor, 1>;

let Constraints = "$src1 = $dst" in {
  def PANDNrr : PDI<0xDF, MRMSrcReg,
                    (outs VR128:$dst), (ins VR128:$src1, VR128:$src2),
                    "pandn\t{$src2, $dst|$dst, $src2}",
                    [(set VR128:$dst, (v2i64 (and (vnot VR128:$src1),
                                              VR128:$src2)))]>;

  def PANDNrm : PDI<0xDF, MRMSrcMem,
                    (outs VR128:$dst), (ins VR128:$src1, i128mem:$src2),
                    "pandn\t{$src2, $dst|$dst, $src2}",
                    [(set VR128:$dst, (v2i64 (and (vnot VR128:$src1),
                                              (memopv2i64 addr:$src2))))]>;
}

// SSE2 Integer comparison
defm PCMPEQB  : PDI_binop_rm_int<0x74, "pcmpeqb", int_x86_sse2_pcmpeq_b>;
defm PCMPEQW  : PDI_binop_rm_int<0x75, "pcmpeqw", int_x86_sse2_pcmpeq_w>;
defm PCMPEQD  : PDI_binop_rm_int<0x76, "pcmpeqd", int_x86_sse2_pcmpeq_d>;
defm PCMPGTB  : PDI_binop_rm_int<0x64, "pcmpgtb", int_x86_sse2_pcmpgt_b>;
defm PCMPGTW  : PDI_binop_rm_int<0x65, "pcmpgtw", int_x86_sse2_pcmpgt_w>;
defm PCMPGTD  : PDI_binop_rm_int<0x66, "pcmpgtd", int_x86_sse2_pcmpgt_d>;

def : Pat<(v16i8 (X86pcmpeqb VR128:$src1, VR128:$src2)),
          (PCMPEQBrr VR128:$src1, VR128:$src2)>;
def : Pat<(v16i8 (X86pcmpeqb VR128:$src1, (memop addr:$src2))),
          (PCMPEQBrm VR128:$src1, addr:$src2)>;
def : Pat<(v8i16 (X86pcmpeqw VR128:$src1, VR128:$src2)),
          (PCMPEQWrr VR128:$src1, VR128:$src2)>;
def : Pat<(v8i16 (X86pcmpeqw VR128:$src1, (memop addr:$src2))),
          (PCMPEQWrm VR128:$src1, addr:$src2)>;
def : Pat<(v4i32 (X86pcmpeqd VR128:$src1, VR128:$src2)),
          (PCMPEQDrr VR128:$src1, VR128:$src2)>;
def : Pat<(v4i32 (X86pcmpeqd VR128:$src1, (memop addr:$src2))),
          (PCMPEQDrm VR128:$src1, addr:$src2)>;

def : Pat<(v16i8 (X86pcmpgtb VR128:$src1, VR128:$src2)),
          (PCMPGTBrr VR128:$src1, VR128:$src2)>;
def : Pat<(v16i8 (X86pcmpgtb VR128:$src1, (memop addr:$src2))),
          (PCMPGTBrm VR128:$src1, addr:$src2)>;
def : Pat<(v8i16 (X86pcmpgtw VR128:$src1, VR128:$src2)),
          (PCMPGTWrr VR128:$src1, VR128:$src2)>;
def : Pat<(v8i16 (X86pcmpgtw VR128:$src1, (memop addr:$src2))),
          (PCMPGTWrm VR128:$src1, addr:$src2)>;
def : Pat<(v4i32 (X86pcmpgtd VR128:$src1, VR128:$src2)),
          (PCMPGTDrr VR128:$src1, VR128:$src2)>;
def : Pat<(v4i32 (X86pcmpgtd VR128:$src1, (memop addr:$src2))),
          (PCMPGTDrm VR128:$src1, addr:$src2)>;


// Pack instructions
defm PACKSSWB : PDI_binop_rm_int<0x63, "packsswb", int_x86_sse2_packsswb_128>;
defm PACKSSDW : PDI_binop_rm_int<0x6B, "packssdw", int_x86_sse2_packssdw_128>;
defm PACKUSWB : PDI_binop_rm_int<0x67, "packuswb", int_x86_sse2_packuswb_128>;

// Shuffle and unpack instructions
def PSHUFDri : PDIi8<0x70, MRMSrcReg,
                     (outs VR128:$dst), (ins VR128:$src1, i8imm:$src2),
                     "pshufd\t{$src2, $src1, $dst|$dst, $src1, $src2}",
                     [(set VR128:$dst, (v4i32 (vector_shuffle
                                               VR128:$src1, (undef),
                                               PSHUFD_shuffle_mask:$src2)))]>;
def PSHUFDmi : PDIi8<0x70, MRMSrcMem,
                     (outs VR128:$dst), (ins i128mem:$src1, i8imm:$src2),
                     "pshufd\t{$src2, $src1, $dst|$dst, $src1, $src2}",
                     [(set VR128:$dst, (v4i32 (vector_shuffle
                                               (bc_v4i32(memopv2i64 addr:$src1)),
                                               (undef),
                                               PSHUFD_shuffle_mask:$src2)))]>;

// SSE2 with ImmT == Imm8 and XS prefix.
def PSHUFHWri : Ii8<0x70, MRMSrcReg,
                    (outs VR128:$dst), (ins VR128:$src1, i8imm:$src2),
                    "pshufhw\t{$src2, $src1, $dst|$dst, $src1, $src2}",
                    [(set VR128:$dst, (v8i16 (vector_shuffle
                                              VR128:$src1, (undef),
                                              PSHUFHW_shuffle_mask:$src2)))]>,
                XS, Requires<[HasSSE2]>;
def PSHUFHWmi : Ii8<0x70, MRMSrcMem,
                    (outs VR128:$dst), (ins i128mem:$src1, i8imm:$src2),
                    "pshufhw\t{$src2, $src1, $dst|$dst, $src1, $src2}",
                    [(set VR128:$dst, (v8i16 (vector_shuffle
                                              (bc_v8i16 (memopv2i64 addr:$src1)),
                                              (undef),
                                              PSHUFHW_shuffle_mask:$src2)))]>,
                XS, Requires<[HasSSE2]>;

// SSE2 with ImmT == Imm8 and XD prefix.
def PSHUFLWri : Ii8<0x70, MRMSrcReg,
                    (outs VR128:$dst), (ins VR128:$src1, i32i8imm:$src2),
                    "pshuflw\t{$src2, $src1, $dst|$dst, $src1, $src2}",
                    [(set VR128:$dst, (v8i16 (vector_shuffle
                                              VR128:$src1, (undef),
                                              PSHUFLW_shuffle_mask:$src2)))]>,
                XD, Requires<[HasSSE2]>;
def PSHUFLWmi : Ii8<0x70, MRMSrcMem,
                    (outs VR128:$dst), (ins i128mem:$src1, i32i8imm:$src2),
                    "pshuflw\t{$src2, $src1, $dst|$dst, $src1, $src2}",
                    [(set VR128:$dst, (v8i16 (vector_shuffle
                                              (bc_v8i16 (memopv2i64 addr:$src1)),
                                              (undef),
                                              PSHUFLW_shuffle_mask:$src2)))]>,
                XD, Requires<[HasSSE2]>;


let Constraints = "$src1 = $dst" in {
  def PUNPCKLBWrr : PDI<0x60, MRMSrcReg, 
                        (outs VR128:$dst), (ins VR128:$src1, VR128:$src2),
                        "punpcklbw\t{$src2, $dst|$dst, $src2}",
                        [(set VR128:$dst,
                          (v16i8 (vector_shuffle VR128:$src1, VR128:$src2,
                                  UNPCKL_shuffle_mask)))]>;
  def PUNPCKLBWrm : PDI<0x60, MRMSrcMem, 
                        (outs VR128:$dst), (ins VR128:$src1, i128mem:$src2),
                        "punpcklbw\t{$src2, $dst|$dst, $src2}",
                        [(set VR128:$dst,
                          (v16i8 (vector_shuffle VR128:$src1,
                                  (bc_v16i8 (memopv2i64 addr:$src2)),
                                  UNPCKL_shuffle_mask)))]>;
  def PUNPCKLWDrr : PDI<0x61, MRMSrcReg, 
                        (outs VR128:$dst), (ins VR128:$src1, VR128:$src2),
                        "punpcklwd\t{$src2, $dst|$dst, $src2}",
                        [(set VR128:$dst,
                          (v8i16 (vector_shuffle VR128:$src1, VR128:$src2,
                                  UNPCKL_shuffle_mask)))]>;
  def PUNPCKLWDrm : PDI<0x61, MRMSrcMem, 
                        (outs VR128:$dst), (ins VR128:$src1, i128mem:$src2),
                        "punpcklwd\t{$src2, $dst|$dst, $src2}",
                        [(set VR128:$dst,
                          (v8i16 (vector_shuffle VR128:$src1,
                                  (bc_v8i16 (memopv2i64 addr:$src2)),
                                  UNPCKL_shuffle_mask)))]>;
  def PUNPCKLDQrr : PDI<0x62, MRMSrcReg, 
                        (outs VR128:$dst), (ins VR128:$src1, VR128:$src2),
                        "punpckldq\t{$src2, $dst|$dst, $src2}",
                        [(set VR128:$dst,
                          (v4i32 (vector_shuffle VR128:$src1, VR128:$src2,
                                  UNPCKL_shuffle_mask)))]>;
  def PUNPCKLDQrm : PDI<0x62, MRMSrcMem, 
                        (outs VR128:$dst), (ins VR128:$src1, i128mem:$src2),
                        "punpckldq\t{$src2, $dst|$dst, $src2}",
                        [(set VR128:$dst,
                          (v4i32 (vector_shuffle VR128:$src1,
                                  (bc_v4i32 (memopv2i64 addr:$src2)),
                                  UNPCKL_shuffle_mask)))]>;
  def PUNPCKLQDQrr : PDI<0x6C, MRMSrcReg, 
                         (outs VR128:$dst), (ins VR128:$src1, VR128:$src2),
                         "punpcklqdq\t{$src2, $dst|$dst, $src2}",
                        [(set VR128:$dst,
                          (v2i64 (vector_shuffle VR128:$src1, VR128:$src2,
                                  UNPCKL_shuffle_mask)))]>;
  def PUNPCKLQDQrm : PDI<0x6C, MRMSrcMem, 
                         (outs VR128:$dst), (ins VR128:$src1, i128mem:$src2),
                         "punpcklqdq\t{$src2, $dst|$dst, $src2}",
                        [(set VR128:$dst,
                          (v2i64 (vector_shuffle VR128:$src1,
                                  (memopv2i64 addr:$src2),
                                  UNPCKL_shuffle_mask)))]>;
  
  def PUNPCKHBWrr : PDI<0x68, MRMSrcReg, 
                        (outs VR128:$dst), (ins VR128:$src1, VR128:$src2),
                        "punpckhbw\t{$src2, $dst|$dst, $src2}",
                        [(set VR128:$dst,
                          (v16i8 (vector_shuffle VR128:$src1, VR128:$src2,
                                  UNPCKH_shuffle_mask)))]>;
  def PUNPCKHBWrm : PDI<0x68, MRMSrcMem, 
                        (outs VR128:$dst), (ins VR128:$src1, i128mem:$src2),
                        "punpckhbw\t{$src2, $dst|$dst, $src2}",
                        [(set VR128:$dst,
                          (v16i8 (vector_shuffle VR128:$src1,
                                  (bc_v16i8 (memopv2i64 addr:$src2)),
                                  UNPCKH_shuffle_mask)))]>;
  def PUNPCKHWDrr : PDI<0x69, MRMSrcReg, 
                        (outs VR128:$dst), (ins VR128:$src1, VR128:$src2),
                        "punpckhwd\t{$src2, $dst|$dst, $src2}",
                        [(set VR128:$dst,
                          (v8i16 (vector_shuffle VR128:$src1, VR128:$src2,
                                  UNPCKH_shuffle_mask)))]>;
  def PUNPCKHWDrm : PDI<0x69, MRMSrcMem, 
                        (outs VR128:$dst), (ins VR128:$src1, i128mem:$src2),
                        "punpckhwd\t{$src2, $dst|$dst, $src2}",
                        [(set VR128:$dst,
                          (v8i16 (vector_shuffle VR128:$src1,
                                  (bc_v8i16 (memopv2i64 addr:$src2)),
                                  UNPCKH_shuffle_mask)))]>;
  def PUNPCKHDQrr : PDI<0x6A, MRMSrcReg, 
                        (outs VR128:$dst), (ins VR128:$src1, VR128:$src2),
                        "punpckhdq\t{$src2, $dst|$dst, $src2}",
                        [(set VR128:$dst,
                          (v4i32 (vector_shuffle VR128:$src1, VR128:$src2,
                                  UNPCKH_shuffle_mask)))]>;
  def PUNPCKHDQrm : PDI<0x6A, MRMSrcMem, 
                        (outs VR128:$dst), (ins VR128:$src1, i128mem:$src2),
                        "punpckhdq\t{$src2, $dst|$dst, $src2}",
                        [(set VR128:$dst,
                          (v4i32 (vector_shuffle VR128:$src1,
                                  (bc_v4i32 (memopv2i64 addr:$src2)),
                                  UNPCKH_shuffle_mask)))]>;
  def PUNPCKHQDQrr : PDI<0x6D, MRMSrcReg, 
                         (outs VR128:$dst), (ins VR128:$src1, VR128:$src2),
                         "punpckhqdq\t{$src2, $dst|$dst, $src2}",
                        [(set VR128:$dst,
                          (v2i64 (vector_shuffle VR128:$src1, VR128:$src2,
                                  UNPCKH_shuffle_mask)))]>;
  def PUNPCKHQDQrm : PDI<0x6D, MRMSrcMem, 
                        (outs VR128:$dst), (ins VR128:$src1, i128mem:$src2),
                        "punpckhqdq\t{$src2, $dst|$dst, $src2}",
                        [(set VR128:$dst,
                          (v2i64 (vector_shuffle VR128:$src1,
                                  (memopv2i64 addr:$src2),
                                  UNPCKH_shuffle_mask)))]>;
}

// Extract / Insert
def PEXTRWri : PDIi8<0xC5, MRMSrcReg,
                    (outs GR32:$dst), (ins VR128:$src1, i32i8imm:$src2),
                    "pextrw\t{$src2, $src1, $dst|$dst, $src1, $src2}",
                    [(set GR32:$dst, (X86pextrw (v8i16 VR128:$src1),
                                                imm:$src2))]>;
let Constraints = "$src1 = $dst" in {
  def PINSRWrri : PDIi8<0xC4, MRMSrcReg,
                       (outs VR128:$dst), (ins VR128:$src1,
                        GR32:$src2, i32i8imm:$src3),
                       "pinsrw\t{$src3, $src2, $dst|$dst, $src2, $src3}",
                       [(set VR128:$dst,
                         (X86pinsrw VR128:$src1, GR32:$src2, imm:$src3))]>;
  def PINSRWrmi : PDIi8<0xC4, MRMSrcMem,
                       (outs VR128:$dst), (ins VR128:$src1,
                        i16mem:$src2, i32i8imm:$src3),
                       "pinsrw\t{$src3, $src2, $dst|$dst, $src2, $src3}",
                       [(set VR128:$dst, 
                         (X86pinsrw VR128:$src1, (extloadi16 addr:$src2),
                                    imm:$src3))]>;
}

// Mask creation
def PMOVMSKBrr : PDI<0xD7, MRMSrcReg, (outs GR32:$dst), (ins VR128:$src),
                     "pmovmskb\t{$src, $dst|$dst, $src}",
                     [(set GR32:$dst, (int_x86_sse2_pmovmskb_128 VR128:$src))]>;

// Conditional store
let Uses = [EDI] in
def MASKMOVDQU : PDI<0xF7, MRMSrcReg, (outs), (ins VR128:$src, VR128:$mask),
                     "maskmovdqu\t{$mask, $src|$src, $mask}",
                     [(int_x86_sse2_maskmov_dqu VR128:$src, VR128:$mask, EDI)]>;

// Non-temporal stores
def MOVNTPDmr : PDI<0x2B, MRMDestMem, (outs), (ins i128mem:$dst, VR128:$src),
                    "movntpd\t{$src, $dst|$dst, $src}",
                    [(int_x86_sse2_movnt_pd addr:$dst, VR128:$src)]>;
def MOVNTDQmr : PDI<0xE7, MRMDestMem, (outs), (ins f128mem:$dst, VR128:$src),
                    "movntdq\t{$src, $dst|$dst, $src}",
                    [(int_x86_sse2_movnt_dq addr:$dst, VR128:$src)]>;
def MOVNTImr  :   I<0xC3, MRMDestMem, (outs), (ins i32mem:$dst, GR32:$src),
                    "movnti\t{$src, $dst|$dst, $src}",
                    [(int_x86_sse2_movnt_i addr:$dst, GR32:$src)]>, 
                  TB, Requires<[HasSSE2]>;

// Flush cache
def CLFLUSH : I<0xAE, MRM7m, (outs), (ins i8mem:$src),
               "clflush\t$src", [(int_x86_sse2_clflush addr:$src)]>,
              TB, Requires<[HasSSE2]>;

// Load, store, and memory fence
def LFENCE : I<0xAE, MRM5r, (outs), (ins),
               "lfence", [(int_x86_sse2_lfence)]>, TB, Requires<[HasSSE2]>;
def MFENCE : I<0xAE, MRM6r, (outs), (ins),
               "mfence", [(int_x86_sse2_mfence)]>, TB, Requires<[HasSSE2]>;

//TODO: custom lower this so as to never even generate the noop
def : Pat<(membarrier (i8 imm:$ll), (i8 imm:$ls), (i8 imm:$sl), (i8 imm:$ss), 
           (i8 0)), (NOOP)>;
def : Pat<(membarrier (i8 0), (i8 0), (i8 0), (i8 1), (i8 1)), (SFENCE)>;
def : Pat<(membarrier (i8 1), (i8 0), (i8 0), (i8 0), (i8 1)), (LFENCE)>;
def : Pat<(membarrier (i8 imm:$ll), (i8 imm:$ls), (i8 imm:$sl), (i8 imm:$ss), 
           (i8 1)), (MFENCE)>;

// Alias instructions that map zero vector to pxor / xorp* for sse.
// We set canFoldAsLoad because this can be converted to a constant-pool
// load of an all-ones value if folding it would be beneficial.
let isReMaterializable = 1, isAsCheapAsAMove = 1, canFoldAsLoad = 1 in
  def V_SETALLONES : PDI<0x76, MRMInitReg, (outs VR128:$dst), (ins),
                         "pcmpeqd\t$dst, $dst",
                         [(set VR128:$dst, (v4i32 immAllOnesV))]>;

// FR64 to 128-bit vector conversion.
def MOVSD2PDrr : SDI<0x10, MRMSrcReg, (outs VR128:$dst), (ins FR64:$src),
                      "movsd\t{$src, $dst|$dst, $src}",
                      [(set VR128:$dst,
                        (v2f64 (scalar_to_vector FR64:$src)))]>;
def MOVSD2PDrm : SDI<0x10, MRMSrcMem, (outs VR128:$dst), (ins f64mem:$src),
                     "movsd\t{$src, $dst|$dst, $src}",
                     [(set VR128:$dst, 
                       (v2f64 (scalar_to_vector (loadf64 addr:$src))))]>;

def MOVDI2PDIrr : PDI<0x6E, MRMSrcReg, (outs VR128:$dst), (ins GR32:$src),
                      "movd\t{$src, $dst|$dst, $src}",
                      [(set VR128:$dst,
                        (v4i32 (scalar_to_vector GR32:$src)))]>;
def MOVDI2PDIrm : PDI<0x6E, MRMSrcMem, (outs VR128:$dst), (ins i32mem:$src),
                      "movd\t{$src, $dst|$dst, $src}",
                      [(set VR128:$dst,
                        (v4i32 (scalar_to_vector (loadi32 addr:$src))))]>;

def MOVDI2SSrr  : PDI<0x6E, MRMSrcReg, (outs FR32:$dst), (ins GR32:$src),
                      "movd\t{$src, $dst|$dst, $src}",
                      [(set FR32:$dst, (bitconvert GR32:$src))]>;

def MOVDI2SSrm  : PDI<0x6E, MRMSrcMem, (outs FR32:$dst), (ins i32mem:$src),
                      "movd\t{$src, $dst|$dst, $src}",
                      [(set FR32:$dst, (bitconvert (loadi32 addr:$src)))]>;

// SSE2 instructions with XS prefix
def MOVQI2PQIrm : I<0x7E, MRMSrcMem, (outs VR128:$dst), (ins i64mem:$src),
                    "movq\t{$src, $dst|$dst, $src}",
                    [(set VR128:$dst,
                      (v2i64 (scalar_to_vector (loadi64 addr:$src))))]>, XS,
                  Requires<[HasSSE2]>;
def MOVPQI2QImr : PDI<0xD6, MRMDestMem, (outs), (ins i64mem:$dst, VR128:$src),
                      "movq\t{$src, $dst|$dst, $src}",
                      [(store (i64 (vector_extract (v2i64 VR128:$src),
                                    (iPTR 0))), addr:$dst)]>;

// FIXME: may not be able to eliminate this movss with coalescing the src and
// dest register classes are different. We really want to write this pattern
// like this:
// def : Pat<(f32 (vector_extract (v4f32 VR128:$src), (iPTR 0))),
//           (f32 FR32:$src)>;
def MOVPD2SDrr : SDI<0x10, MRMSrcReg, (outs FR64:$dst), (ins VR128:$src),
                     "movsd\t{$src, $dst|$dst, $src}",
                     [(set FR64:$dst, (vector_extract (v2f64 VR128:$src),
                                       (iPTR 0)))]>;
def MOVPD2SDmr : SDI<0x11, MRMDestMem, (outs), (ins f64mem:$dst, VR128:$src),
                     "movsd\t{$src, $dst|$dst, $src}",
                     [(store (f64 (vector_extract (v2f64 VR128:$src),
                                   (iPTR 0))), addr:$dst)]>;
def MOVPDI2DIrr  : PDI<0x7E, MRMDestReg, (outs GR32:$dst), (ins VR128:$src),
                       "movd\t{$src, $dst|$dst, $src}",
                       [(set GR32:$dst, (vector_extract (v4i32 VR128:$src),
                                        (iPTR 0)))]>;
def MOVPDI2DImr  : PDI<0x7E, MRMDestMem, (outs), (ins i32mem:$dst, VR128:$src),
                       "movd\t{$src, $dst|$dst, $src}",
                       [(store (i32 (vector_extract (v4i32 VR128:$src),
                                     (iPTR 0))), addr:$dst)]>;

def MOVSS2DIrr  : PDI<0x7E, MRMDestReg, (outs GR32:$dst), (ins FR32:$src),
                      "movd\t{$src, $dst|$dst, $src}",
                      [(set GR32:$dst, (bitconvert FR32:$src))]>;
def MOVSS2DImr  : PDI<0x7E, MRMDestMem, (outs), (ins i32mem:$dst, FR32:$src),
                      "movd\t{$src, $dst|$dst, $src}",
                      [(store (i32 (bitconvert FR32:$src)), addr:$dst)]>;


// Move to lower bits of a VR128, leaving upper bits alone.
// Three operand (but two address) aliases.
let Constraints = "$src1 = $dst" in {
  let neverHasSideEffects = 1 in
  def MOVLSD2PDrr : SDI<0x10, MRMSrcReg,
                        (outs VR128:$dst), (ins VR128:$src1, FR64:$src2),
                        "movsd\t{$src2, $dst|$dst, $src2}", []>;

  let AddedComplexity = 15 in
    def MOVLPDrr : SDI<0x10, MRMSrcReg,
                       (outs VR128:$dst), (ins VR128:$src1, VR128:$src2),
                       "movsd\t{$src2, $dst|$dst, $src2}",
                       [(set VR128:$dst,
                         (v2f64 (vector_shuffle VR128:$src1, VR128:$src2,
                                 MOVL_shuffle_mask)))]>;
}

// Store / copy lower 64-bits of a XMM register.
def MOVLQ128mr : PDI<0xD6, MRMDestMem, (outs), (ins i64mem:$dst, VR128:$src),
                     "movq\t{$src, $dst|$dst, $src}",
                     [(int_x86_sse2_storel_dq addr:$dst, VR128:$src)]>;

// Move to lower bits of a VR128 and zeroing upper bits.
// Loading from memory automatically zeroing upper bits.
let AddedComplexity = 20 in {
def MOVZSD2PDrm : SDI<0x10, MRMSrcMem, (outs VR128:$dst), (ins f64mem:$src),
                      "movsd\t{$src, $dst|$dst, $src}",
                      [(set VR128:$dst,
                        (v2f64 (X86vzmovl (v2f64 (scalar_to_vector
                                                 (loadf64 addr:$src))))))]>;

def : Pat<(v2f64 (X86vzmovl (loadv2f64 addr:$src))),
            (MOVZSD2PDrm addr:$src)>;
def : Pat<(v2f64 (X86vzmovl (bc_v2f64 (loadv4f32 addr:$src)))),
            (MOVZSD2PDrm addr:$src)>;
def : Pat<(v2f64 (X86vzload addr:$src)), (MOVZSD2PDrm addr:$src)>;
}

// movd / movq to XMM register zero-extends
let AddedComplexity = 15 in {
def MOVZDI2PDIrr : PDI<0x6E, MRMSrcReg, (outs VR128:$dst), (ins GR32:$src),
                       "movd\t{$src, $dst|$dst, $src}",
                       [(set VR128:$dst, (v4i32 (X86vzmovl
                                      (v4i32 (scalar_to_vector GR32:$src)))))]>;
// This is X86-64 only.
def MOVZQI2PQIrr : RPDI<0x6E, MRMSrcReg, (outs VR128:$dst), (ins GR64:$src),
                       "mov{d|q}\t{$src, $dst|$dst, $src}",
                       [(set VR128:$dst, (v2i64 (X86vzmovl
                                      (v2i64 (scalar_to_vector GR64:$src)))))]>;
}

let AddedComplexity = 20 in {
def MOVZDI2PDIrm : PDI<0x6E, MRMSrcMem, (outs VR128:$dst), (ins i32mem:$src),
                       "movd\t{$src, $dst|$dst, $src}",
                       [(set VR128:$dst,
                         (v4i32 (X86vzmovl (v4i32 (scalar_to_vector
                                                   (loadi32 addr:$src))))))]>;

def : Pat<(v4i32 (X86vzmovl (loadv4i32 addr:$src))),
            (MOVZDI2PDIrm addr:$src)>;
def : Pat<(v4i32 (X86vzmovl (bc_v4i32 (loadv4f32 addr:$src)))),
            (MOVZDI2PDIrm addr:$src)>;
def : Pat<(v4i32 (X86vzmovl (bc_v4i32 (loadv2i64 addr:$src)))),
            (MOVZDI2PDIrm addr:$src)>;

def MOVZQI2PQIrm : I<0x7E, MRMSrcMem, (outs VR128:$dst), (ins i64mem:$src),
                     "movq\t{$src, $dst|$dst, $src}",
                     [(set VR128:$dst,
                       (v2i64 (X86vzmovl (v2i64 (scalar_to_vector
                                                 (loadi64 addr:$src))))))]>, XS,
                   Requires<[HasSSE2]>;

def : Pat<(v2i64 (X86vzmovl (loadv2i64 addr:$src))),
            (MOVZQI2PQIrm addr:$src)>;
def : Pat<(v2i64 (X86vzmovl (bc_v2i64 (loadv4f32 addr:$src)))),
            (MOVZQI2PQIrm addr:$src)>;
def : Pat<(v2i64 (X86vzload addr:$src)), (MOVZQI2PQIrm addr:$src)>;
}

// Moving from XMM to XMM and clear upper 64 bits. Note, there is a bug in
// IA32 document. movq xmm1, xmm2 does clear the high bits.
let AddedComplexity = 15 in
def MOVZPQILo2PQIrr : I<0x7E, MRMSrcReg, (outs VR128:$dst), (ins VR128:$src),
                        "movq\t{$src, $dst|$dst, $src}",
                    [(set VR128:$dst, (v2i64 (X86vzmovl (v2i64 VR128:$src))))]>,
                      XS, Requires<[HasSSE2]>;

let AddedComplexity = 20 in {
def MOVZPQILo2PQIrm : I<0x7E, MRMSrcMem, (outs VR128:$dst), (ins i128mem:$src),
                        "movq\t{$src, $dst|$dst, $src}",
                    [(set VR128:$dst, (v2i64 (X86vzmovl
                                             (loadv2i64 addr:$src))))]>,
                      XS, Requires<[HasSSE2]>;

def : Pat<(v2i64 (X86vzmovl (bc_v2i64 (loadv4i32 addr:$src)))),
            (MOVZPQILo2PQIrm addr:$src)>;
}

//===----------------------------------------------------------------------===//
// SSE3 Instructions
//===----------------------------------------------------------------------===//

// Move Instructions
def MOVSHDUPrr : S3SI<0x16, MRMSrcReg, (outs VR128:$dst), (ins VR128:$src),
                      "movshdup\t{$src, $dst|$dst, $src}",
                      [(set VR128:$dst, (v4f32 (vector_shuffle
                                                VR128:$src, (undef),
                                                MOVSHDUP_shuffle_mask)))]>;
def MOVSHDUPrm : S3SI<0x16, MRMSrcMem, (outs VR128:$dst), (ins f128mem:$src),
                      "movshdup\t{$src, $dst|$dst, $src}",
                      [(set VR128:$dst, (v4f32 (vector_shuffle
                                                (memopv4f32 addr:$src), (undef),
                                                MOVSHDUP_shuffle_mask)))]>;

def MOVSLDUPrr : S3SI<0x12, MRMSrcReg, (outs VR128:$dst), (ins VR128:$src),
                      "movsldup\t{$src, $dst|$dst, $src}",
                      [(set VR128:$dst, (v4f32 (vector_shuffle
                                                VR128:$src, (undef),
                                                MOVSLDUP_shuffle_mask)))]>;
def MOVSLDUPrm : S3SI<0x12, MRMSrcMem, (outs VR128:$dst), (ins f128mem:$src),
                      "movsldup\t{$src, $dst|$dst, $src}",
                      [(set VR128:$dst, (v4f32 (vector_shuffle
                                                (memopv4f32 addr:$src), (undef),
                                                MOVSLDUP_shuffle_mask)))]>;

def MOVDDUPrr  : S3DI<0x12, MRMSrcReg, (outs VR128:$dst), (ins VR128:$src),
                      "movddup\t{$src, $dst|$dst, $src}",
                      [(set VR128:$dst,
                         (v2f64 (vector_shuffle VR128:$src, (undef),
                                                MOVDDUP_shuffle_mask)))]>;
def MOVDDUPrm  : S3DI<0x12, MRMSrcMem, (outs VR128:$dst), (ins f64mem:$src),
                      "movddup\t{$src, $dst|$dst, $src}",
                    [(set VR128:$dst,
                      (v2f64 (vector_shuffle
                              (scalar_to_vector (loadf64 addr:$src)),
                              (undef), MOVDDUP_shuffle_mask)))]>;

def : Pat<(vector_shuffle
             (bc_v2f64 (v2i64 (scalar_to_vector (loadi64 addr:$src)))),
             (undef), MOVDDUP_shuffle_mask),
          (MOVDDUPrm addr:$src)>, Requires<[HasSSE3]>;
def : Pat<(vector_shuffle
            (memopv2f64 addr:$src), (undef), MOVDDUP_shuffle_mask),
          (MOVDDUPrm addr:$src)>, Requires<[HasSSE3]>;


// Arithmetic
let Constraints = "$src1 = $dst" in {
  def ADDSUBPSrr : S3DI<0xD0, MRMSrcReg,
                        (outs VR128:$dst), (ins VR128:$src1, VR128:$src2),
                        "addsubps\t{$src2, $dst|$dst, $src2}",
                        [(set VR128:$dst, (int_x86_sse3_addsub_ps VR128:$src1,
                                           VR128:$src2))]>;
  def ADDSUBPSrm : S3DI<0xD0, MRMSrcMem,
                        (outs VR128:$dst), (ins VR128:$src1, f128mem:$src2),
                        "addsubps\t{$src2, $dst|$dst, $src2}",
                        [(set VR128:$dst, (int_x86_sse3_addsub_ps VR128:$src1,
                                           (memop addr:$src2)))]>;
  def ADDSUBPDrr : S3I<0xD0, MRMSrcReg,
                       (outs VR128:$dst), (ins VR128:$src1, VR128:$src2),
                       "addsubpd\t{$src2, $dst|$dst, $src2}",
                       [(set VR128:$dst, (int_x86_sse3_addsub_pd VR128:$src1,
                                          VR128:$src2))]>;
  def ADDSUBPDrm : S3I<0xD0, MRMSrcMem,
                       (outs VR128:$dst), (ins VR128:$src1, f128mem:$src2),
                       "addsubpd\t{$src2, $dst|$dst, $src2}",
                       [(set VR128:$dst, (int_x86_sse3_addsub_pd VR128:$src1,
                                          (memop addr:$src2)))]>;
}

def LDDQUrm : S3DI<0xF0, MRMSrcMem, (outs VR128:$dst), (ins i128mem:$src),
                   "lddqu\t{$src, $dst|$dst, $src}",
                   [(set VR128:$dst, (int_x86_sse3_ldu_dq addr:$src))]>;

// Horizontal ops
class S3D_Intrr<bits<8> o, string OpcodeStr, Intrinsic IntId>
  : S3DI<o, MRMSrcReg, (outs VR128:$dst), (ins VR128:$src1, VR128:$src2),
         !strconcat(OpcodeStr, "\t{$src2, $dst|$dst, $src2}"),
         [(set VR128:$dst, (v4f32 (IntId VR128:$src1, VR128:$src2)))]>;
class S3D_Intrm<bits<8> o, string OpcodeStr, Intrinsic IntId>
  : S3DI<o, MRMSrcMem, (outs VR128:$dst), (ins VR128:$src1, f128mem:$src2),
         !strconcat(OpcodeStr, "\t{$src2, $dst|$dst, $src2}"),
         [(set VR128:$dst, (v4f32 (IntId VR128:$src1, (memop addr:$src2))))]>;
class S3_Intrr<bits<8> o, string OpcodeStr, Intrinsic IntId>
  : S3I<o, MRMSrcReg, (outs VR128:$dst), (ins VR128:$src1, VR128:$src2),
        !strconcat(OpcodeStr, "\t{$src2, $dst|$dst, $src2}"),
        [(set VR128:$dst, (v2f64 (IntId VR128:$src1, VR128:$src2)))]>;
class S3_Intrm<bits<8> o, string OpcodeStr, Intrinsic IntId>
  : S3I<o, MRMSrcMem, (outs VR128:$dst), (ins VR128:$src1, f128mem:$src2),
        !strconcat(OpcodeStr, "\t{$src2, $dst|$dst, $src2}"),
      [(set VR128:$dst, (v2f64 (IntId VR128:$src1, (memopv2f64 addr:$src2))))]>;

let Constraints = "$src1 = $dst" in {
  def HADDPSrr : S3D_Intrr<0x7C, "haddps", int_x86_sse3_hadd_ps>;
  def HADDPSrm : S3D_Intrm<0x7C, "haddps", int_x86_sse3_hadd_ps>;
  def HADDPDrr : S3_Intrr <0x7C, "haddpd", int_x86_sse3_hadd_pd>;
  def HADDPDrm : S3_Intrm <0x7C, "haddpd", int_x86_sse3_hadd_pd>;
  def HSUBPSrr : S3D_Intrr<0x7D, "hsubps", int_x86_sse3_hsub_ps>;
  def HSUBPSrm : S3D_Intrm<0x7D, "hsubps", int_x86_sse3_hsub_ps>;
  def HSUBPDrr : S3_Intrr <0x7D, "hsubpd", int_x86_sse3_hsub_pd>;
  def HSUBPDrm : S3_Intrm <0x7D, "hsubpd", int_x86_sse3_hsub_pd>;
}

// Thread synchronization
def MONITOR : I<0xC8, RawFrm, (outs), (ins), "monitor",
                [(int_x86_sse3_monitor EAX, ECX, EDX)]>,TB, Requires<[HasSSE3]>;
def MWAIT   : I<0xC9, RawFrm, (outs), (ins), "mwait",
                [(int_x86_sse3_mwait ECX, EAX)]>, TB, Requires<[HasSSE3]>;

// vector_shuffle v1, <undef> <1, 1, 3, 3>
let AddedComplexity = 15 in
def : Pat<(v4i32 (vector_shuffle VR128:$src, (undef),
                  MOVSHDUP_shuffle_mask)),
          (MOVSHDUPrr VR128:$src)>, Requires<[HasSSE3]>;
let AddedComplexity = 20 in
def : Pat<(v4i32 (vector_shuffle (bc_v4i32 (memopv2i64 addr:$src)), (undef),
                  MOVSHDUP_shuffle_mask)),
          (MOVSHDUPrm addr:$src)>, Requires<[HasSSE3]>;

// vector_shuffle v1, <undef> <0, 0, 2, 2>
let AddedComplexity = 15 in
  def : Pat<(v4i32 (vector_shuffle VR128:$src, (undef),
                    MOVSLDUP_shuffle_mask)),
            (MOVSLDUPrr VR128:$src)>, Requires<[HasSSE3]>;
let AddedComplexity = 20 in
  def : Pat<(v4i32 (vector_shuffle (bc_v4i32 (memopv2i64 addr:$src)), (undef),
                    MOVSLDUP_shuffle_mask)),
            (MOVSLDUPrm addr:$src)>, Requires<[HasSSE3]>;

//===----------------------------------------------------------------------===//
// SSSE3 Instructions
//===----------------------------------------------------------------------===//

/// SS3I_unop_rm_int_8 - Simple SSSE3 unary operator whose type is v*i8.
multiclass SS3I_unop_rm_int_8<bits<8> opc, string OpcodeStr,
                              Intrinsic IntId64, Intrinsic IntId128> {
  def rr64 : SS38I<opc, MRMSrcReg, (outs VR64:$dst), (ins VR64:$src),
                   !strconcat(OpcodeStr, "\t{$src, $dst|$dst, $src}"),
                   [(set VR64:$dst, (IntId64 VR64:$src))]>;

  def rm64 : SS38I<opc, MRMSrcMem, (outs VR64:$dst), (ins i64mem:$src),
                   !strconcat(OpcodeStr, "\t{$src, $dst|$dst, $src}"),
                   [(set VR64:$dst,
                     (IntId64 (bitconvert (memopv8i8 addr:$src))))]>;

  def rr128 : SS38I<opc, MRMSrcReg, (outs VR128:$dst),
                    (ins VR128:$src),
                    !strconcat(OpcodeStr, "\t{$src, $dst|$dst, $src}"),
                    [(set VR128:$dst, (IntId128 VR128:$src))]>,
                    OpSize;

  def rm128 : SS38I<opc, MRMSrcMem, (outs VR128:$dst),
                    (ins i128mem:$src),
                    !strconcat(OpcodeStr, "\t{$src, $dst|$dst, $src}"),
                    [(set VR128:$dst,
                      (IntId128
                       (bitconvert (memopv16i8 addr:$src))))]>, OpSize;
}

/// SS3I_unop_rm_int_16 - Simple SSSE3 unary operator whose type is v*i16.
multiclass SS3I_unop_rm_int_16<bits<8> opc, string OpcodeStr,
                               Intrinsic IntId64, Intrinsic IntId128> {
  def rr64 : SS38I<opc, MRMSrcReg, (outs VR64:$dst),
                   (ins VR64:$src),
                   !strconcat(OpcodeStr, "\t{$src, $dst|$dst, $src}"),
                   [(set VR64:$dst, (IntId64 VR64:$src))]>;

  def rm64 : SS38I<opc, MRMSrcMem, (outs VR64:$dst),
                   (ins i64mem:$src),
                   !strconcat(OpcodeStr, "\t{$src, $dst|$dst, $src}"),
                   [(set VR64:$dst,
                     (IntId64
                      (bitconvert (memopv4i16 addr:$src))))]>;

  def rr128 : SS38I<opc, MRMSrcReg, (outs VR128:$dst),
                    (ins VR128:$src),
                    !strconcat(OpcodeStr, "\t{$src, $dst|$dst, $src}"),
                    [(set VR128:$dst, (IntId128 VR128:$src))]>,
                    OpSize;

  def rm128 : SS38I<opc, MRMSrcMem, (outs VR128:$dst),
                    (ins i128mem:$src),
                    !strconcat(OpcodeStr, "\t{$src, $dst|$dst, $src}"),
                    [(set VR128:$dst,
                      (IntId128
                       (bitconvert (memopv8i16 addr:$src))))]>, OpSize;
}

/// SS3I_unop_rm_int_32 - Simple SSSE3 unary operator whose type is v*i32.
multiclass SS3I_unop_rm_int_32<bits<8> opc, string OpcodeStr,
                               Intrinsic IntId64, Intrinsic IntId128> {
  def rr64 : SS38I<opc, MRMSrcReg, (outs VR64:$dst),
                   (ins VR64:$src),
                   !strconcat(OpcodeStr, "\t{$src, $dst|$dst, $src}"),
                   [(set VR64:$dst, (IntId64 VR64:$src))]>;

  def rm64 : SS38I<opc, MRMSrcMem, (outs VR64:$dst),
                   (ins i64mem:$src),
                   !strconcat(OpcodeStr, "\t{$src, $dst|$dst, $src}"),
                   [(set VR64:$dst,
                     (IntId64
                      (bitconvert (memopv2i32 addr:$src))))]>;

  def rr128 : SS38I<opc, MRMSrcReg, (outs VR128:$dst),
                    (ins VR128:$src),
                    !strconcat(OpcodeStr, "\t{$src, $dst|$dst, $src}"),
                    [(set VR128:$dst, (IntId128 VR128:$src))]>,
                    OpSize;

  def rm128 : SS38I<opc, MRMSrcMem, (outs VR128:$dst),
                    (ins i128mem:$src),
                    !strconcat(OpcodeStr, "\t{$src, $dst|$dst, $src}"),
                    [(set VR128:$dst,
                      (IntId128
                       (bitconvert (memopv4i32 addr:$src))))]>, OpSize;
}

defm PABSB       : SS3I_unop_rm_int_8 <0x1C, "pabsb",
                                       int_x86_ssse3_pabs_b,
                                       int_x86_ssse3_pabs_b_128>;
defm PABSW       : SS3I_unop_rm_int_16<0x1D, "pabsw",
                                       int_x86_ssse3_pabs_w,
                                       int_x86_ssse3_pabs_w_128>;
defm PABSD       : SS3I_unop_rm_int_32<0x1E, "pabsd",
                                       int_x86_ssse3_pabs_d,
                                       int_x86_ssse3_pabs_d_128>;

/// SS3I_binop_rm_int_8 - Simple SSSE3 binary operator whose type is v*i8.
let Constraints = "$src1 = $dst" in {
  multiclass SS3I_binop_rm_int_8<bits<8> opc, string OpcodeStr,
                                 Intrinsic IntId64, Intrinsic IntId128,
                                 bit Commutable = 0> {
    def rr64 : SS38I<opc, MRMSrcReg, (outs VR64:$dst),
                     (ins VR64:$src1, VR64:$src2),
                     !strconcat(OpcodeStr, "\t{$src2, $dst|$dst, $src2}"),
                     [(set VR64:$dst, (IntId64 VR64:$src1, VR64:$src2))]> {
      let isCommutable = Commutable;
    }
    def rm64 : SS38I<opc, MRMSrcMem, (outs VR64:$dst),
                     (ins VR64:$src1, i64mem:$src2),
                     !strconcat(OpcodeStr, "\t{$src2, $dst|$dst, $src2}"),
                     [(set VR64:$dst,
                       (IntId64 VR64:$src1,
                        (bitconvert (memopv8i8 addr:$src2))))]>;

    def rr128 : SS38I<opc, MRMSrcReg, (outs VR128:$dst),
                      (ins VR128:$src1, VR128:$src2),
                      !strconcat(OpcodeStr, "\t{$src2, $dst|$dst, $src2}"),
                      [(set VR128:$dst, (IntId128 VR128:$src1, VR128:$src2))]>,
                      OpSize {
      let isCommutable = Commutable;
    }
    def rm128 : SS38I<opc, MRMSrcMem, (outs VR128:$dst),
                      (ins VR128:$src1, i128mem:$src2),
                      !strconcat(OpcodeStr, "\t{$src2, $dst|$dst, $src2}"),
                      [(set VR128:$dst,
                        (IntId128 VR128:$src1,
                         (bitconvert (memopv16i8 addr:$src2))))]>, OpSize;
  }
}

/// SS3I_binop_rm_int_16 - Simple SSSE3 binary operator whose type is v*i16.
let Constraints = "$src1 = $dst" in {
  multiclass SS3I_binop_rm_int_16<bits<8> opc, string OpcodeStr,
                                  Intrinsic IntId64, Intrinsic IntId128,
                                  bit Commutable = 0> {
    def rr64 : SS38I<opc, MRMSrcReg, (outs VR64:$dst),
                     (ins VR64:$src1, VR64:$src2),
                     !strconcat(OpcodeStr, "\t{$src2, $dst|$dst, $src2}"),
                     [(set VR64:$dst, (IntId64 VR64:$src1, VR64:$src2))]> {
      let isCommutable = Commutable;
    }
    def rm64 : SS38I<opc, MRMSrcMem, (outs VR64:$dst),
                     (ins VR64:$src1, i64mem:$src2),
                     !strconcat(OpcodeStr, "\t{$src2, $dst|$dst, $src2}"),
                     [(set VR64:$dst,
                       (IntId64 VR64:$src1,
                        (bitconvert (memopv4i16 addr:$src2))))]>;

    def rr128 : SS38I<opc, MRMSrcReg, (outs VR128:$dst),
                      (ins VR128:$src1, VR128:$src2),
                      !strconcat(OpcodeStr, "\t{$src2, $dst|$dst, $src2}"),
                      [(set VR128:$dst, (IntId128 VR128:$src1, VR128:$src2))]>,
                      OpSize {
      let isCommutable = Commutable;
    }
    def rm128 : SS38I<opc, MRMSrcMem, (outs VR128:$dst),
                      (ins VR128:$src1, i128mem:$src2),
                      !strconcat(OpcodeStr, "\t{$src2, $dst|$dst, $src2}"),
                      [(set VR128:$dst,
                        (IntId128 VR128:$src1,
                         (bitconvert (memopv8i16 addr:$src2))))]>, OpSize;
  }
}

/// SS3I_binop_rm_int_32 - Simple SSSE3 binary operator whose type is v*i32.
let Constraints = "$src1 = $dst" in {
  multiclass SS3I_binop_rm_int_32<bits<8> opc, string OpcodeStr,
                                  Intrinsic IntId64, Intrinsic IntId128,
                                  bit Commutable = 0> {
    def rr64 : SS38I<opc, MRMSrcReg, (outs VR64:$dst),
                     (ins VR64:$src1, VR64:$src2),
                     !strconcat(OpcodeStr, "\t{$src2, $dst|$dst, $src2}"),
                     [(set VR64:$dst, (IntId64 VR64:$src1, VR64:$src2))]> {
      let isCommutable = Commutable;
    }
    def rm64 : SS38I<opc, MRMSrcMem, (outs VR64:$dst),
                     (ins VR64:$src1, i64mem:$src2),
                     !strconcat(OpcodeStr, "\t{$src2, $dst|$dst, $src2}"),
                     [(set VR64:$dst,
                       (IntId64 VR64:$src1,
                        (bitconvert (memopv2i32 addr:$src2))))]>;

    def rr128 : SS38I<opc, MRMSrcReg, (outs VR128:$dst),
                      (ins VR128:$src1, VR128:$src2),
                      !strconcat(OpcodeStr, "\t{$src2, $dst|$dst, $src2}"),
                      [(set VR128:$dst, (IntId128 VR128:$src1, VR128:$src2))]>,
                      OpSize {
      let isCommutable = Commutable;
    }
    def rm128 : SS38I<opc, MRMSrcMem, (outs VR128:$dst),
                      (ins VR128:$src1, i128mem:$src2),
                      !strconcat(OpcodeStr, "\t{$src2, $dst|$dst, $src2}"),
                      [(set VR128:$dst,
                        (IntId128 VR128:$src1,
                         (bitconvert (memopv4i32 addr:$src2))))]>, OpSize;
  }
}

defm PHADDW      : SS3I_binop_rm_int_16<0x01, "phaddw",
                                        int_x86_ssse3_phadd_w,
                                        int_x86_ssse3_phadd_w_128>;
defm PHADDD      : SS3I_binop_rm_int_32<0x02, "phaddd",
                                        int_x86_ssse3_phadd_d,
                                        int_x86_ssse3_phadd_d_128>;
defm PHADDSW     : SS3I_binop_rm_int_16<0x03, "phaddsw",
                                        int_x86_ssse3_phadd_sw,
                                        int_x86_ssse3_phadd_sw_128>;
defm PHSUBW      : SS3I_binop_rm_int_16<0x05, "phsubw",
                                        int_x86_ssse3_phsub_w,
                                        int_x86_ssse3_phsub_w_128>;
defm PHSUBD      : SS3I_binop_rm_int_32<0x06, "phsubd",
                                        int_x86_ssse3_phsub_d,
                                        int_x86_ssse3_phsub_d_128>;
defm PHSUBSW     : SS3I_binop_rm_int_16<0x07, "phsubsw",
                                        int_x86_ssse3_phsub_sw,
                                        int_x86_ssse3_phsub_sw_128>;
defm PMADDUBSW   : SS3I_binop_rm_int_8 <0x04, "pmaddubsw",
                                        int_x86_ssse3_pmadd_ub_sw,
                                        int_x86_ssse3_pmadd_ub_sw_128>;
defm PMULHRSW    : SS3I_binop_rm_int_16<0x0B, "pmulhrsw",
                                        int_x86_ssse3_pmul_hr_sw,
                                        int_x86_ssse3_pmul_hr_sw_128, 1>;
defm PSHUFB      : SS3I_binop_rm_int_8 <0x00, "pshufb",
                                        int_x86_ssse3_pshuf_b,
                                        int_x86_ssse3_pshuf_b_128>;
defm PSIGNB      : SS3I_binop_rm_int_8 <0x08, "psignb",
                                        int_x86_ssse3_psign_b,
                                        int_x86_ssse3_psign_b_128>;
defm PSIGNW      : SS3I_binop_rm_int_16<0x09, "psignw",
                                        int_x86_ssse3_psign_w,
                                        int_x86_ssse3_psign_w_128>;
defm PSIGND      : SS3I_binop_rm_int_32<0x09, "psignd",
                                        int_x86_ssse3_psign_d,
                                        int_x86_ssse3_psign_d_128>;

let Constraints = "$src1 = $dst" in {
  def PALIGNR64rr  : SS3AI<0x0F, MRMSrcReg, (outs VR64:$dst),
                           (ins VR64:$src1, VR64:$src2, i16imm:$src3),
                           "palignr\t{$src3, $src2, $dst|$dst, $src2, $src3}",
                           [(set VR64:$dst,
                             (int_x86_ssse3_palign_r
                              VR64:$src1, VR64:$src2,
                              imm:$src3))]>;
  def PALIGNR64rm  : SS3AI<0x0F, MRMSrcMem, (outs VR64:$dst),
                           (ins VR64:$src1, i64mem:$src2, i16imm:$src3),
                           "palignr\t{$src3, $src2, $dst|$dst, $src2, $src3}",
                           [(set VR64:$dst,
                             (int_x86_ssse3_palign_r
                              VR64:$src1,
                              (bitconvert (memopv2i32 addr:$src2)),
                              imm:$src3))]>;

  def PALIGNR128rr : SS3AI<0x0F, MRMSrcReg, (outs VR128:$dst),
                           (ins VR128:$src1, VR128:$src2, i32imm:$src3),
                           "palignr\t{$src3, $src2, $dst|$dst, $src2, $src3}",
                           [(set VR128:$dst,
                             (int_x86_ssse3_palign_r_128
                              VR128:$src1, VR128:$src2,
                              imm:$src3))]>, OpSize;
  def PALIGNR128rm : SS3AI<0x0F, MRMSrcMem, (outs VR128:$dst),
                           (ins VR128:$src1, i128mem:$src2, i32imm:$src3),
                           "palignr\t{$src3, $src2, $dst|$dst, $src2, $src3}",
                           [(set VR128:$dst,
                             (int_x86_ssse3_palign_r_128
                              VR128:$src1,
                              (bitconvert (memopv4i32 addr:$src2)),
                              imm:$src3))]>, OpSize;
}

//===----------------------------------------------------------------------===//
// Non-Instruction Patterns
//===----------------------------------------------------------------------===//

// extload f32 -> f64.  This matches load+fextend because we have a hack in 
// the isel (PreprocessForFPConvert) that can introduce loads after dag combine.
// Since these loads aren't folded into the fextend, we have to match it
// explicitly here.
let Predicates = [HasSSE2] in
 def : Pat<(fextend (loadf32 addr:$src)),
           (CVTSS2SDrm addr:$src)>;

// bit_convert
let Predicates = [HasSSE2] in {
  def : Pat<(v2i64 (bitconvert (v4i32 VR128:$src))), (v2i64 VR128:$src)>;
  def : Pat<(v2i64 (bitconvert (v8i16 VR128:$src))), (v2i64 VR128:$src)>;
  def : Pat<(v2i64 (bitconvert (v16i8 VR128:$src))), (v2i64 VR128:$src)>;
  def : Pat<(v2i64 (bitconvert (v2f64 VR128:$src))), (v2i64 VR128:$src)>;
  def : Pat<(v2i64 (bitconvert (v4f32 VR128:$src))), (v2i64 VR128:$src)>;
  def : Pat<(v4i32 (bitconvert (v2i64 VR128:$src))), (v4i32 VR128:$src)>;
  def : Pat<(v4i32 (bitconvert (v8i16 VR128:$src))), (v4i32 VR128:$src)>;
  def : Pat<(v4i32 (bitconvert (v16i8 VR128:$src))), (v4i32 VR128:$src)>;
  def : Pat<(v4i32 (bitconvert (v2f64 VR128:$src))), (v4i32 VR128:$src)>;
  def : Pat<(v4i32 (bitconvert (v4f32 VR128:$src))), (v4i32 VR128:$src)>;
  def : Pat<(v8i16 (bitconvert (v2i64 VR128:$src))), (v8i16 VR128:$src)>;
  def : Pat<(v8i16 (bitconvert (v4i32 VR128:$src))), (v8i16 VR128:$src)>;
  def : Pat<(v8i16 (bitconvert (v16i8 VR128:$src))), (v8i16 VR128:$src)>;
  def : Pat<(v8i16 (bitconvert (v2f64 VR128:$src))), (v8i16 VR128:$src)>;
  def : Pat<(v8i16 (bitconvert (v4f32 VR128:$src))), (v8i16 VR128:$src)>;
  def : Pat<(v16i8 (bitconvert (v2i64 VR128:$src))), (v16i8 VR128:$src)>;
  def : Pat<(v16i8 (bitconvert (v4i32 VR128:$src))), (v16i8 VR128:$src)>;
  def : Pat<(v16i8 (bitconvert (v8i16 VR128:$src))), (v16i8 VR128:$src)>;
  def : Pat<(v16i8 (bitconvert (v2f64 VR128:$src))), (v16i8 VR128:$src)>;
  def : Pat<(v16i8 (bitconvert (v4f32 VR128:$src))), (v16i8 VR128:$src)>;
  def : Pat<(v4f32 (bitconvert (v2i64 VR128:$src))), (v4f32 VR128:$src)>;
  def : Pat<(v4f32 (bitconvert (v4i32 VR128:$src))), (v4f32 VR128:$src)>;
  def : Pat<(v4f32 (bitconvert (v8i16 VR128:$src))), (v4f32 VR128:$src)>;
  def : Pat<(v4f32 (bitconvert (v16i8 VR128:$src))), (v4f32 VR128:$src)>;
  def : Pat<(v4f32 (bitconvert (v2f64 VR128:$src))), (v4f32 VR128:$src)>;
  def : Pat<(v2f64 (bitconvert (v2i64 VR128:$src))), (v2f64 VR128:$src)>;
  def : Pat<(v2f64 (bitconvert (v4i32 VR128:$src))), (v2f64 VR128:$src)>;
  def : Pat<(v2f64 (bitconvert (v8i16 VR128:$src))), (v2f64 VR128:$src)>;
  def : Pat<(v2f64 (bitconvert (v16i8 VR128:$src))), (v2f64 VR128:$src)>;
  def : Pat<(v2f64 (bitconvert (v4f32 VR128:$src))), (v2f64 VR128:$src)>;
}

// Move scalar to XMM zero-extended
// movd to XMM register zero-extends
let AddedComplexity = 15 in {
// Zeroing a VR128 then do a MOVS{S|D} to the lower bits.
def : Pat<(v2f64 (X86vzmovl (v2f64 (scalar_to_vector FR64:$src)))),
          (MOVLSD2PDrr (V_SET0), FR64:$src)>, Requires<[HasSSE2]>;
def : Pat<(v4f32 (X86vzmovl (v4f32 (scalar_to_vector FR32:$src)))),
          (MOVLSS2PSrr (V_SET0), FR32:$src)>, Requires<[HasSSE1]>;
def : Pat<(v4f32 (X86vzmovl (v4f32 VR128:$src))),
          (MOVLPSrr (V_SET0), VR128:$src)>, Requires<[HasSSE1]>;
def : Pat<(v4i32 (X86vzmovl (v4i32 VR128:$src))),
          (MOVLPSrr (V_SET0), VR128:$src)>, Requires<[HasSSE1]>;
}

// Splat v2f64 / v2i64
let AddedComplexity = 10 in {
def : Pat<(vector_shuffle (v2f64 VR128:$src), (undef), SSE_splat_lo_mask:$sm),
          (UNPCKLPDrr VR128:$src, VR128:$src)>,   Requires<[HasSSE2]>;
def : Pat<(vector_shuffle (v2f64 VR128:$src), (undef), UNPCKH_shuffle_mask:$sm),
          (UNPCKHPDrr VR128:$src, VR128:$src)>,   Requires<[HasSSE2]>;
def : Pat<(vector_shuffle (v2i64 VR128:$src), (undef), SSE_splat_lo_mask:$sm),
          (PUNPCKLQDQrr VR128:$src, VR128:$src)>, Requires<[HasSSE2]>;
def : Pat<(vector_shuffle (v2i64 VR128:$src), (undef), UNPCKH_shuffle_mask:$sm),
          (PUNPCKHQDQrr VR128:$src, VR128:$src)>, Requires<[HasSSE2]>;
}

// Special unary SHUFPSrri case.
def : Pat<(v4f32 (vector_shuffle VR128:$src1, (undef),
           SHUFP_unary_shuffle_mask:$sm)),
          (SHUFPSrri VR128:$src1, VR128:$src1, SHUFP_unary_shuffle_mask:$sm)>,
      Requires<[HasSSE1]>;
// Special unary SHUFPDrri case.
def : Pat<(v2f64 (vector_shuffle VR128:$src1, (undef),
           SHUFP_unary_shuffle_mask:$sm)),
          (SHUFPDrri VR128:$src1, VR128:$src1, SHUFP_unary_shuffle_mask:$sm)>,
      Requires<[HasSSE2]>;
// Unary v4f32 shuffle with PSHUF* in order to fold a load.
def : Pat<(vector_shuffle (bc_v4i32 (memopv4f32 addr:$src1)), (undef),
           SHUFP_unary_shuffle_mask:$sm),
          (PSHUFDmi addr:$src1, SHUFP_unary_shuffle_mask:$sm)>,
      Requires<[HasSSE2]>;

// Special binary v4i32 shuffle cases with SHUFPS.
def : Pat<(v4i32 (vector_shuffle VR128:$src1, (v4i32 VR128:$src2),
           PSHUFD_binary_shuffle_mask:$sm)),
          (SHUFPSrri VR128:$src1, VR128:$src2, PSHUFD_binary_shuffle_mask:$sm)>,
           Requires<[HasSSE2]>;
def : Pat<(v4i32 (vector_shuffle VR128:$src1,
           (bc_v4i32 (memopv2i64 addr:$src2)), PSHUFD_binary_shuffle_mask:$sm)),
          (SHUFPSrmi VR128:$src1, addr:$src2, PSHUFD_binary_shuffle_mask:$sm)>,
           Requires<[HasSSE2]>;
// Special binary v2i64 shuffle cases using SHUFPDrri.
def : Pat<(v2i64 (vector_shuffle VR128:$src1, VR128:$src2,
                 SHUFP_shuffle_mask:$sm)),
          (SHUFPDrri VR128:$src1, VR128:$src2, SHUFP_shuffle_mask:$sm)>,
          Requires<[HasSSE2]>;
// Special unary SHUFPDrri case.
def : Pat<(v2i64 (vector_shuffle VR128:$src1, (undef),
                  SHUFP_unary_shuffle_mask:$sm)),
          (SHUFPDrri VR128:$src1, VR128:$src1, SHUFP_unary_shuffle_mask:$sm)>,
      Requires<[HasSSE2]>;

// vector_shuffle v1, <undef>, <0, 0, 1, 1, ...>
let AddedComplexity = 15 in {
def : Pat<(v4i32 (vector_shuffle VR128:$src, (undef),
                                 UNPCKL_v_undef_shuffle_mask:$sm)),
          (PSHUFDri VR128:$src, PSHUFD_shuffle_mask:$sm)>,
          Requires<[OptForSpeed, HasSSE2]>;
def : Pat<(v4f32 (vector_shuffle VR128:$src, (undef),
                                 UNPCKL_v_undef_shuffle_mask:$sm)),
          (PSHUFDri VR128:$src, PSHUFD_shuffle_mask:$sm)>,
          Requires<[OptForSpeed, HasSSE2]>;
}
let AddedComplexity = 10 in {
def : Pat<(v4f32 (vector_shuffle VR128:$src, (undef),
                  UNPCKL_v_undef_shuffle_mask)),
          (UNPCKLPSrr VR128:$src, VR128:$src)>, Requires<[HasSSE1]>;
def : Pat<(v16i8 (vector_shuffle VR128:$src, (undef),
                  UNPCKL_v_undef_shuffle_mask)),
          (PUNPCKLBWrr VR128:$src, VR128:$src)>, Requires<[HasSSE2]>;
def : Pat<(v8i16 (vector_shuffle VR128:$src, (undef),
                  UNPCKL_v_undef_shuffle_mask)),
          (PUNPCKLWDrr VR128:$src, VR128:$src)>, Requires<[HasSSE2]>;
def : Pat<(v4i32 (vector_shuffle VR128:$src, (undef),
                  UNPCKL_v_undef_shuffle_mask)),
          (PUNPCKLDQrr VR128:$src, VR128:$src)>, Requires<[HasSSE2]>;
}

// vector_shuffle v1, <undef>, <2, 2, 3, 3, ...>
let AddedComplexity = 15 in {
def : Pat<(v4i32 (vector_shuffle VR128:$src, (undef),
                 UNPCKH_v_undef_shuffle_mask:$sm)),
          (PSHUFDri VR128:$src, PSHUFD_shuffle_mask:$sm)>,
          Requires<[OptForSpeed, HasSSE2]>;
def : Pat<(v4f32 (vector_shuffle VR128:$src, (undef),
                 UNPCKH_v_undef_shuffle_mask:$sm)),
          (PSHUFDri VR128:$src, PSHUFD_shuffle_mask:$sm)>,
          Requires<[OptForSpeed, HasSSE2]>;
}
let AddedComplexity = 10 in {
def : Pat<(v4f32 (vector_shuffle VR128:$src, (undef),
                  UNPCKH_v_undef_shuffle_mask)),
          (UNPCKHPSrr VR128:$src, VR128:$src)>, Requires<[HasSSE1]>;
def : Pat<(v16i8 (vector_shuffle VR128:$src, (undef),
                  UNPCKH_v_undef_shuffle_mask)),
          (PUNPCKHBWrr VR128:$src, VR128:$src)>, Requires<[HasSSE2]>;
def : Pat<(v8i16 (vector_shuffle VR128:$src, (undef),
                  UNPCKH_v_undef_shuffle_mask)),
          (PUNPCKHWDrr VR128:$src, VR128:$src)>, Requires<[HasSSE2]>;
def : Pat<(v4i32 (vector_shuffle VR128:$src, (undef),
                  UNPCKH_v_undef_shuffle_mask)),
          (PUNPCKHDQrr VR128:$src, VR128:$src)>, Requires<[HasSSE2]>;
}

let AddedComplexity = 20 in {
// vector_shuffle v1, v2 <0, 1, 4, 5> using MOVLHPS
def : Pat<(v4i32 (vector_shuffle VR128:$src1, VR128:$src2,
                  MOVHP_shuffle_mask)),
          (MOVLHPSrr VR128:$src1, VR128:$src2)>;

// vector_shuffle v1, v2 <6, 7, 2, 3> using MOVHLPS
def : Pat<(v4i32 (vector_shuffle VR128:$src1, VR128:$src2,
                  MOVHLPS_shuffle_mask)),
          (MOVHLPSrr VR128:$src1, VR128:$src2)>;

// vector_shuffle v1, undef <2, ?, ?, ?> using MOVHLPS
def : Pat<(v4f32 (vector_shuffle VR128:$src1, (undef),
                  MOVHLPS_v_undef_shuffle_mask)),
          (MOVHLPSrr VR128:$src1, VR128:$src1)>;
def : Pat<(v4i32 (vector_shuffle VR128:$src1, (undef),
                  MOVHLPS_v_undef_shuffle_mask)),
          (MOVHLPSrr VR128:$src1, VR128:$src1)>;
}

let AddedComplexity = 20 in {
// vector_shuffle v1, (load v2) <4, 5, 2, 3> using MOVLPS
// vector_shuffle v1, (load v2) <0, 1, 4, 5> using MOVHPS
def : Pat<(v4f32 (vector_shuffle VR128:$src1, (load addr:$src2),
                  MOVLP_shuffle_mask)),
          (MOVLPSrm VR128:$src1, addr:$src2)>, Requires<[HasSSE1]>;
def : Pat<(v2f64 (vector_shuffle VR128:$src1, (load addr:$src2),
                  MOVLP_shuffle_mask)),
          (MOVLPDrm VR128:$src1, addr:$src2)>, Requires<[HasSSE2]>;
def : Pat<(v4f32 (vector_shuffle VR128:$src1, (load addr:$src2),
                  MOVHP_shuffle_mask)),
          (MOVHPSrm VR128:$src1, addr:$src2)>, Requires<[HasSSE1]>;
def : Pat<(v2f64 (vector_shuffle VR128:$src1, (load addr:$src2),
                  MOVHP_shuffle_mask)),
          (MOVHPDrm VR128:$src1, addr:$src2)>, Requires<[HasSSE2]>;

def : Pat<(v4i32 (vector_shuffle VR128:$src1, (load addr:$src2),
                  MOVLP_shuffle_mask)),
          (MOVLPSrm VR128:$src1, addr:$src2)>, Requires<[HasSSE2]>;
def : Pat<(v2i64 (vector_shuffle VR128:$src1, (load addr:$src2),
                  MOVLP_shuffle_mask)),
          (MOVLPDrm VR128:$src1, addr:$src2)>, Requires<[HasSSE2]>;
def : Pat<(v4i32 (vector_shuffle VR128:$src1, (load addr:$src2),
                  MOVHP_shuffle_mask)),
          (MOVHPSrm VR128:$src1, addr:$src2)>, Requires<[HasSSE1]>;
def : Pat<(v2i64 (vector_shuffle VR128:$src1, (load addr:$src2),
                  MOVHP_shuffle_mask)),
          (MOVHPDrm VR128:$src1, addr:$src2)>, Requires<[HasSSE2]>;
}

// (store (vector_shuffle (load addr), v2, <4, 5, 2, 3>), addr) using MOVLPS
// (store (vector_shuffle (load addr), v2, <0, 1, 4, 5>), addr) using MOVHPS
def : Pat<(store (v4f32 (vector_shuffle (load addr:$src1), VR128:$src2,
                         MOVLP_shuffle_mask)), addr:$src1),
          (MOVLPSmr addr:$src1, VR128:$src2)>, Requires<[HasSSE1]>;
def : Pat<(store (v2f64 (vector_shuffle (load addr:$src1), VR128:$src2,
                         MOVLP_shuffle_mask)), addr:$src1),
          (MOVLPDmr addr:$src1, VR128:$src2)>, Requires<[HasSSE2]>;
def : Pat<(store (v4f32 (vector_shuffle (load addr:$src1), VR128:$src2,
                         MOVHP_shuffle_mask)), addr:$src1),
          (MOVHPSmr addr:$src1, VR128:$src2)>, Requires<[HasSSE1]>;
def : Pat<(store (v2f64 (vector_shuffle (load addr:$src1), VR128:$src2,
                         MOVHP_shuffle_mask)), addr:$src1),
          (MOVHPDmr addr:$src1, VR128:$src2)>, Requires<[HasSSE2]>;

def : Pat<(store (v4i32 (vector_shuffle
                         (bc_v4i32 (loadv2i64 addr:$src1)), VR128:$src2,
                         MOVLP_shuffle_mask)), addr:$src1),
          (MOVLPSmr addr:$src1, VR128:$src2)>, Requires<[HasSSE1]>;
def : Pat<(store (v2i64 (vector_shuffle (load addr:$src1), VR128:$src2,
                         MOVLP_shuffle_mask)), addr:$src1),
          (MOVLPDmr addr:$src1, VR128:$src2)>, Requires<[HasSSE2]>;
def : Pat<(store (v4i32 (vector_shuffle
                         (bc_v4i32 (loadv2i64 addr:$src1)), VR128:$src2,
                         MOVHP_shuffle_mask)), addr:$src1),
          (MOVHPSmr addr:$src1, VR128:$src2)>, Requires<[HasSSE1]>;
def : Pat<(store (v2i64 (vector_shuffle (load addr:$src1), VR128:$src2,
                         MOVHP_shuffle_mask)), addr:$src1),
          (MOVHPDmr addr:$src1, VR128:$src2)>, Requires<[HasSSE2]>;


let AddedComplexity = 15 in {
// Setting the lowest element in the vector.
def : Pat<(v4i32 (vector_shuffle VR128:$src1, VR128:$src2,
                  MOVL_shuffle_mask)),
          (MOVLPSrr VR128:$src1, VR128:$src2)>, Requires<[HasSSE2]>;
def : Pat<(v2i64 (vector_shuffle VR128:$src1, VR128:$src2,
                  MOVL_shuffle_mask)),
          (MOVLPDrr VR128:$src1, VR128:$src2)>, Requires<[HasSSE2]>;

// vector_shuffle v1, v2 <4, 5, 2, 3> using MOVLPDrr (movsd)
def : Pat<(v4f32 (vector_shuffle VR128:$src1, VR128:$src2,
                  MOVLP_shuffle_mask)),
          (MOVLPDrr VR128:$src1, VR128:$src2)>, Requires<[HasSSE2]>;
def : Pat<(v4i32 (vector_shuffle VR128:$src1, VR128:$src2,
                  MOVLP_shuffle_mask)),
          (MOVLPDrr VR128:$src1, VR128:$src2)>, Requires<[HasSSE2]>;
}

// Set lowest element and zero upper elements.
let AddedComplexity = 15 in
def : Pat<(v2f64 (vector_shuffle immAllZerosV_bc, VR128:$src,
           MOVL_shuffle_mask)),
          (MOVZPQILo2PQIrr VR128:$src)>, Requires<[HasSSE2]>;
def : Pat<(v2f64 (X86vzmovl (v2f64 VR128:$src))),
          (MOVZPQILo2PQIrr VR128:$src)>, Requires<[HasSSE2]>;

// Some special case pandn patterns.
def : Pat<(v2i64 (and (xor VR128:$src1, (bc_v2i64 (v4i32 immAllOnesV))),
                  VR128:$src2)),
          (PANDNrr VR128:$src1, VR128:$src2)>, Requires<[HasSSE2]>;
def : Pat<(v2i64 (and (xor VR128:$src1, (bc_v2i64 (v8i16 immAllOnesV))),
                  VR128:$src2)),
          (PANDNrr VR128:$src1, VR128:$src2)>, Requires<[HasSSE2]>;
def : Pat<(v2i64 (and (xor VR128:$src1, (bc_v2i64 (v16i8 immAllOnesV))),
                  VR128:$src2)),
          (PANDNrr VR128:$src1, VR128:$src2)>, Requires<[HasSSE2]>;

def : Pat<(v2i64 (and (xor VR128:$src1, (bc_v2i64 (v4i32 immAllOnesV))),
                  (memop addr:$src2))),
          (PANDNrm VR128:$src1, addr:$src2)>, Requires<[HasSSE2]>;
def : Pat<(v2i64 (and (xor VR128:$src1, (bc_v2i64 (v8i16 immAllOnesV))),
                  (memop addr:$src2))),
          (PANDNrm VR128:$src1, addr:$src2)>, Requires<[HasSSE2]>;
def : Pat<(v2i64 (and (xor VR128:$src1, (bc_v2i64 (v16i8 immAllOnesV))),
                  (memop addr:$src2))),
          (PANDNrm VR128:$src1, addr:$src2)>, Requires<[HasSSE2]>;

// vector -> vector casts
def : Pat<(v4f32 (sint_to_fp (v4i32 VR128:$src))),
          (Int_CVTDQ2PSrr VR128:$src)>, Requires<[HasSSE2]>;
def : Pat<(v4i32 (fp_to_sint (v4f32 VR128:$src))),
          (Int_CVTTPS2DQrr VR128:$src)>, Requires<[HasSSE2]>;
def : Pat<(v2f64 (sint_to_fp (v2i32 VR64:$src))),
          (Int_CVTPI2PDrr VR64:$src)>, Requires<[HasSSE2]>;
def : Pat<(v2i32 (fp_to_sint (v2f64 VR128:$src))),
          (Int_CVTTPD2PIrr VR128:$src)>, Requires<[HasSSE2]>;

// Use movaps / movups for SSE integer load / store (one byte shorter).
def : Pat<(alignedloadv4i32 addr:$src),
          (MOVAPSrm addr:$src)>, Requires<[HasSSE1]>;
def : Pat<(loadv4i32 addr:$src),
          (MOVUPSrm addr:$src)>, Requires<[HasSSE1]>;
def : Pat<(alignedloadv2i64 addr:$src),
          (MOVAPSrm addr:$src)>, Requires<[HasSSE2]>;
def : Pat<(loadv2i64 addr:$src),
          (MOVUPSrm addr:$src)>, Requires<[HasSSE2]>;

def : Pat<(alignedstore (v2i64 VR128:$src), addr:$dst),
          (MOVAPSmr addr:$dst, VR128:$src)>, Requires<[HasSSE2]>;
def : Pat<(alignedstore (v4i32 VR128:$src), addr:$dst),
          (MOVAPSmr addr:$dst, VR128:$src)>, Requires<[HasSSE2]>;
def : Pat<(alignedstore (v8i16 VR128:$src), addr:$dst),
          (MOVAPSmr addr:$dst, VR128:$src)>, Requires<[HasSSE2]>;
def : Pat<(alignedstore (v16i8 VR128:$src), addr:$dst),
          (MOVAPSmr addr:$dst, VR128:$src)>, Requires<[HasSSE2]>;
def : Pat<(store (v2i64 VR128:$src), addr:$dst),
          (MOVUPSmr addr:$dst, VR128:$src)>, Requires<[HasSSE2]>;
def : Pat<(store (v4i32 VR128:$src), addr:$dst),
          (MOVUPSmr addr:$dst, VR128:$src)>, Requires<[HasSSE2]>;
def : Pat<(store (v8i16 VR128:$src), addr:$dst),
          (MOVUPSmr addr:$dst, VR128:$src)>, Requires<[HasSSE2]>;
def : Pat<(store (v16i8 VR128:$src), addr:$dst),
          (MOVUPSmr addr:$dst, VR128:$src)>, Requires<[HasSSE2]>;
          
//===----------------------------------------------------------------------===//
// SSE4.1 Instructions
//===----------------------------------------------------------------------===//

multiclass sse41_fp_unop_rm<bits<8> opcps, bits<8> opcpd,
                            string OpcodeStr,
                            Intrinsic V4F32Int,
                            Intrinsic V2F64Int> {
  // Intrinsic operation, reg.
  // Vector intrinsic operation, reg
  def PSr_Int : SS4AIi8<opcps, MRMSrcReg, 
                    (outs VR128:$dst), (ins VR128:$src1, i32i8imm:$src2),
                    !strconcat(OpcodeStr,
                    "ps\t{$src2, $src1, $dst|$dst, $src1, $src2}"),
                    [(set VR128:$dst, (V4F32Int VR128:$src1, imm:$src2))]>,
                    OpSize;

  // Vector intrinsic operation, mem
  def PSm_Int : SS4AIi8<opcps, MRMSrcMem,
                    (outs VR128:$dst), (ins f128mem:$src1, i32i8imm:$src2),
                    !strconcat(OpcodeStr,
                    "ps\t{$src2, $src1, $dst|$dst, $src1, $src2}"),
                    [(set VR128:$dst,
                          (V4F32Int (memopv4f32 addr:$src1),imm:$src2))]>,
                    OpSize;

  // Vector intrinsic operation, reg
  def PDr_Int : SS4AIi8<opcpd, MRMSrcReg,
                    (outs VR128:$dst), (ins VR128:$src1, i32i8imm:$src2),
                    !strconcat(OpcodeStr,
                    "pd\t{$src2, $src1, $dst|$dst, $src1, $src2}"),
                    [(set VR128:$dst, (V2F64Int VR128:$src1, imm:$src2))]>,
                    OpSize;

  // Vector intrinsic operation, mem
  def PDm_Int : SS4AIi8<opcpd, MRMSrcMem,
                    (outs VR128:$dst), (ins f128mem:$src1, i32i8imm:$src2),
                    !strconcat(OpcodeStr,
                    "pd\t{$src2, $src1, $dst|$dst, $src1, $src2}"),
                    [(set VR128:$dst,
                          (V2F64Int (memopv2f64 addr:$src1),imm:$src2))]>,
                    OpSize;
}

let Constraints = "$src1 = $dst" in {
multiclass sse41_fp_binop_rm<bits<8> opcss, bits<8> opcsd,
                            string OpcodeStr,
                            Intrinsic F32Int,
                            Intrinsic F64Int> {
  // Intrinsic operation, reg.
  def SSr_Int : SS4AIi8<opcss, MRMSrcReg,
                    (outs VR128:$dst), 
                                 (ins VR128:$src1, VR128:$src2, i32i8imm:$src3),
                    !strconcat(OpcodeStr,
                    "ss\t{$src3, $src2, $dst|$dst, $src2, $src3}"),
                    [(set VR128:$dst, 
                            (F32Int VR128:$src1, VR128:$src2, imm:$src3))]>,
                    OpSize;

  // Intrinsic operation, mem.
  def SSm_Int : SS4AIi8<opcss, MRMSrcMem, 
                    (outs VR128:$dst), 
                                (ins VR128:$src1, ssmem:$src2, i32i8imm:$src3),
                    !strconcat(OpcodeStr, 
                    "ss\t{$src3, $src2, $dst|$dst, $src2, $src3}"),
                    [(set VR128:$dst, 
                         (F32Int VR128:$src1, sse_load_f32:$src2, imm:$src3))]>,
                    OpSize;

  // Intrinsic operation, reg.
  def SDr_Int : SS4AIi8<opcsd, MRMSrcReg,
                    (outs VR128:$dst), 
                            (ins VR128:$src1, VR128:$src2, i32i8imm:$src3),
                    !strconcat(OpcodeStr,
                    "sd\t{$src3, $src2, $dst|$dst, $src2, $src3}"),
                    [(set VR128:$dst, 
                            (F64Int VR128:$src1, VR128:$src2, imm:$src3))]>,
                    OpSize;

  // Intrinsic operation, mem.
  def SDm_Int : SS4AIi8<opcsd, MRMSrcMem,
                    (outs VR128:$dst), 
                            (ins VR128:$src1, sdmem:$src2, i32i8imm:$src3),
                    !strconcat(OpcodeStr,
                    "sd\t{$src3, $src2, $dst|$dst, $src2, $src3}"),
                    [(set VR128:$dst, 
                        (F64Int VR128:$src1, sse_load_f64:$src2, imm:$src3))]>,
                    OpSize;
}
}

// FP round - roundss, roundps, roundsd, roundpd
defm ROUND  : sse41_fp_unop_rm<0x08, 0x09, "round",
                               int_x86_sse41_round_ps, int_x86_sse41_round_pd>;
defm ROUND  : sse41_fp_binop_rm<0x0A, 0x0B, "round",
                               int_x86_sse41_round_ss, int_x86_sse41_round_sd>;

// SS41I_unop_rm_int_v16 - SSE 4.1 unary operator whose type is v8i16.
multiclass SS41I_unop_rm_int_v16<bits<8> opc, string OpcodeStr,
                                 Intrinsic IntId128> {
  def rr128 : SS48I<opc, MRMSrcReg, (outs VR128:$dst),
                    (ins VR128:$src),
                    !strconcat(OpcodeStr, "\t{$src, $dst|$dst, $src}"),
                    [(set VR128:$dst, (IntId128 VR128:$src))]>, OpSize;
  def rm128 : SS48I<opc, MRMSrcMem, (outs VR128:$dst),
                     (ins i128mem:$src),
                     !strconcat(OpcodeStr, "\t{$src, $dst|$dst, $src}"),
                     [(set VR128:$dst,
                       (IntId128
                       (bitconvert (memopv8i16 addr:$src))))]>, OpSize;
}

defm PHMINPOSUW : SS41I_unop_rm_int_v16 <0x41, "phminposuw",
                                         int_x86_sse41_phminposuw>;

/// SS41I_binop_rm_int - Simple SSE 4.1 binary operator
let Constraints = "$src1 = $dst" in {
  multiclass SS41I_binop_rm_int<bits<8> opc, string OpcodeStr,
                                Intrinsic IntId128, bit Commutable = 0> {
    def rr : SS48I<opc, MRMSrcReg, (outs VR128:$dst),
                   (ins VR128:$src1, VR128:$src2),
                   !strconcat(OpcodeStr, "\t{$src2, $dst|$dst, $src2}"),
                   [(set VR128:$dst, (IntId128 VR128:$src1, VR128:$src2))]>,
                   OpSize {
      let isCommutable = Commutable;
    }
    def rm : SS48I<opc, MRMSrcMem, (outs VR128:$dst),
                   (ins VR128:$src1, i128mem:$src2),
                   !strconcat(OpcodeStr, "\t{$src2, $dst|$dst, $src2}"),
                   [(set VR128:$dst,
                     (IntId128 VR128:$src1,
                      (bitconvert (memopv16i8 addr:$src2))))]>, OpSize;
  }
}

defm PCMPEQQ      : SS41I_binop_rm_int<0x29, "pcmpeqq",
                                       int_x86_sse41_pcmpeqq, 1>;
defm PACKUSDW     : SS41I_binop_rm_int<0x2B, "packusdw",
                                       int_x86_sse41_packusdw, 0>;
defm PMINSB       : SS41I_binop_rm_int<0x38, "pminsb",
                                       int_x86_sse41_pminsb, 1>;
defm PMINSD       : SS41I_binop_rm_int<0x39, "pminsd",
                                       int_x86_sse41_pminsd, 1>;
defm PMINUD       : SS41I_binop_rm_int<0x3B, "pminud",
                                       int_x86_sse41_pminud, 1>;
defm PMINUW       : SS41I_binop_rm_int<0x3A, "pminuw",
                                       int_x86_sse41_pminuw, 1>;
defm PMAXSB       : SS41I_binop_rm_int<0x3C, "pmaxsb",
                                       int_x86_sse41_pmaxsb, 1>;
defm PMAXSD       : SS41I_binop_rm_int<0x3D, "pmaxsd",
                                       int_x86_sse41_pmaxsd, 1>;
defm PMAXUD       : SS41I_binop_rm_int<0x3F, "pmaxud",
                                       int_x86_sse41_pmaxud, 1>;
defm PMAXUW       : SS41I_binop_rm_int<0x3E, "pmaxuw",
                                       int_x86_sse41_pmaxuw, 1>;

defm PMULDQ       : SS41I_binop_rm_int<0x28, "pmuldq", int_x86_sse41_pmuldq, 1>;

def : Pat<(v2i64 (X86pcmpeqq VR128:$src1, VR128:$src2)),
          (PCMPEQQrr VR128:$src1, VR128:$src2)>;
def : Pat<(v2i64 (X86pcmpeqq VR128:$src1, (memop addr:$src2))),
          (PCMPEQQrm VR128:$src1, addr:$src2)>;

/// SS41I_binop_rm_int - Simple SSE 4.1 binary operator
let Constraints = "$src1 = $dst" in {
  multiclass SS41I_binop_patint<bits<8> opc, string OpcodeStr, ValueType OpVT,
                                SDNode OpNode, Intrinsic IntId128,
                                bit Commutable = 0> {
    def rr : SS48I<opc, MRMSrcReg, (outs VR128:$dst),
                   (ins VR128:$src1, VR128:$src2),
                   !strconcat(OpcodeStr, "\t{$src2, $dst|$dst, $src2}"),
                   [(set VR128:$dst, (OpNode (OpVT VR128:$src1),
                                                   VR128:$src2))]>, OpSize {
      let isCommutable = Commutable;
    }
    def rr_int : SS48I<opc, MRMSrcReg, (outs VR128:$dst),
                      (ins VR128:$src1, VR128:$src2),
                      !strconcat(OpcodeStr, "\t{$src2, $dst|$dst, $src2}"),
                      [(set VR128:$dst, (IntId128 VR128:$src1, VR128:$src2))]>,
                      OpSize {
      let isCommutable = Commutable;
    }
    def rm : SS48I<opc, MRMSrcMem, (outs VR128:$dst),
                   (ins VR128:$src1, i128mem:$src2),
                   !strconcat(OpcodeStr, "\t{$src2, $dst|$dst, $src2}"),
                   [(set VR128:$dst,
                     (OpNode VR128:$src1, (memop addr:$src2)))]>, OpSize;
    def rm_int : SS48I<opc, MRMSrcMem, (outs VR128:$dst),
                       (ins VR128:$src1, i128mem:$src2),
                       !strconcat(OpcodeStr, "\t{$src2, $dst|$dst, $src2}"),
                       [(set VR128:$dst,
                        (IntId128 VR128:$src1, (memop addr:$src2)))]>,
                       OpSize;
  }
}
defm PMULLD       : SS41I_binop_patint<0x40, "pmulld", v4i32, mul,
                                       int_x86_sse41_pmulld, 1>;

/// SS41I_binop_rmi_int - SSE 4.1 binary operator with 8-bit immediate
let Constraints = "$src1 = $dst" in {
  multiclass SS41I_binop_rmi_int<bits<8> opc, string OpcodeStr,
                                 Intrinsic IntId128, bit Commutable = 0> {
    def rri : SS4AIi8<opc, MRMSrcReg, (outs VR128:$dst),
                    (ins VR128:$src1, VR128:$src2, i32i8imm:$src3),
                    !strconcat(OpcodeStr, 
                     "\t{$src3, $src2, $dst|$dst, $src2, $src3}"),
                    [(set VR128:$dst, 
                      (IntId128 VR128:$src1, VR128:$src2, imm:$src3))]>,
                    OpSize {
      let isCommutable = Commutable;
    }
    def rmi : SS4AIi8<opc, MRMSrcMem, (outs VR128:$dst),
                    (ins VR128:$src1, i128mem:$src2, i32i8imm:$src3),
                    !strconcat(OpcodeStr,
                     "\t{$src3, $src2, $dst|$dst, $src2, $src3}"),
                    [(set VR128:$dst,
                      (IntId128 VR128:$src1,
                       (bitconvert (memopv16i8 addr:$src2)), imm:$src3))]>,
                    OpSize;
  }
}

defm BLENDPS      : SS41I_binop_rmi_int<0x0C, "blendps",
                                        int_x86_sse41_blendps, 0>;
defm BLENDPD      : SS41I_binop_rmi_int<0x0D, "blendpd",
                                        int_x86_sse41_blendpd, 0>;
defm PBLENDW      : SS41I_binop_rmi_int<0x0E, "pblendw",
                                        int_x86_sse41_pblendw, 0>;
defm DPPS         : SS41I_binop_rmi_int<0x40, "dpps",
                                        int_x86_sse41_dpps, 1>;
defm DPPD         : SS41I_binop_rmi_int<0x41, "dppd",
                                        int_x86_sse41_dppd, 1>;
defm MPSADBW      : SS41I_binop_rmi_int<0x42, "mpsadbw",
                                        int_x86_sse41_mpsadbw, 1>;


/// SS41I_ternary_int - SSE 4.1 ternary operator
let Uses = [XMM0], Constraints = "$src1 = $dst" in {
  multiclass SS41I_ternary_int<bits<8> opc, string OpcodeStr, Intrinsic IntId> {
    def rr0 : SS48I<opc, MRMSrcReg, (outs VR128:$dst),
                    (ins VR128:$src1, VR128:$src2),
                    !strconcat(OpcodeStr, 
                     "\t{%xmm0, $src2, $dst|$dst, $src2, %xmm0}"),
                    [(set VR128:$dst, (IntId VR128:$src1, VR128:$src2, XMM0))]>,
                    OpSize;

    def rm0 : SS48I<opc, MRMSrcMem, (outs VR128:$dst),
                    (ins VR128:$src1, i128mem:$src2),
                    !strconcat(OpcodeStr,
                     "\t{%xmm0, $src2, $dst|$dst, $src2, %xmm0}"),
                    [(set VR128:$dst,
                      (IntId VR128:$src1,
                       (bitconvert (memopv16i8 addr:$src2)), XMM0))]>, OpSize;
  }
}

defm BLENDVPD     : SS41I_ternary_int<0x15, "blendvpd", int_x86_sse41_blendvpd>;
defm BLENDVPS     : SS41I_ternary_int<0x14, "blendvps", int_x86_sse41_blendvps>;
defm PBLENDVB     : SS41I_ternary_int<0x10, "pblendvb", int_x86_sse41_pblendvb>;


multiclass SS41I_binop_rm_int8<bits<8> opc, string OpcodeStr, Intrinsic IntId> {
  def rr : SS48I<opc, MRMSrcReg, (outs VR128:$dst), (ins VR128:$src),
                 !strconcat(OpcodeStr, "\t{$src, $dst|$dst, $src}"),
                 [(set VR128:$dst, (IntId VR128:$src))]>, OpSize;

  def rm : SS48I<opc, MRMSrcMem, (outs VR128:$dst), (ins i64mem:$src),
                 !strconcat(OpcodeStr, "\t{$src, $dst|$dst, $src}"),
       [(set VR128:$dst,
         (IntId (bitconvert (v2i64 (scalar_to_vector (loadi64 addr:$src))))))]>,
       OpSize;
}

defm PMOVSXBW   : SS41I_binop_rm_int8<0x20, "pmovsxbw", int_x86_sse41_pmovsxbw>;
defm PMOVSXWD   : SS41I_binop_rm_int8<0x23, "pmovsxwd", int_x86_sse41_pmovsxwd>;
defm PMOVSXDQ   : SS41I_binop_rm_int8<0x25, "pmovsxdq", int_x86_sse41_pmovsxdq>;
defm PMOVZXBW   : SS41I_binop_rm_int8<0x30, "pmovzxbw", int_x86_sse41_pmovzxbw>;
defm PMOVZXWD   : SS41I_binop_rm_int8<0x33, "pmovzxwd", int_x86_sse41_pmovzxwd>;
defm PMOVZXDQ   : SS41I_binop_rm_int8<0x35, "pmovzxdq", int_x86_sse41_pmovzxdq>;

// Common patterns involving scalar load.
def : Pat<(int_x86_sse41_pmovsxbw (vzmovl_v2i64 addr:$src)),
          (PMOVSXBWrm addr:$src)>, Requires<[HasSSE41]>;
def : Pat<(int_x86_sse41_pmovsxbw (vzload_v2i64 addr:$src)),
          (PMOVSXBWrm addr:$src)>, Requires<[HasSSE41]>;

def : Pat<(int_x86_sse41_pmovsxwd (vzmovl_v2i64 addr:$src)),
          (PMOVSXWDrm addr:$src)>, Requires<[HasSSE41]>;
def : Pat<(int_x86_sse41_pmovsxwd (vzload_v2i64 addr:$src)),
          (PMOVSXWDrm addr:$src)>, Requires<[HasSSE41]>;

def : Pat<(int_x86_sse41_pmovsxdq (vzmovl_v2i64 addr:$src)),
          (PMOVSXDQrm addr:$src)>, Requires<[HasSSE41]>;
def : Pat<(int_x86_sse41_pmovsxdq (vzload_v2i64 addr:$src)),
          (PMOVSXDQrm addr:$src)>, Requires<[HasSSE41]>;

def : Pat<(int_x86_sse41_pmovzxbw (vzmovl_v2i64 addr:$src)),
          (PMOVZXBWrm addr:$src)>, Requires<[HasSSE41]>;
def : Pat<(int_x86_sse41_pmovzxbw (vzload_v2i64 addr:$src)),
          (PMOVZXBWrm addr:$src)>, Requires<[HasSSE41]>;

def : Pat<(int_x86_sse41_pmovzxwd (vzmovl_v2i64 addr:$src)),
          (PMOVZXWDrm addr:$src)>, Requires<[HasSSE41]>;
def : Pat<(int_x86_sse41_pmovzxwd (vzload_v2i64 addr:$src)),
          (PMOVZXWDrm addr:$src)>, Requires<[HasSSE41]>;

def : Pat<(int_x86_sse41_pmovzxdq (vzmovl_v2i64 addr:$src)),
          (PMOVZXDQrm addr:$src)>, Requires<[HasSSE41]>;
def : Pat<(int_x86_sse41_pmovzxdq (vzload_v2i64 addr:$src)),
          (PMOVZXDQrm addr:$src)>, Requires<[HasSSE41]>;


multiclass SS41I_binop_rm_int4<bits<8> opc, string OpcodeStr, Intrinsic IntId> {
  def rr : SS48I<opc, MRMSrcReg, (outs VR128:$dst), (ins VR128:$src),
                 !strconcat(OpcodeStr, "\t{$src, $dst|$dst, $src}"),
                 [(set VR128:$dst, (IntId VR128:$src))]>, OpSize;

  def rm : SS48I<opc, MRMSrcMem, (outs VR128:$dst), (ins i32mem:$src),
                 !strconcat(OpcodeStr, "\t{$src, $dst|$dst, $src}"),
       [(set VR128:$dst,
         (IntId (bitconvert (v4i32 (scalar_to_vector (loadi32 addr:$src))))))]>,
          OpSize;
}

defm PMOVSXBD   : SS41I_binop_rm_int4<0x21, "pmovsxbd", int_x86_sse41_pmovsxbd>;
defm PMOVSXWQ   : SS41I_binop_rm_int4<0x24, "pmovsxwq", int_x86_sse41_pmovsxwq>;
defm PMOVZXBD   : SS41I_binop_rm_int4<0x31, "pmovzxbd", int_x86_sse41_pmovzxbd>;
defm PMOVZXWQ   : SS41I_binop_rm_int4<0x34, "pmovzxwq", int_x86_sse41_pmovzxwq>;

// Common patterns involving scalar load
def : Pat<(int_x86_sse41_pmovsxbd (vzmovl_v4i32 addr:$src)),
          (PMOVSXBDrm addr:$src)>, Requires<[HasSSE41]>;
def : Pat<(int_x86_sse41_pmovsxwq (vzmovl_v4i32 addr:$src)),
          (PMOVSXWQrm addr:$src)>, Requires<[HasSSE41]>;

def : Pat<(int_x86_sse41_pmovzxbd (vzmovl_v4i32 addr:$src)),
          (PMOVZXBDrm addr:$src)>, Requires<[HasSSE41]>;
def : Pat<(int_x86_sse41_pmovzxwq (vzmovl_v4i32 addr:$src)),
          (PMOVZXWQrm addr:$src)>, Requires<[HasSSE41]>;


multiclass SS41I_binop_rm_int2<bits<8> opc, string OpcodeStr, Intrinsic IntId> {
  def rr : SS48I<opc, MRMSrcReg, (outs VR128:$dst), (ins VR128:$src),
                 !strconcat(OpcodeStr, "\t{$src, $dst|$dst, $src}"),
                 [(set VR128:$dst, (IntId VR128:$src))]>, OpSize;

  // Expecting a i16 load any extended to i32 value.
  def rm : SS48I<opc, MRMSrcMem, (outs VR128:$dst), (ins i16mem:$src),
                 !strconcat(OpcodeStr, "\t{$src, $dst|$dst, $src}"),
                 [(set VR128:$dst, (IntId (bitconvert
                     (v4i32 (scalar_to_vector (loadi16_anyext addr:$src))))))]>,
                 OpSize;
}

defm PMOVSXBQ   : SS41I_binop_rm_int2<0x22, "pmovsxbq", int_x86_sse41_pmovsxbq>;
defm PMOVZXBQ   : SS41I_binop_rm_int2<0x32, "pmovsxbq", int_x86_sse41_pmovzxbq>;

// Common patterns involving scalar load
def : Pat<(int_x86_sse41_pmovsxbq
            (bitconvert (v4i32 (X86vzmovl
                             (v4i32 (scalar_to_vector (loadi32 addr:$src))))))),
          (PMOVSXBQrm addr:$src)>, Requires<[HasSSE41]>;

def : Pat<(int_x86_sse41_pmovzxbq
            (bitconvert (v4i32 (X86vzmovl
                             (v4i32 (scalar_to_vector (loadi32 addr:$src))))))),
          (PMOVZXBQrm addr:$src)>, Requires<[HasSSE41]>;


/// SS41I_binop_ext8 - SSE 4.1 extract 8 bits to 32 bit reg or 8 bit mem
multiclass SS41I_extract8<bits<8> opc, string OpcodeStr> {
  def rr : SS4AIi8<opc, MRMDestReg, (outs GR32:$dst),
                 (ins VR128:$src1, i32i8imm:$src2),
                 !strconcat(OpcodeStr, 
                  "\t{$src2, $src1, $dst|$dst, $src1, $src2}"),
                 [(set GR32:$dst, (X86pextrb (v16i8 VR128:$src1), imm:$src2))]>,
                 OpSize;
  def mr : SS4AIi8<opc, MRMDestMem, (outs),
                 (ins i8mem:$dst, VR128:$src1, i32i8imm:$src2),
                 !strconcat(OpcodeStr, 
                  "\t{$src2, $src1, $dst|$dst, $src1, $src2}"),
                 []>, OpSize;
// FIXME:
// There's an AssertZext in the way of writing the store pattern
// (store (i8 (trunc (X86pextrb (v16i8 VR128:$src1), imm:$src2))), addr:$dst)
}

defm PEXTRB      : SS41I_extract8<0x14, "pextrb">;


/// SS41I_extract16 - SSE 4.1 extract 16 bits to memory destination
multiclass SS41I_extract16<bits<8> opc, string OpcodeStr> {
  def mr : SS4AIi8<opc, MRMDestMem, (outs),
                 (ins i16mem:$dst, VR128:$src1, i32i8imm:$src2),
                 !strconcat(OpcodeStr, 
                  "\t{$src2, $src1, $dst|$dst, $src1, $src2}"),
                 []>, OpSize;
// FIXME:
// There's an AssertZext in the way of writing the store pattern
// (store (i16 (trunc (X86pextrw (v16i8 VR128:$src1), imm:$src2))), addr:$dst)
}

defm PEXTRW      : SS41I_extract16<0x15, "pextrw">;


/// SS41I_extract32 - SSE 4.1 extract 32 bits to int reg or memory destination
multiclass SS41I_extract32<bits<8> opc, string OpcodeStr> {
  def rr : SS4AIi8<opc, MRMDestReg, (outs GR32:$dst),
                 (ins VR128:$src1, i32i8imm:$src2),
                 !strconcat(OpcodeStr, 
                  "\t{$src2, $src1, $dst|$dst, $src1, $src2}"),
                 [(set GR32:$dst,
                  (extractelt (v4i32 VR128:$src1), imm:$src2))]>, OpSize;
  def mr : SS4AIi8<opc, MRMDestMem, (outs),
                 (ins i32mem:$dst, VR128:$src1, i32i8imm:$src2),
                 !strconcat(OpcodeStr, 
                  "\t{$src2, $src1, $dst|$dst, $src1, $src2}"),
                 [(store (extractelt (v4i32 VR128:$src1), imm:$src2),
                          addr:$dst)]>, OpSize;
}

defm PEXTRD      : SS41I_extract32<0x16, "pextrd">;


/// SS41I_extractf32 - SSE 4.1 extract 32 bits fp value to int reg or memory
/// destination
multiclass SS41I_extractf32<bits<8> opc, string OpcodeStr> {
  def rr : SS4AIi8<opc, MRMDestReg, (outs GR32:$dst),
                 (ins VR128:$src1, i32i8imm:$src2),
                 !strconcat(OpcodeStr, 
                  "\t{$src2, $src1, $dst|$dst, $src1, $src2}"),
                 [(set GR32:$dst,
                    (extractelt (bc_v4i32 (v4f32 VR128:$src1)), imm:$src2))]>,
           OpSize;
  def mr : SS4AIi8<opc, MRMDestMem, (outs), 
                 (ins f32mem:$dst, VR128:$src1, i32i8imm:$src2),
                 !strconcat(OpcodeStr, 
                  "\t{$src2, $src1, $dst|$dst, $src1, $src2}"),
                 [(store (extractelt (bc_v4i32 (v4f32 VR128:$src1)), imm:$src2),
                          addr:$dst)]>, OpSize;
}

defm EXTRACTPS   : SS41I_extractf32<0x17, "extractps">;

// Also match an EXTRACTPS store when the store is done as f32 instead of i32.
def : Pat<(store (f32 (bitconvert (extractelt (bc_v4i32 (v4f32 VR128:$src1)),
                                              imm:$src2))),
                 addr:$dst),
          (EXTRACTPSmr addr:$dst, VR128:$src1, imm:$src2)>,
         Requires<[HasSSE41]>;

let Constraints = "$src1 = $dst" in {
  multiclass SS41I_insert8<bits<8> opc, string OpcodeStr> {
    def rr : SS4AIi8<opc, MRMSrcReg, (outs VR128:$dst),
                   (ins VR128:$src1, GR32:$src2, i32i8imm:$src3),
                   !strconcat(OpcodeStr, 
                    "\t{$src3, $src2, $dst|$dst, $src2, $src3}"),
                   [(set VR128:$dst, 
                     (X86pinsrb VR128:$src1, GR32:$src2, imm:$src3))]>, OpSize;
    def rm : SS4AIi8<opc, MRMSrcMem, (outs VR128:$dst),
                   (ins VR128:$src1, i8mem:$src2, i32i8imm:$src3),
                   !strconcat(OpcodeStr,
                    "\t{$src3, $src2, $dst|$dst, $src2, $src3}"),
                   [(set VR128:$dst, 
                     (X86pinsrb VR128:$src1, (extloadi8 addr:$src2),
                                imm:$src3))]>, OpSize;
  }
}

defm PINSRB      : SS41I_insert8<0x20, "pinsrb">;

let Constraints = "$src1 = $dst" in {
  multiclass SS41I_insert32<bits<8> opc, string OpcodeStr> {
    def rr : SS4AIi8<opc, MRMSrcReg, (outs VR128:$dst),
                   (ins VR128:$src1, GR32:$src2, i32i8imm:$src3),
                   !strconcat(OpcodeStr, 
                    "\t{$src3, $src2, $dst|$dst, $src2, $src3}"),
                   [(set VR128:$dst, 
                     (v4i32 (insertelt VR128:$src1, GR32:$src2, imm:$src3)))]>,
                   OpSize;
    def rm : SS4AIi8<opc, MRMSrcMem, (outs VR128:$dst),
                   (ins VR128:$src1, i32mem:$src2, i32i8imm:$src3),
                   !strconcat(OpcodeStr,
                    "\t{$src3, $src2, $dst|$dst, $src2, $src3}"),
                   [(set VR128:$dst, 
                     (v4i32 (insertelt VR128:$src1, (loadi32 addr:$src2),
                                       imm:$src3)))]>, OpSize;
  }
}

defm PINSRD      : SS41I_insert32<0x22, "pinsrd">;

let Constraints = "$src1 = $dst" in {
  multiclass SS41I_insertf32<bits<8> opc, string OpcodeStr> {
    def rr : SS4AIi8<opc, MRMSrcReg, (outs VR128:$dst),
                   (ins VR128:$src1, FR32:$src2, i32i8imm:$src3),
                   !strconcat(OpcodeStr, 
                    "\t{$src3, $src2, $dst|$dst, $src2, $src3}"),
                   [(set VR128:$dst, 
                     (X86insrtps VR128:$src1, FR32:$src2, imm:$src3))]>, OpSize;
    def rm : SS4AIi8<opc, MRMSrcMem, (outs VR128:$dst),
                   (ins VR128:$src1, f32mem:$src2, i32i8imm:$src3),
                   !strconcat(OpcodeStr,
                    "\t{$src3, $src2, $dst|$dst, $src2, $src3}"),
                   [(set VR128:$dst, 
                     (X86insrtps VR128:$src1, (loadf32 addr:$src2),
                                 imm:$src3))]>, OpSize;
  }
}

defm INSERTPS    : SS41I_insertf32<0x21, "insertps">;

let Defs = [EFLAGS] in {
def PTESTrr : SS48I<0x17, MRMSrcReg, (outs), (ins VR128:$src1, VR128:$src2),
                    "ptest \t{$src2, $src1|$src1, $src2}", []>, OpSize;
def PTESTrm : SS48I<0x17, MRMSrcMem, (outs), (ins VR128:$src1, i128mem:$src2),
                    "ptest \t{$src2, $src1|$src1, $src2}", []>, OpSize;
}

def MOVNTDQArm : SS48I<0x2A, MRMSrcMem, (outs VR128:$dst), (ins i128mem:$src),
                       "movntdqa\t{$src, $dst|$dst, $src}",
                       [(set VR128:$dst, (int_x86_sse41_movntdqa addr:$src))]>;

/// SS42I_binop_rm_int - Simple SSE 4.2 binary operator
let Constraints = "$src1 = $dst" in {
  multiclass SS42I_binop_rm_int<bits<8> opc, string OpcodeStr,
                                Intrinsic IntId128, bit Commutable = 0> {
    def rr : SS428I<opc, MRMSrcReg, (outs VR128:$dst),
                   (ins VR128:$src1, VR128:$src2),
                   !strconcat(OpcodeStr, "\t{$src2, $dst|$dst, $src2}"),
                   [(set VR128:$dst, (IntId128 VR128:$src1, VR128:$src2))]>,
                   OpSize {
      let isCommutable = Commutable;
    }
    def rm : SS428I<opc, MRMSrcMem, (outs VR128:$dst),
                   (ins VR128:$src1, i128mem:$src2),
                   !strconcat(OpcodeStr, "\t{$src2, $dst|$dst, $src2}"),
                   [(set VR128:$dst,
                     (IntId128 VR128:$src1,
                      (bitconvert (memopv16i8 addr:$src2))))]>, OpSize;
  }
}

defm PCMPGTQ      : SS42I_binop_rm_int<0x37, "pcmpgtq", int_x86_sse42_pcmpgtq>;

def : Pat<(v2i64 (X86pcmpgtq VR128:$src1, VR128:$src2)),
          (PCMPGTQrr VR128:$src1, VR128:$src2)>;
def : Pat<(v2i64 (X86pcmpgtq VR128:$src1, (memop addr:$src2))),
          (PCMPGTQrm VR128:$src1, addr:$src2)>;

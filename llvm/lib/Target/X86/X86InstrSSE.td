//====- X86InstrSSE.td - Describe the X86 Instruction Set -------*- C++ -*-===//
// 
//                     The LLVM Compiler Infrastructure
//
// This file was developed by the Evan Cheng and is distributed under
// the University of Illinois Open Source License. See LICENSE.TXT for details.
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

def X86loadp   : SDNode<"X86ISD::LOAD_PACK", SDTLoad, [SDNPHasChain]>;
def X86loadu   : SDNode<"X86ISD::LOAD_UA",   SDTLoad, [SDNPHasChain]>;
def X86fmin    : SDNode<"X86ISD::FMIN",      SDTFPBinOp>;
def X86fmax    : SDNode<"X86ISD::FMAX",      SDTFPBinOp>;
def X86fand    : SDNode<"X86ISD::FAND",      SDTFPBinOp,
                        [SDNPCommutative, SDNPAssociative]>;
def X86fxor    : SDNode<"X86ISD::FXOR",      SDTFPBinOp,
                        [SDNPCommutative, SDNPAssociative]>;
def X86comi    : SDNode<"X86ISD::COMI",      SDTX86CmpTest,
                        [SDNPHasChain, SDNPOutFlag]>;
def X86ucomi   : SDNode<"X86ISD::UCOMI",     SDTX86CmpTest,
                        [SDNPHasChain, SDNPOutFlag]>;
def X86s2vec   : SDNode<"X86ISD::S2VEC",  SDTypeProfile<1, 1, []>, []>;
def X86pextrw  : SDNode<"X86ISD::PEXTRW", SDTypeProfile<1, 2, []>, []>;
def X86pinsrw  : SDNode<"X86ISD::PINSRW", SDTypeProfile<1, 3, []>, []>;

//===----------------------------------------------------------------------===//
// SSE Complex Patterns
//===----------------------------------------------------------------------===//

// These are 'extloads' from a scalar to the low element of a vector, zeroing
// the top elements.  These are used for the SSE 'ss' and 'sd' instruction
// forms.
def sse_load_f32 : ComplexPattern<v4f32, 4, "SelectScalarSSELoad", [],
                                  [SDNPHasChain]>;
def sse_load_f64 : ComplexPattern<v2f64, 4, "SelectScalarSSELoad", [],
                                  [SDNPHasChain]>;

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

def X86loadpf32  : PatFrag<(ops node:$ptr), (f32   (X86loadp node:$ptr))>;
def X86loadpf64  : PatFrag<(ops node:$ptr), (f64   (X86loadp node:$ptr))>;

def loadv4f32    : PatFrag<(ops node:$ptr), (v4f32 (load node:$ptr))>;
def loadv2f64    : PatFrag<(ops node:$ptr), (v2f64 (load node:$ptr))>;
def loadv2i64    : PatFrag<(ops node:$ptr), (v2i64 (load node:$ptr))>;

def bc_v4f32 : PatFrag<(ops node:$in), (v4f32 (bitconvert node:$in))>;
def bc_v2f64 : PatFrag<(ops node:$in), (v2f64 (bitconvert node:$in))>;
def bc_v16i8 : PatFrag<(ops node:$in), (v16i8 (bitconvert node:$in))>;
def bc_v8i16 : PatFrag<(ops node:$in), (v8i16 (bitconvert node:$in))>;
def bc_v4i32 : PatFrag<(ops node:$in), (v4i32 (bitconvert node:$in))>;
def bc_v2i64 : PatFrag<(ops node:$in), (v2i64 (bitconvert node:$in))>;

def fp32imm0 : PatLeaf<(f32 fpimm), [{
  return N->isExactlyValue(+0.0);
}]>;

def PSxLDQ_imm  : SDNodeXForm<imm, [{
  // Transformation function: imm >> 3
  return getI32Imm(N->getValue() >> 3);
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

def MOVHLPS_shuffle_mask : PatLeaf<(build_vector), [{
  return X86::isMOVHLPSMask(N);
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

// Instruction templates
// SSI - SSE1 instructions with XS prefix.
// SDI - SSE2 instructions with XD prefix.
// PSI - SSE1 instructions with TB prefix.
// PDI - SSE2 instructions with TB and OpSize prefixes.
// PSIi8 - SSE1 instructions with ImmT == Imm8 and TB prefix.
// PDIi8 - SSE2 instructions with ImmT == Imm8 and TB and OpSize prefixes.
// S3I - SSE3 instructions with TB and OpSize prefixes.
// S3SI - SSE3 instructions with XS prefix.
// S3DI - SSE3 instructions with XD prefix.
class SSI<bits<8> o, Format F, dag ops, string asm, list<dag> pattern>
      : I<o, F, ops, asm, pattern>, XS, Requires<[HasSSE1]>;
class SDI<bits<8> o, Format F, dag ops, string asm, list<dag> pattern>
      : I<o, F, ops, asm, pattern>, XD, Requires<[HasSSE2]>;
class PSI<bits<8> o, Format F, dag ops, string asm, list<dag> pattern>
      : I<o, F, ops, asm, pattern>, TB, Requires<[HasSSE1]>;
class PDI<bits<8> o, Format F, dag ops, string asm, list<dag> pattern>
      : I<o, F, ops, asm, pattern>, TB, OpSize, Requires<[HasSSE2]>;
class PSIi8<bits<8> o, Format F, dag ops, string asm, list<dag> pattern>
      : Ii8<o, F, ops, asm, pattern>, TB, Requires<[HasSSE1]>;
class PDIi8<bits<8> o, Format F, dag ops, string asm, list<dag> pattern>
      : Ii8<o, F, ops, asm, pattern>, TB, OpSize, Requires<[HasSSE2]>;

class S3SI<bits<8> o, Format F, dag ops, string asm, list<dag> pattern>
      : I<o, F, ops, asm, pattern>, XS, Requires<[HasSSE3]>;
class S3DI<bits<8> o, Format F, dag ops, string asm, list<dag> pattern>
      : I<o, F, ops, asm, pattern>, XD, Requires<[HasSSE3]>;
class S3I<bits<8> o, Format F, dag ops, string asm, list<dag> pattern>
      : I<o, F, ops, asm, pattern>, TB, OpSize, Requires<[HasSSE3]>;

//===----------------------------------------------------------------------===//
// Helpers for defining instructions that directly correspond to intrinsics.

multiclass SS_IntUnary<bits<8> o, string OpcodeStr, Intrinsic IntId> {
  def r : SSI<o, MRMSrcReg, (ops VR128:$dst, VR128:$src),
              !strconcat(OpcodeStr, " {$src, $dst|$dst, $src"),
              [(set VR128:$dst, (v4f32 (IntId VR128:$src)))]>;
  def m : SSI<o, MRMSrcMem, (ops VR128:$dst, ssmem:$src),
              !strconcat(OpcodeStr, " {$src, $dst|$dst, $src"),
              [(set VR128:$dst, (v4f32 (IntId sse_load_f32:$src)))]>;
}

multiclass SD_IntUnary<bits<8> o, string OpcodeStr, Intrinsic IntId> {
  def r : SDI<o, MRMSrcReg, (ops VR128:$dst, VR128:$src),
              !strconcat(OpcodeStr, " {$src, $dst|$dst, $src"),
              [(set VR128:$dst, (v2f64 (IntId VR128:$src)))]>;
  def m : SDI<o, MRMSrcMem, (ops VR128:$dst, sdmem:$src),
              !strconcat(OpcodeStr, " {$src, $dst|$dst, $src"),
              [(set VR128:$dst, (v2f64 (IntId sse_load_f64:$src)))]>;
}

class PS_Intr<bits<8> o, string OpcodeStr, Intrinsic IntId>
  : PSI<o, MRMSrcReg, (ops VR128:$dst, VR128:$src),
        !strconcat(OpcodeStr, " {$src, $dst|$dst, $src}"),
        [(set VR128:$dst, (IntId VR128:$src))]>;
class PS_Intm<bits<8> o, string OpcodeStr, Intrinsic IntId>
  : PSI<o, MRMSrcMem, (ops VR128:$dst, f32mem:$src),
        !strconcat(OpcodeStr, " {$src, $dst|$dst, $src}"),
        [(set VR128:$dst, (IntId (load addr:$src)))]>;
class PD_Intr<bits<8> o, string OpcodeStr, Intrinsic IntId>
  : PDI<o, MRMSrcReg, (ops VR128:$dst, VR128:$src),
        !strconcat(OpcodeStr, " {$src, $dst|$dst, $src}"),
        [(set VR128:$dst, (IntId VR128:$src))]>;
class PD_Intm<bits<8> o, string OpcodeStr, Intrinsic IntId>
  : PDI<o, MRMSrcMem, (ops VR128:$dst, f64mem:$src),
        !strconcat(OpcodeStr, " {$src, $dst|$dst, $src}"),
        [(set VR128:$dst, (IntId (load addr:$src)))]>;

class PS_Intrr<bits<8> o, string OpcodeStr, Intrinsic IntId>
  : PSI<o, MRMSrcReg, (ops VR128:$dst, VR128:$src1, VR128:$src2),
        !strconcat(OpcodeStr, " {$src2, $dst|$dst, $src2}"),
        [(set VR128:$dst, (IntId VR128:$src1, VR128:$src2))]>;
class PS_Intrm<bits<8> o, string OpcodeStr, Intrinsic IntId>
  : PSI<o, MRMSrcMem, (ops VR128:$dst, VR128:$src1, f32mem:$src2),
        !strconcat(OpcodeStr, " {$src2, $dst|$dst, $src2}"),
        [(set VR128:$dst, (IntId VR128:$src1, (load addr:$src2)))]>;
class PD_Intrr<bits<8> o, string OpcodeStr, Intrinsic IntId>
  : PDI<o, MRMSrcReg, (ops VR128:$dst, VR128:$src1, VR128:$src2),
        !strconcat(OpcodeStr, " {$src2, $dst|$dst, $src2}"),
        [(set VR128:$dst, (IntId VR128:$src1, VR128:$src2))]>;
class PD_Intrm<bits<8> o, string OpcodeStr, Intrinsic IntId>
  : PDI<o, MRMSrcMem, (ops VR128:$dst, VR128:$src1, f64mem:$src2),
        !strconcat(OpcodeStr, " {$src2, $dst|$dst, $src2}"),
        [(set VR128:$dst, (IntId VR128:$src1, (load addr:$src2)))]>;

// Some 'special' instructions
def IMPLICIT_DEF_FR32 : I<0, Pseudo, (ops FR32:$dst),
                         "#IMPLICIT_DEF $dst",
                         [(set FR32:$dst, (undef))]>, Requires<[HasSSE2]>;
def IMPLICIT_DEF_FR64 : I<0, Pseudo, (ops FR64:$dst),
                         "#IMPLICIT_DEF $dst",
                         [(set FR64:$dst, (undef))]>, Requires<[HasSSE2]>;

// CMOV* - Used to implement the SSE SELECT DAG operation.  Expanded by the
// scheduler into a branch sequence.
let usesCustomDAGSchedInserter = 1 in {  // Expanded by the scheduler.
  def CMOV_FR32 : I<0, Pseudo,
                    (ops FR32:$dst, FR32:$t, FR32:$f, i8imm:$cond),
                    "#CMOV_FR32 PSEUDO!",
                    [(set FR32:$dst, (X86cmov FR32:$t, FR32:$f, imm:$cond))]>;
  def CMOV_FR64 : I<0, Pseudo,
                    (ops FR64:$dst, FR64:$t, FR64:$f, i8imm:$cond),
                    "#CMOV_FR64 PSEUDO!",
                    [(set FR64:$dst, (X86cmov FR64:$t, FR64:$f, imm:$cond))]>;
  def CMOV_V4F32 : I<0, Pseudo,
                    (ops VR128:$dst, VR128:$t, VR128:$f, i8imm:$cond),
                    "#CMOV_V4F32 PSEUDO!",
                    [(set VR128:$dst,
                      (v4f32 (X86cmov VR128:$t, VR128:$f, imm:$cond)))]>;
  def CMOV_V2F64 : I<0, Pseudo,
                    (ops VR128:$dst, VR128:$t, VR128:$f, i8imm:$cond),
                    "#CMOV_V2F64 PSEUDO!",
                    [(set VR128:$dst,
                      (v2f64 (X86cmov VR128:$t, VR128:$f, imm:$cond)))]>;
  def CMOV_V2I64 : I<0, Pseudo,
                    (ops VR128:$dst, VR128:$t, VR128:$f, i8imm:$cond),
                    "#CMOV_V2I64 PSEUDO!",
                    [(set VR128:$dst,
                      (v2i64 (X86cmov VR128:$t, VR128:$f, imm:$cond)))]>;
}

// Move Instructions
def MOVSSrr : SSI<0x10, MRMSrcReg, (ops FR32:$dst, FR32:$src),
                "movss {$src, $dst|$dst, $src}", []>;
def MOVSSrm : SSI<0x10, MRMSrcMem, (ops FR32:$dst, f32mem:$src),
                "movss {$src, $dst|$dst, $src}",
                [(set FR32:$dst, (loadf32 addr:$src))]>;
def MOVSDrr : SDI<0x10, MRMSrcReg, (ops FR64:$dst, FR64:$src),
                "movsd {$src, $dst|$dst, $src}", []>;
def MOVSDrm : SDI<0x10, MRMSrcMem, (ops FR64:$dst, f64mem:$src),
                "movsd {$src, $dst|$dst, $src}",
                [(set FR64:$dst, (loadf64 addr:$src))]>;

def MOVSSmr : SSI<0x11, MRMDestMem, (ops f32mem:$dst, FR32:$src),
                "movss {$src, $dst|$dst, $src}",
                [(store FR32:$src, addr:$dst)]>;
def MOVSDmr : SDI<0x11, MRMDestMem, (ops f64mem:$dst, FR64:$src),
                "movsd {$src, $dst|$dst, $src}",
                [(store FR64:$src, addr:$dst)]>;

/// scalar_sse12_fp_binop_rm - Scalar SSE binops come in four basic forms:
///  1. f32 vs f64 - These come in SSE1/SSE2 forms for float/doubles.
///  2. rr vs rm - They include a reg+reg form and a ref+mem form.
///
/// In addition, scalar SSE ops have an intrinsic form.  This form is unlike the
/// normal form, in that they take an entire vector (instead of a scalar) and
/// leave the top elements undefined.  This adds another two variants of the
/// above permutations, giving us 8 forms for 'instruction'.
///
let isTwoAddress = 1 in {
multiclass scalar_sse12_fp_binop_rm<bits<8> opc, string OpcodeStr,
                                    SDNode OpNode, Intrinsic F32Int,
                                    Intrinsic F64Int, bit Commutable = 0> {
  // Scalar operation, reg+reg.
  def SSrr : SSI<opc, MRMSrcReg, (ops FR32:$dst, FR32:$src1, FR32:$src2),
               !strconcat(OpcodeStr, "ss {$src2, $dst|$dst, $src2"),
               [(set FR32:$dst, (OpNode FR32:$src1, FR32:$src2))]> {
    let isCommutable = Commutable;
  }
  def SDrr : SDI<opc, MRMSrcReg, (ops FR64:$dst, FR64:$src1, FR64:$src2),
               !strconcat(OpcodeStr, "sd {$src2, $dst|$dst, $src2"),
               [(set FR64:$dst, (OpNode FR64:$src1, FR64:$src2))]> {
    let isCommutable = Commutable;
  }
  // Scalar operation, reg+mem.
  def SSrm : SSI<opc, MRMSrcMem, (ops FR32:$dst, FR32:$src1, f32mem:$src2),
                 !strconcat(OpcodeStr, "ss {$src2, $dst|$dst, $src2"),
                 [(set FR32:$dst, (OpNode FR32:$src1, (load addr:$src2)))]>;
  def SDrm : SDI<opc, MRMSrcMem, (ops FR64:$dst, FR64:$src1, f64mem:$src2),
                 !strconcat(OpcodeStr, "sd {$src2, $dst|$dst, $src2"),
                 [(set FR64:$dst, (OpNode FR64:$src1, (load addr:$src2)))]>;
                 
  // Vector intrinsic operation, reg+reg.
  def SSrr_Int : SSI<opc, MRMSrcReg, (ops VR128:$dst, VR128:$src1, VR128:$src2),
                     !strconcat(OpcodeStr, "ss {$src2, $dst|$dst, $src2"),
                     [(set VR128:$dst, (F32Int VR128:$src1, VR128:$src2))]> {
    let isCommutable = Commutable;
  }
  def SDrr_Int : SDI<opc, MRMSrcReg, (ops VR128:$dst, VR128:$src1, VR128:$src2),
               !strconcat(OpcodeStr, "sd {$src2, $dst|$dst, $src2"),
               [(set VR128:$dst, (F64Int VR128:$src1, VR128:$src2))]> {
    let isCommutable = Commutable;
  }
  // Vector intrinsic operation, reg+mem.
  def SSrm_Int : SSI<opc, MRMSrcMem, (ops VR128:$dst, VR128:$src1, ssmem:$src2),
                     !strconcat(OpcodeStr, "ss {$src2, $dst|$dst, $src2"),
                     [(set VR128:$dst, (F32Int VR128:$src1,
                                               sse_load_f32:$src2))]>;
  def SDrm_Int : SDI<opc, MRMSrcMem, (ops VR128:$dst, VR128:$src1, sdmem:$src2),
                     !strconcat(OpcodeStr, "sd {$src2, $dst|$dst, $src2"),
                     [(set VR128:$dst, (F64Int VR128:$src1,
                                               sse_load_f64:$src2))]>;
}
}

// Arithmetic instructions

defm ADD : scalar_sse12_fp_binop_rm<0x58, "add", fadd, 
                                    int_x86_sse_add_ss, int_x86_sse2_add_sd, 1>;
defm MUL : scalar_sse12_fp_binop_rm<0x59, "mul", fmul, 
                                    int_x86_sse_mul_ss, int_x86_sse2_mul_sd, 1>;
defm SUB : scalar_sse12_fp_binop_rm<0x5C, "sub", fsub,
                                    int_x86_sse_sub_ss, int_x86_sse2_sub_sd>;
defm DIV : scalar_sse12_fp_binop_rm<0x5E, "div", fdiv,
                                    int_x86_sse_div_ss, int_x86_sse2_div_sd>;
defm MAX : scalar_sse12_fp_binop_rm<0x5F, "max", X86fmax,
                                    int_x86_sse_max_ss, int_x86_sse2_max_sd>;
defm MIN : scalar_sse12_fp_binop_rm<0x5D, "min", X86fmin,
                                    int_x86_sse_min_ss, int_x86_sse2_min_sd>;


def SQRTSSr : SSI<0x51, MRMSrcReg, (ops FR32:$dst, FR32:$src),
                "sqrtss {$src, $dst|$dst, $src}",
                [(set FR32:$dst, (fsqrt FR32:$src))]>;
def SQRTSSm : SSI<0x51, MRMSrcMem, (ops FR32:$dst, f32mem:$src),
                 "sqrtss {$src, $dst|$dst, $src}",
                 [(set FR32:$dst, (fsqrt (loadf32 addr:$src)))]>;
def SQRTSDr : SDI<0x51, MRMSrcReg, (ops FR64:$dst, FR64:$src),
                 "sqrtsd {$src, $dst|$dst, $src}",
                 [(set FR64:$dst, (fsqrt FR64:$src))]>;
def SQRTSDm : SDI<0x51, MRMSrcMem, (ops FR64:$dst, f64mem:$src),
                 "sqrtsd {$src, $dst|$dst, $src}",
                 [(set FR64:$dst, (fsqrt (loadf64 addr:$src)))]>;

// Aliases to match intrinsics which expect XMM operand(s).

defm SQRTSS_Int  : SS_IntUnary<0x51, "sqrtss" , int_x86_sse_sqrt_ss>;
defm SQRTSD_Int  : SD_IntUnary<0x51, "sqrtsd" , int_x86_sse2_sqrt_sd>;
defm RSQRTSS_Int : SS_IntUnary<0x52, "rsqrtss", int_x86_sse_rsqrt_ss>;
defm RCPSS_Int   : SS_IntUnary<0x53, "rcpss"  , int_x86_sse_rcp_ss>;

// Conversion instructions
def CVTTSS2SIrr: SSI<0x2C, MRMSrcReg, (ops GR32:$dst, FR32:$src),
                    "cvttss2si {$src, $dst|$dst, $src}",
                    [(set GR32:$dst, (fp_to_sint FR32:$src))]>;
def CVTTSS2SIrm: SSI<0x2C, MRMSrcMem, (ops GR32:$dst, f32mem:$src),
                    "cvttss2si {$src, $dst|$dst, $src}",
                    [(set GR32:$dst, (fp_to_sint (loadf32 addr:$src)))]>;
def CVTTSD2SIrr: SDI<0x2C, MRMSrcReg, (ops GR32:$dst, FR64:$src),
                    "cvttsd2si {$src, $dst|$dst, $src}",
                    [(set GR32:$dst, (fp_to_sint FR64:$src))]>;
def CVTTSD2SIrm: SDI<0x2C, MRMSrcMem, (ops GR32:$dst, f64mem:$src),
                    "cvttsd2si {$src, $dst|$dst, $src}",
                    [(set GR32:$dst, (fp_to_sint (loadf64 addr:$src)))]>;
def CVTSD2SSrr: SDI<0x5A, MRMSrcReg, (ops FR32:$dst, FR64:$src),
                   "cvtsd2ss {$src, $dst|$dst, $src}",
                   [(set FR32:$dst, (fround FR64:$src))]>;
def CVTSD2SSrm: SDI<0x5A, MRMSrcMem, (ops FR32:$dst, f64mem:$src), 
                   "cvtsd2ss {$src, $dst|$dst, $src}",
                   [(set FR32:$dst, (fround (loadf64 addr:$src)))]>;
def CVTSI2SSrr: SSI<0x2A, MRMSrcReg, (ops FR32:$dst, GR32:$src),
                  "cvtsi2ss {$src, $dst|$dst, $src}",
                  [(set FR32:$dst, (sint_to_fp GR32:$src))]>;
def CVTSI2SSrm: SSI<0x2A, MRMSrcMem, (ops FR32:$dst, i32mem:$src),
                   "cvtsi2ss {$src, $dst|$dst, $src}",
                   [(set FR32:$dst, (sint_to_fp (loadi32 addr:$src)))]>;
def CVTSI2SDrr: SDI<0x2A, MRMSrcReg, (ops FR64:$dst, GR32:$src),
                   "cvtsi2sd {$src, $dst|$dst, $src}",
                   [(set FR64:$dst, (sint_to_fp GR32:$src))]>;
def CVTSI2SDrm: SDI<0x2A, MRMSrcMem, (ops FR64:$dst, i32mem:$src),
                   "cvtsi2sd {$src, $dst|$dst, $src}",
                   [(set FR64:$dst, (sint_to_fp (loadi32 addr:$src)))]>;

// SSE2 instructions with XS prefix
def CVTSS2SDrr: I<0x5A, MRMSrcReg, (ops FR64:$dst, FR32:$src),
                 "cvtss2sd {$src, $dst|$dst, $src}",
                 [(set FR64:$dst, (fextend FR32:$src))]>, XS,
                Requires<[HasSSE2]>;
def CVTSS2SDrm: I<0x5A, MRMSrcMem, (ops FR64:$dst, f32mem:$src),
                 "cvtss2sd {$src, $dst|$dst, $src}",
                 [(set FR64:$dst, (extloadf32 addr:$src))]>, XS,
                Requires<[HasSSE2]>;

// Match intrinsics which expect XMM operand(s).
def Int_CVTSS2SIrr: SSI<0x2D, MRMSrcReg, (ops GR32:$dst, VR128:$src),
                        "cvtss2si {$src, $dst|$dst, $src}",
                        [(set GR32:$dst, (int_x86_sse_cvtss2si VR128:$src))]>;
def Int_CVTSS2SIrm: SSI<0x2D, MRMSrcMem, (ops GR32:$dst, f32mem:$src),
                        "cvtss2si {$src, $dst|$dst, $src}",
                        [(set GR32:$dst, (int_x86_sse_cvtss2si
                                          (load addr:$src)))]>;
def Int_CVTSD2SIrr: SDI<0x2D, MRMSrcReg, (ops GR32:$dst, VR128:$src),
                        "cvtsd2si {$src, $dst|$dst, $src}",
                        [(set GR32:$dst, (int_x86_sse2_cvtsd2si VR128:$src))]>;
def Int_CVTSD2SIrm: SDI<0x2D, MRMSrcMem, (ops GR32:$dst, f128mem:$src),
                        "cvtsd2si {$src, $dst|$dst, $src}",
                        [(set GR32:$dst, (int_x86_sse2_cvtsd2si
                                          (load addr:$src)))]>;

// Aliases for intrinsics
def Int_CVTTSS2SIrr: SSI<0x2C, MRMSrcReg, (ops GR32:$dst, VR128:$src),
                    "cvttss2si {$src, $dst|$dst, $src}",
                    [(set GR32:$dst, (int_x86_sse_cvttss2si VR128:$src))]>;
def Int_CVTTSS2SIrm: SSI<0x2C, MRMSrcMem, (ops GR32:$dst, f32mem:$src),
                    "cvttss2si {$src, $dst|$dst, $src}",
                    [(set GR32:$dst, (int_x86_sse_cvttss2si(load addr:$src)))]>;
def Int_CVTTSD2SIrr: SDI<0x2C, MRMSrcReg, (ops GR32:$dst, VR128:$src),
                         "cvttsd2si {$src, $dst|$dst, $src}",
                         [(set GR32:$dst, (int_x86_sse2_cvttsd2si VR128:$src))]>;
def Int_CVTTSD2SIrm: SDI<0x2C, MRMSrcMem, (ops GR32:$dst, f128mem:$src),
                         "cvttsd2si {$src, $dst|$dst, $src}",
                         [(set GR32:$dst, (int_x86_sse2_cvttsd2si
                                          (load addr:$src)))]>;

let isTwoAddress = 1 in {
def Int_CVTSI2SSrr: SSI<0x2A, MRMSrcReg,
                        (ops VR128:$dst, VR128:$src1, GR32:$src2),
                        "cvtsi2ss {$src2, $dst|$dst, $src2}",
                        [(set VR128:$dst, (int_x86_sse_cvtsi2ss VR128:$src1,
                                           GR32:$src2))]>;
def Int_CVTSI2SSrm: SSI<0x2A, MRMSrcMem,
                        (ops VR128:$dst, VR128:$src1, i32mem:$src2),
                        "cvtsi2ss {$src2, $dst|$dst, $src2}",
                        [(set VR128:$dst, (int_x86_sse_cvtsi2ss VR128:$src1,
                                           (loadi32 addr:$src2)))]>;
}

// Comparison instructions
let isTwoAddress = 1 in {
def CMPSSrr : SSI<0xC2, MRMSrcReg, 
                (ops FR32:$dst, FR32:$src1, FR32:$src, SSECC:$cc),
                "cmp${cc}ss {$src, $dst|$dst, $src}",
                 []>;
def CMPSSrm : SSI<0xC2, MRMSrcMem, 
                (ops FR32:$dst, FR32:$src1, f32mem:$src, SSECC:$cc),
                "cmp${cc}ss {$src, $dst|$dst, $src}", []>;
def CMPSDrr : SDI<0xC2, MRMSrcReg, 
                (ops FR64:$dst, FR64:$src1, FR64:$src, SSECC:$cc),
                "cmp${cc}sd {$src, $dst|$dst, $src}", []>;
def CMPSDrm : SDI<0xC2, MRMSrcMem, 
                (ops FR64:$dst, FR64:$src1, f64mem:$src, SSECC:$cc),
                "cmp${cc}sd {$src, $dst|$dst, $src}", []>;
}

def UCOMISSrr: PSI<0x2E, MRMSrcReg, (ops FR32:$src1, FR32:$src2),
                 "ucomiss {$src2, $src1|$src1, $src2}",
                 [(X86cmp FR32:$src1, FR32:$src2)]>;
def UCOMISSrm: PSI<0x2E, MRMSrcMem, (ops FR32:$src1, f32mem:$src2),
                 "ucomiss {$src2, $src1|$src1, $src2}",
                 [(X86cmp FR32:$src1, (loadf32 addr:$src2))]>;
def UCOMISDrr: PDI<0x2E, MRMSrcReg, (ops FR64:$src1, FR64:$src2),
                 "ucomisd {$src2, $src1|$src1, $src2}",
                 [(X86cmp FR64:$src1, FR64:$src2)]>;
def UCOMISDrm: PDI<0x2E, MRMSrcMem, (ops FR64:$src1, f64mem:$src2),
                 "ucomisd {$src2, $src1|$src1, $src2}",
                 [(X86cmp FR64:$src1, (loadf64 addr:$src2))]>;

// Aliases to match intrinsics which expect XMM operand(s).
let isTwoAddress = 1 in {
def Int_CMPSSrr : SSI<0xC2, MRMSrcReg, 
                      (ops VR128:$dst, VR128:$src1, VR128:$src, SSECC:$cc),
                      "cmp${cc}ss {$src, $dst|$dst, $src}",
                      [(set VR128:$dst, (int_x86_sse_cmp_ss VR128:$src1,
                                         VR128:$src, imm:$cc))]>;
def Int_CMPSSrm : SSI<0xC2, MRMSrcMem, 
                      (ops VR128:$dst, VR128:$src1, f32mem:$src, SSECC:$cc),
                      "cmp${cc}ss {$src, $dst|$dst, $src}",
                      [(set VR128:$dst, (int_x86_sse_cmp_ss VR128:$src1,
                                         (load addr:$src), imm:$cc))]>;
def Int_CMPSDrr : SDI<0xC2, MRMSrcReg, 
                      (ops VR128:$dst, VR128:$src1, VR128:$src, SSECC:$cc),
                      "cmp${cc}sd {$src, $dst|$dst, $src}", []>;
def Int_CMPSDrm : SDI<0xC2, MRMSrcMem, 
                      (ops VR128:$dst, VR128:$src1, f64mem:$src, SSECC:$cc),
                      "cmp${cc}sd {$src, $dst|$dst, $src}", []>;
}

def Int_UCOMISSrr: PSI<0x2E, MRMSrcReg, (ops VR128:$src1, VR128:$src2),
                       "ucomiss {$src2, $src1|$src1, $src2}",
                       [(X86ucomi (v4f32 VR128:$src1), VR128:$src2)]>;
def Int_UCOMISSrm: PSI<0x2E, MRMSrcMem, (ops VR128:$src1, f128mem:$src2),
                       "ucomiss {$src2, $src1|$src1, $src2}",
                      [(X86ucomi (v4f32 VR128:$src1), (load addr:$src2))]>;
def Int_UCOMISDrr: PDI<0x2E, MRMSrcReg, (ops VR128:$src1, VR128:$src2),
                       "ucomisd {$src2, $src1|$src1, $src2}",
                       [(X86ucomi (v2f64 VR128:$src1), (v2f64 VR128:$src2))]>;
def Int_UCOMISDrm: PDI<0x2E, MRMSrcMem, (ops VR128:$src1, f128mem:$src2),
                       "ucomisd {$src2, $src1|$src1, $src2}",
                      [(X86ucomi (v2f64 VR128:$src1), (load addr:$src2))]>;

def Int_COMISSrr: PSI<0x2F, MRMSrcReg, (ops VR128:$src1, VR128:$src2),
                      "comiss {$src2, $src1|$src1, $src2}",
                      [(X86comi (v4f32 VR128:$src1), VR128:$src2)]>;
def Int_COMISSrm: PSI<0x2F, MRMSrcMem, (ops VR128:$src1, f128mem:$src2),
                      "comiss {$src2, $src1|$src1, $src2}",
                      [(X86comi (v4f32 VR128:$src1), (load addr:$src2))]>;
def Int_COMISDrr: PDI<0x2F, MRMSrcReg, (ops VR128:$src1, VR128:$src2),
                      "comisd {$src2, $src1|$src1, $src2}",
                      [(X86comi (v2f64 VR128:$src1), (v2f64 VR128:$src2))]>;
def Int_COMISDrm: PDI<0x2F, MRMSrcMem, (ops VR128:$src1, f128mem:$src2),
                      "comisd {$src2, $src1|$src1, $src2}",
                      [(X86comi (v2f64 VR128:$src1), (load addr:$src2))]>;

// Aliases of packed instructions for scalar use. These all have names that
// start with 'Fs'.

// Alias instructions that map fld0 to pxor for sse.
def FsFLD0SS : I<0xEF, MRMInitReg, (ops FR32:$dst),
                 "pxor $dst, $dst", [(set FR32:$dst, fp32imm0)]>,
               Requires<[HasSSE1]>, TB, OpSize;
def FsFLD0SD : I<0xEF, MRMInitReg, (ops FR64:$dst),
                 "pxor $dst, $dst", [(set FR64:$dst, fp64imm0)]>,
               Requires<[HasSSE2]>, TB, OpSize;

// Alias instructions to do FR32 / FR64 reg-to-reg copy using movaps / movapd.
// Upper bits are disregarded.
def FsMOVAPSrr : PSI<0x28, MRMSrcReg, (ops FR32:$dst, FR32:$src),
                   "movaps {$src, $dst|$dst, $src}", []>;
def FsMOVAPDrr : PDI<0x28, MRMSrcReg, (ops FR64:$dst, FR64:$src),
                   "movapd {$src, $dst|$dst, $src}", []>;

// Alias instructions to load FR32 / FR64 from f128mem using movaps / movapd.
// Upper bits are disregarded.
def FsMOVAPSrm : PSI<0x28, MRMSrcMem, (ops FR32:$dst, f128mem:$src),
                   "movaps {$src, $dst|$dst, $src}",
                   [(set FR32:$dst, (X86loadpf32 addr:$src))]>;
def FsMOVAPDrm : PDI<0x28, MRMSrcMem, (ops FR64:$dst, f128mem:$src),
                  "movapd {$src, $dst|$dst, $src}",
                  [(set FR64:$dst, (X86loadpf64 addr:$src))]>;

// Alias bitwise logical operations using SSE logical ops on packed FP values.
let isTwoAddress = 1 in {
let isCommutable = 1 in {
def FsANDPSrr : PSI<0x54, MRMSrcReg, (ops FR32:$dst, FR32:$src1, FR32:$src2),
                  "andps {$src2, $dst|$dst, $src2}",
                  [(set FR32:$dst, (X86fand FR32:$src1, FR32:$src2))]>;
def FsANDPDrr : PDI<0x54, MRMSrcReg, (ops FR64:$dst, FR64:$src1, FR64:$src2),
                  "andpd {$src2, $dst|$dst, $src2}",
                  [(set FR64:$dst, (X86fand FR64:$src1, FR64:$src2))]>;
def FsORPSrr  : PSI<0x56, MRMSrcReg, (ops FR32:$dst, FR32:$src1, FR32:$src2),
                  "orps {$src2, $dst|$dst, $src2}", []>;
def FsORPDrr  : PDI<0x56, MRMSrcReg, (ops FR64:$dst, FR64:$src1, FR64:$src2),
                  "orpd {$src2, $dst|$dst, $src2}", []>;
def FsXORPSrr : PSI<0x57, MRMSrcReg, (ops FR32:$dst, FR32:$src1, FR32:$src2),
                  "xorps {$src2, $dst|$dst, $src2}",
                  [(set FR32:$dst, (X86fxor FR32:$src1, FR32:$src2))]>;
def FsXORPDrr : PDI<0x57, MRMSrcReg, (ops FR64:$dst, FR64:$src1, FR64:$src2),
                  "xorpd {$src2, $dst|$dst, $src2}",
                  [(set FR64:$dst, (X86fxor FR64:$src1, FR64:$src2))]>;
}
def FsANDPSrm : PSI<0x54, MRMSrcMem, (ops FR32:$dst, FR32:$src1, f128mem:$src2),
                  "andps {$src2, $dst|$dst, $src2}",
                  [(set FR32:$dst, (X86fand FR32:$src1,
                                    (X86loadpf32 addr:$src2)))]>;
def FsANDPDrm : PDI<0x54, MRMSrcMem, (ops FR64:$dst, FR64:$src1, f128mem:$src2),
                  "andpd {$src2, $dst|$dst, $src2}",
                  [(set FR64:$dst, (X86fand FR64:$src1,
                                    (X86loadpf64 addr:$src2)))]>;
def FsORPSrm  : PSI<0x56, MRMSrcMem, (ops FR32:$dst, FR32:$src1, f128mem:$src2),
                  "orps {$src2, $dst|$dst, $src2}", []>;
def FsORPDrm  : PDI<0x56, MRMSrcMem, (ops FR64:$dst, FR64:$src1, f128mem:$src2),
                  "orpd {$src2, $dst|$dst, $src2}", []>;
def FsXORPSrm : PSI<0x57, MRMSrcMem, (ops FR32:$dst, FR32:$src1, f128mem:$src2),
                  "xorps {$src2, $dst|$dst, $src2}",
                  [(set FR32:$dst, (X86fxor FR32:$src1,
                                    (X86loadpf32 addr:$src2)))]>;
def FsXORPDrm : PDI<0x57, MRMSrcMem, (ops FR64:$dst, FR64:$src1, f128mem:$src2),
                  "xorpd {$src2, $dst|$dst, $src2}",
                  [(set FR64:$dst, (X86fxor FR64:$src1,
                                    (X86loadpf64 addr:$src2)))]>;

def FsANDNPSrr : PSI<0x55, MRMSrcReg, (ops FR32:$dst, FR32:$src1, FR32:$src2),
                   "andnps {$src2, $dst|$dst, $src2}", []>;
def FsANDNPSrm : PSI<0x55, MRMSrcMem, (ops FR32:$dst, FR32:$src1, f128mem:$src2),
                   "andnps {$src2, $dst|$dst, $src2}", []>;
def FsANDNPDrr : PDI<0x55, MRMSrcReg, (ops FR64:$dst, FR64:$src1, FR64:$src2),
                   "andnpd {$src2, $dst|$dst, $src2}", []>;
def FsANDNPDrm : PDI<0x55, MRMSrcMem, (ops FR64:$dst, FR64:$src1, f128mem:$src2),
                   "andnpd {$src2, $dst|$dst, $src2}", []>;
}

//===----------------------------------------------------------------------===//
// SSE packed FP Instructions
//===----------------------------------------------------------------------===//

// Some 'special' instructions
def IMPLICIT_DEF_VR128 : I<0, Pseudo, (ops VR128:$dst),
                           "#IMPLICIT_DEF $dst",
                           [(set VR128:$dst, (v4f32 (undef)))]>,
                         Requires<[HasSSE1]>;

// Move Instructions
def MOVAPSrr : PSI<0x28, MRMSrcReg, (ops VR128:$dst, VR128:$src),
                   "movaps {$src, $dst|$dst, $src}", []>;
def MOVAPSrm : PSI<0x28, MRMSrcMem, (ops VR128:$dst, f128mem:$src),
                   "movaps {$src, $dst|$dst, $src}",
                   [(set VR128:$dst, (loadv4f32 addr:$src))]>;
def MOVAPDrr : PDI<0x28, MRMSrcReg, (ops VR128:$dst, VR128:$src),
                   "movapd {$src, $dst|$dst, $src}", []>;
def MOVAPDrm : PDI<0x28, MRMSrcMem, (ops VR128:$dst, f128mem:$src),
                   "movapd {$src, $dst|$dst, $src}",
                   [(set VR128:$dst, (loadv2f64 addr:$src))]>;

def MOVAPSmr : PSI<0x29, MRMDestMem, (ops f128mem:$dst, VR128:$src),
                   "movaps {$src, $dst|$dst, $src}",
                   [(store (v4f32 VR128:$src), addr:$dst)]>;
def MOVAPDmr : PDI<0x29, MRMDestMem, (ops f128mem:$dst, VR128:$src),
                   "movapd {$src, $dst|$dst, $src}",
                   [(store (v2f64 VR128:$src), addr:$dst)]>;

def MOVUPSrr : PSI<0x10, MRMSrcReg, (ops VR128:$dst, VR128:$src),
                   "movups {$src, $dst|$dst, $src}", []>;
def MOVUPSrm : PSI<0x10, MRMSrcMem, (ops VR128:$dst, f128mem:$src),
                   "movups {$src, $dst|$dst, $src}",
                   [(set VR128:$dst, (int_x86_sse_loadu_ps addr:$src))]>;
def MOVUPSmr : PSI<0x11, MRMDestMem, (ops f128mem:$dst, VR128:$src),
                   "movups {$src, $dst|$dst, $src}",
                   [(int_x86_sse_storeu_ps addr:$dst, VR128:$src)]>;
def MOVUPDrr : PDI<0x10, MRMSrcReg, (ops VR128:$dst, VR128:$src),
                   "movupd {$src, $dst|$dst, $src}", []>;
def MOVUPDrm : PDI<0x10, MRMSrcMem, (ops VR128:$dst, f128mem:$src),
                   "movupd {$src, $dst|$dst, $src}",
                   [(set VR128:$dst, (int_x86_sse2_loadu_pd addr:$src))]>;
def MOVUPDmr : PDI<0x11, MRMDestMem, (ops f128mem:$dst, VR128:$src),
                   "movupd {$src, $dst|$dst, $src}",
                   [(int_x86_sse2_storeu_pd addr:$dst, VR128:$src)]>;

let isTwoAddress = 1 in {
let AddedComplexity = 20 in {
def MOVLPSrm : PSI<0x12, MRMSrcMem, (ops VR128:$dst, VR128:$src1, f64mem:$src2),
                   "movlps {$src2, $dst|$dst, $src2}",
                   [(set VR128:$dst, 
                     (v4f32 (vector_shuffle VR128:$src1,
                     (bc_v4f32 (v2f64 (scalar_to_vector (loadf64 addr:$src2)))),
                             MOVLP_shuffle_mask)))]>;
def MOVLPDrm : PDI<0x12, MRMSrcMem, (ops VR128:$dst, VR128:$src1, f64mem:$src2),
                   "movlpd {$src2, $dst|$dst, $src2}",
                   [(set VR128:$dst, 
                     (v2f64 (vector_shuffle VR128:$src1,
                             (scalar_to_vector (loadf64 addr:$src2)),
                             MOVLP_shuffle_mask)))]>;
def MOVHPSrm : PSI<0x16, MRMSrcMem, (ops VR128:$dst, VR128:$src1, f64mem:$src2),
                   "movhps {$src2, $dst|$dst, $src2}",
                   [(set VR128:$dst, 
                     (v4f32 (vector_shuffle VR128:$src1,
                     (bc_v4f32 (v2f64 (scalar_to_vector (loadf64 addr:$src2)))),
                             MOVHP_shuffle_mask)))]>;
def MOVHPDrm : PDI<0x16, MRMSrcMem, (ops VR128:$dst, VR128:$src1, f64mem:$src2),
                   "movhpd {$src2, $dst|$dst, $src2}",
                   [(set VR128:$dst, 
                     (v2f64 (vector_shuffle VR128:$src1,
                             (scalar_to_vector (loadf64 addr:$src2)),
                             MOVHP_shuffle_mask)))]>;
} // AddedComplexity
}

def MOVLPSmr : PSI<0x13, MRMDestMem, (ops f64mem:$dst, VR128:$src),
                   "movlps {$src, $dst|$dst, $src}",
                   [(store (f64 (vector_extract (bc_v2f64 (v4f32 VR128:$src)),
                                 (iPTR 0))), addr:$dst)]>;
def MOVLPDmr : PDI<0x13, MRMDestMem, (ops f64mem:$dst, VR128:$src),
                   "movlpd {$src, $dst|$dst, $src}",
                   [(store (f64 (vector_extract (v2f64 VR128:$src),
                                 (iPTR 0))), addr:$dst)]>;

// v2f64 extract element 1 is always custom lowered to unpack high to low
// and extract element 0 so the non-store version isn't too horrible.
def MOVHPSmr : PSI<0x17, MRMDestMem, (ops f64mem:$dst, VR128:$src),
                   "movhps {$src, $dst|$dst, $src}",
                   [(store (f64 (vector_extract
                                 (v2f64 (vector_shuffle
                                         (bc_v2f64 (v4f32 VR128:$src)), (undef),
                                         UNPCKH_shuffle_mask)), (iPTR 0))),
                     addr:$dst)]>;
def MOVHPDmr : PDI<0x17, MRMDestMem, (ops f64mem:$dst, VR128:$src),
                   "movhpd {$src, $dst|$dst, $src}",
                   [(store (f64 (vector_extract
                                 (v2f64 (vector_shuffle VR128:$src, (undef),
                                         UNPCKH_shuffle_mask)), (iPTR 0))),
                     addr:$dst)]>;

let isTwoAddress = 1 in {
let AddedComplexity = 15 in {
def MOVLHPSrr : PSI<0x16, MRMSrcReg, (ops VR128:$dst, VR128:$src1, VR128:$src2),
                    "movlhps {$src2, $dst|$dst, $src2}",
                    [(set VR128:$dst,
                      (v4f32 (vector_shuffle VR128:$src1, VR128:$src2,
                              MOVHP_shuffle_mask)))]>;

def MOVHLPSrr : PSI<0x12, MRMSrcReg, (ops VR128:$dst, VR128:$src1, VR128:$src2),
                    "movhlps {$src2, $dst|$dst, $src2}",
                    [(set VR128:$dst,
                      (v4f32 (vector_shuffle VR128:$src1, VR128:$src2,
                              MOVHLPS_shuffle_mask)))]>;
} // AddedComplexity
}

def MOVSHDUPrr : S3SI<0x16, MRMSrcReg, (ops VR128:$dst, VR128:$src),
                      "movshdup {$src, $dst|$dst, $src}",
                      [(set VR128:$dst, (v4f32 (vector_shuffle
                                                VR128:$src, (undef),
                                                MOVSHDUP_shuffle_mask)))]>;
def MOVSHDUPrm : S3SI<0x16, MRMSrcMem, (ops VR128:$dst, f128mem:$src),
                      "movshdup {$src, $dst|$dst, $src}",
                      [(set VR128:$dst, (v4f32 (vector_shuffle
                                                (loadv4f32 addr:$src), (undef),
                                                MOVSHDUP_shuffle_mask)))]>;

def MOVSLDUPrr : S3SI<0x12, MRMSrcReg, (ops VR128:$dst, VR128:$src),
                      "movsldup {$src, $dst|$dst, $src}",
                      [(set VR128:$dst, (v4f32 (vector_shuffle
                                                VR128:$src, (undef),
                                                MOVSLDUP_shuffle_mask)))]>;
def MOVSLDUPrm : S3SI<0x12, MRMSrcMem, (ops VR128:$dst, f128mem:$src),
                      "movsldup {$src, $dst|$dst, $src}",
                      [(set VR128:$dst, (v4f32 (vector_shuffle
                                                (loadv4f32 addr:$src), (undef),
                                                MOVSLDUP_shuffle_mask)))]>;

def MOVDDUPrr : S3DI<0x12, MRMSrcReg, (ops VR128:$dst, VR128:$src),
                      "movddup {$src, $dst|$dst, $src}",
                  [(set VR128:$dst, (v2f64 (vector_shuffle
                                            VR128:$src, (undef),
                                            SSE_splat_lo_mask)))]>;
def MOVDDUPrm : S3DI<0x12, MRMSrcMem, (ops VR128:$dst, f64mem:$src),
                  "movddup {$src, $dst|$dst, $src}",
                  [(set VR128:$dst, (v2f64 (vector_shuffle
                                         (scalar_to_vector (loadf64 addr:$src)),
                                             (undef),
                                            SSE_splat_lo_mask)))]>;

// SSE2 instructions without OpSize prefix
def Int_CVTDQ2PSrr : I<0x5B, MRMSrcReg, (ops VR128:$dst, VR128:$src),
                       "cvtdq2ps {$src, $dst|$dst, $src}",
                       [(set VR128:$dst, (int_x86_sse2_cvtdq2ps VR128:$src))]>,
                     TB, Requires<[HasSSE2]>;
def Int_CVTDQ2PSrm : I<0x5B, MRMSrcMem, (ops VR128:$dst, i128mem:$src),
                       "cvtdq2ps {$src, $dst|$dst, $src}",
                       [(set VR128:$dst, (int_x86_sse2_cvtdq2ps
                                         (bitconvert (loadv2i64 addr:$src))))]>,
                     TB, Requires<[HasSSE2]>;

// SSE2 instructions with XS prefix
def Int_CVTDQ2PDrr : I<0xE6, MRMSrcReg, (ops VR128:$dst, VR128:$src),
                       "cvtdq2pd {$src, $dst|$dst, $src}",
                       [(set VR128:$dst, (int_x86_sse2_cvtdq2pd VR128:$src))]>,
                     XS, Requires<[HasSSE2]>;
def Int_CVTDQ2PDrm : I<0xE6, MRMSrcMem, (ops VR128:$dst, i64mem:$src),
                       "cvtdq2pd {$src, $dst|$dst, $src}",
                       [(set VR128:$dst, (int_x86_sse2_cvtdq2pd
                                          (bitconvert (loadv2i64 addr:$src))))]>,
                     XS, Requires<[HasSSE2]>;

def Int_CVTPS2DQrr : PDI<0x5B, MRMSrcReg, (ops VR128:$dst, VR128:$src),
                         "cvtps2dq {$src, $dst|$dst, $src}",
                         [(set VR128:$dst, (int_x86_sse2_cvtps2dq VR128:$src))]>;
def Int_CVTPS2DQrm : PDI<0x5B, MRMSrcMem, (ops VR128:$dst, f128mem:$src),
                         "cvtps2dq {$src, $dst|$dst, $src}",
                         [(set VR128:$dst, (int_x86_sse2_cvtps2dq
                                            (load addr:$src)))]>;
// SSE2 packed instructions with XS prefix
def Int_CVTTPS2DQrr : I<0x5B, MRMSrcReg, (ops VR128:$dst, VR128:$src),
                        "cvttps2dq {$src, $dst|$dst, $src}",
                        [(set VR128:$dst, (int_x86_sse2_cvttps2dq VR128:$src))]>,
                      XS, Requires<[HasSSE2]>;
def Int_CVTTPS2DQrm : I<0x5B, MRMSrcMem, (ops VR128:$dst, f128mem:$src),
                        "cvttps2dq {$src, $dst|$dst, $src}",
                        [(set VR128:$dst, (int_x86_sse2_cvttps2dq
                                           (load addr:$src)))]>,
                      XS, Requires<[HasSSE2]>;

// SSE2 packed instructions with XD prefix
def Int_CVTPD2DQrr : I<0xE6, MRMSrcReg, (ops VR128:$dst, VR128:$src),
                       "cvtpd2dq {$src, $dst|$dst, $src}",
                       [(set VR128:$dst, (int_x86_sse2_cvtpd2dq VR128:$src))]>,
                     XD, Requires<[HasSSE2]>;
def Int_CVTPD2DQrm : I<0xE6, MRMSrcMem, (ops VR128:$dst, f128mem:$src),
                       "cvtpd2dq {$src, $dst|$dst, $src}",
                       [(set VR128:$dst, (int_x86_sse2_cvtpd2dq
                                          (load addr:$src)))]>,
                     XD, Requires<[HasSSE2]>;
def Int_CVTTPD2DQrr : PDI<0xE6, MRMSrcReg, (ops VR128:$dst, VR128:$src),
                          "cvttpd2dq {$src, $dst|$dst, $src}",
                       [(set VR128:$dst, (int_x86_sse2_cvttpd2dq VR128:$src))]>;
def Int_CVTTPD2DQrm : PDI<0xE6, MRMSrcMem, (ops VR128:$dst, f128mem:$src),
                          "cvttpd2dq {$src, $dst|$dst, $src}",
                          [(set VR128:$dst, (int_x86_sse2_cvttpd2dq
                                             (load addr:$src)))]>;

// SSE2 instructions without OpSize prefix
def Int_CVTPS2PDrr : I<0x5A, MRMSrcReg, (ops VR128:$dst, VR128:$src),
                       "cvtps2pd {$src, $dst|$dst, $src}",
                       [(set VR128:$dst, (int_x86_sse2_cvtps2pd VR128:$src))]>,
                     TB, Requires<[HasSSE2]>;
def Int_CVTPS2PDrm : I<0x5A, MRMSrcReg, (ops VR128:$dst, f64mem:$src),
                       "cvtps2pd {$src, $dst|$dst, $src}",
                       [(set VR128:$dst, (int_x86_sse2_cvtps2pd
                                          (load addr:$src)))]>,
                     TB, Requires<[HasSSE2]>;

def Int_CVTPD2PSrr : PDI<0x5A, MRMSrcReg, (ops VR128:$dst, VR128:$src),
                         "cvtpd2ps {$src, $dst|$dst, $src}",
                        [(set VR128:$dst, (int_x86_sse2_cvtpd2ps VR128:$src))]>;
def Int_CVTPD2PSrm : PDI<0x5A, MRMSrcReg, (ops VR128:$dst, f128mem:$src),
                         "cvtpd2ps {$src, $dst|$dst, $src}",
                         [(set VR128:$dst, (int_x86_sse2_cvtpd2ps
                                            (load addr:$src)))]>;

// Match intrinsics which expect XMM operand(s).
// Aliases for intrinsics
let isTwoAddress = 1 in {
def Int_CVTSI2SDrr: SDI<0x2A, MRMSrcReg,
                        (ops VR128:$dst, VR128:$src1, GR32:$src2),
                        "cvtsi2sd {$src2, $dst|$dst, $src2}",
                        [(set VR128:$dst, (int_x86_sse2_cvtsi2sd VR128:$src1,
                                           GR32:$src2))]>;
def Int_CVTSI2SDrm: SDI<0x2A, MRMSrcMem,
                        (ops VR128:$dst, VR128:$src1, i32mem:$src2),
                        "cvtsi2sd {$src2, $dst|$dst, $src2}",
                        [(set VR128:$dst, (int_x86_sse2_cvtsi2sd VR128:$src1,
                                           (loadi32 addr:$src2)))]>;
def Int_CVTSD2SSrr: SDI<0x5A, MRMSrcReg,
                        (ops VR128:$dst, VR128:$src1, VR128:$src2),
                   "cvtsd2ss {$src2, $dst|$dst, $src2}",
                   [(set VR128:$dst, (int_x86_sse2_cvtsd2ss VR128:$src1,
                                      VR128:$src2))]>;
def Int_CVTSD2SSrm: SDI<0x5A, MRMSrcMem,
                        (ops VR128:$dst, VR128:$src1, f64mem:$src2), 
                   "cvtsd2ss {$src2, $dst|$dst, $src2}",
                   [(set VR128:$dst, (int_x86_sse2_cvtsd2ss VR128:$src1,
                                      (load addr:$src2)))]>;
def Int_CVTSS2SDrr: I<0x5A, MRMSrcReg,
                      (ops VR128:$dst, VR128:$src1, VR128:$src2),
                    "cvtss2sd {$src2, $dst|$dst, $src2}",
                    [(set VR128:$dst, (int_x86_sse2_cvtss2sd VR128:$src1,
                                       VR128:$src2))]>, XS,
                    Requires<[HasSSE2]>;
def Int_CVTSS2SDrm: I<0x5A, MRMSrcMem,
                      (ops VR128:$dst, VR128:$src1, f32mem:$src2),
                    "cvtss2sd {$src2, $dst|$dst, $src2}",
                    [(set VR128:$dst, (int_x86_sse2_cvtss2sd VR128:$src1,
                                       (load addr:$src2)))]>, XS,
                    Requires<[HasSSE2]>;
}

/// packed_sse12_fp_binop_rm - Packed SSE binops come in four basic forms:
///  1. v4f32 vs v2f64 - These come in SSE1/SSE2 forms for float/doubles.
///  2. rr vs rm - They include a reg+reg form and a ref+mem form.
///
let isTwoAddress = 1 in {
multiclass packed_sse12_fp_binop_rm<bits<8> opc, string OpcodeStr,
                                    SDNode OpNode, bit Commutable = 0> {
  // Packed operation, reg+reg.
  def PSrr : PSI<opc, MRMSrcReg, (ops VR128:$dst, VR128:$src1, VR128:$src2),
               !strconcat(OpcodeStr, "ps {$src2, $dst|$dst, $src2"),
               [(set VR128:$dst, (v4f32 (OpNode VR128:$src1, VR128:$src2)))]> {
    let isCommutable = Commutable;
  }
  def PDrr : PDI<opc, MRMSrcReg, (ops VR128:$dst, VR128:$src1, VR128:$src2),
               !strconcat(OpcodeStr, "pd {$src2, $dst|$dst, $src2"),
               [(set VR128:$dst, (v2f64 (OpNode VR128:$src1, VR128:$src2)))]> {
    let isCommutable = Commutable;
  }
  // Packed operation, reg+mem.
  def PSrm : PSI<opc, MRMSrcMem, (ops VR128:$dst, VR128:$src1, f128mem:$src2),
                 !strconcat(OpcodeStr, "ps {$src2, $dst|$dst, $src2"),
                 [(set VR128:$dst, (OpNode VR128:$src1, (loadv4f32 addr:$src2)))]>;
  def PDrm : PDI<opc, MRMSrcMem, (ops VR128:$dst, VR128:$src1, f128mem:$src2),
                 !strconcat(OpcodeStr, "pd {$src2, $dst|$dst, $src2"),
                 [(set VR128:$dst, (OpNode VR128:$src1, (loadv2f64 addr:$src2)))]>;
}
}

defm ADD : packed_sse12_fp_binop_rm<0x58, "add", fadd, 1>;
defm MUL : packed_sse12_fp_binop_rm<0x59, "mul", fmul, 1>;
defm DIV : packed_sse12_fp_binop_rm<0x5E, "div", fdiv>;
defm SUB : packed_sse12_fp_binop_rm<0x5C, "sub", fsub>;

// Arithmetic
let isTwoAddress = 1 in {
def ADDSUBPSrr : S3DI<0xD0, MRMSrcReg,
                      (ops VR128:$dst, VR128:$src1, VR128:$src2),
                      "addsubps {$src2, $dst|$dst, $src2}",
                      [(set VR128:$dst, (int_x86_sse3_addsub_ps VR128:$src1,
                                         VR128:$src2))]>;
def ADDSUBPSrm : S3DI<0xD0, MRMSrcMem,
                      (ops VR128:$dst, VR128:$src1, f128mem:$src2),
                      "addsubps {$src2, $dst|$dst, $src2}",
                      [(set VR128:$dst, (int_x86_sse3_addsub_ps VR128:$src1,
                                         (load addr:$src2)))]>;
def ADDSUBPDrr : S3I<0xD0, MRMSrcReg,
                      (ops VR128:$dst, VR128:$src1, VR128:$src2),
                      "addsubpd {$src2, $dst|$dst, $src2}",
                      [(set VR128:$dst, (int_x86_sse3_addsub_pd VR128:$src1,
                                         VR128:$src2))]>;
def ADDSUBPDrm : S3I<0xD0, MRMSrcMem,
                      (ops VR128:$dst, VR128:$src1, f128mem:$src2),
                      "addsubpd {$src2, $dst|$dst, $src2}",
                      [(set VR128:$dst, (int_x86_sse3_addsub_pd VR128:$src1,
                                         (load addr:$src2)))]>;
}

def SQRTPSr  : PS_Intr<0x51, "sqrtps", int_x86_sse_sqrt_ps>;
def SQRTPSm  : PS_Intm<0x51, "sqrtps", int_x86_sse_sqrt_ps>;
def SQRTPDr  : PD_Intr<0x51, "sqrtpd", int_x86_sse2_sqrt_pd>;
def SQRTPDm  : PD_Intm<0x51, "sqrtpd", int_x86_sse2_sqrt_pd>;

def RSQRTPSr : PS_Intr<0x52, "rsqrtps", int_x86_sse_rsqrt_ps>;
def RSQRTPSm : PS_Intm<0x52, "rsqrtps", int_x86_sse_rsqrt_ps>;
def RCPPSr   : PS_Intr<0x53, "rcpps", int_x86_sse_rcp_ps>;
def RCPPSm   : PS_Intm<0x53, "rcpps", int_x86_sse_rcp_ps>;

let isTwoAddress = 1 in {
let isCommutable = 1 in {
def MAXPSrr  : PS_Intrr<0x5F, "maxps", int_x86_sse_max_ps>;
def MAXPDrr  : PD_Intrr<0x5F, "maxpd", int_x86_sse2_max_pd>;
def MINPSrr  : PS_Intrr<0x5D, "minps", int_x86_sse_min_ps>;
def MINPDrr  : PD_Intrr<0x5D, "minpd", int_x86_sse2_min_pd>;
}
def MAXPSrm  : PS_Intrm<0x5F, "maxps", int_x86_sse_max_ps>;
def MAXPDrm  : PD_Intrm<0x5F, "maxpd", int_x86_sse2_max_pd>;
def MINPSrm  : PS_Intrm<0x5D, "minps", int_x86_sse_min_ps>;
def MINPDrm  : PD_Intrm<0x5D, "minpd", int_x86_sse2_min_pd>;
}

// Logical
let isTwoAddress = 1 in {
let isCommutable = 1 in {
def ANDPSrr : PSI<0x54, MRMSrcReg, (ops VR128:$dst, VR128:$src1, VR128:$src2),
                  "andps {$src2, $dst|$dst, $src2}",
                  [(set VR128:$dst, (v2i64 (and VR128:$src1, VR128:$src2)))]>;
def ANDPDrr : PDI<0x54, MRMSrcReg, (ops VR128:$dst, VR128:$src1, VR128:$src2),
                "andpd {$src2, $dst|$dst, $src2}",
                  [(set VR128:$dst,
                    (and (bc_v2i64 (v2f64 VR128:$src1)),
                         (bc_v2i64 (v2f64 VR128:$src2))))]>;
def ORPSrr  : PSI<0x56, MRMSrcReg, (ops VR128:$dst, VR128:$src1, VR128:$src2),
                  "orps {$src2, $dst|$dst, $src2}",
                  [(set VR128:$dst, (v2i64 (or VR128:$src1, VR128:$src2)))]>;
def ORPDrr  : PDI<0x56, MRMSrcReg, (ops VR128:$dst, VR128:$src1, VR128:$src2),
                  "orpd {$src2, $dst|$dst, $src2}",
                  [(set VR128:$dst,
                    (or (bc_v2i64 (v2f64 VR128:$src1)),
                        (bc_v2i64 (v2f64 VR128:$src2))))]>;
def XORPSrr : PSI<0x57, MRMSrcReg, (ops VR128:$dst, VR128:$src1, VR128:$src2),
                  "xorps {$src2, $dst|$dst, $src2}",
                  [(set VR128:$dst, (v2i64 (xor VR128:$src1, VR128:$src2)))]>;
def XORPDrr : PDI<0x57, MRMSrcReg, (ops VR128:$dst, VR128:$src1, VR128:$src2),
                  "xorpd {$src2, $dst|$dst, $src2}",
                  [(set VR128:$dst,
                    (xor (bc_v2i64 (v2f64 VR128:$src1)),
                         (bc_v2i64 (v2f64 VR128:$src2))))]>;
}
def ANDPSrm : PSI<0x54, MRMSrcMem, (ops VR128:$dst, VR128:$src1, f128mem:$src2),
                  "andps {$src2, $dst|$dst, $src2}",
                  [(set VR128:$dst, (and VR128:$src1,
                                          (bc_v2i64 (loadv4f32 addr:$src2))))]>;
def ANDPDrm : PDI<0x54, MRMSrcMem, (ops VR128:$dst, VR128:$src1, f128mem:$src2),
                  "andpd {$src2, $dst|$dst, $src2}",
                [(set VR128:$dst,
                  (and (bc_v2i64 (v2f64 VR128:$src1)),
                       (bc_v2i64 (loadv2f64 addr:$src2))))]>;
def ORPSrm  : PSI<0x56, MRMSrcMem, (ops VR128:$dst, VR128:$src1, f128mem:$src2),
                  "orps {$src2, $dst|$dst, $src2}",
                  [(set VR128:$dst, (or VR128:$src1,
                                          (bc_v2i64 (loadv4f32 addr:$src2))))]>;
def ORPDrm  : PDI<0x56, MRMSrcMem, (ops VR128:$dst, VR128:$src1, f128mem:$src2),
                "orpd {$src2, $dst|$dst, $src2}",
                 [(set VR128:$dst,
                   (or (bc_v2i64 (v2f64 VR128:$src1)),
                       (bc_v2i64 (loadv2f64 addr:$src2))))]>;
def XORPSrm : PSI<0x57, MRMSrcMem, (ops VR128:$dst, VR128:$src1, f128mem:$src2),
                  "xorps {$src2, $dst|$dst, $src2}",
                  [(set VR128:$dst, (xor VR128:$src1,
                                          (bc_v2i64 (loadv4f32 addr:$src2))))]>;
def XORPDrm : PDI<0x57, MRMSrcMem, (ops VR128:$dst, VR128:$src1, f128mem:$src2),
                  "xorpd {$src2, $dst|$dst, $src2}",
                [(set VR128:$dst,
                  (xor (bc_v2i64 (v2f64 VR128:$src1)),
                       (bc_v2i64 (loadv2f64 addr:$src2))))]>;
def ANDNPSrr : PSI<0x55, MRMSrcReg, (ops VR128:$dst, VR128:$src1, VR128:$src2),
                  "andnps {$src2, $dst|$dst, $src2}",
                  [(set VR128:$dst, (v2i64 (and (xor VR128:$src1,
                                                (bc_v2i64 (v4i32 immAllOnesV))),
                                            VR128:$src2)))]>;
def ANDNPSrm : PSI<0x55, MRMSrcMem, (ops VR128:$dst, VR128:$src1,f128mem:$src2),
                  "andnps {$src2, $dst|$dst, $src2}",
                  [(set VR128:$dst, (v2i64 (and (xor VR128:$src1,
                                                (bc_v2i64 (v4i32 immAllOnesV))),
                                         (bc_v2i64 (loadv4f32 addr:$src2)))))]>;
def ANDNPDrr : PDI<0x55, MRMSrcReg, (ops VR128:$dst, VR128:$src1, VR128:$src2),
                  "andnpd {$src2, $dst|$dst, $src2}",
                [(set VR128:$dst,
                  (and (vnot (bc_v2i64 (v2f64 VR128:$src1))),
                       (bc_v2i64 (v2f64 VR128:$src2))))]>;
def ANDNPDrm : PDI<0x55, MRMSrcMem, (ops VR128:$dst, VR128:$src1,f128mem:$src2),
                  "andnpd {$src2, $dst|$dst, $src2}",
                  [(set VR128:$dst,
                    (and (vnot (bc_v2i64 (v2f64 VR128:$src1))),
                         (bc_v2i64 (loadv2f64 addr:$src2))))]>;
}

let isTwoAddress = 1 in {
def CMPPSrri : PSIi8<0xC2, MRMSrcReg, 
                    (ops VR128:$dst, VR128:$src1, VR128:$src, SSECC:$cc),
                    "cmp${cc}ps {$src, $dst|$dst, $src}",
                    [(set VR128:$dst, (int_x86_sse_cmp_ps VR128:$src1,
                                       VR128:$src, imm:$cc))]>;
def CMPPSrmi : PSIi8<0xC2, MRMSrcMem, 
                    (ops VR128:$dst, VR128:$src1, f128mem:$src, SSECC:$cc),
                    "cmp${cc}ps {$src, $dst|$dst, $src}",
                    [(set VR128:$dst, (int_x86_sse_cmp_ps VR128:$src1,
                                       (load addr:$src), imm:$cc))]>;
def CMPPDrri : PDIi8<0xC2, MRMSrcReg, 
                    (ops VR128:$dst, VR128:$src1, VR128:$src, SSECC:$cc),
                    "cmp${cc}pd {$src, $dst|$dst, $src}",
                    [(set VR128:$dst, (int_x86_sse2_cmp_pd VR128:$src1,
                                       VR128:$src, imm:$cc))]>;
def CMPPDrmi : PDIi8<0xC2, MRMSrcMem, 
                    (ops VR128:$dst, VR128:$src1, f128mem:$src, SSECC:$cc),
                    "cmp${cc}pd {$src, $dst|$dst, $src}",
                    [(set VR128:$dst, (int_x86_sse2_cmp_pd VR128:$src1,
                                       (load addr:$src), imm:$cc))]>;
}

// Shuffle and unpack instructions
let isTwoAddress = 1 in {
let isConvertibleToThreeAddress = 1 in // Convert to pshufd
def SHUFPSrri : PSIi8<0xC6, MRMSrcReg, 
                     (ops VR128:$dst, VR128:$src1, VR128:$src2, i32i8imm:$src3),
                     "shufps {$src3, $src2, $dst|$dst, $src2, $src3}",
                     [(set VR128:$dst, (v4f32 (vector_shuffle
                                               VR128:$src1, VR128:$src2,
                                               SHUFP_shuffle_mask:$src3)))]>;
def SHUFPSrmi : PSIi8<0xC6, MRMSrcMem, 
                   (ops VR128:$dst, VR128:$src1, f128mem:$src2, i32i8imm:$src3),
                     "shufps {$src3, $src2, $dst|$dst, $src2, $src3}",
                     [(set VR128:$dst, (v4f32 (vector_shuffle
                                               VR128:$src1, (load addr:$src2),
                                               SHUFP_shuffle_mask:$src3)))]>;
def SHUFPDrri : PDIi8<0xC6, MRMSrcReg, 
                     (ops VR128:$dst, VR128:$src1, VR128:$src2, i8imm:$src3),
                     "shufpd {$src3, $src2, $dst|$dst, $src2, $src3}",
                     [(set VR128:$dst, (v2f64 (vector_shuffle
                                               VR128:$src1, VR128:$src2,
                                               SHUFP_shuffle_mask:$src3)))]>;
def SHUFPDrmi : PDIi8<0xC6, MRMSrcMem, 
                     (ops VR128:$dst, VR128:$src1, f128mem:$src2, i8imm:$src3),
                     "shufpd {$src3, $src2, $dst|$dst, $src2, $src3}",
                     [(set VR128:$dst, (v2f64 (vector_shuffle
                                               VR128:$src1, (load addr:$src2),
                                               SHUFP_shuffle_mask:$src3)))]>;

let AddedComplexity = 10 in {
def UNPCKHPSrr : PSI<0x15, MRMSrcReg, 
                    (ops VR128:$dst, VR128:$src1, VR128:$src2),
                    "unpckhps {$src2, $dst|$dst, $src2}",
                    [(set VR128:$dst, (v4f32 (vector_shuffle
                                              VR128:$src1, VR128:$src2,
                                              UNPCKH_shuffle_mask)))]>;
def UNPCKHPSrm : PSI<0x15, MRMSrcMem, 
                    (ops VR128:$dst, VR128:$src1, f128mem:$src2),
                    "unpckhps {$src2, $dst|$dst, $src2}",
                    [(set VR128:$dst, (v4f32 (vector_shuffle
                                              VR128:$src1, (load addr:$src2),
                                              UNPCKH_shuffle_mask)))]>;
def UNPCKHPDrr : PDI<0x15, MRMSrcReg, 
                    (ops VR128:$dst, VR128:$src1, VR128:$src2),
                    "unpckhpd {$src2, $dst|$dst, $src2}",
                    [(set VR128:$dst, (v2f64 (vector_shuffle
                                              VR128:$src1, VR128:$src2,
                                              UNPCKH_shuffle_mask)))]>;
def UNPCKHPDrm : PDI<0x15, MRMSrcMem, 
                    (ops VR128:$dst, VR128:$src1, f128mem:$src2),
                    "unpckhpd {$src2, $dst|$dst, $src2}",
                    [(set VR128:$dst, (v2f64 (vector_shuffle
                                              VR128:$src1, (load addr:$src2),
                                              UNPCKH_shuffle_mask)))]>;

def UNPCKLPSrr : PSI<0x14, MRMSrcReg, 
                    (ops VR128:$dst, VR128:$src1, VR128:$src2),
                    "unpcklps {$src2, $dst|$dst, $src2}",
                    [(set VR128:$dst, (v4f32 (vector_shuffle
                                              VR128:$src1, VR128:$src2,
                                              UNPCKL_shuffle_mask)))]>;
def UNPCKLPSrm : PSI<0x14, MRMSrcMem, 
                    (ops VR128:$dst, VR128:$src1, f128mem:$src2),
                    "unpcklps {$src2, $dst|$dst, $src2}",
                    [(set VR128:$dst, (v4f32 (vector_shuffle
                                              VR128:$src1, (load addr:$src2),
                                              UNPCKL_shuffle_mask)))]>;
def UNPCKLPDrr : PDI<0x14, MRMSrcReg, 
                    (ops VR128:$dst, VR128:$src1, VR128:$src2),
                    "unpcklpd {$src2, $dst|$dst, $src2}",
                    [(set VR128:$dst, (v2f64 (vector_shuffle
                                              VR128:$src1, VR128:$src2,
                                              UNPCKL_shuffle_mask)))]>;
def UNPCKLPDrm : PDI<0x14, MRMSrcMem, 
                    (ops VR128:$dst, VR128:$src1, f128mem:$src2),
                    "unpcklpd {$src2, $dst|$dst, $src2}",
                    [(set VR128:$dst, (v2f64 (vector_shuffle
                                              VR128:$src1, (load addr:$src2),
                                              UNPCKL_shuffle_mask)))]>;
} // AddedComplexity
}

// Horizontal ops

class S3D_Intrr<bits<8> o, string OpcodeStr, Intrinsic IntId>
  : S3DI<o, MRMSrcReg, (ops VR128:$dst, VR128:$src1, VR128:$src2),
         !strconcat(OpcodeStr, " {$src2, $dst|$dst, $src2}"),
         [(set VR128:$dst, (v4f32 (IntId VR128:$src1, VR128:$src2)))]>;
class S3D_Intrm<bits<8> o, string OpcodeStr, Intrinsic IntId>
  : S3DI<o, MRMSrcMem, (ops VR128:$dst, VR128:$src1, f128mem:$src2),
         !strconcat(OpcodeStr, " {$src2, $dst|$dst, $src2}"),
         [(set VR128:$dst, (v4f32 (IntId VR128:$src1, (load addr:$src2))))]>;
class S3_Intrr<bits<8> o, string OpcodeStr, Intrinsic IntId>
  : S3I<o, MRMSrcReg, (ops VR128:$dst, VR128:$src1, VR128:$src2),
         !strconcat(OpcodeStr, " {$src2, $dst|$dst, $src2}"),
         [(set VR128:$dst, (v2f64 (IntId VR128:$src1, VR128:$src2)))]>;
class S3_Intrm<bits<8> o, string OpcodeStr, Intrinsic IntId>
  : S3I<o, MRMSrcMem, (ops VR128:$dst, VR128:$src1, f128mem:$src2),
         !strconcat(OpcodeStr, " {$src2, $dst|$dst, $src2}"),
         [(set VR128:$dst, (v2f64 (IntId VR128:$src1, (load addr:$src2))))]>;

let isTwoAddress = 1 in {
def HADDPSrr : S3D_Intrr<0x7C, "haddps", int_x86_sse3_hadd_ps>;
def HADDPSrm : S3D_Intrm<0x7C, "haddps", int_x86_sse3_hadd_ps>;
def HADDPDrr : S3_Intrr <0x7C, "haddpd", int_x86_sse3_hadd_pd>;
def HADDPDrm : S3_Intrm <0x7C, "haddpd", int_x86_sse3_hadd_pd>;
def HSUBPSrr : S3D_Intrr<0x7D, "hsubps", int_x86_sse3_hsub_ps>;
def HSUBPSrm : S3D_Intrm<0x7D, "hsubps", int_x86_sse3_hsub_ps>;
def HSUBPDrr : S3_Intrr <0x7D, "hsubpd", int_x86_sse3_hsub_pd>;
def HSUBPDrm : S3_Intrm <0x7D, "hsubpd", int_x86_sse3_hsub_pd>;
}

//===----------------------------------------------------------------------===//
// SSE integer instructions
//===----------------------------------------------------------------------===//

// Move Instructions
def MOVDQArr : PDI<0x6F, MRMSrcReg, (ops VR128:$dst, VR128:$src),
                   "movdqa {$src, $dst|$dst, $src}", []>;
def MOVDQArm : PDI<0x6F, MRMSrcMem, (ops VR128:$dst, i128mem:$src),
                   "movdqa {$src, $dst|$dst, $src}",
                   [(set VR128:$dst, (loadv2i64 addr:$src))]>;
def MOVDQAmr : PDI<0x7F, MRMDestMem, (ops i128mem:$dst, VR128:$src),
                   "movdqa {$src, $dst|$dst, $src}",
                   [(store (v2i64 VR128:$src), addr:$dst)]>;
def MOVDQUrm :   I<0x6F, MRMSrcMem, (ops VR128:$dst, i128mem:$src),
                   "movdqu {$src, $dst|$dst, $src}",
                   [(set VR128:$dst, (int_x86_sse2_loadu_dq addr:$src))]>,
                 XS, Requires<[HasSSE2]>;
def MOVDQUmr :   I<0x7F, MRMDestMem, (ops i128mem:$dst, VR128:$src),
                   "movdqu {$src, $dst|$dst, $src}",
                   [(int_x86_sse2_storeu_dq addr:$dst, VR128:$src)]>,
                 XS, Requires<[HasSSE2]>;
def LDDQUrm : S3DI<0xF0, MRMSrcMem, (ops VR128:$dst, i128mem:$src),
                   "lddqu {$src, $dst|$dst, $src}",
                   [(set VR128:$dst, (int_x86_sse3_ldu_dq addr:$src))]>;


let isTwoAddress = 1 in {
multiclass PDI_binop_rm_int<bits<8> opc, string OpcodeStr, Intrinsic IntId,
                            bit Commutable = 0> {
  def rr : PDI<opc, MRMSrcReg, (ops VR128:$dst, VR128:$src1, VR128:$src2),
               !strconcat(OpcodeStr, " {$src2, $dst|$dst, $src2"),
               [(set VR128:$dst, (IntId VR128:$src1, VR128:$src2))]> {
    let isCommutable = Commutable;
  }
  def rm : PDI<opc, MRMSrcMem, (ops VR128:$dst, VR128:$src1, i128mem:$src2),
               !strconcat(OpcodeStr, " {$src2, $dst|$dst, $src2"),
               [(set VR128:$dst, (IntId VR128:$src1,
                                        (bitconvert (loadv2i64 addr:$src2))))]>;
}
}

let isTwoAddress = 1 in {
multiclass PDI_binop_rmi_int<bits<8> opc, bits<8> opc2, Format ImmForm,
                             string OpcodeStr, Intrinsic IntId> {
  def rr : PDI<opc, MRMSrcReg, (ops VR128:$dst, VR128:$src1, VR128:$src2),
               !strconcat(OpcodeStr, " {$src2, $dst|$dst, $src2"),
               [(set VR128:$dst, (IntId VR128:$src1, VR128:$src2))]>;
  def rm : PDI<opc, MRMSrcMem, (ops VR128:$dst, VR128:$src1, i128mem:$src2),
               !strconcat(OpcodeStr, " {$src2, $dst|$dst, $src2"),
               [(set VR128:$dst, (IntId VR128:$src1,
                                        (bitconvert (loadv2i64 addr:$src2))))]>;
  def ri : PDIi8<opc2, ImmForm, (ops VR128:$dst, VR128:$src1, i32i8imm:$src2),
               !strconcat(OpcodeStr, " {$src2, $dst|$dst, $src2"),
               [(set VR128:$dst, (IntId VR128:$src1,
                                        (scalar_to_vector (i32 imm:$src2))))]>;
}
}


let isTwoAddress = 1 in {
/// PDI_binop_rm - Simple SSE2 binary operator.
multiclass PDI_binop_rm<bits<8> opc, string OpcodeStr, SDNode OpNode,
                        ValueType OpVT, bit Commutable = 0> {
  def rr : PDI<opc, MRMSrcReg, (ops VR128:$dst, VR128:$src1, VR128:$src2),
               !strconcat(OpcodeStr, " {$src2, $dst|$dst, $src2"),
               [(set VR128:$dst, (OpVT (OpNode VR128:$src1, VR128:$src2)))]> {
    let isCommutable = Commutable;
  }
  def rm : PDI<opc, MRMSrcMem, (ops VR128:$dst, VR128:$src1, i128mem:$src2),
               !strconcat(OpcodeStr, " {$src2, $dst|$dst, $src2"),
               [(set VR128:$dst, (OpVT (OpNode VR128:$src1,
                                       (bitconvert (loadv2i64 addr:$src2)))))]>;
}

/// PDI_binop_rm_v2i64 - Simple SSE2 binary operator whose type is v2i64.
///
/// FIXME: we could eliminate this and use PDI_binop_rm instead if tblgen knew
/// to collapse (bitconvert VT to VT) into its operand.
///
multiclass PDI_binop_rm_v2i64<bits<8> opc, string OpcodeStr, SDNode OpNode,
                              bit Commutable = 0> {
  def rr : PDI<opc, MRMSrcReg, (ops VR128:$dst, VR128:$src1, VR128:$src2),
               !strconcat(OpcodeStr, " {$src2, $dst|$dst, $src2"),
               [(set VR128:$dst, (v2i64 (OpNode VR128:$src1, VR128:$src2)))]> {
    let isCommutable = Commutable;
  }
  def rm : PDI<opc, MRMSrcMem, (ops VR128:$dst, VR128:$src1, i128mem:$src2),
               !strconcat(OpcodeStr, " {$src2, $dst|$dst, $src2"),
               [(set VR128:$dst, (OpNode VR128:$src1,(loadv2i64 addr:$src2)))]>;
}
}


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


defm PSLLW : PDI_binop_rmi_int<0xF1, 0x71, MRM6r, "psllw", int_x86_sse2_psll_w>;
defm PSLLD : PDI_binop_rmi_int<0xF2, 0x72, MRM6r, "pslld", int_x86_sse2_psll_d>;
defm PSLLQ : PDI_binop_rmi_int<0xF3, 0x73, MRM6r, "psllq", int_x86_sse2_psll_q>;

defm PSRLW : PDI_binop_rmi_int<0xD1, 0x71, MRM2r, "psrlw", int_x86_sse2_psrl_w>;
defm PSRLD : PDI_binop_rmi_int<0xD2, 0x72, MRM2r, "psrld", int_x86_sse2_psrl_d>;
defm PSRLQ : PDI_binop_rmi_int<0xD3, 0x73, MRM2r, "psrlq", int_x86_sse2_psrl_q>;

defm PSRAW : PDI_binop_rmi_int<0xE1, 0x71, MRM4r, "psraw", int_x86_sse2_psra_w>;
defm PSRAD : PDI_binop_rmi_int<0xE2, 0x72, MRM4r, "psrad", int_x86_sse2_psra_d>;
// PSRAQ doesn't exist in SSE[1-3].


// 128-bit logical shifts.
let isTwoAddress = 1 in {
def PSLLDQri : PDIi8<0x73, MRM7r, (ops VR128:$dst, VR128:$src1, i32i8imm:$src2),
                     "pslldq {$src2, $dst|$dst, $src2}", []>;
def PSRLDQri : PDIi8<0x73, MRM3r, (ops VR128:$dst, VR128:$src1, i32i8imm:$src2),
                     "psrldq {$src2, $dst|$dst, $src2}", []>;
// PSRADQri doesn't exist in SSE[1-3].
}

let Predicates = [HasSSE2] in {
  def : Pat<(int_x86_sse2_psll_dq VR128:$src1, imm:$src2),
            (v2i64 (PSLLDQri VR128:$src1, (PSxLDQ_imm imm:$src2)))>;
  def : Pat<(int_x86_sse2_psrl_dq VR128:$src1, imm:$src2),
            (v2i64 (PSRLDQri VR128:$src1, (PSxLDQ_imm imm:$src2)))>;
}

// Logical
defm PAND : PDI_binop_rm_v2i64<0xDB, "pand", and, 1>;
defm POR  : PDI_binop_rm_v2i64<0xEB, "por" , or , 1>;
defm PXOR : PDI_binop_rm_v2i64<0xEF, "pxor", xor, 1>;

let isTwoAddress = 1 in {
def PANDNrr : PDI<0xDF, MRMSrcReg, (ops VR128:$dst, VR128:$src1, VR128:$src2),
                  "pandn {$src2, $dst|$dst, $src2}",
                  [(set VR128:$dst, (v2i64 (and (vnot VR128:$src1),
                                            VR128:$src2)))]>;

def PANDNrm : PDI<0xDF, MRMSrcMem, (ops VR128:$dst, VR128:$src1, i128mem:$src2),
                 "pandn {$src2, $dst|$dst, $src2}",
                 [(set VR128:$dst, (v2i64 (and (vnot VR128:$src1),
                                           (load addr:$src2))))]>;
}

// SSE2 Integer comparison
defm PCMPEQB : PDI_binop_rm_int<0x74, "pcmpeqb", int_x86_sse2_pcmpeq_b>;
defm PCMPEQW : PDI_binop_rm_int<0x75, "pcmpeqw", int_x86_sse2_pcmpeq_w>;
defm PCMPEQD : PDI_binop_rm_int<0x76, "pcmpeqd", int_x86_sse2_pcmpeq_d>;
defm PCMPGTB : PDI_binop_rm_int<0x64, "pcmpgtb", int_x86_sse2_pcmpgt_b>;
defm PCMPGTW : PDI_binop_rm_int<0x65, "pcmpgtw", int_x86_sse2_pcmpgt_w>;
defm PCMPGTD : PDI_binop_rm_int<0x66, "pcmpgtd", int_x86_sse2_pcmpgt_d>;

// Pack instructions
defm PACKSSWB : PDI_binop_rm_int<0x63, "packsswb", int_x86_sse2_packsswb_128>;
defm PACKSSDW : PDI_binop_rm_int<0x6B, "packssdw", int_x86_sse2_packssdw_128>;
defm PACKUSWB : PDI_binop_rm_int<0x67, "packuswb", int_x86_sse2_packuswb_128>;

// Shuffle and unpack instructions
def PSHUFDri : PDIi8<0x70, MRMSrcReg,
                     (ops VR128:$dst, VR128:$src1, i8imm:$src2),
                     "pshufd {$src2, $src1, $dst|$dst, $src1, $src2}",
                     [(set VR128:$dst, (v4i32 (vector_shuffle
                                               VR128:$src1, (undef),
                                               PSHUFD_shuffle_mask:$src2)))]>;
def PSHUFDmi : PDIi8<0x70, MRMSrcMem,
                     (ops VR128:$dst, i128mem:$src1, i8imm:$src2),
                     "pshufd {$src2, $src1, $dst|$dst, $src1, $src2}",
                     [(set VR128:$dst, (v4i32 (vector_shuffle
                                               (bc_v4i32(loadv2i64 addr:$src1)),
                                               (undef),
                                               PSHUFD_shuffle_mask:$src2)))]>;

// SSE2 with ImmT == Imm8 and XS prefix.
def PSHUFHWri : Ii8<0x70, MRMSrcReg,
                    (ops VR128:$dst, VR128:$src1, i8imm:$src2),
                    "pshufhw {$src2, $src1, $dst|$dst, $src1, $src2}",
                    [(set VR128:$dst, (v8i16 (vector_shuffle
                                              VR128:$src1, (undef),
                                              PSHUFHW_shuffle_mask:$src2)))]>,
                XS, Requires<[HasSSE2]>;
def PSHUFHWmi : Ii8<0x70, MRMSrcMem,
                    (ops VR128:$dst, i128mem:$src1, i8imm:$src2),
                    "pshufhw {$src2, $src1, $dst|$dst, $src1, $src2}",
                    [(set VR128:$dst, (v8i16 (vector_shuffle
                                              (bc_v8i16 (loadv2i64 addr:$src1)),
                                              (undef),
                                              PSHUFHW_shuffle_mask:$src2)))]>,
                XS, Requires<[HasSSE2]>;

// SSE2 with ImmT == Imm8 and XD prefix.
def PSHUFLWri : Ii8<0x70, MRMSrcReg,
                    (ops VR128:$dst, VR128:$src1, i32i8imm:$src2),
                    "pshuflw {$src2, $src1, $dst|$dst, $src1, $src2}",
                    [(set VR128:$dst, (v8i16 (vector_shuffle
                                              VR128:$src1, (undef),
                                              PSHUFLW_shuffle_mask:$src2)))]>,
                XD, Requires<[HasSSE2]>;
def PSHUFLWmi : Ii8<0x70, MRMSrcMem,
                    (ops VR128:$dst, i128mem:$src1, i32i8imm:$src2),
                    "pshuflw {$src2, $src1, $dst|$dst, $src1, $src2}",
                    [(set VR128:$dst, (v8i16 (vector_shuffle
                                              (bc_v8i16 (loadv2i64 addr:$src1)),
                                              (undef),
                                              PSHUFLW_shuffle_mask:$src2)))]>,
                XD, Requires<[HasSSE2]>;

let isTwoAddress = 1 in {
def PUNPCKLBWrr : PDI<0x60, MRMSrcReg, 
                      (ops VR128:$dst, VR128:$src1, VR128:$src2),
                      "punpcklbw {$src2, $dst|$dst, $src2}",
                      [(set VR128:$dst,
                        (v16i8 (vector_shuffle VR128:$src1, VR128:$src2,
                                UNPCKL_shuffle_mask)))]>;
def PUNPCKLBWrm : PDI<0x60, MRMSrcMem, 
                      (ops VR128:$dst, VR128:$src1, i128mem:$src2),
                      "punpcklbw {$src2, $dst|$dst, $src2}",
                      [(set VR128:$dst,
                        (v16i8 (vector_shuffle VR128:$src1,
                                (bc_v16i8 (loadv2i64 addr:$src2)),
                                UNPCKL_shuffle_mask)))]>;
def PUNPCKLWDrr : PDI<0x61, MRMSrcReg, 
                      (ops VR128:$dst, VR128:$src1, VR128:$src2),
                      "punpcklwd {$src2, $dst|$dst, $src2}",
                      [(set VR128:$dst,
                        (v8i16 (vector_shuffle VR128:$src1, VR128:$src2,
                                UNPCKL_shuffle_mask)))]>;
def PUNPCKLWDrm : PDI<0x61, MRMSrcMem, 
                      (ops VR128:$dst, VR128:$src1, i128mem:$src2),
                      "punpcklwd {$src2, $dst|$dst, $src2}",
                      [(set VR128:$dst,
                        (v8i16 (vector_shuffle VR128:$src1,
                                (bc_v8i16 (loadv2i64 addr:$src2)),
                                UNPCKL_shuffle_mask)))]>;
def PUNPCKLDQrr : PDI<0x62, MRMSrcReg, 
                      (ops VR128:$dst, VR128:$src1, VR128:$src2),
                      "punpckldq {$src2, $dst|$dst, $src2}",
                      [(set VR128:$dst,
                        (v4i32 (vector_shuffle VR128:$src1, VR128:$src2,
                                UNPCKL_shuffle_mask)))]>;
def PUNPCKLDQrm : PDI<0x62, MRMSrcMem, 
                      (ops VR128:$dst, VR128:$src1, i128mem:$src2),
                      "punpckldq {$src2, $dst|$dst, $src2}",
                      [(set VR128:$dst,
                        (v4i32 (vector_shuffle VR128:$src1,
                                (bc_v4i32 (loadv2i64 addr:$src2)),
                                UNPCKL_shuffle_mask)))]>;
def PUNPCKLQDQrr : PDI<0x6C, MRMSrcReg, 
                       (ops VR128:$dst, VR128:$src1, VR128:$src2),
                       "punpcklqdq {$src2, $dst|$dst, $src2}",
                      [(set VR128:$dst,
                        (v2i64 (vector_shuffle VR128:$src1, VR128:$src2,
                                UNPCKL_shuffle_mask)))]>;
def PUNPCKLQDQrm : PDI<0x6C, MRMSrcMem, 
                       (ops VR128:$dst, VR128:$src1, i128mem:$src2),
                       "punpcklqdq {$src2, $dst|$dst, $src2}",
                      [(set VR128:$dst,
                        (v2i64 (vector_shuffle VR128:$src1,
                                (loadv2i64 addr:$src2),
                                UNPCKL_shuffle_mask)))]>;

def PUNPCKHBWrr : PDI<0x68, MRMSrcReg, 
                      (ops VR128:$dst, VR128:$src1, VR128:$src2),
                      "punpckhbw {$src2, $dst|$dst, $src2}",
                      [(set VR128:$dst,
                        (v16i8 (vector_shuffle VR128:$src1, VR128:$src2,
                                UNPCKH_shuffle_mask)))]>;
def PUNPCKHBWrm : PDI<0x68, MRMSrcMem, 
                      (ops VR128:$dst, VR128:$src1, i128mem:$src2),
                      "punpckhbw {$src2, $dst|$dst, $src2}",
                      [(set VR128:$dst,
                        (v16i8 (vector_shuffle VR128:$src1,
                                (bc_v16i8 (loadv2i64 addr:$src2)),
                                UNPCKH_shuffle_mask)))]>;
def PUNPCKHWDrr : PDI<0x69, MRMSrcReg, 
                      (ops VR128:$dst, VR128:$src1, VR128:$src2),
                      "punpckhwd {$src2, $dst|$dst, $src2}",
                      [(set VR128:$dst,
                        (v8i16 (vector_shuffle VR128:$src1, VR128:$src2,
                                UNPCKH_shuffle_mask)))]>;
def PUNPCKHWDrm : PDI<0x69, MRMSrcMem, 
                      (ops VR128:$dst, VR128:$src1, i128mem:$src2),
                      "punpckhwd {$src2, $dst|$dst, $src2}",
                      [(set VR128:$dst,
                        (v8i16 (vector_shuffle VR128:$src1,
                                (bc_v8i16 (loadv2i64 addr:$src2)),
                                UNPCKH_shuffle_mask)))]>;
def PUNPCKHDQrr : PDI<0x6A, MRMSrcReg, 
                      (ops VR128:$dst, VR128:$src1, VR128:$src2),
                      "punpckhdq {$src2, $dst|$dst, $src2}",
                      [(set VR128:$dst,
                        (v4i32 (vector_shuffle VR128:$src1, VR128:$src2,
                                UNPCKH_shuffle_mask)))]>;
def PUNPCKHDQrm : PDI<0x6A, MRMSrcMem, 
                      (ops VR128:$dst, VR128:$src1, i128mem:$src2),
                      "punpckhdq {$src2, $dst|$dst, $src2}",
                      [(set VR128:$dst,
                        (v4i32 (vector_shuffle VR128:$src1,
                                (bc_v4i32 (loadv2i64 addr:$src2)),
                                UNPCKH_shuffle_mask)))]>;
def PUNPCKHQDQrr : PDI<0x6D, MRMSrcReg, 
                       (ops VR128:$dst, VR128:$src1, VR128:$src2),
                       "punpckhqdq {$src2, $dst|$dst, $src2}",
                      [(set VR128:$dst,
                        (v2i64 (vector_shuffle VR128:$src1, VR128:$src2,
                                UNPCKH_shuffle_mask)))]>;
def PUNPCKHQDQrm : PDI<0x6D, MRMSrcMem, 
                      (ops VR128:$dst, VR128:$src1, i128mem:$src2),
                      "punpckhqdq {$src2, $dst|$dst, $src2}",
                      [(set VR128:$dst,
                        (v2i64 (vector_shuffle VR128:$src1,
                                (loadv2i64 addr:$src2),
                                UNPCKH_shuffle_mask)))]>;
}

// Extract / Insert
def PEXTRWri : PDIi8<0xC5, MRMSrcReg,
                    (ops GR32:$dst, VR128:$src1, i32i8imm:$src2),
                    "pextrw {$src2, $src1, $dst|$dst, $src1, $src2}",
                    [(set GR32:$dst, (X86pextrw (v8i16 VR128:$src1),
                                     (iPTR imm:$src2)))]>;
let isTwoAddress = 1 in {
def PINSRWrri : PDIi8<0xC4, MRMSrcReg,
                     (ops VR128:$dst, VR128:$src1, GR32:$src2, i32i8imm:$src3),
                     "pinsrw {$src3, $src2, $dst|$dst, $src2, $src3}",
                     [(set VR128:$dst, (v8i16 (X86pinsrw (v8i16 VR128:$src1),
                                               GR32:$src2, (iPTR imm:$src3))))]>;
def PINSRWrmi : PDIi8<0xC4, MRMSrcMem,
                     (ops VR128:$dst, VR128:$src1, i16mem:$src2, i32i8imm:$src3),
                     "pinsrw {$src3, $src2, $dst|$dst, $src2, $src3}",
                     [(set VR128:$dst,
                       (v8i16 (X86pinsrw (v8i16 VR128:$src1),
                               (i32 (anyext (loadi16 addr:$src2))),
                               (iPTR imm:$src3))))]>;
}

//===----------------------------------------------------------------------===//
// Miscellaneous Instructions
//===----------------------------------------------------------------------===//

// Mask creation
def MOVMSKPSrr : PSI<0x50, MRMSrcReg, (ops GR32:$dst, VR128:$src),
                     "movmskps {$src, $dst|$dst, $src}",
                     [(set GR32:$dst, (int_x86_sse_movmsk_ps VR128:$src))]>;
def MOVMSKPDrr : PSI<0x50, MRMSrcReg, (ops GR32:$dst, VR128:$src),
                     "movmskpd {$src, $dst|$dst, $src}",
                     [(set GR32:$dst, (int_x86_sse2_movmsk_pd VR128:$src))]>;

def PMOVMSKBrr : PDI<0xD7, MRMSrcReg, (ops GR32:$dst, VR128:$src),
                     "pmovmskb {$src, $dst|$dst, $src}",
                     [(set GR32:$dst, (int_x86_sse2_pmovmskb_128 VR128:$src))]>;

// Conditional store
def MASKMOVDQU : PDI<0xF7, MRMSrcReg, (ops VR128:$src, VR128:$mask),
                     "maskmovdqu {$mask, $src|$src, $mask}",
                     [(int_x86_sse2_maskmov_dqu VR128:$src, VR128:$mask, EDI)]>,
                 Imp<[EDI],[]>;

// Prefetching loads.
// TODO: no intrinsics for these?
def PREFETCHT0   : PSI<0x18, MRM1m, (ops i8mem:$src), "prefetcht0 $src", []>;
def PREFETCHT1   : PSI<0x18, MRM2m, (ops i8mem:$src), "prefetcht1 $src", []>;
def PREFETCHT2   : PSI<0x18, MRM3m, (ops i8mem:$src), "prefetcht2 $src", []>;
def PREFETCHTNTA : PSI<0x18, MRM0m, (ops i8mem:$src), "prefetchtnta $src", []>;

// Non-temporal stores
def MOVNTPSmr : PSI<0x2B, MRMDestMem, (ops i128mem:$dst, VR128:$src),
                    "movntps {$src, $dst|$dst, $src}",
                    [(int_x86_sse_movnt_ps addr:$dst, VR128:$src)]>;
def MOVNTPDmr : PDI<0x2B, MRMDestMem, (ops i128mem:$dst, VR128:$src),
                    "movntpd {$src, $dst|$dst, $src}",
                    [(int_x86_sse2_movnt_pd addr:$dst, VR128:$src)]>;
def MOVNTDQmr : PDI<0xE7, MRMDestMem, (ops f128mem:$dst, VR128:$src),
                    "movntdq {$src, $dst|$dst, $src}",
                    [(int_x86_sse2_movnt_dq addr:$dst, VR128:$src)]>;
def MOVNTImr  :   I<0xC3, MRMDestMem, (ops i32mem:$dst, GR32:$src),
                    "movnti {$src, $dst|$dst, $src}",
                    [(int_x86_sse2_movnt_i addr:$dst, GR32:$src)]>, 
                  TB, Requires<[HasSSE2]>;

// Flush cache
def CLFLUSH : I<0xAE, MRM7m, (ops i8mem:$src),
               "clflush $src", [(int_x86_sse2_clflush addr:$src)]>,
              TB, Requires<[HasSSE2]>;

// Load, store, and memory fence
def SFENCE : PSI<0xAE, MRM7m, (ops), "sfence", [(int_x86_sse_sfence)]>;
def LFENCE : I<0xAE, MRM5m, (ops),
               "lfence", [(int_x86_sse2_lfence)]>, TB, Requires<[HasSSE2]>;
def MFENCE : I<0xAE, MRM6m, (ops),
               "mfence", [(int_x86_sse2_mfence)]>, TB, Requires<[HasSSE2]>;

// MXCSR register
def LDMXCSR : PSI<0xAE, MRM2m, (ops i32mem:$src),
                  "ldmxcsr $src", [(int_x86_sse_ldmxcsr addr:$src)]>;
def STMXCSR : PSI<0xAE, MRM3m, (ops i32mem:$dst),
                  "stmxcsr $dst", [(int_x86_sse_stmxcsr addr:$dst)]>;

// Thread synchronization
def MONITOR : I<0xC8, RawFrm, (ops), "monitor",
                [(int_x86_sse3_monitor EAX, ECX, EDX)]>,TB, Requires<[HasSSE3]>;
def MWAIT   : I<0xC9, RawFrm, (ops), "mwait",
                [(int_x86_sse3_mwait ECX, EAX)]>, TB, Requires<[HasSSE3]>;

//===----------------------------------------------------------------------===//
// Alias Instructions
//===----------------------------------------------------------------------===//

// Alias instructions that map zero vector to pxor / xorp* for sse.
// FIXME: remove when we can teach regalloc that xor reg, reg is ok.
def V_SET0 : PSI<0x57, MRMInitReg, (ops VR128:$dst),
                 "xorps $dst, $dst",
                 [(set VR128:$dst, (v4f32 immAllZerosV))]>;

def V_SETALLONES : PDI<0x76, MRMInitReg, (ops VR128:$dst),
                       "pcmpeqd $dst, $dst",
                       [(set VR128:$dst, (v2f64 immAllOnesV))]>;

// FR32 / FR64 to 128-bit vector conversion.
def MOVSS2PSrr : SSI<0x10, MRMSrcReg, (ops VR128:$dst, FR32:$src),
                      "movss {$src, $dst|$dst, $src}",
                      [(set VR128:$dst,
                        (v4f32 (scalar_to_vector FR32:$src)))]>;
def MOVSS2PSrm : SSI<0x10, MRMSrcMem, (ops VR128:$dst, f32mem:$src),
                     "movss {$src, $dst|$dst, $src}",
                     [(set VR128:$dst,
                       (v4f32 (scalar_to_vector (loadf32 addr:$src))))]>;
def MOVSD2PDrr : SDI<0x10, MRMSrcReg, (ops VR128:$dst, FR64:$src),
                      "movsd {$src, $dst|$dst, $src}",
                      [(set VR128:$dst,
                        (v2f64 (scalar_to_vector FR64:$src)))]>;
def MOVSD2PDrm : SDI<0x10, MRMSrcMem, (ops VR128:$dst, f64mem:$src),
                     "movsd {$src, $dst|$dst, $src}",
                     [(set VR128:$dst, 
                       (v2f64 (scalar_to_vector (loadf64 addr:$src))))]>;

def MOVDI2PDIrr : PDI<0x6E, MRMSrcReg, (ops VR128:$dst, GR32:$src),
                      "movd {$src, $dst|$dst, $src}",
                      [(set VR128:$dst,
                        (v4i32 (scalar_to_vector GR32:$src)))]>;
def MOVDI2PDIrm : PDI<0x6E, MRMSrcMem, (ops VR128:$dst, i32mem:$src),
                      "movd {$src, $dst|$dst, $src}",
                      [(set VR128:$dst,
                        (v4i32 (scalar_to_vector (loadi32 addr:$src))))]>;
// SSE2 instructions with XS prefix
def MOVQI2PQIrr : I<0x7E, MRMSrcReg, (ops VR128:$dst, VR64:$src),
                    "movq {$src, $dst|$dst, $src}",
                    [(set VR128:$dst,
                      (v2i64 (scalar_to_vector VR64:$src)))]>, XS,
                  Requires<[HasSSE2]>;
def MOVQI2PQIrm : I<0x7E, MRMSrcMem, (ops VR128:$dst, i64mem:$src),
                    "movq {$src, $dst|$dst, $src}",
                    [(set VR128:$dst,
                      (v2i64 (scalar_to_vector (loadi64 addr:$src))))]>, XS,
                  Requires<[HasSSE2]>;
// FIXME: may not be able to eliminate this movss with coalescing the src and
// dest register classes are different. We really want to write this pattern
// like this:
// def : Pat<(f32 (vector_extract (v4f32 VR128:$src), (iPTR 0))),
//           (f32 FR32:$src)>;
def MOVPS2SSrr : SSI<0x10, MRMSrcReg, (ops FR32:$dst, VR128:$src),
                     "movss {$src, $dst|$dst, $src}",
                     [(set FR32:$dst, (vector_extract (v4f32 VR128:$src),
                                       (iPTR 0)))]>;
def MOVPS2SSmr : SSI<0x11, MRMDestMem, (ops f32mem:$dst, VR128:$src),
                     "movss {$src, $dst|$dst, $src}",
                     [(store (f32 (vector_extract (v4f32 VR128:$src),
                                   (iPTR 0))), addr:$dst)]>;
def MOVPD2SDrr : SDI<0x10, MRMSrcReg, (ops FR64:$dst, VR128:$src),
                     "movsd {$src, $dst|$dst, $src}",
                     [(set FR64:$dst, (vector_extract (v2f64 VR128:$src),
                                       (iPTR 0)))]>;
def MOVPD2SDmr : SDI<0x11, MRMDestMem, (ops f64mem:$dst, VR128:$src),
                     "movsd {$src, $dst|$dst, $src}",
                     [(store (f64 (vector_extract (v2f64 VR128:$src),
                                   (iPTR 0))), addr:$dst)]>;
def MOVPDI2DIrr  : PDI<0x7E, MRMDestReg, (ops GR32:$dst, VR128:$src),
                       "movd {$src, $dst|$dst, $src}",
                       [(set GR32:$dst, (vector_extract (v4i32 VR128:$src),
                                        (iPTR 0)))]>;
def MOVPDI2DImr  : PDI<0x7E, MRMDestMem, (ops i32mem:$dst, VR128:$src),
                       "movd {$src, $dst|$dst, $src}",
                       [(store (i32 (vector_extract (v4i32 VR128:$src),
                                     (iPTR 0))), addr:$dst)]>;

// Move to lower bits of a VR128, leaving upper bits alone.
// Three operand (but two address) aliases.
let isTwoAddress = 1 in {
def MOVLSS2PSrr : SSI<0x10, MRMSrcReg, (ops VR128:$dst, VR128:$src1, FR32:$src2),
                      "movss {$src2, $dst|$dst, $src2}", []>;
def MOVLSD2PDrr : SDI<0x10, MRMSrcReg, (ops VR128:$dst, VR128:$src1, FR64:$src2),
                      "movsd {$src2, $dst|$dst, $src2}", []>;

let AddedComplexity = 15 in {
def MOVLPSrr : SSI<0x10, MRMSrcReg, (ops VR128:$dst, VR128:$src1, VR128:$src2),
                      "movss {$src2, $dst|$dst, $src2}",
                   [(set VR128:$dst,
                     (v4f32 (vector_shuffle VR128:$src1, VR128:$src2,
                             MOVL_shuffle_mask)))]>;
def MOVLPDrr : SDI<0x10, MRMSrcReg, (ops VR128:$dst, VR128:$src1, VR128:$src2),
                      "movsd {$src2, $dst|$dst, $src2}",
                   [(set VR128:$dst,
                     (v2f64 (vector_shuffle VR128:$src1, VR128:$src2,
                             MOVL_shuffle_mask)))]>;
}
}

// Store / copy lower 64-bits of a XMM register.
def MOVLQ128mr : PDI<0xD6, MRMDestMem, (ops i64mem:$dst, VR128:$src),
                     "movq {$src, $dst|$dst, $src}",
                     [(int_x86_sse2_storel_dq addr:$dst, VR128:$src)]>;

// Move to lower bits of a VR128 and zeroing upper bits.
// Loading from memory automatically zeroing upper bits.
let AddedComplexity = 20 in {
def MOVZSS2PSrm : SSI<0x10, MRMSrcMem, (ops VR128:$dst, f32mem:$src),
                      "movss {$src, $dst|$dst, $src}",
                      [(set VR128:$dst, (v4f32 (vector_shuffle immAllZerosV,
                                 (v4f32 (scalar_to_vector (loadf32 addr:$src))),
                                                MOVL_shuffle_mask)))]>;
def MOVZSD2PDrm : SDI<0x10, MRMSrcMem, (ops VR128:$dst, f64mem:$src),
                      "movsd {$src, $dst|$dst, $src}",
                      [(set VR128:$dst, (v2f64 (vector_shuffle immAllZerosV,
                                 (v2f64 (scalar_to_vector (loadf64 addr:$src))),
                                                MOVL_shuffle_mask)))]>;
}
let AddedComplexity = 15 in
// movd / movq to XMM register zero-extends
def MOVZDI2PDIrr : PDI<0x6E, MRMSrcReg, (ops VR128:$dst, GR32:$src),
                       "movd {$src, $dst|$dst, $src}",
                      [(set VR128:$dst, (v4i32 (vector_shuffle immAllZerosV,
                                           (v4i32 (scalar_to_vector GR32:$src)),
                                                MOVL_shuffle_mask)))]>;
let AddedComplexity = 20 in
def MOVZDI2PDIrm : PDI<0x6E, MRMSrcMem, (ops VR128:$dst, i32mem:$src),
                       "movd {$src, $dst|$dst, $src}",
                      [(set VR128:$dst, (v4i32 (vector_shuffle immAllZerosV,
                                 (v4i32 (scalar_to_vector (loadi32 addr:$src))),
                                                MOVL_shuffle_mask)))]>;
// Moving from XMM to XMM but still clear upper 64 bits.
let AddedComplexity = 15 in
def MOVZQI2PQIrr : I<0x7E, MRMSrcReg, (ops VR128:$dst, VR128:$src),
                     "movq {$src, $dst|$dst, $src}",
                     [(set VR128:$dst, (int_x86_sse2_movl_dq VR128:$src))]>,
                   XS, Requires<[HasSSE2]>;
let AddedComplexity = 20 in
def MOVZQI2PQIrm : I<0x7E, MRMSrcMem, (ops VR128:$dst, i64mem:$src),
                     "movq {$src, $dst|$dst, $src}",
                   [(set VR128:$dst, (int_x86_sse2_movl_dq
                                      (bitconvert (loadv2i64 addr:$src))))]>,
                   XS, Requires<[HasSSE2]>;

//===----------------------------------------------------------------------===//
// Non-Instruction Patterns
//===----------------------------------------------------------------------===//

// 128-bit vector undef's.
def : Pat<(v2f64 (undef)), (IMPLICIT_DEF_VR128)>, Requires<[HasSSE2]>;
def : Pat<(v16i8 (undef)), (IMPLICIT_DEF_VR128)>, Requires<[HasSSE2]>;
def : Pat<(v8i16 (undef)), (IMPLICIT_DEF_VR128)>, Requires<[HasSSE2]>;
def : Pat<(v4i32 (undef)), (IMPLICIT_DEF_VR128)>, Requires<[HasSSE2]>;
def : Pat<(v2i64 (undef)), (IMPLICIT_DEF_VR128)>, Requires<[HasSSE2]>;

// 128-bit vector all zero's.
def : Pat<(v16i8 immAllZerosV), (V_SET0)>, Requires<[HasSSE2]>;
def : Pat<(v8i16 immAllZerosV), (V_SET0)>, Requires<[HasSSE2]>;
def : Pat<(v4i32 immAllZerosV), (V_SET0)>, Requires<[HasSSE2]>;
def : Pat<(v2i64 immAllZerosV), (V_SET0)>, Requires<[HasSSE2]>;
def : Pat<(v2f64 immAllZerosV), (V_SET0)>, Requires<[HasSSE2]>;

// 128-bit vector all one's.
def : Pat<(v16i8 immAllOnesV), (V_SETALLONES)>, Requires<[HasSSE2]>;
def : Pat<(v8i16 immAllOnesV), (V_SETALLONES)>, Requires<[HasSSE2]>;
def : Pat<(v4i32 immAllOnesV), (V_SETALLONES)>, Requires<[HasSSE2]>;
def : Pat<(v2i64 immAllOnesV), (V_SETALLONES)>, Requires<[HasSSE2]>;
def : Pat<(v4f32 immAllOnesV), (V_SETALLONES)>, Requires<[HasSSE1]>;

// Store 128-bit integer vector values.
def : Pat<(store (v16i8 VR128:$src), addr:$dst),
          (MOVDQAmr addr:$dst, VR128:$src)>, Requires<[HasSSE2]>;
def : Pat<(store (v8i16 VR128:$src), addr:$dst),
          (MOVDQAmr addr:$dst, VR128:$src)>, Requires<[HasSSE2]>;
def : Pat<(store (v4i32 VR128:$src), addr:$dst),
          (MOVDQAmr addr:$dst, VR128:$src)>, Requires<[HasSSE2]>;

// Scalar to v8i16 / v16i8. The source may be a GR32, but only the lower 8 or
// 16-bits matter.
def : Pat<(v8i16 (X86s2vec GR32:$src)), (MOVDI2PDIrr GR32:$src)>,
      Requires<[HasSSE2]>;
def : Pat<(v16i8 (X86s2vec GR32:$src)), (MOVDI2PDIrr GR32:$src)>,
      Requires<[HasSSE2]>;

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
def : Pat<(v8i16 (vector_shuffle immAllZerosV,
                  (v8i16 (X86s2vec GR32:$src)), MOVL_shuffle_mask)),
          (MOVZDI2PDIrr GR32:$src)>, Requires<[HasSSE2]>;
def : Pat<(v16i8 (vector_shuffle immAllZerosV,
                  (v16i8 (X86s2vec GR32:$src)), MOVL_shuffle_mask)),
          (MOVZDI2PDIrr GR32:$src)>, Requires<[HasSSE2]>;
// Zeroing a VR128 then do a MOVS{S|D} to the lower bits.
def : Pat<(v2f64 (vector_shuffle immAllZerosV,
                  (v2f64 (scalar_to_vector FR64:$src)), MOVL_shuffle_mask)),
          (MOVLSD2PDrr (V_SET0), FR64:$src)>, Requires<[HasSSE2]>;
def : Pat<(v4f32 (vector_shuffle immAllZerosV,
                  (v4f32 (scalar_to_vector FR32:$src)), MOVL_shuffle_mask)),
          (MOVLSS2PSrr (V_SET0), FR32:$src)>, Requires<[HasSSE2]>;
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

// Splat v4f32
def : Pat<(vector_shuffle (v4f32 VR128:$src), (undef), SSE_splat_mask:$sm),
          (SHUFPSrri VR128:$src, VR128:$src, SSE_splat_mask:$sm)>,
      Requires<[HasSSE1]>;

// Special unary SHUFPSrri case.
// FIXME: when we want non two-address code, then we should use PSHUFD?
def : Pat<(vector_shuffle (v4f32 VR128:$src1), (undef),
           SHUFP_unary_shuffle_mask:$sm),
          (SHUFPSrri VR128:$src1, VR128:$src1, SHUFP_unary_shuffle_mask:$sm)>,
      Requires<[HasSSE1]>;
// Unary v4f32 shuffle with PSHUF* in order to fold a load.
def : Pat<(vector_shuffle (loadv4f32 addr:$src1), (undef),
           SHUFP_unary_shuffle_mask:$sm),
          (PSHUFDmi addr:$src1, SHUFP_unary_shuffle_mask:$sm)>,
      Requires<[HasSSE2]>;
// Special binary v4i32 shuffle cases with SHUFPS.
def : Pat<(vector_shuffle (v4i32 VR128:$src1), (v4i32 VR128:$src2),
           PSHUFD_binary_shuffle_mask:$sm),
          (SHUFPSrri VR128:$src1, VR128:$src2, PSHUFD_binary_shuffle_mask:$sm)>,
           Requires<[HasSSE2]>;
def : Pat<(vector_shuffle (v4i32 VR128:$src1),
           (bc_v4i32 (loadv2i64 addr:$src2)), PSHUFD_binary_shuffle_mask:$sm),
          (SHUFPSrmi VR128:$src1, addr:$src2, PSHUFD_binary_shuffle_mask:$sm)>,
           Requires<[HasSSE2]>;

// vector_shuffle v1, <undef>, <0, 0, 1, 1, ...>
let AddedComplexity = 10 in {
def : Pat<(v4f32 (vector_shuffle VR128:$src, (undef),
                  UNPCKL_v_undef_shuffle_mask)),
          (UNPCKLPSrr VR128:$src, VR128:$src)>, Requires<[HasSSE2]>;
def : Pat<(v16i8 (vector_shuffle VR128:$src, (undef),
                  UNPCKL_v_undef_shuffle_mask)),
          (PUNPCKLBWrr VR128:$src, VR128:$src)>, Requires<[HasSSE2]>;
def : Pat<(v8i16 (vector_shuffle VR128:$src, (undef),
                  UNPCKL_v_undef_shuffle_mask)),
          (PUNPCKLWDrr VR128:$src, VR128:$src)>, Requires<[HasSSE2]>;
def : Pat<(v4i32 (vector_shuffle VR128:$src, (undef),
                  UNPCKL_v_undef_shuffle_mask)),
          (PUNPCKLDQrr VR128:$src, VR128:$src)>, Requires<[HasSSE1]>;
}

let AddedComplexity = 15 in
// vector_shuffle v1, <undef> <1, 1, 3, 3>
def : Pat<(v4i32 (vector_shuffle VR128:$src, (undef),
                  MOVSHDUP_shuffle_mask)),
          (MOVSHDUPrr VR128:$src)>, Requires<[HasSSE3]>;
let AddedComplexity = 20 in
def : Pat<(v4i32 (vector_shuffle (bc_v4i32 (loadv2i64 addr:$src)), (undef),
                  MOVSHDUP_shuffle_mask)),
          (MOVSHDUPrm addr:$src)>, Requires<[HasSSE3]>;

// vector_shuffle v1, <undef> <0, 0, 2, 2>
let AddedComplexity = 15 in
def : Pat<(v4i32 (vector_shuffle VR128:$src, (undef),
                  MOVSLDUP_shuffle_mask)),
          (MOVSLDUPrr VR128:$src)>, Requires<[HasSSE3]>;
let AddedComplexity = 20 in
def : Pat<(v4i32 (vector_shuffle (bc_v4i32 (loadv2i64 addr:$src)), (undef),
                  MOVSLDUP_shuffle_mask)),
          (MOVSLDUPrm addr:$src)>, Requires<[HasSSE3]>;

let AddedComplexity = 15 in {
// vector_shuffle v1, v2 <0, 1, 4, 5> using MOVLHPS
def : Pat<(v4i32 (vector_shuffle VR128:$src1, VR128:$src2,
                  MOVHP_shuffle_mask)),
          (MOVLHPSrr VR128:$src1, VR128:$src2)>;

// vector_shuffle v1, v2 <6, 7, 2, 3> using MOVHLPS
def : Pat<(v4i32 (vector_shuffle VR128:$src1, VR128:$src2,
                  MOVHLPS_shuffle_mask)),
          (MOVHLPSrr VR128:$src1, VR128:$src2)>;

// vector_shuffle v1, undef <2, 3, ?, ?> using MOVHLPS
def : Pat<(v4f32 (vector_shuffle VR128:$src1, (undef),
                  UNPCKH_shuffle_mask)),
          (MOVHLPSrr VR128:$src1, VR128:$src1)>;
def : Pat<(v4i32 (vector_shuffle VR128:$src1, (undef),
                  UNPCKH_shuffle_mask)),
          (MOVHLPSrr VR128:$src1, VR128:$src1)>;
}

  let AddedComplexity = 20 in {
// vector_shuffle v1, (load v2) <4, 5, 2, 3> using MOVLPS
// vector_shuffle v1, (load v2) <0, 1, 4, 5> using MOVHPS
def : Pat<(v4f32 (vector_shuffle VR128:$src1, (loadv4f32 addr:$src2),
                  MOVLP_shuffle_mask)),
          (MOVLPSrm VR128:$src1, addr:$src2)>, Requires<[HasSSE1]>;
def : Pat<(v2f64 (vector_shuffle VR128:$src1, (loadv2f64 addr:$src2),
                  MOVLP_shuffle_mask)),
          (MOVLPDrm VR128:$src1, addr:$src2)>, Requires<[HasSSE2]>;
def : Pat<(v4f32 (vector_shuffle VR128:$src1, (loadv4f32 addr:$src2),
                  MOVHP_shuffle_mask)),
          (MOVHPSrm VR128:$src1, addr:$src2)>, Requires<[HasSSE1]>;
def : Pat<(v2f64 (vector_shuffle VR128:$src1, (loadv2f64 addr:$src2),
                  MOVHP_shuffle_mask)),
          (MOVHPDrm VR128:$src1, addr:$src2)>, Requires<[HasSSE2]>;

def : Pat<(v4i32 (vector_shuffle VR128:$src1, (bc_v4i32 (loadv2i64 addr:$src2)),
                  MOVLP_shuffle_mask)),
          (MOVLPSrm VR128:$src1, addr:$src2)>, Requires<[HasSSE2]>;
def : Pat<(v2i64 (vector_shuffle VR128:$src1, (loadv2i64 addr:$src2),
                  MOVLP_shuffle_mask)),
          (MOVLPDrm VR128:$src1, addr:$src2)>, Requires<[HasSSE2]>;
def : Pat<(v4i32 (vector_shuffle VR128:$src1, (bc_v4i32 (loadv2i64 addr:$src2)),
                  MOVHP_shuffle_mask)),
          (MOVHPSrm VR128:$src1, addr:$src2)>, Requires<[HasSSE1]>;
def : Pat<(v2i64 (vector_shuffle VR128:$src1, (loadv2i64 addr:$src2),
                  MOVLP_shuffle_mask)),
          (MOVLPDrm VR128:$src1, addr:$src2)>, Requires<[HasSSE2]>;
}

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
let AddedComplexity = 20 in
def : Pat<(bc_v2i64 (vector_shuffle immAllZerosV,
                     (v2f64 (scalar_to_vector (loadf64 addr:$src))),
                     MOVL_shuffle_mask)),
          (MOVZQI2PQIrm addr:$src)>, Requires<[HasSSE2]>;

// FIXME: Temporary workaround since 2-wide shuffle is broken.
def : Pat<(int_x86_sse2_movs_d  VR128:$src1, VR128:$src2),
          (v2f64 (MOVLPDrr VR128:$src1, VR128:$src2))>, Requires<[HasSSE2]>;
def : Pat<(int_x86_sse2_loadh_pd VR128:$src1, addr:$src2),
          (v2f64 (MOVHPDrm VR128:$src1, addr:$src2))>, Requires<[HasSSE2]>;
def : Pat<(int_x86_sse2_loadl_pd VR128:$src1, addr:$src2),
          (v2f64 (MOVLPDrm VR128:$src1, addr:$src2))>, Requires<[HasSSE2]>;
def : Pat<(int_x86_sse2_shuf_pd VR128:$src1, VR128:$src2, imm:$src3),
          (v2f64 (SHUFPDrri VR128:$src1, VR128:$src2, imm:$src3))>,
      Requires<[HasSSE2]>;
def : Pat<(int_x86_sse2_shuf_pd VR128:$src1, (load addr:$src2), imm:$src3),
          (v2f64 (SHUFPDrmi VR128:$src1, addr:$src2, imm:$src3))>,
      Requires<[HasSSE2]>;
def : Pat<(int_x86_sse2_unpckh_pd VR128:$src1, VR128:$src2),
          (v2f64 (UNPCKHPDrr VR128:$src1, VR128:$src2))>, Requires<[HasSSE2]>;
def : Pat<(int_x86_sse2_unpckh_pd VR128:$src1, (load addr:$src2)),
          (v2f64 (UNPCKHPDrm VR128:$src1, addr:$src2))>, Requires<[HasSSE2]>;
def : Pat<(int_x86_sse2_unpckl_pd VR128:$src1, VR128:$src2),
          (v2f64 (UNPCKLPDrr VR128:$src1, VR128:$src2))>, Requires<[HasSSE2]>;
def : Pat<(int_x86_sse2_unpckl_pd VR128:$src1, (load addr:$src2)),
          (v2f64 (UNPCKLPDrm VR128:$src1, addr:$src2))>, Requires<[HasSSE2]>;
def : Pat<(int_x86_sse2_punpckh_qdq VR128:$src1, VR128:$src2),
          (v2i64 (PUNPCKHQDQrr VR128:$src1, VR128:$src2))>, Requires<[HasSSE2]>;
def : Pat<(int_x86_sse2_punpckh_qdq VR128:$src1, (load addr:$src2)),
          (v2i64 (PUNPCKHQDQrm VR128:$src1, addr:$src2))>, Requires<[HasSSE2]>;
def : Pat<(int_x86_sse2_punpckl_qdq VR128:$src1, VR128:$src2),
          (v2i64 (PUNPCKLQDQrr VR128:$src1, VR128:$src2))>, Requires<[HasSSE2]>;
def : Pat<(int_x86_sse2_punpckl_qdq VR128:$src1, (load addr:$src2)),
          (PUNPCKLQDQrm VR128:$src1, addr:$src2)>, Requires<[HasSSE2]>;

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
                  (load addr:$src2))),
          (PANDNrm VR128:$src1, addr:$src2)>, Requires<[HasSSE2]>;
def : Pat<(v2i64 (and (xor VR128:$src1, (bc_v2i64 (v8i16 immAllOnesV))),
                  (load addr:$src2))),
          (PANDNrm VR128:$src1, addr:$src2)>, Requires<[HasSSE2]>;
def : Pat<(v2i64 (and (xor VR128:$src1, (bc_v2i64 (v16i8 immAllOnesV))),
                  (load addr:$src2))),
          (PANDNrm VR128:$src1, addr:$src2)>, Requires<[HasSSE2]>;

// Unaligned load
def : Pat<(v4f32 (X86loadu addr:$src)), (MOVUPSrm addr:$src)>,
      Requires<[HasSSE1]>;

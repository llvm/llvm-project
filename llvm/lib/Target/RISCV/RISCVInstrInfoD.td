//===-- RISCVInstrInfoD.td - RISC-V 'D' instructions -------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file describes the RISC-V instructions from the standard 'D',
// Double-Precision Floating-Point instruction set extension.
//
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// RISC-V specific DAG Nodes.
//===----------------------------------------------------------------------===//

def SDT_RISCVBuildPairF64 : SDTypeProfile<1, 2, [SDTCisVT<0, f64>,
                                                 SDTCisVT<1, i32>,
                                                 SDTCisSameAs<1, 2>]>;
def SDT_RISCVSplitF64     : SDTypeProfile<2, 1, [SDTCisVT<0, i32>,
                                                 SDTCisVT<1, i32>,
                                                 SDTCisVT<2, f64>]>;

def RISCVBuildPairF64 : SDNode<"RISCVISD::BuildPairF64", SDT_RISCVBuildPairF64>;
def RISCVSplitF64     : SDNode<"RISCVISD::SplitF64", SDT_RISCVSplitF64>;

//===----------------------------------------------------------------------===//
// Instructions
//===----------------------------------------------------------------------===//

let Predicates = [HasStdExtD] in {
def FLD : FPLoad_r<0b011, "fld", FPR64, WriteFLD64>;

// Operands for stores are in the order srcreg, base, offset rather than
// reflecting the order these fields are specified in the instruction
// encoding.
def FSD : FPStore_r<0b011, "fsd", FPR64, WriteFST64>;

let SchedRW = [WriteFMA64, ReadFMA64, ReadFMA64, ReadFMA64] in {
def FMADD_D  : FPFMA_rrr_frm<OPC_MADD,  0b01, "fmadd.d",  FPR64>;
def FMSUB_D  : FPFMA_rrr_frm<OPC_MSUB,  0b01, "fmsub.d",  FPR64>;
def FNMSUB_D : FPFMA_rrr_frm<OPC_NMSUB, 0b01, "fnmsub.d", FPR64>;
def FNMADD_D : FPFMA_rrr_frm<OPC_NMADD, 0b01, "fnmadd.d", FPR64>;
}

def : FPFMADynFrmAlias<FMADD_D,  "fmadd.d",  FPR64>;
def : FPFMADynFrmAlias<FMSUB_D,  "fmsub.d",  FPR64>;
def : FPFMADynFrmAlias<FNMSUB_D, "fnmsub.d", FPR64>;
def : FPFMADynFrmAlias<FNMADD_D, "fnmadd.d", FPR64>;

def FADD_D : FPALU_rr_frm<0b0000001, "fadd.d", FPR64>,
             Sched<[WriteFALU64, ReadFALU64, ReadFALU64]>;
def FSUB_D : FPALU_rr_frm<0b0000101, "fsub.d", FPR64>,
             Sched<[WriteFALU64, ReadFALU64, ReadFALU64]>;
def FMUL_D : FPALU_rr_frm<0b0001001, "fmul.d", FPR64>,
             Sched<[WriteFMul64, ReadFMul64, ReadFMul64]>;
def FDIV_D : FPALU_rr_frm<0b0001101, "fdiv.d", FPR64>,
             Sched<[WriteFDiv64, ReadFDiv64, ReadFDiv64]>;

def        : FPALUDynFrmAlias<FADD_D, "fadd.d", FPR64>;
def        : FPALUDynFrmAlias<FSUB_D, "fsub.d", FPR64>;
def        : FPALUDynFrmAlias<FMUL_D, "fmul.d", FPR64>;
def        : FPALUDynFrmAlias<FDIV_D, "fdiv.d", FPR64>;

def FSQRT_D : FPUnaryOp_r_frm<0b0101101, 0b00000, FPR64, FPR64, "fsqrt.d">,
              Sched<[WriteFSqrt64, ReadFSqrt64]>;
def         : FPUnaryOpDynFrmAlias<FSQRT_D, "fsqrt.d", FPR64, FPR64>;

let SchedRW = [WriteFSGNJ64, ReadFSGNJ64, ReadFSGNJ64],
    mayRaiseFPException = 0 in {
def FSGNJ_D  : FPALU_rr<0b0010001, 0b000, "fsgnj.d", FPR64>;
def FSGNJN_D : FPALU_rr<0b0010001, 0b001, "fsgnjn.d", FPR64>;
def FSGNJX_D : FPALU_rr<0b0010001, 0b010, "fsgnjx.d", FPR64>;
}

let SchedRW = [WriteFMinMax64, ReadFMinMax64, ReadFMinMax64] in {
def FMIN_D   : FPALU_rr<0b0010101, 0b000, "fmin.d", FPR64>;
def FMAX_D   : FPALU_rr<0b0010101, 0b001, "fmax.d", FPR64>;
}

def FCVT_S_D : FPUnaryOp_r_frm<0b0100000, 0b00001, FPR32, FPR64, "fcvt.s.d">,
               Sched<[WriteFCvtF64ToF32, ReadFCvtF64ToF32]>;
def          : FPUnaryOpDynFrmAlias<FCVT_S_D, "fcvt.s.d", FPR32, FPR64>;

def FCVT_D_S : FPUnaryOp_r<0b0100001, 0b00000, 0b000, FPR64, FPR32, "fcvt.d.s">,
               Sched<[WriteFCvtF32ToF64, ReadFCvtF32ToF64]>;

let SchedRW = [WriteFCmp64, ReadFCmp64, ReadFCmp64] in {
def FEQ_D : FPCmp_rr<0b1010001, 0b010, "feq.d", FPR64>;
def FLT_D : FPCmp_rr<0b1010001, 0b001, "flt.d", FPR64>;
def FLE_D : FPCmp_rr<0b1010001, 0b000, "fle.d", FPR64>;
}

let mayRaiseFPException = 0 in
def FCLASS_D : FPUnaryOp_r<0b1110001, 0b00000, 0b001, GPR, FPR64, "fclass.d">,
               Sched<[WriteFClass64, ReadFClass64]>;

def FCVT_W_D : FPUnaryOp_r_frm<0b1100001, 0b00000, GPR, FPR64, "fcvt.w.d">,
               Sched<[WriteFCvtF64ToI32, ReadFCvtF64ToI32]>;
def          : FPUnaryOpDynFrmAlias<FCVT_W_D, "fcvt.w.d", GPR, FPR64>;

def FCVT_WU_D : FPUnaryOp_r_frm<0b1100001, 0b00001, GPR, FPR64, "fcvt.wu.d">,
                Sched<[WriteFCvtF64ToI32, ReadFCvtF64ToI32]>;
def           : FPUnaryOpDynFrmAlias<FCVT_WU_D, "fcvt.wu.d", GPR, FPR64>;

def FCVT_D_W : FPUnaryOp_r<0b1101001, 0b00000, 0b000, FPR64, GPR, "fcvt.d.w">,
               Sched<[WriteFCvtI32ToF64, ReadFCvtI32ToF64]>;

def FCVT_D_WU : FPUnaryOp_r<0b1101001, 0b00001, 0b000, FPR64, GPR, "fcvt.d.wu">,
                Sched<[WriteFCvtI32ToF64, ReadFCvtI32ToF64]>;
} // Predicates = [HasStdExtD]

let Predicates = [HasStdExtD, IsRV64] in {
def FCVT_L_D : FPUnaryOp_r_frm<0b1100001, 0b00010, GPR, FPR64, "fcvt.l.d">,
               Sched<[WriteFCvtF64ToI64, ReadFCvtF64ToI64]>;
def          : FPUnaryOpDynFrmAlias<FCVT_L_D, "fcvt.l.d", GPR, FPR64>;

def FCVT_LU_D : FPUnaryOp_r_frm<0b1100001, 0b00011, GPR, FPR64, "fcvt.lu.d">,
                Sched<[WriteFCvtF64ToI64, ReadFCvtF64ToI64]>;
def           : FPUnaryOpDynFrmAlias<FCVT_LU_D, "fcvt.lu.d", GPR, FPR64>;

let mayRaiseFPException = 0 in
def FMV_X_D : FPUnaryOp_r<0b1110001, 0b00000, 0b000, GPR, FPR64, "fmv.x.d">,
              Sched<[WriteFMovF64ToI64, ReadFMovF64ToI64]>;

def FCVT_D_L : FPUnaryOp_r_frm<0b1101001, 0b00010, FPR64, GPR, "fcvt.d.l">,
               Sched<[WriteFCvtI64ToF64, ReadFCvtI64ToF64]>;
def          : FPUnaryOpDynFrmAlias<FCVT_D_L, "fcvt.d.l", FPR64, GPR>;

def FCVT_D_LU : FPUnaryOp_r_frm<0b1101001, 0b00011, FPR64, GPR, "fcvt.d.lu">,
                Sched<[WriteFCvtI64ToF64, ReadFCvtI64ToF64]>;
def           : FPUnaryOpDynFrmAlias<FCVT_D_LU, "fcvt.d.lu", FPR64, GPR>;

let mayRaiseFPException = 0 in
def FMV_D_X : FPUnaryOp_r<0b1111001, 0b00000, 0b000, FPR64, GPR, "fmv.d.x">,
              Sched<[WriteFMovI64ToF64, ReadFMovI64ToF64]>;
} // Predicates = [HasStdExtD, IsRV64]

//===----------------------------------------------------------------------===//
// Assembler Pseudo Instructions (User-Level ISA, Version 2.2, Chapter 20)
//===----------------------------------------------------------------------===//

let Predicates = [HasStdExtD] in {
def : InstAlias<"fld $rd, (${rs1})",  (FLD FPR64:$rd,  GPR:$rs1, 0), 0>;
def : InstAlias<"fsd $rs2, (${rs1})", (FSD FPR64:$rs2, GPR:$rs1, 0), 0>;

def : InstAlias<"fmv.d $rd, $rs",  (FSGNJ_D  FPR64:$rd, FPR64:$rs, FPR64:$rs)>;
def : InstAlias<"fabs.d $rd, $rs", (FSGNJX_D FPR64:$rd, FPR64:$rs, FPR64:$rs)>;
def : InstAlias<"fneg.d $rd, $rs", (FSGNJN_D FPR64:$rd, FPR64:$rs, FPR64:$rs)>;

// fgt.d/fge.d are recognised by the GNU assembler but the canonical
// flt.d/fle.d forms will always be printed. Therefore, set a zero weight.
def : InstAlias<"fgt.d $rd, $rs, $rt",
                (FLT_D GPR:$rd, FPR64:$rt, FPR64:$rs), 0>;
def : InstAlias<"fge.d $rd, $rs, $rt",
                (FLE_D GPR:$rd, FPR64:$rt, FPR64:$rs), 0>;

def PseudoFLD  : PseudoFloatLoad<"fld", FPR64>;
def PseudoFSD  : PseudoStore<"fsd", FPR64>;
let usesCustomInserter = 1 in {
def PseudoQuietFLE_D : PseudoQuietFCMP<FPR64>;
def PseudoQuietFLT_D : PseudoQuietFCMP<FPR64>;
}
} // Predicates = [HasStdExtD]

//===----------------------------------------------------------------------===//
// Pseudo-instructions and codegen patterns
//===----------------------------------------------------------------------===//

class PatFpr64Fpr64<SDPatternOperator OpNode, RVInstR Inst>
    : Pat<(OpNode FPR64:$rs1, FPR64:$rs2), (Inst $rs1, $rs2)>;

class PatFpr64Fpr64DynFrm<SDPatternOperator OpNode, RVInstRFrm Inst>
    : Pat<(OpNode FPR64:$rs1, FPR64:$rs2), (Inst $rs1, $rs2, 0b111)>;

let Predicates = [HasStdExtD] in {

/// Float conversion operations

// f64 -> f32, f32 -> f64
def : Pat<(any_fpround FPR64:$rs1), (FCVT_S_D FPR64:$rs1, 0b111)>;
def : Pat<(any_fpextend FPR32:$rs1), (FCVT_D_S FPR32:$rs1)>;

// [u]int<->double conversion patterns must be gated on IsRV32 or IsRV64, so
// are defined later.

/// Float arithmetic operations

def : PatFpr64Fpr64DynFrm<any_fadd, FADD_D>;
def : PatFpr64Fpr64DynFrm<any_fsub, FSUB_D>;
def : PatFpr64Fpr64DynFrm<any_fmul, FMUL_D>;
def : PatFpr64Fpr64DynFrm<any_fdiv, FDIV_D>;

def : Pat<(any_fsqrt FPR64:$rs1), (FSQRT_D FPR64:$rs1, 0b111)>;

def : Pat<(fneg FPR64:$rs1), (FSGNJN_D $rs1, $rs1)>;
def : Pat<(fabs FPR64:$rs1), (FSGNJX_D $rs1, $rs1)>;

def : PatFpr64Fpr64<fcopysign, FSGNJ_D>;
def : Pat<(fcopysign FPR64:$rs1, (fneg FPR64:$rs2)), (FSGNJN_D $rs1, $rs2)>;
def : Pat<(fcopysign FPR64:$rs1, FPR32:$rs2), (FSGNJ_D $rs1, (FCVT_D_S $rs2))>;
def : Pat<(fcopysign FPR32:$rs1, FPR64:$rs2), (FSGNJ_S $rs1, (FCVT_S_D $rs2,
                                                              0b111))>;

// fmadd: rs1 * rs2 + rs3
def : Pat<(any_fma FPR64:$rs1, FPR64:$rs2, FPR64:$rs3),
          (FMADD_D $rs1, $rs2, $rs3, 0b111)>;

// fmsub: rs1 * rs2 - rs3
def : Pat<(any_fma FPR64:$rs1, FPR64:$rs2, (fneg FPR64:$rs3)),
          (FMSUB_D FPR64:$rs1, FPR64:$rs2, FPR64:$rs3, 0b111)>;

// fnmsub: -rs1 * rs2 + rs3
def : Pat<(any_fma (fneg FPR64:$rs1), FPR64:$rs2, FPR64:$rs3),
          (FNMSUB_D FPR64:$rs1, FPR64:$rs2, FPR64:$rs3, 0b111)>;

// fnmadd: -rs1 * rs2 - rs3
def : Pat<(any_fma (fneg FPR64:$rs1), FPR64:$rs2, (fneg FPR64:$rs3)),
          (FNMADD_D FPR64:$rs1, FPR64:$rs2, FPR64:$rs3, 0b111)>;

// The ratified 20191213 ISA spec defines fmin and fmax in a way that matches
// LLVM's fminnum and fmaxnum.
// <https://github.com/riscv/riscv-isa-manual/commit/cd20cee7efd9bac7c5aa127ec3b451749d2b3cce>.
def : PatFpr64Fpr64<fminnum, FMIN_D>;
def : PatFpr64Fpr64<fmaxnum, FMAX_D>;

/// Setcc
// FIXME: SETEQ/SETLT/SETLE imply nonans, can we pick better instructions for
// strict versions of those.

// Match non-signaling FEQ_D
def : PatSetCC<FPR64, any_fsetcc, SETEQ, FEQ_D>;
def : PatSetCC<FPR64, any_fsetcc, SETOEQ, FEQ_D>;
def : PatSetCC<FPR64, strict_fsetcc, SETLT, PseudoQuietFLT_D>;
def : PatSetCC<FPR64, strict_fsetcc, SETOLT, PseudoQuietFLT_D>;
def : PatSetCC<FPR64, strict_fsetcc, SETLE, PseudoQuietFLE_D>;
def : PatSetCC<FPR64, strict_fsetcc, SETOLE, PseudoQuietFLE_D>;

// Match signaling FEQ_D
def : Pat<(strict_fsetccs FPR64:$rs1, FPR64:$rs2, SETEQ),
          (AND (FLE_D $rs1, $rs2),
               (FLE_D $rs2, $rs1))>;
def : Pat<(strict_fsetccs FPR64:$rs1, FPR64:$rs2, SETOEQ),
          (AND (FLE_D $rs1, $rs2),
               (FLE_D $rs2, $rs1))>;
// If both operands are the same, use a single FLE.
def : Pat<(strict_fsetccs FPR64:$rs1, FPR64:$rs1, SETEQ),
          (FLE_D $rs1, $rs1)>;
def : Pat<(strict_fsetccs FPR64:$rs1, FPR64:$rs1, SETOEQ),
          (FLE_D $rs1, $rs1)>;

def : PatSetCC<FPR64, any_fsetccs, SETLT, FLT_D>;
def : PatSetCC<FPR64, any_fsetccs, SETOLT, FLT_D>;
def : PatSetCC<FPR64, any_fsetccs, SETLE, FLE_D>;
def : PatSetCC<FPR64, any_fsetccs, SETOLE, FLE_D>;

def Select_FPR64_Using_CC_GPR : SelectCC_rrirr<FPR64, GPR>;

/// Loads

defm : LdPat<load, FLD, f64>;

/// Stores

defm : StPat<store, FSD, FPR64, f64>;

/// Pseudo-instructions needed for the soft-float ABI with RV32D

// Moves two GPRs to an FPR.
let usesCustomInserter = 1 in
def BuildPairF64Pseudo
    : Pseudo<(outs FPR64:$dst), (ins GPR:$src1, GPR:$src2),
             [(set FPR64:$dst, (RISCVBuildPairF64 GPR:$src1, GPR:$src2))]>;

// Moves an FPR to two GPRs.
let usesCustomInserter = 1 in
def SplitF64Pseudo
    : Pseudo<(outs GPR:$dst1, GPR:$dst2), (ins FPR64:$src),
             [(set GPR:$dst1, GPR:$dst2, (RISCVSplitF64 FPR64:$src))]>;

} // Predicates = [HasStdExtD]

let Predicates = [HasStdExtD, IsRV32] in {

/// Float constants
def : Pat<(f64 (fpimm0)), (FCVT_D_W (i32 X0))>;
def : Pat<(f64 (fpimmneg0)), (FSGNJN_D (FCVT_D_W (i32 X0)),
                                       (FCVT_D_W (i32 X0)))>;

// double->[u]int. Round-to-zero must be used.
def : Pat<(i32 (any_fp_to_sint FPR64:$rs1)), (FCVT_W_D FPR64:$rs1, 0b001)>;
def : Pat<(i32 (any_fp_to_uint FPR64:$rs1)), (FCVT_WU_D FPR64:$rs1, 0b001)>;

// Saturating double->[u]int32.
def : Pat<(i32 (riscv_fcvt_x FPR64:$rs1, timm:$frm)), (FCVT_W_D $rs1, timm:$frm)>;
def : Pat<(i32 (riscv_fcvt_xu FPR64:$rs1, timm:$frm)), (FCVT_WU_D $rs1, timm:$frm)>;

// float->int32 with current rounding mode.
def : Pat<(i32 (any_lrint FPR64:$rs1)), (FCVT_W_D $rs1, 0b111)>;

// float->int32 rounded to nearest with ties rounded away from zero.
def : Pat<(i32 (any_lround FPR64:$rs1)), (FCVT_W_D $rs1, 0b100)>;

// [u]int->double.
def : Pat<(any_sint_to_fp (i32 GPR:$rs1)), (FCVT_D_W GPR:$rs1)>;
def : Pat<(any_uint_to_fp (i32 GPR:$rs1)), (FCVT_D_WU GPR:$rs1)>;
} // Predicates = [HasStdExtD, IsRV32]

let Predicates = [HasStdExtD, IsRV64] in {

/// Float constants
def : Pat<(f64 (fpimm0)), (FMV_D_X (i64 X0))>;
def : Pat<(f64 (fpimmneg0)), (FSGNJN_D (FMV_D_X (i64 X0)),
                                       (FMV_D_X (i64 X0)))>;

// Moves (no conversion)
def : Pat<(bitconvert (i64 GPR:$rs1)), (FMV_D_X GPR:$rs1)>;
def : Pat<(i64 (bitconvert FPR64:$rs1)), (FMV_X_D FPR64:$rs1)>;

// Use target specific isd nodes to help us remember the result is sign
// extended. Matching sext_inreg+fptoui/fptosi may cause the conversion to be
// duplicated if it has another user that didn't need the sign_extend.
def : Pat<(riscv_any_fcvt_w_rv64 FPR64:$rs1, timm:$frm),  (FCVT_W_D $rs1, timm:$frm)>;
def : Pat<(riscv_any_fcvt_wu_rv64 FPR64:$rs1, timm:$frm), (FCVT_WU_D $rs1, timm:$frm)>;

// [u]int32->fp
def : Pat<(any_sint_to_fp (i64 (sexti32 (i64 GPR:$rs1)))), (FCVT_D_W $rs1)>;
def : Pat<(any_uint_to_fp (i64 (zexti32 (i64 GPR:$rs1)))), (FCVT_D_WU $rs1)>;

// Saturating double->[u]int64.
def : Pat<(i64 (riscv_fcvt_x FPR64:$rs1, timm:$frm)), (FCVT_L_D $rs1, timm:$frm)>;
def : Pat<(i64 (riscv_fcvt_xu FPR64:$rs1, timm:$frm)), (FCVT_LU_D $rs1, timm:$frm)>;

// double->[u]int64. Round-to-zero must be used.
def : Pat<(i64 (any_fp_to_sint FPR64:$rs1)), (FCVT_L_D FPR64:$rs1, 0b001)>;
def : Pat<(i64 (any_fp_to_uint FPR64:$rs1)), (FCVT_LU_D FPR64:$rs1, 0b001)>;

// double->int64 with current rounding mode.
def : Pat<(i64 (any_lrint FPR64:$rs1)), (FCVT_L_D $rs1, 0b111)>;
def : Pat<(i64 (any_llrint FPR64:$rs1)), (FCVT_L_D $rs1, 0b111)>;

// double->int64 rounded to nearest with ties rounded away from zero.
def : Pat<(i64 (any_lround FPR64:$rs1)), (FCVT_L_D $rs1, 0b100)>;
def : Pat<(i64 (any_llround FPR64:$rs1)), (FCVT_L_D $rs1, 0b100)>;

// [u]int64->fp. Match GCC and default to using dynamic rounding mode.
def : Pat<(any_sint_to_fp (i64 GPR:$rs1)), (FCVT_D_L GPR:$rs1, 0b111)>;
def : Pat<(any_uint_to_fp (i64 GPR:$rs1)), (FCVT_D_LU GPR:$rs1, 0b111)>;
} // Predicates = [HasStdExtD, IsRV64]

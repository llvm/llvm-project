//===-- RISCVInstrInfoZfh.td - RISC-V 'Zfh' instructions ---*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file describes the RISC-V instructions from the standard 'Zfh'
// half-precision floating-point extension, version 1.0.
//
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// RISC-V specific DAG Nodes.
//===----------------------------------------------------------------------===//

def SDT_RISCVFMV_H_X
    : SDTypeProfile<1, 1, [SDTCisVT<0, f16>, SDTCisVT<1, XLenVT>]>;
def SDT_RISCVFMV_X_ANYEXTH
    : SDTypeProfile<1, 1, [SDTCisVT<0, XLenVT>, SDTCisVT<1, f16>]>;

def riscv_fmv_h_x
    : SDNode<"RISCVISD::FMV_H_X", SDT_RISCVFMV_H_X>;
def riscv_fmv_x_anyexth
    : SDNode<"RISCVISD::FMV_X_ANYEXTH", SDT_RISCVFMV_X_ANYEXTH>;

//===----------------------------------------------------------------------===//
// Instructions
//===----------------------------------------------------------------------===//

let Predicates = [HasStdExtZfhmin] in {
def FLH : FPLoad_r<0b001, "flh", FPR16, WriteFLD16>;

// Operands for stores are in the order srcreg, base, offset rather than
// reflecting the order these fields are specified in the instruction
// encoding.
def FSH : FPStore_r<0b001, "fsh", FPR16, WriteFST16>;
} // Predicates = [HasStdExtZfhmin]

let Predicates = [HasStdExtZfh] in {
let SchedRW = [WriteFMA16, ReadFMA16, ReadFMA16, ReadFMA16] in {
def FMADD_H  : FPFMA_rrr_frm<OPC_MADD,  0b10, "fmadd.h",  FPR16>;
def FMSUB_H  : FPFMA_rrr_frm<OPC_MSUB,  0b10, "fmsub.h",  FPR16>;
def FNMSUB_H : FPFMA_rrr_frm<OPC_NMSUB, 0b10, "fnmsub.h", FPR16>;
def FNMADD_H : FPFMA_rrr_frm<OPC_NMADD, 0b10, "fnmadd.h", FPR16>;
}

def : FPFMADynFrmAlias<FMADD_H,  "fmadd.h",  FPR16>;
def : FPFMADynFrmAlias<FMSUB_H,  "fmsub.h",  FPR16>;
def : FPFMADynFrmAlias<FNMSUB_H, "fnmsub.h", FPR16>;
def : FPFMADynFrmAlias<FNMADD_H, "fnmadd.h", FPR16>;

def FADD_H : FPALU_rr_frm<0b0000010, "fadd.h", FPR16>,
             Sched<[WriteFALU16, ReadFALU16, ReadFALU16]>;
def FSUB_H : FPALU_rr_frm<0b0000110, "fsub.h", FPR16>,
             Sched<[WriteFALU16, ReadFALU16, ReadFALU16]>;
def FMUL_H : FPALU_rr_frm<0b0001010, "fmul.h", FPR16>,
             Sched<[WriteFMul16, ReadFMul16, ReadFMul16]>;
def FDIV_H : FPALU_rr_frm<0b0001110, "fdiv.h", FPR16>,
             Sched<[WriteFDiv16, ReadFDiv16, ReadFDiv16]>;

def        : FPALUDynFrmAlias<FADD_H, "fadd.h", FPR16>;
def        : FPALUDynFrmAlias<FSUB_H, "fsub.h", FPR16>;
def        : FPALUDynFrmAlias<FMUL_H, "fmul.h", FPR16>;
def        : FPALUDynFrmAlias<FDIV_H, "fdiv.h", FPR16>;

def FSQRT_H : FPUnaryOp_r_frm<0b0101110, 0b00000, FPR16, FPR16, "fsqrt.h">,
              Sched<[WriteFSqrt16, ReadFSqrt16]>;
def         : FPUnaryOpDynFrmAlias<FSQRT_H, "fsqrt.h", FPR16, FPR16>;

let SchedRW = [WriteFSGNJ16, ReadFSGNJ16, ReadFSGNJ16],
    mayRaiseFPException = 0 in {
def FSGNJ_H  : FPALU_rr<0b0010010, 0b000, "fsgnj.h", FPR16>;
def FSGNJN_H : FPALU_rr<0b0010010, 0b001, "fsgnjn.h", FPR16>;
def FSGNJX_H : FPALU_rr<0b0010010, 0b010, "fsgnjx.h", FPR16>;
}

let SchedRW = [WriteFMinMax16, ReadFMinMax16, ReadFMinMax16] in {
def FMIN_H   : FPALU_rr<0b0010110, 0b000, "fmin.h", FPR16>;
def FMAX_H   : FPALU_rr<0b0010110, 0b001, "fmax.h", FPR16>;
}

def FCVT_W_H : FPUnaryOp_r_frm<0b1100010, 0b00000, GPR, FPR16, "fcvt.w.h">,
               Sched<[WriteFCvtF16ToI32, ReadFCvtF16ToI32]>;
def          : FPUnaryOpDynFrmAlias<FCVT_W_H, "fcvt.w.h", GPR, FPR16>;

def FCVT_WU_H : FPUnaryOp_r_frm<0b1100010, 0b00001, GPR, FPR16, "fcvt.wu.h">,
                Sched<[WriteFCvtF16ToI32, ReadFCvtF16ToI32]>;
def           : FPUnaryOpDynFrmAlias<FCVT_WU_H, "fcvt.wu.h", GPR, FPR16>;

def FCVT_H_W : FPUnaryOp_r_frm<0b1101010, 0b00000, FPR16, GPR, "fcvt.h.w">,
               Sched<[WriteFCvtI32ToF16, ReadFCvtI32ToF16]>;
def          : FPUnaryOpDynFrmAlias<FCVT_H_W, "fcvt.h.w", FPR16, GPR>;

def FCVT_H_WU : FPUnaryOp_r_frm<0b1101010, 0b00001, FPR16, GPR, "fcvt.h.wu">,
                Sched<[WriteFCvtI32ToF16, ReadFCvtI32ToF16]>;
def           : FPUnaryOpDynFrmAlias<FCVT_H_WU, "fcvt.h.wu", FPR16, GPR>;
} // Predicates = [HasStdExtZfh]

let Predicates = [HasStdExtZfhmin] in {
def FCVT_H_S : FPUnaryOp_r_frm<0b0100010, 0b00000, FPR16, FPR32, "fcvt.h.s">,
               Sched<[WriteFCvtF32ToF16, ReadFCvtF32ToF16]>;
def          : FPUnaryOpDynFrmAlias<FCVT_H_S, "fcvt.h.s", FPR16, FPR32>;

def FCVT_S_H : FPUnaryOp_r<0b0100000, 0b00010, 0b000, FPR32, FPR16, "fcvt.s.h">,
               Sched<[WriteFCvtF16ToF32, ReadFCvtF16ToF32]>;

let mayRaiseFPException = 0 in
def FMV_X_H : FPUnaryOp_r<0b1110010, 0b00000, 0b000, GPR, FPR16, "fmv.x.h">,
              Sched<[WriteFMovF16ToI16, ReadFMovF16ToI16]>;

let mayRaiseFPException = 0 in
def FMV_H_X : FPUnaryOp_r<0b1111010, 0b00000, 0b000, FPR16, GPR, "fmv.h.x">,
              Sched<[WriteFMovI16ToF16, ReadFMovI16ToF16]>;
} // Predicates = [HasStdExtZfhmin]

let Predicates = [HasStdExtZfh] in {

let SchedRW = [WriteFCmp16, ReadFCmp16, ReadFCmp16] in {
def FEQ_H : FPCmp_rr<0b1010010, 0b010, "feq.h", FPR16>;
def FLT_H : FPCmp_rr<0b1010010, 0b001, "flt.h", FPR16>;
def FLE_H : FPCmp_rr<0b1010010, 0b000, "fle.h", FPR16>;
}

let mayRaiseFPException = 0 in
def FCLASS_H : FPUnaryOp_r<0b1110010, 0b00000, 0b001, GPR, FPR16, "fclass.h">,
               Sched<[WriteFClass16, ReadFClass16]>;
} // Predicates = [HasStdExtZfh]

let Predicates = [HasStdExtZfh, IsRV64] in {
def FCVT_L_H  : FPUnaryOp_r_frm<0b1100010, 0b00010, GPR, FPR16, "fcvt.l.h">,
                Sched<[WriteFCvtF16ToI64, ReadFCvtF16ToI64]>;
def           : FPUnaryOpDynFrmAlias<FCVT_L_H, "fcvt.l.h", GPR, FPR16>;

def FCVT_LU_H  : FPUnaryOp_r_frm<0b1100010, 0b00011, GPR, FPR16, "fcvt.lu.h">,
                 Sched<[WriteFCvtF16ToI64, ReadFCvtF16ToI64]>;
def            : FPUnaryOpDynFrmAlias<FCVT_LU_H, "fcvt.lu.h", GPR, FPR16>;

def FCVT_H_L : FPUnaryOp_r_frm<0b1101010, 0b00010, FPR16, GPR, "fcvt.h.l">,
               Sched<[WriteFCvtI64ToF16, ReadFCvtI64ToF16]>;
def          : FPUnaryOpDynFrmAlias<FCVT_H_L, "fcvt.h.l", FPR16, GPR>;

def FCVT_H_LU : FPUnaryOp_r_frm<0b1101010, 0b00011, FPR16, GPR, "fcvt.h.lu">,
                Sched<[WriteFCvtI64ToF16, ReadFCvtI64ToF16]>;
def           : FPUnaryOpDynFrmAlias<FCVT_H_LU, "fcvt.h.lu", FPR16, GPR>;
} // Predicates = [HasStdExtZfh, IsRV64]

let Predicates = [HasStdExtZfhmin, HasStdExtD] in {
def FCVT_H_D : FPUnaryOp_r_frm<0b0100010, 0b00001, FPR16, FPR64, "fcvt.h.d">,
               Sched<[WriteFCvtF64ToF16, ReadFCvtF64ToF16]>;
def          : FPUnaryOpDynFrmAlias<FCVT_H_D, "fcvt.h.d", FPR16, FPR64>;

def FCVT_D_H : FPUnaryOp_r<0b0100001, 0b00010, 0b000, FPR64, FPR16, "fcvt.d.h">,
               Sched<[WriteFCvtF16ToF64, ReadFCvtF16ToF64]>;
} // Predicates = [HasStdExtZfhmin, HasStdExtD]

//===----------------------------------------------------------------------===//
// Assembler Pseudo Instructions (User-Level ISA, Version 2.2, Chapter 20)
//===----------------------------------------------------------------------===//

let Predicates = [HasStdExtZfhmin] in {
def : InstAlias<"flh $rd, (${rs1})",  (FLH FPR16:$rd,  GPR:$rs1, 0), 0>;
def : InstAlias<"fsh $rs2, (${rs1})", (FSH FPR16:$rs2, GPR:$rs1, 0), 0>;
} // Predicates = [HasStdExtZfhmin]

let Predicates = [HasStdExtZfh] in {
def : InstAlias<"fmv.h $rd, $rs",  (FSGNJ_H  FPR16:$rd, FPR16:$rs, FPR16:$rs)>;
def : InstAlias<"fabs.h $rd, $rs", (FSGNJX_H FPR16:$rd, FPR16:$rs, FPR16:$rs)>;
def : InstAlias<"fneg.h $rd, $rs", (FSGNJN_H FPR16:$rd, FPR16:$rs, FPR16:$rs)>;

// fgt.h/fge.h are recognised by the GNU assembler but the canonical
// flt.h/fle.h forms will always be printed. Therefore, set a zero weight.
def : InstAlias<"fgt.h $rd, $rs, $rt",
                (FLT_H GPR:$rd, FPR16:$rt, FPR16:$rs), 0>;
def : InstAlias<"fge.h $rd, $rs, $rt",
                (FLE_H GPR:$rd, FPR16:$rt, FPR16:$rs), 0>;
} // Predicates = [HasStdExtZfh]

let Predicates = [HasStdExtZfhmin] in {
def PseudoFLH  : PseudoFloatLoad<"flh", FPR16>;
def PseudoFSH  : PseudoStore<"fsh", FPR16>;
let usesCustomInserter = 1 in {
def PseudoQuietFLE_H : PseudoQuietFCMP<FPR16>;
def PseudoQuietFLT_H : PseudoQuietFCMP<FPR16>;
}
} // Predicates = [HasStdExtZfhmin]

//===----------------------------------------------------------------------===//
// Pseudo-instructions and codegen patterns
//===----------------------------------------------------------------------===//

/// Generic pattern classes
class PatFpr16Fpr16<SDPatternOperator OpNode, RVInstR Inst>
    : Pat<(OpNode FPR16:$rs1, FPR16:$rs2), (Inst $rs1, $rs2)>;

class PatFpr16Fpr16DynFrm<SDPatternOperator OpNode, RVInstRFrm Inst>
    : Pat<(OpNode FPR16:$rs1, FPR16:$rs2), (Inst $rs1, $rs2, 0b111)>;

let Predicates = [HasStdExtZfh] in {

/// Float constants
def : Pat<(f16 (fpimm0)), (FMV_H_X X0)>;

/// Float conversion operations

// [u]int32<->float conversion patterns must be gated on IsRV32 or IsRV64, so
// are defined later.

/// Float arithmetic operations

def : PatFpr16Fpr16DynFrm<any_fadd, FADD_H>;
def : PatFpr16Fpr16DynFrm<any_fsub, FSUB_H>;
def : PatFpr16Fpr16DynFrm<any_fmul, FMUL_H>;
def : PatFpr16Fpr16DynFrm<any_fdiv, FDIV_H>;

def : Pat<(any_fsqrt FPR16:$rs1), (FSQRT_H FPR16:$rs1, 0b111)>;

def : Pat<(fneg FPR16:$rs1), (FSGNJN_H $rs1, $rs1)>;
def : Pat<(fabs FPR16:$rs1), (FSGNJX_H $rs1, $rs1)>;

def : PatFpr16Fpr16<fcopysign, FSGNJ_H>;
def : Pat<(fcopysign FPR16:$rs1, (fneg FPR16:$rs2)), (FSGNJN_H $rs1, $rs2)>;
def : Pat<(fcopysign FPR16:$rs1, FPR32:$rs2),
          (FSGNJ_H $rs1, (FCVT_H_S $rs2, 0b111))>;
def : Pat<(fcopysign FPR32:$rs1, FPR16:$rs2), (FSGNJ_S $rs1, (FCVT_S_H $rs2))>;

// fmadd: rs1 * rs2 + rs3
def : Pat<(any_fma FPR16:$rs1, FPR16:$rs2, FPR16:$rs3),
          (FMADD_H $rs1, $rs2, $rs3, 0b111)>;

// fmsub: rs1 * rs2 - rs3
def : Pat<(any_fma FPR16:$rs1, FPR16:$rs2, (fneg FPR16:$rs3)),
          (FMSUB_H FPR16:$rs1, FPR16:$rs2, FPR16:$rs3, 0b111)>;

// fnmsub: -rs1 * rs2 + rs3
def : Pat<(any_fma (fneg FPR16:$rs1), FPR16:$rs2, FPR16:$rs3),
          (FNMSUB_H FPR16:$rs1, FPR16:$rs2, FPR16:$rs3, 0b111)>;

// fnmadd: -rs1 * rs2 - rs3
def : Pat<(any_fma (fneg FPR16:$rs1), FPR16:$rs2, (fneg FPR16:$rs3)),
          (FNMADD_H FPR16:$rs1, FPR16:$rs2, FPR16:$rs3, 0b111)>;

// The ratified 20191213 ISA spec defines fmin and fmax in a way that matches
// LLVM's fminnum and fmaxnum
// <https://github.com/riscv/riscv-isa-manual/commit/cd20cee7efd9bac7c5aa127ec3b451749d2b3cce>.
def : PatFpr16Fpr16<fminnum, FMIN_H>;
def : PatFpr16Fpr16<fmaxnum, FMAX_H>;

/// Setcc
// FIXME: SETEQ/SETLT/SETLE imply nonans, can we pick better instructions for
// strict versions of those.

// Match non-signaling FEQ_D
def : PatSetCC<FPR16, any_fsetcc, SETEQ, FEQ_H>;
def : PatSetCC<FPR16, any_fsetcc, SETOEQ, FEQ_H>;
def : PatSetCC<FPR16, strict_fsetcc, SETLT, PseudoQuietFLT_H>;
def : PatSetCC<FPR16, strict_fsetcc, SETOLT, PseudoQuietFLT_H>;
def : PatSetCC<FPR16, strict_fsetcc, SETLE, PseudoQuietFLE_H>;
def : PatSetCC<FPR16, strict_fsetcc, SETOLE, PseudoQuietFLE_H>;

// Match signaling FEQ_H
def : Pat<(strict_fsetccs FPR16:$rs1, FPR16:$rs2, SETEQ),
          (AND (FLE_H $rs1, $rs2),
               (FLE_H $rs2, $rs1))>;
def : Pat<(strict_fsetccs FPR16:$rs1, FPR16:$rs2, SETOEQ),
          (AND (FLE_H $rs1, $rs2),
               (FLE_H $rs2, $rs1))>;
// If both operands are the same, use a single FLE.
def : Pat<(strict_fsetccs FPR16:$rs1, FPR16:$rs1, SETEQ),
          (FLE_H $rs1, $rs1)>;
def : Pat<(strict_fsetccs FPR16:$rs1, FPR16:$rs1, SETOEQ),
          (FLE_H $rs1, $rs1)>;

def : PatSetCC<FPR16, any_fsetccs, SETLT, FLT_H>;
def : PatSetCC<FPR16, any_fsetccs, SETOLT, FLT_H>;
def : PatSetCC<FPR16, any_fsetccs, SETLE, FLE_H>;
def : PatSetCC<FPR16, any_fsetccs, SETOLE, FLE_H>;

def Select_FPR16_Using_CC_GPR : SelectCC_rrirr<FPR16, GPR>;
} // Predicates = [HasStdExtZfh]

let Predicates = [HasStdExtZfhmin] in {
/// Loads

defm : LdPat<load, FLH, f16>;

/// Stores

defm : StPat<store, FSH, FPR16, f16>;

/// Float conversion operations

// f32 -> f16, f16 -> f32
def : Pat<(any_fpround FPR32:$rs1), (FCVT_H_S FPR32:$rs1, 0b111)>;
def : Pat<(any_fpextend FPR16:$rs1), (FCVT_S_H FPR16:$rs1)>;

// Moves (no conversion)
def : Pat<(riscv_fmv_h_x GPR:$src), (FMV_H_X GPR:$src)>;
def : Pat<(riscv_fmv_x_anyexth FPR16:$src), (FMV_X_H FPR16:$src)>;
} // Predicates = [HasStdExtZfhmin]

let Predicates = [HasStdExtZfh, IsRV32] in {
// half->[u]int. Round-to-zero must be used.
def : Pat<(i32 (any_fp_to_sint FPR16:$rs1)), (FCVT_W_H $rs1, 0b001)>;
def : Pat<(i32 (any_fp_to_uint FPR16:$rs1)), (FCVT_WU_H $rs1, 0b001)>;

// Saturating float->[u]int32.
def : Pat<(i32 (riscv_fcvt_x FPR16:$rs1, timm:$frm)), (FCVT_W_H $rs1, timm:$frm)>;
def : Pat<(i32 (riscv_fcvt_xu FPR16:$rs1, timm:$frm)), (FCVT_WU_H $rs1, timm:$frm)>;

// half->int32 with current rounding mode.
def : Pat<(i32 (any_lrint FPR16:$rs1)), (FCVT_W_H $rs1, 0b111)>;

// half->int32 rounded to nearest with ties rounded away from zero.
def : Pat<(i32 (any_lround FPR16:$rs1)), (FCVT_W_H $rs1, 0b100)>;

// [u]int->half. Match GCC and default to using dynamic rounding mode.
def : Pat<(any_sint_to_fp (i32 GPR:$rs1)), (FCVT_H_W $rs1, 0b111)>;
def : Pat<(any_uint_to_fp (i32 GPR:$rs1)), (FCVT_H_WU $rs1, 0b111)>;
} // Predicates = [HasStdExtZfh, IsRV32]

let Predicates = [HasStdExtZfh, IsRV64] in {
// Use target specific isd nodes to help us remember the result is sign
// extended. Matching sext_inreg+fptoui/fptosi may cause the conversion to be
// duplicated if it has another user that didn't need the sign_extend.
def : Pat<(riscv_any_fcvt_w_rv64 FPR16:$rs1, timm:$frm),  (FCVT_W_H $rs1, timm:$frm)>;
def : Pat<(riscv_any_fcvt_wu_rv64 FPR16:$rs1, timm:$frm), (FCVT_WU_H $rs1, timm:$frm)>;

// half->[u]int64. Round-to-zero must be used.
def : Pat<(i64 (any_fp_to_sint FPR16:$rs1)), (FCVT_L_H $rs1, 0b001)>;
def : Pat<(i64 (any_fp_to_uint FPR16:$rs1)), (FCVT_LU_H $rs1, 0b001)>;

// Saturating float->[u]int64.
def : Pat<(i64 (riscv_fcvt_x FPR16:$rs1, timm:$frm)), (FCVT_L_H $rs1, timm:$frm)>;
def : Pat<(i64 (riscv_fcvt_xu FPR16:$rs1, timm:$frm)), (FCVT_LU_H $rs1, timm:$frm)>;

// half->int64 with current rounding mode.
def : Pat<(i64 (any_lrint FPR16:$rs1)), (FCVT_L_H $rs1, 0b111)>;
def : Pat<(i64 (any_llrint FPR16:$rs1)), (FCVT_L_H $rs1, 0b111)>;

// half->int64 rounded to nearest with ties rounded away from zero.
def : Pat<(i64 (any_lround FPR16:$rs1)), (FCVT_L_H $rs1, 0b100)>;
def : Pat<(i64 (any_llround FPR16:$rs1)), (FCVT_L_H $rs1, 0b100)>;

// [u]int->fp. Match GCC and default to using dynamic rounding mode.
def : Pat<(any_sint_to_fp (i64 (sexti32 (i64 GPR:$rs1)))), (FCVT_H_W $rs1, 0b111)>;
def : Pat<(any_uint_to_fp (i64 (zexti32 (i64 GPR:$rs1)))), (FCVT_H_WU $rs1, 0b111)>;
def : Pat<(any_sint_to_fp (i64 GPR:$rs1)), (FCVT_H_L $rs1, 0b111)>;
def : Pat<(any_uint_to_fp (i64 GPR:$rs1)), (FCVT_H_LU $rs1, 0b111)>;
} // Predicates = [HasStdExtZfh, IsRV64]

let Predicates = [HasStdExtZfhmin, HasStdExtD] in {
/// Float conversion operations
// f64 -> f16, f16 -> f64
def : Pat<(any_fpround FPR64:$rs1), (FCVT_H_D FPR64:$rs1, 0b111)>;
def : Pat<(any_fpextend FPR16:$rs1), (FCVT_D_H FPR16:$rs1)>;

/// Float arithmetic operations
def : Pat<(fcopysign FPR16:$rs1, FPR64:$rs2),
          (FSGNJ_H $rs1, (FCVT_H_D $rs2, 0b111))>;
def : Pat<(fcopysign FPR64:$rs1, FPR16:$rs2), (FSGNJ_D $rs1, (FCVT_D_H $rs2))>;
} // Predicates = [HasStdExtZfhmin, HasStdExtD]

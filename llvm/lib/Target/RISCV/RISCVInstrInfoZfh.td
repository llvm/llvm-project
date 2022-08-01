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
def SDT_RISCVFMV_X_EXTH
    : SDTypeProfile<1, 1, [SDTCisVT<0, XLenVT>, SDTCisVT<1, f16>]>;

def riscv_fmv_h_x
    : SDNode<"RISCVISD::FMV_H_X", SDT_RISCVFMV_H_X>;
def riscv_fmv_x_anyexth
    : SDNode<"RISCVISD::FMV_X_ANYEXTH", SDT_RISCVFMV_X_EXTH>;
def riscv_fmv_x_signexth
    : SDNode<"RISCVISD::FMV_X_SIGNEXTH", SDT_RISCVFMV_X_EXTH>;

//===----------------------------------------------------------------------===//
// Operand and SDNode transformation definitions.
//===----------------------------------------------------------------------===//

// Zhinxmin and Zhinx

def FPR16INX : RegisterOperand<GPRF16> {
  let ParserMatchClass = GPRAsFPR;
  let DecoderMethod = "DecodeGPRRegisterClass";
}

def ZfhExt           : ExtInfo<0, [HasStdExtZfh]>;
def Zfh64Ext         : ExtInfo<0, [HasStdExtZfh,             IsRV64]>;
def ZfhminExt        : ExtInfo<0, [HasStdExtZfhOrZfhmin]>;
def ZhinxExt         : ExtInfo<1, [HasStdExtZhinx]>;
def ZhinxminExt      : ExtInfo<1, [HasStdExtZhinxOrZhinxmin]>;
def Zhinx64Ext       : ExtInfo<1, [HasStdExtZhinx,           IsRV64]>;

def ZfhminDExt       : ExtInfo<0, [HasStdExtZfhOrZfhmin,     HasStdExtD]>;
def ZhinxminZdinxExt : ExtInfo<1, [HasStdExtZhinxOrZhinxmin, HasStdExtZdinx]>;

def H     : ExtInfo_r<ZfhExt,     FPR16>;
def H_INX : ExtInfo_r<ZhinxExt, FPR16INX>;

def HH        : ExtInfo_rr<ZfhExt,           FPR16,    FPR16>;
def HH_INX    : ExtInfo_rr<ZhinxExt,         FPR16INX, FPR16INX>;
def XH        : ExtInfo_rr<ZfhExt,           GPR,      FPR16>;
def XH_INX    : ExtInfo_rr<ZhinxExt,         GPR,      FPR16INX>;
def HX        : ExtInfo_rr<ZfhExt,           FPR16,    GPR>;
def HX_INX    : ExtInfo_rr<ZhinxExt,         FPR16INX, GPR>;
def XH_64     : ExtInfo_rr<Zfh64Ext,         GPR,      FPR16>;
def HX_64     : ExtInfo_rr<Zfh64Ext,         FPR16,    GPR>;
def XH_INX_64 : ExtInfo_rr<Zhinx64Ext,       GPR,      FPR16INX>;
def HX_INX_64 : ExtInfo_rr<Zhinx64Ext,       FPR16INX, GPR>;
def HFmin     : ExtInfo_rr<ZfhminExt,        FPR16,    FPR32>;
def HF_INXmin : ExtInfo_rr<ZhinxminExt,      FPR16INX, FPR32INX>;
def HF_INX    : ExtInfo_rr<ZhinxExt,         FPR16INX, FPR32INX>;
def FHmin     : ExtInfo_rr<ZfhminExt,        FPR32,    FPR16>;
def FH_INXmin : ExtInfo_rr<ZhinxminExt,      FPR32INX, FPR16INX>;
def FH_INX    : ExtInfo_rr<ZhinxExt,         FPR32INX, FPR16INX>;
def DHmin     : ExtInfo_rr<ZfhminDExt,       FPR64,    FPR16>;
def DH_INXmin : ExtInfo_rr<ZhinxminZdinxExt, FPR64INX, FPR16INX>;
def HDmin     : ExtInfo_rr<ZfhminDExt,       FPR16,    FPR64>;
def HD_INXmin : ExtInfo_rr<ZhinxminZdinxExt, FPR16INX, FPR64INX>;

defvar HINX     = [H,     H_INX];
defvar HHINX    = [HH,    HH_INX];
defvar XHINX    = [XH,    XH_INX];
defvar HXINX    = [HX,    HX_INX];
defvar XHIN64X  = [XH_64, XH_INX_64];
defvar HXIN64X  = [HX_64, HX_INX_64];
defvar HFINXmin = [HFmin, HF_INXmin];
defvar FHINXmin = [FHmin, FH_INXmin];
defvar DHINXmin = [DHmin, DH_INXmin];
defvar HDINXmin = [HDmin, HD_INXmin];

//===----------------------------------------------------------------------===//
// Instructions
//===----------------------------------------------------------------------===//

let Predicates = [HasStdExtZfhOrZfhmin] in {
def FLH : FPLoad_r<0b001, "flh", FPR16, WriteFLD16>;

// Operands for stores are in the order srcreg, base, offset rather than
// reflecting the order these fields are specified in the instruction
// encoding.
def FSH : FPStore_r<0b001, "fsh", FPR16, WriteFST16>;
} // Predicates = [HasStdExtZfhOrZfhmin]

let SchedRW = [WriteFMA16, ReadFMA16, ReadFMA16, ReadFMA16] in {
defm FMADD_H  : FPFMA_rrr_frm_m<OPC_MADD,  0b10, "fmadd.h",  HINX>;
defm FMSUB_H  : FPFMA_rrr_frm_m<OPC_MSUB,  0b10, "fmsub.h",  HINX>;
defm FNMSUB_H : FPFMA_rrr_frm_m<OPC_NMSUB, 0b10, "fnmsub.h", HINX>;
defm FNMADD_H : FPFMA_rrr_frm_m<OPC_NMADD, 0b10, "fnmadd.h", HINX>;
}

defm : FPFMADynFrmAlias_m<FMADD_H,  "fmadd.h",  HINX>;
defm : FPFMADynFrmAlias_m<FMSUB_H,  "fmsub.h",  HINX>;
defm : FPFMADynFrmAlias_m<FNMSUB_H, "fnmsub.h", HINX>;
defm : FPFMADynFrmAlias_m<FNMADD_H, "fnmadd.h", HINX>;

let SchedRW = [WriteFALU16, ReadFALU16, ReadFALU16] in {
defm FADD_H : FPALU_rr_frm_m<0b0000010, "fadd.h", HINX, /*Commutable*/1>;
defm FSUB_H : FPALU_rr_frm_m<0b0000110, "fsub.h", HINX>;
}
let SchedRW = [WriteFMul16, ReadFMul16, ReadFMul16] in
defm FMUL_H : FPALU_rr_frm_m<0b0001010, "fmul.h", HINX, /*Commutable*/1>;

let SchedRW = [WriteFDiv16, ReadFDiv16, ReadFDiv16] in
defm FDIV_H : FPALU_rr_frm_m<0b0001110, "fdiv.h", HINX>;

defm : FPALUDynFrmAlias_m<FADD_H, "fadd.h", HINX>;
defm : FPALUDynFrmAlias_m<FSUB_H, "fsub.h", HINX>;
defm : FPALUDynFrmAlias_m<FMUL_H, "fmul.h", HINX>;
defm : FPALUDynFrmAlias_m<FDIV_H, "fdiv.h", HINX>;

defm FSQRT_H : FPUnaryOp_r_frm_m<0b0101110, 0b00000, HHINX, "fsqrt.h">,
               Sched<[WriteFSqrt16, ReadFSqrt16]>;
defm         : FPUnaryOpDynFrmAlias_m<FSQRT_H, "fsqrt.h", HHINX>;

let SchedRW = [WriteFSGNJ16, ReadFSGNJ16, ReadFSGNJ16],
    mayRaiseFPException = 0 in {
defm FSGNJ_H  : FPALU_rr_m<0b0010010, 0b000, "fsgnj.h",  HINX>;
defm FSGNJN_H : FPALU_rr_m<0b0010010, 0b001, "fsgnjn.h", HINX>;
defm FSGNJX_H : FPALU_rr_m<0b0010010, 0b010, "fsgnjx.h", HINX>;
}

let SchedRW = [WriteFMinMax16, ReadFMinMax16, ReadFMinMax16] in {
defm FMIN_H   : FPALU_rr_m<0b0010110, 0b000, "fmin.h", HINX, /*Commutable*/1>;
defm FMAX_H   : FPALU_rr_m<0b0010110, 0b001, "fmax.h", HINX, /*Commutable*/1>;
}

defm FCVT_W_H : FPUnaryOp_r_frm_m<0b1100010, 0b00000, XHINX, "fcvt.w.h">,
                Sched<[WriteFCvtF16ToI32, ReadFCvtF16ToI32]>;
defm          : FPUnaryOpDynFrmAlias_m<FCVT_W_H, "fcvt.w.h", XHINX>;

defm FCVT_WU_H : FPUnaryOp_r_frm_m<0b1100010, 0b00001, XHINX, "fcvt.wu.h">,
                 Sched<[WriteFCvtF16ToI32, ReadFCvtF16ToI32]>;
defm           : FPUnaryOpDynFrmAlias_m<FCVT_WU_H, "fcvt.wu.h", XHINX>;

defm FCVT_H_W : FPUnaryOp_r_frm_m<0b1101010, 0b00000, HXINX, "fcvt.h.w">,
                Sched<[WriteFCvtI32ToF16, ReadFCvtI32ToF16]>;
defm          : FPUnaryOpDynFrmAlias_m<FCVT_H_W, "fcvt.h.w", HXINX>;

defm FCVT_H_WU : FPUnaryOp_r_frm_m<0b1101010, 0b00001, HXINX, "fcvt.h.wu">,
                 Sched<[WriteFCvtI32ToF16, ReadFCvtI32ToF16]>;
defm           : FPUnaryOpDynFrmAlias_m<FCVT_H_WU, "fcvt.h.wu", HXINX>;

defm FCVT_H_S : FPUnaryOp_r_frm_m<0b0100010, 0b00000, HFINXmin, "fcvt.h.s">,
                Sched<[WriteFCvtF32ToF16, ReadFCvtF32ToF16]>;
defm          : FPUnaryOpDynFrmAlias_m<FCVT_H_S, "fcvt.h.s", HFINXmin>;

defm FCVT_S_H : FPUnaryOp_r_m<0b0100000, 0b00010, 0b000, FHINXmin, "fcvt.s.h">,
               Sched<[WriteFCvtF16ToF32, ReadFCvtF16ToF32]>;

let Predicates = [HasStdExtZfhOrZfhmin] in {
let mayRaiseFPException = 0 in
def FMV_X_H : FPUnaryOp_r<0b1110010, 0b00000, 0b000, GPR, FPR16, "fmv.x.h">,
              Sched<[WriteFMovF16ToI16, ReadFMovF16ToI16]>;

let mayRaiseFPException = 0 in
def FMV_H_X : FPUnaryOp_r<0b1111010, 0b00000, 0b000, FPR16, GPR, "fmv.h.x">,
              Sched<[WriteFMovI16ToF16, ReadFMovI16ToF16]>;
} // Predicates = [HasStdExtZfhOrZfhmin]

let SchedRW = [WriteFCmp16, ReadFCmp16, ReadFCmp16] in {
defm FEQ_H : FPCmp_rr_m<0b1010010, 0b010, "feq.h", HINX, /*Commutable*/1>;
defm FLT_H : FPCmp_rr_m<0b1010010, 0b001, "flt.h", HINX>;
defm FLE_H : FPCmp_rr_m<0b1010010, 0b000, "fle.h", HINX>;
}

let mayRaiseFPException = 0 in
defm FCLASS_H : FPUnaryOp_r_m<0b1110010, 0b00000, 0b001, XHINX, "fclass.h">,
                Sched<[WriteFClass16, ReadFClass16]>;

defm FCVT_L_H  : FPUnaryOp_r_frm_m<0b1100010, 0b00010, XHIN64X, "fcvt.l.h">,
                 Sched<[WriteFCvtF16ToI64, ReadFCvtF16ToI64]>;
defm           : FPUnaryOpDynFrmAlias_m<FCVT_L_H, "fcvt.l.h", XHIN64X>;

defm FCVT_LU_H  : FPUnaryOp_r_frm_m<0b1100010, 0b00011, XHIN64X, "fcvt.lu.h">,
                  Sched<[WriteFCvtF16ToI64, ReadFCvtF16ToI64]>;
defm            : FPUnaryOpDynFrmAlias_m<FCVT_LU_H, "fcvt.lu.h", XHIN64X>;

defm FCVT_H_L : FPUnaryOp_r_frm_m<0b1101010, 0b00010, HXIN64X, "fcvt.h.l">,
                Sched<[WriteFCvtI64ToF16, ReadFCvtI64ToF16]>;
defm          : FPUnaryOpDynFrmAlias_m<FCVT_H_L, "fcvt.h.l", HXIN64X>;

defm FCVT_H_LU : FPUnaryOp_r_frm_m<0b1101010, 0b00011, HXIN64X, "fcvt.h.lu">,
                 Sched<[WriteFCvtI64ToF16, ReadFCvtI64ToF16]>;
defm           : FPUnaryOpDynFrmAlias_m<FCVT_H_LU, "fcvt.h.lu", HXIN64X>;

defm FCVT_H_D : FPUnaryOp_r_frm_m<0b0100010, 0b00001, HDINXmin, "fcvt.h.d">,
                Sched<[WriteFCvtF64ToF16, ReadFCvtF64ToF16]>;
defm          : FPUnaryOpDynFrmAlias_m<FCVT_H_D, "fcvt.h.d", HDINXmin>;

defm FCVT_D_H : FPUnaryOp_r_m<0b0100001, 0b00010, 0b000, DHINXmin, "fcvt.d.h">,
                Sched<[WriteFCvtF16ToF64, ReadFCvtF16ToF64]>;

//===----------------------------------------------------------------------===//
// Assembler Pseudo Instructions (User-Level ISA, Version 2.2, Chapter 20)
//===----------------------------------------------------------------------===//

let Predicates = [HasStdExtZfhOrZfhmin] in {
def : InstAlias<"flh $rd, (${rs1})",  (FLH FPR16:$rd,  GPR:$rs1, 0), 0>;
def : InstAlias<"fsh $rs2, (${rs1})", (FSH FPR16:$rs2, GPR:$rs1, 0), 0>;
} // Predicates = [HasStdExtZfhOrZfhmin]

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

let Predicates = [HasStdExtZfhOrZfhmin] in {
def PseudoFLH  : PseudoFloatLoad<"flh", FPR16>;
def PseudoFSH  : PseudoStore<"fsh", FPR16>;
let usesCustomInserter = 1 in {
def PseudoQuietFLE_H : PseudoQuietFCMP<FPR16>;
def PseudoQuietFLT_H : PseudoQuietFCMP<FPR16>;
}
} // Predicates = [HasStdExtZfhOrZfhmin]

let Predicates = [HasStdExtZhinx] in {
def : InstAlias<"fmv.h $rd, $rs",  (FSGNJ_H_INX  FPR16INX:$rd, FPR16INX:$rs, FPR16INX:$rs)>;
def : InstAlias<"fabs.h $rd, $rs", (FSGNJX_H_INX FPR16INX:$rd, FPR16INX:$rs, FPR16INX:$rs)>;
def : InstAlias<"fneg.h $rd, $rs", (FSGNJN_H_INX FPR16INX:$rd, FPR16INX:$rs, FPR16INX:$rs)>;

def : InstAlias<"fgt.h $rd, $rs, $rt",
                (FLT_H_INX GPR:$rd, FPR16INX:$rt, FPR16INX:$rs), 0>;
def : InstAlias<"fge.h $rd, $rs, $rt",
                (FLE_H_INX GPR:$rd, FPR16INX:$rt, FPR16INX:$rs), 0>;
} // Predicates = [HasStdExtZhinx]

//===----------------------------------------------------------------------===//
// Pseudo-instructions and codegen patterns
//===----------------------------------------------------------------------===//

let Predicates = [HasStdExtZfh] in {

/// Float constants
def : Pat<(f16 (fpimm0)), (FMV_H_X X0)>;
def : Pat<(f16 (fpimmneg0)), (FSGNJN_H (FMV_H_X X0), (FMV_H_X X0))>;

/// Float conversion operations

// [u]int32<->float conversion patterns must be gated on IsRV32 or IsRV64, so
// are defined later.

/// Float arithmetic operations

def : PatFprFprDynFrm<any_fadd, FADD_H, FPR16>;
def : PatFprFprDynFrm<any_fsub, FSUB_H, FPR16>;
def : PatFprFprDynFrm<any_fmul, FMUL_H, FPR16>;
def : PatFprFprDynFrm<any_fdiv, FDIV_H, FPR16>;

def : Pat<(any_fsqrt FPR16:$rs1), (FSQRT_H FPR16:$rs1, 0b111)>;

def : Pat<(fneg FPR16:$rs1), (FSGNJN_H $rs1, $rs1)>;
def : Pat<(fabs FPR16:$rs1), (FSGNJX_H $rs1, $rs1)>;

def : PatFprFpr<fcopysign, FSGNJ_H, FPR16>;
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

// fnmadd: -(rs1 * rs2 + rs3) (the nsz flag on the FMA)
def : Pat<(fneg (any_fma_nsz FPR16:$rs1, FPR16:$rs2, FPR16:$rs3)),
          (FNMADD_H FPR16:$rs1, FPR16:$rs2, FPR16:$rs3, 0b111)>;

// The ratified 20191213 ISA spec defines fmin and fmax in a way that matches
// LLVM's fminnum and fmaxnum
// <https://github.com/riscv/riscv-isa-manual/commit/cd20cee7efd9bac7c5aa127ec3b451749d2b3cce>.
def : PatFprFpr<fminnum, FMIN_H, FPR16>;
def : PatFprFpr<fmaxnum, FMAX_H, FPR16>;

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

defm Select_FPR16 : SelectCC_GPR_rrirr<FPR16>;
} // Predicates = [HasStdExtZfh]

let Predicates = [HasStdExtZfhOrZfhmin] in {
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
def : Pat<(riscv_fmv_x_signexth FPR16:$src), (FMV_X_H FPR16:$src)>;
} // Predicates = [HasStdExtZfhOrZfhmin]

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

let Predicates = [HasStdExtZfhOrZfhmin, HasStdExtD] in {
/// Float conversion operations
// f64 -> f16, f16 -> f64
def : Pat<(any_fpround FPR64:$rs1), (FCVT_H_D FPR64:$rs1, 0b111)>;
def : Pat<(any_fpextend FPR16:$rs1), (FCVT_D_H FPR16:$rs1)>;

/// Float arithmetic operations
def : Pat<(fcopysign FPR16:$rs1, FPR64:$rs2),
          (FSGNJ_H $rs1, (FCVT_H_D $rs2, 0b111))>;
def : Pat<(fcopysign FPR64:$rs1, FPR16:$rs2), (FSGNJ_D $rs1, (FCVT_D_H $rs2))>;
} // Predicates = [HasStdExtZfhOrZfhmin, HasStdExtD]

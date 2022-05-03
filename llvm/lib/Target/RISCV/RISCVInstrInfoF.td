//===-- RISCVInstrInfoF.td - RISC-V 'F' instructions -------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file describes the RISC-V instructions from the standard 'F',
// Single-Precision Floating-Point instruction set extension.
//
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// RISC-V specific DAG Nodes.
//===----------------------------------------------------------------------===//

def SDT_RISCVFMV_W_X_RV64
    : SDTypeProfile<1, 1, [SDTCisVT<0, f32>, SDTCisVT<1, i64>]>;
def SDT_RISCVFMV_X_ANYEXTW_RV64
    : SDTypeProfile<1, 1, [SDTCisVT<0, i64>, SDTCisVT<1, f32>]>;
def SDT_RISCVFCVT_W_RV64
    : SDTypeProfile<1, 2, [SDTCisVT<0, i64>, SDTCisFP<1>,
                           SDTCisVT<2, i64>]>;
def SDT_RISCVFCVT_X
    : SDTypeProfile<1, 2, [SDTCisVT<0, XLenVT>, SDTCisFP<1>,
                           SDTCisVT<2, XLenVT>]>;

def riscv_fmv_w_x_rv64
    : SDNode<"RISCVISD::FMV_W_X_RV64", SDT_RISCVFMV_W_X_RV64>;
def riscv_fmv_x_anyextw_rv64
    : SDNode<"RISCVISD::FMV_X_ANYEXTW_RV64", SDT_RISCVFMV_X_ANYEXTW_RV64>;
def riscv_fcvt_w_rv64
    : SDNode<"RISCVISD::FCVT_W_RV64", SDT_RISCVFCVT_W_RV64>;
def riscv_fcvt_wu_rv64
    : SDNode<"RISCVISD::FCVT_WU_RV64", SDT_RISCVFCVT_W_RV64>;
def riscv_fcvt_x
    : SDNode<"RISCVISD::FCVT_X", SDT_RISCVFCVT_X>;
def riscv_fcvt_xu
    : SDNode<"RISCVISD::FCVT_XU", SDT_RISCVFCVT_X>;

def riscv_strict_fcvt_w_rv64
    : SDNode<"RISCVISD::STRICT_FCVT_W_RV64", SDT_RISCVFCVT_W_RV64,
             [SDNPHasChain]>;
def riscv_strict_fcvt_wu_rv64
    : SDNode<"RISCVISD::STRICT_FCVT_WU_RV64", SDT_RISCVFCVT_W_RV64,
             [SDNPHasChain]>;

def riscv_any_fcvt_w_rv64 : PatFrags<(ops node:$src, node:$frm),
                                     [(riscv_strict_fcvt_w_rv64 node:$src, node:$frm),
                                      (riscv_fcvt_w_rv64 node:$src, node:$frm)]>;
def riscv_any_fcvt_wu_rv64 : PatFrags<(ops node:$src, node:$frm),
                                      [(riscv_strict_fcvt_wu_rv64 node:$src, node:$frm),
                                       (riscv_fcvt_wu_rv64 node:$src, node:$frm)]>;

//===----------------------------------------------------------------------===//
// Operand and SDNode transformation definitions.
//===----------------------------------------------------------------------===//

// Zfinx

def GPRAsFPR : AsmOperandClass {
  let Name = "GPRAsFPR";
  let ParserMethod = "parseGPRAsFPR";
  let RenderMethod = "addRegOperands";
}

def FPR32INX : RegisterOperand<GPRF32> {
  let ParserMatchClass = GPRAsFPR;
  let DecoderMethod = "DecodeGPRRegisterClass";
}

// inx = 0 : f, d, zfh, zfhmin
//     = 1 : zfinx, zdinx, zhinx, zhinxmin
//     = 2 : zdinx_rv32
class ExtInfo<bits<2> inx, list<Predicate> pres> {
  string Suffix = !cond(!eq(inx, 0): "",
                        !eq(inx, 1): "_INX",
                        !eq(inx, 2): "_IN32X");
  list<Predicate> Predicates = pres;
  string Space = !cond(!eq(inx, 0): "",
                       !eq(inx, 1): "RVZfinx",
                       !eq(inx, 2): "RV32Zdinx");
}

class ExtInfo_r<ExtInfo ext, DAGOperand reg> {
  string Suffix = ext.Suffix;
  list<Predicate> Predicates = ext.Predicates;
  string Space = ext.Space;
  DAGOperand Reg = reg;
}

class ExtInfo_rr<ExtInfo ext, DAGOperand rdty, DAGOperand rs1ty> {
  string Suffix = ext.Suffix;
  list<Predicate> Predicates = ext.Predicates;
  string Space = ext.Space;
  DAGOperand RdTy = rdty;
  DAGOperand Rs1Ty = rs1ty;
}

def FExt       : ExtInfo<0, [HasStdExtF]>;
def F64Ext     : ExtInfo<0, [HasStdExtF, IsRV64]>;
def ZfinxExt   : ExtInfo<1, [HasStdExtZfinx]>;
def Zfinx64Ext : ExtInfo<1, [HasStdExtZfinx, IsRV64]>;

def F      : ExtInfo_r<FExt,     FPR32>;
def F_INX  : ExtInfo_r<ZfinxExt, FPR32INX>;

def FF        : ExtInfo_rr<FExt,       FPR32,    FPR32>;
def FF_INX    : ExtInfo_rr<ZfinxExt,   FPR32INX, FPR32INX>;
def FX        : ExtInfo_rr<FExt,       FPR32,    GPR>;
def FX_INX    : ExtInfo_rr<ZfinxExt,   FPR32INX, GPR>;
def FX_64     : ExtInfo_rr<F64Ext,     FPR32,    GPR>;
def FX_INX_64 : ExtInfo_rr<Zfinx64Ext, FPR32INX, GPR>;
def XF        : ExtInfo_rr<FExt,       GPR,      FPR32>;
def XF_64     : ExtInfo_rr<F64Ext,     GPR,      FPR32>;
def XF_INX    : ExtInfo_rr<ZfinxExt,   GPR,      FPR32INX>;
def XF_INX_64 : ExtInfo_rr<Zfinx64Ext, GPR,      FPR32INX>;

defvar FINX    = [F,     F_INX];
defvar FFINX   = [FF,    FF_INX];
defvar FXINX   = [FX,    FX_INX];
defvar XFINX   = [XF,    XF_INX];
defvar XFIN64X = [XF_64, XF_INX_64];
defvar FXIN64X = [FX_64, FX_INX_64];

// Floating-point rounding mode

def FRMArg : AsmOperandClass {
  let Name = "FRMArg";
  let RenderMethod = "addFRMArgOperands";
  let DiagnosticType = "InvalidFRMArg";
}

def frmarg : Operand<XLenVT> {
  let ParserMatchClass = FRMArg;
  let PrintMethod = "printFRMArg";
  let DecoderMethod = "decodeFRMArg";
}

//===----------------------------------------------------------------------===//
// Instruction class templates
//===----------------------------------------------------------------------===//

let hasSideEffects = 0, mayLoad = 1, mayStore = 0 in
class FPLoad_r<bits<3> funct3, string opcodestr, RegisterClass rty,
               SchedWrite sw>
    : RVInstI<funct3, OPC_LOAD_FP, (outs rty:$rd),
              (ins GPR:$rs1, simm12:$imm12),
              opcodestr, "$rd, ${imm12}(${rs1})">,
      Sched<[sw, ReadFMemBase]>;

let hasSideEffects = 0, mayLoad = 0, mayStore = 1 in
class FPStore_r<bits<3> funct3, string opcodestr, RegisterClass rty,
                SchedWrite sw>
    : RVInstS<funct3, OPC_STORE_FP, (outs),
              (ins rty:$rs2, GPR:$rs1, simm12:$imm12),
              opcodestr, "$rs2, ${imm12}(${rs1})">,
      Sched<[sw, ReadStoreData, ReadFMemBase]>;

let hasSideEffects = 0, mayLoad = 0, mayStore = 0, mayRaiseFPException = 1,
    UseNamedOperandTable = 1, hasPostISelHook = 1, isCommutable = 1 in
class FPFMA_rrr_frm<RISCVOpcode opcode, bits<2> funct2, string opcodestr,
                    DAGOperand rty>
    : RVInstR4Frm<funct2, opcode, (outs rty:$rd),
                  (ins rty:$rs1, rty:$rs2, rty:$rs3, frmarg:$frm),
                  opcodestr, "$rd, $rs1, $rs2, $rs3, $frm">;

multiclass FPFMA_rrr_frm_m<RISCVOpcode opcode, bits<2> funct2,
                           string opcodestr, list<ExtInfo_r> Exts> {
  foreach Ext = Exts in
    let Predicates = Ext.Predicates, DecoderNamespace = Ext.Space in
    def Ext.Suffix : FPFMA_rrr_frm<opcode, funct2, opcodestr, Ext.Reg>;
}

class FPFMADynFrmAlias<FPFMA_rrr_frm Inst, string OpcodeStr,
                       DAGOperand rty>
    : InstAlias<OpcodeStr#" $rd, $rs1, $rs2, $rs3",
                (Inst rty:$rd, rty:$rs1, rty:$rs2, rty:$rs3, 0b111)>;
multiclass FPFMADynFrmAlias_m<FPFMA_rrr_frm Inst, string OpcodeStr,
                              list<ExtInfo_r> Exts> {
  foreach Ext = Exts in
    let Predicates = Ext.Predicates in
    def : FPFMADynFrmAlias<!cast<FPFMA_rrr_frm>(Inst#Ext.Suffix), OpcodeStr,
                           Ext.Reg>;
}

let hasSideEffects = 0, mayLoad = 0, mayStore = 0, mayRaiseFPException = 1 in
class FPALU_rr<bits<7> funct7, bits<3> funct3, string opcodestr,
               DAGOperand rty, bit Commutable>
    : RVInstR<funct7, funct3, OPC_OP_FP, (outs rty:$rd),
              (ins rty:$rs1, rty:$rs2), opcodestr, "$rd, $rs1, $rs2"> {
  let isCommutable = Commutable;
}
multiclass FPALU_rr_m<bits<7> funct7, bits<3> funct3, string opcodestr,
                      list<ExtInfo_r> Exts, bit Commutable = 0> {
  foreach Ext = Exts in
    let Predicates = Ext.Predicates, DecoderNamespace = Ext.Space in
    def Ext.Suffix : FPALU_rr<funct7, funct3, opcodestr, Ext.Reg, Commutable>;
}

let hasSideEffects = 0, mayLoad = 0, mayStore = 0, mayRaiseFPException = 1,
    UseNamedOperandTable = 1, hasPostISelHook = 1 in
class FPALU_rr_frm<bits<7> funct7, string opcodestr, DAGOperand rty,
                   bit Commutable>
    : RVInstRFrm<funct7, OPC_OP_FP, (outs rty:$rd),
                 (ins rty:$rs1, rty:$rs2, frmarg:$frm), opcodestr,
                  "$rd, $rs1, $rs2, $frm"> {
  let isCommutable = Commutable;
}
multiclass FPALU_rr_frm_m<bits<7> funct7, string opcodestr,
                          list<ExtInfo_r> Exts, bit Commutable = 0> {
  foreach Ext = Exts in
    let Predicates = Ext.Predicates, DecoderNamespace = Ext.Space in
    def Ext.Suffix : FPALU_rr_frm<funct7, opcodestr, Ext.Reg, Commutable>;
}

class FPALUDynFrmAlias<FPALU_rr_frm Inst, string OpcodeStr,
                       DAGOperand rty>
    : InstAlias<OpcodeStr#" $rd, $rs1, $rs2",
                (Inst rty:$rd, rty:$rs1, rty:$rs2, 0b111)>;
multiclass FPALUDynFrmAlias_m<FPALU_rr_frm Inst, string OpcodeStr,
                              list<ExtInfo_r> Exts> {
  foreach Ext = Exts in
    let Predicates = Ext.Predicates in
    def : FPALUDynFrmAlias<!cast<FPALU_rr_frm>(Inst#Ext.Suffix), OpcodeStr,
                           Ext.Reg>;
}

let hasSideEffects = 0, mayLoad = 0, mayStore = 0, mayRaiseFPException = 1 in
class FPUnaryOp_r<bits<7> funct7, bits<5> rs2val, bits<3> funct3,
                  DAGOperand rdty, DAGOperand rs1ty, string opcodestr>
    : RVInstR<funct7, funct3, OPC_OP_FP, (outs rdty:$rd), (ins rs1ty:$rs1),
              opcodestr, "$rd, $rs1"> {
  let rs2 = rs2val;
}
multiclass FPUnaryOp_r_m<bits<7> funct7, bits<5> rs2val, bits<3> funct3,
                         list<ExtInfo_rr> Exts, string opcodestr> {
  foreach Ext = Exts in
    let Predicates = Ext.Predicates, DecoderNamespace = Ext.Space in
    def Ext.Suffix : FPUnaryOp_r<funct7, rs2val, funct3, Ext.RdTy, Ext.Rs1Ty,
                                 opcodestr>;
}

let hasSideEffects = 0, mayLoad = 0, mayStore = 0, mayRaiseFPException = 1,
    UseNamedOperandTable = 1, hasPostISelHook = 1 in
class FPUnaryOp_r_frm<bits<7> funct7, bits<5> rs2val, DAGOperand rdty,
                      DAGOperand rs1ty, string opcodestr>
    : RVInstRFrm<funct7, OPC_OP_FP, (outs rdty:$rd),
                 (ins rs1ty:$rs1, frmarg:$frm), opcodestr,
                  "$rd, $rs1, $frm"> {
  let rs2 = rs2val;
}
multiclass FPUnaryOp_r_frm_m<bits<7> funct7, bits<5> rs2val,
                             list<ExtInfo_rr> Exts, string opcodestr> {
  foreach Ext = Exts in
    let Predicates = Ext.Predicates, DecoderNamespace = Ext.Space in
    def Ext.Suffix : FPUnaryOp_r_frm<funct7, rs2val, Ext.RdTy, Ext.Rs1Ty,
                                     opcodestr>;
}

class FPUnaryOpDynFrmAlias<FPUnaryOp_r_frm Inst, string OpcodeStr,
                           DAGOperand rdty, DAGOperand rs1ty>
    : InstAlias<OpcodeStr#" $rd, $rs1",
                (Inst rdty:$rd, rs1ty:$rs1, 0b111)>;
multiclass FPUnaryOpDynFrmAlias_m<FPUnaryOp_r_frm Inst, string OpcodeStr,
                                  list<ExtInfo_rr> Exts> {
  foreach Ext = Exts in
    let Predicates = Ext.Predicates in
    def : FPUnaryOpDynFrmAlias<!cast<FPUnaryOp_r_frm>(Inst#Ext.Suffix),
                               OpcodeStr, Ext.RdTy, Ext.Rs1Ty>;
}

let hasSideEffects = 0, mayLoad = 0, mayStore = 0, mayRaiseFPException = 1 in
class FPCmp_rr<bits<7> funct7, bits<3> funct3, string opcodestr,
               DAGOperand rty, bit Commutable>
    : RVInstR<funct7, funct3, OPC_OP_FP, (outs GPR:$rd),
              (ins rty:$rs1, rty:$rs2), opcodestr, "$rd, $rs1, $rs2"> {
  let isCommutable = Commutable;
}
multiclass FPCmp_rr_m<bits<7> funct7, bits<3> funct3, string opcodestr,
                      list<ExtInfo_r> Exts, bit Commutable = 0> {
  foreach Ext = Exts in
    let Predicates = Ext.Predicates, DecoderNamespace = Ext.Space in
    def Ext.Suffix : FPCmp_rr<funct7, funct3, opcodestr, Ext.Reg, Commutable>;
}

//===----------------------------------------------------------------------===//
// Instructions
//===----------------------------------------------------------------------===//

let Predicates = [HasStdExtF] in {
def FLW : FPLoad_r<0b010, "flw", FPR32, WriteFLD32>;

// Operands for stores are in the order srcreg, base, offset rather than
// reflecting the order these fields are specified in the instruction
// encoding.
def FSW : FPStore_r<0b010, "fsw", FPR32, WriteFST32>;
} // Predicates = [HasStdExtF]

let SchedRW = [WriteFMA32, ReadFMA32, ReadFMA32, ReadFMA32] in {
defm FMADD_S  : FPFMA_rrr_frm_m<OPC_MADD,  0b00, "fmadd.s",  FINX>;
defm FMSUB_S  : FPFMA_rrr_frm_m<OPC_MSUB,  0b00, "fmsub.s",  FINX>;
defm FNMSUB_S : FPFMA_rrr_frm_m<OPC_NMSUB, 0b00, "fnmsub.s", FINX>;
defm FNMADD_S : FPFMA_rrr_frm_m<OPC_NMADD, 0b00, "fnmadd.s", FINX>;
}

defm : FPFMADynFrmAlias_m<FMADD_S,  "fmadd.s",  FINX>;
defm : FPFMADynFrmAlias_m<FMSUB_S,  "fmsub.s",  FINX>;
defm : FPFMADynFrmAlias_m<FNMSUB_S, "fnmsub.s", FINX>;
defm : FPFMADynFrmAlias_m<FNMADD_S, "fnmadd.s", FINX>;

let SchedRW = [WriteFALU32, ReadFALU32, ReadFALU32] in {
defm FADD_S : FPALU_rr_frm_m<0b0000000, "fadd.s", FINX, /*Commutable*/1>;
defm FSUB_S : FPALU_rr_frm_m<0b0000100, "fsub.s", FINX>;
}
let SchedRW = [WriteFMul32, ReadFMul32, ReadFMul32] in
defm FMUL_S : FPALU_rr_frm_m<0b0001000, "fmul.s", FINX, /*Commutable*/1>;

let SchedRW = [WriteFDiv32, ReadFDiv32, ReadFDiv32] in
defm FDIV_S : FPALU_rr_frm_m<0b0001100, "fdiv.s", FINX>;

defm : FPALUDynFrmAlias_m<FADD_S, "fadd.s", FINX>;
defm : FPALUDynFrmAlias_m<FSUB_S, "fsub.s", FINX>;
defm : FPALUDynFrmAlias_m<FMUL_S, "fmul.s", FINX>;
defm : FPALUDynFrmAlias_m<FDIV_S, "fdiv.s", FINX>;

defm FSQRT_S : FPUnaryOp_r_frm_m<0b0101100, 0b00000, FFINX, "fsqrt.s">,
               Sched<[WriteFSqrt32, ReadFSqrt32]>;
defm         : FPUnaryOpDynFrmAlias_m<FSQRT_S, "fsqrt.s", FFINX>;

let SchedRW = [WriteFSGNJ32, ReadFSGNJ32, ReadFSGNJ32],
    mayRaiseFPException = 0 in {
defm FSGNJ_S  : FPALU_rr_m<0b0010000, 0b000, "fsgnj.s",  FINX>;
defm FSGNJN_S : FPALU_rr_m<0b0010000, 0b001, "fsgnjn.s", FINX>;
defm FSGNJX_S : FPALU_rr_m<0b0010000, 0b010, "fsgnjx.s", FINX>;
}

let SchedRW = [WriteFMinMax32, ReadFMinMax32, ReadFMinMax32] in {
defm FMIN_S   : FPALU_rr_m<0b0010100, 0b000, "fmin.s", FINX, /*Commutable*/1>;
defm FMAX_S   : FPALU_rr_m<0b0010100, 0b001, "fmax.s", FINX, /*Commutable*/1>;
}

defm FCVT_W_S : FPUnaryOp_r_frm_m<0b1100000, 0b00000, XFINX, "fcvt.w.s">,
                Sched<[WriteFCvtF32ToI32, ReadFCvtF32ToI32]>;
defm          : FPUnaryOpDynFrmAlias_m<FCVT_W_S, "fcvt.w.s", XFINX>;

defm FCVT_WU_S : FPUnaryOp_r_frm_m<0b1100000, 0b00001, XFINX, "fcvt.wu.s">,
                 Sched<[WriteFCvtF32ToI32, ReadFCvtF32ToI32]>;
defm           : FPUnaryOpDynFrmAlias_m<FCVT_WU_S, "fcvt.wu.s", XFINX>;

let mayRaiseFPException = 0 in
def FMV_X_W : FPUnaryOp_r<0b1110000, 0b00000, 0b000, GPR, FPR32, "fmv.x.w">,
              Sched<[WriteFMovF32ToI32, ReadFMovF32ToI32]>;

let SchedRW = [WriteFCmp32, ReadFCmp32, ReadFCmp32] in {
defm FEQ_S : FPCmp_rr_m<0b1010000, 0b010, "feq.s", FINX, /*Commutable*/1>;
defm FLT_S : FPCmp_rr_m<0b1010000, 0b001, "flt.s", FINX>;
defm FLE_S : FPCmp_rr_m<0b1010000, 0b000, "fle.s", FINX>;
}

let mayRaiseFPException = 0 in
defm FCLASS_S : FPUnaryOp_r_m<0b1110000, 0b00000, 0b001, XFINX, "fclass.s">,
                Sched<[WriteFClass32, ReadFClass32]>;

defm FCVT_S_W : FPUnaryOp_r_frm_m<0b1101000, 0b00000, FXINX, "fcvt.s.w">,
                Sched<[WriteFCvtI32ToF32, ReadFCvtI32ToF32]>;
defm          : FPUnaryOpDynFrmAlias_m<FCVT_S_W, "fcvt.s.w", FXINX>;

defm FCVT_S_WU : FPUnaryOp_r_frm_m<0b1101000, 0b00001, FXINX, "fcvt.s.wu">,
                 Sched<[WriteFCvtI32ToF32, ReadFCvtI32ToF32]>;
defm           : FPUnaryOpDynFrmAlias_m<FCVT_S_WU, "fcvt.s.wu", FXINX>;

let mayRaiseFPException = 0 in
def FMV_W_X : FPUnaryOp_r<0b1111000, 0b00000, 0b000, FPR32, GPR, "fmv.w.x">,
              Sched<[WriteFMovI32ToF32, ReadFMovI32ToF32]>;

defm FCVT_L_S  : FPUnaryOp_r_frm_m<0b1100000, 0b00010, XFIN64X, "fcvt.l.s">,
                 Sched<[WriteFCvtF32ToI64, ReadFCvtF32ToI64]>;
defm           : FPUnaryOpDynFrmAlias_m<FCVT_L_S, "fcvt.l.s", XFIN64X>;

defm FCVT_LU_S  : FPUnaryOp_r_frm_m<0b1100000, 0b00011, XFIN64X, "fcvt.lu.s">,
                  Sched<[WriteFCvtF32ToI64, ReadFCvtF32ToI64]>;
defm            : FPUnaryOpDynFrmAlias_m<FCVT_LU_S, "fcvt.lu.s", XFIN64X>;

defm FCVT_S_L : FPUnaryOp_r_frm_m<0b1101000, 0b00010, FXIN64X, "fcvt.s.l">,
                Sched<[WriteFCvtI64ToF32, ReadFCvtI64ToF32]>;
defm          : FPUnaryOpDynFrmAlias_m<FCVT_S_L, "fcvt.s.l", FXIN64X>;

defm FCVT_S_LU : FPUnaryOp_r_frm_m<0b1101000, 0b00011, FXIN64X, "fcvt.s.lu">,
                 Sched<[WriteFCvtI64ToF32, ReadFCvtI64ToF32]>;
defm           : FPUnaryOpDynFrmAlias_m<FCVT_S_LU, "fcvt.s.lu", FXIN64X>;

//===----------------------------------------------------------------------===//
// Assembler Pseudo Instructions (User-Level ISA, Version 2.2, Chapter 20)
//===----------------------------------------------------------------------===//

let Predicates = [HasStdExtF] in {
def : InstAlias<"flw $rd, (${rs1})",  (FLW FPR32:$rd,  GPR:$rs1, 0), 0>;
def : InstAlias<"fsw $rs2, (${rs1})", (FSW FPR32:$rs2, GPR:$rs1, 0), 0>;

def : InstAlias<"fmv.s $rd, $rs",  (FSGNJ_S  FPR32:$rd, FPR32:$rs, FPR32:$rs)>;
def : InstAlias<"fabs.s $rd, $rs", (FSGNJX_S FPR32:$rd, FPR32:$rs, FPR32:$rs)>;
def : InstAlias<"fneg.s $rd, $rs", (FSGNJN_S FPR32:$rd, FPR32:$rs, FPR32:$rs)>;

// fgt.s/fge.s are recognised by the GNU assembler but the canonical
// flt.s/fle.s forms will always be printed. Therefore, set a zero weight.
def : InstAlias<"fgt.s $rd, $rs, $rt",
                (FLT_S GPR:$rd, FPR32:$rt, FPR32:$rs), 0>;
def : InstAlias<"fge.s $rd, $rs, $rt",
                (FLE_S GPR:$rd, FPR32:$rt, FPR32:$rs), 0>;

// The following csr instructions actually alias instructions from the base ISA.
// However, it only makes sense to support them when the F extension is enabled.
// NOTE: "frcsr", "frrm", and "frflags" are more specialized version of "csrr".
def : InstAlias<"frcsr $rd",      (CSRRS GPR:$rd, SysRegFCSR.Encoding, X0), 2>;
def : InstAlias<"fscsr $rd, $rs", (CSRRW GPR:$rd, SysRegFCSR.Encoding, GPR:$rs)>;
def : InstAlias<"fscsr $rs",      (CSRRW      X0, SysRegFCSR.Encoding, GPR:$rs), 2>;

// frsr, fssr are obsolete aliases replaced by frcsr, fscsr, so give them
// zero weight.
def : InstAlias<"frsr $rd",       (CSRRS GPR:$rd, SysRegFCSR.Encoding, X0), 0>;
def : InstAlias<"fssr $rd, $rs",  (CSRRW GPR:$rd, SysRegFCSR.Encoding, GPR:$rs), 0>;
def : InstAlias<"fssr $rs",       (CSRRW      X0, SysRegFCSR.Encoding, GPR:$rs), 0>;

def : InstAlias<"frrm $rd",        (CSRRS  GPR:$rd, SysRegFRM.Encoding, X0), 2>;
def : InstAlias<"fsrm $rd, $rs",   (CSRRW  GPR:$rd, SysRegFRM.Encoding, GPR:$rs)>;
def : InstAlias<"fsrm $rs",        (CSRRW       X0, SysRegFRM.Encoding, GPR:$rs), 2>;
def : InstAlias<"fsrmi $rd, $imm", (CSRRWI GPR:$rd, SysRegFRM.Encoding, uimm5:$imm)>;
def : InstAlias<"fsrmi $imm",      (CSRRWI      X0, SysRegFRM.Encoding, uimm5:$imm), 2>;

def : InstAlias<"frflags $rd",        (CSRRS  GPR:$rd, SysRegFFLAGS.Encoding, X0), 2>;
def : InstAlias<"fsflags $rd, $rs",   (CSRRW  GPR:$rd, SysRegFFLAGS.Encoding, GPR:$rs)>;
def : InstAlias<"fsflags $rs",        (CSRRW       X0, SysRegFFLAGS.Encoding, GPR:$rs), 2>;
def : InstAlias<"fsflagsi $rd, $imm", (CSRRWI GPR:$rd, SysRegFFLAGS.Encoding, uimm5:$imm)>;
def : InstAlias<"fsflagsi $imm",      (CSRRWI      X0, SysRegFFLAGS.Encoding, uimm5:$imm), 2>;

// fmv.w.x and fmv.x.w were previously known as fmv.s.x and fmv.x.s. Both
// spellings should be supported by standard tools.
def : MnemonicAlias<"fmv.s.x", "fmv.w.x">;
def : MnemonicAlias<"fmv.x.s", "fmv.x.w">;

def PseudoFLW  : PseudoFloatLoad<"flw", FPR32>;
def PseudoFSW  : PseudoStore<"fsw", FPR32>;
let usesCustomInserter = 1 in {
def PseudoQuietFLE_S : PseudoQuietFCMP<FPR32>;
def PseudoQuietFLT_S : PseudoQuietFCMP<FPR32>;
}
} // Predicates = [HasStdExtF]

let Predicates = [HasStdExtZfinx] in {
def : InstAlias<"fabs.s $rd, $rs", (FSGNJX_S_INX FPR32INX:$rd, FPR32INX:$rs, FPR32INX:$rs)>;
def : InstAlias<"fneg.s $rd, $rs", (FSGNJN_S_INX FPR32INX:$rd, FPR32INX:$rs, FPR32INX:$rs)>;

def : InstAlias<"fgt.s $rd, $rs, $rt",
                (FLT_S_INX GPR:$rd, FPR32INX:$rt, FPR32INX:$rs), 0>;
def : InstAlias<"fge.s $rd, $rs, $rt",
                (FLE_S_INX GPR:$rd, FPR32INX:$rt, FPR32INX:$rs), 0>;
} // Predicates = [HasStdExtZfinx]

//===----------------------------------------------------------------------===//
// Pseudo-instructions and codegen patterns
//===----------------------------------------------------------------------===//

/// Floating point constants
def fpimm0    : PatLeaf<(fpimm), [{ return N->isExactlyValue(+0.0); }]>;
def fpimmneg0 : PatLeaf<(fpimm), [{ return N->isExactlyValue(-0.0); }]>;

/// Generic pattern classes
class PatSetCC<RegisterClass Ty, SDPatternOperator OpNode, CondCode Cond, RVInst Inst>
    : Pat<(OpNode Ty:$rs1, Ty:$rs2, Cond), (Inst $rs1, $rs2)>;

class PatFprFpr<SDPatternOperator OpNode, RVInstR Inst,
                RegisterClass RegTy>
    : Pat<(OpNode RegTy:$rs1, RegTy:$rs2), (Inst $rs1, $rs2)>;

class PatFprFprDynFrm<SDPatternOperator OpNode, RVInstRFrm Inst,
                      RegisterClass RegTy>
    : Pat<(OpNode RegTy:$rs1, RegTy:$rs2), (Inst $rs1, $rs2, 0b111)>;

let Predicates = [HasStdExtF] in {

/// Float constants
def : Pat<(f32 (fpimm0)), (FMV_W_X X0)>;
def : Pat<(f32 (fpimmneg0)), (FSGNJN_S (FMV_W_X X0), (FMV_W_X X0))>;

/// Float conversion operations

// [u]int32<->float conversion patterns must be gated on IsRV32 or IsRV64, so
// are defined later.

/// Float arithmetic operations

def : PatFprFprDynFrm<any_fadd, FADD_S, FPR32>;
def : PatFprFprDynFrm<any_fsub, FSUB_S, FPR32>;
def : PatFprFprDynFrm<any_fmul, FMUL_S, FPR32>;
def : PatFprFprDynFrm<any_fdiv, FDIV_S, FPR32>;

def : Pat<(any_fsqrt FPR32:$rs1), (FSQRT_S FPR32:$rs1, 0b111)>;

def : Pat<(fneg FPR32:$rs1), (FSGNJN_S $rs1, $rs1)>;
def : Pat<(fabs FPR32:$rs1), (FSGNJX_S $rs1, $rs1)>;

def : PatFprFpr<fcopysign, FSGNJ_S, FPR32>;
def : Pat<(fcopysign FPR32:$rs1, (fneg FPR32:$rs2)), (FSGNJN_S $rs1, $rs2)>;

// fmadd: rs1 * rs2 + rs3
def : Pat<(any_fma FPR32:$rs1, FPR32:$rs2, FPR32:$rs3),
          (FMADD_S $rs1, $rs2, $rs3, 0b111)>;

// fmsub: rs1 * rs2 - rs3
def : Pat<(any_fma FPR32:$rs1, FPR32:$rs2, (fneg FPR32:$rs3)),
          (FMSUB_S FPR32:$rs1, FPR32:$rs2, FPR32:$rs3, 0b111)>;

// fnmsub: -rs1 * rs2 + rs3
def : Pat<(any_fma (fneg FPR32:$rs1), FPR32:$rs2, FPR32:$rs3),
          (FNMSUB_S FPR32:$rs1, FPR32:$rs2, FPR32:$rs3, 0b111)>;

// fnmadd: -rs1 * rs2 - rs3
def : Pat<(any_fma (fneg FPR32:$rs1), FPR32:$rs2, (fneg FPR32:$rs3)),
          (FNMADD_S FPR32:$rs1, FPR32:$rs2, FPR32:$rs3, 0b111)>;

// The ratified 20191213 ISA spec defines fmin and fmax in a way that matches
// LLVM's fminnum and fmaxnum
// <https://github.com/riscv/riscv-isa-manual/commit/cd20cee7efd9bac7c5aa127ec3b451749d2b3cce>.
def : PatFprFpr<fminnum, FMIN_S, FPR32>;
def : PatFprFpr<fmaxnum, FMAX_S, FPR32>;

/// Setcc
// FIXME: SETEQ/SETLT/SETLE imply nonans, can we pick better instructions for
// strict versions of those.

// Match non-signaling FEQ_S
def : PatSetCC<FPR32, any_fsetcc, SETEQ, FEQ_S>;
def : PatSetCC<FPR32, any_fsetcc, SETOEQ, FEQ_S>;
def : PatSetCC<FPR32, strict_fsetcc, SETLT, PseudoQuietFLT_S>;
def : PatSetCC<FPR32, strict_fsetcc, SETOLT, PseudoQuietFLT_S>;
def : PatSetCC<FPR32, strict_fsetcc, SETLE, PseudoQuietFLE_S>;
def : PatSetCC<FPR32, strict_fsetcc, SETOLE, PseudoQuietFLE_S>;

// Match signaling FEQ_S
def : Pat<(strict_fsetccs FPR32:$rs1, FPR32:$rs2, SETEQ),
          (AND (FLE_S $rs1, $rs2),
               (FLE_S $rs2, $rs1))>;
def : Pat<(strict_fsetccs FPR32:$rs1, FPR32:$rs2, SETOEQ),
          (AND (FLE_S $rs1, $rs2),
               (FLE_S $rs2, $rs1))>;
// If both operands are the same, use a single FLE.
def : Pat<(strict_fsetccs FPR32:$rs1, FPR32:$rs1, SETEQ),
          (FLE_S $rs1, $rs1)>;
def : Pat<(strict_fsetccs FPR32:$rs1, FPR32:$rs1, SETOEQ),
          (FLE_S $rs1, $rs1)>;

def : PatSetCC<FPR32, any_fsetccs, SETLT, FLT_S>;
def : PatSetCC<FPR32, any_fsetccs, SETOLT, FLT_S>;
def : PatSetCC<FPR32, any_fsetccs, SETLE, FLE_S>;
def : PatSetCC<FPR32, any_fsetccs, SETOLE, FLE_S>;

def Select_FPR32_Using_CC_GPR : SelectCC_rrirr<FPR32, GPR>;

/// Loads

defm : LdPat<load, FLW, f32>;

/// Stores

defm : StPat<store, FSW, FPR32, f32>;

} // Predicates = [HasStdExtF]

let Predicates = [HasStdExtF, IsRV32] in {
// Moves (no conversion)
def : Pat<(bitconvert (i32 GPR:$rs1)), (FMV_W_X GPR:$rs1)>;
def : Pat<(i32 (bitconvert FPR32:$rs1)), (FMV_X_W FPR32:$rs1)>;

// float->[u]int. Round-to-zero must be used.
def : Pat<(i32 (any_fp_to_sint FPR32:$rs1)), (FCVT_W_S $rs1, 0b001)>;
def : Pat<(i32 (any_fp_to_uint FPR32:$rs1)), (FCVT_WU_S $rs1, 0b001)>;

// Saturating float->[u]int32.
def : Pat<(i32 (riscv_fcvt_x FPR32:$rs1, timm:$frm)), (FCVT_W_S $rs1, timm:$frm)>;
def : Pat<(i32 (riscv_fcvt_xu FPR32:$rs1, timm:$frm)), (FCVT_WU_S $rs1, timm:$frm)>;

// float->int32 with current rounding mode.
def : Pat<(i32 (any_lrint FPR32:$rs1)), (FCVT_W_S $rs1, 0b111)>;

// float->int32 rounded to nearest with ties rounded away from zero.
def : Pat<(i32 (any_lround FPR32:$rs1)), (FCVT_W_S $rs1, 0b100)>;

// [u]int->float. Match GCC and default to using dynamic rounding mode.
def : Pat<(any_sint_to_fp (i32 GPR:$rs1)), (FCVT_S_W $rs1, 0b111)>;
def : Pat<(any_uint_to_fp (i32 GPR:$rs1)), (FCVT_S_WU $rs1, 0b111)>;
} // Predicates = [HasStdExtF, IsRV32]

let Predicates = [HasStdExtF, IsRV64] in {
// Moves (no conversion)
def : Pat<(riscv_fmv_w_x_rv64 GPR:$src), (FMV_W_X GPR:$src)>;
def : Pat<(riscv_fmv_x_anyextw_rv64 FPR32:$src), (FMV_X_W FPR32:$src)>;
def : Pat<(sext_inreg (riscv_fmv_x_anyextw_rv64 FPR32:$src), i32),
          (FMV_X_W FPR32:$src)>;

// Use target specific isd nodes to help us remember the result is sign
// extended. Matching sext_inreg+fptoui/fptosi may cause the conversion to be
// duplicated if it has another user that didn't need the sign_extend.
def : Pat<(riscv_any_fcvt_w_rv64 FPR32:$rs1, timm:$frm),  (FCVT_W_S $rs1, timm:$frm)>;
def : Pat<(riscv_any_fcvt_wu_rv64 FPR32:$rs1, timm:$frm), (FCVT_WU_S $rs1, timm:$frm)>;

// float->[u]int64. Round-to-zero must be used.
def : Pat<(i64 (any_fp_to_sint FPR32:$rs1)), (FCVT_L_S $rs1, 0b001)>;
def : Pat<(i64 (any_fp_to_uint FPR32:$rs1)), (FCVT_LU_S $rs1, 0b001)>;

// Saturating float->[u]int64.
def : Pat<(i64 (riscv_fcvt_x FPR32:$rs1, timm:$frm)), (FCVT_L_S $rs1, timm:$frm)>;
def : Pat<(i64 (riscv_fcvt_xu FPR32:$rs1, timm:$frm)), (FCVT_LU_S $rs1, timm:$frm)>;

// float->int64 with current rounding mode.
def : Pat<(i64 (any_lrint FPR32:$rs1)), (FCVT_L_S $rs1, 0b111)>;
def : Pat<(i64 (any_llrint FPR32:$rs1)), (FCVT_L_S $rs1, 0b111)>;

// float->int64 rounded to neartest with ties rounded away from zero.
def : Pat<(i64 (any_lround FPR32:$rs1)), (FCVT_L_S $rs1, 0b100)>;
def : Pat<(i64 (any_llround FPR32:$rs1)), (FCVT_L_S $rs1, 0b100)>;

// [u]int->fp. Match GCC and default to using dynamic rounding mode.
def : Pat<(any_sint_to_fp (i64 (sexti32 (i64 GPR:$rs1)))), (FCVT_S_W $rs1, 0b111)>;
def : Pat<(any_uint_to_fp (i64 (zexti32 (i64 GPR:$rs1)))), (FCVT_S_WU $rs1, 0b111)>;
def : Pat<(any_sint_to_fp (i64 GPR:$rs1)), (FCVT_S_L $rs1, 0b111)>;
def : Pat<(any_uint_to_fp (i64 GPR:$rs1)), (FCVT_S_LU $rs1, 0b111)>;
} // Predicates = [HasStdExtF, IsRV64]

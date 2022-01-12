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
    UseNamedOperandTable = 1, hasPostISelHook = 1 in
class FPFMA_rrr_frm<RISCVOpcode opcode, bits<2> funct2, string opcodestr,
                    RegisterClass rty>
    : RVInstR4Frm<funct2, opcode, (outs rty:$rd),
                  (ins rty:$rs1, rty:$rs2, rty:$rs3, frmarg:$frm),
                  opcodestr, "$rd, $rs1, $rs2, $rs3, $frm">;

class FPFMADynFrmAlias<FPFMA_rrr_frm Inst, string OpcodeStr,
                       RegisterClass rty>
    : InstAlias<OpcodeStr#" $rd, $rs1, $rs2, $rs3",
                (Inst rty:$rd, rty:$rs1, rty:$rs2, rty:$rs3, 0b111)>;

let hasSideEffects = 0, mayLoad = 0, mayStore = 0, mayRaiseFPException = 1 in
class FPALU_rr<bits<7> funct7, bits<3> funct3, string opcodestr,
               RegisterClass rty>
    : RVInstR<funct7, funct3, OPC_OP_FP, (outs rty:$rd),
              (ins rty:$rs1, rty:$rs2), opcodestr, "$rd, $rs1, $rs2">;

let hasSideEffects = 0, mayLoad = 0, mayStore = 0, mayRaiseFPException = 1,
    UseNamedOperandTable = 1, hasPostISelHook = 1 in
class FPALU_rr_frm<bits<7> funct7, string opcodestr, RegisterClass rty>
    : RVInstRFrm<funct7, OPC_OP_FP, (outs rty:$rd),
                 (ins rty:$rs1, rty:$rs2, frmarg:$frm), opcodestr,
                  "$rd, $rs1, $rs2, $frm">;

class FPALUDynFrmAlias<FPALU_rr_frm Inst, string OpcodeStr,
                       RegisterClass rty>
    : InstAlias<OpcodeStr#" $rd, $rs1, $rs2",
                (Inst rty:$rd, rty:$rs1, rty:$rs2, 0b111)>;

let hasSideEffects = 0, mayLoad = 0, mayStore = 0, mayRaiseFPException = 1 in
class FPUnaryOp_r<bits<7> funct7, bits<5> rs2val, bits<3> funct3,
                  RegisterClass rdty, RegisterClass rs1ty, string opcodestr>
    : RVInstR<funct7, funct3, OPC_OP_FP, (outs rdty:$rd), (ins rs1ty:$rs1),
              opcodestr, "$rd, $rs1"> {
  let rs2 = rs2val;
}

let hasSideEffects = 0, mayLoad = 0, mayStore = 0, mayRaiseFPException = 1,
    UseNamedOperandTable = 1, hasPostISelHook = 1 in
class FPUnaryOp_r_frm<bits<7> funct7, bits<5> rs2val, RegisterClass rdty,
                      RegisterClass rs1ty, string opcodestr>
    : RVInstRFrm<funct7, OPC_OP_FP, (outs rdty:$rd),
                 (ins rs1ty:$rs1, frmarg:$frm), opcodestr,
                  "$rd, $rs1, $frm"> {
  let rs2 = rs2val;
}

class FPUnaryOpDynFrmAlias<FPUnaryOp_r_frm Inst, string OpcodeStr,
                           RegisterClass rdty, RegisterClass rs1ty>
    : InstAlias<OpcodeStr#" $rd, $rs1",
                (Inst rdty:$rd, rs1ty:$rs1, 0b111)>;

let hasSideEffects = 0, mayLoad = 0, mayStore = 0, mayRaiseFPException = 1 in
class FPCmp_rr<bits<7> funct7, bits<3> funct3, string opcodestr,
               RegisterClass rty>
    : RVInstR<funct7, funct3, OPC_OP_FP, (outs GPR:$rd),
              (ins rty:$rs1, rty:$rs2), opcodestr, "$rd, $rs1, $rs2">;

//===----------------------------------------------------------------------===//
// Instructions
//===----------------------------------------------------------------------===//

let Predicates = [HasStdExtF] in {
def FLW : FPLoad_r<0b010, "flw", FPR32, WriteFLD32>;

// Operands for stores are in the order srcreg, base, offset rather than
// reflecting the order these fields are specified in the instruction
// encoding.
def FSW : FPStore_r<0b010, "fsw", FPR32, WriteFST32>;

let SchedRW = [WriteFMA32, ReadFMA32, ReadFMA32, ReadFMA32] in {
def FMADD_S  : FPFMA_rrr_frm<OPC_MADD,  0b00, "fmadd.s",  FPR32>;
def FMSUB_S  : FPFMA_rrr_frm<OPC_MSUB,  0b00, "fmsub.s",  FPR32>;
def FNMSUB_S : FPFMA_rrr_frm<OPC_NMSUB, 0b00, "fnmsub.s", FPR32>;
def FNMADD_S : FPFMA_rrr_frm<OPC_NMADD, 0b00, "fnmadd.s", FPR32>;
}

def : FPFMADynFrmAlias<FMADD_S,  "fmadd.s",  FPR32>;
def : FPFMADynFrmAlias<FMSUB_S,  "fmsub.s",  FPR32>;
def : FPFMADynFrmAlias<FNMSUB_S, "fnmsub.s", FPR32>;
def : FPFMADynFrmAlias<FNMADD_S, "fnmadd.s", FPR32>;

def FADD_S : FPALU_rr_frm<0b0000000, "fadd.s", FPR32>,
             Sched<[WriteFALU32, ReadFALU32, ReadFALU32]>;
def FSUB_S : FPALU_rr_frm<0b0000100, "fsub.s", FPR32>,
             Sched<[WriteFALU32, ReadFALU32, ReadFALU32]>;
def FMUL_S : FPALU_rr_frm<0b0001000, "fmul.s", FPR32>,
             Sched<[WriteFMul32, ReadFMul32, ReadFMul32]>;
def FDIV_S : FPALU_rr_frm<0b0001100, "fdiv.s", FPR32>,
             Sched<[WriteFDiv32, ReadFDiv32, ReadFDiv32]>;

def        : FPALUDynFrmAlias<FADD_S, "fadd.s", FPR32>;
def        : FPALUDynFrmAlias<FSUB_S, "fsub.s", FPR32>;
def        : FPALUDynFrmAlias<FMUL_S, "fmul.s", FPR32>;
def        : FPALUDynFrmAlias<FDIV_S, "fdiv.s", FPR32>;

def FSQRT_S : FPUnaryOp_r_frm<0b0101100, 0b00000, FPR32, FPR32, "fsqrt.s">,
              Sched<[WriteFSqrt32, ReadFSqrt32]>;
def         : FPUnaryOpDynFrmAlias<FSQRT_S, "fsqrt.s", FPR32, FPR32>;

let SchedRW = [WriteFSGNJ32, ReadFSGNJ32, ReadFSGNJ32],
    mayRaiseFPException = 0 in {
def FSGNJ_S  : FPALU_rr<0b0010000, 0b000, "fsgnj.s", FPR32>;
def FSGNJN_S : FPALU_rr<0b0010000, 0b001, "fsgnjn.s", FPR32>;
def FSGNJX_S : FPALU_rr<0b0010000, 0b010, "fsgnjx.s", FPR32>;
}

let SchedRW = [WriteFMinMax32, ReadFMinMax32, ReadFMinMax32] in {
def FMIN_S   : FPALU_rr<0b0010100, 0b000, "fmin.s", FPR32>;
def FMAX_S   : FPALU_rr<0b0010100, 0b001, "fmax.s", FPR32>;
}

def FCVT_W_S : FPUnaryOp_r_frm<0b1100000, 0b00000, GPR, FPR32, "fcvt.w.s">,
               Sched<[WriteFCvtF32ToI32, ReadFCvtF32ToI32]>;
def          : FPUnaryOpDynFrmAlias<FCVT_W_S, "fcvt.w.s", GPR, FPR32>;

def FCVT_WU_S : FPUnaryOp_r_frm<0b1100000, 0b00001, GPR, FPR32, "fcvt.wu.s">,
                Sched<[WriteFCvtF32ToI32, ReadFCvtF32ToI32]>;
def           : FPUnaryOpDynFrmAlias<FCVT_WU_S, "fcvt.wu.s", GPR, FPR32>;

let mayRaiseFPException = 0 in
def FMV_X_W : FPUnaryOp_r<0b1110000, 0b00000, 0b000, GPR, FPR32, "fmv.x.w">,
              Sched<[WriteFMovF32ToI32, ReadFMovF32ToI32]>;

let SchedRW = [WriteFCmp32, ReadFCmp32, ReadFCmp32] in {
def FEQ_S : FPCmp_rr<0b1010000, 0b010, "feq.s", FPR32>;
def FLT_S : FPCmp_rr<0b1010000, 0b001, "flt.s", FPR32>;
def FLE_S : FPCmp_rr<0b1010000, 0b000, "fle.s", FPR32>;
}

let mayRaiseFPException = 0 in
def FCLASS_S : FPUnaryOp_r<0b1110000, 0b00000, 0b001, GPR, FPR32, "fclass.s">,
               Sched<[WriteFClass32, ReadFClass32]>;

def FCVT_S_W : FPUnaryOp_r_frm<0b1101000, 0b00000, FPR32, GPR, "fcvt.s.w">,
               Sched<[WriteFCvtI32ToF32, ReadFCvtI32ToF32]>;
def          : FPUnaryOpDynFrmAlias<FCVT_S_W, "fcvt.s.w", FPR32, GPR>;

def FCVT_S_WU : FPUnaryOp_r_frm<0b1101000, 0b00001, FPR32, GPR, "fcvt.s.wu">,
                Sched<[WriteFCvtI32ToF32, ReadFCvtI32ToF32]>;
def           : FPUnaryOpDynFrmAlias<FCVT_S_WU, "fcvt.s.wu", FPR32, GPR>;

let mayRaiseFPException = 0 in
def FMV_W_X : FPUnaryOp_r<0b1111000, 0b00000, 0b000, FPR32, GPR, "fmv.w.x">,
              Sched<[WriteFMovI32ToF32, ReadFMovI32ToF32]>;
} // Predicates = [HasStdExtF]

let Predicates = [HasStdExtF, IsRV64] in {
def FCVT_L_S  : FPUnaryOp_r_frm<0b1100000, 0b00010, GPR, FPR32, "fcvt.l.s">,
                Sched<[WriteFCvtF32ToI64, ReadFCvtF32ToI64]>;
def           : FPUnaryOpDynFrmAlias<FCVT_L_S, "fcvt.l.s", GPR, FPR32>;

def FCVT_LU_S  : FPUnaryOp_r_frm<0b1100000, 0b00011, GPR, FPR32, "fcvt.lu.s">,
                 Sched<[WriteFCvtF32ToI64, ReadFCvtF32ToI64]>;
def            : FPUnaryOpDynFrmAlias<FCVT_LU_S, "fcvt.lu.s", GPR, FPR32>;

def FCVT_S_L : FPUnaryOp_r_frm<0b1101000, 0b00010, FPR32, GPR, "fcvt.s.l">,
               Sched<[WriteFCvtI64ToF32, ReadFCvtI64ToF32]>;
def          : FPUnaryOpDynFrmAlias<FCVT_S_L, "fcvt.s.l", FPR32, GPR>;

def FCVT_S_LU : FPUnaryOp_r_frm<0b1101000, 0b00011, FPR32, GPR, "fcvt.s.lu">,
                Sched<[WriteFCvtI64ToF32, ReadFCvtI64ToF32]>;
def           : FPUnaryOpDynFrmAlias<FCVT_S_LU, "fcvt.s.lu", FPR32, GPR>;
} // Predicates = [HasStdExtF, IsRV64]

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

//===----------------------------------------------------------------------===//
// Pseudo-instructions and codegen patterns
//===----------------------------------------------------------------------===//

/// Floating point constants
def fpimm0 : PatLeaf<(fpimm), [{ return N->isExactlyValue(+0.0); }]>;

/// Generic pattern classes
class PatSetCC<RegisterClass Ty, SDPatternOperator OpNode, CondCode Cond, RVInst Inst>
    : Pat<(OpNode Ty:$rs1, Ty:$rs2, Cond), (Inst $rs1, $rs2)>;

class PatFpr32Fpr32<SDPatternOperator OpNode, RVInstR Inst>
    : Pat<(OpNode FPR32:$rs1, FPR32:$rs2), (Inst $rs1, $rs2)>;

class PatFpr32Fpr32DynFrm<SDPatternOperator OpNode, RVInstRFrm Inst>
    : Pat<(OpNode FPR32:$rs1, FPR32:$rs2), (Inst $rs1, $rs2, 0b111)>;

let Predicates = [HasStdExtF] in {

/// Float constants
def : Pat<(f32 (fpimm0)), (FMV_W_X X0)>;

/// Float conversion operations

// [u]int32<->float conversion patterns must be gated on IsRV32 or IsRV64, so
// are defined later.

/// Float arithmetic operations

def : PatFpr32Fpr32DynFrm<any_fadd, FADD_S>;
def : PatFpr32Fpr32DynFrm<any_fsub, FSUB_S>;
def : PatFpr32Fpr32DynFrm<any_fmul, FMUL_S>;
def : PatFpr32Fpr32DynFrm<any_fdiv, FDIV_S>;

def : Pat<(any_fsqrt FPR32:$rs1), (FSQRT_S FPR32:$rs1, 0b111)>;

def : Pat<(fneg FPR32:$rs1), (FSGNJN_S $rs1, $rs1)>;
def : Pat<(fabs FPR32:$rs1), (FSGNJX_S $rs1, $rs1)>;

def : PatFpr32Fpr32<fcopysign, FSGNJ_S>;
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
def : PatFpr32Fpr32<fminnum, FMIN_S>;
def : PatFpr32Fpr32<fmaxnum, FMAX_S>;

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

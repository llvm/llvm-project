//===-- RISCVInstrInfoF.td - RISC-V 'F' instructions -------*- tablegen -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file describes the RISC-V instructions from the standard 'F',
// Single-Precision Floating-Point instruction set extension.
//
//===----------------------------------------------------------------------===//

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
  let DecoderMethod = "decodeUImmOperand<3>";
}

//===----------------------------------------------------------------------===//
// Instruction class templates
//===----------------------------------------------------------------------===//

let hasSideEffects = 0, mayLoad = 0, mayStore = 0 in
class FPFMAS_rrr_frm<RISCVOpcode opcode, string opcodestr>
    : RVInstR4<0b00, opcode, (outs FPR32:$rd),
               (ins FPR32:$rs1, FPR32:$rs2, FPR32:$rs3, frmarg:$funct3),
                opcodestr, "$rd, $rs1, $rs2, $rs3, $funct3">;

class FPFMASDynFrmAlias<FPFMAS_rrr_frm Inst, string OpcodeStr>
    : InstAlias<OpcodeStr#" $rd, $rs1, $rs2, $rs3",
                (Inst FPR32:$rd, FPR32:$rs1, FPR32:$rs2, FPR32:$rs3, 0b111)>;

let hasSideEffects = 0, mayLoad = 0, mayStore = 0 in
class FPALUS_rr<bits<7> funct7, bits<3> funct3, string opcodestr>
    : RVInstR<funct7, funct3, OPC_OP_FP, (outs FPR32:$rd),
              (ins FPR32:$rs1, FPR32:$rs2), opcodestr, "$rd, $rs1, $rs2">;

let hasSideEffects = 0, mayLoad = 0, mayStore = 0 in
class FPALUS_rr_frm<bits<7> funct7, string opcodestr>
    : RVInstRFrm<funct7, OPC_OP_FP, (outs FPR32:$rd),
                 (ins FPR32:$rs1, FPR32:$rs2, frmarg:$funct3), opcodestr,
                  "$rd, $rs1, $rs2, $funct3">;

class FPALUSDynFrmAlias<FPALUS_rr_frm Inst, string OpcodeStr>
    : InstAlias<OpcodeStr#" $rd, $rs1, $rs2",
                (Inst FPR32:$rd, FPR32:$rs1, FPR32:$rs2, 0b111)>;

let hasSideEffects = 0, mayLoad = 0, mayStore = 0 in
class FPUnaryOp_r<bits<7> funct7, bits<3> funct3, RegisterClass rdty,
                RegisterClass rs1ty, string opcodestr>
    : RVInstR<funct7, funct3, OPC_OP_FP, (outs rdty:$rd), (ins rs1ty:$rs1),
              opcodestr, "$rd, $rs1">;

let hasSideEffects = 0, mayLoad = 0, mayStore = 0 in
class FPUnaryOp_r_frm<bits<7> funct7, RegisterClass rdty, RegisterClass rs1ty,
                      string opcodestr>
    : RVInstRFrm<funct7, OPC_OP_FP, (outs rdty:$rd),
                 (ins rs1ty:$rs1, frmarg:$funct3), opcodestr,
                  "$rd, $rs1, $funct3">;

class FPUnaryOpDynFrmAlias<FPUnaryOp_r_frm Inst, string OpcodeStr,
                           RegisterClass rdty, RegisterClass rs1ty>
    : InstAlias<OpcodeStr#" $rd, $rs1",
                (Inst rdty:$rd, rs1ty:$rs1, 0b111)>;

let hasSideEffects = 0, mayLoad = 0, mayStore = 0 in
class FPCmpS_rr<bits<3> funct3, string opcodestr>
    : RVInstR<0b1010000, funct3, OPC_OP_FP, (outs GPR:$rd),
              (ins FPR32:$rs1, FPR32:$rs2), opcodestr, "$rd, $rs1, $rs2">;

//===----------------------------------------------------------------------===//
// Instructions
//===----------------------------------------------------------------------===//

let Predicates = [HasStdExtF] in {
let hasSideEffects = 0, mayLoad = 1, mayStore = 0 in
def FLW : RVInstI<0b010, OPC_LOAD_FP, (outs FPR32:$rd),
                  (ins GPR:$rs1, simm12:$imm12),
                   "flw", "$rd, ${imm12}(${rs1})">;

// Operands for stores are in the order srcreg, base, offset rather than
// reflecting the order these fields are specified in the instruction
// encoding.
let hasSideEffects = 0, mayLoad = 0, mayStore = 1 in
def FSW : RVInstS<0b010, OPC_STORE_FP, (outs),
                  (ins FPR32:$rs2, GPR:$rs1, simm12:$imm12),
                   "fsw", "$rs2, ${imm12}(${rs1})">;

def FMADD_S  : FPFMAS_rrr_frm<OPC_MADD, "fmadd.s">;
def          : FPFMASDynFrmAlias<FMADD_S, "fmadd.s">;
def FMSUB_S  : FPFMAS_rrr_frm<OPC_MSUB, "fmsub.s">;
def          : FPFMASDynFrmAlias<FMSUB_S, "fmsub.s">;
def FNMSUB_S : FPFMAS_rrr_frm<OPC_NMSUB, "fnmsub.s">;
def          : FPFMASDynFrmAlias<FNMSUB_S, "fnmsub.s">;
def FNMADD_S : FPFMAS_rrr_frm<OPC_NMADD, "fnmadd.s">;
def          : FPFMASDynFrmAlias<FNMADD_S, "fnmadd.s">;

def FADD_S : FPALUS_rr_frm<0b0000000, "fadd.s">;
def        : FPALUSDynFrmAlias<FADD_S, "fadd.s">;
def FSUB_S : FPALUS_rr_frm<0b0000100, "fsub.s">;
def        : FPALUSDynFrmAlias<FSUB_S, "fsub.s">;
def FMUL_S : FPALUS_rr_frm<0b0001000, "fmul.s">;
def        : FPALUSDynFrmAlias<FMUL_S, "fmul.s">;
def FDIV_S : FPALUS_rr_frm<0b0001100, "fdiv.s">;
def        : FPALUSDynFrmAlias<FDIV_S, "fdiv.s">;

def FSQRT_S : FPUnaryOp_r_frm<0b0101100, FPR32, FPR32, "fsqrt.s"> {
  let rs2 = 0b00000;
}
def         : FPUnaryOpDynFrmAlias<FSQRT_S, "fsqrt.s", FPR32, FPR32>;

def FSGNJ_S  : FPALUS_rr<0b0010000, 0b000, "fsgnj.s">;
def FSGNJN_S : FPALUS_rr<0b0010000, 0b001, "fsgnjn.s">;
def FSGNJX_S : FPALUS_rr<0b0010000, 0b010, "fsgnjx.s">;
def FMIN_S   : FPALUS_rr<0b0010100, 0b000, "fmin.s">;
def FMAX_S   : FPALUS_rr<0b0010100, 0b001, "fmax.s">;

def FCVT_W_S : FPUnaryOp_r_frm<0b1100000, GPR, FPR32, "fcvt.w.s"> {
  let rs2 = 0b00000;
}
def          : FPUnaryOpDynFrmAlias<FCVT_W_S, "fcvt.w.s", GPR, FPR32>;

def FCVT_WU_S : FPUnaryOp_r_frm<0b1100000, GPR, FPR32, "fcvt.wu.s"> {
  let rs2 = 0b00001;
}
def           : FPUnaryOpDynFrmAlias<FCVT_WU_S, "fcvt.wu.s", GPR, FPR32>;

def FMV_X_W : FPUnaryOp_r<0b1110000, 0b000, GPR, FPR32, "fmv.x.w"> {
  let rs2 = 0b00000;
}

def FEQ_S : FPCmpS_rr<0b010, "feq.s">;
def FLT_S : FPCmpS_rr<0b001, "flt.s">;
def FLE_S : FPCmpS_rr<0b000, "fle.s">;

def FCLASS_S : FPUnaryOp_r<0b1110000, 0b001, GPR, FPR32, "fclass.s"> {
  let rs2 = 0b00000;
}

def FCVT_S_W : FPUnaryOp_r_frm<0b1101000, FPR32, GPR, "fcvt.s.w"> {
  let rs2 = 0b00000;
}
def          : FPUnaryOpDynFrmAlias<FCVT_S_W, "fcvt.s.w", FPR32, GPR>;

def FCVT_S_WU : FPUnaryOp_r_frm<0b1101000, FPR32, GPR, "fcvt.s.wu"> {
  let rs2 = 0b00001;
}
def           : FPUnaryOpDynFrmAlias<FCVT_S_WU, "fcvt.s.wu", FPR32, GPR>;

def FMV_W_X : FPUnaryOp_r<0b1111000, 0b000, FPR32, GPR, "fmv.w.x"> {
  let rs2 = 0b00000;
}
} // Predicates = [HasStdExtF]

let Predicates = [HasStdExtF, IsRV64] in {
def FCVT_L_S  : FPUnaryOp_r_frm<0b1100000, GPR, FPR32, "fcvt.l.s"> {
  let rs2 = 0b00010;
}
def           : FPUnaryOpDynFrmAlias<FCVT_L_S, "fcvt.l.s", GPR, FPR32>;

def FCVT_LU_S  : FPUnaryOp_r_frm<0b1100000, GPR, FPR32, "fcvt.lu.s"> {
  let rs2 = 0b00011;
}
def            : FPUnaryOpDynFrmAlias<FCVT_LU_S, "fcvt.lu.s", GPR, FPR32>;

def FCVT_S_L : FPUnaryOp_r_frm<0b1101000, FPR32, GPR, "fcvt.s.l"> {
  let rs2 = 0b00010;
}
def          : FPUnaryOpDynFrmAlias<FCVT_S_L, "fcvt.s.l", FPR32, GPR>;

def FCVT_S_LU : FPUnaryOp_r_frm<0b1101000, FPR32, GPR, "fcvt.s.lu"> {
  let rs2 = 0b00011;
}
def           : FPUnaryOpDynFrmAlias<FCVT_S_LU, "fcvt.s.lu", FPR32, GPR>;
} // Predicates = [HasStdExtF, IsRV64]

//===-- RISCVInstrInfoD.td - RISC-V 'D' instructions -------*- tablegen -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file describes the RISC-V instructions from the standard 'D',
// Double-Precision Floating-Point instruction set extension.
//
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// Instruction Class Templates
//===----------------------------------------------------------------------===//

let hasSideEffects = 0, mayLoad = 0, mayStore = 0 in
class FPFMAD_rrr_frm<RISCVOpcode opcode, string opcodestr>
    : RVInstR4<0b01, opcode, (outs FPR64:$rd),
               (ins FPR64:$rs1, FPR64:$rs2, FPR64:$rs3, frmarg:$funct3),
                opcodestr, "$rd, $rs1, $rs2, $rs3, $funct3">;

class FPFMADDynFrmAlias<FPFMAD_rrr_frm Inst, string OpcodeStr>
    : InstAlias<OpcodeStr#" $rd, $rs1, $rs2, $rs3",
                (Inst FPR64:$rd, FPR64:$rs1, FPR64:$rs2, FPR64:$rs3, 0b111)>;

let hasSideEffects = 0, mayLoad = 0, mayStore = 0 in
class FPALUD_rr<bits<7> funct7, bits<3> funct3, string opcodestr>
    : RVInstR<funct7, funct3, OPC_OP_FP, (outs FPR64:$rd),
              (ins FPR64:$rs1, FPR64:$rs2), opcodestr, "$rd, $rs1, $rs2">;

let hasSideEffects = 0, mayLoad = 0, mayStore = 0 in
class FPALUD_rr_frm<bits<7> funct7, string opcodestr>
    : RVInstRFrm<funct7, OPC_OP_FP, (outs FPR64:$rd),
                (ins FPR64:$rs1, FPR64:$rs2, frmarg:$funct3), opcodestr,
                 "$rd, $rs1, $rs2, $funct3">;

class FPALUDDynFrmAlias<FPALUD_rr_frm Inst, string OpcodeStr>
    : InstAlias<OpcodeStr#" $rd, $rs1, $rs2",
                (Inst FPR64:$rd, FPR64:$rs1, FPR64:$rs2, 0b111)>;

let hasSideEffects = 0, mayLoad = 0, mayStore = 0 in
class FPCmpD_rr<bits<3> funct3, string opcodestr>
    : RVInstR<0b1010001, funct3, OPC_OP_FP, (outs GPR:$rd),
              (ins FPR64:$rs1, FPR64:$rs2), opcodestr, "$rd, $rs1, $rs2">;

//===----------------------------------------------------------------------===//
// Instructions
//===----------------------------------------------------------------------===//

let Predicates = [HasStdExtD] in {

let hasSideEffects = 0, mayLoad = 1, mayStore = 0 in
def FLD : RVInstI<0b011, OPC_LOAD_FP, (outs FPR64:$rd),
                  (ins GPR:$rs1, simm12:$imm12),
                  "fld", "$rd, ${imm12}(${rs1})">;

// Operands for stores are in the order srcreg, base, offset rather than
// reflecting the order these fields are specified in the instruction
// encoding.
let hasSideEffects = 0, mayLoad = 0, mayStore = 1 in
def FSD : RVInstS<0b011, OPC_STORE_FP, (outs),
                  (ins FPR64:$rs2, GPR:$rs1, simm12:$imm12),
                   "fsd", "$rs2, ${imm12}(${rs1})">;

def FMADD_D  : FPFMAD_rrr_frm<OPC_MADD, "fmadd.d">;
def          : FPFMADDynFrmAlias<FMADD_D, "fmadd.d">;
def FMSUB_D  : FPFMAD_rrr_frm<OPC_MSUB, "fmsub.d">;
def          : FPFMADDynFrmAlias<FMSUB_D, "fmsub.d">;
def FNMSUB_D : FPFMAD_rrr_frm<OPC_NMSUB, "fnmsub.d">;
def          : FPFMADDynFrmAlias<FNMSUB_D, "fnmsub.d">;
def FNMADD_D : FPFMAD_rrr_frm<OPC_NMADD, "fnmadd.d">;
def          : FPFMADDynFrmAlias<FNMADD_D, "fnmadd.d">;

def FADD_D : FPALUD_rr_frm<0b0000001, "fadd.d">;
def        : FPALUDDynFrmAlias<FADD_D, "fadd.d">;
def FSUB_D : FPALUD_rr_frm<0b0000101, "fsub.d">;
def        : FPALUDDynFrmAlias<FSUB_D, "fsub.d">;
def FMUL_D : FPALUD_rr_frm<0b0001001, "fmul.d">;
def        : FPALUDDynFrmAlias<FMUL_D, "fmul.d">;
def FDIV_D : FPALUD_rr_frm<0b0001101, "fdiv.d">;
def        : FPALUDDynFrmAlias<FDIV_D, "fdiv.d">;

def FSQRT_D : FPUnaryOp_r_frm<0b0101101, FPR64, FPR64, "fsqrt.d"> {
  let rs2 = 0b00000;
}
def         : FPUnaryOpDynFrmAlias<FSQRT_D, "fsqrt.d", FPR64, FPR64>;

def FSGNJ_D  : FPALUD_rr<0b0010001, 0b000, "fsgnj.d">;
def FSGNJN_D : FPALUD_rr<0b0010001, 0b001, "fsgnjn.d">;
def FSGNJX_D : FPALUD_rr<0b0010001, 0b010, "fsgnjx.d">;
def FMIN_D   : FPALUD_rr<0b0010101, 0b000, "fmin.d">;
def FMAX_D   : FPALUD_rr<0b0010101, 0b001, "fmax.d">;

def FCVT_S_D : FPUnaryOp_r_frm<0b0100000, FPR32, FPR64, "fcvt.s.d"> {
  let rs2 = 0b00001;
}
def          : FPUnaryOpDynFrmAlias<FCVT_S_D, "fcvt.s.d", FPR32, FPR64>;

def FCVT_D_S : FPUnaryOp_r<0b0100001, 0b000, FPR64, FPR32, "fcvt.d.s"> {
  let rs2 = 0b00000;
}

def FEQ_D : FPCmpD_rr<0b010, "feq.d">;
def FLT_D : FPCmpD_rr<0b001, "flt.d">;
def FLE_D : FPCmpD_rr<0b000, "fle.d">;

def FCLASS_D : FPUnaryOp_r<0b1110001, 0b001, GPR, FPR64, "fclass.d"> {
  let rs2 = 0b00000;
}

def FCVT_W_D : FPUnaryOp_r_frm<0b1100001, GPR, FPR64, "fcvt.w.d"> {
  let rs2 = 0b00000;
}
def          : FPUnaryOpDynFrmAlias<FCVT_W_D, "fcvt.w.d", GPR, FPR64>;

def FCVT_WU_D : FPUnaryOp_r_frm<0b1100001, GPR, FPR64, "fcvt.wu.d"> {
  let rs2 = 0b00001;
}
def           : FPUnaryOpDynFrmAlias<FCVT_WU_D, "fcvt.wu.d", GPR, FPR64>;

def FCVT_D_W : FPUnaryOp_r<0b1101001, 0b000, FPR64, GPR, "fcvt.d.w"> {
  let rs2 = 0b00000;
}

def FCVT_D_WU : FPUnaryOp_r<0b1101001, 0b000, FPR64, GPR, "fcvt.d.wu"> {
  let rs2 = 0b00001;
}
} // Predicates = [HasStdExtD]

let Predicates = [HasStdExtD, IsRV64] in {
def FCVT_L_D : FPUnaryOp_r_frm<0b1100001, GPR, FPR64, "fcvt.l.d"> {
  let rs2 = 0b00010;
}
def          : FPUnaryOpDynFrmAlias<FCVT_L_D, "fcvt.l.d", GPR, FPR64>;

def FCVT_LU_D : FPUnaryOp_r_frm<0b1100001, GPR, FPR64, "fcvt.lu.d"> {
  let rs2 = 0b00011;
}
def           : FPUnaryOpDynFrmAlias<FCVT_LU_D, "fcvt.lu.d", GPR, FPR64>;

def FMV_X_D : FPUnaryOp_r<0b1110001, 0b000, GPR, FPR64, "fmv.x.d"> {
  let rs2 = 0b00000;
}

def FCVT_D_L : FPUnaryOp_r_frm<0b1101001, FPR64, GPR, "fcvt.d.l"> {
  let rs2 = 0b00010;
}
def          : FPUnaryOpDynFrmAlias<FCVT_D_L, "fcvt.d.l", FPR64, GPR>;

def FCVT_D_LU : FPUnaryOp_r_frm<0b1101001, FPR64, GPR, "fcvt.d.lu"> {
  let rs2 = 0b00011;
}
def           : FPUnaryOpDynFrmAlias<FCVT_D_LU, "fcvt.d.lu", FPR64, GPR>;

def FMV_D_X : FPUnaryOp_r<0b1111001, 0b000, FPR64, GPR, "fmv.d.x"> {
  let rs2 = 0b00000;
}
} // Predicates = [HasStdExtD, IsRV64]

//===----------------------------------------------------------------------===//
// Assembler Pseudo Instructions (User-Level ISA, Version 2.2, Chapter 20)
//===----------------------------------------------------------------------===//

let Predicates = [HasStdExtD] in {
// TODO fld
// TODO fsd

def : InstAlias<"fmv.d $rd, $rs",  (FSGNJ_D  FPR64:$rd, FPR64:$rs, FPR64:$rs)>;
def : InstAlias<"fabs.d $rd, $rs", (FSGNJX_D FPR64:$rd, FPR64:$rs, FPR64:$rs)>;
def : InstAlias<"fneg.d $rd, $rs", (FSGNJN_D FPR64:$rd, FPR64:$rs, FPR64:$rs)>;
} // Predicates = [HasStdExtD]

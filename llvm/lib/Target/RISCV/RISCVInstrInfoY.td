//===-- RISCVInstrInfoY.td - RISCV instructions -------------*- tblgen-*---===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// Instruction Formats
//===----------------------------------------------------------------------===//

include "RISCVInstrFormatsY.td"

//===----------------------------------------------------------------------===//
// Instruction Class Templates
//===----------------------------------------------------------------------===//

let hasSideEffects = 0, mayLoad = 0, mayStore = 0 in {
class RVY_r<bits<5> funct5, string opcodestr, DAGOperand rdOp = GPR,
            DAGOperand rs1Op = YGPR>
    : RVYInstSrcDst<0x8, funct5, 0x0, OPC_OP, (outs rdOp:$rd),
                    (ins rs1Op:$rs1), opcodestr, "$rd, $rs1">;
class RVY_rr<bits<7> funct7, bits<3> funct3, string opcodestr,
             DAGOperand rdOp = YGPR, DAGOperand rs1Op = YGPR,
             DAGOperand rs2Op = GPR>
    : RVInstR<funct7, funct3, OPC_OP, (outs rdOp:$rd),
              (ins rs1Op:$rs1, rs2Op:$rs2), opcodestr, "$rd, $rs1, $rs2">;
class RVY_ri<bits<3> funct3, RISCVOpcode opcode, string opcodestr,
             DAGOperand rdOp = YGPR, DAGOperand rs1Op = YGPR>
    : RVInstI<funct3, opcode, (outs rdOp:$rd),
              (ins rs1Op:$rs1, simm12_lo:$imm12), opcodestr,
              "$rd, $rs1, $imm12">;
class RVY_bndswi<string opcodestr>
    : RVYInstBNDSWImmFmt<(outs YGPR:$rd), (ins YGPR:$rs1, ybndsw_imm:$imm),
                         opcodestr, "$rd, $rs1, $imm">;
} // hasSideEffects = 0, mayLoad = 0, mayStore = 0

let Predicates = [HasStdExtY] in {
//
// Instructions to Update The Capability Pointer
//
let isReMaterializable = true, isAsCheapAsAMove = true in {
  def ADDY : RVY_rr<0x6, 0x0, "addy", YGPR, YGPR, GPRNoX0>;
  def ADDIY : RVY_ri<0x2, OPC_OP_IMM_32, "addiy">;
}
def : InstAlias<"addy $rd, $rs1, $imm12",
                (ADDIY YGPR:$rd, YGPR:$rs1, simm12_lo:$imm12), 0>;
def YADDRW : RVY_rr<0x6, 0x1, "yaddrw", YGPR, YGPR, GPR>;

//
// Instructions to Manipulate Capabilities
//
def YPERMC : RVY_rr<0x6, 0x2, "ypermc", YGPR, YGPR, GPR>;
let hasSideEffects = 0, mayLoad = 0, mayStore = 0, isReMaterializable = true,
    isAsCheapAsAMove = true, isMoveReg = true in {
  def YMV : RVYInstSrcDst<0x6, 0x0, 0x0, OPC_OP, (outs YGPR:$rd),
                          (ins YGPR:$rs1), "ymv", "$rd, $rs1">;
}
def PACKY : RVY_rr<0x6, 0x3, "packy", YGPR, YGPR, GPR>;
def : MnemonicAlias<"yhiw", "packy">;
def YBNDSW : RVY_rr<0x7, 0x0, "ybndsw", YGPR, YGPR, GPR>;
def YBNDSWI : RVY_bndswi<"ybndswi">;
def : InstAlias<"ybndsw $rd, $rs1, $imm",
                (YBNDSWI YGPR:$rd, YGPR:$rs1, ybndsw_imm:$imm), 0>;
def YBNDSRW : RVY_rr<0x7, 0x1, "ybndsrw", YGPR, YGPR, GPR>;
def YSUNSEAL : RVY_rr<0x7, 0x2, "ysunseal", YGPR, YGPR, YGPR>;

//
// Instructions to Decode Capability Bounds
//
def YBASER : RVY_r<0x5, "ybaser", GPR, YGPR>;
def YLENR : RVY_r<0x6, "ylenr", GPR, YGPR>;

//
// Instructions to Extract Capability Fields
//
def YTAGR : RVY_r<0x0, "ytagr", GPR, YGPR>;
def YPERMR : RVY_r<0x1, "ypermr", GPR, YGPR>;
def YTYPER : RVY_r<0x2, "ytyper", GPR, YGPR>;
// The SRLIY instruction uses the encoding of SRLI with shamt==XLEN, other
// values are not supported. YHIR is a pseudo that expands to the correct
// shift value depending on RV32/RV64.
let hasSideEffects = 0, mayLoad = 0, mayStore = 0 in {
def SRLIY : RVInstIBase<0b101, OPC_OP_IMM, (outs GPR:$rd),
                        (ins YGPR:$rs1, uimm7_srliy:$shamt), "srliy",
                        "$rd, $rs1, $shamt"> {
  bits<7> shamt;
  let Inst{31-27} = 0b00000;
  let Inst{26-20} = shamt;
}
}
let Predicates = [HasStdExtY, IsRV64] in {
def : InstAlias<"yhir $rd, $rs1", (SRLIY GPR:$rd, YGPR:$rs1, 64)>;
}
let Predicates = [HasStdExtY, IsRV32] in {
def : InstAlias<"yhir $rd, $rs1", (SRLIY GPR:$rd, YGPR:$rs1, 32)>;
}
//
// Miscellaneous Instructions to Handle Capability Data
//
def SYEQ : RVY_rr<0x6, 0x4, "syeq", GPR, YGPR, YGPR>;
def YLT : RVY_rr<0x6, 0x6, "ylt", GPR, YGPR, YGPR>;
def YAMASK : RVY_r<0x7, "yamask", GPR, GPR>;
} // Predicates = [HasStdExtY]

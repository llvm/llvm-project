//===-- RISCVInstrInfoY.td - RISCV instructions -------------*- tblgen-*---===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file describes the RISC-V instructions from the standard 'Base Y'
// instruction set.
//
// This version is still experimental as the 'Y' extension hasn't been
// ratified yet.
//
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// Operand definitions.
//===----------------------------------------------------------------------===//

def YBNDSWImmOperand : AsmOperandClass {
  let Name = "YBNDSWImm";
  let RenderMethod = "addImmOperands";
  let DiagnosticType = "InvalidYBNDSWImm";
}

def ybndsw_imm : RISCVOp<XLenVT>, ImmLeaf<XLenVT, [{
  return llvm::RISCV::isValidYBNDSWImm(Imm);
}]> {
  let EncoderMethod = "getYBNDSWImmOpValue";
  let ParserMatchClass = YBNDSWImmOperand;
  let DecoderMethod = "DecodeYBNDSWImm";
  let OperandType = "OPERAND_UIMM9_YBNDSWI";
}

//===----------------------------------------------------------------------===//
// Instruction Formats and class templates
//===----------------------------------------------------------------------===//

let hasSideEffects = 0, mayLoad = 0, mayStore = 0 in {
// All regular 3-operand RVY instructions use opc=CUSTOM-3/RVY-A with funct3=0
defvar OPC_RVY_A = OPC_CUSTOM_3;
class RVY_rr<bits<7> funct7, string opcodestr, DAGOperand rdOp,
             DAGOperand rs1Op, DAGOperand rs2Op>
    : RVInstR<funct7, 0b000, OPC_RVY_A, (outs rdOp:$rd),
              (ins rs1Op:$rs1, rs2Op:$rs2), opcodestr, "$rd, $rs1, $rs2">;

// Like RVY_rr, except rs1 is fixed (e.g. x0 for YSEALE).
class RVY_rr_fixed_rs1<bits<7> funct7, bits<5> rs1val, string opcodestr,
                       DAGOperand rdOp, DAGOperand rs2Op>
    : RVInstR<funct7, 0b000, OPC_RVY_A, (outs rdOp:$rd), (ins rs2Op:$rs2),
              opcodestr, "$rd, $rs2"> {
  let rs1 = rs1val;
}

// Like RVY_rr, except rs2 is now an additional function code.
class RVYInst2Op<bits<7> funct7, bits<5> funct5, DAGOperand rdOp,
                 DAGOperand rs1Op, string opcodestr, string argstr>
    : RVInst<(outs rdOp:$rd), (ins rs1Op:$rs1), opcodestr, argstr, [],
             InstFormatR> {
  bits<5> rs1;
  bits<5> rd;

  let Inst{31-25} = funct7;
  let Inst{24-20} = funct5;
  let Inst{19-15} = rs1;
  let Inst{14-12} = 0b000; // funct3
  let Inst{11-7} = rd;
  let Inst{6-0} = OPC_RVY_A.Value;
}

class RVY_r_xx<bits<5> funct5, string opcodestr>
    : RVYInst2Op<0b1111000, funct5, GPR, GPR, opcodestr, "$rd, $rs1">;
class RVY_r_xy<bits<5> funct5, string opcodestr>
    : RVYInst2Op<0b1111010, funct5, GPR, YGPR, opcodestr, "$rd, $rs1">;
class RVY_ri<bits<3> funct3, string opcodestr, DAGOperand rdOp = YGPR,
             DAGOperand rs1Op = YGPR>
    : RVInstI<funct3, OPC_RVY_A, (outs rdOp:$rd),
              (ins rs1Op:$rs1, simm12_lo:$imm12), opcodestr,
              "$rd, $rs1, $imm12">;
class RVY_bndswi<string opcodestr>
   : RVInstIBase<0b101, OPC_RVY_A, (outs YGPR:$rd),
                 (ins YGPR:$rs1, ybndsw_imm:$imm), opcodestr,
                 "$rd, $rs1, $imm"> {
  bits<5> rd;
  bits<5> rs1;
  bits<9> imm;

  let Inst{31-29} = 0b111;
  let Inst{28-20} = imm;
}
} // hasSideEffects = 0, mayLoad = 0, mayStore = 0

let Predicates = [HasStdExtY] in {
//
// Instructions to Update The Capability Pointer
//
let isReMaterializable = true, isAsCheapAsAMove = true in {
def YADD : RVY_rr<0b0000011, "yadd", YGPR, YGPR, GPRNoX0>;
def YADDI : RVY_ri<0b100, "yaddi">;
}
def : InstAlias<"yadd $rd, $rs1, $imm12",
                (YADDI YGPR:$rd, YGPR:$rs1, simm12_lo:$imm12), 0>;
def YADDRW : RVY_rr<0b0001011, "yaddrw", YGPR, YGPR, GPR>;

//
// Instructions to Manipulate Capabilities
//
def YPERMC : RVY_rr<0b0010011, "ypermc", YGPR, YGPR, GPR>;
// YMV is encoded as YADD with rs2=x0
let isReMaterializable = true, isAsCheapAsAMove = true, isMoveReg = true in
def YMV : RVYInst2Op<0b0000011, 0b00000, YGPR, YGPR, "ymv", "$rd, $rs1">;
def YHIW : RVY_rr<0b0000001, "yhiw", YGPR, GPR, GPR>;
def YBNDSW : RVY_rr<0b0011011, "ybndsw", YGPR, YGPR, GPR>;
def YBNDSWI : RVY_bndswi<"ybndswi">;
def : InstAlias<"ybndsw $rd, $rs1, $imm",
                (YBNDSWI YGPR:$rd, YGPR:$rs1, ybndsw_imm:$imm), 0>;
def YBNDSRW : RVY_rr<0b0100011, "ybndsrw", YGPR, YGPR, GPR>;
def YUNSEALS : RVY_rr<0b0000111, "yunseals", YGPR, YGPR, YGPR>;
def YSEALE : RVY_rr_fixed_rs1<0b0010111, 0b00000, "yseale", YGPR, YGPR>;
def YBLD : RVY_rr<0b0001111, "ybld", YGPR, YGPR, YGPR>;

//
// Instructions to Decode Capability Bounds
//
def YBASER : RVY_r_xy<0b00000, "ybaser">;
def YLENR : RVY_r_xy<0b00011, "ylenr">;
def YTOPR : RVY_r_xy<0b00010, "ytopr">;

//
// Instructions to Extract Capability Fields
//
def YTAGR : RVY_r_xy<0b00100, "ytagr">;
def YPERMR : RVY_r_xy<0b00001, "ypermr">;
def YTYPER : RVY_r_xy<0b00101, "ytyper">;
def YHIR : RVY_r_xy<0b00111, "yhir">;

//
// Miscellaneous Instructions to Handle Capability Data
//
def YEQ : RVY_rr<0b0000110, "yeq", GPR, YGPR, YGPR>;
def YSS : RVY_rr<0b0001110, "yss", GPR, YGPR, YGPR>;
def YAMASK : RVY_r_xx<0b00000, "yamask">;
} // Predicates = [HasStdExtY]

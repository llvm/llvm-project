//===-- RISCVInstrInfoY.td - RISCV instructions -------------*- tblgen-*---===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file describes the RISC-V instructions from the standard 'Base Y'
// Packed SIMD instruction set extension.
//
//  This version is still experimental as the 'Y' extension hasn't been
//  ratified yet.
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

def ybndsw_imm : Operand<XLenVT>, ImmLeaf<XLenVT, [{
  return (Imm >= 1 && Imm <= 256) ||
         (Imm >= 258 && Imm <= 768 && (Imm % 2) == 0) ||
         (Imm >= 772 && Imm <= 1792 && (Imm % 4) == 0) ||
         (Imm >= 1800 && Imm <= 3840 && (Imm % 8) == 0);
}]> {
  let EncoderMethod = "getYBNDSWImmOpValue";
  let ParserMatchClass = YBNDSWImmOperand;
  let DecoderMethod = "DecodeYBNDSWImm";
  let MIOperandInfo = (ops i32imm);
}

def uimm7_srliy : RISCVUImmOp<7>, ImmLeaf<XLenVT, [{
  return Imm == Subtarget->getXLen();
}]> {
  let ParserMatchClass = UImmAsmOperand<7, "Srliy">;
  let DecoderMethod = "decodeUImm7SrliyOperand";
  let MCOperandPredicate = [{
    int64_t Imm;
    if (!MCOp.evaluateAsConstantImm(Imm))
      return false;
    return Imm == Subtarget->getXLen()
  }];
  let OperandType = "OPERAND_UIMM7_SRLIY";
}

//===----------------------------------------------------------------------===//
// Instruction Formats and class templates
//===----------------------------------------------------------------------===//

// Like an RVInstR, except rs2 is now an additional function code.
class RVYInstSrcDst<bits<7> funct7, bits<5> funct5, bits<3> funct3,
                    RISCVOpcode opcode, dag outs, dag ins, string opcodestr,
                    string argstr>
    : RVInst<outs, ins, opcodestr, argstr, [], InstFormatR> {
  bits<5> rs1;
  bits<5> rd;

  let Inst{31-25} = funct7;
  let Inst{24-20} = funct5;
  let Inst{19-15} = rs1;
  let Inst{14-12} = funct3;
  let Inst{11-7} = rd;
  let Inst{6-0} = opcode.Value;
}

let hasSideEffects = 0, mayLoad = 0, mayStore = 0 in {
class RVY_r<bits<5> funct5, string opcodestr, DAGOperand rdOp = GPR,
            DAGOperand rs1Op = YGPR>
    : RVYInstSrcDst<0b0001000, funct5, 0b000, OPC_OP, (outs rdOp:$rd),
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
   : RVInstIBase<0b011, OPC_OP_IMM_32, (outs YGPR:$rd),
                 (ins YGPR:$rs1, ybndsw_imm:$imm), opcodestr,
                 "$rd, $rs1, $imm"> {
  bits<5> rd;
  bits<5> rs1;
  bits<10> imm;

  let Inst{31-30} = 0b00;
  let Inst{29-20} = imm;
}
class RVY_srli<string opcodestr, bits<7> imm>
    : RVInstIBase<0b101, OPC_OP_IMM, (outs GPR:$rd),
                  (ins YGPR:$rs1, uimm7_srliy:$shamt), opcodestr,
                  "$rd, $rs1, $shamt"> {
  bits<7> shamt = imm;

  let Inst{31-27} = 0b00000;
  let Inst{26-20} = shamt;
}
} // hasSideEffects = 0, mayLoad = 0, mayStore = 0

let Predicates = [HasStdExtY] in {
//
// Instructions to Update The Capability Pointer
//
let isReMaterializable = true, isAsCheapAsAMove = true in {
def ADDY : RVY_rr<0b0000110, 0b000, "addy", YGPR, YGPR, GPRNoX0>;
def ADDIY : RVY_ri<0b010, OPC_OP_IMM_32, "addiy">;
}
def : InstAlias<"addy $rd, $rs1, $imm12",
                (ADDIY YGPR:$rd, YGPR:$rs1, simm12_lo:$imm12), 0>;
def YADDRW : RVY_rr<0b0000110, 0b001, "yaddrw", YGPR, YGPR, GPR>;

//
// Instructions to Manipulate Capabilities
//
def YPERMC : RVY_rr<0b0000110, 0b010, "ypermc", YGPR, YGPR, GPR>;
let hasSideEffects = 0, mayLoad = 0, mayStore = 0, isReMaterializable = true,
    isAsCheapAsAMove = true, isMoveReg = true in
def YMV : RVYInstSrcDst<0b0000110, 0b00000, 0b000, OPC_OP, (outs YGPR:$rd),
                        (ins YGPR:$rs1), "ymv", "$rd, $rs1">;
def PACKY : RVY_rr<0b0000100, 0b011, "packy", YGPR, YGPR, GPR>;
def : MnemonicAlias<"yhiw", "packy">;
def YBNDSW : RVY_rr<0b0000111, 0b000, "ybndsw", YGPR, YGPR, GPR>;
def YBNDSWI : RVY_bndswi<"ybndswi">;
def : InstAlias<"ybndsw $rd, $rs1, $imm",
                (YBNDSWI YGPR:$rd, YGPR:$rs1, ybndsw_imm:$imm), 0>;
def YBNDSRW : RVY_rr<0b0000111, 0b001, "ybndsrw", YGPR, YGPR, GPR>;
def YSUNSEAL : RVY_rr<0b0000111, 0b010, "ysunseal", YGPR, YGPR, YGPR>;

//
// Instructions to Decode Capability Bounds
//
def YBASER : RVY_r<0b00101, "ybaser", GPR, YGPR>;
def YLENR : RVY_r<0b00110, "ylenr", GPR, YGPR>;

//
// Instructions to Extract Capability Fields
//
def YTAGR : RVY_r<0b00000, "ytagr", GPR, YGPR>;
def YPERMR : RVY_r<0b00001, "ypermr", GPR, YGPR>;
def YTYPER : RVY_r<0b00010, "ytyper", GPR, YGPR>;
// Extracting the high bits of capabilities is encoded as a XLEN-bit right shift
// (SRLIY), with the only legal immediate being XLEN. YHIR is a pseudo that
// expands to the correct shift value depending on RV32/RV64.
// Since this instruction overlaps with the SRLI encoding we split it into two
// instructions for RV32/RV64 so that it can be decoded correctly since a fixed
// field is decoded before a given variable immediate one.
// The alternative would be to use a decoder namespace
let Predicates = [HasStdExtY, IsRV64] in {
def SRLIY64 : RVY_srli<"srliy", 64>;
def : InstAlias<"yhir $rd, $rs1", (SRLIY64 GPR:$rd, YGPR:$rs1, 64)>;
}
let Predicates = [HasStdExtY, IsRV32] in {
def SRLIY32 : RVY_srli<"srliy", 32>;
def : InstAlias<"yhir $rd, $rs1", (SRLIY32 GPR:$rd, YGPR:$rs1, 32)>;
}
//
// Miscellaneous Instructions to Handle Capability Data
//
def SYEQ : RVY_rr<0b0000110, 0b100, "syeq", GPR, YGPR, YGPR>;
def YLT : RVY_rr<0b0000110, 0b110, "ylt", GPR, YGPR, YGPR>;
def YAMASK : RVY_r<0b00111, "yamask", GPR, GPR>;
} // Predicates = [HasStdExtY]

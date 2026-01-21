//===-- RISCVInstrFormatsY.td --------------------------------*- tablegen -*-=//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
//  This file describes the RISC-V Y extension instruction formats.
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
  return Subtarget->is64Bit() ? Imm == 64 : Imm == 32;
}]> {
  let ParserMatchClass = UImmAsmOperand<7, "Srliy">;
  let DecoderMethod = "decodeUImm7SrliyOperand";
  let MCOperandPredicate = [{
    int64_t Imm;
    if (!MCOp.evaluateAsConstantImm(Imm))
      return false;
    return Subtarget->is64Bit() ? Imm == 64 : Imm == 32;
  }];
  let OperandType = "OPERAND_UIMM7_SRLIY";
}

//===----------------------------------------------------------------------===//
// Instruction Formats
//===----------------------------------------------------------------------===//

// Like an RVInstR, except rs2 is now an additional function code.
class RVYInstSrcDst<bits<7> funct7, bits<5> funct5, bits<3> funct3,
                    RISCVOpcode opcode, dag outs, dag ins, string opcodestr,
                    string argstr>
    : RVInst<outs, ins, opcodestr, argstr, [], InstFormatR> {
  bits<5> rs1;
  bits<5> rd;

  let Inst{31 -25} = funct7;
  let Inst{24 -20} = funct5;
  let Inst{19 -15} = rs1;
  let Inst{14 -12} = funct3;
  let Inst{11 -7} = rd;
  let Inst{6 -0} = opcode.Value;
}

class RVYInstSetBoundsImmFmt<dag outs, dag ins, string opcodestr, string argstr>
    : RVInstIBase<0b011, OPC_OP_IMM_32, outs, ins, opcodestr, argstr> {
  bits<5> rd;
  bits<5> rs1;
  bits<10> imm;

  let Inst{31 -30} = 0b00;
  let Inst{29 -20} = imm;
}

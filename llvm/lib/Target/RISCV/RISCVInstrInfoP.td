//===-- RISCVInstrInfoP.td - RISC-V 'P' instructions -------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file describes the RISC-V instructions from the standard 'Base P'
// Packed SIMD instruction set extension.
//
//  This version is still experimental as the 'P' extension hasn't been
//  ratified yet.
//
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// Operand and SDNode transformation definitions.
//===----------------------------------------------------------------------===//

def simm10 : RISCVSImmLeafOp<10>;

def SImm10UnsignedAsmOperand : SImmAsmOperand<10, "Unsigned"> {
  let RenderMethod = "addSImm10UnsignedOperands";
}

// A 10-bit signed immediate allowing range [-512, 1023]
// but represented as [-512, 511].
def simm10_unsigned : RISCVOp {
  let ParserMatchClass = SImm10UnsignedAsmOperand;
  let EncoderMethod = "getImmOpValue";
  let DecoderMethod = "decodeSImmOperand<10>";
  let OperandType = "OPERAND_SIMM10";
  let MCOperandPredicate = [{
    int64_t Imm;
    if (!MCOp.evaluateAsConstantImm(Imm))
      return false;
    return isInt<10>(Imm);
  }];
}

//===----------------------------------------------------------------------===//
// Instruction class templates
//===----------------------------------------------------------------------===//

let hasSideEffects = 0, mayLoad = 0, mayStore = 0 in
class PLI_i<bits<7> funct7, string opcodestr>
    : RVInst<(outs GPR:$rd), (ins simm10:$imm10), opcodestr, "$rd, $imm10", [],
             InstFormatOther> {
  bits<10> imm10;
  bits<5> rd;

  let Inst{31-25} = funct7;
  let Inst{24-16} = imm10{8-0};
  let Inst{15}    = imm10{9};
  let Inst{14-12} = 0b010;
  let Inst{11-7} = rd;
  let Inst{6-0} = OPC_OP_IMM_32.Value;
}

let hasSideEffects = 0, mayLoad = 0, mayStore = 0 in
class PLUI_i<bits<7> funct7, string opcodestr>
    : RVInst<(outs GPR:$rd), (ins simm10_unsigned:$imm10), opcodestr,
             "$rd, $imm10", [], InstFormatOther> {
  bits<10> imm10;
  bits<5> rd;

  let Inst{31-25} = funct7;
  let Inst{24}    = imm10{0};
  let Inst{23-15} = imm10{9-1};
  let Inst{14-12} = 0b010;
  let Inst{11-7} = rd;
  let Inst{6-0} = OPC_OP_IMM_32.Value;
}

let hasSideEffects = 0, mayLoad = 0, mayStore = 0 in
class PLI_B_i<bits<8> funct8, string opcodestr>
    : RVInst<(outs GPR:$rd), (ins uimm8:$uimm8), opcodestr, "$rd, $uimm8", [],
             InstFormatOther> {
  bits<8> uimm8;
  bits<5> rd;

  let Inst{31-24} = funct8;
  let Inst{23-16} = uimm8;
  let Inst{15}    = 0b0;
  let Inst{14-12} = 0b010;
  let Inst{11-7} = rd;
  let Inst{6-0} = OPC_OP_IMM_32.Value;
}

let hasSideEffects = 0, mayLoad = 0, mayStore = 0 in
class RVPShift_ri<bits<3> f, bits<3> funct3, string opcodestr, Operand ImmType>
    : RVInstIBase<funct3, OPC_OP_IMM_32, (outs GPR:$rd),
                  (ins GPR:$rs1, ImmType:$shamt), opcodestr,
                  "$rd, $rs1, $shamt"> {
  let Inst{31}    = 0b1;
  let Inst{30-28} = f;
  let Inst{27}    = 0b0;
}

class RVPShiftD_ri<bits<3> f, bits<3> funct3, string opcodestr>
    : RVPShift_ri<f, funct3, opcodestr, uimm6> {
  bits<6> shamt;

  let Inst{26} = 0b1;
  let Inst{25-20} = shamt;
}

class RVPShiftW_ri<bits<3> f, bits<3> funct3, string opcodestr>
    : RVPShift_ri<f, funct3, opcodestr, uimm5> {
  bits<5> shamt;

  let Inst{26-25} = 0b01;
  let Inst{24-20} = shamt;
}

class RVPShiftH_ri<bits<3> f, bits<3> funct3, string opcodestr>
    : RVPShift_ri<f, funct3, opcodestr, uimm4> {
  bits<4> shamt;

  let Inst{26-24} = 0b001;
  let Inst{23-20} = shamt;
}

class RVPShiftB_ri<bits<3> f, bits<3> funct3, string opcodestr>
    : RVPShift_ri<f, funct3, opcodestr, uimm3> {
  bits<3> shamt;

  let Inst{26-23} = 0b0001;
  let Inst{22-20} = shamt;
}

let hasSideEffects = 0, mayLoad = 0, mayStore = 0 in
class RVPUnary_ri<bits<2> w, bits<5> uf, string opcodestr>
    : RVInstIBase<0b010, OPC_OP_IMM_32, (outs GPR:$rd), (ins GPR:$rs1),
                  opcodestr, "$rd, $rs1">  {
  let Inst{31-27} = 0b11100;
  let Inst{26-25} = w;
  let Inst{24-20} = uf;
}

let hasSideEffects = 0, mayLoad = 0, mayStore = 0 in
class RVPBinaryScalar_rr<bits<3> f, bits<2> w, bits<3> funct3, string opcodestr>
    : RVInstRBase<funct3, OPC_OP_IMM_32, (outs GPR:$rd),
                  (ins GPR:$rs1, GPR:$rs2), opcodestr, "$rd, $rs1, $rs2"> {
  let Inst{31} = 0b1;
  let Inst{30-28} = f;
  let Inst{27} = 0b1;
  let Inst{26-25} = w;
}

//===----------------------------------------------------------------------===//
// Instructions
//===----------------------------------------------------------------------===//

let Predicates = [HasStdExtP] in {
  let IsSignExtendingOpW = 1 in
  def CLS    : Unary_r<0b011000000011, 0b001, "cls">;
  def ABS    : Unary_r<0b011000000111, 0b001, "abs">;
} // Predicates = [HasStdExtP]

let Predicates = [HasStdExtP, IsRV32] in {
  def REV_RV32  : Unary_r<0b011010011111, 0b101, "rev">;
} // Predicates = [HasStdExtP, IsRV32]

let Predicates = [HasStdExtP, IsRV64] in {
  def REV16      : Unary_r<0b011010110000, 0b101, "rev16">;
  def REV_RV64   : Unary_r<0b011010111111, 0b101, "rev">;

  let IsSignExtendingOpW = 1 in {
    def CLSW  : UnaryW_r<0b011000000011, 0b001, "clsw">;
    def ABSW  : UnaryW_r<0b011000000111, 0b001, "absw">;
  }
} // Predicates = [HasStdExtP, IsRV64]

let Predicates = [HasStdExtP] in {
  def PSLLI_B  : RVPShiftB_ri<0b000, 0b010, "pslli.b">;
  def PSLLI_H  : RVPShiftH_ri<0b000, 0b010, "pslli.h">;
  def PSSLAI_H : RVPShiftH_ri<0b101, 0b010, "psslai.h">;
} // Predicates = [HasStdExtP]
let Predicates = [HasStdExtP, IsRV32], DecoderNamespace = "RV32Only" in {
  def SSLAI    : RVPShiftW_ri<0b101, 0b010, "sslai">;
} // Predicates = [HasStdExtP, IsRV32], DecoderNamespace = "RV32Only"
let Predicates = [HasStdExtP, IsRV64] in {
  def PSLLI_W  : RVPShiftW_ri<0b000, 0b010, "pslli.w">;
  def PSSLAI_W : RVPShiftW_ri<0b101, 0b010, "psslai.w">;
} // Predicates = [HasStdExtP, IsRV64]

let Predicates = [HasStdExtP] in
def PLI_H : PLI_i<0b1011000, "pli.h">;
let Predicates = [HasStdExtP, IsRV64] in
def PLI_W : PLI_i<0b1011001, "pli.w">;
let Predicates = [HasStdExtP] in
def PLI_B : PLI_B_i<0b10110100, "pli.b">;

let Predicates = [HasStdExtP] in {
  def PSEXT_H_B : RVPUnary_ri<0b00, 0b00100, "psext.h.b">;
  def PSABS_H   : RVPUnary_ri<0b00, 0b00111, "psabs.h">;
  def PSABS_B   : RVPUnary_ri<0b10, 0b00111, "psabs.b">;
} // Predicates = [HasStdExtP]
let Predicates = [HasStdExtP, IsRV64] in {
  def PSEXT_W_B : RVPUnary_ri<0b01, 0b00100, "psext.w.b">;
  def PSEXT_W_H : RVPUnary_ri<0b01, 0b00101, "psext.w.h">;
} // Predicates = [HasStdExtP, IsRV64]

let Predicates = [HasStdExtP] in
def PLUI_H : PLUI_i<0b1111000, "plui.h">;
let Predicates = [HasStdExtP, IsRV64] in
def PLUI_W : PLUI_i<0b1111001, "plui.w">;

let Predicates = [HasStdExtP] in {
  def PSLL_HS   : RVPBinaryScalar_rr<0b000, 0b00, 0b010, "psll.hs">;
  def PSLL_BS   : RVPBinaryScalar_rr<0b000, 0b10, 0b010, "psll.bs">;

  def PADD_HS   : RVPBinaryScalar_rr<0b001, 0b00, 0b010, "padd.hs">;
  def PADD_BS   : RVPBinaryScalar_rr<0b001, 0b10, 0b010, "padd.bs">;

  def PSSHA_HS  : RVPBinaryScalar_rr<0b110, 0b00, 0b010, "pssha.hs">;

  def PSSHAR_HS : RVPBinaryScalar_rr<0b111, 0b00, 0b010, "psshar.hs">;
} // Predicates = [HasStdExtP]
let Predicates = [HasStdExtP, IsRV32], DecoderNamespace = "RV32Only" in {
  def SSHA      : RVPBinaryScalar_rr<0b110, 0b01, 0b010, "ssha">;

  def SSHAR     : RVPBinaryScalar_rr<0b111, 0b01, 0b010, "sshar">;
} // Predicates = [HasStdExtP, IsRV32]
let Predicates = [HasStdExtP, IsRV64] in {
  def PSLL_WS   : RVPBinaryScalar_rr<0b000, 0b01, 0b010, "psll.ws">;

  def PADD_WS   : RVPBinaryScalar_rr<0b001, 0b01, 0b010, "padd.ws">;

  def PSSHA_WS  : RVPBinaryScalar_rr<0b110, 0b01, 0b010, "pssha.ws">;
  def SHA       : RVPBinaryScalar_rr<0b110, 0b11, 0b010, "sha">;

  def PSSHAR_WS : RVPBinaryScalar_rr<0b111, 0b01, 0b010, "psshar.ws">;
  def SHAR      : RVPBinaryScalar_rr<0b111, 0b11, 0b010, "shar">;
} // Predicates = [HasStdExtP, IsRV64]

let Predicates = [HasStdExtP] in {
  def PSRLI_B    : RVPShiftB_ri<0b000, 0b100, "psrli.b">;
  def PSRLI_H    : RVPShiftH_ri<0b000, 0b100, "psrli.h">;

  def PUSATI_H   : RVPShiftH_ri<0b010, 0b100, "pusati.h">;

  def PSRAI_B    : RVPShiftB_ri<0b100, 0b100, "psrai.b">;
  def PSRAI_H    : RVPShiftH_ri<0b100, 0b100, "psrai.h">;

  def PSRARI_H   : RVPShiftH_ri<0b101, 0b100, "psrari.h">;

  def PSATI_H    : RVPShiftH_ri<0b110, 0b100, "psati.h">;
} // Predicates = [HasStdExtP]
let Predicates = [HasStdExtP, IsRV32], DecoderNamespace = "RV32Only" in {
  def USATI_RV32 : RVPShiftW_ri<0b010, 0b100, "usati">;

  def SRARI_RV32 : RVPShiftW_ri<0b101, 0b100, "srari">;

  def SATI_RV32  : RVPShiftW_ri<0b110, 0b100, "sati">;
} // Predicates = [HasStdExtP, IsRV32]
let Predicates = [HasStdExtP, IsRV64] in {
  def PSRLI_W    : RVPShiftW_ri<0b000, 0b100, "psrli.w">;
  def PSRAI_W    : RVPShiftW_ri<0b100, 0b100, "psrai.w">;

  def PUSATI_W   : RVPShiftW_ri<0b010, 0b100, "pusati.w">;
  def USATI_RV64 : RVPShiftD_ri<0b010, 0b100, "usati">;

  def PSRARI_W   : RVPShiftW_ri<0b101, 0b100, "psrari.w">;
  def SRARI_RV64 : RVPShiftD_ri<0b101, 0b100, "srari">;

  def PSATI_W    : RVPShiftW_ri<0b110, 0b100, "psati.w">;
  def SATI_RV64  : RVPShiftD_ri<0b110, 0b100, "sati">;
} // Predicates = [HasStdExtP, IsRV64]

let Predicates = [HasStdExtP] in {
  def PSRL_HS     : RVPBinaryScalar_rr<0b000, 0b00, 0b100, "psrl.hs">;
  def PSRL_BS     : RVPBinaryScalar_rr<0b000, 0b10, 0b100, "psrl.bs">;

  def PREDSUM_HS  : RVPBinaryScalar_rr<0b001, 0b00, 0b100, "predsum.hs">;
  def PREDSUM_BS  : RVPBinaryScalar_rr<0b001, 0b10, 0b100, "predsum.bs">;

  def PREDSUMU_HS : RVPBinaryScalar_rr<0b011, 0b00, 0b100, "predsumu.hs">;
  def PREDSUMU_BS : RVPBinaryScalar_rr<0b011, 0b10, 0b100, "predsumu.bs">;

  def PSRA_HS     : RVPBinaryScalar_rr<0b100, 0b00, 0b100, "psra.hs">;
  def PSRA_BS     : RVPBinaryScalar_rr<0b100, 0b10, 0b100, "psra.bs">;
} // Predicates = [HasStdExtP]
let Predicates = [HasStdExtP, IsRV64] in {
  def PSRL_WS     : RVPBinaryScalar_rr<0b000, 0b01, 0b100, "psrl.ws">;

  def PREDSUM_WS  : RVPBinaryScalar_rr<0b001, 0b01, 0b100, "predsum.ws">;

  def PREDSUMU_WS : RVPBinaryScalar_rr<0b011, 0b01, 0b100, "predsumu.ws">;

  def PSRA_WS     : RVPBinaryScalar_rr<0b100, 0b01, 0b100, "psra.ws">;
} // Predicates = [HasStdExtP, IsRV64]

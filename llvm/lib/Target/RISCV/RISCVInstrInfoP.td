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

def simm10 : RISCVSImmOp<10>;

def SImm8UnsignedAsmOperand : SImmAsmOperand<8, "Unsigned"> {
  let RenderMethod = "addSImm8UnsignedOperands";
}

// A 8-bit signed immediate allowing range [-128, 255]
// but represented as [-128, 255].
def simm8_unsigned : RISCVOp {
  let ParserMatchClass = SImm8UnsignedAsmOperand;
  let EncoderMethod = "getImmOpValue";
  let DecoderMethod = "decodeSImmOperand<8>";
  let OperandType = "OPERAND_SIMM8_UNSIGNED";
  let MCOperandPredicate = [{
    int64_t Imm;
    if (!MCOp.evaluateAsConstantImm(Imm))
      return false;
    return isInt<8>(Imm);
  }];
}

def SImm10UnsignedAsmOperand : SImmAsmOperand<10, "Unsigned"> {
  let RenderMethod = "addSImm10UnsignedOperands";
}

// A 10-bit signed immediate allowing range [-512, 1023]
// but represented as [-512, 511].
def simm10_unsigned : RISCVOp {
  let ParserMatchClass = SImm10UnsignedAsmOperand;
  let EncoderMethod = "getImmOpValue";
  let DecoderMethod = "decodeSImmOperand<10>";
  let OperandType = "OPERAND_SIMM10_UNSIGNED";
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

// Common base for pli.b/h/w and plui.h/w
class RVPLoadImm_i<bits<7> funct7, dag ins, string opcodestr,
                   string argstr>
    : RVInst<(outs GPR:$rd), ins, opcodestr, argstr, [],
             InstFormatOther> {
  bits<5> rd;

  let Inst{31-25} = funct7;
  let Inst{14-12} = 0b010;
  let Inst{11-7} = rd;
  let Inst{6-0} = OPC_OP_IMM_32.Value;

  let hasSideEffects = 0;
  let mayLoad = 0;
  let mayStore = 0;
}

// Base for pli.h/w.
class PLI_i<bits<7> funct7, string opcodestr>
    : RVPLoadImm_i<funct7, (ins simm10:$imm10), opcodestr, "$rd, $imm10"> {
  bits<10> imm10;

  let Inst{24-16} = imm10{8-0};
  let Inst{15}    = imm10{9};
}

// Base for plui.h/w.
class PLUI_i<bits<7> funct7, string opcodestr>
    : RVPLoadImm_i<funct7, (ins simm10_unsigned:$imm10), opcodestr,
                   "$rd, $imm10"> {
  bits<10> imm10;

  let Inst{24}    = imm10{0};
  let Inst{23-15} = imm10{9-1};
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

let hasSideEffects = 0, mayLoad = 0, mayStore = 0 in
class RVPBinary_rr<bits<4> f, bits<2> w, bits<3> funct3, string opcodestr>
    : RVInstRBase<funct3, OPC_OP_32, (outs GPR:$rd),
                  (ins GPR:$rs1, GPR:$rs2), opcodestr, "$rd, $rs1, $rs2"> {
  let Inst{31} = 0b1;
  let Inst{30-27} = f;
  let Inst{26-25} = w;
}

let hasSideEffects = 0, mayLoad = 0, mayStore = 0 in
class RVPTernary_rrr<bits<4> f, bits<2> w, bits<3> funct3, string opcodestr>
    : RVInstRBase<funct3, OPC_OP_32, (outs GPR:$rd_wb),
                  (ins GPR:$rd, GPR:$rs1, GPR:$rs2), opcodestr,
                  "$rd, $rs1, $rs2"> {
  let Inst{31} = 0b1;
  let Inst{30-27} = f;
  let Inst{26-25} = w;

  let Constraints = "$rd = $rd_wb";
}

// Common base for pli.db/h/w and plui.dh/w
class RVPPairLoadImm_i<bits<7> funct7, dag ins, string opcodestr,
                       string argstr>
    : RVInst<(outs GPRPairRV32:$rd), ins, opcodestr, argstr, [],
             InstFormatOther> {
  bits<5> rd;

  let Inst{31-25} = funct7;
  let Inst{14-12} = 0b010;
  let Inst{11-8}  = rd{4-1};
  let Inst{7}     = 0b0;
  let Inst{6-0}   = OPC_OP_IMM_32.Value;

  let hasSideEffects = 0;
  let mayLoad = 0;
  let mayStore = 0;
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
let Predicates = [HasStdExtP] in {
  def PLI_B : RVPLoadImm_i<0b1011010, (ins simm8_unsigned:$imm8), "pli.b",
                           "$rd, $imm8"> {
    bits<8> imm8;

    let Inst{24}    = 0b0;
    let Inst{23-16} = imm8;
    let Inst{15}    = 0b0;
  }
}

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

let Predicates = [HasStdExtP] in {
  def PADD_H   : RVPBinary_rr<0b0000, 0b00, 0b000, "padd.h">;
  def PADD_B   : RVPBinary_rr<0b0000, 0b10, 0b000, "padd.b">;

  def PSADD_H  : RVPBinary_rr<0b0010, 0b00, 0b000, "psadd.h">;
  def PSADD_B  : RVPBinary_rr<0b0010, 0b10, 0b000, "psadd.b">;

  def PAADD_H  : RVPBinary_rr<0b0011, 0b00, 0b000, "paadd.h">;
  def PAADD_B  : RVPBinary_rr<0b0011, 0b10, 0b000, "paadd.b">;

  def PSADDU_H : RVPBinary_rr<0b0110, 0b00, 0b000, "psaddu.h">;
  def PSADDU_B : RVPBinary_rr<0b0110, 0b10, 0b000, "psaddu.b">;

  def PAADDU_H : RVPBinary_rr<0b0111, 0b00, 0b000, "paaddu.h">;
  def PAADDU_B : RVPBinary_rr<0b0111, 0b10, 0b000, "paaddu.b">;

  def PSUB_H   : RVPBinary_rr<0b1000, 0b00, 0b000, "psub.h">;
  def PSUB_B   : RVPBinary_rr<0b1000, 0b10, 0b000, "psub.b">;

  def PDIF_H   : RVPBinary_rr<0b1001, 0b00, 0b000, "pdif.h">;
  def PDIF_B   : RVPBinary_rr<0b1001, 0b10, 0b000, "pdif.b">;

  def PSSUB_H  : RVPBinary_rr<0b1010, 0b00, 0b000, "pssub.h">;
  def PSSUB_B  : RVPBinary_rr<0b1010, 0b10, 0b000, "pssub.b">;

  def PASUB_H  : RVPBinary_rr<0b1011, 0b00, 0b000, "pasub.h">;
  def PASUB_B  : RVPBinary_rr<0b1011, 0b10, 0b000, "pasub.b">;

  def PDIFU_H  : RVPBinary_rr<0b1101, 0b00, 0b000, "pdifu.h">;
  def PDIFU_B  : RVPBinary_rr<0b1101, 0b10, 0b000, "pdifu.b">;

  def PSSUBU_H : RVPBinary_rr<0b1110, 0b00, 0b000, "pssubu.h">;
  def PSSUBU_B : RVPBinary_rr<0b1110, 0b10, 0b000, "pssubu.b">;

  def PASUBU_H : RVPBinary_rr<0b1111, 0b00, 0b000, "pasubu.h">;
  def PASUBU_B : RVPBinary_rr<0b1111, 0b10, 0b000, "pasubu.b">;
} // Predicates = [HasStdExtP]
let Predicates = [HasStdExtP, IsRV32], DecoderNamespace = "RV32Only" in {
  def SADD     : RVPBinary_rr<0b0010, 0b01, 0b000, "sadd">;

  def AADD     : RVPBinary_rr<0b0011, 0b01, 0b000, "aadd">;

  def SADDU    : RVPBinary_rr<0b0110, 0b01, 0b000, "saddu">;

  def AADDU    : RVPBinary_rr<0b0111, 0b01, 0b000, "aaddu">;

  def SSUB     : RVPBinary_rr<0b1010, 0b01, 0b000, "ssub">;

  def ASUB     : RVPBinary_rr<0b1011, 0b01, 0b000, "asub">;

  def SSUBU    : RVPBinary_rr<0b1110, 0b01, 0b000, "ssubu">;

  def ASUBU    : RVPBinary_rr<0b1111, 0b01, 0b000, "asubu">;
} // Predicates = [HasStdExtP, IsRV32], DecoderNamespace = "RV32Only"
let Predicates = [HasStdExtP, IsRV64] in {
  def PADD_W   : RVPBinary_rr<0b0000, 0b01, 0b000, "padd.w">;

  def PSADD_W  : RVPBinary_rr<0b0010, 0b01, 0b000, "psadd.w">;

  def PAADD_W  : RVPBinary_rr<0b0011, 0b01, 0b000, "paadd.w">;

  def PSADDU_W : RVPBinary_rr<0b0110, 0b01, 0b000, "psaddu.w">;

  def PAADDU_W : RVPBinary_rr<0b0111, 0b01, 0b000, "paaddu.w">;

  def PSUB_W   : RVPBinary_rr<0b1000, 0b01, 0b000, "psub.w">;

  def PSSUB_W  : RVPBinary_rr<0b1010, 0b01, 0b000, "pssub.w">;

  def PASUB_W  : RVPBinary_rr<0b1011, 0b01, 0b000, "pasub.w">;

  def PSSUBU_W : RVPBinary_rr<0b1110, 0b01, 0b000, "pssubu.w">;

  def PASUBU_W : RVPBinary_rr<0b1111, 0b01, 0b000, "pasubu.w">;
} // Predicates = [HasStdExtP, IsRV64]

let Predicates = [HasStdExtP] in {
  def SLX          : RVPBinary_rr<0b0001, 0b11, 0b001, "slx">;

  def PMUL_H_B01   : RVPBinary_rr<0b0010, 0b00, 0b001, "pmul.h.b01">;

  def MVM          : RVPTernary_rrr<0b0101, 0b00, 0b001, "mvm">;
  def MVMN         : RVPTernary_rrr<0b0101, 0b01, 0b001, "mvmn">;
  def MERGE        : RVPTernary_rrr<0b0101, 0b10, 0b001, "merge">;
  def SRX          : RVPTernary_rrr<0b0101, 0b11, 0b001, "srx">;

  def PMULU_H_B01  : RVPBinary_rr<0b0110, 0b00, 0b001, "pmulu.h.b01">;
  def PDIFSUMU_B   : RVPBinary_rr<0b0110, 0b10, 0b001, "pdifsumu.b">;

  def PDIFSUMAU_B  : RVPTernary_rrr<0b0111, 0b10, 0b001, "pdifsumau.b">;
} // Predicates = [HasStdExtP]
let Predicates = [HasStdExtP, IsRV32], DecoderNamespace = "RV32Only" in {
  def MUL_H01      : RVPBinary_rr<0b0010, 0b01, 0b001, "mul.h01">;

  def MACC_H01     : RVPTernary_rrr<0b0011, 0b01, 0b001, "macc.h01">;

  def MULU_H01     : RVPBinary_rr<0b0110, 0b01, 0b001, "mulu.h01">;

  def MACCU_H01    : RVPTernary_rrr<0b0111, 0b01, 0b001, "maccu.h01">;
} // Predicates = [HasStdExtP, IsRV32], DecoderNamespace = "RV32Only"
let Predicates = [HasStdExtP, IsRV64] in {
  def PMUL_W_H01   : RVPBinary_rr<0b0010, 0b01, 0b001, "pmul.w.h01">;
  def MUL_W01      : RVPBinary_rr<0b0010, 0b11, 0b001, "mul.w01">;

  def PMACC_W_H01  : RVPTernary_rrr<0b0011, 0b01, 0b001, "pmacc.w.h01">;
  def MACC_W01     : RVPTernary_rrr<0b0011, 0b11, 0b001, "macc.w01">;

  def PMULU_W_H01  : RVPBinary_rr<0b0110, 0b01, 0b001, "pmulu.w.h01">;
  def MULU_W01     : RVPBinary_rr<0b0110, 0b11, 0b001, "mulu.w01">;

  def PMACCU_W_H01 : RVPTernary_rrr<0b0111, 0b01, 0b001, "pmaccu.w.h01">;
  def MACCU_W01    : RVPTernary_rrr<0b0111, 0b11, 0b001, "maccu.w01">;
} // Predicates = [HasStdExtP, IsRV64]

// Note the spec has a 3-bit f field in bits 30:28 with 0 in bit 27.
// Here we include the 0 in the f field to reduce number of tablegen classes.
let Predicates = [HasStdExtP] in {
  def PSH1ADD_H   : RVPBinary_rr<0b0100, 0b00, 0b010, "psh1add.h">;

  def PSSH1SADD_H : RVPBinary_rr<0b0110, 0b00, 0b010, "pssh1sadd.h">;
} // Predicates = [HasStdExtP]
let Predicates = [HasStdExtP, IsRV32], DecoderNamespace = "RV32Only" in {
  def SSH1SADD    : RVPBinary_rr<0b0110, 0b01, 0b010, "ssh1sadd">;
} // Predicates = [HasStdExtP, IsRV32], DecoderNamespace = "RV32Only"
let Predicates = [HasStdExtP, IsRV64] in {
  def PSH1ADD_W   : RVPBinary_rr<0b0100, 0b01, 0b010, "psh1add.w">;

  def PSSH1SADD_W : RVPBinary_rr<0b0110, 0b01, 0b010, "pssh1sadd.w">;

  def UNZIP8P     : RVPBinary_rr<0b1100, 0b00, 0b010, "unzip8p">;
  def UNZIP16P    : RVPBinary_rr<0b1100, 0b01, 0b010, "unzip16p">;
  def UNZIP8HP    : RVPBinary_rr<0b1100, 0b10, 0b010, "unzip8hp">;
  def UNZIP16HP   : RVPBinary_rr<0b1100, 0b11, 0b010, "unzip16hp">;

  def ZIP8P       : RVPBinary_rr<0b1110, 0b00, 0b010, "zip8p">;
  def ZIP16P      : RVPBinary_rr<0b1110, 0b01, 0b010, "zip16p">;
  def ZIP8HP      : RVPBinary_rr<0b1110, 0b10, 0b010, "zip8hp">;
  def ZIP16HP     : RVPBinary_rr<0b1110, 0b11, 0b010, "zip16hp">;
} // Predicates = [HasStdExtP, IsRV64]

let Predicates = [HasStdExtP] in {
  def PMUL_H_B00    : RVPBinary_rr<0b0000, 0b00, 0b011, "pmul.h.b00">;

  def PMUL_H_B11    : RVPBinary_rr<0b0010, 0b00, 0b011, "pmul.h.b11">;

  def PMULU_H_B00   : RVPBinary_rr<0b0100, 0b00, 0b011, "pmulu.h.b00">;

  def PMULU_H_B11   : RVPBinary_rr<0b0110, 0b00, 0b011, "pmulu.h.b11">;

  def PMULSU_H_B00  : RVPBinary_rr<0b1100, 0b00, 0b011, "pmulsu.h.b00">;

  def PMULSU_H_B11  : RVPBinary_rr<0b1110, 0b00, 0b011, "pmulsu.h.b11">;
} // Predicates = [HasStdExtP]
let Predicates = [HasStdExtP, IsRV32], DecoderNamespace = "RV32Only" in {
  def MUL_H00       : RVPBinary_rr<0b0000, 0b01, 0b011, "mul.h00">;

  def MACC_H00      : RVPTernary_rrr<0b0001, 0b01, 0b011, "macc.h00">;

  def MUL_H11       : RVPBinary_rr<0b0010, 0b01, 0b011, "mul.h11">;

  def MACC_H11      : RVPTernary_rrr<0b0011, 0b01, 0b011, "macc.h11">;

  def MULU_H00      : RVPBinary_rr<0b0100, 0b01, 0b011, "mulu.h00">;

  def MACCU_H00     : RVPTernary_rrr<0b0101, 0b01, 0b011, "maccu.h00">;

  def MULU_H11      : RVPBinary_rr<0b0110, 0b01, 0b011, "mulu.h11">;

  def MACCU_H11     : RVPTernary_rrr<0b0111, 0b01, 0b011, "maccu.h11">;

  def MULSU_H00     : RVPBinary_rr<0b1100, 0b01, 0b011, "mulsu.h00">;

  def MACCSU_H00    : RVPTernary_rrr<0b1101, 0b01, 0b011, "maccsu.h00">;

  def MULSU_H11     : RVPBinary_rr<0b1110, 0b01, 0b011, "mulsu.h11">;

  def MACCSU_H11    : RVPTernary_rrr<0b1111, 0b01, 0b011, "maccsu.h11">;
} // Predicates = [HasStdExtP, IsRV32], DecoderNamespace = "RV32Only"
let Predicates = [HasStdExtP, IsRV64] in {
  def PMUL_W_H00    : RVPBinary_rr<0b0000, 0b01, 0b011, "pmul.w.h00">;
  def MUL_W00       : RVPBinary_rr<0b0000, 0b11, 0b011, "mul.w00">;

  def PMACC_W_H00   : RVPTernary_rrr<0b0001, 0b01, 0b011, "pmacc.w.h00">;
  def MACC_W00      : RVPTernary_rrr<0b0001, 0b11, 0b011, "macc.w00">;

  def PMUL_W_H11    : RVPBinary_rr<0b0010, 0b01, 0b011, "pmul.w.h11">;
  def MUL_W11       : RVPBinary_rr<0b0010, 0b11, 0b011, "mul.w11">;

  def PMACC_W_H11   : RVPTernary_rrr<0b0011, 0b01, 0b011, "pmacc.w.h11">;
  def MACC_W11      : RVPTernary_rrr<0b0011, 0b11, 0b011, "macc.w11">;

  def PMULU_W_H00   : RVPBinary_rr<0b0100, 0b01, 0b011, "pmulu.w.h00">;
  def MULU_W00      : RVPBinary_rr<0b0100, 0b11, 0b011, "mulu.w00">;

  def PMACCU_W_H00  : RVPTernary_rrr<0b0101, 0b01, 0b011, "pmaccu.w.h00">;
  def MACCU_W00     : RVPTernary_rrr<0b0101, 0b11, 0b011, "maccu.w00">;

  def PMULU_W_H11   : RVPBinary_rr<0b0110, 0b01, 0b011, "pmulu.w.h11">;
  def MULU_W11      : RVPBinary_rr<0b0110, 0b11, 0b011, "mulu.w11">;

  def PMACCU_W_H11  : RVPTernary_rrr<0b0111, 0b01, 0b011, "pmaccu.w.h11">;
  def MACCU_W11     : RVPTernary_rrr<0b0111, 0b11, 0b011, "maccu.w11">;

  def PMULSU_W_H00  : RVPBinary_rr<0b1100, 0b01, 0b011, "pmulsu.w.h00">;
  def MULSU_W00     : RVPBinary_rr<0b1100, 0b11, 0b011, "mulsu.w00">;

  def PMACCSU_W_H00 : RVPTernary_rrr<0b1101, 0b01, 0b011, "pmaccsu.w.h00">;
  def MACCSU_W00    : RVPTernary_rrr<0b1101, 0b11, 0b011, "maccsu.w00">;

  def PMULSU_W_H11  : RVPBinary_rr<0b1110, 0b01, 0b011, "pmulsu.w.h11">;
  def MULSU_W11     : RVPBinary_rr<0b1110, 0b11, 0b011, "mulsu.w11">;

  def PMACCSU_W_H11 : RVPTernary_rrr<0b1111, 0b01, 0b011, "pmaccsu.w.h11">;
  def MACCSU_W11    : RVPTernary_rrr<0b1111, 0b11, 0b011, "maccsu.w11">;
} // Predicates = [HasStdExtP, IsRV64]

// Note the spec has a 3-bit f field in bits 30:28 with 0 in bit 27.
// Here we include the 0 in the f field to reduce number of tablegen classes.
let Predicates = [HasStdExtP] in {
  def PPACK_H     : RVPBinary_rr<0b0000, 0b00, 0b100, "ppack.h">;

  def PPACKBT_H   : RVPBinary_rr<0b0010, 0b00, 0b100, "ppackbt.h">;

  def PPACKTB_H   : RVPBinary_rr<0b0100, 0b00, 0b100, "ppacktb.h">;

  def PPACKT_H    : RVPBinary_rr<0b0110, 0b00, 0b100, "ppackt.h">;
} // Predicates = [HasStdExtP]
let Predicates = [HasStdExtP, IsRV32], DecoderNamespace = "RV32Only" in {
  def PACKBT_RV32 : RVPBinary_rr<0b0010, 0b01, 0b100, "packbt">;

  def PACKTB_RV32 : RVPBinary_rr<0b0100, 0b01, 0b100, "packtb">;

  def PACKT_RV32  : RVPBinary_rr<0b0110, 0b01, 0b100, "packt">;
} // Predicates = [HasStdExtP, IsRV32], DecoderNamespace = "RV32Only"
let Predicates = [HasStdExtP, IsRV64] in {
  def PPACK_W     : RVPBinary_rr<0b0000, 0b01, 0b100, "ppack.w">;

  def PPACKBT_W   : RVPBinary_rr<0b0010, 0b01, 0b100, "ppackbt.w">;
  def PACKBT_RV64 : RVPBinary_rr<0b0010, 0b11, 0b100, "packbt">;

  def PPACKTB_W   : RVPBinary_rr<0b0100, 0b01, 0b100, "ppacktb.w">;
  def PACKTB_RV64 : RVPBinary_rr<0b0100, 0b11, 0b100, "packtb">;

  def PPACKT_W    : RVPBinary_rr<0b0110, 0b01, 0b100, "ppackt.w">;
  def PACKT_RV64  : RVPBinary_rr<0b0110, 0b11, 0b100, "packt">;
} // Predicates = [HasStdExtP, IsRV64]

let Predicates = [HasStdExtP] in {
  def PM2ADD_H    : RVPBinary_rr<0b0000, 0b00, 0b101, "pm2add.h">;
  def PM4ADD_B    : RVPBinary_rr<0b0000, 0b10, 0b101, "pm4add.b">;

  def PM2ADDA_H   : RVPTernary_rrr<0b0001, 0b00, 0b101, "pm2adda.h">;
  def PM4ADDA_B   : RVPTernary_rrr<0b0001, 0b10, 0b101, "pm4adda.b">;

  def PM2ADD_HX   : RVPBinary_rr<0b0010, 0b00, 0b101, "pm2add.hx">;

  def PM2ADDA_HX  : RVPTernary_rrr<0b0011, 0b00, 0b101, "pm2adda.hx">;

  def PM2ADDU_H   : RVPBinary_rr<0b0100, 0b00, 0b101, "pm2addu.h">;
  def PM4ADDU_B   : RVPBinary_rr<0b0100, 0b10, 0b101, "pm4addu.b">;

  def PM2ADDAU_H  : RVPTernary_rrr<0b0101, 0b00, 0b101, "pm2addau.h">;
  def PM4ADDAU_B  : RVPTernary_rrr<0b0101, 0b10, 0b101, "pm4addau.b">;

  def PMQ2ADD_H   : RVPBinary_rr<0b0110, 0b00, 0b101, "pmq2add.h">;
  def PMQR2ADD_H  : RVPBinary_rr<0b0110, 0b10, 0b101, "pmqr2add.h">;

  def PMQ2ADDA_H  : RVPTernary_rrr<0b0111, 0b00, 0b101, "pmq2adda.h">;
  def PMQR2ADDA_H : RVPTernary_rrr<0b0111, 0b10, 0b101, "pmqr2adda.h">;

  def PM2SUB_H    : RVPBinary_rr<0b1000, 0b00, 0b101, "pm2sub.h">;
  def PM2SADD_H   : RVPBinary_rr<0b1000, 0b10, 0b101, "pm2sadd.h">;

  def PM2SUBA_H   : RVPTernary_rrr<0b1001, 0b00, 0b101, "pm2suba.h">;

  def PM2SUB_HX   : RVPBinary_rr<0b1010, 0b00, 0b101, "pm2sub.hx">;
  def PM2SADD_HX  : RVPBinary_rr<0b1010, 0b10, 0b101, "pm2sadd.hx">;

  def PM2SUBA_HX  : RVPTernary_rrr<0b1011, 0b00, 0b101, "pm2suba.hx">;

  def PM2ADDSU_H  : RVPBinary_rr<0b1100, 0b00, 0b101, "pm2addsu.h">;
  def PM4ADDSU_B  : RVPBinary_rr<0b1100, 0b10, 0b101, "pm4addsu.b">;

  def PM2ADDASU_H : RVPBinary_rr<0b1101, 0b00, 0b101, "pm2addasu.h">;
  def PM4ADDASU_B : RVPBinary_rr<0b1101, 0b10, 0b101, "pm4addasu.b">;
} // Predicates = [HasStdExtP]
let Predicates = [HasStdExtP, IsRV32], DecoderNamespace = "RV32Only" in {
  def MQACC_H01  : RVPTernary_rrr<0b1111, 0b00, 0b101, "mqacc.h01">;
  def MQRACC_H01 : RVPTernary_rrr<0b1111, 0b10, 0b101, "mqracc.h01">;
} // // Predicates = [HasStdExtP, IsRV32], DecoderNamespace = "RV32Only"
let Predicates = [HasStdExtP, IsRV64] in {
  def PM2ADD_W      : RVPBinary_rr<0b0000, 0b01, 0b101, "pm2add.w">;
  def PM4ADD_H      : RVPBinary_rr<0b0000, 0b11, 0b101, "pm4add.h">;

  def PM2ADDA_W     : RVPTernary_rrr<0b0001, 0b01, 0b101, "pm2adda.w">;
  def PM4ADDA_H     : RVPTernary_rrr<0b0001, 0b11, 0b101, "pm4adda.h">;

  def PM2ADD_WX     : RVPBinary_rr<0b0010, 0b01, 0b101, "pm2add.wx">;

  def PM2ADDA_WX    : RVPTernary_rrr<0b0011, 0b01, 0b101, "pm2adda.wx">;

  def PM2ADDU_W     : RVPBinary_rr<0b0100, 0b01, 0b101, "pm2addu.w">;
  def PM4ADDU_H     : RVPBinary_rr<0b0100, 0b11, 0b101, "pm4addu.h">;

  def PM2ADDAU_W    : RVPTernary_rrr<0b0101, 0b01, 0b101, "pm2addau.w">;
  def PM4ADDAU_H    : RVPTernary_rrr<0b0101, 0b11, 0b101, "pm4addau.h">;

  def PMQ2ADD_W     : RVPBinary_rr<0b0110, 0b01, 0b101, "pmq2add.w">;
  def PMQR2ADD_W    : RVPBinary_rr<0b0110, 0b11, 0b101, "pmqr2add.w">;

  def PMQ2ADDA_W    : RVPTernary_rrr<0b0111, 0b01, 0b101, "pmq2adda.w">;
  def PMQR2ADDA_W   : RVPTernary_rrr<0b0111, 0b11, 0b101, "pmqr2adda.w">;

  def PM2SUB_W      : RVPBinary_rr<0b1000, 0b01, 0b101, "pm2sub.w">;

  def PM2SUBA_W     : RVPTernary_rrr<0b1001, 0b01, 0b101, "pm2suba.w">;

  def PM2SUB_WX     : RVPBinary_rr<0b1010, 0b01, 0b101, "pm2sub.wx">;

  def PM2SUBA_WX    : RVPBinary_rr<0b1011, 0b01, 0b101, "pm2suba.wx">;

  def PM2ADDSU_W    : RVPBinary_rr<0b1100, 0b01, 0b101, "pm2addsu.w">;
  def PM4ADDSU_H    : RVPBinary_rr<0b1100, 0b11, 0b101, "pm4addsu.h">;

  def PM2ADDASU_W   : RVPTernary_rrr<0b1101, 0b01, 0b101, "pm2addasu.w">;
  def PM4ADDASU_H   : RVPTernary_rrr<0b1101, 0b11, 0b101, "pm4addasu.h">;

  def PMQACC_W_H01  : RVPTernary_rrr<0b1111, 0b00, 0b101, "pmqacc.w.h01">;
  def MQACC_W01     : RVPTernary_rrr<0b1111, 0b01, 0b101, "mqacc.w01">;

  def PMQRACC_W_H01 : RVPTernary_rrr<0b1111, 0b10, 0b101, "pmqracc.w.h01">;
  def MQRACC_W01    : RVPTernary_rrr<0b1111, 0b11, 0b101, "mqracc.w01">;
} // Predicates = [HasStdExtP, IsRV64]

let Predicates = [HasStdExtP] in {
  def PAS_HX   : RVPBinary_rr<0b0000, 0b00, 0b110, "pas.hx">;
  def PSA_HX   : RVPBinary_rr<0b0000, 0b10, 0b110, "psa.hx">;

  def PSAS_HX  : RVPBinary_rr<0b0010, 0b00, 0b110, "psas.hx">;
  def PSSA_HX  : RVPBinary_rr<0b0010, 0b10, 0b110, "pssa.hx">;

  def PAAS_HX  : RVPBinary_rr<0b0011, 0b00, 0b110, "paas.hx">;
  def PASA_HX  : RVPBinary_rr<0b0011, 0b10, 0b110, "pasa.hx">;

  def PMSEQ_H  : RVPBinary_rr<0b1000, 0b00, 0b110, "pmseq.h">;
  def PMSEQ_B  : RVPBinary_rr<0b1000, 0b10, 0b110, "pmseq.b">;

  def PMSLT_H  : RVPBinary_rr<0b1010, 0b00, 0b110, "pmslt.h">;
  def PMSLT_B  : RVPBinary_rr<0b1010, 0b10, 0b110, "pmslt.b">;

  def PMSLTU_H : RVPBinary_rr<0b1011, 0b00, 0b110, "pmsltu.h">;
  def PMSLTU_B : RVPBinary_rr<0b1011, 0b10, 0b110, "pmsltu.b">;

  def PMIN_H   : RVPBinary_rr<0b1100, 0b00, 0b110, "pmin.h">;
  def PMIN_B   : RVPBinary_rr<0b1100, 0b10, 0b110, "pmin.b">;

  def PMINU_H  : RVPBinary_rr<0b1101, 0b00, 0b110, "pminu.h">;
  def PMINU_B  : RVPBinary_rr<0b1101, 0b10, 0b110, "pminu.b">;

  def PMAX_H   : RVPBinary_rr<0b1110, 0b00, 0b110, "pmax.h">;
  def PMAX_B   : RVPBinary_rr<0b1110, 0b10, 0b110, "pmax.b">;

  def PMAXU_H  : RVPBinary_rr<0b1111, 0b00, 0b110, "pmaxu.h">;
  def PMAXU_B  : RVPBinary_rr<0b1111, 0b10, 0b110, "pmaxu.b">;
} // Predicates = [HasStdExtP]
let Predicates = [HasStdExtP, IsRV32], DecoderNamespace = "RV32Only" in {
  def MSEQ  : RVPBinary_rr<0b1000, 0b01, 0b110, "mseq">;

  def MSLT  : RVPBinary_rr<0b1010, 0b01, 0b110, "mslt">;

  def MSLTU : RVPBinary_rr<0b1011, 0b01, 0b110, "msltu">;
} // Predicates = [HasStdExtP, IsRV32], DecoderNamespace = "RV32Only"
let Predicates = [HasStdExtP, IsRV64] in {
  def PAS_WX   : RVPBinary_rr<0b0000, 0b01, 0b110, "pas.wx">;
  def PSA_WX   : RVPBinary_rr<0b0000, 0b11, 0b110, "psa.wx">;

  def PSAS_WX  : RVPBinary_rr<0b0010, 0b01, 0b110, "psas.wx">;
  def PSSA_WX  : RVPBinary_rr<0b0010, 0b11, 0b110, "pssa.wx">;

  def PAAS_WX  : RVPBinary_rr<0b0011, 0b01, 0b110, "paas.wx">;
  def PASA_WX  : RVPBinary_rr<0b0011, 0b11, 0b110, "pasa.wx">;

  def PMSEQ_W  : RVPBinary_rr<0b1000, 0b01, 0b110, "pmseq.w">;

  def PMSLT_W  : RVPBinary_rr<0b1010, 0b01, 0b110, "pmslt.w">;

  def PMSLTU_W : RVPBinary_rr<0b1011, 0b01, 0b110, "pmsltu.w">;

  def PMIN_W   : RVPBinary_rr<0b1100, 0b01, 0b110, "pmin.w">;

  def PMINU_W  : RVPBinary_rr<0b1101, 0b01, 0b110, "pminu.w">;

  def PMAX_W   : RVPBinary_rr<0b1110, 0b01, 0b110, "pmax.w">;

  def PMAXU_W  : RVPBinary_rr<0b1111, 0b01, 0b110, "pmaxu.w">;
} // Predicates = [HasStdExtP, IsRV64]

let Predicates = [HasStdExtP] in {
  def PMULH_H      : RVPBinary_rr<0b0000, 0b00, 0b111, "pmulh.h">;
  def PMULHR_H     : RVPBinary_rr<0b0000, 0b10, 0b111, "pmulhr.h">;

  def PMHACC_H     : RVPTernary_rrr<0b0001, 0b00, 0b111, "pmhacc.h">;
  def PMHRACC_H    : RVPTernary_rrr<0b0001, 0b10, 0b111, "pmhracc.h">;

  def PMULHU_H     : RVPBinary_rr<0b0010, 0b00, 0b111, "pmulhu.h">;
  def PMULHRU_H    : RVPBinary_rr<0b0010, 0b10, 0b111, "pmulhru.h">;

  def PMHACCU_H    : RVPTernary_rrr<0b0011, 0b00, 0b111, "pmhaccu.h">;
  def PMHRACCU_H   : RVPTernary_rrr<0b0011, 0b10, 0b111, "pmhraccu.h">;

  def PMULH_H_B0    : RVPBinary_rr<0b0100, 0b00, 0b111, "pmulh.h.b0">;
  def PMULHSU_H_B0  : RVPBinary_rr<0b0100, 0b10, 0b111, "pmulhsu.h.b0">;

  def PMHACC_H_B0   : RVPTernary_rrr<0b0101, 0b00, 0b111, "pmhacc.h.b0">;
  def PMHACCSU_H_B0 : RVPTernary_rrr<0b0101, 0b10, 0b111, "pmhaccsu.h.b0">;

  def PMULH_H_B1    : RVPBinary_rr<0b0110, 0b00, 0b111, "pmulh.h.b1">;
  def PMULHSU_H_B1  : RVPBinary_rr<0b0110, 0b10, 0b111, "pmulhsu.h.b1">;

  def PMHACC_H_B1   : RVPTernary_rrr<0b0111, 0b00, 0b111, "pmhacc.h.b1">;
  def PMHACCSU_H_B1 : RVPTernary_rrr<0b0111, 0b10, 0b111, "pmhaccsu.h.b1">;

  def PMULHSU_H     : RVPBinary_rr<0b1000, 0b00, 0b111, "pmulhsu.h">;
  def PMULHRSU_H    : RVPBinary_rr<0b1000, 0b10, 0b111, "pmulhrsu.h">;

  def PMHACCSU_H    : RVPTernary_rrr<0b1001, 0b00, 0b111, "pmhaccsu.h">;
  def PMHRACCSU_H   : RVPTernary_rrr<0b1001, 0b10, 0b111, "pmhraccsu.h">;

  def PMULQ_H       : RVPBinary_rr<0b1010, 0b00, 0b111, "pmulq.h">;
  def PMULQR_H      : RVPBinary_rr<0b1010, 0b10, 0b111, "pmulqr.h">;
} // Predicates = [HasStdExtP]
let Predicates = [HasStdExtP, IsRV32], DecoderNamespace = "RV32Only" in {
  def MULHR      : RVPBinary_rr<0b0000, 0b11, 0b111, "mulhr">;

  def MHACC      : RVPTernary_rrr<0b0001, 0b01, 0b111, "mhacc">;
  def MHRACC     : RVPTernary_rrr<0b0001, 0b11, 0b111, "mhracc">;

  def MULHRU     : RVPBinary_rr<0b0010, 0b11, 0b111, "mulhru">;

  def MHACCU     : RVPTernary_rrr<0b0011, 0b01, 0b111, "mhaccu">;
  def MHRACCU    : RVPTernary_rrr<0b0011, 0b11, 0b111, "mhraccu">;

  def MULH_H0    : RVPBinary_rr<0b0100, 0b01, 0b111, "mulh.h0">;
  def MULHSU_H0  : RVPBinary_rr<0b0100, 0b11, 0b111, "mulhsu.h0">;

  def MHACC_H0   : RVPTernary_rrr<0b0101, 0b01, 0b111, "mhacc.h0">;
  def MHACCSU_H0 : RVPTernary_rrr<0b0101, 0b11, 0b111, "mhaccsu.h0">;

  def MULH_H1    : RVPBinary_rr<0b0110, 0b01, 0b111, "mulh.h1">;
  def MULHSU_H1  : RVPBinary_rr<0b0110, 0b11, 0b111, "mulhsu.h1">;

  def MHACC_H1   : RVPTernary_rrr<0b0111, 0b01, 0b111, "mhacc.h1">;
  def MHACCSU_H1 : RVPTernary_rrr<0b0111, 0b11, 0b111, "mhaccsu.h1">;

  def MULHRSU  : RVPBinary_rr<0b1000, 0b11, 0b111, "mulhrsu">;

  def MHACCSU    : RVPTernary_rrr<0b1001, 0b01, 0b111, "mhaccsu">;
  def MHRACCSU   : RVPTernary_rrr<0b1001, 0b11, 0b111, "mhraccsu">;

  def MULQ       : RVPBinary_rr<0b1010, 0b01, 0b111, "mulq">;
  def MULQR      : RVPBinary_rr<0b1010, 0b11, 0b111, "mulqr">;

  def MQACC_H00  : RVPTernary_rrr<0b1101, 0b00, 0b111, "mqacc.h00">;
  def MQRACC_H00 : RVPTernary_rrr<0b1101, 0b10, 0b111, "mqracc.h00">;

  def MQACC_H11  : RVPTernary_rrr<0b1111, 0b00, 0b111, "mqacc.h11">;
  def MQRACC_H11 : RVPTernary_rrr<0b1111, 0b10, 0b111, "mqracc.h11">;
} // Predicates = [HasStdExtP, IsRV32], DecoderNamespace = "RV32Only" in
let Predicates = [HasStdExtP, IsRV64] in {
  def PMULH_W       : RVPBinary_rr<0b0000, 0b01, 0b111, "pmulh.w">;
  def PMULHR_W      : RVPBinary_rr<0b0000, 0b11, 0b111, "pmulhr.w">;

  def PMHACC_W      : RVPTernary_rrr<0b0001, 0b01, 0b111, "pmhacc.w">;
  def PMHRACC_W     : RVPTernary_rrr<0b0001, 0b11, 0b111, "pmhracc.w">;

  def PMULHU_W      : RVPBinary_rr<0b0010, 0b01, 0b111, "pmulhu.w">;
  def PMULHRU_W     : RVPBinary_rr<0b0010, 0b11, 0b111, "pmulhru.w">;

  def PMHACCU_W     : RVPTernary_rrr<0b0011, 0b01, 0b111, "pmhaccu.w">;
  def PMHRACCU_W    : RVPTernary_rrr<0b0011, 0b11, 0b111, "pmhraccu.w">;

  def PMULH_W_H0    : RVPBinary_rr<0b0100, 0b01, 0b111, "pmulh.w.h0">;
  def PMULHSU_W_H0  : RVPBinary_rr<0b0100, 0b11, 0b111, "pmulhsu.w.h0">;

  def PMHACC_W_H0   : RVPTernary_rrr<0b0101, 0b01, 0b111, "pmhacc.w.h0">;
  def PMHACCSU_W_H0 : RVPTernary_rrr<0b0101, 0b11, 0b111, "pmhaccsu.w.h0">;

  def PMULH_W_H1    : RVPBinary_rr<0b0110, 0b01, 0b111, "pmulh.w.h1">;
  def PMULHSU_W_H1  : RVPBinary_rr<0b0110, 0b11, 0b111, "pmulhsu.w.h1">;

  def PMHACC_W_H1   : RVPTernary_rrr<0b0111, 0b01, 0b111, "pmhacc.w.h1">;
  def PMHACCSU_W_H1 : RVPTernary_rrr<0b0111, 0b11, 0b111, "pmhaccsu.w.h1">;

  def PMULHSU_W    : RVPBinary_rr<0b1000, 0b01, 0b111, "pmulhsu.w">;
  def PMULHRSU_W   : RVPBinary_rr<0b1000, 0b11, 0b111, "pmulhrsu.w">;

  def PMHACCSU_W   : RVPTernary_rrr<0b1001, 0b01, 0b111, "pmhaccsu.w">;
  def PMHRACCSU_W  : RVPTernary_rrr<0b1001, 0b11, 0b111, "pmhraccsu.w">;

  def PMULQ_W      : RVPBinary_rr<0b1010, 0b01, 0b111, "pmulq.w">;
  def PMULQR_W     : RVPBinary_rr<0b1010, 0b11, 0b111, "pmulqr.w">;

  def PMQACC_W_H00  : RVPTernary_rrr<0b1101, 0b00, 0b111, "pmqacc.w.h00">;
  def MQACC_W00     : RVPTernary_rrr<0b1101, 0b01, 0b111, "mqacc.w00">;
  def PMQRACC_W_H00 : RVPTernary_rrr<0b1101, 0b10, 0b111, "pmqracc.w.h00">;
  def MQRACC_W00    : RVPTernary_rrr<0b1101, 0b11, 0b111, "mqracc.w00">;

  def PMQACC_W_H11  : RVPTernary_rrr<0b1111, 0b00, 0b111, "pmqacc.w.h11">;
  def MQACC_W11     : RVPTernary_rrr<0b1111, 0b01, 0b111, "mqacc.w11">;
  def PMQRACC_W_H11 : RVPTernary_rrr<0b1111, 0b10, 0b111, "pmqracc.w.h11">;
  def MQRACC_W11    : RVPTernary_rrr<0b1111, 0b11, 0b111, "mqracc.w11">;
} // Predicates = [HasStdExtP, IsRV64]

let Predicates = [HasStdExtP, IsRV32] in {
  def PLI_DH : RVPPairLoadImm_i<0b0011000, (ins simm10:$imm10), "pli.dh",
                                "$rd, $imm10"> {
    bits<10> imm10;

    let Inst{24-16} = imm10{8-0};
    let Inst{15}    = imm10{9};
  }

  def PLI_DB : RVPPairLoadImm_i<0b0011010, (ins simm8_unsigned:$imm8), "pli.db",
                                "$rd, $imm8"> {
    bits<8> imm8;

    let Inst{24}    = 0b0;
    let Inst{23-16} = imm8;
    let Inst{15}    = 0b0;
  }

  def PLUI_DH : RVPPairLoadImm_i<0b0111000, (ins simm10_unsigned:$imm10),
                                 "plui.dh", "$rd, $imm10"> {
    bits<10> imm10;

    let Inst{24}    = imm10{0};
    let Inst{23-15} = imm10{9-1};
  }
}

//===----------------------------------------------------------------------===//
// Codegen patterns
//===----------------------------------------------------------------------===//

class RISCVTImmLeafOp<int bitsNum> :
  RISCVUImmOp<bitsNum>, TImmLeaf<XLenVT, "return isUInt<" # bitsNum # ">(Imm);">;
def timm3 : RISCVTImmLeafOp<3>;
def timm4 : RISCVTImmLeafOp<4>;
def timm5 : RISCVTImmLeafOp<5>;
def timm6 : RISCVTImmLeafOp<6>;

class PatGprTImm<SDPatternOperator OpNode, RVInst Inst, TImmLeaf ImmType,
                 ValueType vt = XLenVT>
    : Pat<(vt (OpNode (vt GPR:$rs1), ImmType:$imm)),
          (Inst GPR:$rs1, ImmType:$imm)>;

let Predicates = [HasStdExtP] in {
def : PatGprTImm<int_riscv_pslli_b,     PSLLI_B,  timm3>;
def : PatGprTImm<int_riscv_pslli_h,     PSLLI_H,  timm4>;
def : PatGprTImm<int_riscv_psslai_h,    PSSLAI_H, timm4>;
def : PatGprGpr<int_riscv_psll_bs,      PSLL_BS>;
def : PatGprGpr<int_riscv_psll_hs,      PSLL_HS>;
def : PatGprGpr<int_riscv_padd_bs,      PADD_BS>;
def : PatGprGpr<int_riscv_padd_hs,      PADD_HS>;
def : PatGprGpr<int_riscv_pssha_hs,     PSSHA_HS>;
def : PatGprGpr<int_riscv_psshar_hs,    PSSHAR_HS>;
def : PatGprTImm<int_riscv_psrli_b,     PSRLI_B,  timm3>;
def : PatGprTImm<int_riscv_psrli_h,     PSRLI_H,  timm4>;
def : PatGprTImm<int_riscv_pusati_h,    PUSATI_H, timm4>;
def : PatGprTImm<int_riscv_psrai_b,     PSRAI_B,  timm3>;
def : PatGprTImm<int_riscv_psrai_h,     PSRAI_H,  timm4>;
def : PatGprTImm<int_riscv_psrari_h,    PSRARI_H, timm4>;
def : PatGprTImm<int_riscv_psati_h,     PSATI_H,  timm4>;
def : PatGprGpr<int_riscv_psrl_bs,      PSRL_BS>;
def : PatGprGpr<int_riscv_psrl_hs,      PSRL_HS>;
def : PatGprGpr<int_riscv_predsum_bs,   PREDSUM_BS>;
def : PatGprGpr<int_riscv_predsum_hs,   PREDSUM_HS>;
def : PatGprGpr<int_riscv_predsumu_bs,  PREDSUMU_BS>;
def : PatGprGpr<int_riscv_predsumu_hs,  PREDSUMU_HS>;
def : PatGprGpr<int_riscv_psra_bs,      PSRA_BS>;
def : PatGprGpr<int_riscv_psra_hs,      PSRA_HS>;
def : PatGprGpr<int_riscv_padd_b,       PADD_B>;
def : PatGprGpr<int_riscv_padd_h,       PADD_H>;
def : PatGprGpr<int_riscv_psadd_b,      PSADD_B>;
def : PatGprGpr<int_riscv_psadd_h,      PSADD_H>;
def : PatGprGpr<int_riscv_paadd_b,      PAADD_B>;
def : PatGprGpr<int_riscv_paadd_h,      PAADD_H>;
def : PatGprGpr<int_riscv_psaddu_b,     PSADDU_B>;
def : PatGprGpr<int_riscv_psaddu_h,     PSADDU_H>;
def : PatGprGpr<int_riscv_paaddu_b,     PAADDU_B>;
def : PatGprGpr<int_riscv_paaddu_h,     PAADDU_H>;
def : PatGprGpr<int_riscv_psub_b,       PSUB_B>;
def : PatGprGpr<int_riscv_psub_h,       PSUB_H>;
def : PatGprGpr<int_riscv_pssub_b,      PSSUB_B>;
def : PatGprGpr<int_riscv_pssub_h,      PSSUB_H>;
def : PatGprGpr<int_riscv_pasub_b,      PASUB_B>;
def : PatGprGpr<int_riscv_pasub_h,      PASUB_H>;
def : PatGprGpr<int_riscv_pssubu_b,     PSSUBU_B>;
def : PatGprGpr<int_riscv_pssubu_h,     PSSUBU_H>;
def : PatGprGpr<int_riscv_pasubu_b,     PASUBU_B>;
def : PatGprGpr<int_riscv_pasubu_h,     PASUBU_H>;
def : PatGprGpr<int_riscv_pdif_b,       PDIF_B>;
def : PatGprGpr<int_riscv_pdif_h,       PDIF_H>;
def : PatGprGpr<int_riscv_pdifu_b,      PDIFU_B>;
def : PatGprGpr<int_riscv_pdifu_h,      PDIFU_H>;
def : PatGprGpr<int_riscv_slx,          SLX>;
def : PatGprGpr<int_riscv_pmul_h_b01,   PMUL_H_B01>;
def : PatGprGpr<int_riscv_pmulu_h_b01,  PMULU_H_B01>;
def : PatGprGpr<int_riscv_psh1add_h,    PSH1ADD_H>;
def : PatGprGpr<int_riscv_pssh1sadd_h,  PSSH1SADD_H>;
def : PatGprGpr<int_riscv_pmul_h_b00,   PMUL_H_B00>;
def : PatGprGpr<int_riscv_pmul_h_b11,   PMUL_H_B11>;
def : PatGprGpr<int_riscv_pmulu_h_b00,  PMULU_H_B00>;
def : PatGprGpr<int_riscv_pmulu_h_b11,  PMULU_H_B11>;
def : PatGprGpr<int_riscv_pmulsu_h_b00, PMULSU_H_B00>;
def : PatGprGpr<int_riscv_pmulsu_h_b11, PMULSU_H_B11>;
def : PatGprGpr<int_riscv_ppack_h,      PPACK_H>;
def : PatGprGpr<int_riscv_ppackbt_h,    PPACKBT_H>;
def : PatGprGpr<int_riscv_ppacktb_h,    PPACKTB_H>;
def : PatGprGpr<int_riscv_ppackt_h,     PPACKT_H>;
def : PatGprGpr<int_riscv_pas_hx,       PAS_HX>;
def : PatGprGpr<int_riscv_psa_hx,       PSA_HX>;
def : PatGprGpr<int_riscv_psas_hx,      PSAS_HX>;
def : PatGprGpr<int_riscv_pssa_hx,      PSSA_HX>;
def : PatGprGpr<int_riscv_paas_hx,      PAAS_HX>;
def : PatGprGpr<int_riscv_pasa_hx,      PASA_HX>;
def : PatGprGpr<int_riscv_pmseq_b,      PMSEQ_B>;
def : PatGprGpr<int_riscv_pmseq_h,      PMSEQ_H>;
def : PatGprGpr<int_riscv_pmslt_b,      PMSLT_B>;
def : PatGprGpr<int_riscv_pmslt_h,      PMSLT_H>;
def : PatGprGpr<int_riscv_pmsltu_b,     PMSLTU_B>;
def : PatGprGpr<int_riscv_pmsltu_h,     PMSLTU_H>;
def : PatGprGpr<int_riscv_pmin_b,       PMIN_B>;
def : PatGprGpr<int_riscv_pmin_h,       PMIN_H>;
def : PatGprGpr<int_riscv_pminu_b,      PMINU_B>;
def : PatGprGpr<int_riscv_pminu_h,      PMINU_H>;
def : PatGprGpr<int_riscv_pmax_b,       PMAX_B>;
def : PatGprGpr<int_riscv_pmax_h,       PMAX_H>;
def : PatGprGpr<int_riscv_pmaxu_b,      PMAXU_B>;
def : PatGprGpr<int_riscv_pmaxu_h,      PMAXU_H>;
def : PatGprGpr<int_riscv_pmulh_h,      PMULH_H>;
def : PatGprGpr<int_riscv_pmulh_h_b0,   PMULH_H_B0>;
def : PatGprGpr<int_riscv_pmulh_h_b1,   PMULH_H_B1>;
def : PatGprGpr<int_riscv_pmulhu_h,     PMULHU_H>;
def : PatGprGpr<int_riscv_pmulhr_h,     PMULHR_H>;
def : PatGprGpr<int_riscv_pmulhru_h,    PMULHRU_H>;
def : PatGprGpr<int_riscv_pmulhsu_h,    PMULHSU_H>;
def : PatGprGpr<int_riscv_pmulhsu_h_b0, PMULHSU_H_B0>;
def : PatGprGpr<int_riscv_pmulhsu_h_b1, PMULHSU_H_B1>;
def : PatGprGpr<int_riscv_pmulhrsu_h,   PMULHRSU_H>;
} // Predicates = [HasStdExtP]

let Predicates = [HasStdExtP, IsRV32] in {
def : PatGprTImm<int_riscv_sslai,    SSLAI, timm5>;
def : PatGprGpr<int_riscv_ssha,      SSHA>;
def : PatGprGpr<int_riscv_sshar,     SSHAR>;
def : PatGprTImm<int_riscv_usati,    USATI_RV32, timm5>;
def : PatGprTImm<int_riscv_srari,    SRARI_RV32, timm5>;
def : PatGprTImm<int_riscv_sati,     SATI_RV32,  timm5>;
def : PatGprGpr<int_riscv_sadd,      SADD>;
def : PatGprGpr<int_riscv_aadd,      AADD>;
def : PatGprGpr<int_riscv_saddu,     SADDU>;
def : PatGprGpr<int_riscv_aaddu,     AADDU>;
def : PatGprGpr<int_riscv_ssub,      SSUB>;
def : PatGprGpr<int_riscv_asub,      ASUB>;
def : PatGprGpr<int_riscv_ssubu,     SSUBU>;
def : PatGprGpr<int_riscv_asubu,     ASUBU>;
def : PatGprGpr<int_riscv_mul_h01,   MUL_H01>;
def : PatGprGpr<int_riscv_mulu_h01,  MULU_H01>;
def : PatGprGpr<int_riscv_ssh1sadd,  SSH1SADD>;
def : PatGprGpr<int_riscv_mul_h00,   MUL_H00>;
def : PatGprGpr<int_riscv_mul_h11,   MUL_H11>;
def : PatGprGpr<int_riscv_mulu_h00,  MULU_H00>;
def : PatGprGpr<int_riscv_mulu_h11,  MULU_H11>;
def : PatGprGpr<int_riscv_mulsu_h00, MULSU_H00>;
def : PatGprGpr<int_riscv_mulsu_h11, MULSU_H11>;
def : PatGprGpr<int_riscv_packbt,    PACKBT_RV32>;
def : PatGprGpr<int_riscv_packtb,    PACKTB_RV32>;
def : PatGprGpr<int_riscv_packt,     PACKT_RV32>;
def : PatGprGpr<int_riscv_mseq,      MSEQ>;
def : PatGprGpr<int_riscv_mslt,      MSLT>;
def : PatGprGpr<int_riscv_msltu,     MSLTU>;
def : PatGprGpr<int_riscv_mulh_h1,   MULH_H1>;
def : PatGprGpr<int_riscv_mulhr,     MULHR>;
def : PatGprGpr<int_riscv_mulhru,    MULHRU>;
def : PatGprGpr<int_riscv_mulh_h0,   MULH_H0>;
def : PatGprGpr<int_riscv_mulhsu_h0, MULHSU_H0>;
def : PatGprGpr<int_riscv_mulhsu_h1, MULHSU_H1>;
def : PatGprGpr<int_riscv_mulhrsu,   MULHRSU>;
} // Predicates = [HasStdExtP, IsRV32]

let Predicates = [HasStdExtP, IsRV64] in {
def : PatGprTImm<int_riscv_pslli_w,     PSLLI_W,  timm5>;
def : PatGprTImm<int_riscv_psslai_w,    PSSLAI_W, timm5>;
def : PatGprGpr<int_riscv_psll_ws,      PSLL_WS>;
def : PatGprGpr<int_riscv_padd_ws,      PADD_WS>;
def : PatGprGpr<int_riscv_pssha_ws,     PSSHA_WS>;
def : PatGprGpr<int_riscv_sha,          SHA>;
def : PatGprGpr<int_riscv_psshar_ws,    PSSHAR_WS>;
def : PatGprGpr<int_riscv_shar,         SHAR>;
def : PatGprTImm<int_riscv_psrli_w,     PSRLI_W,  timm5>;
def : PatGprTImm<int_riscv_pusati_w,    PUSATI_W, timm5>;
def : PatGprTImm<int_riscv_usati,       USATI_RV64, timm6>;
def : PatGprTImm<int_riscv_psrai_w,     PSRAI_W,    timm5>;
def : PatGprTImm<int_riscv_psrari_w,    PSRARI_W,   timm5>;
def : PatGprTImm<int_riscv_srari,       SRARI_RV64, timm6>;
def : PatGprTImm<int_riscv_psati_w,     PSATI_W,    timm5>;
def : PatGprTImm<int_riscv_sati,        SATI_RV64,  timm6>;
def : PatGprGpr<int_riscv_psrl_ws,      PSRL_WS>;
def : PatGprGpr<int_riscv_predsum_ws,   PREDSUM_WS>;
def : PatGprGpr<int_riscv_predsumu_ws,  PREDSUMU_WS>;
def : PatGprGpr<int_riscv_psra_ws,      PSRA_WS>;
def : PatGprGpr<int_riscv_padd_w,       PADD_W>;
def : PatGprGpr<int_riscv_psadd_w,      PSADD_W>;
def : PatGprGpr<int_riscv_paadd_w,      PAADD_W>;
def : PatGprGpr<int_riscv_psaddu_w,     PSADDU_W>;
def : PatGprGpr<int_riscv_paaddu_w,     PAADDU_W>;
def : PatGprGpr<int_riscv_psub_w,       PSUB_W>;
def : PatGprGpr<int_riscv_pssub_w,      PSSUB_W>;
def : PatGprGpr<int_riscv_pasub_w,      PASUB_W>;
def : PatGprGpr<int_riscv_pssubu_w,     PSSUBU_W>;
def : PatGprGpr<int_riscv_pasubu_w,     PASUBU_W>;
def : PatGprGpr<int_riscv_pmul_w_h01,   PMUL_W_H01>;
def : PatGprGpr<int_riscv_pmulu_w_h01,  PMULU_W_H01>;
def : PatGprGpr<int_riscv_mul_w01,      MUL_W01>;
def : PatGprGpr<int_riscv_mulu_w01,     MULU_W01>;
def : PatGprGpr<int_riscv_psh1add_w,    PSH1ADD_W>;
def : PatGprGpr<int_riscv_pssh1sadd_w,  PSSH1SADD_W>;
def : PatGprGpr<int_riscv_unzip8p,      UNZIP8P>;
def : PatGprGpr<int_riscv_unzip16p,     UNZIP16P>;
def : PatGprGpr<int_riscv_unzip8hp,     UNZIP8HP>;
def : PatGprGpr<int_riscv_unzip16hp,    UNZIP16HP>;
def : PatGprGpr<int_riscv_zip8p,        ZIP8P>;
def : PatGprGpr<int_riscv_zip16p,       ZIP16P>;
def : PatGprGpr<int_riscv_zip8hp,       ZIP8HP>;
def : PatGprGpr<int_riscv_zip16hp,      ZIP16HP>;
def : PatGprGpr<int_riscv_pmul_w_h00,   PMUL_W_H00>;
def : PatGprGpr<int_riscv_pmul_w_h11,   PMUL_W_H11>;
def : PatGprGpr<int_riscv_pmulu_w_h00,  PMULU_W_H00>;
def : PatGprGpr<int_riscv_pmulu_w_h11,  PMULU_W_H11>;
def : PatGprGpr<int_riscv_pmulsu_w_h00, PMULSU_W_H00>;
def : PatGprGpr<int_riscv_pmulsu_w_h11, PMULSU_W_H11>;
def : PatGprGpr<int_riscv_mul_w00,      MUL_W00>;
def : PatGprGpr<int_riscv_mul_w11,      MUL_W11>;
def : PatGprGpr<int_riscv_mulu_w00,     MULU_W00>;
def : PatGprGpr<int_riscv_mulu_w11,     MULU_W11>;
def : PatGprGpr<int_riscv_mulsu_w00,    MULSU_W00>;
def : PatGprGpr<int_riscv_mulsu_w11,    MULSU_W11>;
def : PatGprGpr<int_riscv_ppack_w,      PPACK_W>;
def : PatGprGpr<int_riscv_ppackbt_w,    PPACKBT_W>;
def : PatGprGpr<int_riscv_packbt,       PACKBT_RV64>;
def : PatGprGpr<int_riscv_ppacktb_w,    PPACKTB_W>;
def : PatGprGpr<int_riscv_packtb,       PACKTB_RV64>;
def : PatGprGpr<int_riscv_ppackt_w,     PPACKT_W>;
def : PatGprGpr<int_riscv_packt,        PACKT_RV64>;
def : PatGprGpr<int_riscv_pas_wx,       PAS_WX>;
def : PatGprGpr<int_riscv_psa_wx,       PSA_WX>;
def : PatGprGpr<int_riscv_psas_wx,      PSAS_WX>;
def : PatGprGpr<int_riscv_pssa_wx,      PSSA_WX>;
def : PatGprGpr<int_riscv_paas_wx,      PAAS_WX>;
def : PatGprGpr<int_riscv_pasa_wx,      PASA_WX>;
def : PatGprGpr<int_riscv_pmseq_w,      PMSEQ_W>;
def : PatGprGpr<int_riscv_pmslt_w,      PMSLT_W>;
def : PatGprGpr<int_riscv_pmsltu_w,     PMSLTU_W>;
def : PatGprGpr<int_riscv_pmin_w,       PMIN_W>;
def : PatGprGpr<int_riscv_pminu_w,      PMINU_W>;
def : PatGprGpr<int_riscv_pmax_w,       PMAX_W>;
def : PatGprGpr<int_riscv_pmaxu_w,      PMAXU_W>;
def : PatGprGpr<int_riscv_pmulh_w,      PMULH_W>;
def : PatGprGpr<int_riscv_pmulh_w_h0,   PMULH_W_H0>;
def : PatGprGpr<int_riscv_pmulh_w_h1,   PMULH_W_H1>;
def : PatGprGpr<int_riscv_pmulhu_w,     PMULHU_W>;
def : PatGprGpr<int_riscv_pmulhr_w,     PMULHR_W>;
def : PatGprGpr<int_riscv_pmulhru_w,    PMULHRU_W>;
def : PatGprGpr<int_riscv_pmulhsu_w,    PMULHSU_W>;
def : PatGprGpr<int_riscv_pmulhsu_w_h0, PMULHSU_W_H0>;
def : PatGprGpr<int_riscv_pmulhsu_w_h1, PMULHSU_W_H1>;
def : PatGprGpr<int_riscv_pmulhrsu_w,   PMULHRSU_W>;
} // Predicates = [HasStdExtP, IsRV64]
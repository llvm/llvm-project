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
  let OperandType = "OPERAND_SIMM10";
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

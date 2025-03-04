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

def RVPGPRPairRV32 : RegisterOperand<GPRPair> {
  let ParserMatchClass = GPRPairRV32Operand;
  let EncoderMethod = "getRVPGPRPair";
  let DecoderMethod = "decodeRVPGPRPair";
}

def simm10 : RISCVSImmLeafOp<10> {
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
class RVPUnary<bits<5> funct5, bits<7> wuimm,
               bits<3> funct3, RISCVOpcode opcode,
               string opcodestr>
    : RVInstIBase<funct3, opcode, (outs GPR:$rd), (ins GPR:$rs1),
                  opcodestr, "$rd, $rs1"> {
  let Inst{31-27} = funct5;
  let Inst{26-20} = wuimm;
}

let hasSideEffects = 0, mayLoad = 0, mayStore = 0 in
class RVPUnaryImm9<bits<7> funct7, string opcodestr>
    : RVInstIBase<0b010, OPC_OP_IMM_32, (outs GPR:$rd), (ins simm10:$simm10),
                  opcodestr, "$rd, $simm10"> {
  bits<10> simm10;

  let Inst{31-25} = funct7;
  let Inst{24-15} = simm10;
}

let hasSideEffects = 0, mayLoad = 0, mayStore = 0 in
class RVPUnaryImm9Rdp<bits<7> funct7, string opcodestr>
    : RVInstIBase<0b010, OPC_OP_IMM_32, (outs RVPGPRPairRV32:$rdp),
                  (ins simm10:$simm10),
                  opcodestr, "$rdp, $simm10"> {
  bits<10> simm10;
  bits<4> rdp;

  let Inst{31-25} = funct7;
  let Inst{24-15} = simm10;
  let Inst{11-8}  = rdp;
  let Inst{7}     = 0b0;
}

let hasSideEffects = 0, mayLoad = 0, mayStore = 0 in
class RVPUnaryImm8<bits<8> funct8, string opcodestr>
    : RVInstIBase<0b010, OPC_OP_IMM_32, (outs GPR:$rd), (ins uimm8:$uimm8),
                  opcodestr, "$rd, $uimm8"> {
  bits<8> uimm8;

  let Inst{31-24} = funct8;
  let Inst{23-16} = uimm8;
  let Inst{15}    = 0b0;
}

let hasSideEffects = 0, mayLoad = 0, mayStore = 0 in
class RVPUnaryImm8Rdp<bits<8> funct8, string opcodestr>
    : RVInstIBase<0b010, OPC_OP_IMM_32, (outs RVPGPRPairRV32:$rdp),
                  (ins uimm8:$uimm8), opcodestr, "$rdp, $uimm8"> {
  bits<8> uimm8;
  bits<4> rdp;

  let Inst{31-24} = funct8;
  let Inst{23-16} = uimm8;
  let Inst{15}    = 0b0;
  let Inst{11-8}  = rdp;
  let Inst{7}     = 0b0;
}

let hasSideEffects = 0, mayLoad = 0, mayStore = 0 in
class RVPUnaryWUF<bits<2> w, bits<5> uf, string opcodestr>
    : RVInstIBase<0b010, OPC_OP_IMM_32, (outs GPR:$rd), (ins GPR:$rs1),
                  opcodestr, "$rd, $rs1">  {
  let Inst{31-27} = 0b11100;
  let Inst{26-25} = w;
  let Inst{24-20} = uf;
}

let hasSideEffects = 0, mayLoad = 0, mayStore = 0 in
class RVPUnaryWUFRs1pRdp<bits<2> w, bits<5> uf, string opcodestr>
    : RVInstIBase<0b010, OPC_OP_IMM_32, (outs RVPGPRPairRV32:$rdp),
                  (ins RVPGPRPairRV32:$rs1p), opcodestr, "$rdp, $rs1p">  {
  bits<4> rs1p;
  bits<4> rdp;

  let Inst{31-27} = 0b01100;
  let Inst{26-25} = w;
  let Inst{24-20} = uf;
  let Inst{19-16} = rs1p;
  let Inst{15}    = 0b0;
  let Inst{11-8}  = rdp;
  let Inst{7}     = 0b0;
}

let hasSideEffects = 0, mayLoad = 0, mayStore = 0 in
class RVPUnaryF<bit bfr, bits<3> f, bit aft, bits<7> wuimm, string opcodestr,
                bits<3> funct3, dag outs, dag ins, string argstr>
    : RVInstIBase<funct3, OPC_OP_IMM_32, outs, ins, opcodestr, argstr>  {
  let Inst{31}    = bfr;
  let Inst{30-28} = f;
  let Inst{27}    = aft;
  let Inst{26-20} = wuimm;
}

class RVPUnary1F0<bits<3> f, bits<7> wuimm, string opcodestr>
    : RVPUnaryF<1, f, 0, wuimm, opcodestr, 0b100, (outs GPR:$rd),
                (ins GPR:$rs1), "$rd, $rs1">;

class RVPUnary0F0Rdp<bits<3> f, bits<7> wuimm, string opcodestr>
    : RVPUnaryF<0, f, 0, wuimm, opcodestr, 0b010, (outs RVPGPRPairRV32:$rdp),
               (ins GPR:$rs1), "$rdp, $rs1"> {
  bits<4> rdp;

  let Inst{11-8} = rdp;
  let Inst{7}    = 0b0;
}

class RVPUnary0F0Rs1p<bits<3> f, bits<7> wuimm, string opcodestr>
    : RVPUnaryF<0, f, 0, wuimm, opcodestr, 0b100, (outs GPR:$rd),
                (ins RVPGPRPairRV32:$rs1p), "$rd, $rs1p"> {
  bits<4> rs1p;

  let Inst{19-16} = rs1p;
  let Inst{15}    = 0b1;
}

class RVPUnary0F0Rs1pRdp<bits<3> f, bits<7> wuimm, string opcodestr,
                         bit aft = 0b0>
    : RVPUnaryF<0, f, 0, wuimm, opcodestr, 0b110, (outs RVPGPRPairRV32:$rdp),
                (ins RVPGPRPairRV32:$rs1p), "$rdp, $rs1p"> {
  bits<4> rs1p;
  bits<4> rdp;

  let Inst{19-16} = rs1p;
  let Inst{15}    = aft;
  let Inst{11-8}  = rdp;
  let Inst{7}     = 0b0;
}

let hasSideEffects = 0, mayLoad = 0, mayStore = 0 in
class RVPBinaryFW<bit bfr, bits<3> f, bit aft, bits<2> w, bits<3> funct3,
                  string opcodestr, RISCVOpcode Opcode = OPC_OP_32,
                  dag outs = (outs GPR:$rd), dag ins = (ins GPR:$rs1, GPR:$rs2),
                  string argstr = "$rd, $rs1, $rs2">
    : RVInstRBase<funct3, Opcode, outs, ins, opcodestr, argstr>  {
  let Inst{31}    = bfr;
  let Inst{30-28} = f;
  let Inst{27}    = aft;
  let Inst{26-25} = w;
}

class RVPBinary1F1W<bits<3> f, bits<2> w, bits<3> funct3, string opcodestr,
                    RISCVOpcode Opcode = OPC_OP_IMM_32>
    : RVPBinaryFW<1, f, 1, w, funct3, opcodestr, Opcode>;

class RVPBinary1F0W<bits<3> f, bits<2> w, bits<3> funct3, string opcodestr,
                    RISCVOpcode Opcode = OPC_OP_32>
    : RVPBinaryFW<1, f, 0, w, funct3, opcodestr, Opcode>;

class RVPBinary0F1WRdp<bits<3> f, bits<2> w, string opcodestr,
                       RISCVOpcode Opcode = OPC_OP_IMM_32>
    : RVPBinaryFW<0, f, 1, w, 0b010, opcodestr, Opcode, (outs RVPGPRPairRV32:$rdp),
                  (ins GPR:$rs1, GPR:$rs2), "$rdp, $rs1, $rs2"> {
  bits<4> rdp;

  let Inst{11-8} = rdp;
  let Inst{7}    = 0b0;
}

class RVPBinary0F1WRs1p<bits<3> f, bits<2> w, string opcodestr,
                    bit aft = 0b0, RISCVOpcode Opcode = OPC_OP_IMM_32>
    : RVPBinaryFW<0, f, 1, w, 0b100, opcodestr, Opcode, (outs GPR:$rd),
                  (ins RVPGPRPairRV32:$rs1p, GPR:$rs2), "$rd, $rs1p, $rs2"> {
   bits<4> rs1p;

  let Inst{19-16} = rs1p;
  let Inst{15}    = aft;
}

class RVPBinary0F1WRs1pRdp<bits<3> f, bits<2> w, string opcodestr,
                           bit aft = 0b0, RISCVOpcode Opcode = OPC_OP_IMM_32>
    : RVPBinaryFW<0, f, 1, w, 0b110, opcodestr, Opcode, (outs RVPGPRPairRV32:$rdp),
                  (ins RVPGPRPairRV32:$rs1p, GPR:$rs2), "$rdp, $rs1p, $rs2"> {
  bits<4> rs1p;
  bits<4> rdp;

  let Inst{19-16} = rs1p;
  let Inst{15}    = aft;
  let Inst{11-8}  = rdp;
  let Inst{7}     = 0b0;
}

let hasSideEffects = 0, mayLoad = 0, mayStore = 0 in
class RVPBinary1FWRs2pRs1pRdp<bits<4> f, bits<2> w, string opcodestr,
                              bit aft = 0b0, RISCVOpcode Opcode = OPC_OP_IMM_32>
    : RVInstRBase<0b110, Opcode, (outs RVPGPRPairRV32:$rdp),
                  (ins RVPGPRPairRV32:$rs1p, RVPGPRPairRV32:$rs2p),
                  opcodestr, "$rdp, $rs1p, $rs2p"> {
  bits<4> rs1p;
  bits<4> rs2p;
  bits<4> rdp;

  let Inst{31}    = 0b1;
  let Inst{30-27} = f;
  let Inst{26-25} = w;
  let Inst{24-21} = rs2p;
  let Inst{20}    = aft;
  let Inst{19-16} = rs1p;
  let Inst{15}    = aft;
  let Inst{11-8}  = rdp;
  let Inst{7}     = 0b0;
}

class RVPBinary1F0WRs2pRs1pRdp<bits<3> f, bits<2> w, string opcodestr,
                               bit bfr = 0b1, bit aft = 0b0,
                               RISCVOpcode Opcode = OPC_OP_IMM_32>
    : RVPBinaryFW<1, f, 0, w, 0b110, opcodestr, Opcode, (outs RVPGPRPairRV32:$rdp),
                  (ins RVPGPRPairRV32:$rs1p, RVPGPRPairRV32:$rs2p), "$rdp, $rs1p, $rs2p"> {
  bits<4> rs1p;
  bits<4> rs2p;
  bits<4> rdp;

  let Inst{24-21} = rs2p;
  let Inst{20}    = bfr;
  let Inst{19-16} = rs1p;
  let Inst{15}    = aft;
  let Inst{11-8}  = rdp;
  let Inst{7}     = 0b0;
}

let hasSideEffects = 0, mayLoad = 0, mayStore = 0 in
class RVPBinaryLongFW<bit bfr = 1, bits<4> f, bits<2> w, bits<3> funct3,
                      string opcodestr, dag outs, dag ins, string argstr>
    : RVInstRBase<funct3, OPC_OP_32, outs, ins,
                        opcodestr, argstr>  {
  let Inst{31}    = bfr;
  let Inst{30-27} = f;
  let Inst{26-25} = w;
}

class RVPBinary1LongFW<bits<4> f, bits<2> w, bits<3> funct3, string opcodestr>
    : RVPBinaryLongFW<1, f, w, funct3, opcodestr, (outs GPR:$rd),
                      (ins GPR:$rs1, GPR:$rs2), "$rd, $rs1, $rs2">;

class RVPBinary0LongFW<bits<4> f, bits<2> w, string opcodestr>
    : RVPBinaryLongFW<0, f, w, 0b010, opcodestr, (outs RVPGPRPairRV32:$rdp),
                      (ins GPR:$rs1, RVPGPRPairRV32:$rs2), "$rdp, $rs1, $rs2"> {
  bits<4> rdp;

  let Inst{11-8} = rdp;
  let Inst{7}    = 0b1;
}

//===----------------------------------------------------------------------===//
// Instructions
//===----------------------------------------------------------------------===//

let Predicates = [HasStdExtP] in {
def CLS    : RVPUnary<0b01100, 0b0000011, 0b001, OPC_OP_IMM, "cls">;
def ABS    : RVPUnary<0b01100, 0b0000111, 0b001, OPC_OP_IMM, "abs">;
} // Predicates = [HasStdExtP]
let Predicates = [HasStdExtP, IsRV32] in
def REV_RV32  : RVPUnary<0b01101, 0b0011111, 0b101, OPC_OP_IMM, "rev">;

let Predicates = [HasStdExtP, IsRV64] in {
def REV16 : RVPUnary<0b01101, 0b0110000, 0b101, OPC_OP_IMM, "rev16">;
def REV_RV64   : RVPUnary<0b01111, 0b0111111, 0b101, OPC_OP_IMM, "rev">;

def CLSW  : RVPUnary<0b01100, 0b0000011, 0b001, OPC_OP_IMM_32, "clsw">;
def ABSW  : RVPUnary<0b01100, 0b0000111, 0b001, OPC_OP_IMM_32, "absw">;
} // Predicates = [HasStdExtP, IsRV64]

let Predicates = [HasStdExtP] in {
def PSLLI_B  : RVPUnary<0b10000, 0b0001000, 0b010, OPC_OP_IMM_32, "pslli.b">;
def PSLLI_H  : RVPUnary<0b10000, 0b0010000, 0b010, OPC_OP_IMM_32, "pslli.h">;
def PSSLAI_H : RVPUnary<0b11010, 0b0010000, 0b010, OPC_OP_IMM_32, "psslai.h">;
} // Predicates = [HasStdExtP]
let DecoderNamespace = "RISCV32Only_",
    Predicates = [HasStdExtP, IsRV32] in
def SSLAI : RVPUnary<0b11010, 0b0100000, 0b010, OPC_OP_IMM_32, "sslai">;
let Predicates = [HasStdExtP, IsRV64] in {
def PSLLI_W  : RVPUnary<0b10000, 0b0100000, 0b010, OPC_OP_IMM_32, "pslli.w">;
def PSSLAI_W : RVPUnary<0b11010, 0b0100000, 0b010, OPC_OP_IMM_32, "psslai.w">;
} // Predicates = [HasStdExtP, IsRV64]

let Predicates = [HasStdExtP] in
def PLI_H : RVPUnaryImm9<0b1011000, "pli.h">;
let Predicates = [HasStdExtP, IsRV64] in
def PLI_W : RVPUnaryImm9<0b1011001, "pli.w">;
let Predicates = [HasStdExtP] in
def PLI_B : RVPUnaryImm8<0b10110100, "pli.b">;

let DecoderNamespace = "RISCV32Only_",
    Predicates = [HasStdExtP, IsRV32] in {
def PSEXT_H_B_RV32 : RVPUnaryWUF<0b00, 0b00100, "psext.h.b">;
def PSABS_H_RV32  : RVPUnaryWUF<0b00, 0b00111, "psabs.h">;
def PSABS_B_RV32  : RVPUnaryWUF<0b01, 0b00111, "psabs.b">;
} // Predicates = [HasStdExtP, IsRV32]

let Predicates = [HasStdExtP, IsRV64] in {
def PSEXT_H_B_RV64 : RVPUnaryWUF<0b00, 0b00100, "psext.h.b">;
def PSEXT_W_B      : RVPUnaryWUF<0b01, 0b00100, "psext.w.b">;
def PSEXT_W_H      : RVPUnaryWUF<0b01, 0b00101, "psext.w.h">;
def PSABS_H_RV64  : RVPUnaryWUF<0b00, 0b00111, "psabs.h">;
def PSABS_B_RV64  : RVPUnaryWUF<0b10, 0b00111, "psabs.b">;
} // Predicates = [HasStdExtP, IsRV64]

let Predicates = [HasStdExtP] in
def PLUI_H : RVPUnaryImm9<0b1111000, "plui.h">;
let Predicates = [HasStdExtP, IsRV64] in
def PLUI_W : RVPUnaryImm9<0b1111001, "plui.w">;

let Predicates = [HasStdExtP] in {
def PSLL_HS   : RVPBinary1F1W<0b000, 0b00, 0b010, "psll.hs">;
def PSLL_BS   : RVPBinary1F1W<0b000, 0b10, 0b010, "psll.bs">;
def PADD_HS   : RVPBinary1F1W<0b001, 0b00, 0b010, "padd.hs">;
def PADD_BS   : RVPBinary1F1W<0b001, 0b10, 0b010, "padd.bs">;
def PSSHA_HS  : RVPBinary1F1W<0b110, 0b00, 0b010, "pssha.hs">;
def PSSHAR_HS : RVPBinary1F1W<0b111, 0b00, 0b010, "psshar.hs">;
} // Predicates = [HasStdExtP]
let DecoderNamespace = "RISCV32Only_",
    Predicates = [HasStdExtP, IsRV32] in {
def SSHA        : RVPBinary1F1W<0b110, 0b01, 0b010, "ssha">;
def SSHAR       : RVPBinary1F1W<0b111, 0b01, 0b010, "sshar">;
} // Predicates = [HasStdExtP, IsRV32]

let Predicates = [HasStdExtP, IsRV64] in {
def PSLL_WS     : RVPBinary1F1W<0b000, 0b01, 0b010, "psll.ws">;
def PADD_WS     : RVPBinary1F1W<0b001, 0b01, 0b010, "padd.ws">;
def PSSHA_WS    : RVPBinary1F1W<0b110, 0b01, 0b010, "pssha.ws">;
def PSSHAR_WS   : RVPBinary1F1W<0b111, 0b01, 0b010, "psshar.ws">;
def SHA         : RVPBinary1F1W<0b110, 0b11, 0b010, "sha">;
def SHAR        : RVPBinary1F1W<0b111, 0b11, 0b010, "shar">;
} // Predicates = [HasStdExtP, IsRV64]

let Predicates = [HasStdExtP] in {
def PSRLI_B    : RVPUnary1F0<0b000, 0b0001000, "psrli.b">;
def PSRLI_H    : RVPUnary1F0<0b000, 0b0010000, "psrli.h">;
def PUSATI_H   : RVPUnary1F0<0b010, 0b0010000, "pusati.h">;
def PSRAI_B    : RVPUnary1F0<0b100, 0b0001000, "psrai.b">;
def PSRAI_H    : RVPUnary1F0<0b100, 0b0010000, "psrai.h">;
def PSRARI_H   : RVPUnary1F0<0b101, 0b0010000, "psrari.h">;
def PSATI_H    : RVPUnary1F0<0b110, 0b0010000, "psati.h">;
} // Predicates = [HasStdExtP]

let DecoderNamespace = "RISCV32Only_",
    Predicates = [HasStdExtP, IsRV32] in {
def USATI_RV32 : RVPUnary1F0<0b010, 0b0100000, "usati">;
def SRARI      : RVPUnary1F0<0b101, 0b0100000, "srari">;
def SATI       : RVPUnary1F0<0b110, 0b0100000, "sati">;
} // Predicates = [HasStdExtP, IsRV32]
let Predicates = [HasStdExtP, IsRV64] in {
def PSRLI_W    : RVPUnary1F0<0b000, 0b0100000, "psrli.w">;
def PUSATI_W   : RVPUnary1F0<0b010, 0b0100000, "pusati.w">;
def USATI_RV64 : RVPUnary1F0<0b010, 0b1000000, "usati">;
} // Predicates = [HasStdExtP, IsRV64]

let Predicates = [HasStdExtP] in {
def PSRL_HS     : RVPBinary1F1W<0b000, 0b00, 0b100, "psrl.hs">;
def PSRL_BS     : RVPBinary1F1W<0b000, 0b10, 0b100, "psrl.bs">;
def PREDSUM_HS  : RVPBinary1F1W<0b001, 0b00, 0b100, "predsum.hs">;
def PREDSUM_BS  : RVPBinary1F1W<0b001, 0b10, 0b100, "predsum.bs">;
def PREDSUMU_HS : RVPBinary1F1W<0b011, 0b00, 0b100, "predsumu.hs">;
def PREDSUMU_BS : RVPBinary1F1W<0b011, 0b10, 0b100, "predsumu.bs">;
def PSRA_HS     : RVPBinary1F1W<0b100, 0b00, 0b100, "psra.hs">;
def PSRA_BS     : RVPBinary1F1W<0b100, 0b10, 0b100, "psra.bs">;
} // Predicates = [HasStdExtP]

let Predicates = [HasStdExtP, IsRV64] in {
def PSRL_WS     : RVPBinary1F1W<0b000, 0b01, 0b100, "psrl.ws">;
def PREDSUM_WS  : RVPBinary1F1W<0b001, 0b01, 0b100, "predsum.ws">;
def PREDSUMU_WS : RVPBinary1F1W<0b011, 0b01, 0b100, "predsumu.ws">;
def PSRA_WS     : RVPBinary1F1W<0b100, 0b01, 0b100, "psra.ws">;
} // Predicates = [HasStdExtP, IsRV64]


let Predicates = [HasStdExtP] in {
def PADD_H   : RVPBinary1LongFW<0b0000, 0b00, 0b000, "padd.h">;
def PADD_B   : RVPBinary1LongFW<0b0000, 0b10, 0b000, "padd.b">;
def PSADD_H  : RVPBinary1LongFW<0b0010, 0b00, 0b000, "psadd.h">;
def PSADD_B  : RVPBinary1LongFW<0b0010, 0b10, 0b000, "psadd.b">;
def PAADD_H  : RVPBinary1LongFW<0b0011, 0b00, 0b000, "paadd.h">;
def PAADD_B  : RVPBinary1LongFW<0b0011, 0b10, 0b000, "paadd.b">;

def PSADDU_H : RVPBinary1LongFW<0b0110, 0b00, 0b000, "psaddu.h">;
def PSADDU_B : RVPBinary1LongFW<0b0110, 0b10, 0b000, "psaddu.b">;
def PAADDU_H : RVPBinary1LongFW<0b0111, 0b00, 0b000, "paaddu.h">;
def PAADDU_B : RVPBinary1LongFW<0b0111, 0b10, 0b000, "paaddu.b">;

def PSUB_H   : RVPBinary1LongFW<0b1000, 0b00, 0b000, "psub.h">;
def PSUB_B   : RVPBinary1LongFW<0b1000, 0b10, 0b000, "psub.b">;
def PDIF_H   : RVPBinary1LongFW<0b1001, 0b00, 0b000, "pdif.h">;
def PDIF_B   : RVPBinary1LongFW<0b1001, 0b10, 0b000, "pdif.b">;
def PSSUB_H  : RVPBinary1LongFW<0b1010, 0b00, 0b000, "pssub.h">;
def PSSUB_B  : RVPBinary1LongFW<0b1010, 0b10, 0b000, "pssub.b">;
def PASUB_H  : RVPBinary1LongFW<0b1011, 0b00, 0b000, "pasub.h">;
def PASUB_B  : RVPBinary1LongFW<0b1011, 0b10, 0b000, "pasub.b">;

def PDIFU_H  : RVPBinary1LongFW<0b1101, 0b00, 0b000, "pdifu.h">;
def PDIFU_B  : RVPBinary1LongFW<0b1101, 0b10, 0b000, "pdifu.b">;
def PSSUBU_H : RVPBinary1LongFW<0b1110, 0b00, 0b000, "pssubu.h">;
def PSSUBU_B : RVPBinary1LongFW<0b1110, 0b10, 0b000, "pssubu.b">;
def PASUBU_H : RVPBinary1LongFW<0b1111, 0b00, 0b000, "pasubu.h">;
def PASUBU_B : RVPBinary1LongFW<0b1111, 0b10, 0b000, "pasubu.b">;
} // Predicates = [HasStdExtP]

let DecoderNamespace = "RISCV32Only_",
    Predicates = [HasStdExtP, IsRV32] in {
def SADD  : RVPBinary1LongFW<0b0010, 0b01, 0b000, "sadd">;
def AADD  : RVPBinary1LongFW<0b0011, 0b01, 0b000, "aadd">;

def SADDU : RVPBinary1LongFW<0b0110, 0b01, 0b000, "saddu">;
def AADDU : RVPBinary1LongFW<0b0111, 0b01, 0b000, "aaddu">;

def SSUB  : RVPBinary1LongFW<0b1010, 0b01, 0b000, "ssub">;
def ASUB  : RVPBinary1LongFW<0b1011, 0b01, 0b000, "asub">;

def SSUBU : RVPBinary1LongFW<0b1110, 0b01, 0b000, "ssubu">;
def ASUBU : RVPBinary1LongFW<0b1111, 0b01, 0b000, "asubu">;
} // Predicates = [HasStdExtP, IsRV32]

let Predicates = [HasStdExtP, IsRV64] in {
def PADD_W   : RVPBinary1LongFW<0b0000, 0b01, 0b000, "padd.w">;
def PSADD_W  : RVPBinary1LongFW<0b0010, 0b01, 0b000, "psadd.w">;
def PAADD_W  : RVPBinary1LongFW<0b0011, 0b01, 0b000, "paadd.w">;

def PSADDU_W : RVPBinary1LongFW<0b0110, 0b01, 0b000, "psaddu.w">;
def PAADDU_W : RVPBinary1LongFW<0b0111, 0b01, 0b000, "paaddu.w">;

def PSUB_W   : RVPBinary1LongFW<0b1000, 0b01, 0b000, "psub.w">;
def PSSUB_W  : RVPBinary1LongFW<0b1010, 0b01, 0b000, "pssub.w">;
def PASUB_W  : RVPBinary1LongFW<0b1011, 0b01, 0b000, "pasub.w">;

def PSSUBU_W : RVPBinary1LongFW<0b1110, 0b01, 0b000, "pssubu.w">;
def PASUBU_W : RVPBinary1LongFW<0b1111, 0b01, 0b000, "pasubu.w">;
} // Predicates = [HasStdExtP, IsRV64]


let Predicates = [HasStdExtP] in {
def SLX         : RVPBinary1LongFW<0b0001, 0b11, 0b001, "slx">;
def PMUL_H_B01  : RVPBinary1LongFW<0b0010, 0b00, 0b001, "pmul.h.b01">;

def MVM         : RVPBinary1LongFW<0b0101, 0b00, 0b001, "mvm">;
def MVMN        : RVPBinary1LongFW<0b0101, 0b01, 0b001, "mvmn">;
def MERGE       : RVPBinary1LongFW<0b0101, 0b10, 0b001, "merge">;
def SRX         : RVPBinary1LongFW<0b0101, 0b11, 0b001, "srx">;
def PMULU_H_B01 : RVPBinary1LongFW<0b0110, 0b00, 0b001, "pmulu.h.b01">;
def PDIFSUMU_B  : RVPBinary1LongFW<0b0110, 0b10, 0b001, "pdifsumu.b">;
def PDIFSUMAU_B : RVPBinary1LongFW<0b0111, 0b10, 0b001, "pdifsumau.b">;
} // Predicates = [HasStdExtP]

let DecoderNamespace = "RISCV32Only_",
    Predicates = [HasStdExtP, IsRV32] in {
def MUL_H01   : RVPBinary1LongFW<0b0010, 0b01, 0b001, "mul.h01">;
def MACC_H01  : RVPBinary1LongFW<0b0011, 0b01, 0b001, "macc.h01">;

def MULU_H01  : RVPBinary1LongFW<0b0110, 0b01, 0b001, "mulu.h01">;
def MACCU_H01 : RVPBinary1LongFW<0b0111, 0b01, 0b001, "maccu.h01">;
} // Predicates = [HasStdExtP, IsRV32]

let Predicates = [HasStdExtP, IsRV64] in {
def PMUL_W_H01   : RVPBinary1LongFW<0b0010, 0b01, 0b001, "pmul.w.h01">;
def MUL_W01      : RVPBinary1LongFW<0b0010, 0b11, 0b001, "mul.w01">;
def PMACC_W_H01  : RVPBinary1LongFW<0b0011, 0b01, 0b001, "pmacc.w.h01">;
def MACC_W01     : RVPBinary1LongFW<0b0011, 0b11, 0b001, "macc.w01">;

def PMULU_W_H01  : RVPBinary1LongFW<0b0110, 0b01, 0b001, "pmulu.w.h01">;
def MULU_W01     : RVPBinary1LongFW<0b0110, 0b11, 0b001, "mulu.w01">;
def PMACCU_W_H01 : RVPBinary1LongFW<0b0111, 0b01, 0b001, "pmaccu.w.h01">;
def MACCU_W01    : RVPBinary1LongFW<0b0111, 0b11, 0b001, "maccu.w01">;
} // Predicates = [HasStdExtP, IsRV64]


let Predicates = [HasStdExtP] in {
def PSH1ADD_H   : RVPBinary1F0W<0b010, 0b00, 0b010, "psh1add.h">;
def PSSH1SADD_H : RVPBinary1F0W<0b011, 0b00, 0b010, "pssh1sadd.h">;
} // Predicates = [HasStdExtP]

let DecoderNamespace = "RISCV32Only_",
    Predicates = [HasStdExtP, IsRV32] in {
def SSH1SADD : RVPBinary1F0W<0b010, 0b01, 0b010, "ssh1sadd">;
} // Predicates = [HasStdExtP, IsRV32]
let Predicates = [HasStdExtP, IsRV64] in {
def PSH1ADD_W   : RVPBinary1F0W<0b010, 0b01, 0b010, "psh1add.w">;
def PSSH1SADD_W : RVPBinary1F0W<0b011, 0b01, 0b010, "pssh1sadd.w">;

def UNZIP8P     : RVPBinary1F0W<0b110, 0b00, 0b010, "unzip8p">;
def UNZIP16P    : RVPBinary1F0W<0b110, 0b01, 0b010, "unzip16p">;
def UNZIP8HP    : RVPBinary1F0W<0b110, 0b10, 0b010, "unzip8hp">;
def UNZIP16HP   : RVPBinary1F0W<0b110, 0b11, 0b010, "unzip16hp">;
def ZIP8P       : RVPBinary1F0W<0b111, 0b00, 0b010, "zip8p">;
def ZIP16P      : RVPBinary1F0W<0b111, 0b01, 0b010, "zip16p">;
def ZIP8HP      : RVPBinary1F0W<0b111, 0b10, 0b010, "zip8hp">;
def ZIP16HP     : RVPBinary1F0W<0b111, 0b11, 0b010, "zip16hp">;
} // Predicates = [HasStdExtP, IsRV64]


let Predicates = [HasStdExtP] in {
def PMUL_H_B00   : RVPBinary1LongFW<0b0000, 0b00, 0b011, "pmul.h.b00">;
def PMUL_H_B11   : RVPBinary1LongFW<0b0010, 0b00, 0b011, "pmul.h.b11">;

def PMULU_H_B00  : RVPBinary1LongFW<0b0100, 0b00, 0b011, "pmulu.h.b00">;
def PMULU_H_B11  : RVPBinary1LongFW<0b0110, 0b00, 0b011, "pmulu.h.b11">;

def PMULSU_H_B00 : RVPBinary1LongFW<0b1100, 0b00, 0b011, "pmulsu.h.b00">;
def PMULSU_H_B11 : RVPBinary1LongFW<0b1110, 0b00, 0b011, "pmulsu.h.b11">;
} // Predicates = [HasStdExtP]
let DecoderNamespace = "RISCV32Only_",
    Predicates = [HasStdExtP, IsRV32] in {
def MUL_H00    : RVPBinary1LongFW<0b0000, 0b01, 0b011, "mul.h00">;
def MACC_H00   : RVPBinary1LongFW<0b0001, 0b01, 0b011, "macc.h00">;
def MUL_H11    : RVPBinary1LongFW<0b0010, 0b01, 0b011, "mul.h11">;
def MACC_H11   : RVPBinary1LongFW<0b0011, 0b01, 0b011, "macc.h11">;

def MULU_H00   : RVPBinary1LongFW<0b0100, 0b01, 0b011, "mulu.h00">;
def MACCU_H00  : RVPBinary1LongFW<0b0101, 0b01, 0b011, "maccu.h00">;
def MULU_H11   : RVPBinary1LongFW<0b0110, 0b01, 0b011, "mulu.h11">;
def MACCU_H11  : RVPBinary1LongFW<0b0111, 0b01, 0b011, "maccu.h11">;

def MULSU_H00  : RVPBinary1LongFW<0b1100, 0b01, 0b011, "mulsu.h00">;
def MACCSU_H00 : RVPBinary1LongFW<0b1101, 0b01, 0b011, "maccsu.h00">;
def MULSU_H11  : RVPBinary1LongFW<0b1110, 0b01, 0b011, "mulsu.h11">;
def MACCSU_H11 : RVPBinary1LongFW<0b1111, 0b01, 0b011, "maccsu.h11">;
} // Predicates = [HasStdExtP, IsRV32]

let Predicates = [HasStdExtP, IsRV64] in {
def PMUL_W_H00    : RVPBinary1LongFW<0b0000, 0b01, 0b011, "pmul.w.h00">;
def MUL_W00       : RVPBinary1LongFW<0b0000, 0b11, 0b011, "mul.w00">;
def PMACC_W_H00   : RVPBinary1LongFW<0b0001, 0b01, 0b011, "pmacc.w.h00">;
def MACC_W00      : RVPBinary1LongFW<0b0001, 0b11, 0b011, "macc.w00">;
def PMUL_W_H11    : RVPBinary1LongFW<0b0010, 0b01, 0b011, "pmul.w.h11">;
def MUL_W11       : RVPBinary1LongFW<0b0010, 0b11, 0b011, "mul.w11">;
def PMACC_W_H11   : RVPBinary1LongFW<0b0011, 0b01, 0b011, "pmacc.w.h11">;
def MACC_W11      : RVPBinary1LongFW<0b0011, 0b11, 0b011, "macc.w11">;

def PMULU_W_H00   : RVPBinary1LongFW<0b0100, 0b01, 0b011, "pmulu.w.h00">;
def MULU_W00      : RVPBinary1LongFW<0b0100, 0b11, 0b011, "mulu.w00">;
def PMACCU_W_H00  : RVPBinary1LongFW<0b0101, 0b01, 0b011, "pmaccu.w.h00">;
def MACCU_W00     : RVPBinary1LongFW<0b0101, 0b11, 0b011, "maccu.w00">;
def PMULU_W_H11   : RVPBinary1LongFW<0b0110, 0b01, 0b011, "pmulu.w.h11">;
def MULU_W11      : RVPBinary1LongFW<0b0110, 0b11, 0b011, "mulu.w11">;
def PMACCU_W_H11  : RVPBinary1LongFW<0b0111, 0b01, 0b011, "pmaccu.w.h11">;
def MACCU_W11     : RVPBinary1LongFW<0b0111, 0b11, 0b011, "maccu.w11">;

def PMULSU_W_H00  : RVPBinary1LongFW<0b1100, 0b01, 0b011, "pmulsu.w.h00">;
def MULSU_W00     : RVPBinary1LongFW<0b1100, 0b11, 0b011, "mulsu.w00">;
def PMACCSU_W_H00 : RVPBinary1LongFW<0b1101, 0b01, 0b011, "pmaccsu.w.h00">;
def MACCSU_W00    : RVPBinary1LongFW<0b1101, 0b11, 0b011, "maccsu.w00">;
def PMULSU_W_H11  : RVPBinary1LongFW<0b1110, 0b01, 0b011, "pmulsu.w.h11">;
def MULSU_W11     : RVPBinary1LongFW<0b1110, 0b11, 0b011, "mulsu.w11">;
def PMACCSU_W_H11 : RVPBinary1LongFW<0b1111, 0b01, 0b011, "pmaccsu.w.h11">;
def MACCSU_W11    : RVPBinary1LongFW<0b1111, 0b11, 0b011, "maccsu.w11">;
} // Predicates = [HasStdExtP, IsRV64]


let Predicates = [HasStdExtP] in {
def PPACK_H   : RVPBinary1F0W<0b000, 0b00, 0b100, "ppack.h">;
def PPACKBT_H : RVPBinary1F0W<0b001, 0b00, 0b100, "ppackbt.h">;
def PPACKTB_H : RVPBinary1F0W<0b010, 0b00, 0b100, "ppacktb.h">;
def PPACKT_H  : RVPBinary1F0W<0b011, 0b00, 0b100, "ppackt.h">;
} // Predicates = [HasStdExtP]
let DecoderNamespace = "RISCV32Only_",
    Predicates = [HasStdExtP, IsRV32] in {
def PACKBT : RVPBinary1F0W<0b001, 0b01, 0b100, "packbt">;
def PACKTB : RVPBinary1F0W<0b010, 0b01, 0b100, "packtb">;
def PACKT  : RVPBinary1F0W<0b011, 0b01, 0b100, "packt">;
} // Predicates = [HasStdExtP, IsRV32]

let Predicates = [HasStdExtP, IsRV64] in {
def PPACK_W   : RVPBinary1F0W<0b000, 0b01, 0b100, "ppack.w">;
def PPACKBT_W : RVPBinary1F0W<0b001, 0b01, 0b100, "ppackbt.w">;
def PPACKBT   : RVPBinary1F0W<0b001, 0b11, 0b100, "ppackbt">;
def PPACKTB_W : RVPBinary1F0W<0b010, 0b01, 0b100, "ppacktb.w">;
def PPACKTB   : RVPBinary1F0W<0b010, 0b11, 0b100, "ppacktb">;
def PPACKT_W  : RVPBinary1F0W<0b011, 0b01, 0b100, "ppackt.w">;
def PPACKT    : RVPBinary1F0W<0b011, 0b11, 0b100, "ppackt">;
} // Predicates = [HasStdExtP, IsRV64]


let Predicates = [HasStdExtP] in {
def PM2ADD_H    : RVPBinary1LongFW<0b0000, 0b00, 0b101, "pm2add.h">;
def PM4ADD_B    : RVPBinary1LongFW<0b0000, 0b10, 0b101, "pm4add.b">;
def PM2ADDA_H   : RVPBinary1LongFW<0b0001, 0b00, 0b101, "pm2adda.h">;
def PM4ADDA_B   : RVPBinary1LongFW<0b0001, 0b10, 0b101, "pm4adda.b">;
def PM2ADD_HX   : RVPBinary1LongFW<0b0010, 0b00, 0b101, "pm2add.hx">;
def PM2ADDA_HX  : RVPBinary1LongFW<0b0011, 0b00, 0b101, "pm2adda.hx">;

def PM2ADDU_H   : RVPBinary1LongFW<0b0100, 0b00, 0b101, "pm2addu.h">;
def PM4ADDU_B   : RVPBinary1LongFW<0b0100, 0b10, 0b101, "pm4addu.b">;
def PM2ADDAU_H  : RVPBinary1LongFW<0b0101, 0b00, 0b101, "pm2addau.h">;
def PM4ADDAU_B  : RVPBinary1LongFW<0b0101, 0b10, 0b101, "pm4addau.b">;
def PMQ2ADD_H   : RVPBinary1LongFW<0b0110, 0b00, 0b101, "pmq2add.h">;
def PMQR2ADD_H  : RVPBinary1LongFW<0b0110, 0b10, 0b101, "pmqr2add.h">;
def PMQ2ADDA_H  : RVPBinary1LongFW<0b0111, 0b00, 0b101, "pmq2adda.h">;
def PMQR2ADDA_H : RVPBinary1LongFW<0b0111, 0b10, 0b101, "pmqr2adda.h">;

def PM2SUB_H    : RVPBinary1LongFW<0b1000, 0b00, 0b101, "pm2sub.h">;
def PM2SADD_H   : RVPBinary1LongFW<0b1000, 0b10, 0b101, "pm2sadd.h">;
def PM2SUBA_H   : RVPBinary1LongFW<0b1001, 0b00, 0b101, "pm2suba.h">;
def PM2SUB_HX   : RVPBinary1LongFW<0b1010, 0b00, 0b101, "pm2sub.hx">;
def PM2SADD_HX  : RVPBinary1LongFW<0b1010, 0b10, 0b101, "pm2sadd.hx">;
def PM2SUBA_HX  : RVPBinary1LongFW<0b1011, 0b00, 0b101, "pm2suba.hx">;

def PM2ADDSU_H  : RVPBinary1LongFW<0b1100, 0b00, 0b101, "pm2addsu.h">;
def PM4ADDSU_B  : RVPBinary1LongFW<0b1100, 0b10, 0b101, "pm4addsu.b">;
def PM2ADDASU_H : RVPBinary1LongFW<0b1101, 0b00, 0b101, "pm2addasu.h">;
def PM4ADDASU_B : RVPBinary1LongFW<0b1101, 0b10, 0b101, "pm4addasu.b">;
} // Predicates = [HasStdExtP]
let DecoderNamespace = "RISCV32Only_",
    Predicates = [HasStdExtP, IsRV32] in {
def MQACC_H01  : RVPBinary1LongFW<0b1111, 0b00, 0b101, "mqacc.h01">;
def MQRACC_H01 : RVPBinary1LongFW<0b1111, 0b10, 0b101, "mqracc.h01">;
} // Predicates = [HasStdExtP, IsRV32]

let Predicates = [HasStdExtP, IsRV64] in {
def PM2ADD_W    : RVPBinary1LongFW<0b0000, 0b01, 0b101, "pm2add.w">;
def PM4ADD_H    : RVPBinary1LongFW<0b0000, 0b11, 0b101, "pm4add.h">;
def PM2ADDA_W   : RVPBinary1LongFW<0b0001, 0b01, 0b101, "pm2adda.w">;
def PM4ADDA_H   : RVPBinary1LongFW<0b0001, 0b11, 0b101, "pm4adda.h">;
def PM2ADD_WX   : RVPBinary1LongFW<0b0010, 0b01, 0b101, "pm2add.wx">;
def PM2ADDA_WX  : RVPBinary1LongFW<0b0011, 0b01, 0b101, "pm2adda.wx">;

def PM2ADDU_W   : RVPBinary1LongFW<0b0100, 0b01, 0b101, "pm2addu.w">;
def PM4ADDU_H   : RVPBinary1LongFW<0b0100, 0b11, 0b101, "pm4addu.h">;
def PM2ADDAU_W  : RVPBinary1LongFW<0b0101, 0b01, 0b101, "pm2addau.w">;
def PM4ADDAU_H  : RVPBinary1LongFW<0b0101, 0b11, 0b101, "pm4addau.h">;
def PMQ2ADD_W   : RVPBinary1LongFW<0b0110, 0b01, 0b101, "pmq2add.w">;
def PMQR2ADD_W  : RVPBinary1LongFW<0b0110, 0b11, 0b101, "pmqr2add.w">;
def PMQ2ADDA_W  : RVPBinary1LongFW<0b0111, 0b01, 0b101, "pmq2adda.w">;
def PMQR2ADDA_W : RVPBinary1LongFW<0b0111, 0b11, 0b101, "pmqr2adda.w">;

def PM2SUB_W    : RVPBinary1LongFW<0b1000, 0b01, 0b101, "pm2sub.w">;
def PM2SUBA_W   : RVPBinary1LongFW<0b1001, 0b01, 0b101, "pm2suba.w">;
def PM2SUB_WX   : RVPBinary1LongFW<0b1010, 0b01, 0b101, "pm2sub.wx">;
def PM2SUBA_WX  : RVPBinary1LongFW<0b1011, 0b01, 0b101, "pm2suba.wx">;

def PM2ADDSU_W  : RVPBinary1LongFW<0b1100, 0b01, 0b101, "pm2addsu.w">;
def PM4ADDSU_H  : RVPBinary1LongFW<0b1100, 0b11, 0b101, "pm4addsu.h">;
def PM2ADDASU_W : RVPBinary1LongFW<0b1101, 0b01, 0b101, "pm2addasu.w">;
def PM4ADDASU_H : RVPBinary1LongFW<0b1101, 0b11, 0b101, "pm4addasu.h">;

def PMQACC_W_H01  : RVPBinary1LongFW<0b1111, 0b00, 0b101, "pmqacc.w.h01">;
def MQACC_W01     : RVPBinary1LongFW<0b1111, 0b01, 0b101, "mqacc.w01">;
def PMQRACC_W_H01 : RVPBinary1LongFW<0b1111, 0b10, 0b101, "pmqracc.w.h01">;
def MQRACC_W01    : RVPBinary1LongFW<0b1111, 0b11, 0b101, "mqracc.w01">;
} // Predicates = [HasStdExtP, IsRV64]

let Predicates = [HasStdExtP] in {
def PAS_HX   : RVPBinary1LongFW<0b0000, 0b00, 0b110, "pas.hx">;
def PSA_HX   : RVPBinary1LongFW<0b0000, 0b10, 0b110, "psa.hx">;
def PSAS_HX  : RVPBinary1LongFW<0b0010, 0b00, 0b110, "psas.hx">;
def PSSA_HX  : RVPBinary1LongFW<0b0010, 0b10, 0b110, "pssa.hx">;

def PMSEQ_H  : RVPBinary1LongFW<0b1000, 0b00, 0b110, "pmseq.h">;
def PMSEQ_B  : RVPBinary1LongFW<0b1000, 0b10, 0b110, "pmseq.b">;
def PMSLT_H  : RVPBinary1LongFW<0b1010, 0b00, 0b110, "pmslt.h">;
def PMSLT_B  : RVPBinary1LongFW<0b1010, 0b10, 0b110, "pmslt.b">;
def PMSLTU_H : RVPBinary1LongFW<0b1011, 0b00, 0b110, "pmsltu.h">;
def PMSLTU_B : RVPBinary1LongFW<0b1011, 0b10, 0b110, "pmsltu.b">;

def PMIN_H   : RVPBinary1LongFW<0b1100, 0b00, 0b110, "pmin.h">;
def PMIN_B   : RVPBinary1LongFW<0b1100, 0b10, 0b110, "pmin.b">;
def PMINU_H  : RVPBinary1LongFW<0b1101, 0b00, 0b110, "pminu.h">;
def PMINU_B  : RVPBinary1LongFW<0b1101, 0b10, 0b110, "pminu.b">;
def PMAX_H   : RVPBinary1LongFW<0b1110, 0b00, 0b110, "pmax.h">;
def PMAX_B   : RVPBinary1LongFW<0b1110, 0b10, 0b110, "pmax.b">;
def PMAXU_H  : RVPBinary1LongFW<0b1111, 0b00, 0b110, "pmaxu.h">;
def PMAXU_B  : RVPBinary1LongFW<0b1111, 0b10, 0b110, "pmaxu.b">;
} // Predicates = [HasStdExtP]
let DecoderNamespace = "RISCV32Only_",
    Predicates = [HasStdExtP, IsRV32] in {
def MSEQ  : RVPBinary1LongFW<0b1000, 0b01, 0b110, "mseq">;
def MSLT  : RVPBinary1LongFW<0b1010, 0b01, 0b110, "mslt">;
def MSLTU : RVPBinary1LongFW<0b1011, 0b01, 0b110, "msltu">;
} // Predicates = [HasStdExtP, IsRV32]
let Predicates = [HasStdExtP, IsRV64] in {
def PAS_WX   : RVPBinary1LongFW<0b0000, 0b01, 0b110, "pas.wx">;
def PSA_WX   : RVPBinary1LongFW<0b0000, 0b11, 0b110, "psa.wx">;
def PSAS_WX  : RVPBinary1LongFW<0b0010, 0b01, 0b110, "psas.wx">;
def PSSA_WX  : RVPBinary1LongFW<0b0010, 0b11, 0b110, "pssa.wx">;
def PAAS_WX  : RVPBinary1LongFW<0b0011, 0b01, 0b110, "paas.wx">;
def PASA_WX  : RVPBinary1LongFW<0b0011, 0b11, 0b110, "pasa.wx">;

def PMSEQ_W  : RVPBinary1LongFW<0b1000, 0b01, 0b110, "pmseq.w">;
def PMSLT_W  : RVPBinary1LongFW<0b1010, 0b01, 0b110, "pmslt.w">;
def PMSLTU_W : RVPBinary1LongFW<0b1011, 0b01, 0b110, "pmsltu.w">;

def PMIN_W   : RVPBinary1LongFW<0b1100, 0b01, 0b110, "pmin.w">;
def PMINU_W  : RVPBinary1LongFW<0b1101, 0b01, 0b110, "pminu.w">;
def PMAX_W   : RVPBinary1LongFW<0b1110, 0b01, 0b110, "pmax.w">;
def PMAXU_W  : RVPBinary1LongFW<0b1111, 0b01, 0b110, "pmaxu.w">;
} // Predicates = [HasStdExtP, IsRV64]


let Predicates = [HasStdExtP] in {
def PMULH_H      : RVPBinary1LongFW<0b0000, 0b00, 0b111, "pmulh.h">;
def PMULHR_H     : RVPBinary1LongFW<0b0000, 0b10, 0b111, "pmulhr.h">;
def PMHACC_H     : RVPBinary1LongFW<0b0001, 0b00, 0b111, "pmhacc.h">;
def PMHRACC_H    : RVPBinary1LongFW<0b0001, 0b10, 0b111, "pmhracc.h">;
def PMULHU_H     : RVPBinary1LongFW<0b0010, 0b00, 0b111, "pmulhu.h">;
def PMULHRU_H    : RVPBinary1LongFW<0b0010, 0b10, 0b111, "pmulhru.h">;
def PMHACCU_H    : RVPBinary1LongFW<0b0011, 0b00, 0b111, "pmhaccu.h">;
def PMHRACCU_H   : RVPBinary1LongFW<0b0011, 0b10, 0b111, "pmhraccu.h">;

def PMULH_H_B0    : RVPBinary1LongFW<0b0100, 0b00, 0b111, "pmulh.h.b0">;
def PMULHSU_H_B0  : RVPBinary1LongFW<0b0100, 0b10, 0b111, "pmulhsu.h.b0">;
def PMHACCU_H_B0  : RVPBinary1LongFW<0b0101, 0b00, 0b111, "pmhaccu.h.b0">;
def PMHACCSU_H_B0 : RVPBinary1LongFW<0b0101, 0b10, 0b111, "pmhaccsu.h.b0">;
def PMULH_H_B1    : RVPBinary1LongFW<0b0110, 0b00, 0b111, "pmulh.h.b1">;
def PMULHSU_H_B1  : RVPBinary1LongFW<0b0110, 0b10, 0b111, "pmulhsu.h.b1">;
def PMHACC_H_B1   : RVPBinary1LongFW<0b0111, 0b00, 0b111, "pmhacc.h.b1">;
def PMHACCSU_H_B1 : RVPBinary1LongFW<0b0111, 0b10, 0b111, "pmhaccsu.h.b1">;

def PMULHSU_H    : RVPBinary1LongFW<0b1000, 0b00, 0b111, "pmulhsu.h">;
def PMULHRSU_H   : RVPBinary1LongFW<0b1000, 0b10, 0b111, "pmulhrsu.h">;
def PMHACCSU_H   : RVPBinary1LongFW<0b1001, 0b00, 0b111, "pmhaccsu.h">;
def PMHRACCSU_H  : RVPBinary1LongFW<0b1001, 0b10, 0b111, "pmhraccsu.h">;
def PMULQ_H      : RVPBinary1LongFW<0b1010, 0b00, 0b111, "pmulq.h">;
def PMULQR_H     : RVPBinary1LongFW<0b1010, 0b10, 0b111, "pmulqr.h">;
} // Predicates = [HasStdExtP]

let DecoderNamespace = "RISCV32Only_",
    Predicates = [HasStdExtP, IsRV32] in {
def MULHR      : RVPBinary1LongFW<0b0000, 0b11, 0b111, "mulhr">;
def MHACC      : RVPBinary1LongFW<0b0001, 0b01, 0b111, "mhacc">;
def MHRACC     : RVPBinary1LongFW<0b0001, 0b11, 0b111, "mhracc">;
def MULHRU     : RVPBinary1LongFW<0b0010, 0b11, 0b111, "mulhru">;
def MHACCU     : RVPBinary1LongFW<0b0011, 0b01, 0b111, "mhaccu">;
def MHRACCU    : RVPBinary1LongFW<0b0011, 0b11, 0b111, "mhraccu">;

def MULH_H0    : RVPBinary1LongFW<0b0100, 0b01, 0b111, "mulh.h0">;
def MULHSU_H0  : RVPBinary1LongFW<0b0100, 0b11, 0b111, "mulhsu.h0">;
def MHACC_H0   : RVPBinary1LongFW<0b0101, 0b01, 0b111, "mhacc.h0">;
def MHACCSU_H0 : RVPBinary1LongFW<0b0101, 0b11, 0b111, "mhaccsu.h0">;
def MULH_H1    : RVPBinary1LongFW<0b0110, 0b01, 0b111, "mulh.h1">;
def MULHSU_H1  : RVPBinary1LongFW<0b0110, 0b11, 0b111, "mulhsu.h1">;
def MHACC_H1   : RVPBinary1LongFW<0b0111, 0b01, 0b111, "mhacc.h1">;
def MHACCSU_H1 : RVPBinary1LongFW<0b0111, 0b11, 0b111, "mhaccsu.h1">;

def MULHRSU_H  : RVPBinary1LongFW<0b1000, 0b11, 0b111, "mulhrsu.h">;
def MHACCSU    : RVPBinary1LongFW<0b1001, 0b01, 0b111, "mhaccsu">;
def MHRACCSU   : RVPBinary1LongFW<0b1001, 0b11, 0b111, "mhraccsu">;
def MULQ       : RVPBinary1LongFW<0b1010, 0b01, 0b111, "mulq">;
def MULQR      : RVPBinary1LongFW<0b1010, 0b11, 0b111, "mulqr">;

def MQACC_H00  : RVPBinary1LongFW<0b1101, 0b00, 0b111, "mqacc.h00">;
def MQRACC_H00 : RVPBinary1LongFW<0b1101, 0b10, 0b111, "mqracc.h00">;
def MQACC_H11  : RVPBinary1LongFW<0b1111, 0b00, 0b111, "mqacc.h11">;
def MQRACC_H11 : RVPBinary1LongFW<0b1111, 0b10, 0b111, "mqracc.h11">;
} // Predicates = [HasStdExtP, IsRV32]

let Predicates = [HasStdExtP, IsRV64] in {
def PMULH_W      : RVPBinary1LongFW<0b0000, 0b01, 0b111, "pmulh.w">;
def PMULHR_W     : RVPBinary1LongFW<0b0000, 0b11, 0b111, "pmulhr.w">;
def PMHACC_W     : RVPBinary1LongFW<0b0001, 0b01, 0b111, "pmhacc.w">;
def PMHRACC_W    : RVPBinary1LongFW<0b0001, 0b11, 0b111, "pmhracc.w">;
def PMULHU_W     : RVPBinary1LongFW<0b0010, 0b01, 0b111, "pmulhu.w">;
def PMULHRU_W    : RVPBinary1LongFW<0b0010, 0b11, 0b111, "pmulhru.w">;
def PMHACCU_W    : RVPBinary1LongFW<0b0011, 0b01, 0b111, "pmhaccu.w">;
def PMHRACCU_W   : RVPBinary1LongFW<0b0011, 0b11, 0b111, "pmhraccu.w">;

def PMULH_W_H0   : RVPBinary1LongFW<0b0100, 0b01, 0b111, "pmulh.w.h0">;
def PMULHSU_W_H0 : RVPBinary1LongFW<0b0100, 0b11, 0b111, "pmulhsu.w.h0">;
def PMHACC_W_H0  : RVPBinary1LongFW<0b0101, 0b01, 0b111, "pmhacc.w.h0">;
def PMHACCSU_W_H0: RVPBinary1LongFW<0b0101, 0b11, 0b111, "pmhaccsu.w.h0">;
def PMULH_W_H1   : RVPBinary1LongFW<0b0110, 0b01, 0b111, "pmulh.w.h1">;
def PMULHSU_W_H1 : RVPBinary1LongFW<0b0110, 0b11, 0b111, "pmulhsu.w.h1">;
def PMHACC_W_H1  : RVPBinary1LongFW<0b0111, 0b01, 0b111, "pmhacc.w.h1">;
def PMHACCSU_W_H1: RVPBinary1LongFW<0b0111, 0b11, 0b111, "pmhaccsu.w.h1">;

def PMULHSU_W    : RVPBinary1LongFW<0b1000, 0b01, 0b111, "pmulhsu.w">;
def PMULHRSU_W   : RVPBinary1LongFW<0b1000, 0b11, 0b111, "pmulhrsu.w">;
def PMHACCSU_W   : RVPBinary1LongFW<0b1001, 0b01, 0b111, "pmhaccsu.w">;
def PMHRACCSU_W  : RVPBinary1LongFW<0b1001, 0b11, 0b111, "pmhraccsu.w">;
def PMULQ_W      : RVPBinary1LongFW<0b1010, 0b01, 0b111, "pmulq.w">;
def PMULQR_W     : RVPBinary1LongFW<0b1010, 0b11, 0b111, "pmulqr.w">;

def PMQACC_W_H00 : RVPBinary1LongFW<0b1101, 0b00, 0b111, "pmqacc.w.h00">;
def MQACC_W00    : RVPBinary1LongFW<0b1101, 0b01, 0b111, "mqacc.w00">;
def PMQRACC_W_H00: RVPBinary1LongFW<0b1101, 0b10, 0b111, "pmqracc.w.h00">;
def MQRACC_W00   : RVPBinary1LongFW<0b1101, 0b11, 0b111, "mqracc.w00">;
def PMQACC_W_H11 : RVPBinary1LongFW<0b1111, 0b00, 0b111, "pmqacc.w.h11">;
def MQACC_W11    : RVPBinary1LongFW<0b1111, 0b01, 0b111, "mqacc.w11">;
def PMQRACC_W_H11: RVPBinary1LongFW<0b1111, 0b10, 0b111, "pmqracc.w.h11">;
def MQRACC_W11   : RVPBinary1LongFW<0b1111, 0b11, 0b111, "mqracc.w11">;
} // Predicates = [HasStdExtP, IsRV64]


let Predicates = [HasStdExtP, IsRV32] in {
def PWSLLI_B : RVPUnary0F0Rdp<0b000, 0b0010000, "pwslli.b">;
def PWSLLI_H : RVPUnary0F0Rdp<0b001, 0b0100000, "pwslli.h">;
def WSLLI    : RVPUnary0F0Rdp<0b010, 0b1000000, "wslli">;

def PWSLAI_B : RVPUnary0F0Rdp<0b100, 0b0010000, "pwslai.b">;
def PWSLAI_H : RVPUnary0F0Rdp<0b100, 0b0100000, "pwslai.h">;
def WSLAI    : RVPUnary0F0Rdp<0b100, 0b1000000, "wslai">;

def PLI_DH   : RVPUnaryImm9Rdp<0b0011000,  "pli.dh">;
def PLI_DB   : RVPUnaryImm8Rdp<0b00110100, "pli.db">;
def PLUI_DH  : RVPUnaryImm9Rdp<0b0111000,  "plui.dh">;


def PWSLLI_BS : RVPBinary0F1WRdp<0b000, 0b00, "pwslli.bs">;
def PWSLL_HS  : RVPBinary0F1WRdp<0b000, 0b01, "pwsll.hs">;
def WSLL        : RVPBinary0F1WRdp<0b000, 0b11, "wsll">;

def PWSLA_BS : RVPBinary0F1WRdp<0b100, 0b00, "pwsla.bs">;
def PWSLA_HS : RVPBinary0F1WRdp<0b100, 0b01, "pwsla.hs">;
def WSLA       : RVPBinary0F1WRdp<0b100, 0b11, "wsla">;
def WZIP8P     : RVPBinary0F1WRdp<0b111, 0b00, "wzip8p">;
def WZIP16P    : RVPBinary0F1WRdp<0b111, 0b01, "wzip16p">;

def PWADD_H     : RVPBinary0LongFW<0b0000, 0b00, "pwadd.h">;
def WADD        : RVPBinary0LongFW<0b0000, 0b01, "wadd">;
def PWADD_B     : RVPBinary0LongFW<0b0000, 0b10, "pwadd.b">;
def PW2WADD_H   : RVPBinary0LongFW<0b0000, 0b11, "pw2wadd.h">;
def PWADDA_H    : RVPBinary0LongFW<0b0001, 0b00, "pwadda.h">;
def WADDA       : RVPBinary0LongFW<0b0001, 0b01, "wadda">;
def PWADDA_B    : RVPBinary0LongFW<0b0001, 0b10, "pwadda.b">;
def PW2WADDA_H  : RVPBinary0LongFW<0b0001, 0b11, "pw2wadda.h">;
def PWADDU_H    : RVPBinary0LongFW<0b0010, 0b00, "pwaddu.h">;
def WADDU       : RVPBinary0LongFW<0b0010, 0b01, "waddu">;
def PWADDU_B    : RVPBinary0LongFW<0b0010, 0b10, "pwaddu.b">;
def PW2WADD_HX  : RVPBinary0LongFW<0b0010, 0b11, "pw2wadd.hx">;
def PWADDAU_H   : RVPBinary0LongFW<0b0011, 0b00, "pwaddau.h">;
def WADDAU      : RVPBinary0LongFW<0b0011, 0b01, "waddau">;
def PWADDAU_B   : RVPBinary0LongFW<0b0011, 0b10, "pwaddau.b">;
def PW2WADDA_HX : RVPBinary0LongFW<0b0011, 0b11, "pw2wadda.hx">;

def PWMUL_H     : RVPBinary0LongFW<0b0100, 0b00, "pwmul.h">;
def WMUL        : RVPBinary0LongFW<0b0100, 0b01, "wmul">;
def PWMUL_B     : RVPBinary0LongFW<0b0100, 0b10, "pwmul.b">;
def PW2WADDU_H  : RVPBinary0LongFW<0b0100, 0b11, "pw2waddu.h">;
def PWMACC_H    : RVPBinary0LongFW<0b0101, 0b00, "pwmacc.h">;
def WMACC       : RVPBinary0LongFW<0b0101, 0b01, "wmacc">;
def PM2WADDAU_H : RVPBinary0LongFW<0b0101, 0b11, "pm2waddau.h">;
def PWMULU_H    : RVPBinary0LongFW<0b0110, 0b00, "pwmulu.h">;
def WMULU       : RVPBinary0LongFW<0b0110, 0b01, "wmulu">;
def PWMULU_B    : RVPBinary0LongFW<0b0110, 0b10, "pwmulu.b">;
def PWMACCU_H   : RVPBinary0LongFW<0b0111, 0b00, "pwmaccu.h">;
def WMACCU      : RVPBinary0LongFW<0b0111, 0b01, "wmaccu">;

def PWSUB_H     : RVPBinary0LongFW<0b1000, 0b00, "pwsub.h">;
def WSUB        : RVPBinary0LongFW<0b1000, 0b01, "wsub">;
def PWSUB_B     : RVPBinary0LongFW<0b1000, 0b10, "pwsub.b">;
def PW2WSUB_H   : RVPBinary0LongFW<0b1000, 0b11, "pw2wsub.h">;
def PWSUBA_H    : RVPBinary0LongFW<0b1001, 0b00, "pwsuba.h">;
def WSUBA       : RVPBinary0LongFW<0b1001, 0b01, "wsuba">;
def PWSUBA_B    : RVPBinary0LongFW<0b1001, 0b10, "pwsuba.b">;
def PW2WSUBA_H  : RVPBinary0LongFW<0b1001, 0b11, "pw2wsuba.h">;
def PWSUBU_H    : RVPBinary0LongFW<0b1010, 0b00, "pwsubu.h">;
def WSUBU       : RVPBinary0LongFW<0b1010, 0b01, "wsubu">;
def PWSUBU_B    : RVPBinary0LongFW<0b1010, 0b10, "pwsubu.b">;
def PW2WSUB_HX  : RVPBinary0LongFW<0b1010, 0b11, "pw2wsub.hx">;
def PWSUBAU_H   : RVPBinary0LongFW<0b1011, 0b00, "pwsubau.h">;
def WSUBAU      : RVPBinary0LongFW<0b1011, 0b01, "wsubau">;
def PWSUBAU_B   : RVPBinary0LongFW<0b1011, 0b10, "pwsubau.b">;
def PW2WSUBA_HX : RVPBinary0LongFW<0b1011, 0b11, "pw2wsuba.hx">;

def PWMULSU_H    : RVPBinary0LongFW<0b1100, 0b00, "pwmulsu.h">;
def WMULSU       : RVPBinary0LongFW<0b1100, 0b01, "wmulsu">;
def PWMULSU_B    : RVPBinary0LongFW<0b1100, 0b10, "pwmulsu.b">;
def PM2WADDSU_H  : RVPBinary0LongFW<0b1100, 0b11, "pm2waddsu.h">;
def PWMACCSU_H   : RVPBinary0LongFW<0b1101, 0b00, "pwmaccsu.h">;
def WMACCSU      : RVPBinary0LongFW<0b1101, 0b01, "wmaccsu">;
def PM2WADDASU_H : RVPBinary0LongFW<0b1101, 0b11, "pm2waddasu.h">;
def PMQWACC_H    : RVPBinary0LongFW<0b1111, 0b00, "pmqwacc.h">;
def PMQWACC      : RVPBinary0LongFW<0b1111, 0b01, "pmqwacc">;
def PMQRWACC_H   : RVPBinary0LongFW<0b1111, 0b10, "pmqrwacc.h">;
def PMQRWACC     : RVPBinary0LongFW<0b1111, 0b11, "pmqrwacc">;

def PREDSUM_DHS   : RVPBinary0F1WRs1p<0b000, 0b00, "predsum.dhs">;
def PREDSUM_DBS   : RVPBinary0F1WRs1p<0b000, 0b10, "predsum.dbs">;
def PREDSUMU_DHS  : RVPBinary0F1WRs1p<0b011, 0b00, "predsumu.dhs">;
def PREDSUMU_DBS  : RVPBinary0F1WRs1p<0b011, 0b10, "predsumu.dbs">;

def PNSRLI_B     : RVPUnary0F0Rs1p<0b000, 0b0010000, "pnsrli.b">;
def PNSRLI_H     : RVPUnary0F0Rs1p<0b000, 0b0100000, "pnsrli.h">;
def NSRLI        : RVPUnary0F0Rs1p<0b000, 0b1000000, "nsrli">;
def PNCLIPIU_B   : RVPUnary0F0Rs1p<0b010, 0b0010000, "pnclipiu.b">;
def PNCLIPIU_H   : RVPUnary0F0Rs1p<0b010, 0b0100000, "pnclipiu.h">;
def NCLIPIU      : RVPUnary0F0Rs1p<0b010, 0b1000000, "nclipiu">;
def PNCLIPRIU_B  : RVPUnary0F0Rs1p<0b011, 0b0010000, "pnclipriu.b">;
def PNCLIPRIU_H  : RVPUnary0F0Rs1p<0b011, 0b0100000, "pnclipriu.h">;
def NCLIPRIU     : RVPUnary0F0Rs1p<0b011, 0b1000000, "nclipriu">;

def PNSRAI_B     : RVPUnary0F0Rs1p<0b100, 0b0010000, "pnsrai.b">;
def PNSRAI_H     : RVPUnary0F0Rs1p<0b100, 0b0100000, "pnsrai.h">;
def NSRAI        : RVPUnary0F0Rs1p<0b100, 0b1000000, "nsrai">;
def PNSARI_B     : RVPUnary0F0Rs1p<0b101, 0b0010000, "pnsari.b">;
def PNSARI_H     : RVPUnary0F0Rs1p<0b101, 0b0100000, "pnsari.h">;
def NSARI        : RVPUnary0F0Rs1p<0b101, 0b1000000, "nsari">;
def PNCLIPI_B    : RVPUnary0F0Rs1p<0b110, 0b0010000, "pnclipi.b">;
def PNCLIPI_H    : RVPUnary0F0Rs1p<0b110, 0b0100000, "pnclipi.h">;
def NCLIPI       : RVPUnary0F0Rs1p<0b110, 0b1000000, "nclipi">;
def PNCLIPRI_B   : RVPUnary0F0Rs1p<0b111, 0b0010000, "pnclipri.b">;
def PNCLIPRI_H   : RVPUnary0F0Rs1p<0b111, 0b0100000, "pnclipri.h">;
def NCLIPRI      : RVPUnary0F0Rs1p<0b111, 0b1000000, "nclipri">;

def PNSRL_BS     : RVPBinary0F1WRs1p<0b000, 0b00, "pnsrl.bs",    0b1>;
def PNSRL_HS     : RVPBinary0F1WRs1p<0b000, 0b01, "pnsrl.hs",    0b1>;
def NSRL         : RVPBinary0F1WRs1p<0b000, 0b11, "nsrl",        0b1>;
def PNCLIPU_BS   : RVPBinary0F1WRs1p<0b010, 0b00, "pnclipu.bs",  0b1>;
def PNCLIPU_HS   : RVPBinary0F1WRs1p<0b010, 0b01, "pnclipu.hs",  0b1>;
def NCLIPU       : RVPBinary0F1WRs1p<0b010, 0b11, "nclipu",      0b1>;
def PNCLIPRU_BS  : RVPBinary0F1WRs1p<0b011, 0b00, "pnclipru.bs", 0b1>;
def PNCLIPRU_HS  : RVPBinary0F1WRs1p<0b011, 0b01, "pnclipru.hs", 0b1>;
def NCLIPRU      : RVPBinary0F1WRs1p<0b011, 0b11, "nclipru",     0b1>;

def PNSRA_BS     : RVPBinary0F1WRs1p<0b100, 0b00, "pnsra.bs",   0b1>;
def PNSRA_HS     : RVPBinary0F1WRs1p<0b100, 0b01, "pnsra.hs",   0b1>;
def NSRA         : RVPBinary0F1WRs1p<0b100, 0b11, "nsra",       0b1>;
def PNSRAR_BS    : RVPBinary0F1WRs1p<0b101, 0b00, "pnsrar.bs",  0b1>;
def PNSRAR_HS    : RVPBinary0F1WRs1p<0b101, 0b01, "pnsrar.hs",  0b1>;
def NSRAR        : RVPBinary0F1WRs1p<0b101, 0b11, "nsrar",      0b1>;
def PNCLIP_BS    : RVPBinary0F1WRs1p<0b110, 0b00, "pnclip.bs",  0b1>;
def PNCLIP_HS    : RVPBinary0F1WRs1p<0b110, 0b01, "pnclip.hs",  0b1>;
def NCLIP        : RVPBinary0F1WRs1p<0b110, 0b11, "nclip",      0b1>;
def PNCLIPR_BS   : RVPBinary0F1WRs1p<0b111, 0b00, "pnclipr.bs", 0b1>;
def PNCLIPR_HS   : RVPBinary0F1WRs1p<0b111, 0b01, "pnclipr.hs", 0b1>;
def NCLIPR       : RVPBinary0F1WRs1p<0b111, 0b11, "nclipr",     0b1>;


def PSLLI_DB      : RVPUnary0F0Rs1pRdp<0b000, 0b0001000, "pslli.db">;
def PSLLI_DH      : RVPUnary0F0Rs1pRdp<0b000, 0b0010000, "pslli.dh">;
def PSLLI_DW      : RVPUnary0F0Rs1pRdp<0b000, 0b0100000, "pslli.dw">;

def PSSLAI_DH    : RVPUnary0F0Rs1pRdp<0b101, 0b0010000, "psslai.dh">;
def PSSLAI_DW    : RVPUnary0F0Rs1pRdp<0b101, 0b0100000, "psslai.dw">;

def PSEXT_DH_B    : RVPUnaryWUFRs1pRdp<0b00, 0b00100, "psext.dh.b">;
def PSEXT_DW_B    : RVPUnaryWUFRs1pRdp<0b01, 0b00100, "psext.dw.b">;
def PSEXT_DW_H    : RVPUnaryWUFRs1pRdp<0b01, 0b00101, "psext.dw.h">;
def PSABS_DH      : RVPUnaryWUFRs1pRdp<0b00, 0b00111, "psabs.dh">;
def PSABS_DB      : RVPUnaryWUFRs1pRdp<0b10, 0b00111, "psabs.db">;

def PSLL_DHS    : RVPBinary0F1WRs1pRdp<0b000, 0b00, "psll.dhs">;
def PSLL_DWS    : RVPBinary0F1WRs1pRdp<0b000, 0b01, "psll.dws">;
def PSLL_DBS    : RVPBinary0F1WRs1pRdp<0b000, 0b10, "psll.dbs">;
def PADD_DHS    : RVPBinary0F1WRs1pRdp<0b001, 0b00, "padd.dhs">;
def PADD_DWS    : RVPBinary0F1WRs1pRdp<0b001, 0b01, "padd.dws">;
def PADD_DBS    : RVPBinary0F1WRs1pRdp<0b001, 0b10, "padd.dbs">;

def PSSHA_DHS   : RVPBinary0F1WRs1pRdp<0b110, 0b00, "pssha.dhs">;
def PSSHA_DWS   : RVPBinary0F1WRs1pRdp<0b110, 0b01, "pssha.dws">;
def PSSHAR_DHS  : RVPBinary0F1WRs1pRdp<0b111, 0b00, "psshar.dhs">;
def PSSHAR_DWS  : RVPBinary0F1WRs1pRdp<0b111, 0b01, "psshar.dws">;

def PSRLI_DB    : RVPUnary0F0Rs1pRdp<0b000, 0b0001000, "psrli.db",  0b1>;
def PSRLI_DH    : RVPUnary0F0Rs1pRdp<0b000, 0b0010000, "psrli.dh",  0b1>;
def PSRLI_DW    : RVPUnary0F0Rs1pRdp<0b000, 0b0100000, "psrli.dw",  0b1>;
def PUSATI_DH   : RVPUnary0F0Rs1pRdp<0b010, 0b0010000, "pusati.dh", 0b1>;
def PUSATI_DW   : RVPUnary0F0Rs1pRdp<0b010, 0b0100000, "pusati.dw", 0b1>;

def PSRAI_DB    : RVPUnary0F0Rs1pRdp<0b100, 0b0001000, "psrai.db",  0b1>;
def PSRAI_DH    : RVPUnary0F0Rs1pRdp<0b100, 0b0010000, "psrai.dh",  0b1>;
def PSRAI_DW    : RVPUnary0F0Rs1pRdp<0b100, 0b0100000, "psrai.dw",  0b1>;
def PSRARI_DH   : RVPUnary0F0Rs1pRdp<0b101, 0b0010000, "psrari.dh", 0b1>;
def PSRARI_DW   : RVPUnary0F0Rs1pRdp<0b101, 0b0100000, "psrari.dw", 0b1>;
def PSATI_DH    : RVPUnary0F0Rs1pRdp<0b110, 0b0010000, "psati.dh",  0b1>;
def PSATI_DW    : RVPUnary0F0Rs1pRdp<0b110, 0b0100000, "psati.dw",  0b1>;

def PSRL_DHS  : RVPBinary0F1WRs1pRdp<0b000, 0b00, "psrl.dhs", 0b1>;
def PSRL_DWS  : RVPBinary0F1WRs1pRdp<0b000, 0b01, "psrl.dws", 0b1>;
def PSRL_DBS  : RVPBinary0F1WRs1pRdp<0b000, 0b10, "psrl.dbs", 0b1>;

def PSRA_DHS  : RVPBinary0F1WRs1pRdp<0b100, 0b00, "psra.dhs", 0b1>;
def PSRA_DWS  : RVPBinary0F1WRs1pRdp<0b100, 0b01, "psra.dws", 0b1>;
def PSRA_DBS  : RVPBinary0F1WRs1pRdp<0b100, 0b10, "psra.dbs", 0b1>;

def PADD_DH     : RVPBinary1FWRs2pRs1pRdp<0b0000, 0b00, "padd.dh">;
def PADD_DW     : RVPBinary1FWRs2pRs1pRdp<0b0000, 0b01, "padd.dw">;
def PADD_DB     : RVPBinary1FWRs2pRs1pRdp<0b0000, 0b10, "padd.db">;
def ADDD        : RVPBinary1FWRs2pRs1pRdp<0b0000, 0b11, "addd">;
def PSADD_DH    : RVPBinary1FWRs2pRs1pRdp<0b0010, 0b00, "psadd.dh">;
def PSADD_DW    : RVPBinary1FWRs2pRs1pRdp<0b0010, 0b01, "psadd.dw">;
def PSADD_DB    : RVPBinary1FWRs2pRs1pRdp<0b0010, 0b10, "psadd.db">;
def PAADD_DH    : RVPBinary1FWRs2pRs1pRdp<0b0011, 0b00, "paadd.dh">;
def PAADD_DW    : RVPBinary1FWRs2pRs1pRdp<0b0011, 0b01, "paadd.dw">;
def PAADD_DB    : RVPBinary1FWRs2pRs1pRdp<0b0011, 0b10, "paadd.db">;

def PSADDU_DH   : RVPBinary1FWRs2pRs1pRdp<0b0110, 0b00, "psaddu.dh">;
def PSADDU_DW   : RVPBinary1FWRs2pRs1pRdp<0b0110, 0b01, "psaddu.dw">;
def PSADDU_DB   : RVPBinary1FWRs2pRs1pRdp<0b0110, 0b10, "psaddu.db">;
def PAADDU_DH   : RVPBinary1FWRs2pRs1pRdp<0b0111, 0b00, "paaddu.dh">;
def PAADDU_DW   : RVPBinary1FWRs2pRs1pRdp<0b0111, 0b01, "paaddu.dw">;
def PAADDU_DB   : RVPBinary1FWRs2pRs1pRdp<0b0111, 0b10, "paaddu.db">;

def PSUB_DH     : RVPBinary1FWRs2pRs1pRdp<0b1000, 0b00, "psub.dh">;
def PSUB_DW     : RVPBinary1FWRs2pRs1pRdp<0b1000, 0b01, "psub.dw">;
def PSUB_DB     : RVPBinary1FWRs2pRs1pRdp<0b1000, 0b10, "psub.db">;
def SUBD        : RVPBinary1FWRs2pRs1pRdp<0b1000, 0b11, "subd">;
def PDIF_DH     : RVPBinary1FWRs2pRs1pRdp<0b1001, 0b00, "pdif.dh">;
def PDIF_DB     : RVPBinary1FWRs2pRs1pRdp<0b1001, 0b10, "pdif.db">;
def PSSUB_DH    : RVPBinary1FWRs2pRs1pRdp<0b1010, 0b00, "pssub.dh">;
def PSSUB_DW    : RVPBinary1FWRs2pRs1pRdp<0b1010, 0b01, "pssub.dw">;
def PSSUB_DB    : RVPBinary1FWRs2pRs1pRdp<0b1010, 0b10, "pssub.db">;
def PASUB_DH    : RVPBinary1FWRs2pRs1pRdp<0b1011, 0b00, "pasub.dh">;
def PASUB_DW    : RVPBinary1FWRs2pRs1pRdp<0b1011, 0b01, "pasub.dw">;
def PASUB_DB    : RVPBinary1FWRs2pRs1pRdp<0b1011, 0b10, "pasub.db">;

def PDIFU_DH    : RVPBinary1FWRs2pRs1pRdp<0b1101, 0b00, "pdifu.dh">;
def PDIFU_DB    : RVPBinary1FWRs2pRs1pRdp<0b1101, 0b10, "pdifu.db">;
def PSSUBU_DH   : RVPBinary1FWRs2pRs1pRdp<0b1110, 0b00, "pssubu.dh">;
def PSSUBU_DW   : RVPBinary1FWRs2pRs1pRdp<0b1110, 0b01, "pssubu.dw">;
def PSSUBU_DB   : RVPBinary1FWRs2pRs1pRdp<0b1110, 0b10, "pssubu.db">;
def PASUBU_DH   : RVPBinary1FWRs2pRs1pRdp<0b1111, 0b00, "pasubu.dh">;
def PASUBU_DW   : RVPBinary1FWRs2pRs1pRdp<0b1111, 0b01, "pasubu.dw">;
def PASUBU_DB   : RVPBinary1FWRs2pRs1pRdp<0b1111, 0b10, "pasubu.db">;

def PSH1ADD_DH   : RVPBinary1F0WRs2pRs1pRdp<0b010, 0b00, "psh1add.dh">;
def PSH1ADD_DW   : RVPBinary1F0WRs2pRs1pRdp<0b010, 0b01, "psh1add.dw">;
def PSSH1SADD_DH : RVPBinary1F0WRs2pRs1pRdp<0b011, 0b00, "pssh1sadd.dh">;
def PSSH1SADD_DW : RVPBinary1F0WRs2pRs1pRdp<0b011, 0b01, "pssh1sadd.dw">;

def PPACK_DH   : RVPBinary1F0WRs2pRs1pRdp<0b000, 0b00, "ppack.dh",   0b0, 0b1>;
def PPACK_DW   : RVPBinary1F0WRs2pRs1pRdp<0b000, 0b01, "ppack.dw",   0b0, 0b1>;
def PPACKBT_DH : RVPBinary1F0WRs2pRs1pRdp<0b001, 0b00, "ppackbt.dh", 0b0, 0b1>;
def PPACKBT_DW : RVPBinary1F0WRs2pRs1pRdp<0b001, 0b01, "ppackbt.dw", 0b0, 0b1>;
def PPACKTB_DH : RVPBinary1F0WRs2pRs1pRdp<0b010, 0b00, "ppacktb.dh", 0b0, 0b1>;
def PPACKTB_DW : RVPBinary1F0WRs2pRs1pRdp<0b010, 0b01, "ppacktb.dw", 0b0, 0b1>;
def PPACKT_DH  : RVPBinary1F0WRs2pRs1pRdp<0b011, 0b00, "ppackt.dh",  0b0, 0b1>;
def PPACKT_DW  : RVPBinary1F0WRs2pRs1pRdp<0b011, 0b01, "ppackt.dw",  0b0, 0b1>;

def PAS_DHX      : RVPBinary1FWRs2pRs1pRdp<0b0000, 0b00, "pas.dhx",  0b1>;
def PSA_DHX      : RVPBinary1FWRs2pRs1pRdp<0b0000, 0b10, "psa.dhx",  0b1>;
def PSAS_DHX     : RVPBinary1FWRs2pRs1pRdp<0b0010, 0b00, "psas.dhx", 0b1>;
def PSSA_DHX     : RVPBinary1FWRs2pRs1pRdp<0b0010, 0b10, "pssa.dhx", 0b1>;
def PAAX_DHX     : RVPBinary1FWRs2pRs1pRdp<0b0011, 0b00, "paax.dhx", 0b1>;
def PASA_DHX     : RVPBinary1FWRs2pRs1pRdp<0b0011, 0b10, "pasa.dhx", 0b1>;

def PMSEQ_DH     : RVPBinary1FWRs2pRs1pRdp<0b1000, 0b00, "pmseq.dh",  0b1>;
def PMSEQ_DW     : RVPBinary1FWRs2pRs1pRdp<0b1000, 0b01, "pmseq.dw",  0b1>;
def PMSEQ_DB     : RVPBinary1FWRs2pRs1pRdp<0b1000, 0b10, "pmseq.db",  0b1>;
def PMSLT_DH     : RVPBinary1FWRs2pRs1pRdp<0b1010, 0b00, "pmslt.dh",  0b1>;
def PMSLT_DW     : RVPBinary1FWRs2pRs1pRdp<0b1010, 0b01, "pmslt.dw",  0b1>;
def PMSLT_DB     : RVPBinary1FWRs2pRs1pRdp<0b1010, 0b10, "pmslt.db",  0b1>;
def PMSLTU_DH    : RVPBinary1FWRs2pRs1pRdp<0b1011, 0b00, "pmsltu.dh", 0b1>;
def PMSLTU_DW    : RVPBinary1FWRs2pRs1pRdp<0b1011, 0b01, "pmsltu.dw", 0b1>;
def PMSLTU_DB    : RVPBinary1FWRs2pRs1pRdp<0b1011, 0b10, "pmsltu.db", 0b1>;

def PMIN_DH      : RVPBinary1FWRs2pRs1pRdp<0b1100, 0b00, "pmin.dh",  0b1>;
def PMIN_DB      : RVPBinary1FWRs2pRs1pRdp<0b1100, 0b10, "pmin.db",  0b1>;
def PMINU_DH     : RVPBinary1FWRs2pRs1pRdp<0b1101, 0b00, "pminu.dh", 0b1>;
def PMINU_DB     : RVPBinary1FWRs2pRs1pRdp<0b1101, 0b10, "pminu.db", 0b1>;
def PMAX_DH      : RVPBinary1FWRs2pRs1pRdp<0b1110, 0b00, "pmax.dh",  0b1>;
def PMAX_DB      : RVPBinary1FWRs2pRs1pRdp<0b1110, 0b10, "pmax.db",  0b1>;
def PMAXU_DH     : RVPBinary1FWRs2pRs1pRdp<0b1111, 0b00, "pmaxu.dh", 0b1>;
def PMAXU_DB     : RVPBinary1FWRs2pRs1pRdp<0b1111, 0b10, "pmaxu.db", 0b1>;
} // Predicates = [HasStdExtP, IsRV32]

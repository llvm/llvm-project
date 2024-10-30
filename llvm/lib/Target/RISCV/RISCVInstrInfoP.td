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
///
/// This version is still experimental as the 'P' extension hasn't been
/// ratified yet.
///
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// Operand and SDNode transformation definitions.
//===----------------------------------------------------------------------===//

def simm10 : RISCVSImmLeafOp<10> {
  let MCOperandPredicate = [{
    int64_t Imm;
    if (MCOp.evaluateAsConstantImm(Imm))
      return isInt<10>(Imm);
    return MCOp.isBareSymbolRef();
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
class RVPUnaryImm9<bits<7> funct7, RISCVOpcode opcode,
                                 string opcodestr, DAGOperand TyRd = GPR>
  : RVInstIBase<0b010, opcode, (outs TyRd:$rd), (ins simm10:$simm10),
                opcodestr, "$rd, $simm10"> {
  bits<10> simm10;

  let Inst{31-25} = funct7;
  let Inst{24-15} = simm10;
}

let hasSideEffects = 0, mayLoad = 0, mayStore = 0 in
class RVPUnaryImm8<bits<8> funct8, RISCVOpcode opcode,
                                 string opcodestr, DAGOperand TyRd = GPR>
  : RVInstIBase<0b010, opcode, (outs TyRd:$rd), (ins uimm8:$uimm8),
                opcodestr, "$rd, $uimm8"> {
  bits<8> uimm8;
  let Inst{31-24} = funct8;
  let Inst{23-16} = uimm8;
  let Inst{15} = 0b0;
}

let hasSideEffects = 0, mayLoad = 0, mayStore = 0 in
class RVPUnaryWUF<bits<2> w, bits<5> uf,
               string opcodestr>
    : RVInstIBase<0b010, OPC_OP_IMM_32, (outs GPR:$rd), (ins GPR:$rs1),
                        opcodestr, "$rd, $rs1">  {
  let Inst{31-27} = 0b11100;
  let Inst{26-25} = w;
  let Inst{24-20} = uf;
}

let hasSideEffects = 0, mayLoad = 0, mayStore = 0 in
class RVPUnaryWUFRs1pRdp<bits<2> w, bits<5> uf, string opcodestr>
    : RVInstIBase<0b010, OPC_OP_IMM_32, (outs GPRPairRV32:$rdp), (ins GPRPairRV32:$rs1p),
                        opcodestr, "$rdp, $rs1p">  {
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
class RVPUnaryF<bit bfr, bits<3> f, bit aft, bits<7> wuimm,
                string opcodestr, bits<3> funct3 = 0b100,
                dag outs = (outs GPR:$rd), dag ins = (ins GPR:$rs1),
                string argstr = "$rd, $rs1">
    : RVInstIBase<funct3, OPC_OP_IMM_32, outs, ins, opcodestr, argstr>  {
  let Inst{31} = bfr;
  let Inst{30-28} = f;
  let Inst{27} = aft;
  let Inst{26-20} = wuimm;
}

class RVPUnary1F0<bits<3> f, bits<7> wuimm, string opcodestr>
    : RVPUnaryF<1, f, 0, wuimm, opcodestr>;

class RVPUnary0F0Rdp<bits<3> f, bits<7> wuimm, string opcodestr>
    : RVPUnaryF<0, f, 0, wuimm, opcodestr, 0b100, (outs GPRPairRV32:$rdp),
               (ins GPR:$rs1), "$rdp, $rs1"> {
  bits<4> rdp;

  let Inst{11-8} = rdp;
  let Inst{7}    = 0b0;
}

class RVPUnary0F0Rs1p<bits<3> f, bits<7> wuimm, string opcodestr>
    : RVPUnaryF<0, f, 0, wuimm, opcodestr, 0b100, (outs GPR:$rd),
               (ins GPRPairRV32:$rs1p), "$rd, $rs1p"> {
  bits<4> rs1p;

  let Inst{19-16} = rs1p;
  let Inst{15}    = 0b1;
}

class RVPUnary0F0Rs1pRdp<bits<3> f, bits<7> wuimm, string opcodestr,
                         bit aft = 0b0>
    : RVPUnaryF<0, f, 0, wuimm, opcodestr, 0b100, (outs GPRPairRV32:$rdp),
               (ins GPRPairRV32:$rs1p), "$rdp, $rs1p"> {
  bits<4> rs1p;
  bits<4> rdp;

  let Inst{19-16} = rs1p;
  let Inst{15}    = aft;
  let Inst{11-8}  = rdp;
  let Inst{7}     = 0b0;
}

let hasSideEffects = 0, mayLoad = 0, mayStore = 0 in
class RVPBinaryFW<bit bfr, bits<3> f, bit aft, bits<2> w,
               string opcodestr, bits<3> funct3, RISCVOpcode Opcode = OPC_OP_32,
               dag outs = (outs GPR:$rd), dag ins = (ins GPR:$rs1, GPR:$rs2),
               string argstr = "$rd, $rs1, $rs2">
    : RVInstRBase<funct3, Opcode, outs, ins, opcodestr, argstr>  {
  let Inst{31} = bfr;
  let Inst{30-28} = f;
  let Inst{27} = aft;
  let Inst{26-25} = w;
}

class RVPBinary1F1W<bits<3> f, bits<2> w, string opcodestr, bits<3> funct3,
                    RISCVOpcode Opcode = OPC_OP_32>
    : RVPBinaryFW<1, f, 1, w, opcodestr, funct3, Opcode>;

class RVPBinary1F0W<bits<3> f, bits<2> w, string opcodestr, bits<3> funct3,
                    RISCVOpcode Opcode = OPC_OP_32>
    : RVPBinaryFW<1, f, 0, w, opcodestr, funct3, Opcode>;

class RVPBinary0F1WRdp<bits<3> f, bits<2> w, string opcodestr,
                    RISCVOpcode Opcode = OPC_OP_IMM_32>
    : RVPBinaryFW<0, f, 1, w, opcodestr, 0b010, Opcode, (outs GPRPairRV32:$rdp),
                  (ins GPR:$rs1, GPR:$rs2), "$rdp, $rs1, $rs2"> {
  bits<4> rdp;

  let Inst{11-8} = rdp;
  let Inst{7}    = 0b0;
}

class RVPBinary0F1WRs1p<bits<3> f, bits<2> w, string opcodestr,
                    bit aft = 0b0, RISCVOpcode Opcode = OPC_OP_IMM_32>
    : RVPBinaryFW<0, f, 1, w, opcodestr, 0b100, Opcode, (outs GPR:$rd),
                  (ins GPRPairRV32:$rs1p, GPR:$rs2), "$rd, $rs1p, $rs2"> {
   bits<4> rs1p;

  let Inst{19-16} = rs1p;
  let Inst{15}    = aft;
}

class RVPBinary0F1WRs1pRdp<bits<3> f, bits<2> w, string opcodestr,
                           bit aft = 0b0, RISCVOpcode Opcode = OPC_OP_IMM_32>
    : RVPBinaryFW<0, f, 1, w, opcodestr, 0b110, Opcode, (outs GPRPairRV32:$rdp),
                  (ins GPRPairRV32:$rs1p, GPR:$rs2), "$rdp, $rs1p, $rs2"> {
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
    : RVInstRBase<0b110, Opcode, (outs GPRPairRV32:$rdp), (ins GPRPairRV32:$rs1p, GPRPairRV32:$rs2p),
                  opcodestr, "$rdp, $rs1p, $rs2p"> {
  bits<4> rs1p;
  bits<4> rs2p;
  bits<4> rdp;

  let Inst{31} = 0b1;
  let Inst{30-27} = f;
  let Inst{26-25} = w;
  let Inst{24-21} = rs2p;
  let Inst{20} = aft;
  let Inst{19-16} = rs1p;
  let Inst{15} = aft;
  let Inst{11-8} = rdp;
  let Inst{7} = 0b0;
}

class RVPBinary1F0WRs2pRs1pRdp<bits<3> f, bits<2> w, string opcodestr,
                    bit bfr = 0b1, bit aft = 0b0, RISCVOpcode Opcode = OPC_OP_IMM_32>
    : RVPBinaryFW<1, f, 0, w, opcodestr, 0b110, Opcode, (outs GPRPairRV32:$rdp),
                  (ins GPRPairRV32:$rs1p, GPRPairRV32:$rs2p), "$rdp, $rs1p, $rs2p"> {
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
class RVPBinaryLongFW<bit bfr, bits<4> f, bits<2> w,
               string opcodestr, bits<3> funct3, dag outs = (outs GPR:$rd),
               dag ins = (ins GPR:$rs1, GPR:$rs2),
               string argstr = "$rd, $rs1, $rs2">
    : RVInstRBase<funct3, OPC_OP_32, outs, ins,
                        opcodestr, argstr>  {
  let Inst{31} = bfr;
  let Inst{30-27} = f;
  let Inst{26-25} = w;
}

class RVPBinary1LongFW<bits<4> f, bits<2> w, string opcodestr, bits<3> funct3>
    : RVPBinaryLongFW<1, f, w, opcodestr, funct3>;

class RVPBinary0LongFW<bits<4> f, bits<2> w, string opcodestr>
    : RVPBinaryLongFW<0, f, w, opcodestr, 0b010, (outs GPRPairRV32:$rdp),
                      (ins GPR:$rs1, GPRPairRV32:$rs2), "$rdp, $rs1, $rs2"> {
  bits<4> rdp;

  let Inst{11-8} = rdp;
  let Inst{7}    = 0b1;
}

multiclass RVPUnaryBH<bits<5> funct5, string opcodestr> {
  def NAME # _B : RVPUnary<funct5, 0b0001000, 0b010, OPC_OP_IMM_32, opcodestr # ".b">;
  def NAME # _H : RVPUnary<funct5, 0b0010000, 0b010, OPC_OP_IMM_32, opcodestr # ".h">;
}

multiclass RVPUnaryHNonPacked<bits<5> funct5, string opcodestr> {
  def P # NAME # _H : RVPUnary<funct5, 0b0010000, 0b010, OPC_OP_IMM_32, "p" # opcodestr # ".h">;
  def NAME: RVPUnary<funct5, 0b0100000, 0b010, OPC_OP_IMM_32, opcodestr>;
}

multiclass RVPUnaryBHW<bits<5> funct5, string opcodestr> {
  defm NAME : RVPUnaryBH<funct5, opcodestr>;
  def NAME # _W: RVPUnary<funct5, 0b0100000, 0b010, OPC_OP_IMM_32, opcodestr # ".w">;
}

multiclass RVPUnaryHW<bits<5> funct5, string opcodestr> {
  def NAME # _H : RVPUnary<funct5, 0b0010000, 0b010, OPC_OP_IMM_32, opcodestr # ".h">;
  def NAME # _W: RVPUnary<funct5, 0b0100000, 0b010, OPC_OP_IMM_32, opcodestr # ".w">;
}

//===----------------------------------------------------------------------===//
// Instructions
//===----------------------------------------------------------------------===//

let Predicates = [HasStdExtP] in {
def CLS    : RVPUnary<0b01100, 0b0000011, 0b001, OPC_OP_IMM, "cls">;
def ABS    : RVPUnary<0b01100, 0b0000111, 0b001, OPC_OP_IMM, "abs">;
} // Predicates = [HasStdExtP]
let DecoderNamespace = "RISCV32Only_",
    Predicates = [HasStdExtP, IsRV32] in {
def REV_RV32  : RVPUnary<0b01101, 0b0011111, 0b101, OPC_OP_IMM, "rev">;
} // Predicates = [HasStdExtP, IsRV32]

let Predicates = [HasStdExtP, IsRV64] in {
def REV16 : RVPUnary<0b01101, 0b0110000, 0b101, OPC_OP_IMM, "rev16">;
def REV_RV64   : RVPUnary<0b01111, 0b0111111, 0b101, OPC_OP_IMM, "rev">;

def CLSW  : RVPUnary<0b01100, 0b0000011, 0b001, OPC_OP_IMM_32, "clsw">;
def ABSW  : RVPUnary<0b01100, 0b0000111, 0b001, OPC_OP_IMM_32, "absw">;
} // Predicates = [HasStdExtP, IsRV64]

let DecoderNamespace = "RISCV32Only_",
    Predicates = [HasStdExtP, IsRV32] in {
  defm PSLLI_RV32 : RVPUnaryBH<0b10000, "pslli">;
  defm SSLAI_RV32 : RVPUnaryHNonPacked<0b11010, "sslai">;
} // Predicates = [HasStdExtP, IsRV32]

let Predicates = [HasStdExtP, IsRV64] in {
  defm PSLLI_RV64  : RVPUnaryBHW<0b10000, "pslli">;
  defm PSSLAI_RV64 : RVPUnaryHW<0b01010, "psslai">;
} // Predicates = [HasStdExtP, IsRV64]

let Predicates = [HasStdExtP] in
def PLI_H : RVPUnaryImm9<0b1011000, OPC_OP_IMM_32, "pli.h">;
let Predicates = [HasStdExtP, IsRV64] in {
def PLI_W : RVPUnaryImm9<0b1011001, OPC_OP_IMM_32, "pli.w">;
} // Predicates = [HasStdExtP, IsRV64]
let Predicates = [HasStdExtP] in
def PLI_B : RVPUnaryImm8<0b10110100, OPC_OP_IMM_32, "pli.b">;

let DecoderNamespace = "RISCV32Only_",
    Predicates = [HasStdExtP, IsRV32] in {
def PSEXTB_H_RV32 : RVPUnaryWUF<0b00, 0b00100, "psextb.h">;
def PSABS_H_RV32  : RVPUnaryWUF<0b00, 0b00111, "psabs.h">;
def PSABS_B_RV32  : RVPUnaryWUF<0b01, 0b00111, "psabs.b">;
} // Predicates = [HasStdExtP, IsRV32]

let Predicates = [HasStdExtP, IsRV64] in {
def PSEXTB_H_RV64 : RVPUnaryWUF<0b00, 0b00100, "psextb.h">;
def PSEXTB_W      : RVPUnaryWUF<0b01, 0b00100, "psextb.w">;
def PSEXTH_W      : RVPUnaryWUF<0b01, 0b00101, "psexth.w">;
def PSABS_H_RV64  : RVPUnaryWUF<0b00, 0b00111, "psabs.h">;
def PSABS_B_RV64  : RVPUnaryWUF<0b10, 0b00111, "psabs.b">;
} // Predicates = [HasStdExtP, IsRV64]

let Predicates = [HasStdExtP] in
def PLUI_H : RVPUnaryImm9<0b1111000, OPC_OP_IMM_32, "plui.h">;
let Predicates = [HasStdExtP, IsRV64] in
def PLUI_W : RVPUnaryImm9<0b1111001, OPC_OP_IMM_32, "plui.w">;

let Predicates = [HasStdExtP] in {
def PSLL_H_H0   : RVPBinary1F1W<0b000, 0b00, "psll.h.h0", 0b010, OPC_OP_IMM_32>;
def PSLL_B_B0   : RVPBinary1F1W<0b000, 0b10, "psll.b.b0", 0b010, OPC_OP_IMM_32>;
def PADD_H_H0   : RVPBinary1F1W<0b001, 0b00, "padd.h.h0", 0b010, OPC_OP_IMM_32>;
def PADD_B_B0   : RVPBinary1F1W<0b001, 0b10, "padd.b.b0", 0b010, OPC_OP_IMM_32>;
def PSSHA_H_H0  : RVPBinary1F1W<0b110, 0b00, "pssha.h.h0", 0b010, OPC_OP_IMM_32>;
def PSSHAR_H_H0 : RVPBinary1F1W<0b111, 0b00, "psshar.h.h0", 0b010, OPC_OP_IMM_32>;
} // Predicates = [HasStdExtP]
let DecoderNamespace = "RISCV32Only_",
    Predicates = [HasStdExtP, IsRV32] in {
def SSHA        : RVPBinary1F1W<0b110, 0b01, "ssha", 0b010, OPC_OP_IMM_32>;
def SSHAR       : RVPBinary1F1W<0b111, 0b01, "sshar", 0b010, OPC_OP_IMM_32>;
} // Predicates = [HasStdExtP, IsRV32]

let Predicates = [HasStdExtP, IsRV64] in {
def PSLL_W_W0   : RVPBinary1F1W<0b000, 0b01, "psll.w.w0", 0b010, OPC_OP_IMM_32>;
def PADD_W_W0   : RVPBinary1F1W<0b001, 0b01, "padd.w.w0", 0b010, OPC_OP_IMM_32>;
def PSSHA_W_W0  : RVPBinary1F1W<0b110, 0b01, "pssha.w.w0", 0b010, OPC_OP_IMM_32>;
def PSSHAR_W_W0 : RVPBinary1F1W<0b111, 0b01, "psshar.w.w0", 0b010, OPC_OP_IMM_32>;
def SHA         : RVPBinary1F1W<0b110, 0b11, "sha", 0b010, OPC_OP_IMM_32>;
def SHAR        : RVPBinary1F1W<0b111, 0b11, "shar", 0b010, OPC_OP_IMM_32>;
} // Predicates = [HasStdExtP, IsRV64]

let Predicates = [HasStdExtP] in {
def PSRLI_B  : RVPUnary1F0<0b000, 0b0001000, "psrli.b">;
def PSRLI_H  : RVPUnary1F0<0b000, 0b0010000, "psrli.h">;
def PUSATI_H : RVPUnary1F0<0b010, 0b0010000, "pusati.h">;
def PSRAI_B  : RVPUnary1F0<0b100, 0b0001000, "psrai.b">;
def PSRAI_H  : RVPUnary1F0<0b100, 0b0010000, "psrai.h">;
def PSRARI_H : RVPUnary1F0<0b101, 0b0010000, "psrari.h">;
def PSATI_H  : RVPUnary1F0<0b110, 0b0010000, "psati.h">;
} // Predicates = [HasStdExtP]

let DecoderNamespace = "RISCV32Only_",
    Predicates = [HasStdExtP, IsRV32] in {
def USATI_RV32 : RVPUnary1F0<0b010, 0b0100000, "usati">;
def SRARI      : RVPUnary1F0<0b101, 0b0100000, "srari">;
def SATI       : RVPUnary1F0<0b110, 0b0100000, "sati">;
} // Predicates = [HasStdExtP, IsRV32]
let Predicates = [HasStdExtP, IsRV64] in {
def PSRLI_W       : RVPUnary1F0<0b000, 0b0100000, "psrli.w">;
def PUSATI_W      : RVPUnary1F0<0b010, 0b0100000, "pusati.w">;
def USATI_RV64    : RVPUnary1F0<0b010, 0b1000000, "usati">;
} // Predicates = [HasStdExtP, IsRV64]

let Predicates = [HasStdExtP] in {
def PSRL_H_H0  : RVPBinary1F1W<0b000, 0b00, "psrl.h.h0", 0b100, OPC_OP_IMM_32>;
def PSRL_B_B0  : RVPBinary1F1W<0b000, 0b10, "psrl.b.b0", 0b100, OPC_OP_IMM_32>;
def PREDSUM_H  : RVPBinary1F1W<0b001, 0b00, "predsum.h", 0b100, OPC_OP_IMM_32>;
def PREDSUM_B  : RVPBinary1F1W<0b001, 0b10, "predsum.b", 0b100, OPC_OP_IMM_32>;
def PREDSUMU_H : RVPBinary1F1W<0b011, 0b00, "predsumu.h", 0b100, OPC_OP_IMM_32>;
def PREDSUMU_B : RVPBinary1F1W<0b011, 0b10, "predsumu.b", 0b100, OPC_OP_IMM_32>;
def PSRA_H_H0  : RVPBinary1F1W<0b100, 0b00, "psra.h.h0", 0b100, OPC_OP_IMM_32>;
def PSRA_B_B0  : RVPBinary1F1W<0b100, 0b10, "psra.b.b0", 0b100, OPC_OP_IMM_32>;
} // Predicates = [HasStdExtP]

let Predicates = [HasStdExtP, IsRV64] in {
def PSRL_W_W0  : RVPBinary1F1W<0b000, 0b01, "psrl.w.w0", 0b100, OPC_OP_IMM_32>;
def PREDSUM_W  : RVPBinary1F1W<0b001, 0b01, "predsum.w", 0b100, OPC_OP_IMM_32>;
def PREDSUMU_W : RVPBinary1F1W<0b011, 0b01, "predsumu.w", 0b100, OPC_OP_IMM_32>;
def PSRA_W_W0  : RVPBinary1F1W<0b100, 0b01, "psra.w.w0", 0b100, OPC_OP_IMM_32>;
} // Predicates = [HasStdExtP, IsRV64]


let Predicates = [HasStdExtP] in {
def PADD_H  : RVPBinary1LongFW<0b0000, 0b00, "padd.h", 0b000>;
def PADD_B  : RVPBinary1LongFW<0b0000, 0b10, "padd.b", 0b000>;
def PSADD_H : RVPBinary1LongFW<0b0010, 0b00, "psadd.h", 0b000>;
def PSADD_B : RVPBinary1LongFW<0b0010, 0b10, "psadd.b", 0b000>;
def PAADD_H : RVPBinary1LongFW<0b0011, 0b00, "paadd.h", 0b000>;
def PAADD_B : RVPBinary1LongFW<0b0011, 0b10, "paadd.b", 0b000>;

def PSADDU_H : RVPBinary1LongFW<0b0110, 0b00, "psaddu.h", 0b000>;
def PSADDU_B : RVPBinary1LongFW<0b0110, 0b10, "psaddu.b", 0b000>;
def PAADDU_H : RVPBinary1LongFW<0b0111, 0b00, "paaddu.h", 0b000>;
def PAADDU_B : RVPBinary1LongFW<0b0111, 0b10, "paaddu.b", 0b000>;

def PSUB_H  : RVPBinary1LongFW<0b1000, 0b00, "psub.h", 0b000>;
def PSUB_B  : RVPBinary1LongFW<0b1000, 0b10, "psub.b", 0b000>;
def PDIF_H  : RVPBinary1LongFW<0b1001, 0b00, "pdif.h", 0b000>;
def PDIF_B  : RVPBinary1LongFW<0b1001, 0b10, "pdif.b", 0b000>;
def PSSUB_H : RVPBinary1LongFW<0b1010, 0b00, "pssub.h", 0b000>;
def PSSUB_B : RVPBinary1LongFW<0b1010, 0b10, "pssub.b", 0b000>;
def PASUB_H : RVPBinary1LongFW<0b1011, 0b00, "pasub.h", 0b000>;
def PASUB_B : RVPBinary1LongFW<0b1011, 0b10, "pasub.b", 0b000>;

def PDIFU_H  : RVPBinary1LongFW<0b1101, 0b00, "pdifu.h", 0b000>;
def PDIFU_B  : RVPBinary1LongFW<0b1101, 0b10, "pdifu.b", 0b000>;
def PSSUBU_H : RVPBinary1LongFW<0b1110, 0b00, "pssubu.h", 0b000>;
def PSSUBU_B : RVPBinary1LongFW<0b1110, 0b10, "pssubu.b", 0b000>;
def PASUBU_H : RVPBinary1LongFW<0b1111, 0b00, "pasubu.h", 0b000>;
def PASUBU_B : RVPBinary1LongFW<0b1111, 0b10, "pasubu.b", 0b000>;
} // Predicates = [HasStdExtP]

let DecoderNamespace = "RISCV32Only_",
    Predicates = [HasStdExtP, IsRV32] in {
def SADD : RVPBinary1LongFW<0b0010, 0b01, "sadd", 0b000>;
def AADD : RVPBinary1LongFW<0b0011, 0b01, "aadd", 0b000>;

def SADDU : RVPBinary1LongFW<0b0110, 0b01, "saddu", 0b000>;
def AADDU : RVPBinary1LongFW<0b0111, 0b01, "aaddu", 0b000>;

def SSUB : RVPBinary1LongFW<0b1010, 0b01, "ssub", 0b000>;
def ASUB : RVPBinary1LongFW<0b1011, 0b01, "asub", 0b000>;

def SSUBU : RVPBinary1LongFW<0b1110, 0b01, "ssubu", 0b000>;
def ASUBU : RVPBinary1LongFW<0b1111, 0b01, "asubu", 0b000>;
} // Predicates = [HasStdExtP, IsRV32]

let Predicates = [HasStdExtP, IsRV64] in {
def PADD_W  : RVPBinary1LongFW<0b0000, 0b01, "padd.w", 0b000>;
def PSADD_W : RVPBinary1LongFW<0b0010, 0b01, "psadd.w", 0b000>;
def PAADD_W : RVPBinary1LongFW<0b0011, 0b01, "paadd.w", 0b000>;

def PSADDU_W : RVPBinary1LongFW<0b0110, 0b01, "psaddu.w", 0b000>;
def PAADDU_W : RVPBinary1LongFW<0b0111, 0b01, "paaddu.w", 0b000>;

def PSUB_W  : RVPBinary1LongFW<0b1000, 0b01, "psub.w", 0b000>;
def PSSUB_W : RVPBinary1LongFW<0b1010, 0b01, "pssub.w", 0b000>;
def PASUB_W : RVPBinary1LongFW<0b1011, 0b01, "pasub.w", 0b000>;

def PSSUBU_W : RVPBinary1LongFW<0b1110, 0b01, "pssubu.w", 0b000>;
def PASUBU_W : RVPBinary1LongFW<0b1111, 0b01, "pasubu.w", 0b000>;
} // Predicates = [HasStdExtP, IsRV64]


let Predicates = [HasStdExtP] in {
def SLX        : RVPBinary1LongFW<0b0001, 0b11, "slx", 0b001>;
def PMUL_H_BEO : RVPBinary1LongFW<0b0010, 0b00, "pmul.h.beo", 0b001>;

def MVM         : RVPBinary1LongFW<0b0101, 0b00, "mvm", 0b001>;
def MVMN        : RVPBinary1LongFW<0b0101, 0b01, "mvmn", 0b001>;
def MERGE       : RVPBinary1LongFW<0b0101, 0b10, "merge", 0b001>;
def SRX         : RVPBinary1LongFW<0b0101, 0b11, "srx", 0b001>;
def PMULU_H_BEO : RVPBinary1LongFW<0b0110, 0b00, "pmulu.h.beo", 0b001>;
def PDIFSUMU_B  : RVPBinary1LongFW<0b0110, 0b10, "pdifsumu.b", 0b001>;
def PDIFSUMAU_B : RVPBinary1LongFW<0b0111, 0b10, "pdifsumau.b", 0b001>;
} // Predicates = [HasStdExtP]

let DecoderNamespace = "RISCV32Only_",
    Predicates = [HasStdExtP, IsRV32] in {
def MUL_H01  : RVPBinary1LongFW<0b0010, 0b01, "mul.h01", 0b001>;
def MACC_H01 : RVPBinary1LongFW<0b0011, 0b01, "macc.h01", 0b001>;

def MULU_H01  : RVPBinary1LongFW<0b0110, 0b01, "mulu.h01", 0b001>;
def MACCU_H01 : RVPBinary1LongFW<0b0111, 0b01, "maccu.h01", 0b001>;
} // Predicates = [HasStdExtP, IsRV32]

let Predicates = [HasStdExtP, IsRV64] in {
def PMUL_W_HEO  : RVPBinary1LongFW<0b0010, 0b01, "pmul.w.heo", 0b001>;
def MUL_W01     : RVPBinary1LongFW<0b0010, 0b11, "mul.w01", 0b001>;
def PMACC_W_HEO : RVPBinary1LongFW<0b0011, 0b01, "pmacc.w.heo", 0b001>;
def MACC_W01    : RVPBinary1LongFW<0b0011, 0b11, "macc.w01", 0b001>;

def PMULU_W_HEO  : RVPBinary1LongFW<0b0110, 0b01, "pmulu.w.heo", 0b001>;
def MULU_W01     : RVPBinary1LongFW<0b0110, 0b11, "mulu.w01", 0b001>;
def PMACCU_W_HEO : RVPBinary1LongFW<0b0111, 0b01, "pmaccu.w.heo", 0b001>;
def MACCU_W01    : RVPBinary1LongFW<0b0111, 0b11, "maccu.w01", 0b001>;
} // Predicates = [HasStdExtP, IsRV64]


let Predicates = [HasStdExtP] in {
def PSH1ADD_H   : RVPBinary1F0W<0b010, 0b00, "psh1add.h", 0b010>;
def PSSH1SADD_H : RVPBinary1F0W<0b011, 0b00, "pssh1sadd.h", 0b010>;
} // Predicates = [HasStdExtP]

let DecoderNamespace = "RISCV32Only_",
    Predicates = [HasStdExtP, IsRV32] in {
def SSH1SADD : RVPBinary1F0W<0b010, 0b01, "ssh1sadd", 0b010>;
} // Predicates = [HasStdExtP, IsRV32]
let Predicates = [HasStdExtP, IsRV64] in {
def PSH1ADD_W   : RVPBinary1F0W<0b010, 0b01, "psh1add.w", 0b010>;
def PSSH1SADD_W : RVPBinary1F0W<0b011, 0b01, "pssh1sadd.w", 0b010>;

def UNZIP8P   : RVPBinary1F0W<0b110, 0b00, "unzip8p", 0b010>;
def UNZIP16P  : RVPBinary1F0W<0b110, 0b01, "unzip16p", 0b010>;
def UNZIP8HP  : RVPBinary1F0W<0b110, 0b10, "unzip8hp", 0b010>;
def UNZIP16HP : RVPBinary1F0W<0b110, 0b11, "unzip16hp", 0b010>;
def ZIP8P     : RVPBinary1F0W<0b111, 0b00, "zip8p", 0b010>;
def ZIP16P    : RVPBinary1F0W<0b111, 0b01, "zip16p", 0b010>;
def ZIP8HP    : RVPBinary1F0W<0b111, 0b10, "zip8hp", 0b010>;
def ZIP16HP   : RVPBinary1F0W<0b111, 0b11, "zip16hp", 0b010>;
} // Predicates = [HasStdExtP, IsRV64]


let Predicates = [HasStdExtP] in {
def PMUL_H_BEE : RVPBinary1LongFW<0b0000, 0b00, "pmul.h.bee", 0b011>;
def PMUL_H_BOO : RVPBinary1LongFW<0b0010, 0b00, "pmul.h.boo", 0b011>;

def PMULU_H_BEE : RVPBinary1LongFW<0b0100, 0b00, "pmulu.h.bee", 0b011>;
def PMULU_H_BOO : RVPBinary1LongFW<0b0110, 0b00, "pmulu.h.boo", 0b011>;

def PMULSU_H_BEE : RVPBinary1LongFW<0b1100, 0b00, "pmulsu.h.bee", 0b011>;
def PMULSU_H_BOO : RVPBinary1LongFW<0b1110, 0b00, "pmulsu.h.boo", 0b011>;
} // Predicates = [HasStdExtP]
let DecoderNamespace = "RISCV32Only_",
    Predicates = [HasStdExtP, IsRV32] in {
def MUL_H00  : RVPBinary1LongFW<0b0000, 0b01, "mul.h00", 0b011>;
def MACC_H00 : RVPBinary1LongFW<0b0001, 0b01, "macc.h00", 0b011>;
def MUL_H11  : RVPBinary1LongFW<0b0010, 0b01, "mul.h11", 0b011>;
def MACC_H11 : RVPBinary1LongFW<0b0011, 0b01, "macc.h11", 0b011>;

def MULU_H00  : RVPBinary1LongFW<0b0100, 0b01, "mulu.h00", 0b011>;
def MACCU_H00 : RVPBinary1LongFW<0b0101, 0b01, "maccu.h00", 0b011>;
def MULU_H11  : RVPBinary1LongFW<0b0110, 0b01, "mulu.h11", 0b011>;
def MACCU_H11 : RVPBinary1LongFW<0b0111, 0b01, "maccu.h11", 0b011>;

def MULSU_H00  : RVPBinary1LongFW<0b1100, 0b01, "mulsu.h00", 0b011>;
def MACCSU_H00 : RVPBinary1LongFW<0b1101, 0b01, "maccsu.h00", 0b011>;
def MULSU_H11  : RVPBinary1LongFW<0b1110, 0b01, "mulsu.h11", 0b011>;
def MACCSU_H11 : RVPBinary1LongFW<0b1111, 0b01, "maccsu.h11", 0b011>;
} // Predicates = [HasStdExtP, IsRV32]

let Predicates = [HasStdExtP, IsRV64] in {
def PMUL_W_BEE  : RVPBinary1LongFW<0b0000, 0b01, "pmul.w.bee", 0b011>;
def MUL_W00     : RVPBinary1LongFW<0b0000, 0b11, "mul.w00", 0b011>;
def PMACC_W_HEE : RVPBinary1LongFW<0b0001, 0b01, "pmacc.w.hee", 0b011>;
def MACC_W00    : RVPBinary1LongFW<0b0001, 0b11, "macc.w00", 0b011>;
def PMUL_W_HOO  : RVPBinary1LongFW<0b0010, 0b01, "pmul.w.hoo", 0b011>;
def MUL_W11     : RVPBinary1LongFW<0b0010, 0b11, "mul.w11", 0b011>;
def PMACC_W_HOO : RVPBinary1LongFW<0b0011, 0b01, "pmacc.w.hoo", 0b011>;
def MACC_W11    : RVPBinary1LongFW<0b0011, 0b11, "macc.w11", 0b011>;

def PMULU_W_HEE  : RVPBinary1LongFW<0b0100, 0b01, "pmulu.w.hee", 0b011>;
def MULU_W00     : RVPBinary1LongFW<0b0100, 0b11, "mulu.w00", 0b011>;
def PMACCU_W_HEE : RVPBinary1LongFW<0b0101, 0b01, "pmaccu.w.hee", 0b011>;
def MACCU_W00    : RVPBinary1LongFW<0b0101, 0b11, "maccu.w00", 0b011>;
def PMULU_W_HOO  : RVPBinary1LongFW<0b0110, 0b01, "pmulu.w.hoo", 0b011>;
def MULU_W11     : RVPBinary1LongFW<0b0110, 0b11, "mulu.w11", 0b011>;
def PMACCU_W_HOO : RVPBinary1LongFW<0b0111, 0b01, "pmaccu.w.hoo", 0b011>;
def MACCU_W11    : RVPBinary1LongFW<0b0111, 0b11, "maccu.w11", 0b011>;

def PMULSU_W_HEE  : RVPBinary1LongFW<0b1100, 0b01, "pmulsu.w.hee", 0b011>;
def MULSU_W00     : RVPBinary1LongFW<0b1100, 0b11, "mulsu.w00", 0b011>;
def PMACCSU_W_HEE : RVPBinary1LongFW<0b1101, 0b01, "pmaccsu.w.hee", 0b011>;
def MACCSU_W00    : RVPBinary1LongFW<0b1101, 0b11, "maccsu.w00", 0b011>;
def PMULSU_W_HOO  : RVPBinary1LongFW<0b1110, 0b01, "pmulsu.w.hoo", 0b011>;
def MULSU_W11     : RVPBinary1LongFW<0b1110, 0b11, "mulsu.w11", 0b011>;
def PMACCSU_W_HOO : RVPBinary1LongFW<0b1111, 0b01, "pmaccsu.w.hoo", 0b011>;
def MACCSU_W11    : RVPBinary1LongFW<0b1111, 0b11, "maccsu.w11", 0b011>;
} // Predicates = [HasStdExtP, IsRV64]


let Predicates = [HasStdExtP] in {
def PPACK_H   : RVPBinary1F0W<0b000, 0b00, "ppack.h", 0b100>;
def PPACKBT_H : RVPBinary1F0W<0b001, 0b00, "ppackbt.h", 0b100>;
def PPACKTB_H : RVPBinary1F0W<0b010, 0b00, "ppacktb.h", 0b100>;
def PPACKT_H  : RVPBinary1F0W<0b011, 0b00, "ppackt.h", 0b100>;
} // Predicates = [HasStdExtP]
let DecoderNamespace = "RISCV32Only_",
    Predicates = [HasStdExtP, IsRV32] in {
def PACKBT : RVPBinary1F0W<0b001, 0b01, "packbt", 0b100>;
def PACKTB : RVPBinary1F0W<0b010, 0b01, "packtb", 0b100>;
def PACKT  : RVPBinary1F0W<0b011, 0b01, "packt", 0b100>;
} // Predicates = [HasStdExtP, IsRV32]

let Predicates = [HasStdExtP, IsRV64] in {
def PPACK_W   : RVPBinary1F0W<0b000, 0b01, "ppack.w", 0b100>;
def PPACKBT_W : RVPBinary1F0W<0b001, 0b01, "ppackbt.w", 0b100>;
def PPACKBT   : RVPBinary1F0W<0b001, 0b11, "ppackbt", 0b100>;
def PPACKTB_W : RVPBinary1F0W<0b010, 0b01, "ppacktb.w", 0b100>;
def PPACKTB   : RVPBinary1F0W<0b010, 0b11, "ppacktb", 0b100>;
def PPACKT_W  : RVPBinary1F0W<0b011, 0b01, "ppackt.w", 0b100>;
def PPACKT    : RVPBinary1F0W<0b011, 0b11, "ppackt", 0b100>;
} // Predicates = [HasStdExtP, IsRV64]


let Predicates = [HasStdExtP] in {
def PM2ADD_H   : RVPBinary1LongFW<0b0000, 0b00, "pm2add.h", 0b101>;
def PM4ADD_B   : RVPBinary1LongFW<0b0000, 0b10, "pm4add.b", 0b101>;
def PM2ADDA_H  : RVPBinary1LongFW<0b0001, 0b00, "pm2adda.h", 0b101>;
def PM4ADDA_B  : RVPBinary1LongFW<0b0001, 0b10, "pm4adda.b", 0b101>;
def PM2ADD_HX  : RVPBinary1LongFW<0b0010, 0b00, "pm2add.hx", 0b101>;
def PM2ADDA_HX : RVPBinary1LongFW<0b0011, 0b00, "pm2adda.hx", 0b101>;

def PM2ADDU_H   : RVPBinary1LongFW<0b0100, 0b00, "pm2addu.h", 0b101>;
def PM4ADDU_B   : RVPBinary1LongFW<0b0100, 0b10, "pm4addu.b", 0b101>;
def PM2ADDAU_H  : RVPBinary1LongFW<0b0101, 0b00, "pm2addau.h", 0b101>;
def PM4ADDAU_B  : RVPBinary1LongFW<0b0101, 0b10, "pm4addau.b", 0b101>;
def PMQ2ADD_H   : RVPBinary1LongFW<0b0110, 0b00, "pmq2add.h", 0b101>;
def PMQR2ADD_H  : RVPBinary1LongFW<0b0110, 0b10, "pmqr2add.h", 0b101>;
def PMQ2ADDA_H  : RVPBinary1LongFW<0b0111, 0b00, "pmq2adda.h", 0b101>;
def PMQR2ADDA_H : RVPBinary1LongFW<0b0111, 0b10, "pmqr2adda.h", 0b101>;

def PM2SUB_H   : RVPBinary1LongFW<0b1000, 0b00, "pm2sub.h", 0b101>;
def PM2SADD_H  : RVPBinary1LongFW<0b1000, 0b10, "pm2sadd.h", 0b101>;
def PM2SUBA_H  : RVPBinary1LongFW<0b1001, 0b00, "pm2suba.h", 0b101>;
def PM2SUB_HX  : RVPBinary1LongFW<0b1010, 0b00, "pm2sub.hx", 0b101>;
def PM2SADD_HX : RVPBinary1LongFW<0b1010, 0b10, "pm2sadd.hx", 0b101>;
def PM2SUBA_HX : RVPBinary1LongFW<0b1011, 0b00, "pm2suba.hx", 0b101>;

def PM2ADDSU_H  : RVPBinary1LongFW<0b1100, 0b00, "pm2addsu.h", 0b101>;
def PM4ADDSU_B  : RVPBinary1LongFW<0b1100, 0b10, "pm4addsu.b", 0b101>;
def PM2ADDASU_H : RVPBinary1LongFW<0b1101, 0b00, "pm2addasu.h", 0b101>;
def PM4ADDASU_B : RVPBinary1LongFW<0b1101, 0b10, "pm4addasu.b", 0b101>;
} // Predicates = [HasStdExtP]
let DecoderNamespace = "RISCV32Only_",
    Predicates = [HasStdExtP, IsRV32] in {
def MQACC_H01  : RVPBinary1LongFW<0b0111, 0b00, "mqacc.h01", 0b101>;
def MQRACC_H01 : RVPBinary1LongFW<0b0111, 0b10, "mqracc.h01", 0b101>;
} // Predicates = [HasStdExtP, IsRV32]

let Predicates = [HasStdExtP, IsRV64] in {
def PM2ADD_W   : RVPBinary1LongFW<0b0000, 0b01, "pm2add.w", 0b101>;
def PM4ADD_H   : RVPBinary1LongFW<0b0000, 0b11, "pm4add.h", 0b101>;
def PM2ADDA_W  : RVPBinary1LongFW<0b0001, 0b01, "pm2adda.w", 0b101>;
def PM4ADDA_H  : RVPBinary1LongFW<0b0001, 0b11, "pm4adda.h", 0b101>;
def PM2ADD_WX  : RVPBinary1LongFW<0b0010, 0b01, "pm2add.wx", 0b101>;
def PM2ADDA_WX : RVPBinary1LongFW<0b0011, 0b01, "pm2adda.wx", 0b101>;

def PM2ADDU_W   : RVPBinary1LongFW<0b0100, 0b01, "pm2addu.w", 0b101>;
def PM4ADDU_H   : RVPBinary1LongFW<0b0100, 0b11, "pm4addu.h", 0b101>;
def PM2ADDAU_W  : RVPBinary1LongFW<0b0101, 0b01, "pm2addau.w", 0b101>;
def PM4ADDAU_H  : RVPBinary1LongFW<0b0101, 0b11, "pm4addau.h", 0b101>;
def PMQ2ADD_W   : RVPBinary1LongFW<0b0110, 0b01, "pmq2add.w", 0b101>;
def PMQR2ADD_W  : RVPBinary1LongFW<0b0110, 0b11, "pmqr2add.w", 0b101>;
def PMQ2ADDA_W  : RVPBinary1LongFW<0b0111, 0b01, "pmq2adda.w", 0b101>;
def PMQR2ADDA_W : RVPBinary1LongFW<0b0111, 0b11, "pmqr2adda.w", 0b101>;

def PM2SUB_W   : RVPBinary1LongFW<0b1000, 0b01, "pm2sub.w", 0b101>;
def PM2SUBA_W  : RVPBinary1LongFW<0b1001, 0b01, "pm2suba.w", 0b101>;
def PM2SUB_WX  : RVPBinary1LongFW<0b1010, 0b01, "pm2sub.wx", 0b101>;
def PM2SUBA_WX : RVPBinary1LongFW<0b1011, 0b01, "pm2suba.wx", 0b101>;

def PM2ADDSU_W  : RVPBinary1LongFW<0b1100, 0b01, "pm2addsu.w", 0b101>;
def PM4ADDSU_H  : RVPBinary1LongFW<0b1100, 0b11, "pm4addsu.h", 0b101>;
def PM2ADDASU_W : RVPBinary1LongFW<0b1101, 0b01, "pm2addasu.w", 0b101>;
def PM4ADDASU_H : RVPBinary1LongFW<0b1101, 0b11, "pm4addasu.h", 0b101>;

def PMQACC_W_HEO  : RVPBinary1LongFW<0b1111, 0b00, "pmqacc.w.heo", 0b101>;
def MQACC_W01     : RVPBinary1LongFW<0b1111, 0b01, "mqacc.w01", 0b101>;
def PMQRACC_W_HEO : RVPBinary1LongFW<0b1111, 0b10, "pmqracc.w.heo", 0b101>;
def MQRACC_W01    : RVPBinary1LongFW<0b1111, 0b11, "mqracc.w01", 0b101>;
} // Predicates = [HasStdExtP, IsRV64]

let Predicates = [HasStdExtP] in {
def PAS_HX  : RVPBinary1LongFW<0b0000, 0b00, "pas.hx", 0b110>;
def PSA_HX  : RVPBinary1LongFW<0b0000, 0b10, "psa.hx", 0b110>;
def PSAS_HX : RVPBinary1LongFW<0b0010, 0b00, "psas.hx", 0b110>;
def PSSA_HX : RVPBinary1LongFW<0b0010, 0b10, "pssa.hx", 0b110>;

def PMSEQ_H  : RVPBinary1LongFW<0b1000, 0b00, "pmseq.h", 0b110>;
def PMSEQ_B  : RVPBinary1LongFW<0b1000, 0b10, "pmseq.b", 0b110>;
def PMSLT_H  : RVPBinary1LongFW<0b1010, 0b00, "pmslt.h", 0b110>;
def PMSLT_B  : RVPBinary1LongFW<0b1010, 0b10, "pmslt.b", 0b110>;
def PMSLTU_H : RVPBinary1LongFW<0b1011, 0b00, "pmsltu.h", 0b110>;
def PMSLTU_B : RVPBinary1LongFW<0b1011, 0b10, "pmsltu.b", 0b110>;

def PMIN_H  : RVPBinary1LongFW<0b1100, 0b00, "pmin.h", 0b110>;
def PMIN_B  : RVPBinary1LongFW<0b1100, 0b10, "pmin.b", 0b110>;
def PMINU_H : RVPBinary1LongFW<0b1101, 0b00, "pminu.h", 0b110>;
def PMINU_B : RVPBinary1LongFW<0b1101, 0b10, "pminu.b", 0b110>;
def PMAX_H  : RVPBinary1LongFW<0b1110, 0b00, "pmax.h", 0b110>;
def PMAX_B  : RVPBinary1LongFW<0b1110, 0b10, "pmax.b", 0b110>;
def PMAXU_H : RVPBinary1LongFW<0b1111, 0b00, "pmaxu.h", 0b110>;
def PMAXU_B : RVPBinary1LongFW<0b1111, 0b10, "pmaxu.b", 0b110>;
} // Predicates = [HasStdExtP]
let DecoderNamespace = "RISCV32Only_",
    Predicates = [HasStdExtP, IsRV32] in {
def MSEQ  : RVPBinary1LongFW<0b1000, 0b01, "mseq", 0b110>;
def MSLT  : RVPBinary1LongFW<0b1010, 0b01, "mslt", 0b110>;
def MSLTU : RVPBinary1LongFW<0b1011, 0b01, "msltu", 0b110>;
} // Predicates = [HasStdExtP, IsRV32]
let Predicates = [HasStdExtP, IsRV64] in {
def PAS_WX  : RVPBinary1LongFW<0b0000, 0b01, "pas.wx", 0b110>;
def PSA_WX  : RVPBinary1LongFW<0b0000, 0b11, "psa.wx", 0b110>;
def PSAS_WX : RVPBinary1LongFW<0b0010, 0b01, "psas.wx", 0b110>;
def PSSA_WX : RVPBinary1LongFW<0b0010, 0b11, "pssa.wx", 0b110>;
def PAAS_WX : RVPBinary1LongFW<0b0011, 0b01, "paas.wx", 0b110>;
def PASA_WX : RVPBinary1LongFW<0b0011, 0b11, "pasa.wx", 0b110>;

def PMSEQ_W  : RVPBinary1LongFW<0b1000, 0b01, "pmseq.w", 0b110>;
def PMSLT_W  : RVPBinary1LongFW<0b1010, 0b01, "pmslt.w", 0b110>;
def PMSLTU_W : RVPBinary1LongFW<0b1011, 0b01, "pmsltu.w", 0b110>;

def PMIN_W  : RVPBinary1LongFW<0b1100, 0b01, "pmin.w", 0b110>;
def PMINU_W : RVPBinary1LongFW<0b1101, 0b01, "pminu.w", 0b110>;
def PMAX_W  : RVPBinary1LongFW<0b1110, 0b01, "pmax.w", 0b110>;
def PMAXU_W : RVPBinary1LongFW<0b1111, 0b01, "pmaxu.w", 0b110>;
} // Predicates = [HasStdExtP, IsRV64]


let Predicates = [HasStdExtP] in {
def PMULH_H    : RVPBinary1LongFW<0b0000, 0b00, "pmulh.h", 0b111>;
def PMULHR_H   : RVPBinary1LongFW<0b0000, 0b10, "pmulhr.h", 0b111>;
def PMHACC_H   : RVPBinary1LongFW<0b0001, 0b00, "pmhacc.h", 0b111>;
def PMHRACC_H  : RVPBinary1LongFW<0b0001, 0b10, "pmhracc.h", 0b111>;
def PMULHU_H   : RVPBinary1LongFW<0b0010, 0b00, "pmulhu.h", 0b111>;
def PMULHRU_H  : RVPBinary1LongFW<0b0010, 0b10, "pmulhru.h", 0b111>;
def PMHACCU_H  : RVPBinary1LongFW<0b0011, 0b00, "pmhaccu.h", 0b111>;
def PMHRACCU_H : RVPBinary1LongFW<0b0011, 0b10, "pmhraccu.h", 0b111>;

def PMULH_H_BE    : RVPBinary1LongFW<0b0100, 0b00, "pmulh.h.be", 0b111>;
def PMULHSU_H_BE  : RVPBinary1LongFW<0b0100, 0b10, "pmulhsu.h.be", 0b111>;
def PMHACCU_H_BE  : RVPBinary1LongFW<0b0101, 0b00, "pmhaccu.h.be", 0b111>;
def PMHACCSU_H_BE : RVPBinary1LongFW<0b0101, 0b10, "pmhaccsu.h.be", 0b111>;
def PMULH_H_BO    : RVPBinary1LongFW<0b0110, 0b00, "pmulh.h.bo", 0b111>;
def PMULHSU_H_BO  : RVPBinary1LongFW<0b0110, 0b10, "pmulhsu.h.bo", 0b111>;
def PMHACC_H_BO   : RVPBinary1LongFW<0b0111, 0b00, "pmhacc.h.bo", 0b111>;
def PMHACCSU_H_BO : RVPBinary1LongFW<0b0111, 0b10, "pmhaccsu.h.bo", 0b111>;

def PMULHSU_H   : RVPBinary1LongFW<0b1000, 0b00, "pmulhsu.h", 0b111>;
def PMULHRSU_H  : RVPBinary1LongFW<0b1000, 0b10, "pmulhrsu.h", 0b111>;
def PMHACCSU_H  : RVPBinary1LongFW<0b1001, 0b00, "pmhaccsu.h", 0b111>;
def PMHRACCSU_H : RVPBinary1LongFW<0b1001, 0b10, "pmhraccsu.h", 0b111>;
def PMULQ_H     : RVPBinary1LongFW<0b1010, 0b00, "pmulq.h", 0b111>;
def PMULQR_H    : RVPBinary1LongFW<0b1010, 0b10, "pmulqr.h", 0b111>;
} // Predicates = [HasStdExtP]

let DecoderNamespace = "RISCV32Only_",
    Predicates = [HasStdExtP, IsRV32] in {
def MULHR   : RVPBinary1LongFW<0b0000, 0b11, "mulhr", 0b111>;
def MHACC   : RVPBinary1LongFW<0b0001, 0b01, "mhacc", 0b111>;
def MHRACC  : RVPBinary1LongFW<0b0001, 0b11, "mhracc", 0b111>;
def MULHRU  : RVPBinary1LongFW<0b0010, 0b11, "mulhru", 0b111>;
def MHACCU  : RVPBinary1LongFW<0b0011, 0b01, "mhaccu", 0b111>;
def MHRACCU : RVPBinary1LongFW<0b0011, 0b11, "mhraccu", 0b111>;

def MULH_H0    : RVPBinary1LongFW<0b0100, 0b01, "mulh.h0", 0b111>;
def MULHSU_H0  : RVPBinary1LongFW<0b0100, 0b11, "mulhsu.h0", 0b111>;
def MHACC_H0   : RVPBinary1LongFW<0b0101, 0b01, "mhacc.h0", 0b111>;
def MHACCSU_H0 : RVPBinary1LongFW<0b0101, 0b11, "mhaccsu.h0", 0b111>;
def MULH_H1    : RVPBinary1LongFW<0b0110, 0b01, "mulh.h1", 0b111>;
def MULHSU_H1  : RVPBinary1LongFW<0b0110, 0b11, "mulhsu.h1", 0b111>;
def MHACC_H1   : RVPBinary1LongFW<0b0111, 0b01, "mhacc.h1", 0b111>;
def MHACCSU_H1 : RVPBinary1LongFW<0b0111, 0b11, "mhaccsu.h1", 0b111>;

def MULHRSU_H : RVPBinary1LongFW<0b1000, 0b11, "mulhrsu.h", 0b111>;
def MHACCSU   : RVPBinary1LongFW<0b1001, 0b01, "mhaccsu", 0b111>;
def MHRACCSU  : RVPBinary1LongFW<0b1001, 0b11, "mhraccsu", 0b111>;
def MULQ      : RVPBinary1LongFW<0b1010, 0b01, "mulq", 0b111>;
def MULQR     : RVPBinary1LongFW<0b1010, 0b11, "mulqr", 0b111>;

def MQACC_H00  : RVPBinary1LongFW<0b1101, 0b00, "mqacc.h00", 0b111>;
def MQRACC_H00 : RVPBinary1LongFW<0b1101, 0b10, "mqracc.h00", 0b111>;
def MQACC_H11  : RVPBinary1LongFW<0b1111, 0b00, "mqacc.h11", 0b111>;
def MQRACC_H11 : RVPBinary1LongFW<0b1111, 0b10, "mqracc.h11", 0b111>;
} // Predicates = [HasStdExtP, IsRV32]

let Predicates = [HasStdExtP, IsRV64] in {
def PMULH_W    : RVPBinary1LongFW<0b0000, 0b01, "pmulh.w", 0b111>;
def PMULHR_W   : RVPBinary1LongFW<0b0000, 0b11, "pmulhr.w", 0b111>;
def PMHACC_W   : RVPBinary1LongFW<0b0001, 0b01, "pmhacc.w", 0b111>;
def PMHRACC_W  : RVPBinary1LongFW<0b0001, 0b11, "pmhracc.w", 0b111>;
def PMULHU_W   : RVPBinary1LongFW<0b0010, 0b01, "pmulhu.w", 0b111>;
def PMULHRU_W  : RVPBinary1LongFW<0b0010, 0b11, "pmulhru.w", 0b111>;
def PMHACCU_W  : RVPBinary1LongFW<0b0011, 0b01, "pmhaccu.w", 0b111>;
def PMHRACCU_W : RVPBinary1LongFW<0b0011, 0b11, "pmhraccu.w", 0b111>;

def PMULH_W_HE    : RVPBinary1LongFW<0b0100, 0b01, "pmulh.w.he", 0b111>;
def PMULHSU_W_HE  : RVPBinary1LongFW<0b0100, 0b11, "pmulhsu.w.he", 0b111>;
def PMHACC_W_HE   : RVPBinary1LongFW<0b0101, 0b01, "pmhacc.w.he", 0b111>;
def PMHACCSU_W_HE : RVPBinary1LongFW<0b0101, 0b11, "pmhaccsu.w.he", 0b111>;
def PMULH_W_HO    : RVPBinary1LongFW<0b0110, 0b01, "pmulh.w.ho", 0b111>;
def PMULHSU_W_HO  : RVPBinary1LongFW<0b0110, 0b11, "pmulhsu.w.ho", 0b111>;
def PMHACC_W_HO   : RVPBinary1LongFW<0b0111, 0b01, "pmhacc.w.ho", 0b111>;
def PMHACCSU_W_HO : RVPBinary1LongFW<0b0111, 0b11, "pmhaccsu.w.ho", 0b111>;

def PMULHSU_W   : RVPBinary1LongFW<0b1000, 0b01, "pmulhsu.w", 0b111>;
def PMULHRSU_W  : RVPBinary1LongFW<0b1000, 0b11, "pmulhrsu.w", 0b111>;
def PMHACCSU_W  : RVPBinary1LongFW<0b1001, 0b01, "pmhaccsu.w", 0b111>;
def PMHRACCSU_W : RVPBinary1LongFW<0b1001, 0b11, "pmhraccsu.w", 0b111>;
def PMULQ_W     : RVPBinary1LongFW<0b1010, 0b01, "pmulq.w", 0b111>;
def PMULQR_W    : RVPBinary1LongFW<0b1010, 0b11, "pmulqr.w", 0b111>;

def PMQACC_W_HEE  : RVPBinary1LongFW<0b1101, 0b00, "pmqacc.w.hee", 0b111>;
def MQACC_W00     : RVPBinary1LongFW<0b1101, 0b01, "mqacc.w00", 0b111>;
def PMQRACC_W_HEE : RVPBinary1LongFW<0b1101, 0b10, "pmqracc.w.hee", 0b111>;
def MQRACC_W00    : RVPBinary1LongFW<0b1101, 0b11, "mqracc.w00", 0b111>;
def PMQACC_W_HOO  : RVPBinary1LongFW<0b1111, 0b00, "pmqacc.w.hoo", 0b111>;
def MQACC_W11     : RVPBinary1LongFW<0b1111, 0b01, "mqacc.w11", 0b111>;
def PMQRACC_W_HOO : RVPBinary1LongFW<0b1111, 0b10, "pmqracc.w.hoo", 0b111>;
def MQRACC_W11    : RVPBinary1LongFW<0b1111, 0b11, "mqracc.w11", 0b111>;
} // Predicates = [HasStdExtP, IsRV64]


let Predicates = [HasStdExtP, IsRV32] in {
def PWSLLI_B : RVPUnary0F0Rdp<0b000, 0b0010000, "pwslli.b">;
def PWSLLI_H : RVPUnary0F0Rdp<0b001, 0b0100000, "pwslli.h">;
def WSLLI    : RVPUnary0F0Rdp<0b010, 0b1000000, "wslli">;

def PWSLAI_B : RVPUnary0F0Rdp<0b100, 0b0010000, "pwslai.b">;
def PWSLAI_H : RVPUnary0F0Rdp<0b100, 0b0100000, "pwslai.h">;
def WSLAI    : RVPUnary0F0Rdp<0b100, 0b1000000, "wslai">;

def PLI_DH  : RVPUnaryImm9<0b0011000, OPC_OP_IMM_32, "pli.dh", GPRPairRV32>;
def PLI_DB  : RVPUnaryImm8<0b00110100, OPC_OP_IMM_32, "pli.db", GPRPairRV32>;
def PLUI_DH : RVPUnaryImm9<0b0111000, OPC_OP_IMM_32, "plui.dh", GPRPairRV32>;


def PWSLLI_B_B0 : RVPBinary0F1WRdp<0b000, 0b00, "pwslli.b.b0">;
def PWSLL_H_H0  : RVPBinary0F1WRdp<0b000, 0b01, "pwsll.h.h0">;
def WSLL        : RVPBinary0F1WRdp<0b000, 0b11, "wsll">;

def PWSLA_B_B0 : RVPBinary0F1WRdp<0b100, 0b00, "pwsla.b.b0">;
def PWSLA_H_H0 : RVPBinary0F1WRdp<0b100, 0b01, "pwsla.h.h0">;
def WSLA       : RVPBinary0F1WRdp<0b100, 0b11, "wsla">;
def WZIP8P     : RVPBinary0F1WRdp<0b111, 0b00, "wzip8p">;
def WZIP16P    : RVPBinary0F1WRdp<0b111, 0b01, "wzip16p">;

def PWADD_H    : RVPBinary0LongFW<0b0000, 0b00, "pwadd.h">;
def WADD       : RVPBinary0LongFW<0b0000, 0b01, "wadd">;
def PWADD_B    : RVPBinary0LongFW<0b0000, 0b10, "pwadd.b">;
def PW2WADD_H  : RVPBinary0LongFW<0b0000, 0b11, "pw2wadd.h">;
def PWADDA_H   : RVPBinary0LongFW<0b0001, 0b00, "pwadda.h">;
def WADDA      : RVPBinary0LongFW<0b0001, 0b01, "wadda">;
def PWADDA_B   : RVPBinary0LongFW<0b0001, 0b10, "pwadda.b">;
def PW2WADDA_H : RVPBinary0LongFW<0b0001, 0b11, "pw2wadda.h">;
def PWADDU_H   : RVPBinary0LongFW<0b0010, 0b00, "pwaddu.h">;
def WADDU      : RVPBinary0LongFW<0b0010, 0b01, "waddu">;
def PWADDU_B   : RVPBinary0LongFW<0b0010, 0b10, "pwaddu.b">;
def PW2WADD_HX : RVPBinary0LongFW<0b0010, 0b11, "pw2wadd.hx">;
def PWADDAU_H  : RVPBinary0LongFW<0b0011, 0b00, "pwaddau.h">;
def WADDAU     : RVPBinary0LongFW<0b0011, 0b01, "waddau">;
def PWADDAU_B  : RVPBinary0LongFW<0b0011, 0b10, "pwaddau.b">;
def PW2WADDA_HX: RVPBinary0LongFW<0b0011, 0b11, "pw2wadda.hx">;

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

def PWSUB_H    : RVPBinary0LongFW<0b1000, 0b00, "pwsub.h">;
def WSUB       : RVPBinary0LongFW<0b1000, 0b01, "wsub">;
def PWSUB_B    : RVPBinary0LongFW<0b1000, 0b10, "pwsub.b">;
def PW2WSUB_H  : RVPBinary0LongFW<0b1000, 0b11, "pw2wsub.h">;
def PWSUBA_H   : RVPBinary0LongFW<0b1001, 0b00, "pwsuba.h">;
def WSUBA      : RVPBinary0LongFW<0b1001, 0b01, "wsuba">;
def PWSUBA_B   : RVPBinary0LongFW<0b1001, 0b10, "pwsuba.b">;
def PW2WSUBA_H : RVPBinary0LongFW<0b1001, 0b11, "pw2wsuba.h">;
def PWSUBU_H   : RVPBinary0LongFW<0b1010, 0b00, "pwsubu.h">;
def WSUBU      : RVPBinary0LongFW<0b1010, 0b01, "wsubu">;
def PWSUBU_B   : RVPBinary0LongFW<0b1010, 0b10, "pwsubu.b">;
def PW2WSUB_HX : RVPBinary0LongFW<0b1010, 0b11, "pw2wsub.hx">;
def PWSUBAU_H  : RVPBinary0LongFW<0b1011, 0b00, "pwsubau.h">;
def WSUBAU     : RVPBinary0LongFW<0b1011, 0b01, "wsubau">;
def PWSUBAU_B  : RVPBinary0LongFW<0b1011, 0b10, "pwsubau.b">;
def PW2WSUBA_HX: RVPBinary0LongFW<0b1011, 0b11, "pw2wsuba.hx">;

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

def PREDSUM_DH   : RVPBinary0F1WRs1p<0b000, 0b00, "predsum.dh">;
def PREDSUM_DB   : RVPBinary0F1WRs1p<0b000, 0b10, "predsum.db">;
def PREDSUMU_DH  : RVPBinary0F1WRs1p<0b011, 0b00, "predsumu.dh">;
def PREDSUMU_DB  : RVPBinary0F1WRs1p<0b011, 0b10, "predsumu.db">;

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

def PNSRL_B_B0    : RVPBinary0F1WRs1p<0b000, 0b00, "pnsrl.b.b0", 0b1>;
def PNSRL_H_H0    : RVPBinary0F1WRs1p<0b000, 0b01, "pnsrl.h.h0", 0b1>;
def NSRL          : RVPBinary0F1WRs1p<0b000, 0b11, "nsrl", 0b1>;
def PNCLIPU_B_B0  : RVPBinary0F1WRs1p<0b010, 0b00, "pnclipu.b.b0", 0b1>;
def PNCLIPU_H_H0  : RVPBinary0F1WRs1p<0b010, 0b01, "pnclipu.h.h0", 0b1>;
def NCLIPU        : RVPBinary0F1WRs1p<0b010, 0b11, "nclipu", 0b1>;
def PNCLIPRU_B_B0 : RVPBinary0F1WRs1p<0b011, 0b00, "pnclipru.b.b0", 0b1>;
def PNCLIPRU_H_H0 : RVPBinary0F1WRs1p<0b011, 0b01, "pnclipru.h.h0", 0b1>;
def NCLIPRU       : RVPBinary0F1WRs1p<0b011, 0b11, "nclipru", 0b1>;

def PNSRA_B_B0    : RVPBinary0F1WRs1p<0b100, 0b00, "pnsra.b.b0", 0b1>;
def PNSRA_H_H0    : RVPBinary0F1WRs1p<0b100, 0b01, "pnsra.h.h0", 0b1>;
def NSRA          : RVPBinary0F1WRs1p<0b100, 0b11, "nsra", 0b1>;
def PNSRAR_B_B0   : RVPBinary0F1WRs1p<0b101, 0b00, "pnsrar.b.b0", 0b1>;
def PNSRAR_H_H0   : RVPBinary0F1WRs1p<0b101, 0b01, "pnsrar.h.h0", 0b1>;
def NSRAR         : RVPBinary0F1WRs1p<0b101, 0b11, "nsrar", 0b1>;
def PNCLIP_B_B0   : RVPBinary0F1WRs1p<0b110, 0b00, "pnclip.b.b0", 0b1>;
def PNCLIP_H_H0   : RVPBinary0F1WRs1p<0b110, 0b01, "pnclip.h.h0", 0b1>;
def NCLIP         : RVPBinary0F1WRs1p<0b110, 0b11, "nclip", 0b1>;
def PNCLIPR_B_B0  : RVPBinary0F1WRs1p<0b111, 0b00, "pnclipr.b.b0", 0b1>;
def PNCLIPR_H_H0  : RVPBinary0F1WRs1p<0b111, 0b01, "pnclipr.h.h0", 0b1>;
def NCLIPR        : RVPBinary0F1WRs1p<0b111, 0b11, "nclipr", 0b1>;


def PSLLI_DB      : RVPUnary0F0Rs1pRdp<0b000, 0b0001000, "pslli.db">;
def PSLLI_DH      : RVPUnary0F0Rs1pRdp<0b000, 0b0010000, "pslli.dh">;
def PSLLI_DW      : RVPUnary0F0Rs1pRdp<0b000, 0b0100000, "pslli.dw">;

def PSSLAI_DH    : RVPUnary0F0Rs1pRdp<0b101, 0b0010000, "psslai.dh">;
def PSSLAI_DW    : RVPUnary0F0Rs1pRdp<0b101, 0b0100000, "psslai.dw">;

def PSEXTB_DH    : RVPUnaryWUFRs1pRdp<0b00, 0b00100, "psextb.dh">;
def PSEXTB_DW    : RVPUnaryWUFRs1pRdp<0b01, 0b00100, "psextb.dw">;
def PSEXTW_DW    : RVPUnaryWUFRs1pRdp<0b01, 0b00101, "psextw.dw">;
def PSABS_DH     : RVPUnaryWUFRs1pRdp<0b00, 0b00111, "psabs.dh">;
def PSABS_DB     : RVPUnaryWUFRs1pRdp<0b10, 0b00111, "psabs.db">;

def PSLL_DH_H0   : RVPBinary0F1WRs1pRdp<0b000, 0b00, "psll.dh.h0">;
def PSLL_DW_W    : RVPBinary0F1WRs1pRdp<0b000, 0b01, "psll.dw.w">;
def PSLL_DB_B0   : RVPBinary0F1WRs1pRdp<0b000, 0b10, "psll.db.b0">;
def PADD_DH_H0   : RVPBinary0F1WRs1pRdp<0b001, 0b00, "padd.dh.h0">;
def PADD_DW_W    : RVPBinary0F1WRs1pRdp<0b001, 0b01, "padd.dw.w">;
def PADD_DB_B0   : RVPBinary0F1WRs1pRdp<0b001, 0b10, "padd.db.b0">;

def PSSHA_DH_H0  : RVPBinary0F1WRs1pRdp<0b110, 0b00, "pssha.dh.h0">;
def PSSHA_DW_W   : RVPBinary0F1WRs1pRdp<0b110, 0b01, "pssha.dw.w">;
def PSSHAR_DH_H0 : RVPBinary0F1WRs1pRdp<0b111, 0b00, "psshar.dh.h0">;
def PSSHAR_DW_W  : RVPBinary0F1WRs1pRdp<0b111, 0b01, "psshar.dw.w">;

def PSRLI_DB    : RVPUnary0F0Rs1pRdp<0b000, 0b0001000, "psrli.db", 0b1>;
def PSRLI_DH    : RVPUnary0F0Rs1pRdp<0b000, 0b0010000, "psrli.dh", 0b1>;
def PSRLI_DW    : RVPUnary0F0Rs1pRdp<0b000, 0b0100000, "psrli.dw", 0b1>;
def PUSATI_DH   : RVPUnary0F0Rs1pRdp<0b010, 0b0010000, "pusati.dh", 0b1>;
def PUSATI_DW   : RVPUnary0F0Rs1pRdp<0b010, 0b0100000, "pusati.dw", 0b1>;

def PSRAI_DB    : RVPUnary0F0Rs1pRdp<0b100, 0b0001000, "psrai.db", 0b1>;
def PSRAI_DH    : RVPUnary0F0Rs1pRdp<0b100, 0b0010000, "psrai.dh", 0b1>;
def PSRAI_DW    : RVPUnary0F0Rs1pRdp<0b100, 0b0100000, "psrai.dw", 0b1>;
def PSRARI_DH   : RVPUnary0F0Rs1pRdp<0b101, 0b0010000, "psrari.dh", 0b1>;
def PSRARI_DW   : RVPUnary0F0Rs1pRdp<0b101, 0b0100000, "psrari.dw", 0b1>;
def PSATI_DH    : RVPUnary0F0Rs1pRdp<0b110, 0b0010000, "psati.dh", 0b1>;
def PSATI_DW    : RVPUnary0F0Rs1pRdp<0b110, 0b0100000, "psati.dw", 0b1>;

def PSRL_DH_H0  :RVPBinary0F1WRs1pRdp<0b000, 0b00, "psrl.dh.h0", 0b1>;
def PSRL_DW_W   :RVPBinary0F1WRs1pRdp<0b000, 0b01, "psrl.dw.w", 0b1>;
def PSRL_DB_B0  :RVPBinary0F1WRs1pRdp<0b000, 0b10, "psrl.db.b0", 0b1>;

def PSRA_DH_H0  :RVPBinary0F1WRs1pRdp<0b100, 0b00, "psra.dh.h0", 0b1>;
def PSRA_DW_W   :RVPBinary0F1WRs1pRdp<0b100, 0b01, "psra.dw.w", 0b1>;
def PSRA_DB_B0  :RVPBinary0F1WRs1pRdp<0b100, 0b10, "psra.db.b0", 0b1>;

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

def PPACK_DH     : RVPBinary1F0WRs2pRs1pRdp<0b000, 0b00, "ppack.dh", 0b0, 0b1>;
def PPACK_DW     : RVPBinary1F0WRs2pRs1pRdp<0b000, 0b01, "ppack.dw", 0b0, 0b1>;
def PPACKBT_DH   : RVPBinary1F0WRs2pRs1pRdp<0b001, 0b00, "ppackbt.dh", 0b0, 0b1>;
def PPACKBT_DW   : RVPBinary1F0WRs2pRs1pRdp<0b001, 0b01, "ppackbt.dw", 0b0, 0b1>;
def PPACKTB_DH   : RVPBinary1F0WRs2pRs1pRdp<0b010, 0b00, "ppacktb.dh", 0b0, 0b1>;
def PPACKTB_DW   : RVPBinary1F0WRs2pRs1pRdp<0b010, 0b01, "ppacktb.dw", 0b0, 0b1>;
def PPACKT_DH    : RVPBinary1F0WRs2pRs1pRdp<0b011, 0b00, "ppackt.dh", 0b0, 0b1>;
def PPACKT_DW    : RVPBinary1F0WRs2pRs1pRdp<0b011, 0b01, "ppackt.dw", 0b0, 0b1>;

def PAS_DHX      : RVPBinary1FWRs2pRs1pRdp<0b0000, 0b00, "pas.dhx", 0b1>;
def PSA_DHX      : RVPBinary1FWRs2pRs1pRdp<0b0000, 0b10, "psa.dhx", 0b1>;
def PSAS_DHX     : RVPBinary1FWRs2pRs1pRdp<0b0010, 0b00, "psas.dhx", 0b1>;
def PSSA_DHX     : RVPBinary1FWRs2pRs1pRdp<0b0010, 0b10, "pssa.dhx", 0b1>;
def PAAX_DHX     : RVPBinary1FWRs2pRs1pRdp<0b0011, 0b00, "paax.dhx", 0b1>;
def PASA_DHX     : RVPBinary1FWRs2pRs1pRdp<0b0011, 0b10, "pasa.dhx", 0b1>;

def PMSEQ_DH     : RVPBinary1FWRs2pRs1pRdp<0b1000, 0b00, "pmseq.dh", 0b1>;
def PMSEQ_DW     : RVPBinary1FWRs2pRs1pRdp<0b1000, 0b01, "pmseq.dw", 0b1>;
def PMSEQ_DB     : RVPBinary1FWRs2pRs1pRdp<0b1000, 0b10, "pmseq.db", 0b1>;
def PMSLT_DH     : RVPBinary1FWRs2pRs1pRdp<0b1010, 0b00, "pmslt.dh", 0b1>;
def PMSLT_DW     : RVPBinary1FWRs2pRs1pRdp<0b1010, 0b01, "pmslt.dw", 0b1>;
def PMSLT_DB     : RVPBinary1FWRs2pRs1pRdp<0b1010, 0b10, "pmslt.db", 0b1>;
def PMSLTU_DH    : RVPBinary1FWRs2pRs1pRdp<0b1011, 0b00, "pmsltu.dh", 0b1>;
def PMSLTU_DW    : RVPBinary1FWRs2pRs1pRdp<0b1011, 0b01, "pmsltu.dw", 0b1>;
def PMSLTU_DB    : RVPBinary1FWRs2pRs1pRdp<0b1011, 0b10, "pmsltu.db", 0b1>;

def PMIN_DH      : RVPBinary1FWRs2pRs1pRdp<0b1100, 0b00, "pmin.dh", 0b1>;
def PMIN_DB      : RVPBinary1FWRs2pRs1pRdp<0b1100, 0b10, "pmin.db", 0b1>;
def PMINU_DH     : RVPBinary1FWRs2pRs1pRdp<0b1101, 0b00, "pminu.dh", 0b1>;
def PMINU_DB     : RVPBinary1FWRs2pRs1pRdp<0b1101, 0b10, "pminu.db", 0b1>;
def PMAX_DH      : RVPBinary1FWRs2pRs1pRdp<0b1110, 0b00, "pmax.dh", 0b1>;
def PMAX_DB      : RVPBinary1FWRs2pRs1pRdp<0b1110, 0b10, "pmax.db", 0b1>;
def PMAXU_DH     : RVPBinary1FWRs2pRs1pRdp<0b1111, 0b00, "pmaxu.dh", 0b1>;
def PMAXU_DB     : RVPBinary1FWRs2pRs1pRdp<0b1111, 0b10, "pmaxu.db", 0b1>;
} // Predicates = [HasStdExtP, IsRV32]

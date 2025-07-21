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
class RVPUnaryImm10<bits<7> funct7, string opcodestr,
                    DAGOperand TyImm10 = simm10>
    : RVInstIBase<0b010, OPC_OP_IMM_32, (outs GPR:$rd), (ins TyImm10:$imm10),
                  opcodestr, "$rd, $imm10"> {
  bits<10> imm10;

  let Inst{31-25} = funct7;
  let Inst{24-16} = imm10{8-0};
  let Inst{15}    = imm10{9};
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
class RVPUnary<bits<3> f, string opcodestr, dag operands, string argstr>
    : RVInstIBase<0b010, OPC_OP_IMM_32, (outs GPR:$rd), operands, opcodestr, argstr> {
  let Inst{31}    = 0b1;
  let Inst{30-28} = f;
  let Inst{27}    = 0b0;
}

class RVPUnaryImm5<bits<3> f, string opcodestr>
    : RVPUnary<f, opcodestr, (ins GPR:$rs1, uimm5:$uimm5), "$rd, $rs1, $uimm5"> {
  bits<5> uimm5;

  let Inst{26-25} = 0b01;
  let Inst{24-20} = uimm5;
}

class RVPUnaryImm4<bits<3> f, string opcodestr>
    : RVPUnary<f, opcodestr, (ins GPR:$rs1, uimm4:$uimm4), "$rd, $rs1, $uimm4"> {
  bits<4> uimm4;

  let Inst{26-24} = 0b001;
  let Inst{23-20} = uimm4;
}

class RVPUnaryImm3<bits<3> f, string opcodestr>
    : RVPUnary<f, opcodestr, (ins GPR:$rs1, uimm3:$uimm3), "$rd, $rs1, $uimm3"> {
  bits<3> uimm3;

  let Inst{26-23} = 0b0001;
  let Inst{22-20} = uimm3;
}

let hasSideEffects = 0, mayLoad = 0, mayStore = 0 in
class RVPUnaryWUF<bits<2> w, bits<5> uf, string opcodestr>
    : RVInstIBase<0b010, OPC_OP_IMM_32, (outs GPR:$rd), (ins GPR:$rs1),
                  opcodestr, "$rd, $rs1">  {
  let Inst{31-27} = 0b11100;
  let Inst{26-25} = w;
  let Inst{24-20} = uf;
}

//===----------------------------------------------------------------------===//
// Instructions
//===----------------------------------------------------------------------===//

let Predicates = [HasStdExtP] in {
def CLS    : Unary_r<0b011000000011, 0b001, "cls">;
def ABS    : Unary_r<0b011000000111, 0b001, "abs">;
} // Predicates = [HasStdExtP]
let Predicates = [HasStdExtP, IsRV32] in
def REV_RV32  : Unary_r<0b011010011111, 0b101, "rev">;

let Predicates = [HasStdExtP, IsRV64] in {
def REV16      : Unary_r<0b011010110000, 0b101, "rev16">;
def REV_RV64   : Unary_r<0b011010111111, 0b101, "rev">;

def CLSW  : UnaryW_r<0b011000000011, 0b001, "clsw">;
def ABSW  : UnaryW_r<0b011000000111, 0b001, "absw">;
} // Predicates = [HasStdExtP, IsRV64]

let Predicates = [HasStdExtP] in {
def PSLLI_B  : RVPUnaryImm3<0b000, "pslli.b">;
def PSLLI_H  : RVPUnaryImm4<0b000, "pslli.h">;
def PSSLAI_H : RVPUnaryImm4<0b101, "psslai.h">;
} // Predicates = [HasStdExtP]
let DecoderNamespace = "RV32Only",
    Predicates = [HasStdExtP, IsRV32] in
def SSLAI    : RVPUnaryImm5<0b101, "sslai">;
let Predicates = [HasStdExtP, IsRV64] in {
def PSLLI_W  : RVPUnaryImm5<0b000, "pslli.w">;
def PSSLAI_W : RVPUnaryImm5<0b101, "psslai.w">;
} // Predicates = [HasStdExtP, IsRV64]

let Predicates = [HasStdExtP] in
def PLI_H : RVPUnaryImm10<0b1011000, "pli.h">;
let Predicates = [HasStdExtP, IsRV64] in
def PLI_W : RVPUnaryImm10<0b1011001, "pli.w">;
let Predicates = [HasStdExtP] in
def PLI_B : RVPUnaryImm8<0b10110100, "pli.b">;

let Predicates = [HasStdExtP] in {
def PSEXT_H_B : RVPUnaryWUF<0b00, 0b00100, "psext.h.b">;
def PSABS_H   : RVPUnaryWUF<0b00, 0b00111, "psabs.h">;
def PSABS_B   : RVPUnaryWUF<0b10, 0b00111, "psabs.b">;
} // Predicates = [HasStdExtP]
let Predicates = [HasStdExtP, IsRV64] in {
def PSEXT_W_B      : RVPUnaryWUF<0b01, 0b00100, "psext.w.b">;
def PSEXT_W_H      : RVPUnaryWUF<0b01, 0b00101, "psext.w.h">;
} // Predicates = [HasStdExtP, IsRV64]

let Predicates = [HasStdExtP] in
def PLUI_H : RVPUnaryImm10<0b1111000, "plui.h", simm10_unsigned>;
let Predicates = [HasStdExtP, IsRV64] in
def PLUI_W : RVPUnaryImm10<0b1111001, "plui.w", simm10_unsigned>;

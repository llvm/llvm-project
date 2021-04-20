//===-- CSKYInstrInfo.td - Target Description for CSKY -----*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file describes the CSKY instructions in TableGen format.
//
//===----------------------------------------------------------------------===//


//===----------------------------------------------------------------------===//
// CSKY specific DAG Nodes.
//===----------------------------------------------------------------------===//

// TODO: Add CSKY specific DAG Nodes.

//===----------------------------------------------------------------------===//
// Operand and SDNode transformation definitions.
//===----------------------------------------------------------------------===//
class ImmAsmOperand<string prefix, int width, string suffix> : AsmOperandClass {
  let Name = prefix # "Imm" # width # suffix;
  let RenderMethod = "addImmOperands";
  let DiagnosticType = !strconcat("Invalid", Name);
}

class SImmAsmOperand<int width, string suffix = "">
    : ImmAsmOperand<"S", width, suffix> {
}

class UImmAsmOperand<int width, string suffix = "">
    : ImmAsmOperand<"U", width, suffix> {
}

class OImmAsmOperand<int width, string suffix = "">
    : ImmAsmOperand<"O", width, suffix> {
}

class oimm<int num> : Operand<i32>,
  ImmLeaf<i32, "return isUInt<"#num#">(Imm - 1);"> {
  let EncoderMethod = "getOImmOpValue";
  let ParserMatchClass = OImmAsmOperand<num>;
}

class uimm<int num, int shift = 0> : Operand<i32>,
  ImmLeaf<i32, "return isShiftedUInt<"#num#", "#shift#">(Imm);"> {
  let EncoderMethod = "getImmOpValue<"#shift#">";
  let ParserMatchClass =
    !if(!ne(shift, 0),
        UImmAsmOperand<num, "Shift"#shift>,
        UImmAsmOperand<num>);
}

class simm<int num, int shift = 0> : Operand<i32>,
  ImmLeaf<i32, "return isShiftedInt<"#num#", "#shift#">(Imm);"> {
  let EncoderMethod = "getImmOpValue<"#shift#">";
  let ParserMatchClass = SImmAsmOperand<num>;
}

def nimm_XFORM : SDNodeXForm<imm, [{
  return CurDAG->getTargetConstant(~N->getSExtValue(), SDLoc(N), MVT::i32);
}]>;
class nimm<int num> : Operand<i32>,
  ImmLeaf<i32, "return isUInt<"#num#">(~Imm);", nimm_XFORM> {
  let ParserMatchClass = UImmAsmOperand<num>;
}

def uimm32_hi16 : SDNodeXForm<imm, [{
  return CurDAG->getTargetConstant((N->getZExtValue() >> 16) & 0xFFFF,
    SDLoc(N), MVT::i32);
}]>;
def uimm16_16_xform : Operand<i32>,
  ImmLeaf<i32, "return isShiftedUInt<16, 16>(Imm);", uimm32_hi16> {
  let ParserMatchClass = UImmAsmOperand<16>;
}

def uimm_shift : Operand<i32>, ImmLeaf<i32, "return isUInt<2>(Imm);"> {
  let EncoderMethod = "getImmShiftOpValue";
  let ParserMatchClass = UImmAsmOperand<2>;
}


def oimm12 : oimm<12>;
def oimm16 : oimm<16>;

def nimm12 : nimm<12>;

def uimm5 : uimm<5>;
def uimm12 : uimm<12>;
def uimm12_1 : uimm<12, 1>;
def uimm12_2 : uimm<12, 2>;
def uimm16 : uimm<16>;


//===----------------------------------------------------------------------===//
// Instruction Formats
//===----------------------------------------------------------------------===//

include "CSKYInstrFormats.td"

//===----------------------------------------------------------------------===//
// Instruction definitions.
//===----------------------------------------------------------------------===//

class TriOpFrag<dag res> : PatFrag<(ops node: $LHS, node:$MHS, node:$RHS), res>;
class BinOpFrag<dag res> : PatFrag<(ops node:$LHS, node:$RHS), res>;
class UnOpFrag<dag res> : PatFrag<(ops node:$Src), res>;



//===----------------------------------------------------------------------===//
// Basic ALU instructions.
//===----------------------------------------------------------------------===//

def ADDI32 : I_12<0x0, "addi32", add, oimm12>;
def SUBI32 : I_12<0x1, "subi32", sub, oimm12>;
def ANDI32 : I_12<0x2, "andi32", and, uimm12>;
def ANDNI32 : I_12<0x3, "andni32", and, nimm12>;
def ORI32 : I_16_ZX<"ori32", uimm16,
  [(set GPR:$rz, (or GPR:$rx, uimm16:$imm16))]>;
def XORI32 : I_12<0x4, "xori32", xor, uimm12>;
def LSLI32 : I_5_XZ<0x12, 0x1, "lsli32",
  (outs GPR:$rz), (ins GPR:$rx, uimm5:$imm5),
  [(set GPR:$rz, (shl GPR:$rx, uimm5:$imm5))]>;
def LSRI32 : I_5_XZ<0x12, 0x2, "lsri32",
  (outs GPR:$rz), (ins GPR:$rx, uimm5:$imm5),
  [(set GPR:$rz, (srl GPR:$rx, uimm5:$imm5))]>;
def ASRI32 : I_5_XZ<0x12, 0x4, "asri32",
  (outs GPR:$rz), (ins GPR:$rx, uimm5:$imm5),
  [(set GPR:$rz, (sra GPR:$rx, uimm5:$imm5))]>;
def ROTLI32 : I_5_XZ<0x12, 0x8, "rotli32",
  (outs GPR:$rz), (ins GPR:$rx, uimm5:$imm5),
  [(set GPR:$rz, (rotl GPR:$rx, uimm5:$imm5))]>;

def ADDU32 : R_YXZ_SP_F1<0x0, 0x1,
  BinOpFrag<(add node:$LHS, node:$RHS)>, "addu32", 1>;
def SUBU32 : R_YXZ_SP_F1<0x0, 0x4,
  BinOpFrag<(sub node:$LHS, node:$RHS)>, "subu32">;
def AND32 : R_YXZ_SP_F1<0x8, 0x1,
  BinOpFrag<(and node:$LHS, node:$RHS)>, "and32", 1>;
def ANDN32 : R_YXZ_SP_F1<0x8, 0x2,
  BinOpFrag<(and node:$LHS, (not node:$RHS))>, "andn32">;
def OR32: R_YXZ_SP_F1<0x9, 0x1,
  BinOpFrag<(or node:$LHS, node:$RHS)>, "or32", 1>;
def XOR32 : R_YXZ_SP_F1<0x9, 0x2,
  BinOpFrag<(xor node:$LHS, node:$RHS)>, "xor32", 1>;
def NOR32 : R_YXZ_SP_F1<0x9, 0x4,
  BinOpFrag<(not (or node:$LHS, node:$RHS))>, "nor32", 1>;
def LSL32 : R_YXZ_SP_F1<0x10, 0x1,
  BinOpFrag<(shl node:$LHS, node:$RHS)>, "lsl32">;
def LSR32 : R_YXZ_SP_F1<0x10, 0x2,
  BinOpFrag<(srl node:$LHS, node:$RHS)>, "lsr32">;
def ASR32 : R_YXZ_SP_F1<0x10, 0x4,
  BinOpFrag<(sra node:$LHS, node:$RHS)>, "asr32">;
def MULT32 : R_YXZ_SP_F1<0x21, 0x1,
  BinOpFrag<(mul node:$LHS, node:$RHS)>, "mult32", 1>;
def DIVS32 : R_YXZ_SP_F1<0x20, 0x2,
  BinOpFrag<(sdiv node:$LHS, node:$RHS)>, "divs32">;
def DIVU32 : R_YXZ_SP_F1<0x20, 0x1,
  BinOpFrag<(udiv node:$LHS, node:$RHS)>, "divu32">;
def IXH32 : R_YXZ_SP_F1<0x2, 0x1,
  BinOpFrag<(add node:$LHS, (shl node:$RHS, (i32 1)))>, "ixh32">;
def IXW32 : R_YXZ_SP_F1<0x2, 0x2,
  BinOpFrag<(add node:$LHS, (shl node:$RHS, (i32 2)))>, "ixw32">;
def IXD32 : R_YXZ_SP_F1<0x2, 0x4,
  BinOpFrag<(add node:$LHS, (shl node:$RHS, (i32 3)))>, "ixd32">;

def NOT32 : R_XXZ<0b001001, 0b00100, (outs GPR:$rz), (ins GPR:$rx),
  "not32", [(set GPR:$rz, (not GPR:$rx))]>;

def SEXT : I_5_XZ_U<0x16, (outs GPR:$rz), (ins GPR:$rx, uimm5:$msb, uimm5:$lsb), "sext32",
    []>;
def ZEXT : I_5_XZ_U<0x15, (outs GPR:$rz), (ins GPR:$rx, uimm5:$msb, uimm5:$lsb), "zext32",
    []>;


let isCommutable = 1 in
  def ADDC32 : R_YXZ<0x31, 0x0, 0x2, (outs GPR:$rz, CARRY:$cout),
    (ins GPR:$rx, GPR:$ry, CARRY:$cin), "addc32", []>;
def SUBC32 : R_YXZ<0x31, 0x0, 0x8, (outs GPR:$rz, CARRY:$cout),
  (ins GPR:$rx, GPR:$ry, CARRY:$cin), "subc32", []>;

//===----------------------------------------------------------------------===//
// Load & Store instructions.
//===----------------------------------------------------------------------===//

def LD32B : I_LD<AddrMode32B, 0x0, "ld32.b", uimm12>;
def LD32H : I_LD<AddrMode32H, 0x1, "ld32.h", uimm12_1>;
def LD32W : I_LD<AddrMode32WD, 0x2, "ld32.w", uimm12_2>;
def LD32BS : I_LD<AddrMode32B, 0x4, "ld32.bs", uimm12>;
def LD32HS : I_LD<AddrMode32H, 0x5, "ld32.hs", uimm12_1>;

def LDR32B :  I_LDR<0x0, "ldr32.b">;
def LDR32BS :  I_LDR<0x4, "ldr32.bs">;
def LDR32H :  I_LDR<0x1, "ldr32.h">;
def LDR32HS :  I_LDR<0x5, "ldr32.hs">;
def LDR32W :  I_LDR<0x2, "ldr32.w">;

def ST32B : I_ST<AddrMode32B, 0x0, "st32.b", uimm12>;
def ST32H : I_ST<AddrMode32H, 0x1, "st32.h", uimm12_1>;
def ST32W : I_ST<AddrMode32WD, 0x2, "st32.w", uimm12_2>;

def STR32B :  I_STR<0x0, "str32.b">;
def STR32H :  I_STR<0x1, "str32.h">;
def STR32W :  I_STR<0x2, "str32.w">;

//===----------------------------------------------------------------------===//
// Compare instructions.
//===----------------------------------------------------------------------===//

def CMPNEI32 : I_16_X<0x1A, "cmpnei32", uimm16>;
def CMPHSI32 : I_16_X<0x18, "cmphsi32", oimm16>;
def CMPLTI32 : I_16_X<0x19, "cmplti32", oimm16>;
def CMPNE32 : R_YX<0x1, 0x4, "cmpne32">;
def CMPHS32 : R_YX<0x1, 0x1, "cmphs32">;
def CMPLT32 : R_YX<0x1, 0x2, "cmplt32">;

//===----------------------------------------------------------------------===//
// Data move instructions.
//===----------------------------------------------------------------------===//

def MOVT32 : R_ZX<0x3, 0x2, "movt32", []>;
def MOVF32 : R_ZX<0x3, 0x1, "movf32", []>;
def MOV32 : R_XZ<0x12, 0x1, "mov32">;
def MOVI32 : I_16_MOV<0x10, "movi32", uimm16>;
def MOVIH32 : I_16_MOV<0x11, "movih32", uimm16_16_xform>;

def MVC32 : R_Z_1<0x1, 0x8, "mvc32">;
def MVCV32 : R_Z_1<0x1, 0x10, "mvcv32">;

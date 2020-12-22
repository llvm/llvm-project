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

include "CSKYInstrFormats.td"

//===----------------------------------------------------------------------===//
// CSKY specific DAG Nodes.
//===----------------------------------------------------------------------===//

// TODO: Add CSKY specific DAG Nodes.

//===----------------------------------------------------------------------===//
// Operand and SDNode transformation definitions.
//===----------------------------------------------------------------------===//

class oimm<int num> : Operand<i32>,
  ImmLeaf<i32, "return isUInt<"#num#">(Imm - 1);"> {
  let EncoderMethod = "getOImmOpValue";
}

class uimm<int num, int shift = 0> : Operand<i32>,
  ImmLeaf<i32, "return isShiftedUInt<"#num#", "#shift#">(Imm);"> {
  let EncoderMethod = "getImmOpValue<"#shift#">";
}

class simm<int num, int shift = 0> : Operand<i32>,
  ImmLeaf<i32, "return isShiftedInt<"#num#", "#shift#">(Imm);"> {
  let EncoderMethod = "getImmOpValue<"#shift#">";
}

def nimm_XFORM : SDNodeXForm<imm, [{
  return CurDAG->getTargetConstant(~N->getSExtValue(), SDLoc(N), MVT::i32);
}]>;
class nimm<int num> : Operand<i32>,
  ImmLeaf<i32, "return isUInt<"#num#">(~Imm);", nimm_XFORM> {
}


def oimm12 : oimm<12>;

def nimm12 : nimm<12>;

def uimm5 : uimm<5>;
def uimm12 : uimm<12>;

//===----------------------------------------------------------------------===//
// Instruction definitions.
//===----------------------------------------------------------------------===//

class TriOpFrag<dag res> : PatFrag<(ops node: $LHS, node:$MHS, node:$RHS), res>;
class BinOpFrag<dag res> : PatFrag<(ops node:$LHS, node:$RHS), res>;
class UnOpFrag<dag res> : PatFrag<(ops node:$Src), res>;

def ADDI32 : I_12<0x0, "addi32", add, oimm12>;
def SUBI32 : I_12<0x1, "subi32", sub, oimm12>;
def ANDI32 : I_12<0x2, "andi32", and, uimm12>;
def ANDNI32 : I_12<0x3, "andni32", and, nimm12>;
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

def NOT32 : R_XXZ<0b001001, 0b00100, (outs GPR:$rz), (ins GPR:$rx),
  "not", [(set GPR:$rz, (not GPR:$rx))]>;

//===-- RISCVGIsel.td - RISC-V GlobalISel Patterns ---------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
/// \file
/// This file contains patterns that are relevant to GlobalISel, including
/// GIComplexOperandMatcher definitions for equivalent SelectionDAG
/// ComplexPatterns.
//
//===----------------------------------------------------------------------===//

include "RISCV.td"
include "RISCVCombine.td"

def simm12Plus1 : ImmLeaf<XLenVT, [{
    return (isInt<12>(Imm) && Imm != -2048) || Imm == 2048;}]>;
def simm12Plus1i32 : ImmLeaf<i32, [{
    return (isInt<12>(Imm) && Imm != -2048) || Imm == 2048;}]>;

// FIXME: This doesn't check that the G_CONSTANT we're deriving the immediate
// from is only used once
def simm12Minus1Nonzero : ImmLeaf<XLenVT, [{
  return (Imm >= -2049 && Imm < 0) || (Imm > 0 && Imm <= 2046);}]>;

def simm12Minus1NonzeroNonNeg1 : ImmLeaf<XLenVT, [{
  return (Imm >= -2049 && Imm < -1) || (Imm > 0 && Imm <= 2046);}]>;

// Return an immediate value plus 1.
def ImmPlus1 : SDNodeXForm<imm, [{
  return CurDAG->getTargetConstant(N->getSExtValue() + 1, SDLoc(N),
                                   N->getValuePtrVTpe(0));}]>;
def GIImmPlus1 :
  GICustomOperandRenderer<"renderImmPlus1">,
  GISDNodeXFormEquiv<ImmPlus1>;

// Ptr type used in patterns with GlobalISelEmitter
def PtrVT : PtrValueTypeByHwMode<XLenVT, 0>;

// Define pattern expansions for pointer ult/slt conditional codes
def : Pat<(XLenVT (setult (PtrVT GPR:$rs1), simm12:$imm12)),
          (SLTIU GPR:$rs1, simm12:$imm12)>;
def : Pat<(XLenVT (setult (PtrVT GPR:$rs1), (PtrVT GPR:$rs2))),
          (SLTU GPR:$rs1, GPR:$rs2)>;
def : Pat<(XLenVT (setlt (PtrVT GPR:$rs1), simm12:$imm12)),
          (SLTI GPR:$rs1, simm12:$imm12)>;
def : Pat<(XLenVT (setlt (PtrVT GPR:$rs1), (PtrVT GPR:$rs2))),
          (SLT GPR:$rs1, GPR:$rs2)>;

// Define pattern expansions for setcc operations that aren't directly
// handled by a RISC-V instruction.
foreach Ty = [PtrVT, XLenVT] in {
def : Pat<(XLenVT (seteq (Ty GPR:$rs1), (Ty 0))), (SLTIU GPR:$rs1, 1)>;
def : Pat<(XLenVT (seteq (Ty GPR:$rs1), (Ty simm12Plus1:$imm12))),
          (SLTIU (ADDI GPR:$rs1, (NegImm simm12Plus1:$imm12)), 1)>;
def : Pat<(XLenVT (seteq (Ty GPR:$rs1), (Ty GPR:$rs2))),
          (SLTIU (XOR GPR:$rs1, GPR:$rs2), 1)>;
def : Pat<(XLenVT (setne (Ty GPR:$rs1), (Ty 0))), (SLTU (XLenVT X0), GPR:$rs1)>;
def : Pat<(XLenVT (setne (Ty GPR:$rs1), (Ty simm12Plus1:$imm12))),
          (SLTU (XLenVT X0), (ADDI GPR:$rs1, (NegImm simm12Plus1:$imm12)))>;
def : Pat<(XLenVT (setne (Ty GPR:$rs1), (Ty GPR:$rs2))),
          (SLTU (XLenVT X0), (XOR GPR:$rs1, GPR:$rs2))>;
def : Pat<(XLenVT (setugt (Ty GPR:$rs1), (Ty simm12Minus1NonzeroNonNeg1:$imm))),
          (XORI (SLTIU GPR:$rs1,
                       (ImmPlus1 simm12Minus1NonzeroNonNeg1:$imm)), 1)>;
def : Pat<(XLenVT (setugt (Ty GPR:$rs1), (Ty GPR:$rs2))),
          (SLTU GPR:$rs2, GPR:$rs1)>;
def : Pat<(XLenVT (setgt (Ty GPR:$rs1), (Ty simm12Minus1Nonzero:$imm))),
          (XORI (SLTI GPR:$rs1, (ImmPlus1 simm12Minus1Nonzero:$imm)), 1)>;
def : Pat<(XLenVT (setgt (Ty GPR:$rs1), (Ty GPR:$rs2))),
          (SLT GPR:$rs2, GPR:$rs1)>;
def : Pat<(XLenVT (setuge (XLenVT GPR:$rs1), (Ty simm12:$imm))),
          (XORI (SLTIU GPR:$rs1, simm12:$imm), 1)>;
def : Pat<(XLenVT (setuge (Ty GPR:$rs1), (Ty GPR:$rs2))),
          (XORI (SLTU GPR:$rs1, GPR:$rs2), 1)>;
def : Pat<(XLenVT (setge (Ty GPR:$rs1), (Ty simm12:$imm))),
          (XORI (SLTI GPR:$rs1, simm12:$imm), 1)>;
def : Pat<(XLenVT (setge (Ty GPR:$rs1), (Ty GPR:$rs2))),
          (XORI (SLT GPR:$rs1, GPR:$rs2), 1)>;
def : Pat<(XLenVT (setule (Ty GPR:$rs1), (Ty simm12Minus1NonzeroNonNeg1:$imm))),
          (SLTIU GPR:$rs1, (ImmPlus1 simm12Minus1NonzeroNonNeg1:$imm))>;
def : Pat<(XLenVT (setule (Ty GPR:$rs1), (Ty GPR:$rs2))),
          (XORI (SLTU GPR:$rs2, GPR:$rs1), 1)>;
def : Pat<(XLenVT (setle (Ty GPR:$rs1), (Ty simm12Minus1Nonzero:$imm))),
          (SLTI GPR:$rs1, (ImmPlus1 simm12Minus1Nonzero:$imm))>;
def : Pat<(XLenVT (setle (Ty GPR:$rs1), (Ty GPR:$rs2))),
          (XORI (SLT GPR:$rs2, GPR:$rs1), 1)>;
}

let Predicates = [IsRV32] in {
def : LdPat<load, LW, PtrVT>;
def : StPat<store, SW, GPR, PtrVT>;
}

let Predicates = [IsRV64] in {
def : LdPat<load, LD, PtrVT>;
def : StPat<store, SD, GPR, PtrVT>;
}

// Load and store patterns for i16, needed because Zfh makes s16 load/store
// legal and regbank select may not constrain registers to FP.
def : LdPat<load, LH, i16>;
def : StPat<store, SH, GPR, i16>;

def : LdPat<extloadi8, LBU, i16>; // Prefer unsigned due to no c.lb in Zcb.
def : StPat<truncstorei8, SB, GPR, i16>;

let Predicates = [HasAtomicLdSt] in {
  def : LdPat<atomic_load_8,  LB, i16>;
  def : LdPat<atomic_load_16, LH, i16>;

  def : StPat<atomic_store_8,  SB, GPR, i16>;
  def : StPat<atomic_store_16, SH, GPR, i16>;
}

let Predicates = [HasAtomicLdSt, IsRV64] in {
  def : LdPat<atomic_load_32, LW, i32>;
  def : StPat<atomic_store_32, SW, GPR, i32>;
}

//===----------------------------------------------------------------------===//
// RV64 i32 patterns not used by SelectionDAG
//===----------------------------------------------------------------------===//

def uimm5i32 : ImmLeaf<i32, [{return isUInt<5>(Imm);}]>;

def zext_is_sext : PatFrag<(ops node:$src), (zext node:$src), [{
  KnownBits Known = CurDAG->computeKnownBits(N->getOperand(0), 0);
  return Known.isNonNegative();
}]>;

let Predicates = [IsRV64] in {
def : LdPat<extloadi8, LBU, i32>; // Prefer unsigned due to no c.lb in Zcb.
def : LdPat<extloadi16, LH, i32>;

def : StPat<truncstorei8, SB, GPR, i32>;
def : StPat<truncstorei16, SH, GPR, i32>;

def : Pat<(anyext (i32 GPR:$src)), (COPY GPR:$src)>;
def : Pat<(sext (i32 GPR:$src)), (ADDIW GPR:$src, 0)>;
def : Pat<(i32 (trunc GPR:$src)), (COPY GPR:$src)>;

// Use sext if the sign bit of the input is 0.
def : Pat<(zext_is_sext (i32 GPR:$src)), (ADDIW GPR:$src, 0)>;
}

let Predicates = [IsRV64, NotHasStdExtZba] in
def : Pat<(zext (i32 GPR:$src)), (SRLI (i64 (SLLI GPR:$src, 32)), 32)>;

let Predicates = [IsRV32, NoStdExtZbb, NoStdExtZbkb] in 
def : Pat<(XLenVT (zext (i16 GPR:$src))),
          (SRLI (XLenVT (SLLI GPR:$src, 16)), 16)>;

let Predicates = [IsRV64, NoStdExtZbb, NoStdExtZbkb] in {
def : Pat<(i64 (zext (i16 GPR:$src))),
          (SRLI (XLenVT (SLLI GPR:$src, 48)), 48)>;
def : Pat<(i32 (zext (i16 GPR:$src))),
          (SRLI (XLenVT (SLLI GPR:$src, 48)), 48)>;
}

let Predicates = [IsRV32, NoStdExtZbb] in
def : Pat<(XLenVT (sext (i16 GPR:$src))),
          (SRAI (XLenVT (SLLI GPR:$src, 16)), 16)>;

let Predicates = [IsRV64, NoStdExtZbb] in {
def : Pat<(i64 (sext (i16 GPR:$src))),
          (SRAI (XLenVT (SLLI GPR:$src, 48)), 48)>;
def : Pat<(i32 (sext (i16 GPR:$src))),
          (SRAI (XLenVT (SLLI GPR:$src, 48)), 48)>;
}

//===----------------------------------------------------------------------===//
// Zb* RV64 patterns not used by SelectionDAG.
//===----------------------------------------------------------------------===//

let Predicates = [HasStdExtZba, IsRV64] in {
def : Pat<(zext (i32 GPR:$src)), (ADD_UW GPR:$src, (XLenVT X0))>;
}

let Predicates = [HasStdExtZbb] in
def : Pat<(i32 (sext (i16 GPR:$rs))), (SEXT_H GPR:$rs)>;
let Predicates = [HasStdExtZbb, IsRV64] in
def : Pat<(i64 (sext (i16 GPR:$rs))), (SEXT_H GPR:$rs)>;

let Predicates = [HasStdExtZbb, IsRV32] in
def : Pat<(i32 (zext (i16 GPR:$rs))), (ZEXT_H_RV32 GPR:$rs)>;
let Predicates = [HasStdExtZbb, IsRV64] in {
def : Pat<(i64 (zext (i16 GPR:$rs))), (ZEXT_H_RV64 GPR:$rs)>;
def : Pat<(i32 (zext (i16 GPR:$rs))), (ZEXT_H_RV64 GPR:$rs)>;
}

let Predicates = [HasStdExtZbkb, NoStdExtZbb, IsRV32] in
def : Pat<(i32 (zext (i16 GPR:$rs))), (PACK GPR:$rs, (XLenVT X0))>;
let Predicates = [HasStdExtZbkb, NoStdExtZbb, IsRV64] in {
def : Pat<(i64 (zext (i16 GPR:$rs))), (PACKW GPR:$rs, (XLenVT X0))>;
def : Pat<(i32 (zext (i16 GPR:$rs))), (PACKW GPR:$rs, (XLenVT X0))>;
}

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

def GINegImm : GICustomOperandRenderer<"renderNegImm">,
  GISDNodeXFormEquiv<NegImm>;

def GIImmSubFromXLen : GICustomOperandRenderer<"renderImmSubFromXLen">,
  GISDNodeXFormEquiv<ImmSubFromXLen>;
def GIImmSubFrom32 : GICustomOperandRenderer<"renderImmSubFrom32">,
  GISDNodeXFormEquiv<ImmSubFrom32>;

def GIImmPlus1 :
  GICustomOperandRenderer<"renderImmPlus1">,
  GISDNodeXFormEquiv<ImmPlus1>;

def GIAddrRegImm :
  GIComplexOperandMatcher<s32, "selectAddrRegImm">,
  GIComplexPatternEquiv<AddrRegImm>;

// FIXME: This is labelled as handling 's32', however the ComplexPattern it
// refers to handles both i32 and i64 based on the HwMode. Currently this LLT
// parameter appears to be ignored so this pattern works for both, however we
// should add a LowLevelTypeByHwMode, and use that to define our XLenLLT instead
// here.
def GIVLOp : GIComplexOperandMatcher<s32, "renderVLOp">,
             GIComplexPatternEquiv<VLOp>;

// Convert from i32 immediate to i64 target immediate to make SelectionDAG type
// checking happy so we can use ADDIW which expects an XLen immediate.
def as_i64imm : SDNodeXForm<imm, [{
  return CurDAG->getTargetConstant(N->getSExtValue(), SDLoc(N), MVT::i64);
}]>;

def gi_as_i64imm : GICustomOperandRenderer<"renderImm">,
  GISDNodeXFormEquiv<as_i64imm>;

def gi_trailing_zero : GICustomOperandRenderer<"renderTrailingZeros">,
  GISDNodeXFormEquiv<TrailingZeros>;

// FIXME: This is labelled as handling 's32', however the ComplexPattern it
// refers to handles both i32 and i64 based on the HwMode. Currently this LLT
// parameter appears to be ignored so this pattern works for both, however we
// should add a LowLevelTypeByHwMode, and use that to define our XLenLLT instead
// here.
def GIShiftMaskXLen :
    GIComplexOperandMatcher<s32, "selectShiftMask">,
    GIComplexPatternEquiv<shiftMaskXLen>;
def GIShiftMask32 :
    GIComplexOperandMatcher<s32, "selectShiftMask">,
    GIComplexPatternEquiv<shiftMask32>;

def gi_sh1add_op : GIComplexOperandMatcher<s32, "selectSHXADDOp<1>">,
                   GIComplexPatternEquiv<sh1add_op>;
def gi_sh2add_op : GIComplexOperandMatcher<s32, "selectSHXADDOp<2>">,
                   GIComplexPatternEquiv<sh2add_op>;
def gi_sh3add_op : GIComplexOperandMatcher<s32, "selectSHXADDOp<3>">,
                   GIComplexPatternEquiv<sh3add_op>;

def gi_sh1add_uw_op : GIComplexOperandMatcher<s32, "selectSHXADD_UWOp<1>">,
                      GIComplexPatternEquiv<sh1add_uw_op>;
def gi_sh2add_uw_op : GIComplexOperandMatcher<s32, "selectSHXADD_UWOp<2>">,
                      GIComplexPatternEquiv<sh2add_uw_op>;
def gi_sh3add_uw_op : GIComplexOperandMatcher<s32, "selectSHXADD_UWOp<3>">,
                      GIComplexPatternEquiv<sh3add_uw_op>;

def gi_zexti32 : GIComplexOperandMatcher<s64, "selectZExtBits<32>">,
                 GIComplexPatternEquiv<zexti32>;
def gi_zexti16 : GIComplexOperandMatcher<s32, "selectZExtBits<16>">,
                 GIComplexPatternEquiv<zexti16>;
def gi_zexti8  : GIComplexOperandMatcher<s32, "selectZExtBits<8>">,
                 GIComplexPatternEquiv<zexti8>;

// FIXME: Canonicalize (sub X, C) -> (add X, -C) earlier.
def : Pat<(XLenVT (sub GPR:$rs1, simm12Plus1:$imm)),
          (ADDI GPR:$rs1, (NegImm simm12Plus1:$imm))>;

let Predicates = [IsRV64] in {
def : Pat<(i32 (sub GPR:$rs1, simm12Plus1i32:$imm)),
          (ADDIW GPR:$rs1, (i64 (NegImm $imm)))>;
}

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

//===----------------------------------------------------------------------===//
// RV64 i32 patterns not used by SelectionDAG
//===----------------------------------------------------------------------===//

def simm12i32 : ImmLeaf<i32, [{return isInt<12>(Imm);}]>;
def uimm5i32 : ImmLeaf<i32, [{return isUInt<5>(Imm);}]>;

def zext_is_sext : PatFrag<(ops node:$src), (zext node:$src), [{
  KnownBits Known = CurDAG->computeKnownBits(N->getOperand(0), 0);
  return Known.isNonNegative();
}]>;

let Predicates = [IsRV64] in {
def : LdPat<sextloadi8, LB, i32>;
def : LdPat<extloadi8, LBU, i32>; // Prefer unsigned due to no c.lb in Zcb.
def : LdPat<sextloadi16, LH, i32>;
def : LdPat<extloadi16, LH, i32>;
def : LdPat<zextloadi8, LBU, i32>;
def : LdPat<zextloadi16, LHU, i32>;

def : StPat<truncstorei8, SB, GPR, i32>;
def : StPat<truncstorei16, SH, GPR, i32>;

def : Pat<(anyext GPR:$src), (COPY GPR:$src)>;
def : Pat<(sext GPR:$src), (ADDIW GPR:$src, 0)>;
def : Pat<(trunc GPR:$src), (COPY GPR:$src)>;

def : PatGprGpr<add, ADDW, i32, i32>;
def : PatGprGpr<sub, SUBW, i32, i32>;
def : PatGprGpr<and, AND, i32, i32>;
def : PatGprGpr<or, OR, i32, i32>;
def : PatGprGpr<xor, XOR, i32, i32>;
def : PatGprGpr<shl, SLLW, i32, i32>;
def : PatGprGpr<srl, SRLW, i32, i32>;
def : PatGprGpr<sra, SRAW, i32, i32>;

def : Pat<(i32 (add GPR:$rs1, simm12i32:$imm)),
          (ADDIW GPR:$rs1, (i64 (as_i64imm $imm)))>;
def : Pat<(i32 (and GPR:$rs1, simm12i32:$imm)),
          (ANDI GPR:$rs1, (i64 (as_i64imm $imm)))>;
def : Pat<(i32 (or GPR:$rs1, simm12i32:$imm)),
          (ORI GPR:$rs1, (i64 (as_i64imm $imm)))>;
def : Pat<(i32 (xor GPR:$rs1, simm12i32:$imm)),
          (XORI GPR:$rs1, (i64 (as_i64imm $imm)))>;

def : Pat<(i32 (shl GPR:$rs1, uimm5i32:$imm)),
          (SLLIW GPR:$rs1, (i64 (as_i64imm $imm)))>;
def : Pat<(i32 (srl GPR:$rs1, uimm5i32:$imm)),
          (SRLIW GPR:$rs1, (i64 (as_i64imm $imm)))>;
def : Pat<(i32 (sra GPR:$rs1, uimm5i32:$imm)),
          (SRAIW GPR:$rs1, (i64 (as_i64imm $imm)))>;

def : Pat<(i32 (and GPR:$rs, TrailingOnesMask:$mask)),
          (SRLI (i32 (SLLI $rs, (i64 (XLenSubTrailingOnes $mask)))),
                (i64 (XLenSubTrailingOnes $mask)))>;

// Use sext if the sign bit of the input is 0.
def : Pat<(zext_is_sext GPR:$src), (ADDIW GPR:$src, 0)>;
}

let Predicates = [IsRV64, NotHasStdExtZba] in {
def : Pat<(zext GPR:$src), (SRLI (i64 (SLLI GPR:$src, 32)), 32)>;

// If we're shifting a 32-bit zero extended value left by 0-31 bits, use 2
// shifts instead of 3. This can occur when unsigned is used to index an array.
def : Pat<(shl (zext GPR:$rs), uimm5:$shamt),
          (SRLI (i64 (SLLI GPR:$rs, 32)), (ImmSubFrom32 uimm5:$shamt))>;
}

//===----------------------------------------------------------------------===//
// M RV64 i32 legalization patterns.
//===----------------------------------------------------------------------===//

let Predicates = [HasStdExtZmmul, IsRV64] in {
def : PatGprGpr<mul, MULW, i32, i32>;
}

let Predicates = [HasStdExtM, IsRV64] in {
def : PatGprGpr<sdiv, DIVW, i32, i32>;
def : PatGprGpr<udiv, DIVUW, i32, i32>;
def : PatGprGpr<srem, REMW, i32, i32>;
def : PatGprGpr<urem, REMUW, i32, i32>;
}

//===----------------------------------------------------------------------===//
// Zb* RV64 i32 patterns not used by SelectionDAG.
//===----------------------------------------------------------------------===//

let Predicates = [HasStdExtZbb, IsRV64] in {
def : PatGpr<ctlz, CLZW, i32>;
def : PatGpr<cttz, CTZW, i32>;
def : PatGpr<ctpop, CPOPW, i32>;

def : Pat<(i32 (sext_inreg GPR:$rs1, i8)), (SEXT_B GPR:$rs1)>;
def : Pat<(i32 (sext_inreg GPR:$rs1, i16)), (SEXT_H GPR:$rs1)>;

def : Pat<(i32 (and GPR:$rs, 0xFFFF)), (ZEXT_H_RV64 GPR:$rs)>;
} // Predicates = [HasStdExtZbb, IsRV64]

let Predicates = [HasStdExtZbbOrZbkb, IsRV64] in {
def : Pat<(i32 (and GPR:$rs1, (not GPR:$rs2))), (ANDN GPR:$rs1, GPR:$rs2)>;
def : Pat<(i32 (or  GPR:$rs1, (not GPR:$rs2))), (ORN  GPR:$rs1, GPR:$rs2)>;
def : Pat<(i32 (xor GPR:$rs1, (not GPR:$rs2))), (XNOR GPR:$rs1, GPR:$rs2)>;

def : PatGprGpr<rotl, ROLW, i32, i32>;
def : PatGprGpr<rotr, RORW, i32, i32>;
def : Pat<(i32 (rotr GPR:$rs1, uimm5i32:$imm)),
          (RORIW GPR:$rs1, (i64 (as_i64imm $imm)))>;

def : Pat<(i32 (rotl GPR:$rs1, uimm5i32:$rs2)),
          (RORIW GPR:$rs1, (ImmSubFrom32 uimm5i32:$rs2))>;
} // Predicates = [HasStdExtZbbOrZbkb, IsRV64]

let Predicates = [HasStdExtZba, IsRV64] in {
def : Pat<(shl (i64 (zext i32:$rs1)), uimm5:$shamt),
          (SLLI_UW GPR:$rs1, uimm5:$shamt)>;

def : Pat<(i64 (add_like_non_imm12 (zext GPR:$rs1), GPR:$rs2)),
          (ADD_UW GPR:$rs1, GPR:$rs2)>;
def : Pat<(zext GPR:$src), (ADD_UW GPR:$src, (XLenVT X0))>;

foreach i = {1,2,3} in {
  defvar shxadd = !cast<Instruction>("SH"#i#"ADD");
  def : Pat<(i32 (add_like_non_imm12 (shl GPR:$rs1, (i32 i)), GPR:$rs2)),
            (shxadd GPR:$rs1, GPR:$rs2)>;
}
}

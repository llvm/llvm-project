//===- ARCInstrInfo.td - Target Description for ARC --------*- tablegen -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file describes the ARC instructions in TableGen format.
//
//===----------------------------------------------------------------------===//

include "ARCInstrFormats.td"

// ---------------------------------------------------------------------------
// Selection DAG Nodes.
// ---------------------------------------------------------------------------

// Selection DAG types.
def SDT_ARCcmptst : SDTypeProfile<0, 2, [SDTCisSameAs<0, 1>]>;
def SDT_ARCcmov : SDTypeProfile<1, 3, [SDTCisSameAs<0, 1>]>;
def SDT_ARCmov : SDTypeProfile<1, 1, [SDTCisSameAs<0, 1>]>;
def SDT_ARCbrcc : SDTypeProfile<0, 4, []>;
def SDT_ARCBranchLink : SDTypeProfile<0, 1, [SDTCisPtrTy<0>]>;
def SDT_ARCCallSeqStart : SDCallSeqStart<[ SDTCisVT<0, i32>,
                                           SDTCisVT<1, i32> ]>;
def SDT_ARCCallSeqEnd   : SDCallSeqEnd<[ SDTCisVT<0, i32>,
                                         SDTCisVT<1, i32> ]>;


// Global Address.
def ARCGAWrapper : SDNode<"ARCISD::GAWRAPPER", SDT_ARCmov, []>;

// Comparison
def ARCcmp : SDNode<"ARCISD::CMP", SDT_ARCcmptst, [SDNPOutGlue]>;

// Conditionanal mov
def ARCcmov : SDNode<"ARCISD::CMOV", SDT_ARCcmov, [SDNPInGlue]>;

// Conditional Branch
def ARCbrcc : SDNode<"ARCISD::BRcc", SDT_ARCbrcc,
                       [SDNPHasChain, SDNPInGlue, SDNPOutGlue]>;

// Direct Call
def ARCBranchLink     : SDNode<"ARCISD::BL",SDT_ARCBranchLink,
                            [SDNPHasChain, SDNPOptInGlue, SDNPOutGlue,
                             SDNPVariadic]>;

// Indirect Call
def ARCJumpLink       : SDNode<"ARCISD::JL",SDT_ARCBranchLink,
                                 [SDNPHasChain, SDNPOptInGlue, SDNPOutGlue,
                                  SDNPVariadic]>;
// Call return
def ret      : SDNode<"ARCISD::RET", SDTNone,
                      [SDNPHasChain, SDNPOptInGlue, SDNPVariadic]>;

// Call sequencing nodes.
// These are target-independent nodes, but have target-specific formats.
def callseq_start : SDNode<"ISD::CALLSEQ_START", SDT_ARCCallSeqStart,
                           [SDNPHasChain, SDNPOutGlue]>;
def callseq_end   : SDNode<"ISD::CALLSEQ_END",   SDT_ARCCallSeqEnd,
                           [SDNPHasChain, SDNPOptInGlue, SDNPOutGlue]>;

//===----------------------------------------------------------------------===//
// Instruction Pattern Stuff
//===----------------------------------------------------------------------===//

def imm32 : ImmLeaf<i32, [{
  return (Imm & 0xFFFFFFFF) == Imm;
}]>;

// Addressing modes
def FrameADDR_ri : ComplexPattern<i32, 2, "SelectFrameADDR_ri",
                                  [add, frameindex], []>;
def AddrModeS9 : ComplexPattern<i32, 2, "SelectAddrModeS9", []>;
def AddrModeImm : ComplexPattern<i32, 2, "SelectAddrModeImm", []>;
def AddrModeFar : ComplexPattern<i32, 2, "SelectAddrModeFar", []>;

//===----------------------------------------------------------------------===//
// Instruction Class Templates
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// Pseudo Instructions
//===----------------------------------------------------------------------===//

let Defs = [SP], Uses = [SP] in {
def ADJCALLSTACKDOWN : PseudoInstARC<(outs), (ins i32imm:$amt, i32imm:$amt2),
                               "# ADJCALLSTACKDOWN $amt, $amt2",
                               [(callseq_start timm:$amt, timm:$amt2)]>;
def ADJCALLSTACKUP : PseudoInstARC<(outs), (ins i32imm:$amt1, i32imm:$amt2),
                            "# ADJCALLSTACKUP $amt1",
                            [(callseq_end timm:$amt1, timm:$amt2)]>;
}

def GETFI : PseudoInstARC<(outs GPR32:$dst), (ins MEMii:$addr),
                             "pldfi $dst, $addr",
                             [(set GPR32:$dst, FrameADDR_ri:$addr)]>;


def ST_FAR : PseudoInstARC<(outs), (ins GPR32:$dst, MEMrlimm:$addr),
                             "ST_FAR $dst, $addr",
                             [(store GPR32:$dst, AddrModeFar:$addr)]>;

def STH_FAR : PseudoInstARC<(outs), (ins GPR32:$dst, MEMrlimm:$addr),
                             "STH_FAR $dst, $addr",
                             [(truncstorei16 GPR32:$dst, AddrModeFar:$addr)]>;

def STB_FAR : PseudoInstARC<(outs), (ins GPR32:$dst, MEMrlimm:$addr),
                             "STB_FAR $dst, $addr",
                             [(truncstorei8 GPR32:$dst, AddrModeFar:$addr)]>;

//===----------------------------------------------------------------------===//
// Instruction Generation multiclasses.
// Generate many variants of a single instruction with a single defining
// multiclass.  These classes do not contain Selection DAG patterns.
//===----------------------------------------------------------------------===//

// Generic 3 operand binary instructions (i.e., add, r0, r1, r2).
multiclass ArcBinaryInst<bits<5> major, bits<6> mincode,
                       string opasm> {
  // 3 register variant.
  def _rrr : F32_DOP_RR<major, mincode, 0, (outs GPR32:$A),
                        (ins GPR32:$B, GPR32:$C),
                        !strconcat(opasm, "\t$A, $B, $C"),
                        []>;

  // 2 register with unsigned 6-bit immediate variant.
  def _rru6 : F32_DOP_RU6<major, mincode, 0, (outs GPR32:$A),
                          (ins GPR32:$B, immU6:$U6),
                          !strconcat(opasm, "\t$A, $B, $U6"),
                          []>;
  // 2 register with 32-bit immediate variant.
  def _rrlimm : F32_DOP_RLIMM<major, mincode, 0,
                      (outs GPR32:$A),
                      (ins GPR32:$B, i32imm:$LImm),
                      !strconcat(opasm, "\t$A, $B, $LImm"),
                      []>;
  // 2 matched-register with signed 12-bit immediate variant (add r0, r0, -1).
  def _rrs12 : F32_DOP_RS12<major, mincode, 0,
                            (outs GPR32:$B),
                            (ins GPR32:$in, immS12:$S12),
                            !strconcat(opasm, "\t$B, $in, $S12"),
                            []>
  { let Constraints = "$B = $in"; }
}

// Special multivariant GEN4 DOP format instruction that take 2 registers.
// This is the class that is used for various comparison instructions.
multiclass ArcSpecialDOPInst<bits<6> subop, string opasm, bit F> {
  def _rr : F32_DOP_RR<0b00100, subop, F, (outs), (ins GPR32:$B, GPR32:$C),
               !strconcat(opasm, "\t$B, $C"),
               []>;

  def _ru6 : F32_DOP_RU6<0b00100, subop, F, (outs), (ins GPR32:$B, i32imm:$U6),
               !strconcat(opasm, "\t$B, $U6"),
               []>;

  def _rlimm : F32_DOP_RLIMM<0b00100, subop, F, (outs),
               (ins GPR32:$B, i32imm:$LImm),
               !strconcat(opasm, "\t$B, $LImm"),
               []>;
}

// Generic 2-operand unary instructions.
multiclass ArcUnaryInst<bits<5> major, bits<6> subop,
                        string opasm> {
  def _rr : F32_SOP_RR<major, subop, 0, (outs GPR32:$B), (ins GPR32:$C),
                       !strconcat(opasm, "\t$B, $C"), []>;
}


multiclass ArcBinaryGEN4Inst<bits<6> mincode, string opasm> :
  ArcBinaryInst<0b00100, mincode, opasm>;
multiclass ArcBinaryEXT5Inst<bits<6> mincode, string opasm> :
  ArcBinaryInst<0b00101, mincode, opasm>;

multiclass ArcUnaryGEN4Inst<bits<6> mincode, string opasm> :
  ArcUnaryInst<0b00100, mincode, opasm>;

// Pattern generation for differnt instruction variants.
multiclass MultiPat<SDPatternOperator InFrag,
               Instruction RRR, Instruction RRU6, Instruction RRLImm> {
  def _rrr : Pat<(InFrag i32:$B, i32:$C), (RRR i32:$B, i32:$C)>;
  def _rru6 : Pat<(InFrag i32:$B, immU6:$U6), (RRU6 i32:$B, immU6:$U6)>;
  def _rrlimm : Pat<(InFrag i32:$B, imm32:$LImm), (RRLImm i32:$B, imm32:$LImm)>;
}

// ---------------------------------------------------------------------------
// Instruction defintions and patterns for 3 operand binary instructions.
// ---------------------------------------------------------------------------

// Definitions for 3 operand binary instructions.
defm ADD : ArcBinaryGEN4Inst<0b000000, "add">;
defm SUB : ArcBinaryGEN4Inst<0b000010, "sub">;
defm OR  : ArcBinaryGEN4Inst<0b000101, "or">;
defm AND : ArcBinaryGEN4Inst<0b000100, "and">;
defm XOR : ArcBinaryGEN4Inst<0b000111, "xor">;
defm MAX : ArcBinaryGEN4Inst<0b001000, "max">;
defm MIN : ArcBinaryGEN4Inst<0b001001, "min">;
defm ASL : ArcBinaryEXT5Inst<0b000000, "asl">;
defm LSR : ArcBinaryEXT5Inst<0b000001, "lsr">;
defm ASR : ArcBinaryEXT5Inst<0b000010, "asr">;
defm ROR : ArcBinaryEXT5Inst<0b000011, "ror">;
defm MPY  : ArcBinaryGEN4Inst<0b011010, "mpy">;
defm MPYM : ArcBinaryGEN4Inst<0b011011, "mpym">;
defm MPYMU : ArcBinaryGEN4Inst<0b011100, "mpymu">;

// Patterns for 3 operand binary instructions.
defm : MultiPat<add, ADD_rrr, ADD_rru6, ADD_rrlimm>;
defm : MultiPat<sub, SUB_rrr, SUB_rru6, SUB_rrlimm>;
defm : MultiPat<or, OR_rrr, OR_rru6, OR_rrlimm>;
defm : MultiPat<and, AND_rrr, AND_rru6, AND_rrlimm>;
defm : MultiPat<xor, XOR_rrr, XOR_rru6, XOR_rrlimm>;
defm : MultiPat<smax, MAX_rrr, MAX_rru6, MAX_rrlimm>;
defm : MultiPat<smin, MIN_rrr, MIN_rru6, MIN_rrlimm>;
defm : MultiPat<shl, ASL_rrr, ASL_rru6, ASL_rrlimm>;
defm : MultiPat<srl, LSR_rrr, LSR_rru6, LSR_rrlimm>;
defm : MultiPat<sra, ASR_rrr, ASR_rru6, ASR_rrlimm>;
defm : MultiPat<rotr, ROR_rrr, ROR_rru6, ROR_rrlimm>;
defm : MultiPat<mul, MPY_rrr, MPY_rru6, MPY_rrlimm>;
defm : MultiPat<mulhs, MPYM_rrr, MPYM_rru6, MPYM_rrlimm>;
defm : MultiPat<mulhu, MPYMU_rrr, MPYMU_rru6, MPYMU_rrlimm>;


// ---------------------------------------------------------------------------
// Unary Instruction definitions.
// ---------------------------------------------------------------------------
// General unary instruction definitions.
defm SEXB : ArcUnaryGEN4Inst<0b000101, "sexb">;
defm SEXH : ArcUnaryGEN4Inst<0b000110, "sexh">;

// General Unary Instruction fragments.
def : Pat<(sext_inreg i32:$a, i8), (SEXB_rr i32:$a)>;
def : Pat<(sext_inreg i32:$a, i16), (SEXH_rr i32:$a)>;

// Comparison instruction definition
let isCompare = 1, Defs = [STATUS32] in {
defm CMP : ArcSpecialDOPInst<0b001100, "cmp", 1>;
}

def cmp : PatFrag<(ops node:$op1, node:$op2), (ARCcmp $op1, $op2)>;
defm : MultiPat<cmp, CMP_rr, CMP_ru6, CMP_rlimm>;

// ---------------------------------------------------------------------------
// MOV instruction and variants (conditional mov).
// ---------------------------------------------------------------------------
let isReMaterializable = 1, isAsCheapAsAMove = 1, isMoveImm = 1 in {
def MOV_rs12 : F32_DOP_RS12<0b00100, 0b001010, 0,
                 (outs GPR32:$B), (ins immS12:$S12),
                 "mov\t$B, $S12",
                 [(set GPR32:$B, immS12:$S12)]>;
}

def MOV_rr : F32_DOP_RR<0b00100, 0b001010, 0,
                (outs GPR32:$B), (ins GPR32:$C),
                "mov\t$B, $C", []>;

def MOV_rlimm : F32_DOP_RLIMM<0b00100, 0b001010, 0,
                      (outs GPR32:$B), (ins i32imm:$LImm),
                      "mov\t$B, $LImm", []>;

def MOV_ru6 : F32_DOP_RU6<0b00100, 0b001010, 0,
                          (outs GPR32:$B), (ins immU6:$U6),
                          "mov\t$B, $U6", []>;

def cmov : PatFrag<(ops node:$op1, node:$op2, node:$cc),
                   (ARCcmov $op1, $op2, $cc)>;
let Uses = [STATUS32] in {
def MOVcc : F32_DOP_CC_RR<0b00100, 0b001010, 0,
               (outs GPR32:$B),
               (ins GPR32:$C, GPR32:$fval, cmovpred:$cc),
               !strconcat("mov.", "$cc\t$B, $C"),
               [(set GPR32:$B, (cmov i32:$C, i32:$fval, cmovpred:$cc))]> {
  let Constraints = "$B = $fval";
}
}
def : Pat<(ARCGAWrapper tglobaladdr:$addr),
           (MOV_rlimm tglobaladdr:$addr)>;

def : Pat<(ARCGAWrapper tjumptable:$addr),
           (MOV_rlimm tjumptable:$addr)>;


// ---------------------------------------------------------------------------
// Control flow instructions (branch, return, calls, etc).
// ---------------------------------------------------------------------------

// Branch instructions
let isBranch = 1, isTerminator = 1, isBarrier = 1 in {
// Unconditional branch.
def BR : F32_BR0_UCOND_FAR<(outs), (ins btargetS25:$S25),
                  "b\t$S25", [(br bb:$S25)]>;

let Uses=[STATUS32] in {
// Conditional branch.
def Bcc : F32_BR0_COND<(outs), (ins btargetS21:$S21, ccond:$cc),
                    "b$cc\t$S21", []>;
}

// Compare and branch (limited range).
def BRcc_rr  : F32_BR1_BCC<(outs),
                           (ins btargetS9:$S9, GPR32:$B, GPR32:$C, brccond:$cc),
                           "br$cc\t$B, $C, $S9", 0, []>;
def BRcc_ru6 : F32_BR1_BCC<(outs),
                           (ins btargetS9:$S9, GPR32:$B, immU6:$C, brccond:$cc),
                           "br$cc\t$B, $C, $S9", 1, []>;

// Pseudo compare and branch.
// After register allocation, this can expand into either a limited range
// Compare and branch (BRcc), or into CMP + Bcc.
// At worst, this expands into 2 4-byte instructions.
def BRcc_rr_p : PseudoInstARC<(outs),
                              (ins btarget:$T, GPR32:$B, GPR32:$C, ccond:$cc),
                              "pbr$cc\t$B, $C, $T", 
                              [(ARCbrcc bb:$T, i32:$B, i32:$C, imm32:$cc)]>
                              { let Size = 8; }

def BRcc_ru6_p : PseudoInstARC<(outs),
                               (ins btarget:$T, GPR32:$B, i32imm:$C, ccond:$cc),
                               "pbr$cc\t$B, $C, $T",
                               [(ARCbrcc bb:$T, i32:$B, immU6:$C, imm32:$cc)]>
                               { let Size = 8; }
}

// Indirect, unconditional Jump.
let isBranch = 1, isTerminator = 1, isBarrier = 1, isIndirectBranch = 1 in {
def J :  F32_DOP_RR<0b00100, 0b100000, 0,
               (outs), (ins GPR32:$C),
               "j\t[$C]", [(brind i32:$C)]>;
}

// Call instructions.
let isCall = 1, Defs = [BLINK], Uses = [SP] in {
// Direct unconditional call.
def BL : F32_BR1_BL_UCOND_FAR<(outs), (ins calltargetS25:$S25),
                    "bl\t$S25", [(ARCBranchLink tglobaladdr:$S25)]>;

// Indirect unconditional call.
let isIndirectBranch = 1, Defs = [BLINK], Uses = [SP] in {
def JL : F32_DOP_RR<0b00100, 0b100010, 0, (outs), (ins GPR32:$C),
                   "jl\t[$C]", [(ARCJumpLink i32:$C)]>;
}
}

// Pattern to generate BL instruction.
def : Pat<(ARCBranchLink texternalsym:$dst), (BL texternalsym:$dst)>;

// Return from call.
let isReturn = 1, isTerminator = 1, isBarrier = 1  in {
// This is a specialized 2-byte instruction that doesn't generalize
// to any larger 2-byte class, so go ahead and define it here.
def J_S_BLINK : InstARC<2, (outs), (ins), "j_s\t[%blink]", [(ret)]> {
  let Inst{15-0} = 0b0111111011100000;
}
}

//----------------------------------------------------------------------------
// Load/Store instructions.
//----------------------------------------------------------------------------

// 2-byte push/pop blink instructions commonly used for prolog/epilog
// generation.  These 2 instructions are actually specialized 2-byte
// format instructions that aren't generalized to a larger 2-byte
// class, so we might as well have them here.
let Uses = [BLINK], Defs = [SP] in {
def PUSH_S_BLINK : InstARC<2, (outs), (ins),
                             "push_s\t%blink", []> {
  let Inst{15-0} = 0b1100000011110001;
}
}

let Defs = [BLINK, SP] in {
def POP_S_BLINK : InstARC<2, (outs), (ins),
                             "pop_s\t%blink", []> {
  let Inst{15-0} = 0b1100000011010001;
}
}

// Load instruction variants:
// Control bits: x, aa, di, zz
// x - sign extend.
// aa - incrementing mode. (N/A for LIMM).
// di - uncached.
// zz - data size.
multiclass ArcLdInst<bits<2> zz, string asmop> {
  let mayLoad = 1 in {
  def _rs9 : F32_LD_ADDR<0, 0b00, 0, zz,
                         (outs GPR32:$A), (ins MEMrs9:$addr),
                         !strconcat(asmop, "\t$A, [$addr]"), []>;

  def _limm : F32_LD_LIMM<0, 0, zz,
                         (outs GPR32:$A), (ins MEMii:$addr),
                         !strconcat(asmop, "\t$A, [$addr]"), []>;

  def _rlimm : F32_LD_RLIMM<0, 0b00, 0, zz,
                           (outs GPR32:$A), (ins MEMrlimm:$addr),
                           !strconcat(asmop, "\t$A, [$addr]"), []>;

  def _X_rs9 : F32_LD_ADDR<1, 0b00, 0, zz,
                         (outs GPR32:$A), (ins MEMrs9:$addr),
                         !strconcat(asmop, ".x\t$A, [$addr]"), []>;

  def _X_limm : F32_LD_LIMM<1, 0, zz,
                         (outs GPR32:$A), (ins MEMii:$addr),
                         !strconcat(asmop, ".x\t$A, [$addr]"), []>;

  def _X_rlimm : F32_LD_RLIMM<1, 0b00, 0, zz,
                           (outs GPR32:$A), (ins MEMrlimm:$addr),
                           !strconcat(asmop, ".x\t$A, [$addr]"), []>;

  def _AB_rs9 : F32_LD_RS9<0, 0b10, 0, zz,
                      (outs GPR32:$addrout, GPR32:$A),
                      (ins GPR32:$B, immS9:$S9),
                      !strconcat(asmop, ".ab\t$A, [$B,$S9]"), []>
    { let Constraints = "$addrout = $B"; }
  }
}
                         
// Load instruction definitions.
defm LD  : ArcLdInst<0b00, "ld">;
defm LDH : ArcLdInst<0b10, "ldh">;
defm LDB : ArcLdInst<0b01, "ldb">;

// Load instruction patterns.
// 32-bit loads.
def : Pat<(load AddrModeS9:$addr), (LD_rs9 AddrModeS9:$addr)>;
def : Pat<(load AddrModeImm:$addr), (LD_limm AddrModeImm:$addr)>;
def : Pat<(load AddrModeFar:$addr), (LD_rs9 AddrModeFar:$addr)>;

// 16-bit loads
def : Pat<(zextloadi16 AddrModeS9:$addr), (LDH_rs9 AddrModeS9:$addr)>;
def : Pat<(extloadi16 AddrModeS9:$addr), (LDH_rs9 AddrModeS9:$addr)>;
def : Pat<(zextloadi16 AddrModeImm:$addr), (LDH_limm AddrModeImm:$addr)>;
def : Pat<(extloadi16 AddrModeImm:$addr), (LDH_limm AddrModeImm:$addr)>;
def : Pat<(zextloadi16 AddrModeFar:$addr), (LDH_rlimm AddrModeFar:$addr)>;
def : Pat<(extloadi16 AddrModeFar:$addr), (LDH_rlimm AddrModeFar:$addr)>;
def : Pat<(sextloadi16 AddrModeImm:$addr),(LDH_X_limm AddrModeImm:$addr)>;
def : Pat<(sextloadi16 AddrModeFar:$addr),(LDH_X_rlimm AddrModeFar:$addr)>;
def : Pat<(sextloadi16 AddrModeS9:$addr),(LDH_X_rs9 AddrModeS9:$addr)>;

// 8-bit loads.
def : Pat<(zextloadi8 AddrModeS9:$addr), (LDB_rs9 AddrModeS9:$addr)>;
def : Pat<(extloadi8 AddrModeS9:$addr), (LDB_rs9 AddrModeS9:$addr)>;
def : Pat<(zextloadi8 AddrModeImm:$addr), (LDB_limm AddrModeImm:$addr)>;
def : Pat<(extloadi8 AddrModeImm:$addr), (LDB_limm AddrModeImm:$addr)>;
def : Pat<(zextloadi8 AddrModeFar:$addr), (LDB_rlimm AddrModeFar:$addr)>;
def : Pat<(extloadi8 AddrModeFar:$addr), (LDB_rlimm AddrModeFar:$addr)>;
def : Pat<(zextloadi1 AddrModeS9:$addr), (LDB_rs9 AddrModeS9:$addr)>;
def : Pat<(extloadi1 AddrModeS9:$addr), (LDB_rs9 AddrModeS9:$addr)>;
def : Pat<(zextloadi1 AddrModeImm:$addr), (LDB_limm AddrModeImm:$addr)>;
def : Pat<(extloadi1 AddrModeImm:$addr), (LDB_limm AddrModeImm:$addr)>;
def : Pat<(zextloadi1 AddrModeFar:$addr), (LDB_rlimm AddrModeFar:$addr)>;
def : Pat<(extloadi1 AddrModeFar:$addr), (LDB_rlimm AddrModeFar:$addr)>;
def : Pat<(sextloadi8 AddrModeImm:$addr),(LDB_X_limm AddrModeImm:$addr)>;
def : Pat<(sextloadi8 AddrModeFar:$addr),(LDB_X_rlimm AddrModeFar:$addr)>;
def : Pat<(sextloadi8 AddrModeS9:$addr),(LDB_X_rs9 AddrModeS9:$addr)>;


// Store instruction variants:
// Control bits: aa, di, zz
// aa - incrementing mode. (N/A for LIMM).
// di - uncached.
// zz - data size.
multiclass ArcStInst<bits<2> zz, string asmop> {
  let mayStore = 1 in {
  def _rs9  : F32_ST_ADDR<0b00, 0, zz, (outs), (ins GPR32:$C, MEMrs9:$addr),
                         !strconcat(asmop, "\t$C, [$addr]"), []>;

  def _limm : F32_ST_LIMM<0, zz, (outs), (ins GPR32:$C, MEMii:$addr),
                         !strconcat(asmop, "\t$C, [$addr]"), []>;

  def _AW_rs9 : F32_ST_RS9<0b01, 0, zz, (outs GPR32:$addrout),
                      (ins GPR32:$C, GPR32:$B, immS9:$S9),
                      !strconcat(asmop, ".aw\t$C, [$B,$S9]"), []>
    { let Constraints = "$addrout = $B"; }
  }
}

// Store instruction definitions.
defm ST  : ArcStInst<0b00, "st">;
defm STH : ArcStInst<0b10, "sth">;
defm STB : ArcStInst<0b01, "stb">;

// Store instruction patterns.
// 32-bit stores
def : Pat<(store i32:$C, AddrModeS9:$addr),
          (ST_rs9 i32:$C, AddrModeS9:$addr)>;
def : Pat<(store i32:$C, AddrModeImm:$addr),
          (ST_limm i32:$C, AddrModeImm:$addr)>;

// 16-bit stores
def : Pat<(truncstorei16 i32:$C, AddrModeS9:$addr),
          (STH_rs9 i32:$C, AddrModeS9:$addr)>;
def : Pat<(truncstorei16 i32:$C, AddrModeImm:$addr),
          (STH_limm i32:$C, AddrModeImm:$addr)>;

// 8-bit stores
def : Pat<(truncstorei8 i32:$C, AddrModeS9:$addr),
          (STB_rs9 i32:$C, AddrModeS9:$addr)>;
def : Pat<(truncstorei8 i32:$C, AddrModeImm:$addr),
          (STB_limm i32:$C, AddrModeImm:$addr)>;


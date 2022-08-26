//== LoongArchInstrInfo.td - Target Description for LoongArch -*- tablegen -*-//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file describes the LoongArch instructions in TableGen format.
//
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// LoongArch specific DAG Nodes.
//===----------------------------------------------------------------------===//

// Target-independent type requirements, but with target-specific formats.
def SDT_CallSeqStart : SDCallSeqStart<[SDTCisVT<0, i32>,
                                       SDTCisVT<1, i32>]>;
def SDT_CallSeqEnd : SDCallSeqEnd<[SDTCisVT<0, i32>,
                                   SDTCisVT<1, i32>]>;

// Target-dependent type requirements.
def SDT_LoongArchCall : SDTypeProfile<0, -1, [SDTCisVT<0, GRLenVT>]>;
def SDT_LoongArchIntBinOpW : SDTypeProfile<1, 2, [
  SDTCisSameAs<0, 1>, SDTCisSameAs<0, 2>, SDTCisVT<0, i64>
]>;

def SDT_LoongArchBStrIns: SDTypeProfile<1, 4, [
  SDTCisInt<0>, SDTCisSameAs<0, 1>, SDTCisSameAs<0, 2>, SDTCisInt<3>,
  SDTCisSameAs<3, 4>
]>;

def SDT_LoongArchBStrPick: SDTypeProfile<1, 3, [
  SDTCisInt<0>, SDTCisSameAs<0, 1>, SDTCisInt<2>, SDTCisSameAs<2, 3>
]>;

// TODO: Add LoongArch specific DAG Nodes
// Target-independent nodes, but with target-specific formats.
def callseq_start : SDNode<"ISD::CALLSEQ_START", SDT_CallSeqStart,
                           [SDNPHasChain, SDNPOutGlue]>;
def callseq_end : SDNode<"ISD::CALLSEQ_END", SDT_CallSeqEnd,
                         [SDNPHasChain, SDNPOptInGlue, SDNPOutGlue]>;

// Target-dependent nodes.
def loongarch_call : SDNode<"LoongArchISD::CALL", SDT_LoongArchCall,
                            [SDNPHasChain, SDNPOptInGlue, SDNPOutGlue,
                             SDNPVariadic]>;
def loongarch_ret : SDNode<"LoongArchISD::RET", SDTNone,
                           [SDNPHasChain, SDNPOptInGlue, SDNPVariadic]>;
def loongarch_sll_w : SDNode<"LoongArchISD::SLL_W", SDT_LoongArchIntBinOpW>;
def loongarch_sra_w : SDNode<"LoongArchISD::SRA_W", SDT_LoongArchIntBinOpW>;
def loongarch_srl_w : SDNode<"LoongArchISD::SRL_W", SDT_LoongArchIntBinOpW>;
def loongarch_rotr_w : SDNode<"LoongArchISD::ROTR_W", SDT_LoongArchIntBinOpW>;
def loongarch_rotl_w : SDNode<"LoongArchISD::ROTL_W", SDT_LoongArchIntBinOpW>;
def loongarch_bstrins
    : SDNode<"LoongArchISD::BSTRINS", SDT_LoongArchBStrIns>;
def loongarch_bstrpick
    : SDNode<"LoongArchISD::BSTRPICK", SDT_LoongArchBStrPick>;
def loongarch_revb_2h : SDNode<"LoongArchISD::REVB_2H", SDTUnaryOp>;
def loongarch_revb_2w : SDNode<"LoongArchISD::REVB_2W", SDTUnaryOp>;
def loongarch_bitrev_4b : SDNode<"LoongArchISD::BITREV_4B", SDTUnaryOp>;
def loongarch_bitrev_w : SDNode<"LoongArchISD::BITREV_W", SDTUnaryOp>;
def loongarch_clzw : SDNode<"LoongArchISD::CLZ_W", SDTIntBitCountUnaryOp>;
def loongarch_ctzw : SDNode<"LoongArchISD::CTZ_W", SDTIntBitCountUnaryOp>;

//===----------------------------------------------------------------------===//
// Operand and SDNode transformation definitions.
//===----------------------------------------------------------------------===//

class ImmAsmOperand<string prefix, int width, string suffix>
    : AsmOperandClass {
  let Name = prefix # "Imm" # width # suffix;
  let DiagnosticType = !strconcat("Invalid", Name);
  let RenderMethod = "addImmOperands";
}

class SImmAsmOperand<int width, string suffix = "">
    : ImmAsmOperand<"S", width, suffix> {
}

class UImmAsmOperand<int width, string suffix = "">
    : ImmAsmOperand<"U", width, suffix> {
}

def uimm2 : Operand<GRLenVT> {
  let ParserMatchClass = UImmAsmOperand<2>;
}

def uimm2_plus1 : Operand<GRLenVT> {
  let ParserMatchClass = UImmAsmOperand<2, "plus1">;
  let EncoderMethod = "getImmOpValueSub1";
  let DecoderMethod = "decodeUImmOperand<2, 1>";
}

def uimm3 : Operand<GRLenVT> {
  let ParserMatchClass = UImmAsmOperand<3>;
}

def uimm5 : Operand<GRLenVT>, ImmLeaf<GRLenVT, [{return isUInt<5>(Imm);}]> {
  let ParserMatchClass = UImmAsmOperand<5>;
}

def uimm6 : Operand<GRLenVT>, ImmLeaf<GRLenVT, [{return isUInt<6>(Imm);}]> {
  let ParserMatchClass = UImmAsmOperand<6>;
}

def uimm8 : Operand<GRLenVT> {
  let ParserMatchClass = UImmAsmOperand<8>;
}

def uimm12 : Operand<GRLenVT>, ImmLeaf<GRLenVT, [{return isUInt<12>(Imm);}]> {
  let ParserMatchClass = UImmAsmOperand<12>;
}

def uimm14 : Operand<GRLenVT> {
  let ParserMatchClass = UImmAsmOperand<14>;
}

def uimm15 : Operand<GRLenVT> {
  let ParserMatchClass = UImmAsmOperand<15>;
}

def simm12 : Operand<GRLenVT>, ImmLeaf<GRLenVT, [{return isInt<12>(Imm);}]> {
  let ParserMatchClass = SImmAsmOperand<12>;
  let DecoderMethod = "decodeSImmOperand<12>";
}

def simm14_lsl2 : Operand<GRLenVT>,
    ImmLeaf<GRLenVT, [{return isShiftedInt<14,2>(Imm);}]> {
  let ParserMatchClass = SImmAsmOperand<14, "lsl2">;
  let EncoderMethod = "getImmOpValueAsr2";
  let DecoderMethod = "decodeSImmOperand<14, 2>";
}

def simm16 : Operand<GRLenVT> {
  let ParserMatchClass = SImmAsmOperand<16>;
  let DecoderMethod = "decodeSImmOperand<16>";
}

def simm16_lsl2 : Operand<GRLenVT>,
    ImmLeaf<GRLenVT, [{return isInt<16>(Imm>>2);}]> {
  let ParserMatchClass = SImmAsmOperand<16, "lsl2">;
  let EncoderMethod = "getImmOpValueAsr2";
  let DecoderMethod = "decodeSImmOperand<16, 2>";
}

def simm16_lsl2_br : Operand<OtherVT> {
  let ParserMatchClass = SImmAsmOperand<16, "lsl2">;
  let EncoderMethod = "getImmOpValueAsr2";
  let DecoderMethod = "decodeSImmOperand<16, 2>";
}

class SImm20Operand : Operand<GRLenVT> {
  let DecoderMethod = "decodeSImmOperand<20>";
}

def simm20 : SImm20Operand {
  let ParserMatchClass = SImmAsmOperand<20>;
}

def simm20_pcalau12i : SImm20Operand {
  let ParserMatchClass = SImmAsmOperand<20, "pcalau12i">;
}

def simm21_lsl2 : Operand<OtherVT> {
  let ParserMatchClass = SImmAsmOperand<21, "lsl2">;
  let EncoderMethod = "getImmOpValueAsr2";
  let DecoderMethod = "decodeSImmOperand<21, 2>";
}

// TODO: Need split the ParserMethod/PredicateMethod for call/jump/tailcall.
def SImm26Operand: AsmOperandClass {
  let Name = "SImm26Operand";
  let RenderMethod = "addImmOperands";
  let DiagnosticType = "InvalidSImm26Operand";
  let ParserMethod = "parseSImm26Operand";
}

// A symbol or an imm used in B/PseudoBR.
def simm26_b : Operand<OtherVT> {
  let ParserMatchClass = SImm26Operand;
  let EncoderMethod = "getImmOpValueAsr2";
  let DecoderMethod = "decodeSImmOperand<26, 2>";
}

// A symbol or an imm used in BL/PseudoCALL.
def simm26_bl : Operand<GRLenVT> {
  let ParserMatchClass = SImm26Operand;
  let EncoderMethod = "getImmOpValueAsr2";
  let DecoderMethod = "decodeSImmOperand<26, 2>";
}

// Standalone (codegen-only) immleaf patterns.

// A 12-bit signed immediate plus one where the imm range will be [-2047, 2048].
def simm12_plus1 : ImmLeaf<GRLenVT,
  [{return (isInt<12>(Imm) && Imm != -2048) || Imm == 2048;}]>;

// Return the negation of an immediate value.
def NegImm : SDNodeXForm<imm, [{
  return CurDAG->getTargetConstant(-N->getSExtValue(), SDLoc(N),
                                   N->getValueType(0));
}]>;

// FP immediate patterns.
def fpimm0    : PatLeaf<(fpimm), [{return N->isExactlyValue(+0.0);}]>;
def fpimm0neg : PatLeaf<(fpimm), [{return N->isExactlyValue(-0.0);}]>;
def fpimm1    : PatLeaf<(fpimm), [{return N->isExactlyValue(+1.0);}]>;

// Return an immediate subtracted from 32.
def ImmSubFrom32 : SDNodeXForm<imm, [{
  return CurDAG->getTargetConstant(32 - N->getZExtValue(), SDLoc(N),
                                   N->getValueType(0));
}]>;

def BaseAddr : ComplexPattern<iPTR, 1, "SelectBaseAddr">;

//===----------------------------------------------------------------------===//
// Instruction Formats
//===----------------------------------------------------------------------===//

include "LoongArchInstrFormats.td"
include "LoongArchFloatInstrFormats.td"

//===----------------------------------------------------------------------===//
// Instruction Class Templates
//===----------------------------------------------------------------------===//

class ALU_3R<bits<17> op, string opstr>
    : Fmt3R<op, (outs GPR:$rd), (ins GPR:$rj, GPR:$rk), opstr, "$rd, $rj, $rk">;
class ALU_2R<bits<22> op, string opstr>
    : Fmt2R<op, (outs GPR:$rd), (ins GPR:$rj), opstr, "$rd, $rj">;

class ALU_3RI2<bits<15> op, string opstr, Operand ImmOpnd>
    : Fmt3RI2<op, (outs GPR:$rd), (ins GPR:$rj, GPR:$rk, ImmOpnd:$imm2), opstr,
              "$rd, $rj, $rk, $imm2">;
class ALU_3RI3<bits<14> op, string opstr, Operand ImmOpnd>
    : Fmt3RI3<op, (outs GPR:$rd), (ins GPR:$rj, GPR:$rk, ImmOpnd:$imm3), opstr,
              "$rd, $rj, $rk, $imm3">;
class ALU_2RI5<bits<17> op, string opstr, Operand ImmOpnd>
    : Fmt2RI5<op, (outs GPR:$rd), (ins GPR:$rj, ImmOpnd:$imm5), opstr,
              "$rd, $rj, $imm5">;
class ALU_2RI6<bits<16> op, string opstr, Operand ImmOpnd>
    : Fmt2RI6<op, (outs GPR:$rd), (ins GPR:$rj, ImmOpnd:$imm6), opstr,
              "$rd, $rj, $imm6">;
class ALU_2RI12<bits<10> op, string opstr, Operand ImmOpnd>
    : Fmt2RI12<op, (outs GPR:$rd), (ins GPR:$rj, ImmOpnd:$imm12), opstr,
               "$rd, $rj, $imm12">;
class ALU_2RI16<bits<6> op, string opstr, Operand ImmOpnd>
    : Fmt2RI16<op, (outs GPR:$rd), (ins GPR:$rj, ImmOpnd:$imm16), opstr,
               "$rd, $rj, $imm16">;
class ALU_1RI20<bits<7> op, string opstr, Operand ImmOpnd>
    : Fmt1RI20<op, (outs GPR:$rd), (ins ImmOpnd:$imm20), opstr, "$rd, $imm20">;

class MISC_I15<bits<17> op, string opstr>
    : FmtI15<op, (outs), (ins uimm15:$imm15), opstr, "$imm15">;

class RDTIME_2R<bits<22> op, string opstr>
    : Fmt2R<op, (outs GPR:$rd, GPR:$rj), (ins), opstr, "$rd, $rj">;

class BrCC_2RI16<bits<6> op, string opstr>
    : Fmt2RI16<op, (outs), (ins GPR:$rj, GPR:$rd, simm16_lsl2_br:$imm16), opstr,
               "$rj, $rd, $imm16"> {
  let isBranch = 1;
  let isTerminator = 1;
}
class BrCCZ_1RI21<bits<6> op, string opstr>
    : Fmt1RI21<op, (outs), (ins GPR:$rj, simm21_lsl2:$imm21), opstr,
               "$rj, $imm21"> {
  let isBranch = 1;
  let isTerminator = 1;
}
class Br_I26<bits<6> op, string opstr>
    : FmtI26<op, (outs), (ins simm26_b:$imm26), opstr, "$imm26"> {
  let isBranch = 1;
  let isTerminator = 1;
}

let mayLoad = 1 in {
class LOAD_3R<bits<17> op, string opstr>
    : Fmt3R<op, (outs GPR:$rd), (ins GPR:$rj, GPR:$rk), opstr, "$rd, $rj, $rk">;
class LOAD_2RI12<bits<10> op, string opstr>
    : Fmt2RI12<op, (outs GPR:$rd), (ins GPR:$rj, simm12:$imm12), opstr,
               "$rd, $rj, $imm12">;
class LOAD_2RI14<bits<8> op, string opstr>
    : Fmt2RI14<op, (outs GPR:$rd), (ins GPR:$rj, simm14_lsl2:$imm14), opstr,
               "$rd, $rj, $imm14">;
} // mayLoad = 1

let mayStore = 1 in {
class STORE_3R<bits<17> op, string opstr>
    : Fmt3R<op, (outs), (ins GPR:$rd, GPR:$rj, GPR:$rk), opstr,
            "$rd, $rj, $rk">;
class STORE_2RI12<bits<10> op, string opstr>
    : Fmt2RI12<op, (outs), (ins GPR:$rd, GPR:$rj, simm12:$imm12), opstr,
               "$rd, $rj, $imm12">;
class STORE_2RI14<bits<8> op, string opstr>
    : Fmt2RI14<op, (outs), (ins GPR:$rd, GPR:$rj, simm14_lsl2:$imm14), opstr,
               "$rd, $rj, $imm14">;
} // mayStore = 1

let mayLoad = 1, mayStore = 1 in
class AM_3R<bits<17> op, string opstr>
    : Fmt3R<op, (outs GPR:$rd), (ins GPR:$rk, GPR:$rj), opstr, "$rd, $rk, $rj">;

let mayLoad = 1 in
class LLBase<bits<8> op, string opstr>
    : Fmt2RI14<op, (outs GPR:$rd), (ins GPR:$rj, simm14_lsl2:$imm14), opstr,
               "$rd, $rj, $imm14">;

let mayStore = 1, Constraints = "$rd = $dst" in
class SCBase<bits<8> op, string opstr>
    : Fmt2RI14<op, (outs GPR:$dst), (ins GPR:$rd, GPR:$rj, simm14_lsl2:$imm14),
               opstr, "$rd, $rj, $imm14">;

class IOCSRRD<bits<22> op, string opstr>
    : Fmt2R<op, (outs GPR:$rd), (ins GPR:$rj), opstr, "$rd, $rj">;

class IOCSRWR<bits<22> op, string opstr>
    : Fmt2R<op, (outs), (ins GPR:$rd, GPR:$rj), opstr, "$rd, $rj">;

//===----------------------------------------------------------------------===//
// Basic Integer Instructions
//===----------------------------------------------------------------------===//

// Arithmetic Operation Instructions
def ADD_W : ALU_3R<0b00000000000100000, "add.w">;
def SUB_W : ALU_3R<0b00000000000100010, "sub.w">;
def ADDI_W : ALU_2RI12<0b0000001010, "addi.w", simm12>;
def ALSL_W : ALU_3RI2<0b000000000000010, "alsl.w", uimm2_plus1>;
def LU12I_W : ALU_1RI20<0b0001010, "lu12i.w", simm20>;
def SLT  : ALU_3R<0b00000000000100100, "slt">;
def SLTU : ALU_3R<0b00000000000100101, "sltu">;
def SLTI  : ALU_2RI12<0b0000001000, "slti", simm12>;
def SLTUI : ALU_2RI12<0b0000001001, "sltui", simm12>;
def PCADDI    : ALU_1RI20<0b0001100, "pcaddi", simm20>;
def PCADDU12I : ALU_1RI20<0b0001110, "pcaddu12i", simm20>;
def PCALAU12I : ALU_1RI20<0b0001101, "pcalau12i", simm20_pcalau12i>;
def AND  : ALU_3R<0b00000000000101001, "and">;
def OR   : ALU_3R<0b00000000000101010, "or">;
def NOR  : ALU_3R<0b00000000000101000, "nor">;
def XOR  : ALU_3R<0b00000000000101011, "xor">;
def ANDN : ALU_3R<0b00000000000101101, "andn">;
def ORN  : ALU_3R<0b00000000000101100, "orn">;
def ANDI : ALU_2RI12<0b0000001101, "andi", uimm12>;
def ORI  : ALU_2RI12<0b0000001110, "ori", uimm12>;
def XORI : ALU_2RI12<0b0000001111, "xori", uimm12>;
def MUL_W   : ALU_3R<0b00000000000111000, "mul.w">;
def MULH_W  : ALU_3R<0b00000000000111001, "mulh.w">;
def MULH_WU : ALU_3R<0b00000000000111010, "mulh.wu">;
let usesCustomInserter = true in {
def DIV_W   : ALU_3R<0b00000000001000000, "div.w">;
def MOD_W   : ALU_3R<0b00000000001000001, "mod.w">;
def DIV_WU  : ALU_3R<0b00000000001000010, "div.wu">;
def MOD_WU  : ALU_3R<0b00000000001000011, "mod.wu">;
} // usesCustomInserter = true

// Bit-shift Instructions
def SLL_W  : ALU_3R<0b00000000000101110, "sll.w">;
def SRL_W  : ALU_3R<0b00000000000101111, "srl.w">;
def SRA_W  : ALU_3R<0b00000000000110000, "sra.w">;
def ROTR_W : ALU_3R<0b00000000000110110, "rotr.w">;

def SLLI_W  : ALU_2RI5<0b00000000010000001, "slli.w", uimm5>;
def SRLI_W  : ALU_2RI5<0b00000000010001001, "srli.w", uimm5>;
def SRAI_W  : ALU_2RI5<0b00000000010010001, "srai.w", uimm5>;
def ROTRI_W : ALU_2RI5<0b00000000010011001, "rotri.w", uimm5>;

// Bit-manipulation Instructions
def EXT_W_B : ALU_2R<0b0000000000000000010111, "ext.w.b">;
def EXT_W_H : ALU_2R<0b0000000000000000010110, "ext.w.h">;
def CLO_W   : ALU_2R<0b0000000000000000000100, "clo.w">;
def CLZ_W   : ALU_2R<0b0000000000000000000101, "clz.w">;
def CTO_W   : ALU_2R<0b0000000000000000000110, "cto.w">;
def CTZ_W   : ALU_2R<0b0000000000000000000111, "ctz.w">;
def BYTEPICK_W : ALU_3RI2<0b000000000000100, "bytepick.w", uimm2>;
def REVB_2H   : ALU_2R<0b0000000000000000001100, "revb.2h">;
def BITREV_4B : ALU_2R<0b0000000000000000010010, "bitrev.4b">;
def BITREV_W  : ALU_2R<0b0000000000000000010100, "bitrev.w">;
let Constraints = "$rd = $dst" in {
def BSTRINS_W  : FmtBSTR_W<0b000000000110, (outs GPR:$dst),
                           (ins GPR:$rd, GPR:$rj, uimm5:$msbw, uimm5:$lsbw),
                           "bstrins.w", "$rd, $rj, $msbw, $lsbw">;
}
def BSTRPICK_W : FmtBSTR_W<0b000000000111, (outs GPR:$rd),
                           (ins GPR:$rj, uimm5:$msbw, uimm5:$lsbw),
                           "bstrpick.w", "$rd, $rj, $msbw, $lsbw">;
def MASKEQZ : ALU_3R<0b00000000000100110, "maskeqz">;
def MASKNEZ : ALU_3R<0b00000000000100111, "masknez">;

// Branch Instructions
def BEQ  : BrCC_2RI16<0b010110, "beq">;
def BNE  : BrCC_2RI16<0b010111, "bne">;
def BLT  : BrCC_2RI16<0b011000, "blt">;
def BGE  : BrCC_2RI16<0b011001, "bge">;
def BLTU : BrCC_2RI16<0b011010, "bltu">;
def BGEU : BrCC_2RI16<0b011011, "bgeu">;
def BEQZ : BrCCZ_1RI21<0b010000, "beqz">;
def BNEZ : BrCCZ_1RI21<0b010001, "bnez">;
def B : Br_I26<0b010100, "b">;

let isCall = 1, Defs=[R1] in
def BL : FmtI26<0b010101, (outs), (ins simm26_bl:$imm26), "bl", "$imm26">;
def JIRL : Fmt2RI16<0b010011, (outs GPR:$rd),
                    (ins GPR:$rj, simm16_lsl2:$imm16), "jirl",
                    "$rd, $rj, $imm16">;

// Common Memory Access Instructions
def LD_B  : LOAD_2RI12<0b0010100000, "ld.b">;
def LD_H  : LOAD_2RI12<0b0010100001, "ld.h">;
def LD_W  : LOAD_2RI12<0b0010100010, "ld.w">;
def LD_BU : LOAD_2RI12<0b0010101000, "ld.bu">;
def LD_HU : LOAD_2RI12<0b0010101001, "ld.hu">;
def ST_B : STORE_2RI12<0b0010100100, "st.b">;
def ST_H : STORE_2RI12<0b0010100101, "st.h">;
def ST_W : STORE_2RI12<0b0010100110, "st.w">;
def PRELD : FmtPRELD<(outs), (ins uimm5:$imm5, GPR:$rj, simm12:$imm12), "preld",
                     "$imm5, $rj, $imm12">;

// Atomic Memory Access Instructions
def LL_W : LLBase<0b00100000, "ll.w">;
def SC_W : SCBase<0b00100001, "sc.w">;

// Barrier Instructions
def DBAR : MISC_I15<0b00111000011100100, "dbar">;
def IBAR : MISC_I15<0b00111000011100101, "ibar">;

// Other Miscellaneous Instructions
def SYSCALL : MISC_I15<0b00000000001010110, "syscall">;
def BREAK   : MISC_I15<0b00000000001010100, "break">;
def RDTIMEL_W : RDTIME_2R<0b0000000000000000011000, "rdtimel.w">;
def RDTIMEH_W : RDTIME_2R<0b0000000000000000011001, "rdtimeh.w">;
def CPUCFG : ALU_2R<0b0000000000000000011011, "cpucfg">;

/// LA64 instructions

let Predicates = [IsLA64] in {

// Arithmetic Operation Instructions for 64-bits
def ADD_D : ALU_3R<0b00000000000100001, "add.d">;
def SUB_D : ALU_3R<0b00000000000100011, "sub.d">;
def ADDI_D : ALU_2RI12<0b0000001011, "addi.d", simm12>;
def ADDU16I_D : ALU_2RI16<0b000100, "addu16i.d", simm16>;
def ALSL_WU : ALU_3RI2<0b000000000000011, "alsl.wu", uimm2_plus1>;
def ALSL_D  : ALU_3RI2<0b000000000010110, "alsl.d", uimm2_plus1>;
let Constraints = "$rd = $dst" in {
def LU32I_D : Fmt1RI20<0b0001011, (outs GPR:$dst),
                       (ins GPR:$rd, simm20:$imm20), "lu32i.d",
                       "$rd, $imm20">;
}
def LU52I_D : ALU_2RI12<0b0000001100, "lu52i.d", simm12>;
def PCADDU18I : ALU_1RI20<0b0001111, "pcaddu18i", simm20>;
def MUL_D     : ALU_3R<0b00000000000111011, "mul.d">;
def MULH_D    : ALU_3R<0b00000000000111100, "mulh.d">;
def MULH_DU   : ALU_3R<0b00000000000111101, "mulh.du">;
def MULW_D_W  : ALU_3R<0b00000000000111110, "mulw.d.w">;
def MULW_D_WU : ALU_3R<0b00000000000111111, "mulw.d.wu">;
let usesCustomInserter = true in {
def DIV_D     : ALU_3R<0b00000000001000100, "div.d">;
def MOD_D     : ALU_3R<0b00000000001000101, "mod.d">;
def DIV_DU    : ALU_3R<0b00000000001000110, "div.du">;
def MOD_DU    : ALU_3R<0b00000000001000111, "mod.du">;
} // usesCustomInserter = true

// Bit-shift Instructions for 64-bits
def SLL_D  : ALU_3R<0b00000000000110001, "sll.d">;
def SRL_D  : ALU_3R<0b00000000000110010, "srl.d">;
def SRA_D  : ALU_3R<0b00000000000110011, "sra.d">;
def ROTR_D : ALU_3R<0b00000000000110111, "rotr.d">;
def SLLI_D  : ALU_2RI6<0b0000000001000001, "slli.d", uimm6>;
def SRLI_D  : ALU_2RI6<0b0000000001000101, "srli.d", uimm6>;
def SRAI_D  : ALU_2RI6<0b0000000001001001, "srai.d", uimm6>;
def ROTRI_D : ALU_2RI6<0b0000000001001101, "rotri.d", uimm6>;

// Bit-manipulation Instructions for 64-bits
def CLO_D : ALU_2R<0b0000000000000000001000, "clo.d">;
def CLZ_D : ALU_2R<0b0000000000000000001001, "clz.d">;
def CTO_D : ALU_2R<0b0000000000000000001010, "cto.d">;
def CTZ_D : ALU_2R<0b0000000000000000001011, "ctz.d">;
def BYTEPICK_D : ALU_3RI3<0b00000000000011, "bytepick.d", uimm3>;
def REVB_4H   : ALU_2R<0b0000000000000000001101, "revb.4h">;
def REVB_2W   : ALU_2R<0b0000000000000000001110, "revb.2w">;
def REVB_D    : ALU_2R<0b0000000000000000001111, "revb.d">;
def REVH_2W   : ALU_2R<0b0000000000000000010000, "revh.2w">;
def REVH_D    : ALU_2R<0b0000000000000000010001, "revh.d">;
def BITREV_8B : ALU_2R<0b0000000000000000010011, "bitrev.8b">;
def BITREV_D  : ALU_2R<0b0000000000000000010101, "bitrev.d">;
let Constraints = "$rd = $dst" in {
def BSTRINS_D  : FmtBSTR_D<0b0000000010, (outs GPR:$dst),
                           (ins GPR:$rd, GPR:$rj, uimm6:$msbd, uimm6:$lsbd),
                           "bstrins.d", "$rd, $rj, $msbd, $lsbd">;
}
def BSTRPICK_D : FmtBSTR_D<0b0000000011, (outs GPR:$rd),
                           (ins GPR:$rj, uimm6:$msbd, uimm6:$lsbd),
                           "bstrpick.d", "$rd, $rj, $msbd, $lsbd">;

// Common Memory Access Instructions for 64-bits
def LD_WU : LOAD_2RI12<0b0010101010, "ld.wu">;
def LD_D  : LOAD_2RI12<0b0010100011, "ld.d">;
def ST_D : STORE_2RI12<0b0010100111, "st.d">;
def LDX_B  : LOAD_3R<0b00111000000000000, "ldx.b">;
def LDX_H  : LOAD_3R<0b00111000000001000, "ldx.h">;
def LDX_W  : LOAD_3R<0b00111000000010000, "ldx.w">;
def LDX_D  : LOAD_3R<0b00111000000011000, "ldx.d">;
def LDX_BU : LOAD_3R<0b00111000001000000, "ldx.bu">;
def LDX_HU : LOAD_3R<0b00111000001001000, "ldx.hu">;
def LDX_WU : LOAD_3R<0b00111000001010000, "ldx.wu">;
def STX_B : STORE_3R<0b00111000000100000, "stx.b">;
def STX_H : STORE_3R<0b00111000000101000, "stx.h">;
def STX_W : STORE_3R<0b00111000000110000, "stx.w">;
def STX_D : STORE_3R<0b00111000000111000, "stx.d">;
def LDPTR_W : LOAD_2RI14<0b00100100, "ldptr.w">;
def LDPTR_D : LOAD_2RI14<0b00100110, "ldptr.d">;
def STPTR_W : STORE_2RI14<0b00100101, "stptr.w">;
def STPTR_D : STORE_2RI14<0b00100111, "stptr.d">;
def PRELDX : FmtPRELDX<(outs), (ins uimm5:$imm5, GPR:$rj, GPR:$rk), "preldx",
                       "$imm5, $rj, $rk">;

// Bound Check Memory Access Instructions
def LDGT_B : LOAD_3R<0b00111000011110000, "ldgt.b">;
def LDGT_H : LOAD_3R<0b00111000011110001, "ldgt.h">;
def LDGT_W : LOAD_3R<0b00111000011110010, "ldgt.w">;
def LDGT_D : LOAD_3R<0b00111000011110011, "ldgt.d">;
def LDLE_B : LOAD_3R<0b00111000011110100, "ldle.b">;
def LDLE_H : LOAD_3R<0b00111000011110101, "ldle.h">;
def LDLE_W : LOAD_3R<0b00111000011110110, "ldle.w">;
def LDLE_D : LOAD_3R<0b00111000011110111, "ldle.d">;
def STGT_B : STORE_3R<0b00111000011111000, "stgt.b">;
def STGT_H : STORE_3R<0b00111000011111001, "stgt.h">;
def STGT_W : STORE_3R<0b00111000011111010, "stgt.w">;
def STGT_D : STORE_3R<0b00111000011111011, "stgt.d">;
def STLE_B : STORE_3R<0b00111000011111100, "stle.b">;
def STLE_H : STORE_3R<0b00111000011111101, "stle.h">;
def STLE_W : STORE_3R<0b00111000011111110, "stle.w">;
def STLE_D : STORE_3R<0b00111000011111111, "stle.d">;

// Atomic Memory Access Instructions for 64-bits
def AMSWAP_W    : AM_3R<0b00111000011000000, "amswap.w">;
def AMSWAP_D    : AM_3R<0b00111000011000001, "amswap.d">;
def AMADD_W     : AM_3R<0b00111000011000010, "amadd.w">;
def AMADD_D     : AM_3R<0b00111000011000011, "amadd.d">;
def AMAND_W     : AM_3R<0b00111000011000100, "amand.w">;
def AMAND_D     : AM_3R<0b00111000011000101, "amand.d">;
def AMOR_W      : AM_3R<0b00111000011000110, "amor.w">;
def AMOR_D      : AM_3R<0b00111000011000111, "amor.d">;
def AMXOR_W     : AM_3R<0b00111000011001000, "amxor.w">;
def AMXOR_D     : AM_3R<0b00111000011001001, "amxor.d">;
def AMMAX_W     : AM_3R<0b00111000011001010, "ammax.w">;
def AMMAX_D     : AM_3R<0b00111000011001011, "ammax.d">;
def AMMIN_W     : AM_3R<0b00111000011001100, "ammin.w">;
def AMMIN_D     : AM_3R<0b00111000011001101, "ammin.d">;
def AMMAX_WU    : AM_3R<0b00111000011001110, "ammax.wu">;
def AMMAX_DU    : AM_3R<0b00111000011001111, "ammax.du">;
def AMMIN_WU    : AM_3R<0b00111000011010000, "ammin.wu">;
def AMMIN_DU    : AM_3R<0b00111000011010001, "ammin.du">;
def AMSWAP_DB_W : AM_3R<0b00111000011010010, "amswap_db.w">;
def AMSWAP_DB_D : AM_3R<0b00111000011010011, "amswap_db.d">;
def AMADD_DB_W  : AM_3R<0b00111000011010100, "amadd_db.w">;
def AMADD_DB_D  : AM_3R<0b00111000011010101, "amadd_db.d">;
def AMAND_DB_W  : AM_3R<0b00111000011010110, "amand_db.w">;
def AMAND_DB_D  : AM_3R<0b00111000011010111, "amand_db.d">;
def AMOR_DB_W   : AM_3R<0b00111000011011000, "amor_db.w">;
def AMOR_DB_D   : AM_3R<0b00111000011011001, "amor_db.d">;
def AMXOR_DB_W  : AM_3R<0b00111000011011010, "amxor_db.w">;
def AMXOR_DB_D  : AM_3R<0b00111000011011011, "amxor_db.d">;
def AMMAX_DB_W  : AM_3R<0b00111000011011100, "ammax_db.w">;
def AMMAX_DB_D  : AM_3R<0b00111000011011101, "ammax_db.d">;
def AMMIN_DB_W  : AM_3R<0b00111000011011110, "ammin_db.w">;
def AMMIN_DB_D  : AM_3R<0b00111000011011111, "ammin_db.d">;
def AMMAX_DB_WU : AM_3R<0b00111000011100000, "ammax_db.wu">;
def AMMAX_DB_DU : AM_3R<0b00111000011100001, "ammax_db.du">;
def AMMIN_DB_WU : AM_3R<0b00111000011100010, "ammin_db.wu">;
def AMMIN_DB_DU : AM_3R<0b00111000011100011, "ammin_db.du">;
def LL_D : LLBase<0b00100010, "ll.d">;
def SC_D : SCBase<0b00100011, "sc.d">;

// CRC Check Instructions
def CRC_W_B_W  : ALU_3R<0b00000000001001000, "crc.w.b.w">;
def CRC_W_H_W  : ALU_3R<0b00000000001001001, "crc.w.h.w">;
def CRC_W_W_W  : ALU_3R<0b00000000001001010, "crc.w.w.w">;
def CRC_W_D_W  : ALU_3R<0b00000000001001011, "crc.w.d.w">;
def CRCC_W_B_W : ALU_3R<0b00000000001001100, "crcc.w.b.w">;
def CRCC_W_H_W : ALU_3R<0b00000000001001101, "crcc.w.h.w">;
def CRCC_W_W_W : ALU_3R<0b00000000001001110, "crcc.w.w.w">;
def CRCC_W_D_W : ALU_3R<0b00000000001001111, "crcc.w.d.w">;

// Other Miscellaneous Instructions for 64-bits
def ASRTLE_D : FmtASRT<0b00000000000000010, (outs), (ins GPR:$rj, GPR:$rk),
                       "asrtle.d", "$rj, $rk">;
def ASRTGT_D : FmtASRT<0b00000000000000011, (outs), (ins GPR:$rj, GPR:$rk),
                       "asrtgt.d", "$rj, $rk">;
def RDTIME_D : RDTIME_2R<0b0000000000000000011010, "rdtime.d">;
} // Predicates = [IsLA64]

//===----------------------------------------------------------------------===//
// Pseudo-instructions and codegen patterns
//
// Naming convention: For 'generic' pattern classes, we use the naming
// convention PatTy1Ty2.
//===----------------------------------------------------------------------===//

/// Generic pattern classes

class PatGprGpr<SDPatternOperator OpNode, LAInst Inst>
    : Pat<(OpNode GPR:$rj, GPR:$rk), (Inst GPR:$rj, GPR:$rk)>;
class PatGprGpr_32<SDPatternOperator OpNode, LAInst Inst>
    : Pat<(sext_inreg (OpNode GPR:$rj, GPR:$rk), i32), (Inst GPR:$rj, GPR:$rk)>;
class PatGpr<SDPatternOperator OpNode, LAInst Inst>
    : Pat<(OpNode GPR:$rj), (Inst GPR:$rj)>;

class PatGprImm<SDPatternOperator OpNode, LAInst Inst, Operand ImmOpnd>
    : Pat<(OpNode GPR:$rj, ImmOpnd:$imm),
          (Inst GPR:$rj, ImmOpnd:$imm)>;
class PatGprImm_32<SDPatternOperator OpNode, LAInst Inst, Operand ImmOpnd>
    : Pat<(sext_inreg (OpNode GPR:$rj, ImmOpnd:$imm), i32),
          (Inst GPR:$rj, ImmOpnd:$imm)>;

/// Predicates
def AddLike: PatFrags<(ops node:$A, node:$B),
                      [(add node:$A, node:$B), (or node:$A, node:$B)], [{
    return N->getOpcode() == ISD::ADD || isOrEquivalentToAdd(N);
}]>;

/// Simple arithmetic operations

// Match both a plain shift and one where the shift amount is masked (this is
// typically introduced when the legalizer promotes the shift amount and
// zero-extends it). For LoongArch, the mask is unnecessary as shifts in the
// base ISA only read the least significant 5 bits (LA32) or 6 bits (LA64).
def shiftMaskGRLen
    : ComplexPattern<GRLenVT, 1, "selectShiftMaskGRLen", [], [], 0>;
def shiftMask32 : ComplexPattern<i64, 1, "selectShiftMask32", [], [], 0>;

def sexti32 : ComplexPattern<i64, 1, "selectSExti32">;
def zexti32 : ComplexPattern<i64, 1, "selectZExti32">;

class shiftop<SDPatternOperator operator>
    : PatFrag<(ops node:$val, node:$count),
              (operator node:$val, (GRLenVT (shiftMaskGRLen node:$count)))>;
class shiftopw<SDPatternOperator operator>
    : PatFrag<(ops node:$val, node:$count),
              (operator node:$val, (i64 (shiftMask32 node:$count)))>;

let Predicates = [IsLA32] in {
def : PatGprGpr<add, ADD_W>;
def : PatGprImm<add, ADDI_W, simm12>;
def : PatGprGpr<sub, SUB_W>;
def : PatGprGpr<sdiv, DIV_W>;
def : PatGprGpr<udiv, DIV_WU>;
def : PatGprGpr<srem, MOD_W>;
def : PatGprGpr<urem, MOD_WU>;
def : PatGprGpr<mul, MUL_W>;
def : PatGprGpr<mulhs, MULH_W>;
def : PatGprGpr<mulhu, MULH_WU>;
def : PatGprGpr<rotr, ROTR_W>;
def : PatGprImm<rotr, ROTRI_W, uimm5>;
} // Predicates = [IsLA32]

let Predicates = [IsLA64] in {
def : PatGprGpr<add, ADD_D>;
def : PatGprGpr_32<add, ADD_W>;
def : PatGprImm<add, ADDI_D, simm12>;
def : PatGprImm_32<add, ADDI_W, simm12>;
def : PatGprGpr<sub, SUB_D>;
def : PatGprGpr_32<sub, SUB_W>;
def : PatGprGpr<sdiv, DIV_D>;
def : PatGprGpr<udiv, DIV_DU>;
def : PatGprGpr<srem, MOD_D>;
def : PatGprGpr<urem, MOD_DU>;
def : PatGprGpr<rotr, ROTR_D>;
def : PatGprGpr<loongarch_rotr_w, ROTR_W>;
def : PatGprImm<rotr, ROTRI_D, uimm6>;
def : PatGprImm_32<rotr, ROTRI_W, uimm5>;
def : Pat<(loongarch_rotl_w GPR:$rj, uimm5:$imm),
          (ROTRI_W GPR:$rj, (ImmSubFrom32 uimm5:$imm))>;
def : Pat<(sext_inreg (loongarch_rotl_w GPR:$rj, uimm5:$imm), i32),
          (ROTRI_W GPR:$rj, (ImmSubFrom32 uimm5:$imm))>;
// TODO: Select "_W[U]" instructions for i32xi32 if only lower 32 bits of the
// product are used.
def : PatGprGpr<mul, MUL_D>;
def : PatGprGpr<mulhs, MULH_D>;
def : PatGprGpr<mulhu, MULH_DU>;
// Select MULW_D_W for calculating the full 64 bits product of i32xi32 signed
// multiplication.
def : Pat<(i64 (mul (sext_inreg GPR:$rj, i32), (sext_inreg GPR:$rk, i32))),
          (MULW_D_W GPR:$rj, GPR:$rk)>;
// Select MULW_D_WU for calculating the full 64 bits product of i32xi32
// unsigned multiplication.
def : Pat<(i64 (mul (loongarch_bstrpick GPR:$rj, (i64 31), (i64 0)),
                    (loongarch_bstrpick GPR:$rk, (i64 31), (i64 0)))),
          (MULW_D_WU GPR:$rj, GPR:$rk)>;
} // Predicates = [IsLA64]

def : PatGprGpr<and, AND>;
def : PatGprImm<and, ANDI, uimm12>;
def : PatGprGpr<or, OR>;
def : PatGprImm<or, ORI, uimm12>;
def : PatGprGpr<xor, XOR>;
def : PatGprImm<xor, XORI, uimm12>;
def : Pat<(not GPR:$rj), (NOR GPR:$rj, R0)>;
def : Pat<(not (or GPR:$rj, GPR:$rk)), (NOR GPR:$rj, GPR:$rk)>;
def : Pat<(or GPR:$rj, (not GPR:$rk)), (ORN GPR:$rj, GPR:$rk)>;
def : Pat<(and GPR:$rj, (not GPR:$rk)), (ANDN GPR:$rj, GPR:$rk)>;

/// Bit counting operations

let Predicates = [IsLA64] in {
def : PatGpr<ctlz, CLZ_D>;
def : PatGpr<cttz, CTZ_D>;
def : Pat<(ctlz (not GPR:$rj)), (CLO_D GPR:$rj)>;
def : Pat<(cttz (not GPR:$rj)), (CTO_D GPR:$rj)>;
def : PatGpr<loongarch_clzw, CLZ_W>;
def : PatGpr<loongarch_ctzw, CTZ_W>;
def : Pat<(loongarch_clzw (not GPR:$rj)), (CLO_W GPR:$rj)>;
def : Pat<(loongarch_ctzw (not GPR:$rj)), (CTO_W GPR:$rj)>;
} // Predicates = [IsLA64]

let Predicates = [IsLA32] in {
def : PatGpr<ctlz, CLZ_W>;
def : PatGpr<cttz, CTZ_W>;
def : Pat<(ctlz (not GPR:$rj)), (CLO_W GPR:$rj)>;
def : Pat<(cttz (not GPR:$rj)), (CTO_W GPR:$rj)>;
} // Predicates = [IsLA32]

/// FrameIndex calculations
let Predicates = [IsLA32] in {
def : Pat<(AddLike (i32 BaseAddr:$rj), simm12:$imm12),
          (ADDI_W (i32 BaseAddr:$rj), simm12:$imm12)>;
} // Predicates = [IsLA32]
let Predicates = [IsLA64] in {
def : Pat<(AddLike (i64 BaseAddr:$rj), simm12:$imm12),
          (ADDI_D (i64 BaseAddr:$rj), simm12:$imm12)>;
} // Predicates = [IsLA64]

/// Shift

let Predicates = [IsLA32] in {
def : PatGprGpr<shiftop<shl>, SLL_W>;
def : PatGprGpr<shiftop<sra>, SRA_W>;
def : PatGprGpr<shiftop<srl>, SRL_W>;
def : PatGprImm<shl, SLLI_W, uimm5>;
def : PatGprImm<sra, SRAI_W, uimm5>;
def : PatGprImm<srl, SRLI_W, uimm5>;
} // Predicates = [IsLA32]

let Predicates = [IsLA64] in {
def : PatGprGpr<shiftopw<loongarch_sll_w>, SLL_W>;
def : PatGprGpr<shiftopw<loongarch_sra_w>, SRA_W>;
def : PatGprGpr<shiftopw<loongarch_srl_w>, SRL_W>;
def : PatGprGpr<shiftop<shl>, SLL_D>;
def : PatGprGpr<shiftop<sra>, SRA_D>;
def : PatGprGpr<shiftop<srl>, SRL_D>;
def : PatGprImm<shl, SLLI_D, uimm6>;
def : PatGprImm<sra, SRAI_D, uimm6>;
def : PatGprImm<srl, SRLI_D, uimm6>;
} // Predicates = [IsLA64]

/// sext and zext

def : Pat<(sext_inreg GPR:$rj, i8), (EXT_W_B GPR:$rj)>;
def : Pat<(sext_inreg GPR:$rj, i16), (EXT_W_H GPR:$rj)>;

let Predicates = [IsLA64] in {
def : Pat<(sext_inreg GPR:$rj, i32), (ADDI_W GPR:$rj, 0)>;
} // Predicates = [IsLA64]

/// Setcc

def : PatGprGpr<setlt, SLT>;
def : PatGprImm<setlt, SLTI, simm12>;
def : PatGprGpr<setult, SLTU>;
def : PatGprImm<setult, SLTUI, simm12>;

// Define pattern expansions for setcc operations that aren't directly
// handled by a LoongArch instruction.
def : Pat<(seteq GPR:$rj, 0), (SLTUI GPR:$rj, 1)>;
def : Pat<(seteq GPR:$rj, GPR:$rk), (SLTUI (XOR GPR:$rj, GPR:$rk), 1)>;
let Predicates = [IsLA32] in {
def : Pat<(seteq GPR:$rj, simm12_plus1:$imm12),
          (SLTUI (ADDI_W GPR:$rj, (NegImm simm12_plus1:$imm12)), 1)>;
} // Predicates = [IsLA32]
let Predicates = [IsLA64] in {
def : Pat<(seteq GPR:$rj, simm12_plus1:$imm12),
          (SLTUI (ADDI_D GPR:$rj, (NegImm simm12_plus1:$imm12)), 1)>;
} // Predicates = [IsLA64]
def : Pat<(setne GPR:$rj, 0), (SLTU R0, GPR:$rj)>;
def : Pat<(setne GPR:$rj, GPR:$rk), (SLTU R0, (XOR GPR:$rj, GPR:$rk))>;
let Predicates = [IsLA32] in {
def : Pat<(setne GPR:$rj, simm12_plus1:$imm12),
          (SLTU R0, (ADDI_W GPR:$rj, (NegImm simm12_plus1:$imm12)))>;
} // Predicates = [IsLA32]
let Predicates = [IsLA64] in {
def : Pat<(setne GPR:$rj, simm12_plus1:$imm12),
          (SLTU R0, (ADDI_D GPR:$rj, (NegImm simm12_plus1:$imm12)))>;
} // Predicates = [IsLA64]
def : Pat<(setugt GPR:$rj, GPR:$rk), (SLTU GPR:$rk, GPR:$rj)>;
def : Pat<(setuge GPR:$rj, GPR:$rk), (XORI (SLTU GPR:$rj, GPR:$rk), 1)>;
def : Pat<(setule GPR:$rj, GPR:$rk), (XORI (SLTU GPR:$rk, GPR:$rj), 1)>;
def : Pat<(setgt GPR:$rj, GPR:$rk), (SLT GPR:$rk, GPR:$rj)>;
def : Pat<(setge GPR:$rj, GPR:$rk), (XORI (SLT GPR:$rj, GPR:$rk), 1)>;
def : Pat<(setle GPR:$rj, GPR:$rk), (XORI (SLT GPR:$rk, GPR:$rj), 1)>;

/// Select

def : Pat<(select GPR:$cond, GPR:$t, GPR:$f),
          (OR (MASKEQZ GPR:$t, GPR:$cond), (MASKNEZ GPR:$f, GPR:$cond))>;

/// Branches and jumps

class BccPat<PatFrag CondOp, LAInst Inst>
    : Pat<(brcond (GRLenVT (CondOp GPR:$rj, GPR:$rd)), bb:$imm16),
          (Inst GPR:$rj, GPR:$rd, bb:$imm16)>;

def : BccPat<seteq, BEQ>;
def : BccPat<setne, BNE>;
def : BccPat<setlt, BLT>;
def : BccPat<setge, BGE>;
def : BccPat<setult, BLTU>;
def : BccPat<setuge, BGEU>;

class BccSwapPat<PatFrag CondOp, LAInst InstBcc>
    : Pat<(brcond (GRLenVT (CondOp GPR:$rd, GPR:$rj)), bb:$imm16),
          (InstBcc GPR:$rj, GPR:$rd, bb:$imm16)>;

// Condition codes that don't have matching LoongArch branch instructions, but
// are trivially supported by swapping the two input operands.
def : BccSwapPat<setgt, BLT>;
def : BccSwapPat<setle, BGE>;
def : BccSwapPat<setugt, BLTU>;
def : BccSwapPat<setule, BGEU>;

// An extra pattern is needed for a brcond without a setcc (i.e. where the
// condition was calculated elsewhere).
def : Pat<(brcond GPR:$rj, bb:$imm21), (BNEZ GPR:$rj, bb:$imm21)>;

def : Pat<(brcond (GRLenVT (seteq GPR:$rj, 0)), bb:$imm21),
          (BEQZ GPR:$rj, bb:$imm21)>;
def : Pat<(brcond (GRLenVT (setne GPR:$rj, 0)), bb:$imm21),
          (BNEZ GPR:$rj, bb:$imm21)>;

let isBarrier = 1, isBranch = 1, isTerminator = 1 in
def PseudoBR : Pseudo<(outs), (ins simm26_b:$imm26), [(br bb:$imm26)]>,
               PseudoInstExpansion<(B simm26_b:$imm26)>;

let isBarrier = 1, isBranch = 1, isIndirectBranch = 1, isTerminator = 1 in
def PseudoBRIND : Pseudo<(outs), (ins GPR:$rj, simm16_lsl2:$imm16), []>,
                  PseudoInstExpansion<(JIRL R0, GPR:$rj, simm16_lsl2:$imm16)>;

def : Pat<(brind GPR:$rj), (PseudoBRIND GPR:$rj, 0)>;
def : Pat<(brind (add GPR:$rj, simm16_lsl2:$imm16)),
          (PseudoBRIND GPR:$rj, simm16_lsl2:$imm16)>;

let isCall = 1, Defs = [R1] in
def PseudoCALL : Pseudo<(outs), (ins simm26_bl:$func), []> {
  let AsmString = "bl\t$func";
}

def : Pat<(loongarch_call tglobaladdr:$func), (PseudoCALL tglobaladdr:$func)>;
def : Pat<(loongarch_call texternalsym:$func), (PseudoCALL texternalsym:$func)>;

let isCall = 1, Defs = [R1] in
def PseudoCALLIndirect : Pseudo<(outs), (ins GPR:$rj),
                                [(loongarch_call GPR:$rj)]>,
                         PseudoInstExpansion<(JIRL R1, GPR:$rj, 0)>;

let isBarrier = 1, isReturn = 1, isTerminator = 1 in
def PseudoRET : Pseudo<(outs), (ins), [(loongarch_ret)]>,
                PseudoInstExpansion<(JIRL R0, R1, 0)>;

/// BSTRINS and BSTRPICK

let Predicates = [IsLA32] in {
def : Pat<(loongarch_bstrins GPR:$rd, GPR:$rj, uimm5:$msbd, uimm5:$lsbd),
          (BSTRINS_W GPR:$rd, GPR:$rj, uimm5:$msbd, uimm5:$lsbd)>;
def : Pat<(loongarch_bstrpick GPR:$rj, uimm5:$msbd, uimm5:$lsbd),
          (BSTRPICK_W GPR:$rj, uimm5:$msbd, uimm5:$lsbd)>;
} // Predicates = [IsLA32]

let Predicates = [IsLA64] in {
def : Pat<(loongarch_bstrins GPR:$rd, GPR:$rj, uimm6:$msbd, uimm6:$lsbd),
          (BSTRINS_D GPR:$rd, GPR:$rj, uimm6:$msbd, uimm6:$lsbd)>;
def : Pat<(loongarch_bstrpick GPR:$rj, uimm6:$msbd, uimm6:$lsbd),
          (BSTRPICK_D GPR:$rj, uimm6:$msbd, uimm6:$lsbd)>;
} // Predicates = [IsLA64]

/// Byte-swapping and bit-reversal

def : Pat<(loongarch_revb_2h GPR:$rj), (REVB_2H GPR:$rj)>;
def : Pat<(loongarch_bitrev_4b GPR:$rj), (BITREV_4B GPR:$rj)>;

let Predicates = [IsLA32] in {
def : Pat<(bswap GPR:$rj), (ROTRI_W (REVB_2H GPR:$rj), 16)>;
def : Pat<(bitreverse GPR:$rj), (BITREV_W GPR:$rj)>;
def : Pat<(bswap (bitreverse GPR:$rj)), (BITREV_4B GPR:$rj)>;
def : Pat<(bitreverse (bswap GPR:$rj)), (BITREV_4B GPR:$rj)>;
} // Predicates = [IsLA32]

let Predicates = [IsLA64] in {
def : Pat<(loongarch_revb_2w GPR:$rj), (REVB_2W GPR:$rj)>;
def : Pat<(bswap GPR:$rj), (REVB_D GPR:$rj)>;
def : Pat<(loongarch_bitrev_w GPR:$rj), (BITREV_W GPR:$rj)>;
def : Pat<(bitreverse GPR:$rj), (BITREV_D GPR:$rj)>;
def : Pat<(bswap (bitreverse GPR:$rj)), (BITREV_8B GPR:$rj)>;
def : Pat<(bitreverse (bswap GPR:$rj)), (BITREV_8B GPR:$rj)>;
} // Predicates = [IsLA64]

/// Loads

multiclass LdPat<PatFrag LoadOp, LAInst Inst, ValueType vt = GRLenVT> {
  def : Pat<(vt (LoadOp BaseAddr:$rj)), (Inst BaseAddr:$rj, 0)>;
  def : Pat<(vt (LoadOp (AddLike BaseAddr:$rj, simm12:$imm12))),
            (Inst BaseAddr:$rj, simm12:$imm12)>;
}

defm : LdPat<sextloadi8, LD_B>;
defm : LdPat<extloadi8, LD_B>;
defm : LdPat<sextloadi16, LD_H>;
defm : LdPat<extloadi16, LD_H>;
defm : LdPat<load, LD_W>, Requires<[IsLA32]>;
defm : LdPat<zextloadi8, LD_BU>;
defm : LdPat<zextloadi16, LD_HU>;
let Predicates = [IsLA64] in {
defm : LdPat<sextloadi32, LD_W, i64>;
defm : LdPat<extloadi32, LD_W, i64>;
defm : LdPat<zextloadi32, LD_WU, i64>;
defm : LdPat<load, LD_D, i64>;
} // Predicates = [IsLA64]

// LA64 register-register-addressed loads
let Predicates = [IsLA64] in {
class RegRegLdPat<PatFrag LoadOp, LAInst Inst, ValueType vt>
  : Pat<(vt (LoadOp (add BaseAddr:$rj, GPR:$rk))),
        (Inst BaseAddr:$rj, GPR:$rk)>;

def : RegRegLdPat<extloadi8, LDX_B, i64>;
def : RegRegLdPat<sextloadi8, LDX_B, i64>;
def : RegRegLdPat<zextloadi8, LDX_BU, i64>;
def : RegRegLdPat<extloadi16, LDX_H, i64>;
def : RegRegLdPat<sextloadi16, LDX_H, i64>;
def : RegRegLdPat<zextloadi16, LDX_HU, i64>;
def : RegRegLdPat<extloadi32, LDX_W, i64>;
def : RegRegLdPat<sextloadi32, LDX_W, i64>;
def : RegRegLdPat<zextloadi32, LDX_WU, i64>;
def : RegRegLdPat<load, LDX_D, i64>;
} // Predicates = [IsLA64]

/// Stores

multiclass StPat<PatFrag StoreOp, LAInst Inst, RegisterClass StTy,
                 ValueType vt> {
  def : Pat<(StoreOp (vt StTy:$rd), BaseAddr:$rj),
            (Inst StTy:$rd, BaseAddr:$rj, 0)>;
  def : Pat<(StoreOp (vt StTy:$rd), (AddLike BaseAddr:$rj, simm12:$imm12)),
            (Inst StTy:$rd, BaseAddr:$rj, simm12:$imm12)>;
}

defm : StPat<truncstorei8, ST_B, GPR, GRLenVT>;
defm : StPat<truncstorei16, ST_H, GPR, GRLenVT>;
defm : StPat<store, ST_W, GPR, i32>, Requires<[IsLA32]>;
let Predicates = [IsLA64] in {
defm : StPat<truncstorei32, ST_W, GPR, i64>;
defm : StPat<store, ST_D, GPR, i64>;
} // Predicates = [IsLA64]

let Predicates = [IsLA64] in {
def : Pat<(i64 (sextloadi32 (AddLike BaseAddr:$rj, simm14_lsl2:$imm14))),
          (LDPTR_W BaseAddr:$rj, simm14_lsl2:$imm14)>;
def : Pat<(i64 (load (AddLike BaseAddr:$rj, simm14_lsl2:$imm14))),
          (LDPTR_D BaseAddr:$rj, simm14_lsl2:$imm14)>;
def : Pat<(truncstorei32 (i64 GPR:$rd),
                         (AddLike BaseAddr:$rj, simm14_lsl2:$imm14)),
          (STPTR_W GPR:$rd, BaseAddr:$rj, simm14_lsl2:$imm14)>;
def : Pat<(store (i64 GPR:$rd), (AddLike BaseAddr:$rj, simm14_lsl2:$imm14)),
          (STPTR_D GPR:$rd, BaseAddr:$rj, simm14_lsl2:$imm14)>;
} // Predicates = [IsLA64]

// LA64 register-register-addressed stores
let Predicates = [IsLA64] in {
class RegRegStPat<PatFrag StoreOp, LAInst Inst, RegisterClass StTy,
                  ValueType vt>
  : Pat<(StoreOp (vt StTy:$rd), (add BaseAddr:$rj, GPR:$rk)),
        (Inst StTy:$rd, BaseAddr:$rj, GPR:$rk)>;

def : RegRegStPat<truncstorei8, STX_B, GPR, i64>;
def : RegRegStPat<truncstorei16, STX_H, GPR, i64>;
def : RegRegStPat<truncstorei32, STX_W, GPR, i64>;
def : RegRegStPat<store, STX_D, GPR, i64>;
} // Predicates = [IsLA64]

/// Atomic loads and stores

def : Pat<(atomic_fence timm, timm), (DBAR 0)>;

defm : LdPat<atomic_load_8, LD_B>;
defm : LdPat<atomic_load_16, LD_H>;
defm : LdPat<atomic_load_32, LD_W>;

class release_seqcst_store<PatFrag base>
    : PatFrag<(ops node:$ptr, node:$val), (base node:$ptr, node:$val), [{
  AtomicOrdering Ordering = cast<AtomicSDNode>(N)->getSuccessOrdering();
  return isReleaseOrStronger(Ordering);
}]>;

class unordered_monotonic_store<PatFrag base>
    : PatFrag<(ops node:$ptr, node:$val), (base node:$ptr, node:$val), [{
  AtomicOrdering Ordering = cast<AtomicSDNode>(N)->getSuccessOrdering();
  return !isReleaseOrStronger(Ordering);
}]>;

def atomic_store_release_seqcst_32 : release_seqcst_store<atomic_store_32>;
def atomic_store_release_seqcst_64 : release_seqcst_store<atomic_store_64>;
def atomic_store_unordered_monotonic_32
    : unordered_monotonic_store<atomic_store_32>;
def atomic_store_unordered_monotonic_64
    : unordered_monotonic_store<atomic_store_64>;

defm : StPat<atomic_store_8, ST_B, GPR, GRLenVT>;
defm : StPat<atomic_store_16, ST_H, GPR, GRLenVT>;
defm : StPat<atomic_store_unordered_monotonic_32, ST_W, GPR, i32>,
             Requires<[IsLA32]>;

def PseudoAtomicStoreW : Pseudo<(outs GPR:$dst), (ins GPR:$rj, GPR:$rk), []>,
                                 PseudoInstExpansion<(AMSWAP_DB_W R0,
                                                      GPR:$rk, GPR:$rj)>;

def : Pat<(atomic_store_release_seqcst_32 GPR:$rj, GPR:$rk),
          (PseudoAtomicStoreW GPR:$rj, GPR:$rk)>;

let Predicates = [IsLA64] in {
def PseudoAtomicStoreD : Pseudo<(outs GPR:$dst), (ins GPR:$rj, GPR:$rk), []>,
                                 PseudoInstExpansion<(AMSWAP_DB_D R0,
                                                      GPR:$rk, GPR:$rj)>;

def : Pat<(atomic_store_release_seqcst_64 GPR:$rj, GPR:$rk),
          (PseudoAtomicStoreD GPR:$rj, GPR:$rk)>;

defm : LdPat<atomic_load_64, LD_D>;
defm : StPat<atomic_store_unordered_monotonic_32, ST_W, GPR, i64>;
defm : StPat<atomic_store_unordered_monotonic_64, ST_D, GPR, i64>;
} // Predicates = [IsLA64]

/// Other pseudo-instructions

// Pessimistically assume the stack pointer will be clobbered
let Defs = [R3], Uses = [R3] in {
def ADJCALLSTACKDOWN : Pseudo<(outs), (ins i32imm:$amt1, i32imm:$amt2),
                              [(callseq_start timm:$amt1, timm:$amt2)]>;
def ADJCALLSTACKUP   : Pseudo<(outs), (ins i32imm:$amt1, i32imm:$amt2),
                              [(callseq_end timm:$amt1, timm:$amt2)]>;
} // Defs = [R3], Uses = [R3]

//===----------------------------------------------------------------------===//
// Assembler Pseudo Instructions
//===----------------------------------------------------------------------===//

def : InstAlias<"nop", (ANDI R0, R0, 0)>;
def : InstAlias<"move $dst, $src", (OR GPR:$dst, GPR:$src, R0)>;
// `ret` is supported since binutils commit 20f2e2686c79a5ac (version 2.40 and
// later).
def : InstAlias<"ret", (JIRL R0, R1, 0)>;
def : InstAlias<"jr $rj", (JIRL R0, GPR:$rj, 0)>;

//===----------------------------------------------------------------------===//
// Basic Floating-Point Instructions
//===----------------------------------------------------------------------===//

include "LoongArchFloat32InstrInfo.td"
include "LoongArchFloat64InstrInfo.td"

//===----------------------------------------------------------------------===//
// Privilege Instructions
//===----------------------------------------------------------------------===//

// CSR Access Instructions
def CSRRD : FmtCSR<0b0000010000000, (outs GPR:$rd), (ins uimm14:$csr_num),
                   "csrrd", "$rd, $csr_num">;
let Constraints = "$rd = $dst" in {
def CSRWR : FmtCSR<0b0000010000001, (outs GPR:$dst),
                   (ins GPR:$rd, uimm14:$csr_num), "csrwr", "$rd, $csr_num">;
def CSRXCHG : FmtCSRXCHG<0b00000100, (outs GPR:$dst),
                         (ins GPR:$rd, GPR:$rj, uimm14:$csr_num),
                         "csrxchg", "$rd, $rj, $csr_num">;
} // Constraints = "$rd = $dst"

// IOCSR Access Instructions
def IOCSRRD_B : IOCSRRD<0b0000011001001000000000, "iocsrrd.b">;
def IOCSRRD_H : IOCSRRD<0b0000011001001000000001, "iocsrrd.h">;
def IOCSRRD_W : IOCSRRD<0b0000011001001000000010, "iocsrrd.w">;
def IOCSRWR_B : IOCSRWR<0b0000011001001000000100, "iocsrwr.b">;
def IOCSRWR_H : IOCSRWR<0b0000011001001000000101, "iocsrwr.h">;
def IOCSRWR_W : IOCSRWR<0b0000011001001000000110, "iocsrwr.w">;
let Predicates = [IsLA64] in {
def IOCSRRD_D : IOCSRRD<0b0000011001001000000011, "iocsrrd.d">;
def IOCSRWR_D : IOCSRWR<0b0000011001001000000111, "iocsrwr.d">;
} // Predicates = [IsLA64]

// Cache Maintenance Instructions
def CACOP : FmtCACOP<(outs), (ins uimm5:$op, GPR:$rj, simm12:$imm12), "cacop",
                     "$op, $rj, $imm12">;

// TLB Maintenance Instructions
def TLBSRCH  : FmtI32<0b00000110010010000010100000000000, "tlbsrch">;
def TLBRD    : FmtI32<0b00000110010010000010110000000000, "tlbrd">;
def TLBWR    : FmtI32<0b00000110010010000011000000000000, "tlbwr">;
def TLBFILL  : FmtI32<0b00000110010010000011010000000000, "tlbfill">;
def TLBCLR   : FmtI32<0b00000110010010000010000000000000, "tlbclr">;
def TLBFLUSH : FmtI32<0b00000110010010000010010000000000, "tlbflush">;
def INVTLB : FmtINVTLB<(outs), (ins GPR:$rk, GPR:$rj, uimm5:$op), "invtlb",
                       "$op, $rj, $rk">;

// Software Page Walking Instructions
def LDDIR : Fmt2RI8<0b00000110010000, (outs GPR:$rd),
                    (ins GPR:$rj, uimm8:$imm8), "lddir", "$rd, $rj, $imm8">;
def LDPTE : FmtLDPTE<(outs), (ins GPR:$rj, uimm8:$seq), "ldpte", "$rj, $seq">;


// Other Miscellaneous Instructions
def ERTN : FmtI32<0b00000110010010000011100000000000, "ertn">;
def DBCL : MISC_I15<0b00000000001010101, "dbcl">;
def IDLE : MISC_I15<0b00000110010010001, "idle">;

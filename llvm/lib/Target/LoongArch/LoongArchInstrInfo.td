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

// TODO: Add LoongArch specific DAG Nodes

//===----------------------------------------------------------------------===//
// Operand and SDNode transformation definitions.
//===----------------------------------------------------------------------===//
class ImmAsmOperand<string prefix, int width, string suffix> : AsmOperandClass {
  let Name = prefix # "Imm" # width # suffix;
  let DiagnosticType = !strconcat("Invalid", Name);
}

class SImmAsmOperand<int width, string suffix = "">
    : ImmAsmOperand<"S", width, suffix> {
}

class UImmAsmOperand<int width, string suffix = "">
    : ImmAsmOperand<"U", width, suffix> {
}

def uimm2 : Operand<GRLenVT>;
def uimm2_plus1 : Operand<GRLenVT>;
def uimm3 : Operand<GRLenVT>;
def uimm5 : Operand<GRLenVT>;
def uimm6 : Operand<GRLenVT>;
def uimm12 : Operand<GRLenVT>;
def uimm15 : Operand<GRLenVT>;
def simm12 : Operand<GRLenVT>, ImmLeaf<GRLenVT, [{return isInt<12>(Imm);}]>;
def simm14 : Operand<GRLenVT>;
def simm14_lsl2 : Operand<GRLenVT>;
def simm16 : Operand<GRLenVT>;
def simm16_lsl2 : Operand<GRLenVT>;
def simm20 : Operand<GRLenVT>;
def simm21_lsl2 : Operand<GRLenVT>;
def simm26_lsl2 : Operand<GRLenVT>;

//===----------------------------------------------------------------------===//
// Instruction Formats
//===----------------------------------------------------------------------===//

include "LoongArchInstrFormats.td"

//===----------------------------------------------------------------------===//
// Instruction Class Templates
//===----------------------------------------------------------------------===//
class ALU_3R<bits<17> op, string opstr>
    : Fmt3R<op, (outs GPR:$rd), (ins GPR:$rj, GPR:$rk),
            !strconcat(opstr, "\t$rd, $rj, $rk")>;
class ALU_2R<bits<22> op, string opstr>
    : Fmt2R<op, (outs GPR:$rd), (ins GPR:$rj),
            !strconcat(opstr, "\t$rd, $rj")>;

class ALU_3RI2<bits<15> op, string opstr, Operand ImmOpnd>
    : Fmt3RI2<op, (outs GPR:$rd), (ins GPR:$rj, GPR:$rk, ImmOpnd:$imm2),
              !strconcat(opstr, "\t$rd, $rj, $rk, $imm2")>;
class ALU_3RI3<bits<14> op, string opstr, Operand ImmOpnd>
    : Fmt3RI3<op, (outs GPR:$rd), (ins GPR:$rj, GPR:$rk, ImmOpnd:$imm3),
              !strconcat(opstr, "\t$rd, $rj, $rk, $imm3")>;
class ALU_2RI5<bits<17> op, string opstr, Operand ImmOpnd>
    : Fmt2RI5<op, (outs GPR:$rd), (ins GPR:$rj, ImmOpnd:$imm5),
              !strconcat(opstr, "\t$rd, $rj, $imm5")>;
class ALU_2RI6<bits<16> op, string opstr, Operand ImmOpnd>
    : Fmt2RI6<op, (outs GPR:$rd), (ins GPR:$rj, ImmOpnd:$imm6),
              !strconcat(opstr, "\t$rd, $rj, $imm6")>;
class ALU_2RI12<bits<10> op, string opstr, Operand ImmOpnd>
    : Fmt2RI12<op, (outs GPR:$rd), (ins GPR:$rj, ImmOpnd:$imm12),
               !strconcat(opstr, "\t$rd, $rj, $imm12")>;
class ALU_2RI16<bits<6> op, string opstr, Operand ImmOpnd>
    : Fmt2RI16<op, (outs GPR:$rd), (ins GPR:$rj, ImmOpnd:$imm16),
               !strconcat(opstr, "\t$rd, $rj, $imm16")>;
class ALU_1RI20<bits<7> op, string opstr, Operand ImmOpnd>
    : Fmt1RI20<op, (outs GPR:$rd), (ins ImmOpnd:$imm20),
               !strconcat(opstr, "\t$rd, $imm20")>;

class MISC_I15<bits<17> op, string opstr>
    : FmtI15<op, (outs), (ins uimm15:$imm15), !strconcat(opstr, "\t$imm15")>;

class RDTIME_2R<bits<22> op, string opstr>
    : Fmt2R<op, (outs GPR:$rd, GPR:$rj), (ins),
                 !strconcat(opstr, "\t$rd, $rj")>;

class BrCC_2RI16<bits<6> op, string opstr>
    : Fmt2RI16<op, (outs), (ins GPR:$rj, GPR:$rd, simm16_lsl2:$imm16),
               !strconcat(opstr, "\t$rj, $rd, $imm16")> {
  let isBranch = 1;
  let isTerminator = 1;
}
class BrCCZ_1RI21<bits<6> op, string opstr>
    : Fmt1RI21<op, (outs), (ins GPR:$rj, simm21_lsl2:$imm21),
               !strconcat(opstr, "\t$rj, $imm21")> {
  let isBranch = 1;
  let isTerminator = 1;
}
class Br_I26<bits<6> op, string opstr>
    : FmtI26<op, (outs), (ins simm26_lsl2:$imm26),
             !strconcat(opstr, "\t$imm26")> {
  let isBranch = 1;
  let isTerminator = 1;
}

let mayLoad = 1 in {
class LOAD_3R<bits<17> op, string opstr>
    : Fmt3R<op, (outs GPR:$rd), (ins GPR:$rj, GPR:$rk),
            !strconcat(opstr, "\t$rd, $rj, $rk")>;
class LOAD_2RI12<bits<10> op, string opstr>
    : Fmt2RI12<op, (outs GPR:$rd), (ins GPR:$rj, simm12:$imm12),
               !strconcat(opstr, "\t$rd, $rj, $imm12")>;
class LOAD_2RI14<bits<8> op, string opstr>
    : Fmt2RI14<op, (outs GPR:$rd), (ins GPR:$rj, simm14_lsl2:$imm14),
               !strconcat(opstr, "\t$rd, $rj, $imm14")>;
} // mayLoad = 1

let mayStore = 1 in {
class STORE_3R<bits<17> op, string opstr>
    : Fmt3R<op, (outs), (ins GPR:$rd, GPR:$rj, GPR:$rk),
            !strconcat(opstr, "\t$rd, $rj, $rk")>;
class STORE_2RI12<bits<10> op, string opstr>
    : Fmt2RI12<op, (outs), (ins GPR:$rd, GPR:$rj, simm12:$imm12),
               !strconcat(opstr, "\t$rd, $rj, $imm12")>;
class STORE_2RI14<bits<8> op, string opstr>
    : Fmt2RI14<op, (outs), (ins GPR:$rd, GPR:$rj, simm14_lsl2:$imm14),
               !strconcat(opstr, "\t$rd, $rj, $imm14")>;
} // mayStore = 1

let mayLoad = 1, mayStore = 1 in
class AM_3R<bits<17> op, string opstr>
    : Fmt3R<op, (outs GPR:$rd), (ins GPR:$rk, GPR:$rj),
            !strconcat(opstr, "\t$rd, $rk, $rj")>;

let mayLoad = 1 in
class LLBase<bits<8> op, string opstr>
    : Fmt2RI14<op, (outs GPR:$rd), (ins GPR:$rj, simm14_lsl2:$imm14),
               !strconcat(opstr, "\t$rd, $rj, $imm14")>;

let mayStore = 1, Constraints = "$rd = $dst" in
class SCBase<bits<8> op, string opstr>
    : Fmt2RI14<op, (outs GPR:$dst), (ins GPR:$rd, GPR:$rj, simm14_lsl2:$imm14),
               !strconcat(opstr, "\t$rd, $rj, $imm14")>;

//===----------------------------------------------------------------------===//
// Instructions
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
def PCALAU12I : ALU_1RI20<0b0001101, "pcalau12i", simm20>;
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
def DIV_W   : ALU_3R<0b00000000001000000, "div.w">;
def MOD_W   : ALU_3R<0b00000000001000001, "mod.w">;
def DIV_WU  : ALU_3R<0b00000000001000010, "div.wu">;
def MOD_WU  : ALU_3R<0b00000000001000011, "mod.wu">;

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
                           "bstrins.w\t$rd, $rj, $msbw, $lsbw">;
}
def BSTRPICK_W : FmtBSTR_W<0b000000000111, (outs GPR:$rd),
                           (ins GPR:$rj, uimm5:$msbw, uimm5:$lsbw),
                           "bstrpick.w\t$rd, $rj, $msbw, $lsbw">;
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

let isCall = 1 in
def BL : FmtI26<0b010101, (outs), (ins simm26_lsl2:$imm26), "bl\t$imm26">;
def JIRL : Fmt2RI16<0b010011, (outs GPR:$rd),
                    (ins GPR:$rj, simm16_lsl2:$imm16),
                    "jirl\t$rd, $rj, $imm16">;

// Common Memory Access Instructions
def LD_B  : LOAD_2RI12<0b0010100000, "ld.b">;
def LD_H  : LOAD_2RI12<0b0010100001, "ld.h">;
def LD_W  : LOAD_2RI12<0b0010100010, "ld.w">;
def LD_BU : LOAD_2RI12<0b0010101000, "ld.bu">;
def LD_HU : LOAD_2RI12<0b0010101001, "ld.hu">;
def ST_B : STORE_2RI12<0b0010100100, "st.b">;
def ST_H : STORE_2RI12<0b0010100101, "st.h">;
def ST_W : STORE_2RI12<0b0010100110, "st.w">;
def PRELD : FmtPRELD<(outs), (ins uimm5:$imm5, GPR:$rj, simm12:$imm12),
                     "preld\t$imm5, $rj, $imm12">;

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
def LU32I_D : ALU_1RI20<0b0001011, "lu32i.d", simm20>;
def LU52I_D : ALU_2RI12<0b0000001100, "lu52i.d", simm12>;
def PCADDU18I : ALU_1RI20<0b0001111, "pcaddu18i", simm20>;
def MUL_D     : ALU_3R<0b00000000000111011, "mul.d">;
def MULH_D    : ALU_3R<0b00000000000111100, "mulh.d">;
def MULH_DU   : ALU_3R<0b00000000000111101, "mulh.du">;
def MULW_D_W  : ALU_3R<0b00000000000111110, "mulw.d.w">;
def MULW_D_WU : ALU_3R<0b00000000000111111, "mulw.d.wu">;
def DIV_D     : ALU_3R<0b00000000001000100, "div.d">;
def MOD_D     : ALU_3R<0b00000000001000101, "mod.d">;
def DIV_DU    : ALU_3R<0b00000000001000110, "div.du">;
def MOD_DU    : ALU_3R<0b00000000001000111, "mod.du">;

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
                           "bstrins.d\t$rd, $rj, $msbd, $lsbd">;
}
def BSTRPICK_D : FmtBSTR_D<0b0000000011, (outs GPR:$rd),
                           (ins GPR:$rj, uimm6:$msbd, uimm6:$lsbd),
                           "bstrpick.d\t$rd, $rj, $msbd, $lsbd">;

// Common Memory Access Instructions for 64-bits
def LD_WU : LOAD_2RI12<0b0010101010, "ld.wu">;
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
def PRELDX : FmtPRELDX<(outs), (ins uimm5:$imm5, GPR:$rj, GPR:$rk),
                       "preldx\t$imm5, $rj, $rk">;

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
                       "asrtle.d\t$rj, $rk">;
def ASRTGT_D : FmtASRT<0b00000000000000011, (outs), (ins GPR:$rj, GPR:$rk),
                       "asrtgt.d\t$rj, $rk">;
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

class PatGprImm<SDPatternOperator OpNode, LAInst Inst, Operand ImmOpnd>
    : Pat<(OpNode GPR:$rj, ImmOpnd:$imm),
          (Inst GPR:$rj, ImmOpnd:$imm)>;
class PatGprImm_32<SDPatternOperator OpNode, LAInst Inst, Operand ImmOpnd>
    : Pat<(sext_inreg (OpNode GPR:$rj, ImmOpnd:$imm), i32),
          (Inst GPR:$rj, ImmOpnd:$imm)>;

def : PatGprGpr<and, AND>;
def : PatGprGpr<or, OR>;
def : PatGprGpr<xor, XOR>;

/// LA32 patterns
let Predicates = [IsLA32] in {
def : PatGprGpr<add, ADD_W>;
def : PatGprImm<add, ADDI_W, simm12>;
} // Predicates = [IsLA32]

/// LA64 patterns
let Predicates = [IsLA64] in {
def : Pat<(sext_inreg GPR:$rj, i32), (ADDI_W GPR:$rj, 0)>;

def : PatGprGpr<add, ADD_D>;
def : PatGprGpr_32<add, ADD_W>;

def : PatGprImm<add, ADDI_D, simm12>;
def : PatGprImm_32<add, ADDI_W, simm12>;
} // Predicates = [IsLA64]

//===----------------------------------------------------------------------===//
// Assembler Pseudo Instructions
//===----------------------------------------------------------------------===//
def : InstAlias<"nop", (ANDI R0, R0, 0)>;
def : InstAlias<"move $dst, $src", (OR GPR:$dst, GPR:$src, R0)>;

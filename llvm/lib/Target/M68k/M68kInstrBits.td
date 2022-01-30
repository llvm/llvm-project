//===-- M68kInstrBits.td - Bit Manipulation Instrs ---------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This file describes the bit manipulation instructions in the M68k
/// architecture. Here is the current status of the file:
///
///  Machine:
///
///    BCHG    [ ]   BCLR    [ ]   BSET     [ ]   BTST     [~]
///
///  Map:
///
///   [ ] - was not touched at all
///   [!] - requires extarnal stuff implemented
///   [~] - in progress but usable
///   [x] - done
///
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// BTST
//===----------------------------------------------------------------------===//

/// ------------+---------+---------+---------+---------
///  F  E  D  C | B  A  9 | 8  7  6 | 5  4  3 | 2  1  0
/// ------------+---------+---------+---------+---------
///  0  0  0  0 |   REG   | 1  0  0 |   MODE  |   REG
/// ------------+---------+---------+---------+---------
class MxBTSTEnc_R<MxBeadDReg REG, MxEncEA EA, MxEncExt EXT>
    : MxEncoding<EA.Reg, EA.DA, EA.Mode, MxBead3Bits<0b100>, REG, MxBead4Bits<0b0000>,
                 EXT.Imm, EXT.B8, EXT.Scale, EXT.WL, EXT.DAReg>;

/// -------------------------------+---------+---------
///  F  E  D  C  B  A  9  8 . 7  6 | 5  4  3 | 2  1  0
/// -------------------------------+---------+---------
///  0  0  0  0  1  0  0  0 . 0  0 |   MODE  |   REG
/// ------------------------+------+---------+---------
///  0  0  0  0  0  0  0  0 |        BIT NUMBER
/// ------------------------+--------------------------
class MxBTSTEnc_I<MxBead8Imm IMM, MxEncEA EA, MxEncExt EXT>
    : MxEncoding<EA.Reg, EA.DA, EA.Mode, MxBead2Bits<0b00>,
                 MxBead4Bits<0b1000>, MxBead4Bits<0b0000>, IMM,
                 EXT.Imm, EXT.B8, EXT.Scale, EXT.WL, EXT.DAReg>;

let Defs = [CCR] in {
class MxBTST_RR<MxType TYPE>
    : MxInst<(outs), (ins TYPE.ROp:$dst, TYPE.ROp:$bitno), "btst\t$bitno, $dst",
             [(set CCR, (MxBtst TYPE.VT:$dst, TYPE.VT:$bitno))],
             MxBTSTEnc_R<MxBeadDReg<1>, MxEncEAd_0, MxExtEmpty>>;

class MxBTST_RI<MxType TYPE>
    : MxInst<(outs), (ins TYPE.ROp:$dst, TYPE.IOp:$bitno), "btst\t$bitno, $dst",
             [(set CCR, (MxBtst TYPE.VT:$dst, TYPE.IPat:$bitno))],
             MxBTSTEnc_I<MxBead8Imm<1>, MxEncEAd_0, MxExtEmpty>>;

class MxBTST_MR<MxType TYPE, MxOperand MEMOpd, ComplexPattern MEMPat,
                MxEncEA EA, MxEncExt EXT>
    : MxInst<(outs), (ins MEMOpd:$dst, TYPE.ROp:$bitno), "btst\t$bitno, $dst",
             [(set CCR, (MxBtst (TYPE.Load MEMPat:$dst), TYPE.VT:$bitno))],
             MxBTSTEnc_R<MxBeadDReg<1>, EA, EXT>>;

class MxBTST_MI<MxType TYPE, MxOperand MEMOpd, ComplexPattern MEMPat,
                MxEncEA EA, MxEncExt EXT>
    : MxInst<(outs), (ins MEMOpd:$dst, TYPE.IOp:$bitno), "btst\t$bitno, $dst",
             [(set CCR, (MxBtst (TYPE.Load MEMPat:$dst), TYPE.IPat:$bitno))],
             MxBTSTEnc_I<MxBead8Imm<1>, EA, EXT>>;
} // Defs = [CCR]

// Register BTST limited to 32 bits only
def BTST32dd : MxBTST_RR<MxType32d>;
def BTST32di : MxBTST_RI<MxType32d>;

// Memory BTST limited to 8 bits only
def BTST8jd : MxBTST_MR<MxType8d, MxType8.JOp, MxType8.JPat,
                        MxEncEAj_0, MxExtEmpty>;
def BTST8od : MxBTST_MR<MxType8d, MxType8.OOp, MxType8.OPat,
                        MxEncEAo_0, MxExtEmpty>;
def BTST8ed : MxBTST_MR<MxType8d, MxType8.EOp, MxType8.EPat,
                        MxEncEAe_0, MxExtEmpty>;
def BTST8pd : MxBTST_MR<MxType8d, MxType8.POp, MxType8.PPat,
                        MxEncEAp_0, MxExtI16_0>;
def BTST8fd : MxBTST_MR<MxType8d, MxType8.FOp, MxType8.FPat,
                        MxEncEAf_0, MxExtBrief_0>;
def BTST8qd : MxBTST_MR<MxType8d, MxType8.QOp, MxType8.QPat,
                        MxEncEAq,   MxExtI16_0>;
def BTST8kd : MxBTST_MR<MxType8d, MxType8.KOp, MxType8.KPat,
                        MxEncEAk,   MxExtBrief_0>;

def BTST8ji : MxBTST_MI<MxType8d, MxType8.JOp, MxType8.JPat,
                        MxEncEAj_0, MxExtEmpty>;
def BTST8oi : MxBTST_MI<MxType8d, MxType8.OOp, MxType8.OPat,
                        MxEncEAo_0, MxExtEmpty>;
def BTST8ei : MxBTST_MI<MxType8d, MxType8.EOp, MxType8.EPat,
                        MxEncEAe_0, MxExtEmpty>;
def BTST8pi : MxBTST_MI<MxType8d, MxType8.POp, MxType8.PPat,
                        MxEncEAp_0, MxExtI16_0>;
def BTST8fi : MxBTST_MI<MxType8d, MxType8.FOp, MxType8.FPat,
                        MxEncEAf_0, MxExtBrief_0>;
def BTST8qi : MxBTST_MI<MxType8d, MxType8.QOp, MxType8.QPat,
                        MxEncEAq,   MxExtI16_0>;
def BTST8ki : MxBTST_MI<MxType8d, MxType8.KOp, MxType8.KPat,
                        MxEncEAk,   MxExtBrief_0>;

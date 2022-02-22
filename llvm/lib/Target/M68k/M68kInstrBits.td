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
class MxBTSTEnc_R<MxEncMemOp dst_enc, string bitno_name> {
  dag Value = (ascend
    (descend 0b0000,
      (operand "$"#bitno_name, 3),
      0b100, dst_enc.EA
    ),
    dst_enc.Supplement
  );
}

/// -------------------------------+---------+---------
///  F  E  D  C  B  A  9  8 . 7  6 | 5  4  3 | 2  1  0
/// -------------------------------+---------+---------
///  0  0  0  0  1  0  0  0 . 0  0 |   MODE  |   REG
/// ------------------------+------+---------+---------
///  0  0  0  0  0  0  0  0 |        BIT NUMBER
/// ------------------------+--------------------------
class MxBTSTEnc_I<MxEncMemOp dst_enc, string bitno_name> {
  dag Value = (ascend
    (descend 0b0000100000, dst_enc.EA),
    (descend 0b00000000, (operand "$"#bitno_name, 8)),
    dst_enc.Supplement
  );
}

let Defs = [CCR] in {
class MxBTST_RR<MxType TYPE>
    : MxInst<(outs), (ins TYPE.ROp:$dst, TYPE.ROp:$bitno), "btst\t$bitno, $dst",
             [(set CCR, (MxBtst TYPE.VT:$dst, TYPE.VT:$bitno))]> {
  let Inst = MxBTSTEnc_R<MxEncAddrMode_r<"dst">, "bitno">.Value;
}

class MxBTST_RI<MxType TYPE>
    : MxInst<(outs), (ins TYPE.ROp:$dst, TYPE.IOp:$bitno), "btst\t$bitno, $dst",
             [(set CCR, (MxBtst TYPE.VT:$dst, TYPE.IPat:$bitno))]> {
  let Inst = MxBTSTEnc_I<MxEncAddrMode_r<"dst">, "bitno">.Value;
}

class MxBTST_MR<MxType TYPE, MxOperand MEMOpd, ComplexPattern MEMPat,
                MxEncMemOp DST_ENC>
    : MxInst<(outs), (ins MEMOpd:$dst, TYPE.ROp:$bitno), "btst\t$bitno, $dst",
             [(set CCR, (MxBtst (TYPE.Load MEMPat:$dst), TYPE.VT:$bitno))]> {
  let Inst = MxBTSTEnc_R<DST_ENC, "bitno">.Value;
}

class MxBTST_MI<MxType TYPE, MxOperand MEMOpd, ComplexPattern MEMPat,
                MxEncMemOp DST_ENC>
    : MxInst<(outs), (ins MEMOpd:$dst, TYPE.IOp:$bitno), "btst\t$bitno, $dst",
             [(set CCR, (MxBtst (TYPE.Load MEMPat:$dst), TYPE.IPat:$bitno))]> {
  let Inst = MxBTSTEnc_I<DST_ENC, "bitno">.Value;
}
} // Defs = [CCR]

// Register BTST limited to 32 bits only
def BTST32dd : MxBTST_RR<MxType32d>;
def BTST32di : MxBTST_RI<MxType32d>;

// Memory BTST limited to 8 bits only
def BTST8jd : MxBTST_MR<MxType8d, MxType8.JOp, MxType8.JPat,
                        MxEncAddrMode_j<"dst">>;
def BTST8od : MxBTST_MR<MxType8d, MxType8.OOp, MxType8.OPat,
                        MxEncAddrMode_o<"dst">>;
def BTST8ed : MxBTST_MR<MxType8d, MxType8.EOp, MxType8.EPat,
                        MxEncAddrMode_e<"dst">>;
def BTST8pd : MxBTST_MR<MxType8d, MxType8.POp, MxType8.PPat,
                        MxEncAddrMode_p<"dst">>;
def BTST8fd : MxBTST_MR<MxType8d, MxType8.FOp, MxType8.FPat,
                        MxEncAddrMode_f<"dst">>;
def BTST8qd : MxBTST_MR<MxType8d, MxType8.QOp, MxType8.QPat,
                        MxEncAddrMode_q<"dst">>;
def BTST8kd : MxBTST_MR<MxType8d, MxType8.KOp, MxType8.KPat,
                        MxEncAddrMode_k<"dst">>;

def BTST8ji : MxBTST_MI<MxType8d, MxType8.JOp, MxType8.JPat,
                        MxEncAddrMode_j<"dst">>;
def BTST8oi : MxBTST_MI<MxType8d, MxType8.OOp, MxType8.OPat,
                        MxEncAddrMode_o<"dst">>;
def BTST8ei : MxBTST_MI<MxType8d, MxType8.EOp, MxType8.EPat,
                        MxEncAddrMode_e<"dst">>;
def BTST8pi : MxBTST_MI<MxType8d, MxType8.POp, MxType8.PPat,
                        MxEncAddrMode_p<"dst">>;
def BTST8fi : MxBTST_MI<MxType8d, MxType8.FOp, MxType8.FPat,
                        MxEncAddrMode_f<"dst">>;
def BTST8qi : MxBTST_MI<MxType8d, MxType8.QOp, MxType8.QPat,
                        MxEncAddrMode_q<"dst">>;
def BTST8ki : MxBTST_MI<MxType8d, MxType8.KOp, MxType8.KPat,
                        MxEncAddrMode_k<"dst">>;

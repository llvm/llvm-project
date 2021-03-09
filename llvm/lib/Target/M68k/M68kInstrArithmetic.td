//===-- M68kInstrArithmetic.td - Integer Arith Instrs ------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This file describes the integer arithmetic instructions in the M68k
/// architecture. Here is the current status of the file:
///
///  Machine:
///
///    ADD       [~]   ADDA      [~]   ADDI        [~]   ADDQ [ ]   ADDX [~]
///    CLR       [ ]   CMP       [~]   CMPA        [~]   CMPI [~]   CMPM [ ]
///    CMP2      [ ]   DIVS/DIVU [~]   DIVSL/DIVUL [ ]   EXT  [~]   EXTB [ ]
///    MULS/MULU [~]   NEG       [~]   NEGX        [~]   SUB  [~]   SUBA [~]
///    SUBI      [~]   SUBQ      [ ]   SUBX        [~]
///
///  Map:
///
///   [ ] - was not touched at all
///   [!] - requires extarnal stuff implemented
///   [~] - functional implementation
///   [X] - complete implementation
///
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// Encoding
//===----------------------------------------------------------------------===//

/// Encoding for Normal forms
/// ----------------------------------------------------
///  F  E  D  C | B  A  9 | 8  7  6 | 5  4  3 | 2  1  0
/// ----------------------------------------------------
///             |         |         | EFFECTIVE ADDRESS
///  x  x  x  x |   REG   | OP MODE |   MODE  |   REG
/// ----------------------------------------------------
class MxArithEncoding<MxBead4Bits CMD, MxEncOpMode OPMODE, MxBeadReg REG,
                      MxEncEA EA, MxEncExt EXT>
    : MxEncoding<EA.Reg, EA.DA, EA.Mode, OPMODE.B0, OPMODE.B1, OPMODE.B2, REG,
                 CMD,EXT.Imm, EXT.B8, EXT.Scale, EXT.WL, EXT.DAReg>;

/// Encoding for Extended forms
/// ------------------------------------------------------
///  F  E  D  C | B  A  9 | 8 | 7  6 | 5  4 | 3 | 2  1  0
/// ------------------------------------------------------
///  x  x  x  x |  REG Rx | 1 | SIZE | 0  0 | M |  REG Ry
/// ------------------------------------------------------
/// Rx - destination
/// Ry - source
/// M  - address mode switch
class MxArithXEncoding<MxBead4Bits CMD, MxEncSize SIZE, MxBead1Bit MODE,
                       MxBeadReg SRC, MxBeadReg DST>
    : MxEncoding<SRC, MODE, MxBead2Bits<0b00>, SIZE, MxBead1Bit<0b1>, DST, CMD>;

/// Encoding for Immediate forms
/// ---------------------------------------------------
///  F  E  D  C  B  A  9  8 | 7  6 | 5  4  3 | 2  1  0
/// ---------------------------------------------------
///                         |      | EFFECTIVE ADDRESS
///  x  x  x  x  x  x  x  x | SIZE |   MODE  |   REG
/// ---------------------------------------------------
///     16-BIT WORD DATA    |     8-BIT BYTE DATA
/// ---------------------------------------------------
///                 32-BIT LONG DATA
/// ---------------------------------------------------
/// NOTE It is used to store an immediate to memory, imm-to-reg are handled with
/// normal version
class MxArithImmEncoding<MxBead4Bits CMD, MxEncSize SIZE,
                         MxEncEA DST_EA, MxEncExt DST_EXT, MxEncExt SRC_EXT>
    : MxEncoding<DST_EA.Reg, DST_EA.DA, DST_EA.Mode, SIZE, CMD, MxBead4Bits<0>,
                 // Source
                 SRC_EXT.Imm, SRC_EXT.B8, SRC_EXT.Scale,
                 SRC_EXT.WL, SRC_EXT.DAReg,
                 // Destination
                 DST_EXT.Imm, DST_EXT.B8, DST_EXT.Scale,
                 DST_EXT.WL, DST_EXT.DAReg>;


//===----------------------------------------------------------------------===//
// Add/Sub
//===----------------------------------------------------------------------===//

let Defs = [CCR] in {
let Constraints = "$src = $dst" in {

// $reg, $ccr <- $reg op $reg
class MxBiArOp_RFRR_xEA<string MN, SDNode NODE, MxType TYPE, bits<4> CMD>
    : MxInst<(outs TYPE.ROp:$dst), (ins TYPE.ROp:$src, TYPE.ROp:$opd),
           MN#"."#TYPE.Prefix#"\t$opd, $dst",
           [(set TYPE.VT:$dst, CCR, (NODE TYPE.VT:$src, TYPE.VT:$opd))],
           MxArithEncoding<MxBead4Bits<CMD>,
                           !cast<MxEncOpMode>("MxOpMode"#TYPE.Size#TYPE.RLet#"EA"),
                           MxBeadReg<0>,
                           !cast<MxEncEA>("MxEncEA"#TYPE.RLet#"_2"),
                           MxExtEmpty>>;

/// This Op is similar to the one above except it uses reversed opmode, some
/// commands(e.g. eor) do not support dEA or rEA modes and require EAd for
/// register only operations.
/// NOTE when using dd commands it is irrelevant which opmode to use(as it seems)
/// but some opcodes support address register and some do not which creates this
/// mess.
class MxBiArOp_RFRR_EAd<string MN, SDNode NODE, MxType TYPE, bits<4> CMD>
    : MxInst<(outs TYPE.ROp:$dst), (ins TYPE.ROp:$src, TYPE.ROp:$opd),
           MN#"."#TYPE.Prefix#"\t$opd, $dst",
           [(set TYPE.VT:$dst, CCR, (NODE TYPE.VT:$src, TYPE.VT:$opd))],
           MxArithEncoding<MxBead4Bits<CMD>,
                           !cast<MxEncOpMode>("MxOpMode"#TYPE.Size#"EAd"),
                           MxBeadReg<2>, MxEncEAd_0, MxExtEmpty>>;

// $reg <- $reg op $imm
class MxBiArOp_RFRI_xEA<string MN, SDNode NODE, MxType TYPE, bits<4> CMD>
    : MxInst<(outs TYPE.ROp:$dst), (ins TYPE.ROp:$src, TYPE.IOp:$opd),
              MN#"."#TYPE.Prefix#"\t$opd, $dst",
              [(set TYPE.VT:$dst, CCR, (NODE TYPE.VT:$src, TYPE.IPat:$opd))],
              MxArithEncoding<MxBead4Bits<CMD>,
                              !cast<MxEncOpMode>("MxOpMode"#TYPE.Size#TYPE.RLet#"EA"),
                              MxBeadReg<0>, MxEncEAi,
                              !cast<MxEncExt>("MxExtI"#TYPE.Size#"_2")>>;

// Again, there are two ways to write an immediate to Dn register either dEA
// opmode or using *I encoding, and again some instrucitons also support address
// registers some do not.
class MxBiArOp_RFRI<string MN, SDNode NODE, MxType TYPE, bits<4> CMD>
    : MxInst<(outs TYPE.ROp:$dst), (ins TYPE.ROp:$src, TYPE.IOp:$opd),
              MN#"i."#TYPE.Prefix#"\t$opd, $dst",
              [(set TYPE.VT:$dst, CCR, (NODE TYPE.VT:$src, TYPE.IPat:$opd))],
              MxArithImmEncoding<MxBead4Bits<CMD>, !cast<MxEncSize>("MxEncSize"#TYPE.Size),
                   !cast<MxEncEA>("MxEncEA"#TYPE.RLet#"_0"), MxExtEmpty,
                   !cast<MxEncExt>("MxExtI"#TYPE.Size#"_2")>>;

let mayLoad = 1 in
class MxBiArOp_RFRM<string MN, SDNode NODE, MxType TYPE, MxOperand OPD, ComplexPattern PAT,
                    bits<4> CMD, MxEncEA EA, MxEncExt EXT>
    : MxInst<(outs TYPE.ROp:$dst), (ins TYPE.ROp:$src, OPD:$opd),
           MN#"."#TYPE.Prefix#"\t$opd, $dst",
           [(set TYPE.VT:$dst, CCR, (NODE TYPE.VT:$src, (TYPE.Load PAT:$opd)))],
           MxArithEncoding<MxBead4Bits<CMD>,
                           !cast<MxEncOpMode>("MxOpMode"#TYPE.Size#TYPE.RLet#"EA"),
                           MxBeadReg<0>, EA, EXT>>;

} // Constraints

let mayLoad = 1, mayStore = 1 in {

// FIXME MxBiArOp_FMR/FMI cannot consume CCR from MxAdd/MxSub which leads for
// MxAdd to survive the match and subsequent mismatch.
class MxBiArOp_FMR<string MN, SDNode NODE, MxType TYPE,
                   MxOperand MEMOpd, ComplexPattern MEMPat,
                   bits<4> CMD, MxEncEA EA, MxEncExt EXT>
    : MxInst<(outs), (ins MEMOpd:$dst, TYPE.ROp:$opd),
        MN#"."#TYPE.Prefix#"\t$opd, $dst",
        [],
        MxArithEncoding<MxBead4Bits<CMD>,
                        !cast<MxEncOpMode>("MxOpMode"#TYPE.Size#"EA"#TYPE.RLet),
                        MxBeadReg<1>, EA, EXT>>;

class MxBiArOp_FMI<string MN, SDNode NODE, MxType TYPE,
                   MxOperand MEMOpd, ComplexPattern MEMPat,
                   bits<4> CMD, MxEncEA MEMEA, MxEncExt MEMExt>
    : MxInst<(outs), (ins MEMOpd:$dst, TYPE.IOp:$opd),
        MN#"."#TYPE.Prefix#"\t$opd, $dst",
        [],
        MxArithImmEncoding<MxBead4Bits<CMD>,
                            !cast<MxEncSize>("MxEncSize"#TYPE.Size),
                            MEMEA, MEMExt,
                            !cast<MxEncExt>("MxExtI"#TYPE.Size#"_1")>>;
} // mayLoad, mayStore
} // Defs = [CCR]

multiclass MxBiArOp_DF<string MN, SDNode NODE, bit isComm,
                       bits<4> CMD, bits<4> CMDI> {

  // op $mem, $reg
  def NAME#"8dk"  : MxBiArOp_RFRM<MN, NODE, MxType8d,  MxType8.KOp,  MxType8.KPat,
                                  CMD, MxEncEAk, MxExtBrief_2>;
  def NAME#"16dk" : MxBiArOp_RFRM<MN, NODE, MxType16d, MxType16.KOp, MxType16.KPat,
                                  CMD, MxEncEAk, MxExtBrief_2>;
  def NAME#"32dk" : MxBiArOp_RFRM<MN, NODE, MxType32d, MxType32.KOp, MxType32.KPat,
                                  CMD, MxEncEAk, MxExtBrief_2>;

  def NAME#"8dq"  : MxBiArOp_RFRM<MN, NODE, MxType8d,  MxType8.QOp,  MxType8.QPat,
                                  CMD, MxEncEAq, MxExtI16_2>;
  def NAME#"16dq" : MxBiArOp_RFRM<MN, NODE, MxType16d, MxType16.QOp, MxType16.QPat,
                                  CMD, MxEncEAq, MxExtI16_2>;
  def NAME#"32dq" : MxBiArOp_RFRM<MN, NODE, MxType32d, MxType32.QOp, MxType32.QPat,
                                  CMD, MxEncEAq, MxExtI16_2>;

  def NAME#"8dp"  : MxBiArOp_RFRM<MN, NODE, MxType8d,  MxType8.POp,  MxType8.PPat,
                                  CMD, MxEncEAp_2, MxExtI16_2>;
  def NAME#"16dp" : MxBiArOp_RFRM<MN, NODE, MxType16d, MxType16.POp, MxType16.PPat,
                                  CMD, MxEncEAp_2, MxExtI16_2>;
  def NAME#"32dp" : MxBiArOp_RFRM<MN, NODE, MxType32d, MxType32.POp, MxType32.PPat,
                                  CMD, MxEncEAp_2, MxExtI16_2>;

  def NAME#"8df"  : MxBiArOp_RFRM<MN, NODE, MxType8d,  MxType8.FOp,  MxType8.FPat,
                                  CMD, MxEncEAf_2, MxExtBrief_2>;
  def NAME#"16df" : MxBiArOp_RFRM<MN, NODE, MxType16d, MxType16.FOp, MxType16.FPat,
                                  CMD, MxEncEAf_2, MxExtBrief_2>;
  def NAME#"32df" : MxBiArOp_RFRM<MN, NODE, MxType32d, MxType32.FOp, MxType32.FPat,
                                  CMD, MxEncEAf_2, MxExtBrief_2>;

  def NAME#"8dj"  : MxBiArOp_RFRM<MN, NODE, MxType8d,  MxType8.JOp,  MxType8.JPat,
                                  CMD, MxEncEAj_2, MxExtEmpty>;
  def NAME#"16dj" : MxBiArOp_RFRM<MN, NODE, MxType16d, MxType16.JOp, MxType16.JPat,
                                  CMD, MxEncEAj_2, MxExtEmpty>;
  def NAME#"32dj" : MxBiArOp_RFRM<MN, NODE, MxType32d, MxType32.JOp, MxType32.JPat,
                                  CMD, MxEncEAj_2, MxExtEmpty>;

  // op $imm, $reg
  def NAME#"8di"  : MxBiArOp_RFRI_xEA<MN, NODE, MxType8d,  CMD>;
  def NAME#"16di" : MxBiArOp_RFRI_xEA<MN, NODE, MxType16d, CMD>;
  def NAME#"32di" : MxBiArOp_RFRI_xEA<MN, NODE, MxType32d, CMD>;

  // op $reg, $mem
  def NAME#"8pd"  : MxBiArOp_FMR<MN, NODE, MxType8d,  MxType8.POp,  MxType8.PPat,
                                 CMD, MxEncEAp_0, MxExtI16_0>;
  def NAME#"16pd" : MxBiArOp_FMR<MN, NODE, MxType16d, MxType16.POp, MxType16.PPat,
                                 CMD, MxEncEAp_0, MxExtI16_0>;
  def NAME#"32pd" : MxBiArOp_FMR<MN, NODE, MxType32d, MxType32.POp, MxType32.PPat,
                                 CMD, MxEncEAp_0, MxExtI16_0>;

  def NAME#"8fd"  : MxBiArOp_FMR<MN, NODE, MxType8d,  MxType8.FOp,  MxType8.FPat,
                                 CMD, MxEncEAf_0, MxExtBrief_0>;
  def NAME#"16fd" : MxBiArOp_FMR<MN, NODE, MxType16d, MxType16.FOp, MxType16.FPat,
                                 CMD, MxEncEAf_0, MxExtBrief_0>;
  def NAME#"32fd" : MxBiArOp_FMR<MN, NODE, MxType32d, MxType32.FOp, MxType32.FPat,
                                 CMD, MxEncEAf_0, MxExtBrief_0>;

  def NAME#"8jd"  : MxBiArOp_FMR<MN, NODE, MxType8d,  MxType8.JOp,  MxType8.JPat,
                                 CMD, MxEncEAj_0, MxExtEmpty>;
  def NAME#"16jd" : MxBiArOp_FMR<MN, NODE, MxType16d, MxType16.JOp, MxType16.JPat,
                                 CMD, MxEncEAj_0, MxExtEmpty>;
  def NAME#"32jd" : MxBiArOp_FMR<MN, NODE, MxType32d, MxType32.JOp, MxType32.JPat,
                                 CMD, MxEncEAj_0, MxExtEmpty>;

  // op $imm, $mem
  def NAME#"8pi"  : MxBiArOp_FMI<MN, NODE, MxType8,  MxType8.POp,  MxType8.PPat,
                                 CMDI, MxEncEAp_0, MxExtI16_0>;
  def NAME#"16pi" : MxBiArOp_FMI<MN, NODE, MxType16, MxType16.POp, MxType16.PPat,
                                 CMDI, MxEncEAp_0, MxExtI16_0>;
  def NAME#"32pi" : MxBiArOp_FMI<MN, NODE, MxType32, MxType32.POp, MxType32.PPat,
                                 CMDI, MxEncEAp_0, MxExtI16_0>;

  def NAME#"8fi"  : MxBiArOp_FMI<MN, NODE, MxType8,  MxType8.FOp,  MxType8.FPat,
                                 CMDI, MxEncEAf_0, MxExtBrief_0>;
  def NAME#"16fi" : MxBiArOp_FMI<MN, NODE, MxType16, MxType16.FOp, MxType16.FPat,
                                 CMDI, MxEncEAf_0, MxExtBrief_0>;
  def NAME#"32fi" : MxBiArOp_FMI<MN, NODE, MxType32, MxType32.FOp, MxType32.FPat,
                                 CMDI, MxEncEAf_0, MxExtBrief_0>;

  def NAME#"8ji"  : MxBiArOp_FMI<MN, NODE, MxType8,  MxType8.JOp,  MxType8.JPat,
                                 CMDI, MxEncEAj_0, MxExtEmpty>;
  def NAME#"16ji" : MxBiArOp_FMI<MN, NODE, MxType16, MxType16.JOp, MxType16.JPat,
                                 CMDI, MxEncEAj_0, MxExtEmpty>;
  def NAME#"32ji" : MxBiArOp_FMI<MN, NODE, MxType32, MxType32.JOp, MxType32.JPat,
                                 CMDI, MxEncEAj_0, MxExtEmpty>;

  let isCommutable = isComm in {

    def NAME#"8dd"  : MxBiArOp_RFRR_xEA<MN, NODE, MxType8d,  CMD>;
    def NAME#"16dd" : MxBiArOp_RFRR_xEA<MN, NODE, MxType16d, CMD>;
    def NAME#"32dd" : MxBiArOp_RFRR_xEA<MN, NODE, MxType32d, CMD>;

  } // isComm

} // MxBiArOp_DF


// These special snowflakes allowed to match address registers but since *A
// operations do not produce CCR we should not match them against Mx nodes that
// produce it.
let Pattern = [(null_frag)] in
multiclass MxBiArOp_AF<string MN, SDNode NODE, bit isComm,
                       bits<4> CMD, bits<4> CMDI> {

  def NAME#"32rk" : MxBiArOp_RFRM<MN, NODE, MxType32r, MxType32.KOp, MxType32.KPat,
                                  CMD, MxEncEAk, MxExtBrief_2>;
  def NAME#"32rq" : MxBiArOp_RFRM<MN, NODE, MxType32r, MxType32.QOp, MxType32.QPat,
                                  CMD, MxEncEAq, MxExtI16_2>;
  def NAME#"32rf" : MxBiArOp_RFRM<MN, NODE, MxType32r, MxType32.FOp, MxType32.FPat,
                                  CMD, MxEncEAf_2, MxExtBrief_2>;
  def NAME#"32rp" : MxBiArOp_RFRM<MN, NODE, MxType32r, MxType32.POp, MxType32.PPat,
                                  CMD, MxEncEAp_2, MxExtI16_2>;
  def NAME#"32rj" : MxBiArOp_RFRM<MN, NODE, MxType32r, MxType32.JOp, MxType32.JPat,
                                  CMD, MxEncEAj_2, MxExtEmpty>;
  def NAME#"32ri" : MxBiArOp_RFRI_xEA<MN, NODE, MxType32r, CMD>;

  let isCommutable = isComm in
  def NAME#"32rr" : MxBiArOp_RFRR_xEA<MN, NODE, MxType32r, CMD>;

} // MxBiArOp_AF

// NOTE These naturally produce CCR

defm ADD : MxBiArOp_DF<"add", MxAdd, 1, 0xD, 0x6>;
defm ADD : MxBiArOp_AF<"add", MxAdd, 1, 0xD, 0x6>;
defm SUB : MxBiArOp_DF<"sub", MxSub, 0, 0x9, 0x4>;
defm SUB : MxBiArOp_AF<"sub", MxSub, 0, 0x9, 0x4>;


let Uses = [CCR], Defs = [CCR] in {
let Constraints = "$src = $dst" in {

// $reg, ccr <- $reg op $reg op ccr
class MxBiArOp_RFRRF<string MN, SDNode NODE, MxType TYPE, bits<4> CMD>
    : MxInst<(outs TYPE.ROp:$dst), (ins TYPE.ROp:$src, TYPE.ROp:$opd),
             MN#"."#TYPE.Prefix#"\t$opd, $dst",
             [(set TYPE.VT:$dst, CCR, (NODE TYPE.VT:$src, TYPE.VT:$opd, CCR))],
             MxArithXEncoding<MxBead4Bits<CMD>,
                              !cast<MxEncSize>("MxEncSize"#TYPE.Size),
                              MxBead1Bit<0>, MxBeadReg<2>, MxBeadReg<0>>>;

} // Constraints
} // Uses, Defs

multiclass MxBiArOp_RFF<string MN, SDNode NODE, bit isComm, bits<4> CMD> {

let isCommutable = isComm in {

  def NAME#"8dd"  : MxBiArOp_RFRRF<MN, NODE, MxType8d,  CMD>;
  def NAME#"16dd" : MxBiArOp_RFRRF<MN, NODE, MxType16d, CMD>;
  def NAME#"32dd" : MxBiArOp_RFRRF<MN, NODE, MxType32d, CMD>;

} // isComm

} // MxBiArOp_RFF

// NOTE These consume and produce CCR
defm ADDX : MxBiArOp_RFF<"addx", MxAddX, 1, 0xD>;
defm SUBX : MxBiArOp_RFF<"subx", MxSubX, 0, 0x9>;


//===----------------------------------------------------------------------===//
// And/Xor/Or
//===----------------------------------------------------------------------===//

defm AND : MxBiArOp_DF<"and", MxAnd, 1, 0xC, 0x2>;
defm OR  : MxBiArOp_DF<"or",  MxOr,  1, 0x8, 0x0>;

multiclass MxBiArOp_DF_EAd<string MN, SDNode NODE, bits<4> CMD, bits<4> CMDI> {

  let isCommutable = 1 in {

  def NAME#"8dd"  : MxBiArOp_RFRR_EAd<MN, NODE, MxType8d,  CMD>;
  def NAME#"16dd" : MxBiArOp_RFRR_EAd<MN, NODE, MxType16d, CMD>;
  def NAME#"32dd" : MxBiArOp_RFRR_EAd<MN, NODE, MxType32d, CMD>;

  } // isCommutable = 1

  def NAME#"8di"  : MxBiArOp_RFRI<MN, NODE,  MxType8d, CMDI>;
  def NAME#"16di" : MxBiArOp_RFRI<MN, NODE, MxType16d, CMDI>;
  def NAME#"32di" : MxBiArOp_RFRI<MN, NODE, MxType32d, CMDI>;

} // MxBiArOp_DF_EAd

defm XOR : MxBiArOp_DF_EAd<"eor", MxXor, 0xB, 0xA>;


//===----------------------------------------------------------------------===//
// CMP
//===----------------------------------------------------------------------===//

let Defs = [CCR] in {
class MxCmp_RR<MxType TYPE>
    : MxInst<(outs), (ins TYPE.ROp:$lhs, TYPE.ROp:$rhs),
             "cmp."#TYPE.Prefix#"\t$lhs, $rhs",
             [(set CCR, (MxCmp TYPE.VT:$lhs, TYPE.VT:$rhs))],
             MxArithEncoding<MxBead4Bits<0xB>,
                             !cast<MxEncOpMode>("MxOpMode"#TYPE.Size#"dEA"),
                             MxBeadReg<1>, MxEncEAd_0, MxExtEmpty>>;

class MxCmp_RI<MxType TYPE>
    : MxInst<(outs), (ins TYPE.IOp:$imm, TYPE.ROp:$reg),
             "cmpi."#TYPE.Prefix#"\t$imm, $reg",
             [(set CCR, (MxCmp TYPE.IPat:$imm, TYPE.VT:$reg))],
             MxArithImmEncoding<MxBead4Bits<0xC>,
                                !cast<MxEncSize>("MxEncSize"#TYPE.Size),
                                MxEncEAd_1, MxExtEmpty,
                                !cast<MxEncExt>("MxExtI"#TYPE.Size#"_0")>>;

let mayLoad = 1 in {

class MxCmp_MI<MxType TYPE, MxOperand MEMOpd, ComplexPattern MEMPat,
               MxEncEA EA, MxEncExt EXT>
    : MxInst<(outs), (ins TYPE.IOp:$imm, MEMOpd:$mem),
             "cmpi."#TYPE.Prefix#"\t$imm, $mem",
             [(set CCR, (MxCmp TYPE.IPat:$imm, (load MEMPat:$mem)))],
             MxArithImmEncoding<MxBead4Bits<0xC>,
                                !cast<MxEncSize>("MxEncSize"#TYPE.Size),
                                EA, EXT,
                                !cast<MxEncExt>("MxExtI"#TYPE.Size#"_0")>>;

class MxCmp_BI<MxType TYPE>
    : MxInst<(outs), (ins TYPE.IOp:$imm, MxAL32:$abs),
             "cmpi."#TYPE.Prefix#"\t$imm, $abs",
             [(set CCR, (MxCmp TYPE.IPat:$imm,
                               (load (i32 (MxWrapper tglobaladdr:$abs)))))],
             MxArithImmEncoding<MxBead4Bits<0xC>,
                                !cast<MxEncSize>("MxEncSize"#TYPE.Size),
                                MxEncEAb, MxExtI32_1,
                                !cast<MxEncExt>("MxExtI"#TYPE.Size#"_0")>>;

class MxCmp_RM<MxType TYPE, MxOperand MEMOpd, ComplexPattern MEMPat,
               MxEncEA EA, MxEncExt EXT>
    : MxInst<(outs), (ins TYPE.ROp:$reg, MEMOpd:$mem),
             "cmp."#TYPE.Prefix#"\t$mem, $reg",
             [(set CCR, (MxCmp (load MEMPat:$mem), TYPE.ROp:$reg))],
             MxArithEncoding<MxBead4Bits<0xB>,
                             !cast<MxEncOpMode>("MxOpMode"#TYPE.Size#"dEA"),
                             MxBeadReg<0>, EA, EXT>>;
} // let mayLoad = 1

} // let Defs = [CCR]

multiclass MMxCmp_RM<MxType TYPE> {
  def NAME#TYPE.KOp.Letter : MxCmp_RM<TYPE, TYPE.KOp, TYPE.KPat, MxEncEAk,
                                      MxExtBrief_1>;
  def NAME#TYPE.QOp.Letter : MxCmp_RM<TYPE, TYPE.QOp, TYPE.QPat, MxEncEAq,
                                      MxExtI16_1>;
  def NAME#TYPE.POp.Letter : MxCmp_RM<TYPE, TYPE.POp, TYPE.PPat, MxEncEAp_1,
                                      MxExtI16_1>;
  def NAME#TYPE.FOp.Letter : MxCmp_RM<TYPE, TYPE.FOp, TYPE.FPat, MxEncEAf_1,
                                      MxExtBrief_1>;
  def NAME#TYPE.JOp.Letter : MxCmp_RM<TYPE, TYPE.JOp, TYPE.JPat, MxEncEAj_1,
                                      MxExtEmpty>;
}

multiclass MMxCmp_MI<MxType TYPE> {
  def NAME#TYPE.KOp.Letter#"i" : MxCmp_MI<TYPE, TYPE.KOp, TYPE.KPat, MxEncEAk,
                                          MxExtBrief_1>;
  def NAME#TYPE.QOp.Letter#"i" : MxCmp_MI<TYPE, TYPE.QOp, TYPE.QPat, MxEncEAq,
                                          MxExtI16_1>;
  def NAME#TYPE.POp.Letter#"i" : MxCmp_MI<TYPE, TYPE.POp, TYPE.PPat, MxEncEAp_1,
                                          MxExtI16_1>;
  def NAME#TYPE.FOp.Letter#"i" : MxCmp_MI<TYPE, TYPE.FOp, TYPE.FPat, MxEncEAf_1,
                                          MxExtBrief_1>;
  def NAME#TYPE.JOp.Letter#"i" : MxCmp_MI<TYPE, TYPE.JOp, TYPE.JPat, MxEncEAj_1,
                                          MxExtEmpty>;
}

foreach S = [8, 16, 32] in {
  def CMP#S#dd : MxCmp_RR<!cast<MxType>("MxType"#S#"d")>;
  def CMP#S#di : MxCmp_RI<!cast<MxType>("MxType"#S#"d")>;
  def CMP#S#bi : MxCmp_BI<!cast<MxType>("MxType"#S#"d")>;
} // foreach

// cmp mem, Dn
defm CMP8d  : MMxCmp_RM<MxType8d>;
defm CMP16d : MMxCmp_RM<MxType16d>;
defm CMP32d : MMxCmp_RM<MxType32d>;

// cmp #imm, mem
defm CMP8  : MMxCmp_MI<MxType8d>;
defm CMP16 : MMxCmp_MI<MxType16d>;
defm CMP32 : MMxCmp_MI<MxType32d>;


//===----------------------------------------------------------------------===//
// EXT
//===----------------------------------------------------------------------===//

def MxExtOpmode_wb : MxBead3Bits<0b010>;
def MxExtOpmode_lw : MxBead3Bits<0b011>;
def MxExtOpmode_lb : MxBead3Bits<0b111>;

/// ---------------------------------------------------
///  F  E  D  C  B  A  9 | 8  7  6 | 5  4  3 | 2  1  0
/// ---------------------------------------------------
///  0  1  0  0  1  0  0 |  OPMODE | 0  0  0 |   REG
/// ---------------------------------------------------
class MxExtEncoding<MxBead3Bits OPMODE>
    : MxEncoding<MxBeadReg<0>, MxBead3Bits<0b000>, OPMODE,
                 MxBead3Bits<0b100>, MxBead4Bits<0b0100>>;

let Defs = [CCR] in
let Constraints = "$src = $dst" in
class MxExt<MxType TO, MxType FROM>
    : MxInst<(outs TO.ROp:$dst), (ins TO.ROp:$src),
              "ext."#TO.Prefix#"\t$src", [],
              MxExtEncoding<!cast<MxBead3Bits>("MxExtOpmode_"#TO.Prefix#FROM.Prefix)>>;

def EXT16 : MxExt<MxType16d, MxType8d>;
def EXT32 : MxExt<MxType32d, MxType16d>;

def : Pat<(sext_inreg i16:$src, i8),  (EXT16 $src)>;
def : Pat<(sext_inreg i32:$src, i16), (EXT32 $src)>;
def : Pat<(sext_inreg i32:$src, i8),
          (EXT32 (MOVXd32d16 (EXT16 (EXTRACT_SUBREG $src, MxSubRegIndex16Lo))))>;


//===----------------------------------------------------------------------===//
// DIV/MUL
//===----------------------------------------------------------------------===//

def MxSDiMuOpmode : MxBead3Bits<0b111>;
def MxUDiMuOpmode : MxBead3Bits<0b011>;

/// Word operation:
/// ----------------------------------------------------
///  F  E  D  C | B  A  9 | 8  7  6 | 5  4  3 | 2  1  0
/// ----------------------------------------------------
///             |         |         | EFFECTIVE ADDRESS
///  x  x  x  x |   REG   | OP MODE |   MODE  |   REG
/// ----------------------------------------------------
class MxDiMuEncoding<MxBead4Bits CMD, MxBead3Bits OPMODE, MxEncEA EA, MxEncExt EXT>
    : MxEncoding<EA.Reg, EA.DA, EA.Mode, OPMODE, MxBeadReg<0>, CMD,
                 EXT.Imm, EXT.B8, EXT.Scale, EXT.WL, EXT.DAReg>;

let Defs = [CCR] in {
let Constraints = "$src = $dst" in {
// $reg <- $reg op $reg
class MxDiMuOp_DD<string MN, bits<4> CMD, MxBead3Bits OPMODE,
                  MxOperand DST, MxOperand OPD>
    : MxInst<(outs DST:$dst), (ins DST:$src, OPD:$opd), MN#"\t$opd, $dst", [],
             MxDiMuEncoding<MxBead4Bits<CMD>, OPMODE, MxEncEAd_2, MxExtEmpty>>;

// $reg <- $reg op $imm
class MxDiMuOp_DI<string MN, bits<4> CMD, MxBead3Bits OPMODE,
                  MxOperand DST, MxOperand OPD>
    : MxInst<(outs DST:$dst), (ins DST:$src, unknown:$opd), MN#"\t$opd, $dst", [],
             MxDiMuEncoding<MxBead4Bits<CMD>, OPMODE, MxEncEAi, MxExtI16_2>>;
} // let Constraints
} // Defs = [CCR]

multiclass MxDiMuOp<string MN, bits<4> CMD, bit isComm = 0> {

  let isCommutable = isComm in {
    def "S"#NAME#"d32d16" : MxDiMuOp_DD<MN#"s", CMD, MxSDiMuOpmode, MxDRD32,
                                        MxDRD16>;
    def "U"#NAME#"d32d16" : MxDiMuOp_DD<MN#"u", CMD, MxUDiMuOpmode, MxDRD32,
                                        MxDRD16>;
  }

  def "S"#NAME#"d32i16" : MxDiMuOp_DI<MN#"s", CMD, MxSDiMuOpmode, MxDRD32,
                                      Mxi16imm>;
  def "U"#NAME#"d32i16" : MxDiMuOp_DI<MN#"u", CMD, MxUDiMuOpmode, MxDRD32,
                                      Mxi16imm>;

}

defm DIV : MxDiMuOp<"div", 0x8>;

// RR i8
def : Pat<(sdiv i8:$dst, i8:$opd),
          (EXTRACT_SUBREG
            (SDIVd32d16 (MOVSXd32d8 $dst), (MOVSXd16d8 $opd)),
             MxSubRegIndex8Lo)>;

def : Pat<(udiv i8:$dst, i8:$opd),
          (EXTRACT_SUBREG
            (UDIVd32d16 (MOVZXd32d8 $dst), (MOVZXd16d8 $opd)),
             MxSubRegIndex8Lo)>;

def : Pat<(srem i8:$dst, i8:$opd),
          (EXTRACT_SUBREG
            (ASR32di (ASR32di (SDIVd32d16 (MOVSXd32d8 $dst), (MOVSXd16d8 $opd)), 8), 8),
             MxSubRegIndex8Lo)>;

def : Pat<(urem i8:$dst, i8:$opd),
          (EXTRACT_SUBREG
            (LSR32di (LSR32di (UDIVd32d16 (MOVZXd32d8 $dst), (MOVZXd16d8 $opd)), 8), 8),
             MxSubRegIndex8Lo)>;

// RR i16
def : Pat<(sdiv i16:$dst, i16:$opd),
          (EXTRACT_SUBREG
            (SDIVd32d16 (MOVSXd32d16 $dst), $opd),
             MxSubRegIndex16Lo)>;

def : Pat<(udiv i16:$dst, i16:$opd),
          (EXTRACT_SUBREG
            (UDIVd32d16 (MOVZXd32d16 $dst), $opd),
             MxSubRegIndex16Lo)>;

def : Pat<(srem i16:$dst, i16:$opd),
          (EXTRACT_SUBREG
            (ASR32di (ASR32di (SDIVd32d16 (MOVSXd32d16 $dst), $opd), 8), 8),
             MxSubRegIndex16Lo)>;

def : Pat<(urem i16:$dst, i16:$opd),
          (EXTRACT_SUBREG
            (LSR32di (LSR32di (UDIVd32d16 (MOVZXd32d16 $dst), $opd), 8), 8),
             MxSubRegIndex16Lo)>;


// RI i8
def : Pat<(sdiv i8:$dst, MximmSExt8:$opd),
          (EXTRACT_SUBREG
            (SDIVd32i16 (MOVSXd32d8 $dst), imm:$opd),
             MxSubRegIndex8Lo)>;

def : Pat<(udiv i8:$dst, MximmSExt8:$opd),
          (EXTRACT_SUBREG
            (UDIVd32i16 (MOVZXd32d8 $dst), imm:$opd),
             MxSubRegIndex8Lo)>;

def : Pat<(srem i8:$dst, MximmSExt8:$opd),
          (EXTRACT_SUBREG
            (ASR32di (ASR32di (SDIVd32i16 (MOVSXd32d8 $dst), imm:$opd), 8), 8),
             MxSubRegIndex8Lo)>;

def : Pat<(urem i8:$dst, MximmSExt8:$opd),
          (EXTRACT_SUBREG
            (LSR32di (LSR32di (UDIVd32i16 (MOVZXd32d8 $dst), imm:$opd), 8), 8),
             MxSubRegIndex8Lo)>;

// RI i16
def : Pat<(sdiv i16:$dst, MximmSExt16:$opd),
          (EXTRACT_SUBREG
            (SDIVd32i16 (MOVSXd32d16 $dst), imm:$opd),
             MxSubRegIndex16Lo)>;

def : Pat<(udiv i16:$dst, MximmSExt16:$opd),
          (EXTRACT_SUBREG
            (UDIVd32i16 (MOVZXd32d16 $dst), imm:$opd),
             MxSubRegIndex16Lo)>;

def : Pat<(srem i16:$dst, MximmSExt16:$opd),
          (EXTRACT_SUBREG
            (ASR32di (ASR32di (SDIVd32i16 (MOVSXd32d16 $dst), imm:$opd), 8), 8),
             MxSubRegIndex16Lo)>;

def : Pat<(urem i16:$dst, MximmSExt16:$opd),
          (EXTRACT_SUBREG
            (LSR32di (LSR32di (UDIVd32i16 (MOVZXd32d16 $dst), imm:$opd), 8), 8),
             MxSubRegIndex16Lo)>;


defm MUL : MxDiMuOp<"mul", 0xC, 1>;

// RR
def : Pat<(mul i16:$dst, i16:$opd),
          (EXTRACT_SUBREG
            (SMULd32d16 (MOVXd32d16 $dst), $opd),
             MxSubRegIndex16Lo)>;

def : Pat<(mulhs i16:$dst, i16:$opd),
          (EXTRACT_SUBREG
            (ASR32di (ASR32di (SMULd32d16 (MOVXd32d16 $dst), $opd), 8), 8),
             MxSubRegIndex16Lo)>;

def : Pat<(mulhu i16:$dst, i16:$opd),
          (EXTRACT_SUBREG
            (LSR32di (LSR32di (UMULd32d16 (MOVXd32d16 $dst), $opd), 8), 8),
             MxSubRegIndex16Lo)>;


// RI
def : Pat<(mul i16:$dst, MximmSExt16:$opd),
          (EXTRACT_SUBREG
            (SMULd32i16 (MOVXd32d16 $dst), imm:$opd),
             MxSubRegIndex16Lo)>;

def : Pat<(mulhs i16:$dst, MximmSExt16:$opd),
          (EXTRACT_SUBREG
            (ASR32di (ASR32di (SMULd32i16 (MOVXd32d16 $dst), imm:$opd), 8), 8),
             MxSubRegIndex16Lo)>;

def : Pat<(mulhu i16:$dst, MximmSExt16:$opd),
          (EXTRACT_SUBREG
            (LSR32di (LSR32di (UMULd32i16 (MOVXd32d16 $dst), imm:$opd), 8), 8),
             MxSubRegIndex16Lo)>;


//===----------------------------------------------------------------------===//
// NEG/NEGX
//===----------------------------------------------------------------------===//

/// ------------+------------+------+---------+---------
///  F  E  D  C | B  A  9  8 | 7  6 | 5  4  3 | 2  1  0
/// ------------+------------+------+-------------------
///             |            |      | EFFECTIVE ADDRESS
///  0  1  0  0 | x  x  x  x | SIZE |   MODE  |   REG
/// ------------+------------+------+---------+---------
class MxNEGEncoding<MxBead4Bits CMD, MxEncSize SIZE, MxEncEA EA, MxEncExt EXT>
    : MxEncoding<EA.Reg, EA.DA, EA.Mode, SIZE, CMD, MxBead4Bits<0b0100>,
                 EXT.Imm, EXT.B8, EXT.Scale, EXT.WL, EXT.DAReg>;

let Defs = [CCR] in {
let Constraints = "$src = $dst" in {

class MxNeg_D<MxType TYPE>
    : MxInst<(outs TYPE.ROp:$dst), (ins TYPE.ROp:$src),
             "neg."#TYPE.Prefix#"\t$dst",
             [(set TYPE.VT:$dst, (ineg TYPE.VT:$src))],
             MxNEGEncoding<MxBead4Bits<0x4>,
                           !cast<MxEncSize>("MxEncSize"#TYPE.Size),
                           MxEncEAd_0, MxExtEmpty>>;

let Uses = [CCR] in {
class MxNegX_D<MxType TYPE>
    : MxInst<(outs TYPE.ROp:$dst), (ins TYPE.ROp:$src),
             "negx."#TYPE.Prefix#"\t$dst",
             [(set TYPE.VT:$dst, (MxSubX 0, TYPE.VT:$src, CCR))],
             MxNEGEncoding<MxBead4Bits<0x0>,
                           !cast<MxEncSize>("MxEncSize"#TYPE.Size),
                           MxEncEAd_0, MxExtEmpty>>;
}

} // let Constraints
} // let Defs = [CCR]

foreach S = [8, 16, 32] in {
  def NEG#S#d  : MxNeg_D<!cast<MxType>("MxType"#S#"d")>;
  def NEGX#S#d : MxNegX_D<!cast<MxType>("MxType"#S#"d")>;
}

def : Pat<(MxSub 0, i8 :$src), (NEG8d  MxDRD8 :$src)>;
def : Pat<(MxSub 0, i16:$src), (NEG16d MxDRD16:$src)>;
def : Pat<(MxSub 0, i32:$src), (NEG32d MxDRD32:$src)>;

//===----------------------------------------------------------------------===//
// no-CCR Patterns
//===----------------------------------------------------------------------===//

/// Basically the reason for this stuff is that add and addc share the same
/// operand types constraints for whatever reasons and I had to define a common
/// MxAdd and MxSub instructions that produce CCR and then pattern-map add and addc
/// to it.
/// NOTE On the other hand I see no reason why I cannot just drop explicit CCR
/// result. Anyway works for now, hopefully I will better understand how this stuff
/// is designed later
foreach N = ["add", "addc"] in {

  // add reg, reg
  def : Pat<(!cast<SDNode>(N) i8 :$src, i8 :$opd),
            (ADD8dd  MxDRD8 :$src, MxDRD8 :$opd)>;
  def : Pat<(!cast<SDNode>(N) i16:$src, i16:$opd),
            (ADD16dd MxDRD16:$src, MxDRD16:$opd)>;
  def : Pat<(!cast<SDNode>(N) i32:$src, i32:$opd),
            (ADD32rr MxXRD32:$src, MxXRD32:$opd)>;

  // add (An), reg
  def : Pat<(!cast<SDNode>(N) MxType8.VT:$src, (Mxloadi8 MxType8.JPat:$opd)),
            (ADD8dj MxDRD8:$src, MxType8.JOp:$opd)>;
  def : Pat<(!cast<SDNode>(N) MxType16.VT:$src, (Mxloadi16 MxType16.JPat:$opd)),
            (ADD16dj MxDRD16:$src, MxType16.JOp:$opd)>;
  def : Pat<(!cast<SDNode>(N) MxType32.VT:$src, (Mxloadi32 MxType32.JPat:$opd)),
            (ADD32rj MxXRD32:$src, MxType32.JOp:$opd)>;

  // add (i,An), reg
  def : Pat<(!cast<SDNode>(N) MxType8.VT:$src, (Mxloadi8 MxType8.PPat:$opd)),
            (ADD8dp MxDRD8:$src, MxType8.POp:$opd)>;
  def : Pat<(!cast<SDNode>(N) MxType16.VT:$src, (Mxloadi16 MxType16.PPat:$opd)),
            (ADD16dp MxDRD16:$src, MxType16.POp:$opd)>;
  def : Pat<(!cast<SDNode>(N) MxType32.VT:$src, (Mxloadi32 MxType32.PPat:$opd)),
            (ADD32rp MxXRD32:$src, MxType32.POp:$opd)>;

  // add (i,An,Xn), reg
  def : Pat<(!cast<SDNode>(N) MxType8.VT:$src, (Mxloadi8 MxType8.FPat:$opd)),
            (ADD8df MxDRD8:$src, MxType8.FOp:$opd)>;
  def : Pat<(!cast<SDNode>(N) MxType16.VT:$src, (Mxloadi16 MxType16.FPat:$opd)),
            (ADD16df MxDRD16:$src, MxType16.FOp:$opd)>;
  def : Pat<(!cast<SDNode>(N) MxType32.VT:$src, (Mxloadi32 MxType32.FPat:$opd)),
            (ADD32rf MxXRD32:$src, MxType32.FOp:$opd)>;

  // add reg, imm
  def : Pat<(!cast<SDNode>(N) i8: $src, MximmSExt8:$opd),
            (ADD8di  MxDRD8 :$src, imm:$opd)>;
  def : Pat<(!cast<SDNode>(N) i16:$src, MximmSExt16:$opd),
            (ADD16di MxDRD16:$src, imm:$opd)>;

  // LEAp is more complex and thus will be selected over normal ADD32ri but it cannot
  // be used with data registers, here by adding complexity to a simple ADD32ri insts
  // we make sure it will be selected over LEAp
  let AddedComplexity = 15 in {
  def : Pat<(!cast<SDNode>(N) i32:$src, MximmSExt32:$opd),
            (ADD32ri MxXRD32:$src, imm:$opd)>;
  } // AddedComplexity = 15

  // add imm, (An)
  def : Pat<(store (!cast<SDNode>(N) (load MxType8.JPat:$dst), MxType8.IPat:$opd),
                    MxType8.JPat:$dst),
            (ADD8ji MxType8.JOp:$dst, imm:$opd)>;
  def : Pat<(store (!cast<SDNode>(N) (load MxType16.JPat:$dst), MxType16.IPat:$opd),
                    MxType16.JPat:$dst),
            (ADD16ji MxType16.JOp:$dst, imm:$opd)>;
  def : Pat<(store (!cast<SDNode>(N) (load MxType32.JPat:$dst), MxType32.IPat:$opd),
                    MxType32.JPat:$dst),
            (ADD32ji MxType32.JOp:$dst, imm:$opd)>;

} // foreach add, addc

def : Pat<(adde i8 :$src, i8 :$opd), (ADDX8dd  MxDRD8 :$src, MxDRD8 :$opd)>;
def : Pat<(adde i16:$src, i16:$opd), (ADDX16dd MxDRD16:$src, MxDRD16:$opd)>;
def : Pat<(adde i32:$src, i32:$opd), (ADDX32dd MxDRD32:$src, MxDRD32:$opd)>;



foreach N = ["sub", "subc"] in {

  // sub reg, reg
  def : Pat<(!cast<SDNode>(N) i8 :$src, i8 :$opd),
            (SUB8dd  MxDRD8 :$src, MxDRD8 :$opd)>;
  def : Pat<(!cast<SDNode>(N) i16:$src, i16:$opd),
            (SUB16dd MxDRD16:$src, MxDRD16:$opd)>;
  def : Pat<(!cast<SDNode>(N) i32:$src, i32:$opd),
            (SUB32rr MxXRD32:$src, MxXRD32:$opd)>;


  // sub (An), reg
  def : Pat<(!cast<SDNode>(N) MxType8.VT:$src, (Mxloadi8 MxType8.JPat:$opd)),
            (SUB8dj MxDRD8:$src, MxType8.JOp:$opd)>;
  def : Pat<(!cast<SDNode>(N) MxType16.VT:$src, (Mxloadi16 MxType16.JPat:$opd)),
            (SUB16dj MxDRD16:$src, MxType16.JOp:$opd)>;
  def : Pat<(!cast<SDNode>(N) MxType32.VT:$src, (Mxloadi32 MxType32.JPat:$opd)),
            (SUB32rj MxXRD32:$src, MxType32.JOp:$opd)>;

  // sub (i,An), reg
  def : Pat<(!cast<SDNode>(N) MxType8.VT:$src, (Mxloadi8 MxType8.PPat:$opd)),
            (SUB8dp MxDRD8:$src, MxType8.POp:$opd)>;
  def : Pat<(!cast<SDNode>(N) MxType16.VT:$src, (Mxloadi16 MxType16.PPat:$opd)),
            (SUB16dp MxDRD16:$src, MxType16.POp:$opd)>;
  def : Pat<(!cast<SDNode>(N) MxType32.VT:$src, (Mxloadi32 MxType32.PPat:$opd)),
            (SUB32rp MxXRD32:$src, MxType32.POp:$opd)>;

  // sub (i,An,Xn), reg
  def : Pat<(!cast<SDNode>(N) MxType8.VT:$src, (Mxloadi8 MxType8.FPat:$opd)),
            (SUB8df MxDRD8:$src, MxType8.FOp:$opd)>;
  def : Pat<(!cast<SDNode>(N) MxType16.VT:$src, (Mxloadi16 MxType16.FPat:$opd)),
            (SUB16df MxDRD16:$src, MxType16.FOp:$opd)>;
  def : Pat<(!cast<SDNode>(N) MxType32.VT:$src, (Mxloadi32 MxType32.FPat:$opd)),
            (SUB32rf MxXRD32:$src, MxType32.FOp:$opd)>;

  // sub reg, imm
  def : Pat<(!cast<SDNode>(N) i8 :$src, MximmSExt8 :$opd),
            (SUB8di  MxDRD8 :$src, imm:$opd)>;
  def : Pat<(!cast<SDNode>(N) i16:$src, MximmSExt16:$opd),
            (SUB16di MxDRD16:$src, imm:$opd)>;
  def : Pat<(!cast<SDNode>(N) i32:$src, MximmSExt32:$opd),
            (SUB32ri MxXRD32:$src, imm:$opd)>;

  // sub imm, (An)
  def : Pat<(store (!cast<SDNode>(N) (load MxType8.JPat:$dst), MxType8.IPat:$opd),
                   MxType8.JPat:$dst),
            (SUB8ji MxType8.JOp:$dst, imm:$opd)>;
  def : Pat<(store (!cast<SDNode>(N) (load MxType16.JPat:$dst), MxType16.IPat:$opd),
                   MxType16.JPat:$dst),
            (SUB16ji MxType16.JOp:$dst, imm:$opd)>;
  def : Pat<(store (!cast<SDNode>(N) (load MxType32.JPat:$dst), MxType32.IPat:$opd),
                   MxType32.JPat:$dst),
            (SUB32ji MxType32.JOp:$dst, imm:$opd)>;

} // foreach sub, subx

def : Pat<(sube i8 :$src, i8 :$opd), (SUBX8dd  MxDRD8 :$src, MxDRD8 :$opd)>;
def : Pat<(sube i16:$src, i16:$opd), (SUBX16dd MxDRD16:$src, MxDRD16:$opd)>;
def : Pat<(sube i32:$src, i32:$opd), (SUBX32dd MxDRD32:$src, MxDRD32:$opd)>;


// and reg, reg
def : Pat<(and i8 :$src, i8 :$opd), (AND8dd  MxDRD8 :$src, MxDRD8 :$opd)>;
def : Pat<(and i16:$src, i16:$opd), (AND16dd MxDRD16:$src, MxDRD16:$opd)>;
def : Pat<(and i32:$src, i32:$opd), (AND32dd MxDRD32:$src, MxDRD32:$opd)>;

// and reg, imm
def : Pat<(and i8: $src, MximmSExt8 :$opd), (AND8di  MxDRD8 :$src, imm:$opd)>;
def : Pat<(and i16:$src, MximmSExt16:$opd), (AND16di MxDRD16:$src, imm:$opd)>;
def : Pat<(and i32:$src, MximmSExt32:$opd), (AND32di MxDRD32:$src, imm:$opd)>;


// xor reg,reg
def : Pat<(xor i8 :$src, i8 :$opd), (XOR8dd  MxDRD8 :$src, MxDRD8 :$opd)>;
def : Pat<(xor i16:$src, i16:$opd), (XOR16dd MxDRD16:$src, MxDRD16:$opd)>;
def : Pat<(xor i32:$src, i32:$opd), (XOR32dd MxDRD32:$src, MxDRD32:$opd)>;

// xor reg, imm
def : Pat<(xor i8: $src, MximmSExt8 :$opd), (XOR8di  MxDRD8 :$src, imm:$opd)>;
def : Pat<(xor i16:$src, MximmSExt16:$opd), (XOR16di MxDRD16:$src, imm:$opd)>;
def : Pat<(xor i32:$src, MximmSExt32:$opd), (XOR32di MxDRD32:$src, imm:$opd)>;


// or reg, reg
def : Pat<(or i8 :$src, i8 :$opd), (OR8dd  MxDRD8 :$src, MxDRD8 :$opd)>;
def : Pat<(or i16:$src, i16:$opd), (OR16dd MxDRD16:$src, MxDRD16:$opd)>;
def : Pat<(or i32:$src, i32:$opd), (OR32dd MxDRD32:$src, MxDRD32:$opd)>;

// or reg, imm
def : Pat<(or i8: $src, MximmSExt8 :$opd), (OR8di  MxDRD8 :$src, imm:$opd)>;
def : Pat<(or i16:$src, MximmSExt16:$opd), (OR16di MxDRD16:$src, imm:$opd)>;
def : Pat<(or i32:$src, MximmSExt32:$opd), (OR32di MxDRD32:$src, imm:$opd)>;

//== M68kInstrData.td - M68k Data Movement Instructions -*- tablegen --===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This file describes the Motorola 680x0 data movement instructions which are
/// the basic means of transferring and storing addresses and data. Here is the
/// current status of the file:
///
///  Machine:
///
///     EXG   [ ]     FMOVE [ ]     FSMOVE [ ]     FDMOVE [ ]     FMOVEM [ ]
///     LEA   [~]     PEA   [ ]     MOVE   [~]     MOVE16 [ ]     MOVEA  [ ]
///     MOVEM [ ]     MOVEP [ ]     MOVEQ  [ ]     LINK   [ ]     UNLK   [ ]
///
///  Pseudo:
///
///     MOVSX [x]     MOVZX [x]     MOVX   [x]
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
// MOVE
//===----------------------------------------------------------------------===//

/// -----------------------------------------------------
///  F  E | D  C | B  A  9 | 8  7  6 | 5  4  3 | 2  1  0
/// -----------------------------------------------------
///       |      |    DESTINATION    |       SOURCE
///  0  0 | SIZE |   REG   |   MODE  |   MODE  |   REG
/// -----------------------------------------------------
///
/// NOTE Move requires EA X version for direct register destination(0)
class MxMoveEncoding<MxBead2Bits size,
                     MxEncEA srcEA, MxEncExt srcExt,
                     MxEncEA dstEA, MxEncExt dstExt>
    : MxEncoding<srcEA.Reg, srcEA.DA, srcEA.Mode, dstEA.DA, dstEA.Mode, dstEA.Reg,
                 size, MxBead2Bits<0b00>,
                 srcExt.Imm, srcExt.B8, srcExt.Scale, srcExt.WL, srcExt.DAReg,
                 dstExt.Imm, dstExt.B8, dstExt.Scale, dstExt.WL, dstExt.DAReg>;

/// MOVE has alternate size encoding
class MxMoveSize<bits<2> value> : MxBead2Bits<value>;
def MxMoveSize8  : MxMoveSize<0b01>;
def MxMoveSize16 : MxMoveSize<0b11>;
def MxMoveSize32 : MxMoveSize<0b10>;

let Defs = [CCR] in
class MxMove<string size, dag outs, dag ins, list<dag> pattern, MxEncoding enc>
    : MxInst<outs, ins, "move."#size#"\t$src, $dst", pattern, enc>;

class MxMove_RR<MxType DST, MxType SRC, MxMoveEncoding ENC>
    : MxMove<DST.Prefix, (outs DST.ROp:$dst), (ins SRC.ROp:$src),
             [(null_frag)], ENC>;

let mayStore = 1 in {
class MxMove_MR<MxOperand MEMOpd, ComplexPattern MEMPat, MxType REG,
                MxMoveEncoding ENC>
    : MxMove<REG.Prefix, (outs), (ins MEMOpd:$dst, REG.ROp:$src),
             [(store REG.VT:$src, MEMPat:$dst)], ENC>;

class MxMove_MI<MxOperand MEMOpd, ComplexPattern MEMPat, MxType TYPE,
                MxMoveEncoding ENC>
    : MxMove<TYPE.Prefix, (outs), (ins MEMOpd:$dst, TYPE.IOp:$src),
             [(store TYPE.IPat:$src, MEMPat:$dst)], ENC>;
} // let mayStore = 1

class MxMove_RI<MxType DST, MxMoveEncoding ENC>
    : MxMove<DST.Prefix, (outs DST.ROp:$dst), (ins DST.IOp:$src),
              [(set DST.VT:$dst, DST.IPat:$src)], ENC>;


let mayLoad = 1 in
class MxMove_RM<MxType REG, MxOperand MEMOpd, ComplexPattern MEMPat,
                MxBead2Bits SIZE,
                MxEncEA SRCEA, MxEncExt SRCEXT,
                MxEncEA DSTEA, MxEncExt DSTEXT>
    : MxMove<REG.Prefix, (outs REG.ROp:$dst), (ins MEMOpd:$src),
             [(set REG.VT:$dst, (REG.Load MEMPat:$src))],
             MxMoveEncoding<SIZE, SRCEA, SRCEXT, DSTEA, DSTEXT>>;

multiclass MMxMove_RM<MxType REG, MxMoveSize SIZE, MxEncEA EA_0> {

  // REG <- (An)+
  def NAME#REG.OOp.Letter#REG.Postfix : MxMove_RM<REG, REG.OOp, REG.OPat,
      SIZE, MxEncEAo_1, MxExtEmpty, EA_0, MxExtEmpty>;

  // REG <- -(An)
  def NAME#REG.EOp.Letter#REG.Postfix : MxMove_RM<REG, REG.EOp, REG.EPat,
      SIZE, MxEncEAe_1, MxExtEmpty, EA_0, MxExtEmpty>;

  // REG <- (i,PC,Xn)
  def NAME#REG.KOp.Letter#REG.Postfix : MxMove_RM<REG, REG.KOp, REG.KPat,
      SIZE, MxEncEAk, MxExtBrief_1, EA_0, MxExtEmpty>;

  // REG <- (i,PC)
  def NAME#REG.QOp.Letter#REG.Postfix : MxMove_RM<REG, REG.QOp, REG.QPat,
      SIZE, MxEncEAq, MxExtI16_1, EA_0, MxExtEmpty>;

  // REG <- (i,An,Xn)
  def NAME#REG.FOp.Letter#REG.Postfix : MxMove_RM<REG, REG.FOp, REG.FPat,
      SIZE, MxEncEAf_1, MxExtBrief_1, EA_0, MxExtEmpty>;

  // REG <- (i,An)
  def NAME#REG.POp.Letter#REG.Postfix : MxMove_RM<REG, REG.POp, REG.PPat,
      SIZE, MxEncEAp_1, MxExtI16_1, EA_0, MxExtEmpty>;

  // REG <- (ABS)
  def NAME#REG.BOp.Letter#REG.Postfix : MxMove_RM<REG, REG.BOp, REG.BPat,
      SIZE, MxEncEAb, MxExtI32_1, EA_0, MxExtEmpty>;

  // REG <- (An)
  def NAME#REG.JOp.Letter#REG.Postfix : MxMove_RM<REG, REG.JOp, REG.JPat,
      SIZE, MxEncEAj_1, MxExtEmpty, EA_0, MxExtEmpty>;
}

let mayLoad = 1, mayStore = 1 in {
class MxMove_MM<string SIZE, PatFrag LOAD,
                MxOperand DSTOpd, ComplexPattern DSTPat,
                MxOperand SRCOpd, ComplexPattern SRCPat,
                MxBead2Bits ESIZE,
                MxEncEA SRCEA, MxEncExt SRCEXT,
                MxEncEA DSTEA, MxEncExt DSTEXT>
    : MxMove<SIZE, (outs), (ins DSTOpd:$dst, SRCOpd:$src),
             [(store (LOAD SRCPat:$src), DSTPat:$dst)],
             MxMoveEncoding<ESIZE, SRCEA, SRCEXT, DSTEA, DSTEXT>>;
} // let mayLoad = 1, mayStore = 1

multiclass MMxMove_MM<MxType TYPE, MxOperand DSTOpd, ComplexPattern DSTPat,
                      MxMoveSize SIZE, MxEncEA EA_0, MxEncExt EXT_0> {

  // MEM <- (An)+
  def NAME#TYPE.OOp.Letter#TYPE.Postfix
    : MxMove_MM<TYPE.Prefix, TYPE.Load, DSTOpd, DSTPat, TYPE.OOp, TYPE.OPat,
                SIZE, MxEncEAo_1, MxExtEmpty, EA_0, EXT_0>;

  // MEM <- -(An)
  def NAME#TYPE.EOp.Letter#TYPE.Postfix
    : MxMove_MM<TYPE.Prefix, TYPE.Load, DSTOpd, DSTPat, TYPE.EOp, TYPE.EPat,
                SIZE, MxEncEAe_1, MxExtEmpty, EA_0, EXT_0>;

  // MEM <- (i,An)
  def NAME#TYPE.POp.Letter#TYPE.Postfix
    : MxMove_MM<TYPE.Prefix, TYPE.Load, DSTOpd, DSTPat, TYPE.POp, TYPE.PPat,
                SIZE, MxEncEAp_1, MxExtI16_1, EA_0, EXT_0>;

  // MEM <- (i,An,Xn)
  def NAME#TYPE.FOp.Letter#TYPE.Postfix
    : MxMove_MM<TYPE.Prefix, TYPE.Load, DSTOpd, DSTPat, TYPE.FOp, TYPE.FPat,
                SIZE, MxEncEAf_1, MxExtBrief_1, EA_0, EXT_0>;

  // MEM <- (i,PC,Xn)
  def NAME#TYPE.KOp.Letter#TYPE.Postfix
    : MxMove_MM<TYPE.Prefix, TYPE.Load, DSTOpd, DSTPat, TYPE.KOp, TYPE.KPat,
                SIZE, MxEncEAk, MxExtBrief_1, EA_0, EXT_0>;

  // MEM <- (i,PC)
  def NAME#TYPE.QOp.Letter#TYPE.Postfix
    : MxMove_MM<TYPE.Prefix, TYPE.Load, DSTOpd, DSTPat, TYPE.QOp, TYPE.QPat,
                SIZE, MxEncEAq, MxExtI16_1, EA_0, EXT_0>;

  // MEM <- (ABS)
  def NAME#TYPE.BOp.Letter#TYPE.Postfix
    : MxMove_MM<TYPE.Prefix, TYPE.Load, DSTOpd, DSTPat, TYPE.BOp, TYPE.BPat,
                SIZE, MxEncEAb, MxExtI32_1, EA_0, EXT_0>;

  // MEM <- (An)
  def NAME#TYPE.JOp.Letter#TYPE.Postfix
    : MxMove_MM<TYPE.Prefix, TYPE.Load, DSTOpd, DSTPat, TYPE.JOp, TYPE.JPat,
                SIZE, MxEncEAj_1, MxExtEmpty, EA_0, EXT_0>;
}

def MOV8dd
  : MxMove_RR<MxType8d, MxType8d,
    MxMoveEncoding<MxMoveSize8, MxEncEAd_1, MxExtEmpty, MxEncEAd_0, MxExtEmpty>>;

// M <- R
def MOV8fd : MxMove_MR<MxType8.FOp, MxType8.FPat, MxType8d,
                       MxMoveEncoding<MxMoveSize8,
                            /*src*/   MxEncEAd_1, MxExtEmpty,
                            /*dst*/   MxEncEAf_0, MxExtBrief_0>>;

def MOV8pd : MxMove_MR<MxType8.POp, MxType8.PPat, MxType8d,
                       MxMoveEncoding<MxMoveSize8,
                            /*src*/   MxEncEAd_1, MxExtEmpty,
                            /*dst*/   MxEncEAp_0, MxExtI16_0>>;

def MOV8ed : MxMove_MR<MxType8.EOp, MxType8.EPat, MxType8d,
                       MxMoveEncoding<MxMoveSize8,
                            /*src*/   MxEncEAd_1, MxExtEmpty,
                            /*dst*/   MxEncEAe_0, MxExtEmpty>>;

def MOV8od : MxMove_MR<MxType8.OOp, MxType8.OPat, MxType8d,
                       MxMoveEncoding<MxMoveSize8,
                            /*src*/   MxEncEAd_1, MxExtEmpty,
                            /*dst*/   MxEncEAo_0, MxExtEmpty>>;

def MOV8bd : MxMove_MR<MxType8.BOp, MxType8.BPat, MxType8d,
                       MxMoveEncoding<MxMoveSize8,
                            /*src*/   MxEncEAd_1, MxExtEmpty,
                            /*dst*/   MxEncEAb,   MxExtI32_0>>;

def MOV8jd : MxMove_MR<MxType8.JOp, MxType8.JPat, MxType8d,
                       MxMoveEncoding<MxMoveSize8,
                            /*src*/   MxEncEAd_1, MxExtEmpty,
                            /*dst*/   MxEncEAj_0, MxExtEmpty>>;


// R <- I
def MOV8di : MxMove_RI<MxType8d,
    MxMoveEncoding<MxMoveSize8, MxEncEAi, MxExtI8_1, MxEncEAd_0, MxExtEmpty>>;

foreach S = [16, 32] in {
  foreach D = [ "r", "a" ] in {

    foreach O = [ "r", "a" ] in {
      def MOV#S#D#O : MxMove_RR<
        !cast<MxType>("MxType"#S#D),
        !cast<MxType>("MxType"#S#O),
        MxMoveEncoding<!cast<MxMoveSize>("MxMoveSize"#S),
                       !cast<MxEncEA>("MxEncEA"#D#"_1"), MxExtEmpty,
                       !cast<MxEncEA>("MxEncEA"#D#"_0_reflected"), MxExtEmpty>>;
    }

    // M <- R
    def MOV#S#"f"#D : MxMove_MR<
      !cast<MxType>("MxType"#S).FOp,
      !cast<MxType>("MxType"#S).FPat,
      !cast<MxType>("MxType"#S#D),
      MxMoveEncoding<!cast<MxMoveSize>("MxMoveSize"#S),
                     !cast<MxEncEA>("MxEncEA"#D#"_1"), MxExtEmpty,
                     MxEncEAf_0, MxExtBrief_0>>;

    def MOV#S#"p"#D : MxMove_MR<
      !cast<MxType>("MxType"#S).POp,
      !cast<MxType>("MxType"#S).PPat,
      !cast<MxType>("MxType"#S#D),
      MxMoveEncoding<!cast<MxMoveSize>("MxMoveSize"#S),
                     !cast<MxEncEA>("MxEncEA"#D#"_1"), MxExtEmpty,
                     MxEncEAp_0, MxExtI16_0>>;

    def MOV#S#"e"#D : MxMove_MR<
      !cast<MxType>("MxType"#S).EOp,
      !cast<MxType>("MxType"#S).EPat,
      !cast<MxType>("MxType"#S#D),
      MxMoveEncoding<!cast<MxMoveSize>("MxMoveSize"#S),
                     !cast<MxEncEA>("MxEncEA"#D#"_1"), MxExtEmpty,
                     MxEncEAe_0, MxExtEmpty>>;

    def MOV#S#"o"#D : MxMove_MR<
      !cast<MxType>("MxType"#S).OOp,
      !cast<MxType>("MxType"#S).OPat,
      !cast<MxType>("MxType"#S#D),
      MxMoveEncoding<!cast<MxMoveSize>("MxMoveSize"#S),
                     !cast<MxEncEA>("MxEncEA"#D#"_1"), MxExtEmpty,
                     MxEncEAo_0, MxExtEmpty>>;

    def MOV#S#"b"#D : MxMove_MR<
      !cast<MxType>("MxType"#S).BOp,
      !cast<MxType>("MxType"#S).BPat,
      !cast<MxType>("MxType"#S#D),
      MxMoveEncoding<!cast<MxMoveSize>("MxMoveSize"#S),
                     !cast<MxEncEA>("MxEncEA"#D#"_1"), MxExtEmpty,
                     MxEncEAb, MxExtI32_0>>;

    def MOV#S#"j"#D : MxMove_MR<
      !cast<MxType>("MxType"#S).JOp,
      !cast<MxType>("MxType"#S).JPat,
      !cast<MxType>("MxType"#S#D),
      MxMoveEncoding<!cast<MxMoveSize>("MxMoveSize"#S),
                     !cast<MxEncEA>("MxEncEA"#D#"_1"), MxExtEmpty,
                     MxEncEAj_0, MxExtEmpty>>;


    // R <- I
    def MOV#S#D#"i" : MxMove_RI<
      !cast<MxType>("MxType"#S#D),
      MxMoveEncoding<!cast<MxMoveSize>("MxMoveSize"#S),
                     MxEncEAi, !cast<MxEncExt>("MxExtI"#S#"_1"),
                     !cast<MxEncEA>("MxEncEA"#D#"_0_reflected"), MxExtEmpty>>;
  }
}

// M <- I
foreach S = [8, 16, 32] in {
  def MOV#S#"f"#"i" : MxMove_MI<
    !cast<MxType>("MxType"#S).FOp,
    !cast<MxType>("MxType"#S).FPat,
    !cast<MxType>("MxType"#S),
    MxMoveEncoding<!cast<MxMoveSize>("MxMoveSize"#S),
                   MxEncEAi, !cast<MxEncExt>("MxExtI"#S#"_1"),
                   MxEncEAf_0, MxExtBrief_0>>;

  def MOV#S#"p"#"i" : MxMove_MI<
    !cast<MxType>("MxType"#S).POp,
    !cast<MxType>("MxType"#S).PPat,
    !cast<MxType>("MxType"#S),
    MxMoveEncoding<!cast<MxMoveSize>("MxMoveSize"#S),
                   MxEncEAi, !cast<MxEncExt>("MxExtI"#S#"_1"),
                   MxEncEAp_0, MxExtI16_0>>;

  def MOV#S#"b"#"i" : MxMove_MI<
    !cast<MxType>("MxType"#S).BOp,
    !cast<MxType>("MxType"#S).BPat,
    !cast<MxType>("MxType"#S),
    MxMoveEncoding<!cast<MxMoveSize>("MxMoveSize"#S),
                   MxEncEAi, !cast<MxEncExt>("MxExtI"#S#"_1"),
                   MxEncEAb, MxExtI32_0>>;

  def MOV#S#"j"#"i" : MxMove_MI<
    !cast<MxType>("MxType"#S).JOp,
    !cast<MxType>("MxType"#S).JPat,
    !cast<MxType>("MxType"#S),
    MxMoveEncoding<!cast<MxMoveSize>("MxMoveSize"#S),
                   MxEncEAi, !cast<MxEncExt>("MxExtI"#S#"_1"),
                   MxEncEAj_0, MxExtEmpty>>;
}

// Store ABS(basically pointer) as Immdiate to Mem
def : Pat<(store   MxType32.BPat :$src, MxType32.PPat :$dst),
          (MOV32pi MxType32.POp  :$dst, MxType32.IOp  :$src)>;

def : Pat<(store   MxType32.BPat :$src, MxType32.FPat :$dst),
          (MOV32fi MxType32.FOp  :$dst, MxType32.IOp  :$src)>;

def : Pat<(store   MxType32.BPat :$src, MxType32.BPat :$dst),
          (MOV32bi MxType32.BOp  :$dst, MxType32.IOp  :$src)>;

def : Pat<(store   MxType32.BPat :$src, MxType32.JPat :$dst),
          (MOV32ji MxType32.JOp  :$dst, MxType32.IOp  :$src)>;

// R <- M
defm MOV8d  : MMxMove_RM<MxType8d, MxMoveSize8, MxEncEAd_0>;

defm MOV16r : MMxMove_RM<MxType16r, MxMoveSize16, MxEncEAr_0_reflected>;
defm MOV16a : MMxMove_RM<MxType16a, MxMoveSize16, MxEncEAa_0>;

defm MOV32r : MMxMove_RM<MxType32r, MxMoveSize32, MxEncEAr_0_reflected>;
defm MOV32a : MMxMove_RM<MxType32a, MxMoveSize32, MxEncEAa_0>;

let Pattern = [(null_frag)] in {
defm MOV16r : MMxMove_RM<MxType16r_TC, MxMoveSize16, MxEncEAr_0_reflected>;
defm MOV16a : MMxMove_RM<MxType16a_TC, MxMoveSize16, MxEncEAa_0>;

defm MOV32r : MMxMove_RM<MxType32r_TC, MxMoveSize32, MxEncEAr_0_reflected>;
defm MOV32a : MMxMove_RM<MxType32a_TC, MxMoveSize32, MxEncEAa_0>;
} // Pattern

// M <- M
defm MOV8p  : MMxMove_MM<MxType8,      MxType8.POp,  MxType8.PPat,
                         MxMoveSize8,  MxEncEAp_0,   MxExtI16_0>;
defm MOV16p : MMxMove_MM<MxType16,     MxType16.POp, MxType16.PPat,
                         MxMoveSize16, MxEncEAp_0,   MxExtI16_0>;
defm MOV32p : MMxMove_MM<MxType32,     MxType32.POp, MxType32.PPat,
                         MxMoveSize32, MxEncEAp_0,   MxExtI16_0>;

defm MOV8f  : MMxMove_MM<MxType8,      MxType8.FOp,  MxType8.FPat,
                         MxMoveSize8,  MxEncEAf_0,   MxExtBrief_0>;
defm MOV16f : MMxMove_MM<MxType16,     MxType16.FOp, MxType16.FPat,
                         MxMoveSize16, MxEncEAf_0,   MxExtBrief_0>;
defm MOV32f : MMxMove_MM<MxType32,     MxType32.FOp, MxType32.FPat,
                         MxMoveSize32, MxEncEAf_0,   MxExtBrief_0>;

defm MOV8b  : MMxMove_MM<MxType8,      MxType8.BOp,  MxType8.BPat,
                         MxMoveSize8,  MxEncEAb,     MxExtI32_0>;
defm MOV16b : MMxMove_MM<MxType16,     MxType16.BOp, MxType16.BPat,
                         MxMoveSize16, MxEncEAb,     MxExtI32_0>;
defm MOV32b : MMxMove_MM<MxType32,     MxType32.BOp, MxType32.BPat,
                         MxMoveSize32, MxEncEAb,     MxExtI32_0>;

defm MOV8e  : MMxMove_MM<MxType8,      MxType8.EOp,  MxType8.EPat,
                         MxMoveSize8,  MxEncEAe_0,   MxExtEmpty>;
defm MOV16e : MMxMove_MM<MxType16,     MxType16.EOp, MxType16.EPat,
                         MxMoveSize16, MxEncEAe_0,   MxExtEmpty>;
defm MOV32e : MMxMove_MM<MxType32,     MxType32.EOp, MxType32.EPat,
                         MxMoveSize32, MxEncEAe_0,   MxExtEmpty>;

defm MOV8o  : MMxMove_MM<MxType8,      MxType8.OOp,  MxType8.OPat,
                         MxMoveSize8,  MxEncEAo_0,   MxExtEmpty>;
defm MOV16o : MMxMove_MM<MxType16,     MxType16.OOp, MxType16.OPat,
                         MxMoveSize16, MxEncEAo_0,   MxExtEmpty>;
defm MOV32o : MMxMove_MM<MxType32,     MxType32.OOp, MxType32.OPat,
                         MxMoveSize32, MxEncEAo_0,   MxExtEmpty>;

defm MOV8j  : MMxMove_MM<MxType8,      MxType8.JOp,  MxType8.JPat,
                         MxMoveSize8,  MxEncEAj_0,   MxExtEmpty>;
defm MOV16j : MMxMove_MM<MxType16,     MxType16.JOp, MxType16.JPat,
                         MxMoveSize16, MxEncEAj_0,   MxExtEmpty>;
defm MOV32j : MMxMove_MM<MxType32,     MxType32.JOp, MxType32.JPat,
                         MxMoveSize32, MxEncEAj_0,   MxExtEmpty>;

//===----------------------------------------------------------------------===//
// MOVEM
//
// The mask is already pre-processed by the save/restore spill hook
//===----------------------------------------------------------------------===//

// Direction
def MxMOVEM_MR : MxBead1Bit<0>;
def MxMOVEM_RM : MxBead1Bit<1>;

// Size
def MxMOVEM_W  : MxBead1Bit<0>;
def MxMOVEM_L  : MxBead1Bit<1>;

/// ---------------+-------------+-------------+---------
///  F  E  D  C  B | A | 9  8  7 | 6 | 5  4  3 | 2  1  0
/// ---------------+---+---------+---+---------+---------
///  0  1  0  0  1 | D | 0  0  1 | S |   MODE  |   REG
/// ---------------+---+---------+---+---------+---------
///                  REGISTER LIST MASK
/// -----------------------------------------------------
/// D - direction(RM,MR)
/// S - size(W,L)
class MxMOVEMEncoding<MxEncEA EA, MxEncExt EXT, MxBead1Bit SIZE, MxBead1Bit DIR,
                      MxBead16Imm IMM>
    : MxEncoding<EA.Reg, EA.DA, EA.Mode, SIZE, MxBead3Bits<0b001>, DIR,
                 MxBead1Bit<1>, MxBead4Bits<0b0100>, IMM,
                 EXT.Imm, EXT.B8, EXT.Scale, EXT.WL, EXT.DAReg>;

let mayStore = 1 in
class MxMOVEM_MR<MxType TYPE, MxBead1Bit SIZE,
                 MxOperand MEMOp, MxEncEA EA, MxEncExt EXT>
    : MxInst<(outs), (ins MEMOp:$dst, MxMoveMask:$mask),
             "movem."#TYPE.Prefix#"\t$mask, $dst", [],
             MxMOVEMEncoding<EA, EXT, SIZE, MxMOVEM_MR, MxBead16Imm<1>>>;

let mayLoad = 1 in
class MxMOVEM_RM<MxType TYPE, MxBead1Bit SIZE,
                 MxOperand MEMOp, MxEncEA EA, MxEncExt EXT>
    : MxInst<(outs), (ins MxMoveMask:$mask, MEMOp:$src),
             "movem."#TYPE.Prefix#"\t$src, $mask", [],
             MxMOVEMEncoding<EA, EXT, SIZE, MxMOVEM_RM, MxBead16Imm<0>>>;

def MOVM32jm : MxMOVEM_MR<MxType32, MxMOVEM_L, MxType32.JOp, MxEncEAj_0, MxExtEmpty>;
def MOVM32pm : MxMOVEM_MR<MxType32, MxMOVEM_L, MxType32.POp, MxEncEAp_0, MxExtI16_0>;

def MOVM32mj : MxMOVEM_RM<MxType32, MxMOVEM_L, MxType32.JOp, MxEncEAj_1, MxExtEmpty>;
def MOVM32mp : MxMOVEM_RM<MxType32, MxMOVEM_L, MxType32.POp, MxEncEAp_1, MxExtI16_1>;

// Pseudo versions. These a required by virtual register spill/restore since
// the mask requires real register to encode. These instruction will be expanded
// into real MOVEM after RA finishes.
let mayStore = 1 in
class MxMOVEM_MR_Pseudo<MxType TYPE, MxOperand MEMOp>
    : MxPseudo<(outs), (ins MEMOp:$dst, TYPE.ROp:$reg)>;
let mayLoad = 1 in
class MxMOVEM_RM_Pseudo<MxType TYPE, MxOperand MEMOp>
    : MxPseudo<(outs TYPE.ROp:$dst), (ins MEMOp:$src)>;

// Mem <- Reg
def MOVM8jm_P  : MxMOVEM_MR_Pseudo<MxType8d,  MxType8.JOp>;
def MOVM16jm_P : MxMOVEM_MR_Pseudo<MxType16r, MxType16.JOp>;
def MOVM32jm_P : MxMOVEM_MR_Pseudo<MxType32r, MxType32.JOp>;

def MOVM8pm_P  : MxMOVEM_MR_Pseudo<MxType8d,  MxType8.POp>;
def MOVM16pm_P : MxMOVEM_MR_Pseudo<MxType16r, MxType16.POp>;
def MOVM32pm_P : MxMOVEM_MR_Pseudo<MxType32r, MxType32.POp>;

// Reg <- Mem
def MOVM8mj_P  : MxMOVEM_RM_Pseudo<MxType8d,  MxType8.JOp>;
def MOVM16mj_P : MxMOVEM_RM_Pseudo<MxType16r, MxType16.JOp>;
def MOVM32mj_P : MxMOVEM_RM_Pseudo<MxType32r, MxType32.JOp>;

def MOVM8mp_P  : MxMOVEM_RM_Pseudo<MxType8d,  MxType8.POp>;
def MOVM16mp_P : MxMOVEM_RM_Pseudo<MxType16r, MxType16.POp>;
def MOVM32mp_P : MxMOVEM_RM_Pseudo<MxType32r, MxType32.POp>;


//===----------------------------------------------------------------------===//
// MOVE to/from SR/CCR
//
// A special care must be taken working with to/from CCR since it is basically
// word-size SR register truncated for user mode thus it only supports word-size
// instructions. Plus the original M68000 does not support moves from CCR. So in
// order to use CCR effectively one MUST use proper byte-size pseudo instructi-
// ons that will be resolved sometime after RA pass.
//===----------------------------------------------------------------------===//

/// --------------------------------------------------
///  F  E  D  C  B  A  9  8  7  6 | 5  4  3 | 2  1  0
/// --------------------------------------------------
///                               | EFFECTIVE ADDRESS
///  0  1  0  0  0  1  0  0  1  1 |   MODE  |   REG
/// --------------------------------------------------
let Defs = [CCR] in
class MxMoveToCCR<dag INS, MxEncEA EA, MxEncExt EXT>
    : MxInst<(outs CCRC:$dst), INS, "move.w\t$src, $dst", [],
             MxEncoding<EA.Reg, EA.DA, EA.Mode,
                        MxBead4Bits<0b0011>, MxBead4Bits<0b0001>, MxBead2Bits<0b01>,
                        EXT.Imm, EXT.B8, EXT.Scale, EXT.WL, EXT.DAReg>>;

class MxMoveToCCRPseudo<dag INS> : MxPseudo<(outs CCRC:$dst), INS>;

let mayLoad = 1 in {
def MOV16cp : MxMoveToCCR<(ins MxType16d.POp:$src), MxEncEAp_1, MxExtI16_1>;
def  MOV8cp : MxMoveToCCRPseudo<(ins MxType8d.POp:$src)>;
} // let mayLoad = 1

def MOV16cd : MxMoveToCCR<(ins MxType16d.ROp:$src), MxEncEAd_1, MxExtEmpty>;
def  MOV8cd : MxMoveToCCRPseudo<(ins MxType8d.ROp:$src)>;

/// Move from CCR
/// --------------------------------------------------
///  F  E  D  C  B  A  9  8  7  6 | 5  4  3 | 2  1  0
/// --------------------------------------------------
///                               | EFFECTIVE ADDRESS
///  0  1  0  0  0  0  1  0  1  1 |   MODE  |   REG
/// --------------------------------------------------
let Uses = [CCR] in
class MxMoveFromCCR<dag OUTS, dag INS, MxEncEA EA, MxEncExt EXT>
    : MxInst<OUTS, INS, "move.w\t$src, $dst", [],
             MxEncoding<EA.Reg, EA.DA, EA.Mode,
                        MxBead4Bits<0b1011>, MxBead4Bits<0b0000>, MxBead2Bits<0b01>,
                        EXT.Imm, EXT.B8, EXT.Scale, EXT.WL, EXT.DAReg>>,
      Requires<[ IsM68010 ]>;

class MxMoveFromCCRPseudo<dag INS> : MxPseudo<(outs), INS>;

let mayStore = 1 in {
def MOV16pc
  : MxMoveFromCCR<(outs), (ins MxType16d.POp:$dst, CCRC:$src), MxEncEAp_0, MxExtI16_0>;
def MOV8pc : MxMoveFromCCRPseudo<(ins MxType8d.POp:$dst, CCRC:$src)>;
} // let mayStore = 1

def MOV16dc
  : MxMoveFromCCR<(outs MxType16d.ROp:$dst), (ins CCRC:$src), MxEncEAd_0, MxExtEmpty>;

def MOV8dc : MxMoveFromCCRPseudo<(ins MxType8d.ROp:$dst, CCRC:$src)>;


//===----------------------------------------------------------------------===//
// LEA
//===----------------------------------------------------------------------===//

/// ----------------------------------------------------
///  F  E  D  C | B  A  9 | 8  7  6 | 5  4  3 | 2  1  0
/// ----------------------------------------------------
///  0  1  0  0 | DST REG | 1  1  1 |   MODE  |   REG
/// ----------------------------------------------------
class MxLEA<MxOperand SRCOpd, ComplexPattern SRCPat, MxEncEA EA, MxEncExt EXT>
    : MxInst<(outs MxARD32:$dst), (ins SRCOpd:$src),
             "lea\t$src, $dst", [(set i32:$dst, SRCPat:$src)],
             MxEncoding<EA.Reg, EA.DA, EA.Mode,
                        MxBead3Bits<0b111>, MxBeadReg<0>, MxBead4Bits<0x4>,
                        EXT.Imm, EXT.B8, EXT.Scale, EXT.WL, EXT.DAReg>>;

def LEA32p : MxLEA<MxARID32, MxCP_ARID, MxEncEAp_1, MxExtI16_1>;
def LEA32f : MxLEA<MxARII32, MxCP_ARII, MxEncEAf_1, MxExtBrief_1>;
def LEA32q : MxLEA<MxPCD32,  MxCP_PCD,  MxEncEAq,   MxExtI16_1>;
def LEA32b : MxLEA<MxAL32,   MxCP_AL,   MxEncEAb,   MxExtI32_1>;


//===----------------------------------------------------------------------===//
// Pseudos
//===----------------------------------------------------------------------===//

/// Pushe/Pop to/from SP for simplicity
let Uses = [SP], Defs = [SP], hasSideEffects = 0 in {

// SP <- SP - <size>; (SP) <- Dn
let mayStore = 1 in {
def PUSH8d  : MxPseudo<(outs), (ins DR8:$reg)>;
def PUSH16d : MxPseudo<(outs), (ins DR16:$reg)>;
def PUSH32r : MxPseudo<(outs), (ins XR32:$reg)>;
} // let mayStore = 1

// Dn <- (SP); SP <- SP + <size>
let mayLoad = 1 in {
def POP8d  : MxPseudo<(outs DR8:$reg),  (ins)>;
def POP16d : MxPseudo<(outs DR16:$reg), (ins)>;
def POP32r : MxPseudo<(outs XR32:$reg), (ins)>;
} // let mayLoad = 1

} // let Uses/Defs = [SP], hasSideEffects = 0


let Defs = [CCR] in {
class MxPseudoMove_RR<MxType DST, MxType SRC, list<dag> PAT = []>
    : MxPseudo<(outs DST.ROp:$dst), (ins SRC.ROp:$src), PAT>;

class MxPseudoMove_RM<MxType DST, MxOperand SRCOpd, list<dag> PAT = []>
    : MxPseudo<(outs DST.ROp:$dst), (ins SRCOpd:$src), PAT>;
}

/// This group of Pseudos is analogues to the real x86 extending moves, but
/// since M68k does not have those we need to emulate. These instructions
/// will be expanded right after RA completed because we need to know precisely
/// what registers are allocated for the operands and if they overlap we just
/// extend the value if the registers are completely different we need to move
/// first.
foreach EXT = ["S", "Z"] in {
  let hasSideEffects = 0 in {

    def MOV#EXT#Xd16d8  : MxPseudoMove_RR<MxType16d,  MxType8d>;
    def MOV#EXT#Xd32d8  : MxPseudoMove_RR<MxType32d,  MxType8d>;
    def MOV#EXT#Xd32d16 : MxPseudoMove_RR<MxType32r, MxType16r>;

    let mayLoad = 1 in {

      def MOV#EXT#Xd16j8   : MxPseudoMove_RM<MxType16d,  MxType8.JOp>;
      def MOV#EXT#Xd32j8   : MxPseudoMove_RM<MxType32d,  MxType8.JOp>;
      def MOV#EXT#Xd32j16  : MxPseudoMove_RM<MxType32d, MxType16.JOp>;

      def MOV#EXT#Xd16p8   : MxPseudoMove_RM<MxType16d,  MxType8.POp>;
      def MOV#EXT#Xd32p8   : MxPseudoMove_RM<MxType32d,  MxType8.POp>;
      def MOV#EXT#Xd32p16  : MxPseudoMove_RM<MxType32d, MxType16.POp>;

      def MOV#EXT#Xd16f8   : MxPseudoMove_RM<MxType16d,  MxType8.FOp>;
      def MOV#EXT#Xd32f8   : MxPseudoMove_RM<MxType32d,  MxType8.FOp>;
      def MOV#EXT#Xd32f16  : MxPseudoMove_RM<MxType32d, MxType16.FOp>;

    }
  }
}

/// This group of instructions is similar to the group above but DOES NOT do
/// any value extension, they just load a smaller register into the lower part
/// of another register if operands' real registers are different or does
/// nothing if they are the same.
def MOVXd16d8  : MxPseudoMove_RR<MxType16d,  MxType8d>;
def MOVXd32d8  : MxPseudoMove_RR<MxType32d,  MxType8d>;
def MOVXd32d16 : MxPseudoMove_RR<MxType32r, MxType16r>;

//===----------------------------------------------------------------------===//
// Extend/Truncate Patterns
//===----------------------------------------------------------------------===//

// i16 <- sext i8
def: Pat<(i16 (sext i8:$src)),
          (EXTRACT_SUBREG (MOVSXd32d8 MxDRD8:$src), MxSubRegIndex16Lo)>;
def: Pat<(MxSExtLoadi16i8 MxCP_ARI:$src),
          (EXTRACT_SUBREG (MOVSXd32j8 MxARI8:$src), MxSubRegIndex16Lo)>;
def: Pat<(MxSExtLoadi16i8 MxCP_ARID:$src),
          (EXTRACT_SUBREG (MOVSXd32p8 MxARID8:$src), MxSubRegIndex16Lo)>;
def: Pat<(MxSExtLoadi16i8 MxCP_ARII:$src),
          (EXTRACT_SUBREG (MOVSXd32f8 MxARII8:$src), MxSubRegIndex16Lo)>;

// i32 <- sext i8
def: Pat<(i32 (sext i8:$src)), (MOVSXd32d8 MxDRD8:$src)>;
def: Pat<(MxSExtLoadi32i8 MxCP_ARI :$src), (MOVSXd32j8 MxARI8 :$src)>;
def: Pat<(MxSExtLoadi32i8 MxCP_ARID:$src), (MOVSXd32p8 MxARID8:$src)>;
def: Pat<(MxSExtLoadi32i8 MxCP_ARII:$src), (MOVSXd32f8 MxARII8:$src)>;

// i32 <- sext i16
def: Pat<(i32 (sext i16:$src)), (MOVSXd32d16 MxDRD16:$src)>;
def: Pat<(MxSExtLoadi32i16 MxCP_ARI :$src), (MOVSXd32j16 MxARI16 :$src)>;
def: Pat<(MxSExtLoadi32i16 MxCP_ARID:$src), (MOVSXd32p16 MxARID16:$src)>;
def: Pat<(MxSExtLoadi32i16 MxCP_ARII:$src), (MOVSXd32f16 MxARII16:$src)>;

// i16 <- zext i8
def: Pat<(i16 (zext i8:$src)),
          (EXTRACT_SUBREG (MOVZXd32d8 MxDRD8:$src), MxSubRegIndex16Lo)>;
def: Pat<(MxZExtLoadi16i8 MxCP_ARI:$src),
          (EXTRACT_SUBREG (MOVZXd32j8 MxARI8:$src), MxSubRegIndex16Lo)>;
def: Pat<(MxZExtLoadi16i8 MxCP_ARID:$src),
          (EXTRACT_SUBREG (MOVZXd32p8 MxARID8:$src), MxSubRegIndex16Lo)>;
def: Pat<(MxZExtLoadi16i8 MxCP_ARII:$src),
          (EXTRACT_SUBREG (MOVZXd32f8 MxARII8:$src), MxSubRegIndex16Lo)>;

// i32 <- zext i8
def: Pat<(i32 (zext i8:$src)), (MOVZXd32d8 MxDRD8:$src)>;
def: Pat<(MxZExtLoadi32i8 MxCP_ARI :$src), (MOVZXd32j8 MxARI8 :$src)>;
def: Pat<(MxZExtLoadi32i8 MxCP_ARID:$src), (MOVZXd32p8 MxARID8:$src)>;
def: Pat<(MxZExtLoadi32i8 MxCP_ARII:$src), (MOVZXd32f8 MxARII8:$src)>;

// i32 <- zext i16
def: Pat<(i32 (zext i16:$src)), (MOVZXd32d16 MxDRD16:$src)>;
def: Pat<(MxZExtLoadi32i16 MxCP_ARI :$src), (MOVZXd32j16 MxARI16 :$src)>;
def: Pat<(MxZExtLoadi32i16 MxCP_ARID:$src), (MOVZXd32p16 MxARID16:$src)>;
def: Pat<(MxZExtLoadi32i16 MxCP_ARII:$src), (MOVZXd32f16 MxARII16:$src)>;

// i16 <- anyext i8
def: Pat<(i16 (anyext i8:$src)),
          (EXTRACT_SUBREG (MOVZXd32d8 MxDRD8:$src), MxSubRegIndex16Lo)>;
def: Pat<(MxExtLoadi16i8 MxCP_ARI:$src),
          (EXTRACT_SUBREG (MOVZXd32j8 MxARI8:$src), MxSubRegIndex16Lo)>;
def: Pat<(MxExtLoadi16i8 MxCP_ARID:$src),
          (EXTRACT_SUBREG (MOVZXd32p8 MxARID8:$src), MxSubRegIndex16Lo)>;
def: Pat<(MxExtLoadi16i8 MxCP_ARII:$src),
          (EXTRACT_SUBREG (MOVZXd32f8 MxARII8:$src), MxSubRegIndex16Lo)>;

// i32 <- anyext i8
def: Pat<(i32 (anyext i8:$src)), (MOVZXd32d8 MxDRD8:$src)>;
def: Pat<(MxExtLoadi32i8 MxCP_ARI :$src), (MOVZXd32j8 MxARI8 :$src)>;
def: Pat<(MxExtLoadi32i8 MxCP_ARID:$src), (MOVZXd32p8 MxARID8:$src)>;
def: Pat<(MxExtLoadi32i8 MxCP_ARII:$src), (MOVZXd32f8 MxARII8:$src)>;

// i32 <- anyext i16
def: Pat<(i32 (anyext i16:$src)), (MOVZXd32d16 MxDRD16:$src)>;
def: Pat<(MxExtLoadi32i16 MxCP_ARI :$src), (MOVZXd32j16 MxARI16 :$src)>;
def: Pat<(MxExtLoadi32i16 MxCP_ARID:$src), (MOVZXd32p16 MxARID16:$src)>;
def: Pat<(MxExtLoadi32i16 MxCP_ARII:$src), (MOVZXd32f16 MxARII16:$src)>;

// trunc patterns
def : Pat<(i16 (trunc i32:$src)),
          (EXTRACT_SUBREG MxXRD32:$src, MxSubRegIndex16Lo)>;
def : Pat<(i8  (trunc i32:$src)),
          (EXTRACT_SUBREG MxXRD32:$src, MxSubRegIndex8Lo)>;
def : Pat<(i8  (trunc i16:$src)),
          (EXTRACT_SUBREG MxXRD16:$src, MxSubRegIndex8Lo)>;

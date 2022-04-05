//===-- M68kInstrData.td - M68k Data Movement Instructions -*- tablegen -*-===//
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

// MOVE has a different size encoding.
class MxMoveSize<bits<2> value> {
  bits<2> Value = value;
}
def MxMoveSize8  : MxMoveSize<0b01>;
def MxMoveSize16 : MxMoveSize<0b11>;
def MxMoveSize32 : MxMoveSize<0b10>;

class MxMoveEncoding<MxMoveSize size, MxEncMemOp dst_enc, MxEncMemOp src_enc> {
  dag Value = (ascend
    (descend 0b00, size.Value,
             !cond(
               !eq(!getdagop(dst_enc.EA), descend): !setdagop(dst_enc.EA, ascend),
               !eq(!getdagop(dst_enc.EA), ascend): !setdagop(dst_enc.EA, descend)),
             src_enc.EA),
    // Source extension
    src_enc.Supplement,
    // Destination extension
    dst_enc.Supplement
  );
}

// Special encoding for Xn
class MxMoveEncAddrMode_r<string reg_opnd> : MxEncMemOp {
  let EA = (descend (descend 0b00, (slice "$"#reg_opnd, 3, 3)),
                    (operand "$"#reg_opnd, 3));
}

multiclass MxMoveOperandEncodings<string opnd_name> {
  // Dn
  def MxMove#NAME#OpEnc_d : MxEncAddrMode_d<opnd_name>;
  // An
  def MxMove#NAME#OpEnc_a : MxEncAddrMode_a<opnd_name>;
  // Xn
  def MxMove#NAME#OpEnc_r : MxMoveEncAddrMode_r<opnd_name>;
  // (An)+
  def MxMove#NAME#OpEnc_o : MxEncAddrMode_o<opnd_name>;
  // -(An)
  def MxMove#NAME#OpEnc_e : MxEncAddrMode_e<opnd_name>;
  // (i,PC,Xn)
  def MxMove#NAME#OpEnc_k : MxEncAddrMode_k<opnd_name>;
  // (i,PC)
  def MxMove#NAME#OpEnc_q : MxEncAddrMode_q<opnd_name>;
  // (i,An,Xn)
  def MxMove#NAME#OpEnc_f : MxEncAddrMode_f<opnd_name>;
  // (i,An)
  def MxMove#NAME#OpEnc_p : MxEncAddrMode_p<opnd_name>;
  // (ABS).L
  def MxMove#NAME#OpEnc_b : MxEncAddrMode_abs<opnd_name, /*W/L=*/true>;
  // (An)
  def MxMove#NAME#OpEnc_j : MxEncAddrMode_j<opnd_name>;
}

defm Src : MxMoveOperandEncodings<"src">;
defm Dst : MxMoveOperandEncodings<"dst">;

defvar MxMoveSupportedAMs = ["o", "e", "k", "q", "f", "p", "b", "j"];

let Defs = [CCR] in
class MxMove<string size, dag outs, dag ins, list<dag> pattern, MxMoveEncoding enc>
    : MxInst<outs, ins, "move."#size#"\t$src, $dst", pattern> {
  let Inst = enc.Value;
}

class MxMove_RR<MxType TYPE, string DST_REG, string SRC_REG,
                MxMoveEncoding ENC,
                MxOpBundle DST = !cast<MxOpBundle>("MxOp"#TYPE.Size#"AddrMode_"#DST_REG),
                MxOpBundle SRC = !cast<MxOpBundle>("MxOp"#TYPE.Size#"AddrMode_"#SRC_REG)>
    : MxMove<TYPE.Prefix,
             (outs DST.Op:$dst), (ins SRC.Op:$src),
             [(null_frag)], ENC>;

let mayStore = 1 in {
class MxMove_MR<MxType TYPE, MxOpBundle DST, string SRC_REG, MxMoveEncoding ENC,
                MxOpBundle SRC = !cast<MxOpBundle>("MxOp"#TYPE.Size#"AddrMode_"#SRC_REG)>
    : MxMove<TYPE.Prefix, (outs), (ins DST.Op:$dst, SRC.Op:$src),
             [(store TYPE.VT:$src, DST.Pat:$dst)], ENC>;

class MxMove_MI<MxType TYPE, MxOpBundle DST, MxMoveEncoding ENC,
                MxImmOpBundle SRC = !cast<MxImmOpBundle>("MxOp"#TYPE.Size#"AddrMode_i")>
    : MxMove<TYPE.Prefix, (outs), (ins DST.Op:$dst, SRC.Op:$src),
             [(store SRC.ImmPat:$src, DST.Pat:$dst)], ENC>;
} // let mayStore = 1

class MxMove_RI<MxType TYPE, string DST_REG, MxMoveEncoding ENC,
                MxImmOpBundle SRC = !cast<MxImmOpBundle>("MxOp"#TYPE.Size#"AddrMode_i"),
                MxOpBundle DST = !cast<MxOpBundle>("MxOp"#TYPE.Size#"AddrMode_"#DST_REG)>
    : MxMove<TYPE.Prefix, (outs DST.Op:$dst), (ins SRC.Op:$src),
              [(set TYPE.VT:$dst, SRC.ImmPat:$src)], ENC>;


let mayLoad = 1 in
class MxMove_RM<MxType TYPE, string DST_REG, MxOpBundle SRC, MxEncMemOp SRC_ENC,
                MxMoveSize SIZE_ENC = !cast<MxMoveSize>("MxMoveSize"#TYPE.Size),
                MxOpBundle DST = !cast<MxOpBundle>("MxOp"#TYPE.Size#"AddrMode_"#DST_REG),
                MxEncMemOp DST_ENC = !cast<MxEncMemOp>("MxMoveDstOpEnc_"#DST_REG)>
    : MxMove<TYPE.Prefix, (outs DST.Op:$dst), (ins SRC.Op:$src),
             [(set TYPE.VT:$dst, (TYPE.Load SRC.Pat:$src))],
             MxMoveEncoding<SIZE_ENC, DST_ENC, SRC_ENC>>;

foreach REG = ["r", "a", "d"] in
foreach AM = MxMoveSupportedAMs in {
  foreach TYPE = !if(!eq(REG, "d"), [MxType8, MxType16, MxType32], [MxType16, MxType32]) in
  def MOV # TYPE.Size # REG # AM # TYPE.Postfix
      : MxMove_RM<TYPE, REG, !cast<MxOpBundle>("MxOp"#TYPE.Size#"AddrMode_"#AM),
                  !cast<MxEncMemOp>("MxMoveSrcOpEnc_"#AM)>;
} // foreach AM

let mayLoad = 1, mayStore = 1 in
class MxMove_MM<MxType TYPE, MxOpBundle DST, MxOpBundle SRC,
                MxEncMemOp DST_ENC, MxEncMemOp SRC_ENC>
    : MxMove<TYPE.Prefix, (outs), (ins DST.Op:$dst, SRC.Op:$src),
             [(store (TYPE.Load SRC.Pat:$src), DST.Pat:$dst)],
             MxMoveEncoding<!cast<MxMoveSize>("MxMoveSize"#TYPE.Size),
                            DST_ENC, SRC_ENC>>;

foreach DST_AM = MxMoveSupportedAMs in
foreach SRC_AM = MxMoveSupportedAMs in {
  foreach TYPE = [MxType8, MxType16, MxType32] in
  def MOV # TYPE.Size # DST_AM # SRC_AM # TYPE.Postfix
      : MxMove_MM<TYPE, !cast<MxOpBundle>("MxOp"#TYPE.Size#"AddrMode_"#DST_AM),
                  !cast<MxOpBundle>("MxOp"#TYPE.Size#"AddrMode_"#SRC_AM),
                  !cast<MxEncMemOp>("MxMoveDstOpEnc_"#DST_AM),
                  !cast<MxEncMemOp>("MxMoveSrcOpEnc_"#SRC_AM)>;
} // foreach SRC_AM

foreach DST_REG = ["r", "a"] in {
  foreach SRC_REG = ["r", "a"] in
  foreach TYPE = [MxType16, MxType32] in
  def MOV # TYPE.Size # DST_REG # SRC_REG # TYPE.Postfix
      : MxMove_RR<TYPE, DST_REG, SRC_REG,
                  MxMoveEncoding<!cast<MxMoveSize>("MxMoveSize"#TYPE.Size),
                                 !cast<MxEncMemOp>("MxMoveDstOpEnc_"#DST_REG),
                                 !cast<MxEncMemOp>("MxMoveSrcOpEnc_"#SRC_REG)>>;
} // foreach DST_REG
foreach TYPE = [MxType8, MxType16, MxType32] in
def MOV # TYPE.Size # dd # TYPE.Postfix
    : MxMove_RR<TYPE, "d", "d",
                MxMoveEncoding<!cast<MxMoveSize>("MxMoveSize"#TYPE.Size),
                               MxMoveDstOpEnc_d, MxMoveSrcOpEnc_d>>;


foreach REG = ["r", "a", "d"] in
foreach AM = MxMoveSupportedAMs in {
  foreach TYPE = !if(!eq(REG, "d"), [MxType8, MxType16, MxType32], [MxType16, MxType32]) in
  def MOV # TYPE.Size # AM # REG # TYPE.Postfix
      : MxMove_MR<TYPE, !cast<MxOpBundle>("MxOp"#TYPE.Size#"AddrMode_"#AM), REG,
                  MxMoveEncoding<!cast<MxMoveSize>("MxMoveSize"#TYPE.Size),
                                 !cast<MxEncMemOp>("MxMoveDstOpEnc_"#AM),
                                 !cast<MxEncMemOp>("MxMoveSrcOpEnc_"#REG)>>;
} // foreach AM

foreach REG = ["r", "a", "d"] in {
  foreach TYPE = !if(!eq(REG, "d"), [MxType8, MxType16, MxType32], [MxType16, MxType32]) in
  def MOV # TYPE.Size # REG # i # TYPE.Postfix
      : MxMove_RI<TYPE, REG,
                  MxMoveEncoding<!cast<MxMoveSize>("MxMoveSize"#TYPE.Size),
                                 !cast<MxEncMemOp>("MxMoveDstOpEnc_"#REG),
                                 MxEncAddrMode_i<"src", TYPE.Size>>>;
} // foreach REG

foreach AM = MxMoveSupportedAMs in {
  foreach TYPE = [MxType8, MxType16, MxType32] in
  def MOV # TYPE.Size # AM # i # TYPE.Postfix
      : MxMove_MI<TYPE, !cast<MxOpBundle>("MxOp"#TYPE.Size#"AddrMode_"#AM),
                  MxMoveEncoding<!cast<MxMoveSize>("MxMoveSize"#TYPE.Size),
                                 !cast<MxEncMemOp>("MxMoveDstOpEnc_"#AM),
                                 MxEncAddrMode_i<"src", TYPE.Size>>>;
} // foreach AM

// Store ABS(basically pointer) as Immdiate to Mem
def : Pat<(store   MxType32.BPat :$src, MxType32.PPat :$dst),
          (MOV32pi MxType32.POp  :$dst, MxType32.IOp  :$src)>;

def : Pat<(store   MxType32.BPat :$src, MxType32.FPat :$dst),
          (MOV32fi MxType32.FOp  :$dst, MxType32.IOp  :$src)>;

def : Pat<(store   MxType32.BPat :$src, MxType32.BPat :$dst),
          (MOV32bi MxType32.BOp  :$dst, MxType32.IOp  :$src)>;

def : Pat<(store   MxType32.BPat :$src, MxType32.JPat :$dst),
          (MOV32ji MxType32.JOp  :$dst, MxType32.IOp  :$src)>;

let Pattern = [(null_frag)] in {
  foreach REG = ["r", "a"] in
  foreach AM = MxMoveSupportedAMs in {
    foreach TYPE = [MxType16, MxType32] in
    def MOV # TYPE.Size # REG # AM # _TC
        : MxMove_RM<TYPE, REG, !cast<MxOpBundle>("MxOp"#TYPE.Size#"AddrMode_"#AM),
                    !cast<MxEncMemOp>("MxMoveSrcOpEnc_"#AM)>;
  } // foreach AM
} // let Pattern

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

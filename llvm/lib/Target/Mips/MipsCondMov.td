//===-- MipsCondMov.td - Describe Mips Conditional Moves --*- tablegen -*--===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This is the Conditional Moves implementation.
//
//===----------------------------------------------------------------------===//

// Conditional moves:
// These instructions are expanded in
// MipsISelLowering::EmitInstrWithCustomInserter if target does not have
// conditional move instructions.
// cond:int, data:int
class CMov_I_I_FT<string opstr, RegisterOperand CRC, RegisterOperand DRC,
                  InstrItinClass Itin> :
  InstSE<(outs DRC:$rd), (ins DRC:$rs, CRC:$rt, DRC:$F),
         !strconcat(opstr, "\t$rd, $rs, $rt"), [], Itin, FrmFR, opstr> {
  let Constraints = "$F = $rd";
}

// cond:int, data:float
class CMov_I_F_FT<string opstr, RegisterOperand CRC, RegisterOperand DRC,
                  InstrItinClass Itin> :
  InstSE<(outs DRC:$fd), (ins DRC:$fs, CRC:$rt, DRC:$F),
         !strconcat(opstr, "\t$fd, $fs, $rt"), [], Itin, FrmFR, opstr>,
  HARDFLOAT {
  let Constraints = "$F = $fd";
}

// cond:float, data:int
class CMov_F_I_FT<string opstr, RegisterOperand RC, InstrItinClass Itin,
                  SDPatternOperator OpNode = null_frag> :
  InstSE<(outs RC:$rd), (ins RC:$rs, FCCRegsOpnd:$fcc, RC:$F),
         !strconcat(opstr, "\t$rd, $rs, $fcc"),
         [(set RC:$rd, (OpNode RC:$rs, FCCRegsOpnd:$fcc, RC:$F))],
         Itin, FrmFR, opstr>, HARDFLOAT {
  let Constraints = "$F = $rd";
}

// cond:float, data:float
class CMov_F_F_FT<string opstr, RegisterOperand RC, InstrItinClass Itin,
                  SDPatternOperator OpNode = null_frag> :
  InstSE<(outs RC:$fd), (ins RC:$fs, FCCRegsOpnd:$fcc, RC:$F),
         !strconcat(opstr, "\t$fd, $fs, $fcc"),
         [(set RC:$fd, (OpNode RC:$fs, FCCRegsOpnd:$fcc, RC:$F))],
         Itin, FrmFR, opstr>, HARDFLOAT {
  let Constraints = "$F = $fd";
}

// Instantiation of instructions.
def MOVZ_I_I : MMRel, CMov_I_I_FT<"movz", GPR32Opnd, GPR32Opnd, II_MOVZ>,
               ADD_FM<0, 0xa>, INSN_MIPS4_32_NOT_32R6_64R6;

let isCodeGenOnly = 1 in {
  def MOVZ_I_I64   : CMov_I_I_FT<"movz", GPR32Opnd, GPR64Opnd, II_MOVZ>,
                     ADD_FM<0, 0xa>, INSN_MIPS4_32_NOT_32R6_64R6;
  def MOVZ_I64_I   : CMov_I_I_FT<"movz", GPR64Opnd, GPR32Opnd, II_MOVZ>,
                     ADD_FM<0, 0xa>, INSN_MIPS4_32_NOT_32R6_64R6;
  def MOVZ_I64_I64 : CMov_I_I_FT<"movz", GPR64Opnd, GPR64Opnd, II_MOVZ>,
                     ADD_FM<0, 0xa>, INSN_MIPS4_32_NOT_32R6_64R6;
}

def MOVN_I_I       : MMRel, CMov_I_I_FT<"movn", GPR32Opnd, GPR32Opnd, II_MOVN>,
                     ADD_FM<0, 0xb>, INSN_MIPS4_32_NOT_32R6_64R6;

let isCodeGenOnly = 1 in {
  def MOVN_I_I64   : CMov_I_I_FT<"movn", GPR32Opnd, GPR64Opnd, II_MOVN>,
                     ADD_FM<0, 0xb>, INSN_MIPS4_32_NOT_32R6_64R6;
  def MOVN_I64_I   : CMov_I_I_FT<"movn", GPR64Opnd, GPR32Opnd, II_MOVN>,
                     ADD_FM<0, 0xb>, INSN_MIPS4_32_NOT_32R6_64R6;
  def MOVN_I64_I64 : CMov_I_I_FT<"movn", GPR64Opnd, GPR64Opnd, II_MOVN>,
                     ADD_FM<0, 0xb>, INSN_MIPS4_32_NOT_32R6_64R6;
}

def MOVZ_I_S : MMRel, CMov_I_F_FT<"movz.s", GPR32Opnd, FGR32Opnd, II_MOVZ_S>,
               CMov_I_F_FM<18, 16>, INSN_MIPS4_32_NOT_32R6_64R6;

let isCodeGenOnly = 1 in
def MOVZ_I64_S : CMov_I_F_FT<"movz.s", GPR64Opnd, FGR32Opnd, II_MOVZ_S>,
                 CMov_I_F_FM<18, 16>, INSN_MIPS4_32_NOT_32R6_64R6,
                 AdditionalRequires<[HasMips64]>;

def MOVN_I_S : MMRel, CMov_I_F_FT<"movn.s", GPR32Opnd, FGR32Opnd, II_MOVN_S>,
               CMov_I_F_FM<19, 16>, INSN_MIPS4_32_NOT_32R6_64R6;

let isCodeGenOnly = 1 in
def MOVN_I64_S : CMov_I_F_FT<"movn.s", GPR64Opnd, FGR32Opnd, II_MOVN_S>,
                 CMov_I_F_FM<19, 16>, INSN_MIPS4_32_NOT_32R6_64R6,
                 AdditionalRequires<[IsGP64bit]>;

def MOVZ_I_D32 : MMRel, CMov_I_F_FT<"movz.d", GPR32Opnd, AFGR64Opnd,
                                    II_MOVZ_D>, CMov_I_F_FM<18, 17>,
                 INSN_MIPS4_32_NOT_32R6_64R6, FGR_32;
def MOVN_I_D32 : MMRel, CMov_I_F_FT<"movn.d", GPR32Opnd, AFGR64Opnd,
                                    II_MOVN_D>, CMov_I_F_FM<19, 17>,
                 INSN_MIPS4_32_NOT_32R6_64R6, FGR_32;

let DecoderNamespace = "Mips64" in {
  def MOVZ_I_D64 : CMov_I_F_FT<"movz.d", GPR32Opnd, FGR64Opnd, II_MOVZ_D>,
                   CMov_I_F_FM<18, 17>, INSN_MIPS4_32_NOT_32R6_64R6, FGR_64;
  def MOVN_I_D64 : CMov_I_F_FT<"movn.d", GPR32Opnd, FGR64Opnd, II_MOVN_D>,
                   CMov_I_F_FM<19, 17>, INSN_MIPS4_32_NOT_32R6_64R6, FGR_64;
  let isCodeGenOnly = 1 in {
    def MOVZ_I64_D64 : CMov_I_F_FT<"movz.d", GPR64Opnd, FGR64Opnd, II_MOVZ_D>,
                       CMov_I_F_FM<18, 17>, INSN_MIPS4_32_NOT_32R6_64R6, FGR_64;
    def MOVN_I64_D64 : CMov_I_F_FT<"movn.d", GPR64Opnd, FGR64Opnd, II_MOVN_D>,
                       CMov_I_F_FM<19, 17>, INSN_MIPS4_32_NOT_32R6_64R6, FGR_64;
  }
}

def MOVT_I : MMRel, CMov_F_I_FT<"movt", GPR32Opnd, II_MOVT, MipsCMovFP_T>,
             CMov_F_I_FM<1>, INSN_MIPS4_32_NOT_32R6_64R6;

let isCodeGenOnly = 1 in
def MOVT_I64 : CMov_F_I_FT<"movt", GPR64Opnd, II_MOVT, MipsCMovFP_T>,
               CMov_F_I_FM<1>, INSN_MIPS4_32_NOT_32R6_64R6,
               AdditionalRequires<[IsGP64bit]>;

def MOVF_I : MMRel, CMov_F_I_FT<"movf", GPR32Opnd, II_MOVF, MipsCMovFP_F>,
             CMov_F_I_FM<0>, INSN_MIPS4_32_NOT_32R6_64R6;

let isCodeGenOnly = 1 in
def MOVF_I64 : CMov_F_I_FT<"movf", GPR64Opnd, II_MOVF, MipsCMovFP_F>,
               CMov_F_I_FM<0>, INSN_MIPS4_32_NOT_32R6_64R6,
               AdditionalRequires<[IsGP64bit]>;

def MOVT_S : MMRel, CMov_F_F_FT<"movt.s", FGR32Opnd, II_MOVT_S, MipsCMovFP_T>,
             CMov_F_F_FM<16, 1>, INSN_MIPS4_32_NOT_32R6_64R6;
def MOVF_S : MMRel, CMov_F_F_FT<"movf.s", FGR32Opnd, II_MOVF_S, MipsCMovFP_F>,
             CMov_F_F_FM<16, 0>, INSN_MIPS4_32_NOT_32R6_64R6;

def MOVT_D32 : MMRel, CMov_F_F_FT<"movt.d", AFGR64Opnd, II_MOVT_D,
                                  MipsCMovFP_T>, CMov_F_F_FM<17, 1>,
               INSN_MIPS4_32_NOT_32R6_64R6, FGR_32;
def MOVF_D32 : MMRel, CMov_F_F_FT<"movf.d", AFGR64Opnd, II_MOVF_D,
                                  MipsCMovFP_F>, CMov_F_F_FM<17, 0>,
               INSN_MIPS4_32_NOT_32R6_64R6, FGR_32;

let DecoderNamespace = "Mips64" in {
  def MOVT_D64 : CMov_F_F_FT<"movt.d", FGR64Opnd, II_MOVT_D, MipsCMovFP_T>,
                 CMov_F_F_FM<17, 1>, INSN_MIPS4_32_NOT_32R6_64R6, FGR_64;
  def MOVF_D64 : CMov_F_F_FT<"movf.d", FGR64Opnd, II_MOVF_D, MipsCMovFP_F>,
                 CMov_F_F_FM<17, 0>, INSN_MIPS4_32_NOT_32R6_64R6, FGR_64;
}

// select patterns
multiclass MovzPats0<RegisterClass CRC, RegisterClass DRC,
                     Instruction MOVZInst, Instruction SLTOp,
                     Instruction SLTuOp, Instruction SLTiOp,
                     Instruction SLTiuOp, ValueType VT> {
// reg, reg
def : MipsPat<(select (VT (setge CRC:$lhs, CRC:$rhs)), DRC:$T, DRC:$F),
              (MOVZInst DRC:$T, (SLTOp CRC:$lhs, CRC:$rhs), DRC:$F)>;
def : MipsPat<(select (VT (setuge CRC:$lhs, CRC:$rhs)), DRC:$T, DRC:$F),
              (MOVZInst DRC:$T, (SLTuOp CRC:$lhs, CRC:$rhs), DRC:$F)>;
def : MipsPat<(select (VT (setle CRC:$lhs, CRC:$rhs)), DRC:$T, DRC:$F),
              (MOVZInst DRC:$T, (SLTOp CRC:$rhs, CRC:$lhs), DRC:$F)>;
def : MipsPat<(select (VT (setule CRC:$lhs, CRC:$rhs)), DRC:$T, DRC:$F),
              (MOVZInst DRC:$T, (SLTuOp CRC:$rhs, CRC:$lhs), DRC:$F)>;

// reg, imm
def : MipsPat<(select (VT (setge CRC:$lhs, immSExt16:$rhs)), DRC:$T, DRC:$F),
              (MOVZInst DRC:$T, (SLTiOp CRC:$lhs, immSExt16:$rhs), DRC:$F)>;
def : MipsPat<(select (VT (setuge CRC:$lh, immSExt16:$rh)), DRC:$T, DRC:$F),
              (MOVZInst DRC:$T, (SLTiuOp CRC:$lh, immSExt16:$rh), DRC:$F)>;
def : MipsPat<
        (select (VT (setgt CRC:$lhs, immSExt16Plus1:$rhs)), DRC:$T, DRC:$F),
        (MOVZInst DRC:$T, (SLTiOp CRC:$lhs, (Plus1 imm:$rhs)), DRC:$F)>;
def : MipsPat<
        (select (VT (setugt CRC:$lhs, immSExt16Plus1:$rhs)), DRC:$T, DRC:$F),
        (MOVZInst DRC:$T, (SLTiuOp CRC:$lhs, (Plus1 imm:$rhs)), DRC:$F)>;
}

// We have to wrap the instantiation of the MovzPats0 patterns in another
// multiclass that specifies the result type of the SETCC nodes. The patterns
// with VT=i64 (or i32) will be ignored when GPR-width=i32 (or i64).
multiclass MovzPats0_SuperClass<ValueType VT> {
defm : MovzPats0<GPR32, GPR32, MOVZ_I_I, SLT, SLTu, SLTi, SLTiu, VT>,
       INSN_MIPS4_32_NOT_32R6_64R6;
defm : MovzPats0<GPR32, GPR64, MOVZ_I_I64, SLT, SLTu, SLTi, SLTiu, VT>,
       INSN_MIPS4_32_NOT_32R6_64R6, GPR_64;
defm : MovzPats0<GPR32, FGR32, MOVZ_I_S, SLT, SLTu, SLTi, SLTiu, VT>,
       INSN_MIPS4_32_NOT_32R6_64R6;
defm : MovzPats0<GPR32, AFGR64, MOVZ_I_D32, SLT, SLTu, SLTi, SLTiu, VT>,
       INSN_MIPS4_32_NOT_32R6_64R6, FGR_32;
defm : MovzPats0<GPR32, FGR64, MOVZ_I_D64, SLT, SLTu, SLTi, SLTiu, VT>,
       INSN_MIPS4_32_NOT_32R6_64R6, FGR_64;

defm : MovzPats0<GPR32, GPR32, MOVZ_I64_I, SLT64_32, SLTu64_32, SLTi64_32,
                 SLTiu64_32, VT>, INSN_MIPS4_32_NOT_32R6_64R6, GPR_64;
defm : MovzPats0<GPR32, GPR64, MOVZ_I64_I64, SLT64_32, SLTu64_32, SLTi64_32,
                 SLTiu64_32, VT>, INSN_MIPS4_32_NOT_32R6_64R6, GPR_64;
defm : MovzPats0<GPR32, FGR32, MOVZ_I64_S, SLT64_32, SLTu64_32, SLTi64_32,
                 SLTiu64_32, VT>, INSN_MIPS4_32_NOT_32R6_64R6, GPR_64;
defm : MovzPats0<GPR32, FGR64, MOVZ_I64_D64, SLT64_32, SLTu64_32, SLTi64_32,
                 SLTiu64_32, VT>, INSN_MIPS4_32_NOT_32R6_64R6, GPR_64, FGR_64;

defm : MovzPats0<GPR64, GPR32, MOVZ_I64_I, SLT64, SLTu64, SLTi64, SLTiu64, VT>,
       INSN_MIPS4_32_NOT_32R6_64R6, GPR_64;
defm : MovzPats0<GPR64, GPR64, MOVZ_I64_I64, SLT64, SLTu64, SLTi64, SLTiu64, VT>,
       INSN_MIPS4_32_NOT_32R6_64R6, GPR_64;
defm : MovzPats0<GPR64, FGR32, MOVZ_I64_S, SLT64, SLTu64, SLTi64, SLTiu64, VT>,
       INSN_MIPS4_32_NOT_32R6_64R6, GPR_64;
defm : MovzPats0<GPR64, FGR64, MOVZ_I64_D64, SLT64, SLTu64, SLTi64, SLTiu64, VT>,
       INSN_MIPS4_32_NOT_32R6_64R6, GPR_64, FGR_64;
}

defm : MovzPats0_SuperClass<i32>;
defm : MovzPats0_SuperClass<i64>;

multiclass MovzPats1<RegisterClass CRC, RegisterClass DRC,
                     Instruction MOVZInst, Instruction XOROp, ValueType VT> {
def : MipsPat<(select (VT (seteq CRC:$lhs, CRC:$rhs)), DRC:$T, DRC:$F),
              (MOVZInst DRC:$T, (XOROp CRC:$lhs, CRC:$rhs), DRC:$F)>;
def : MipsPat<(select (VT (seteq CRC:$lhs, 0)), DRC:$T, DRC:$F),
              (MOVZInst DRC:$T, CRC:$lhs, DRC:$F)>;
}

multiclass MovzPats1_SuperClass<ValueType VT> {
defm : MovzPats1<GPR32, GPR32, MOVZ_I_I, XOR, VT>, INSN_MIPS4_32_NOT_32R6_64R6;
defm : MovzPats1<GPR32, GPR64, MOVZ_I_I64, XOR, VT>,
       INSN_MIPS4_32_NOT_32R6_64R6, GPR_64;
defm : MovzPats1<GPR32, FGR32, MOVZ_I_S, XOR, VT>, INSN_MIPS4_32_NOT_32R6_64R6;
defm : MovzPats1<GPR32, AFGR64, MOVZ_I_D32, XOR, VT>,
       INSN_MIPS4_32_NOT_32R6_64R6, FGR_32;
defm : MovzPats1<GPR32, FGR64, MOVZ_I_D64, XOR, VT>,
       INSN_MIPS4_32_NOT_32R6_64R6, FGR_64;

defm : MovzPats1<GPR64, GPR32, MOVZ_I64_I, XOR64, VT>,
       INSN_MIPS4_32_NOT_32R6_64R6, GPR_64;
defm : MovzPats1<GPR64, GPR64, MOVZ_I64_I64, XOR64, VT>,
       INSN_MIPS4_32_NOT_32R6_64R6, GPR_64;
defm : MovzPats1<GPR64, FGR32, MOVZ_I64_S, XOR64, VT>,
       INSN_MIPS4_32_NOT_32R6_64R6, GPR_64;
defm : MovzPats1<GPR64, FGR64, MOVZ_I64_D64, XOR64, VT>,
       INSN_MIPS4_32_NOT_32R6_64R6, FGR_64;
}

defm : MovzPats1_SuperClass<i32>;
defm : MovzPats1_SuperClass<i64>;

multiclass MovzPats2<RegisterClass CRC, RegisterClass DRC,
                     Instruction MOVZInst, Instruction XORiOp, ValueType VT> {
def : MipsPat<
    (select (VT (seteq CRC:$lhs, immZExt16:$uimm16)), DRC:$T, DRC:$F),
    (MOVZInst DRC:$T, (XORiOp CRC:$lhs, immZExt16:$uimm16), DRC:$F)>;
}

multiclass MovzPats2_SuperClass<ValueType VT> {
defm : MovzPats2<GPR32, GPR32, MOVZ_I_I, XORi, VT>, INSN_MIPS4_32_NOT_32R6_64R6;
defm : MovzPats2<GPR32, GPR64, MOVZ_I_I64, XORi, VT>,
       INSN_MIPS4_32_NOT_32R6_64R6, GPR_64;
defm : MovzPats2<GPR64, GPR32, MOVZ_I64_I, XORi64, VT>,
       INSN_MIPS4_32_NOT_32R6_64R6, GPR_64;
defm : MovzPats2<GPR64, GPR64, MOVZ_I64_I64, XORi64, VT>,
       INSN_MIPS4_32_NOT_32R6_64R6, GPR_64;
}

defm : MovzPats2_SuperClass<i32>;
defm : MovzPats2_SuperClass<i64>;

multiclass MovnPats<RegisterClass CRC, RegisterClass DRC, Instruction MOVNInst,
                    Instruction XOROp, ValueType VT = i32> {
  def : MipsPat<(select (VT (setne CRC:$lhs, CRC:$rhs)), DRC:$T, DRC:$F),
                (MOVNInst DRC:$T, (XOROp CRC:$lhs, CRC:$rhs), DRC:$F)>;
  def : MipsPat<(select CRC:$cond, DRC:$T, DRC:$F),
                (MOVNInst DRC:$T, CRC:$cond, DRC:$F)>;
  def : MipsPat<(select (VT (setne CRC:$lhs, 0)),DRC:$T, DRC:$F),
                (MOVNInst DRC:$T, CRC:$lhs, DRC:$F)>;
}

defm : MovnPats<GPR32, GPR32, MOVN_I_I, XOR>, INSN_MIPS4_32_NOT_32R6_64R6;
defm : MovnPats<GPR32, FGR32, MOVN_I_S, XOR>, INSN_MIPS4_32_NOT_32R6_64R6;
defm : MovnPats<GPR32, AFGR64, MOVN_I_D32, XOR>, INSN_MIPS4_32_NOT_32R6_64R6,
       FGR_32;
defm : MovnPats<GPR32, FGR64, MOVN_I_D64, XOR>, INSN_MIPS4_32_NOT_32R6_64R6,
       FGR_64;

defm : MovnPats<GPR32, GPR32, MOVN_I_I, XOR, i64>, INSN_MIPS4_32_NOT_32R6_64R6,
       GPR_64;
defm : MovnPats<GPR32, GPR64, MOVN_I_I64, XOR, i64>,
       INSN_MIPS4_32_NOT_32R6_64R6, GPR_64;
defm : MovnPats<GPR32, FGR32, MOVN_I_S, XOR, i64>, INSN_MIPS4_32_NOT_32R6_64R6,
       GPR_64;
defm : MovnPats<GPR32, AFGR64, MOVN_I_D32, XOR, i64>,
       INSN_MIPS4_32_NOT_32R6_64R6, FGR_32;
defm : MovnPats<GPR32, FGR64, MOVN_I_D64, XOR, i64>,
       INSN_MIPS4_32_NOT_32R6_64R6, GPR_64, FGR_64;

defm : MovnPats<GPR64, GPR32, MOVN_I64_I, XOR64, i64>,
       INSN_MIPS4_32_NOT_32R6_64R6, GPR_64;
defm : MovnPats<GPR64, GPR64, MOVN_I64_I64, XOR64, i64>,
       INSN_MIPS4_32_NOT_32R6_64R6, GPR_64;
defm : MovnPats<GPR64, FGR32, MOVN_I64_S, XOR64, i64>,
       INSN_MIPS4_32_NOT_32R6_64R6, GPR_64;
defm : MovnPats<GPR64, FGR64, MOVN_I64_D64, XOR64, i64>,
       INSN_MIPS4_32_NOT_32R6_64R6, GPR_64, FGR_64;

// For targets that don't have conditional-move instructions
// we have to match SELECT nodes with pseudo instructions.
let usesCustomInserter = 1 in {
  class Select_Pseudo<RegisterOperand RC, RegisterOperand RO> :
    PseudoSE<(outs RO:$dst), (ins RC:$cond, RO:$T, RO:$F),
             [(set RO:$dst, (select RC:$cond, RO:$T, RO:$F))]>,
    ISA_MIPS1_NOT_4_32;

  class SelectFP_Pseudo_T<RegisterOperand RO> :
    PseudoSE<(outs RO:$dst), (ins GPR32Opnd:$cond, RO:$T, RO:$F),
             [(set RO:$dst, (MipsCMovFP_T RO:$T, GPR32Opnd:$cond, RO:$F))]>,
    ISA_MIPS1_NOT_4_32;

  class SelectFP_Pseudo_F<RegisterOperand RO> :
    PseudoSE<(outs RO:$dst), (ins GPR32Opnd:$cond, RO:$T, RO:$F),
             [(set RO:$dst, (MipsCMovFP_F RO:$T, GPR32Opnd:$cond, RO:$F))]>,
    ISA_MIPS1_NOT_4_32;
}

def PseudoSELECT_I : Select_Pseudo<GPR32Opnd, GPR32Opnd>;
def PseudoSELECT_I64 : Select_Pseudo<GPR32Opnd, GPR64Opnd>;
def PseudoSELECT_S : Select_Pseudo<GPR32Opnd, FGR32Opnd>;
def PseudoSELECT_D32 : Select_Pseudo<GPR32Opnd, AFGR64Opnd>, FGR_32;
def PseudoSELECT_D64 : Select_Pseudo<GPR32Opnd, FGR64Opnd>, FGR_64;

def PseudoSELECT64_I : Select_Pseudo<GPR64Opnd, GPR32Opnd>, GPR_64;
def PseudoSELECT64_I64 : Select_Pseudo<GPR64Opnd, GPR64Opnd>, GPR_64;
def PseudoSELECT64_S : Select_Pseudo<GPR64Opnd, FGR32Opnd>, GPR_64;
def PseudoSELECT64_D32 : Select_Pseudo<GPR64Opnd, AFGR64Opnd>, GPR_64, FGR_32;
def PseudoSELECT64_D64 : Select_Pseudo<GPR64Opnd, FGR64Opnd>, GPR_64, FGR_64;

def PseudoSELECTFP_T_I : SelectFP_Pseudo_T<GPR32Opnd>;
def PseudoSELECTFP_T_I64 : SelectFP_Pseudo_T<GPR64Opnd>;
def PseudoSELECTFP_T_S : SelectFP_Pseudo_T<FGR32Opnd>;
def PseudoSELECTFP_T_D32 : SelectFP_Pseudo_T<AFGR64Opnd>, FGR_32;
def PseudoSELECTFP_T_D64 : SelectFP_Pseudo_T<FGR64Opnd>, FGR_64;

def PseudoSELECTFP_F_I : SelectFP_Pseudo_F<GPR32Opnd>;
def PseudoSELECTFP_F_I64 : SelectFP_Pseudo_F<GPR64Opnd>;
def PseudoSELECTFP_F_S : SelectFP_Pseudo_F<FGR32Opnd>;
def PseudoSELECTFP_F_D32 : SelectFP_Pseudo_F<AFGR64Opnd>, FGR_32;
def PseudoSELECTFP_F_D64 : SelectFP_Pseudo_F<FGR64Opnd>, FGR_64;

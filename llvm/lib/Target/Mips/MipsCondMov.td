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
class CondMovIntInt<RegisterClass CRC, RegisterClass DRC, bits<6> funct,
                    string instr_asm> :
  FR<0, funct, (outs DRC:$rd), (ins DRC:$rs, CRC:$rt, DRC:$F),
     !strconcat(instr_asm, "\t$rd, $rs, $rt"), [], NoItinerary> {
  let shamt = 0;
  let Constraints = "$F = $rd";
}

// cond:int, data:float
class CondMovIntFP<RegisterClass CRC, RegisterClass DRC, bits<5> fmt,
                   bits<6> func, string instr_asm> :
  FFR<0x11, func, fmt, (outs DRC:$fd), (ins DRC:$fs, CRC:$rt, DRC:$F),
      !strconcat(instr_asm, "\t$fd, $fs, $rt"), []> {
  bits<5> rt;
  let ft = rt;
  let Constraints = "$F = $fd";
}

// cond:float, data:int
class CondMovFPInt<RegisterClass RC, SDNode cmov, bits<1> tf,
                   string instr_asm> :
  FCMOV<tf, (outs RC:$rd), (ins RC:$rs, RC:$F),
        !strconcat(instr_asm, "\t$rd, $rs, $$fcc0"),
        [(set RC:$rd, (cmov RC:$rs, RC:$F))]> {
  let cc = 0;
  let Uses = [FCR31];
  let Constraints = "$F = $rd";
}

// cond:float, data:float
class CondMovFPFP<RegisterClass RC, SDNode cmov, bits<5> fmt, bits<1> tf,
                  string instr_asm> :
  FFCMOV<fmt, tf, (outs RC:$fd), (ins RC:$fs, RC:$F),
         !strconcat(instr_asm, "\t$fd, $fs, $$fcc0"),
         [(set RC:$fd, (cmov RC:$fs, RC:$F))]> {
  let cc = 0;
  let Uses = [FCR31];
  let Constraints = "$F = $fd";
}

// select patterns
multiclass MovzPats0<RegisterClass CRC, RegisterClass DRC,
                     Instruction MOVZInst, Instruction SLTOp,
                     Instruction SLTuOp, Instruction SLTiOp,
                     Instruction SLTiuOp> {
  def : MipsPat<(select (i32 (setge CRC:$lhs, CRC:$rhs)), DRC:$T, DRC:$F),
                (MOVZInst DRC:$T, (SLTOp CRC:$lhs, CRC:$rhs), DRC:$F)>;
  def : MipsPat<
          (select (i32 (setuge CRC:$lhs, CRC:$rhs)), DRC:$T, DRC:$F),
          (MOVZInst DRC:$T, (SLTuOp CRC:$lhs, CRC:$rhs), DRC:$F)>;
  def : MipsPat<
          (select (i32 (setge CRC:$lhs, immSExt16:$rhs)), DRC:$T, DRC:$F),
          (MOVZInst DRC:$T, (SLTiOp CRC:$lhs, immSExt16:$rhs), DRC:$F)>;
  def : MipsPat<
          (select (i32 (setuge CRC:$lh, immSExt16:$rh)), DRC:$T, DRC:$F),
          (MOVZInst DRC:$T, (SLTiuOp CRC:$lh, immSExt16:$rh), DRC:$F)>;
  def : MipsPat<
          (select (i32 (setle CRC:$lhs, CRC:$rhs)), DRC:$T, DRC:$F),
          (MOVZInst DRC:$T, (SLTOp CRC:$rhs, CRC:$lhs), DRC:$F)>;
  def : MipsPat<
          (select (i32 (setule CRC:$lhs, CRC:$rhs)), DRC:$T, DRC:$F),
          (MOVZInst DRC:$T, (SLTuOp CRC:$rhs, CRC:$lhs), DRC:$F)>;
}

multiclass MovzPats1<RegisterClass CRC, RegisterClass DRC,
                     Instruction MOVZInst, Instruction XOROp> {
  def : MipsPat<(select (i32 (seteq CRC:$lhs, CRC:$rhs)), DRC:$T, DRC:$F),
                (MOVZInst DRC:$T, (XOROp CRC:$lhs, CRC:$rhs), DRC:$F)>;
  def : MipsPat<(select (i32 (seteq CRC:$lhs, 0)), DRC:$T, DRC:$F),
                (MOVZInst DRC:$T, CRC:$lhs, DRC:$F)>;
}

multiclass MovzPats2<RegisterClass CRC, RegisterClass DRC,
                     Instruction MOVZInst, Instruction XORiOp> {
  def : MipsPat<
            (select (i32 (seteq CRC:$lhs, immZExt16:$uimm16)), DRC:$T, DRC:$F),
            (MOVZInst DRC:$T, (XORiOp CRC:$lhs, immZExt16:$uimm16), DRC:$F)>;
}

multiclass MovnPats<RegisterClass CRC, RegisterClass DRC, Instruction MOVNInst,
                    Instruction XOROp> {
  def : MipsPat<(select (i32 (setne CRC:$lhs, CRC:$rhs)), DRC:$T, DRC:$F),
                (MOVNInst DRC:$T, (XOROp CRC:$lhs, CRC:$rhs), DRC:$F)>;
  def : MipsPat<(select CRC:$cond, DRC:$T, DRC:$F),
                (MOVNInst DRC:$T, CRC:$cond, DRC:$F)>;
  def : MipsPat<(select (i32 (setne CRC:$lhs, 0)),DRC:$T, DRC:$F),
                (MOVNInst DRC:$T, CRC:$lhs, DRC:$F)>;
}

// Instantiation of instructions.
def MOVZ_I_I     : CondMovIntInt<CPURegs, CPURegs, 0x0a, "movz">;
let Predicates = [HasMips64, HasStandardEncoding],
                  DecoderNamespace = "Mips64" in {
  def MOVZ_I_I64   : CondMovIntInt<CPURegs, CPU64Regs, 0x0a, "movz">;
  def MOVZ_I64_I   : CondMovIntInt<CPU64Regs, CPURegs, 0x0a, "movz"> {
    let isCodeGenOnly = 1;
  }
  def MOVZ_I64_I64 : CondMovIntInt<CPU64Regs, CPU64Regs, 0x0a, "movz"> {
    let isCodeGenOnly = 1;
  }
}

def MOVN_I_I     : CondMovIntInt<CPURegs, CPURegs, 0x0b, "movn">;
let Predicates = [HasMips64, HasStandardEncoding],
                  DecoderNamespace = "Mips64" in {
  def MOVN_I_I64   : CondMovIntInt<CPURegs, CPU64Regs, 0x0b, "movn">;
  def MOVN_I64_I   : CondMovIntInt<CPU64Regs, CPURegs, 0x0b, "movn"> {
    let isCodeGenOnly = 1;
  }
  def MOVN_I64_I64 : CondMovIntInt<CPU64Regs, CPU64Regs, 0x0b, "movn"> {
    let isCodeGenOnly = 1;
  }
}

def MOVZ_I_S   : CondMovIntFP<CPURegs, FGR32, 16, 18, "movz.s">;
def MOVZ_I64_S : CondMovIntFP<CPU64Regs, FGR32, 16, 18, "movz.s">,
                 Requires<[HasMips64, HasStandardEncoding]> {
  let DecoderNamespace = "Mips64";
}

def MOVN_I_S   : CondMovIntFP<CPURegs, FGR32, 16, 19, "movn.s">;
def MOVN_I64_S : CondMovIntFP<CPU64Regs, FGR32, 16, 19, "movn.s">,
                 Requires<[HasMips64, HasStandardEncoding]> {
  let DecoderNamespace = "Mips64";
}

let Predicates = [NotFP64bit, HasStandardEncoding] in {
  def MOVZ_I_D32   : CondMovIntFP<CPURegs, AFGR64, 17, 18, "movz.d">;
  def MOVN_I_D32   : CondMovIntFP<CPURegs, AFGR64, 17, 19, "movn.d">;
}
let Predicates = [IsFP64bit, HasStandardEncoding],
                  DecoderNamespace = "Mips64" in {
  def MOVZ_I_D64   : CondMovIntFP<CPURegs, FGR64, 17, 18, "movz.d">;
  def MOVZ_I64_D64 : CondMovIntFP<CPU64Regs, FGR64, 17, 18, "movz.d"> {
    let isCodeGenOnly = 1;
  }
  def MOVN_I_D64   : CondMovIntFP<CPURegs, FGR64, 17, 19, "movn.d">;
  def MOVN_I64_D64 : CondMovIntFP<CPU64Regs, FGR64, 17, 19, "movn.d"> {
    let isCodeGenOnly = 1;
  }
}

def MOVT_I   : CondMovFPInt<CPURegs, MipsCMovFP_T, 1, "movt">;
def MOVT_I64 : CondMovFPInt<CPU64Regs, MipsCMovFP_T, 1, "movt">,
               Requires<[HasMips64, HasStandardEncoding]> {
  let DecoderNamespace = "Mips64";
}

def MOVF_I   : CondMovFPInt<CPURegs, MipsCMovFP_F, 0, "movf">;
def MOVF_I64 : CondMovFPInt<CPU64Regs, MipsCMovFP_F, 0, "movf">,
               Requires<[HasMips64, HasStandardEncoding]> {
  let DecoderNamespace = "Mips64";
}

def MOVT_S : CondMovFPFP<FGR32, MipsCMovFP_T, 16, 1, "movt.s">;
def MOVF_S : CondMovFPFP<FGR32, MipsCMovFP_F, 16, 0, "movf.s">;

let Predicates = [NotFP64bit, HasStandardEncoding] in {
  def MOVT_D32 : CondMovFPFP<AFGR64, MipsCMovFP_T, 17, 1, "movt.d">;
  def MOVF_D32 : CondMovFPFP<AFGR64, MipsCMovFP_F, 17, 0, "movf.d">;
}
let Predicates = [IsFP64bit, HasStandardEncoding],
    DecoderNamespace = "Mips64" in {
  def MOVT_D64 : CondMovFPFP<FGR64, MipsCMovFP_T, 17, 1, "movt.d">;
  def MOVF_D64 : CondMovFPFP<FGR64, MipsCMovFP_F, 17, 0, "movf.d">;
}

// Instantiation of conditional move patterns.
defm : MovzPats0<CPURegs, CPURegs, MOVZ_I_I, SLT, SLTu, SLTi, SLTiu>;
defm : MovzPats1<CPURegs, CPURegs, MOVZ_I_I, XOR>;
defm : MovzPats2<CPURegs, CPURegs, MOVZ_I_I, XORi>;
let Predicates = [HasMips64, HasStandardEncoding] in {
  defm : MovzPats0<CPURegs, CPU64Regs, MOVZ_I_I64, SLT, SLTu, SLTi, SLTiu>;
  defm : MovzPats0<CPU64Regs, CPURegs, MOVZ_I_I, SLT64, SLTu64, SLTi64,
                   SLTiu64>;
  defm : MovzPats0<CPU64Regs, CPU64Regs, MOVZ_I_I64, SLT64, SLTu64, SLTi64,
                   SLTiu64>;
  defm : MovzPats1<CPURegs, CPU64Regs, MOVZ_I_I64, XOR>;
  defm : MovzPats1<CPU64Regs, CPURegs, MOVZ_I64_I, XOR64>;
  defm : MovzPats1<CPU64Regs, CPU64Regs, MOVZ_I64_I64, XOR64>;
  defm : MovzPats2<CPURegs, CPU64Regs, MOVZ_I_I64, XORi>;
  defm : MovzPats2<CPU64Regs, CPURegs, MOVZ_I64_I, XORi64>;
  defm : MovzPats2<CPU64Regs, CPU64Regs, MOVZ_I64_I64, XORi64>;
}

defm : MovnPats<CPURegs, CPURegs, MOVN_I_I, XOR>;
let Predicates = [HasMips64, HasStandardEncoding] in {
  defm : MovnPats<CPURegs, CPU64Regs, MOVN_I_I64, XOR>;
  defm : MovnPats<CPU64Regs, CPURegs, MOVN_I64_I, XOR64>;
  defm : MovnPats<CPU64Regs, CPU64Regs, MOVN_I64_I64, XOR64>;
}

defm : MovzPats0<CPURegs, FGR32, MOVZ_I_S, SLT, SLTu, SLTi, SLTiu>;
defm : MovzPats1<CPURegs, FGR32, MOVZ_I_S, XOR>;
defm : MovnPats<CPURegs, FGR32, MOVN_I_S, XOR>;
let Predicates = [HasMips64, HasStandardEncoding] in {
  defm : MovzPats0<CPU64Regs, FGR32, MOVZ_I_S, SLT64, SLTu64, SLTi64,
                   SLTiu64>;
  defm : MovzPats1<CPU64Regs, FGR32, MOVZ_I64_S, XOR64>;
  defm : MovnPats<CPU64Regs, FGR32, MOVN_I64_S, XOR64>;
}

let Predicates = [NotFP64bit, HasStandardEncoding] in {
  defm : MovzPats0<CPURegs, AFGR64, MOVZ_I_D32, SLT, SLTu, SLTi, SLTiu>;
  defm : MovzPats1<CPURegs, AFGR64, MOVZ_I_D32, XOR>;
  defm : MovnPats<CPURegs, AFGR64, MOVN_I_D32, XOR>;
}
let Predicates = [IsFP64bit, HasStandardEncoding] in {
  defm : MovzPats0<CPURegs, FGR64, MOVZ_I_D64, SLT, SLTu, SLTi, SLTiu>;
  defm : MovzPats0<CPU64Regs, FGR64, MOVZ_I_D64, SLT64, SLTu64, SLTi64,
                   SLTiu64>;
  defm : MovzPats1<CPURegs, FGR64, MOVZ_I_D64, XOR>;
  defm : MovzPats1<CPU64Regs, FGR64, MOVZ_I64_D64, XOR64>;
  defm : MovnPats<CPURegs, FGR64, MOVN_I_D64, XOR>;
  defm : MovnPats<CPU64Regs, FGR64, MOVN_I64_D64, XOR64>;
}

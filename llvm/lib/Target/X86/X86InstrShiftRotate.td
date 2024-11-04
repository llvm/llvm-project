//===-- X86InstrShiftRotate.td - Shift and Rotate Instrs ---*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file describes the shift and rotate instructions.
//
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// Shift/Rotate instructions
//===----------------------------------------------------------------------===//

multiclass ShiftRotate<string m, Format RegMRM, Format MemMRM, SDPatternOperator node,
                       SchedReadWrite rCL, SchedReadWrite ri, SchedReadWrite mCL,
                       SchedReadWrite mi, list<Register> uses = []> {

  let Uses = uses in {
    let isConvertibleToThreeAddress = !if(!eq(m, "shl"), 1, 0) in {
      def 8ri  : BinOpRI8U_R<m, RegMRM, Xi8, node>, Sched<[ri]>, DefEFLAGS;
      def 16ri : BinOpRI8U_R<m, RegMRM, Xi16, node>, Sched<[ri]>, DefEFLAGS, OpSize16;
      def 32ri : BinOpRI8U_R<m, RegMRM, Xi32, node>, Sched<[ri]>, DefEFLAGS, OpSize32;
      def 64ri : BinOpRI8U_R<m, RegMRM, Xi64, node>, Sched<[ri]>, DefEFLAGS;
    }

    def 8mi  : BinOpMI8U_M<m, MemMRM, Xi8, node>, Sched<[mi, WriteRMW]>, DefEFLAGS;
    def 16mi : BinOpMI8U_M<m, MemMRM, Xi16, node>, Sched<[mi, WriteRMW]>, DefEFLAGS, OpSize16;
    def 32mi : BinOpMI8U_M<m, MemMRM, Xi32, node>, Sched<[mi, WriteRMW]>, DefEFLAGS, OpSize32;
    def 64mi : BinOpMI8U_M<m, MemMRM, Xi64, node>, Sched<[mi, WriteRMW]>, DefEFLAGS, Requires<[In64BitMode]>;

    let SchedRW = [ri] in {
      def 8r1  : UnaryOpR_RF<0xD1, RegMRM, m, Xi8, null_frag>;
      def 16r1 : UnaryOpR_RF<0xD1, RegMRM, m, Xi16, null_frag>, OpSize16;
      def 32r1 : UnaryOpR_RF<0xD1, RegMRM, m, Xi32, null_frag>, OpSize32;
      def 64r1 : UnaryOpR_RF<0xD1, RegMRM, m, Xi64, null_frag>;
    }

    let SchedRW = [mi, WriteRMW] in {
      def 8m1  : UnaryOpM_MF<0xD1, MemMRM, m, Xi8, null_frag>;
      def 16m1 : UnaryOpM_MF<0xD1, MemMRM, m, Xi16, null_frag>, OpSize16;
      def 32m1 : UnaryOpM_MF<0xD1, MemMRM, m, Xi32, null_frag>, OpSize32;
      def 64m1 : UnaryOpM_MF<0xD1, MemMRM, m, Xi64, null_frag>, Requires<[In64BitMode]>;
    }
  }

  let Uses = !listconcat([CL], uses) in {
    def 8rCL  : BinOpRC_R<m, RegMRM, Xi8, node>, Sched<[rCL]>, DefEFLAGS;
    def 16rCL : BinOpRC_R<m, RegMRM, Xi16, node>, Sched<[rCL]>, DefEFLAGS, OpSize16;
    def 32rCL : BinOpRC_R<m, RegMRM, Xi32, node>, Sched<[rCL]>, DefEFLAGS, OpSize32;
    def 64rCL : BinOpRC_R<m, RegMRM, Xi64, node>, Sched<[rCL]>, DefEFLAGS;

    def 8mCL  : BinOpMC_M<m, MemMRM, Xi8, node>, Sched<[mCL, WriteRMW]>, DefEFLAGS;
    def 16mCL : BinOpMC_M<m, MemMRM, Xi16, node>, Sched<[mCL, WriteRMW]>, DefEFLAGS, OpSize16;
    def 32mCL : BinOpMC_M<m, MemMRM, Xi32, node>, Sched<[mCL, WriteRMW]>, DefEFLAGS, OpSize32;
    def 64mCL : BinOpMC_M<m, MemMRM, Xi64, node>, Sched<[mCL, WriteRMW]>, DefEFLAGS, Requires<[In64BitMode]>;
  }
}

defm SHL: ShiftRotate<"shl", MRM4r, MRM4m, shl, WriteShiftCL, WriteShift, WriteShiftCLLd, WriteShiftLd>;
defm SHR: ShiftRotate<"shr", MRM5r, MRM5m, srl, WriteShiftCL, WriteShift, WriteShiftCLLd, WriteShiftLd>;
defm SAR: ShiftRotate<"sar", MRM7r, MRM7m, sra, WriteShiftCL, WriteShift, WriteShiftCLLd, WriteShiftLd>;

defm ROL: ShiftRotate<"rol", MRM0r, MRM0m, rotl, WriteRotateCL, WriteRotate, WriteRotateCLLd, WriteRotateLd>;
defm ROR: ShiftRotate<"ror", MRM1r, MRM1m, rotr, WriteRotateCL, WriteRotate, WriteRotateCLLd, WriteRotateLd>;
defm RCL: ShiftRotate<"rcl", MRM2r, MRM2m, null_frag, WriteRotateCL, WriteRotate, WriteRotateCLLd, WriteRotateLd, [EFLAGS]>;
defm RCR: ShiftRotate<"rcr", MRM3r, MRM3m, null_frag, WriteRotateCL, WriteRotate, WriteRotateCLLd, WriteRotateLd, [EFLAGS]>;

//===----------------------------------------------------------------------===//
// Double shift instructions (generalizations of rotate)
//===----------------------------------------------------------------------===//

let Defs = [EFLAGS], hasSideEffects = 0 in {
let Constraints = "$src1 = $dst" in {

let Uses = [CL], SchedRW = [WriteSHDrrcl] in {
def SHLD16rrCL : I<0xA5, MRMDestReg, (outs GR16:$dst),
                   (ins GR16:$src1, GR16:$src2),
                   "shld{w}\t{%cl, $src2, $dst|$dst, $src2, cl}",
                   [(set GR16:$dst, (X86fshl GR16:$src1, GR16:$src2, CL))]>,
                   TB, OpSize16;
def SHRD16rrCL : I<0xAD, MRMDestReg, (outs GR16:$dst),
                   (ins GR16:$src1, GR16:$src2),
                   "shrd{w}\t{%cl, $src2, $dst|$dst, $src2, cl}",
                   [(set GR16:$dst, (X86fshr GR16:$src2, GR16:$src1, CL))]>,
                   TB, OpSize16;
def SHLD32rrCL : I<0xA5, MRMDestReg, (outs GR32:$dst),
                   (ins GR32:$src1, GR32:$src2),
                   "shld{l}\t{%cl, $src2, $dst|$dst, $src2, cl}",
                   [(set GR32:$dst, (fshl GR32:$src1, GR32:$src2, CL))]>,
                   TB, OpSize32;
def SHRD32rrCL : I<0xAD, MRMDestReg, (outs GR32:$dst),
                   (ins GR32:$src1, GR32:$src2),
                   "shrd{l}\t{%cl, $src2, $dst|$dst, $src2, cl}",
                   [(set GR32:$dst, (fshr GR32:$src2, GR32:$src1, CL))]>,
                   TB, OpSize32;
def SHLD64rrCL : RI<0xA5, MRMDestReg, (outs GR64:$dst),
                    (ins GR64:$src1, GR64:$src2),
                    "shld{q}\t{%cl, $src2, $dst|$dst, $src2, cl}",
                    [(set GR64:$dst, (fshl GR64:$src1, GR64:$src2, CL))]>,
                    TB;
def SHRD64rrCL : RI<0xAD, MRMDestReg, (outs GR64:$dst),
                    (ins GR64:$src1, GR64:$src2),
                    "shrd{q}\t{%cl, $src2, $dst|$dst, $src2, cl}",
                    [(set GR64:$dst, (fshr GR64:$src2, GR64:$src1, CL))]>,
                    TB;
} // Uses, SchedRW

let isCommutable = 1, SchedRW = [WriteSHDrri] in {  // These instructions commute to each other.
def SHLD16rri8 : Ii8<0xA4, MRMDestReg,
                     (outs GR16:$dst),
                     (ins GR16:$src1, GR16:$src2, u8imm:$src3),
                     "shld{w}\t{$src3, $src2, $dst|$dst, $src2, $src3}",
                     [(set GR16:$dst, (X86fshl GR16:$src1, GR16:$src2,
                                      (i8 imm:$src3)))]>,
                     TB, OpSize16;
def SHRD16rri8 : Ii8<0xAC, MRMDestReg,
                     (outs GR16:$dst),
                     (ins GR16:$src1, GR16:$src2, u8imm:$src3),
                     "shrd{w}\t{$src3, $src2, $dst|$dst, $src2, $src3}",
                     [(set GR16:$dst, (X86fshr GR16:$src2, GR16:$src1,
                                      (i8 imm:$src3)))]>,
                     TB, OpSize16;
def SHLD32rri8 : Ii8<0xA4, MRMDestReg,
                     (outs GR32:$dst),
                     (ins GR32:$src1, GR32:$src2, u8imm:$src3),
                     "shld{l}\t{$src3, $src2, $dst|$dst, $src2, $src3}",
                     [(set GR32:$dst, (fshl GR32:$src1, GR32:$src2,
                                      (i8 imm:$src3)))]>,
                 TB, OpSize32;
def SHRD32rri8 : Ii8<0xAC, MRMDestReg,
                     (outs GR32:$dst),
                     (ins GR32:$src1, GR32:$src2, u8imm:$src3),
                     "shrd{l}\t{$src3, $src2, $dst|$dst, $src2, $src3}",
                     [(set GR32:$dst, (fshr GR32:$src2, GR32:$src1,
                                      (i8 imm:$src3)))]>,
                 TB, OpSize32;
def SHLD64rri8 : RIi8<0xA4, MRMDestReg,
                      (outs GR64:$dst),
                      (ins GR64:$src1, GR64:$src2, u8imm:$src3),
                      "shld{q}\t{$src3, $src2, $dst|$dst, $src2, $src3}",
                      [(set GR64:$dst, (fshl GR64:$src1, GR64:$src2,
                                       (i8 imm:$src3)))]>,
                 TB;
def SHRD64rri8 : RIi8<0xAC, MRMDestReg,
                      (outs GR64:$dst),
                      (ins GR64:$src1, GR64:$src2, u8imm:$src3),
                      "shrd{q}\t{$src3, $src2, $dst|$dst, $src2, $src3}",
                      [(set GR64:$dst, (fshr GR64:$src2, GR64:$src1,
                                       (i8 imm:$src3)))]>,
                 TB;
} // SchedRW
} // Constraints = "$src = $dst"

let Uses = [CL], SchedRW = [WriteSHDmrcl] in {
def SHLD16mrCL : I<0xA5, MRMDestMem, (outs), (ins i16mem:$dst, GR16:$src2),
                   "shld{w}\t{%cl, $src2, $dst|$dst, $src2, cl}",
                   [(store (X86fshl (loadi16 addr:$dst), GR16:$src2, CL),
                                    addr:$dst)]>, TB, OpSize16;
def SHRD16mrCL : I<0xAD, MRMDestMem, (outs), (ins i16mem:$dst, GR16:$src2),
                  "shrd{w}\t{%cl, $src2, $dst|$dst, $src2, cl}",
                  [(store (X86fshr GR16:$src2, (loadi16 addr:$dst), CL),
                                   addr:$dst)]>, TB, OpSize16;

def SHLD32mrCL : I<0xA5, MRMDestMem, (outs), (ins i32mem:$dst, GR32:$src2),
                   "shld{l}\t{%cl, $src2, $dst|$dst, $src2, cl}",
                   [(store (fshl (loadi32 addr:$dst), GR32:$src2, CL),
                     addr:$dst)]>, TB, OpSize32;
def SHRD32mrCL : I<0xAD, MRMDestMem, (outs), (ins i32mem:$dst, GR32:$src2),
                  "shrd{l}\t{%cl, $src2, $dst|$dst, $src2, cl}",
                  [(store (fshr GR32:$src2, (loadi32 addr:$dst), CL),
                                addr:$dst)]>, TB, OpSize32;

def SHLD64mrCL : RI<0xA5, MRMDestMem, (outs), (ins i64mem:$dst, GR64:$src2),
                    "shld{q}\t{%cl, $src2, $dst|$dst, $src2, cl}",
                    [(store (fshl (loadi64 addr:$dst), GR64:$src2, CL),
                                  addr:$dst)]>, TB;
def SHRD64mrCL : RI<0xAD, MRMDestMem, (outs), (ins i64mem:$dst, GR64:$src2),
                    "shrd{q}\t{%cl, $src2, $dst|$dst, $src2, cl}",
                    [(store (fshr GR64:$src2, (loadi64 addr:$dst), CL),
                                  addr:$dst)]>, TB;
} // Uses, SchedRW

let SchedRW = [WriteSHDmri] in {
def SHLD16mri8 : Ii8<0xA4, MRMDestMem,
                    (outs), (ins i16mem:$dst, GR16:$src2, u8imm:$src3),
                    "shld{w}\t{$src3, $src2, $dst|$dst, $src2, $src3}",
                    [(store (X86fshl (loadi16 addr:$dst), GR16:$src2,
                                     (i8 imm:$src3)), addr:$dst)]>,
                    TB, OpSize16;
def SHRD16mri8 : Ii8<0xAC, MRMDestMem,
                     (outs), (ins i16mem:$dst, GR16:$src2, u8imm:$src3),
                     "shrd{w}\t{$src3, $src2, $dst|$dst, $src2, $src3}",
                    [(store (X86fshr GR16:$src2, (loadi16 addr:$dst),
                                     (i8 imm:$src3)), addr:$dst)]>,
                     TB, OpSize16;

def SHLD32mri8 : Ii8<0xA4, MRMDestMem,
                    (outs), (ins i32mem:$dst, GR32:$src2, u8imm:$src3),
                    "shld{l}\t{$src3, $src2, $dst|$dst, $src2, $src3}",
                    [(store (fshl (loadi32 addr:$dst), GR32:$src2,
                                  (i8 imm:$src3)), addr:$dst)]>,
                    TB, OpSize32;
def SHRD32mri8 : Ii8<0xAC, MRMDestMem,
                     (outs), (ins i32mem:$dst, GR32:$src2, u8imm:$src3),
                     "shrd{l}\t{$src3, $src2, $dst|$dst, $src2, $src3}",
                     [(store (fshr GR32:$src2, (loadi32 addr:$dst),
                                   (i8 imm:$src3)), addr:$dst)]>,
                     TB, OpSize32;

def SHLD64mri8 : RIi8<0xA4, MRMDestMem,
                      (outs), (ins i64mem:$dst, GR64:$src2, u8imm:$src3),
                      "shld{q}\t{$src3, $src2, $dst|$dst, $src2, $src3}",
                      [(store (fshl (loadi64 addr:$dst), GR64:$src2,
                                    (i8 imm:$src3)), addr:$dst)]>,
                 TB;
def SHRD64mri8 : RIi8<0xAC, MRMDestMem,
                      (outs), (ins i64mem:$dst, GR64:$src2, u8imm:$src3),
                      "shrd{q}\t{$src3, $src2, $dst|$dst, $src2, $src3}",
                      [(store (fshr GR64:$src2, (loadi64 addr:$dst),
                                    (i8 imm:$src3)), addr:$dst)]>,
                 TB;
} // SchedRW

} // Defs = [EFLAGS], hasSideEffects

// Use the opposite rotate if allows us to use the rotate by 1 instruction.
def : Pat<(rotl GR8:$src1,  (i8 7)),  (ROR8r1  GR8:$src1)>;
def : Pat<(rotl GR16:$src1, (i8 15)), (ROR16r1 GR16:$src1)>;
def : Pat<(rotl GR32:$src1, (i8 31)), (ROR32r1 GR32:$src1)>;
def : Pat<(rotl GR64:$src1, (i8 63)), (ROR64r1 GR64:$src1)>;
def : Pat<(rotr GR8:$src1,  (i8 7)),  (ROL8r1  GR8:$src1)>;
def : Pat<(rotr GR16:$src1, (i8 15)), (ROL16r1 GR16:$src1)>;
def : Pat<(rotr GR32:$src1, (i8 31)), (ROL32r1 GR32:$src1)>;
def : Pat<(rotr GR64:$src1, (i8 63)), (ROL64r1 GR64:$src1)>;

def : Pat<(store (rotl (loadi8 addr:$dst), (i8 7)), addr:$dst),
          (ROR8m1 addr:$dst)>;
def : Pat<(store (rotl (loadi16 addr:$dst), (i8 15)), addr:$dst),
          (ROR16m1 addr:$dst)>;
def : Pat<(store (rotl (loadi32 addr:$dst), (i8 31)), addr:$dst),
          (ROR32m1 addr:$dst)>;
def : Pat<(store (rotl (loadi64 addr:$dst), (i8 63)), addr:$dst),
          (ROR64m1 addr:$dst)>, Requires<[In64BitMode]>;

def : Pat<(store (rotr (loadi8 addr:$dst), (i8 7)), addr:$dst),
          (ROL8m1 addr:$dst)>;
def : Pat<(store (rotr (loadi16 addr:$dst), (i8 15)), addr:$dst),
          (ROL16m1 addr:$dst)>;
def : Pat<(store (rotr (loadi32 addr:$dst), (i8 31)), addr:$dst),
          (ROL32m1 addr:$dst)>;
def : Pat<(store (rotr (loadi64 addr:$dst), (i8 63)), addr:$dst),
          (ROL64m1 addr:$dst)>, Requires<[In64BitMode]>;

// Sandy Bridge and newer Intel processors support faster rotates using
// SHLD to avoid a partial flag update on the normal rotate instructions.
// Use a pseudo so that TwoInstructionPass and register allocation will see
// this as unary instruction.
let Predicates = [HasFastSHLDRotate], AddedComplexity = 5,
    Defs = [EFLAGS], isPseudo = 1, SchedRW = [WriteSHDrri],
    Constraints = "$src1 = $dst" in {
  def SHLDROT32ri  : I<0, Pseudo, (outs GR32:$dst),
                       (ins GR32:$src1, u8imm:$shamt), "",
                     [(set GR32:$dst, (rotl GR32:$src1, (i8 imm:$shamt)))]>;
  def SHLDROT64ri  : I<0, Pseudo, (outs GR64:$dst),
                       (ins GR64:$src1, u8imm:$shamt), "",
                     [(set GR64:$dst, (rotl GR64:$src1, (i8 imm:$shamt)))]>;

  def SHRDROT32ri  : I<0, Pseudo, (outs GR32:$dst),
                       (ins GR32:$src1, u8imm:$shamt), "",
                     [(set GR32:$dst, (rotr GR32:$src1, (i8 imm:$shamt)))]>;
  def SHRDROT64ri  : I<0, Pseudo, (outs GR64:$dst),
                       (ins GR64:$src1, u8imm:$shamt), "",
                     [(set GR64:$dst, (rotr GR64:$src1, (i8 imm:$shamt)))]>;
}

def ROT32L2R_imm8  : SDNodeXForm<imm, [{
  // Convert a ROTL shamt to a ROTR shamt on 32-bit integer.
  return getI8Imm(32 - N->getZExtValue(), SDLoc(N));
}]>;

def ROT64L2R_imm8  : SDNodeXForm<imm, [{
  // Convert a ROTL shamt to a ROTR shamt on 64-bit integer.
  return getI8Imm(64 - N->getZExtValue(), SDLoc(N));
}]>;

// NOTE: We use WriteShift for these rotates as they avoid the stalls
// of many of the older x86 rotate instructions.
multiclass bmi_rotate<string asm, RegisterClass RC, X86MemOperand x86memop,
                      string Suffix = ""> {
let hasSideEffects = 0 in {
  def ri#Suffix : Ii8<0xF0, MRMSrcReg, (outs RC:$dst), (ins RC:$src1, u8imm:$src2),
                      !strconcat(asm, "\t{$src2, $src1, $dst|$dst, $src1, $src2}"), []>,
                  TA, XD, VEX, Sched<[WriteShift]>;
  let mayLoad = 1 in
  def mi#Suffix : Ii8<0xF0, MRMSrcMem, (outs RC:$dst),
                      (ins x86memop:$src1, u8imm:$src2),
                      !strconcat(asm, "\t{$src2, $src1, $dst|$dst, $src1, $src2}"), []>,
                  TA, XD, VEX, Sched<[WriteShiftLd]>;
}
}

multiclass bmi_shift<string asm, RegisterClass RC, X86MemOperand x86memop,
                     string Suffix = ""> {
let hasSideEffects = 0 in {
  def rr#Suffix : I<0xF7, MRMSrcReg4VOp3, (outs RC:$dst), (ins RC:$src1, RC:$src2),
                    !strconcat(asm, "\t{$src2, $src1, $dst|$dst, $src1, $src2}"), []>,
                    VEX, Sched<[WriteShift]>;
  let mayLoad = 1 in
  def rm#Suffix : I<0xF7, MRMSrcMem4VOp3,
                    (outs RC:$dst), (ins x86memop:$src1, RC:$src2),
                    !strconcat(asm, "\t{$src2, $src1, $dst|$dst, $src1, $src2}"), []>,
                  VEX, Sched<[WriteShift.Folded,
                              // x86memop:$src1
                              ReadDefault, ReadDefault, ReadDefault, ReadDefault,
                              ReadDefault,
                              // RC:$src2
                              WriteShift.ReadAfterFold]>;
}
}

let Predicates = [HasBMI2, NoEGPR] in {
  defm RORX32 : bmi_rotate<"rorx{l}", GR32, i32mem>;
  defm RORX64 : bmi_rotate<"rorx{q}", GR64, i64mem>, REX_W;
  defm SARX32 : bmi_shift<"sarx{l}", GR32, i32mem>, T8, XS;
  defm SARX64 : bmi_shift<"sarx{q}", GR64, i64mem>, T8, XS, REX_W;
  defm SHRX32 : bmi_shift<"shrx{l}", GR32, i32mem>, T8, XD;
  defm SHRX64 : bmi_shift<"shrx{q}", GR64, i64mem>, T8, XD, REX_W;
  defm SHLX32 : bmi_shift<"shlx{l}", GR32, i32mem>, T8, PD;
  defm SHLX64 : bmi_shift<"shlx{q}", GR64, i64mem>, T8, PD, REX_W;
}

let Predicates = [HasBMI2, HasEGPR, In64BitMode] in {
  defm RORX32 : bmi_rotate<"rorx{l}", GR32, i32mem, "_EVEX">, EVEX;
  defm RORX64 : bmi_rotate<"rorx{q}", GR64, i64mem, "_EVEX">, REX_W, EVEX;
  defm SARX32 : bmi_shift<"sarx{l}", GR32, i32mem, "_EVEX">, T8, XS, EVEX;
  defm SARX64 : bmi_shift<"sarx{q}", GR64, i64mem, "_EVEX">, T8, XS, REX_W, EVEX;
  defm SHRX32 : bmi_shift<"shrx{l}", GR32, i32mem, "_EVEX">, T8, XD, EVEX;
  defm SHRX64 : bmi_shift<"shrx{q}", GR64, i64mem, "_EVEX">, T8, XD, REX_W, EVEX;
  defm SHLX32 : bmi_shift<"shlx{l}", GR32, i32mem, "_EVEX">, T8, PD, EVEX;
  defm SHLX64 : bmi_shift<"shlx{q}", GR64, i64mem, "_EVEX">, T8, PD, REX_W, EVEX;
}

let Predicates = [HasBMI2] in {
  // Prefer RORX which is non-destructive and doesn't update EFLAGS.
  let AddedComplexity = 10 in {
    def : Pat<(rotr GR32:$src, (i8 imm:$shamt)),
              (RORX32ri GR32:$src, imm:$shamt)>;
    def : Pat<(rotr GR64:$src, (i8 imm:$shamt)),
              (RORX64ri GR64:$src, imm:$shamt)>;

    def : Pat<(rotl GR32:$src, (i8 imm:$shamt)),
              (RORX32ri GR32:$src, (ROT32L2R_imm8 imm:$shamt))>;
    def : Pat<(rotl GR64:$src, (i8 imm:$shamt)),
              (RORX64ri GR64:$src, (ROT64L2R_imm8 imm:$shamt))>;
  }

  def : Pat<(rotr (loadi32 addr:$src), (i8 imm:$shamt)),
            (RORX32mi addr:$src, imm:$shamt)>;
  def : Pat<(rotr (loadi64 addr:$src), (i8 imm:$shamt)),
            (RORX64mi addr:$src, imm:$shamt)>;

  def : Pat<(rotl (loadi32 addr:$src), (i8 imm:$shamt)),
            (RORX32mi addr:$src, (ROT32L2R_imm8 imm:$shamt))>;
  def : Pat<(rotl (loadi64 addr:$src), (i8 imm:$shamt)),
            (RORX64mi addr:$src, (ROT64L2R_imm8 imm:$shamt))>;

  // Prefer SARX/SHRX/SHLX over SAR/SHR/SHL with variable shift BUT not
  // immediate shift, i.e. the following code is considered better
  //
  //  mov %edi, %esi
  //  shl $imm, %esi
  //  ... %edi, ...
  //
  // than
  //
  //  movb $imm, %sil
  //  shlx %sil, %edi, %esi
  //  ... %edi, ...
  //
  let AddedComplexity = 1 in {
    def : Pat<(sra GR32:$src1, GR8:$src2),
              (SARX32rr GR32:$src1,
                        (INSERT_SUBREG
                          (i32 (IMPLICIT_DEF)), GR8:$src2, sub_8bit))>;
    def : Pat<(sra GR64:$src1, GR8:$src2),
              (SARX64rr GR64:$src1,
                        (INSERT_SUBREG
                          (i64 (IMPLICIT_DEF)), GR8:$src2, sub_8bit))>;

    def : Pat<(srl GR32:$src1, GR8:$src2),
              (SHRX32rr GR32:$src1,
                        (INSERT_SUBREG
                          (i32 (IMPLICIT_DEF)), GR8:$src2, sub_8bit))>;
    def : Pat<(srl GR64:$src1, GR8:$src2),
              (SHRX64rr GR64:$src1,
                        (INSERT_SUBREG
                          (i64 (IMPLICIT_DEF)), GR8:$src2, sub_8bit))>;

    def : Pat<(shl GR32:$src1, GR8:$src2),
              (SHLX32rr GR32:$src1,
                        (INSERT_SUBREG
                          (i32 (IMPLICIT_DEF)), GR8:$src2, sub_8bit))>;
    def : Pat<(shl GR64:$src1, GR8:$src2),
              (SHLX64rr GR64:$src1,
                        (INSERT_SUBREG
                          (i64 (IMPLICIT_DEF)), GR8:$src2, sub_8bit))>;
  }

  // We prefer to use
  //  mov (%ecx), %esi
  //  shl $imm, $esi
  //
  // over
  //
  //  movb $imm, %al
  //  shlx %al, (%ecx), %esi
  //
  // This priority is enforced by IsProfitableToFoldLoad.
  def : Pat<(sra (loadi32 addr:$src1), GR8:$src2),
            (SARX32rm addr:$src1,
                      (INSERT_SUBREG
                        (i32 (IMPLICIT_DEF)), GR8:$src2, sub_8bit))>;
  def : Pat<(sra (loadi64 addr:$src1), GR8:$src2),
            (SARX64rm addr:$src1,
                      (INSERT_SUBREG
                        (i64 (IMPLICIT_DEF)), GR8:$src2, sub_8bit))>;

  def : Pat<(srl (loadi32 addr:$src1), GR8:$src2),
            (SHRX32rm addr:$src1,
                      (INSERT_SUBREG
                        (i32 (IMPLICIT_DEF)), GR8:$src2, sub_8bit))>;
  def : Pat<(srl (loadi64 addr:$src1), GR8:$src2),
            (SHRX64rm addr:$src1,
                      (INSERT_SUBREG
                        (i64 (IMPLICIT_DEF)), GR8:$src2, sub_8bit))>;

  def : Pat<(shl (loadi32 addr:$src1), GR8:$src2),
            (SHLX32rm addr:$src1,
                      (INSERT_SUBREG
                        (i32 (IMPLICIT_DEF)), GR8:$src2, sub_8bit))>;
  def : Pat<(shl (loadi64 addr:$src1), GR8:$src2),
            (SHLX64rm addr:$src1,
                      (INSERT_SUBREG
                        (i64 (IMPLICIT_DEF)), GR8:$src2, sub_8bit))>;
}

def : Pat<(rotl GR8:$src1, (i8 relocImm:$src2)),
          (ROL8ri GR8:$src1, relocImm:$src2)>;
def : Pat<(rotl GR16:$src1, (i8 relocImm:$src2)),
          (ROL16ri GR16:$src1, relocImm:$src2)>;
def : Pat<(rotl GR32:$src1, (i8 relocImm:$src2)),
          (ROL32ri GR32:$src1, relocImm:$src2)>;
def : Pat<(rotl GR64:$src1, (i8 relocImm:$src2)),
          (ROL64ri GR64:$src1, relocImm:$src2)>;

def : Pat<(rotr GR8:$src1, (i8 relocImm:$src2)),
          (ROR8ri GR8:$src1, relocImm:$src2)>;
def : Pat<(rotr GR16:$src1, (i8 relocImm:$src2)),
          (ROR16ri GR16:$src1, relocImm:$src2)>;
def : Pat<(rotr GR32:$src1, (i8 relocImm:$src2)),
          (ROR32ri GR32:$src1, relocImm:$src2)>;
def : Pat<(rotr GR64:$src1, (i8 relocImm:$src2)),
          (ROR64ri GR64:$src1, relocImm:$src2)>;

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
// Shift instructions
//===----------------------------------------------------------------------===//

class ShiftOpRCL_R<string m, Format f, X86TypeInfo t, SDPatternOperator node>
  : ITy<0xD3, f, t, (outs t.RegClass:$dst), (ins t.RegClass:$src1), m,
        "{%cl, $src1|$src1, cl}",
        [(set t.RegClass:$dst, (node t.RegClass:$src1, CL))]>,
    Sched<[WriteShiftCL]> {
  let Uses = [CL];
}

class ShiftOpRI_R<string m, Format f, X86TypeInfo t, SDPatternOperator node>
  : ITy<0xC1, f, t, (outs t.RegClass:$dst), (ins t.RegClass:$src1, u8imm:$src2), m,
        binop_args,
        [(set t.RegClass:$dst, (node t.RegClass:$src1, (i8 imm:$src2)))]>,
    Sched<[WriteShift]> {
  let ImmT = Imm8;
}

class ShiftOpR1<string m, Format f, X86TypeInfo t>
  : ITy<0xD1, f, t, (outs t.RegClass:$dst), (ins t.RegClass:$src1), m,
        "$src1", []>, Sched<[WriteShift]>;

class ShiftOpMCL<string m, Format f, X86TypeInfo t, SDPatternOperator node>
  : ITy<0xD3, f, t, (outs), (ins t.MemOperand:$src1), m,
        "{%cl, $src1|$src1, cl}",
        [(store (node (t.LoadNode addr:$src1), CL), addr:$src1)]>,
    Sched<[WriteShiftCLLd, WriteRMW]> {
  let Uses = [CL];
  let mayLoad = 1;
  let mayStore = 1;
}

class ShiftOpMI<string m, Format f, X86TypeInfo t, SDPatternOperator node>
  : ITy<0xC1, f, t, (outs), (ins t.MemOperand:$src1, u8imm:$src2), m,
        binop_args, [(store (node (t.LoadNode addr:$src1), (i8 imm:$src2)), addr:$src1)]>,
    Sched<[WriteShiftLd, WriteRMW]> {
  let ImmT = Imm8;
  let mayLoad = 1;
  let mayStore = 1;
}

class ShiftOpM1<string m, Format f, X86TypeInfo t>
  : ITy<0xD1, f, t, (outs), (ins t.MemOperand:$src1), m,
        "$src1", []>, Sched<[WriteShiftLd, WriteRMW]> {
  let mayLoad = 1;
  let mayStore = 1;
}

multiclass Shift<string m, Format RegMRM, Format MemMRM, SDPatternOperator node,
                 bit ConvertibleToThreeAddress> {
  let Constraints = "$src1 = $dst" in {
    def 8rCL  : ShiftOpRCL_R<m, RegMRM, Xi8, node>, DefEFLAGS;
    def 16rCL : ShiftOpRCL_R<m, RegMRM, Xi16, node>, DefEFLAGS, OpSize16;
    def 32rCL : ShiftOpRCL_R<m, RegMRM, Xi32, node>, DefEFLAGS, OpSize32;
    def 64rCL : ShiftOpRCL_R<m, RegMRM, Xi64, node>, DefEFLAGS;

    let isConvertibleToThreeAddress = ConvertibleToThreeAddress in {
      def 8ri  : ShiftOpRI_R<m, RegMRM, Xi8, node>, DefEFLAGS;
      def 16ri : ShiftOpRI_R<m, RegMRM, Xi16, node>, DefEFLAGS, OpSize16;
      def 32ri : ShiftOpRI_R<m, RegMRM, Xi32, node>, DefEFLAGS, OpSize32;
      def 64ri : ShiftOpRI_R<m, RegMRM, Xi64, node>, DefEFLAGS;
    }

    def 8r1  : ShiftOpR1<m, RegMRM, Xi8>, DefEFLAGS;
    def 16r1 : ShiftOpR1<m, RegMRM, Xi16>, DefEFLAGS, OpSize16;
    def 32r1 : ShiftOpR1<m, RegMRM, Xi32>, DefEFLAGS, OpSize32;
    def 64r1 : ShiftOpR1<m, RegMRM, Xi64>, DefEFLAGS;
  }

  def 8mCL  : ShiftOpMCL<m, MemMRM, Xi8, node>, DefEFLAGS;
  def 16mCL : ShiftOpMCL<m, MemMRM, Xi16, node>, DefEFLAGS, OpSize16;
  def 32mCL : ShiftOpMCL<m, MemMRM, Xi32, node>, DefEFLAGS, OpSize32;
  def 64mCL : ShiftOpMCL<m, MemMRM, Xi64, node>, DefEFLAGS, Requires<[In64BitMode]>;

  def 8mi  : ShiftOpMI<m, MemMRM, Xi8, node>, DefEFLAGS;
  def 16mi : ShiftOpMI<m, MemMRM, Xi16, node>, DefEFLAGS, OpSize16;
  def 32mi : ShiftOpMI<m, MemMRM, Xi32, node>, DefEFLAGS, OpSize32;
  def 64mi : ShiftOpMI<m, MemMRM, Xi64, node>, DefEFLAGS, Requires<[In64BitMode]>;

  def 8m1  : ShiftOpM1<m, MemMRM, Xi8>, DefEFLAGS;
  def 16m1 : ShiftOpM1<m, MemMRM, Xi16>, DefEFLAGS, OpSize16;
  def 32m1 : ShiftOpM1<m, MemMRM, Xi32>, DefEFLAGS, OpSize32;
  def 64m1 : ShiftOpM1<m, MemMRM, Xi64>, DefEFLAGS, Requires<[In64BitMode]>;
}

defm SHL: Shift<"shl", MRM4r, MRM4m, shl, 1>;
defm SHR: Shift<"shr", MRM5r, MRM5m, srl, 0>;
defm SAR: Shift<"sar", MRM7r, MRM7m, sra, 0>;

//===----------------------------------------------------------------------===//
// Rotate instructions
//===----------------------------------------------------------------------===//

let Defs = [EFLAGS], hasSideEffects = 0 in {

let Constraints = "$src1 = $dst" in {

let Uses = [CL, EFLAGS], SchedRW = [WriteRotateCL] in {
def RCL8rCL : I<0xD2, MRM2r, (outs GR8:$dst), (ins GR8:$src1),
                "rcl{b}\t{%cl, $dst|$dst, cl}", []>;
def RCL16rCL : I<0xD3, MRM2r, (outs GR16:$dst), (ins GR16:$src1),
                 "rcl{w}\t{%cl, $dst|$dst, cl}", []>, OpSize16;
def RCL32rCL : I<0xD3, MRM2r, (outs GR32:$dst), (ins GR32:$src1),
                 "rcl{l}\t{%cl, $dst|$dst, cl}", []>, OpSize32;
def RCL64rCL : RI<0xD3, MRM2r, (outs GR64:$dst), (ins GR64:$src1),
                  "rcl{q}\t{%cl, $dst|$dst, cl}", []>;
} // Uses = [CL, EFLAGS], SchedRW

let Uses = [EFLAGS], SchedRW = [WriteRotate] in {
def RCL8r1 : I<0xD0, MRM2r, (outs GR8:$dst), (ins GR8:$src1),
               "rcl{b}\t$dst", []>;
def RCL8ri : Ii8<0xC0, MRM2r, (outs GR8:$dst), (ins GR8:$src1, u8imm:$cnt),
                 "rcl{b}\t{$cnt, $dst|$dst, $cnt}", []>;
def RCL16r1 : I<0xD1, MRM2r, (outs GR16:$dst), (ins GR16:$src1),
                "rcl{w}\t$dst", []>, OpSize16;
def RCL16ri : Ii8<0xC1, MRM2r, (outs GR16:$dst), (ins GR16:$src1, u8imm:$cnt),
                  "rcl{w}\t{$cnt, $dst|$dst, $cnt}", []>, OpSize16;
def RCL32r1 : I<0xD1, MRM2r, (outs GR32:$dst), (ins GR32:$src1),
                "rcl{l}\t$dst", []>, OpSize32;
def RCL32ri : Ii8<0xC1, MRM2r, (outs GR32:$dst), (ins GR32:$src1, u8imm:$cnt),
                  "rcl{l}\t{$cnt, $dst|$dst, $cnt}", []>, OpSize32;
def RCL64r1 : RI<0xD1, MRM2r, (outs GR64:$dst), (ins GR64:$src1),
                 "rcl{q}\t$dst", []>;
def RCL64ri : RIi8<0xC1, MRM2r, (outs GR64:$dst), (ins GR64:$src1, u8imm:$cnt),
                   "rcl{q}\t{$cnt, $dst|$dst, $cnt}", []>;
} // Uses = [EFLAGS], SchedRW

let Uses = [CL, EFLAGS], SchedRW = [WriteRotateCL] in {
def RCR8rCL : I<0xD2, MRM3r, (outs GR8:$dst), (ins GR8:$src1),
                "rcr{b}\t{%cl, $dst|$dst, cl}", []>;
def RCR16rCL : I<0xD3, MRM3r, (outs GR16:$dst), (ins GR16:$src1),
                 "rcr{w}\t{%cl, $dst|$dst, cl}", []>, OpSize16;
def RCR32rCL : I<0xD3, MRM3r, (outs GR32:$dst), (ins GR32:$src1),
                 "rcr{l}\t{%cl, $dst|$dst, cl}", []>, OpSize32;
def RCR64rCL : RI<0xD3, MRM3r, (outs GR64:$dst), (ins GR64:$src1),
                  "rcr{q}\t{%cl, $dst|$dst, cl}", []>;
} // Uses = [CL, EFLAGS], SchedRW

let Uses = [EFLAGS], SchedRW = [WriteRotate] in {
def RCR8r1 : I<0xD0, MRM3r, (outs GR8:$dst), (ins GR8:$src1),
               "rcr{b}\t$dst", []>;
def RCR8ri : Ii8<0xC0, MRM3r, (outs GR8:$dst), (ins GR8:$src1, u8imm:$cnt),
                 "rcr{b}\t{$cnt, $dst|$dst, $cnt}", []>;
def RCR16r1 : I<0xD1, MRM3r, (outs GR16:$dst), (ins GR16:$src1),
                "rcr{w}\t$dst", []>, OpSize16;
def RCR16ri : Ii8<0xC1, MRM3r, (outs GR16:$dst), (ins GR16:$src1, u8imm:$cnt),
                  "rcr{w}\t{$cnt, $dst|$dst, $cnt}", []>, OpSize16;
def RCR32r1 : I<0xD1, MRM3r, (outs GR32:$dst), (ins GR32:$src1),
                "rcr{l}\t$dst", []>, OpSize32;
def RCR32ri : Ii8<0xC1, MRM3r, (outs GR32:$dst), (ins GR32:$src1, u8imm:$cnt),
                  "rcr{l}\t{$cnt, $dst|$dst, $cnt}", []>, OpSize32;
def RCR64r1 : RI<0xD1, MRM3r, (outs GR64:$dst), (ins GR64:$src1),
                 "rcr{q}\t$dst", []>;
def RCR64ri : RIi8<0xC1, MRM3r, (outs GR64:$dst), (ins GR64:$src1, u8imm:$cnt),
                   "rcr{q}\t{$cnt, $dst|$dst, $cnt}", []>;
} // Uses = [EFLAGS], SchedRW
} // Constraints = "$src = $dst"

let mayLoad = 1, mayStore = 1 in {
let Uses = [EFLAGS], SchedRW = [WriteRotateLd, WriteRMW] in {
def RCL8m1 : I<0xD0, MRM2m, (outs), (ins i8mem:$dst),
               "rcl{b}\t$dst", []>;
def RCL8mi : Ii8<0xC0, MRM2m, (outs), (ins i8mem:$dst, u8imm:$cnt),
                 "rcl{b}\t{$cnt, $dst|$dst, $cnt}", []>;
def RCL16m1 : I<0xD1, MRM2m, (outs), (ins i16mem:$dst),
                "rcl{w}\t$dst", []>, OpSize16;
def RCL16mi : Ii8<0xC1, MRM2m, (outs), (ins i16mem:$dst, u8imm:$cnt),
                  "rcl{w}\t{$cnt, $dst|$dst, $cnt}", []>, OpSize16;
def RCL32m1 : I<0xD1, MRM2m, (outs), (ins i32mem:$dst),
                "rcl{l}\t$dst", []>, OpSize32;
def RCL32mi : Ii8<0xC1, MRM2m, (outs), (ins i32mem:$dst, u8imm:$cnt),
                  "rcl{l}\t{$cnt, $dst|$dst, $cnt}", []>, OpSize32;
def RCL64m1 : RI<0xD1, MRM2m, (outs), (ins i64mem:$dst),
                 "rcl{q}\t$dst", []>, Requires<[In64BitMode]>;
def RCL64mi : RIi8<0xC1, MRM2m, (outs), (ins i64mem:$dst, u8imm:$cnt),
                   "rcl{q}\t{$cnt, $dst|$dst, $cnt}", []>,
                   Requires<[In64BitMode]>;

def RCR8m1 : I<0xD0, MRM3m, (outs), (ins i8mem:$dst),
               "rcr{b}\t$dst", []>;
def RCR8mi : Ii8<0xC0, MRM3m, (outs), (ins i8mem:$dst, u8imm:$cnt),
                 "rcr{b}\t{$cnt, $dst|$dst, $cnt}", []>;
def RCR16m1 : I<0xD1, MRM3m, (outs), (ins i16mem:$dst),
                "rcr{w}\t$dst", []>, OpSize16;
def RCR16mi : Ii8<0xC1, MRM3m, (outs), (ins i16mem:$dst, u8imm:$cnt),
                  "rcr{w}\t{$cnt, $dst|$dst, $cnt}", []>, OpSize16;
def RCR32m1 : I<0xD1, MRM3m, (outs), (ins i32mem:$dst),
                "rcr{l}\t$dst", []>, OpSize32;
def RCR32mi : Ii8<0xC1, MRM3m, (outs), (ins i32mem:$dst, u8imm:$cnt),
                  "rcr{l}\t{$cnt, $dst|$dst, $cnt}", []>, OpSize32;
def RCR64m1 : RI<0xD1, MRM3m, (outs), (ins i64mem:$dst),
                 "rcr{q}\t$dst", []>, Requires<[In64BitMode]>;
def RCR64mi : RIi8<0xC1, MRM3m, (outs), (ins i64mem:$dst, u8imm:$cnt),
                   "rcr{q}\t{$cnt, $dst|$dst, $cnt}", []>,
                   Requires<[In64BitMode]>;
} // Uses = [EFLAGS], SchedRW

let Uses = [CL, EFLAGS], SchedRW = [WriteRotateCLLd, WriteRMW] in {
def RCL8mCL : I<0xD2, MRM2m, (outs), (ins i8mem:$dst),
                "rcl{b}\t{%cl, $dst|$dst, cl}", []>;
def RCL16mCL : I<0xD3, MRM2m, (outs), (ins i16mem:$dst),
                 "rcl{w}\t{%cl, $dst|$dst, cl}", []>, OpSize16;
def RCL32mCL : I<0xD3, MRM2m, (outs), (ins i32mem:$dst),
                 "rcl{l}\t{%cl, $dst|$dst, cl}", []>, OpSize32;
def RCL64mCL : RI<0xD3, MRM2m, (outs), (ins i64mem:$dst),
                  "rcl{q}\t{%cl, $dst|$dst, cl}", []>,
                  Requires<[In64BitMode]>;

def RCR8mCL : I<0xD2, MRM3m, (outs), (ins i8mem:$dst),
                "rcr{b}\t{%cl, $dst|$dst, cl}", []>;
def RCR16mCL : I<0xD3, MRM3m, (outs), (ins i16mem:$dst),
                 "rcr{w}\t{%cl, $dst|$dst, cl}", []>, OpSize16;
def RCR32mCL : I<0xD3, MRM3m, (outs), (ins i32mem:$dst),
                 "rcr{l}\t{%cl, $dst|$dst, cl}", []>, OpSize32;
def RCR64mCL : RI<0xD3, MRM3m, (outs), (ins i64mem:$dst),
                  "rcr{q}\t{%cl, $dst|$dst, cl}", []>,
                  Requires<[In64BitMode]>;
} // Uses = [CL, EFLAGS], SchedRW
} // mayLoad, mayStore

let Constraints = "$src1 = $dst" in {
// FIXME: provide shorter instructions when imm8 == 1
let Uses = [CL], SchedRW = [WriteRotateCL] in {
def ROL8rCL  : I<0xD2, MRM0r, (outs GR8 :$dst), (ins GR8 :$src1),
                 "rol{b}\t{%cl, $dst|$dst, cl}",
                 [(set GR8:$dst, (rotl GR8:$src1, CL))]>;
def ROL16rCL : I<0xD3, MRM0r, (outs GR16:$dst), (ins GR16:$src1),
                 "rol{w}\t{%cl, $dst|$dst, cl}",
                 [(set GR16:$dst, (rotl GR16:$src1, CL))]>, OpSize16;
def ROL32rCL : I<0xD3, MRM0r, (outs GR32:$dst), (ins GR32:$src1),
                 "rol{l}\t{%cl, $dst|$dst, cl}",
                 [(set GR32:$dst, (rotl GR32:$src1, CL))]>, OpSize32;
def ROL64rCL : RI<0xD3, MRM0r, (outs GR64:$dst), (ins GR64:$src1),
                  "rol{q}\t{%cl, $dst|$dst, cl}",
                  [(set GR64:$dst, (rotl GR64:$src1, CL))]>;
} // Uses, SchedRW

let SchedRW = [WriteRotate] in {
def ROL8ri   : Ii8<0xC0, MRM0r, (outs GR8 :$dst), (ins GR8 :$src1, u8imm:$src2),
                   "rol{b}\t{$src2, $dst|$dst, $src2}",
                   [(set GR8:$dst, (rotl GR8:$src1, (i8 imm:$src2)))]>;
def ROL16ri  : Ii8<0xC1, MRM0r, (outs GR16:$dst), (ins GR16:$src1, u8imm:$src2),
                   "rol{w}\t{$src2, $dst|$dst, $src2}",
                   [(set GR16:$dst, (rotl GR16:$src1, (i8 imm:$src2)))]>,
                   OpSize16;
def ROL32ri  : Ii8<0xC1, MRM0r, (outs GR32:$dst), (ins GR32:$src1, u8imm:$src2),
                   "rol{l}\t{$src2, $dst|$dst, $src2}",
                   [(set GR32:$dst, (rotl GR32:$src1, (i8 imm:$src2)))]>,
                   OpSize32;
def ROL64ri  : RIi8<0xC1, MRM0r, (outs GR64:$dst),
                    (ins GR64:$src1, u8imm:$src2),
                    "rol{q}\t{$src2, $dst|$dst, $src2}",
                    [(set GR64:$dst, (rotl GR64:$src1, (i8 imm:$src2)))]>;

// Rotate by 1
def ROL8r1   : I<0xD0, MRM0r, (outs GR8 :$dst), (ins GR8 :$src1),
                 "rol{b}\t$dst", []>;
def ROL16r1  : I<0xD1, MRM0r, (outs GR16:$dst), (ins GR16:$src1),
                 "rol{w}\t$dst", []>, OpSize16;
def ROL32r1  : I<0xD1, MRM0r, (outs GR32:$dst), (ins GR32:$src1),
                 "rol{l}\t$dst", []>, OpSize32;
def ROL64r1  : RI<0xD1, MRM0r, (outs GR64:$dst), (ins GR64:$src1),
                  "rol{q}\t$dst", []>;
} // SchedRW
} // Constraints = "$src = $dst"

let Uses = [CL], SchedRW = [WriteRotateCLLd, WriteRMW] in {
def ROL8mCL  : I<0xD2, MRM0m, (outs), (ins i8mem :$dst),
                 "rol{b}\t{%cl, $dst|$dst, cl}",
                 [(store (rotl (loadi8 addr:$dst), CL), addr:$dst)]>;
def ROL16mCL : I<0xD3, MRM0m, (outs), (ins i16mem:$dst),
                 "rol{w}\t{%cl, $dst|$dst, cl}",
                 [(store (rotl (loadi16 addr:$dst), CL), addr:$dst)]>, OpSize16;
def ROL32mCL : I<0xD3, MRM0m, (outs), (ins i32mem:$dst),
                 "rol{l}\t{%cl, $dst|$dst, cl}",
                 [(store (rotl (loadi32 addr:$dst), CL), addr:$dst)]>, OpSize32;
def ROL64mCL :  RI<0xD3, MRM0m, (outs), (ins i64mem:$dst),
                   "rol{q}\t{%cl, $dst|$dst, cl}",
                   [(store (rotl (loadi64 addr:$dst), CL), addr:$dst)]>,
                   Requires<[In64BitMode]>;
} // Uses, SchedRW

let SchedRW = [WriteRotateLd, WriteRMW], mayLoad = 1, mayStore = 1 in {
def ROL8mi   : Ii8<0xC0, MRM0m, (outs), (ins i8mem :$dst, u8imm:$src1),
                   "rol{b}\t{$src1, $dst|$dst, $src1}",
               [(store (rotl (loadi8 addr:$dst), (i8 imm:$src1)), addr:$dst)]>;
def ROL16mi  : Ii8<0xC1, MRM0m, (outs), (ins i16mem:$dst, u8imm:$src1),
                   "rol{w}\t{$src1, $dst|$dst, $src1}",
              [(store (rotl (loadi16 addr:$dst), (i8 imm:$src1)), addr:$dst)]>,
              OpSize16;
def ROL32mi  : Ii8<0xC1, MRM0m, (outs), (ins i32mem:$dst, u8imm:$src1),
                   "rol{l}\t{$src1, $dst|$dst, $src1}",
              [(store (rotl (loadi32 addr:$dst), (i8 imm:$src1)), addr:$dst)]>,
              OpSize32;
def ROL64mi  : RIi8<0xC1, MRM0m, (outs), (ins i64mem:$dst, u8imm:$src1),
                    "rol{q}\t{$src1, $dst|$dst, $src1}",
                [(store (rotl (loadi64 addr:$dst), (i8 imm:$src1)), addr:$dst)]>,
                Requires<[In64BitMode]>;

// Rotate by 1
def ROL8m1   : I<0xD0, MRM0m, (outs), (ins i8mem :$dst),
                 "rol{b}\t$dst", []>;
def ROL16m1  : I<0xD1, MRM0m, (outs), (ins i16mem:$dst),
                 "rol{w}\t$dst", []>, OpSize16;
def ROL32m1  : I<0xD1, MRM0m, (outs), (ins i32mem:$dst),
                 "rol{l}\t$dst", []>, OpSize32;
def ROL64m1  : RI<0xD1, MRM0m, (outs), (ins i64mem:$dst),
                 "rol{q}\t$dst", []>, Requires<[In64BitMode]>;
} // SchedRW, mayLoad, mayStore

let Constraints = "$src1 = $dst" in {
let Uses = [CL], SchedRW = [WriteRotateCL] in {
def ROR8rCL  : I<0xD2, MRM1r, (outs GR8 :$dst), (ins GR8 :$src1),
                 "ror{b}\t{%cl, $dst|$dst, cl}",
                 [(set GR8:$dst, (rotr GR8:$src1, CL))]>;
def ROR16rCL : I<0xD3, MRM1r, (outs GR16:$dst), (ins GR16:$src1),
                 "ror{w}\t{%cl, $dst|$dst, cl}",
                 [(set GR16:$dst, (rotr GR16:$src1, CL))]>, OpSize16;
def ROR32rCL : I<0xD3, MRM1r, (outs GR32:$dst), (ins GR32:$src1),
                 "ror{l}\t{%cl, $dst|$dst, cl}",
                 [(set GR32:$dst, (rotr GR32:$src1, CL))]>, OpSize32;
def ROR64rCL : RI<0xD3, MRM1r, (outs GR64:$dst), (ins GR64:$src1),
                  "ror{q}\t{%cl, $dst|$dst, cl}",
                  [(set GR64:$dst, (rotr GR64:$src1, CL))]>;
}

let SchedRW = [WriteRotate] in {
def ROR8ri   : Ii8<0xC0, MRM1r, (outs GR8 :$dst), (ins GR8 :$src1, u8imm:$src2),
                   "ror{b}\t{$src2, $dst|$dst, $src2}",
                   [(set GR8:$dst, (rotr GR8:$src1, (i8 imm:$src2)))]>;
def ROR16ri  : Ii8<0xC1, MRM1r, (outs GR16:$dst), (ins GR16:$src1, u8imm:$src2),
                   "ror{w}\t{$src2, $dst|$dst, $src2}",
                   [(set GR16:$dst, (rotr GR16:$src1, (i8 imm:$src2)))]>,
                   OpSize16;
def ROR32ri  : Ii8<0xC1, MRM1r, (outs GR32:$dst), (ins GR32:$src1, u8imm:$src2),
                   "ror{l}\t{$src2, $dst|$dst, $src2}",
                   [(set GR32:$dst, (rotr GR32:$src1, (i8 imm:$src2)))]>,
                   OpSize32;
def ROR64ri  : RIi8<0xC1, MRM1r, (outs GR64:$dst),
                    (ins GR64:$src1, u8imm:$src2),
                    "ror{q}\t{$src2, $dst|$dst, $src2}",
                    [(set GR64:$dst, (rotr GR64:$src1, (i8 imm:$src2)))]>;

// Rotate by 1
def ROR8r1   : I<0xD0, MRM1r, (outs GR8 :$dst), (ins GR8 :$src1),
                 "ror{b}\t$dst", []>;
def ROR16r1  : I<0xD1, MRM1r, (outs GR16:$dst), (ins GR16:$src1),
                 "ror{w}\t$dst", []>, OpSize16;
def ROR32r1  : I<0xD1, MRM1r, (outs GR32:$dst), (ins GR32:$src1),
                 "ror{l}\t$dst", []>, OpSize32;
def ROR64r1  : RI<0xD1, MRM1r, (outs GR64:$dst), (ins GR64:$src1),
                  "ror{q}\t$dst", []>;
} // SchedRW
} // Constraints = "$src = $dst", SchedRW

let Uses = [CL], SchedRW = [WriteRotateCLLd, WriteRMW] in {
def ROR8mCL  : I<0xD2, MRM1m, (outs), (ins i8mem :$dst),
                 "ror{b}\t{%cl, $dst|$dst, cl}",
                 [(store (rotr (loadi8 addr:$dst), CL), addr:$dst)]>;
def ROR16mCL : I<0xD3, MRM1m, (outs), (ins i16mem:$dst),
                 "ror{w}\t{%cl, $dst|$dst, cl}",
                 [(store (rotr (loadi16 addr:$dst), CL), addr:$dst)]>, OpSize16;
def ROR32mCL : I<0xD3, MRM1m, (outs), (ins i32mem:$dst),
                 "ror{l}\t{%cl, $dst|$dst, cl}",
                 [(store (rotr (loadi32 addr:$dst), CL), addr:$dst)]>, OpSize32;
def ROR64mCL : RI<0xD3, MRM1m, (outs), (ins i64mem:$dst),
                  "ror{q}\t{%cl, $dst|$dst, cl}",
                  [(store (rotr (loadi64 addr:$dst), CL), addr:$dst)]>,
                  Requires<[In64BitMode]>;
} // Uses, SchedRW

let SchedRW = [WriteRotateLd, WriteRMW], mayLoad = 1, mayStore =1 in {
def ROR8mi   : Ii8<0xC0, MRM1m, (outs), (ins i8mem :$dst, u8imm:$src),
                   "ror{b}\t{$src, $dst|$dst, $src}",
                   [(store (rotr (loadi8 addr:$dst), (i8 imm:$src)), addr:$dst)]>;
def ROR16mi  : Ii8<0xC1, MRM1m, (outs), (ins i16mem:$dst, u8imm:$src),
                   "ror{w}\t{$src, $dst|$dst, $src}",
                   [(store (rotr (loadi16 addr:$dst), (i8 imm:$src)), addr:$dst)]>,
                   OpSize16;
def ROR32mi  : Ii8<0xC1, MRM1m, (outs), (ins i32mem:$dst, u8imm:$src),
                   "ror{l}\t{$src, $dst|$dst, $src}",
                   [(store (rotr (loadi32 addr:$dst), (i8 imm:$src)), addr:$dst)]>,
                   OpSize32;
def ROR64mi  : RIi8<0xC1, MRM1m, (outs), (ins i64mem:$dst, u8imm:$src),
                    "ror{q}\t{$src, $dst|$dst, $src}",
                    [(store (rotr (loadi64 addr:$dst), (i8 imm:$src)), addr:$dst)]>,
                    Requires<[In64BitMode]>;

// Rotate by 1
def ROR8m1   : I<0xD0, MRM1m, (outs), (ins i8mem :$dst),
                 "ror{b}\t$dst", []>;
def ROR16m1  : I<0xD1, MRM1m, (outs), (ins i16mem:$dst),
                 "ror{w}\t$dst", []>, OpSize16;
def ROR32m1  : I<0xD1, MRM1m, (outs), (ins i32mem:$dst),
                 "ror{l}\t$dst", []>,
                 OpSize32;
def ROR64m1  : RI<0xD1, MRM1m, (outs), (ins i64mem:$dst),
                 "ror{q}\t$dst", []>, Requires<[In64BitMode]>;
} // SchedRW, mayLoad, mayStore


//===----------------------------------------------------------------------===//
// Double shift instructions (generalizations of rotate)
//===----------------------------------------------------------------------===//

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

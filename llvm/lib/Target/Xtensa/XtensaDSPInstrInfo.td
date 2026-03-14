//===- XtensaDSPInstrInfo.td - Xtensa Target Description ---*- tablegen -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file describes the Xtensa DSP instructions in TableGen format.
//
//===----------------------------------------------------------------------===//

// Multiply
class UMUL_AA<bits<4> oper1, string instrAsm>
  : RRR_Inst<0x04, oper1, 0x07, (outs), (ins AR:$s, AR:$t),
             instrAsm#"\t$s, $t", []>, Requires<[HasMAC16]> {
  let r = 0;
  let Defs = [M1, M2, ACCLO, ACCHI];
}

def UMUL_AA_LL : UMUL_AA<0x00, "umul.aa.ll">;
def UMUL_AA_HL : UMUL_AA<0x01, "umul.aa.hl">;
def UMUL_AA_LH : UMUL_AA<0x02, "umul.aa.lh">;
def UMUL_AA_HH : UMUL_AA<0x03, "umul.aa.hh">;

class MUL_AA<bits<4> oper1, string instrAsm>
  : RRR_Inst<0x04, oper1, 0x07, (outs), (ins AR:$s, AR:$t),
             instrAsm#"\t$s, $t", []>, Requires<[HasMAC16]> {
  let r = 0;
  let Defs = [M1, M2, ACCLO, ACCHI];
}

def MUL_AA_LL : MUL_AA<0x04, "mul.aa.ll">;
def MUL_AA_HL : MUL_AA<0x05, "mul.aa.hl">;
def MUL_AA_LH : MUL_AA<0x06, "mul.aa.lh">;
def MUL_AA_HH : MUL_AA<0x07, "mul.aa.hh">;

class MUL_AD<bits<4> oper1, string instrAsm>
  : RRR_Inst<0x04, oper1, 0x03, (outs), (ins AR:$s, MR23:$y),
             instrAsm#"\t$s, $y", []>, Requires<[HasMAC16]> {
  bits<2> y;

  let r = 0;
  let t{3} = 0;
  let t{2} = y{0};
  let t{1-0} = 0;
  let Defs = [M1, M2, ACCLO, ACCHI];
}

def MUL_AD_LL : MUL_AD<0x04, "mul.ad.ll">;
def MUL_AD_HL : MUL_AD<0x05, "mul.ad.hl">;
def MUL_AD_LH : MUL_AD<0x06, "mul.ad.lh">;
def MUL_AD_HH : MUL_AD<0x07, "mul.ad.hh">;

class MUL_DA<bits<4> oper1, string instrAsm>
  : RRR_Inst<0x04, oper1, 0x06, (outs), (ins MR01:$x, AR:$t),
             instrAsm#"\t$x, $t", []>, Requires<[HasMAC16]> {
  bits<2> x;

  let r{3} = 0;
  let r{2} = x{0};
  let r{1-0} = 0;
  let s = 0;
  let Defs = [M1, M2, ACCLO, ACCHI];
}

def MUL_DA_LL : MUL_DA<0x04, "mul.da.ll">;
def MUL_DA_HL : MUL_DA<0x05, "mul.da.hl">;
def MUL_DA_LH : MUL_DA<0x06, "mul.da.lh">;
def MUL_DA_HH : MUL_DA<0x07, "mul.da.hh">;

class MUL_DD<bits<4> oper1, string instrAsm>
  : RRR_Inst<0x04, oper1, 0x02, (outs), (ins MR01:$x, MR23:$y),
             instrAsm#"\t$x, $y", []>, Requires<[HasMAC16]> {
  bits<2> x;
  bits<2> y;

  let r{3} = 0;
  let r{2} = x{0};
  let r{1-0} = 0;
  let s = 0;
  let t{3} = 0;
  let t{2} = y{0};
  let t{1-0} = 0;
  let Defs = [M1, M2, ACCLO, ACCHI];
}

def MUL_DD_LL : MUL_DD<0x04, "mul.dd.ll">;
def MUL_DD_HL : MUL_DD<0x05, "mul.dd.hl">;
def MUL_DD_LH : MUL_DD<0x06, "mul.dd.lh">;
def MUL_DD_HH : MUL_DD<0x07, "mul.dd.hh">;

class MULA_AA<bits<4> oper1, string instrAsm>
  : RRR_Inst<0x04, oper1, 0x07, (outs), (ins AR:$s, AR:$t),
             instrAsm#"\t$s, $t", []>, Requires<[HasMAC16]> {
  let r = 0;
  let Defs = [M1, M2, ACCLO, ACCHI];
}

def MULA_AA_LL : MULA_AA<0x08, "mula.aa.ll">;
def MULA_AA_HL : MULA_AA<0x09, "mula.aa.hl">;
def MULA_AA_LH : MULA_AA<0x0A, "mula.aa.lh">;
def MULA_AA_HH : MULA_AA<0x0B, "mula.aa.hh">;

class MULA_AD<bits<4> oper1, string instrAsm>
  : RRR_Inst<0x04, oper1, 0x03, (outs), (ins AR:$s, MR23:$y),
             instrAsm#"\t$s, $y", []>, Requires<[HasMAC16]> {
  bits<2> y;

  let r = 0;
  let t{3} = 0;
  let t{2} = y{0};
  let t{1-0} = 0;

  let Uses = [ACCLO, ACCHI];
  let Defs = [M1, M2, ACCLO, ACCHI];
}

def MULA_AD_LL : MULA_AD<0x08, "mula.ad.ll">;
def MULA_AD_HL : MULA_AD<0x09, "mula.ad.hl">;
def MULA_AD_LH : MULA_AD<0x0A, "mula.ad.lh">;
def MULA_AD_HH : MULA_AD<0x0B, "mula.ad.hh">;

class MULA_DA<bits<4> oper1, string instrAsm>
  : RRR_Inst<0x04, oper1, 0x06, (outs), (ins MR01:$x, AR:$t),
             instrAsm#"\t$x, $t", []>, Requires<[HasMAC16]> {
  bits<2> x;

  let r{3} = 0;
  let r{2} = x{0};
  let r{1-0} = 0;
  let s = 0;

  let Uses = [ACCLO, ACCHI];
  let Defs = [M1, M2, ACCLO, ACCHI];
}

def MULA_DA_LL : MULA_DA<0x08, "mula.da.ll">;
def MULA_DA_HL : MULA_DA<0x09, "mula.da.hl">;
def MULA_DA_LH : MULA_DA<0x0A, "mula.da.lh">;
def MULA_DA_HH : MULA_DA<0x0B, "mula.da.hh">;

class MULA_DD<bits<4> oper1, string instrAsm>
  : RRR_Inst<0x04, oper1, 0x02, (outs), (ins MR01:$x, MR23:$y),
             instrAsm#"\t$x, $y", []>, Requires<[HasMAC16]> {
  bits<2> x;
  bits<2> y;

  let r{3} = 0;
  let r{2} = x{0};
  let r{1-0} = 0;
  let s = 0;
  let t{3} = 0;
  let t{2} = y{0};
  let t{1-0} = 0;

  let Uses = [ACCLO, ACCHI];
  let Defs = [M1, M2, ACCLO, ACCHI];
}

def MULA_DD_LL : MULA_DD<0x08, "mula.dd.ll">;
def MULA_DD_HL : MULA_DD<0x09, "mula.dd.hl">;
def MULA_DD_LH : MULA_DD<0x0A, "mula.dd.lh">;
def MULA_DD_HH : MULA_DD<0x0B, "mula.dd.hh">;

class MULS_AA<bits<4> oper1, string instrAsm>
  : RRR_Inst<0x04, oper1, 0x07, (outs), (ins AR:$s, AR:$t),
             instrAsm#"\t$s, $t", []>, Requires<[HasMAC16]> {
  let r = 0;
  let Uses = [ACCLO, ACCHI];
  let Defs = [M1, M2, ACCLO, ACCHI];
}

def MULS_AA_LL : MULS_AA<0x0C, "muls.aa.ll">;
def MULS_AA_HL : MULS_AA<0x0D, "muls.aa.hl">;
def MULS_AA_LH : MULS_AA<0x0E, "muls.aa.lh">;
def MULS_AA_HH : MULS_AA<0x0F, "muls.aa.hh">;

class MULS_AD<bits<4> oper1, string instrAsm>
  : RRR_Inst<0x04, oper1, 0x03, (outs), (ins AR:$s, MR23:$y),
             instrAsm#"\t$s, $y", []>, Requires<[HasMAC16]> {
  bits<2> y;

  let r = 0;
  let t{3} = 0;
  let t{2} = y{0};
  let t{1-0} = 0;

  let Uses = [ACCLO, ACCHI];
  let Defs = [M1, M2, ACCLO, ACCHI];
}

def MULS_AD_LL : MULS_AD<0x0C, "muls.ad.ll">;
def MULS_AD_HL : MULS_AD<0x0D, "muls.ad.hl">;
def MULS_AD_LH : MULS_AD<0x0E, "muls.ad.lh">;
def MULS_AD_HH : MULS_AD<0x0F, "muls.ad.hh">;

class MULS_DA<bits<4> oper1, string instrAsm>
  : RRR_Inst<0x04, oper1, 0x06, (outs), (ins MR01:$x, AR:$t),
             instrAsm#"\t$x, $t", []>, Requires<[HasMAC16]> {
  bits<2> x;

  let r{3} = 0;
  let r{2} = x{0};
  let r{1-0} = 0;
  let s = 0;

  let Uses = [ACCLO, ACCHI];
  let Defs = [M1, M2, ACCLO, ACCHI];
}

def MULS_DA_LL : MULS_DA<0x0C, "muls.da.ll">;
def MULS_DA_HL : MULS_DA<0x0D, "muls.da.hl">;
def MULS_DA_LH : MULS_DA<0x0E, "muls.da.lh">;
def MULS_DA_HH : MULS_DA<0x0F, "muls.da.hh">;

class MULS_DD<bits<4> oper1, string instrAsm>
  : RRR_Inst<0x04, oper1, 0x02, (outs), (ins MR01:$x, MR23:$y),
             instrAsm#"\t$x, $y", []>, Requires<[HasMAC16]> {
  bits<2> x;
  bits<2> y;

  let r{3} = 0;
  let r{2} = x{0};
  let r{1-0} = 0;
  let s = 0;
  let t{3} = 0;
  let t{2} = y{0};
  let t{1-0} = 0;

  let Uses = [ACCLO, ACCHI];
  let Defs = [M1, M2, ACCLO, ACCHI];
}

def MULS_DD_LL : MULS_DD<0x0C, "muls.dd.ll">;
def MULS_DD_HL : MULS_DD<0x0D, "muls.dd.hl">;
def MULS_DD_LH : MULS_DD<0x0E, "muls.dd.lh">;
def MULS_DD_HH : MULS_DD<0x0F, "muls.dd.hh">;

//===----------------------------------------------------------------------===//
// Multiply-accumulate with load

class MULA_DA_LDDEC<bits<4> oper1, string instrAsm>
  : RRR_Inst<0x04, oper1, 0x05, (outs MR:$w, AR:$d), (ins AR:$s, MR01:$x, AR:$t),
             instrAsm#"\t $w, $s, $x, $t", []>, Requires<[HasMAC16]> {
  bits<2> x;
  bits<2> w;

  let Constraints = "$s = $d";
  let mayLoad = 1;
  let r{3} = 0;
  let r{2} = x{0};
  let r{1-0} = w{1-0};
  let Uses = [ACCLO, ACCHI];
  let Defs = [M1, M2, ACCLO, ACCHI];
}

def MULA_DA_LL_LDDEC : MULA_DA_LDDEC<0x08, "mula.da.ll.lddec">;
def MULA_DA_HL_LDDEC : MULA_DA_LDDEC<0x09, "mula.da.hl.lddec">;
def MULA_DA_LH_LDDEC : MULA_DA_LDDEC<0x0A, "mula.da.lh.lddec">;
def MULA_DA_HH_LDDEC : MULA_DA_LDDEC<0x0B, "mula.da.hh.lddec">;

class MULA_DA_LDINC<bits<4> oper1, string instrAsm>
  : RRR_Inst<0x04, oper1, 0x04, (outs MR:$w, AR:$d), (ins AR:$s, MR:$x, AR:$t),
             instrAsm#"\t $w, $s, $x, $t", []>, Requires<[HasMAC16]> {
  bits<1> x;
  bits<2> w;

  let Constraints = "$s = $d";
  let mayLoad = 1;
  let r{3} = 0;
  let r{2} = x{0};
  let r{1-0} = w{1-0};
  let Uses = [ACCLO, ACCHI];
  let Defs = [M1, M2, ACCLO, ACCHI];
}

def MULA_DA_LL_LDINC: MULA_DA_LDINC<0x08, "mula.da.ll.ldinc">;
def MULA_DA_HL_LDINC: MULA_DA_LDINC<0x09, "mula.da.hl.ldinc">;
def MULA_DA_LH_LDINC: MULA_DA_LDINC<0x0A, "mula.da.lh.ldinc">;
def MULA_DA_HH_LDINC: MULA_DA_LDINC<0x0B, "mula.da.hh.ldinc">;

class MULA_DD_LDDEC<bits<4> oper1, string instrAsm>
  : RRR_Inst<0x04, oper1, 0x01, (outs MR:$w, AR:$d), (ins AR:$s, MR01:$x, MR23:$y),
             instrAsm#"\t $w, $s, $x, $y", []>, Requires<[HasMAC16]> {
  bits<2> x;
  bits<2> y;
  bits<2> w;

  let Constraints = "$s = $d";
  let mayLoad = 1;
  let r{3} = 0;
  let r{2} = x{0};
  let r{1-0} = w{1-0};
  let t{3} = 0;
  let t{2} = y{0};
  let t{1-0} = 0;
  let Uses = [ACCLO, ACCHI];
  let Defs = [M1, M2, ACCLO, ACCHI];
}

def MULA_DD_LL_LDDEC : MULA_DD_LDDEC<0x08, "mula.dd.ll.lddec">;
def MULA_DD_HL_LDDEC : MULA_DD_LDDEC<0x09, "mula.dd.hl.lddec">;
def MULA_DD_LH_LDDEC : MULA_DD_LDDEC<0x0A, "mula.dd.lh.lddec">;
def MULA_DD_HH_LDDEC : MULA_DD_LDDEC<0x0B, "mula.dd.hh.lddec">;

class MULA_DD_LDINC<bits<4> oper1, string instrAsm>
  : RRR_Inst<0x04, oper1, 0x00, (outs MR:$w, AR:$d), (ins AR:$s, MR01:$x, MR23:$y),
             instrAsm#"\t $w, $s, $x, $y", []>, Requires<[HasMAC16]> {
  bits<2> x;
  bits<2> y;
  bits<2> w;

  let Constraints = "$s = $d";
  let mayLoad = 1;
  let r{3} = 0;
  let r{2} = x{0};
  let r{1-0} = w{1-0};
  let t{3} = 0;
  let t{2} = y{0};
  let t{1-0} = 0;
  let Uses = [ACCLO, ACCHI];
  let Defs = [M1, M2, ACCLO, ACCHI];
}

def MULA_DD_LL_LDINC : MULA_DD_LDINC<0x08, "mula.dd.ll.ldinc">;
def MULA_DD_HL_LDINC : MULA_DD_LDINC<0x09, "mula.dd.hl.ldinc">;
def MULA_DD_LH_LDINC : MULA_DD_LDINC<0x0A, "mula.dd.lh.ldinc">;
def MULA_DD_HH_LDINC : MULA_DD_LDINC<0x0B, "mula.dd.hh.ldinc">;

def LDDEC : RRR_Inst<0x04, 0x00, 0x09, (outs MR:$w, AR:$d), (ins AR:$s),
                    "lddec\t $w, $s", []>, Requires<[HasMAC16]> {
  bits<2> w;

  let Constraints = "$s = $d";
  let mayLoad = 1;
  let r{3-2} = 0;
  let r{1-0} = w{1-0};
  let t = 0x00;
}

def LDINC : RRR_Inst<0x04, 0x00, 0x08, (outs MR:$w, AR:$d), (ins AR:$s),
                    "ldinc\t $w, $s", []>, Requires<[HasMAC16]> {
  bits<2> w;

  let Constraints = "$s = $d";
  let mayLoad = 1;
  let r{3-2} = 0;
  let r{1-0} = w{1-0};
  let t = 0;
}

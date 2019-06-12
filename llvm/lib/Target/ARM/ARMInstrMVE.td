//===-- ARMInstrMVE.td - MVE support for ARM ---------------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file describes the ARM MVE instruction set.
//
//===----------------------------------------------------------------------===//

class MVE_MI_with_pred<dag oops, dag iops, InstrItinClass itin, string asm,
                       string ops, string cstr, list<dag> pattern>
  : Thumb2I<oops, iops, AddrModeNone, 4, itin, asm, !strconcat("\t", ops), cstr,
             pattern>,
    Requires<[HasV8_1MMainline, HasMVEInt]> {
  let D = MVEDomain;
  let DecoderNamespace = "MVE";
}

class t2MVEShift<string iname, dag oops, dag iops, string asm, string cstr,
            list<dag> pattern=[]>
  : MVE_MI_with_pred<oops, iops, NoItinerary, iname, asm, cstr, pattern> {
  let Inst{31-20} = 0b111010100101;
  let Inst{8} = 0b1;

}

class t2MVEShiftSingleReg<string iname, dag iops, string asm, string cstr,
                    list<dag> pattern=[]>
  : t2MVEShift<iname, (outs rGPR:$RdaDest), iops, asm, cstr, pattern> {
  bits<4> RdaDest;

  let Inst{19-16} = RdaDest{3-0};
}

class t2MVEShiftSRegImm<string iname, bits<2> op5_4, list<dag> pattern=[]>
  : t2MVEShiftSingleReg<iname, (ins rGPR:$RdaSrc, long_shift:$imm),
                     "$RdaSrc, $imm", "$RdaDest = $RdaSrc", pattern> {
  bits<5> imm;

  let Inst{15} = 0b0;
  let Inst{14-12} = imm{4-2};
  let Inst{11-8} = 0b1111;
  let Inst{7-6} = imm{1-0};
  let Inst{5-4} = op5_4{1-0};
  let Inst{3-0} = 0b1111;
}

def t2SQSHL : t2MVEShiftSRegImm<"sqshl", 0b11>;
def t2SRSHR : t2MVEShiftSRegImm<"srshr", 0b10>;
def t2UQSHL : t2MVEShiftSRegImm<"uqshl", 0b00>;
def t2URSHR : t2MVEShiftSRegImm<"urshr", 0b01>;

class t2MVEShiftSRegReg<string iname, bits<2> op5_4, list<dag> pattern=[]>
  : t2MVEShiftSingleReg<iname, (ins rGPR:$RdaSrc, rGPR:$Rm),
                     "$RdaSrc, $Rm", "$RdaDest = $RdaSrc", pattern> {
  bits<4> Rm;

  let Inst{15-12} = Rm{3-0};
  let Inst{11-8} = 0b1111;
  let Inst{7-6} = 0b00;
  let Inst{5-4} = op5_4{1-0};
  let Inst{3-0} = 0b1101;
}

def t2SQRSHR : t2MVEShiftSRegReg<"sqrshr", 0b10>;
def t2UQRSHL : t2MVEShiftSRegReg<"uqrshl", 0b00>;

class t2MVEShiftDoubleReg<string iname, dag iops, string asm, string cstr,
                       list<dag> pattern=[]>
  : t2MVEShift<iname, (outs tGPREven:$RdaLo, tGPROdd:$RdaHi), iops, asm, cstr,
            pattern> {
  bits<4> RdaLo;
  bits<4> RdaHi;

  let Inst{19-17} = RdaLo{3-1};
  let Inst{11-9} = RdaHi{3-1};
}

class t2MVEShiftDRegImm<string iname, bits<2> op5_4, bit op16, list<dag> pattern=[]>
  : t2MVEShiftDoubleReg<iname, (ins tGPREven:$RdaLo_src, tGPROdd:$RdaHi_src,
                        long_shift:$imm), "$RdaLo, $RdaHi, $imm",
                        "$RdaLo = $RdaLo_src,$RdaHi = $RdaHi_src", pattern> {
  bits<5> imm;

  let Inst{16} = op16;
  let Inst{15} = 0b0;
  let Inst{14-12} = imm{4-2};
  let Inst{7-6} = imm{1-0};
  let Inst{5-4} = op5_4{1-0};
  let Inst{3-0} = 0b1111;
}

class t2MVEShiftDRegReg<string iname, bit op5, bit op16, list<dag> pattern=[]>
  : t2MVEShiftDoubleReg<iname, (ins tGPREven:$RdaLo_src, tGPROdd:$RdaHi_src, rGPR:$Rm),
                    "$RdaLo, $RdaHi, $Rm",
                    "@earlyclobber $RdaHi,@earlyclobber $RdaLo,$RdaLo = $RdaLo_src,$RdaHi = $RdaHi_src",
                    pattern> {
  bits<4> Rm;

  let Inst{16} = op16;
  let Inst{15-12} = Rm{3-0};
  let Inst{7-6} = 0b00;
  let Inst{5} = op5;
  let Inst{4} = 0b0;
  let Inst{3-0} = 0b1101;

  // Custom decoder method because of the following overlapping encodings:
  // ASRL and SQRSHR
  // LSLL and UQRSHL
  // SQRSHRL and SQRSHR
  // UQRSHLL and UQRSHL
  let DecoderMethod = "DecodeMVEOverlappingLongShift";
}

def t2ASRLr   : t2MVEShiftDRegReg<"asrl",    0b1,  0b0>;
def t2ASRLi   : t2MVEShiftDRegImm<"asrl",    0b10, ?>;
def t2LSLLr   : t2MVEShiftDRegReg<"lsll",    0b0,  0b0>;
def t2LSLLi   : t2MVEShiftDRegImm<"lsll",    0b00, ?>;
def t2LSRL    : t2MVEShiftDRegImm<"lsrl",    0b01, ?>;

def t2SQRSHRL : t2MVEShiftDRegReg<"sqrshrl", 0b1,  0b1>;
def t2SQSHLL  : t2MVEShiftDRegImm<"sqshll",  0b11, 0b1>;
def t2SRSHRL  : t2MVEShiftDRegImm<"srshrl",  0b10, 0b1>;

def t2UQRSHLL : t2MVEShiftDRegReg<"uqrshll", 0b0,  0b1>;
def t2UQSHLL  : t2MVEShiftDRegImm<"uqshll",  0b00, 0b1>;
def t2URSHRL  : t2MVEShiftDRegImm<"urshrl",  0b01, 0b1>;

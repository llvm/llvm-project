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

// VPT condition mask
def vpt_mask : Operand<i32> {
  let PrintMethod = "printVPTMask";
  let ParserMatchClass = it_mask_asmoperand;
  let EncoderMethod = "getVPTMaskOpValue";
  let DecoderMethod = "DecodeVPTMaskOperand";
}

// VPT/VCMP restricted predicate for sign invariant types
def pred_restricted_i_asmoperand : AsmOperandClass {
  let Name = "CondCodeRestrictedI";
  let RenderMethod = "addITCondCodeOperands";
  let PredicateMethod = "isITCondCodeRestrictedI";
  let ParserMethod = "parseITCondCode";
}

// VPT/VCMP restricted predicate for signed types
def pred_restricted_s_asmoperand : AsmOperandClass {
  let Name = "CondCodeRestrictedS";
  let RenderMethod = "addITCondCodeOperands";
  let PredicateMethod = "isITCondCodeRestrictedS";
  let ParserMethod = "parseITCondCode";
}

// VPT/VCMP restricted predicate for unsigned types
def pred_restricted_u_asmoperand : AsmOperandClass {
  let Name = "CondCodeRestrictedU";
  let RenderMethod = "addITCondCodeOperands";
  let PredicateMethod = "isITCondCodeRestrictedU";
  let ParserMethod = "parseITCondCode";
}

// VPT/VCMP restricted predicate for floating point
def pred_restricted_fp_asmoperand : AsmOperandClass {
  let Name = "CondCodeRestrictedFP";
  let RenderMethod = "addITCondCodeOperands";
  let PredicateMethod = "isITCondCodeRestrictedFP";
  let ParserMethod = "parseITCondCode";
}

def pred_basic_i : Operand<i32> {
  let PrintMethod = "printMandatoryRestrictedPredicateOperand";
  let ParserMatchClass = pred_restricted_i_asmoperand;
  let DecoderMethod = "DecodeRestrictedIPredicateOperand";
  let EncoderMethod = "getRestrictedCondCodeOpValue";
}

def pred_basic_u : Operand<i32> {
  let PrintMethod = "printMandatoryRestrictedPredicateOperand";
  let ParserMatchClass = pred_restricted_u_asmoperand;
  let DecoderMethod = "DecodeRestrictedUPredicateOperand";
  let EncoderMethod = "getRestrictedCondCodeOpValue";
}

def pred_basic_s : Operand<i32> {
  let PrintMethod = "printMandatoryRestrictedPredicateOperand";
  let ParserMatchClass = pred_restricted_s_asmoperand;
  let DecoderMethod = "DecodeRestrictedSPredicateOperand";
  let EncoderMethod = "getRestrictedCondCodeOpValue";
}

def pred_basic_fp : Operand<i32> {
  let PrintMethod = "printMandatoryRestrictedPredicateOperand";
  let ParserMatchClass = pred_restricted_fp_asmoperand;
  let DecoderMethod = "DecodeRestrictedFPPredicateOperand";
  let EncoderMethod = "getRestrictedCondCodeOpValue";
}

class MVE_MI<dag oops, dag iops, InstrItinClass itin, string asm,
             string ops, string cstr, list<dag> pattern>
  : Thumb2XI<oops, iops, AddrModeNone, 4, itin, !strconcat(asm, "\t", ops), cstr,
             pattern>,
    Requires<[HasMVEInt]> {
  let D = MVEDomain;
  let DecoderNamespace = "MVE";
}

// MVE_p is used for most predicated instructions, to add the cluster
// of input operands that provides the VPT suffix (none, T or E) and
// the input predicate register.
class MVE_p<dag oops, dag iops, InstrItinClass itin, string iname,
            string suffix, string ops, vpred_ops vpred, string cstr,
            list<dag> pattern=[]>
  : MVE_MI<oops, !con(iops, (ins vpred:$vp)), itin,
           // If the instruction has a suffix, like vadd.f32, then the
           // VPT predication suffix goes before the dot, so the full
           // name has to be "vadd${vp}.f32".
           !strconcat(iname, "${vp}",
                      !if(!eq(suffix, ""), "", !strconcat(".", suffix))),
           ops, !strconcat(cstr, vpred.vpred_constraint), pattern> {
  let Inst{31-29} = 0b111;
  let Inst{27-26} = 0b11;
}

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

// start of mve_rDest instructions

class MVE_rDest<dag oops, dag iops, InstrItinClass itin, string iname, string suffix,
                string ops, string cstr, list<dag> pattern=[]>
// Always use vpred_n and not vpred_r: with the output register being
// a GPR and not a vector register, there can't be any question of
// what to put in its inactive lanes.
  : MVE_p<oops, iops, itin, iname, suffix, ops, vpred_n, cstr, pattern> {

  let Inst{25-23} = 0b101;
  let Inst{11-9} = 0b111;
  let Inst{4} = 0b0;
}

class t2VABAV<string suffix, bit U, bits<2> size, list<dag> pattern=[]>
  : MVE_rDest<(outs rGPR:$Rda), (ins rGPR:$Rda_src, MQPR:$Qn, MQPR:$Qm),
              NoItinerary, "vabav", suffix, "$Rda, $Qn, $Qm", "$Rda = $Rda_src",
              pattern> {
  bits<4> Qm;
  bits<4> Qn;
  bits<4> Rda;

  let Inst{28} = U;
  let Inst{22} = 0b0;
  let Inst{21-20} = size{1-0};
  let Inst{19-17} = Qn{2-0};
  let Inst{16} = 0b0;
  let Inst{15-12} = Rda{3-0};
  let Inst{8} = 0b1;
  let Inst{7} = Qn{3};
  let Inst{6} = 0b0;
  let Inst{5} = Qm{3};
  let Inst{3-1} = Qm{2-0};
  let Inst{0} = 0b1;
}

def VABAVs8  : t2VABAV<"s8", 0b0, 0b00>;
def VABAVs16 : t2VABAV<"s16", 0b0, 0b01>;
def VABAVs32 : t2VABAV<"s32", 0b0, 0b10>;
def VABAVu8  : t2VABAV<"u8", 0b1, 0b00>;
def VABAVu16 : t2VABAV<"u16", 0b1, 0b01>;
def VABAVu32 : t2VABAV<"u32", 0b1, 0b10>;

// end of mve_rDest instructions

// start of mve_comp instructions

class MVE_comp<InstrItinClass itin, string iname, string suffix,
               string cstr, list<dag> pattern=[]>
  : MVE_p<(outs MQPR:$Qd), (ins MQPR:$Qn, MQPR:$Qm), itin, iname, suffix,
           "$Qd, $Qn, $Qm", vpred_r, cstr, pattern> {
  bits<4> Qd;
  bits<4> Qn;
  bits<4> Qm;

  let Inst{22} = Qd{3};
  let Inst{19-17} = Qn{2-0};
  let Inst{16} = 0b0;
  let Inst{15-13} = Qd{2-0};
  let Inst{12} = 0b0;
  let Inst{10-9} = 0b11;
  let Inst{7} = Qn{3};
  let Inst{5} = Qm{3};
  let Inst{3-1} = Qm{2-0};
  let Inst{0} = 0b0;
}

class VMINMAXNM<string iname, string suffix, bit sz, bit bit_21,
                list<dag> pattern=[]>
  : MVE_comp<NoItinerary, iname, suffix, "", pattern> {

  let Inst{28} = 0b1;
  let Inst{25-24} = 0b11;
  let Inst{23} = 0b0;
  let Inst{21} = bit_21;
  let Inst{20} = sz;
  let Inst{11} = 0b1;
  let Inst{8} = 0b1;
  let Inst{6} = 0b1;
  let Inst{4} = 0b1;

  let Predicates = [HasMVEFloat];
}

def VMAXNMf32 : VMINMAXNM<"vmaxnm", "f32", 0b0, 0b0>;
def VMAXNMf16 : VMINMAXNM<"vmaxnm", "f16", 0b1, 0b0>;

def VMINNMf32 : VMINMAXNM<"vminnm", "f32", 0b0, 0b1>;
def VMINNMf16 : VMINMAXNM<"vminnm", "f16", 0b1, 0b1>;

// end of mve_comp instructions

class t2VPT<string suffix, bits<2> size, dag iops, string asm, list<dag> pattern=[]>
  : MVE_MI<(outs ), iops, NoItinerary, !strconcat("vpt", "${Mk}", ".", suffix), asm, "", pattern> {
  bits<3> fc;
  bits<4> Mk;
  bits<3> Qn;

  let Inst{31-23} = 0b111111100;
  let Inst{22} = Mk{3};
  let Inst{21-20} = size;
  let Inst{19-17} = Qn{2-0};
  let Inst{16} = 0b1;
  let Inst{15-13} = Mk{2-0};
  let Inst{12} = fc{2};
  let Inst{11-8} = 0b1111;
  let Inst{7} = fc{0};
  let Inst{4} = 0b0;

  let Defs = [VPR, P0];
}

class t2VPTt1<string suffix, bits<2> size, dag iops>
  : t2VPT<suffix, size, iops, "$fc, $Qn, $Qm"> {
  bits<4> Qm;
  bits<4> Mk;

  let Inst{6} = 0b0;
  let Inst{5} = Qm{3};
  let Inst{3-1} = Qm{2-0};
  let Inst{0} = fc{1};
}

class t2VPTt1i<string suffix, bits<2> size>
 : t2VPTt1<suffix, size,
           (ins vpt_mask:$Mk, pred_basic_i:$fc, MQPR:$Qn, MQPR:$Qm)> {
  let Inst{12} = 0b0;
  let Inst{0} = 0b0;
}

def t2VPTv4i32 : t2VPTt1i<"i32", 0b10>;
def t2VPTv8i16 : t2VPTt1i<"i16", 0b01>;
def t2VPTv16i8 : t2VPTt1i<"i8", 0b00>;

class t2VPTt1u<string suffix, bits<2> size>
 : t2VPTt1<suffix, size,
           (ins vpt_mask:$Mk, pred_basic_u:$fc, MQPR:$Qn, MQPR:$Qm)> {
  let Inst{12} = 0b0;
  let Inst{0} = 0b1;
}

def t2VPTv4u32 : t2VPTt1u<"u32", 0b10>;
def t2VPTv8u16 : t2VPTt1u<"u16", 0b01>;
def t2VPTv16u8 : t2VPTt1u<"u8", 0b00>;

class t2VPTt1s<string suffix, bits<2> size>
 : t2VPTt1<suffix, size,
           (ins vpt_mask:$Mk, pred_basic_s:$fc, MQPR:$Qn, MQPR:$Qm)> {
  let Inst{12} = 0b1;
}

def t2VPTv4s32 : t2VPTt1s<"s32", 0b10>;
def t2VPTv8s16 : t2VPTt1s<"s16", 0b01>;
def t2VPTv16s8 : t2VPTt1s<"s8", 0b00>;

class t2VPTt2<string suffix, bits<2> size, dag iops>
  : t2VPT<suffix, size, iops,
          "$fc, $Qn, $Rm"> {
  bits<4> Rm;
  bits<3> fc;
  bits<4> Mk;

  let Inst{6} = 0b1;
  let Inst{5} = fc{1};
  let Inst{3-0} = Rm{3-0};
}

class t2VPTt2i<string suffix, bits<2> size>
  : t2VPTt2<suffix, size,
            (ins vpt_mask:$Mk, pred_basic_i:$fc, MQPR:$Qn, GPRwithZR:$Rm)> {
  let Inst{12} = 0b0;
  let Inst{5} = 0b0;
}

def t2VPTv4i32r : t2VPTt2i<"i32", 0b10>;
def t2VPTv8i16r : t2VPTt2i<"i16", 0b01>;
def t2VPTv16i8r : t2VPTt2i<"i8", 0b00>;

class t2VPTt2u<string suffix, bits<2> size>
  : t2VPTt2<suffix, size,
            (ins vpt_mask:$Mk, pred_basic_u:$fc, MQPR:$Qn, GPRwithZR:$Rm)> {
  let Inst{12} = 0b0;
  let Inst{5} = 0b1;
}

def t2VPTv4u32r : t2VPTt2u<"u32", 0b10>;
def t2VPTv8u16r : t2VPTt2u<"u16", 0b01>;
def t2VPTv16u8r : t2VPTt2u<"u8", 0b00>;

class t2VPTt2s<string suffix, bits<2> size>
  : t2VPTt2<suffix, size,
            (ins vpt_mask:$Mk, pred_basic_s:$fc, MQPR:$Qn, GPRwithZR:$Rm)> {
  let Inst{12} = 0b1;
}

def t2VPTv4s32r : t2VPTt2s<"s32", 0b10>;
def t2VPTv8s16r : t2VPTt2s<"s16", 0b01>;
def t2VPTv16s8r : t2VPTt2s<"s8", 0b00>;


class t2VPTf<string suffix, bit size, dag iops, string asm, list<dag> pattern=[]>
  : MVE_MI<(outs ), iops, NoItinerary, !strconcat("vpt", "${Mk}", ".", suffix), asm,
            "", pattern> {
  bits<3> fc;
  bits<4> Mk;
  bits<3> Qn;

  let Inst{31-29} = 0b111;
  let Inst{28} = size;
  let Inst{27-23} = 0b11100;
  let Inst{22} = Mk{3};
  let Inst{21-20} = 0b11;
  let Inst{19-17} = Qn{2-0};
  let Inst{16} = 0b1;
  let Inst{15-13} = Mk{2-0};
  let Inst{12} = fc{2};
  let Inst{11-8} = 0b1111;
  let Inst{7} = fc{0};
  let Inst{4} = 0b0;

  let Defs = [P0];
  let Predicates = [HasMVEFloat];
}

class t2VPTft1<string suffix, bit size>
  : t2VPTf<suffix, size, (ins vpt_mask:$Mk, pred_basic_fp:$fc, MQPR:$Qn, MQPR:$Qm),
          "$fc, $Qn, $Qm"> {
  bits<3> fc;
  bits<4> Qm;

  let Inst{6} = 0b0;
  let Inst{5} = Qm{3};
  let Inst{3-1} = Qm{2-0};
  let Inst{0} = fc{1};
}

def t2VPTv4f32         : t2VPTft1<"f32", 0b0>;
def t2VPTv8f16         : t2VPTft1<"f16", 0b1>;

class t2VPTft2<string suffix, bit size>
  : t2VPTf<suffix, size, (ins vpt_mask:$Mk, pred_basic_fp:$fc, MQPR:$Qn, GPRwithZR:$Rm),
          "$fc, $Qn, $Rm"> {
  bits<3> fc;
  bits<4> Rm;

  let Inst{6} = 0b1;
  let Inst{5} = fc{1};
  let Inst{3-0} = Rm{3-0};
}

def t2VPTv4f32r        : t2VPTft2<"f32", 0b0>;
def t2VPTv8f16r        : t2VPTft2<"f16", 0b1>;

def t2VPST : MVE_MI<(outs ), (ins vpt_mask:$Mk), NoItinerary,
       !strconcat("vpst", "${Mk}"), "", "", []> {
  bits<4> Mk;

  let Inst{31-23} = 0b111111100;
  let Inst{22} = Mk{3};
  let Inst{21-16} = 0b110001;
  let Inst{15-13} = Mk{2-0};
  let Inst{12-0} = 0b0111101001101;
  let Unpredictable{12} = 0b1;
  let Unpredictable{7} = 0b1;
  let Unpredictable{5} = 0b1;

  let Defs = [P0];
}

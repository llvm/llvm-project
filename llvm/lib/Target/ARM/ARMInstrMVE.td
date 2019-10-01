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

class MVE_ScalarShift<string iname, dag oops, dag iops, string asm, string cstr,
            list<dag> pattern=[]>
  : MVE_MI_with_pred<oops, iops, NoItinerary, iname, asm, cstr, pattern> {
  let Inst{31-20} = 0b111010100101;
  let Inst{8} = 0b1;

}

class MVE_ScalarShiftSingleReg<string iname, dag iops, string asm, string cstr,
                    list<dag> pattern=[]>
  : MVE_ScalarShift<iname, (outs rGPR:$RdaDest), iops, asm, cstr, pattern> {
  bits<4> RdaDest;

  let Inst{19-16} = RdaDest{3-0};
}

class MVE_ScalarShiftSRegImm<string iname, bits<2> op5_4, list<dag> pattern=[]>
  : MVE_ScalarShiftSingleReg<iname, (ins rGPR:$RdaSrc, long_shift:$imm),
                     "$RdaSrc, $imm", "$RdaDest = $RdaSrc", pattern> {
  bits<5> imm;

  let Inst{15} = 0b0;
  let Inst{14-12} = imm{4-2};
  let Inst{11-8} = 0b1111;
  let Inst{7-6} = imm{1-0};
  let Inst{5-4} = op5_4{1-0};
  let Inst{3-0} = 0b1111;
}

def MVE_SQSHL : MVE_ScalarShiftSRegImm<"sqshl", 0b11>;
def MVE_SRSHR : MVE_ScalarShiftSRegImm<"srshr", 0b10>;
def MVE_UQSHL : MVE_ScalarShiftSRegImm<"uqshl", 0b00>;
def MVE_URSHR : MVE_ScalarShiftSRegImm<"urshr", 0b01>;

class MVE_ScalarShiftSRegReg<string iname, bits<2> op5_4, list<dag> pattern=[]>
  : MVE_ScalarShiftSingleReg<iname, (ins rGPR:$RdaSrc, rGPR:$Rm),
                     "$RdaSrc, $Rm", "$RdaDest = $RdaSrc", pattern> {
  bits<4> Rm;

  let Inst{15-12} = Rm{3-0};
  let Inst{11-8} = 0b1111;
  let Inst{7-6} = 0b00;
  let Inst{5-4} = op5_4{1-0};
  let Inst{3-0} = 0b1101;
}

def MVE_SQRSHR : MVE_ScalarShiftSRegReg<"sqrshr", 0b10>;
def MVE_UQRSHL : MVE_ScalarShiftSRegReg<"uqrshl", 0b00>;

class MVE_ScalarShiftDoubleReg<string iname, dag iops, string asm,
                               string cstr, list<dag> pattern=[]>
  : MVE_ScalarShift<iname, (outs tGPREven:$RdaLo, tGPROdd:$RdaHi),
                    iops, asm, cstr, pattern> {
  bits<4> RdaLo;
  bits<4> RdaHi;

  let Inst{19-17} = RdaLo{3-1};
  let Inst{11-9} = RdaHi{3-1};
}

class MVE_ScalarShiftDRegImm<string iname, bits<2> op5_4, bit op16,
                             list<dag> pattern=[]>
  : MVE_ScalarShiftDoubleReg<
      iname, (ins tGPREven:$RdaLo_src, tGPROdd:$RdaHi_src, long_shift:$imm),
      "$RdaLo, $RdaHi, $imm", "$RdaLo = $RdaLo_src,$RdaHi = $RdaHi_src",
      pattern> {
  bits<5> imm;

  let Inst{16} = op16;
  let Inst{15} = 0b0;
  let Inst{14-12} = imm{4-2};
  let Inst{7-6} = imm{1-0};
  let Inst{5-4} = op5_4{1-0};
  let Inst{3-0} = 0b1111;
}

class MVE_ScalarShiftDRegReg<string iname, bit op5, bit op16,
                             list<dag> pattern=[]>
  : MVE_ScalarShiftDoubleReg<
     iname, (ins tGPREven:$RdaLo_src, tGPROdd:$RdaHi_src, rGPR:$Rm),
     "$RdaLo, $RdaHi, $Rm", "@earlyclobber $RdaHi,@earlyclobber $RdaLo,"
                            "$RdaLo = $RdaLo_src,$RdaHi = $RdaHi_src",
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

def MVE_ASRLr   : MVE_ScalarShiftDRegReg<"asrl",    0b1,  0b0>;
def MVE_ASRLi   : MVE_ScalarShiftDRegImm<"asrl",    0b10, ?>;
def MVE_LSLLr   : MVE_ScalarShiftDRegReg<"lsll",    0b0,  0b0>;
def MVE_LSLLi   : MVE_ScalarShiftDRegImm<"lsll",    0b00, ?>;
def MVE_LSRL    : MVE_ScalarShiftDRegImm<"lsrl",    0b01, ?>;

def MVE_SQRSHRL : MVE_ScalarShiftDRegReg<"sqrshrl", 0b1,  0b1>;
def MVE_SQSHLL  : MVE_ScalarShiftDRegImm<"sqshll",  0b11, 0b1>;
def MVE_SRSHRL  : MVE_ScalarShiftDRegImm<"srshrl",  0b10, 0b1>;

def MVE_UQRSHLL : MVE_ScalarShiftDRegReg<"uqrshll", 0b0,  0b1>;
def MVE_UQSHLL  : MVE_ScalarShiftDRegImm<"uqshll",  0b00, 0b1>;
def MVE_URSHRL  : MVE_ScalarShiftDRegImm<"urshrl",  0b01, 0b1>;

// start of mve_rDest instructions

class MVE_rDest<dag oops, dag iops, InstrItinClass itin,
                string iname, string suffix,
                string ops, string cstr, list<dag> pattern=[]>
// Always use vpred_n and not vpred_r: with the output register being
// a GPR and not a vector register, there can't be any question of
// what to put in its inactive lanes.
  : MVE_p<oops, iops, itin, iname, suffix, ops, vpred_n, cstr, pattern> {

  let Inst{25-23} = 0b101;
  let Inst{11-9} = 0b111;
  let Inst{4} = 0b0;
}

class MVE_VABAV<string suffix, bit U, bits<2> size, list<dag> pattern=[]>
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

def MVE_VABAVs8  : MVE_VABAV<"s8", 0b0, 0b00>;
def MVE_VABAVs16 : MVE_VABAV<"s16", 0b0, 0b01>;
def MVE_VABAVs32 : MVE_VABAV<"s32", 0b0, 0b10>;
def MVE_VABAVu8  : MVE_VABAV<"u8", 0b1, 0b00>;
def MVE_VABAVu16 : MVE_VABAV<"u16", 0b1, 0b01>;
def MVE_VABAVu32 : MVE_VABAV<"u32", 0b1, 0b10>;

class MVE_VADDV<string iname, string suffix, dag iops, string cstr,
              bit A, bit U, bits<2> size, list<dag> pattern=[]>
  : MVE_rDest<(outs tGPREven:$Rda), iops, NoItinerary,
              iname, suffix, "$Rda, $Qm", cstr, pattern> {
  bits<3> Qm;
  bits<4> Rda;

  let Inst{28} = U;
  let Inst{22-20} = 0b111;
  let Inst{19-18} = size{1-0};
  let Inst{17-16} = 0b01;
  let Inst{15-13} = Rda{3-1};
  let Inst{12} = 0b0;
  let Inst{8-6} = 0b100;
  let Inst{5} = A;
  let Inst{3-1} = Qm{2-0};
  let Inst{0} = 0b0;
}

multiclass MVE_VADDV_A<string suffix, bit U, bits<2> size,
                       list<dag> pattern=[]> {
  def acc    : MVE_VADDV<"vaddva", suffix,
                         (ins tGPREven:$Rda_src, MQPR:$Qm), "$Rda = $Rda_src",
                         0b1, U, size, pattern>;
  def no_acc : MVE_VADDV<"vaddv", suffix,
                         (ins MQPR:$Qm), "",
                         0b0, U, size, pattern>;
}

defm MVE_VADDVs8  : MVE_VADDV_A<"s8",  0b0, 0b00>;
defm MVE_VADDVs16 : MVE_VADDV_A<"s16", 0b0, 0b01>;
defm MVE_VADDVs32 : MVE_VADDV_A<"s32", 0b0, 0b10>;
defm MVE_VADDVu8  : MVE_VADDV_A<"u8",  0b1, 0b00>;
defm MVE_VADDVu16 : MVE_VADDV_A<"u16", 0b1, 0b01>;
defm MVE_VADDVu32 : MVE_VADDV_A<"u32", 0b1, 0b10>;

class MVE_VADDLV<string iname, string suffix, dag iops, string cstr,
               bit A, bit U, list<dag> pattern=[]>
  : MVE_rDest<(outs tGPREven:$RdaLo, tGPROdd:$RdaHi), iops, NoItinerary, iname,
              suffix, "$RdaLo, $RdaHi, $Qm", cstr, pattern> {
  bits<3> Qm;
  bits<4> RdaLo;
  bits<4> RdaHi;

  let Inst{28} = U;
  let Inst{22-20} = RdaHi{3-1};
  let Inst{19-18} = 0b10;
  let Inst{17-16} = 0b01;
  let Inst{15-13} = RdaLo{3-1};
  let Inst{12} = 0b0;
  let Inst{8-6} = 0b100;
  let Inst{5} = A;
  let Inst{3-1} = Qm{2-0};
  let Inst{0} = 0b0;
}

multiclass MVE_VADDLV_A<string suffix, bit U, list<dag> pattern=[]> {
  def acc    : MVE_VADDLV<"vaddlva", suffix,
                        (ins tGPREven:$RdaLo_src, tGPROdd:$RdaHi_src, MQPR:$Qm),
                        "$RdaLo = $RdaLo_src,$RdaHi = $RdaHi_src",
                        0b1, U, pattern>;
  def no_acc : MVE_VADDLV<"vaddlv", suffix,
                        (ins MQPR:$Qm), "",
                        0b0, U, pattern>;
}


defm MVE_VADDLVs32 : MVE_VADDLV_A<"s32", 0b0>;
defm MVE_VADDLVu32 : MVE_VADDLV_A<"u32", 0b1>;

class MVE_VMINMAXNMV<string iname, string suffix, bit sz,
                     bit bit_17, bit bit_7, list<dag> pattern=[]>
  : MVE_rDest<(outs rGPR:$RdaDest), (ins rGPR:$RdaSrc, MQPR:$Qm),
              NoItinerary, iname, suffix, "$RdaSrc, $Qm",
              "$RdaDest = $RdaSrc", pattern> {
  bits<3> Qm;
  bits<4> RdaDest;

  let Inst{28} = sz;
  let Inst{22-20} = 0b110;
  let Inst{19-18} = 0b11;
  let Inst{17} = bit_17;
  let Inst{16} = 0b0;
  let Inst{15-12} = RdaDest{3-0};
  let Inst{8} = 0b1;
  let Inst{7} = bit_7;
  let Inst{6-5} = 0b00;
  let Inst{3-1} = Qm{2-0};
  let Inst{0} = 0b0;

  let Predicates = [HasMVEFloat];
}

multiclass MVE_VMINMAXNMV_fty<string iname, bit bit_7, list<dag> pattern=[]> {
  def f32 : MVE_VMINMAXNMV<iname, "f32", 0b0, 0b1, bit_7, pattern>;
  def f16 : MVE_VMINMAXNMV<iname, "f16", 0b1, 0b1, bit_7, pattern>;
}

defm MVE_VMINNMV : MVE_VMINMAXNMV_fty<"vminnmv", 0b1>;
defm MVE_VMAXNMV : MVE_VMINMAXNMV_fty<"vmaxnmv", 0b0>;

multiclass MVE_VMINMAXNMAV_fty<string iname, bit bit_7, list<dag> pattern=[]> {
  def f32 : MVE_VMINMAXNMV<iname, "f32", 0b0, 0b0, bit_7, pattern>;
  def f16 : MVE_VMINMAXNMV<iname, "f16", 0b1, 0b0, bit_7, pattern>;
}

defm MVE_VMINNMAV : MVE_VMINMAXNMAV_fty<"vminnmav", 0b1>;
defm MVE_VMAXNMAV : MVE_VMINMAXNMAV_fty<"vmaxnmav", 0b0>;

class MVE_VMINMAXV<string iname, string suffix, bit U, bits<2> size,
                 bit bit_17, bit bit_7, list<dag> pattern=[]>
  : MVE_rDest<(outs rGPR:$RdaDest), (ins rGPR:$RdaSrc, MQPR:$Qm), NoItinerary,
              iname, suffix, "$RdaSrc, $Qm", "$RdaDest = $RdaSrc", pattern> {
  bits<3> Qm;
  bits<4> RdaDest;

  let Inst{28} = U;
  let Inst{22-20} = 0b110;
  let Inst{19-18} = size{1-0};
  let Inst{17} = bit_17;
  let Inst{16} = 0b0;
  let Inst{15-12} = RdaDest{3-0};
  let Inst{8} = 0b1;
  let Inst{7} = bit_7;
  let Inst{6-5} = 0b00;
  let Inst{3-1} = Qm{2-0};
  let Inst{0} = 0b0;
}

multiclass MVE_VMINMAXV_ty<string iname, bit bit_7, list<dag> pattern=[]> {
  def s8  : MVE_VMINMAXV<iname, "s8",  0b0, 0b00, 0b1, bit_7>;
  def s16 : MVE_VMINMAXV<iname, "s16", 0b0, 0b01, 0b1, bit_7>;
  def s32 : MVE_VMINMAXV<iname, "s32", 0b0, 0b10, 0b1, bit_7>;
  def u8  : MVE_VMINMAXV<iname, "u8",  0b1, 0b00, 0b1, bit_7>;
  def u16 : MVE_VMINMAXV<iname, "u16", 0b1, 0b01, 0b1, bit_7>;
  def u32 : MVE_VMINMAXV<iname, "u32", 0b1, 0b10, 0b1, bit_7>;
}

defm MVE_VMINV : MVE_VMINMAXV_ty<"vminv", 0b1>;
defm MVE_VMAXV : MVE_VMINMAXV_ty<"vmaxv", 0b0>;

multiclass MVE_VMINMAXAV_ty<string iname, bit bit_7, list<dag> pattern=[]> {
  def s8  : MVE_VMINMAXV<iname, "s8",  0b0, 0b00, 0b0, bit_7>;
  def s16 : MVE_VMINMAXV<iname, "s16", 0b0, 0b01, 0b0, bit_7>;
  def s32 : MVE_VMINMAXV<iname, "s32", 0b0, 0b10, 0b0, bit_7>;
}

defm MVE_VMINAV : MVE_VMINMAXAV_ty<"vminav", 0b1>;
defm MVE_VMAXAV : MVE_VMINMAXAV_ty<"vmaxav", 0b0>;

class MVE_VMLAMLSDAV<string iname, string suffix, dag iops, string cstr,
                   bit sz, bit bit_28, bit A, bit X, bit bit_8, bit bit_0,
                   list<dag> pattern=[]>
  : MVE_rDest<(outs tGPREven:$RdaDest), iops, NoItinerary, iname, suffix,
              "$RdaDest, $Qn, $Qm", cstr, pattern> {
  bits<4> RdaDest;
  bits<3> Qm;
  bits<3> Qn;

  let Inst{28} = bit_28;
  let Inst{22-20} = 0b111;
  let Inst{19-17} = Qn{2-0};
  let Inst{16} = sz;
  let Inst{15-13} = RdaDest{3-1};
  let Inst{12} = X;
  let Inst{8} = bit_8;
  let Inst{7-6} = 0b00;
  let Inst{5} = A;
  let Inst{3-1} = Qm{2-0};
  let Inst{0} = bit_0;
}

multiclass MVE_VMLAMLSDAV_X<string iname, string suffix, dag iops, string cstr,
                          bit sz, bit bit_28, bit A, bit bit_8, bit bit_0,
                          list<dag> pattern=[]> {
  def _noexch : MVE_VMLAMLSDAV<iname, suffix, iops, cstr, sz,
                            bit_28, A, 0b0, bit_8, bit_0, pattern>;
  def _exch   : MVE_VMLAMLSDAV<iname # "x", suffix, iops, cstr, sz,
                            bit_28, A, 0b1, bit_8, bit_0, pattern>;
}

multiclass MVE_VMLAMLSDAV_XA<string iname, string suffix, bit sz, bit bit_28,
                           bit bit_8, bit bit_0, list<dag> pattern=[]> {
  defm _noacc : MVE_VMLAMLSDAV_X<iname, suffix, (ins MQPR:$Qn, MQPR:$Qm), "",
                              sz, bit_28, 0b0, bit_8, bit_0, pattern>;
  defm _acc   : MVE_VMLAMLSDAV_X<iname # "a", suffix,
                             (ins tGPREven:$RdaSrc, MQPR:$Qn, MQPR:$Qm),
                             "$RdaDest = $RdaSrc",
                              sz, bit_28, 0b1, bit_8, bit_0, pattern>;
}

multiclass MVE_VMLADAV_multi<string suffix, bit sz, bit U, bit bit_8,
                           list<dag> pattern=[]> {
  defm "" : MVE_VMLAMLSDAV_XA<"vmladav", suffix, sz, U, bit_8, 0b0, pattern>;
}

defm MVE_VMLADAVs16 : MVE_VMLADAV_multi<"s16", 0b0, 0b0, 0b0>;
defm MVE_VMLADAVs32 : MVE_VMLADAV_multi<"s32", 0b1, 0b0, 0b0>;
defm MVE_VMLADAVu16 : MVE_VMLADAV_multi<"u16", 0b0, 0b1, 0b0>;
defm MVE_VMLADAVu32 : MVE_VMLADAV_multi<"u32", 0b1, 0b1, 0b0>;

defm MVE_VMLADAVs8 : MVE_VMLADAV_multi<"s8", 0b0, 0b0, 0b1>;
defm MVE_VMLADAVu8 : MVE_VMLADAV_multi<"u8", 0b0, 0b1, 0b1>;

// vmlav aliases vmladav
foreach acc = ["_acc", "_noacc"] in {
  foreach suffix = ["s8", "s16", "s32", "u8", "u16", "u32"] in {
    def : MVEInstAlias<!strconcat("vmlav", !if(!eq(acc, "_acc"), "a", ""),
                       "${vp}.", suffix, "\t$RdaDest, $Qn, $Qm"),
                       (!cast<Instruction>("MVE_VMLADAV"#suffix#acc#"_noexch")
                        tGPREven:$RdaDest, MQPR:$Qn, MQPR:$Qm, vpred_n:$vp)>;
  }
}

multiclass MVE_VMLSDAV_multi<string suffix, bit sz, bit bit_28,
                           list<dag> pattern=[]> {
  defm "" : MVE_VMLAMLSDAV_XA<"vmlsdav", suffix, sz, bit_28, 0b0, 0b1, pattern>;
}

defm MVE_VMLSDAVs8  : MVE_VMLSDAV_multi<"s8", 0, 0b1>;
defm MVE_VMLSDAVs16 : MVE_VMLSDAV_multi<"s16", 0, 0b0>;
defm MVE_VMLSDAVs32 : MVE_VMLSDAV_multi<"s32", 1, 0b0>;

// Base class for VMLALDAV and VMLSLDAV, VRMLALDAVH, VRMLSLDAVH
class MVE_VMLALDAVBase<string iname, string suffix, dag iops, string cstr,
                       bit sz, bit bit_28, bit A, bit X, bit bit_8, bit bit_0,
                       list<dag> pattern=[]>
  : MVE_rDest<(outs tGPREven:$RdaLoDest, tGPROdd:$RdaHiDest), iops, NoItinerary,
              iname, suffix, "$RdaLoDest, $RdaHiDest, $Qn, $Qm", cstr, pattern> {
  bits<4> RdaLoDest;
  bits<4> RdaHiDest;
  bits<3> Qm;
  bits<3> Qn;

  let Inst{28} = bit_28;
  let Inst{22-20} = RdaHiDest{3-1};
  let Inst{19-17} = Qn{2-0};
  let Inst{16} = sz;
  let Inst{15-13} = RdaLoDest{3-1};
  let Inst{12} = X;
  let Inst{8} = bit_8;
  let Inst{7-6} = 0b00;
  let Inst{5} = A;
  let Inst{3-1} = Qm{2-0};
  let Inst{0} = bit_0;
}

multiclass MVE_VMLALDAVBase_X<string iname, string suffix, dag iops,
                              string cstr, bit sz, bit bit_28, bit A,
                              bit bit_8, bit bit_0, list<dag> pattern=[]> {
  def _noexch : MVE_VMLALDAVBase<iname, suffix, iops, cstr, sz,
                               bit_28, A, 0b0, bit_8, bit_0, pattern>;
  def _exch   : MVE_VMLALDAVBase<iname # "x", suffix, iops, cstr, sz,
                               bit_28, A, 0b1, bit_8, bit_0, pattern>;
}

multiclass MVE_VMLALDAVBase_XA<string iname, string suffix, bit sz, bit bit_28,
                             bit bit_8, bit bit_0, list<dag> pattern=[]> {
  defm _noacc : MVE_VMLALDAVBase_X<
     iname, suffix, (ins MQPR:$Qn, MQPR:$Qm), "",
     sz, bit_28, 0b0, bit_8, bit_0, pattern>;
  defm _acc   : MVE_VMLALDAVBase_X<
     iname # "a", suffix, (ins tGPREven:$RdaLoSrc, tGPROdd:$RdaHiSrc,
                               MQPR:$Qn, MQPR:$Qm),
     "$RdaLoDest = $RdaLoSrc,$RdaHiDest = $RdaHiSrc",
     sz, bit_28, 0b1, bit_8, bit_0, pattern>;
}

multiclass MVE_VRMLALDAVH_multi<string suffix, bit U, list<dag> pattern=[]> {
  defm "" : MVE_VMLALDAVBase_XA<
     "vrmlaldavh", suffix, 0b0, U, 0b1, 0b0, pattern>;
}

defm MVE_VRMLALDAVHs32 : MVE_VRMLALDAVH_multi<"s32", 0>;
defm MVE_VRMLALDAVHu32 : MVE_VRMLALDAVH_multi<"u32", 1>;

// vrmlalvh aliases for vrmlaldavh
def : MVEInstAlias<"vrmlalvh${vp}.s32\t$RdaLo, $RdaHi, $Qn, $Qm",
                  (MVE_VRMLALDAVHs32_noacc_noexch
                   tGPREven:$RdaLo, tGPROdd:$RdaHi,
                   MQPR:$Qn, MQPR:$Qm, vpred_n:$vp)>;
def : MVEInstAlias<"vrmlalvha${vp}.s32\t$RdaLo, $RdaHi, $Qn, $Qm",
                  (MVE_VRMLALDAVHs32_acc_noexch
                   tGPREven:$RdaLo, tGPROdd:$RdaHi,
                   MQPR:$Qn, MQPR:$Qm, vpred_n:$vp)>;
def : MVEInstAlias<"vrmlalvh${vp}.u32\t$RdaLo, $RdaHi, $Qn, $Qm",
                  (MVE_VRMLALDAVHu32_noacc_noexch
                   tGPREven:$RdaLo, tGPROdd:$RdaHi,
                   MQPR:$Qn, MQPR:$Qm, vpred_n:$vp)>;
def : MVEInstAlias<"vrmlalvha${vp}.u32\t$RdaLo, $RdaHi, $Qn, $Qm",
                  (MVE_VRMLALDAVHu32_acc_noexch
                   tGPREven:$RdaLo, tGPROdd:$RdaHi,
                   MQPR:$Qn, MQPR:$Qm, vpred_n:$vp)>;

multiclass MVE_VMLALDAV_multi<string suffix, bit sz, bit U,
                              list<dag> pattern=[]> {
  defm "" : MVE_VMLALDAVBase_XA<"vmlaldav", suffix, sz, U, 0b0, 0b0, pattern>;
}

defm MVE_VMLALDAVs16 : MVE_VMLALDAV_multi<"s16", 0b0, 0b0>;
defm MVE_VMLALDAVs32 : MVE_VMLALDAV_multi<"s32", 0b1, 0b0>;
defm MVE_VMLALDAVu16 : MVE_VMLALDAV_multi<"u16", 0b0, 0b1>;
defm MVE_VMLALDAVu32 : MVE_VMLALDAV_multi<"u32", 0b1, 0b1>;

// vmlalv aliases vmlaldav
foreach acc = ["_acc", "_noacc"] in {
  foreach suffix = ["s16", "s32", "u16", "u32"] in {
    def : MVEInstAlias<!strconcat("vmlalv", !if(!eq(acc, "_acc"), "a", ""),
                       "${vp}.", suffix, "\t$RdaLoDest, $RdaHiDest, $Qn, $Qm"),
                       (!cast<Instruction>("MVE_VMLALDAV"#suffix#acc#"_noexch")
                       tGPREven:$RdaLoDest, tGPROdd:$RdaHiDest,
                       MQPR:$Qn, MQPR:$Qm, vpred_n:$vp)>;
  }
}

multiclass MVE_VMLSLDAV_multi<string iname, string suffix, bit sz,
                            bit bit_28, list<dag> pattern=[]> {
  defm "" : MVE_VMLALDAVBase_XA<iname, suffix, sz, bit_28, 0b0, 0b1, pattern>;
}

defm MVE_VMLSLDAVs16   : MVE_VMLSLDAV_multi<"vmlsldav", "s16", 0b0, 0b0>;
defm MVE_VMLSLDAVs32   : MVE_VMLSLDAV_multi<"vmlsldav", "s32", 0b1, 0b0>;
defm MVE_VRMLSLDAVHs32 : MVE_VMLSLDAV_multi<"vrmlsldavh", "s32", 0b0, 0b1>;

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

class MVE_VMINMAXNM<string iname, string suffix, bit sz, bit bit_21,
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

def MVE_VMAXNMf32 : MVE_VMINMAXNM<"vmaxnm", "f32", 0b0, 0b0>;
def MVE_VMAXNMf16 : MVE_VMINMAXNM<"vmaxnm", "f16", 0b1, 0b0>;

def MVE_VMINNMf32 : MVE_VMINMAXNM<"vminnm", "f32", 0b0, 0b1>;
def MVE_VMINNMf16 : MVE_VMINMAXNM<"vminnm", "f16", 0b1, 0b1>;

class MVE_VMINMAX<string iname, string suffix, bit U, bits<2> size,
              bit bit_4, list<dag> pattern=[]>
  : MVE_comp<NoItinerary, iname, suffix, "", pattern> {

  let Inst{28} = U;
  let Inst{25-24} = 0b11;
  let Inst{23} = 0b0;
  let Inst{21-20} = size{1-0};
  let Inst{11} = 0b0;
  let Inst{8} = 0b0;
  let Inst{6} = 0b1;
  let Inst{4} = bit_4;
}

multiclass MVE_VMINMAX_all_sizes<string iname, bit bit_4> {
  def s8  : MVE_VMINMAX<iname, "s8",  0b0, 0b00, bit_4>;
  def s16 : MVE_VMINMAX<iname, "s16", 0b0, 0b01, bit_4>;
  def s32 : MVE_VMINMAX<iname, "s32", 0b0, 0b10, bit_4>;
  def u8  : MVE_VMINMAX<iname, "u8",  0b1, 0b00, bit_4>;
  def u16 : MVE_VMINMAX<iname, "u16", 0b1, 0b01, bit_4>;
  def u32 : MVE_VMINMAX<iname, "u32", 0b1, 0b10, bit_4>;
}

defm MVE_VMAX : MVE_VMINMAX_all_sizes<"vmax", 0b0>;
defm MVE_VMIN : MVE_VMINMAX_all_sizes<"vmin", 0b1>;

// end of mve_comp instructions

// start of mve_imm_shift instructions

def MVE_VSHLC : MVE_p<(outs rGPR:$RdmDest, MQPR:$Qd),
                      (ins MQPR:$QdSrc, rGPR:$RdmSrc, long_shift:$imm),
                      NoItinerary, "vshlc", "", "$QdSrc, $RdmSrc, $imm",
                      vpred_n, "$RdmDest = $RdmSrc,$Qd = $QdSrc"> {
  bits<5> imm;
  bits<4> Qd;
  bits<4> RdmDest;

  let Inst{28} = 0b0;
  let Inst{25-23} = 0b101;
  let Inst{22} = Qd{3};
  let Inst{21} = 0b1;
  let Inst{20-16} = imm{4-0};
  let Inst{15-13} = Qd{2-0};
  let Inst{12-4} = 0b011111100;
  let Inst{3-0} = RdmDest{3-0};
}

class MVE_shift_imm<dag oops, dag iops, string iname, string suffix,
                    string ops, vpred_ops vpred, string cstr,
                    list<dag> pattern=[]>
  : MVE_p<oops, iops, NoItinerary, iname, suffix, ops, vpred, cstr, pattern> {
  bits<4> Qd;
  bits<4> Qm;

  let Inst{22} = Qd{3};
  let Inst{15-13} = Qd{2-0};
  let Inst{5} = Qm{3};
  let Inst{3-1} = Qm{2-0};
}

class MVE_VMOVL<string iname, string suffix, bits<2> sz, bit U,
              list<dag> pattern=[]>
  : MVE_shift_imm<(outs MQPR:$Qd), (ins MQPR:$Qm),
                  iname, suffix, "$Qd, $Qm", vpred_r, "",
                  pattern> {
  let Inst{28} = U;
  let Inst{25-23} = 0b101;
  let Inst{21} = 0b1;
  let Inst{20-19} = sz{1-0};
  let Inst{18-16} = 0b000;
  let Inst{11-6} = 0b111101;
  let Inst{4} = 0b0;
  let Inst{0} = 0b0;
}

multiclass MVE_VMOVL_shift_half<string iname, string suffix, bits<2> sz, bit U,
                                list<dag> pattern=[]> {
  def bh : MVE_VMOVL<!strconcat(iname, "b"), suffix, sz, U, pattern> {
    let Inst{12} = 0b0;
  }
  def th : MVE_VMOVL<!strconcat(iname, "t"), suffix, sz, U, pattern> {
    let Inst{12} = 0b1;
  }
}

defm MVE_VMOVLs8 : MVE_VMOVL_shift_half<"vmovl", "s8", 0b01, 0b0>;
defm MVE_VMOVLu8 : MVE_VMOVL_shift_half<"vmovl", "u8", 0b01, 0b1>;
defm MVE_VMOVLs16 : MVE_VMOVL_shift_half<"vmovl", "s16", 0b10, 0b0>;
defm MVE_VMOVLu16 : MVE_VMOVL_shift_half<"vmovl", "u16", 0b10, 0b1>;

class MVE_VSHLL_imm<string iname, string suffix, bit U, bit th,
                    dag immops, list<dag> pattern=[]>
  : MVE_shift_imm<(outs MQPR:$Qd), !con((ins MQPR:$Qm), immops),
                  iname, suffix, "$Qd, $Qm, $imm", vpred_r, "", pattern> {
  let Inst{28} = U;
  let Inst{25-23} = 0b101;
  let Inst{21} = 0b1;
  let Inst{12} = th;
  let Inst{11-6} = 0b111101;
  let Inst{4} = 0b0;
  let Inst{0} = 0b0;
}

// The immediate VSHLL instructions accept shift counts from 1 up to
// the lane width (8 or 16), but the full-width shifts have an
// entirely separate encoding, given below with 'lw' in the name.

class MVE_VSHLL_imm8<string iname, string suffix,
                     bit U, bit th, list<dag> pattern=[]>
  : MVE_VSHLL_imm<iname, suffix, U, th, (ins mve_shift_imm1_7:$imm), pattern> {
  bits<3> imm;
  let Inst{20-19} = 0b01;
  let Inst{18-16} = imm;
}

class MVE_VSHLL_imm16<string iname, string suffix,
                      bit U, bit th, list<dag> pattern=[]>
  : MVE_VSHLL_imm<iname, suffix, U, th, (ins mve_shift_imm1_15:$imm), pattern> {
  bits<4> imm;
  let Inst{20} = 0b1;
  let Inst{19-16} = imm;
}

def MVE_VSHLL_imms8bh  : MVE_VSHLL_imm8 <"vshllb", "s8", 0b0, 0b0>;
def MVE_VSHLL_imms8th  : MVE_VSHLL_imm8 <"vshllt", "s8", 0b0, 0b1>;
def MVE_VSHLL_immu8bh  : MVE_VSHLL_imm8 <"vshllb", "u8", 0b1, 0b0>;
def MVE_VSHLL_immu8th  : MVE_VSHLL_imm8 <"vshllt", "u8", 0b1, 0b1>;
def MVE_VSHLL_imms16bh : MVE_VSHLL_imm16<"vshllb", "s16", 0b0, 0b0>;
def MVE_VSHLL_imms16th : MVE_VSHLL_imm16<"vshllt", "s16", 0b0, 0b1>;
def MVE_VSHLL_immu16bh : MVE_VSHLL_imm16<"vshllb", "u16", 0b1, 0b0>;
def MVE_VSHLL_immu16th : MVE_VSHLL_imm16<"vshllt", "u16", 0b1, 0b1>;

class MVE_VSHLL_by_lane_width<string iname, string suffix, bits<2> size,
                              bit U, string ops, list<dag> pattern=[]>
  : MVE_shift_imm<(outs MQPR:$Qd), (ins MQPR:$Qm),
                  iname, suffix, ops, vpred_r, "", pattern> {
  let Inst{28} = U;
  let Inst{25-23} = 0b100;
  let Inst{21-20} = 0b11;
  let Inst{19-18} = size{1-0};
  let Inst{17-16} = 0b01;
  let Inst{11-6} = 0b111000;
  let Inst{4} = 0b0;
  let Inst{0} = 0b1;
}

multiclass MVE_VSHLL_lw<string iname, string suffix, bits<2> sz, bit U,
                              string ops, list<dag> pattern=[]> {
  def bh : MVE_VSHLL_by_lane_width<iname#"b", suffix, sz, U, ops, pattern> {
    let Inst{12} = 0b0;
  }
  def th : MVE_VSHLL_by_lane_width<iname#"t", suffix, sz, U, ops, pattern> {
    let Inst{12} = 0b1;
  }
}

defm MVE_VSHLL_lws8  : MVE_VSHLL_lw<"vshll", "s8",  0b00, 0b0, "$Qd, $Qm, #8">;
defm MVE_VSHLL_lws16 : MVE_VSHLL_lw<"vshll", "s16", 0b01, 0b0, "$Qd, $Qm, #16">;
defm MVE_VSHLL_lwu8  : MVE_VSHLL_lw<"vshll", "u8",  0b00, 0b1, "$Qd, $Qm, #8">;
defm MVE_VSHLL_lwu16 : MVE_VSHLL_lw<"vshll", "u16", 0b01, 0b1, "$Qd, $Qm, #16">;

class MVE_VxSHRN<string iname, string suffix, bit bit_12, bit bit_28,
               dag immops, list<dag> pattern=[]>
  : MVE_shift_imm<(outs MQPR:$Qd), !con((ins MQPR:$QdSrc, MQPR:$Qm), immops),
                  iname, suffix, "$Qd, $Qm, $imm", vpred_n, "$Qd = $QdSrc",
                  pattern> {
  bits<5> imm;

  let Inst{28} = bit_28;
  let Inst{25-23} = 0b101;
  let Inst{21} = 0b0;
  let Inst{20-16} = imm{4-0};
  let Inst{12} = bit_12;
  let Inst{11-6} = 0b111111;
  let Inst{4} = 0b0;
  let Inst{0} = 0b1;
}

def MVE_VRSHRNi16bh : MVE_VxSHRN<
    "vrshrnb", "i16", 0b0, 0b1, (ins shr_imm8:$imm)> {
  let Inst{20-19} = 0b01;
}
def MVE_VRSHRNi16th : MVE_VxSHRN<
    "vrshrnt", "i16", 0b1, 0b1,(ins shr_imm8:$imm)> {
  let Inst{20-19} = 0b01;
}
def MVE_VRSHRNi32bh : MVE_VxSHRN<
    "vrshrnb", "i32", 0b0, 0b1, (ins shr_imm16:$imm)> {
  let Inst{20} = 0b1;
}
def MVE_VRSHRNi32th : MVE_VxSHRN<
    "vrshrnt", "i32", 0b1, 0b1, (ins shr_imm16:$imm)> {
  let Inst{20} = 0b1;
}

def MVE_VSHRNi16bh : MVE_VxSHRN<
    "vshrnb", "i16", 0b0, 0b0, (ins shr_imm8:$imm)> {
  let Inst{20-19} = 0b01;
}
def MVE_VSHRNi16th : MVE_VxSHRN<
    "vshrnt", "i16", 0b1, 0b0, (ins shr_imm8:$imm)> {
  let Inst{20-19} = 0b01;
}
def MVE_VSHRNi32bh : MVE_VxSHRN<
    "vshrnb", "i32", 0b0, 0b0, (ins shr_imm16:$imm)> {
  let Inst{20} = 0b1;
}
def MVE_VSHRNi32th : MVE_VxSHRN<
    "vshrnt", "i32", 0b1, 0b0, (ins shr_imm16:$imm)> {
  let Inst{20} = 0b1;
}

class MVE_VxQRSHRUN<string iname, string suffix, bit bit_28, bit bit_12, dag immops,
                 list<dag> pattern=[]>
  : MVE_shift_imm<(outs MQPR:$Qd), !con((ins MQPR:$QdSrc, MQPR:$Qm), immops),
                  iname, suffix, "$Qd, $Qm, $imm", vpred_n, "$Qd = $QdSrc",
                  pattern> {
  bits<5> imm;

  let Inst{28} = bit_28;
  let Inst{25-23} = 0b101;
  let Inst{21} = 0b0;
  let Inst{20-16} = imm{4-0};
  let Inst{12} = bit_12;
  let Inst{11-6} = 0b111111;
  let Inst{4} = 0b0;
  let Inst{0} = 0b0;
}

def MVE_VQRSHRUNs16bh : MVE_VxQRSHRUN<
    "vqrshrunb", "s16", 0b1, 0b0, (ins shr_imm8:$imm)> {
  let Inst{20-19} = 0b01;
}
def MVE_VQRSHRUNs16th : MVE_VxQRSHRUN<
    "vqrshrunt", "s16", 0b1, 0b1, (ins shr_imm8:$imm)> {
  let Inst{20-19} = 0b01;
}
def MVE_VQRSHRUNs32bh : MVE_VxQRSHRUN<
    "vqrshrunb", "s32", 0b1, 0b0, (ins shr_imm16:$imm)> {
  let Inst{20} = 0b1;
}
def MVE_VQRSHRUNs32th : MVE_VxQRSHRUN<
    "vqrshrunt", "s32", 0b1, 0b1, (ins shr_imm16:$imm)> {
  let Inst{20} = 0b1;
}

def MVE_VQSHRUNs16bh : MVE_VxQRSHRUN<
    "vqshrunb", "s16", 0b0, 0b0, (ins shr_imm8:$imm)> {
  let Inst{20-19} = 0b01;
}
def MVE_VQSHRUNs16th : MVE_VxQRSHRUN<
    "vqshrunt", "s16", 0b0, 0b1, (ins shr_imm8:$imm)> {
  let Inst{20-19} = 0b01;
}
def MVE_VQSHRUNs32bh : MVE_VxQRSHRUN<
    "vqshrunb", "s32", 0b0, 0b0, (ins shr_imm16:$imm)> {
  let Inst{20} = 0b1;
}
def MVE_VQSHRUNs32th : MVE_VxQRSHRUN<
    "vqshrunt", "s32", 0b0, 0b1, (ins shr_imm16:$imm)> {
  let Inst{20} = 0b1;
}

class MVE_VxQRSHRN<string iname, string suffix, bit bit_0, bit bit_12,
                   dag immops, list<dag> pattern=[]>
  : MVE_shift_imm<(outs MQPR:$Qd), !con((ins MQPR:$QdSrc, MQPR:$Qm), immops),
                  iname, suffix, "$Qd, $Qm, $imm", vpred_n, "$Qd = $QdSrc",
                  pattern> {
  bits<5> imm;

  let Inst{25-23} = 0b101;
  let Inst{21} = 0b0;
  let Inst{20-16} = imm{4-0};
  let Inst{12} = bit_12;
  let Inst{11-6} = 0b111101;
  let Inst{4} = 0b0;
  let Inst{0} = bit_0;
}

multiclass MVE_VxQRSHRN_types<string iname, bit bit_0, bit bit_12> {
  def s16 : MVE_VxQRSHRN<iname, "s16", bit_0, bit_12, (ins shr_imm8:$imm)> {
    let Inst{28} = 0b0;
    let Inst{20-19} = 0b01;
  }
  def u16 : MVE_VxQRSHRN<iname, "u16", bit_0, bit_12, (ins shr_imm8:$imm)> {
    let Inst{28} = 0b1;
    let Inst{20-19} = 0b01;
  }
  def s32 : MVE_VxQRSHRN<iname, "s32", bit_0, bit_12, (ins shr_imm16:$imm)> {
    let Inst{28} = 0b0;
    let Inst{20} = 0b1;
  }
  def u32 : MVE_VxQRSHRN<iname, "u32", bit_0, bit_12, (ins shr_imm16:$imm)> {
    let Inst{28} = 0b1;
    let Inst{20} = 0b1;
  }
}

defm MVE_VQRSHRNbh : MVE_VxQRSHRN_types<"vqrshrnb", 0b1, 0b0>;
defm MVE_VQRSHRNth : MVE_VxQRSHRN_types<"vqrshrnt", 0b1, 0b1>;
defm MVE_VQSHRNbh  : MVE_VxQRSHRN_types<"vqshrnb", 0b0, 0b0>;
defm MVE_VQSHRNth  : MVE_VxQRSHRN_types<"vqshrnt", 0b0, 0b1>;

// end of mve_imm_shift instructions

// start of mve_shift instructions

class MVE_shift_by_vec<string iname, string suffix, bit U,
                       bits<2> size, bit bit_4, bit bit_8>
  : MVE_p<(outs MQPR:$Qd), (ins MQPR:$Qm, MQPR:$Qn), NoItinerary,
           iname, suffix, "$Qd, $Qm, $Qn", vpred_r, "", []> {
  // Shift instructions which take a vector of shift counts
  bits<4> Qd;
  bits<4> Qm;
  bits<4> Qn;

  let Inst{28} = U;
  let Inst{25-24} = 0b11;
  let Inst{23} = 0b0;
  let Inst{22} = Qd{3};
  let Inst{21-20} = size;
  let Inst{19-17} = Qn{2-0};
  let Inst{16} = 0b0;
  let Inst{15-13} = Qd{2-0};
  let Inst{12-9} = 0b0010;
  let Inst{8} = bit_8;
  let Inst{7} = Qn{3};
  let Inst{6} = 0b1;
  let Inst{5} = Qm{3};
  let Inst{4} = bit_4;
  let Inst{3-1} = Qm{2-0};
  let Inst{0} = 0b0;
}

multiclass mve_shift_by_vec_multi<string iname, bit bit_4, bit bit_8> {
  def s8  : MVE_shift_by_vec<iname, "s8", 0b0, 0b00, bit_4, bit_8>;
  def s16 : MVE_shift_by_vec<iname, "s16", 0b0, 0b01, bit_4, bit_8>;
  def s32 : MVE_shift_by_vec<iname, "s32", 0b0, 0b10, bit_4, bit_8>;
  def u8  : MVE_shift_by_vec<iname, "u8", 0b1, 0b00, bit_4, bit_8>;
  def u16 : MVE_shift_by_vec<iname, "u16", 0b1, 0b01, bit_4, bit_8>;
  def u32 : MVE_shift_by_vec<iname, "u32", 0b1, 0b10, bit_4, bit_8>;
}

defm MVE_VSHL_by_vec   : mve_shift_by_vec_multi<"vshl",   0b0, 0b0>;
defm MVE_VQSHL_by_vec  : mve_shift_by_vec_multi<"vqshl",  0b1, 0b0>;
defm MVE_VQRSHL_by_vec : mve_shift_by_vec_multi<"vqrshl", 0b1, 0b1>;
defm MVE_VRSHL_by_vec  : mve_shift_by_vec_multi<"vrshl",  0b0, 0b1>;

class MVE_shift_with_imm<string iname, string suffix, dag oops, dag iops,
                         string ops, vpred_ops vpred, string cstr,
                         list<dag> pattern=[]>
  : MVE_p<oops, iops, NoItinerary, iname, suffix, ops, vpred, cstr, pattern> {
  bits<4> Qd;
  bits<4> Qm;

  let Inst{23} = 0b1;
  let Inst{22} = Qd{3};
  let Inst{15-13} = Qd{2-0};
  let Inst{12-11} = 0b00;
  let Inst{7-6} = 0b01;
  let Inst{5} = Qm{3};
  let Inst{4} = 0b1;
  let Inst{3-1} = Qm{2-0};
  let Inst{0} = 0b0;
}

class MVE_VSxI_imm<string iname, string suffix, bit bit_8, dag imm>
  : MVE_shift_with_imm<iname, suffix, (outs MQPR:$Qd),
                       !con((ins MQPR:$Qd_src, MQPR:$Qm), imm),
                       "$Qd, $Qm, $imm", vpred_n, "$Qd = $Qd_src"> {
  bits<6> imm;
  let Inst{28} = 0b1;
  let Inst{25-24} = 0b11;
  let Inst{21-16} = imm;
  let Inst{10-9} = 0b10;
  let Inst{8} = bit_8;
}

def MVE_VSRIimm8 : MVE_VSxI_imm<"vsri", "8", 0b0, (ins shr_imm8:$imm)> {
  let Inst{21-19} = 0b001;
}

def MVE_VSRIimm16 : MVE_VSxI_imm<"vsri", "16", 0b0, (ins shr_imm16:$imm)> {
  let Inst{21-20} = 0b01;
}

def MVE_VSRIimm32 : MVE_VSxI_imm<"vsri", "32", 0b0, (ins shr_imm32:$imm)> {
  let Inst{21} = 0b1;
}

def MVE_VSLIimm8 : MVE_VSxI_imm<"vsli", "8", 0b1, (ins imm0_7:$imm)> {
  let Inst{21-19} = 0b001;
}

def MVE_VSLIimm16 : MVE_VSxI_imm<"vsli", "16", 0b1, (ins imm0_15:$imm)> {
  let Inst{21-20} = 0b01;
}

def MVE_VSLIimm32 : MVE_VSxI_imm<"vsli", "32", 0b1,(ins imm0_31:$imm)> {
  let Inst{21} = 0b1;
}

class MVE_VQSHL_imm<string suffix, dag imm>
  : MVE_shift_with_imm<"vqshl", suffix, (outs MQPR:$Qd),
                       !con((ins MQPR:$Qm), imm), "$Qd, $Qm, $imm",
                       vpred_r, ""> {
  bits<6> imm;

  let Inst{25-24} = 0b11;
  let Inst{21-16} = imm;
  let Inst{10-8} = 0b111;
}

def MVE_VSLIimms8 : MVE_VQSHL_imm<"s8", (ins imm0_7:$imm)> {
  let Inst{28} = 0b0;
  let Inst{21-19} = 0b001;
}

def MVE_VSLIimmu8 : MVE_VQSHL_imm<"u8", (ins imm0_7:$imm)> {
  let Inst{28} = 0b1;
  let Inst{21-19} = 0b001;
}

def MVE_VSLIimms16 : MVE_VQSHL_imm<"s16", (ins imm0_15:$imm)> {
  let Inst{28} = 0b0;
  let Inst{21-20} = 0b01;
}

def MVE_VSLIimmu16 : MVE_VQSHL_imm<"u16", (ins imm0_15:$imm)> {
  let Inst{28} = 0b1;
  let Inst{21-20} = 0b01;
}

def MVE_VSLIimms32 : MVE_VQSHL_imm<"s32", (ins imm0_31:$imm)> {
  let Inst{28} = 0b0;
  let Inst{21} = 0b1;
}

def MVE_VSLIimmu32 : MVE_VQSHL_imm<"u32", (ins imm0_31:$imm)> {
  let Inst{28} = 0b1;
  let Inst{21} = 0b1;
}

class MVE_VQSHLU_imm<string suffix, dag imm>
  : MVE_shift_with_imm<"vqshlu", suffix, (outs MQPR:$Qd),
                       !con((ins MQPR:$Qm), imm), "$Qd, $Qm, $imm",
                       vpred_r, ""> {
  bits<6> imm;

  let Inst{28} = 0b1;
  let Inst{25-24} = 0b11;
  let Inst{21-16} = imm;
  let Inst{10-8} = 0b110;
}

def MVE_VQSHLU_imms8 : MVE_VQSHLU_imm<"s8", (ins imm0_7:$imm)> {
  let Inst{21-19} = 0b001;
}

def MVE_VQSHLU_imms16 : MVE_VQSHLU_imm<"s16", (ins imm0_15:$imm)> {
  let Inst{21-20} = 0b01;
}

def MVE_VQSHLU_imms32 : MVE_VQSHLU_imm<"s32", (ins imm0_31:$imm)> {
  let Inst{21} = 0b1;
}

class MVE_VRSHR_imm<string suffix, dag imm>
  : MVE_shift_with_imm<"vrshr", suffix, (outs MQPR:$Qd),
                       !con((ins MQPR:$Qm), imm), "$Qd, $Qm, $imm",
                       vpred_r, ""> {
  bits<6> imm;

  let Inst{25-24} = 0b11;
  let Inst{21-16} = imm;
  let Inst{10-8} = 0b010;
}

def MVE_VRSHR_imms8 : MVE_VRSHR_imm<"s8", (ins shr_imm8:$imm)> {
  let Inst{28} = 0b0;
  let Inst{21-19} = 0b001;
}

def MVE_VRSHR_immu8 : MVE_VRSHR_imm<"u8", (ins shr_imm8:$imm)> {
  let Inst{28} = 0b1;
  let Inst{21-19} = 0b001;
}

def MVE_VRSHR_imms16 : MVE_VRSHR_imm<"s16", (ins shr_imm16:$imm)> {
  let Inst{28} = 0b0;
  let Inst{21-20} = 0b01;
}

def MVE_VRSHR_immu16 : MVE_VRSHR_imm<"u16", (ins shr_imm16:$imm)> {
  let Inst{28} = 0b1;
  let Inst{21-20} = 0b01;
}

def MVE_VRSHR_imms32 : MVE_VRSHR_imm<"s32", (ins shr_imm32:$imm)> {
  let Inst{28} = 0b0;
  let Inst{21} = 0b1;
}

def MVE_VRSHR_immu32 : MVE_VRSHR_imm<"u32", (ins shr_imm32:$imm)> {
  let Inst{28} = 0b1;
  let Inst{21} = 0b1;
}

class MVE_VSHR_imm<string suffix, dag imm>
  : MVE_shift_with_imm<"vshr", suffix, (outs MQPR:$Qd),
                       !con((ins MQPR:$Qm), imm), "$Qd, $Qm, $imm",
                       vpred_r, ""> {
  bits<6> imm;

  let Inst{25-24} = 0b11;
  let Inst{21-16} = imm;
  let Inst{10-8} = 0b000;
}

def MVE_VSHR_imms8 : MVE_VSHR_imm<"s8", (ins shr_imm8:$imm)> {
  let Inst{28} = 0b0;
  let Inst{21-19} = 0b001;
}

def MVE_VSHR_immu8 : MVE_VSHR_imm<"u8", (ins shr_imm8:$imm)> {
  let Inst{28} = 0b1;
  let Inst{21-19} = 0b001;
}

def MVE_VSHR_imms16 : MVE_VSHR_imm<"s16", (ins shr_imm16:$imm)> {
  let Inst{28} = 0b0;
  let Inst{21-20} = 0b01;
}

def MVE_VSHR_immu16 : MVE_VSHR_imm<"u16", (ins shr_imm16:$imm)> {
  let Inst{28} = 0b1;
  let Inst{21-20} = 0b01;
}

def MVE_VSHR_imms32 : MVE_VSHR_imm<"s32", (ins shr_imm32:$imm)> {
  let Inst{28} = 0b0;
  let Inst{21} = 0b1;
}

def MVE_VSHR_immu32 : MVE_VSHR_imm<"u32", (ins shr_imm32:$imm)> {
  let Inst{28} = 0b1;
  let Inst{21} = 0b1;
}

class MVE_VSHL_imm<string suffix, dag imm>
  : MVE_shift_with_imm<"vshl", suffix, (outs MQPR:$Qd),
                       !con((ins MQPR:$Qm), imm), "$Qd, $Qm, $imm",
                       vpred_r, ""> {
  bits<6> imm;

  let Inst{28} = 0b0;
  let Inst{25-24} = 0b11;
  let Inst{21-16} = imm;
  let Inst{10-8} = 0b101;
}

def MVE_VSHL_immi8 : MVE_VSHL_imm<"i8", (ins imm0_7:$imm)> {
  let Inst{21-19} = 0b001;
}

def MVE_VSHL_immi16 : MVE_VSHL_imm<"i16", (ins imm0_15:$imm)> {
  let Inst{21-20} = 0b01;
}

def MVE_VSHL_immi32 : MVE_VSHL_imm<"i32", (ins imm0_31:$imm)> {
  let Inst{21} = 0b1;
}
// end of mve_shift instructions

class MVE_VPT<string suffix, bits<2> size, dag iops, string asm, list<dag> pattern=[]>
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

class MVE_VPTt1<string suffix, bits<2> size, dag iops>
  : MVE_VPT<suffix, size, iops, "$fc, $Qn, $Qm"> {
  bits<4> Qm;
  bits<4> Mk;

  let Inst{6} = 0b0;
  let Inst{5} = Qm{3};
  let Inst{3-1} = Qm{2-0};
  let Inst{0} = fc{1};
}

class MVE_VPTt1i<string suffix, bits<2> size>
 : MVE_VPTt1<suffix, size,
           (ins vpt_mask:$Mk, pred_basic_i:$fc, MQPR:$Qn, MQPR:$Qm)> {
  let Inst{12} = 0b0;
  let Inst{0} = 0b0;
}

def MVE_VPTv4i32 : MVE_VPTt1i<"i32", 0b10>;
def MVE_VPTv8i16 : MVE_VPTt1i<"i16", 0b01>;
def MVE_VPTv16i8 : MVE_VPTt1i<"i8", 0b00>;

class MVE_VPTt1u<string suffix, bits<2> size>
 : MVE_VPTt1<suffix, size,
           (ins vpt_mask:$Mk, pred_basic_u:$fc, MQPR:$Qn, MQPR:$Qm)> {
  let Inst{12} = 0b0;
  let Inst{0} = 0b1;
}

def MVE_VPTv4u32 : MVE_VPTt1u<"u32", 0b10>;
def MVE_VPTv8u16 : MVE_VPTt1u<"u16", 0b01>;
def MVE_VPTv16u8 : MVE_VPTt1u<"u8", 0b00>;

class MVE_VPTt1s<string suffix, bits<2> size>
 : MVE_VPTt1<suffix, size,
           (ins vpt_mask:$Mk, pred_basic_s:$fc, MQPR:$Qn, MQPR:$Qm)> {
  let Inst{12} = 0b1;
}

def MVE_VPTv4s32 : MVE_VPTt1s<"s32", 0b10>;
def MVE_VPTv8s16 : MVE_VPTt1s<"s16", 0b01>;
def MVE_VPTv16s8 : MVE_VPTt1s<"s8", 0b00>;

class MVE_VPTt2<string suffix, bits<2> size, dag iops>
  : MVE_VPT<suffix, size, iops,
          "$fc, $Qn, $Rm"> {
  bits<4> Rm;
  bits<3> fc;
  bits<4> Mk;

  let Inst{6} = 0b1;
  let Inst{5} = fc{1};
  let Inst{3-0} = Rm{3-0};
}

class MVE_VPTt2i<string suffix, bits<2> size>
  : MVE_VPTt2<suffix, size,
            (ins vpt_mask:$Mk, pred_basic_i:$fc, MQPR:$Qn, GPRwithZR:$Rm)> {
  let Inst{12} = 0b0;
  let Inst{5} = 0b0;
}

def MVE_VPTv4i32r : MVE_VPTt2i<"i32", 0b10>;
def MVE_VPTv8i16r : MVE_VPTt2i<"i16", 0b01>;
def MVE_VPTv16i8r : MVE_VPTt2i<"i8", 0b00>;

class MVE_VPTt2u<string suffix, bits<2> size>
  : MVE_VPTt2<suffix, size,
            (ins vpt_mask:$Mk, pred_basic_u:$fc, MQPR:$Qn, GPRwithZR:$Rm)> {
  let Inst{12} = 0b0;
  let Inst{5} = 0b1;
}

def MVE_VPTv4u32r : MVE_VPTt2u<"u32", 0b10>;
def MVE_VPTv8u16r : MVE_VPTt2u<"u16", 0b01>;
def MVE_VPTv16u8r : MVE_VPTt2u<"u8", 0b00>;

class MVE_VPTt2s<string suffix, bits<2> size>
  : MVE_VPTt2<suffix, size,
            (ins vpt_mask:$Mk, pred_basic_s:$fc, MQPR:$Qn, GPRwithZR:$Rm)> {
  let Inst{12} = 0b1;
}

def MVE_VPTv4s32r : MVE_VPTt2s<"s32", 0b10>;
def MVE_VPTv8s16r : MVE_VPTt2s<"s16", 0b01>;
def MVE_VPTv16s8r : MVE_VPTt2s<"s8", 0b00>;


class MVE_VPTf<string suffix, bit size, dag iops, string asm, list<dag> pattern=[]>
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

class MVE_VPTft1<string suffix, bit size>
  : MVE_VPTf<suffix, size, (ins vpt_mask:$Mk, pred_basic_fp:$fc, MQPR:$Qn, MQPR:$Qm),
          "$fc, $Qn, $Qm"> {
  bits<3> fc;
  bits<4> Qm;

  let Inst{6} = 0b0;
  let Inst{5} = Qm{3};
  let Inst{3-1} = Qm{2-0};
  let Inst{0} = fc{1};
}

def MVE_VPTv4f32         : MVE_VPTft1<"f32", 0b0>;
def MVE_VPTv8f16         : MVE_VPTft1<"f16", 0b1>;

class MVE_VPTft2<string suffix, bit size>
  : MVE_VPTf<suffix, size, (ins vpt_mask:$Mk, pred_basic_fp:$fc, MQPR:$Qn, GPRwithZR:$Rm),
          "$fc, $Qn, $Rm"> {
  bits<3> fc;
  bits<4> Rm;

  let Inst{6} = 0b1;
  let Inst{5} = fc{1};
  let Inst{3-0} = Rm{3-0};
}

def MVE_VPTv4f32r        : MVE_VPTft2<"f32", 0b0>;
def MVE_VPTv8f16r        : MVE_VPTft2<"f16", 0b1>;

def MVE_VPST : MVE_MI<(outs ), (ins vpt_mask:$Mk), NoItinerary,
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

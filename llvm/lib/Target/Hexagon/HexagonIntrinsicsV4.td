//===- HexagonIntrinsicsV4.td - V4 Instruction intrinsics --*- tablegen -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
// This is populated based on the following specs:
// Hexagon V4 Architecture Extensions
// Application-Level Specification
// 80-V9418-12 Rev. A
// June 15, 2010


//
// ALU 32 types.
//

class si_ALU32_sisi_not<string opc, Intrinsic IntID>
  : ALU32_rr<(outs IntRegs:$dst), (ins IntRegs:$src1, IntRegs:$src2),
             !strconcat("$dst = ", !strconcat(opc , "($src1, ~$src2)")),
             [(set IntRegs:$dst, (IntID IntRegs:$src1, IntRegs:$src2))]>;

class di_ALU32_s8si<string opc, Intrinsic IntID>
  : ALU32_rr<(outs DoubleRegs:$dst), (ins s8Imm:$src1, IntRegs:$src2),
             !strconcat("$dst = ", !strconcat(opc , "(#$src1, $src2)")),
             [(set DoubleRegs:$dst, (IntID imm:$src1, IntRegs:$src2))]>;

class di_ALU32_sis8<string opc, Intrinsic IntID>
  : ALU32_rr<(outs DoubleRegs:$dst), (ins IntRegs:$src1, s8Imm:$src2),
             !strconcat("$dst = ", !strconcat(opc , "($src1, #$src2)")),
             [(set DoubleRegs:$dst, (IntID IntRegs:$src1, imm:$src2))]>;

class qi_neg_ALU32_sisi<string opc, Intrinsic IntID>
  : ALU32_rr<(outs PredRegs:$dst), (ins IntRegs:$src1, IntRegs:$src2),
             !strconcat("$dst = !", !strconcat(opc , "($src1, $src2)")),
             [(set PredRegs:$dst, (IntID IntRegs:$src1, IntRegs:$src2))]>;

class qi_neg_ALU32_sis10<string opc, Intrinsic IntID>
  : ALU32_rr<(outs PredRegs:$dst), (ins IntRegs:$src1, s10Imm:$src2),
             !strconcat("$dst = !", !strconcat(opc , "($src1, #$src2)")),
             [(set PredRegs:$dst, (IntID IntRegs:$src1, imm:$src2))]>;

class qi_neg_ALU32_siu9<string opc, Intrinsic IntID>
  : ALU32_rr<(outs PredRegs:$dst), (ins IntRegs:$src1, u9Imm:$src2),
             !strconcat("$dst = !", !strconcat(opc , "($src1, #$src2)")),
             [(set PredRegs:$dst, (IntID IntRegs:$src1, imm:$src2))]>;

class si_neg_ALU32_sisi<string opc, Intrinsic IntID>
  : ALU32_rr<(outs IntRegs:$dst), (ins IntRegs:$src1, IntRegs:$src2),
             !strconcat("$dst = !", !strconcat(opc , "($src1, $src2)")),
             [(set IntRegs:$dst, (IntID IntRegs:$src1, IntRegs:$src2))]>;

class si_neg_ALU32_sis8<string opc, Intrinsic IntID>
  : ALU32_rr<(outs IntRegs:$dst), (ins IntRegs:$src1, s8Imm:$src2),
             !strconcat("$dst = !", !strconcat(opc , "($src1, #$src2)")),
             [(set IntRegs:$dst, (IntID IntRegs:$src1, imm:$src2))]>;

class si_ALU32_sis8<string opc, Intrinsic IntID>
  : ALU32_rr<(outs IntRegs:$dst), (ins IntRegs:$src1, s8Imm:$src2),
             !strconcat("$dst = ", !strconcat(opc , "($src1, #$src2)")),
             [(set IntRegs:$dst, (IntID IntRegs:$src1, imm:$src2))]>;


//
// SInst Classes.
//
class qi_neg_SInst_qiqi<string opc, Intrinsic IntID>
  : SInst<(outs PredRegs:$dst), (ins IntRegs:$src1, IntRegs:$src2),
             !strconcat("$dst = !", !strconcat(opc , "($src1, $src2)")),
             [(set PredRegs:$dst, (IntID IntRegs:$src1, IntRegs:$src2))]>;

class qi_SInst_qi_andqiqi_neg<string opc, Intrinsic IntID>
  : SInst<(outs PredRegs:$dst), (ins IntRegs:$src1, IntRegs:$src2,
                                     IntRegs:$src3),
             !strconcat("$dst = ", !strconcat(opc ,
                                              "($src1, and($src2, !$src3)")),
             [(set PredRegs:$dst, (IntID IntRegs:$src1, IntRegs:$src2,
                                         IntRegs:$src3))]>;

class qi_SInst_qi_andqiqi<string opc, Intrinsic IntID>
  : SInst<(outs PredRegs:$dst), (ins IntRegs:$src1, IntRegs:$src2,
                                     IntRegs:$src3),
             !strconcat("$dst = ", !strconcat(opc ,
                                              "($src1, and($src2, $src3)")),
             [(set PredRegs:$dst, (IntID IntRegs:$src1, IntRegs:$src2,
                                         IntRegs:$src3))]>;

class qi_SInst_qi_orqiqi_neg<string opc, Intrinsic IntID>
  : SInst<(outs PredRegs:$dst), (ins IntRegs:$src1, IntRegs:$src2,
                                     IntRegs:$src3),
             !strconcat("$dst = ", !strconcat(opc ,
                                              "($src1, or($src2, !$src3)")),
             [(set PredRegs:$dst, (IntID IntRegs:$src1, IntRegs:$src2,
                                         IntRegs:$src3))]>;

class qi_SInst_qi_orqiqi<string opc, Intrinsic IntID>
  : SInst<(outs PredRegs:$dst), (ins IntRegs:$src1, IntRegs:$src2,
                                     IntRegs:$src3),
             !strconcat("$dst = ", !strconcat(opc ,
                                              "($src1, or($src2, $src3)")),
             [(set PredRegs:$dst, (IntID IntRegs:$src1, IntRegs:$src2,
                                         IntRegs:$src3))]>;

class si_SInst_si_addsis6<string opc, Intrinsic IntID>
  : SInst<(outs IntRegs:$dst), (ins IntRegs:$src1, IntRegs:$src2, s6Imm:$src3),
             !strconcat("$dst = ", !strconcat(opc ,
                                              "($src1, add($src2, #$src3)")),
             [(set IntRegs:$dst, (IntID IntRegs:$src1, IntRegs:$src2,
                                        imm:$src3))]>;

class si_SInst_si_subs6si<string opc, Intrinsic IntID>
  : SInst<(outs IntRegs:$dst), (ins IntRegs:$src1, s6Imm:$src2, IntRegs:$src3),
             !strconcat("$dst = ", !strconcat(opc ,
                                              "($src1, sub(#$src2, $src3)")),
             [(set IntRegs:$dst, (IntID IntRegs:$src1, imm:$src2,
                                        IntRegs:$src3))]>;

class di_ALU64_didi_neg<string opc, Intrinsic IntID>
  : ALU64_rr<(outs DoubleRegs:$dst), (ins DoubleRegs:$src1, DoubleRegs:$src2),
          !strconcat("$dst = ", !strconcat(opc , "($src1, ~$src2)")),
          [(set DoubleRegs:$dst, (IntID DoubleRegs:$src1, DoubleRegs:$src2))]>;

class di_MInst_dididi_xacc<string opc, Intrinsic IntID>
  : MInst_acc<(outs DoubleRegs:$dst), (ins DoubleRegs:$dst2, DoubleRegs:$src1,
                                           DoubleRegs:$src2),
               !strconcat("$dst ^= ", !strconcat(opc , "($src1, $src2)")),
               [(set DoubleRegs:$dst, (IntID DoubleRegs:$dst2, DoubleRegs:$src1,
                                             DoubleRegs:$src2))],
               "$dst2 = $dst">;

class si_MInst_sisisi_and<string opc, Intrinsic IntID>
  : MInst<(outs IntRegs:$dst), (ins IntRegs:$dst1, IntRegs:$src2,
                                    IntRegs:$src3),
             !strconcat("$dst &= ", !strconcat(opc , "($src2, $src3)")),
             [(set IntRegs:$dst, (IntID IntRegs:$dst1, IntRegs:$src2,
                                        IntRegs:$src3))]>;

class si_MInst_sisisi_andn<string opc, Intrinsic IntID>
  : MInst<(outs IntRegs:$dst), (ins IntRegs:$dst1, IntRegs:$src2,
                                    IntRegs:$src3),
             !strconcat("$dst &= ", !strconcat(opc , "($src2, ~$src3)")),
             [(set IntRegs:$dst, (IntID IntRegs:$dst1, IntRegs:$src2,
                                        IntRegs:$src3))]>;

class si_SInst_sisis10_andi<string opc, Intrinsic IntID>
  : SInst<(outs IntRegs:$dst), (ins IntRegs:$src1, IntRegs:$src2, s10Imm:$src3),
             !strconcat("$dst = ", !strconcat(opc ,
                                              "($src1, and($src2, #$src3))")),
             [(set IntRegs:$dst, (IntID IntRegs:$src1, IntRegs:$src2,
                                        imm:$src3))]>;

class si_MInst_sisisi_xor<string opc, Intrinsic IntID>
  : MInst<(outs IntRegs:$dst), (ins IntRegs:$dst1, IntRegs:$src2,
                                    IntRegs:$src3),
             !strconcat("$dst ^= ", !strconcat(opc , "($src2, $src3)")),
             [(set IntRegs:$dst, (IntID IntRegs:$dst1, IntRegs:$src2,
                                        IntRegs:$src3))]>;

class si_MInst_sisisi_xorn<string opc, Intrinsic IntID>
  : MInst<(outs IntRegs:$dst), (ins IntRegs:$dst1, IntRegs:$src2,
                                    IntRegs:$src3),
             !strconcat("$dst ^= ", !strconcat(opc , "($src2, ~$src3)")),
             [(set IntRegs:$dst, (IntID IntRegs:$dst1, IntRegs:$src2,
                                        IntRegs:$src3))]>;

class si_SInst_sisis10_or<string opc, Intrinsic IntID>
  : SInst<(outs IntRegs:$dst), (ins IntRegs:$dst1, IntRegs:$src2, s10Imm:$src3),
             !strconcat("$dst |= ", !strconcat(opc , "($src2, #$src3)")),
             [(set IntRegs:$dst, (IntID IntRegs:$dst1, IntRegs:$src2,
                                        imm:$src3))]>;

class si_MInst_sisisi_or<string opc, Intrinsic IntID>
  : MInst<(outs IntRegs:$dst), (ins IntRegs:$dst1, IntRegs:$src2,
                                    IntRegs:$src3),
             !strconcat("$dst |= ", !strconcat(opc , "($src2, $src3)")),
             [(set IntRegs:$dst, (IntID IntRegs:$dst1, IntRegs:$src2,
                                        IntRegs:$src3))]>;

class si_MInst_sisisi_orn<string opc, Intrinsic IntID>
  : MInst<(outs IntRegs:$dst), (ins IntRegs:$dst1, IntRegs:$src2,
                                    IntRegs:$src3),
             !strconcat("$dst |= ", !strconcat(opc , "($src2, ~$src3)")),
             [(set IntRegs:$dst, (IntID IntRegs:$dst1, IntRegs:$src2,
                                        IntRegs:$src3))]>;

class si_SInst_siu5_sat<string opc, Intrinsic IntID>
  : SInst<(outs IntRegs:$dst), (ins IntRegs:$src1, u5Imm:$src2),
          !strconcat("$dst = ", !strconcat(opc , "($src1, #$src2):sat")),
          [(set IntRegs:$dst, (IntID IntRegs:$src1, imm:$src2))]>;


/********************************************************************
*            ALU32/ALU                                              *
*********************************************************************/

// ALU32 / ALU / Logical Operations.
def Hexagon_A4_orn  : si_ALU32_sisi_not <"or",  int_hexagon_A4_orn>;
def Hexagon_A4_andn : si_ALU32_sisi_not <"and", int_hexagon_A4_andn>;


/********************************************************************
*            ALU32/PERM                                             *
*********************************************************************/

// ALU32 / PERM / Combine Words Into Doublewords.
def Hexagon_A4_combineir : di_ALU32_s8si  <"combine", int_hexagon_A4_combineir>;
def Hexagon_A4_combineri : di_ALU32_sis8  <"combine", int_hexagon_A4_combineri>;


/********************************************************************
*            ALU32/PRED                                             *
*********************************************************************/

// ALU32 / PRED / Conditional Shift Halfword.
// ALU32 / PRED / Conditional Sign Extend.
// ALU32 / PRED / Conditional Zero Extend.
// ALU32 / PRED / Compare.
def Hexagon_C4_cmpneq  : qi_neg_ALU32_sisi  <"cmp.eq", int_hexagon_C4_cmpneq>;
def Hexagon_C4_cmpneqi : qi_neg_ALU32_sis10 <"cmp.eq", int_hexagon_C4_cmpneqi>;
def Hexagon_C4_cmplte  : qi_neg_ALU32_sisi  <"cmp.gt", int_hexagon_C4_cmplte>;
def Hexagon_C4_cmpltei : qi_neg_ALU32_sis10 <"cmp.gt", int_hexagon_C4_cmpltei>;
def Hexagon_C4_cmplteu : qi_neg_ALU32_sisi  <"cmp.gtu",int_hexagon_C4_cmplteu>;
def Hexagon_C4_cmplteui: qi_neg_ALU32_siu9  <"cmp.gtu",int_hexagon_C4_cmplteui>;

// ALU32 / PRED / cmpare To General Register.
def Hexagon_A4_rcmpneq : si_neg_ALU32_sisi <"cmp.eq", int_hexagon_A4_rcmpneq>;
def Hexagon_A4_rcmpneqi: si_neg_ALU32_sis8 <"cmp.eq", int_hexagon_A4_rcmpneqi>;
def Hexagon_A4_rcmpeq  : si_ALU32_sisi     <"cmp.eq", int_hexagon_A4_rcmpeq>;
def Hexagon_A4_rcmpeqi : si_ALU32_sis8     <"cmp.eq", int_hexagon_A4_rcmpeqi>;


/********************************************************************
*            CR                                                     *
*********************************************************************/

// CR / Corner Detection Acceleration.
def Hexagon_C4_fastcorner9:
  qi_SInst_qiqi<"fastcorner9", int_hexagon_C4_fastcorner9>;
def Hexagon_C4_fastcorner9_not:
  qi_neg_SInst_qiqi<"fastcorner9",int_hexagon_C4_fastcorner9_not>;

// CR / Logical Operations On Predicates.
def Hexagon_C4_and_andn:
  qi_SInst_qi_andqiqi_neg         <"and",      int_hexagon_C4_and_andn>;
def Hexagon_C4_and_and:
  qi_SInst_qi_andqiqi             <"and",      int_hexagon_C4_and_and>;
def Hexagon_C4_and_orn:
  qi_SInst_qi_orqiqi_neg          <"and",      int_hexagon_C4_and_orn>;
def Hexagon_C4_and_or:
  qi_SInst_qi_orqiqi              <"and",      int_hexagon_C4_and_or>;
def Hexagon_C4_or_andn:
  qi_SInst_qi_andqiqi_neg         <"or",       int_hexagon_C4_or_andn>;
def Hexagon_C4_or_and:
  qi_SInst_qi_andqiqi             <"or",       int_hexagon_C4_or_and>;
def Hexagon_C4_or_orn:
  qi_SInst_qi_orqiqi_neg          <"or",       int_hexagon_C4_or_orn>;
def Hexagon_C4_or_or:
  qi_SInst_qi_orqiqi              <"or",       int_hexagon_C4_or_or>;


/********************************************************************
*            XTYPE/ALU                                              *
*********************************************************************/

// XTYPE / ALU / Add And Accumulate.
def Hexagon_S4_addaddi:
  si_SInst_si_addsis6             <"add",      int_hexagon_S4_addaddi>;
def Hexagon_S4_subaddi:
  si_SInst_si_subs6si             <"add",      int_hexagon_S4_subaddi>;

// XTYPE / ALU / Logical Doublewords.
def Hexagon_S4_andnp:
  di_ALU64_didi_neg               <"and",      int_hexagon_A4_andnp>;
def Hexagon_S4_ornp:
  di_ALU64_didi_neg               <"or",       int_hexagon_A4_ornp>;

// XTYPE / ALU / Logical-logical Doublewords.
def Hexagon_M4_xor_xacc:
  di_MInst_dididi_xacc            <"xor",      int_hexagon_M4_xor_xacc>;

// XTYPE / ALU / Logical-logical Words.
def HEXAGON_M4_and_and:
  si_MInst_sisisi_and             <"and",      int_hexagon_M4_and_and>;
def HEXAGON_M4_and_or:
  si_MInst_sisisi_and             <"or",       int_hexagon_M4_and_or>;
def HEXAGON_M4_and_xor:
  si_MInst_sisisi_and             <"xor",      int_hexagon_M4_and_xor>;
def HEXAGON_M4_and_andn:
  si_MInst_sisisi_andn            <"and",      int_hexagon_M4_and_andn>;
def HEXAGON_M4_xor_and:
  si_MInst_sisisi_xor             <"and",      int_hexagon_M4_xor_and>;
def HEXAGON_M4_xor_or:
  si_MInst_sisisi_xor             <"or",       int_hexagon_M4_xor_or>;
def HEXAGON_M4_xor_andn:
  si_MInst_sisisi_xorn            <"and",      int_hexagon_M4_xor_andn>;
def HEXAGON_M4_or_and:
  si_MInst_sisisi_or              <"and",      int_hexagon_M4_or_and>;
def HEXAGON_M4_or_or:
  si_MInst_sisisi_or              <"or",       int_hexagon_M4_or_or>;
def HEXAGON_M4_or_xor:
  si_MInst_sisisi_or              <"xor",      int_hexagon_M4_or_xor>;
def HEXAGON_M4_or_andn:
  si_MInst_sisisi_orn             <"and",      int_hexagon_M4_or_andn>;
def HEXAGON_S4_or_andix:
  si_SInst_sisis10_andi           <"or",       int_hexagon_S4_or_andix>;
def HEXAGON_S4_or_andi:
  si_SInst_sisis10_or             <"and",      int_hexagon_S4_or_andi>;
def HEXAGON_S4_or_ori:
  si_SInst_sisis10_or             <"or",       int_hexagon_S4_or_ori>;

// XTYPE / ALU / Modulo wrap.
def HEXAGON_A4_modwrapu:
  si_ALU64_sisi                   <"modwrap",  int_hexagon_A4_modwrapu>;

// XTYPE / ALU / Round.
def HEXAGON_A4_cround_ri:
  si_SInst_siu5                   <"cround",   int_hexagon_A4_cround_ri>;
def HEXAGON_A4_cround_rr:
  si_SInst_sisi                   <"cround",   int_hexagon_A4_cround_rr>;
def HEXAGON_A4_round_ri:
  si_SInst_siu5                   <"round",    int_hexagon_A4_round_ri>;
def HEXAGON_A4_round_rr:
  si_SInst_sisi                   <"round",    int_hexagon_A4_round_rr>;
def HEXAGON_A4_round_ri_sat:
  si_SInst_siu5_sat               <"round",    int_hexagon_A4_round_ri_sat>;
def HEXAGON_A4_round_rr_sat:
  si_SInst_sisi_sat               <"round",    int_hexagon_A4_round_rr_sat>;

// XTYPE / ALU / Vector reduce add unsigned halfwords.
// XTYPE / ALU / Vector add bytes.
// XTYPE / ALU / Vector conditional negate.
// XTYPE / ALU / Vector maximum bytes.
// XTYPE / ALU / Vector reduce maximum halfwords.
// XTYPE / ALU / Vector reduce maximum words.
// XTYPE / ALU / Vector minimum bytes.
// XTYPE / ALU / Vector reduce minimum halfwords.
// XTYPE / ALU / Vector reduce minimum words.
// XTYPE / ALU / Vector subtract bytes.


/********************************************************************
*            XTYPE/BIT                                              *
*********************************************************************/

// XTYPE / BIT / Count leading.
// XTYPE / BIT / Count trailing.
// XTYPE / BIT / Extract bitfield.
// XTYPE / BIT / Masked parity.
// XTYPE / BIT / Bit reverse.
// XTYPE / BIT / Split bitfield.


/********************************************************************
*            XTYPE/COMPLEX                                          *
*********************************************************************/

// XTYPE / COMPLEX / Complex add/sub halfwords.
// XTYPE / COMPLEX / Complex add/sub words.
// XTYPE / COMPLEX / Complex multiply 32x16.
// XTYPE / COMPLEX / Vector reduce complex rotate.


/********************************************************************
*            XTYPE/MPY                                              *
*********************************************************************/

// XTYPE / COMPLEX / Complex add/sub halfwords.

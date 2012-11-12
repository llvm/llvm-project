//===-- HexagonIntrinsics.td - Instruction intrinsics ------*- tablegen -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
// This is populated based on the following specs:
// Hexagon V2 Architecture
// Application-Level Specification
// 80-V9418-8 Rev. B
// March 4, 2008
//===----------------------------------------------------------------------===//

//
// ALU 32 types.
//

class qi_ALU32_sisi<string opc, Intrinsic IntID>
  : ALU32_rr<(outs PredRegs:$dst), (ins IntRegs:$src1, IntRegs:$src2),
             !strconcat("$dst = ", !strconcat(opc , "($src1, $src2)")),
             [(set PredRegs:$dst, (IntID IntRegs:$src1, IntRegs:$src2))]>;

class qi_ALU32_sis10<string opc, Intrinsic IntID>
  : ALU32_rr<(outs PredRegs:$dst), (ins IntRegs:$src1, s10Imm:$src2),
             !strconcat("$dst = ", !strconcat(opc , "($src1, #$src2)")),
             [(set PredRegs:$dst, (IntID IntRegs:$src1, imm:$src2))]>;

class qi_ALU32_sis8<string opc, Intrinsic IntID>
  : ALU32_rr<(outs PredRegs:$dst), (ins IntRegs:$src1, s8Imm:$src2),
             !strconcat("$dst = ", !strconcat(opc , "($src1, #$src2)")),
             [(set PredRegs:$dst, (IntID IntRegs:$src1, imm:$src2))]>;

class qi_ALU32_siu8<string opc, Intrinsic IntID>
  : ALU32_rr<(outs PredRegs:$dst), (ins IntRegs:$src1, u8Imm:$src2),
             !strconcat("$dst = ", !strconcat(opc , "($src1, #$src2)")),
             [(set PredRegs:$dst, (IntID IntRegs:$src1, imm:$src2))]>;

class qi_ALU32_siu9<string opc, Intrinsic IntID>
  : ALU32_rr<(outs PredRegs:$dst), (ins IntRegs:$src1, u9Imm:$src2),
             !strconcat("$dst = ", !strconcat(opc , "($src1, #$src2)")),
             [(set PredRegs:$dst, (IntID IntRegs:$src1, imm:$src2))]>;

class si_ALU32_qisisi<string opc, Intrinsic IntID>
  : ALU32_rr<(outs IntRegs:$dst), (ins IntRegs:$src1, IntRegs:$src2,
                                      IntRegs:$src3),
             !strconcat("$dst = ", !strconcat(opc , "($src1, $src2, $src3)")),
             [(set IntRegs:$dst, (IntID IntRegs:$src1, IntRegs:$src2,
                                        IntRegs:$src3))]>;

class si_ALU32_qis8si<string opc, Intrinsic IntID>
  : ALU32_rr<(outs IntRegs:$dst), (ins IntRegs:$src1, s8Imm:$src2,
                                       IntRegs:$src3),
             !strconcat("$dst = ", !strconcat(opc , "($src1, #$src2, $src3)")),
             [(set IntRegs:$dst, (IntID IntRegs:$src1, imm:$src2,
                                        IntRegs:$src3))]>;

class si_ALU32_qisis8<string opc, Intrinsic IntID>
  : ALU32_rr<(outs IntRegs:$dst), (ins IntRegs:$src1, IntRegs:$src2,
                                       s8Imm:$src3),
             !strconcat("$dst = ", !strconcat(opc , "($src1, $src2, #$src3)")),
             [(set IntRegs:$dst, (IntID IntRegs:$src1, IntRegs:$src2,
                                        imm:$src3))]>;

class si_ALU32_qis8s8<string opc, Intrinsic IntID>
  : ALU32_rr<(outs IntRegs:$dst), (ins IntRegs:$src1, s8Imm:$src2, s8Imm:$src3),
             !strconcat("$dst = ", !strconcat(opc , "($src1, #$src2, #$src3)")),
             [(set IntRegs:$dst, (IntID IntRegs:$src1, imm:$src2, imm:$src3))]>;

class si_ALU32_sisi<string opc, Intrinsic IntID>
  : ALU32_rr<(outs IntRegs:$dst), (ins IntRegs:$src1, IntRegs:$src2),
             !strconcat("$dst = ", !strconcat(opc , "($src1, $src2)")),
             [(set IntRegs:$dst, (IntID IntRegs:$src1, IntRegs:$src2))]>;

class si_ALU32_sisi_sat<string opc, Intrinsic IntID>
  : ALU32_rr<(outs IntRegs:$dst), (ins IntRegs:$src1, IntRegs:$src2),
             !strconcat("$dst = ", !strconcat(opc , "($src1, $src2):sat")),
             [(set IntRegs:$dst, (IntID IntRegs:$src1, IntRegs:$src2))]>;

class si_ALU32_sisi_rnd<string opc, Intrinsic IntID>
  : ALU32_rr<(outs IntRegs:$dst), (ins IntRegs:$src1, IntRegs:$src2),
             !strconcat("$dst = ", !strconcat(opc , "($src1, $src2):rnd")),
             [(set IntRegs:$dst, (IntID IntRegs:$src1, IntRegs:$src2))]>;

class si_ALU32_sis16<string opc, Intrinsic IntID>
  : ALU32_rr<(outs IntRegs:$dst), (ins IntRegs:$src1, s16Imm:$src2),
             !strconcat("$dst = ", !strconcat(opc , "($src1, #$src2)")),
             [(set IntRegs:$dst, (IntID IntRegs:$src1, imm:$src2))]>;

class si_ALU32_sis10<string opc, Intrinsic IntID>
  : ALU32_rr<(outs IntRegs:$dst), (ins IntRegs:$src1, s10Imm:$src2),
             !strconcat("$dst = ", !strconcat(opc , "($src1, #$src2)")),
             [(set IntRegs:$dst, (IntID IntRegs:$src1, imm:$src2))]>;

class si_ALU32_s10si<string opc, Intrinsic IntID>
  : ALU32_rr<(outs IntRegs:$dst), (ins s10Imm:$src1, IntRegs:$src2),
             !strconcat("$dst = ", !strconcat(opc , "(#$src1, $src2)")),
             [(set IntRegs:$dst, (IntID imm:$src1, IntRegs:$src2))]>;

class si_lo_ALU32_siu16<string opc, Intrinsic IntID>
  : ALU32_rr<(outs IntRegs:$dst), (ins IntRegs:$src1, u16Imm:$src2),
             !strconcat("$dst.l = ", !strconcat(opc , "#$src2")),
             [(set IntRegs:$dst, (IntID IntRegs:$src1, imm:$src2))]>;

class si_hi_ALU32_siu16<string opc, Intrinsic IntID>
  : ALU32_rr<(outs IntRegs:$dst), (ins IntRegs:$src1, u16Imm:$src2),
             !strconcat("$dst.h = ", !strconcat(opc , "#$src2")),
             [(set IntRegs:$dst, (IntID IntRegs:$src1, imm:$src2))]>;

class si_ALU32_s16<string opc, Intrinsic IntID>
  : ALU32_rr<(outs IntRegs:$dst), (ins s16Imm:$src1),
             !strconcat("$dst = ", !strconcat(opc , "#$src1")),
             [(set IntRegs:$dst, (IntID imm:$src1))]>;

class di_ALU32_s8<string opc, Intrinsic IntID>
  : ALU32_rr<(outs DoubleRegs:$dst), (ins s8Imm:$src1),
             !strconcat("$dst = ", !strconcat(opc , "#$src1")),
             [(set DoubleRegs:$dst, (IntID imm:$src1))]>;

class di_ALU64_di<string opc, Intrinsic IntID>
  : ALU64_rr<(outs DoubleRegs:$dst), (ins DoubleRegs:$src),
             !strconcat("$dst = ", !strconcat(opc , "$src")),
             [(set DoubleRegs:$dst, (IntID DoubleRegs:$src))]>;

class si_ALU32_si<string opc, Intrinsic IntID>
  : ALU32_rr<(outs IntRegs:$dst), (ins IntRegs:$src),
             !strconcat("$dst = ", !strconcat(opc , "($src)")),
             [(set IntRegs:$dst, (IntID IntRegs:$src))]>;

class si_ALU32_si_tfr<string opc, Intrinsic IntID>
  : ALU32_rr<(outs IntRegs:$dst), (ins IntRegs:$src),
             !strconcat("$dst = ", !strconcat(opc , "$src")),
             [(set IntRegs:$dst, (IntID IntRegs:$src))]>;

//
// ALU 64 types.
//

class si_ALU64_si_sat<string opc, Intrinsic IntID>
  : ALU64_rr<(outs IntRegs:$dst), (ins IntRegs:$src),
             !strconcat("$dst = ", !strconcat(opc , "($src):sat")),
             [(set IntRegs:$dst, (IntID IntRegs:$src))]>;

class si_ALU64_didi<string opc, Intrinsic IntID>
  : ALU64_rr<(outs IntRegs:$dst), (ins DoubleRegs:$src1, DoubleRegs:$src2),
             !strconcat("$dst = ", !strconcat(opc , "($src1, $src2)")),
             [(set IntRegs:$dst, (IntID DoubleRegs:$src1, DoubleRegs:$src2))]>;

class di_ALU64_sidi<string opc, Intrinsic IntID>
  : ALU64_rr<(outs DoubleRegs:$dst), (ins IntRegs:$src1, DoubleRegs:$src2),
             !strconcat("$dst = ", !strconcat(opc , "($src1, $src2)")),
             [(set DoubleRegs:$dst, (IntID IntRegs:$src1, DoubleRegs:$src2))]>;

class di_ALU64_didi<string opc, Intrinsic IntID>
  : ALU64_rr<(outs DoubleRegs:$dst), (ins DoubleRegs:$src1, DoubleRegs:$src2),
             !strconcat("$dst = ", !strconcat(opc , "($src1, $src2)")),
             [(set DoubleRegs:$dst, (IntID DoubleRegs:$src1,
                                           DoubleRegs:$src2))]>;

class di_ALU64_qididi<string opc, Intrinsic IntID>
  : ALU64_rr<(outs DoubleRegs:$dst), (ins IntRegs:$src1, DoubleRegs:$src2,
                                          DoubleRegs:$src3),
             !strconcat("$dst = ", !strconcat(opc , "($src1, $src2, $src3)")),
             [(set DoubleRegs:$dst, (IntID IntRegs:$src1, DoubleRegs:$src2,
                                           DoubleRegs:$src3))]>;

class di_ALU64_sisi<string opc, Intrinsic IntID>
  : ALU64_rr<(outs DoubleRegs:$dst), (ins IntRegs:$src1, IntRegs:$src2),
             !strconcat("$dst = ", !strconcat(opc , "($src1, $src2)")),
             [(set DoubleRegs:$dst, (IntID IntRegs:$src1, IntRegs:$src2))]>;

class di_ALU64_didi_sat<string opc, Intrinsic IntID>
  : ALU64_rr<(outs DoubleRegs:$dst), (ins DoubleRegs:$src1, DoubleRegs:$src2),
             !strconcat("$dst = ", !strconcat(opc , "($src1, $src2):sat")),
             [(set DoubleRegs:$dst, (IntID DoubleRegs:$src1,
                                           DoubleRegs:$src2))]>;

class di_ALU64_didi_rnd<string opc, Intrinsic IntID>
  : ALU64_rr<(outs DoubleRegs:$dst), (ins DoubleRegs:$src1, DoubleRegs:$src2),
             !strconcat("$dst = ", !strconcat(opc , "($src1, $src2):rnd")),
             [(set DoubleRegs:$dst, (IntID DoubleRegs:$src1,
                                           DoubleRegs:$src2))]>;

class di_ALU64_didi_crnd<string opc, Intrinsic IntID>
  : ALU64_rr<(outs DoubleRegs:$dst), (ins DoubleRegs:$src1, DoubleRegs:$src2),
             !strconcat("$dst = ", !strconcat(opc , "($src1, $src2):crnd")),
             [(set DoubleRegs:$dst, (IntID DoubleRegs:$src1,
                                           DoubleRegs:$src2))]>;

class di_ALU64_didi_rnd_sat<string opc, Intrinsic IntID>
  : ALU64_rr<(outs DoubleRegs:$dst), (ins DoubleRegs:$src1, DoubleRegs:$src2),
             !strconcat("$dst = ", !strconcat(opc , "($src1, $src2):rnd:sat")),
             [(set DoubleRegs:$dst, (IntID DoubleRegs:$src1,
                                           DoubleRegs:$src2))]>;

class di_ALU64_didi_crnd_sat<string opc, Intrinsic IntID>
  : ALU64_rr<(outs DoubleRegs:$dst), (ins DoubleRegs:$src1, DoubleRegs:$src2),
             !strconcat("$dst = ", !strconcat(opc , "($src1, $src2):crnd:sat")),
             [(set DoubleRegs:$dst, (IntID DoubleRegs:$src1,
                                           DoubleRegs:$src2))]>;

class qi_ALU64_didi<string opc, Intrinsic IntID>
  : ALU64_rr<(outs PredRegs:$dst), (ins DoubleRegs:$src1, DoubleRegs:$src2),
             !strconcat("$dst = ", !strconcat(opc , "($src1, $src2)")),
             [(set PredRegs:$dst, (IntID DoubleRegs:$src1, DoubleRegs:$src2))]>;

class si_ALU64_sisi<string opc, Intrinsic IntID>
  : ALU64_rr<(outs IntRegs:$dst), (ins IntRegs:$src1, IntRegs:$src2),
             !strconcat("$dst = ", !strconcat(opc , "($src1, $src2)")),
             [(set IntRegs:$dst, (IntID IntRegs:$src1, IntRegs:$src2))]>;

class si_ALU64_sisi_sat_lh<string opc, Intrinsic IntID>
  : ALU64_rr<(outs IntRegs:$dst), (ins IntRegs:$src1, IntRegs:$src2),
             !strconcat("$dst = ", !strconcat(opc , "($src1.L, $src2.H):sat")),
             [(set IntRegs:$dst, (IntID IntRegs:$src1, IntRegs:$src2))]>;

class si_ALU64_sisi_l16_sat_hh<string opc, Intrinsic IntID>
  : ALU64_rr<(outs IntRegs:$dst), (ins IntRegs:$src1, IntRegs:$src2),
             !strconcat("$dst = ", !strconcat(opc , "($src1.H, $src2.H):sat")),
             [(set IntRegs:$dst, (IntID IntRegs:$src1, IntRegs:$src2))]>;

class si_ALU64_sisi_l16_sat_lh<string opc, Intrinsic IntID>
  : ALU64_rr<(outs IntRegs:$dst), (ins IntRegs:$src1, IntRegs:$src2),
             !strconcat("$dst = ", !strconcat(opc , "($src1.L, $src2.H):sat")),
             [(set IntRegs:$dst, (IntID IntRegs:$src1, IntRegs:$src2))]>;

class si_ALU64_sisi_l16_sat_hl<string opc, Intrinsic IntID>
  : ALU64_rr<(outs IntRegs:$dst), (ins IntRegs:$src1, IntRegs:$src2),
             !strconcat("$dst = ", !strconcat(opc , "($src1.H, $src2.L):sat")),
             [(set IntRegs:$dst, (IntID IntRegs:$src1, IntRegs:$src2))]>;

class si_ALU64_sisi_l16_sat_ll<string opc, Intrinsic IntID>
  : ALU64_rr<(outs IntRegs:$dst), (ins IntRegs:$src1, IntRegs:$src2),
             !strconcat("$dst = ", !strconcat(opc , "($src1.L, $src2.L):sat")),
             [(set IntRegs:$dst, (IntID IntRegs:$src1, IntRegs:$src2))]>;

class si_ALU64_sisi_l16_hh<string opc, Intrinsic IntID>
  : ALU64_rr<(outs IntRegs:$dst), (ins IntRegs:$src1, IntRegs:$src2),
             !strconcat("$dst = ", !strconcat(opc , "($src1.H, $src2.H)")),
             [(set IntRegs:$dst, (IntID IntRegs:$src1, IntRegs:$src2))]>;

class si_ALU64_sisi_l16_hl<string opc, Intrinsic IntID>
  : ALU64_rr<(outs IntRegs:$dst), (ins IntRegs:$src1, IntRegs:$src2),
             !strconcat("$dst = ", !strconcat(opc , "($src1.H, $src2.L)")),
             [(set IntRegs:$dst, (IntID IntRegs:$src1, IntRegs:$src2))]>;

class si_ALU64_sisi_l16_lh<string opc, Intrinsic IntID>
  : ALU64_rr<(outs IntRegs:$dst), (ins IntRegs:$src1, IntRegs:$src2),
             !strconcat("$dst = ", !strconcat(opc , "($src1.L, $src2.H)")),
             [(set IntRegs:$dst, (IntID IntRegs:$src1, IntRegs:$src2))]>;

class si_ALU64_sisi_l16_ll<string opc, Intrinsic IntID>
  : ALU64_rr<(outs IntRegs:$dst), (ins IntRegs:$src1, IntRegs:$src2),
             !strconcat("$dst = ", !strconcat(opc , "($src1.L, $src2.L)")),
             [(set IntRegs:$dst, (IntID IntRegs:$src1, IntRegs:$src2))]>;

class si_ALU64_sisi_h16_sat_hh<string opc, Intrinsic IntID>
  : ALU64_rr<(outs IntRegs:$dst), (ins IntRegs:$src1, IntRegs:$src2),
             !strconcat("$dst = ", !strconcat(opc ,
                                              "($src1.H, $src2.H):sat:<<16")),
             [(set IntRegs:$dst, (IntID IntRegs:$src1, IntRegs:$src2))]>;

class si_ALU64_sisi_h16_sat_lh<string opc, Intrinsic IntID>
  : ALU64_rr<(outs IntRegs:$dst), (ins IntRegs:$src1, IntRegs:$src2),
             !strconcat("$dst = ", !strconcat(opc ,
                                              "($src1.L, $src2.H):sat:<<16")),
             [(set IntRegs:$dst, (IntID IntRegs:$src1, IntRegs:$src2))]>;

class si_ALU64_sisi_h16_sat_hl<string opc, Intrinsic IntID>
  : ALU64_rr<(outs IntRegs:$dst), (ins IntRegs:$src1, IntRegs:$src2),
             !strconcat("$dst = ", !strconcat(opc ,
                                              "($src1.H, $src2.L):sat:<<16")),
             [(set IntRegs:$dst, (IntID IntRegs:$src1, IntRegs:$src2))]>;

class si_ALU64_sisi_h16_sat_ll<string opc, Intrinsic IntID>
  : ALU64_rr<(outs IntRegs:$dst), (ins IntRegs:$src1, IntRegs:$src2),
             !strconcat("$dst = ", !strconcat(opc ,
                                              "($src1.L, $src2.L):sat:<<16")),
             [(set IntRegs:$dst, (IntID IntRegs:$src1, IntRegs:$src2))]>;

class si_ALU64_sisi_h16_hh<string opc, Intrinsic IntID>
  : ALU64_rr<(outs IntRegs:$dst), (ins IntRegs:$src1, IntRegs:$src2),
             !strconcat("$dst = ", !strconcat(opc , "($src1.H, $src2.H):<<16")),
             [(set IntRegs:$dst, (IntID IntRegs:$src1, IntRegs:$src2))]>;

class si_ALU64_sisi_h16_hl<string opc, Intrinsic IntID>
  : ALU64_rr<(outs IntRegs:$dst), (ins IntRegs:$src1, IntRegs:$src2),
             !strconcat("$dst = ", !strconcat(opc , "($src1.H, $src2.L):<<16")),
             [(set IntRegs:$dst, (IntID IntRegs:$src1, IntRegs:$src2))]>;

class si_ALU64_sisi_h16_lh<string opc, Intrinsic IntID>
  : ALU64_rr<(outs IntRegs:$dst), (ins IntRegs:$src1, IntRegs:$src2),
             !strconcat("$dst = ", !strconcat(opc , "($src1.L, $src2.H):<<16")),
             [(set IntRegs:$dst, (IntID IntRegs:$src1, IntRegs:$src2))]>;

class si_ALU64_sisi_h16_ll<string opc, Intrinsic IntID>
  : ALU64_rr<(outs IntRegs:$dst), (ins IntRegs:$src1, IntRegs:$src2),
             !strconcat("$dst = ", !strconcat(opc , "($src1.L, $src2.L):<<16")),
             [(set IntRegs:$dst, (IntID IntRegs:$src1, IntRegs:$src2))]>;

class si_ALU64_sisi_lh<string opc, Intrinsic IntID>
  : ALU64_rr<(outs IntRegs:$dst), (ins IntRegs:$src1, IntRegs:$src2),
             !strconcat("$dst = ", !strconcat(opc , "($src1.L, $src2.H)")),
             [(set IntRegs:$dst, (IntID IntRegs:$src1, IntRegs:$src2))]>;

class si_ALU64_sisi_ll<string opc, Intrinsic IntID>
  : ALU64_rr<(outs IntRegs:$dst), (ins IntRegs:$src1, IntRegs:$src2),
             !strconcat("$dst = ", !strconcat(opc , "($src1.L, $src2.L)")),
             [(set IntRegs:$dst, (IntID IntRegs:$src1, IntRegs:$src2))]>;

class si_ALU64_sisi_sat<string opc, Intrinsic IntID>
  : ALU64_rr<(outs IntRegs:$dst), (ins IntRegs:$src1, IntRegs:$src2),
             !strconcat("$dst = ", !strconcat(opc , "($src1, $src2):sat")),
             [(set IntRegs:$dst, (IntID IntRegs:$src1, IntRegs:$src2))]>;

//
// SInst classes.
//

class qi_SInst_qi<string opc, Intrinsic IntID>
  : SInst<(outs PredRegs:$dst), (ins IntRegs:$src),
             !strconcat("$dst = ", !strconcat(opc , "($src)")),
             [(set PredRegs:$dst, (IntID IntRegs:$src))]>;

class qi_SInst_qi_pxfer<string opc, Intrinsic IntID>
  : SInst<(outs PredRegs:$dst), (ins IntRegs:$src),
             !strconcat("$dst = ", !strconcat(opc , "$src")),
             [(set PredRegs:$dst, (IntID IntRegs:$src))]>;

class qi_SInst_qiqi<string opc, Intrinsic IntID>
  : SInst<(outs PredRegs:$dst), (ins IntRegs:$src1, IntRegs:$src2),
             !strconcat("$dst = ", !strconcat(opc , "($src1, $src2)")),
             [(set PredRegs:$dst, (IntID IntRegs:$src1, IntRegs:$src2))]>;

class qi_SInst_qiqi_neg<string opc, Intrinsic IntID>
  : SInst<(outs PredRegs:$dst), (ins IntRegs:$src1, IntRegs:$src2),
             !strconcat("$dst = ", !strconcat(opc , "($src1, !$src2)")),
             [(set PredRegs:$dst, (IntID IntRegs:$src1, IntRegs:$src2))]>;

class di_SInst_di<string opc, Intrinsic IntID>
  : SInst<(outs DoubleRegs:$dst), (ins DoubleRegs:$src),
             !strconcat("$dst = ", !strconcat(opc , "($src)")),
             [(set DoubleRegs:$dst, (IntID DoubleRegs:$src))]>;

class di_SInst_di_sat<string opc, Intrinsic IntID>
  : SInst<(outs DoubleRegs:$dst), (ins DoubleRegs:$src),
             !strconcat("$dst = ", !strconcat(opc , "($src):sat")),
             [(set DoubleRegs:$dst, (IntID DoubleRegs:$src))]>;

class si_SInst_di<string opc, Intrinsic IntID>
  : SInst<(outs IntRegs:$dst), (ins DoubleRegs:$src),
          !strconcat("$dst = ", !strconcat(opc , "($src)")),
          [(set IntRegs:$dst, (IntID DoubleRegs:$src))]>;

class si_SInst_di_sat<string opc, Intrinsic IntID>
  : SInst<(outs IntRegs:$dst), (ins DoubleRegs:$src),
          !strconcat("$dst = ", !strconcat(opc , "($src):sat")),
          [(set IntRegs:$dst, (IntID DoubleRegs:$src))]>;

class di_SInst_disi<string opc, Intrinsic IntID>
  : SInst<(outs DoubleRegs:$dst), (ins DoubleRegs:$src1, IntRegs:$src2),
          !strconcat("$dst = ", !strconcat(opc , "($src1, $src2)")),
          [(set DoubleRegs:$dst, (IntID DoubleRegs:$src1, IntRegs:$src2))]>;

class di_SInst_didi<string opc, Intrinsic IntID>
  : SInst<(outs DoubleRegs:$dst), (ins DoubleRegs:$src1, DoubleRegs:$src2),
          !strconcat("$dst = ", !strconcat(opc , "($src1, $src2)")),
          [(set DoubleRegs:$dst, (IntID DoubleRegs:$src1, DoubleRegs:$src2))]>;

class di_SInst_si<string opc, Intrinsic IntID>
  : SInst<(outs DoubleRegs:$dst), (ins IntRegs:$src1),
          !strconcat("$dst = ", !strconcat(opc , "($src1)")),
          [(set DoubleRegs:$dst, (IntID IntRegs:$src1))]>;

class si_SInst_sisiu3<string opc, Intrinsic IntID>
  : SInst<(outs IntRegs:$dst), (ins IntRegs:$src1, IntRegs:$src2, u3Imm:$src3),
          !strconcat("$dst = ", !strconcat(opc , "($src1, $src2, #$src3)")),
          [(set IntRegs:$dst, (IntID IntRegs:$src1, IntRegs:$src2,
                                     imm:$src3))]>;

class si_SInst_diu5<string opc, Intrinsic IntID>
  : SInst<(outs IntRegs:$dst), (ins DoubleRegs:$src1, u5Imm:$src2),
          !strconcat("$dst = ", !strconcat(opc , "($src1, #$src2)")),
          [(set IntRegs:$dst, (IntID DoubleRegs:$src1, imm:$src2))]>;

class si_SInst_disi<string opc, Intrinsic IntID>
  : SInst<(outs IntRegs:$dst), (ins DoubleRegs:$src1, IntRegs:$src2),
          !strconcat("$dst = ", !strconcat(opc , "($src1, $src2)")),
          [(set IntRegs:$dst, (IntID DoubleRegs:$src1, IntRegs:$src2))]>;

class si_SInst_sidi<string opc, Intrinsic IntID>
  : SInst<(outs IntRegs:$dst), (ins IntRegs:$src1, DoubleRegs:$src2),
          !strconcat("$dst = ", !strconcat(opc , "($src1, $src2)")),
          [(set IntRegs:$dst, (IntID IntRegs:$src1, DoubleRegs:$src2))]>;

class di_SInst_disisi<string opc, Intrinsic IntID>
  : SInst<(outs DoubleRegs:$dst), (ins DoubleRegs:$src1, IntRegs:$src2,
                                       IntRegs:$src3),
          !strconcat("$dst = ", !strconcat(opc , "($src1, $src2, $src3)")),
          [(set DoubleRegs:$dst, (IntID DoubleRegs:$src1, IntRegs:$src2,
                                        IntRegs:$src3))]>;

class di_SInst_sisi<string opc, Intrinsic IntID>
  : SInst<(outs DoubleRegs:$dst), (ins IntRegs:$src1, IntRegs:$src2),
          !strconcat("$dst = ", !strconcat(opc , "($src1, $src2)")),
          [(set DoubleRegs:$dst, (IntID IntRegs:$src1, IntRegs:$src2))]>;

class qi_SInst_siu5<string opc, Intrinsic IntID>
  : SInst<(outs PredRegs:$dst), (ins IntRegs:$src1, u5Imm:$src2),
          !strconcat("$dst = ", !strconcat(opc , "($src1, #$src2)")),
          [(set PredRegs:$dst, (IntID IntRegs:$src1, imm:$src2))]>;

class qi_SInst_siu6<string opc, Intrinsic IntID>
  : SInst<(outs PredRegs:$dst), (ins IntRegs:$src1, u6Imm:$src2),
          !strconcat("$dst = ", !strconcat(opc , "($src1, #$src2)")),
          [(set PredRegs:$dst, (IntID IntRegs:$src1, imm:$src2))]>;

class qi_SInst_sisi<string opc, Intrinsic IntID>
  : SInst<(outs PredRegs:$dst), (ins IntRegs:$src1, IntRegs:$src2),
          !strconcat("$dst = ", !strconcat(opc , "($src1, $src2)")),
          [(set PredRegs:$dst, (IntID IntRegs:$src1, IntRegs:$src2))]>;

class si_SInst_si<string opc, Intrinsic IntID>
  : SInst<(outs IntRegs:$dst), (ins IntRegs:$src),
          !strconcat("$dst = ", !strconcat(opc , "($src)")),
          [(set IntRegs:$dst, (IntID IntRegs:$src))]>;

class si_SInst_si_sat<string opc, Intrinsic IntID>
  : SInst<(outs IntRegs:$dst), (ins IntRegs:$src),
          !strconcat("$dst = ", !strconcat(opc , "($src):sat")),
          [(set IntRegs:$dst, (IntID IntRegs:$src))]>;

class di_SInst_qi<string opc, Intrinsic IntID>
  : SInst<(outs DoubleRegs:$dst), (ins IntRegs:$src),
          !strconcat("$dst = ", !strconcat(opc , "($src)")),
          [(set DoubleRegs:$dst, (IntID IntRegs:$src))]>;

class si_SInst_qi<string opc, Intrinsic IntID>
  : SInst<(outs IntRegs:$dst), (ins IntRegs:$src),
          !strconcat("$dst = ", !strconcat(opc , "$src")),
          [(set IntRegs:$dst, (IntID IntRegs:$src))]>;

class si_SInst_qiqi<string opc, Intrinsic IntID>
  : SInst<(outs IntRegs:$dst), (ins IntRegs:$src1, IntRegs:$src2),
          !strconcat("$dst = ", !strconcat(opc , "($src1, $src2)")),
          [(set IntRegs:$dst, (IntID IntRegs:$src1, IntRegs:$src2))]>;

class qi_SInst_si<string opc, Intrinsic IntID>
  : SInst<(outs PredRegs:$dst), (ins IntRegs:$src),
          !strconcat("$dst = ", !strconcat(opc , "$src")),
          [(set PredRegs:$dst, (IntID IntRegs:$src))]>;

class si_SInst_sisi<string opc, Intrinsic IntID>
  : SInst<(outs IntRegs:$dst), (ins IntRegs:$src1, IntRegs:$src2),
          !strconcat("$dst = ", !strconcat(opc , "($src1, $src2)")),
          [(set IntRegs:$dst, (IntID IntRegs:$src1, IntRegs:$src2))]>;

class di_SInst_diu6<string opc, Intrinsic IntID>
  : SInst<(outs DoubleRegs:$dst), (ins DoubleRegs:$src1, u6Imm:$src2),
          !strconcat("$dst = ", !strconcat(opc , "($src1, #$src2)")),
          [(set DoubleRegs:$dst, (IntID DoubleRegs:$src1, imm:$src2))]>;

class si_SInst_siu5<string opc, Intrinsic IntID>
  : SInst<(outs IntRegs:$dst), (ins IntRegs:$src1, u5Imm:$src2),
          !strconcat("$dst = ", !strconcat(opc , "($src1, #$src2)")),
          [(set IntRegs:$dst, (IntID IntRegs:$src1, imm:$src2))]>;

class si_SInst_siu5_rnd<string opc, Intrinsic IntID>
  : SInst<(outs IntRegs:$dst), (ins IntRegs:$src1, u5Imm:$src2),
          !strconcat("$dst = ", !strconcat(opc , "($src1, #$src2):rnd")),
          [(set IntRegs:$dst, (IntID IntRegs:$src1, imm:$src2))]>;

class si_SInst_siu5u5<string opc, Intrinsic IntID>
  : SInst<(outs IntRegs:$dst), (ins IntRegs:$src1, u5Imm:$src2, u5Imm:$src3),
          !strconcat("$dst = ", !strconcat(opc , "($src1, #$src2, #$src3)")),
          [(set IntRegs:$dst, (IntID IntRegs:$src1, imm:$src2, imm:$src3))]>;

class si_SInst_sisisi_acc<string opc, Intrinsic IntID>
  : SInst_acc<(outs IntRegs:$dst), (ins IntRegs:$dst2, IntRegs:$src1,
                                        IntRegs:$src2),
              !strconcat("$dst += ", !strconcat(opc , "($src1, $src2)")),
              [(set IntRegs:$dst, (IntID IntRegs:$dst2, IntRegs:$src1,
                                         IntRegs:$src2))],
              "$dst2 = $dst">;

class si_SInst_sisisi_nac<string opc, Intrinsic IntID>
  : SInst_acc<(outs IntRegs:$dst), (ins IntRegs:$dst2, IntRegs:$src1,
                                        IntRegs:$src2),
              !strconcat("$dst -= ", !strconcat(opc , "($src1, $src2)")),
              [(set IntRegs:$dst, (IntID IntRegs:$dst2, IntRegs:$src1,
                                         IntRegs:$src2))],
              "$dst2 = $dst">;

class di_SInst_didisi_acc<string opc, Intrinsic IntID>
  : SInst_acc<(outs DoubleRegs:$dst), (ins DoubleRegs:$dst2, DoubleRegs:$src1,
                                           IntRegs:$src2),
               !strconcat("$dst += ", !strconcat(opc , "($src1, $src2)")),
               [(set DoubleRegs:$dst, (IntID DoubleRegs:$dst2,
                                             DoubleRegs:$src1,
                                             IntRegs:$src2))],
               "$dst2 = $dst">;

class di_SInst_didisi_nac<string opc, Intrinsic IntID>
  : SInst_acc<(outs DoubleRegs:$dst), (ins DoubleRegs:$dst2, DoubleRegs:$src1,
                                           IntRegs:$src2),
          !strconcat("$dst -= ", !strconcat(opc , "($src1, $src2)")),
          [(set DoubleRegs:$dst, (IntID DoubleRegs:$dst2,
                                        DoubleRegs:$src1, IntRegs:$src2))],
          "$dst2 = $dst">;

class si_SInst_sisiu5u5<string opc, Intrinsic IntID>
  : SInst_acc<(outs IntRegs:$dst), (ins IntRegs:$dst2, IntRegs:$src1,
                                        u5Imm:$src2, u5Imm:$src3),
              !strconcat("$dst = ", !strconcat(opc ,
                                               "($src1, #$src2, #$src3)")),
              [(set IntRegs:$dst, (IntID IntRegs:$dst2, IntRegs:$src1,
                                         imm:$src2, imm:$src3))],
              "$dst2 = $dst">;

class si_SInst_sisidi<string opc, Intrinsic IntID>
  : SInst_acc<(outs IntRegs:$dst), (ins IntRegs:$dst2, IntRegs:$src1,
                                        DoubleRegs:$src2),
              !strconcat("$dst = ", !strconcat(opc , "($src1, $src2)")),
              [(set IntRegs:$dst, (IntID IntRegs:$dst2, IntRegs:$src1,
                                         DoubleRegs:$src2))],
              "$dst2 = $dst">;

class di_SInst_didiu6u6<string opc, Intrinsic IntID>
  : SInst_acc<(outs DoubleRegs:$dst), (ins DoubleRegs:$dst2, DoubleRegs:$src1,
                                           u6Imm:$src2, u6Imm:$src3),
              !strconcat("$dst = ", !strconcat(opc ,
                                               "($src1, #$src2, #$src3)")),
              [(set DoubleRegs:$dst, (IntID DoubleRegs:$dst2, DoubleRegs:$src1,
                                            imm:$src2, imm:$src3))],
              "$dst2 = $dst">;

class di_SInst_dididi<string opc, Intrinsic IntID>
  : SInst_acc<(outs DoubleRegs:$dst), (ins DoubleRegs:$dst2, DoubleRegs:$src1,
                                           DoubleRegs:$src2),
              !strconcat("$dst = ", !strconcat(opc , "($src1, $src2)")),
              [(set DoubleRegs:$dst, (IntID DoubleRegs:$dst2,
                                            DoubleRegs:$src1,
                                            DoubleRegs:$src2))],
              "$dst2 = $dst">;

class di_SInst_diu6u6<string opc, Intrinsic IntID>
  : SInst<(outs DoubleRegs:$dst), (ins DoubleRegs:$src1, u6Imm:$src2,
                                       u6Imm:$src3),
          !strconcat("$dst = ", !strconcat(opc , "($src1, #$src2, #$src3)")),
          [(set DoubleRegs:$dst, (IntID DoubleRegs:$src1, imm:$src2,
                                        imm:$src3))]>;

class di_SInst_didiqi<string opc, Intrinsic IntID>
  : SInst<(outs DoubleRegs:$dst), (ins DoubleRegs:$src1, DoubleRegs:$src2,
                                       IntRegs:$src3),
          !strconcat("$dst = ", !strconcat(opc , "($src1, $src2, $src3)")),
          [(set DoubleRegs:$dst, (IntID DoubleRegs:$src1, DoubleRegs:$src2,
                                        IntRegs:$src3))]>;

class di_SInst_didiu3<string opc, Intrinsic IntID>
  : SInst<(outs DoubleRegs:$dst), (ins DoubleRegs:$src1, DoubleRegs:$src2,
                                       u3Imm:$src3),
          !strconcat("$dst = ", !strconcat(opc , "($src1, $src2, #$src3)")),
          [(set DoubleRegs:$dst, (IntID DoubleRegs:$src1, DoubleRegs:$src2,
                                        imm:$src3))]>;

class di_SInst_didisi_or<string opc, Intrinsic IntID>
  : SInst_acc<(outs DoubleRegs:$dst), (ins DoubleRegs:$dst2, DoubleRegs:$src1,
                                           IntRegs:$src2),
          !strconcat("$dst |= ", !strconcat(opc , "($src1, $src2)")),
          [(set DoubleRegs:$dst, (IntID DoubleRegs:$dst2, DoubleRegs:$src1,
                                        IntRegs:$src2))],
          "$dst2 = $dst">;

class di_SInst_didisi_and<string opc, Intrinsic IntID>
  : SInst_acc<(outs DoubleRegs:$dst), (ins DoubleRegs:$dst2, DoubleRegs:$src1,
                                           IntRegs:$src2),
          !strconcat("$dst &= ", !strconcat(opc , "($src1, $src2)")),
          [(set DoubleRegs:$dst, (IntID DoubleRegs:$dst2, DoubleRegs:$src1,
                                        IntRegs:$src2))],
          "$dst2 = $dst">;

class di_SInst_didiu6_and<string opc, Intrinsic IntID>
  : SInst_acc<(outs DoubleRegs:$dst), (ins DoubleRegs:$dst2, DoubleRegs:$src1,
                                           u6Imm:$src2),
          !strconcat("$dst &= ", !strconcat(opc , "($src1, #$src2)")),
          [(set DoubleRegs:$dst, (IntID DoubleRegs:$dst2, DoubleRegs:$src1,
                                        imm:$src2))],
          "$dst2 = $dst">;

class di_SInst_didiu6_or<string opc, Intrinsic IntID>
  : SInst_acc<(outs DoubleRegs:$dst), (ins DoubleRegs:$dst2, DoubleRegs:$src1,
                                           u6Imm:$src2),
          !strconcat("$dst |= ", !strconcat(opc , "($src1, #$src2)")),
          [(set DoubleRegs:$dst, (IntID DoubleRegs:$dst2, DoubleRegs:$src1,
                                        imm:$src2))],
          "$dst2 = $dst">;

class di_SInst_didiu6_xor<string opc, Intrinsic IntID>
  : SInst_acc<(outs DoubleRegs:$dst), (ins DoubleRegs:$dst2, DoubleRegs:$src1,
                                           u6Imm:$src2),
          !strconcat("$dst ^= ", !strconcat(opc , "($src1, #$src2)")),
          [(set DoubleRegs:$dst, (IntID DoubleRegs:$dst2, DoubleRegs:$src1,
                                        imm:$src2))],
          "$dst2 = $dst">;

class si_SInst_sisisi_and<string opc, Intrinsic IntID>
  : SInst_acc<(outs IntRegs:$dst), (ins IntRegs:$dst2, IntRegs:$src1,
                                        IntRegs:$src2),
              !strconcat("$dst &= ", !strconcat(opc , "($src1, $src2)")),
              [(set IntRegs:$dst, (IntID IntRegs:$dst2, IntRegs:$src1,
                                         IntRegs:$src2))],
              "$dst2 = $dst">;

class si_SInst_sisisi_or<string opc, Intrinsic IntID>
  : SInst_acc<(outs IntRegs:$dst), (ins IntRegs:$dst2, IntRegs:$src1,
                                        IntRegs:$src2),
              !strconcat("$dst |= ", !strconcat(opc , "($src1, $src2)")),
              [(set IntRegs:$dst, (IntID IntRegs:$dst2, IntRegs:$src1,
                                         IntRegs:$src2))],
              "$dst2 = $dst">;


class si_SInst_sisiu5_and<string opc, Intrinsic IntID>
  : SInst_acc<(outs IntRegs:$dst), (ins IntRegs:$dst2, IntRegs:$src1,
                                        u5Imm:$src2),
              !strconcat("$dst &= ", !strconcat(opc , "($src1, #$src2)")),
              [(set IntRegs:$dst, (IntID IntRegs:$dst2, IntRegs:$src1,
                                         imm:$src2))],
              "$dst2 = $dst">;

class si_SInst_sisiu5_or<string opc, Intrinsic IntID>
  : SInst_acc<(outs IntRegs:$dst), (ins IntRegs:$dst2, IntRegs:$src1,
                                        u5Imm:$src2),
              !strconcat("$dst |= ", !strconcat(opc , "($src1, #$src2)")),
              [(set IntRegs:$dst, (IntID IntRegs:$dst2, IntRegs:$src1,
                                         imm:$src2))],
              "$dst2 = $dst">;

class si_SInst_sisiu5_xor<string opc, Intrinsic IntID>
  : SInst_acc<(outs IntRegs:$dst), (ins IntRegs:$dst2, IntRegs:$src1,
                                        u5Imm:$src2),
              !strconcat("$dst ^= ", !strconcat(opc , "($src1, #$src2)")),
              [(set IntRegs:$dst, (IntID IntRegs:$dst2, IntRegs:$src1,
                                         imm:$src2))],
              "$dst2 = $dst">;

class si_SInst_sisiu5_acc<string opc, Intrinsic IntID>
  : SInst_acc<(outs IntRegs:$dst), (ins IntRegs:$dst2, IntRegs:$src1,
                                        u5Imm:$src2),
              !strconcat("$dst += ", !strconcat(opc , "($src1, #$src2)")),
              [(set IntRegs:$dst, (IntID IntRegs:$dst2, IntRegs:$src1,
                                         imm:$src2))],
              "$dst2 = $dst">;

class si_SInst_sisiu5_nac<string opc, Intrinsic IntID>
  : SInst_acc<(outs IntRegs:$dst), (ins IntRegs:$dst2, IntRegs:$src1,
                                        u5Imm:$src2),
              !strconcat("$dst -= ", !strconcat(opc , "($src1, #$src2)")),
              [(set IntRegs:$dst, (IntID IntRegs:$dst2, IntRegs:$src1,
                                         imm:$src2))],
              "$dst2 = $dst">;

class di_SInst_didiu6_acc<string opc, Intrinsic IntID>
  : SInst_acc<(outs DoubleRegs:$dst), (ins DoubleRegs:$dst2, DoubleRegs:$src1,
                                           u5Imm:$src2),
              !strconcat("$dst += ", !strconcat(opc , "($src1, #$src2)")),
              [(set DoubleRegs:$dst, (IntID DoubleRegs:$dst2,
                                            DoubleRegs:$src1, imm:$src2))],
              "$dst2 = $dst">;

class di_SInst_didiu6_nac<string opc, Intrinsic IntID>
  : SInst_acc<(outs DoubleRegs:$dst), (ins DoubleRegs:$dst2, DoubleRegs:$src1,
                                           u5Imm:$src2),
              !strconcat("$dst -= ", !strconcat(opc , "($src1, #$src2)")),
              [(set DoubleRegs:$dst, (IntID DoubleRegs:$dst2, DoubleRegs:$src1,
                                            imm:$src2))],
              "$dst2 = $dst">;


//
// MInst classes.
//

class di_MInst_sisi_rnd_hh_s1<string opc, Intrinsic IntID>
  : MInst<(outs DoubleRegs:$dst), (ins IntRegs:$src1, IntRegs:$src2),
               !strconcat("$dst = ", !strconcat(opc ,
                                                "($src1.H, $src2.H):<<1:rnd")),
               [(set DoubleRegs:$dst, (IntID IntRegs:$src1, IntRegs:$src2))]>;

class di_MInst_sisi_rnd_hh<string opc, Intrinsic IntID>
  : MInst<(outs DoubleRegs:$dst), (ins IntRegs:$src1, IntRegs:$src2),
               !strconcat("$dst = ", !strconcat(opc ,
                                                "($src1.H, $src2.H):rnd")),
               [(set DoubleRegs:$dst, (IntID IntRegs:$src1, IntRegs:$src2))]>;

class di_MInst_sisi_rnd_hl_s1<string opc, Intrinsic IntID>
  : MInst<(outs DoubleRegs:$dst), (ins IntRegs:$src1, IntRegs:$src2),
               !strconcat("$dst = ", !strconcat(opc ,
                                                "($src1.H, $src2.L):<<1:rnd")),
               [(set DoubleRegs:$dst, (IntID IntRegs:$src1, IntRegs:$src2))]>;

class di_MInst_sisi_rnd_hl<string opc, Intrinsic IntID>
  : MInst<(outs DoubleRegs:$dst), (ins IntRegs:$src1, IntRegs:$src2),
               !strconcat("$dst = ", !strconcat(opc ,
                                                "($src1.H, $src2.L):rnd")),
               [(set DoubleRegs:$dst, (IntID IntRegs:$src1, IntRegs:$src2))]>;

class di_MInst_sisi_rnd_lh_s1<string opc, Intrinsic IntID>
  : MInst<(outs DoubleRegs:$dst), (ins IntRegs:$src1, IntRegs:$src2),
               !strconcat("$dst = ", !strconcat(opc ,
                                                "($src1.L, $src2.H):<<1:rnd")),
               [(set DoubleRegs:$dst, (IntID IntRegs:$src1, IntRegs:$src2))]>;

class di_MInst_sisi_rnd_lh<string opc, Intrinsic IntID>
  : MInst<(outs DoubleRegs:$dst), (ins IntRegs:$src1, IntRegs:$src2),
               !strconcat("$dst = ", !strconcat(opc ,
                                                "($src1.L, $src2.H):rnd")),
               [(set DoubleRegs:$dst, (IntID IntRegs:$src1, IntRegs:$src2))]>;

class di_MInst_sisi_rnd_ll_s1<string opc, Intrinsic IntID>
  : MInst<(outs DoubleRegs:$dst), (ins IntRegs:$src1, IntRegs:$src2),
               !strconcat("$dst = ", !strconcat(opc ,
                                                "($src1.L, $src2.L):<<1:rnd")),
               [(set DoubleRegs:$dst, (IntID IntRegs:$src1, IntRegs:$src2))]>;

class di_MInst_sisi_rnd_ll<string opc, Intrinsic IntID>
  : MInst<(outs DoubleRegs:$dst), (ins IntRegs:$src1, IntRegs:$src2),
               !strconcat("$dst = ", !strconcat(opc ,
                                                "($src1.L, $src2.L):rnd")),
               [(set DoubleRegs:$dst, (IntID IntRegs:$src1, IntRegs:$src2))]>;

class di_MInst_disisi_acc<string opc, Intrinsic IntID>
  : MInst_acc<(outs DoubleRegs:$dst), (ins DoubleRegs:$dst2, IntRegs:$src1,
                                           IntRegs:$src2),
             !strconcat("$dst += ", !strconcat(opc , "($src1, $src2)")),
             [(set DoubleRegs:$dst, (IntID DoubleRegs:$dst2, IntRegs:$src1,
                                           IntRegs:$src2))],
             "$dst2 = $dst">;

class di_MInst_disisi_nac<string opc, Intrinsic IntID>
  : MInst_acc<(outs DoubleRegs:$dst), (ins DoubleRegs:$dst2, IntRegs:$src1,
                                           IntRegs:$src2),
             !strconcat("$dst -= ", !strconcat(opc , "($src1, $src2)")),
             [(set DoubleRegs:$dst, (IntID DoubleRegs:$dst2, IntRegs:$src1,
                                           IntRegs:$src2))],
             "$dst2 = $dst">;

class di_MInst_disisi_acc_sat<string opc, Intrinsic IntID>
  : MInst_acc<(outs DoubleRegs:$dst), (ins DoubleRegs:$dst2, IntRegs:$src1,
                                           IntRegs:$src2),
             !strconcat("$dst += ", !strconcat(opc , "($src1, $src2):sat")),
             [(set DoubleRegs:$dst, (IntID DoubleRegs:$dst2, IntRegs:$src1,
                                           IntRegs:$src2))],
             "$dst2 = $dst">;

class di_MInst_disisi_nac_sat<string opc, Intrinsic IntID>
  : MInst_acc<(outs DoubleRegs:$dst), (ins DoubleRegs:$dst2, IntRegs:$src1,
                                           IntRegs:$src2),
             !strconcat("$dst -= ", !strconcat(opc , "($src1, $src2):sat")),
             [(set DoubleRegs:$dst, (IntID DoubleRegs:$dst2, IntRegs:$src1,
                                           IntRegs:$src2))],
             "$dst2 = $dst">;

class di_MInst_disisi_acc_sat_conj<string opc, Intrinsic IntID>
  : MInst_acc<(outs DoubleRegs:$dst), (ins DoubleRegs:$dst2, IntRegs:$src1,
                                           IntRegs:$src2),
             !strconcat("$dst += ", !strconcat(opc , "($src1, $src2*):sat")),
             [(set DoubleRegs:$dst, (IntID DoubleRegs:$dst2, IntRegs:$src1,
                                           IntRegs:$src2))],
             "$dst2 = $dst">;

class di_MInst_disisi_nac_sat_conj<string opc, Intrinsic IntID>
  : MInst_acc<(outs DoubleRegs:$dst), (ins DoubleRegs:$dst2, IntRegs:$src1,
                                           IntRegs:$src2),
             !strconcat("$dst -= ", !strconcat(opc , "($src1, $src2*):sat")),
             [(set DoubleRegs:$dst, (IntID DoubleRegs:$dst2, IntRegs:$src1,
                                           IntRegs:$src2))],
             "$dst2 = $dst">;

class di_MInst_disisi_nac_s1_sat<string opc, Intrinsic IntID>
  : MInst_acc<(outs DoubleRegs:$dst), (ins DoubleRegs:$dst2, IntRegs:$src1,
                                           IntRegs:$src2),
             !strconcat("$dst -= ", !strconcat(opc ,
                                               "($src1, $src2):<<1:sat")),
             [(set DoubleRegs:$dst, (IntID DoubleRegs:$dst2, IntRegs:$src1,
                                           IntRegs:$src2))],
             "$dst2 = $dst">;

class di_MInst_disisi_acc_s1_sat_conj<string opc, Intrinsic IntID>
  : MInst_acc<(outs DoubleRegs:$dst), (ins DoubleRegs:$dst2, IntRegs:$src1,
                                           IntRegs:$src2),
             !strconcat("$dst += ", !strconcat(opc ,
                                               "($src1, $src2*):<<1:sat")),
             [(set DoubleRegs:$dst, (IntID DoubleRegs:$dst2, IntRegs:$src1,
                                           IntRegs:$src2))],
             "$dst2 = $dst">;

class di_MInst_disisi_nac_s1_sat_conj<string opc, Intrinsic IntID>
  : MInst_acc<(outs DoubleRegs:$dst), (ins DoubleRegs:$dst2, IntRegs:$src1,
                                           IntRegs:$src2),
             !strconcat("$dst -= ", !strconcat(opc ,
                                               "($src1, $src2*):<<1:sat")),
             [(set DoubleRegs:$dst, (IntID DoubleRegs:$dst2, IntRegs:$src1,
                                           IntRegs:$src2))],
             "$dst2 = $dst">;

class di_MInst_s8s8<string opc, Intrinsic IntID>
  : MInst<(outs DoubleRegs:$dst), (ins s8Imm:$src1, s8Imm:$src2),
             !strconcat("$dst = ", !strconcat(opc , "(#$src1, #$src2)")),
             [(set DoubleRegs:$dst, (IntID imm:$src1, imm:$src2))]>;

class si_MInst_sis9<string opc, Intrinsic IntID>
  : MInst<(outs IntRegs:$dst), (ins IntRegs:$src1, s9Imm:$src2),
             !strconcat("$dst = ", !strconcat(opc , "($src1, #$src2)")),
             [(set IntRegs:$dst, (IntID IntRegs:$src1, imm:$src2))]>;

class si_MInst_sisi<string opc, Intrinsic IntID>
  : MInst<(outs IntRegs:$dst), (ins IntRegs:$src1, IntRegs:$src2),
             !strconcat("$dst = ", !strconcat(opc , "($src1, $src2)")),
             [(set IntRegs:$dst, (IntID IntRegs:$src1, IntRegs:$src2))]>;

class di_MInst_sisi_hh<string opc, Intrinsic IntID>
  : MInst<(outs DoubleRegs:$dst), (ins IntRegs:$src1, IntRegs:$src2),
             !strconcat("$dst = ", !strconcat(opc , "($src1.H, $src2.H)")),
             [(set DoubleRegs:$dst, (IntID IntRegs:$src1, IntRegs:$src2))]>;

class di_MInst_sisi_hh_s1<string opc, Intrinsic IntID>
  : MInst<(outs DoubleRegs:$dst), (ins IntRegs:$src1, IntRegs:$src2),
             !strconcat("$dst = ", !strconcat(opc , "($src1.H, $src2.H):<<1")),
             [(set DoubleRegs:$dst, (IntID IntRegs:$src1, IntRegs:$src2))]>;

class di_MInst_sisi_lh<string opc, Intrinsic IntID>
  : MInst<(outs DoubleRegs:$dst), (ins IntRegs:$src1, IntRegs:$src2),
             !strconcat("$dst = ", !strconcat(opc , "($src1.L, $src2.H)")),
             [(set DoubleRegs:$dst, (IntID IntRegs:$src1, IntRegs:$src2))]>;

class di_MInst_sisi_lh_s1<string opc, Intrinsic IntID>
  : MInst<(outs DoubleRegs:$dst), (ins IntRegs:$src1, IntRegs:$src2),
             !strconcat("$dst = ", !strconcat(opc , "($src1.L, $src2.H):<<1")),
             [(set DoubleRegs:$dst, (IntID IntRegs:$src1, IntRegs:$src2))]>;

class di_MInst_sisi_hl<string opc, Intrinsic IntID>
  : MInst<(outs DoubleRegs:$dst), (ins IntRegs:$src1, IntRegs:$src2),
             !strconcat("$dst = ", !strconcat(opc , "($src1.H, $src2.L)")),
             [(set DoubleRegs:$dst, (IntID IntRegs:$src1, IntRegs:$src2))]>;

class di_MInst_sisi_hl_s1<string opc, Intrinsic IntID>
  : MInst<(outs DoubleRegs:$dst), (ins IntRegs:$src1, IntRegs:$src2),
             !strconcat("$dst = ", !strconcat(opc , "($src1.H, $src2.L):<<1")),
             [(set DoubleRegs:$dst, (IntID IntRegs:$src1, IntRegs:$src2))]>;

class di_MInst_sisi_ll<string opc, Intrinsic IntID>
  : MInst<(outs DoubleRegs:$dst), (ins IntRegs:$src1, IntRegs:$src2),
             !strconcat("$dst = ", !strconcat(opc , "($src1.L, $src2.L)")),
             [(set DoubleRegs:$dst, (IntID IntRegs:$src1, IntRegs:$src2))]>;

class di_MInst_sisi_ll_s1<string opc, Intrinsic IntID>
  : MInst<(outs DoubleRegs:$dst), (ins IntRegs:$src1, IntRegs:$src2),
             !strconcat("$dst = ", !strconcat(opc , "($src1.L, $src2.L):<<1")),
             [(set DoubleRegs:$dst, (IntID IntRegs:$src1, IntRegs:$src2))]>;


class si_MInst_sisi_hh<string opc, Intrinsic IntID>
  : MInst<(outs IntRegs:$dst), (ins IntRegs:$src1, IntRegs:$src2),
             !strconcat("$dst = ", !strconcat(opc , "($src1.H, $src2.H)")),
             [(set IntRegs:$dst, (IntID IntRegs:$src1, IntRegs:$src2))]>;

class si_MInst_sisi_hh_s1<string opc, Intrinsic IntID>
  : MInst<(outs IntRegs:$dst), (ins IntRegs:$src1, IntRegs:$src2),
             !strconcat("$dst = ", !strconcat(opc , "($src1.H, $src2.H):<<1")),
             [(set IntRegs:$dst, (IntID IntRegs:$src1, IntRegs:$src2))]>;

class si_MInst_sisi_lh<string opc, Intrinsic IntID>
  : MInst<(outs IntRegs:$dst), (ins IntRegs:$src1, IntRegs:$src2),
             !strconcat("$dst = ", !strconcat(opc , "($src1.L, $src2.H)")),
             [(set IntRegs:$dst, (IntID IntRegs:$src1, IntRegs:$src2))]>;

class si_MInst_sisi_lh_s1<string opc, Intrinsic IntID>
  : MInst<(outs IntRegs:$dst), (ins IntRegs:$src1, IntRegs:$src2),
             !strconcat("$dst = ", !strconcat(opc , "($src1.L, $src2.H):<<1")),
             [(set IntRegs:$dst, (IntID IntRegs:$src1, IntRegs:$src2))]>;

class si_MInst_sisi_hl<string opc, Intrinsic IntID>
  : MInst<(outs IntRegs:$dst), (ins IntRegs:$src1, IntRegs:$src2),
             !strconcat("$dst = ", !strconcat(opc , "($src1.H, $src2.L)")),
             [(set IntRegs:$dst, (IntID IntRegs:$src1, IntRegs:$src2))]>;

class si_MInst_sisi_hl_s1<string opc, Intrinsic IntID>
  : MInst<(outs IntRegs:$dst), (ins IntRegs:$src1, IntRegs:$src2),
             !strconcat("$dst = ", !strconcat(opc , "($src1.H, $src2.L):<<1")),
             [(set IntRegs:$dst, (IntID IntRegs:$src1, IntRegs:$src2))]>;

class si_MInst_sisi_ll<string opc, Intrinsic IntID>
  : MInst<(outs IntRegs:$dst), (ins IntRegs:$src1, IntRegs:$src2),
             !strconcat("$dst = ", !strconcat(opc , "($src1.L, $src2.L)")),
             [(set IntRegs:$dst, (IntID IntRegs:$src1, IntRegs:$src2))]>;

class si_MInst_sisi_ll_s1<string opc, Intrinsic IntID>
  : MInst<(outs IntRegs:$dst), (ins IntRegs:$src1, IntRegs:$src2),
             !strconcat("$dst = ", !strconcat(opc , "($src1.L, $src2.L):<<1")),
             [(set IntRegs:$dst, (IntID IntRegs:$src1, IntRegs:$src2))]>;

class si_MInst_sisi_up<string opc, Intrinsic IntID>
  : MInst<(outs IntRegs:$dst), (ins IntRegs:$src1, IntRegs:$src2),
             !strconcat("$dst = ", !strconcat(opc , "($src1, $src2)")),
             [(set IntRegs:$dst, (IntID IntRegs:$src1, IntRegs:$src2))]>;

class di_MInst_didi<string opc, Intrinsic IntID>
  : MInst<(outs DoubleRegs:$dst), (ins DoubleRegs:$src1, DoubleRegs:$src2),
             !strconcat("$dst = ", !strconcat(opc , "($src1, $src2)")),
             [(set DoubleRegs:$dst, (IntID DoubleRegs:$src1,
                                           DoubleRegs:$src2))]>;

class di_MInst_didi_conj<string opc, Intrinsic IntID>
  : MInst<(outs DoubleRegs:$dst), (ins DoubleRegs:$src1, DoubleRegs:$src2),
             !strconcat("$dst = ", !strconcat(opc , "($src1, $src2*)")),
             [(set DoubleRegs:$dst, (IntID DoubleRegs:$src1,
                                           DoubleRegs:$src2))]>;

class di_MInst_sisi_s1_sat_conj<string opc, Intrinsic IntID>
  : MInst<(outs DoubleRegs:$dst), (ins IntRegs:$src1, IntRegs:$src2),
             !strconcat("$dst = ", !strconcat(opc ,
                                              "($src1, $src2*):<<1:sat")),
             [(set DoubleRegs:$dst, (IntID IntRegs:$src1, IntRegs:$src2))]>;

class di_MInst_didi_s1_rnd_sat<string opc, Intrinsic IntID>
  : MInst<(outs DoubleRegs:$dst), (ins DoubleRegs:$src1, DoubleRegs:$src2),
             !strconcat("$dst = ", !strconcat(opc ,
                                              "($src1, $src2):<<1:rnd:sat")),
             [(set DoubleRegs:$dst, (IntID DoubleRegs:$src1,
                                           DoubleRegs:$src2))]>;

class di_MInst_didi_sat<string opc, Intrinsic IntID>
  : MInst<(outs DoubleRegs:$dst), (ins DoubleRegs:$src1, DoubleRegs:$src2),
             !strconcat("$dst = ", !strconcat(opc , "($src1, $src2):sat")),
             [(set DoubleRegs:$dst, (IntID DoubleRegs:$src1,
                                           DoubleRegs:$src2))]>;

class di_MInst_didi_rnd_sat<string opc, Intrinsic IntID>
  : MInst<(outs DoubleRegs:$dst), (ins DoubleRegs:$src1, DoubleRegs:$src2),
             !strconcat("$dst = ", !strconcat(opc ,
                                              "($src1, $src2):rnd:sat")),
             [(set DoubleRegs:$dst, (IntID DoubleRegs:$src1,
                                           DoubleRegs:$src2))]>;

class si_SInst_sisi_sat<string opc, Intrinsic IntID>
  : SInst<(outs IntRegs:$dst), (ins IntRegs:$src1, IntRegs:$src2),
          !strconcat("$dst = ", !strconcat(opc , "($src1, $src2):sat")),
          [(set IntRegs:$dst, (IntID IntRegs:$src1, IntRegs:$src2))]>;

class si_SInst_didi_sat<string opc, Intrinsic IntID>
  : SInst<(outs IntRegs:$dst), (ins DoubleRegs:$src1, DoubleRegs:$src2),
          !strconcat("$dst = ", !strconcat(opc , "($src1, $src2):sat")),
          [(set IntRegs:$dst, (IntID DoubleRegs:$src1, DoubleRegs:$src2))]>;

class si_SInst_disi_s1_rnd_sat<string opc, Intrinsic IntID>
  : MInst<(outs IntRegs:$dst), (ins DoubleRegs:$src1, IntRegs:$src2),
             !strconcat("$dst = ", !strconcat(opc ,
                                              "($src1, $src2):<<1:rnd:sat")),
             [(set IntRegs:$dst, (IntID DoubleRegs:$src1, IntRegs:$src2))]>;

class si_MInst_sisi_s1_rnd_sat<string opc, Intrinsic IntID>
  : MInst<(outs IntRegs:$dst), (ins IntRegs:$src1, IntRegs:$src2),
             !strconcat("$dst = ", !strconcat(opc ,
                                              "($src1, $src2):<<1:rnd:sat")),
             [(set IntRegs:$dst, (IntID IntRegs:$src1, IntRegs:$src2))]>;

class si_MInst_sisi_l_s1_rnd_sat<string opc, Intrinsic IntID>
  : MInst<(outs IntRegs:$dst), (ins IntRegs:$src1, IntRegs:$src2),
             !strconcat("$dst = ", !strconcat(opc ,
                                              "($src1, $src2.L):<<1:rnd:sat")),
             [(set IntRegs:$dst, (IntID IntRegs:$src1, IntRegs:$src2))]>;

class si_MInst_sisi_h_s1_rnd_sat<string opc, Intrinsic IntID>
  : MInst<(outs IntRegs:$dst), (ins IntRegs:$src1, IntRegs:$src2),
             !strconcat("$dst = ", !strconcat(opc ,
                                              "($src1, $src2.H):<<1:rnd:sat")),
             [(set IntRegs:$dst, (IntID IntRegs:$src1, IntRegs:$src2))]>;

class si_MInst_sisi_rnd_sat_conj<string opc, Intrinsic IntID>
  : MInst<(outs IntRegs:$dst), (ins IntRegs:$src1, IntRegs:$src2),
             !strconcat("$dst = ", !strconcat(opc ,
                                              "($src1, $src2*):rnd:sat")),
             [(set IntRegs:$dst, (IntID IntRegs:$src1, IntRegs:$src2))]>;

class si_MInst_sisi_s1_rnd_sat_conj<string opc, Intrinsic IntID>
  : MInst<(outs IntRegs:$dst), (ins IntRegs:$src1, IntRegs:$src2),
             !strconcat("$dst = ", !strconcat(opc ,
                                              "($src1, $src2*):<<1:rnd:sat")),
             [(set IntRegs:$dst, (IntID IntRegs:$src1, IntRegs:$src2))]>;

class si_MInst_sisi_rnd_sat<string opc, Intrinsic IntID>
  : MInst<(outs IntRegs:$dst), (ins IntRegs:$src1, IntRegs:$src2),
             !strconcat("$dst = ", !strconcat(opc ,
                                              "($src1, $src2):rnd:sat")),
             [(set IntRegs:$dst, (IntID IntRegs:$src1, IntRegs:$src2))]>;

class si_MInst_sisi_rnd<string opc, Intrinsic IntID>
  : MInst<(outs IntRegs:$dst), (ins IntRegs:$src1, IntRegs:$src2),
             !strconcat("$dst = ", !strconcat(opc , "($src1, $src2):rnd")),
             [(set IntRegs:$dst, (IntID IntRegs:$src1, IntRegs:$src2))]>;

class si_MInst_sisisi_xacc<string opc, Intrinsic IntID>
  : MInst_acc<(outs IntRegs:$dst), (ins IntRegs:$dst2, IntRegs:$src2,
                                        IntRegs:$src3),
             !strconcat("$dst ^= ", !strconcat(opc , "($src2, $src3)")),
             [(set IntRegs:$dst, (IntID IntRegs:$dst2, IntRegs:$src2,
                                        IntRegs:$src3))],
             "$dst2 = $dst">;

class si_MInst_sisisi_acc<string opc, Intrinsic IntID>
  : MInst_acc<(outs IntRegs:$dst), (ins IntRegs:$dst2, IntRegs:$src2,
                                        IntRegs:$src3),
             !strconcat("$dst += ", !strconcat(opc , "($src2, $src3)")),
             [(set IntRegs:$dst, (IntID IntRegs:$dst2, IntRegs:$src2,
                                        IntRegs:$src3))],
             "$dst2 = $dst">;

class si_MInst_sisisi_nac<string opc, Intrinsic IntID>
  : MInst_acc<(outs IntRegs:$dst), (ins IntRegs:$dst2, IntRegs:$src2,
                                        IntRegs:$src3),
             !strconcat("$dst -= ", !strconcat(opc , "($src2, $src3)")),
             [(set IntRegs:$dst, (IntID IntRegs:$dst2, IntRegs:$src2,
                                        IntRegs:$src3))],
             "$dst2 = $dst">;

class si_MInst_sisis8_acc<string opc, Intrinsic IntID>
  : MInst_acc<(outs IntRegs:$dst), (ins IntRegs:$dst2, IntRegs:$src2,
                                        s8Imm:$src3),
             !strconcat("$dst += ", !strconcat(opc , "($src2, #$src3)")),
             [(set IntRegs:$dst, (IntID IntRegs:$dst2, IntRegs:$src2,
                                        imm:$src3))],
             "$dst2 = $dst">;

class si_MInst_sisis8_nac<string opc, Intrinsic IntID>
  : MInst_acc<(outs IntRegs:$dst), (ins IntRegs:$dst2, IntRegs:$src2,
                                        s8Imm:$src3),
             !strconcat("$dst -= ", !strconcat(opc , "($src2, #$src3)")),
             [(set IntRegs:$dst, (IntID IntRegs:$dst2, IntRegs:$src2,
                                        imm:$src3))],
             "$dst2 = $dst">;

class si_MInst_sisiu4u5<string opc, Intrinsic IntID>
  : MInst_acc<(outs IntRegs:$dst), (ins IntRegs:$dst2, IntRegs:$src1,
                                        u4Imm:$src2, u5Imm:$src3),
               !strconcat("$dst = ", !strconcat(opc ,
                                                "($src1, #$src2, #$src3)")),
               [(set IntRegs:$dst, (IntID IntRegs:$dst2, IntRegs:$src1,
                                          imm:$src2, imm:$src3))],
               "$dst2 = $dst">;

class si_MInst_sisiu8_acc<string opc, Intrinsic IntID>
  : MInst_acc<(outs IntRegs:$dst), (ins IntRegs:$dst2, IntRegs:$src2,
                                        u8Imm:$src3),
               !strconcat("$dst += ", !strconcat(opc , "($src2, #$src3)")),
               [(set IntRegs:$dst, (IntID IntRegs:$dst2, IntRegs:$src2,
                                          imm:$src3))],
               "$dst2 = $dst">;

class si_MInst_sisiu8_nac<string opc, Intrinsic IntID>
  : MInst_acc<(outs IntRegs:$dst), (ins IntRegs:$dst2, IntRegs:$src2,
                                        u8Imm:$src3),
               !strconcat("$dst -= ", !strconcat(opc , "($src2, #$src3)")),
               [(set IntRegs:$dst, (IntID IntRegs:$dst2, IntRegs:$src2,
                                          imm:$src3))],
               "$dst2 = $dst">;

class si_MInst_sisisi_acc_hh<string opc, Intrinsic IntID>
  : MInst_acc<(outs IntRegs:$dst), (ins IntRegs:$dst2, IntRegs:$src1,
                                        IntRegs:$src2),
               !strconcat("$dst += ", !strconcat(opc , "($src1.H, $src2.H)")),
               [(set IntRegs:$dst, (IntID IntRegs:$dst2, IntRegs:$src1,
                                          IntRegs:$src2))],
               "$dst2 = $dst">;

class si_MInst_sisisi_acc_sat_lh<string opc, Intrinsic IntID>
  : MInst_acc<(outs IntRegs:$dst), (ins IntRegs:$dst2, IntRegs:$src1,
                                        IntRegs:$src2),
               !strconcat("$dst += ", !strconcat(opc ,
                                                 "($src1.L, $src2.H):sat")),
               [(set IntRegs:$dst, (IntID IntRegs:$dst2, IntRegs:$src1,
                                          IntRegs:$src2))],
               "$dst2 = $dst">;

class si_MInst_sisisi_acc_sat_lh_s1<string opc, Intrinsic IntID>
  : MInst_acc<(outs IntRegs:$dst), (ins IntRegs:$dst2, IntRegs:$src1,
                                        IntRegs:$src2),
               !strconcat("$dst += ", !strconcat(opc ,
                                                 "($src1.L, $src2.H):<<1:sat")),
               [(set IntRegs:$dst, (IntID IntRegs:$dst2, IntRegs:$src1,
                                          IntRegs:$src2))],
               "$dst2 = $dst">;

class si_MInst_sisisi_acc_sat_hh<string opc, Intrinsic IntID>
  : MInst_acc<(outs IntRegs:$dst), (ins IntRegs:$dst2, IntRegs:$src1,
                                        IntRegs:$src2),
               !strconcat("$dst += ", !strconcat(opc ,
                                                 "($src1.H, $src2.H):sat")),
               [(set IntRegs:$dst, (IntID IntRegs:$dst2, IntRegs:$src1,
                                          IntRegs:$src2))],
               "$dst2 = $dst">;

class si_MInst_sisisi_acc_sat_hh_s1<string opc, Intrinsic IntID>
  : MInst_acc<(outs IntRegs:$dst), (ins IntRegs:$dst2, IntRegs:$src1,
                                        IntRegs:$src2),
               !strconcat("$dst += ", !strconcat(opc ,
                                                 "($src1.H, $src2.H):<<1:sat")),
               [(set IntRegs:$dst, (IntID IntRegs:$dst2, IntRegs:$src1,
                                          IntRegs:$src2))],
               "$dst2 = $dst">;

class si_MInst_sisisi_acc_hh_s1<string opc, Intrinsic IntID>
  : MInst_acc<(outs IntRegs:$dst), (ins IntRegs:$dst2, IntRegs:$src1,
                                        IntRegs:$src2),
               !strconcat("$dst += ", !strconcat(opc ,
                                                 "($src1.H, $src2.H):<<1")),
               [(set IntRegs:$dst, (IntID IntRegs:$dst2, IntRegs:$src1,
                                          IntRegs:$src2))],
               "$dst2 = $dst">;

class si_MInst_sisisi_nac_hh<string opc, Intrinsic IntID>
  : MInst_acc<(outs IntRegs:$dst), (ins IntRegs:$dst2, IntRegs:$src1,
                                        IntRegs:$src2),
               !strconcat("$dst -= ", !strconcat(opc , "($src1.H, $src2.H)")),
               [(set IntRegs:$dst, (IntID IntRegs:$dst2, IntRegs:$src1,
                                          IntRegs:$src2))],
               "$dst2 = $dst">;

class si_MInst_sisisi_nac_sat_hh_s1<string opc, Intrinsic IntID>
  : MInst_acc<(outs IntRegs:$dst), (ins IntRegs:$dst2, IntRegs:$src1,
                                        IntRegs:$src2),
               !strconcat("$dst -= ", !strconcat(opc ,
                                                 "($src1.H, $src2.H):<<1:sat")),
               [(set IntRegs:$dst, (IntID IntRegs:$dst2, IntRegs:$src1,
                                          IntRegs:$src2))],
               "$dst2 = $dst">;

class si_MInst_sisisi_nac_sat_hh<string opc, Intrinsic IntID>
  : MInst_acc<(outs IntRegs:$dst), (ins IntRegs:$dst2, IntRegs:$src1,
                                        IntRegs:$src2),
               !strconcat("$dst -= ", !strconcat(opc ,
                                                 "($src1.H, $src2.H):sat")),
               [(set IntRegs:$dst, (IntID IntRegs:$dst2, IntRegs:$src1,
                                          IntRegs:$src2))],
               "$dst2 = $dst">;

class si_MInst_sisisi_nac_sat_hl_s1<string opc, Intrinsic IntID>
  : MInst_acc<(outs IntRegs:$dst), (ins IntRegs:$dst2, IntRegs:$src1,
                                        IntRegs:$src2),
               !strconcat("$dst -= ", !strconcat(opc ,
                                                 "($src1.H, $src2.L):<<1:sat")),
               [(set IntRegs:$dst, (IntID IntRegs:$dst2, IntRegs:$src1,
                                          IntRegs:$src2))],
               "$dst2 = $dst">;

class si_MInst_sisisi_nac_sat_hl<string opc, Intrinsic IntID>
  : MInst_acc<(outs IntRegs:$dst), (ins IntRegs:$dst2, IntRegs:$src1,
                                        IntRegs:$src2),
               !strconcat("$dst -= ", !strconcat(opc ,
                                                 "($src1.H, $src2.L):sat")),
               [(set IntRegs:$dst, (IntID IntRegs:$dst2, IntRegs:$src1,
                                          IntRegs:$src2))],
               "$dst2 = $dst">;

class si_MInst_sisisi_nac_sat_lh_s1<string opc, Intrinsic IntID>
  : MInst_acc<(outs IntRegs:$dst), (ins IntRegs:$dst2, IntRegs:$src1,
                                        IntRegs:$src2),
               !strconcat("$dst -= ", !strconcat(opc ,
                                                 "($src1.L, $src2.H):<<1:sat")),
               [(set IntRegs:$dst, (IntID IntRegs:$dst2, IntRegs:$src1,
                                          IntRegs:$src2))],
               "$dst2 = $dst">;

class si_MInst_sisisi_nac_sat_lh<string opc, Intrinsic IntID>
  : MInst_acc<(outs IntRegs:$dst), (ins IntRegs:$dst2, IntRegs:$src1,
                                        IntRegs:$src2),
               !strconcat("$dst -= ", !strconcat(opc ,
                                                 "($src1.L, $src2.H):sat")),
               [(set IntRegs:$dst, (IntID IntRegs:$dst2, IntRegs:$src1,
                                          IntRegs:$src2))],
               "$dst2 = $dst">;

class si_MInst_sisisi_nac_sat_ll_s1<string opc, Intrinsic IntID>
  : MInst_acc<(outs IntRegs:$dst), (ins IntRegs:$dst2, IntRegs:$src1,
                                        IntRegs:$src2),
               !strconcat("$dst -= ", !strconcat(opc ,
                                                 "($src1.L, $src2.L):<<1:sat")),
               [(set IntRegs:$dst, (IntID IntRegs:$dst2, IntRegs:$src1,
                                          IntRegs:$src2))],
               "$dst2 = $dst">;

class si_MInst_sisisi_nac_sat_ll<string opc, Intrinsic IntID>
  : MInst_acc<(outs IntRegs:$dst), (ins IntRegs:$dst2, IntRegs:$src1,
                                        IntRegs:$src2),
               !strconcat("$dst -= ", !strconcat(opc ,
                                                 "($src1.L, $src2.L):sat")),
               [(set IntRegs:$dst, (IntID IntRegs:$dst2, IntRegs:$src1,
                                          IntRegs:$src2))],
               "$dst2 = $dst">;

class si_MInst_sisisi_nac_hh_s1<string opc, Intrinsic IntID>
  : MInst_acc<(outs IntRegs:$dst), (ins IntRegs:$dst2, IntRegs:$src1,
                                        IntRegs:$src2),
               !strconcat("$dst -= ", !strconcat(opc ,
                                                 "($src1.H, $src2.H):<<1")),
               [(set IntRegs:$dst, (IntID IntRegs:$dst2, IntRegs:$src1,
                                          IntRegs:$src2))],
               "$dst2 = $dst">;

class si_MInst_sisisi_acc_hl<string opc, Intrinsic IntID>
  : MInst_acc<(outs IntRegs:$dst), (ins IntRegs:$dst2, IntRegs:$src1,
                                        IntRegs:$src2),
               !strconcat("$dst += ", !strconcat(opc , "($src1.H, $src2.L)")),
               [(set IntRegs:$dst, (IntID IntRegs:$dst2, IntRegs:$src1,
                                          IntRegs:$src2))],
               "$dst2 = $dst">;

class si_MInst_sisisi_acc_hl_s1<string opc, Intrinsic IntID>
  : MInst_acc<(outs IntRegs:$dst), (ins IntRegs:$dst2, IntRegs:$src1,
                                        IntRegs:$src2),
               !strconcat("$dst += ", !strconcat(opc ,
                                                 "($src1.H, $src2.L):<<1")),
               [(set IntRegs:$dst, (IntID IntRegs:$dst2, IntRegs:$src1,
                                          IntRegs:$src2))],
               "$dst2 = $dst">;

class si_MInst_sisisi_nac_hl<string opc, Intrinsic IntID>
  : MInst_acc<(outs IntRegs:$dst), (ins IntRegs:$dst2, IntRegs:$src1,
                                        IntRegs:$src2),
               !strconcat("$dst -= ", !strconcat(opc , "($src1.H, $src2.L)")),
               [(set IntRegs:$dst, (IntID IntRegs:$dst2, IntRegs:$src1,
                                          IntRegs:$src2))],
               "$dst2 = $dst">;

class si_MInst_sisisi_nac_hl_s1<string opc, Intrinsic IntID>
  : MInst_acc<(outs IntRegs:$dst), (ins IntRegs:$dst2, IntRegs:$src1,
                                        IntRegs:$src2),
               !strconcat("$dst -= ", !strconcat(opc ,
                                                 "($src1.H, $src2.L):<<1")),
               [(set IntRegs:$dst, (IntID IntRegs:$dst2, IntRegs:$src1,
                                          IntRegs:$src2))],
               "$dst2 = $dst">;

class si_MInst_sisisi_acc_lh<string opc, Intrinsic IntID>
  : MInst_acc<(outs IntRegs:$dst), (ins IntRegs:$dst2, IntRegs:$src1,
                                        IntRegs:$src2),
               !strconcat("$dst += ", !strconcat(opc , "($src1.L, $src2.H)")),
               [(set IntRegs:$dst, (IntID IntRegs:$dst2, IntRegs:$src1,
                                          IntRegs:$src2))],
               "$dst2 = $dst">;

class si_MInst_sisisi_acc_lh_s1<string opc, Intrinsic IntID>
  : MInst_acc<(outs IntRegs:$dst), (ins IntRegs:$dst2, IntRegs:$src1,
                                        IntRegs:$src2),
               !strconcat("$dst += ", !strconcat(opc ,
                                                 "($src1.L, $src2.H):<<1")),
               [(set IntRegs:$dst, (IntID IntRegs:$dst2, IntRegs:$src1,
                                          IntRegs:$src2))],
               "$dst2 = $dst">;

class si_MInst_sisisi_nac_lh<string opc, Intrinsic IntID>
  : MInst_acc<(outs IntRegs:$dst), (ins IntRegs:$dst2, IntRegs:$src1,
                                        IntRegs:$src2),
               !strconcat("$dst -= ", !strconcat(opc , "($src1.L, $src2.H)")),
               [(set IntRegs:$dst, (IntID IntRegs:$dst2, IntRegs:$src1,
                                          IntRegs:$src2))],
               "$dst2 = $dst">;

class si_MInst_sisisi_nac_lh_s1<string opc, Intrinsic IntID>
  : MInst_acc<(outs IntRegs:$dst), (ins IntRegs:$dst2, IntRegs:$src1,
                                        IntRegs:$src2),
               !strconcat("$dst -= ", !strconcat(opc ,
                                                 "($src1.L, $src2.H):<<1")),
               [(set IntRegs:$dst, (IntID IntRegs:$dst2, IntRegs:$src1,
                                          IntRegs:$src2))],
               "$dst2 = $dst">;

class si_MInst_sisisi_acc_ll<string opc, Intrinsic IntID>
  : MInst_acc<(outs IntRegs:$dst), (ins IntRegs:$dst2, IntRegs:$src1,
                                        IntRegs:$src2),
               !strconcat("$dst += ", !strconcat(opc , "($src1.L, $src2.L)")),
               [(set IntRegs:$dst, (IntID IntRegs:$dst2, IntRegs:$src1,
                                          IntRegs:$src2))],
               "$dst2 = $dst">;

class si_MInst_sisisi_acc_ll_s1<string opc, Intrinsic IntID>
  : MInst_acc<(outs IntRegs:$dst), (ins IntRegs:$dst2, IntRegs:$src1,
                                        IntRegs:$src2),
               !strconcat("$dst += ", !strconcat(opc ,
                                                 "($src1.L, $src2.L):<<1")),
               [(set IntRegs:$dst, (IntID IntRegs:$dst2, IntRegs:$src1,
                                          IntRegs:$src2))],
               "$dst2 = $dst">;

class si_MInst_sisisi_acc_sat_ll_s1<string opc, Intrinsic IntID>
  : MInst_acc<(outs IntRegs:$dst), (ins IntRegs:$dst2, IntRegs:$src1,
                                        IntRegs:$src2),
               !strconcat("$dst += ", !strconcat(opc ,
                                                 "($src1.L, $src2.L):<<1:sat")),
               [(set IntRegs:$dst, (IntID IntRegs:$dst2, IntRegs:$src1,
                                          IntRegs:$src2))],
               "$dst2 = $dst">;

class si_MInst_sisisi_acc_sat_hl_s1<string opc, Intrinsic IntID>
  : MInst_acc<(outs IntRegs:$dst), (ins IntRegs:$dst2, IntRegs:$src1,
                                        IntRegs:$src2),
               !strconcat("$dst += ", !strconcat(opc ,
                                                 "($src1.H, $src2.L):<<1:sat")),
               [(set IntRegs:$dst, (IntID IntRegs:$dst2, IntRegs:$src1,
                                          IntRegs:$src2))],
               "$dst2 = $dst">;

class si_MInst_sisisi_acc_sat_ll<string opc, Intrinsic IntID>
  : MInst_acc<(outs IntRegs:$dst), (ins IntRegs:$dst2, IntRegs:$src1,
                                        IntRegs:$src2),
               !strconcat("$dst += ", !strconcat(opc ,
                                                 "($src1.L, $src2.L):sat")),
               [(set IntRegs:$dst, (IntID IntRegs:$dst2, IntRegs:$src1,
                                          IntRegs:$src2))],
               "$dst2 = $dst">;

class si_MInst_sisisi_acc_sat_hl<string opc, Intrinsic IntID>
  : MInst_acc<(outs IntRegs:$dst), (ins IntRegs:$dst2, IntRegs:$src1,
                                        IntRegs:$src2),
               !strconcat("$dst += ", !strconcat(opc ,
                                                 "($src1.H, $src2.L):sat")),
               [(set IntRegs:$dst, (IntID IntRegs:$dst2, IntRegs:$src1,
                                          IntRegs:$src2))],
               "$dst2 = $dst">;

class si_MInst_sisisi_nac_ll<string opc, Intrinsic IntID>
  : MInst_acc<(outs IntRegs:$dst), (ins IntRegs:$dst2, IntRegs:$src1,
                                        IntRegs:$src2),
               !strconcat("$dst -= ", !strconcat(opc , "($src1.L, $src2.L)")),
               [(set IntRegs:$dst, (IntID IntRegs:$dst2, IntRegs:$src1,
                                          IntRegs:$src2))],
               "$dst2 = $dst">;

class si_MInst_sisisi_nac_ll_s1<string opc, Intrinsic IntID>
  : MInst_acc<(outs IntRegs:$dst), (ins IntRegs:$dst2, IntRegs:$src1,
                                        IntRegs:$src2),
               !strconcat("$dst -= ", !strconcat(opc ,
                                                 "($src1.L, $src2.L):<<1")),
               [(set IntRegs:$dst, (IntID IntRegs:$dst2, IntRegs:$src1,
                                          IntRegs:$src2))],
               "$dst2 = $dst">;

class si_MInst_sisisi_nac_hh_sat<string opc, Intrinsic IntID>
  : MInst_acc<(outs IntRegs:$dst), (ins IntRegs:$dst2, IntRegs:$src1,
                                        IntRegs:$src2),
               !strconcat("$dst -= ", !strconcat(opc ,
                                                 "($src1.H, $src2.H):sat")),
               [(set IntRegs:$dst, (IntID IntRegs:$dst2, IntRegs:$src1,
                                          IntRegs:$src2))],
               "$dst2 = $dst">;

class si_MInst_sisisi_nac_hh_s1_sat<string opc, Intrinsic IntID>
  : MInst_acc<(outs IntRegs:$dst), (ins IntRegs:$dst2, IntRegs:$src1,
                                        IntRegs:$src2),
               !strconcat("$dst -= ", !strconcat(opc ,
                                                 "($src1.H, $src2.H):<<1:sat")),
               [(set IntRegs:$dst, (IntID IntRegs:$dst2, IntRegs:$src1,
                                          IntRegs:$src2))],
               "$dst2 = $dst">;

class si_MInst_sisisi_nac_hl_sat<string opc, Intrinsic IntID>
  : MInst_acc<(outs IntRegs:$dst), (ins IntRegs:$dst2, IntRegs:$src1,
                                        IntRegs:$src2),
               !strconcat("$dst -= ", !strconcat(opc ,
                                                 "($src1.H, $src2.L):sat")),
               [(set IntRegs:$dst, (IntID IntRegs:$dst2, IntRegs:$src1,
                                          IntRegs:$src2))],
               "$dst2 = $dst">;

class si_MInst_sisisi_nac_hl_s1_sat<string opc, Intrinsic IntID>
  : MInst_acc<(outs IntRegs:$dst), (ins IntRegs:$dst2, IntRegs:$src1,
                                        IntRegs:$src2),
               !strconcat("$dst -= ", !strconcat(opc ,
                                                 "($src1.H, $src2.L):<<1:sat")),
               [(set IntRegs:$dst, (IntID IntRegs:$dst2, IntRegs:$src1,
                                          IntRegs:$src2))],
               "$dst2 = $dst">;

class si_MInst_sisisi_nac_lh_sat<string opc, Intrinsic IntID>
  : MInst_acc<(outs IntRegs:$dst), (ins IntRegs:$dst2, IntRegs:$src1,
                                        IntRegs:$src2),
               !strconcat("$dst -= ", !strconcat(opc ,
                                                 "($src1.L, $src2.H):sat")),
               [(set IntRegs:$dst, (IntID IntRegs:$dst2, IntRegs:$src1,
                                          IntRegs:$src2))],
               "$dst2 = $dst">;

class si_MInst_sisisi_nac_lh_s1_sat<string opc, Intrinsic IntID>
  : MInst_acc<(outs IntRegs:$dst), (ins IntRegs:$dst2, IntRegs:$src1,
                                        IntRegs:$src2),
               !strconcat("$dst -= ", !strconcat(opc ,
                                                 "($src1.L, $src2.H):<<1:sat")),
               [(set IntRegs:$dst, (IntID IntRegs:$dst2, IntRegs:$src1,
                                          IntRegs:$src2))],
               "$dst2 = $dst">;

class si_MInst_sisisi_nac_ll_sat<string opc, Intrinsic IntID>
  : MInst_acc<(outs IntRegs:$dst), (ins IntRegs:$dst2, IntRegs:$src1,
                                        IntRegs:$src2),
               !strconcat("$dst -= ", !strconcat(opc ,
                                                 "($src1.L, $src2.L):sat")),
               [(set IntRegs:$dst, (IntID IntRegs:$dst2, IntRegs:$src1,
                                          IntRegs:$src2))],
               "$dst2 = $dst">;

class si_MInst_sisisi_nac_ll_s1_sat<string opc, Intrinsic IntID>
  : MInst_acc<(outs IntRegs:$dst), (ins IntRegs:$dst2, IntRegs:$src1,
                                        IntRegs:$src2),
               !strconcat("$dst -= ", !strconcat(opc ,
                                                 "($src1.L, $src2.L):<<1:sat")),
               [(set IntRegs:$dst, (IntID IntRegs:$dst2, IntRegs:$src1,
                                          IntRegs:$src2))],
               "$dst2 = $dst">;

class di_ALU32_sisi<string opc, Intrinsic IntID>
  : ALU32_rr<(outs DoubleRegs:$dst), (ins IntRegs:$src1, IntRegs:$src2),
             !strconcat("$dst = ", !strconcat(opc , "($src1, $src2)")),
             [(set DoubleRegs:$dst, (IntID IntRegs:$src1, IntRegs:$src2))]>;

class di_MInst_sisi<string opc, Intrinsic IntID>
  : MInst<(outs DoubleRegs:$dst), (ins IntRegs:$src1, IntRegs:$src2),
             !strconcat("$dst = ", !strconcat(opc , "($src1, $src2)")),
             [(set DoubleRegs:$dst, (IntID IntRegs:$src1, IntRegs:$src2))]>;

class di_MInst_sisi_sat<string opc, Intrinsic IntID>
  : MInst<(outs DoubleRegs:$dst), (ins IntRegs:$src1, IntRegs:$src2),
             !strconcat("$dst = ", !strconcat(opc , "($src1, $src2):sat")),
             [(set DoubleRegs:$dst, (IntID IntRegs:$src1, IntRegs:$src2))]>;

class di_MInst_sisi_sat_conj<string opc, Intrinsic IntID>
  : MInst<(outs DoubleRegs:$dst), (ins IntRegs:$src1, IntRegs:$src2),
             !strconcat("$dst = ", !strconcat(opc , "($src1, $src2*):sat")),
             [(set DoubleRegs:$dst, (IntID IntRegs:$src1, IntRegs:$src2))]>;

class di_MInst_sisi_s1_sat<string opc, Intrinsic IntID>
  : MInst<(outs DoubleRegs:$dst), (ins IntRegs:$src1, IntRegs:$src2),
             !strconcat("$dst = ", !strconcat(opc , "($src1, $src2):<<1:sat")),
             [(set DoubleRegs:$dst, (IntID IntRegs:$src1, IntRegs:$src2))]>;

class di_MInst_didi_s1_sat<string opc, Intrinsic IntID>
  : MInst<(outs DoubleRegs:$dst), (ins DoubleRegs:$src1, DoubleRegs:$src2),
             !strconcat("$dst = ", !strconcat(opc , "($src1, $src2):<<1:sat")),
             [(set DoubleRegs:$dst, (IntID DoubleRegs:$src1,
                                           DoubleRegs:$src2))]>;

class si_MInst_didi_s1_rnd_sat<string opc, Intrinsic IntID>
  : MInst<(outs IntRegs:$dst), (ins DoubleRegs:$src1, DoubleRegs:$src2),
             !strconcat("$dst = ", !strconcat(opc ,
                                              "($src1, $src2):<<1:rnd:sat")),
             [(set IntRegs:$dst, (IntID DoubleRegs:$src1, DoubleRegs:$src2))]>;

class si_MInst_didi_rnd_sat<string opc, Intrinsic IntID>
  : MInst<(outs IntRegs:$dst), (ins DoubleRegs:$src1, DoubleRegs:$src2),
             !strconcat("$dst = ", !strconcat(opc , "($src1, $src2):rnd:sat")),
             [(set IntRegs:$dst, (IntID DoubleRegs:$src1, DoubleRegs:$src2))]>;

class si_MInst_sisi_sat_hh<string opc, Intrinsic IntID>
  : MInst<(outs IntRegs:$dst), (ins IntRegs:$src1, IntRegs:$src2),
             !strconcat("$dst = ", !strconcat(opc , "($src1.H, $src2.H):sat")),
             [(set IntRegs:$dst, (IntID IntRegs:$src1, IntRegs:$src2))]>;

class si_MInst_sisi_sat_hh_s1<string opc, Intrinsic IntID>
  : MInst<(outs IntRegs:$dst), (ins IntRegs:$src1, IntRegs:$src2),
               !strconcat("$dst = ", !strconcat(opc ,
                                                "($src1.H, $src2.H):<<1:sat")),
               [(set IntRegs:$dst, (IntID IntRegs:$src1, IntRegs:$src2))]>;

class si_MInst_sisi_sat_hl<string opc, Intrinsic IntID>
  : MInst<(outs IntRegs:$dst), (ins IntRegs:$src1, IntRegs:$src2),
             !strconcat("$dst = ", !strconcat(opc , "($src1.H, $src2.L):sat")),
             [(set IntRegs:$dst, (IntID IntRegs:$src1, IntRegs:$src2))]>;

class si_MInst_sisi_sat_hl_s1<string opc, Intrinsic IntID>
  : MInst<(outs IntRegs:$dst), (ins IntRegs:$src1, IntRegs:$src2),
               !strconcat("$dst = ", !strconcat(opc ,
                                                "($src1.H, $src2.L):<<1:sat")),
               [(set IntRegs:$dst, (IntID IntRegs:$src1, IntRegs:$src2))]>;

class si_MInst_sisi_sat_lh<string opc, Intrinsic IntID>
  : MInst<(outs IntRegs:$dst), (ins IntRegs:$src1, IntRegs:$src2),
             !strconcat("$dst = ", !strconcat(opc , "($src1.L, $src2.H):sat")),
             [(set IntRegs:$dst, (IntID IntRegs:$src1, IntRegs:$src2))]>;

class si_MInst_sisi_sat_lh_s1<string opc, Intrinsic IntID>
  : MInst<(outs IntRegs:$dst), (ins IntRegs:$src1, IntRegs:$src2),
               !strconcat("$dst = ", !strconcat(opc ,
                                                "($src1.L, $src2.H):<<1:sat")),
               [(set IntRegs:$dst, (IntID IntRegs:$src1, IntRegs:$src2))]>;

class si_MInst_sisi_sat_ll<string opc, Intrinsic IntID>
  : MInst<(outs IntRegs:$dst), (ins IntRegs:$src1, IntRegs:$src2),
             !strconcat("$dst = ", !strconcat(opc , "($src1.L, $src2.L):sat")),
             [(set IntRegs:$dst, (IntID IntRegs:$src1, IntRegs:$src2))]>;

class si_MInst_sisi_sat_ll_s1<string opc, Intrinsic IntID>
  : MInst<(outs IntRegs:$dst), (ins IntRegs:$src1, IntRegs:$src2),
               !strconcat("$dst = ", !strconcat(opc ,
                                                "($src1.L, $src2.L):<<1:sat")),
               [(set IntRegs:$dst, (IntID IntRegs:$src1, IntRegs:$src2))]>;

class si_MInst_sisi_sat_rnd_hh<string opc, Intrinsic IntID>
  : MInst<(outs IntRegs:$dst), (ins IntRegs:$src1, IntRegs:$src2),
               !strconcat("$dst = ", !strconcat(opc ,
                                                "($src1.H, $src2.H):rnd:sat")),
               [(set IntRegs:$dst, (IntID IntRegs:$src1, IntRegs:$src2))]>;

class si_MInst_sisi_rnd_hh<string opc, Intrinsic IntID>
  : MInst<(outs IntRegs:$dst), (ins IntRegs:$src1, IntRegs:$src2),
               !strconcat("$dst = ", !strconcat(opc ,
                                                "($src1.H, $src2.H):rnd")),
               [(set IntRegs:$dst, (IntID IntRegs:$src1, IntRegs:$src2))]>;

class si_MInst_sisi_rnd_hh_s1<string opc, Intrinsic IntID>
  : MInst<(outs IntRegs:$dst), (ins IntRegs:$src1, IntRegs:$src2),
               !strconcat("$dst = ", !strconcat(opc ,
                                                "($src1.H, $src2.H):<<1:rnd")),
               [(set IntRegs:$dst, (IntID IntRegs:$src1, IntRegs:$src2))]>;

class si_MInst_sisi_sat_rnd_hh_s1<string opc, Intrinsic IntID>
  : MInst<(outs IntRegs:$dst), (ins IntRegs:$src1, IntRegs:$src2),
               !strconcat("$dst = ",
                          !strconcat(opc ,
                                     "($src1.H, $src2.H):<<1:rnd:sat")),
               [(set IntRegs:$dst, (IntID IntRegs:$src1, IntRegs:$src2))]>;

class si_MInst_sisi_rnd_hl<string opc, Intrinsic IntID>
  : MInst<(outs IntRegs:$dst), (ins IntRegs:$src1, IntRegs:$src2),
               !strconcat("$dst = ",
                          !strconcat(opc , "($src1.H, $src2.L):rnd")),
               [(set IntRegs:$dst, (IntID IntRegs:$src1, IntRegs:$src2))]>;

class si_MInst_sisi_rnd_hl_s1<string opc, Intrinsic IntID>
  : MInst<(outs IntRegs:$dst), (ins IntRegs:$src1, IntRegs:$src2),
               !strconcat("$dst = ",
                          !strconcat(opc , "($src1.H, $src2.L):<<1:rnd")),
               [(set IntRegs:$dst, (IntID IntRegs:$src1, IntRegs:$src2))]>;

class si_MInst_sisi_sat_rnd_hl<string opc, Intrinsic IntID>
  : MInst<(outs IntRegs:$dst), (ins IntRegs:$src1, IntRegs:$src2),
               !strconcat("$dst = ",
                          !strconcat(opc , "($src1.H, $src2.L):rnd:sat")),
               [(set IntRegs:$dst, (IntID IntRegs:$src1, IntRegs:$src2))]>;

class si_MInst_sisi_sat_rnd_hl_s1<string opc, Intrinsic IntID>
  : MInst<(outs IntRegs:$dst), (ins IntRegs:$src1, IntRegs:$src2),
               !strconcat("$dst = ",
                          !strconcat(opc , "($src1.H, $src2.L):<<1:rnd:sat")),
               [(set IntRegs:$dst, (IntID IntRegs:$src1, IntRegs:$src2))]>;

class si_MInst_sisi_rnd_lh<string opc, Intrinsic IntID>
  : MInst<(outs IntRegs:$dst), (ins IntRegs:$src1, IntRegs:$src2),
               !strconcat("$dst = ",
                          !strconcat(opc , "($src1.L, $src2.H):rnd")),
               [(set IntRegs:$dst, (IntID IntRegs:$src1, IntRegs:$src2))]>;

class si_MInst_sisi_sat_rnd_lh<string opc, Intrinsic IntID>
  : MInst<(outs IntRegs:$dst), (ins IntRegs:$src1, IntRegs:$src2),
               !strconcat("$dst = ",
                          !strconcat(opc , "($src1.L, $src2.H):rnd:sat")),
               [(set IntRegs:$dst, (IntID IntRegs:$src1, IntRegs:$src2))]>;

class si_MInst_sisi_sat_rnd_lh_s1<string opc, Intrinsic IntID>
  : MInst<(outs IntRegs:$dst), (ins IntRegs:$src1, IntRegs:$src2),
               !strconcat("$dst = ",
                          !strconcat(opc , "($src1.L, $src2.H):<<1:rnd:sat")),
               [(set IntRegs:$dst, (IntID IntRegs:$src1, IntRegs:$src2))]>;

class si_MInst_sisi_rnd_lh_s1<string opc, Intrinsic IntID>
  : MInst<(outs IntRegs:$dst), (ins IntRegs:$src1, IntRegs:$src2),
               !strconcat("$dst = ",
                          !strconcat(opc , "($src1.L, $src2.H):<<1:rnd")),
               [(set IntRegs:$dst, (IntID IntRegs:$src1, IntRegs:$src2))]>;

class si_MInst_sisi_sat_rnd_ll<string opc, Intrinsic IntID>
  : MInst<(outs IntRegs:$dst), (ins IntRegs:$src1, IntRegs:$src2),
               !strconcat("$dst = ",
                          !strconcat(opc , "($src1.L, $src2.L):rnd:sat")),
               [(set IntRegs:$dst, (IntID IntRegs:$src1, IntRegs:$src2))]>;

class si_MInst_sisi_sat_rnd_ll_s1<string opc, Intrinsic IntID>
  : MInst<(outs IntRegs:$dst), (ins IntRegs:$src1, IntRegs:$src2),
               !strconcat("$dst = ",
                          !strconcat(opc , "($src1.L, $src2.L):<<1:rnd:sat")),
               [(set IntRegs:$dst, (IntID IntRegs:$src1, IntRegs:$src2))]>;

class si_MInst_sisi_rnd_ll<string opc, Intrinsic IntID>
  : MInst<(outs IntRegs:$dst), (ins IntRegs:$src1, IntRegs:$src2),
               !strconcat("$dst = ",
                          !strconcat(opc , "($src1.L, $src2.L):rnd")),
               [(set IntRegs:$dst, (IntID IntRegs:$src1, IntRegs:$src2))]>;

class si_MInst_sisi_rnd_ll_s1<string opc, Intrinsic IntID>
  : MInst<(outs IntRegs:$dst), (ins IntRegs:$src1, IntRegs:$src2),
               !strconcat("$dst = ",
                          !strconcat(opc , "($src1.L, $src2.L):<<1:rnd")),
               [(set IntRegs:$dst, (IntID IntRegs:$src1, IntRegs:$src2))]>;

class di_MInst_dididi_acc_sat<string opc, Intrinsic IntID>
  : MInst_acc<(outs DoubleRegs:$dst), (ins DoubleRegs:$dst2,
                                           DoubleRegs:$src1, DoubleRegs:$src2),
               !strconcat("$dst += ", !strconcat(opc , "($src1, $src2):sat")),
               [(set DoubleRegs:$dst, (IntID DoubleRegs:$dst2,
                                             DoubleRegs:$src1,
                                             DoubleRegs:$src2))],
               "$dst2 = $dst">;

class di_MInst_dididi_acc_rnd_sat<string opc, Intrinsic IntID>
  : MInst_acc<(outs DoubleRegs:$dst), (ins DoubleRegs:$dst2, DoubleRegs:$src1,
                                           DoubleRegs:$src2),
               !strconcat("$dst += ",
                          !strconcat(opc , "($src1, $src2):rnd:sat")),
               [(set DoubleRegs:$dst, (IntID DoubleRegs:$dst2,
                                             DoubleRegs:$src1,
                                             DoubleRegs:$src2))],
               "$dst2 = $dst">;

class di_MInst_dididi_acc_s1<string opc, Intrinsic IntID>
  : MInst_acc<(outs DoubleRegs:$dst), (ins DoubleRegs:$dst2,
                                           DoubleRegs:$src1,
                                           DoubleRegs:$src2),
               !strconcat("$dst += ",
                          !strconcat(opc , "($src1, $src2):<<1")),
               [(set DoubleRegs:$dst, (IntID DoubleRegs:$dst2,
                                             DoubleRegs:$src1,
                                             DoubleRegs:$src2))],
               "$dst2 = $dst">;


class di_MInst_dididi_acc_s1_sat<string opc, Intrinsic IntID>
  : MInst_acc<(outs DoubleRegs:$dst), (ins DoubleRegs:$dst2,
                                           DoubleRegs:$src1,
                                           DoubleRegs:$src2),
               !strconcat("$dst += ",
                          !strconcat(opc , "($src1, $src2):<<1:sat")),
               [(set DoubleRegs:$dst, (IntID DoubleRegs:$dst2,
                                             DoubleRegs:$src1,
                                             DoubleRegs:$src2))],
               "$dst2 = $dst">;

class di_MInst_dididi_acc_s1_rnd_sat<string opc, Intrinsic IntID>
  : MInst_acc<(outs DoubleRegs:$dst), (ins DoubleRegs:$dst2, DoubleRegs:$src1,
                                           DoubleRegs:$src2),
               !strconcat("$dst += ",
                          !strconcat(opc , "($src1, $src2):<<1:rnd:sat")),
               [(set DoubleRegs:$dst, (IntID DoubleRegs:$dst2,
                                             DoubleRegs:$src1,
                                             DoubleRegs:$src2))],
               "$dst2 = $dst">;

class di_MInst_dididi_acc<string opc, Intrinsic IntID>
  : MInst_acc<(outs DoubleRegs:$dst), (ins DoubleRegs:$dst2, DoubleRegs:$src1,
                                           DoubleRegs:$src2),
               !strconcat("$dst += ", !strconcat(opc , "($src1, $src2)")),
               [(set DoubleRegs:$dst, (IntID DoubleRegs:$dst2,
                                             DoubleRegs:$src1,
                                             DoubleRegs:$src2))],
               "$dst2 = $dst">;

class di_MInst_dididi_acc_conj<string opc, Intrinsic IntID>
  : MInst_acc<(outs DoubleRegs:$dst), (ins DoubleRegs:$dst2, DoubleRegs:$src1,
                                           DoubleRegs:$src2),
               !strconcat("$dst += ", !strconcat(opc , "($src1, $src2*)")),
               [(set DoubleRegs:$dst, (IntID DoubleRegs:$dst2,
                                             DoubleRegs:$src1,
                                             DoubleRegs:$src2))],
               "$dst2 = $dst">;

class di_MInst_disisi_acc_hh<string opc, Intrinsic IntID>
  : MInst_acc<(outs DoubleRegs:$dst), (ins DoubleRegs:$dst2, IntRegs:$src1,
                                           IntRegs:$src2),
               !strconcat("$dst += ", !strconcat(opc , "($src1.H, $src2.H)")),
               [(set DoubleRegs:$dst, (IntID DoubleRegs:$dst2, IntRegs:$src1,
                                             IntRegs:$src2))],
               "$dst2 = $dst">;

class di_MInst_disisi_acc_hl<string opc, Intrinsic IntID>
  : MInst_acc<(outs DoubleRegs:$dst), (ins DoubleRegs:$dst2, IntRegs:$src1,
                                           IntRegs:$src2),
               !strconcat("$dst += ", !strconcat(opc , "($src1.H, $src2.L)")),
               [(set DoubleRegs:$dst, (IntID DoubleRegs:$dst2, IntRegs:$src1,
                                             IntRegs:$src2))],
               "$dst2 = $dst">;

class di_MInst_disisi_acc_lh<string opc, Intrinsic IntID>
  : MInst_acc<(outs DoubleRegs:$dst), (ins DoubleRegs:$dst2, IntRegs:$src1,
                                           IntRegs:$src2),
               !strconcat("$dst += ", !strconcat(opc , "($src1.L, $src2.H)")),
               [(set DoubleRegs:$dst, (IntID DoubleRegs:$dst2, IntRegs:$src1,
                                             IntRegs:$src2))],
               "$dst2 = $dst">;

class di_MInst_disisi_acc_ll<string opc, Intrinsic IntID>
  : MInst_acc<(outs DoubleRegs:$dst), (ins DoubleRegs:$dst2, IntRegs:$src1,
                                           IntRegs:$src2),
               !strconcat("$dst += ", !strconcat(opc , "($src1.L, $src2.L)")),
               [(set DoubleRegs:$dst, (IntID DoubleRegs:$dst2, IntRegs:$src1,
                                             IntRegs:$src2))],
               "$dst2 = $dst">;

class di_MInst_disisi_acc_hh_s1<string opc, Intrinsic IntID>
  : MInst_acc<(outs DoubleRegs:$dst), (ins DoubleRegs:$dst2, IntRegs:$src1,
                                           IntRegs:$src2),
               !strconcat("$dst += ",
                          !strconcat(opc , "($src1.H, $src2.H):<<1")),
               [(set DoubleRegs:$dst, (IntID DoubleRegs:$dst2, IntRegs:$src1,
                                             IntRegs:$src2))],
               "$dst2 = $dst">;

class di_MInst_disisi_acc_hl_s1<string opc, Intrinsic IntID>
  : MInst_acc<(outs DoubleRegs:$dst), (ins DoubleRegs:$dst2, IntRegs:$src1,
                                           IntRegs:$src2),
               !strconcat("$dst += ",
                          !strconcat(opc , "($src1.H, $src2.L):<<1")),
               [(set DoubleRegs:$dst, (IntID DoubleRegs:$dst2, IntRegs:$src1,
                                             IntRegs:$src2))],
               "$dst2 = $dst">;

class di_MInst_disisi_acc_lh_s1<string opc, Intrinsic IntID>
  : MInst_acc<(outs DoubleRegs:$dst), (ins DoubleRegs:$dst2, IntRegs:$src1,
                                           IntRegs:$src2),
               !strconcat("$dst += ",
                          !strconcat(opc , "($src1.L, $src2.H):<<1")),
               [(set DoubleRegs:$dst, (IntID DoubleRegs:$dst2, IntRegs:$src1,
                                             IntRegs:$src2))],
               "$dst2 = $dst">;

class di_MInst_disisi_acc_ll_s1<string opc, Intrinsic IntID>
  : MInst_acc<(outs DoubleRegs:$dst), (ins DoubleRegs:$dst2, IntRegs:$src1,
                                           IntRegs:$src2),
               !strconcat("$dst += ",
                          !strconcat(opc , "($src1.L, $src2.L):<<1")),
               [(set DoubleRegs:$dst, (IntID DoubleRegs:$dst2, IntRegs:$src1,
                                             IntRegs:$src2))],
               "$dst2 = $dst">;

class di_MInst_disisi_nac_hh<string opc, Intrinsic IntID>
  : MInst_acc<(outs DoubleRegs:$dst), (ins DoubleRegs:$dst2, IntRegs:$src1,
                                           IntRegs:$src2),
               !strconcat("$dst -= ", !strconcat(opc , "($src1.H, $src2.H)")),
               [(set DoubleRegs:$dst, (IntID DoubleRegs:$dst2, IntRegs:$src1,
                                             IntRegs:$src2))],
               "$dst2 = $dst">;

class di_MInst_disisi_nac_hl<string opc, Intrinsic IntID>
  : MInst_acc<(outs DoubleRegs:$dst), (ins DoubleRegs:$dst2, IntRegs:$src1,
                                           IntRegs:$src2),
               !strconcat("$dst -= ", !strconcat(opc , "($src1.H, $src2.L)")),
               [(set DoubleRegs:$dst, (IntID DoubleRegs:$dst2, IntRegs:$src1,
                                             IntRegs:$src2))],
               "$dst2 = $dst">;

class di_MInst_disisi_nac_lh<string opc, Intrinsic IntID>
  : MInst_acc<(outs DoubleRegs:$dst), (ins DoubleRegs:$dst2, IntRegs:$src1,
                                           IntRegs:$src2),
               !strconcat("$dst -= ", !strconcat(opc , "($src1.L, $src2.H)")),
               [(set DoubleRegs:$dst, (IntID DoubleRegs:$dst2, IntRegs:$src1,
                                             IntRegs:$src2))],
               "$dst2 = $dst">;

class di_MInst_disisi_nac_ll<string opc, Intrinsic IntID>
  : MInst_acc<(outs DoubleRegs:$dst), (ins DoubleRegs:$dst2, IntRegs:$src1,
                                           IntRegs:$src2),
               !strconcat("$dst -= ", !strconcat(opc , "($src1.L, $src2.L)")),
               [(set DoubleRegs:$dst, (IntID DoubleRegs:$dst2, IntRegs:$src1,
                                             IntRegs:$src2))],
               "$dst2 = $dst">;

class di_MInst_disisi_nac_hh_s1<string opc, Intrinsic IntID>
  : MInst_acc<(outs DoubleRegs:$dst), (ins DoubleRegs:$dst2, IntRegs:$src1,
                                           IntRegs:$src2),
               !strconcat("$dst -= ",
                          !strconcat(opc , "($src1.H, $src2.H):<<1")),
               [(set DoubleRegs:$dst, (IntID DoubleRegs:$dst2, IntRegs:$src1,
                                             IntRegs:$src2))],
               "$dst2 = $dst">;

class di_MInst_disisi_nac_hl_s1<string opc, Intrinsic IntID>
  : MInst_acc<(outs DoubleRegs:$dst), (ins DoubleRegs:$dst2, IntRegs:$src1,
                                           IntRegs:$src2),
               !strconcat("$dst -= ",
                          !strconcat(opc , "($src1.H, $src2.L):<<1")),
               [(set DoubleRegs:$dst, (IntID DoubleRegs:$dst2, IntRegs:$src1,
                                             IntRegs:$src2))],
               "$dst2 = $dst">;

class di_MInst_disisi_nac_lh_s1<string opc, Intrinsic IntID>
  : MInst_acc<(outs DoubleRegs:$dst), (ins DoubleRegs:$dst2, IntRegs:$src1,
                                           IntRegs:$src2),
               !strconcat("$dst -= ",
                          !strconcat(opc , "($src1.L, $src2.H):<<1")),
               [(set DoubleRegs:$dst, (IntID DoubleRegs:$dst2, IntRegs:$src1,
                                             IntRegs:$src2))],
               "$dst2 = $dst">;

class di_MInst_disisi_nac_ll_s1<string opc, Intrinsic IntID>
  : MInst_acc<(outs DoubleRegs:$dst), (ins DoubleRegs:$dst2, IntRegs:$src1,
                                           IntRegs:$src2),
               !strconcat("$dst -= ",
                          !strconcat(opc , "($src1.L, $src2.L):<<1")),
               [(set DoubleRegs:$dst, (IntID DoubleRegs:$dst2, IntRegs:$src1,
                                             IntRegs:$src2))],
               "$dst2 = $dst">;

class di_MInst_disisi_acc_s1_sat<string opc, Intrinsic IntID>
  : MInst_acc<(outs DoubleRegs:$dst), (ins DoubleRegs:$dst2, IntRegs:$src1,
                                           IntRegs:$src2),
               !strconcat("$dst += ",
                          !strconcat(opc , "($src1, $src2):<<1:sat")),
               [(set DoubleRegs:$dst, (IntID DoubleRegs:$dst2, IntRegs:$src1,
                                             IntRegs:$src2))],
               "$dst2 = $dst">;

class di_MInst_disi_s1_sat<string opc, Intrinsic IntID>
  : MInst<(outs DoubleRegs:$dst), (ins DoubleRegs:$src1, IntRegs:$src2),
             !strconcat("$dst = ", !strconcat(opc , "($src1, $src2):<<1:sat")),
             [(set DoubleRegs:$dst, (IntID DoubleRegs:$src1, IntRegs:$src2))]>;

class di_MInst_didisi_acc_s1_sat<string opc, Intrinsic IntID>
  : MInst_acc<(outs DoubleRegs:$dst), (ins DoubleRegs:$dst2, DoubleRegs:$src1,
                                           IntRegs:$src2),
               !strconcat("$dst += ",
                          !strconcat(opc , "($src1, $src2):<<1:sat")),
               [(set DoubleRegs:$dst, (IntID DoubleRegs:$dst2,
                                             DoubleRegs:$src1,
                                             IntRegs:$src2))],
               "$dst2 = $dst">;

class si_MInst_disi_s1_rnd_sat<string opc, Intrinsic IntID>
  : MInst<(outs IntRegs:$dst), (ins DoubleRegs:$src1, IntRegs:$src2),
             !strconcat("$dst = ",
                        !strconcat(opc , "($src1, $src2):<<1:rnd:sat")),
             [(set IntRegs:$dst, (IntID DoubleRegs:$src1, IntRegs:$src2))]>;

class si_MInst_didi<string opc, Intrinsic IntID>
  : MInst<(outs IntRegs:$dst), (ins DoubleRegs:$src1, DoubleRegs:$src2),
             !strconcat("$dst = ", !strconcat(opc , "($src1, $src2)")),
             [(set IntRegs:$dst, (IntID DoubleRegs:$src1, DoubleRegs:$src2))]>;

//
// LDInst classes.
//
let mayLoad = 1, neverHasSideEffects = 1 in
class di_LDInstPI_diu4<string opc, Intrinsic IntID>
  : LDInstPI<(outs IntRegs:$dst, DoubleRegs:$dst2),
           (ins IntRegs:$src1, IntRegs:$src2, CRRegs:$src3, s4Imm:$offset),
           "$dst2 = memd($src1++#$offset:circ($src3))",
           [],
           "$src1 = $dst">;

/********************************************************************
*            ALU32/ALU                                              *
*********************************************************************/

// ALU32 / ALU / Add.
def HEXAGON_A2_add:
  si_ALU32_sisi                   <"add",      int_hexagon_A2_add>;
def HEXAGON_A2_addi:
  si_ALU32_sis16                  <"add",      int_hexagon_A2_addi>;

// ALU32 / ALU / Logical operations.
def HEXAGON_A2_and:
  si_ALU32_sisi                   <"and",      int_hexagon_A2_and>;
def HEXAGON_A2_andir:
  si_ALU32_sis10                  <"and",      int_hexagon_A2_andir>;
def HEXAGON_A2_not:
  si_ALU32_si                     <"not",      int_hexagon_A2_not>;
def HEXAGON_A2_or:
  si_ALU32_sisi                   <"or",       int_hexagon_A2_or>;
def HEXAGON_A2_orir:
  si_ALU32_sis10                  <"or",       int_hexagon_A2_orir>;
def HEXAGON_A2_xor:
  si_ALU32_sisi                   <"xor",      int_hexagon_A2_xor>;

// ALU32 / ALU / Negate.
def HEXAGON_A2_neg:
  si_ALU32_si                     <"neg",      int_hexagon_A2_neg>;

// ALU32 / ALU / Subtract.
def HEXAGON_A2_sub:
  si_ALU32_sisi                   <"sub",      int_hexagon_A2_sub>;
def HEXAGON_A2_subri:
  si_ALU32_s10si                  <"sub",      int_hexagon_A2_subri>;

// ALU32 / ALU / Transfer Immediate.
def HEXAGON_A2_tfril:
  si_lo_ALU32_siu16               <"",         int_hexagon_A2_tfril>;
def HEXAGON_A2_tfrih:
  si_hi_ALU32_siu16               <"",         int_hexagon_A2_tfrih>;
def HEXAGON_A2_tfrsi:
  si_ALU32_s16                    <"",         int_hexagon_A2_tfrsi>;
def HEXAGON_A2_tfrpi:
  di_ALU32_s8                     <"",         int_hexagon_A2_tfrpi>;

// ALU32 / ALU / Transfer Register.
def HEXAGON_A2_tfr:
  si_ALU32_si_tfr                  <"",        int_hexagon_A2_tfr>;

/********************************************************************
*            ALU32/PERM                                             *
*********************************************************************/

// ALU32 / PERM / Combine.
def HEXAGON_A2_combinew:
  di_ALU32_sisi                   <"combine",  int_hexagon_A2_combinew>;
def HEXAGON_A2_combine_hh:
  si_MInst_sisi_hh                <"combine",  int_hexagon_A2_combine_hh>;
def HEXAGON_A2_combine_lh:
  si_MInst_sisi_lh                <"combine",  int_hexagon_A2_combine_lh>;
def HEXAGON_A2_combine_hl:
  si_MInst_sisi_hl                <"combine",  int_hexagon_A2_combine_hl>;
def HEXAGON_A2_combine_ll:
  si_MInst_sisi_ll                <"combine",  int_hexagon_A2_combine_ll>;
def HEXAGON_A2_combineii:
  di_MInst_s8s8                   <"combine",  int_hexagon_A2_combineii>;

// ALU32 / PERM / Mux.
def HEXAGON_C2_mux:
  si_ALU32_qisisi                 <"mux",      int_hexagon_C2_mux>;
def HEXAGON_C2_muxri:
  si_ALU32_qis8si                 <"mux",      int_hexagon_C2_muxri>;
def HEXAGON_C2_muxir:
  si_ALU32_qisis8                 <"mux",      int_hexagon_C2_muxir>;
def HEXAGON_C2_muxii:
  si_ALU32_qis8s8                 <"mux",      int_hexagon_C2_muxii>;

// ALU32 / PERM / Shift halfword.
def HEXAGON_A2_aslh:
  si_ALU32_si                     <"aslh",     int_hexagon_A2_aslh>;
def HEXAGON_A2_asrh:
  si_ALU32_si                     <"asrh",     int_hexagon_A2_asrh>;
def SI_to_SXTHI_asrh:
  si_ALU32_si                     <"asrh",     int_hexagon_SI_to_SXTHI_asrh>;

// ALU32 / PERM / Sign/zero extend.
def HEXAGON_A2_sxth:
  si_ALU32_si                     <"sxth",     int_hexagon_A2_sxth>;
def HEXAGON_A2_sxtb:
  si_ALU32_si                     <"sxtb",     int_hexagon_A2_sxtb>;
def HEXAGON_A2_zxth:
  si_ALU32_si                     <"zxth",     int_hexagon_A2_zxth>;
def HEXAGON_A2_zxtb:
  si_ALU32_si                     <"zxtb",     int_hexagon_A2_zxtb>;

/********************************************************************
*            ALU32/PRED                                             *
*********************************************************************/

// ALU32 / PRED / Compare.
def HEXAGON_C2_cmpeq:
  qi_ALU32_sisi                   <"cmp.eq",   int_hexagon_C2_cmpeq>;
def HEXAGON_C2_cmpeqi:
  qi_ALU32_sis10                  <"cmp.eq",   int_hexagon_C2_cmpeqi>;
def HEXAGON_C2_cmpgei:
  qi_ALU32_sis8                   <"cmp.ge",   int_hexagon_C2_cmpgei>;
def HEXAGON_C2_cmpgeui:
  qi_ALU32_siu8                   <"cmp.geu",  int_hexagon_C2_cmpgeui>;
def HEXAGON_C2_cmpgt:
  qi_ALU32_sisi                   <"cmp.gt",   int_hexagon_C2_cmpgt>;
def HEXAGON_C2_cmpgti:
  qi_ALU32_sis10                  <"cmp.gt",   int_hexagon_C2_cmpgti>;
def HEXAGON_C2_cmpgtu:
  qi_ALU32_sisi                   <"cmp.gtu",  int_hexagon_C2_cmpgtu>;
def HEXAGON_C2_cmpgtui:
  qi_ALU32_siu9                   <"cmp.gtu",  int_hexagon_C2_cmpgtui>;
def HEXAGON_C2_cmplt:
  qi_ALU32_sisi                   <"cmp.lt",   int_hexagon_C2_cmplt>;
def HEXAGON_C2_cmpltu:
  qi_ALU32_sisi                   <"cmp.ltu",  int_hexagon_C2_cmpltu>;

/********************************************************************
*            ALU32/VH                                               *
*********************************************************************/

// ALU32 / VH / Vector add halfwords.
// Rd32=vadd[u]h(Rs32,Rt32:sat]
def HEXAGON_A2_svaddh:
  si_ALU32_sisi                   <"vaddh",    int_hexagon_A2_svaddh>;
def HEXAGON_A2_svaddhs:
  si_ALU32_sisi_sat               <"vaddh",    int_hexagon_A2_svaddhs>;
def HEXAGON_A2_svadduhs:
  si_ALU32_sisi_sat               <"vadduh",   int_hexagon_A2_svadduhs>;

// ALU32 / VH / Vector average halfwords.
def HEXAGON_A2_svavgh:
  si_ALU32_sisi                   <"vavgh",    int_hexagon_A2_svavgh>;
def HEXAGON_A2_svavghs:
  si_ALU32_sisi_rnd               <"vavgh",    int_hexagon_A2_svavghs>;
def HEXAGON_A2_svnavgh:
  si_ALU32_sisi                   <"vnavgh",   int_hexagon_A2_svnavgh>;

// ALU32 / VH / Vector subtract halfwords.
def HEXAGON_A2_svsubh:
  si_ALU32_sisi                   <"vsubh",    int_hexagon_A2_svsubh>;
def HEXAGON_A2_svsubhs:
  si_ALU32_sisi_sat               <"vsubh",    int_hexagon_A2_svsubhs>;
def HEXAGON_A2_svsubuhs:
  si_ALU32_sisi_sat               <"vsubuh",   int_hexagon_A2_svsubuhs>;

/********************************************************************
*            ALU64/ALU                                              *
*********************************************************************/

// ALU64 / ALU / Add.
def HEXAGON_A2_addp:
  di_ALU64_didi                   <"add",      int_hexagon_A2_addp>;
def HEXAGON_A2_addsat:
  si_ALU64_sisi_sat               <"add",      int_hexagon_A2_addsat>;

// ALU64 / ALU / Add halfword.
// Even though the definition says hl, it should be lh -
//so DON'T change the class " si_ALU64_sisi_l16_lh " it inherits.
def HEXAGON_A2_addh_l16_hl:
  si_ALU64_sisi_l16_lh            <"add",      int_hexagon_A2_addh_l16_hl>;
def HEXAGON_A2_addh_l16_ll:
  si_ALU64_sisi_l16_ll            <"add",      int_hexagon_A2_addh_l16_ll>;

def HEXAGON_A2_addh_l16_sat_hl:
  si_ALU64_sisi_l16_sat_lh        <"add",      int_hexagon_A2_addh_l16_sat_hl>;
def HEXAGON_A2_addh_l16_sat_ll:
  si_ALU64_sisi_l16_sat_ll        <"add",      int_hexagon_A2_addh_l16_sat_ll>;

def HEXAGON_A2_addh_h16_hh:
  si_ALU64_sisi_h16_hh            <"add",      int_hexagon_A2_addh_h16_hh>;
def HEXAGON_A2_addh_h16_hl:
  si_ALU64_sisi_h16_hl            <"add",      int_hexagon_A2_addh_h16_hl>;
def HEXAGON_A2_addh_h16_lh:
  si_ALU64_sisi_h16_lh            <"add",      int_hexagon_A2_addh_h16_lh>;
def HEXAGON_A2_addh_h16_ll:
  si_ALU64_sisi_h16_ll            <"add",      int_hexagon_A2_addh_h16_ll>;

def HEXAGON_A2_addh_h16_sat_hh:
  si_ALU64_sisi_h16_sat_hh        <"add",      int_hexagon_A2_addh_h16_sat_hh>;
def HEXAGON_A2_addh_h16_sat_hl:
  si_ALU64_sisi_h16_sat_hl        <"add",      int_hexagon_A2_addh_h16_sat_hl>;
def HEXAGON_A2_addh_h16_sat_lh:
  si_ALU64_sisi_h16_sat_lh        <"add",      int_hexagon_A2_addh_h16_sat_lh>;
def HEXAGON_A2_addh_h16_sat_ll:
  si_ALU64_sisi_h16_sat_ll        <"add",      int_hexagon_A2_addh_h16_sat_ll>;

// ALU64 / ALU / Compare.
def HEXAGON_C2_cmpeqp:
  qi_ALU64_didi                   <"cmp.eq",   int_hexagon_C2_cmpeqp>;
def HEXAGON_C2_cmpgtp:
  qi_ALU64_didi                   <"cmp.gt",   int_hexagon_C2_cmpgtp>;
def HEXAGON_C2_cmpgtup:
  qi_ALU64_didi                   <"cmp.gtu",  int_hexagon_C2_cmpgtup>;

// ALU64 / ALU / Logical operations.
def HEXAGON_A2_andp:
  di_ALU64_didi                   <"and",      int_hexagon_A2_andp>;
def HEXAGON_A2_orp:
  di_ALU64_didi                   <"or",       int_hexagon_A2_orp>;
def HEXAGON_A2_xorp:
  di_ALU64_didi                   <"xor",      int_hexagon_A2_xorp>;

// ALU64 / ALU / Maximum.
def HEXAGON_A2_max:
  si_ALU64_sisi                   <"max",      int_hexagon_A2_max>;
def HEXAGON_A2_maxu:
  si_ALU64_sisi                   <"maxu",     int_hexagon_A2_maxu>;

// ALU64 / ALU / Minimum.
def HEXAGON_A2_min:
  si_ALU64_sisi                   <"min",      int_hexagon_A2_min>;
def HEXAGON_A2_minu:
  si_ALU64_sisi                   <"minu",     int_hexagon_A2_minu>;

// ALU64 / ALU / Subtract.
def HEXAGON_A2_subp:
  di_ALU64_didi                   <"sub",      int_hexagon_A2_subp>;
def HEXAGON_A2_subsat:
  si_ALU64_sisi_sat               <"sub",      int_hexagon_A2_subsat>;

// ALU64 / ALU / Subtract halfword.
// Even though the definition says hl, it should be lh -
//so DON'T change the class " si_ALU64_sisi_l16_lh " it inherits.
def HEXAGON_A2_subh_l16_hl:
  si_ALU64_sisi_l16_lh            <"sub",      int_hexagon_A2_subh_l16_hl>;
def HEXAGON_A2_subh_l16_ll:
  si_ALU64_sisi_l16_ll            <"sub",      int_hexagon_A2_subh_l16_ll>;

def HEXAGON_A2_subh_l16_sat_hl:
  si_ALU64_sisi_l16_sat_lh        <"sub",      int_hexagon_A2_subh_l16_sat_hl>;
def HEXAGON_A2_subh_l16_sat_ll:
  si_ALU64_sisi_l16_sat_ll        <"sub",      int_hexagon_A2_subh_l16_sat_ll>;

def HEXAGON_A2_subh_h16_hh:
  si_ALU64_sisi_h16_hh            <"sub",      int_hexagon_A2_subh_h16_hh>;
def HEXAGON_A2_subh_h16_hl:
  si_ALU64_sisi_h16_hl            <"sub",      int_hexagon_A2_subh_h16_hl>;
def HEXAGON_A2_subh_h16_lh:
  si_ALU64_sisi_h16_lh            <"sub",      int_hexagon_A2_subh_h16_lh>;
def HEXAGON_A2_subh_h16_ll:
  si_ALU64_sisi_h16_ll            <"sub",      int_hexagon_A2_subh_h16_ll>;

def HEXAGON_A2_subh_h16_sat_hh:
  si_ALU64_sisi_h16_sat_hh        <"sub",      int_hexagon_A2_subh_h16_sat_hh>;
def HEXAGON_A2_subh_h16_sat_hl:
  si_ALU64_sisi_h16_sat_hl        <"sub",      int_hexagon_A2_subh_h16_sat_hl>;
def HEXAGON_A2_subh_h16_sat_lh:
  si_ALU64_sisi_h16_sat_lh        <"sub",      int_hexagon_A2_subh_h16_sat_lh>;
def HEXAGON_A2_subh_h16_sat_ll:
  si_ALU64_sisi_h16_sat_ll        <"sub",      int_hexagon_A2_subh_h16_sat_ll>;

// ALU64 / ALU / Transfer register.
def HEXAGON_A2_tfrp:
  di_ALU64_di                     <"",         int_hexagon_A2_tfrp>;

/********************************************************************
*            ALU64/BIT                                              *
*********************************************************************/

// ALU64 / BIT / Masked parity.
def HEXAGON_S2_parityp:
  si_ALU64_didi                   <"parity",   int_hexagon_S2_parityp>;

/********************************************************************
*            ALU64/PERM                                             *
*********************************************************************/

// ALU64 / PERM / Vector pack high and low halfwords.
def HEXAGON_S2_packhl:
  di_ALU64_sisi                   <"packhl",   int_hexagon_S2_packhl>;

/********************************************************************
*            ALU64/VB                                               *
*********************************************************************/

// ALU64 / VB / Vector add unsigned bytes.
def HEXAGON_A2_vaddub:
  di_ALU64_didi                   <"vaddub",   int_hexagon_A2_vaddub>;
def HEXAGON_A2_vaddubs:
  di_ALU64_didi_sat               <"vaddub",   int_hexagon_A2_vaddubs>;

// ALU64 / VB / Vector average unsigned bytes.
def HEXAGON_A2_vavgub:
  di_ALU64_didi                   <"vavgub",   int_hexagon_A2_vavgub>;
def HEXAGON_A2_vavgubr:
  di_ALU64_didi_rnd               <"vavgub",   int_hexagon_A2_vavgubr>;

// ALU64 / VB / Vector compare unsigned bytes.
def HEXAGON_A2_vcmpbeq:
  qi_ALU64_didi                   <"vcmpb.eq", int_hexagon_A2_vcmpbeq>;
def HEXAGON_A2_vcmpbgtu:
  qi_ALU64_didi                   <"vcmpb.gtu",int_hexagon_A2_vcmpbgtu>;

// ALU64 / VB / Vector maximum/minimum unsigned bytes.
def HEXAGON_A2_vmaxub:
  di_ALU64_didi                   <"vmaxub",   int_hexagon_A2_vmaxub>;
def HEXAGON_A2_vminub:
  di_ALU64_didi                   <"vminub",   int_hexagon_A2_vminub>;

// ALU64 / VB / Vector subtract unsigned bytes.
def HEXAGON_A2_vsubub:
  di_ALU64_didi                   <"vsubub",   int_hexagon_A2_vsubub>;
def HEXAGON_A2_vsububs:
  di_ALU64_didi_sat               <"vsubub",   int_hexagon_A2_vsububs>;

// ALU64 / VB / Vector mux.
def HEXAGON_C2_vmux:
  di_ALU64_qididi                 <"vmux",     int_hexagon_C2_vmux>;


/********************************************************************
*            ALU64/VH                                               *
*********************************************************************/

// ALU64 / VH / Vector add halfwords.
// Rdd64=vadd[u]h(Rss64,Rtt64:sat]
def HEXAGON_A2_vaddh:
  di_ALU64_didi                   <"vaddh",    int_hexagon_A2_vaddh>;
def HEXAGON_A2_vaddhs:
  di_ALU64_didi_sat               <"vaddh",    int_hexagon_A2_vaddhs>;
def HEXAGON_A2_vadduhs:
  di_ALU64_didi_sat               <"vadduh",   int_hexagon_A2_vadduhs>;

// ALU64 / VH / Vector average halfwords.
// Rdd64=v[n]avg[u]h(Rss64,Rtt64:rnd/:crnd][:sat]
def HEXAGON_A2_vavgh:
  di_ALU64_didi                   <"vavgh",    int_hexagon_A2_vavgh>;
def HEXAGON_A2_vavghcr:
  di_ALU64_didi_crnd              <"vavgh",    int_hexagon_A2_vavghcr>;
def HEXAGON_A2_vavghr:
  di_ALU64_didi_rnd               <"vavgh",    int_hexagon_A2_vavghr>;
def HEXAGON_A2_vavguh:
  di_ALU64_didi                   <"vavguh",   int_hexagon_A2_vavguh>;
def HEXAGON_A2_vavguhr:
  di_ALU64_didi_rnd               <"vavguh",   int_hexagon_A2_vavguhr>;
def HEXAGON_A2_vnavgh:
  di_ALU64_didi                   <"vnavgh",   int_hexagon_A2_vnavgh>;
def HEXAGON_A2_vnavghcr:
  di_ALU64_didi_crnd_sat          <"vnavgh",   int_hexagon_A2_vnavghcr>;
def HEXAGON_A2_vnavghr:
  di_ALU64_didi_rnd_sat           <"vnavgh",   int_hexagon_A2_vnavghr>;

// ALU64 / VH / Vector compare halfwords.
def HEXAGON_A2_vcmpheq:
  qi_ALU64_didi                   <"vcmph.eq", int_hexagon_A2_vcmpheq>;
def HEXAGON_A2_vcmphgt:
  qi_ALU64_didi                   <"vcmph.gt", int_hexagon_A2_vcmphgt>;
def HEXAGON_A2_vcmphgtu:
  qi_ALU64_didi                   <"vcmph.gtu",int_hexagon_A2_vcmphgtu>;

// ALU64 / VH / Vector maximum halfwords.
def HEXAGON_A2_vmaxh:
  di_ALU64_didi                   <"vmaxh",    int_hexagon_A2_vmaxh>;
def HEXAGON_A2_vmaxuh:
  di_ALU64_didi                   <"vmaxuh",   int_hexagon_A2_vmaxuh>;

// ALU64 / VH / Vector minimum halfwords.
def HEXAGON_A2_vminh:
  di_ALU64_didi                   <"vminh",    int_hexagon_A2_vminh>;
def HEXAGON_A2_vminuh:
  di_ALU64_didi                   <"vminuh",   int_hexagon_A2_vminuh>;

// ALU64 / VH / Vector subtract halfwords.
def HEXAGON_A2_vsubh:
  di_ALU64_didi                   <"vsubh",    int_hexagon_A2_vsubh>;
def HEXAGON_A2_vsubhs:
  di_ALU64_didi_sat               <"vsubh",    int_hexagon_A2_vsubhs>;
def HEXAGON_A2_vsubuhs:
  di_ALU64_didi_sat               <"vsubuh",   int_hexagon_A2_vsubuhs>;


/********************************************************************
*            ALU64/VW                                               *
*********************************************************************/

// ALU64 / VW / Vector add words.
// Rdd32=vaddw(Rss32,Rtt32)[:sat]
def HEXAGON_A2_vaddw:
  di_ALU64_didi                   <"vaddw",    int_hexagon_A2_vaddw>;
def HEXAGON_A2_vaddws:
  di_ALU64_didi_sat               <"vaddw",   int_hexagon_A2_vaddws>;

// ALU64 / VW / Vector average words.
def HEXAGON_A2_vavguw:
  di_ALU64_didi                   <"vavguw",   int_hexagon_A2_vavguw>;
def HEXAGON_A2_vavguwr:
  di_ALU64_didi_rnd               <"vavguw",   int_hexagon_A2_vavguwr>;
def HEXAGON_A2_vavgw:
  di_ALU64_didi                   <"vavgw",    int_hexagon_A2_vavgw>;
def HEXAGON_A2_vavgwcr:
  di_ALU64_didi_crnd              <"vavgw",    int_hexagon_A2_vavgwcr>;
def HEXAGON_A2_vavgwr:
  di_ALU64_didi_rnd               <"vavgw",    int_hexagon_A2_vavgwr>;
def HEXAGON_A2_vnavgw:
  di_ALU64_didi                   <"vnavgw",   int_hexagon_A2_vnavgw>;
def HEXAGON_A2_vnavgwcr:
  di_ALU64_didi_crnd_sat          <"vnavgw",   int_hexagon_A2_vnavgwcr>;
def HEXAGON_A2_vnavgwr:
  di_ALU64_didi_rnd_sat           <"vnavgw",   int_hexagon_A2_vnavgwr>;

// ALU64 / VW / Vector compare words.
def HEXAGON_A2_vcmpweq:
  qi_ALU64_didi                   <"vcmpw.eq", int_hexagon_A2_vcmpweq>;
def HEXAGON_A2_vcmpwgt:
  qi_ALU64_didi                   <"vcmpw.gt", int_hexagon_A2_vcmpwgt>;
def HEXAGON_A2_vcmpwgtu:
  qi_ALU64_didi                   <"vcmpw.gtu",int_hexagon_A2_vcmpwgtu>;

// ALU64 / VW / Vector maximum words.
def HEXAGON_A2_vmaxw:
  di_ALU64_didi                   <"vmaxw",    int_hexagon_A2_vmaxw>;
def HEXAGON_A2_vmaxuw:
  di_ALU64_didi                   <"vmaxuw",   int_hexagon_A2_vmaxuw>;

// ALU64 / VW / Vector minimum words.
def HEXAGON_A2_vminw:
  di_ALU64_didi                   <"vminw",    int_hexagon_A2_vminw>;
def HEXAGON_A2_vminuw:
  di_ALU64_didi                   <"vminuw",   int_hexagon_A2_vminuw>;

// ALU64 / VW / Vector subtract words.
def HEXAGON_A2_vsubw:
  di_ALU64_didi                   <"vsubw",    int_hexagon_A2_vsubw>;
def HEXAGON_A2_vsubws:
  di_ALU64_didi_sat               <"vsubw",    int_hexagon_A2_vsubws>;


/********************************************************************
*            CR                                                     *
*********************************************************************/

// CR / Logical reductions on predicates.
def HEXAGON_C2_all8:
  qi_SInst_qi                     <"all8",     int_hexagon_C2_all8>;
def HEXAGON_C2_any8:
  qi_SInst_qi                     <"any8",     int_hexagon_C2_any8>;

// CR / Logical operations on predicates.
def HEXAGON_C2_pxfer_map:
  qi_SInst_qi_pxfer               <"",         int_hexagon_C2_pxfer_map>;
def HEXAGON_C2_and:
  qi_SInst_qiqi                   <"and",      int_hexagon_C2_and>;
def HEXAGON_C2_andn:
  qi_SInst_qiqi_neg               <"and",      int_hexagon_C2_andn>;
def HEXAGON_C2_not:
  qi_SInst_qi                     <"not",      int_hexagon_C2_not>;
def HEXAGON_C2_or:
  qi_SInst_qiqi                   <"or",       int_hexagon_C2_or>;
def HEXAGON_C2_orn:
  qi_SInst_qiqi_neg               <"or",       int_hexagon_C2_orn>;
def HEXAGON_C2_xor:
  qi_SInst_qiqi                   <"xor",      int_hexagon_C2_xor>;


/********************************************************************
*            MTYPE/ALU                                              *
*********************************************************************/

// MTYPE / ALU / Add and accumulate.
def HEXAGON_M2_acci:
  si_MInst_sisisi_acc             <"add",      int_hexagon_M2_acci>;
def HEXAGON_M2_accii:
  si_MInst_sisis8_acc             <"add",      int_hexagon_M2_accii>;
def HEXAGON_M2_nacci:
  si_MInst_sisisi_nac             <"add",      int_hexagon_M2_nacci>;
def HEXAGON_M2_naccii:
  si_MInst_sisis8_nac             <"add",      int_hexagon_M2_naccii>;

// MTYPE / ALU / Subtract and accumulate.
def HEXAGON_M2_subacc:
  si_MInst_sisisi_acc             <"sub",      int_hexagon_M2_subacc>;

// MTYPE / ALU / Vector absolute difference.
def HEXAGON_M2_vabsdiffh:
  di_MInst_didi                   <"vabsdiffh",int_hexagon_M2_vabsdiffh>;
def HEXAGON_M2_vabsdiffw:
  di_MInst_didi                   <"vabsdiffw",int_hexagon_M2_vabsdiffw>;

// MTYPE / ALU / XOR and xor with destination.
def HEXAGON_M2_xor_xacc:
  si_MInst_sisisi_xacc            <"xor",      int_hexagon_M2_xor_xacc>;


/********************************************************************
*            MTYPE/COMPLEX                                          *
*********************************************************************/

// MTYPE / COMPLEX / Complex multiply.
// Rdd[-+]=cmpy(Rs, Rt:<<1]:sat
def HEXAGON_M2_cmpys_s1:
  di_MInst_sisi_s1_sat            <"cmpy",     int_hexagon_M2_cmpys_s1>;
def HEXAGON_M2_cmpys_s0:
  di_MInst_sisi_sat               <"cmpy",     int_hexagon_M2_cmpys_s0>;
def HEXAGON_M2_cmpysc_s1:
  di_MInst_sisi_s1_sat_conj       <"cmpy",     int_hexagon_M2_cmpysc_s1>;
def HEXAGON_M2_cmpysc_s0:
  di_MInst_sisi_sat_conj          <"cmpy",     int_hexagon_M2_cmpysc_s0>;

def HEXAGON_M2_cmacs_s1:
  di_MInst_disisi_acc_s1_sat      <"cmpy",     int_hexagon_M2_cmacs_s1>;
def HEXAGON_M2_cmacs_s0:
  di_MInst_disisi_acc_sat         <"cmpy",     int_hexagon_M2_cmacs_s0>;
def HEXAGON_M2_cmacsc_s1:
  di_MInst_disisi_acc_s1_sat_conj <"cmpy",     int_hexagon_M2_cmacsc_s1>;
def HEXAGON_M2_cmacsc_s0:
  di_MInst_disisi_acc_sat_conj    <"cmpy",     int_hexagon_M2_cmacsc_s0>;

def HEXAGON_M2_cnacs_s1:
  di_MInst_disisi_nac_s1_sat      <"cmpy",     int_hexagon_M2_cnacs_s1>;
def HEXAGON_M2_cnacs_s0:
  di_MInst_disisi_nac_sat         <"cmpy",     int_hexagon_M2_cnacs_s0>;
def HEXAGON_M2_cnacsc_s1:
  di_MInst_disisi_nac_s1_sat_conj <"cmpy",     int_hexagon_M2_cnacsc_s1>;
def HEXAGON_M2_cnacsc_s0:
  di_MInst_disisi_nac_sat_conj    <"cmpy",     int_hexagon_M2_cnacsc_s0>;

// MTYPE / COMPLEX / Complex multiply real or imaginary.
def HEXAGON_M2_cmpyr_s0:
  di_MInst_sisi                   <"cmpyr",    int_hexagon_M2_cmpyr_s0>;
def HEXAGON_M2_cmacr_s0:
  di_MInst_disisi_acc             <"cmpyr",    int_hexagon_M2_cmacr_s0>;

def HEXAGON_M2_cmpyi_s0:
  di_MInst_sisi                   <"cmpyi",    int_hexagon_M2_cmpyi_s0>;
def HEXAGON_M2_cmaci_s0:
  di_MInst_disisi_acc             <"cmpyi",    int_hexagon_M2_cmaci_s0>;

// MTYPE / COMPLEX / Complex multiply with round and pack.
// Rxx32+=cmpy(Rs32,[*]Rt32:<<1]:rnd:sat
def HEXAGON_M2_cmpyrs_s0:
  si_MInst_sisi_rnd_sat           <"cmpy",     int_hexagon_M2_cmpyrs_s0>;
def HEXAGON_M2_cmpyrs_s1:
  si_MInst_sisi_s1_rnd_sat        <"cmpy",     int_hexagon_M2_cmpyrs_s1>;

def HEXAGON_M2_cmpyrsc_s0:
  si_MInst_sisi_rnd_sat_conj      <"cmpy",     int_hexagon_M2_cmpyrsc_s0>;
def HEXAGON_M2_cmpyrsc_s1:
  si_MInst_sisi_s1_rnd_sat_conj   <"cmpy",     int_hexagon_M2_cmpyrsc_s1>;

//MTYPE / COMPLEX / Vector complex multiply real or imaginary.
def HEXAGON_M2_vcmpy_s0_sat_i:
  di_MInst_didi_sat               <"vcmpyi",   int_hexagon_M2_vcmpy_s0_sat_i>;
def HEXAGON_M2_vcmpy_s1_sat_i:
  di_MInst_didi_s1_sat            <"vcmpyi",   int_hexagon_M2_vcmpy_s1_sat_i>;

def HEXAGON_M2_vcmpy_s0_sat_r:
  di_MInst_didi_sat               <"vcmpyr",   int_hexagon_M2_vcmpy_s0_sat_r>;
def HEXAGON_M2_vcmpy_s1_sat_r:
  di_MInst_didi_s1_sat            <"vcmpyr",   int_hexagon_M2_vcmpy_s1_sat_r>;

def HEXAGON_M2_vcmac_s0_sat_i:
  di_MInst_dididi_acc_sat         <"vcmpyi",   int_hexagon_M2_vcmac_s0_sat_i>;
def HEXAGON_M2_vcmac_s0_sat_r:
  di_MInst_dididi_acc_sat         <"vcmpyr",   int_hexagon_M2_vcmac_s0_sat_r>;

//MTYPE / COMPLEX / Vector reduce complex multiply real or imaginary.
def HEXAGON_M2_vrcmpyi_s0:
  di_MInst_didi                   <"vrcmpyi",  int_hexagon_M2_vrcmpyi_s0>;
def HEXAGON_M2_vrcmpyr_s0:
  di_MInst_didi                   <"vrcmpyr",  int_hexagon_M2_vrcmpyr_s0>;

def HEXAGON_M2_vrcmpyi_s0c:
  di_MInst_didi_conj              <"vrcmpyi",  int_hexagon_M2_vrcmpyi_s0c>;
def HEXAGON_M2_vrcmpyr_s0c:
  di_MInst_didi_conj              <"vrcmpyr",  int_hexagon_M2_vrcmpyr_s0c>;

def HEXAGON_M2_vrcmaci_s0:
  di_MInst_dididi_acc             <"vrcmpyi",  int_hexagon_M2_vrcmaci_s0>;
def HEXAGON_M2_vrcmacr_s0:
  di_MInst_dididi_acc             <"vrcmpyr",  int_hexagon_M2_vrcmacr_s0>;

def HEXAGON_M2_vrcmaci_s0c:
  di_MInst_dididi_acc_conj        <"vrcmpyi",  int_hexagon_M2_vrcmaci_s0c>;
def HEXAGON_M2_vrcmacr_s0c:
  di_MInst_dididi_acc_conj        <"vrcmpyr",  int_hexagon_M2_vrcmacr_s0c>;


/********************************************************************
*            MTYPE/MPYH                                             *
*********************************************************************/

// MTYPE / MPYH / Multiply and use lower result.
//def HEXAGON_M2_mpysmi:
//FIXME: Hexagon_M2_mpysmi should really by of the type si_MInst_sim9,
// not si_MInst_sis9 - but for now, we will use s9.
// def Hexagon_M2_mpysmi:
//  si_MInst_sim9                   <"mpyi",     int_hexagon_M2_mpysmi>;
def Hexagon_M2_mpysmi:
  si_MInst_sis9                   <"mpyi",     int_hexagon_M2_mpysmi>;
def HEXAGON_M2_mpyi:
  si_MInst_sisi                   <"mpyi",     int_hexagon_M2_mpyi>;
def HEXAGON_M2_mpyui:
  si_MInst_sisi                   <"mpyui",    int_hexagon_M2_mpyui>;
def HEXAGON_M2_macsip:
  si_MInst_sisiu8_acc             <"mpyi",     int_hexagon_M2_macsip>;
def HEXAGON_M2_maci:
  si_MInst_sisisi_acc             <"mpyi",     int_hexagon_M2_maci>;
def HEXAGON_M2_macsin:
  si_MInst_sisiu8_nac             <"mpyi",     int_hexagon_M2_macsin>;

// MTYPE / MPYH / Multiply word by half (32x16).
//Rdd[+]=vmpywoh(Rss,Rtt)[:<<1][:rnd][:sat]
//Rdd[+]=vmpyweh(Rss,Rtt)[:<<1][:rnd][:sat]
def HEXAGON_M2_mmpyl_rs1:
  di_MInst_didi_s1_rnd_sat        <"vmpyweh",  int_hexagon_M2_mmpyl_rs1>;
def HEXAGON_M2_mmpyl_s1:
  di_MInst_didi_s1_sat            <"vmpyweh",  int_hexagon_M2_mmpyl_s1>;
def HEXAGON_M2_mmpyl_rs0:
  di_MInst_didi_rnd_sat           <"vmpyweh",  int_hexagon_M2_mmpyl_rs0>;
def HEXAGON_M2_mmpyl_s0:
  di_MInst_didi_sat               <"vmpyweh",  int_hexagon_M2_mmpyl_s0>;
def HEXAGON_M2_mmpyh_rs1:
  di_MInst_didi_s1_rnd_sat        <"vmpywoh",  int_hexagon_M2_mmpyh_rs1>;
def HEXAGON_M2_mmpyh_s1:
  di_MInst_didi_s1_sat            <"vmpywoh",  int_hexagon_M2_mmpyh_s1>;
def HEXAGON_M2_mmpyh_rs0:
  di_MInst_didi_rnd_sat           <"vmpywoh",  int_hexagon_M2_mmpyh_rs0>;
def HEXAGON_M2_mmpyh_s0:
  di_MInst_didi_sat               <"vmpywoh",  int_hexagon_M2_mmpyh_s0>;
def HEXAGON_M2_mmacls_rs1:
  di_MInst_dididi_acc_s1_rnd_sat  <"vmpyweh",  int_hexagon_M2_mmacls_rs1>;
def HEXAGON_M2_mmacls_s1:
  di_MInst_dididi_acc_s1_sat      <"vmpyweh",  int_hexagon_M2_mmacls_s1>;
def HEXAGON_M2_mmacls_rs0:
  di_MInst_dididi_acc_rnd_sat     <"vmpyweh",  int_hexagon_M2_mmacls_rs0>;
def HEXAGON_M2_mmacls_s0:
  di_MInst_dididi_acc_sat         <"vmpyweh",  int_hexagon_M2_mmacls_s0>;
def HEXAGON_M2_mmachs_rs1:
  di_MInst_dididi_acc_s1_rnd_sat  <"vmpywoh",  int_hexagon_M2_mmachs_rs1>;
def HEXAGON_M2_mmachs_s1:
  di_MInst_dididi_acc_s1_sat      <"vmpywoh",  int_hexagon_M2_mmachs_s1>;
def HEXAGON_M2_mmachs_rs0:
  di_MInst_dididi_acc_rnd_sat     <"vmpywoh",  int_hexagon_M2_mmachs_rs0>;
def HEXAGON_M2_mmachs_s0:
  di_MInst_dididi_acc_sat         <"vmpywoh",  int_hexagon_M2_mmachs_s0>;

// MTYPE / MPYH / Multiply word by unsigned half (32x16).
//Rdd[+]=vmpywouh(Rss,Rtt)[:<<1][:rnd][:sat]
//Rdd[+]=vmpyweuh(Rss,Rtt)[:<<1][:rnd][:sat]
def HEXAGON_M2_mmpyul_rs1:
  di_MInst_didi_s1_rnd_sat        <"vmpyweuh", int_hexagon_M2_mmpyul_rs1>;
def HEXAGON_M2_mmpyul_s1:
  di_MInst_didi_s1_sat            <"vmpyweuh", int_hexagon_M2_mmpyul_s1>;
def HEXAGON_M2_mmpyul_rs0:
  di_MInst_didi_rnd_sat           <"vmpyweuh", int_hexagon_M2_mmpyul_rs0>;
def HEXAGON_M2_mmpyul_s0:
  di_MInst_didi_sat               <"vmpyweuh", int_hexagon_M2_mmpyul_s0>;
def HEXAGON_M2_mmpyuh_rs1:
  di_MInst_didi_s1_rnd_sat        <"vmpywouh", int_hexagon_M2_mmpyuh_rs1>;
def HEXAGON_M2_mmpyuh_s1:
  di_MInst_didi_s1_sat            <"vmpywouh", int_hexagon_M2_mmpyuh_s1>;
def HEXAGON_M2_mmpyuh_rs0:
  di_MInst_didi_rnd_sat           <"vmpywouh", int_hexagon_M2_mmpyuh_rs0>;
def HEXAGON_M2_mmpyuh_s0:
  di_MInst_didi_sat               <"vmpywouh", int_hexagon_M2_mmpyuh_s0>;
def HEXAGON_M2_mmaculs_rs1:
  di_MInst_dididi_acc_s1_rnd_sat  <"vmpyweuh", int_hexagon_M2_mmaculs_rs1>;
def HEXAGON_M2_mmaculs_s1:
  di_MInst_dididi_acc_s1_sat      <"vmpyweuh", int_hexagon_M2_mmaculs_s1>;
def HEXAGON_M2_mmaculs_rs0:
  di_MInst_dididi_acc_rnd_sat     <"vmpyweuh", int_hexagon_M2_mmaculs_rs0>;
def HEXAGON_M2_mmaculs_s0:
  di_MInst_dididi_acc_sat         <"vmpyweuh", int_hexagon_M2_mmaculs_s0>;
def HEXAGON_M2_mmacuhs_rs1:
  di_MInst_dididi_acc_s1_rnd_sat  <"vmpywouh", int_hexagon_M2_mmacuhs_rs1>;
def HEXAGON_M2_mmacuhs_s1:
  di_MInst_dididi_acc_s1_sat      <"vmpywouh", int_hexagon_M2_mmacuhs_s1>;
def HEXAGON_M2_mmacuhs_rs0:
  di_MInst_dididi_acc_rnd_sat     <"vmpywouh", int_hexagon_M2_mmacuhs_rs0>;
def HEXAGON_M2_mmacuhs_s0:
  di_MInst_dididi_acc_sat         <"vmpywouh", int_hexagon_M2_mmacuhs_s0>;

// MTYPE / MPYH / Multiply and use upper result.
def HEXAGON_M2_hmmpyh_rs1:
  si_MInst_sisi_h_s1_rnd_sat      <"mpy",      int_hexagon_M2_hmmpyh_rs1>;
def HEXAGON_M2_hmmpyl_rs1:
  si_MInst_sisi_l_s1_rnd_sat      <"mpy",      int_hexagon_M2_hmmpyl_rs1>;
def HEXAGON_M2_mpy_up:
  si_MInst_sisi                   <"mpy",      int_hexagon_M2_mpy_up>;
def HEXAGON_M2_dpmpyss_rnd_s0:
  si_MInst_sisi_rnd               <"mpy",      int_hexagon_M2_dpmpyss_rnd_s0>;
def HEXAGON_M2_mpyu_up:
  si_MInst_sisi                   <"mpyu",     int_hexagon_M2_mpyu_up>;

// MTYPE / MPYH / Multiply and use full result.
def HEXAGON_M2_dpmpyuu_s0:
  di_MInst_sisi                   <"mpyu",     int_hexagon_M2_dpmpyuu_s0>;
def HEXAGON_M2_dpmpyuu_acc_s0:
  di_MInst_disisi_acc             <"mpyu",     int_hexagon_M2_dpmpyuu_acc_s0>;
def HEXAGON_M2_dpmpyuu_nac_s0:
  di_MInst_disisi_nac             <"mpyu",     int_hexagon_M2_dpmpyuu_nac_s0>;
def HEXAGON_M2_dpmpyss_s0:
  di_MInst_sisi                   <"mpy",      int_hexagon_M2_dpmpyss_s0>;
def HEXAGON_M2_dpmpyss_acc_s0:
  di_MInst_disisi_acc             <"mpy",      int_hexagon_M2_dpmpyss_acc_s0>;
def HEXAGON_M2_dpmpyss_nac_s0:
  di_MInst_disisi_nac             <"mpy",      int_hexagon_M2_dpmpyss_nac_s0>;


/********************************************************************
*            MTYPE/MPYS                                             *
*********************************************************************/

// MTYPE / MPYS / Scalar 16x16 multiply signed.
//Rd=mpy(Rs.[H|L],Rt.[H|L:<<0|:<<1]|
//          [:<<0[:rnd|:sat|:rnd:sat]|:<<1[:rnd|:sat|:rnd:sat]]]
def HEXAGON_M2_mpy_hh_s0:
  si_MInst_sisi_hh                <"mpy",     int_hexagon_M2_mpy_hh_s0>;
def HEXAGON_M2_mpy_hh_s1:
  si_MInst_sisi_hh_s1             <"mpy",     int_hexagon_M2_mpy_hh_s1>;
def HEXAGON_M2_mpy_rnd_hh_s1:
  si_MInst_sisi_rnd_hh_s1         <"mpy",     int_hexagon_M2_mpy_rnd_hh_s1>;
def HEXAGON_M2_mpy_sat_rnd_hh_s1:
  si_MInst_sisi_sat_rnd_hh_s1     <"mpy",     int_hexagon_M2_mpy_sat_rnd_hh_s1>;
def HEXAGON_M2_mpy_sat_hh_s1:
  si_MInst_sisi_sat_hh_s1         <"mpy",     int_hexagon_M2_mpy_sat_hh_s1>;
def HEXAGON_M2_mpy_rnd_hh_s0:
  si_MInst_sisi_rnd_hh            <"mpy",     int_hexagon_M2_mpy_rnd_hh_s0>;
def HEXAGON_M2_mpy_sat_rnd_hh_s0:
  si_MInst_sisi_sat_rnd_hh        <"mpy",     int_hexagon_M2_mpy_sat_rnd_hh_s0>;
def HEXAGON_M2_mpy_sat_hh_s0:
  si_MInst_sisi_sat_hh            <"mpy",     int_hexagon_M2_mpy_sat_hh_s0>;

def HEXAGON_M2_mpy_hl_s0:
  si_MInst_sisi_hl                <"mpy",     int_hexagon_M2_mpy_hl_s0>;
def HEXAGON_M2_mpy_hl_s1:
  si_MInst_sisi_hl_s1             <"mpy",     int_hexagon_M2_mpy_hl_s1>;
def HEXAGON_M2_mpy_rnd_hl_s1:
  si_MInst_sisi_rnd_hl_s1         <"mpy",     int_hexagon_M2_mpy_rnd_hl_s1>;
def HEXAGON_M2_mpy_sat_rnd_hl_s1:
  si_MInst_sisi_sat_rnd_hl_s1     <"mpy",     int_hexagon_M2_mpy_sat_rnd_hl_s1>;
def HEXAGON_M2_mpy_sat_hl_s1:
  si_MInst_sisi_sat_hl_s1         <"mpy",     int_hexagon_M2_mpy_sat_hl_s1>;
def HEXAGON_M2_mpy_rnd_hl_s0:
  si_MInst_sisi_rnd_hl            <"mpy",     int_hexagon_M2_mpy_rnd_hl_s0>;
def HEXAGON_M2_mpy_sat_rnd_hl_s0:
  si_MInst_sisi_sat_rnd_hl        <"mpy",     int_hexagon_M2_mpy_sat_rnd_hl_s0>;
def HEXAGON_M2_mpy_sat_hl_s0:
  si_MInst_sisi_sat_hl            <"mpy",     int_hexagon_M2_mpy_sat_hl_s0>;

def HEXAGON_M2_mpy_lh_s0:
  si_MInst_sisi_lh                <"mpy",     int_hexagon_M2_mpy_lh_s0>;
def HEXAGON_M2_mpy_lh_s1:
  si_MInst_sisi_lh_s1             <"mpy",     int_hexagon_M2_mpy_lh_s1>;
def HEXAGON_M2_mpy_rnd_lh_s1:
  si_MInst_sisi_rnd_lh_s1         <"mpy",     int_hexagon_M2_mpy_rnd_lh_s1>;
def HEXAGON_M2_mpy_sat_rnd_lh_s1:
  si_MInst_sisi_sat_rnd_lh_s1     <"mpy",     int_hexagon_M2_mpy_sat_rnd_lh_s1>;
def HEXAGON_M2_mpy_sat_lh_s1:
  si_MInst_sisi_sat_lh_s1         <"mpy",     int_hexagon_M2_mpy_sat_lh_s1>;
def HEXAGON_M2_mpy_rnd_lh_s0:
  si_MInst_sisi_rnd_lh            <"mpy",     int_hexagon_M2_mpy_rnd_lh_s0>;
def HEXAGON_M2_mpy_sat_rnd_lh_s0:
  si_MInst_sisi_sat_rnd_lh        <"mpy",     int_hexagon_M2_mpy_sat_rnd_lh_s0>;
def HEXAGON_M2_mpy_sat_lh_s0:
  si_MInst_sisi_sat_lh            <"mpy",     int_hexagon_M2_mpy_sat_lh_s0>;

def HEXAGON_M2_mpy_ll_s0:
  si_MInst_sisi_ll                <"mpy",     int_hexagon_M2_mpy_ll_s0>;
def HEXAGON_M2_mpy_ll_s1:
  si_MInst_sisi_ll_s1             <"mpy",     int_hexagon_M2_mpy_ll_s1>;
def HEXAGON_M2_mpy_rnd_ll_s1:
  si_MInst_sisi_rnd_ll_s1         <"mpy",     int_hexagon_M2_mpy_rnd_ll_s1>;
def HEXAGON_M2_mpy_sat_rnd_ll_s1:
  si_MInst_sisi_sat_rnd_ll_s1     <"mpy",     int_hexagon_M2_mpy_sat_rnd_ll_s1>;
def HEXAGON_M2_mpy_sat_ll_s1:
  si_MInst_sisi_sat_ll_s1         <"mpy",     int_hexagon_M2_mpy_sat_ll_s1>;
def HEXAGON_M2_mpy_rnd_ll_s0:
  si_MInst_sisi_rnd_ll            <"mpy",     int_hexagon_M2_mpy_rnd_ll_s0>;
def HEXAGON_M2_mpy_sat_rnd_ll_s0:
  si_MInst_sisi_sat_rnd_ll        <"mpy",     int_hexagon_M2_mpy_sat_rnd_ll_s0>;
def HEXAGON_M2_mpy_sat_ll_s0:
  si_MInst_sisi_sat_ll            <"mpy",     int_hexagon_M2_mpy_sat_ll_s0>;

//Rdd=mpy(Rs.[H|L],Rt.[H|L])[[:<<0|:<<1]|[:<<0:rnd|:<<1:rnd]]
def HEXAGON_M2_mpyd_hh_s0:
  di_MInst_sisi_hh                <"mpy",     int_hexagon_M2_mpyd_hh_s0>;
def HEXAGON_M2_mpyd_hh_s1:
  di_MInst_sisi_hh_s1             <"mpy",     int_hexagon_M2_mpyd_hh_s1>;
def HEXAGON_M2_mpyd_rnd_hh_s1:
  di_MInst_sisi_rnd_hh_s1         <"mpy",     int_hexagon_M2_mpyd_rnd_hh_s1>;
def HEXAGON_M2_mpyd_rnd_hh_s0:
  di_MInst_sisi_rnd_hh            <"mpy",     int_hexagon_M2_mpyd_rnd_hh_s0>;

def HEXAGON_M2_mpyd_hl_s0:
  di_MInst_sisi_hl                <"mpy",     int_hexagon_M2_mpyd_hl_s0>;
def HEXAGON_M2_mpyd_hl_s1:
  di_MInst_sisi_hl_s1             <"mpy",     int_hexagon_M2_mpyd_hl_s1>;
def HEXAGON_M2_mpyd_rnd_hl_s1:
  di_MInst_sisi_rnd_hl_s1         <"mpy",     int_hexagon_M2_mpyd_rnd_hl_s1>;
def HEXAGON_M2_mpyd_rnd_hl_s0:
  di_MInst_sisi_rnd_hl            <"mpy",     int_hexagon_M2_mpyd_rnd_hl_s0>;

def HEXAGON_M2_mpyd_lh_s0:
  di_MInst_sisi_lh                <"mpy",     int_hexagon_M2_mpyd_lh_s0>;
def HEXAGON_M2_mpyd_lh_s1:
  di_MInst_sisi_lh_s1             <"mpy",     int_hexagon_M2_mpyd_lh_s1>;
def HEXAGON_M2_mpyd_rnd_lh_s1:
  di_MInst_sisi_rnd_lh_s1         <"mpy",     int_hexagon_M2_mpyd_rnd_lh_s1>;
def HEXAGON_M2_mpyd_rnd_lh_s0:
  di_MInst_sisi_rnd_lh            <"mpy",     int_hexagon_M2_mpyd_rnd_lh_s0>;

def HEXAGON_M2_mpyd_ll_s0:
  di_MInst_sisi_ll                <"mpy",     int_hexagon_M2_mpyd_ll_s0>;
def HEXAGON_M2_mpyd_ll_s1:
  di_MInst_sisi_ll_s1             <"mpy",     int_hexagon_M2_mpyd_ll_s1>;
def HEXAGON_M2_mpyd_rnd_ll_s1:
  di_MInst_sisi_rnd_ll_s1         <"mpy",     int_hexagon_M2_mpyd_rnd_ll_s1>;
def HEXAGON_M2_mpyd_rnd_ll_s0:
  di_MInst_sisi_rnd_ll            <"mpy",     int_hexagon_M2_mpyd_rnd_ll_s0>;

//Rx+=mpy(Rs.[H|L],Rt.[H|L])[[[:<<0|:<<1]|[:<<0:sat|:<<1:sat]]
def HEXAGON_M2_mpy_acc_hh_s0:
  si_MInst_sisisi_acc_hh            <"mpy",  int_hexagon_M2_mpy_acc_hh_s0>;
def HEXAGON_M2_mpy_acc_hh_s1:
  si_MInst_sisisi_acc_hh_s1         <"mpy",  int_hexagon_M2_mpy_acc_hh_s1>;
def HEXAGON_M2_mpy_acc_sat_hh_s1:
  si_MInst_sisisi_acc_sat_hh_s1     <"mpy",  int_hexagon_M2_mpy_acc_sat_hh_s1>;
def HEXAGON_M2_mpy_acc_sat_hh_s0:
  si_MInst_sisisi_acc_sat_hh        <"mpy",  int_hexagon_M2_mpy_acc_sat_hh_s0>;

def HEXAGON_M2_mpy_acc_hl_s0:
  si_MInst_sisisi_acc_hl            <"mpy",  int_hexagon_M2_mpy_acc_hl_s0>;
def HEXAGON_M2_mpy_acc_hl_s1:
  si_MInst_sisisi_acc_hl_s1         <"mpy",  int_hexagon_M2_mpy_acc_hl_s1>;
def HEXAGON_M2_mpy_acc_sat_hl_s1:
  si_MInst_sisisi_acc_sat_hl_s1     <"mpy",  int_hexagon_M2_mpy_acc_sat_hl_s1>;
def HEXAGON_M2_mpy_acc_sat_hl_s0:
  si_MInst_sisisi_acc_sat_hl        <"mpy",  int_hexagon_M2_mpy_acc_sat_hl_s0>;

def HEXAGON_M2_mpy_acc_lh_s0:
  si_MInst_sisisi_acc_lh            <"mpy",  int_hexagon_M2_mpy_acc_lh_s0>;
def HEXAGON_M2_mpy_acc_lh_s1:
  si_MInst_sisisi_acc_lh_s1         <"mpy",  int_hexagon_M2_mpy_acc_lh_s1>;
def HEXAGON_M2_mpy_acc_sat_lh_s1:
  si_MInst_sisisi_acc_sat_lh_s1     <"mpy",  int_hexagon_M2_mpy_acc_sat_lh_s1>;
def HEXAGON_M2_mpy_acc_sat_lh_s0:
  si_MInst_sisisi_acc_sat_lh        <"mpy",  int_hexagon_M2_mpy_acc_sat_lh_s0>;

def HEXAGON_M2_mpy_acc_ll_s0:
  si_MInst_sisisi_acc_ll            <"mpy",  int_hexagon_M2_mpy_acc_ll_s0>;
def HEXAGON_M2_mpy_acc_ll_s1:
  si_MInst_sisisi_acc_ll_s1         <"mpy",  int_hexagon_M2_mpy_acc_ll_s1>;
def HEXAGON_M2_mpy_acc_sat_ll_s1:
  si_MInst_sisisi_acc_sat_ll_s1     <"mpy",  int_hexagon_M2_mpy_acc_sat_ll_s1>;
def HEXAGON_M2_mpy_acc_sat_ll_s0:
  si_MInst_sisisi_acc_sat_ll        <"mpy",  int_hexagon_M2_mpy_acc_sat_ll_s0>;

//Rx-=mpy(Rs.[H|L],Rt.[H|L])[[[:<<0|:<<1]|[:<<0:sat|:<<1:sat]]
def HEXAGON_M2_mpy_nac_hh_s0:
  si_MInst_sisisi_nac_hh            <"mpy",  int_hexagon_M2_mpy_nac_hh_s0>;
def HEXAGON_M2_mpy_nac_hh_s1:
  si_MInst_sisisi_nac_hh_s1         <"mpy",  int_hexagon_M2_mpy_nac_hh_s1>;
def HEXAGON_M2_mpy_nac_sat_hh_s1:
  si_MInst_sisisi_nac_sat_hh_s1     <"mpy",  int_hexagon_M2_mpy_nac_sat_hh_s1>;
def HEXAGON_M2_mpy_nac_sat_hh_s0:
  si_MInst_sisisi_nac_sat_hh        <"mpy",  int_hexagon_M2_mpy_nac_sat_hh_s0>;

def HEXAGON_M2_mpy_nac_hl_s0:
  si_MInst_sisisi_nac_hl            <"mpy",  int_hexagon_M2_mpy_nac_hl_s0>;
def HEXAGON_M2_mpy_nac_hl_s1:
  si_MInst_sisisi_nac_hl_s1         <"mpy",  int_hexagon_M2_mpy_nac_hl_s1>;
def HEXAGON_M2_mpy_nac_sat_hl_s1:
  si_MInst_sisisi_nac_sat_hl_s1     <"mpy",  int_hexagon_M2_mpy_nac_sat_hl_s1>;
def HEXAGON_M2_mpy_nac_sat_hl_s0:
  si_MInst_sisisi_nac_sat_hl        <"mpy",  int_hexagon_M2_mpy_nac_sat_hl_s0>;

def HEXAGON_M2_mpy_nac_lh_s0:
  si_MInst_sisisi_nac_lh            <"mpy",  int_hexagon_M2_mpy_nac_lh_s0>;
def HEXAGON_M2_mpy_nac_lh_s1:
  si_MInst_sisisi_nac_lh_s1         <"mpy",  int_hexagon_M2_mpy_nac_lh_s1>;
def HEXAGON_M2_mpy_nac_sat_lh_s1:
  si_MInst_sisisi_nac_sat_lh_s1     <"mpy",  int_hexagon_M2_mpy_nac_sat_lh_s1>;
def HEXAGON_M2_mpy_nac_sat_lh_s0:
  si_MInst_sisisi_nac_sat_lh        <"mpy",  int_hexagon_M2_mpy_nac_sat_lh_s0>;

def HEXAGON_M2_mpy_nac_ll_s0:
  si_MInst_sisisi_nac_ll            <"mpy",  int_hexagon_M2_mpy_nac_ll_s0>;
def HEXAGON_M2_mpy_nac_ll_s1:
  si_MInst_sisisi_nac_ll_s1         <"mpy",  int_hexagon_M2_mpy_nac_ll_s1>;
def HEXAGON_M2_mpy_nac_sat_ll_s1:
  si_MInst_sisisi_nac_sat_ll_s1     <"mpy",  int_hexagon_M2_mpy_nac_sat_ll_s1>;
def HEXAGON_M2_mpy_nac_sat_ll_s0:
  si_MInst_sisisi_nac_sat_ll        <"mpy",  int_hexagon_M2_mpy_nac_sat_ll_s0>;

//Rx+=mpy(Rs.[H|L],Rt.[H|L:<<0|:<<1]
def HEXAGON_M2_mpyd_acc_hh_s0:
  di_MInst_disisi_acc_hh          <"mpy",    int_hexagon_M2_mpyd_acc_hh_s0>;
def HEXAGON_M2_mpyd_acc_hh_s1:
  di_MInst_disisi_acc_hh_s1       <"mpy",    int_hexagon_M2_mpyd_acc_hh_s1>;

def HEXAGON_M2_mpyd_acc_hl_s0:
  di_MInst_disisi_acc_hl          <"mpy",    int_hexagon_M2_mpyd_acc_hl_s0>;
def HEXAGON_M2_mpyd_acc_hl_s1:
  di_MInst_disisi_acc_hl_s1       <"mpy",    int_hexagon_M2_mpyd_acc_hl_s1>;

def HEXAGON_M2_mpyd_acc_lh_s0:
  di_MInst_disisi_acc_lh          <"mpy",    int_hexagon_M2_mpyd_acc_lh_s0>;
def HEXAGON_M2_mpyd_acc_lh_s1:
  di_MInst_disisi_acc_lh_s1       <"mpy",    int_hexagon_M2_mpyd_acc_lh_s1>;

def HEXAGON_M2_mpyd_acc_ll_s0:
  di_MInst_disisi_acc_ll          <"mpy",    int_hexagon_M2_mpyd_acc_ll_s0>;
def HEXAGON_M2_mpyd_acc_ll_s1:
  di_MInst_disisi_acc_ll_s1       <"mpy",    int_hexagon_M2_mpyd_acc_ll_s1>;

//Rx-=mpy(Rs.[H|L],Rt.[H|L:<<0|:<<1]
def HEXAGON_M2_mpyd_nac_hh_s0:
  di_MInst_disisi_nac_hh          <"mpy",    int_hexagon_M2_mpyd_nac_hh_s0>;
def HEXAGON_M2_mpyd_nac_hh_s1:
  di_MInst_disisi_nac_hh_s1       <"mpy",    int_hexagon_M2_mpyd_nac_hh_s1>;

def HEXAGON_M2_mpyd_nac_hl_s0:
  di_MInst_disisi_nac_hl          <"mpy",    int_hexagon_M2_mpyd_nac_hl_s0>;
def HEXAGON_M2_mpyd_nac_hl_s1:
  di_MInst_disisi_nac_hl_s1       <"mpy",    int_hexagon_M2_mpyd_nac_hl_s1>;

def HEXAGON_M2_mpyd_nac_lh_s0:
  di_MInst_disisi_nac_lh          <"mpy",    int_hexagon_M2_mpyd_nac_lh_s0>;
def HEXAGON_M2_mpyd_nac_lh_s1:
  di_MInst_disisi_nac_lh_s1       <"mpy",    int_hexagon_M2_mpyd_nac_lh_s1>;

def HEXAGON_M2_mpyd_nac_ll_s0:
  di_MInst_disisi_nac_ll          <"mpy",    int_hexagon_M2_mpyd_nac_ll_s0>;
def HEXAGON_M2_mpyd_nac_ll_s1:
  di_MInst_disisi_nac_ll_s1       <"mpy",    int_hexagon_M2_mpyd_nac_ll_s1>;

// MTYPE / MPYS / Scalar 16x16 multiply unsigned.
//Rd=mpyu(Rs.[H|L],Rt.[H|L])[:<<0|:<<1]
def HEXAGON_M2_mpyu_hh_s0:
  si_MInst_sisi_hh                <"mpyu",    int_hexagon_M2_mpyu_hh_s0>;
def HEXAGON_M2_mpyu_hh_s1:
  si_MInst_sisi_hh_s1             <"mpyu",    int_hexagon_M2_mpyu_hh_s1>;
def HEXAGON_M2_mpyu_hl_s0:
  si_MInst_sisi_hl                <"mpyu",    int_hexagon_M2_mpyu_hl_s0>;
def HEXAGON_M2_mpyu_hl_s1:
  si_MInst_sisi_hl_s1             <"mpyu",    int_hexagon_M2_mpyu_hl_s1>;
def HEXAGON_M2_mpyu_lh_s0:
  si_MInst_sisi_lh                <"mpyu",    int_hexagon_M2_mpyu_lh_s0>;
def HEXAGON_M2_mpyu_lh_s1:
  si_MInst_sisi_lh_s1             <"mpyu",    int_hexagon_M2_mpyu_lh_s1>;
def HEXAGON_M2_mpyu_ll_s0:
  si_MInst_sisi_ll                <"mpyu",    int_hexagon_M2_mpyu_ll_s0>;
def HEXAGON_M2_mpyu_ll_s1:
  si_MInst_sisi_ll_s1             <"mpyu",    int_hexagon_M2_mpyu_ll_s1>;

//Rdd=mpyu(Rs.[H|L],Rt.[H|L])[:<<0|:<<1]
def HEXAGON_M2_mpyud_hh_s0:
  di_MInst_sisi_hh                <"mpyu",    int_hexagon_M2_mpyud_hh_s0>;
def HEXAGON_M2_mpyud_hh_s1:
  di_MInst_sisi_hh_s1             <"mpyu",    int_hexagon_M2_mpyud_hh_s1>;
def HEXAGON_M2_mpyud_hl_s0:
  di_MInst_sisi_hl                <"mpyu",    int_hexagon_M2_mpyud_hl_s0>;
def HEXAGON_M2_mpyud_hl_s1:
  di_MInst_sisi_hl_s1             <"mpyu",    int_hexagon_M2_mpyud_hl_s1>;
def HEXAGON_M2_mpyud_lh_s0:
  di_MInst_sisi_lh                <"mpyu",    int_hexagon_M2_mpyud_lh_s0>;
def HEXAGON_M2_mpyud_lh_s1:
  di_MInst_sisi_lh_s1             <"mpyu",    int_hexagon_M2_mpyud_lh_s1>;
def HEXAGON_M2_mpyud_ll_s0:
  di_MInst_sisi_ll                <"mpyu",    int_hexagon_M2_mpyud_ll_s0>;
def HEXAGON_M2_mpyud_ll_s1:
  di_MInst_sisi_ll_s1             <"mpyu",    int_hexagon_M2_mpyud_ll_s1>;

//Rd+=mpyu(Rs.[H|L],Rt.[H|L])[:<<0|:<<1]
def HEXAGON_M2_mpyu_acc_hh_s0:
  si_MInst_sisisi_acc_hh            <"mpyu",    int_hexagon_M2_mpyu_acc_hh_s0>;
def HEXAGON_M2_mpyu_acc_hh_s1:
  si_MInst_sisisi_acc_hh_s1         <"mpyu",    int_hexagon_M2_mpyu_acc_hh_s1>;
def HEXAGON_M2_mpyu_acc_hl_s0:
  si_MInst_sisisi_acc_hl            <"mpyu",    int_hexagon_M2_mpyu_acc_hl_s0>;
def HEXAGON_M2_mpyu_acc_hl_s1:
  si_MInst_sisisi_acc_hl_s1         <"mpyu",    int_hexagon_M2_mpyu_acc_hl_s1>;
def HEXAGON_M2_mpyu_acc_lh_s0:
  si_MInst_sisisi_acc_lh            <"mpyu",    int_hexagon_M2_mpyu_acc_lh_s0>;
def HEXAGON_M2_mpyu_acc_lh_s1:
  si_MInst_sisisi_acc_lh_s1         <"mpyu",    int_hexagon_M2_mpyu_acc_lh_s1>;
def HEXAGON_M2_mpyu_acc_ll_s0:
  si_MInst_sisisi_acc_ll            <"mpyu",    int_hexagon_M2_mpyu_acc_ll_s0>;
def HEXAGON_M2_mpyu_acc_ll_s1:
  si_MInst_sisisi_acc_ll_s1         <"mpyu",    int_hexagon_M2_mpyu_acc_ll_s1>;

//Rd+=mpyu(Rs.[H|L],Rt.[H|L])[:<<0|:<<1]
def HEXAGON_M2_mpyu_nac_hh_s0:
  si_MInst_sisisi_nac_hh            <"mpyu",    int_hexagon_M2_mpyu_nac_hh_s0>;
def HEXAGON_M2_mpyu_nac_hh_s1:
  si_MInst_sisisi_nac_hh_s1         <"mpyu",    int_hexagon_M2_mpyu_nac_hh_s1>;
def HEXAGON_M2_mpyu_nac_hl_s0:
  si_MInst_sisisi_nac_hl            <"mpyu",    int_hexagon_M2_mpyu_nac_hl_s0>;
def HEXAGON_M2_mpyu_nac_hl_s1:
  si_MInst_sisisi_nac_hl_s1         <"mpyu",    int_hexagon_M2_mpyu_nac_hl_s1>;
def HEXAGON_M2_mpyu_nac_lh_s0:
  si_MInst_sisisi_nac_lh            <"mpyu",    int_hexagon_M2_mpyu_nac_lh_s0>;
def HEXAGON_M2_mpyu_nac_lh_s1:
  si_MInst_sisisi_nac_lh_s1         <"mpyu",    int_hexagon_M2_mpyu_nac_lh_s1>;
def HEXAGON_M2_mpyu_nac_ll_s0:
  si_MInst_sisisi_nac_ll            <"mpyu",    int_hexagon_M2_mpyu_nac_ll_s0>;
def HEXAGON_M2_mpyu_nac_ll_s1:
  si_MInst_sisisi_nac_ll_s1         <"mpyu",    int_hexagon_M2_mpyu_nac_ll_s1>;

//Rdd+=mpyu(Rs.[H|L],Rt.[H|L])[:<<0|:<<1]
def HEXAGON_M2_mpyud_acc_hh_s0:
  di_MInst_disisi_acc_hh            <"mpyu", int_hexagon_M2_mpyud_acc_hh_s0>;
def HEXAGON_M2_mpyud_acc_hh_s1:
  di_MInst_disisi_acc_hh_s1         <"mpyu", int_hexagon_M2_mpyud_acc_hh_s1>;
def HEXAGON_M2_mpyud_acc_hl_s0:
  di_MInst_disisi_acc_hl            <"mpyu", int_hexagon_M2_mpyud_acc_hl_s0>;
def HEXAGON_M2_mpyud_acc_hl_s1:
  di_MInst_disisi_acc_hl_s1         <"mpyu", int_hexagon_M2_mpyud_acc_hl_s1>;
def HEXAGON_M2_mpyud_acc_lh_s0:
  di_MInst_disisi_acc_lh            <"mpyu", int_hexagon_M2_mpyud_acc_lh_s0>;
def HEXAGON_M2_mpyud_acc_lh_s1:
  di_MInst_disisi_acc_lh_s1         <"mpyu", int_hexagon_M2_mpyud_acc_lh_s1>;
def HEXAGON_M2_mpyud_acc_ll_s0:
  di_MInst_disisi_acc_ll            <"mpyu", int_hexagon_M2_mpyud_acc_ll_s0>;
def HEXAGON_M2_mpyud_acc_ll_s1:
  di_MInst_disisi_acc_ll_s1         <"mpyu", int_hexagon_M2_mpyud_acc_ll_s1>;

//Rdd-=mpyu(Rs.[H|L],Rt.[H|L])[:<<0|:<<1]
def HEXAGON_M2_mpyud_nac_hh_s0:
  di_MInst_disisi_nac_hh            <"mpyu", int_hexagon_M2_mpyud_nac_hh_s0>;
def HEXAGON_M2_mpyud_nac_hh_s1:
  di_MInst_disisi_nac_hh_s1         <"mpyu", int_hexagon_M2_mpyud_nac_hh_s1>;
def HEXAGON_M2_mpyud_nac_hl_s0:
  di_MInst_disisi_nac_hl            <"mpyu", int_hexagon_M2_mpyud_nac_hl_s0>;
def HEXAGON_M2_mpyud_nac_hl_s1:
  di_MInst_disisi_nac_hl_s1         <"mpyu", int_hexagon_M2_mpyud_nac_hl_s1>;
def HEXAGON_M2_mpyud_nac_lh_s0:
  di_MInst_disisi_nac_lh            <"mpyu", int_hexagon_M2_mpyud_nac_lh_s0>;
def HEXAGON_M2_mpyud_nac_lh_s1:
  di_MInst_disisi_nac_lh_s1         <"mpyu", int_hexagon_M2_mpyud_nac_lh_s1>;
def HEXAGON_M2_mpyud_nac_ll_s0:
  di_MInst_disisi_nac_ll            <"mpyu", int_hexagon_M2_mpyud_nac_ll_s0>;
def HEXAGON_M2_mpyud_nac_ll_s1:
  di_MInst_disisi_nac_ll_s1         <"mpyu", int_hexagon_M2_mpyud_nac_ll_s1>;


/********************************************************************
*            MTYPE/VB                                               *
*********************************************************************/

// MTYPE / VB / Vector reduce add unsigned bytes.
def HEXAGON_A2_vraddub:
  di_MInst_didi                   <"vraddub", int_hexagon_A2_vraddub>;
def HEXAGON_A2_vraddub_acc:
  di_MInst_dididi_acc             <"vraddub", int_hexagon_A2_vraddub_acc>;

// MTYPE / VB / Vector sum of absolute differences unsigned bytes.
def HEXAGON_A2_vrsadub:
  di_MInst_didi                   <"vrsadub", int_hexagon_A2_vrsadub>;
def HEXAGON_A2_vrsadub_acc:
  di_MInst_dididi_acc             <"vrsadub", int_hexagon_A2_vrsadub_acc>;

/********************************************************************
*            MTYPE/VH                                               *
*********************************************************************/

// MTYPE / VH / Vector dual multiply.
def HEXAGON_M2_vdmpys_s1:
  di_MInst_didi_s1_sat            <"vdmpy",   int_hexagon_M2_vdmpys_s1>;
def HEXAGON_M2_vdmpys_s0:
  di_MInst_didi_sat               <"vdmpy",   int_hexagon_M2_vdmpys_s0>;
def HEXAGON_M2_vdmacs_s1:
  di_MInst_dididi_acc_s1_sat      <"vdmpy",   int_hexagon_M2_vdmacs_s1>;
def HEXAGON_M2_vdmacs_s0:
  di_MInst_dididi_acc_sat         <"vdmpy",   int_hexagon_M2_vdmacs_s0>;

// MTYPE / VH / Vector dual multiply with round and pack.
def HEXAGON_M2_vdmpyrs_s0:
  si_MInst_didi_rnd_sat           <"vdmpy",   int_hexagon_M2_vdmpyrs_s0>;
def HEXAGON_M2_vdmpyrs_s1:
  si_MInst_didi_s1_rnd_sat        <"vdmpy",   int_hexagon_M2_vdmpyrs_s1>;

// MTYPE / VH / Vector multiply even halfwords.
def HEXAGON_M2_vmpy2es_s1:
  di_MInst_didi_s1_sat            <"vmpyeh",  int_hexagon_M2_vmpy2es_s1>;
def HEXAGON_M2_vmpy2es_s0:
  di_MInst_didi_sat               <"vmpyeh",  int_hexagon_M2_vmpy2es_s0>;
def HEXAGON_M2_vmac2es:
  di_MInst_dididi_acc             <"vmpyeh",  int_hexagon_M2_vmac2es>;
def HEXAGON_M2_vmac2es_s1:
  di_MInst_dididi_acc_s1_sat      <"vmpyeh",  int_hexagon_M2_vmac2es_s1>;
def HEXAGON_M2_vmac2es_s0:
  di_MInst_dididi_acc_sat         <"vmpyeh",  int_hexagon_M2_vmac2es_s0>;

// MTYPE / VH / Vector multiply halfwords.
def HEXAGON_M2_vmpy2s_s0:
  di_MInst_sisi_sat               <"vmpyh",   int_hexagon_M2_vmpy2s_s0>;
def HEXAGON_M2_vmpy2s_s1:
  di_MInst_sisi_s1_sat            <"vmpyh",   int_hexagon_M2_vmpy2s_s1>;
def HEXAGON_M2_vmac2:
  di_MInst_disisi_acc             <"vmpyh",   int_hexagon_M2_vmac2>;
def HEXAGON_M2_vmac2s_s0:
  di_MInst_disisi_acc_sat         <"vmpyh",   int_hexagon_M2_vmac2s_s0>;
def HEXAGON_M2_vmac2s_s1:
  di_MInst_disisi_acc_s1_sat      <"vmpyh",   int_hexagon_M2_vmac2s_s1>;

// MTYPE / VH / Vector multiply halfwords with round and pack.
def HEXAGON_M2_vmpy2s_s0pack:
  si_MInst_sisi_rnd_sat           <"vmpyh",   int_hexagon_M2_vmpy2s_s0pack>;
def HEXAGON_M2_vmpy2s_s1pack:
  si_MInst_sisi_s1_rnd_sat        <"vmpyh",   int_hexagon_M2_vmpy2s_s1pack>;

// MTYPE / VH / Vector reduce multiply halfwords.
// Rxx32+=vrmpyh(Rss32,Rtt32)
def HEXAGON_M2_vrmpy_s0:
  di_MInst_didi                   <"vrmpyh",  int_hexagon_M2_vrmpy_s0>;
def HEXAGON_M2_vrmac_s0:
  di_MInst_dididi_acc             <"vrmpyh",  int_hexagon_M2_vrmac_s0>;


/********************************************************************
*            STYPE/ALU                                              *
*********************************************************************/

// STYPE / ALU / Absolute value.
def HEXAGON_A2_abs:
  si_SInst_si                     <"abs",     int_hexagon_A2_abs>;
def HEXAGON_A2_absp:
  di_SInst_di                     <"abs",     int_hexagon_A2_absp>;
def HEXAGON_A2_abssat:
  si_SInst_si_sat                 <"abs",     int_hexagon_A2_abssat>;

// STYPE / ALU / Negate.
def HEXAGON_A2_negp:
  di_SInst_di                     <"neg",     int_hexagon_A2_negp>;
def HEXAGON_A2_negsat:
  si_SInst_si_sat                 <"neg",     int_hexagon_A2_negsat>;

// STYPE / ALU / Logical Not.
def HEXAGON_A2_notp:
  di_SInst_di                     <"not",     int_hexagon_A2_notp>;

// STYPE / ALU / Sign extend word to doubleword.
def HEXAGON_A2_sxtw:
  di_SInst_si                     <"sxtw",     int_hexagon_A2_sxtw>;


/********************************************************************
*            STYPE/BIT                                              *
*********************************************************************/

// STYPE / BIT / Count leading.
def HEXAGON_S2_cl0:
  si_SInst_si                     <"cl0",     int_hexagon_S2_cl0>;
def HEXAGON_S2_cl0p:
  si_SInst_di                     <"cl0",     int_hexagon_S2_cl0p>;
def HEXAGON_S2_cl1:
  si_SInst_si                     <"cl1",     int_hexagon_S2_cl1>;
def HEXAGON_S2_cl1p:
  si_SInst_di                     <"cl1",     int_hexagon_S2_cl1p>;
def HEXAGON_S2_clb:
  si_SInst_si                     <"clb",     int_hexagon_S2_clb>;
def HEXAGON_S2_clbp:
  si_SInst_di                     <"clb",     int_hexagon_S2_clbp>;
def HEXAGON_S2_clbnorm:
  si_SInst_si                     <"normamt", int_hexagon_S2_clbnorm>;

// STYPE / BIT / Count trailing.
def HEXAGON_S2_ct0:
  si_SInst_si                     <"ct0",     int_hexagon_S2_ct0>;
def HEXAGON_S2_ct1:
  si_SInst_si                     <"ct1",     int_hexagon_S2_ct1>;

// STYPE / BIT / Compare bit mask.
def Hexagon_C2_bitsclr:
  qi_SInst_sisi                   <"bitsclr", int_hexagon_C2_bitsclr>;
def Hexagon_C2_bitsclri:
  qi_SInst_siu6                   <"bitsclr", int_hexagon_C2_bitsclri>;
def Hexagon_C2_bitsset:
  qi_SInst_sisi                   <"bitsset", int_hexagon_C2_bitsset>;

// STYPE / BIT / Extract unsigned.
// Rd[d][32/64]=extractu(Rs[s],Rt[t],[imm])
def HEXAGON_S2_extractu:
  si_SInst_siu5u5                 <"extractu",int_hexagon_S2_extractu>;
def HEXAGON_S2_extractu_rp:
  si_SInst_sidi                   <"extractu",int_hexagon_S2_extractu_rp>;
def HEXAGON_S2_extractup:
  di_SInst_diu6u6                 <"extractu",int_hexagon_S2_extractup>;
def HEXAGON_S2_extractup_rp:
  di_SInst_didi                   <"extractu",int_hexagon_S2_extractup_rp>;

// STYPE / BIT / Insert bitfield.
def Hexagon_S2_insert:
  si_SInst_sisiu5u5               <"insert",  int_hexagon_S2_insert>;
def Hexagon_S2_insert_rp:
  si_SInst_sisidi                 <"insert",  int_hexagon_S2_insert_rp>;
def Hexagon_S2_insertp:
  di_SInst_didiu6u6               <"insert",  int_hexagon_S2_insertp>;
def Hexagon_S2_insertp_rp:
  di_SInst_dididi                 <"insert",  int_hexagon_S2_insertp_rp>;

// STYPE / BIT / Innterleave/deinterleave.
def Hexagon_S2_interleave:
  di_SInst_di                     <"interleave", int_hexagon_S2_interleave>;
def Hexagon_S2_deinterleave:
  di_SInst_di                     <"deinterleave", int_hexagon_S2_deinterleave>;

// STYPE / BIT / Linear feedback-shift Iteration.
def Hexagon_S2_lfsp:
  di_SInst_didi                   <"lfs",     int_hexagon_S2_lfsp>;

// STYPE / BIT / Bit reverse.
def Hexagon_S2_brev:
  si_SInst_si                     <"brev",    int_hexagon_S2_brev>;

// STYPE / BIT / Set/Clear/Toggle Bit.
def HEXAGON_S2_setbit_i:
  si_SInst_siu5                   <"setbit",  int_hexagon_S2_setbit_i>;
def HEXAGON_S2_togglebit_i:
  si_SInst_siu5                   <"togglebit", int_hexagon_S2_togglebit_i>;
def HEXAGON_S2_clrbit_i:
  si_SInst_siu5                   <"clrbit",  int_hexagon_S2_clrbit_i>;
def HEXAGON_S2_setbit_r:
  si_SInst_sisi                   <"setbit",  int_hexagon_S2_setbit_r>;
def HEXAGON_S2_togglebit_r:
  si_SInst_sisi                   <"togglebit", int_hexagon_S2_togglebit_r>;
def HEXAGON_S2_clrbit_r:
  si_SInst_sisi                   <"clrbit",  int_hexagon_S2_clrbit_r>;

// STYPE / BIT / Test Bit.
def HEXAGON_S2_tstbit_i:
  qi_SInst_siu5                   <"tstbit",  int_hexagon_S2_tstbit_i>;
def HEXAGON_S2_tstbit_r:
  qi_SInst_sisi                   <"tstbit",  int_hexagon_S2_tstbit_r>;


/********************************************************************
*            STYPE/COMPLEX                                          *
*********************************************************************/

// STYPE / COMPLEX / Vector Complex conjugate.
def HEXAGON_A2_vconj:
  di_SInst_di_sat                 <"vconj",   int_hexagon_A2_vconj>;

// STYPE / COMPLEX / Vector Complex rotate.
def HEXAGON_S2_vcrotate:
  di_SInst_disi                   <"vcrotate",int_hexagon_S2_vcrotate>;


/********************************************************************
*            STYPE/PERM                                             *
*********************************************************************/

// STYPE / PERM / Saturate.
def HEXAGON_A2_sat:
  si_SInst_di                     <"sat",     int_hexagon_A2_sat>;
def HEXAGON_A2_satb:
  si_SInst_si                     <"satb",    int_hexagon_A2_satb>;
def HEXAGON_A2_sath:
  si_SInst_si                     <"sath",    int_hexagon_A2_sath>;
def HEXAGON_A2_satub:
  si_SInst_si                     <"satub",   int_hexagon_A2_satub>;
def HEXAGON_A2_satuh:
  si_SInst_si                     <"satuh",   int_hexagon_A2_satuh>;

// STYPE / PERM / Swizzle bytes.
def HEXAGON_A2_swiz:
  si_SInst_si                     <"swiz",    int_hexagon_A2_swiz>;

// STYPE / PERM / Vector align.
// Need custom lowering
def HEXAGON_S2_valignib:
  di_SInst_didiu3                 <"valignb", int_hexagon_S2_valignib>;
def HEXAGON_S2_valignrb:
  di_SInst_didiqi                 <"valignb", int_hexagon_S2_valignrb>;

// STYPE / PERM / Vector round and pack.
def HEXAGON_S2_vrndpackwh:
  si_SInst_di                     <"vrndwh",  int_hexagon_S2_vrndpackwh>;
def HEXAGON_S2_vrndpackwhs:
  si_SInst_di_sat                 <"vrndwh",  int_hexagon_S2_vrndpackwhs>;

// STYPE / PERM / Vector saturate and pack.
def HEXAGON_S2_svsathb:
  si_SInst_si                     <"vsathb",  int_hexagon_S2_svsathb>;
def HEXAGON_S2_vsathb:
  si_SInst_di                     <"vsathb",  int_hexagon_S2_vsathb>;
def HEXAGON_S2_svsathub:
  si_SInst_si                     <"vsathub", int_hexagon_S2_svsathub>;
def HEXAGON_S2_vsathub:
  si_SInst_di                     <"vsathub", int_hexagon_S2_vsathub>;
def HEXAGON_S2_vsatwh:
  si_SInst_di                     <"vsatwh",  int_hexagon_S2_vsatwh>;
def HEXAGON_S2_vsatwuh:
  si_SInst_di                     <"vsatwuh", int_hexagon_S2_vsatwuh>;

// STYPE / PERM / Vector saturate without pack.
def HEXAGON_S2_vsathb_nopack:
  di_SInst_di                     <"vsathb",  int_hexagon_S2_vsathb_nopack>;
def HEXAGON_S2_vsathub_nopack:
  di_SInst_di                     <"vsathub", int_hexagon_S2_vsathub_nopack>;
def HEXAGON_S2_vsatwh_nopack:
  di_SInst_di                     <"vsatwh",  int_hexagon_S2_vsatwh_nopack>;
def HEXAGON_S2_vsatwuh_nopack:
  di_SInst_di                     <"vsatwuh", int_hexagon_S2_vsatwuh_nopack>;

// STYPE / PERM / Vector shuffle.
def HEXAGON_S2_shuffeb:
  di_SInst_didi                   <"shuffeb", int_hexagon_S2_shuffeb>;
def HEXAGON_S2_shuffeh:
  di_SInst_didi                   <"shuffeh", int_hexagon_S2_shuffeh>;
def HEXAGON_S2_shuffob:
  di_SInst_didi                   <"shuffob", int_hexagon_S2_shuffob>;
def HEXAGON_S2_shuffoh:
  di_SInst_didi                   <"shuffoh", int_hexagon_S2_shuffoh>;

// STYPE / PERM / Vector splat bytes.
def HEXAGON_S2_vsplatrb:
  si_SInst_si                     <"vsplatb", int_hexagon_S2_vsplatrb>;

// STYPE / PERM / Vector splat halfwords.
def HEXAGON_S2_vsplatrh:
  di_SInst_si                     <"vsplath", int_hexagon_S2_vsplatrh>;

// STYPE / PERM / Vector splice.
def Hexagon_S2_vsplicerb:
  di_SInst_didiqi                 <"vspliceb",int_hexagon_S2_vsplicerb>;
def Hexagon_S2_vspliceib:
  di_SInst_didiu3                 <"vspliceb",int_hexagon_S2_vspliceib>;

// STYPE / PERM / Sign extend.
def HEXAGON_S2_vsxtbh:
  di_SInst_si                     <"vsxtbh",  int_hexagon_S2_vsxtbh>;
def HEXAGON_S2_vsxthw:
  di_SInst_si                     <"vsxthw",  int_hexagon_S2_vsxthw>;

// STYPE / PERM / Truncate.
def HEXAGON_S2_vtrunehb:
  si_SInst_di                     <"vtrunehb",int_hexagon_S2_vtrunehb>;
def HEXAGON_S2_vtrunohb:
  si_SInst_di                     <"vtrunohb",int_hexagon_S2_vtrunohb>;
def HEXAGON_S2_vtrunewh:
  di_SInst_didi                   <"vtrunewh",int_hexagon_S2_vtrunewh>;
def HEXAGON_S2_vtrunowh:
  di_SInst_didi                   <"vtrunowh",int_hexagon_S2_vtrunowh>;

// STYPE / PERM / Zero extend.
def HEXAGON_S2_vzxtbh:
  di_SInst_si                     <"vzxtbh",  int_hexagon_S2_vzxtbh>;
def HEXAGON_S2_vzxthw:
  di_SInst_si                     <"vzxthw",  int_hexagon_S2_vzxthw>;


/********************************************************************
*            STYPE/PRED                                             *
*********************************************************************/

// STYPE / PRED / Mask generate from predicate.
def HEXAGON_C2_mask:
  di_SInst_qi                     <"mask",   int_hexagon_C2_mask>;

// STYPE / PRED / Predicate transfer.
def HEXAGON_C2_tfrpr:
  si_SInst_qi                     <"",       int_hexagon_C2_tfrpr>;
def HEXAGON_C2_tfrrp:
  qi_SInst_si                     <"",       int_hexagon_C2_tfrrp>;

// STYPE / PRED / Viterbi pack even and odd predicate bits.
def HEXAGON_C2_vitpack:
  si_SInst_qiqi                   <"vitpack",int_hexagon_C2_vitpack>;


/********************************************************************
*            STYPE/SHIFT                                            *
*********************************************************************/

// STYPE / SHIFT / Shift by immediate.
def HEXAGON_S2_asl_i_r:
  si_SInst_siu5                   <"asl",     int_hexagon_S2_asl_i_r>;
def HEXAGON_S2_asr_i_r:
  si_SInst_siu5                   <"asr",     int_hexagon_S2_asr_i_r>;
def HEXAGON_S2_lsr_i_r:
  si_SInst_siu5                   <"lsr",     int_hexagon_S2_lsr_i_r>;
def HEXAGON_S2_asl_i_p:
  di_SInst_diu6                   <"asl",     int_hexagon_S2_asl_i_p>;
def HEXAGON_S2_asr_i_p:
  di_SInst_diu6                   <"asr",     int_hexagon_S2_asr_i_p>;
def HEXAGON_S2_lsr_i_p:
  di_SInst_diu6                   <"lsr",     int_hexagon_S2_lsr_i_p>;

// STYPE / SHIFT / Shift by immediate and accumulate.
def HEXAGON_S2_asl_i_r_acc:
  si_SInst_sisiu5_acc             <"asl",     int_hexagon_S2_asl_i_r_acc>;
def HEXAGON_S2_asr_i_r_acc:
  si_SInst_sisiu5_acc             <"asr",     int_hexagon_S2_asr_i_r_acc>;
def HEXAGON_S2_lsr_i_r_acc:
  si_SInst_sisiu5_acc             <"lsr",     int_hexagon_S2_lsr_i_r_acc>;
def HEXAGON_S2_asl_i_r_nac:
  si_SInst_sisiu5_nac             <"asl",     int_hexagon_S2_asl_i_r_nac>;
def HEXAGON_S2_asr_i_r_nac:
  si_SInst_sisiu5_nac             <"asr",     int_hexagon_S2_asr_i_r_nac>;
def HEXAGON_S2_lsr_i_r_nac:
  si_SInst_sisiu5_nac             <"lsr",     int_hexagon_S2_lsr_i_r_nac>;
def HEXAGON_S2_asl_i_p_acc:
  di_SInst_didiu6_acc             <"asl",     int_hexagon_S2_asl_i_p_acc>;
def HEXAGON_S2_asr_i_p_acc:
  di_SInst_didiu6_acc             <"asr",     int_hexagon_S2_asr_i_p_acc>;
def HEXAGON_S2_lsr_i_p_acc:
  di_SInst_didiu6_acc             <"lsr",     int_hexagon_S2_lsr_i_p_acc>;
def HEXAGON_S2_asl_i_p_nac:
  di_SInst_didiu6_nac             <"asl",     int_hexagon_S2_asl_i_p_nac>;
def HEXAGON_S2_asr_i_p_nac:
  di_SInst_didiu6_nac             <"asr",     int_hexagon_S2_asr_i_p_nac>;
def HEXAGON_S2_lsr_i_p_nac:
  di_SInst_didiu6_nac             <"lsr",     int_hexagon_S2_lsr_i_p_nac>;

// STYPE / SHIFT / Shift by immediate and add.
def HEXAGON_S2_addasl_rrri:
  si_SInst_sisiu3                 <"addasl",  int_hexagon_S2_addasl_rrri>;

// STYPE / SHIFT / Shift by immediate and logical.
def HEXAGON_S2_asl_i_r_and:
  si_SInst_sisiu5_and             <"asl",     int_hexagon_S2_asl_i_r_and>;
def HEXAGON_S2_asr_i_r_and:
  si_SInst_sisiu5_and             <"asr",     int_hexagon_S2_asr_i_r_and>;
def HEXAGON_S2_lsr_i_r_and:
  si_SInst_sisiu5_and             <"lsr",     int_hexagon_S2_lsr_i_r_and>;

def HEXAGON_S2_asl_i_r_xacc:
  si_SInst_sisiu5_xor             <"asl",     int_hexagon_S2_asl_i_r_xacc>;
def HEXAGON_S2_lsr_i_r_xacc:
  si_SInst_sisiu5_xor             <"lsr",     int_hexagon_S2_lsr_i_r_xacc>;

def HEXAGON_S2_asl_i_r_or:
  si_SInst_sisiu5_or              <"asl",     int_hexagon_S2_asl_i_r_or>;
def HEXAGON_S2_asr_i_r_or:
  si_SInst_sisiu5_or              <"asr",     int_hexagon_S2_asr_i_r_or>;
def HEXAGON_S2_lsr_i_r_or:
  si_SInst_sisiu5_or              <"lsr",     int_hexagon_S2_lsr_i_r_or>;

def HEXAGON_S2_asl_i_p_and:
  di_SInst_didiu6_and             <"asl",     int_hexagon_S2_asl_i_p_and>;
def HEXAGON_S2_asr_i_p_and:
  di_SInst_didiu6_and             <"asr",     int_hexagon_S2_asr_i_p_and>;
def HEXAGON_S2_lsr_i_p_and:
  di_SInst_didiu6_and             <"lsr",     int_hexagon_S2_lsr_i_p_and>;

def HEXAGON_S2_asl_i_p_xacc:
  di_SInst_didiu6_xor             <"asl",     int_hexagon_S2_asl_i_p_xacc>;
def HEXAGON_S2_lsr_i_p_xacc:
  di_SInst_didiu6_xor             <"lsr",     int_hexagon_S2_lsr_i_p_xacc>;

def HEXAGON_S2_asl_i_p_or:
  di_SInst_didiu6_or              <"asl",     int_hexagon_S2_asl_i_p_or>;
def HEXAGON_S2_asr_i_p_or:
  di_SInst_didiu6_or              <"asr",     int_hexagon_S2_asr_i_p_or>;
def HEXAGON_S2_lsr_i_p_or:
  di_SInst_didiu6_or              <"lsr",     int_hexagon_S2_lsr_i_p_or>;

// STYPE / SHIFT / Shift right by immediate with rounding.
def HEXAGON_S2_asr_i_r_rnd:
  si_SInst_siu5_rnd               <"asr",     int_hexagon_S2_asr_i_r_rnd>;
def HEXAGON_S2_asr_i_r_rnd_goodsyntax:
  si_SInst_siu5              <"asrrnd",  int_hexagon_S2_asr_i_r_rnd_goodsyntax>;

// STYPE / SHIFT / Shift left by immediate with saturation.
def HEXAGON_S2_asl_i_r_sat:
  si_SInst_sisi_sat               <"asl",     int_hexagon_S2_asl_i_r_sat>;

// STYPE / SHIFT / Shift by register.
def HEXAGON_S2_asl_r_r:
  si_SInst_sisi                   <"asl",     int_hexagon_S2_asl_r_r>;
def HEXAGON_S2_asr_r_r:
  si_SInst_sisi                   <"asr",     int_hexagon_S2_asr_r_r>;
def HEXAGON_S2_lsl_r_r:
  si_SInst_sisi                   <"lsl",     int_hexagon_S2_lsl_r_r>;
def HEXAGON_S2_lsr_r_r:
  si_SInst_sisi                   <"lsr",     int_hexagon_S2_lsr_r_r>;
def HEXAGON_S2_asl_r_p:
  di_SInst_disi                   <"asl",     int_hexagon_S2_asl_r_p>;
def HEXAGON_S2_asr_r_p:
  di_SInst_disi                   <"asr",     int_hexagon_S2_asr_r_p>;
def HEXAGON_S2_lsl_r_p:
  di_SInst_disi                   <"lsl",     int_hexagon_S2_lsl_r_p>;
def HEXAGON_S2_lsr_r_p:
  di_SInst_disi                   <"lsr",     int_hexagon_S2_lsr_r_p>;

// STYPE / SHIFT / Shift by register and accumulate.
def HEXAGON_S2_asl_r_r_acc:
  si_SInst_sisisi_acc             <"asl",     int_hexagon_S2_asl_r_r_acc>;
def HEXAGON_S2_asr_r_r_acc:
  si_SInst_sisisi_acc             <"asr",     int_hexagon_S2_asr_r_r_acc>;
def HEXAGON_S2_lsl_r_r_acc:
  si_SInst_sisisi_acc             <"lsl",     int_hexagon_S2_lsl_r_r_acc>;
def HEXAGON_S2_lsr_r_r_acc:
  si_SInst_sisisi_acc             <"lsr",     int_hexagon_S2_lsr_r_r_acc>;
def HEXAGON_S2_asl_r_p_acc:
  di_SInst_didisi_acc             <"asl",     int_hexagon_S2_asl_r_p_acc>;
def HEXAGON_S2_asr_r_p_acc:
  di_SInst_didisi_acc             <"asr",     int_hexagon_S2_asr_r_p_acc>;
def HEXAGON_S2_lsl_r_p_acc:
  di_SInst_didisi_acc             <"lsl",     int_hexagon_S2_lsl_r_p_acc>;
def HEXAGON_S2_lsr_r_p_acc:
  di_SInst_didisi_acc             <"lsr",     int_hexagon_S2_lsr_r_p_acc>;

def HEXAGON_S2_asl_r_r_nac:
  si_SInst_sisisi_nac             <"asl",     int_hexagon_S2_asl_r_r_nac>;
def HEXAGON_S2_asr_r_r_nac:
  si_SInst_sisisi_nac             <"asr",     int_hexagon_S2_asr_r_r_nac>;
def HEXAGON_S2_lsl_r_r_nac:
  si_SInst_sisisi_nac             <"lsl",     int_hexagon_S2_lsl_r_r_nac>;
def HEXAGON_S2_lsr_r_r_nac:
  si_SInst_sisisi_nac             <"lsr",     int_hexagon_S2_lsr_r_r_nac>;
def HEXAGON_S2_asl_r_p_nac:
  di_SInst_didisi_nac             <"asl",     int_hexagon_S2_asl_r_p_nac>;
def HEXAGON_S2_asr_r_p_nac:
  di_SInst_didisi_nac             <"asr",     int_hexagon_S2_asr_r_p_nac>;
def HEXAGON_S2_lsl_r_p_nac:
  di_SInst_didisi_nac             <"lsl",     int_hexagon_S2_lsl_r_p_nac>;
def HEXAGON_S2_lsr_r_p_nac:
  di_SInst_didisi_nac             <"lsr",     int_hexagon_S2_lsr_r_p_nac>;

// STYPE / SHIFT / Shift by register and logical.
def HEXAGON_S2_asl_r_r_and:
  si_SInst_sisisi_and             <"asl",     int_hexagon_S2_asl_r_r_and>;
def HEXAGON_S2_asr_r_r_and:
  si_SInst_sisisi_and             <"asr",     int_hexagon_S2_asr_r_r_and>;
def HEXAGON_S2_lsl_r_r_and:
  si_SInst_sisisi_and             <"lsl",     int_hexagon_S2_lsl_r_r_and>;
def HEXAGON_S2_lsr_r_r_and:
  si_SInst_sisisi_and             <"lsr",     int_hexagon_S2_lsr_r_r_and>;

def HEXAGON_S2_asl_r_r_or:
  si_SInst_sisisi_or              <"asl",     int_hexagon_S2_asl_r_r_or>;
def HEXAGON_S2_asr_r_r_or:
  si_SInst_sisisi_or              <"asr",     int_hexagon_S2_asr_r_r_or>;
def HEXAGON_S2_lsl_r_r_or:
  si_SInst_sisisi_or              <"lsl",     int_hexagon_S2_lsl_r_r_or>;
def HEXAGON_S2_lsr_r_r_or:
  si_SInst_sisisi_or              <"lsr",     int_hexagon_S2_lsr_r_r_or>;

def HEXAGON_S2_asl_r_p_and:
  di_SInst_didisi_and             <"asl",     int_hexagon_S2_asl_r_p_and>;
def HEXAGON_S2_asr_r_p_and:
  di_SInst_didisi_and             <"asr",     int_hexagon_S2_asr_r_p_and>;
def HEXAGON_S2_lsl_r_p_and:
  di_SInst_didisi_and             <"lsl",     int_hexagon_S2_lsl_r_p_and>;
def HEXAGON_S2_lsr_r_p_and:
  di_SInst_didisi_and             <"lsr",     int_hexagon_S2_lsr_r_p_and>;

def HEXAGON_S2_asl_r_p_or:
  di_SInst_didisi_or              <"asl",     int_hexagon_S2_asl_r_p_or>;
def HEXAGON_S2_asr_r_p_or:
  di_SInst_didisi_or              <"asr",     int_hexagon_S2_asr_r_p_or>;
def HEXAGON_S2_lsl_r_p_or:
  di_SInst_didisi_or              <"lsl",     int_hexagon_S2_lsl_r_p_or>;
def HEXAGON_S2_lsr_r_p_or:
  di_SInst_didisi_or              <"lsr",     int_hexagon_S2_lsr_r_p_or>;

// STYPE / SHIFT / Shift by register with saturation.
def HEXAGON_S2_asl_r_r_sat:
  si_SInst_sisi_sat               <"asl",     int_hexagon_S2_asl_r_r_sat>;
def HEXAGON_S2_asr_r_r_sat:
  si_SInst_sisi_sat               <"asr",     int_hexagon_S2_asr_r_r_sat>;

// STYPE / SHIFT / Table Index.
def Hexagon_S2_tableidxb_goodsyntax:
  si_MInst_sisiu4u5          <"tableidxb",int_hexagon_S2_tableidxb_goodsyntax>;
def Hexagon_S2_tableidxd_goodsyntax:
  si_MInst_sisiu4u5          <"tableidxd",int_hexagon_S2_tableidxd_goodsyntax>;
def Hexagon_S2_tableidxh_goodsyntax:
  si_MInst_sisiu4u5          <"tableidxh",int_hexagon_S2_tableidxh_goodsyntax>;
def Hexagon_S2_tableidxw_goodsyntax:
  si_MInst_sisiu4u5          <"tableidxw",int_hexagon_S2_tableidxw_goodsyntax>;


/********************************************************************
*            STYPE/VH                                               *
*********************************************************************/

// STYPE / VH / Vector absolute value halfwords.
// Rdd64=vabsh(Rss64)
def HEXAGON_A2_vabsh:
  di_SInst_di                     <"vabsh",   int_hexagon_A2_vabsh>;
def HEXAGON_A2_vabshsat:
  di_SInst_di_sat                 <"vabsh",   int_hexagon_A2_vabshsat>;

// STYPE / VH / Vector shift halfwords by immediate.
// Rdd64=v[asl/asr/lsr]h(Rss64,Rt32)
def HEXAGON_S2_asl_i_vh:
  di_SInst_disi                   <"vaslh",   int_hexagon_S2_asl_i_vh>;
def HEXAGON_S2_asr_i_vh:
  di_SInst_disi                   <"vasrh",   int_hexagon_S2_asr_i_vh>;
def HEXAGON_S2_lsr_i_vh:
  di_SInst_disi                   <"vlsrh",   int_hexagon_S2_lsr_i_vh>;

// STYPE / VH / Vector shift halfwords by register.
// Rdd64=v[asl/asr/lsl/lsr]w(Rss64,Rt32)
def HEXAGON_S2_asl_r_vh:
  di_SInst_disi                   <"vaslh",   int_hexagon_S2_asl_r_vh>;
def HEXAGON_S2_asr_r_vh:
  di_SInst_disi                   <"vasrh",   int_hexagon_S2_asr_r_vh>;
def HEXAGON_S2_lsl_r_vh:
  di_SInst_disi                   <"vlslh",   int_hexagon_S2_lsl_r_vh>;
def HEXAGON_S2_lsr_r_vh:
  di_SInst_disi                   <"vlsrh",   int_hexagon_S2_lsr_r_vh>;


/********************************************************************
*            STYPE/VW                                               *
*********************************************************************/

// STYPE / VW / Vector absolute value words.
def HEXAGON_A2_vabsw:
  di_SInst_di                     <"vabsw",   int_hexagon_A2_vabsw>;
def HEXAGON_A2_vabswsat:
  di_SInst_di_sat                 <"vabsw",   int_hexagon_A2_vabswsat>;

// STYPE / VW / Vector shift words by immediate.
// Rdd64=v[asl/vsl]w(Rss64,Rt32)
def HEXAGON_S2_asl_i_vw:
  di_SInst_disi                   <"vaslw",   int_hexagon_S2_asl_i_vw>;
def HEXAGON_S2_asr_i_vw:
  di_SInst_disi                   <"vasrw",   int_hexagon_S2_asr_i_vw>;
def HEXAGON_S2_lsr_i_vw:
  di_SInst_disi                   <"vlsrw",   int_hexagon_S2_lsr_i_vw>;

// STYPE / VW / Vector shift words by register.
// Rdd64=v[asl/vsl]w(Rss64,Rt32)
def HEXAGON_S2_asl_r_vw:
  di_SInst_disi                   <"vaslw",   int_hexagon_S2_asl_r_vw>;
def HEXAGON_S2_asr_r_vw:
  di_SInst_disi                   <"vasrw",   int_hexagon_S2_asr_r_vw>;
def HEXAGON_S2_lsl_r_vw:
  di_SInst_disi                   <"vlslw",   int_hexagon_S2_lsl_r_vw>;
def HEXAGON_S2_lsr_r_vw:
  di_SInst_disi                   <"vlsrw",   int_hexagon_S2_lsr_r_vw>;

// STYPE / VW / Vector shift words with truncate and pack.
def HEXAGON_S2_asr_r_svw_trun:
  si_SInst_disi                   <"vasrw",   int_hexagon_S2_asr_r_svw_trun>;
def HEXAGON_S2_asr_i_svw_trun:
  si_SInst_diu5                   <"vasrw",   int_hexagon_S2_asr_i_svw_trun>;

// LD / Circular loads.
def HEXAGON_circ_ldd:
  di_LDInstPI_diu4                <"circ_ldd", int_hexagon_circ_ldd>;

include "HexagonIntrinsicsV3.td"
include "HexagonIntrinsicsV4.td"
include "HexagonIntrinsicsV5.td"

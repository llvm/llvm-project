//===-- X86InstrAVX10.td - AVX10 Instruction Set -----------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file describes the X86 AVX10 instruction set, defining the
// instructions, and properties of the instructions which are needed for code
// generation, machine code emission, and analysis.
//
//===----------------------------------------------------------------------===//

// VNNI FP16
let ExeDomain = SSEPackedSingle in
defm VDPPHPS : avx512_dpf16ps_sizes<0x52, "vdpphps", X86dpfp16ps, avx512vl_f16_info,
                                    [HasAVX10_2], [HasAVX10_2_512]>,
                    T8, PS, EVEX_CD8<32, CD8VF>;

// VNNI INT8
defm VPDPBSSD   : VNNI_common<0x50, "vpdpbssd", X86vpdpbssd, SchedWriteVecIMul, 1,
                              [HasAVX10_2], [HasAVX10_2_512]>, XD;
defm VPDPBSSDS  : VNNI_common<0x51, "vpdpbssds", X86vpdpbssds, SchedWriteVecIMul, 1,
                              [HasAVX10_2], [HasAVX10_2_512]>, XD;
defm VPDPBSUD   : VNNI_common<0x50, "vpdpbsud", X86vpdpbsud, SchedWriteVecIMul, 0,
                              [HasAVX10_2], [HasAVX10_2_512]>, XS;
defm VPDPBSUDS  : VNNI_common<0x51, "vpdpbsuds", X86vpdpbsuds, SchedWriteVecIMul, 0,
                              [HasAVX10_2], [HasAVX10_2_512]>, XS;
defm VPDPBUUD   : VNNI_common<0x50, "vpdpbuud", X86vpdpbuud, SchedWriteVecIMul, 1,
                              [HasAVX10_2], [HasAVX10_2_512]>, PS;
defm VPDPBUUDS  : VNNI_common<0x51, "vpdpbuuds", X86vpdpbuuds, SchedWriteVecIMul, 1,
                              [HasAVX10_2], [HasAVX10_2_512]>, PS;

// VNNI INT16
defm VPDPWSUD   : VNNI_common<0xd2, "vpdpwsud", X86vpdpwsud, SchedWriteVecIMul, 0,
                              [HasAVX10_2], [HasAVX10_2_512]>, XS;
defm VPDPWSUDS  : VNNI_common<0xd3, "vpdpwsuds", X86vpdpwsuds, SchedWriteVecIMul, 0,
                              [HasAVX10_2], [HasAVX10_2_512]>, XS;
defm VPDPWUSD   : VNNI_common<0xd2, "vpdpwusd", X86vpdpwusd, SchedWriteVecIMul, 0,
                              [HasAVX10_2], [HasAVX10_2_512]>, PD;
defm VPDPWUSDS  : VNNI_common<0xd3, "vpdpwusds", X86vpdpwusds, SchedWriteVecIMul, 0,
                              [HasAVX10_2], [HasAVX10_2_512]>, PD;
defm VPDPWUUD   : VNNI_common<0xd2, "vpdpwuud", X86vpdpwuud, SchedWriteVecIMul, 1,
                              [HasAVX10_2], [HasAVX10_2_512]>, PS;
defm VPDPWUUDS  : VNNI_common<0xd3, "vpdpwuuds", X86vpdpwuuds, SchedWriteVecIMul, 1,
                              [HasAVX10_2], [HasAVX10_2_512]>, PS;

// VMPSADBW
defm VMPSADBW : avx512_common_3Op_rm_imm8<0x42, X86Vmpsadbw, "vmpsadbw", SchedWritePSADBW,
                                          avx512vl_i16_info, avx512vl_i8_info,
                                          HasAVX10_2>,
                    XS, EVEX_CD8<32, CD8VF>;

// YMM Rounding
multiclass avx256_fp_binop_p_round<bits<8> opc, string OpcodeStr, SDNode OpNodeRnd,
                                   X86SchedWriteSizes sched> {
  defm PHZ256 : avx512_fp_round_packed<opc, OpcodeStr, OpNodeRnd, sched.PH.YMM,
                                       v16f16x_info>, T_MAP5,PS, EVEX_CD8<16, CD8VF>;
  defm PSZ256 : avx512_fp_round_packed<opc, OpcodeStr, OpNodeRnd, sched.PS.YMM,
                                       v8f32x_info>, TB, PS, EVEX_CD8<32, CD8VF>;
  defm PDZ256 : avx512_fp_round_packed<opc, OpcodeStr, OpNodeRnd, sched.PD.YMM,
                                       v4f64x_info>, TB, PD, EVEX_CD8<64, CD8VF>, REX_W;
}

multiclass avx256_fp_binop_p_sae<bits<8> opc, string OpcodeStr, SDNode OpNodeRnd,
                                 X86SchedWriteSizes sched> {
  defm PHZ256 : avx512_fp_sae_packed<opc, OpcodeStr, OpNodeRnd, sched.PH.YMM,
                                     v16f16x_info>, T_MAP5,PS, EVEX_CD8<16, CD8VF>;
  defm PSZ256 : avx512_fp_sae_packed<opc, OpcodeStr, OpNodeRnd, sched.PS.YMM,
                                     v8f32x_info>, TB, PS, EVEX_CD8<32, CD8VF>;
  defm PDZ256 : avx512_fp_sae_packed<opc, OpcodeStr, OpNodeRnd, sched.PD.YMM,
                                     v4f64x_info>, TB, PD, EVEX_CD8<64, CD8VF>, REX_W;
}

multiclass avx256_vcmp_p_sae<X86SchedWriteWidths sched> {
  defm PHZ256 : avx512_vcmp_sae<sched.YMM, v16f16x_info>, AVX512PSIi8Base, EVEX_CD8<16, CD8VF>, TA;
  defm PSZ256 : avx512_vcmp_sae<sched.YMM, v8f32x_info>, AVX512PSIi8Base, EVEX_CD8<32, CD8VF>;
  defm PDZ256 : avx512_vcmp_sae<sched.YMM, v4f64x_info>, AVX512PDIi8Base, EVEX_CD8<64, CD8VF>, REX_W;
}

multiclass avx256_fixupimm_packed_all<bits<8> opc, string OpcodeStr,
                                      X86SchedWriteWidths sched> {
  defm PSZ256 : avx512_fixupimm_packed_sae<opc, OpcodeStr, sched.YMM, v8f32x_info,
                                           v8i32x_info>, EVEX_CD8<32, CD8VF>;
  defm PDZ256 : avx512_fixupimm_packed_sae<opc, OpcodeStr, sched.YMM, v4f64x_info,
                                           v4i64x_info>, EVEX_CD8<64, CD8VF>, REX_W;
}

multiclass avx256_vgetexp<bits<8> opc, string OpcodeStr, SDNode OpNodeSAE,
                          X86SchedWriteWidths sched> {
  defm PHZ256 : avx512_fp28_p_sae<opc, OpcodeStr#"ph", v16f16x_info, OpNodeSAE,
                                  sched.YMM>, T_MAP6,PD, EVEX_CD8<16, CD8VF>;
  defm PSZ256 : avx512_fp28_p_sae<opc, OpcodeStr#"ps", v8f32x_info, OpNodeSAE,
                                  sched.YMM>, T8,PD, EVEX_CD8<32, CD8VF>;
  defm PDZ256 : avx512_fp28_p_sae<opc, OpcodeStr#"pd", v4f64x_info, OpNodeSAE,
                                  sched.YMM>, T8,PD, EVEX_CD8<64, CD8VF>, REX_W;
}

multiclass avx256_unary_fp_sae<string OpcodeStr, bits<8> opcPs, bits<8> opcPd,
                               SDNode OpNodeSAE, X86SchedWriteWidths sched> {
  defm PHZ256 : avx512_unary_fp_sae_packed_imm<opcPs, OpcodeStr, OpNodeSAE, sched.YMM,
                                               v16f16x_info>, AVX512PSIi8Base, TA, EVEX_CD8<16, CD8VF>;
  defm PSZ256 : avx512_unary_fp_sae_packed_imm<opcPs, OpcodeStr, OpNodeSAE, sched.YMM,
                                               v8f32x_info>, AVX512AIi8Base, EVEX_CD8<32, CD8VF>;
  defm PDZ256 : avx512_unary_fp_sae_packed_imm<opcPd, OpcodeStr, OpNodeSAE, sched.YMM,
                                               v4f64x_info>, AVX512AIi8Base, EVEX_CD8<64, CD8VF>, REX_W;
}

multiclass avx256_common_fp_sae_packed_imm<bits<8> opc, string OpcodeStr, SDNode OpNodeSAE,
                                           X86SchedWriteWidths sched> {
  defm PSZ256 : avx512_fp_sae_packed_imm<opc, OpcodeStr#"ps", OpNodeSAE, sched.YMM,
                                         v8f32x_info>, EVEX_CD8<32, CD8VF>;
  defm PDZ256 : avx512_fp_sae_packed_imm<opc, OpcodeStr#"pd", OpNodeSAE, sched.YMM,
                                         v4f64x_info>, EVEX_CD8<64, CD8VF>, REX_W;
}

multiclass avx256_fp_scalef_round<bits<8> opc, string OpcodeStr, SDNode OpNodeRnd,
                                  X86SchedWriteWidths sched> {
  defm PHZ256 : avx512_fp_round_packed<opc, OpcodeStr, OpNodeRnd, sched.YMM,
                                       v16f16x_info>, T_MAP6,PD, EVEX_CD8<16, CD8VF>;
  defm PSZ256 : avx512_fp_round_packed<opc, OpcodeStr, OpNodeRnd, sched.YMM,
                                       v8f32x_info>, T8,PD, EVEX_CD8<32, CD8VF>;
  defm PDZ256 : avx512_fp_round_packed<opc, OpcodeStr, OpNodeRnd, sched.YMM,
                                       v4f64x_info>, T8,PD, EVEX_CD8<64, CD8VF>, REX_W;
}

multiclass avx256_sqrt_packed_all_round<bits<8> opc, string OpcodeStr,
                                        X86SchedWriteSizes sched> {
  defm PHZ256 : avx512_sqrt_packed_round<opc, !strconcat(OpcodeStr, "ph"),
                                         sched.PH.YMM, v16f16x_info>, T_MAP5,PS, EVEX_CD8<16, CD8VF>;
  defm PSZ256 : avx512_sqrt_packed_round<opc, !strconcat(OpcodeStr, "ps"),
                                         sched.PS.YMM, v8f32x_info>, TB, PS, EVEX_CD8<32, CD8VF>;
  defm PDZ256 : avx512_sqrt_packed_round<opc, !strconcat(OpcodeStr, "pd"),
                                         sched.PD.YMM, v4f64x_info>, TB, PD, EVEX_CD8<64, CD8VF>, REX_W;
}

multiclass avx256_vcvtw_rc<string OpcodeStr, SDNode OpNodeRnd> {
  defm PHZ256 : avx512_vcvt_fp_rc<0x7D, OpcodeStr, v16f16x_info, v16i16x_info, OpNodeRnd,
                                  SchedWriteCvtPD2DQ.YMM>, EVEX_CD8<16, CD8VF>;
}

multiclass avx256_cvtdq2fp_rc<bits<8> opc, string OpcodeStr, SDNode OpNodeRnd,
                              X86SchedWriteWidths sched> {
  defm PHZ256 : avx512_vcvt_fp_rc<opc, !strconcat(OpcodeStr, "ph"), v8f16x_info,
                                  v8i32x_info, OpNodeRnd, sched.YMM>, T_MAP5,PS, EVEX_CD8<32, CD8VF>;
  defm PSZ256 : avx512_vcvt_fp_rc<opc, !strconcat(OpcodeStr, "ps"), v8f32x_info,
                                  v8i32x_info, OpNodeRnd, sched.YMM>, TB, PS, EVEX_CD8<32, CD8VF>;
}

multiclass avx256_cvtudq2fp_rc<bits<8> opc, string OpcodeStr, SDNode OpNodeRnd,
                               X86SchedWriteWidths sched> {
  defm PHZ256 : avx512_vcvt_fp_rc<opc, !strconcat(OpcodeStr, "ph"), v8f16x_info,
                                  v8i32x_info, OpNodeRnd, sched.YMM>, T_MAP5,XD, EVEX_CD8<32, CD8VF>;
  defm PSZ256 : avx512_vcvt_fp_rc<opc, !strconcat(OpcodeStr, "ps"), v8f32x_info,
                                  v8i32x_info, OpNodeRnd, sched.YMM>, TB, XD, EVEX_CD8<32, CD8VF>;
}

multiclass avx256_cvtqq2fp_rc<string OpcodeStr, X86VectorVTInfo _Src> {
  defm PHZ256 : avx512_vcvt_fp_rc<0x5B, !strconcat(OpcodeStr, "ph"), v8f16x_info,
                                  _Src, X86VSintToFpRnd, SchedWriteCvtDQ2PS.YMM>, T_MAP5,PS;
  defm PSZ256 : avx512_vcvt_fp_rc<0x5B, !strconcat(OpcodeStr, "ps"), v4f32x_info,
                                  _Src, X86VSintToFpRnd, SchedWriteCvtDQ2PS.YMM>, TB, PS;
  defm PDZ256 : avx512_vcvt_fp_rc<0xE6, !strconcat(OpcodeStr, "pd"), v4f64x_info,
                                  _Src, X86VSintToFpRnd, SchedWriteCvtDQ2PD.YMM>, TB, XS;
}

multiclass avx256_cvtuqq2fp_rc<string OpcodeStr, X86VectorVTInfo _Src> {
  defm PHZ256 : avx512_vcvt_fp_rc<0x7A, !strconcat(OpcodeStr, "ph"), v8f16x_info,
                                  _Src, X86VUintToFpRnd, SchedWriteCvtDQ2PS.YMM>, T_MAP5,XD;
  defm PSZ256 : avx512_vcvt_fp_rc<0x7A, !strconcat(OpcodeStr, "ps"), v4f32x_info,
                                  _Src, X86VUintToFpRnd, SchedWriteCvtDQ2PS.YMM>, TB, XD;
  defm PDZ256 : avx512_vcvt_fp_rc<0x7A, !strconcat(OpcodeStr, "pd"), v4f64x_info,
                                  _Src, X86VUintToFpRnd, SchedWriteCvtDQ2PD.YMM>, TB, XS;
}

multiclass avx256_vcvt_pd2<string OpcodeStr, X86VectorVTInfo _Src> {
  defm PHZ256 : avx512_vcvt_fp_rc<0x5A, !strconcat(OpcodeStr, "ph"), v8f16x_info,
                                  _Src, X86vfproundRnd, SchedWriteCvtPD2PS.YMM>, T_MAP5,PD;
  defm PSZ256 : avx512_vcvt_fp_rc<0x5A, !strconcat(OpcodeStr, "ps"), v4f32x_info,
                                  _Src, X86vfproundRnd, SchedWriteCvtPD2PS.YMM>, TB, PD;
  defm DQZ256 : avx512_vcvt_fp_rc<0xE6, !strconcat(OpcodeStr, "dq"), v4i32x_info,
                                  _Src, X86cvtp2IntRnd, SchedWriteCvtPD2DQ.YMM>, TB, XD;
  defm QQZ256 : avx512_vcvt_fp_rc<0x7B, !strconcat(OpcodeStr, "qq"), v4i64x_info,
                                  _Src, X86cvtp2IntRnd, SchedWriteCvtPD2DQ.YMM>, TB, PD;
  defm UDQZ256 : avx512_vcvt_fp_rc<0x79, !strconcat(OpcodeStr, "udq"), v4i32x_info,
                                   _Src, X86cvtp2UIntRnd, SchedWriteCvtPD2DQ.YMM>, TB, PS;
  defm UQQZ256 : avx512_vcvt_fp_rc<0x79, !strconcat(OpcodeStr, "uqq"), v4i64x_info,
                                   _Src, X86cvtp2UIntRnd, SchedWriteCvtPD2DQ.YMM>, TB, PD;
}

multiclass avx256_vcvt_ps2<string OpcodeStr> {
  defm PHZ256 : avx512_cvtps2ph_sae<v8i16x_info, v8f32x_info, WriteCvtPS2PHZ>, EVEX_CD8<32, CD8VH>;
  defm PHXZ256 : avx512_vcvt_fp_rc<0x1D, !strconcat(OpcodeStr, "phx"), v8f16x_info, v8f32x_info,
                                   X86vfproundRnd, SchedWriteCvtPD2PS.YMM>, T_MAP5,PD, EVEX_CD8<32, CD8VF>;
  defm PDZ256 : avx512_vcvt_fp_sae<0x5A, !strconcat(OpcodeStr, "pd"), v4f64x_info, v4f32x_info,
                                   X86vfpextSAE, SchedWriteCvtPS2PD.YMM>, TB, PS, EVEX_CD8<32, CD8VF>;
  defm DQZ256 : avx512_vcvt_fp_rc<0x5B, !strconcat(OpcodeStr, "dq"), v8i32x_info, v8f32x_info,
                                  X86cvtp2IntRnd, SchedWriteCvtPS2DQ.YMM>, TB, PD, EVEX_CD8<32, CD8VF>;
  defm QQZ256 : avx512_vcvt_fp_rc<0x7B, !strconcat(OpcodeStr, "qq"), v4i64x_info, v4f32x_info,
                                  X86cvtp2IntRnd, SchedWriteCvtPS2DQ.YMM>, TB, PD, EVEX_CD8<32, CD8VF>;
  defm UDQZ256 : avx512_vcvt_fp_rc<0x79, !strconcat(OpcodeStr, "udq"), v8i32x_info, v8f32x_info,
                                   X86cvtp2UIntRnd, SchedWriteCvtPS2DQ.YMM>, TB, PS, EVEX_CD8<32, CD8VF>;
  defm UQQZ256 : avx512_vcvt_fp_rc<0x79, !strconcat(OpcodeStr, "uqq"), v4i64x_info, v4f32x_info,
                                   X86cvtp2UIntRnd, SchedWriteCvtPS2DQ.YMM>, TB, PD, EVEX_CD8<32, CD8VF>;
}

multiclass avx256_vcvt_ph2<string OpcodeStr> {
  defm PSZ256 : avx512_cvtph2ps_sae<v8f32x_info, v8i16x_info, WriteCvtPH2PSZ>, EVEX_CD8<32, CD8VH>;
  defm PSXZ256 : avx512_vcvt_fp_sae<0x13, !strconcat(OpcodeStr, "psx"), v8f32x_info, v8f16x_info,
                                    X86vfpextSAE, SchedWriteCvtPS2PD.YMM>, T_MAP6,PD, EVEX_CD8<16, CD8VH>;
  defm PDZ256 : avx512_vcvt_fp_sae<0x5A, !strconcat(OpcodeStr, "pd"), v4f64x_info, v8f16x_info,
                                   X86vfpextSAE, SchedWriteCvtPS2PD.YMM>, T_MAP5,PS, EVEX_CD8<16, CD8VQ>;
  defm WZ256 : avx512_vcvt_fp_rc<0x7D, !strconcat(OpcodeStr, "w"), v16i16x_info, v16f16x_info,
                                 X86cvtp2IntRnd, SchedWriteCvtPD2DQ.YMM>, T_MAP5,PD, EVEX_CD8<16, CD8VF>;
  defm DQZ256 : avx512_vcvt_fp_rc<0x5B, !strconcat(OpcodeStr, "dq"), v8i32x_info, v8f16x_info,
                                  X86cvtp2IntRnd, SchedWriteCvtPS2DQ.YMM>, T_MAP5,PD, EVEX_CD8<16, CD8VH>;
  defm QQZ256 : avx512_vcvt_fp_rc<0x7B, !strconcat(OpcodeStr, "qq"), v4i64x_info, v8f16x_info,
                                  X86cvtp2IntRnd, SchedWriteCvtPS2DQ.YMM>, T_MAP5,PD, EVEX_CD8<16, CD8VQ>;
  defm UWZ256 : avx512_vcvt_fp_rc<0x7D, !strconcat(OpcodeStr, "uw"), v16i16x_info, v16f16x_info,
                                  X86cvtp2UIntRnd, SchedWriteCvtPD2DQ.YMM>, T_MAP5,PS, EVEX_CD8<16, CD8VF>;
  defm UDQZ256 : avx512_vcvt_fp_rc<0x79, !strconcat(OpcodeStr, "udq"), v8i32x_info, v8f16x_info,
                                   X86cvtp2UIntRnd, SchedWriteCvtPS2DQ.YMM>, T_MAP5,PS, EVEX_CD8<16, CD8VH>;
  defm UQQZ256 : avx512_vcvt_fp_rc<0x79, !strconcat(OpcodeStr, "uqq"), v4i64x_info, v8f16x_info,
                                   X86cvtp2UIntRnd, SchedWriteCvtPS2DQ.YMM>, T_MAP5,PD, EVEX_CD8<16, CD8VQ>;
}

multiclass avx256_vcvtt_pd2<string OpcodeStr, X86VectorVTInfo _Src> {
  defm DQZ256 : avx512_vcvt_fp_sae<0xE6, !strconcat(OpcodeStr, "dq"), v4i32x_info,
                                   _Src, X86cvttp2siSAE, SchedWriteCvtPD2DQ.YMM>, PD;
  defm QQZ256 : avx512_vcvt_fp_sae<0x7A, !strconcat(OpcodeStr, "qq"), v4i64x_info,
                                   _Src, X86cvttp2siSAE, SchedWriteCvtPD2DQ.YMM>, PD;
  defm UDQZ256 : avx512_vcvt_fp_sae<0x78, !strconcat(OpcodeStr, "udq"), v4i32x_info,
                                    _Src, X86cvttp2uiSAE, SchedWriteCvtPD2DQ.YMM>, PS;
  defm UQQZ256 : avx512_vcvt_fp_sae<0x78, !strconcat(OpcodeStr, "uqq"), v4i64x_info,
                                    _Src, X86cvttp2uiSAE, SchedWriteCvtPD2DQ.YMM>, PD;
}

multiclass avx256_vcvtt_ps2<string OpcodeStr> {
  defm DQZ256 : avx512_vcvt_fp_sae<0x5B, !strconcat(OpcodeStr, "dq"), v8i32x_info, v8f32x_info,
                                   X86cvttp2siSAE, SchedWriteCvtPS2DQ.YMM>, XS, EVEX_CD8<32, CD8VF>;
  defm QQZ256 : avx512_vcvt_fp_sae<0x7A, !strconcat(OpcodeStr, "qq"), v4i64x_info, v4f32x_info,
                                   X86cvttp2siSAE, SchedWriteCvtPS2DQ.YMM>, PD, EVEX_CD8<32, CD8VH>;
  defm UDQZ256 : avx512_vcvt_fp_sae<0x78, !strconcat(OpcodeStr, "udq"), v8i32x_info, v8f32x_info,
                                    X86cvttp2uiSAE, SchedWriteCvtPS2DQ.YMM>, PS, EVEX_CD8<32, CD8VF>;
  defm UQQZ256 : avx512_vcvt_fp_sae<0x78, !strconcat(OpcodeStr, "uqq"), v4i64x_info, v4f32x_info,
                                    X86cvttp2uiSAE, SchedWriteCvtPS2DQ.YMM>, PD, EVEX_CD8<32, CD8VH>;
}

multiclass avx256_vcvtt_ph2<string OpcodeStr> {
  defm WZ256 : avx512_vcvt_fp_sae<0x7C, !strconcat(OpcodeStr, "w"), v16i16x_info, v16f16x_info,
                                  X86cvttp2siSAE, SchedWriteCvtPD2DQ.YMM>, T_MAP5,PD, EVEX_CD8<16, CD8VF>;
  defm DQZ256 : avx512_vcvt_fp_sae<0x5B, !strconcat(OpcodeStr, "dq"), v8i32x_info, v8f16x_info,
                                   X86cvttp2siSAE, SchedWriteCvtPS2DQ.YMM>, T_MAP5,XS, EVEX_CD8<16, CD8VH>;
  defm QQZ256 : avx512_vcvt_fp_sae<0x7A, !strconcat(OpcodeStr, "qq"), v4i64x_info, v8f16x_info,
                                   X86cvttp2siSAE, SchedWriteCvtPS2DQ.YMM>, T_MAP5,PD, EVEX_CD8<16, CD8VQ>;
  defm UWZ256 : avx512_vcvt_fp_sae<0x7C, !strconcat(OpcodeStr, "uw"), v16i16x_info, v16f16x_info,
                                   X86cvttp2uiSAE, SchedWriteCvtPD2DQ.YMM>, T_MAP5,PS, EVEX_CD8<16, CD8VF>;
  defm UDQZ256 : avx512_vcvt_fp_sae<0x78, !strconcat(OpcodeStr, "udq"), v8i32x_info, v8f16x_info,
                                    X86cvttp2uiSAE, SchedWriteCvtPS2DQ.YMM>, T_MAP5,PS, EVEX_CD8<16, CD8VH>;
  defm UQQZ256 : avx512_vcvt_fp_sae<0x78, !strconcat(OpcodeStr, "uqq"), v4i64x_info, v8f16x_info,
                                    X86cvttp2uiSAE, SchedWriteCvtPS2DQ.YMM>, T_MAP5,PD, EVEX_CD8<16, CD8VQ>;
}

multiclass avx256_fma3_132_round<bits<8> opc, string OpcodeStr, SDNode OpNodeRnd> {
  defm PHZ256 : avx512_fma3_132_round<opc, !strconcat(OpcodeStr, "ph"), OpNodeRnd,
                                      SchedWriteFMA.YMM, v16f16x_info>, T_MAP6,PD, EVEX_CD8<16, CD8VF>;
  defm PSZ256 : avx512_fma3_132_round<opc, !strconcat(OpcodeStr, "ps"), OpNodeRnd,
                                      SchedWriteFMA.YMM, v8f32x_info>, T8,PD, EVEX_CD8<32, CD8VF>;
  defm PDZ256 : avx512_fma3_132_round<opc, !strconcat(OpcodeStr, "pd"), OpNodeRnd,
                                      SchedWriteFMA.YMM, v4f64x_info>, T8,PD, EVEX_CD8<64, CD8VF>, REX_W;
}

multiclass avx256_fma3_213_round<bits<8> opc, string OpcodeStr, SDNode OpNodeRnd> {
  defm PHZ256 : avx512_fma3_213_round<opc, !strconcat(OpcodeStr, "ph"), OpNodeRnd,
                                      SchedWriteFMA.YMM, v16f16x_info>, T_MAP6,PD, EVEX_CD8<16, CD8VF>;
  defm PSZ256 : avx512_fma3_213_round<opc, !strconcat(OpcodeStr, "ps"), OpNodeRnd,
                                      SchedWriteFMA.YMM, v8f32x_info>, T8,PD, EVEX_CD8<32, CD8VF>;
  defm PDZ256 : avx512_fma3_213_round<opc, !strconcat(OpcodeStr, "pd"), OpNodeRnd,
                                      SchedWriteFMA.YMM, v4f64x_info>, T8,PD, EVEX_CD8<64, CD8VF>, REX_W;
}

multiclass avx256_fma3_231_round<bits<8> opc, string OpcodeStr, SDNode OpNodeRnd> {
  defm PHZ256 : avx512_fma3_231_round<opc, !strconcat(OpcodeStr, "ph"), OpNodeRnd,
                                      SchedWriteFMA.YMM, v16f16x_info>, T_MAP6,PD, EVEX_CD8<16, CD8VF>;
  defm PSZ256 : avx512_fma3_231_round<opc, !strconcat(OpcodeStr, "ps"), OpNodeRnd,
                                      SchedWriteFMA.YMM, v8f32x_info>, T8,PD, EVEX_CD8<32, CD8VF>;
  defm PDZ256 : avx512_fma3_231_round<opc, !strconcat(OpcodeStr, "pd"), OpNodeRnd,
                                      SchedWriteFMA.YMM, v4f64x_info>, T8,PD, EVEX_CD8<64, CD8VF>, REX_W;
}

multiclass avx256_fma3_round3<bits<8> opc132, bits<8> opc213, bits<8> opc231,
                              string OpcodeStr, SDNode OpNodeRnd> {
  defm NAME#132 : avx256_fma3_132_round<opc132, !strconcat(OpcodeStr, "132"), OpNodeRnd>;
  defm NAME#213 : avx256_fma3_213_round<opc213, !strconcat(OpcodeStr, "213"), OpNodeRnd>;
  defm NAME#231 : avx256_fma3_231_round<opc231, !strconcat(OpcodeStr, "231"), OpNodeRnd>;
}

let Predicates = [HasAVX10_2], hasEVEX_U = 1, OpEnc = EncEVEX in {
  defm VADD : avx256_fp_binop_p_round<0x58, "vadd", X86faddRnd, SchedWriteFAddSizes>;
  defm VMUL : avx256_fp_binop_p_round<0x59, "vmul", X86fmulRnd, SchedWriteFMulSizes>;
  defm VSUB : avx256_fp_binop_p_round<0x5C, "vsub", X86fsubRnd, SchedWriteFAddSizes>;
  defm VDIV : avx256_fp_binop_p_round<0x5E, "vdiv", X86fdivRnd, SchedWriteFDivSizes>;
  defm VMIN : avx256_fp_binop_p_sae<0x5D, "vmin", X86fminSAE, SchedWriteFCmpSizes>;
  defm VMAX : avx256_fp_binop_p_sae<0x5F, "vmax", X86fmaxSAE, SchedWriteFCmpSizes>;
  defm VCMP : avx256_vcmp_p_sae<SchedWriteFCmp>, EVEX, VVVV;
  defm VFIXUPIMM : avx256_fixupimm_packed_all<0x54, "vfixupimm", SchedWriteFAdd>, AVX512AIi8Base, EVEX, VVVV;
  defm VGETEXP : avx256_vgetexp<0x42, "vgetexp", X86fgetexpSAE, SchedWriteFRnd>;
  defm VREDUCE : avx256_unary_fp_sae<"vreduce", 0x56, 0x56, X86VReduceSAE, SchedWriteFRnd>;
  defm VRNDSCALE : avx256_unary_fp_sae<"vrndscale", 0x08, 0x09, X86VRndScaleSAE, SchedWriteFRnd>;
  defm VGETMANT : avx256_unary_fp_sae<"vgetmant", 0x26, 0x26, X86VGetMantSAE, SchedWriteFRnd>;
  defm VRANGE : avx256_common_fp_sae_packed_imm<0x50, "vrange", X86VRangeSAE, SchedWriteFAdd>, AVX512AIi8Base, EVEX, VVVV;
  defm VSCALEF : avx256_fp_scalef_round<0x2C, "vscalef", X86scalefRnd, SchedWriteFAdd>;
  defm VSQRT : avx256_sqrt_packed_all_round<0x51, "vsqrt", SchedWriteFSqrtSizes>;
  defm VCVTW2 : avx256_vcvtw_rc<"vcvtw2ph", X86VSintToFpRnd>, T_MAP5, XS;
  defm VCVTDQ2 : avx256_cvtdq2fp_rc<0x5B, "vcvtdq2", X86VSintToFpRnd, SchedWriteCvtDQ2PS>;
  defm VCVTQQ2 : avx256_cvtqq2fp_rc<"vcvtqq2", v4i64x_info>, EVEX_CD8<64, CD8VF>, REX_W;
  defm VCVTUW2 : avx256_vcvtw_rc<"vcvtuw2ph", X86VUintToFpRnd>, T_MAP5,XD;
  defm VCVTUDQ2 : avx256_cvtudq2fp_rc<0x7A, "vcvtudq2", X86VUintToFpRnd, SchedWriteCvtDQ2PS>;
  defm VCVTUQQ2 : avx256_cvtuqq2fp_rc<"vcvtuqq2", v4i64x_info>, EVEX_CD8<64, CD8VF>, REX_W;
  defm VCVTPD2 : avx256_vcvt_pd2<"vcvtpd2", v4f64x_info>, EVEX_CD8<64, CD8VF>, REX_W;
  defm VCVTPS2 : avx256_vcvt_ps2<"vcvtps2">;
  defm VCVTPH2 : avx256_vcvt_ph2<"vcvtph2">;
  defm VCVTTPD2 : avx256_vcvtt_pd2<"vcvttpd2", v4f64x_info>, EVEX_CD8<64, CD8VF>, TB, REX_W;
  defm VCVTTPS2 : avx256_vcvtt_ps2<"vcvttps2">, TB;
  defm VCVTTPH2 : avx256_vcvtt_ph2<"vcvttph2">;
  defm VFMADD : avx256_fma3_round3<0x98, 0xA8, 0xB8, "vfmadd", X86FmaddRnd>;
  defm VFMSUB : avx256_fma3_round3<0x9A, 0xAA, 0xBA, "vfmsub", X86FmsubRnd>;
  defm VFMADDSUB : avx256_fma3_round3<0x96, 0xA6, 0xB6, "vfmaddsub", X86FmaddsubRnd>;
  defm VFMSUBADD : avx256_fma3_round3<0x97, 0xA7, 0xB7, "vfmsubadd", X86FmsubaddRnd>;
  defm VFNMADD : avx256_fma3_round3<0x9C, 0xAC, 0xBC, "vfnmadd", X86FnmaddRnd>;
  defm VFNMSUB : avx256_fma3_round3<0x9E, 0xAE, 0xBE, "vfnmsub", X86FnmsubRnd>;
  defm VFMULCPHZ256 : avx512_fp_round_packed<0xD6, "vfmulcph", x86vfmulcRnd, SchedWriteFMA.YMM,
                                             v8f32x_info, "", "@earlyclobber $dst">, T_MAP6,XS, EVEX_CD8<32, CD8VF>;
  defm VFCMULCPHZ256 : avx512_fp_round_packed<0xD6, "vfcmulcph", x86vfcmulcRnd, SchedWriteFMA.YMM,
                                              v8f32x_info, "", "@earlyclobber $dst">, T_MAP6,XD, EVEX_CD8<32, CD8VF>;
  defm VFMADDCPHZ256 : avx512_cfmaop_round<0x56, "vfmaddcph", x86vfmaddcRnd,
                                           v8f32x_info>, T_MAP6,XS, EVEX_CD8<32, CD8VF>, Sched<[WriteFMAY]>;
  defm VFCMADDCPHZ256 : avx512_cfmaop_round<0x56, "vfcmaddcph", x86vfcmaddcRnd,
                                            v8f32x_info>, T_MAP6,XD, EVEX_CD8<32, CD8VF>, Sched<[WriteFMAY]>;
}

//-------------------------------------------------
// AVX10 MINMAX instructions
//-------------------------------------------------

multiclass avx10_minmax_packed_base<string OpStr, X86VectorVTInfo VTI, SDNode OpNode> {
  let ExeDomain = VTI.ExeDomain, Uses = [MXCSR], mayRaiseFPException = 1 in {
    defm rri : AVX512_maskable<0x52, MRMSrcReg, VTI, (outs VTI.RC:$dst),
                                (ins VTI.RC:$src1, VTI.RC:$src2, i32u8imm:$src3), OpStr,
                                "$src3, $src2, $src1", "$src1, $src2, $src3",
                                (VTI.VT (OpNode VTI.RC:$src1, VTI.RC:$src2,
                                                (i32 timm:$src3)))>,
                                EVEX, VVVV, Sched<[WriteFMAX]>;
    defm rmi : AVX512_maskable<0x52, MRMSrcMem, VTI, (outs VTI.RC:$dst),
                                (ins VTI.RC:$src1, VTI.MemOp:$src2, i32u8imm:$src3), OpStr,
                                "$src3, $src2, $src1", "$src1, $src2, $src3",
                                (VTI.VT (OpNode VTI.RC:$src1, (VTI.LdFrag addr:$src2),
                                                (i32 timm:$src3)))>,
                                EVEX, VVVV,
                                Sched<[WriteFMAX.Folded, WriteFMAX.ReadAfterFold]>;
    defm rmbi : AVX512_maskable<0x52, MRMSrcMem, VTI, (outs VTI.RC:$dst),
                                (ins VTI.RC:$src1, VTI.ScalarMemOp:$src2, i32u8imm:$src3),
                                OpStr, "$src3, ${src2}"#VTI.BroadcastStr#", $src1",
                                "$src1, ${src2}"#VTI.BroadcastStr#", $src3",
                                (VTI.VT (OpNode VTI.RC:$src1, (VTI.BroadcastLdFrag addr:$src2),
                                                (i32 timm:$src3)))>,
                                EVEX, VVVV, EVEX_B,
                                Sched<[WriteFMAX.Folded, WriteFMAX.ReadAfterFold]>;
  }
}

multiclass avx10_minmax_packed_sae<string OpStr, AVX512VLVectorVTInfo VTI, SDNode OpNode> {
  let Uses = []<Register>, mayRaiseFPException = 0 in {
    defm Zrrib : AVX512_maskable<0x52, MRMSrcReg, VTI.info512, (outs VTI.info512.RC:$dst),
                                (ins VTI.info512.RC:$src1, VTI.info512.RC:$src2, i32u8imm:$src3), OpStr,
                                "$src3, {sae}, $src2, $src1", "$src1, $src2, {sae}, $src3",
                                (VTI.info512.VT (OpNode (VTI.info512.VT VTI.info512.RC:$src1),
                                                        (VTI.info512.VT VTI.info512.RC:$src2),
                                                        (i32 timm:$src3)))>,
                                EVEX, VVVV, EVEX_B, EVEX_V512, Sched<[WriteFMAX]>;
    let hasEVEX_U = 1 in
    defm Z256rrib : AVX512_maskable<0x52, MRMSrcReg, VTI.info256, (outs VTI.info256.RC:$dst),
                                (ins VTI.info256.RC:$src1, VTI.info256.RC:$src2, i32u8imm:$src3), OpStr,
                                "$src3, {sae}, $src2, $src1", "$src1, $src2, {sae}, $src3",
                                (VTI.info256.VT (OpNode (VTI.info256.VT VTI.info256.RC:$src1),
                                                        (VTI.info256.VT VTI.info256.RC:$src2),
                                                        (i32 timm:$src3)))>,
                                EVEX, VVVV, EVEX_B, EVEX_V256, Sched<[WriteFMAX]>;
  }
}

multiclass avx10_minmax_packed<string OpStr, AVX512VLVectorVTInfo VTI, SDNode OpNode> {
  let Predicates = [HasAVX10_2_512] in
    defm Z    :   avx10_minmax_packed_base<OpStr, VTI.info512, OpNode>, EVEX_V512;
  let Predicates = [HasAVX10_2] in {
    defm Z256 :   avx10_minmax_packed_base<OpStr, VTI.info256, OpNode>, EVEX_V256;
    defm Z128 :   avx10_minmax_packed_base<OpStr, VTI.info128, OpNode>, EVEX_V128;
  }
}

multiclass avx10_minmax_scalar<string OpStr, X86VectorVTInfo _, SDNode OpNode,
                                SDNode OpNodeSAE> {
  let ExeDomain = _.ExeDomain, Predicates = [HasAVX10_2] in {
    let mayRaiseFPException = 1 in {
      defm rri : AVX512_maskable<0x53, MRMSrcReg, _, (outs VR128X:$dst),
                               (ins VR128X:$src1, VR128X:$src2, i32u8imm:$src3),
                                OpStr, "$src3, $src2, $src1", "$src1, $src2, $src3",
                                (_.VT (OpNode (_.VT _.RC:$src1), (_.VT _.RC:$src2),
                                              (i32 timm:$src3)))>,
                                Sched<[WriteFMAX]>;

      defm rmi : AVX512_maskable<0x53, MRMSrcMem, _, (outs VR128X:$dst),
                       (ins VR128X:$src1, _.ScalarMemOp:$src2, i32u8imm:$src3),
                       OpStr, "$src3, $src2, $src1", "$src1, $src2, $src3",
                       (_.VT (OpNode (_.VT _.RC:$src1), (_.ScalarIntMemFrags addr:$src2),
                                     (i32 timm:$src3)))>,
                       Sched<[WriteFMAX.Folded, WriteFMAX.ReadAfterFold]>;
    }
    let Uses = []<Register>, mayRaiseFPException = 0 in
      defm rrib : AVX512_maskable<0x53, MRMSrcReg, _, (outs VR128X:$dst),
                        (ins VR128X:$src1, VR128X:$src2, i32u8imm:$src3),
                        OpStr, "$src3, {sae}, $src2, $src1",
                        "$src1, $src2, {sae}, $src3",
                        (_.VT (OpNodeSAE (_.VT _.RC:$src1), (_.VT _.RC:$src2),
                                         (i32 timm:$src3)))>,
                        Sched<[WriteFMAX]>, EVEX_B;
  }
}


let mayRaiseFPException = 0 in
defm VMINMAXNEPBF16 : avx10_minmax_packed<"vminmaxnepbf16", avx512vl_bf16_info, X86vminmax>,
                      AVX512XDIi8Base, EVEX_CD8<16, CD8VF>, TA;

defm VMINMAXPD : avx10_minmax_packed<"vminmaxpd", avx512vl_f64_info, X86vminmax>,
                 avx10_minmax_packed_sae<"vminmaxpd", avx512vl_f64_info, X86vminmaxSae>,
                 AVX512PDIi8Base, REX_W, TA, EVEX_CD8<64, CD8VF>;

defm VMINMAXPH : avx10_minmax_packed<"vminmaxph", avx512vl_f16_info, X86vminmax>,
                 avx10_minmax_packed_sae<"vminmaxph", avx512vl_f16_info, X86vminmaxSae>,
                 AVX512PSIi8Base, TA, EVEX_CD8<16, CD8VF>;

defm VMINMAXPS : avx10_minmax_packed<"vminmaxps", avx512vl_f32_info, X86vminmax>,
                 avx10_minmax_packed_sae<"vminmaxps", avx512vl_f32_info, X86vminmaxSae>,
                 AVX512PDIi8Base, TA, EVEX_CD8<32, CD8VF>;

defm VMINMAXSD : avx10_minmax_scalar<"vminmaxsd", v2f64x_info, X86vminmaxs, X86vminmaxsSae>,
                 AVX512AIi8Base, VEX_LIG, EVEX, VVVV, EVEX_CD8<64, CD8VT1>, REX_W;
defm VMINMAXSH : avx10_minmax_scalar<"vminmaxsh", v8f16x_info, X86vminmaxs, X86vminmaxsSae>,
                 AVX512PSIi8Base, VEX_LIG, EVEX, VVVV, EVEX_CD8<16, CD8VT1>, TA;
defm VMINMAXSS : avx10_minmax_scalar<"vminmaxss", v4f32x_info, X86vminmaxs, X86vminmaxsSae>,
                 AVX512AIi8Base, VEX_LIG, EVEX, VVVV, EVEX_CD8<32, CD8VT1>;

//-------------------------------------------------
// AVX10 SATCVT instructions
//-------------------------------------------------

multiclass avx10_sat_cvt_rmb<bits<8> Opc, string OpStr, X86FoldableSchedWrite sched,
                             X86VectorVTInfo DestInfo,
                             X86VectorVTInfo SrcInfo,
                             SDNode MaskNode> {
  defm rr: AVX512_maskable<Opc, MRMSrcReg, DestInfo, (outs DestInfo.RC:$dst),
                           (ins SrcInfo.RC:$src), OpStr, "$src", "$src",
                           (DestInfo.VT (MaskNode (SrcInfo.VT SrcInfo.RC:$src)))>,
                          Sched<[sched]>;
  defm rm: AVX512_maskable<Opc, MRMSrcMem, DestInfo, (outs DestInfo.RC:$dst),
                           (ins SrcInfo.MemOp:$src), OpStr, "$src", "$src",
                           (DestInfo.VT (MaskNode (SrcInfo.VT
                           (SrcInfo.LdFrag addr:$src))))>,
                          Sched<[sched.Folded, sched.ReadAfterFold]>;
  defm rmb: AVX512_maskable<Opc, MRMSrcMem, DestInfo, (outs DestInfo.RC:$dst),
                            (ins SrcInfo.ScalarMemOp:$src), OpStr,
                            "${src}"#SrcInfo.BroadcastStr, "${src}"#SrcInfo.BroadcastStr,
                            (DestInfo.VT (MaskNode (SrcInfo.VT
                            (SrcInfo.BroadcastLdFrag addr:$src))))>, EVEX_B,
                            Sched<[sched.Folded, sched.ReadAfterFold]>;
}

// Conversion with rounding control (RC)
multiclass avx10_sat_cvt_rc<bits<8> Opc, string OpStr, X86SchedWriteWidths sched,
                            AVX512VLVectorVTInfo DestInfo, AVX512VLVectorVTInfo SrcInfo,
                            SDNode MaskNode> {
  let Predicates = [HasAVX10_2_512], Uses = [MXCSR] in
  defm Zrrb : AVX512_maskable<Opc, MRMSrcReg, DestInfo.info512,
                              (outs DestInfo.info512.RC:$dst),
                              (ins SrcInfo.info512.RC:$src, AVX512RC:$rc),
                              OpStr, "$rc, $src", "$src, $rc",
                              (DestInfo.info512.VT
                                (MaskNode (SrcInfo.info512.VT SrcInfo.info512.RC:$src),
                                          (i32 timm:$rc)))>,
                             Sched<[sched.ZMM]>, EVEX, EVEX_RC, EVEX_B;
  let Predicates = [HasAVX10_2], hasEVEX_U = 1 in {
  defm Z256rrb : AVX512_maskable<Opc, MRMSrcReg, DestInfo.info256,
                                (outs DestInfo.info256.RC:$dst),
                                (ins SrcInfo.info256.RC:$src, AVX512RC:$rc),
                                OpStr, "$rc, $src", "$src, $rc",
                                (DestInfo.info256.VT
                                  (MaskNode (SrcInfo.info256.VT SrcInfo.info256.RC:$src),
                                            (i32 timm:$rc)))>,
                               Sched<[sched.YMM]>, EVEX, EVEX_RC, EVEX_B;
  }
}

// Conversion with SAE
multiclass avx10_sat_cvt_sae<bits<8> Opc, string OpStr, X86SchedWriteWidths sched,
                             AVX512VLVectorVTInfo DestInfo, AVX512VLVectorVTInfo SrcInfo,
                             SDNode Node> {
  let Predicates = [HasAVX10_2_512], Uses = [MXCSR] in
  defm Zrrb : AVX512_maskable<Opc, MRMSrcReg, DestInfo.info512,
                             (outs DestInfo.info512.RC:$dst),
                             (ins SrcInfo.info512.RC:$src),
                             OpStr, "{sae}, $src", "$src, {sae}",
                             (DestInfo.info512.VT
                               (Node (SrcInfo.info512.VT SrcInfo.info512.RC:$src)))>,
                             Sched<[sched.ZMM]>, EVEX, EVEX_B;
  let Predicates = [HasAVX10_2], hasEVEX_U = 1 in {
  defm Z256rrb : AVX512_maskable<Opc, MRMSrcReg, DestInfo.info256,
                                (outs DestInfo.info256.RC:$dst),
                                (ins SrcInfo.info256.RC:$src),
                                OpStr, "{sae}, $src", "$src, {sae}",
                                (DestInfo.info256.VT
                                  (Node (SrcInfo.info256.VT SrcInfo.info256.RC:$src)))>,
                                Sched<[sched.YMM]>, EVEX, EVEX_B;
  }
}

multiclass avx10_sat_cvt_base<bits<8> Opc, string OpStr, X86SchedWriteWidths sched,
                               SDNode MaskNode, AVX512VLVectorVTInfo DestInfo,
                               AVX512VLVectorVTInfo SrcInfo> {
  let Predicates = [HasAVX10_2_512] in
  defm Z : avx10_sat_cvt_rmb<Opc, OpStr, sched.ZMM,
           DestInfo.info512, SrcInfo.info512,
           MaskNode>,
      EVEX, EVEX_V512;
  let Predicates = [HasAVX10_2] in {
    defm Z256
        : avx10_sat_cvt_rmb<Opc, OpStr, sched.YMM,
           DestInfo.info256, SrcInfo.info256,
          MaskNode>,
          EVEX, EVEX_V256;
    defm Z128
        : avx10_sat_cvt_rmb<Opc, OpStr, sched.XMM,
          DestInfo.info128, SrcInfo.info128,
          MaskNode>,
          EVEX, EVEX_V128;
  }
}

defm VCVTNEBF162IBS : avx10_sat_cvt_base<0x69, "vcvtnebf162ibs",
                                         SchedWriteVecIMul, X86vcvtp2ibs,
                                         avx512vl_i16_info, avx512vl_bf16_info>,
                      AVX512XDIi8Base, T_MAP5, EVEX_CD8<16, CD8VF>;
defm VCVTNEBF162IUBS : avx10_sat_cvt_base<0x6b, "vcvtnebf162iubs",
                                          SchedWriteVecIMul, X86vcvtp2iubs,
                                          avx512vl_i16_info, avx512vl_bf16_info>,
                       AVX512XDIi8Base, T_MAP5, EVEX_CD8<16, CD8VF>;

defm VCVTPH2IBS : avx10_sat_cvt_base<0x69, "vcvtph2ibs", SchedWriteVecIMul,
                                     X86vcvtp2ibs, avx512vl_i16_info,
                                     avx512vl_f16_info>,
                  avx10_sat_cvt_rc<0x69, "vcvtph2ibs", SchedWriteVecIMul,
                                   avx512vl_i16_info, avx512vl_f16_info,
                                   X86vcvtp2ibsRnd>,
                  AVX512PSIi8Base, T_MAP5, EVEX_CD8<16, CD8VF>;
defm VCVTPH2IUBS : avx10_sat_cvt_base<0x6b, "vcvtph2iubs", SchedWriteVecIMul,
                                      X86vcvtp2iubs, avx512vl_i16_info,
                                      avx512vl_f16_info>,
                   avx10_sat_cvt_rc<0x6b, "vcvtph2iubs", SchedWriteVecIMul,
                                    avx512vl_i16_info, avx512vl_f16_info,
                                    X86vcvtp2iubsRnd>,
                   AVX512PSIi8Base, T_MAP5, EVEX_CD8<16, CD8VF>;

defm VCVTPS2IBS : avx10_sat_cvt_base<0x69, "vcvtps2ibs", SchedWriteVecIMul,
                                     X86vcvtp2ibs, avx512vl_i32_info,
                                     avx512vl_f32_info>,
                  avx10_sat_cvt_rc<0x69, "vcvtps2ibs", SchedWriteVecIMul,
                                   avx512vl_i32_info, avx512vl_f32_info,
                                   X86vcvtp2ibsRnd>,
                  AVX512PDIi8Base, T_MAP5, EVEX_CD8<32, CD8VF>;
defm VCVTPS2IUBS : avx10_sat_cvt_base<0x6b, "vcvtps2iubs", SchedWriteVecIMul,
                                      X86vcvtp2iubs, avx512vl_i32_info,
                                      avx512vl_f32_info>,
                   avx10_sat_cvt_rc<0x6b, "vcvtps2iubs", SchedWriteVecIMul,
                                    avx512vl_i32_info, avx512vl_f32_info,
                                    X86vcvtp2iubsRnd>,
                   AVX512PDIi8Base, T_MAP5, EVEX_CD8<32, CD8VF>;

defm VCVTTNEBF162IBS : avx10_sat_cvt_base<0x68, "vcvttnebf162ibs",
                                          SchedWriteVecIMul, X86vcvttp2ibs,
                                          avx512vl_i16_info, avx512vl_bf16_info>,
                       AVX512XDIi8Base, T_MAP5, EVEX_CD8<16, CD8VF>;
defm VCVTTNEBF162IUBS : avx10_sat_cvt_base<0x6a, "vcvttnebf162iubs",
                                           SchedWriteVecIMul, X86vcvttp2iubs,
                                           avx512vl_i16_info, avx512vl_bf16_info>,
                        AVX512XDIi8Base, T_MAP5, EVEX_CD8<16, CD8VF>;

defm VCVTTPH2IBS : avx10_sat_cvt_base<0x68, "vcvttph2ibs", SchedWriteVecIMul,
                                      X86vcvttp2ibs, avx512vl_i16_info,
                                      avx512vl_f16_info>,
                   avx10_sat_cvt_sae<0x68, "vcvttph2ibs", SchedWriteVecIMul,
                                     avx512vl_i16_info, avx512vl_f16_info,
                                     X86vcvttp2ibsSAE>,
                   AVX512PSIi8Base, T_MAP5, EVEX_CD8<16, CD8VF>;
defm VCVTTPH2IUBS : avx10_sat_cvt_base<0x6a, "vcvttph2iubs", SchedWriteVecIMul,
                                       X86vcvttp2iubs, avx512vl_i16_info,
                                       avx512vl_f16_info>,
                    avx10_sat_cvt_sae<0x6a, "vcvttph2iubs", SchedWriteVecIMul,
                                      avx512vl_i16_info, avx512vl_f16_info,
                                      X86vcvttp2iubsSAE>,
                    AVX512PSIi8Base, T_MAP5, EVEX_CD8<16, CD8VF>;

defm VCVTTPS2IBS : avx10_sat_cvt_base<0x68, "vcvttps2ibs", SchedWriteVecIMul,
                                      X86vcvttp2ibs, avx512vl_i32_info,
                                      avx512vl_f32_info>,
                   avx10_sat_cvt_sae<0x68, "vcvttps2ibs", SchedWriteVecIMul,
                                     avx512vl_i32_info, avx512vl_f32_info,
                                     X86vcvttp2ibsSAE>,
                   AVX512PDIi8Base, T_MAP5, EVEX_CD8<32, CD8VF>;
defm VCVTTPS2IUBS : avx10_sat_cvt_base<0x6a, "vcvttps2iubs", SchedWriteVecIMul,
                                       X86vcvttp2iubs, avx512vl_i32_info,
                                       avx512vl_f32_info>,
                    avx10_sat_cvt_sae<0x6a, "vcvttps2iubs", SchedWriteVecIMul,
                                      avx512vl_i32_info, avx512vl_f32_info,
                                      X86vcvttp2iubsSAE>,
                    AVX512PDIi8Base, T_MAP5, EVEX_CD8<32, CD8VF>;

//-------------------------------------------------
// AVX10 CONVERT instructions
//-------------------------------------------------

multiclass avx10_cvt2ps2ph_rc<bits<8> opc, string OpcodeStr, X86FoldableSchedWrite sched,
                              X86VectorVTInfo _Src, X86VectorVTInfo _,
                              SDNode OpNodeRnd> {
  let Uses = [MXCSR] in
    defm rrb : AVX512_maskable<opc, MRMSrcReg, _, (outs _.RC:$dst),
                               (ins _Src.RC:$src1, _Src.RC:$src2, AVX512RC:$rc), OpcodeStr,
                               "$rc, $src2, $src1", "$src1, $src2, $rc",
                               (_.VT (OpNodeRnd (_Src.VT _Src.RC:$src1),
                                                (_Src.VT _Src.RC:$src2), (i32 timm:$rc)))>,
                              EVEX, VVVV, EVEX_B, EVEX_RC, PD, Sched<[sched]>;
}

//TODO: Merge into avx512_binop_all, difference is rounding control added here.
multiclass avx10_cvt2ps2ph<bits<8> opc, string OpcodeStr,
                           X86SchedWriteWidths sched,
                           AVX512VLVectorVTInfo _SrcVTInfo,
                           AVX512VLVectorVTInfo _DstVTInfo,
                           SDNode OpNode, SDNode OpNodeRnd> {
  let Predicates = [HasAVX10_2_512], Uses = [MXCSR] in {
    defm Z : avx512_binop_rm2<opc, OpcodeStr, sched.ZMM, OpNode,
                              _SrcVTInfo.info512, _DstVTInfo.info512,
                              _SrcVTInfo.info512>,
             avx10_cvt2ps2ph_rc<opc, OpcodeStr, sched.ZMM,
                                _SrcVTInfo.info512, _DstVTInfo.info512,
                                OpNodeRnd>,
             EVEX_V512, EVEX_CD8<32, CD8VF>;
  }
  let Predicates = [HasAVX10_2] in {
    defm Z256 : avx512_binop_rm2<opc, OpcodeStr, sched.YMM, OpNode,
                                 _SrcVTInfo.info256, _DstVTInfo.info256,
                                 _SrcVTInfo.info256>,
                                EVEX_V256, EVEX_CD8<32, CD8VF>;
    defm Z128 : avx512_binop_rm2<opc, OpcodeStr, sched.XMM, OpNode,
                                 _SrcVTInfo.info128, _DstVTInfo.info128,
                                 _SrcVTInfo.info128>,
                EVEX_V128, EVEX_CD8<32, CD8VF>;
  }

  let Predicates = [HasAVX10_2], hasEVEX_U = 1 in {
    defm Z256 : avx10_cvt2ps2ph_rc<opc, OpcodeStr, sched.YMM,
                                   _SrcVTInfo.info256, _DstVTInfo.info256,
                                   OpNodeRnd>;
  }
}

defm VCVT2PS2PHX : avx10_cvt2ps2ph<0x67, "vcvt2ps2phx",
                                   SchedWriteCvtPD2PS,
                                   avx512vl_f32_info, avx512vl_f16_info,
                                   X86vfpround2, X86vfpround2Rnd>, T8;

defm VCVTNE2PH2BF8 : avx512_binop_all<0x74, "vcvtne2ph2bf8", SchedWriteCvtPD2PS,
                                     avx512vl_f16_info, avx512vl_i8_info,
                                     X86vcvtne2ph2bf8, [HasAVX10_2_512], [HasAVX10_2]>,
                                    EVEX_CD8<16, CD8VF>, T8, XD;
defm VCVTNE2PH2BF8S : avx512_binop_all<0x74, "vcvtne2ph2bf8s", SchedWriteCvtPD2PS,
                                      avx512vl_f16_info, avx512vl_i8_info,
                                      X86vcvtne2ph2bf8s, [HasAVX10_2_512], [HasAVX10_2]>,
                                     EVEX_CD8<16, CD8VF>, T_MAP5, XD;
defm VCVTNE2PH2HF8 : avx512_binop_all<0x18, "vcvtne2ph2hf8", SchedWriteCvtPD2PS,
                                     avx512vl_f16_info, avx512vl_i8_info,
                                     X86vcvtne2ph2hf8, [HasAVX10_2_512], [HasAVX10_2]>,
                                    EVEX_CD8<16, CD8VF>, T_MAP5, XD;
defm VCVTNE2PH2HF8S : avx512_binop_all<0x1b, "vcvtne2ph2hf8s", SchedWriteCvtPD2PS,
                                      avx512vl_f16_info, avx512vl_i8_info,
                                      X86vcvtne2ph2hf8s, [HasAVX10_2_512], [HasAVX10_2]>,
                                     EVEX_CD8<16, CD8VF>, T_MAP5, XD;

//TODO: Merge into avx512_vcvt_fp, diffrence is one more source register here.
multiclass avx10_convert_3op_packed<bits<8> OpCode, string OpcodeStr,
                                    X86VectorVTInfo vt_dst, X86VectorVTInfo vt_src1,
                                    X86VectorVTInfo vt_src2, SDPatternOperator OpNode,
                                    SDPatternOperator MaskOpNode, X86FoldableSchedWrite sched,
                                    string Broadcast = vt_src2.BroadcastStr,
                                    X86MemOperand MemOp = vt_src2.MemOp,
                                    RegisterClass MaskRC = vt_src2.KRCWM,
                                    dag LdDAG = (vt_dst.VT (OpNode (vt_src1.VT vt_src1.RC:$src1),
                                                           (vt_src2.VT (vt_src2.LdFrag addr:$src2)))),
                                    dag MaskLdDAG = (vt_dst.VT (MaskOpNode (vt_src1.VT vt_src1.RC:$src1),
                                                               (vt_src2.VT (vt_src2.LdFrag addr:$src2))))> {
  defm rr : AVX512_maskable_cvt<OpCode, MRMSrcReg, vt_dst, (outs vt_dst.RC:$dst),
                      (ins vt_src1.RC:$src1, vt_src2.RC:$src2),
                      (ins vt_dst.RC:$src0, MaskRC:$mask, vt_src1.RC:$src1, vt_src2.RC:$src2),
                      (ins MaskRC:$mask, vt_src1.RC:$src1, vt_src2.RC:$src2),
                      OpcodeStr, "$src2, $src1", "$src1, $src2",
                      (vt_dst.VT (OpNode (vt_src1.VT vt_src1.RC:$src1),
                                         (vt_src2.VT vt_src2.RC:$src2))),
                      (vselect_mask MaskRC:$mask,
                        (vt_dst.VT (MaskOpNode (vt_src1.VT vt_src1.RC:$src1),
                        (vt_src2.VT vt_src2.RC:$src2))),
                        vt_dst.RC:$src0),
                      (vselect_mask MaskRC:$mask,
                        (vt_dst.VT (MaskOpNode (vt_src1.VT vt_src1.RC:$src1),
                        (vt_src2.VT vt_src2.RC:$src2))),
                        vt_dst.ImmAllZerosV)>,
                      EVEX, VVVV, Sched<[sched]>;
  let mayLoad = 1 in
  defm rm : AVX512_maskable_cvt<OpCode, MRMSrcMem, vt_dst, (outs vt_dst.RC:$dst),
                      (ins vt_src1.RC:$src1, MemOp:$src2),
                      (ins vt_dst.RC:$src0, MaskRC:$mask, vt_src1.RC:$src1, MemOp:$src2),
                      (ins MaskRC:$mask, vt_src1.RC:$src1, MemOp:$src2),
                      OpcodeStr, "$src2, $src1", "$src1, $src2",
                      LdDAG,
                      (vselect_mask MaskRC:$mask, MaskLdDAG, vt_dst.RC:$src0),
                      (vselect_mask MaskRC:$mask, MaskLdDAG, vt_dst.ImmAllZerosV)>,
                      EVEX, VVVV, Sched<[sched]>;

  let mayLoad = 1 in
  defm rmb : AVX512_maskable_cvt<OpCode, MRMSrcMem, vt_dst, (outs vt_dst.RC:$dst),
                      (ins vt_src1.RC:$src1, vt_src2.ScalarMemOp:$src2),
                      (ins vt_dst.RC:$src0, MaskRC:$mask, vt_src1.RC:$src1,
                           vt_src2.ScalarMemOp:$src2),
                      (ins MaskRC:$mask, vt_src1.RC:$src1, vt_src2.ScalarMemOp:$src2),
                      OpcodeStr,
                      "${src2}"#Broadcast#", $src1", "$src1, ${src2}"#Broadcast,
                      (vt_dst.VT (OpNode (vt_src1.VT vt_src1.RC:$src1), (vt_src2.VT
                                  (vt_src2.BroadcastLdFrag addr:$src2)))),
                      (vselect_mask MaskRC:$mask,
                                       (vt_dst.VT
                                        (MaskOpNode
                                         (vt_src1.VT vt_src1.RC:$src1), (vt_src2.VT
                                          (vt_src2.BroadcastLdFrag addr:$src2)))),
                                       vt_dst.RC:$src0),
                      (vselect_mask MaskRC:$mask,
                                       (vt_dst.VT
                                        (MaskOpNode
                                         (vt_src1.VT vt_src1.RC:$src1),
                                         (vt_src2.VT
                                          (vt_src2.BroadcastLdFrag addr:$src2)))),
                                       vt_dst.ImmAllZerosV)>,
                      EVEX, VVVV, EVEX_B, Sched<[sched]>;
}

//TODO: Merge into avx512_cvt_trunc
multiclass avx10_convert_3op<bits<8> OpCode, string OpcodeStr,
           AVX512VLVectorVTInfo vt_dst, AVX512VLVectorVTInfo vt_src,
           X86SchedWriteWidths sched,
           SDPatternOperator OpNode,
           SDPatternOperator MaskOpNode,
           PatFrag bcast128 = vt_src.info128.BroadcastLdFrag,
           PatFrag loadVT128 = vt_src.info128.LdFrag,
           RegisterClass maskRC128 = vt_src.info128.KRCWM> {
  let Predicates = [HasAVX10_2_512] in
    defm Z : avx10_convert_3op_packed<OpCode, OpcodeStr, vt_dst.info256,
               vt_dst.info512, vt_src.info512, OpNode, OpNode, sched.ZMM>,
               EVEX_V512, EVEX_CD8<16, CD8VF>;
  let Predicates = [HasAVX10_2] in {
    defm Z256 : avx10_convert_3op_packed<OpCode, OpcodeStr, vt_dst.info128,
                  vt_dst.info256, vt_src.info256, OpNode, OpNode, sched.YMM>,
                  EVEX_V256, EVEX_CD8<16, CD8VF>;
    defm Z128 : avx10_convert_3op_packed<OpCode, OpcodeStr, vt_dst.info128,
                  vt_dst.info128, vt_src.info128,
                  null_frag, null_frag, sched.XMM>,
                  EVEX_V128, EVEX_CD8<16, CD8VF>;
    // Special patterns to allow use of MaskOpNode for masking 128 version. Instruction
    // patterns have been disabled with null_frag.
    def : Pat<(vt_dst.info128.VT (OpNode (vt_dst.info128.VT VR128X:$src1),
                                         (vt_src.info128.VT VR128X:$src2))),
              (!cast<Instruction>(NAME # "Z128rr") VR128X:$src1, VR128X:$src2)>;
    def : Pat<(MaskOpNode (vt_dst.info128.VT VR128X:$src1),
                          (vt_src.info128.VT VR128X:$src2),
                          (vt_dst.info128.VT VR128X:$src0), maskRC128:$mask),
              (!cast<Instruction>(NAME # "Z128rrk") VR128X:$src0, maskRC128:$mask,
                          VR128X:$src1, VR128X:$src2)>;
    def : Pat<(MaskOpNode (vt_dst.info128.VT VR128X:$src1),
                          (vt_src.info128.VT VR128X:$src2),
                          vt_dst.info128.ImmAllZerosV, maskRC128:$mask),
              (!cast<Instruction>(NAME # "Z128rrkz") maskRC128:$mask,
                          VR128X:$src1, VR128X:$src2)>;

    def : Pat<(vt_dst.info128.VT (OpNode (vt_dst.info128.VT VR128X:$src1),
                                         (loadVT128 addr:$src2))),
              (!cast<Instruction>(NAME # "Z128rm") VR128X:$src1, addr:$src2)>;
    def : Pat<(MaskOpNode (vt_dst.info128.VT VR128X:$src1),
                          (loadVT128 addr:$src2),
                          (vt_dst.info128.VT VR128X:$src0),
                          maskRC128:$mask),
              (!cast<Instruction>(NAME # "Z128rmk") VR128X:$src0, maskRC128:$mask,
                          VR128X:$src1, addr:$src2)>;
    def : Pat<(MaskOpNode (vt_dst.info128.VT VR128X:$src1),
                          (loadVT128 addr:$src2),
                          vt_dst.info128.ImmAllZerosV,
                          maskRC128:$mask),
              (!cast<Instruction>(NAME # "Z128rmkz") maskRC128:$mask,
                          VR128X:$src1, addr:$src2)>;

    def : Pat<(vt_dst.info128.VT (OpNode (vt_dst.info128.VT VR128X:$src1),
                                         (vt_src.info128.VT (bcast128 addr:$src2)))),
              (!cast<Instruction>(NAME # "Z128rmb") VR128X:$src1, addr:$src2)>;
    def : Pat<(MaskOpNode (vt_dst.info128.VT VR128X:$src1),
                          (vt_src.info128.VT (bcast128 addr:$src2)),
                          (vt_dst.info128.VT VR128X:$src0), maskRC128:$mask),
              (!cast<Instruction>(NAME # "Z128rmbk") VR128X:$src0, maskRC128:$mask,
                           VR128X:$src1, addr:$src2)>;
    def : Pat<(MaskOpNode (vt_dst.info128.VT VR128X:$src1),
                          (vt_src.info128.VT (bcast128 addr:$src2)),
                          vt_dst.info128.ImmAllZerosV, maskRC128:$mask),
              (!cast<Instruction>(NAME # "Z128rmbkz") maskRC128:$mask,
                           VR128X:$src1, addr:$src2)>;
  }
}

defm VCVTBIASPH2BF8 : avx10_convert_3op<0x74, "vcvtbiasph2bf8",
                                        avx512vl_i8_info, avx512vl_f16_info,
                                        SchedWriteCvtPD2PS,
                                        X86vcvtbiasph2bf8, X86vmcvtbiasph2bf8>,
                                        T8, PS;
defm VCVTBIASPH2BF8S : avx10_convert_3op<0x74, "vcvtbiasph2bf8s",
                                         avx512vl_i8_info, avx512vl_f16_info,
                                         SchedWriteCvtPD2PS,
                                         X86vcvtbiasph2bf8s, X86vmcvtbiasph2bf8s>,
                                         T_MAP5, PS;
defm VCVTBIASPH2HF8 : avx10_convert_3op<0x18, "vcvtbiasph2hf8",
                                        avx512vl_i8_info, avx512vl_f16_info,
                                        SchedWriteCvtPD2PS,
                                        X86vcvtbiasph2hf8, X86vmcvtbiasph2hf8>,
                                        T_MAP5, PS;
defm VCVTBIASPH2HF8S : avx10_convert_3op<0x1b, "vcvtbiasph2hf8s",
                                         avx512vl_i8_info, avx512vl_f16_info,
                                         SchedWriteCvtPD2PS,
                                         X86vcvtbiasph2hf8s, X86vmcvtbiasph2hf8s>,
                                         T_MAP5, PS;

defm VCVTNEPH2BF8 : avx512_cvt_trunc_ne<0x74, "vcvtneph2bf8", avx512vl_i8_info,
                                        avx512vl_f16_info, SchedWriteCvtPD2PS,
                                        X86vcvtneph2bf8, X86vmcvtneph2bf8,
                                        [HasAVX10_2], [HasAVX10_2_512]>,
                                        T8, XS, EVEX_CD8<16, CD8VF>;

defm VCVTNEPH2BF8S : avx512_cvt_trunc_ne<0x74, "vcvtneph2bf8s", avx512vl_i8_info,
                                         avx512vl_f16_info, SchedWriteCvtPD2PS,
                                         X86vcvtneph2bf8s, X86vmcvtneph2bf8s,
                                         [HasAVX10_2], [HasAVX10_2_512]>,
                                         T_MAP5, XS, EVEX_CD8<16, CD8VF>;

defm VCVTNEPH2HF8 : avx512_cvt_trunc_ne<0x18, "vcvtneph2hf8", avx512vl_i8_info,
                                        avx512vl_f16_info, SchedWriteCvtPD2PS,
                                        X86vcvtneph2hf8, X86vmcvtneph2hf8,
                                        [HasAVX10_2], [HasAVX10_2_512]>,
                                        T_MAP5, XS, EVEX_CD8<16, CD8VF>;

defm VCVTNEPH2HF8S : avx512_cvt_trunc_ne<0x1b, "vcvtneph2hf8s", avx512vl_i8_info,
                                         avx512vl_f16_info, SchedWriteCvtPD2PS,
                                         X86vcvtneph2hf8s, X86vmcvtneph2hf8s,
                                         [HasAVX10_2], [HasAVX10_2_512]>,
                                         T_MAP5, XS, EVEX_CD8<16, CD8VF>;

multiclass avx10_convert_2op_nomb_packed<bits<8> opc, string OpcodeStr,
                                         X86VectorVTInfo _dest, X86VectorVTInfo _src,
                                         SDNode OpNode, X86MemOperand x86memop,
                                         X86FoldableSchedWrite sched,
                                         dag ld_dag = (load addr:$src)> {
  let ExeDomain = _dest.ExeDomain in {
  defm rr : AVX512_maskable_split<opc, MRMSrcReg, _dest ,(outs _dest.RC:$dst),
                                  (ins _src.RC:$src), OpcodeStr, "$src", "$src",
                                  (OpNode (_src.VT _src.RC:$src)),
                                  (OpNode (_src.VT _src.RC:$src))>,
                                 Sched<[sched]>;
  defm rm : AVX512_maskable_split<opc, MRMSrcMem, _dest, (outs _dest.RC:$dst),
                                  (ins x86memop:$src), OpcodeStr, "$src", "$src",
                                  (OpNode (_src.VT ld_dag)),
                                  (OpNode (_src.VT ld_dag))>,
                                 Sched<[sched.Folded]>;
  }
}

multiclass avx10_convert_2op_nomb<string OpcodeStr, AVX512VLVectorVTInfo _dest,
                                  AVX512VLVectorVTInfo _src, bits<8> opc, SDNode OpNode> {
  let Predicates = [HasAVX10_2_512] in
  defm Z : avx10_convert_2op_nomb_packed<opc, OpcodeStr, _dest.info512, _src.info256,
                                         OpNode, f256mem, WriteCvtPH2PSZ>, EVEX_V512;
  let Predicates = [HasAVX10_2] in {
  defm Z128 : avx10_convert_2op_nomb_packed<opc, OpcodeStr, _dest.info128, _src.info128,
                                            OpNode, f64mem, WriteCvtPH2PSZ>, EVEX_V128;
  defm Z256 : avx10_convert_2op_nomb_packed<opc, OpcodeStr, _dest.info256, _src.info128,
                                            OpNode, f128mem, WriteCvtPH2PSZ>, EVEX_V256;
  }
}

defm VCVTHF82PH : avx10_convert_2op_nomb<"vcvthf82ph", avx512vl_f16_info,
                                         avx512vl_i8_info, 0x1e, X86vcvthf82ph>,
                  AVX512XDIi8Base, T_MAP5, EVEX, EVEX_CD8<16, CD8VH>;

//-------------------------------------------------
// AVX10 BF16 instructions
//-------------------------------------------------

// VADDNEPBF16, VSUBNEPBF16, VMULNEPBF16, VDIVNEPBF16, VMAXPBF16, VMINPBF16
multiclass avx10_fp_binopne_int_pbf16<bits<8> opc, string OpcodeStr,
                                      X86SchedWriteSizes sched,
                                      bit IsCommutable = 0> {
  let Predicates = [HasAVX10_2_512] in
    defm Z : avx512_fp_packed<opc, OpcodeStr,
                              !cast<Intrinsic>("int_x86_avx10_"#OpcodeStr#"pbf16512"),
                              !cast<Intrinsic>("int_x86_avx10_"#OpcodeStr#"pbf16512"),
                              v32bf16_info, sched.PH.ZMM, IsCommutable>, EVEX_V512,
                              T_MAP5, PD, EVEX_CD8<16, CD8VF>;
  let Predicates = [HasAVX10_2] in {
    defm Z128 : avx512_fp_packed<opc, OpcodeStr,
                                 !cast<Intrinsic>("int_x86_avx10_"#OpcodeStr#"pbf16128"),
                                 !cast<Intrinsic>("int_x86_avx10_"#OpcodeStr#"pbf16128"),
                                 v8bf16x_info, sched.PH.XMM, IsCommutable>, EVEX_V128,
                                 T_MAP5, PD, EVEX_CD8<16, CD8VF>;
    defm Z256 : avx512_fp_packed<opc, OpcodeStr,
                                 !cast<Intrinsic>("int_x86_avx10_"#OpcodeStr#"pbf16256"),
                                 !cast<Intrinsic>("int_x86_avx10_"#OpcodeStr#"pbf16256"),
                                 v16bf16x_info, sched.PH.YMM, IsCommutable>, EVEX_V256,
                                 T_MAP5, PD, EVEX_CD8<16, CD8VF>;
  }
}

multiclass avx10_fp_binop_pbf16<bits<8> opc, string OpcodeStr, SDPatternOperator OpNode,
                                X86SchedWriteSizes sched,
                                bit IsCommutable = 0,
                                SDPatternOperator MaskOpNode = OpNode> {
  let Predicates = [HasAVX10_2_512] in
    defm Z : avx512_fp_packed<opc, OpcodeStr, OpNode, MaskOpNode, 
                              v32bf16_info, sched.PH.ZMM, IsCommutable>, EVEX_V512,
                              T_MAP5, PD, EVEX_CD8<16, CD8VF>;
  let Predicates = [HasAVX10_2] in {
    defm Z128 : avx512_fp_packed<opc, OpcodeStr, OpNode, MaskOpNode, 
                                 v8bf16x_info, sched.PH.XMM, IsCommutable>, EVEX_V128,
                                 T_MAP5, PD, EVEX_CD8<16, CD8VF>;
    defm Z256 : avx512_fp_packed<opc, OpcodeStr, OpNode, MaskOpNode, 
                                 v16bf16x_info, sched.PH.YMM, IsCommutable>, EVEX_V256,
                                 T_MAP5, PD, EVEX_CD8<16, CD8VF>;
  }
}

let Uses = []<Register>, mayRaiseFPException = 0 in {
defm VADDNEPBF16 : avx10_fp_binop_pbf16<0x58, "vaddne", fadd, SchedWriteFAddSizes, 1>;
defm VSUBNEPBF16 : avx10_fp_binop_pbf16<0x5C, "vsubne", fsub, SchedWriteFAddSizes, 0>;
defm VMULNEPBF16 : avx10_fp_binop_pbf16<0x59, "vmulne", fmul, SchedWriteFMulSizes, 1>;
defm VDIVNEPBF16 : avx10_fp_binop_pbf16<0x5E, "vdivne", fdiv, SchedWriteFDivSizes, 0>;
defm VMINPBF16 : avx10_fp_binopne_int_pbf16<0x5D, "vmin", SchedWriteFCmpSizes, 0>;
defm VMAXPBF16 : avx10_fp_binopne_int_pbf16<0x5F, "vmax", SchedWriteFCmpSizes, 0>;
}

// VCOMSBF16
let Uses = []<Register>, mayRaiseFPException = 0,
  Defs = [EFLAGS], Predicates = [HasAVX10_2] in {
  //TODO: Replace null_frag with X86fcmp to support lowering `fcmp oeq bfloat *`
  //which may require extend supports on BFR16X, loadbf16, ...
  defm VCOMSBF16Z : sse12_ord_cmp<0x2F, FR16X, null_frag, bf16, f16mem, loadf16,
                                  "comsbf16", SSEPackedSingle>, T_MAP5, PD, EVEX,
                                  VEX_LIG, EVEX_CD8<16, CD8VT1>;

  let isCodeGenOnly = 1 in {
    defm VCOMSBF16Z : sse12_ord_cmp_int<0x2F, VR128X, X86comi, v8bf16, f16mem,
                                        sse_load_bf16, "comsbf16", SSEPackedSingle>,
                                        T_MAP5, PD, EVEX, VEX_LIG, EVEX_CD8<16, CD8VT1>;
  }
}

// VCMPPBF16
multiclass avx10_vcmp_common_bf16<X86FoldableSchedWrite sched, X86VectorVTInfo _> {
  let mayRaiseFPException = 0 in {
  defm rri  : AVX512_maskable_cmp<0xC2, MRMSrcReg, _,
                   (outs _.KRC:$dst), (ins _.RC:$src1, _.RC:$src2, u8imm:$cc),
                   "vcmp"#_.Suffix,
                   "$cc, $src2, $src1", "$src1, $src2, $cc",
                   (X86cmpm (_.VT _.RC:$src1), (_.VT _.RC:$src2), timm:$cc),
                   (X86cmpm_su (_.VT _.RC:$src1), (_.VT _.RC:$src2), timm:$cc),
                   1>, Sched<[sched]>;

  defm rmi  : AVX512_maskable_cmp<0xC2, MRMSrcMem, _,
                (outs _.KRC:$dst),(ins _.RC:$src1, _.MemOp:$src2, u8imm:$cc),
                "vcmp"#_.Suffix,
                "$cc, $src2, $src1", "$src1, $src2, $cc",
                (X86cmpm (_.VT _.RC:$src1), (_.VT (_.LdFrag addr:$src2)),
                         timm:$cc),
                (X86cmpm_su (_.VT _.RC:$src1), (_.VT (_.LdFrag addr:$src2)),
                            timm:$cc)>,
                Sched<[sched.Folded, sched.ReadAfterFold]>;

  defm rmbi : AVX512_maskable_cmp<0xC2, MRMSrcMem, _,
                (outs _.KRC:$dst),
                (ins _.RC:$src1, _.ScalarMemOp:$src2, u8imm:$cc),
                "vcmp"#_.Suffix,
                "$cc, ${src2}"#_.BroadcastStr#", $src1",
                "$src1, ${src2}"#_.BroadcastStr#", $cc",
                (X86cmpm (_.VT _.RC:$src1),
                         (_.VT (_.BroadcastLdFrag addr:$src2)),
                         timm:$cc),
                (X86cmpm_su (_.VT _.RC:$src1),
                            (_.VT (_.BroadcastLdFrag addr:$src2)),
                            timm:$cc)>,
                EVEX_B, Sched<[sched.Folded, sched.ReadAfterFold]>;
  }
}

multiclass avx10_vcmp_bf16<X86SchedWriteWidths sched, AVX512VLVectorVTInfo _> {
  let Predicates = [HasAVX10_2_512] in
    defm Z : avx10_vcmp_common_bf16<sched.ZMM, _.info512>, EVEX_V512;
  let Predicates = [HasAVX10_2] in {
    defm Z128 : avx10_vcmp_common_bf16<sched.XMM, _.info128>, EVEX_V128;
    defm Z256 : avx10_vcmp_common_bf16<sched.YMM, _.info256>, EVEX_V256;
  }
}

defm VCMPPBF16 : avx10_vcmp_bf16<SchedWriteFCmp, avx512vl_bf16_info>,
                                 AVX512XDIi8Base, EVEX, VVVV,
                                 EVEX_CD8<16, CD8VF>, TA;


// VSQRTNEPBF16
multiclass avx10_sqrt_packed_bf16<bits<8> opc, string OpcodeStr,
                                  X86SchedWriteSizes sched> {
  let Predicates = [HasAVX10_2_512] in
  defm Z : avx512_sqrt_packed<opc, !strconcat(OpcodeStr, "pbf16"),
                              sched.PH.ZMM, v32bf16_info>,
                              EVEX_V512, PD, T_MAP5, EVEX_CD8<16, CD8VF>;
  let Predicates = [HasAVX10_2] in {
    defm Z128 : avx512_sqrt_packed<opc, !strconcat(OpcodeStr, "pbf16"),
                                   sched.PH.XMM, v8bf16x_info>,
                                   EVEX_V128, PD, T_MAP5, EVEX_CD8<16, CD8VF>;
    defm Z256 : avx512_sqrt_packed<opc, !strconcat(OpcodeStr, "pbf16"),
                                   sched.PH.YMM, v16bf16x_info>,
                                   EVEX_V256, PD, T_MAP5, EVEX_CD8<16, CD8VF>;
  }
}

let Uses = []<Register>, mayRaiseFPException = 0 in
defm VSQRTNEPBF16 : avx10_sqrt_packed_bf16<0x51, "vsqrtne", SchedWriteFSqrtSizes>;

// VRSQRTPBF16, VRCPPBF16, VSRQTPBF16, VGETEXPPBF16
multiclass avx10_fp14_pbf16<bits<8> opc, string OpcodeStr, SDNode OpNode,
                            X86SchedWriteWidths sched> {
  let Predicates = [HasAVX10_2_512] in
  defm PBF16Z : avx512_fp14_p<opc, !strconcat(OpcodeStr, "pbf16"),
                              OpNode, sched.ZMM, v32bf16_info>,
                              EVEX_V512;
  let Predicates = [HasAVX10_2] in {
    defm PBF16Z128 : avx512_fp14_p<opc, !strconcat(OpcodeStr, "pbf16"),
                                   OpNode, sched.XMM, v8bf16x_info>,
                                   EVEX_V128;
    defm PBF16Z256 : avx512_fp14_p<opc, !strconcat(OpcodeStr, "pbf16"),
                                   OpNode, sched.YMM, v16bf16x_info>,
                                   EVEX_V256;
  }
}

defm VRSQRT  : avx10_fp14_pbf16<0x4E, "vrsqrt", X86rsqrt14, SchedWriteFRsqrt>,
                                T_MAP6, PS, EVEX_CD8<16, CD8VF>;
defm VRCP    : avx10_fp14_pbf16<0x4C, "vrcp", X86rcp14, SchedWriteFRcp>,
                                T_MAP6, PS, EVEX_CD8<16, CD8VF>;
defm VGETEXP : avx10_fp14_pbf16<0x42, "vgetexp", X86fgetexp, SchedWriteFRnd>,
                                T_MAP5, EVEX_CD8<16, CD8VF>;

// VSCALEFPBF16
multiclass avx10_fp_scalef_bf16<bits<8> opc, string OpcodeStr,
                                X86SchedWriteWidths sched> {
  let Predicates = [HasAVX10_2_512] in
    defm Z : avx512_fp_scalef_p<opc, OpcodeStr, X86scalef, sched.ZMM, v32bf16_info>,
                                EVEX_V512, T_MAP6, PS, EVEX_CD8<16, CD8VF>;
  let Predicates = [HasAVX10_2] in {
    defm Z128 : avx512_fp_scalef_p<opc, OpcodeStr, X86scalef, sched.XMM, v8bf16x_info>,
                                   EVEX_V128, EVEX_CD8<16, CD8VF>, T_MAP6, PS;
    defm Z256 : avx512_fp_scalef_p<opc, OpcodeStr, X86scalef, sched.YMM, v16bf16x_info>,
                                   EVEX_V256, EVEX_CD8<16, CD8VF>, T_MAP6, PS;
  }
}

let Uses = []<Register>, mayRaiseFPException = 0 in
defm VSCALEFPBF16 : avx10_fp_scalef_bf16<0x2C, "vscalef", SchedWriteFAdd>;

// VREDUCENEPBF16, VRNDSCALENEPBF16, VGETMANTPBF16
multiclass avx10_common_unary_fp_packed_imm_bf16<string OpcodeStr,
            AVX512VLVectorVTInfo _, bits<8> opc, SDPatternOperator OpNode,
            SDPatternOperator MaskOpNode, X86SchedWriteWidths sched> {
  let Predicates = [HasAVX10_2_512] in
    defm Z    : avx512_unary_fp_packed_imm<opc, OpcodeStr, OpNode, MaskOpNode,
                                           sched.ZMM, _.info512>, EVEX_V512;
  let Predicates = [HasAVX10_2] in {
    defm Z128 : avx512_unary_fp_packed_imm<opc, OpcodeStr, OpNode, MaskOpNode,
                                           sched.XMM, _.info128>, EVEX_V128;
    defm Z256 : avx512_unary_fp_packed_imm<opc, OpcodeStr, OpNode, MaskOpNode,
                                           sched.YMM, _.info256>, EVEX_V256;
  }
}

let Uses = []<Register>, mayRaiseFPException = 0 in {
defm VREDUCENEPBF16 : avx10_common_unary_fp_packed_imm_bf16<"vreducene", avx512vl_bf16_info, 0x56,
                            X86VReduce, X86VReduce, SchedWriteFRnd>,
                            AVX512XDIi8Base, TA, EVEX, EVEX_CD8<16, CD8VF>;
defm VRNDSCALENEPBF16 : avx10_common_unary_fp_packed_imm_bf16<"vrndscalene", avx512vl_bf16_info, 0x08,
                            X86any_VRndScale, X86VRndScale, SchedWriteFRnd>,
                            AVX512XDIi8Base, TA, EVEX, EVEX_CD8<16, CD8VF>;
defm VGETMANTPBF16 : avx10_common_unary_fp_packed_imm_bf16<"vgetmant", avx512vl_bf16_info, 0x26,
                            X86VGetMant, X86VGetMant, SchedWriteFRnd>,
                            AVX512XDIi8Base, TA, EVEX, EVEX_CD8<16, CD8VF>;
}

// VFPCLASSPBF16
multiclass avx10_fp_fpclass_bf16<string OpcodeStr, bits<8> opcVec,
                                  X86SchedWriteWidths sched> {
  let Predicates = [HasAVX10_2_512] in
    defm Z : avx512_vector_fpclass<opcVec, OpcodeStr, sched.ZMM,
                                   avx512vl_bf16_info.info512, "z",
                                   []<Register>>, EVEX_V512;
  let Predicates = [HasAVX10_2] in {
    defm Z128 : avx512_vector_fpclass<opcVec, OpcodeStr, sched.XMM,
                                      avx512vl_bf16_info.info128, "x",
                                      []<Register>>, EVEX_V128;
    defm Z256 : avx512_vector_fpclass<opcVec, OpcodeStr, sched.YMM,
                                      avx512vl_bf16_info.info256, "y",
                                      []<Register>>, EVEX_V256;
  }
}

defm VFPCLASSPBF16 : avx10_fp_fpclass_bf16<"vfpclass", 0x66, SchedWriteFCmp>,
                                      AVX512XDIi8Base, TA, EVEX, EVEX_CD8<16, CD8VF>;

// VF[,N]M[ADD,SUB][132,213,231]NEPBF16
multiclass avx10_fma3p_213_bf16<bits<8> opc, string OpcodeStr,
                                 SDPatternOperator OpNode, SDNode MaskOpNode,
                                 X86SchedWriteWidths sched> {
  let Predicates = [HasAVX10_2_512] in
    defm Z : avx512_fma3p_213_rm<opc, OpcodeStr, OpNode, MaskOpNode,
                               sched.ZMM, v32bf16_info>, EVEX_V512, T_MAP6, PS,
                               EVEX_CD8<16, CD8VF>;
  let Predicates = [HasAVX10_2] in {
    defm Z128 : avx512_fma3p_213_rm<opc, OpcodeStr, OpNode, MaskOpNode,
                               sched.XMM, v8bf16x_info>, EVEX_V128, T_MAP6, PS,
                               EVEX_CD8<16, CD8VF>;
    defm Z256 : avx512_fma3p_213_rm<opc, OpcodeStr, OpNode, MaskOpNode,
                               sched.YMM, v16bf16x_info>, EVEX_V256, T_MAP6, PS,
                               EVEX_CD8<16, CD8VF>;
  }
}

let Uses = []<Register>, mayRaiseFPException = 0 in {
defm VFMADD213NEPBF16 : avx10_fma3p_213_bf16<0xA8, "vfmadd213nepbf16", any_fma,
                                         fma, SchedWriteFMA>;
defm VFMSUB213NEPBF16 : avx10_fma3p_213_bf16<0xAA, "vfmsub213nepbf16", X86any_Fmsub,
                                         X86Fmsub, SchedWriteFMA>;
defm VFNMADD213NEPBF16 : avx10_fma3p_213_bf16<0xAC, "vfnmadd213nepbf16", X86any_Fnmadd,
                                          X86Fnmadd, SchedWriteFMA>;
defm VFNMSUB213NEPBF16 : avx10_fma3p_213_bf16<0xAE, "vfnmsub213nepbf16", X86any_Fnmsub,
                                          X86Fnmsub, SchedWriteFMA>;
}

multiclass avx10_fma3p_231_bf16<bits<8> opc, string OpcodeStr,
                                 SDPatternOperator OpNode, SDNode MaskOpNode,
                                 X86SchedWriteWidths sched> {
  let Predicates = [HasAVX10_2_512] in
    defm Z : avx512_fma3p_231_rm<opc, OpcodeStr, OpNode, MaskOpNode,
                               sched.ZMM, v32bf16_info>, EVEX_V512, T_MAP6, PS,
                               EVEX_CD8<16, CD8VF>;
  let Predicates = [HasAVX10_2] in {
    defm Z128 : avx512_fma3p_231_rm<opc, OpcodeStr, OpNode, MaskOpNode,
                               sched.XMM, v8bf16x_info>, EVEX_V128, T_MAP6, PS,
                               EVEX_CD8<16, CD8VF>;
    defm Z256 : avx512_fma3p_231_rm<opc, OpcodeStr, OpNode, MaskOpNode,
                               sched.YMM, v16bf16x_info>, EVEX_V256, T_MAP6, PS,
                               EVEX_CD8<16, CD8VF>;
  }
}

let Uses = []<Register>, mayRaiseFPException = 0 in {
defm VFMADD231NEPBF16 : avx10_fma3p_231_bf16<0xB8, "vfmadd231nepbf16", any_fma,
                                         fma, SchedWriteFMA>;
defm VFMSUB231NEPBF16 : avx10_fma3p_231_bf16<0xBA, "vfmsub231nepbf16", X86any_Fmsub,
                                         X86Fmsub, SchedWriteFMA>;
defm VFNMADD231NEPBF16 : avx10_fma3p_231_bf16<0xBC, "vfnmadd231nepbf16", X86any_Fnmadd,
                                          X86Fnmadd, SchedWriteFMA>;
defm VFNMSUB231NEPBF16 : avx10_fma3p_231_bf16<0xBE, "vfnmsub231nepbf16", X86any_Fnmsub,
                                          X86Fnmsub, SchedWriteFMA>;
}

multiclass avx10_fma3p_132_bf16<bits<8> opc, string OpcodeStr,
                                 SDPatternOperator OpNode, SDNode MaskOpNode,
                                 X86SchedWriteWidths sched> {
  let Predicates = [HasAVX10_2_512] in
    defm Z : avx512_fma3p_132_rm<opc, OpcodeStr, OpNode, MaskOpNode,
                                 sched.ZMM, v32bf16_info>, EVEX_V512, T_MAP6, PS,
                                 EVEX_CD8<16, CD8VF>;
  let Predicates = [HasAVX10_2] in {
    defm Z128 : avx512_fma3p_132_rm<opc, OpcodeStr, OpNode, MaskOpNode,
                                    sched.XMM, v8bf16x_info>, EVEX_V128, T_MAP6, PS,
                                    EVEX_CD8<16, CD8VF>;
    defm Z256 : avx512_fma3p_132_rm<opc, OpcodeStr, OpNode, MaskOpNode,
                                    sched.YMM, v16bf16x_info>, EVEX_V256, T_MAP6, PS,
                                    EVEX_CD8<16, CD8VF>;
  }
}

let Uses = []<Register>, mayRaiseFPException = 0 in {
defm VFMADD132NEPBF16 : avx10_fma3p_132_bf16<0x98, "vfmadd132nepbf16", any_fma,
                                             fma, SchedWriteFMA>;
defm VFMSUB132NEPBF16 : avx10_fma3p_132_bf16<0x9A, "vfmsub132nepbf16", X86any_Fmsub,
                                             X86Fmsub, SchedWriteFMA>;
defm VFNMADD132NEPBF16 : avx10_fma3p_132_bf16<0x9C, "vfnmadd132nepbf16", X86any_Fnmadd,
                                              X86Fnmadd, SchedWriteFMA>;
defm VFNMSUB132NEPBF16 : avx10_fma3p_132_bf16<0x9E, "vfnmsub132nepbf16", X86any_Fnmsub,
                                              X86Fnmsub, SchedWriteFMA>;
}

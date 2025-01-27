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
      let isCodeGenOnly = 1 in {
        def rri : AVX512Ii8<0x53, MRMSrcReg, (outs _.FRC:$dst),
                            (ins _.FRC:$src1, _.FRC:$src2, i32u8imm:$src3),
                             !strconcat(OpStr, "\t{$src3, $src2, $src1|$src1, $src2, $src3}"),
                             [(set _.FRC:$dst, (OpNode _.FRC:$src1, _.FRC:$src2, (i32 timm:$src3)))]>,
                       Sched<[WriteFMAX]>;

        def rmi : AVX512Ii8<0x53, MRMSrcMem, (outs _.FRC:$dst),
                            (ins _.FRC:$src1, _.ScalarMemOp:$src2, i32u8imm:$src3),
                             !strconcat(OpStr, "\t{$src3, $src2, $src1|$src1, $src2, $src3}"),
                             [(set _.FRC:$dst, (OpNode _.FRC:$src1, (_.ScalarLdFrag addr:$src2),
                                                       (i32 timm:$src3)))]>,
                       Sched<[WriteFMAX.Folded, WriteFMAX.ReadAfterFold]>;
      }
      defm rri : AVX512_maskable<0x53, MRMSrcReg, _, (outs VR128X:$dst),
                                 (ins VR128X:$src1, VR128X:$src2, i32u8imm:$src3),
                                  OpStr, "$src3, $src2, $src1", "$src1, $src2, $src3",
                                  (_.VT (OpNode (_.VT _.RC:$src1), (_.VT _.RC:$src2),
                                                (i32 timm:$src3))),
                                 0, 0, 0, vselect_mask, "", "_Int">,
                       Sched<[WriteFMAX]>;

      defm rmi : AVX512_maskable<0x53, MRMSrcMem, _, (outs VR128X:$dst),
                                 (ins VR128X:$src1, _.ScalarMemOp:$src2, i32u8imm:$src3),
                                  OpStr, "$src3, $src2, $src1", "$src1, $src2, $src3",
                                  (_.VT (OpNode (_.VT _.RC:$src1), (_.ScalarIntMemFrags addr:$src2),
                                                (i32 timm:$src3))),
                                 0, 0, 0, vselect_mask, "", "_Int">,
                       Sched<[WriteFMAX.Folded, WriteFMAX.ReadAfterFold]>;
    }
    let Uses = []<Register>, mayRaiseFPException = 0 in
      defm rrib : AVX512_maskable<0x53, MRMSrcReg, _, (outs VR128X:$dst),
                                  (ins VR128X:$src1, VR128X:$src2, i32u8imm:$src3),
                                   OpStr, "$src3, {sae}, $src2, $src1",
                                   "$src1, $src2, {sae}, $src3",
                                   (_.VT (OpNodeSAE (_.VT _.RC:$src1), (_.VT _.RC:$src2),
                                                    (i32 timm:$src3))),
                                  0, 0, 0, vselect_mask, "", "_Int">,
                       Sched<[WriteFMAX]>, EVEX_B;
  }
}


let mayRaiseFPException = 0 in
defm VMINMAXBF16 : avx10_minmax_packed<"vminmaxbf16", avx512vl_bf16_info, X86vminmax>,
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

defm VCVTBF162IBS : avx10_sat_cvt_base<0x69, "vcvtbf162ibs",
                                       SchedWriteVecIMul, X86vcvtp2ibs,
                                       avx512vl_i16_info, avx512vl_bf16_info>,
                      AVX512XDIi8Base, T_MAP5, EVEX_CD8<16, CD8VF>;
defm VCVTBF162IUBS : avx10_sat_cvt_base<0x6b, "vcvtbf162iubs",
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

defm VCVTTBF162IBS : avx10_sat_cvt_base<0x68, "vcvttbf162ibs",
                                        SchedWriteVecIMul, X86vcvttp2ibs,
                                        avx512vl_i16_info, avx512vl_bf16_info>,
                       AVX512XDIi8Base, T_MAP5, EVEX_CD8<16, CD8VF>;
defm VCVTTBF162IUBS : avx10_sat_cvt_base<0x6a, "vcvttbf162iubs",
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
// AVX10 SATCVT-DS instructions
//-------------------------------------------------

// Convert Double to Signed/Unsigned Doubleword with truncation.
multiclass avx10_cvttpd2dqs<bits<8> opc, string OpcodeStr, SDPatternOperator OpNode,
                            SDNode MaskOpNode, SDNode OpNodeSAE,
                            X86SchedWriteWidths sched> {
  let Predicates = [HasAVX10_2_512] in {
    defm Z : avx512_vcvt_fp<opc, OpcodeStr, v8i32x_info, v8f64_info, OpNode,
                            MaskOpNode, sched.ZMM>,
             avx512_vcvt_fp_sae<opc, OpcodeStr, v8i32x_info, v8f64_info,
                               OpNodeSAE, sched.ZMM>, EVEX_V512;
  }
  let Predicates = [HasAVX10_2] in {
    defm Z128 : avx512_vcvt_fp<opc, OpcodeStr, v4i32x_info, v2f64x_info,
                              null_frag, null_frag, sched.XMM, "{1to2}", "{x}",
                              f128mem, VK2WM>, EVEX_V128;
    defm Z256 : avx512_vcvt_fp<opc, OpcodeStr, v4i32x_info, v4f64x_info, OpNode,
                              MaskOpNode, sched.YMM, "{1to4}", "{y}">, EVEX_V256;
  }

  let Predicates = [HasAVX10_2], hasEVEX_U=1 in {
    defm Z256 : avx512_vcvt_fp_sae<opc, OpcodeStr, v4i32x_info, v4f64x_info, OpNodeSAE,
                                  sched.YMM>, EVEX_V256;
  }


  def : InstAlias<OpcodeStr#"x\t{$src, $dst|$dst, $src}",
                  (!cast<Instruction>(NAME # "Z128rr") VR128X:$dst,
                   VR128X:$src), 0, "att">;
  def : InstAlias<OpcodeStr#"x\t{$src, $dst {${mask}}|$dst {${mask}}, $src}",
                  (!cast<Instruction>(NAME # "Z128rrk") VR128X:$dst,
                   VK2WM:$mask, VR128X:$src), 0, "att">;
  def : InstAlias<OpcodeStr#"x\t{$src, $dst {${mask}} {z}|$dst {${mask}} {z}, $src}",
                  (!cast<Instruction>(NAME # "Z128rrkz") VR128X:$dst,
                   VK2WM:$mask, VR128X:$src), 0, "att">;
  def : InstAlias<OpcodeStr#"x\t{${src}{1to2}, $dst|$dst, ${src}{1to2}}",
                  (!cast<Instruction>(NAME # "Z128rmb") VR128X:$dst,
                   f64mem:$src), 0, "att">;
  def : InstAlias<OpcodeStr#"x\t{${src}{1to2}, $dst {${mask}}|"
                  "$dst {${mask}}, ${src}{1to2}}",
                  (!cast<Instruction>(NAME # "Z128rmbk") VR128X:$dst,
                   VK2WM:$mask, f64mem:$src), 0, "att">;
  def : InstAlias<OpcodeStr#"x\t{${src}{1to2}, $dst {${mask}} {z}|"
                  "$dst {${mask}} {z}, ${src}{1to2}}",
                  (!cast<Instruction>(NAME # "Z128rmbkz") VR128X:$dst,
                   VK2WM:$mask, f64mem:$src), 0, "att">;

  def : InstAlias<OpcodeStr#"y\t{$src, $dst|$dst, $src}",
                  (!cast<Instruction>(NAME # "Z256rr") VR128X:$dst,
                   VR256X:$src), 0, "att">;
  def : InstAlias<OpcodeStr#"y\t{{sae} $src, $dst|$dst, $src {sae}}",
                  (!cast<Instruction>(NAME # "Z256rrb") VR128X:$dst,
                   VR256X:$src), 0, "att">;
  def : InstAlias<OpcodeStr#"y\t{$src, $dst {${mask}}|$dst {${mask}}, $src}",
                  (!cast<Instruction>(NAME # "Z256rrk") VR128X:$dst,
                   VK4WM:$mask, VR256X:$src), 0, "att">;
  def : InstAlias<OpcodeStr#"y\t{{sae} $src, $dst {${mask}}|$dst {${mask}}, $src {sae}}",
                  (!cast<Instruction>(NAME # "Z256rrbk") VR128X:$dst,
                   VK4WM:$mask, VR256X:$src), 0, "att">;
  def : InstAlias<OpcodeStr#"y\t{$src, $dst {${mask}} {z}|$dst {${mask}} {z}, $src}",
                  (!cast<Instruction>(NAME # "Z256rrkz") VR128X:$dst,
                   VK4WM:$mask, VR256X:$src), 0, "att">;
  def : InstAlias<OpcodeStr#"y\t{{sae} $src, $dst {${mask}} {z}|$dst {${mask}} {z}, $src {sae}}",
                  (!cast<Instruction>(NAME # "Z256rrbkz") VR128X:$dst,
                   VK4WM:$mask, VR256X:$src), 0, "att">;
  def : InstAlias<OpcodeStr#"y\t{${src}{1to4}, $dst|$dst, ${src}{1to4}}",
                  (!cast<Instruction>(NAME # "Z256rmb") VR128X:$dst,
                   f64mem:$src), 0, "att">;
  def : InstAlias<OpcodeStr#"y\t{${src}{1to4}, $dst {${mask}}|"
                  "$dst {${mask}}, ${src}{1to4}}",
                  (!cast<Instruction>(NAME # "Z256rmbk") VR128X:$dst,
                   VK4WM:$mask, f64mem:$src), 0, "att">;
  def : InstAlias<OpcodeStr#"y\t{${src}{1to4}, $dst {${mask}} {z}|"
                  "$dst {${mask}} {z}, ${src}{1to4}}",
                  (!cast<Instruction>(NAME # "Z256rmbkz") VR128X:$dst,
                   VK4WM:$mask, f64mem:$src), 0, "att">;
}

// Convert Double to Signed/Unsigned Quardword with truncation saturationn enabled
multiclass avx10_cvttpd2qqs<bits<8> opc, string OpcodeStr, SDPatternOperator OpNode,
                            SDNode MaskOpNode, SDNode OpNodeRnd,
                            X86SchedWriteWidths sched> {
   let Predicates = [HasAVX10_2_512] in {
     defm Z : avx512_vcvt_fp<opc, OpcodeStr, v8i64_info, v8f64_info, OpNode,
                            MaskOpNode, sched.ZMM>,
              avx512_vcvt_fp_sae<opc, OpcodeStr, v8i64_info, v8f64_info,
                                OpNodeRnd, sched.ZMM>, EVEX_V512;
   }
   let Predicates = [HasAVX10_2] in {
     defm Z128 : avx512_vcvt_fp<opc, OpcodeStr, v2i64x_info, v2f64x_info, OpNode,
                               MaskOpNode, sched.XMM>, EVEX_V128;
     defm Z256 : avx512_vcvt_fp<opc, OpcodeStr, v4i64x_info, v4f64x_info, OpNode,
                               MaskOpNode, sched.YMM>, EVEX_V256;
   }
   let Predicates = [HasAVX10_2], hasEVEX_U=1 in {
     defm Z256 : avx512_vcvt_fp_sae<opc, OpcodeStr, v4i64x_info, v4f64x_info,
                                   OpNodeRnd, sched.YMM>, EVEX_V256;
   }
}

// Convert Float to Signed/Unsigned Quardword with truncation
multiclass avx10_cvttps2qqs<bits<8> opc, string OpcodeStr, SDPatternOperator OpNode,
                            SDNode MaskOpNode, SDNode OpNodeRnd,
                            X86SchedWriteWidths sched> {
  let Predicates = [HasAVX10_2_512] in {
    defm Z : avx512_vcvt_fp<opc, OpcodeStr, v8i64_info, v8f32x_info, OpNode,
                           MaskOpNode, sched.ZMM>,
             avx512_vcvt_fp_sae<opc, OpcodeStr, v8i64_info, v8f32x_info,
                               OpNodeRnd, sched.ZMM>, EVEX_V512;
  }
  let Predicates = [HasAVX10_2] in {
    defm Z128 : avx512_vcvt_fp<opc, OpcodeStr, v2i64x_info, v4f32x_info, OpNode,
                              MaskOpNode, sched.XMM, "{1to2}", "", f64mem, VK2WM,
                              (v2i64 (OpNode (bc_v4f32 (v2f64
                               (scalar_to_vector (loadf64 addr:$src)))))),
                              (v2i64 (MaskOpNode (bc_v4f32 (v2f64
                               (scalar_to_vector (loadf64 addr:$src))))))>,
                              EVEX_V128;
    defm Z256 : avx512_vcvt_fp<opc, OpcodeStr, v4i64x_info, v4f32x_info, OpNode,
                              MaskOpNode, sched.YMM>, EVEX_V256;
  }

  let Predicates = [HasAVX10_2], hasEVEX_U=1 in {
    defm Z256 : avx512_vcvt_fp_sae<opc, OpcodeStr, v4i64x_info, v4f32x_info, OpNodeRnd,
                                  sched.YMM>, EVEX_V256;
  }
}

// Convert Float to Signed/Unsigned Doubleword with truncation
multiclass avx10_cvttps2dqs<bits<8> opc, string OpcodeStr, SDPatternOperator OpNode,
                            SDNode MaskOpNode,
                            SDNode OpNodeSAE, X86SchedWriteWidths sched> {
  let Predicates = [HasAVX10_2_512] in {
    defm Z : avx512_vcvt_fp<opc, OpcodeStr, v16i32_info, v16f32_info, OpNode,
                           MaskOpNode, sched.ZMM>,
             avx512_vcvt_fp_sae<opc, OpcodeStr, v16i32_info, v16f32_info,
                                OpNodeSAE, sched.ZMM>, EVEX_V512;
  }

  let Predicates = [HasAVX10_2] in {
    defm Z128 : avx512_vcvt_fp<opc, OpcodeStr, v4i32x_info, v4f32x_info, OpNode,
                              MaskOpNode, sched.XMM>, EVEX_V128;
    defm Z256 : avx512_vcvt_fp<opc, OpcodeStr, v8i32x_info, v8f32x_info, OpNode,
                              MaskOpNode, sched.YMM>, EVEX_V256;
  }

  let Predicates = [HasAVX10_2], hasEVEX_U=1 in {
    defm Z256 : avx512_vcvt_fp_sae<opc, OpcodeStr, v8i32x_info, v8f32x_info,
                                  OpNodeSAE, sched.YMM>, EVEX_V256;
  }
}

defm VCVTTPD2DQS : avx10_cvttpd2dqs<0x6D, "vcvttpd2dqs", X86cvttp2sis,
                                    X86cvttp2sis, X86cvttp2sisSAE,
                                    SchedWriteCvtPD2DQ>,
                                    PD, REX_W, T_MAP5,PS, EVEX_CD8<64, CD8VF>;
defm VCVTTPD2UDQS : avx10_cvttpd2dqs<0x6C, "vcvttpd2udqs", X86cvttp2uis,
                                     X86cvttp2uis, X86cvttp2uisSAE,
                                     SchedWriteCvtPD2DQ>,
                                     REX_W, T_MAP5,PS, EVEX_CD8<64, CD8VF>;
defm VCVTTPS2DQS : avx10_cvttps2dqs<0x6D, "vcvttps2dqs", X86cvttp2sis,
                                    X86cvttp2sis, X86cvttp2sisSAE,
                                    SchedWriteCvtPS2DQ>, T_MAP5,PS,
                                    EVEX_CD8<32, CD8VF>;
defm VCVTTPS2UDQS : avx10_cvttps2dqs<0x6C, "vcvttps2udqs", X86cvttp2uis,
                                     X86cvttp2uis, X86cvttp2uisSAE,
                                     SchedWriteCvtPS2DQ>, T_MAP5,PS,
                                     EVEX_CD8<32, CD8VF>;
defm VCVTTPD2QQS : avx10_cvttpd2qqs<0x6D, "vcvttpd2qqs", X86cvttp2sis,
                                    X86cvttp2sis, X86cvttp2sisSAE,
                                    SchedWriteCvtPD2DQ>, REX_W, T_MAP5,PD,
                                    EVEX_CD8<64, CD8VF>;
defm VCVTTPS2QQS : avx10_cvttps2qqs<0x6D, "vcvttps2qqs", X86cvttp2sis,
                                    X86cvttp2sis, X86cvttp2sisSAE,
                                    SchedWriteCvtPS2DQ>, T_MAP5,PD,
                                    EVEX_CD8<32, CD8VH>;
defm VCVTTPD2UQQS : avx10_cvttpd2qqs<0x6C, "vcvttpd2uqqs", X86cvttp2uis,
                                     X86cvttp2uis, X86cvttp2uisSAE,
                                     SchedWriteCvtPD2DQ>, REX_W, T_MAP5,PD,
                                     EVEX_CD8<64, CD8VF>;
defm VCVTTPS2UQQS : avx10_cvttps2qqs<0x6C, "vcvttps2uqqs", X86cvttp2uis,
                                     X86cvttp2uis, X86cvttp2uisSAE,
                                     SchedWriteCvtPS2DQ>, T_MAP5,PD,
                                     EVEX_CD8<32, CD8VH>;

let Predicates = [HasAVX10_2] in {
// Special patterns to allow use of X86mcvttp2si for masking. Instruction
// patterns have been disabled with null_frag.
// Patterns VCVTTPD2DQSZ128

// VCVTTPD2DQS
def : Pat<(v4i32(X86fp2sisat(v2f64 VR128X:$src))),
          (VCVTTPD2DQSZ128rr VR128X:$src)>;
def : Pat<(v4i32(fp_to_sint_sat(v4f64 VR256X:$src), i32)),
          (VCVTTPD2DQSZ256rr VR256X:$src)>;
def : Pat<(v8i32(fp_to_sint_sat(v8f64 VR512:$src), i32)),
          (VCVTTPD2DQSZrr VR512:$src)>;

// VCVTTPD2QQS
def : Pat<(v2i64(fp_to_sint_sat(v2f64 VR128X:$src), i64)),
          (VCVTTPD2QQSZ128rr VR128X:$src)>;
def : Pat<(v4i64(fp_to_sint_sat(v4f64 VR256X:$src), i64)),
          (VCVTTPD2QQSZ256rr VR256X:$src)>;
def : Pat<(v8i64(fp_to_sint_sat(v8f64 VR512:$src), i64)),
          (VCVTTPD2QQSZrr VR512:$src)>;

// VCVTTPD2UDQS
def : Pat<(v4i32(X86fp2uisat(v2f64 VR128X:$src))),
          (VCVTTPD2UDQSZ128rr VR128X:$src)>;
def : Pat<(v4i32(fp_to_uint_sat(v4f64 VR256X:$src), i32)),
          (VCVTTPD2UDQSZ256rr VR256X:$src)>;
def : Pat<(v8i32(fp_to_uint_sat(v8f64 VR512:$src), i32)),
          (VCVTTPD2UDQSZrr VR512:$src)>;

// VCVTTPD2UQQS
def : Pat<(v2i64(fp_to_uint_sat(v2f64 VR128X:$src), i64)),
          (VCVTTPD2UQQSZ128rr VR128X:$src)>;
def : Pat<(v4i64(fp_to_uint_sat(v4f64 VR256X:$src), i64)),
          (VCVTTPD2UQQSZ256rr VR256X:$src)>;
def : Pat<(v8i64(fp_to_uint_sat(v8f64 VR512:$src), i64)),
          (VCVTTPD2UQQSZrr VR512:$src)>;

// VCVTTPS2DQS
def : Pat<(v4i32(fp_to_sint_sat(v4f32 VR128X:$src), i32)),
          (VCVTTPS2DQSZ128rr VR128X:$src)>;
def : Pat<(v8i32(fp_to_sint_sat(v8f32 VR256X:$src), i32)),
          (VCVTTPS2DQSZ256rr VR256X:$src)>;
def : Pat<(v16i32(fp_to_sint_sat(v16f32 VR512:$src), i32)),
          (VCVTTPS2DQSZrr VR512:$src)>;

// VCVTTPS2QQS
def : Pat<(v2i64(X86fp2sisat(v4f32 VR128X:$src))),
          (VCVTTPS2QQSZ128rr VR128X:$src)>;
def : Pat<(v4i64(fp_to_sint_sat(v4f32 VR128X:$src), i64)),
          (VCVTTPS2QQSZ256rr VR128X:$src)>;
def : Pat<(v8i64(fp_to_sint_sat(v8f32 VR256X:$src), i64)),
          (VCVTTPS2QQSZrr VR256X:$src)>;

// VCVTTPS2UDQS
def : Pat<(v4i32(fp_to_uint_sat(v4f32 VR128X:$src), i32)),
          (VCVTTPS2UDQSZ128rr VR128X:$src)>;
def : Pat<(v8i32(fp_to_uint_sat(v8f32 VR256X:$src), i32)),
          (VCVTTPS2UDQSZ256rr VR256X:$src)>;
def : Pat<(v16i32(fp_to_uint_sat(v16f32 VR512:$src), i32)),
          (VCVTTPS2UDQSZrr VR512:$src)>;

// VCVTTPS2UQQS
def : Pat<(v2i64(X86fp2uisat(v4f32 VR128X:$src))),
          (VCVTTPS2UQQSZ128rr VR128X:$src)>;
def : Pat<(v4i64(fp_to_uint_sat(v4f32 VR128X:$src), i64)),
          (VCVTTPS2UQQSZ256rr VR128X:$src)>;
def : Pat<(v8i64(fp_to_uint_sat(v8f32 VR256X:$src), i64)),
          (VCVTTPS2UQQSZrr VR256X:$src)>;

def : Pat<(v4i32 (X86cvttp2sis (v2f64 VR128X:$src))),
          (VCVTTPD2DQSZ128rr VR128X:$src)>;
def : Pat<(v4i32 (X86cvttp2sis (loadv2f64 addr:$src))),
          (VCVTTPD2DQSZ128rm addr:$src)>;
def : Pat<(v4i32 (X86cvttp2sis (v2f64 (X86VBroadcastld64 addr:$src)))),
          (VCVTTPD2DQSZ128rmb addr:$src)>;
def : Pat<(X86mcvttp2sis (v2f64 VR128X:$src), (v4i32 VR128X:$src0),
           VK2WM:$mask),
          (VCVTTPD2DQSZ128rrk VR128X:$src0, VK2WM:$mask, VR128X:$src)>;
def : Pat<(X86mcvttp2sis (v2f64 VR128X:$src), v4i32x_info.ImmAllZerosV,
           VK2WM:$mask),
          (VCVTTPD2DQSZ128rrkz VK2WM:$mask, VR128X:$src)>;
def : Pat<(X86mcvttp2sis (loadv2f64 addr:$src), (v4i32 VR128X:$src0),
           VK2WM:$mask),
          (VCVTTPD2DQSZ128rmk VR128X:$src0, VK2WM:$mask, addr:$src)>;
def : Pat<(X86mcvttp2sis (loadv2f64 addr:$src), v4i32x_info.ImmAllZerosV,
           VK2WM:$mask),
          (VCVTTPD2DQSZ128rmkz VK2WM:$mask, addr:$src)>;
def : Pat<(X86mcvttp2sis (v2f64 (X86VBroadcastld64 addr:$src)),
           (v4i32 VR128X:$src0), VK2WM:$mask),
          (VCVTTPD2DQSZ128rmbk VR128X:$src0, VK2WM:$mask, addr:$src)>;
def : Pat<(X86mcvttp2sis (v2f64 (X86VBroadcastld64 addr:$src)),
           v4i32x_info.ImmAllZerosV, VK2WM:$mask),
          (VCVTTPD2DQSZ128rmbkz VK2WM:$mask, addr:$src)>;

// Patterns VCVTTPD2UDQSZ128
def : Pat<(v4i32 (X86cvttp2uis (v2f64 (X86VBroadcastld64 addr:$src)))),
          (VCVTTPD2UDQSZ128rmb addr:$src)>;
def : Pat<(v4i32 (X86cvttp2uis (v2f64 VR128X:$src))),
          (VCVTTPD2UDQSZ128rr VR128X:$src)>;
def : Pat<(v4i32 (X86cvttp2uis (v2f64 (X86VBroadcastld64 addr:$src)))),
          (VCVTTPD2UDQSZ128rmb addr:$src)>;
def : Pat<(X86mcvttp2uis (v2f64 VR128X:$src), (v4i32 VR128X:$src0),
           VK2WM:$mask),
          (VCVTTPD2UDQSZ128rrk VR128X:$src0, VK2WM:$mask, VR128X:$src)>;
def : Pat<(X86mcvttp2uis (v2f64 VR128X:$src), v4i32x_info.ImmAllZerosV,
           VK2WM:$mask),
          (VCVTTPD2UDQSZ128rrkz VK2WM:$mask, VR128X:$src)>;
def : Pat<(X86mcvttp2uis (loadv2f64 addr:$src), (v4i32 VR128X:$src0),
           VK2WM:$mask),
          (VCVTTPD2UDQSZ128rmk VR128X:$src0, VK2WM:$mask, addr:$src)>;
def : Pat<(X86mcvttp2uis (loadv2f64 addr:$src), v4i32x_info.ImmAllZerosV,
           VK2WM:$mask),
          (VCVTTPD2UDQSZ128rmkz VK2WM:$mask, addr:$src)>;
def : Pat<(X86mcvttp2uis (v2f64 (X86VBroadcastld64 addr:$src)),
           (v4i32 VR128X:$src0), VK2WM:$mask),
          (VCVTTPD2UDQSZ128rmbk VR128X:$src0, VK2WM:$mask, addr:$src)>;
def : Pat<(X86mcvttp2uis (v2f64 (X86VBroadcastld64 addr:$src)),
           v4i32x_info.ImmAllZerosV, VK2WM:$mask),
          (VCVTTPD2UDQSZ128rmbkz VK2WM:$mask, addr:$src)>;
}

// Convert scalar float/double to signed/unsigned int 32/64 with truncation and saturation.
multiclass avx10_cvt_s_ds<bits<8> opc, string asm, X86VectorVTInfo _SrcRC,
                          X86VectorVTInfo _DstRC, SDPatternOperator OpNode,
                          SDNode OpNodeInt, SDNode OpNodeSAE,
                          X86FoldableSchedWrite sched> {
  let Predicates = [HasAVX10_2], ExeDomain = _SrcRC.ExeDomain in {
    let isCodeGenOnly = 1 in {
    def rr : AVX512<opc, MRMSrcReg, (outs _DstRC.RC:$dst), (ins _SrcRC.FRC:$src),
                    !strconcat(asm,"\t{$src, $dst|$dst, $src}"),
                      [(set _DstRC.RC:$dst, (OpNode _SrcRC.FRC:$src, _DstRC.EltVT))]>,
                    EVEX, VEX_LIG, Sched<[sched]>, SIMD_EXC;
    def rm : AVX512<opc, MRMSrcMem, (outs _DstRC.RC:$dst), (ins _SrcRC.ScalarMemOp:$src),
                    !strconcat(asm,"\t{$src, $dst|$dst, $src}"),
                      [(set _DstRC.RC:$dst, (OpNode (_SrcRC.ScalarLdFrag addr:$src), _DstRC.EltVT))]>,
                    EVEX, VEX_LIG, Sched<[sched.Folded, sched.ReadAfterFold]>, SIMD_EXC;
  }
  def rr_Int : AVX512<opc, MRMSrcReg, (outs _DstRC.RC:$dst), (ins _SrcRC.RC:$src),
                      !strconcat(asm,"\t{$src, $dst|$dst, $src}"),
                        [(set _DstRC.RC:$dst, (OpNodeInt (_SrcRC.VT _SrcRC.RC:$src)))]>,
                      EVEX, VEX_LIG, Sched<[sched]>, SIMD_EXC;
  let Uses = [MXCSR] in
  def rrb_Int : AVX512<opc, MRMSrcReg, (outs _DstRC.RC:$dst), (ins _SrcRC.RC:$src),
                       !strconcat(asm,"\t{{sae}, $src, $dst|$dst, $src, {sae}}"),
                        [(set _DstRC.RC:$dst, (OpNodeSAE (_SrcRC.VT _SrcRC.RC:$src)))]>,
                      EVEX, VEX_LIG, EVEX_B, Sched<[sched]>;
  def rm_Int : AVX512<opc, MRMSrcMem, (outs _DstRC.RC:$dst),
                      (ins _SrcRC.IntScalarMemOp:$src),
                      !strconcat(asm,"\t{$src, $dst|$dst, $src}"),
                        [(set _DstRC.RC:$dst,
                          (OpNodeInt (_SrcRC.ScalarIntMemFrags addr:$src)))]>,
                      EVEX, VEX_LIG, Sched<[sched.Folded, sched.ReadAfterFold]>,
                      SIMD_EXC;
  }
}

defm VCVTTSS2SIS: avx10_cvt_s_ds<0x6D, "vcvttss2sis", f32x_info, i32x_info,
                                 fp_to_sint_sat, X86cvttss2Int,
                                 X86cvttss2IntSAE, WriteCvtSS2I>,
                                 T_MAP5,XS, EVEX_CD8<32, CD8VT1>;
defm VCVTTSS2SI64S: avx10_cvt_s_ds<0x6D, "vcvttss2sis", f32x_info, i64x_info,
                                   fp_to_sint_sat, X86cvttss2Int,
                                   X86cvttss2IntSAE, WriteCvtSS2I>,
                                   REX_W, T_MAP5,XS, EVEX_CD8<32, CD8VT1>;
defm VCVTTSD2SIS: avx10_cvt_s_ds<0x6D, "vcvttsd2sis", f64x_info, i32x_info,
                                 fp_to_sint_sat, X86cvttss2Int,
                                 X86cvttss2IntSAE, WriteCvtSD2I>,
                                 T_MAP5,XD, EVEX_CD8<64, CD8VT1>;
defm VCVTTSD2SI64S: avx10_cvt_s_ds<0x6D, "vcvttsd2sis", f64x_info, i64x_info,
                                   fp_to_sint_sat, X86cvttss2Int,
                                   X86cvttss2IntSAE, WriteCvtSD2I>,
                                   REX_W, T_MAP5,XD, EVEX_CD8<64, CD8VT1>;
defm VCVTTSS2USIS: avx10_cvt_s_ds<0x6C, "vcvttss2usis", f32x_info, i32x_info,
                                  fp_to_uint_sat, X86cvttss2UInt,
                                  X86cvttss2UIntSAE, WriteCvtSS2I>,
                                  T_MAP5,XS, EVEX_CD8<32, CD8VT1>;
defm VCVTTSS2USI64S: avx10_cvt_s_ds<0x6C, "vcvttss2usis", f32x_info, i64x_info,
                                    fp_to_uint_sat, X86cvttss2UInt,
                                     X86cvttss2UIntSAE, WriteCvtSS2I>,
                                    T_MAP5,XS,REX_W, EVEX_CD8<32, CD8VT1>;
defm VCVTTSD2USIS: avx10_cvt_s_ds<0x6C, "vcvttsd2usis", f64x_info, i32x_info,
                                  fp_to_uint_sat, X86cvttss2UInt,
                                  X86cvttss2UIntSAE, WriteCvtSD2I>,
                                  T_MAP5,XD, EVEX_CD8<64, CD8VT1>;
defm VCVTTSD2USI64S: avx10_cvt_s_ds<0x6C, "vcvttsd2usis", f64x_info, i64x_info,
                                    fp_to_uint_sat, X86cvttss2UInt,
                                    X86cvttss2UIntSAE, WriteCvtSD2I>,
                                    T_MAP5,XD, REX_W, EVEX_CD8<64, CD8VT1>;

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

defm VCVT2PH2BF8 : avx512_binop_all<0x74, "vcvt2ph2bf8", SchedWriteCvtPD2PS,
                                     avx512vl_f16_info, avx512vl_i8_info,
                                     X86vcvt2ph2bf8, [HasAVX10_2_512], [HasAVX10_2]>,
                                    EVEX_CD8<16, CD8VF>, T8, XD;
defm VCVT2PH2BF8S : avx512_binop_all<0x74, "vcvt2ph2bf8s", SchedWriteCvtPD2PS,
                                      avx512vl_f16_info, avx512vl_i8_info,
                                      X86vcvt2ph2bf8s, [HasAVX10_2_512], [HasAVX10_2]>,
                                     EVEX_CD8<16, CD8VF>, T_MAP5, XD;
defm VCVT2PH2HF8 : avx512_binop_all<0x18, "vcvt2ph2hf8", SchedWriteCvtPD2PS,
                                     avx512vl_f16_info, avx512vl_i8_info,
                                     X86vcvt2ph2hf8, [HasAVX10_2_512], [HasAVX10_2]>,
                                    EVEX_CD8<16, CD8VF>, T_MAP5, XD;
defm VCVT2PH2HF8S : avx512_binop_all<0x1b, "vcvt2ph2hf8s", SchedWriteCvtPD2PS,
                                      avx512vl_f16_info, avx512vl_i8_info,
                                      X86vcvt2ph2hf8s, [HasAVX10_2_512], [HasAVX10_2]>,
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

defm VCVTPH2BF8 : avx512_cvt_trunc_ne<0x74, "vcvtph2bf8", avx512vl_i8_info,
                                        avx512vl_f16_info, SchedWriteCvtPD2PS,
                                        X86vcvtph2bf8, X86vmcvtph2bf8,
                                        [HasAVX10_2], [HasAVX10_2_512]>,
                                        T8, XS, EVEX_CD8<16, CD8VF>;

defm VCVTPH2BF8S : avx512_cvt_trunc_ne<0x74, "vcvtph2bf8s", avx512vl_i8_info,
                                         avx512vl_f16_info, SchedWriteCvtPD2PS,
                                         X86vcvtph2bf8s, X86vmcvtph2bf8s,
                                         [HasAVX10_2], [HasAVX10_2_512]>,
                                         T_MAP5, XS, EVEX_CD8<16, CD8VF>;

defm VCVTPH2HF8 : avx512_cvt_trunc_ne<0x18, "vcvtph2hf8", avx512vl_i8_info,
                                        avx512vl_f16_info, SchedWriteCvtPD2PS,
                                        X86vcvtph2hf8, X86vmcvtph2hf8,
                                        [HasAVX10_2], [HasAVX10_2_512]>,
                                        T_MAP5, XS, EVEX_CD8<16, CD8VF>;

defm VCVTPH2HF8S : avx512_cvt_trunc_ne<0x1b, "vcvtph2hf8s", avx512vl_i8_info,
                                         avx512vl_f16_info, SchedWriteCvtPD2PS,
                                         X86vcvtph2hf8s, X86vmcvtph2hf8s,
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

// VADDBF16, VSUBBF16, VMULBF16, VDIVBF16, VMAXBF16, VMINBF16
multiclass avx10_fp_binop_int_bf16<bits<8> opc, string OpcodeStr,
                                      X86SchedWriteSizes sched,
                                      bit IsCommutable = 0> {
  let Predicates = [HasAVX10_2_512] in
    defm Z : avx512_fp_packed<opc, OpcodeStr,
                              !cast<Intrinsic>("int_x86_avx10_"#OpcodeStr#"bf16512"),
                              !cast<Intrinsic>("int_x86_avx10_"#OpcodeStr#"bf16512"),
                              v32bf16_info, sched.PH.ZMM, IsCommutable>, EVEX_V512,
                              T_MAP5, PD, EVEX_CD8<16, CD8VF>;
  let Predicates = [HasAVX10_2] in {
    defm Z128 : avx512_fp_packed<opc, OpcodeStr,
                                 !cast<Intrinsic>("int_x86_avx10_"#OpcodeStr#"bf16128"),
                                 !cast<Intrinsic>("int_x86_avx10_"#OpcodeStr#"bf16128"),
                                 v8bf16x_info, sched.PH.XMM, IsCommutable>, EVEX_V128,
                                 T_MAP5, PD, EVEX_CD8<16, CD8VF>;
    defm Z256 : avx512_fp_packed<opc, OpcodeStr,
                                 !cast<Intrinsic>("int_x86_avx10_"#OpcodeStr#"bf16256"),
                                 !cast<Intrinsic>("int_x86_avx10_"#OpcodeStr#"bf16256"),
                                 v16bf16x_info, sched.PH.YMM, IsCommutable>, EVEX_V256,
                                 T_MAP5, PD, EVEX_CD8<16, CD8VF>;
  }
}

multiclass avx10_fp_binop_bf16<bits<8> opc, string OpcodeStr, SDPatternOperator OpNode,
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
defm VADDBF16 : avx10_fp_binop_bf16<0x58, "vadd", fadd, SchedWriteFAddSizes, 1>;
defm VSUBBF16 : avx10_fp_binop_bf16<0x5C, "vsub", fsub, SchedWriteFAddSizes, 0>;
defm VMULBF16 : avx10_fp_binop_bf16<0x59, "vmul", fmul, SchedWriteFMulSizes, 1>;
defm VDIVBF16 : avx10_fp_binop_bf16<0x5E, "vdiv", fdiv, SchedWriteFDivSizes, 0>;
defm VMINBF16 : avx10_fp_binop_int_bf16<0x5D, "vmin", SchedWriteFCmpSizes, 0>;
defm VMAXBF16 : avx10_fp_binop_int_bf16<0x5F, "vmax", SchedWriteFCmpSizes, 0>;
}

// VCOMISBF16
let Uses = []<Register>, mayRaiseFPException = 0,
  Defs = [EFLAGS], Predicates = [HasAVX10_2] in {
  //TODO: Replace null_frag with X86fcmp to support lowering `fcmp oeq bfloat *`
  //which may require extend supports on BFR16X, loadbf16, ...
  defm VCOMISBF16Z : sse12_ord_cmp<0x2F, FR16X, null_frag, bf16, f16mem, loadf16,
                                   "comisbf16", SSEPackedSingle>, T_MAP5, PD, EVEX,
                                   VEX_LIG, EVEX_CD8<16, CD8VT1>;

  let isCodeGenOnly = 1 in {
    defm VCOMISBF16Z : sse12_ord_cmp_int<0x2F, VR128X, X86comi, v8bf16, f16mem,
                                         sse_load_bf16, "comisbf16", SSEPackedSingle>,
                                         T_MAP5, PD, EVEX, VEX_LIG, EVEX_CD8<16, CD8VT1>;
  }
}

// VCMPBF16
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

defm VCMPBF16 : avx10_vcmp_bf16<SchedWriteFCmp, avx512vl_bf16_info>,
                                 AVX512XDIi8Base, EVEX, VVVV,
                                 EVEX_CD8<16, CD8VF>, TA;


// VSQRTBF16
multiclass avx10_sqrt_packed_bf16<bits<8> opc, string OpcodeStr,
                                  X86SchedWriteSizes sched> {
  let Predicates = [HasAVX10_2_512] in
  defm Z : avx512_sqrt_packed<opc, !strconcat(OpcodeStr, "bf16"),
                              sched.PH.ZMM, v32bf16_info>,
                              EVEX_V512, PD, T_MAP5, EVEX_CD8<16, CD8VF>;
  let Predicates = [HasAVX10_2] in {
    defm Z128 : avx512_sqrt_packed<opc, !strconcat(OpcodeStr, "bf16"),
                                   sched.PH.XMM, v8bf16x_info>,
                                   EVEX_V128, PD, T_MAP5, EVEX_CD8<16, CD8VF>;
    defm Z256 : avx512_sqrt_packed<opc, !strconcat(OpcodeStr, "bf16"),
                                   sched.PH.YMM, v16bf16x_info>,
                                   EVEX_V256, PD, T_MAP5, EVEX_CD8<16, CD8VF>;
  }
}

let Uses = []<Register>, mayRaiseFPException = 0 in
defm VSQRTBF16 : avx10_sqrt_packed_bf16<0x51, "vsqrt", SchedWriteFSqrtSizes>;

// VRSQRTBF16, VRCPBF16, VSRQTBF16, VGETEXPBF16
multiclass avx10_fp14_bf16<bits<8> opc, string OpcodeStr, SDNode OpNode,
                            X86SchedWriteWidths sched> {
  let Predicates = [HasAVX10_2_512] in
  defm BF16Z : avx512_fp14_p<opc, !strconcat(OpcodeStr, "bf16"),
                             OpNode, sched.ZMM, v32bf16_info>,
                             EVEX_V512;
  let Predicates = [HasAVX10_2] in {
    defm BF16Z128 : avx512_fp14_p<opc, !strconcat(OpcodeStr, "bf16"),
                                  OpNode, sched.XMM, v8bf16x_info>,
                                  EVEX_V128;
    defm BF16Z256 : avx512_fp14_p<opc, !strconcat(OpcodeStr, "bf16"),
                                  OpNode, sched.YMM, v16bf16x_info>,
                                  EVEX_V256;
  }
}

defm VRSQRT  : avx10_fp14_bf16<0x4E, "vrsqrt", X86rsqrt14, SchedWriteFRsqrt>,
                                T_MAP6, PS, EVEX_CD8<16, CD8VF>;
defm VRCP    : avx10_fp14_bf16<0x4C, "vrcp", X86rcp14, SchedWriteFRcp>,
                                T_MAP6, PS, EVEX_CD8<16, CD8VF>;
defm VGETEXP : avx10_fp14_bf16<0x42, "vgetexp", X86fgetexp, SchedWriteFRnd>,
                                T_MAP5, EVEX_CD8<16, CD8VF>;

// VSCALEFBF16
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
defm VSCALEFBF16 : avx10_fp_scalef_bf16<0x2C, "vscalef", SchedWriteFAdd>;

// VREDUCEBF16, VRNDSCALEBF16, VGETMANTBF16
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
defm VREDUCEBF16 : avx10_common_unary_fp_packed_imm_bf16<"vreduce", avx512vl_bf16_info, 0x56,
                            X86VReduce, X86VReduce, SchedWriteFRnd>,
                            AVX512XDIi8Base, TA, EVEX, EVEX_CD8<16, CD8VF>;
defm VRNDSCALEBF16 : avx10_common_unary_fp_packed_imm_bf16<"vrndscale", avx512vl_bf16_info, 0x08,
                            X86any_VRndScale, X86VRndScale, SchedWriteFRnd>,
                            AVX512XDIi8Base, TA, EVEX, EVEX_CD8<16, CD8VF>;
defm VGETMANTBF16 : avx10_common_unary_fp_packed_imm_bf16<"vgetmant", avx512vl_bf16_info, 0x26,
                            X86VGetMant, X86VGetMant, SchedWriteFRnd>,
                            AVX512XDIi8Base, TA, EVEX, EVEX_CD8<16, CD8VF>;
}

// VFPCLASSBF16
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

defm VFPCLASSBF16 : avx10_fp_fpclass_bf16<"vfpclass", 0x66, SchedWriteFCmp>,
                                      AVX512XDIi8Base, TA, EVEX, EVEX_CD8<16, CD8VF>;

// VF[,N]M[ADD,SUB][132,213,231]BF16
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
defm VFMADD213BF16 : avx10_fma3p_213_bf16<0xA8, "vfmadd213bf16", any_fma,
                                          fma, SchedWriteFMA>;
defm VFMSUB213BF16 : avx10_fma3p_213_bf16<0xAA, "vfmsub213bf16", X86any_Fmsub,
                                          X86Fmsub, SchedWriteFMA>;
defm VFNMADD213BF16 : avx10_fma3p_213_bf16<0xAC, "vfnmadd213bf16", X86any_Fnmadd,
                                           X86Fnmadd, SchedWriteFMA>;
defm VFNMSUB213BF16 : avx10_fma3p_213_bf16<0xAE, "vfnmsub213bf16", X86any_Fnmsub,
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
defm VFMADD231BF16 : avx10_fma3p_231_bf16<0xB8, "vfmadd231bf16", any_fma,
                                          fma, SchedWriteFMA>;
defm VFMSUB231BF16 : avx10_fma3p_231_bf16<0xBA, "vfmsub231bf16", X86any_Fmsub,
                                          X86Fmsub, SchedWriteFMA>;
defm VFNMADD231BF16 : avx10_fma3p_231_bf16<0xBC, "vfnmadd231bf16", X86any_Fnmadd,
                                           X86Fnmadd, SchedWriteFMA>;
defm VFNMSUB231BF16 : avx10_fma3p_231_bf16<0xBE, "vfnmsub231bf16", X86any_Fnmsub,
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
defm VFMADD132BF16 : avx10_fma3p_132_bf16<0x98, "vfmadd132bf16", any_fma,
                                          fma, SchedWriteFMA>;
defm VFMSUB132BF16 : avx10_fma3p_132_bf16<0x9A, "vfmsub132bf16", X86any_Fmsub,
                                             X86Fmsub, SchedWriteFMA>;
defm VFNMADD132BF16 : avx10_fma3p_132_bf16<0x9C, "vfnmadd132bf16", X86any_Fnmadd,
                                           X86Fnmadd, SchedWriteFMA>;
defm VFNMSUB132BF16 : avx10_fma3p_132_bf16<0x9E, "vfnmsub132bf16", X86any_Fnmsub,
                                           X86Fnmsub, SchedWriteFMA>;
}

//-------------------------------------------------
// AVX10  COMEF instructions
//-------------------------------------------------
multiclass avx10_com_ef<bits<8> Opc, RegisterClass RC, ValueType VT,
                        SDPatternOperator OpNode, string OpcodeStr,
                        X86MemOperand x86memop, PatFrag ld_frag,
                        Domain d, X86FoldableSchedWrite sched = WriteFComX>{
  let ExeDomain = d, mayRaiseFPException = 1, isCodeGenOnly = 1 in {
    def rr : AVX512<Opc, MRMSrcReg, (outs), (ins RC:$src1, RC:$src2),
                    !strconcat(OpcodeStr, "\t{$src2, $src1|$src1, $src2}"),
                    [(set EFLAGS, (OpNode (VT RC:$src1), RC:$src2))]>,
                    EVEX, EVEX_V128, Sched<[sched]>, SIMD_EXC;
    let mayLoad = 1 in {
      def rm : AVX512<Opc, MRMSrcMem, (outs), (ins RC:$src1, x86memop:$src2),
                      !strconcat(OpcodeStr, "\t{$src2, $src1|$src1, $src2}"),
                      [(set EFLAGS, (OpNode (VT RC:$src1), (ld_frag addr:$src2)))]>,
                      EVEX, EVEX_V128, Sched<[sched.Folded, sched.ReadAfterFold]>, SIMD_EXC;
    }
  }
}

multiclass avx10_com_ef_int<bits<8> Opc, X86VectorVTInfo _, SDNode OpNode,
                             string OpcodeStr,
                             Domain d,
                             X86FoldableSchedWrite sched = WriteFComX> {
  let ExeDomain = d, mayRaiseFPException = 1 in {
    def rr_Int : AVX512<Opc, MRMSrcReg, (outs), (ins _.RC:$src1, _.RC:$src2),
                        !strconcat(OpcodeStr, "\t{$src2, $src1|$src1, $src2}"),
                        [(set EFLAGS, (OpNode (_.VT _.RC:$src1), _.RC:$src2))]>,
                        EVEX, EVEX_V128, Sched<[sched]>, SIMD_EXC;
    let mayLoad = 1 in {
      def rm_Int : AVX512<Opc, MRMSrcMem, (outs), (ins _.RC:$src1, _.ScalarMemOp:$src2),
                          !strconcat(OpcodeStr, "\t{$src2, $src1|$src1, $src2}"),
                          [(set EFLAGS, (OpNode (_.VT _.RC:$src1), (_.LdFrag addr:$src2)))]>,
                          EVEX, EVEX_V128, Sched<[sched]>, SIMD_EXC;
    }
    def rrb_Int : AVX512<Opc, MRMSrcReg, (outs), (ins _.RC:$src1, _.RC:$src2),
                        !strconcat(OpcodeStr, "\t{{sae}, $src2, $src1|$src1, $src2, {sae}}"),
                        []>,
                        EVEX, EVEX_V128, EVEX_B, Sched<[sched]>, SIMD_EXC;
  }
}

let Defs = [EFLAGS], Uses = [MXCSR], Predicates = [HasAVX10_2] in {
  defm VUCOMXSDZ  :  avx10_com_ef<0x2e, FR64X, f64, X86ucomi512,
                                  "vucomxsd", f64mem, loadf64, SSEPackedDouble>,
                                  TB, XS, VEX_LIG, REX_W, EVEX_CD8<64, CD8VT1>;
  defm VUCOMXSHZ  :  avx10_com_ef<0x2e, FR16X, f16, X86ucomi512,
                                  "vucomxsh", f16mem, loadf16, SSEPackedSingle>,
                                  T_MAP5, XD, EVEX_CD8<16, CD8VT1>;
  defm VUCOMXSSZ  :  avx10_com_ef<0x2e, FR32X, f32, X86ucomi512,
                                  "vucomxss", f32mem, loadf32, SSEPackedSingle>,
                                  TB, XD, VEX_LIG, EVEX_CD8<32, CD8VT1>;
  defm VCOMXSDZ   :  avx10_com_ef_int<0x2f, v2f64x_info, X86comi512,
                                      "vcomxsd", SSEPackedDouble>,
                                      TB, XS, VEX_LIG, REX_W, EVEX_CD8<64, CD8VT1>;
  defm VCOMXSHZ   :  avx10_com_ef_int<0x2f, v8f16x_info, X86comi512,
                                      "vcomxsh", SSEPackedSingle>,
                                      T_MAP5, XD, EVEX_CD8<16, CD8VT1>;
  defm VCOMXSSZ   :  avx10_com_ef_int<0x2f, v4f32x_info, X86comi512,
                                      "vcomxss", SSEPackedSingle>,
                                      TB, XD, VEX_LIG, EVEX_CD8<32, CD8VT1>;
  defm VUCOMXSDZ  :  avx10_com_ef_int<0x2e, v2f64x_info, X86ucomi512,
                                      "vucomxsd", SSEPackedDouble>,
                                      TB, XS, VEX_LIG, REX_W, EVEX_CD8<64, CD8VT1>;
  defm VUCOMXSHZ  :  avx10_com_ef_int<0x2e, v8f16x_info, X86ucomi512,
                                      "vucomxsh", SSEPackedSingle>,
                                      T_MAP5, XD, EVEX_CD8<16, CD8VT1>;
  defm VUCOMXSSZ  :  avx10_com_ef_int<0x2e, v4f32x_info, X86ucomi512,
                                      "vucomxss", SSEPackedSingle>,
                                      TB, XD, VEX_LIG, EVEX_CD8<32, CD8VT1>;
}

//-------------------------------------------------
// AVX10 MOVZXC (COPY) instructions
//-------------------------------------------------
let Predicates = [HasAVX10_2] in {
  def VMOVZPDILo2PDIZrr : AVX512XSI<0x7E, MRMSrcReg, (outs VR128X:$dst),
                                    (ins VR128X:$src),
                                    "vmovd\t{$src, $dst|$dst, $src}",
                                    [(set VR128X:$dst, (v4i32 (X86vzmovl
                                    (v4i32 VR128X:$src))))]>, EVEX,
                                    Sched<[WriteVecMoveFromGpr]>;

let isCodeGenOnly = 1, ForceDisassemble = 1, hasSideEffects = 0, mayLoad = 1 in
  def VMOVZPDILo2PDIZrm : AVX512XSI<0x7E, MRMSrcMem, (outs VR128X:$dst),
                                    (ins i32mem:$src),
                                    "vmovd\t{$src, $dst|$dst, $src}", []>, EVEX,
                                     EVEX_CD8<32, CD8VT1>,
                                     Sched<[WriteVecLoad]>;

let isCodeGenOnly = 1, ForceDisassemble = 1, hasSideEffects = 0, mayStore = 1 in
  def VMOVZPDILo2PDIZmr : AVX512PDI<0xD6, MRMDestMem, (outs),
                                    (ins i32mem:$dst, VR128X:$src),
                                    "vmovd\t{$src, $dst|$dst, $src}", []>, EVEX,
                                    EVEX_CD8<32, CD8VT1>,
                                    Sched<[WriteVecStore]>;

let isCodeGenOnly = 1, ForceDisassemble = 1, hasSideEffects = 0 in
  def VMOVZPDILo2PDIZrr2 : AVX512PDI<0xD6, MRMSrcReg, (outs VR128X:$dst),
                                     (ins VR128X:$src),
                                     "vmovd\t{$src, $dst|$dst, $src}", []>, EVEX,
                                     Sched<[WriteVecMoveFromGpr]>;
  def : InstAlias<"vmovd.s\t{$src, $dst|$dst, $src}",
                  (VMOVZPDILo2PDIZrr2 VR128X:$dst, VR128X:$src), 0>;

def VMOVZPWILo2PWIZrr : AVX512XSI<0x6E, MRMSrcReg, (outs VR128X:$dst),
                                  (ins VR128X:$src),
                                  "vmovw\t{$src, $dst|$dst, $src}",
                                  [(set VR128X:$dst, (v8i16 (X86vzmovl
                                  (v8i16 VR128X:$src))))]>, EVEX, T_MAP5,
                                  Sched<[WriteVecMoveFromGpr]>;

let isCodeGenOnly = 1, ForceDisassemble = 1, hasSideEffects = 0, mayLoad = 1 in
  def VMOVZPWILo2PWIZrm : AVX512XSI<0x6E, MRMSrcMem, (outs VR128X:$dst),
                                    (ins i16mem:$src),
                                    "vmovw\t{$src, $dst|$dst, $src}", []>, EVEX,
                                    EVEX_CD8<16, CD8VT1>, T_MAP5,
                                    Sched<[WriteVecLoad]>;

let isCodeGenOnly = 1, ForceDisassemble = 1, hasSideEffects = 0, mayStore = 1 in
  def VMOVZPWILo2PWIZmr : AVX512XSI<0x7E, MRMDestMem, (outs),
                                    (ins i32mem:$dst, VR128X:$src),
                                    "vmovw\t{$src, $dst|$dst, $src}", []>, EVEX,
                                    EVEX_CD8<16, CD8VT1>, T_MAP5,
                                    Sched<[WriteVecStore]>;

let isCodeGenOnly = 1, ForceDisassemble = 1, hasSideEffects = 0 in
  def VMOVZPWILo2PWIZrr2 : AVX512XSI<0x7E, MRMSrcReg, (outs VR128X:$dst),
                                     (ins VR128X:$src),
                                     "vmovw\t{$src, $dst|$dst, $src}",
                                     []>, EVEX, T_MAP5,
                                     Sched<[WriteVecMoveFromGpr]>;
  def : InstAlias<"vmovw.s\t{$src, $dst|$dst, $src}",
                  (VMOVZPWILo2PWIZrr2 VR128X:$dst, VR128X:$src), 0>;
}

// MOVRS
multiclass vmovrs_p<bits<8> opc, string OpStr, X86VectorVTInfo _> {
  let ExeDomain = _.ExeDomain in {
    defm m: AVX512_maskable<opc, MRMSrcMem, _, (outs _.RC:$dst),
                            (ins _.MemOp:$src), OpStr, "$src", "$src",
                            (_.VT (!cast<Intrinsic>("int_x86_avx10_"#OpStr#_.Size)
                                   addr:$src))>, EVEX;
  }
}

multiclass vmovrs_p_vl<bits<8> opc, string OpStr, AVX512VLVectorVTInfo _Vec> {
  let Predicates = [HasMOVRS, HasAVX10_2_512, In64BitMode] in
    defm Z : vmovrs_p<opc, OpStr, _Vec.info512>, EVEX_V512;
  let Predicates = [HasMOVRS, HasAVX10_2, In64BitMode] in {
    defm Z128 : vmovrs_p<opc, OpStr, _Vec.info128>, EVEX_V128;
    defm Z256 : vmovrs_p<opc, OpStr, _Vec.info256>, EVEX_V256;
  }
}

defm VMOVRSB : vmovrs_p_vl<0x6f, "vmovrsb", avx512vl_i8_info>,
                          T_MAP5, XD, EVEX_CD8<8, CD8VF>, Sched<[WriteVecLoad]>;
defm VMOVRSW : vmovrs_p_vl<0x6f, "vmovrsw", avx512vl_i16_info>,
                          T_MAP5, XD, REX_W, EVEX_CD8<16, CD8VF>, Sched<[WriteVecLoad]>;
defm VMOVRSD : vmovrs_p_vl<0x6f, "vmovrsd", avx512vl_i32_info>,
                          T_MAP5, XS, EVEX_CD8<32, CD8VF>, Sched<[WriteVecLoad]>;
defm VMOVRSQ : vmovrs_p_vl<0x6f, "vmovrsq", avx512vl_i64_info>,
                          T_MAP5, XS, REX_W, EVEX_CD8<64, CD8VF>, Sched<[WriteVecLoad]>;

// SM4(EVEX)
multiclass avx10_sm4_base<string OpStr> {
  // SM4_Base is in X86InstrSSE.td.
  let Predicates = [HasSM4, HasAVX10_2], AddedComplexity = 1 in {
    defm Z128 : SM4_Base<OpStr, VR128X, "128", loadv4i32, i128mem>, EVEX_V128;
    defm Z256 : SM4_Base<OpStr, VR256X, "256", loadv8i32, i256mem>, EVEX_V256;
  }
  let Predicates = [HasSM4, HasAVX10_2_512] in
    defm Z : SM4_Base<OpStr, VR512, "512", loadv16i32, i512mem>, EVEX_V512;
}

defm VSM4KEY4 : avx10_sm4_base<"vsm4key4">, T8, XS, EVEX, VVVV;
defm VSM4RNDS4 : avx10_sm4_base<"vsm4rnds4">, T8, XD, EVEX, VVVV;

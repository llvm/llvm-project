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

let Predicates = [HasAVX10_2], hasEVEX_U = 1, OpEnc = EncEVEX in
  defm VADD : avx256_fp_binop_p_round<0x58, "vadd", X86faddRnd, SchedWriteFAddSizes>;

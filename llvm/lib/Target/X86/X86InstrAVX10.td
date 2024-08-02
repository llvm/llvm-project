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
  let ExeDomain = _.ExeDomain, Predicates = [HasAVX10_2_512] in {
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

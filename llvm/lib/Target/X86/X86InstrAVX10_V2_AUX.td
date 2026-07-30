//===-- X86InstrAVX10_V2_AUX.td - AVX10 V2 AUX Instructions --*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file describes the X86 AVX10 V2 AUX instruction set, defining the
// instructions and their encoding.
//
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// AVX10 V2 AUX Multiclass Definitions
//===----------------------------------------------------------------------===//

// Convert from FP32 to FP8: truncating conversion, quarter-size output
// Output is always xmm for all VL variants.
// Adapts avx512_vcvt_fp for each VL level with explicit dest/src type mappings.
multiclass avx10_v2aux_cvt_trunc_ps2i8<bits<8> opc, string OpcodeStr,
                                        SDPatternOperator OpNode,
                                        SDPatternOperator MaskOpNode> {
  let ExeDomain = SSEPackedSingle in {
    let Uses = []<Register>, mayRaiseFPException = 0 in {
      defm Z : avx512_vcvt_fp<opc, OpcodeStr, v16i8x_info, v16f32_info,
                              OpNode, OpNode, WriteCvtPH2PSZ>, EVEX_V512;
      // Z256/Z128: use null_frag because element count mismatch between
      // dest (v16i8) and source (v8f32/v4f32) prevents avx512_vcvt_fp from
      // generating correct masked patterns. Explicit Pat patterns below.
      defm Z256 : avx512_vcvt_fp<opc, OpcodeStr, v16i8x_info, v8f32x_info,
                                 null_frag, null_frag,
                                 WriteCvtPH2PSZ, v8f32x_info.BroadcastStr,
                                 "{y}", v8f32x_info.MemOp,
                                 v8f32x_info.KRCWM>, EVEX_V256;
      defm Z128 : avx512_vcvt_fp<opc, OpcodeStr, v16i8x_info, v4f32x_info,
                                 null_frag, null_frag,
                                 WriteCvtPH2PSZ, v4f32x_info.BroadcastStr,
                                 "{x}", f128mem,
                                 v4f32x_info.KRCWM>, EVEX_V128;
    }
  }

  // InstAliases for x/y suffixes
  def : InstAlias<OpcodeStr#"x\t{$src, $dst|$dst, $src}",
                  (!cast<Instruction>(NAME # "Z128rr") VR128X:$dst,
                  VR128X:$src), 0>;
  def : InstAlias<OpcodeStr#"x\t{$src, $dst|$dst, $src}",
                  (!cast<Instruction>(NAME # "Z128rm") VR128X:$dst,
                  f128mem:$src), 0, "intel">;
  def : InstAlias<OpcodeStr#"y\t{$src, $dst|$dst, $src}",
                  (!cast<Instruction>(NAME # "Z256rr") VR128X:$dst,
                  VR256X:$src), 0>;
  def : InstAlias<OpcodeStr#"y\t{$src, $dst|$dst, $src}",
                  (!cast<Instruction>(NAME # "Z256rm") VR128X:$dst,
                  f256mem:$src), 0, "intel">;

  // Explicit patterns for Z256 (8 source elements, VK8WM mask)
  // Unmasked
  def : Pat<(v16i8 (OpNode (v8f32 VR256X:$src))),
            (!cast<Instruction>(NAME # "Z256rr") VR256X:$src)>;
  // Masked (merge)
  def : Pat<(MaskOpNode (v8f32 VR256X:$src), (v16i8 VR128X:$src0),
                         VK8WM:$mask),
            (!cast<Instruction>(NAME # "Z256rrk") VR128X:$src0, VK8WM:$mask,
                                VR256X:$src)>;
  // Masked (zero)
  def : Pat<(MaskOpNode (v8f32 VR256X:$src), v16i8x_info.ImmAllZerosV,
                         VK8WM:$mask),
            (!cast<Instruction>(NAME # "Z256rrkz") VK8WM:$mask,
                                VR256X:$src)>;
  // Memory
  def : Pat<(v16i8 (OpNode (loadv8f32 addr:$src))),
            (!cast<Instruction>(NAME # "Z256rm") addr:$src)>;
  def : Pat<(MaskOpNode (loadv8f32 addr:$src), (v16i8 VR128X:$src0),
                         VK8WM:$mask),
            (!cast<Instruction>(NAME # "Z256rmk") VR128X:$src0, VK8WM:$mask,
                                addr:$src)>;
  def : Pat<(MaskOpNode (loadv8f32 addr:$src), v16i8x_info.ImmAllZerosV,
                         VK8WM:$mask),
            (!cast<Instruction>(NAME # "Z256rmkz") VK8WM:$mask, addr:$src)>;
  // Broadcast
  def : Pat<(v16i8 (OpNode (v8f32 (X86VBroadcastld32 addr:$src)))),
            (!cast<Instruction>(NAME # "Z256rmb") addr:$src)>;
  def : Pat<(MaskOpNode (v8f32 (X86VBroadcastld32 addr:$src)),
                          (v16i8 VR128X:$src0), VK8WM:$mask),
            (!cast<Instruction>(NAME # "Z256rmbk") VR128X:$src0, VK8WM:$mask,
                                addr:$src)>;
  def : Pat<(MaskOpNode (v8f32 (X86VBroadcastld32 addr:$src)),
                          v16i8x_info.ImmAllZerosV, VK8WM:$mask),
            (!cast<Instruction>(NAME # "Z256rmbkz") VK8WM:$mask, addr:$src)>;

  // Explicit patterns for Z128 (4 source elements, VK4WM mask)
  // Unmasked
  def : Pat<(v16i8 (OpNode (v4f32 VR128X:$src))),
            (!cast<Instruction>(NAME # "Z128rr") VR128X:$src)>;
  // Masked (merge)
  def : Pat<(MaskOpNode (v4f32 VR128X:$src), (v16i8 VR128X:$src0),
                         VK4WM:$mask),
            (!cast<Instruction>(NAME # "Z128rrk") VR128X:$src0, VK4WM:$mask,
                                VR128X:$src)>;
  // Masked (zero)
  def : Pat<(MaskOpNode (v4f32 VR128X:$src), v16i8x_info.ImmAllZerosV,
                         VK4WM:$mask),
            (!cast<Instruction>(NAME # "Z128rrkz") VK4WM:$mask,
                                VR128X:$src)>;
  // Memory
  def : Pat<(v16i8 (OpNode (loadv4f32 addr:$src))),
            (!cast<Instruction>(NAME # "Z128rm") addr:$src)>;
  def : Pat<(MaskOpNode (loadv4f32 addr:$src), (v16i8 VR128X:$src0),
                         VK4WM:$mask),
            (!cast<Instruction>(NAME # "Z128rmk") VR128X:$src0, VK4WM:$mask,
                                addr:$src)>;
  def : Pat<(MaskOpNode (loadv4f32 addr:$src), v16i8x_info.ImmAllZerosV,
                         VK4WM:$mask),
            (!cast<Instruction>(NAME # "Z128rmkz") VK4WM:$mask, addr:$src)>;
  // Broadcast
  def : Pat<(v16i8 (OpNode (v4f32 (X86VBroadcastld32 addr:$src)))),
            (!cast<Instruction>(NAME # "Z128rmb") addr:$src)>;
  def : Pat<(MaskOpNode (v4f32 (X86VBroadcastld32 addr:$src)),
                          (v16i8 VR128X:$src0), VK4WM:$mask),
            (!cast<Instruction>(NAME # "Z128rmbk") VR128X:$src0, VK4WM:$mask,
                                addr:$src)>;
  def : Pat<(MaskOpNode (v4f32 (X86VBroadcastld32 addr:$src)),
                          v16i8x_info.ImmAllZerosV, VK4WM:$mask),
            (!cast<Instruction>(NAME # "Z128rmbkz") VK4WM:$mask, addr:$src)>;
}

// Convert from FP32 to FP8 with bias: 3-operand, quarter-size output
// bias + f32 source -> i8 dest
// Reuses avx10_convert_3op_packed from X86InstrAVX10.td for each VL variant.
multiclass avx10_v2aux_cvt_3op_ps<bits<8> opc, string OpcodeStr,
                                  SDPatternOperator OpNode,
                                  SDPatternOperator MaskOpNode> {
  // Z (512-bit): bias=v64i8(zmm), src=v16f32(zmm), dst=v16i8(xmm)
  // Element counts match (16), so vselect_mask works directly.
  defm Z : avx10_convert_3op_packed<opc, OpcodeStr, v16i8x_info,
             v64i8_info, v16f32_info, OpNode, OpNode, WriteCvtPH2PSZ>,
             EVEX_V512, EVEX_CD8<32, CD8VF>;
  // Z256/Z128: use null_frag because element count mismatch between
  // dest (v16i8) and source (v8f32/v4f32) prevents vselect_mask from
  // generating correct masked patterns. Explicit Pat patterns below.
  defm Z256 : avx10_convert_3op_packed<opc, OpcodeStr, v16i8x_info,
                v32i8x_info, v8f32x_info,
                null_frag, null_frag, WriteCvtPH2PSZ>,
                EVEX_V256, EVEX_CD8<32, CD8VF>;
  defm Z128 : avx10_convert_3op_packed<opc, OpcodeStr, v16i8x_info,
                v16i8x_info, v4f32x_info,
                null_frag, null_frag, WriteCvtPH2PSZ>,
                EVEX_V128, EVEX_CD8<32, CD8VF>;

  // Explicit patterns for Z256 (8 source elements, VK8WM mask)
  def : Pat<(v16i8 (OpNode (v32i8 VR256X:$src1), (v8f32 VR256X:$src2))),
            (!cast<Instruction>(NAME # "Z256rr") VR256X:$src1, VR256X:$src2)>;
  def : Pat<(MaskOpNode (v32i8 VR256X:$src1), (v8f32 VR256X:$src2),
                         (v16i8 VR128X:$src0), VK8WM:$mask),
            (!cast<Instruction>(NAME # "Z256rrk") VR128X:$src0, VK8WM:$mask,
                                VR256X:$src1, VR256X:$src2)>;
  def : Pat<(MaskOpNode (v32i8 VR256X:$src1), (v8f32 VR256X:$src2),
                         v16i8x_info.ImmAllZerosV, VK8WM:$mask),
            (!cast<Instruction>(NAME # "Z256rrkz") VK8WM:$mask,
                                VR256X:$src1, VR256X:$src2)>;
  // Memory
  def : Pat<(v16i8 (OpNode (v32i8 VR256X:$src1), (loadv8f32 addr:$src2))),
            (!cast<Instruction>(NAME # "Z256rm") VR256X:$src1, addr:$src2)>;
  def : Pat<(MaskOpNode (v32i8 VR256X:$src1), (loadv8f32 addr:$src2),
                         (v16i8 VR128X:$src0), VK8WM:$mask),
            (!cast<Instruction>(NAME # "Z256rmk") VR128X:$src0, VK8WM:$mask,
                                VR256X:$src1, addr:$src2)>;
  def : Pat<(MaskOpNode (v32i8 VR256X:$src1), (loadv8f32 addr:$src2),
                         v16i8x_info.ImmAllZerosV, VK8WM:$mask),
            (!cast<Instruction>(NAME # "Z256rmkz") VK8WM:$mask,
                                VR256X:$src1, addr:$src2)>;
  // Broadcast
  def : Pat<(v16i8 (OpNode (v32i8 VR256X:$src1),
                            (v8f32 (X86VBroadcastld32 addr:$src2)))),
            (!cast<Instruction>(NAME # "Z256rmb") VR256X:$src1, addr:$src2)>;
  def : Pat<(MaskOpNode (v32i8 VR256X:$src1),
                          (v8f32 (X86VBroadcastld32 addr:$src2)),
                          (v16i8 VR128X:$src0), VK8WM:$mask),
            (!cast<Instruction>(NAME # "Z256rmbk") VR128X:$src0, VK8WM:$mask,
                                VR256X:$src1, addr:$src2)>;
  def : Pat<(MaskOpNode (v32i8 VR256X:$src1),
                          (v8f32 (X86VBroadcastld32 addr:$src2)),
                          v16i8x_info.ImmAllZerosV, VK8WM:$mask),
            (!cast<Instruction>(NAME # "Z256rmbkz") VK8WM:$mask,
                                VR256X:$src1, addr:$src2)>;

  // Explicit patterns for Z128 (4 source elements, VK4WM mask)
  def : Pat<(v16i8 (OpNode (v16i8 VR128X:$src1), (v4f32 VR128X:$src2))),
            (!cast<Instruction>(NAME # "Z128rr") VR128X:$src1, VR128X:$src2)>;
  def : Pat<(MaskOpNode (v16i8 VR128X:$src1), (v4f32 VR128X:$src2),
                         (v16i8 VR128X:$src0), VK4WM:$mask),
            (!cast<Instruction>(NAME # "Z128rrk") VR128X:$src0, VK4WM:$mask,
                                VR128X:$src1, VR128X:$src2)>;
  def : Pat<(MaskOpNode (v16i8 VR128X:$src1), (v4f32 VR128X:$src2),
                         v16i8x_info.ImmAllZerosV, VK4WM:$mask),
            (!cast<Instruction>(NAME # "Z128rrkz") VK4WM:$mask,
                                VR128X:$src1, VR128X:$src2)>;
  // Memory
  def : Pat<(v16i8 (OpNode (v16i8 VR128X:$src1), (loadv4f32 addr:$src2))),
            (!cast<Instruction>(NAME # "Z128rm") VR128X:$src1, addr:$src2)>;
  def : Pat<(MaskOpNode (v16i8 VR128X:$src1), (loadv4f32 addr:$src2),
                         (v16i8 VR128X:$src0), VK4WM:$mask),
            (!cast<Instruction>(NAME # "Z128rmk") VR128X:$src0, VK4WM:$mask,
                                VR128X:$src1, addr:$src2)>;
  def : Pat<(MaskOpNode (v16i8 VR128X:$src1), (loadv4f32 addr:$src2),
                         v16i8x_info.ImmAllZerosV, VK4WM:$mask),
            (!cast<Instruction>(NAME # "Z128rmkz") VK4WM:$mask,
                                VR128X:$src1, addr:$src2)>;
  // Broadcast
  def : Pat<(v16i8 (OpNode (v16i8 VR128X:$src1),
                            (v4f32 (X86VBroadcastld32 addr:$src2)))),
            (!cast<Instruction>(NAME # "Z128rmb") VR128X:$src1, addr:$src2)>;
  def : Pat<(MaskOpNode (v16i8 VR128X:$src1),
                          (v4f32 (X86VBroadcastld32 addr:$src2)),
                          (v16i8 VR128X:$src0), VK4WM:$mask),
            (!cast<Instruction>(NAME # "Z128rmbk") VR128X:$src0, VK4WM:$mask,
                                VR128X:$src1, addr:$src2)>;
  def : Pat<(MaskOpNode (v16i8 VR128X:$src1),
                          (v4f32 (X86VBroadcastld32 addr:$src2)),
                          v16i8x_info.ImmAllZerosV, VK4WM:$mask),
            (!cast<Instruction>(NAME # "Z128rmbkz") VK4WM:$mask,
                                VR128X:$src1, addr:$src2)>;
}

// Convert from FP8 to FP32: expanding conversion (4x, no broadcast)
multiclass avx10_v2aux_cvt_2op_i8_to_f32<bits<8> opc, string OpcodeStr,
                                         SDNode OpNode> {
  defm Z : avx10_convert_2op_nomb_packed<opc, OpcodeStr, v16f32_info,
                                         v16i8x_info, OpNode, f128mem,
                                         WriteCvtPH2PSZ>, EVEX_V512;
  defm Z128 : avx10_convert_2op_nomb_packed<opc, OpcodeStr, v4f32x_info,
                                            v16i8x_info, OpNode, f32mem,
                                            WriteCvtPH2PSZ>, EVEX_V128;
  defm Z256 : avx10_convert_2op_nomb_packed<opc, OpcodeStr, v8f32x_info,
                                            v16i8x_info, OpNode, f64mem,
                                            WriteCvtPH2PSZ>, EVEX_V256;
}

// Convert from FP8 to FP4: truncating conversion (reg/mem dest, no masking)
// Uses MRMDestReg/MRMDestMem since destination can be memory operand.
// Source is in reg field, destination is in r/m field.
multiclass avx10_v2aux_cvt_trunc_base<bits<8> opc, string OpcodeStr,
                                      X86VectorVTInfo _src,
                                      X86VectorVTInfo _dest,
                                      X86MemOperand x86memop,
                                      X86FoldableSchedWrite sched> {
  let hasSideEffects = 0 in {
    def rr : I<opc, MRMDestReg, (outs _dest.RC:$dst),
               (ins _src.RC:$src),
               OpcodeStr # "\t{$src, $dst|$dst, $src}", []>,
             Sched<[sched]>, EVEX, EVEX_CD8<8, CD8VH>;
    let mayStore = 1 in
      def mr : I<opc, MRMDestMem, (outs),
                 (ins x86memop:$dst, _src.RC:$src),
                 OpcodeStr # "\t{$src, $dst|$dst, $src}", []>,
               Sched<[sched.Folded]>, EVEX, EVEX_CD8<8, CD8VH>;
  }

  def : Pat<(_dest.VT (!cast<Intrinsic>("int_x86_avx10_"#OpcodeStr#"_"#_src.Size)
                        (_src.VT _src.RC:$src))),
            (!cast<Instruction>(NAME # "rr") _src.RC:$src)>;
  def : Pat<(!cast<Intrinsic>("int_x86_avx10_"#OpcodeStr#"_"#_src.Size#"_mem")
                addr:$dst, (_src.VT _src.RC:$src)),
            (!cast<Instruction>(NAME # "mr") addr:$dst, _src.RC:$src)>;
}

multiclass avx10_v2aux_cvt_trunc_b<bits<8> opc, string OpcodeStr> {
  defm Z    : avx10_v2aux_cvt_trunc_base<opc, OpcodeStr, v64i8_info,
                                         v32i8x_info, i256mem,
                                         WriteCvtPH2PSZ>, EVEX_V512;
  defm Z256 : avx10_v2aux_cvt_trunc_base<opc, OpcodeStr, v32i8x_info,
                                         v16i8x_info, i128mem,
                                         WriteCvtPH2PSY>, EVEX_V256;
  defm Z128 : avx10_v2aux_cvt_trunc_base<opc, OpcodeStr, v16i8x_info,
                                         v16i8x_info, i64mem,
                                         WriteCvtPH2PS>, EVEX_V128;
}

// Convert from FP4 to FP8: expanding with masking, no broadcast (reg+mem)
multiclass avx10_v2aux_cvt_expand_masked_base<bits<8> opc, string OpcodeStr,
                                              X86VectorVTInfo _,
                                              X86VectorVTInfo _src,
                                              X86MemOperand x86memop,
                                              X86FoldableSchedWrite sched> {
  let ExeDomain = _.ExeDomain in {
    defm rr : AVX512_maskable<opc, MRMSrcReg, _, (outs _.RC:$dst),
                              (ins _src.RC:$src),
                              OpcodeStr, "$src", "$src",
                              (_.VT (!cast<Intrinsic>("int_x86_avx10_"#OpcodeStr#"_"#_.Size)
                                     (_src.VT _src.RC:$src)))>,
                              Sched<[sched]>, EVEX, EVEX_CD8<8, CD8VH>;
    let mayLoad = 1 in
      defm rm : AVX512_maskable<opc, MRMSrcMem, _, (outs _.RC:$dst),
                                (ins x86memop:$src),
                                OpcodeStr, "$src", "$src",
                                (_.VT (!cast<Intrinsic>("int_x86_avx10_"#OpcodeStr#"_"#_.Size)
                                       (_src.VT (load addr:$src))))>,
                                Sched<[sched.Folded]>, EVEX, EVEX_CD8<8, CD8VH>;
  }
}

multiclass avx10_v2aux_cvt_expand_masked_b<bits<8> opc, string OpcodeStr> {
  defm Z    : avx10_v2aux_cvt_expand_masked_base<opc, OpcodeStr, v64i8_info,
                                                 v32i8x_info, f256mem,
                                                 WriteCvtPH2PSZ>, EVEX_V512;
  defm Z256 : avx10_v2aux_cvt_expand_masked_base<opc, OpcodeStr, v32i8x_info,
                                                 v16i8x_info, f128mem,
                                                 WriteCvtPH2PSY>, EVEX_V256;
  defm Z128 : avx10_v2aux_cvt_expand_masked_base<opc, OpcodeStr, v16i8x_info,
                                                 v16i8x_info, f64mem,
                                                 WriteCvtPH2PS>, EVEX_V128;
}

// Convert from FP6 to FP8: widening conversion with masking (reg-only)
multiclass avx10_v2aux_cvt_widen_masked_base<bits<8> opc, string OpcodeStr,
                                             X86VectorVTInfo _,
                                             X86FoldableSchedWrite sched> {
  let ExeDomain = _.ExeDomain in {
    defm rr : AVX512_maskable<opc, MRMSrcReg, _, (outs _.RC:$dst),
                              (ins _.RC:$src),
                              OpcodeStr, "$src", "$src",
                              (_.VT (!cast<Intrinsic>("int_x86_avx10_"#OpcodeStr#"_"#_.Size)
                                     (_.VT _.RC:$src)))>,
                              Sched<[sched]>, EVEX;
  }
}

multiclass avx10_v2aux_cvt_widen_masked_b<bits<8> opc, string OpcodeStr> {
  defm Z    : avx10_v2aux_cvt_widen_masked_base<opc, OpcodeStr, v64i8_info,
                                                WriteCvtPH2PSZ>, EVEX_V512;
  defm Z256 : avx10_v2aux_cvt_widen_masked_base<opc, OpcodeStr, v32i8x_info,
                                                WriteCvtPH2PSY>, EVEX_V256;
  defm Z128 : avx10_v2aux_cvt_widen_masked_base<opc, OpcodeStr, v16i8x_info,
                                                WriteCvtPH2PS>, EVEX_V128;
}

// Unpack to Byte: byte unpack with immediate
multiclass avx10_v2aux_shuffle_base<bits<8> opc, string OpcodeStr,
                                    X86VectorVTInfo _,
                                    X86FoldableSchedWrite sched> {
  let ImmT = Imm8 in {
    defm rri : AVX512_maskable<opc, MRMSrcReg, _, (outs _.RC:$dst),
                               (ins _.RC:$src1, u8imm:$src2),
                               OpcodeStr, "$src2, $src1", "$src1, $src2",
                               (_.VT (!cast<Intrinsic>("int_x86_avx10_"#OpcodeStr#"_"#_.Size)
                                      (_.VT _.RC:$src1), (i8 timm:$src2)))>,
                               Sched<[sched]>, EVEX, EVEX_CD8<8, CD8VF>;
    let mayLoad = 1 in
      defm rmi : AVX512_maskable<opc, MRMSrcMem, _, (outs _.RC:$dst),
                                 (ins _.MemOp:$src1, u8imm:$src2),
                                 OpcodeStr, "$src2, $src1", "$src1, $src2",
                                 (_.VT (!cast<Intrinsic>("int_x86_avx10_"#OpcodeStr#"_"#_.Size)
                                        (_.VT (load addr:$src1)), (i8 timm:$src2)))>,
                                 Sched<[sched.Folded]>, EVEX, EVEX_CD8<8, CD8VF>;
  }
}

multiclass avx10_v2aux_shuffle_b<bits<8> opc, string OpcodeStr> {
  defm Z    : avx10_v2aux_shuffle_base<opc, OpcodeStr, v64i8_info,
                                       WriteShuffleZ>, EVEX_V512;
  defm Z256 : avx10_v2aux_shuffle_base<opc, OpcodeStr, v32i8x_info,
                                       WriteShuffle256>, EVEX_V256;
  defm Z128 : avx10_v2aux_shuffle_base<opc, OpcodeStr, v16i8x_info,
                                       WriteShuffle>, EVEX_V128;
}

//===----------------------------------------------------------------------===//
// AVX10 V2 AUX Instruction Definitions
//===----------------------------------------------------------------------===//

//-------------------------------------------------
// Convert from FP32 to FP8
//-------------------------------------------------

let Predicates = [HasAVX10_V2_AUX] in {
  defm VCVTPS2BF8 : avx10_v2aux_cvt_trunc_ps2i8<0x39, "vcvtps2bf8",
                                                X86vcvtps2bf8, X86vmcvtps2bf8>,
                    T_MAP5, XS, EVEX_CD8<32, CD8VF>;
  defm VCVTPS2BF8S : avx10_v2aux_cvt_trunc_ps2i8<0x3B, "vcvtps2bf8s",
                                                 X86vcvtps2bf8s, X86vmcvtps2bf8s>,
                     T_MAP5, XS, EVEX_CD8<32, CD8VF>;
  defm VCVTPS2HF8 : avx10_v2aux_cvt_trunc_ps2i8<0x38, "vcvtps2hf8",
                                                X86vcvtps2hf8, X86vmcvtps2hf8>,
                    T_MAP5, XS, EVEX_CD8<32, CD8VF>;
  defm VCVTPS2HF8S : avx10_v2aux_cvt_trunc_ps2i8<0x3A, "vcvtps2hf8s",
                                                 X86vcvtps2hf8s, X86vmcvtps2hf8s>,
                     T_MAP5, XS, EVEX_CD8<32, CD8VF>;
  defm VCVTROPS2HF8 : avx10_v2aux_cvt_trunc_ps2i8<0x38, "vcvtrops2hf8",
                                                  X86vcvtrops2hf8, X86vmcvtrops2hf8>,
                      T_MAP5, PD, EVEX_CD8<32, CD8VF>;
  defm VCVTROPS2HF8S : avx10_v2aux_cvt_trunc_ps2i8<0x3A, "vcvtrops2hf8s",
                                                   X86vcvtrops2hf8s, X86vmcvtrops2hf8s>,
                       T_MAP5, PD, EVEX_CD8<32, CD8VF>;
}

//-------------------------------------------------
// Convert from FP32 to FP8 with bias
//-------------------------------------------------

let Predicates = [HasAVX10_V2_AUX] in {
  defm VCVTBIASPS2BF8 : avx10_v2aux_cvt_3op_ps<0x39, "vcvtbiasps2bf8",
                                               X86vcvtbiasps2bf8,
                                               X86vmcvtbiasps2bf8>,
                        T_MAP5, PS;
  defm VCVTBIASPS2BF8S : avx10_v2aux_cvt_3op_ps<0x3B, "vcvtbiasps2bf8s",
                                                X86vcvtbiasps2bf8s,
                                                X86vmcvtbiasps2bf8s>,
                         T_MAP5, PS;
  defm VCVTBIASPS2HF8 : avx10_v2aux_cvt_3op_ps<0x38, "vcvtbiasps2hf8",
                                               X86vcvtbiasps2hf8,
                                               X86vmcvtbiasps2hf8>,
                        T_MAP5, PS;
  defm VCVTBIASPS2HF8S : avx10_v2aux_cvt_3op_ps<0x3A, "vcvtbiasps2hf8s",
                                                X86vcvtbiasps2hf8s,
                                                X86vmcvtbiasps2hf8s>,
                         T_MAP5, PS;
}

//-------------------------------------------------
// Convert from FP8 to FP32
//-------------------------------------------------

let Predicates = [HasAVX10_V2_AUX] in {
  defm VCVTBF82PS : avx10_v2aux_cvt_2op_i8_to_f32<0x36, "vcvtbf82ps",
                                                  X86vcvtbf82ps>,
                    PS, T_MAP5, EVEX, EVEX_CD8<32, CD8VQ>, REX_W;
  defm VCVTHF82PS : avx10_v2aux_cvt_2op_i8_to_f32<0x36, "vcvthf82ps",
                                                  X86vcvthf82ps>,
                    PS, T_MAP5, EVEX, EVEX_CD8<32, CD8VQ>;
}

//-------------------------------------------------
// Convert from FP8 to FP4
//-------------------------------------------------

let Predicates = [HasAVX10_V2_AUX] in {
  defm VCVTBF82BF4S : avx10_v2aux_cvt_trunc_b<0x3D, "vcvtbf82bf4s">,
                      T_MAP5, XS, REX_W;
  defm VCVTHF82BF4S : avx10_v2aux_cvt_trunc_b<0x3D, "vcvthf82bf4s">,
                      T_MAP5, XS;
}

//-------------------------------------------------
// Convert from FP8 to FP6
//-------------------------------------------------

// Convert from FP8 to FP6: narrowing conversion, reg-only, no masking
multiclass avx10_v2aux_cvt_narrow_base<bits<8> opc, string OpcodeStr,
                                       X86VectorVTInfo _,
                                       X86FoldableSchedWrite sched> {
  def rr : I<opc, MRMSrcReg, (outs _.RC:$dst),
             (ins _.RC:$src),
             OpcodeStr # "\t{$src, $dst|$dst, $src}",
             [(set _.RC:$dst,
               (_.VT (!cast<Intrinsic>("int_x86_avx10_"#OpcodeStr#"_"#_.Size)
                       (_.VT _.RC:$src))))]>,
           EVEX, Sched<[sched]>;
}

multiclass avx10_v2aux_cvt_narrow_b<bits<8> opc, string OpcodeStr> {
  defm Z    : avx10_v2aux_cvt_narrow_base<opc, OpcodeStr, v64i8_info,
                                          WriteCvtPH2PSZ>, EVEX_V512;
  defm Z256 : avx10_v2aux_cvt_narrow_base<opc, OpcodeStr, v32i8x_info,
                                          WriteCvtPH2PSY>, EVEX_V256;
  defm Z128 : avx10_v2aux_cvt_narrow_base<opc, OpcodeStr, v16i8x_info,
                                          WriteCvtPH2PS>, EVEX_V128;
}

let Predicates = [HasAVX10_V2_AUX] in {
  defm VCVTBF82BF6S : avx10_v2aux_cvt_narrow_b<0x3E, "vcvtbf82bf6s">,
                      T_MAP5, XS, REX_W;
  defm VCVTHF82HF6S : avx10_v2aux_cvt_narrow_b<0x3C, "vcvthf82hf6s">,
                      T_MAP5, XS;
}

//-------------------------------------------------
// Convert from FP4 to FP8 and from FP6 to FP8
//-------------------------------------------------

// VCVTBF42HF8: expanding (2x), with masking, no broadcast
// Z128: xmm{k}{z}, xmm/m64  Z256: ymm{k}{z}, xmm/m128  Z: zmm{k}{z}, ymm/m256
let Predicates = [HasAVX10_V2_AUX] in {
  defm VCVTBF42HF8 : avx10_v2aux_cvt_expand_masked_b<0x37, "vcvtbf42hf8">,
                     T_MAP5, PS;
}
let Predicates = [HasAVX10_V2_AUX] in {
  // VCVTBF62HF8: widening (6-bit to 8-bit) with masking, reg-only
  defm VCVTBF62HF8 : avx10_v2aux_cvt_widen_masked_b<0x37, "vcvtbf62hf8">,
                     T_MAP5, PD, REX_W;

  // VCVTHF62HF8: widening (6-bit to 8-bit) with masking, reg-only
  defm VCVTHF62HF8 : avx10_v2aux_cvt_widen_masked_b<0x37, "vcvthf62hf8">,
                     T_MAP5, PD;
}

//-------------------------------------------------
// Down convert DWord to Byte with symmetric signed saturation
//-------------------------------------------------

let Predicates = [HasAVX10_V2_AUX] in {
  defm VPMOVSSDB : avx512_trunc_db<0x41, "vpmovssdb", X86vtruncss, select_truncss,
                                   SchedWriteVecTruncate, truncstore_ss_vi8,
                                   masked_truncstore_ss_vi8, X86vtruncss,
                                   X86vmtruncss>;

  // Explicit patterns for 512-bit VMTRUNCSS (intrinsic lowering produces this
  // SDNode directly, but avx512_trunc_db generates vselect_mask patterns for Z)
  def : Pat<(v16i8 (X86vmtruncss (v16i32 VR512:$src), (v16i8 VR128X:$src0),
                                  VK16WM:$mask)),
            (VPMOVSSDBZrrk VR128X:$src0, VK16WM:$mask, VR512:$src)>;
  def : Pat<(v16i8 (X86vmtruncss (v16i32 VR512:$src), v16i8x_info.ImmAllZerosV,
                                  VK16WM:$mask)),
            (VPMOVSSDBZrrkz VK16WM:$mask, VR512:$src)>;
}

//-------------------------------------------------
// Unpack to Byte
//-------------------------------------------------

let Predicates = [HasAVX10_V2_AUX] in {
  defm VUNPACKB : avx10_v2aux_shuffle_b<0x3D, "vunpackb">, TA, PS;
}

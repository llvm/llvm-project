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

// Group A multiclass: PS(f32) -> i8 truncating conversion (quarter-size output)
// Output is always xmm for all VL variants.
// Adapts avx512_vcvt_fp for each VL level with explicit dest/src type mappings.
multiclass avx10_v2aux_cvt_trunc_ps2i8<bits<8> opc, string OpcodeStr,
                                        SDPatternOperator OpNode,
                                        SDPatternOperator MaskOpNode> {
  let Predicates = [HasAVX10_V2_AUX] in {
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
}

// Group B multiclass: 3-operand bias PS(f32) -> i8 conversion (quarter-size output)
// bias + f32 source -> i8 dest
// Reuses avx10_convert_3op_packed from X86InstrAVX10.td for each VL variant.
multiclass avx10_v2aux_convert_3op_ps<bits<8> OpCode, string OpcodeStr,
                                       SDPatternOperator OpNode,
                                       SDPatternOperator MaskOpNode> {
  let Predicates = [HasAVX10_V2_AUX] in {
    // Z (512-bit): bias=v64i8(zmm), src=v16f32(zmm), dst=v16i8(xmm)
    // Element counts match (16), so vselect_mask works directly.
    defm Z : avx10_convert_3op_packed<OpCode, OpcodeStr, v16i8x_info,
               v64i8_info, v16f32_info, OpNode, OpNode, WriteCvtPH2PSZ>,
               EVEX_V512, EVEX_CD8<32, CD8VF>;
    // Z256/Z128: use null_frag because element count mismatch between
    // dest (v16i8) and source (v8f32/v4f32) prevents vselect_mask from
    // generating correct masked patterns. Explicit Pat patterns below.
    defm Z256 : avx10_convert_3op_packed<OpCode, OpcodeStr, v16i8x_info,
                  v32i8x_info, v8f32x_info,
                  null_frag, null_frag, WriteCvtPH2PSZ>,
                  EVEX_V256, EVEX_CD8<32, CD8VF>;
    defm Z128 : avx10_convert_3op_packed<OpCode, OpcodeStr, v16i8x_info,
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
}

// Group C multiclass: i8 -> f32 expanding conversion (4x expansion, no broadcast)
multiclass avx10_v2aux_convert_2op_i8_to_f32<string OpcodeStr, bits<8> opc,
                                              SDNode OpNode> {
  let Predicates = [HasAVX10_V2_AUX] in {
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
}

// Group D multiclass: Store-like truncation (reg/mem dest, no masking)
// Uses MRMDestReg/MRMDestMem since destination can be memory operand.
// Source is in reg field, destination is in r/m field.
multiclass avx10_v2aux_trunc_store<bits<8> opc, string OpcodeStr,
                                    X86VectorVTInfo SrcInfo,
                                    X86VectorVTInfo DestInfo,
                                    X86MemOperand x86memop> {
  let hasSideEffects = 0 in {
    def rr : I<opc, MRMDestReg, (outs DestInfo.RC:$dst),
               (ins SrcInfo.RC:$src),
               OpcodeStr # "\t{$src, $dst|$dst, $src}", []>,
               EVEX, Sched<[WriteCvtPH2PSZ]>;
    let mayStore = 1 in
    def mr : I<opc, MRMDestMem, (outs),
               (ins x86memop:$dst, SrcInfo.RC:$src),
               OpcodeStr # "\t{$src, $dst|$dst, $src}", []>,
               EVEX, Sched<[WriteCvtPH2PSZ.Folded]>;
  }
}

// Group F helper: expanding conversion with masking, no broadcast (reg+mem)
multiclass avx10_v2aux_convert_expand_masked<bits<8> opc, string OpcodeStr,
                                              X86VectorVTInfo _dest,
                                              X86VectorVTInfo _src,
                                              X86MemOperand x86memop> {
  let ExeDomain = _dest.ExeDomain in {
    defm rr : AVX512_maskable_custom<opc, MRMSrcReg,
                (outs _dest.RC:$dst),
                (ins _src.RC:$src),
                (ins _dest.RC:$src0, _dest.KRCWM:$mask, _src.RC:$src),
                (ins _dest.KRCWM:$mask, _src.RC:$src),
                OpcodeStr, "$src", "$src", [], [], [],
                "$src0 = $dst">,
               EVEX, Sched<[WriteCvtPH2PSZ]>;
    let mayLoad = 1 in
    defm rm : AVX512_maskable_custom<opc, MRMSrcMem,
                (outs _dest.RC:$dst),
                (ins x86memop:$src),
                (ins _dest.RC:$src0, _dest.KRCWM:$mask, x86memop:$src),
                (ins _dest.KRCWM:$mask, x86memop:$src),
                OpcodeStr, "$src", "$src", [], [], [],
                "$src0 = $dst">,
               EVEX, Sched<[WriteCvtPH2PSZ.Folded]>;
  }
}

// Group F helper: same-size conversion with masking (reg-only)
multiclass avx10_v2aux_convert_samesize_masked<bits<8> opc, string OpcodeStr,
                                                X86VectorVTInfo _> {
  let ExeDomain = _.ExeDomain in {
    defm rr : AVX512_maskable_custom<opc, MRMSrcReg,
                (outs _.RC:$dst),
                (ins _.RC:$src),
                (ins _.RC:$src0, _.KRCWM:$mask, _.RC:$src),
                (ins _.KRCWM:$mask, _.RC:$src),
                OpcodeStr, "$src", "$src", [], [], [],
                "$src0 = $dst">,
               EVEX, Sched<[WriteCvtPH2PSZ]>;
  }
}

// Group H multiclass: Byte unpack with immediate
multiclass avx10_v2aux_unpackb<bits<8> opc, string OpcodeStr,
                                X86VectorVTInfo _> {
  let ImmT = Imm8 in {
    defm ri : AVX512_maskable_custom<opc, MRMSrcReg,
                    (outs _.RC:$dst),
                    (ins _.RC:$src1, u8imm:$src2),
                    (ins _.RC:$src0, _.KRCWM:$mask, _.RC:$src1, u8imm:$src2),
                    (ins _.KRCWM:$mask, _.RC:$src1, u8imm:$src2),
                    OpcodeStr, "$src2, $src1", "$src1, $src2",
                    [], [], [],
                    "$src0 = $dst">,
                    Sched<[WriteShuffle]>;
    let mayLoad = 1 in
    defm mi : AVX512_maskable_custom<opc, MRMSrcMem,
                    (outs _.RC:$dst),
                    (ins _.MemOp:$src1, u8imm:$src2),
                    (ins _.RC:$src0, _.KRCWM:$mask, _.MemOp:$src1, u8imm:$src2),
                    (ins _.KRCWM:$mask, _.MemOp:$src1, u8imm:$src2),
                    OpcodeStr, "$src2, $src1", "$src1, $src2",
                    [], [], [],
                    "$src0 = $dst">,
                    Sched<[WriteShuffle.Folded]>;
  }
}

//===----------------------------------------------------------------------===//
// AVX10 V2 AUX Instruction Definitions
//===----------------------------------------------------------------------===//

//-------------------------------------------------
// Group A: PS->8bit truncating conversions
//-------------------------------------------------

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

//-------------------------------------------------
// Group B: Bias PS->8bit conversions (3-operand)
//-------------------------------------------------

defm VCVTBIASPS2BF8 : avx10_v2aux_convert_3op_ps<0x39, "vcvtbiasps2bf8",
                                                   X86vcvtbiasps2bf8,
                                                   X86vmcvtbiasps2bf8>,
                      T_MAP5, PS;
defm VCVTBIASPS2BF8S : avx10_v2aux_convert_3op_ps<0x3B, "vcvtbiasps2bf8s",
                                                    X86vcvtbiasps2bf8s,
                                                    X86vmcvtbiasps2bf8s>,
                       T_MAP5, PS;
defm VCVTBIASPS2HF8 : avx10_v2aux_convert_3op_ps<0x38, "vcvtbiasps2hf8",
                                                   X86vcvtbiasps2hf8,
                                                   X86vmcvtbiasps2hf8>,
                      T_MAP5, PS;
defm VCVTBIASPS2HF8S : avx10_v2aux_convert_3op_ps<0x3A, "vcvtbiasps2hf8s",
                                                    X86vcvtbiasps2hf8s,
                                                    X86vmcvtbiasps2hf8s>,
                       T_MAP5, PS;

//-------------------------------------------------
// Group C: 8bit->PS expanding conversions
//-------------------------------------------------

defm VCVTBF82PS : avx10_v2aux_convert_2op_i8_to_f32<"vcvtbf82ps", 0x36,
                                                      X86vcvtbf82ps>,
                  PS, T_MAP5, EVEX, EVEX_CD8<32, CD8VQ>, REX_W;
defm VCVTHF82PS : avx10_v2aux_convert_2op_i8_to_f32<"vcvthf82ps", 0x36,
                                                      X86vcvthf82ps>,
                  PS, T_MAP5, EVEX, EVEX_CD8<32, CD8VQ>;

// X //-------------------------------------------------
// Group D: BF8/HF8->BF4S store-like truncations
//-------------------------------------------------

let Predicates = [HasAVX10_V2_AUX] in {
  defm VCVTBF82BF4SZ    : avx10_v2aux_trunc_store<0x3D, "vcvtbf82bf4s",
                             v64i8_info, v32i8x_info, i256mem>,
                           T_MAP5, XS, REX_W, EVEX_V512, EVEX_CD8<8, CD8VH>;
  defm VCVTBF82BF4SZ256 : avx10_v2aux_trunc_store<0x3D, "vcvtbf82bf4s",
                             v32i8x_info, v16i8x_info, i128mem>,
                           T_MAP5, XS, REX_W, EVEX_V256, EVEX_CD8<8, CD8VH>;
  defm VCVTBF82BF4SZ128 : avx10_v2aux_trunc_store<0x3D, "vcvtbf82bf4s",
                             v16i8x_info, v16i8x_info, i64mem>,
                           T_MAP5, XS, REX_W, EVEX_V128, EVEX_CD8<8, CD8VH>;

  defm VCVTHF82BF4SZ    : avx10_v2aux_trunc_store<0x3D, "vcvthf82bf4s",
                             v64i8_info, v32i8x_info, i256mem>,
                           T_MAP5, XS, EVEX_V512, EVEX_CD8<8, CD8VH>;
  defm VCVTHF82BF4SZ256 : avx10_v2aux_trunc_store<0x3D, "vcvthf82bf4s",
                             v32i8x_info, v16i8x_info, i128mem>,
                           T_MAP5, XS, EVEX_V256, EVEX_CD8<8, CD8VH>;
  defm VCVTHF82BF4SZ128 : avx10_v2aux_trunc_store<0x3D, "vcvthf82bf4s",
                             v16i8x_info, v16i8x_info, i64mem>,
                           T_MAP5, XS, EVEX_V128, EVEX_CD8<8, CD8VH>;
}

//-------------------------------------------------
// Group E: Same-size reg-only conversions (no masking)
//-------------------------------------------------

let Predicates = [HasAVX10_V2_AUX], hasSideEffects = 0 in {
  def VCVTBF82BF6SZrr : I<0x3E, MRMSrcReg, (outs VR512:$dst),
                           (ins VR512:$src),
                           "vcvtbf82bf6s\t{$src, $dst|$dst, $src}", []>,
                         EVEX, T_MAP5, XS, REX_W, EVEX_V512,
                         Sched<[WriteCvtPH2PSZ]>;
  def VCVTBF82BF6SZ256rr : I<0x3E, MRMSrcReg, (outs VR256X:$dst),
                              (ins VR256X:$src),
                              "vcvtbf82bf6s\t{$src, $dst|$dst, $src}", []>,
                            EVEX, T_MAP5, XS, REX_W, EVEX_V256,
                            Sched<[WriteCvtPH2PSZ]>;
  def VCVTBF82BF6SZ128rr : I<0x3E, MRMSrcReg, (outs VR128X:$dst),
                              (ins VR128X:$src),
                              "vcvtbf82bf6s\t{$src, $dst|$dst, $src}", []>,
                            EVEX, T_MAP5, XS, REX_W, EVEX_V128,
                            Sched<[WriteCvtPH2PSZ]>;

  def VCVTHF82HF6SZrr : I<0x3C, MRMSrcReg, (outs VR512:$dst),
                           (ins VR512:$src),
                           "vcvthf82hf6s\t{$src, $dst|$dst, $src}", []>,
                         EVEX, T_MAP5, XS, EVEX_V512,
                         Sched<[WriteCvtPH2PSZ]>;
  def VCVTHF82HF6SZ256rr : I<0x3C, MRMSrcReg, (outs VR256X:$dst),
                              (ins VR256X:$src),
                              "vcvthf82hf6s\t{$src, $dst|$dst, $src}", []>,
                            EVEX, T_MAP5, XS, EVEX_V256,
                            Sched<[WriteCvtPH2PSZ]>;
  def VCVTHF82HF6SZ128rr : I<0x3C, MRMSrcReg, (outs VR128X:$dst),
                              (ins VR128X:$src),
                              "vcvthf82hf6s\t{$src, $dst|$dst, $src}", []>,
                            EVEX, T_MAP5, XS, EVEX_V128,
                            Sched<[WriteCvtPH2PSZ]>;
}

// Intrinsic patterns for Group E
let Predicates = [HasAVX10_V2_AUX] in {
  def : Pat<(v16i8 (int_x86_avx10_vcvtbf82bf6s_128 (v16i8 VR128X:$src))),
            (VCVTBF82BF6SZ128rr VR128X:$src)>;
  def : Pat<(v32i8 (int_x86_avx10_vcvtbf82bf6s_256 (v32i8 VR256X:$src))),
            (VCVTBF82BF6SZ256rr VR256X:$src)>;
  def : Pat<(v64i8 (int_x86_avx10_vcvtbf82bf6s_512 (v64i8 VR512:$src))),
            (VCVTBF82BF6SZrr VR512:$src)>;

  def : Pat<(v16i8 (int_x86_avx10_vcvthf82hf6s_128 (v16i8 VR128X:$src))),
            (VCVTHF82HF6SZ128rr VR128X:$src)>;
  def : Pat<(v32i8 (int_x86_avx10_vcvthf82hf6s_256 (v32i8 VR256X:$src))),
            (VCVTHF82HF6SZ256rr VR256X:$src)>;
  def : Pat<(v64i8 (int_x86_avx10_vcvthf82hf6s_512 (v64i8 VR512:$src))),
            (VCVTHF82HF6SZrr VR512:$src)>;
}

//-------------------------------------------------
// Group F: Expanding/same-size conversions with masking
//-------------------------------------------------

// VCVTBF42HF8: expanding (2x), with masking, no broadcast
// Z128: xmm{k}{z}, xmm/m64  Z256: ymm{k}{z}, xmm/m128  Z: zmm{k}{z}, ymm/m256
let Predicates = [HasAVX10_V2_AUX] in {
  defm VCVTBF42HF8Z : avx10_v2aux_convert_expand_masked<0x37, "vcvtbf42hf8",
                         v64i8_info, v32i8x_info, f256mem>,
                       T_MAP5, PS, EVEX_V512, EVEX_CD8<8, CD8VH>;
  defm VCVTBF42HF8Z128 : avx10_v2aux_convert_expand_masked<0x37, "vcvtbf42hf8",
                            v16i8x_info, v16i8x_info, f64mem>,
                          T_MAP5, PS, EVEX_V128, EVEX_CD8<8, CD8VH>;
  defm VCVTBF42HF8Z256 : avx10_v2aux_convert_expand_masked<0x37, "vcvtbf42hf8",
                            v32i8x_info, v16i8x_info, f128mem>,
                          T_MAP5, PS, EVEX_V256, EVEX_CD8<8, CD8VH>;
}
let Predicates = [HasAVX10_V2_AUX] in {
  let ExeDomain = SSEPackedInt in {
    // VCVTBF62HF8: same-size with masking, reg-only (66.MAP5.W1)
    defm VCVTBF62HF8Z    : avx10_v2aux_convert_samesize_masked<0x37,
                              "vcvtbf62hf8", v64i8_info>,
                            T_MAP5, PD, REX_W, EVEX_V512;
    defm VCVTBF62HF8Z256 : avx10_v2aux_convert_samesize_masked<0x37,
                              "vcvtbf62hf8", v32i8x_info>,
                            T_MAP5, PD, REX_W, EVEX_V256;
    defm VCVTBF62HF8Z128 : avx10_v2aux_convert_samesize_masked<0x37,
                              "vcvtbf62hf8", v16i8x_info>,
                            T_MAP5, PD, REX_W, EVEX_V128;

    // VCVTHF62HF8: same-size with masking, reg-only (66.MAP5.W0)
    defm VCVTHF62HF8Z    : avx10_v2aux_convert_samesize_masked<0x37,
                              "vcvthf62hf8", v64i8_info>,
                            T_MAP5, PD, EVEX_V512;
    defm VCVTHF62HF8Z256 : avx10_v2aux_convert_samesize_masked<0x37,
                              "vcvthf62hf8", v32i8x_info>,
                            T_MAP5, PD, EVEX_V256;
    defm VCVTHF62HF8Z128 : avx10_v2aux_convert_samesize_masked<0x37,
                              "vcvthf62hf8", v16i8x_info>,
                            T_MAP5, PD, EVEX_V128;
  }
}

// Intrinsic patterns for Group F (unmasked only; masking via selectb in C header)

let Predicates = [HasAVX10_V2_AUX] in {
  // VCVTBF42HF8
  def : Pat<(v16i8 (int_x86_avx10_vcvtbf42hf8_128 (v16i8 VR128X:$src))),
            (VCVTBF42HF8Z128rr VR128X:$src)>;
  def : Pat<(v32i8 (int_x86_avx10_vcvtbf42hf8_256 (v16i8 VR128X:$src))),
            (VCVTBF42HF8Z256rr VR128X:$src)>;
  def : Pat<(v64i8 (int_x86_avx10_vcvtbf42hf8_512 (v32i8 VR256X:$src))),
            (VCVTBF42HF8Zrr VR256X:$src)>;

  // VCVTBF62HF8
  def : Pat<(v16i8 (int_x86_avx10_vcvtbf62hf8_128 (v16i8 VR128X:$src))),
            (VCVTBF62HF8Z128rr VR128X:$src)>;
  def : Pat<(v32i8 (int_x86_avx10_vcvtbf62hf8_256 (v32i8 VR256X:$src))),
            (VCVTBF62HF8Z256rr VR256X:$src)>;
  def : Pat<(v64i8 (int_x86_avx10_vcvtbf62hf8_512 (v64i8 VR512:$src))),
            (VCVTBF62HF8Zrr VR512:$src)>;

  // VCVTHF62HF8
  def : Pat<(v16i8 (int_x86_avx10_vcvthf62hf8_128 (v16i8 VR128X:$src))),
            (VCVTHF62HF8Z128rr VR128X:$src)>;
  def : Pat<(v32i8 (int_x86_avx10_vcvthf62hf8_256 (v32i8 VR256X:$src))),
            (VCVTHF62HF8Z256rr VR256X:$src)>;
  def : Pat<(v64i8 (int_x86_avx10_vcvthf62hf8_512 (v64i8 VR512:$src))),
            (VCVTHF62HF8Zrr VR512:$src)>;
}

//-------------------------------------------------
// Group G: VPMOVSSDB - Integer DWord->Byte signed saturation
// F3.0F38.W0 0x41
//-------------------------------------------------

let Predicates = [HasAVX10_V2_AUX] in
  defm VPMOVSSDZ : avx512_trunc_common<0x41, "vpmovssdb", X86vtruncs,
                     X86vmtruncs, SchedWriteVecTruncate.ZMM,
                     v16i32_info, v16i8x_info, i128mem>,
                   EVEX_V512, EVEX_CD8<8, CD8VQ>;
let Predicates = [HasVLX, HasAVX10_V2_AUX] in {
  defm VPMOVSSDZ256 : avx512_trunc_common<0x41, "vpmovssdb", X86vtruncs,
                        X86vmtruncs, SchedWriteVecTruncate.YMM,
                        v8i32x_info, v16i8x_info, i64mem>,
                      EVEX_V256, EVEX_CD8<8, CD8VQ>;
  defm VPMOVSSDZ128 : avx512_trunc_common<0x41, "vpmovssdb", X86vtruncs,
                        X86vmtruncs, SchedWriteVecTruncate.XMM,
                        v4i32x_info, v16i8x_info, i32mem>,
                      EVEX_V128, EVEX_CD8<8, CD8VQ>;
}

//-------------------------------------------------
// Group H: VUNPACKB - Byte unpack with immediate
//-------------------------------------------------

let Predicates = [HasAVX10_V2_AUX] in {
  defm VUNPACKBZ    : avx10_v2aux_unpackb<0x3D, "vunpackb", v64i8_info>,
                      EVEX_V512, TA, PS, EVEX, EVEX_CD8<8, CD8VF>;
  defm VUNPACKBZ256 : avx10_v2aux_unpackb<0x3D, "vunpackb", v32i8x_info>,
                      EVEX_V256, TA, PS, EVEX, EVEX_CD8<8, CD8VF>;
  defm VUNPACKBZ128 : avx10_v2aux_unpackb<0x3D, "vunpackb", v16i8x_info>,
                      EVEX_V128, TA, PS, EVEX, EVEX_CD8<8, CD8VF>;

  // Intrinsic patterns for Group H (unmasked only; masking via selectb in C header)
  def : Pat<(v16i8 (int_x86_avx10_vunpackb_128
              (v16i8 VR128X:$src1), timm:$src2)),
            (VUNPACKBZ128ri VR128X:$src1, timm:$src2)>;
  def : Pat<(v32i8 (int_x86_avx10_vunpackb_256
              (v32i8 VR256X:$src1), timm:$src2)),
            (VUNPACKBZ256ri VR256X:$src1, timm:$src2)>;
  def : Pat<(v64i8 (int_x86_avx10_vunpackb_512
              (v64i8 VR512:$src1), timm:$src2)),
            (VUNPACKBZri VR512:$src1, timm:$src2)>;
}

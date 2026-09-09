//===---- X86InstrACE.td - ACE Instruction Set Extension --*- tablegen -*--===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file describes the instructions that make up the ACE (AI Compute
// Extensions) instruction set.
//
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// ACE instructions

let Predicates = [HasACEV1, In64BitMode] in {

// BSRINIT - Initialize Block Scale Register
let SchedRW = [WriteSystem], hasSideEffects = 1, Defs = [BSR0] in {
  def BSRINIT : I<0x49, MRM_C0, (outs), (ins),
                  "bsrinit\t{%bsr0|bsr0}", []>,
                  VEX, XD, T8, REX_W;
}

// BSRMOVF - Move Full to BSR
let SchedRW = [WriteSystem], Defs = [BSR0] in {
  def BSRMOVFrr : I<0x95, MRM0r, (outs),
                    (ins VR512:$src1, VR512:$src2),
                    "bsrmovf\t{$src2, $src1, %bsr0|bsr0, $src1, $src2}",
                    [(int_x86_acev1_bsr0movf (v64i8 VR512:$src1), (v64i8 VR512:$src2))]>,
                    EVEX, T_MAP6, REX_W, EVEX_V512, VVVV;

  let mayLoad = 1 in
  def BSRMOVFrm : I<0x95, MRM0m, (outs),
                    (ins VR512:$src1, f512mem:$src2),
                    "bsrmovf\t{$src2, $src1, %bsr0|bsr0, $src1, $src2}", []>,
                    EVEX, T_MAP6, REX_W, EVEX_V512, EVEX_CD8<64, CD8VF>, VVVV;
}

// BSRMOVH - Move Half to/from BSR
let SchedRW = [WriteSystem] in {
  let Defs = [BSR0], Uses = [BSR0] in {
    def BSRMOVHrr_set : I<0x95, MRM0r, (outs),
                          (ins VR512:$src),
                          "bsrmovh\t{$src, %bsr0|bsr0, $src}",
                          [(int_x86_acev1_bsr0movhinsert (v64i8 VR512:$src))]>,
                          EVEX, XD, T_MAP6, REX_W, EVEX_V512;

    let mayLoad = 1 in
    def BSRMOVHrm_set : I<0x95, MRM0m, (outs),
                          (ins f512mem:$src),
                          "bsrmovh\t{$src, %bsr0|bsr0, $src}", []>,
                          EVEX, XD, T_MAP6, REX_W, EVEX_V512, EVEX_CD8<64, CD8VF>;
  }

  let Uses = [BSR0] in {
    def BSRMOVHrr_get : I<0x95, MRM0r, (outs VR512:$dst),
                          (ins),
                          "bsrmovh\t{%bsr0, $dst|$dst, bsr0}",
                          [(set (v64i8 VR512:$dst), (int_x86_acev1_bsr0movhextract))]>,
                          EVEX, XD, T_MAP6, EVEX_V512;

    let mayStore = 1 in
    def BSRMOVHmr_get : I<0x95, MRM0m, (outs),
                          (ins f512mem:$dst),
                          "bsrmovh\t{%bsr0, $dst|$dst, bsr0}",
                          [(store (v64i8 (int_x86_acev1_bsr0movhextract)), addr:$dst)]>,
                          EVEX, XD, T_MAP6, EVEX_V512, EVEX_CD8<64, CD8VF>;
  }
}

// BSRMOVL - Move Low to/from BSR
let SchedRW = [WriteSystem] in {
  let Defs = [BSR0], Uses = [BSR0] in {
    def BSRMOVLrr_set : I<0x95, MRM0r, (outs),
                          (ins VR512:$src),
                          "bsrmovl\t{$src, %bsr0|bsr0, $src}",
                          [(int_x86_acev1_bsr0movlinsert (v64i8 VR512:$src))]>,
                          EVEX, XS, T_MAP6, REX_W, EVEX_V512;

    let mayLoad = 1 in
    def BSRMOVLrm_set : I<0x95, MRM0m, (outs),
                          (ins f512mem:$src),
                          "bsrmovl\t{$src, %bsr0|bsr0, $src}", []>,
                          EVEX, XS, T_MAP6, REX_W, EVEX_V512, EVEX_CD8<64, CD8VF>;
  }

  let Uses = [BSR0] in {
    def BSRMOVLrr_get : I<0x95, MRM0r, (outs VR512:$dst),
                          (ins),
                          "bsrmovl\t{%bsr0, $dst|$dst, bsr0}",
                          [(set (v64i8 VR512:$dst), (int_x86_acev1_bsr0movlextract))]>,
                          EVEX, XS, T_MAP6, EVEX_V512;

    let mayStore = 1 in
    def BSRMOVLmr_get : I<0x95, MRM0m, (outs),
                          (ins f512mem:$dst),
                          "bsrmovl\t{%bsr0, $dst|$dst, bsr0}",
                          [(store (v64i8 (int_x86_acev1_bsr0movlextract)), addr:$dst)]>,
                          EVEX, XS, T_MAP6, EVEX_V512, EVEX_CD8<64, CD8VF>;
  }
}

// TILEMOVCOL - Move Column from Vector to Tile
let SchedRW = [WriteSystem] in {
  def TILEMOVCOLtri : Ii8<0x2F, MRMSrcReg, (outs TILE:$dst),
                          (ins VR512:$src, u8imm:$idx),
                          "tilemovcol\t{$idx, $src, $dst|$dst, $src, $idx}", []>,
                          EVEX, PD, TA, REX_W, EVEX_V512;

  def TILEMOVCOLtre : I<0x4B, MRMSrcReg4VOp3, (outs TILE:$dst),
                        (ins VR512:$src, GR32:$idx),
                        "tilemovcol\t{$idx, $src, $dst|$dst, $src, $idx}", []>,
                        EVEX, PD, T8, REX_W, VVVV, EVEX_V512;
}

// TILEMOVROW - Move Row from Vector to Tile
let SchedRW = [WriteSystem] in {
  def TILEMOVROWtri : Ii8<0x07, MRMSrcReg, (outs TILE:$dst),
                          (ins VR512:$src, u8imm:$idx),
                          "tilemovrow\t{$idx, $src, $dst|$dst, $src, $idx}", []>,
                          EVEX, PD, TA, REX_W, EVEX_V512;

  def TILEMOVROWtre : I<0x4A, MRMSrcReg4VOp3, (outs TILE:$dst),
                        (ins VR512:$src, GR32:$idx),
                        "tilemovrow\t{$idx, $src, $dst|$dst, $src, $idx}", []>,
                        EVEX, PD, T8, REX_W, VVVV, EVEX_V512;
}

// Outer Product Instructions
// TOP2BF16PS - Outer Product BF16 to Single Precision
let Constraints = "$src1 = $dst", SchedRW = [WriteSystem] in {
  def TOP2BF16PStrr : I<0x5C, MRMSrcReg4VOp3, (outs TILE:$dst),
                        (ins TILE:$src1, VR512:$src2, VR512:$src3),
                        "top2bf16ps\t{$src3, $src2, $dst|$dst, $src2, $src3}", []>,
                        EVEX, VVVV, XS, T8, EVEX_V512;
}

// TOP4 Instructions - integer outer products
let Constraints = "$src1 = $dst", SchedRW = [WriteSystem] in {
  def TOP4BUUDtrr : I<0x5E, MRMSrcReg4VOp3, (outs TILE:$dst),
                      (ins TILE:$src1, VR512:$src2, VR512:$src3),
                      "top4buud\t{$src3, $src2, $dst|$dst, $src2, $src3}", []>,
                      EVEX, VVVV, T8, EVEX_V512;

  def TOP4BUSDtrr : I<0x5E, MRMSrcReg4VOp3, (outs TILE:$dst),
                      (ins TILE:$src1, VR512:$src2, VR512:$src3),
                      "top4busd\t{$src3, $src2, $dst|$dst, $src2, $src3}", []>,
                      EVEX, VVVV, PD, T8, EVEX_V512;

  def TOP4BSSDtrr : I<0x5E, MRMSrcReg4VOp3, (outs TILE:$dst),
                      (ins TILE:$src1, VR512:$src2, VR512:$src3),
                      "top4bssd\t{$src3, $src2, $dst|$dst, $src2, $src3}", []>,
                      EVEX, VVVV, XD, T8, EVEX_V512;

  def TOP4BSUDtrr : I<0x5E, MRMSrcReg4VOp3, (outs TILE:$dst),
                      (ins TILE:$src1, VR512:$src2, VR512:$src3),
                      "top4bsud\t{$src3, $src2, $dst|$dst, $src2, $src3}", []>,
                      EVEX, VVVV, XS, T8, EVEX_V512;
}

// TOP4MX Instructions - Mixed Precision Outer Products with Immediate
let Constraints = "$src1 = $dst", SchedRW = [WriteSystem], Uses = [BSR0] in {
  def TOP4MXBF8PStrri : Ii8<0x8D, MRMSrcReg, (outs TILE:$dst),
                            (ins TILE:$src1, VR512:$src3, VR512:$src2, u8imm:$imm),
                            "top4mxbf8ps\t{$imm, $src3, $src2, $dst|$dst, $src2, $src3, $imm}", []>,
                            EVEX, VVVV, TA, EVEX_V512;

  def TOP4MXHF8PStrri : Ii8<0x8D, MRMSrcReg, (outs TILE:$dst),
                            (ins TILE:$src1, VR512:$src3, VR512:$src2, u8imm:$imm),
                            "top4mxhf8ps\t{$imm, $src3, $src2, $dst|$dst, $src2, $src3, $imm}", []>,
                            EVEX, VVVV, PD, TA, EVEX_V512;

  def TOP4MXBHF8PStrri : Ii8<0x8D, MRMSrcReg, (outs TILE:$dst),
                             (ins TILE:$src1, VR512:$src3, VR512:$src2, u8imm:$imm),
                             "top4mxbhf8ps\t{$imm, $src3, $src2, $dst|$dst, $src2, $src3, $imm}", []>,
                             EVEX, VVVV, XD, TA, EVEX_V512;

  def TOP4MXHBF8PStrri : Ii8<0x8D, MRMSrcReg, (outs TILE:$dst),
                             (ins TILE:$src1, VR512:$src3, VR512:$src2, u8imm:$imm),
                             "top4mxhbf8ps\t{$imm, $src3, $src2, $dst|$dst, $src2, $src3, $imm}", []>,
                             EVEX, VVVV, XS, TA, EVEX_V512;

  def TOP4MXBSSPStrri : Ii8<0x8F, MRMSrcReg, (outs TILE:$dst),
                            (ins TILE:$src1, VR512:$src3, VR512:$src2, u8imm:$imm),
                            "top4mxbssps\t{$imm, $src3, $src2, $dst|$dst, $src2, $src3, $imm}", []>,
                            EVEX, VVVV, XD, TA, EVEX_V512;
}

//===----------------------------------------------------------------------===//
// ACE Pseudo Instructions
// These are used by intrinsics and expanded via custom inserter
//===----------------------------------------------------------------------===//

// Pseudo instructions for user-facing intrinsics (immediate tile IDs).
// usesCustomInserter expands these to real instructions in X86ISelLowering.
let usesCustomInserter = 1, hasNoSchedulingInfo = 1 in {
  def PBSRINIT : PseudoI<(outs), (ins),
                         [(int_x86_acev1_bsr0init)]>;

  def PTILEMOVCOLtri : PseudoI<(outs), (ins u8imm:$dst, VR512:$src, u8imm:$idx), []>;

  def PTILEMOVCOLtre : PseudoI<(outs), (ins u8imm:$dst, VR512:$src, GR32:$idx), []>;

  def PTILEMOVROWtri : PseudoI<(outs), (ins u8imm:$dst, VR512:$src, u8imm:$idx), []>;

  def PTILEMOVROWtre : PseudoI<(outs), (ins u8imm:$dst, VR512:$src, GR32:$idx), []>;

  def PTOP2BF16PStrr : PseudoI<(outs), (ins u8imm:$dst, VR512:$src1, VR512:$src2),
                               [(int_x86_acev1_top2bf16ps timm:$dst, VR512:$src1, VR512:$src2)]>;

  def PTOP4BUUDtrr : PseudoI<(outs), (ins u8imm:$dst, VR512:$src1, VR512:$src2),
                             [(int_x86_acev1_top4buud timm:$dst, VR512:$src1, VR512:$src2)]>;

  def PTOP4BUSDtrr : PseudoI<(outs), (ins u8imm:$dst, VR512:$src1, VR512:$src2),
                             [(int_x86_acev1_top4busd timm:$dst, VR512:$src1, VR512:$src2)]>;

  def PTOP4BSSDtrr : PseudoI<(outs), (ins u8imm:$dst, VR512:$src1, VR512:$src2),
                             [(int_x86_acev1_top4bssd timm:$dst, VR512:$src1, VR512:$src2)]>;

  def PTOP4BSUDtrr : PseudoI<(outs), (ins u8imm:$dst, VR512:$src1, VR512:$src2),
                             [(int_x86_acev1_top4bsud timm:$dst, VR512:$src1, VR512:$src2)]>;

  def PTOP4MXHF8PStrri : PseudoI<(outs), (ins u8imm:$dst, VR512:$src1, VR512:$src2, u8imm:$imm),
                                 [(int_x86_acev1_top4mxhf8ps timm:$dst, VR512:$src1, VR512:$src2, timm:$imm)]>;

  def PTOP4MXBHF8PStrri : PseudoI<(outs), (ins u8imm:$dst, VR512:$src1, VR512:$src2, u8imm:$imm),
                                  [(int_x86_acev1_top4mxbhf8ps timm:$dst, VR512:$src1, VR512:$src2, timm:$imm)]>;

  def PTOP4MXHBF8PStrri : PseudoI<(outs), (ins u8imm:$dst, VR512:$src1, VR512:$src2, u8imm:$imm),
                                  [(int_x86_acev1_top4mxhbf8ps timm:$dst, VR512:$src1, VR512:$src2, timm:$imm)]>;

  def PTOP4MXBF8PStrri : PseudoI<(outs), (ins u8imm:$dst, VR512:$src1, VR512:$src2, u8imm:$imm),
                                 [(int_x86_acev1_top4mxbf8ps timm:$dst, VR512:$src1, VR512:$src2, timm:$imm)]>;

  def PTOP4MXBSSPStrri : PseudoI<(outs), (ins u8imm:$dst, VR512:$src1, VR512:$src2, u8imm:$imm),
                                 [(int_x86_acev1_top4mxbssps timm:$dst, VR512:$src1, VR512:$src2, timm:$imm)]>;
}

// Pseudo instructions for internal intrinsics (register allocation).
let isPseudo = true, hasNoSchedulingInfo = 1, usesCustomInserter = 1 in {
  def PTILEMOVCOLtreV : PseudoI<(outs TILE:$dst),
                                (ins GR16:$row, GR16:$col, VR512:$src, GR32:$idx),
                                [(set TILE:$dst,
                                  (int_x86_acev1_tilemovcolinsert_internal GR16:$row, GR16:$col,
                                   VR512:$src, GR32:$idx))]>;

  def PTILEMOVROWtreV : PseudoI<(outs TILE:$dst),
                                (ins GR16:$row, GR16:$col, VR512:$src, GR32:$idx),
                                [(set TILE:$dst,
                                  (int_x86_acev1_tilemovrowinsert_internal GR16:$row, GR16:$col,
                                   VR512:$src, GR32:$idx))]>;

  let Constraints = "$src1 = $dst" in
  def PTOP2BF16PStrrV : PseudoI<(outs TILE:$dst),
                                (ins GR16:$row, GR16:$col, GR16:$k,
                                     TILE:$src1, VR512:$src2, VR512:$src3),
                                [(set TILE:$dst,
                                  (int_x86_acev1_top2bf16ps_internal GR16:$row, GR16:$col, GR16:$k,
                                   TILE:$src1, VR512:$src2, VR512:$src3))]>;

  let Constraints = "$src1 = $dst" in
  def PTOP4BUUDtrrV : PseudoI<(outs TILE:$dst),
                              (ins GR16:$row, GR16:$col, GR16:$k,
                                   TILE:$src1, VR512:$src2, VR512:$src3),
                              [(set TILE:$dst,
                                (int_x86_acev1_top4buud_internal GR16:$row, GR16:$col, GR16:$k,
                                 TILE:$src1, VR512:$src2, VR512:$src3))]>;

  let Constraints = "$src1 = $dst" in
  def PTOP4BUSDtrrV : PseudoI<(outs TILE:$dst),
                              (ins GR16:$row, GR16:$col, GR16:$k,
                                   TILE:$src1, VR512:$src2, VR512:$src3),
                              [(set TILE:$dst,
                                (int_x86_acev1_top4busd_internal GR16:$row, GR16:$col, GR16:$k,
                                 TILE:$src1, VR512:$src2, VR512:$src3))]>;

  let Constraints = "$src1 = $dst" in
  def PTOP4BSSDtrrV : PseudoI<(outs TILE:$dst),
                              (ins GR16:$row, GR16:$col, GR16:$k,
                                   TILE:$src1, VR512:$src2, VR512:$src3),
                              [(set TILE:$dst,
                                (int_x86_acev1_top4bssd_internal GR16:$row, GR16:$col, GR16:$k,
                                 TILE:$src1, VR512:$src2, VR512:$src3))]>;

  let Constraints = "$src1 = $dst" in
  def PTOP4BSUDtrrV : PseudoI<(outs TILE:$dst),
                              (ins GR16:$row, GR16:$col, GR16:$k,
                                   TILE:$src1, VR512:$src2, VR512:$src3),
                              [(set TILE:$dst,
                                (int_x86_acev1_top4bsud_internal GR16:$row, GR16:$col, GR16:$k,
                                 TILE:$src1, VR512:$src2, VR512:$src3))]>;

  let Constraints = "$src1 = $dst", Uses = [BSR0] in
  def PTOP4MXHF8PStrriV : PseudoI<(outs TILE:$dst),
                                  (ins GR16:$row, GR16:$col, GR16:$k, u8imm:$imm,
                                       TILE:$src1, VR512:$src2, VR512:$src3),
                                  [(set TILE:$dst,
                                    (int_x86_acev1_top4mxhf8ps_internal GR16:$row, GR16:$col, GR16:$k, timm:$imm,
                                     TILE:$src1, VR512:$src2, VR512:$src3))]>;

  let Constraints = "$src1 = $dst", Uses = [BSR0] in
  def PTOP4MXBHF8PStrriV : PseudoI<(outs TILE:$dst),
                                   (ins GR16:$row, GR16:$col, GR16:$k, u8imm:$imm,
                                        TILE:$src1, VR512:$src2, VR512:$src3),
                                   [(set TILE:$dst,
                                     (int_x86_acev1_top4mxbhf8ps_internal GR16:$row, GR16:$col, GR16:$k, timm:$imm,
                                      TILE:$src1, VR512:$src2, VR512:$src3))]>;

  let Constraints = "$src1 = $dst", Uses = [BSR0] in
  def PTOP4MXHBF8PStrriV : PseudoI<(outs TILE:$dst),
                                   (ins GR16:$row, GR16:$col, GR16:$k, u8imm:$imm,
                                        TILE:$src1, VR512:$src2, VR512:$src3),
                                   [(set TILE:$dst,
                                     (int_x86_acev1_top4mxhbf8ps_internal GR16:$row, GR16:$col, GR16:$k, timm:$imm,
                                      TILE:$src1, VR512:$src2, VR512:$src3))]>;

  let Constraints = "$src1 = $dst", Uses = [BSR0] in
  def PTOP4MXBF8PStrriV : PseudoI<(outs TILE:$dst),
                                  (ins GR16:$row, GR16:$col, GR16:$k, u8imm:$imm,
                                       TILE:$src1, VR512:$src2, VR512:$src3),
                                  [(set TILE:$dst,
                                    (int_x86_acev1_top4mxbf8ps_internal GR16:$row, GR16:$col, GR16:$k, timm:$imm,
                                     TILE:$src1, VR512:$src2, VR512:$src3))]>;

  let Constraints = "$src1 = $dst", Uses = [BSR0] in
  def PTOP4MXBSSPStrriV : PseudoI<(outs TILE:$dst),
                                  (ins GR16:$row, GR16:$col, GR16:$k, u8imm:$imm,
                                       TILE:$src1, VR512:$src2, VR512:$src3),
                                  [(set TILE:$dst,
                                    (int_x86_acev1_top4mxbssps_internal GR16:$row, GR16:$col, GR16:$k, timm:$imm,
                                     TILE:$src1, VR512:$src2, VR512:$src3))]>;
}

// Pseudo instructions for tile spill and reload (register allocation)
let isPseudo = true, hasNoSchedulingInfo = 1 in {
  let mayStore = 1 in
  def ACE_TILESPILL : PseudoI<(outs), (ins opaquemem:$dst, TILE:$src), []>;

  let mayLoad = 1 in
  def ACE_TILERELOAD : PseudoI<(outs TILE:$dst), (ins opaquemem:$src), []>;
}

} // end HasACEV1, In64BitMode

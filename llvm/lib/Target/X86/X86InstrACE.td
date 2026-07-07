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
// VEX.128.F2.0F38.W1 49 11:000:000 - BSRINIT bsr0
// BSR is implicit destination, ModRM = 11:000:000 (mod=11, reg=000, r/m=000)
// Note: BSR0 is implicit (via Defs), not in operand list. Asm syntax has no explicit operand.
let SchedRW = [WriteSystem], hasSideEffects = 1, Defs = [BSR0] in {
  def BSRINIT : I<0x49, MRM_C0, (outs), (ins),
                  "bsrinit", []>,
               VEX, XD, T8, REX_W;
}

// BSRMOVF - Move Full to BSR
// EVEX.512.NP.MAP6.W1 95 11:000:bbb - BSRMOVF bsr0, zmm1, zmm2/m512
// ModRM.reg = 000 (opcode extension), EVEX.vvvv encodes zmm1, ModRM.r/m encodes zmm2
// BSR is implicit destination (via Defs)
let SchedRW = [WriteVecLogic], Defs = [BSR0] in {
  def BSRMOVFrr : I<0x95, MRM0r, (outs),
                    (ins VR512:$src1, VR512:$src2),
                    "bsrmovf\t{$src2, $src1|$src1, $src2}",
                    [(int_x86_bsrmovf (v16i32 VR512:$src1), (v16i32 VR512:$src2))]>,
                 EVEX, T_MAP6, REX_W, EVEX_V512, VVVV;

  let mayLoad = 1 in
  def BSRMOVFrm : I<0x95, MRM0m, (outs),
                    (ins VR512:$src1, f512mem:$src2),
                    "bsrmovf\t{$src2, $src1|$src1, $src2}", []>,
                 EVEX, T_MAP6, REX_W, EVEX_V512, EVEX_CD8<64, CD8VF>, VVVV;
}

// BSRMOVH - Move Half to/from BSR
// EVEX.512.F2.MAP6.W1 95 /r - BSRMOVH bsr0, zmm1/m512, bsr0 (load, W1)
// EVEX.512.F2.MAP6.W0 95 /r - BSRMOVH zmm1/m512, bsr0 (store, W0)
// BSR is implicit operand (via Defs/Uses), ModRM.reg field = 0
// Use MRM0r/MRM0m: ModRM.reg = 0 (BSR implicit), ModRM.r/m = source/destination
// AsmString uses "_set"/"_get" suffix for asm matching; printer uses custom print.
let SchedRW = [WriteVecLogic] in {
  // Set BSR high half (W1): BSR is both input and output (read-modify-write)
  let Defs = [BSR0], Uses = [BSR0] in {
    def BSRMOVHrr_set : I<0x95, MRM0r, (outs),
                          (ins VR512:$src),
                          "bsrmovh_set\t$src",
                          [(int_x86_bsrmovh_set (v16i32 VR512:$src))]>,
                       EVEX, XD, T_MAP6, REX_W, EVEX_V512;

    let mayLoad = 1 in
    def BSRMOVHrm_set : I<0x95, MRM0m, (outs),
                          (ins f512mem:$src),
                          "bsrmovh_set\t$src", []>,
                       EVEX, XD, T_MAP6, REX_W, EVEX_V512, EVEX_CD8<64, CD8VF>;
  }

  // Get BSR high half (W0): BSR is implicit source, ZMM/mem is explicit destination
  let Uses = [BSR0] in {
    def BSRMOVHrr_get : I<0x95, MRM0r, (outs VR512:$dst),
                          (ins),
                          "bsrmovh_get\t$dst",
                          [(set (v16i32 VR512:$dst), (int_x86_bsrmovh_get))]>,
                       EVEX, XD, T_MAP6, EVEX_V512;

    let mayStore = 1 in
    def BSRMOVHmr_get : I<0x95, MRM0m, (outs),
                          (ins f512mem:$dst),
                          "bsrmovh_get\t$dst", []>,
                       EVEX, XD, T_MAP6, EVEX_V512, EVEX_CD8<64, CD8VF>;
  }
}

// BSRMOVL - Move Low to/from BSR
// EVEX.512.F3.MAP6.W1 95 /r - BSRMOVL bsr0, zmm1/m512 (load, W1)
// EVEX.512.F3.MAP6.W0 95 /r - BSRMOVL zmm1/m512, bsr0 (store, W0)
// BSR is implicit operand (via Defs/Uses), ModRM.reg field = 0
// Use MRM0r/MRM0m: ModRM.reg = 0 (BSR implicit), ModRM.r/m = source/destination
// AsmString uses "_set"/"_get" suffix for asm matching; printer uses custom print.
let SchedRW = [WriteVecLogic] in {
  // Set BSR low half (W1): BSR is both input and output (read-modify-write)
  let Defs = [BSR0], Uses = [BSR0] in {
    def BSRMOVLrr_set : I<0x95, MRM0r, (outs),
                          (ins VR512:$src),
                          "bsrmovl_set\t$src",
                          [(int_x86_bsrmovl_set (v16i32 VR512:$src))]>,
                       EVEX, XS, T_MAP6, REX_W, EVEX_V512;

    let mayLoad = 1 in
    def BSRMOVLrm_set : I<0x95, MRM0m, (outs),
                          (ins f512mem:$src),
                          "bsrmovl_set\t$src", []>,
                       EVEX, XS, T_MAP6, REX_W, EVEX_V512, EVEX_CD8<64, CD8VF>;
  }

  // Get BSR low half (W0): BSR is implicit source, ZMM/mem is explicit destination
  let Uses = [BSR0] in {
    def BSRMOVLrr_get : I<0x95, MRM0r, (outs VR512:$dst),
                          (ins),
                          "bsrmovl_get\t$dst",
                          [(set (v16i32 VR512:$dst), (int_x86_bsrmovl_get))]>,
                       EVEX, XS, T_MAP6, EVEX_V512;

    let mayStore = 1 in
    def BSRMOVLmr_get : I<0x95, MRM0m, (outs),
                          (ins f512mem:$dst),
                          "bsrmovl_get\t$dst", []>,
                       EVEX, XS, T_MAP6, EVEX_V512, EVEX_CD8<64, CD8VF>;
  }
}

// TILEMOVCOL - Move Column from Vector to Tile
// EVEX.512.66.0F3A.W1 2F /r ib - TILEMOVCOL tmm1, zmm2, imm8
// EVEX.512.66.0F38.W1 4B /r - TILEMOVCOL tmm1, zmm2, r32
// For r32 variant: GR32 is encoded in EVEX.vvvv (use MRMSrcReg4VOp3 + VVVV)
let SchedRW = [WriteVecLogic] in {
  def TILEMOVCOLri : Ii8<0x2F, MRMSrcReg, (outs TILE:$dst),
                         (ins VR512:$src, u8imm:$idx),
                         "tilemovcol\t{$idx, $src, $dst|$dst, $src, $idx}", []>,
                    EVEX, PD, TA, REX_W, EVEX_V512;

  def TILEMOVCOLrr : I<0x4B, MRMSrcReg4VOp3, (outs TILE:$dst),
                       (ins VR512:$src, GR32:$idx),
                       "tilemovcol\t{$idx, $src, $dst|$dst, $src, $idx}", []>,
                    EVEX, PD, T8, REX_W, VVVV, EVEX_V512;
}

// TILEMOVROW - Move Row from Vector to Tile
// EVEX.512.66.0F3A.W1 07 /r ib - TILEMOVROW tmm1, zmm2, imm8
// EVEX.512.66.0F38.W1 4A /r - TILEMOVROW tmm1, zmm2, r32
// For r32 variant: GR32 is encoded in EVEX.vvvv (use MRMSrcReg4VOp3 + VVVV)
let SchedRW = [WriteVecLogic] in {
  def TILEMOVROWri : Ii8<0x07, MRMSrcReg, (outs TILE:$dst),
                         (ins VR512:$src, u8imm:$idx),
                         "tilemovrow\t{$idx, $src, $dst|$dst, $src, $idx}", []>,
                    EVEX, PD, TA, REX_W, EVEX_V512;

  def TILEMOVROWrr : I<0x4A, MRMSrcReg4VOp3, (outs TILE:$dst),
                       (ins VR512:$src, GR32:$idx),
                       "tilemovrow\t{$idx, $src, $dst|$dst, $src, $idx}", []>,
                    EVEX, PD, T8, REX_W, VVVV, EVEX_V512;
}

// Outer Product Instructions
// TOP2BF16PS - Outer Product BF16 to Single Precision
// EVEX.512.F3.0F38.W0 5C /r - TOP2BF16PS tmm1, zmm2, zmm3
// Operand encoding from ACE spec page 52:
//   Operand 1: ModRM:reg (rw) = tmm1
//   Operand 2: ModRM:r/m (r) = zmm2
//   Operand 3: EVEX.vvvv (r) = zmm3
// Use MRMSrcReg4VOp3: operand 1 → r/m, operand 2 → vvvv (so swap src2/src3 order)
let Constraints = "$src1 = $dst", SchedRW = [WriteVecLogic] in {
  def TOP2BF16PSrrr : I<0x5C, MRMSrcReg4VOp3, (outs TILE:$dst),
                        (ins TILE:$src1, VR512:$src2, VR512:$src3),
                        "top2bf16ps\t{$src3, $src2, $dst|$dst, $src2, $src3}", []>,
                     EVEX, VVVV, XS, T8, EVEX_V512;
}

// TOP4 Instructions - All use opcode 0x5E, differentiated by prefix
// EVEX.512.NP.0F38.W0 5E /r - TOP4BUUD tmm1, zmm2, zmm3 (unsigned/unsigned)
// EVEX.512.66.0F38.W0 5E /r - TOP4BUSD tmm1, zmm2, zmm3 (unsigned/signed)
// EVEX.512.F2.0F38.W0 5E /r - TOP4BSSD tmm1, zmm2, zmm3 (signed/signed)
// EVEX.512.F3.0F38.W0 5E /r - TOP4BSUD tmm1, zmm2, zmm3 (signed/unsigned)
// Operand encoding from ACE spec page 54:
//   Operand 1: ModRM:reg (rw) = tmm1
//   Operand 2: ModRM:r/m (r) = zmm2
//   Operand 3: EVEX.vvvv (r) = zmm3
// Use MRMSrcReg4VOp3: operand 1 → r/m, operand 2 → vvvv (so swap src2/src3 order)
let Constraints = "$src1 = $dst", SchedRW = [WriteVecLogic] in {
  def TOP4BUUDrrr : I<0x5E, MRMSrcReg4VOp3, (outs TILE:$dst),
                      (ins TILE:$src1, VR512:$src2, VR512:$src3),
                      "top4buud\t{$src3, $src2, $dst|$dst, $src2, $src3}", []>,
                   EVEX, VVVV, T8, EVEX_V512;

  def TOP4BUSDrrr : I<0x5E, MRMSrcReg4VOp3, (outs TILE:$dst),
                      (ins TILE:$src1, VR512:$src2, VR512:$src3),
                      "top4busd\t{$src3, $src2, $dst|$dst, $src2, $src3}", []>,
                   EVEX, VVVV, PD, T8, EVEX_V512;

  def TOP4BSSDrrr : I<0x5E, MRMSrcReg4VOp3, (outs TILE:$dst),
                      (ins TILE:$src1, VR512:$src2, VR512:$src3),
                      "top4bssd\t{$src3, $src2, $dst|$dst, $src2, $src3}", []>,
                   EVEX, VVVV, XD, T8, EVEX_V512;

  def TOP4BSUDrrr : I<0x5E, MRMSrcReg4VOp3, (outs TILE:$dst),
                      (ins TILE:$src1, VR512:$src2, VR512:$src3),
                      "top4bsud\t{$src3, $src2, $dst|$dst, $src2, $src3}", []>,
                   EVEX, VVVV, XS, T8, EVEX_V512;
}

// TOP4MX Instructions - Mixed Precision Outer Products with Immediate
// All use opcode 0x8D with different prefixes, or 0x8F
// EVEX.512.NP.0F3A.W0 8D /r ib - TOP4MXBF8PS tmm1, zmm2, zmm3, imm8
// EVEX.512.66.0F3A.W0 8D /r ib - TOP4MXHF8PS tmm1, zmm2, zmm3, imm8
// EVEX.512.F2.0F3A.W0 8D /r ib - TOP4MXBHF8PS tmm1, zmm2, zmm3, imm8
// EVEX.512.F3.0F3A.W0 8D /r ib - TOP4MXHBF8PS tmm1, zmm2, zmm3, imm8
// EVEX.512.F2.0F3A.W0 8F /r ib - TOP4MXBSSPS tmm1, zmm2, zmm3, imm8
// Operand encoding from ACE spec page 47:
//   Operand 1: ModRM:reg (rw) = tmm1
//   Operand 2: ModRM:r/m (r) = zmm2
//   Operand 3: EVEX.vvvv (r) = zmm3
//   Operand 4: imm8
// MRMSrcReg encodes: src1→reg, src2→r/m, src3→vvvv (so swap to get spec order)
let Constraints = "$src1 = $dst", SchedRW = [WriteVecLogic] in {
  def TOP4MXBF8PSrrri : Ii8<0x8D, MRMSrcReg, (outs TILE:$dst),
                            (ins TILE:$src1, VR512:$src3, VR512:$src2, u8imm:$imm),
                            "top4mxbf8ps\t{$imm, $src3, $src2, $dst|$dst, $src2, $src3, $imm}", []>,
                       EVEX, VVVV, TA, EVEX_V512;

  def TOP4MXHF8PSrrri : Ii8<0x8D, MRMSrcReg, (outs TILE:$dst),
                            (ins TILE:$src1, VR512:$src3, VR512:$src2, u8imm:$imm),
                            "top4mxhf8ps\t{$imm, $src3, $src2, $dst|$dst, $src2, $src3, $imm}", []>,
                       EVEX, VVVV, PD, TA, EVEX_V512;

  def TOP4MXBHF8PSrrri : Ii8<0x8D, MRMSrcReg, (outs TILE:$dst),
                             (ins TILE:$src1, VR512:$src3, VR512:$src2, u8imm:$imm),
                             "top4mxbhf8ps\t{$imm, $src3, $src2, $dst|$dst, $src2, $src3, $imm}", []>,
                        EVEX, VVVV, XD, TA, EVEX_V512;

  def TOP4MXHBF8PSrrri : Ii8<0x8D, MRMSrcReg, (outs TILE:$dst),
                             (ins TILE:$src1, VR512:$src3, VR512:$src2, u8imm:$imm),
                             "top4mxhbf8ps\t{$imm, $src3, $src2, $dst|$dst, $src2, $src3, $imm}", []>,
                        EVEX, VVVV, XS, TA, EVEX_V512;

  def TOP4MXBSSPSrrri : Ii8<0x8F, MRMSrcReg, (outs TILE:$dst),
                            (ins TILE:$src1, VR512:$src3, VR512:$src2, u8imm:$imm),
                            "top4mxbssps\t{$imm, $src3, $src2, $dst|$dst, $src2, $src3, $imm}", []>,
                       EVEX, VVVV, XD, TA, EVEX_V512;
}

//===----------------------------------------------------------------------===//
// ACE Pseudo Instructions
// These are used by intrinsics and expanded via custom inserter
//===----------------------------------------------------------------------===//

// Pseudo instructions for user-facing intrinsics (immediate tile IDs)
// These use usesCustomInserter to expand to real instructions in X86ISelLowering
let usesCustomInserter = 1, hasNoSchedulingInfo = 1 in {
  // BSRINIT pseudo - no operands, just triggers BSR initialization
  def PBSRINIT : PseudoI<(outs), (ins),
                         [(int_x86_bsrinit)]>;

  // Note: BSRMOVF, BSRMOVH, BSRMOVL intrinsics are handled directly
  // by patterns on the real instructions (BSRMOVFrr, BSRMOVHrr_set, etc.)

  // TILEMOVCOL pseudo - immediate index variant
  // dst: tile ID (imm), src: ZMM value (VR512), idx: column index (imm)
  // Pattern matching removed - handled by custom lowering in X86ISelLowering.cpp
  def PTILEMOVCOL : PseudoI<(outs), (ins u8imm:$dst, VR512:$src, u8imm:$idx), []>;

  // TILEMOVCOL pseudo - register index variant
  def PTILEMOVCOL_REG : PseudoI<(outs), (ins u8imm:$dst, VR512:$src, GR32:$idx), []>;

  // TILEMOVROW pseudo - immediate index variant (ACE: write to tile)
  // dst: tile ID (imm), src: ZMM value (VR512), idx: row index (imm)
  // Pattern matching removed - handled by custom lowering in X86ISelLowering.cpp
  def PTILEMOVROW : PseudoI<(outs), (ins u8imm:$dst, VR512:$src, u8imm:$idx), []>;

  // TILEMOVROW pseudo - register index variant (ACE: write to tile)
  def PTILEMOVROW_REG : PseudoI<(outs), (ins u8imm:$dst, VR512:$src, GR32:$idx), []>;

  // TOP2BF16PS pseudo
  def PTOP2BF16PS : PseudoI<(outs), (ins u8imm:$dst, u8imm:$src1, u8imm:$src2),
                            [(int_x86_top2bf16ps timm:$dst, timm:$src1, timm:$src2)]>;

  // TOP4 pseudos - three tile operands (dst is also accumulator)
  def PTOP4BUUD : PseudoI<(outs), (ins u8imm:$dst, u8imm:$src1, u8imm:$src2),
                          [(int_x86_top4buud timm:$dst, timm:$src1, timm:$src2)]>;

  def PTOP4BUSD : PseudoI<(outs), (ins u8imm:$dst, u8imm:$src1, u8imm:$src2),
                          [(int_x86_top4busd timm:$dst, timm:$src1, timm:$src2)]>;

  def PTOP4BSSD : PseudoI<(outs), (ins u8imm:$dst, u8imm:$src1, u8imm:$src2),
                          [(int_x86_top4bssd timm:$dst, timm:$src1, timm:$src2)]>;

  def PTOP4BSUD : PseudoI<(outs), (ins u8imm:$dst, u8imm:$src1, u8imm:$src2),
                          [(int_x86_top4bsud timm:$dst, timm:$src1, timm:$src2)]>;

  // TOP4MX pseudos - four operands (dst, src1, src2, imm)
  def PTOP4MXHF8PS : PseudoI<(outs), (ins u8imm:$dst, u8imm:$src1, u8imm:$src2, u8imm:$imm),
                             [(int_x86_top4mxhf8ps timm:$dst, timm:$src1, timm:$src2, timm:$imm)]>;

  def PTOP4MXBHF8PS : PseudoI<(outs), (ins u8imm:$dst, u8imm:$src1, u8imm:$src2, u8imm:$imm),
                              [(int_x86_top4mxbhf8ps timm:$dst, timm:$src1, timm:$src2, timm:$imm)]>;

  def PTOP4MXHBF8PS : PseudoI<(outs), (ins u8imm:$dst, u8imm:$src1, u8imm:$src2, u8imm:$imm),
                              [(int_x86_top4mxhbf8ps timm:$dst, timm:$src1, timm:$src2, timm:$imm)]>;

  def PTOP4MXBF8PS : PseudoI<(outs), (ins u8imm:$dst, u8imm:$src1, u8imm:$src2, u8imm:$imm),
                             [(int_x86_top4mxbf8ps timm:$dst, timm:$src1, timm:$src2, timm:$imm)]>;

  def PTOP4MXBSSPS : PseudoI<(outs), (ins u8imm:$dst, u8imm:$src1, u8imm:$src2, u8imm:$imm),
                             [(int_x86_top4mxbssps timm:$dst, timm:$src1, timm:$src2, timm:$imm)]>;
}

// Pseudo instructions for internal intrinsics (register allocation)
// These are used for IR-level optimization and register allocation
// usesCustomInserter is needed to set ACEProgModel in EmitInstrWithCustomInserter
let isPseudo = true, hasNoSchedulingInfo = 1, usesCustomInserter = 1 in {
  // TILEMOVCOL internal - output is TILE, input is ZMM + index
  def PTILEMOVCOLV : PseudoI<(outs TILE:$dst),
                             (ins GR16:$row, GR16:$col, VR512:$src, GR32:$idx),
                             [(set TILE:$dst,
                               (int_x86_tilesetcol_internal GR16:$row, GR16:$col,
                                VR512:$src, GR32:$idx))]>;

  // TILEMOVROW internal - output is TILE, input is ZMM + index
  // ACE TILEMOVROW internal - write vector to tile row
  def PTILEMOVROWV : PseudoI<(outs TILE:$dst),
                             (ins GR16:$row, GR16:$col, VR512:$src, GR32:$idx),
                             [(set TILE:$dst,
                               (int_x86_tilesetrow_internal GR16:$row, GR16:$col,
                                VR512:$src, GR32:$idx))]>;

  // TOP2BF16PS internal - output TILE, inputs: dimensions, TILE acc, two ZMMs
  let Constraints = "$src1 = $dst" in
  def PTOP2BF16PSV : PseudoI<(outs TILE:$dst),
                             (ins GR16:$row, GR16:$col, GR16:$k,
                                  TILE:$src1, VR512:$src2, VR512:$src3),
                             [(set TILE:$dst,
                               (int_x86_top2bf16ps_internal GR16:$row, GR16:$col, GR16:$k,
                                TILE:$src1, VR512:$src2, VR512:$src3))]>;

  // TOP4BUUD internal
  let Constraints = "$src1 = $dst" in
  def PTOP4BUUDV : PseudoI<(outs TILE:$dst),
                           (ins GR16:$row, GR16:$col, GR16:$k,
                                TILE:$src1, VR512:$src2, VR512:$src3),
                           [(set TILE:$dst,
                             (int_x86_top4buud_internal GR16:$row, GR16:$col, GR16:$k,
                              TILE:$src1, VR512:$src2, VR512:$src3))]>;

  // TOP4BUSD internal
  let Constraints = "$src1 = $dst" in
  def PTOP4BUSDV : PseudoI<(outs TILE:$dst),
                           (ins GR16:$row, GR16:$col, GR16:$k,
                                TILE:$src1, VR512:$src2, VR512:$src3),
                           [(set TILE:$dst,
                             (int_x86_top4busd_internal GR16:$row, GR16:$col, GR16:$k,
                              TILE:$src1, VR512:$src2, VR512:$src3))]>;

  // TOP4BSSD internal
  let Constraints = "$src1 = $dst" in
  def PTOP4BSSDV : PseudoI<(outs TILE:$dst),
                           (ins GR16:$row, GR16:$col, GR16:$k,
                                TILE:$src1, VR512:$src2, VR512:$src3),
                           [(set TILE:$dst,
                             (int_x86_top4bssd_internal GR16:$row, GR16:$col, GR16:$k,
                              TILE:$src1, VR512:$src2, VR512:$src3))]>;

  // TOP4BSUD internal
  let Constraints = "$src1 = $dst" in
  def PTOP4BSUDV : PseudoI<(outs TILE:$dst),
                           (ins GR16:$row, GR16:$col, GR16:$k,
                                TILE:$src1, VR512:$src2, VR512:$src3),
                           [(set TILE:$dst,
                             (int_x86_top4bsud_internal GR16:$row, GR16:$col, GR16:$k,
                              TILE:$src1, VR512:$src2, VR512:$src3))]>;

  // TOP4MXHFBPS internal - imm is BSR index (u8imm matches timm in pattern)
  let Constraints = "$src1 = $dst" in
  def PTOP4MXHF8PSV : PseudoI<(outs TILE:$dst),
                              (ins GR16:$row, GR16:$col, GR16:$k, u8imm:$imm,
                                   TILE:$src1, VR512:$src2, VR512:$src3),
                              [(set TILE:$dst,
                                (int_x86_top4mxhf8ps_internal GR16:$row, GR16:$col, GR16:$k, timm:$imm,
                                 TILE:$src1, VR512:$src2, VR512:$src3))]>;

  // TOP4MXBHFBPS internal
  let Constraints = "$src1 = $dst" in
  def PTOP4MXBHF8PSV : PseudoI<(outs TILE:$dst),
                               (ins GR16:$row, GR16:$col, GR16:$k, u8imm:$imm,
                                    TILE:$src1, VR512:$src2, VR512:$src3),
                               [(set TILE:$dst,
                                 (int_x86_top4mxbhf8ps_internal GR16:$row, GR16:$col, GR16:$k, timm:$imm,
                                  TILE:$src1, VR512:$src2, VR512:$src3))]>;

  // TOP4MXHBFBPS internal
  let Constraints = "$src1 = $dst" in
  def PTOP4MXHBF8PSV : PseudoI<(outs TILE:$dst),
                               (ins GR16:$row, GR16:$col, GR16:$k, u8imm:$imm,
                                    TILE:$src1, VR512:$src2, VR512:$src3),
                               [(set TILE:$dst,
                                 (int_x86_top4mxhbf8ps_internal GR16:$row, GR16:$col, GR16:$k, timm:$imm,
                                  TILE:$src1, VR512:$src2, VR512:$src3))]>;

  // TOP4MXBF8PS internal
  let Constraints = "$src1 = $dst" in
  def PTOP4MXBF8PSV : PseudoI<(outs TILE:$dst),
                              (ins GR16:$row, GR16:$col, GR16:$k, u8imm:$imm,
                                   TILE:$src1, VR512:$src2, VR512:$src3),
                              [(set TILE:$dst,
                                (int_x86_top4mxbf8ps_internal GR16:$row, GR16:$col, GR16:$k, timm:$imm,
                                 TILE:$src1, VR512:$src2, VR512:$src3))]>;

  // TOP4MXBSSPS internal
  let Constraints = "$src1 = $dst" in
  def PTOP4MXBSSPSV : PseudoI<(outs TILE:$dst),
                              (ins GR16:$row, GR16:$col, GR16:$k, u8imm:$imm,
                                   TILE:$src1, VR512:$src2, VR512:$src3),
                              [(set TILE:$dst,
                                (int_x86_top4mxbssps_internal GR16:$row, GR16:$col, GR16:$k, timm:$imm,
                                 TILE:$src1, VR512:$src2, VR512:$src3))]>;
}

} // end HasACEV1, In64BitMode

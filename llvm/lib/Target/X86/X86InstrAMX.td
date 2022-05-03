//===---- X86InstrAMX.td - AMX Instruction Set Extension --*- tablegen -*--===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file describes the instructions that make up the Intel AMX instruction
// set.
//
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// AMX instructions

let Predicates = [HasAMXTILE, In64BitMode] in {
  let SchedRW = [WriteSystem] in {
    let hasSideEffects = 1,
        Defs = [TMM0,TMM1,TMM2,TMM3,TMM4,TMM5,TMM6,TMM7] in
    def LDTILECFG : I <0x49, MRM0m, (outs), (ins opaquemem:$src),
                       "ldtilecfg\t$src",
                       [(int_x86_ldtilecfg addr:$src)]>, VEX, T8PS;
    let hasSideEffects = 1 in
    def STTILECFG : I <0x49, MRM0m, (outs), (ins opaquemem:$src),
                       "sttilecfg\t$src",
                       [(int_x86_sttilecfg addr:$src)]>, VEX, T8PD;
    let mayLoad = 1 in
    def TILELOADD : I<0x4b, MRMSrcMemFSIB, (outs TILE:$dst),
                      (ins sibmem:$src),
                      "tileloadd\t{$src, $dst|$dst, $src}", []>,
                      VEX, T8XD;
    let mayLoad = 1 in
    def TILELOADDT1 : I<0x4b, MRMSrcMemFSIB, (outs TILE:$dst),
                        (ins sibmem:$src),
                        "tileloaddt1\t{$src, $dst|$dst, $src}", []>,
                        VEX, T8PD;
    let Defs = [TMM0,TMM1,TMM2,TMM3,TMM4,TMM5,TMM6,TMM7] in
    def TILERELEASE : I<0x49, MRM_C0, (outs), (ins),
                        "tilerelease", [(int_x86_tilerelease)]>, VEX, T8PS;
    let mayStore = 1 in
    def TILESTORED : I<0x4b, MRMDestMemFSIB, (outs),
                       (ins sibmem:$dst, TILE:$src),
                       "tilestored\t{$src, $dst|$dst, $src}", []>,
                       VEX, T8XS;
    def TILEZERO : I<0x49, MRMr0, (outs TILE:$dst), (ins),
                     "tilezero\t$dst", []>,
                     VEX, T8XD;

    // Pseduo instruction for RA.
    let mayLoad = 1 in
    def PLDTILECFGV : PseudoI<(outs), (ins opaquemem:$src),
                              [(int_x86_ldtilecfg_internal addr:$src)]>;
    let mayLoad = 1 in
    def PTILELOADDV : PseudoI<(outs TILE:$dst), (ins GR16:$src1,
                                                     GR16:$src2,
                                                     opaquemem:$src3), []>;
    let mayLoad = 1 in
    def PTILELOADDT1V : PseudoI<(outs TILE:$dst), (ins GR16:$src1,
                                                       GR16:$src2,
                                                       opaquemem:$src3), []>;
    let mayStore = 1 in
    def PTILESTOREDV : PseudoI<(outs), (ins GR16:$src1,
                                            GR16:$src2, opaquemem:$src3,
                                            TILE:$src4), []>;
    let isReMaterializable = 1, isAsCheapAsAMove = 1, canFoldAsLoad = 1 in
      def PTILEZEROV : PseudoI<(outs TILE:$dst), (ins GR16:$src1, GR16:$src2),
                                [(set TILE:$dst, (int_x86_tilezero_internal
                                  GR16:$src1, GR16:$src2))]>;

    let usesCustomInserter = 1 in {
      // Pseudo instructions, using immediates instead of tile registers.
      // To be translated to the actual instructions in X86ISelLowering.cpp
      let mayLoad = 1 in
      def PTILELOADD : PseudoI<(outs), (ins u8imm:$src1, sibmem:$src2), []>;
      let mayLoad = 1 in
      def PTILELOADDT1 : PseudoI<(outs), (ins u8imm:$src1,
                                          sibmem:$src2), []>;
      let mayStore = 1 in
      def PTILESTORED : PseudoI<(outs), (ins i8mem:$dst, u8imm:$src), []>;
      def PTILEZERO : PseudoI<(outs), (ins u8imm:$src),
                              [(int_x86_tilezero timm:$src)]>;
    }
  } // SchedRW
} // HasAMXTILE

let Predicates = [HasAMXINT8, In64BitMode] in {
  let SchedRW = [WriteSystem] in {
    let Constraints = "$src1 = $dst" in {
      def TDPBSSD : I<0x5e, MRMSrcReg4VOp3, (outs TILE:$dst),
                      (ins TILE:$src1, TILE:$src2, TILE:$src3),
                      "tdpbssd\t{$src3, $src2, $dst|$dst, $src2, $src3}", []>,
                      VEX_4V, T8XD;
      def TDPBSUD : I<0x5e, MRMSrcReg4VOp3, (outs TILE:$dst),
                      (ins TILE:$src1, TILE:$src2, TILE:$src3),
                      "tdpbsud\t{$src3, $src2, $dst|$dst, $src2, $src3}", []>,
                      VEX_4V, T8XS;
      def TDPBUSD : I<0x5e, MRMSrcReg4VOp3, (outs TILE:$dst),
                      (ins TILE:$src1, TILE:$src2, TILE:$src3),
                      "tdpbusd\t{$src3, $src2, $dst|$dst, $src2, $src3}", []>,
                      VEX_4V, T8PD;
      def TDPBUUD : I<0x5e, MRMSrcReg4VOp3, (outs TILE:$dst),
                      (ins TILE:$src1, TILE:$src2, TILE:$src3),
                      "tdpbuud\t{$src3, $src2, $dst|$dst, $src2, $src3}", []>,
                      VEX_4V, T8PS;
    }

    // Pseduo instruction for RA.
    let Constraints = "$src4 = $dst" in {
      def PTDPBSSDV : PseudoI<(outs TILE:$dst), (ins GR16:$src1,
                              GR16:$src2, GR16:$src3, TILE:$src4,
                              TILE:$src5, TILE:$src6),
                              [(set TILE: $dst,
                              (int_x86_tdpbssd_internal GR16:$src1, GR16:$src2,
                              GR16:$src3, TILE:$src4, TILE:$src5, TILE:$src6))]>;
      def PTDPBSUDV : PseudoI<(outs TILE: $dst), (ins GR16:$src1,
                              GR16:$src2, GR16:$src3, TILE:$src4,
                              TILE:$src5, TILE:$src6),
                              [(set TILE: $dst,
                              (int_x86_tdpbsud_internal GR16:$src1, GR16:$src2,
                               GR16:$src3, TILE:$src4, TILE:$src5, TILE:$src6))]>;
      def PTDPBUSDV : PseudoI<(outs TILE: $dst), (ins GR16:$src1,
                              GR16:$src2, GR16:$src3, TILE:$src4,
                              TILE:$src5, TILE:$src6),
                              [(set TILE: $dst,
                              (int_x86_tdpbusd_internal GR16:$src1, GR16:$src2,
                              GR16:$src3, TILE:$src4, TILE:$src5, TILE:$src6))]>;
      def PTDPBUUDV : PseudoI<(outs TILE: $dst), (ins GR16:$src1,
                              GR16:$src2, GR16:$src3, TILE:$src4,
                              TILE:$src5, TILE:$src6),
                              [(set TILE: $dst,
                              (int_x86_tdpbuud_internal GR16:$src1, GR16:$src2,
                              GR16:$src3, TILE:$src4, TILE:$src5, TILE:$src6))]>;
    }

    let usesCustomInserter = 1 in {
      // Pseudo instructions, using immediates instead of tile registers.
      // To be translated to the actual instructions in X86ISelLowering.cpp
      def PTDPBSSD : PseudoI<(outs), (ins u8imm:$src1,
                             u8imm:$src2, u8imm:$src3),
                             [(int_x86_tdpbssd timm:$src1,
                               timm:$src2, timm:$src3)]>;
      def PTDPBSUD : PseudoI<(outs), (ins u8imm:$src1,
                             u8imm:$src2, u8imm:$src3),
                             [(int_x86_tdpbsud timm:$src1,
                               timm:$src2, timm:$src3)]>;
      def PTDPBUSD : PseudoI<(outs), (ins u8imm:$src1,
                             u8imm:$src2, u8imm:$src3),
                             [(int_x86_tdpbusd timm:$src1,
                               timm:$src2, timm:$src3)]>;
      def PTDPBUUD : PseudoI<(outs), (ins u8imm:$src1,
                             u8imm:$src2, u8imm:$src3),
                             [(int_x86_tdpbuud timm:$src1,
                               timm:$src2, timm:$src3)]>;
    }
  }
} // HasAMXTILE

let Predicates = [HasAMXBF16, In64BitMode] in {
  let SchedRW = [WriteSystem] in {
    let Constraints = "$src1 = $dst" in
    def TDPBF16PS : I<0x5c, MRMSrcReg4VOp3, (outs TILE:$dst),
                      (ins TILE:$src1, TILE:$src2, TILE:$src3),
                      "tdpbf16ps\t{$src3, $src2, $dst|$dst, $src2, $src3}",
                      []>, VEX_4V, T8XS;

    // Pseduo instruction for RA.
    let Constraints = "$src4 = $dst" in
      def PTDPBF16PSV : PseudoI<(outs TILE: $dst), (ins GR16:$src1,
                                 GR16:$src2, GR16:$src3, TILE:$src4,
                                 TILE:$src5, TILE:$src6),
                                 [(set TILE: $dst,
                                  (int_x86_tdpbf16ps_internal GR16:$src1,
                                   GR16:$src2, GR16:$src3, TILE:$src4,
                                   TILE:$src5, TILE:$src6))]>;

    let usesCustomInserter = 1 in {
      // Pseudo instructions, using immediates instead of tile registers.
      // To be translated to the actual instructions in X86ISelLowering.cpp
      def PTDPBF16PS : PseudoI<(outs), (ins u8imm:$src1,
                               u8imm:$src2, u8imm:$src3),
                               [(int_x86_tdpbf16ps timm:$src1,
                                 timm:$src2, timm:$src3)]>;
    }
  }
} // HasAMXTILE, HasAMXBF16

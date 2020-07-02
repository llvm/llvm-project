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
    let Defs = [TMM0,TMM1,TMM2,TMM3,TMM4,TMM5,TMM6,TMM7] in
    def LDTILECFG : I <0x49, MRM0m, (outs), (ins opaquemem:$src),
                       "ldtilecfg\t$src", []>, VEX, T8PS;
    def STTILECFG : I <0x49, MRM0m, (outs), (ins opaquemem:$src),
                       "sttilecfg\t$src", []>, VEX, T8PD;
    def TILELOADD : I<0x4b, MRMSrcMemFSIB, (outs TILE:$dst),
                      (ins sibmem:$src),
                      "tileloadd\t{$src, $dst|$dst, $src}", []>,
                      VEX, T8XD;
    def TILELOADDT1 : I<0x4b, MRMSrcMemFSIB, (outs TILE:$dst),
                        (ins sibmem:$src),
                        "tileloaddt1\t{$src, $dst|$dst, $src}", []>,
                        VEX, T8PD;
    let Defs = [TMM0,TMM1,TMM2,TMM3,TMM4,TMM5,TMM6,TMM7] in
    def TILERELEASE : I<0x49, MRM_C0, (outs), (ins),
                        "tilerelease", []>, VEX, T8PS;
    def TILESTORED : I<0x4b, MRMDestMemFSIB, (outs),
                       (ins sibmem:$dst, TILE:$src),
                       "tilestored\t{$src, $dst|$dst, $src}", []>,
                       VEX, T8XS;
    def TILEZERO : I<0x49, MRMr0, (outs TILE:$dst), (ins),
                     "tilezero\t$dst", []>,
                     VEX, T8XD;
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
  }
} // HasAMXTILE

let Predicates = [HasAMXBF16, In64BitMode] in {
  let SchedRW = [WriteSystem] in {
    let Constraints = "$src1 = $dst" in
    def TDPBF16PS : I<0x5c, MRMSrcReg4VOp3, (outs TILE:$dst),
                      (ins TILE:$src1, TILE:$src2, TILE:$src3),
                      "tdpbf16ps\t{$src3, $src2, $dst|$dst, $src2, $src3}",
                      []>, VEX_4V, T8XS;
  }
} // HasAMXTILE, HasAMXBF16

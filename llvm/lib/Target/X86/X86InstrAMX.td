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

multiclass AMX_TILE_COMMON<string Suffix, Predicate HasEGPR> {
let Predicates = [HasAMXTILE, HasEGPR, In64BitMode] in {
  let hasSideEffects = 1,
      Defs = [TMM0,TMM1,TMM2,TMM3,TMM4,TMM5,TMM6,TMM7] in
  def LDTILECFG#Suffix : I<0x49, MRM0m, (outs), (ins opaquemem:$src),
                           "ldtilecfg\t$src",
                           [(int_x86_ldtilecfg addr:$src)]>,
                         T8, PS;
  let hasSideEffects = 1 in
  def STTILECFG#Suffix : I<0x49, MRM0m, (outs), (ins opaquemem:$src),
                           "sttilecfg\t$src",
                           [(int_x86_sttilecfg addr:$src)]>,
                         T8, PD;
  let mayLoad = 1 in
  def TILELOADD#Suffix : I<0x4b, MRMSrcMemFSIB, (outs TILE:$dst),
                           (ins sibmem:$src),
                           "tileloadd\t{$src, $dst|$dst, $src}", []>,
                         T8, XD;
  let mayLoad = 1 in
  def TILELOADDT1#Suffix : I<0x4b, MRMSrcMemFSIB, (outs TILE:$dst),
                             (ins sibmem:$src),
                             "tileloaddt1\t{$src, $dst|$dst, $src}", []>,
                           T8, PD;
  let mayStore = 1 in
  def TILESTORED#Suffix : I<0x4b, MRMDestMemFSIB, (outs),
                            (ins sibmem:$dst, TILE:$src),
                            "tilestored\t{$src, $dst|$dst, $src}", []>,
                          T8, XS;
}
}

let SchedRW = [WriteSystem] in {
  defm "" : AMX_TILE_COMMON<"", NoEGPR>, VEX;
  defm "" : AMX_TILE_COMMON<"_EVEX", HasEGPR>, EVEX, NoCD8;

  let Predicates = [HasAMXTILE, In64BitMode] in {
    let Defs = [TMM0,TMM1,TMM2,TMM3,TMM4,TMM5,TMM6,TMM7] in
    def TILERELEASE : I<0x49, MRM_C0, (outs), (ins),
                        "tilerelease", [(int_x86_tilerelease)]>, VEX, T8, PS;
    def TILEZERO : I<0x49, MRMr0, (outs TILE:$dst), (ins),
                     "tilezero\t$dst", []>,
                     VEX, T8, XD;

    // Pseduo instruction for RA.
    let isPseudo = true, mayLoad = 1, hasSideEffects = 1,
        Defs = [TMM0,TMM1,TMM2,TMM3,TMM4,TMM5,TMM6,TMM7] in
    def PLDTILECFGV : PseudoI<(outs), (ins opaquemem:$src), []>;
    let isPseudo = true, mayLoad = 1 in
    def PTILELOADDV : PseudoI<(outs TILE:$dst), (ins GR16:$src1,
                                                     GR16:$src2,
                                                     opaquemem:$src3), []>;
    let isPseudo = true, mayLoad = 1 in
    def PTILELOADDT1V : PseudoI<(outs TILE:$dst), (ins GR16:$src1,
                                                       GR16:$src2,
                                                       opaquemem:$src3), []>;
    let isPseudo = true, mayStore = 1 in
    def PTILESTOREDV : PseudoI<(outs), (ins GR16:$src1,
                                            GR16:$src2, opaquemem:$src3,
                                            TILE:$src4), []>;
    let isPseudo = true, isReMaterializable = 1, isAsCheapAsAMove = 1,
        canFoldAsLoad = 1, usesCustomInserter = 1 in
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
  } // Predicates
} // SchedRW

let Predicates = [HasAMXINT8, In64BitMode] in {
  let SchedRW = [WriteSystem] in {
    let Constraints = "$src1 = $dst" in {
      def TDPBSSD : I<0x5e, MRMSrcReg4VOp3, (outs TILE:$dst),
                      (ins TILE:$src1, TILE:$src2, TILE:$src3),
                      "tdpbssd\t{$src3, $src2, $dst|$dst, $src2, $src3}", []>,
                      VEX, VVVV, T8, XD;
      def TDPBSUD : I<0x5e, MRMSrcReg4VOp3, (outs TILE:$dst),
                      (ins TILE:$src1, TILE:$src2, TILE:$src3),
                      "tdpbsud\t{$src3, $src2, $dst|$dst, $src2, $src3}", []>,
                      VEX, VVVV, T8, XS;
      def TDPBUSD : I<0x5e, MRMSrcReg4VOp3, (outs TILE:$dst),
                      (ins TILE:$src1, TILE:$src2, TILE:$src3),
                      "tdpbusd\t{$src3, $src2, $dst|$dst, $src2, $src3}", []>,
                      VEX, VVVV, T8, PD;
      def TDPBUUD : I<0x5e, MRMSrcReg4VOp3, (outs TILE:$dst),
                      (ins TILE:$src1, TILE:$src2, TILE:$src3),
                      "tdpbuud\t{$src3, $src2, $dst|$dst, $src2, $src3}", []>,
                      VEX, VVVV, T8;
    }

    // Pseduo instruction for RA.
    let isPseudo = true, Constraints = "$src4 = $dst" in {
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
                      []>, VEX, VVVV, T8, XS;

    // Pseduo instruction for RA.
    let isPseudo = true, Constraints = "$src4 = $dst" in
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

//AMX-FP16
let Predicates = [HasAMXFP16, In64BitMode] in {
  let SchedRW = [WriteSystem] in {
    let Constraints = "$src1 = $dst" in {
      def TDPFP16PS : I<0x5c, MRMSrcReg4VOp3, (outs TILE:$dst),
                        (ins TILE:$src1, TILE:$src2, TILE:$src3),
                        "tdpfp16ps\t{$src3, $src2, $src1|$src1, $src2, $src3}",
                        []>, VEX, VVVV, T8, XD;
    }

    // Pseduo instruction for RA.
    let isPseudo = true, Constraints = "$src4 = $dst" in {
      def PTDPFP16PSV : PseudoI<(outs TILE: $dst), (ins GR16:$src1,
                                 GR16:$src2, GR16:$src3, TILE:$src4,
                                 TILE:$src5, TILE:$src6),
                                 [(set TILE: $dst,
                                  (int_x86_tdpfp16ps_internal GR16:$src1,
                                   GR16:$src2, GR16:$src3, TILE:$src4,
                                   TILE:$src5, TILE:$src6))]>;
    }

    let  usesCustomInserter = 1 in {
      def PTDPFP16PS : PseudoI<(outs), (ins u8imm:$src1,
                               u8imm:$src2, u8imm:$src3),
                               [(int_x86_tdpfp16ps timm:$src1,
                                 timm:$src2, timm:$src3)]>;
    }
  }
} // HasAMXTILE, HasAMXFP16

let Predicates = [HasAMXCOMPLEX, In64BitMode] in {
  let SchedRW = [WriteSystem] in {
    let Constraints = "$src1 = $dst" in {
      def TCMMIMFP16PS   : I<0x6c, MRMSrcReg4VOp3, (outs TILE:$dst),
                            (ins TILE:$src1, TILE:$src2, TILE:$src3),
                            "tcmmimfp16ps\t{$src3, $src2, $src1|$src1, $src2, $src3}",
                            []>, T8, PD, VEX, VVVV;
      def TCMMRLFP16PS : I<0x6c, MRMSrcReg4VOp3, (outs TILE:$dst),
                            (ins TILE:$src1, TILE:$src2, TILE:$src3),
                            "tcmmrlfp16ps\t{$src3, $src2, $src1|$src1, $src2, $src3}",
                            []>, VEX, VVVV, WIG, T8;

    } // Constraints = "$src1 = $dst"

    let Constraints = "$src4 = $dst" in {
      def PTCMMIMFP16PSV : PseudoI<(outs TILE:$dst), (ins GR16:$src1,
                                  GR16:$src2, GR16:$src3, TILE:$src4,
                                  TILE:$src5, TILE:$src6),
                                  [(set TILE: $dst,
                                  (int_x86_tcmmimfp16ps_internal GR16:$src1, GR16:$src2,
                                   GR16:$src3, TILE:$src4, TILE:$src5, TILE:$src6))]>;
      def PTCMMRLFP16PSV : PseudoI<(outs TILE:$dst), (ins GR16:$src1,
                                  GR16:$src2, GR16:$src3, TILE:$src4,
                                  TILE:$src5, TILE:$src6),
                                  [(set TILE: $dst,
                                  (int_x86_tcmmrlfp16ps_internal GR16:$src1, GR16:$src2,
                                   GR16:$src3, TILE:$src4, TILE:$src5, TILE:$src6))]>;
    }

    let usesCustomInserter = 1 in {
      def PTCMMIMFP16PS : PseudoI<(outs), (ins u8imm:$src1,
                                u8imm:$src2, u8imm:$src3),
                                [(int_x86_tcmmimfp16ps timm:$src1,
                                  timm:$src2, timm:$src3)]>;
      def PTCMMRLFP16PS : PseudoI<(outs), (ins u8imm:$src1,
                                u8imm:$src2, u8imm:$src3),
                                [(int_x86_tcmmrlfp16ps timm:$src1,
                                  timm:$src2, timm:$src3)]>;
    }
  } // SchedRW = [WriteSystem]
}

// AMX-FP8
let Predicates = [HasAMXFP8, In64BitMode] in {
  let SchedRW = [WriteSystem] in {
    let Constraints = "$src1 = $dst" in {
      class AMX_FP8_BASE<bits<8> Opcode, string Opstr> :
        I<Opcode, MRMSrcReg4VOp3, (outs TILE:$dst),
          (ins TILE:$src1, TILE:$src2, TILE:$src3),
          !strconcat(Opstr, "\t{$src3, $src2, $dst|$dst, $src2, $src3}"),
          []>, VEX, VVVV;
    }

    def TDPBF8PS : AMX_FP8_BASE<0xfd, "tdpbf8ps">, T_MAP5, PS;
    def TDPBHF8PS : AMX_FP8_BASE<0xfd, "tdpbhf8ps">, T_MAP5, XD;
    def TDPHBF8PS : AMX_FP8_BASE<0xfd, "tdphbf8ps">, T_MAP5, XS;
    def TDPHF8PS : AMX_FP8_BASE<0xfd, "tdphf8ps">, T_MAP5, PD;

    let usesCustomInserter = 1 in {
      // Pseudo instructions, using immediates instead of tile registers.
      // To be translated to the actual instructions in X86ISelLowering.cpp
      def PTDPBF8PS : PseudoI<(outs),
                              (ins u8imm:$src1, u8imm:$src2, u8imm:$src3),
                              [(int_x86_tdpbf8ps timm:$src1, timm:$src2,
                                timm:$src3)]>;
      def PTDPBHF8PS : PseudoI<(outs),
                               (ins u8imm:$src1, u8imm:$src2, u8imm:$src3),
                               [(int_x86_tdpbhf8ps timm:$src1, timm:$src2,
                                 timm:$src3)]>;
      def PTDPHBF8PS : PseudoI<(outs),
                               (ins u8imm:$src1, u8imm:$src2, u8imm:$src3),
                               [(int_x86_tdphbf8ps timm:$src1, timm:$src2,
                                 timm:$src3)]>;
      def PTDPHF8PS : PseudoI<(outs),
                              (ins u8imm:$src1, u8imm:$src2, u8imm:$src3),
                              [(int_x86_tdphf8ps timm:$src1, timm:$src2,
                                timm:$src3)]>;
    }

    let Constraints = "$src4 = $dst" in {
      def PTDPBF8PSV : PseudoI<(outs TILE:$dst),
                               (ins GR16:$src1, GR16:$src2, GR16:$src3,
                                    TILE:$src4, TILE:$src5, TILE:$src6),
                               [(set TILE:$dst,
                                (int_x86_tdpbf8ps_internal GR16:$src1,
                                 GR16:$src2, GR16:$src3, TILE:$src4,
                                 TILE:$src5, TILE:$src6))]>;
      def PTDPBHF8PSV : PseudoI<(outs TILE:$dst),
                               (ins GR16:$src1, GR16:$src2, GR16:$src3,
                                    TILE:$src4, TILE:$src5, TILE:$src6),
                               [(set TILE:$dst,
                                (int_x86_tdpbhf8ps_internal GR16:$src1,
                                 GR16:$src2, GR16:$src3, TILE:$src4,
                                 TILE:$src5, TILE:$src6))]>;
      def PTDPHBF8PSV : PseudoI<(outs TILE:$dst),
                               (ins GR16:$src1, GR16:$src2, GR16:$src3,
                                    TILE:$src4, TILE:$src5, TILE:$src6),
                               [(set TILE:$dst,
                                (int_x86_tdphbf8ps_internal GR16:$src1,
                                 GR16:$src2, GR16:$src3, TILE:$src4,
                                 TILE:$src5, TILE:$src6))]>;
      def PTDPHF8PSV : PseudoI<(outs TILE:$dst),
                               (ins GR16:$src1, GR16:$src2, GR16:$src3,
                                    TILE:$src4, TILE:$src5, TILE:$src6),
                               [(set TILE:$dst,
                                (int_x86_tdphf8ps_internal GR16:$src1,
                                 GR16:$src2, GR16:$src3, TILE:$src4,
                                 TILE:$src5, TILE:$src6))]>;
    }
  }
}

let Predicates = [HasAMXTILE, In64BitMode], isPseudo = true, SchedRW = [WriteSystem] in {
  let mayStore = 1 in
  def PTILEPAIRSTORE : PseudoI<(outs), (ins opaquemem:$src1, TILEPair:$src2), []>;
  let mayLoad = 1 in
  def PTILEPAIRLOAD : PseudoI<(outs TILEPair:$dst), (ins opaquemem:$src), []>;
}

multiclass T2RPNTLVW_Base<bits<8> op1, bits<8> op2, string rs, string suffix> {
  def Z0#rs#suffix    : I<op1, MRMSrcMemFSIB, (outs TILEPair:$dst), (ins sibmem:$src),
                          "t2rpntlvwz0" #!tolower(rs)# "\t{$src, $dst|$dst, $src}", []>, PS;
  def Z0#rs#T1#suffix : I<op2, MRMSrcMemFSIB, (outs TILEPair:$dst), (ins sibmem:$src),
                          "t2rpntlvwz0" #!tolower(rs)# "t1\t{$src, $dst|$dst, $src}", []>, PS;
  def Z1#rs#suffix    : I<op1, MRMSrcMemFSIB, (outs TILEPair:$dst), (ins sibmem:$src),
                          "t2rpntlvwz1" #!tolower(rs)# "\t{$src, $dst|$dst, $src}", []>, PD;
  def Z1#rs#T1#suffix : I<op2, MRMSrcMemFSIB, (outs TILEPair:$dst), (ins sibmem:$src),
                          "t2rpntlvwz1" #!tolower(rs)# "t1\t{$src, $dst|$dst, $src}", []>, PD;
}

let Predicates = [HasAMXTRANSPOSE, In64BitMode], SchedRW = [WriteSystem] in
  defm T2RPNTLVW : T2RPNTLVW_Base<0x6e, 0x6f, "", "">, T8, VEX;

let Predicates = [HasAMXTRANSPOSE, HasEGPR, In64BitMode], SchedRW = [WriteSystem] in
  defm T2RPNTLVW : T2RPNTLVW_Base<0x6e, 0x6f, "", "_EVEX">, T8, EVEX, NoCD8;

let Predicates = [HasAMXMOVRS, HasAMXTRANSPOSE, In64BitMode], SchedRW = [WriteSystem] in
  defm T2RPNTLVW : T2RPNTLVW_Base<0xf8, 0xf9, "RS", "">, T_MAP5, VEX;

let Predicates = [HasAMXMOVRS, HasAMXTRANSPOSE, HasEGPR, In64BitMode], SchedRW = [WriteSystem] in
  defm T2RPNTLVW : T2RPNTLVW_Base<0xf8, 0xf9, "RS", "_EVEX">, T_MAP5, EVEX, NoCD8;

let Predicates = [HasAMXTRANSPOSE, In64BitMode] in {
  let SchedRW = [WriteSystem] in {
    def TTRANSPOSED : I<0x5f, MRMSrcReg, (outs TILE:$dst), (ins TILE:$src),
                        "ttransposed\t{$src, $dst|$dst, $src}", []>, VEX, T8, XS;
    let isPseudo = true in {
      def PT2RPNTLVWZ0V : PseudoI<(outs TILEPair:$dst),
                                  (ins GR16:$src1, GR16:$src2, GR16:$src3, opaquemem:$src4),
                                  []>;
      def PT2RPNTLVWZ0T1V : PseudoI<(outs TILEPair:$dst),
                                  (ins GR16:$src1, GR16:$src2, GR16:$src3, opaquemem:$src4),
                                  []>;
      def PT2RPNTLVWZ1V : PseudoI<(outs TILEPair:$dst),
                                  (ins GR16:$src1, GR16:$src2, GR16:$src3, opaquemem:$src4),
                                  []>;
      def PT2RPNTLVWZ1T1V : PseudoI<(outs TILEPair:$dst),
                                  (ins GR16:$src1, GR16:$src2, GR16:$src3, opaquemem:$src4),
                                  []>;
    }

    def PTTRANSPOSEDV : PseudoI<(outs TILE:$dst),
                                (ins GR16:$src1, GR16:$src2, TILE:$src),
                                [(set TILE: $dst,
                                 (int_x86_ttransposed_internal GR16:$src1, GR16:$src2,
                                  TILE:$src))]>;

    let usesCustomInserter = 1 in {
      def PT2RPNTLVWZ0 : PseudoI<(outs), (ins u8imm:$dst,
                                 sibmem:$src1), []>;
      def PT2RPNTLVWZ0T1 : PseudoI<(outs), (ins u8imm:$dst,
                                   sibmem:$src1), []>;
      def PT2RPNTLVWZ1 : PseudoI<(outs), (ins u8imm:$dst,
                                 sibmem:$src1), []>;
      def PT2RPNTLVWZ1T1 : PseudoI<(outs), (ins u8imm:$dst,
                                   sibmem:$src1), []>;
      def PTTRANSPOSED : PseudoI<(outs), (ins u8imm:$dst, u8imm:$src),
                                 [(int_x86_ttransposed timm:$dst, timm:$src)]>;
    }
  }
} // HasAMXTILE, HasAMXTRANSPOSE

let Predicates = [HasAMXBF16, HasAMXTRANSPOSE, In64BitMode], SchedRW = [WriteSystem] in {
  let Constraints = "$src1 = $dst" in
    def TTDPBF16PS : I<0x6c, MRMSrcReg4VOp3, (outs TILE:$dst),
                       (ins TILE:$src1, TILE:$src2, TILE:$src3),
                       "ttdpbf16ps\t{$src3, $src2, $dst|$dst, $src2, $src3}",
                       []>, VEX, VVVV, T8,XS;
  let Constraints = "$src4 = $dst" in
    def PTTDPBF16PSV : PseudoI<(outs TILE:$dst), (ins GR16:$src1,
                                GR16:$src2, GR16:$src3, TILE:$src4,
                                TILE:$src5, TILE:$src6),
                                [(set TILE: $dst,
                                  (int_x86_ttdpbf16ps_internal GR16:$src1, GR16:$src2,
                                   GR16:$src3, TILE:$src4, TILE:$src5, TILE:$src6))]>;
  let usesCustomInserter = 1 in
    def PTTDPBF16PS : PseudoI<(outs), (ins u8imm:$src1, u8imm:$src2, u8imm:$src3),
                              [(int_x86_ttdpbf16ps timm:$src1, timm:$src2, timm:$src3)]>;
}

let Predicates = [HasAMXFP16, HasAMXTRANSPOSE, In64BitMode], SchedRW = [WriteSystem] in {
  let Constraints = "$src1 = $dst" in
    def TTDPFP16PS : I<0x6c, MRMSrcReg4VOp3, (outs TILE:$dst),
                       (ins TILE:$src1, TILE:$src2, TILE:$src3),
                       "ttdpfp16ps\t{$src3, $src2, $dst|$dst, $src2, $src3}",
                       []>, VEX, VVVV, T8,XD;
  let Constraints = "$src4 = $dst" in
    def PTTDPFP16PSV : PseudoI<(outs TILE:$dst), (ins GR16:$src1,
                                GR16:$src2, GR16:$src3, TILE:$src4,
                                TILE:$src5, TILE:$src6),
                                [(set TILE: $dst,
                                  (int_x86_ttdpfp16ps_internal GR16:$src1, GR16:$src2,
                                   GR16:$src3, TILE:$src4, TILE:$src5, TILE:$src6))]>;
  let usesCustomInserter = 1 in
    def PTTDPFP16PS : PseudoI<(outs), (ins u8imm:$src1, u8imm:$src2, u8imm:$src3),
                              [(int_x86_ttdpfp16ps timm:$src1, timm:$src2, timm:$src3)]>;
}

let Predicates = [HasAMXCOMPLEX, HasAMXTRANSPOSE, In64BitMode], SchedRW = [WriteSystem] in {
  let Constraints = "$src1 = $dst" in {
    def TTCMMIMFP16PS : I<0x6b, MRMSrcReg4VOp3, (outs TILE:$dst),
                          (ins TILE:$src1, TILE:$src2, TILE:$src3),
                          "ttcmmimfp16ps\t{$src3, $src2, $src1|$src1, $src2, $src3}",
                          []>, VEX, VVVV, T8,XD;
    def TTCMMRLFP16PS: I<0x6b, MRMSrcReg4VOp3, (outs TILE:$dst),
                         (ins TILE:$src1, TILE:$src2, TILE:$src3),
                         "ttcmmrlfp16ps\t{$src3, $src2, $src1|$src1, $src2, $src3}",
                         []>, VEX, VVVV, T8,XS;
    def TCONJTCMMIMFP16PS : I<0x6b, MRMSrcReg4VOp3, (outs TILE:$dst),
                          (ins TILE:$src1, TILE:$src2, TILE:$src3),
                          "tconjtcmmimfp16ps\t{$src3, $src2, $src1|$src1, $src2, $src3}",
                          []>, VEX, VVVV, WIG, T8,PS;
  }
  def TCONJTFP16 : I<0x6b, MRMSrcReg, (outs TILE:$dst), (ins TILE:$src),
                     "tconjtfp16\t{$src, $dst|$dst, $src}", []>, VEX, T8,PD;

  let Constraints = "$src4 = $dst" in {
    def PTTCMMIMFP16PSV : PseudoI<(outs TILE:$dst), (ins GR16:$src1,
                                  GR16:$src2, GR16:$src3, TILE:$src4,
                                  TILE:$src5, TILE:$src6),
                                  [(set TILE: $dst,
                                    (int_x86_ttcmmimfp16ps_internal GR16:$src1, GR16:$src2,
                                     GR16:$src3, TILE:$src4, TILE:$src5, TILE:$src6))]>;
    def PTTCMMRLFP16PSV : PseudoI<(outs TILE:$dst), (ins GR16:$src1,
                                  GR16:$src2, GR16:$src3, TILE:$src4,
                                  TILE:$src5, TILE:$src6),
                                  [(set TILE: $dst,
                                    (int_x86_ttcmmrlfp16ps_internal GR16:$src1, GR16:$src2,
                                     GR16:$src3, TILE:$src4, TILE:$src5, TILE:$src6))]>;
    def PTCONJTCMMIMFP16PSV : PseudoI<(outs TILE:$dst), (ins GR16:$src1,
                                      GR16:$src2, GR16:$src3, TILE:$src4,
                                      TILE:$src5, TILE:$src6),
                                      [(set TILE: $dst,
                                        (int_x86_tconjtcmmimfp16ps_internal GR16:$src1, GR16:$src2,
                                         GR16:$src3, TILE:$src4, TILE:$src5, TILE:$src6))]>;
  }
  def PTCONJTFP16V : PseudoI<(outs TILE:$dst), (ins GR16:$src1, GR16:$src2, TILE:$src3),
                             [(set TILE: $dst, (int_x86_tconjtfp16_internal GR16:$src1, GR16:$src2, TILE:$src3))]>;

  let usesCustomInserter = 1 in {
    def PTTCMMIMFP16PS : PseudoI<(outs), (ins u8imm:$src1, u8imm:$src2, u8imm:$src3),
                                 [(int_x86_ttcmmimfp16ps timm:$src1, timm:$src2, timm:$src3)]>;
    def PTTCMMRLFP16PS : PseudoI<(outs), (ins u8imm:$src1, u8imm:$src2, u8imm:$src3),
                                 [(int_x86_ttcmmrlfp16ps timm:$src1, timm:$src2, timm:$src3)]>;
    def PTCONJTCMMIMFP16PS : PseudoI<(outs), (ins u8imm:$src1, u8imm:$src2, u8imm:$src3),
                                     [(int_x86_tconjtcmmimfp16ps timm:$src1, timm:$src2, timm:$src3)]>;
    def PTCONJTFP16 : PseudoI<(outs), (ins u8imm:$dst, u8imm:$src),
                              [(int_x86_tconjtfp16 timm:$dst, timm:$src)]>;
  }
}

let Predicates = [HasAMXMOVRS, HasAMXTRANSPOSE, In64BitMode], SchedRW = [WriteSystem] in {
  let isPseudo = true in {
    def PT2RPNTLVWZ0RSV   : PseudoI<(outs TILEPair:$dst),
                              (ins GR16:$src1, GR16:$src2, GR16:$src3, opaquemem:$src4),
                              []>;
    def PT2RPNTLVWZ0RST1V : PseudoI<(outs TILEPair:$dst),
                              (ins GR16:$src1, GR16:$src2, GR16:$src3, opaquemem:$src4),
                              []>;
    def PT2RPNTLVWZ1RSV   : PseudoI<(outs TILEPair:$dst),
                              (ins GR16:$src1, GR16:$src2, GR16:$src3, opaquemem:$src4),
                              []>;
    def PT2RPNTLVWZ1RST1V : PseudoI<(outs TILEPair:$dst),
                              (ins GR16:$src1, GR16:$src2, GR16:$src3, opaquemem:$src4),
                              []>;
  }
  let  usesCustomInserter = 1 in {
    def PT2RPNTLVWZ0RS   : PseudoI<(outs), (ins u8imm:$dst, sibmem:$src1), []>;
    def PT2RPNTLVWZ0RST1 : PseudoI<(outs), (ins u8imm:$dst, sibmem:$src1), []>;
    def PT2RPNTLVWZ1RS   : PseudoI<(outs), (ins u8imm:$dst, sibmem:$src1), []>;
    def PT2RPNTLVWZ1RST1 : PseudoI<(outs), (ins u8imm:$dst, sibmem:$src1), []>;
  }
} // HasAMXMOVRS, HasAMXTRANSPOSE

multiclass TILELOADDRS_Base<string suffix> {
  def suffix    : I<0x4a, MRMSrcMemFSIB, (outs TILE:$dst), (ins sibmem:$src1),
                    "tileloaddrs\t{$src1, $dst|$dst, $src1}", []>, T8, XD;
  def T1#suffix : I<0x4a, MRMSrcMemFSIB, (outs TILE:$dst), (ins sibmem:$src1),
                    "tileloaddrst1\t{$src1, $dst|$dst, $src1}", []>, T8, PD;
}

let Predicates = [HasAMXMOVRS, In64BitMode], SchedRW = [WriteSystem] in
  defm TILELOADDRS : TILELOADDRS_Base<"">, VEX;

let Predicates = [HasAMXMOVRS, HasEGPR, In64BitMode], SchedRW = [WriteSystem] in
  defm TILELOADDRS : TILELOADDRS_Base<"_EVEX">, EVEX, NoCD8;

let Predicates = [HasAMXMOVRS, In64BitMode], SchedRW = [WriteSystem] in {
  let isPseudo = true, mayLoad = 1 in {
    def PTILELOADDRSV : PseudoI<(outs TILE:$dst), (ins GR16:$src1,
                                                  GR16:$src2,
                                                  opaquemem:$src3), []>;
    def PTILELOADDRST1V : PseudoI<(outs TILE:$dst), (ins GR16:$src1,
                                                    GR16:$src2,
                                                    opaquemem:$src3), []>;
  }

  let usesCustomInserter = 1, mayLoad = 1 in {
    def PTILELOADDRS : PseudoI<(outs), (ins u8imm:$src1, sibmem:$src2), []>;
    def PTILELOADDRST1 : PseudoI<(outs), (ins u8imm:$src1, sibmem:$src2), []>;
  }
} // HasAMXMOVRS, In64BitMode

multiclass m_tcvtrowd2ps {
  let Predicates = [HasAMXAVX512, HasAVX10_2_512, In64BitMode] in {
    let SchedRW = [WriteSystem] in {
      def rri : Ii8<0x7, MRMSrcReg, (outs VR512:$dst),
                    (ins TILE:$src1, i32u8imm:$src2),
                    "tcvtrowd2ps\t{$src2, $src1, $dst|$dst, $src1, $src2}",
                    []>, TA,XS, EVEX, EVEX_V512;
      def rre : I<0x4A, MRMSrcReg4VOp3, (outs VR512:$dst),
                  (ins TILE:$src1, GR32:$src2),
                  "tcvtrowd2ps\t{$src2, $src1, $dst|$dst, $src1, $src2}",
                  []>, T8,XS, EVEX, VVVV, EVEX_V512;
    }
  } // HasAMXAVX512, HasAVX10_2_512, In64BitMode
}

defm TCVTROWD2PS : m_tcvtrowd2ps;

let Predicates = [HasAMXAVX512, HasAVX10_2_512, In64BitMode] in {
  let SchedRW = [WriteSystem] in {
    let  usesCustomInserter = 1 in {
      def PTCVTROWD2PSrri : PseudoI<(outs VR512:$dst), (ins u8imm:$src1, i32u8imm:$src2),
                                    [(set VR512:$dst, (int_x86_tcvtrowd2ps timm:$src1, imm:$src2))]>;
      def PTCVTROWD2PSrre : PseudoI<(outs VR512:$dst), (ins u8imm:$src1, GR32:$src2),
                                    [(set VR512:$dst, (int_x86_tcvtrowd2ps timm:$src1, GR32:$src2))]>;
    }

    def PTCVTROWD2PSrriV : PseudoI<(outs VR512:$dst),
                                   (ins GR16:$src1, GR16:$src2, TILE:$src3, i32u8imm:$src4),
                                    [(set VR512: $dst,
                                      (int_x86_tcvtrowd2ps_internal GR16:$src1, GR16:$src2,
                                       TILE:$src3, imm:$src4))]>;
    def PTCVTROWD2PSrreV : PseudoI<(outs VR512:$dst),
                                   (ins GR16:$src1, GR16:$src2, TILE:$src3, GR32:$src4),
                                   [(set VR512: $dst,
                                     (int_x86_tcvtrowd2ps_internal GR16:$src1, GR16:$src2,
                                      TILE:$src3, GR32:$src4))]>;
    def PTCVTROWPS2BF16HrriV : PseudoI<(outs VR512:$dst),
                                       (ins GR16:$src1, GR16:$src2, TILE:$src3, i32u8imm:$src4),
                                       [(set VR512: $dst,
                                         (int_x86_tcvtrowps2bf16h_internal GR16:$src1, GR16:$src2,
                                          TILE:$src3, imm:$src4))]>;
    def PTCVTROWPS2BF16HrreV : PseudoI<(outs VR512:$dst),
                                       (ins GR16:$src1, GR16:$src2, TILE:$src3, GR32:$src4),
                                       [(set VR512: $dst,
                                         (int_x86_tcvtrowps2bf16h_internal GR16:$src1, GR16:$src2,
                                          TILE:$src3, GR32:$src4))]>;
    def PTCVTROWPS2BF16LrriV : PseudoI<(outs VR512:$dst),
                                       (ins GR16:$src1, GR16:$src2, TILE:$src3, i32u8imm:$src4),
                                       [(set VR512: $dst,
                                         (int_x86_tcvtrowps2bf16l_internal GR16:$src1, GR16:$src2,
                                          TILE:$src3, imm:$src4))]>;
    def PTCVTROWPS2BF16LrreV : PseudoI<(outs VR512:$dst),
                                       (ins GR16:$src1, GR16:$src2, TILE:$src3, GR32:$src4),
                                       [(set VR512: $dst,
                                         (int_x86_tcvtrowps2bf16l_internal GR16:$src1, GR16:$src2,
                                          TILE:$src3, GR32:$src4))]>;
    def PTCVTROWPS2PHHrriV : PseudoI<(outs VR512:$dst),
                                     (ins GR16:$src1, GR16:$src2, TILE:$src3, i32u8imm:$src4),
                                     [(set VR512: $dst,
                                       (int_x86_tcvtrowps2phh_internal GR16:$src1, GR16:$src2,
                                        TILE:$src3, imm:$src4))]>;
    def PTCVTROWPS2PHHrreV : PseudoI<(outs VR512:$dst),
                                     (ins GR16:$src1, GR16:$src2, TILE:$src3, GR32:$src4),
                                     [(set VR512: $dst,
                                       (int_x86_tcvtrowps2phh_internal GR16:$src1, GR16:$src2,
                                        TILE:$src3, GR32:$src4))]>;
    def PTCVTROWPS2PHLrriV : PseudoI<(outs VR512:$dst),
                                     (ins GR16:$src1, GR16:$src2, TILE:$src3, i32u8imm:$src4),
                                     [(set VR512: $dst,
                                       (int_x86_tcvtrowps2phl_internal GR16:$src1, GR16:$src2,
                                        TILE:$src3, imm:$src4))]>;
    def PTCVTROWPS2PHLrreV : PseudoI<(outs VR512:$dst),
                                     (ins GR16:$src1, GR16:$src2, TILE:$src3, GR32:$src4),
                                     [(set VR512: $dst,
                                       (int_x86_tcvtrowps2phl_internal GR16:$src1, GR16:$src2,
                                        TILE:$src3, GR32:$src4))]>;
  }
}

multiclass AMXAVX512_BASE<bits<8> Opcode1, bits<8> Opcode2, string Opstr,
                                Prefix P1, Prefix P2> {
  let Predicates = [HasAMXAVX512, HasAVX10_2_512, In64BitMode], SchedRW = [WriteSystem] in {
    let OpPrefix = P1 in
      def rre : I<Opcode1, MRMSrcReg4VOp3, (outs VR512:$dst),
                  (ins TILE:$src1, GR32:$src2),
                  !strconcat(Opstr, "\t{$src2, $src1, $dst|$dst, $src1, $src2}"),
                  []>, EVEX, VVVV, EVEX_V512, T8;
    let OpPrefix = P2 in
      def rri : Ii8<Opcode2, MRMSrcReg, (outs VR512:$dst),
                    (ins TILE:$src1, i32u8imm:$src2),
                    !strconcat(Opstr, "\t{$src2, $src1, $dst|$dst, $src1, $src2}"),
                    []>, EVEX, EVEX_V512, TA;
    let usesCustomInserter = 1 in {
      def "P"#NAME#"rre" : PseudoI<(outs VR512:$dst), (ins u8imm:$src1, GR32:$src2),
                                   [(set VR512:$dst,
                                    (!cast<Intrinsic>("int_x86_"#Opstr) timm:$src1, GR32:$src2))]>;
      def "P"#NAME#"rri" : PseudoI<(outs VR512:$dst), (ins u8imm:$src1, i32u8imm:$src2),
                                   [(set VR512:$dst,
                                    (!cast<Intrinsic>("int_x86_"#Opstr) timm:$src1, imm:$src2))]>;
    }
  }
}

defm TCVTROWPS2PHH : AMXAVX512_BASE<0x6d, 0x07, "tcvtrowps2phh", PS, PS>;
defm TCVTROWPS2PHL : AMXAVX512_BASE<0x6d, 0x77, "tcvtrowps2phl", PD, XD>;
defm TCVTROWPS2BF16H : AMXAVX512_BASE<0x6d, 0x07, "tcvtrowps2bf16h", XD, XD>;
defm TCVTROWPS2BF16L : AMXAVX512_BASE<0x6d, 0x77, "tcvtrowps2bf16l", XS, XS>;

multiclass m_tilemovrow {
  let Predicates = [HasAMXAVX512, HasAVX10_2_512, In64BitMode] in {
    let SchedRW = [WriteSystem] in {
      def rri : Ii8<0x7, MRMSrcReg, (outs VR512:$dst),
                    (ins TILE:$src1, u8imm:$src2),
                    "tilemovrow\t{$src2, $src1, $dst|$dst, $src1, $src2}",
                    []>, TA,PD, EVEX, EVEX_V512;
      def rre : I<0x4A, MRMSrcReg4VOp3, (outs VR512:$dst),
                  (ins TILE:$src1, GR32:$src2),
                  "tilemovrow\t{$src2, $src1, $dst|$dst, $src1, $src2}",
                  []>, T8,PD, EVEX, VVVV, EVEX_V512;
    }
  } // HasAMXAVX512, HasAVX10_2_512, In64BitMode
}

defm TILEMOVROW : m_tilemovrow;

let Predicates = [HasAMXAVX512, HasAVX10_2_512, In64BitMode] in {
  let SchedRW = [WriteSystem] in {
    let  usesCustomInserter = 1 in {
      def PTILEMOVROWrri : PseudoI<(outs VR512:$dst), (ins u8imm:$src1, i32u8imm:$src2),
                                   [(set VR512:$dst, (int_x86_tilemovrow timm:$src1, imm:$src2))]>;
      def PTILEMOVROWrre : PseudoI<(outs VR512:$dst), (ins u8imm:$src1, GR32:$src2),
                                   [(set VR512:$dst, (int_x86_tilemovrow timm:$src1, GR32:$src2))]>;
    }

    def PTILEMOVROWrriV : PseudoI<(outs VR512:$dst),
                                  (ins GR16:$src1, GR16:$src2, TILE:$src3, i32u8imm:$src4),
                                  [(set VR512: $dst,
                                    (int_x86_tilemovrow_internal GR16:$src1, GR16:$src2,
                                     TILE:$src3, imm:$src4))]>;
    def PTILEMOVROWrreV : PseudoI<(outs VR512:$dst),
                                  (ins GR16:$src1, GR16:$src2, TILE:$src3, GR32:$src4),
                                  [(set VR512: $dst,
                                    (int_x86_tilemovrow_internal GR16:$src1, GR16:$src2,
                                     TILE:$src3, GR32:$src4))]>;
  }
}

let Predicates = [HasAMXTF32, In64BitMode] in {
  let SchedRW = [WriteSystem] in {
    let Constraints = "$src1 = $dst" in {
      def TMMULTF32PS: I<0x48, MRMSrcReg4VOp3, (outs TILE:$dst),
                         (ins TILE:$src1, TILE:$src2, TILE:$src3),
                         "tmmultf32ps\t{$src3, $src2, $dst|$dst, $src2, $src3}",
                         []>, VEX, VVVV, T8, PD;
    }
    let Constraints = "$src4 = $dst" in {
      def PTMMULTF32PSV : PseudoI<(outs TILE:$dst),
                                  (ins GR16:$src1, GR16:$src2, GR16:$src3,
                                   TILE:$src4, TILE:$src5, TILE:$src6),
                                  [(set TILE:$dst,
                                    (int_x86_tmmultf32ps_internal GR16:$src1,
                                     GR16:$src2, GR16:$src3, TILE:$src4,
                                     TILE:$src5, TILE:$src6))]>;
    }
    let usesCustomInserter = 1 in {
      def PTMMULTF32PS : PseudoI<(outs),
                                 (ins u8imm:$src1, u8imm:$src2, u8imm:$src3),
                                 [(int_x86_tmmultf32ps timm:$src1, timm:$src2,
                                   timm:$src3)]>;
    }
  } // SchedRW = [WriteSystem]
} // HasAMXTF32

let Predicates = [HasAMXTF32, HasAMXTRANSPOSE, In64BitMode] in {
  let SchedRW = [WriteSystem] in {
    let Constraints = "$src1 = $dst" in {
      def TTMMULTF32PS: I<0x48, MRMSrcReg4VOp3, (outs TILE:$dst),
                         (ins TILE:$src1, TILE:$src2, TILE:$src3),
                         "ttmmultf32ps\t{$src3, $src2, $dst|$dst, $src2, $src3}",
                         []>, VEX, VVVV, T8, PS;
    }
    let Constraints = "$src4 = $dst" in {
      def PTTMMULTF32PSV : PseudoI<(outs TILE:$dst),
                                   (ins GR16:$src1, GR16:$src2, GR16:$src3,
                                    TILE:$src4, TILE:$src5, TILE:$src6),
                                   [(set TILE:$dst,
                                     (int_x86_ttmmultf32ps_internal GR16:$src1,
                                      GR16:$src2, GR16:$src3, TILE:$src4,
                                      TILE:$src5, TILE:$src6))]>;
    }
    let usesCustomInserter = 1 in {
      def PTTMMULTF32PS : PseudoI<(outs),
                                  (ins u8imm:$src1, u8imm:$src2, u8imm:$src3),
                                  [(int_x86_ttmmultf32ps timm:$src1, timm:$src2,
                                    timm:$src3)]>;
    }
  } // SchedRW = [WriteSystem]
} // HasAMXTF32, HasAMXTRANSPOSE

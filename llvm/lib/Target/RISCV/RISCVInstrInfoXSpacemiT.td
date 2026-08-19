//===-- RISCVInstrInfoXSpacemiT.td -------------------------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file describes the vendor extensions defined by SpacemiT.
//
//===----------------------------------------------------------------------===//

include "RISCVInstrFormatsSpacemitV.td"

let DecoderNamespace = "XSMT" in {
  let Predicates = [HasVendorXSMTVDot, IsRV64], ElementsDependOn = EltDepsNone,
      VS1VS2Constraint = WidenV, DestEEW = EEWSEWx4 in {
    // Base vector dot product (no slide) instructions
    // NOTE: Destination registers (vd) MUST be even-numbered (v0, v2, ..., v30)
    //       due to the number of elements processed requires a pair of register operations.
    // Using odd registers may cause undefined behavior.

    foreach a = VTypeEncodes in
      def SMT_VMADOT#!toupper(a.Name) :
            SMTVMADot<a.Encoding, "smt.vmadot"#a.Name>, Sched<[]>;

    //===----------------------------------------------------------------------===//
    // Sliding-window Vector Dot Product Instructions
    //
    // The numeric suffix (1, 2, 3) specifies the stride of the sliding window:
    //   1: Window slides by 1 element per operation
    //   2: Window slides by 2 elements per operation
    //   3: Window slides by 3 elements per operation
    //
    // These instructions compute dot products with overlapping operand windows
    // where the window position increments by <N> elements between
    // computations.
    //===----------------------------------------------------------------------===//
    // NOTE: Destination registers (vd) and first source register (vs1) MUST be
    //       even-numbered (v0, v2, ..., v30) due to the number of elements processed
    //       requires a pair of register operations.
    //       Using odd registers may cause undefined behavior.

    foreach a = SlideEncodes in
      foreach b = VTypeEncodes in
        def SMT_VMADOT#!toupper(a.Name)#!toupper(b.Name) :
            SMTVMADotSlide<a.Encoding, b.Encoding, "smt.vmadot"#a.Name#b.Name>, Sched<[]>;
  }

  class VFWMADTSLIDE_INSN<bits<3> funct3, string opcodestr>
      : SMTVFWMADOT<funct3, (outs VRM2:$vd_wb),
                    (ins VRM2:$vd, VRM2:$vs1, VR:$vs2),
                    opcodestr, "$vd, $vs1, $vs2">, Sched<[]>;

  class VNPACK_INSN<bits<7> funct7, bit funct1, string opcodestr>
      : SMTVPACK<funct7, funct1, (outs VR:$vd),
                 (ins VR:$vs1, VR:$vs2, uimm2:$imm2), opcodestr,
                 "$vd, $vs1, $vs2, $imm2">, Sched<[]>;

  class VPACK_INSN<bits<7> funct7, bit funct1, string opcodestr>
      : SMTVPACK<funct7, funct1, (outs VRM2:$vd),
                 (ins VR:$vs1, VR:$vs2, uimm2:$imm2), opcodestr,
                 "$vd, $vs1, $vs2, $imm2">, Sched<[]>;

  let Predicates = [HasVendorXSMTVDotII, IsRV64], ElementsDependOn = EltDepsNone in {
    let VS1VS2Constraint = WidenV in {
      // Base VMADOT II instructions
      // Support smt.vmadot/smt.vmadotu/smt.vmadotus/smt.vmadotsu
      foreach a = VTypeEncodes in
        def VMADOT#!toupper(a.Name) :
            SMTVMADotII<a.Encoding, "smt.vmadot"#a.Name>, Sched<[]>;

      // VMADOT Slide
      foreach a = SlideEncodes in
        foreach b = VTypeEncodes in
          def VMADOT#!toupper(a.Name)#!toupper(b.Name):
              SMTVMADotSlideII<a.Encoding, b.Encoding, "smt.vmadot"#a.Name#b.Name>, Sched<[]>;

      // Support smt.vmadot.sp/smt.vmadotu.sp/smt.vmadotus.sp/smt.vmadotsu.sp
      foreach a = VTypeEncodes in
        def VMADOT#!toupper(a.Name)#_SP :
            SMTVMADOTSP<a.Encoding, "smt.vmadot"#a.Name#".sp",
                        "$vd, $vs1, $vs2, $vmask, $imm2$vtype">, Sched<[]>;

      // Support smt.vmadot.hp/smt.vmadotu.hp/smt.vmadotus.hp/smt.vmadotsu.hp
      foreach c = HPEncodes in
        def VMADOT#!toupper(c.Name)#_HP :
            SMTVMADOTHP<c.Encoding, "smt.vmadot"#c.Name#".hp",
                        "$vd, $vs1, $vs2, $vmask, $imm3$vtype">, Sched<[]>;

      let DestEEW = EEWSEWx2, Constraints = "$vd = $vd_wb" in {
        // VFWMADOT
        def VFWMADOT :
            SMTVFWMADOT<0b100, (outs VRM2:$vd_wb),
                        (ins VRM2:$vd, VR:$vs1, VR:$vs2),
                        "smt.vfwmadot", "$vd, $vs1, $vs2">, Sched<[]>;
        // VFWMADOT Slide
        def VFWMADOT1 : VFWMADTSLIDE_INSN<0b101, "smt.vfwmadot1">;
        def VFWMADOT2 : VFWMADTSLIDE_INSN<0b110, "smt.vfwmadot2">;
        def VFWMADOT3 : VFWMADTSLIDE_INSN<0b111, "smt.vfwmadot3">;
      }

      def VNPACK_VV : VNPACK_INSN<0b0110001, 0b0, "smt.vnpack.vv">;
      def VNSPACK_VV : VNPACK_INSN<0b0110001, 0b1, "smt.vnspack.vv">;
      def VNPACK4_VV : VNPACK_INSN<0b0100001, 0b0, "smt.vnpack4.vv">;
      def VNSPACK4_VV : VNPACK_INSN<0b0100001, 0b1, "smt.vnspack4.vv">;
      def VPACK_VV : VPACK_INSN<0b0110011, 0b0, "smt.vpack.vv">;
      def VUPACK_VV : VPACK_INSN<0b0110011, 0b1, "smt.vupack.vv">;
    }
  }
} // DecoderNamespace = "XSMT"

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

//===----------------------------------------------------------------------===//
// Operand definitions.
//===----------------------------------------------------------------------===//

class SMTVDotOpcode<bits<7> val> {
  bits<7> Value = val;
}

class SMTVEncoding2<bits<2> val> {
  bits<2> Value = val;
}

def OPMMA       : SMTVDotOpcode<0b1110001>;
def OPMMA_SLIDE : SMTVDotOpcode<0b1110011>;

//===----------------------------------------------------------------------===//
// Vector Dot-Product Sign Encoding
// Defines the signed/unsigned mixing modes for vector dot-product operations.
// Encoding format: [1:0] bits
//   00: UU (Unsigned x Unsigned)
//   01: US (Unsigned x Signed)
//   10: SU (Signed x Unsigned)
//   11: SS (Signed x Signed)
//===----------------------------------------------------------------------===//
def SMT_VDot_UU        : SMTVEncoding2<0b00>;
def SMT_VDot_US        : SMTVEncoding2<0b01>;
def SMT_VDot_SU        : SMTVEncoding2<0b10>;
def SMT_VDot_SS        : SMTVEncoding2<0b11>;

//===----------------------------------------------------------------------===//
// Vector Dot-Product Sliding Window Modes
// Encoding format: [1:0] bits
//   00: Slide1 (1-element sliding stride)
//   01: Slide2 (2-element sliding stride)
//   10: Slide3 (3-element sliding stride)
//   11: Reserved
//
// Used in sliding-window dot-product operations:
//   vd = vs1 • vs2.slide{1|2|3}  // • = dot product
//===----------------------------------------------------------------------===//
def SMT_VDot_Slide1 : SMTVEncoding2<0b00>;
def SMT_VDot_Slide2 : SMTVEncoding2<0b01>;
def SMT_VDot_Slide3 : SMTVEncoding2<0b10>;

//===----------------------------------------------------------------------===//
// Instruction formats
//===----------------------------------------------------------------------===//

let hasSideEffects = 0, mayLoad = 0, mayStore = 0 in {
// Base vector dot product (no slide) format.
class RVInstSMTVDot<SMTVEncoding2 sign, string opcodestr, string argstr>
    : RVInst<(outs VRM2:$vd), (ins VR:$vs1, VR:$vs2), opcodestr, argstr, [], InstFormatR> {
  bits<5> vd;
  bits<5> vs1;
  bits<5> vs2;

  let Inst{31-25} = OPMMA.Value;
  let Inst{24-20} = vs2;
  let Inst{19-15} = vs1;
  let Inst{14} = 0b0;
  let Inst{13-12} = sign.Value;
  let Inst{11-8} = vd{4-1};
  let Inst{7} = 0b0;
  let Inst{6-0} = OPC_CUSTOM_1.Value;
}

// Sliding-window vector dot product format.
class RVInstSMTVDotSlide<SMTVEncoding2 funct2, SMTVEncoding2 sign, string opcodestr, string argstr>
    : RVInst<(outs VRM2:$vd), (ins VRM2:$vs1, VR:$vs2), opcodestr, argstr, [], InstFormatR> {
  bits<5> vd;
  bits<5> vs1;
  bits<5> vs2;

  let Inst{31-25} = OPMMA_SLIDE.Value;
  let Inst{24-20} = vs2;
  let Inst{19-16} = vs1{4-1};
  let Inst{15-14} = funct2.Value;
  let Inst{13-12} = sign.Value;
  let Inst{11-8} = vd{4-1};
  let Inst{7} = 0b0;
  let Inst{6-0} = OPC_CUSTOM_1.Value;
}
}

//===----------------------------------------------------------------------===//
// Instructions
//===----------------------------------------------------------------------===//

let DecoderNamespace = "XSMT" in {

let Predicates = [HasVendorXSMTVDot], ElementsDependOn = EltDepsVL in {
// Base vector dot product (no slide) instructions
// NOTE: Destination registers (vd) MUST be even-numbered (v0, v2, ..., v30)
//       due to hardware alignment constraints. Using odd registers may cause undefined behavior.
def SMT_VMADOT   : RVInstSMTVDot<SMT_VDot_SS, "smt.vmadot",   "$vd, $vs1, $vs2">;
def SMT_VMADOTU  : RVInstSMTVDot<SMT_VDot_UU, "smt.vmadotu",  "$vd, $vs1, $vs2">;
def SMT_VMADOTSU : RVInstSMTVDot<SMT_VDot_SU, "smt.vmadotsu", "$vd, $vs1, $vs2">;
def SMT_VMADOTUS : RVInstSMTVDot<SMT_VDot_US, "smt.vmadotus", "$vd, $vs1, $vs2">;

//===----------------------------------------------------------------------===//
// Sliding-window Vector Dot Product Instructions
//
// The numeric suffix (1, 2, 3) specifies the stride of the sliding window:
//   1: Window slides by 1 element per operation
//   2: Window slides by 2 elements per operation
//   3: Window slides by 3 elements per operation
//
// These instructions compute dot products with overlapping operand windows
// where the window position increments by <N> elements between computations.
//===----------------------------------------------------------------------===//
// NOTE: Destination registers (vd) and first source register (vs1) MUST be
//       even-numbered (v0, v2, ..., v30) due to hardware alignment constraints.
//       Using odd registers may cause undefined behavior.
def SMT_VMADOT1   : RVInstSMTVDotSlide<SMT_VDot_Slide1, SMT_VDot_SS, "smt.vmadot1",   "$vd, $vs1, $vs2">;
def SMT_VMADOT1U  : RVInstSMTVDotSlide<SMT_VDot_Slide1, SMT_VDot_UU, "smt.vmadot1u",  "$vd, $vs1, $vs2">;
def SMT_VMADOT1SU : RVInstSMTVDotSlide<SMT_VDot_Slide1, SMT_VDot_SU, "smt.vmadot1su", "$vd, $vs1, $vs2">;
def SMT_VMADOT1US : RVInstSMTVDotSlide<SMT_VDot_Slide1, SMT_VDot_US, "smt.vmadot1us", "$vd, $vs1, $vs2">;
def SMT_VMADOT2   : RVInstSMTVDotSlide<SMT_VDot_Slide2, SMT_VDot_SS, "smt.vmadot2",   "$vd, $vs1, $vs2">;
def SMT_VMADOT2U  : RVInstSMTVDotSlide<SMT_VDot_Slide2, SMT_VDot_UU, "smt.vmadot2u",  "$vd, $vs1, $vs2">;
def SMT_VMADOT2SU : RVInstSMTVDotSlide<SMT_VDot_Slide2, SMT_VDot_SU, "smt.vmadot2su", "$vd, $vs1, $vs2">;
def SMT_VMADOT2US : RVInstSMTVDotSlide<SMT_VDot_Slide2, SMT_VDot_US, "smt.vmadot2us", "$vd, $vs1, $vs2">;
def SMT_VMADOT3   : RVInstSMTVDotSlide<SMT_VDot_Slide3, SMT_VDot_SS, "smt.vmadot3",   "$vd, $vs1, $vs2">;
def SMT_VMADOT3U  : RVInstSMTVDotSlide<SMT_VDot_Slide3, SMT_VDot_UU, "smt.vmadot3u",  "$vd, $vs1, $vs2">;
def SMT_VMADOT3SU : RVInstSMTVDotSlide<SMT_VDot_Slide3, SMT_VDot_SU, "smt.vmadot3su", "$vd, $vs1, $vs2">;
def SMT_VMADOT3US : RVInstSMTVDotSlide<SMT_VDot_Slide3, SMT_VDot_US, "smt.vmadot3us", "$vd, $vs1, $vs2">;
}
} // DecoderNamespace = "XSMT"

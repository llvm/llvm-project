//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file describes the RISC-V instructions from the standard 'Zvvm' family
// of Integrated Matrix extensions.
// These extensions are still experimental as they haven't been ratified yet.
//
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// Instructions
//===----------------------------------------------------------------------===//

def VScaleAsmOperand : AsmOperandClass {
  let Name = "RVVScaleRegOpOperand";
  let RenderMethod = "addRegOperands";
  let PredicateMethod = "isV0Reg";
  let ParserMethod = "parseVScaleReg";
  let DiagnosticType = "InvalidVScaleRegister";
  let DiagnosticString = "operand must be v0.scale";
}

// An always present v0.scale operand encoded with vm=0. Classes that use this
// must set the vm field in RVInstV* to 0.
def VScaleOp : RegisterOperand<VMV0> {
  let ParserMatchClass = VScaleAsmOperand;
  let PrintMethod = "printVScaleReg";
  let EncoderMethod = "getVMaskReg";
  let DecoderMethod = "decodeVMaskReg";
}

class VIMEMACVV<bits<6> funct6, string opcodestr>
    : RVInstVV<funct6, OPIVV, (outs VR:$vd_wb),
               (ins VR:$vd, VR:$vs1, VR:$vs2), opcodestr,
               "$vd, $vs1, $vs2"> {
  let mayLoad = 0;
  let mayStore = 0;
  let hasSideEffects = 0;
  let Constraints = "$vd = $vd_wb";
  let vm = 1;
  let VMConstraint = false;
}

class VFMEMACVV<bits<6> funct6, string opcodestr>
    : RVInstVV<funct6, OPFVV, (outs VR:$vd_wb),
               (ins VR:$vd, VR:$vs1, VR:$vs2), opcodestr,
               "$vd, $vs1, $vs2"> {
  let mayLoad = 0;
  let mayStore = 0;
  let hasSideEffects = 0;
  let Constraints = "$vd = $vd_wb";
  let Uses = [FRM, VL, VTYPE];
  let mayRaiseFPException = true;
  let vm = 1;
  let VMConstraint = false;
}

class VFMEMACScaleVV<bits<6> funct6, string opcodestr>
    : RVInstVV<funct6, OPFVV, (outs VR:$vd_wb),
               (ins VR:$vd, VR:$vs1, VR:$vs2, VScaleOp:$vm), opcodestr,
               "$vd, $vs1, $vs2$vm"> {
  let mayLoad = 0;
  let mayStore = 0;
  let hasSideEffects = 0;
  let Constraints = "$vd = $vd_wb";
  let Uses = [FRM, VL, VTYPE];
  let mayRaiseFPException = true;
  let vm = 0;
  let VMConstraint = false;
}

class VIFMEMACScaleVV<bits<6> funct6, string opcodestr>
    : RVInstVV<funct6, OPIVV, (outs VR:$vd_wb),
               (ins VR:$vd, VR:$vs1, VR:$vs2, VScaleOp:$vm), opcodestr,
               "$vd, $vs1, $vs2$vm"> {
  let mayLoad = 0;
  let mayStore = 0;
  let hasSideEffects = 0;
  let Constraints = "$vd = $vd_wb";
  let Uses = [FRM, VL, VTYPE];
  let mayRaiseFPException = true;
  let vm = 0;
  let VMConstraint = false;
}

let Predicates = [HasStdExtZvvmm] in {
  def VMMACC_VV : VIMEMACVV<0b111000, "vmmacc.vv">;
  def VWMMACC_VV : VIMEMACVV<0b111001, "vwmmacc.vv">;
  def VQMMACC_VV : VIMEMACVV<0b111010, "vqmmacc.vv">;
  def V8WMMACC_VV : VIMEMACVV<0b111011, "v8wmmacc.vv">;
} // Predicates = [HasStdExtZvvmm]

let Predicates = [HasStdExtZvvfmm] in {
  def VFMMACC_VV : VFMEMACVV<0b010100, "vfmmacc.vv">;
  def VFWMMACC_VV : VFMEMACVV<0b010101, "vfwmmacc.vv">;
  def VFQMMACC_VV : VFMEMACVV<0b010110, "vfqmmacc.vv">;
  def VF8WMMACC_VV : VFMEMACVV<0b010111, "vf8wmmacc.vv">;
  def VFWMMACC_VV_SCALE : VFMEMACScaleVV<0b010101, "vfwmmacc.vv">;
  def VFQMMACC_VV_SCALE : VFMEMACScaleVV<0b010110, "vfqmmacc.vv">;
  def VF8WMMACC_VV_SCALE : VFMEMACScaleVV<0b010111, "vf8wmmacc.vv">;
  // FIXME: The integer-input MX forms should be gated by the
  // Zvvxi*/Zvvxni* microscaling extensions once LLVM models those
  // individual extension names. They are temporarily enabled under
  // experimental-zvvfmm for MC bring-up.
  def VFWIMMACC_VV : VIFMEMACScaleVV<0b111001, "vfwimmacc.vv">;
  def VFQIMMACC_VV : VIFMEMACScaleVV<0b111010, "vfqimmacc.vv">;
  def VF8WIMMACC_VV : VIFMEMACScaleVV<0b111011, "vf8wimmacc.vv">;
} // Predicates = [HasStdExtZvvfmm]

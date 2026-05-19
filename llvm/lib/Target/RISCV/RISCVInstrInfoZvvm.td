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

def TileLambdaAsmOperand : AsmOperandClass {
  let Name = "TileLambda";
  let RenderMethod = "addImmOperands";
  let ParserMethod = "parseTileLambda";
  let DiagnosticType = "InvalidTileLambda";
  let DiagnosticString = "operand must be L1, L2, L4, L8, L16, L32, or L64";
}

def TileLambdaOp : RISCVOp {
  let ParserMatchClass = TileLambdaAsmOperand;
  let PrintMethod = "printTileLambda";
  let EncoderMethod = "getImmOpValue";
  let DecoderMethod = "decodeUImmNonZeroOperand<3>";
  let OperandType = "OPERAND_UIMM3";
}

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

class VTileLoadBase<bits<2> mop, dag ins, string opcodestr, string argstr>
    : RVInst<(outs VR:$vd),
             ins, opcodestr, argstr, [], InstFormatR> {
  bits<5> vd;
  bits<5> rs1;
  bits<5> rs2;
  bit vm;
  bits<3> vlambda;

  let Inst{31-29} = vlambda;
  let Inst{28} = 1;
  let Inst{27-26} = mop;
  let Inst{25} = vm;
  let Inst{24-20} = rs2;
  let Inst{19-15} = rs1;
  let Inst{14-12} = 0b111;
  let Inst{11-7} = vd;
  let Inst{6-0} = OPC_LOAD_FP.Value;

  let mayLoad = true;
  let mayStore = false;
  let hasSideEffects = false;
  let Uses = [VL, VTYPE];
  let VMConstraint = true;
  let UseNamedOperandTable = true;
}

class VTileStoreBase<bits<2> mop, dag ins, string opcodestr, string argstr>
    : RVInst<(outs), ins, opcodestr, argstr, [], InstFormatR> {
  bits<5> vs3;
  bits<5> rs1;
  bits<5> rs2;
  bit vm;
  bits<3> vlambda;

  let Inst{31-29} = vlambda;
  let Inst{28} = 1;
  let Inst{27-26} = mop;
  let Inst{25} = vm;
  let Inst{24-20} = rs2;
  let Inst{19-15} = rs1;
  let Inst{14-12} = 0b111;
  let Inst{11-7} = vs3;
  let Inst{6-0} = OPC_STORE_FP.Value;

  let mayLoad = false;
  let mayStore = true;
  let hasSideEffects = false;
  let Uses = [VL, VTYPE];
  let UseNamedOperandTable = true;
}

class VTileLoad<bits<2> mop, string opcodestr>
    : VTileLoadBase<mop, (ins GPRMemZeroOffset:$rs1, GPR:$rs2, VMaskOp:$vm),
                    opcodestr, "$vd, $rs1, $rs2$vm"> {
  let vlambda = 0;
}

class VTileLoadLambda<bits<2> mop, string opcodestr>
    : VTileLoadBase<mop, (ins GPRMemZeroOffset:$rs1, GPR:$rs2,
                              TileLambdaOp:$vlambda, VMaskOp:$vm),
                    opcodestr, "$vd, $rs1, $rs2$vlambda$vm">;

class VTileStore<bits<2> mop, string opcodestr>
    : VTileStoreBase<mop,
                     (ins VR:$vs3, GPRMemZeroOffset:$rs1, GPR:$rs2,
                          VMaskOp:$vm),
                     opcodestr, "$vs3, $rs1, $rs2$vm"> {
  let vlambda = 0;
}

class VTileStoreLambda<bits<2> mop, string opcodestr>
    : VTileStoreBase<mop,
                     (ins VR:$vs3, GPRMemZeroOffset:$rs1, GPR:$rs2,
                          TileLambdaOp:$vlambda, VMaskOp:$vm),
                     opcodestr, "$vs3, $rs1, $rs2$vlambda$vm">;

class VIMEMACVV<bits<6> funct6, string opcodestr>
    : RVInstVV<funct6, OPIVV, (outs VR:$vd_wb),
               (ins VR:$vd, VR:$vs1, VR:$vs2), opcodestr,
               "$vd, $vs1, $vs2"> {
  let mayLoad = false;
  let mayStore = false;
  let hasSideEffects = false;
  let Constraints = "$vd = $vd_wb";
  let Uses = [VL, VTYPE];
  let vm = 1;
  let VMConstraint = false;
}

class VFMEMACVV<bits<6> funct6, string opcodestr>
    : RVInstVV<funct6, OPFVV, (outs VR:$vd_wb),
               (ins VR:$vd, VR:$vs1, VR:$vs2), opcodestr,
               "$vd, $vs1, $vs2"> {
  let mayLoad = false;
  let mayStore = false;
  let hasSideEffects = false;
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
  let mayLoad = false;
  let mayStore = false;
  let hasSideEffects = false;
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
  let mayLoad = false;
  let mayStore = false;
  let hasSideEffects = false;
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

let Predicates = [HasStdExtZvvmtls] in {
  def VMTL_V_L : VTileLoadLambda<0b00, "vmtl.v">;
  def VMTL_V : VTileLoad<0b00, "vmtl.v">;
  def VMTS_V_L : VTileStoreLambda<0b00, "vmts.v">;
  def VMTS_V : VTileStore<0b00, "vmts.v">;
} // Predicates = [HasStdExtZvvmtls]

let Predicates = [HasStdExtZvvmttls] in {
  def VMTTL_V_L : VTileLoadLambda<0b01, "vmttl.v">;
  def VMTTL_V : VTileLoad<0b01, "vmttl.v">;
  def VMTTS_V_L : VTileStoreLambda<0b01, "vmtts.v">;
  def VMTTS_V : VTileStore<0b01, "vmtts.v">;
} // Predicates = [HasStdExtZvvmttls]

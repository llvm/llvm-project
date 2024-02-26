//===-- RISCVInstrInfoXTHeadV.td ---------------------------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// This file describes the RISC-V instructions from the standard 'V' Vector
/// extension, version 0.7.1.
///
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// Operand and SDNode transformation definitions.
//===----------------------------------------------------------------------===//

def XTHeadVTypeIAsmOperand : AsmOperandClass {
  let Name = "XTHeadVTypeI";
  let ParserMethod = "parseXTHeadVTypeI";
  let DiagnosticType = "InvalidXTHeadVTypeI";
  let RenderMethod = "addVTypeIOperands";
}

def XTHeadVTypeI : Operand<XLenVT> {
  let ParserMatchClass = XTHeadVTypeIAsmOperand;
  let PrintMethod = "printXTHeadVTypeI";
  let DecoderMethod = "decodeUImmOperand<11>";
  let OperandType = "OPERAND_XTHEADVTYPEI";
  let OperandNamespace = "RISCVOp";
  let MCOperandPredicate = [{
    int64_t Imm;
    if (MCOp.evaluateAsConstantImm(Imm))
      return isUInt<11>(Imm);
    return MCOp.isBareSymbolRef();
  }];
}

//===----------------------------------------------------------------------===//
// Instruction class templates
//===----------------------------------------------------------------------===//

class TH_VLoadStore<bits<3> nf, RISCVOpcode opcode,
                    bits<3> mop, bits<3> width, dag outs, dag ins,
                    string opcodestr, string argstr>
    : RVInst<outs, ins, opcodestr, argstr, [], InstFormatR> {
  bits<5> rs2;
  bits<5> rs1;
  bits<5> rd;
  bit vm;

  let Inst{31-29} = nf;
  let Inst{28-26} = mop;
  let Inst{25} = vm;
  let Inst{24-20} = rs2;
  let Inst{19-15} = rs1;
  let Inst{14-12} = width;
  let Inst{11-7} = rd;
  let Inst{6-0} = opcode.Value;

  let Uses = [VTYPE, VL];
}

let hasSideEffects = 0, mayLoad = 1, mayStore = 0, RVVConstraint = VMConstraint in {
  class TH_VLxU<bits<3> nf, bits<3> width, string opcodestr>
      : TH_VLoadStore<nf, OPC_LOAD_FP, 0b000, width, (outs VR:$rd),
                      (ins GPRMemZeroOffset:$rs1, VMaskOp:$vm),
                      opcodestr, "$rd, ${rs1}$vm"> {
    let rs2 = 0b00000;
  }
  class TH_VLx<bits<3> nf, bits<3> width, string opcodestr>
      : TH_VLoadStore<nf, OPC_LOAD_FP, 0b100, width, (outs VR:$rd),
                      (ins GPRMemZeroOffset:$rs1, VMaskOp:$vm),
                      opcodestr, "$rd, ${rs1}$vm"> {
    let rs2 = 0b00000;
  }
  class TH_VLxUFF<bits<3> nf, bits<3> width, string opcodestr>
      : TH_VLoadStore<nf, OPC_LOAD_FP, 0b000, width, (outs VR:$rd),
                      (ins GPRMemZeroOffset:$rs1, VMaskOp:$vm),
                      opcodestr, "$rd, ${rs1}$vm"> {
    let rs2 = 0b10000;
  }
  class TH_VLxFF<bits<3> nf, bits<3> width, string opcodestr>
      : TH_VLoadStore<nf, OPC_LOAD_FP, 0b100, width, (outs VR:$rd),
                      (ins GPRMemZeroOffset:$rs1, VMaskOp:$vm),
                      opcodestr, "$rd, ${rs1}$vm"> {
    let rs2 = 0b10000;
  }
  class TH_VLSxU<bits<3> nf, bits<3> width, string opcodestr>
      : TH_VLoadStore<nf, OPC_LOAD_FP, 0b010, width, (outs VR:$rd),
                      (ins GPRMemZeroOffset:$rs1, GPR:$rs2, VMaskOp:$vm),
                      opcodestr, "$rd, $rs1, $rs2$vm">;
  class TH_VLSx<bits<3> nf, bits<3> width, string opcodestr>
      : TH_VLoadStore<nf, OPC_LOAD_FP, 0b110, width, (outs VR:$rd),
                      (ins GPRMemZeroOffset:$rs1, GPR:$rs2, VMaskOp:$vm),
                      opcodestr, "$rd, $rs1, $rs2$vm">;
  class TH_VLXxU<bits<3> nf, bits<3> width, string opcodestr>
      : TH_VLoadStore<nf, OPC_LOAD_FP, 0b011, width, (outs VR:$rd),
                      (ins GPRMemZeroOffset:$rs1, VR:$rs2, VMaskOp:$vm),
                      opcodestr, "$rd, $rs1, $rs2$vm">;
  class TH_VLXx<bits<3> nf, bits<3> width, string opcodestr>
      : TH_VLoadStore<nf, OPC_LOAD_FP, 0b111, width, (outs VR:$rd),
                      (ins GPRMemZeroOffset:$rs1, VR:$rs2, VMaskOp:$vm),
                      opcodestr, "$rd, $rs1, $rs2$vm">;
}

let hasSideEffects = 0, mayLoad = 0, mayStore = 1 in {
  class TH_VSx<bits<3> nf, bits<3> width, string opcodestr>
      : TH_VLoadStore<nf, OPC_STORE_FP, 0b000, width, (outs),
                      (ins VR:$rd, GPRMemZeroOffset:$rs1, VMaskOp:$vm),
                      opcodestr, "$rd, ${rs1}$vm"> {
    let rs2 = 0b00000;
  }
  class TH_VSSx<bits<3> nf, bits<3> width, string opcodestr>
      : TH_VLoadStore<nf, OPC_STORE_FP, 0b010, width, (outs),
                      (ins VR:$rd, GPRMemZeroOffset:$rs1, GPR:$rs2, VMaskOp:$vm),
                      opcodestr, "$rd, $rs1, $rs2$vm">;
  class TH_VSXx<bits<3> nf, bits<3> width, string opcodestr>
      : TH_VLoadStore<nf, OPC_STORE_FP, 0b011, width, (outs),
                      (ins VR:$rd, GPRMemZeroOffset:$rs1, VR:$rs2, VMaskOp:$vm),
                      opcodestr, "$rd, $rs1, $rs2$vm">;
  class TH_VSUXx<bits<3> nf, bits<3> width, string opcodestr>
      : TH_VLoadStore<nf, OPC_STORE_FP, 0b111, width, (outs),
                      (ins VR:$rd, GPRMemZeroOffset:$rs1, VR:$rs2, VMaskOp:$vm),
                      opcodestr, "$rd, $rs1, $rs2$vm">;
}

multiclass TH_VWSMAC_V_X<string opcodestr, bits<6> funct6> {
  def V : VALUrVV<funct6, OPIVV, opcodestr # ".vv">,
          Sched<[WriteVIWMulAddV_WorstCase, ReadVIWMulAddV_WorstCase,
                 ReadVIWMulAddV_WorstCase, ReadVMask]>;
  def X : VALUrVX<funct6, OPIVX, opcodestr # ".vx">,
          Sched<[WriteVIWMulAddX_WorstCase, ReadVIWMulAddV_WorstCase,
                 ReadVIWMulAddX_WorstCase, ReadVMask]>;
}

multiclass TH_VWSMAC_X<string opcodestr, bits<6> funct6> {
  def X : VALUrVX<funct6, OPIVX, opcodestr # ".vx">,
          Sched<[WriteVIWMulAddX_WorstCase, ReadVIWMulAddV_WorstCase,
                 ReadVIWMulAddX_WorstCase, ReadVMask]>;
}

multiclass TH_VNCLP_IV_V_X_I<string opcodestr, bits<6> funct6> {
  def V  : VALUVV<funct6, OPIVV, opcodestr # ".vv">,
           Sched<[WriteVNClipV_WorstCase, ReadVNClipV_WorstCase,
                  ReadVNClipV_WorstCase, ReadVMask]>;
  def X  : VALUVX<funct6, OPIVX, opcodestr # ".vx">,
           Sched<[WriteVNClipX_WorstCase, ReadVNClipV_WorstCase,
                  ReadVNClipX_WorstCase, ReadVMask]>;
  def I  : VALUVI<funct6, opcodestr # ".vi", uimm5>,
           Sched<[WriteVNClipI_WorstCase, ReadVNClipV_WorstCase,
                  ReadVMask]>;
}

multiclass TH_VNSHT_IV_V_X_I<string opcodestr, bits<6> funct6> {
  def V  : VALUVV<funct6, OPIVV, opcodestr # ".vv">,
           Sched<[WriteVNShiftV_WorstCase, ReadVNShiftV_WorstCase,
                  ReadVNShiftV_WorstCase, ReadVMask]>;
  def X  : VALUVX<funct6, OPIVX, opcodestr # ".vx">,
           Sched<[WriteVNShiftX_WorstCase, ReadVNShiftV_WorstCase,
                  ReadVNShiftX_WorstCase, ReadVMask]>;
  def I  : VALUVI<funct6, opcodestr # ".vi", uimm5>,
           Sched<[WriteVNShiftI_WorstCase, ReadVNShiftV_WorstCase,
                  ReadVMask]>;
}

let hasSideEffects = 0, mayLoad = 0, mayStore = 0 in {
// rvv 0.7.1 set vm=1 for `op vd, vs2, vs1, v0`
class TH_VALUmVV<bits<6> funct6, RISCVVFormat opv, string opcodestr>
    : RVInstVV<funct6, opv, (outs VR:$vd),
               (ins VR:$vs2, VR:$vs1, VMV0:$v0),
               opcodestr, "$vd, $vs2, $vs1, v0"> {
  let vm = 1;
}

// rvv 0.7.1 set vm=1 for `op vd, vs2, rs1, v0`
class TH_VALUmVX<bits<6> funct6, RISCVVFormat opv, string opcodestr>
    : RVInstVX<funct6, opv, (outs VR:$vd),
               (ins VR:$vs2, GPR:$rs1, VMV0:$v0),
               opcodestr, "$vd, $vs2, $rs1, v0"> {
  let vm = 1;
}

// rvv 0.7.1 set vm=1 for `op vd, vs2, imm, v0`
class TH_VALUmVI<bits<6> funct6, string opcodestr, Operand optype = simm5>
    : RVInstIVI<funct6, (outs VR:$vd),
                (ins VR:$vs2, optype:$imm, VMV0:$v0),
                opcodestr, "$vd, $vs2, $imm, v0"> {
  let vm = 1;
}

multiclass TH_VALUm_IV_V_X<string opcodestr, bits<6> funct6> {
  def VM : TH_VALUmVV<funct6, OPIVV, opcodestr # ".vvm">,
           Sched<[WriteVICALUV_WorstCase, ReadVICALUV_WorstCase,
                  ReadVICALUV_WorstCase, ReadVMask]>;
  def XM : TH_VALUmVX<funct6, OPIVX, opcodestr # ".vxm">,
           Sched<[WriteVICALUX_WorstCase, ReadVICALUV_WorstCase,
                  ReadVICALUX_WorstCase, ReadVMask]>;
}

multiclass TH_VALUm_IV_V_X_I<string opcodestr, bits<6> funct6>
    : TH_VALUm_IV_V_X<opcodestr, funct6> {
  def IM : TH_VALUmVI<funct6, opcodestr # ".vim">,
           Sched<[WriteVICALUI_WorstCase, ReadVICALUV_WorstCase,
                  ReadVMask]>;
}
} // hasSideEffects = 0, mayLoad = 0, mayStore = 0

class TH_InstVAMO<bits<5> amoop, bits<3> width, dag outs,
                  dag ins, string opcodestr, string argstr>
    : RVInst<outs, ins, opcodestr, argstr, [], InstFormatR> {
  bits<5> vs2;
  bits<5> rs1;
  bit wd;
  bit vm;

  let Inst{31-27} = amoop;
  let Inst{26} = wd;
  let Inst{25} = vm;
  let Inst{24-20} = vs2;
  let Inst{19-15} = rs1;
  let Inst{14-12} = width;
  let Inst{6-0} = OPC_AMO.Value;

  let Uses = [VTYPE, VL];
}

let hasSideEffects = 0, mayLoad = 1, mayStore = 1 in {
// vamo vd, vs2, (rs1), vd, vm
class TH_VAMOWd<bits<5> amoop, bits<3> width, string opcodestr>
    : TH_InstVAMO<amoop, width, (outs VR:$vd_wd),
                  (ins VR:$vs2, GPR:$rs1, VR:$vd, VMaskOp:$vm),
                  opcodestr, "$vd_wd, $vs2, (${rs1}), $vd$vm"> {
  let Constraints = "$vd_wd = $vd";
  let wd = 1;
  bits<5> vd;
  let Inst{11-7} = vd;
}

// vamo x0, vs2, (rs1), vs3, vm
class TH_VAMONoWd<bits<5> amoop, bits<3> width, string opcodestr>
    : TH_InstVAMO<amoop, width, (outs),
                  (ins VR:$vs2, GPR:$rs1, VR:$vs3, VMaskOp:$vm),
                  opcodestr, "x0, $vs2, (${rs1}), $vs3$vm"> {
  let wd = 0;
  bits<5> vs3;
  let Inst{11-7} = vs3;
}

} // hasSideEffects = 0, mayLoad = 1, mayStore = 1

multiclass TH_VAMO<bits<5> amoop, bits<3> width, string opcodestr> {
  def _WD_V : TH_VAMOWd<amoop, width, opcodestr>;
  def _UNWD_V : TH_VAMONoWd<amoop, width, opcodestr>;
}

//===----------------------------------------------------------------------===//
// Instructions
//===----------------------------------------------------------------------===//

let DecoderNamespace = "RVV0p71" in {
let Predicates = [HasVendorXTHeadV] in {
// Configuration-Setting Instructions
let hasSideEffects = 1, mayLoad = 0, mayStore = 0 in {
def TH_VSETVLI : RVInstSetVLi<(outs GPR:$rd), (ins GPR:$rs1, XTHeadVTypeI:$vtypei),
                              "th.vsetvli", "$rd, $rs1, $vtypei">,
                 Sched<[WriteVSETVLI, ReadVSETVLI]>;
def TH_VSETVL : RVInstSetVL<(outs GPR:$rd), (ins GPR:$rs1, GPR:$rs2),
                            "th.vsetvl", "$rd, $rs1, $rs2">,
                Sched<[WriteVSETVL, ReadVSETVL, ReadVSETVL]>;
} // hasSideEffects = 1, mayLoad = 0, mayStore = 0

// Vector Unit-Stride Instructions
def TH_VLB_V : TH_VLx<0b000, 0b000, "th.vlb.v">;
def TH_VLH_V : TH_VLx<0b000, 0b101, "th.vlh.v">;
def TH_VLW_V : TH_VLx<0b000, 0b110, "th.vlw.v">;
def TH_VLBU_V : TH_VLxU<0b000, 0b000, "th.vlbu.v">;
def TH_VLHU_V : TH_VLxU<0b000, 0b101, "th.vlhu.v">;
def TH_VLWU_V : TH_VLxU<0b000, 0b110, "th.vlwu.v">;
def TH_VLE_V : TH_VLxU<0b000, 0b111, "th.vle.v">;
def TH_VSB_V : TH_VSx<0b000, 0b000, "th.vsb.v">;
def TH_VSH_V : TH_VSx<0b000, 0b101, "th.vsh.v">;
def TH_VSW_V : TH_VSx<0b000, 0b110, "th.vsw.v">;
def TH_VSE_V : TH_VSx<0b000, 0b111, "th.vse.v">;

// Vector Strided Instructions
def TH_VLSB_V : TH_VLSx<0b000, 0b000, "th.vlsb.v">;
def TH_VLSH_V : TH_VLSx<0b000, 0b101, "th.vlsh.v">;
def TH_VLSW_V : TH_VLSx<0b000, 0b110, "th.vlsw.v">;
def TH_VLSBU_V : TH_VLSxU<0b000, 0b000, "th.vlsbu.v">;
def TH_VLSHU_V : TH_VLSxU<0b000, 0b101, "th.vlshu.v">;
def TH_VLSWU_V : TH_VLSxU<0b000, 0b110, "th.vlswu.v">;
def TH_VLSE_V : TH_VLSxU<0b000, 0b111, "th.vlse.v">;
def TH_VSSB_V : TH_VSSx<0b000, 0b000, "th.vssb.v">;
def TH_VSSH_V : TH_VSSx<0b000, 0b101, "th.vssh.v">;
def TH_VSSW_V : TH_VSSx<0b000, 0b110, "th.vssw.v">;
def TH_VSSE_V : TH_VSSx<0b000, 0b111, "th.vsse.v">;

// Vector indexed loads and stores
def TH_VLXB_V : TH_VLXx<0b000, 0b000, "th.vlxb.v">;
def TH_VLXH_V : TH_VLXx<0b000, 0b101, "th.vlxh.v">;
def TH_VLXW_V : TH_VLXx<0b000, 0b110, "th.vlxw.v">;
def TH_VLXBU_V : TH_VLXxU<0b000, 0b000, "th.vlxbu.v">;
def TH_VLXHU_V : TH_VLXxU<0b000, 0b101, "th.vlxhu.v">;
def TH_VLXWU_V : TH_VLXxU<0b000, 0b110, "th.vlxwu.v">;
def TH_VLXE_V : TH_VLXxU<0b000, 0b111, "th.vlxe.v">;
def TH_VSXB_V : TH_VSXx<0b000, 0b000, "th.vsxb.v">;
def TH_VSXH_V : TH_VSXx<0b000, 0b101, "th.vsxh.v">;
def TH_VSXW_V : TH_VSXx<0b000, 0b110, "th.vsxw.v">;
def TH_VSXE_V : TH_VSXx<0b000, 0b111, "th.vsxe.v">;
def TH_VSUXB_V : TH_VSUXx<0b000, 0b000, "th.vsuxb.v">;
def TH_VSUXH_V : TH_VSUXx<0b000, 0b101, "th.vsuxh.v">;
def TH_VSUXW_V : TH_VSUXx<0b000, 0b110, "th.vsuxw.v">;
def TH_VSUXE_V : TH_VSUXx<0b000, 0b111, "th.vsuxe.v">;

// Unit-stride Fault-Only-First Loads
def TH_VLBFF_V : TH_VLxFF<0b000, 0b000, "th.vlbff.v">;
def TH_VLHFF_V : TH_VLxFF<0b000, 0b101, "th.vlhff.v">;
def TH_VLWFF_V : TH_VLxFF<0b000, 0b110, "th.vlwff.v">;
def TH_VLBUFF_V : TH_VLxUFF<0b000, 0b000, "th.vlbuff.v">;
def TH_VLHUFF_V : TH_VLxUFF<0b000, 0b101, "th.vlhuff.v">;
def TH_VLWUFF_V : TH_VLxUFF<0b000, 0b110, "th.vlwuff.v">;
def TH_VLEFF_V : TH_VLxUFF<0b000, 0b111, "th.vleff.v">;
} // Predicates = [HasVendorXTHeadV]

// The extension `Zvlsseg (chapter 7.8)` is not a subextension but a mandatory part of `XTheadVector`.
let Predicates = [HasVendorXTHeadV] in {
foreach nf=2-8 in {
  // Vector Unit-Stride Segment Loads and Stores
  def TH_VLSEG#nf#B_V : TH_VLx<!add(nf, -1), 0b000, "th.vlseg"#nf#"b.v">;
  def TH_VLSEG#nf#H_V : TH_VLx<!add(nf, -1), 0b101, "th.vlseg"#nf#"h.v">;
  def TH_VLSEG#nf#W_V : TH_VLx<!add(nf, -1), 0b110, "th.vlseg"#nf#"w.v">;
  def TH_VLSEG#nf#BU_V : TH_VLxU<!add(nf, -1), 0b000, "th.vlseg"#nf#"bu.v">;
  def TH_VLSEG#nf#HU_V : TH_VLxU<!add(nf, -1), 0b101, "th.vlseg"#nf#"hu.v">;
  def TH_VLSEG#nf#WU_V : TH_VLxU<!add(nf, -1), 0b110, "th.vlseg"#nf#"wu.v">;
  def TH_VLSEG#nf#E_V : TH_VLxU<!add(nf, -1), 0b111, "th.vlseg"#nf#"e.v">;
  def TH_VSSEG#nf#B_V : TH_VSx<!add(nf, -1), 0b000, "th.vsseg"#nf#"b.v">;
  def TH_VSSEG#nf#H_V : TH_VSx<!add(nf, -1), 0b101, "th.vsseg"#nf#"h.v">;
  def TH_VSSEG#nf#W_V : TH_VSx<!add(nf, -1), 0b110, "th.vsseg"#nf#"w.v">;
  def TH_VSSEG#nf#E_V : TH_VSx<!add(nf, -1), 0b111, "th.vsseg"#nf#"e.v">;
  def TH_VLSEG#nf#BFF_V : TH_VLxFF<!add(nf, -1), 0b000, "th.vlseg"#nf#"bff.v">;
  def TH_VLSEG#nf#HFF_V : TH_VLxFF<!add(nf, -1), 0b101, "th.vlseg"#nf#"hff.v">;
  def TH_VLSEG#nf#WFF_V : TH_VLxFF<!add(nf, -1), 0b110, "th.vlseg"#nf#"wff.v">;
  def TH_VLSEG#nf#BUFF_V : TH_VLxUFF<!add(nf, -1), 0b000, "th.vlseg"#nf#"buff.v">;
  def TH_VLSEG#nf#HUFF_V : TH_VLxUFF<!add(nf, -1), 0b101, "th.vlseg"#nf#"huff.v">;
  def TH_VLSEG#nf#WUFF_V : TH_VLxUFF<!add(nf, -1), 0b110, "th.vlseg"#nf#"wuff.v">;
  def TH_VLSEG#nf#EFF_V : TH_VLxUFF<!add(nf, -1), 0b111, "th.vlseg"#nf#"eff.v">;

  // Vector Strided Segment Loads and Stores
  def TH_VLSSEG#nf#B_V : TH_VLSx<!add(nf, -1), 0b000, "th.vlsseg"#nf#"b.v">;
  def TH_VLSSEG#nf#H_V : TH_VLSx<!add(nf, -1), 0b101, "th.vlsseg"#nf#"h.v">;
  def TH_VLSSEG#nf#W_V : TH_VLSx<!add(nf, -1), 0b110, "th.vlsseg"#nf#"w.v">;
  def TH_VLSSEG#nf#BU_V : TH_VLSxU<!add(nf, -1), 0b000, "th.vlsseg"#nf#"bu.v">;
  def TH_VLSSEG#nf#HU_V : TH_VLSxU<!add(nf, -1), 0b101, "th.vlsseg"#nf#"hu.v">;
  def TH_VLSSEG#nf#WU_V : TH_VLSxU<!add(nf, -1), 0b110, "th.vlsseg"#nf#"wu.v">;
  def TH_VLSSEG#nf#E_V : TH_VLSxU<!add(nf, -1), 0b111, "th.vlsseg"#nf#"e.v">;
  def TH_VSSSEG#nf#B_V : TH_VSSx<!add(nf, -1), 0b000, "th.vssseg"#nf#"b.v">;
  def TH_VSSSEG#nf#H_V : TH_VSSx<!add(nf, -1), 0b101, "th.vssseg"#nf#"h.v">;
  def TH_VSSSEG#nf#W_V : TH_VSSx<!add(nf, -1), 0b110, "th.vssseg"#nf#"w.v">;
  def TH_VSSSEG#nf#E_V : TH_VSSx<!add(nf, -1), 0b111, "th.vssseg"#nf#"e.v">;

  // Vector Indexed Segment Loads and Stores
  def TH_VLXSEG#nf#B_V : TH_VLXx<!add(nf, -1), 0b000, "th.vlxseg"#nf#"b.v">;
  def TH_VLXSEG#nf#H_V : TH_VLXx<!add(nf, -1), 0b101, "th.vlxseg"#nf#"h.v">;
  def TH_VLXSEG#nf#W_V : TH_VLXx<!add(nf, -1), 0b110, "th.vlxseg"#nf#"w.v">;
  def TH_VLXSEG#nf#BU_V : TH_VLXxU<!add(nf, -1), 0b000, "th.vlxseg"#nf#"bu.v">;
  def TH_VLXSEG#nf#HU_V : TH_VLXxU<!add(nf, -1), 0b101, "th.vlxseg"#nf#"hu.v">;
  def TH_VLXSEG#nf#WU_V : TH_VLXxU<!add(nf, -1), 0b110, "th.vlxseg"#nf#"wu.v">;
  def TH_VLXSEG#nf#E_V : TH_VLXxU<!add(nf, -1), 0b111, "th.vlxseg"#nf#"e.v">;
  def TH_VSXSEG#nf#B_V : TH_VSXx<!add(nf, -1), 0b000, "th.vsxseg"#nf#"b.v">;
  def TH_VSXSEG#nf#H_V : TH_VSXx<!add(nf, -1), 0b101, "th.vsxseg"#nf#"h.v">;
  def TH_VSXSEG#nf#W_V : TH_VSXx<!add(nf, -1), 0b110, "th.vsxseg"#nf#"w.v">;
  def TH_VSXSEG#nf#E_V : TH_VSXx<!add(nf, -1), 0b111, "th.vsxseg"#nf#"e.v">;
}
} // Predicates = [HasVendorXTHeadV]

// The extension `Zvamo` is renamed to `XTheadZvamo`.
let Predicates = [HasVendorXTHeadV, HasVendorXTHeadZvamo, HasStdExtA] in {
  // Vector AMO Instruction
  defm TH_VAMOSWAPW : TH_VAMO<0b00001, 0b110, "th.vamoswapw.v">;
  defm TH_VAMOADDW : TH_VAMO<0b00000, 0b110, "th.vamoaddw.v">;
  defm TH_VAMOXORW : TH_VAMO<0b00100, 0b110, "th.vamoxorw.v">;
  defm TH_VAMOANDW : TH_VAMO<0b01100, 0b110, "th.vamoandw.v">;
  defm TH_VAMOORW : TH_VAMO<0b01000, 0b110, "th.vamoorw.v">;
  defm TH_VAMOMINW : TH_VAMO<0b10000, 0b110, "th.vamominw.v">;
  defm TH_VAMOMAXW : TH_VAMO<0b10100, 0b110, "th.vamomaxw.v">;
  defm TH_VAMOMINUW : TH_VAMO<0b11000, 0b110, "th.vamominuw.v">;
  defm TH_VAMOMAXUW : TH_VAMO<0b11100, 0b110, "th.vamomaxuw.v">;

  defm TH_VAMOSWAPD : TH_VAMO<0b00001, 0b111, "th.vamoswapd.v">;
  defm TH_VAMOADDD : TH_VAMO<0b00000, 0b111, "th.vamoaddd.v">;
  defm TH_VAMOXORD : TH_VAMO<0b00100, 0b111, "th.vamoxord.v">;
  defm TH_VAMOANDD : TH_VAMO<0b01100, 0b111, "th.vamoandd.v">;
  defm TH_VAMOORD : TH_VAMO<0b01000, 0b111, "th.vamoord.v">;
  defm TH_VAMOMIND : TH_VAMO<0b10000, 0b111, "th.vamomind.v">;
  defm TH_VAMOMAXD : TH_VAMO<0b10100, 0b111, "th.vamomaxd.v">;
  defm TH_VAMOMINUD : TH_VAMO<0b11000, 0b111, "th.vamominud.v">;
  defm TH_VAMOMAXUD : TH_VAMO<0b11100, 0b111, "th.vamomaxud.v">;

  defm TH_VAMOSWAPQ : TH_VAMO<0b00001, 0b000, "th.vamoswapq.v">;
  defm TH_VAMOADDQ : TH_VAMO<0b00000, 0b000, "th.vamoaddq.v">;
  defm TH_VAMOXORQ : TH_VAMO<0b00100, 0b000, "th.vamoxorq.v">;
  defm TH_VAMOANDQ : TH_VAMO<0b01100, 0b000, "th.vamoandq.v">;
  defm TH_VAMOORQ : TH_VAMO<0b01000, 0b000, "th.vamoorq.v">;
  defm TH_VAMOMINQ : TH_VAMO<0b10000, 0b000, "th.vamominq.v">;
  defm TH_VAMOMAXQ : TH_VAMO<0b10100, 0b000, "th.vamomaxq.v">;
  defm TH_VAMOMINUQ : TH_VAMO<0b11000, 0b000, "th.vamominuq.v">;
  defm TH_VAMOMAXUQ : TH_VAMO<0b11100, 0b000, "th.vamomaxuq.v">;
} // Predicates = [HasVendorXTHeadV, HasVendorXTHeadZvamo, HasStdExtA]

let Predicates = [HasVendorXTHeadV] in {
// Vector Single-Width Integer Add and Subtract
defm TH_VADD_V : VALU_IV_V_X_I<"th.vadd", 0b000000>;
defm TH_VSUB_V : VALU_IV_V_X<"th.vsub", 0b000010>;
defm TH_VRSUB_V : VALU_IV_X_I<"th.vrsub", 0b000011>;

// Vector Widening Integer Add/Subtract
// Refer to 11.2 Widening Vector Arithmetic Instructions
// The destination vector register group cannot overlap a source vector
// register group of a different element width (including the mask register
// if masked), otherwise an illegal instruction exception is raised.
let Constraints = "@earlyclobber $vd" in {
let RVVConstraint = WidenV in {
defm TH_VWADDU_V : VALU_MV_V_X<"th.vwaddu", 0b110000, "v">;
defm TH_VWSUBU_V : VALU_MV_V_X<"th.vwsubu", 0b110010, "v">;
defm TH_VWADD_V : VALU_MV_V_X<"th.vwadd", 0b110001, "v">;
defm TH_VWSUB_V : VALU_MV_V_X<"th.vwsub", 0b110011, "v">;
} // RVVConstraint = WidenV
// Set earlyclobber for following instructions for second and mask operands.
// This has the downside that the earlyclobber constraint is too coarse and
// will impose unnecessary restrictions by not allowing the destination to
// overlap with the first (wide) operand.
let RVVConstraint = WidenW in {
defm TH_VWADDU_W : VALU_MV_V_X<"th.vwaddu", 0b110100, "w">;
defm TH_VWSUBU_W : VALU_MV_V_X<"th.vwsubu", 0b110110, "w">;
defm TH_VWADD_W : VALU_MV_V_X<"th.vwadd", 0b110101, "w">;
defm TH_VWSUB_W : VALU_MV_V_X<"th.vwsub", 0b110111, "w">;
} // RVVConstraint = WidenW
} // Constraints = "@earlyclobber $vd"

// Vector Integer Add-with-Carry / Subtract-with-Borrow Instructions
defm TH_VADC_V : TH_VALUm_IV_V_X_I<"th.vadc", 0b010000>;
let Constraints = "@earlyclobber $vd", RVVConstraint = NoConstraint in {
defm TH_VMADC_V : TH_VALUm_IV_V_X_I<"th.vmadc", 0b010001>;
} // Constraints = "@earlyclobber $vd", RVVConstraint = NoConstraint
defm TH_VSBC_V : TH_VALUm_IV_V_X<"th.vsbc", 0b010010>;
let Constraints = "@earlyclobber $vd", RVVConstraint = NoConstraint in {
defm TH_VMSBC_V : TH_VALUm_IV_V_X<"th.vmsbc", 0b010011>;
} // Constraints = "@earlyclobber $vd", RVVConstraint = NoConstraint

// Vector Bitwise Logical Instructions
defm TH_VAND_V : VALU_IV_V_X_I<"th.vand", 0b001001>;
defm TH_VOR_V : VALU_IV_V_X_I<"th.vor", 0b001010>;
defm TH_VXOR_V : VALU_IV_V_X_I<"th.vxor", 0b001011>;

// Vector Single-Width Bit Shift Instructions
defm TH_VSLL_V : VSHT_IV_V_X_I<"th.vsll", 0b100101>;
defm TH_VSRL_V : VSHT_IV_V_X_I<"th.vsrl", 0b101000>;
defm TH_VSRA_V : VSHT_IV_V_X_I<"th.vsra", 0b101001>;

// Vector Narrowing Integer Right Shift Instructions
// Refer to 11.3. Narrowing Vector Arithmetic Instructions
// The destination vector register group cannot overlap the first source
// vector register group (specified by vs2). The destination vector register
// group cannot overlap the mask register if used, unless LMUL=1.
let Constraints = "@earlyclobber $vd" in {
defm TH_VNSRL_W : TH_VNSHT_IV_V_X_I<"th.vnsrl", 0b101100>;
defm TH_VNSRA_W : TH_VNSHT_IV_V_X_I<"th.vnsra", 0b101101>;
} // Constraints = "@earlyclobber $vd"

// Vector Integer Comparison Instructions
let RVVConstraint = NoConstraint in {
defm TH_VMSEQ_V : VCMP_IV_V_X_I<"th.vmseq", 0b011000>;
defm TH_VMSNE_V : VCMP_IV_V_X_I<"th.vmsne", 0b011001>;
defm TH_VMSLTU_V : VCMP_IV_V_X<"th.vmsltu", 0b011010>;
defm TH_VMSLT_V : VCMP_IV_V_X<"th.vmslt", 0b011011>;
defm TH_VMSLEU_V : VCMP_IV_V_X_I<"th.vmsleu", 0b011100>;
defm TH_VMSLE_V : VCMP_IV_V_X_I<"th.vmsle", 0b011101>;
defm TH_VMSGTU_V : VCMP_IV_X_I<"th.vmsgtu", 0b011110>;
defm TH_VMSGT_V : VCMP_IV_X_I<"th.vmsgt", 0b011111>;
} // RVVConstraint = NoConstraint

// Vector Integer Min/Max Instructions
defm TH_VMINU_V : VCMP_IV_V_X<"th.vminu", 0b000100>;
defm TH_VMIN_V : VCMP_IV_V_X<"th.vmin", 0b000101>;
defm TH_VMAXU_V : VCMP_IV_V_X<"th.vmaxu", 0b000110>;
defm TH_VMAX_V : VCMP_IV_V_X<"th.vmax", 0b000111>;

// Vector Single-Width Integer Multiply Instructions
defm TH_VMUL_V : VMUL_MV_V_X<"th.vmul", 0b100101>;
defm TH_VMULH_V : VMUL_MV_V_X<"th.vmulh", 0b100111>;
defm TH_VMULHU_V : VMUL_MV_V_X<"th.vmulhu", 0b100100>;
defm TH_VMULHSU_V : VMUL_MV_V_X<"th.vmulhsu", 0b100110>;

// Vector Integer Divide Instructions
defm TH_VDIVU_V : VDIV_MV_V_X<"th.vdivu", 0b100000>;
defm TH_VDIV_V : VDIV_MV_V_X<"th.vdiv", 0b100001>;
defm TH_VREMU_V : VDIV_MV_V_X<"th.vremu", 0b100010>;
defm TH_VREM_V : VDIV_MV_V_X<"th.vrem", 0b100011>;

// Vector Widening Integer Multiply Instructions
let Constraints = "@earlyclobber $vd", RVVConstraint = WidenV in {
defm TH_VWMUL_V : VWMUL_MV_V_X<"th.vwmul", 0b111011>;
defm TH_VWMULU_V : VWMUL_MV_V_X<"th.vwmulu", 0b111000>;
defm TH_VWMULSU_V : VWMUL_MV_V_X<"th.vwmulsu", 0b111010>;
} // Constraints = "@earlyclobber $vd", RVVConstraint = WidenV

// Vector Single-Width Integer Multiply-Add Instructions
defm TH_VMACC_V : VMAC_MV_V_X<"th.vmacc", 0b101101>;
defm TH_VNMSAC_V : VMAC_MV_V_X<"th.vnmsac", 0b101111>;
defm TH_VMADD_V : VMAC_MV_V_X<"th.vmadd", 0b101001>;
defm TH_VNMSUB_V : VMAC_MV_V_X<"th.vnmsub", 0b101011>;

// Vector Widening Integer Multiply-Add Instructions
let Constraints = "@earlyclobber $vd", RVVConstraint = WidenV in {
defm TH_VWMACCU_V : VWMAC_MV_V_X<"th.vwmaccu", 0b111100>;
defm TH_VWMACC_V : VWMAC_MV_V_X<"th.vwmacc", 0b111101>;
defm TH_VWMACCSU_V : VWMAC_MV_V_X<"th.vwmaccsu", 0b111110>;
defm TH_VWMACCUS_V : VWMAC_MV_X<"th.vwmaccus", 0b111111>;
} // Constraints = "@earlyclobber $vd", RVVConstraint = WidenV

// Vector Integer Merge Instructions
defm TH_VMERGE_V : VMRG_IV_V_X_I<"th.vmerge", 0b010111>;

// Vector Integer Move Instructions
let hasSideEffects = 0, mayLoad = 0, mayStore = 0, vs2 = 0, vm = 1,
    RVVConstraint = NoConstraint  in {
// op vd, vs1
def TH_VMV_V_V : RVInstVV<0b010111, OPIVV, (outs VR:$vd),
                       (ins VR:$vs1), "th.vmv.v.v", "$vd, $vs1">,
                 Sched<[WriteVIMovV_WorstCase, ReadVIMovV_WorstCase]>;
// op vd, rs1
def TH_VMV_V_X : RVInstVX<0b010111, OPIVX, (outs VR:$vd),
                       (ins GPR:$rs1), "th.vmv.v.x", "$vd, $rs1">,
                 Sched<[WriteVIMovX_WorstCase, ReadVIMovX_WorstCase]>;
// op vd, imm
def TH_VMV_V_I : RVInstIVI<0b010111, (outs VR:$vd),
                       (ins simm5:$imm), "th.vmv.v.i", "$vd, $imm">,
                 Sched<[WriteVIMovI_WorstCase]>;
} // hasSideEffects = 0, mayLoad = 0, mayStore = 0

// Vector Fixed-Point Arithmetic Instructions
defm TH_VSADDU_V : VSALU_IV_V_X_I<"th.vsaddu", 0b100000>;
defm TH_VSADD_V : VSALU_IV_V_X_I<"th.vsadd", 0b100001>;
defm TH_VSSUBU_V : VSALU_IV_V_X<"th.vssubu", 0b100010>;
defm TH_VSSUB_V : VSALU_IV_V_X<"th.vssub", 0b100011>;

// Vector Single-Width Averaging Add and Subtract
defm TH_VAADD_V : VSALU_IV_V_X_I<"th.vaadd", 0b100100>;
defm TH_VASUB_V : VSALU_IV_V_X<"th.vasub", 0b100110>;

// Vector Single-Width Fractional Multiply with Rounding and Saturation
defm TH_VSMUL_V : VSMUL_IV_V_X<"th.vsmul", 0b100111>;

// Vector Widening Saturating Scaled Multiply-Add
let Constraints = "@earlyclobber $vd", RVVConstraint = WidenV in {
defm TH_VWSMACCU_V : TH_VWSMAC_V_X<"th.vwsmaccu", 0b111100>;
defm TH_VWSMACC_V : TH_VWSMAC_V_X<"th.vwsmacc", 0b111101>;
defm TH_VWSMACCSU_V : TH_VWSMAC_V_X<"th.vwsmaccsu", 0b111110>;
defm TH_VWSMACCUS_V : TH_VWSMAC_X<"th.vwsmaccus", 0b111111>;
} // Constraints = "@earlyclobber $vd", RVVConstraint = WidenV

// Vector Single-Width Scaling Shift Instructions
defm TH_VSSRL_V : VSSHF_IV_V_X_I<"th.vssrl", 0b101010>;
defm TH_VSSRA_V : VSSHF_IV_V_X_I<"th.vssra", 0b101011>;

// Vector Narrowing Fixed-Point Clip Instructions
let Constraints = "@earlyclobber $vd" in {
defm TH_VNCLIPU_W : TH_VNCLP_IV_V_X_I<"th.vnclipu", 0b101110>;
defm TH_VNCLIP_W : TH_VNCLP_IV_V_X_I<"th.vnclip", 0b101111>;
} // Constraints = "@earlyclobber $vd"
} // Predicates = [HasVendorXTHeadV]

let Predicates = [HasVendorXTHeadV, HasStdExtF] in {
// Vector Single-Width Floating-Point Add/Subtract Instructions
let Uses = [FRM], mayRaiseFPException = true in {
defm TH_VFADD_V : VALU_FV_V_F<"th.vfadd", 0b000000>;
defm TH_VFSUB_V : VALU_FV_V_F<"th.vfsub", 0b000010>;
defm TH_VFRSUB_V : VALU_FV_F<"th.vfrsub", 0b100111>;
}

// Vector Widening Floating-Point Add/Subtract Instructions
let Constraints = "@earlyclobber $vd",
    Uses = [FRM],
    mayRaiseFPException = true in {
let RVVConstraint = WidenV in {
defm TH_VFWADD_V : VWALU_FV_V_F<"th.vfwadd", 0b110000, "v">;
defm TH_VFWSUB_V : VWALU_FV_V_F<"th.vfwsub", 0b110010, "v">;
} // RVVConstraint = WidenV
// Set earlyclobber for following instructions for second and mask operands.
// This has the downside that the earlyclobber constraint is too coarse and
// will impose unnecessary restrictions by not allowing the destination to
// overlap with the first (wide) operand.
let RVVConstraint = WidenW in {
defm TH_VFWADD_W : VWALU_FV_V_F<"th.vfwadd", 0b110100, "w">;
defm TH_VFWSUB_W : VWALU_FV_V_F<"th.vfwsub", 0b110110, "w">;
} // RVVConstraint = WidenW
} // Constraints = "@earlyclobber $vd", Uses = [FRM], mayRaiseFPException = true

// Vector Single-Width Floating-Point Multiply/Divide Instructions
let Uses = [FRM], mayRaiseFPException = true in {
defm TH_VFMUL_V : VMUL_FV_V_F<"th.vfmul", 0b100100>;
defm TH_VFDIV_V : VDIV_FV_V_F<"th.vfdiv", 0b100000>;
defm TH_VFRDIV_V : VDIV_FV_F<"th.vfrdiv", 0b100001>;
}

// Vector Widening Floating-Point Multiply
let Constraints = "@earlyclobber $vd", RVVConstraint = WidenV,
    Uses = [FRM], mayRaiseFPException = true in {
defm TH_VFWMUL_V : VWMUL_FV_V_F<"th.vfwmul", 0b111000>;
} // Constraints = "@earlyclobber $vd", RVVConstraint = WidenV, Uses = [FRM], mayRaiseFPException = true

// Vector Single-Width Floating-Point Fused Multiply-Add Instructions
let Uses = [FRM], mayRaiseFPException = true in {
defm TH_VFMACC_V : VMAC_FV_V_F<"th.vfmacc", 0b101100>;
defm TH_VFNMACC_V : VMAC_FV_V_F<"th.vfnmacc", 0b101101>;
defm TH_VFMSAC_V : VMAC_FV_V_F<"th.vfmsac", 0b101110>;
defm TH_VFNMSAC_V : VMAC_FV_V_F<"th.vfnmsac", 0b101111>;
defm TH_VFMADD_V : VMAC_FV_V_F<"th.vfmadd", 0b101000>;
defm TH_VFNMADD_V : VMAC_FV_V_F<"th.vfnmadd", 0b101001>;
defm TH_VFMSUB_V : VMAC_FV_V_F<"th.vfmsub", 0b101010>;
defm TH_VFNMSUB_V : VMAC_FV_V_F<"th.vfnmsub", 0b101011>;
}

// Vector Widening Floating-Point Fused Multiply-Add Instructions
let Constraints = "@earlyclobber $vd", RVVConstraint = WidenV,
    Uses = [FRM], mayRaiseFPException = true in {
defm TH_VFWMACC_V : VWMAC_FV_V_F<"th.vfwmacc", 0b111100>;
defm TH_VFWNMACC_V : VWMAC_FV_V_F<"th.vfwnmacc", 0b111101>;
defm TH_VFWMSAC_V : VWMAC_FV_V_F<"th.vfwmsac", 0b111110>;
defm TH_VFWNMSAC_V : VWMAC_FV_V_F<"th.vfwnmsac", 0b111111>;
} // Constraints = "@earlyclobber $vd", RVVConstraint = WidenV, Uses = [FRM], mayRaiseFPException = true

// Vector Floating-Point Square-Root Instruction
let Uses = [FRM], mayRaiseFPException = true in {
defm TH_VFSQRT_V : VSQR_FV_VS2<"th.vfsqrt.v", 0b100011, 0b00000>;
}

// Vector Floating-Point MIN/MAX Instructions
let mayRaiseFPException = true in {
defm TH_VFMIN_V : VCMP_FV_V_F<"th.vfmin", 0b000100>;
defm TH_VFMAX_V : VCMP_FV_V_F<"th.vfmax", 0b000110>;
}

// Vector Floating-Point Sign-Injection Instructions
defm TH_VFSGNJ_V : VSGNJ_FV_V_F<"th.vfsgnj", 0b001000>;
defm TH_VFSGNJN_V : VSGNJ_FV_V_F<"th.vfsgnjn", 0b001001>;
defm TH_VFSGNJX_V : VSGNJ_FV_V_F<"th.vfsgnjx", 0b001010>;

// Vector Floating-Point Compare Instructions
let RVVConstraint = NoConstraint, mayRaiseFPException = true in {
defm TH_VMFEQ_V : VCMP_FV_V_F<"th.vmfeq", 0b011000>;
defm TH_VMFNE_V : VCMP_FV_V_F<"th.vmfne", 0b011100>;
defm TH_VMFLT_V : VCMP_FV_V_F<"th.vmflt", 0b011011>;
defm TH_VMFLE_V : VCMP_FV_V_F<"th.vmfle", 0b011001>;
defm TH_VMFGT_V : VCMP_FV_F<"th.vmfgt", 0b011101>;
defm TH_VMFGE_V : VCMP_FV_F<"th.vmfge", 0b011111>;
defm TH_VMFORD_V : VCMP_FV_V_F<"th.vmford", 0b011010>;
} // RVVConstraint = NoConstraint, mayRaiseFPException = true

// Vector Floating-Point Classify Instruction
defm TH_VFCLASS_V : VCLS_FV_VS2<"th.vfclass.v", 0b100011, 0b10000>;

let hasSideEffects = 0, mayLoad = 0, mayStore = 0 in {

// Vector Floating-Point Merge Instruction
let vm = 0 in
def TH_VFMERGE_VFM : RVInstVX<0b010111, OPFVF, (outs VR:$vd),
                              (ins VR:$vs2, FPR32:$rs1, VMV0:$v0),
                              "th.vfmerge.vfm", "$vd, $vs2, $rs1, v0">,
                     Sched<[WriteVFMergeV_WorstCase, ReadVFMergeV_WorstCase,
                            ReadVFMergeF_WorstCase, ReadVMask]>;

// Vector Floating-Point Move Instruction
let RVVConstraint = NoConstraint in
let vm = 1, vs2 = 0 in
def TH_VFMV_V_F : RVInstVX<0b010111, OPFVF, (outs VR:$vd),
                           (ins FPR32:$rs1), "th.vfmv.v.f", "$vd, $rs1">,
                  Sched<[WriteVFMovV_WorstCase, ReadVFMovF_WorstCase]>;

} // hasSideEffects = 0, mayLoad = 0, mayStore = 0

// Single-Width Floating-Point/Integer Type-Convert Instructions
let mayRaiseFPException = true in {
let Uses = [FRM] in {
defm TH_VFCVT_XU_F_V : VCVTI_FV_VS2<"th.vfcvt.xu.f.v", 0b100010, 0b00000>;
defm TH_VFCVT_X_F_V : VCVTI_FV_VS2<"th.vfcvt.x.f.v", 0b100010, 0b00001>;
defm TH_VFCVT_F_XU_V : VCVTF_IV_VS2<"th.vfcvt.f.xu.v", 0b100010, 0b00010>;
defm TH_VFCVT_F_X_V : VCVTF_IV_VS2<"th.vfcvt.f.x.v", 0b100010, 0b00011>;
}
} // mayRaiseFPException = true

// Widening Floating-Point/Integer Type-Convert Instructions
let Constraints = "@earlyclobber $vd", RVVConstraint = WidenCvt,
    mayRaiseFPException = true in {
let Uses = [FRM] in {
defm TH_VFWCVT_XU_F_V : VWCVTI_FV_VS2<"th.vfwcvt.xu.f.v", 0b100010, 0b01000>;
defm TH_VFWCVT_X_F_V : VWCVTI_FV_VS2<"th.vfwcvt.x.f.v", 0b100010, 0b01001>;
}
defm TH_VFWCVT_F_XU_V : VWCVTF_IV_VS2<"th.vfwcvt.f.xu.v", 0b100010, 0b01010>;
defm TH_VFWCVT_F_X_V : VWCVTF_IV_VS2<"th.vfwcvt.f.x.v", 0b100010, 0b01011>;
defm TH_VFWCVT_F_F_V : VWCVTF_FV_VS2<"th.vfwcvt.f.f.v", 0b100010, 0b01100>;
} // Constraints = "@earlyclobber $vd", RVVConstraint = WidenCvt

// Narrowing Floating-Point/Integer Type-Convert Instructions
let Constraints = "@earlyclobber $vd", mayRaiseFPException = true in {
let Uses = [FRM] in {
defm TH_VFNCVT_XU_F_W : VNCVTI_FV_VS2<"th.vfncvt.xu.f.v", 0b100010, 0b10000>;
defm TH_VFNCVT_X_F_W : VNCVTI_FV_VS2<"th.vfncvt.x.f.v", 0b100010, 0b10001>;
defm TH_VFNCVT_F_XU_W : VNCVTF_IV_VS2<"th.vfncvt.f.xu.v", 0b100010, 0b10010>;
defm TH_VFNCVT_F_X_W : VNCVTF_IV_VS2<"th.vfncvt.f.x.v", 0b100010, 0b10011>;
defm TH_VFNCVT_F_F_W : VNCVTF_FV_VS2<"th.vfncvt.f.f.v", 0b100010, 0b10100>;
}
} // Constraints = "@earlyclobber $vd", mayRaiseFPException = true
} // Predicates = [HasVendorXTHeadV, HasStdExtF]

let Predicates = [HasVendorXTHeadV] in {
// Vector Single-Width Integer Reduction Instructions
let RVVConstraint = NoConstraint in {
defm TH_VREDSUM : VRED_MV_V<"th.vredsum", 0b000000>;
defm TH_VREDMAXU : VRED_MV_V<"th.vredmaxu", 0b000110>;
defm TH_VREDMAX : VRED_MV_V<"th.vredmax", 0b000111>;
defm TH_VREDMINU : VRED_MV_V<"th.vredminu", 0b000100>;
defm TH_VREDMIN : VRED_MV_V<"th.vredmin", 0b000101>;
defm TH_VREDAND : VRED_MV_V<"th.vredand", 0b000001>;
defm TH_VREDOR : VRED_MV_V<"th.vredor", 0b000010>;
defm TH_VREDXOR : VRED_MV_V<"th.vredxor", 0b000011>;
} // RVVConstraint = NoConstraint

// Vector Widening Integer Reduction Instructions
let Constraints = "@earlyclobber $vd", RVVConstraint = NoConstraint in {
// Set earlyclobber for following instructions for second and mask operands.
// This has the downside that the earlyclobber constraint is too coarse and
// will impose unnecessary restrictions by not allowing the destination to
// overlap with the first (wide) operand.
defm TH_VWREDSUMU : VWRED_IV_V<"th.vwredsumu", 0b110000>;
defm TH_VWREDSUM : VWRED_IV_V<"th.vwredsum", 0b110001>;
} // Constraints = "@earlyclobber $vd", RVVConstraint = NoConstraint

} // Predicates = [HasVendorXTHeadV]

let Predicates = [HasVendorXTHeadV, HasStdExtF] in {
// Vector Single-Width Floating-Point Reduction Instructions
let RVVConstraint = NoConstraint in {
let Uses = [FRM], mayRaiseFPException = true in {
defm TH_VFREDOSUM : VREDO_FV_V<"th.vfredosum", 0b000011>;
defm TH_VFREDUSUM : VRED_FV_V<"th.vfredsum", 0b000001>;
}
let mayRaiseFPException = true in {
defm TH_VFREDMAX : VRED_FV_V<"th.vfredmax", 0b000111>;
defm TH_VFREDMIN : VRED_FV_V<"th.vfredmin", 0b000101>;
}
} // RVVConstraint = NoConstraint

// Vector Widening Floating-Point Reduction Instructions
let Constraints = "@earlyclobber $vd", RVVConstraint = NoConstraint in {
// Set earlyclobber for following instructions for second and mask operands.
// This has the downside that the earlyclobber constraint is too coarse and
// will impose unnecessary restrictions by not allowing the destination to
// overlap with the first (wide) operand.
let Uses = [FRM], mayRaiseFPException = true in {
defm TH_VFWREDOSUM : VWREDO_FV_V<"th.vfwredosum", 0b110011>;
defm TH_VFWREDUSUM : VWRED_FV_V<"th.vfwredsum", 0b110001>;
}
} // Constraints = "@earlyclobber $vd", RVVConstraint = NoConstraint
} // Predicates = [HasVendorXTHeadV, HasStdExtF]

let Predicates = [HasVendorXTHeadV] in {
// Vector Mask-Register Logical Instructions
let RVVConstraint = NoConstraint in {
defm TH_VMAND_M : VMALU_MV_Mask<"th.vmand", 0b011001, "m">;
defm TH_VMNAND_M : VMALU_MV_Mask<"th.vmnand", 0b011101, "m">;
defm TH_VMANDN_M : VMALU_MV_Mask<"th.vmandnot", 0b011000, "m">;
defm TH_VMXOR_M : VMALU_MV_Mask<"th.vmxor", 0b011011, "m">;
defm TH_VMOR_M : VMALU_MV_Mask<"th.vmor", 0b011010, "m">;
defm TH_VMNOR_M : VMALU_MV_Mask<"th.vmnor", 0b011110, "m">;
defm TH_VMORN_M : VMALU_MV_Mask<"th.vmornot", 0b011100, "m">;
defm TH_VMXNOR_M : VMALU_MV_Mask<"th.vmxnor", 0b011111, "m">;
}

let hasSideEffects = 0, mayLoad = 0, mayStore = 0,
    RVVConstraint = NoConstraint  in {

// Vector mask population count vmpopc
def TH_VMPOPC_M : RVInstV<0b010100, 0b00000, OPMVV, (outs GPR:$vd),
                          (ins VR:$vs2, VMaskOp:$vm),
                          "th.vmpopc.m", "$vd, $vs2$vm">,
                  Sched<[WriteVMPopV_WorstCase, ReadVMPopV_WorstCase,
                         ReadVMask]>;

// vmfirst find-first-set mask bit
def TH_VMFIRST_M : RVInstV<0b010101, 0b00000, OPMVV, (outs GPR:$vd),
                           (ins VR:$vs2, VMaskOp:$vm),
                           "th.vmfirst.m", "$vd, $vs2$vm">,
                   Sched<[WriteVMFFSV_WorstCase, ReadVMFFSV_WorstCase,
                          ReadVMask]>;

} // hasSideEffects = 0, mayLoad = 0, mayStore = 0

let Constraints = "@earlyclobber $vd", RVVConstraint = Iota in {

// vmsbf.m set-before-first mask bit
defm TH_VMSBF_M : VMSFS_MV_V<"th.vmsbf.m", 0b010110, 0b00001>;
// vmsif.m set-including-first mask bit
defm TH_VMSIF_M : VMSFS_MV_V<"th.vmsif.m", 0b010110, 0b00011>;
// vmsof.m set-only-first mask bit
defm TH_VMSOF_M : VMSFS_MV_V<"th.vmsof.m", 0b010110, 0b00010>;
// Vector Iota Instruction
defm TH_VIOTA_M : VIOTA_MV_V<"th.viota.m", 0b010110, 0b10000>;

} // Constraints = "@earlyclobber $vd", RVVConstraint = Iota

let hasSideEffects = 0, mayLoad = 0, mayStore = 0 in {
// Vector Element Index Instruction
let vs2 = 0 in
def TH_VID_V : RVInstV<0b010110, 0b10001, OPMVV, (outs VR:$vd),
                       (ins VMaskOp:$vm),
                       "th.vid.v", "$vd$vm">,
               SchedNullaryMC<"WriteVIdxV">;

let vm = 1, RVVConstraint = NoConstraint in {
// Integer Extract Instruction
def TH_VEXT_X_V : RVInstVX<0b001100, OPMVV, (outs GPR:$vd),
                           (ins VR:$vs2, GPR:$rs1),
                           "th.vext.x.v", "$vd, $vs2, $rs1">,
                  Sched<[WriteVIMovVX, ReadVIMovVX, ReadVIMovXX]>;

// Integer Scalar Move Instruction
let Constraints = "$vd = $vd_wb" in
def TH_VMV_S_X : RVInstV2<0b001101, 0b00000, OPMVX, (outs VR:$vd_wb),
                          (ins VR:$vd, GPR:$rs1),
                          "th.vmv.s.x", "$vd, $rs1">,
                 Sched<[WriteVIMovXV, ReadVIMovXV,  ReadVIMovXX]>;
}
} // hasSideEffects = 0, mayLoad = 0, mayStore = 0
} // Predicates = [HasVendorXTHeadV]

let Predicates = [HasVendorXTHeadV, HasStdExtF] in {
let hasSideEffects = 0, mayLoad = 0, mayStore = 0, vm = 1,
    RVVConstraint = NoConstraint  in {
// Floating-Point Scalar Move Instructions
def TH_VFMV_F_S : RVInstV<0b001100, 0b00000, OPFVV, (outs FPR32:$vd),
                          (ins VR:$vs2),
                          "th.vfmv.f.s", "$vd, $vs2">,
                  Sched<[WriteVFMovVF, ReadVFMovVF]>;
let Constraints = "$vd = $vd_wb" in
def TH_VFMV_S_F : RVInstV2<0b001101, 0b00000, OPFVF, (outs VR:$vd_wb),
                           (ins VR:$vd, FPR32:$rs1),
                           "th.vfmv.s.f", "$vd, $rs1">,
                  Sched<[WriteVFMovFV, ReadVFMovFV, ReadVFMovFX]>;

} // hasSideEffects = 0, mayLoad = 0, mayStore = 0, vm = 1
} // Predicates = [HasVendorXTHeadV, HasStdExtF]

let Predicates = [HasVendorXTHeadV] in  {
// Vector Slide Instructions
let Constraints = "@earlyclobber $vd", RVVConstraint = SlideUp in {
defm TH_VSLIDEUP_V : VSLD_IV_X_I<"th.vslideup", 0b001110>;
defm TH_VSLIDE1UP_V : VSLD1_MV_X<"th.vslide1up", 0b001110>;
} // Constraints = "@earlyclobber $vd", RVVConstraint = SlideUp
defm TH_VSLIDEDOWN_V : VSLD_IV_X_I<"th.vslidedown", 0b001111>;
defm TH_VSLIDE1DOWN_V : VSLD1_MV_X<"th.vslide1down", 0b001111>;

// Vector Register Gather Instruction
let Constraints = "@earlyclobber $vd", RVVConstraint = Vrgather in {
defm TH_VRGATHER_V : VGTR_IV_V_X_I<"th.vrgather", 0b001100>;
} // Constraints = "@earlyclobber $vd", RVVConstraint = Vrgather

// Vector Compress Instruction
let Constraints = "@earlyclobber $vd", RVVConstraint = Vcompress in {
defm TH_VCOMPRESS_V : VCPR_MV_Mask<"th.vcompress", 0b010111>;
} // Constraints = "@earlyclobber $vd", RVVConstraint = Vcompress
} // Predicates = [HasVendorXTHeadV]
} // DecoderNamespace = "RVV0p71"

// Pseudo instructions
let Predicates = [HasVendorXTHeadV] in {
// Vector Integer Comparison Instructions
def : InstAlias<"th.vmsgtu.vv $vd, $va, $vb$vm",
                (TH_VMSLTU_VV VR:$vd, VR:$vb, VR:$va, VMaskOp:$vm), 0>;
def : InstAlias<"th.vmsgt.vv $vd, $va, $vb$vm",
                (TH_VMSLT_VV VR:$vd, VR:$vb, VR:$va, VMaskOp:$vm), 0>;
def : InstAlias<"th.vmsgeu.vv $vd, $va, $vb$vm",
                (TH_VMSLEU_VV VR:$vd, VR:$vb, VR:$va, VMaskOp:$vm), 0>;
def : InstAlias<"th.vmsge.vv $vd, $va, $vb$vm",
                (TH_VMSLE_VV VR:$vd, VR:$vb, VR:$va, VMaskOp:$vm), 0>;

// These pseudos need to be handled in RISCVAsmParser::processInstruction
let isCodeGenOnly = 0, isAsmParserOnly = 1, hasSideEffects = 0, mayLoad = 0,
    mayStore = 0 in {
// For unsigned comparisons we need to special case 0 immediate to maintain
// the always true/false semantics we would invert if we just decremented the
// immediate like we do for signed. To match the GNU assembler we will use
// vmseq/vmsne.vv with the same register for both operands which we can't do
// from an InstAlias.
def PseudoTH_VMSGEU_VI : Pseudo<(outs VR:$vd),
                                (ins VR:$vs2, simm5_plus1:$imm, VMaskOp:$vm),
                                [], "th.vmsgeu.vi", "$vd, $vs2, $imm$vm">;
def PseudoTH_VMSLTU_VI : Pseudo<(outs VR:$vd),
                                (ins VR:$vs2, simm5_plus1:$imm, VMaskOp:$vm),
                                [], "th.vmsltu.vi", "$vd, $vs2, $imm$vm">;
// Handle signed with pseudos as well for more consistency in the
// implementation.
def PseudoTH_VMSGE_VI : Pseudo<(outs VR:$vd),
                               (ins VR:$vs2, simm5_plus1:$imm, VMaskOp:$vm),
                               [], "th.vmsge.vi", "$vd, $vs2, $imm$vm">;
def PseudoTH_VMSLT_VI : Pseudo<(outs VR:$vd),
                               (ins VR:$vs2, simm5_plus1:$imm, VMaskOp:$vm),
                               [], "th.vmslt.vi", "$vd, $vs2, $imm$vm">;
}

// These pseudos need to be handled in RISCVAsmParser::processInstruction
let isCodeGenOnly = 0, isAsmParserOnly = 1, hasSideEffects = 0, mayLoad = 0,
    mayStore = 0 in {
def PseudoTH_VMSGEU_VX : Pseudo<(outs VR:$vd),
                                (ins VR:$vs2, GPR:$rs1),
                                [], "th.vmsgeu.vx", "$vd, $vs2, $rs1">;
def PseudoTH_VMSGE_VX : Pseudo<(outs VR:$vd),
                               (ins VR:$vs2, GPR:$rs1),
                               [], "th.vmsge.vx", "$vd, $vs2, $rs1">;
def PseudoTH_VMSGEU_VX_M : Pseudo<(outs VRNoV0:$vd),
                                  (ins VR:$vs2, GPR:$rs1, VMaskOp:$vm),
                                  [], "th.vmsgeu.vx", "$vd, $vs2, $rs1$vm">;
def PseudoTH_VMSGE_VX_M : Pseudo<(outs VRNoV0:$vd),
                                 (ins VR:$vs2, GPR:$rs1, VMaskOp:$vm),
                                 [], "th.vmsge.vx", "$vd, $vs2, $rs1$vm">;
def PseudoTH_VMSGEU_VX_M_T : Pseudo<(outs VR:$vd, VRNoV0:$scratch),
                                    (ins VR:$vs2, GPR:$rs1, VMaskOp:$vm),
                                    [], "th.vmsgeu.vx", "$vd, $vs2, $rs1$vm, $scratch">;
def PseudoTH_VMSGE_VX_M_T : Pseudo<(outs VR:$vd, VRNoV0:$scratch),
                                   (ins VR:$vs2, GPR:$rs1, VMaskOp:$vm),
                                   [], "th.vmsge.vx", "$vd, $vs2, $rs1$vm, $scratch">;
} // isCodeGenOnly = 0, isAsmParserOnly = 1, hasSideEffects = 0, mayLoad = 0, mayStore = 0


// Vector Floating-Point Compare Instructions
def : InstAlias<"th.vmfgt.vv $vd, $va, $vb$vm",
                (TH_VMFLT_VV VR:$vd, VR:$vb, VR:$va, VMaskOp:$vm), 0>;
def : InstAlias<"th.vmfge.vv $vd, $va, $vb$vm",
                (TH_VMFLE_VV VR:$vd, VR:$vb, VR:$va, VMaskOp:$vm), 0>;

// Vector Bitwise Logical Instructions
def : InstAlias<"th.vnot.v $vd, $vs$vm",
                (TH_VXOR_VI VR:$vd, VR:$vs, -1, VMaskOp:$vm)>;
def : InstAlias<"th.vnot.v $vd, $vs",
                (TH_VXOR_VI VR:$vd, VR:$vs, -1, zero_reg)>;

// Vector Widening Integer Add/Subtract
def : InstAlias<"th.vwcvt.x.x.v $vd, $vs$vm",
                (TH_VWADD_VX VR:$vd, VR:$vs, X0, VMaskOp:$vm)>;
def : InstAlias<"th.vwcvt.x.x.v $vd, $vs",
                (TH_VWADD_VX VR:$vd, VR:$vs, X0, zero_reg)>;
def : InstAlias<"th.vwcvtu.x.x.v $vd, $vs$vm",
                (TH_VWADDU_VX VR:$vd, VR:$vs, X0, VMaskOp:$vm)>;
def : InstAlias<"th.vwcvtu.x.x.v $vd, $vs",
                (TH_VWADDU_VX VR:$vd, VR:$vs, X0, zero_reg)>;

// Integer Extract Instruction
def : InstAlias<"th.vmv.x.s $rd, $vs",
                (TH_VEXT_X_V GPR:$rd, VR:$vs, X0)>;

// Vector Mask-Register Logical Instructions
def : InstAlias<"th.vmcpy.m $vd, $vs",
                (TH_VMAND_MM VR:$vd, VR:$vs, VR:$vs)>;
def : InstAlias<"th.vmclr.m $vd",
                (TH_VMXOR_MM VR:$vd, VR:$vd, VR:$vd)>;
def : InstAlias<"th.vmset.m $vd",
                (TH_VMXNOR_MM VR:$vd, VR:$vd, VR:$vd)>;
def : InstAlias<"th.vmnot.m $vd, $vs",
                (TH_VMNAND_MM VR:$vd, VR:$vs, VR:$vs)>;

} // Predicates = [HasVendorXTHeadV]

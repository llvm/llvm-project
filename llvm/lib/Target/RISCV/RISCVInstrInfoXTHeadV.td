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

class XVLoadStore<bits<3> nf, RISCVOpcode opcode,
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
  class XVLxU<bits<3> nf, bits<3> width, string opcodestr>
      : XVLoadStore<nf, OPC_LOAD_FP, 0b000, width, (outs VR:$rd),
                    (ins GPRMemZeroOffset:$rs1, VMaskOp:$vm),
                    opcodestr, "$rd, ${rs1}$vm"> {
    let rs2 = 0b00000;
  }
  class XVLx<bits<3> nf, bits<3> width, string opcodestr>
      : XVLoadStore<nf, OPC_LOAD_FP, 0b100, width, (outs VR:$rd),
                    (ins GPRMemZeroOffset:$rs1, VMaskOp:$vm),
                    opcodestr, "$rd, ${rs1}$vm"> {
    let rs2 = 0b00000;
  }
  class XVLxUFF<bits<3> nf, bits<3> width, string opcodestr>
      : XVLoadStore<nf, OPC_LOAD_FP, 0b000, width, (outs VR:$rd),
                    (ins GPRMemZeroOffset:$rs1, VMaskOp:$vm),
                    opcodestr, "$rd, ${rs1}$vm"> {
    let rs2 = 0b10000;
  }
  class XVLxFF<bits<3> nf, bits<3> width, string opcodestr>
      : XVLoadStore<nf, OPC_LOAD_FP, 0b100, width, (outs VR:$rd),
                    (ins GPRMemZeroOffset:$rs1, VMaskOp:$vm),
                    opcodestr, "$rd, ${rs1}$vm"> {
    let rs2 = 0b10000;
  }
  class XVLSxU<bits<3> nf, bits<3> width, string opcodestr>
      : XVLoadStore<nf, OPC_LOAD_FP, 0b010, width, (outs VR:$rd),
                    (ins GPRMemZeroOffset:$rs1, GPR:$rs2, VMaskOp:$vm),
                    opcodestr, "$rd, $rs1, $rs2$vm">;
  class XVLSx<bits<3> nf, bits<3> width, string opcodestr>
      : XVLoadStore<nf, OPC_LOAD_FP, 0b110, width, (outs VR:$rd),
                    (ins GPRMemZeroOffset:$rs1, GPR:$rs2, VMaskOp:$vm),
                    opcodestr, "$rd, $rs1, $rs2$vm">;
  class XVLXxU<bits<3> nf, bits<3> width, string opcodestr>
      : XVLoadStore<nf, OPC_LOAD_FP, 0b011, width, (outs VR:$rd),
                    (ins GPRMemZeroOffset:$rs1, VR:$rs2, VMaskOp:$vm),
                    opcodestr, "$rd, $rs1, $rs2$vm">;
  class XVLXx<bits<3> nf, bits<3> width, string opcodestr>
      : XVLoadStore<nf, OPC_LOAD_FP, 0b111, width, (outs VR:$rd),
                    (ins GPRMemZeroOffset:$rs1, VR:$rs2, VMaskOp:$vm),
                    opcodestr, "$rd, $rs1, $rs2$vm">;
}

let hasSideEffects = 0, mayLoad = 0, mayStore = 1 in {
  class XVSx<bits<3> nf, bits<3> width, string opcodestr>
      : XVLoadStore<nf, OPC_STORE_FP, 0b000, width, (outs),
                    (ins VR:$rd, GPRMemZeroOffset:$rs1, VMaskOp:$vm),
                    opcodestr, "$rd, ${rs1}$vm"> {
    let rs2 = 0b00000;
  }
  class XVSSx<bits<3> nf, bits<3> width, string opcodestr>
      : XVLoadStore<nf, OPC_STORE_FP, 0b010, width, (outs),
                    (ins VR:$rd, GPRMemZeroOffset:$rs1, GPR:$rs2, VMaskOp:$vm),
                    opcodestr, "$rd, $rs1, $rs2$vm">;
  class XVSXx<bits<3> nf, bits<3> width, string opcodestr>
      : XVLoadStore<nf, OPC_STORE_FP, 0b011, width, (outs),
                    (ins VR:$rd, GPRMemZeroOffset:$rs1, VR:$rs2, VMaskOp:$vm),
                    opcodestr, "$rd, $rs1, $rs2$vm">;
  class XVSUXx<bits<3> nf, bits<3> width, string opcodestr>
      : XVLoadStore<nf, OPC_STORE_FP, 0b111, width, (outs),
                    (ins VR:$rd, GPRMemZeroOffset:$rs1, VR:$rs2, VMaskOp:$vm),
                    opcodestr, "$rd, $rs1, $rs2$vm">;
}

multiclass VWSMAC_V_X<string opcodestr, bits<6> funct6> {
  def V : VALUrVV<funct6, OPIVV, opcodestr # ".vv">,
          Sched<[WriteVIWMulAddV_WorstCase, ReadVIWMulAddV_WorstCase,
                 ReadVIWMulAddV_WorstCase, ReadVMask]>;
  def X : VALUrVX<funct6, OPIVX, opcodestr # ".vx">,
          Sched<[WriteVIWMulAddX_WorstCase, ReadVIWMulAddV_WorstCase,
                 ReadVIWMulAddX_WorstCase, ReadVMask]>;
}

multiclass VWSMAC_X<string opcodestr, bits<6> funct6> {
  def X : VALUrVX<funct6, OPIVX, opcodestr # ".vx">,
          Sched<[WriteVIWMulAddX_WorstCase, ReadVIWMulAddV_WorstCase,
                 ReadVIWMulAddX_WorstCase, ReadVMask]>;
}

multiclass XVNCLP_IV_V_X_I<string opcodestr, bits<6> funct6> {
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

multiclass XVNSHT_IV_V_X_I<string opcodestr, bits<6> funct6> {
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

//===----------------------------------------------------------------------===//
// Instructions
//===----------------------------------------------------------------------===//

let AsmVariantName = "RVV0p71", DecoderNamespace = "RVV0p71" in {
let Predicates = [HasVendorXTHeadV] in {
// Configuration-Setting Instructions
let hasSideEffects = 1, mayLoad = 0, mayStore = 0 in {
def XVSETVLI : RVInstSetVLi<(outs GPR:$rd), (ins GPR:$rs1, XTHeadVTypeI:$vtypei),
                            "vsetvli", "$rd, $rs1, $vtypei">,
                            Sched<[WriteVSETVLI, ReadVSETVLI]>;
def XVSETVL : RVInstSetVL<(outs GPR:$rd), (ins GPR:$rs1, GPR:$rs2),
                          "vsetvl", "$rd, $rs1, $rs2">,
                          Sched<[WriteVSETVL, ReadVSETVL, ReadVSETVL]>;
} // hasSideEffects = 1, mayLoad = 0, mayStore = 0

// Vector Unit-Stride Instructions
def XVLB_V : XVLx<0b000, 0b000, "vlb.v">;
def XVLH_V : XVLx<0b000, 0b101, "vlh.v">;
def XVLW_V : XVLx<0b000, 0b110, "vlw.v">;
def XVLBU_V : XVLxU<0b000, 0b000, "vlbu.v">;
def XVLHU_V : XVLxU<0b000, 0b101, "vlhu.v">;
def XVLWU_V : XVLxU<0b000, 0b110, "vlwu.v">;
def XVLE_V : XVLxU<0b000, 0b111, "vle.v">;
def XVSB_V : XVSx<0b000, 0b000, "vsb.v">;
def XVSH_V : XVSx<0b000, 0b101, "vsh.v">;
def XVSW_V : XVSx<0b000, 0b110, "vsw.v">;
def XVSE_V : XVSx<0b000, 0b111, "vse.v">;

// Vector Strided Instructions
def XVLSB_V : XVLSx<0b000, 0b000, "vlsb.v">;
def XVLSH_V : XVLSx<0b000, 0b101, "vlsh.v">;
def XVLSW_V : XVLSx<0b000, 0b110, "vlsw.v">;
def XVLSBU_V : XVLSxU<0b000, 0b000, "vlsbu.v">;
def XVLSHU_V : XVLSxU<0b000, 0b101, "vlshu.v">;
def XVLSWU_V : XVLSxU<0b000, 0b110, "vlswu.v">;
def XVLSE_V : XVLSxU<0b000, 0b111, "vlse.v">;
def XVSSB_V : XVSSx<0b000, 0b000, "vssb.v">;
def XVSSH_V : XVSSx<0b000, 0b101, "vssh.v">;
def XVSSW_V : XVSSx<0b000, 0b110, "vssw.v">;
def XVSSE_V : XVSSx<0b000, 0b111, "vsse.v">;

// Vector indexed loads and stores
def XVLXB_V : XVLXx<0b000, 0b000, "vlxb.v">;
def XVLXH_V : XVLXx<0b000, 0b101, "vlxh.v">;
def XVLXW_V : XVLXx<0b000, 0b110, "vlxw.v">;
def XVLXBU_V : XVLXxU<0b000, 0b000, "vlxbu.v">;
def XVLXHU_V : XVLXxU<0b000, 0b101, "vlxhu.v">;
def XVLXWU_V : XVLXxU<0b000, 0b110, "vlxwu.v">;
def XVLXE_V : XVLXxU<0b000, 0b111, "vlxe.v">;
def XVSXB_V : XVSXx<0b000, 0b000, "vsxb.v">;
def XVSXH_V : XVSXx<0b000, 0b101, "vsxh.v">;
def XVSXW_V : XVSXx<0b000, 0b110, "vsxw.v">;
def XVSXE_V : XVSXx<0b000, 0b111, "vsxe.v">;
def XVSUXB_V : XVSUXx<0b000, 0b000, "vsuxb.v">;
def XVSUXH_V : XVSUXx<0b000, 0b101, "vsuxh.v">;
def XVSUXW_V : XVSUXx<0b000, 0b110, "vsuxw.v">;
def XVSUXE_V : XVSUXx<0b000, 0b111, "vsuxe.v">;

// Unit-stride Fault-Only-First Loads
def XVLBFF_V : XVLxFF<0b000, 0b000, "vlbff.v">;
def XVLHFF_V : XVLxFF<0b000, 0b101, "vlhff.v">;
def XVLWFF_V : XVLxFF<0b000, 0b110, "vlwff.v">;
def XVLBUFF_V : XVLxUFF<0b000, 0b000, "vlbuff.v">;
def XVLHUFF_V : XVLxUFF<0b000, 0b101, "vlhuff.v">;
def XVLWUFF_V : XVLxUFF<0b000, 0b110, "vlwuff.v">;
def XVLEFF_V : XVLxUFF<0b000, 0b111, "vleff.v">;
} // Predicates = [HasVendorXTHeadV]

let Predicates = [HasVendorXTHeadV, HasVendorXTHeadVlsseg] in {
foreach nf=2-8 in {
  // Vector Unit-Stride Segment Loads and Stores
  def XVLSEG#nf#B_V : XVLx<!add(nf, -1), 0b000, "vlseg"#nf#"b.v">;
  def XVLSEG#nf#H_V : XVLx<!add(nf, -1), 0b101, "vlseg"#nf#"h.v">;
  def XVLSEG#nf#W_V : XVLx<!add(nf, -1), 0b110, "vlseg"#nf#"w.v">;
  def XVLSEG#nf#BU_V : XVLxU<!add(nf, -1), 0b000, "vlseg"#nf#"bu.v">;
  def XVLSEG#nf#HU_V : XVLxU<!add(nf, -1), 0b101, "vlseg"#nf#"hu.v">;
  def XVLSEG#nf#WU_V : XVLxU<!add(nf, -1), 0b110, "vlseg"#nf#"wu.v">;
  def XVLSEG#nf#E_V : XVLxU<!add(nf, -1), 0b111, "vlseg"#nf#"e.v">;
  def XVSSEG#nf#B_V : XVSx<!add(nf, -1), 0b000, "vsseg"#nf#"b.v">;
  def XVSSEG#nf#H_V : XVSx<!add(nf, -1), 0b101, "vsseg"#nf#"h.v">;
  def XVSSEG#nf#W_V : XVSx<!add(nf, -1), 0b110, "vsseg"#nf#"w.v">;
  def XVSSEG#nf#E_V : XVSx<!add(nf, -1), 0b111, "vsseg"#nf#"e.v">;
  def XVLSEG#nf#BFF_V : XVLxFF<!add(nf, -1), 0b000, "vlseg"#nf#"bff.v">;
  def XVLSEG#nf#HFF_V : XVLxFF<!add(nf, -1), 0b101, "vlseg"#nf#"hff.v">;
  def XVLSEG#nf#WFF_V : XVLxFF<!add(nf, -1), 0b110, "vlseg"#nf#"wff.v">;
  def XVLSEG#nf#BUFF_V : XVLxUFF<!add(nf, -1), 0b000, "vlseg"#nf#"buff.v">;
  def XVLSEG#nf#HUFF_V : XVLxUFF<!add(nf, -1), 0b101, "vlseg"#nf#"huff.v">;
  def XVLSEG#nf#WUFF_V : XVLxUFF<!add(nf, -1), 0b110, "vlseg"#nf#"wuff.v">;
  def XVLSEG#nf#EFF_V : XVLxUFF<!add(nf, -1), 0b111, "vlseg"#nf#"eff.v">;

  // Vector Strided Segment Loads and Stores
  def XVLSSEG#nf#B_V : XVLSx<!add(nf, -1), 0b000, "vlsseg"#nf#"b.v">;
  def XVLSSEG#nf#H_V : XVLSx<!add(nf, -1), 0b101, "vlsseg"#nf#"h.v">;
  def XVLSSEG#nf#W_V : XVLSx<!add(nf, -1), 0b110, "vlsseg"#nf#"w.v">;
  def XVLSSEG#nf#BU_V : XVLSxU<!add(nf, -1), 0b000, "vlsseg"#nf#"bu.v">;
  def XVLSSEG#nf#HU_V : XVLSxU<!add(nf, -1), 0b101, "vlsseg"#nf#"hu.v">;
  def XVLSSEG#nf#WU_V : XVLSxU<!add(nf, -1), 0b110, "vlsseg"#nf#"wu.v">;
  def XVLSSEG#nf#E_V : XVLSxU<!add(nf, -1), 0b111, "vlsseg"#nf#"e.v">;
  def XVSSSEG#nf#B_V : XVSSx<!add(nf, -1), 0b000, "vssseg"#nf#"b.v">;
  def XVSSSEG#nf#H_V : XVSSx<!add(nf, -1), 0b101, "vssseg"#nf#"h.v">;
  def XVSSSEG#nf#W_V : XVSSx<!add(nf, -1), 0b110, "vssseg"#nf#"w.v">;
  def XVSSSEG#nf#E_V : XVSSx<!add(nf, -1), 0b111, "vssseg"#nf#"e.v">;

  // Vector Indexed Segment Loads and Stores
  def XVLXSEG#nf#B_V : XVLXx<!add(nf, -1), 0b000, "vlxseg"#nf#"b.v">;
  def XVLXSEG#nf#H_V : XVLXx<!add(nf, -1), 0b101, "vlxseg"#nf#"h.v">;
  def XVLXSEG#nf#W_V : XVLXx<!add(nf, -1), 0b110, "vlxseg"#nf#"w.v">;
  def XVLXSEG#nf#BU_V : XVLXxU<!add(nf, -1), 0b000, "vlxseg"#nf#"bu.v">;
  def XVLXSEG#nf#HU_V : XVLXxU<!add(nf, -1), 0b101, "vlxseg"#nf#"hu.v">;
  def XVLXSEG#nf#WU_V : XVLXxU<!add(nf, -1), 0b110, "vlxseg"#nf#"wu.v">;
  def XVLXSEG#nf#E_V : XVLXxU<!add(nf, -1), 0b111, "vlxseg"#nf#"e.v">;
  def XVSXSEG#nf#B_V : XVSXx<!add(nf, -1), 0b000, "vsxseg"#nf#"b.v">;
  def XVSXSEG#nf#H_V : XVSXx<!add(nf, -1), 0b101, "vsxseg"#nf#"h.v">;
  def XVSXSEG#nf#W_V : XVSXx<!add(nf, -1), 0b110, "vsxseg"#nf#"w.v">;
  def XVSXSEG#nf#E_V : XVSXx<!add(nf, -1), 0b111, "vsxseg"#nf#"e.v">;
}
} // Predicates = [HasVendorXTHeadV, HasVendorXTHeadVlsseg]

let Predicates = [HasVendorXTHeadV] in {
// Vector Single-Width Integer Add and Subtract
defm XVADD_V : VALU_IV_V_X_I<"vadd", 0b000000>;
defm XVSUB_V : VALU_IV_V_X<"vsub", 0b000010>;
defm XVRSUB_V : VALU_IV_X_I<"vrsub", 0b000011>;

// Vector Widening Integer Add/Subtract
// Refer to 11.2 Widening Vector Arithmetic Instructions
// The destination vector register group cannot overlap a source vector
// register group of a different element width (including the mask register
// if masked), otherwise an illegal instruction exception is raised.
let Constraints = "@earlyclobber $vd" in {
let RVVConstraint = WidenV in {
defm XVWADDU_V : VALU_MV_V_X<"vwaddu", 0b110000, "v">;
defm XVWSUBU_V : VALU_MV_V_X<"vwsubu", 0b110010, "v">;
defm XVWADD_V : VALU_MV_V_X<"vwadd", 0b110001, "v">;
defm XVWSUB_V : VALU_MV_V_X<"vwsub", 0b110011, "v">;
} // RVVConstraint = WidenV
// Set earlyclobber for following instructions for second and mask operands.
// This has the downside that the earlyclobber constraint is too coarse and
// will impose unnecessary restrictions by not allowing the destination to
// overlap with the first (wide) operand.
let RVVConstraint = WidenW in {
defm XVWADDU_W : VALU_MV_V_X<"vwaddu", 0b110100, "w">;
defm XVWSUBU_W : VALU_MV_V_X<"vwsubu", 0b110110, "w">;
defm XVWADD_W : VALU_MV_V_X<"vwadd", 0b110101, "w">;
defm XVWSUB_W : VALU_MV_V_X<"vwsub", 0b110111, "w">;
} // RVVConstraint = WidenW
} // Constraints = "@earlyclobber $vd"

// Vector Integer Add-with-Carry / Subtract-with-Borrow Instructions
defm XVADC_V : VALUm_IV_V_X_I<"vadc", 0b010000>;
let Constraints = "@earlyclobber $vd", RVVConstraint = NoConstraint in {
defm XVMADC_V : VALUm_IV_V_X_I<"vmadc", 0b010001>;
defm XVMADC_V : VALUNoVm_IV_V_X_I<"vmadc", 0b010001>;
} // Constraints = "@earlyclobber $vd", RVVConstraint = NoConstraint
defm XVSBC_V : VALUm_IV_V_X<"vsbc", 0b010010>;
let Constraints = "@earlyclobber $vd", RVVConstraint = NoConstraint in {
defm XVMSBC_V : VALUm_IV_V_X<"vmsbc", 0b010011>;
defm XVMSBC_V : VALUNoVm_IV_V_X<"vmsbc", 0b010011>;
} // Constraints = "@earlyclobber $vd", RVVConstraint = NoConstraint

// Vector Bitwise Logical Instructions
defm XVAND_V : VALU_IV_V_X_I<"vand", 0b001001>;
defm XVOR_V : VALU_IV_V_X_I<"vor", 0b001010>;
defm XVXOR_V : VALU_IV_V_X_I<"vxor", 0b001011>;

// Vector Single-Width Bit Shift Instructions
defm XVSLL_V : VSHT_IV_V_X_I<"vsll", 0b100101>;
defm XVSRL_V : VSHT_IV_V_X_I<"vsrl", 0b101000>;
defm XVSRA_V : VSHT_IV_V_X_I<"vsra", 0b101001>;

// Vector Narrowing Integer Right Shift Instructions
// Refer to 11.3. Narrowing Vector Arithmetic Instructions
// The destination vector register group cannot overlap the first source
// vector register group (specified by vs2). The destination vector register
// group cannot overlap the mask register if used, unless LMUL=1.
let Constraints = "@earlyclobber $vd" in {
defm XVNSRL_W : XVNSHT_IV_V_X_I<"vnsrl", 0b101100>;
defm XVNSRA_W : XVNSHT_IV_V_X_I<"vnsra", 0b101101>;
} // Constraints = "@earlyclobber $vd"

// Vector Integer Comparison Instructions
let RVVConstraint = NoConstraint in {
defm XVMSEQ_V : VCMP_IV_V_X_I<"vmseq", 0b011000>;
defm XVMSNE_V : VCMP_IV_V_X_I<"vmsne", 0b011001>;
defm XVMSLTU_V : VCMP_IV_V_X<"vmsltu", 0b011010>;
defm XVMSLT_V : VCMP_IV_V_X<"vmslt", 0b011011>;
defm XVMSLEU_V : VCMP_IV_V_X_I<"vmsleu", 0b011100>;
defm XVMSLE_V : VCMP_IV_V_X_I<"vmsle", 0b011101>;
defm XVMSGTU_V : VCMP_IV_X_I<"vmsgtu", 0b011110>;
defm XVMSGT_V : VCMP_IV_X_I<"vmsgt", 0b011111>;
} // RVVConstraint = NoConstraint

// Vector Integer Min/Max Instructions
defm XVMINU_V : VCMP_IV_V_X<"vminu", 0b000100>;
defm XVMIN_V : VCMP_IV_V_X<"vmin", 0b000101>;
defm XVMAXU_V : VCMP_IV_V_X<"vmaxu", 0b000110>;
defm XVMAX_V : VCMP_IV_V_X<"vmax", 0b000111>;

// Vector Single-Width Integer Multiply Instructions
defm XVMUL_V : VMUL_MV_V_X<"vmul", 0b100101>;
defm XVMULH_V : VMUL_MV_V_X<"vmulh", 0b100111>;
defm XVMULHU_V : VMUL_MV_V_X<"vmulhu", 0b100100>;
defm XVMULHSU_V : VMUL_MV_V_X<"vmulhsu", 0b100110>;

// Vector Integer Divide Instructions
defm XVDIVU_V : VDIV_MV_V_X<"vdivu", 0b100000>;
defm XVDIV_V : VDIV_MV_V_X<"vdiv", 0b100001>;
defm XVREMU_V : VDIV_MV_V_X<"vremu", 0b100010>;
defm XVREM_V : VDIV_MV_V_X<"vrem", 0b100011>;

// Vector Widening Integer Multiply Instructions
let Constraints = "@earlyclobber $vd", RVVConstraint = WidenV in {
defm XVWMUL_V : VWMUL_MV_V_X<"vwmul", 0b111011>;
defm XVWMULU_V : VWMUL_MV_V_X<"vwmulu", 0b111000>;
defm XVWMULSU_V : VWMUL_MV_V_X<"vwmulsu", 0b111010>;
} // Constraints = "@earlyclobber $vd", RVVConstraint = WidenV

// Vector Single-Width Integer Multiply-Add Instructions
defm XVMACC_V : VMAC_MV_V_X<"vmacc", 0b101101>;
defm XVNMSAC_V : VMAC_MV_V_X<"vnmsac", 0b101111>;
defm XVMADD_V : VMAC_MV_V_X<"vmadd", 0b101001>;
defm XVNMSUB_V : VMAC_MV_V_X<"vnmsub", 0b101011>;

// Vector Widening Integer Multiply-Add Instructions
let Constraints = "@earlyclobber $vd", RVVConstraint = WidenV in {
defm XVWMACCU_V : VWMAC_MV_V_X<"vwmaccu", 0b111100>;
defm XVWMACC_V : VWMAC_MV_V_X<"vwmacc", 0b111101>;
defm XVWMACCSU_V : VWMAC_MV_V_X<"vwmaccsu", 0b111111>;
defm XVWMACCUS_V : VWMAC_MV_X<"vwmaccus", 0b111110>;
} // Constraints = "@earlyclobber $vd", RVVConstraint = WidenV

// Vector Integer Merge Instructions
defm XVMERGE_V : VMRG_IV_V_X_I<"vmerge", 0b010111>;

// Vector Integer Move Instructions
let hasSideEffects = 0, mayLoad = 0, mayStore = 0, vs2 = 0, vm = 1,
    RVVConstraint = NoConstraint  in {
// op vd, vs1
def XVMV_V_V : RVInstVV<0b010111, OPIVV, (outs VR:$vd),
                       (ins VR:$vs1), "vmv.v.v", "$vd, $vs1">,
              Sched<[WriteVIMovV_WorstCase, ReadVIMovV_WorstCase]>;
// op vd, rs1
def XVMV_V_X : RVInstVX<0b010111, OPIVX, (outs VR:$vd),
                       (ins GPR:$rs1), "vmv.v.x", "$vd, $rs1">,
              Sched<[WriteVIMovX_WorstCase, ReadVIMovX_WorstCase]>;
// op vd, imm
def XVMV_V_I : RVInstIVI<0b010111, (outs VR:$vd),
                       (ins simm5:$imm), "vmv.v.i", "$vd, $imm">,
              Sched<[WriteVIMovI_WorstCase]>;
} // hasSideEffects = 0, mayLoad = 0, mayStore = 0

// Vector Fixed-Point Arithmetic Instructions
defm XVSADDU_V : VSALU_IV_V_X_I<"vsaddu", 0b100000>;
defm XVSADD_V : VSALU_IV_V_X_I<"vsadd", 0b100001>;
defm XVSSUBU_V : VSALU_IV_V_X<"vssubu", 0b100010>;
defm XVSSUB_V : VSALU_IV_V_X<"vssub", 0b100011>;

// Vector Single-Width Averaging Add and Subtract
defm XVAADD_V : VSALU_IV_V_X_I<"vaadd", 0b100100>;
defm XVASUB_V : VSALU_IV_V_X<"vasub", 0b100110>;

// Vector Single-Width Fractional Multiply with Rounding and Saturation
defm XVSMUL_V : VSMUL_IV_V_X<"vsmul", 0b100111>;

// Vector Widening Saturating Scaled Multiply-Add
let Constraints = "@earlyclobber $vd", RVVConstraint = WidenV in {
defm XVWSMACCU_V : VWSMAC_V_X<"vwsmaccu", 0b111100>;
defm XVWSMACC_V : VWSMAC_V_X<"vwsmacc", 0b111101>;
defm XVWSMACCSU_V : VWSMAC_V_X<"vwsmaccsu", 0b111110>;
defm XVWSMACCUS_V : VWSMAC_X<"vwsmaccus", 0b111111>;
} // Constraints = "@earlyclobber $vd", RVVConstraint = WidenV

// Vector Single-Width Scaling Shift Instructions
defm XVSSRL_V : VSSHF_IV_V_X_I<"vssrl", 0b101010>;
defm XVSSRA_V : VSSHF_IV_V_X_I<"vssra", 0b101011>;

// Vector Narrowing Fixed-Point Clip Instructions
let Constraints = "@earlyclobber $vd" in {
defm XVNCLIPU_W : XVNCLP_IV_V_X_I<"vnclipu", 0b101110>;
defm XVNCLIP_W : XVNCLP_IV_V_X_I<"vnclip", 0b101111>;
} // Constraints = "@earlyclobber $vd"
} // Predicates = [HasVendorXTHeadV]

let Predicates = [HasVendorXTHeadV, HasStdExtF] in {
// Vector Single-Width Floating-Point Add/Subtract Instructions
let Uses = [FRM], mayRaiseFPException = true in {
defm XVFADD_V : VALU_FV_V_F<"vfadd", 0b000000>;
defm XVFSUB_V : VALU_FV_V_F<"vfsub", 0b000010>;
defm XVFRSUB_V : VALU_FV_F<"vfrsub", 0b100111>;
}

// Vector Widening Floating-Point Add/Subtract Instructions
let Constraints = "@earlyclobber $vd",
    Uses = [FRM],
    mayRaiseFPException = true in {
let RVVConstraint = WidenV in {
defm XVFWADD_V : VWALU_FV_V_F<"vfwadd", 0b110000, "v">;
defm XVFWSUB_V : VWALU_FV_V_F<"vfwsub", 0b110010, "v">;
} // RVVConstraint = WidenV
// Set earlyclobber for following instructions for second and mask operands.
// This has the downside that the earlyclobber constraint is too coarse and
// will impose unnecessary restrictions by not allowing the destination to
// overlap with the first (wide) operand.
let RVVConstraint = WidenW in {
defm XVFWADD_W : VWALU_FV_V_F<"vfwadd", 0b110100, "w">;
defm XVFWSUB_W : VWALU_FV_V_F<"vfwsub", 0b110110, "w">;
} // RVVConstraint = WidenW
} // Constraints = "@earlyclobber $vd", Uses = [FRM], mayRaiseFPException = true

// Vector Single-Width Floating-Point Multiply/Divide Instructions
let Uses = [FRM], mayRaiseFPException = true in {
defm XVFMUL_V : VMUL_FV_V_F<"vfmul", 0b100100>;
defm XVFDIV_V : VDIV_FV_V_F<"vfdiv", 0b100000>;
defm XVFRDIV_V : VDIV_FV_F<"vfrdiv", 0b100001>;
}

// Vector Widening Floating-Point Multiply
let Constraints = "@earlyclobber $vd", RVVConstraint = WidenV,
    Uses = [FRM], mayRaiseFPException = true in {
defm XVFWMUL_V : VWMUL_FV_V_F<"vfwmul", 0b111000>;
} // Constraints = "@earlyclobber $vd", RVVConstraint = WidenV, Uses = [FRM], mayRaiseFPException = true

// Vector Single-Width Floating-Point Fused Multiply-Add Instructions
let Uses = [FRM], mayRaiseFPException = true in {
defm XVFMACC_V : VMAC_FV_V_F<"vfmacc", 0b101100>;
defm XVFNMACC_V : VMAC_FV_V_F<"vfnmacc", 0b101101>;
defm XVFMSAC_V : VMAC_FV_V_F<"vfmsac", 0b101110>;
defm XVFNMSAC_V : VMAC_FV_V_F<"vfnmsac", 0b101111>;
defm XVFMADD_V : VMAC_FV_V_F<"vfmadd", 0b101000>;
defm XVFNMADD_V : VMAC_FV_V_F<"vfnmadd", 0b101001>;
defm XVFMSUB_V : VMAC_FV_V_F<"vfmsub", 0b101010>;
defm XVFNMSUB_V : VMAC_FV_V_F<"vfnmsub", 0b101011>;
}

// Vector Widening Floating-Point Fused Multiply-Add Instructions
let Constraints = "@earlyclobber $vd", RVVConstraint = WidenV,
    Uses = [FRM], mayRaiseFPException = true in {
defm XVFWMACC_V : VWMAC_FV_V_F<"vfwmacc", 0b111100>;
defm XVFWNMACC_V : VWMAC_FV_V_F<"vfwnmacc", 0b111101>;
defm XVFWMSAC_V : VWMAC_FV_V_F<"vfwmsac", 0b111110>;
defm XVFWNMSAC_V : VWMAC_FV_V_F<"vfwnmsac", 0b111111>;
} // Constraints = "@earlyclobber $vd", RVVConstraint = WidenV, Uses = [FRM], mayRaiseFPException = true

// Vector Floating-Point Square-Root Instruction
let Uses = [FRM], mayRaiseFPException = true in {
defm XVFSQRT_V : VSQR_FV_VS2<"vfsqrt.v", 0b100011, 0b00000>;
}

// Vector Floating-Point MIN/MAX Instructions
let mayRaiseFPException = true in {
defm XVFMIN_V : VCMP_FV_V_F<"vfmin", 0b000100>;
defm XVFMAX_V : VCMP_FV_V_F<"vfmax", 0b000110>;
}

// Vector Floating-Point Sign-Injection Instructions
defm XVFSGNJ_V : VSGNJ_FV_V_F<"vfsgnj", 0b001000>;
defm XVFSGNJN_V : VSGNJ_FV_V_F<"vfsgnjn", 0b001001>;
defm XVFSGNJX_V : VSGNJ_FV_V_F<"vfsgnjx", 0b001010>;

// Vector Floating-Point Compare Instructions
let RVVConstraint = NoConstraint, mayRaiseFPException = true in {
defm XVMFEQ_V : VCMP_FV_V_F<"vmfeq", 0b011000>;
defm XVMFNE_V : VCMP_FV_V_F<"vmfne", 0b011100>;
defm XVMFLT_V : VCMP_FV_V_F<"vmflt", 0b011011>;
defm XVMFLE_V : VCMP_FV_V_F<"vmfle", 0b011001>;
defm XVMFGT_V : VCMP_FV_F<"vmfgt", 0b011101>;
defm XVMFGE_V : VCMP_FV_F<"vmfge", 0b011111>;
defm XVMFORD_V : VCMP_FV_V_F<"vmford", 0b011010>;
} // RVVConstraint = NoConstraint, mayRaiseFPException = true

// Vector Floating-Point Classify Instruction
defm XVFCLASS_V : VCLS_FV_VS2<"vfclass.v", 0b100011, 0b10000>;

let hasSideEffects = 0, mayLoad = 0, mayStore = 0 in {

// Vector Floating-Point Merge Instruction
let vm = 0 in
def XVFMERGE_VFM : RVInstVX<0b010111, OPFVF, (outs VR:$vd),
                           (ins VR:$vs2, FPR32:$rs1, VMV0:$v0),
                           "vfmerge.vfm", "$vd, $vs2, $rs1, v0">,
                  Sched<[WriteVFMergeV_WorstCase, ReadVFMergeV_WorstCase,
                         ReadVFMergeF_WorstCase, ReadVMask]>;

// Vector Floating-Point Move Instruction
let RVVConstraint = NoConstraint in
let vm = 1, vs2 = 0 in
def XVFMV_V_F : RVInstVX<0b010111, OPFVF, (outs VR:$vd),
                       (ins FPR32:$rs1), "vfmv.v.f", "$vd, $rs1">,
               Sched<[WriteVFMovV_WorstCase, ReadVFMovF_WorstCase]>;

} // hasSideEffects = 0, mayLoad = 0, mayStore = 0

// Single-Width Floating-Point/Integer Type-Convert Instructions
let mayRaiseFPException = true in {
let Uses = [FRM] in {
defm XVFCVT_XU_F_V : VCVTI_FV_VS2<"vfcvt.xu.f.v", 0b100010, 0b00000>;
defm XVFCVT_X_F_V : VCVTI_FV_VS2<"vfcvt.x.f.v", 0b100010, 0b00001>;
defm XVFCVT_F_XU_V : VCVTF_IV_VS2<"vfcvt.f.xu.v", 0b100010, 0b00010>;
defm XVFCVT_F_X_V : VCVTF_IV_VS2<"vfcvt.f.x.v", 0b100010, 0b00011>;
}
} // mayRaiseFPException = true

// Widening Floating-Point/Integer Type-Convert Instructions
let Constraints = "@earlyclobber $vd", RVVConstraint = WidenCvt,
    mayRaiseFPException = true in {
let Uses = [FRM] in {
defm XVFWCVT_XU_F_V : VWCVTI_FV_VS2<"vfwcvt.xu.f.v", 0b100010, 0b01000>;
defm XVFWCVT_X_F_V : VWCVTI_FV_VS2<"vfwcvt.x.f.v", 0b100010, 0b01001>;
}
defm XVFWCVT_F_XU_V : VWCVTF_IV_VS2<"vfwcvt.f.xu.v", 0b100010, 0b01010>;
defm XVFWCVT_F_X_V : VWCVTF_IV_VS2<"vfwcvt.f.x.v", 0b100010, 0b01011>;
defm XVFWCVT_F_F_V : VWCVTF_FV_VS2<"vfwcvt.f.f.v", 0b100010, 0b01100>;
} // Constraints = "@earlyclobber $vd", RVVConstraint = WidenCvt

// Narrowing Floating-Point/Integer Type-Convert Instructions
let Constraints = "@earlyclobber $vd", mayRaiseFPException = true in {
let Uses = [FRM] in {
defm XVFNCVT_XU_F_W : VNCVTI_FV_VS2<"vfncvt.xu.f.v", 0b100010, 0b10000>;
defm XVFNCVT_X_F_W : VNCVTI_FV_VS2<"vfncvt.x.f.v", 0b100010, 0b10001>;
defm XVFNCVT_F_XU_W : VNCVTF_IV_VS2<"vfncvt.f.xu.v", 0b100010, 0b10010>;
defm XVFNCVT_F_X_W : VNCVTF_IV_VS2<"vfncvt.f.x.v", 0b100010, 0b10011>;
defm XVFNCVT_F_F_W : VNCVTF_FV_VS2<"vfncvt.f.f.v", 0b100010, 0b10100>;
}
} // Constraints = "@earlyclobber $vd", mayRaiseFPException = true
} // Predicates = [HasVendorXTHeadV, HasStdExtF]

let Predicates = [HasVendorXTHeadV] in {
// Vector Single-Width Integer Reduction Instructions
let RVVConstraint = NoConstraint in {
defm XVREDSUM : VRED_MV_V<"vredsum", 0b000000>;
defm XVREDMAXU : VRED_MV_V<"vredmaxu", 0b000110>;
defm XVREDMAX : VRED_MV_V<"vredmax", 0b000111>;
defm XVREDMINU : VRED_MV_V<"vredminu", 0b000100>;
defm XVREDMIN : VRED_MV_V<"vredmin", 0b000101>;
defm XVREDAND : VRED_MV_V<"vredand", 0b000001>;
defm XVREDOR : VRED_MV_V<"vredor", 0b000010>;
defm XVREDXOR : VRED_MV_V<"vredxor", 0b000011>;
} // RVVConstraint = NoConstraint

// Vector Widening Integer Reduction Instructions
let Constraints = "@earlyclobber $vd", RVVConstraint = NoConstraint in {
// Set earlyclobber for following instructions for second and mask operands.
// This has the downside that the earlyclobber constraint is too coarse and
// will impose unnecessary restrictions by not allowing the destination to
// overlap with the first (wide) operand.
defm XVWREDSUMU : VWRED_IV_V<"vwredsumu", 0b110000>;
defm XVWREDSUM : VWRED_IV_V<"vwredsum", 0b110001>;
} // Constraints = "@earlyclobber $vd", RVVConstraint = NoConstraint

} // Predicates = [HasVendorXTHeadV]

let Predicates = [HasVendorXTHeadV, HasStdExtF] in {
// Vector Single-Width Floating-Point Reduction Instructions
let RVVConstraint = NoConstraint in {
let Uses = [FRM], mayRaiseFPException = true in {
defm XVFREDOSUM : VREDO_FV_V<"vfredosum", 0b000011>;
defm XVFREDUSUM : VRED_FV_V<"vfredsum", 0b000001>;
}
let mayRaiseFPException = true in {
defm XVFREDMAX : VRED_FV_V<"vfredmax", 0b000111>;
defm XVFREDMIN : VRED_FV_V<"vfredmin", 0b000101>;
}
} // RVVConstraint = NoConstraint

// Vector Widening Floating-Point Reduction Instructions
let Constraints = "@earlyclobber $vd", RVVConstraint = NoConstraint in {
// Set earlyclobber for following instructions for second and mask operands.
// This has the downside that the earlyclobber constraint is too coarse and
// will impose unnecessary restrictions by not allowing the destination to
// overlap with the first (wide) operand.
let Uses = [FRM], mayRaiseFPException = true in {
defm XVFWREDOSUM : VWREDO_FV_V<"vfwredosum", 0b110011>;
defm XVFWREDUSUM : VWRED_FV_V<"vfwredsum", 0b110001>;
}
} // Constraints = "@earlyclobber $vd", RVVConstraint = NoConstraint
} // Predicates = [HasVendorXTHeadV, HasStdExtF]

let Predicates = [HasVendorXTHeadV] in {
// Vector Mask-Register Logical Instructions
let RVVConstraint = NoConstraint in {
defm XVMAND_M : VMALU_MV_Mask<"vmand", 0b011001, "m">;
defm XVMNAND_M : VMALU_MV_Mask<"vmnand", 0b011101, "m">;
defm XVMANDN_M : VMALU_MV_Mask<"vmandnot", 0b011000, "m">;
defm XVMXOR_M : VMALU_MV_Mask<"vmxor", 0b011011, "m">;
defm XVMOR_M : VMALU_MV_Mask<"vmor", 0b011010, "m">;
defm XVMNOR_M : VMALU_MV_Mask<"vmnor", 0b011110, "m">;
defm XVMORN_M : VMALU_MV_Mask<"vmornot", 0b011100, "m">;
defm XVMXNOR_M : VMALU_MV_Mask<"vmxnor", 0b011111, "m">;
}

let hasSideEffects = 0, mayLoad = 0, mayStore = 0,
    RVVConstraint = NoConstraint  in {

// Vector mask population count vmpopc
def XVMPOPC_M : RVInstV<0b010100, 0b00000, OPMVV, (outs GPR:$vd),
                      (ins VR:$vs2, VMaskOp:$vm),
                      "vmpopc.m", "$vd, $vs2$vm">,
              Sched<[WriteVMPopV_WorstCase, ReadVMPopV_WorstCase,
                     ReadVMask]>;

// vmfirst find-first-set mask bit
def XVMFIRST_M : RVInstV<0b010101, 0b00000, OPMVV, (outs GPR:$vd),
                       (ins VR:$vs2, VMaskOp:$vm),
                       "vmfirst.m", "$vd, $vs2$vm">,
              Sched<[WriteVMFFSV_WorstCase, ReadVMFFSV_WorstCase,
                     ReadVMask]>;

} // hasSideEffects = 0, mayLoad = 0, mayStore = 0

let Constraints = "@earlyclobber $vd", RVVConstraint = Iota in {

// vmsbf.m set-before-first mask bit
defm XVMSBF_M : VMSFS_MV_V<"vmsbf.m", 0b010110, 0b00001>;
// vmsif.m set-including-first mask bit
defm XVMSIF_M : VMSFS_MV_V<"vmsif.m", 0b010110, 0b00011>;
// vmsof.m set-only-first mask bit
defm XVMSOF_M : VMSFS_MV_V<"vmsof.m", 0b010110, 0b00010>;
// Vector Iota Instruction
 defm XVIOTA_M : VMIOT_MV_V<"viota.m", 0b010110, 0b10000>;

} // Constraints = "@earlyclobber $vd", RVVConstraint = Iota

let hasSideEffects = 0, mayLoad = 0, mayStore = 0 in {
// Vector Element Index Instruction
let vs2 = 0 in
def XVID_V : RVInstV<0b010110, 0b10001, OPMVV, (outs VR:$vd),
                    (ins VMaskOp:$vm), "vid.v", "$vd$vm">,
            Sched<[WriteVMIdxV_WorstCase, ReadVMask]>;

let vm = 1, RVVConstraint = NoConstraint in {
// Integer Extract Instruction
def XVEXT_X_V : RVInstVX<0b001100, OPMVV, (outs GPR:$vd),
                         (ins VR:$vs2, GPR:$rs1),
                         "vext.x.v", "$vd, $vs2, $rs1">,
                Sched<[WriteVIMovVX, ReadVIMovVX,
                       ReadVIMovXX]>;

// Integer Scalar Move Instruction
let Constraints = "$vd = $vd_wb" in
def XVMV_S_X : RVInstV2<0b001101, 0b00000, OPMVX, (outs VR:$vd_wb),
                       (ins VR:$vd, GPR:$rs1), "vmv.s.x", "$vd, $rs1">,
               Sched<[WriteVIMovXV, ReadVIMovXV,
                      ReadVIMovXX]>;
}
} // hasSideEffects = 0, mayLoad = 0, mayStore = 0
} // Predicates = [HasVendorXTHeadV]

let Predicates = [HasVendorXTHeadV, HasStdExtF] in {
let hasSideEffects = 0, mayLoad = 0, mayStore = 0, vm = 1,
    RVVConstraint = NoConstraint  in {
// Floating-Point Scalar Move Instructions
def XVFMV_F_S : RVInstV<0b001100, 0b00000, OPFVV, (outs FPR32:$vd),
                        (ins VR:$vs2), "vfmv.f.s", "$vd, $vs2">,
                Sched<[WriteVFMovVF, ReadVFMovVF]>;
let Constraints = "$vd = $vd_wb" in
def XVFMV_S_F : RVInstV2<0b001101, 0b00000, OPFVF, (outs VR:$vd_wb),
                         (ins VR:$vd, FPR32:$rs1), "vfmv.s.f", "$vd, $rs1">,
                Sched<[WriteVFMovFV, ReadVFMovFV,
                       ReadVFMovFX]>;

} // hasSideEffects = 0, mayLoad = 0, mayStore = 0, vm = 1
} // Predicates = [HasVendorXTHeadV, HasStdExtF]

let Predicates = [HasVendorXTHeadV] in  {
// Vector Slide Instructions
let Constraints = "@earlyclobber $vd", RVVConstraint = SlideUp in {
defm XVSLIDEUP_V : VSLD_IV_X_I<"vslideup", 0b001110>;
defm XVSLIDE1UP_V : VSLD1_MV_X<"vslide1up", 0b001110>;
} // Constraints = "@earlyclobber $vd", RVVConstraint = SlideUp
defm XVSLIDEDOWN_V : VSLD_IV_X_I<"vslidedown", 0b001111>;
defm XVSLIDE1DOWN_V : VSLD1_MV_X<"vslide1down", 0b001111>;

// Vector Register Gather Instruction
let Constraints = "@earlyclobber $vd", RVVConstraint = Vrgather in {
defm XVRGATHER_V : VGTR_IV_V_X_I<"vrgather", 0b001100>;
} // Constraints = "@earlyclobber $vd", RVVConstraint = Vrgather

// Vector Compress Instruction
let Constraints = "@earlyclobber $vd", RVVConstraint = Vcompress in {
defm XVCOMPRESS_V : VCPR_MV_Mask<"vcompress", 0b010111>;
} // Constraints = "@earlyclobber $vd", RVVConstraint = Vcompress
} // Predicates = [HasVendorXTHeadV]
} // AsmVariantName = "RVV0p71", DecoderNamespace = "RVV0p71"

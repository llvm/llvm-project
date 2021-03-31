//===-- RISCVInstrInfoV.td - RISC-V 'V' instructions -------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// This file describes the RISC-V instructions from the standard 'V' Vector
/// extension, version 0.10.
/// This version is still experimental as the 'V' extension hasn't been
/// ratified yet.
///
//===----------------------------------------------------------------------===//

include "RISCVInstrFormatsV.td"

//===----------------------------------------------------------------------===//
// Operand and SDNode transformation definitions.
//===----------------------------------------------------------------------===//

def VTypeIAsmOperand : AsmOperandClass {
  let Name = "VTypeI";
  let ParserMethod = "parseVTypeI";
  let DiagnosticType = "InvalidVTypeI";
}

def VTypeIOp : Operand<XLenVT> {
  let ParserMatchClass = VTypeIAsmOperand;
  let PrintMethod = "printVTypeI";
  let DecoderMethod = "decodeUImmOperand<11>";
}

def VMaskAsmOperand : AsmOperandClass {
  let Name = "RVVMaskRegOpOperand";
  let RenderMethod = "addRegOperands";
  let PredicateMethod = "isV0Reg";
  let ParserMethod = "parseMaskReg";
  let IsOptional = 1;
  let DefaultMethod = "defaultMaskRegOp";
  let DiagnosticType = "InvalidVMaskRegister";
}

def VMaskOp : RegisterOperand<VMV0> {
  let ParserMatchClass = VMaskAsmOperand;
  let PrintMethod = "printVMaskReg";
  let EncoderMethod = "getVMaskReg";
  let DecoderMethod = "decodeVMaskReg";
}

def simm5 : Operand<XLenVT>, ImmLeaf<XLenVT, [{return isInt<5>(Imm);}]> {
  let ParserMatchClass = SImmAsmOperand<5>;
  let EncoderMethod = "getImmOpValue";
  let DecoderMethod = "decodeSImmOperand<5>";
  let MCOperandPredicate = [{
    int64_t Imm;
    if (MCOp.evaluateAsConstantImm(Imm))
      return isInt<5>(Imm);
    return MCOp.isBareSymbolRef();
  }];
}

def SImm5Plus1AsmOperand : AsmOperandClass {
  let Name = "SImm5Plus1";
  let RenderMethod = "addImmOperands";
  let DiagnosticType = "InvalidSImm5Plus1";
}

def simm5_plus1 : Operand<XLenVT>, ImmLeaf<XLenVT,
                                           [{return isInt<5>(Imm - 1);}]> {
  let ParserMatchClass = SImm5Plus1AsmOperand;
  let MCOperandPredicate = [{
    int64_t Imm;
    if (MCOp.evaluateAsConstantImm(Imm))
      return isInt<5>(Imm - 1);
    return MCOp.isBareSymbolRef();
  }];
}

//===----------------------------------------------------------------------===//
// Instruction class templates
//===----------------------------------------------------------------------===//

let hasSideEffects = 0, mayLoad = 1, mayStore = 0 in {
// load vd, (rs1)
class VUnitStrideLoadMask<string opcodestr>
    : RVInstVLU<0b000, LSWidth8.Value{3}, LUMOPUnitStrideMask, LSWidth8.Value{2-0},
                (outs VR:$vd),
                (ins GPR:$rs1), opcodestr, "$vd, (${rs1})"> {
  let vm = 1;
}

// load vd, (rs1), vm
class VUnitStrideLoad<RISCVLSUMOP lumop, RISCVWidth width,
                      string opcodestr>
    : RVInstVLU<0b000, width.Value{3}, lumop, width.Value{2-0},
                (outs VR:$vd),
                (ins GPR:$rs1, VMaskOp:$vm), opcodestr, "$vd, (${rs1})$vm">;

// load vd, (rs1), rs2, vm
class VStridedLoad<RISCVWidth width, string opcodestr>
    : RVInstVLS<0b000, width.Value{3}, width.Value{2-0},
                (outs VR:$vd),
                (ins GPR:$rs1, GPR:$rs2, VMaskOp:$vm), opcodestr,
                "$vd, (${rs1}), $rs2$vm">;

// load vd, (rs1), vs2, vm
class VIndexedLoad<RISCVMOP mop, RISCVWidth width, string opcodestr>
    : RVInstVLX<0b000, width.Value{3}, mop, width.Value{2-0},
                (outs VR:$vd),
                (ins GPR:$rs1, VR:$vs2, VMaskOp:$vm), opcodestr,
                "$vd, (${rs1}), $vs2$vm">;

// vl<nf>r.v vd, (rs1)
class VWholeLoad<bits<3> nf, RISCVWidth width, string opcodestr, RegisterClass VRC>
    : RVInstVLU<nf, width.Value{3}, LUMOPUnitStrideWholeReg,
                width.Value{2-0}, (outs VRC:$vd), (ins GPR:$rs1),
                opcodestr, "$vd, (${rs1})"> {
  let vm = 1;
  let Uses = [];
  let RVVConstraint = NoConstraint;
}

// segment load vd, (rs1), vm
class VUnitStrideSegmentLoad<bits<3> nf, RISCVLSUMOP lumop,
                             RISCVWidth width, string opcodestr>
    : RVInstVLU<nf, width.Value{3}, lumop, width.Value{2-0},
                (outs VR:$vd),
                (ins GPR:$rs1, VMaskOp:$vm), opcodestr, "$vd, (${rs1})$vm">;

// segment load vd, (rs1), rs2, vm
class VStridedSegmentLoad<bits<3> nf, RISCVWidth width, string opcodestr>
    : RVInstVLS<nf, width.Value{3}, width.Value{2-0},
                (outs VR:$vd),
                (ins GPR:$rs1, GPR:$rs2, VMaskOp:$vm), opcodestr,
                "$vd, (${rs1}), $rs2$vm">;

// segment load vd, (rs1), vs2, vm
class VIndexedSegmentLoad<bits<3> nf, RISCVMOP mop, RISCVWidth width,
                          string opcodestr>
    : RVInstVLX<nf, width.Value{3}, mop, width.Value{2-0},
                (outs VR:$vd),
                (ins GPR:$rs1, VR:$vs2, VMaskOp:$vm), opcodestr,
                "$vd, (${rs1}), $vs2$vm">;
} // hasSideEffects = 0, mayLoad = 1, mayStore = 0

let hasSideEffects = 0, mayLoad = 0, mayStore = 1 in {
// store vd, vs3, (rs1)
class VUnitStrideStoreMask<string opcodestr>
    : RVInstVSU<0b000, LSWidth8.Value{3}, SUMOPUnitStrideMask, LSWidth8.Value{2-0},
                (outs), (ins VR:$vs3, GPR:$rs1), opcodestr,
                "$vs3, (${rs1})"> {
  let vm = 1;
}

// store vd, vs3, (rs1), vm
class VUnitStrideStore<RISCVLSUMOP sumop, RISCVWidth width,
                         string opcodestr>
    : RVInstVSU<0b000, width.Value{3}, sumop, width.Value{2-0},
                (outs), (ins VR:$vs3, GPR:$rs1, VMaskOp:$vm), opcodestr,
                "$vs3, (${rs1})$vm">;

// store vd, vs3, (rs1), rs2, vm
class VStridedStore<RISCVWidth width, string opcodestr>
    : RVInstVSS<0b000, width.Value{3}, width.Value{2-0}, (outs),
                (ins VR:$vs3, GPR:$rs1, GPR:$rs2, VMaskOp:$vm),
                opcodestr, "$vs3, (${rs1}), $rs2$vm">;

// store vd, vs3, (rs1), vs2, vm
class VIndexedStore<RISCVMOP mop, RISCVWidth width, string opcodestr>
    : RVInstVSX<0b000, width.Value{3}, mop, width.Value{2-0}, (outs),
                (ins VR:$vs3, GPR:$rs1, VR:$vs2, VMaskOp:$vm),
                opcodestr, "$vs3, (${rs1}), $vs2$vm">;

// vs<nf>r.v vd, (rs1)
class VWholeStore<bits<3> nf, string opcodestr, RegisterClass VRC>
    : RVInstVSU<nf, 0, SUMOPUnitStrideWholeReg,
                0b000, (outs), (ins VRC:$vs3, GPR:$rs1),
                opcodestr, "$vs3, (${rs1})"> {
  let vm = 1;
  let Uses = [];
}

// segment store vd, vs3, (rs1), vm
class VUnitStrideSegmentStore<bits<3> nf, RISCVWidth width, string opcodestr>
    : RVInstVSU<nf, width.Value{3}, SUMOPUnitStride, width.Value{2-0},
                (outs), (ins VR:$vs3, GPR:$rs1, VMaskOp:$vm), opcodestr,
                "$vs3, (${rs1})$vm">;

// segment store vd, vs3, (rs1), rs2, vm
class VStridedSegmentStore<bits<3> nf, RISCVWidth width, string opcodestr>
    : RVInstVSS<nf, width.Value{3}, width.Value{2-0}, (outs),
                (ins VR:$vs3, GPR:$rs1, GPR:$rs2, VMaskOp:$vm),
                opcodestr, "$vs3, (${rs1}), $rs2$vm">;

// segment store vd, vs3, (rs1), vs2, vm
class VIndexedSegmentStore<bits<3> nf, RISCVMOP mop, RISCVWidth width,
                           string opcodestr>
    : RVInstVSX<nf, width.Value{3}, mop, width.Value{2-0}, (outs),
                (ins VR:$vs3, GPR:$rs1, VR:$vs2, VMaskOp:$vm),
                opcodestr, "$vs3, (${rs1}), $vs2$vm">;
} // hasSideEffects = 0, mayLoad = 0, mayStore = 1

let hasSideEffects = 0, mayLoad = 0, mayStore = 0 in {
// op vd, vs2, vs1, vm
class VALUVV<bits<6> funct6, RISCVVFormat opv, string opcodestr>
    : RVInstVV<funct6, opv, (outs VR:$vd),
                (ins VR:$vs2, VR:$vs1, VMaskOp:$vm),
                opcodestr, "$vd, $vs2, $vs1$vm">;

// op vd, vs2, vs1, v0 (without mask, use v0 as carry input)
class VALUmVV<bits<6> funct6, RISCVVFormat opv, string opcodestr>
    : RVInstVV<funct6, opv, (outs VR:$vd),
                (ins VR:$vs2, VR:$vs1, VMV0:$v0),
                opcodestr, "$vd, $vs2, $vs1, v0"> {
  let vm = 0;
}

// op vd, vs1, vs2, vm (reverse the order of vs1 and vs2)
class VALUrVV<bits<6> funct6, RISCVVFormat opv, string opcodestr>
    : RVInstVV<funct6, opv, (outs VR:$vd),
                (ins VR:$vs1, VR:$vs2, VMaskOp:$vm),
                opcodestr, "$vd, $vs1, $vs2$vm">;

// op vd, vs2, vs1
class VALUVVNoVm<bits<6> funct6, RISCVVFormat opv, string opcodestr>
    : RVInstVV<funct6, opv, (outs VR:$vd),
               (ins VR:$vs2, VR:$vs1),
               opcodestr, "$vd, $vs2, $vs1"> {
  let vm = 1;
}

// op vd, vs2, rs1, vm
class VALUVX<bits<6> funct6, RISCVVFormat opv, string opcodestr>
    : RVInstVX<funct6, opv, (outs VR:$vd),
                (ins VR:$vs2, GPR:$rs1, VMaskOp:$vm),
                opcodestr, "$vd, $vs2, $rs1$vm">;

// op vd, vs2, rs1, v0 (without mask, use v0 as carry input)
class VALUmVX<bits<6> funct6, RISCVVFormat opv, string opcodestr>
    : RVInstVX<funct6, opv, (outs VR:$vd),
                (ins VR:$vs2, GPR:$rs1, VMV0:$v0),
                opcodestr, "$vd, $vs2, $rs1, v0"> {
  let vm = 0;
}

// op vd, rs1, vs2, vm (reverse the order of rs1 and vs2)
class VALUrVX<bits<6> funct6, RISCVVFormat opv, string opcodestr>
    : RVInstVX<funct6, opv, (outs VR:$vd),
                (ins GPR:$rs1, VR:$vs2, VMaskOp:$vm),
                opcodestr, "$vd, $rs1, $vs2$vm">;

// op vd, vs1, vs2
class VALUVXNoVm<bits<6> funct6, RISCVVFormat opv, string opcodestr>
    : RVInstVX<funct6, opv, (outs VR:$vd),
               (ins VR:$vs2, GPR:$rs1),
               opcodestr, "$vd, $vs2, $rs1"> {
  let vm = 1;
}

// op vd, vs2, imm, vm
class VALUVI<bits<6> funct6, string opcodestr, Operand optype = simm5>
    : RVInstIVI<funct6, (outs VR:$vd),
                (ins VR:$vs2, optype:$imm, VMaskOp:$vm),
                opcodestr, "$vd, $vs2, $imm$vm">;

// op vd, vs2, imm, v0 (without mask, use v0 as carry input)
class VALUmVI<bits<6> funct6, string opcodestr, Operand optype = simm5>
    : RVInstIVI<funct6, (outs VR:$vd),
                (ins VR:$vs2, optype:$imm, VMV0:$v0),
                opcodestr, "$vd, $vs2, $imm, v0"> {
  let vm = 0;
}

// op vd, vs2, imm, vm
class VALUVINoVm<bits<6> funct6, string opcodestr, Operand optype = simm5>
    : RVInstIVI<funct6, (outs VR:$vd),
                (ins VR:$vs2, optype:$imm),
                opcodestr, "$vd, $vs2, $imm"> {
  let vm = 1;
}

// op vd, vs2, rs1, vm (Float)
class VALUVF<bits<6> funct6, RISCVVFormat opv, string opcodestr>
    : RVInstVX<funct6, opv, (outs VR:$vd),
                (ins VR:$vs2, FPR32:$rs1, VMaskOp:$vm),
                opcodestr, "$vd, $vs2, $rs1$vm">;

// op vd, rs1, vs2, vm (Float) (with mask, reverse the order of rs1 and vs2)
class VALUrVF<bits<6> funct6, RISCVVFormat opv, string opcodestr>
    : RVInstVX<funct6, opv, (outs VR:$vd),
                (ins FPR32:$rs1, VR:$vs2, VMaskOp:$vm),
                opcodestr, "$vd, $rs1, $vs2$vm">;

// op vd, vs2, vm (use vs1 as instruction encoding)
class VALUVs2<bits<6> funct6, bits<5> vs1, RISCVVFormat opv, string opcodestr>
    : RVInstV<funct6, vs1, opv, (outs VR:$vd),
               (ins VR:$vs2, VMaskOp:$vm),
               opcodestr, "$vd, $vs2$vm">;
} // hasSideEffects = 0, mayLoad = 0, mayStore = 0

let hasSideEffects = 0, mayLoad = 1, mayStore = 1 in {
// vamo vd, (rs1), vs2, vd, vm
class VAMOWd<RISCVAMOOP amoop, RISCVWidth width, string opcodestr>
    : RVInstVAMO<amoop, width.Value{2-0}, (outs VR:$vd_wd),
            (ins GPR:$rs1, VR:$vs2, VR:$vd, VMaskOp:$vm),
            opcodestr, "$vd_wd, (${rs1}), $vs2, $vd$vm"> {
    let Constraints = "$vd_wd = $vd";
    let wd = 1;
    bits<5> vd;
    let Inst{11-7} = vd;
}

// vamo x0, (rs1), vs2, vs3, vm
class VAMONoWd<RISCVAMOOP amoop, RISCVWidth width, string opcodestr>
    : RVInstVAMO<amoop, width.Value{2-0}, (outs),
            (ins GPR:$rs1, VR:$vs2, VR:$vs3, VMaskOp:$vm),
            opcodestr, "x0, (${rs1}), $vs2, $vs3$vm"> {
    bits<5> vs3;
    let Inst{11-7} = vs3;
}

} // hasSideEffects = 0, mayLoad = 1, mayStore = 1

//===----------------------------------------------------------------------===//
// Combination of instruction classes.
// Use these multiclasses to define instructions more easily.
//===----------------------------------------------------------------------===//
multiclass VALU_IV_V_X_I<string opcodestr, bits<6> funct6, Operand optype = simm5, string vw = "v"> {
  def V  : VALUVV<funct6, OPIVV, opcodestr # "." # vw # "v">;
  def X  : VALUVX<funct6, OPIVX, opcodestr # "." # vw # "x">;
  def I  : VALUVI<funct6, opcodestr # "." # vw # "i", optype>;
}

multiclass VALU_IV_V_X<string opcodestr, bits<6> funct6, string vw = "v"> {
  def V  : VALUVV<funct6, OPIVV, opcodestr # "." # vw # "v">;
  def X  : VALUVX<funct6, OPIVX, opcodestr # "." # vw # "x">;
}

multiclass VALUr_IV_V_X<string opcodestr, bits<6> funct6, string vw = "v"> {
  def V : VALUrVV<funct6, OPIVV, opcodestr # "." # vw # "v">;
  def X : VALUrVX<funct6, OPIVX, opcodestr # "." # vw # "x">;
}

multiclass VALU_IV_X_I<string opcodestr, bits<6> funct6, Operand optype = simm5, string vw = "v"> {
  def X  : VALUVX<funct6, OPIVX, opcodestr # "." # vw # "x">;
  def I  : VALUVI<funct6, opcodestr # "." # vw # "i", optype>;
}

multiclass VALU_IV_V<string opcodestr, bits<6> funct6> {
  def _VS  : VALUVV<funct6, OPIVV, opcodestr # ".vs">;
}

multiclass VALUr_IV_X<string opcodestr, bits<6> funct6, string vw = "v"> {
  def X : VALUrVX<funct6, OPIVX, opcodestr # "." # vw # "x">;
}

multiclass VALU_MV_V_X<string opcodestr, bits<6> funct6, string vw = "v"> {
  def V : VALUVV<funct6, OPMVV, opcodestr # "." # vw # "v">;
  def X  : VALUVX<funct6, OPMVX, opcodestr # "." # vw # "x">;
}

multiclass VALU_MV_V<string opcodestr, bits<6> funct6> {
  def _VS : VALUVV<funct6, OPMVV, opcodestr # ".vs">;
}

multiclass VALU_MV_Mask<string opcodestr, bits<6> funct6, string vm = "v"> {
  def M : VALUVVNoVm<funct6, OPMVV, opcodestr # "." # vm # "m">;
}

multiclass VALU_MV_X<string opcodestr, bits<6> funct6, string vw = "v"> {
  def X  : VALUVX<funct6, OPMVX, opcodestr # "." # vw # "x">;
}

multiclass VALUr_MV_V_X<string opcodestr, bits<6> funct6, string vw = "v"> {
  def V : VALUrVV<funct6, OPMVV, opcodestr # "." # vw # "v">;
  def X : VALUrVX<funct6, OPMVX, opcodestr # "." # vw # "x">;
}

multiclass VALUr_MV_X<string opcodestr, bits<6> funct6, string vw = "v"> {
  def X : VALUrVX<funct6, OPMVX, opcodestr # "." # vw # "x">;
}

multiclass VALU_MV_VS2<string opcodestr, bits<6> funct6, bits<5> vs1> {
  def "" : VALUVs2<funct6, vs1, OPMVV, opcodestr>;
}

multiclass VALUm_IV_V_X_I<string opcodestr, bits<6> funct6> {
  def VM : VALUmVV<funct6, OPIVV, opcodestr # ".vvm">;
  def XM : VALUmVX<funct6, OPIVX, opcodestr # ".vxm">;
  def IM : VALUmVI<funct6, opcodestr # ".vim">;
}

multiclass VALUm_IV_V_X<string opcodestr, bits<6> funct6> {
  def VM : VALUmVV<funct6, OPIVV, opcodestr # ".vvm">;
  def XM : VALUmVX<funct6, OPIVX, opcodestr # ".vxm">;
}

multiclass VALUNoVm_IV_V_X_I<string opcodestr, bits<6> funct6, Operand optype = simm5> {
  def V : VALUVVNoVm<funct6, OPIVV, opcodestr # ".vv">;
  def X : VALUVXNoVm<funct6, OPIVX, opcodestr # ".vx">;
  def I : VALUVINoVm<funct6, opcodestr # ".vi", optype>;
}

multiclass VALUNoVm_IV_V_X<string opcodestr, bits<6> funct6> {
  def V : VALUVVNoVm<funct6, OPIVV, opcodestr # ".vv">;
  def X : VALUVXNoVm<funct6, OPIVX, opcodestr # ".vx">;
}

multiclass VALU_FV_V_F<string opcodestr, bits<6> funct6, string vw = "v"> {
  def V : VALUVV<funct6, OPFVV, opcodestr # "." # vw # "v">;
  def F : VALUVF<funct6, OPFVF, opcodestr # "." # vw # "f">;
}

multiclass VALU_FV_F<string opcodestr, bits<6> funct6, string vw = "v"> {
  def F : VALUVF<funct6, OPFVF, opcodestr # "." # vw # "f">;
}

multiclass VALUr_FV_V_F<string opcodestr, bits<6> funct6, string vw = "v"> {
  def V : VALUrVV<funct6, OPFVV, opcodestr # "." # vw # "v">;
  def F : VALUrVF<funct6, OPFVF, opcodestr # "." # vw # "f">;
}

multiclass VALU_FV_V<string opcodestr, bits<6> funct6> {
  def _VS : VALUVV<funct6, OPFVV, opcodestr # ".vs">;
}

multiclass VALU_FV_VS2<string opcodestr, bits<6> funct6, bits<5> vs1> {
  def "" : VALUVs2<funct6, vs1, OPFVV, opcodestr>;
}

multiclass VAMO<RISCVAMOOP amoop, RISCVWidth width, string opcodestr> {
  def _WD : VAMOWd<amoop, width, opcodestr>;
  def _UNWD : VAMONoWd<amoop, width, opcodestr>;
}

multiclass VWholeLoad<bits<3> nf, string opcodestr, RegisterClass VRC> {
  def E8_V : VWholeLoad<nf, LSWidth8, opcodestr # "e8.v", VRC>;
  def E16_V : VWholeLoad<nf, LSWidth16, opcodestr # "e16.v", VRC>;
  def E32_V : VWholeLoad<nf, LSWidth32, opcodestr # "e32.v", VRC>;
  def E64_V : VWholeLoad<nf, LSWidth64, opcodestr # "e64.v", VRC>;
}

//===----------------------------------------------------------------------===//
// Instructions
//===----------------------------------------------------------------------===//

let Predicates = [HasStdExtV] in {
let hasSideEffects = 1, mayLoad = 0, mayStore = 0 in {
def VSETVLI : RVInstSetVLi<(outs GPR:$rd), (ins GPR:$rs1, VTypeIOp:$vtypei),
                           "vsetvli", "$rd, $rs1, $vtypei">;

def VSETIVLI : RVInstSetiVLi<(outs GPR:$rd), (ins uimm5:$uimm, VTypeIOp:$vtypei),
                             "vsetivli", "$rd, $uimm, $vtypei">;

def VSETVL : RVInstSetVL<(outs GPR:$rd), (ins GPR:$rs1, GPR:$rs2),
                         "vsetvl", "$rd, $rs1, $rs2">;
} // hasSideEffects = 1, mayLoad = 0, mayStore = 0

// Vector Unit-Stride Instructions
def VLE8_V : VUnitStrideLoad<LUMOPUnitStride, LSWidth8, "vle8.v">;
def VLE16_V : VUnitStrideLoad<LUMOPUnitStride, LSWidth16, "vle16.v">;
def VLE32_V : VUnitStrideLoad<LUMOPUnitStride, LSWidth32, "vle32.v">;
def VLE64_V : VUnitStrideLoad<LUMOPUnitStride, LSWidth64, "vle64.v">;

def VLE8FF_V : VUnitStrideLoad<LUMOPUnitStrideFF, LSWidth8, "vle8ff.v">;
def VLE16FF_V : VUnitStrideLoad<LUMOPUnitStrideFF, LSWidth16, "vle16ff.v">;
def VLE32FF_V : VUnitStrideLoad<LUMOPUnitStrideFF, LSWidth32, "vle32ff.v">;
def VLE64FF_V : VUnitStrideLoad<LUMOPUnitStrideFF, LSWidth64, "vle64ff.v">;

def VLE1_V : VUnitStrideLoadMask<"vle1.v">;
def VSE1_V : VUnitStrideStoreMask<"vse1.v">;

def VSE8_V : VUnitStrideStore<SUMOPUnitStride, LSWidth8, "vse8.v">;
def VSE16_V : VUnitStrideStore<SUMOPUnitStride, LSWidth16, "vse16.v">;
def VSE32_V : VUnitStrideStore<SUMOPUnitStride, LSWidth32, "vse32.v">;
def VSE64_V : VUnitStrideStore<SUMOPUnitStride, LSWidth64, "vse64.v">;

// Vector Strided Instructions
def VLSE8_V : VStridedLoad<LSWidth8, "vlse8.v">;
def VLSE16_V : VStridedLoad<LSWidth16, "vlse16.v">;
def VLSE32_V : VStridedLoad<LSWidth32, "vlse32.v">;
def VLSE64_V : VStridedLoad<LSWidth64, "vlse64.v">;

def VSSE8_V : VStridedStore<LSWidth8, "vsse8.v">;
def VSSE16_V : VStridedStore<LSWidth16, "vsse16.v">;
def VSSE32_V : VStridedStore<LSWidth32, "vsse32.v">;
def VSSE64_V : VStridedStore<LSWidth64, "vsse64.v">;

// Vector Indexed Instructions
def VLUXEI8_V : VIndexedLoad<MOPLDIndexedUnord, LSWidth8, "vluxei8.v">;
def VLUXEI16_V : VIndexedLoad<MOPLDIndexedUnord, LSWidth16, "vluxei16.v">;
def VLUXEI32_V : VIndexedLoad<MOPLDIndexedUnord, LSWidth32, "vluxei32.v">;
def VLUXEI64_V : VIndexedLoad<MOPLDIndexedUnord, LSWidth64, "vluxei64.v">;

def VLOXEI8_V : VIndexedLoad<MOPLDIndexedOrder, LSWidth8, "vloxei8.v">;
def VLOXEI16_V : VIndexedLoad<MOPLDIndexedOrder, LSWidth16, "vloxei16.v">;
def VLOXEI32_V : VIndexedLoad<MOPLDIndexedOrder, LSWidth32, "vloxei32.v">;
def VLOXEI64_V : VIndexedLoad<MOPLDIndexedOrder, LSWidth64, "vloxei64.v">;

def VSUXEI8_V : VIndexedStore<MOPSTIndexedUnord, LSWidth8, "vsuxei8.v">;
def VSUXEI16_V : VIndexedStore<MOPSTIndexedUnord, LSWidth16, "vsuxei16.v">;
def VSUXEI32_V : VIndexedStore<MOPSTIndexedUnord, LSWidth32, "vsuxei32.v">;
def VSUXEI64_V : VIndexedStore<MOPSTIndexedUnord, LSWidth64, "vsuxei64.v">;

def VSOXEI8_V : VIndexedStore<MOPSTIndexedOrder, LSWidth8, "vsoxei8.v">;
def VSOXEI16_V : VIndexedStore<MOPSTIndexedOrder, LSWidth16, "vsoxei16.v">;
def VSOXEI32_V : VIndexedStore<MOPSTIndexedOrder, LSWidth32, "vsoxei32.v">;
def VSOXEI64_V : VIndexedStore<MOPSTIndexedOrder, LSWidth64, "vsoxei64.v">;

defm VL1R : VWholeLoad<0, "vl1r", VR>;
defm VL2R : VWholeLoad<1, "vl2r", VRM2>;
defm VL4R : VWholeLoad<3, "vl4r", VRM4>;
defm VL8R : VWholeLoad<7, "vl8r", VRM8>;
def : InstAlias<"vl1r.v $vd, (${rs1})", (VL1RE8_V VR:$vd, GPR:$rs1)>;
def : InstAlias<"vl2r.v $vd, (${rs1})", (VL2RE8_V VRM2:$vd, GPR:$rs1)>;
def : InstAlias<"vl4r.v $vd, (${rs1})", (VL4RE8_V VRM4:$vd, GPR:$rs1)>;
def : InstAlias<"vl8r.v $vd, (${rs1})", (VL8RE8_V VRM8:$vd, GPR:$rs1)>;

def VS1R_V : VWholeStore<0, "vs1r.v", VR>;
def VS2R_V : VWholeStore<1, "vs2r.v", VRM2>;
def VS4R_V : VWholeStore<3, "vs4r.v", VRM4>;
def VS8R_V : VWholeStore<7, "vs8r.v", VRM8>;

// Vector Single-Width Integer Add and Subtract
defm VADD_V : VALU_IV_V_X_I<"vadd", 0b000000>;
defm VSUB_V : VALU_IV_V_X<"vsub", 0b000010>;
defm VRSUB_V : VALU_IV_X_I<"vrsub", 0b000011>;

def : InstAlias<"vneg.v $vd, $vs$vm", (VRSUB_VX VR:$vd, VR:$vs, X0, VMaskOp:$vm)>;

// Vector Widening Integer Add/Subtract
// Refer to 11.2 Widening Vector Arithmetic Instructions
// The destination vector register group cannot overlap a source vector
// register group of a different element width (including the mask register
// if masked), otherwise an illegal instruction exception is raised.
let Constraints = "@earlyclobber $vd" in {
let RVVConstraint = WidenV in {
defm VWADDU_V : VALU_MV_V_X<"vwaddu", 0b110000>;
defm VWSUBU_V : VALU_MV_V_X<"vwsubu", 0b110010>;
defm VWADD_V : VALU_MV_V_X<"vwadd", 0b110001>;
defm VWSUB_V : VALU_MV_V_X<"vwsub", 0b110011>;
} // RVVConstraint = WidenV
// Set earlyclobber for following instructions for second and mask operands.
// This has the downside that the earlyclobber constraint is too coarse and
// will impose unnecessary restrictions by not allowing the destination to
// overlap with the first (wide) operand.
let RVVConstraint = WidenW in {
defm VWADDU_W : VALU_MV_V_X<"vwaddu", 0b110100, "w">;
defm VWSUBU_W : VALU_MV_V_X<"vwsubu", 0b110110, "w">;
defm VWADD_W : VALU_MV_V_X<"vwadd", 0b110101, "w">;
defm VWSUB_W : VALU_MV_V_X<"vwsub", 0b110111, "w">;
} // RVVConstraint = WidenW
} // Constraints = "@earlyclobber $vd"

def : InstAlias<"vwcvt.x.x.v $vd, $vs$vm",
                (VWADD_VX VR:$vd, VR:$vs, X0, VMaskOp:$vm)>;
def : InstAlias<"vwcvtu.x.x.v $vd, $vs$vm",
                (VWADDU_VX VR:$vd, VR:$vs, X0, VMaskOp:$vm)>;

// Vector Integer Extension
defm VZEXT_VF8 : VALU_MV_VS2<"vzext.vf8", 0b010010, 0b00010>;
defm VSEXT_VF8 : VALU_MV_VS2<"vsext.vf8", 0b010010, 0b00011>;
defm VZEXT_VF4 : VALU_MV_VS2<"vzext.vf4", 0b010010, 0b00100>;
defm VSEXT_VF4 : VALU_MV_VS2<"vsext.vf4", 0b010010, 0b00101>;
defm VZEXT_VF2 : VALU_MV_VS2<"vzext.vf2", 0b010010, 0b00110>;
defm VSEXT_VF2 : VALU_MV_VS2<"vsext.vf2", 0b010010, 0b00111>;

// Vector Integer Add-with-Carry / Subtract-with-Borrow Instructions
defm VADC_V : VALUm_IV_V_X_I<"vadc", 0b010000>;
let Constraints = "@earlyclobber $vd", RVVConstraint = NoConstraint in {
defm VMADC_V : VALUm_IV_V_X_I<"vmadc", 0b010001>;
defm VMADC_V : VALUNoVm_IV_V_X_I<"vmadc", 0b010001>;
} // Constraints = "@earlyclobber $vd", RVVConstraint = NoConstraint
defm VSBC_V : VALUm_IV_V_X<"vsbc", 0b010010>;
let Constraints = "@earlyclobber $vd", RVVConstraint = NoConstraint in {
defm VMSBC_V : VALUm_IV_V_X<"vmsbc", 0b010011>;
defm VMSBC_V : VALUNoVm_IV_V_X<"vmsbc", 0b010011>;
} // Constraints = "@earlyclobber $vd", RVVConstraint = NoConstraint

// Vector Bitwise Logical Instructions
defm VAND_V : VALU_IV_V_X_I<"vand", 0b001001>;
defm VOR_V : VALU_IV_V_X_I<"vor", 0b001010>;
defm VXOR_V : VALU_IV_V_X_I<"vxor", 0b001011>;

def : InstAlias<"vnot.v $vd, $vs$vm",
                (VXOR_VI VR:$vd, VR:$vs, -1, VMaskOp:$vm)>;

// Vector Single-Width Bit Shift Instructions
defm VSLL_V : VALU_IV_V_X_I<"vsll", 0b100101, uimm5>;
defm VSRL_V : VALU_IV_V_X_I<"vsrl", 0b101000, uimm5>;
defm VSRA_V : VALU_IV_V_X_I<"vsra", 0b101001, uimm5>;

// Vector Narrowing Integer Right Shift Instructions
// Refer to 11.3. Narrowing Vector Arithmetic Instructions
// The destination vector register group cannot overlap the first source
// vector register group (specified by vs2). The destination vector register
// group cannot overlap the mask register if used, unless LMUL=1.
let Constraints = "@earlyclobber $vd" in {
defm VNSRL_W : VALU_IV_V_X_I<"vnsrl", 0b101100, uimm5, "w">;
defm VNSRA_W : VALU_IV_V_X_I<"vnsra", 0b101101, uimm5, "w">;
} // Constraints = "@earlyclobber $vd"

def : InstAlias<"vncvt.x.x.w $vd, $vs$vm",
                (VNSRL_WX VR:$vd, VR:$vs, X0, VMaskOp:$vm)>;

// Vector Integer Comparison Instructions
let RVVConstraint = NoConstraint in {
defm VMSEQ_V : VALU_IV_V_X_I<"vmseq", 0b011000>;
defm VMSNE_V : VALU_IV_V_X_I<"vmsne", 0b011001>;
defm VMSLTU_V : VALU_IV_V_X<"vmsltu", 0b011010>;
defm VMSLT_V : VALU_IV_V_X<"vmslt", 0b011011>;
defm VMSLEU_V : VALU_IV_V_X_I<"vmsleu", 0b011100>;
defm VMSLE_V : VALU_IV_V_X_I<"vmsle", 0b011101>;
defm VMSGTU_V : VALU_IV_X_I<"vmsgtu", 0b011110>;
defm VMSGT_V : VALU_IV_X_I<"vmsgt", 0b011111>;
} // RVVConstraint = NoConstraint

def : InstAlias<"vmsgtu.vv $vd, $va, $vb$vm",
                (VMSLTU_VV VR:$vd, VR:$vb, VR:$va, VMaskOp:$vm), 0>;
def : InstAlias<"vmsgt.vv $vd, $va, $vb$vm",
                (VMSLT_VV VR:$vd, VR:$vb, VR:$va, VMaskOp:$vm), 0>;
def : InstAlias<"vmsgeu.vv $vd, $va, $vb$vm",
                (VMSLEU_VV VR:$vd, VR:$vb, VR:$va, VMaskOp:$vm), 0>;
def : InstAlias<"vmsge.vv $vd, $va, $vb$vm",
                (VMSLE_VV VR:$vd, VR:$vb, VR:$va, VMaskOp:$vm), 0>;

let isCodeGenOnly = 0, isAsmParserOnly = 1, hasSideEffects = 0, mayLoad = 0,
    mayStore = 0 in {
// For unsigned comparisons we need to special case 0 immediate to maintain
// the always true/false semantics we would invert if we just decremented the
// immediate like we do for signed. To match the GNU assembler we will use
// vmseq/vmsne.vv with the same register for both operands which we can't do
// from an InstAlias.
def PseudoVMSGEU_VI : Pseudo<(outs VR:$vd),
                             (ins VR:$vs2, simm5_plus1:$imm, VMaskOp:$vm),
                             [], "vmsgeu.vi", "$vd, $vs2, $imm$vm">;
def PseudoVMSLTU_VI : Pseudo<(outs VR:$vd),
                             (ins VR:$vs2, simm5_plus1:$imm, VMaskOp:$vm),
                             [], "vmsltu.vi", "$vd, $vs2, $imm$vm">;
// Handle signed with pseudos as well for more consistency in the
// implementation.
def PseudoVMSGE_VI : Pseudo<(outs VR:$vd),
                            (ins VR:$vs2, simm5_plus1:$imm, VMaskOp:$vm),
                            [], "vmsge.vi", "$vd, $vs2, $imm$vm">;
def PseudoVMSLT_VI : Pseudo<(outs VR:$vd),
                            (ins VR:$vs2, simm5_plus1:$imm, VMaskOp:$vm),
                            [], "vmslt.vi", "$vd, $vs2, $imm$vm">;
}

let isCodeGenOnly = 0, isAsmParserOnly = 1, hasSideEffects = 0, mayLoad = 0,
    mayStore = 0 in {
def PseudoVMSGEU_VX : Pseudo<(outs VR:$vd),
                             (ins VR:$vs2, GPR:$rs1),
                             [], "vmsgeu.vx", "$vd, $vs2, $rs1">;
def PseudoVMSGE_VX : Pseudo<(outs VR:$vd),
                            (ins VR:$vs2, GPR:$rs1),
                            [], "vmsge.vx", "$vd, $vs2, $rs1">;
def PseudoVMSGEU_VX_M : Pseudo<(outs VRNoV0:$vd),
                               (ins VR:$vs2, GPR:$rs1, VMaskOp:$vm),
                               [], "vmsgeu.vx", "$vd, $vs2, $rs1$vm">;
def PseudoVMSGE_VX_M : Pseudo<(outs VRNoV0:$vd),
                              (ins VR:$vs2, GPR:$rs1, VMaskOp:$vm),
                              [], "vmsge.vx", "$vd, $vs2, $rs1$vm">;
def PseudoVMSGEU_VX_M_T : Pseudo<(outs VMV0:$vd, VR:$scratch),
                                 (ins VR:$vs2, GPR:$rs1, VMaskOp:$vm),
                                 [], "vmsgeu.vx", "$vd, $vs2, $rs1$vm, $scratch">;
def PseudoVMSGE_VX_M_T : Pseudo<(outs VMV0:$vd, VR:$scratch),
                                (ins VR:$vs2, GPR:$rs1, VMaskOp:$vm),
                                [], "vmsge.vx", "$vd, $vs2, $rs1$vm, $scratch">;
}

// Vector Integer Min/Max Instructions
defm VMINU_V : VALU_IV_V_X<"vminu", 0b000100>;
defm VMIN_V : VALU_IV_V_X<"vmin", 0b000101>;
defm VMAXU_V : VALU_IV_V_X<"vmaxu", 0b000110>;
defm VMAX_V : VALU_IV_V_X<"vmax", 0b000111>;

// Vector Single-Width Integer Multiply Instructions
defm VMUL_V : VALU_MV_V_X<"vmul", 0b100101>;
defm VMULH_V : VALU_MV_V_X<"vmulh", 0b100111>;
defm VMULHU_V : VALU_MV_V_X<"vmulhu", 0b100100>;
defm VMULHSU_V : VALU_MV_V_X<"vmulhsu", 0b100110>;

// Vector Integer Divide Instructions
defm VDIVU_V : VALU_MV_V_X<"vdivu", 0b100000>;
defm VDIV_V : VALU_MV_V_X<"vdiv", 0b100001>;
defm VREMU_V : VALU_MV_V_X<"vremu", 0b100010>;
defm VREM_V : VALU_MV_V_X<"vrem", 0b100011>;

// Vector Widening Integer Multiply Instructions
let Constraints = "@earlyclobber $vd", RVVConstraint = WidenV in {
defm VWMUL_V : VALU_MV_V_X<"vwmul", 0b111011>;
defm VWMULU_V : VALU_MV_V_X<"vwmulu", 0b111000>;
defm VWMULSU_V : VALU_MV_V_X<"vwmulsu", 0b111010>;
} // Constraints = "@earlyclobber $vd", RVVConstraint = WidenV

// Vector Single-Width Integer Multiply-Add Instructions
defm VMACC_V : VALUr_MV_V_X<"vmacc", 0b101101>;
defm VNMSAC_V : VALUr_MV_V_X<"vnmsac", 0b101111>;
defm VMADD_V : VALUr_MV_V_X<"vmadd", 0b101001>;
defm VNMSUB_V : VALUr_MV_V_X<"vnmsub", 0b101011>;

// Vector Widening Integer Multiply-Add Instructions
let Constraints = "@earlyclobber $vd", RVVConstraint = WidenV in {
defm VWMACCU_V : VALUr_MV_V_X<"vwmaccu", 0b111100>;
defm VWMACC_V : VALUr_MV_V_X<"vwmacc", 0b111101>;
defm VWMACCSU_V : VALUr_MV_V_X<"vwmaccsu", 0b111111>;
defm VWMACCUS_V : VALUr_MV_X<"vwmaccus", 0b111110>;
} // Constraints = "@earlyclobber $vd", RVVConstraint = WidenV

// Vector Integer Merge Instructions
defm VMERGE_V : VALUm_IV_V_X_I<"vmerge", 0b010111>;

// Vector Integer Move Instructions
let hasSideEffects = 0, mayLoad = 0, mayStore = 0, vs2 = 0, vm = 1,
    RVVConstraint = NoConstraint  in {
// op vd, vs1
def VMV_V_V : RVInstVV<0b010111, OPIVV, (outs VR:$vd),
                       (ins VR:$vs1), "vmv.v.v", "$vd, $vs1">;
// op vd, rs1
def VMV_V_X : RVInstVX<0b010111, OPIVX, (outs VR:$vd),
                       (ins GPR:$rs1), "vmv.v.x", "$vd, $rs1">;
// op vd, imm
def VMV_V_I : RVInstIVI<0b010111, (outs VR:$vd),
                       (ins simm5:$imm), "vmv.v.i", "$vd, $imm">;
} // hasSideEffects = 0, mayLoad = 0, mayStore = 0

// Vector Fixed-Point Arithmetic Instructions
defm VSADDU_V : VALU_IV_V_X_I<"vsaddu", 0b100000>;
defm VSADD_V : VALU_IV_V_X_I<"vsadd", 0b100001>;
defm VSSUBU_V : VALU_IV_V_X<"vssubu", 0b100010>;
defm VSSUB_V : VALU_IV_V_X<"vssub", 0b100011>;

// Vector Single-Width Averaging Add and Subtract
defm VAADDU_V : VALU_MV_V_X<"vaaddu", 0b001000>;
defm VAADD_V : VALU_MV_V_X<"vaadd", 0b001001>;
defm VASUBU_V : VALU_MV_V_X<"vasubu", 0b001010>;
defm VASUB_V : VALU_MV_V_X<"vasub", 0b001011>;

// Vector Single-Width Fractional Multiply with Rounding and Saturation
defm VSMUL_V : VALU_IV_V_X<"vsmul", 0b100111>;

// Vector Single-Width Scaling Shift Instructions
defm VSSRL_V : VALU_IV_V_X_I<"vssrl", 0b101010, uimm5>;
defm VSSRA_V : VALU_IV_V_X_I<"vssra", 0b101011, uimm5>;

// Vector Narrowing Fixed-Point Clip Instructions
let Constraints = "@earlyclobber $vd" in {
defm VNCLIPU_W : VALU_IV_V_X_I<"vnclipu", 0b101110, uimm5, "w">;
defm VNCLIP_W : VALU_IV_V_X_I<"vnclip", 0b101111, uimm5, "w">;
} // Constraints = "@earlyclobber $vd"
} // Predicates = [HasStdExtV]

let Predicates = [HasStdExtV, HasStdExtF] in {
// Vector Single-Width Floating-Point Add/Subtract Instructions
defm VFADD_V : VALU_FV_V_F<"vfadd", 0b000000>;
defm VFSUB_V : VALU_FV_V_F<"vfsub", 0b000010>;
defm VFRSUB_V : VALU_FV_F<"vfrsub", 0b100111>;

// Vector Widening Floating-Point Add/Subtract Instructions
let Constraints = "@earlyclobber $vd" in {
let RVVConstraint = WidenV in {
defm VFWADD_V : VALU_FV_V_F<"vfwadd", 0b110000>;
defm VFWSUB_V : VALU_FV_V_F<"vfwsub", 0b110010>;
} // RVVConstraint = WidenV
// Set earlyclobber for following instructions for second and mask operands.
// This has the downside that the earlyclobber constraint is too coarse and
// will impose unnecessary restrictions by not allowing the destination to
// overlap with the first (wide) operand.
let RVVConstraint = WidenW in {
defm VFWADD_W : VALU_FV_V_F<"vfwadd", 0b110100, "w">;
defm VFWSUB_W : VALU_FV_V_F<"vfwsub", 0b110110, "w">;
} // RVVConstraint = WidenW
} // Constraints = "@earlyclobber $vd"

// Vector Single-Width Floating-Point Multiply/Divide Instructions
defm VFMUL_V : VALU_FV_V_F<"vfmul", 0b100100>;
defm VFDIV_V : VALU_FV_V_F<"vfdiv", 0b100000>;
defm VFRDIV_V : VALU_FV_F<"vfrdiv", 0b100001>;

// Vector Widening Floating-Point Multiply
let Constraints = "@earlyclobber $vd", RVVConstraint = WidenV in {
defm VFWMUL_V : VALU_FV_V_F<"vfwmul", 0b111000>;
} // Constraints = "@earlyclobber $vd", RVVConstraint = WidenV

// Vector Single-Width Floating-Point Fused Multiply-Add Instructions
defm VFMACC_V : VALUr_FV_V_F<"vfmacc", 0b101100>;
defm VFNMACC_V : VALUr_FV_V_F<"vfnmacc", 0b101101>;
defm VFMSAC_V : VALUr_FV_V_F<"vfmsac", 0b101110>;
defm VFNMSAC_V : VALUr_FV_V_F<"vfnmsac", 0b101111>;
defm VFMADD_V : VALUr_FV_V_F<"vfmadd", 0b101000>;
defm VFNMADD_V : VALUr_FV_V_F<"vfnmadd", 0b101001>;
defm VFMSUB_V : VALUr_FV_V_F<"vfmsub", 0b101010>;
defm VFNMSUB_V : VALUr_FV_V_F<"vfnmsub", 0b101011>;

// Vector Widening Floating-Point Fused Multiply-Add Instructions
let Constraints = "@earlyclobber $vd", RVVConstraint = WidenV in {
defm VFWMACC_V : VALUr_FV_V_F<"vfwmacc", 0b111100>;
defm VFWNMACC_V : VALUr_FV_V_F<"vfwnmacc", 0b111101>;
defm VFWMSAC_V : VALUr_FV_V_F<"vfwmsac", 0b111110>;
defm VFWNMSAC_V : VALUr_FV_V_F<"vfwnmsac", 0b111111>;
} // Constraints = "@earlyclobber $vd", RVVConstraint = WidenV

// Vector Floating-Point Square-Root Instruction
defm VFSQRT_V : VALU_FV_VS2<"vfsqrt.v", 0b010011, 0b00000>;
defm VFRSQRT7_V : VALU_FV_VS2<"vfrsqrt7.v", 0b010011, 0b00100>;
defm VFREC7_V : VALU_FV_VS2<"vfrec7.v", 0b010011, 0b00101>;

// Vector Floating-Point MIN/MAX Instructions
defm VFMIN_V : VALU_FV_V_F<"vfmin", 0b000100>;
defm VFMAX_V : VALU_FV_V_F<"vfmax", 0b000110>;

// Vector Floating-Point Sign-Injection Instructions
defm VFSGNJ_V : VALU_FV_V_F<"vfsgnj", 0b001000>;
defm VFSGNJN_V : VALU_FV_V_F<"vfsgnjn", 0b001001>;
defm VFSGNJX_V : VALU_FV_V_F<"vfsgnjx", 0b001010>;

def : InstAlias<"vfneg.v $vd, $vs$vm",
                (VFSGNJN_VV VR:$vd, VR:$vs, VR:$vs, VMaskOp:$vm)>;
def : InstAlias<"vfabs.v $vd, $vs$vm",
                (VFSGNJX_VV VR:$vd, VR:$vs, VR:$vs, VMaskOp:$vm)>;

// Vector Floating-Point Compare Instructions
let RVVConstraint = NoConstraint in {
defm VMFEQ_V : VALU_FV_V_F<"vmfeq", 0b011000>;
defm VMFNE_V : VALU_FV_V_F<"vmfne", 0b011100>;
defm VMFLT_V : VALU_FV_V_F<"vmflt", 0b011011>;
defm VMFLE_V : VALU_FV_V_F<"vmfle", 0b011001>;
defm VMFGT_V : VALU_FV_F<"vmfgt", 0b011101>;
defm VMFGE_V : VALU_FV_F<"vmfge", 0b011111>;
} // RVVConstraint = NoConstraint

def : InstAlias<"vmfgt.vv $vd, $va, $vb$vm",
                (VMFLT_VV VR:$vd, VR:$vb, VR:$va, VMaskOp:$vm), 0>;
def : InstAlias<"vmfge.vv $vd, $va, $vb$vm",
                (VMFLE_VV VR:$vd, VR:$vb, VR:$va, VMaskOp:$vm), 0>;

// Vector Floating-Point Classify Instruction
defm VFCLASS_V : VALU_FV_VS2<"vfclass.v", 0b010011, 0b10000>;

let hasSideEffects = 0, mayLoad = 0, mayStore = 0 in {
// Vector Floating-Point Merge Instruction
def VFMERGE_VFM : RVInstVX<0b010111, OPFVF, (outs VR:$vd),
                           (ins VR:$vs2, FPR32:$rs1, VMV0:$v0),
                           "vfmerge.vfm", "$vd, $vs2, $rs1, v0"> {
  let vm = 0;
}

// Vector Floating-Point Move Instruction
let RVVConstraint = NoConstraint in
def VFMV_V_F : RVInstVX<0b010111, OPFVF, (outs VR:$vd),
                       (ins FPR32:$rs1), "vfmv.v.f", "$vd, $rs1"> {
  let vs2 = 0;
  let vm = 1;
}
} // hasSideEffects = 0, mayLoad = 0, mayStore = 0

// Single-Width Floating-Point/Integer Type-Convert Instructions
defm VFCVT_XU_F_V : VALU_FV_VS2<"vfcvt.xu.f.v", 0b010010, 0b00000>;
defm VFCVT_X_F_V : VALU_FV_VS2<"vfcvt.x.f.v", 0b010010, 0b00001>;
defm VFCVT_RTZ_XU_F_V : VALU_FV_VS2<"vfcvt.rtz.xu.f.v", 0b010010, 0b00110>;
defm VFCVT_RTZ_X_F_V : VALU_FV_VS2<"vfcvt.rtz.x.f.v", 0b010010, 0b00111>;
defm VFCVT_F_XU_V : VALU_FV_VS2<"vfcvt.f.xu.v", 0b010010, 0b00010>;
defm VFCVT_F_X_V : VALU_FV_VS2<"vfcvt.f.x.v", 0b010010, 0b00011>;

// Widening Floating-Point/Integer Type-Convert Instructions
let Constraints = "@earlyclobber $vd", RVVConstraint = WidenCvt in {
defm VFWCVT_XU_F_V : VALU_FV_VS2<"vfwcvt.xu.f.v", 0b010010, 0b01000>;
defm VFWCVT_X_F_V : VALU_FV_VS2<"vfwcvt.x.f.v", 0b010010, 0b01001>;
defm VFWCVT_RTZ_XU_F_V : VALU_FV_VS2<"vfwcvt.rtz.xu.f.v", 0b010010, 0b01110>;
defm VFWCVT_RTZ_X_F_V : VALU_FV_VS2<"vfwcvt.rtz.x.f.v", 0b010010, 0b01111>;
defm VFWCVT_F_XU_V : VALU_FV_VS2<"vfwcvt.f.xu.v", 0b010010, 0b01010>;
defm VFWCVT_F_X_V : VALU_FV_VS2<"vfwcvt.f.x.v", 0b010010, 0b01011>;
defm VFWCVT_F_F_V : VALU_FV_VS2<"vfwcvt.f.f.v", 0b010010, 0b01100>;
} // Constraints = "@earlyclobber $vd", RVVConstraint = WidenCvt

// Narrowing Floating-Point/Integer Type-Convert Instructions
let Constraints = "@earlyclobber $vd" in {
defm VFNCVT_XU_F_W : VALU_FV_VS2<"vfncvt.xu.f.w", 0b010010, 0b10000>;
defm VFNCVT_X_F_W : VALU_FV_VS2<"vfncvt.x.f.w", 0b010010, 0b10001>;
defm VFNCVT_RTZ_XU_F_W : VALU_FV_VS2<"vfncvt.rtz.xu.f.w", 0b010010, 0b10110>;
defm VFNCVT_RTZ_X_F_W : VALU_FV_VS2<"vfncvt.rtz.x.f.w", 0b010010, 0b10111>;
defm VFNCVT_F_XU_W : VALU_FV_VS2<"vfncvt.f.xu.w", 0b010010, 0b10010>;
defm VFNCVT_F_X_W : VALU_FV_VS2<"vfncvt.f.x.w", 0b010010, 0b10011>;
defm VFNCVT_F_F_W : VALU_FV_VS2<"vfncvt.f.f.w", 0b010010, 0b10100>;
defm VFNCVT_ROD_F_F_W : VALU_FV_VS2<"vfncvt.rod.f.f.w", 0b010010, 0b10101>;
} // Constraints = "@earlyclobber $vd"
} // Predicates = [HasStdExtV, HasStdExtF]

let Predicates = [HasStdExtV] in {
// Vector Single-Width Integer Reduction Instructions
let RVVConstraint = NoConstraint in {
defm VREDSUM : VALU_MV_V<"vredsum", 0b000000>;
defm VREDMAXU : VALU_MV_V<"vredmaxu", 0b000110>;
defm VREDMAX : VALU_MV_V<"vredmax", 0b000111>;
defm VREDMINU : VALU_MV_V<"vredminu", 0b000100>;
defm VREDMIN : VALU_MV_V<"vredmin", 0b000101>;
defm VREDAND : VALU_MV_V<"vredand", 0b000001>;
defm VREDOR : VALU_MV_V<"vredor", 0b000010>;
defm VREDXOR : VALU_MV_V<"vredxor", 0b000011>;
} // RVVConstraint = NoConstraint

// Vector Widening Integer Reduction Instructions
let Constraints = "@earlyclobber $vd", RVVConstraint = NoConstraint in {
// Set earlyclobber for following instructions for second and mask operands.
// This has the downside that the earlyclobber constraint is too coarse and
// will impose unnecessary restrictions by not allowing the destination to
// overlap with the first (wide) operand.
defm VWREDSUMU : VALU_IV_V<"vwredsumu", 0b110000>;
defm VWREDSUM : VALU_IV_V<"vwredsum", 0b110001>;
} // Constraints = "@earlyclobber $vd", RVVConstraint = NoConstraint
} // Predicates = [HasStdExtV]

let Predicates = [HasStdExtV, HasStdExtF] in {
// Vector Single-Width Floating-Point Reduction Instructions
let RVVConstraint = NoConstraint in {
defm VFREDOSUM : VALU_FV_V<"vfredosum", 0b000011>;
defm VFREDSUM : VALU_FV_V<"vfredsum", 0b000001>;
defm VFREDMAX : VALU_FV_V<"vfredmax", 0b000111>;
defm VFREDMIN : VALU_FV_V<"vfredmin", 0b000101>;
} // RVVConstraint = NoConstraint

// Vector Widening Floating-Point Reduction Instructions
let Constraints = "@earlyclobber $vd", RVVConstraint = NoConstraint in {
// Set earlyclobber for following instructions for second and mask operands.
// This has the downside that the earlyclobber constraint is too coarse and
// will impose unnecessary restrictions by not allowing the destination to
// overlap with the first (wide) operand.
defm VFWREDOSUM : VALU_FV_V<"vfwredosum", 0b110011>;
defm VFWREDSUM : VALU_FV_V<"vfwredsum", 0b110001>;
} // Constraints = "@earlyclobber $vd", RVVConstraint = NoConstraint
} // Predicates = [HasStdExtV, HasStdExtF]

let Predicates = [HasStdExtV] in {
// Vector Mask-Register Logical Instructions
let RVVConstraint = NoConstraint in {
defm VMAND_M : VALU_MV_Mask<"vmand", 0b011001, "m">;
defm VMNAND_M : VALU_MV_Mask<"vmnand", 0b011101, "m">;
defm VMANDNOT_M : VALU_MV_Mask<"vmandnot", 0b011000, "m">;
defm VMXOR_M : VALU_MV_Mask<"vmxor", 0b011011, "m">;
defm VMOR_M : VALU_MV_Mask<"vmor", 0b011010, "m">;
defm VMNOR_M : VALU_MV_Mask<"vmnor", 0b011110, "m">;
defm VMORNOT_M : VALU_MV_Mask<"vmornot", 0b011100, "m">;
defm VMXNOR_M : VALU_MV_Mask<"vmxnor", 0b011111, "m">;
}

def : InstAlias<"vmmv.m $vd, $vs",
                (VMAND_MM VR:$vd, VR:$vs, VR:$vs)>;
def : InstAlias<"vmclr.m $vd",
                (VMXOR_MM VR:$vd, VR:$vd, VR:$vd)>;
def : InstAlias<"vmset.m $vd",
                (VMXNOR_MM VR:$vd, VR:$vd, VR:$vd)>;
def : InstAlias<"vmnot.m $vd, $vs",
                (VMNAND_MM VR:$vd, VR:$vs, VR:$vs)>;

let hasSideEffects = 0, mayLoad = 0, mayStore = 0,
    RVVConstraint = NoConstraint  in {
// Vector mask population count vpopc
def VPOPC_M : RVInstV<0b010000, 0b10000, OPMVV, (outs GPR:$vd),
                        (ins VR:$vs2, VMaskOp:$vm),
                        "vpopc.m", "$vd, $vs2$vm">;

// vfirst find-first-set mask bit
def VFIRST_M : RVInstV<0b010000, 0b10001, OPMVV, (outs GPR:$vd),
                        (ins VR:$vs2, VMaskOp:$vm),
                        "vfirst.m", "$vd, $vs2$vm">;
} // hasSideEffects = 0, mayLoad = 0, mayStore = 0

let Constraints = "@earlyclobber $vd", RVVConstraint = Iota in {
// vmsbf.m set-before-first mask bit
defm VMSBF_M : VALU_MV_VS2<"vmsbf.m", 0b010100, 0b00001>;
// vmsif.m set-including-first mask bit
defm VMSIF_M : VALU_MV_VS2<"vmsif.m", 0b010100, 0b00011>;
// vmsof.m set-only-first mask bit
defm VMSOF_M : VALU_MV_VS2<"vmsof.m", 0b010100, 0b00010>;
// Vector Iota Instruction
defm VIOTA_M : VALU_MV_VS2<"viota.m", 0b010100, 0b10000>;
} // Constraints = "@earlyclobber $vd", RVVConstraint = Iota

// Vector Element Index Instruction
let hasSideEffects = 0, mayLoad = 0, mayStore = 0 in {
def VID_V : RVInstV<0b010100, 0b10001, OPMVV, (outs VR:$vd),
                      (ins VMaskOp:$vm), "vid.v", "$vd$vm"> {
  let vs2 = 0;
}

// Integer Scalar Move Instructions
let vm = 1, RVVConstraint = NoConstraint in {
def VMV_X_S : RVInstV<0b010000, 0b00000, OPMVV, (outs GPR:$vd),
                      (ins VR:$vs2), "vmv.x.s", "$vd, $vs2">;
let Constraints = "$vd = $vd_wb" in
def VMV_S_X : RVInstV2<0b010000, 0b00000, OPMVX, (outs VR:$vd_wb),
                      (ins VR:$vd, GPR:$rs1), "vmv.s.x", "$vd, $rs1">;

}
} // hasSideEffects = 0, mayLoad = 0, mayStore = 0
} // Predicates = [HasStdExtV]

let Predicates = [HasStdExtV, HasStdExtF] in {
let hasSideEffects = 0, mayLoad = 0, mayStore = 0, vm = 1,
    RVVConstraint = NoConstraint  in {
// Floating-Point Scalar Move Instructions
def VFMV_F_S : RVInstV<0b010000, 0b00000, OPFVV, (outs FPR32:$vd),
                      (ins VR:$vs2), "vfmv.f.s", "$vd, $vs2">;
let Constraints = "$vd = $vd_wb" in
def VFMV_S_F : RVInstV2<0b010000, 0b00000, OPFVF, (outs VR:$vd_wb),
                      (ins VR:$vd, FPR32:$rs1), "vfmv.s.f", "$vd, $rs1">;

} // hasSideEffects = 0, mayLoad = 0, mayStore = 0, vm = 1
} // Predicates = [HasStdExtV, HasStdExtF]

let Predicates = [HasStdExtV] in {
// Vector Slide Instructions
let Constraints = "@earlyclobber $vd", RVVConstraint = SlideUp in {
defm VSLIDEUP_V : VALU_IV_X_I<"vslideup", 0b001110, uimm5>;
defm VSLIDE1UP_V : VALU_MV_X<"vslide1up", 0b001110>;
} // Constraints = "@earlyclobber $vd", RVVConstraint = SlideUp
defm VSLIDEDOWN_V : VALU_IV_X_I<"vslidedown", 0b001111, uimm5>;
defm VSLIDE1DOWN_V : VALU_MV_X<"vslide1down", 0b001111>;
} // Predicates = [HasStdExtV]

let Predicates = [HasStdExtV, HasStdExtF] in {
let Constraints = "@earlyclobber $vd", RVVConstraint = SlideUp in {
defm VFSLIDE1UP_V : VALU_FV_F<"vfslide1up", 0b001110>;
} // Constraints = "@earlyclobber $vd", RVVConstraint = SlideUp
defm VFSLIDE1DOWN_V : VALU_FV_F<"vfslide1down", 0b001111>;
} // Predicates = [HasStdExtV, HasStdExtF]

let Predicates = [HasStdExtV] in {
// Vector Register Gather Instruction
let Constraints = "@earlyclobber $vd", RVVConstraint = Vrgather in {
defm VRGATHER_V : VALU_IV_V_X_I<"vrgather", 0b001100, uimm5>;
def VRGATHEREI16_VV : VALUVV<0b001110, OPIVV, "vrgatherei16.vv">;
} // Constraints = "@earlyclobber $vd", RVVConstraint = Vrgather

// Vector Compress Instruction
let Constraints = "@earlyclobber $vd", RVVConstraint = Vcompress in {
defm VCOMPRESS_V : VALU_MV_Mask<"vcompress", 0b010111>;
} // Constraints = "@earlyclobber $vd", RVVConstraint = Vcompress

let hasSideEffects = 0, mayLoad = 0, mayStore = 0,
    RVVConstraint = NoConstraint in {
foreach nf = [1, 2, 4, 8] in {
  def VMV#nf#R_V  : RVInstV<0b100111, !add(nf, -1), OPIVI, (outs VR:$vd),
                            (ins VR:$vs2), "vmv" # nf # "r.v",
                            "$vd, $vs2"> {
    let Uses = [];
    let vm = 1;
  }
}
} // hasSideEffects = 0, mayLoad = 0, mayStore = 0
} // Predicates = [HasStdExtV]

let Predicates = [HasStdExtZvlsseg] in {
  foreach nf=2-8 in {
    def VLSEG#nf#E8_V : VUnitStrideSegmentLoad<!add(nf, -1), LUMOPUnitStride, LSWidth8, "vlseg"#nf#"e8.v">;
    def VLSEG#nf#E16_V : VUnitStrideSegmentLoad<!add(nf, -1), LUMOPUnitStride, LSWidth16, "vlseg"#nf#"e16.v">;
    def VLSEG#nf#E32_V : VUnitStrideSegmentLoad<!add(nf, -1), LUMOPUnitStride, LSWidth32, "vlseg"#nf#"e32.v">;
    def VLSEG#nf#E64_V : VUnitStrideSegmentLoad<!add(nf, -1), LUMOPUnitStride, LSWidth64, "vlseg"#nf#"e64.v">;

    def VLSEG#nf#E8FF_V : VUnitStrideSegmentLoad<!add(nf, -1), LUMOPUnitStrideFF, LSWidth8, "vlseg"#nf#"e8ff.v">;
    def VLSEG#nf#E16FF_V : VUnitStrideSegmentLoad<!add(nf, -1), LUMOPUnitStrideFF, LSWidth16, "vlseg"#nf#"e16ff.v">;
    def VLSEG#nf#E32FF_V : VUnitStrideSegmentLoad<!add(nf, -1), LUMOPUnitStrideFF, LSWidth32, "vlseg"#nf#"e32ff.v">;
    def VLSEG#nf#E64FF_V : VUnitStrideSegmentLoad<!add(nf, -1), LUMOPUnitStrideFF, LSWidth64, "vlseg"#nf#"e64ff.v">;

    def VSSEG#nf#E8_V : VUnitStrideSegmentStore<!add(nf, -1), LSWidth8, "vsseg"#nf#"e8.v">;
    def VSSEG#nf#E16_V : VUnitStrideSegmentStore<!add(nf, -1), LSWidth16, "vsseg"#nf#"e16.v">;
    def VSSEG#nf#E32_V : VUnitStrideSegmentStore<!add(nf, -1), LSWidth32, "vsseg"#nf#"e32.v">;
    def VSSEG#nf#E64_V : VUnitStrideSegmentStore<!add(nf, -1), LSWidth64, "vsseg"#nf#"e64.v">;

    // Vector Strided Instructions
    def VLSSEG#nf#E8_V : VStridedSegmentLoad<!add(nf, -1), LSWidth8, "vlsseg"#nf#"e8.v">;
    def VLSSEG#nf#E16_V : VStridedSegmentLoad<!add(nf, -1), LSWidth16, "vlsseg"#nf#"e16.v">;
    def VLSSEG#nf#E32_V : VStridedSegmentLoad<!add(nf, -1), LSWidth32, "vlsseg"#nf#"e32.v">;
    def VLSSEG#nf#E64_V : VStridedSegmentLoad<!add(nf, -1), LSWidth64, "vlsseg"#nf#"e64.v">;

    def VSSSEG#nf#E8_V : VStridedSegmentStore<!add(nf, -1), LSWidth8, "vssseg"#nf#"e8.v">;
    def VSSSEG#nf#E16_V : VStridedSegmentStore<!add(nf, -1), LSWidth16, "vssseg"#nf#"e16.v">;
    def VSSSEG#nf#E32_V : VStridedSegmentStore<!add(nf, -1), LSWidth32, "vssseg"#nf#"e32.v">;
    def VSSSEG#nf#E64_V : VStridedSegmentStore<!add(nf, -1), LSWidth64, "vssseg"#nf#"e64.v">;

    // Vector Indexed Instructions
    def VLUXSEG#nf#EI8_V : VIndexedSegmentLoad<!add(nf, -1), MOPLDIndexedUnord,
                             LSWidth8, "vluxseg"#nf#"ei8.v">;
    def VLUXSEG#nf#EI16_V : VIndexedSegmentLoad<!add(nf, -1), MOPLDIndexedUnord,
                              LSWidth16, "vluxseg"#nf#"ei16.v">;
    def VLUXSEG#nf#EI32_V : VIndexedSegmentLoad<!add(nf, -1), MOPLDIndexedUnord,
                              LSWidth32, "vluxseg"#nf#"ei32.v">;
    def VLUXSEG#nf#EI64_V : VIndexedSegmentLoad<!add(nf, -1), MOPLDIndexedUnord,
                              LSWidth64, "vluxseg"#nf#"ei64.v">;

    def VLOXSEG#nf#EI8_V : VIndexedSegmentLoad<!add(nf, -1), MOPLDIndexedOrder,
                             LSWidth8, "vloxseg"#nf#"ei8.v">;
    def VLOXSEG#nf#EI16_V : VIndexedSegmentLoad<!add(nf, -1), MOPLDIndexedOrder,
                              LSWidth16, "vloxseg"#nf#"ei16.v">;
    def VLOXSEG#nf#EI32_V : VIndexedSegmentLoad<!add(nf, -1), MOPLDIndexedOrder,
                              LSWidth32, "vloxseg"#nf#"ei32.v">;
    def VLOXSEG#nf#EI64_V : VIndexedSegmentLoad<!add(nf, -1), MOPLDIndexedOrder,
                              LSWidth64, "vloxseg"#nf#"ei64.v">;

    def VSUXSEG#nf#EI8_V : VIndexedSegmentStore<!add(nf, -1), MOPSTIndexedUnord,
                             LSWidth8, "vsuxseg"#nf#"ei8.v">;
    def VSUXSEG#nf#EI16_V : VIndexedSegmentStore<!add(nf, -1), MOPSTIndexedUnord,
                              LSWidth16, "vsuxseg"#nf#"ei16.v">;
    def VSUXSEG#nf#EI32_V : VIndexedSegmentStore<!add(nf, -1), MOPSTIndexedUnord,
                              LSWidth32, "vsuxseg"#nf#"ei32.v">;
    def VSUXSEG#nf#EI64_V : VIndexedSegmentStore<!add(nf, -1), MOPSTIndexedUnord,
                              LSWidth64, "vsuxseg"#nf#"ei64.v">;

    def VSOXSEG#nf#EI8_V : VIndexedSegmentStore<!add(nf, -1), MOPSTIndexedOrder,
                             LSWidth8, "vsoxseg"#nf#"ei8.v">;
    def VSOXSEG#nf#EI16_V : VIndexedSegmentStore<!add(nf, -1), MOPSTIndexedOrder,
                              LSWidth16, "vsoxseg"#nf#"ei16.v">;
    def VSOXSEG#nf#EI32_V : VIndexedSegmentStore<!add(nf, -1), MOPSTIndexedOrder,
                              LSWidth32, "vsoxseg"#nf#"ei32.v">;
    def VSOXSEG#nf#EI64_V : VIndexedSegmentStore<!add(nf, -1), MOPSTIndexedOrder,
                              LSWidth64, "vsoxseg"#nf#"ei64.v">;
  }
} // Predicates = [HasStdExtZvlsseg]

let Predicates = [HasStdExtZvamo, HasStdExtA] in {
  defm VAMOSWAPEI8 : VAMO<AMOOPVamoSwap, LSWidth8, "vamoswapei8.v">;
  defm VAMOSWAPEI16 : VAMO<AMOOPVamoSwap, LSWidth16, "vamoswapei16.v">;
  defm VAMOSWAPEI32 : VAMO<AMOOPVamoSwap, LSWidth32, "vamoswapei32.v">;

  defm VAMOADDEI8 : VAMO<AMOOPVamoAdd, LSWidth8, "vamoaddei8.v">;
  defm VAMOADDEI16 : VAMO<AMOOPVamoAdd, LSWidth16, "vamoaddei16.v">;
  defm VAMOADDEI32 : VAMO<AMOOPVamoAdd, LSWidth32, "vamoaddei32.v">;

  defm VAMOXOREI8 : VAMO<AMOOPVamoXor, LSWidth8, "vamoxorei8.v">;
  defm VAMOXOREI16 : VAMO<AMOOPVamoXor, LSWidth16, "vamoxorei16.v">;
  defm VAMOXOREI32 : VAMO<AMOOPVamoXor, LSWidth32, "vamoxorei32.v">;

  defm VAMOANDEI8 : VAMO<AMOOPVamoAnd, LSWidth8, "vamoandei8.v">;
  defm VAMOANDEI16 : VAMO<AMOOPVamoAnd, LSWidth16, "vamoandei16.v">;
  defm VAMOANDEI32 : VAMO<AMOOPVamoAnd, LSWidth32, "vamoandei32.v">;

  defm VAMOOREI8 : VAMO<AMOOPVamoOr, LSWidth8, "vamoorei8.v">;
  defm VAMOOREI16 : VAMO<AMOOPVamoOr, LSWidth16, "vamoorei16.v">;
  defm VAMOOREI32 : VAMO<AMOOPVamoOr, LSWidth32, "vamoorei32.v">;

  defm VAMOMINEI8 : VAMO<AMOOPVamoMin, LSWidth8, "vamominei8.v">;
  defm VAMOMINEI16 : VAMO<AMOOPVamoMin, LSWidth16, "vamominei16.v">;
  defm VAMOMINEI32 : VAMO<AMOOPVamoMin, LSWidth32, "vamominei32.v">;

  defm VAMOMAXEI8 : VAMO<AMOOPVamoMax, LSWidth8, "vamomaxei8.v">;
  defm VAMOMAXEI16 : VAMO<AMOOPVamoMax, LSWidth16, "vamomaxei16.v">;
  defm VAMOMAXEI32 : VAMO<AMOOPVamoMax, LSWidth32, "vamomaxei32.v">;

  defm VAMOMINUEI8 : VAMO<AMOOPVamoMinu, LSWidth8, "vamominuei8.v">;
  defm VAMOMINUEI16 : VAMO<AMOOPVamoMinu, LSWidth16, "vamominuei16.v">;
  defm VAMOMINUEI32 : VAMO<AMOOPVamoMinu, LSWidth32, "vamominuei32.v">;

  defm VAMOMAXUEI8 : VAMO<AMOOPVamoMaxu, LSWidth8, "vamomaxuei8.v">;
  defm VAMOMAXUEI16 : VAMO<AMOOPVamoMaxu, LSWidth16, "vamomaxuei16.v">;
  defm VAMOMAXUEI32 : VAMO<AMOOPVamoMaxu, LSWidth32, "vamomaxuei32.v">;
} // Predicates = [HasStdExtZvamo, HasStdExtA]

let Predicates = [HasStdExtZvamo, HasStdExtA, IsRV64] in {
  defm VAMOSWAPEI64 : VAMO<AMOOPVamoSwap, LSWidth64, "vamoswapei64.v">;
  defm VAMOADDEI64 : VAMO<AMOOPVamoAdd, LSWidth64, "vamoaddei64.v">;
  defm VAMOXOREI64 : VAMO<AMOOPVamoXor, LSWidth64, "vamoxorei64.v">;
  defm VAMOANDEI64 : VAMO<AMOOPVamoAnd, LSWidth64, "vamoandei64.v">;
  defm VAMOOREI64 : VAMO<AMOOPVamoOr, LSWidth64, "vamoorei64.v">;
  defm VAMOMINEI64 : VAMO<AMOOPVamoMin, LSWidth64, "vamominei64.v">;
  defm VAMOMAXEI64 : VAMO<AMOOPVamoMax, LSWidth64, "vamomaxei64.v">;
  defm VAMOMINUEI64 : VAMO<AMOOPVamoMinu, LSWidth64, "vamominuei64.v">;
  defm VAMOMAXUEI64 : VAMO<AMOOPVamoMaxu, LSWidth64, "vamomaxuei64.v">;
} // Predicates = [HasStdExtZvamo, HasStdExtA, IsRV64]

include "RISCVInstrInfoVPseudos.td"

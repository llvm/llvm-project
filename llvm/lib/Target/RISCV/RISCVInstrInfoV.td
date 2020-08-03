//===-- RISCVInstrInfoV.td - RISC-V 'V' instructions -------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// This file describes the RISC-V instructions from the standard 'V' Vector
/// extension, version 0.9.
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

def VRegAsmOperand : AsmOperandClass {
  let Name = "RVVRegOpOperand";
  let RenderMethod = "addRegOperands";
  let PredicateMethod = "isReg";
  let ParserMethod = "parseRegister";
}

def VRegOp : RegisterOperand<VR> {
  let ParserMatchClass = VRegAsmOperand;
  let PrintMethod = "printOperand";
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
  let RenderMethod = "addSImm5Plus1Operands";
  let DiagnosticType = "InvalidSImm5Plus1";
}

def simm5_plus1 : Operand<XLenVT>, ImmLeaf<XLenVT,
                                           [{return isInt<5>(Imm - 1);}]> {
  let ParserMatchClass = SImm5Plus1AsmOperand;
  let PrintMethod = "printSImm5Plus1";
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
// load vd, (rs1), vm
class VUnitStrideLoad<RISCVLSUMOP lumop, RISCVWidth width,
                      string opcodestr>
    : RVInstVLU<0b000, width.Value{3}, lumop, width.Value{2-0},
                (outs VRegOp:$vd),
                (ins GPR:$rs1, VMaskOp:$vm), opcodestr, "$vd, (${rs1})$vm">;

// load vd, (rs1), rs2, vm
class VStridedLoad<RISCVWidth width, string opcodestr>
    : RVInstVLS<0b000, width.Value{3}, width.Value{2-0},
                (outs VRegOp:$vd),
                (ins GPR:$rs1, GPR:$rs2, VMaskOp:$vm), opcodestr,
                "$vd, (${rs1}), $rs2$vm">;

// load vd, (rs1), vs2, vm
class VIndexedLoad<RISCVWidth width, string opcodestr>
    : RVInstVLX<0b000, width.Value{3}, width.Value{2-0},
                (outs VRegOp:$vd),
                (ins GPR:$rs1, VRegOp:$vs2, VMaskOp:$vm), opcodestr,
                "$vd, (${rs1}), $vs2$vm">;

// vl<nf>r.v vd, (rs1)
class VWholeLoad<bits<3> nf, string opcodestr>
    : RVInstVLU<nf, 0b0, LUMOPUnitStrideWholeReg,
                0b000, (outs VRegOp:$vd), (ins GPR:$rs1),
                opcodestr, "$vd, (${rs1})"> {
  let vm = 1;
  let Uses = [];
}
} // hasSideEffects = 0, mayLoad = 1, mayStore = 0

let hasSideEffects = 0, mayLoad = 0, mayStore = 1 in {
// store vd, vs3, (rs1), vm
class VUnitStrideStore<RISCVLSUMOP sumop, RISCVWidth width,
                         string opcodestr>
    : RVInstVSU<0b000, width.Value{3}, sumop, width.Value{2-0},
                (outs), (ins VRegOp:$vs3, GPR:$rs1, VMaskOp:$vm), opcodestr,
                "$vs3, (${rs1})$vm">;

// store vd, vs3, (rs1), rs2, vm
class VStridedStore<RISCVWidth width, string opcodestr>
    : RVInstVSS<0b000, width.Value{3}, width.Value{2-0}, (outs),
                (ins VRegOp:$vs3, GPR:$rs1, GPR:$rs2, VMaskOp:$vm),
                opcodestr, "$vs3, (${rs1}), $rs2$vm">;

// store vd, vs3, (rs1), vs2, vm
class VIndexedStore<RISCVMOP mop, RISCVWidth width, string opcodestr>
    : RVInstVSX<0b000, width.Value{3}, mop, width.Value{2-0}, (outs),
                (ins VRegOp:$vs3, GPR:$rs1, VRegOp:$vs2, VMaskOp:$vm),
                opcodestr, "$vs3, (${rs1}), $vs2$vm">;

// vs<nf>r.v vd, (rs1)
class VWholeStore<bits<3> nf, string opcodestr>
    : RVInstVSU<nf, 0b0, SUMOPUnitStrideWholeReg,
                0b000, (outs), (ins VRegOp:$vs3, GPR:$rs1),
                opcodestr, "$vs3, (${rs1})"> {
  let vm = 1;
  let Uses = [];
}
} // hasSideEffects = 0, mayLoad = 0, mayStore = 1

let hasSideEffects = 0, mayLoad = 0, mayStore = 0 in {
// op vd, vs2, vs1, vm
class VALUVV<bits<6> funct6, RISCVVFormat opv, string opcodestr>
    : RVInstVV<funct6, opv, (outs VRegOp:$vd),
                (ins VRegOp:$vs2, VRegOp:$vs1, VMaskOp:$vm),
                opcodestr, "$vd, $vs2, $vs1$vm">;

// op vd, vs2, vs1, v0 (without mask, use v0 as carry input)
class VALUmVV<bits<6> funct6, RISCVVFormat opv, string opcodestr>
    : RVInstVV<funct6, opv, (outs VRegOp:$vd),
                (ins VRegOp:$vs2, VRegOp:$vs1, VMV0:$v0),
                opcodestr, "$vd, $vs2, $vs1, v0"> {
  let vm = 0;
}

// op vd, vs1, vs2, vm (reverse the order of vs1 and vs2)
class VALUrVV<bits<6> funct6, RISCVVFormat opv, string opcodestr>
    : RVInstVV<funct6, opv, (outs VRegOp:$vd),
                (ins VRegOp:$vs1, VRegOp:$vs2, VMaskOp:$vm),
                opcodestr, "$vd, $vs1, $vs2$vm">;

// op vd, vs1, vs2
class VALUVVNoVm<bits<6> funct6, RISCVVFormat opv, string opcodestr>
    : RVInstVV<funct6, opv, (outs VRegOp:$vd),
               (ins VRegOp:$vs2, VRegOp:$vs1),
               opcodestr, "$vd, $vs2, $vs1"> {
  let vm = 1;
}

// op vd, vs2, rs1, vm
class VALUVX<bits<6> funct6, RISCVVFormat opv, string opcodestr>
    : RVInstVX<funct6, opv, (outs VRegOp:$vd),
                (ins VRegOp:$vs2, GPR:$rs1, VMaskOp:$vm),
                opcodestr, "$vd, $vs2, $rs1$vm">;

// op vd, vs2, rs1, v0 (without mask, use v0 as carry input)
class VALUmVX<bits<6> funct6, RISCVVFormat opv, string opcodestr>
    : RVInstVX<funct6, opv, (outs VRegOp:$vd),
                (ins VRegOp:$vs2, GPR:$rs1, VMV0:$v0),
                opcodestr, "$vd, $vs2, $rs1, v0"> {
  let vm = 0;
}

// op vd, rs1, vs2, vm (reverse the order of rs1 and vs2)
class VALUrVX<bits<6> funct6, RISCVVFormat opv, string opcodestr>
    : RVInstVX<funct6, opv, (outs VRegOp:$vd),
                (ins GPR:$rs1, VRegOp:$vs2, VMaskOp:$vm),
                opcodestr, "$vd, $rs1, $vs2$vm">;

// op vd, vs1, vs2
class VALUVXNoVm<bits<6> funct6, RISCVVFormat opv, string opcodestr>
    : RVInstVX<funct6, opv, (outs VRegOp:$vd),
               (ins VRegOp:$vs2, GPR:$rs1),
               opcodestr, "$vd, $vs2, $rs1"> {
  let vm = 1;
}

// op vd, vs2, imm, vm
class VALUVI<bits<6> funct6, string opcodestr, Operand optype = simm5>
    : RVInstIVI<funct6, (outs VRegOp:$vd),
                (ins VRegOp:$vs2, optype:$imm, VMaskOp:$vm),
                opcodestr, "$vd, $vs2, $imm$vm">;

// op vd, vs2, imm, v0 (without mask, use v0 as carry input)
class VALUmVI<bits<6> funct6, string opcodestr, Operand optype = simm5>
    : RVInstIVI<funct6, (outs VRegOp:$vd),
                (ins VRegOp:$vs2, optype:$imm, VMV0:$v0),
                opcodestr, "$vd, $vs2, $imm, v0"> {
  let vm = 0;
}

// op vd, vs2, imm, vm
class VALUVINoVm<bits<6> funct6, string opcodestr, Operand optype = simm5>
    : RVInstIVI<funct6, (outs VRegOp:$vd),
                (ins VRegOp:$vs2, optype:$imm),
                opcodestr, "$vd, $vs2, $imm"> {
  let vm = 1;
}

// op vd, vs2, rs1, vm (Float)
class VALUVF<bits<6> funct6, RISCVVFormat opv, string opcodestr>
    : RVInstVX<funct6, opv, (outs VRegOp:$vd),
                (ins VRegOp:$vs2, FPR32:$rs1, VMaskOp:$vm),
                opcodestr, "$vd, $vs2, $rs1$vm">;

// op vd, rs1, vs2, vm (Float) (with mask, reverse the order of rs1 and vs2)
class VALUrVF<bits<6> funct6, RISCVVFormat opv, string opcodestr>
    : RVInstVX<funct6, opv, (outs VRegOp:$vd),
                (ins FPR32:$rs1, VRegOp:$vs2, VMaskOp:$vm),
                opcodestr, "$vd, $rs1, $vs2$vm">;

// op vd, vs2, vm (use vs1 as instruction encoding)
class VALUVs2<bits<6> funct6, bits<5> vs1, RISCVVFormat opv, string opcodestr>
    : RVInstV<funct6, vs1, opv, (outs VRegOp:$vd),
               (ins VRegOp:$vs2, VMaskOp:$vm),
               opcodestr, "$vd, $vs2$vm">;
} // hasSideEffects = 0, mayLoad = 0, mayStore = 0

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

//===----------------------------------------------------------------------===//
// Instructions
//===----------------------------------------------------------------------===//

let Predicates = [HasStdExtV] in {
let hasSideEffects = 1, mayLoad = 0, mayStore = 0 in {
def VSETVLI : RVInstSetVLi<(outs GPR:$rd), (ins GPR:$rs1, VTypeIOp:$vtypei),
                           "vsetvli", "$rd, $rs1, $vtypei">;

def VSETVL : RVInstSetVL<(outs GPR:$rd), (ins GPR:$rs1, GPR:$rs2),
                         "vsetvl", "$rd, $rs1, $rs2">;
} // hasSideEffects = 1, mayLoad = 0, mayStore = 0

// Vector Unit-Stride Instructions
def VLE8_V : VUnitStrideLoad<LUMOPUnitStride, LSWidth8, "vle8.v">;
def VLE16_V : VUnitStrideLoad<LUMOPUnitStride, LSWidth16, "vle16.v">;
def VLE32_V : VUnitStrideLoad<LUMOPUnitStride, LSWidth32, "vle32.v">;
def VLE64_V : VUnitStrideLoad<LUMOPUnitStride, LSWidth64, "vle64.v">;
def VLE128_V : VUnitStrideLoad<LUMOPUnitStride, LSWidth128, "vle128.v">;
def VLE256_V : VUnitStrideLoad<LUMOPUnitStride, LSWidth256, "vle256.v">;
def VLE512_V : VUnitStrideLoad<LUMOPUnitStride, LSWidth512, "vle512.v">;
def VLE1024_V : VUnitStrideLoad<LUMOPUnitStride, LSWidth1024, "vle1024.v">;

def VLE8FF_V : VUnitStrideLoad<LUMOPUnitStrideFF, LSWidth8, "vle8ff.v">;
def VLE16FF_V : VUnitStrideLoad<LUMOPUnitStrideFF, LSWidth16, "vle16ff.v">;
def VLE32FF_V : VUnitStrideLoad<LUMOPUnitStrideFF, LSWidth32, "vle32ff.v">;
def VLE64FF_V : VUnitStrideLoad<LUMOPUnitStrideFF, LSWidth64, "vle64ff.v">;
def VLE128FF_V : VUnitStrideLoad<LUMOPUnitStrideFF, LSWidth128, "vle128ff.v">;
def VLE256FF_V : VUnitStrideLoad<LUMOPUnitStrideFF, LSWidth256, "vle256ff.v">;
def VLE512FF_V : VUnitStrideLoad<LUMOPUnitStrideFF, LSWidth512, "vle512ff.v">;
def VLE1024FF_V : VUnitStrideLoad<LUMOPUnitStrideFF, LSWidth1024, "vle1024ff.v">;

def VSE8_V : VUnitStrideStore<SUMOPUnitStride, LSWidth8, "vse8.v">;
def VSE16_V : VUnitStrideStore<SUMOPUnitStride, LSWidth16, "vse16.v">;
def VSE32_V : VUnitStrideStore<SUMOPUnitStride, LSWidth32, "vse32.v">;
def VSE64_V : VUnitStrideStore<SUMOPUnitStride, LSWidth64, "vse64.v">;
def VSE128_V : VUnitStrideStore<SUMOPUnitStride, LSWidth128, "vse128.v">;
def VSE256_V : VUnitStrideStore<SUMOPUnitStride, LSWidth256, "vse256.v">;
def VSE512_V : VUnitStrideStore<SUMOPUnitStride, LSWidth512, "vse512.v">;
def VSE1024_V : VUnitStrideStore<SUMOPUnitStride, LSWidth1024, "vse1024.v">;

// Vector Strided Instructions
def VLSE8_V : VStridedLoad<LSWidth8, "vlse8.v">;
def VLSE16_V : VStridedLoad<LSWidth16, "vlse16.v">;
def VLSE32_V : VStridedLoad<LSWidth32, "vlse32.v">;
def VLSE64_V : VStridedLoad<LSWidth64, "vlse64.v">;
def VLSE128_V : VStridedLoad<LSWidth128, "vlse128.v">;
def VLSE256_V : VStridedLoad<LSWidth256, "vlse256.v">;
def VLSE512_V : VStridedLoad<LSWidth512, "vlse512.v">;
def VLSE1024_V : VStridedLoad<LSWidth1024, "vlse1024.v">;

def VSSE8_V : VStridedStore<LSWidth8, "vsse8.v">;
def VSSE16_V : VStridedStore<LSWidth16, "vsse16.v">;
def VSSE32_V : VStridedStore<LSWidth32, "vsse32.v">;
def VSSE64_V : VStridedStore<LSWidth64, "vsse64.v">;
def VSSE128_V : VStridedStore<LSWidth128, "vsse128.v">;
def VSSE256_V : VStridedStore<LSWidth256, "vsse256.v">;
def VSSE512_V : VStridedStore<LSWidth512, "vsse512.v">;
def VSSE1024_V : VStridedStore<LSWidth1024, "vsse1024.v">;

// Vector Indexed Instructions
def VLXEI8_V : VIndexedLoad<LSWidth8, "vlxei8.v">;
def VLXEI16_V : VIndexedLoad<LSWidth16, "vlxei16.v">;
def VLXEI32_V : VIndexedLoad<LSWidth32, "vlxei32.v">;
def VLXEI64_V : VIndexedLoad<LSWidth64, "vlxei64.v">;
def VLXEI128_V : VIndexedLoad<LSWidth128, "vlxei128.v">;
def VLXEI256_V : VIndexedLoad<LSWidth256, "vlxei256.v">;
def VLXEI512_V : VIndexedLoad<LSWidth512, "vlxei512.v">;
def VLXEI1024_V : VIndexedLoad<LSWidth1024, "vlxei1024.v">;

def VSXEI8_V : VIndexedStore<MOPSTIndexedOrder, LSWidth8, "vsxei8.v">;
def VSXEI16_V : VIndexedStore<MOPSTIndexedOrder, LSWidth16, "vsxei16.v">;
def VSXEI32_V : VIndexedStore<MOPSTIndexedOrder, LSWidth32, "vsxei32.v">;
def VSXEI64_V : VIndexedStore<MOPSTIndexedOrder, LSWidth64, "vsxei64.v">;
def VSXEI128_V : VIndexedStore<MOPSTIndexedOrder, LSWidth128, "vsxei128.v">;
def VSXEI256_V : VIndexedStore<MOPSTIndexedOrder, LSWidth256, "vsxei256.v">;
def VSXEI512_V : VIndexedStore<MOPSTIndexedOrder, LSWidth512, "vsxei512.v">;
def VSXEI1024_V : VIndexedStore<MOPSTIndexedOrder, LSWidth1024, "vsxei1024.v">;

def VSUXEI8_V : VIndexedStore<MOPSTIndexedUnord, LSWidth8, "vsuxei8.v">;
def VSUXEI16_V : VIndexedStore<MOPSTIndexedUnord, LSWidth16, "vsuxei16.v">;
def VSUXEI32_V : VIndexedStore<MOPSTIndexedUnord, LSWidth32, "vsuxei32.v">;
def VSUXEI64_V : VIndexedStore<MOPSTIndexedUnord, LSWidth64, "vsuxei64.v">;
def VSUXEI128_V : VIndexedStore<MOPSTIndexedUnord, LSWidth128, "vsuxei128.v">;
def VSUXEI256_V : VIndexedStore<MOPSTIndexedUnord, LSWidth256, "vsuxei256.v">;
def VSUXEI512_V : VIndexedStore<MOPSTIndexedUnord, LSWidth512, "vsuxei512.v">;
def VSUXEI1024_V : VIndexedStore<MOPSTIndexedUnord, LSWidth1024, "vsuxei1024.v">;

def VL1R_V : VWholeLoad<0, "vl1r.v">;
def VS1R_V : VWholeStore<0, "vs1r.v">;

// Vector Single-Width Integer Add and Subtract
defm VADD_V : VALU_IV_V_X_I<"vadd", 0b000000>;
defm VSUB_V : VALU_IV_V_X<"vsub", 0b000010>;
defm VRSUB_V : VALU_IV_X_I<"vrsub", 0b000011>;

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
                (VWADD_VX VRegOp:$vd, VRegOp:$vs, X0, VMaskOp:$vm)>;
def : InstAlias<"vwcvtu.x.x.v $vd, $vs$vm",
                (VWADDU_VX VRegOp:$vd, VRegOp:$vs, X0, VMaskOp:$vm)>;

// Vector Integer Extension
defm VZEXT_VF8 : VALU_MV_VS2<"vzext.vf8", 0b010010, 0b00010>;
defm VSEXT_VF8 : VALU_MV_VS2<"vsext.vf8", 0b010010, 0b00011>;
defm VZEXT_VF4 : VALU_MV_VS2<"vzext.vf4", 0b010010, 0b00100>;
defm VSEXT_VF4 : VALU_MV_VS2<"vsext.vf4", 0b010010, 0b00101>;
defm VZEXT_VF2 : VALU_MV_VS2<"vzext.vf2", 0b010010, 0b00110>;
defm VSEXT_VF2 : VALU_MV_VS2<"vsext.vf2", 0b010010, 0b00111>;

// Vector Integer Add-with-Carry / Subtract-with-Borrow Instructions
defm VADC_V : VALUm_IV_V_X_I<"vadc", 0b010000>;
let Constraints = "@earlyclobber $vd", RVVConstraint = Vmadc in {
defm VMADC_V : VALUm_IV_V_X_I<"vmadc", 0b010001>;
defm VMADC_V : VALUNoVm_IV_V_X_I<"vmadc", 0b010001>;
} // Constraints = "@earlyclobber $vd", RVVConstraint = Vmadc
defm VSBC_V : VALUm_IV_V_X<"vsbc", 0b010010>;
let Constraints = "@earlyclobber $vd", RVVConstraint = Vmadc in {
defm VMSBC_V : VALUm_IV_V_X<"vmsbc", 0b010011>;
defm VMSBC_V : VALUNoVm_IV_V_X<"vmsbc", 0b010011>;
} // Constraints = "@earlyclobber $vd", RVVConstraint = Vmadc

// Vector Bitwise Logical Instructions
defm VAND_V : VALU_IV_V_X_I<"vand", 0b001001>;
defm VOR_V : VALU_IV_V_X_I<"vor", 0b001010>;
defm VXOR_V : VALU_IV_V_X_I<"vxor", 0b001011>;

def : InstAlias<"vnot.v $vd, $vs$vm",
                (VXOR_VI VRegOp:$vd, VRegOp:$vs, -1, VMaskOp:$vm)>;

// Vector Single-Width Bit Shift Instructions
defm VSLL_V : VALU_IV_V_X_I<"vsll", 0b100101, uimm5>;
defm VSRL_V : VALU_IV_V_X_I<"vsrl", 0b101000, uimm5>;
defm VSRA_V : VALU_IV_V_X_I<"vsra", 0b101001, uimm5>;

// Vector Narrowing Integer Right Shift Instructions
// Refer to 11.3. Narrowing Vector Arithmetic Instructions
// The destination vector register group cannot overlap the first source
// vector register group (specified by vs2). The destination vector register
// group cannot overlap the mask register if used, unless LMUL=1.
let Constraints = "@earlyclobber $vd", RVVConstraint = Narrow in {
defm VNSRL_W : VALU_IV_V_X_I<"vnsrl", 0b101100, uimm5, "w">;
defm VNSRA_W : VALU_IV_V_X_I<"vnsra", 0b101101, uimm5, "w">;
} // Constraints = "@earlyclobber $vd", RVVConstraint = Narrow

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
                (VMSLTU_VV VRegOp:$vd, VRegOp:$vb, VRegOp:$va, VMaskOp:$vm), 0>;
def : InstAlias<"vmsgt.vv $vd, $va, $vb$vm",
                (VMSLT_VV VRegOp:$vd, VRegOp:$vb, VRegOp:$va, VMaskOp:$vm), 0>;
def : InstAlias<"vmsgeu.vv $vd, $va, $vb$vm",
                (VMSLEU_VV VRegOp:$vd, VRegOp:$vb, VRegOp:$va, VMaskOp:$vm), 0>;
def : InstAlias<"vmsge.vv $vd, $va, $vb$vm",
                (VMSLE_VV VRegOp:$vd, VRegOp:$vb, VRegOp:$va, VMaskOp:$vm), 0>;
def : InstAlias<"vmsltu.vi $vd, $va, $imm$vm",
                (VMSLEU_VI VRegOp:$vd, VRegOp:$va, simm5_plus1:$imm,
                 VMaskOp:$vm), 0>;
def : InstAlias<"vmslt.vi $vd, $va, $imm$vm",
                (VMSLE_VI VRegOp:$vd, VRegOp:$va, simm5_plus1:$imm,
                 VMaskOp:$vm), 0>;
def : InstAlias<"vmsgeu.vi $vd, $va, $imm$vm",
                (VMSGTU_VI VRegOp:$vd, VRegOp:$va, simm5_plus1:$imm,
                 VMaskOp:$vm), 0>;
def : InstAlias<"vmsge.vi $vd, $va, $imm$vm",
                (VMSGT_VI VRegOp:$vd, VRegOp:$va, simm5_plus1:$imm,
                 VMaskOp:$vm), 0>;

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
let hasSideEffects = 0, mayLoad = 0, mayStore = 0, vs2 = 0, vm = 1 in {
// op vd, vs1
def VMV_V_V : RVInstVV<0b010111, OPIVV, (outs VRegOp:$vd),
                       (ins VRegOp:$vs1), "vmv.v.v", "$vd, $vs1">;
// op vd, rs1
def VMV_V_X : RVInstVX<0b010111, OPIVX, (outs VRegOp:$vd),
                       (ins GPR:$rs1), "vmv.v.x", "$vd, $rs1">;
// op vd, imm
def VMV_V_I : RVInstIVI<0b010111, (outs VRegOp:$vd),
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
let Constraints = "@earlyclobber $vd", RVVConstraint = Narrow in {
defm VNCLIPU_W : VALU_IV_V_X_I<"vnclipu", 0b101110, uimm5, "w">;
defm VNCLIP_W : VALU_IV_V_X_I<"vnclip", 0b101111, uimm5, "w">;
} // Constraints = "@earlyclobber $vd", RVVConstraint = Narrow

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

// Vector Floating-Point MIN/MAX Instructions
defm VFMIN_V : VALU_FV_V_F<"vfmin", 0b000100>;
defm VFMAX_V : VALU_FV_V_F<"vfmax", 0b000110>;

// Vector Floating-Point Sign-Injection Instructions
defm VFSGNJ_V : VALU_FV_V_F<"vfsgnj", 0b001000>;
defm VFSGNJN_V : VALU_FV_V_F<"vfsgnjn", 0b001001>;
defm VFSGNJX_V : VALU_FV_V_F<"vfsgnjx", 0b001010>;

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
                (VMFLT_VV VRegOp:$vd, VRegOp:$vb, VRegOp:$va, VMaskOp:$vm), 0>;
def : InstAlias<"vmfge.vv $vd, $va, $vb$vm",
                (VMFLE_VV VRegOp:$vd, VRegOp:$vb, VRegOp:$va, VMaskOp:$vm), 0>;

// Vector Floating-Point Classify Instruction
defm VFCLASS_V : VALU_FV_VS2<"vfclass.v", 0b010011, 0b10000>;

let hasSideEffects = 0, mayLoad = 0, mayStore = 0 in {
// Vector Floating-Point Merge Instruction
def VFMERGE_VFM : RVInstVX<0b010111, OPFVF, (outs VRegOp:$vd),
                           (ins VRegOp:$vs2, FPR32:$rs1, VMV0:$v0),
                           "vfmerge.vfm", "$vd, $vs2, $rs1, v0"> {
  let vm = 0;
}

// Vector Floating-Point Move Instruction
def VFMV_V_F : RVInstVX<0b010111, OPFVF, (outs VRegOp:$vd),
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
let Constraints = "@earlyclobber $vd", RVVConstraint = NarrowCvt in {
defm VFNCVT_XU_F_W : VALU_FV_VS2<"vfncvt.xu.f.w", 0b010010, 0b10000>;
defm VFNCVT_X_F_W : VALU_FV_VS2<"vfncvt.x.f.w", 0b010010, 0b10001>;
defm VFNCVT_RTZ_XU_F_W : VALU_FV_VS2<"vfncvt.rtz.xu.f.w", 0b010010, 0b10110>;
defm VFNCVT_RTZ_X_F_W : VALU_FV_VS2<"vfncvt.rtz.x.f.w", 0b010010, 0b10111>;
defm VFNCVT_F_XU_W : VALU_FV_VS2<"vfncvt.f.xu.w", 0b010010, 0b10010>;
defm VFNCVT_F_X_W : VALU_FV_VS2<"vfncvt.f.x.w", 0b010010, 0b10011>;
defm VFNCVT_F_F_W : VALU_FV_VS2<"vfncvt.f.f.w", 0b010010, 0b10100>;
defm VFNCVT_ROD_F_F_W : VALU_FV_VS2<"vfncvt.rod.f.f.w", 0b010010, 0b10101>;
} // Constraints = "@earlyclobber $vd", RVVConstraint = NarrowCvt

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

// Vector Mask-Register Logical Instructions
defm VMAND_M : VALU_MV_Mask<"vmand", 0b011001, "m">;
defm VMNAND_M : VALU_MV_Mask<"vmnand", 0b011101, "m">;
defm VMANDNOT_M : VALU_MV_Mask<"vmandnot", 0b011000, "m">;
defm VMXOR_M : VALU_MV_Mask<"vmxor", 0b011011, "m">;
defm VMOR_M : VALU_MV_Mask<"vmor", 0b011010, "m">;
defm VMNOR_M : VALU_MV_Mask<"vmnor", 0b011110, "m">;
defm VMORNOT_M : VALU_MV_Mask<"vmornot", 0b011100, "m">;
defm VMXNOR_M : VALU_MV_Mask<"vmxnor", 0b011111, "m">;

def : InstAlias<"vmmv.m $vd, $vs",
                (VMAND_MM VRegOp:$vd, VRegOp:$vs, VRegOp:$vs)>;
def : InstAlias<"vmclr.m $vd",
                (VMXOR_MM VRegOp:$vd, VRegOp:$vd, VRegOp:$vd)>;
def : InstAlias<"vmset.m $vd",
                (VMXNOR_MM VRegOp:$vd, VRegOp:$vd, VRegOp:$vd)>;
def : InstAlias<"vmnot.m $vd, $vs",
                (VMNAND_MM VRegOp:$vd, VRegOp:$vs, VRegOp:$vs)>;

let hasSideEffects = 0, mayLoad = 0, mayStore = 0 in {
// Vector mask population count vpopc
def VPOPC_M : RVInstV<0b010000, 0b10000, OPMVV, (outs GPR:$vd),
                        (ins VRegOp:$vs2, VMaskOp:$vm),
                        "vpopc.m", "$vd, $vs2$vm">;

// vfirst find-first-set mask bit
def VFIRST_M : RVInstV<0b010000, 0b10001, OPMVV, (outs GPR:$vd),
                        (ins VRegOp:$vs2, VMaskOp:$vm),
                        "vfirst.m", "$vd, $vs2$vm">;
} // hasSideEffects = 0, mayLoad = 0, mayStore = 0

// vmsbf.m set-before-first mask bit
defm VMSBF_M : VALU_MV_VS2<"vmsbf.m", 0b010100, 0b00001>;

// vmsif.m set-including-first mask bit
defm VMSIF_M : VALU_MV_VS2<"vmsif.m", 0b010100, 0b00011>;

// vmsof.m set-only-first mask bit
defm VMSOF_M : VALU_MV_VS2<"vmsof.m", 0b010100, 0b00010>;

// Vector Iota Instruction
let Constraints = "@earlyclobber $vd", RVVConstraint = Iota in {
defm VIOTA_M : VALU_MV_VS2<"viota.m", 0b010100, 0b10000>;
} // Constraints = "@earlyclobber $vd", RVVConstraint = Iota

// Vector Element Index Instruction
let hasSideEffects = 0, mayLoad = 0, mayStore = 0 in {
def VID_V : RVInstV<0b010100, 0b10001, OPMVV, (outs VRegOp:$vd),
                      (ins VMaskOp:$vm), "vid.v", "$vd$vm"> {
  let vs2 = 0;
}

// Integer Scalar Move Instructions
let vm = 1 in {
def VMV_X_S : RVInstV<0b010000, 0b00000, OPMVV, (outs GPR:$vd),
                      (ins VRegOp:$vs2), "vmv.x.s", "$vd, $vs2">;
def VMV_S_X : RVInstV2<0b010000, 0b00000, OPMVX, (outs VRegOp:$vd),
                      (ins GPR:$rs1), "vmv.s.x", "$vd, $rs1">;

}
} // hasSideEffects = 0, mayLoad = 0, mayStore = 0

let hasSideEffects = 0, mayLoad = 0, mayStore = 0, vm = 1 in {
// Floating-Point Scalar Move Instructions
def VFMV_F_S : RVInstV<0b010000, 0b00000, OPFVV, (outs FPR32:$vd),
                      (ins VRegOp:$vs2), "vfmv.f.s", "$vd, $vs2">;
def VFMV_S_F : RVInstV2<0b010000, 0b00000, OPFVF, (outs VRegOp:$vd),
                      (ins FPR32:$rs1), "vfmv.s.f", "$vd, $rs1">;

} // hasSideEffects = 0, mayLoad = 0, mayStore = 0, vm = 1

// Vector Slide Instructions
let Constraints = "@earlyclobber $vd", RVVConstraint = SlideUp in {
defm VSLIDEUP_V : VALU_IV_X_I<"vslideup", 0b001110, uimm5>;
} // Constraints = "@earlyclobber $vd", RVVConstraint = SlideUp
defm VSLIDEDOWN_V : VALU_IV_X_I<"vslidedown", 0b001111, uimm5>;

let Constraints = "@earlyclobber $vd", RVVConstraint = SlideUp in {
defm VSLIDE1UP_V : VALU_MV_X<"vslide1up", 0b001110>;
defm VFSLIDE1UP_V : VALU_FV_F<"vfslide1up", 0b001110>;
} // Constraints = "@earlyclobber $vd", RVVConstraint = SlideUp
defm VSLIDE1DOWN_V : VALU_MV_X<"vslide1down", 0b001111>;
defm VFSLIDE1DOWN_V : VALU_FV_F<"vfslide1down", 0b001111>;

// Vector Register Gather Instruction
let Constraints = "@earlyclobber $vd", RVVConstraint = Vrgather in {
defm VRGATHER_V : VALU_IV_V_X_I<"vrgather", 0b001100, uimm5>;
} // Constraints = "@earlyclobber $vd", RVVConstraint = Vrgather

// Vector Compress Instruction
let Constraints = "@earlyclobber $vd", RVVConstraint = Vcompress in {
defm VCOMPRESS_V : VALU_MV_Mask<"vcompress", 0b010111>;
} // Constraints = "@earlyclobber $vd", RVVConstraint = Vcompress

let hasSideEffects = 0, mayLoad = 0, mayStore = 0 in {
foreach nf = [1, 2, 4, 8] in {
  def VMV#nf#R_V  : RVInstV<0b100111, !add(nf, -1), OPIVI, (outs VRegOp:$vd),
                            (ins VRegOp:$vs2), "vmv" # nf # "r.v",
                            "$vd, $vs2"> {
    let Uses = [];
    let vm = 1;
  }
}
} // hasSideEffects = 0, mayLoad = 0, mayStore = 0
} // Predicates = [HasStdExtV]

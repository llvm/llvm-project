//===-- RISCVInstrInfoZvk.td - RISC-V 'Zvk' instructions -------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file describes the RISC-V instructions from the standard 'Zvk',
// Vector Cryptography Instructions extension, version 0.3.
//
//===----------------------------------------------------------------------===//

let hasSideEffects = 0, mayLoad = 0, mayStore = 0 in {
multiclass VCLMUL_MV_V_X<string opcodestr, bits<6> funct6> {
  def V  : VALUVV<funct6, OPMVV, opcodestr # "." # "vv">,
           Sched<[WriteVIALUV_WorstCase, ReadVIALUV_WorstCase,
                  ReadVIALUV_WorstCase, ReadVMask]>;
  def X  : VALUVX<funct6, OPMVX, opcodestr # "." # "vx">,
           Sched<[WriteVIALUX_WorstCase, ReadVIALUV_WorstCase,
                  ReadVIALUX_WorstCase, ReadVMask]>;
}

class RVInstIVI_VROR<bits<6> funct6, dag outs, dag ins, string opcodestr,
                     string argstr>
    : RVInst<outs, ins, opcodestr, argstr, [], InstFormatR> {
  bits<5> vs2;
  bits<6> imm;
  bits<5> vd;
  bit vm;

  let Inst{31-27} = funct6{5-1};
  let Inst{26} = imm{5};
  let Inst{25} = vm;
  let Inst{24-20} = vs2;
  let Inst{19-15} = imm{4-0};
  let Inst{14-12} = OPIVI.Value;
  let Inst{11-7} = vd;
  let Opcode = OPC_OP_V.Value;

  let Uses = [VTYPE, VL];
  let RVVConstraint = VMConstraint;
}

multiclass VROR_IV_V_X_I<string opcodestr, bits<6> funct6,
                              Operand optype = uimm6, string vw = "v">
    : VALU_IV_V_X<opcodestr, funct6> {
  def I : RVInstIVI_VROR<funct6, (outs VR:$vd),
              (ins VR:$vs2, optype:$imm, VMaskOp:$vm),
              opcodestr # "." # vw # "i", "$vd, $vs2, $imm$vm">,
         Sched<[WriteVIALUI_WorstCase, ReadVIALUV_WorstCase,
                ReadVMask]>;
}

// op vd, vs2, vs1
class PALUVVNoVm<bits<6> funct6, RISCVVFormat opv, string opcodestr>
    : VALUVVNoVm<funct6, opv, opcodestr> {
  let Opcode = OPC_OP_P.Value;
}

// op vd, vs2, imm, vm
class PALUVINoVm<bits<6> funct6, string opcodestr, Operand optype = simm5>
    : VALUVINoVm<funct6, opcodestr, optype> {
  let Opcode = OPC_OP_P.Value;
  let Inst{14-12} = OPMVV.Value;
}

// op vd, vs2 (use vs1 as instruction encoding)
class PALUVs2NoVm<bits<6> funct6, bits<5> vs1, RISCVVFormat opv, string opcodestr>
    : VALUVs2NoVm<funct6, vs1, opv, opcodestr> {
  let Opcode = OPC_OP_P.Value;
}

multiclass VAES_MV_V_S<bits<6> funct6_vv, bits<6> funct6_vs, bits<5> vs1,
                         RISCVVFormat opv, string opcodestr> {
  def NAME # _VV : PALUVs2NoVm<funct6_vv, vs1, opv, opcodestr # ".vv">;
  def NAME # _VS : PALUVs2NoVm<funct6_vs, vs1, opv, opcodestr # ".vs">;
}

// vaeskf1.vi and vaeskf2.vi uses different opcode and format, we need
// to customize one for them.
class VAESKF_MV_I<bits<6> funct6, string opcodestr, Operand optype>
    : VALUVINoVm<funct6, opcodestr, optype> {
  let Opcode = OPC_OP_P.Value;
  let Inst{14-12} = OPMVV.Value;
}
} // hasSideEffects = 0, mayLoad = 0, mayStore = 0

def RnumArg_0_7 : AsmOperandClass {
  let Name = "RnumArg_0_7";
  let RenderMethod = "addImmOperands";
  let DiagnosticType = "InvalidRnumArg_0_7";
}

def RnumArg_1_10 : AsmOperandClass {
  let Name = "RnumArg_1_10";
  let RenderMethod = "addImmOperands";
  let DiagnosticType = "InvalidRnumArg_1_10";
}

def RnumArg_2_14 : AsmOperandClass {
  let Name = "RnumArg_2_14";
  let RenderMethod = "addImmOperands";
  let DiagnosticType = "InvalidRnumArg_2_14";
}

def rnum_0_7 : Operand<XLenVT>, ImmLeaf<XLenVT,
                                         [{return (0 <= Imm && Imm <= 7);}]> {
  let ParserMatchClass = RnumArg_0_7;
  let DecoderMethod = "decodeUImmOperand<5>";
  let OperandType = "OPERAND_RVKRNUM_0_7";
  let OperandNamespace = "RISCVOp";
}

def rnum_1_10 : Operand<XLenVT>, ImmLeaf<XLenVT,
                                         [{return (1 <= Imm && Imm <= 10);}]> {
  let ParserMatchClass = RnumArg_1_10;
  let DecoderMethod = "decodeUImmOperand<5>";
  let OperandType = "OPERAND_RVKRNUM_1_10";
  let OperandNamespace = "RISCVOp";
}

def rnum_2_14 : Operand<XLenVT>, ImmLeaf<XLenVT,
                                         [{return (2 <= Imm && Imm <= 14);}]> {
  let ParserMatchClass = RnumArg_2_14;
  let DecoderMethod = "decodeUImmOperand<5>";
  let OperandType = "OPERAND_RVKRNUM_2_14";
  let OperandNamespace = "RISCVOp";
}

let Predicates = [HasStdExtZvkb] in {
  defm VANDN_V : VALU_IV_V_X<"vandn", 0b000001>;
  def VBREV8_V : VALUVs2<0b010010, 0b01000, OPMVV, "vbrev8.v">;
  defm VCLMUL_V : VCLMUL_MV_V_X<"vclmul", 0b001100>;
  defm VCLMULH_V : VCLMUL_MV_V_X<"vclmulh", 0b001101>;
  def VREV8_V : VALUVs2<0b010010, 0b01001, OPMVV, "vrev8.v">;
  defm VROL_V : VALU_IV_V_X<"vrol", 0b010101>;
  defm VROR_V : VROR_IV_V_X_I<"vror", 0b010100>;
} // Predicates = [HasStdExtZvkb]

let Predicates = [HasStdExtZvkg], RVVConstraint = NoConstraint in {
  def VGHSH_VV : PALUVVNoVm<0b101100, OPMVV, "vghsh.vv">;
  def VGMUL_VV : PALUVs2NoVm<0b101000, 0b10001, OPMVV, "vgmul.vv">;
} // Predicates = [HasStdExtZvkg]

let Predicates = [HasStdExtZvknha], RVVConstraint = NoConstraint in {
  def VSHA2CH_VV : PALUVVNoVm<0b101110, OPMVV, "vsha2ch.vv">;
  def VSHA2CL_VV : PALUVVNoVm<0b101111, OPMVV, "vsha2cl.vv">;
  def VSHA2MS_VV : PALUVVNoVm<0b101101, OPMVV, "vsha2ms.vv">;
} // Predicates = [HasStdExtZvknha]

let Predicates = [HasStdExtZvkned], RVVConstraint = NoConstraint in {
  defm VAESDF : VAES_MV_V_S<0b101000, 0b101001, 0b00001, OPMVV, "vaesdf">;
  defm VAESDM : VAES_MV_V_S<0b101000, 0b101001, 0b00000, OPMVV, "vaesdm">;
  defm VAESEF : VAES_MV_V_S<0b101000, 0b101001, 0b00011, OPMVV, "vaesef">;
  defm VAESEM : VAES_MV_V_S<0b101000, 0b101001, 0b00010, OPMVV, "vaesem">;
  def VAESKF1_VI : VAESKF_MV_I<0b100010, "vaeskf1.vi", rnum_1_10>;
  def VAESKF2_VI : VAESKF_MV_I<0b101010, "vaeskf2.vi", rnum_2_14>;
  def VAESZ_VS : PALUVs2NoVm<0b101001, 0b00111, OPMVV, "vaesz.vs">;
} // Predicates = [HasStdExtZvkned]

let Predicates = [HasStdExtZvksed], RVVConstraint = NoConstraint in {
  def VSM4K_VI : PALUVINoVm<0b100001, "vsm4k.vi", rnum_0_7>;
  defm VSM4R : VAES_MV_V_S<0b101000, 0b101001, 0b10000, OPMVV, "vsm4r">;
} // Predicates = [HasStdExtZvksed]

let Predicates = [HasStdExtZvksh], RVVConstraint = NoConstraint in {
  def VSM3C_VI : PALUVINoVm<0b101011, "vsm3c.vi", uimm5>;
  def VSM3ME_VV : PALUVVNoVm<0b100000, OPMVV, "vsm3me.vv">;
} // Predicates = [HasStdExtZvksh]

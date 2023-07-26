//===-- RISCVInstrInfoZvk.td - RISC-V 'Zvk' instructions -------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file describes the RISC-V instructions from the standard 'Zvk',
// Vector Cryptography Instructions extension, version 1.0.0-rc1.
//
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// Operand and SDNode transformation definitions.
//===----------------------------------------------------------------------===//

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

//===----------------------------------------------------------------------===//
// Instruction class templates
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
  let Inst{6-0} = OPC_OP_V.Value;

  let Uses = [VTYPE, VL];
  let RVVConstraint = VMConstraint;
}

multiclass VROR_IV_V_X_I<string opcodestr, bits<6> funct6>
    : VALU_IV_V_X<opcodestr, funct6> {
  def I : RVInstIVI_VROR<funct6, (outs VR:$vd),
              (ins VR:$vs2, uimm6:$imm, VMaskOp:$vm),
              opcodestr # ".vi", "$vd, $vs2, $imm$vm">,
         Sched<[WriteVIALUI_WorstCase, ReadVIALUV_WorstCase,
                ReadVMask]>;
}

// op vd, vs2, vs1
class PALUVVNoVm<bits<6> funct6, RISCVVFormat opv, string opcodestr>
    : VALUVVNoVm<funct6, opv, opcodestr> {
  let Inst{6-0} = OPC_OP_P.Value;
}

// op vd, vs2, imm, vm
class PALUVINoVm<bits<6> funct6, string opcodestr, Operand optype = simm5>
    : VALUVINoVm<funct6, opcodestr, optype> {
  let Inst{6-0} = OPC_OP_P.Value;
  let Inst{14-12} = OPMVV.Value;
}

// op vd, vs2 (use vs1 as instruction encoding)
class PALUVs2NoVm<bits<6> funct6, bits<5> vs1, RISCVVFormat opv, string opcodestr>
    : VALUVs2NoVm<funct6, vs1, opv, opcodestr> {
  let Inst{6-0} = OPC_OP_P.Value;
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
  let Inst{6-0} = OPC_OP_P.Value;
  let Inst{14-12} = OPMVV.Value;
}
} // hasSideEffects = 0, mayLoad = 0, mayStore = 0

//===----------------------------------------------------------------------===//
// Instructions
//===----------------------------------------------------------------------===//

let Predicates = [HasStdExtZvbb] in {
  defm VANDN_V  : VALU_IV_V_X<"vandn", 0b000001>;
  def  VBREV8_V : VALUVs2<0b010010, 0b01000, OPMVV, "vbrev8.v">;
  def  VBREV_V  : VALUVs2<0b010010, 0b01010, OPMVV, "vbrev.v">;
  def  VCLZ_V   : VALUVs2<0b010010, 0b01100, OPMVV, "vclz.v">;
  def  VCPOP_V  : VALUVs2<0b010010, 0b01110, OPMVV, "vcpop.v">;
  def  VCTZ_V   : VALUVs2<0b010010, 0b01101, OPMVV, "vctz.v">;
  def  VREV8_V  : VALUVs2<0b010010, 0b01001, OPMVV, "vrev8.v">;
  defm VROL_V   : VALU_IV_V_X<"vrol", 0b010101>;
  defm VROR_V   : VROR_IV_V_X_I<"vror", 0b010100>;
  let Constraints = "@earlyclobber $vd", RVVConstraint = WidenV in
  defm VWSLL_V  : VSHT_IV_V_X_I<"vwsll", 0b110101>;
} // Predicates = [HasStdExtZvbb]

let Predicates = [HasStdExtZvbc] in {
  defm VCLMUL_V  : VCLMUL_MV_V_X<"vclmul", 0b001100>;
  defm VCLMULH_V : VCLMUL_MV_V_X<"vclmulh", 0b001101>;
} // Predicates = [HasStdExtZvbc]

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
  defm VAESDF     : VAES_MV_V_S<0b101000, 0b101001, 0b00001, OPMVV, "vaesdf">;
  defm VAESDM     : VAES_MV_V_S<0b101000, 0b101001, 0b00000, OPMVV, "vaesdm">;
  defm VAESEF     : VAES_MV_V_S<0b101000, 0b101001, 0b00011, OPMVV, "vaesef">;
  defm VAESEM     : VAES_MV_V_S<0b101000, 0b101001, 0b00010, OPMVV, "vaesem">;
  def  VAESKF1_VI : VAESKF_MV_I<0b100010, "vaeskf1.vi", uimm5>;
  def  VAESKF2_VI : VAESKF_MV_I<0b101010, "vaeskf2.vi", uimm5>;
  def  VAESZ_VS   : PALUVs2NoVm<0b101001, 0b00111, OPMVV, "vaesz.vs">;
} // Predicates = [HasStdExtZvkned]

let Predicates = [HasStdExtZvksed], RVVConstraint = NoConstraint in {
  def  VSM4K_VI : PALUVINoVm<0b100001, "vsm4k.vi", uimm5>;
  defm VSM4R    : VAES_MV_V_S<0b101000, 0b101001, 0b10000, OPMVV, "vsm4r">;
} // Predicates = [HasStdExtZvksed]

let Predicates = [HasStdExtZvksh], RVVConstraint = NoConstraint in {
  def VSM3C_VI  : PALUVINoVm<0b101011, "vsm3c.vi", uimm5>;
  def VSM3ME_VV : PALUVVNoVm<0b100000, OPMVV, "vsm3me.vv">;
} // Predicates = [HasStdExtZvksh]

//===----------------------------------------------------------------------===//
// Pseudo instructions
//===----------------------------------------------------------------------===//

defm PseudoVANDN : VPseudoVALU_VV_VX;

multiclass VPseudoUnaryV_V {
  foreach m = MxList in {
    let VLMul = m.value in {
      def "_V_" # m.MX : VPseudoUnaryNoMask<m.vrclass, m.vrclass>;
      def "_V_" # m.MX # "_MASK" : VPseudoUnaryMask<m.vrclass, m.vrclass>,
                                   RISCVMaskedPseudo<MaskIdx=2>;
    }
  }
}

defm PseudoVBREV : VPseudoUnaryV_V;
defm PseudoVREV8 : VPseudoUnaryV_V;
defm PseudoVCLZ : VPseudoUnaryV_V;
defm PseudoVCTZ : VPseudoUnaryV_V;
defm PseudoVCPOP : VPseudoUnaryV_V;

defm PseudoVROL : VPseudoVALU_VV_VX;
defm PseudoVROR : VPseudoVALU_VV_VX_VI<uimm6>;

//===----------------------------------------------------------------------===//
// SDNode patterns
//===----------------------------------------------------------------------===//

multiclass VPatUnarySDNode_V<SDPatternOperator op, string instruction_name> {
  foreach vti = AllIntegerVectors in {
    let Predicates = !listconcat([HasStdExtZvbb],
                                 GetVTypePredicates<vti>.Predicates) in {
      def : Pat<(vti.Vector (op (vti.Vector vti.RegClass:$rs1))),
                (!cast<Instruction>(instruction_name#"_V_"#vti.LMul.MX)
                   (vti.Vector (IMPLICIT_DEF)),
                   vti.RegClass:$rs1,
                   vti.AVL, vti.Log2SEW, TA_MA)>;
    }
  }
}

// Helpers for detecting splats since we preprocess splat_vector to vmv.v.x
// This should match the logic in RISCVDAGToDAGISel::selectVSplat
def riscv_splat_vector : PatFrag<(ops node:$rs1),
                                 (riscv_vmv_v_x_vl undef, node:$rs1, srcvalue)>;
def riscv_vnot : PatFrag<(ops node:$rs1), (xor node:$rs1,
                                               (riscv_splat_vector -1))>;

foreach vti = AllIntegerVectors in {
  let Predicates = !listconcat([HasStdExtZvbb],
                               GetVTypePredicates<vti>.Predicates) in {
    def : Pat<(vti.Vector (and (riscv_vnot vti.RegClass:$rs1),
                               vti.RegClass:$rs2)),
              (!cast<Instruction>("PseudoVANDN_VV_"#vti.LMul.MX)
                 (vti.Vector (IMPLICIT_DEF)),
                 vti.RegClass:$rs2,
                 vti.RegClass:$rs1,
                 vti.AVL, vti.Log2SEW, TA_MA)>;
    def : Pat<(vti.Vector (and (riscv_splat_vector
                                 (not vti.ScalarRegClass:$rs1)),
                               vti.RegClass:$rs2)),
              (!cast<Instruction>("PseudoVANDN_VX_"#vti.LMul.MX)
                 (vti.Vector (IMPLICIT_DEF)),
                 vti.RegClass:$rs2,
                 vti.ScalarRegClass:$rs1,
                 vti.AVL, vti.Log2SEW, TA_MA)>;
  }
}

defm : VPatUnarySDNode_V<bitreverse, "PseudoVBREV">;
defm : VPatUnarySDNode_V<bswap, "PseudoVREV8">;
defm : VPatUnarySDNode_V<ctlz, "PseudoVCLZ">;
defm : VPatUnarySDNode_V<cttz, "PseudoVCTZ">;
defm : VPatUnarySDNode_V<ctpop, "PseudoVCPOP">;

defm : VPatBinarySDNode_VV_VX<rotl, "PseudoVROL">;

def NegImm64 : SDNodeXForm<imm, [{
  return CurDAG->getTargetConstant(0x3f & (64 - N->getZExtValue()), SDLoc(N),
                                   N->getValueType(0));
}]>;

// Although there is no vrol.vi, an immediate rotate left can be achieved by
// negating the immediate in vror.vi
foreach vti = AllIntegerVectors in {
  let Predicates = !listconcat([HasStdExtZvbb],
                               GetVTypePredicates<vti>.Predicates) in {
    def : Pat<(vti.Vector (rotl vti.RegClass:$rs2,
                                (vti.Vector (SplatPat_uimm6 uimm6:$rs1)))),
              (!cast<Instruction>("PseudoVROR_VI_"#vti.LMul.MX)
                 (vti.Vector (IMPLICIT_DEF)),
                 vti.RegClass:$rs2,
                 (NegImm64 uimm6:$rs1),
                 vti.AVL, vti.Log2SEW, TA_MA)>;
  }
}
defm : VPatBinarySDNode_VV_VX_VI<rotr, "PseudoVROR", uimm6>;

//===----------------------------------------------------------------------===//
// VL patterns
//===----------------------------------------------------------------------===//

multiclass VPatUnaryVL_V<SDPatternOperator op, string instruction_name> {
  foreach vti = AllIntegerVectors in {
    let Predicates = !listconcat([HasStdExtZvbb],
                                 GetVTypePredicates<vti>.Predicates) in {
      def : Pat<(vti.Vector (op (vti.Vector vti.RegClass:$rs1),
                                (vti.Vector vti.RegClass:$merge),
                                (vti.Mask V0),
                                VLOpFrag)),
                (!cast<Instruction>(instruction_name#"_V_"#vti.LMul.MX#"_MASK")
                   vti.RegClass:$merge,
                   vti.RegClass:$rs1,
                   (vti.Mask V0),
                   GPR:$vl,
                   vti.Log2SEW,
                   TAIL_AGNOSTIC)>;
    }
  }
}

foreach vti = AllIntegerVectors in {
  let Predicates = !listconcat([HasStdExtZvbb],
                               GetVTypePredicates<vti>.Predicates) in {
    def : Pat<(vti.Vector (riscv_and_vl (riscv_xor_vl
                                           (vti.Vector vti.RegClass:$rs1),
                                           (riscv_splat_vector -1),
                                           (vti.Vector vti.RegClass:$merge),
                                           (vti.Mask V0),
                                           VLOpFrag),
                                        (vti.Vector vti.RegClass:$rs2),
                                        (vti.Vector vti.RegClass:$merge),
                                        (vti.Mask V0),
                                        VLOpFrag)),
              (!cast<Instruction>("PseudoVANDN_VV_"#vti.LMul.MX#"_MASK")
                 vti.RegClass:$merge,
                 vti.RegClass:$rs2,
                 vti.RegClass:$rs1,
                 (vti.Mask V0),
                 GPR:$vl,
                 vti.Log2SEW,
                 TAIL_AGNOSTIC)>;

    def : Pat<(vti.Vector (riscv_and_vl (riscv_splat_vector
                                           (not vti.ScalarRegClass:$rs1)),
                                        (vti.Vector vti.RegClass:$rs2),
                                        (vti.Vector vti.RegClass:$merge),
                                        (vti.Mask V0),
                                        VLOpFrag)),
              (!cast<Instruction>("PseudoVANDN_VX_"#vti.LMul.MX#"_MASK")
                 vti.RegClass:$merge,
                 vti.RegClass:$rs2,
                 vti.ScalarRegClass:$rs1,
                 (vti.Mask V0),
                 GPR:$vl,
                 vti.Log2SEW,
                 TAIL_AGNOSTIC)>;
  }
}

defm : VPatUnaryVL_V<riscv_bitreverse_vl, "PseudoVBREV">;
defm : VPatUnaryVL_V<riscv_bswap_vl, "PseudoVREV8">;
defm : VPatUnaryVL_V<riscv_ctlz_vl, "PseudoVCLZ">;
defm : VPatUnaryVL_V<riscv_cttz_vl, "PseudoVCTZ">;
defm : VPatUnaryVL_V<riscv_ctpop_vl, "PseudoVCPOP">;

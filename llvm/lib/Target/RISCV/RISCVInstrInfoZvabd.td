//===-- RISCVInstrInfoZvabd.td - 'Zvabd' instructions ------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// This file describes the RISC-V instructions for 'Zvabd' (Vector Absolute
/// Difference).
///
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// Instruction Definitions
//===----------------------------------------------------------------------===//
let Predicates = [HasStdExtZvabd], ElementsDependOn = EltDepsNone in {
  def VABD_VV  : VALUVV<0b010101, OPMVV, "vabd.vv">;
  def VABD_VX  : VALUVX<0b010101, OPMVX, "vabd.vx">;
  def VABDU_VV : VALUVV<0b010110, OPMVV, "vabdu.vv">;
  def VABDU_VX : VALUVX<0b010110, OPMVX, "vabdu.vx">;

  let Constraints = "@earlyclobber $vd", DestEEW = EEWSEWx2,
      VS1VS2Constraint = WidenV in {
    def VWABDA_VV  : VALUVV<0b111101, OPIVV, "vwabda.vv">;
    def VWABDA_VX  : VALUVX<0b111101, OPIVX, "vwabda.vx">;
    def VWABDAU_VV : VALUVV<0b111110, OPIVV, "vwabdau.vv">;
    def VWABDAU_VX : VALUVX<0b111110, OPIVX, "vwabdau.vx">;
  } // Constraints = "@earlyclobber $vd", DestEEW = EEWSEWx2,
    // VS1VS2Constraint = WidenV
} // Predicates = [HasStdExtZvabd], ElementsDependOn = EltDepsNone

let Predicates = [HasStdExtZvabd] in {
  def : InstAlias<"vabs.v $vd, $vs$vm",
                  (VABD_VX VR:$vd, VR:$vs, X0, VMaskOp:$vm)>;
  def : InstAlias<"vabs.v $vd, $vs",
                  (VABD_VX VR:$vd, VR:$vs, X0, zero_reg)>;
}

//===----------------------------------------------------------------------===//
// Pseudos
//===----------------------------------------------------------------------===//
multiclass VPseudoVWABD_VV_VX {
  foreach m = MxListW in {
    defvar mx = m.MX;
    defm "" : VPseudoTernaryW_VV<m, Commutable = 1>,
              SchedTernary<"WriteVIWMulAddV", "ReadVIWMulAddV",
                           "ReadVIWMulAddV", "ReadVIWMulAddV", mx>;
    defvar constraint = "@earlyclobber $rd";
    defm "_VX" : VPseudoTernaryWithPolicy<m.wvrclass, m.vrclass, GPR, m,
                                           constraint,
                                           TargetConstraintType = 3>,
                  SchedTernary<"WriteVIWMulAddX", "ReadVIWMulAddV",
                               "ReadVIWMulAddV", "ReadVIWMulAddX", mx>;
  }
}

let Predicates = [HasStdExtZvabd] in {
  defm PseudoVABD : VPseudoVALU_VV_VX<Commutable = 1>;
  defm PseudoVABDU : VPseudoVALU_VV_VX<Commutable = 1>;
  defm PseudoVWABDA : VPseudoVWABD_VV_VX;
  defm PseudoVWABDAU : VPseudoVWABD_VV_VX;
} // Predicates = [HasStdExtZvabd]

//===----------------------------------------------------------------------===//
// CodeGen Patterns
//===----------------------------------------------------------------------===//
let HasPassthruOp = true, HasMaskOp = true in {
def riscv_abs_vl  : RVSDNode<"ABS_VL", SDT_RISCVIntUnOp_VL>;
def riscv_abds_vl : RVSDNode<"ABDS_VL", SDT_RISCVIntBinOp_VL, [SDNPCommutative]>;
def riscv_abdu_vl : RVSDNode<"ABDU_VL", SDT_RISCVIntBinOp_VL, [SDNPCommutative]>;
def rvv_vwabda_vl  : RVSDNode<"VWABDA_VL", SDT_RISCVVWIntTernOp_VL, [SDNPCommutative]>;
def rvv_vwabdau_vl : RVSDNode<"VWABDAU_VL", SDT_RISCVVWIntTernOp_VL, [SDNPCommutative]>;
} // let HasPassthruOp = true, HasMaskOp = true

// The widening instructions are defined for SEW=8 and SEW=16, otherwise the
// instruction encoding is reserved.
defvar ABDAIntVectors = !filter(vtiTowti, AllWidenableIntVectors,
                                          !or(!eq(vtiTowti.Vti.SEW, 8),
                                              !eq(vtiTowti.Vti.SEW, 16)));

multiclass VPatTernaryW_VX_VS2First<string intrinsic, string instruction,
                                    list<VTypeInfoToWide> vtilist> {
  foreach vtiTowti = vtilist in {
    defvar vti = vtiTowti.Vti;
    defvar wti = vtiTowti.Wti;
    let Predicates = !listconcat(GetVTypePredicates<vti>.Predicates,
                                 GetVTypePredicates<wti>.Predicates) in
    defm : VPatTernaryWithPolicy<intrinsic, instruction,
                                 "V"#vti.ScalarSuffix,
                                 wti.Vector, vti.Vector, vti.Scalar,
                                 vti.Mask, vti.Log2SEW, vti.LMul,
                                 wti.RegClass, vti.RegClass,
                                 vti.ScalarRegClass>;
  }
}

multiclass VPatWidenABDVL_VV_VX<SDNode op, string instruction> {
  defm : VPatWidenMultiplyAddVL_VV<op, instruction, ABDAIntVectors>;
  foreach vtiTowti = ABDAIntVectors in {
    defvar vti = vtiTowti.Vti;
    defvar wti = vtiTowti.Wti;
    let Predicates = !listconcat(GetVTypePredicates<vti>.Predicates,
                                 GetVTypePredicates<wti>.Predicates) in
    def : Pat<(op (SplatPat XLenVT:$rs1),
                  (vti.Vector vti.RegClass:$rs2),
                  (wti.Vector wti.RegClass:$rd),
                  (vti.Mask VMV0:$vm), VLOpFrag),
              (!cast<Instruction>(instruction#"_VX_"#vti.LMul.MX#"_MASK")
                  wti.RegClass:$rd, vti.RegClass:$rs2,
                  vti.ScalarRegClass:$rs1, (vti.Mask VMV0:$vm), GPR:$vl,
                  vti.Log2SEW, TAIL_AGNOSTIC)>;
  }
}

let Predicates = [HasStdExtZvabd] in {
defm : VPatBinarySDNode_VV_VX<abds, "PseudoVABD", AllIntegerVectors>;
defm : VPatBinarySDNode_VV_VX<abdu, "PseudoVABDU", AllIntegerVectors>;

defm : VPatBinaryVL_VV_VX<riscv_abds_vl, "PseudoVABD", AllIntegerVectors>;
defm : VPatBinaryVL_VV_VX<riscv_abdu_vl, "PseudoVABDU", AllIntegerVectors>;

defm : VPatBinaryV_VV_VX<"int_riscv_vabd", "PseudoVABD", AllIntegerVectors>;
defm : VPatBinaryV_VV_VX<"int_riscv_vabdu", "PseudoVABDU", AllIntegerVectors>;
defm : VPatTernaryW_VV<"int_riscv_vwabda", "PseudoVWABDA", ABDAIntVectors>;
defm : VPatTernaryW_VV<"int_riscv_vwabdau", "PseudoVWABDAU", ABDAIntVectors>;
defm : VPatTernaryW_VX_VS2First<"int_riscv_vwabda", "PseudoVWABDA",
                                 ABDAIntVectors>;
defm : VPatTernaryW_VX_VS2First<"int_riscv_vwabdau", "PseudoVWABDAU",
                                 ABDAIntVectors>;

foreach vti = AllIntegerVectors in {
  let Predicates = GetVTypePredicates<vti>.Predicates in {
    def : Pat<(vti.Vector (abs (vti.Vector vti.RegClass:$rs2))),
              (!cast<Instruction>("PseudoVABD_VX_"#vti.LMul.MX)
                  (vti.Vector (IMPLICIT_DEF)), vti.RegClass:$rs2,
                  (XLenVT X0), vti.AVL, vti.Log2SEW, TA_MA)>;
    def : Pat<(riscv_abs_vl (vti.Vector vti.RegClass:$rs2),
                             (vti.Vector vti.RegClass:$passthru),
                             (vti.Mask VMV0:$vm), VLOpFrag),
              (!cast<Instruction>("PseudoVABD_VX_"#vti.LMul.MX#"_MASK")
                  vti.RegClass:$passthru, vti.RegClass:$rs2, (XLenVT X0),
                  (vti.Mask VMV0:$vm), GPR:$vl, vti.Log2SEW,
                  TAIL_AGNOSTIC)>;
  }
}

defm : VPatWidenABDVL_VV_VX<rvv_vwabda_vl, "PseudoVWABDA">;
defm : VPatWidenABDVL_VV_VX<rvv_vwabdau_vl, "PseudoVWABDAU">;
} // Predicates = [HasStdExtZvabd]

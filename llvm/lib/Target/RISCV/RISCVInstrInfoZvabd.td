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
let Predicates = [HasStdExtZvabd] in {
  def VABS_V : VALUVs2<0b010010, 0b10000, OPMVV, "vabs.v">;

  def VABD_VV  : VALUVV<0b010001, OPMVV, "vabd.vv">;
  def VABDU_VV : VALUVV<0b010011, OPMVV, "vabdu.vv">;

  let Constraints = "@earlyclobber $vd", RVVConstraint = WidenV in {
    def VWABDA_VV  : VALUVV<0b010101, OPMVV, "vwabda.vv">;
    def VWABDAU_VV : VALUVV<0b010110, OPMVV, "vwabdau.vv">;
  } // Constraints = "@earlyclobber $vd", RVVConstraint = WidenV
} // Predicates = [HasStdExtZvabd]

//===----------------------------------------------------------------------===//
// Pseudos
//===----------------------------------------------------------------------===//
multiclass PseudoVABS {
  foreach m = MxList in {
    defvar mx = m.MX;
    let VLMul = m.value in {
      def "_V_" # mx : VPseudoUnaryNoMask<m.vrclass, m.vrclass>,
                       SchedUnary<"WriteVIALUV", "ReadVIALUV", mx, forcePassthruRead=true>;
      def "_V_" # mx # "_MASK" :
        VPseudoUnaryMask<m.vrclass, m.vrclass>,
        RISCVMaskedPseudo<MaskIdx=2>,
        SchedUnary<"WriteVIALUV", "ReadVIALUV", mx, forcePassthruRead=true>;
    }
  }
}

multiclass VPseudoVWABD_VV {
  foreach m = MxListW in {
    defvar mx = m.MX;
    defm "" : VPseudoTernaryW_VV<m, Commutable = 1>,
              SchedTernary<"WriteVIWMulAddV", "ReadVIWMulAddV",
                           "ReadVIWMulAddV", "ReadVIWMulAddV", mx>;
  }
}

let Predicates = [HasStdExtZvabd] in {
  defm PseudoVABS : PseudoVABS;
  defm PseudoVABD : VPseudoVALU_VV<Commutable = 1>;
  defm PseudoVABDU : VPseudoVALU_VV<Commutable = 1>;
  defm PseudoVWABDA : VPseudoVWABD_VV;
  defm PseudoVWABDAU : VPseudoVWABD_VV;
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

// These instructions are defined for SEW=8 and SEW=16, otherwise the instruction
// encoding is reserved.
defvar ABDIntVectors = !filter(vti, AllIntegerVectors, !or(!eq(vti.SEW, 8),
                                                           !eq(vti.SEW, 16)));
defvar ABDAIntVectors = !filter(vtiTowti, AllWidenableIntVectors,
                                          !or(!eq(vtiTowti.Vti.SEW, 8),
                                              !eq(vtiTowti.Vti.SEW, 16)));

let Predicates = [HasStdExtZvabd] in {
defm : VPatBinarySDNode_VV<abds, "PseudoVABD", ABDIntVectors>;
defm : VPatBinarySDNode_VV<abdu, "PseudoVABDU", ABDIntVectors>;

defm : VPatBinaryVL_VV<riscv_abds_vl, "PseudoVABD", ABDIntVectors>;
defm : VPatBinaryVL_VV<riscv_abdu_vl, "PseudoVABDU", ABDIntVectors>;

defm : VPatUnaryV_V<"int_riscv_vabs", "PseudoVABS", AllIntegerVectors>;
defm : VPatBinaryV_VV<"int_riscv_vabd", "PseudoVABD", ABDIntVectors>;
defm : VPatBinaryV_VV<"int_riscv_vabdu", "PseudoVABDU", ABDIntVectors>;
defm : VPatTernaryW_VV<"int_riscv_vwabda", "PseudoVWABDA", ABDAIntVectors>;
defm : VPatTernaryW_VV<"int_riscv_vwabdau", "PseudoVWABDAU", ABDAIntVectors>;

foreach vti = AllIntegerVectors in {
  def : Pat<(vti.Vector (abs (vti.Vector vti.RegClass:$rs2))),
            (!cast<Instruction>("PseudoVABS_V_"#vti.LMul.MX)
                    (vti.Vector (IMPLICIT_DEF)),
                    vti.RegClass:$rs2, vti.AVL, vti.Log2SEW, TA_MA)>;
}

defm : VPatUnaryVL_V<riscv_abs_vl, "PseudoVABS", HasStdExtZvabd>;

defm : VPatWidenMultiplyAddVL_VV<rvv_vwabda_vl, "PseudoVWABDA", ABDAIntVectors>;
defm : VPatWidenMultiplyAddVL_VV<rvv_vwabdau_vl, "PseudoVWABDAU", ABDAIntVectors>;
} // Predicates = [HasStdExtZvabd]

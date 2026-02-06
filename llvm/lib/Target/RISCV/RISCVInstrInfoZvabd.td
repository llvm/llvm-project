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
  defm VABS_V : VALU_MV_VS2<"vabs.v", 0b010010, 0b10000>;

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

let Predicates = [HasStdExtZvabd] in {
  defm PseudoVABS : PseudoVABS;
  defm PseudoVABD : VPseudoVALU_VV<Commutable = 1>;
  defm PseudoVABDU : VPseudoVALU_VV<Commutable = 1>;
} // Predicates = [HasStdExtZvabd]

//===----------------------------------------------------------------------===//
// CodeGen Patterns
//===----------------------------------------------------------------------===//
let HasPassthruOp = true, HasMaskOp = true in {
def riscv_abs_vl  : RVSDNode<"ABS_VL", SDT_RISCVIntUnOp_VL>;
def riscv_abds_vl : RVSDNode<"ABDS_VL", SDT_RISCVIntBinOp_VL, [SDNPCommutative]>;
def riscv_abdu_vl : RVSDNode<"ABDU_VL", SDT_RISCVIntBinOp_VL, [SDNPCommutative]>;
} // let HasPassthruOp = true, HasMaskOp = true

// These instructions are defined for SEW=8 and SEW=16, otherwise the instruction
// encoding is reserved.
defvar ABDIntVectors = !filter(vti, AllIntegerVectors, !or(!eq(vti.SEW, 8),
                                                           !eq(vti.SEW, 16)));

let Predicates = [HasStdExtZvabd] in {
defm : VPatBinarySDNode_VV<abds, "PseudoVABD", ABDIntVectors>;
defm : VPatBinarySDNode_VV<abdu, "PseudoVABDU", ABDIntVectors>;

defm : VPatBinaryVL_VV<riscv_abds_vl, "PseudoVABD", ABDIntVectors>;
defm : VPatBinaryVL_VV<riscv_abdu_vl, "PseudoVABDU", ABDIntVectors>;

foreach vti = AllIntegerVectors in {
  def : Pat<(vti.Vector (abs (vti.Vector vti.RegClass:$rs2))),
            (!cast<Instruction>("PseudoVABS_V_"#vti.LMul.MX)
                    (vti.Vector (IMPLICIT_DEF)),
                    vti.RegClass:$rs2, vti.AVL, vti.Log2SEW, TA_MA)>;
}

defm : VPatUnaryVL_V<riscv_abs_vl, "PseudoVABS", HasStdExtZvabd>;
} // Predicates = [HasStdExtZvabd]

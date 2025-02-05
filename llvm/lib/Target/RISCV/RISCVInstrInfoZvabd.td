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
  defm VABD_V : VAALU_MV_V_X<"vabd", 0b010001>;
  defm VABDU_V : VAALU_MV_V_X<"vabdu", 0b010011>;

  let Constraints = "@earlyclobber $vd", RVVConstraint = WidenV in {
    defm VWABDACC_V : VALU_MV_V_X<"vwabdacc", 0b010101, "v">;
    defm VWABDACCU_V : VALU_MV_V_X<"vwabdaccu", 0b010110, "v">;
  } // Constraints = "@earlyclobber $vd", RVVConstraint = WidenV

  def : InstAlias<"vabs.v $vd, $vs$vm", (VABD_VX VR:$vd, VR:$vs, X0, VMaskOp:$vm)>;
  def : InstAlias<"vabs.v $vd, $vs", (VABD_VX VR:$vd, VR:$vs, X0, zero_reg)>;
} // Predicates = [HasStdExtZvabd]

//===----------------------------------------------------------------------===//
// Pseudos
//===----------------------------------------------------------------------===//
let Predicates = [HasStdExtZvabd] in {
  defm PseudoVABD : VPseudoVALU_VV_VX<Commutable = 1>;
  defm PseudoVABDU : VPseudoVALU_VV_VX<Commutable = 1>;
} // Predicates = [HasStdExtZvabd]

//===----------------------------------------------------------------------===//
// CodeGen Patterns
//===----------------------------------------------------------------------===//
let HasPassthruOp = true, HasMaskOp = true in {
def riscv_abds_vl : RVSDNode<"ABDS_VL", SDT_RISCVIntBinOp_VL, [SDNPCommutative]>;
def riscv_abdu_vl : RVSDNode<"ABDU_VL", SDT_RISCVIntBinOp_VL, [SDNPCommutative]>;
} // let HasPassthruOp = true, HasMaskOp = true

let Predicates = [HasStdExtZvabd] in {
defm : VPatBinarySDNode_VV_VX<abds, "PseudoVABD">;
defm : VPatBinarySDNode_VV_VX<abdu, "PseudoVABDU">;

foreach vti = AllIntegerVectors in {
  def : Pat<(vti.Vector (abs (vti.Vector vti.RegClass:$rs2))),
            (!cast<Instruction>("PseudoVABD_VX_"#vti.LMul.MX)
                    (vti.Vector (IMPLICIT_DEF)),
                vti.RegClass:$rs2, (XLenVT X0), vti.AVL, vti.Log2SEW, TA_MA)>;
}

defm : VPatBinaryVL_VV_VX<riscv_abds_vl, "PseudoVABD">;
defm : VPatBinaryVL_VV_VX<riscv_abdu_vl, "PseudoVABDU">;
} // Predicates = [HasStdExtZvabd]

//===-- RISCVInstrInfoZvqdot.td - 'Zvqdotq' instructions ---*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file describes the RISC-V instructions from the standard 'Zvqdotq'
// extension.
// This version is still experimental as the 'Zvqdotq' extension hasn't been
// ratified yet.
//
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// Instructions
//===----------------------------------------------------------------------===//

let Predicates = [HasStdExtZvqdotq] in {
  def VQDOT_VV   : VALUVV<0b101100, OPMVV, "vqdot.vv">;
  def VQDOT_VX   : VALUVX<0b101100, OPMVX, "vqdot.vx">;
  def VQDOTU_VV  : VALUVV<0b101000, OPMVV, "vqdotu.vv">;
  def VQDOTU_VX  : VALUVX<0b101000, OPMVX, "vqdotu.vx">;
  def VQDOTSU_VV : VALUVV<0b101010, OPMVV, "vqdotsu.vv">;
  def VQDOTSU_VX : VALUVX<0b101010, OPMVX, "vqdotsu.vx">;
  def VQDOTUS_VX : VALUVX<0b101110, OPMVX, "vqdotus.vx">;
} // Predicates = [HasStdExtZvqdotq]


let HasPassthruOp = true, HasMaskOp = true in {
  def riscv_vqdot_vl : SDNode<"RISCVISD::VQDOT_VL", SDT_RISCVIntBinOp_VL>;
  def riscv_vqdotu_vl : SDNode<"RISCVISD::VQDOTU_VL", SDT_RISCVIntBinOp_VL>;
  def riscv_vqdotsu_vl : SDNode<"RISCVISD::VQDOTSU_VL", SDT_RISCVIntBinOp_VL>;
} // let HasPassthruOp = true, HasMaskOp = true

multiclass VPseudoVQDOT_VV_VX {
  foreach m = MxSet<32>.m in {
    defm "" : VPseudoBinaryV_VV<m>,
            SchedBinary<"WriteVIALUV", "ReadVIALUV", "ReadVIALUV", m.MX,
                        forcePassthruRead=true>;
    defm "" : VPseudoBinaryV_VX<m>,
            SchedBinary<"WriteVIALUX", "ReadVIALUV", "ReadVIALUX", m.MX,
                        forcePassthruRead=true>;
  }
}

// TODO: Add pseudo and patterns for vqdotus.vx
// TODO: Add isCommutable for VQDOT and VQDOTU
let Predicates = [HasStdExtZvqdotq], mayLoad = 0, mayStore = 0,
    hasSideEffects = 0 in {
  defm PseudoVQDOT : VPseudoVQDOT_VV_VX;
  defm PseudoVQDOTU : VPseudoVQDOT_VV_VX;
  defm PseudoVQDOTSU : VPseudoVQDOT_VV_VX;
}

defvar AllE32Vectors = [VI32MF2, VI32M1, VI32M2, VI32M4, VI32M8];
defm : VPatBinaryVL_VV_VX<riscv_vqdot_vl, "PseudoVQDOT", AllE32Vectors>;
defm : VPatBinaryVL_VV_VX<riscv_vqdotu_vl, "PseudoVQDOTU", AllE32Vectors>;
defm : VPatBinaryVL_VV_VX<riscv_vqdotsu_vl, "PseudoVQDOTSU", AllE32Vectors>;


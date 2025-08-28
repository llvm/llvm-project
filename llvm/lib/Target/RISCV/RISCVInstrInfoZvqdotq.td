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

class VQDOTVV<bits<6> funct6, RISCVVFormat opv, string opcodestr>
    : RVInstVV<funct6, opv, (outs VR:$vd_wb),
                (ins VR:$vd, VR:$vs2, VR:$vs1, VMaskOp:$vm),
                opcodestr, "$vd, $vs2, $vs1$vm"> {
  let mayLoad = 0;
  let mayStore = 0;
  let hasSideEffects = 0;
  let Constraints = "$vd = $vd_wb";
}

class VQDOTVX<bits<6> funct6, RISCVVFormat opv, string opcodestr>
    : RVInstVX<funct6, opv, (outs VR:$vd_wb),
                (ins VR:$vd, VR:$vs2, GPR:$rs1, VMaskOp:$vm),
                opcodestr, "$vd, $vs2, $rs1$vm"> {
  let mayLoad = 0;
  let mayStore = 0;
  let hasSideEffects = 0;
  let Constraints = "$vd = $vd_wb";
}

let Predicates = [HasStdExtZvqdotq] in {
  def VQDOT_VV   : VQDOTVV<0b101100, OPMVV, "vqdot.vv">;
  def VQDOT_VX   : VQDOTVX<0b101100, OPMVX, "vqdot.vx">;
  def VQDOTU_VV  : VQDOTVV<0b101000, OPMVV, "vqdotu.vv">;
  def VQDOTU_VX  : VQDOTVX<0b101000, OPMVX, "vqdotu.vx">;
  def VQDOTSU_VV : VQDOTVV<0b101010, OPMVV, "vqdotsu.vv">;
  def VQDOTSU_VX : VQDOTVX<0b101010, OPMVX, "vqdotsu.vx">;
  def VQDOTUS_VX : VQDOTVX<0b101110, OPMVX, "vqdotus.vx">;
} // Predicates = [HasStdExtZvqdotq]

//===----------------------------------------------------------------------===//
// Helpers to define the VL patterns.
//===----------------------------------------------------------------------===//

let HasPassthruOp = true, HasMaskOp = true in {
  def riscv_vqdot_vl : RVSDNode<"VQDOT_VL", SDT_RISCVIntBinOp_VL>;
  def riscv_vqdotu_vl : RVSDNode<"VQDOTU_VL", SDT_RISCVIntBinOp_VL>;
  def riscv_vqdotsu_vl : RVSDNode<"VQDOTSU_VL", SDT_RISCVIntBinOp_VL>;
} // let HasPassthruOp = true, HasMaskOp = true

//===----------------------------------------------------------------------===//
// Pseudo Instructions for CodeGen
//===----------------------------------------------------------------------===//

multiclass VPseudoVQDOT_VV_VX {
  foreach m = MxSet<32>.m in {
    defm "" : VPseudoBinaryV_VV<m>,
              SchedBinary<"WriteVIMulAddV", "ReadVIMulAddV", "ReadVIMulAddV", m.MX,
                          forcePassthruRead=true>;
    defm "" : VPseudoBinaryV_VX<m>,
              SchedBinary<"WriteVIMulAddX", "ReadVIMulAddV", "ReadVIMulAddX", m.MX,
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

//===----------------------------------------------------------------------===//
// Patterns.
//===----------------------------------------------------------------------===//

defvar AllE32Vectors = [VI32MF2, VI32M1, VI32M2, VI32M4, VI32M8];
defm : VPatBinaryVL_VV_VX<riscv_vqdot_vl, "PseudoVQDOT", AllE32Vectors>;
defm : VPatBinaryVL_VV_VX<riscv_vqdotu_vl, "PseudoVQDOTU", AllE32Vectors>;
defm : VPatBinaryVL_VV_VX<riscv_vqdotsu_vl, "PseudoVQDOTSU", AllE32Vectors>;


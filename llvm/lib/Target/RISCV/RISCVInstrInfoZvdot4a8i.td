//==-- RISCVInstrInfoZvdot4a8i.td - 'Zvdot4a8i' instructions -*- tablegen -*-=//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file describes the RISC-V instructions from the standard 'Zvdot4a8i'
// extension.
// This version is still experimental as the 'Zvdot4a8i' extension hasn't been
// ratified yet.
//
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// Instructions
//===----------------------------------------------------------------------===//

class VDOT4AVV<bits<6> funct6, RISCVVFormat opv, string opcodestr>
    : RVInstVV<funct6, opv, (outs VR:$vd_wb),
                (ins VR:$vd, VR:$vs2, VR:$vs1, VMaskOp:$vm),
                opcodestr, "$vd, $vs2, $vs1$vm"> {
  let mayLoad = 0;
  let mayStore = 0;
  let hasSideEffects = 0;
  let Constraints = "$vd = $vd_wb";
}

class VDOT4AVX<bits<6> funct6, RISCVVFormat opv, string opcodestr>
    : RVInstVX<funct6, opv, (outs VR:$vd_wb),
                (ins VR:$vd, VR:$vs2, GPR:$rs1, VMaskOp:$vm),
                opcodestr, "$vd, $vs2, $rs1$vm"> {
  let mayLoad = 0;
  let mayStore = 0;
  let hasSideEffects = 0;
  let Constraints = "$vd = $vd_wb";
}

let Predicates = [HasStdExtZvdot4a8i] in {
  def VDOT4A_VV   : VDOT4AVV<0b101100, OPMVV, "vdot4a.vv">;
  def VDOT4A_VX   : VDOT4AVX<0b101100, OPMVX, "vdot4a.vx">;
  def VDOT4AU_VV  : VDOT4AVV<0b101000, OPMVV, "vdot4au.vv">;
  def VDOT4AU_VX  : VDOT4AVX<0b101000, OPMVX, "vdot4au.vx">;
  def VDOT4ASU_VV : VDOT4AVV<0b101010, OPMVV, "vdot4asu.vv">;
  def VDOT4ASU_VX : VDOT4AVX<0b101010, OPMVX, "vdot4asu.vx">;
  def VDOT4AUS_VX : VDOT4AVX<0b101110, OPMVX, "vdot4aus.vx">;
} // Predicates = [HasStdExtZvdot4a8i]

//===----------------------------------------------------------------------===//
// Helpers to define the VL patterns.
//===----------------------------------------------------------------------===//

let HasPassthruOp = true, HasMaskOp = true in {
  def riscv_vdot4a_vl : RVSDNode<"VDOT4A_VL", SDT_RISCVIntBinOp_VL>;
  def riscv_vdot4au_vl : RVSDNode<"VDOT4AU_VL", SDT_RISCVIntBinOp_VL>;
  def riscv_vdot4asu_vl : RVSDNode<"VDOT4ASU_VL", SDT_RISCVIntBinOp_VL>;
} // let HasPassthruOp = true, HasMaskOp = true

//===----------------------------------------------------------------------===//
// Pseudo Instructions for CodeGen
//===----------------------------------------------------------------------===//

multiclass VPseudoVDOT4A_VV_VX<bit Commutable = 0> {
  foreach m = MxSet<32>.m in {
    defm "" : VPseudoBinaryV_VV<m, Commutable=Commutable>,
              SchedBinary<"WriteVIMulAddV", "ReadVIMulAddV", "ReadVIMulAddV", m.MX,
                          forcePassthruRead=true>;
    defm "" : VPseudoBinaryV_VX<m>,
              SchedBinary<"WriteVIMulAddX", "ReadVIMulAddV", "ReadVIMulAddX", m.MX,
                          forcePassthruRead=true>;
  }
}

let Predicates = [HasStdExtZvdot4a8i], mayLoad = 0, mayStore = 0,
    hasSideEffects = 0 in {
  defm PseudoVDOT4A : VPseudoVDOT4A_VV_VX<Commutable=1>;
  defm PseudoVDOT4AU : VPseudoVDOT4A_VV_VX<Commutable=1>;
  defm PseudoVDOT4ASU : VPseudoVDOT4A_VV_VX;
  // VDOT4AUS does not have a VV variant
  foreach m = MxListVF4 in {
    defm "PseudoVDOT4AUS_VX" : VPseudoTernaryWithPolicy<m.vrclass, m.vrclass, GPR, m>;
  }
}

//===----------------------------------------------------------------------===//
// Patterns.
//===----------------------------------------------------------------------===//

defvar AllE32Vectors = [VI32MF2, VI32M1, VI32M2, VI32M4, VI32M8];
defm : VPatBinaryVL_VV_VX<riscv_vdot4a_vl, "PseudoVDOT4A", AllE32Vectors, ExtraPreds=[HasStdExtZvdot4a8i]>;
defm : VPatBinaryVL_VV_VX<riscv_vdot4au_vl, "PseudoVDOT4AU", AllE32Vectors, ExtraPreds=[HasStdExtZvdot4a8i]>;
defm : VPatBinaryVL_VV_VX<riscv_vdot4asu_vl, "PseudoVDOT4ASU", AllE32Vectors, ExtraPreds=[HasStdExtZvdot4a8i]>;

// These VPat definitions are for vdot4a because they have a different operand
// order with other ternary instructions (i.e. vop.vx vd, vs2, rs1)
multiclass VPatTernaryV_VX_AAAX<string intrinsic, string instruction,
                                list<VTypeInfo> vtilist,
                                list<Predicate> ExtraPreds> {
  foreach vti = vtilist in
    let Predicates = !listconcat(ExtraPreds, GetVTypePredicates<vti>.Predicates) in
    defm : VPatTernaryWithPolicy<intrinsic, instruction,
                                 "V"#vti.ScalarSuffix,
                                 vti.Vector, vti.Vector, vti.Scalar,
                                 vti.Mask, vti.Log2SEW, vti.LMul,
                                 vti.RegClass, vti.RegClass,
                                 vti.ScalarRegClass>;
}

multiclass VPatTernaryV_VV_AAAX<string intrinsic, string instruction,
                                list<VTypeInfo> vtilist,
                                list<Predicate> ExtraPreds> {
  foreach vti = vtilist in
    let Predicates = !listconcat(ExtraPreds, GetVTypePredicates<vti>.Predicates) in
    defm : VPatTernaryWithPolicy<intrinsic, instruction,
                                 "VV",
                                 vti.Vector, vti.Vector, vti.Vector,
                                 vti.Mask, vti.Log2SEW, vti.LMul,
                                 vti.RegClass, vti.RegClass,
                                 vti.RegClass>;
}

multiclass VPatTernaryV_VV_VX_AAAX<string intrinsic, string instruction,
                                   list<VTypeInfo> vtilist,
                                   list<Predicate> ExtraPreds>
    : VPatTernaryV_VV_AAAX<intrinsic, instruction, vtilist, ExtraPreds>,
      VPatTernaryV_VX_AAAX<intrinsic, instruction, vtilist, ExtraPreds>;

defm : VPatTernaryV_VV_VX_AAAX<"int_riscv_vdot4a", "PseudoVDOT4A", AllE32Vectors, [HasStdExtZvdot4a8i]>;
defm : VPatTernaryV_VV_VX_AAAX<"int_riscv_vdot4au", "PseudoVDOT4AU", AllE32Vectors, [HasStdExtZvdot4a8i]>;
defm : VPatTernaryV_VV_VX_AAAX<"int_riscv_vdot4asu", "PseudoVDOT4ASU", AllE32Vectors, [HasStdExtZvdot4a8i]>;
defm : VPatTernaryV_VX_AAAX<"int_riscv_vdot4aus", "PseudoVDOT4AUS", AllE32Vectors, [HasStdExtZvdot4a8i]>;

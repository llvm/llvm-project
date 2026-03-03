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

class VDOTA4VV<bits<6> funct6, RISCVVFormat opv, string opcodestr>
    : RVInstVV<funct6, opv, (outs VR:$vd_wb),
                (ins VR:$vd, VR:$vs2, VR:$vs1, VMaskOp:$vm),
                opcodestr, "$vd, $vs2, $vs1$vm"> {
  let mayLoad = 0;
  let mayStore = 0;
  let hasSideEffects = 0;
  let Constraints = "$vd = $vd_wb";
}

class VDOTA4VX<bits<6> funct6, RISCVVFormat opv, string opcodestr>
    : RVInstVX<funct6, opv, (outs VR:$vd_wb),
                (ins VR:$vd, VR:$vs2, GPR:$rs1, VMaskOp:$vm),
                opcodestr, "$vd, $vs2, $rs1$vm"> {
  let mayLoad = 0;
  let mayStore = 0;
  let hasSideEffects = 0;
  let Constraints = "$vd = $vd_wb";
}

let Predicates = [HasStdExtZvdot4a8i] in {
  def VDOTA4_VV   : VDOTA4VV<0b101100, OPMVV, "vdota4.vv">;
  def VDOTA4_VX   : VDOTA4VX<0b101100, OPMVX, "vdota4.vx">;
  def VDOTA4U_VV  : VDOTA4VV<0b101000, OPMVV, "vdota4u.vv">;
  def VDOTA4U_VX  : VDOTA4VX<0b101000, OPMVX, "vdota4u.vx">;
  def VDOTA4SU_VV : VDOTA4VV<0b101010, OPMVV, "vdota4su.vv">;
  def VDOTA4SU_VX : VDOTA4VX<0b101010, OPMVX, "vdota4su.vx">;
  def VDOTA4US_VX : VDOTA4VX<0b101110, OPMVX, "vdota4us.vx">;
} // Predicates = [HasStdExtZvdot4a8i]

//===----------------------------------------------------------------------===//
// Helpers to define the VL patterns.
//===----------------------------------------------------------------------===//

let HasPassthruOp = true, HasMaskOp = true in {
  def riscv_vdota4_vl : RVSDNode<"VDOTA4_VL", SDT_RISCVIntBinOp_VL>;
  def riscv_vdota4u_vl : RVSDNode<"VDOTA4U_VL", SDT_RISCVIntBinOp_VL>;
  def riscv_vdota4su_vl : RVSDNode<"VDOTA4SU_VL", SDT_RISCVIntBinOp_VL>;
} // let HasPassthruOp = true, HasMaskOp = true

//===----------------------------------------------------------------------===//
// Pseudo Instructions for CodeGen
//===----------------------------------------------------------------------===//

multiclass VPseudoVDOTA4_VV_VX {
  foreach m = MxSet<32>.m in {
    defm "" : VPseudoBinaryV_VV<m>,
              SchedBinary<"WriteVIMulAddV", "ReadVIMulAddV", "ReadVIMulAddV", m.MX,
                          forcePassthruRead=true>;
    defm "" : VPseudoBinaryV_VX<m>,
              SchedBinary<"WriteVIMulAddX", "ReadVIMulAddV", "ReadVIMulAddX", m.MX,
                          forcePassthruRead=true>;
  }
}

// TODO: Add isCommutable for VDOTA4 and VDOTA4U
let Predicates = [HasStdExtZvdot4a8i], mayLoad = 0, mayStore = 0,
    hasSideEffects = 0 in {
  defm PseudoVDOTA4 : VPseudoVDOTA4_VV_VX;
  defm PseudoVDOTA4U : VPseudoVDOTA4_VV_VX;
  defm PseudoVDOTA4SU : VPseudoVDOTA4_VV_VX;
  // VDOTA4US does not have a VV variant
  foreach m = MxListVF4 in {
    defm "PseudoVDOTA4US_VX" : VPseudoTernaryWithPolicy<m.vrclass, m.vrclass, GPR, m>;
  }
}

//===----------------------------------------------------------------------===//
// Patterns.
//===----------------------------------------------------------------------===//

defvar AllE32Vectors = [VI32MF2, VI32M1, VI32M2, VI32M4, VI32M8];
defm : VPatBinaryVL_VV_VX<riscv_vdota4_vl, "PseudoVDOTA4", AllE32Vectors, ExtraPreds=[HasStdExtZvdot4a8i]>;
defm : VPatBinaryVL_VV_VX<riscv_vdota4u_vl, "PseudoVDOTA4U", AllE32Vectors, ExtraPreds=[HasStdExtZvdot4a8i]>;
defm : VPatBinaryVL_VV_VX<riscv_vdota4su_vl, "PseudoVDOTA4SU", AllE32Vectors, ExtraPreds=[HasStdExtZvdot4a8i]>;

// These VPat definitions are for vdota4 because they have a different operand
// order with other ternary instructions (i.e. vop.vx vd, vs2, rs1)
multiclass VPatTernaryV_VX_AABX<string intrinsic, string instruction,
                                list<VTypeInfoToWide> info_pairs,
                                list<Predicate> ExtraPreds> {
  foreach pair = info_pairs in {
    defvar VdInfo = pair.Wti;
    defvar Vs2Info = pair.Vti;
    let Predicates = !listconcat(ExtraPreds, GetVTypePredicates<VdInfo>.Predicates) in
    defm : VPatTernaryWithPolicy<intrinsic, instruction,
                                 "V"#VdInfo.ScalarSuffix,
                                 VdInfo.Vector, Vs2Info.Vector, Vs2Info.Scalar,
                                 VdInfo.Mask, VdInfo.Log2SEW, VdInfo.LMul,
                                 VdInfo.RegClass, Vs2Info.RegClass,
                                 Vs2Info.ScalarRegClass>;
  }
}

multiclass VPatTernaryV_VV_AABX<string intrinsic, string instruction,
                                list<VTypeInfoToWide> info_pairs,
                                list<Predicate> ExtraPreds> {
  foreach pair = info_pairs in {
    defvar VdInfo = pair.Wti;
    defvar Vs2Info = pair.Vti;
    let Predicates = !listconcat(ExtraPreds, GetVTypePredicates<VdInfo>.Predicates) in
    defm : VPatTernaryWithPolicy<intrinsic, instruction,
                                 "VV",
                                 VdInfo.Vector, Vs2Info.Vector, Vs2Info.Vector,
                                 VdInfo.Mask, VdInfo.Log2SEW, VdInfo.LMul,
                                 VdInfo.RegClass, Vs2Info.RegClass,
                                 Vs2Info.RegClass>;
  }
}

multiclass VPatTernaryV_VV_VX_AABX<string intrinsic, string instruction,
                                   list<VTypeInfoToWide> info_pairs,
                                   list<Predicate> ExtraPreds>
    : VPatTernaryV_VV_AABX<intrinsic, instruction, info_pairs, ExtraPreds>,
      VPatTernaryV_VX_AABX<intrinsic, instruction, info_pairs, ExtraPreds>;

defset list<VTypeInfoToWide> VDOTA4InfoPairs = {
  def : VTypeInfoToWide<VI8MF2, VI32MF2>;
  def : VTypeInfoToWide<VI8M1, VI32M1>;
  def : VTypeInfoToWide<VI8M2, VI32M2>;
  def : VTypeInfoToWide<VI8M4, VI32M4>;
  def : VTypeInfoToWide<VI8M8, VI32M8>;
}

defm : VPatTernaryV_VV_VX_AABX<"int_riscv_vdota4", "PseudoVDOTA4", VDOTA4InfoPairs, [HasStdExtZvdot4a8i]>;
defm : VPatTernaryV_VV_VX_AABX<"int_riscv_vdota4u", "PseudoVDOTA4U", VDOTA4InfoPairs, [HasStdExtZvdot4a8i]>;
defm : VPatTernaryV_VV_VX_AABX<"int_riscv_vdota4su", "PseudoVDOTA4SU", VDOTA4InfoPairs, [HasStdExtZvdot4a8i]>;
defm : VPatTernaryV_VX_AABX<"int_riscv_vdota4us", "PseudoVDOTA4US", VDOTA4InfoPairs, [HasStdExtZvdot4a8i]>;

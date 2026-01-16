//===- RISCVInstrInfoZvfofp8min.td - 'Zvfofp8min' ----------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file describes the RISC-V instructions from the standard 'Zvfofp8min'
// extension, providing vector conversion instructions for OFP8.
// This version is still experimental as the 'Zvfofp8min' extension hasn't been
// ratified yet.
//
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// Instructions
//===----------------------------------------------------------------------===//

let Predicates = [HasStdExtZvfofp8min], Constraints = "@earlyclobber $vd",
    mayRaiseFPException = true, Uses = [FRM, VL, VTYPE] in {
  defm VFNCVTBF16_SAT_F_F_W
      : VNCVTF_FV_VS2<"vfncvtbf16.sat.f.f.w", 0b010010, 0b11111>;
  defm VFNCVT_F_F_Q : VNCVTF_FV_VS2<"vfncvt.f.f.q", 0b010010, 0b11001>;
  defm VFNCVT_SAT_F_F_Q : VNCVTF_FV_VS2<"vfncvt.sat.f.f.q", 0b010010, 0b11011>;
}

//===----------------------------------------------------------------------===//
// Pseudo instructions
//===----------------------------------------------------------------------===//
defvar MxListQ = [V_MF8, V_MF4, V_MF2, V_M1, V_M2];

defset list<VTypeInfoToWide> AllWidenableIntToBFloatVectors = {
  def : VTypeInfoToWide<VI8MF8, VBF16MF4>;
  def : VTypeInfoToWide<VI8MF4, VBF16MF2>;
  def : VTypeInfoToWide<VI8MF2, VBF16M1>;
  def : VTypeInfoToWide<VI8M1, VBF16M2>;
  def : VTypeInfoToWide<VI8M2, VBF16M4>;
  def : VTypeInfoToWide<VI8M4, VBF16M8>;
}

defset list<VTypeInfoToWide> AllWidenableInt8ToFloat32Vectors = {
  def : VTypeInfoToWide<VI8MF8, VF32MF2>;
  def : VTypeInfoToWide<VI8MF4, VF32M1>;
  def : VTypeInfoToWide<VI8MF2, VF32M2>;
  def : VTypeInfoToWide<VI8M1, VF32M4>;
  def : VTypeInfoToWide<VI8M2, VF32M8>;
}

class QVRClass<LMULInfo m> {
  LMULInfo c = !cond(!eq(m, V_MF8): V_MF2,
                     !eq(m, V_MF4): V_M1,
                     !eq(m, V_MF2): V_M2,
                     !eq(m, V_M1): V_M4,
                     !eq(m, V_M2): V_M8);
}

multiclass VPseudoVWCVTD_V_NoSched_Zvfofp8min {
  defvar constraint = "@earlyclobber $rd";
  foreach m = MxListW in {
    defm _V : VPseudoConversion<m.wvrclass, m.vrclass, m, constraint, sew=8,
                                TargetConstraintType=3>;
  }
}

multiclass VPseudoVNCVTD_W_RM_NoSched_Zvfofp8min {
  defvar constraint = "@earlyclobber $rd";
  foreach m = MxListW in {
    defm _W : VPseudoConversionRoundingMode<m.vrclass, m.wvrclass, m,
                                            constraint, sew=8,
                                            TargetConstraintType=2>;
  }
}

multiclass VPseudoVNCVTD_Q_RM_NoSched_Zvfofp8min {
  defvar constraint = "@earlyclobber $rd";
  foreach m = MxListQ in {
    defm _Q : VPseudoConversionRoundingMode<m.vrclass, QVRClass<m>.c.vrclass, m,
                                            constraint, sew=8,
                                            TargetConstraintType=2>;
  }
}

let Predicates = [HasStdExtZvfofp8min] in {
  let AltFmtType = IS_NOT_ALTFMT in
    defm PseudoVFWCVTBF16_F_F : VPseudoVWCVTD_V_NoSched_Zvfofp8min;
  let AltFmtType = IS_ALTFMT in
    defm PseudoVFWCVTBF16_F_F_ALT : VPseudoVWCVTD_V_NoSched_Zvfofp8min;
  let mayRaiseFPException = true in {
    let AltFmtType = IS_NOT_ALTFMT in {
      defm PseudoVFNCVTBF16_F_F :     VPseudoVNCVTD_W_RM_NoSched_Zvfofp8min;
      defm PseudoVFNCVTBF16_SAT_F_F : VPseudoVNCVTD_W_RM_NoSched_Zvfofp8min;
      defm PseudoVFNCVT_F_F :         VPseudoVNCVTD_Q_RM_NoSched_Zvfofp8min;
      defm PseudoVFNCVT_SAT_F_F :     VPseudoVNCVTD_Q_RM_NoSched_Zvfofp8min;
    }
    let AltFmtType = IS_ALTFMT in {
      defm PseudoVFNCVTBF16_F_F_ALT :     VPseudoVNCVTD_W_RM_NoSched_Zvfofp8min;
      defm PseudoVFNCVTBF16_SAT_F_F_ALT : VPseudoVNCVTD_W_RM_NoSched_Zvfofp8min;
      defm PseudoVFNCVT_F_F_ALT :         VPseudoVNCVTD_Q_RM_NoSched_Zvfofp8min;
      defm PseudoVFNCVT_SAT_F_F_ALT :     VPseudoVNCVTD_Q_RM_NoSched_Zvfofp8min;
    }
  }
}

//===----------------------------------------------------------------------===//
// Patterns
//===----------------------------------------------------------------------===//
multiclass VPatConversionQF_RM<string intrinsic, string instruction,
                               bit isSEWAware = 0> {
  foreach fvtiToFWti = AllWidenableInt8ToFloat32Vectors in {
    defvar fvti = fvtiToFWti.Vti;
    defvar fwti = fvtiToFWti.Wti;
    let Predicates = [HasStdExtZvfofp8min] in
    defm : VPatConversionRoundingMode<intrinsic, instruction, "Q",
                                      fvti.Vector, fwti.Vector, fvti.Mask, fvti.Log2SEW,
                                      fvti.LMul, fvti.RegClass, fwti.RegClass,
                                      isSEWAware>;
  }
}

let Predicates = [HasStdExtZvfofp8min] in {
  // OFP8 to BF16 conversion instructions
  defm : VPatConversionWF_VF<"int_riscv_vfwcvt_f_f_v",
                             "PseudoVFWCVTBF16_F_F",
                             wlist=AllWidenableIntToBFloatVectors,
                             isSEWAware=1>;
  defm : VPatConversionWF_VF<"int_riscv_vfwcvt_f_f_v_alt",
                             "PseudoVFWCVTBF16_F_F_ALT",
                             wlist=AllWidenableIntToBFloatVectors,
                             isSEWAware=1>;
  // BF16 to OFP8 conversion instructions
  defm : VPatConversionVF_WF_RM<"int_riscv_vfncvt_f_f_w", 
                                "PseudoVFNCVTBF16_F_F",
                                wlist=AllWidenableIntToBFloatVectors,
                                isSEWAware=1>;
  defm : VPatConversionVF_WF_RM<"int_riscv_vfncvt_sat_f_f_w", 
                                "PseudoVFNCVTBF16_SAT_F_F",
                                wlist=AllWidenableIntToBFloatVectors,
                                isSEWAware=1>;
  defm : VPatConversionVF_WF_RM<"int_riscv_vfncvt_f_f_w_alt", 
                                "PseudoVFNCVTBF16_F_F_ALT",
                                wlist=AllWidenableIntToBFloatVectors,
                                isSEWAware=1>;
  defm : VPatConversionVF_WF_RM<"int_riscv_vfncvt_sat_f_f_w_alt", 
                                "PseudoVFNCVTBF16_SAT_F_F_ALT",
                                 wlist=AllWidenableIntToBFloatVectors,
                                 isSEWAware=1>;
  // FP32 to OFP8 conversion instructions
  defm : VPatConversionQF_RM<"int_riscv_vfncvt_f_f_q",
                             "PseudoVFNCVT_F_F", isSEWAware=1>;
  defm : VPatConversionQF_RM<"int_riscv_vfncvt_sat_f_f_q",
                             "PseudoVFNCVT_SAT_F_F", isSEWAware=1>;
  defm : VPatConversionQF_RM<"int_riscv_vfncvt_f_f_q_alt",
                             "PseudoVFNCVT_F_F_ALT", isSEWAware=1>;
  defm : VPatConversionQF_RM<"int_riscv_vfncvt_sat_f_f_q_alt",
                             "PseudoVFNCVT_SAT_F_F_ALT", isSEWAware=1>;
}

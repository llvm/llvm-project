//===-- RISCVInstrInfoZvfbf.td - 'Zvfbf*' instructions -----*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file describes the RISC-V instructions from the standard 'Zvfbfmin'
// extension, providing vector conversion instructions for BFloat16.
// This version is still experimental as the 'Zvfbfmin' extension hasn't been
// ratified yet.
//
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// Instructions
//===----------------------------------------------------------------------===//

let Predicates = [HasStdExtZvfbfminOrZvfofp8min],
    Constraints = "@earlyclobber $vd",
    mayRaiseFPException = true in {
let RVVConstraint = WidenCvt, DestEEW = EEWSEWx2 in
defm VFWCVTBF16_F_F_V : VWCVTF_FV_VS2<"vfwcvtbf16.f.f.v", 0b010010, 0b01101>;
let Uses = [FRM, VL, VTYPE] in
defm VFNCVTBF16_F_F_W : VNCVTF_FV_VS2<"vfncvtbf16.f.f.w", 0b010010, 0b11101>;
}

let Predicates = [HasStdExtZvfbfwma],
    Constraints = "@earlyclobber $vd_wb, $vd = $vd_wb",
    RVVConstraint = WidenV, Uses = [FRM, VL, VTYPE], mayRaiseFPException = true,
    DestEEW = EEWSEWx2 in {
defm VFWMACCBF16_V : VWMAC_FV_V_F<"vfwmaccbf16", 0b111011>;
}

//===----------------------------------------------------------------------===//
// Pseudo instructions
//===----------------------------------------------------------------------===//
let Predicates = [HasStdExtZvfbfmin] in {
  defm PseudoVFWCVTBF16_F_F : VPseudoVWCVTD_V;
  defm PseudoVFNCVTBF16_F_F : VPseudoVNCVTD_W_RM;
}

let mayRaiseFPException = true, Predicates = [HasStdExtZvfbfwma] in
  defm PseudoVFWMACCBF16 : VPseudoVWMAC_VV_VF_BF_RM;

//===----------------------------------------------------------------------===//
// Patterns
//===----------------------------------------------------------------------===//
multiclass VPatConversionWF_VF_BF<string intrinsic, string instruction,
                                  bit isSEWAware = 0> {
  foreach fvtiToFWti = AllWidenableBF16ToFloatVectors in
  {
    defvar fvti = fvtiToFWti.Vti;
    defvar fwti = fvtiToFWti.Wti;
    defm : VPatConversion<intrinsic, instruction, "V",
                          fwti.Vector, fvti.Vector, fwti.Mask, fvti.Log2SEW,
                          fvti.LMul, fwti.RegClass, fvti.RegClass, isSEWAware>;
  }
}

multiclass VPatConversionVF_WF_BF_RM<string intrinsic, string instruction,
                                     bit isSEWAware = 0> {
  foreach fvtiToFWti = AllWidenableBF16ToFloatVectors in {
    defvar fvti = fvtiToFWti.Vti;
    defvar fwti = fvtiToFWti.Wti;
    defm : VPatConversionRoundingMode<intrinsic, instruction, "W",
                                      fvti.Vector, fwti.Vector, fvti.Mask, fvti.Log2SEW,
                                      fvti.LMul, fvti.RegClass, fwti.RegClass,
                                      isSEWAware>;
  }
}

let Predicates = [HasStdExtZvfbfmin] in {
  defm : VPatConversionWF_VF_BF<"int_riscv_vfwcvtbf16_f_f_v",
                                "PseudoVFWCVTBF16_F_F", isSEWAware=1>;
  defm : VPatConversionVF_WF_BF_RM<"int_riscv_vfncvtbf16_f_f_w",
                                   "PseudoVFNCVTBF16_F_F", isSEWAware=1>;

  foreach fvtiToFWti = AllWidenableBF16ToFloatVectors in {
    defvar fvti = fvtiToFWti.Vti;
    defvar fwti = fvtiToFWti.Wti;
    def : Pat<(fwti.Vector (any_riscv_fpextend_vl
                               (fvti.Vector fvti.RegClass:$rs1),
                               (fvti.Mask VMV0:$vm),
                               VLOpFrag)),
              (!cast<Instruction>("PseudoVFWCVTBF16_F_F_V_"#fvti.LMul.MX#"_E"#fvti.SEW#"_MASK")
                  (fwti.Vector (IMPLICIT_DEF)), fvti.RegClass:$rs1,
                  (fvti.Mask VMV0:$vm),
                  GPR:$vl, fvti.Log2SEW, TA_MA)>;

    def : Pat<(fvti.Vector (any_riscv_fpround_vl
                               (fwti.Vector fwti.RegClass:$rs1),
                               (fwti.Mask VMV0:$vm), VLOpFrag)),
              (!cast<Instruction>("PseudoVFNCVTBF16_F_F_W_"#fvti.LMul.MX#"_E"#fvti.SEW#"_MASK")
                  (fvti.Vector (IMPLICIT_DEF)), fwti.RegClass:$rs1,
                  (fwti.Mask VMV0:$vm),
                  // Value to indicate no rounding mode change in
                  // RISCVInsertReadWriteCSR
                  FRM_DYN,
                  GPR:$vl, fvti.Log2SEW, TA_MA)>;
    def : Pat<(fvti.Vector (fpround (fwti.Vector fwti.RegClass:$rs1))),
              (!cast<Instruction>("PseudoVFNCVTBF16_F_F_W_"#fvti.LMul.MX#"_E"#fvti.SEW)
                  (fvti.Vector (IMPLICIT_DEF)),
                  fwti.RegClass:$rs1,
                  // Value to indicate no rounding mode change in
                  // RISCVInsertReadWriteCSR
                  FRM_DYN,
                  fvti.AVL, fvti.Log2SEW, TA_MA)>;
  }
}

let Predicates = [HasStdExtZvfbfwma] in {
  defm : VPatTernaryW_VV_VX_RM<"int_riscv_vfwmaccbf16", "PseudoVFWMACCBF16",
                               AllWidenableBF16ToFloatVectors, isSEWAware=1>;
  defm : VPatWidenFPMulAccVL_VV_VF_RM<riscv_vfwmadd_vl, "PseudoVFWMACCBF16",
                                      AllWidenableBF16ToFloatVectors>;
  defm : VPatWidenFPMulAccSDNode_VV_VF_RM<"PseudoVFWMACCBF16",
                                          AllWidenableBF16ToFloatVectors>;
}

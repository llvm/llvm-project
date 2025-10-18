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

defset list<VTypeInfoToWide> AllWidenableIntToBF16Vectors = {
  def : VTypeInfoToWide<VI8MF8, VBF16MF4>;
  def : VTypeInfoToWide<VI8MF4, VBF16MF2>;
  def : VTypeInfoToWide<VI8MF2, VBF16M1>;
  def : VTypeInfoToWide<VI8M1, VBF16M2>;
  def : VTypeInfoToWide<VI8M2, VBF16M4>;
  def : VTypeInfoToWide<VI8M4, VBF16M8>;
}

multiclass VPseudoVALU_VV_VF_RM_BF16 {
  foreach m = MxListF in {
    defm "" : VPseudoBinaryFV_VV_RM<m, 16/*sew*/>,
              SchedBinary<"WriteVFALUV", "ReadVFALUV", "ReadVFALUV", m.MX,
                          16/*sew*/, forcePassthruRead=true>;
  }

  defvar f = SCALAR_F16;
  foreach m = f.MxList in {
    defm "" : VPseudoBinaryV_VF_RM<m, f, f.SEW>,
              SchedBinary<"WriteVFALUF", "ReadVFALUV", "ReadVFALUF", m.MX,
                          f.SEW, forcePassthruRead=true>;
  }
}

multiclass VPseudoVALU_VF_RM_BF16 {
  defvar f = SCALAR_F16;
  foreach m = f.MxList in {
    defm "" : VPseudoBinaryV_VF_RM<m, f, f.SEW>,
              SchedBinary<"WriteVFALUF", "ReadVFALUV", "ReadVFALUF", m.MX,
                          f.SEW, forcePassthruRead=true>;
  }
}

multiclass VPseudoVFWALU_VV_VF_RM_BF16 {
  foreach m = MxListFW in {
    defm "" : VPseudoBinaryW_VV_RM<m, sew=16>,
              SchedBinary<"WriteVFWALUV", "ReadVFWALUV", "ReadVFWALUV", m.MX,
                          16/*sew*/, forcePassthruRead=true>;
  }

  defvar f = SCALAR_F16;
  foreach m = f.MxListFW in {
    defm "" : VPseudoBinaryW_VF_RM<m, f, sew=f.SEW>,
              SchedBinary<"WriteVFWALUF", "ReadVFWALUV", "ReadVFWALUF", m.MX,
                        f.SEW, forcePassthruRead=true>;
  }
}

multiclass VPseudoVFWALU_WV_WF_RM_BF16 {
  foreach m = MxListFW in {
    defm "" : VPseudoBinaryW_WV_RM<m, sew=16>,
              SchedBinary<"WriteVFWALUV", "ReadVFWALUV", "ReadVFWALUV", m.MX,
                          16/*sew*/, forcePassthruRead=true>;
  }
  defvar f = SCALAR_F16;
  foreach m = f.MxListFW in {
    defm "" : VPseudoBinaryW_WF_RM<m, f, sew=f.SEW>,
              SchedBinary<"WriteVFWALUF", "ReadVFWALUV", "ReadVFWALUF", m.MX,
                          f.SEW, forcePassthruRead=true>;
  }
}

multiclass VPseudoVFMUL_VV_VF_RM_BF16 {
  foreach m = MxListF in {
    defm "" : VPseudoBinaryFV_VV_RM<m, 16/*sew*/>,
              SchedBinary<"WriteVFMulV", "ReadVFMulV", "ReadVFMulV", m.MX,
                          16/*sew*/, forcePassthruRead=true>;
  }

  defvar f = SCALAR_F16;
  foreach m = f.MxList in {
    defm "" : VPseudoBinaryV_VF_RM<m, f, f.SEW>,
              SchedBinary<"WriteVFMulF", "ReadVFMulV", "ReadVFMulF", m.MX,
                          f.SEW, forcePassthruRead=true>;
  }
}

multiclass VPseudoVWMUL_VV_VF_RM_BF16 {
  foreach m = MxListFW in {
    defm "" : VPseudoBinaryW_VV_RM<m, sew=16>,
              SchedBinary<"WriteVFWMulV", "ReadVFWMulV", "ReadVFWMulV", m.MX,
                          16/*sew*/, forcePassthruRead=true>;
  }

  defvar f = SCALAR_F16;
  foreach m = f.MxListFW in {
    defm "" : VPseudoBinaryW_VF_RM<m, f, sew=f.SEW>,
              SchedBinary<"WriteVFWMulF", "ReadVFWMulV", "ReadVFWMulF", m.MX,
                          f.SEW, forcePassthruRead=true>;
  }
}

multiclass VPseudoVMAC_VV_VF_AAXA_RM_BF16 {
  foreach m = MxListF in {
    defm "" : VPseudoTernaryV_VV_AAXA_RM<m, 16/*sew*/>,
              SchedTernary<"WriteVFMulAddV", "ReadVFMulAddV", "ReadVFMulAddV", 
                           "ReadVFMulAddV", m.MX, 16/*sew*/>;
  }

  defvar f = SCALAR_F16;
  foreach m = f.MxList in {
    defm "" : VPseudoTernaryV_VF_AAXA_RM<m, f, f.SEW>,
              SchedTernary<"WriteVFMulAddF", "ReadVFMulAddV", "ReadVFMulAddF", 
                           "ReadVFMulAddV", m.MX, f.SEW>;
  }
}

multiclass VPseudoVWMAC_VV_VF_RM_BF16 {
  foreach m = MxListFW in {
    defm "" : VPseudoTernaryW_VV_RM<m, sew=16>,
              SchedTernary<"WriteVFWMulAddV", "ReadVFWMulAddV",
                           "ReadVFWMulAddV", "ReadVFWMulAddV", m.MX, 16/*sew*/>;
  }

  defvar f = SCALAR_F16;
  foreach m = f.MxListFW in {
    defm "" : VPseudoTernaryW_VF_RM<m, f, sew=f.SEW>,
              SchedTernary<"WriteVFWMulAddF", "ReadVFWMulAddV",
                           "ReadVFWMulAddF", "ReadVFWMulAddV", m.MX, f.SEW>;
  }
}

multiclass VPseudoVRCP_V_BF16 {
  foreach m = MxListF in {
    defvar mx = m.MX;
    let VLMul = m.value in {
      def "_V_" # mx # "_E16"
          : VPseudoUnaryNoMask<m.vrclass, m.vrclass>,
            SchedUnary<"WriteVFRecpV", "ReadVFRecpV", mx, 16/*sew*/,
                       forcePassthruRead=true>;
      def "_V_" # mx # "_E16_MASK"
          : VPseudoUnaryMask<m.vrclass, m.vrclass>,
            RISCVMaskedPseudo<MaskIdx = 2>,
            SchedUnary<"WriteVFRecpV", "ReadVFRecpV", mx, 16/*sew*/,
                       forcePassthruRead=true>;
    }
  }
}

multiclass VPseudoVRCP_V_RM_BF16 {
  foreach m = MxListF in {
    defvar mx = m.MX;
    let VLMul = m.value in {
      def "_V_" # mx # "_E16"
          : VPseudoUnaryNoMaskRoundingMode<m.vrclass, m.vrclass>,
            SchedUnary<"WriteVFRecpV", "ReadVFRecpV", mx, 16/*sew*/,
                       forcePassthruRead=true>;
      def "_V_" # mx # "_E16_MASK"
          : VPseudoUnaryMaskRoundingMode<m.vrclass, m.vrclass>,
            RISCVMaskedPseudo<MaskIdx = 2>,
            SchedUnary<"WriteVFRecpV", "ReadVFRecpV", mx, 16/*sew*/,
                       forcePassthruRead=true>;
    }
  }
}

multiclass VPseudoVMAX_VV_VF_BF16 {
  foreach m = MxListF in {
    defm "" : VPseudoBinaryV_VV<m, sew=16>,
              SchedBinary<"WriteVFMinMaxV", "ReadVFMinMaxV", "ReadVFMinMaxV", 
                          m.MX, 16/*sew*/, forcePassthruRead=true>;
  }

  defvar f = SCALAR_F16;
  foreach m = f.MxList in {
    defm "" : VPseudoBinaryV_VF<m, f, f.SEW>,
              SchedBinary<"WriteVFMinMaxF", "ReadVFMinMaxV", "ReadVFMinMaxF", 
                          m.MX, f.SEW, forcePassthruRead=true>;
  }
}

multiclass VPseudoVSGNJ_VV_VF_BF16 {
  foreach m = MxListF in {
    defm "" : VPseudoBinaryV_VV<m, sew=16>,
              SchedBinary<"WriteVFSgnjV", "ReadVFSgnjV", "ReadVFSgnjV", m.MX,
                          16/*sew*/, forcePassthruRead=true>;
  }

  defvar f = SCALAR_F16;
  foreach m = f.MxList in {
    defm "" : VPseudoBinaryV_VF<m, f, f.SEW>,
              SchedBinary<"WriteVFSgnjF", "ReadVFSgnjV", "ReadVFSgnjF", m.MX,
                          f.SEW, forcePassthruRead=true>;
  }
}

multiclass VPseudoVWCVTF_V_BF16 {
  defvar constraint = "@earlyclobber $rd";
  foreach m = MxListW in
    defm _V : VPseudoConversion<m.wvrclass, m.vrclass, m, constraint, sew=8,
                                TargetConstraintType=3>,
              SchedUnary<"WriteVFWCvtIToFV", "ReadVFWCvtIToFV", m.MX, 8/*sew*/,
                         forcePassthruRead=true>;
}

multiclass VPseudoVWCVTD_V_BF16 {
  defvar constraint = "@earlyclobber $rd";
  foreach m = MxListFW in
    defm _V : VPseudoConversion<m.wvrclass, m.vrclass, m, constraint, sew=16,
                                TargetConstraintType=3>,
              SchedUnary<"WriteVFWCvtFToFV", "ReadVFWCvtFToFV", m.MX, 16/*sew*/,
                         forcePassthruRead=true>;
}

multiclass VPseudoVNCVTD_W_BF16 {
  defvar constraint = "@earlyclobber $rd";
  foreach m = MxListFW in
    defm _W : VPseudoConversion<m.vrclass, m.wvrclass, m, constraint, sew=16,
                                TargetConstraintType=2>,
              SchedUnary<"WriteVFNCvtFToFV", "ReadVFNCvtFToFV", m.MX, 16/*sew*/,
                         forcePassthruRead=true>;
}

multiclass VPseudoVNCVTD_W_RM_BF16 {
  defvar constraint = "@earlyclobber $rd";
  foreach m = MxListFW in
    defm _W : VPseudoConversionRoundingMode<m.vrclass, m.wvrclass, m,
                                            constraint, sew=16,
                                            TargetConstraintType=2>,
              SchedUnary<"WriteVFNCvtFToFV", "ReadVFNCvtFToFV", m.MX, 16/*sew*/,
                         forcePassthruRead=true>;
}

let Predicates = [HasStdExtZvfbfa], AltFmtType = IS_ALTFMT in {
let mayRaiseFPException = true in {
defm PseudoVFADD_ALT : VPseudoVALU_VV_VF_RM_BF16;
defm PseudoVFSUB_ALT  : VPseudoVALU_VV_VF_RM_BF16;
defm PseudoVFRSUB_ALT : VPseudoVALU_VF_RM_BF16;
}

let mayRaiseFPException = true in {
defm PseudoVFWADD_ALT : VPseudoVFWALU_VV_VF_RM_BF16;
defm PseudoVFWSUB_ALT : VPseudoVFWALU_VV_VF_RM_BF16;
defm PseudoVFWADD_ALT : VPseudoVFWALU_WV_WF_RM_BF16;
defm PseudoVFWSUB_ALT : VPseudoVFWALU_WV_WF_RM_BF16;
}

let mayRaiseFPException = true in
defm PseudoVFMUL_ALT : VPseudoVFMUL_VV_VF_RM_BF16;

let mayRaiseFPException = true in
defm PseudoVFWMUL_ALT : VPseudoVWMUL_VV_VF_RM_BF16;

let mayRaiseFPException = true in {
defm PseudoVFMACC_ALT  : VPseudoVMAC_VV_VF_AAXA_RM_BF16;
defm PseudoVFNMACC_ALT : VPseudoVMAC_VV_VF_AAXA_RM_BF16;
defm PseudoVFMSAC_ALT  : VPseudoVMAC_VV_VF_AAXA_RM_BF16;
defm PseudoVFNMSAC_ALT : VPseudoVMAC_VV_VF_AAXA_RM_BF16;
defm PseudoVFMADD_ALT  : VPseudoVMAC_VV_VF_AAXA_RM_BF16;
defm PseudoVFNMADD_ALT : VPseudoVMAC_VV_VF_AAXA_RM_BF16;
defm PseudoVFMSUB_ALT  : VPseudoVMAC_VV_VF_AAXA_RM_BF16;
defm PseudoVFNMSUB_ALT : VPseudoVMAC_VV_VF_AAXA_RM_BF16;
}

let mayRaiseFPException = true in {
defm PseudoVFWMACC_ALT  : VPseudoVWMAC_VV_VF_RM_BF16;
defm PseudoVFWNMACC_ALT : VPseudoVWMAC_VV_VF_RM_BF16;
defm PseudoVFWMSAC_ALT  : VPseudoVWMAC_VV_VF_RM_BF16;
defm PseudoVFWNMSAC_ALT : VPseudoVWMAC_VV_VF_RM_BF16;
}

let mayRaiseFPException = true in
defm PseudoVFRSQRT7_ALT : VPseudoVRCP_V_BF16;

let mayRaiseFPException = true in
defm PseudoVFREC7_ALT : VPseudoVRCP_V_RM_BF16;

let mayRaiseFPException = true in {
defm PseudoVFMIN_ALT : VPseudoVMAX_VV_VF_BF16;
defm PseudoVFMAX_ALT : VPseudoVMAX_VV_VF_BF16;
}

defm PseudoVFSGNJ_ALT  : VPseudoVSGNJ_VV_VF_BF16;
defm PseudoVFSGNJN_ALT : VPseudoVSGNJ_VV_VF_BF16;
defm PseudoVFSGNJX_ALT : VPseudoVSGNJ_VV_VF_BF16;

let mayRaiseFPException = true in {
defm PseudoVMFEQ_ALT : VPseudoVCMPM_VV_VF;
defm PseudoVMFNE_ALT : VPseudoVCMPM_VV_VF;
defm PseudoVMFLT_ALT : VPseudoVCMPM_VV_VF;
defm PseudoVMFLE_ALT : VPseudoVCMPM_VV_VF;
defm PseudoVMFGT_ALT : VPseudoVCMPM_VF;
defm PseudoVMFGE_ALT : VPseudoVCMPM_VF;
}

defm PseudoVFCLASS_ALT : VPseudoVCLS_V;

defm PseudoVFMERGE_ALT : VPseudoVMRG_FM;

defm PseudoVFMV_V_ALT : VPseudoVMV_F;

let mayRaiseFPException = true in {
defm PseudoVFWCVT_F_XU_ALT : VPseudoVWCVTF_V_BF16;
defm PseudoVFWCVT_F_X_ALT  : VPseudoVWCVTF_V_BF16;

defm PseudoVFWCVT_F_F_ALT  : VPseudoVWCVTD_V_BF16;
} // mayRaiseFPException = true

let mayRaiseFPException = true in {
let hasSideEffects = 0, hasPostISelHook = 1 in {
defm PseudoVFNCVT_XU_F_ALT : VPseudoVNCVTI_W_RM;
defm PseudoVFNCVT_X_F_ALT  : VPseudoVNCVTI_W_RM;
}

defm PseudoVFNCVT_RTZ_XU_F_ALT : VPseudoVNCVTI_W;
defm PseudoVFNCVT_RTZ_X_F_ALT  : VPseudoVNCVTI_W;

defm PseudoVFNCVT_F_F_ALT  : VPseudoVNCVTD_W_RM_BF16;

defm PseudoVFNCVT_ROD_F_F_ALT : VPseudoVNCVTD_W_BF16;
} // mayRaiseFPException = true

let mayLoad = 0, mayStore = 0, hasSideEffects = 0 in {
  defvar f = SCALAR_F16;
  let HasSEWOp = 1, BaseInstr = VFMV_F_S in
  def "PseudoVFMV_" # f.FX # "_S_ALT" :
    RISCVVPseudo<(outs f.fprclass:$rd), (ins VR:$rs2, sew:$sew)>,
    Sched<[WriteVMovFS, ReadVMovFS]>;
  let HasVLOp = 1, HasSEWOp = 1, BaseInstr = VFMV_S_F, isReMaterializable = 1,
      Constraints = "$rd = $passthru" in
  def "PseudoVFMV_S_" # f.FX # "_ALT" :
    RISCVVPseudo<(outs VR:$rd),
                 (ins VR:$passthru, f.fprclass:$rs1, AVL:$vl, sew:$sew)>,
    Sched<[WriteVMovSF, ReadVMovSF_V, ReadVMovSF_F]>;
}

defm PseudoVFSLIDE1UP_ALT   : VPseudoVSLD1_VF<"@earlyclobber $rd">;
defm PseudoVFSLIDE1DOWN_ALT : VPseudoVSLD1_VF;
} // Predicates = [HasStdExtZvfbfa], AltFmtType = IS_ALTFMT

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

  defm : VPatUnaryV_V_AnyMask<"int_riscv_vcompress", "PseudoVCOMPRESS", AllBF16Vectors>;
  defm : VPatBinaryV_VV_VX_VI_INT<"int_riscv_vrgather", "PseudoVRGATHER",
                                  AllBF16Vectors, uimm5>;
  defm : VPatBinaryV_VV_INT_EEW<"int_riscv_vrgatherei16_vv", "PseudoVRGATHEREI16",
                                eew=16, vtilist=AllBF16Vectors>;
  defm : VPatTernaryV_VX_VI<"int_riscv_vslideup", "PseudoVSLIDEUP", AllBF16Vectors, uimm5>;
  defm : VPatTernaryV_VX_VI<"int_riscv_vslidedown", "PseudoVSLIDEDOWN", AllBF16Vectors, uimm5>;

  foreach fvti = AllBF16Vectors in {
    defm : VPatBinaryCarryInTAIL<"int_riscv_vmerge", "PseudoVMERGE", "VVM",
                                 fvti.Vector,
                                 fvti.Vector, fvti.Vector, fvti.Mask,
                                 fvti.Log2SEW, fvti.LMul, fvti.RegClass,
                                 fvti.RegClass, fvti.RegClass>;
    defm : VPatBinaryCarryInTAIL<"int_riscv_vfmerge", "PseudoVFMERGE",
                                 "V"#fvti.ScalarSuffix#"M",
                                 fvti.Vector,
                                 fvti.Vector, fvti.Scalar, fvti.Mask,
                                 fvti.Log2SEW, fvti.LMul, fvti.RegClass,
                                 fvti.RegClass, fvti.ScalarRegClass>;
    defvar instr = !cast<Instruction>("PseudoVMERGE_VIM_"#fvti.LMul.MX);
    def : Pat<(fvti.Vector (int_riscv_vfmerge (fvti.Vector fvti.RegClass:$passthru),
                                              (fvti.Vector fvti.RegClass:$rs2),
                                              (fvti.Scalar (fpimm0)),
                                              (fvti.Mask VMV0:$vm), VLOpFrag)),
              (instr fvti.RegClass:$passthru, fvti.RegClass:$rs2, 0,
                     (fvti.Mask VMV0:$vm), GPR:$vl, fvti.Log2SEW)>;

    defvar ivti = GetIntVTypeInfo<fvti>.Vti;
    def : Pat<(fvti.Vector (vselect (fvti.Mask VMV0:$vm), fvti.RegClass:$rs1,
                                                          fvti.RegClass:$rs2)),
              (!cast<Instruction>("PseudoVMERGE_VVM_"#fvti.LMul.MX)
                   (fvti.Vector (IMPLICIT_DEF)),
                   fvti.RegClass:$rs2, fvti.RegClass:$rs1, (fvti.Mask VMV0:$vm),
                   fvti.AVL, fvti.Log2SEW)>;

    def : Pat<(fvti.Vector (vselect (fvti.Mask VMV0:$vm),
                                    (SplatFPOp (SelectScalarFPAsInt (XLenVT GPR:$imm))),
                                    fvti.RegClass:$rs2)),
              (!cast<Instruction>("PseudoVMERGE_VXM_"#fvti.LMul.MX)
                   (fvti.Vector (IMPLICIT_DEF)),
                   fvti.RegClass:$rs2, GPR:$imm, (fvti.Mask VMV0:$vm), fvti.AVL, fvti.Log2SEW)>;

    def : Pat<(fvti.Vector (vselect (fvti.Mask VMV0:$vm),
                                    (SplatFPOp (fvti.Scalar fpimm0)),
                                    fvti.RegClass:$rs2)),
              (!cast<Instruction>("PseudoVMERGE_VIM_"#fvti.LMul.MX)
                   (fvti.Vector (IMPLICIT_DEF)),
                   fvti.RegClass:$rs2, 0, (fvti.Mask VMV0:$vm), fvti.AVL, fvti.Log2SEW)>;

    def : Pat<(fvti.Vector (vselect (fvti.Mask VMV0:$vm),
                                    (SplatFPOp fvti.ScalarRegClass:$rs1),
                                    fvti.RegClass:$rs2)),
              (!cast<Instruction>("PseudoVFMERGE_V"#fvti.ScalarSuffix#"M_"#fvti.LMul.MX)
                   (fvti.Vector (IMPLICIT_DEF)),
                   fvti.RegClass:$rs2,
                   (fvti.Scalar fvti.ScalarRegClass:$rs1),
                   (fvti.Mask VMV0:$vm), fvti.AVL, fvti.Log2SEW)>;

    def : Pat<(fvti.Vector (riscv_vmerge_vl (fvti.Mask VMV0:$vm),
                                            fvti.RegClass:$rs1,
                                            fvti.RegClass:$rs2,
                                            fvti.RegClass:$passthru,
                                            VLOpFrag)),
              (!cast<Instruction>("PseudoVMERGE_VVM_"#fvti.LMul.MX)
                   fvti.RegClass:$passthru, fvti.RegClass:$rs2, fvti.RegClass:$rs1, (fvti.Mask VMV0:$vm),
                   GPR:$vl, fvti.Log2SEW)>;

    def : Pat<(fvti.Vector (riscv_vmerge_vl (fvti.Mask VMV0:$vm),
                                            (SplatFPOp (SelectScalarFPAsInt (XLenVT GPR:$imm))),
                                            fvti.RegClass:$rs2,
                                            fvti.RegClass:$passthru,
                                            VLOpFrag)),
              (!cast<Instruction>("PseudoVMERGE_VXM_"#fvti.LMul.MX)
                   fvti.RegClass:$passthru, fvti.RegClass:$rs2, GPR:$imm, (fvti.Mask VMV0:$vm),
                   GPR:$vl, fvti.Log2SEW)>;


    def : Pat<(fvti.Vector (riscv_vmerge_vl (fvti.Mask VMV0:$vm),
                                            (SplatFPOp (fvti.Scalar fpimm0)),
                                            fvti.RegClass:$rs2,
                                            fvti.RegClass:$passthru,
                                            VLOpFrag)),
              (!cast<Instruction>("PseudoVMERGE_VIM_"#fvti.LMul.MX)
                   fvti.RegClass:$passthru, fvti.RegClass:$rs2, 0, (fvti.Mask VMV0:$vm),
                   GPR:$vl, fvti.Log2SEW)>;

    def : Pat<(fvti.Vector (riscv_vmerge_vl (fvti.Mask VMV0:$vm),
                                            (SplatFPOp fvti.ScalarRegClass:$rs1),
                                            fvti.RegClass:$rs2,
                                            fvti.RegClass:$passthru,
                                            VLOpFrag)),
              (!cast<Instruction>("PseudoVFMERGE_V"#fvti.ScalarSuffix#"M_"#fvti.LMul.MX)
                   fvti.RegClass:$passthru, fvti.RegClass:$rs2,
                   (fvti.Scalar fvti.ScalarRegClass:$rs1),
                   (fvti.Mask VMV0:$vm), GPR:$vl, fvti.Log2SEW)>;

    def : Pat<(fvti.Vector
               (riscv_vrgather_vv_vl fvti.RegClass:$rs2,
                                     (ivti.Vector fvti.RegClass:$rs1),
                                     fvti.RegClass:$passthru,
                                     (fvti.Mask VMV0:$vm),
                                     VLOpFrag)),
              (!cast<Instruction>("PseudoVRGATHER_VV_"# fvti.LMul.MX#"_E"# fvti.SEW#"_MASK")
                   fvti.RegClass:$passthru, fvti.RegClass:$rs2, fvti.RegClass:$rs1,
                   (fvti.Mask VMV0:$vm), GPR:$vl, fvti.Log2SEW, TAIL_AGNOSTIC)>;
    def : Pat<(fvti.Vector (riscv_vrgather_vx_vl fvti.RegClass:$rs2, GPR:$rs1,
                                                fvti.RegClass:$passthru,
                                                (fvti.Mask VMV0:$vm),
                                                VLOpFrag)),
              (!cast<Instruction>("PseudoVRGATHER_VX_"# fvti.LMul.MX#"_MASK")
                   fvti.RegClass:$passthru, fvti.RegClass:$rs2, GPR:$rs1,
                   (fvti.Mask VMV0:$vm), GPR:$vl, fvti.Log2SEW, TAIL_AGNOSTIC)>;
    def : Pat<(fvti.Vector
               (riscv_vrgather_vx_vl fvti.RegClass:$rs2,
                                     uimm5:$imm,
                                     fvti.RegClass:$passthru,
                                     (fvti.Mask VMV0:$vm),
                                     VLOpFrag)),
              (!cast<Instruction>("PseudoVRGATHER_VI_"# fvti.LMul.MX#"_MASK")
                   fvti.RegClass:$passthru, fvti.RegClass:$rs2, uimm5:$imm,
                   (fvti.Mask VMV0:$vm), GPR:$vl, fvti.Log2SEW, TAIL_AGNOSTIC)>;
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

multiclass VPatConversionVI_VF_BF16<string intrinsic, string instruction> {
  foreach fvti = AllBF16Vectors in {
    defvar ivti = GetIntVTypeInfo<fvti>.Vti;
    let Predicates = !listconcat(GetVTypePredicates<fvti>.Predicates,
                                 GetVTypePredicates<ivti>.Predicates) in
    defm : VPatConversion<intrinsic, instruction, "V",
                          ivti.Vector, fvti.Vector, ivti.Mask, fvti.Log2SEW,
                          fvti.LMul, ivti.RegClass, fvti.RegClass>;
  }
}

multiclass VPatConversionWF_VI_BF16<string intrinsic, string instruction,
                                    bit isSEWAware = 0> {
  foreach vtiToWti = AllWidenableIntToBF16Vectors in {
    defvar vti = vtiToWti.Vti;
    defvar fwti = vtiToWti.Wti;
    let Predicates = !listconcat(GetVTypePredicates<vti>.Predicates,
                                 GetVTypePredicates<fwti>.Predicates) in
    defm : VPatConversion<intrinsic, instruction, "V",
                          fwti.Vector, vti.Vector, fwti.Mask, vti.Log2SEW,
                          vti.LMul, fwti.RegClass, vti.RegClass, isSEWAware>;
  }
}

multiclass VPatConversionWF_VF_BF16<string intrinsic, string instruction,
                                    bit isSEWAware = 0> {
  foreach fvtiToFWti = AllWidenableBF16ToFloatVectors in {
    defvar fvti = fvtiToFWti.Vti;
    defvar fwti = fvtiToFWti.Wti;
    let Predicates = !listconcat(GetVTypeMinimalPredicates<fvti>.Predicates,
                                 GetVTypeMinimalPredicates<fwti>.Predicates) in
    defm : VPatConversion<intrinsic, instruction, "V",
                          fwti.Vector, fvti.Vector, fwti.Mask, fvti.Log2SEW,
                          fvti.LMul, fwti.RegClass, fvti.RegClass, isSEWAware>;
  }
}

multiclass VPatConversionVI_WF_BF16<string intrinsic, string instruction> {
  foreach vtiToWti = AllWidenableIntToBF16Vectors in {
    defvar vti = vtiToWti.Vti;
    defvar fwti = vtiToWti.Wti;
    let Predicates = !listconcat(GetVTypePredicates<vti>.Predicates,
                                 GetVTypePredicates<fwti>.Predicates) in
    defm : VPatConversion<intrinsic, instruction, "W",
                          vti.Vector, fwti.Vector, vti.Mask, vti.Log2SEW,
                          vti.LMul, vti.RegClass, fwti.RegClass>;
  }
}

multiclass VPatConversionVI_WF_RM_BF16<string intrinsic, string instruction> {
  foreach vtiToWti = AllWidenableIntToBF16Vectors in {
    defvar vti = vtiToWti.Vti;
    defvar fwti = vtiToWti.Wti;
    let Predicates = !listconcat(GetVTypePredicates<vti>.Predicates,
                                 GetVTypePredicates<fwti>.Predicates) in
    defm : VPatConversionRoundingMode<intrinsic, instruction, "W",
                                      vti.Vector, fwti.Vector, vti.Mask, vti.Log2SEW,
                                      vti.LMul, vti.RegClass, fwti.RegClass>;
  }
}

multiclass VPatConversionVF_WF_BF16<string intrinsic, string instruction,
                                    bit isSEWAware = 0> {
  foreach fvtiToFWti = AllWidenableBF16ToFloatVectors in {
    defvar fvti = fvtiToFWti.Vti;
    defvar fwti = fvtiToFWti.Wti;
    let Predicates = !listconcat(GetVTypePredicates<fvti>.Predicates,
                                 GetVTypePredicates<fwti>.Predicates) in
    defm : VPatConversion<intrinsic, instruction, "W",
                          fvti.Vector, fwti.Vector, fvti.Mask, fvti.Log2SEW,
                          fvti.LMul, fvti.RegClass, fwti.RegClass, isSEWAware>;
  }
}

let Predicates = [HasStdExtZvfbfa] in {
defm : VPatBinaryV_VV_VX_RM<"int_riscv_vfadd", "PseudoVFADD_ALT",
                            AllBF16Vectors, isSEWAware = 1>;
defm : VPatBinaryV_VV_VX_RM<"int_riscv_vfsub", "PseudoVFSUB_ALT",
                            AllBF16Vectors, isSEWAware = 1>;
defm : VPatBinaryV_VX_RM<"int_riscv_vfrsub", "PseudoVFRSUB_ALT",
                         AllBF16Vectors, isSEWAware = 1>;
defm : VPatBinaryW_VV_VX_RM<"int_riscv_vfwadd", "PseudoVFWADD_ALT",
                            AllWidenableBF16ToFloatVectors, isSEWAware=1>;
defm : VPatBinaryW_VV_VX_RM<"int_riscv_vfwsub", "PseudoVFWSUB_ALT",
                            AllWidenableBF16ToFloatVectors, isSEWAware=1>;
defm : VPatBinaryW_WV_WX_RM<"int_riscv_vfwadd_w", "PseudoVFWADD_ALT",
                            AllWidenableBF16ToFloatVectors, isSEWAware=1>;
defm : VPatBinaryW_WV_WX_RM<"int_riscv_vfwsub_w", "PseudoVFWSUB_ALT",
                            AllWidenableBF16ToFloatVectors, isSEWAware=1>;
defm : VPatBinaryV_VV_VX_RM<"int_riscv_vfmul", "PseudoVFMUL_ALT",
                            AllBF16Vectors, isSEWAware=1>;
defm : VPatBinaryW_VV_VX_RM<"int_riscv_vfwmul", "PseudoVFWMUL_ALT",
                            AllWidenableBF16ToFloatVectors, isSEWAware=1>;
defm : VPatTernaryV_VV_VX_AAXA_RM<"int_riscv_vfmacc", "PseudoVFMACC_ALT",
                                  AllBF16Vectors, isSEWAware=1>;
defm : VPatTernaryV_VV_VX_AAXA_RM<"int_riscv_vfnmacc", "PseudoVFNMACC_ALT",
                                  AllBF16Vectors, isSEWAware=1>;
defm : VPatTernaryV_VV_VX_AAXA_RM<"int_riscv_vfmsac", "PseudoVFMSAC_ALT",
                                  AllBF16Vectors, isSEWAware=1>;
defm : VPatTernaryV_VV_VX_AAXA_RM<"int_riscv_vfnmsac", "PseudoVFNMSAC_ALT",
                                  AllBF16Vectors, isSEWAware=1>;
defm : VPatTernaryV_VV_VX_AAXA_RM<"int_riscv_vfmadd", "PseudoVFMADD_ALT",
                                  AllBF16Vectors, isSEWAware=1>;
defm : VPatTernaryV_VV_VX_AAXA_RM<"int_riscv_vfnmadd", "PseudoVFNMADD_ALT",
                                  AllBF16Vectors, isSEWAware=1>;
defm : VPatTernaryV_VV_VX_AAXA_RM<"int_riscv_vfmsub", "PseudoVFMSUB_ALT",
                                  AllBF16Vectors, isSEWAware=1>;
defm : VPatTernaryV_VV_VX_AAXA_RM<"int_riscv_vfnmsub", "PseudoVFNMSUB_ALT",
                                  AllBF16Vectors, isSEWAware=1>;
defm : VPatTernaryW_VV_VX_RM<"int_riscv_vfwmacc", "PseudoVFWMACC_ALT",
                             AllWidenableBF16ToFloatVectors, isSEWAware=1>;
defm : VPatTernaryW_VV_VX_RM<"int_riscv_vfwnmacc", "PseudoVFWNMACC_ALT",
                             AllWidenableBF16ToFloatVectors, isSEWAware=1>;
defm : VPatTernaryW_VV_VX_RM<"int_riscv_vfwmsac", "PseudoVFWMSAC_ALT",
                             AllWidenableBF16ToFloatVectors, isSEWAware=1>;
defm : VPatTernaryW_VV_VX_RM<"int_riscv_vfwnmsac", "PseudoVFWNMSAC_ALT",
                             AllWidenableBF16ToFloatVectors, isSEWAware=1>;
defm : VPatUnaryV_V<"int_riscv_vfrsqrt7", "PseudoVFRSQRT7_ALT",
                    AllBF16Vectors, isSEWAware=1>;
defm : VPatUnaryV_V_RM<"int_riscv_vfrec7", "PseudoVFREC7_ALT",
                       AllBF16Vectors, isSEWAware=1>;
defm : VPatBinaryV_VV_VX<"int_riscv_vfmin", "PseudoVFMIN_ALT",
                         AllBF16Vectors, isSEWAware=1>;
defm : VPatBinaryV_VV_VX<"int_riscv_vfmax", "PseudoVFMAX_ALT",
                         AllBF16Vectors, isSEWAware=1>;
defm : VPatBinaryV_VV_VX<"int_riscv_vfsgnj", "PseudoVFSGNJ_ALT",
                         AllBF16Vectors, isSEWAware=1>;
defm : VPatBinaryV_VV_VX<"int_riscv_vfsgnjn", "PseudoVFSGNJN_ALT",
                         AllBF16Vectors, isSEWAware=1>;
defm : VPatBinaryV_VV_VX<"int_riscv_vfsgnjx", "PseudoVFSGNJX_ALT",
                         AllBF16Vectors, isSEWAware=1>;
defm : VPatBinaryM_VV_VX<"int_riscv_vmfeq", "PseudoVMFEQ_ALT", AllBF16Vectors>;
defm : VPatBinaryM_VV_VX<"int_riscv_vmfle", "PseudoVMFLE_ALT", AllBF16Vectors>;
defm : VPatBinaryM_VV_VX<"int_riscv_vmflt", "PseudoVMFLT_ALT", AllBF16Vectors>;
defm : VPatBinaryM_VV_VX<"int_riscv_vmfne", "PseudoVMFNE_ALT", AllBF16Vectors>;
defm : VPatBinaryM_VX<"int_riscv_vmfgt", "PseudoVMFGT_ALT", AllBF16Vectors>;
defm : VPatBinaryM_VX<"int_riscv_vmfge", "PseudoVMFGE_ALT", AllBF16Vectors>;
defm : VPatBinarySwappedM_VV<"int_riscv_vmfgt", "PseudoVMFLT_ALT", AllBF16Vectors>;
defm : VPatBinarySwappedM_VV<"int_riscv_vmfge", "PseudoVMFLE_ALT", AllBF16Vectors>;
defm : VPatConversionVI_VF_BF16<"int_riscv_vfclass", "PseudoVFCLASS_ALT">;
foreach vti = AllBF16Vectors in {
  let Predicates = GetVTypePredicates<vti>.Predicates in
    defm : VPatBinaryCarryInTAIL<"int_riscv_vfmerge", "PseudoVFMERGE_ALT",
                                 "V"#vti.ScalarSuffix#"M",
                                 vti.Vector,
                                 vti.Vector, vti.Scalar, vti.Mask,
                                 vti.Log2SEW, vti.LMul, vti.RegClass,
                                 vti.RegClass, vti.ScalarRegClass>;
}
defm : VPatConversionWF_VI_BF16<"int_riscv_vfwcvt_f_xu_v", "PseudoVFWCVT_F_XU_ALT",
                                isSEWAware=1>;
defm : VPatConversionWF_VI_BF16<"int_riscv_vfwcvt_f_x_v", "PseudoVFWCVT_F_X_ALT",
                                isSEWAware=1>;
defm : VPatConversionWF_VF_BF16<"int_riscv_vfwcvt_f_f_v", "PseudoVFWCVT_F_F_ALT",
                                isSEWAware=1>;
defm : VPatConversionVI_WF_RM_BF16<"int_riscv_vfncvt_xu_f_w", "PseudoVFNCVT_XU_F_ALT">;
defm : VPatConversionVI_WF_RM_BF16<"int_riscv_vfncvt_x_f_w", "PseudoVFNCVT_X_F_ALT">;
defm : VPatConversionVI_WF_BF16<"int_riscv_vfncvt_rtz_xu_f_w", "PseudoVFNCVT_RTZ_XU_F_ALT">;
defm : VPatConversionVI_WF_BF16<"int_riscv_vfncvt_rtz_x_f_w", "PseudoVFNCVT_RTZ_X_F_ALT">;
defm : VPatConversionVF_WF_RM<"int_riscv_vfncvt_f_f_w", "PseudoVFNCVT_F_F_ALT",
                              AllWidenableBF16ToFloatVectors, isSEWAware=1>;
defm : VPatConversionVF_WF_BF16<"int_riscv_vfncvt_rod_f_f_w", "PseudoVFNCVT_ROD_F_F_ALT",
                                isSEWAware=1>;
defm : VPatBinaryV_VX<"int_riscv_vfslide1up", "PseudoVFSLIDE1UP_ALT", AllBF16Vectors>;
defm : VPatBinaryV_VX<"int_riscv_vfslide1down", "PseudoVFSLIDE1DOWN_ALT", AllBF16Vectors>;

foreach fvti = AllBF16Vectors in {
  defvar ivti = GetIntVTypeInfo<fvti>.Vti;
  let Predicates = GetVTypePredicates<ivti>.Predicates in {
    // 13.16. Vector Floating-Point Move Instruction
    // If we're splatting fpimm0, use vmv.v.x vd, x0.
    def : Pat<(fvti.Vector (riscv_vfmv_v_f_vl
                           fvti.Vector:$passthru, (fvti.Scalar (fpimm0)), VLOpFrag)),
              (!cast<Instruction>("PseudoVMV_V_I_"#fvti.LMul.MX)
               $passthru, 0, GPR:$vl, fvti.Log2SEW, TU_MU)>;
    def : Pat<(fvti.Vector (riscv_vfmv_v_f_vl
                           fvti.Vector:$passthru, (fvti.Scalar (SelectScalarFPAsInt (XLenVT GPR:$imm))), VLOpFrag)),
              (!cast<Instruction>("PseudoVMV_V_X_"#fvti.LMul.MX)
               $passthru, GPR:$imm, GPR:$vl, fvti.Log2SEW, TU_MU)>;
  }

  let Predicates = GetVTypePredicates<fvti>.Predicates in {
    def : Pat<(fvti.Vector (riscv_vfmv_v_f_vl
                           fvti.Vector:$passthru, (fvti.Scalar fvti.ScalarRegClass:$rs2), VLOpFrag)),
              (!cast<Instruction>("PseudoVFMV_V_ALT_" # fvti.ScalarSuffix # "_" #
                                  fvti.LMul.MX)
               $passthru, (fvti.Scalar fvti.ScalarRegClass:$rs2),
               GPR:$vl, fvti.Log2SEW, TU_MU)>;
  }
}

foreach vti = NoGroupBF16Vectors in {
  let Predicates = GetVTypePredicates<vti>.Predicates in {
    def : Pat<(vti.Vector (riscv_vfmv_s_f_vl (vti.Vector vti.RegClass:$passthru),
                                             (vti.Scalar (fpimm0)),
                                             VLOpFrag)),
              (PseudoVMV_S_X $passthru, (XLenVT X0), GPR:$vl, vti.Log2SEW)>;
    def : Pat<(vti.Vector (riscv_vfmv_s_f_vl (vti.Vector vti.RegClass:$passthru),
                                             (vti.Scalar (SelectScalarFPAsInt (XLenVT GPR:$imm))),
                                             VLOpFrag)),
              (PseudoVMV_S_X $passthru, GPR:$imm, GPR:$vl, vti.Log2SEW)>;
    def : Pat<(vti.Vector (riscv_vfmv_s_f_vl (vti.Vector vti.RegClass:$passthru),
                                             vti.ScalarRegClass:$rs1,
                                             VLOpFrag)),
              (!cast<Instruction>("PseudoVFMV_S_"#vti.ScalarSuffix#"_ALT")
                  vti.RegClass:$passthru,
                  (vti.Scalar vti.ScalarRegClass:$rs1), GPR:$vl, vti.Log2SEW)>;
  }

  defvar vfmv_f_s_inst = !cast<Instruction>(!strconcat("PseudoVFMV_",
                                                       vti.ScalarSuffix,
                                                       "_S_ALT"));
  // Only pattern-match extract-element operations where the index is 0. Any
  // other index will have been custom-lowered to slide the vector correctly
  // into place.
  let Predicates = GetVTypePredicates<vti>.Predicates in
  def : Pat<(vti.Scalar (extractelt (vti.Vector vti.RegClass:$rs2), 0)),
            (vfmv_f_s_inst vti.RegClass:$rs2, vti.Log2SEW)>;
}
} // Predicates = [HasStdExtZvfbfa]

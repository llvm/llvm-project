//===---- RISCVInstrInfoZvzip.td - 'Zvzip' instructions ----*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file describes the RISC-V instructions from the Zvzip standard extension
// for reordering structured data in vector registers.
//
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// Instructions
//===----------------------------------------------------------------------===//

let Predicates = [HasStdExtZvzip], Constraints = "@earlyclobber $vd" in {
  let VS1VS2Constraint = WidenV in
  def VZIP_VV : VALUVV<0b111110, OPMVV, "vzip.vv">;

  let VS1VS2Constraint = VS2Constraint in {
    def VUNZIPE_V : VALUVs2<0b010010, 0b01011, OPMVV, "vunzipe.v">;
    def VUNZIPO_V : VALUVs2<0b010010, 0b01111, OPMVV, "vunzipo.v">;
  }

  let VS1VS2Constraint = Vrgather in {
    def VPAIRE_VV : VALUVV<0b001111, OPIVV, "vpaire.vv">;
    def VPAIRO_VV : VALUVV<0b001111, OPMVV, "vpairo.vv">;
  }
} // Predicates = [HasStdExtZvzip]

defset list<VTypeInfoToWide> AllZvzipVectors = {
  def : VTypeInfoToWide<VI8MF8,  VI8MF4>;
  def : VTypeInfoToWide<VI8MF4,  VI8MF2>;
  def : VTypeInfoToWide<VI8MF2,  VI8M1>;
  def : VTypeInfoToWide<VI8M1,   VI8M2>;
  def : VTypeInfoToWide<VI8M2,   VI8M4>;
  def : VTypeInfoToWide<VI8M4,   VI8M8>;

  def : VTypeInfoToWide<VI16MF4, VI16MF2>;
  def : VTypeInfoToWide<VI16MF2, VI16M1>;
  def : VTypeInfoToWide<VI16M1,  VI16M2>;
  def : VTypeInfoToWide<VI16M2,  VI16M4>;
  def : VTypeInfoToWide<VI16M4,  VI16M8>;

  def : VTypeInfoToWide<VI32MF2, VI32M1>;
  def : VTypeInfoToWide<VI32M1,  VI32M2>;
  def : VTypeInfoToWide<VI32M2,  VI32M4>;
  def : VTypeInfoToWide<VI32M4,  VI32M8>;

  def : VTypeInfoToWide<VI64M1,  VI64M2>;
  def : VTypeInfoToWide<VI64M2,  VI64M4>;
  def : VTypeInfoToWide<VI64M4,  VI64M8>;

  // Floating-point 16-bit
  def : VTypeInfoToWide<VF16MF4, VF16MF2>;
  def : VTypeInfoToWide<VF16MF2, VF16M1>;
  def : VTypeInfoToWide<VF16M1,  VF16M2>;
  def : VTypeInfoToWide<VF16M2,  VF16M4>;
  def : VTypeInfoToWide<VF16M4,  VF16M8>;

  // BF16
  def : VTypeInfoToWide<VBF16MF4, VBF16MF2>;
  def : VTypeInfoToWide<VBF16MF2, VBF16M1>;
  def : VTypeInfoToWide<VBF16M1,  VBF16M2>;
  def : VTypeInfoToWide<VBF16M2,  VBF16M4>;
  def : VTypeInfoToWide<VBF16M4,  VBF16M8>;

  // Floating-point 32-bit
  def : VTypeInfoToWide<VF32MF2, VF32M1>;
  def : VTypeInfoToWide<VF32M1,  VF32M2>;
  def : VTypeInfoToWide<VF32M2,  VF32M4>;
  def : VTypeInfoToWide<VF32M4,  VF32M8>;

  // Floating-point 64-bit
  def : VTypeInfoToWide<VF64M1,  VF64M2>;
  def : VTypeInfoToWide<VF64M2,  VF64M4>;
  def : VTypeInfoToWide<VF64M4,  VF64M8>;
}

multiclass VPseudoVZIP {
  foreach m = MxListW in
    defm "" : VPseudoBinaryW_VV<m, Commutable=0>,
              SchedBinary<"WriteVIWALUV", "ReadVIWALUV", "ReadVIWALUV", m.MX,
                          forcePassthruRead=true>;
}

multiclass VPseudoVUNZIP {
  foreach m = MxListW in {
    defvar mx = m.MX;
    let VLMul = m.value in {
      def "_V_" # mx : VPseudoUnaryNoMask<m.vrclass, m.wvrclass>,
                       SchedUnary<"WriteVIALUV", "ReadVIALUV", mx,
                                  forcePassthruRead=true>;
      def "_V_" # mx # "_MASK" :
        VPseudoUnaryMask<m.vrclass, m.wvrclass>,
        RISCVMaskedPseudo<MaskIdx=2>,
        SchedUnary<"WriteVIALUV", "ReadVIALUV", mx, forcePassthruRead=true>;
    }
  }
}

let Predicates = [HasStdExtZvzip],
    Constraints = "@earlyclobber $rd, $rd = $passthru" in {
  defm PseudoVZIP    : VPseudoVZIP;
  defm PseudoVUNZIPE : VPseudoVUNZIP;
  defm PseudoVUNZIPO : VPseudoVUNZIP;
  defm PseudoVPAIRE  : VPseudoVALU_VV;
  defm PseudoVPAIRO  : VPseudoVALU_VV;
}

multiclass VPatVUnzipIntrinsic<string intrinsic_name,
                               string instruction_name> {
  foreach VtiToWti = AllZvzipVectors in {
    defvar vti = VtiToWti.Vti;
    defvar wti = VtiToWti.Wti;
    let Predicates = !listconcat(GetVTypeMinimalPredicates<wti>.Predicates,
                                 [HasStdExtZvzip]) in {
      def : VPatUnaryNoMask<intrinsic_name, instruction_name, "V",
                            vti.Vector, wti.Vector, vti.Log2SEW,
                            vti.LMul, vti.RegClass, wti.RegClass>;
      def : VPatUnaryMask<intrinsic_name, instruction_name, "V",
                          vti.Vector, wti.Vector, vti.Mask, vti.Log2SEW,
                          vti.LMul, vti.RegClass, wti.RegClass>;
    }
  }
}

defm : VPatVUnzipIntrinsic<"int_riscv_vunzipe", "PseudoVUNZIPE">;
defm : VPatVUnzipIntrinsic<"int_riscv_vunzipo", "PseudoVUNZIPO">;
defm : VPatBinaryV_VV<"int_riscv_vpaire", "PseudoVPAIRE", AllVectors,
                      ExtraPreds = [HasStdExtZvzip],
                      requireMinimal = true>;
defm : VPatBinaryV_VV<"int_riscv_vpairo", "PseudoVPAIRO", AllVectors,
                      ExtraPreds = [HasStdExtZvzip],
                      requireMinimal = true>;
defm : VPatBinaryW_VV<"int_riscv_vzip", "PseudoVZIP", AllZvzipVectors,
                      ExtraPreds = [HasStdExtZvzip],
                      requireMinimal = true>;

// (vd (op vs2, vs1, passthru, mask, vl))
def SDT_RISCVZip_VL : SDTypeProfile<1, 5, [SDTCisVec<0>, SDTCisVec<1>,
                                           SDTCisSubVecOfVec<1, 0>,
                                           SDTCisSameAs<1, 2>,
                                           SDTCisSameAs<0, 3>,
                                           SDTCVecEltisVT<4, i1>,
                                           SDTCisSameNumEltsAs<1, 4>,
                                           SDTCisVT<5, XLenVT>]>;

def SDT_RISCVVecBinOp_VL : SDTypeProfile<1, 5, [SDTCisSameAs<0, 1>,
                                                SDTCisSameAs<0, 2>,
                                                SDTCisVec<0>,
                                                SDTCisSameAs<0, 3>,
                                                SDTCVecEltisVT<4, i1>,
                                                SDTCisSameNumEltsAs<0, 4>,
                                                SDTCisVT<5, XLenVT>]>;

let HasPassthruOp = true, HasMaskOp = true in {
def vpaire_vl  : RVSDNode<"VPAIRE_VL", SDT_RISCVVecBinOp_VL>;
def vpairo_vl  : RVSDNode<"VPAIRO_VL", SDT_RISCVVecBinOp_VL>;
def vzip_vl    : RVSDNode<"VZIP_VL", SDT_RISCVZip_VL>;
} // HasPassthruOp = true, HasMaskOp = true

multiclass VPatVZIP<SDPatternOperator vop, string instruction_name> {
  foreach VtiToWti = AllZvzipVectors in {
    defvar vti = VtiToWti.Vti;
    defvar wti = VtiToWti.Wti;
    let Predicates = !listconcat([HasStdExtZvzip],
                                 GetVTypeMinimalPredicates<vti>.Predicates) in {
      def : VPatBinaryVL_V<vop, instruction_name, "VV",
                           wti.Vector, vti.Vector, vti.Vector, vti.Mask,
                           vti.Log2SEW, vti.LMul, wti.RegClass, vti.RegClass,
                           vti.RegClass>;
    }
  }
}

defm : VPatVZIP<vzip_vl, "PseudoVZIP">;
defm : VPatBinaryVL_VV<vpaire_vl, "PseudoVPAIRE", AllVectors,
                       ExtraPreds = [HasStdExtZvzip],
                       requireMinimal = true>;
defm : VPatBinaryVL_VV<vpairo_vl, "PseudoVPAIRO", AllVectors,
                       ExtraPreds = [HasStdExtZvzip],
                       requireMinimal = true>;

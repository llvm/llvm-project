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

def SDT_RISCVZip_VL : SDTypeProfile<1, 5, [SDTCisVec<0>, SDTCisVec<1>,
                                           SDTCisInt<0>, SDTCisInt<1>,
                                           SDTCisSubVecOfVec<1, 0>,
                                           SDTCisSameAs<1, 2>,
                                           SDTCisSameAs<0, 3>,
                                           SDTCVecEltisVT<4, i1>,
                                           SDTCisSameNumEltsAs<1, 4>,
                                           SDTCisVT<5, XLenVT>]>;
def SDT_RISCVUnzip_VL : SDTypeProfile<1, 4, [SDTCisVec<0>, SDTCisVec<1>,
                                             SDTCisInt<0>, SDTCisInt<1>,
                                             SDTCisSubVecOfVec<0, 1>,
                                             SDTCisSameAs<0, 2>,
                                             SDTCVecEltisVT<3, i1>,
                                             SDTCisSameNumEltsAs<1, 3>,
                                             SDTCisVT<4, XLenVT>]>;
// These are modeled after the int binop VL nodes
let HasPassthruOp = true, HasMaskOp = true in {
def vzip_vl    : RVSDNode<"VZIP_VL", SDT_RISCVZip_VL>;
def vunzipe_vl : RVSDNode<"VUNZIPE_VL", SDT_RISCVUnzip_VL>;
def vunzipo_vl : RVSDNode<"VUNZIPO_VL", SDT_RISCVUnzip_VL>;
def vpaire_vl  : RVSDNode<"VPAIRE_VL", SDT_RISCVIntBinOp_VL>;
def vpairo_vl  : RVSDNode<"VPAIRO_VL", SDT_RISCVIntBinOp_VL>;
} // HasPassthruOp = true, HasMaskOp = true
multiclass VPseudoVZIP {
  foreach m = MxListW in
    defm "" : VPseudoBinaryW_VV<m, Commutable=0>;
}
multiclass VPseudoVUNZIP {
  foreach m = MxListW in {
    defvar mx = m.MX;
    let VLMul = m.value in {
      def "_V_" # mx : VPseudoUnaryNoMask<m.vrclass, m.wvrclass>,
                       SchedUnary<"WriteVIALUV", "ReadVIALUV", mx, forcePassthruRead=true>;
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
}
multiclass VPatVZIP<SDPatternOperator vop, string instruction_name> {
  foreach VtiToWti = AllZvzipVectors in {
    defvar vti = VtiToWti.Vti;
    defvar wti = VtiToWti.Wti;
    let Predicates = !listconcat([HasStdExtZvzip], GetVTypePredicates<vti>.Predicates) in {
      def : Pat<(wti.Vector (vop
                       (vti.Vector vti.RegClass:$rs1),
                       (vti.Vector vti.RegClass:$rs2),
                       (wti.Vector wti.RegClass:$passthru),
                       (vti.Mask VMV0:$vm),
                       VLOpFrag)),
                (!cast<Instruction>(instruction_name#"_VV_"#vti.LMul.MX#"_MASK")
                             wti.RegClass:$passthru,
                             vti.RegClass:$rs1,
                             vti.RegClass:$rs2,
                             (vti.Mask VMV0:$vm), GPR:$vl, vti.Log2SEW, TAIL_AGNOSTIC)>;
    }
  }
}
multiclass VPatVUNZIP<SDPatternOperator op, string instruction_name> {
  foreach VtiToWti = AllZvzipVectors in {
    defvar vti = VtiToWti.Vti;
    defvar wti = VtiToWti.Wti;
    let Predicates = !listconcat([HasStdExtZvzip],
                                 GetVTypePredicates<vti>.Predicates) in {
      def : Pat<(vti.Vector (op (wti.Vector wti.RegClass:$rs1),
                                (vti.Vector vti.RegClass:$passthru),
                                (wti.Mask VMV0:$vm),
                                VLOpFrag)),
                (!cast<Instruction>(instruction_name#"_V_"#vti.LMul.MX#"_MASK")
                  vti.RegClass:$passthru,
                  wti.RegClass:$rs1,
                  (wti.Mask VMV0:$vm),
                  GPR:$vl,
                  vti.Log2SEW,
                  TAIL_AGNOSTIC)>;
    }
  }
}
defm : VPatVZIP<vzip_vl, "PseudoVZIP">;
defm : VPatVUNZIP<vunzipe_vl, "PseudoVUNZIPE">;
defm : VPatVUNZIP<vunzipo_vl, "PseudoVUNZIPO">;
defm : VPatBinaryVL_VV<vpaire_vl, "PseudoVPAIRE">;
defm : VPatBinaryVL_VV<vpairo_vl, "PseudoVPAIRO">;

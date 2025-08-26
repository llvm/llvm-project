//===-- RISCVInstrInfoXRivos.td ----------------------------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file describes the vendor extensions defined by Rivos Inc.
//
//===----------------------------------------------------------------------===//

class CustomRivosVXI<bits<6> funct6, RISCVVFormat opv, dag outs, dag ins,
                string opcodestr, string argstr>
    : RVInst<outs, ins, opcodestr, argstr, [], InstFormatR> {
  bits<5> imm;
  bits<5> rs1;
  bits<5> vd;
  bit vm = 0;

  let Inst{31-26} = funct6;
  let Inst{25} = vm;
  let Inst{24-20} = imm;
  let Inst{19-15} = rs1;
  let Inst{14-12} = opv.Value;
  let Inst{11-7} = vd;
  let Inst{6-0} = OPC_CUSTOM_2.Value;

  let Uses = [VL, VTYPE];
  let RVVConstraint = NoConstraint;
  let Constraints = "$vd = $vd_wb";
}

class CustomRivosXVI<bits<6> funct6, RISCVVFormat opv, dag outs, dag ins,
                string opcodestr, string argstr>
    : RVInst<outs, ins, opcodestr, argstr, [], InstFormatR> {
  bits<5> imm;
  bits<5> vs2;
  bits<5> rd;
  bit vm = 1;

  let Inst{31-26} = funct6;
  let Inst{25} = vm;
  let Inst{24-20} = vs2;
  let Inst{19-15} = imm;
  let Inst{14-12} = opv.Value;
  let Inst{11-7} = rd;
  let Inst{6-0} = OPC_CUSTOM_2.Value;

  let Uses = [VL, VTYPE];
  let RVVConstraint = NoConstraint;
}

//===----------------------------------------------------------------------===//
// XRivosVizip
//===----------------------------------------------------------------------===//


let Predicates = [HasVendorXRivosVizip], DecoderNamespace = "XRivos",
  Constraints = "@earlyclobber $vd", RVVConstraint = Vrgather,
  Inst<6-0> = OPC_CUSTOM_2.Value, ReadsPastVL = 1 in  {
defm RI_VZIPEVEN_V : VALU_IV_V<"ri.vzipeven", 0b001100>;
defm RI_VZIPODD_V : VALU_IV_V<"ri.vzipodd", 0b011100>;
defm RI_VZIP2A_V : VALU_IV_V<"ri.vzip2a", 0b000100>;
defm RI_VZIP2B_V : VALU_IV_V<"ri.vzip2b", 0b010100>;
defm RI_VUNZIP2A_V : VALU_IV_V<"ri.vunzip2a", 0b001000>;
defm RI_VUNZIP2B_V : VALU_IV_V<"ri.vunzip2b", 0b011000>;
}

// These are modeled after the int binop VL nodes
let HasMaskOp = true in {
  def ri_vzipeven_vl : RVSDNode<"RI_VZIPEVEN_VL", SDT_RISCVIntBinOp_VL>;
  def ri_vzipodd_vl : RVSDNode<"RI_VZIPODD_VL", SDT_RISCVIntBinOp_VL>;
  def ri_vzip2a_vl : RVSDNode<"RI_VZIP2A_VL", SDT_RISCVIntBinOp_VL>;
  def ri_vzip2b_vl : RVSDNode<"RI_VZIP2B_VL", SDT_RISCVIntBinOp_VL>;
  def ri_vunzip2a_vl : RVSDNode<"RI_VUNZIP2A_VL", SDT_RISCVIntBinOp_VL>;
  def ri_vunzip2b_vl : RVSDNode<"RI_VUNZIP2B_VL", SDT_RISCVIntBinOp_VL>;
}

multiclass RIVPseudoVALU_VV {
  foreach m = MxList in
    defm "" : VPseudoBinaryV_VV<m, Commutable=0>;
}

let Predicates = [HasVendorXRivosVizip],
  Constraints = "@earlyclobber $rd, $rd = $passthru" in {
defm PseudoRI_VZIPEVEN   : RIVPseudoVALU_VV;
defm PseudoRI_VZIPODD   : RIVPseudoVALU_VV;
defm PseudoRI_VZIP2A   : RIVPseudoVALU_VV;
defm PseudoRI_VZIP2B   : RIVPseudoVALU_VV;
defm PseudoRI_VUNZIP2A   : RIVPseudoVALU_VV;
defm PseudoRI_VUNZIP2B   : RIVPseudoVALU_VV;
}

multiclass RIVPatBinaryVL_VV<SDPatternOperator vop, string instruction_name,
                              list<VTypeInfo> vtilist = AllIntegerVectors,
                              bit isSEWAware = false> {
  foreach vti = vtilist in
    let Predicates = GetVTypePredicates<vti>.Predicates in
      def : VPatBinaryVL_V<vop, instruction_name, "VV",
                           vti.Vector, vti.Vector, vti.Vector, vti.Mask,
                           vti.Log2SEW, vti.LMul, vti.RegClass, vti.RegClass,
                           vti.RegClass, isSEWAware>;
}

defm : RIVPatBinaryVL_VV<ri_vzipeven_vl, "PseudoRI_VZIPEVEN">;
defm : RIVPatBinaryVL_VV<ri_vzipodd_vl, "PseudoRI_VZIPODD">;
defm : RIVPatBinaryVL_VV<ri_vzip2a_vl, "PseudoRI_VZIP2A">;
defm : RIVPatBinaryVL_VV<ri_vzip2b_vl, "PseudoRI_VZIP2B">;
defm : RIVPatBinaryVL_VV<ri_vunzip2a_vl, "PseudoRI_VUNZIP2A">;
defm : RIVPatBinaryVL_VV<ri_vunzip2b_vl, "PseudoRI_VUNZIP2B">;

//===----------------------------------------------------------------------===//
// XRivosVisni
//===----------------------------------------------------------------------===//

let Predicates = [HasVendorXRivosVisni], DecoderNamespace = "XRivos",
  mayLoad = false, mayStore = false, hasSideEffects = false in {

let vm = 0, vs2=0, Inst<6-0> = OPC_CUSTOM_2.Value,
    isReMaterializable = 1, isAsCheapAsAMove = 1 in
def RI_VZERO : RVInstV<0b000000, 0b00000, OPCFG, (outs VR:$vd),
                     (ins), "ri.vzero.v", "$vd">;

def RI_VINSERT : CustomRivosVXI<0b010000, OPMVX, (outs VR:$vd_wb),
                                (ins VR:$vd, GPR:$rs1, uimm5:$imm),
                                "ri.vinsert.v.x", "$vd, $rs1, $imm">;

let ReadsPastVL = 1 in
def RI_VEXTRACT : CustomRivosXVI<0b010111, OPMVV, (outs GPR:$rd),
                                (ins VR:$vs2, uimm5:$imm),
                                "ri.vextract.x.v", "$rd, $vs2, $imm">;
}

// RI_VEXTRACT matches the semantics of ri.vextract.x.v. The result is always
// XLenVT sign extended from the vector element size.  VEXTRACT does *not*
// have a VL operand.
def ri_vextract : RVSDNode<"RI_VEXTRACT",
                           SDTypeProfile<1, 2, [SDTCisInt<0>, SDTCisVec<1>,
                                                SDTCisInt<2>,
                                                SDTCisInt<1>]>>;

// RI_VINSERT_VL matches the semantics of ri.vinsert.v.x. It carries a VL operand.
def ri_vinsert_vl : RVSDNode<"RI_VINSERT_VL",
                             SDTypeProfile<1, 5, [SDTCisSameAs<0, 1>,
                                                  SDTCisInt<0>,
                                                  SDTCisVT<2, XLenVT>,
                                                  SDTCisVT<3, XLenVT>,
                                                  SDTCisVT<4, XLenVT>]>>;

let Predicates = [HasVendorXRivosVisni], mayLoad = 0, mayStore = 0,
    hasSideEffects = 0, HasSEWOp = 1 in
foreach m = MxList in {
  defvar mx = m.MX;
  let VLMul = m.value in {
    let BaseInstr = RI_VEXTRACT in
    def PseudoRI_VEXTRACT_  # mx :
      RISCVVPseudo<(outs GPR:$rd),
                   (ins m.vrclass:$rs2, uimm5:$idx, ixlenimm:$sew),
                   []>;

    let HasVLOp = 1, BaseInstr = RI_VINSERT, HasVecPolicyOp = 1,
        Constraints = "$rd = $rs1" in
    def PseudoRI_VINSERT_ # mx :
      RISCVVPseudo<(outs m.vrclass:$rd),
                   (ins m.vrclass:$rs1, GPR:$rs2, uimm5:$idx, AVL:$vl,
                        ixlenimm:$sew, ixlenimm:$policy),
                   []>;
  }
}



foreach vti = AllIntegerVectors in
  let Predicates = GetVTypePredicates<vti>.Predicates in {
    def : Pat<(XLenVT (ri_vextract (vti.Vector vti.RegClass:$vs2), uimm5:$imm)),
              (!cast<Instruction>("PseudoRI_VEXTRACT_" # vti.LMul.MX)
               $vs2, uimm5:$imm, vti.Log2SEW)>;

    def : Pat<(vti.Vector (ri_vinsert_vl (vti.Vector vti.RegClass:$merge),
                                          vti.ScalarRegClass:$rs1,
                                          uimm5:$imm,
                                          VLOpFrag,
                                          (XLenVT timm:$policy))),
              (!cast<Instruction>("PseudoRI_VINSERT_" # vti.LMul.MX)
               $merge, vti.ScalarRegClass:$rs1, uimm5:$imm,
               GPR:$vl, vti.Log2SEW, (XLenVT timm:$policy))>;

  }

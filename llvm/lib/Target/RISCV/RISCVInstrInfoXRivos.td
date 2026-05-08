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
  let VMConstraint = false;
  let Constraints = "$vd = $vd_wb";

  let UseNamedOperandTable = true;
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
  let VMConstraint = false;

  let UseNamedOperandTable = true;
}

//===----------------------------------------------------------------------===//
// XRivosVizip
//===----------------------------------------------------------------------===//


let Predicates = [HasVendorXRivosVizip], DecoderNamespace = "XRivos",
  Constraints = "@earlyclobber $vd", VS1VS2Constraint = Vrgather,
  Inst{6-0} = OPC_CUSTOM_2.Value, ReadsPastVL = 1,
  ElementsDependOn = EltDepsNone in  {
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
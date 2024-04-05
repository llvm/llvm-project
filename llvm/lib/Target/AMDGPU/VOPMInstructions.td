//===-- VOPMInstructions.td - Vector Instruction Definitions --------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// VOPM Encodings
//===----------------------------------------------------------------------===//

class VOPMProfile<list<ValueType> ArgTy, bit _HasAux = 0, bit _HasClamp = 0> : VOPProfile<ArgTy> {
  bit HasAux = _HasAux;
  let HasClamp = _HasClamp;

  let HasModifiers = 0;
  let HasOMod = 0;
}

class VOP2Me <bits<8> op, VOPMProfile pfl> : Enc64 {
  bits<10> vdst;
  bits<11> src0;
  bits<11> src1;
  bits<6> aux_data;

  // TODO-GFX13: Inst{9-0} = vdst;
  let Inst{7-0} = vdst{7-0};
  let Inst{11-10} = ?;    // vdst_idx
  let Inst{13} = ?;       // supr

  let Inst{22-15} = op;
  let Inst{31-24} = 0xdc; // encoding

  // TODO-GFX13: Inst{42-32} = src0;
  let Inst{40-32} = src0{8-0};
  let Inst{44-43} = ?;    // src0_idx

  // TODO-GFX13: Inst{55-45} = !if(pfl.HasSrc1, src1, ?);
  let Inst{53-45} = !if(pfl.HasSrc1, src1{8-0}, ?);
  let Inst{57-56} = ?;    // src1_idx

  let Inst{63-58} = !if(pfl.HasAux, aux_data, ?);
}

class VOP3Me <bits<8> op, VOPMProfile pfl> : Enc96 {

  bits<10> vdst;
  bits<11> src0;
  bits<11> src1;
  bits<11> src2;
  bits<6> aux_data;

  bit clamp;

  let Inst{7-0} = vdst{7-0};
  let Inst{9-8} = ?;      // vdst cont
  let Inst{11-10} = ?;    // vdst_idx
  let Inst{12} = !if(pfl.HasClamp, clamp, ?);
  let Inst{13} = ?;       // supr

  let Inst{23-16} = op;
  let Inst{31-24} = 0xde; // encoding

  let Inst{40-32} = src0{8-0};
  let Inst{42-41} = ?;    // src0 cont
  let Inst{44-43} = ?;    // src0_idx
  let Inst{53-45} = src1{8-0};
  let Inst{55-54} = ?;    // src1 cont
  let Inst{57-56} = ?;    // src1_idx
  let Inst{63-58} = !if(pfl.HasAux, aux_data, ?);
  let Inst{72-64} = src2{8-0};
  let Inst{74-73} = ?;    // src2 cont
  let Inst{76-75} = ?;    // src2_idx

  let Inst{78-77} = ?;    // sparse_sel
  let Inst{88-79} = ?;    // aux_data1
}

class VOP5Me <bits<8> op, VOPMProfile pfl> : Enc128 {
  bits<10> vdst;
  bits<11> src0;
  bits<11> src1;
  bits<11> src2;
  bits<10> src3;
  bits<10> src4;
  bits<6> aux_data;

  bit clamp;

  let Inst{7-0} = vdst{7-0};
  let Inst{9-8} = ?;      // vdst cont
  let Inst{11-10} = ?;    // vdst_idx
  let Inst{12} = !if(pfl.HasClamp, clamp, ?);
  let Inst{13} = ?;       // supr

  let Inst{23-16} = op;
  let Inst{31-24} = 0xde; // encoding

  let Inst{40-32} = src0{8-0};
  let Inst{42-41} = ?;    // src0 cont
  let Inst{44-43} = ?;    // src0_idx
  let Inst{53-45} = src1{8-0};
  let Inst{55-54} = ?;    // src1 cont
  let Inst{57-56} = ?;    // src1_idx
  let Inst{63-58} = !if(pfl.HasAux, aux_data, ?);
  let Inst{72-64} = src2{8-0};
  let Inst{74-73} = ?;    // src2 cont
  let Inst{76-75} = ?;    // src2_idx

  let Inst{78-77} = ?;    // sparse_sel
  let Inst{88-79} = ?;    // aux_data1

  let Inst{104-96} = src3{8-0};
  let Inst{105} = ?;     // src3 cont
  let Inst{107-106} = ?; // src3_idx
  let Inst{116-108} = src4{8-0};
  let Inst{117} = ?;     // src4 cont
  let Inst{119-118} = ?; // src4_idx
}

class VOP6Me <bits<8> op, VOPMProfile pfl> : Enc128 {
  bits<10> vdst;
  bits<11> src0;
  bits<11> src1;
  bits<11> src2;
  bits<10> src3;
  bits<10> src4;
  bits<10> src5;
  bits<6> aux_data;

  bit clamp;

  let Inst{7-0} = vdst{7-0};
  let Inst{9-8} = ?;      // vdst cont
  let Inst{11-10} = ?;    // vdst_idx
  let Inst{12} = !if(pfl.HasClamp, clamp, ?);
  let Inst{13} = ?;       // supr

  let Inst{23-16} = op;
  let Inst{31-24} = 0xde; // encoding

  let Inst{40-32} = src0{8-0};
  let Inst{42-41} = ?;    // src0 cont
  let Inst{44-43} = ?;    // src0_idx
  let Inst{53-45} = src1{8-0};
  let Inst{55-54} = ?;    // src1 cont
  let Inst{57-56} = ?;    // src1_idx
  let Inst{63-58} = !if(pfl.HasAux, aux_data, ?);
  let Inst{72-64} = src2{8-0};
  let Inst{74-73} = ?;    // src2 cont
  let Inst{76-75} = ?;    // src2_idx

  let Inst{92-84} = src5{8-0};
  let Inst{93}    = ?;    // src5 cont
  let Inst{95-94} = ?;    // src5_idx

  let Inst{104-96} = src3{8-0};
  let Inst{105} = ?;     // src3 cont
  let Inst{107-106} = ?; // src3_idx
  let Inst{116-108} = src4{8-0};
  let Inst{117} = ?;     // src4 cont
  let Inst{119-118} = ?; // src4_idx
}

//===----------------------------------------------------------------------===//
// VOPM Pseudos
//===----------------------------------------------------------------------===//

class VOP2M_Profile<list<ValueType> ArgTy, bit _HasAux = 0> : VOPMProfile<ArgTy, _HasAux, 0> {
  bit HasAux = _HasAux;

  let Src1RC64 = !if(!eq(Src1VT, i64), SCSrc_b64, VSrc_f32);
  let Ins64 = !con((ins Src0RC64:$src0), !if(HasSrc1, (ins Src1RC64:$src1), (ins)), !if(HasAux, (ins AuxData:$aux_data), (ins)));

  string Src1Asm = !if(HasSrc1, ", $src1", "");
  string AuxDataAsm = !if(HasAux, "$aux_data", "");

  let Asm64 = "$vdst, $src0"#Src1Asm#AuxDataAsm;
}

def VOPM_I32_I32_I32 : VOP2M_Profile<[i32, i32, i32, untyped]>;
def VOPM_I32_I32_I32_AUX : VOP2M_Profile<[i32, i32, i32, untyped], 1>;
def VOPM_I32_I32_AUX : VOP2M_Profile<[i32, i32, untyped, untyped], 1>;
def VOPM_I32_I32_I64 : VOP2M_Profile<[i32, i32, i64, untyped]>;

class VOP3M_Permute_Profile<list<ValueType> ArgTy> : VOPMProfile<ArgTy, 1> {

  let Outs64 = (outs DstRC64:$vdst, Src2RC64:$src2);
  let Ins64 = (ins Src0RC64:$src0, Src1RC64:$src1, AuxData:$aux_data);

  let Asm64 = "$vdst, $src2, $src0, $src1$aux_data";
}

def VOPM_PERMUTE_I32_I32_I32 : VOP3M_Permute_Profile<[i32, i32, i32, untyped]>;

class Conv_Profile<list<ValueType> ArgTy, int PixelShape, bit FilterType3x3> : VOPMProfile<ArgTy, 1, 1> {
  let Outs64 = (outs DstRC64:$vdst);
  let Ins64 = !if(FilterType3x3, (ins Src0RC64:$src0, Src1RC64:$src1, Src2RC64:$src2, Src2RC64:$src3, Src2RC64:$src4, AuxData:$aux_data, clampmod0:$clamp),
                                 (ins Src0RC64:$src0, Src1RC64:$src1, Src2RC64:$src2, Src2RC64:$src3, Src2RC64:$src4, Src2RC64:$src5, AuxData:$aux_data, clampmod0:$clamp));

  dag ConvInPat = !if(FilterType3x3, (ins Src0VT:$src0, Src1VT:$src1, Src2VT:$src2, Src2VT:$src3, Src2VT:$src4, timm:$aux_data, i1:$clamp),
                                     (ins Src0VT:$src0, Src1VT:$src1, Src2VT:$src2, Src2VT:$src3, Src2VT:$src4, Src2VT:$src5, timm:$aux_data, i1:$clamp));
  dag ConvOutPat = !if(FilterType3x3, (ins Src0VT:$src0, Src1VT:$src1, Src2VT:$src2, Src2VT:$src3, Src2VT:$src4, timm:$aux_data, i1:$clamp),
                                      (ins Src0VT:$src0, Src1VT:$src1, Src2VT:$src2, Src2VT:$src3, Src2VT:$src4, Src2VT:$src5, timm:$aux_data, i1:$clamp));

  let Asm64 = !if(FilterType3x3, "$vdst, $src0, $src1, $src2, $src3, $src4$aux_data$clamp",
                                 "$vdst, $src0, $src1, $src2, $src3, $src4, $src5$aux_data$clamp");

  int PrimType = PixelShape;
  int FilterType = FilterType3x3;
}

// V_WCONV_FP16_FP16
//                                               dest   acc    weights tensor
def VOP5M_CONV_FP16_FP16_3x3_8x4 : Conv_Profile<[v8f16, v8f16, v10f16, v8f16], 1, 1>;
def VOP5M_CONV_FP16_FP16_3x3_4x4 : Conv_Profile<[v8f16, v8f16, v18f16, v6f16], 0, 1>;
def VOP5M_CONV_FP16_FP16_3x3_4x2 : Conv_Profile<[v8f16, v8f16, v36f16, v6f16], 2, 1>;

def VOP5M_CONV_FP16_FP16_1x1_8x4 : Conv_Profile<[v8f16, v8f16, v2f16, v4f16], 1, 0>;
def VOP5M_CONV_FP16_FP16_1x1_4x4 : Conv_Profile<[v8f16, v8f16, v2f16, v2f16], 0, 0>;
def VOP5M_CONV_FP16_FP16_1x1_4x2 : Conv_Profile<[v8f16, v8f16, v4f16, v2f16], 2, 0>;

class VOPM_Pseudo <VOPProfile P, list<dag> pattern = [], string opName = !tolower(NAME)> :
  VOP_Pseudo<opName, "", P, P.Outs64, P.Ins64, "", pattern> {
  let mayRaiseFPException = 0;
  let ReadsModeReg = 0;
  let Uses = [EXEC];
}

def V_MOV_2SRC_B64               : VOPM_Pseudo<VOPM_I32_I32_I32>;
def V_BPERMUTE_B32               : VOPM_Pseudo<VOPM_I32_I32_I32,
                                               [(set i32:$vdst, (int_amdgcn_bpermute_b32 i32:$src0, i32:$src1))]>;

def V_PERMUTE_PAIR_GENSGPR_B32   : VOPM_Pseudo<VOPM_I32_I32_I64,
                                               [(set i32:$vdst, (int_amdgcn_permute_pair_gensgpr_b32 i32:$src0, i64:$src1))]>;
def V_PERMUTE_PAIR_BCAST_B32     : VOPM_Pseudo<VOPM_I32_I32_AUX,
                                               [(set i32:$vdst, (int_amdgcn_permute_pair_bcast_b32 i32:$src0, timm:$aux_data))]>;
def V_PERMUTE_PAIR_2SRC_ROTATE_GROUP_B32 : VOPM_Pseudo<VOPM_I32_I32_I32_AUX,
                                                       [(set i32:$vdst, (int_amdgcn_permute_pair_2src_rotate_group_b32 i32:$src0, i32:$src1, timm:$aux_data))]>;

let SchedRW = [Write32Bit, Write32Bit] in {
  def V_PERMUTE_PAIR_2SRC_INTERLEAVE_B64   : VOPM_Pseudo<VOPM_PERMUTE_I32_I32_I32,
                                                         [(set i32:$vdst, i32:$src2, (int_amdgcn_permute_pair_2src_interleave_b64 i32:$src0, i32:$src1, timm:$aux_data))]>;
  def V_PERMUTE_PACK_TENSOR_2SRC_B64       : VOPM_Pseudo<VOPM_PERMUTE_I32_I32_I32,
                                                         [(set i32:$vdst, i32:$src2, (int_amdgcn_permute_pack_tensor_2src_b64 i32:$src0, i32:$src1, timm:$aux_data))]>;
}

multiclass VOPM_Conv_MultiPseudos<string ProfilePrefix>
{
  def _1x1_8x4 : VOPM_Pseudo<!cast<Conv_Profile>(ProfilePrefix #"_1x1_8x4")>;
  def _1x1_4x4 : VOPM_Pseudo<!cast<Conv_Profile>(ProfilePrefix #"_1x1_4x4")>;
  def _1x1_4x2 : VOPM_Pseudo<!cast<Conv_Profile>(ProfilePrefix #"_1x1_4x2")>;

  def _3x3_8x4 : VOPM_Pseudo<!cast<Conv_Profile>(ProfilePrefix #"_3x3_8x4")>;
  def _3x3_4x4 : VOPM_Pseudo<!cast<Conv_Profile>(ProfilePrefix #"_3x3_4x4")>;
  def _3x3_4x2 : VOPM_Pseudo<!cast<Conv_Profile>(ProfilePrefix #"_3x3_4x2")>;

}

defm V_WCONV_FP16_FP16 : VOPM_Conv_MultiPseudos<"VOP5M_CONV_FP16_FP16">;

//===----------------------------------------------------------------------===//
// VOPM Reals
//===----------------------------------------------------------------------===//

class VOPM_Real <VOPM_Pseudo ps, int EncodingFamily, string real_name = ps.Mnemonic > :
  VOP_Real <ps>,
  InstSI <ps.OutOperandList, ps.InOperandList, real_name # ps.AsmOperands, []>,
  SIMCInstr <ps.PseudoInstr, EncodingFamily> {

  let VALU = 1;
  let isPseudo = 0;
  let isCodeGenOnly = 0;
  let IsPacked = 1;

  let Constraints     = ps.Constraints;
  let DisableEncoding = ps.DisableEncoding;


  // copy relevant pseudo op flags
  let SubtargetPredicate = ps.SubtargetPredicate;
  let OtherPredicates    = ps.OtherPredicates;
  let AsmMatchConverter  = ps.AsmMatchConverter;
  let AsmVariantName     = ps.AsmVariantName;
  let DisableEncoding    = ps.DisableEncoding;
  let TSFlags            = ps.TSFlags;
  let UseNamedOperandTable = ps.UseNamedOperandTable;
  let Uses                 = ps.Uses;
  let Defs                 = ps.Defs;
  let SchedRW              = ps.SchedRW;
  let mayLoad              = ps.mayLoad;
  let mayStore             = ps.mayStore;
  let TRANS                = ps.TRANS;
  let Constraints     = ps.Constraints;

  VOPProfile Pfl = ps.Pfl;
}

class VOPM_Real_Gen <VOPM_Pseudo ps, GFXGen Gen, string real_name = ps.Mnemonic> :
  VOPM_Real <ps, Gen.Subtarget, real_name> {
  let AssemblerPredicate = Gen.AssemblerPredicate;
  let DecoderNamespace = Gen.DecoderNamespace;
}

multiclass VOP2M_Real_Base<GFXGen Gen, bits<8> op,
                           string backing_ps_name = NAME,
                           string asmName = !cast<VOPM_Pseudo>(NAME).Mnemonic> {
  defvar ps = !cast<VOPM_Pseudo>(backing_ps_name);

  def Gen.Suffix :
    VOPM_Real_Gen<ps, Gen, asmName>,
    VOP2Me<op, !cast<VOPMProfile>(ps.Pfl)> {
      let AsmString = asmName # ps.Pfl.Asm64;
  }
}

multiclass VOP2M_Real_gfx13 <bits<8> op> : VOP2M_Real_Base <GFX13Gen, op>;

multiclass VOP3M_Real_Base<GFXGen Gen, bits<8> op,
                           string backing_ps_name = NAME,
                           string asmName = !cast<VOPM_Pseudo>(NAME).Mnemonic> {
  defvar ps = !cast<VOPM_Pseudo>(backing_ps_name);

  def Gen.Suffix :
    VOPM_Real_Gen<ps, Gen, asmName>,
    VOP3Me<op, !cast<VOPMProfile>(ps.Pfl)> {
      let AsmString = asmName # ps.Pfl.Asm64;
  }
}

multiclass VOP3M_Real_gfx13 <bits<8> op> : VOP3M_Real_Base <GFX13Gen, op>;

multiclass VOP5M_Real_Base<GFXGen Gen, bits<8> op,
                                string backing_ps_name = NAME,
                                string asmName = !cast<VOPM_Pseudo>(NAME).Mnemonic> {
  defvar ps = !cast<VOPM_Pseudo>(backing_ps_name);

  def Gen.Suffix :
    VOPM_Real_Gen<ps, Gen, asmName>,
    VOP5Me<op, !cast<VOPMProfile>(ps.Pfl)> {
      let AsmString = asmName # ps.Pfl.Asm64;

      defvar PrimType = !cast<Conv_Profile>(ps.Pfl).PrimType;
      let Inst{59-58} = PrimType;
      defvar FilterType = !cast<Conv_Profile>(ps.Pfl).FilterType;
      let Inst{60} = FilterType;
  }
}

multiclass VOP5M_Conv_Real_gfx13<bits<8> op, string psName> {
  defm _8x4 : VOP5M_Real_Base<GFX13Gen, op, psName#"_3x3_8x4", !tolower(psName)>;
  defm _4x4 : VOP5M_Real_Base<GFX13Gen, op, psName#"_3x3_4x4", !tolower(psName)>;
  defm _4x2 : VOP5M_Real_Base<GFX13Gen, op, psName#"_3x3_4x2", !tolower(psName)>;
}

multiclass VOP6M_Real_Base<GFXGen Gen, bits<8> op,
                                string backing_ps_name = NAME,
                                string asmName = !cast<VOPM_Pseudo>(NAME).Mnemonic> {
  defvar ps = !cast<VOPM_Pseudo>(backing_ps_name);

  def Gen.Suffix :
    VOPM_Real_Gen<ps, Gen, asmName>,
    VOP6Me<op, !cast<VOPMProfile>(ps.Pfl)> {
      let AsmString = asmName # ps.Pfl.Asm64;

      defvar PrimType = !cast<Conv_Profile>(ps.Pfl).PrimType;
      let Inst{59-58} = PrimType;
      defvar FilterType = !cast<Conv_Profile>(ps.Pfl).FilterType;
      let Inst{60} = FilterType;
  }
}

multiclass VOP6M_Conv_Real_gfx13<bits<8> op, string psName> {
  defm _8x4 : VOP6M_Real_Base<GFX13Gen, op, psName#"_1x1_8x4", !tolower(psName)>;
  defm _4x4 : VOP6M_Real_Base<GFX13Gen, op, psName#"_1x1_4x4", !tolower(psName)>;
  defm _4x2 : VOP6M_Real_Base<GFX13Gen, op, psName#"_1x1_4x2", !tolower(psName)>;
}

multiclass Conv_Real_gfx13<bits<8> op> {
  defm _3x3 : VOP5M_Conv_Real_gfx13<op, NAME>;
  defm _1x1 : VOP6M_Conv_Real_gfx13<op, NAME>;
}

defm V_PERMUTE_PAIR_2SRC_ROTATE_GROUP_B32 : VOP2M_Real_gfx13 <0x0>;
defm V_PERMUTE_PAIR_GENSGPR_B32           : VOP2M_Real_gfx13 <0x1>;
defm V_PERMUTE_PAIR_BCAST_B32             : VOP2M_Real_gfx13 <0x2>;
defm V_BPERMUTE_B32                       : VOP2M_Real_gfx13 <0x3>;

// TODO-GFX13: Replace invented opcodes

defm V_MOV_2SRC_B64                       : VOP2M_Real_gfx13 <0x4>;

defm V_PERMUTE_PAIR_2SRC_INTERLEAVE_B64   : VOP3M_Real_gfx13 <0x3E>;
defm V_PERMUTE_PACK_TENSOR_2SRC_B64       : VOP3M_Real_gfx13 <0x3F>;

defm V_WCONV_FP16_FP16 : Conv_Real_gfx13 <0x7e>;

//===----------------------------------------------------------------------===//
// VOPM Patterns
//===----------------------------------------------------------------------===//

multiclass ConvPat_VOPM<string Inst, SDPatternOperator node, string ProfileName> {
  defvar P = !cast<Conv_Profile>(ProfileName);

  def : GCNPat <(P.DstVT !setdagop(P.ConvInPat, node)),
                (P.DstVT !setdagop(P.ConvOutPat, !cast<Instruction>(Inst)))>;
}

multiclass ConvPatMulti_VOPM<string Inst, SDPatternOperator node, string ProfilePrefix>
{
  defm : ConvPat_VOPM<Inst # "_8x4", node, ProfilePrefix # "_8x4">;
  defm : ConvPat_VOPM<Inst # "_4x4", node, ProfilePrefix # "_4x4">;
  defm : ConvPat_VOPM<Inst # "_4x2", node, ProfilePrefix # "_4x2">;
}

defm : ConvPatMulti_VOPM<"V_WCONV_FP16_FP16_1x1", int_amdgcn_wconv_1x1, "VOP5M_CONV_FP16_FP16_1x1">;
defm : ConvPatMulti_VOPM<"V_WCONV_FP16_FP16_3x3", int_amdgcn_wconv_3x3, "VOP5M_CONV_FP16_FP16_3x3">;


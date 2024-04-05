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
  bits<11> src2;
  bits<6> aux_data;

  let Inst{7-0} = vdst{7-0};
  let Inst{11-10} = ?;    // vdst_idx
  let Inst{12} = ?;       // clmp
  let Inst{13} = ?;       // supr

  let Inst{24-17} = op;
  let Inst{31-25} = 0x37; // encoding

  let Inst{40-32} = src0{8-0};
  let Inst{44-43} = ?;    // src0_idx
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

class VOP4Me <bits<8> op> : Enc128 {

  let Inst{9-0} = ?;      // vdst
  let Inst{11-10} = ?;    // vdst_idx
  let Inst{12} = ?;       // clmp
  let Inst{13} = ?;       // supr

  let Inst{23-16} = op;
  let Inst{31-24} = 0xdf; // encoding

  let Inst{42-32} = ?;    // src0
  let Inst{44-43} = ?;    // src0_idx
  let Inst{55-45} = ?;    // src1
  let Inst{57-56} = ?;    // src1_idx
  let Inst{74-64} = ?;    // src2
  let Inst{76-75} = ?;    // src2_idx

  let Inst{78-77} = ?;    // sparse_sel
  let Inst{88-79} = ?;    // aux_data

  let Inst{106-96} = ?;   // src3
  let Inst{108-107} = ?;  // src3_idx
  let Inst{119-109} = ?;  // src4
  let Inst{121-120} = ?;  // src4_idx

  let Inst{127-122} = ?;  // aux_data_cont
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

// TODO-GFX13: Replace invented opcodes

defm V_MOV_2SRC_B64              : VOP2M_Real_gfx13 <0x42>;
defm V_BPERMUTE_B32              : VOP2M_Real_gfx13 <0x43>;
defm V_PERMUTE_PAIR_BCAST_B32    : VOP2M_Real_gfx13 <0x44>;
defm V_PERMUTE_PAIR_GENSGPR_B32  : VOP2M_Real_gfx13 <0x45>;
defm V_PERMUTE_PAIR_2SRC_ROTATE_GROUP_B32 : VOP2M_Real_gfx13 <0x46>;

defm V_PERMUTE_PAIR_2SRC_INTERLEAVE_B64   : VOP3M_Real_gfx13 <0x3E>;
defm V_PERMUTE_PACK_TENSOR_2SRC_B64       : VOP3M_Real_gfx13 <0x3F>;


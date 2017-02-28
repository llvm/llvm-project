//===-- VOP3PInstructions.td - Vector Instruction Defintions --------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// VOP3P Classes
//===----------------------------------------------------------------------===//

class VOP3PInst<string OpName, VOPProfile P, SDPatternOperator node = null_frag> :
  VOP3P_Pseudo<OpName, P,
    !if(P.HasModifiers, getVOP3PModPat<P, node>.ret, getVOP3Pat<P, node>.ret)
>;

// Non-packed instructions that use the VOP3P encoding. i.e. where
// omod/abs are used.
class VOP3_VOP3PInst<string OpName, VOPProfile P, SDPatternOperator node = null_frag> :
  VOP3P_Pseudo<OpName, P,
    !if(P.HasModifiers, getVOP3ModPat<P, node>.ret, getVOP3Pat<P, node>.ret)
>;

let isCommutable = 1 in {
def V_PK_FMA_F16 : VOP3PInst<"v_pk_fma_f16", VOP3_Profile<VOP_V2F16_V2F16_V2F16_V2F16>, fma>;
def V_PK_ADD_F16 : VOP3PInst<"v_pk_add_f16", VOP3_Profile<VOP_V2F16_V2F16_V2F16>, fadd>;
def V_PK_MUL_F16 : VOP3PInst<"v_pk_mul_f16", VOP3_Profile<VOP_V2F16_V2F16_V2F16>, fmul>;
def V_PK_MAX_F16 : VOP3PInst<"v_pk_max_f16", VOP3_Profile<VOP_V2F16_V2F16_V2F16>, fmaxnum>;
def V_PK_MIN_F16 : VOP3PInst<"v_pk_min_f16", VOP3_Profile<VOP_V2F16_V2F16_V2F16>, fminnum>;

def V_PK_ADD_U16 : VOP3PInst<"v_pk_add_u16", VOP3_Profile<VOP_V2I16_V2I16_V2I16>, add>;
def V_PK_ADD_I16 : VOP3PInst<"v_pk_add_i16", VOP3_Profile<VOP_V2I16_V2I16_V2I16>>;
def V_PK_SUB_I16 : VOP3PInst<"v_pk_sub_i16", VOP3_Profile<VOP_V2I16_V2I16_V2I16>, sub>;
def V_PK_MUL_LO_U16 : VOP3PInst<"v_pk_mul_lo_u16", VOP3_Profile<VOP_V2I16_V2I16_V2I16>, mul>;

def V_PK_MIN_I16 : VOP3PInst<"v_pk_min_i16", VOP3_Profile<VOP_V2I16_V2I16_V2I16>, smin>;
def V_PK_MIN_U16 : VOP3PInst<"v_pk_min_u16", VOP3_Profile<VOP_V2I16_V2I16_V2I16>, umin>;
def V_PK_MAX_I16 : VOP3PInst<"v_pk_max_i16", VOP3_Profile<VOP_V2I16_V2I16_V2I16>, smax>;
def V_PK_MAX_U16 : VOP3PInst<"v_pk_max_u16", VOP3_Profile<VOP_V2I16_V2I16_V2I16>, umax>;
}

def V_PK_LSHLREV_B16 : VOP3PInst<"v_pk_lshlrev_b16", VOP3_Profile<VOP_V2I16_V2I16_V2I16>, lshl_rev>;
def V_PK_ASHRREV_I16 : VOP3PInst<"v_pk_ashrrev_i16", VOP3_Profile<VOP_V2I16_V2I16_V2I16>, ashr_rev>;
def V_PK_LSHRREV_B16 : VOP3PInst<"v_pk_lshrrev_b16", VOP3_Profile<VOP_V2I16_V2I16_V2I16>, lshr_rev>;

// XXX - Commutable?
def V_MAD_MIX_F32 : VOP3_VOP3PInst<"v_mad_mix_f32", VOP3_Profile<VOP_F32_F32_F32_F32>>;
def V_MAD_MIXLO_F16 : VOP3_VOP3PInst<"v_mad_mixlo_f16", VOP3_Profile<VOP_F16_F16_F16_F16>>;
def V_MAD_MIXHI_F16 : VOP3_VOP3PInst<"v_mad_mixhi_f16", VOP3_Profile<VOP_F16_F16_F16_F16>>;


multiclass VOP3P_Real_vi<bits<10> op> {
  def _vi : VOP3P_Real<!cast<VOP3P_Pseudo>(NAME), SIEncodingFamily.VI>,
            VOP3Pe <op, !cast<VOP3P_Pseudo>(NAME).Pfl> {
    let AssemblerPredicates = [HasVOP3PInsts];
    let DecoderNamespace = "VI";
  }
}

defm V_PK_MUL_LO_U16 : VOP3P_Real_vi <0x381>;
defm V_PK_ADD_I16 : VOP3P_Real_vi <0x382>;
defm V_PK_SUB_I16 : VOP3P_Real_vi <0x383>;
defm V_PK_LSHLREV_B16 : VOP3P_Real_vi <0x384>;
defm V_PK_LSHRREV_B16 : VOP3P_Real_vi <0x385>;
defm V_PK_ASHRREV_I16 : VOP3P_Real_vi <0x386>;
defm V_PK_MAX_I16 : VOP3P_Real_vi <0x387>;
defm V_PK_MIN_I16 : VOP3P_Real_vi <0x388>;

defm V_PK_ADD_U16 : VOP3P_Real_vi <0x38a>;
defm V_PK_MAX_U16 : VOP3P_Real_vi <0x38c>;
defm V_PK_MIN_U16 : VOP3P_Real_vi <0x38d>;
defm V_PK_FMA_F16 : VOP3P_Real_vi <0x38e>;
defm V_PK_ADD_F16 : VOP3P_Real_vi <0x38f>;
defm V_PK_MUL_F16 : VOP3P_Real_vi <0x390>;
defm V_PK_MIN_F16 : VOP3P_Real_vi <0x391>;
defm V_PK_MAX_F16 : VOP3P_Real_vi <0x392>;

defm V_MAD_MIX_F32 : VOP3P_Real_vi <0x3a0>;
defm V_MAD_MIXLO_F16 : VOP3P_Real_vi <0x3a1>;
defm V_MAD_MIXHI_F16 : VOP3P_Real_vi <0x3a2>;

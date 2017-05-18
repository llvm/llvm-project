//===-- VOP3Instructions.td - Vector Instruction Defintions ---------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// VOP3 Classes
//===----------------------------------------------------------------------===//

class getVOP3ModPat<VOPProfile P, SDPatternOperator node> {
  list<dag> ret3 = [(set P.DstVT:$vdst,
    (node (P.Src0VT (VOP3Mods0 P.Src0VT:$src0, i32:$src0_modifiers, i1:$clamp, i32:$omod)),
          (P.Src1VT (VOP3Mods P.Src1VT:$src1, i32:$src1_modifiers)),
          (P.Src2VT (VOP3Mods P.Src2VT:$src2, i32:$src2_modifiers))))];

  list<dag> ret2 = [(set P.DstVT:$vdst,
    (node (P.Src0VT (VOP3Mods0 P.Src0VT:$src0, i32:$src0_modifiers, i1:$clamp, i32:$omod)),
          (P.Src1VT (VOP3Mods P.Src1VT:$src1, i32:$src1_modifiers))))];

  list<dag> ret1 = [(set P.DstVT:$vdst,
    (node (P.Src0VT (VOP3Mods0 P.Src0VT:$src0, i32:$src0_modifiers, i1:$clamp, i32:$omod))))];

  list<dag> ret = !if(!eq(P.NumSrcArgs, 3), ret3,
                  !if(!eq(P.NumSrcArgs, 2), ret2,
                  ret1));
}

class getVOP3PModPat<VOPProfile P, SDPatternOperator node> {
  list<dag> ret3 = [(set P.DstVT:$vdst,
    (node (P.Src0VT !if(P.HasClamp, (VOP3PMods0 P.Src0VT:$src0, i32:$src0_modifiers, i1:$clamp),
                                    (VOP3PMods P.Src0VT:$src0, i32:$src0_modifiers))),
          (P.Src1VT (VOP3PMods P.Src1VT:$src1, i32:$src1_modifiers)),
          (P.Src2VT (VOP3PMods P.Src2VT:$src2, i32:$src2_modifiers))))];

  list<dag> ret2 = [(set P.DstVT:$vdst,
    (node !if(P.HasClamp, (P.Src0VT (VOP3PMods0 P.Src0VT:$src0, i32:$src0_modifiers, i1:$clamp)),
                          (P.Src0VT (VOP3PMods P.Src0VT:$src0, i32:$src0_modifiers))),
          (P.Src1VT (VOP3PMods P.Src1VT:$src1, i32:$src1_modifiers))))];

  list<dag> ret1 = [(set P.DstVT:$vdst,
    (node (P.Src0VT (VOP3PMods0 P.Src0VT:$src0, i32:$src0_modifiers, i1:$clamp))))];

  list<dag> ret = !if(!eq(P.NumSrcArgs, 3), ret3,
                  !if(!eq(P.NumSrcArgs, 2), ret2,
                  ret1));
}

class getVOP3Pat<VOPProfile P, SDPatternOperator node> {
  list<dag> ret3 = [(set P.DstVT:$vdst, (node P.Src0VT:$src0, P.Src1VT:$src1, P.Src2VT:$src2))];
  list<dag> ret2 = [(set P.DstVT:$vdst, (node P.Src0VT:$src0, P.Src1VT:$src1))];
  list<dag> ret1 = [(set P.DstVT:$vdst, (node P.Src0VT:$src0))];
  list<dag> ret = !if(!eq(P.NumSrcArgs, 3), ret3,
                  !if(!eq(P.NumSrcArgs, 2), ret2,
                  ret1));
}

class VOP3Inst<string OpName, VOPProfile P, SDPatternOperator node = null_frag, bit VOP3Only = 0> :
  VOP3_Pseudo<OpName, P,
    !if(P.HasModifiers, getVOP3ModPat<P, node>.ret, getVOP3Pat<P, node>.ret),
    VOP3Only>;

// Special case for v_div_fmas_{f32|f64}, since it seems to be the
// only VOP instruction that implicitly reads VCC.
let Asm64 = " $vdst, $src0_modifiers, $src1_modifiers, $src2_modifiers$clamp$omod" in {
def VOP_F32_F32_F32_F32_VCC : VOPProfile<[f32, f32, f32, f32]> {
  let Outs64 = (outs DstRC.RegClass:$vdst);
}
def VOP_F64_F64_F64_F64_VCC : VOPProfile<[f64, f64, f64, f64]> {
  let Outs64 = (outs DstRC.RegClass:$vdst);
}
}

class getVOP3VCC<VOPProfile P, SDPatternOperator node> {
  list<dag> ret =
    [(set P.DstVT:$vdst,
      (node (P.Src0VT (VOP3Mods0 P.Src0VT:$src0, i32:$src0_modifiers, i1:$clamp, i32:$omod)),
            (P.Src1VT (VOP3Mods P.Src1VT:$src1, i32:$src1_modifiers)),
            (P.Src2VT (VOP3Mods P.Src2VT:$src2, i32:$src2_modifiers)),
            (i1 VCC)))];
}

class VOP3_Profile<VOPProfile P> : VOPProfile<P.ArgVT> {
  // FIXME: Hack to stop printing _e64
  let Outs64 = (outs DstRC.RegClass:$vdst);
  let Asm64 = " " # P.Asm64;
}

class VOP3b_Profile<ValueType vt> : VOPProfile<[vt, vt, vt, vt]> {
  // v_div_scale_{f32|f64} do not support input modifiers.
  let HasModifiers = 0;
  let Outs64 = (outs DstRC:$vdst, SReg_64:$sdst);
  let Asm64 = " $vdst, $sdst, $src0, $src1, $src2";
}

def VOP3b_F32_I1_F32_F32_F32 : VOP3b_Profile<f32> {
  // FIXME: Hack to stop printing _e64
  let DstRC = RegisterOperand<VGPR_32>;
}

def VOP3b_F64_I1_F64_F64_F64 : VOP3b_Profile<f64> {
  // FIXME: Hack to stop printing _e64
  let DstRC = RegisterOperand<VReg_64>;
}

def VOP3b_I64_I1_I32_I32_I64 : VOPProfile<[i64, i32, i32, i64]> {
  // FIXME: Hack to stop printing _e64
  let DstRC = RegisterOperand<VReg_64>;

  let Outs64 = (outs DstRC:$vdst, SReg_64:$sdst);
  let Asm64 = " $vdst, $sdst, $src0, $src1, $src2";
}

//===----------------------------------------------------------------------===//
// VOP3 Instructions
//===----------------------------------------------------------------------===//

let isCommutable = 1 in {

def V_MAD_LEGACY_F32 : VOP3Inst <"v_mad_legacy_f32", VOP3_Profile<VOP_F32_F32_F32_F32>>;
def V_MAD_F32 : VOP3Inst <"v_mad_f32", VOP3_Profile<VOP_F32_F32_F32_F32>, fmad>;
def V_MAD_I32_I24 : VOP3Inst <"v_mad_i32_i24", VOP3_Profile<VOP_I32_I32_I32_I32>, AMDGPUmad_i24>;
def V_MAD_U32_U24 : VOP3Inst <"v_mad_u32_u24", VOP3_Profile<VOP_I32_I32_I32_I32>, AMDGPUmad_u24>;
def V_FMA_F32 : VOP3Inst <"v_fma_f32", VOP3_Profile<VOP_F32_F32_F32_F32>, fma>;
def V_FMA_F64 : VOP3Inst <"v_fma_f64", VOP3_Profile<VOP_F64_F64_F64_F64>, fma>;
def V_LERP_U8 : VOP3Inst <"v_lerp_u8", VOP3_Profile<VOP_I32_I32_I32_I32>, int_amdgcn_lerp>;

let SchedRW = [WriteDoubleAdd] in {
def V_ADD_F64 : VOP3Inst <"v_add_f64", VOP3_Profile<VOP_F64_F64_F64>, fadd, 1>;
def V_MUL_F64 : VOP3Inst <"v_mul_f64", VOP3_Profile<VOP_F64_F64_F64>, fmul, 1>;
def V_MIN_F64 : VOP3Inst <"v_min_f64", VOP3_Profile<VOP_F64_F64_F64>, fminnum, 1>;
def V_MAX_F64 : VOP3Inst <"v_max_f64", VOP3_Profile<VOP_F64_F64_F64>, fmaxnum, 1>;
} // End SchedRW = [WriteDoubleAdd]

let SchedRW = [WriteQuarterRate32] in {
def V_MUL_LO_U32 : VOP3Inst <"v_mul_lo_u32", VOP3_Profile<VOP_I32_I32_I32>>;
def V_MUL_HI_U32 : VOP3Inst <"v_mul_hi_u32", VOP3_Profile<VOP_I32_I32_I32>, mulhu>;
def V_MUL_LO_I32 : VOP3Inst <"v_mul_lo_i32", VOP3_Profile<VOP_I32_I32_I32>>;
def V_MUL_HI_I32 : VOP3Inst <"v_mul_hi_i32", VOP3_Profile<VOP_I32_I32_I32>, mulhs>;
} // End SchedRW = [WriteQuarterRate32]

let Uses = [VCC, EXEC] in {
// v_div_fmas_f32:
//   result = src0 * src1 + src2
//   if (vcc)
//     result *= 2^32
//
def V_DIV_FMAS_F32 : VOP3_Pseudo <"v_div_fmas_f32", VOP_F32_F32_F32_F32_VCC,
  getVOP3VCC<VOP_F32_F32_F32_F32_VCC, AMDGPUdiv_fmas>.ret> {
  let SchedRW = [WriteFloatFMA];
}
// v_div_fmas_f64:
//   result = src0 * src1 + src2
//   if (vcc)
//     result *= 2^64
//
def V_DIV_FMAS_F64 : VOP3_Pseudo <"v_div_fmas_f64", VOP_F64_F64_F64_F64_VCC,
  getVOP3VCC<VOP_F64_F64_F64_F64_VCC, AMDGPUdiv_fmas>.ret> {
  let SchedRW = [WriteDouble];
}
} // End Uses = [VCC, EXEC]

} // End isCommutable = 1

def V_CUBEID_F32 : VOP3Inst <"v_cubeid_f32", VOP3_Profile<VOP_F32_F32_F32_F32>, int_amdgcn_cubeid>;
def V_CUBESC_F32 : VOP3Inst <"v_cubesc_f32", VOP3_Profile<VOP_F32_F32_F32_F32>, int_amdgcn_cubesc>;
def V_CUBETC_F32 : VOP3Inst <"v_cubetc_f32", VOP3_Profile<VOP_F32_F32_F32_F32>, int_amdgcn_cubetc>;
def V_CUBEMA_F32 : VOP3Inst <"v_cubema_f32", VOP3_Profile<VOP_F32_F32_F32_F32>, int_amdgcn_cubema>;
def V_BFE_U32 : VOP3Inst <"v_bfe_u32", VOP3_Profile<VOP_I32_I32_I32_I32>, AMDGPUbfe_u32>;
def V_BFE_I32 : VOP3Inst <"v_bfe_i32", VOP3_Profile<VOP_I32_I32_I32_I32>, AMDGPUbfe_i32>;
def V_BFI_B32 : VOP3Inst <"v_bfi_b32", VOP3_Profile<VOP_I32_I32_I32_I32>, AMDGPUbfi>;
def V_ALIGNBIT_B32 : VOP3Inst <"v_alignbit_b32", VOP3_Profile<VOP_I32_I32_I32_I32>>;
def V_ALIGNBYTE_B32 : VOP3Inst <"v_alignbyte_b32", VOP3_Profile<VOP_I32_I32_I32_I32>>;
def V_MIN3_F32 : VOP3Inst <"v_min3_f32", VOP3_Profile<VOP_F32_F32_F32_F32>, AMDGPUfmin3>;
def V_MIN3_I32 : VOP3Inst <"v_min3_i32", VOP3_Profile<VOP_I32_I32_I32_I32>, AMDGPUsmin3>;
def V_MIN3_U32 : VOP3Inst <"v_min3_u32", VOP3_Profile<VOP_I32_I32_I32_I32>, AMDGPUumin3>;
def V_MAX3_F32 : VOP3Inst <"v_max3_f32", VOP3_Profile<VOP_F32_F32_F32_F32>, AMDGPUfmax3>;
def V_MAX3_I32 : VOP3Inst <"v_max3_i32", VOP3_Profile<VOP_I32_I32_I32_I32>, AMDGPUsmax3>;
def V_MAX3_U32 : VOP3Inst <"v_max3_u32", VOP3_Profile<VOP_I32_I32_I32_I32>, AMDGPUumax3>;
def V_MED3_F32 : VOP3Inst <"v_med3_f32", VOP3_Profile<VOP_F32_F32_F32_F32>, AMDGPUfmed3>;
def V_MED3_I32 : VOP3Inst <"v_med3_i32", VOP3_Profile<VOP_I32_I32_I32_I32>, AMDGPUsmed3>;
def V_MED3_U32 : VOP3Inst <"v_med3_u32", VOP3_Profile<VOP_I32_I32_I32_I32>, AMDGPUumed3>;
def V_SAD_U8 : VOP3Inst <"v_sad_u8", VOP3_Profile<VOP_I32_I32_I32_I32>, int_amdgcn_sad_u8>;
def V_SAD_HI_U8 : VOP3Inst <"v_sad_hi_u8", VOP3_Profile<VOP_I32_I32_I32_I32>, int_amdgcn_sad_hi_u8>;
def V_SAD_U16 : VOP3Inst <"v_sad_u16", VOP3_Profile<VOP_I32_I32_I32_I32>, int_amdgcn_sad_u16>;
def V_SAD_U32 : VOP3Inst <"v_sad_u32", VOP3_Profile<VOP_I32_I32_I32_I32>>;
def V_CVT_PK_U8_F32 : VOP3Inst<"v_cvt_pk_u8_f32", VOP3_Profile<VOP_I32_F32_I32_I32>, int_amdgcn_cvt_pk_u8_f32>;
def V_DIV_FIXUP_F32 : VOP3Inst <"v_div_fixup_f32", VOP3_Profile<VOP_F32_F32_F32_F32>, AMDGPUdiv_fixup>;

let SchedRW = [WriteDoubleAdd] in {
def V_DIV_FIXUP_F64 : VOP3Inst <"v_div_fixup_f64", VOP3_Profile<VOP_F64_F64_F64_F64>, AMDGPUdiv_fixup>;
def V_LDEXP_F64 : VOP3Inst <"v_ldexp_f64", VOP3_Profile<VOP_F64_F64_I32>, AMDGPUldexp, 1>;
} // End SchedRW = [WriteDoubleAdd]

def V_DIV_SCALE_F32 : VOP3_Pseudo <"v_div_scale_f32", VOP3b_F32_I1_F32_F32_F32, [], 1> {
  let SchedRW = [WriteFloatFMA, WriteSALU];
  let hasExtraSrcRegAllocReq = 1;
  let AsmMatchConverter = "";
}

// Double precision division pre-scale.
def V_DIV_SCALE_F64 : VOP3_Pseudo <"v_div_scale_f64", VOP3b_F64_I1_F64_F64_F64, [], 1> {
  let SchedRW = [WriteDouble, WriteSALU];
  let hasExtraSrcRegAllocReq = 1;
  let AsmMatchConverter = "";
}

def V_MSAD_U8 : VOP3Inst <"v_msad_u8", VOP3_Profile<VOP_I32_I32_I32_I32>, int_amdgcn_msad_u8>;
def V_MQSAD_PK_U16_U8 : VOP3Inst <"v_mqsad_pk_u16_u8", VOP3_Profile<VOP_I64_I64_I32_I64>, int_amdgcn_mqsad_pk_u16_u8>;

def V_TRIG_PREOP_F64 : VOP3Inst <"v_trig_preop_f64", VOP3_Profile<VOP_F64_F64_I32>, AMDGPUtrig_preop> {
  let SchedRW = [WriteDouble];
}

// These instructions only exist on SI and CI
let SubtargetPredicate = isSICI in {
def V_LSHL_B64 : VOP3Inst <"v_lshl_b64", VOP3_Profile<VOP_I64_I64_I32>>;
def V_LSHR_B64 : VOP3Inst <"v_lshr_b64", VOP3_Profile<VOP_I64_I64_I32>>;
def V_ASHR_I64 : VOP3Inst <"v_ashr_i64", VOP3_Profile<VOP_I64_I64_I32>>;
def V_MULLIT_F32 : VOP3Inst <"v_mullit_f32", VOP3_Profile<VOP_F32_F32_F32_F32>>;
} // End SubtargetPredicate = isSICI

let SubtargetPredicate = isVI in {
def V_LSHLREV_B64 : VOP3Inst <"v_lshlrev_b64", VOP3_Profile<VOP_I64_I32_I64>>;
def V_LSHRREV_B64 : VOP3Inst <"v_lshrrev_b64", VOP3_Profile<VOP_I64_I32_I64>>;
def V_ASHRREV_I64 : VOP3Inst <"v_ashrrev_i64", VOP3_Profile<VOP_I64_I32_I64>>;
} // End SubtargetPredicate = isVI


let SubtargetPredicate = isCIVI in {

def V_QSAD_PK_U16_U8 : VOP3Inst <"v_qsad_pk_u16_u8", VOP3_Profile<VOP_I64_I64_I32_I64>, int_amdgcn_qsad_pk_u16_u8>;
def V_MQSAD_U32_U8 : VOP3Inst <"v_mqsad_u32_u8", VOP3_Profile<VOP_V4I32_I64_I32_V4I32>, int_amdgcn_mqsad_u32_u8>;

let isCommutable = 1 in {
def V_MAD_U64_U32 : VOP3Inst <"v_mad_u64_u32", VOP3b_I64_I1_I32_I32_I64>;
def V_MAD_I64_I32 : VOP3Inst <"v_mad_i64_i32", VOP3b_I64_I1_I32_I32_I64>;
} // End isCommutable = 1

} // End SubtargetPredicate = isCIVI


let SubtargetPredicate = isVI in {

let isCommutable = 1 in {

def V_DIV_FIXUP_F16   : VOP3Inst <"v_div_fixup_f16", VOP3_Profile<VOP_F16_F16_F16_F16>, AMDGPUdiv_fixup>;
def V_FMA_F16         : VOP3Inst <"v_fma_f16", VOP3_Profile<VOP_F16_F16_F16_F16>, fma>;
def V_INTERP_P1LL_F16 : VOP3Inst <"v_interp_p1ll_f16", VOP3_Profile<VOP_F32_F32_F16>>;
def V_INTERP_P1LV_F16 : VOP3Inst <"v_interp_p1lv_f16", VOP3_Profile<VOP_F32_F32_F16_F16>>;
def V_INTERP_P2_F16   : VOP3Inst <"v_interp_p2_f16", VOP3_Profile<VOP_F16_F32_F16_F32>>;
def V_MAD_F16         : VOP3Inst <"v_mad_f16", VOP3_Profile<VOP_F16_F16_F16_F16>, fmad>;

def V_MAD_U16 : VOP3Inst <"v_mad_u16", VOP3_Profile<VOP_I16_I16_I16_I16>>;
def V_MAD_I16 : VOP3Inst <"v_mad_i16", VOP3_Profile<VOP_I16_I16_I16_I16>>;

}  // End isCommutable = 1

def V_PERM_B32 : VOP3Inst <"v_perm_b32", VOP3_Profile<VOP_I32_I32_I32_I32>>;

} // End SubtargetPredicate = isVI

let Predicates = [isVI] in {

multiclass Ternary_i16_Pats <SDPatternOperator op1, SDPatternOperator op2,
                             Instruction inst, SDPatternOperator op3> {
def : Pat<
  (op2 (op1 i16:$src0, i16:$src1), i16:$src2),
  (inst i16:$src0, i16:$src1, i16:$src2)
>;

def : Pat<
  (i32 (op3 (op2 (op1 i16:$src0, i16:$src1), i16:$src2))),
  (inst i16:$src0, i16:$src1, i16:$src2)
>;

def : Pat<
  (i64 (op3 (op2 (op1 i16:$src0, i16:$src1), i16:$src2))),
   (REG_SEQUENCE VReg_64,
     (inst i16:$src0, i16:$src1, i16:$src2), sub0,
     (V_MOV_B32_e32 (i32 0)), sub1)
>;
}

defm: Ternary_i16_Pats<mul, add, V_MAD_U16, zext>;
defm: Ternary_i16_Pats<mul, add, V_MAD_I16, sext>;

} // End Predicates = [isVI]

let SubtargetPredicate = isGFX9 in {
def V_PACK_B32_F16 : VOP3Inst <"v_pack_b32_f16", VOP3_Profile<VOP_B32_F16_F16>>;
def V_LSHL_ADD_U32 : VOP3Inst <"v_lshl_add_u32", VOP3_Profile<VOP_I32_I32_I32_I32>>;
def V_ADD_LSHL_U32 : VOP3Inst <"v_add_lshl_u32", VOP3_Profile<VOP_I32_I32_I32_I32>>;
def V_ADD3_U32 : VOP3Inst <"v_add3_u32", VOP3_Profile<VOP_I32_I32_I32_I32>>;
def V_LSHL_OR_B32 : VOP3Inst <"v_lshl_or_b32", VOP3_Profile<VOP_I32_I32_I32_I32>>;
def V_AND_OR_B32 : VOP3Inst <"v_and_or_b32", VOP3_Profile<VOP_I32_I32_I32_I32>>;
def V_OR3_B32 : VOP3Inst <"v_or3_b32", VOP3_Profile<VOP_I32_I32_I32_I32>>;

def V_XAD_U32 : VOP3Inst <"v_xad_u32", VOP3_Profile<VOP_I32_I32_I32_I32>>;

def V_MED3_F16 : VOP3Inst <"v_med3_f16", VOP3_Profile<VOP_F16_F16_F16_F16>, AMDGPUfmed3>;
def V_MED3_I16 : VOP3Inst <"v_med3_i16", VOP3_Profile<VOP_I16_I16_I16_I16>, AMDGPUsmed3>;
def V_MED3_U16 : VOP3Inst <"v_med3_u16", VOP3_Profile<VOP_I16_I16_I16_I16>, AMDGPUumed3>;

def V_MIN3_F16 : VOP3Inst <"v_min3_f16", VOP3_Profile<VOP_F16_F16_F16_F16>, AMDGPUfmin3>;
def V_MIN3_I16 : VOP3Inst <"v_min3_i16", VOP3_Profile<VOP_I16_I16_I16_I16>, AMDGPUsmin3>;
def V_MIN3_U16 : VOP3Inst <"v_min3_u16", VOP3_Profile<VOP_I16_I16_I16_I16>, AMDGPUumin3>;

def V_MAX3_F16 : VOP3Inst <"v_max3_f16", VOP3_Profile<VOP_F16_F16_F16_F16>, AMDGPUfmax3>;
def V_MAX3_I16 : VOP3Inst <"v_max3_i16", VOP3_Profile<VOP_I16_I16_I16_I16>, AMDGPUsmax3>;
def V_MAX3_U16 : VOP3Inst <"v_max3_u16", VOP3_Profile<VOP_I16_I16_I16_I16>, AMDGPUumax3>;
} // End SubtargetPredicate = isGFX9


//===----------------------------------------------------------------------===//
// Target
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// SI
//===----------------------------------------------------------------------===//

let AssemblerPredicates = [isSICI], DecoderNamespace = "SICI" in {

multiclass VOP3_Real_si<bits<9> op> {
  def _si : VOP3_Real<!cast<VOP3_Pseudo>(NAME), SIEncodingFamily.SI>,
            VOP3e_si <op, !cast<VOP3_Pseudo>(NAME).Pfl>;
}

multiclass VOP3be_Real_si<bits<9> op> {
  def _si : VOP3_Real<!cast<VOP3_Pseudo>(NAME), SIEncodingFamily.SI>,
            VOP3be_si <op, !cast<VOP3_Pseudo>(NAME).Pfl>;
}

} // End AssemblerPredicates = [isSICI], DecoderNamespace = "SICI"

defm V_MAD_LEGACY_F32   : VOP3_Real_si <0x140>;
defm V_MAD_F32          : VOP3_Real_si <0x141>;
defm V_MAD_I32_I24      : VOP3_Real_si <0x142>;
defm V_MAD_U32_U24      : VOP3_Real_si <0x143>;
defm V_CUBEID_F32       : VOP3_Real_si <0x144>;
defm V_CUBESC_F32       : VOP3_Real_si <0x145>;
defm V_CUBETC_F32       : VOP3_Real_si <0x146>;
defm V_CUBEMA_F32       : VOP3_Real_si <0x147>;
defm V_BFE_U32          : VOP3_Real_si <0x148>;
defm V_BFE_I32          : VOP3_Real_si <0x149>;
defm V_BFI_B32          : VOP3_Real_si <0x14a>;
defm V_FMA_F32          : VOP3_Real_si <0x14b>;
defm V_FMA_F64          : VOP3_Real_si <0x14c>;
defm V_LERP_U8          : VOP3_Real_si <0x14d>;
defm V_ALIGNBIT_B32     : VOP3_Real_si <0x14e>;
defm V_ALIGNBYTE_B32    : VOP3_Real_si <0x14f>;
defm V_MULLIT_F32       : VOP3_Real_si <0x150>;
defm V_MIN3_F32         : VOP3_Real_si <0x151>;
defm V_MIN3_I32         : VOP3_Real_si <0x152>;
defm V_MIN3_U32         : VOP3_Real_si <0x153>;
defm V_MAX3_F32         : VOP3_Real_si <0x154>;
defm V_MAX3_I32         : VOP3_Real_si <0x155>;
defm V_MAX3_U32         : VOP3_Real_si <0x156>;
defm V_MED3_F32         : VOP3_Real_si <0x157>;
defm V_MED3_I32         : VOP3_Real_si <0x158>;
defm V_MED3_U32         : VOP3_Real_si <0x159>;
defm V_SAD_U8           : VOP3_Real_si <0x15a>;
defm V_SAD_HI_U8        : VOP3_Real_si <0x15b>;
defm V_SAD_U16          : VOP3_Real_si <0x15c>;
defm V_SAD_U32          : VOP3_Real_si <0x15d>;
defm V_CVT_PK_U8_F32    : VOP3_Real_si <0x15e>;
defm V_DIV_FIXUP_F32    : VOP3_Real_si <0x15f>;
defm V_DIV_FIXUP_F64    : VOP3_Real_si <0x160>;
defm V_LSHL_B64         : VOP3_Real_si <0x161>;
defm V_LSHR_B64         : VOP3_Real_si <0x162>;
defm V_ASHR_I64         : VOP3_Real_si <0x163>;
defm V_ADD_F64          : VOP3_Real_si <0x164>;
defm V_MUL_F64          : VOP3_Real_si <0x165>;
defm V_MIN_F64          : VOP3_Real_si <0x166>;
defm V_MAX_F64          : VOP3_Real_si <0x167>;
defm V_LDEXP_F64        : VOP3_Real_si <0x168>;
defm V_MUL_LO_U32       : VOP3_Real_si <0x169>;
defm V_MUL_HI_U32       : VOP3_Real_si <0x16a>;
defm V_MUL_LO_I32       : VOP3_Real_si <0x16b>;
defm V_MUL_HI_I32       : VOP3_Real_si <0x16c>;
defm V_DIV_SCALE_F32    : VOP3be_Real_si <0x16d>;
defm V_DIV_SCALE_F64    : VOP3be_Real_si <0x16e>;
defm V_DIV_FMAS_F32     : VOP3_Real_si <0x16f>;
defm V_DIV_FMAS_F64     : VOP3_Real_si <0x170>;
defm V_MSAD_U8          : VOP3_Real_si <0x171>;
defm V_MQSAD_PK_U16_U8  : VOP3_Real_si <0x173>;
defm V_TRIG_PREOP_F64   : VOP3_Real_si <0x174>;

//===----------------------------------------------------------------------===//
// CI
//===----------------------------------------------------------------------===//

multiclass VOP3_Real_ci<bits<9> op> {
  def _ci : VOP3_Real<!cast<VOP3_Pseudo>(NAME), SIEncodingFamily.SI>,
            VOP3e_si <op, !cast<VOP3_Pseudo>(NAME).Pfl> {
    let AssemblerPredicates = [isCIOnly];
    let DecoderNamespace = "CI";
  }
}

multiclass VOP3be_Real_ci<bits<9> op> {
  def _ci : VOP3_Real<!cast<VOP3_Pseudo>(NAME), SIEncodingFamily.SI>,
            VOP3be_si <op, !cast<VOP3_Pseudo>(NAME).Pfl> {
    let AssemblerPredicates = [isCIOnly];
    let DecoderNamespace = "CI";
  }
}

defm V_QSAD_PK_U16_U8   : VOP3_Real_ci <0x172>;
defm V_MQSAD_U32_U8     : VOP3_Real_ci <0x175>;
defm V_MAD_U64_U32      : VOP3be_Real_ci <0x176>;
defm V_MAD_I64_I32      : VOP3be_Real_ci <0x177>;

//===----------------------------------------------------------------------===//
// VI
//===----------------------------------------------------------------------===//

let AssemblerPredicates = [isVI], DecoderNamespace = "VI" in {

multiclass VOP3_Real_vi<bits<10> op> {
  def _vi : VOP3_Real<!cast<VOP3_Pseudo>(NAME), SIEncodingFamily.VI>,
            VOP3e_vi <op, !cast<VOP3_Pseudo>(NAME).Pfl>;
}

multiclass VOP3be_Real_vi<bits<10> op> {
  def _vi : VOP3_Real<!cast<VOP3_Pseudo>(NAME), SIEncodingFamily.VI>,
            VOP3be_vi <op, !cast<VOP3_Pseudo>(NAME).Pfl>;
}

} // End AssemblerPredicates = [isVI], DecoderNamespace = "VI"

defm V_MAD_U64_U32      : VOP3be_Real_vi <0x1E8>;
defm V_MAD_I64_I32      : VOP3be_Real_vi <0x1E9>;

defm V_MAD_LEGACY_F32   : VOP3_Real_vi <0x1c0>;
defm V_MAD_F32          : VOP3_Real_vi <0x1c1>;
defm V_MAD_I32_I24      : VOP3_Real_vi <0x1c2>;
defm V_MAD_U32_U24      : VOP3_Real_vi <0x1c3>;
defm V_CUBEID_F32       : VOP3_Real_vi <0x1c4>;
defm V_CUBESC_F32       : VOP3_Real_vi <0x1c5>;
defm V_CUBETC_F32       : VOP3_Real_vi <0x1c6>;
defm V_CUBEMA_F32       : VOP3_Real_vi <0x1c7>;
defm V_BFE_U32          : VOP3_Real_vi <0x1c8>;
defm V_BFE_I32          : VOP3_Real_vi <0x1c9>;
defm V_BFI_B32          : VOP3_Real_vi <0x1ca>;
defm V_FMA_F32          : VOP3_Real_vi <0x1cb>;
defm V_FMA_F64          : VOP3_Real_vi <0x1cc>;
defm V_LERP_U8          : VOP3_Real_vi <0x1cd>;
defm V_ALIGNBIT_B32     : VOP3_Real_vi <0x1ce>;
defm V_ALIGNBYTE_B32    : VOP3_Real_vi <0x1cf>;
defm V_MIN3_F32         : VOP3_Real_vi <0x1d0>;
defm V_MIN3_I32         : VOP3_Real_vi <0x1d1>;
defm V_MIN3_U32         : VOP3_Real_vi <0x1d2>;
defm V_MAX3_F32         : VOP3_Real_vi <0x1d3>;
defm V_MAX3_I32         : VOP3_Real_vi <0x1d4>;
defm V_MAX3_U32         : VOP3_Real_vi <0x1d5>;
defm V_MED3_F32         : VOP3_Real_vi <0x1d6>;
defm V_MED3_I32         : VOP3_Real_vi <0x1d7>;
defm V_MED3_U32         : VOP3_Real_vi <0x1d8>;
defm V_SAD_U8           : VOP3_Real_vi <0x1d9>;
defm V_SAD_HI_U8        : VOP3_Real_vi <0x1da>;
defm V_SAD_U16          : VOP3_Real_vi <0x1db>;
defm V_SAD_U32          : VOP3_Real_vi <0x1dc>;
defm V_CVT_PK_U8_F32    : VOP3_Real_vi <0x1dd>;
defm V_DIV_FIXUP_F32    : VOP3_Real_vi <0x1de>;
defm V_DIV_FIXUP_F64    : VOP3_Real_vi <0x1df>;
defm V_DIV_SCALE_F32    : VOP3be_Real_vi <0x1e0>;
defm V_DIV_SCALE_F64    : VOP3be_Real_vi <0x1e1>;
defm V_DIV_FMAS_F32     : VOP3_Real_vi <0x1e2>;
defm V_DIV_FMAS_F64     : VOP3_Real_vi <0x1e3>;
defm V_MSAD_U8          : VOP3_Real_vi <0x1e4>;
defm V_QSAD_PK_U16_U8   : VOP3_Real_vi <0x1e5>;
defm V_MQSAD_PK_U16_U8  : VOP3_Real_vi <0x1e6>;
defm V_MQSAD_U32_U8     : VOP3_Real_vi <0x1e7>;

defm V_MAD_F16          : VOP3_Real_vi <0x1ea>;
defm V_MAD_U16          : VOP3_Real_vi <0x1eb>;
defm V_MAD_I16          : VOP3_Real_vi <0x1ec>;

defm V_PERM_B32         : VOP3_Real_vi <0x1ed>;

defm V_FMA_F16          : VOP3_Real_vi <0x1ee>;
defm V_DIV_FIXUP_F16    : VOP3_Real_vi <0x1ef>;

defm V_INTERP_P1LL_F16  : VOP3_Real_vi <0x274>;
defm V_INTERP_P1LV_F16  : VOP3_Real_vi <0x275>;
defm V_INTERP_P2_F16    : VOP3_Real_vi <0x276>;
defm V_ADD_F64          : VOP3_Real_vi <0x280>;
defm V_MUL_F64          : VOP3_Real_vi <0x281>;
defm V_MIN_F64          : VOP3_Real_vi <0x282>;
defm V_MAX_F64          : VOP3_Real_vi <0x283>;
defm V_LDEXP_F64        : VOP3_Real_vi <0x284>;
defm V_MUL_LO_U32       : VOP3_Real_vi <0x285>;

// removed from VI as identical to V_MUL_LO_U32
let isAsmParserOnly = 1 in {
defm V_MUL_LO_I32       : VOP3_Real_vi <0x285>;
}

defm V_MUL_HI_U32       : VOP3_Real_vi <0x286>;
defm V_MUL_HI_I32       : VOP3_Real_vi <0x287>;

defm V_LSHLREV_B64      : VOP3_Real_vi <0x28f>;
defm V_LSHRREV_B64      : VOP3_Real_vi <0x290>;
defm V_ASHRREV_I64      : VOP3_Real_vi <0x291>;
defm V_TRIG_PREOP_F64   : VOP3_Real_vi <0x292>;

defm V_LSHL_ADD_U32 : VOP3_Real_vi <0x1fd>;
defm V_ADD_LSHL_U32 : VOP3_Real_vi <0x1fe>;
defm V_ADD3_U32 : VOP3_Real_vi <0x1ff>;
defm V_LSHL_OR_B32 : VOP3_Real_vi <0x200>;
defm V_AND_OR_B32 : VOP3_Real_vi <0x201>;
defm V_OR3_B32 : VOP3_Real_vi <0x202>;
defm V_PACK_B32_F16 : VOP3_Real_vi <0x2a0>;

defm V_XAD_U32 : VOP3_Real_vi <0x1f3>;

defm V_MIN3_F16 : VOP3_Real_vi <0x1f4>;
defm V_MIN3_I16 : VOP3_Real_vi <0x1f5>;
defm V_MIN3_U16 : VOP3_Real_vi <0x1f6>;

defm V_MAX3_F16 : VOP3_Real_vi <0x1f7>;
defm V_MAX3_I16 : VOP3_Real_vi <0x1f8>;
defm V_MAX3_U16 : VOP3_Real_vi <0x1f9>;

defm V_MED3_F16 : VOP3_Real_vi <0x1fa>;
defm V_MED3_I16 : VOP3_Real_vi <0x1fb>;
defm V_MED3_U16 : VOP3_Real_vi <0x1fc>;

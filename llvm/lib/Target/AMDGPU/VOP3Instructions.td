//===-- VOP3Instructions.td - Vector Instruction Definitions --------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// Special case for v_div_fmas_{f32|f64}, since it seems to be the
// only VOP instruction that implicitly reads VCC.
let Asm64 = " $vdst, $src0_modifiers, $src1_modifiers, $src2_modifiers$clamp$omod" in {
def VOP_F32_F32_F32_F32_VCC : VOPProfile<[f32, f32, f32, f32]> {
  let Outs64 = (outs DstRC.RegClass:$vdst);
  let HasExtVOP3DPP = 0;
  let HasExtDPP = 0;
}
def VOP_F64_F64_F64_F64_VCC : VOPProfile<[f64, f64, f64, f64]> {
  let Outs64 = (outs DstRC.RegClass:$vdst);
}
}

class VOP3b_Profile<ValueType vt> : VOPProfile<[vt, vt, vt, vt]> {
  let Outs64 = (outs DstRC:$vdst, VOPDstS64orS32:$sdst);
  let Asm64 = "$vdst, $sdst, $src0_modifiers, $src1_modifiers, $src2_modifiers$clamp$omod";
  let IsSingle = 1;
  let HasExtVOP3DPP = 0;
  let HasExtDPP = 0;
}

def VOP3b_F32_I1_F32_F32_F32 : VOP3b_Profile<f32>;
def VOP3b_F64_I1_F64_F64_F64 : VOP3b_Profile<f64>;

def VOP3b_I64_I1_I32_I32_I64 : VOPProfile<[i64, i32, i32, i64]> {
  let HasClamp = 1;

  let IsSingle = 1;
  let Outs64 = (outs DstRC:$vdst, VOPDstS64orS32:$sdst);
  let Asm64 = "$vdst, $sdst, $src0, $src1, $src2$clamp";
}

class V_MUL_PROF<VOPProfile P> : VOP3_Profile<P> {
  let HasExtVOP3DPP = 0;
  let HasExtDPP = 0;
}

def DIV_FIXUP_F32_PROF : VOP3_Profile<VOP_F32_F32_F32_F32> {
  let HasExtVOP3DPP = 0;
  let HasExtDPP = 0;
}

//===----------------------------------------------------------------------===//
// VOP3 INTERP
//===----------------------------------------------------------------------===//

class VOP3Interp<string OpName, VOPProfile P, list<dag> pattern = []> :
                 VOP3_Pseudo<OpName, P, pattern> {
  let AsmMatchConverter = "cvtVOP3Interp";
  let mayRaiseFPException = 0;
}

def VOP3_INTERP : VOPProfile<[f32, f32, i32, untyped]> {
  let Ins64 = (ins Src0Mod:$src0_modifiers, VRegSrc_32:$src0,
                   Attr:$attr, AttrChan:$attrchan,
                   clampmod0:$clamp, omod0:$omod);

  let Asm64 = "$vdst, $src0_modifiers, $attr$attrchan$clamp$omod";
}

def VOP3_INTERP_MOV : VOPProfile<[f32, i32, i32, untyped]> {
  let Ins64 = (ins InterpSlot:$src0,
                   Attr:$attr, AttrChan:$attrchan,
                   clampmod0:$clamp, omod0:$omod);

  let Asm64 = "$vdst, $src0, $attr$attrchan$clamp$omod";

  let HasClamp = 1;
  let HasSrc0Mods = 0;
}

class getInterp16Asm <bit HasSrc2, bit HasOMod> {
  string src2 = !if(HasSrc2, ", $src2_modifiers", "");
  string omod = !if(HasOMod, "$omod", "");
  string ret =
    " $vdst, $src0_modifiers, $attr$attrchan"#src2#"$high$clamp"#omod;
}

class getInterp16Ins <bit HasSrc2, bit HasOMod,
                      Operand Src0Mod, Operand Src2Mod> {
  dag ret = !if(HasSrc2,
                !if(HasOMod,
                    (ins Src0Mod:$src0_modifiers, VRegSrc_32:$src0,
                         Attr:$attr, AttrChan:$attrchan,
                         Src2Mod:$src2_modifiers, VRegSrc_32:$src2,
                         highmod:$high, clampmod0:$clamp, omod0:$omod),
                    (ins Src0Mod:$src0_modifiers, VRegSrc_32:$src0,
                         Attr:$attr, AttrChan:$attrchan,
                         Src2Mod:$src2_modifiers, VRegSrc_32:$src2,
                         highmod:$high, clampmod0:$clamp)
                ),
                (ins Src0Mod:$src0_modifiers, VRegSrc_32:$src0,
                     Attr:$attr, AttrChan:$attrchan,
                     highmod:$high, clampmod0:$clamp, omod0:$omod)
            );
}

class VOP3_INTERP16 <list<ValueType> ArgVT> : VOPProfile<ArgVT> {

  let HasOMod = !ne(DstVT.Value, f16.Value);
  let HasHigh = 1;

  let Outs64 = (outs DstRC.RegClass:$vdst);
  let Ins64 = getInterp16Ins<HasSrc2, HasOMod, Src0Mod, Src2Mod>.ret;
  let Asm64 = getInterp16Asm<HasSrc2, HasOMod>.ret;
}

//===----------------------------------------------------------------------===//
// VOP3 Instructions
//===----------------------------------------------------------------------===//

let isCommutable = 1 in {

let isReMaterializable = 1 in {
let mayRaiseFPException = 0 in {
let SubtargetPredicate = HasMadMacF32Insts in {
defm V_MAD_LEGACY_F32 : VOP3Inst <"v_mad_legacy_f32", VOP3_Profile<VOP_F32_F32_F32_F32>>;
defm V_MAD_F32 : VOP3Inst <"v_mad_f32", VOP3_Profile<VOP_F32_F32_F32_F32>, fmad>;
} // End SubtargetPredicate = HasMadMacInsts

let SubtargetPredicate = HasFmaLegacy32 in
defm V_FMA_LEGACY_F32 : VOP3Inst <"v_fma_legacy_f32",
                                 VOP3_Profile<VOP_F32_F32_F32_F32>,
                                 int_amdgcn_fma_legacy>;
}

defm V_MAD_I32_I24 : VOP3Inst <"v_mad_i32_i24", VOP3_Profile<VOP_I32_I32_I32_I32, VOP3_CLAMP>>;
defm V_MAD_U32_U24 : VOP3Inst <"v_mad_u32_u24", VOP3_Profile<VOP_I32_I32_I32_I32, VOP3_CLAMP>>;
defm V_FMA_F32 : VOP3Inst <"v_fma_f32", VOP3_Profile<VOP_F32_F32_F32_F32>, any_fma>;
defm V_LERP_U8 : VOP3Inst <"v_lerp_u8", VOP3_Profile<VOP_I32_I32_I32_I32>, int_amdgcn_lerp>;

let SchedRW = [WriteDoubleAdd] in {
let FPDPRounding = 1 in {
defm V_FMA_F64 : VOP3Inst <"v_fma_f64", VOP3_Profile<VOP_F64_F64_F64_F64>, any_fma>;
defm V_ADD_F64 : VOP3Inst <"v_add_f64", VOP3_Profile<VOP_F64_F64_F64>, any_fadd>;
defm V_MUL_F64 : VOP3Inst <"v_mul_f64", VOP3_Profile<VOP_F64_F64_F64>, fmul>;
} // End FPDPRounding = 1
defm V_MIN_F64 : VOP3Inst <"v_min_f64", VOP3_Profile<VOP_F64_F64_F64>, fminnum_like>;
defm V_MAX_F64 : VOP3Inst <"v_max_f64", VOP3_Profile<VOP_F64_F64_F64>, fmaxnum_like>;
} // End SchedRW = [WriteDoubleAdd]

let SchedRW = [WriteIntMul] in {
defm V_MUL_LO_U32 : VOP3Inst <"v_mul_lo_u32", V_MUL_PROF<VOP_I32_I32_I32>, DivergentBinFrag<mul>>;
defm V_MUL_HI_U32 : VOP3Inst <"v_mul_hi_u32", V_MUL_PROF<VOP_I32_I32_I32>, mulhu>;
defm V_MUL_LO_I32 : VOP3Inst <"v_mul_lo_i32", V_MUL_PROF<VOP_I32_I32_I32>>;
defm V_MUL_HI_I32 : VOP3Inst <"v_mul_hi_i32", V_MUL_PROF<VOP_I32_I32_I32>, mulhs>;
} // End SchedRW = [WriteIntMul]
} // End isReMaterializable = 1

let Uses = [MODE, VCC, EXEC] in {
// v_div_fmas_f32:
//   result = src0 * src1 + src2
//   if (vcc)
//     result *= 2^32
//
let SchedRW = [WriteFloatFMA] in
defm V_DIV_FMAS_F32 : VOP3Inst_Pseudo_Wrapper <"v_div_fmas_f32", VOP_F32_F32_F32_F32_VCC, []>;
// v_div_fmas_f64:
//   result = src0 * src1 + src2
//   if (vcc)
//     result *= 2^64
//
let SchedRW = [WriteDouble], FPDPRounding = 1 in
defm V_DIV_FMAS_F64 : VOP3Inst_Pseudo_Wrapper  <"v_div_fmas_f64", VOP_F64_F64_F64_F64_VCC, []>;
} // End Uses = [MODE, VCC, EXEC]

} // End isCommutable = 1

let isReMaterializable = 1 in {
let mayRaiseFPException = 0 in {
defm V_CUBEID_F32 : VOP3Inst <"v_cubeid_f32", VOP3_Profile<VOP_F32_F32_F32_F32>, int_amdgcn_cubeid>;
defm V_CUBESC_F32 : VOP3Inst <"v_cubesc_f32", VOP3_Profile<VOP_F32_F32_F32_F32>, int_amdgcn_cubesc>;
defm V_CUBETC_F32 : VOP3Inst <"v_cubetc_f32", VOP3_Profile<VOP_F32_F32_F32_F32>, int_amdgcn_cubetc>;
defm V_CUBEMA_F32 : VOP3Inst <"v_cubema_f32", VOP3_Profile<VOP_F32_F32_F32_F32>, int_amdgcn_cubema>;
} // End mayRaiseFPException

defm V_BFE_U32 : VOP3Inst <"v_bfe_u32", VOP3_Profile<VOP_I32_I32_I32_I32>, AMDGPUbfe_u32>;
defm V_BFE_I32 : VOP3Inst <"v_bfe_i32", VOP3_Profile<VOP_I32_I32_I32_I32>, AMDGPUbfe_i32>;
defm V_BFI_B32 : VOP3Inst <"v_bfi_b32", VOP3_Profile<VOP_I32_I32_I32_I32>, AMDGPUbfi>;
defm V_ALIGNBIT_B32 : VOP3Inst <"v_alignbit_b32", VOP3_Profile<VOP_I32_I32_I32_I32>, fshr>;
defm V_ALIGNBYTE_B32 : VOP3Inst <"v_alignbyte_b32", VOP3_Profile<VOP_I32_I32_I32_I32>, int_amdgcn_alignbyte>;

// XXX - No FPException seems suspect but manual doesn't say it does
let mayRaiseFPException = 0 in {
  let isCommutable = 1 in {
    defm V_MIN3_I32 : VOP3Inst <"v_min3_i32", VOP3_Profile<VOP_I32_I32_I32_I32>, AMDGPUsmin3>;
    defm V_MIN3_U32 : VOP3Inst <"v_min3_u32", VOP3_Profile<VOP_I32_I32_I32_I32>, AMDGPUumin3>;
    defm V_MAX3_I32 : VOP3Inst <"v_max3_i32", VOP3_Profile<VOP_I32_I32_I32_I32>, AMDGPUsmax3>;
    defm V_MAX3_U32 : VOP3Inst <"v_max3_u32", VOP3_Profile<VOP_I32_I32_I32_I32>, AMDGPUumax3>;
    defm V_MED3_I32 : VOP3Inst <"v_med3_i32", VOP3_Profile<VOP_I32_I32_I32_I32>, AMDGPUsmed3>;
    defm V_MED3_U32 : VOP3Inst <"v_med3_u32", VOP3_Profile<VOP_I32_I32_I32_I32>, AMDGPUumed3>;
  } // End isCommutable = 1
  defm V_MIN3_F32 : VOP3Inst <"v_min3_f32", VOP3_Profile<VOP_F32_F32_F32_F32>, AMDGPUfmin3>;
  defm V_MAX3_F32 : VOP3Inst <"v_max3_f32", VOP3_Profile<VOP_F32_F32_F32_F32>, AMDGPUfmax3>;
  defm V_MED3_F32 : VOP3Inst <"v_med3_f32", VOP3_Profile<VOP_F32_F32_F32_F32>, AMDGPUfmed3>;
} // End mayRaiseFPException = 0

let isCommutable = 1 in {
  defm V_SAD_U8 : VOP3Inst <"v_sad_u8", VOP3_Profile<VOP_I32_I32_I32_I32, VOP3_CLAMP>>;
  defm V_SAD_HI_U8 : VOP3Inst <"v_sad_hi_u8", VOP3_Profile<VOP_I32_I32_I32_I32, VOP3_CLAMP>>;
  defm V_SAD_U16 : VOP3Inst <"v_sad_u16", VOP3_Profile<VOP_I32_I32_I32_I32, VOP3_CLAMP>>;
  defm V_SAD_U32 : VOP3Inst <"v_sad_u32", VOP3_Profile<VOP_I32_I32_I32_I32, VOP3_CLAMP>>;
} // End isCommutable = 1
defm V_CVT_PK_U8_F32 : VOP3Inst<"v_cvt_pk_u8_f32", VOP3_Profile<VOP_I32_F32_I32_I32>, int_amdgcn_cvt_pk_u8_f32>;

defm V_DIV_FIXUP_F32 : VOP3Inst <"v_div_fixup_f32", DIV_FIXUP_F32_PROF, AMDGPUdiv_fixup>;

let SchedRW = [WriteDoubleAdd], FPDPRounding = 1 in {
  defm V_DIV_FIXUP_F64 : VOP3Inst <"v_div_fixup_f64", VOP3_Profile<VOP_F64_F64_F64_F64>, AMDGPUdiv_fixup>;
  defm V_LDEXP_F64 : VOP3Inst <"v_ldexp_f64", VOP3_Profile<VOP_F64_F64_I32>, AMDGPUldexp>;
} // End SchedRW = [WriteDoubleAdd], FPDPRounding = 1
} // End isReMaterializable = 1


let mayRaiseFPException = 0 in { // Seems suspicious but manual doesn't say it does.
  let SchedRW = [WriteFloatFMA, WriteSALU] in
  defm V_DIV_SCALE_F32 : VOP3Inst_Pseudo_Wrapper <"v_div_scale_f32", VOP3b_F32_I1_F32_F32_F32> ;

  // Double precision division pre-scale.
  let SchedRW = [WriteDouble, WriteSALU], FPDPRounding = 1 in
  defm V_DIV_SCALE_F64 : VOP3Inst_Pseudo_Wrapper <"v_div_scale_f64", VOP3b_F64_I1_F64_F64_F64>;
} // End mayRaiseFPException = 0

let isReMaterializable = 1 in
defm V_MSAD_U8 : VOP3Inst <"v_msad_u8", VOP3_Profile<VOP_I32_I32_I32_I32, VOP3_CLAMP>>;

let Constraints = "@earlyclobber $vdst" in {
defm V_MQSAD_PK_U16_U8 : VOP3Inst <"v_mqsad_pk_u16_u8", VOP3_Profile<VOP_I64_I64_I32_I64, VOP3_CLAMP>>;
} // End Constraints = "@earlyclobber $vdst"


let isReMaterializable = 1 in {
let SchedRW = [WriteDouble] in {
defm V_TRIG_PREOP_F64 : VOP3Inst <"v_trig_preop_f64", VOP3_Profile<VOP_F64_F64_I32>, int_amdgcn_trig_preop>;
} // End SchedRW = [WriteDouble]

let SchedRW = [Write64Bit] in {
  let SubtargetPredicate = isGFX6GFX7 in {
  defm V_LSHL_B64 : VOP3Inst <"v_lshl_b64", VOP3_Profile<VOP_I64_I64_I32>, cshl_64>;
  defm V_LSHR_B64 : VOP3Inst <"v_lshr_b64", VOP3_Profile<VOP_I64_I64_I32>, csrl_64>;
  defm V_ASHR_I64 : VOP3Inst <"v_ashr_i64", VOP3_Profile<VOP_I64_I64_I32>, csra_64>;
  } // End SubtargetPredicate = isGFX6GFX7

  let SubtargetPredicate = isGFX8Plus in {
  defm V_LSHLREV_B64 : VOP3Inst <"v_lshlrev_b64", VOP3_Profile<VOP_I64_I32_I64>, clshl_rev_64>;
  defm V_LSHRREV_B64 : VOP3Inst <"v_lshrrev_b64", VOP3_Profile<VOP_I64_I32_I64>, clshr_rev_64>;
  defm V_ASHRREV_I64 : VOP3Inst <"v_ashrrev_i64", VOP3_Profile<VOP_I64_I32_I64>, cashr_rev_64>;
  } // End SubtargetPredicate = isGFX8Plus
} // End SchedRW = [Write64Bit]
} // End isReMaterializable = 1

def : GCNPat<
  (i32 (getDivergentFrag<sext>.ret i16:$src)),
  (i32 (V_BFE_I32_e64 $src, (S_MOV_B32 (i32 0)), (S_MOV_B32 (i32 0x10))))
>;

let isReMaterializable = 1 in {
let SubtargetPredicate = isGFX6GFX7GFX10Plus in {
defm V_MULLIT_F32 : VOP3Inst <"v_mullit_f32", VOP3_Profile<VOP_F32_F32_F32_F32>>;
} // End SubtargetPredicate = isGFX6GFX7GFX10Plus

let SchedRW = [Write32Bit] in {
let SubtargetPredicate = isGFX8Plus in {
defm V_PERM_B32 : VOP3Inst <"v_perm_b32", VOP3_Profile<VOP_I32_I32_I32_I32>, AMDGPUperm>;
} // End SubtargetPredicate = isGFX8Plus
} // End SchedRW = [Write32Bit]
} // End isReMaterializable = 1

def VOPProfileMQSAD : VOP3_Profile<VOP_V4I32_I64_I32_V4I32, VOP3_CLAMP> {
  let HasModifiers = 0;
}

let SubtargetPredicate = isGFX7Plus in {
let Constraints = "@earlyclobber $vdst", SchedRW = [WriteQuarterRate32] in {
defm V_QSAD_PK_U16_U8 : VOP3Inst <"v_qsad_pk_u16_u8", VOP3_Profile<VOP_I64_I64_I32_I64, VOP3_CLAMP>>;
defm V_MQSAD_U32_U8 : VOP3Inst <"v_mqsad_u32_u8", VOPProfileMQSAD>;
} // End Constraints = "@earlyclobber $vdst", SchedRW = [WriteQuarterRate32]
} // End SubtargetPredicate = isGFX7Plus

let isCommutable = 1 in {
let SchedRW = [WriteIntMul, WriteSALU] in {
let SubtargetPredicate = isGFX7GFX8GFX9GFX10 in {
defm V_MAD_U64_U32 : VOP3Inst <"v_mad_u64_u32", VOP3b_I64_I1_I32_I32_I64>;
defm V_MAD_I64_I32 : VOP3Inst <"v_mad_i64_i32", VOP3b_I64_I1_I32_I32_I64>;
}
let SubtargetPredicate = isGFX11Only, Constraints = "@earlyclobber $vdst" in {
defm V_MAD_U64_U32_gfx11 : VOP3Inst <"v_mad_u64_u32_gfx11", VOP3b_I64_I1_I32_I32_I64>;
defm V_MAD_I64_I32_gfx11 : VOP3Inst <"v_mad_i64_i32_gfx11", VOP3b_I64_I1_I32_I32_I64>;
} // End SubtargetPredicate = isGFX11Only, Constraints = "@earlyclobber $vdst"
} // End SchedRW = [WriteIntMul, WriteSALU]
} // End isCommutable = 1


let FPDPRounding = 1 in {
  let Predicates = [Has16BitInsts, isGFX8Only] in {
    defm V_DIV_FIXUP_F16 : VOP3Inst <"v_div_fixup_f16", VOP3_Profile<VOP_F16_F16_F16_F16>, AMDGPUdiv_fixup>;
    defm V_FMA_F16 : VOP3Inst <"v_fma_f16", VOP3_Profile<VOP_F16_F16_F16_F16>, any_fma>;
  } // End Predicates = [Has16BitInsts, isGFX8Only]

  let renamedInGFX9 = 1, Predicates = [Has16BitInsts, isGFX9Plus] in {
    defm V_DIV_FIXUP_F16_gfx9 : VOP3Inst <"v_div_fixup_f16_gfx9",
                                          VOP3_Profile<VOP_F16_F16_F16_F16, VOP3_OPSEL>, AMDGPUdiv_fixup>;
    defm V_FMA_F16_gfx9 : VOP3Inst <"v_fma_f16_gfx9", VOP3_Profile<VOP_F16_F16_F16_F16, VOP3_OPSEL>, any_fma>;
  } // End renamedInGFX9 = 1, Predicates = [Has16BitInsts, isGFX9Plus]
} // End FPDPRounding = 1

let SubtargetPredicate = Has16BitInsts, isCommutable = 1 in {

let renamedInGFX9 = 1 in {
  defm V_MAD_U16 : VOP3Inst <"v_mad_u16", VOP3_Profile<VOP_I16_I16_I16_I16, VOP3_CLAMP>>;
  defm V_MAD_I16 : VOP3Inst <"v_mad_i16", VOP3_Profile<VOP_I16_I16_I16_I16, VOP3_CLAMP>>;
  let FPDPRounding = 1 in {
    defm V_MAD_F16 : VOP3Inst <"v_mad_f16", VOP3_Profile<VOP_F16_F16_F16_F16>, fmad>;
    let Uses = [MODE, M0, EXEC] in {
    let OtherPredicates = [isNotGFX90APlus] in
    // For some reason the intrinsic operands are in a different order
    // from the instruction operands.
    def V_INTERP_P2_F16 : VOP3Interp <"v_interp_p2_f16", VOP3_INTERP16<[f16, f32, i32, f32]>,
           [(set f16:$vdst,
             (int_amdgcn_interp_p2_f16 (VOP3Mods f32:$src2, i32:$src2_modifiers),
                                       (VOP3Mods f32:$src0, i32:$src0_modifiers),
                                       (i32 timm:$attrchan),
                                       (i32 timm:$attr),
                                       (i1 timm:$high),
                                       M0))]>;
    } // End Uses = [M0, MODE, EXEC]
  } // End FPDPRounding = 1
} // End renamedInGFX9 = 1

let SubtargetPredicate = isGFX9Only, FPDPRounding = 1 in {
  defm V_MAD_F16_gfx9   : VOP3Inst <"v_mad_f16_gfx9", VOP3_Profile<VOP_F16_F16_F16_F16, VOP3_OPSEL>> ;
} // End SubtargetPredicate = isGFX9Only, FPDPRounding = 1

let SubtargetPredicate = isGFX9Plus in {
defm V_MAD_U16_gfx9   : VOP3Inst <"v_mad_u16_gfx9", VOP3_Profile<VOP_I16_I16_I16_I16, VOP3_OPSEL>>;
defm V_MAD_I16_gfx9   : VOP3Inst <"v_mad_i16_gfx9", VOP3_Profile<VOP_I16_I16_I16_I16, VOP3_OPSEL>>;
let OtherPredicates = [isNotGFX90APlus] in
def V_INTERP_P2_F16_gfx9 : VOP3Interp <"v_interp_p2_f16_gfx9", VOP3_INTERP16<[f16, f32, i32, f32]>>;
} // End SubtargetPredicate = isGFX9Plus

// This predicate should only apply to the selection pattern. The
// instruction still exists and should decode on subtargets with
// other bank counts.
let OtherPredicates = [isNotGFX90APlus, has32BankLDS], Uses = [MODE, M0, EXEC], FPDPRounding = 1 in {
def V_INTERP_P1LL_F16 : VOP3Interp <"v_interp_p1ll_f16", VOP3_INTERP16<[f32, f32, i32, untyped]>,
       [(set f32:$vdst, (int_amdgcn_interp_p1_f16 (VOP3Mods f32:$src0, i32:$src0_modifiers),
                                                  (i32 timm:$attrchan),
                                                  (i32 timm:$attr),
                                                  (i1 timm:$high), M0))]>;
} // End OtherPredicates = [isNotGFX90APlus, has32BankLDS], Uses = [MODE, M0, EXEC], FPDPRounding = 1

let OtherPredicates = [isNotGFX90APlus], Uses = [MODE, M0, EXEC], FPDPRounding = 1 in {
def V_INTERP_P1LV_F16 : VOP3Interp <"v_interp_p1lv_f16", VOP3_INTERP16<[f32, f32, i32, f16]>>;
} // End OtherPredicates = [isNotGFX90APlus], Uses = [MODE, M0, EXEC], FPDPRounding = 1

} // End SubtargetPredicate = Has16BitInsts, isCommutable = 1

def : GCNPat<
  (i64 (getDivergentFrag<sext>.ret i16:$src)),
    (REG_SEQUENCE VReg_64,
      (i32 (V_BFE_I32_e64 $src, (S_MOV_B32 (i32 0)), (S_MOV_B32 (i32 0x10)))), sub0,
      (i32 (COPY_TO_REGCLASS
         (V_ASHRREV_I32_e32 (S_MOV_B32 (i32 0x1f)), (i32 (V_BFE_I32_e64 $src, (S_MOV_B32 (i32 0)), (S_MOV_B32 (i32 0x10))))
      ), VGPR_32)), sub1)
>;

let SubtargetPredicate = isGFX8Plus, Uses = [MODE, M0, EXEC], OtherPredicates = [isNotGFX90APlus] in {
def V_INTERP_P1_F32_e64  : VOP3Interp <"v_interp_p1_f32", VOP3_INTERP>;
def V_INTERP_P2_F32_e64  : VOP3Interp <"v_interp_p2_f32", VOP3_INTERP>;
def V_INTERP_MOV_F32_e64 : VOP3Interp <"v_interp_mov_f32", VOP3_INTERP_MOV>;
} // End SubtargetPredicate = isGFX8Plus, Uses = [MODE, M0, EXEC], OtherPredicates = [isNotGFX90APlus]

let Predicates = [Has16BitInsts, isGFX6GFX7GFX8GFX9] in {

multiclass Ternary_i16_Pats <SDPatternOperator op1, SDPatternOperator op2,
                             Instruction inst> {
def : GCNPat <
  (op2 (op1 i16:$src0, i16:$src1), i16:$src2),
  (inst i16:$src0, i16:$src1, i16:$src2, (i1 0))
>;

}

defm: Ternary_i16_Pats<mul, add, V_MAD_U16_e64>;
defm: Ternary_i16_Pats<mul, add, V_MAD_I16_e64>;

} // End Predicates = [Has16BitInsts, isGFX6GFX7GFX8GFX9]

let Predicates = [Has16BitInsts, isGFX10Plus] in {

multiclass Ternary_i16_Pats_gfx9<SDPatternOperator op1, SDPatternOperator op2,
                                 Instruction inst> {
def : GCNPat <
  (op2 (op1 i16:$src0, i16:$src1), i16:$src2),
  (inst SRCMODS.NONE, $src0, SRCMODS.NONE, $src1, SRCMODS.NONE, $src2, DSTCLAMP.NONE)
>;

}

defm: Ternary_i16_Pats_gfx9<mul, add, V_MAD_U16_gfx9_e64>;
defm: Ternary_i16_Pats_gfx9<mul, add, V_MAD_I16_gfx9_e64>;

} // End Predicates = [Has16BitInsts, isGFX10Plus]

class ThreeOpFragSDAG<SDPatternOperator op1, SDPatternOperator op2> : PatFrag<
  (ops node:$x, node:$y, node:$z),
  // When the inner operation is used multiple times, selecting 3-op
  // instructions may still be beneficial -- if the other users can be
  // combined similarly. Let's be conservative for now.
  (op2 (HasOneUseBinOp<op1> node:$x, node:$y), node:$z),
  [{
    // Only use VALU ops when the result is divergent.
    if (!N->isDivergent())
      return false;

    // Check constant bus limitations.
    //
    // Note: Use !isDivergent as a conservative proxy for whether the value
    //       is in an SGPR (uniform values can end up in VGPRs as well).
    unsigned ConstantBusUses = 0;
    for (unsigned i = 0; i < 3; ++i) {
      if (!Operands[i]->isDivergent() &&
          !isInlineImmediate(Operands[i].getNode())) {
        ConstantBusUses++;
        // This uses AMDGPU::V_ADD3_U32_e64, but all three operand instructions
        // have the same constant bus limit.
        if (ConstantBusUses > Subtarget->getConstantBusLimit(AMDGPU::V_ADD3_U32_e64))
          return false;
      }
    }

    return true;
  }]> {
  let PredicateCodeUsesOperands = 1;
}

class ThreeOpFrag<SDPatternOperator op1, SDPatternOperator op2> : ThreeOpFragSDAG<op1, op2> {
  // The divergence predicate is irrelevant in GlobalISel, as we have
  // proper register bank checks. We just need to verify the constant
  // bus restriction when all the sources are considered.
  //
  // FIXME: With unlucky SGPR operands, we could penalize code by
  // blocking folding SGPR->VGPR copies later.
  // FIXME: There's no register bank verifier
  let GISelPredicateCode = [{
    const int ConstantBusLimit = Subtarget->getConstantBusLimit(AMDGPU::V_ADD3_U32_e64);
    int ConstantBusUses = 0;
    for (unsigned i = 0; i < 3; ++i) {
      const RegisterBank *RegBank = RBI.getRegBank(Operands[i]->getReg(), MRI, TRI);
      if (RegBank->getID() == AMDGPU::SGPRRegBankID) {
        if (++ConstantBusUses > ConstantBusLimit)
          return false;
      }
    }
    return true;
  }];
}

def shl_0_to_4 : PatFrag<
  (ops node:$src0, node:$src1), (shl node:$src0, node:$src1),
  [{
     if (auto *C = dyn_cast<ConstantSDNode>(N->getOperand(1))) {
       return C->getZExtValue() <= 4;
     }
     return false;
   }]> {
  let GISelPredicateCode = [{
    int64_t Imm = 0;
    if (!mi_match(MI.getOperand(2).getReg(), MRI, m_ICst(Imm)) &&
        !mi_match(MI.getOperand(2).getReg(), MRI, m_Copy(m_ICst(Imm))))
      return false;
    return (uint64_t)Imm <= 4;
  }];
}

def VOP3_CVT_PK_F8_F32_Profile : VOP3_Profile<VOP_I32_F32_F32, VOP3_OPSEL> {
  let InsVOP3OpSel = (ins FP32InputMods:$src0_modifiers, Src0RC64:$src0,
                          FP32InputMods:$src1_modifiers, Src1RC64:$src1,
                          VGPR_32:$vdst_in, op_sel0:$op_sel);
  let HasClamp = 0;
  let HasExtVOP3DPP = 0;
}

def VOP3_CVT_SR_F8_F32_Profile : VOP3_Profile<VOPProfile<[i32, f32, i32, f32]>,
                                              VOP3_OPSEL> {
  let InsVOP3OpSel = (ins FP32InputMods:$src0_modifiers, Src0RC64:$src0,
                          FP32InputMods:$src1_modifiers, Src1RC64:$src1,
                          FP32InputMods:$src2_modifiers, VGPR_32:$src2,
                          op_sel0:$op_sel);
  let HasClamp = 0;
  let HasSrc2 = 0;
  let HasSrc2Mods = 1;
  let AsmVOP3OpSel = !subst(", $src2_modifiers", "",
                            getAsmVOP3OpSel<3, HasClamp,
                                            HasSrc0FloatMods, HasSrc1FloatMods,
                                            HasSrc2FloatMods>.ret);
  let HasExtVOP3DPP = 0;
}

let SubtargetPredicate = isGFX9Plus in {
let isCommutable = 1, isReMaterializable = 1 in {
  defm V_ADD3_U32 : VOP3Inst <"v_add3_u32", VOP3_Profile<VOP_I32_I32_I32_I32>>;
  defm V_AND_OR_B32 : VOP3Inst <"v_and_or_b32", VOP3_Profile<VOP_I32_I32_I32_I32>>;
  defm V_OR3_B32 : VOP3Inst <"v_or3_b32", VOP3_Profile<VOP_I32_I32_I32_I32>>;
  defm V_XAD_U32 : VOP3Inst <"v_xad_u32", VOP3_Profile<VOP_I32_I32_I32_I32>>;
  defm V_ADD_I32 : VOP3Inst <"v_add_i32", VOP3_Profile<VOP_I32_I32_I32_ARITH>>;
  defm V_ADD_LSHL_U32 : VOP3Inst <"v_add_lshl_u32", VOP3_Profile<VOP_I32_I32_I32_I32>>;
} // End isCommutable = 1, isReMaterializable = 1
// TODO src0 contains the opsel bit for dst, so if we commute, need to mask and swap this
// to the new src0.
defm V_MED3_F16 : VOP3Inst <"v_med3_f16", VOP3_Profile<VOP_F16_F16_F16_F16, VOP3_OPSEL>, AMDGPUfmed3>;
defm V_MED3_I16 : VOP3Inst <"v_med3_i16", VOP3_Profile<VOP_I16_I16_I16_I16, VOP3_OPSEL>, AMDGPUsmed3>;
defm V_MED3_U16 : VOP3Inst <"v_med3_u16", VOP3_Profile<VOP_I16_I16_I16_I16, VOP3_OPSEL>, AMDGPUumed3>;

defm V_MIN3_F16 : VOP3Inst <"v_min3_f16", VOP3_Profile<VOP_F16_F16_F16_F16, VOP3_OPSEL>, AMDGPUfmin3>;
defm V_MIN3_I16 : VOP3Inst <"v_min3_i16", VOP3_Profile<VOP_I16_I16_I16_I16, VOP3_OPSEL>, AMDGPUsmin3>;
defm V_MIN3_U16 : VOP3Inst <"v_min3_u16", VOP3_Profile<VOP_I16_I16_I16_I16, VOP3_OPSEL>, AMDGPUumin3>;

defm V_MAX3_F16 : VOP3Inst <"v_max3_f16", VOP3_Profile<VOP_F16_F16_F16_F16, VOP3_OPSEL>, AMDGPUfmax3>;
defm V_MAX3_I16 : VOP3Inst <"v_max3_i16", VOP3_Profile<VOP_I16_I16_I16_I16, VOP3_OPSEL>, AMDGPUsmax3>;
defm V_MAX3_U16 : VOP3Inst <"v_max3_u16", VOP3_Profile<VOP_I16_I16_I16_I16, VOP3_OPSEL>, AMDGPUumax3>;

defm V_ADD_I16 : VOP3Inst <"v_add_i16", VOP3_Profile<VOP_I16_I16_I16, VOP3_OPSEL>>;
defm V_SUB_I16 : VOP3Inst <"v_sub_i16", VOP3_Profile<VOP_I16_I16_I16, VOP3_OPSEL>>;

defm V_MAD_U32_U16 : VOP3Inst <"v_mad_u32_u16", VOP3_Profile<VOP_I32_I16_I16_I32, VOP3_OPSEL>>;
defm V_MAD_I32_I16 : VOP3Inst <"v_mad_i32_i16", VOP3_Profile<VOP_I32_I16_I16_I32, VOP3_OPSEL>>;

defm V_CVT_PKNORM_I16_F16 : VOP3Inst <"v_cvt_pknorm_i16_f16", VOP3_Profile<VOP_B32_F16_F16, VOP3_OPSEL>>;
defm V_CVT_PKNORM_U16_F16 : VOP3Inst <"v_cvt_pknorm_u16_f16", VOP3_Profile<VOP_B32_F16_F16, VOP3_OPSEL>>;

defm V_PACK_B32_F16 : VOP3Inst <"v_pack_b32_f16", VOP3_Profile<VOP_B32_F16_F16, VOP3_OPSEL>>;

let isReMaterializable = 1 in {
defm V_SUB_I32 : VOP3Inst <"v_sub_i32", VOP3_Profile<VOP_I32_I32_I32_ARITH>>;
defm V_LSHL_ADD_U32 : VOP3Inst <"v_lshl_add_u32", VOP3_Profile<VOP_I32_I32_I32_I32>>;
defm V_LSHL_OR_B32 : VOP3Inst <"v_lshl_or_b32", VOP3_Profile<VOP_I32_I32_I32_I32>>;
} // End isReMaterializable = 1

// V_LSHL_ADD_U64: D0.u64 = (S0.u64 << S1.u[2:0]) + S2.u64
// src0 is shifted left by 0-4 (use “0” to get ADD_U64).
let SubtargetPredicate = isGFX940Plus in
defm V_LSHL_ADD_U64 : VOP3Inst <"v_lshl_add_u64", VOP3_Profile<VOP_I64_I64_I32_I64>>;

let SubtargetPredicate = HasFP8Insts, mayRaiseFPException = 0,
    SchedRW = [WriteFloatCvt] in {
  let Constraints = "$vdst = $vdst_in", DisableEncoding = "$vdst_in" in {
    defm V_CVT_PK_FP8_F32 : VOP3Inst<"v_cvt_pk_fp8_f32", VOP3_CVT_PK_F8_F32_Profile>;
    defm V_CVT_PK_BF8_F32 : VOP3Inst<"v_cvt_pk_bf8_f32", VOP3_CVT_PK_F8_F32_Profile>;
  }

  // These instructions have non-standard use of op_sel. In particular they are
  // using op_sel bits 2 and 3 while only having two sources. Therefore dummy
  // src2 is used to hold the op_sel value.
  let Constraints = "$vdst = $src2", DisableEncoding = "$src2" in {
    defm V_CVT_SR_FP8_F32 : VOP3Inst<"v_cvt_sr_fp8_f32", VOP3_CVT_SR_F8_F32_Profile>;
    defm V_CVT_SR_BF8_F32 : VOP3Inst<"v_cvt_sr_bf8_f32", VOP3_CVT_SR_F8_F32_Profile>;
  }
}

class Cvt_PK_F8_F32_Pat<SDPatternOperator node, int index, VOP3_Pseudo inst> : GCNPat<
    (i32 (node f32:$src0, f32:$src1, i32:$old, index)),
    (inst !if(index, SRCMODS.DST_OP_SEL, 0), $src0, 0, $src1, $old, !if(index, SRCMODS.OP_SEL_0, 0))
>;

class Cvt_SR_F8_F32_Pat<SDPatternOperator node, bits<2> index, VOP3_Pseudo inst> : GCNPat<
    (i32 (node f32:$src0, i32:$src1, i32:$old, index)),
    (inst !if(index{1}, SRCMODS.DST_OP_SEL, 0), $src0, 0, $src1,
          !if(index{0}, SRCMODS.OP_SEL_0, 0), $old, !if(index{1}, SRCMODS.OP_SEL_0, 0))
>;

foreach Index = [0, -1] in {
  def : Cvt_PK_F8_F32_Pat<int_amdgcn_cvt_pk_fp8_f32, Index, V_CVT_PK_FP8_F32_e64>;
  def : Cvt_PK_F8_F32_Pat<int_amdgcn_cvt_pk_bf8_f32, Index, V_CVT_PK_BF8_F32_e64>;
}

foreach Index = [0, 1, 2, 3] in {
  def : Cvt_SR_F8_F32_Pat<int_amdgcn_cvt_sr_fp8_f32, Index, V_CVT_SR_FP8_F32_e64>;
  def : Cvt_SR_F8_F32_Pat<int_amdgcn_cvt_sr_bf8_f32, Index, V_CVT_SR_BF8_F32_e64>;
}

class ThreeOp_i32_Pats <SDPatternOperator op1, SDPatternOperator op2, Instruction inst> : GCNPat <
  // This matches (op2 (op1 i32:$src0, i32:$src1), i32:$src2) with conditions.
  (ThreeOpFrag<op1, op2> i32:$src0, i32:$src1, i32:$src2),
  (inst VSrc_b32:$src0, VSrc_b32:$src1, VSrc_b32:$src2)
>;

def : ThreeOp_i32_Pats<cshl_32, add, V_LSHL_ADD_U32_e64>;
def : ThreeOp_i32_Pats<add, cshl_32, V_ADD_LSHL_U32_e64>;
def : ThreeOp_i32_Pats<add, add, V_ADD3_U32_e64>;
def : ThreeOp_i32_Pats<cshl_32, or, V_LSHL_OR_B32_e64>;
def : ThreeOp_i32_Pats<and, or, V_AND_OR_B32_e64>;
def : ThreeOp_i32_Pats<or, or, V_OR3_B32_e64>;
def : ThreeOp_i32_Pats<xor, add, V_XAD_U32_e64>;

let SubtargetPredicate = isGFX940Plus in
def : GCNPat<
  (ThreeOpFrag<shl_0_to_4, add> i64:$src0, i32:$src1, i64:$src2),
  (V_LSHL_ADD_U64_e64 VSrc_b64:$src0, VSrc_b32:$src1, VSrc_b64:$src2)
>;

def : VOPBinOpClampPat<saddsat, V_ADD_I32_e64, i32>;
def : VOPBinOpClampPat<ssubsat, V_SUB_I32_e64, i32>;

def : GCNPat<(getDivergentFrag<or>.ret (or_oneuse i64:$src0, i64:$src1), i64:$src2),
             (REG_SEQUENCE VReg_64,
               (V_OR3_B32_e64 (i32 (EXTRACT_SUBREG $src0, sub0)),
                              (i32 (EXTRACT_SUBREG $src1, sub0)),
                              (i32 (EXTRACT_SUBREG $src2, sub0))), sub0,
               (V_OR3_B32_e64 (i32 (EXTRACT_SUBREG $src0, sub1)),
                              (i32 (EXTRACT_SUBREG $src1, sub1)),
                              (i32 (EXTRACT_SUBREG $src2, sub1))), sub1)>;

// FIXME: Probably should hardcode clamp bit in pseudo and avoid this.
class OpSelBinOpClampPat<SDPatternOperator node,
                         Instruction inst> : GCNPat<
 (node (i16 (VOP3OpSel i16:$src0, i32:$src0_modifiers)),
       (i16 (VOP3OpSel i16:$src1, i32:$src1_modifiers))),
  (inst $src0_modifiers, $src0, $src1_modifiers, $src1, DSTCLAMP.ENABLE, 0)
>;

def : OpSelBinOpClampPat<saddsat, V_ADD_I16_e64>;
def : OpSelBinOpClampPat<ssubsat, V_SUB_I16_e64>;
} // End SubtargetPredicate = isGFX9Plus

// FIXME: GlobalISel in general does not handle instructions with 2 results,
// so it cannot use these patterns.
multiclass IMAD32_Pats <VOP3_Pseudo inst> {
  def : GCNPat <
        (ThreeOpFrag<mul, add> i32:$src0, i32:$src1, i32:$src2),
        (EXTRACT_SUBREG (inst $src0, $src1,
                              (REG_SEQUENCE SReg_64, // Use scalar and let it be legalized
                                            $src2, sub0,
                                            (i32 (IMPLICIT_DEF)), sub1),
                                            0 /* clamp */),
                        sub0)
        >;
  // Immediate src2 in the pattern above will not fold because it would be partially
  // undef. Hence define specialized pattern for this case.
  // FIXME: GlobalISel pattern exporter fails to export a pattern like this and asserts,
  // make it SDAG only.
  def : GCNPat <
        (ThreeOpFragSDAG<mul, add> i32:$src0, i32:$src1, (i32 imm:$src2)),
        (EXTRACT_SUBREG (inst $src0, $src1, (i64 (as_i64imm $src2)), 0 /* clamp */), sub0)
        >;
}

let SubtargetPredicate = isGFX9GFX10 in // exclude pre-GFX9 where it was slow
defm : IMAD32_Pats<V_MAD_U64_U32_e64>;
let SubtargetPredicate = isGFX11Only in
defm : IMAD32_Pats<V_MAD_U64_U32_gfx11_e64>;

def VOP3_PERMLANE_Profile : VOP3_Profile<VOPProfile <[i32, i32, i32, i32]>, VOP3_OPSEL> {
  let Src0RC64 = VRegSrc_32;
  let Src1RC64 = SCSrc_b32;
  let Src2RC64 = SCSrc_b32;
  let InsVOP3OpSel = (ins IntOpSelMods:$src0_modifiers, VRegSrc_32:$src0,
                          IntOpSelMods:$src1_modifiers, SCSrc_b32:$src1,
                          IntOpSelMods:$src2_modifiers, SCSrc_b32:$src2,
                          VGPR_32:$vdst_in, op_sel0:$op_sel);
  let HasClamp = 0;
  let HasExtVOP3DPP = 0;
  let HasExtDPP = 0;
}

class PermlanePat<SDPatternOperator permlane,
  Instruction inst> : GCNPat<
  (permlane i32:$vdst_in, i32:$src0, i32:$src1, i32:$src2,
            timm:$fi, timm:$bc),
  (inst (as_i1timm $fi), VGPR_32:$src0, (as_i1timm $bc),
        SCSrc_b32:$src1, 0, SCSrc_b32:$src2, VGPR_32:$vdst_in)
>;

// Permlane intrinsic that has either fetch invalid or bound control
// fields enabled.
class BoundControlOrFetchInvalidPermlane<SDPatternOperator permlane> :
  PatFrag<(ops node:$vdst_in, node:$src0, node:$src1, node:$src2,
               node:$fi, node:$bc),
          (permlane node:$vdst_in, node:$src0, node:
                    $src1, node:$src2, node:$fi, node:$bc)> {
  let PredicateCode = [{ return N->getConstantOperandVal(5) != 0 ||
                                N->getConstantOperandVal(6) != 0; }];
  let GISelPredicateCode = [{
    return MI.getOperand(6).getImm() != 0 ||
           MI.getOperand(7).getImm() != 0;
  }];
}

// Drop the input value if it won't be read.
class PermlaneDiscardVDstIn<SDPatternOperator permlane,
                            Instruction inst> : GCNPat<
  (permlane srcvalue, i32:$src0, i32:$src1, i32:$src2,
            timm:$fi, timm:$bc),
  (inst (as_i1timm $fi), VGPR_32:$src0, (as_i1timm $bc),
        SCSrc_b32:$src1, 0, SCSrc_b32:$src2,
        (IMPLICIT_DEF))
>;


let SubtargetPredicate = isGFX10Plus in {
  let isCommutable = 1, isReMaterializable = 1 in {
    defm V_XOR3_B32 : VOP3Inst <"v_xor3_b32", VOP3_Profile<VOP_I32_I32_I32_I32>>;
  } // End isCommutable = 1, isReMaterializable = 1
  def : ThreeOp_i32_Pats<xor, xor, V_XOR3_B32_e64>;

  let Constraints = "$vdst = $vdst_in", DisableEncoding="$vdst_in" in {
    defm V_PERMLANE16_B32 : VOP3Inst<"v_permlane16_b32", VOP3_PERMLANE_Profile>;
    defm V_PERMLANEX16_B32 : VOP3Inst<"v_permlanex16_b32", VOP3_PERMLANE_Profile>;
  } // End $vdst = $vdst_in, DisableEncoding $vdst_in

  def : PermlanePat<int_amdgcn_permlane16, V_PERMLANE16_B32_e64>;
  def : PermlanePat<int_amdgcn_permlanex16, V_PERMLANEX16_B32_e64>;

  def : PermlaneDiscardVDstIn<
    BoundControlOrFetchInvalidPermlane<int_amdgcn_permlane16>,
    V_PERMLANE16_B32_e64>;
  def : PermlaneDiscardVDstIn<
    BoundControlOrFetchInvalidPermlane<int_amdgcn_permlanex16>,
    V_PERMLANEX16_B32_e64>;

  defm V_ADD_NC_U16 : VOP3Inst <"v_add_nc_u16", VOP3_Profile<VOP_I16_I16_I16, VOP3_OPSEL>, add>;
  defm V_SUB_NC_U16 : VOP3Inst <"v_sub_nc_u16", VOP3_Profile<VOP_I16_I16_I16, VOP3_OPSEL>, sub>;

  def : OpSelBinOpClampPat<uaddsat, V_ADD_NC_U16_e64>;
  def : OpSelBinOpClampPat<usubsat, V_SUB_NC_U16_e64>;

  // Undo sub x, c -> add x, -c canonicalization since c is more likely
  // an inline immediate than -c.
  def : GCNPat<
    (add i16:$src0, (i16 NegSubInlineIntConst16:$src1)),
    (V_SUB_NC_U16_e64 0, VSrc_b16:$src0, 0, NegSubInlineIntConst16:$src1, 0, 0)
  >;

} // End SubtargetPredicate = isGFX10Plus

class DivFmasPat<ValueType vt, Instruction inst, Register CondReg> : GCNPat<
  (AMDGPUdiv_fmas (vt (VOP3Mods vt:$src0, i32:$src0_modifiers)),
                  (vt (VOP3Mods vt:$src1, i32:$src1_modifiers)),
                  (vt (VOP3Mods vt:$src2, i32:$src2_modifiers)),
                  (i1 CondReg)),
  (inst $src0_modifiers, $src0, $src1_modifiers, $src1, $src2_modifiers, $src2)
>;

let WaveSizePredicate = isWave64 in {
def : DivFmasPat<f32, V_DIV_FMAS_F32_e64, VCC>;
def : DivFmasPat<f64, V_DIV_FMAS_F64_e64, VCC>;
}

let WaveSizePredicate = isWave32 in {
def : DivFmasPat<f32, V_DIV_FMAS_F32_e64, VCC_LO>;
def : DivFmasPat<f64, V_DIV_FMAS_F64_e64, VCC_LO>;
}

class VOP3_DOT_Profile<VOPProfile P, VOP3Features Features = VOP3_REGULAR> : VOP3_Profile<P, Features> {
  let HasClamp = 0;
  let HasOMod = 0;
  // Override modifiers for bf16(i16) (same as float modifiers).
  let HasSrc0Mods = 1;
  let HasSrc1Mods = 1;
  let HasSrc2Mods = 1;
  let Src0ModDPP = FPVRegInputMods;
  let Src1ModDPP = FPVRegInputMods;
  let Src2ModVOP3DPP = FPVRegInputMods;
  let InsVOP3OpSel = getInsVOP3OpSel<Src0RC64, Src1RC64, Src2RC64, NumSrcArgs,
                                     HasClamp, HasOMod, FPVRegInputMods,
                                     FPVRegInputMods, FPVRegInputMods>.ret;
  let AsmVOP3OpSel = getAsmVOP3OpSel<NumSrcArgs, HasClamp, 1, 1, 1>.ret;
}

let SubtargetPredicate = isGFX11Plus in {
  defm V_MAXMIN_F32     : VOP3Inst<"v_maxmin_f32", VOP3_Profile<VOP_F32_F32_F32_F32>>;
  defm V_MINMAX_F32     : VOP3Inst<"v_minmax_f32", VOP3_Profile<VOP_F32_F32_F32_F32>>;
  defm V_MAXMIN_F16     : VOP3Inst<"v_maxmin_f16", VOP3_Profile<VOP_F16_F16_F16_F16>>;
  defm V_MINMAX_F16     : VOP3Inst<"v_minmax_f16", VOP3_Profile<VOP_F16_F16_F16_F16>>;
  defm V_MAXMIN_U32     : VOP3Inst<"v_maxmin_u32", VOP3_Profile<VOP_I32_I32_I32_I32>>;
  defm V_MINMAX_U32     : VOP3Inst<"v_minmax_u32", VOP3_Profile<VOP_I32_I32_I32_I32>>;
  defm V_MAXMIN_I32     : VOP3Inst<"v_maxmin_i32", VOP3_Profile<VOP_I32_I32_I32_I32>>;
  defm V_MINMAX_I32     : VOP3Inst<"v_minmax_i32", VOP3_Profile<VOP_I32_I32_I32_I32>>;
  defm V_CVT_PK_I16_F32 : VOP3Inst<"v_cvt_pk_i16_f32", VOP3_Profile<VOP_V2I16_F32_F32>>;
  defm V_CVT_PK_U16_F32 : VOP3Inst<"v_cvt_pk_u16_f32", VOP3_Profile<VOP_V2I16_F32_F32>>;
} // End SubtargetPredicate = isGFX11Plus

let SubtargetPredicate = HasDot8Insts, IsDOT=1 in {
  defm V_DOT2_F16_F16 :   VOP3Inst<"v_dot2_f16_f16",   VOP3_DOT_Profile<VOP_F16_V2F16_V2F16_F16>, int_amdgcn_fdot2_f16_f16>;
  defm V_DOT2_BF16_BF16 : VOP3Inst<"v_dot2_bf16_bf16", VOP3_DOT_Profile<VOP_I16_V2I16_V2I16_I16>, int_amdgcn_fdot2_bf16_bf16>;
}

//===----------------------------------------------------------------------===//
// Integer Clamp Patterns
//===----------------------------------------------------------------------===//

class getClampPat<VOPProfile P, SDPatternOperator node> {
  dag ret3 = (P.DstVT (node P.Src0VT:$src0, P.Src1VT:$src1, P.Src2VT:$src2));
  dag ret2 = (P.DstVT (node P.Src0VT:$src0, P.Src1VT:$src1));
  dag ret1 = (P.DstVT (node P.Src0VT:$src0));
  dag ret = !if(!eq(P.NumSrcArgs, 3), ret3,
            !if(!eq(P.NumSrcArgs, 2), ret2,
            ret1));
}

class getClampRes<VOPProfile P, Instruction inst> {
  dag ret3 = (inst P.Src0VT:$src0, P.Src1VT:$src1, P.Src2VT:$src2, (i1 0));
  dag ret2 = (inst P.Src0VT:$src0, P.Src1VT:$src1, (i1 0));
  dag ret1 = (inst P.Src0VT:$src0, (i1 0));
  dag ret = !if(!eq(P.NumSrcArgs, 3), ret3,
            !if(!eq(P.NumSrcArgs, 2), ret2,
            ret1));
}

class IntClampPat<VOP3InstBase inst, SDPatternOperator node> : GCNPat<
  getClampPat<inst.Pfl, node>.ret,
  getClampRes<inst.Pfl, inst>.ret
>;

def : IntClampPat<V_MAD_I32_I24_e64, AMDGPUmad_i24>;
def : IntClampPat<V_MAD_U32_U24_e64, AMDGPUmad_u24>;

def : IntClampPat<V_SAD_U8_e64, int_amdgcn_sad_u8>;
def : IntClampPat<V_SAD_HI_U8_e64, int_amdgcn_sad_hi_u8>;
def : IntClampPat<V_SAD_U16_e64, int_amdgcn_sad_u16>;

def : IntClampPat<V_MSAD_U8_e64, int_amdgcn_msad_u8>;
def : IntClampPat<V_MQSAD_PK_U16_U8_e64, int_amdgcn_mqsad_pk_u16_u8>;

def : IntClampPat<V_QSAD_PK_U16_U8_e64, int_amdgcn_qsad_pk_u16_u8>;
def : IntClampPat<V_MQSAD_U32_U8_e64, int_amdgcn_mqsad_u32_u8>;

//===----------------------------------------------------------------------===//
// Target-specific instruction encodings.
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// GFX11.
//===----------------------------------------------------------------------===//

defm V_FMA_DX9_ZERO_F32    : VOP3_Real_with_name_gfx11<0x209, "V_FMA_LEGACY_F32", "v_fma_dx9_zero_f32">;
defm V_MAD_I32_I24         : VOP3_Realtriple_gfx11<0x20a>;
defm V_MAD_U32_U24         : VOP3_Realtriple_gfx11<0x20b>;
defm V_CUBEID_F32          : VOP3_Realtriple_gfx11<0x20c>;
defm V_CUBESC_F32          : VOP3_Realtriple_gfx11<0x20d>;
defm V_CUBETC_F32          : VOP3_Realtriple_gfx11<0x20e>;
defm V_CUBEMA_F32          : VOP3_Realtriple_gfx11<0x20f>;
defm V_BFE_U32             : VOP3_Realtriple_gfx11<0x210>;
defm V_BFE_I32             : VOP3_Realtriple_gfx11<0x211>;
defm V_BFI_B32             : VOP3_Realtriple_gfx11<0x212>;
defm V_FMA_F32             : VOP3_Realtriple_gfx11<0x213>;
defm V_FMA_F64             : VOP3_Real_Base_gfx11<0x214>;
defm V_LERP_U8             : VOP3_Realtriple_gfx11<0x215>;
defm V_ALIGNBIT_B32        : VOP3_Realtriple_gfx11<0x216>;
defm V_ALIGNBYTE_B32       : VOP3_Realtriple_gfx11<0x217>;
defm V_MULLIT_F32          : VOP3_Realtriple_gfx11<0x218>;
defm V_MIN3_F32            : VOP3_Realtriple_gfx11<0x219>;
defm V_MIN3_I32            : VOP3_Realtriple_gfx11<0x21a>;
defm V_MIN3_U32            : VOP3_Realtriple_gfx11<0x21b>;
defm V_MAX3_F32            : VOP3_Realtriple_gfx11<0x21c>;
defm V_MAX3_I32            : VOP3_Realtriple_gfx11<0x21d>;
defm V_MAX3_U32            : VOP3_Realtriple_gfx11<0x21e>;
defm V_MED3_F32            : VOP3_Realtriple_gfx11<0x21f>;
defm V_MED3_I32            : VOP3_Realtriple_gfx11<0x220>;
defm V_MED3_U32            : VOP3_Realtriple_gfx11<0x221>;
defm V_SAD_U8              : VOP3_Realtriple_gfx11<0x222>;
defm V_SAD_HI_U8           : VOP3_Realtriple_gfx11<0x223>;
defm V_SAD_U16             : VOP3_Realtriple_gfx11<0x224>;
defm V_SAD_U32             : VOP3_Realtriple_gfx11<0x225>;
defm V_CVT_PK_U8_F32       : VOP3_Realtriple_gfx11<0x226>;
defm V_DIV_FIXUP_F32       : VOP3_Real_Base_gfx11<0x227>;
defm V_DIV_FIXUP_F64       : VOP3_Real_Base_gfx11<0x228>;
defm V_DIV_FMAS_F32        : VOP3_Real_Base_gfx11<0x237>;
defm V_DIV_FMAS_F64        : VOP3_Real_Base_gfx11<0x238>;
defm V_MSAD_U8             : VOP3_Realtriple_gfx11<0x239>;
defm V_QSAD_PK_U16_U8      : VOP3_Real_Base_gfx11<0x23a>;
defm V_MQSAD_PK_U16_U8     : VOP3_Real_Base_gfx11<0x23b>;
defm V_MQSAD_U32_U8        : VOP3_Real_Base_gfx11<0x23d>;
defm V_XOR3_B32            : VOP3_Realtriple_gfx11<0x240>;
defm V_MAD_U16             : VOP3_Realtriple_with_name_gfx11<0x241, "V_MAD_U16_gfx9", "v_mad_u16">;
defm V_PERM_B32            : VOP3_Realtriple_gfx11<0x244>;
defm V_XAD_U32             : VOP3_Realtriple_gfx11<0x245>;
defm V_LSHL_ADD_U32        : VOP3_Realtriple_gfx11<0x246>;
defm V_ADD_LSHL_U32        : VOP3_Realtriple_gfx11<0x247>;
defm V_FMA_F16             : VOP3_Realtriple_with_name_gfx11<0x248, "V_FMA_F16_gfx9", "v_fma_f16">;
defm V_MIN3_F16            : VOP3_Realtriple_gfx11<0x249>;
defm V_MIN3_I16            : VOP3_Realtriple_gfx11<0x24a>;
defm V_MIN3_U16            : VOP3_Realtriple_gfx11<0x24b>;
defm V_MAX3_F16            : VOP3_Realtriple_gfx11<0x24c>;
defm V_MAX3_I16            : VOP3_Realtriple_gfx11<0x24d>;
defm V_MAX3_U16            : VOP3_Realtriple_gfx11<0x24e>;
defm V_MED3_F16            : VOP3_Realtriple_gfx11<0x24f>;
defm V_MED3_I16            : VOP3_Realtriple_gfx11<0x250>;
defm V_MED3_U16            : VOP3_Realtriple_gfx11<0x251>;
defm V_MAD_I16             : VOP3_Realtriple_with_name_gfx11<0x253, "V_MAD_I16_gfx9", "v_mad_i16">;
defm V_DIV_FIXUP_F16       : VOP3_Realtriple_with_name_gfx11<0x254, "V_DIV_FIXUP_F16_gfx9", "v_div_fixup_f16">;
defm V_ADD3_U32            : VOP3_Realtriple_gfx11<0x255>;
defm V_LSHL_OR_B32         : VOP3_Realtriple_gfx11<0x256>;
defm V_AND_OR_B32          : VOP3_Realtriple_gfx11<0x257>;
defm V_OR3_B32             : VOP3_Realtriple_gfx11<0x258>;
defm V_MAD_U32_U16         : VOP3_Realtriple_gfx11<0x259>;
defm V_MAD_I32_I16         : VOP3_Realtriple_gfx11<0x25a>;
defm V_PERMLANE16_B32      : VOP3_Real_Base_gfx11<0x25b>;
defm V_PERMLANEX16_B32     : VOP3_Real_Base_gfx11<0x25c>;
defm V_MAXMIN_F32          : VOP3_Realtriple_gfx11<0x25e>;
defm V_MINMAX_F32          : VOP3_Realtriple_gfx11<0x25f>;
defm V_MAXMIN_F16          : VOP3_Realtriple_gfx11<0x260>;
defm V_MINMAX_F16          : VOP3_Realtriple_gfx11<0x261>;
defm V_MAXMIN_U32          : VOP3_Realtriple_gfx11<0x262>;
defm V_MINMAX_U32          : VOP3_Realtriple_gfx11<0x263>;
defm V_MAXMIN_I32          : VOP3_Realtriple_gfx11<0x264>;
defm V_MINMAX_I32          : VOP3_Realtriple_gfx11<0x265>;
defm V_DOT2_F16_F16        : VOP3Dot_Realtriple_gfx11<0x266>;
defm V_DOT2_BF16_BF16      : VOP3Dot_Realtriple_gfx11<0x267>;
defm V_DIV_SCALE_F32       : VOP3be_Real_gfx11<0x2fc, "V_DIV_SCALE_F32", "v_div_scale_f32">;
defm V_DIV_SCALE_F64       : VOP3be_Real_gfx11<0x2fd, "V_DIV_SCALE_F64", "v_div_scale_f64">;
defm V_MAD_U64_U32_gfx11   : VOP3be_Real_gfx11<0x2fe, "V_MAD_U64_U32_gfx11", "v_mad_u64_u32">;
defm V_MAD_I64_I32_gfx11   : VOP3be_Real_gfx11<0x2ff, "V_MAD_I64_I32_gfx11", "v_mad_i64_i32">;
defm V_ADD_NC_U16          : VOP3Only_Realtriple_gfx11<0x303>;
defm V_SUB_NC_U16          : VOP3Only_Realtriple_gfx11<0x304>;
defm V_MUL_LO_U16          : VOP3Only_Realtriple_gfx11<0x305>;
defm V_CVT_PK_I16_F32      : VOP3_Realtriple_gfx11<0x306>;
defm V_CVT_PK_U16_F32      : VOP3_Realtriple_gfx11<0x307>;
defm V_MAX_U16             : VOP3Only_Realtriple_gfx11<0x309>;
defm V_MAX_I16             : VOP3Only_Realtriple_gfx11<0x30a>;
defm V_MIN_U16             : VOP3Only_Realtriple_gfx11<0x30b>;
defm V_MIN_I16             : VOP3Only_Realtriple_gfx11<0x30c>;
defm V_ADD_NC_I16          : VOP3_Realtriple_with_name_gfx11<0x30d, "V_ADD_I16", "v_add_nc_i16">;
defm V_SUB_NC_I16          : VOP3_Realtriple_with_name_gfx11<0x30e, "V_SUB_I16", "v_sub_nc_i16">;
defm V_PACK_B32_F16        : VOP3_Realtriple_gfx11<0x311>;
defm V_CVT_PK_NORM_I16_F16 : VOP3_Realtriple_with_name_gfx11<0x312, "V_CVT_PKNORM_I16_F16" , "v_cvt_pk_norm_i16_f16" >;
defm V_CVT_PK_NORM_U16_F16 : VOP3_Realtriple_with_name_gfx11<0x313, "V_CVT_PKNORM_U16_F16" , "v_cvt_pk_norm_u16_f16" >;
defm V_SUB_NC_I32          : VOP3_Realtriple_with_name_gfx11<0x325, "V_SUB_I32", "v_sub_nc_i32">;
defm V_ADD_NC_I32          : VOP3_Realtriple_with_name_gfx11<0x326, "V_ADD_I32", "v_add_nc_i32">;
defm V_ADD_F64             : VOP3_Real_Base_gfx11<0x327>;
defm V_MUL_F64             : VOP3_Real_Base_gfx11<0x328>;
defm V_MIN_F64             : VOP3_Real_Base_gfx11<0x329>;
defm V_MAX_F64             : VOP3_Real_Base_gfx11<0x32a>;
defm V_LDEXP_F64           : VOP3_Real_Base_gfx11<0x32b>;
defm V_MUL_LO_U32          : VOP3_Real_Base_gfx11<0x32c>;
defm V_MUL_HI_U32          : VOP3_Real_Base_gfx11<0x32d>;
defm V_MUL_HI_I32          : VOP3_Real_Base_gfx11<0x32e>;
defm V_TRIG_PREOP_F64      : VOP3_Real_Base_gfx11<0x32f>;
defm V_LSHLREV_B16         : VOP3Only_Realtriple_gfx11<0x338>;
defm V_LSHRREV_B16         : VOP3Only_Realtriple_gfx11<0x339>;
defm V_ASHRREV_I16         : VOP3Only_Realtriple_gfx11<0x33a>;
defm V_LSHLREV_B64         : VOP3_Real_Base_gfx11<0x33c>;
defm V_LSHRREV_B64         : VOP3_Real_Base_gfx11<0x33d>;
defm V_ASHRREV_I64         : VOP3_Real_Base_gfx11<0x33e>;
defm V_READLANE_B32        : VOP3_Real_No_Suffix_gfx11<0x360>; // Pseudo in VOP2
let InOperandList = (ins SSrcOrLds_b32:$src0, SCSrc_b32:$src1, VGPR_32:$vdst_in) in {
  defm V_WRITELANE_B32     : VOP3_Real_No_Suffix_gfx11<0x361>; // Pseudo in VOP2
} // End InOperandList = (ins SSrcOrLds_b32:$src0, SCSrc_b32:$src1, VGPR_32:$vdst_in)
defm V_AND_B16             : VOP3Only_Realtriple_gfx11<0x362>;
defm V_OR_B16              : VOP3Only_Realtriple_gfx11<0x363>;
defm V_XOR_B16             : VOP3Only_Realtriple_gfx11<0x364>;

//===----------------------------------------------------------------------===//
// GFX10.
//===----------------------------------------------------------------------===//

let AssemblerPredicate = isGFX10Only, DecoderNamespace = "GFX10" in {
  multiclass VOP3_Real_gfx10<bits<10> op> {
    def _gfx10 :
      VOP3_Real<!cast<VOP_Pseudo>(NAME#"_e64"), SIEncodingFamily.GFX10>,
      VOP3e_gfx10<op, !cast<VOP_Pseudo>(NAME#"_e64").Pfl>;
  }
  multiclass VOP3_Real_No_Suffix_gfx10<bits<10> op> {
    def _gfx10 :
      VOP3_Real<!cast<VOP_Pseudo>(NAME), SIEncodingFamily.GFX10>,
      VOP3e_gfx10<op, !cast<VOP_Pseudo>(NAME).Pfl>;
  }
  multiclass VOP3_Real_gfx10_with_name<bits<10> op, string opName,
                                       string asmName> {
    def _gfx10 :
      VOP3_Real<!cast<VOP3_Pseudo>(opName#"_e64"), SIEncodingFamily.GFX10>,
      VOP3e_gfx10<op, !cast<VOP3_Pseudo>(opName#"_e64").Pfl> {
        VOP3_Pseudo ps = !cast<VOP3_Pseudo>(opName#"_e64");
        let AsmString = asmName # ps.AsmOperands;
        let IsSingle = 1;
      }
  }
  multiclass VOP3be_Real_gfx10<bits<10> op> {
    def _gfx10 :
      VOP3_Real<!cast<VOP3_Pseudo>(NAME#"_e64"), SIEncodingFamily.GFX10>,
      VOP3be_gfx10<op, !cast<VOP3_Pseudo>(NAME#"_e64").Pfl>;
  }
  multiclass VOP3Interp_Real_gfx10<bits<10> op> {
    def _gfx10 :
      VOP3_Real<!cast<VOP3_Pseudo>(NAME), SIEncodingFamily.GFX10>,
      VOP3Interp_gfx10<op, !cast<VOP3_Pseudo>(NAME).Pfl>;
  }
  multiclass VOP3OpSel_Real_gfx10<bits<10> op> {
    def _gfx10 :
      VOP3_Real<!cast<VOP3_Pseudo>(NAME#"_e64"), SIEncodingFamily.GFX10>,
      VOP3OpSel_gfx10<op, !cast<VOP3_Pseudo>(NAME#"_e64").Pfl>;
  }
  multiclass VOP3OpSel_Real_gfx10_with_name<bits<10> op, string opName,
                                            string asmName> {
    def _gfx10 :
      VOP3_Real<!cast<VOP3_Pseudo>(opName#"_e64"), SIEncodingFamily.GFX10>,
      VOP3OpSel_gfx10<op, !cast<VOP3_Pseudo>(opName#"_e64").Pfl> {
        VOP3_Pseudo ps = !cast<VOP3_Pseudo>(opName#"_e64");
        let AsmString = asmName # ps.AsmOperands;
      }
  }
} // End AssemblerPredicate = isGFX10Only, DecoderNamespace = "GFX10"

defm V_READLANE_B32  : VOP3_Real_No_Suffix_gfx10<0x360>;

let InOperandList = (ins SSrcOrLds_b32:$src0, SCSrc_b32:$src1, VGPR_32:$vdst_in) in {
  defm V_WRITELANE_B32 : VOP3_Real_No_Suffix_gfx10<0x361>;
} // End InOperandList = (ins SSrcOrLds_b32:$src0, SCSrc_b32:$src1, VGPR_32:$vdst_in)

let SubtargetPredicate = isGFX10Before1030 in {
  defm V_MUL_LO_I32      : VOP3_Real_gfx10<0x16b>;
}

defm V_XOR3_B32           : VOP3_Real_gfx10<0x178>;
defm V_LSHLREV_B64        : VOP3_Real_gfx10<0x2ff>;
defm V_LSHRREV_B64        : VOP3_Real_gfx10<0x300>;
defm V_ASHRREV_I64        : VOP3_Real_gfx10<0x301>;
defm V_PERM_B32           : VOP3_Real_gfx10<0x344>;
defm V_XAD_U32            : VOP3_Real_gfx10<0x345>;
defm V_LSHL_ADD_U32       : VOP3_Real_gfx10<0x346>;
defm V_ADD_LSHL_U32       : VOP3_Real_gfx10<0x347>;
defm V_ADD3_U32           : VOP3_Real_gfx10<0x36d>;
defm V_LSHL_OR_B32        : VOP3_Real_gfx10<0x36f>;
defm V_AND_OR_B32         : VOP3_Real_gfx10<0x371>;
defm V_OR3_B32            : VOP3_Real_gfx10<0x372>;

// TODO-GFX10: add MC tests for v_add/sub_nc_i16
defm V_ADD_NC_I16 :
  VOP3OpSel_Real_gfx10_with_name<0x30d, "V_ADD_I16", "v_add_nc_i16">;
defm V_SUB_NC_I16 :
  VOP3OpSel_Real_gfx10_with_name<0x30e, "V_SUB_I16", "v_sub_nc_i16">;
defm V_SUB_NC_I32 :
  VOP3_Real_gfx10_with_name<0x376, "V_SUB_I32", "v_sub_nc_i32">;
defm V_ADD_NC_I32 :
  VOP3_Real_gfx10_with_name<0x37f, "V_ADD_I32", "v_add_nc_i32">;

defm V_INTERP_P1_F32_e64  : VOP3Interp_Real_gfx10<0x200>;
defm V_INTERP_P2_F32_e64  : VOP3Interp_Real_gfx10<0x201>;
defm V_INTERP_MOV_F32_e64 : VOP3Interp_Real_gfx10<0x202>;

defm V_INTERP_P1LL_F16    : VOP3Interp_Real_gfx10<0x342>;
defm V_INTERP_P1LV_F16    : VOP3Interp_Real_gfx10<0x343>;
defm V_INTERP_P2_F16      : VOP3Interp_Real_gfx10<0x35a>;

defm V_PACK_B32_F16       : VOP3OpSel_Real_gfx10<0x311>;
defm V_CVT_PKNORM_I16_F16 : VOP3OpSel_Real_gfx10<0x312>;
defm V_CVT_PKNORM_U16_F16 : VOP3OpSel_Real_gfx10<0x313>;

defm V_MIN3_F16           : VOP3OpSel_Real_gfx10<0x351>;
defm V_MIN3_I16           : VOP3OpSel_Real_gfx10<0x352>;
defm V_MIN3_U16           : VOP3OpSel_Real_gfx10<0x353>;
defm V_MAX3_F16           : VOP3OpSel_Real_gfx10<0x354>;
defm V_MAX3_I16           : VOP3OpSel_Real_gfx10<0x355>;
defm V_MAX3_U16           : VOP3OpSel_Real_gfx10<0x356>;
defm V_MED3_F16           : VOP3OpSel_Real_gfx10<0x357>;
defm V_MED3_I16           : VOP3OpSel_Real_gfx10<0x358>;
defm V_MED3_U16           : VOP3OpSel_Real_gfx10<0x359>;
defm V_MAD_U32_U16        : VOP3OpSel_Real_gfx10<0x373>;
defm V_MAD_I32_I16        : VOP3OpSel_Real_gfx10<0x375>;

defm V_MAD_U16 :
  VOP3OpSel_Real_gfx10_with_name<0x340, "V_MAD_U16_gfx9", "v_mad_u16">;
defm V_FMA_F16 :
  VOP3OpSel_Real_gfx10_with_name<0x34b, "V_FMA_F16_gfx9", "v_fma_f16">;
defm V_MAD_I16 :
  VOP3OpSel_Real_gfx10_with_name<0x35e, "V_MAD_I16_gfx9", "v_mad_i16">;
defm V_DIV_FIXUP_F16 :
  VOP3OpSel_Real_gfx10_with_name<0x35f, "V_DIV_FIXUP_F16_gfx9", "v_div_fixup_f16">;

defm V_ADD_NC_U16      : VOP3OpSel_Real_gfx10<0x303>;
defm V_SUB_NC_U16      : VOP3OpSel_Real_gfx10<0x304>;

// FIXME-GFX10-OPSEL: Need to add "selective" opsel support to some of these
// (they do not support SDWA or DPP).
defm V_MUL_LO_U16      : VOP3_Real_gfx10_with_name<0x305, "V_MUL_LO_U16", "v_mul_lo_u16">;
defm V_LSHRREV_B16     : VOP3_Real_gfx10_with_name<0x307, "V_LSHRREV_B16", "v_lshrrev_b16">;
defm V_ASHRREV_I16     : VOP3_Real_gfx10_with_name<0x308, "V_ASHRREV_I16", "v_ashrrev_i16">;
defm V_MAX_U16         : VOP3_Real_gfx10_with_name<0x309, "V_MAX_U16", "v_max_u16">;
defm V_MAX_I16         : VOP3_Real_gfx10_with_name<0x30a, "V_MAX_I16", "v_max_i16">;
defm V_MIN_U16         : VOP3_Real_gfx10_with_name<0x30b, "V_MIN_U16", "v_min_u16">;
defm V_MIN_I16         : VOP3_Real_gfx10_with_name<0x30c, "V_MIN_I16", "v_min_i16">;
defm V_LSHLREV_B16     : VOP3_Real_gfx10_with_name<0x314, "V_LSHLREV_B16", "v_lshlrev_b16">;
defm V_PERMLANE16_B32  : VOP3OpSel_Real_gfx10<0x377>;
defm V_PERMLANEX16_B32 : VOP3OpSel_Real_gfx10<0x378>;

//===----------------------------------------------------------------------===//
// GFX7, GFX10.
//===----------------------------------------------------------------------===//

let AssemblerPredicate = isGFX7Only, DecoderNamespace = "GFX7" in {
  multiclass VOP3_Real_gfx7<bits<10> op> {
    def _gfx7 :
      VOP3_Real<!cast<VOP3_Pseudo>(NAME#"_e64"), SIEncodingFamily.SI>,
      VOP3e_gfx6_gfx7<op{8-0}, !cast<VOP3_Pseudo>(NAME#"_e64").Pfl>;
  }
  multiclass VOP3be_Real_gfx7<bits<10> op> {
    def _gfx7 :
      VOP3_Real<!cast<VOP3_Pseudo>(NAME#"_e64"), SIEncodingFamily.SI>,
      VOP3be_gfx6_gfx7<op{8-0}, !cast<VOP3_Pseudo>(NAME#"_e64").Pfl>;
  }
} // End AssemblerPredicate = isGFX7Only, DecoderNamespace = "GFX7"

multiclass VOP3_Real_gfx7_gfx10<bits<10> op> :
  VOP3_Real_gfx7<op>, VOP3_Real_gfx10<op>;

multiclass VOP3be_Real_gfx7_gfx10<bits<10> op> :
  VOP3be_Real_gfx7<op>, VOP3be_Real_gfx10<op>;

defm V_QSAD_PK_U16_U8   : VOP3_Real_gfx7_gfx10<0x172>;
defm V_MQSAD_U32_U8     : VOP3_Real_gfx7_gfx10<0x175>;
defm V_MAD_U64_U32      : VOP3be_Real_gfx7_gfx10<0x176>;
defm V_MAD_I64_I32      : VOP3be_Real_gfx7_gfx10<0x177>;

//===----------------------------------------------------------------------===//
// GFX6, GFX7, GFX10.
//===----------------------------------------------------------------------===//

let AssemblerPredicate = isGFX6GFX7, DecoderNamespace = "GFX6GFX7" in {
  multiclass VOP3_Real_gfx6_gfx7<bits<10> op> {
    def _gfx6_gfx7 :
      VOP3_Real<!cast<VOP3_Pseudo>(NAME#"_e64"), SIEncodingFamily.SI>,
      VOP3e_gfx6_gfx7<op{8-0}, !cast<VOP3_Pseudo>(NAME#"_e64").Pfl>;
  }
  multiclass VOP3be_Real_gfx6_gfx7<bits<10> op> {
    def _gfx6_gfx7 :
      VOP3_Real<!cast<VOP3_Pseudo>(NAME#"_e64"), SIEncodingFamily.SI>,
      VOP3be_gfx6_gfx7<op{8-0}, !cast<VOP3_Pseudo>(NAME#"_e64").Pfl>;
  }
} // End AssemblerPredicate = isGFX6GFX7, DecoderNamespace = "GFX6GFX7"

multiclass VOP3_Real_gfx6_gfx7_gfx10<bits<10> op> :
  VOP3_Real_gfx6_gfx7<op>, VOP3_Real_gfx10<op>;

multiclass VOP3be_Real_gfx6_gfx7_gfx10<bits<10> op> :
  VOP3be_Real_gfx6_gfx7<op>, VOP3be_Real_gfx10<op>;

defm V_LSHL_B64        : VOP3_Real_gfx6_gfx7<0x161>;
defm V_LSHR_B64        : VOP3_Real_gfx6_gfx7<0x162>;
defm V_ASHR_I64        : VOP3_Real_gfx6_gfx7<0x163>;
defm V_MUL_LO_I32      : VOP3_Real_gfx6_gfx7<0x16b>;

defm V_MAD_LEGACY_F32  : VOP3_Real_gfx6_gfx7_gfx10<0x140>;
defm V_MAD_F32         : VOP3_Real_gfx6_gfx7_gfx10<0x141>;
defm V_MAD_I32_I24     : VOP3_Real_gfx6_gfx7_gfx10<0x142>;
defm V_MAD_U32_U24     : VOP3_Real_gfx6_gfx7_gfx10<0x143>;
defm V_CUBEID_F32      : VOP3_Real_gfx6_gfx7_gfx10<0x144>;
defm V_CUBESC_F32      : VOP3_Real_gfx6_gfx7_gfx10<0x145>;
defm V_CUBETC_F32      : VOP3_Real_gfx6_gfx7_gfx10<0x146>;
defm V_CUBEMA_F32      : VOP3_Real_gfx6_gfx7_gfx10<0x147>;
defm V_BFE_U32         : VOP3_Real_gfx6_gfx7_gfx10<0x148>;
defm V_BFE_I32         : VOP3_Real_gfx6_gfx7_gfx10<0x149>;
defm V_BFI_B32         : VOP3_Real_gfx6_gfx7_gfx10<0x14a>;
defm V_FMA_F32         : VOP3_Real_gfx6_gfx7_gfx10<0x14b>;
defm V_FMA_F64         : VOP3_Real_gfx6_gfx7_gfx10<0x14c>;
defm V_LERP_U8         : VOP3_Real_gfx6_gfx7_gfx10<0x14d>;
defm V_ALIGNBIT_B32    : VOP3_Real_gfx6_gfx7_gfx10<0x14e>;
defm V_ALIGNBYTE_B32   : VOP3_Real_gfx6_gfx7_gfx10<0x14f>;
defm V_MULLIT_F32      : VOP3_Real_gfx6_gfx7_gfx10<0x150>;
defm V_MIN3_F32        : VOP3_Real_gfx6_gfx7_gfx10<0x151>;
defm V_MIN3_I32        : VOP3_Real_gfx6_gfx7_gfx10<0x152>;
defm V_MIN3_U32        : VOP3_Real_gfx6_gfx7_gfx10<0x153>;
defm V_MAX3_F32        : VOP3_Real_gfx6_gfx7_gfx10<0x154>;
defm V_MAX3_I32        : VOP3_Real_gfx6_gfx7_gfx10<0x155>;
defm V_MAX3_U32        : VOP3_Real_gfx6_gfx7_gfx10<0x156>;
defm V_MED3_F32        : VOP3_Real_gfx6_gfx7_gfx10<0x157>;
defm V_MED3_I32        : VOP3_Real_gfx6_gfx7_gfx10<0x158>;
defm V_MED3_U32        : VOP3_Real_gfx6_gfx7_gfx10<0x159>;
defm V_SAD_U8          : VOP3_Real_gfx6_gfx7_gfx10<0x15a>;
defm V_SAD_HI_U8       : VOP3_Real_gfx6_gfx7_gfx10<0x15b>;
defm V_SAD_U16         : VOP3_Real_gfx6_gfx7_gfx10<0x15c>;
defm V_SAD_U32         : VOP3_Real_gfx6_gfx7_gfx10<0x15d>;
defm V_CVT_PK_U8_F32   : VOP3_Real_gfx6_gfx7_gfx10<0x15e>;
defm V_DIV_FIXUP_F32   : VOP3_Real_gfx6_gfx7_gfx10<0x15f>;
defm V_DIV_FIXUP_F64   : VOP3_Real_gfx6_gfx7_gfx10<0x160>;
defm V_ADD_F64         : VOP3_Real_gfx6_gfx7_gfx10<0x164>;
defm V_MUL_F64         : VOP3_Real_gfx6_gfx7_gfx10<0x165>;
defm V_MIN_F64         : VOP3_Real_gfx6_gfx7_gfx10<0x166>;
defm V_MAX_F64         : VOP3_Real_gfx6_gfx7_gfx10<0x167>;
defm V_LDEXP_F64       : VOP3_Real_gfx6_gfx7_gfx10<0x168>;
defm V_MUL_LO_U32      : VOP3_Real_gfx6_gfx7_gfx10<0x169>;
defm V_MUL_HI_U32      : VOP3_Real_gfx6_gfx7_gfx10<0x16a>;
defm V_MUL_HI_I32      : VOP3_Real_gfx6_gfx7_gfx10<0x16c>;
defm V_DIV_FMAS_F32    : VOP3_Real_gfx6_gfx7_gfx10<0x16f>;
defm V_DIV_FMAS_F64    : VOP3_Real_gfx6_gfx7_gfx10<0x170>;
defm V_MSAD_U8         : VOP3_Real_gfx6_gfx7_gfx10<0x171>;
defm V_MQSAD_PK_U16_U8 : VOP3_Real_gfx6_gfx7_gfx10<0x173>;
defm V_TRIG_PREOP_F64  : VOP3_Real_gfx6_gfx7_gfx10<0x174>;
defm V_DIV_SCALE_F32   : VOP3be_Real_gfx6_gfx7_gfx10<0x16d>;
defm V_DIV_SCALE_F64   : VOP3be_Real_gfx6_gfx7_gfx10<0x16e>;

// NB: Same opcode as v_mad_legacy_f32
let DecoderNamespace = "GFX10_B" in
defm V_FMA_LEGACY_F32  : VOP3_Real_gfx10<0x140>;

//===----------------------------------------------------------------------===//
// GFX8, GFX9 (VI).
//===----------------------------------------------------------------------===//

let AssemblerPredicate = isGFX8GFX9, DecoderNamespace = "GFX8" in {

multiclass VOP3_Real_vi<bits<10> op> {
  def _vi : VOP3_Real<!cast<VOP_Pseudo>(NAME#"_e64"), SIEncodingFamily.VI>,
            VOP3e_vi <op, !cast<VOP_Pseudo>(NAME#"_e64").Pfl>;
}
multiclass VOP3_Real_No_Suffix_vi<bits<10> op> {
  def _vi : VOP3_Real<!cast<VOP_Pseudo>(NAME), SIEncodingFamily.VI>,
            VOP3e_vi <op, !cast<VOP_Pseudo>(NAME).Pfl>;
}

multiclass VOP3be_Real_vi<bits<10> op> {
  def _vi : VOP3_Real<!cast<VOP_Pseudo>(NAME#"_e64"), SIEncodingFamily.VI>,
            VOP3be_vi <op, !cast<VOP_Pseudo>(NAME#"_e64").Pfl>;
}

multiclass VOP3OpSel_Real_gfx9<bits<10> op> {
  def _vi : VOP3_Real<!cast<VOP_Pseudo>(NAME#"_e64"), SIEncodingFamily.VI>,
            VOP3OpSel_gfx9 <op, !cast<VOP_Pseudo>(NAME#"_e64").Pfl>;
}

multiclass VOP3OpSel_Real_gfx9_forced_opsel2<bits<10> op> {
  def _vi : VOP3_Real<!cast<VOP_Pseudo>(NAME#"_e64"), SIEncodingFamily.VI>,
            VOP3OpSel_gfx9 <op, !cast<VOP_Pseudo>(NAME#"_e64").Pfl> {
    let Inst{13} = src2_modifiers{2}; // op_sel(2)
  }
}

multiclass VOP3Interp_Real_vi<bits<10> op> {
  def _vi : VOP3_Real<!cast<VOP_Pseudo>(NAME), SIEncodingFamily.VI>,
            VOP3Interp_vi <op, !cast<VOP_Pseudo>(NAME).Pfl>;
}

} // End AssemblerPredicate = isGFX8GFX9, DecoderNamespace = "GFX8"

let AssemblerPredicate = isGFX8Only, DecoderNamespace = "GFX8" in {

multiclass VOP3_F16_Real_vi<bits<10> op> {
  def _vi : VOP3_Real<!cast<VOP3_Pseudo>(NAME#"_e64"), SIEncodingFamily.VI>,
            VOP3e_vi <op, !cast<VOP3_Pseudo>(NAME#"_e64").Pfl>;
}

multiclass VOP3Interp_F16_Real_vi<bits<10> op> {
  def _vi : VOP3_Real<!cast<VOP3_Pseudo>(NAME), SIEncodingFamily.VI>,
            VOP3Interp_vi <op, !cast<VOP3_Pseudo>(NAME).Pfl>;
}

} // End AssemblerPredicate = isGFX8Only, DecoderNamespace = "GFX8"

let AssemblerPredicate = isGFX9Only, DecoderNamespace = "GFX9" in {

multiclass VOP3_F16_Real_gfx9<bits<10> op, string OpName, string AsmName> {
  def _gfx9 : VOP3_Real<!cast<VOP3_Pseudo>(OpName#"_e64"), SIEncodingFamily.GFX9>,
            VOP3e_vi <op, !cast<VOP3_Pseudo>(OpName#"_e64").Pfl> {
              VOP3_Pseudo ps = !cast<VOP3_Pseudo>(OpName#"_e64");
              let AsmString = AsmName # ps.AsmOperands;
            }
}

multiclass VOP3OpSel_F16_Real_gfx9<bits<10> op, string AsmName> {
  def _gfx9 : VOP3_Real<!cast<VOP3_Pseudo>(NAME#"_e64"), SIEncodingFamily.GFX9>,
            VOP3OpSel_gfx9 <op, !cast<VOP3_Pseudo>(NAME#"_e64").Pfl> {
              VOP3_Pseudo ps = !cast<VOP3_Pseudo>(NAME#"_e64");
              let AsmString = AsmName # ps.AsmOperands;
            }
}

multiclass VOP3Interp_F16_Real_gfx9<bits<10> op, string OpName, string AsmName> {
  def _gfx9 : VOP3_Real<!cast<VOP3_Pseudo>(OpName), SIEncodingFamily.GFX9>,
            VOP3Interp_vi <op, !cast<VOP3_Pseudo>(OpName).Pfl> {
              VOP3_Pseudo ps = !cast<VOP3_Pseudo>(OpName);
              let AsmString = AsmName # ps.AsmOperands;
            }
}

multiclass VOP3_Real_gfx9<bits<10> op, string AsmName> {
  def _gfx9 : VOP3_Real<!cast<VOP_Pseudo>(NAME#"_e64"), SIEncodingFamily.GFX9>,
              VOP3e_vi <op, !cast<VOP_Pseudo>(NAME#"_e64").Pfl> {
              VOP_Pseudo ps = !cast<VOP_Pseudo>(NAME#"_e64");
              let AsmString = AsmName # ps.AsmOperands;
            }
}

} // End AssemblerPredicate = isGFX9Only, DecoderNamespace = "GFX9"

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

defm V_PERM_B32         : VOP3_Real_vi <0x1ed>;

defm V_MAD_F16          : VOP3_F16_Real_vi <0x1ea>;
defm V_MAD_U16          : VOP3_F16_Real_vi <0x1eb>;
defm V_MAD_I16          : VOP3_F16_Real_vi <0x1ec>;
defm V_FMA_F16          : VOP3_F16_Real_vi <0x1ee>;
defm V_DIV_FIXUP_F16    : VOP3_F16_Real_vi <0x1ef>;
defm V_INTERP_P2_F16    : VOP3Interp_F16_Real_vi <0x276>;

let FPDPRounding = 1 in {
defm V_MAD_LEGACY_F16       : VOP3_F16_Real_gfx9 <0x1ea, "V_MAD_F16",       "v_mad_legacy_f16">;
defm V_FMA_LEGACY_F16       : VOP3_F16_Real_gfx9 <0x1ee, "V_FMA_F16",       "v_fma_legacy_f16">;
defm V_DIV_FIXUP_LEGACY_F16 : VOP3_F16_Real_gfx9 <0x1ef, "V_DIV_FIXUP_F16", "v_div_fixup_legacy_f16">;
defm V_INTERP_P2_LEGACY_F16 : VOP3Interp_F16_Real_gfx9 <0x276, "V_INTERP_P2_F16", "v_interp_p2_legacy_f16">;
} // End FPDPRounding = 1

defm V_MAD_LEGACY_U16       : VOP3_F16_Real_gfx9 <0x1eb, "V_MAD_U16",       "v_mad_legacy_u16">;
defm V_MAD_LEGACY_I16       : VOP3_F16_Real_gfx9 <0x1ec, "V_MAD_I16",       "v_mad_legacy_i16">;

defm V_MAD_F16_gfx9         : VOP3OpSel_F16_Real_gfx9 <0x203, "v_mad_f16">;
defm V_MAD_U16_gfx9         : VOP3OpSel_F16_Real_gfx9 <0x204, "v_mad_u16">;
defm V_MAD_I16_gfx9         : VOP3OpSel_F16_Real_gfx9 <0x205, "v_mad_i16">;
defm V_FMA_F16_gfx9         : VOP3OpSel_F16_Real_gfx9 <0x206, "v_fma_f16">;
defm V_DIV_FIXUP_F16_gfx9   : VOP3OpSel_F16_Real_gfx9 <0x207, "v_div_fixup_f16">;
defm V_INTERP_P2_F16_gfx9   : VOP3Interp_F16_Real_gfx9 <0x277, "V_INTERP_P2_F16_gfx9", "v_interp_p2_f16">;

defm V_ADD_I32         : VOP3_Real_vi <0x29c>;
defm V_SUB_I32         : VOP3_Real_vi <0x29d>;

defm V_INTERP_P1_F32_e64  : VOP3Interp_Real_vi <0x270>;
defm V_INTERP_P2_F32_e64  : VOP3Interp_Real_vi <0x271>;
defm V_INTERP_MOV_F32_e64 : VOP3Interp_Real_vi <0x272>;

defm V_INTERP_P1LL_F16  : VOP3Interp_Real_vi <0x274>;
defm V_INTERP_P1LV_F16  : VOP3Interp_Real_vi <0x275>;
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

defm V_READLANE_B32     : VOP3_Real_No_Suffix_vi <0x289>;
defm V_WRITELANE_B32    : VOP3_Real_No_Suffix_vi <0x28a>;

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
defm V_PACK_B32_F16 : VOP3OpSel_Real_gfx9 <0x2a0>;

defm V_XAD_U32 : VOP3_Real_vi <0x1f3>;

defm V_MIN3_F16 : VOP3OpSel_Real_gfx9 <0x1f4>;
defm V_MIN3_I16 : VOP3OpSel_Real_gfx9 <0x1f5>;
defm V_MIN3_U16 : VOP3OpSel_Real_gfx9 <0x1f6>;

defm V_MAX3_F16 : VOP3OpSel_Real_gfx9 <0x1f7>;
defm V_MAX3_I16 : VOP3OpSel_Real_gfx9 <0x1f8>;
defm V_MAX3_U16 : VOP3OpSel_Real_gfx9 <0x1f9>;

defm V_MED3_F16 : VOP3OpSel_Real_gfx9 <0x1fa>;
defm V_MED3_I16 : VOP3OpSel_Real_gfx9 <0x1fb>;
defm V_MED3_U16 : VOP3OpSel_Real_gfx9 <0x1fc>;

defm V_ADD_I16  : VOP3OpSel_Real_gfx9 <0x29e>;
defm V_SUB_I16  : VOP3OpSel_Real_gfx9 <0x29f>;

defm V_MAD_U32_U16 : VOP3OpSel_Real_gfx9 <0x1f1>;
defm V_MAD_I32_I16 : VOP3OpSel_Real_gfx9 <0x1f2>;

defm V_CVT_PKNORM_I16_F16 : VOP3OpSel_Real_gfx9 <0x299>;
defm V_CVT_PKNORM_U16_F16 : VOP3OpSel_Real_gfx9 <0x29a>;

defm V_LSHL_ADD_U64 : VOP3_Real_vi <0x208>;

let OtherPredicates = [HasFP8Insts] in {
defm V_CVT_PK_FP8_F32 : VOP3OpSel_Real_gfx9 <0x2a2>;
defm V_CVT_PK_BF8_F32 : VOP3OpSel_Real_gfx9 <0x2a3>;
defm V_CVT_SR_FP8_F32 : VOP3OpSel_Real_gfx9_forced_opsel2 <0x2a4>;
defm V_CVT_SR_BF8_F32 : VOP3OpSel_Real_gfx9_forced_opsel2 <0x2a5>;
}

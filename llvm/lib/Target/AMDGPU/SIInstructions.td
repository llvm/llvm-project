//===-- SIInstructions.td - SI Instruction Defintions ---------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
// This file was originally auto-generated from a GPU register header file and
// all the instruction definitions were originally commented out.  Instructions
// that are not yet supported remain commented out.
//===----------------------------------------------------------------------===//

def isGCN : Predicate<"Subtarget->getGeneration() "
                      ">= SISubtarget::SOUTHERN_ISLANDS">,
            AssemblerPredicate<"FeatureGCN">;
def isSI : Predicate<"Subtarget->getGeneration() "
                      "== SISubtarget::SOUTHERN_ISLANDS">,
           AssemblerPredicate<"FeatureSouthernIslands">;

def has16BankLDS : Predicate<"Subtarget->getLDSBankCount() == 16">;
def has32BankLDS : Predicate<"Subtarget->getLDSBankCount() == 32">;

include "VOPInstructions.td"
include "SOPInstructions.td"
include "SMInstructions.td"
include "FLATInstructions.td"
include "BUFInstructions.td"

let SubtargetPredicate = isGCN in {

//===----------------------------------------------------------------------===//
// EXP Instructions
//===----------------------------------------------------------------------===//

defm EXP : EXP_m;

//===----------------------------------------------------------------------===//
// VOP1 Instructions
//===----------------------------------------------------------------------===//

let vdst = 0, src0 = 0, VOPAsmPrefer32Bit = 1 in {
defm V_NOP : VOP1Inst <vop1<0x0>, "v_nop", VOP_NONE>;
}

let isMoveImm = 1, isReMaterializable = 1, isAsCheapAsAMove = 1 in {
defm V_MOV_B32 : VOP1Inst <vop1<0x1>, "v_mov_b32", VOP_I32_I32>;
} // End isMoveImm = 1

let Uses = [EXEC] in {

// FIXME: Specify SchedRW for READFIRSTLANE_B32

def V_READFIRSTLANE_B32 : VOP1 <
  0x00000002,
  (outs SReg_32:$vdst),
  (ins VGPR_32:$src0),
  "v_readfirstlane_b32 $vdst, $src0",
  [(set i32:$vdst, (int_amdgcn_readfirstlane i32:$src0))]
> {
  let isConvergent = 1;
}

}

let SchedRW = [WriteQuarterRate32] in {

defm V_CVT_I32_F64 : VOP1Inst <vop1<0x3>, "v_cvt_i32_f64",
  VOP_I32_F64, fp_to_sint
>;
defm V_CVT_F64_I32 : VOP1Inst <vop1<0x4>, "v_cvt_f64_i32",
  VOP_F64_I32, sint_to_fp
>;
defm V_CVT_F32_I32 : VOP1Inst <vop1<0x5>, "v_cvt_f32_i32",
  VOP_F32_I32, sint_to_fp
>;
defm V_CVT_F32_U32 : VOP1Inst <vop1<0x6>, "v_cvt_f32_u32",
  VOP_F32_I32, uint_to_fp
>;
defm V_CVT_U32_F32 : VOP1Inst <vop1<0x7>, "v_cvt_u32_f32",
  VOP_I32_F32, fp_to_uint
>;
defm V_CVT_I32_F32 : VOP1Inst <vop1<0x8>, "v_cvt_i32_f32",
  VOP_I32_F32, fp_to_sint
>;
defm V_CVT_F16_F32 : VOP1Inst <vop1<0xa>, "v_cvt_f16_f32",
  VOP_I32_F32, fp_to_f16
>;
defm V_CVT_F32_F16 : VOP1Inst <vop1<0xb>, "v_cvt_f32_f16",
  VOP_F32_I32, f16_to_fp
>;
defm V_CVT_RPI_I32_F32 : VOP1Inst <vop1<0xc>, "v_cvt_rpi_i32_f32",
  VOP_I32_F32, cvt_rpi_i32_f32>;
defm V_CVT_FLR_I32_F32 : VOP1Inst <vop1<0xd>, "v_cvt_flr_i32_f32",
  VOP_I32_F32, cvt_flr_i32_f32>;
defm V_CVT_OFF_F32_I4 : VOP1Inst  <vop1<0x0e>, "v_cvt_off_f32_i4", VOP_F32_I32>;
defm V_CVT_F32_F64 : VOP1Inst <vop1<0xf>, "v_cvt_f32_f64",
  VOP_F32_F64, fpround
>;
defm V_CVT_F64_F32 : VOP1Inst <vop1<0x10>, "v_cvt_f64_f32",
  VOP_F64_F32, fpextend
>;
defm V_CVT_F32_UBYTE0 : VOP1Inst <vop1<0x11>, "v_cvt_f32_ubyte0",
  VOP_F32_I32, AMDGPUcvt_f32_ubyte0
>;
defm V_CVT_F32_UBYTE1 : VOP1Inst <vop1<0x12>, "v_cvt_f32_ubyte1",
  VOP_F32_I32, AMDGPUcvt_f32_ubyte1
>;
defm V_CVT_F32_UBYTE2 : VOP1Inst <vop1<0x13>, "v_cvt_f32_ubyte2",
  VOP_F32_I32, AMDGPUcvt_f32_ubyte2
>;
defm V_CVT_F32_UBYTE3 : VOP1Inst <vop1<0x14>, "v_cvt_f32_ubyte3",
  VOP_F32_I32, AMDGPUcvt_f32_ubyte3
>;
defm V_CVT_U32_F64 : VOP1Inst <vop1<0x15>, "v_cvt_u32_f64",
  VOP_I32_F64, fp_to_uint
>;
defm V_CVT_F64_U32 : VOP1Inst <vop1<0x16>, "v_cvt_f64_u32",
  VOP_F64_I32, uint_to_fp
>;

} // End SchedRW = [WriteQuarterRate32]

defm V_FRACT_F32 : VOP1Inst <vop1<0x20, 0x1b>, "v_fract_f32",
  VOP_F32_F32, AMDGPUfract
>;
defm V_TRUNC_F32 : VOP1Inst <vop1<0x21, 0x1c>, "v_trunc_f32",
  VOP_F32_F32, ftrunc
>;
defm V_CEIL_F32 : VOP1Inst <vop1<0x22, 0x1d>, "v_ceil_f32",
  VOP_F32_F32, fceil
>;
defm V_RNDNE_F32 : VOP1Inst <vop1<0x23, 0x1e>, "v_rndne_f32",
  VOP_F32_F32, frint
>;
defm V_FLOOR_F32 : VOP1Inst <vop1<0x24, 0x1f>, "v_floor_f32",
  VOP_F32_F32, ffloor
>;
defm V_EXP_F32 : VOP1Inst <vop1<0x25, 0x20>, "v_exp_f32",
  VOP_F32_F32, fexp2
>;

let SchedRW = [WriteQuarterRate32] in {

defm V_LOG_F32 : VOP1Inst <vop1<0x27, 0x21>, "v_log_f32",
  VOP_F32_F32, flog2
>;
defm V_RCP_F32 : VOP1Inst <vop1<0x2a, 0x22>, "v_rcp_f32",
  VOP_F32_F32, AMDGPUrcp
>;
defm V_RCP_IFLAG_F32 : VOP1Inst <vop1<0x2b, 0x23>, "v_rcp_iflag_f32",
  VOP_F32_F32
>;
defm V_RSQ_F32 : VOP1Inst <vop1<0x2e, 0x24>, "v_rsq_f32",
  VOP_F32_F32, AMDGPUrsq
>;

} // End SchedRW = [WriteQuarterRate32]

let SchedRW = [WriteDouble] in {

defm V_RCP_F64 : VOP1Inst <vop1<0x2f, 0x25>, "v_rcp_f64",
  VOP_F64_F64, AMDGPUrcp
>;
defm V_RSQ_F64 : VOP1Inst <vop1<0x31, 0x26>, "v_rsq_f64",
  VOP_F64_F64, AMDGPUrsq
>;

} // End SchedRW = [WriteDouble];

defm V_SQRT_F32 : VOP1Inst <vop1<0x33, 0x27>, "v_sqrt_f32",
  VOP_F32_F32, fsqrt
>;

let SchedRW = [WriteDouble] in {

defm V_SQRT_F64 : VOP1Inst <vop1<0x34, 0x28>, "v_sqrt_f64",
  VOP_F64_F64, fsqrt
>;

} // End SchedRW = [WriteDouble]

let SchedRW = [WriteQuarterRate32] in {

defm V_SIN_F32 : VOP1Inst <vop1<0x35, 0x29>, "v_sin_f32",
  VOP_F32_F32, AMDGPUsin
>;
defm V_COS_F32 : VOP1Inst <vop1<0x36, 0x2a>, "v_cos_f32",
  VOP_F32_F32, AMDGPUcos
>;

} // End SchedRW = [WriteQuarterRate32]

defm V_NOT_B32 : VOP1Inst <vop1<0x37, 0x2b>, "v_not_b32", VOP_I32_I32>;
defm V_BFREV_B32 : VOP1Inst <vop1<0x38, 0x2c>, "v_bfrev_b32", VOP_I32_I32>;
defm V_FFBH_U32 : VOP1Inst <vop1<0x39, 0x2d>, "v_ffbh_u32", VOP_I32_I32>;
defm V_FFBL_B32 : VOP1Inst <vop1<0x3a, 0x2e>, "v_ffbl_b32", VOP_I32_I32>;
defm V_FFBH_I32 : VOP1Inst <vop1<0x3b, 0x2f>, "v_ffbh_i32", VOP_I32_I32>;
defm V_FREXP_EXP_I32_F64 : VOP1Inst <vop1<0x3c,0x30>, "v_frexp_exp_i32_f64",
  VOP_I32_F64, int_amdgcn_frexp_exp
>;

let SchedRW = [WriteDoubleAdd] in {
defm V_FREXP_MANT_F64 : VOP1Inst <vop1<0x3d, 0x31>, "v_frexp_mant_f64",
  VOP_F64_F64, int_amdgcn_frexp_mant
>;

defm V_FRACT_F64 : VOP1Inst <vop1<0x3e, 0x32>, "v_fract_f64",
  VOP_F64_F64, AMDGPUfract
>;
} // End SchedRW = [WriteDoubleAdd]


defm V_FREXP_EXP_I32_F32 : VOP1Inst <vop1<0x3f, 0x33>, "v_frexp_exp_i32_f32",
  VOP_I32_F32, int_amdgcn_frexp_exp
>;
defm V_FREXP_MANT_F32 : VOP1Inst <vop1<0x40, 0x34>, "v_frexp_mant_f32",
  VOP_F32_F32, int_amdgcn_frexp_mant
>;
let vdst = 0, src0 = 0, VOPAsmPrefer32Bit = 1 in {
defm V_CLREXCP : VOP1Inst <vop1<0x41,0x35>, "v_clrexcp", VOP_NO_EXT<VOP_NONE>>;
}

let Uses = [M0, EXEC] in {
// v_movreld_b32 is a special case because the destination output
 // register is really a source. It isn't actually read (but may be
 // written), and is only to provide the base register to start
 // indexing from. Tablegen seems to not let you define an implicit
 // virtual register output for the super register being written into,
 // so this must have an implicit def of the register added to it.
defm V_MOVRELD_B32 : VOP1Inst <vop1<0x42, 0x36>, "v_movreld_b32", VOP_MOVRELD>;
defm V_MOVRELS_B32 : VOP1Inst <vop1<0x43, 0x37>, "v_movrels_b32", VOP_I32_VI32_NO_EXT>;
defm V_MOVRELSD_B32 : VOP1Inst <vop1<0x44, 0x38>, "v_movrelsd_b32", VOP_NO_EXT<VOP_I32_I32>>;

} // End Uses = [M0, EXEC]

// These instruction only exist on SI and CI
let SubtargetPredicate = isSICI in {

let SchedRW = [WriteQuarterRate32] in {

defm V_MOV_FED_B32 : VOP1InstSI <vop1<0x9>, "v_mov_fed_b32", VOP_I32_I32>;
defm V_LOG_CLAMP_F32 : VOP1InstSI <vop1<0x26>, "v_log_clamp_f32",
  VOP_F32_F32, int_amdgcn_log_clamp>;
defm V_RCP_CLAMP_F32 : VOP1InstSI <vop1<0x28>, "v_rcp_clamp_f32", VOP_F32_F32>;
defm V_RCP_LEGACY_F32 : VOP1InstSI <vop1<0x29>, "v_rcp_legacy_f32",
  VOP_F32_F32, AMDGPUrcp_legacy>;
defm V_RSQ_CLAMP_F32 : VOP1InstSI <vop1<0x2c>, "v_rsq_clamp_f32",
  VOP_F32_F32, AMDGPUrsq_clamp
>;
defm V_RSQ_LEGACY_F32 : VOP1InstSI <vop1<0x2d>, "v_rsq_legacy_f32",
  VOP_F32_F32, AMDGPUrsq_legacy
>;

} // End SchedRW = [WriteQuarterRate32]

let SchedRW = [WriteDouble] in {

defm V_RCP_CLAMP_F64 : VOP1InstSI <vop1<0x30>, "v_rcp_clamp_f64", VOP_F64_F64>;
defm V_RSQ_CLAMP_F64 : VOP1InstSI <vop1<0x32>, "v_rsq_clamp_f64",
  VOP_F64_F64, AMDGPUrsq_clamp
>;

} // End SchedRW = [WriteDouble]

} // End SubtargetPredicate = isSICI

//===----------------------------------------------------------------------===//
// VINTRP Instructions
//===----------------------------------------------------------------------===//

let Uses = [M0, EXEC] in {

// FIXME: Specify SchedRW for VINTRP insturctions.

multiclass V_INTERP_P1_F32_m : VINTRP_m <
  0x00000000,
  (outs VGPR_32:$dst),
  (ins VGPR_32:$i, i32imm:$attr_chan, i32imm:$attr),
  "v_interp_p1_f32 $dst, $i, $attr_chan, $attr, [m0]",
  [(set f32:$dst, (AMDGPUinterp_p1 i32:$i, (i32 imm:$attr_chan),
                                           (i32 imm:$attr)))]
>;

let OtherPredicates = [has32BankLDS] in {

defm V_INTERP_P1_F32 : V_INTERP_P1_F32_m;

} // End OtherPredicates = [has32BankLDS]

let OtherPredicates = [has16BankLDS], Constraints = "@earlyclobber $dst", isAsmParserOnly=1 in {

defm V_INTERP_P1_F32_16bank : V_INTERP_P1_F32_m;

} // End OtherPredicates = [has32BankLDS], Constraints = "@earlyclobber $dst", isAsmParserOnly=1

let DisableEncoding = "$src0", Constraints = "$src0 = $dst" in {

defm V_INTERP_P2_F32 : VINTRP_m <
  0x00000001,
  (outs VGPR_32:$dst),
  (ins VGPR_32:$src0, VGPR_32:$j, i32imm:$attr_chan, i32imm:$attr),
  "v_interp_p2_f32 $dst, [$src0], $j, $attr_chan, $attr, [m0]",
  [(set f32:$dst, (AMDGPUinterp_p2 f32:$src0, i32:$j, (i32 imm:$attr_chan),
                                                     (i32 imm:$attr)))]>;

} // End DisableEncoding = "$src0", Constraints = "$src0 = $dst"

defm V_INTERP_MOV_F32 : VINTRP_m <
  0x00000002,
  (outs VGPR_32:$dst),
  (ins InterpSlot:$src0, i32imm:$attr_chan, i32imm:$attr),
  "v_interp_mov_f32 $dst, $src0, $attr_chan, $attr, [m0]",
  [(set f32:$dst, (AMDGPUinterp_mov (i32 imm:$src0), (i32 imm:$attr_chan),
                                    (i32 imm:$attr)))]>;

} // End Uses = [M0, EXEC]

//===----------------------------------------------------------------------===//
// VOP2 Instructions
//===----------------------------------------------------------------------===//

defm V_CNDMASK_B32 : VOP2eInst <vop2<0x0, 0x0>, "v_cndmask_b32",
  VOP2e_I32_I32_I32_I1
>;

let isCommutable = 1 in {
defm V_ADD_F32 : VOP2Inst <vop2<0x3, 0x1>, "v_add_f32",
  VOP_F32_F32_F32, fadd
>;

defm V_SUB_F32 : VOP2Inst <vop2<0x4, 0x2>, "v_sub_f32", VOP_F32_F32_F32, fsub>;
defm V_SUBREV_F32 : VOP2Inst <vop2<0x5, 0x3>, "v_subrev_f32",
  VOP_F32_F32_F32, null_frag, "v_sub_f32"
>;
} // End isCommutable = 1

let isCommutable = 1 in {

defm V_MUL_LEGACY_F32 : VOP2Inst <vop2<0x7, 0x4>, "v_mul_legacy_f32",
  VOP_F32_F32_F32, AMDGPUfmul_legacy
>;

defm V_MUL_F32 : VOP2Inst <vop2<0x8, 0x5>, "v_mul_f32",
  VOP_F32_F32_F32, fmul
>;

defm V_MUL_I32_I24 : VOP2Inst <vop2<0x9, 0x6>, "v_mul_i32_i24",
  VOP_I32_I32_I32, AMDGPUmul_i24
>;

defm V_MUL_HI_I32_I24 : VOP2Inst <vop2<0xa,0x7>, "v_mul_hi_i32_i24",
  VOP_I32_I32_I32, AMDGPUmulhi_i24
>;

defm V_MUL_U32_U24 : VOP2Inst <vop2<0xb, 0x8>, "v_mul_u32_u24",
  VOP_I32_I32_I32, AMDGPUmul_u24
>;

defm V_MUL_HI_U32_U24 : VOP2Inst <vop2<0xc,0x9>, "v_mul_hi_u32_u24",
 VOP_I32_I32_I32, AMDGPUmulhi_u24
>;

defm V_MIN_F32 : VOP2Inst <vop2<0xf, 0xa>, "v_min_f32", VOP_F32_F32_F32,
  fminnum>;
defm V_MAX_F32 : VOP2Inst <vop2<0x10, 0xb>, "v_max_f32", VOP_F32_F32_F32,
  fmaxnum>;
defm V_MIN_I32 : VOP2Inst <vop2<0x11, 0xc>, "v_min_i32", VOP_I32_I32_I32>;
defm V_MAX_I32 : VOP2Inst <vop2<0x12, 0xd>, "v_max_i32", VOP_I32_I32_I32>;
defm V_MIN_U32 : VOP2Inst <vop2<0x13, 0xe>, "v_min_u32", VOP_I32_I32_I32>;
defm V_MAX_U32 : VOP2Inst <vop2<0x14, 0xf>, "v_max_u32", VOP_I32_I32_I32>;

defm V_LSHRREV_B32 : VOP2Inst <
  vop2<0x16, 0x10>, "v_lshrrev_b32", VOP_I32_I32_I32, null_frag,
    "v_lshr_b32"
>;

defm V_ASHRREV_I32 : VOP2Inst <
  vop2<0x18, 0x11>, "v_ashrrev_i32", VOP_I32_I32_I32, null_frag,
    "v_ashr_i32"
>;

defm V_LSHLREV_B32 : VOP2Inst <
  vop2<0x1a, 0x12>, "v_lshlrev_b32", VOP_I32_I32_I32, null_frag,
    "v_lshl_b32"
>;

defm V_AND_B32 : VOP2Inst <vop2<0x1b, 0x13>, "v_and_b32", VOP_I32_I32_I32>;
defm V_OR_B32 : VOP2Inst <vop2<0x1c, 0x14>, "v_or_b32", VOP_I32_I32_I32>;
defm V_XOR_B32 : VOP2Inst <vop2<0x1d, 0x15>, "v_xor_b32", VOP_I32_I32_I32>;

let Constraints = "$vdst = $src2", DisableEncoding="$src2",
    isConvertibleToThreeAddress = 1 in {
defm V_MAC_F32 : VOP2Inst <vop2<0x1f, 0x16>, "v_mac_f32", VOP_MAC>;
}
} // End isCommutable = 1

defm V_MADMK_F32 : VOP2MADK <vop2<0x20, 0x17>, "v_madmk_f32", VOP_MADMK>;

let isCommutable = 1 in {
defm V_MADAK_F32 : VOP2MADK <vop2<0x21, 0x18>, "v_madak_f32", VOP_MADAK>;
} // End isCommutable = 1

let isCommutable = 1 in {
// No patterns so that the scalar instructions are always selected.
// The scalar versions will be replaced with vector when needed later.

// V_ADD_I32, V_SUB_I32, and V_SUBREV_I32 where renamed to *_U32 in VI,
// but the VI instructions behave the same as the SI versions.
defm V_ADD_I32 : VOP2bInst <vop2<0x25, 0x19>, "v_add_i32",
  VOP2b_I32_I1_I32_I32
>;
defm V_SUB_I32 : VOP2bInst <vop2<0x26, 0x1a>, "v_sub_i32", VOP2b_I32_I1_I32_I32>;

defm V_SUBREV_I32 : VOP2bInst <vop2<0x27, 0x1b>, "v_subrev_i32",
  VOP2b_I32_I1_I32_I32, null_frag, "v_sub_i32"
>;

defm V_ADDC_U32 : VOP2bInst <vop2<0x28, 0x1c>, "v_addc_u32",
  VOP2b_I32_I1_I32_I32_I1
>;
defm V_SUBB_U32 : VOP2bInst <vop2<0x29, 0x1d>, "v_subb_u32",
  VOP2b_I32_I1_I32_I32_I1
>;
defm V_SUBBREV_U32 : VOP2bInst <vop2<0x2a, 0x1e>, "v_subbrev_u32",
  VOP2b_I32_I1_I32_I32_I1, null_frag, "v_subb_u32"
>;

} // End isCommutable = 1

// These are special and do not read the exec mask.
let isConvergent = 1, Uses = []<Register> in {

defm V_READLANE_B32 : VOP2SI_3VI_m <
  vop3 <0x001, 0x289>,
  "v_readlane_b32",
  (outs SReg_32:$vdst),
  (ins VGPR_32:$src0, SCSrc_b32:$src1),
  "v_readlane_b32 $vdst, $src0, $src1",
  [(set i32:$vdst, (int_amdgcn_readlane i32:$src0, i32:$src1))]
>;

defm V_WRITELANE_B32 : VOP2SI_3VI_m <
  vop3 <0x002, 0x28a>,
  "v_writelane_b32",
  (outs VGPR_32:$vdst),
  (ins SReg_32:$src0, SCSrc_b32:$src1),
  "v_writelane_b32 $vdst, $src0, $src1"
>;

} // End isConvergent = 1

// These instructions only exist on SI and CI
let SubtargetPredicate = isSICI in {

let isCommutable = 1 in {
defm V_MAC_LEGACY_F32 : VOP2InstSI <vop2<0x6>, "v_mac_legacy_f32",
  VOP_F32_F32_F32
>;
} // End isCommutable = 1

defm V_MIN_LEGACY_F32 : VOP2InstSI <vop2<0xd>, "v_min_legacy_f32",
  VOP_F32_F32_F32, AMDGPUfmin_legacy
>;
defm V_MAX_LEGACY_F32 : VOP2InstSI <vop2<0xe>, "v_max_legacy_f32",
  VOP_F32_F32_F32, AMDGPUfmax_legacy
>;

let isCommutable = 1 in {
defm V_LSHR_B32 : VOP2InstSI <vop2<0x15>, "v_lshr_b32", VOP_I32_I32_I32>;
defm V_ASHR_I32 : VOP2InstSI <vop2<0x17>, "v_ashr_i32", VOP_I32_I32_I32>;
defm V_LSHL_B32 : VOP2InstSI <vop2<0x19>, "v_lshl_b32", VOP_I32_I32_I32>;
} // End isCommutable = 1
} // End let SubtargetPredicate = SICI

defm V_BFM_B32 : VOP2_VI3_Inst <vop23<0x1e, 0x293>, "v_bfm_b32",
  VOP_I32_I32_I32
>;
defm V_BCNT_U32_B32 : VOP2_VI3_Inst <vop23<0x22, 0x28b>, "v_bcnt_u32_b32",
  VOP_I32_I32_I32
>;
defm V_MBCNT_LO_U32_B32 : VOP2_VI3_Inst <vop23<0x23, 0x28c>, "v_mbcnt_lo_u32_b32",
  VOP_I32_I32_I32, int_amdgcn_mbcnt_lo
>;
defm V_MBCNT_HI_U32_B32 : VOP2_VI3_Inst <vop23<0x24, 0x28d>, "v_mbcnt_hi_u32_b32",
  VOP_I32_I32_I32, int_amdgcn_mbcnt_hi
>;
defm V_LDEXP_F32 : VOP2_VI3_Inst <vop23<0x2b, 0x288>, "v_ldexp_f32",
  VOP_F32_F32_I32, AMDGPUldexp
>;

defm V_CVT_PKACCUM_U8_F32 : VOP2_VI3_Inst <vop23<0x2c, 0x1f0>, "v_cvt_pkaccum_u8_f32",
  VOP_I32_F32_I32>; // TODO: set "Uses = dst"

defm V_CVT_PKNORM_I16_F32 : VOP2_VI3_Inst <vop23<0x2d, 0x294>, "v_cvt_pknorm_i16_f32",
  VOP_I32_F32_F32
>;
defm V_CVT_PKNORM_U16_F32 : VOP2_VI3_Inst <vop23<0x2e, 0x295>, "v_cvt_pknorm_u16_f32",
  VOP_I32_F32_F32
>;
defm V_CVT_PKRTZ_F16_F32 : VOP2_VI3_Inst <vop23<0x2f, 0x296>, "v_cvt_pkrtz_f16_f32",
  VOP_I32_F32_F32, int_SI_packf16
>;
defm V_CVT_PK_U16_U32 : VOP2_VI3_Inst <vop23<0x30, 0x297>, "v_cvt_pk_u16_u32",
  VOP_I32_I32_I32
>;
defm V_CVT_PK_I16_I32 : VOP2_VI3_Inst <vop23<0x31, 0x298>, "v_cvt_pk_i16_i32",
  VOP_I32_I32_I32
>;

//===----------------------------------------------------------------------===//
// Pseudo Instructions
//===----------------------------------------------------------------------===//

let hasSideEffects = 0, mayLoad = 0, mayStore = 0, Uses = [EXEC] in {

// For use in patterns
def V_CNDMASK_B64_PSEUDO : VOP3Common <(outs VReg_64:$vdst),
  (ins VSrc_b64:$src0, VSrc_b64:$src1, SSrc_b64:$src2), "", []> {
  let isPseudo = 1;
  let isCodeGenOnly = 1;
  let usesCustomInserter = 1;
}

// 64-bit vector move instruction.  This is mainly used by the SIFoldOperands
// pass to enable folding of inline immediates.
def V_MOV_B64_PSEUDO : PseudoInstSI <(outs VReg_64:$vdst), (ins VSrc_b64:$src0)> {
  let VALU = 1;
}
} // End let hasSideEffects = 0, mayLoad = 0, mayStore = 0, Uses = [EXEC]

let usesCustomInserter = 1, SALU = 1 in {
def GET_GROUPSTATICSIZE : PseudoInstSI <(outs SReg_32:$sdst), (ins),
  [(set SReg_32:$sdst, (int_amdgcn_groupstaticsize))]>;
} // End let usesCustomInserter = 1, SALU = 1

// SI pseudo instructions. These are used by the CFG structurizer pass
// and should be lowered to ISA instructions prior to codegen.

// Dummy terminator instruction to use after control flow instructions
// replaced with exec mask operations.
def SI_MASK_BRANCH : PseudoInstSI <
  (outs), (ins brtarget:$target)> {
  let isBranch = 0;
  let isTerminator = 1;
  let isBarrier = 0;
  let SALU = 1;
  let Uses = [EXEC];
}

let isTerminator = 1 in {

def SI_IF: CFPseudoInstSI <
  (outs SReg_64:$dst), (ins SReg_64:$vcc, brtarget:$target),
  [(set i64:$dst, (int_amdgcn_if i1:$vcc, bb:$target))], 1, 1> {
  let Constraints = "";
  let Size = 8;
  let mayStore = 1;
  let mayLoad = 1;
  let hasSideEffects = 1;
}

def SI_ELSE : CFPseudoInstSI <
  (outs SReg_64:$dst), (ins SReg_64:$src, brtarget:$target, i1imm:$execfix), [], 1, 1> {
  let Constraints = "$src = $dst";
  let Size = 12;
  let mayStore = 1;
  let mayLoad = 1;
  let hasSideEffects = 1;
}

def SI_LOOP : CFPseudoInstSI <
  (outs), (ins SReg_64:$saved, brtarget:$target),
  [(int_amdgcn_loop i64:$saved, bb:$target)], 1, 1> {
  let Size = 8;
  let isBranch = 1;
  let hasSideEffects = 1;
  let mayLoad = 1;
  let mayStore = 1;
}

} // End isBranch = 1, isTerminator = 1

def SI_END_CF : CFPseudoInstSI <
  (outs), (ins SReg_64:$saved),
  [(int_amdgcn_end_cf i64:$saved)], 1, 1> {
  let Size = 4;
  let isAsCheapAsAMove = 1;
  let isReMaterializable = 1;
  let mayLoad = 1;
  let mayStore = 1;
  let hasSideEffects = 1;
}

def SI_BREAK : CFPseudoInstSI <
  (outs SReg_64:$dst), (ins SReg_64:$src),
  [(set i64:$dst, (int_amdgcn_break i64:$src))], 1> {
  let Size = 4;
  let isAsCheapAsAMove = 1;
  let isReMaterializable = 1;
}

def SI_IF_BREAK : CFPseudoInstSI <
  (outs SReg_64:$dst), (ins SReg_64:$vcc, SReg_64:$src),
  [(set i64:$dst, (int_amdgcn_if_break i1:$vcc, i64:$src))]> {
  let Size = 4;
  let isAsCheapAsAMove = 1;
  let isReMaterializable = 1;
}

def SI_ELSE_BREAK : CFPseudoInstSI <
  (outs SReg_64:$dst), (ins SReg_64:$src0, SReg_64:$src1),
  [(set i64:$dst, (int_amdgcn_else_break i64:$src0, i64:$src1))]> {
  let Size = 4;
  let isAsCheapAsAMove = 1;
  let isReMaterializable = 1;
}

let Uses = [EXEC], Defs = [EXEC,VCC] in {
def SI_KILL : PseudoInstSI <
  (outs), (ins VSrc_b32:$src),
  [(AMDGPUkill i32:$src)]> {
  let isConvergent = 1;
  let usesCustomInserter = 1;
}

def SI_KILL_TERMINATOR : SPseudoInstSI <
  (outs), (ins VSrc_b32:$src)> {
  let isTerminator = 1;
}

} // End Uses = [EXEC], Defs = [EXEC,VCC]


def SI_PS_LIVE : PseudoInstSI <
  (outs SReg_64:$dst), (ins),
  [(set i1:$dst, (int_amdgcn_ps_live))]> {
  let SALU = 1;
}

// Used as an isel pseudo to directly emit initialization with an
// s_mov_b32 rather than a copy of another initialized
// register. MachineCSE skips copies, and we don't want to have to
// fold operands before it runs.
def SI_INIT_M0 : SPseudoInstSI <(outs), (ins SSrc_b32:$src)> {
  let Defs = [M0];
  let usesCustomInserter = 1;
  let isAsCheapAsAMove = 1;
  let isReMaterializable = 1;
}

def SI_RETURN : SPseudoInstSI <
  (outs), (ins variable_ops), [(AMDGPUreturn)]> {
  let isTerminator = 1;
  let isBarrier = 1;
  let isReturn = 1;
  let hasSideEffects = 1;
  let hasNoSchedulingInfo = 1;
  let DisableWQM = 1;
}

let Defs = [M0, EXEC],
  UseNamedOperandTable = 1 in {

class SI_INDIRECT_SRC<RegisterClass rc> : VPseudoInstSI <
  (outs VGPR_32:$vdst),
  (ins rc:$src, VS_32:$idx, i32imm:$offset)> {
  let usesCustomInserter = 1;
}

class SI_INDIRECT_DST<RegisterClass rc> : VPseudoInstSI <
  (outs rc:$vdst),
  (ins rc:$src, VS_32:$idx, i32imm:$offset, VGPR_32:$val)> {
  let Constraints = "$src = $vdst";
  let usesCustomInserter = 1;
}

// TODO: We can support indirect SGPR access.
def SI_INDIRECT_SRC_V1 : SI_INDIRECT_SRC<VGPR_32>;
def SI_INDIRECT_SRC_V2 : SI_INDIRECT_SRC<VReg_64>;
def SI_INDIRECT_SRC_V4 : SI_INDIRECT_SRC<VReg_128>;
def SI_INDIRECT_SRC_V8 : SI_INDIRECT_SRC<VReg_256>;
def SI_INDIRECT_SRC_V16 : SI_INDIRECT_SRC<VReg_512>;

def SI_INDIRECT_DST_V1 : SI_INDIRECT_DST<VGPR_32>;
def SI_INDIRECT_DST_V2 : SI_INDIRECT_DST<VReg_64>;
def SI_INDIRECT_DST_V4 : SI_INDIRECT_DST<VReg_128>;
def SI_INDIRECT_DST_V8 : SI_INDIRECT_DST<VReg_256>;
def SI_INDIRECT_DST_V16 : SI_INDIRECT_DST<VReg_512>;

} // End Uses = [EXEC], Defs = [M0, EXEC]

multiclass SI_SPILL_SGPR <RegisterClass sgpr_class> {
  let UseNamedOperandTable = 1, SGPRSpill = 1, Uses = [EXEC] in {
    def _SAVE : PseudoInstSI <
      (outs),
      (ins sgpr_class:$data, i32imm:$addr)> {
      let mayStore = 1;
      let mayLoad = 0;
    }

    def _RESTORE : PseudoInstSI <
      (outs sgpr_class:$data),
      (ins i32imm:$addr)> {
      let mayStore = 0;
      let mayLoad = 1;
    }
  } // End UseNamedOperandTable = 1
}

// You cannot use M0 as the output of v_readlane_b32 instructions or
// use it in the sdata operand of SMEM instructions. We still need to
// be able to spill the physical register m0, so allow it for
// SI_SPILL_32_* instructions.
defm SI_SPILL_S32  : SI_SPILL_SGPR <SReg_32>;
defm SI_SPILL_S64  : SI_SPILL_SGPR <SReg_64>;
defm SI_SPILL_S128 : SI_SPILL_SGPR <SReg_128>;
defm SI_SPILL_S256 : SI_SPILL_SGPR <SReg_256>;
defm SI_SPILL_S512 : SI_SPILL_SGPR <SReg_512>;

multiclass SI_SPILL_VGPR <RegisterClass vgpr_class> {
  let UseNamedOperandTable = 1, VGPRSpill = 1,
       SchedRW = [WriteVMEM] in {
    def _SAVE : VPseudoInstSI <
      (outs),
      (ins vgpr_class:$vdata, i32imm:$vaddr, SReg_128:$srsrc,
           SReg_32:$soffset, i32imm:$offset)> {
      let mayStore = 1;
      let mayLoad = 0;
      // (2 * 4) + (8 * num_subregs) bytes maximum
      let Size = !add(!shl(!srl(vgpr_class.Size, 5), 3), 8);
    }

    def _RESTORE : VPseudoInstSI <
      (outs vgpr_class:$vdata),
      (ins i32imm:$vaddr, SReg_128:$srsrc, SReg_32:$soffset,
           i32imm:$offset)> {
      let mayStore = 0;
      let mayLoad = 1;

      // (2 * 4) + (8 * num_subregs) bytes maximum
      let Size = !add(!shl(!srl(vgpr_class.Size, 5), 3), 8);
    }
  } // End UseNamedOperandTable = 1, VGPRSpill = 1, SchedRW = [WriteVMEM]
}

defm SI_SPILL_V32  : SI_SPILL_VGPR <VGPR_32>;
defm SI_SPILL_V64  : SI_SPILL_VGPR <VReg_64>;
defm SI_SPILL_V96  : SI_SPILL_VGPR <VReg_96>;
defm SI_SPILL_V128 : SI_SPILL_VGPR <VReg_128>;
defm SI_SPILL_V256 : SI_SPILL_VGPR <VReg_256>;
defm SI_SPILL_V512 : SI_SPILL_VGPR <VReg_512>;

def SI_PC_ADD_REL_OFFSET : SPseudoInstSI <
  (outs SReg_64:$dst),
  (ins si_ga:$ptr),
  [(set SReg_64:$dst, (i64 (SIpc_add_rel_offset (tglobaladdr:$ptr))))]> {
  let Defs = [SCC];
}

} // End SubtargetPredicate = isGCN

let Predicates = [isGCN] in {

def : Pat<
  (int_amdgcn_else i64:$src, bb:$target),
  (SI_ELSE $src, $target, 0)
>;

def : Pat <
  (int_AMDGPU_kilp),
  (SI_KILL 0xbf800000)
>;

def : Pat <
  (int_SI_export imm:$en, imm:$vm, imm:$done, imm:$tgt, imm:$compr,
                 f32:$src0, f32:$src1, f32:$src2, f32:$src3),
  (EXP imm:$en, imm:$tgt, imm:$compr, imm:$done, imm:$vm,
       $src0, $src1, $src2, $src3)
>;

//===----------------------------------------------------------------------===//
// VOP1 Patterns
//===----------------------------------------------------------------------===//

let Predicates = [UnsafeFPMath] in {

//def : RcpPat<V_RCP_F64_e32, f64>;
//defm : RsqPat<V_RSQ_F64_e32, f64>;
//defm : RsqPat<V_RSQ_F32_e32, f32>;

def : RsqPat<V_RSQ_F32_e32, f32>;
def : RsqPat<V_RSQ_F64_e32, f64>;

// Convert (x - floor(x)) to fract(x)
def : Pat <
  (f32 (fsub (f32 (VOP3Mods f32:$x, i32:$mods)),
             (f32 (ffloor (f32 (VOP3Mods f32:$x, i32:$mods)))))),
  (V_FRACT_F32_e64 $mods, $x, DSTCLAMP.NONE, DSTOMOD.NONE)
>;

// Convert (x + (-floor(x))) to fract(x)
def : Pat <
  (f64 (fadd (f64 (VOP3Mods f64:$x, i32:$mods)),
             (f64 (fneg (f64 (ffloor (f64 (VOP3Mods f64:$x, i32:$mods)))))))),
  (V_FRACT_F64_e64 $mods, $x, DSTCLAMP.NONE, DSTOMOD.NONE)
>;

} // End Predicates = [UnsafeFPMath]

//===----------------------------------------------------------------------===//
// VOP2 Patterns
//===----------------------------------------------------------------------===//

def : Pat <
  (i32 (add (i32 (ctpop i32:$popcnt)), i32:$val)),
  (V_BCNT_U32_B32_e64 $popcnt, $val)
>;

def : Pat <
  (i32 (select i1:$src0, i32:$src1, i32:$src2)),
  (V_CNDMASK_B32_e64 $src2, $src1, $src0)
>;

// Pattern for V_MAC_F32
def : Pat <
  (fmad  (VOP3NoMods0 f32:$src0, i32:$src0_modifiers, i1:$clamp, i32:$omod),
         (VOP3NoMods f32:$src1, i32:$src1_modifiers),
         (VOP3NoMods f32:$src2, i32:$src2_modifiers)),
  (V_MAC_F32_e64 $src0_modifiers, $src0, $src1_modifiers, $src1,
                 $src2_modifiers, $src2, $clamp, $omod)
>;

/********** ============================================ **********/
/********** Extraction, Insertion, Building and Casting  **********/
/********** ============================================ **********/

foreach Index = 0-2 in {
  def Extract_Element_v2i32_#Index : Extract_Element <
    i32, v2i32, Index, !cast<SubRegIndex>(sub#Index)
  >;
  def Insert_Element_v2i32_#Index : Insert_Element <
    i32, v2i32, Index, !cast<SubRegIndex>(sub#Index)
  >;

  def Extract_Element_v2f32_#Index : Extract_Element <
    f32, v2f32, Index, !cast<SubRegIndex>(sub#Index)
  >;
  def Insert_Element_v2f32_#Index : Insert_Element <
    f32, v2f32, Index, !cast<SubRegIndex>(sub#Index)
  >;
}

foreach Index = 0-3 in {
  def Extract_Element_v4i32_#Index : Extract_Element <
    i32, v4i32, Index, !cast<SubRegIndex>(sub#Index)
  >;
  def Insert_Element_v4i32_#Index : Insert_Element <
    i32, v4i32, Index, !cast<SubRegIndex>(sub#Index)
  >;

  def Extract_Element_v4f32_#Index : Extract_Element <
    f32, v4f32, Index, !cast<SubRegIndex>(sub#Index)
  >;
  def Insert_Element_v4f32_#Index : Insert_Element <
    f32, v4f32, Index, !cast<SubRegIndex>(sub#Index)
  >;
}

foreach Index = 0-7 in {
  def Extract_Element_v8i32_#Index : Extract_Element <
    i32, v8i32, Index, !cast<SubRegIndex>(sub#Index)
  >;
  def Insert_Element_v8i32_#Index : Insert_Element <
    i32, v8i32, Index, !cast<SubRegIndex>(sub#Index)
  >;

  def Extract_Element_v8f32_#Index : Extract_Element <
    f32, v8f32, Index, !cast<SubRegIndex>(sub#Index)
  >;
  def Insert_Element_v8f32_#Index : Insert_Element <
    f32, v8f32, Index, !cast<SubRegIndex>(sub#Index)
  >;
}

foreach Index = 0-15 in {
  def Extract_Element_v16i32_#Index : Extract_Element <
    i32, v16i32, Index, !cast<SubRegIndex>(sub#Index)
  >;
  def Insert_Element_v16i32_#Index : Insert_Element <
    i32, v16i32, Index, !cast<SubRegIndex>(sub#Index)
  >;

  def Extract_Element_v16f32_#Index : Extract_Element <
    f32, v16f32, Index, !cast<SubRegIndex>(sub#Index)
  >;
  def Insert_Element_v16f32_#Index : Insert_Element <
    f32, v16f32, Index, !cast<SubRegIndex>(sub#Index)
  >;
}

// FIXME: Why do only some of these type combinations for SReg and
// VReg?
// 32-bit bitcast
def : BitConvert <i32, f32, VGPR_32>;
def : BitConvert <f32, i32, VGPR_32>;
def : BitConvert <i32, f32, SReg_32>;
def : BitConvert <f32, i32, SReg_32>;

// 64-bit bitcast
def : BitConvert <i64, f64, VReg_64>;
def : BitConvert <f64, i64, VReg_64>;
def : BitConvert <v2i32, v2f32, VReg_64>;
def : BitConvert <v2f32, v2i32, VReg_64>;
def : BitConvert <i64, v2i32, VReg_64>;
def : BitConvert <v2i32, i64, VReg_64>;
def : BitConvert <i64, v2f32, VReg_64>;
def : BitConvert <v2f32, i64, VReg_64>;
def : BitConvert <f64, v2f32, VReg_64>;
def : BitConvert <v2f32, f64, VReg_64>;
def : BitConvert <f64, v2i32, VReg_64>;
def : BitConvert <v2i32, f64, VReg_64>;
def : BitConvert <v4i32, v4f32, VReg_128>;
def : BitConvert <v4f32, v4i32, VReg_128>;

// 128-bit bitcast
def : BitConvert <v2i64, v4i32, SReg_128>;
def : BitConvert <v4i32, v2i64, SReg_128>;
def : BitConvert <v2f64, v4f32, VReg_128>;
def : BitConvert <v2f64, v4i32, VReg_128>;
def : BitConvert <v4f32, v2f64, VReg_128>;
def : BitConvert <v4i32, v2f64, VReg_128>;
def : BitConvert <v2i64, v2f64, VReg_128>;
def : BitConvert <v2f64, v2i64, VReg_128>;

// 256-bit bitcast
def : BitConvert <v8i32, v8f32, SReg_256>;
def : BitConvert <v8f32, v8i32, SReg_256>;
def : BitConvert <v8i32, v8f32, VReg_256>;
def : BitConvert <v8f32, v8i32, VReg_256>;

// 512-bit bitcast
def : BitConvert <v16i32, v16f32, VReg_512>;
def : BitConvert <v16f32, v16i32, VReg_512>;

/********** =================== **********/
/********** Src & Dst modifiers **********/
/********** =================== **********/

def : Pat <
  (AMDGPUclamp (VOP3Mods0Clamp f32:$src0, i32:$src0_modifiers, i32:$omod),
               (f32 FP_ZERO), (f32 FP_ONE)),
  (V_ADD_F32_e64 $src0_modifiers, $src0, 0, 0, 1, $omod)
>;

/********** ================================ **********/
/********** Floating point absolute/negative **********/
/********** ================================ **********/

// Prevent expanding both fneg and fabs.

def : Pat <
  (fneg (fabs f32:$src)),
  (S_OR_B32 $src, (S_MOV_B32 0x80000000)) // Set sign bit
>;

// FIXME: Should use S_OR_B32
def : Pat <
  (fneg (fabs f64:$src)),
  (REG_SEQUENCE VReg_64,
    (i32 (EXTRACT_SUBREG f64:$src, sub0)),
    sub0,
    (V_OR_B32_e32 (EXTRACT_SUBREG f64:$src, sub1),
                  (V_MOV_B32_e32 0x80000000)), // Set sign bit.
    sub1)
>;

def : Pat <
  (fabs f32:$src),
  (V_AND_B32_e64 $src, (V_MOV_B32_e32 0x7fffffff))
>;

def : Pat <
  (fneg f32:$src),
  (V_XOR_B32_e32 $src, (V_MOV_B32_e32 0x80000000))
>;

def : Pat <
  (fabs f64:$src),
  (REG_SEQUENCE VReg_64,
    (i32 (EXTRACT_SUBREG f64:$src, sub0)),
    sub0,
    (V_AND_B32_e64 (EXTRACT_SUBREG f64:$src, sub1),
                   (V_MOV_B32_e32 0x7fffffff)), // Set sign bit.
     sub1)
>;

def : Pat <
  (fneg f64:$src),
  (REG_SEQUENCE VReg_64,
    (i32 (EXTRACT_SUBREG f64:$src, sub0)),
    sub0,
    (V_XOR_B32_e32 (EXTRACT_SUBREG f64:$src, sub1),
                   (V_MOV_B32_e32 0x80000000)),
    sub1)
>;

/********** ================== **********/
/********** Immediate Patterns **********/
/********** ================== **********/

def : Pat <
  (SGPRImm<(i32 imm)>:$imm),
  (S_MOV_B32 imm:$imm)
>;

def : Pat <
  (SGPRImm<(f32 fpimm)>:$imm),
  (S_MOV_B32 (f32 (bitcast_fpimm_to_i32 $imm)))
>;

def : Pat <
  (i32 imm:$imm),
  (V_MOV_B32_e32 imm:$imm)
>;

def : Pat <
  (f32 fpimm:$imm),
  (V_MOV_B32_e32 (f32 (bitcast_fpimm_to_i32 $imm)))
>;

def : Pat <
 (i32 frameindex:$fi),
 (V_MOV_B32_e32 (i32 (frameindex_to_targetframeindex $fi)))
>;

def : Pat <
  (i64 InlineImm<i64>:$imm),
  (S_MOV_B64 InlineImm<i64>:$imm)
>;

// XXX - Should this use a s_cmp to set SCC?

// Set to sign-extended 64-bit value (true = -1, false = 0)
def : Pat <
  (i1 imm:$imm),
  (S_MOV_B64 (i64 (as_i64imm $imm)))
>;

def : Pat <
  (f64 InlineFPImm<f64>:$imm),
  (S_MOV_B64 (f64 (bitcast_fpimm_to_i64 InlineFPImm<f64>:$imm)))
>;

/********** ================== **********/
/********** Intrinsic Patterns **********/
/********** ================== **********/

def : POW_Common <V_LOG_F32_e32, V_EXP_F32_e32, V_MUL_LEGACY_F32_e32>;

def : Pat <
  (int_AMDGPU_cube v4f32:$src),
  (REG_SEQUENCE VReg_128,
    (V_CUBETC_F32 0 /* src0_modifiers */, (EXTRACT_SUBREG $src, sub0),
                  0 /* src1_modifiers */, (EXTRACT_SUBREG $src, sub1),
                  0 /* src2_modifiers */, (EXTRACT_SUBREG $src, sub2),
                  0 /* clamp */, 0 /* omod */), sub0,
    (V_CUBESC_F32 0 /* src0_modifiers */, (EXTRACT_SUBREG $src, sub0),
                  0 /* src1_modifiers */,(EXTRACT_SUBREG $src, sub1),
                  0 /* src2_modifiers */,(EXTRACT_SUBREG $src, sub2),
                  0 /* clamp */, 0 /* omod */), sub1,
    (V_CUBEMA_F32 0 /* src1_modifiers */,(EXTRACT_SUBREG $src, sub0),
                  0 /* src1_modifiers */,(EXTRACT_SUBREG $src, sub1),
                  0 /* src1_modifiers */,(EXTRACT_SUBREG $src, sub2),
                  0 /* clamp */, 0 /* omod */), sub2,
    (V_CUBEID_F32 0 /* src1_modifiers */,(EXTRACT_SUBREG $src, sub0),
                  0 /* src1_modifiers */,(EXTRACT_SUBREG $src, sub1),
                  0 /* src1_modifiers */,(EXTRACT_SUBREG $src, sub2),
                  0 /* clamp */, 0 /* omod */), sub3)
>;

def : Pat <
  (i32 (sext i1:$src0)),
  (V_CNDMASK_B32_e64 (i32 0), (i32 -1), $src0)
>;

class Ext32Pat <SDNode ext> : Pat <
  (i32 (ext i1:$src0)),
  (V_CNDMASK_B32_e64 (i32 0), (i32 1), $src0)
>;

def : Ext32Pat <zext>;
def : Ext32Pat <anyext>;

// The multiplication scales from [0,1] to the unsigned integer range
def : Pat <
  (AMDGPUurecip i32:$src0),
  (V_CVT_U32_F32_e32
    (V_MUL_F32_e32 CONST.FP_UINT_MAX_PLUS_1,
                   (V_RCP_IFLAG_F32_e32 (V_CVT_F32_U32_e32 $src0))))
>;

//===----------------------------------------------------------------------===//
// VOP3 Patterns
//===----------------------------------------------------------------------===//

def : IMad24Pat<V_MAD_I32_I24>;
def : UMad24Pat<V_MAD_U32_U24>;

defm : BFIPatterns <V_BFI_B32, S_MOV_B32, SReg_64>;
def : ROTRPattern <V_ALIGNBIT_B32>;

/********** ====================== **********/
/**********   Indirect adressing   **********/
/********** ====================== **********/

multiclass SI_INDIRECT_Pattern <ValueType vt, ValueType eltvt, string VecSize> {
  // Extract with offset
  def : Pat<
    (eltvt (extractelt vt:$src, (MOVRELOffset i32:$idx, (i32 imm:$offset)))),
    (!cast<Instruction>("SI_INDIRECT_SRC_"#VecSize) $src, $idx, imm:$offset)
  >;

  // Insert with offset
  def : Pat<
    (insertelt vt:$src, eltvt:$val, (MOVRELOffset i32:$idx, (i32 imm:$offset))),
    (!cast<Instruction>("SI_INDIRECT_DST_"#VecSize) $src, $idx, imm:$offset, $val)
  >;
}

defm : SI_INDIRECT_Pattern <v2f32, f32, "V2">;
defm : SI_INDIRECT_Pattern <v4f32, f32, "V4">;
defm : SI_INDIRECT_Pattern <v8f32, f32, "V8">;
defm : SI_INDIRECT_Pattern <v16f32, f32, "V16">;

defm : SI_INDIRECT_Pattern <v2i32, i32, "V2">;
defm : SI_INDIRECT_Pattern <v4i32, i32, "V4">;
defm : SI_INDIRECT_Pattern <v8i32, i32, "V8">;
defm : SI_INDIRECT_Pattern <v16i32, i32, "V16">;

//===----------------------------------------------------------------------===//
// SAD Patterns
//===----------------------------------------------------------------------===//

def : Pat <
  (add (sub_oneuse (umax i32:$src0, i32:$src1),
                   (umin i32:$src0, i32:$src1)),
       i32:$src2),
  (V_SAD_U32 $src0, $src1, $src2)
>;

def : Pat <
  (add (select_oneuse (i1 (setugt i32:$src0, i32:$src1)),
                      (sub i32:$src0, i32:$src1),
                      (sub i32:$src1, i32:$src0)),
       i32:$src2),
  (V_SAD_U32 $src0, $src1, $src2)
>;

//===----------------------------------------------------------------------===//
// Conversion Patterns
//===----------------------------------------------------------------------===//

def : Pat<(i32 (sext_inreg i32:$src, i1)),
  (S_BFE_I32 i32:$src, 65536)>; // 0 | 1 << 16

// Handle sext_inreg in i64
def : Pat <
  (i64 (sext_inreg i64:$src, i1)),
  (S_BFE_I64 i64:$src, 0x10000) // 0 | 1 << 16
>;

def : Pat <
  (i64 (sext_inreg i64:$src, i8)),
  (S_BFE_I64 i64:$src, 0x80000) // 0 | 8 << 16
>;

def : Pat <
  (i64 (sext_inreg i64:$src, i16)),
  (S_BFE_I64 i64:$src, 0x100000) // 0 | 16 << 16
>;

def : Pat <
  (i64 (sext_inreg i64:$src, i32)),
  (S_BFE_I64 i64:$src, 0x200000) // 0 | 32 << 16
>;

def : Pat <
  (i64 (zext i32:$src)),
  (REG_SEQUENCE SReg_64, $src, sub0, (S_MOV_B32 0), sub1)
>;

def : Pat <
  (i64 (anyext i32:$src)),
  (REG_SEQUENCE SReg_64, $src, sub0, (i32 (IMPLICIT_DEF)), sub1)
>;

class ZExt_i64_i1_Pat <SDNode ext> : Pat <
  (i64 (ext i1:$src)),
    (REG_SEQUENCE VReg_64,
      (V_CNDMASK_B32_e64 (i32 0), (i32 1), $src), sub0,
      (S_MOV_B32 0), sub1)
>;


def : ZExt_i64_i1_Pat<zext>;
def : ZExt_i64_i1_Pat<anyext>;

// FIXME: We need to use COPY_TO_REGCLASS to work-around the fact that
// REG_SEQUENCE patterns don't support instructions with multiple outputs.
def : Pat <
  (i64 (sext i32:$src)),
    (REG_SEQUENCE SReg_64, $src, sub0,
    (i32 (COPY_TO_REGCLASS (S_ASHR_I32 $src, 31), SReg_32_XM0)), sub1)
>;

def : Pat <
  (i64 (sext i1:$src)),
  (REG_SEQUENCE VReg_64,
    (V_CNDMASK_B32_e64 0, -1, $src), sub0,
    (V_CNDMASK_B32_e64 0, -1, $src), sub1)
>;

class FPToI1Pat<Instruction Inst, int KOne, ValueType vt, SDPatternOperator fp_to_int> : Pat <
  (i1 (fp_to_int (vt (VOP3Mods vt:$src0, i32:$src0_modifiers)))),
  (i1 (Inst 0, KOne, $src0_modifiers, $src0, DSTCLAMP.NONE, DSTOMOD.NONE))
>;

def : FPToI1Pat<V_CMP_EQ_F32_e64, CONST.FP32_ONE, f32, fp_to_uint>;
def : FPToI1Pat<V_CMP_EQ_F32_e64, CONST.FP32_NEG_ONE, f32, fp_to_sint>;
def : FPToI1Pat<V_CMP_EQ_F64_e64, CONST.FP64_ONE, f64, fp_to_uint>;
def : FPToI1Pat<V_CMP_EQ_F64_e64, CONST.FP64_NEG_ONE, f64, fp_to_sint>;

// If we need to perform a logical operation on i1 values, we need to
// use vector comparisons since there is only one SCC register. Vector
// comparisions still write to a pair of SGPRs, so treat these as
// 64-bit comparisons. When legalizing SGPR copies, instructions
// resulting in the copies from SCC to these instructions will be
// moved to the VALU.
def : Pat <
  (i1 (and i1:$src0, i1:$src1)),
  (S_AND_B64 $src0, $src1)
>;

def : Pat <
  (i1 (or i1:$src0, i1:$src1)),
  (S_OR_B64 $src0, $src1)
>;

def : Pat <
  (i1 (xor i1:$src0, i1:$src1)),
  (S_XOR_B64 $src0, $src1)
>;

def : Pat <
  (f32 (sint_to_fp i1:$src)),
  (V_CNDMASK_B32_e64 (i32 0), CONST.FP32_NEG_ONE, $src)
>;

def : Pat <
  (f32 (uint_to_fp i1:$src)),
  (V_CNDMASK_B32_e64 (i32 0), CONST.FP32_ONE, $src)
>;

def : Pat <
  (f64 (sint_to_fp i1:$src)),
  (V_CVT_F64_I32_e32 (V_CNDMASK_B32_e64 (i32 0), (i32 -1), $src))
>;

def : Pat <
  (f64 (uint_to_fp i1:$src)),
  (V_CVT_F64_U32_e32 (V_CNDMASK_B32_e64 (i32 0), (i32 1), $src))
>;

//===----------------------------------------------------------------------===//
// Miscellaneous Patterns
//===----------------------------------------------------------------------===//

def : Pat <
  (i32 (trunc i64:$a)),
  (EXTRACT_SUBREG $a, sub0)
>;

def : Pat <
  (i1 (trunc i32:$a)),
  (V_CMP_EQ_I32_e64 (S_AND_B32 (i32 1), $a), 1)
>;

def : Pat <
  (i1 (trunc i64:$a)),
  (V_CMP_EQ_I32_e64 (S_AND_B32 (i32 1),
                    (EXTRACT_SUBREG $a, sub0)), 1)
>;

def : Pat <
  (i32 (bswap i32:$a)),
  (V_BFI_B32 (S_MOV_B32 0x00ff00ff),
             (V_ALIGNBIT_B32 $a, $a, 24),
             (V_ALIGNBIT_B32 $a, $a, 8))
>;

def : Pat <
  (f32 (select i1:$src2, f32:$src1, f32:$src0)),
  (V_CNDMASK_B32_e64 $src0, $src1, $src2)
>;

multiclass BFMPatterns <ValueType vt, InstSI BFM, InstSI MOV> {
  def : Pat <
    (vt (shl (vt (add (vt (shl 1, vt:$a)), -1)), vt:$b)),
    (BFM $a, $b)
  >;

  def : Pat <
    (vt (add (vt (shl 1, vt:$a)), -1)),
    (BFM $a, (MOV 0))
  >;
}

defm : BFMPatterns <i32, S_BFM_B32, S_MOV_B32>;
// FIXME: defm : BFMPatterns <i64, S_BFM_B64, S_MOV_B64>;

def : BFEPattern <V_BFE_U32, S_MOV_B32>;

def : Pat<
  (fcanonicalize f32:$src),
  (V_MUL_F32_e64 0, CONST.FP32_ONE, 0, $src, 0, 0)
>;

def : Pat<
  (fcanonicalize f64:$src),
  (V_MUL_F64 0, CONST.FP64_ONE, 0, $src, 0, 0)
>;

//===----------------------------------------------------------------------===//
// Fract Patterns
//===----------------------------------------------------------------------===//

let Predicates = [isSI] in {

// V_FRACT is buggy on SI, so the F32 version is never used and (x-floor(x)) is
// used instead. However, SI doesn't have V_FLOOR_F64, so the most efficient
// way to implement it is using V_FRACT_F64.
// The workaround for the V_FRACT bug is:
//    fract(x) = isnan(x) ? x : min(V_FRACT(x), 0.99999999999999999)

// Convert floor(x) to (x - fract(x))
def : Pat <
  (f64 (ffloor (f64 (VOP3Mods f64:$x, i32:$mods)))),
  (V_ADD_F64
      $mods,
      $x,
      SRCMODS.NEG,
      (V_CNDMASK_B64_PSEUDO
         (V_MIN_F64
             SRCMODS.NONE,
             (V_FRACT_F64_e64 $mods, $x, DSTCLAMP.NONE, DSTOMOD.NONE),
             SRCMODS.NONE,
             (V_MOV_B64_PSEUDO 0x3fefffffffffffff),
             DSTCLAMP.NONE, DSTOMOD.NONE),
         $x,
         (V_CMP_CLASS_F64_e64 SRCMODS.NONE, $x, 3/*NaN*/)),
      DSTCLAMP.NONE, DSTOMOD.NONE)
>;

} // End Predicates = [isSI]

//============================================================================//
// Miscellaneous Optimization Patterns
//============================================================================//

def : SHA256MaPattern <V_BFI_B32, V_XOR_B32_e64>;

def : IntMed3Pat<V_MED3_I32, smax, smax_oneuse, smin_oneuse>;
def : IntMed3Pat<V_MED3_U32, umax, umax_oneuse, umin_oneuse>;

//============================================================================//
// Assembler aliases
//============================================================================//

def : MnemonicAlias<"v_add_u32", "v_add_i32">;
def : MnemonicAlias<"v_sub_u32", "v_sub_i32">;
def : MnemonicAlias<"v_subrev_u32", "v_subrev_i32">;

} // End isGCN predicate

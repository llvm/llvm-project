//===-- VIInstructions.td - VI Instruction Defintions ---------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
// Instruction definitions for VI and newer.
//===----------------------------------------------------------------------===//

let SIAssemblerPredicate = DisableInst, SubtargetPredicate = isVI in {

//===----------------------------------------------------------------------===//
// VOP1 Instructions
//===----------------------------------------------------------------------===//

defm V_CVT_F16_U16 : VOP1Inst <vop1<0, 0x39>, "v_cvt_f16_u16", VOP_F16_I16>;
defm V_CVT_F16_I16 : VOP1Inst <vop1<0, 0x3a>, "v_cvt_f16_i16", VOP_F16_I16>;
defm V_CVT_U16_F16 : VOP1Inst <vop1<0, 0x3b>, "v_cvt_u16_f16", VOP_I16_F16>;
defm V_CVT_I16_F16 : VOP1Inst <vop1<0, 0x3c>, "v_cvt_i16_f16", VOP_I16_F16>;
defm V_RCP_F16 : VOP1Inst <vop1<0, 0x3d>, "v_rcp_f16", VOP_F16_F16>;
defm V_SQRT_F16 : VOP1Inst <vop1<0, 0x3e>, "v_sqrt_f16", VOP_F16_F16>;
defm V_RSQ_F16 : VOP1Inst <vop1<0, 0x3f>, "v_rsq_f16", VOP_F16_F16>;
defm V_LOG_F16 : VOP1Inst <vop1<0, 0x40>, "v_log_f16", VOP_F16_F16>;
defm V_EXP_F16 : VOP1Inst <vop1<0, 0x41>, "v_exp_f16", VOP_F16_F16>;
defm V_FREXP_MANT_F16 : VOP1Inst <vop1<0, 0x42>, "v_frexp_mant_f16",
  VOP_F16_F16
>;
defm V_FREXP_EXP_I16_F16 : VOP1Inst <vop1<0, 0x43>, "v_frexp_exp_i16_f16",
  VOP_I16_F16
>;
defm V_FLOOR_F16 : VOP1Inst <vop1<0, 0x44>, "v_floor_f16", VOP_F16_F16>;
defm V_CEIL_F16 : VOP1Inst <vop1<0, 0x45>, "v_ceil_f16", VOP_F16_F16>;
defm V_TRUNC_F16 : VOP1Inst <vop1<0, 0x46>, "v_trunc_f16", VOP_F16_F16>;
defm V_RNDNE_F16 : VOP1Inst <vop1<0, 0x47>, "v_rndne_f16", VOP_F16_F16>;
defm V_FRACT_F16 : VOP1Inst <vop1<0, 0x48>, "v_fract_f16", VOP_F16_F16>;
defm V_SIN_F16 : VOP1Inst <vop1<0, 0x49>, "v_sin_f16", VOP_F16_F16>;
defm V_COS_F16 : VOP1Inst <vop1<0, 0x4a>, "v_cos_f16", VOP_F16_F16>;

//===----------------------------------------------------------------------===//
// VOP2 Instructions
//===----------------------------------------------------------------------===//

let isCommutable = 1 in {

defm V_ADD_F16 : VOP2Inst <vop2<0, 0x1f>, "v_add_f16", VOP_F16_F16_F16>;
defm V_SUB_F16 : VOP2Inst <vop2<0, 0x20>, "v_sub_f16", VOP_F16_F16_F16>;
defm V_SUBREV_F16 : VOP2Inst <vop2<0, 0x21>, "v_subrev_f16", VOP_F16_F16_F16,
  null_frag, "v_sub_f16"
>;
defm V_MUL_F16 : VOP2Inst <vop2<0, 0x22>, "v_mul_f16", VOP_F16_F16_F16>;
defm V_MAC_F16 : VOP2Inst <vop2<0, 0x23>, "v_mac_f16", VOP_F16_F16_F16>;
} // End isCommutable = 1
defm V_MADMK_F16 : VOP2MADK <vop2<0,0x24>, "v_madmk_f16">;
let isCommutable = 1 in {
defm V_MADAK_F16 : VOP2MADK <vop2<0,0x25>, "v_madak_f16">;
defm V_ADD_U16 : VOP2Inst <vop2<0,0x26>, "v_add_u16", VOP_I16_I16_I16>;
defm V_SUB_U16 : VOP2Inst <vop2<0,0x27>, "v_sub_u16" , VOP_I16_I16_I16>;
defm V_SUBREV_U16 : VOP2Inst <vop2<0,0x28>, "v_subrev_u16", VOP_I16_I16_I16>;
defm V_MUL_LO_U16 : VOP2Inst <vop2<0,0x29>, "v_mul_lo_u16", VOP_I16_I16_I16>;
} // End isCommutable = 1
defm V_LSHLREV_B16 : VOP2Inst <vop2<0,0x2a>, "v_lshlrev_b16", VOP_I16_I16_I16>;
defm V_LSHRREV_B16 : VOP2Inst <vop2<0,0x2b>, "v_lshrrev_b16", VOP_I16_I16_I16>;
defm V_ASHRREV_B16 : VOP2Inst <vop2<0,0x2c>, "v_ashrrev_b16", VOP_I16_I16_I16>;
let isCommutable = 1 in {
defm V_MAX_F16 : VOP2Inst <vop2<0,0x2d>, "v_max_f16", VOP_F16_F16_F16>;
defm V_MIN_F16 : VOP2Inst <vop2<0,0x2e>, "v_min_f16", VOP_F16_F16_F16>;
defm V_MAX_U16 : VOP2Inst <vop2<0,0x2f>, "v_max_u16", VOP_I16_I16_I16>;
defm V_MAX_I16 : VOP2Inst <vop2<0,0x30>, "v_max_i16", VOP_I16_I16_I16>;
defm V_MIN_U16 : VOP2Inst <vop2<0,0x31>, "v_min_u16", VOP_I16_I16_I16>;
defm V_MIN_I16 : VOP2Inst <vop2<0,0x32>, "v_min_i16", VOP_I16_I16_I16>;
} // End isCommutable = 1
defm V_LDEXP_F16 : VOP2Inst <vop2<0,0x33>, "v_ldexp_f16", VOP_F16_F16_I16>;

// Aliases to simplify matching of floating-point instructions that
// are VOP2 on SI and VOP3 on VI.

class SI2_VI3Alias <string name, Instruction inst> : InstAlias <
  name#" $dst, $src0, $src1",
  (inst VGPR_32:$dst, 0, VCSrc_32:$src0, 0, VCSrc_32:$src1, 0, 0)
>, PredicateControl {
  let UseInstAsmMatchConverter = 0;
}

def : SI2_VI3Alias <"v_ldexp_f32", V_LDEXP_F32_e64_vi>;
def : SI2_VI3Alias <"v_cvt_pkaccum_u8_f32", V_CVT_PKACCUM_U8_F32_e64_vi>;
def : SI2_VI3Alias <"v_cvt_pknorm_i16_f32", V_CVT_PKNORM_I16_F32_e64_vi>;
def : SI2_VI3Alias <"v_cvt_pknorm_u16_f32", V_CVT_PKNORM_U16_F32_e64_vi>;
def : SI2_VI3Alias <"v_cvt_pkrtz_f16_f32", V_CVT_PKRTZ_F16_F32_e64_vi>;

//===----------------------------------------------------------------------===//
// SMEM Instructions
//===----------------------------------------------------------------------===//

def S_DCACHE_WB : SMEM_Inval <0x21,
  "s_dcache_wb", int_amdgcn_s_dcache_wb>;

def S_DCACHE_WB_VOL : SMEM_Inval <0x23,
  "s_dcache_wb_vol", int_amdgcn_s_dcache_wb_vol>;

} // End SIAssemblerPredicate = DisableInst, SubtargetPredicate = isVI

//===----------------------------------------------------------------------===//
// SMEM Patterns
//===----------------------------------------------------------------------===//

let Predicates = [isVI] in {

// 1. Offset as 20bit DWORD immediate
def : Pat <
  (SIload_constant v4i32:$sbase, IMM20bit:$offset),
  (S_BUFFER_LOAD_DWORD_IMM $sbase, (as_i32imm $offset))
>;

// Patterns for global loads with no offset
class FlatLoadPat <FLAT inst, SDPatternOperator node, ValueType vt> : Pat <
  (vt (node i64:$addr)),
  (inst $addr, 0, 0, 0)
>;

def : FlatLoadPat <FLAT_LOAD_UBYTE, az_extloadi8_global, i32>;
def : FlatLoadPat <FLAT_LOAD_SBYTE, sextloadi8_global, i32>;
def : FlatLoadPat <FLAT_LOAD_USHORT, az_extloadi16_global, i32>;
def : FlatLoadPat <FLAT_LOAD_SSHORT, sextloadi16_global, i32>;
def : FlatLoadPat <FLAT_LOAD_DWORD, global_load, i32>;
def : FlatLoadPat <FLAT_LOAD_DWORDX2, global_load, v2i32>;
def : FlatLoadPat <FLAT_LOAD_DWORDX4, global_load, v4i32>;

class FlatStorePat <FLAT inst, SDPatternOperator node, ValueType vt> : Pat <
  (node vt:$data, i64:$addr),
  (inst $data, $addr, 0, 0, 0)
>;

def : FlatStorePat <FLAT_STORE_BYTE, truncstorei8_global, i32>;
def : FlatStorePat <FLAT_STORE_SHORT, truncstorei16_global, i32>;
def : FlatStorePat <FLAT_STORE_DWORD, global_store, i32>;
def : FlatStorePat <FLAT_STORE_DWORDX2, global_store, v2i32>;
def : FlatStorePat <FLAT_STORE_DWORDX4, global_store, v4i32>;

class FlatAtomicPat <FLAT inst, SDPatternOperator node, ValueType vt> : Pat <
  (vt (node i64:$addr, vt:$data)),
  (inst $addr, $data, 0, 0)
>;

def : FlatAtomicPat <FLAT_ATOMIC_ADD_RTN, atomic_add_global, i32>;
def : FlatAtomicPat <FLAT_ATOMIC_AND_RTN, atomic_and_global, i32>;
def : FlatAtomicPat <FLAT_ATOMIC_SUB_RTN, atomic_sub_global, i32>;
def : FlatAtomicPat <FLAT_ATOMIC_SMAX_RTN, atomic_max_global, i32>;
def : FlatAtomicPat <FLAT_ATOMIC_UMAX_RTN, atomic_umax_global, i32>;
def : FlatAtomicPat <FLAT_ATOMIC_SMIN_RTN, atomic_min_global, i32>;
def : FlatAtomicPat <FLAT_ATOMIC_UMIN_RTN, atomic_umin_global, i32>;
def : FlatAtomicPat <FLAT_ATOMIC_OR_RTN, atomic_or_global, i32>;
def : FlatAtomicPat <FLAT_ATOMIC_SWAP_RTN, atomic_swap_global, i32>;
def : FlatAtomicPat <FLAT_ATOMIC_XOR_RTN, atomic_xor_global, i32>;


} // End Predicates = [isVI]

//===-- CIInstructions.td - CI Instruction Defintions ---------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
// Instruction definitions for CI and newer.
//===----------------------------------------------------------------------===//
// Remaining instructions:
// S_CBRANCH_CDBGUSER
// S_CBRANCH_CDBGSYS
// S_CBRANCH_CDBGSYS_OR_USER
// S_CBRANCH_CDBGSYS_AND_USER
// DS_NOP
// DS_GWS_SEMA_RELEASE_ALL
// DS_WRAP_RTN_B32
// DS_CNDXCHG32_RTN_B64
// DS_WRITE_B96
// DS_WRITE_B128
// DS_CONDXCHG32_RTN_B128
// DS_READ_B96
// DS_READ_B128
// BUFFER_LOAD_DWORDX3
// BUFFER_STORE_DWORDX3


def isCIVI : Predicate <
  "Subtarget->getGeneration() == AMDGPUSubtarget::SEA_ISLANDS || "
  "Subtarget->getGeneration() == AMDGPUSubtarget::VOLCANIC_ISLANDS"
>, AssemblerPredicate<"FeatureCIInsts">;

def HasFlatAddressSpace : Predicate<"Subtarget->hasFlatAddressSpace()">;

//===----------------------------------------------------------------------===//
// VOP1 Instructions
//===----------------------------------------------------------------------===//

let SubtargetPredicate = isCIVI in {

let SchedRW = [WriteDoubleAdd] in {
defm V_TRUNC_F64 : VOP1Inst <vop1<0x17>, "v_trunc_f64",
  VOP_F64_F64, ftrunc
>;
defm V_CEIL_F64 : VOP1Inst <vop1<0x18>, "v_ceil_f64",
  VOP_F64_F64, fceil
>;
defm V_FLOOR_F64 : VOP1Inst <vop1<0x1A>, "v_floor_f64",
  VOP_F64_F64, ffloor
>;
defm V_RNDNE_F64 : VOP1Inst <vop1<0x19>, "v_rndne_f64",
  VOP_F64_F64, frint
>;
} // End SchedRW = [WriteDoubleAdd]

let SchedRW = [WriteQuarterRate32] in {
defm V_LOG_LEGACY_F32 : VOP1Inst <vop1<0x45, 0x4c>, "v_log_legacy_f32",
  VOP_F32_F32
>;
defm V_EXP_LEGACY_F32 : VOP1Inst <vop1<0x46, 0x4b>, "v_exp_legacy_f32",
  VOP_F32_F32
>;
} // End SchedRW = [WriteQuarterRate32]

//===----------------------------------------------------------------------===//
// VOP3 Instructions
//===----------------------------------------------------------------------===//

defm V_QSAD_PK_U16_U8 : VOP3Inst <vop3<0x173>, "v_qsad_pk_u16_u8",
  VOP_I32_I32_I32
>;
defm V_MQSAD_U16_U8 : VOP3Inst <vop3<0x172>, "v_mqsad_u16_u8",
  VOP_I32_I32_I32
>;
defm V_MQSAD_U32_U8 : VOP3Inst <vop3<0x175>, "v_mqsad_u32_u8",
  VOP_I32_I32_I32
>;

let isCommutable = 1 in {
defm V_MAD_U64_U32 : VOP3Inst <vop3<0x176>, "v_mad_u64_u32",
  VOP_I64_I32_I32_I64
>;

// XXX - Does this set VCC?
defm V_MAD_I64_I32 : VOP3Inst <vop3<0x177>, "v_mad_i64_i32",
  VOP_I64_I32_I32_I64
>;
} // End isCommutable = 1


//===----------------------------------------------------------------------===//
// DS Instructions
//===----------------------------------------------------------------------===//
defm DS_WRAP_RTN_F32 : DS_1A1D_RET <0x34, "ds_wrap_rtn_f32", VGPR_32, "ds_wrap_f32">;

// DS_CONDXCHG32_RTN_B64
// DS_CONDXCHG32_RTN_B128

//===----------------------------------------------------------------------===//
// SMRD Instructions
//===----------------------------------------------------------------------===//

defm S_DCACHE_INV_VOL : SMRD_Inval <smrd<0x1d, 0x22>,
  "s_dcache_inv_vol", int_amdgcn_s_dcache_inv_vol>;

//===----------------------------------------------------------------------===//
// MUBUF Instructions
//===----------------------------------------------------------------------===//

defm BUFFER_WBINVL1_VOL : MUBUF_Invalidate <mubuf<0x70, 0x3f>,
  "buffer_wbinvl1_vol", int_amdgcn_buffer_wbinvl1_vol
>;

//===----------------------------------------------------------------------===//
// Flat Instructions
//===----------------------------------------------------------------------===//

def FLAT_LOAD_UBYTE : FLAT_Load_Helper <0x8, "flat_load_ubyte", VGPR_32>;
def FLAT_LOAD_SBYTE : FLAT_Load_Helper <0x9, "flat_load_sbyte", VGPR_32>;
def FLAT_LOAD_USHORT : FLAT_Load_Helper <0xa, "flat_load_ushort", VGPR_32>;
def FLAT_LOAD_SSHORT : FLAT_Load_Helper <0xb, "flat_load_sshort", VGPR_32>;
def FLAT_LOAD_DWORD : FLAT_Load_Helper <0xc, "flat_load_dword", VGPR_32>;
def FLAT_LOAD_DWORDX2 : FLAT_Load_Helper <0xd, "flat_load_dwordx2", VReg_64>;
def FLAT_LOAD_DWORDX4 : FLAT_Load_Helper <0xe, "flat_load_dwordx4", VReg_128>;
def FLAT_LOAD_DWORDX3 : FLAT_Load_Helper <0xf, "flat_load_dwordx3", VReg_96>;
def FLAT_STORE_BYTE : FLAT_Store_Helper <0x18, "flat_store_byte", VGPR_32>;
def FLAT_STORE_SHORT : FLAT_Store_Helper <0x1a, "flat_store_short", VGPR_32>;
def FLAT_STORE_DWORD : FLAT_Store_Helper <0x1c, "flat_store_dword", VGPR_32>;
def FLAT_STORE_DWORDX2 : FLAT_Store_Helper <
  0x1d, "flat_store_dwordx2", VReg_64
>;
def FLAT_STORE_DWORDX4 : FLAT_Store_Helper <
  0x1e, "flat_store_dwordx4", VReg_128
>;
def FLAT_STORE_DWORDX3 : FLAT_Store_Helper <
  0x1f, "flat_store_dwordx3", VReg_96
>;
defm FLAT_ATOMIC_SWAP : FLAT_ATOMIC <0x30, "flat_atomic_swap", VGPR_32>;
defm FLAT_ATOMIC_CMPSWAP : FLAT_ATOMIC <
  0x31, "flat_atomic_cmpswap", VGPR_32, VReg_64
>;
defm FLAT_ATOMIC_ADD : FLAT_ATOMIC <0x32, "flat_atomic_add", VGPR_32>;
defm FLAT_ATOMIC_SUB : FLAT_ATOMIC <0x33, "flat_atomic_sub", VGPR_32>;
defm FLAT_ATOMIC_RSUB : FLAT_ATOMIC <0x34, "flat_atomic_rsub", VGPR_32>;
defm FLAT_ATOMIC_SMIN : FLAT_ATOMIC <0x35, "flat_atomic_smin", VGPR_32>;
defm FLAT_ATOMIC_UMIN : FLAT_ATOMIC <0x36, "flat_atomic_umin", VGPR_32>;
defm FLAT_ATOMIC_SMAX : FLAT_ATOMIC <0x37, "flat_atomic_smax", VGPR_32>;
defm FLAT_ATOMIC_UMAX : FLAT_ATOMIC <0x38, "flat_atomic_umax", VGPR_32>;
defm FLAT_ATOMIC_AND : FLAT_ATOMIC <0x39, "flat_atomic_and", VGPR_32>;
defm FLAT_ATOMIC_OR : FLAT_ATOMIC <0x3a, "flat_atomic_or", VGPR_32>;
defm FLAT_ATOMIC_XOR : FLAT_ATOMIC <0x3b, "flat_atomic_xor", VGPR_32>;
defm FLAT_ATOMIC_INC : FLAT_ATOMIC <0x3c, "flat_atomic_inc", VGPR_32>;
defm FLAT_ATOMIC_DEC : FLAT_ATOMIC <0x3d, "flat_atomic_dec", VGPR_32>;
defm FLAT_ATOMIC_FCMPSWAP : FLAT_ATOMIC <
  0x3e, "flat_atomic_fcmpswap", VGPR_32, VReg_64
>;
defm FLAT_ATOMIC_FMIN : FLAT_ATOMIC <0x3f, "flat_atomic_fmin", VGPR_32>;
defm FLAT_ATOMIC_FMAX : FLAT_ATOMIC <0x40, "flat_atomic_fmax", VGPR_32>;
defm FLAT_ATOMIC_SWAP_X2 : FLAT_ATOMIC <0x50, "flat_atomic_swap_x2", VReg_64>;
defm FLAT_ATOMIC_CMPSWAP_X2 : FLAT_ATOMIC <
  0x51, "flat_atomic_cmpswap_x2", VReg_64, VReg_128
>;
defm FLAT_ATOMIC_ADD_X2 : FLAT_ATOMIC <0x52, "flat_atomic_add_x2", VReg_64>;
defm FLAT_ATOMIC_SUB_X2 : FLAT_ATOMIC <0x53, "flat_atomic_sub_x2", VReg_64>;
defm FLAT_ATOMIC_RSUB_X2 : FLAT_ATOMIC <0x54, "flat_atomic_rsub_x2", VReg_64>;
defm FLAT_ATOMIC_SMIN_X2 : FLAT_ATOMIC <0x55, "flat_atomic_smin_x2", VReg_64>;
defm FLAT_ATOMIC_UMIN_X2 : FLAT_ATOMIC <0x56, "flat_atomic_umin_x2", VReg_64>;
defm FLAT_ATOMIC_SMAX_X2 : FLAT_ATOMIC <0x57, "flat_atomic_smax_x2", VReg_64>;
defm FLAT_ATOMIC_UMAX_X2 : FLAT_ATOMIC <0x58, "flat_atomic_umax_x2", VReg_64>;
defm FLAT_ATOMIC_AND_X2 : FLAT_ATOMIC <0x59, "flat_atomic_and_x2", VReg_64>;
defm FLAT_ATOMIC_OR_X2 : FLAT_ATOMIC <0x5a, "flat_atomic_or_x2", VReg_64>;
defm FLAT_ATOMIC_XOR_X2 : FLAT_ATOMIC <0x5b, "flat_atomic_xor_x2", VReg_64>;
defm FLAT_ATOMIC_INC_X2 : FLAT_ATOMIC <0x5c, "flat_atomic_inc_x2", VReg_64>;
defm FLAT_ATOMIC_DEC_X2 : FLAT_ATOMIC <0x5d, "flat_atomic_dec_x2", VReg_64>;
defm FLAT_ATOMIC_FCMPSWAP_X2 : FLAT_ATOMIC <
  0x5e, "flat_atomic_fcmpswap_x2", VReg_64, VReg_128
>;
defm FLAT_ATOMIC_FMIN_X2 : FLAT_ATOMIC <0x5f, "flat_atomic_fmin_x2", VReg_64>;
defm FLAT_ATOMIC_FMAX_X2 : FLAT_ATOMIC <0x60, "flat_atomic_fmax_x2", VReg_64>;

} // End SubtargetPredicate = isCIVI

//===----------------------------------------------------------------------===//
// Flat Patterns
//===----------------------------------------------------------------------===//

let Predicates = [HasFlatAddressSpace] in {

class FLATLoad_Pattern <FLAT Instr_ADDR64, ValueType vt,
                             PatFrag flat_ld> :
  Pat <(vt (flat_ld i64:$ptr)),
       (Instr_ADDR64 $ptr, 0, 0, 0)
>;

def : FLATLoad_Pattern <FLAT_LOAD_SBYTE, i32, sextloadi8_flat>;
def : FLATLoad_Pattern <FLAT_LOAD_UBYTE, i32, az_extloadi8_flat>;
def : FLATLoad_Pattern <FLAT_LOAD_SSHORT, i32, sextloadi16_flat>;
def : FLATLoad_Pattern <FLAT_LOAD_USHORT, i32, az_extloadi16_flat>;
def : FLATLoad_Pattern <FLAT_LOAD_DWORD, i32, flat_load>;
def : FLATLoad_Pattern <FLAT_LOAD_DWORDX2, i64, flat_load>;
def : FLATLoad_Pattern <FLAT_LOAD_DWORDX2, i64, az_extloadi32_flat>;
def : FLATLoad_Pattern <FLAT_LOAD_DWORDX2, v2i32, flat_load>;
def : FLATLoad_Pattern <FLAT_LOAD_DWORDX4, v4i32, flat_load>;

class FLATStore_Pattern <FLAT Instr, ValueType vt, PatFrag st> :
  Pat <(st vt:$value, i64:$ptr),
        (Instr $value, $ptr, 0, 0, 0)
  >;

def : FLATStore_Pattern <FLAT_STORE_BYTE, i32, truncstorei8_flat>;
def : FLATStore_Pattern <FLAT_STORE_SHORT, i32, truncstorei16_flat>;
def : FLATStore_Pattern <FLAT_STORE_DWORD, i32, flat_store>;
def : FLATStore_Pattern <FLAT_STORE_DWORDX2, i64, flat_store>;
def : FLATStore_Pattern <FLAT_STORE_DWORDX2, v2i32, flat_store>;
def : FLATStore_Pattern <FLAT_STORE_DWORDX4, v4i32, flat_store>;

} // End HasFlatAddressSpace predicate

let Predicates = [isCI] in {

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

} // End Predicates = [isCI]

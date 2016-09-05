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
// BUFFER_LOAD_DWORDX3
// BUFFER_STORE_DWORDX3

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

defm V_MQSAD_U16_U8 : VOP3Inst <vop3<0x172>, "v_mqsad_u16_u8",
  VOP_I32_I32_I32
>;

defm V_QSAD_PK_U16_U8 : VOP3Inst <vop3<0x172, 0x1e5>, "v_qsad_pk_u16_u8",
  VOP_I64_I64_I32_I64, int_amdgcn_qsad_pk_u16_u8>;

defm V_MQSAD_U32_U8 : VOP3Inst <vop3<0x174, 0x1e7>, "v_mqsad_u32_u8",
  VOP_I32_I32_I32_I32, int_amdgcn_mqsad_u32_u8>;

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
// MUBUF Instructions
//===----------------------------------------------------------------------===//

let DisableSIDecoder = 1 in {
defm BUFFER_WBINVL1_VOL : MUBUF_Invalidate <mubuf<0x70, 0x3f>,
  "buffer_wbinvl1_vol", int_amdgcn_buffer_wbinvl1_vol
>;
}

} // End SubtargetPredicate = isCIVI

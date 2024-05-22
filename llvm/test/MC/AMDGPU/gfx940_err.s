// RUN: not llvm-mc -triple=amdgcn -mcpu=gfx940 %s 2>&1 | FileCheck --check-prefix=GFX940 --implicit-check-not=error: %s

v_mac_f32 v0, v1, v2
// GFX940: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_mac_f32_e64 v5, v1, v2
// GFX940: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_mac_f32_dpp v5, v1, v2 quad_perm:[0,1,2,3] row_mask:0xf bank_mask:0xf
// GFX940: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_mac_f32_dpp v5, v1, v2 dpp8:[7,6,5,4,3,2,1,0]
// GFX940: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_mac_f32_sdwa v5, v1, v2 dst_sel:DWORD dst_unused:UNUSED_PRESERVE src0_sel:DWORD src1_sel:DWORD
// GFX940: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_mad_f32 v0, v1, v2, v3
// GFX940: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_madak_f32 v0, v1, v2, 0
// GFX940: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_madmk_f32 v0, v1, 0, v2
// GFX940: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_mad_legacy_f32 v0, v1, v2, v3
// GFX940: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_mov_b64 v[2:3], v[4:5] row_shl:1
// GFX940: :[[@LINE-1]]:{{[0-9]+}}: error: DP ALU dpp only supports row_newbcast

v_mov_b64 v[2:3], -v[4:5]
// GFX940: :[[@LINE-1]]:{{[0-9]+}}: error: not a valid operand.

v_mov_b64 v[2:3], |v[4:5]|
// GFX940: :[[@LINE-1]]:{{[0-9]+}}: error: not a valid operand.

v_mov_b64 v[2:3], v[4:5] dst_sel:BYTE_0 dst_unused:UNUSED_PRESERVE src0_sel:DWORD
// GFX940: :[[@LINE-1]]:{{[0-9]+}}: error: not a valid operand.

v_mov_b64_sdwa v[2:3], v[4:5]
// GFX940: :[[@LINE-1]]:{{[0-9]+}}: error: sdwa variant of this instruction is not supported

buffer_invl2
// GFX940: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

global_load_dword v2, v[2:3], off glc
// GFX940: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

global_load_dword v2, v[2:3], off slc
// GFX940: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

global_load_dword v2, v[2:3], off scc
// GFX940: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

s_load_dword s2, s[2:3], 0x0 sc0
// GFX940: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

buffer_atomic_swap v5, off, s[8:11], s3 glc
// GFX940: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

buffer_atomic_swap v5, off, s[8:11], s3 slc
// GFX940: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

buffer_wbl2 glc
// GFX940: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

buffer_wbl2 scc
// GFX940: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

v_dot2_u32_u16 v0, 1, v0, s2 op_sel:[0,1,0,1] op_sel_hi:[0,0,1,1]
// GFX940: :[[@LINE-1]]:{{[0-9]+}}: error: invalid op_sel operand

v_cvt_f32_fp8 v1, sext(v3) src0_sel:BYTE_1
// GFX940: :[[@LINE-1]]:{{[0-9]+}}: error: not a valid operand.

v_cvt_pk_f32_bf8 v[2:3], sext(v3) src0_sel:BYTE_1
// GFX940: :[[@LINE-1]]:{{[0-9]+}}: error: not a valid operand.

v_cvt_sr_bf8_f32 v1, v2, -v3
// GFX940: :[[@LINE-1]]:{{[0-9]+}}: error: not a valid operand.

v_cvt_sr_fp8_f32 v1, v2, -v3
// GFX940: :[[@LINE-1]]:{{[0-9]+}}: error: not a valid operand.

v_cvt_sr_fp8_f32 v1, v2, v3 clamp
// GFX940: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

v_cvt_sr_fp8_f32 v1, v2, v3 mul:2
// GFX940: :[[@LINE-1]]:{{[0-9]+}}: error: not a valid operand.

v_cvt_pk_fp8_f32 v1, v2, v3 clamp
// GFX940: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

v_cvt_pk_fp8_f32 v1, v2, v3 mul:2
// GFX940: :[[@LINE-1]]:{{[0-9]+}}: error: not a valid operand.

s_getreg_b32 s1, hwreg(HW_REG_FLAT_SCR_LO)
// GFX940: :[[@LINE-1]]:{{[0-9]+}}: error: specified hardware register is not supported on this GPU

s_getreg_b32 s1, hwreg(HW_REG_FLAT_SCR_HI)
// GFX940: :[[@LINE-1]]:{{[0-9]+}}: error: specified hardware register is not supported on this GPU

s_getreg_b32 s1, hwreg(HW_REG_XNACK_MASK)
// GFX940: :[[@LINE-1]]:{{[0-9]+}}: error: specified hardware register is not supported on this GPU

s_getreg_b32 s1, hwreg(HW_REG_HW_ID1)
// GFX940: :[[@LINE-1]]:{{[0-9]+}}: error: specified hardware register is not supported on this GPU

s_getreg_b32 s1, hwreg(HW_REG_HW_ID2)
// GFX940: :[[@LINE-1]]:{{[0-9]+}}: error: specified hardware register is not supported on this GPU

s_getreg_b32 s1, hwreg(HW_REG_POPS_PACKER)
// GFX940: :[[@LINE-1]]:{{[0-9]+}}: error: specified hardware register is not supported on this GPU

ds_ordered_count v5, v1 offset:65535 gds
// GFX940: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

exp pos0 v3, v2, v1, v0
// GFX940: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

global_load_dword v[2:3], off lds
// GFX940: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

scratch_load_dword v2, off lds
// GFX940: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

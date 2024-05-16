// RUN: not llvm-mc -triple=amdgcn -mcpu=gfx1200 -show-encoding %s 2>&1 | FileCheck --check-prefixes=GFX12-ERR --implicit-check-not=error: -strict-whitespace %s

v_cubesc_f32_e64_dpp v5, v1, v2, 12345678 row_shr:4 row_mask:0xf bank_mask:0xf
// GFX12-ERR: [[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

v_add3_u32_e64_dpp v5, v1, v2, 49812340 dpp8:[7,6,5,4,3,2,1,0]
// GFX12-ERR: [[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

v_cvt_f32_i32_e64_dpp v5, s1 dpp8:[7,6,5,4,3,2,1,0]
// GFX12-ERR: [[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

v_cvt_f32_i32_e64_dpp v5, s1 row_shl:15 row_mask:0xf bank_mask:0xf
// GFX12-ERR: [[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

v_cvt_f16_u16_e64_dpp v5, s1 dpp8:[7,6,5,4,3,2,1,0]
// GFX12-ERR: [[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

v_cvt_f16_u16_e64_dpp v5, s1 row_shl:1 row_mask:0xf bank_mask:0xf
// GFX12-ERR: [[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

; disallow space between colons
v_dual_mul_f32 v0, v0, v2 : : v_dual_mul_f32 v1, v1, v3
// GFX12-ERR: [[@LINE-1]]:{{[0-9]+}}: error: unknown token in expression

// On GFX12, v_dot8_i32_i4 is a valid SP3 alias for v_dot8_i32_iu4.
// However, we intentionally leave it unimplemented because on other
// processors v_dot8_i32_i4 denotes an instruction of a different
// behaviour, which is considered potentially dangerous.
v_dot8_i32_i4 v0, v1, v2, v3
// GFX12-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

// On GFX12, v_dot4_i32_i8 is a valid SP3 alias for v_dot4_i32_iu8.
// However, we intentionally leave it unimplemented because on other
// processors v_dot4_i32_i8 denotes an instruction of a different
// behaviour, which is considered potentially dangerous.
v_dot4_i32_i8 v0, v1, v2, v3
// GFX12-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_dot4c_i32_i8 v0, v1, v2
// GFX12-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_cmp_class_f16_e64_dpp s105, s2, v2 row_ror:15
// GFX12-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmpx_class_f32_e64_dpp s1, v2 dpp8:[7,6,5,4,3,2,1,0] fi:1
// GFX12-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

v_fma_mix_f32_e64_dpp v5, s1, v3, v4 quad_perm:[3,2,1,0]
// GFX12-ERR: [[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

v_fma_mix_f32_e64_dpp v5, v1, s3, v4 quad_perm:[3,2,1,0]
// GFX12-ERR: [[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

v_fma_mix_f32_e64_dpp v5, s1, v3, v4 dpp8:[7,6,5,4,3,2,1,0]
// GFX12-ERR: [[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

v_fma_mix_f32_e64_dpp v5, v1, s3, v4 dpp8:[7,6,5,4,3,2,1,0]
// GFX12-ERR: [[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

v_fma_mixhi_f16_e64_dpp v5, v1, 0, v4 quad_perm:[3,2,1,0]
// GFX12-ERR: [[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

v_fma_mixlo_f16_e64_dpp v5, v1, 1, v4 dpp8:[7,6,5,4,3,2,1,0]
// GFX12-ERR: [[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

v_lshlrev_b64 v[5:6], s2, s[0:1]
// GFX12-ERR: [[@LINE-1]]:{{[0-9]+}}: error: invalid operand (violates constant bus restrictions)

v_lshrrev_b64 v[5:6], s2, s[0:1]
// GFX12-ERR: [[@LINE-1]]:{{[0-9]+}}: error: invalid operand (violates constant bus restrictions)

v_ashrrev_i64 v[5:6], s2, s[0:1]
// GFX12-ERR: [[@LINE-1]]:{{[0-9]+}}: error: invalid operand (violates constant bus restrictions)

image_load v0, v0, s[0:7] dmask:0x1 dim:SQ_RSRC_IMG_1D th:0x7
// GFX12-ERR: [[@LINE-1]]:{{[0-9]+}}: error: expected an identifier

image_load v0, v0, s[0:7] dmask:0x1 dim:SQ_RSRC_IMG_1D th:TH_STORE_NT
// GFX12-ERR: [[@LINE-1]]:{{[0-9]+}}: error: invalid th value for load instructions

image_load v0, v0, s[0:7] dmask:0x1 dim:SQ_RSRC_IMG_1D th:TH_ATOMIC_NT
// GFX12-ERR: [[@LINE-1]]:{{[0-9]+}}: error: invalid th value for load instructions

image_store v0, v0, s[0:7] dmask:0x1 dim:SQ_RSRC_IMG_1D th:TH_LOAD_NT
// GFX12-ERR: [[@LINE-1]]:{{[0-9]+}}: error: invalid th value for store instructions

image_store v0, v0, s[0:7] dmask:0x1 dim:SQ_RSRC_IMG_1D th:TH_ATOMIC_NT
// GFX12-ERR: [[@LINE-1]]:{{[0-9]+}}: error: invalid th value for store instructions

image_atomic_swap v0, v0, s[0:7] dmask:0x1 dim:SQ_RSRC_IMG_1D th:TH_LOAD_NT
// GFX12-ERR: [[@LINE-1]]:{{[0-9]+}}: error: invalid th value for atomic instructions

image_atomic_swap v0, v0, s[0:7] dmask:0x1 dim:SQ_RSRC_IMG_1D th:TH_STORE_NT
// GFX12-ERR: [[@LINE-1]]:{{[0-9]+}}: error: invalid th value for atomic instructions

image_store v0, v0, s[0:7] dmask:0x1 dim:SQ_RSRC_IMG_1D th:TH_STORE_LU
// GFX12-ERR: [[@LINE-1]]:{{[0-9]+}}: error: invalid th value

image_load v0, v0, s[0:7] dmask:0x1 dim:SQ_RSRC_IMG_1D th:TH_LOAD_RT_WB
// GFX12-ERR: [[@LINE-1]]:{{[0-9]+}}: error: invalid th value

image_load v0, v0, s[0:7] dmask:0x1 dim:SQ_RSRC_IMG_1D th:TH_LOAD_NT_WB
// GFX12-ERR: [[@LINE-1]]:{{[0-9]+}}: error: invalid th value

image_store v0, v0, s[0:7] dmask:0x1 dim:SQ_RSRC_IMG_1D th:TH_STORE_RT_WB scope:SCOPE_SYS
// GFX12-ERR: [[@LINE-1]]:{{[0-9]+}}: error: scope and th combination is not valid

image_store v0, v0, s[0:7] dmask:0x1 dim:SQ_RSRC_IMG_1D th:TH_STORE_BYPASS scope:SCOPE_DEV
// GFX12-ERR: [[@LINE-1]]:{{[0-9]+}}: error: scope and th combination is not valid

s_load_b32 s5, s[4:5], s0 offset:0x0 th:TH_LOAD_NT_RT
// GFX12-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: invalid th value for SMEM instruction

s_buffer_load_b64 s[10:11], s[4:7], s0 offset:0x0 th:TH_LOAD_RT_NT
// GFX12-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: invalid th value for SMEM instruction

s_load_b128 s[20:23], s[2:3], vcc_lo th:TH_LOAD_NT_HT
// GFX12-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: invalid th value for SMEM instruction

image_load v0, v0, s[0:7] dmask:0x1 dim:SQ_RSRC_IMG_1D th:TH_LOAD_HT scope:SCOPE_SE th:TH_LOAD_HT
// GFX12-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: not a valid operand

image_load v0, v0, s[0:7] dmask:0x1 dim:SQ_RSRC_IMG_1D scope:SCOPE_SE th:TH_LOAD_HT scope:SCOPE_SE
// GFX12-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: not a valid operand

s_prefetch_inst s[14:15], 0xffffff, m0, 7
// GFX12-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: expected a 24-bit signed offset
// GFX12-ERR: s_prefetch_inst s[14:15], 0xffffff, m0, 7
// GFX12-ERR:                           ^

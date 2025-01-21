// RUN: not llvm-mc -triple=amdgcn -mcpu=gfx1200 %s 2>&1 | FileCheck --check-prefixes=NOGFX12 --implicit-check-not=error: %s

buffer_atomic_add_f32 v5, v0, null, s3 idxen
// NOGFX12: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

buffer_atomic_add_u32 v5, v0, null, s3 idxen
// NOGFX12: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

buffer_atomic_add_u64 v[5:6], v0, null, s3 idxen
// NOGFX12: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

buffer_atomic_and_b32 v5, v0, null, s3 idxen
// NOGFX12: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

buffer_atomic_and_b64 v[5:6], v0, null, s3 idxen
// NOGFX12: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

buffer_atomic_cmpswap_b32 v[5:6], v0, null, s3 idxen
// NOGFX12: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

buffer_atomic_cmpswap_b64 v[5:8], v0, null, s3 idxen
// NOGFX12: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

buffer_atomic_cond_sub_u32 v5, v0, null, s3 idxen
// NOGFX12: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

buffer_atomic_dec_u32 v5, v0, null, s3 idxen
// NOGFX12: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

buffer_atomic_dec_u64 v[5:6], v0, null, s3 idxen
// NOGFX12: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

buffer_atomic_inc_u32 v5, v0, null, s3 idxen
// NOGFX12: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

buffer_atomic_inc_u64 v[5:6], v0, null, s3 idxen
// NOGFX12: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

buffer_atomic_max_i32 v5, v0, null, s3 idxen
// NOGFX12: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

buffer_atomic_max_i64 v[5:6], v0, null, s3 idxen
// NOGFX12: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

buffer_atomic_max_num_f32 v5, v0, null, s3 idxen
// NOGFX12: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

buffer_atomic_max_u32 v5, v0, null, s3 idxen
// NOGFX12: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

buffer_atomic_max_u64 v[5:6], v0, null, s3 idxen
// NOGFX12: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

buffer_atomic_min_i32 v5, v0, null, s3 idxen
// NOGFX12: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

buffer_atomic_min_i64 v[5:6], v0, null, s3 idxen
// NOGFX12: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

buffer_atomic_min_u32 v5, v0, null, s3 idxen
// NOGFX12: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

buffer_atomic_min_u64 v[5:6], v0, null, s3 idxen
// NOGFX12: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

buffer_atomic_min_num_f32 v5, v0, null, s3 idxen
// NOGFX12: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

buffer_atomic_or_b32 v5, v0, null, s3 idxen
// NOGFX12: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

buffer_atomic_or_b64 v[5:6], v0, null, s3 idxen
// NOGFX12: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

buffer_atomic_pk_add_bf16 v5, v0, null, s3 idxen
// NOGFX12: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

buffer_atomic_pk_add_f16 v5, v0, null, s3 idxen
// NOGFX12: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

buffer_atomic_sub_clamp_u32 v5, v0, null, s3 idxen
// NOGFX12: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

buffer_atomic_sub_u32 v5, v0, null, s3 idxen
// NOGFX12: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

buffer_atomic_sub_u64 v[5:6], v0, null, s3 idxen
// NOGFX12: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

buffer_atomic_swap_b32 v5, v0, null, s3 idxen
// NOGFX12: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

buffer_atomic_swap_b64 v[5:6], v0, null, s3 idxen
// NOGFX12: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

buffer_atomic_xor_b32 v5, v0, null, s3 idxen
// NOGFX12: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

buffer_atomic_xor_b64 v[5:6], v0, null, s3 idxen
// NOGFX12: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

buffer_load_b128 v[5:8], v0, null, s3 offen offset:4095
// NOGFX12: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

buffer_load_b32 v5, v0, null, s3 offen offset:4095
// NOGFX12: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

buffer_store_b64 v[1:2], v0, null, s4 idxen offset:4095
// NOGFX12: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

buffer_store_b96 v[1:3], v0, null, s4 idxen offset:4095
// NOGFX12: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

buffer_load_d16_b16 v5, v0, null, s3 offen offset:4095
// NOGFX12: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

buffer_load_d16_format_x v3, v0, null, s1 offen offset:4095
// NOGFX12: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

buffer_load_d16_format_xy v3, v0, null, s1 offen offset:4095
// NOGFX12: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

buffer_load_d16_format_xyz v[3:4], v0, null, s1 offen offset:4095
// NOGFX12: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

buffer_load_d16_format_xyzw v[3:4], v0, null, s1 offen offset:4095
// NOGFX12: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

buffer_load_d16_hi_b16 v3, v0, null, s1 offen offset:4095
// NOGFX12: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

buffer_load_d16_hi_format_x v3, v0, null, s1 offen offset:4095
// NOGFX12: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

buffer_load_d16_hi_i8 v3, v0, null, s1 offen offset:4095
// NOGFX12: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

buffer_load_d16_hi_u8 v3, v0, null, s1 offen offset:4095
// NOGFX12: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

buffer_load_d16_i8 v3, v0, null, s1 offen offset:4095
// NOGFX12: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

buffer_load_d16_u8 v3, v0, null, s1 offen offset:4095
// NOGFX12: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

buffer_load_format_x v3, v0, null, s1 offen offset:4095
// NOGFX12: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

buffer_load_format_xy v[3:4], v0, null, s1 offen offset:4095
// NOGFX12: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

buffer_load_format_xyz v[3:5], v0, null, s1 offen offset:4095
// NOGFX12: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

buffer_load_format_xyzw v[3:6], v0, null, s1 offen offset:4095
// NOGFX12: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

buffer_load_i16 v3, v0, null, s1 offen offset:4095
// NOGFX12: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

buffer_load_i8 v3, v0, null, s1 offen offset:4095
// NOGFX12: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

buffer_load_u16 v3, v0, null, s1 offen offset:4095
// NOGFX12: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

buffer_load_u8 v3, v0, null, s1 offen offset:4095
// NOGFX12: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

buffer_store_b128 v[3:6], v0, null, s1 offen offset:4095
// NOGFX12: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

buffer_store_b16 v3, v0, null, s1 offen offset:4095
// NOGFX12: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

buffer_store_b32 v3, v0, null, s1 offen offset:4095
// NOGFX12: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

buffer_store_b64 v[3:4], v0, null, s1 offen offset:4095
// NOGFX12: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

buffer_store_b8 v3, v0, null, s1 offen offset:4095
// NOGFX12: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

buffer_store_b96 v[3:5], v0, null, s1 offen offset:4095
// NOGFX12: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

buffer_store_d16_format_x v3, v0, null, s1 offen offset:4095
// NOGFX12: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

buffer_store_d16_format_xy v3, v0, null, s1 offen offset:4095
// NOGFX12: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

buffer_store_d16_format_xyz v[3:4], v0, null, s1 offen offset:4095
// NOGFX12: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

buffer_store_d16_format_xyzw v[3:4], v0, null, s1 offen offset:4095
// NOGFX12: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

buffer_store_d16_hi_b16 v3, v0, null, s1 offen offset:4095
// NOGFX12: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

buffer_store_d16_hi_b8 v3, v0, null, s1 offen offset:4095
// NOGFX12: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

buffer_store_d16_hi_format_x v3, v0, null, s1 offen offset:4095
// NOGFX12: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

buffer_store_format_x v1, v0, null, s1 offen offset:4095
// NOGFX12: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

buffer_store_format_xy v[1:2], v0, null, s1 offen offset:4095
// NOGFX12: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

buffer_store_format_xyz v[1:3], v0, null, s1 offen offset:4095
// NOGFX12: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

buffer_store_format_xyzw v[1:4], v0, null, s1 offen offset:4095
// NOGFX12: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

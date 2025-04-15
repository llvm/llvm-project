// RUN: not llvm-mc -triple=amdgcn -mcpu=gfx1200 %s 2>&1 | FileCheck --check-prefixes=GFX12 --implicit-check-not=error: %s

global_atomic_add_f32 v0, v2, null
// GFX12: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

global_atomic_add_f32 v0, v2, v4, null glc
// GFX12: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

global_atomic_add_u32 v0, v2, null
// GFX12: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

global_atomic_add_u32 v0, v2, v4, null glc
// GFX12: :[[@LINE-1]]:35: error: invalid operand for instruction

global_atomic_add_u64 v0, v[2:3], null
// GFX12: :[[@LINE-1]]:35: error: invalid operand for instruction

global_atomic_add_u64 v[0:1], v2, v[4:5], null
// GFX12: :[[@LINE-1]]:43: error: invalid operand for instruction

global_atomic_and_b32 v0, v2, null
// GFX12: :[[@LINE-1]]:31: error: invalid operand for instruction

global_atomic_and_b32 v0, v2, v4, null
// GFX12: :[[@LINE-1]]:35: error: invalid operand for instruction

global_atomic_and_b64 v0, v[2:3], null
// GFX12: :[[@LINE-1]]:35: error: invalid operand for instruction

global_atomic_and_b64 v[0:1], v2, v[4:5], null
// GFX12: :[[@LINE-1]]:43: error: invalid operand for instruction

global_atomic_cmpswap_b32 v0, v[2:3], null
// GFX12: :[[@LINE-1]]:39: error: invalid operand for instruction

global_atomic_cmpswap_b32 v0, v2, v[4:5], null
// GFX12: :[[@LINE-1]]:43: error: invalid operand for instruction

global_atomic_cmpswap_b64 v0, v[2:5], null
// GFX12: :[[@LINE-1]]:39: error: invalid operand for instruction

global_atomic_cmpswap_b64 v[0:1], v2, v[4:7], null
// GFX12: :[[@LINE-1]]:47: error: invalid operand for instruction

global_atomic_cond_sub_u32 v0, v2, null
// GFX12: :[[@LINE-1]]:36: error: invalid operand for instruction

global_atomic_cond_sub_u32 v0, v2, v4, null
// GFX12: :[[@LINE-1]]:40: error: invalid operand for instruction

global_atomic_dec_u32 v0, v2, null
// GFX12: :[[@LINE-1]]:31: error: invalid operand for instruction

global_atomic_dec_u32 v0, v2, v4, null
// GFX12: :[[@LINE-1]]:35: error: invalid operand for instruction

global_atomic_dec_u64 v0, v[2:3], null
// GFX12: :[[@LINE-1]]:35: error: invalid operand for instruction

global_atomic_dec_u64 v[0:1], v2, v[4:5], null
// GFX12: :[[@LINE-1]]:43: error: invalid operand for instruction

global_atomic_inc_u32 v0, v2, null
// GFX12: :[[@LINE-1]]:31: error: invalid operand for instruction

global_atomic_inc_u32 v0, v2, v4, null
// GFX12: :[[@LINE-1]]:35: error: invalid operand for instruction

global_atomic_inc_u64 v0, v[2:3], null
// GFX12: :[[@LINE-1]]:35: error: invalid operand for instruction

global_atomic_inc_u64 v[0:1], v2, v[4:5], null
// GFX12: :[[@LINE-1]]:43: error: invalid operand for instruction

global_atomic_max_i32 v0, v2, null
// GFX12: :[[@LINE-1]]:31: error: invalid operand for instruction

global_atomic_max_i32 v0, v2, v4, null
// GFX12: :[[@LINE-1]]:35: error: invalid operand for instruction

global_atomic_max_i64 v0, v[2:3], null
// GFX12: :[[@LINE-1]]:35: error: invalid operand for instruction

global_atomic_max_i64 v[0:1], v2, v[4:5], null
// GFX12: :[[@LINE-1]]:43: error: invalid operand for instruction

global_atomic_max_num_f32 v0, v2, null
// GFX12: :[[@LINE-1]]:35: error: invalid operand for instruction

global_atomic_max_num_f32 v0, v2, v4, null
// GFX12: :[[@LINE-1]]:39: error: invalid operand for instruction

global_atomic_max_u32 v0, v2, null
// GFX12: :[[@LINE-1]]:31: error: invalid operand for instruction

global_atomic_max_u32 v0, v2, v4, null
// GFX12: :[[@LINE-1]]:35: error: invalid operand for instruction

global_atomic_max_u64 v0, v[2:3], null
// GFX12: :[[@LINE-1]]:35: error: invalid operand for instruction

global_atomic_max_u64 v[0:1], v2, v[4:5], null
// GFX12: :[[@LINE-1]]:43: error: invalid operand for instruction

global_atomic_min_i32 v0, v2, v4, null
// GFX12: :[[@LINE-1]]:35: error: invalid operand for instruction

global_atomic_min_i32 v0, v2, null
// GFX12: :[[@LINE-1]]:31: error: invalid operand for instruction

global_atomic_min_i64 v0, v[2:3], null
// GFX12: :[[@LINE-1]]:35: error: invalid operand for instruction

global_atomic_min_i64 v[0:1], v2, v[4:5], null
// GFX12: :[[@LINE-1]]:43: error: invalid operand for instruction

global_atomic_min_num_f32 v0, v2, null
// GFX12: :[[@LINE-1]]:35: error: invalid operand for instruction

global_atomic_min_num_f32 v0, v2, v4, null
// GFX12: :[[@LINE-1]]:39: error: invalid operand for instruction

global_atomic_min_u32 v0, v2, null
// GFX12: :[[@LINE-1]]:31: error: invalid operand for instruction

global_atomic_min_u32 v0, v2, v4, null
// GFX12: :[[@LINE-1]]:35: error: invalid operand for instruction

global_atomic_min_u64 v0, v[2:3], null
// GFX12: :[[@LINE-1]]:35: error: invalid operand for instruction

global_atomic_min_u64 v[0:1], v2, v[4:5], null
// GFX12: :[[@LINE-1]]:43: error: invalid operand for instruction

global_atomic_or_b32 v0, v2, null
// GFX12: :[[@LINE-1]]:30: error: invalid operand for instruction

global_atomic_or_b32 v0, v2, v4, null
// GFX12: :[[@LINE-1]]:34: error: invalid operand for instruction

global_atomic_or_b64 v0, v[2:3], null
// GFX12: :[[@LINE-1]]:34: error: invalid operand for instruction

global_atomic_or_b64 v[0:1], v2, v[4:5], null
// GFX12: :[[@LINE-1]]:42: error: invalid operand for instruction

global_atomic_ordered_add_b64 v0, v[2:3], null
// GFX12: :[[@LINE-1]]:43: error: invalid operand for instruction

global_atomic_ordered_add_b64 v[0:1], v2, v[4:5], null
// GFX12: :[[@LINE-1]]:51: error: invalid operand for instruction

global_atomic_pk_add_bf16 v0, v2, null
// GFX12: :[[@LINE-1]]:35: error: invalid operand for instruction

global_atomic_pk_add_bf16 v0, v2, v4, null
// GFX12: :[[@LINE-1]]:39: error: invalid operand for instruction

global_atomic_pk_add_f16 v0, v2, null
// GFX12: :[[@LINE-1]]:34: error: invalid operand for instruction

global_atomic_pk_add_f16 v0, v2, v4, null
// GFX12: :[[@LINE-1]]:38: error: invalid operand for instruction

global_atomic_sub_clamp_u32 v0, v2, null
// GFX12: :[[@LINE-1]]:37: error: invalid operand for instruction

global_atomic_sub_clamp_u32 v0, v2, v4, null
// GFX12: :[[@LINE-1]]:41: error: invalid operand for instruction

global_atomic_sub_u32 v0, v2, null
// GFX12: :[[@LINE-1]]:31: error: invalid operand for instruction

global_atomic_sub_u32 v0, v2, v4, null
// GFX12: :[[@LINE-1]]:35: error: invalid operand for instruction

global_atomic_sub_u64 v0, v[2:3], null
// GFX12: :[[@LINE-1]]:35: error: invalid operand for instruction

global_atomic_sub_u64 v[0:1], v2, v[4:5], null
// GFX12: :[[@LINE-1]]:43: error: invalid operand for instruction

global_atomic_swap_b32 v0, v2, null
// GFX12: :[[@LINE-1]]:32: error: invalid operand for instruction

global_atomic_swap_b32 v0, v2, v4, null
// GFX12: :[[@LINE-1]]:36: error: invalid operand for instruction

global_atomic_swap_b64 v0, v[2:3], null
// GFX12: :[[@LINE-1]]:36: error: invalid operand for instruction

global_atomic_swap_b64 v[0:1], v2, v[4:5], null
// GFX12: :[[@LINE-1]]:44: error: invalid operand for instruction

global_atomic_xor_b32 v0, v2, null
// GFX12: :[[@LINE-1]]:31: error: invalid operand for instruction

global_atomic_xor_b32 v0, v2, v4, null
// GFX12: :[[@LINE-1]]:35: error: invalid operand for instruction

global_atomic_xor_b64 v0, v[2:3], null
// GFX12: :[[@LINE-1]]:35: error: invalid operand for instruction

global_atomic_xor_b64 v[0:1], v2, v[4:5], null
// GFX12: :[[@LINE-1]]:43: error: invalid operand for instruction

global_load_b128 v[0:3], v4, null
// GFX12: :[[@LINE-1]]:30: error: invalid operand for instruction

global_load_b32 v0, v4, null
// GFX12: :[[@LINE-1]]:25: error: invalid operand for instruction

global_load_b64 v[0:1], v4, null
// GFX12: :[[@LINE-1]]:29: error: invalid operand for instruction

global_load_b96 v[0:2], v4, null
// GFX12: :[[@LINE-1]]:29: error: invalid operand for instruction

global_load_block v[0:31], v32, null
// GFX12: :[[@LINE-1]]:33: error: invalid operand for instruction

global_load_d16_b16 v0, v2, null
// GFX12: :[[@LINE-1]]:29: error: invalid operand for instruction

global_load_d16_hi_b16 v0, v2, null
// GFX12: :[[@LINE-1]]:32: error: invalid operand for instruction

global_load_d16_hi_i8 v0, v2, null
// GFX12: :[[@LINE-1]]:31: error: invalid operand for instruction

global_load_d16_hi_u8 v0, v2, null
// GFX12: :[[@LINE-1]]:31: error: invalid operand for instruction

global_load_d16_i8 v0, v2, null
// GFX12: :[[@LINE-1]]:28: error: invalid operand for instruction

global_load_d16_u8 v0, v2, null
// GFX12: :[[@LINE-1]]:28: error: invalid operand for instruction

global_load_i16 v0, v2, null
// GFX12: :[[@LINE-1]]:25: error: invalid operand for instruction

global_load_i8 v0, v2, null
// GFX12: :[[@LINE-1]]:24: error: invalid operand for instruction

global_load_tr_b128 v[0:3], v4, null
// GFX12: :[[@LINE-1]]:33: error: invalid operand for instruction

global_load_tr_b128 v[0:1], v4, null
// GFX12: :[[@LINE-1]]:33: error: invalid operand for instruction

global_load_tr_b64 v[0:1], v4, null
// GFX12: :[[@LINE-1]]:32: error: invalid operand for instruction

global_load_tr_b64 v0, v4, null
// GFX12: :[[@LINE-1]]:28: error: invalid operand for instruction

global_load_u16 v0, v2, null
// GFX12: :[[@LINE-1]]:25: error: invalid operand for instruction

global_load_u8 v0, v2, null
// GFX12: :[[@LINE-1]]:24: error: invalid operand for instruction

global_store_b128 v0, v[2:5], null
// GFX12: :[[@LINE-1]]:31: error: invalid operand for instruction

global_store_b16 v0, v2, null
// GFX12: :[[@LINE-1]]:26: error: invalid operand for instruction

global_store_b32 v0, v2, null
// GFX12: :[[@LINE-1]]:26: error: invalid operand for instruction

global_store_b64 v0, v[2:3], null
// GFX12: :[[@LINE-1]]:30: error: invalid operand for instruction

global_store_b8 v0, v2, null
// GFX12: :[[@LINE-1]]:25: error: invalid operand for instruction

global_store_b96 v0, v[2:4], null
// GFX12: :[[@LINE-1]]:30: error: invalid operand for instruction

global_store_block v32, v[0:31], null
// GFX12: :[[@LINE-1]]:34: error: invalid operand for instruction

global_store_d16_hi_b16 v0, v2, null
// GFX12: :[[@LINE-1]]:33: error: invalid operand for instruction

global_store_d16_hi_b8 v0, v2, null
// GFX12: :[[@LINE-1]]:32: error: invalid operand for instruction

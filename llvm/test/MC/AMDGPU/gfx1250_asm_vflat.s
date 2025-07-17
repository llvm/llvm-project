// RUN: llvm-mc -triple=amdgcn -mcpu=gfx1250 -show-encoding %s | FileCheck --check-prefix=GFX1250 %s
// RUN: not llvm-mc -triple=amdgcn -mcpu=gfx1200 -show-encoding %s 2>&1 | FileCheck --check-prefix=GFX12-ERR --implicit-check-not=error: --strict-whitespace %s

tensor_save s[0:1]
// GFX1250: tensor_save s[0:1] ; encoding: [0x00,0x80,0x1b,0xee,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00]
// GFX12-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: instruction not supported on this GPU

tensor_save s[0:1] th:TH_STORE_BYPASS scope:SCOPE_SYS
// GFX1250: tensor_save s[0:1] th:TH_STORE_BYPASS scope:SCOPE_SYS ; encoding: [0x00,0x80,0x1b,0xee,0x00,0x00,0x3c,0x00,0x00,0x00,0x00,0x00]
// GFX12-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: instruction not supported on this GPU

tensor_save s[0:1] offset:32
// GFX1250: tensor_save s[0:1] offset:32 ; encoding: [0x00,0x80,0x1b,0xee,0x00,0x00,0x00,0x00,0x00,0x20,0x00,0x00]
// GFX12-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: instruction not supported on this GPU

tensor_stop
// GFX1250: tensor_stop ; encoding: [0x7c,0xc0,0x1b,0xee,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00]
// GFX12-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: instruction not supported on this GPU

tensor_stop th:TH_STORE_BYPASS scope:SCOPE_SYS
// GFX1250: tensor_stop th:TH_STORE_BYPASS scope:SCOPE_SYS ; encoding: [0x7c,0xc0,0x1b,0xee,0x00,0x00,0x3c,0x00,0x00,0x00,0x00,0x00]
// GFX12-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: instruction not supported on this GPU

flat_atomic_add_f32 v2, v3, s[2:3] offset:64
// GFX1250: flat_atomic_add_f32 v2, v3, s[2:3] offset:64 ; encoding: [0x02,0x80,0x15,0xec,0x00,0x00,0x80,0x01,0x02,0x40,0x00,0x00]
// GFX12-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

flat_atomic_add_u32 v2, v3, s[2:3] offset:-64
// GFX1250: flat_atomic_add_u32 v2, v3, s[2:3] offset:-64 ; encoding: [0x02,0x40,0x0d,0xec,0x00,0x00,0x80,0x01,0x02,0xc0,0xff,0xff]
// GFX12-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

flat_atomic_add_u64 v2, v[2:3], s[2:3] offset:64
// GFX1250: flat_atomic_add_u64 v2, v[2:3], s[2:3] offset:64 ; encoding: [0x02,0xc0,0x10,0xec,0x00,0x00,0x00,0x01,0x02,0x40,0x00,0x00]
// GFX12-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

flat_atomic_and_b32 v2, v3, s[2:3] offset:64
// GFX1250: flat_atomic_and_b32 v2, v3, s[2:3] offset:64 ; encoding: [0x02,0x00,0x0f,0xec,0x00,0x00,0x80,0x01,0x02,0x40,0x00,0x00]
// GFX12-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

flat_atomic_and_b64 v2, v[2:3], s[2:3] offset:64
// GFX1250: flat_atomic_and_b64 v2, v[2:3], s[2:3] offset:64 ; encoding: [0x02,0x40,0x12,0xec,0x00,0x00,0x00,0x01,0x02,0x40,0x00,0x00]
// GFX12-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

flat_atomic_cmpswap_b32 v2, v[2:3], s[2:3] offset:64
// GFX1250: flat_atomic_cmpswap_b32 v2, v[2:3], s[2:3] offset:64 ; encoding: [0x02,0x00,0x0d,0xec,0x00,0x00,0x00,0x01,0x02,0x40,0x00,0x00]
// GFX12-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

flat_atomic_cmpswap_b64 v2, v[2:5], s[2:3]
// GFX1250: flat_atomic_cmpswap_b64 v2, v[2:5], s[2:3] ; encoding: [0x02,0x80,0x10,0xec,0x00,0x00,0x00,0x01,0x02,0x00,0x00,0x00]
// GFX12-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

flat_atomic_cond_sub_u32 v2, v3, s[2:3] offset:64
// GFX1250: flat_atomic_cond_sub_u32 v2, v3, s[2:3] offset:64 ; encoding: [0x02,0x00,0x14,0xec,0x00,0x00,0x80,0x01,0x02,0x40,0x00,0x00]
// GFX12-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

flat_atomic_dec_u32 v2, v3, s[2:3] offset:64
// GFX1250: flat_atomic_dec_u32 v2, v3, s[2:3] offset:64 ; encoding: [0x02,0x00,0x10,0xec,0x00,0x00,0x80,0x01,0x02,0x40,0x00,0x00]
// GFX12-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

flat_atomic_dec_u64 v2, v[2:3], s[2:3] offset:64
// GFX1250: flat_atomic_dec_u64 v2, v[2:3], s[2:3] offset:64 ; encoding: [0x02,0x40,0x13,0xec,0x00,0x00,0x00,0x01,0x02,0x40,0x00,0x00]
// GFX12-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

flat_atomic_inc_u32 v2, v3, s[2:3] offset:64
// GFX1250: flat_atomic_inc_u32 v2, v3, s[2:3] offset:64 ; encoding: [0x02,0xc0,0x0f,0xec,0x00,0x00,0x80,0x01,0x02,0x40,0x00,0x00]
// GFX12-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

flat_atomic_inc_u64 v2, v[2:3], s[2:3] offset:64
// GFX1250: flat_atomic_inc_u64 v2, v[2:3], s[2:3] offset:64 ; encoding: [0x02,0x00,0x13,0xec,0x00,0x00,0x00,0x01,0x02,0x40,0x00,0x00]
// GFX12-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

flat_atomic_max_num_f32 v2, v3, s[2:3] offset:64
// GFX1250: flat_atomic_max_num_f32 v2, v3, s[2:3] offset:64 ; encoding: [0x02,0x80,0x14,0xec,0x00,0x00,0x80,0x01,0x02,0x40,0x00,0x00]
// GFX12-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

flat_atomic_max_i32 v2, v3, s[2:3] offset:64
// GFX1250: flat_atomic_max_i32 v2, v3, s[2:3] offset:64 ; encoding: [0x02,0x80,0x0e,0xec,0x00,0x00,0x80,0x01,0x02,0x40,0x00,0x00]
// GFX12-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

flat_atomic_max_i64 v2, v[2:3], s[2:3] offset:64
// GFX1250: flat_atomic_max_i64 v2, v[2:3], s[2:3] offset:64 ; encoding: [0x02,0xc0,0x11,0xec,0x00,0x00,0x00,0x01,0x02,0x40,0x00,0x00]
// GFX12-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

flat_atomic_max_u32 v2, v3, s[2:3] offset:64
// GFX1250: flat_atomic_max_u32 v2, v3, s[2:3] offset:64 ; encoding: [0x02,0xc0,0x0e,0xec,0x00,0x00,0x80,0x01,0x02,0x40,0x00,0x00]
// GFX12-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

flat_atomic_max_u64 v2, v[2:3], s[2:3] offset:64
// GFX1250: flat_atomic_max_u64 v2, v[2:3], s[2:3] offset:64 ; encoding: [0x02,0x00,0x12,0xec,0x00,0x00,0x00,0x01,0x02,0x40,0x00,0x00]
// GFX12-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

flat_atomic_min_num_f32 v2, v3, s[2:3] offset:64
// GFX1250: flat_atomic_min_num_f32 v2, v3, s[2:3] offset:64 ; encoding: [0x02,0x40,0x14,0xec,0x00,0x00,0x80,0x01,0x02,0x40,0x00,0x00]
// GFX12-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

flat_atomic_min_i32 v2, v3, s[2:3] offset:64
// GFX1250: flat_atomic_min_i32 v2, v3, s[2:3] offset:64 ; encoding: [0x02,0x00,0x0e,0xec,0x00,0x00,0x80,0x01,0x02,0x40,0x00,0x00]
// GFX12-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

flat_atomic_min_i64 v2, v[2:3], s[2:3] offset:64
// GFX1250: flat_atomic_min_i64 v2, v[2:3], s[2:3] offset:64 ; encoding: [0x02,0x40,0x11,0xec,0x00,0x00,0x00,0x01,0x02,0x40,0x00,0x00]
// GFX12-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

flat_atomic_min_u32 v2, v3, s[2:3] offset:64
// GFX1250: flat_atomic_min_u32 v2, v3, s[2:3] offset:64 ; encoding: [0x02,0x40,0x0e,0xec,0x00,0x00,0x80,0x01,0x02,0x40,0x00,0x00]
// GFX12-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

flat_atomic_min_u64 v2, v[2:3], s[2:3] offset:64
// GFX1250: flat_atomic_min_u64 v2, v[2:3], s[2:3] offset:64 ; encoding: [0x02,0x80,0x11,0xec,0x00,0x00,0x00,0x01,0x02,0x40,0x00,0x00]
// GFX12-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

flat_atomic_or_b32 v2, v3, s[2:3] offset:64
// GFX1250: flat_atomic_or_b32 v2, v3, s[2:3] offset:64 ; encoding: [0x02,0x40,0x0f,0xec,0x00,0x00,0x80,0x01,0x02,0x40,0x00,0x00]
// GFX12-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

flat_atomic_or_b64 v2, v[2:3], s[2:3] offset:64
// GFX1250: flat_atomic_or_b64 v2, v[2:3], s[2:3] offset:64 ; encoding: [0x02,0x80,0x12,0xec,0x00,0x00,0x00,0x01,0x02,0x40,0x00,0x00]
// GFX12-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

flat_atomic_sub_clamp_u32 v2, v3, s[2:3] offset:64
// GFX1250: flat_atomic_sub_clamp_u32 v2, v3, s[2:3] offset:64 ; encoding: [0x02,0xc0,0x0d,0xec,0x00,0x00,0x80,0x01,0x02,0x40,0x00,0x00]
// GFX12-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

flat_atomic_sub_u32 v2, v3, s[2:3] offset:64
// GFX1250: flat_atomic_sub_u32 v2, v3, s[2:3] offset:64 ; encoding: [0x02,0x80,0x0d,0xec,0x00,0x00,0x80,0x01,0x02,0x40,0x00,0x00]
// GFX12-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

flat_atomic_sub_u64 v2, v[2:3], s[2:3] offset:64
// GFX1250: flat_atomic_sub_u64 v2, v[2:3], s[2:3] offset:64 ; encoding: [0x02,0x00,0x11,0xec,0x00,0x00,0x00,0x01,0x02,0x40,0x00,0x00]
// GFX12-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

flat_atomic_swap_b32 v2, v3, s[2:3] offset:64
// GFX1250: flat_atomic_swap_b32 v2, v3, s[2:3] offset:64 ; encoding: [0x02,0xc0,0x0c,0xec,0x00,0x00,0x80,0x01,0x02,0x40,0x00,0x00]
// GFX12-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

flat_atomic_swap_b64 v2, v[2:3], s[2:3] offset:64
// GFX1250: flat_atomic_swap_b64 v2, v[2:3], s[2:3] offset:64 ; encoding: [0x02,0x40,0x10,0xec,0x00,0x00,0x00,0x01,0x02,0x40,0x00,0x00]
// GFX12-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

flat_atomic_xor_b32 v2, v3, s[2:3] offset:64
// GFX1250: flat_atomic_xor_b32 v2, v3, s[2:3] offset:64 ; encoding: [0x02,0x80,0x0f,0xec,0x00,0x00,0x80,0x01,0x02,0x40,0x00,0x00]
// GFX12-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

flat_atomic_xor_b64 v2, v[2:3], s[2:3] offset:64
// GFX1250: flat_atomic_xor_b64 v2, v[2:3], s[2:3] offset:64 ; encoding: [0x02,0xc0,0x12,0xec,0x00,0x00,0x00,0x01,0x02,0x40,0x00,0x00]
// GFX12-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

flat_atomic_pk_add_f16 v2, v3, s[2:3] offset:64
// GFX1250: flat_atomic_pk_add_f16 v2, v3, s[2:3] offset:64 ; encoding: [0x02,0x40,0x16,0xec,0x00,0x00,0x80,0x01,0x02,0x40,0x00,0x00]
// GFX12-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

flat_atomic_pk_add_bf16 v2, v3, s[2:3] offset:64
// GFX1250: flat_atomic_pk_add_bf16 v2, v3, s[2:3] offset:64 ; encoding: [0x02,0x80,0x16,0xec,0x00,0x00,0x80,0x01,0x02,0x40,0x00,0x00]
// GFX12-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

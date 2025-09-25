// RUN: llvm-mc -triple=amdgcn -show-encoding -mcpu=gfx1250 %s | FileCheck --check-prefix=GFX1250 %s
// RUN: not llvm-mc -triple=amdgcn -mcpu=gfx1200 -show-encoding %s 2>&1 | FileCheck --check-prefix=GFX12-ERR --implicit-check-not=error: --strict-whitespace %s

s_add_pc_i64 s[2:3]
// GFX1250: s_add_pc_i64 s[2:3]                     ; encoding: [0x02,0x4b,0x80,0xbe]
// GFX12-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: instruction not supported on this GPU

s_add_pc_i64 4
// GFX1250: s_add_pc_i64 4                          ; encoding: [0x84,0x4b,0x80,0xbe]
// GFX12-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: instruction not supported on this GPU

s_add_pc_i64 100
// GFX1250: s_add_pc_i64 0x64                       ; encoding: [0xff,0x4b,0x80,0xbe,0x64,0x00,0x00,0x00]
// GFX12-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: instruction not supported on this GPU

s_add_pc_i64 0x12345678abcd0
// GFX1250: s_add_pc_i64 lit64(0x12345678abcd0)            ; encoding: [0xfe,0x4b,0x80,0xbe,0xd0,0xbc,0x8a,0x67,0x45,0x23,0x01,0x00]
// GFX12-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: instruction not supported on this GPU

s_get_pc_i64 s[2:3]
// GFX1250: s_get_pc_i64 s[2:3]                     ; encoding: [0x00,0x47,0x82,0xbe]
// GFX12-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: instruction not supported on this GPU

s_getpc_b64 s[2:3]
// GFX1250: s_get_pc_i64 s[2:3]                     ; encoding: [0x00,0x47,0x82,0xbe]

s_set_pc_i64 s[2:3]
// GFX1250: s_set_pc_i64 s[2:3]                     ; encoding: [0x02,0x48,0x80,0xbe]
// GFX12-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: instruction not supported on this GPU

s_setpc_b64 s[2:3]
// GFX1250: s_set_pc_i64 s[2:3]                     ; encoding: [0x02,0x48,0x80,0xbe]

s_swap_pc_i64 s[2:3], 10
// GFX1250: s_swap_pc_i64 s[2:3], 10                ; encoding: [0x8a,0x49,0x82,0xbe]
// GFX12-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: instruction not supported on this GPU

s_swappc_b64 s[2:3], 10
// GFX1250: s_swap_pc_i64 s[2:3], 10                ; encoding: [0x8a,0x49,0x82,0xbe]

s_rfe_i64 s[2:3]
// GFX1250: s_rfe_i64 s[2:3]                        ; encoding: [0x02,0x4a,0x80,0xbe]
// GFX12-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: instruction not supported on this GPU

s_rfe_b64 s[2:3]
// GFX1250: s_rfe_i64 s[2:3]                        ; encoding: [0x02,0x4a,0x80,0xbe]

s_sendmsg_rtn_b32 s2, sendmsg(MSG_RTN_GET_CLUSTER_BARRIER_STATE)
// GFX1250: s_sendmsg_rtn_b32 s2, sendmsg(MSG_RTN_GET_CLUSTER_BARRIER_STATE) ; encoding: [0x88,0x4c,0x82,0xbe]
// GFX12-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: specified message id is not supported on this GPU

s_sendmsg_rtn_b64 s[2:3], sendmsg(MSG_RTN_GET_CLUSTER_BARRIER_STATE)
// GFX1250: s_sendmsg_rtn_b64 s[2:3], sendmsg(MSG_RTN_GET_CLUSTER_BARRIER_STATE) ; encoding: [0x88,0x4d,0x82,0xbe]
// GFX12-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: specified message id is not supported on this GPU

s_get_shader_cycles_u64 s[2:3]
// GFX1250: s_get_shader_cycles_u64 s[2:3]          ; encoding: [0x00,0x06,0x82,0xbe]
// GFX12-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: instruction not supported on this GPU

s_barrier_signal -3
// GFX1250: s_barrier_signal -3                     ; encoding: [0xc3,0x4e,0x80,0xbe]

s_get_barrier_state s3, -3
// GFX1250: s_get_barrier_state s3, -3              ; encoding: [0xc3,0x50,0x83,0xbe]

s_get_barrier_state s3, -4
// GFX1250: s_get_barrier_state s3, -4              ; encoding: [0xc4,0x50,0x83,0xbe]

s_get_barrier_state s3, m0
// GFX1250: s_get_barrier_state s3, m0              ; encoding: [0x7d,0x50,0x83,0xbe]

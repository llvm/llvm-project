// RUN: llvm-mc -arch=amdgcn -show-encoding -mcpu=gfx1210 %s | FileCheck --check-prefix=GFX1210 %s
// RUN: not llvm-mc -arch=amdgcn -mcpu=gfx1200 -show-encoding %s 2>&1 | FileCheck --check-prefix=GFX12-ERR --implicit-check-not=error: --strict-whitespace %s

s_add_pc_i64 s[2:3]
// GFX1210: s_add_pc_i64 s[2:3]                     ; encoding: [0x02,0x4b,0x80,0xbe]
// GFX12-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: instruction not supported on this GPU

s_add_pc_i64 4
// GFX1210: s_add_pc_i64 4                          ; encoding: [0x84,0x4b,0x80,0xbe]
// GFX12-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: instruction not supported on this GPU

s_add_pc_i64 100
// GFX1210: s_add_pc_i64 0x64                       ; encoding: [0xff,0x4b,0x80,0xbe,0x64,0x00,0x00,0x00]
// GFX12-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: instruction not supported on this GPU

s_add_pc_i64 0x12345678abcd0
// GFX1210: s_add_pc_i64 0x12345678abcd0            ; encoding: [0xfe,0x4b,0x80,0xbe,0xd0,0xbc,0x8a,0x67,0x45,0x23,0x01,0x00]
// GFX12-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: instruction not supported on this GPU

s_sendmsg_rtn_b32 s2, sendmsg(MSG_RTN_GET_CLUSTER_BARRIER_STATE)
// GFX1210: s_sendmsg_rtn_b32 s2, sendmsg(MSG_RTN_GET_CLUSTER_BARRIER_STATE) ; encoding: [0x88,0x4c,0x82,0xbe]
// GFX12-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: specified message id is not supported on this GPU

s_sendmsg_rtn_b64 s[2:3], sendmsg(MSG_RTN_GET_CLUSTER_BARRIER_STATE)
// GFX1210: s_sendmsg_rtn_b64 s[2:3], sendmsg(MSG_RTN_GET_CLUSTER_BARRIER_STATE) ; encoding: [0x88,0x4d,0x82,0xbe]
// GFX12-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: specified message id is not supported on this GPU

// RUN: llvm-mc -triple=amdgcn -show-encoding -mcpu=gfx1250 %s | FileCheck --check-prefix=GFX1250 %s
// RUN: not llvm-mc -triple=amdgcn -mcpu=gfx1200 -show-encoding %s 2>&1 | FileCheck --check-prefix=GFX12-ERR --implicit-check-not=error: --strict-whitespace %s

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

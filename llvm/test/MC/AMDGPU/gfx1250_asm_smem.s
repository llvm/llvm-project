// RUN: llvm-mc -triple=amdgcn -mcpu=gfx1250 -show-encoding %s | FileCheck --check-prefix=GFX1250 %s
// RUN: not llvm-mc -triple=amdgcn -mcpu=gfx1200 -show-encoding %s 2>&1 | FileCheck --check-prefix=GFX12-ERR --implicit-check-not=error: --strict-whitespace %s

s_load_b32 s4, s[2:3], 10 nv
// GFX1250: s_load_b32 s4, s[2:3], 0xa nv           ; encoding: [0x01,0x01,0x10,0xf4,0x0a,0x00,0x00,0xf8]
// GFX12-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: nv is not supported on this GPU
// GFX12-ERR-NEXT:{{^}}s_load_b32 s4, s[2:3], 10 nv
// GFX12-ERR-NEXT:{{^}}                          ^

s_buffer_load_i8 s5, s[4:7], s0 nv
// GFX1250: s_buffer_load_i8 s5, s[4:7], s0 offset:0x0 nv ; encoding: [0x42,0x01,0x13,0xf4,0x00,0x00,0x00,0x00]
// GFX12-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: nv is not supported on this GPU
// GFX12-ERR-NEXT:{{^}}s_buffer_load_i8 s5, s[4:7], s0 nv
// GFX12-ERR-NEXT:{{^}}                                ^

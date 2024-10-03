// RUN: llvm-mc -arch=amdgcn -show-encoding -mcpu=gfx1210 %s | FileCheck --check-prefix=GFX1210 %s
// RUN: not llvm-mc -arch=amdgcn -mcpu=gfx1200 -show-encoding %s 2>&1 | FileCheck --check-prefixes=GFX12-ERR --implicit-check-not=error: -strict-whitespace %s

s_call_i64 s[0:1], 4660
// GFX1210: s_call_i64 s[0:1], 4660                 ; encoding: [0x34,0x12,0x00,0xba]
// GFX12-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: instruction not supported on this GPU

s_call_b64 s[0:1], 4660
// GFX1210: s_call_i64 s[0:1], 4660                 ; encoding: [0x34,0x12,0x00,0xba]

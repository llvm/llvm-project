// RUN: llvm-mc -triple=amdgcn -show-encoding -mcpu=gfx1250 %s | FileCheck --check-prefix=GFX1250 %s
// RUN: not llvm-mc -triple=amdgcn -mcpu=gfx1200 -filetype=null %s 2>&1 | FileCheck --check-prefixes=GFX12-ERR --implicit-check-not=error: -strict-whitespace %s

s_call_i64 s[0:1], 4660
// GFX1250: s_call_i64 s[0:1], 4660                 ; encoding: [0x34,0x12,0x00,0xba]
// GFX12-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: instruction not supported on this GPU

s_call_b64 s[0:1], 4660
// GFX1250: s_call_i64 s[0:1], 4660                 ; encoding: [0x34,0x12,0x00,0xba]

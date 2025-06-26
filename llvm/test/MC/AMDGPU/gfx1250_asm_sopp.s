// RUN: llvm-mc -triple=amdgcn -show-encoding -mcpu=gfx1250 %s | FileCheck --check-prefix=GFX1250 %s
// RUN: not llvm-mc -triple=amdgcn -mcpu=gfx1200 -show-encoding %s 2>&1 | FileCheck --check-prefixes=GFX12-ERR --implicit-check-not=error: -strict-whitespace %s

s_wait_xcnt 0x0
// GFX1250: [0x00,0x00,0xc5,0xbf]
// GFX12-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: instruction not supported on this GPU

s_wait_xcnt 0x7
// GFX1250: [0x07,0x00,0xc5,0xbf]
// GFX12-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: instruction not supported on this GPU

s_wait_xcnt 0xf
// GFX1250: [0x0f,0x00,0xc5,0xbf]
// GFX12-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: instruction not supported on this GPU

s_setprio_inc_wg 100
// GFX1250: [0x64,0x00,0xbe,0xbf]
// GFX12-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: instruction not supported on this GPU

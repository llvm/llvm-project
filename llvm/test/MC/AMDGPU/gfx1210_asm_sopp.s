// RUN: llvm-mc -arch=amdgcn -show-encoding -mcpu=gfx1210 %s | FileCheck --check-prefix=GFX1210 %s

s_wait_asynccnt 0x1234
// GFX1210: [0x34,0x12,0xca,0xbf]

s_wait_asynccnt 0xc1d1
// GFX1210: [0xd1,0xc1,0xca,0xbf]

s_wait_asynccnt 0x1234
// GFX1210: [0x34,0x12,0xca,0xbf]

s_wait_asynccnt 0xc1d1
// GFX1210: [0xd1,0xc1,0xca,0xbf]

s_setprio_inc_wg 100
// GFX1210: [0x64,0x00,0xbe,0xbf]

s_set_vgpr_msb 10
// GFX1210: [0x0a,0x00,0x86,0xbf]
// GFX12-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: instruction not supported on this GPU

s_set_vgpr_msb 255
// GFX1210: [0xff,0x00,0x86,0xbf]
// GFX12-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: instruction not supported on this GPU

// RUN: llvm-mc -arch=amdgcn -show-encoding -mcpu=gfx1210 %s | FileCheck --check-prefix=GFX1210 %s

s_set_vgpr_msb 10
// GFX1210: [0x0a,0x00,0x86,0xbf]

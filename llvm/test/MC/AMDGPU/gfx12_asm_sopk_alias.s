// RUN: llvm-mc -arch=amdgcn -show-encoding -mcpu=gfx1200 %s | FileCheck --check-prefix=GFX12 %s

s_addk_i32 s0, 0x1234
// GFX12: encoding: [0x34,0x12,0x80,0xb7]

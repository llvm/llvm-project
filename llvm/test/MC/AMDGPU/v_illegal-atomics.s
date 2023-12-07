// RUN: llvm-mc -arch=amdgcn -mcpu=gfx1030 -show-encoding %s | FileCheck --check-prefix=GFX1030 %s
// RUN: llvm-mc -arch=amdgcn -mcpu=gfx1100 -show-encoding %s | FileCheck --check-prefix=GFX1100 %s

v_illegal
// GFX1030: encoding: [0x00,0x00,0x00,0x00]
// GFX1100: encoding: [0x00,0x00,0x00,0x00]

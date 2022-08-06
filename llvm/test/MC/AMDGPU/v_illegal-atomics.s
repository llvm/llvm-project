// RUN: llvm-mc -arch=amdgcn -mcpu=gfx906 -show-encoding %s | FileCheck --check-prefix=GFX906 %s
// RUN: llvm-mc -arch=amdgcn -mcpu=gfx908 -show-encoding %s | FileCheck --check-prefix=GFX908 %s
// RUN: llvm-mc -arch=amdgcn -mcpu=gfx90a -show-encoding %s | FileCheck --check-prefix=GFX90A %s
// RUN: llvm-mc -arch=amdgcn -mcpu=gfx940 -show-encoding %s | FileCheck --check-prefix=GFX940 %s
// RUN: llvm-mc -arch=amdgcn -mcpu=gfx1030 -show-encoding %s | FileCheck --check-prefix=GFX1030 %s
// RUN: llvm-mc -arch=amdgcn -mcpu=gfx1100 -show-encoding %s | FileCheck --check-prefix=GFX1100 %s

v_illegal
// GFX906: encoding: [0xff,0xff,0xff,0xff]
// GFX908: encoding: [0xff,0xff,0xff,0xff]
// GFX90A: encoding: [0xff,0xff,0xff,0xff]
// GFX940: encoding: [0xff,0xff,0xff,0xff]
// GFX1030: encoding: [0x00,0x00,0x00,0x00]
// GFX1100: encoding: [0x00,0x00,0x00,0x00]

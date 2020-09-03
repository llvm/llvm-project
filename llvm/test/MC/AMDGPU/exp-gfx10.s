// RUN: not llvm-mc -arch=amdgcn -mcpu=verde %s 2>&1 | FileCheck -check-prefix=SI --implicit-check-not=error: %s
// RUN: not llvm-mc -arch=amdgcn -mcpu=tonga %s 2>&1 | FileCheck -check-prefix=VI --implicit-check-not=error: %s
// RUN: llvm-mc -arch=amdgcn -mcpu=gfx1010 -show-encoding %s | FileCheck -check-prefix=GFX10 %s

exp prim v1, off, off, off
// SI: :5: error: invalid operand for instruction
// VI: :5: error: invalid operand for instruction
// GFX10: exp prim v1, off, off, off ; encoding: [0x41,0x01,0x00,0xf8,0x01,0x00,0x00,0x00]

exp prim v2, v3, off, off
// SI: :5: error: invalid operand for instruction
// VI: :5: error: invalid operand for instruction
// GFX10: exp prim v2, v3, off, off ; encoding: [0x43,0x01,0x00,0xf8,0x02,0x03,0x00,0x00]

exp pos4 v4, v3, v2, v1
// SI: error: invalid exp target
// VI: error: invalid exp target
// GFX10: exp pos4 v4, v3, v2, v1 ; encoding: [0x0f,0x01,0x00,0xf8,0x04,0x03,0x02,0x01]

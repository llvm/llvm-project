// RUN: llvm-mc -triple=amdgcn -mcpu=gfx1170 -show-encoding %s | FileCheck -check-prefix=GFX1170 %s

//===----------------------------------------------------------------------===//
// A VOPD OpY mov_b32 instruction uses SRC2 source-cache if OpX is also mov_b32
//===----------------------------------------------------------------------===//

v_dual_mov_b32 v2, v5 :: v_dual_mov_b32 v3, v1
// GFX1170: encoding: [0x05,0x01,0x10,0xca,0x01,0x01,0x02,0x02]

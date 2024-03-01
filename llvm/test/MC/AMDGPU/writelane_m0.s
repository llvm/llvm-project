// RUN: llvm-mc --triple=amdgcn --mcpu=gfx600 -show-encoding %s | FileCheck %s -check-prefix=GFX6
// RUN: llvm-mc --triple=amdgcn --mcpu=gfx700 -show-encoding %s | FileCheck %s -check-prefix=GFX7
// RUN: llvm-mc --triple=amdgcn --mcpu=gfx904 -show-encoding %s | FileCheck %s -check-prefix=GFX9
// RUN: llvm-mc --triple=amdgcn --mcpu=gfx940 -show-encoding %s | FileCheck %s -check-prefix=GFX9
// RUN: llvm-mc --triple=amdgcn --mcpu=gfx1010 -show-encoding %s | FileCheck %s -check-prefix=GFX10
// RUN: llvm-mc --triple=amdgcn --mcpu=gfx1030 -show-encoding %s | FileCheck %s -check-prefix=GFX10
// RUN: llvm-mc --triple=amdgcn --mcpu=gfx1100 -show-encoding %s | FileCheck %s -check-prefix=GFX11
// RUN: llvm-mc --triple=amdgcn --mcpu=gfx1200 -show-encoding %s | FileCheck %s -check-prefix=GFX12

.text
  v_writelane_b32 v1, s13, m0

// GFX6: v_writelane_b32 v1, s13, m0 ; encoding: [0x0d,0xf8,0x02,0x04]
// GFX7: v_writelane_b32 v1, s13, m0 ; encoding: [0x0d,0xf8,0x02,0x04]
// GFX9: v_writelane_b32 v1, s13, m0 ; encoding: [0x01,0x00,0x8a,0xd2,0x0d,0xf8,0x00,0x00]
// GFX10: v_writelane_b32 v1, s13, m0 ; encoding: [0x01,0x00,0x61,0xd7,0x0d,0xf8,0x00,0x00]
// GFX11: v_writelane_b32 v1, s13, m0 ; encoding: [0x01,0x00,0x61,0xd7,0x0d,0xfa,0x00,0x00]
// GFX12: v_writelane_b32 v1, s13, m0 ; encoding: [0x01,0x00,0x61,0xd7,0x0d,0xfa,0x00,0x00]

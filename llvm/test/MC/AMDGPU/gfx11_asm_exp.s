// RUN: not llvm-mc -arch=amdgcn -mcpu=gfx900 %s 2>&1 | FileCheck -check-prefix=PREGFX11 --implicit-check-not=error: %s
// RUN: not llvm-mc -arch=amdgcn -mcpu=gfx1010 %s 2>&1 | FileCheck -check-prefix=PREGFX11 --implicit-check-not=error: %s
// RUN: llvm-mc -arch=amdgcn -mcpu=gfx1100 -show-encoding %s | FileCheck -check-prefix=GFX11 --implicit-check-not=error: %s

exp dual_src_blend0 v4, v3, v2, v1
// PREGFX11: :[[@LINE-1]]:{{[0-9]+}}: error: exp target is not supported on this GPU
// GFX11: exp dual_src_blend0 v4, v3, v2, v1      ; encoding: [0x5f,0x01,0x00,0xf8,0x04,0x03,0x02,0x01]

exp dual_src_blend1 v2, v3, off, off
// PREGFX11: :[[@LINE-1]]:{{[0-9]+}}: error: exp target is not supported on this GPU
// GFX11: exp dual_src_blend1 v2, v3, off, off    ; encoding: [0x63,0x01,0x00,0xf8,0x02,0x03,0x00,0x00]

exp mrtz v4, v3, v2, v1 row_en
// PREGFX11: :[[@LINE-1]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode
// GFX11: exp mrtz v4, v3, v2, v1 row_en          ; encoding: [0x8f,0x20,0x00,0xf8,0x04,0x03,0x02,0x01]

exp mrtz v4, v3, off, off done row_en
// PREGFX11: :[[@LINE-1]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode
// GFX11: exp mrtz v4, v3, off, off done row_en   ; encoding: [0x83,0x28,0x00,0xf8,0x04,0x03,0x00,0x00]

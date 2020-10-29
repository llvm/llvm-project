# RUN: llvm-mc -arch=amdgcn -mcpu=gfx1100 -disassemble -show-encoding < %s | FileCheck -check-prefix=GFX11 %s

# GFX11:  s_mov_b32 s0, s1
0x01,0x00,0x80,0xbe

# GFX11:  s_ctz_i32_b32 s0, s104
0x68,0x08,0x80,0xbe

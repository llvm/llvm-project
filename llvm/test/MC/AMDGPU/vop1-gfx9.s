// RUN: llvm-mc -arch=amdgcn -mcpu=gfx901 -show-encoding %s | FileCheck -check-prefix=GFX9 %s
// RUN: not llvm-mc -arch=amdgcn -mcpu=tahiti -show-encoding %s 2>&1 | FileCheck -check-prefix=NOVI %s
// RUN: not llvm-mc -arch=amdgcn -mcpu=hawaii -show-encoding %s 2>&1 | FileCheck -check-prefix=NOVI %s
// RUN: not llvm-mc -arch=amdgcn -mcpu=tonga -show-encoding %s 2>&1 | FileCheck -check-prefix=NOVI %s

v_swap_b32 v1, v2
// GFX9: v_swap_b32 v1, v2 ; encoding: [0x02,0xa3,0x02,0x7e]
// NOVI: :1: error: instruction not supported on this GPU

// FIXME: Error for it requiring VOP1 encoding
v_swap_b32_e32 v1, v2
// GFX9: v_swap_b32 v1, v2 ; encoding: [0x02,0xa3,0x02,0x7e]
// NOVI: :1: error: instruction not supported on this GPU

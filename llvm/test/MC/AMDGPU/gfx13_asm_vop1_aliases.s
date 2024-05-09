// RUN: llvm-mc -triple=amdgcn -mcpu=gfx1300 -mattr=+wavefrontsize32,-wavefrontsize64 -show-encoding %s | FileCheck --check-prefixes=GFX13 %s

v_brev_b32_e32 v5, v1
// GFX13: v_bfrev_b32_e32 v5, v1 ; encoding: [0x01,0x71,0x0a,0x7e]

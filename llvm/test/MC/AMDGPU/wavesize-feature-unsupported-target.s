// RUN: llvm-mc -triple=amdgcn -mcpu=gfx1250 -mattr=+wavefrontsize64 -o - %s | FileCheck -check-prefix=GFX1250 %s
// RUN: llvm-mc -triple=amdgcn -mcpu=gfx900 -mattr=+wavefrontsize32 -o - %s | FileCheck -check-prefix=GFX900 %s

// Make sure setting both modes is supported at the same time.
// RUN: llvm-mc -triple=amdgcn -mcpu=gfx1010 -mattr=+wavefrontsize32,+wavefrontsize64 %s | FileCheck -check-prefixes=GFX10 %s

// Test that there is no assertion when using an explicit
// wavefrontsize attribute on a target which does not support it.

// GFX1250: v_add_f64_e32 v[0:1], 1.0, v[0:1]
// GFX900: v_add_f64 v[0:1], 1.0, v[0:1]
// GFX10: v_add_f64 v[0:1], 1.0, v[0:1]
v_add_f64 v[0:1], 1.0, v[0:1]

// GFX1250: v_cmp_eq_u32_e64 s[0:1], 1.0, s1
// GFX900: v_cmp_eq_u32_e64 s[0:1], 1.0, s1
// GFX10: v_cmp_eq_u32_e64 s[0:1], 1.0, s1
v_cmp_eq_u32_e64 s[0:1], 1.0, s1

// GFX1250: v_cndmask_b32_e64 v1, v2, v3, s[0:1]
// GFX900: v_cndmask_b32_e64 v1, v2, v3, s[0:1]
// GFX10: v_cndmask_b32_e64 v1, v2, v3, s[0:1]
v_cndmask_b32 v1, v2, v3, s[0:1]

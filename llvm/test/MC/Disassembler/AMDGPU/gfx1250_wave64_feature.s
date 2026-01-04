# RUN: llvm-mc -triple=amdgcn -mcpu=gfx1250 -mattr=+wavefrontsize64 -disassemble -o - %s | FileCheck %s

# Make sure there's no assertion when trying to use an unsupported
# wave64 on a wave32-only target

# CHECK: v_add_f64_e32 v[0:1], 1.0, v[0:1]
0xf2,0x00,0x00,0x04

# CHECK: v_cmp_eq_u32_e64 s[0:1], 1.0, s1
0x00,0x00,0x4a,0xd4,0xf2,0x02,0x00,0x00

# CHECK: v_cndmask_b32_e64 v1, v2, v3, s[0:1]
0x01,0x00,0x01,0xd5,0x02,0x07,0x02,0x00

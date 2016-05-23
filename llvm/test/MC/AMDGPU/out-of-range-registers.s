// RUN: not llvm-mc -arch=amdgcn -mcpu=tahiti %s 2>&1 | FileCheck %s
// RUN: not llvm-mc -arch=amdgcn -mcpu=tonga %s 2>&1 | FileCheck %s

s_add_i32 s104, s0, s1
// CHECK: error: not a valid operand

s_add_i32 s105, s0, s1
// CHECK: error: not a valid operand

v_add_i32 v256, v0, v1
// CHECK: error: not a valid operand

v_add_i32 v257, v0, v1
// CHECK: error: not a valid operand

s_mov_b64 s[0:17], -1
// CHECK: error: not a valid operand

s_mov_b64 s[103:104], -1
// CHECK: error: not a valid operand

s_mov_b64 s[104:105], -1
// CHECK: error: not a valid operand

s_load_dwordx4 s[102:105], s[2:3], s4
// CHECK: error: not a valid operand

s_load_dwordx4 s[104:108], s[2:3], s4
// CHECK: error: not a valid operand

s_load_dwordx4 s[108:112], s[2:3], s4
// CHECK: error: not a valid operand

s_load_dwordx4 s[1:4], s[2:3], s4
// CHECK: error: not a valid operand

s_load_dwordx4 s[1:4], s[2:3], s4
// CHECK: error: not a valid operand

s_load_dwordx8 s[104:111], s[2:3], s4
// CHECK: error: not a valid operand

s_load_dwordx8 s[100:107], s[2:3], s4
// CHECK: error: not a valid operand

s_load_dwordx8 s[108:115], s[2:3], s4
// CHECK: error: not a valid operand

s_load_dwordx16 s[92:107], s[2:3], s4
// CHECK: error: not a valid operand

s_load_dwordx16 s[96:111], s[2:3], s4
// CHECK: error: not a valid operand

s_load_dwordx16 s[100:115], s[2:3], s4
// CHECK: error: not a valid operand

s_load_dwordx16 s[104:119], s[2:3], s4
// CHECK: error: not a valid operand

s_load_dwordx16 s[108:123], s[2:3], s4
// CHECK: error: not a valid operand

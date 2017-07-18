// RUN: not llvm-mc -arch=amdgcn -show-encoding %s 2>&1 | FileCheck %s
// RUN: not llvm-mc -arch=amdgcn -mcpu=SI -show-encoding %s 2>&1 | FileCheck %s

v_add_f32_e64 v0, v1
// CHECK: error: too few operands for instruction

v_div_scale_f32  v24, vcc, v22, 1.1, v22
// CHECK: error: invalid operand for instruction

v_mqsad_u32_u8 v[0:3], s[2:3], v4, v[0:3]
// CHECK: error: instruction not supported on this GPU

v_mqsad_pk_u16_u8 v[0:1], v[1:2], v9, v[4:5]
// CHECK: error: destination must be different than all sources

v_mqsad_pk_u16_u8 v[1:2], v[1:2], v9, v[4:5]
// CHECK: error: destination must be different than all sources

v_mqsad_pk_u16_u8 v[2:3], v[1:2], v9, v[4:5]
// CHECK: error: destination must be different than all sources

v_mqsad_pk_u16_u8 v[3:4], v[0:1], v9, v[4:5]
// CHECK: error: destination must be different than all sources

v_mqsad_pk_u16_u8 v[4:5], v[1:2], v9, v[4:5]
// CHECK: error: destination must be different than all sources

v_mqsad_pk_u16_u8 v[5:6], v[1:2], v9, v[4:5]
// CHECK: error: destination must be different than all sources

v_mqsad_pk_u16_u8 v[8:9], v[1:2], v9, v[4:5]
// CHECK: error: destination must be different than all sources

v_mqsad_pk_u16_u8 v[9:10], v[1:2], v9, v[4:5]
// CHECK: error: destination must be different than all sources

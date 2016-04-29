// RUN: not llvm-mc -arch=amdgcn %s 2>&1 | FileCheck %s
// RUN: not llvm-mc -arch=amdgcn -mcpu=SI %s 2>&1 | FileCheck %s

// offset too big
// CHECK: error: invalid operand for instruction
ds_add_u32 v2, v4 offset:1000000000

// offset0 twice
// CHECK:  error: invalid operand for instruction
ds_write2_b32 v2, v4, v6 offset0:4 offset0:8

// offset1 twice
// CHECK:  error: invalid operand for instruction
ds_write2_b32 v2, v4, v6 offset1:4 offset1:8

// offset0 too big
// CHECK: invalid operand for instruction
ds_write2_b32 v2, v4, v6 offset0:1000000000

// offset1 too big
// CHECK: invalid operand for instruction
ds_write2_b32 v2, v4, v6 offset1:1000000000


// RUN: not llvm-mc -triple=amdgcn -mcpu=gfx950 %s 2>&1 | FileCheck --implicit-check-not=error: %s

v_mfma_ld_scale_b32 v0, 65
// CHECK: :[[@LINE-1]]:25: error: literal operands are not supported

v_mfma_ld_scale_b32 65, v0
// CHECK: :[[@LINE-1]]:21: error: literal operands are not supported

v_mfma_ld_scale_b32 65, 65
// CHECK: :[[@LINE-1]]:25: error: literal operands are not supported

v_mfma_ld_scale_b32 s0, s1
// CHECK: :[[@LINE-1]]:25: error: invalid operand (violates constant bus restrictions)

v_mfma_ld_scale_b32 v0, v0 clamp
// CHECK: :[[@LINE-1]]:28: error: invalid operand for instruction

v_mfma_ld_scale_b32 v0, v0 neg_lo:[0,1]
// CHECK: :[[@LINE-1]]:28: error: not a valid operand

v_mfma_ld_scale_b32 v0, v0 neg_lo:[1,1]
// CHECK: :[[@LINE-1]]:28: error: not a valid operand

v_mfma_ld_scale_b32 v0, v0 neg_hi:[1,1]
// CHECK: :[[@LINE-1]]:28: error: not a valid operand

v_mfma_ld_scale_b32 v0, v0 neg_hi:[0,1]
// CHECK: :[[@LINE-1]]:28: error: not a valid operand

v_mfma_ld_scale_b32 v0, v0 neg_lo:[0,1] neg_hi:[0,1]
// CHECK: :[[@LINE-1]]:28: error: not a valid operand

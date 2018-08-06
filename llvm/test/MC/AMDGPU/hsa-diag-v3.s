// RUN: not llvm-mc -mattr=+code-object-v3 -triple amdgcn-amd-amdhsa -mcpu=gfx803 -mattr=+xnack -show-encoding %s 2>&1 >/dev/null | FileCheck %s
// RUN: not llvm-mc -mattr=+code-object-v3 -triple amdgcn-amd- -mcpu=gfx803 -mattr=+xnack -show-encoding %s 2>&1 >/dev/null | FileCheck %s --check-prefix=NOT-AMDHSA

.text

.amdgcn_target "amdgcn--amdhsa-gfx803+xnack"
// CHECK: error: target must match options

.amdhsa_kernel
// CHECK: error: unknown directive
.end_amdhsa_kernel

.amdhsa_kernel foo
  .amdhsa_group_segment_fixed_size -1
  // CHECK: error: value out of range
.end_amdhsa_kernel

.amdhsa_kernel foo
  .amdhsa_group_segment_fixed_size 10000000000 + 1
  // CHECK: error: value out of range
.end_amdhsa_kernel

.amdhsa_kernel foo
  // NOT-AMDHSA: error: directive only supported for amdhsa OS
.end_amdhsa_kernel

.amdhsa_kernel foo
  .amdhsa_group_segment_fixed_size 1
  .amdhsa_group_segment_fixed_size 1
  // CHECK: error: .amdhsa_ directives cannot be repeated
.end_amdhsa_kernel

.amdhsa_kernel foo
  // CHECK: error: .amdhsa_next_free_vgpr directive is required
.end_amdhsa_kernel

.amdhsa_kernel foo
  .amdhsa_next_free_vgpr 0
  // CHECK: error: .amdhsa_next_free_sgpr directive is required
.end_amdhsa_kernel

.amdhsa_kernel foo
  1
  // CHECK: error: expected .amdhsa_ directive or .end_amdhsa_kernel
.end_amdhsa_kernel

.set .amdgcn.next_free_vgpr, "foo"
v_mov_b32_e32 v0, s0
// CHECK: error: .amdgcn.next_free_{v,s}gpr symbols must be absolute expressions

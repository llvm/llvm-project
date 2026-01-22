// RUN: llvm-mc -triple=amdgcn-amd-amdhsa -mcpu=gfx1250 --amdhsa-code-object-version=6 -show-encoding %s | FileCheck %s

// CHECK: .amdgpu_metadata
// CHECK: amdhsa.kernels:
// CHECK: - .cluster_dims:
// CHECK-NEXT: - 4
// CHECK-NEXT: - 2
// CHECK-NEXT: - 1
.amdgpu_metadata
  amdhsa.version:
    - 1
    - 0
  amdhsa.printf:
    - '1:1:4:%d\n'
    - '2:1:8:%g\n'
  amdhsa.kernels:
    - .name:            test_kernel
      .symbol:      test_kernel@kd
      .language:        OpenCL C
      .language_version:
        - 2
        - 0
      .kernarg_segment_size: 8
      .group_segment_fixed_size: 16
      .private_segment_fixed_size: 32
      .kernarg_segment_align: 64
      .wavefront_size: 128
      .sgpr_count: 14
      .vgpr_count: 40
      .max_flat_workgroup_size: 256
      .cluster_dims:
        - 4
        - 2
        - 1
.end_amdgpu_metadata

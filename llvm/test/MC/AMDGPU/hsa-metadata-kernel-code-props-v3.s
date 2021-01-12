// RUN: llvm-mc -triple=amdgcn-amd-amdhsa -mcpu=gfx700 -show-encoding %s | FileCheck --check-prefix=CHECK %s
// RUN: llvm-mc -triple=amdgcn-amd-amdhsa -mcpu=gfx800 -show-encoding %s | FileCheck --check-prefix=CHECK %s
// RUN: llvm-mc -triple=amdgcn-amd-amdhsa -mcpu=gfx900 -show-encoding %s | FileCheck --check-prefix=CHECK %s

// CHECK:      	.amdgpu_metadata
// CHECK:      amdhsa.kernels:  
// CHECK:        - .group_segment_fixed_size: 24
// CHECK:          .kernarg_segment_align: 16
// CHECK:          .kernarg_segment_size: 24
// CHECK:          .max_flat_workgroup_size: 256
// CHECK:          .name:           test_kernel
// CHECK:          .private_segment_fixed_size: 16
// CHECK:          .sgpr_count:     40
// CHECK:          .sgpr_spill_count: 1
// CHECK:          .symbol:         'test_kernel@kd'
// CHECK:          .vgpr_count:     14
// CHECK:          .vgpr_spill_count: 1
// CHECK:          .wavefront_size: 64
// CHECK:      amdhsa.version:  
// CHECK-NEXT:   - 1
// CHECK-NEXT:   - 0
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
      .kernarg_segment_size:      24
      .group_segment_fixed_size:   24
      .private_segment_fixed_size: 16
      .kernarg_segment_align:     16
      .wavefront_size:           64
      .max_flat_workgroup_size:    256
      .sgpr_count:               40
      .vgpr_count:               14
      .sgpr_spill_count:         1
      .vgpr_spill_count:         1
.end_amdgpu_metadata

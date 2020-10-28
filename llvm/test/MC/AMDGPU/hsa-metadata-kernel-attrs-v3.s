// RUN: llvm-mc -triple=amdgcn-amd-amdhsa -mcpu=gfx700 -show-encoding %s | FileCheck --check-prefix=CHECK --check-prefix=GFX700 %s
// RUN: llvm-mc -triple=amdgcn-amd-amdhsa -mcpu=gfx800 -show-encoding %s | FileCheck --check-prefix=CHECK --check-prefix=GFX800 %s
// RUN: llvm-mc -triple=amdgcn-amd-amdhsa -mcpu=gfx900 -show-encoding %s | FileCheck --check-prefix=CHECK --check-prefix=GFX900 %s

// CHECK:      	.amdgpu_metadata
// CHECK:      amdhsa.kernels:  
// CHECK:        - .group_segment_fixed_size: 16
// CHECK:          .kernarg_segment_align: 64
// CHECK:          .kernarg_segment_size: 8
// CHECK:          .language:       OpenCL C
// CHECK:          .language_version: 
// CHECK-NEXT:       - 2
// CHECK-NEXT:       - 0
// CHECK:          .max_flat_workgroup_size: 256
// CHECK:          .name:           test_kernel
// CHECK:          .private_segment_fixed_size: 32
// CHECK:          .reqd_workgroup_size: 
// CHECK-NEXT:       - 1
// CHECK-NEXT:       - 2
// CHECK-NEXT:       - 4
// CHECK:          .sgpr_count:     14
// CHECK:          .symbol:         'test_kernel@kd'
// CHECK:          .vec_type_hint:  int
// CHECK:          .vgpr_count:     40
// CHECK:          .wavefront_size: 128
// CHECK:          .workgroup_size_hint: 
// CHECK-NEXT:       - 8
// CHECK-NEXT:       - 16
// CHECK-NEXT:       - 32
// CHECK:      amdhsa.printf:   
// CHECK:        - '1:1:4:%d\n'
// CHECK:        - '2:1:8:%g\n'
// CHECK:      amdhsa.version:  
// CHECK-NEXT:   - 1
// CHECK-NEXT:   - 0
// CHECK:      	.end_amdgpu_metadata
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
      .reqd_workgroup_size:
        - 1
        - 2
        - 4
      .workgroup_size_hint:
        - 8
        - 16
        - 32
      .vec_type_hint:       int
.end_amdgpu_metadata

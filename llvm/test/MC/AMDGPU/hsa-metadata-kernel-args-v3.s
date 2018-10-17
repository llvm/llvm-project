// RUN: llvm-mc -mattr=+code-object-v3 -triple=amdgcn-amd-amdhsa -mcpu=gfx700 -show-encoding %s | FileCheck --check-prefix=CHECK --check-prefix=GFX700 %s
// RUN: llvm-mc -mattr=+code-object-v3 -triple=amdgcn-amd-amdhsa -mcpu=gfx800 -show-encoding %s | FileCheck --check-prefix=CHECK --check-prefix=GFX800 %s
// RUN: llvm-mc -mattr=+code-object-v3 -triple=amdgcn-amd-amdhsa -mcpu=gfx900 -show-encoding %s | FileCheck --check-prefix=CHECK --check-prefix=GFX900 %s

// CHECK:  .amdgpu_metadata
// CHECK:    amdhsa.kernels:
// CHECK-NEXT:      - .max_flat_workgroup_size: 256
// CHECK-NEXT:        .wavefront_size: 128
// CHECK-NEXT:        .symbol:      'test_kernel@kd'
// CHECK-NEXT:        .kernarg_segment_size: 8
// CHECK-NEXT:        .private_segment_fixed_size: 32
// CHECK-NEXT:        .name:            test_kernel
// CHECK-NEXT:        .language:        OpenCL C
// CHECK-NEXT:        .sgpr_count: 14
// CHECK-NEXT:        .kernarg_segment_align: 64
// CHECK-NEXT:        .vgpr_count: 40
// CHECK-NEXT:        .group_segment_fixed_size: 16
// CHECK-NEXT:        .language_version:
// CHECK-NEXT:          - 2
// CHECK-NEXT:          - 0
// CHECK-NEXT:        .args:
// CHECK-NEXT:          - .type_name:      char
// CHECK-NEXT:            .value_kind:     by_value
// CHECK-NEXT:            .offset:         1
// CHECK-NEXT:            .size:          1
// CHECK-NEXT:            .value_type:     i8
// CHECK-NEXT:          - .value_kind:     hidden_global_offset_x
// CHECK-NEXT:            .offset:         8
// CHECK-NEXT:            .size:          8
// CHECK-NEXT:            .value_type:     i64
// CHECK-NEXT:          - .value_kind:     hidden_global_offset_y
// CHECK-NEXT:            .offset:         8
// CHECK-NEXT:            .size:          8
// CHECK-NEXT:            .value_type:     i64
// CHECK-NEXT:          - .value_kind:     hidden_global_offset_z
// CHECK-NEXT:            .offset:         8
// CHECK-NEXT:            .size:          8
// CHECK-NEXT:            .value_type:     i64
// CHECK-NEXT:          - .value_kind:     hidden_printf_buffer
// CHECK-NEXT:            .offset:         8
// CHECK-NEXT:            .size:          8
// CHECK-NEXT:            .value_type:     i8
// CHECK-NEXT:            .address_space: global
// CHECK:    amdhsa.version:
// CHECK-NEXT:       - 1
// CHECK-NEXT:       - 0
// CHECK:    amdhsa.printf:
// CHECK-NEXT:      - '1:1:4:%d\n'
// CHECK-NEXt:      - '2:1:8:%g\n'
// CHECK:  .end_amdgpu_metadata
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
      .args:
        - .type_name:      char
          .size:          1
          .offset:         1
          .value_kind:     by_value
          .value_type:     i8
        - .size:          8
          .offset:         8
          .value_kind:     hidden_global_offset_x
          .value_type:     i64
        - .size:          8
          .offset:         8
          .value_kind:     hidden_global_offset_y
          .value_type:     i64
        - .size:          8
          .offset:         8
          .value_kind:     hidden_global_offset_z
          .value_type:     i64
        - .size:          8
          .offset:         8
          .value_kind:     hidden_printf_buffer
          .value_type:     i8
          .address_space: global
.end_amdgpu_metadata

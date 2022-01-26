; RUN: llc -mtriple=amdgcn-amd-amdhsa -mcpu=gfx700 --amdhsa-code-object-version=3 -filetype=obj -o - < %s | llvm-readelf --notes - | FileCheck --check-prefix=CHECK %s
; RUN: llc -mtriple=amdgcn-amd-amdhsa -mcpu=gfx802 --amdhsa-code-object-version=3 -filetype=obj -o - < %s | llvm-readelf --notes - | FileCheck --check-prefix=CHECK %s
; RUN: llc -mtriple=amdgcn-amd-amdhsa -mcpu=gfx900 --amdhsa-code-object-version=3 -filetype=obj -o - < %s | llvm-readelf --notes - | FileCheck --check-prefix=CHECK %s
; RUN: llc -mtriple=amdgcn-amd-amdhsa -mcpu=gfx700 --amdhsa-code-object-version=3 -amdgpu-dump-hsa-metadata -amdgpu-verify-hsa-metadata -filetype=obj -o - < %s 2>&1 | FileCheck --check-prefix=PARSER %s
; RUN: llc -mtriple=amdgcn-amd-amdhsa -mcpu=gfx802 --amdhsa-code-object-version=3 -amdgpu-dump-hsa-metadata -amdgpu-verify-hsa-metadata -filetype=obj -o - < %s 2>&1 | FileCheck --check-prefix=PARSER %s
; RUN: llc -mtriple=amdgcn-amd-amdhsa -mcpu=gfx900 --amdhsa-code-object-version=3 -amdgpu-dump-hsa-metadata -amdgpu-verify-hsa-metadata -filetype=obj -o - < %s 2>&1 | FileCheck --check-prefix=PARSER %s

%struct.A = type { i8, float }
%opencl.image1d_t = type opaque
%opencl.image2d_t = type opaque
%opencl.image3d_t = type opaque
%opencl.queue_t = type opaque
%opencl.pipe_t = type opaque
%struct.B = type { i32 addrspace(1)*}
%opencl.clk_event_t = type opaque

@__test_block_invoke_kernel_runtime_handle = external addrspace(1) externally_initialized constant i8 addrspace(1)*

; CHECK:              ---
; CHECK-NEXT: amdhsa.kernels:
; CHECK-NEXT:   - .args:
; CHECK-NEXT:       - .name:           a
; CHECK-NEXT:         .offset:         0
; CHECK-NEXT:         .size:           1
; CHECK-NEXT:         .type_name:      char
; CHECK-NEXT:         .value_kind:     by_value
; CHECK-NEXT:       - .offset:         8
; CHECK-NEXT:         .size:           8
; CHECK-NEXT:         .value_kind:     hidden_global_offset_x
; CHECK-NEXT:       - .offset:         16
; CHECK-NEXT:         .size:           8
; CHECK-NEXT:         .value_kind:     hidden_global_offset_y
; CHECK-NEXT:       - .offset:         24
; CHECK-NEXT:         .size:           8
; CHECK-NEXT:         .value_kind:     hidden_global_offset_z
; CHECK-NEXT:       - .address_space:  global
; CHECK-NEXT:         .offset:         32
; CHECK-NEXT:         .size:           8
; CHECK-NOT:          .value_kind:     hidden_default_queue
; CHECK-NOT:          .value_kind:     hidden_completion_action
; CHECK-NOT:          .value_kind:     hidden_hostcall_buffer
; CHECK-NEXT:         .value_kind:     hidden_printf_buffer
; CHECK:              .value_kind:     hidden_multigrid_sync_arg
; CHECK:          .language:       OpenCL C
; CHECK-NEXT:     .language_version:
; CHECK-NEXT:       - 2
; CHECK-NEXT:       - 0
; CHECK:          .name:           test_char
; CHECK:          .symbol:         test_char.kd
define amdgpu_kernel void @test_char(i8 %a) #0
    !kernel_arg_addr_space !1 !kernel_arg_access_qual !2 !kernel_arg_type !9
    !kernel_arg_base_type !9 !kernel_arg_type_qual !4 {
  ret void
}

; CHECK:        - .args:
; CHECK-NEXT:       - .name:           a
; CHECK-NEXT:         .offset:         0
; CHECK-NEXT:         .size:           1
; CHECK-NEXT:         .type_name:      char
; CHECK-NEXT:         .value_kind:     by_value
; CHECK-NEXT:       - .offset:         8
; CHECK-NEXT:         .size:           8
; CHECK-NEXT:         .value_kind:     hidden_global_offset_x
; CHECK-NEXT:       - .offset:         16
; CHECK-NEXT:         .size:           8
; CHECK-NEXT:         .value_kind:     hidden_global_offset_y
; CHECK-NEXT:       - .offset:         24
; CHECK-NEXT:         .size:           8
; CHECK-NEXT:         .value_kind:     hidden_global_offset_z
; CHECK-NEXT:       - .address_space:  global
; CHECK-NEXT:         .offset:         32
; CHECK-NEXT:         .size:           8
; CHECK-NOT:          .value_kind:     hidden_default_queue
; CHECK-NOT:          .value_kind:     hidden_completion_action
; CHECK-NOT:          .value_kind:     hidden_hostcall_buffer
; CHECK-NEXT:         .value_kind:     hidden_printf_buffer
; CHECK:              .value_kind:     hidden_multigrid_sync_arg
; CHECK:          .language:       OpenCL C
; CHECK-NEXT:     .language_version:
; CHECK-NEXT:       - 2
; CHECK-NEXT:       - 0
; CHECK:          .name:           test_char_byref_constant
; CHECK:          .symbol:         test_char_byref_constant.kd
define amdgpu_kernel void @test_char_byref_constant(i8 addrspace(4)* byref(i8) %a) #0
    !kernel_arg_addr_space !1 !kernel_arg_access_qual !2 !kernel_arg_type !9
    !kernel_arg_base_type !9 !kernel_arg_type_qual !4 {
  ret void
}

; CHECK:        - .args:
; CHECK-NEXT:       - .offset:         0
; CHECK-NEXT:         .size:           1
; CHECK-NEXT:         .type_name:      char
; CHECK-NEXT:         .value_kind:     by_value
; CHECK-NEXT:       - .name:           a
; CHECK-NEXT:         .offset:         512
; CHECK-NEXT:         .size:           1
; CHECK-NEXT:         .type_name:      char
; CHECK-NEXT:         .value_kind:     by_value
; CHECK-NEXT:       - .offset:         520
; CHECK-NEXT:         .size:           8
; CHECK-NEXT:         .value_kind:     hidden_global_offset_x
; CHECK-NEXT:       - .offset:         528
; CHECK-NEXT:         .size:           8
; CHECK-NEXT:         .value_kind:     hidden_global_offset_y
; CHECK-NEXT:       - .offset:         536
; CHECK-NEXT:         .size:           8
; CHECK-NEXT:         .value_kind:     hidden_global_offset_z
; CHECK-NEXT:       - .address_space:  global
; CHECK-NEXT:         .offset:         544
; CHECK-NEXT:         .size:           8
; CHECK-NOT:          .value_kind:     hidden_default_queue
; CHECK-NOT:          .value_kind:     hidden_completion_action
; CHECK-NOT:          .value_kind:     hidden_hostcall_buffer
; CHECK-NEXT:         .value_kind:     hidden_printf_buffer
; CHECK:              .value_kind:     hidden_multigrid_sync_arg
; CHECK:          .language:       OpenCL C
; CHECK-NEXT:     .language_version:
; CHECK-NEXT:       - 2
; CHECK-NEXT:       - 0
; CHECK:          .name:           test_char_byref_constant_align512
; CHECK:          .symbol:         test_char_byref_constant_align512.kd
define amdgpu_kernel void @test_char_byref_constant_align512(i8, i8 addrspace(4)* byref(i8) align(512) %a) #0
    !kernel_arg_addr_space !1 !kernel_arg_access_qual !2 !kernel_arg_type !111
    !kernel_arg_base_type !9 !kernel_arg_type_qual !4 {
  ret void
}

; CHECK:        - .args:
; CHECK-NEXT:       - .name:           a
; CHECK-NEXT:         .offset:         0
; CHECK-NEXT:         .size:           4
; CHECK-NEXT:         .type_name:      ushort2
; CHECK-NEXT:         .value_kind:     by_value
; CHECK-NEXT:       - .offset:         8
; CHECK-NEXT:         .size:           8
; CHECK-NEXT:         .value_kind:     hidden_global_offset_x
; CHECK-NEXT:       - .offset:         16
; CHECK-NEXT:         .size:           8
; CHECK-NEXT:         .value_kind:     hidden_global_offset_y
; CHECK-NEXT:       - .offset:         24
; CHECK-NEXT:         .size:           8
; CHECK-NEXT:         .value_kind:     hidden_global_offset_z
; CHECK-NEXT:       - .address_space:  global
; CHECK-NEXT:         .offset:         32
; CHECK-NEXT:         .size:           8
; CHECK-NEXT:         .value_kind:     hidden_printf_buffer
; CHECK-NEXT:       - .address_space:  global
; CHECK-NEXT:         .offset:         40
; CHECK-NEXT:         .size:           8
; CHECK-NEXT:         .value_kind:     hidden_none
; CHECK-NEXT:       - .address_space:  global
; CHECK-NEXT:         .offset:         48
; CHECK-NEXT:         .size:           8
; CHECK-NEXT:         .value_kind:     hidden_none
; CHECK-NEXT:       - .address_space:  global
; CHECK-NEXT:         .offset:         56
; CHECK-NEXT:         .size:           8
; CHECK-NEXT:         .value_kind:     hidden_multigrid_sync_arg
; CHECK:          .language:       OpenCL C
; CHECK-NEXT:     .language_version:
; CHECK-NEXT:       - 2
; CHECK-NEXT:       - 0
; CHECK:          .name:           test_ushort2
; CHECK:          .symbol:         test_ushort2.kd
define amdgpu_kernel void @test_ushort2(<2 x i16> %a) #0
    !kernel_arg_addr_space !1 !kernel_arg_access_qual !2 !kernel_arg_type !10
    !kernel_arg_base_type !10 !kernel_arg_type_qual !4 {
  ret void
}

; CHECK:        - .args:
; CHECK-NEXT:       - .name:           a
; CHECK-NEXT:         .offset:         0
; CHECK-NEXT:         .size:           16
; CHECK-NEXT:         .type_name:      int3
; CHECK-NEXT:         .value_kind:     by_value
; CHECK-NEXT:       - .offset:         16
; CHECK-NEXT:         .size:           8
; CHECK-NEXT:         .value_kind:     hidden_global_offset_x
; CHECK-NEXT:       - .offset:         24
; CHECK-NEXT:         .size:           8
; CHECK-NEXT:         .value_kind:     hidden_global_offset_y
; CHECK-NEXT:       - .offset:         32
; CHECK-NEXT:         .size:           8
; CHECK-NEXT:         .value_kind:     hidden_global_offset_z
; CHECK-NEXT:       - .address_space:  global
; CHECK-NEXT:         .offset:         40
; CHECK-NEXT:         .size:           8
; CHECK-NEXT:         .value_kind:     hidden_printf_buffer
; CHECK-NEXT:       - .address_space:  global
; CHECK-NEXT:         .offset:         48
; CHECK-NEXT:         .size:           8
; CHECK-NEXT:         .value_kind:     hidden_none
; CHECK-NEXT:       - .address_space:  global
; CHECK-NEXT:         .offset:         56
; CHECK-NEXT:         .size:           8
; CHECK-NEXT:         .value_kind:     hidden_none
; CHECK-NEXT:       - .address_space:  global
; CHECK-NEXT:         .offset:         64
; CHECK-NEXT:         .size:           8
; CHECK-NEXT:         .value_kind:     hidden_multigrid_sync_arg
; CHECK:          .language:       OpenCL C
; CHECK-NEXT:     .language_version:
; CHECK-NEXT:       - 2
; CHECK-NEXT:       - 0
; CHECK:          .name:           test_int3
; CHECK:          .symbol:         test_int3.kd
define amdgpu_kernel void @test_int3(<3 x i32> %a) #0
    !kernel_arg_addr_space !1 !kernel_arg_access_qual !2 !kernel_arg_type !11
    !kernel_arg_base_type !11 !kernel_arg_type_qual !4 {
  ret void
}

; CHECK:        - .args:
; CHECK-NEXT:       - .name:           a
; CHECK-NEXT:         .offset:         0
; CHECK-NEXT:         .size:           32
; CHECK-NEXT:         .type_name:      ulong4
; CHECK-NEXT:         .value_kind:     by_value
; CHECK-NEXT:       - .offset:         32
; CHECK-NEXT:         .size:           8
; CHECK-NEXT:         .value_kind:     hidden_global_offset_x
; CHECK-NEXT:       - .offset:         40
; CHECK-NEXT:         .size:           8
; CHECK-NEXT:         .value_kind:     hidden_global_offset_y
; CHECK-NEXT:       - .offset:         48
; CHECK-NEXT:         .size:           8
; CHECK-NEXT:         .value_kind:     hidden_global_offset_z
; CHECK-NEXT:       - .address_space:  global
; CHECK-NEXT:         .offset:         56
; CHECK-NEXT:         .size:           8
; CHECK-NEXT:         .value_kind:     hidden_printf_buffer
; CHECK-NEXT:       - .address_space:  global
; CHECK-NEXT:         .offset:         64
; CHECK-NEXT:         .size:           8
; CHECK-NEXT:         .value_kind:     hidden_none
; CHECK-NEXT:       - .address_space:  global
; CHECK-NEXT:         .offset:         72
; CHECK-NEXT:         .size:           8
; CHECK-NEXT:         .value_kind:     hidden_none
; CHECK-NEXT:       - .address_space:  global
; CHECK-NEXT:         .offset:         80
; CHECK-NEXT:         .size:           8
; CHECK-NEXT:         .value_kind:     hidden_multigrid_sync_arg
; CHECK:          .language:       OpenCL C
; CHECK-NEXT:     .language_version:
; CHECK-NEXT:       - 2
; CHECK-NEXT:       - 0
; CHECK:          .name:           test_ulong4
; CHECK:          .symbol:         test_ulong4.kd
define amdgpu_kernel void @test_ulong4(<4 x i64> %a) #0
    !kernel_arg_addr_space !1 !kernel_arg_access_qual !2 !kernel_arg_type !12
    !kernel_arg_base_type !12 !kernel_arg_type_qual !4 {
  ret void
}

; CHECK:        - .args:
; CHECK-NEXT:       - .name:           a
; CHECK-NEXT:         .offset:         0
; CHECK-NEXT:         .size:           16
; CHECK-NEXT:         .type_name:      half8
; CHECK-NEXT:         .value_kind:     by_value
; CHECK-NEXT:       - .offset:         16
; CHECK-NEXT:         .size:           8
; CHECK-NEXT:         .value_kind:     hidden_global_offset_x
; CHECK-NEXT:       - .offset:         24
; CHECK-NEXT:         .size:           8
; CHECK-NEXT:         .value_kind:     hidden_global_offset_y
; CHECK-NEXT:       - .offset:         32
; CHECK-NEXT:         .size:           8
; CHECK-NEXT:         .value_kind:     hidden_global_offset_z
; CHECK-NEXT:       - .address_space:  global
; CHECK-NEXT:         .offset:         40
; CHECK-NEXT:         .size:           8
; CHECK-NEXT:         .value_kind:     hidden_printf_buffer
; CHECK-NEXT:       - .address_space:  global
; CHECK-NEXT:         .offset:         48
; CHECK-NEXT:         .size:           8
; CHECK-NEXT:         .value_kind:     hidden_none
; CHECK-NEXT:       - .address_space:  global
; CHECK-NEXT:         .offset:         56
; CHECK-NEXT:         .size:           8
; CHECK-NEXT:         .value_kind:     hidden_none
; CHECK-NEXT:       - .address_space:  global
; CHECK-NEXT:         .offset:         64
; CHECK-NEXT:         .size:           8
; CHECK-NEXT:         .value_kind:     hidden_multigrid_sync_arg
; CHECK:          .language:       OpenCL C
; CHECK-NEXT:     .language_version:
; CHECK-NEXT:       - 2
; CHECK-NEXT:       - 0
; CHECK:          .name:           test_half8
; CHECK:          .symbol:         test_half8.kd
define amdgpu_kernel void @test_half8(<8 x half> %a) #0
    !kernel_arg_addr_space !1 !kernel_arg_access_qual !2 !kernel_arg_type !13
    !kernel_arg_base_type !13 !kernel_arg_type_qual !4 {
  ret void
}

; CHECK:        - .args:
; CHECK-NEXT:       - .name:           a
; CHECK-NEXT:         .offset:         0
; CHECK-NEXT:         .size:           64
; CHECK-NEXT:         .type_name:      float16
; CHECK-NEXT:         .value_kind:     by_value
; CHECK-NEXT:       - .offset:         64
; CHECK-NEXT:         .size:           8
; CHECK-NEXT:         .value_kind:     hidden_global_offset_x
; CHECK-NEXT:       - .offset:         72
; CHECK-NEXT:         .size:           8
; CHECK-NEXT:         .value_kind:     hidden_global_offset_y
; CHECK-NEXT:       - .offset:         80
; CHECK-NEXT:         .size:           8
; CHECK-NEXT:         .value_kind:     hidden_global_offset_z
; CHECK-NEXT:       - .address_space:  global
; CHECK-NEXT:         .offset:         88
; CHECK-NEXT:         .size:           8
; CHECK-NEXT:         .value_kind:     hidden_printf_buffer
; CHECK-NEXT:       - .address_space:  global
; CHECK-NEXT:         .offset:         96
; CHECK-NEXT:         .size:           8
; CHECK-NEXT:         .value_kind:     hidden_none
; CHECK-NEXT:       - .address_space:  global
; CHECK-NEXT:         .offset:         104
; CHECK-NEXT:         .size:           8
; CHECK-NEXT:         .value_kind:     hidden_none
; CHECK-NEXT:       - .address_space:  global
; CHECK-NEXT:         .offset:         112
; CHECK-NEXT:         .size:           8
; CHECK-NEXT:         .value_kind:     hidden_multigrid_sync_arg
; CHECK:          .language:       OpenCL C
; CHECK-NEXT:     .language_version:
; CHECK-NEXT:       - 2
; CHECK-NEXT:       - 0
; CHECK:          .name:           test_float16
; CHECK:          .symbol:         test_float16.kd
define amdgpu_kernel void @test_float16(<16 x float> %a) #0
    !kernel_arg_addr_space !1 !kernel_arg_access_qual !2 !kernel_arg_type !14
    !kernel_arg_base_type !14 !kernel_arg_type_qual !4 {
  ret void
}

; CHECK:        - .args:
; CHECK-NEXT:       - .name:           a
; CHECK-NEXT:         .offset:         0
; CHECK-NEXT:         .size:           128
; CHECK-NEXT:         .type_name:      double16
; CHECK-NEXT:         .value_kind:     by_value
; CHECK-NEXT:       - .offset:         128
; CHECK-NEXT:         .size:           8
; CHECK-NEXT:         .value_kind:     hidden_global_offset_x
; CHECK-NEXT:       - .offset:         136
; CHECK-NEXT:         .size:           8
; CHECK-NEXT:         .value_kind:     hidden_global_offset_y
; CHECK-NEXT:       - .offset:         144
; CHECK-NEXT:         .size:           8
; CHECK-NEXT:         .value_kind:     hidden_global_offset_z
; CHECK-NEXT:       - .address_space:  global
; CHECK-NEXT:         .offset:         152
; CHECK-NEXT:         .size:           8
; CHECK-NEXT:         .value_kind:     hidden_printf_buffer
; CHECK-NEXT:       - .address_space:  global
; CHECK-NEXT:         .offset:         160
; CHECK-NEXT:         .size:           8
; CHECK-NEXT:         .value_kind:     hidden_none
; CHECK-NEXT:       - .address_space:  global
; CHECK-NEXT:         .offset:         168
; CHECK-NEXT:         .size:           8
; CHECK-NEXT:         .value_kind:     hidden_none
; CHECK-NEXT:       - .address_space:  global
; CHECK-NEXT:         .offset:         176
; CHECK-NEXT:         .size:           8
; CHECK-NEXT:         .value_kind:     hidden_multigrid_sync_arg
; CHECK:          .language:       OpenCL C
; CHECK-NEXT:     .language_version:
; CHECK-NEXT:       - 2
; CHECK-NEXT:       - 0
; CHECK:          .name:           test_double16
; CHECK:          .symbol:         test_double16.kd
define amdgpu_kernel void @test_double16(<16 x double> %a) #0
    !kernel_arg_addr_space !1 !kernel_arg_access_qual !2 !kernel_arg_type !15
    !kernel_arg_base_type !15 !kernel_arg_type_qual !4 {
  ret void
}

; CHECK:        - .args:
; CHECK-NEXT:       - .address_space:  global
; CHECK-NEXT:         .name:           a
; CHECK-NEXT:         .offset:         0
; CHECK-NEXT:         .size:           8
; CHECK-NEXT:         .type_name:      'int  addrspace(5)*'
; CHECK-NEXT:         .value_kind:     global_buffer
; CHECK-NEXT:       - .offset:         8
; CHECK-NEXT:         .size:           8
; CHECK-NEXT:         .value_kind:     hidden_global_offset_x
; CHECK-NEXT:       - .offset:         16
; CHECK-NEXT:         .size:           8
; CHECK-NEXT:         .value_kind:     hidden_global_offset_y
; CHECK-NEXT:       - .offset:         24
; CHECK-NEXT:         .size:           8
; CHECK-NEXT:         .value_kind:     hidden_global_offset_z
; CHECK-NEXT:       - .address_space:  global
; CHECK-NEXT:         .offset:         32
; CHECK-NEXT:         .size:           8
; CHECK-NEXT:         .value_kind:     hidden_printf_buffer
; CHECK-NEXT:       - .address_space:  global
; CHECK-NEXT:         .offset:         40
; CHECK-NEXT:         .size:           8
; CHECK-NEXT:         .value_kind:     hidden_none
; CHECK-NEXT:       - .address_space:  global
; CHECK-NEXT:         .offset:         48
; CHECK-NEXT:         .size:           8
; CHECK-NEXT:         .value_kind:     hidden_none
; CHECK-NEXT:       - .address_space:  global
; CHECK-NEXT:         .offset:         56
; CHECK-NEXT:         .size:           8
; CHECK-NEXT:         .value_kind:     hidden_multigrid_sync_arg
; CHECK:          .language:       OpenCL C
; CHECK-NEXT:     .language_version:
; CHECK-NEXT:       - 2
; CHECK-NEXT:       - 0
; CHECK:          .name:           test_pointer
; CHECK:          .symbol:         test_pointer.kd
define amdgpu_kernel void @test_pointer(i32 addrspace(1)* %a) #0
    !kernel_arg_addr_space !1 !kernel_arg_access_qual !2 !kernel_arg_type !16
    !kernel_arg_base_type !16 !kernel_arg_type_qual !4 {
  ret void
}

; CHECK:        - .args:
; CHECK-NEXT:       - .address_space:  global
; CHECK-NEXT:         .name:           a
; CHECK-NEXT:         .offset:         0
; CHECK-NEXT:         .size:           8
; CHECK-NEXT:         .type_name:      image2d_t
; CHECK-NEXT:         .value_kind:     image
; CHECK-NEXT:       - .offset:         8
; CHECK-NEXT:         .size:           8
; CHECK-NEXT:         .value_kind:     hidden_global_offset_x
; CHECK-NEXT:       - .offset:         16
; CHECK-NEXT:         .size:           8
; CHECK-NEXT:         .value_kind:     hidden_global_offset_y
; CHECK-NEXT:       - .offset:         24
; CHECK-NEXT:         .size:           8
; CHECK-NEXT:         .value_kind:     hidden_global_offset_z
; CHECK-NEXT:       - .address_space:  global
; CHECK-NEXT:         .offset:         32
; CHECK-NEXT:         .size:           8
; CHECK-NEXT:         .value_kind:     hidden_printf_buffer
; CHECK-NEXT:       - .address_space:  global
; CHECK-NEXT:         .offset:         40
; CHECK-NEXT:         .size:           8
; CHECK-NEXT:         .value_kind:     hidden_none
; CHECK-NEXT:       - .address_space:  global
; CHECK-NEXT:         .offset:         48
; CHECK-NEXT:         .size:           8
; CHECK-NEXT:         .value_kind:     hidden_none
; CHECK-NEXT:       - .address_space:  global
; CHECK-NEXT:         .offset:         56
; CHECK-NEXT:         .size:           8
; CHECK-NEXT:         .value_kind:     hidden_multigrid_sync_arg
; CHECK:          .language:       OpenCL C
; CHECK-NEXT:     .language_version:
; CHECK-NEXT:       - 2
; CHECK-NEXT:       - 0
; CHECK:          .name:           test_image
; CHECK:          .symbol:         test_image.kd
define amdgpu_kernel void @test_image(%opencl.image2d_t addrspace(1)* %a) #0
    !kernel_arg_addr_space !1 !kernel_arg_access_qual !2 !kernel_arg_type !17
    !kernel_arg_base_type !17 !kernel_arg_type_qual !4 {
  ret void
}

; CHECK:        - .args:
; CHECK-NEXT:       - .name:           a
; CHECK-NEXT:         .offset:         0
; CHECK-NEXT:         .size:           4
; CHECK-NEXT:         .type_name:      sampler_t
; CHECK-NEXT:         .value_kind:     sampler
; CHECK-NEXT:       - .offset:         8
; CHECK-NEXT:         .size:           8
; CHECK-NEXT:         .value_kind:     hidden_global_offset_x
; CHECK-NEXT:       - .offset:         16
; CHECK-NEXT:         .size:           8
; CHECK-NEXT:         .value_kind:     hidden_global_offset_y
; CHECK-NEXT:       - .offset:         24
; CHECK-NEXT:         .size:           8
; CHECK-NEXT:         .value_kind:     hidden_global_offset_z
; CHECK-NEXT:       - .address_space:  global
; CHECK-NEXT:         .offset:         32
; CHECK-NEXT:         .size:           8
; CHECK-NEXT:         .value_kind:     hidden_printf_buffer
; CHECK-NEXT:       - .address_space:  global
; CHECK-NEXT:         .offset:         40
; CHECK-NEXT:         .size:           8
; CHECK-NEXT:         .value_kind:     hidden_none
; CHECK-NEXT:       - .address_space:  global
; CHECK-NEXT:         .offset:         48
; CHECK-NEXT:         .size:           8
; CHECK-NEXT:         .value_kind:     hidden_none
; CHECK-NEXT:       - .address_space:  global
; CHECK-NEXT:         .offset:         56
; CHECK-NEXT:         .size:           8
; CHECK-NEXT:         .value_kind:     hidden_multigrid_sync_arg
; CHECK:          .language:       OpenCL C
; CHECK-NEXT:     .language_version:
; CHECK-NEXT:       - 2
; CHECK-NEXT:       - 0
; CHECK:          .name:           test_sampler
; CHECK:          .symbol:         test_sampler.kd
define amdgpu_kernel void @test_sampler(i32 %a) #0
    !kernel_arg_addr_space !1 !kernel_arg_access_qual !2 !kernel_arg_type !18
    !kernel_arg_base_type !18 !kernel_arg_type_qual !4 {
  ret void
}

; CHECK:        - .args:
; CHECK-NEXT:       - .address_space:  global
; CHECK-NEXT:         .name:           a
; CHECK-NEXT:         .offset:         0
; CHECK-NEXT:         .size:           8
; CHECK-NEXT:         .type_name:      queue_t
; CHECK-NEXT:         .value_kind:     queue
; CHECK-NEXT:       - .offset:         8
; CHECK-NEXT:         .size:           8
; CHECK-NEXT:         .value_kind:     hidden_global_offset_x
; CHECK-NEXT:       - .offset:         16
; CHECK-NEXT:         .size:           8
; CHECK-NEXT:         .value_kind:     hidden_global_offset_y
; CHECK-NEXT:       - .offset:         24
; CHECK-NEXT:         .size:           8
; CHECK-NEXT:         .value_kind:     hidden_global_offset_z
; CHECK-NEXT:       - .address_space:  global
; CHECK-NEXT:         .offset:         32
; CHECK-NEXT:         .size:           8
; CHECK-NEXT:         .value_kind:     hidden_printf_buffer
; CHECK-NEXT:       - .address_space:  global
; CHECK-NEXT:         .offset:         40
; CHECK-NEXT:         .size:           8
; CHECK-NEXT:         .value_kind:     hidden_none
; CHECK-NEXT:       - .address_space:  global
; CHECK-NEXT:         .offset:         48
; CHECK-NEXT:         .size:           8
; CHECK-NEXT:         .value_kind:     hidden_none
; CHECK-NEXT:       - .address_space:  global
; CHECK-NEXT:         .offset:         56
; CHECK-NEXT:         .size:           8
; CHECK-NEXT:         .value_kind:     hidden_multigrid_sync_arg
; CHECK:          .language:       OpenCL C
; CHECK-NEXT:     .language_version:
; CHECK-NEXT:       - 2
; CHECK-NEXT:       - 0
; CHECK:          .name:           test_queue
; CHECK:          .symbol:         test_queue.kd
define amdgpu_kernel void @test_queue(%opencl.queue_t addrspace(1)* %a) #0
    !kernel_arg_addr_space !1 !kernel_arg_access_qual !2 !kernel_arg_type !19
    !kernel_arg_base_type !19 !kernel_arg_type_qual !4 {
  ret void
}

; CHECK:        - .args:
; CHECK-NEXT:         .name:           a
; CHECK-NEXT:         .offset:         0
; CHECK-NEXT:         .size:           8
; CHECK-NEXT:         .type_name:      struct A
; CHECK-NEXT:         .value_kind:     by_value
; CHECK-NEXT:       - .offset:         8
; CHECK-NEXT:         .size:           8
; CHECK-NEXT:         .value_kind:     hidden_global_offset_x
; CHECK-NEXT:       - .offset:         16
; CHECK-NEXT:         .size:           8
; CHECK-NEXT:         .value_kind:     hidden_global_offset_y
; CHECK-NEXT:       - .offset:         24
; CHECK-NEXT:         .size:           8
; CHECK-NEXT:         .value_kind:     hidden_global_offset_z
; CHECK-NEXT:       - .address_space:  global
; CHECK-NEXT:         .offset:         32
; CHECK-NEXT:         .size:           8
; CHECK-NEXT:         .value_kind:     hidden_printf_buffer
; CHECK-NEXT:       - .address_space:  global
; CHECK-NEXT:         .offset:         40
; CHECK-NEXT:         .size:           8
; CHECK-NEXT:         .value_kind:     hidden_none
; CHECK-NEXT:       - .address_space:  global
; CHECK-NEXT:         .offset:         48
; CHECK-NEXT:         .size:           8
; CHECK-NEXT:         .value_kind:     hidden_none
; CHECK-NEXT:       - .address_space:  global
; CHECK-NEXT:         .offset:         56
; CHECK-NEXT:         .size:           8
; CHECK-NEXT:         .value_kind:     hidden_multigrid_sync_arg
; CHECK:          .language:       OpenCL C
; CHECK-NEXT:     .language_version:
; CHECK-NEXT:       - 2
; CHECK-NEXT:       - 0
; CHECK:          .name:           test_struct
; CHECK:          .symbol:         test_struct.kd
define amdgpu_kernel void @test_struct(%struct.A %a) #0
    !kernel_arg_addr_space !1 !kernel_arg_access_qual !2 !kernel_arg_type !20
    !kernel_arg_base_type !20 !kernel_arg_type_qual !4 {
  ret void
}

; CHECK:        - .args:
; CHECK-NEXT:         .name:           a
; CHECK-NEXT:         .offset:         0
; CHECK-NEXT:         .size:           8
; CHECK-NEXT:         .type_name:      struct A
; CHECK-NEXT:         .value_kind:     by_value
; CHECK-NEXT:       - .offset:         8
; CHECK-NEXT:         .size:           8
; CHECK-NEXT:         .value_kind:     hidden_global_offset_x
; CHECK-NEXT:       - .offset:         16
; CHECK-NEXT:         .size:           8
; CHECK-NEXT:         .value_kind:     hidden_global_offset_y
; CHECK-NEXT:       - .offset:         24
; CHECK-NEXT:         .size:           8
; CHECK-NEXT:         .value_kind:     hidden_global_offset_z
; CHECK-NEXT:       - .address_space:  global
; CHECK-NEXT:         .offset:         32
; CHECK-NEXT:         .size:           8
; CHECK-NEXT:         .value_kind:     hidden_printf_buffer
; CHECK-NEXT:       - .address_space:  global
; CHECK-NEXT:         .offset:         40
; CHECK-NEXT:         .size:           8
; CHECK-NEXT:         .value_kind:     hidden_none
; CHECK-NEXT:       - .address_space:  global
; CHECK-NEXT:         .offset:         48
; CHECK-NEXT:         .size:           8
; CHECK-NEXT:         .value_kind:     hidden_none
; CHECK-NEXT:       - .address_space:  global
; CHECK-NEXT:         .offset:         56
; CHECK-NEXT:         .size:           8
; CHECK-NEXT:         .value_kind:     hidden_multigrid_sync_arg
; CHECK:          .language:       OpenCL C
; CHECK-NEXT:     .language_version:
; CHECK-NEXT:       - 2
; CHECK-NEXT:       - 0
; CHECK:          .name:           test_struct_byref_constant
; CHECK:          .symbol:         test_struct_byref_constant.kd
define amdgpu_kernel void @test_struct_byref_constant(%struct.A addrspace(4)* byref(%struct.A) %a) #0
    !kernel_arg_addr_space !1 !kernel_arg_access_qual !2 !kernel_arg_type !20
    !kernel_arg_base_type !20 !kernel_arg_type_qual !4 {
  ret void
}

; CHECK:        - .args:
; CHECK-NEXT:         .name:           a
; CHECK-NEXT:         .offset:         0
; CHECK-NEXT:         .size:           32
; CHECK-NEXT:         .type_name:      struct A
; CHECK-NEXT:         .value_kind:     by_value
; CHECK-NEXT:       - .offset:         32
; CHECK-NEXT:         .size:           8
; CHECK-NEXT:         .value_kind:     hidden_global_offset_x
; CHECK-NEXT:       - .offset:         40
; CHECK-NEXT:         .size:           8
; CHECK-NEXT:         .value_kind:     hidden_global_offset_y
; CHECK-NEXT:       - .offset:         48
; CHECK-NEXT:         .size:           8
; CHECK-NEXT:         .value_kind:     hidden_global_offset_z
; CHECK-NEXT:       - .address_space:  global
; CHECK-NEXT:         .offset:         56
; CHECK-NEXT:         .size:           8
; CHECK-NEXT:         .value_kind:     hidden_printf_buffer
; CHECK-NEXT:       - .address_space:  global
; CHECK-NEXT:         .offset:         64
; CHECK-NEXT:         .size:           8
; CHECK-NEXT:         .value_kind:     hidden_none
; CHECK-NEXT:       - .address_space:  global
; CHECK-NEXT:         .offset:         72
; CHECK-NEXT:         .size:           8
; CHECK-NEXT:         .value_kind:     hidden_none
; CHECK-NEXT:       - .address_space:  global
; CHECK-NEXT:         .offset:         80
; CHECK-NEXT:         .size:           8
; CHECK-NEXT:         .value_kind:     hidden_multigrid_sync_arg
; CHECK:          .language:       OpenCL C
; CHECK-NEXT:     .language_version:
; CHECK-NEXT:       - 2
; CHECK-NEXT:       - 0
; CHECK:          .name:           test_array
; CHECK:          .symbol:         test_array.kd
define amdgpu_kernel void @test_array([32 x i8] %a) #0
    !kernel_arg_addr_space !1 !kernel_arg_access_qual !2 !kernel_arg_type !20
    !kernel_arg_base_type !20 !kernel_arg_type_qual !4 {
  ret void
}

; CHECK:        - .args:
; CHECK-NEXT:         .name:           a
; CHECK-NEXT:         .offset:         0
; CHECK-NEXT:         .size:           32
; CHECK-NEXT:         .type_name:      struct A
; CHECK-NEXT:         .value_kind:     by_value
; CHECK-NEXT:       - .offset:         32
; CHECK-NEXT:         .size:           8
; CHECK-NEXT:         .value_kind:     hidden_global_offset_x
; CHECK-NEXT:       - .offset:         40
; CHECK-NEXT:         .size:           8
; CHECK-NEXT:         .value_kind:     hidden_global_offset_y
; CHECK-NEXT:       - .offset:         48
; CHECK-NEXT:         .size:           8
; CHECK-NEXT:         .value_kind:     hidden_global_offset_z
; CHECK-NEXT:       - .address_space:  global
; CHECK-NEXT:         .offset:         56
; CHECK-NEXT:         .size:           8
; CHECK-NEXT:         .value_kind:     hidden_printf_buffer
; CHECK-NEXT:       - .address_space:  global
; CHECK-NEXT:         .offset:         64
; CHECK-NEXT:         .size:           8
; CHECK-NEXT:         .value_kind:     hidden_none
; CHECK-NEXT:       - .address_space:  global
; CHECK-NEXT:         .offset:         72
; CHECK-NEXT:         .size:           8
; CHECK-NEXT:         .value_kind:     hidden_none
; CHECK-NEXT:       - .address_space:  global
; CHECK-NEXT:         .offset:         80
; CHECK-NEXT:         .size:           8
; CHECK-NEXT:         .value_kind:     hidden_multigrid_sync_arg
; CHECK:          .language:       OpenCL C
; CHECK-NEXT:     .language_version:
; CHECK-NEXT:       - 2
; CHECK-NEXT:       - 0
; CHECK:          .name:           test_array_byref_constant
; CHECK:          .symbol:         test_array_byref_constant.kd
define amdgpu_kernel void @test_array_byref_constant([32 x i8] addrspace(4)* byref([32 x i8]) %a) #0
    !kernel_arg_addr_space !1 !kernel_arg_access_qual !2 !kernel_arg_type !20
    !kernel_arg_base_type !20 !kernel_arg_type_qual !4 {
  ret void
}

; CHECK:        - .args:
; CHECK-NEXT:       - .name:           a
; CHECK-NEXT:         .offset:         0
; CHECK-NEXT:         .size:           16
; CHECK-NEXT:         .type_name:      i128
; CHECK-NEXT:         .value_kind:     by_value
; CHECK-NEXT:       - .offset:         16
; CHECK-NEXT:         .size:           8
; CHECK-NEXT:         .value_kind:     hidden_global_offset_x
; CHECK-NEXT:       - .offset:         24
; CHECK-NEXT:         .size:           8
; CHECK-NEXT:         .value_kind:     hidden_global_offset_y
; CHECK-NEXT:       - .offset:         32
; CHECK-NEXT:         .size:           8
; CHECK-NEXT:         .value_kind:     hidden_global_offset_z
; CHECK-NEXT:       - .address_space:  global
; CHECK-NEXT:         .offset:         40
; CHECK-NEXT:         .size:           8
; CHECK-NEXT:         .value_kind:     hidden_printf_buffer
; CHECK-NEXT:       - .address_space:  global
; CHECK-NEXT:         .offset:         48
; CHECK-NEXT:         .size:           8
; CHECK-NEXT:         .value_kind:     hidden_none
; CHECK-NEXT:       - .address_space:  global
; CHECK-NEXT:         .offset:         56
; CHECK-NEXT:         .size:           8
; CHECK-NEXT:         .value_kind:     hidden_none
; CHECK-NEXT:       - .address_space:  global
; CHECK-NEXT:         .offset:         64
; CHECK-NEXT:         .size:           8
; CHECK-NEXT:         .value_kind:     hidden_multigrid_sync_arg
; CHECK:          .language:       OpenCL C
; CHECK-NEXT:     .language_version:
; CHECK-NEXT:       - 2
; CHECK-NEXT:       - 0
; CHECK:          .name:           test_i128
; CHECK:          .symbol:         test_i128.kd
define amdgpu_kernel void @test_i128(i128 %a) #0
    !kernel_arg_addr_space !1 !kernel_arg_access_qual !2 !kernel_arg_type !21
    !kernel_arg_base_type !21 !kernel_arg_type_qual !4 {
  ret void
}

; CHECK:        - .args:
; CHECK-NEXT:       - .name:           a
; CHECK-NEXT:         .offset:         0
; CHECK-NEXT:         .size:           4
; CHECK-NEXT:         .type_name:      int
; CHECK-NEXT:         .value_kind:     by_value
; CHECK-NEXT:       - .name:           b
; CHECK-NEXT:         .offset:         4
; CHECK-NEXT:         .size:           4
; CHECK-NEXT:         .type_name:      short2
; CHECK-NEXT:         .value_kind:     by_value
; CHECK-NEXT:       - .name:           c
; CHECK-NEXT:         .offset:         8
; CHECK-NEXT:         .size:           4
; CHECK-NEXT:         .type_name:      char3
; CHECK-NEXT:         .value_kind:     by_value
; CHECK-NEXT:       - .offset:         16
; CHECK-NEXT:         .size:           8
; CHECK-NEXT:         .value_kind:     hidden_global_offset_x
; CHECK-NEXT:       - .offset:         24
; CHECK-NEXT:         .size:           8
; CHECK-NEXT:         .value_kind:     hidden_global_offset_y
; CHECK-NEXT:       - .offset:         32
; CHECK-NEXT:         .size:           8
; CHECK-NEXT:         .value_kind:     hidden_global_offset_z
; CHECK-NEXT:       - .address_space:  global
; CHECK-NEXT:         .offset:         40
; CHECK-NEXT:         .size:           8
; CHECK-NEXT:         .value_kind:     hidden_printf_buffer
; CHECK-NEXT:       - .address_space:  global
; CHECK-NEXT:         .offset:         48
; CHECK-NEXT:         .size:           8
; CHECK-NEXT:         .value_kind:     hidden_none
; CHECK-NEXT:       - .address_space:  global
; CHECK-NEXT:         .offset:         56
; CHECK-NEXT:         .size:           8
; CHECK-NEXT:         .value_kind:     hidden_none
; CHECK-NEXT:       - .address_space:  global
; CHECK-NEXT:         .offset:         64
; CHECK-NEXT:         .size:           8
; CHECK-NEXT:         .value_kind:     hidden_multigrid_sync_arg
; CHECK:          .language:       OpenCL C
; CHECK-NEXT:     .language_version:
; CHECK-NEXT:       - 2
; CHECK-NEXT:       - 0
; CHECK:          .name:           test_multi_arg
; CHECK:          .symbol:         test_multi_arg.kd
define amdgpu_kernel void @test_multi_arg(i32 %a, <2 x i16> %b, <3 x i8> %c) #0
    !kernel_arg_addr_space !22 !kernel_arg_access_qual !23 !kernel_arg_type !24
    !kernel_arg_base_type !24 !kernel_arg_type_qual !25 {
  ret void
}

; CHECK:        - .args:
; CHECK-NEXT:       - .address_space:  global
; CHECK-NEXT:         .name:           g
; CHECK-NEXT:         .offset:         0
; CHECK-NEXT:         .size:           8
; CHECK-NEXT:         .type_name:      'int  addrspace(5)*'
; CHECK-NEXT:         .value_kind:     global_buffer
; CHECK-NEXT:       - .address_space:  constant
; CHECK-NEXT:         .name:           c
; CHECK-NEXT:         .offset:         8
; CHECK-NEXT:         .size:           8
; CHECK-NEXT:         .type_name:      'int  addrspace(5)*'
; CHECK-NEXT:         .value_kind:     global_buffer
; CHECK-NEXT:       - .address_space:  local
; CHECK-NEXT:         .name:           l
; CHECK-NEXT:         .offset:         16
; CHECK-NEXT:         .pointee_align:  4
; CHECK-NEXT:         .size:           4
; CHECK-NEXT:         .type_name:      'int  addrspace(5)*'
; CHECK-NEXT:         .value_kind:     dynamic_shared_pointer
; CHECK-NEXT:       - .offset:         24
; CHECK-NEXT:         .size:           8
; CHECK-NEXT:         .value_kind:     hidden_global_offset_x
; CHECK-NEXT:       - .offset:         32
; CHECK-NEXT:         .size:           8
; CHECK-NEXT:         .value_kind:     hidden_global_offset_y
; CHECK-NEXT:       - .offset:         40
; CHECK-NEXT:         .size:           8
; CHECK-NEXT:         .value_kind:     hidden_global_offset_z
; CHECK-NEXT:       - .address_space:  global
; CHECK-NEXT:         .offset:         48
; CHECK-NEXT:         .size:           8
; CHECK-NEXT:         .value_kind:     hidden_printf_buffer
; CHECK-NEXT:       - .address_space:  global
; CHECK-NEXT:         .offset:         56
; CHECK-NEXT:         .size:           8
; CHECK-NEXT:         .value_kind:     hidden_none
; CHECK-NEXT:       - .address_space:  global
; CHECK-NEXT:         .offset:         64
; CHECK-NEXT:         .size:           8
; CHECK-NEXT:         .value_kind:     hidden_none
; CHECK-NEXT:       - .address_space:  global
; CHECK-NEXT:         .offset:         72
; CHECK-NEXT:         .size:           8
; CHECK-NEXT:         .value_kind:     hidden_multigrid_sync_arg
; CHECK:          .language:       OpenCL C
; CHECK-NEXT:     .language_version:
; CHECK-NEXT:       - 2
; CHECK-NEXT:       - 0
; CHECK:          .name:           test_addr_space
; CHECK:          .symbol:         test_addr_space.kd
define amdgpu_kernel void @test_addr_space(i32 addrspace(1)* %g,
                                           i32 addrspace(4)* %c,
                                           i32 addrspace(3)* align 4 %l) #0
    !kernel_arg_addr_space !50 !kernel_arg_access_qual !23 !kernel_arg_type !51
    !kernel_arg_base_type !51 !kernel_arg_type_qual !25 {
  ret void
}

; CHECK:        - .args:
; CHECK-NEXT:       - .address_space:  global
; CHECK-NEXT:         .is_volatile:    true
; CHECK-NEXT:         .name:           a
; CHECK-NEXT:         .offset:         0
; CHECK-NEXT:         .size:           8
; CHECK-NEXT:         .type_name:      'int  addrspace(5)*'
; CHECK-NEXT:         .value_kind:     global_buffer
; CHECK-NEXT:       - .address_space:  global
; CHECK-NEXT:         .is_const:       true
; CHECK-NEXT:         .is_restrict:    true
; CHECK-NEXT:         .name:           b
; CHECK-NEXT:         .offset:         8
; CHECK-NEXT:         .size:           8
; CHECK-NEXT:         .type_name:      'int  addrspace(5)*'
; CHECK-NEXT:         .value_kind:     global_buffer
; CHECK-NEXT:       - .address_space:  global
; CHECK-NEXT:         .is_pipe:        true
; CHECK-NEXT:         .name:           c
; CHECK-NEXT:         .offset:         16
; CHECK-NEXT:         .size:           8
; CHECK-NEXT:         .type_name:      'int  addrspace(5)*'
; CHECK-NEXT:         .value_kind:     pipe
; CHECK-NEXT:       - .offset:         24
; CHECK-NEXT:         .size:           8
; CHECK-NEXT:         .value_kind:     hidden_global_offset_x
; CHECK-NEXT:       - .offset:         32
; CHECK-NEXT:         .size:           8
; CHECK-NEXT:         .value_kind:     hidden_global_offset_y
; CHECK-NEXT:       - .offset:         40
; CHECK-NEXT:         .size:           8
; CHECK-NEXT:         .value_kind:     hidden_global_offset_z
; CHECK-NEXT:       - .address_space:  global
; CHECK-NEXT:         .offset:         48
; CHECK-NEXT:         .size:           8
; CHECK-NEXT:         .value_kind:     hidden_printf_buffer
; CHECK-NEXT:       - .address_space:  global
; CHECK-NEXT:         .offset:         56
; CHECK-NEXT:         .size:           8
; CHECK-NEXT:         .value_kind:     hidden_none
; CHECK-NEXT:       - .address_space:  global
; CHECK-NEXT:         .offset:         64
; CHECK-NEXT:         .size:           8
; CHECK-NEXT:         .value_kind:     hidden_none
; CHECK-NEXT:       - .address_space:  global
; CHECK-NEXT:         .offset:         72
; CHECK-NEXT:         .size:           8
; CHECK-NEXT:         .value_kind:     hidden_multigrid_sync_arg
; CHECK:          .language:       OpenCL C
; CHECK-NEXT:     .language_version:
; CHECK-NEXT:       - 2
; CHECK-NEXT:       - 0
; CHECK:          .name:           test_type_qual
; CHECK:          .symbol:         test_type_qual.kd
define amdgpu_kernel void @test_type_qual(i32 addrspace(1)* %a,
                                          i32 addrspace(1)* %b,
                                          %opencl.pipe_t addrspace(1)* %c) #0
    !kernel_arg_addr_space !22 !kernel_arg_access_qual !23 !kernel_arg_type !51
    !kernel_arg_base_type !51 !kernel_arg_type_qual !70 {
  ret void
}

; CHECK:        - .args:
; CHECK-NEXT:       - .access:         read_only
; CHECK-NEXT:         .address_space:  global
; CHECK-NEXT:         .name:           ro
; CHECK-NEXT:         .offset:         0
; CHECK-NEXT:         .size:           8
; CHECK-NEXT:         .type_name:      image1d_t
; CHECK-NEXT:         .value_kind:     image
; CHECK-NEXT:       - .access:         write_only
; CHECK-NEXT:         .address_space:  global
; CHECK-NEXT:         .name:           wo
; CHECK-NEXT:         .offset:         8
; CHECK-NEXT:         .size:           8
; CHECK-NEXT:         .type_name:      image2d_t
; CHECK-NEXT:         .value_kind:     image
; CHECK-NEXT:       - .access:         read_write
; CHECK-NEXT:         .address_space:  global
; CHECK-NEXT:         .name:           rw
; CHECK-NEXT:         .offset:         16
; CHECK-NEXT:         .size:           8
; CHECK-NEXT:         .type_name:      image3d_t
; CHECK-NEXT:         .value_kind:     image
; CHECK-NEXT:       - .offset:         24
; CHECK-NEXT:         .size:           8
; CHECK-NEXT:         .value_kind:     hidden_global_offset_x
; CHECK-NEXT:       - .offset:         32
; CHECK-NEXT:         .size:           8
; CHECK-NEXT:         .value_kind:     hidden_global_offset_y
; CHECK-NEXT:       - .offset:         40
; CHECK-NEXT:         .size:           8
; CHECK-NEXT:         .value_kind:     hidden_global_offset_z
; CHECK-NEXT:       - .address_space:  global
; CHECK-NEXT:         .offset:         48
; CHECK-NEXT:         .size:           8
; CHECK-NEXT:         .value_kind:     hidden_printf_buffer
; CHECK-NEXT:       - .address_space:  global
; CHECK-NEXT:         .offset:         56
; CHECK-NEXT:         .size:           8
; CHECK-NEXT:         .value_kind:     hidden_none
; CHECK-NEXT:       - .address_space:  global
; CHECK-NEXT:         .offset:         64
; CHECK-NEXT:         .size:           8
; CHECK-NEXT:         .value_kind:     hidden_none
; CHECK-NEXT:       - .address_space:  global
; CHECK-NEXT:         .offset:         72
; CHECK-NEXT:         .size:           8
; CHECK-NEXT:         .value_kind:     hidden_multigrid_sync_arg
; CHECK:          .language:       OpenCL C
; CHECK-NEXT:     .language_version:
; CHECK-NEXT:       - 2
; CHECK-NEXT:       - 0
; CHECK:          .name:           test_access_qual
; CHECK:          .symbol:         test_access_qual.kd
define amdgpu_kernel void @test_access_qual(%opencl.image1d_t addrspace(1)* %ro,
                                            %opencl.image2d_t addrspace(1)* %wo,
                                            %opencl.image3d_t addrspace(1)* %rw) #0
    !kernel_arg_addr_space !60 !kernel_arg_access_qual !61 !kernel_arg_type !62
    !kernel_arg_base_type !62 !kernel_arg_type_qual !25 {
  ret void
}

; CHECK:        - .args:
; CHECK-NEXT:       - .name:           a
; CHECK-NEXT:         .offset:         0
; CHECK-NEXT:         .size:           4
; CHECK-NEXT:         .type_name:      int
; CHECK-NEXT:         .value_kind:     by_value
; CHECK-NEXT:       - .offset:         8
; CHECK-NEXT:         .size:           8
; CHECK-NEXT:         .value_kind:     hidden_global_offset_x
; CHECK-NEXT:       - .offset:         16
; CHECK-NEXT:         .size:           8
; CHECK-NEXT:         .value_kind:     hidden_global_offset_y
; CHECK-NEXT:       - .offset:         24
; CHECK-NEXT:         .size:           8
; CHECK-NEXT:         .value_kind:     hidden_global_offset_z
; CHECK-NEXT:       - .address_space:  global
; CHECK-NEXT:         .offset:         32
; CHECK-NEXT:         .size:           8
; CHECK-NEXT:         .value_kind:     hidden_printf_buffer
; CHECK-NEXT:       - .address_space:  global
; CHECK-NEXT:         .offset:         40
; CHECK-NEXT:         .size:           8
; CHECK-NEXT:         .value_kind:     hidden_none
; CHECK-NEXT:       - .address_space:  global
; CHECK-NEXT:         .offset:         48
; CHECK-NEXT:         .size:           8
; CHECK-NEXT:         .value_kind:     hidden_none
; CHECK-NEXT:       - .address_space:  global
; CHECK-NEXT:         .offset:         56
; CHECK-NEXT:         .size:           8
; CHECK-NEXT:         .value_kind:     hidden_multigrid_sync_arg
; CHECK:          .language:       OpenCL C
; CHECK-NEXT:     .language_version:
; CHECK-NEXT:       - 2
; CHECK-NEXT:       - 0
; CHECK:          .name:           test_vec_type_hint_half
; CHECK:          .symbol:         test_vec_type_hint_half.kd
; CHECK:          .vec_type_hint:  half
define amdgpu_kernel void @test_vec_type_hint_half(i32 %a) #0
    !kernel_arg_addr_space !1 !kernel_arg_access_qual !2 !kernel_arg_type !3
    !kernel_arg_base_type !3 !kernel_arg_type_qual !4 !vec_type_hint !26 {
  ret void
}

; CHECK:        - .args:
; CHECK-NEXT:       - .name:           a
; CHECK-NEXT:         .offset:         0
; CHECK-NEXT:         .size:           4
; CHECK-NEXT:         .type_name:      int
; CHECK-NEXT:         .value_kind:     by_value
; CHECK-NEXT:       - .offset:         8
; CHECK-NEXT:         .size:           8
; CHECK-NEXT:         .value_kind:     hidden_global_offset_x
; CHECK-NEXT:       - .offset:         16
; CHECK-NEXT:         .size:           8
; CHECK-NEXT:         .value_kind:     hidden_global_offset_y
; CHECK-NEXT:       - .offset:         24
; CHECK-NEXT:         .size:           8
; CHECK-NEXT:         .value_kind:     hidden_global_offset_z
; CHECK-NEXT:       - .address_space:  global
; CHECK-NEXT:         .offset:         32
; CHECK-NEXT:         .size:           8
; CHECK-NEXT:         .value_kind:     hidden_printf_buffer
; CHECK-NEXT:       - .address_space:  global
; CHECK-NEXT:         .offset:         40
; CHECK-NEXT:         .size:           8
; CHECK-NEXT:         .value_kind:     hidden_none
; CHECK-NEXT:       - .address_space:  global
; CHECK-NEXT:         .offset:         48
; CHECK-NEXT:         .size:           8
; CHECK-NEXT:         .value_kind:     hidden_none
; CHECK-NEXT:       - .address_space:  global
; CHECK-NEXT:         .offset:         56
; CHECK-NEXT:         .size:           8
; CHECK-NEXT:         .value_kind:     hidden_multigrid_sync_arg
; CHECK:          .language:       OpenCL C
; CHECK-NEXT:     .language_version:
; CHECK-NEXT:       - 2
; CHECK-NEXT:       - 0
; CHECK:          .name:           test_vec_type_hint_float
; CHECK:          .symbol:         test_vec_type_hint_float.kd
; CHECK:          .vec_type_hint:  float
define amdgpu_kernel void @test_vec_type_hint_float(i32 %a) #0
    !kernel_arg_addr_space !1 !kernel_arg_access_qual !2 !kernel_arg_type !3
    !kernel_arg_base_type !3 !kernel_arg_type_qual !4 !vec_type_hint !27 {
  ret void
}

; CHECK:        - .args:
; CHECK-NEXT:       - .name:           a
; CHECK-NEXT:         .offset:         0
; CHECK-NEXT:         .size:           4
; CHECK-NEXT:         .type_name:      int
; CHECK-NEXT:         .value_kind:     by_value
; CHECK-NEXT:       - .offset:         8
; CHECK-NEXT:         .size:           8
; CHECK-NEXT:         .value_kind:     hidden_global_offset_x
; CHECK-NEXT:       - .offset:         16
; CHECK-NEXT:         .size:           8
; CHECK-NEXT:         .value_kind:     hidden_global_offset_y
; CHECK-NEXT:       - .offset:         24
; CHECK-NEXT:         .size:           8
; CHECK-NEXT:         .value_kind:     hidden_global_offset_z
; CHECK-NEXT:       - .address_space:  global
; CHECK-NEXT:         .offset:         32
; CHECK-NEXT:         .size:           8
; CHECK-NEXT:         .value_kind:     hidden_printf_buffer
; CHECK-NEXT:       - .address_space:  global
; CHECK-NEXT:         .offset:         40
; CHECK-NEXT:         .size:           8
; CHECK-NEXT:         .value_kind:     hidden_none
; CHECK-NEXT:       - .address_space:  global
; CHECK-NEXT:         .offset:         48
; CHECK-NEXT:         .size:           8
; CHECK-NEXT:         .value_kind:     hidden_none
; CHECK-NEXT:       - .address_space:  global
; CHECK-NEXT:         .offset:         56
; CHECK-NEXT:         .size:           8
; CHECK-NEXT:         .value_kind:     hidden_multigrid_sync_arg
; CHECK:          .language:       OpenCL C
; CHECK-NEXT:     .language_version:
; CHECK-NEXT:       - 2
; CHECK-NEXT:       - 0
; CHECK:          .name:           test_vec_type_hint_double
; CHECK:          .symbol:         test_vec_type_hint_double.kd
; CHECK:          .vec_type_hint:  double
define amdgpu_kernel void @test_vec_type_hint_double(i32 %a) #0
    !kernel_arg_addr_space !1 !kernel_arg_access_qual !2 !kernel_arg_type !3
    !kernel_arg_base_type !3 !kernel_arg_type_qual !4 !vec_type_hint !28 {
  ret void
}

; CHECK:        - .args:
; CHECK-NEXT:       - .name:           a
; CHECK-NEXT:         .offset:         0
; CHECK-NEXT:         .size:           4
; CHECK-NEXT:         .type_name:      int
; CHECK-NEXT:         .value_kind:     by_value
; CHECK-NEXT:       - .offset:         8
; CHECK-NEXT:         .size:           8
; CHECK-NEXT:         .value_kind:     hidden_global_offset_x
; CHECK-NEXT:       - .offset:         16
; CHECK-NEXT:         .size:           8
; CHECK-NEXT:         .value_kind:     hidden_global_offset_y
; CHECK-NEXT:       - .offset:         24
; CHECK-NEXT:         .size:           8
; CHECK-NEXT:         .value_kind:     hidden_global_offset_z
; CHECK-NEXT:       - .address_space:  global
; CHECK-NEXT:         .offset:         32
; CHECK-NEXT:         .size:           8
; CHECK-NEXT:         .value_kind:     hidden_printf_buffer
; CHECK-NEXT:       - .address_space:  global
; CHECK-NEXT:         .offset:         40
; CHECK-NEXT:         .size:           8
; CHECK-NEXT:         .value_kind:     hidden_none
; CHECK-NEXT:       - .address_space:  global
; CHECK-NEXT:         .offset:         48
; CHECK-NEXT:         .size:           8
; CHECK-NEXT:         .value_kind:     hidden_none
; CHECK-NEXT:       - .address_space:  global
; CHECK-NEXT:         .offset:         56
; CHECK-NEXT:         .size:           8
; CHECK-NEXT:         .value_kind:     hidden_multigrid_sync_arg
; CHECK:          .language:       OpenCL C
; CHECK-NEXT:     .language_version:
; CHECK-NEXT:       - 2
; CHECK-NEXT:       - 0
; CHECK:          .name:           test_vec_type_hint_char
; CHECK:          .symbol:         test_vec_type_hint_char.kd
; CHECK:          .vec_type_hint:  char
define amdgpu_kernel void @test_vec_type_hint_char(i32 %a) #0
    !kernel_arg_addr_space !1 !kernel_arg_access_qual !2 !kernel_arg_type !3
    !kernel_arg_base_type !3 !kernel_arg_type_qual !4 !vec_type_hint !29 {
  ret void
}

; CHECK:        - .args:
; CHECK-NEXT:       - .name:           a
; CHECK-NEXT:         .offset:         0
; CHECK-NEXT:         .size:           4
; CHECK-NEXT:         .type_name:      int
; CHECK-NEXT:         .value_kind:     by_value
; CHECK-NEXT:       - .offset:         8
; CHECK-NEXT:         .size:           8
; CHECK-NEXT:         .value_kind:     hidden_global_offset_x
; CHECK-NEXT:       - .offset:         16
; CHECK-NEXT:         .size:           8
; CHECK-NEXT:         .value_kind:     hidden_global_offset_y
; CHECK-NEXT:       - .offset:         24
; CHECK-NEXT:         .size:           8
; CHECK-NEXT:         .value_kind:     hidden_global_offset_z
; CHECK-NEXT:       - .address_space:  global
; CHECK-NEXT:         .offset:         32
; CHECK-NEXT:         .size:           8
; CHECK-NEXT:         .value_kind:     hidden_printf_buffer
; CHECK-NEXT:       - .address_space:  global
; CHECK-NEXT:         .offset:         40
; CHECK-NEXT:         .size:           8
; CHECK-NEXT:         .value_kind:     hidden_none
; CHECK-NEXT:       - .address_space:  global
; CHECK-NEXT:         .offset:         48
; CHECK-NEXT:         .size:           8
; CHECK-NEXT:         .value_kind:     hidden_none
; CHECK-NEXT:       - .address_space:  global
; CHECK-NEXT:         .offset:         56
; CHECK-NEXT:         .size:           8
; CHECK-NEXT:         .value_kind:     hidden_multigrid_sync_arg
; CHECK:          .language:       OpenCL C
; CHECK-NEXT:     .language_version:
; CHECK-NEXT:       - 2
; CHECK-NEXT:       - 0
; CHECK:          .name:           test_vec_type_hint_short
; CHECK:          .symbol:         test_vec_type_hint_short.kd
; CHECK:          .vec_type_hint:  short
define amdgpu_kernel void @test_vec_type_hint_short(i32 %a) #0
    !kernel_arg_addr_space !1 !kernel_arg_access_qual !2 !kernel_arg_type !3
    !kernel_arg_base_type !3 !kernel_arg_type_qual !4 !vec_type_hint !30 {
  ret void
}

; CHECK:        - .args:
; CHECK-NEXT:       - .name:           a
; CHECK-NEXT:         .offset:         0
; CHECK-NEXT:         .size:           4
; CHECK-NEXT:         .type_name:      int
; CHECK-NEXT:         .value_kind:     by_value
; CHECK-NEXT:       - .offset:         8
; CHECK-NEXT:         .size:           8
; CHECK-NEXT:         .value_kind:     hidden_global_offset_x
; CHECK-NEXT:       - .offset:         16
; CHECK-NEXT:         .size:           8
; CHECK-NEXT:         .value_kind:     hidden_global_offset_y
; CHECK-NEXT:       - .offset:         24
; CHECK-NEXT:         .size:           8
; CHECK-NEXT:         .value_kind:     hidden_global_offset_z
; CHECK-NEXT:       - .address_space:  global
; CHECK-NEXT:         .offset:         32
; CHECK-NEXT:         .size:           8
; CHECK-NEXT:         .value_kind:     hidden_printf_buffer
; CHECK-NEXT:       - .address_space:  global
; CHECK-NEXT:         .offset:         40
; CHECK-NEXT:         .size:           8
; CHECK-NEXT:         .value_kind:     hidden_none
; CHECK-NEXT:       - .address_space:  global
; CHECK-NEXT:         .offset:         48
; CHECK-NEXT:         .size:           8
; CHECK-NEXT:         .value_kind:     hidden_none
; CHECK-NEXT:       - .address_space:  global
; CHECK-NEXT:         .offset:         56
; CHECK-NEXT:         .size:           8
; CHECK-NEXT:         .value_kind:     hidden_multigrid_sync_arg
; CHECK:          .language:       OpenCL C
; CHECK-NEXT:     .language_version:
; CHECK-NEXT:       - 2
; CHECK-NEXT:       - 0
; CHECK:          .name:           test_vec_type_hint_long
; CHECK:          .symbol:         test_vec_type_hint_long.kd
; CHECK:          .vec_type_hint:  long
define amdgpu_kernel void @test_vec_type_hint_long(i32 %a) #0
    !kernel_arg_addr_space !1 !kernel_arg_access_qual !2 !kernel_arg_type !3
    !kernel_arg_base_type !3 !kernel_arg_type_qual !4 !vec_type_hint !31 {
  ret void
}

; CHECK:        - .args:
; CHECK-NEXT:       - .name:           a
; CHECK-NEXT:         .offset:         0
; CHECK-NEXT:         .size:           4
; CHECK-NEXT:         .type_name:      int
; CHECK-NEXT:         .value_kind:     by_value
; CHECK-NEXT:       - .offset:         8
; CHECK-NEXT:         .size:           8
; CHECK-NEXT:         .value_kind:     hidden_global_offset_x
; CHECK-NEXT:       - .offset:         16
; CHECK-NEXT:         .size:           8
; CHECK-NEXT:         .value_kind:     hidden_global_offset_y
; CHECK-NEXT:       - .offset:         24
; CHECK-NEXT:         .size:           8
; CHECK-NEXT:         .value_kind:     hidden_global_offset_z
; CHECK-NEXT:       - .address_space:  global
; CHECK-NEXT:         .offset:         32
; CHECK-NEXT:         .size:           8
; CHECK-NEXT:         .value_kind:     hidden_printf_buffer
; CHECK-NEXT:       - .address_space:  global
; CHECK-NEXT:         .offset:         40
; CHECK-NEXT:         .size:           8
; CHECK-NEXT:         .value_kind:     hidden_none
; CHECK-NEXT:       - .address_space:  global
; CHECK-NEXT:         .offset:         48
; CHECK-NEXT:         .size:           8
; CHECK-NEXT:         .value_kind:     hidden_none
; CHECK-NEXT:       - .address_space:  global
; CHECK-NEXT:         .offset:         56
; CHECK-NEXT:         .size:           8
; CHECK-NEXT:         .value_kind:     hidden_multigrid_sync_arg
; CHECK:          .language:       OpenCL C
; CHECK-NEXT:     .language_version:
; CHECK-NEXT:       - 2
; CHECK-NEXT:       - 0
; CHECK:          .name:           test_vec_type_hint_unknown
; CHECK:          .symbol:         test_vec_type_hint_unknown.kd
; CHECK:          .vec_type_hint:  unknown
define amdgpu_kernel void @test_vec_type_hint_unknown(i32 %a) #0
    !kernel_arg_addr_space !1 !kernel_arg_access_qual !2 !kernel_arg_type !3
    !kernel_arg_base_type !3 !kernel_arg_type_qual !4 !vec_type_hint !32 {
  ret void
}

; CHECK:        - .args:
; CHECK-NEXT:       - .name:           a
; CHECK-NEXT:         .offset:         0
; CHECK-NEXT:         .size:           4
; CHECK-NEXT:         .type_name:      int
; CHECK-NEXT:         .value_kind:     by_value
; CHECK-NEXT:       - .offset:         8
; CHECK-NEXT:         .size:           8
; CHECK-NEXT:         .value_kind:     hidden_global_offset_x
; CHECK-NEXT:       - .offset:         16
; CHECK-NEXT:         .size:           8
; CHECK-NEXT:         .value_kind:     hidden_global_offset_y
; CHECK-NEXT:       - .offset:         24
; CHECK-NEXT:         .size:           8
; CHECK-NEXT:         .value_kind:     hidden_global_offset_z
; CHECK-NEXT:       - .address_space:  global
; CHECK-NEXT:         .offset:         32
; CHECK-NEXT:         .size:           8
; CHECK-NEXT:         .value_kind:     hidden_printf_buffer
; CHECK-NEXT:       - .address_space:  global
; CHECK-NEXT:         .offset:         40
; CHECK-NEXT:         .size:           8
; CHECK-NEXT:         .value_kind:     hidden_none
; CHECK-NEXT:       - .address_space:  global
; CHECK-NEXT:         .offset:         48
; CHECK-NEXT:         .size:           8
; CHECK-NEXT:         .value_kind:     hidden_none
; CHECK-NEXT:       - .address_space:  global
; CHECK-NEXT:         .offset:         56
; CHECK-NEXT:         .size:           8
; CHECK-NEXT:         .value_kind:     hidden_multigrid_sync_arg
; CHECK:          .language:       OpenCL C
; CHECK-NEXT:     .language_version:
; CHECK-NEXT:       - 2
; CHECK-NEXT:       - 0
; CHECK:          .name:           test_reqd_wgs_vec_type_hint
; CHECK:          .reqd_workgroup_size:
; CHECK-NEXT:       - 1
; CHECK-NEXT:       - 2
; CHECK-NEXT:       - 4
; CHECK:          .symbol:         test_reqd_wgs_vec_type_hint.kd
; CHECK:          .vec_type_hint:  int
define amdgpu_kernel void @test_reqd_wgs_vec_type_hint(i32 %a) #0
    !kernel_arg_addr_space !1 !kernel_arg_access_qual !2 !kernel_arg_type !3
    !kernel_arg_base_type !3 !kernel_arg_type_qual !4 !vec_type_hint !5
    !reqd_work_group_size !6 {
  ret void
}

; CHECK:        - .args:
; CHECK-NEXT:       - .name:           a
; CHECK-NEXT:         .offset:         0
; CHECK-NEXT:         .size:           4
; CHECK-NEXT:         .type_name:      int
; CHECK-NEXT:         .value_kind:     by_value
; CHECK-NEXT:       - .offset:         8
; CHECK-NEXT:         .size:           8
; CHECK-NEXT:         .value_kind:     hidden_global_offset_x
; CHECK-NEXT:       - .offset:         16
; CHECK-NEXT:         .size:           8
; CHECK-NEXT:         .value_kind:     hidden_global_offset_y
; CHECK-NEXT:       - .offset:         24
; CHECK-NEXT:         .size:           8
; CHECK-NEXT:         .value_kind:     hidden_global_offset_z
; CHECK-NEXT:       - .address_space:  global
; CHECK-NEXT:         .offset:         32
; CHECK-NEXT:         .size:           8
; CHECK-NEXT:         .value_kind:     hidden_printf_buffer
; CHECK-NEXT:       - .address_space:  global
; CHECK-NEXT:         .offset:         40
; CHECK-NEXT:         .size:           8
; CHECK-NEXT:         .value_kind:     hidden_none
; CHECK-NEXT:       - .address_space:  global
; CHECK-NEXT:         .offset:         48
; CHECK-NEXT:         .size:           8
; CHECK-NEXT:         .value_kind:     hidden_none
; CHECK-NEXT:       - .address_space:  global
; CHECK-NEXT:         .offset:         56
; CHECK-NEXT:         .size:           8
; CHECK-NEXT:         .value_kind:     hidden_multigrid_sync_arg
; CHECK:          .language:       OpenCL C
; CHECK-NEXT:     .language_version:
; CHECK-NEXT:       - 2
; CHECK-NEXT:       - 0
; CHECK:          .name:           test_wgs_hint_vec_type_hint
; CHECK:          .symbol:         test_wgs_hint_vec_type_hint.kd
; CHECK:          .vec_type_hint:  uint4
; CHECK:          .workgroup_size_hint:
; CHECK-NEXT:       - 8
; CHECK-NEXT:       - 16
; CHECK-NEXT:       - 32
define amdgpu_kernel void @test_wgs_hint_vec_type_hint(i32 %a) #0
    !kernel_arg_addr_space !1 !kernel_arg_access_qual !2 !kernel_arg_type !3
    !kernel_arg_base_type !3 !kernel_arg_type_qual !4 !vec_type_hint !7
    !work_group_size_hint !8 {
  ret void
}

; CHECK:        - .args:
; CHECK-NEXT:       - .address_space:  global
; CHECK-NEXT:         .name:           a
; CHECK-NEXT:         .offset:         0
; CHECK-NEXT:         .size:           8
; CHECK-NEXT:         .type_name:      'int  addrspace(5)* addrspace(5)*'
; CHECK-NEXT:         .value_kind:     global_buffer
; CHECK-NEXT:       - .offset:         8
; CHECK-NEXT:         .size:           8
; CHECK-NEXT:         .value_kind:     hidden_global_offset_x
; CHECK-NEXT:       - .offset:         16
; CHECK-NEXT:         .size:           8
; CHECK-NEXT:         .value_kind:     hidden_global_offset_y
; CHECK-NEXT:       - .offset:         24
; CHECK-NEXT:         .size:           8
; CHECK-NEXT:         .value_kind:     hidden_global_offset_z
; CHECK-NEXT:       - .address_space:  global
; CHECK-NEXT:         .offset:         32
; CHECK-NEXT:         .size:           8
; CHECK-NEXT:         .value_kind:     hidden_printf_buffer
; CHECK-NEXT:       - .address_space:  global
; CHECK-NEXT:         .offset:         40
; CHECK-NEXT:         .size:           8
; CHECK-NEXT:         .value_kind:     hidden_none
; CHECK-NEXT:       - .address_space:  global
; CHECK-NEXT:         .offset:         48
; CHECK-NEXT:         .size:           8
; CHECK-NEXT:         .value_kind:     hidden_none
; CHECK-NEXT:       - .address_space:  global
; CHECK-NEXT:         .offset:         56
; CHECK-NEXT:         .size:           8
; CHECK-NEXT:         .value_kind:     hidden_multigrid_sync_arg
; CHECK:          .language:       OpenCL C
; CHECK-NEXT:     .language_version:
; CHECK-NEXT:       - 2
; CHECK-NEXT:       - 0
; CHECK:          .name:           test_arg_ptr_to_ptr
; CHECK:          .symbol:         test_arg_ptr_to_ptr.kd
define amdgpu_kernel void @test_arg_ptr_to_ptr(i32 addrspace(5)* addrspace(1)* %a) #0
    !kernel_arg_addr_space !81 !kernel_arg_access_qual !2 !kernel_arg_type !80
    !kernel_arg_base_type !80 !kernel_arg_type_qual !4 {
  ret void
}

; CHECK:        - .args:
; CHECK-NEXT:         .name:           a
; CHECK-NEXT:         .offset:         0
; CHECK-NEXT:         .size:           8
; CHECK-NEXT:         .type_name:      struct B
; CHECK-NEXT:         .value_kind:     by_value
; CHECK-NEXT:       - .offset:         8
; CHECK-NEXT:         .size:           8
; CHECK-NEXT:         .value_kind:     hidden_global_offset_x
; CHECK-NEXT:       - .offset:         16
; CHECK-NEXT:         .size:           8
; CHECK-NEXT:         .value_kind:     hidden_global_offset_y
; CHECK-NEXT:       - .offset:         24
; CHECK-NEXT:         .size:           8
; CHECK-NEXT:         .value_kind:     hidden_global_offset_z
; CHECK-NEXT:       - .address_space:  global
; CHECK-NEXT:         .offset:         32
; CHECK-NEXT:         .size:           8
; CHECK-NEXT:         .value_kind:     hidden_printf_buffer
; CHECK-NEXT:       - .address_space:  global
; CHECK-NEXT:         .offset:         40
; CHECK-NEXT:         .size:           8
; CHECK-NEXT:         .value_kind:     hidden_none
; CHECK-NEXT:       - .address_space:  global
; CHECK-NEXT:         .offset:         48
; CHECK-NEXT:         .size:           8
; CHECK-NEXT:         .value_kind:     hidden_none
; CHECK-NEXT:       - .address_space:  global
; CHECK-NEXT:         .offset:         56
; CHECK-NEXT:         .size:           8
; CHECK-NEXT:         .value_kind:     hidden_multigrid_sync_arg
; CHECK:          .language:       OpenCL C
; CHECK-NEXT:     .language_version:
; CHECK-NEXT:       - 2
; CHECK-NEXT:       - 0
; CHECK:          .name:           test_arg_struct_contains_ptr
; CHECK:          .symbol:         test_arg_struct_contains_ptr.kd
define amdgpu_kernel void @test_arg_struct_contains_ptr(%struct.B %a) #0
    !kernel_arg_addr_space !1 !kernel_arg_access_qual !2 !kernel_arg_type !82
    !kernel_arg_base_type !82 !kernel_arg_type_qual !4 {
 ret void
}

; CHECK:        - .args:
; CHECK-NEXT:       - .name:           a
; CHECK-NEXT:         .offset:         0
; CHECK-NEXT:         .size:           16
; CHECK-NEXT:         .type_name:      'global int addrspace(5)* __attribute__((ext_vector_type(2)))'
; CHECK-NEXT:         .value_kind:     by_value
; CHECK-NEXT:       - .offset:         16
; CHECK-NEXT:         .size:           8
; CHECK-NEXT:         .value_kind:     hidden_global_offset_x
; CHECK-NEXT:       - .offset:         24
; CHECK-NEXT:         .size:           8
; CHECK-NEXT:         .value_kind:     hidden_global_offset_y
; CHECK-NEXT:       - .offset:         32
; CHECK-NEXT:         .size:           8
; CHECK-NEXT:         .value_kind:     hidden_global_offset_z
; CHECK-NEXT:       - .address_space:  global
; CHECK-NEXT:         .offset:         40
; CHECK-NEXT:         .size:           8
; CHECK-NEXT:         .value_kind:     hidden_printf_buffer
; CHECK-NEXT:       - .address_space:  global
; CHECK-NEXT:         .offset:         48
; CHECK-NEXT:         .size:           8
; CHECK-NEXT:         .value_kind:     hidden_none
; CHECK-NEXT:       - .address_space:  global
; CHECK-NEXT:         .offset:         56
; CHECK-NEXT:         .size:           8
; CHECK-NEXT:         .value_kind:     hidden_none
; CHECK-NEXT:       - .address_space:  global
; CHECK-NEXT:         .offset:         64
; CHECK-NEXT:         .size:           8
; CHECK-NEXT:         .value_kind:     hidden_multigrid_sync_arg
; CHECK:          .language:       OpenCL C
; CHECK-NEXT:     .language_version:
; CHECK-NEXT:       - 2
; CHECK-NEXT:       - 0
; CHECK:          .name:           test_arg_vector_of_ptr
; CHECK:          .symbol:         test_arg_vector_of_ptr.kd
define amdgpu_kernel void @test_arg_vector_of_ptr(<2 x i32 addrspace(1)*> %a) #0
    !kernel_arg_addr_space !1 !kernel_arg_access_qual !2 !kernel_arg_type !83
    !kernel_arg_base_type !83 !kernel_arg_type_qual !4 {
  ret void
}

; CHECK:        - .args:
; CHECK-NEXT:       - .address_space:  global
; CHECK-NEXT:         .name:           a
; CHECK-NEXT:         .offset:         0
; CHECK-NEXT:         .size:           8
; CHECK-NEXT:         .type_name:      clk_event_t
; CHECK-NEXT:         .value_kind:     global_buffer
; CHECK-NEXT:       - .offset:         8
; CHECK-NEXT:         .size:           8
; CHECK-NEXT:         .value_kind:     hidden_global_offset_x
; CHECK-NEXT:       - .offset:         16
; CHECK-NEXT:         .size:           8
; CHECK-NEXT:         .value_kind:     hidden_global_offset_y
; CHECK-NEXT:       - .offset:         24
; CHECK-NEXT:         .size:           8
; CHECK-NEXT:         .value_kind:     hidden_global_offset_z
; CHECK-NEXT:       - .address_space:  global
; CHECK-NEXT:         .offset:         32
; CHECK-NEXT:         .size:           8
; CHECK-NEXT:         .value_kind:     hidden_printf_buffer
; CHECK-NEXT:       - .address_space:  global
; CHECK-NEXT:         .offset:         40
; CHECK-NEXT:         .size:           8
; CHECK-NEXT:         .value_kind:     hidden_none
; CHECK-NEXT:       - .address_space:  global
; CHECK-NEXT:         .offset:         48
; CHECK-NEXT:         .size:           8
; CHECK-NEXT:         .value_kind:     hidden_none
; CHECK-NEXT:       - .address_space:  global
; CHECK-NEXT:         .offset:         56
; CHECK-NEXT:         .size:           8
; CHECK-NEXT:         .value_kind:     hidden_multigrid_sync_arg
; CHECK:          .language:       OpenCL C
; CHECK-NEXT:     .language_version:
; CHECK-NEXT:       - 2
; CHECK-NEXT:       - 0
; CHECK:          .name:           test_arg_unknown_builtin_type
; CHECK:          .symbol:         test_arg_unknown_builtin_type.kd
define amdgpu_kernel void @test_arg_unknown_builtin_type(
    %opencl.clk_event_t addrspace(1)* %a) #0
    !kernel_arg_addr_space !81 !kernel_arg_access_qual !2 !kernel_arg_type !84
    !kernel_arg_base_type !84 !kernel_arg_type_qual !4 {
  ret void
}

; CHECK:        - .args:
; CHECK-NEXT:       - .address_space:  global
; CHECK-NEXT:         .name:           a
; CHECK-NEXT:         .offset:         0
; CHECK-NEXT:         .size:           8
; CHECK-NEXT:         .type_name:      'long  addrspace(5)*'
; CHECK-NEXT:         .value_kind:     global_buffer
; CHECK-NEXT:       - .address_space:  local
; CHECK-NEXT:         .name:           b
; CHECK-NEXT:         .offset:         8
; CHECK-NEXT:         .pointee_align:  1
; CHECK-NEXT:         .size:           4
; CHECK-NEXT:         .type_name:      'char  addrspace(5)*'
; CHECK-NEXT:         .value_kind:     dynamic_shared_pointer
; CHECK-NEXT:       - .address_space:  local
; CHECK-NEXT:         .name:           c
; CHECK-NEXT:         .offset:         12
; CHECK-NEXT:         .pointee_align:  2
; CHECK-NEXT:         .size:           4
; CHECK-NEXT:         .type_name:      'char2  addrspace(5)*'
; CHECK-NEXT:         .value_kind:     dynamic_shared_pointer
; CHECK-NEXT:       - .address_space:  local
; CHECK-NEXT:         .name:           d
; CHECK-NEXT:         .offset:         16
; CHECK-NEXT:         .pointee_align:  4
; CHECK-NEXT:         .size:           4
; CHECK-NEXT:         .type_name:      'char3  addrspace(5)*'
; CHECK-NEXT:         .value_kind:     dynamic_shared_pointer
; CHECK-NEXT:       - .address_space:  local
; CHECK-NEXT:         .name:           e
; CHECK-NEXT:         .offset:         20
; CHECK-NEXT:         .pointee_align:  4
; CHECK-NEXT:         .size:           4
; CHECK-NEXT:         .type_name:      'char4  addrspace(5)*'
; CHECK-NEXT:         .value_kind:     dynamic_shared_pointer
; CHECK-NEXT:       - .address_space:  local
; CHECK-NEXT:         .name:           f
; CHECK-NEXT:         .offset:         24
; CHECK-NEXT:         .pointee_align:  8
; CHECK-NEXT:         .size:           4
; CHECK-NEXT:         .type_name:      'char8  addrspace(5)*'
; CHECK-NEXT:         .value_kind:     dynamic_shared_pointer
; CHECK-NEXT:       - .address_space:  local
; CHECK-NEXT:         .name:           g
; CHECK-NEXT:         .offset:         28
; CHECK-NEXT:         .pointee_align:  16
; CHECK-NEXT:         .size:           4
; CHECK-NEXT:         .type_name:      'char16  addrspace(5)*'
; CHECK-NEXT:         .value_kind:     dynamic_shared_pointer
; CHECK-NEXT:       - .address_space:  local
; CHECK-NEXT:         .name:           h
; CHECK-NEXT:         .offset:         32
; CHECK-NEXT:         .pointee_align:  1
; CHECK-NEXT:         .size:           4
; CHECK-NEXT:         .value_kind:     dynamic_shared_pointer
; CHECK-NEXT:       - .offset:         40
; CHECK-NEXT:         .size:           8
; CHECK-NEXT:         .value_kind:     hidden_global_offset_x
; CHECK-NEXT:       - .offset:         48
; CHECK-NEXT:         .size:           8
; CHECK-NEXT:         .value_kind:     hidden_global_offset_y
; CHECK-NEXT:       - .offset:         56
; CHECK-NEXT:         .size:           8
; CHECK-NEXT:         .value_kind:     hidden_global_offset_z
; CHECK-NEXT:       - .address_space:  global
; CHECK-NEXT:         .offset:         64
; CHECK-NEXT:         .size:           8
; CHECK-NEXT:         .value_kind:     hidden_printf_buffer
; CHECK-NEXT:       - .address_space:  global
; CHECK-NEXT:         .offset:         72
; CHECK-NEXT:         .size:           8
; CHECK-NEXT:         .value_kind:     hidden_none
; CHECK-NEXT:       - .address_space:  global
; CHECK-NEXT:         .offset:         80
; CHECK-NEXT:         .size:           8
; CHECK-NEXT:         .value_kind:     hidden_none
; CHECK-NEXT:       - .address_space:  global
; CHECK-NEXT:         .offset:         88
; CHECK-NEXT:         .size:           8
; CHECK-NEXT:         .value_kind:     hidden_multigrid_sync_arg
; CHECK:          .language:       OpenCL C
; CHECK-NEXT:     .language_version:
; CHECK-NEXT:       - 2
; CHECK-NEXT:       - 0
; CHECK:          .name:           test_pointee_align
; CHECK:          .symbol:         test_pointee_align.kd
define amdgpu_kernel void @test_pointee_align(i64 addrspace(1)* %a,
                                              i8 addrspace(3)* %b,
                                              <2 x i8> addrspace(3)* align 2 %c,
                                              <3 x i8> addrspace(3)* align 4 %d,
                                              <4 x i8> addrspace(3)* align 4 %e,
                                              <8 x i8> addrspace(3)* align 8 %f,
                                              <16 x i8> addrspace(3)* align 16 %g,
                                              {} addrspace(3)* %h) #0
    !kernel_arg_addr_space !91 !kernel_arg_access_qual !92 !kernel_arg_type !93
    !kernel_arg_base_type !93 !kernel_arg_type_qual !94 {
  ret void
}

; CHECK:        - .args:
; CHECK-NEXT:       - .address_space:  global
; CHECK-NEXT:         .name:           a
; CHECK-NEXT:         .offset:         0
; CHECK-NEXT:         .size:           8
; CHECK-NEXT:         .type_name:      'long  addrspace(5)*'
; CHECK-NEXT:         .value_kind:     global_buffer
; CHECK-NEXT:       - .address_space:  local
; CHECK-NEXT:         .name:           b
; CHECK-NEXT:         .offset:         8
; CHECK-NEXT:         .pointee_align:  8
; CHECK-NEXT:         .size:           4
; CHECK-NEXT:         .type_name:      'char  addrspace(5)*'
; CHECK-NEXT:         .value_kind:     dynamic_shared_pointer
; CHECK-NEXT:       - .address_space:  local
; CHECK-NEXT:         .name:           c
; CHECK-NEXT:         .offset:         12
; CHECK-NEXT:         .pointee_align:  32
; CHECK-NEXT:         .size:           4
; CHECK-NEXT:         .type_name:      'char2  addrspace(5)*'
; CHECK-NEXT:         .value_kind:     dynamic_shared_pointer
; CHECK-NEXT:       - .address_space:  local
; CHECK-NEXT:         .name:           d
; CHECK-NEXT:         .offset:         16
; CHECK-NEXT:         .pointee_align:  64
; CHECK-NEXT:         .size:           4
; CHECK-NEXT:         .type_name:      'char3  addrspace(5)*'
; CHECK-NEXT:         .value_kind:     dynamic_shared_pointer
; CHECK-NEXT:       - .address_space:  local
; CHECK-NEXT:         .name:           e
; CHECK-NEXT:         .offset:         20
; CHECK-NEXT:         .pointee_align:  256
; CHECK-NEXT:         .size:           4
; CHECK-NEXT:         .type_name:      'char4  addrspace(5)*'
; CHECK-NEXT:         .value_kind:     dynamic_shared_pointer
; CHECK-NEXT:       - .address_space:  local
; CHECK-NEXT:         .name:           f
; CHECK-NEXT:         .offset:         24
; CHECK-NEXT:         .pointee_align:  128
; CHECK-NEXT:         .size:           4
; CHECK-NEXT:         .type_name:      'char8  addrspace(5)*'
; CHECK-NEXT:         .value_kind:     dynamic_shared_pointer
; CHECK-NEXT:       - .address_space:  local
; CHECK-NEXT:         .name:           g
; CHECK-NEXT:         .offset:         28
; CHECK-NEXT:         .pointee_align:  1024
; CHECK-NEXT:         .size:           4
; CHECK-NEXT:         .type_name:      'char16  addrspace(5)*'
; CHECK-NEXT:         .value_kind:     dynamic_shared_pointer
; CHECK-NEXT:       - .address_space:  local
; CHECK-NEXT:         .name:           h
; CHECK-NEXT:         .offset:         32
; CHECK-NEXT:         .pointee_align:  16
; CHECK-NEXT:         .size:           4
; CHECK-NEXT:         .value_kind:     dynamic_shared_pointer
; CHECK-NEXT:       - .offset:         40
; CHECK-NEXT:         .size:           8
; CHECK-NEXT:         .value_kind:     hidden_global_offset_x
; CHECK-NEXT:       - .offset:         48
; CHECK-NEXT:         .size:           8
; CHECK-NEXT:         .value_kind:     hidden_global_offset_y
; CHECK-NEXT:       - .offset:         56
; CHECK-NEXT:         .size:           8
; CHECK-NEXT:         .value_kind:     hidden_global_offset_z
; CHECK-NEXT:       - .address_space:  global
; CHECK-NEXT:         .offset:         64
; CHECK-NEXT:         .size:           8
; CHECK-NEXT:         .value_kind:     hidden_printf_buffer
; CHECK-NEXT:       - .address_space:  global
; CHECK-NEXT:         .offset:         72
; CHECK-NEXT:         .size:           8
; CHECK-NEXT:         .value_kind:     hidden_none
; CHECK-NEXT:       - .address_space:  global
; CHECK-NEXT:         .offset:         80
; CHECK-NEXT:         .size:           8
; CHECK-NEXT:         .value_kind:     hidden_none
; CHECK-NEXT:       - .address_space:  global
; CHECK-NEXT:         .offset:         88
; CHECK-NEXT:         .size:           8
; CHECK-NEXT:         .value_kind:     hidden_multigrid_sync_arg
; CHECK:          .language:       OpenCL C
; CHECK-NEXT:     .language_version:
; CHECK-NEXT:       - 2
; CHECK-NEXT:       - 0
; CHECK:          .name:           test_pointee_align_attribute
; CHECK:          .symbol:         test_pointee_align_attribute.kd
define amdgpu_kernel void @test_pointee_align_attribute(i64 addrspace(1)* align 16 %a,
                                                        i8 addrspace(3)* align 8 %b,
                                                        <2 x i8> addrspace(3)* align 32 %c,
                                                        <3 x i8> addrspace(3)* align 64 %d,
                                                        <4 x i8> addrspace(3)* align 256 %e,
                                                        <8 x i8> addrspace(3)* align 128 %f,
                                                        <16 x i8> addrspace(3)* align 1024 %g,
                                                        {} addrspace(3)* align 16 %h) #0
    !kernel_arg_addr_space !91 !kernel_arg_access_qual !92 !kernel_arg_type !93
    !kernel_arg_base_type !93 !kernel_arg_type_qual !94 {
  ret void
}
; CHECK:        - .args:
; CHECK-NEXT:       - .name:           arg
; CHECK-NEXT:         .offset:         0
; CHECK-NEXT:         .size:           25
; CHECK-NEXT:         .type_name:      __block_literal
; CHECK-NEXT:         .value_kind:     by_value
; CHECK-NEXT:       - .offset:         32
; CHECK-NEXT:         .size:           8
; CHECK-NEXT:         .value_kind:     hidden_global_offset_x
; CHECK-NEXT:       - .offset:         40
; CHECK-NEXT:         .size:           8
; CHECK-NEXT:         .value_kind:     hidden_global_offset_y
; CHECK-NEXT:       - .offset:         48
; CHECK-NEXT:         .size:           8
; CHECK-NEXT:         .value_kind:     hidden_global_offset_z
; CHECK-NEXT:       - .address_space:  global
; CHECK-NEXT:         .offset:         56
; CHECK-NEXT:         .size:           8
; CHECK-NEXT:         .value_kind:     hidden_printf_buffer
; CHECK-NEXT:       - .address_space:  global
; CHECK-NEXT:         .offset:         64
; CHECK-NEXT:         .size:           8
; CHECK-NEXT:         .value_kind:     hidden_none
; CHECK-NEXT:       - .address_space:  global
; CHECK-NEXT:         .offset:         72
; CHECK-NEXT:         .size:           8
; CHECK-NEXT:         .value_kind:     hidden_none
; CHECK-NEXT:       - .address_space:  global
; CHECK-NEXT:         .offset:         80
; CHECK-NEXT:         .size:           8
; CHECK-NEXT:         .value_kind:     hidden_multigrid_sync_arg
; CHECK:          .device_enqueue_symbol: __test_block_invoke_kernel_runtime_handle
; CHECK:          .language:       OpenCL C
; CHECK-NEXT:     .language_version:
; CHECK-NEXT:       - 2
; CHECK-NEXT:       - 0
; CHECK:          .name:           __test_block_invoke_kernel
; CHECK:          .symbol:         __test_block_invoke_kernel.kd
define amdgpu_kernel void @__test_block_invoke_kernel(
    <{ i32, i32, i8*, i8 addrspace(1)*, i8 }> %arg) #1
    !kernel_arg_addr_space !1 !kernel_arg_access_qual !2 !kernel_arg_type !110
    !kernel_arg_base_type !110 !kernel_arg_type_qual !4 {
  ret void
}

; CHECK:        - .args:
; CHECK-NEXT:       - .name:           a
; CHECK-NEXT:         .offset:         0
; CHECK-NEXT:         .size:           1
; CHECK-NEXT:         .type_name:      char
; CHECK-NEXT:         .value_kind:     by_value
; CHECK-NEXT:       - .offset:         8
; CHECK-NEXT:         .size:           8
; CHECK-NEXT:         .value_kind:     hidden_global_offset_x
; CHECK-NEXT:       - .offset:         16
; CHECK-NEXT:         .size:           8
; CHECK-NEXT:         .value_kind:     hidden_global_offset_y
; CHECK-NEXT:       - .offset:         24
; CHECK-NEXT:         .size:           8
; CHECK-NEXT:         .value_kind:     hidden_global_offset_z
; CHECK-NEXT:       - .address_space:  global
; CHECK-NEXT:         .offset:         32
; CHECK-NEXT:         .size:           8
; CHECK-NEXT:         .value_kind:     hidden_printf_buffer
; CHECK-NEXT:       - .address_space:  global
; CHECK-NEXT:         .offset:         40
; CHECK-NEXT:         .size:           8
; CHECK-NEXT:         .value_kind:     hidden_default_queue
; CHECK-NEXT:       - .address_space:  global
; CHECK-NEXT:         .offset:         48
; CHECK-NEXT:         .size:           8
; CHECK-NEXT:         .value_kind:     hidden_completion_action
; CHECK-NEXT:       - .address_space:  global
; CHECK-NEXT:         .offset:         56
; CHECK-NEXT:         .size:           8
; CHECK-NEXT:         .value_kind:     hidden_multigrid_sync_arg
; CHECK:          .language:       OpenCL C
; CHECK-NEXT:     .language_version:
; CHECK-NEXT:       - 2
; CHECK-NEXT:       - 0
; CHECK:          .name:           test_enqueue_kernel_caller
; CHECK:          .symbol:         test_enqueue_kernel_caller.kd
define amdgpu_kernel void @test_enqueue_kernel_caller(i8 %a) #2
    !kernel_arg_addr_space !1 !kernel_arg_access_qual !2 !kernel_arg_type !9
    !kernel_arg_base_type !9 !kernel_arg_type_qual !4 {
  ret void
}

; CHECK:        - .args:
; CHECK-NEXT:       - .name:           ptr
; CHECK-NEXT:         .offset:         0
; CHECK-NEXT:         .size:           8
; CHECK-NEXT:         .value_kind:     global_buffer
; CHECK:          .name:           unknown_addrspace_kernarg
; CHECK:          .symbol:         unknown_addrspace_kernarg.kd
define amdgpu_kernel void @unknown_addrspace_kernarg(i32 addrspace(12345)* %ptr) #0 {
  ret void
}

; CHECK:  amdhsa.printf:
; CHECK-NEXT: - '1:1:4:%d\n'
; CHECK-NEXT: - '2:1:8:%g\n'
; CHECK:  amdhsa.version:
; CHECK-NEXT: - 1
; CHECK-NEXT: - 0

attributes #0 = { optnone noinline "amdgpu-implicitarg-num-bytes"="56" }
attributes #1 = { optnone noinline "amdgpu-implicitarg-num-bytes"="56" "runtime-handle"="__test_block_invoke_kernel_runtime_handle" }
attributes #2 = { optnone noinline "amdgpu-implicitarg-num-bytes"="56" "calls-enqueue-kernel" }

!llvm.printf.fmts = !{!100, !101}

!1 = !{i32 0}
!2 = !{!"none"}
!3 = !{!"int"}
!4 = !{!""}
!5 = !{i32 undef, i32 1}
!6 = !{i32 1, i32 2, i32 4}
!7 = !{<4 x i32> undef, i32 0}
!8 = !{i32 8, i32 16, i32 32}
!9 = !{!"char"}
!10 = !{!"ushort2"}
!11 = !{!"int3"}
!12 = !{!"ulong4"}
!13 = !{!"half8"}
!14 = !{!"float16"}
!15 = !{!"double16"}
!16 = !{!"int  addrspace(5)*"}
!17 = !{!"image2d_t"}
!18 = !{!"sampler_t"}
!19 = !{!"queue_t"}
!20 = !{!"struct A"}
!21 = !{!"i128"}
!22 = !{i32 0, i32 0, i32 0}
!23 = !{!"none", !"none", !"none"}
!24 = !{!"int", !"short2", !"char3"}
!25 = !{!"", !"", !""}
!26 = !{half undef, i32 1}
!27 = !{float undef, i32 1}
!28 = !{double undef, i32 1}
!29 = !{i8 undef, i32 1}
!30 = !{i16 undef, i32 1}
!31 = !{i64 undef, i32 1}
!32 = !{i32  addrspace(5)*undef, i32 1}
!50 = !{i32 1, i32 2, i32 3}
!51 = !{!"int  addrspace(5)*", !"int  addrspace(5)*", !"int  addrspace(5)*"}
!60 = !{i32 1, i32 1, i32 1}
!61 = !{!"read_only", !"write_only", !"read_write"}
!62 = !{!"image1d_t", !"image2d_t", !"image3d_t"}
!70 = !{!"volatile", !"const restrict", !"pipe"}
!80 = !{!"int  addrspace(5)* addrspace(5)*"}
!81 = !{i32 1}
!82 = !{!"struct B"}
!83 = !{!"global int addrspace(5)* __attribute__((ext_vector_type(2)))"}
!84 = !{!"clk_event_t"}
!opencl.ocl.version = !{!90}
!90 = !{i32 2, i32 0}
!91 = !{i32 0, i32 3, i32 3, i32 3, i32 3, i32 3, i32 3}
!92 = !{!"none", !"none", !"none", !"none", !"none", !"none", !"none"}
!93 = !{!"long  addrspace(5)*", !"char  addrspace(5)*", !"char2  addrspace(5)*", !"char3  addrspace(5)*", !"char4  addrspace(5)*", !"char8  addrspace(5)*", !"char16  addrspace(5)*"}
!94 = !{!"", !"", !"", !"", !"", !"", !""}
!100 = !{!"1:1:4:%d\5Cn"}
!101 = !{!"2:1:8:%g\5Cn"}
!110 = !{!"__block_literal"}
!111 = !{!"char", !"char"}

; PARSER: AMDGPU HSA Metadata Parser Test: PASS

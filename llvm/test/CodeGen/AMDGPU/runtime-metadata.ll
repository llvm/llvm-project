; RUN: llc -mtriple=amdgcn--amdhsa < %s | FileCheck %s

%struct.A = type { i8, float }
%opencl.image1d_t = type opaque
%opencl.image2d_t = type opaque
%opencl.image3d_t = type opaque
%opencl.queue_t = type opaque
%opencl.pipe_t = type opaque
%struct.B = type { i32 addrspace(1)*}
%opencl.clk_event_t = type opaque

; CHECK: .section        .AMDGPU.runtime_metadata
; CHECK-NEXT: .byte	1
; CHECK-NEXT: .short	256
; CHECK-NEXT: .byte	2
; CHECK-NEXT: .byte	0
; CHECK-NEXT: .byte	3
; CHECK-NEXT: .short	200
; CHECK-NEXT: .byte	30
; CHECK-NEXT: .long	10
; CHECK-NEXT: .ascii	"1:1:4:%d\\n"
; CHECK-NEXT: .byte	30
; CHECK-NEXT: .long	10
; CHECK-NEXT: .ascii	"2:1:8:%g\\n"

; CHECK-LABEL:{{^}}test_char:
; CHECK: .section        .AMDGPU.runtime_metadata
; CHECK-NEXT: .byte	4
; CHECK-NEXT: .byte	6
; CHECK-NEXT: .long	9
; CHECK-NEXT: .ascii	"test_char"
; CHECK-NEXT: .byte	7
; CHECK-NEXT: .byte	9
; CHECK-NEXT: .long	1
; CHECK-NEXT: .byte	10
; CHECK-NEXT: .long	1
; CHECK-NEXT: .byte	11
; CHECK-NEXT: .long	4
; CHECK-NEXT: .ascii	"char"
; CHECK-NEXT: .byte	13
; CHECK-NEXT: .byte	0
; CHECK-NEXT: .byte	14
; CHECK-NEXT: .short	1
; CHECK-NEXT: .byte	16
; CHECK-NEXT: .byte	0
; CHECK-NEXT: .byte	8
; CHECK-NEXT: .byte	7
; CHECK-NEXT: .byte	9
; CHECK-NEXT: .long	8
; CHECK-NEXT: .byte	10
; CHECK-NEXT: .long	8
; CHECK-NEXT: .byte	13
; CHECK-NEXT: .byte	7
; CHECK-NEXT: .byte	14
; CHECK-NEXT: .short	9
; CHECK-NEXT: .byte	8
; CHECK-NEXT: .byte	7
; CHECK-NEXT: .byte	9
; CHECK-NEXT: .long	8
; CHECK-NEXT: .byte	10
; CHECK-NEXT: .long	8
; CHECK-NEXT: .byte	13
; CHECK-NEXT: .byte	8
; CHECK-NEXT: .byte	14
; CHECK-NEXT: .short	9
; CHECK-NEXT: .byte	8
; CHECK-NEXT: .byte	7
; CHECK-NEXT: .byte	9
; CHECK-NEXT: .long	8
; CHECK-NEXT: .byte	10
; CHECK-NEXT: .long	8
; CHECK-NEXT: .byte	13
; CHECK-NEXT: .byte	9
; CHECK-NEXT: .byte	14
; CHECK-NEXT: .short	9
; CHECK-NEXT: .byte	8
; CHECK-NEXT: .byte	7
; CHECK-NEXT: .byte	9
; CHECK-NEXT: .long	8
; CHECK-NEXT: .byte	10
; CHECK-NEXT: .long	8
; CHECK-NEXT: .byte	13
; CHECK-NEXT: .byte	11
; CHECK-NEXT: .byte	14
; CHECK-NEXT: .short	1
; CHECK-NEXT: .byte	15
; CHECK-NEXT: .byte	1
; CHECK-NEXT: .byte	8
; CHECK-NEXT: .byte	5

define amdgpu_kernel void @test_char(i8 %a) !kernel_arg_addr_space !1 !kernel_arg_access_qual !2 !kernel_arg_type !9 !kernel_arg_base_type !9 !kernel_arg_type_qual !4 {
  ret void
}

; CHECK-LABEL:{{^}}test_ushort2:
; CHECK: .section        .AMDGPU.runtime_metadata
; CHECK-NEXT: .byte	4
; CHECK-NEXT: .byte	6
; CHECK-NEXT: .long	12
; CHECK-NEXT: .ascii	"test_ushort2"
; CHECK-NEXT: .byte	7
; CHECK-NEXT: .byte	9
; CHECK-NEXT: .long	4
; CHECK-NEXT: .byte	10
; CHECK-NEXT: .long	4
; CHECK-NEXT: .byte	11
; CHECK-NEXT: .long	7
; CHECK-NEXT: .ascii	"ushort2"
; CHECK-NEXT: .byte	13
; CHECK-NEXT: .byte	0
; CHECK-NEXT: .byte	14
; CHECK-NEXT: .short	4
; CHECK-NEXT: .byte	16
; CHECK-NEXT: .byte	0
; CHECK-NEXT: .byte	8
; CHECK-NEXT: .byte	7
; CHECK-NEXT: .byte	9
; CHECK-NEXT: .long	8
; CHECK-NEXT: .byte	10
; CHECK-NEXT: .long	8
; CHECK-NEXT: .byte	13
; CHECK-NEXT: .byte	7
; CHECK-NEXT: .byte	14
; CHECK-NEXT: .short	9
; CHECK-NEXT: .byte	8
; CHECK-NEXT: .byte	7
; CHECK-NEXT: .byte	9
; CHECK-NEXT: .long	8
; CHECK-NEXT: .byte	10
; CHECK-NEXT: .long	8
; CHECK-NEXT: .byte	13
; CHECK-NEXT: .byte	8
; CHECK-NEXT: .byte	14
; CHECK-NEXT: .short	9
; CHECK-NEXT: .byte	8
; CHECK-NEXT: .byte	7
; CHECK-NEXT: .byte	9
; CHECK-NEXT: .long	8
; CHECK-NEXT: .byte	10
; CHECK-NEXT: .long	8
; CHECK-NEXT: .byte	13
; CHECK-NEXT: .byte	9
; CHECK-NEXT: .byte	14
; CHECK-NEXT: .short	9
; CHECK-NEXT: .byte	8
; CHECK-NEXT: .byte	7
; CHECK-NEXT: .byte	9
; CHECK-NEXT: .long	8
; CHECK-NEXT: .byte	10
; CHECK-NEXT: .long	8
; CHECK-NEXT: .byte	13
; CHECK-NEXT: .byte	11
; CHECK-NEXT: .byte	14
; CHECK-NEXT: .short	1
; CHECK-NEXT: .byte	15
; CHECK-NEXT: .byte	1
; CHECK-NEXT: .byte	8
; CHECK-NEXT: .byte	5

define amdgpu_kernel void @test_ushort2(<2 x i16> %a) !kernel_arg_addr_space !1 !kernel_arg_access_qual !2 !kernel_arg_type !10 !kernel_arg_base_type !10 !kernel_arg_type_qual !4 {
  ret void
}

; CHECK-LABEL:{{^}}test_int3:
; CHECK: .section        .AMDGPU.runtime_metadata
; CHECK-NEXT: .byte	4
; CHECK-NEXT: .byte	6
; CHECK-NEXT: .long	9
; CHECK-NEXT: .ascii	"test_int3"
; CHECK-NEXT: .byte	7
; CHECK-NEXT: .byte	9
; CHECK-NEXT: .long	16
; CHECK-NEXT: .byte	10
; CHECK-NEXT: .long	16
; CHECK-NEXT: .byte	11
; CHECK-NEXT: .long	4
; CHECK-NEXT: .ascii	"int3"
; CHECK-NEXT: .byte	13
; CHECK-NEXT: .byte	0
; CHECK-NEXT: .byte	14
; CHECK-NEXT: .short	6
; CHECK-NEXT: .byte	16
; CHECK-NEXT: .byte	0
; CHECK-NEXT: .byte	8
; CHECK-NEXT: .byte	7
; CHECK-NEXT: .byte	9
; CHECK-NEXT: .long	8
; CHECK-NEXT: .byte	10
; CHECK-NEXT: .long	8
; CHECK-NEXT: .byte	13
; CHECK-NEXT: .byte	7
; CHECK-NEXT: .byte	14
; CHECK-NEXT: .short	9
; CHECK-NEXT: .byte	8
; CHECK-NEXT: .byte	7
; CHECK-NEXT: .byte	9
; CHECK-NEXT: .long	8
; CHECK-NEXT: .byte	10
; CHECK-NEXT: .long	8
; CHECK-NEXT: .byte	13
; CHECK-NEXT: .byte	8
; CHECK-NEXT: .byte	14
; CHECK-NEXT: .short	9
; CHECK-NEXT: .byte	8
; CHECK-NEXT: .byte	7
; CHECK-NEXT: .byte	9
; CHECK-NEXT: .long	8
; CHECK-NEXT: .byte	10
; CHECK-NEXT: .long	8
; CHECK-NEXT: .byte	13
; CHECK-NEXT: .byte	9
; CHECK-NEXT: .byte	14
; CHECK-NEXT: .short	9
; CHECK-NEXT: .byte	8
; CHECK-NEXT: .byte	7
; CHECK-NEXT: .byte	9
; CHECK-NEXT: .long	8
; CHECK-NEXT: .byte	10
; CHECK-NEXT: .long	8
; CHECK-NEXT: .byte	13
; CHECK-NEXT: .byte	11
; CHECK-NEXT: .byte	14
; CHECK-NEXT: .short	1
; CHECK-NEXT: .byte	15
; CHECK-NEXT: .byte	1
; CHECK-NEXT: .byte	8
; CHECK-NEXT: .byte	5

define amdgpu_kernel void @test_int3(<3 x i32> %a) !kernel_arg_addr_space !1 !kernel_arg_access_qual !2 !kernel_arg_type !11 !kernel_arg_base_type !11 !kernel_arg_type_qual !4 {
  ret void
}

; CHECK-LABEL:{{^}}test_ulong4:
; CHECK: .section        .AMDGPU.runtime_metadata
; CHECK-NEXT: .byte	4
; CHECK-NEXT: .byte	6
; CHECK-NEXT: .long	11
; CHECK-NEXT: .ascii	"test_ulong4"
; CHECK-NEXT: .byte	7
; CHECK-NEXT: .byte	9
; CHECK-NEXT: .long	32
; CHECK-NEXT: .byte	10
; CHECK-NEXT: .long	32
; CHECK-NEXT: .byte	11
; CHECK-NEXT: .long	6
; CHECK-NEXT: .ascii	"ulong4"
; CHECK-NEXT: .byte	13
; CHECK-NEXT: .byte	0
; CHECK-NEXT: .byte	14
; CHECK-NEXT: .short	10
; CHECK-NEXT: .byte	16
; CHECK-NEXT: .byte	0
; CHECK-NEXT: .byte	8
; CHECK-NEXT: .byte	7
; CHECK-NEXT: .byte	9
; CHECK-NEXT: .long	8
; CHECK-NEXT: .byte	10
; CHECK-NEXT: .long	8
; CHECK-NEXT: .byte	13
; CHECK-NEXT: .byte	7
; CHECK-NEXT: .byte	14
; CHECK-NEXT: .short	9
; CHECK-NEXT: .byte	8
; CHECK-NEXT: .byte	7
; CHECK-NEXT: .byte	9
; CHECK-NEXT: .long	8
; CHECK-NEXT: .byte	10
; CHECK-NEXT: .long	8
; CHECK-NEXT: .byte	13
; CHECK-NEXT: .byte	8
; CHECK-NEXT: .byte	14
; CHECK-NEXT: .short	9
; CHECK-NEXT: .byte	8
; CHECK-NEXT: .byte	7
; CHECK-NEXT: .byte	9
; CHECK-NEXT: .long	8
; CHECK-NEXT: .byte	10
; CHECK-NEXT: .long	8
; CHECK-NEXT: .byte	13
; CHECK-NEXT: .byte	9
; CHECK-NEXT: .byte	14
; CHECK-NEXT: .short	9
; CHECK-NEXT: .byte	8
; CHECK-NEXT: .byte	7
; CHECK-NEXT: .byte	9
; CHECK-NEXT: .long	8
; CHECK-NEXT: .byte	10
; CHECK-NEXT: .long	8
; CHECK-NEXT: .byte	13
; CHECK-NEXT: .byte	11
; CHECK-NEXT: .byte	14
; CHECK-NEXT: .short	1
; CHECK-NEXT: .byte	15
; CHECK-NEXT: .byte	1
; CHECK-NEXT: .byte	8
; CHECK-NEXT: .byte	5

define amdgpu_kernel void @test_ulong4(<4 x i64> %a) !kernel_arg_addr_space !1 !kernel_arg_access_qual !2 !kernel_arg_type !12 !kernel_arg_base_type !12 !kernel_arg_type_qual !4 {
  ret void
}

; CHECK-LABEL:{{^}}test_half8:
; CHECK: .section        .AMDGPU.runtime_metadata
; CHECK-NEXT: .byte	4
; CHECK-NEXT: .byte	6
; CHECK-NEXT: .long	10
; CHECK-NEXT: .ascii	"test_half8"
; CHECK-NEXT: .byte	7
; CHECK-NEXT: .byte	9
; CHECK-NEXT: .long	16
; CHECK-NEXT: .byte	10
; CHECK-NEXT: .long	16
; CHECK-NEXT: .byte	11
; CHECK-NEXT: .long	5
; CHECK-NEXT: .ascii	"half8"
; CHECK-NEXT: .byte	13
; CHECK-NEXT: .byte	0
; CHECK-NEXT: .byte	14
; CHECK-NEXT: .short	5
; CHECK-NEXT: .byte	16
; CHECK-NEXT: .byte	0
; CHECK-NEXT: .byte	8
; CHECK-NEXT: .byte	7
; CHECK-NEXT: .byte	9
; CHECK-NEXT: .long	8
; CHECK-NEXT: .byte	10
; CHECK-NEXT: .long	8
; CHECK-NEXT: .byte	13
; CHECK-NEXT: .byte	7
; CHECK-NEXT: .byte	14
; CHECK-NEXT: .short	9
; CHECK-NEXT: .byte	8
; CHECK-NEXT: .byte	7
; CHECK-NEXT: .byte	9
; CHECK-NEXT: .long	8
; CHECK-NEXT: .byte	10
; CHECK-NEXT: .long	8
; CHECK-NEXT: .byte	13
; CHECK-NEXT: .byte	8
; CHECK-NEXT: .byte	14
; CHECK-NEXT: .short	9
; CHECK-NEXT: .byte	8
; CHECK-NEXT: .byte	7
; CHECK-NEXT: .byte	9
; CHECK-NEXT: .long	8
; CHECK-NEXT: .byte	10
; CHECK-NEXT: .long	8
; CHECK-NEXT: .byte	13
; CHECK-NEXT: .byte	9
; CHECK-NEXT: .byte	14
; CHECK-NEXT: .short	9
; CHECK-NEXT: .byte	8
; CHECK-NEXT: .byte	7
; CHECK-NEXT: .byte	9
; CHECK-NEXT: .long	8
; CHECK-NEXT: .byte	10
; CHECK-NEXT: .long	8
; CHECK-NEXT: .byte	13
; CHECK-NEXT: .byte	11
; CHECK-NEXT: .byte	14
; CHECK-NEXT: .short	1
; CHECK-NEXT: .byte	15
; CHECK-NEXT: .byte	1
; CHECK-NEXT: .byte	8
; CHECK-NEXT: .byte	5

define amdgpu_kernel void @test_half8(<8 x half> %a) !kernel_arg_addr_space !1 !kernel_arg_access_qual !2 !kernel_arg_type !13 !kernel_arg_base_type !13 !kernel_arg_type_qual !4 {
  ret void
}

; CHECK-LABEL:{{^}}test_float16:
; CHECK: .section        .AMDGPU.runtime_metadata
; CHECK-NEXT: .byte	4
; CHECK-NEXT: .byte	6
; CHECK-NEXT: .long	12
; CHECK-NEXT: .ascii	"test_float16"
; CHECK-NEXT: .byte	7
; CHECK-NEXT: .byte	9
; CHECK-NEXT: .long	64
; CHECK-NEXT: .byte	10
; CHECK-NEXT: .long	64
; CHECK-NEXT: .byte	11
; CHECK-NEXT: .long	7
; CHECK-NEXT: .ascii	"float16"
; CHECK-NEXT: .byte	13
; CHECK-NEXT: .byte	0
; CHECK-NEXT: .byte	14
; CHECK-NEXT: .short	8
; CHECK-NEXT: .byte	16
; CHECK-NEXT: .byte	0
; CHECK-NEXT: .byte	8
; CHECK-NEXT: .byte	7
; CHECK-NEXT: .byte	9
; CHECK-NEXT: .long	8
; CHECK-NEXT: .byte	10
; CHECK-NEXT: .long	8
; CHECK-NEXT: .byte	13
; CHECK-NEXT: .byte	7
; CHECK-NEXT: .byte	14
; CHECK-NEXT: .short	9
; CHECK-NEXT: .byte	8
; CHECK-NEXT: .byte	7
; CHECK-NEXT: .byte	9
; CHECK-NEXT: .long	8
; CHECK-NEXT: .byte	10
; CHECK-NEXT: .long	8
; CHECK-NEXT: .byte	13
; CHECK-NEXT: .byte	8
; CHECK-NEXT: .byte	14
; CHECK-NEXT: .short	9
; CHECK-NEXT: .byte	8
; CHECK-NEXT: .byte	7
; CHECK-NEXT: .byte	9
; CHECK-NEXT: .long	8
; CHECK-NEXT: .byte	10
; CHECK-NEXT: .long	8
; CHECK-NEXT: .byte	13
; CHECK-NEXT: .byte	9
; CHECK-NEXT: .byte	14
; CHECK-NEXT: .short	9
; CHECK-NEXT: .byte	8
; CHECK-NEXT: .byte	7
; CHECK-NEXT: .byte	9
; CHECK-NEXT: .long	8
; CHECK-NEXT: .byte	10
; CHECK-NEXT: .long	8
; CHECK-NEXT: .byte	13
; CHECK-NEXT: .byte	11
; CHECK-NEXT: .byte	14
; CHECK-NEXT: .short	1
; CHECK-NEXT: .byte	15
; CHECK-NEXT: .byte	1
; CHECK-NEXT: .byte	8
; CHECK-NEXT: .byte	5

define amdgpu_kernel void @test_float16(<16 x float> %a) !kernel_arg_addr_space !1 !kernel_arg_access_qual !2 !kernel_arg_type !14 !kernel_arg_base_type !14 !kernel_arg_type_qual !4 {
  ret void
}

; CHECK-LABEL:{{^}}test_double16:
; CHECK: .section        .AMDGPU.runtime_metadata
; CHECK-NEXT: .byte	4
; CHECK-NEXT: .byte	6
; CHECK-NEXT: .long	13
; CHECK-NEXT: .ascii	"test_double16"
; CHECK-NEXT: .byte	7
; CHECK-NEXT: .byte	9
; CHECK-NEXT: .long	128
; CHECK-NEXT: .byte	10
; CHECK-NEXT: .long	128
; CHECK-NEXT: .byte	11
; CHECK-NEXT: .long	8
; CHECK-NEXT: .ascii	"double16"
; CHECK-NEXT: .byte	13
; CHECK-NEXT: .byte	0
; CHECK-NEXT: .byte	14
; CHECK-NEXT: .short	11
; CHECK-NEXT: .byte	16
; CHECK-NEXT: .byte	0
; CHECK-NEXT: .byte	8
; CHECK-NEXT: .byte	7
; CHECK-NEXT: .byte	9
; CHECK-NEXT: .long	8
; CHECK-NEXT: .byte	10
; CHECK-NEXT: .long	8
; CHECK-NEXT: .byte	13
; CHECK-NEXT: .byte	7
; CHECK-NEXT: .byte	14
; CHECK-NEXT: .short	9
; CHECK-NEXT: .byte	8
; CHECK-NEXT: .byte	7
; CHECK-NEXT: .byte	9
; CHECK-NEXT: .long	8
; CHECK-NEXT: .byte	10
; CHECK-NEXT: .long	8
; CHECK-NEXT: .byte	13
; CHECK-NEXT: .byte	8
; CHECK-NEXT: .byte	14
; CHECK-NEXT: .short	9
; CHECK-NEXT: .byte	8
; CHECK-NEXT: .byte	7
; CHECK-NEXT: .byte	9
; CHECK-NEXT: .long	8
; CHECK-NEXT: .byte	10
; CHECK-NEXT: .long	8
; CHECK-NEXT: .byte	13
; CHECK-NEXT: .byte	9
; CHECK-NEXT: .byte	14
; CHECK-NEXT: .short	9
; CHECK-NEXT: .byte	8
; CHECK-NEXT: .byte	7
; CHECK-NEXT: .byte	9
; CHECK-NEXT: .long	8
; CHECK-NEXT: .byte	10
; CHECK-NEXT: .long	8
; CHECK-NEXT: .byte	13
; CHECK-NEXT: .byte	11
; CHECK-NEXT: .byte	14
; CHECK-NEXT: .short	1
; CHECK-NEXT: .byte	15
; CHECK-NEXT: .byte	1
; CHECK-NEXT: .byte	8
; CHECK-NEXT: .byte	5

define amdgpu_kernel void @test_double16(<16 x double> %a) !kernel_arg_addr_space !1 !kernel_arg_access_qual !2 !kernel_arg_type !15 !kernel_arg_base_type !15 !kernel_arg_type_qual !4 {
  ret void
}

; CHECK-LABEL:{{^}}test_pointer:
; CHECK: .section        .AMDGPU.runtime_metadata
; CHECK-NEXT: .byte	4
; CHECK-NEXT: .byte	6
; CHECK-NEXT: .long	12
; CHECK-NEXT: .ascii	"test_pointer"
; CHECK-NEXT: .byte	7
; CHECK-NEXT: .byte	9
; CHECK-NEXT: .long	8
; CHECK-NEXT: .byte	10
; CHECK-NEXT: .long	8
; CHECK-NEXT: .byte	11
; CHECK-NEXT: .long	5
; CHECK-NEXT: .ascii	"int *"
; CHECK-NEXT: .byte	13
; CHECK-NEXT: .byte	1
; CHECK-NEXT: .byte	14
; CHECK-NEXT: .short	6
; CHECK-NEXT: .byte	16
; CHECK-NEXT: .byte	0
; CHECK-NEXT: .byte	15
; CHECK-NEXT: .byte	1
; CHECK-NEXT: .byte	8
; CHECK-NEXT: .byte	7
; CHECK-NEXT: .byte	9
; CHECK-NEXT: .long	8
; CHECK-NEXT: .byte	10
; CHECK-NEXT: .long	8
; CHECK-NEXT: .byte	13
; CHECK-NEXT: .byte	7
; CHECK-NEXT: .byte	14
; CHECK-NEXT: .short	9
; CHECK-NEXT: .byte	8
; CHECK-NEXT: .byte	7
; CHECK-NEXT: .byte	9
; CHECK-NEXT: .long	8
; CHECK-NEXT: .byte	10
; CHECK-NEXT: .long	8
; CHECK-NEXT: .byte	13
; CHECK-NEXT: .byte	8
; CHECK-NEXT: .byte	14
; CHECK-NEXT: .short	9
; CHECK-NEXT: .byte	8
; CHECK-NEXT: .byte	7
; CHECK-NEXT: .byte	9
; CHECK-NEXT: .long	8
; CHECK-NEXT: .byte	10
; CHECK-NEXT: .long	8
; CHECK-NEXT: .byte	13
; CHECK-NEXT: .byte	9
; CHECK-NEXT: .byte	14
; CHECK-NEXT: .short	9
; CHECK-NEXT: .byte	8
; CHECK-NEXT: .byte	7
; CHECK-NEXT: .byte	9
; CHECK-NEXT: .long	8
; CHECK-NEXT: .byte	10
; CHECK-NEXT: .long	8
; CHECK-NEXT: .byte	13
; CHECK-NEXT: .byte	11
; CHECK-NEXT: .byte	14
; CHECK-NEXT: .short	1
; CHECK-NEXT: .byte	15
; CHECK-NEXT: .byte	1
; CHECK-NEXT: .byte	8
; CHECK-NEXT: .byte	5

define amdgpu_kernel void @test_pointer(i32 addrspace(1)* %a) !kernel_arg_addr_space !1 !kernel_arg_access_qual !2 !kernel_arg_type !16 !kernel_arg_base_type !16 !kernel_arg_type_qual !4 {
  ret void
}

; CHECK-LABEL:{{^}}test_image:
; CHECK: .section        .AMDGPU.runtime_metadata
; CHECK-NEXT: .byte	4
; CHECK-NEXT: .byte	6
; CHECK-NEXT: .long	10
; CHECK-NEXT: .ascii	"test_image"
; CHECK-NEXT: .byte	7
; CHECK-NEXT: .byte	9
; CHECK-NEXT: .long	8
; CHECK-NEXT: .byte	10
; CHECK-NEXT: .long	8
; CHECK-NEXT: .byte	11
; CHECK-NEXT: .long	9
; CHECK-NEXT: .ascii	"image2d_t"
; CHECK-NEXT: .byte	13
; CHECK-NEXT: .byte	4
; CHECK-NEXT: .byte	14
; CHECK-NEXT: .short	0
; CHECK-NEXT: .byte	16
; CHECK-NEXT: .byte	0
; CHECK-NEXT: .byte	15
; CHECK-NEXT: .byte	1
; CHECK-NEXT: .byte	8
; CHECK-NEXT: .byte	7
; CHECK-NEXT: .byte	9
; CHECK-NEXT: .long	8
; CHECK-NEXT: .byte	10
; CHECK-NEXT: .long	8
; CHECK-NEXT: .byte	13
; CHECK-NEXT: .byte	7
; CHECK-NEXT: .byte	14
; CHECK-NEXT: .short	9
; CHECK-NEXT: .byte	8
; CHECK-NEXT: .byte	7
; CHECK-NEXT: .byte	9
; CHECK-NEXT: .long	8
; CHECK-NEXT: .byte	10
; CHECK-NEXT: .long	8
; CHECK-NEXT: .byte	13
; CHECK-NEXT: .byte	8
; CHECK-NEXT: .byte	14
; CHECK-NEXT: .short	9
; CHECK-NEXT: .byte	8
; CHECK-NEXT: .byte	7
; CHECK-NEXT: .byte	9
; CHECK-NEXT: .long	8
; CHECK-NEXT: .byte	10
; CHECK-NEXT: .long	8
; CHECK-NEXT: .byte	13
; CHECK-NEXT: .byte	9
; CHECK-NEXT: .byte	14
; CHECK-NEXT: .short	9
; CHECK-NEXT: .byte	8
; CHECK-NEXT: .byte	7
; CHECK-NEXT: .byte	9
; CHECK-NEXT: .long	8
; CHECK-NEXT: .byte	10
; CHECK-NEXT: .long	8
; CHECK-NEXT: .byte	13
; CHECK-NEXT: .byte	11
; CHECK-NEXT: .byte	14
; CHECK-NEXT: .short	1
; CHECK-NEXT: .byte	15
; CHECK-NEXT: .byte	1
; CHECK-NEXT: .byte	8
; CHECK-NEXT: .byte	5

define amdgpu_kernel void @test_image(%opencl.image2d_t addrspace(1)* %a) !kernel_arg_addr_space !1 !kernel_arg_access_qual !2 !kernel_arg_type !17 !kernel_arg_base_type !17 !kernel_arg_type_qual !4 {
  ret void
}

; CHECK-LABEL:{{^}}test_sampler:
; CHECK: .section        .AMDGPU.runtime_metadata
; CHECK-NEXT: .byte	4
; CHECK-NEXT: .byte	6
; CHECK-NEXT: .long	12
; CHECK-NEXT: .ascii	"test_sampler"
; CHECK-NEXT: .byte	7
; CHECK-NEXT: .byte	9
; CHECK-NEXT: .long	4
; CHECK-NEXT: .byte	10
; CHECK-NEXT: .long	4
; CHECK-NEXT: .byte	11
; CHECK-NEXT: .long	9
; CHECK-NEXT: .ascii	"sampler_t"
; CHECK-NEXT: .byte	13
; CHECK-NEXT: .byte	3
; CHECK-NEXT: .byte	14
; CHECK-NEXT: .short	6
; CHECK-NEXT: .byte	16
; CHECK-NEXT: .byte	0
; CHECK-NEXT: .byte	8
; CHECK-NEXT: .byte	7
; CHECK-NEXT: .byte	9
; CHECK-NEXT: .long	8
; CHECK-NEXT: .byte	10
; CHECK-NEXT: .long	8
; CHECK-NEXT: .byte	13
; CHECK-NEXT: .byte	7
; CHECK-NEXT: .byte	14
; CHECK-NEXT: .short	9
; CHECK-NEXT: .byte	8
; CHECK-NEXT: .byte	7
; CHECK-NEXT: .byte	9
; CHECK-NEXT: .long	8
; CHECK-NEXT: .byte	10
; CHECK-NEXT: .long	8
; CHECK-NEXT: .byte	13
; CHECK-NEXT: .byte	8
; CHECK-NEXT: .byte	14
; CHECK-NEXT: .short	9
; CHECK-NEXT: .byte	8
; CHECK-NEXT: .byte	7
; CHECK-NEXT: .byte	9
; CHECK-NEXT: .long	8
; CHECK-NEXT: .byte	10
; CHECK-NEXT: .long	8
; CHECK-NEXT: .byte	13
; CHECK-NEXT: .byte	9
; CHECK-NEXT: .byte	14
; CHECK-NEXT: .short	9
; CHECK-NEXT: .byte	8
; CHECK-NEXT: .byte	7
; CHECK-NEXT: .byte	9
; CHECK-NEXT: .long	8
; CHECK-NEXT: .byte	10
; CHECK-NEXT: .long	8
; CHECK-NEXT: .byte	13
; CHECK-NEXT: .byte	11
; CHECK-NEXT: .byte	14
; CHECK-NEXT: .short	1
; CHECK-NEXT: .byte	15
; CHECK-NEXT: .byte	1
; CHECK-NEXT: .byte	8
; CHECK-NEXT: .byte	5

define amdgpu_kernel void @test_sampler(i32 %a) !kernel_arg_addr_space !1 !kernel_arg_access_qual !2 !kernel_arg_type !18 !kernel_arg_base_type !18 !kernel_arg_type_qual !4 {
  ret void
}

; CHECK-LABEL:{{^}}test_queue:
; CHECK: .section        .AMDGPU.runtime_metadata
; CHECK-NEXT: .byte	4
; CHECK-NEXT: .byte	6
; CHECK-NEXT: .long	10
; CHECK-NEXT: .ascii	"test_queue"
; CHECK-NEXT: .byte	7
; CHECK-NEXT: .byte	9
; CHECK-NEXT: .long	8
; CHECK-NEXT: .byte	10
; CHECK-NEXT: .long	8
; CHECK-NEXT: .byte	11
; CHECK-NEXT: .long	7
; CHECK-NEXT: .ascii	"queue_t"
; CHECK-NEXT: .byte	13
; CHECK-NEXT: .byte	6
; CHECK-NEXT: .byte	14
; CHECK-NEXT: .short	0
; CHECK-NEXT: .byte	16
; CHECK-NEXT: .byte	0
; CHECK-NEXT: .byte	15
; CHECK-NEXT: .byte	1
; CHECK-NEXT: .byte	8
; CHECK-NEXT: .byte	7
; CHECK-NEXT: .byte	9
; CHECK-NEXT: .long	8
; CHECK-NEXT: .byte	10
; CHECK-NEXT: .long	8
; CHECK-NEXT: .byte	13
; CHECK-NEXT: .byte	7
; CHECK-NEXT: .byte	14
; CHECK-NEXT: .short	9
; CHECK-NEXT: .byte	8
; CHECK-NEXT: .byte	7
; CHECK-NEXT: .byte	9
; CHECK-NEXT: .long	8
; CHECK-NEXT: .byte	10
; CHECK-NEXT: .long	8
; CHECK-NEXT: .byte	13
; CHECK-NEXT: .byte	8
; CHECK-NEXT: .byte	14
; CHECK-NEXT: .short	9
; CHECK-NEXT: .byte	8
; CHECK-NEXT: .byte	7
; CHECK-NEXT: .byte	9
; CHECK-NEXT: .long	8
; CHECK-NEXT: .byte	10
; CHECK-NEXT: .long	8
; CHECK-NEXT: .byte	13
; CHECK-NEXT: .byte	9
; CHECK-NEXT: .byte	14
; CHECK-NEXT: .short	9
; CHECK-NEXT: .byte	8
; CHECK-NEXT: .byte	7
; CHECK-NEXT: .byte	9
; CHECK-NEXT: .long	8
; CHECK-NEXT: .byte	10
; CHECK-NEXT: .long	8
; CHECK-NEXT: .byte	13
; CHECK-NEXT: .byte	11
; CHECK-NEXT: .byte	14
; CHECK-NEXT: .short	1
; CHECK-NEXT: .byte	15
; CHECK-NEXT: .byte	1
; CHECK-NEXT: .byte	8
; CHECK-NEXT: .byte	5

define amdgpu_kernel void @test_queue(%opencl.queue_t addrspace(1)* %a) !kernel_arg_addr_space !1 !kernel_arg_access_qual !2 !kernel_arg_type !19 !kernel_arg_base_type !19 !kernel_arg_type_qual !4 {
  ret void
}

; CHECK-LABEL:{{^}}test_struct:
; CHECK: .section        .AMDGPU.runtime_metadata
; CHECK-NEXT: .byte	4
; CHECK-NEXT: .byte	6
; CHECK-NEXT: .long	11
; CHECK-NEXT: .ascii	"test_struct"
; CHECK-NEXT: .byte	7
; CHECK-NEXT: .byte	9
; CHECK-NEXT: .long	4
; CHECK-NEXT: .byte	10
; CHECK-NEXT: .long	4
; CHECK-NEXT: .byte	11
; CHECK-NEXT: .long	8
; CHECK-NEXT: .ascii	"struct A"
; CHECK-NEXT: .byte	13
; CHECK-NEXT: .byte	1
; CHECK-NEXT: .byte	14
; CHECK-NEXT: .short	0
; CHECK-NEXT: .byte	16
; CHECK-NEXT: .byte	0
; CHECK-NEXT: .byte	15
; CHECK-NEXT: .byte	0
; CHECK-NEXT: .byte	8
; CHECK-NEXT: .byte	7
; CHECK-NEXT: .byte	9
; CHECK-NEXT: .long	8
; CHECK-NEXT: .byte	10
; CHECK-NEXT: .long	8
; CHECK-NEXT: .byte	13
; CHECK-NEXT: .byte	7
; CHECK-NEXT: .byte	14
; CHECK-NEXT: .short	9
; CHECK-NEXT: .byte	8
; CHECK-NEXT: .byte	7
; CHECK-NEXT: .byte	9
; CHECK-NEXT: .long	8
; CHECK-NEXT: .byte	10
; CHECK-NEXT: .long	8
; CHECK-NEXT: .byte	13
; CHECK-NEXT: .byte	8
; CHECK-NEXT: .byte	14
; CHECK-NEXT: .short	9
; CHECK-NEXT: .byte	8
; CHECK-NEXT: .byte	7
; CHECK-NEXT: .byte	9
; CHECK-NEXT: .long	8
; CHECK-NEXT: .byte	10
; CHECK-NEXT: .long	8
; CHECK-NEXT: .byte	13
; CHECK-NEXT: .byte	9
; CHECK-NEXT: .byte	14
; CHECK-NEXT: .short	9
; CHECK-NEXT: .byte	8
; CHECK-NEXT: .byte	7
; CHECK-NEXT: .byte	9
; CHECK-NEXT: .long	8
; CHECK-NEXT: .byte	10
; CHECK-NEXT: .long	8
; CHECK-NEXT: .byte	13
; CHECK-NEXT: .byte	11
; CHECK-NEXT: .byte	14
; CHECK-NEXT: .short	1
; CHECK-NEXT: .byte	15
; CHECK-NEXT: .byte	1
; CHECK-NEXT: .byte	8
; CHECK-NEXT: .byte	5

define amdgpu_kernel void @test_struct(%struct.A* byval %a) !kernel_arg_addr_space !1 !kernel_arg_access_qual !2 !kernel_arg_type !20 !kernel_arg_base_type !20 !kernel_arg_type_qual !4 {
  ret void
}

; CHECK-LABEL:{{^}}test_i128:
; CHECK: .section        .AMDGPU.runtime_metadata
; CHECK-NEXT: .byte	4
; CHECK-NEXT: .byte	6
; CHECK-NEXT: .long	9
; CHECK-NEXT: .ascii	"test_i128"
; CHECK-NEXT: .byte	7
; CHECK-NEXT: .byte	9
; CHECK-NEXT: .long	16
; CHECK-NEXT: .byte	10
; CHECK-NEXT: .long	8
; CHECK-NEXT: .byte	11
; CHECK-NEXT: .long	4
; CHECK-NEXT: .ascii	"i128"
; CHECK-NEXT: .byte	13
; CHECK-NEXT: .byte	0
; CHECK-NEXT: .byte	14
; CHECK-NEXT: .short	0
; CHECK-NEXT: .byte	16
; CHECK-NEXT: .byte	0
; CHECK-NEXT: .byte	8
; CHECK-NEXT: .byte	7
; CHECK-NEXT: .byte	9
; CHECK-NEXT: .long	8
; CHECK-NEXT: .byte	10
; CHECK-NEXT: .long	8
; CHECK-NEXT: .byte	13
; CHECK-NEXT: .byte	7
; CHECK-NEXT: .byte	14
; CHECK-NEXT: .short	9
; CHECK-NEXT: .byte	8
; CHECK-NEXT: .byte	7
; CHECK-NEXT: .byte	9
; CHECK-NEXT: .long	8
; CHECK-NEXT: .byte	10
; CHECK-NEXT: .long	8
; CHECK-NEXT: .byte	13
; CHECK-NEXT: .byte	8
; CHECK-NEXT: .byte	14
; CHECK-NEXT: .short	9
; CHECK-NEXT: .byte	8
; CHECK-NEXT: .byte	7
; CHECK-NEXT: .byte	9
; CHECK-NEXT: .long	8
; CHECK-NEXT: .byte	10
; CHECK-NEXT: .long	8
; CHECK-NEXT: .byte	13
; CHECK-NEXT: .byte	9
; CHECK-NEXT: .byte	14
; CHECK-NEXT: .short	9
; CHECK-NEXT: .byte	8
; CHECK-NEXT: .byte	7
; CHECK-NEXT: .byte	9
; CHECK-NEXT: .long	8
; CHECK-NEXT: .byte	10
; CHECK-NEXT: .long	8
; CHECK-NEXT: .byte	13
; CHECK-NEXT: .byte	11
; CHECK-NEXT: .byte	14
; CHECK-NEXT: .short	1
; CHECK-NEXT: .byte	15
; CHECK-NEXT: .byte	1
; CHECK-NEXT: .byte	8
; CHECK-NEXT: .byte	5

define amdgpu_kernel void @test_i128(i128 %a) !kernel_arg_addr_space !1 !kernel_arg_access_qual !2 !kernel_arg_type !21 !kernel_arg_base_type !21 !kernel_arg_type_qual !4 {
  ret void
}

; CHECK-LABEL:{{^}}test_multi_arg:
; CHECK: .section        .AMDGPU.runtime_metadata
; CHECK-NEXT: .byte	4
; CHECK-NEXT: .byte	6
; CHECK-NEXT: .long	14
; CHECK-NEXT: .ascii	"test_multi_arg"
; CHECK-NEXT: .byte	7
; CHECK-NEXT: .byte	9
; CHECK-NEXT: .long	4
; CHECK-NEXT: .byte	10
; CHECK-NEXT: .long	4
; CHECK-NEXT: .byte	11
; CHECK-NEXT: .long	3
; CHECK-NEXT: .ascii	"int"
; CHECK-NEXT: .byte	13
; CHECK-NEXT: .byte	0
; CHECK-NEXT: .byte	14
; CHECK-NEXT: .short	6
; CHECK-NEXT: .byte	16
; CHECK-NEXT: .byte	0
; CHECK-NEXT: .byte	8
; CHECK-NEXT: .byte	7
; CHECK-NEXT: .byte	9
; CHECK-NEXT: .long	4
; CHECK-NEXT: .byte	10
; CHECK-NEXT: .long	4
; CHECK-NEXT: .byte	11
; CHECK-NEXT: .long	6
; CHECK-NEXT: .ascii	"short2"
; CHECK-NEXT: .byte	13
; CHECK-NEXT: .byte	0
; CHECK-NEXT: .byte	14
; CHECK-NEXT: .short	3
; CHECK-NEXT: .byte	16
; CHECK-NEXT: .byte	0
; CHECK-NEXT: .byte	8
; CHECK-NEXT: .byte	7
; CHECK-NEXT: .byte	9
; CHECK-NEXT: .long	4
; CHECK-NEXT: .byte	10
; CHECK-NEXT: .long	4
; CHECK-NEXT: .byte	11
; CHECK-NEXT: .long	5
; CHECK-NEXT: .ascii	"char3"
; CHECK-NEXT: .byte	13
; CHECK-NEXT: .byte	0
; CHECK-NEXT: .byte	14
; CHECK-NEXT: .short	1
; CHECK-NEXT: .byte	16
; CHECK-NEXT: .byte	0
; CHECK-NEXT: .byte	8
; CHECK-NEXT: .byte	7
; CHECK-NEXT: .byte	9
; CHECK-NEXT: .long	8
; CHECK-NEXT: .byte	10
; CHECK-NEXT: .long	8
; CHECK-NEXT: .byte	13
; CHECK-NEXT: .byte	7
; CHECK-NEXT: .byte	14
; CHECK-NEXT: .short	9
; CHECK-NEXT: .byte	8
; CHECK-NEXT: .byte	7
; CHECK-NEXT: .byte	9
; CHECK-NEXT: .long	8
; CHECK-NEXT: .byte	10
; CHECK-NEXT: .long	8
; CHECK-NEXT: .byte	13
; CHECK-NEXT: .byte	8
; CHECK-NEXT: .byte	14
; CHECK-NEXT: .short	9
; CHECK-NEXT: .byte	8
; CHECK-NEXT: .byte	7
; CHECK-NEXT: .byte	9
; CHECK-NEXT: .long	8
; CHECK-NEXT: .byte	10
; CHECK-NEXT: .long	8
; CHECK-NEXT: .byte	13
; CHECK-NEXT: .byte	9
; CHECK-NEXT: .byte	14
; CHECK-NEXT: .short	9
; CHECK-NEXT: .byte	8
; CHECK-NEXT: .byte	7
; CHECK-NEXT: .byte	9
; CHECK-NEXT: .long	8
; CHECK-NEXT: .byte	10
; CHECK-NEXT: .long	8
; CHECK-NEXT: .byte	13
; CHECK-NEXT: .byte	11
; CHECK-NEXT: .byte	14
; CHECK-NEXT: .short	1
; CHECK-NEXT: .byte	15
; CHECK-NEXT: .byte	1
; CHECK-NEXT: .byte	8
; CHECK-NEXT: .byte	5

define amdgpu_kernel void @test_multi_arg(i32 %a, <2 x i16> %b, <3 x i8> %c) !kernel_arg_addr_space !22 !kernel_arg_access_qual !23 !kernel_arg_type !24 !kernel_arg_base_type !24 !kernel_arg_type_qual !25 {
  ret void
}

; CHECK-LABEL:{{^}}test_addr_space:
; CHECK: .section        .AMDGPU.runtime_metadata
; CHECK-NEXT: .byte	4
; CHECK-NEXT: .byte	6
; CHECK-NEXT: .long	15
; CHECK-NEXT: .ascii	"test_addr_space"
; CHECK-NEXT: .byte	7
; CHECK-NEXT: .byte	9
; CHECK-NEXT: .long	8
; CHECK-NEXT: .byte	10
; CHECK-NEXT: .long	8
; CHECK-NEXT: .byte	11
; CHECK-NEXT: .long	5
; CHECK-NEXT: .ascii	"int *"
; CHECK-NEXT: .byte	13
; CHECK-NEXT: .byte	1
; CHECK-NEXT: .byte	14
; CHECK-NEXT: .short	6
; CHECK-NEXT: .byte	16
; CHECK-NEXT: .byte	0
; CHECK-NEXT: .byte	15
; CHECK-NEXT: .byte	1
; CHECK-NEXT: .byte	8
; CHECK-NEXT: .byte	7
; CHECK-NEXT: .byte	9
; CHECK-NEXT: .long	8
; CHECK-NEXT: .byte	10
; CHECK-NEXT: .long	8
; CHECK-NEXT: .byte	11
; CHECK-NEXT: .long	5
; CHECK-NEXT: .ascii	"int *"
; CHECK-NEXT: .byte	13
; CHECK-NEXT: .byte	1
; CHECK-NEXT: .byte	14
; CHECK-NEXT: .short	6
; CHECK-NEXT: .byte	16
; CHECK-NEXT: .byte	0
; CHECK-NEXT: .byte	15
; CHECK-NEXT: .byte	2
; CHECK-NEXT: .byte	8
; CHECK-NEXT: .byte	7
; CHECK-NEXT: .byte	9
; CHECK-NEXT: .long	4
; CHECK-NEXT: .byte	10
; CHECK-NEXT: .long	4
; CHECK-NEXT: .byte	32
; CHECK-NEXT: .long	4
; CHECK-NEXT: .byte	11
; CHECK-NEXT: .long	5
; CHECK-NEXT: .ascii	"int *"
; CHECK-NEXT: .byte	13
; CHECK-NEXT: .byte	2
; CHECK-NEXT: .byte	14
; CHECK-NEXT: .short	6
; CHECK-NEXT: .byte	16
; CHECK-NEXT: .byte	0
; CHECK-NEXT: .byte	15
; CHECK-NEXT: .byte	3
; CHECK-NEXT: .byte	8
; CHECK-NEXT: .byte	7
; CHECK-NEXT: .byte	9
; CHECK-NEXT: .long	8
; CHECK-NEXT: .byte	10
; CHECK-NEXT: .long	8
; CHECK-NEXT: .byte	13
; CHECK-NEXT: .byte	7
; CHECK-NEXT: .byte	14
; CHECK-NEXT: .short	9
; CHECK-NEXT: .byte	8
; CHECK-NEXT: .byte	7
; CHECK-NEXT: .byte	9
; CHECK-NEXT: .long	8
; CHECK-NEXT: .byte	10
; CHECK-NEXT: .long	8
; CHECK-NEXT: .byte	13
; CHECK-NEXT: .byte	8
; CHECK-NEXT: .byte	14
; CHECK-NEXT: .short	9
; CHECK-NEXT: .byte	8
; CHECK-NEXT: .byte	7
; CHECK-NEXT: .byte	9
; CHECK-NEXT: .long	8
; CHECK-NEXT: .byte	10
; CHECK-NEXT: .long	8
; CHECK-NEXT: .byte	13
; CHECK-NEXT: .byte	9
; CHECK-NEXT: .byte	14
; CHECK-NEXT: .short	9
; CHECK-NEXT: .byte	8
; CHECK-NEXT: .byte	7
; CHECK-NEXT: .byte	9
; CHECK-NEXT: .long	8
; CHECK-NEXT: .byte	10
; CHECK-NEXT: .long	8
; CHECK-NEXT: .byte	13
; CHECK-NEXT: .byte	11
; CHECK-NEXT: .byte	14
; CHECK-NEXT: .short	1
; CHECK-NEXT: .byte	15
; CHECK-NEXT: .byte	1
; CHECK-NEXT: .byte	8
; CHECK-NEXT: .byte	5

define amdgpu_kernel void @test_addr_space(i32 addrspace(1)* %g, i32 addrspace(2)* %c, i32 addrspace(3)* %l) !kernel_arg_addr_space !50 !kernel_arg_access_qual !23 !kernel_arg_type !51 !kernel_arg_base_type !51 !kernel_arg_type_qual !25 {
  ret void
}

; CHECK-LABEL:{{^}}test_type_qual:
; CHECK: .section        .AMDGPU.runtime_metadata
; CHECK-NEXT: .byte	4
; CHECK-NEXT: .byte	6
; CHECK-NEXT: .long	14
; CHECK-NEXT: .ascii	"test_type_qual"
; CHECK-NEXT: .byte	7
; CHECK-NEXT: .byte	9
; CHECK-NEXT: .long	8
; CHECK-NEXT: .byte	10
; CHECK-NEXT: .long	8
; CHECK-NEXT: .byte	11
; CHECK-NEXT: .long	5
; CHECK-NEXT: .ascii	"int *"
; CHECK-NEXT: .byte	19
; CHECK-NEXT: .byte	13
; CHECK-NEXT: .byte	1
; CHECK-NEXT: .byte	14
; CHECK-NEXT: .short	6
; CHECK-NEXT: .byte	16
; CHECK-NEXT: .byte	0
; CHECK-NEXT: .byte	15
; CHECK-NEXT: .byte	1
; CHECK-NEXT: .byte	8
; CHECK-NEXT: .byte	7
; CHECK-NEXT: .byte	9
; CHECK-NEXT: .long	8
; CHECK-NEXT: .byte	10
; CHECK-NEXT: .long	8
; CHECK-NEXT: .byte	11
; CHECK-NEXT: .long	5
; CHECK-NEXT: .ascii	"int *"
; CHECK-NEXT: .byte	17
; CHECK-NEXT: .byte	18
; CHECK-NEXT: .byte	13
; CHECK-NEXT: .byte	1
; CHECK-NEXT: .byte	14
; CHECK-NEXT: .short	6
; CHECK-NEXT: .byte	16
; CHECK-NEXT: .byte	0
; CHECK-NEXT: .byte	15
; CHECK-NEXT: .byte	1
; CHECK-NEXT: .byte	8
; CHECK-NEXT: .byte	7
; CHECK-NEXT: .byte	9
; CHECK-NEXT: .long	8
; CHECK-NEXT: .byte	10
; CHECK-NEXT: .long	8
; CHECK-NEXT: .byte	11
; CHECK-NEXT: .long	5
; CHECK-NEXT: .ascii	"int *"
; CHECK-NEXT: .byte	20
; CHECK-NEXT: .byte	13
; CHECK-NEXT: .byte	5
; CHECK-NEXT: .byte	14
; CHECK-NEXT: .short	0
; CHECK-NEXT: .byte	16
; CHECK-NEXT: .byte	0
; CHECK-NEXT: .byte	15
; CHECK-NEXT: .byte	1
; CHECK-NEXT: .byte	8
; CHECK-NEXT: .byte	7
; CHECK-NEXT: .byte	9
; CHECK-NEXT: .long	8
; CHECK-NEXT: .byte	10
; CHECK-NEXT: .long	8
; CHECK-NEXT: .byte	13
; CHECK-NEXT: .byte	7
; CHECK-NEXT: .byte	14
; CHECK-NEXT: .short	9
; CHECK-NEXT: .byte	8
; CHECK-NEXT: .byte	7
; CHECK-NEXT: .byte	9
; CHECK-NEXT: .long	8
; CHECK-NEXT: .byte	10
; CHECK-NEXT: .long	8
; CHECK-NEXT: .byte	13
; CHECK-NEXT: .byte	8
; CHECK-NEXT: .byte	14
; CHECK-NEXT: .short	9
; CHECK-NEXT: .byte	8
; CHECK-NEXT: .byte	7
; CHECK-NEXT: .byte	9
; CHECK-NEXT: .long	8
; CHECK-NEXT: .byte	10
; CHECK-NEXT: .long	8
; CHECK-NEXT: .byte	13
; CHECK-NEXT: .byte	9
; CHECK-NEXT: .byte	14
; CHECK-NEXT: .short	9
; CHECK-NEXT: .byte	8
; CHECK-NEXT: .byte	7
; CHECK-NEXT: .byte	9
; CHECK-NEXT: .long	8
; CHECK-NEXT: .byte	10
; CHECK-NEXT: .long	8
; CHECK-NEXT: .byte	13
; CHECK-NEXT: .byte	11
; CHECK-NEXT: .byte	14
; CHECK-NEXT: .short	1
; CHECK-NEXT: .byte	15
; CHECK-NEXT: .byte	1
; CHECK-NEXT: .byte	8
; CHECK-NEXT: .byte	5

define amdgpu_kernel void @test_type_qual(i32 addrspace(1)* %a, i32 addrspace(1)* %b, %opencl.pipe_t addrspace(1)* %c) !kernel_arg_addr_space !22 !kernel_arg_access_qual !23 !kernel_arg_type !51 !kernel_arg_base_type !51 !kernel_arg_type_qual !70 {
  ret void
}

; CHECK-LABEL:{{^}}test_access_qual:
; CHECK: .section        .AMDGPU.runtime_metadata
; CHECK-NEXT: .byte	4
; CHECK-NEXT: .byte	6
; CHECK-NEXT: .long	16
; CHECK-NEXT: .ascii	"test_access_qual"
; CHECK-NEXT: .byte	7
; CHECK-NEXT: .byte	9
; CHECK-NEXT: .long	8
; CHECK-NEXT: .byte	10
; CHECK-NEXT: .long	8
; CHECK-NEXT: .byte	11
; CHECK-NEXT: .long	9
; CHECK-NEXT: .ascii	"image1d_t"
; CHECK-NEXT: .byte	13
; CHECK-NEXT: .byte	4
; CHECK-NEXT: .byte	14
; CHECK-NEXT: .short	0
; CHECK-NEXT: .byte	16
; CHECK-NEXT: .byte	1
; CHECK-NEXT: .byte	15
; CHECK-NEXT: .byte	1
; CHECK-NEXT: .byte	8
; CHECK-NEXT: .byte	7
; CHECK-NEXT: .byte	9
; CHECK-NEXT: .long	8
; CHECK-NEXT: .byte	10
; CHECK-NEXT: .long	8
; CHECK-NEXT: .byte	11
; CHECK-NEXT: .long	9
; CHECK-NEXT: .ascii	"image2d_t"
; CHECK-NEXT: .byte	13
; CHECK-NEXT: .byte	4
; CHECK-NEXT: .byte	14
; CHECK-NEXT: .short	0
; CHECK-NEXT: .byte	16
; CHECK-NEXT: .byte	2
; CHECK-NEXT: .byte	15
; CHECK-NEXT: .byte	1
; CHECK-NEXT: .byte	8
; CHECK-NEXT: .byte	7
; CHECK-NEXT: .byte	9
; CHECK-NEXT: .long	8
; CHECK-NEXT: .byte	10
; CHECK-NEXT: .long	8
; CHECK-NEXT: .byte	11
; CHECK-NEXT: .long	9
; CHECK-NEXT: .ascii	"image3d_t"
; CHECK-NEXT: .byte	13
; CHECK-NEXT: .byte	4
; CHECK-NEXT: .byte	14
; CHECK-NEXT: .short	0
; CHECK-NEXT: .byte	16
; CHECK-NEXT: .byte	3
; CHECK-NEXT: .byte	15
; CHECK-NEXT: .byte	1
; CHECK-NEXT: .byte	8
; CHECK-NEXT: .byte	7
; CHECK-NEXT: .byte	9
; CHECK-NEXT: .long	8
; CHECK-NEXT: .byte	10
; CHECK-NEXT: .long	8
; CHECK-NEXT: .byte	13
; CHECK-NEXT: .byte	7
; CHECK-NEXT: .byte	14
; CHECK-NEXT: .short	9
; CHECK-NEXT: .byte	8
; CHECK-NEXT: .byte	7
; CHECK-NEXT: .byte	9
; CHECK-NEXT: .long	8
; CHECK-NEXT: .byte	10
; CHECK-NEXT: .long	8
; CHECK-NEXT: .byte	13
; CHECK-NEXT: .byte	8
; CHECK-NEXT: .byte	14
; CHECK-NEXT: .short	9
; CHECK-NEXT: .byte	8
; CHECK-NEXT: .byte	7
; CHECK-NEXT: .byte	9
; CHECK-NEXT: .long	8
; CHECK-NEXT: .byte	10
; CHECK-NEXT: .long	8
; CHECK-NEXT: .byte	13
; CHECK-NEXT: .byte	9
; CHECK-NEXT: .byte	14
; CHECK-NEXT: .short	9
; CHECK-NEXT: .byte	8
; CHECK-NEXT: .byte	7
; CHECK-NEXT: .byte	9
; CHECK-NEXT: .long	8
; CHECK-NEXT: .byte	10
; CHECK-NEXT: .long	8
; CHECK-NEXT: .byte	13
; CHECK-NEXT: .byte	11
; CHECK-NEXT: .byte	14
; CHECK-NEXT: .short	1
; CHECK-NEXT: .byte	15
; CHECK-NEXT: .byte	1
; CHECK-NEXT: .byte	8
; CHECK-NEXT: .byte	5

define amdgpu_kernel void @test_access_qual(%opencl.image1d_t addrspace(1)* %ro, %opencl.image2d_t addrspace(1)* %wo, %opencl.image3d_t addrspace(1)* %rw) !kernel_arg_addr_space !60 !kernel_arg_access_qual !61 !kernel_arg_type !62 !kernel_arg_base_type !62 !kernel_arg_type_qual !25 {
  ret void
}

; CHECK-LABEL:{{^}}test_vec_type_hint_half:
; CHECK: .section        .AMDGPU.runtime_metadata
; CHECK-NEXT: .byte	4
; CHECK-NEXT: .byte	6
; CHECK-NEXT: .long	23
; CHECK-NEXT: .ascii	"test_vec_type_hint_half"
; CHECK-NEXT: .byte	7
; CHECK-NEXT: .byte	9
; CHECK-NEXT: .long	4
; CHECK-NEXT: .byte	10
; CHECK-NEXT: .long	4
; CHECK-NEXT: .byte	11
; CHECK-NEXT: .long	3
; CHECK-NEXT: .ascii	"int"
; CHECK-NEXT: .byte	13
; CHECK-NEXT: .byte	0
; CHECK-NEXT: .byte	14
; CHECK-NEXT: .short	6
; CHECK-NEXT: .byte	16
; CHECK-NEXT: .byte	0
; CHECK-NEXT: .byte	8
; CHECK-NEXT: .byte	7
; CHECK-NEXT: .byte	9
; CHECK-NEXT: .long	8
; CHECK-NEXT: .byte	10
; CHECK-NEXT: .long	8
; CHECK-NEXT: .byte	13
; CHECK-NEXT: .byte	7
; CHECK-NEXT: .byte	14
; CHECK-NEXT: .short	9
; CHECK-NEXT: .byte	8
; CHECK-NEXT: .byte	7
; CHECK-NEXT: .byte	9
; CHECK-NEXT: .long	8
; CHECK-NEXT: .byte	10
; CHECK-NEXT: .long	8
; CHECK-NEXT: .byte	13
; CHECK-NEXT: .byte	8
; CHECK-NEXT: .byte	14
; CHECK-NEXT: .short	9
; CHECK-NEXT: .byte	8
; CHECK-NEXT: .byte	7
; CHECK-NEXT: .byte	9
; CHECK-NEXT: .long	8
; CHECK-NEXT: .byte	10
; CHECK-NEXT: .long	8
; CHECK-NEXT: .byte	13
; CHECK-NEXT: .byte	9
; CHECK-NEXT: .byte	14
; CHECK-NEXT: .short	9
; CHECK-NEXT: .byte	8
; CHECK-NEXT: .byte	7
; CHECK-NEXT: .byte	9
; CHECK-NEXT: .long	8
; CHECK-NEXT: .byte	10
; CHECK-NEXT: .long	8
; CHECK-NEXT: .byte	13
; CHECK-NEXT: .byte	11
; CHECK-NEXT: .byte	14
; CHECK-NEXT: .short	1
; CHECK-NEXT: .byte	15
; CHECK-NEXT: .byte	1
; CHECK-NEXT: .byte	8
; CHECK-NEXT: .byte	23
; CHECK-NEXT: .long	4
; CHECK-NEXT: .ascii	"half"
; CHECK-NEXT: .byte	5

define amdgpu_kernel void @test_vec_type_hint_half(i32 %a) !kernel_arg_addr_space !1 !kernel_arg_access_qual !2 !kernel_arg_type !3 !kernel_arg_base_type !3 !kernel_arg_type_qual !4 !vec_type_hint !26 {
  ret void
}

; CHECK-LABEL:{{^}}test_vec_type_hint_float:
; CHECK: .section        .AMDGPU.runtime_metadata
; CHECK-NEXT: .byte	4
; CHECK-NEXT: .byte	6
; CHECK-NEXT: .long	24
; CHECK-NEXT: .ascii	"test_vec_type_hint_float"
; CHECK-NEXT: .byte	7
; CHECK-NEXT: .byte	9
; CHECK-NEXT: .long	4
; CHECK-NEXT: .byte	10
; CHECK-NEXT: .long	4
; CHECK-NEXT: .byte	11
; CHECK-NEXT: .long	3
; CHECK-NEXT: .ascii	"int"
; CHECK-NEXT: .byte	13
; CHECK-NEXT: .byte	0
; CHECK-NEXT: .byte	14
; CHECK-NEXT: .short	6
; CHECK-NEXT: .byte	16
; CHECK-NEXT: .byte	0
; CHECK-NEXT: .byte	8
; CHECK-NEXT: .byte	7
; CHECK-NEXT: .byte	9
; CHECK-NEXT: .long	8
; CHECK-NEXT: .byte	10
; CHECK-NEXT: .long	8
; CHECK-NEXT: .byte	13
; CHECK-NEXT: .byte	7
; CHECK-NEXT: .byte	14
; CHECK-NEXT: .short	9
; CHECK-NEXT: .byte	8
; CHECK-NEXT: .byte	7
; CHECK-NEXT: .byte	9
; CHECK-NEXT: .long	8
; CHECK-NEXT: .byte	10
; CHECK-NEXT: .long	8
; CHECK-NEXT: .byte	13
; CHECK-NEXT: .byte	8
; CHECK-NEXT: .byte	14
; CHECK-NEXT: .short	9
; CHECK-NEXT: .byte	8
; CHECK-NEXT: .byte	7
; CHECK-NEXT: .byte	9
; CHECK-NEXT: .long	8
; CHECK-NEXT: .byte	10
; CHECK-NEXT: .long	8
; CHECK-NEXT: .byte	13
; CHECK-NEXT: .byte	9
; CHECK-NEXT: .byte	14
; CHECK-NEXT: .short	9
; CHECK-NEXT: .byte	8
; CHECK-NEXT: .byte	7
; CHECK-NEXT: .byte	9
; CHECK-NEXT: .long	8
; CHECK-NEXT: .byte	10
; CHECK-NEXT: .long	8
; CHECK-NEXT: .byte	13
; CHECK-NEXT: .byte	11
; CHECK-NEXT: .byte	14
; CHECK-NEXT: .short	1
; CHECK-NEXT: .byte	15
; CHECK-NEXT: .byte	1
; CHECK-NEXT: .byte	8
; CHECK-NEXT: .byte	23
; CHECK-NEXT: .long	5
; CHECK-NEXT: .ascii	"float"
; CHECK-NEXT: .byte	5

define amdgpu_kernel void @test_vec_type_hint_float(i32 %a) !kernel_arg_addr_space !1 !kernel_arg_access_qual !2 !kernel_arg_type !3 !kernel_arg_base_type !3 !kernel_arg_type_qual !4 !vec_type_hint !27 {
  ret void
}

; CHECK-LABEL:{{^}}test_vec_type_hint_double:
; CHECK: .section        .AMDGPU.runtime_metadata
; CHECK-NEXT: .byte	4
; CHECK-NEXT: .byte	6
; CHECK-NEXT: .long	25
; CHECK-NEXT: .ascii	"test_vec_type_hint_double"
; CHECK-NEXT: .byte	7
; CHECK-NEXT: .byte	9
; CHECK-NEXT: .long	4
; CHECK-NEXT: .byte	10
; CHECK-NEXT: .long	4
; CHECK-NEXT: .byte	11
; CHECK-NEXT: .long	3
; CHECK-NEXT: .ascii	"int"
; CHECK-NEXT: .byte	13
; CHECK-NEXT: .byte	0
; CHECK-NEXT: .byte	14
; CHECK-NEXT: .short	6
; CHECK-NEXT: .byte	16
; CHECK-NEXT: .byte	0
; CHECK-NEXT: .byte	8
; CHECK-NEXT: .byte	7
; CHECK-NEXT: .byte	9
; CHECK-NEXT: .long	8
; CHECK-NEXT: .byte	10
; CHECK-NEXT: .long	8
; CHECK-NEXT: .byte	13
; CHECK-NEXT: .byte	7
; CHECK-NEXT: .byte	14
; CHECK-NEXT: .short	9
; CHECK-NEXT: .byte	8
; CHECK-NEXT: .byte	7
; CHECK-NEXT: .byte	9
; CHECK-NEXT: .long	8
; CHECK-NEXT: .byte	10
; CHECK-NEXT: .long	8
; CHECK-NEXT: .byte	13
; CHECK-NEXT: .byte	8
; CHECK-NEXT: .byte	14
; CHECK-NEXT: .short	9
; CHECK-NEXT: .byte	8
; CHECK-NEXT: .byte	7
; CHECK-NEXT: .byte	9
; CHECK-NEXT: .long	8
; CHECK-NEXT: .byte	10
; CHECK-NEXT: .long	8
; CHECK-NEXT: .byte	13
; CHECK-NEXT: .byte	9
; CHECK-NEXT: .byte	14
; CHECK-NEXT: .short	9
; CHECK-NEXT: .byte	8
; CHECK-NEXT: .byte	7
; CHECK-NEXT: .byte	9
; CHECK-NEXT: .long	8
; CHECK-NEXT: .byte	10
; CHECK-NEXT: .long	8
; CHECK-NEXT: .byte	13
; CHECK-NEXT: .byte	11
; CHECK-NEXT: .byte	14
; CHECK-NEXT: .short	1
; CHECK-NEXT: .byte	15
; CHECK-NEXT: .byte	1
; CHECK-NEXT: .byte	8
; CHECK-NEXT: .byte	23
; CHECK-NEXT: .long	6
; CHECK-NEXT: .ascii	"double"
; CHECK-NEXT: .byte	5

define amdgpu_kernel void @test_vec_type_hint_double(i32 %a) !kernel_arg_addr_space !1 !kernel_arg_access_qual !2 !kernel_arg_type !3 !kernel_arg_base_type !3 !kernel_arg_type_qual !4 !vec_type_hint !28 {
  ret void
}

; CHECK-LABEL:{{^}}test_vec_type_hint_char:
; CHECK: .section        .AMDGPU.runtime_metadata
; CHECK-NEXT: .byte	4
; CHECK-NEXT: .byte	6
; CHECK-NEXT: .long	23
; CHECK-NEXT: .ascii	"test_vec_type_hint_char"
; CHECK-NEXT: .byte	7
; CHECK-NEXT: .byte	9
; CHECK-NEXT: .long	4
; CHECK-NEXT: .byte	10
; CHECK-NEXT: .long	4
; CHECK-NEXT: .byte	11
; CHECK-NEXT: .long	3
; CHECK-NEXT: .ascii	"int"
; CHECK-NEXT: .byte	13
; CHECK-NEXT: .byte	0
; CHECK-NEXT: .byte	14
; CHECK-NEXT: .short	6
; CHECK-NEXT: .byte	16
; CHECK-NEXT: .byte	0
; CHECK-NEXT: .byte	8
; CHECK-NEXT: .byte	7
; CHECK-NEXT: .byte	9
; CHECK-NEXT: .long	8
; CHECK-NEXT: .byte	10
; CHECK-NEXT: .long	8
; CHECK-NEXT: .byte	13
; CHECK-NEXT: .byte	7
; CHECK-NEXT: .byte	14
; CHECK-NEXT: .short	9
; CHECK-NEXT: .byte	8
; CHECK-NEXT: .byte	7
; CHECK-NEXT: .byte	9
; CHECK-NEXT: .long	8
; CHECK-NEXT: .byte	10
; CHECK-NEXT: .long	8
; CHECK-NEXT: .byte	13
; CHECK-NEXT: .byte	8
; CHECK-NEXT: .byte	14
; CHECK-NEXT: .short	9
; CHECK-NEXT: .byte	8
; CHECK-NEXT: .byte	7
; CHECK-NEXT: .byte	9
; CHECK-NEXT: .long	8
; CHECK-NEXT: .byte	10
; CHECK-NEXT: .long	8
; CHECK-NEXT: .byte	13
; CHECK-NEXT: .byte	9
; CHECK-NEXT: .byte	14
; CHECK-NEXT: .short	9
; CHECK-NEXT: .byte	8
; CHECK-NEXT: .byte	7
; CHECK-NEXT: .byte	9
; CHECK-NEXT: .long	8
; CHECK-NEXT: .byte	10
; CHECK-NEXT: .long	8
; CHECK-NEXT: .byte	13
; CHECK-NEXT: .byte	11
; CHECK-NEXT: .byte	14
; CHECK-NEXT: .short	1
; CHECK-NEXT: .byte	15
; CHECK-NEXT: .byte	1
; CHECK-NEXT: .byte	8
; CHECK-NEXT: .byte	23
; CHECK-NEXT: .long	4
; CHECK-NEXT: .ascii	"char"
; CHECK-NEXT: .byte	5

define amdgpu_kernel void @test_vec_type_hint_char(i32 %a) !kernel_arg_addr_space !1 !kernel_arg_access_qual !2 !kernel_arg_type !3 !kernel_arg_base_type !3 !kernel_arg_type_qual !4 !vec_type_hint !29 {
  ret void
}

; CHECK-LABEL:{{^}}test_vec_type_hint_short:
; CHECK: .section        .AMDGPU.runtime_metadata
; CHECK-NEXT: .byte	4
; CHECK-NEXT: .byte	6
; CHECK-NEXT: .long	24
; CHECK-NEXT: .ascii	"test_vec_type_hint_short"
; CHECK-NEXT: .byte	7
; CHECK-NEXT: .byte	9
; CHECK-NEXT: .long	4
; CHECK-NEXT: .byte	10
; CHECK-NEXT: .long	4
; CHECK-NEXT: .byte	11
; CHECK-NEXT: .long	3
; CHECK-NEXT: .ascii	"int"
; CHECK-NEXT: .byte	13
; CHECK-NEXT: .byte	0
; CHECK-NEXT: .byte	14
; CHECK-NEXT: .short	6
; CHECK-NEXT: .byte	16
; CHECK-NEXT: .byte	0
; CHECK-NEXT: .byte	8
; CHECK-NEXT: .byte	7
; CHECK-NEXT: .byte	9
; CHECK-NEXT: .long	8
; CHECK-NEXT: .byte	10
; CHECK-NEXT: .long	8
; CHECK-NEXT: .byte	13
; CHECK-NEXT: .byte	7
; CHECK-NEXT: .byte	14
; CHECK-NEXT: .short	9
; CHECK-NEXT: .byte	8
; CHECK-NEXT: .byte	7
; CHECK-NEXT: .byte	9
; CHECK-NEXT: .long	8
; CHECK-NEXT: .byte	10
; CHECK-NEXT: .long	8
; CHECK-NEXT: .byte	13
; CHECK-NEXT: .byte	8
; CHECK-NEXT: .byte	14
; CHECK-NEXT: .short	9
; CHECK-NEXT: .byte	8
; CHECK-NEXT: .byte	7
; CHECK-NEXT: .byte	9
; CHECK-NEXT: .long	8
; CHECK-NEXT: .byte	10
; CHECK-NEXT: .long	8
; CHECK-NEXT: .byte	13
; CHECK-NEXT: .byte	9
; CHECK-NEXT: .byte	14
; CHECK-NEXT: .short	9
; CHECK-NEXT: .byte	8
; CHECK-NEXT: .byte	7
; CHECK-NEXT: .byte	9
; CHECK-NEXT: .long	8
; CHECK-NEXT: .byte	10
; CHECK-NEXT: .long	8
; CHECK-NEXT: .byte	13
; CHECK-NEXT: .byte	11
; CHECK-NEXT: .byte	14
; CHECK-NEXT: .short	1
; CHECK-NEXT: .byte	15
; CHECK-NEXT: .byte	1
; CHECK-NEXT: .byte	8
; CHECK-NEXT: .byte	23
; CHECK-NEXT: .long	5
; CHECK-NEXT: .ascii	"short"
; CHECK-NEXT: .byte	5

define amdgpu_kernel void @test_vec_type_hint_short(i32 %a) !kernel_arg_addr_space !1 !kernel_arg_access_qual !2 !kernel_arg_type !3 !kernel_arg_base_type !3 !kernel_arg_type_qual !4 !vec_type_hint !30 {
  ret void
}

; CHECK-LABEL:{{^}}test_vec_type_hint_long:
; CHECK: .section        .AMDGPU.runtime_metadata
; CHECK-NEXT: .byte	4
; CHECK-NEXT: .byte	6
; CHECK-NEXT: .long	23
; CHECK-NEXT: .ascii	"test_vec_type_hint_long"
; CHECK-NEXT: .byte	7
; CHECK-NEXT: .byte	9
; CHECK-NEXT: .long	4
; CHECK-NEXT: .byte	10
; CHECK-NEXT: .long	4
; CHECK-NEXT: .byte	11
; CHECK-NEXT: .long	3
; CHECK-NEXT: .ascii	"int"
; CHECK-NEXT: .byte	13
; CHECK-NEXT: .byte	0
; CHECK-NEXT: .byte	14
; CHECK-NEXT: .short	6
; CHECK-NEXT: .byte	16
; CHECK-NEXT: .byte	0
; CHECK-NEXT: .byte	8
; CHECK-NEXT: .byte	7
; CHECK-NEXT: .byte	9
; CHECK-NEXT: .long	8
; CHECK-NEXT: .byte	10
; CHECK-NEXT: .long	8
; CHECK-NEXT: .byte	13
; CHECK-NEXT: .byte	7
; CHECK-NEXT: .byte	14
; CHECK-NEXT: .short	9
; CHECK-NEXT: .byte	8
; CHECK-NEXT: .byte	7
; CHECK-NEXT: .byte	9
; CHECK-NEXT: .long	8
; CHECK-NEXT: .byte	10
; CHECK-NEXT: .long	8
; CHECK-NEXT: .byte	13
; CHECK-NEXT: .byte	8
; CHECK-NEXT: .byte	14
; CHECK-NEXT: .short	9
; CHECK-NEXT: .byte	8
; CHECK-NEXT: .byte	7
; CHECK-NEXT: .byte	9
; CHECK-NEXT: .long	8
; CHECK-NEXT: .byte	10
; CHECK-NEXT: .long	8
; CHECK-NEXT: .byte	13
; CHECK-NEXT: .byte	9
; CHECK-NEXT: .byte	14
; CHECK-NEXT: .short	9
; CHECK-NEXT: .byte	8
; CHECK-NEXT: .byte	7
; CHECK-NEXT: .byte	9
; CHECK-NEXT: .long	8
; CHECK-NEXT: .byte	10
; CHECK-NEXT: .long	8
; CHECK-NEXT: .byte	13
; CHECK-NEXT: .byte	11
; CHECK-NEXT: .byte	14
; CHECK-NEXT: .short	1
; CHECK-NEXT: .byte	15
; CHECK-NEXT: .byte	1
; CHECK-NEXT: .byte	8
; CHECK-NEXT: .byte	23
; CHECK-NEXT: .long	4
; CHECK-NEXT: .ascii	"long"
; CHECK-NEXT: .byte	5

define amdgpu_kernel void @test_vec_type_hint_long(i32 %a) !kernel_arg_addr_space !1 !kernel_arg_access_qual !2 !kernel_arg_type !3 !kernel_arg_base_type !3 !kernel_arg_type_qual !4 !vec_type_hint !31 {
  ret void
}

; CHECK-LABEL:{{^}}test_vec_type_hint_unknown:
; CHECK: .section        .AMDGPU.runtime_metadata
; CHECK-NEXT: .byte	4
; CHECK-NEXT: .byte	6
; CHECK-NEXT: .long	26
; CHECK-NEXT: .ascii	"test_vec_type_hint_unknown"
; CHECK-NEXT: .byte	7
; CHECK-NEXT: .byte	9
; CHECK-NEXT: .long	4
; CHECK-NEXT: .byte	10
; CHECK-NEXT: .long	4
; CHECK-NEXT: .byte	11
; CHECK-NEXT: .long	3
; CHECK-NEXT: .ascii	"int"
; CHECK-NEXT: .byte	13
; CHECK-NEXT: .byte	0
; CHECK-NEXT: .byte	14
; CHECK-NEXT: .short	6
; CHECK-NEXT: .byte	16
; CHECK-NEXT: .byte	0
; CHECK-NEXT: .byte	8
; CHECK-NEXT: .byte	7
; CHECK-NEXT: .byte	9
; CHECK-NEXT: .long	8
; CHECK-NEXT: .byte	10
; CHECK-NEXT: .long	8
; CHECK-NEXT: .byte	13
; CHECK-NEXT: .byte	7
; CHECK-NEXT: .byte	14
; CHECK-NEXT: .short	9
; CHECK-NEXT: .byte	8
; CHECK-NEXT: .byte	7
; CHECK-NEXT: .byte	9
; CHECK-NEXT: .long	8
; CHECK-NEXT: .byte	10
; CHECK-NEXT: .long	8
; CHECK-NEXT: .byte	13
; CHECK-NEXT: .byte	8
; CHECK-NEXT: .byte	14
; CHECK-NEXT: .short	9
; CHECK-NEXT: .byte	8
; CHECK-NEXT: .byte	7
; CHECK-NEXT: .byte	9
; CHECK-NEXT: .long	8
; CHECK-NEXT: .byte	10
; CHECK-NEXT: .long	8
; CHECK-NEXT: .byte	13
; CHECK-NEXT: .byte	9
; CHECK-NEXT: .byte	14
; CHECK-NEXT: .short	9
; CHECK-NEXT: .byte	8
; CHECK-NEXT: .byte	7
; CHECK-NEXT: .byte	9
; CHECK-NEXT: .long	8
; CHECK-NEXT: .byte	10
; CHECK-NEXT: .long	8
; CHECK-NEXT: .byte	13
; CHECK-NEXT: .byte	11
; CHECK-NEXT: .byte	14
; CHECK-NEXT: .short	1
; CHECK-NEXT: .byte	15
; CHECK-NEXT: .byte	1
; CHECK-NEXT: .byte	8
; CHECK-NEXT: .byte	23
; CHECK-NEXT: .long	7
; CHECK-NEXT: .ascii	"unknown"
; CHECK-NEXT: .byte	5

define amdgpu_kernel void @test_vec_type_hint_unknown(i32 %a) !kernel_arg_addr_space !1 !kernel_arg_access_qual !2 !kernel_arg_type !3 !kernel_arg_base_type !3 !kernel_arg_type_qual !4 !vec_type_hint !32 {
  ret void
}

; CHECK-LABEL:{{^}}test_reqd_wgs_vec_type_hint:
; CHECK: .section        .AMDGPU.runtime_metadata
; CHECK-NEXT: .byte	4
; CHECK-NEXT: .byte	6
; CHECK-NEXT: .long	27
; CHECK-NEXT: .ascii	"test_reqd_wgs_vec_type_hint"
; CHECK-NEXT: .byte	7
; CHECK-NEXT: .byte	9
; CHECK-NEXT: .long	4
; CHECK-NEXT: .byte	10
; CHECK-NEXT: .long	4
; CHECK-NEXT: .byte	11
; CHECK-NEXT: .long	3
; CHECK-NEXT: .ascii	"int"
; CHECK-NEXT: .byte	13
; CHECK-NEXT: .byte	0
; CHECK-NEXT: .byte	14
; CHECK-NEXT: .short	6
; CHECK-NEXT: .byte	16
; CHECK-NEXT: .byte	0
; CHECK-NEXT: .byte	8
; CHECK-NEXT: .byte	7
; CHECK-NEXT: .byte	9
; CHECK-NEXT: .long	8
; CHECK-NEXT: .byte	10
; CHECK-NEXT: .long	8
; CHECK-NEXT: .byte	13
; CHECK-NEXT: .byte	7
; CHECK-NEXT: .byte	14
; CHECK-NEXT: .short	9
; CHECK-NEXT: .byte	8
; CHECK-NEXT: .byte	7
; CHECK-NEXT: .byte	9
; CHECK-NEXT: .long	8
; CHECK-NEXT: .byte	10
; CHECK-NEXT: .long	8
; CHECK-NEXT: .byte	13
; CHECK-NEXT: .byte	8
; CHECK-NEXT: .byte	14
; CHECK-NEXT: .short	9
; CHECK-NEXT: .byte	8
; CHECK-NEXT: .byte	7
; CHECK-NEXT: .byte	9
; CHECK-NEXT: .long	8
; CHECK-NEXT: .byte	10
; CHECK-NEXT: .long	8
; CHECK-NEXT: .byte	13
; CHECK-NEXT: .byte	9
; CHECK-NEXT: .byte	14
; CHECK-NEXT: .short	9
; CHECK-NEXT: .byte	8
; CHECK-NEXT: .byte	7
; CHECK-NEXT: .byte	9
; CHECK-NEXT: .long	8
; CHECK-NEXT: .byte	10
; CHECK-NEXT: .long	8
; CHECK-NEXT: .byte	13
; CHECK-NEXT: .byte	11
; CHECK-NEXT: .byte	14
; CHECK-NEXT: .short	1
; CHECK-NEXT: .byte	15
; CHECK-NEXT: .byte	1
; CHECK-NEXT: .byte	8
; CHECK-NEXT: .byte	21
; CHECK-NEXT: .long	1
; CHECK-NEXT: .long	2
; CHECK-NEXT: .long	4
; CHECK-NEXT: .byte	23
; CHECK-NEXT: .long	3
; CHECK-NEXT: .ascii	"int"
; CHECK-NEXT: .byte	5

define amdgpu_kernel void @test_reqd_wgs_vec_type_hint(i32 %a) !kernel_arg_addr_space !1 !kernel_arg_access_qual !2 !kernel_arg_type !3 !kernel_arg_base_type !3 !kernel_arg_type_qual !4 !vec_type_hint !5 !reqd_work_group_size !6 {
  ret void
}

; CHECK-LABEL:{{^}}test_wgs_hint_vec_type_hint:
; CHECK: .section        .AMDGPU.runtime_metadata
; CHECK-NEXT: .byte	4
; CHECK-NEXT: .byte	6
; CHECK-NEXT: .long	27
; CHECK-NEXT: .ascii	"test_wgs_hint_vec_type_hint"
; CHECK-NEXT: .byte	7
; CHECK-NEXT: .byte	9
; CHECK-NEXT: .long	4
; CHECK-NEXT: .byte	10
; CHECK-NEXT: .long	4
; CHECK-NEXT: .byte	11
; CHECK-NEXT: .long	3
; CHECK-NEXT: .ascii	"int"
; CHECK-NEXT: .byte	13
; CHECK-NEXT: .byte	0
; CHECK-NEXT: .byte	14
; CHECK-NEXT: .short	6
; CHECK-NEXT: .byte	16
; CHECK-NEXT: .byte	0
; CHECK-NEXT: .byte	8
; CHECK-NEXT: .byte	7
; CHECK-NEXT: .byte	9
; CHECK-NEXT: .long	8
; CHECK-NEXT: .byte	10
; CHECK-NEXT: .long	8
; CHECK-NEXT: .byte	13
; CHECK-NEXT: .byte	7
; CHECK-NEXT: .byte	14
; CHECK-NEXT: .short	9
; CHECK-NEXT: .byte	8
; CHECK-NEXT: .byte	7
; CHECK-NEXT: .byte	9
; CHECK-NEXT: .long	8
; CHECK-NEXT: .byte	10
; CHECK-NEXT: .long	8
; CHECK-NEXT: .byte	13
; CHECK-NEXT: .byte	8
; CHECK-NEXT: .byte	14
; CHECK-NEXT: .short	9
; CHECK-NEXT: .byte	8
; CHECK-NEXT: .byte	7
; CHECK-NEXT: .byte	9
; CHECK-NEXT: .long	8
; CHECK-NEXT: .byte	10
; CHECK-NEXT: .long	8
; CHECK-NEXT: .byte	13
; CHECK-NEXT: .byte	9
; CHECK-NEXT: .byte	14
; CHECK-NEXT: .short	9
; CHECK-NEXT: .byte	8
; CHECK-NEXT: .byte	7
; CHECK-NEXT: .byte	9
; CHECK-NEXT: .long	8
; CHECK-NEXT: .byte	10
; CHECK-NEXT: .long	8
; CHECK-NEXT: .byte	13
; CHECK-NEXT: .byte	11
; CHECK-NEXT: .byte	14
; CHECK-NEXT: .short	1
; CHECK-NEXT: .byte	15
; CHECK-NEXT: .byte	1
; CHECK-NEXT: .byte	8
; CHECK-NEXT: .byte	22
; CHECK-NEXT: .long	8
; CHECK-NEXT: .long	16
; CHECK-NEXT: .long	32
; CHECK-NEXT: .byte	23
; CHECK-NEXT: .long	5
; CHECK-NEXT: .ascii	"uint4"
; CHECK-NEXT: .byte	5

define amdgpu_kernel void @test_wgs_hint_vec_type_hint(i32 %a) !kernel_arg_addr_space !1 !kernel_arg_access_qual !2 !kernel_arg_type !3 !kernel_arg_base_type !3 !kernel_arg_type_qual !4 !vec_type_hint !7 !work_group_size_hint !8 {
  ret void
}

; CHECK-LABEL:{{^}}test_arg_ptr_to_ptr:
; CHECK: .section        .AMDGPU.runtime_metadata
; CHECK-NEXT: .byte	4
; CHECK-NEXT: .byte	6
; CHECK-NEXT: .long	19
; CHECK-NEXT: .ascii	"test_arg_ptr_to_ptr"
; CHECK-NEXT: .byte	7
; CHECK-NEXT: .byte	9
; CHECK-NEXT: .long	8
; CHECK-NEXT: .byte	10
; CHECK-NEXT: .long	8
; CHECK-NEXT: .byte	11
; CHECK-NEXT: .long	6
; CHECK-NEXT: .ascii	"int **"
; CHECK-NEXT: .byte	13
; CHECK-NEXT: .byte	1
; CHECK-NEXT: .byte	14
; CHECK-NEXT: .short	6
; CHECK-NEXT: .byte	16
; CHECK-NEXT: .byte	0
; CHECK-NEXT: .byte	15
; CHECK-NEXT: .byte	1
; CHECK-NEXT: .byte	8
; CHECK-NEXT: .byte	7
; CHECK-NEXT: .byte	9
; CHECK-NEXT: .long	8
; CHECK-NEXT: .byte	10
; CHECK-NEXT: .long	8
; CHECK-NEXT: .byte	13
; CHECK-NEXT: .byte	7
; CHECK-NEXT: .byte	14
; CHECK-NEXT: .short	9
; CHECK-NEXT: .byte	8
; CHECK-NEXT: .byte	7
; CHECK-NEXT: .byte	9
; CHECK-NEXT: .long	8
; CHECK-NEXT: .byte	10
; CHECK-NEXT: .long	8
; CHECK-NEXT: .byte	13
; CHECK-NEXT: .byte	8
; CHECK-NEXT: .byte	14
; CHECK-NEXT: .short	9
; CHECK-NEXT: .byte	8
; CHECK-NEXT: .byte	7
; CHECK-NEXT: .byte	9
; CHECK-NEXT: .long	8
; CHECK-NEXT: .byte	10
; CHECK-NEXT: .long	8
; CHECK-NEXT: .byte	13
; CHECK-NEXT: .byte	9
; CHECK-NEXT: .byte	14
; CHECK-NEXT: .short	9
; CHECK-NEXT: .byte	8
; CHECK-NEXT: .byte	7
; CHECK-NEXT: .byte	9
; CHECK-NEXT: .long	8
; CHECK-NEXT: .byte	10
; CHECK-NEXT: .long	8
; CHECK-NEXT: .byte	13
; CHECK-NEXT: .byte	11
; CHECK-NEXT: .byte	14
; CHECK-NEXT: .short	1
; CHECK-NEXT: .byte	15
; CHECK-NEXT: .byte	1
; CHECK-NEXT: .byte	8
; CHECK-NEXT: .byte	5

define amdgpu_kernel void @test_arg_ptr_to_ptr(i32 * addrspace(1)* %a) !kernel_arg_addr_space !81 !kernel_arg_access_qual !2 !kernel_arg_type !80 !kernel_arg_base_type !80 !kernel_arg_type_qual !4 {
  ret void
}

; CHECK-LABEL:{{^}}test_arg_struct_contains_ptr:
; CHECK: .section        .AMDGPU.runtime_metadata
; CHECK-NEXT: .byte	4
; CHECK-NEXT: .byte	6
; CHECK-NEXT: .long	28
; CHECK-NEXT: .ascii	"test_arg_struct_contains_ptr"
; CHECK-NEXT: .byte	7
; CHECK-NEXT: .byte	9
; CHECK-NEXT: .long	4
; CHECK-NEXT: .byte	10
; CHECK-NEXT: .long	4
; CHECK-NEXT: .byte	11
; CHECK-NEXT: .long	8
; CHECK-NEXT: .ascii	"struct B"
; CHECK-NEXT: .byte	13
; CHECK-NEXT: .byte	1
; CHECK-NEXT: .byte	14
; CHECK-NEXT: .short	0
; CHECK-NEXT: .byte	16
; CHECK-NEXT: .byte	0
; CHECK-NEXT: .byte	15
; CHECK-NEXT: .byte	0
; CHECK-NEXT: .byte	8
; CHECK-NEXT: .byte	7
; CHECK-NEXT: .byte	9
; CHECK-NEXT: .long	8
; CHECK-NEXT: .byte	10
; CHECK-NEXT: .long	8
; CHECK-NEXT: .byte	13
; CHECK-NEXT: .byte	7
; CHECK-NEXT: .byte	14
; CHECK-NEXT: .short	9
; CHECK-NEXT: .byte	8
; CHECK-NEXT: .byte	7
; CHECK-NEXT: .byte	9
; CHECK-NEXT: .long	8
; CHECK-NEXT: .byte	10
; CHECK-NEXT: .long	8
; CHECK-NEXT: .byte	13
; CHECK-NEXT: .byte	8
; CHECK-NEXT: .byte	14
; CHECK-NEXT: .short	9
; CHECK-NEXT: .byte	8
; CHECK-NEXT: .byte	7
; CHECK-NEXT: .byte	9
; CHECK-NEXT: .long	8
; CHECK-NEXT: .byte	10
; CHECK-NEXT: .long	8
; CHECK-NEXT: .byte	13
; CHECK-NEXT: .byte	9
; CHECK-NEXT: .byte	14
; CHECK-NEXT: .short	9
; CHECK-NEXT: .byte	8
; CHECK-NEXT: .byte	7
; CHECK-NEXT: .byte	9
; CHECK-NEXT: .long	8
; CHECK-NEXT: .byte	10
; CHECK-NEXT: .long	8
; CHECK-NEXT: .byte	13
; CHECK-NEXT: .byte	11
; CHECK-NEXT: .byte	14
; CHECK-NEXT: .short	1
; CHECK-NEXT: .byte	15
; CHECK-NEXT: .byte	1
; CHECK-NEXT: .byte	8
; CHECK-NEXT: .byte	5

define amdgpu_kernel void @test_arg_struct_contains_ptr(%struct.B * byval %a) !kernel_arg_addr_space !1 !kernel_arg_access_qual !2 !kernel_arg_type !82 !kernel_arg_base_type !82 !kernel_arg_type_qual !4 {
  ret void
}

; CHECK-LABEL:{{^}}test_arg_vector_of_ptr:
; CHECK: .section        .AMDGPU.runtime_metadata
; CHECK-NEXT: .byte	4
; CHECK-NEXT: .byte	6
; CHECK-NEXT: .long	22
; CHECK-NEXT: .ascii	"test_arg_vector_of_ptr"
; CHECK-NEXT: .byte	7
; CHECK-NEXT: .byte	9
; CHECK-NEXT: .long	16
; CHECK-NEXT: .byte	10
; CHECK-NEXT: .long	16
; CHECK-NEXT: .byte	11
; CHECK-NEXT: .long	47
; CHECK-NEXT: .ascii	"global int* __attribute__((ext_vector_type(2)))"
; CHECK-NEXT: .byte	13
; CHECK-NEXT: .byte	0
; CHECK-NEXT: .byte	14
; CHECK-NEXT: .short	6
; CHECK-NEXT: .byte	16
; CHECK-NEXT: .byte	0
; CHECK-NEXT: .byte	8
; CHECK-NEXT: .byte	7
; CHECK-NEXT: .byte	9
; CHECK-NEXT: .long	8
; CHECK-NEXT: .byte	10
; CHECK-NEXT: .long	8
; CHECK-NEXT: .byte	13
; CHECK-NEXT: .byte	7
; CHECK-NEXT: .byte	14
; CHECK-NEXT: .short	9
; CHECK-NEXT: .byte	8
; CHECK-NEXT: .byte	7
; CHECK-NEXT: .byte	9
; CHECK-NEXT: .long	8
; CHECK-NEXT: .byte	10
; CHECK-NEXT: .long	8
; CHECK-NEXT: .byte	13
; CHECK-NEXT: .byte	8
; CHECK-NEXT: .byte	14
; CHECK-NEXT: .short	9
; CHECK-NEXT: .byte	8
; CHECK-NEXT: .byte	7
; CHECK-NEXT: .byte	9
; CHECK-NEXT: .long	8
; CHECK-NEXT: .byte	10
; CHECK-NEXT: .long	8
; CHECK-NEXT: .byte	13
; CHECK-NEXT: .byte	9
; CHECK-NEXT: .byte	14
; CHECK-NEXT: .short	9
; CHECK-NEXT: .byte	8
; CHECK-NEXT: .byte	7
; CHECK-NEXT: .byte	9
; CHECK-NEXT: .long	8
; CHECK-NEXT: .byte	10
; CHECK-NEXT: .long	8
; CHECK-NEXT: .byte	13
; CHECK-NEXT: .byte	11
; CHECK-NEXT: .byte	14
; CHECK-NEXT: .short	1
; CHECK-NEXT: .byte	15
; CHECK-NEXT: .byte	1
; CHECK-NEXT: .byte	8
; CHECK-NEXT: .byte	5

define amdgpu_kernel void @test_arg_vector_of_ptr(<2 x i32 addrspace(1)*> %a) !kernel_arg_addr_space !1 !kernel_arg_access_qual !2 !kernel_arg_type !83 !kernel_arg_base_type !83 !kernel_arg_type_qual !4 {
  ret void
}

; CHECK-LABEL:{{^}}test_arg_unknown_builtin_type:
; CHECK: .section        .AMDGPU.runtime_metadata
; CHECK-NEXT: .byte	4
; CHECK-NEXT: .byte	6
; CHECK-NEXT: .long	29
; CHECK-NEXT: .ascii	"test_arg_unknown_builtin_type"
; CHECK-NEXT: .byte	7
; CHECK-NEXT: .byte	9
; CHECK-NEXT: .long	8
; CHECK-NEXT: .byte	10
; CHECK-NEXT: .long	8
; CHECK-NEXT: .byte	11
; CHECK-NEXT: .long	11
; CHECK-NEXT: .ascii	"clk_event_t"
; CHECK-NEXT: .byte	13
; CHECK-NEXT: .byte	1
; CHECK-NEXT: .byte	14
; CHECK-NEXT: .short	0
; CHECK-NEXT: .byte	16
; CHECK-NEXT: .byte	0
; CHECK-NEXT: .byte	15
; CHECK-NEXT: .byte	1
; CHECK-NEXT: .byte	8
; CHECK-NEXT: .byte	7
; CHECK-NEXT: .byte	9
; CHECK-NEXT: .long	8
; CHECK-NEXT: .byte	10
; CHECK-NEXT: .long	8
; CHECK-NEXT: .byte	13
; CHECK-NEXT: .byte	7
; CHECK-NEXT: .byte	14
; CHECK-NEXT: .short	9
; CHECK-NEXT: .byte	8
; CHECK-NEXT: .byte	7
; CHECK-NEXT: .byte	9
; CHECK-NEXT: .long	8
; CHECK-NEXT: .byte	10
; CHECK-NEXT: .long	8
; CHECK-NEXT: .byte	13
; CHECK-NEXT: .byte	8
; CHECK-NEXT: .byte	14
; CHECK-NEXT: .short	9
; CHECK-NEXT: .byte	8
; CHECK-NEXT: .byte	7
; CHECK-NEXT: .byte	9
; CHECK-NEXT: .long	8
; CHECK-NEXT: .byte	10
; CHECK-NEXT: .long	8
; CHECK-NEXT: .byte	13
; CHECK-NEXT: .byte	9
; CHECK-NEXT: .byte	14
; CHECK-NEXT: .short	9
; CHECK-NEXT: .byte	8
; CHECK-NEXT: .byte	7
; CHECK-NEXT: .byte	9
; CHECK-NEXT: .long	8
; CHECK-NEXT: .byte	10
; CHECK-NEXT: .long	8
; CHECK-NEXT: .byte	13
; CHECK-NEXT: .byte	11
; CHECK-NEXT: .byte	14
; CHECK-NEXT: .short	1
; CHECK-NEXT: .byte	15
; CHECK-NEXT: .byte	1
; CHECK-NEXT: .byte	8
; CHECK-NEXT: .byte	5


define amdgpu_kernel void @test_arg_unknown_builtin_type(%opencl.clk_event_t addrspace(1)* %a) !kernel_arg_addr_space !81 !kernel_arg_access_qual !2 !kernel_arg_type !84 !kernel_arg_base_type !84 !kernel_arg_type_qual !4 {
  ret void
}

; CHECK-LABEL:{{^}}test_pointee_align:
; CHECK: .section        .AMDGPU.runtime_metadata
; CHECK-NEXT: .byte	4
; CHECK-NEXT: .byte	6
; CHECK-NEXT: .long	18
; CHECK-NEXT: .ascii	"test_pointee_align"
; CHECK-NEXT: .byte	7
; CHECK-NEXT: .byte	9
; CHECK-NEXT: .long	8
; CHECK-NEXT: .byte	10
; CHECK-NEXT: .long	8
; CHECK-NEXT: .byte	11
; CHECK-NEXT: .long	6
; CHECK-NEXT: .ascii	"long *"
; CHECK-NEXT: .byte	13
; CHECK-NEXT: .byte	1
; CHECK-NEXT: .byte	14
; CHECK-NEXT: .short	9
; CHECK-NEXT: .byte	16
; CHECK-NEXT: .byte	0
; CHECK-NEXT: .byte	15
; CHECK-NEXT: .byte	1
; CHECK-NEXT: .byte	8
; CHECK-NEXT: .byte	7
; CHECK-NEXT: .byte	9
; CHECK-NEXT: .long	4
; CHECK-NEXT: .byte	10
; CHECK-NEXT: .long	4
; CHECK-NEXT: .byte	32
; CHECK-NEXT: .long	1
; CHECK-NEXT: .byte	11
; CHECK-NEXT: .long	6
; CHECK-NEXT: .ascii	"char *"
; CHECK-NEXT: .byte	13
; CHECK-NEXT: .byte	2
; CHECK-NEXT: .byte	14
; CHECK-NEXT: .short	1
; CHECK-NEXT: .byte	16
; CHECK-NEXT: .byte	0
; CHECK-NEXT: .byte	15
; CHECK-NEXT: .byte	3
; CHECK-NEXT: .byte	8
; CHECK-NEXT: .byte	7
; CHECK-NEXT: .byte	9
; CHECK-NEXT: .long	4
; CHECK-NEXT: .byte	10
; CHECK-NEXT: .long	4
; CHECK-NEXT: .byte	32
; CHECK-NEXT: .long	2
; CHECK-NEXT: .byte	11
; CHECK-NEXT: .long	7
; CHECK-NEXT: .ascii	"char2 *"
; CHECK-NEXT: .byte	13
; CHECK-NEXT: .byte	2
; CHECK-NEXT: .byte	14
; CHECK-NEXT: .short	1
; CHECK-NEXT: .byte	16
; CHECK-NEXT: .byte	0
; CHECK-NEXT: .byte	15
; CHECK-NEXT: .byte	3
; CHECK-NEXT: .byte	8
; CHECK-NEXT: .byte	7
; CHECK-NEXT: .byte	9
; CHECK-NEXT: .long	4
; CHECK-NEXT: .byte	10
; CHECK-NEXT: .long	4
; CHECK-NEXT: .byte	32
; CHECK-NEXT: .long	4
; CHECK-NEXT: .byte	11
; CHECK-NEXT: .long	7
; CHECK-NEXT: .ascii	"char3 *"
; CHECK-NEXT: .byte	13
; CHECK-NEXT: .byte	2
; CHECK-NEXT: .byte	14
; CHECK-NEXT: .short	1
; CHECK-NEXT: .byte	16
; CHECK-NEXT: .byte	0
; CHECK-NEXT: .byte	15
; CHECK-NEXT: .byte	3
; CHECK-NEXT: .byte	8
; CHECK-NEXT: .byte	7
; CHECK-NEXT: .byte	9
; CHECK-NEXT: .long	4
; CHECK-NEXT: .byte	10
; CHECK-NEXT: .long	4
; CHECK-NEXT: .byte	32
; CHECK-NEXT: .long	4
; CHECK-NEXT: .byte	11
; CHECK-NEXT: .long	7
; CHECK-NEXT: .ascii	"char4 *"
; CHECK-NEXT: .byte	13
; CHECK-NEXT: .byte	2
; CHECK-NEXT: .byte	14
; CHECK-NEXT: .short	1
; CHECK-NEXT: .byte	16
; CHECK-NEXT: .byte	0
; CHECK-NEXT: .byte	15
; CHECK-NEXT: .byte	3
; CHECK-NEXT: .byte	8
; CHECK-NEXT: .byte	7
; CHECK-NEXT: .byte	9
; CHECK-NEXT: .long	4
; CHECK-NEXT: .byte	10
; CHECK-NEXT: .long	4
; CHECK-NEXT: .byte	32
; CHECK-NEXT: .long	8
; CHECK-NEXT: .byte	11
; CHECK-NEXT: .long	7
; CHECK-NEXT: .ascii	"char8 *"
; CHECK-NEXT: .byte	13
; CHECK-NEXT: .byte	2
; CHECK-NEXT: .byte	14
; CHECK-NEXT: .short	1
; CHECK-NEXT: .byte	16
; CHECK-NEXT: .byte	0
; CHECK-NEXT: .byte	15
; CHECK-NEXT: .byte	3
; CHECK-NEXT: .byte	8
; CHECK-NEXT: .byte	7
; CHECK-NEXT: .byte	9
; CHECK-NEXT: .long	4
; CHECK-NEXT: .byte	10
; CHECK-NEXT: .long	4
; CHECK-NEXT: .byte	32
; CHECK-NEXT: .long	16
; CHECK-NEXT: .byte	11
; CHECK-NEXT: .long	8
; CHECK-NEXT: .ascii	"char16 *"
; CHECK-NEXT: .byte	13
; CHECK-NEXT: .byte	2
; CHECK-NEXT: .byte	14
; CHECK-NEXT: .short	1
; CHECK-NEXT: .byte	16
; CHECK-NEXT: .byte	0
; CHECK-NEXT: .byte	15
; CHECK-NEXT: .byte	3
; CHECK-NEXT: .byte	8
; CHECK-NEXT: .byte	7
; CHECK-NEXT: .byte	9
; CHECK-NEXT: .long	8
; CHECK-NEXT: .byte	10
; CHECK-NEXT: .long	8
; CHECK-NEXT: .byte	13
; CHECK-NEXT: .byte	7
; CHECK-NEXT: .byte	14
; CHECK-NEXT: .short	9
; CHECK-NEXT: .byte	8
; CHECK-NEXT: .byte	7
; CHECK-NEXT: .byte	9
; CHECK-NEXT: .long	8
; CHECK-NEXT: .byte	10
; CHECK-NEXT: .long	8
; CHECK-NEXT: .byte	13
; CHECK-NEXT: .byte	8
; CHECK-NEXT: .byte	14
; CHECK-NEXT: .short	9
; CHECK-NEXT: .byte	8
; CHECK-NEXT: .byte	7
; CHECK-NEXT: .byte	9
; CHECK-NEXT: .long	8
; CHECK-NEXT: .byte	10
; CHECK-NEXT: .long	8
; CHECK-NEXT: .byte	13
; CHECK-NEXT: .byte	9
; CHECK-NEXT: .byte	14
; CHECK-NEXT: .short	9
; CHECK-NEXT: .byte	8
; CHECK-NEXT: .byte	7
; CHECK-NEXT: .byte	9
; CHECK-NEXT: .long	8
; CHECK-NEXT: .byte	10
; CHECK-NEXT: .long	8
; CHECK-NEXT: .byte	13
; CHECK-NEXT: .byte	11
; CHECK-NEXT: .byte	14
; CHECK-NEXT: .short	1
; CHECK-NEXT: .byte	15
; CHECK-NEXT: .byte	1
; CHECK-NEXT: .byte	8
; CHECK-NEXT: .byte	5

define amdgpu_kernel void @test_pointee_align(i64 addrspace(1)* %a, i8 addrspace(3)* %b, <2 x i8> addrspace(3)* %c, <3 x i8> addrspace(3)* %d, <4 x i8> addrspace(3)* %e, <8 x i8> addrspace(3)* %f, <16 x i8> addrspace(3)* %g) !kernel_arg_addr_space !91 !kernel_arg_access_qual !92 !kernel_arg_type !93 !kernel_arg_base_type !93 !kernel_arg_type_qual !94 {
  ret void
}

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
!16 = !{!"int *"}
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
!32 = !{i32 *undef, i32 1}
!50 = !{i32 1, i32 2, i32 3}
!51 = !{!"int *", !"int *", !"int *"}
!60 = !{i32 1, i32 1, i32 1}
!61 = !{!"read_only", !"write_only", !"read_write"}
!62 = !{!"image1d_t", !"image2d_t", !"image3d_t"}
!70 = !{!"volatile", !"const restrict", !"pipe"}
!80 = !{!"int **"}
!81 = !{i32 1}
!82 = !{!"struct B"}
!83 = !{!"global int* __attribute__((ext_vector_type(2)))"}
!84 = !{!"clk_event_t"}
!opencl.ocl.version = !{!90}
!90 = !{i32 2, i32 0}
!91 = !{i32 0, i32 3, i32 3, i32 3, i32 3, i32 3, i32 3}
!92 = !{!"none", !"none", !"none", !"none", !"none", !"none", !"none"}
!93 = !{!"long *", !"char *", !"char2 *", !"char3 *", !"char4 *", !"char8 *", !"char16 *"}
!94 = !{!"", !"", !"", !"", !"", !"", !""}
!100 = !{!"1:1:4:%d\5Cn"}
!101 = !{!"2:1:8:%g\5Cn"}

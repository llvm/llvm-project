; RUN: llc -mtriple=amdgcn--amdhsa -filetype=obj -o - < %s | llvm-readobj -amdgpu-runtime-metadata -elf-output-style=GNU -notes | FileCheck %s --check-prefix=NOTES --check-prefix=SI
; RUN: llc -mtriple=amdgcn--amdhsa -mcpu=fiji -filetype=obj -o - < %s | llvm-readobj -amdgpu-runtime-metadata -elf-output-style=GNU -notes | FileCheck %s --check-prefix=NOTES --check-prefix=VI
; RUN: llc -mtriple=amdgcn--amdhsa -filetype=obj -amdgpu-dump-rtmd -amdgpu-check-rtmd-parser %s -o - 2>&1 | FileCheck --check-prefix=CHECK --check-prefix=PARSER %s

%struct.A = type { i8, float }
%opencl.image1d_t = type opaque
%opencl.image2d_t = type opaque
%opencl.image3d_t = type opaque
%opencl.queue_t = type opaque
%opencl.pipe_t = type opaque
%struct.B = type { i32 addrspace(1)*}
%opencl.clk_event_t = type opaque

; CHECK: ---
; SI: { amd.MDVersion: [ 2, 1 ], amd.IsaInfo: { amd.IsaInfoWavefrontSize: 64, amd.IsaInfoLocalMemorySize: 65536, amd.IsaInfoEUsPerCU: 4, amd.IsaInfoMaxWavesPerEU: 10, amd.IsaInfoMaxFlatWorkGroupSize: 2048, amd.IsaInfoSGPRAllocGranule: 8, amd.IsaInfoTotalNumSGPRs: 512, amd.IsaInfoAddressableNumSGPRs: 104, amd.IsaInfoVGPRAllocGranule: 4, amd.IsaInfoTotalNumVGPRs: 256, amd.IsaInfoAddressableNumVGPRs: 256 }, amd.PrintfInfo: [ '1:1:4:%d\n', '2:1:8:%g\n' ], amd.Kernels: 
; VI: { amd.MDVersion: [ 2, 1 ], amd.IsaInfo: { amd.IsaInfoWavefrontSize: 64, amd.IsaInfoLocalMemorySize: 65536, amd.IsaInfoEUsPerCU: 4, amd.IsaInfoMaxWavesPerEU: 10, amd.IsaInfoMaxFlatWorkGroupSize: 2048, amd.IsaInfoSGPRAllocGranule: 16, amd.IsaInfoTotalNumSGPRs: 800, amd.IsaInfoAddressableNumSGPRs: 102, amd.IsaInfoVGPRAllocGranule: 4, amd.IsaInfoTotalNumVGPRs: 256, amd.IsaInfoAddressableNumVGPRs: 256 }, amd.PrintfInfo: [ '1:1:4:%d\n', '2:1:8:%g\n' ], amd.Kernels: 

; CHECK:   - { amd.KernelName: test_char, amd.Language: OpenCL C, amd.LanguageVersion: [ 2, 0 ], amd.Args: 
; CHECK-NEXT:       - { amd.ArgSize: 1, amd.ArgAlign: 1, amd.ArgKind: 0, amd.ArgValueType: 1, amd.ArgTypeName: char, amd.ArgAccQual: 0 }
; CHECK-NEXT:       - { amd.ArgSize: 8, amd.ArgAlign: 8, amd.ArgKind: 7, amd.ArgValueType: 9 }
; CHECK-NEXT:       - { amd.ArgSize: 8, amd.ArgAlign: 8, amd.ArgKind: 8, amd.ArgValueType: 9 }
; CHECK-NEXT:       - { amd.ArgSize: 8, amd.ArgAlign: 8, amd.ArgKind: 9, amd.ArgValueType: 9 }
; CHECK-NEXT:       - { amd.ArgSize: 8, amd.ArgAlign: 8, amd.ArgKind: 11, amd.ArgValueType: 1, amd.ArgAddrQual: 1 } }
define amdgpu_kernel void @test_char(i8 %a) !kernel_arg_addr_space !1 !kernel_arg_access_qual !2 !kernel_arg_type !9 !kernel_arg_base_type !9 !kernel_arg_type_qual !4 {
  ret void
}

; CHECK-NEXT:   - { amd.KernelName: test_ushort2, amd.Language: OpenCL C, amd.LanguageVersion: [ 2, 0 ], amd.Args: 
; CHECK-NEXT:       - { amd.ArgSize: 4, amd.ArgAlign: 4, amd.ArgKind: 0, amd.ArgValueType: 4, amd.ArgTypeName: ushort2, amd.ArgAccQual: 0 }
; CHECK-NEXT:       - { amd.ArgSize: 8, amd.ArgAlign: 8, amd.ArgKind: 7, amd.ArgValueType: 9 }
; CHECK-NEXT:       - { amd.ArgSize: 8, amd.ArgAlign: 8, amd.ArgKind: 8, amd.ArgValueType: 9 }
; CHECK-NEXT:       - { amd.ArgSize: 8, amd.ArgAlign: 8, amd.ArgKind: 9, amd.ArgValueType: 9 }
; CHECK-NEXT:       - { amd.ArgSize: 8, amd.ArgAlign: 8, amd.ArgKind: 11, amd.ArgValueType: 1, amd.ArgAddrQual: 1 } }
define amdgpu_kernel void @test_ushort2(<2 x i16> %a) !kernel_arg_addr_space !1 !kernel_arg_access_qual !2 !kernel_arg_type !10 !kernel_arg_base_type !10 !kernel_arg_type_qual !4 {
  ret void
}

; CHECK-NEXT:   - { amd.KernelName: test_int3, amd.Language: OpenCL C, amd.LanguageVersion: [ 2, 0 ], amd.Args: 
; CHECK-NEXT:       - { amd.ArgSize: 16, amd.ArgAlign: 16, amd.ArgKind: 0, amd.ArgValueType: 6, amd.ArgTypeName: int3, amd.ArgAccQual: 0 }
; CHECK-NEXT:       - { amd.ArgSize: 8, amd.ArgAlign: 8, amd.ArgKind: 7, amd.ArgValueType: 9 }
; CHECK-NEXT:       - { amd.ArgSize: 8, amd.ArgAlign: 8, amd.ArgKind: 8, amd.ArgValueType: 9 }
; CHECK-NEXT:       - { amd.ArgSize: 8, amd.ArgAlign: 8, amd.ArgKind: 9, amd.ArgValueType: 9 }
; CHECK-NEXT:       - { amd.ArgSize: 8, amd.ArgAlign: 8, amd.ArgKind: 11, amd.ArgValueType: 1, amd.ArgAddrQual: 1 } }
define amdgpu_kernel void @test_int3(<3 x i32> %a) !kernel_arg_addr_space !1 !kernel_arg_access_qual !2 !kernel_arg_type !11 !kernel_arg_base_type !11 !kernel_arg_type_qual !4 {
  ret void
}

; CHECK-NEXT:   - { amd.KernelName: test_ulong4, amd.Language: OpenCL C, amd.LanguageVersion: [ 2, 0 ], amd.Args: 
; CHECK-NEXT:       - { amd.ArgSize: 32, amd.ArgAlign: 32, amd.ArgKind: 0, amd.ArgValueType: 10, amd.ArgTypeName: ulong4, amd.ArgAccQual: 0 }
; CHECK-NEXT:       - { amd.ArgSize: 8, amd.ArgAlign: 8, amd.ArgKind: 7, amd.ArgValueType: 9 }
; CHECK-NEXT:       - { amd.ArgSize: 8, amd.ArgAlign: 8, amd.ArgKind: 8, amd.ArgValueType: 9 }
; CHECK-NEXT:       - { amd.ArgSize: 8, amd.ArgAlign: 8, amd.ArgKind: 9, amd.ArgValueType: 9 }
; CHECK-NEXT:       - { amd.ArgSize: 8, amd.ArgAlign: 8, amd.ArgKind: 11, amd.ArgValueType: 1, amd.ArgAddrQual: 1 } }
define amdgpu_kernel void @test_ulong4(<4 x i64> %a) !kernel_arg_addr_space !1 !kernel_arg_access_qual !2 !kernel_arg_type !12 !kernel_arg_base_type !12 !kernel_arg_type_qual !4 {
  ret void
}

; CHECK-NEXT:   - { amd.KernelName: test_half8, amd.Language: OpenCL C, amd.LanguageVersion: [ 2, 0 ], amd.Args: 
; CHECK-NEXT:       - { amd.ArgSize: 16, amd.ArgAlign: 16, amd.ArgKind: 0, amd.ArgValueType: 5, amd.ArgTypeName: half8, amd.ArgAccQual: 0 }
; CHECK-NEXT:       - { amd.ArgSize: 8, amd.ArgAlign: 8, amd.ArgKind: 7, amd.ArgValueType: 9 }
; CHECK-NEXT:       - { amd.ArgSize: 8, amd.ArgAlign: 8, amd.ArgKind: 8, amd.ArgValueType: 9 }
; CHECK-NEXT:       - { amd.ArgSize: 8, amd.ArgAlign: 8, amd.ArgKind: 9, amd.ArgValueType: 9 }
; CHECK-NEXT:       - { amd.ArgSize: 8, amd.ArgAlign: 8, amd.ArgKind: 11, amd.ArgValueType: 1, amd.ArgAddrQual: 1 } }
define amdgpu_kernel void @test_half8(<8 x half> %a) !kernel_arg_addr_space !1 !kernel_arg_access_qual !2 !kernel_arg_type !13 !kernel_arg_base_type !13 !kernel_arg_type_qual !4 {
  ret void
}

; CHECK-NEXT:   - { amd.KernelName: test_float16, amd.Language: OpenCL C, amd.LanguageVersion: [ 2, 0 ], amd.Args: 
; CHECK-NEXT:       - { amd.ArgSize: 64, amd.ArgAlign: 64, amd.ArgKind: 0, amd.ArgValueType: 8, amd.ArgTypeName: float16, amd.ArgAccQual: 0 }
; CHECK-NEXT:       - { amd.ArgSize: 8, amd.ArgAlign: 8, amd.ArgKind: 7, amd.ArgValueType: 9 }
; CHECK-NEXT:       - { amd.ArgSize: 8, amd.ArgAlign: 8, amd.ArgKind: 8, amd.ArgValueType: 9 }
; CHECK-NEXT:       - { amd.ArgSize: 8, amd.ArgAlign: 8, amd.ArgKind: 9, amd.ArgValueType: 9 }
; CHECK-NEXT:       - { amd.ArgSize: 8, amd.ArgAlign: 8, amd.ArgKind: 11, amd.ArgValueType: 1, amd.ArgAddrQual: 1 } }
define amdgpu_kernel void @test_float16(<16 x float> %a) !kernel_arg_addr_space !1 !kernel_arg_access_qual !2 !kernel_arg_type !14 !kernel_arg_base_type !14 !kernel_arg_type_qual !4 {
  ret void
}

; CHECK-NEXT:   - { amd.KernelName: test_double16, amd.Language: OpenCL C, amd.LanguageVersion: [ 2, 0 ], amd.Args: 
; CHECK-NEXT:       - { amd.ArgSize: 128, amd.ArgAlign: 128, amd.ArgKind: 0, amd.ArgValueType: 11, amd.ArgTypeName: double16, amd.ArgAccQual: 0 }
; CHECK-NEXT:       - { amd.ArgSize: 8, amd.ArgAlign: 8, amd.ArgKind: 7, amd.ArgValueType: 9 }
; CHECK-NEXT:       - { amd.ArgSize: 8, amd.ArgAlign: 8, amd.ArgKind: 8, amd.ArgValueType: 9 }
; CHECK-NEXT:       - { amd.ArgSize: 8, amd.ArgAlign: 8, amd.ArgKind: 9, amd.ArgValueType: 9 }
; CHECK-NEXT:       - { amd.ArgSize: 8, amd.ArgAlign: 8, amd.ArgKind: 11, amd.ArgValueType: 1, amd.ArgAddrQual: 1 } }
define amdgpu_kernel void @test_double16(<16 x double> %a) !kernel_arg_addr_space !1 !kernel_arg_access_qual !2 !kernel_arg_type !15 !kernel_arg_base_type !15 !kernel_arg_type_qual !4 {
  ret void
}

; CHECK-NEXT:   - { amd.KernelName: test_pointer, amd.Language: OpenCL C, amd.LanguageVersion: [ 2, 0 ], amd.Args: 
; CHECK-NEXT:       - { amd.ArgSize: 8, amd.ArgAlign: 8, amd.ArgKind: 1, amd.ArgValueType: 6, amd.ArgTypeName: 'int *', amd.ArgAddrQual: 1, amd.ArgAccQual: 0 }
; CHECK-NEXT:       - { amd.ArgSize: 8, amd.ArgAlign: 8, amd.ArgKind: 7, amd.ArgValueType: 9 }
; CHECK-NEXT:       - { amd.ArgSize: 8, amd.ArgAlign: 8, amd.ArgKind: 8, amd.ArgValueType: 9 }
; CHECK-NEXT:       - { amd.ArgSize: 8, amd.ArgAlign: 8, amd.ArgKind: 9, amd.ArgValueType: 9 }
; CHECK-NEXT:       - { amd.ArgSize: 8, amd.ArgAlign: 8, amd.ArgKind: 11, amd.ArgValueType: 1, amd.ArgAddrQual: 1 } }
define amdgpu_kernel void @test_pointer(i32 addrspace(1)* %a) !kernel_arg_addr_space !1 !kernel_arg_access_qual !2 !kernel_arg_type !16 !kernel_arg_base_type !16 !kernel_arg_type_qual !4 {
  ret void
}

; CHECK-NEXT:   - { amd.KernelName: test_image, amd.Language: OpenCL C, amd.LanguageVersion: [ 2, 0 ], amd.Args: 
; CHECK-NEXT:       - { amd.ArgSize: 8, amd.ArgAlign: 8, amd.ArgKind: 4, amd.ArgValueType: 0, amd.ArgTypeName: image2d_t, amd.ArgAddrQual: 1, amd.ArgAccQual: 0 }
; CHECK-NEXT:       - { amd.ArgSize: 8, amd.ArgAlign: 8, amd.ArgKind: 7, amd.ArgValueType: 9 }
; CHECK-NEXT:       - { amd.ArgSize: 8, amd.ArgAlign: 8, amd.ArgKind: 8, amd.ArgValueType: 9 }
; CHECK-NEXT:       - { amd.ArgSize: 8, amd.ArgAlign: 8, amd.ArgKind: 9, amd.ArgValueType: 9 }
; CHECK-NEXT:       - { amd.ArgSize: 8, amd.ArgAlign: 8, amd.ArgKind: 11, amd.ArgValueType: 1, amd.ArgAddrQual: 1 } }
define amdgpu_kernel void @test_image(%opencl.image2d_t addrspace(1)* %a) !kernel_arg_addr_space !1 !kernel_arg_access_qual !2 !kernel_arg_type !17 !kernel_arg_base_type !17 !kernel_arg_type_qual !4 {
  ret void
}

; CHECK-NEXT:   - { amd.KernelName: test_sampler, amd.Language: OpenCL C, amd.LanguageVersion: [ 2, 0 ], amd.Args: 
; CHECK-NEXT:       - { amd.ArgSize: 4, amd.ArgAlign: 4, amd.ArgKind: 3, amd.ArgValueType: 6, amd.ArgTypeName: sampler_t, amd.ArgAccQual: 0 }
; CHECK-NEXT:       - { amd.ArgSize: 8, amd.ArgAlign: 8, amd.ArgKind: 7, amd.ArgValueType: 9 }
; CHECK-NEXT:       - { amd.ArgSize: 8, amd.ArgAlign: 8, amd.ArgKind: 8, amd.ArgValueType: 9 }
; CHECK-NEXT:       - { amd.ArgSize: 8, amd.ArgAlign: 8, amd.ArgKind: 9, amd.ArgValueType: 9 }
; CHECK-NEXT:       - { amd.ArgSize: 8, amd.ArgAlign: 8, amd.ArgKind: 11, amd.ArgValueType: 1, amd.ArgAddrQual: 1 } }
define amdgpu_kernel void @test_sampler(i32 %a) !kernel_arg_addr_space !1 !kernel_arg_access_qual !2 !kernel_arg_type !18 !kernel_arg_base_type !18 !kernel_arg_type_qual !4 {
  ret void
}

; CHECK-NEXT:   - { amd.KernelName: test_queue, amd.Language: OpenCL C, amd.LanguageVersion: [ 2, 0 ], amd.Args: 
; CHECK-NEXT:       - { amd.ArgSize: 8, amd.ArgAlign: 8, amd.ArgKind: 6, amd.ArgValueType: 0, amd.ArgTypeName: queue_t, amd.ArgAddrQual: 1, amd.ArgAccQual: 0 }
; CHECK-NEXT:       - { amd.ArgSize: 8, amd.ArgAlign: 8, amd.ArgKind: 7, amd.ArgValueType: 9 }
; CHECK-NEXT:       - { amd.ArgSize: 8, amd.ArgAlign: 8, amd.ArgKind: 8, amd.ArgValueType: 9 }
; CHECK-NEXT:       - { amd.ArgSize: 8, amd.ArgAlign: 8, amd.ArgKind: 9, amd.ArgValueType: 9 }
; CHECK-NEXT:       - { amd.ArgSize: 8, amd.ArgAlign: 8, amd.ArgKind: 11, amd.ArgValueType: 1, amd.ArgAddrQual: 1 } }
define amdgpu_kernel void @test_queue(%opencl.queue_t addrspace(1)* %a) !kernel_arg_addr_space !1 !kernel_arg_access_qual !2 !kernel_arg_type !19 !kernel_arg_base_type !19 !kernel_arg_type_qual !4 {
  ret void
}

; CHECK-NEXT:   - { amd.KernelName: test_struct, amd.Language: OpenCL C, amd.LanguageVersion: [ 2, 0 ], amd.Args: 
; CHECK-NEXT:       - { amd.ArgSize: 4, amd.ArgAlign: 4, amd.ArgKind: 1, amd.ArgValueType: 0, amd.ArgTypeName: struct A, amd.ArgAddrQual: 0, amd.ArgAccQual: 0 }
; CHECK-NEXT:       - { amd.ArgSize: 8, amd.ArgAlign: 8, amd.ArgKind: 7, amd.ArgValueType: 9 }
; CHECK-NEXT:       - { amd.ArgSize: 8, amd.ArgAlign: 8, amd.ArgKind: 8, amd.ArgValueType: 9 }
; CHECK-NEXT:       - { amd.ArgSize: 8, amd.ArgAlign: 8, amd.ArgKind: 9, amd.ArgValueType: 9 }
; CHECK-NEXT:       - { amd.ArgSize: 8, amd.ArgAlign: 8, amd.ArgKind: 11, amd.ArgValueType: 1, amd.ArgAddrQual: 1 } }
define amdgpu_kernel void @test_struct(%struct.A* byval %a) !kernel_arg_addr_space !1 !kernel_arg_access_qual !2 !kernel_arg_type !20 !kernel_arg_base_type !20 !kernel_arg_type_qual !4 {
  ret void
}

; CHECK-NEXT:   - { amd.KernelName: test_i128, amd.Language: OpenCL C, amd.LanguageVersion: [ 2, 0 ], amd.Args: 
; CHECK-NEXT:       - { amd.ArgSize: 16, amd.ArgAlign: 8, amd.ArgKind: 0, amd.ArgValueType: 0, amd.ArgTypeName: i128, amd.ArgAccQual: 0 }
; CHECK-NEXT:       - { amd.ArgSize: 8, amd.ArgAlign: 8, amd.ArgKind: 7, amd.ArgValueType: 9 }
; CHECK-NEXT:       - { amd.ArgSize: 8, amd.ArgAlign: 8, amd.ArgKind: 8, amd.ArgValueType: 9 }
; CHECK-NEXT:       - { amd.ArgSize: 8, amd.ArgAlign: 8, amd.ArgKind: 9, amd.ArgValueType: 9 }
; CHECK-NEXT:       - { amd.ArgSize: 8, amd.ArgAlign: 8, amd.ArgKind: 11, amd.ArgValueType: 1, amd.ArgAddrQual: 1 } }
define amdgpu_kernel void @test_i128(i128 %a) !kernel_arg_addr_space !1 !kernel_arg_access_qual !2 !kernel_arg_type !21 !kernel_arg_base_type !21 !kernel_arg_type_qual !4 {
  ret void
}

; CHECK-NEXT:   - { amd.KernelName: test_multi_arg, amd.Language: OpenCL C, amd.LanguageVersion: [ 2, 0 ], amd.Args: 
; CHECK-NEXT:       - { amd.ArgSize: 4, amd.ArgAlign: 4, amd.ArgKind: 0, amd.ArgValueType: 6, amd.ArgTypeName: int, amd.ArgAccQual: 0 }
; CHECK-NEXT:       - { amd.ArgSize: 4, amd.ArgAlign: 4, amd.ArgKind: 0, amd.ArgValueType: 3, amd.ArgTypeName: short2, amd.ArgAccQual: 0 }
; CHECK-NEXT:       - { amd.ArgSize: 4, amd.ArgAlign: 4, amd.ArgKind: 0, amd.ArgValueType: 1, amd.ArgTypeName: char3, amd.ArgAccQual: 0 }
; CHECK-NEXT:       - { amd.ArgSize: 8, amd.ArgAlign: 8, amd.ArgKind: 7, amd.ArgValueType: 9 }
; CHECK-NEXT:       - { amd.ArgSize: 8, amd.ArgAlign: 8, amd.ArgKind: 8, amd.ArgValueType: 9 }
; CHECK-NEXT:       - { amd.ArgSize: 8, amd.ArgAlign: 8, amd.ArgKind: 9, amd.ArgValueType: 9 }
; CHECK-NEXT:       - { amd.ArgSize: 8, amd.ArgAlign: 8, amd.ArgKind: 11, amd.ArgValueType: 1, amd.ArgAddrQual: 1 } }
define amdgpu_kernel void @test_multi_arg(i32 %a, <2 x i16> %b, <3 x i8> %c) !kernel_arg_addr_space !22 !kernel_arg_access_qual !23 !kernel_arg_type !24 !kernel_arg_base_type !24 !kernel_arg_type_qual !25 {
  ret void
}

; CHECK-NEXT:   - { amd.KernelName: test_addr_space, amd.Language: OpenCL C, amd.LanguageVersion: [ 2, 0 ], amd.Args: 
; CHECK-NEXT:       - { amd.ArgSize: 8, amd.ArgAlign: 8, amd.ArgKind: 1, amd.ArgValueType: 6, amd.ArgTypeName: 'int *', amd.ArgAddrQual: 1, amd.ArgAccQual: 0 }
; CHECK-NEXT:       - { amd.ArgSize: 8, amd.ArgAlign: 8, amd.ArgKind: 1, amd.ArgValueType: 6, amd.ArgTypeName: 'int *', amd.ArgAddrQual: 2, amd.ArgAccQual: 0 }
; CHECK-NEXT:       - { amd.ArgSize: 4, amd.ArgAlign: 4, amd.ArgPointeeAlign: 4, amd.ArgKind: 2, amd.ArgValueType: 6, amd.ArgTypeName: 'int *', amd.ArgAddrQual: 3, amd.ArgAccQual: 0 }
; CHECK-NEXT:       - { amd.ArgSize: 8, amd.ArgAlign: 8, amd.ArgKind: 7, amd.ArgValueType: 9 }
; CHECK-NEXT:       - { amd.ArgSize: 8, amd.ArgAlign: 8, amd.ArgKind: 8, amd.ArgValueType: 9 }
; CHECK-NEXT:       - { amd.ArgSize: 8, amd.ArgAlign: 8, amd.ArgKind: 9, amd.ArgValueType: 9 }
; CHECK-NEXT:       - { amd.ArgSize: 8, amd.ArgAlign: 8, amd.ArgKind: 11, amd.ArgValueType: 1, amd.ArgAddrQual: 1 } }
define amdgpu_kernel void @test_addr_space(i32 addrspace(1)* %g, i32 addrspace(2)* %c, i32 addrspace(3)* %l) !kernel_arg_addr_space !50 !kernel_arg_access_qual !23 !kernel_arg_type !51 !kernel_arg_base_type !51 !kernel_arg_type_qual !25 {
  ret void
}

; CHECK-NEXT:   - { amd.KernelName: test_type_qual, amd.Language: OpenCL C, amd.LanguageVersion: [ 2, 0 ], amd.Args: 
; CHECK-NEXT:       - { amd.ArgSize: 8, amd.ArgAlign: 8, amd.ArgKind: 1, amd.ArgValueType: 6, amd.ArgTypeName: 'int *', amd.ArgAddrQual: 1, amd.ArgAccQual: 0, amd.ArgIsVolatile: 1 }
; CHECK-NEXT:       - { amd.ArgSize: 8, amd.ArgAlign: 8, amd.ArgKind: 1, amd.ArgValueType: 6, amd.ArgTypeName: 'int *', amd.ArgAddrQual: 1, amd.ArgAccQual: 0, amd.ArgIsConst: 1, amd.ArgIsRestrict: 1 }
; CHECK-NEXT:       - { amd.ArgSize: 8, amd.ArgAlign: 8, amd.ArgKind: 5, amd.ArgValueType: 0, amd.ArgTypeName: 'int *', amd.ArgAddrQual: 1, amd.ArgAccQual: 0, amd.ArgIsPipe: 1 }
; CHECK-NEXT:       - { amd.ArgSize: 8, amd.ArgAlign: 8, amd.ArgKind: 7, amd.ArgValueType: 9 }
; CHECK-NEXT:       - { amd.ArgSize: 8, amd.ArgAlign: 8, amd.ArgKind: 8, amd.ArgValueType: 9 }
; CHECK-NEXT:       - { amd.ArgSize: 8, amd.ArgAlign: 8, amd.ArgKind: 9, amd.ArgValueType: 9 }
; CHECK-NEXT:       - { amd.ArgSize: 8, amd.ArgAlign: 8, amd.ArgKind: 11, amd.ArgValueType: 1, amd.ArgAddrQual: 1 } }
define amdgpu_kernel void @test_type_qual(i32 addrspace(1)* %a, i32 addrspace(1)* %b, %opencl.pipe_t addrspace(1)* %c) !kernel_arg_addr_space !22 !kernel_arg_access_qual !23 !kernel_arg_type !51 !kernel_arg_base_type !51 !kernel_arg_type_qual !70 {
  ret void
}

; CHECK-NEXT:   - { amd.KernelName: test_access_qual, amd.Language: OpenCL C, amd.LanguageVersion: [ 2, 0 ], amd.Args: 
; CHECK-NEXT:       - { amd.ArgSize: 8, amd.ArgAlign: 8, amd.ArgKind: 4, amd.ArgValueType: 0, amd.ArgTypeName: image1d_t, amd.ArgAddrQual: 1, amd.ArgAccQual: 1 }
; CHECK-NEXT:       - { amd.ArgSize: 8, amd.ArgAlign: 8, amd.ArgKind: 4, amd.ArgValueType: 0, amd.ArgTypeName: image2d_t, amd.ArgAddrQual: 1, amd.ArgAccQual: 2 }
; CHECK-NEXT:       - { amd.ArgSize: 8, amd.ArgAlign: 8, amd.ArgKind: 4, amd.ArgValueType: 0, amd.ArgTypeName: image3d_t, amd.ArgAddrQual: 1, amd.ArgAccQual: 3 }
; CHECK-NEXT:       - { amd.ArgSize: 8, amd.ArgAlign: 8, amd.ArgKind: 7, amd.ArgValueType: 9 }
; CHECK-NEXT:       - { amd.ArgSize: 8, amd.ArgAlign: 8, amd.ArgKind: 8, amd.ArgValueType: 9 }
; CHECK-NEXT:       - { amd.ArgSize: 8, amd.ArgAlign: 8, amd.ArgKind: 9, amd.ArgValueType: 9 }
; CHECK-NEXT:       - { amd.ArgSize: 8, amd.ArgAlign: 8, amd.ArgKind: 11, amd.ArgValueType: 1, amd.ArgAddrQual: 1 } }
define amdgpu_kernel void @test_access_qual(%opencl.image1d_t addrspace(1)* %ro, %opencl.image2d_t addrspace(1)* %wo, %opencl.image3d_t addrspace(1)* %rw) !kernel_arg_addr_space !60 !kernel_arg_access_qual !61 !kernel_arg_type !62 !kernel_arg_base_type !62 !kernel_arg_type_qual !25 {
  ret void
}

; CHECK-NEXT:   - { amd.KernelName: test_vec_type_hint_half, amd.Language: OpenCL C, amd.LanguageVersion: [ 2, 0 ], amd.VecTypeHint: half, amd.Args: 
; CHECK-NEXT:       - { amd.ArgSize: 4, amd.ArgAlign: 4, amd.ArgKind: 0, amd.ArgValueType: 6, amd.ArgTypeName: int, amd.ArgAccQual: 0 }
; CHECK-NEXT:       - { amd.ArgSize: 8, amd.ArgAlign: 8, amd.ArgKind: 7, amd.ArgValueType: 9 }
; CHECK-NEXT:       - { amd.ArgSize: 8, amd.ArgAlign: 8, amd.ArgKind: 8, amd.ArgValueType: 9 }
; CHECK-NEXT:       - { amd.ArgSize: 8, amd.ArgAlign: 8, amd.ArgKind: 9, amd.ArgValueType: 9 }
; CHECK-NEXT:       - { amd.ArgSize: 8, amd.ArgAlign: 8, amd.ArgKind: 11, amd.ArgValueType: 1, amd.ArgAddrQual: 1 } }
define amdgpu_kernel void @test_vec_type_hint_half(i32 %a) !kernel_arg_addr_space !1 !kernel_arg_access_qual !2 !kernel_arg_type !3 !kernel_arg_base_type !3 !kernel_arg_type_qual !4 !vec_type_hint !26 {
  ret void
}

; CHECK-NEXT:   - { amd.KernelName: test_vec_type_hint_float, amd.Language: OpenCL C, amd.LanguageVersion: [ 2, 0 ], amd.VecTypeHint: float, amd.Args: 
; CHECK-NEXT:       - { amd.ArgSize: 4, amd.ArgAlign: 4, amd.ArgKind: 0, amd.ArgValueType: 6, amd.ArgTypeName: int, amd.ArgAccQual: 0 }
; CHECK-NEXT:       - { amd.ArgSize: 8, amd.ArgAlign: 8, amd.ArgKind: 7, amd.ArgValueType: 9 }
; CHECK-NEXT:       - { amd.ArgSize: 8, amd.ArgAlign: 8, amd.ArgKind: 8, amd.ArgValueType: 9 }
; CHECK-NEXT:       - { amd.ArgSize: 8, amd.ArgAlign: 8, amd.ArgKind: 9, amd.ArgValueType: 9 }
; CHECK-NEXT:       - { amd.ArgSize: 8, amd.ArgAlign: 8, amd.ArgKind: 11, amd.ArgValueType: 1, amd.ArgAddrQual: 1 } }
define amdgpu_kernel void @test_vec_type_hint_float(i32 %a) !kernel_arg_addr_space !1 !kernel_arg_access_qual !2 !kernel_arg_type !3 !kernel_arg_base_type !3 !kernel_arg_type_qual !4 !vec_type_hint !27 {
  ret void
}

; CHECK-NEXT:   - { amd.KernelName: test_vec_type_hint_double, amd.Language: OpenCL C, amd.LanguageVersion: [ 2, 0 ], amd.VecTypeHint: double, amd.Args: 
; CHECK-NEXT:       - { amd.ArgSize: 4, amd.ArgAlign: 4, amd.ArgKind: 0, amd.ArgValueType: 6, amd.ArgTypeName: int, amd.ArgAccQual: 0 }
; CHECK-NEXT:       - { amd.ArgSize: 8, amd.ArgAlign: 8, amd.ArgKind: 7, amd.ArgValueType: 9 }
; CHECK-NEXT:       - { amd.ArgSize: 8, amd.ArgAlign: 8, amd.ArgKind: 8, amd.ArgValueType: 9 }
; CHECK-NEXT:       - { amd.ArgSize: 8, amd.ArgAlign: 8, amd.ArgKind: 9, amd.ArgValueType: 9 }
; CHECK-NEXT:       - { amd.ArgSize: 8, amd.ArgAlign: 8, amd.ArgKind: 11, amd.ArgValueType: 1, amd.ArgAddrQual: 1 } }
define amdgpu_kernel void @test_vec_type_hint_double(i32 %a) !kernel_arg_addr_space !1 !kernel_arg_access_qual !2 !kernel_arg_type !3 !kernel_arg_base_type !3 !kernel_arg_type_qual !4 !vec_type_hint !28 {
  ret void
}

; CHECK-NEXT:   - { amd.KernelName: test_vec_type_hint_char, amd.Language: OpenCL C, amd.LanguageVersion: [ 2, 0 ], amd.VecTypeHint: char, amd.Args: 
; CHECK-NEXT:       - { amd.ArgSize: 4, amd.ArgAlign: 4, amd.ArgKind: 0, amd.ArgValueType: 6, amd.ArgTypeName: int, amd.ArgAccQual: 0 }
; CHECK-NEXT:       - { amd.ArgSize: 8, amd.ArgAlign: 8, amd.ArgKind: 7, amd.ArgValueType: 9 }
; CHECK-NEXT:       - { amd.ArgSize: 8, amd.ArgAlign: 8, amd.ArgKind: 8, amd.ArgValueType: 9 }
; CHECK-NEXT:       - { amd.ArgSize: 8, amd.ArgAlign: 8, amd.ArgKind: 9, amd.ArgValueType: 9 }
; CHECK-NEXT:       - { amd.ArgSize: 8, amd.ArgAlign: 8, amd.ArgKind: 11, amd.ArgValueType: 1, amd.ArgAddrQual: 1 } }
define amdgpu_kernel void @test_vec_type_hint_char(i32 %a) !kernel_arg_addr_space !1 !kernel_arg_access_qual !2 !kernel_arg_type !3 !kernel_arg_base_type !3 !kernel_arg_type_qual !4 !vec_type_hint !29 {
  ret void
}

; CHECK-NEXT:   - { amd.KernelName: test_vec_type_hint_short, amd.Language: OpenCL C, amd.LanguageVersion: [ 2, 0 ], amd.VecTypeHint: short, amd.Args: 
; CHECK-NEXT:       - { amd.ArgSize: 4, amd.ArgAlign: 4, amd.ArgKind: 0, amd.ArgValueType: 6, amd.ArgTypeName: int, amd.ArgAccQual: 0 }
; CHECK-NEXT:       - { amd.ArgSize: 8, amd.ArgAlign: 8, amd.ArgKind: 7, amd.ArgValueType: 9 }
; CHECK-NEXT:       - { amd.ArgSize: 8, amd.ArgAlign: 8, amd.ArgKind: 8, amd.ArgValueType: 9 }
; CHECK-NEXT:       - { amd.ArgSize: 8, amd.ArgAlign: 8, amd.ArgKind: 9, amd.ArgValueType: 9 }
; CHECK-NEXT:       - { amd.ArgSize: 8, amd.ArgAlign: 8, amd.ArgKind: 11, amd.ArgValueType: 1, amd.ArgAddrQual: 1 } }
define amdgpu_kernel void @test_vec_type_hint_short(i32 %a) !kernel_arg_addr_space !1 !kernel_arg_access_qual !2 !kernel_arg_type !3 !kernel_arg_base_type !3 !kernel_arg_type_qual !4 !vec_type_hint !30 {
  ret void
}

; CHECK-NEXT:   - { amd.KernelName: test_vec_type_hint_long, amd.Language: OpenCL C, amd.LanguageVersion: [ 2, 0 ], amd.VecTypeHint: long, amd.Args: 
; CHECK-NEXT:       - { amd.ArgSize: 4, amd.ArgAlign: 4, amd.ArgKind: 0, amd.ArgValueType: 6, amd.ArgTypeName: int, amd.ArgAccQual: 0 }
; CHECK-NEXT:       - { amd.ArgSize: 8, amd.ArgAlign: 8, amd.ArgKind: 7, amd.ArgValueType: 9 }
; CHECK-NEXT:       - { amd.ArgSize: 8, amd.ArgAlign: 8, amd.ArgKind: 8, amd.ArgValueType: 9 }
; CHECK-NEXT:       - { amd.ArgSize: 8, amd.ArgAlign: 8, amd.ArgKind: 9, amd.ArgValueType: 9 }
; CHECK-NEXT:       - { amd.ArgSize: 8, amd.ArgAlign: 8, amd.ArgKind: 11, amd.ArgValueType: 1, amd.ArgAddrQual: 1 } }
define amdgpu_kernel void @test_vec_type_hint_long(i32 %a) !kernel_arg_addr_space !1 !kernel_arg_access_qual !2 !kernel_arg_type !3 !kernel_arg_base_type !3 !kernel_arg_type_qual !4 !vec_type_hint !31 {
  ret void
}

; CHECK-NEXT:   - { amd.KernelName: test_vec_type_hint_unknown, amd.Language: OpenCL C, amd.LanguageVersion: [ 2, 0 ], amd.VecTypeHint: unknown, amd.Args: 
; CHECK-NEXT:       - { amd.ArgSize: 4, amd.ArgAlign: 4, amd.ArgKind: 0, amd.ArgValueType: 6, amd.ArgTypeName: int, amd.ArgAccQual: 0 }
; CHECK-NEXT:       - { amd.ArgSize: 8, amd.ArgAlign: 8, amd.ArgKind: 7, amd.ArgValueType: 9 }
; CHECK-NEXT:       - { amd.ArgSize: 8, amd.ArgAlign: 8, amd.ArgKind: 8, amd.ArgValueType: 9 }
; CHECK-NEXT:       - { amd.ArgSize: 8, amd.ArgAlign: 8, amd.ArgKind: 9, amd.ArgValueType: 9 }
; CHECK-NEXT:       - { amd.ArgSize: 8, amd.ArgAlign: 8, amd.ArgKind: 11, amd.ArgValueType: 1, amd.ArgAddrQual: 1 } }
define amdgpu_kernel void @test_vec_type_hint_unknown(i32 %a) !kernel_arg_addr_space !1 !kernel_arg_access_qual !2 !kernel_arg_type !3 !kernel_arg_base_type !3 !kernel_arg_type_qual !4 !vec_type_hint !32 {
  ret void
}

; CHECK-NEXT:   - { amd.KernelName: test_reqd_wgs_vec_type_hint, amd.Language: OpenCL C, amd.LanguageVersion: [ 2, 0 ], amd.ReqdWorkGroupSize: [ 1, 2, 4 ], amd.VecTypeHint: int, amd.Args: 
; CHECK-NEXT:       - { amd.ArgSize: 4, amd.ArgAlign: 4, amd.ArgKind: 0, amd.ArgValueType: 6, amd.ArgTypeName: int, amd.ArgAccQual: 0 }
; CHECK-NEXT:       - { amd.ArgSize: 8, amd.ArgAlign: 8, amd.ArgKind: 7, amd.ArgValueType: 9 }
; CHECK-NEXT:       - { amd.ArgSize: 8, amd.ArgAlign: 8, amd.ArgKind: 8, amd.ArgValueType: 9 }
; CHECK-NEXT:       - { amd.ArgSize: 8, amd.ArgAlign: 8, amd.ArgKind: 9, amd.ArgValueType: 9 }
; CHECK-NEXT:       - { amd.ArgSize: 8, amd.ArgAlign: 8, amd.ArgKind: 11, amd.ArgValueType: 1, amd.ArgAddrQual: 1 } }
define amdgpu_kernel void @test_reqd_wgs_vec_type_hint(i32 %a) !kernel_arg_addr_space !1 !kernel_arg_access_qual !2 !kernel_arg_type !3 !kernel_arg_base_type !3 !kernel_arg_type_qual !4 !vec_type_hint !5 !reqd_work_group_size !6 {
  ret void
}

; CHECK-NEXT:   - { amd.KernelName: test_wgs_hint_vec_type_hint, amd.Language: OpenCL C, amd.LanguageVersion: [ 2, 0 ], amd.WorkGroupSizeHint: [ 8, 16, 32 ], amd.VecTypeHint: uint4, amd.Args: 
; CHECK-NEXT:       - { amd.ArgSize: 4, amd.ArgAlign: 4, amd.ArgKind: 0, amd.ArgValueType: 6, amd.ArgTypeName: int, amd.ArgAccQual: 0 }
; CHECK-NEXT:       - { amd.ArgSize: 8, amd.ArgAlign: 8, amd.ArgKind: 7, amd.ArgValueType: 9 }
; CHECK-NEXT:       - { amd.ArgSize: 8, amd.ArgAlign: 8, amd.ArgKind: 8, amd.ArgValueType: 9 }
; CHECK-NEXT:       - { amd.ArgSize: 8, amd.ArgAlign: 8, amd.ArgKind: 9, amd.ArgValueType: 9 }
; CHECK-NEXT:       - { amd.ArgSize: 8, amd.ArgAlign: 8, amd.ArgKind: 11, amd.ArgValueType: 1, amd.ArgAddrQual: 1 } }
define amdgpu_kernel void @test_wgs_hint_vec_type_hint(i32 %a) !kernel_arg_addr_space !1 !kernel_arg_access_qual !2 !kernel_arg_type !3 !kernel_arg_base_type !3 !kernel_arg_type_qual !4 !vec_type_hint !7 !work_group_size_hint !8 {
  ret void
}

; CHECK-NEXT:   - { amd.KernelName: test_arg_ptr_to_ptr, amd.Language: OpenCL C, amd.LanguageVersion: [ 2, 0 ], amd.Args: 
; CHECK-NEXT:       - { amd.ArgSize: 8, amd.ArgAlign: 8, amd.ArgKind: 1, amd.ArgValueType: 6, amd.ArgTypeName: 'int **', amd.ArgAddrQual: 1, amd.ArgAccQual: 0 }
; CHECK-NEXT:       - { amd.ArgSize: 8, amd.ArgAlign: 8, amd.ArgKind: 7, amd.ArgValueType: 9 }
; CHECK-NEXT:       - { amd.ArgSize: 8, amd.ArgAlign: 8, amd.ArgKind: 8, amd.ArgValueType: 9 }
; CHECK-NEXT:       - { amd.ArgSize: 8, amd.ArgAlign: 8, amd.ArgKind: 9, amd.ArgValueType: 9 }
; CHECK-NEXT:       - { amd.ArgSize: 8, amd.ArgAlign: 8, amd.ArgKind: 11, amd.ArgValueType: 1, amd.ArgAddrQual: 1 } }
define amdgpu_kernel void @test_arg_ptr_to_ptr(i32 * addrspace(1)* %a) !kernel_arg_addr_space !81 !kernel_arg_access_qual !2 !kernel_arg_type !80 !kernel_arg_base_type !80 !kernel_arg_type_qual !4 {
  ret void
}
; CHECK-NEXT:   - { amd.KernelName: test_arg_struct_contains_ptr, amd.Language: OpenCL C, amd.LanguageVersion: [ 2, 0 ], amd.Args: 
; CHECK-NEXT:       - { amd.ArgSize: 4, amd.ArgAlign: 4, amd.ArgKind: 1, amd.ArgValueType: 0, amd.ArgTypeName: struct B, amd.ArgAddrQual: 0, amd.ArgAccQual: 0 }
; CHECK-NEXT:       - { amd.ArgSize: 8, amd.ArgAlign: 8, amd.ArgKind: 7, amd.ArgValueType: 9 }
; CHECK-NEXT:       - { amd.ArgSize: 8, amd.ArgAlign: 8, amd.ArgKind: 8, amd.ArgValueType: 9 }
; CHECK-NEXT:       - { amd.ArgSize: 8, amd.ArgAlign: 8, amd.ArgKind: 9, amd.ArgValueType: 9 }
; CHECK-NEXT:       - { amd.ArgSize: 8, amd.ArgAlign: 8, amd.ArgKind: 11, amd.ArgValueType: 1, amd.ArgAddrQual: 1 } }
define amdgpu_kernel void @test_arg_struct_contains_ptr(%struct.B * byval %a) !kernel_arg_addr_space !1 !kernel_arg_access_qual !2 !kernel_arg_type !82 !kernel_arg_base_type !82 !kernel_arg_type_qual !4 {
  ret void
}

; CHECK-NEXT:   - { amd.KernelName: test_arg_vector_of_ptr, amd.Language: OpenCL C, amd.LanguageVersion: [ 2, 0 ], amd.Args: 
; CHECK-NEXT:       - { amd.ArgSize: 16, amd.ArgAlign: 16, amd.ArgKind: 0, amd.ArgValueType: 6, amd.ArgTypeName: 'global int* __attribute__((ext_vector_type(2)))', amd.ArgAccQual: 0 }
; CHECK-NEXT:       - { amd.ArgSize: 8, amd.ArgAlign: 8, amd.ArgKind: 7, amd.ArgValueType: 9 }
; CHECK-NEXT:       - { amd.ArgSize: 8, amd.ArgAlign: 8, amd.ArgKind: 8, amd.ArgValueType: 9 }
; CHECK-NEXT:       - { amd.ArgSize: 8, amd.ArgAlign: 8, amd.ArgKind: 9, amd.ArgValueType: 9 }
; CHECK-NEXT:       - { amd.ArgSize: 8, amd.ArgAlign: 8, amd.ArgKind: 11, amd.ArgValueType: 1, amd.ArgAddrQual: 1 } }
define amdgpu_kernel void @test_arg_vector_of_ptr(<2 x i32 addrspace(1)*> %a) !kernel_arg_addr_space !1 !kernel_arg_access_qual !2 !kernel_arg_type !83 !kernel_arg_base_type !83 !kernel_arg_type_qual !4 {
  ret void
}


; CHECK-NEXT:   - { amd.KernelName: test_arg_unknown_builtin_type, amd.Language: OpenCL C, amd.LanguageVersion: [ 2, 0 ], amd.Args: 
; CHECK-NEXT:       - { amd.ArgSize: 8, amd.ArgAlign: 8, amd.ArgKind: 1, amd.ArgValueType: 0, amd.ArgTypeName: clk_event_t, amd.ArgAddrQual: 1, amd.ArgAccQual: 0 }
; CHECK-NEXT:       - { amd.ArgSize: 8, amd.ArgAlign: 8, amd.ArgKind: 7, amd.ArgValueType: 9 }
; CHECK-NEXT:       - { amd.ArgSize: 8, amd.ArgAlign: 8, amd.ArgKind: 8, amd.ArgValueType: 9 }
; CHECK-NEXT:       - { amd.ArgSize: 8, amd.ArgAlign: 8, amd.ArgKind: 9, amd.ArgValueType: 9 }
; CHECK-NEXT:       - { amd.ArgSize: 8, amd.ArgAlign: 8, amd.ArgKind: 11, amd.ArgValueType: 1, amd.ArgAddrQual: 1 } }
define amdgpu_kernel void @test_arg_unknown_builtin_type(%opencl.clk_event_t addrspace(1)* %a) !kernel_arg_addr_space !81 !kernel_arg_access_qual !2 !kernel_arg_type !84 !kernel_arg_base_type !84 !kernel_arg_type_qual !4 {
  ret void
}

; CHECK-NEXT:   - { amd.KernelName: test_pointee_align, amd.Language: OpenCL C, amd.LanguageVersion: [ 2, 0 ], amd.Args: 
; CHECK-NEXT:       - { amd.ArgSize: 8, amd.ArgAlign: 8, amd.ArgKind: 1, amd.ArgValueType: 9, amd.ArgTypeName: 'long *', amd.ArgAddrQual: 1, amd.ArgAccQual: 0 }
; CHECK-NEXT:       - { amd.ArgSize: 4, amd.ArgAlign: 4, amd.ArgPointeeAlign: 1, amd.ArgKind: 2, amd.ArgValueType: 1, amd.ArgTypeName: 'char *', amd.ArgAddrQual: 3, amd.ArgAccQual: 0 }
; CHECK-NEXT:       - { amd.ArgSize: 4, amd.ArgAlign: 4, amd.ArgPointeeAlign: 2, amd.ArgKind: 2, amd.ArgValueType: 1, amd.ArgTypeName: 'char2 *', amd.ArgAddrQual: 3, amd.ArgAccQual: 0 }
; CHECK-NEXT:       - { amd.ArgSize: 4, amd.ArgAlign: 4, amd.ArgPointeeAlign: 4, amd.ArgKind: 2, amd.ArgValueType: 1, amd.ArgTypeName: 'char3 *', amd.ArgAddrQual: 3, amd.ArgAccQual: 0 }
; CHECK-NEXT:       - { amd.ArgSize: 4, amd.ArgAlign: 4, amd.ArgPointeeAlign: 4, amd.ArgKind: 2, amd.ArgValueType: 1, amd.ArgTypeName: 'char4 *', amd.ArgAddrQual: 3, amd.ArgAccQual: 0 }
; CHECK-NEXT:       - { amd.ArgSize: 4, amd.ArgAlign: 4, amd.ArgPointeeAlign: 8, amd.ArgKind: 2, amd.ArgValueType: 1, amd.ArgTypeName: 'char8 *', amd.ArgAddrQual: 3, amd.ArgAccQual: 0 }
; CHECK-NEXT:       - { amd.ArgSize: 4, amd.ArgAlign: 4, amd.ArgPointeeAlign: 16, amd.ArgKind: 2, amd.ArgValueType: 1, amd.ArgTypeName: 'char16 *', amd.ArgAddrQual: 3, amd.ArgAccQual: 0 }
; CHECK-NEXT:       - { amd.ArgSize: 8, amd.ArgAlign: 8, amd.ArgKind: 7, amd.ArgValueType: 9 }
; CHECK-NEXT:       - { amd.ArgSize: 8, amd.ArgAlign: 8, amd.ArgKind: 8, amd.ArgValueType: 9 }
; CHECK-NEXT:       - { amd.ArgSize: 8, amd.ArgAlign: 8, amd.ArgKind: 9, amd.ArgValueType: 9 }
; CHECK-NEXT:       - { amd.ArgSize: 8, amd.ArgAlign: 8, amd.ArgKind: 11, amd.ArgValueType: 1, amd.ArgAddrQual: 1 } } }
define amdgpu_kernel void @test_pointee_align(i64 addrspace(1)* %a, i8 addrspace(3)* %b, <2 x i8> addrspace(3)* %c, <3 x i8> addrspace(3)* %d, <4 x i8> addrspace(3)* %e, <8 x i8> addrspace(3)* %f, <16 x i8> addrspace(3)* %g) !kernel_arg_addr_space !91 !kernel_arg_access_qual !92 !kernel_arg_type !93 !kernel_arg_base_type !93 !kernel_arg_type_qual !94 {
  ret void
}

; CHECK-NEXT:...

; PARSER: AMDGPU runtime metadata parser test passes.

; NOTES: Displaying notes found at file offset 0x{{[0-9]+}}
; NOTES-NEXT: Owner    Data size    Description
; NOTES-NEXT: AMD      0x00000008   Unknown note type: (0x00000001)
; NOTES-NEXT: AMD      0x0000001b   Unknown note type: (0x00000003)

; SI:         AMD      0x0000530d   Unknown note type: (0x00000008)
; VI:         AMD      0x0000530e   Unknown note type: (0x00000008)

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

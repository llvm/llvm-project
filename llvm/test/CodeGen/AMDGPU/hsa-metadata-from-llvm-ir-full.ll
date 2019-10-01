; RUN: llc -mtriple=amdgcn-amd-amdhsa -mcpu=gfx700 -mattr=-code-object-v3 -filetype=obj -o - < %s | llvm-readelf --notes | FileCheck --check-prefix=CHECK --check-prefix=GFX700 --check-prefix=NOTES %s
; RUN: llc -mtriple=amdgcn-amd-amdhsa -mcpu=gfx802 -mattr=-code-object-v3 -filetype=obj -o - < %s | llvm-readelf --notes | FileCheck --check-prefix=CHECK --check-prefix=GFX802 --check-prefix=NOTES %s
; RUN: llc -mtriple=amdgcn-amd-amdhsa -mcpu=gfx900 -mattr=-code-object-v3 -filetype=obj -o - < %s | llvm-readelf --notes | FileCheck --check-prefix=CHECK --check-prefix=GFX900 --check-prefix=NOTES %s
; RUN: llc -mtriple=amdgcn-amd-amdhsa -mcpu=gfx700 -mattr=-code-object-v3 -amdgpu-dump-hsa-metadata -amdgpu-verify-hsa-metadata -filetype=obj -o - < %s 2>&1 | FileCheck --check-prefix=PARSER %s
; RUN: llc -mtriple=amdgcn-amd-amdhsa -mcpu=gfx802 -mattr=-code-object-v3 -amdgpu-dump-hsa-metadata -amdgpu-verify-hsa-metadata -filetype=obj -o - < %s 2>&1 | FileCheck --check-prefix=PARSER %s
; RUN: llc -mtriple=amdgcn-amd-amdhsa -mcpu=gfx900 -mattr=-code-object-v3 -amdgpu-dump-hsa-metadata -amdgpu-verify-hsa-metadata -filetype=obj -o - < %s 2>&1 | FileCheck --check-prefix=PARSER %s

%struct.A = type { i8, float }
%opencl.image1d_t = type opaque
%opencl.image2d_t = type opaque
%opencl.image3d_t = type opaque
%opencl.queue_t = type opaque
%opencl.pipe_t = type opaque
%struct.B = type { i32 addrspace(1)*}
%opencl.clk_event_t = type opaque

@__test_block_invoke_kernel_runtime_handle = external addrspace(1) externally_initialized constant i8 addrspace(1)*

; CHECK: ---
; CHECK:  Version: [ 1, 0 ]
; CHECK:  Printf:
; CHECK:    - '1:1:4:%d\n'
; CHECK:    - '2:1:8:%g\n'
; CHECK:  Kernels:

; CHECK:      - Name:            test_char
; CHECK-NEXT:   SymbolName:      'test_char@kd'
; CHECK-NEXT:   Language:        OpenCL C
; CHECK-NEXT:   LanguageVersion: [ 2, 0 ]
; CHECK-NEXT:   Args:
; CHECK-NEXT:     - Name:          a
; CHECK-NEXT:       TypeName:      char
; CHECK-NEXT:       Size:          1
; CHECK-NEXT:       Align:         1
; CHECK-NEXT:       ValueKind:     ByValue
; CHECK-NEXT:       ValueType:     I8
; CHECK-NEXT:       AccQual:       Default
; CHECK-NEXT:     - Size:          8
; CHECK-NEXT:       Align:         8
; CHECK-NEXT:       ValueKind:     HiddenGlobalOffsetX
; CHECK-NEXT:       ValueType:     I64
; CHECK-NEXT:     - Size:          8
; CHECK-NEXT:       Align:         8
; CHECK-NEXT:       ValueKind:     HiddenGlobalOffsetY
; CHECK-NEXT:       ValueType:     I64
; CHECK-NEXT:     - Size:          8
; CHECK-NEXT:       Align:         8
; CHECK-NEXT:       ValueKind:     HiddenGlobalOffsetZ
; CHECK-NEXT:       ValueType:     I64
; CHECK-NEXT:     - Size:          8
; CHECK-NEXT:       Align:         8
; CHECK-NEXT:       ValueKind:     HiddenPrintfBuffer
; CHECK-NEXT:       ValueType:     I8
; CHECK-NEXT:       AddrSpaceQual: Global
; CHECK-NOT:        ValueKind:     HiddenDefaultQueue
; CHECK-NOT:        ValueKind:     HiddenCompletionAction
define amdgpu_kernel void @test_char(i8 %a) #0
    !kernel_arg_addr_space !1 !kernel_arg_access_qual !2 !kernel_arg_type !9
    !kernel_arg_base_type !9 !kernel_arg_type_qual !4 {
  ret void
}

; CHECK:      - Name:            test_ushort2
; CHECK-NEXT:   SymbolName:      'test_ushort2@kd'
; CHECK-NEXT:   Language:        OpenCL C
; CHECK-NEXT:   LanguageVersion: [ 2, 0 ]
; CHECK-NEXT:   Args:
; CHECK-NEXT:     - Name:          a
; CHECK-NEXT:       TypeName:      ushort2
; CHECK-NEXT:       Size:          4
; CHECK-NEXT:       Align:         4
; CHECK-NEXT:       ValueKind:     ByValue
; CHECK-NEXT:       ValueType:     U16
; CHECK-NEXT:       AccQual:       Default
; CHECK-NEXT:     - Size:          8
; CHECK-NEXT:       Align:         8
; CHECK-NEXT:       ValueKind:     HiddenGlobalOffsetX
; CHECK-NEXT:       ValueType:     I64
; CHECK-NEXT:     - Size:          8
; CHECK-NEXT:       Align:         8
; CHECK-NEXT:       ValueKind:     HiddenGlobalOffsetY
; CHECK-NEXT:       ValueType:     I64
; CHECK-NEXT:     - Size:          8
; CHECK-NEXT:       Align:         8
; CHECK-NEXT:       ValueKind:     HiddenGlobalOffsetZ
; CHECK-NEXT:       ValueType:     I64
; CHECK-NEXT:     - Size:          8
; CHECK-NEXT:       Align:         8
; CHECK-NEXT:       ValueKind:     HiddenPrintfBuffer
; CHECK-NEXT:       ValueType:     I8
; CHECK-NEXT:       AddrSpaceQual: Global
define amdgpu_kernel void @test_ushort2(<2 x i16> %a) #0
    !kernel_arg_addr_space !1 !kernel_arg_access_qual !2 !kernel_arg_type !10
    !kernel_arg_base_type !10 !kernel_arg_type_qual !4 {
  ret void
}

; CHECK:      - Name:            test_int3
; CHECK-NEXT:   SymbolName:      'test_int3@kd'
; CHECK-NEXT:   Language:        OpenCL C
; CHECK-NEXT:   LanguageVersion: [ 2, 0 ]
; CHECK-NEXT:   Args:
; CHECK-NEXT:     - Name:          a
; CHECK-NEXT:       TypeName:      int3
; CHECK-NEXT:       Size:          16
; CHECK-NEXT:       Align:         16
; CHECK-NEXT:       ValueKind:     ByValue
; CHECK-NEXT:       ValueType:     I32
; CHECK-NEXT:       AccQual:       Default
; CHECK-NEXT:     - Size:          8
; CHECK-NEXT:       Align:         8
; CHECK-NEXT:       ValueKind:     HiddenGlobalOffsetX
; CHECK-NEXT:       ValueType:     I64
; CHECK-NEXT:     - Size:          8
; CHECK-NEXT:       Align:         8
; CHECK-NEXT:       ValueKind:     HiddenGlobalOffsetY
; CHECK-NEXT:       ValueType:     I64
; CHECK-NEXT:     - Size:          8
; CHECK-NEXT:       Align:         8
; CHECK-NEXT:       ValueKind:     HiddenGlobalOffsetZ
; CHECK-NEXT:       ValueType:     I64
; CHECK-NEXT:     - Size:          8
; CHECK-NEXT:       Align:         8
; CHECK-NEXT:       ValueKind:     HiddenPrintfBuffer
; CHECK-NEXT:       ValueType:     I8
; CHECK-NEXT:       AddrSpaceQual: Global
define amdgpu_kernel void @test_int3(<3 x i32> %a) #0
    !kernel_arg_addr_space !1 !kernel_arg_access_qual !2 !kernel_arg_type !11
    !kernel_arg_base_type !11 !kernel_arg_type_qual !4 {
  ret void
}

; CHECK:      - Name:            test_ulong4
; CHECK-NEXT:   SymbolName:      'test_ulong4@kd'
; CHECK-NEXT:   Language:        OpenCL C
; CHECK-NEXT:   LanguageVersion: [ 2, 0 ]
; CHECK-NEXT:   Args:
; CHECK-NEXT:     - Name:          a
; CHECK-NEXT:       TypeName:      ulong4
; CHECK-NEXT:       Size:          32
; CHECK-NEXT:       Align:         32
; CHECK-NEXT:       ValueKind:     ByValue
; CHECK-NEXT:       ValueType:     U64
; CHECK-NEXT:       AccQual:       Default
; CHECK-NEXT:     - Size:          8
; CHECK-NEXT:       Align:         8
; CHECK-NEXT:       ValueKind:     HiddenGlobalOffsetX
; CHECK-NEXT:       ValueType:     I64
; CHECK-NEXT:     - Size:          8
; CHECK-NEXT:       Align:         8
; CHECK-NEXT:       ValueKind:     HiddenGlobalOffsetY
; CHECK-NEXT:       ValueType:     I64
; CHECK-NEXT:     - Size:          8
; CHECK-NEXT:       Align:         8
; CHECK-NEXT:       ValueKind:     HiddenGlobalOffsetZ
; CHECK-NEXT:       ValueType:     I64
; CHECK-NEXT:     - Size:          8
; CHECK-NEXT:       Align:         8
; CHECK-NEXT:       ValueKind:     HiddenPrintfBuffer
; CHECK-NEXT:       ValueType:     I8
; CHECK-NEXT:       AddrSpaceQual: Global
define amdgpu_kernel void @test_ulong4(<4 x i64> %a) #0
    !kernel_arg_addr_space !1 !kernel_arg_access_qual !2 !kernel_arg_type !12
    !kernel_arg_base_type !12 !kernel_arg_type_qual !4 {
  ret void
}

; CHECK:      - Name:            test_half8
; CHECK-NEXT:   SymbolName:      'test_half8@kd'
; CHECK-NEXT:   Language:        OpenCL C
; CHECK-NEXT:   LanguageVersion: [ 2, 0 ]
; CHECK-NEXT:   Args:
; CHECK-NEXT:     - Name:          a
; CHECK-NEXT:       TypeName:      half8
; CHECK-NEXT:       Size:          16
; CHECK-NEXT:       Align:         16
; CHECK-NEXT:       ValueKind:     ByValue
; CHECK-NEXT:       ValueType:     F16
; CHECK-NEXT:       AccQual:       Default
; CHECK-NEXT:     - Size:          8
; CHECK-NEXT:       Align:         8
; CHECK-NEXT:       ValueKind:     HiddenGlobalOffsetX
; CHECK-NEXT:       ValueType:     I64
; CHECK-NEXT:     - Size:          8
; CHECK-NEXT:       Align:         8
; CHECK-NEXT:       ValueKind:     HiddenGlobalOffsetY
; CHECK-NEXT:       ValueType:     I64
; CHECK-NEXT:     - Size:          8
; CHECK-NEXT:       Align:         8
; CHECK-NEXT:       ValueKind:     HiddenGlobalOffsetZ
; CHECK-NEXT:       ValueType:     I64
; CHECK-NEXT:     - Size:          8
; CHECK-NEXT:       Align:         8
; CHECK-NEXT:       ValueKind:     HiddenPrintfBuffer
; CHECK-NEXT:       ValueType:     I8
; CHECK-NEXT:       AddrSpaceQual: Global
define amdgpu_kernel void @test_half8(<8 x half> %a) #0
    !kernel_arg_addr_space !1 !kernel_arg_access_qual !2 !kernel_arg_type !13
    !kernel_arg_base_type !13 !kernel_arg_type_qual !4 {
  ret void
}

; CHECK:      - Name:            test_float16
; CHECK-NEXT:   SymbolName:      'test_float16@kd'
; CHECK-NEXT:   Language:        OpenCL C
; CHECK-NEXT:   LanguageVersion: [ 2, 0 ]
; CHECK-NEXT:   Args:
; CHECK-NEXT:     - Name:          a
; CHECK-NEXT:       TypeName:      float16
; CHECK-NEXT:       Size:          64
; CHECK-NEXT:       Align:         64
; CHECK-NEXT:       ValueKind:     ByValue
; CHECK-NEXT:       ValueType:     F32
; CHECK-NEXT:       AccQual:       Default
; CHECK-NEXT:     - Size:          8
; CHECK-NEXT:       Align:         8
; CHECK-NEXT:       ValueKind:     HiddenGlobalOffsetX
; CHECK-NEXT:       ValueType:     I64
; CHECK-NEXT:     - Size:          8
; CHECK-NEXT:       Align:         8
; CHECK-NEXT:       ValueKind:     HiddenGlobalOffsetY
; CHECK-NEXT:       ValueType:     I64
; CHECK-NEXT:     - Size:          8
; CHECK-NEXT:       Align:         8
; CHECK-NEXT:       ValueKind:     HiddenGlobalOffsetZ
; CHECK-NEXT:       ValueType:     I64
; CHECK-NEXT:     - Size:          8
; CHECK-NEXT:       Align:         8
; CHECK-NEXT:       ValueKind:     HiddenPrintfBuffer
; CHECK-NEXT:       ValueType:     I8
; CHECK-NEXT:       AddrSpaceQual: Global
define amdgpu_kernel void @test_float16(<16 x float> %a) #0
    !kernel_arg_addr_space !1 !kernel_arg_access_qual !2 !kernel_arg_type !14
    !kernel_arg_base_type !14 !kernel_arg_type_qual !4 {
  ret void
}

; CHECK:      - Name:            test_double16
; CHECK-NEXT:   SymbolName:      'test_double16@kd'
; CHECK-NEXT:   Language:        OpenCL C
; CHECK-NEXT:   LanguageVersion: [ 2, 0 ]
; CHECK-NEXT:   Args:
; CHECK-NEXT:     - Name:          a
; CHECK-NEXT:       TypeName:      double16
; CHECK-NEXT:       Size:          128
; CHECK-NEXT:       Align:         128
; CHECK-NEXT:       ValueKind:     ByValue
; CHECK-NEXT:       ValueType:     F64
; CHECK-NEXT:       AccQual:       Default
; CHECK-NEXT:     - Size:          8
; CHECK-NEXT:       Align:         8
; CHECK-NEXT:       ValueKind:     HiddenGlobalOffsetX
; CHECK-NEXT:       ValueType:     I64
; CHECK-NEXT:     - Size:          8
; CHECK-NEXT:       Align:         8
; CHECK-NEXT:       ValueKind:     HiddenGlobalOffsetY
; CHECK-NEXT:       ValueType:     I64
; CHECK-NEXT:     - Size:          8
; CHECK-NEXT:       Align:         8
; CHECK-NEXT:       ValueKind:     HiddenGlobalOffsetZ
; CHECK-NEXT:       ValueType:     I64
; CHECK-NEXT:     - Size:          8
; CHECK-NEXT:       Align:         8
; CHECK-NEXT:       ValueKind:     HiddenPrintfBuffer
; CHECK-NEXT:       ValueType:     I8
; CHECK-NEXT:       AddrSpaceQual: Global
define amdgpu_kernel void @test_double16(<16 x double> %a) #0
    !kernel_arg_addr_space !1 !kernel_arg_access_qual !2 !kernel_arg_type !15
    !kernel_arg_base_type !15 !kernel_arg_type_qual !4 {
  ret void
}

; CHECK:      - Name:            test_pointer
; CHECK-NEXT:   SymbolName:      'test_pointer@kd'
; CHECK-NEXT:   Language:        OpenCL C
; CHECK-NEXT:   LanguageVersion: [ 2, 0 ]
; CHECK-NEXT:   Args:
; CHECK-NEXT:     - Name:          a
; CHECK-NEXT:       TypeName:      'int  addrspace(5)*'
; CHECK-NEXT:       Size:          8
; CHECK-NEXT:       Align:         8
; CHECK-NEXT:       ValueKind:     GlobalBuffer
; CHECK-NEXT:       ValueType:     I32
; CHECK-NEXT:       AddrSpaceQual: Global
; CHECK-NEXT:       AccQual:       Default
; CHECK-NEXT:     - Size:          8
; CHECK-NEXT:       Align:         8
; CHECK-NEXT:       ValueKind:     HiddenGlobalOffsetX
; CHECK-NEXT:       ValueType:     I64
; CHECK-NEXT:     - Size:          8
; CHECK-NEXT:       Align:         8
; CHECK-NEXT:       ValueKind:     HiddenGlobalOffsetY
; CHECK-NEXT:       ValueType:     I64
; CHECK-NEXT:     - Size:          8
; CHECK-NEXT:       Align:         8
; CHECK-NEXT:       ValueKind:     HiddenGlobalOffsetZ
; CHECK-NEXT:       ValueType:     I64
; CHECK-NEXT:     - Size:          8
; CHECK-NEXT:       Align:         8
; CHECK-NEXT:       ValueKind:     HiddenPrintfBuffer
; CHECK-NEXT:       ValueType:     I8
; CHECK-NEXT:       AddrSpaceQual: Global
define amdgpu_kernel void @test_pointer(i32 addrspace(1)* %a) #0
    !kernel_arg_addr_space !1 !kernel_arg_access_qual !2 !kernel_arg_type !16
    !kernel_arg_base_type !16 !kernel_arg_type_qual !4 {
  ret void
}

; CHECK:      - Name:            test_image
; CHECK-NEXT:   SymbolName:      'test_image@kd'
; CHECK-NEXT:   Language:        OpenCL C
; CHECK-NEXT:   LanguageVersion: [ 2, 0 ]
; CHECK-NEXT:   Args:
; CHECK-NEXT:     - Name:          a
; CHECK-NEXT:       TypeName:      image2d_t
; CHECK-NEXT:       Size:          8
; CHECK-NEXT:       Align:         8
; CHECK-NEXT:       ValueKind:     Image
; CHECK-NEXT:       ValueType:     Struct
; CHECK-NEXT:       AddrSpaceQual: Global
; CHECK-NEXT:       AccQual:       Default
; CHECK-NEXT:     - Size:          8
; CHECK-NEXT:       Align:         8
; CHECK-NEXT:       ValueKind:     HiddenGlobalOffsetX
; CHECK-NEXT:       ValueType:     I64
; CHECK-NEXT:     - Size:          8
; CHECK-NEXT:       Align:         8
; CHECK-NEXT:       ValueKind:     HiddenGlobalOffsetY
; CHECK-NEXT:       ValueType:     I64
; CHECK-NEXT:     - Size:          8
; CHECK-NEXT:       Align:         8
; CHECK-NEXT:       ValueKind:     HiddenGlobalOffsetZ
; CHECK-NEXT:       ValueType:     I64
; CHECK-NEXT:     - Size:          8
; CHECK-NEXT:       Align:         8
; CHECK-NEXT:       ValueKind:     HiddenPrintfBuffer
; CHECK-NEXT:       ValueType:     I8
; CHECK-NEXT:       AddrSpaceQual: Global
define amdgpu_kernel void @test_image(%opencl.image2d_t addrspace(1)* %a) #0
    !kernel_arg_addr_space !1 !kernel_arg_access_qual !2 !kernel_arg_type !17
    !kernel_arg_base_type !17 !kernel_arg_type_qual !4 {
  ret void
}

; CHECK:      - Name:            test_sampler
; CHECK-NEXT:   SymbolName:      'test_sampler@kd'
; CHECK-NEXT:   Language:        OpenCL C
; CHECK-NEXT:   LanguageVersion: [ 2, 0 ]
; CHECK-NEXT:   Args:
; CHECK-NEXT:     - Name:          a
; CHECK-NEXT:       TypeName:      sampler_t
; CHECK-NEXT:       Size:          4
; CHECK-NEXT:       Align:         4
; CHECK-NEXT:       ValueKind:     Sampler
; CHECK-NEXT:       ValueType:     I32
; CHECK-NEXT:       AccQual:       Default
; CHECK-NEXT:     - Size:          8
; CHECK-NEXT:       Align:         8
; CHECK-NEXT:       ValueKind:     HiddenGlobalOffsetX
; CHECK-NEXT:       ValueType:     I64
; CHECK-NEXT:     - Size:          8
; CHECK-NEXT:       Align:         8
; CHECK-NEXT:       ValueKind:     HiddenGlobalOffsetY
; CHECK-NEXT:       ValueType:     I64
; CHECK-NEXT:     - Size:          8
; CHECK-NEXT:       Align:         8
; CHECK-NEXT:       ValueKind:     HiddenGlobalOffsetZ
; CHECK-NEXT:       ValueType:     I64
; CHECK-NEXT:     - Size:          8
; CHECK-NEXT:       Align:         8
; CHECK-NEXT:       ValueKind:     HiddenPrintfBuffer
; CHECK-NEXT:       ValueType:     I8
; CHECK-NEXT:       AddrSpaceQual: Global
define amdgpu_kernel void @test_sampler(i32 %a) #0
    !kernel_arg_addr_space !1 !kernel_arg_access_qual !2 !kernel_arg_type !18
    !kernel_arg_base_type !18 !kernel_arg_type_qual !4 {
  ret void
}

; CHECK:      - Name:            test_queue
; CHECK-NEXT:   SymbolName:      'test_queue@kd'
; CHECK-NEXT:   Language:        OpenCL C
; CHECK-NEXT:   LanguageVersion: [ 2, 0 ]
; CHECK-NEXT:   Args:
; CHECK-NEXT:     - Name:          a
; CHECK-NEXT:       TypeName:      queue_t
; CHECK-NEXT:       Size:          8
; CHECK-NEXT:       Align:         8
; CHECK-NEXT:       ValueKind:     Queue
; CHECK-NEXT:       ValueType:     Struct
; CHECK-NEXT:       AddrSpaceQual: Global
; CHECK-NEXT:       AccQual:       Default
; CHECK-NEXT:     - Size:          8
; CHECK-NEXT:       Align:         8
; CHECK-NEXT:       ValueKind:     HiddenGlobalOffsetX
; CHECK-NEXT:       ValueType:     I64
; CHECK-NEXT:     - Size:          8
; CHECK-NEXT:       Align:         8
; CHECK-NEXT:       ValueKind:     HiddenGlobalOffsetY
; CHECK-NEXT:       ValueType:     I64
; CHECK-NEXT:     - Size:          8
; CHECK-NEXT:       Align:         8
; CHECK-NEXT:       ValueKind:     HiddenGlobalOffsetZ
; CHECK-NEXT:       ValueType:     I64
; CHECK-NEXT:     - Size:          8
; CHECK-NEXT:       Align:         8
; CHECK-NEXT:       ValueKind:     HiddenPrintfBuffer
; CHECK-NEXT:       ValueType:     I8
; CHECK-NEXT:       AddrSpaceQual: Global
define amdgpu_kernel void @test_queue(%opencl.queue_t addrspace(1)* %a) #0
    !kernel_arg_addr_space !1 !kernel_arg_access_qual !2 !kernel_arg_type !19
    !kernel_arg_base_type !19 !kernel_arg_type_qual !4 {
  ret void
}

; CHECK:      - Name:            test_struct
; CHECK-NEXT:   SymbolName:      'test_struct@kd'
; CHECK-NEXT:   Language:        OpenCL C
; CHECK-NEXT:   LanguageVersion: [ 2, 0 ]
; CHECK-NEXT:   Args:
; CHECK-NEXT:     - Name:          a
; CHECK-NEXT:       TypeName:      struct A
; CHECK-NEXT:       Size:          4
; CHECK-NEXT:       Align:         4
; CHECK-NEXT:       ValueKind:     GlobalBuffer
; CHECK-NEXT:       ValueType:     Struct
; CHECK-NEXT:       AddrSpaceQual: Private
; CHECK-NEXT:       AccQual:       Default
; CHECK-NEXT:     - Size:          8
; CHECK-NEXT:       Align:         8
; CHECK-NEXT:       ValueKind:     HiddenGlobalOffsetX
; CHECK-NEXT:       ValueType:     I64
; CHECK-NEXT:     - Size:          8
; CHECK-NEXT:       Align:         8
; CHECK-NEXT:       ValueKind:     HiddenGlobalOffsetY
; CHECK-NEXT:       ValueType:     I64
; CHECK-NEXT:     - Size:          8
; CHECK-NEXT:       Align:         8
; CHECK-NEXT:       ValueKind:     HiddenGlobalOffsetZ
; CHECK-NEXT:       ValueType:     I64
; CHECK-NEXT:     - Size:          8
; CHECK-NEXT:       Align:         8
; CHECK-NEXT:       ValueKind:     HiddenPrintfBuffer
; CHECK-NEXT:       ValueType:     I8
; CHECK-NEXT:       AddrSpaceQual: Global
define amdgpu_kernel void @test_struct(%struct.A addrspace(5)* byval %a) #0
    !kernel_arg_addr_space !1 !kernel_arg_access_qual !2 !kernel_arg_type !20
    !kernel_arg_base_type !20 !kernel_arg_type_qual !4 {
  ret void
}

; CHECK:      - Name:            test_i128
; CHECK-NEXT:   SymbolName:      'test_i128@kd'
; CHECK-NEXT:   Language:        OpenCL C
; CHECK-NEXT:   LanguageVersion: [ 2, 0 ]
; CHECK-NEXT:   Args:
; CHECK-NEXT:     - Name:          a
; CHECK-NEXT:       TypeName:      i128
; CHECK-NEXT:       Size:          16
; CHECK-NEXT:       Align:         8
; CHECK-NEXT:       ValueKind:     ByValue
; CHECK-NEXT:       ValueType:     Struct
; CHECK-NEXT:       AccQual:       Default
; CHECK-NEXT:     - Size:          8
; CHECK-NEXT:       Align:         8
; CHECK-NEXT:       ValueKind:     HiddenGlobalOffsetX
; CHECK-NEXT:       ValueType:     I64
; CHECK-NEXT:     - Size:          8
; CHECK-NEXT:       Align:         8
; CHECK-NEXT:       ValueKind:     HiddenGlobalOffsetY
; CHECK-NEXT:       ValueType:     I64
; CHECK-NEXT:     - Size:          8
; CHECK-NEXT:       Align:         8
; CHECK-NEXT:       ValueKind:     HiddenGlobalOffsetZ
; CHECK-NEXT:       ValueType:     I64
; CHECK-NEXT:     - Size:          8
; CHECK-NEXT:       Align:         8
; CHECK-NEXT:       ValueKind:     HiddenPrintfBuffer
; CHECK-NEXT:       ValueType:     I8
; CHECK-NEXT:       AddrSpaceQual: Global
define amdgpu_kernel void @test_i128(i128 %a) #0
    !kernel_arg_addr_space !1 !kernel_arg_access_qual !2 !kernel_arg_type !21
    !kernel_arg_base_type !21 !kernel_arg_type_qual !4 {
  ret void
}

; CHECK:      - Name:            test_multi_arg
; CHECK-NEXT:   SymbolName:      'test_multi_arg@kd'
; CHECK-NEXT:   Language:        OpenCL C
; CHECK-NEXT:   LanguageVersion: [ 2, 0 ]
; CHECK-NEXT:   Args:
; CHECK-NEXT:     - Name:          a
; CHECK-NEXT:       TypeName:      int
; CHECK-NEXT:       Size:          4
; CHECK-NEXT:       Align:         4
; CHECK-NEXT:       ValueKind:     ByValue
; CHECK-NEXT:       ValueType:     I32
; CHECK-NEXT:       AccQual:       Default
; CHECK-NEXT:     - Name:          b
; CHECK-NEXT:       TypeName:      short2
; CHECK-NEXT:       Size:          4
; CHECK-NEXT:       Align:         4
; CHECK-NEXT:       ValueKind:     ByValue
; CHECK-NEXT:       ValueType:     I16
; CHECK-NEXT:       AccQual:       Default
; CHECK-NEXT:     - Name:          c
; CHECK-NEXT:       TypeName:      char3
; CHECK-NEXT:       Size:          4
; CHECK-NEXT:       Align:         4
; CHECK-NEXT:       ValueKind:     ByValue
; CHECK-NEXT:       ValueType:     I8
; CHECK-NEXT:       AccQual:       Default
; CHECK-NEXT:     - Size:          8
; CHECK-NEXT:       Align:         8
; CHECK-NEXT:       ValueKind:     HiddenGlobalOffsetX
; CHECK-NEXT:       ValueType:     I64
; CHECK-NEXT:     - Size:          8
; CHECK-NEXT:       Align:         8
; CHECK-NEXT:       ValueKind:     HiddenGlobalOffsetY
; CHECK-NEXT:       ValueType:     I64
; CHECK-NEXT:     - Size:          8
; CHECK-NEXT:       Align:         8
; CHECK-NEXT:       ValueKind:     HiddenGlobalOffsetZ
; CHECK-NEXT:       ValueType:     I64
; CHECK-NEXT:     - Size:          8
; CHECK-NEXT:       Align:         8
; CHECK-NEXT:       ValueKind:     HiddenPrintfBuffer
; CHECK-NEXT:       ValueType:     I8
; CHECK-NEXT:       AddrSpaceQual: Global
define amdgpu_kernel void @test_multi_arg(i32 %a, <2 x i16> %b, <3 x i8> %c) #0
    !kernel_arg_addr_space !22 !kernel_arg_access_qual !23 !kernel_arg_type !24
    !kernel_arg_base_type !24 !kernel_arg_type_qual !25 {
  ret void
}

; CHECK:      - Name:            test_addr_space
; CHECK-NEXT:   SymbolName:      'test_addr_space@kd'
; CHECK-NEXT:   Language:        OpenCL C
; CHECK-NEXT:   LanguageVersion: [ 2, 0 ]
; CHECK-NEXT:   Args:
; CHECK-NEXT:     - Name:          g
; CHECK-NEXT:       TypeName:      'int  addrspace(5)*'
; CHECK-NEXT:       Size:          8
; CHECK-NEXT:       Align:         8
; CHECK-NEXT:       ValueKind:     GlobalBuffer
; CHECK-NEXT:       ValueType:     I32
; CHECK-NEXT:       AddrSpaceQual: Global
; CHECK-NEXT:       AccQual:       Default
; CHECK-NEXT:     - Name:          c
; CHECK-NEXT:       TypeName:      'int  addrspace(5)*'
; CHECK-NEXT:       Size:          8
; CHECK-NEXT:       Align:         8
; CHECK-NEXT:       ValueKind:     GlobalBuffer
; CHECK-NEXT:       ValueType:     I32
; CHECK-NEXT:       AddrSpaceQual: Constant
; CHECK-NEXT:       AccQual:       Default
; CHECK-NEXT:     - Name:          l
; CHECK-NEXT:       TypeName:      'int  addrspace(5)*'
; CHECK-NEXT:       Size:          4
; CHECK-NEXT:       Align:         4
; CHECK-NEXT:       ValueKind:     DynamicSharedPointer
; CHECK-NEXT:       ValueType:     I32
; CHECK-NEXT:       PointeeAlign:  4
; CHECK-NEXT:       AddrSpaceQual: Local
; CHECK-NEXT:       AccQual:       Default
; CHECK-NEXT:     - Size:          8
; CHECK-NEXT:       Align:         8
; CHECK-NEXT:       ValueKind:     HiddenGlobalOffsetX
; CHECK-NEXT:       ValueType:     I64
; CHECK-NEXT:     - Size:          8
; CHECK-NEXT:       Align:         8
; CHECK-NEXT:       ValueKind:     HiddenGlobalOffsetY
; CHECK-NEXT:       ValueType:     I64
; CHECK-NEXT:     - Size:          8
; CHECK-NEXT:       Align:         8
; CHECK-NEXT:       ValueKind:     HiddenGlobalOffsetZ
; CHECK-NEXT:       ValueType:     I64
; CHECK-NEXT:     - Size:          8
; CHECK-NEXT:       Align:         8
; CHECK-NEXT:       ValueKind:     HiddenPrintfBuffer
; CHECK-NEXT:       ValueType:     I8
; CHECK-NEXT:       AddrSpaceQual: Global
define amdgpu_kernel void @test_addr_space(i32 addrspace(1)* %g,
                                           i32 addrspace(4)* %c,
                                           i32 addrspace(3)* %l) #0
    !kernel_arg_addr_space !50 !kernel_arg_access_qual !23 !kernel_arg_type !51
    !kernel_arg_base_type !51 !kernel_arg_type_qual !25 {
  ret void
}

; CHECK:      - Name:            test_type_qual
; CHECK-NEXT:   SymbolName:      'test_type_qual@kd'
; CHECK-NEXT:   Language:        OpenCL C
; CHECK-NEXT:   LanguageVersion: [ 2, 0 ]
; CHECK-NEXT:   Args:
; CHECK-NEXT:     - Name:          a
; CHECK-NEXT:       TypeName:      'int  addrspace(5)*'
; CHECK-NEXT:       Size:          8
; CHECK-NEXT:       Align:         8
; CHECK-NEXT:       ValueKind:     GlobalBuffer
; CHECK-NEXT:       ValueType:     I32
; CHECK-NEXT:       AddrSpaceQual: Global
; CHECK-NEXT:       AccQual:       Default
; CHECK-NEXT:       IsVolatile:    true
; CHECK-NEXT:     - Name:          b
; CHECK-NEXT:       TypeName:      'int  addrspace(5)*'
; CHECK-NEXT:       Size:          8
; CHECK-NEXT:       Align:         8
; CHECK-NEXT:       ValueKind:     GlobalBuffer
; CHECK-NEXT:       ValueType:     I32
; CHECK-NEXT:       AddrSpaceQual: Global
; CHECK-NEXT:       AccQual:       Default
; CHECK-NEXT:       IsConst:       true
; CHECK-NEXT:       IsRestrict:    true
; CHECK-NEXT:     - Name:          c
; CHECK-NEXT:       TypeName:      'int  addrspace(5)*'
; CHECK-NEXT:       Size:          8
; CHECK-NEXT:       Align:         8
; CHECK-NEXT:       ValueKind:     Pipe
; CHECK-NEXT:       ValueType:     Struct
; CHECK-NEXT:       AddrSpaceQual: Global
; CHECK-NEXT:       AccQual:       Default
; CHECK-NEXT:       IsPipe:        true
; CHECK-NEXT:     - Size:          8
; CHECK-NEXT:       Align:         8
; CHECK-NEXT:       ValueKind:     HiddenGlobalOffsetX
; CHECK-NEXT:       ValueType:     I64
; CHECK-NEXT:     - Size:          8
; CHECK-NEXT:       Align:         8
; CHECK-NEXT:       ValueKind:     HiddenGlobalOffsetY
; CHECK-NEXT:       ValueType:     I64
; CHECK-NEXT:     - Size:          8
; CHECK-NEXT:       Align:         8
; CHECK-NEXT:       ValueKind:     HiddenGlobalOffsetZ
; CHECK-NEXT:       ValueType:     I64
; CHECK-NEXT:     - Size:          8
; CHECK-NEXT:       Align:         8
; CHECK-NEXT:       ValueKind:     HiddenPrintfBuffer
; CHECK-NEXT:       ValueType:     I8
; CHECK-NEXT:       AddrSpaceQual: Global
define amdgpu_kernel void @test_type_qual(i32 addrspace(1)* %a,
                                          i32 addrspace(1)* %b,
                                          %opencl.pipe_t addrspace(1)* %c) #0
    !kernel_arg_addr_space !22 !kernel_arg_access_qual !23 !kernel_arg_type !51
    !kernel_arg_base_type !51 !kernel_arg_type_qual !70 {
  ret void
}

; CHECK:      - Name:            test_access_qual
; CHECK-NEXT:   SymbolName:      'test_access_qual@kd'
; CHECK-NEXT:   Language:        OpenCL C
; CHECK-NEXT:   LanguageVersion: [ 2, 0 ]
; CHECK-NEXT:   Args:
; CHECK-NEXT:     - Name:          ro
; CHECK-NEXT:       TypeName:      image1d_t
; CHECK-NEXT:       Size:          8
; CHECK-NEXT:       Align:         8
; CHECK-NEXT:       ValueKind:     Image
; CHECK-NEXT:       ValueType:     Struct
; CHECK-NEXT:       AddrSpaceQual: Global
; CHECK-NEXT:       AccQual:       ReadOnly
; CHECK-NEXT:     - Name:          wo
; CHECK-NEXT:       TypeName:      image2d_t
; CHECK-NEXT:       Size:          8
; CHECK-NEXT:       Align:         8
; CHECK-NEXT:       ValueKind:     Image
; CHECK-NEXT:       ValueType:     Struct
; CHECK-NEXT:       AddrSpaceQual: Global
; CHECK-NEXT:       AccQual:       WriteOnly
; CHECK-NEXT:     - Name:          rw
; CHECK-NEXT:       TypeName:      image3d_t
; CHECK-NEXT:       Size:          8
; CHECK-NEXT:       Align:         8
; CHECK-NEXT:       ValueKind:     Image
; CHECK-NEXT:       ValueType:     Struct
; CHECK-NEXT:       AddrSpaceQual: Global
; CHECK-NEXT:       AccQual:       ReadWrite
; CHECK-NEXT:     - Size:          8
; CHECK-NEXT:       Align:         8
; CHECK-NEXT:       ValueKind:     HiddenGlobalOffsetX
; CHECK-NEXT:       ValueType:     I64
; CHECK-NEXT:     - Size:          8
; CHECK-NEXT:       Align:         8
; CHECK-NEXT:       ValueKind:     HiddenGlobalOffsetY
; CHECK-NEXT:       ValueType:     I64
; CHECK-NEXT:     - Size:          8
; CHECK-NEXT:       Align:         8
; CHECK-NEXT:       ValueKind:     HiddenGlobalOffsetZ
; CHECK-NEXT:       ValueType:     I64
; CHECK-NEXT:     - Size:          8
; CHECK-NEXT:       Align:         8
; CHECK-NEXT:       ValueKind:     HiddenPrintfBuffer
; CHECK-NEXT:       ValueType:     I8
; CHECK-NEXT:       AddrSpaceQual: Global
define amdgpu_kernel void @test_access_qual(%opencl.image1d_t addrspace(1)* %ro,
                                            %opencl.image2d_t addrspace(1)* %wo,
                                            %opencl.image3d_t addrspace(1)* %rw) #0
    !kernel_arg_addr_space !60 !kernel_arg_access_qual !61 !kernel_arg_type !62
    !kernel_arg_base_type !62 !kernel_arg_type_qual !25 {
  ret void
}

; CHECK:      - Name:            test_vec_type_hint_half
; CHECK-NEXT:   SymbolName:      'test_vec_type_hint_half@kd'
; CHECK-NEXT:   Language:        OpenCL C
; CHECK-NEXT:   LanguageVersion: [ 2, 0 ]
; CHECK-NEXT:   Attrs:
; CHECK-NEXT:       VecTypeHint:   half
; CHECK-NEXT:   Args:
; CHECK-NEXT:     - Name:          a
; CHECK-NEXT:       TypeName:      int
; CHECK-NEXT:       Size:          4
; CHECK-NEXT:       Align:         4
; CHECK-NEXT:       ValueKind:     ByValue
; CHECK-NEXT:       ValueType:     I32
; CHECK-NEXT:       AccQual:       Default
; CHECK-NEXT:     - Size:          8
; CHECK-NEXT:       Align:         8
; CHECK-NEXT:       ValueKind:     HiddenGlobalOffsetX
; CHECK-NEXT:       ValueType:     I64
; CHECK-NEXT:     - Size:          8
; CHECK-NEXT:       Align:         8
; CHECK-NEXT:       ValueKind:     HiddenGlobalOffsetY
; CHECK-NEXT:       ValueType:     I64
; CHECK-NEXT:     - Size:          8
; CHECK-NEXT:       Align:         8
; CHECK-NEXT:       ValueKind:     HiddenGlobalOffsetZ
; CHECK-NEXT:       ValueType:     I64
; CHECK-NEXT:     - Size:          8
; CHECK-NEXT:       Align:         8
; CHECK-NEXT:       ValueKind:     HiddenPrintfBuffer
; CHECK-NEXT:       ValueType:     I8
; CHECK-NEXT:       AddrSpaceQual: Global
define amdgpu_kernel void @test_vec_type_hint_half(i32 %a) #0
    !kernel_arg_addr_space !1 !kernel_arg_access_qual !2 !kernel_arg_type !3
    !kernel_arg_base_type !3 !kernel_arg_type_qual !4 !vec_type_hint !26 {
  ret void
}

; CHECK:      - Name:            test_vec_type_hint_float
; CHECK-NEXT:   SymbolName:      'test_vec_type_hint_float@kd'
; CHECK-NEXT:   Language:        OpenCL C
; CHECK-NEXT:   LanguageVersion: [ 2, 0 ]
; CHECK-NEXT:   Attrs:
; CHECK-NEXT:       VecTypeHint:   float
; CHECK-NEXT:   Args:
; CHECK-NEXT:     - Name:          a
; CHECK-NEXT:       TypeName:      int
; CHECK-NEXT:       Size:          4
; CHECK-NEXT:       Align:         4
; CHECK-NEXT:       ValueKind:     ByValue
; CHECK-NEXT:       ValueType:     I32
; CHECK-NEXT:       AccQual:       Default
; CHECK-NEXT:     - Size:          8
; CHECK-NEXT:       Align:         8
; CHECK-NEXT:       ValueKind:     HiddenGlobalOffsetX
; CHECK-NEXT:       ValueType:     I64
; CHECK-NEXT:     - Size:          8
; CHECK-NEXT:       Align:         8
; CHECK-NEXT:       ValueKind:     HiddenGlobalOffsetY
; CHECK-NEXT:       ValueType:     I64
; CHECK-NEXT:     - Size:          8
; CHECK-NEXT:       Align:         8
; CHECK-NEXT:       ValueKind:     HiddenGlobalOffsetZ
; CHECK-NEXT:       ValueType:     I64
; CHECK-NEXT:     - Size:          8
; CHECK-NEXT:       Align:         8
; CHECK-NEXT:       ValueKind:     HiddenPrintfBuffer
; CHECK-NEXT:       ValueType:     I8
; CHECK-NEXT:       AddrSpaceQual: Global
define amdgpu_kernel void @test_vec_type_hint_float(i32 %a) #0
    !kernel_arg_addr_space !1 !kernel_arg_access_qual !2 !kernel_arg_type !3
    !kernel_arg_base_type !3 !kernel_arg_type_qual !4 !vec_type_hint !27 {
  ret void
}

; CHECK:      - Name:            test_vec_type_hint_double
; CHECK-NEXT:   SymbolName:      'test_vec_type_hint_double@kd'
; CHECK-NEXT:   Language:        OpenCL C
; CHECK-NEXT:   LanguageVersion: [ 2, 0 ]
; CHECK-NEXT:   Attrs:
; CHECK-NEXT:       VecTypeHint:   double
; CHECK-NEXT:   Args:
; CHECK-NEXT:     - Name:          a
; CHECK-NEXT:       TypeName:      int
; CHECK-NEXT:       Size:          4
; CHECK-NEXT:       Align:         4
; CHECK-NEXT:       ValueKind:     ByValue
; CHECK-NEXT:       ValueType:     I32
; CHECK-NEXT:       AccQual:       Default
; CHECK-NEXT:     - Size:          8
; CHECK-NEXT:       Align:         8
; CHECK-NEXT:       ValueKind:     HiddenGlobalOffsetX
; CHECK-NEXT:       ValueType:     I64
; CHECK-NEXT:     - Size:          8
; CHECK-NEXT:       Align:         8
; CHECK-NEXT:       ValueKind:     HiddenGlobalOffsetY
; CHECK-NEXT:       ValueType:     I64
; CHECK-NEXT:     - Size:          8
; CHECK-NEXT:       Align:         8
; CHECK-NEXT:       ValueKind:     HiddenGlobalOffsetZ
; CHECK-NEXT:       ValueType:     I64
; CHECK-NEXT:     - Size:          8
; CHECK-NEXT:       Align:         8
; CHECK-NEXT:       ValueKind:     HiddenPrintfBuffer
; CHECK-NEXT:       ValueType:     I8
; CHECK-NEXT:       AddrSpaceQual: Global
define amdgpu_kernel void @test_vec_type_hint_double(i32 %a) #0
    !kernel_arg_addr_space !1 !kernel_arg_access_qual !2 !kernel_arg_type !3
    !kernel_arg_base_type !3 !kernel_arg_type_qual !4 !vec_type_hint !28 {
  ret void
}

; CHECK:      - Name:            test_vec_type_hint_char
; CHECK-NEXT:   SymbolName:      'test_vec_type_hint_char@kd'
; CHECK-NEXT:   Language:        OpenCL C
; CHECK-NEXT:   LanguageVersion: [ 2, 0 ]
; CHECK-NEXT:   Attrs:
; CHECK-NEXT:       VecTypeHint:   char
; CHECK-NEXT:   Args:
; CHECK-NEXT:     - Name:          a
; CHECK-NEXT:       TypeName:      int
; CHECK-NEXT:       Size:          4
; CHECK-NEXT:       Align:         4
; CHECK-NEXT:       ValueKind:     ByValue
; CHECK-NEXT:       ValueType:     I32
; CHECK-NEXT:       AccQual:       Default
; CHECK-NEXT:     - Size:          8
; CHECK-NEXT:       Align:         8
; CHECK-NEXT:       ValueKind:     HiddenGlobalOffsetX
; CHECK-NEXT:       ValueType:     I64
; CHECK-NEXT:     - Size:          8
; CHECK-NEXT:       Align:         8
; CHECK-NEXT:       ValueKind:     HiddenGlobalOffsetY
; CHECK-NEXT:       ValueType:     I64
; CHECK-NEXT:     - Size:          8
; CHECK-NEXT:       Align:         8
; CHECK-NEXT:       ValueKind:     HiddenGlobalOffsetZ
; CHECK-NEXT:       ValueType:     I64
; CHECK-NEXT:     - Size:          8
; CHECK-NEXT:       Align:         8
; CHECK-NEXT:       ValueKind:     HiddenPrintfBuffer
; CHECK-NEXT:       ValueType:     I8
; CHECK-NEXT:       AddrSpaceQual: Global
define amdgpu_kernel void @test_vec_type_hint_char(i32 %a) #0
    !kernel_arg_addr_space !1 !kernel_arg_access_qual !2 !kernel_arg_type !3
    !kernel_arg_base_type !3 !kernel_arg_type_qual !4 !vec_type_hint !29 {
  ret void
}

; CHECK:      - Name:            test_vec_type_hint_short
; CHECK-NEXT:   SymbolName:      'test_vec_type_hint_short@kd'
; CHECK-NEXT:   Language:        OpenCL C
; CHECK-NEXT:   LanguageVersion: [ 2, 0 ]
; CHECK-NEXT:   Attrs:
; CHECK-NEXT:       VecTypeHint:   short
; CHECK-NEXT:   Args:
; CHECK-NEXT:     - Name:          a
; CHECK-NEXT:       TypeName:      int
; CHECK-NEXT:       Size:          4
; CHECK-NEXT:       Align:         4
; CHECK-NEXT:       ValueKind:     ByValue
; CHECK-NEXT:       ValueType:     I32
; CHECK-NEXT:       AccQual:       Default
; CHECK-NEXT:     - Size:          8
; CHECK-NEXT:       Align:         8
; CHECK-NEXT:       ValueKind:     HiddenGlobalOffsetX
; CHECK-NEXT:       ValueType:     I64
; CHECK-NEXT:     - Size:          8
; CHECK-NEXT:       Align:         8
; CHECK-NEXT:       ValueKind:     HiddenGlobalOffsetY
; CHECK-NEXT:       ValueType:     I64
; CHECK-NEXT:     - Size:          8
; CHECK-NEXT:       Align:         8
; CHECK-NEXT:       ValueKind:     HiddenGlobalOffsetZ
; CHECK-NEXT:       ValueType:     I64
; CHECK-NEXT:     - Size:          8
; CHECK-NEXT:       Align:         8
; CHECK-NEXT:       ValueKind:     HiddenPrintfBuffer
; CHECK-NEXT:       ValueType:     I8
; CHECK-NEXT:       AddrSpaceQual: Global
define amdgpu_kernel void @test_vec_type_hint_short(i32 %a) #0
    !kernel_arg_addr_space !1 !kernel_arg_access_qual !2 !kernel_arg_type !3
    !kernel_arg_base_type !3 !kernel_arg_type_qual !4 !vec_type_hint !30 {
  ret void
}

; CHECK:      - Name:            test_vec_type_hint_long
; CHECK-NEXT:   SymbolName:      'test_vec_type_hint_long@kd'
; CHECK-NEXT:   Language:        OpenCL C
; CHECK-NEXT:   LanguageVersion: [ 2, 0 ]
; CHECK-NEXT:   Attrs:
; CHECK-NEXT:       VecTypeHint:   long
; CHECK-NEXT:   Args:
; CHECK-NEXT:     - Name:          a
; CHECK-NEXT:       TypeName:      int
; CHECK-NEXT:       Size:          4
; CHECK-NEXT:       Align:         4
; CHECK-NEXT:       ValueKind:     ByValue
; CHECK-NEXT:       ValueType:     I32
; CHECK-NEXT:       AccQual:       Default
; CHECK-NEXT:     - Size:          8
; CHECK-NEXT:       Align:         8
; CHECK-NEXT:       ValueKind:     HiddenGlobalOffsetX
; CHECK-NEXT:       ValueType:     I64
; CHECK-NEXT:     - Size:          8
; CHECK-NEXT:       Align:         8
; CHECK-NEXT:       ValueKind:     HiddenGlobalOffsetY
; CHECK-NEXT:       ValueType:     I64
; CHECK-NEXT:     - Size:          8
; CHECK-NEXT:       Align:         8
; CHECK-NEXT:       ValueKind:     HiddenGlobalOffsetZ
; CHECK-NEXT:       ValueType:     I64
; CHECK-NEXT:     - Size:          8
; CHECK-NEXT:       Align:         8
; CHECK-NEXT:       ValueKind:     HiddenPrintfBuffer
; CHECK-NEXT:       ValueType:     I8
; CHECK-NEXT:       AddrSpaceQual: Global
define amdgpu_kernel void @test_vec_type_hint_long(i32 %a) #0
    !kernel_arg_addr_space !1 !kernel_arg_access_qual !2 !kernel_arg_type !3
    !kernel_arg_base_type !3 !kernel_arg_type_qual !4 !vec_type_hint !31 {
  ret void
}

; CHECK:      - Name:            test_vec_type_hint_unknown
; CHECK-NEXT:   SymbolName:      'test_vec_type_hint_unknown@kd'
; CHECK-NEXT:   Language:        OpenCL C
; CHECK-NEXT:   LanguageVersion: [ 2, 0 ]
; CHECK-NEXT:   Attrs:
; CHECK-NEXT:       VecTypeHint:   unknown
; CHECK-NEXT:   Args:
; CHECK-NEXT:     - Name:          a
; CHECK-NEXT:       TypeName:      int
; CHECK-NEXT:       Size:          4
; CHECK-NEXT:       Align:         4
; CHECK-NEXT:       ValueKind:     ByValue
; CHECK-NEXT:       ValueType:     I32
; CHECK-NEXT:       AccQual:       Default
; CHECK-NEXT:     - Size:          8
; CHECK-NEXT:       Align:         8
; CHECK-NEXT:       ValueKind:     HiddenGlobalOffsetX
; CHECK-NEXT:       ValueType:     I64
; CHECK-NEXT:     - Size:          8
; CHECK-NEXT:       Align:         8
; CHECK-NEXT:       ValueKind:     HiddenGlobalOffsetY
; CHECK-NEXT:       ValueType:     I64
; CHECK-NEXT:     - Size:          8
; CHECK-NEXT:       Align:         8
; CHECK-NEXT:       ValueKind:     HiddenGlobalOffsetZ
; CHECK-NEXT:       ValueType:     I64
; CHECK-NEXT:     - Size:          8
; CHECK-NEXT:       Align:         8
; CHECK-NEXT:       ValueKind:     HiddenPrintfBuffer
; CHECK-NEXT:       ValueType:     I8
; CHECK-NEXT:       AddrSpaceQual: Global
define amdgpu_kernel void @test_vec_type_hint_unknown(i32 %a) #0
    !kernel_arg_addr_space !1 !kernel_arg_access_qual !2 !kernel_arg_type !3
    !kernel_arg_base_type !3 !kernel_arg_type_qual !4 !vec_type_hint !32 {
  ret void
}

; CHECK:      - Name:            test_reqd_wgs_vec_type_hint
; CHECK-NEXT:   SymbolName:      'test_reqd_wgs_vec_type_hint@kd'
; CHECK-NEXT:   Language:        OpenCL C
; CHECK-NEXT:   LanguageVersion: [ 2, 0 ]
; CHECK-NEXT:   Attrs:
; CHECK-NEXT:       ReqdWorkGroupSize: [ 1, 2, 4 ]
; CHECK-NEXT:       VecTypeHint:       int
; CHECK-NEXT:   Args:
; CHECK-NEXT:     - Name:              a
; CHECK-NEXT:       TypeName:          int
; CHECK-NEXT:       Size:              4
; CHECK-NEXT:       Align:             4
; CHECK-NEXT:       ValueKind:         ByValue
; CHECK-NEXT:       ValueType:         I32
; CHECK-NEXT:       AccQual:           Default
; CHECK-NEXT:     - Size:              8
; CHECK-NEXT:       Align:             8
; CHECK-NEXT:       ValueKind:         HiddenGlobalOffsetX
; CHECK-NEXT:       ValueType:         I64
; CHECK-NEXT:     - Size:              8
; CHECK-NEXT:       Align:             8
; CHECK-NEXT:       ValueKind:         HiddenGlobalOffsetY
; CHECK-NEXT:       ValueType:         I64
; CHECK-NEXT:     - Size:              8
; CHECK-NEXT:       Align:             8
; CHECK-NEXT:       ValueKind:         HiddenGlobalOffsetZ
; CHECK-NEXT:       ValueType:         I64
; CHECK-NEXT:     - Size:              8
; CHECK-NEXT:       Align:             8
; CHECK-NEXT:       ValueKind:         HiddenPrintfBuffer
; CHECK-NEXT:       ValueType:         I8
; CHECK-NEXT:       AddrSpaceQual:     Global
define amdgpu_kernel void @test_reqd_wgs_vec_type_hint(i32 %a) #0
    !kernel_arg_addr_space !1 !kernel_arg_access_qual !2 !kernel_arg_type !3
    !kernel_arg_base_type !3 !kernel_arg_type_qual !4 !vec_type_hint !5
    !reqd_work_group_size !6 {
  ret void
}

; CHECK:      - Name:            test_wgs_hint_vec_type_hint
; CHECK-NEXT:   SymbolName:      'test_wgs_hint_vec_type_hint@kd'
; CHECK-NEXT:   Language:        OpenCL C
; CHECK-NEXT:   LanguageVersion: [ 2, 0 ]
; CHECK-NEXT:   Attrs:
; CHECK-NEXT:       WorkGroupSizeHint: [ 8, 16, 32 ]
; CHECK-NEXT:       VecTypeHint:       uint4
; CHECK-NEXT:   Args:
; CHECK-NEXT:     - Name:              a
; CHECK-NEXT:       TypeName:          int
; CHECK-NEXT:       Size:              4
; CHECK-NEXT:       Align:             4
; CHECK-NEXT:       ValueKind:         ByValue
; CHECK-NEXT:       ValueType:         I32
; CHECK-NEXT:       AccQual:           Default
; CHECK-NEXT:     - Size:              8
; CHECK-NEXT:       Align:             8
; CHECK-NEXT:       ValueKind:         HiddenGlobalOffsetX
; CHECK-NEXT:       ValueType:         I64
; CHECK-NEXT:     - Size:              8
; CHECK-NEXT:       Align:             8
; CHECK-NEXT:       ValueKind:         HiddenGlobalOffsetY
; CHECK-NEXT:       ValueType:         I64
; CHECK-NEXT:     - Size:              8
; CHECK-NEXT:       Align:             8
; CHECK-NEXT:       ValueKind:         HiddenGlobalOffsetZ
; CHECK-NEXT:       ValueType:         I64
; CHECK-NEXT:     - Size:              8
; CHECK-NEXT:       Align:             8
; CHECK-NEXT:       ValueKind:         HiddenPrintfBuffer
; CHECK-NEXT:       ValueType:         I8
; CHECK-NEXT:       AddrSpaceQual:     Global
define amdgpu_kernel void @test_wgs_hint_vec_type_hint(i32 %a) #0
    !kernel_arg_addr_space !1 !kernel_arg_access_qual !2 !kernel_arg_type !3
    !kernel_arg_base_type !3 !kernel_arg_type_qual !4 !vec_type_hint !7
    !work_group_size_hint !8 {
  ret void
}

; CHECK:      - Name:            test_arg_ptr_to_ptr
; CHECK-NEXT:   SymbolName:      'test_arg_ptr_to_ptr@kd'
; CHECK-NEXT:   Language:        OpenCL C
; CHECK-NEXT:   LanguageVersion: [ 2, 0 ]
; CHECK-NEXT:   Args:
; CHECK-NEXT:     - Name:          a
; CHECK-NEXT:       TypeName:      'int  addrspace(5)* addrspace(5)*'
; CHECK-NEXT:       Size:          8
; CHECK-NEXT:       Align:         8
; CHECK-NEXT:       ValueKind:     GlobalBuffer
; CHECK-NEXT:       ValueType:     I32
; CHECK-NEXT:       AddrSpaceQual: Global
; CHECK-NEXT:       AccQual:       Default
; CHECK-NEXT:     - Size:          8
; CHECK-NEXT:       Align:         8
; CHECK-NEXT:       ValueKind:     HiddenGlobalOffsetX
; CHECK-NEXT:       ValueType:     I64
; CHECK-NEXT:     - Size:          8
; CHECK-NEXT:       Align:         8
; CHECK-NEXT:       ValueKind:     HiddenGlobalOffsetY
; CHECK-NEXT:       ValueType:     I64
; CHECK-NEXT:     - Size:          8
; CHECK-NEXT:       Align:         8
; CHECK-NEXT:       ValueKind:     HiddenGlobalOffsetZ
; CHECK-NEXT:       ValueType:     I64
; CHECK-NEXT:     - Size:          8
; CHECK-NEXT:       Align:         8
; CHECK-NEXT:       ValueKind:     HiddenPrintfBuffer
; CHECK-NEXT:       ValueType:     I8
; CHECK-NEXT:       AddrSpaceQual: Global
define amdgpu_kernel void @test_arg_ptr_to_ptr(i32 addrspace(5)* addrspace(1)* %a) #0
    !kernel_arg_addr_space !81 !kernel_arg_access_qual !2 !kernel_arg_type !80
    !kernel_arg_base_type !80 !kernel_arg_type_qual !4 {
  ret void
}

; CHECK:      - Name:            test_arg_struct_contains_ptr
; CHECK-NEXT:   SymbolName:      'test_arg_struct_contains_ptr@kd'
; CHECK-NEXT:   Language:        OpenCL C
; CHECK-NEXT:   LanguageVersion: [ 2, 0 ]
; CHECK-NEXT:   Args:
; CHECK-NEXT:     - Name:          a
; CHECK-NEXT:       TypeName:      struct B
; CHECK-NEXT:       Size:          4
; CHECK-NEXT:       Align:         4
; CHECK-NEXT:       ValueKind:     GlobalBuffer
; CHECK-NEXT:       ValueType:     Struct
; CHECK-NEXT:       AddrSpaceQual: Private
; CHECK-NEXT:       AccQual:       Default
; CHECK-NEXT:     - Size:          8
; CHECK-NEXT:       Align:         8
; CHECK-NEXT:       ValueKind:     HiddenGlobalOffsetX
; CHECK-NEXT:       ValueType:     I64
; CHECK-NEXT:     - Size:          8
; CHECK-NEXT:       Align:         8
; CHECK-NEXT:       ValueKind:     HiddenGlobalOffsetY
; CHECK-NEXT:       ValueType:     I64
; CHECK-NEXT:     - Size:          8
; CHECK-NEXT:       Align:         8
; CHECK-NEXT:       ValueKind:     HiddenGlobalOffsetZ
; CHECK-NEXT:       ValueType:     I64
; CHECK-NEXT:     - Size:          8
; CHECK-NEXT:       Align:         8
; CHECK-NEXT:       ValueKind:     HiddenPrintfBuffer
; CHECK-NEXT:       ValueType:     I8
; CHECK-NEXT:       AddrSpaceQual: Global
define amdgpu_kernel void @test_arg_struct_contains_ptr(%struct.B addrspace(5)* byval %a) #0
    !kernel_arg_addr_space !1 !kernel_arg_access_qual !2 !kernel_arg_type !82
    !kernel_arg_base_type !82 !kernel_arg_type_qual !4 {
 ret void
}

; CHECK:      - Name:            test_arg_vector_of_ptr
; CHECK-NEXT:   SymbolName:      'test_arg_vector_of_ptr@kd'
; CHECK-NEXT:   Language:        OpenCL C
; CHECK-NEXT:   LanguageVersion: [ 2, 0 ]
; CHECK-NEXT:   Args:
; CHECK-NEXT:     - Name:          a
; CHECK-NEXT:       TypeName:      'global int addrspace(5)* __attribute__((ext_vector_type(2)))'
; CHECK-NEXT:       Size:          16
; CHECK-NEXT:       Align:         16
; CHECK-NEXT:       ValueKind:     ByValue
; CHECK-NEXT:       ValueType:     I32
; CHECK-NEXT:       AccQual:       Default
; CHECK-NEXT:     - Size:          8
; CHECK-NEXT:       Align:         8
; CHECK-NEXT:       ValueKind:     HiddenGlobalOffsetX
; CHECK-NEXT:       ValueType:     I64
; CHECK-NEXT:     - Size:          8
; CHECK-NEXT:       Align:         8
; CHECK-NEXT:       ValueKind:     HiddenGlobalOffsetY
; CHECK-NEXT:       ValueType:     I64
; CHECK-NEXT:     - Size:          8
; CHECK-NEXT:       Align:         8
; CHECK-NEXT:       ValueKind:     HiddenGlobalOffsetZ
; CHECK-NEXT:       ValueType:     I64
; CHECK-NEXT:     - Size:          8
; CHECK-NEXT:       Align:         8
; CHECK-NEXT:       ValueKind:     HiddenPrintfBuffer
; CHECK-NEXT:       ValueType:     I8
; CHECK-NEXT:       AddrSpaceQual: Global
define amdgpu_kernel void @test_arg_vector_of_ptr(<2 x i32 addrspace(1)*> %a) #0
    !kernel_arg_addr_space !1 !kernel_arg_access_qual !2 !kernel_arg_type !83
    !kernel_arg_base_type !83 !kernel_arg_type_qual !4 {
  ret void
}

; CHECK:      - Name:            test_arg_unknown_builtin_type
; CHECK-NEXT:   SymbolName:      'test_arg_unknown_builtin_type@kd'
; CHECK-NEXT:   Language:        OpenCL C
; CHECK-NEXT:   LanguageVersion: [ 2, 0 ]
; CHECK-NEXT:   Args:
; CHECK-NEXT:     - Name:          a
; CHECK-NEXT:       TypeName:      clk_event_t
; CHECK-NEXT:       Size:          8
; CHECK-NEXT:       Align:         8
; CHECK-NEXT:       ValueKind:     GlobalBuffer
; CHECK-NEXT:       ValueType:     Struct
; CHECK-NEXT:       AddrSpaceQual: Global
; CHECK-NEXT:       AccQual:       Default
; CHECK-NEXT:     - Size:          8
; CHECK-NEXT:       Align:         8
; CHECK-NEXT:       ValueKind:     HiddenGlobalOffsetX
; CHECK-NEXT:       ValueType:     I64
; CHECK-NEXT:     - Size:          8
; CHECK-NEXT:       Align:         8
; CHECK-NEXT:       ValueKind:     HiddenGlobalOffsetY
; CHECK-NEXT:       ValueType:     I64
; CHECK-NEXT:     - Size:          8
; CHECK-NEXT:       Align:         8
; CHECK-NEXT:       ValueKind:     HiddenGlobalOffsetZ
; CHECK-NEXT:       ValueType:     I64
; CHECK-NEXT:     - Size:          8
; CHECK-NEXT:       Align:         8
; CHECK-NEXT:       ValueKind:     HiddenPrintfBuffer
; CHECK-NEXT:       ValueType:     I8
; CHECK-NEXT:       AddrSpaceQual: Global
define amdgpu_kernel void @test_arg_unknown_builtin_type(
    %opencl.clk_event_t addrspace(1)* %a) #0
    !kernel_arg_addr_space !81 !kernel_arg_access_qual !2 !kernel_arg_type !84
    !kernel_arg_base_type !84 !kernel_arg_type_qual !4 {
  ret void
}

; CHECK:      - Name:            test_pointee_align
; CHECK-NEXT:   SymbolName:      'test_pointee_align@kd'
; CHECK-NEXT:   Language:        OpenCL C
; CHECK-NEXT:   LanguageVersion: [ 2, 0 ]
; CHECK-NEXT:   Args:
; CHECK-NEXT:     - Name:          a
; CHECK-NEXT:       TypeName:      'long  addrspace(5)*'
; CHECK-NEXT:       Size:          8
; CHECK-NEXT:       Align:         8
; CHECK-NEXT:       ValueKind:     GlobalBuffer
; CHECK-NEXT:       ValueType:     I64
; CHECK-NEXT:       AddrSpaceQual: Global
; CHECK-NEXT:       AccQual:       Default
; CHECK-NEXT:     - Name:          b
; CHECK-NEXT:       TypeName:      'char  addrspace(5)*'
; CHECK-NEXT:       Size:          4
; CHECK-NEXT:       Align:         4
; CHECK-NEXT:       ValueKind:     DynamicSharedPointer
; CHECK-NEXT:       ValueType:     I8
; CHECK-NEXT:       PointeeAlign:  1
; CHECK-NEXT:       AddrSpaceQual: Local
; CHECK-NEXT:       AccQual:       Default
; CHECK-NEXT:     - Name:          c
; CHECK-NEXT:       TypeName:      'char2  addrspace(5)*'
; CHECK-NEXT:       Size:          4
; CHECK-NEXT:       Align:         4
; CHECK-NEXT:       ValueKind:     DynamicSharedPointer
; CHECK-NEXT:       ValueType:     I8
; CHECK-NEXT:       PointeeAlign:  2
; CHECK-NEXT:       AddrSpaceQual: Local
; CHECK-NEXT:       AccQual:       Default
; CHECK-NEXT:     - Name:          d
; CHECK-NEXT:       TypeName:      'char3  addrspace(5)*'
; CHECK-NEXT:       Size:          4
; CHECK-NEXT:       Align:         4
; CHECK-NEXT:       ValueKind:     DynamicSharedPointer
; CHECK-NEXT:       ValueType:     I8
; CHECK-NEXT:       PointeeAlign:  4
; CHECK-NEXT:       AddrSpaceQual: Local
; CHECK-NEXT:       AccQual:       Default
; CHECK-NEXT:     - Name:          e
; CHECK-NEXT:       TypeName:      'char4  addrspace(5)*'
; CHECK-NEXT:       Size:          4
; CHECK-NEXT:       Align:         4
; CHECK-NEXT:       ValueKind:     DynamicSharedPointer
; CHECK-NEXT:       ValueType:     I8
; CHECK-NEXT:       PointeeAlign:  4
; CHECK-NEXT:       AddrSpaceQual: Local
; CHECK-NEXT:       AccQual:       Default
; CHECK-NEXT:     - Name:          f
; CHECK-NEXT:       TypeName:      'char8  addrspace(5)*'
; CHECK-NEXT:       Size:          4
; CHECK-NEXT:       Align:         4
; CHECK-NEXT:       ValueKind:     DynamicSharedPointer
; CHECK-NEXT:       ValueType:     I8
; CHECK-NEXT:       PointeeAlign:  8
; CHECK-NEXT:       AddrSpaceQual: Local
; CHECK-NEXT:       AccQual:       Default
; CHECK-NEXT:     - Name:          g
; CHECK-NEXT:       TypeName:      'char16  addrspace(5)*'
; CHECK-NEXT:       Size:          4
; CHECK-NEXT:       Align:         4
; CHECK-NEXT:       ValueKind:     DynamicSharedPointer
; CHECK-NEXT:       ValueType:     I8
; CHECK-NEXT:       PointeeAlign:  16
; CHECK-NEXT:       AddrSpaceQual: Local
; CHECK-NEXT:       AccQual:       Default
; CHECK-NEXT:     - Name:          h
; CHECK-NEXT:       Size:          4
; CHECK-NEXT:       Align:         4
; CHECK-NEXT:       ValueKind:     DynamicSharedPointer
; CHECK-NEXT:       ValueType:     Struct
; CHECK-NEXT:       PointeeAlign:  1
; CHECK-NEXT:       AddrSpaceQual: Local
; CHECK-NEXT:     - Size:          8
; CHECK-NEXT:       Align:         8
; CHECK-NEXT:       ValueKind:     HiddenGlobalOffsetX
; CHECK-NEXT:       ValueType:     I64
; CHECK-NEXT:     - Size:          8
; CHECK-NEXT:       Align:         8
; CHECK-NEXT:       ValueKind:     HiddenGlobalOffsetY
; CHECK-NEXT:       ValueType:     I64
; CHECK-NEXT:     - Size:          8
; CHECK-NEXT:       Align:         8
; CHECK-NEXT:       ValueKind:     HiddenGlobalOffsetZ
; CHECK-NEXT:       ValueType:     I64
; CHECK-NEXT:     - Size:          8
; CHECK-NEXT:       Align:         8
; CHECK-NEXT:       ValueKind:     HiddenPrintfBuffer
; CHECK-NEXT:       ValueType:     I8
; CHECK-NEXT:       AddrSpaceQual: Global
define amdgpu_kernel void @test_pointee_align(i64 addrspace(1)* %a,
                                              i8 addrspace(3)* %b,
                                              <2 x i8> addrspace(3)* %c,
                                              <3 x i8> addrspace(3)* %d,
                                              <4 x i8> addrspace(3)* %e,
                                              <8 x i8> addrspace(3)* %f,
                                              <16 x i8> addrspace(3)* %g,
                                              {} addrspace(3)* %h) #0
    !kernel_arg_addr_space !91 !kernel_arg_access_qual !92 !kernel_arg_type !93
    !kernel_arg_base_type !93 !kernel_arg_type_qual !94 {
  ret void
}

; CHECK:      - Name:            test_pointee_align_attribute
; CHECK-NEXT:   SymbolName:      'test_pointee_align_attribute@kd'
; CHECK-NEXT:   Language:        OpenCL C
; CHECK-NEXT:   LanguageVersion: [ 2, 0 ]
; CHECK-NEXT:   Args:
; CHECK-NEXT:     - Name:          a
; CHECK-NEXT:       TypeName:      'long  addrspace(5)*'
; CHECK-NEXT:       Size:          8
; CHECK-NEXT:       Align:         8
; CHECK-NEXT:       ValueKind:     GlobalBuffer
; CHECK-NEXT:       ValueType:     I64
; CHECK-NEXT:       AddrSpaceQual: Global
; CHECK-NEXT:       AccQual:       Default
; CHECK-NEXT:     - Name:          b
; CHECK-NEXT:       TypeName:      'char  addrspace(5)*'
; CHECK-NEXT:       Size:          4
; CHECK-NEXT:       Align:         4
; CHECK-NEXT:       ValueKind:     DynamicSharedPointer
; CHECK-NEXT:       ValueType:     I8
; CHECK-NEXT:       PointeeAlign:  8
; CHECK-NEXT:       AddrSpaceQual: Local
; CHECK-NEXT:       AccQual:       Default
; CHECK-NEXT:     - Name:          c
; CHECK-NEXT:       TypeName:      'char2  addrspace(5)*'
; CHECK-NEXT:       Size:          4
; CHECK-NEXT:       Align:         4
; CHECK-NEXT:       ValueKind:     DynamicSharedPointer
; CHECK-NEXT:       ValueType:     I8
; CHECK-NEXT:       PointeeAlign:  32
; CHECK-NEXT:       AddrSpaceQual: Local
; CHECK-NEXT:       AccQual:       Default
; CHECK-NEXT:     - Name:          d
; CHECK-NEXT:       TypeName:      'char3  addrspace(5)*'
; CHECK-NEXT:       Size:          4
; CHECK-NEXT:       Align:         4
; CHECK-NEXT:       ValueKind:     DynamicSharedPointer
; CHECK-NEXT:       ValueType:     I8
; CHECK-NEXT:       PointeeAlign:  64
; CHECK-NEXT:       AddrSpaceQual: Local
; CHECK-NEXT:       AccQual:       Default
; CHECK-NEXT:     - Name:          e
; CHECK-NEXT:       TypeName:      'char4  addrspace(5)*'
; CHECK-NEXT:       Size:          4
; CHECK-NEXT:       Align:         4
; CHECK-NEXT:       ValueKind:     DynamicSharedPointer
; CHECK-NEXT:       ValueType:     I8
; CHECK-NEXT:       PointeeAlign:  256
; CHECK-NEXT:       AddrSpaceQual: Local
; CHECK-NEXT:       AccQual:       Default
; CHECK-NEXT:     - Name:          f
; CHECK-NEXT:       TypeName:      'char8  addrspace(5)*'
; CHECK-NEXT:       Size:          4
; CHECK-NEXT:       Align:         4
; CHECK-NEXT:       ValueKind:     DynamicSharedPointer
; CHECK-NEXT:       ValueType:     I8
; CHECK-NEXT:       PointeeAlign:  128
; CHECK-NEXT:       AddrSpaceQual: Local
; CHECK-NEXT:       AccQual:       Default
; CHECK-NEXT:     - Name:          g
; CHECK-NEXT:       TypeName:      'char16  addrspace(5)*'
; CHECK-NEXT:       Size:          4
; CHECK-NEXT:       Align:         4
; CHECK-NEXT:       ValueKind:     DynamicSharedPointer
; CHECK-NEXT:       ValueType:     I8
; CHECK-NEXT:       PointeeAlign:  1024
; CHECK-NEXT:       AddrSpaceQual: Local
; CHECK-NEXT:       AccQual:       Default
; CHECK-NEXT:      - Name:            h
; CHECK-NEXT:        Size:            4
; CHECK-NEXT:        Align:           4
; CHECK-NEXT:        ValueKind:       DynamicSharedPointer
; CHECK-NEXT:        ValueType:       Struct
; CHECK-NEXT:        PointeeAlign:    16
; CHECK-NEXT:        AddrSpaceQual:   Local
; CHECK-NEXT:     - Size:          8
; CHECK-NEXT:       Align:         8
; CHECK-NEXT:       ValueKind:     HiddenGlobalOffsetX
; CHECK-NEXT:       ValueType:     I64
; CHECK-NEXT:     - Size:          8
; CHECK-NEXT:       Align:         8
; CHECK-NEXT:       ValueKind:     HiddenGlobalOffsetY
; CHECK-NEXT:       ValueType:     I64
; CHECK-NEXT:     - Size:          8
; CHECK-NEXT:       Align:         8
; CHECK-NEXT:       ValueKind:     HiddenGlobalOffsetZ
; CHECK-NEXT:       ValueType:     I64
; CHECK-NEXT:     - Size:          8
; CHECK-NEXT:       Align:         8
; CHECK-NEXT:       ValueKind:     HiddenPrintfBuffer
; CHECK-NEXT:       ValueType:     I8
; CHECK-NEXT:       AddrSpaceQual: Global
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


; CHECK:      - Name:            __test_block_invoke_kernel
; CHECK-NEXT:   SymbolName:      '__test_block_invoke_kernel@kd'
; CHECK-NEXT:   Language:        OpenCL C
; CHECK-NEXT:   LanguageVersion: [ 2, 0 ]
; CHECK-NEXT:   Attrs:
; CHECK-NEXT:       RuntimeHandle: __test_block_invoke_kernel_runtime_handle
; CHECK-NEXT:   Args:
; CHECK-NEXT:     - Name:          arg
; CHECK-NEXT:       TypeName:      __block_literal
; CHECK-NEXT:       Size:          25
; CHECK-NEXT:       Align:         1
; CHECK-NEXT:       ValueKind:     ByValue
; CHECK-NEXT:       ValueType:     Struct
; CHECK-NEXT:       AccQual:       Default
; CHECK-NEXT:     - Size:          8
; CHECK-NEXT:       Align:         8
; CHECK-NEXT:       ValueKind:     HiddenGlobalOffsetX
; CHECK-NEXT:       ValueType:     I64
; CHECK-NEXT:     - Size:          8
; CHECK-NEXT:       Align:         8
; CHECK-NEXT:       ValueKind:     HiddenGlobalOffsetY
; CHECK-NEXT:       ValueType:     I64
; CHECK-NEXT:     - Size:          8
; CHECK-NEXT:       Align:         8
; CHECK-NEXT:       ValueKind:     HiddenGlobalOffsetZ
; CHECK-NEXT:       ValueType:     I64
; CHECK-NEXT:     - Size:          8
; CHECK-NEXT:       Align:         8
; CHECK-NEXT:       ValueKind:     HiddenPrintfBuffer
; CHECK-NEXT:       ValueType:     I8
; CHECK-NEXT:       AddrSpaceQual: Global
define amdgpu_kernel void @__test_block_invoke_kernel(
    <{ i32, i32, i8*, i8 addrspace(1)*, i8 }> %arg) #1
    !kernel_arg_addr_space !1 !kernel_arg_access_qual !2 !kernel_arg_type !110
    !kernel_arg_base_type !110 !kernel_arg_type_qual !4 {
  ret void
}

; CHECK:      - Name:            test_enqueue_kernel_caller
; CHECK-NEXT:   SymbolName:      'test_enqueue_kernel_caller@kd'
; CHECK-NEXT:   Language:        OpenCL C
; CHECK-NEXT:   LanguageVersion: [ 2, 0 ]
; CHECK-NEXT:   Args:
; CHECK-NEXT:     - Name:          a
; CHECK-NEXT:       TypeName:      char
; CHECK-NEXT:       Size:          1
; CHECK-NEXT:       Align:         1
; CHECK-NEXT:       ValueKind:     ByValue
; CHECK-NEXT:       ValueType:     I8
; CHECK-NEXT:       AccQual:       Default
; CHECK-NEXT:     - Size:          8
; CHECK-NEXT:       Align:         8
; CHECK-NEXT:       ValueKind:     HiddenGlobalOffsetX
; CHECK-NEXT:       ValueType:     I64
; CHECK-NEXT:     - Size:          8
; CHECK-NEXT:       Align:         8
; CHECK-NEXT:       ValueKind:     HiddenGlobalOffsetY
; CHECK-NEXT:       ValueType:     I64
; CHECK-NEXT:     - Size:          8
; CHECK-NEXT:       Align:         8
; CHECK-NEXT:       ValueKind:     HiddenGlobalOffsetZ
; CHECK-NEXT:       ValueType:     I64
; CHECK-NEXT:     - Size:          8
; CHECK-NEXT:       Align:         8
; CHECK-NEXT:       ValueKind:     HiddenPrintfBuffer
; CHECK-NEXT:       ValueType:     I8
; CHECK-NEXT:       AddrSpaceQual: Global
; CHECK-NEXT:     - Size:          8
; CHECK-NEXT:       Align:         8
; CHECK-NEXT:       ValueKind:     HiddenDefaultQueue
; CHECK-NEXT:       ValueType:     I8
; CHECK-NEXT:       AddrSpaceQual: Global
; CHECK-NEXT:     - Size:          8
; CHECK-NEXT:       Align:         8
; CHECK-NEXT:       ValueKind:     HiddenCompletionAction
; CHECK-NEXT:       ValueType:     I8
; CHECK-NEXT:       AddrSpaceQual: Global
define amdgpu_kernel void @test_enqueue_kernel_caller(i8 %a) #2
    !kernel_arg_addr_space !1 !kernel_arg_access_qual !2 !kernel_arg_type !9
    !kernel_arg_base_type !9 !kernel_arg_type_qual !4 {
  ret void
}

; CHECK: - Name:            unknown_addrspace_kernarg
; CHECK: Args:
; CHECK-NEXT: - Name:            ptr
; CHECK-NEXT: Size:            8
; CHECK-NEXT: Align:           8
; CHECK-NEXT: ValueKind:       GlobalBuffer
; CHECK-NEXT: ValueType:       I32
define amdgpu_kernel void @unknown_addrspace_kernarg(i32 addrspace(12345)* %ptr) #0 {
  ret void
}

attributes #0 = { "amdgpu-implicitarg-num-bytes"="48" }
attributes #1 = { "amdgpu-implicitarg-num-bytes"="48" "runtime-handle"="__test_block_invoke_kernel_runtime_handle" }
attributes #2 = { "amdgpu-implicitarg-num-bytes"="48" "calls-enqueue-kernel" }

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

; PARSER: AMDGPU HSA Metadata Parser Test: PASS

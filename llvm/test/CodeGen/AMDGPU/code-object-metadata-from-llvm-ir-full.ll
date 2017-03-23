; RUN: llc -mtriple=amdgcn-amd-amdhsa -mcpu=gfx700 -filetype=obj -o - < %s | llvm-readobj -amdgpu-code-object-metadata -elf-output-style=GNU -notes | FileCheck --check-prefix=CHECK --check-prefix=GFX700 --check-prefix=NOTES %s
; RUN: llc -mtriple=amdgcn-amd-amdhsa -mcpu=gfx800 -filetype=obj -o - < %s | llvm-readobj -amdgpu-code-object-metadata -elf-output-style=GNU -notes | FileCheck --check-prefix=CHECK --check-prefix=GFX800 --check-prefix=NOTES %s
; RUN: llc -mtriple=amdgcn-amd-amdhsa -mcpu=gfx900 -filetype=obj -o - < %s | llvm-readobj -amdgpu-code-object-metadata -elf-output-style=GNU -notes | FileCheck --check-prefix=CHECK --check-prefix=GFX900 --check-prefix=NOTES %s
; RUN: llc -mtriple=amdgcn-amd-amdhsa -mcpu=gfx700 -amdgpu-dump-comd -amdgpu-verify-comd -filetype=obj -o - < %s 2>&1 | FileCheck --check-prefix=PARSER %s
; RUN: llc -mtriple=amdgcn-amd-amdhsa -mcpu=gfx800 -amdgpu-dump-comd -amdgpu-verify-comd -filetype=obj -o - < %s 2>&1 | FileCheck --check-prefix=PARSER %s
; RUN: llc -mtriple=amdgcn-amd-amdhsa -mcpu=gfx900 -amdgpu-dump-comd -amdgpu-verify-comd -filetype=obj -o - < %s 2>&1 | FileCheck --check-prefix=PARSER %s

%struct.A = type { i8, float }
%opencl.image1d_t = type opaque
%opencl.image2d_t = type opaque
%opencl.image3d_t = type opaque
%opencl.queue_t = type opaque
%opencl.pipe_t = type opaque
%struct.B = type { i32 addrspace(1)*}
%opencl.clk_event_t = type opaque

; CHECK: ---
; CHECK:  Version: [ 1, 0 ]
; CHECK:  Printf: [ '1:1:4:%d\n', '2:1:8:%g\n' ]
; CHECK:  Kernels:

; CHECK:      - Name:            test_char
; CHECK-NEXT:   Language:        OpenCL C
; CHECK-NEXT:   LanguageVersion: [ 2, 0 ]
; CHECK-NEXT:   Args:
; CHECK-NEXT:     - Size:          1
; CHECK-NEXT:       Align:         1
; CHECK-NEXT:       Kind:          ByValue
; CHECK-NEXT:       ValueType:     I8
; CHECK-NEXT:       AccQual:       Default
; CHECK-NEXT:       TypeName:      char
; CHECK-NEXT:     - Size:          8
; CHECK-NEXT:       Align:         8
; CHECK-NEXT:       Kind:          HiddenGlobalOffsetX
; CHECK-NEXT:       ValueType:     I64
; CHECK-NEXT:     - Size:          8
; CHECK-NEXT:       Align:         8
; CHECK-NEXT:       Kind:          HiddenGlobalOffsetY
; CHECK-NEXT:       ValueType:     I64
; CHECK-NEXT:     - Size:          8
; CHECK-NEXT:       Align:         8
; CHECK-NEXT:       Kind:          HiddenGlobalOffsetZ
; CHECK-NEXT:       ValueType:     I64
; CHECK-NEXT:     - Size:          8
; CHECK-NEXT:       Align:         8
; CHECK-NEXT:       Kind:          HiddenPrintfBuffer
; CHECK-NEXT:       ValueType:     I8
; CHECK-NEXT:       AddrSpaceQual: Global
define amdgpu_kernel void @test_char(i8 %a)
    !kernel_arg_addr_space !1 !kernel_arg_access_qual !2 !kernel_arg_type !9
    !kernel_arg_base_type !9 !kernel_arg_type_qual !4 {
  ret void
}

; CHECK:      - Name:            test_ushort2
; CHECK-NEXT:   Language:        OpenCL C
; CHECK-NEXT:   LanguageVersion: [ 2, 0 ]
; CHECK-NEXT:   Args:
; CHECK-NEXT:     - Size:          4
; CHECK-NEXT:       Align:         4
; CHECK-NEXT:       Kind:          ByValue
; CHECK-NEXT:       ValueType:     U16
; CHECK-NEXT:       AccQual:       Default
; CHECK-NEXT:       TypeName:      ushort2
; CHECK-NEXT:     - Size:          8
; CHECK-NEXT:       Align:         8
; CHECK-NEXT:       Kind:          HiddenGlobalOffsetX
; CHECK-NEXT:       ValueType:     I64
; CHECK-NEXT:     - Size:          8
; CHECK-NEXT:       Align:         8
; CHECK-NEXT:       Kind:          HiddenGlobalOffsetY
; CHECK-NEXT:       ValueType:     I64
; CHECK-NEXT:     - Size:          8
; CHECK-NEXT:       Align:         8
; CHECK-NEXT:       Kind:          HiddenGlobalOffsetZ
; CHECK-NEXT:       ValueType:     I64
; CHECK-NEXT:     - Size:          8
; CHECK-NEXT:       Align:         8
; CHECK-NEXT:       Kind:          HiddenPrintfBuffer
; CHECK-NEXT:       ValueType:     I8
; CHECK-NEXT:       AddrSpaceQual: Global
define amdgpu_kernel void @test_ushort2(<2 x i16> %a)
    !kernel_arg_addr_space !1 !kernel_arg_access_qual !2 !kernel_arg_type !10
    !kernel_arg_base_type !10 !kernel_arg_type_qual !4 {
  ret void
}

; CHECK:      - Name:            test_int3
; CHECK-NEXT:   Language:        OpenCL C
; CHECK-NEXT:   LanguageVersion: [ 2, 0 ]
; CHECK-NEXT:   Args:
; CHECK-NEXT:     - Size:          16
; CHECK-NEXT:       Align:         16
; CHECK-NEXT:       Kind:          ByValue
; CHECK-NEXT:       ValueType:     I32
; CHECK-NEXT:       AccQual:       Default
; CHECK-NEXT:       TypeName:      int3
; CHECK-NEXT:     - Size:          8
; CHECK-NEXT:       Align:         8
; CHECK-NEXT:       Kind:          HiddenGlobalOffsetX
; CHECK-NEXT:       ValueType:     I64
; CHECK-NEXT:     - Size:          8
; CHECK-NEXT:       Align:         8
; CHECK-NEXT:       Kind:          HiddenGlobalOffsetY
; CHECK-NEXT:       ValueType:     I64
; CHECK-NEXT:     - Size:          8
; CHECK-NEXT:       Align:         8
; CHECK-NEXT:       Kind:          HiddenGlobalOffsetZ
; CHECK-NEXT:       ValueType:     I64
; CHECK-NEXT:     - Size:          8
; CHECK-NEXT:       Align:         8
; CHECK-NEXT:       Kind:          HiddenPrintfBuffer
; CHECK-NEXT:       ValueType:     I8
; CHECK-NEXT:       AddrSpaceQual: Global
define amdgpu_kernel void @test_int3(<3 x i32> %a)
    !kernel_arg_addr_space !1 !kernel_arg_access_qual !2 !kernel_arg_type !11
    !kernel_arg_base_type !11 !kernel_arg_type_qual !4 {
  ret void
}

; CHECK:      - Name:            test_ulong4
; CHECK-NEXT:   Language:        OpenCL C
; CHECK-NEXT:   LanguageVersion: [ 2, 0 ]
; CHECK-NEXT:   Args:
; CHECK-NEXT:     - Size:          32
; CHECK-NEXT:       Align:         32
; CHECK-NEXT:       Kind:          ByValue
; CHECK-NEXT:       ValueType:     U64
; CHECK-NEXT:       AccQual:       Default
; CHECK-NEXT:       TypeName:      ulong4
; CHECK-NEXT:     - Size:          8
; CHECK-NEXT:       Align:         8
; CHECK-NEXT:       Kind:          HiddenGlobalOffsetX
; CHECK-NEXT:       ValueType:     I64
; CHECK-NEXT:     - Size:          8
; CHECK-NEXT:       Align:         8
; CHECK-NEXT:       Kind:          HiddenGlobalOffsetY
; CHECK-NEXT:       ValueType:     I64
; CHECK-NEXT:     - Size:          8
; CHECK-NEXT:       Align:         8
; CHECK-NEXT:       Kind:          HiddenGlobalOffsetZ
; CHECK-NEXT:       ValueType:     I64
; CHECK-NEXT:     - Size:          8
; CHECK-NEXT:       Align:         8
; CHECK-NEXT:       Kind:          HiddenPrintfBuffer
; CHECK-NEXT:       ValueType:     I8
; CHECK-NEXT:       AddrSpaceQual: Global
define amdgpu_kernel void @test_ulong4(<4 x i64> %a)
    !kernel_arg_addr_space !1 !kernel_arg_access_qual !2 !kernel_arg_type !12
    !kernel_arg_base_type !12 !kernel_arg_type_qual !4 {
  ret void
}

; CHECK:      - Name:            test_half8
; CHECK-NEXT:   Language:        OpenCL C
; CHECK-NEXT:   LanguageVersion: [ 2, 0 ]
; CHECK-NEXT:   Args:
; CHECK-NEXT:     - Size:          16
; CHECK-NEXT:       Align:         16
; CHECK-NEXT:       Kind:          ByValue
; CHECK-NEXT:       ValueType:     F16
; CHECK-NEXT:       AccQual:       Default
; CHECK-NEXT:       TypeName:      half8
; CHECK-NEXT:     - Size:          8
; CHECK-NEXT:       Align:         8
; CHECK-NEXT:       Kind:          HiddenGlobalOffsetX
; CHECK-NEXT:       ValueType:     I64
; CHECK-NEXT:     - Size:          8
; CHECK-NEXT:       Align:         8
; CHECK-NEXT:       Kind:          HiddenGlobalOffsetY
; CHECK-NEXT:       ValueType:     I64
; CHECK-NEXT:     - Size:          8
; CHECK-NEXT:       Align:         8
; CHECK-NEXT:       Kind:          HiddenGlobalOffsetZ
; CHECK-NEXT:       ValueType:     I64
; CHECK-NEXT:     - Size:          8
; CHECK-NEXT:       Align:         8
; CHECK-NEXT:       Kind:          HiddenPrintfBuffer
; CHECK-NEXT:       ValueType:     I8
; CHECK-NEXT:       AddrSpaceQual: Global
define amdgpu_kernel void @test_half8(<8 x half> %a)
    !kernel_arg_addr_space !1 !kernel_arg_access_qual !2 !kernel_arg_type !13
    !kernel_arg_base_type !13 !kernel_arg_type_qual !4 {
  ret void
}

; CHECK:      - Name:            test_float16
; CHECK-NEXT:   Language:        OpenCL C
; CHECK-NEXT:   LanguageVersion: [ 2, 0 ]
; CHECK-NEXT:   Args:
; CHECK-NEXT:     - Size:          64
; CHECK-NEXT:       Align:         64
; CHECK-NEXT:       Kind:          ByValue
; CHECK-NEXT:       ValueType:     F32
; CHECK-NEXT:       AccQual:       Default
; CHECK-NEXT:       TypeName:      float16
; CHECK-NEXT:     - Size:          8
; CHECK-NEXT:       Align:         8
; CHECK-NEXT:       Kind:          HiddenGlobalOffsetX
; CHECK-NEXT:       ValueType:     I64
; CHECK-NEXT:     - Size:          8
; CHECK-NEXT:       Align:         8
; CHECK-NEXT:       Kind:          HiddenGlobalOffsetY
; CHECK-NEXT:       ValueType:     I64
; CHECK-NEXT:     - Size:          8
; CHECK-NEXT:       Align:         8
; CHECK-NEXT:       Kind:          HiddenGlobalOffsetZ
; CHECK-NEXT:       ValueType:     I64
; CHECK-NEXT:     - Size:          8
; CHECK-NEXT:       Align:         8
; CHECK-NEXT:       Kind:          HiddenPrintfBuffer
; CHECK-NEXT:       ValueType:     I8
; CHECK-NEXT:       AddrSpaceQual: Global
define amdgpu_kernel void @test_float16(<16 x float> %a)
    !kernel_arg_addr_space !1 !kernel_arg_access_qual !2 !kernel_arg_type !14
    !kernel_arg_base_type !14 !kernel_arg_type_qual !4 {
  ret void
}

; CHECK:      - Name:            test_double16
; CHECK-NEXT:   Language:        OpenCL C
; CHECK-NEXT:   LanguageVersion: [ 2, 0 ]
; CHECK-NEXT:   Args:
; CHECK-NEXT:     - Size:          128
; CHECK-NEXT:       Align:         128
; CHECK-NEXT:       Kind:          ByValue
; CHECK-NEXT:       ValueType:     F64
; CHECK-NEXT:       AccQual:       Default
; CHECK-NEXT:       TypeName:      double16
; CHECK-NEXT:     - Size:          8
; CHECK-NEXT:       Align:         8
; CHECK-NEXT:       Kind:          HiddenGlobalOffsetX
; CHECK-NEXT:       ValueType:     I64
; CHECK-NEXT:     - Size:          8
; CHECK-NEXT:       Align:         8
; CHECK-NEXT:       Kind:          HiddenGlobalOffsetY
; CHECK-NEXT:       ValueType:     I64
; CHECK-NEXT:     - Size:          8
; CHECK-NEXT:       Align:         8
; CHECK-NEXT:       Kind:          HiddenGlobalOffsetZ
; CHECK-NEXT:       ValueType:     I64
; CHECK-NEXT:     - Size:          8
; CHECK-NEXT:       Align:         8
; CHECK-NEXT:       Kind:          HiddenPrintfBuffer
; CHECK-NEXT:       ValueType:     I8
; CHECK-NEXT:       AddrSpaceQual: Global
define amdgpu_kernel void @test_double16(<16 x double> %a)
    !kernel_arg_addr_space !1 !kernel_arg_access_qual !2 !kernel_arg_type !15
    !kernel_arg_base_type !15 !kernel_arg_type_qual !4 {
  ret void
}

; CHECK:      - Name:            test_pointer
; CHECK-NEXT:   Language:        OpenCL C
; CHECK-NEXT:   LanguageVersion: [ 2, 0 ]
; CHECK-NEXT:   Args:
; CHECK-NEXT:     - Size:          8
; CHECK-NEXT:       Align:         8
; CHECK-NEXT:       Kind:          GlobalBuffer
; CHECK-NEXT:       ValueType:     I32
; CHECK-NEXT:       AccQual:       Default
; CHECK-NEXT:       AddrSpaceQual: Global
; CHECK-NEXT:       TypeName:      'int *'
; CHECK-NEXT:     - Size:          8
; CHECK-NEXT:       Align:         8
; CHECK-NEXT:       Kind:          HiddenGlobalOffsetX
; CHECK-NEXT:       ValueType:     I64
; CHECK-NEXT:     - Size:          8
; CHECK-NEXT:       Align:         8
; CHECK-NEXT:       Kind:          HiddenGlobalOffsetY
; CHECK-NEXT:       ValueType:     I64
; CHECK-NEXT:     - Size:          8
; CHECK-NEXT:       Align:         8
; CHECK-NEXT:       Kind:          HiddenGlobalOffsetZ
; CHECK-NEXT:       ValueType:     I64
; CHECK-NEXT:     - Size:          8
; CHECK-NEXT:       Align:         8
; CHECK-NEXT:       Kind:          HiddenPrintfBuffer
; CHECK-NEXT:       ValueType:     I8
; CHECK-NEXT:       AddrSpaceQual: Global
define amdgpu_kernel void @test_pointer(i32 addrspace(1)* %a)
    !kernel_arg_addr_space !1 !kernel_arg_access_qual !2 !kernel_arg_type !16
    !kernel_arg_base_type !16 !kernel_arg_type_qual !4 {
  ret void
}

; CHECK:      - Name:            test_image
; CHECK-NEXT:   Language:        OpenCL C
; CHECK-NEXT:   LanguageVersion: [ 2, 0 ]
; CHECK-NEXT:   Args:
; CHECK-NEXT:     - Size:          8
; CHECK-NEXT:       Align:         8
; CHECK-NEXT:       Kind:          Image
; CHECK-NEXT:       ValueType:     Struct
; CHECK-NEXT:       AccQual:       Default
; CHECK-NEXT:       AddrSpaceQual: Global
; CHECK-NEXT:       TypeName:      image2d_t
; CHECK-NEXT:     - Size:          8
; CHECK-NEXT:       Align:         8
; CHECK-NEXT:       Kind:          HiddenGlobalOffsetX
; CHECK-NEXT:       ValueType:     I64
; CHECK-NEXT:     - Size:          8
; CHECK-NEXT:       Align:         8
; CHECK-NEXT:       Kind:          HiddenGlobalOffsetY
; CHECK-NEXT:       ValueType:     I64
; CHECK-NEXT:     - Size:          8
; CHECK-NEXT:       Align:         8
; CHECK-NEXT:       Kind:          HiddenGlobalOffsetZ
; CHECK-NEXT:       ValueType:     I64
; CHECK-NEXT:     - Size:          8
; CHECK-NEXT:       Align:         8
; CHECK-NEXT:       Kind:          HiddenPrintfBuffer
; CHECK-NEXT:       ValueType:     I8
; CHECK-NEXT:       AddrSpaceQual: Global
define amdgpu_kernel void @test_image(%opencl.image2d_t addrspace(1)* %a)
    !kernel_arg_addr_space !1 !kernel_arg_access_qual !2 !kernel_arg_type !17
    !kernel_arg_base_type !17 !kernel_arg_type_qual !4 {
  ret void
}

; CHECK:      - Name:            test_sampler
; CHECK-NEXT:   Language:        OpenCL C
; CHECK-NEXT:   LanguageVersion: [ 2, 0 ]
; CHECK-NEXT:   Args:
; CHECK-NEXT:     - Size:          4
; CHECK-NEXT:       Align:         4
; CHECK-NEXT:       Kind:          Sampler
; CHECK-NEXT:       ValueType:     I32
; CHECK-NEXT:       AccQual:       Default
; CHECK-NEXT:       TypeName:      sampler_t
; CHECK-NEXT:     - Size:          8
; CHECK-NEXT:       Align:         8
; CHECK-NEXT:       Kind:          HiddenGlobalOffsetX
; CHECK-NEXT:       ValueType:     I64
; CHECK-NEXT:     - Size:          8
; CHECK-NEXT:       Align:         8
; CHECK-NEXT:       Kind:          HiddenGlobalOffsetY
; CHECK-NEXT:       ValueType:     I64
; CHECK-NEXT:     - Size:          8
; CHECK-NEXT:       Align:         8
; CHECK-NEXT:       Kind:          HiddenGlobalOffsetZ
; CHECK-NEXT:       ValueType:     I64
; CHECK-NEXT:     - Size:          8
; CHECK-NEXT:       Align:         8
; CHECK-NEXT:       Kind:          HiddenPrintfBuffer
; CHECK-NEXT:       ValueType:     I8
; CHECK-NEXT:       AddrSpaceQual: Global
define amdgpu_kernel void @test_sampler(i32 %a)
    !kernel_arg_addr_space !1 !kernel_arg_access_qual !2 !kernel_arg_type !18
    !kernel_arg_base_type !18 !kernel_arg_type_qual !4 {
  ret void
}

; CHECK:      - Name:            test_queue
; CHECK-NEXT:   Language:        OpenCL C
; CHECK-NEXT:   LanguageVersion: [ 2, 0 ]
; CHECK-NEXT:   Args:
; CHECK-NEXT:     - Size:          8
; CHECK-NEXT:       Align:         8
; CHECK-NEXT:       Kind:          Queue
; CHECK-NEXT:       ValueType:     Struct
; CHECK-NEXT:       AccQual:       Default
; CHECK-NEXT:       AddrSpaceQual: Global
; CHECK-NEXT:       TypeName:      queue_t
; CHECK-NEXT:     - Size:          8
; CHECK-NEXT:       Align:         8
; CHECK-NEXT:       Kind:          HiddenGlobalOffsetX
; CHECK-NEXT:       ValueType:     I64
; CHECK-NEXT:     - Size:          8
; CHECK-NEXT:       Align:         8
; CHECK-NEXT:       Kind:          HiddenGlobalOffsetY
; CHECK-NEXT:       ValueType:     I64
; CHECK-NEXT:     - Size:          8
; CHECK-NEXT:       Align:         8
; CHECK-NEXT:       Kind:          HiddenGlobalOffsetZ
; CHECK-NEXT:       ValueType:     I64
; CHECK-NEXT:     - Size:          8
; CHECK-NEXT:       Align:         8
; CHECK-NEXT:       Kind:          HiddenPrintfBuffer
; CHECK-NEXT:       ValueType:     I8
; CHECK-NEXT:       AddrSpaceQual: Global
define amdgpu_kernel void @test_queue(%opencl.queue_t addrspace(1)* %a)
    !kernel_arg_addr_space !1 !kernel_arg_access_qual !2 !kernel_arg_type !19
    !kernel_arg_base_type !19 !kernel_arg_type_qual !4 {
  ret void
}

; CHECK:      - Name:            test_struct
; CHECK-NEXT:   Language:        OpenCL C
; CHECK-NEXT:   LanguageVersion: [ 2, 0 ]
; CHECK-NEXT:   Args:
; CHECK-NEXT:     - Size:          4
; CHECK-NEXT:       Align:         4
; CHECK-NEXT:       Kind:          GlobalBuffer
; CHECK-NEXT:       ValueType:     Struct
; CHECK-NEXT:       AccQual:       Default
; CHECK-NEXT:       AddrSpaceQual: Private
; CHECK-NEXT:       TypeName:      struct A
; CHECK-NEXT:     - Size:          8
; CHECK-NEXT:       Align:         8
; CHECK-NEXT:       Kind:          HiddenGlobalOffsetX
; CHECK-NEXT:       ValueType:     I64
; CHECK-NEXT:     - Size:          8
; CHECK-NEXT:       Align:         8
; CHECK-NEXT:       Kind:          HiddenGlobalOffsetY
; CHECK-NEXT:       ValueType:     I64
; CHECK-NEXT:     - Size:          8
; CHECK-NEXT:       Align:         8
; CHECK-NEXT:       Kind:          HiddenGlobalOffsetZ
; CHECK-NEXT:       ValueType:     I64
; CHECK-NEXT:     - Size:          8
; CHECK-NEXT:       Align:         8
; CHECK-NEXT:       Kind:          HiddenPrintfBuffer
; CHECK-NEXT:       ValueType:     I8
; CHECK-NEXT:       AddrSpaceQual: Global
define amdgpu_kernel void @test_struct(%struct.A* byval %a)
    !kernel_arg_addr_space !1 !kernel_arg_access_qual !2 !kernel_arg_type !20
    !kernel_arg_base_type !20 !kernel_arg_type_qual !4 {
  ret void
}

; CHECK:      - Name:            test_i128
; CHECK-NEXT:   Language:        OpenCL C
; CHECK-NEXT:   LanguageVersion: [ 2, 0 ]
; CHECK-NEXT:   Args:
; CHECK-NEXT:     - Size:          16
; CHECK-NEXT:       Align:         8
; CHECK-NEXT:       Kind:          ByValue
; CHECK-NEXT:       ValueType:     Struct
; CHECK-NEXT:       AccQual:       Default
; CHECK-NEXT:       TypeName:      i128
; CHECK-NEXT:     - Size:          8
; CHECK-NEXT:       Align:         8
; CHECK-NEXT:       Kind:          HiddenGlobalOffsetX
; CHECK-NEXT:       ValueType:     I64
; CHECK-NEXT:     - Size:          8
; CHECK-NEXT:       Align:         8
; CHECK-NEXT:       Kind:          HiddenGlobalOffsetY
; CHECK-NEXT:       ValueType:     I64
; CHECK-NEXT:     - Size:          8
; CHECK-NEXT:       Align:         8
; CHECK-NEXT:       Kind:          HiddenGlobalOffsetZ
; CHECK-NEXT:       ValueType:     I64
; CHECK-NEXT:     - Size:          8
; CHECK-NEXT:       Align:         8
; CHECK-NEXT:       Kind:          HiddenPrintfBuffer
; CHECK-NEXT:       ValueType:     I8
; CHECK-NEXT:       AddrSpaceQual: Global
define amdgpu_kernel void @test_i128(i128 %a)
    !kernel_arg_addr_space !1 !kernel_arg_access_qual !2 !kernel_arg_type !21
    !kernel_arg_base_type !21 !kernel_arg_type_qual !4 {
  ret void
}

; CHECK:      - Name:            test_multi_arg
; CHECK-NEXT:   Language:        OpenCL C
; CHECK-NEXT:   LanguageVersion: [ 2, 0 ]
; CHECK-NEXT:   Args:
; CHECK-NEXT:     - Size:          4
; CHECK-NEXT:       Align:         4
; CHECK-NEXT:       Kind:          ByValue
; CHECK-NEXT:       ValueType:     I32
; CHECK-NEXT:       AccQual:       Default
; CHECK-NEXT:       TypeName:      int
; CHECK-NEXT:     - Size:          4
; CHECK-NEXT:       Align:         4
; CHECK-NEXT:       Kind:          ByValue
; CHECK-NEXT:       ValueType:     I16
; CHECK-NEXT:       AccQual:       Default
; CHECK-NEXT:       TypeName:      short2
; CHECK-NEXT:     - Size:          4
; CHECK-NEXT:       Align:         4
; CHECK-NEXT:       Kind:          ByValue
; CHECK-NEXT:       ValueType:     I8
; CHECK-NEXT:       AccQual:       Default
; CHECK-NEXT:       TypeName:      char3
; CHECK-NEXT:     - Size:          8
; CHECK-NEXT:       Align:         8
; CHECK-NEXT:       Kind:          HiddenGlobalOffsetX
; CHECK-NEXT:       ValueType:     I64
; CHECK-NEXT:     - Size:          8
; CHECK-NEXT:       Align:         8
; CHECK-NEXT:       Kind:          HiddenGlobalOffsetY
; CHECK-NEXT:       ValueType:     I64
; CHECK-NEXT:     - Size:          8
; CHECK-NEXT:       Align:         8
; CHECK-NEXT:       Kind:          HiddenGlobalOffsetZ
; CHECK-NEXT:       ValueType:     I64
; CHECK-NEXT:     - Size:          8
; CHECK-NEXT:       Align:         8
; CHECK-NEXT:       Kind:          HiddenPrintfBuffer
; CHECK-NEXT:       ValueType:     I8
; CHECK-NEXT:       AddrSpaceQual: Global
define amdgpu_kernel void @test_multi_arg(i32 %a, <2 x i16> %b, <3 x i8> %c)
    !kernel_arg_addr_space !22 !kernel_arg_access_qual !23 !kernel_arg_type !24
    !kernel_arg_base_type !24 !kernel_arg_type_qual !25 {
  ret void
}

; CHECK:      - Name:            test_addr_space
; CHECK-NEXT:   Language:        OpenCL C
; CHECK-NEXT:   LanguageVersion: [ 2, 0 ]
; CHECK-NEXT:   Args:
; CHECK-NEXT:     - Size:          8
; CHECK-NEXT:       Align:         8
; CHECK-NEXT:       Kind:          GlobalBuffer
; CHECK-NEXT:       ValueType:     I32
; CHECK-NEXT:       AccQual:       Default
; CHECK-NEXT:       AddrSpaceQual: Global
; CHECK-NEXT:       TypeName:      'int *'
; CHECK-NEXT:     - Size:          8
; CHECK-NEXT:       Align:         8
; CHECK-NEXT:       Kind:          GlobalBuffer
; CHECK-NEXT:       ValueType:     I32
; CHECK-NEXT:       AccQual:       Default
; CHECK-NEXT:       AddrSpaceQual: Constant
; CHECK-NEXT:       TypeName:      'int *'
; CHECK-NEXT:     - Size:          4
; CHECK-NEXT:       Align:         4
; CHECK-NEXT:       Kind:          DynamicSharedPointer
; CHECK-NEXT:       ValueType:     I32
; CHECK-NEXT:       PointeeAlign:  4
; CHECK-NEXT:       AccQual:       Default
; CHECK-NEXT:       AddrSpaceQual: Local
; CHECK-NEXT:       TypeName:      'int *'
; CHECK-NEXT:     - Size:          8
; CHECK-NEXT:       Align:         8
; CHECK-NEXT:       Kind:          HiddenGlobalOffsetX
; CHECK-NEXT:       ValueType:     I64
; CHECK-NEXT:     - Size:          8
; CHECK-NEXT:       Align:         8
; CHECK-NEXT:       Kind:          HiddenGlobalOffsetY
; CHECK-NEXT:       ValueType:     I64
; CHECK-NEXT:     - Size:          8
; CHECK-NEXT:       Align:         8
; CHECK-NEXT:       Kind:          HiddenGlobalOffsetZ
; CHECK-NEXT:       ValueType:     I64
; CHECK-NEXT:     - Size:          8
; CHECK-NEXT:       Align:         8
; CHECK-NEXT:       Kind:          HiddenPrintfBuffer
; CHECK-NEXT:       ValueType:     I8
; CHECK-NEXT:       AddrSpaceQual: Global
define amdgpu_kernel void @test_addr_space(i32 addrspace(1)* %g,
                                           i32 addrspace(2)* %c,
                                           i32 addrspace(3)* %l)
    !kernel_arg_addr_space !50 !kernel_arg_access_qual !23 !kernel_arg_type !51
    !kernel_arg_base_type !51 !kernel_arg_type_qual !25 {
  ret void
}

; CHECK:      - Name:            test_type_qual
; CHECK-NEXT:   Language:        OpenCL C
; CHECK-NEXT:   LanguageVersion: [ 2, 0 ]
; CHECK-NEXT:   Args:
; CHECK-NEXT:     - Size:          8
; CHECK-NEXT:       Align:         8
; CHECK-NEXT:       Kind:          GlobalBuffer
; CHECK-NEXT:       ValueType:     I32
; CHECK-NEXT:       AccQual:       Default
; CHECK-NEXT:       AddrSpaceQual: Global
; CHECK-NEXT:       IsVolatile:    true
; CHECK-NEXT:       TypeName:      'int *'
; CHECK-NEXT:     - Size:          8
; CHECK-NEXT:       Align:         8
; CHECK-NEXT:       Kind:          GlobalBuffer
; CHECK-NEXT:       ValueType:     I32
; CHECK-NEXT:       AccQual:       Default
; CHECK-NEXT:       AddrSpaceQual: Global
; CHECK-NEXT:       IsConst:       true
; CHECK-NEXT:       IsRestrict:    true
; CHECK-NEXT:       TypeName:      'int *'
; CHECK-NEXT:     - Size:          8
; CHECK-NEXT:       Align:         8
; CHECK-NEXT:       Kind:          Pipe
; CHECK-NEXT:       ValueType:     Struct
; CHECK-NEXT:       AccQual:       Default
; CHECK-NEXT:       AddrSpaceQual: Global
; CHECK-NEXT:       IsPipe:        true
; CHECK-NEXT:       TypeName:      'int *'
; CHECK-NEXT:     - Size:          8
; CHECK-NEXT:       Align:         8
; CHECK-NEXT:       Kind:          HiddenGlobalOffsetX
; CHECK-NEXT:       ValueType:     I64
; CHECK-NEXT:     - Size:          8
; CHECK-NEXT:       Align:         8
; CHECK-NEXT:       Kind:          HiddenGlobalOffsetY
; CHECK-NEXT:       ValueType:     I64
; CHECK-NEXT:     - Size:          8
; CHECK-NEXT:       Align:         8
; CHECK-NEXT:       Kind:          HiddenGlobalOffsetZ
; CHECK-NEXT:       ValueType:     I64
; CHECK-NEXT:     - Size:          8
; CHECK-NEXT:       Align:         8
; CHECK-NEXT:       Kind:          HiddenPrintfBuffer
; CHECK-NEXT:       ValueType:     I8
; CHECK-NEXT:       AddrSpaceQual: Global
define amdgpu_kernel void @test_type_qual(i32 addrspace(1)* %a,
                                          i32 addrspace(1)* %b,
                                          %opencl.pipe_t addrspace(1)* %c)
    !kernel_arg_addr_space !22 !kernel_arg_access_qual !23 !kernel_arg_type !51
    !kernel_arg_base_type !51 !kernel_arg_type_qual !70 {
  ret void
}

; CHECK:      - Name:            test_access_qual
; CHECK-NEXT:   Language:        OpenCL C
; CHECK-NEXT:   LanguageVersion: [ 2, 0 ]
; CHECK-NEXT:   Args:
; CHECK-NEXT:     - Size:          8
; CHECK-NEXT:       Align:         8
; CHECK-NEXT:       Kind:          Image
; CHECK-NEXT:       ValueType:     Struct
; CHECK-NEXT:       AccQual:       ReadOnly
; CHECK-NEXT:       AddrSpaceQual: Global
; CHECK-NEXT:       TypeName:      image1d_t
; CHECK-NEXT:     - Size:          8
; CHECK-NEXT:       Align:         8
; CHECK-NEXT:       Kind:          Image
; CHECK-NEXT:       ValueType:     Struct
; CHECK-NEXT:       AccQual:       WriteOnly
; CHECK-NEXT:       AddrSpaceQual: Global
; CHECK-NEXT:       TypeName:      image2d_t
; CHECK-NEXT:     - Size:          8
; CHECK-NEXT:       Align:         8
; CHECK-NEXT:       Kind:          Image
; CHECK-NEXT:       ValueType:     Struct
; CHECK-NEXT:       AccQual:       ReadWrite
; CHECK-NEXT:       AddrSpaceQual: Global
; CHECK-NEXT:       TypeName:      image3d_t
; CHECK-NEXT:     - Size:          8
; CHECK-NEXT:       Align:         8
; CHECK-NEXT:       Kind:          HiddenGlobalOffsetX
; CHECK-NEXT:       ValueType:     I64
; CHECK-NEXT:     - Size:          8
; CHECK-NEXT:       Align:         8
; CHECK-NEXT:       Kind:          HiddenGlobalOffsetY
; CHECK-NEXT:       ValueType:     I64
; CHECK-NEXT:     - Size:          8
; CHECK-NEXT:       Align:         8
; CHECK-NEXT:       Kind:          HiddenGlobalOffsetZ
; CHECK-NEXT:       ValueType:     I64
; CHECK-NEXT:     - Size:          8
; CHECK-NEXT:       Align:         8
; CHECK-NEXT:       Kind:          HiddenPrintfBuffer
; CHECK-NEXT:       ValueType:     I8
; CHECK-NEXT:       AddrSpaceQual: Global
define amdgpu_kernel void @test_access_qual(%opencl.image1d_t addrspace(1)* %ro,
                                            %opencl.image2d_t addrspace(1)* %wo,
                                            %opencl.image3d_t addrspace(1)* %rw)
    !kernel_arg_addr_space !60 !kernel_arg_access_qual !61 !kernel_arg_type !62
    !kernel_arg_base_type !62 !kernel_arg_type_qual !25 {
  ret void
}

; CHECK:      - Name:            test_vec_type_hint_half
; CHECK-NEXT:   Language:        OpenCL C
; CHECK-NEXT:   LanguageVersion: [ 2, 0 ]
; CHECK-NEXT:   Attrs:
; CHECK-NEXT:       VecTypeHint:   half
; CHECK-NEXT:   Args:
; CHECK-NEXT:     - Size:          4
; CHECK-NEXT:       Align:         4
; CHECK-NEXT:       Kind:          ByValue
; CHECK-NEXT:       ValueType:     I32
; CHECK-NEXT:       AccQual:       Default
; CHECK-NEXT:       TypeName:      int
; CHECK-NEXT:     - Size:          8
; CHECK-NEXT:       Align:         8
; CHECK-NEXT:       Kind:          HiddenGlobalOffsetX
; CHECK-NEXT:       ValueType:     I64
; CHECK-NEXT:     - Size:          8
; CHECK-NEXT:       Align:         8
; CHECK-NEXT:       Kind:          HiddenGlobalOffsetY
; CHECK-NEXT:       ValueType:     I64
; CHECK-NEXT:     - Size:          8
; CHECK-NEXT:       Align:         8
; CHECK-NEXT:       Kind:          HiddenGlobalOffsetZ
; CHECK-NEXT:       ValueType:     I64
; CHECK-NEXT:     - Size:          8
; CHECK-NEXT:       Align:         8
; CHECK-NEXT:       Kind:          HiddenPrintfBuffer
; CHECK-NEXT:       ValueType:     I8
; CHECK-NEXT:       AddrSpaceQual: Global
define amdgpu_kernel void @test_vec_type_hint_half(i32 %a)
    !kernel_arg_addr_space !1 !kernel_arg_access_qual !2 !kernel_arg_type !3
    !kernel_arg_base_type !3 !kernel_arg_type_qual !4 !vec_type_hint !26 {
  ret void
}

; CHECK:      - Name:            test_vec_type_hint_float
; CHECK-NEXT:   Language:        OpenCL C
; CHECK-NEXT:   LanguageVersion: [ 2, 0 ]
; CHECK-NEXT:   Attrs:
; CHECK-NEXT:       VecTypeHint:   float
; CHECK-NEXT:   Args:
; CHECK-NEXT:     - Size:          4
; CHECK-NEXT:       Align:         4
; CHECK-NEXT:       Kind:          ByValue
; CHECK-NEXT:       ValueType:     I32
; CHECK-NEXT:       AccQual:       Default
; CHECK-NEXT:       TypeName:      int
; CHECK-NEXT:     - Size:          8
; CHECK-NEXT:       Align:         8
; CHECK-NEXT:       Kind:          HiddenGlobalOffsetX
; CHECK-NEXT:       ValueType:     I64
; CHECK-NEXT:     - Size:          8
; CHECK-NEXT:       Align:         8
; CHECK-NEXT:       Kind:          HiddenGlobalOffsetY
; CHECK-NEXT:       ValueType:     I64
; CHECK-NEXT:     - Size:          8
; CHECK-NEXT:       Align:         8
; CHECK-NEXT:       Kind:          HiddenGlobalOffsetZ
; CHECK-NEXT:       ValueType:     I64
; CHECK-NEXT:     - Size:          8
; CHECK-NEXT:       Align:         8
; CHECK-NEXT:       Kind:          HiddenPrintfBuffer
; CHECK-NEXT:       ValueType:     I8
; CHECK-NEXT:       AddrSpaceQual: Global
define amdgpu_kernel void @test_vec_type_hint_float(i32 %a)
    !kernel_arg_addr_space !1 !kernel_arg_access_qual !2 !kernel_arg_type !3
    !kernel_arg_base_type !3 !kernel_arg_type_qual !4 !vec_type_hint !27 {
  ret void
}

; CHECK:      - Name:            test_vec_type_hint_double
; CHECK-NEXT:   Language:        OpenCL C
; CHECK-NEXT:   LanguageVersion: [ 2, 0 ]
; CHECK-NEXT:   Attrs:
; CHECK-NEXT:       VecTypeHint:   double
; CHECK-NEXT:   Args:
; CHECK-NEXT:     - Size:          4
; CHECK-NEXT:       Align:         4
; CHECK-NEXT:       Kind:          ByValue
; CHECK-NEXT:       ValueType:     I32
; CHECK-NEXT:       AccQual:       Default
; CHECK-NEXT:       TypeName:      int
; CHECK-NEXT:     - Size:          8
; CHECK-NEXT:       Align:         8
; CHECK-NEXT:       Kind:          HiddenGlobalOffsetX
; CHECK-NEXT:       ValueType:     I64
; CHECK-NEXT:     - Size:          8
; CHECK-NEXT:       Align:         8
; CHECK-NEXT:       Kind:          HiddenGlobalOffsetY
; CHECK-NEXT:       ValueType:     I64
; CHECK-NEXT:     - Size:          8
; CHECK-NEXT:       Align:         8
; CHECK-NEXT:       Kind:          HiddenGlobalOffsetZ
; CHECK-NEXT:       ValueType:     I64
; CHECK-NEXT:     - Size:          8
; CHECK-NEXT:       Align:         8
; CHECK-NEXT:       Kind:          HiddenPrintfBuffer
; CHECK-NEXT:       ValueType:     I8
; CHECK-NEXT:       AddrSpaceQual: Global
define amdgpu_kernel void @test_vec_type_hint_double(i32 %a)
    !kernel_arg_addr_space !1 !kernel_arg_access_qual !2 !kernel_arg_type !3
    !kernel_arg_base_type !3 !kernel_arg_type_qual !4 !vec_type_hint !28 {
  ret void
}

; CHECK:      - Name:            test_vec_type_hint_char
; CHECK-NEXT:   Language:        OpenCL C
; CHECK-NEXT:   LanguageVersion: [ 2, 0 ]
; CHECK-NEXT:   Attrs:
; CHECK-NEXT:       VecTypeHint:   char
; CHECK-NEXT:   Args:
; CHECK-NEXT:     - Size:          4
; CHECK-NEXT:       Align:         4
; CHECK-NEXT:       Kind:          ByValue
; CHECK-NEXT:       ValueType:     I32
; CHECK-NEXT:       AccQual:       Default
; CHECK-NEXT:       TypeName:      int
; CHECK-NEXT:     - Size:          8
; CHECK-NEXT:       Align:         8
; CHECK-NEXT:       Kind:          HiddenGlobalOffsetX
; CHECK-NEXT:       ValueType:     I64
; CHECK-NEXT:     - Size:          8
; CHECK-NEXT:       Align:         8
; CHECK-NEXT:       Kind:          HiddenGlobalOffsetY
; CHECK-NEXT:       ValueType:     I64
; CHECK-NEXT:     - Size:          8
; CHECK-NEXT:       Align:         8
; CHECK-NEXT:       Kind:          HiddenGlobalOffsetZ
; CHECK-NEXT:       ValueType:     I64
; CHECK-NEXT:     - Size:          8
; CHECK-NEXT:       Align:         8
; CHECK-NEXT:       Kind:          HiddenPrintfBuffer
; CHECK-NEXT:       ValueType:     I8
; CHECK-NEXT:       AddrSpaceQual: Global
define amdgpu_kernel void @test_vec_type_hint_char(i32 %a)
    !kernel_arg_addr_space !1 !kernel_arg_access_qual !2 !kernel_arg_type !3
    !kernel_arg_base_type !3 !kernel_arg_type_qual !4 !vec_type_hint !29 {
  ret void
}

; CHECK:      - Name:            test_vec_type_hint_short
; CHECK-NEXT:   Language:        OpenCL C
; CHECK-NEXT:   LanguageVersion: [ 2, 0 ]
; CHECK-NEXT:   Attrs:
; CHECK-NEXT:       VecTypeHint:   short
; CHECK-NEXT:   Args:
; CHECK-NEXT:     - Size:          4
; CHECK-NEXT:       Align:         4
; CHECK-NEXT:       Kind:          ByValue
; CHECK-NEXT:       ValueType:     I32
; CHECK-NEXT:       AccQual:       Default
; CHECK-NEXT:       TypeName:      int
; CHECK-NEXT:     - Size:          8
; CHECK-NEXT:       Align:         8
; CHECK-NEXT:       Kind:          HiddenGlobalOffsetX
; CHECK-NEXT:       ValueType:     I64
; CHECK-NEXT:     - Size:          8
; CHECK-NEXT:       Align:         8
; CHECK-NEXT:       Kind:          HiddenGlobalOffsetY
; CHECK-NEXT:       ValueType:     I64
; CHECK-NEXT:     - Size:          8
; CHECK-NEXT:       Align:         8
; CHECK-NEXT:       Kind:          HiddenGlobalOffsetZ
; CHECK-NEXT:       ValueType:     I64
; CHECK-NEXT:     - Size:          8
; CHECK-NEXT:       Align:         8
; CHECK-NEXT:       Kind:          HiddenPrintfBuffer
; CHECK-NEXT:       ValueType:     I8
; CHECK-NEXT:       AddrSpaceQual: Global
define amdgpu_kernel void @test_vec_type_hint_short(i32 %a)
    !kernel_arg_addr_space !1 !kernel_arg_access_qual !2 !kernel_arg_type !3
    !kernel_arg_base_type !3 !kernel_arg_type_qual !4 !vec_type_hint !30 {
  ret void
}

; CHECK:      - Name:            test_vec_type_hint_long
; CHECK-NEXT:   Language:        OpenCL C
; CHECK-NEXT:   LanguageVersion: [ 2, 0 ]
; CHECK-NEXT:   Attrs:
; CHECK-NEXT:       VecTypeHint:   long
; CHECK-NEXT:   Args:
; CHECK-NEXT:     - Size:          4
; CHECK-NEXT:       Align:         4
; CHECK-NEXT:       Kind:          ByValue
; CHECK-NEXT:       ValueType:     I32
; CHECK-NEXT:       AccQual:       Default
; CHECK-NEXT:       TypeName:      int
; CHECK-NEXT:     - Size:          8
; CHECK-NEXT:       Align:         8
; CHECK-NEXT:       Kind:          HiddenGlobalOffsetX
; CHECK-NEXT:       ValueType:     I64
; CHECK-NEXT:     - Size:          8
; CHECK-NEXT:       Align:         8
; CHECK-NEXT:       Kind:          HiddenGlobalOffsetY
; CHECK-NEXT:       ValueType:     I64
; CHECK-NEXT:     - Size:          8
; CHECK-NEXT:       Align:         8
; CHECK-NEXT:       Kind:          HiddenGlobalOffsetZ
; CHECK-NEXT:       ValueType:     I64
; CHECK-NEXT:     - Size:          8
; CHECK-NEXT:       Align:         8
; CHECK-NEXT:       Kind:          HiddenPrintfBuffer
; CHECK-NEXT:       ValueType:     I8
; CHECK-NEXT:       AddrSpaceQual: Global
define amdgpu_kernel void @test_vec_type_hint_long(i32 %a)
    !kernel_arg_addr_space !1 !kernel_arg_access_qual !2 !kernel_arg_type !3
    !kernel_arg_base_type !3 !kernel_arg_type_qual !4 !vec_type_hint !31 {
  ret void
}

; CHECK:      - Name:            test_vec_type_hint_unknown
; CHECK-NEXT:   Language:        OpenCL C
; CHECK-NEXT:   LanguageVersion: [ 2, 0 ]
; CHECK-NEXT:   Attrs:
; CHECK-NEXT:       VecTypeHint:   unknown
; CHECK-NEXT:   Args:
; CHECK-NEXT:     - Size:          4
; CHECK-NEXT:       Align:         4
; CHECK-NEXT:       Kind:          ByValue
; CHECK-NEXT:       ValueType:     I32
; CHECK-NEXT:       AccQual:       Default
; CHECK-NEXT:       TypeName:      int
; CHECK-NEXT:     - Size:          8
; CHECK-NEXT:       Align:         8
; CHECK-NEXT:       Kind:          HiddenGlobalOffsetX
; CHECK-NEXT:       ValueType:     I64
; CHECK-NEXT:     - Size:          8
; CHECK-NEXT:       Align:         8
; CHECK-NEXT:       Kind:          HiddenGlobalOffsetY
; CHECK-NEXT:       ValueType:     I64
; CHECK-NEXT:     - Size:          8
; CHECK-NEXT:       Align:         8
; CHECK-NEXT:       Kind:          HiddenGlobalOffsetZ
; CHECK-NEXT:       ValueType:     I64
; CHECK-NEXT:     - Size:          8
; CHECK-NEXT:       Align:         8
; CHECK-NEXT:       Kind:          HiddenPrintfBuffer
; CHECK-NEXT:       ValueType:     I8
; CHECK-NEXT:       AddrSpaceQual: Global
define amdgpu_kernel void @test_vec_type_hint_unknown(i32 %a)
    !kernel_arg_addr_space !1 !kernel_arg_access_qual !2 !kernel_arg_type !3
    !kernel_arg_base_type !3 !kernel_arg_type_qual !4 !vec_type_hint !32 {
  ret void
}

; CHECK:      - Name:            test_reqd_wgs_vec_type_hint
; CHECK-NEXT:   Language:        OpenCL C
; CHECK-NEXT:   LanguageVersion: [ 2, 0 ]
; CHECK-NEXT:   Attrs:
; CHECK-NEXT:       ReqdWorkGroupSize: [ 1, 2, 4 ]
; CHECK-NEXT:       VecTypeHint:       int
; CHECK-NEXT:   Args:
; CHECK-NEXT:     - Size:              4
; CHECK-NEXT:       Align:             4
; CHECK-NEXT:       Kind:              ByValue
; CHECK-NEXT:       ValueType:         I32
; CHECK-NEXT:       AccQual:           Default
; CHECK-NEXT:       TypeName:          int
; CHECK-NEXT:     - Size:              8
; CHECK-NEXT:       Align:             8
; CHECK-NEXT:       Kind:              HiddenGlobalOffsetX
; CHECK-NEXT:       ValueType:         I64
; CHECK-NEXT:     - Size:              8
; CHECK-NEXT:       Align:             8
; CHECK-NEXT:       Kind:              HiddenGlobalOffsetY
; CHECK-NEXT:       ValueType:         I64
; CHECK-NEXT:     - Size:              8
; CHECK-NEXT:       Align:             8
; CHECK-NEXT:       Kind:              HiddenGlobalOffsetZ
; CHECK-NEXT:       ValueType:         I64
; CHECK-NEXT:     - Size:              8
; CHECK-NEXT:       Align:             8
; CHECK-NEXT:       Kind:              HiddenPrintfBuffer
; CHECK-NEXT:       ValueType:         I8
; CHECK-NEXT:       AddrSpaceQual:     Global
define amdgpu_kernel void @test_reqd_wgs_vec_type_hint(i32 %a)
    !kernel_arg_addr_space !1 !kernel_arg_access_qual !2 !kernel_arg_type !3
    !kernel_arg_base_type !3 !kernel_arg_type_qual !4 !vec_type_hint !5
    !reqd_work_group_size !6 {
  ret void
}

; CHECK:      - Name:            test_wgs_hint_vec_type_hint
; CHECK-NEXT:   Language:        OpenCL C
; CHECK-NEXT:   LanguageVersion: [ 2, 0 ]
; CHECK-NEXT:   Attrs:
; CHECK-NEXT:       WorkGroupSizeHint: [ 8, 16, 32 ]
; CHECK-NEXT:       VecTypeHint:       uint4
; CHECK-NEXT:   Args:
; CHECK-NEXT:     - Size:              4
; CHECK-NEXT:       Align:             4
; CHECK-NEXT:       Kind:              ByValue
; CHECK-NEXT:       ValueType:         I32
; CHECK-NEXT:       AccQual:           Default
; CHECK-NEXT:       TypeName:          int
; CHECK-NEXT:     - Size:              8
; CHECK-NEXT:       Align:             8
; CHECK-NEXT:       Kind:              HiddenGlobalOffsetX
; CHECK-NEXT:       ValueType:         I64
; CHECK-NEXT:     - Size:              8
; CHECK-NEXT:       Align:             8
; CHECK-NEXT:       Kind:              HiddenGlobalOffsetY
; CHECK-NEXT:       ValueType:         I64
; CHECK-NEXT:     - Size:              8
; CHECK-NEXT:       Align:             8
; CHECK-NEXT:       Kind:              HiddenGlobalOffsetZ
; CHECK-NEXT:       ValueType:         I64
; CHECK-NEXT:     - Size:              8
; CHECK-NEXT:       Align:             8
; CHECK-NEXT:       Kind:              HiddenPrintfBuffer
; CHECK-NEXT:       ValueType:         I8
; CHECK-NEXT:       AddrSpaceQual:     Global
define amdgpu_kernel void @test_wgs_hint_vec_type_hint(i32 %a)
    !kernel_arg_addr_space !1 !kernel_arg_access_qual !2 !kernel_arg_type !3
    !kernel_arg_base_type !3 !kernel_arg_type_qual !4 !vec_type_hint !7
    !work_group_size_hint !8 {
  ret void
}

; CHECK:      - Name:            test_arg_ptr_to_ptr
; CHECK-NEXT:   Language:        OpenCL C
; CHECK-NEXT:   LanguageVersion: [ 2, 0 ]
; CHECK-NEXT:   Args:
; CHECK-NEXT:     - Size:          8
; CHECK-NEXT:       Align:         8
; CHECK-NEXT:       Kind:          GlobalBuffer
; CHECK-NEXT:       ValueType:     I32
; CHECK-NEXT:       AccQual:       Default
; CHECK-NEXT:       AddrSpaceQual: Global
; CHECK-NEXT:       TypeName:      'int **'
; CHECK-NEXT:     - Size:          8
; CHECK-NEXT:       Align:         8
; CHECK-NEXT:       Kind:          HiddenGlobalOffsetX
; CHECK-NEXT:       ValueType:     I64
; CHECK-NEXT:     - Size:          8
; CHECK-NEXT:       Align:         8
; CHECK-NEXT:       Kind:          HiddenGlobalOffsetY
; CHECK-NEXT:       ValueType:     I64
; CHECK-NEXT:     - Size:          8
; CHECK-NEXT:       Align:         8
; CHECK-NEXT:       Kind:          HiddenGlobalOffsetZ
; CHECK-NEXT:       ValueType:     I64
; CHECK-NEXT:     - Size:          8
; CHECK-NEXT:       Align:         8
; CHECK-NEXT:       Kind:          HiddenPrintfBuffer
; CHECK-NEXT:       ValueType:     I8
; CHECK-NEXT:       AddrSpaceQual: Global
define amdgpu_kernel void @test_arg_ptr_to_ptr(i32* addrspace(1)* %a)
    !kernel_arg_addr_space !81 !kernel_arg_access_qual !2 !kernel_arg_type !80
    !kernel_arg_base_type !80 !kernel_arg_type_qual !4 {
  ret void
}

; CHECK:      - Name:            test_arg_struct_contains_ptr
; CHECK-NEXT:   Language:        OpenCL C
; CHECK-NEXT:   LanguageVersion: [ 2, 0 ]
; CHECK-NEXT:   Args:
; CHECK-NEXT:     - Size:          4
; CHECK-NEXT:       Align:         4
; CHECK-NEXT:       Kind:          GlobalBuffer
; CHECK-NEXT:       ValueType:     Struct
; CHECK-NEXT:       AccQual:       Default
; CHECK-NEXT:       AddrSpaceQual: Private
; CHECK-NEXT:       TypeName:      struct B
; CHECK-NEXT:     - Size:          8
; CHECK-NEXT:       Align:         8
; CHECK-NEXT:       Kind:          HiddenGlobalOffsetX
; CHECK-NEXT:       ValueType:     I64
; CHECK-NEXT:     - Size:          8
; CHECK-NEXT:       Align:         8
; CHECK-NEXT:       Kind:          HiddenGlobalOffsetY
; CHECK-NEXT:       ValueType:     I64
; CHECK-NEXT:     - Size:          8
; CHECK-NEXT:       Align:         8
; CHECK-NEXT:       Kind:          HiddenGlobalOffsetZ
; CHECK-NEXT:       ValueType:     I64
; CHECK-NEXT:     - Size:          8
; CHECK-NEXT:       Align:         8
; CHECK-NEXT:       Kind:          HiddenPrintfBuffer
; CHECK-NEXT:       ValueType:     I8
; CHECK-NEXT:       AddrSpaceQual: Global
define amdgpu_kernel void @test_arg_struct_contains_ptr(%struct.B* byval %a)
    !kernel_arg_addr_space !1 !kernel_arg_access_qual !2 !kernel_arg_type !82
    !kernel_arg_base_type !82 !kernel_arg_type_qual !4 {
 ret void
}

; CHECK:      - Name:            test_arg_vector_of_ptr
; CHECK-NEXT:   Language:        OpenCL C
; CHECK-NEXT:   LanguageVersion: [ 2, 0 ]
; CHECK-NEXT:   Args:
; CHECK-NEXT:     - Size:          16
; CHECK-NEXT:       Align:         16
; CHECK-NEXT:       Kind:          ByValue
; CHECK-NEXT:       ValueType:     I32
; CHECK-NEXT:       AccQual:       Default
; CHECK-NEXT:       TypeName:      'global int* __attribute__((ext_vector_type(2)))'
; CHECK-NEXT:     - Size:          8
; CHECK-NEXT:       Align:         8
; CHECK-NEXT:       Kind:          HiddenGlobalOffsetX
; CHECK-NEXT:       ValueType:     I64
; CHECK-NEXT:     - Size:          8
; CHECK-NEXT:       Align:         8
; CHECK-NEXT:       Kind:          HiddenGlobalOffsetY
; CHECK-NEXT:       ValueType:     I64
; CHECK-NEXT:     - Size:          8
; CHECK-NEXT:       Align:         8
; CHECK-NEXT:       Kind:          HiddenGlobalOffsetZ
; CHECK-NEXT:       ValueType:     I64
; CHECK-NEXT:     - Size:          8
; CHECK-NEXT:       Align:         8
; CHECK-NEXT:       Kind:          HiddenPrintfBuffer
; CHECK-NEXT:       ValueType:     I8
; CHECK-NEXT:       AddrSpaceQual: Global
define amdgpu_kernel void @test_arg_vector_of_ptr(<2 x i32 addrspace(1)*> %a)
    !kernel_arg_addr_space !1 !kernel_arg_access_qual !2 !kernel_arg_type !83
    !kernel_arg_base_type !83 !kernel_arg_type_qual !4 {
  ret void
}

; CHECK:      - Name:            test_arg_unknown_builtin_type
; CHECK-NEXT:   Language:        OpenCL C
; CHECK-NEXT:   LanguageVersion: [ 2, 0 ]
; CHECK-NEXT:   Args:
; CHECK-NEXT:     - Size:          8
; CHECK-NEXT:       Align:         8
; CHECK-NEXT:       Kind:          GlobalBuffer
; CHECK-NEXT:       ValueType:     Struct
; CHECK-NEXT:       AccQual:       Default
; CHECK-NEXT:       AddrSpaceQual: Global
; CHECK-NEXT:       TypeName:      clk_event_t
; CHECK-NEXT:     - Size:          8
; CHECK-NEXT:       Align:         8
; CHECK-NEXT:       Kind:          HiddenGlobalOffsetX
; CHECK-NEXT:       ValueType:     I64
; CHECK-NEXT:     - Size:          8
; CHECK-NEXT:       Align:         8
; CHECK-NEXT:       Kind:          HiddenGlobalOffsetY
; CHECK-NEXT:       ValueType:     I64
; CHECK-NEXT:     - Size:          8
; CHECK-NEXT:       Align:         8
; CHECK-NEXT:       Kind:          HiddenGlobalOffsetZ
; CHECK-NEXT:       ValueType:     I64
; CHECK-NEXT:     - Size:          8
; CHECK-NEXT:       Align:         8
; CHECK-NEXT:       Kind:          HiddenPrintfBuffer
; CHECK-NEXT:       ValueType:     I8
; CHECK-NEXT:       AddrSpaceQual: Global
define amdgpu_kernel void @test_arg_unknown_builtin_type(
    %opencl.clk_event_t addrspace(1)* %a)
    !kernel_arg_addr_space !81 !kernel_arg_access_qual !2 !kernel_arg_type !84
    !kernel_arg_base_type !84 !kernel_arg_type_qual !4 {
  ret void
}

; CHECK:      - Name:            test_pointee_align
; CHECK-NEXT:   Language:        OpenCL C
; CHECK-NEXT:   LanguageVersion: [ 2, 0 ]
; CHECK-NEXT:   Args:
; CHECK-NEXT:     - Size:          8
; CHECK-NEXT:       Align:         8
; CHECK-NEXT:       Kind:          GlobalBuffer
; CHECK-NEXT:       ValueType:     I64
; CHECK-NEXT:       AccQual:       Default
; CHECK-NEXT:       AddrSpaceQual: Global
; CHECK-NEXT:       TypeName:      'long *'
; CHECK-NEXT:     - Size:          4
; CHECK-NEXT:       Align:         4
; CHECK-NEXT:       Kind:          DynamicSharedPointer
; CHECK-NEXT:       ValueType:     I8
; CHECK-NEXT:       PointeeAlign:  1
; CHECK-NEXT:       AccQual:       Default
; CHECK-NEXT:       AddrSpaceQual: Local
; CHECK-NEXT:       TypeName:      'char *'
; CHECK-NEXT:     - Size:          4
; CHECK-NEXT:       Align:         4
; CHECK-NEXT:       Kind:          DynamicSharedPointer
; CHECK-NEXT:       ValueType:     I8
; CHECK-NEXT:       PointeeAlign:  2
; CHECK-NEXT:       AccQual:       Default
; CHECK-NEXT:       AddrSpaceQual: Local
; CHECK-NEXT:       TypeName:      'char2 *'
; CHECK-NEXT:     - Size:          4
; CHECK-NEXT:       Align:         4
; CHECK-NEXT:       Kind:          DynamicSharedPointer
; CHECK-NEXT:       ValueType:     I8
; CHECK-NEXT:       PointeeAlign:  4
; CHECK-NEXT:       AccQual:       Default
; CHECK-NEXT:       AddrSpaceQual: Local
; CHECK-NEXT:       TypeName:      'char3 *'
; CHECK-NEXT:     - Size:          4
; CHECK-NEXT:       Align:         4
; CHECK-NEXT:       Kind:          DynamicSharedPointer
; CHECK-NEXT:       ValueType:     I8
; CHECK-NEXT:       PointeeAlign:  4
; CHECK-NEXT:       AccQual:       Default
; CHECK-NEXT:       AddrSpaceQual: Local
; CHECK-NEXT:       TypeName:      'char4 *'
; CHECK-NEXT:     - Size:          4
; CHECK-NEXT:       Align:         4
; CHECK-NEXT:       Kind:          DynamicSharedPointer
; CHECK-NEXT:       ValueType:     I8
; CHECK-NEXT:       PointeeAlign:  8
; CHECK-NEXT:       AccQual:       Default
; CHECK-NEXT:       AddrSpaceQual: Local
; CHECK-NEXT:       TypeName:      'char8 *'
; CHECK-NEXT:     - Size:          4
; CHECK-NEXT:       Align:         4
; CHECK-NEXT:       Kind:          DynamicSharedPointer
; CHECK-NEXT:       ValueType:     I8
; CHECK-NEXT:       PointeeAlign:  16
; CHECK-NEXT:       AccQual:       Default
; CHECK-NEXT:       AddrSpaceQual: Local
; CHECK-NEXT:       TypeName:      'char16 *'
; CHECK-NEXT:     - Size:          8
; CHECK-NEXT:       Align:         8
; CHECK-NEXT:       Kind:          HiddenGlobalOffsetX
; CHECK-NEXT:       ValueType:     I64
; CHECK-NEXT:     - Size:          8
; CHECK-NEXT:       Align:         8
; CHECK-NEXT:       Kind:          HiddenGlobalOffsetY
; CHECK-NEXT:       ValueType:     I64
; CHECK-NEXT:     - Size:          8
; CHECK-NEXT:       Align:         8
; CHECK-NEXT:       Kind:          HiddenGlobalOffsetZ
; CHECK-NEXT:       ValueType:     I64
; CHECK-NEXT:     - Size:          8
; CHECK-NEXT:       Align:         8
; CHECK-NEXT:       Kind:          HiddenPrintfBuffer
; CHECK-NEXT:       ValueType:     I8
; CHECK-NEXT:       AddrSpaceQual: Global
define amdgpu_kernel void @test_pointee_align(i64 addrspace(1)* %a,
                                              i8 addrspace(3)* %b,
                                              <2 x i8> addrspace(3)* %c,
                                              <3 x i8> addrspace(3)* %d,
                                              <4 x i8> addrspace(3)* %e,
                                              <8 x i8> addrspace(3)* %f,
                                              <16 x i8> addrspace(3)* %g)
    !kernel_arg_addr_space !91 !kernel_arg_access_qual !92 !kernel_arg_type !93
    !kernel_arg_base_type !93 !kernel_arg_type_qual !94 {
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

; NOTES: Displaying notes found at file offset 0x{{[0-9]+}}
; NOTES-NEXT: Owner    Data size    Description
; NOTES-NEXT: AMD      0x00000008   Unknown note type: (0x00000001)
; NOTES-NEXT: AMD      0x0000001b   Unknown note type: (0x00000003)
; GFX700:     AMD      0x00009171   Unknown note type: (0x0000000a)
; GFX800:     AMD      0x00009190   Unknown note type: (0x0000000a)
; GFX900:     AMD      0x00009171   Unknown note type: (0x0000000a)

; PARSER: AMDGPU Code Object Metadata Parser Test: PASS

; RUN: llc -mtriple=amdgcn-amd-amdhsa --amdhsa-code-object-version=2 -mcpu=gfx900 -filetype=obj -o - < %s | llvm-readelf --notes - | FileCheck --check-prefix=CHECK --check-prefix=GFX900 --check-prefix=NOTES %s
; RUN: llc -mtriple=amdgcn-amd-amdhsa --amdhsa-code-object-version=2 -mcpu=gfx900 -amdgpu-dump-hsa-metadata -amdgpu-verify-hsa-metadata -filetype=obj -o - < %s 2>&1 | FileCheck --check-prefix=PARSER %s

; CHECK: ---
; CHECK:  Version: [ 1, 0 ]
; CHECK-NOT:  Printf:
; CHECK:  Kernels:

; CHECK:      - Name:            test_non_enqueue_kernel_caller
; CHECK-NEXT:   SymbolName:      'test_non_enqueue_kernel_caller@kd'
; CHECK-NEXT:   Language:        OpenCL C
; CHECK-NEXT:   LanguageVersion: [ 2, 0 ]
; CHECK-NEXT:   Args:
; CHECK-NEXT:     - Name:          a
; CHECK-NEXT:       TypeName:      char
; CHECK-NEXT:       Size:          1
; CHECK-NEXT:       Align:         1
; CHECK-NEXT:       ValueKind:     ByValue
; CHECK-NEXT:       AccQual:       Default
; CHECK-NEXT:     - Size:          8
; CHECK-NEXT:       Align:         8
; CHECK-NEXT:       ValueKind:     HiddenGlobalOffsetX
; CHECK-NEXT:     - Size:          8
; CHECK-NEXT:       Align:         8
; CHECK-NEXT:       ValueKind:     HiddenGlobalOffsetY
; CHECK-NEXT:     - Size:          8
; CHECK-NEXT:       Align:         8
; CHECK-NEXT:       ValueKind:     HiddenGlobalOffsetZ
; CHECK-NOT:        ValueKind:     HiddenDefaultQueue
; CHECK-NOT:        ValueKind:     HiddenCompletionAction
define amdgpu_kernel void @test_non_enqueue_kernel_caller(i8 %a) #0
    !kernel_arg_addr_space !1 !kernel_arg_access_qual !2 !kernel_arg_type !3
    !kernel_arg_base_type !3 !kernel_arg_type_qual !4 {
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
; CHECK-NEXT:       AccQual:       Default
; CHECK-NEXT:     - Size:          8
; CHECK-NEXT:       Align:         8
; CHECK-NEXT:       ValueKind:     HiddenGlobalOffsetX
; CHECK-NEXT:     - Size:          8
; CHECK-NEXT:       Align:         8
; CHECK-NEXT:       ValueKind:     HiddenGlobalOffsetY
; CHECK-NEXT:     - Size:          8
; CHECK-NEXT:       Align:         8
; CHECK-NEXT:       ValueKind:     HiddenGlobalOffsetZ
; CHECK-NEXT:     - Size:          8
; CHECK-NEXT:       Align:         8
; CHECK-NEXT:       ValueKind:     HiddenNone
; CHECK-NEXT:       AddrSpaceQual: Global
; CHECK-NEXT:     - Size:          8
; CHECK-NEXT:       Align:         8
; CHECK-NEXT:       ValueKind:     HiddenDefaultQueue
; CHECK-NEXT:       AddrSpaceQual: Global
; CHECK-NEXT:     - Size:          8
; CHECK-NEXT:       Align:         8
; CHECK-NEXT:       ValueKind:     HiddenCompletionAction
; CHECK-NEXT:       AddrSpaceQual: Global
define amdgpu_kernel void @test_enqueue_kernel_caller(i8 %a) #1
    !kernel_arg_addr_space !1 !kernel_arg_access_qual !2 !kernel_arg_type !3
    !kernel_arg_base_type !3 !kernel_arg_type_qual !4 {
  ret void
}

attributes #0 = { "amdgpu-implicitarg-num-bytes"="48" }
attributes #1 = { "calls-enqueue-kernel" "amdgpu-implicitarg-num-bytes"="48" }

!1 = !{i32 0}
!2 = !{!"none"}
!3 = !{!"char"}
!4 = !{!""}

!opencl.ocl.version = !{!90}
!90 = !{i32 2, i32 0}

; PARSER: AMDGPU HSA Metadata Parser Test: PASS

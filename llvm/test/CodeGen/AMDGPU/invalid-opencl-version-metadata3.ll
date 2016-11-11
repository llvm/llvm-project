; RUN: llc -mtriple=amdgcn--amdhsa < %s | FileCheck %s
; check llc does not crash for invalid opencl version metadata

; CHECK: .section        .note,#alloc
; CHECK-NEXT: .long   4
; CHECK-NEXT: .long   {{.+}}
; CHECK-NEXT: .long   7
; CHECK-NEXT: .asciz  "AMD"

!opencl.ocl.version = !{!0}
!0 = !{i32 1}

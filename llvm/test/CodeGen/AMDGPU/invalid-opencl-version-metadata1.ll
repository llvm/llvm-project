; RUN: llc -mtriple=amdgcn--amdhsa < %s | FileCheck %s
; check llc does not crash for invalid opencl version metadata

; CHECK: .section        .AMDGPU.runtime_metadata
; CHECK-NEXT: .byte	1
; CHECK-NEXT: .short	256

!opencl.ocl.version = !{}

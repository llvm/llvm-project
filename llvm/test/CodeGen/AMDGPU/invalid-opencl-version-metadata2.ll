; RUN: llc -mtriple=amdgcn--amdhsa -filetype=obj -o - < %s | llvm-readobj -amdgpu-runtime-metadata | FileCheck %s
; check llc does not crash for invalid opencl version metadata

; CHECK: { amd.MDVersion: [ 2, 0 ] }

!opencl.ocl.version = !{!0}
!0 = !{}

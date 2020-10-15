; RUN: llc -mtriple=amdgcn-amd-amdhsa --amdhsa-code-object-version=2 -filetype=obj -o - < %s | llvm-readelf --notes - | FileCheck %s

; Make sure llc does not crash for invalid opencl version metadata.

; CHECK: ---
; CHECK: Version: [ 1, 0 ]
; CHECK: ...

!opencl.ocl.version = !{!0}
!0 = !{i32 1}

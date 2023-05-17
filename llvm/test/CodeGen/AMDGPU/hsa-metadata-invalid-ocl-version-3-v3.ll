; RUN: llc -mtriple=amdgcn-amd-amdhsa -filetype=obj -o - < %s | llvm-readelf --notes - | FileCheck %s

; Make sure llc does not crash for invalid opencl version metadata.

; CHECK: ---
; CHECK: amdhsa.version:
; CHECK-NEXT: - 1
; CHECK-NEXT: - 0
; CHECK: ...

!opencl.ocl.version = !{!0}
!llvm.module.flags = !{!1}
!0 = !{i32 1}
!1 = !{i32 1, !"amdgpu_code_object_version", i32 300}

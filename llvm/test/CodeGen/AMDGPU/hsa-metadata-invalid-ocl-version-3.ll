; RUN: llc -mtriple=amdgcn-amd-amdhsa -filetype=obj -o - < %s | llvm-readobj -elf-output-style=GNU -notes | FileCheck %s

; Make sure llc does not crash for invalid opencl version metadata.

; CHECK: ---
; CHECK: Version: [ 1, 0 ]
; CHECK: ...

!opencl.ocl.version = !{!0}
!0 = !{i32 1}

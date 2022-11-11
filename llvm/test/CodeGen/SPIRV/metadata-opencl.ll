; RUN: split-file %s %t
; RUN: llc -O0 -mtriple=spirv32-unknown-unknown %t/metadata-opencl12.ll -o - | FileCheck %t/metadata-opencl12.ll
; RUN: llc -O0 -mtriple=spirv32-unknown-unknown %t/metadata-opencl20.ll -o - | FileCheck %t/metadata-opencl20.ll
; RUN: llc -O0 -mtriple=spirv32-unknown-unknown %t/metadata-opencl22.ll -o - | FileCheck %t/metadata-opencl22.ll

;--- metadata-opencl12.ll
!opencl.ocl.version = !{!0}
!0 = !{i32 1, i32 2}
; CHECK: OpSource OpenCL_C 102000

;--- metadata-opencl20.ll
!opencl.ocl.version = !{!0}
!0 = !{i32 2, i32 0}
; CHECK: OpSource OpenCL_C 200000

;--- metadata-opencl22.ll
!opencl.ocl.version = !{!0}
!0 = !{i32 2, i32 2}
; CHECK: OpSource OpenCL_C 202000

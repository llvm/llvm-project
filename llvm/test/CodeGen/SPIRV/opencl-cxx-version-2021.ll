; RUN: llc -O0 -mtriple=spirv64-unknown-unknown %s -o - | FileCheck %s
; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv64-unknown-unknown %s -o - -filetype=obj | spirv-val %}

; When both opencl.ocl.version (3.0) and opencl.cxx.version (2021) are present,
; the source language should be OpenCL C++ with version 202100000.

; CHECK: OpSource CPP_for_OpenCL 202100000

define spir_kernel void @foo() {
entry:
  ret void
}

!opencl.ocl.version = !{!0}
!opencl.cxx.version = !{!1}
!opencl.spir.version = !{!0}

!0 = !{i32 3, i32 0}
!1 = !{i32 2021, i32 0}

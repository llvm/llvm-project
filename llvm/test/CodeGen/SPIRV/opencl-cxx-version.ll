; RUN: llc -O0 -mtriple=spirv64-unknown-unknown %s -o - | FileCheck %s
; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv64-unknown-unknown %s -o - -filetype=obj | spirv-val %}

; When both opencl.ocl.version (2.0) and opencl.cxx.version (1.0) are present,
; the source language should be OpenCL C++ with version 100000

; CHECK: OpSource CPP_for_OpenCL 100000

target datalayout = "e-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024-G1"
target triple = "spir64-unknown-unknown"

define spir_kernel void @foo() {
entry:
  ret void
}

!opencl.ocl.version = !{!0}
!opencl.cxx.version = !{!1}
!opencl.spir.version = !{!0}

!0 = !{i32 2, i32 0}
!1 = !{i32 1, i32 0}

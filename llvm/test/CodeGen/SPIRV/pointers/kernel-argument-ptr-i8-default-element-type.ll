; RUN: llc -verify-machineinstrs -O0 -mtriple=spirv64-unknown-unknown %s -o - | FileCheck %s
; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv64-unknown-unknown %s -o - -filetype=obj | spirv-val %}

; CHECK-DAG: %[[#CHAR:]] = OpTypeInt 8
; CHECK-DAG: %[[#GLOBAL_PTR_CHAR:]] = OpTypePointer CrossWorkgroup %[[#CHAR]]

define spir_kernel void @foo(ptr addrspace(1) %arg) {
  ret void
}

; CHECK: %[[#]] = OpFunctionParameter %[[#GLOBAL_PTR_CHAR]]

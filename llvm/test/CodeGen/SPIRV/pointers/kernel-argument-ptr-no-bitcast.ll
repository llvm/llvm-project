; RUN: llc -O0 -mtriple=spirv64-unknown-unknown %s -o - | FileCheck %s
; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv64-unknown-unknown %s -o - -filetype=obj | spirv-val %}

; CHECK-DAG: %[[#CHAR:]] = OpTypeInt 8
; CHECK-DAG: %[[#GLOBAL_PTR_CHAR:]] = OpTypePointer CrossWorkgroup %[[#CHAR]]

define spir_kernel void @foo(i8 %a, ptr addrspace(1) %p) {
  store i8 %a, ptr addrspace(1) %p
  ret void
}

; CHECK: %[[#A:]] = OpFunctionParameter %[[#CHAR]]
; CHECK: %[[#P:]] = OpFunctionParameter %[[#GLOBAL_PTR_CHAR]]
; CHECK: OpStore %[[#P]] %[[#A]]

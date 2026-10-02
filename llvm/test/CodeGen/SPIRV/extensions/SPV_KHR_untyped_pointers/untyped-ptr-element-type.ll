; RUN: llc -O0 -verify-machineinstrs -mtriple=spirv64-unknown-unknown --spirv-ext=+SPV_KHR_untyped_pointers %s -o - | FileCheck %s
; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv64-unknown-unknown --spirv-ext=+SPV_KHR_untyped_pointers %s -o - -filetype=obj | spirv-val %}

; Untyped access chains carry the real element type as Base Type, not a fallback.

; CHECK-DAG: %[[#I32:]] = OpTypeInt 32 0
; CHECK-DAG: %[[#ARR:]] = OpTypeArray %[[#I32]]
; CHECK-DAG: %[[#CROSS:]] = OpTypeUntypedPointerKHR CrossWorkgroup
; CHECK-DAG: %[[#FUNC:]] = OpTypeUntypedPointerKHR Function

; A chained GEP carries the i32 Base Type on both links, including the one based
; on the previous chain's result.
; CHECK: %[[#P1:]] = OpUntypedPtrAccessChainKHR %[[#CROSS]] %[[#I32]] %[[#]] %[[#]]
; CHECK: OpUntypedPtrAccessChainKHR %[[#CROSS]] %[[#I32]] %[[#P1]] %[[#]]
define spir_kernel void @chain(ptr addrspace(1) %base, ptr addrspace(1) %out) {
  %p1 = getelementptr i32, ptr addrspace(1) %base, i64 5
  %p2 = getelementptr i32, ptr addrspace(1) %p1, i64 3
  %v = load i32, ptr addrspace(1) %p2
  store i32 %v, ptr addrspace(1) %out
  ret void
}

; An aggregate alloca uses the array type for both Data Type and Base Type.
; CHECK: %[[#VAR:]] = OpUntypedVariableKHR %[[#FUNC]] Function %[[#ARR]]
; CHECK: OpUntypedPtrAccessChainKHR %[[#FUNC]] %[[#ARR]] %[[#VAR]]
define spir_kernel void @agg(ptr addrspace(1) %out) {
  %arr = alloca [4 x i32]
  %e = getelementptr [4 x i32], ptr %arr, i64 0, i64 2
  store i32 7, ptr %e
  ret void
}

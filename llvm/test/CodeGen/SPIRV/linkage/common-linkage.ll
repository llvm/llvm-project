; Test that common linkage globals are emitted without an initializer

; RUN: llc -O0 -mtriple=spirv64-unknown-unknown %s -o - | FileCheck %s
; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv64-unknown-unknown %s -o - -filetype=obj | spirv-val %}
; RUN: llc -O0 -mtriple=spirv32-unknown-unknown %s -o - | FileCheck %s
; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv32-unknown-unknown %s -o - -filetype=obj | spirv-val %}

; CHECK-DAG: OpName %[[#COMMON_VAR:]] "c"
; CHECK-DAG: OpDecorate %[[#COMMON_VAR]] LinkageAttributes "c" Export

; CHECK-DAG: %[[#I32:]] = OpTypeInt 32 0
; CHECK-DAG: %[[#PTR_CW_I32:]] = OpTypePointer CrossWorkgroup %[[#I32]]

; Ensure OpVariable for @c has NO initializer operand.
; CHECK: %[[#COMMON_VAR]] = OpVariable %[[#PTR_CW_I32]] CrossWorkgroup
; CHECK-NOT: OpConstantNull

@c = common addrspace(1) global i32 0, align 4

define spir_kernel void @test(ptr addrspace(1) %out) {
entry:
  %val = load i32, ptr addrspace(1) @c, align 4
  store i32 %val, ptr addrspace(1) %out, align 4
  ret void
}

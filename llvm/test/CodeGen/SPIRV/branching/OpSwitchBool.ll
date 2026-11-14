; Check that a switch on an i1 is lowered to a conditional branch, since
; OpSwitch cannot take an OpTypeBool selector.

; RUN: llc -verify-machineinstrs -O0 -mtriple=spirv64-unknown-unknown %s -o - | FileCheck %s --implicit-check-not=OpSwitch
; RUN: llc -verify-machineinstrs -O2 -mtriple=spirv64-unknown-unknown %s -o - | FileCheck %s --implicit-check-not=OpSwitch
; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv64-unknown-unknown %s -o - -filetype=obj | spirv-val %}
; RUN: %if spirv-tools %{ llc -O2 -mtriple=spirv64-unknown-unknown %s -o - -filetype=obj | spirv-val %}

; CHECK-DAG: %[[#BOOL:]] = OpTypeBool
; CHECK-DAG: %[[#I32:]] = OpTypeInt 32 0
; CHECK-DAG: %[[#ONE:]] = OpConstant %[[#I32]] 1
; CHECK-DAG: %[[#TWO:]] = OpConstant %[[#I32]] 2
; CHECK: %[[#COND:]] = OpFunctionParameter %[[#BOOL]]
; CHECK: OpBranchConditional %[[#COND]] %[[#TRUE:]] %[[#FALSE:]]
; CHECK: %[[#FALSE]] = OpLabel
; CHECK-NEXT: OpStore %[[#]] %[[#ONE]]
; CHECK: %[[#TRUE]] = OpLabel
; CHECK-NEXT: OpStore %[[#]] %[[#TWO]]

define spir_kernel void @test(ptr addrspace(1) %p, i1 %c) {
entry:
  switch i1 %c, label %default [
    i1 false, label %false
    i1 true, label %true
  ]
false:
  store i32 1, ptr addrspace(1) %p
  ret void
true:
  store i32 2, ptr addrspace(1) %p
  ret void
default:
  ret void
}

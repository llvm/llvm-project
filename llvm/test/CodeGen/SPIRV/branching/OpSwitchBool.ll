; A switch on an i1, which InstCombine can leave behind when it narrows a
; switch's condition. OpSwitch takes an integer selector, so the condition is
; zero-extended rather than used as an OpTypeBool.

; RUN: llc -verify-machineinstrs -O0 -mtriple=spirv64-unknown-unknown %s -o - | FileCheck %s --check-prefixes=CHECK,CHECK-O0
; RUN: llc -verify-machineinstrs -O2 -mtriple=spirv64-unknown-unknown %s -o - | FileCheck %s
; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv64-unknown-unknown %s -o - -filetype=obj | spirv-val %}
; RUN: %if spirv-tools %{ llc -O2 -mtriple=spirv64-unknown-unknown %s -o - -filetype=obj | spirv-val %}

; CHECK-DAG: %[[#BOOL:]] = OpTypeBool
; CHECK-DAG: %[[#I32:]] = OpTypeInt 32 0
; CHECK-DAG: %[[#ONE:]] = OpConstant %[[#I32]] 1
; CHECK-DAG: %[[#ZERO:]] = OpConstantNull %[[#I32]]
; CHECK: %[[#COND:]] = OpFunctionParameter %[[#BOOL]]
; CHECK: %[[#SEL:]] = OpSelect %[[#I32]] %[[#COND]] %[[#ONE]] %[[#ZERO]]
; CHECK: OpSwitch %[[#SEL]] %[[#DEFAULT:]] 0 %[[#FALSE:]] 1 %[[#TRUE:]]

define spir_kernel void @both_cases(ptr addrspace(1) %p, i1 %c) {
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

; (at -O2, a switch with one case becomes a conditional branch first)
; CHECK-O0: %[[#COND2:]] = OpFunctionParameter %[[#BOOL]]
; CHECK-O0: %[[#SEL2:]] = OpSelect %[[#I32]] %[[#COND2]] %[[#ONE]] %[[#ZERO]]
; CHECK-O0: OpSwitch %[[#SEL2]] %[[#]] 1 %[[#]]

define spir_kernel void @one_case(ptr addrspace(1) %p, i1 %c) {
entry:
  switch i1 %c, label %default [
    i1 true, label %true
  ]
true:
  store i32 2, ptr addrspace(1) %p
  br label %default
default:
  ret void
}

; RUN: llc -verify-machineinstrs -O0 -mtriple=spirv64-unknown-unknown %s -o - | FileCheck %s
; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv64-unknown-unknown %s -o - -filetype=obj | spirv-val %}

; RUN: llc -verify-machineinstrs -O0 -mtriple=spirv32-unknown-unknown %s -o - | FileCheck %s
; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv32-unknown-unknown %s -o - -filetype=obj | spirv-val %}

; CHECK-DAG: %[[#BoolTy:]] = OpTypeBool
; CHECK-DAG: %[[#BoolPtrTy:]] = OpTypePointer Function %[[#BoolTy]]
; CHECK-DAG: %[[#False:]] = OpConstantFalse %[[#BoolTy]]
; CHECK: OpFunction
; CHECK: %[[#Ptr:]] = OpVariable %[[#BoolPtrTy]] Function
; CHECK: OpStore %[[#Ptr]] %[[#False]] Aligned 1

define spir_func void @foo() {
entry:
  %bvar = alloca i1, align 1
  store i1 false, ptr %bvar, align 1
  ret void
}

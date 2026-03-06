; RUN: llc -verify-machineinstrs -O0 -mtriple=spirv64-unknown-unknown %s -o - | FileCheck %s
; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv64-unknown-unknown %s -o - -filetype=obj | spirv-val %}

; Check that hidden visibility does not cause a crash and that hidden
; declarations get Import linkage while hidden definitions do not.

; CHECK-DAG: OpName %[[#HIDDEN_HELPER:]] "hidden_helper"
; CHECK-DAG: OpName %[[#HIDDEN_DEF:]] "hidden_def"
; CHECK-DAG: OpName %[[#HIDDEN_VAR:]] "hidden_leaf_var"
; CHECK-DAG: OpName %[[#KERN:]] "test_kernel"

; CHECK-DAG: OpDecorate %[[#HIDDEN_HELPER]] LinkageAttributes "hidden_helper" Import
; CHECK-DAG: OpDecorate %[[#HIDDEN_VAR]] LinkageAttributes "hidden_leaf_var" Import
; CHECK-NOT: OpDecorate %[[#HIDDEN_DEF]] LinkageAttributes

@hidden_leaf_var = external hidden addrspace(1) global i32

declare hidden spir_func void @hidden_helper(ptr addrspace(1))

define hidden spir_func void @hidden_def(ptr addrspace(1) %x) {
entry:
  ret void
}

define spir_kernel void @test_kernel(ptr addrspace(1) %data) {
entry:
  %val = load i32, ptr addrspace(1) @hidden_leaf_var
  store i32 %val, ptr addrspace(1) %data
  call spir_func void @hidden_helper(ptr addrspace(1) %data)
  call spir_func void @hidden_def(ptr addrspace(1) %data)
  ret void
}

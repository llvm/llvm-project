; RUN: llc -verify-machineinstrs -O0 -mtriple=spirv64-unknown-unknown %s -o - | FileCheck %s
; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv64-unknown-unknown %s -o - -filetype=obj | spirv-val %}

; Check that protected visibility does not cause a crash, that protected
; declarations get Import linkage, and protected definitions get Export.

; CHECK-DAG: OpName %[[#PROTECTED_DECL:]] "protected_decl"
; CHECK-DAG: OpName %[[#PROTECTED_DEF:]] "protected_def"
; CHECK-DAG: OpName %[[#KERN:]] "test_kernel"

; CHECK-DAG: OpDecorate %[[#PROTECTED_DECL]] LinkageAttributes "protected_decl" Import
; CHECK-DAG: OpDecorate %[[#PROTECTED_DEF]] LinkageAttributes "protected_def" Export

declare protected spir_func void @protected_decl(ptr addrspace(1))

define protected spir_func void @protected_def(ptr addrspace(1) %x) {
entry:
  ret void
}

define spir_kernel void @test_kernel(ptr addrspace(1) %data) {
entry:
  call spir_func void @protected_decl(ptr addrspace(1) %data)
  call spir_func void @protected_def(ptr addrspace(1) %data)
  ret void
}

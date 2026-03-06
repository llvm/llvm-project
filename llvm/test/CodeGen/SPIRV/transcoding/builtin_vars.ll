; RUN: llc -verify-machineinstrs -O0 -mtriple=spirv32-unknown-unknown %s -o - | FileCheck %s
; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv32-unknown-unknown %s -o - -filetype=obj | spirv-val %}

; RUN: llc -verify-machineinstrs -O0 -mtriple=spirv64-unknown-unknown %s -o - | FileCheck %s
; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv64-unknown-unknown %s -o - -filetype=obj | spirv-val %}

; CHECK: OpDecorate %[[#Id:]] BuiltIn GlobalLinearId
; CHECK: %[[#Id:]] = OpVariable %[[#]]

@__spirv_BuiltInGlobalLinearId = external addrspace(1) global i32

define spir_kernel void @f(ptr addrspace(1) nocapture %order) {
entry:
  %0 = load i32, ptr addrspace(4) addrspacecast (ptr addrspace(1) @__spirv_BuiltInGlobalLinearId to ptr addrspace(4)), align 4
  ;; Need to store the result somewhere, otherwise the access to GlobalLinearId
  ;; may be removed.
  store i32 %0, ptr addrspace(1) %order, align 4
  ret void
}

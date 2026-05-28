; RUN: llc -O0 -verify-machineinstrs -mtriple=spirv64-unknown-unknown --spirv-ext=+SPV_KHR_untyped_pointers %s -o - | FileCheck %s
; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv64-unknown-unknown --spirv-ext=+SPV_KHR_untyped_pointers %s -o - -filetype=obj | spirv-val %}

; memcpy lowers to OpCopyMemorySized with untyped pointer operands.

; CHECK-DAG: %[[#CROSS:]] = OpTypeUntypedPointerKHR CrossWorkgroup
; CHECK: OpCopyMemorySized %[[#]] %[[#]] %[[#]]
define spir_kernel void @t(ptr addrspace(1) %dst, ptr addrspace(1) %src) {
  call void @llvm.memcpy.p1.p1.i64(ptr addrspace(1) %dst, ptr addrspace(1) %src, i64 128, i1 false)
  ret void
}
declare void @llvm.memcpy.p1.p1.i64(ptr addrspace(1), ptr addrspace(1), i64, i1)

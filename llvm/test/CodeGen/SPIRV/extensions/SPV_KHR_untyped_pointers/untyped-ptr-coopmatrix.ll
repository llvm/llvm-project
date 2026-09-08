; RUN: llc -O0 -verify-machineinstrs -mtriple=spirv64-unknown-unknown --spirv-ext=+SPV_KHR_cooperative_matrix,+SPV_KHR_untyped_pointers %s -o - | FileCheck %s
; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv64-unknown-unknown --spirv-ext=+SPV_KHR_cooperative_matrix,+SPV_KHR_untyped_pointers %s -o - -filetype=obj | spirv-val %}

; Cooperative matrix load/store accept an untyped pointer operand unchanged.

; CHECK-DAG: %[[#CROSS:]] = OpTypeUntypedPointerKHR CrossWorkgroup
; CHECK: OpCooperativeMatrixLoadKHR
; CHECK: OpCooperativeMatrixStoreKHR
define spir_kernel void @t(ptr addrspace(1) %src, ptr addrspace(1) %dst, i64 %stride) {
  %m = tail call spir_func target("spirv.CooperativeMatrixKHR", i32, 3, 12, 12, 2) @_Z32__spirv_CooperativeMatrixLoadKHR(ptr addrspace(1) %src, i32 0, i64 %stride, i32 1)
  tail call spir_func void @_Z33__spirv_CooperativeMatrixStoreKHR(ptr addrspace(1) %dst, target("spirv.CooperativeMatrixKHR", i32, 3, 12, 12, 2) %m, i32 0, i64 %stride, i32 1)
  ret void
}
declare spir_func target("spirv.CooperativeMatrixKHR", i32, 3, 12, 12, 2) @_Z32__spirv_CooperativeMatrixLoadKHR(ptr addrspace(1), i32, i64, i32)
declare spir_func void @_Z33__spirv_CooperativeMatrixStoreKHR(ptr addrspace(1), target("spirv.CooperativeMatrixKHR", i32, 3, 12, 12, 2), i32, i64, i32)

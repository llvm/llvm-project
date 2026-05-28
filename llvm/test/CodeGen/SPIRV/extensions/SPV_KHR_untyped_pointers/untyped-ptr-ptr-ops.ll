; RUN: llc -O0 -verify-machineinstrs -mtriple=spirv64-unknown-unknown --spirv-ext=+SPV_KHR_untyped_pointers %s -o - | FileCheck %s
; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv64-unknown-unknown --spirv-ext=+SPV_KHR_untyped_pointers %s -o - -filetype=obj | spirv-val --target-env spv1.4 %}

; icmp on untyped pointers uses OpPtrEqual/OpPtrNotEqual, and ptrtoint/inttoptr
; use the pointer conversion ops.

; CHECK-DAG: %[[#PTR:]] = OpTypeUntypedPointerKHR CrossWorkgroup
; CHECK-DAG: %[[#I64:]] = OpTypeInt 64 0

; CHECK: OpPtrEqual %[[#]] %[[#]] %[[#]]
define spir_func i1 @ptr_eq(ptr addrspace(1) %a, ptr addrspace(1) %b) {
  %c = icmp eq ptr addrspace(1) %a, %b
  ret i1 %c
}

; CHECK: OpPtrNotEqual %[[#]] %[[#]] %[[#]]
define spir_func i1 @ptr_ne(ptr addrspace(1) %a, ptr addrspace(1) %b) {
  %c = icmp ne ptr addrspace(1) %a, %b
  ret i1 %c
}

; CHECK: OpConvertPtrToU %[[#I64]] %[[#]]
define spir_func i64 @p2i(ptr addrspace(1) %a) {
  %i = ptrtoint ptr addrspace(1) %a to i64
  ret i64 %i
}

; CHECK: OpConvertUToPtr %[[#PTR]] %[[#]]
define spir_func ptr addrspace(1) @i2p(i64 %i) {
  %p = inttoptr i64 %i to ptr addrspace(1)
  ret ptr addrspace(1) %p
}

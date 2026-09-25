; RUN: llc -O0 -verify-machineinstrs -mtriple=spirv64-unknown-unknown --spirv-ext=+SPV_KHR_untyped_pointers %s -o - | FileCheck %s
; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv64-unknown-unknown --spirv-ext=+SPV_KHR_untyped_pointers %s -o - -filetype=obj | spirv-val %}

; prefetch on an untyped pointer uses OpUntypedPrefetchKHR with Num Bytes equal
; to num_elements scaled by the element byte size.

; CHECK-DAG: OpCapability UntypedPointersKHR
; CHECK-DAG: %[[#I32:]] = OpTypeInt 32 0
; CHECK-DAG: %[[#FOUR:]] = OpConstant %[[#I32]] 4

; i32 element, so Num Bytes is num_elements * 4.
; CHECK: %[[#NB:]] = OpIMul %[[#I32]] %[[#]] %[[#FOUR]]
; CHECK: OpUntypedPrefetchKHR %[[#]] %[[#NB]]
define spir_kernel void @pf_i32(ptr addrspace(1) %p, i32 %n) {
  call spir_func void @_Z8prefetchPU3AS1Kij(ptr addrspace(1) %p, i32 %n)
  ret void
}

; i8 element, so Num Bytes is the count with no multiply.
; CHECK: OpUntypedPrefetchKHR
; CHECK-NOT: OpIMul
define spir_kernel void @pf_i8(ptr addrspace(1) %p) {
  call spir_func void @_Z20__spirv_ocl_prefetchPU3AS1Kcm(ptr addrspace(1) %p, i64 4)
  ret void
}

declare spir_func void @_Z8prefetchPU3AS1Kij(ptr addrspace(1), i32)
declare spir_func void @_Z20__spirv_ocl_prefetchPU3AS1Kcm(ptr addrspace(1), i64)

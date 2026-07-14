;; OpenCL C source:
;; #pragma OPENCL EXTENSION cl_khr_int64_base_atomics : enable
;; #pragma OPENCL EXTENSION cl_khr_int64_extended_atomics : enable
;;
;; void foo (volatile atomic_long *object, long *expected, long desired) {
;;   atomic_compare_exchange_weak(object, expected, desired);
;; }

; RUN: llc -O0 -mtriple=spirv64-unknown-unknown %s -o - | FileCheck %s
; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv64v1.2-unknown-unknown %s -o - -filetype=obj | spirv-val %}

; CHECK: OpCapability Int64Atomics

define spir_func void @foo(ptr addrspace(4) %object, ptr addrspace(4) %expected, i64 %desired) {
entry:
  %call = tail call spir_func zeroext i1 @_Z28atomic_compare_exchange_weakPVU3AS4U7_AtomiclPU3AS4ll(ptr addrspace(4) %object, ptr addrspace(4) %expected, i64 %desired)
  ret void
}

declare spir_func zeroext i1 @_Z28atomic_compare_exchange_weakPVU3AS4U7_AtomiclPU3AS4ll(ptr addrspace(4), ptr addrspace(4), i64)

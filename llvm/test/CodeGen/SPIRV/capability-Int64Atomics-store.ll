;; OpenCL C source:
;; #pragma OPENCL EXTENSION cl_khr_int64_base_atomics : enable
;; #pragma OPENCL EXTENSION cl_khr_int64_extended_atomics : enable
;;
;; void foo (volatile atomic_long *object, long desired) {
;;   atomic_store(object, desired);
;; }

; RUN: llc -O0 -mtriple=spirv64-unknown-unknown %s -o - | FileCheck %s

; CHECK: OpCapability Int64Atomics

define spir_func void @foo(i64 addrspace(4)* %object, i64 %desired) {
entry:
  tail call spir_func void @_Z12atomic_storePVU3AS4U7_Atomicll(i64 addrspace(4)* %object, i64 %desired)
  ret void
}

declare spir_func void @_Z12atomic_storePVU3AS4U7_Atomicll(i64 addrspace(4)*, i64)

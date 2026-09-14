; RUN: llc -O0 -verify-machineinstrs -mtriple=spirv64-unknown-unknown --spirv-ext=+SPV_KHR_untyped_pointers %s -o - | FileCheck %s
; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv64-unknown-unknown --spirv-ext=+SPV_KHR_untyped_pointers %s -o - -filetype=obj | spirv-val %}

; A GEP based on a module-scope global carries the global's value type as the
; access-chain Base Type, and cmpxchg through an untyped pointer selects
; OpAtomicCompareExchange.

; CHECK-DAG: %[[#I32:]] = OpTypeInt 32 0
; CHECK-DAG: %[[#ARR:]] = OpTypeArray %[[#I32]]
; CHECK-DAG: %[[#CROSS:]] = OpTypeUntypedPointerKHR CrossWorkgroup

; The global's Data Type and the access chain's Base Type are both the array.
; CHECK: %[[#G:]] = OpUntypedVariableKHR %[[#CROSS]] CrossWorkgroup %[[#ARR]]
; CHECK: OpUntypedInBoundsPtrAccessChainKHR %[[#CROSS]] %[[#ARR]] %[[#G]]
@arr = addrspace(1) global [4 x i32] zeroinitializer, align 4
define spir_kernel void @gep_global(ptr addrspace(1) %out) {
  %p = getelementptr inbounds [4 x i32], ptr addrspace(1) @arr, i64 0, i64 2
  %v = load i32, ptr addrspace(1) %p, align 4
  store i32 %v, ptr addrspace(1) %out, align 4
  ret void
}

; CHECK: OpAtomicCompareExchange %[[#I32]]
define spir_func i32 @cmpxchg_fn(ptr addrspace(1) %p, i32 %cmp, i32 %new) {
  %pair = cmpxchg ptr addrspace(1) %p, i32 %cmp, i32 %new seq_cst seq_cst
  %old = extractvalue { i32, i1 } %pair, 0
  ret i32 %old
}

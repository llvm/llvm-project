; RUN: llc -O0 -mtriple=spirv64-unknown-unknown %s -o - | FileCheck %s
; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv64-unknown-unknown %s -o - -filetype=obj | spirv-val %}

;; Types:
; CHECK:         %[[#F32:]] = OpTypeFloat 32
; CHECK:         %[[#I32:]] = OpTypeInt 32 0
;; Constants:
; CHECK-DAG:     %[[#CONST:]] = OpConstant %[[#F32]] 1
; CHECK-DAG:     %[[#RELAXED:]] = OpConstantNull %[[#I32]]
; CHECK-DAG:     %[[#DEVICE:]] = OpConstant %[[#I32]] 1
; CHECK-DAG:     %[[#WORKGROUP:]] = OpConstant %[[#I32]] 2
; CHECK-DAG:     %[[#SEQCST:]] = OpConstant %[[#I32]] 16
;; Atomic instructions:
; CHECK:         OpStore %[[#]] %[[#CONST]]
; CHECK:         OpAtomicStore %[[#]] %[[#DEVICE]] %[[#SEQCST]] %[[#CONST]]
; CHECK:         OpAtomicStore %[[#]] %[[#DEVICE]] %[[#RELAXED]] %[[#CONST]]
; CHECK:         OpAtomicStore %[[#]] %[[#WORKGROUP]] %[[#RELAXED]] %[[#CONST]]
; CHECK:         OpAtomicLoad %[[#]] %[[#]] %[[#DEVICE]] %[[#SEQCST]]
; CHECK:         OpAtomicLoad %[[#]] %[[#]] %[[#DEVICE]] %[[#RELAXED]]
; CHECK:         OpAtomicLoad %[[#]] %[[#]] %[[#WORKGROUP]] %[[#RELAXED]]
; CHECK-COUNT-3: OpAtomicExchange

define spir_kernel void @test_atomic_kernel(ptr addrspace(3) %ff) local_unnamed_addr #0 !kernel_arg_addr_space !3 !kernel_arg_access_qual !4 !kernel_arg_type !5 !kernel_arg_base_type !6 !kernel_arg_type_qual !7 {
entry:
  %0 = addrspacecast ptr addrspace(3) %ff to ptr addrspace(4)
  tail call spir_func void @_Z11atomic_initPU3AS4VU7_Atomicff(ptr addrspace(4) %0, float 1.000000e+00) #2
  tail call spir_func void @_Z12atomic_storePU3AS4VU7_Atomicff(ptr addrspace(4) %0, float 1.000000e+00) #2
  tail call spir_func void @_Z21atomic_store_explicitPU3AS4VU7_Atomicff12memory_order(ptr addrspace(4) %0, float 1.000000e+00, i32 0) #2
  tail call spir_func void @_Z21atomic_store_explicitPU3AS4VU7_Atomicff12memory_order12memory_scope(ptr addrspace(4) %0, float 1.000000e+00, i32 0, i32 1) #2
  %call = tail call spir_func float @_Z11atomic_loadPU3AS4VU7_Atomicf(ptr addrspace(4) %0) #2
  %call1 = tail call spir_func float @_Z20atomic_load_explicitPU3AS4VU7_Atomicf12memory_order(ptr addrspace(4) %0, i32 0) #2
  %call2 = tail call spir_func float @_Z20atomic_load_explicitPU3AS4VU7_Atomicf12memory_order12memory_scope(ptr addrspace(4) %0, i32 0, i32 1) #2
  %call3 = tail call spir_func float @_Z15atomic_exchangePU3AS4VU7_Atomicff(ptr addrspace(4) %0, float 1.000000e+00) #2
  %call4 = tail call spir_func float @_Z24atomic_exchange_explicitPU3AS4VU7_Atomicff12memory_order(ptr addrspace(4) %0, float 1.000000e+00, i32 0) #2
  %call5 = tail call spir_func float @_Z24atomic_exchange_explicitPU3AS4VU7_Atomicff12memory_order12memory_scope(ptr addrspace(4) %0, float 1.000000e+00, i32 0, i32 1) #2
  ret void
}

declare spir_func void @_Z11atomic_initPU3AS4VU7_Atomicff(ptr addrspace(4), float)

declare spir_func void @_Z12atomic_storePU3AS4VU7_Atomicff(ptr addrspace(4), float)

declare spir_func void @_Z21atomic_store_explicitPU3AS4VU7_Atomicff12memory_order(ptr addrspace(4), float, i32)

declare spir_func void @_Z21atomic_store_explicitPU3AS4VU7_Atomicff12memory_order12memory_scope(ptr addrspace(4), float, i32, i32)

declare spir_func float @_Z11atomic_loadPU3AS4VU7_Atomicf(ptr addrspace(4))

declare spir_func float @_Z20atomic_load_explicitPU3AS4VU7_Atomicf12memory_order(ptr addrspace(4), i32)

declare spir_func float @_Z20atomic_load_explicitPU3AS4VU7_Atomicf12memory_order12memory_scope(ptr addrspace(4), i32, i32)

declare spir_func float @_Z15atomic_exchangePU3AS4VU7_Atomicff(ptr addrspace(4), float)

declare spir_func float @_Z24atomic_exchange_explicitPU3AS4VU7_Atomicff12memory_order(ptr addrspace(4), float, i32)

declare spir_func float @_Z24atomic_exchange_explicitPU3AS4VU7_Atomicff12memory_order12memory_scope(ptr addrspace(4), float, i32, i32)

!3 = !{i32 3}
!4 = !{!"none"}
!5 = !{!"atomic_float*"}
!6 = !{!"_Atomic(float)*"}
!7 = !{!"volatile"}

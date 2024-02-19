; RUN: llc -O0 -mtriple=spirv64-unknown-unknown %s -o - | FileCheck %s

;; Types:
; CHECK:         %[[#F32:]] = OpTypeFloat 32
;; Constants:
; CHECK:         %[[#CONST:]] = OpConstant %[[#F32]] 1
;; Atomic instructions:
; CHECK:         OpStore %[[#]] %[[#CONST]]
; CHECK-COUNT-3: OpAtomicStore
; CHECK-COUNT-3: OpAtomicLoad
; CHECK-COUNT-3: OpAtomicExchange

define spir_kernel void @test_atomic_kernel(float addrspace(3)* %ff) local_unnamed_addr #0 !kernel_arg_addr_space !3 !kernel_arg_access_qual !4 !kernel_arg_type !5 !kernel_arg_base_type !6 !kernel_arg_type_qual !7 {
entry:
  %0 = addrspacecast float addrspace(3)* %ff to float addrspace(4)*
  tail call spir_func void @_Z11atomic_initPU3AS4VU7_Atomicff(float addrspace(4)* %0, float 1.000000e+00) #2
  tail call spir_func void @_Z12atomic_storePU3AS4VU7_Atomicff(float addrspace(4)* %0, float 1.000000e+00) #2
  tail call spir_func void @_Z21atomic_store_explicitPU3AS4VU7_Atomicff12memory_order(float addrspace(4)* %0, float 1.000000e+00, i32 0) #2
  tail call spir_func void @_Z21atomic_store_explicitPU3AS4VU7_Atomicff12memory_order12memory_scope(float addrspace(4)* %0, float 1.000000e+00, i32 0, i32 1) #2
  %call = tail call spir_func float @_Z11atomic_loadPU3AS4VU7_Atomicf(float addrspace(4)* %0) #2
  %call1 = tail call spir_func float @_Z20atomic_load_explicitPU3AS4VU7_Atomicf12memory_order(float addrspace(4)* %0, i32 0) #2
  %call2 = tail call spir_func float @_Z20atomic_load_explicitPU3AS4VU7_Atomicf12memory_order12memory_scope(float addrspace(4)* %0, i32 0, i32 1) #2
  %call3 = tail call spir_func float @_Z15atomic_exchangePU3AS4VU7_Atomicff(float addrspace(4)* %0, float 1.000000e+00) #2
  %call4 = tail call spir_func float @_Z24atomic_exchange_explicitPU3AS4VU7_Atomicff12memory_order(float addrspace(4)* %0, float 1.000000e+00, i32 0) #2
  %call5 = tail call spir_func float @_Z24atomic_exchange_explicitPU3AS4VU7_Atomicff12memory_order12memory_scope(float addrspace(4)* %0, float 1.000000e+00, i32 0, i32 1) #2
  ret void
}

declare spir_func void @_Z11atomic_initPU3AS4VU7_Atomicff(float addrspace(4)*, float)

declare spir_func void @_Z12atomic_storePU3AS4VU7_Atomicff(float addrspace(4)*, float)

declare spir_func void @_Z21atomic_store_explicitPU3AS4VU7_Atomicff12memory_order(float addrspace(4)*, float, i32)

declare spir_func void @_Z21atomic_store_explicitPU3AS4VU7_Atomicff12memory_order12memory_scope(float addrspace(4)*, float, i32, i32)

declare spir_func float @_Z11atomic_loadPU3AS4VU7_Atomicf(float addrspace(4)*)

declare spir_func float @_Z20atomic_load_explicitPU3AS4VU7_Atomicf12memory_order(float addrspace(4)*, i32)

declare spir_func float @_Z20atomic_load_explicitPU3AS4VU7_Atomicf12memory_order12memory_scope(float addrspace(4)*, i32, i32)

declare spir_func float @_Z15atomic_exchangePU3AS4VU7_Atomicff(float addrspace(4)*, float)

declare spir_func float @_Z24atomic_exchange_explicitPU3AS4VU7_Atomicff12memory_order(float addrspace(4)*, float, i32)

declare spir_func float @_Z24atomic_exchange_explicitPU3AS4VU7_Atomicff12memory_order12memory_scope(float addrspace(4)*, float, i32, i32)

!3 = !{i32 3}
!4 = !{!"none"}
!5 = !{!"atomic_float*"}
!6 = !{!"_Atomic(float)*"}
!7 = !{!"volatile"}

; RUN: opt -mtriple=amdgcn-- -amdgpu-lower-opencl-atomic-builtins -mcpu=fiji -S < %s | FileCheck %s
; CHECK: %[[CMPXCHG1:[0-9]+]] = cmpxchg weak volatile i32 addrspace(4)* null, i32 %[[EXPECTED1:[0-9]+]], i32 0 syncscope(2) seq_cst seq_cst
; CHECK: %[[CMPXCHG1_RES_TY:[0-9]+]] = extractvalue { i32, i1 } %[[CMPXCHG1]], 0
; CHECK: %[[CMPXCHG1_RES_BOOL:[0-9]+]] = extractvalue { i32, i1 } %[[CMPXCHG1]], 1
; CHECK: %[[CMPXCHG1_FAIL_OR_NOT:[0-9]+]] = select i1 %[[CMPXCHG1_RES_BOOL]], i32 %[[EXPECTED1]], i32 %[[CMPXCHG1_RES_TY]]
; CHECK: store i32 %[[CMPXCHG1_FAIL_OR_NOT]], i32 addrspace(4)* null

; CHECK: %[[CMPXCHG2:[0-9]+]] = cmpxchg volatile i32 addrspace(4)* null, i32 %[[EXPECTED2:[0-9]+]], i32 0 syncscope(4) release acquire
; CHECK: %[[CMPXCHG2_RES_TY:[0-9]+]] = extractvalue { i32, i1 } %[[CMPXCHG2]], 0
; CHECK: %[[CMPXCHG2_RES_BOOL:[0-9]+]] = extractvalue { i32, i1 } %[[CMPXCHG2]], 1
; CHECK: %[[CMPXCHG2_FAIL_OR_NOT:[0-9]+]] = select i1 %[[CMPXCHG2_RES_BOOL]], i32 %[[EXPECTED2]], i32 %[[CMPXCHG2_RES_TY]]
; CHECK: store i32 %[[CMPXCHG2_FAIL_OR_NOT]], i32 addrspace(4)* null

; CHECK: atomicrmw add i32 addrspace(4)* null, i32 1 syncscope(3) release
; CHECK: store volatile i32 42, i32 addrspace(4)* addrspacecast (i32 addrspace(3)* @test.guide to i32 addrspace(4)*)
; CHECK: load atomic volatile i32, i32 addrspace(4)* null syncscope(2) monotonic, align 4
; CHECK: store atomic volatile i32 0, i32 addrspace(4)* null syncscope(2) release, align 4
; CHECK: call void @my_atomic_compare_exchange_strong_explicit(i32 addrspace(4)* null, i32 addrspace(4)* null, i32 0)
; CHECK: store atomic volatile i32 0, i32 addrspace(4)* null syncscope(2) seq_cst, align 4

@test.guide = internal addrspace(3) global i32 undef, align 4

define amdgpu_kernel void @test() {
entry:

  %call = call zeroext i1 @_Z30atomic_compare_exchange_weakPU3AS4VU7_AtomiciPU3AS4ii(i32 addrspace(4)* null, i32 addrspace(4)* null, i32 0)
  %call1 = call zeroext i1 @_Z39atomic_compare_exchange_strong_explicitPU3AS4VU7_AtomiciPU3AS4ii12memory_orderS4_12memory_scope(i32 addrspace(4)* null, i32 addrspace(4)* null, i32 0, i32 2, i32 1, i32 4)
  %call2 = call i32 @_Z25atomic_fetch_add_explicitPU3AS4VU7_Atomicii12memory_order12memory_scope(i32 addrspace(4)* null, i32 1, i32 2, i32 1)
           call void @_Z11atomic_initPU3AS4VU7_Atomicii(i32 addrspace(4)* addrspacecast (i32 addrspace(3)* @test.guide to i32 addrspace(4)*), i32 42)
  %call3 = call i32 @_Z20atomic_load_explicitPU3AS4VU7_Atomici12memory_order(i32 addrspace(4)* null, i32 0)
           call void @_Z21atomic_store_explicitPU3AS4VU7_Atomicii12memory_order(i32 addrspace(4)* null, i32 0, i32 2)
           call void @my_atomic_compare_exchange_strong_explicit(i32 addrspace(4)* null, i32 addrspace(4)* null, i32 0)
           call void @_Z17atomic_flag_clearPU3AS4VU7_Atomici(i32 addrspace(4)* null)
  ret void
}
declare i32 @_Z20atomic_load_explicitPU3AS4VU7_Atomici12memory_order(i32 addrspace(4)*, i32) #2
declare void @_Z21atomic_store_explicitPU3AS4VU7_Atomicii12memory_order(i32 addrspace(4)*, i32, i32) #2
declare void @_Z11atomic_initPU3AS4VU7_Atomicii(i32 addrspace(4)*, i32)
declare i32 @_Z25atomic_fetch_add_explicitPU3AS4VU7_Atomicii12memory_order12memory_scope(i32 addrspace(4)*, i32, i32, i32) #1
declare zeroext i1 @_Z30atomic_compare_exchange_weakPU3AS4VU7_AtomiciPU3AS4ii(i32 addrspace(4)*, i32 addrspace(4)*, i32) #2
declare zeroext i1 @_Z39atomic_compare_exchange_strong_explicitPU3AS4VU7_AtomiciPU3AS4ii12memory_orderS4_12memory_scope(i32 addrspace(4)*, i32 addrspace(4)*, i32, i32, i32, i32) #2
declare void @my_atomic_compare_exchange_strong_explicit(i32 addrspace(4)*, i32 addrspace(4)*, i32)
declare void @_Z17atomic_flag_clearPU3AS4VU7_Atomici(i32 addrspace(4)*)

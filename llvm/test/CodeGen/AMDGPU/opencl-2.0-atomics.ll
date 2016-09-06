; RUN: opt -mtriple=amdgcn--amdhsa -amdgpu-lower-opencl-atomic-builtins -mcpu=fiji -S < %s | FileCheck %s
; CHECK: cmpxchg i32 addrspace(4)* null, i32 %{{[0-9]+}}, i32 0 synchscope(2) seq_cst seq_cst
; CHECK: cmpxchg i32 addrspace(4)* null, i32 %{{[0-9]+}}, i32 0 synchscope(4) release acquire
; CHECK: atomicrmw add i32 addrspace(4)* null, i32 1 synchscope(3) release
; CHECK: store volatile i32 42, i32 addrspace(4)* addrspacecast (i32 addrspace(3)* @test.guide to i32 addrspace(4)*)
; CHECK: load atomic volatile i32, i32 addrspace(4)* null synchscope(2) monotonic, align 4
; CHECK: store atomic volatile i32 0, i32 addrspace(4)* null synchscope(2) release, align 4
; CHECK: call void @my_atomic_compare_exchange_strong_explicit(

@test.guide = internal addrspace(3) global i32 undef, align 4

define amdgpu_kernel void @test() {
entry:

  %call = call zeroext i1 @_Z30atomic_compare_exchange_strongPU3AS4VU7_AtomiciPU3AS4ii(i32 addrspace(4)* null, i32 addrspace(4)* null, i32 0)
  %call1 = call zeroext i1 @_Z39atomic_compare_exchange_strong_explicitPU3AS4VU7_AtomiciPU3AS4ii12memory_orderS4_12memory_scope(i32 addrspace(4)* null, i32 addrspace(4)* null, i32 0, i32 2, i32 1, i32 4)
  %call2 = call i32 @_Z25atomic_fetch_add_explicitPU3AS4VU7_Atomicii12memory_order12memory_scope(i32 addrspace(4)* null, i32 1, i32 2, i32 1)
           call void @_Z11atomic_initPU3AS4VU7_Atomicii(i32 addrspace(4)* addrspacecast (i32 addrspace(3)* @test.guide to i32 addrspace(4)*), i32 42)
  %call3 = call i32 @_Z20atomic_load_explicitPU3AS4VU7_Atomici12memory_order(i32 addrspace(4)* null, i32 0)
           call void @_Z21atomic_store_explicitPU3AS4VU7_Atomicii12memory_order(i32 addrspace(4)* null, i32 0, i32 2)
           call void @my_atomic_compare_exchange_strong_explicit(i32 addrspace(4)* null, i32 addrspace(4)* null, i32 0)
  ret void
}
declare i32 @_Z20atomic_load_explicitPU3AS4VU7_Atomici12memory_order(i32 addrspace(4)*, i32) #2
declare void @_Z21atomic_store_explicitPU3AS4VU7_Atomicii12memory_order(i32 addrspace(4)*, i32, i32) #2
declare void @_Z11atomic_initPU3AS4VU7_Atomicii(i32 addrspace(4)*, i32)
declare i32 @_Z25atomic_fetch_add_explicitPU3AS4VU7_Atomicii12memory_order12memory_scope(i32 addrspace(4)*, i32, i32, i32) #1
declare zeroext i1 @_Z30atomic_compare_exchange_strongPU3AS4VU7_AtomiciPU3AS4ii(i32 addrspace(4)*, i32 addrspace(4)*, i32) #2
declare zeroext i1 @_Z39atomic_compare_exchange_strong_explicitPU3AS4VU7_AtomiciPU3AS4ii12memory_orderS4_12memory_scope(i32 addrspace(4)*, i32 addrspace(4)*, i32, i32, i32, i32) #2
declare void @my_atomic_compare_exchange_strong_explicit(i32 addrspace(4)*, i32 addrspace(4)*, i32)

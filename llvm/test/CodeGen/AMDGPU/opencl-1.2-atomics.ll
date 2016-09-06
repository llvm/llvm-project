; RUN:   opt -mtriple=amdgcn--amdhsa -amdgpu-lower-opencl-atomic-builtins -mcpu=fiji -S < %s | FileCheck %s
; CHECK: atomicrmw add i32 addrspace(1)* null, i32 1 synchscope(2) monotonic
; CHECK: atomicrmw add i32 addrspace(1)* null, i32 0 synchscope(2) monotonic
; CHECK: cmpxchg i32 addrspace(1)* null, i32 0, i32 0 synchscope(2) monotonic monotonic
; CHECK: atomicrmw xchg i32 addrspace(1)* null, i32 0 synchscope(2) monotonic
define amdgpu_kernel void @test() {
entry:
  %call0 = call i32 @_Z8atom_incPU3AS1Vi(i32 addrspace(1)* null)
  %call1 = call i32 @_Z10atomic_addPU3AS1Vii(i32 addrspace(1)* null, i32 0)
  %call2 = call i32 @_Z14atomic_cmpxchgPU3AS1Viii(i32 addrspace(1)* null, i32 0, i32 0)
  %call3 = call i32 @_Z11atomic_xchgPU3AS1Vii(i32 addrspace(1)* null, i32 0)
  ret void
}
declare i32 @_Z8atom_incPU3AS1Vi(i32 addrspace(1)*)
declare i32 @_Z10atomic_addPU3AS1Vii(i32 addrspace(1)*, i32)
declare i32 @_Z14atomic_cmpxchgPU3AS1Viii(i32 addrspace(1)*, i32, i32)
declare i32 @_Z11atomic_xchgPU3AS1Vii(i32 addrspace(1)*, i32)

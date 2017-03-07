; RUN:   opt -mtriple=amdgcn--amdhsa -amdgpu-lower-opencl-atomic-builtins -mcpu=fiji -S < %s | FileCheck %s
; CHECK: atomicrmw add i32 addrspace(1)* null, i32 1 syncscope(2) monotonic
; CHECK: atomicrmw add i32 addrspace(1)* null, i32 0 syncscope(2) monotonic
; CHECK: cmpxchg i32 addrspace(1)* null, i32 0, i32 0 syncscope(2) monotonic monotonic
; CHECK: atomicrmw xchg i32 addrspace(1)* null, i32 0 syncscope(2) monotonic
; CHECK: [[A:%[0-9]*]] = atomicrmw xchg i32 addrspace(3)* null, i32 0 syncscope(2) monotonic
; CHECK: bitcast i32 [[A]] to float
; CHECK: call void @_ZN9__gnu_cxxL21__atomic_add_dispatchEPii(i32 0, i32 1)
; CHECK: declare void @_ZN9__gnu_cxxL21__atomic_add_dispatchEPii(i32, i32)

define amdgpu_kernel void @test() {
entry:
  %call0 = call i32 @_Z8atom_incPU3AS1Vi(i32 addrspace(1)* null)
  %call1 = call i32 @_Z10atomic_addPU3AS1Vii(i32 addrspace(1)* null, i32 0)
  %call2 = call i32 @_Z14atomic_cmpxchgPU3AS1Viii(i32 addrspace(1)* null, i32 0, i32 0)
  %call3 = call i32 @_Z11atomic_xchgPU3AS1Vii(i32 addrspace(1)* null, i32 0)
  %call4 = call float @_Z11atomic_xchgPU3AS3Vff(float addrspace(3)* null, float 0.0)
  call void @_ZN9__gnu_cxxL21__atomic_add_dispatchEPii(i32 0, i32 1)
  ret void
}
declare i32 @_Z8atom_incPU3AS1Vi(i32 addrspace(1)*)
declare i32 @_Z10atomic_addPU3AS1Vii(i32 addrspace(1)*, i32)
declare i32 @_Z14atomic_cmpxchgPU3AS1Viii(i32 addrspace(1)*, i32, i32)
declare i32 @_Z11atomic_xchgPU3AS1Vii(i32 addrspace(1)*, i32)
declare float @_Z11atomic_xchgPU3AS3Vff(float addrspace(3)*, float)
declare void @_ZN9__gnu_cxxL21__atomic_add_dispatchEPii(i32, i32)

; RUN: opt -mtriple=amdgcn--amdhsa -amdgpu-opencl-12-adapter -mcpu=fiji -S < %s | FileCheck %s
; CHECK-LABEL: define extern_weak <2 x i32> @_Z6vload2mPU3AS1Ki(i64, i32 addrspace(1)*)
; CHECK: %2 = addrspacecast i32 addrspace(1)* %1 to i32 addrspace(4)*
; CHECK: %3 = call <2 x i32> @_Z6vload2mPU3AS4Ki(i64 %0, i32 addrspace(4)* %2)
; CHECK: ret <2 x i32> %3

define amdgpu_kernel void @test_fn() {
entry:

call <2 x i32> @_Z6vload2mPU3AS1Ki(i64 0, i32 addrspace(1)* null)
ret void
}

declare <2 x i32> @_Z6vload2mPU3AS1Ki(i64, i32 addrspace(1)*)

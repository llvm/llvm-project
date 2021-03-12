; RUN: opt -mtriple=amdgcn--amdhsa -S -passes=inline -inline-threshold=0 -debug-only=inline-cost < %s 2>&1 | FileCheck %s

; REQUIRES: asserts

target datalayout = "A5"

; Verify we are properly adding cost of the -amdgpu-inline-arg-alloca-cost to the threshold.

; CHECK: NumAllocaArgs: 1
; CHECK: Threshold: 66000

define void @use_private_ptr_arg(float addrspace(5)* nocapture %p) {
  ret void
}

define amdgpu_kernel void @test_inliner_pvt_ptr(float addrspace(1)* nocapture %a, i32 %n) {
entry:
  %pvt_arr = alloca [64 x float], align 4, addrspace(5)
  %to.ptr = getelementptr inbounds [64 x float], [64 x float] addrspace(5)* %pvt_arr, i32 0, i32 0
  call void @use_private_ptr_arg(float addrspace(5)* %to.ptr)
  ret void
}

; RUN: opt -data-layout=A5 -S -mtriple=amdgcn-unknown-unknown -passes=amdgpu-promote-alloca -disable-promote-alloca-to-vector < %s | FileCheck %s

; Make sure that AMDGPUPromoteAlloca doesn't crash if the called
; function is a constantexpr cast of a function.

declare void @foo(ptr addrspace(5)) nounwind
declare void @foo.varargs(...) nounwind

; CHECK-LABEL: @crash_call_constexpr_cast(
; CHECK: alloca
; CHECK: call void
define amdgpu_kernel void @crash_call_constexpr_cast() nounwind {
  %alloca = alloca i32, addrspace(5)
  call void @foo(ptr addrspace(5) %alloca) nounwind
  ret void
}

; CHECK-LABEL: @crash_call_constexpr_cast_varargs(
; CHECK: alloca
; CHECK: call void
define amdgpu_kernel void @crash_call_constexpr_cast_varargs() nounwind {
  %alloca = alloca i32, addrspace(5)
  call void @foo.varargs(ptr addrspace(5) %alloca) nounwind
  ret void
}

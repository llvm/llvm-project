; RUN: opt -S -mtriple=amdgcn-unknown-amdhsa -passes=amdgpu-promote-alloca < %s | FileCheck %s

; CHECK-LABEL: @volatile_load(
; CHECK: alloca [4 x i32]
; CHECK: load volatile i32, ptr addrspace(5)
define amdgpu_kernel void @volatile_load(ptr addrspace(1) nocapture %out, ptr addrspace(1) nocapture %in) {
entry:
  %stack = alloca [4 x i32], align 4, addrspace(5)
  %tmp = load i32, ptr addrspace(1) %in, align 4
  %arrayidx1 = getelementptr inbounds [4 x i32], ptr addrspace(5) %stack, i32 0, i32 %tmp
  %load = load volatile i32, ptr addrspace(5) %arrayidx1
  store i32 %load, ptr addrspace(1) %out
 ret void
}

; CHECK-LABEL: @volatile_store(
; CHECK: alloca [4 x i32]
; CHECK: store volatile i32 %tmp, ptr addrspace(5)
define amdgpu_kernel void @volatile_store(ptr addrspace(1) nocapture %out, ptr addrspace(1) nocapture %in) {
entry:
  %stack = alloca [4 x i32], align 4, addrspace(5)
  %tmp = load i32, ptr addrspace(1) %in, align 4
  %arrayidx1 = getelementptr inbounds [4 x i32], ptr addrspace(5) %stack, i32 0, i32 %tmp
  store volatile i32 %tmp, ptr addrspace(5) %arrayidx1
 ret void
}

; Has on OK non-volatile user but also a volatile user
; CHECK-LABEL: @volatile_and_non_volatile_load(
; CHECK: alloca double
; CHECK: load double
; CHECK: load volatile double
define amdgpu_kernel void @volatile_and_non_volatile_load(ptr addrspace(1) nocapture %arg, i32 %arg1) #0 {
bb:
  %tmp = alloca double, align 8, addrspace(5)
  store double 0.000000e+00, ptr addrspace(5) %tmp, align 8

  %tmp4 = load double, ptr addrspace(5) %tmp, align 8
  %tmp5 = load volatile double, ptr addrspace(5) %tmp, align 8

  store double %tmp4, ptr addrspace(1) %arg
  ret void
}

attributes #0 = { nounwind }

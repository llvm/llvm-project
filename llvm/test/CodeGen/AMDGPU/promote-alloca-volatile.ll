; RUN: opt -S -mtriple=amdgcn-unknown-amdhsa -amdgpu-promote-alloca < %s | FileCheck %s

; CHECK-LABEL: @volatile_load(
; CHECK: alloca [5 x i32]
; CHECK load volatile i32, i32*
define void @volatile_load(i32 addrspace(1)* nocapture %out, i32 addrspace(1)* nocapture %in) {
entry:
  %stack = alloca [5 x i32], align 4
  %tmp = load i32, i32 addrspace(1)* %in, align 4
  %arrayidx1 = getelementptr inbounds [5 x i32], [5 x i32]* %stack, i32 0, i32 %tmp
  %load = load volatile i32, i32* %arrayidx1
  store i32 %load, i32 addrspace(1)* %out
 ret void
}

; CHECK-LABEL: @volatile_store(
; CHECK: alloca [5 x i32]
; CHECK store volatile i32 %tmp, i32*
define void @volatile_store(i32 addrspace(1)* nocapture %out, i32 addrspace(1)* nocapture %in) {
entry:
  %stack = alloca [5 x i32], align 4
  %tmp = load i32, i32 addrspace(1)* %in, align 4
  %arrayidx1 = getelementptr inbounds [5 x i32], [5 x i32]* %stack, i32 0, i32 %tmp
  store volatile i32 %tmp, i32* %arrayidx1
 ret void
}

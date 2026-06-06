; RUN: opt -S -mtriple=amdgcn-unknown-amdhsa -mcpu=kaveri -debug-only=amdgpu-promote-alloca -amdgpu-promote-alloca-to-vector-limit=512 -passes=amdgpu-promote-alloca %s -o - 2>&1 | FileCheck %s
; REQUIRES: asserts

; CHECK-LABEL: Analyzing:   %simpleuser = alloca [4 x i64], align 4, addrspace(5)
; CHECK-NEXT: Scoring:   %simpleuser = alloca [4 x i64], align 4, addrspace(5)
; CHECK-NEXT:   [+1]:   store i64 42, ptr addrspace(5) %simpleuser, align 8
; CHECK-NEXT:   => Final Score:1
; CHECK-LABEL: Analyzing:   %manyusers = alloca [4 x i64], align 4, addrspace(5)
; CHECK-NEXT: Scoring:   %manyusers = alloca [4 x i64], align 4, addrspace(5)
; CHECK-NEXT:   [+1]:   store i64 %v0.add, ptr addrspace(5) %manyusers.1, align 8
; CHECK-NEXT:   [+1]:   %v0 = load i64, ptr addrspace(5) %manyusers.1, align 8
; CHECK-NEXT:   [+1]:   store i64 %v1.add, ptr addrspace(5) %manyusers.2, align 8
; CHECK-NEXT:   [+1]:   %v1 = load i64, ptr addrspace(5) %manyusers.2, align 8
; CHECK-NEXT:   => Final Score:4
; CHECK-NEXT: Sorted Worklist:
; CHECK-NEXT:     %manyusers = alloca [4 x i64], align 4, addrspace(5)
; CHECK-NEXT:     %simpleuser = alloca [4 x i64], align 4, addrspace(5)
define amdgpu_kernel void @simple_users_scores() #0 {
entry:
  ; should get a score of 1
  %simpleuser = alloca [4 x i64], align 4, addrspace(5)
  ; should get a score of 4
  %manyusers = alloca [4 x i64], align 4, addrspace(5)

  store i64 42, ptr addrspace(5) %simpleuser

  %manyusers.1 = getelementptr i64, ptr addrspace(5) %manyusers, i64 2
  %v0 = load i64, ptr addrspace(5)  %manyusers.1
  %v0.add = add i64 %v0, 1
  store i64 %v0.add, ptr addrspace(5) %manyusers.1

  %manyusers.2 = getelementptr i64, ptr addrspace(5) %manyusers, i64 1
  %v1 = load i64, ptr addrspace(5)  %manyusers.2
  %v1.add = add i64 %v0, 1
  store i64 %v1.add, ptr addrspace(5) %manyusers.2

  ret void
}

; CHECK-LABEL: Analyzing:   %stack = alloca [4 x i64], align 4, addrspace(5)
; CHECK-NEXT: Scoring:   %stack = alloca [4 x i64], align 4, addrspace(5)
; CHECK-NEXT:   [+5]:   store i64 32, ptr addrspace(5) %stack, align 8
; CHECK-NEXT:   [+1]:   store i64 42, ptr addrspace(5) %stack, align 8
; CHECK-NEXT:   [+9]:   store i64 32, ptr addrspace(5) %stack.1, align 8
; CHECK-NEXT:   [+5]:   %outer = load i64, ptr addrspace(5) %stack.1, align 8
; CHECK-NEXT:   [+1]:   store i64 64, ptr addrspace(5) %stack.2, align 8
; CHECK-NEXT:   [+9]:   %inner = load i64, ptr addrspace(5) %stack.2, align 8
; CHECK-NEXT:   => Final Score:30
define amdgpu_kernel void @loop_users_alloca(i1 %x, i2) #0 {
entry:
  ; should get a score of 1
  %stack = alloca [4 x i64], align 4, addrspace(5)
  %stack.1 = getelementptr i8, ptr addrspace(5) %stack, i64 8
  %stack.2 = getelementptr i8, ptr addrspace(5) %stack, i64 16

  store i64 42, ptr addrspace(5) %stack
  br label %loop.outer

loop.outer:
  store i64 32, ptr addrspace(5) %stack
  %outer = load i64, ptr addrspace(5) %stack.1
  br label %loop.inner

loop.inner:
  store i64 32, ptr addrspace(5) %stack.1
  %inner = load i64, ptr addrspace(5) %stack.2
  %inner.cmp = icmp sge i64 %inner, 0
  br i1 %inner.cmp, label %loop.inner, label %loop.outer

exit:
  store i64 64, ptr addrspace(5) %stack.2
  ret void
}

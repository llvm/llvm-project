; RUN: opt -S -mtriple=amdgcn-unknown-amdhsa -mcpu=kaveri -debug-only=amdgpu-promote-alloca -amdgpu-promote-alloca-to-vector-limit=512 -passes=amdgpu-promote-alloca %s -o - 2>&1 | FileCheck %s
; REQUIRES: asserts

; CHECK:      Scoring:   %simpleuser = alloca [4 x i64], align 4, addrspace(5)
; CHECK-NEXT:   [+1]:   store i32 42, ptr addrspace(5) %simpleuser, align 4
; CHECK-NEXT:   => Final Score:1
; CHECK-NEXT: Scoring:   %manyusers = alloca [4 x i64], align 4, addrspace(5)
; CHECK-NEXT:   [+1]:   store i32 %v0.ext, ptr addrspace(5) %manyusers.1, align 4
; CHECK-NEXT:   [+1]:   %v0 = load i8, ptr addrspace(5) %manyusers.1, align 1
; CHECK-NEXT:   [+1]:   store i32 %v1.ext, ptr addrspace(5) %manyusers.2, align 4
; CHECK-NEXT:   [+1]:   %v1 = load i8, ptr addrspace(5) %manyusers.2, align 1
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

  store i32 42, ptr addrspace(5) %simpleuser

  %manyusers.1 = getelementptr i8, ptr addrspace(5) %manyusers, i64 2
  %v0 = load i8, ptr addrspace(5)  %manyusers.1
  %v0.ext = zext i8 %v0 to i32
  store i32 %v0.ext, ptr addrspace(5) %manyusers.1

  %manyusers.2 = getelementptr i8, ptr addrspace(5) %manyusers, i64 1
  %v1 = load i8, ptr addrspace(5)  %manyusers.2
  %v1.ext = zext i8 %v0 to i32
  store i32 %v1.ext, ptr addrspace(5) %manyusers.2

  ret void
}

; CHECK:      Scoring:   %stack = alloca [4 x i64], align 4, addrspace(5)
; CHECK-NEXT:   [+5]:   store i32 32, ptr addrspace(5) %stack, align 4
; CHECK-NEXT:   [+1]:   store i32 42, ptr addrspace(5) %stack, align 4
; CHECK-NEXT:   [+9]:   store i32 32, ptr addrspace(5) %stack.1, align 4
; CHECK-NEXT:   [+5]:   %outer.cmp = load i1, ptr addrspace(5) %stack.1, align 1
; CHECK-NEXT:   [+1]:   store i32 64, ptr addrspace(5) %stack.2, align 4
; CHECK-NEXT:   [+9]:   %inner.cmp = load i1, ptr addrspace(5) %stack.2, align 1
; CHECK-NEXT:   => Final Score:30
define amdgpu_kernel void @loop_users_alloca(i1 %x, i2) #0 {
entry:
  ; should get a score of 1
  %stack = alloca [4 x i64], align 4, addrspace(5)
  %stack.1 = getelementptr i8, ptr addrspace(5) %stack, i64 4
  %stack.2 = getelementptr i8, ptr addrspace(5) %stack, i64 8

  store i32 42, ptr addrspace(5) %stack
  br label %loop.outer

loop.outer:
  store i32 32, ptr addrspace(5) %stack
  %outer.cmp = load i1, ptr addrspace(5) %stack.1
  br label %loop.inner

loop.inner:
  store i32 32, ptr addrspace(5) %stack.1
  %inner.cmp = load i1, ptr addrspace(5) %stack.2
  br i1 %inner.cmp, label %loop.inner, label %loop.outer

exit:
  store i32 64, ptr addrspace(5) %stack.2
  ret void
}

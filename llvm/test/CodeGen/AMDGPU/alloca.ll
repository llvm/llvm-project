; RUN: llvm-as < %s | llvm-dis | FileCheck %s
; RUN: llc -mtriple=amdgcn-amd-amdhsa < %s
; RUN: llvm-as < %s | llc -mtriple=amdgcn-amd-amdhsa
; RUN: opt -S < %s
; RUN: llvm-as < %s | opt -S

; CHECK: %tmp = alloca i32, align 4, addrspace(5)
define amdgpu_kernel void @test() {
  %tmp = alloca i32, addrspace(5)
  ret void
}


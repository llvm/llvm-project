; RUN: llvm-as -data-layout=A5 < %s | llvm-dis | FileCheck %s
; RUN: llc -mtriple amdgcn-amd-amdhsa < %s
; RUN: llvm-as -data-layout=A5 < %s | llc -mtriple amdgcn-amd-amdhsa
; RUN: opt -data-layout=A5 -S < %s
; RUN: llvm-as -data-layout=A5 < %s | opt -S

; CHECK: %tmp = alloca i32, addrspace(5)
define amdgpu_kernel void @test() {
  %tmp = alloca i32, addrspace(5)
  ret void
}


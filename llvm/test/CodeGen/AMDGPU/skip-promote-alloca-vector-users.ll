; RUN: opt -S -mtriple=amdgcn-amd-amdhsa -passes=amdgpu-promote-alloca < %s | FileCheck %s

; Do not promote an alloca with users of vector/aggregate type.

; CHECK-LABEL: @test_insertelement(
; CHECK:  %alloca = alloca i16
; CHECK-NEXT:  insertelement <2 x ptr addrspace(5)> undef, ptr addrspace(5) %alloca, i32 0
define amdgpu_kernel void @test_insertelement() #0 {
entry:
  %alloca = alloca i16, align 4, addrspace(5)
  %in = insertelement <2 x ptr addrspace(5)> undef, ptr addrspace(5) %alloca, i32 0
  store <2 x ptr addrspace(5)> %in, ptr undef, align 4
  ret void
}

; CHECK-LABEL: @test_insertvalue(
; CHECK:  %alloca = alloca i16
; CHECK-NEXT:  insertvalue { ptr addrspace(5) } undef, ptr addrspace(5) %alloca, 0
define amdgpu_kernel void @test_insertvalue() #0 {
entry:
  %alloca = alloca i16, align 4, addrspace(5)
  %in = insertvalue { ptr addrspace(5) } undef, ptr addrspace(5) %alloca, 0
  store { ptr addrspace(5) } %in, ptr undef, align 4
  ret void
}

; CHECK-LABEL: @test_insertvalue_array(
; CHECK:  %alloca = alloca i16
; CHECK-NEXT:  insertvalue [2 x ptr addrspace(5)] undef, ptr addrspace(5) %alloca, 0
define amdgpu_kernel void @test_insertvalue_array() #0 {
entry:
  %alloca = alloca i16, align 4, addrspace(5)
  %in = insertvalue [2 x ptr addrspace(5)] undef, ptr addrspace(5) %alloca, 0
  store [2 x ptr addrspace(5)] %in, ptr undef, align 4
  ret void
}

attributes #0 = { nounwind }

; RUN: opt -S -mtriple=amdgcn-amd-amdhsa -amdgpu-promote-alloca < %s | FileCheck %s

; Do not promote an alloca with users of vector/aggregate type.

; CHECK-LABEL: @test_insertelement(
; CHECK:  %alloca = alloca i16
; CHECK-NEXT:  insertelement <2 x i16*> undef, i16* %alloca, i32 0
define amdgpu_kernel void @test_insertelement() #0 {
entry:
  %alloca = alloca i16, align 4
  %in = insertelement <2 x i16*> undef, i16* %alloca, i32 0
  store <2 x i16*> %in, <2 x i16*>* undef, align 4
  ret void
}

; CHECK-LABEL: @test_insertvalue(
; CHECK:  %alloca = alloca i16
; CHECK-NEXT:  insertvalue { i16* } undef, i16* %alloca, 0
define amdgpu_kernel void @test_insertvalue() #0 {
entry:
  %alloca = alloca i16, align 4
  %in = insertvalue { i16* } undef, i16* %alloca, 0
  store { i16* } %in, { i16* }* undef, align 4
  ret void
}

; CHECK-LABEL: @test_insertvalue_array(
; CHECK:  %alloca = alloca i16
; CHECK-NEXT:  insertvalue [2 x i16*] undef, i16* %alloca, 0
define amdgpu_kernel void @test_insertvalue_array() #0 {
entry:
  %alloca = alloca i16, align 4
  %in = insertvalue [2 x i16*] undef, i16* %alloca, 0
  store [2 x i16*] %in, [2 x i16*]* undef, align 4
  ret void
}

attributes #0 = { nounwind }

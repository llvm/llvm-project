; RUN: split-file %s %t
; RUN: opt -mtriple=amdgcn-amd-amdhsa -passes=amdgpu-printf-runtime-binding -S < %t/openmp.ll | FileCheck %s
; RUN: opt -mtriple=amdgcn-amd-amdhsa-llvm -passes=amdgpu-printf-runtime-binding -S < %t/llvm-env.ll | FileCheck %s

; CHECK-LABEL: define void @test(
; CHECK: call i32 (ptr addrspace(4), ...) @printf(ptr addrspace(4) @format.str, i32 %n)
; CHECK-NOT: __printf_alloc

;--- openmp.ll
@format.str = private unnamed_addr addrspace(4) constant [8 x i8] c"arst %d\00", align 1

define void @test(i32 %n) {
  %call = call i32 (ptr addrspace(4), ...) @printf(ptr addrspace(4) @format.str, i32 %n)
  ret void
}

declare i32 @printf(ptr addrspace(4), ...)

!llvm.module.flags = !{!0}
!0 = !{i32 7, !"openmp", i32 51}

;--- llvm-env.ll
@format.str = private unnamed_addr addrspace(4) constant [8 x i8] c"arst %d\00", align 1

define void @test(i32 %n) {
  %call = call i32 (ptr addrspace(4), ...) @printf(ptr addrspace(4) @format.str, i32 %n)
  ret void
}

declare i32 @printf(ptr addrspace(4), ...)

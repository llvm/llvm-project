; RUN: opt -S -mtriple=amdgpu9.50-amd-amdhsa -passes=amdgpu-promote-alloca -disable-promote-alloca-to-vector < %s | FileCheck %s

@lds_12800 = internal addrspace(3) global [12800 x i8] poison, align 16
attributes #0 = { "amdgpu-flat-work-group-size"="64,64" }

; This is a regression test for a bug in getMaxLocalMemSizeWithWaveCount
; which did not round to the LDS allocation block size, leading
; AMDGPUPromoteAlloca pass to overestimate the available LDS.

; We have LocalMemorySize = 163840, allowing for floor(163840/ 12800)
; = 12 workgroups and occupancy 12 / 4 = 3 with the 12800 bytes of LDS
; usage and one wave per workgroup.

; Without aligning down to the LDS granularity of 1280 in
; getMaxLocalMemSizeWithWaveCount, the limit in alloca promotion is
; 163840 / 12 = 13653 bytes which led to the promotion of the alloca
; in the function.

; With rounding down, the promotion limit is 12800 bytes. The alloca
; would add 64 * 10 bytes which exceeds the promotion limit.
; This prevents the promotion of the alloca.

; CHECK-LABEL: @test(
; CHECK: %stack = alloca [10 x i8], align 1, addrspace(5)

define amdgpu_kernel void @test(ptr addrspace(1) %out, i32 %idx) #0 {

  %stack = alloca [10 x i8], align 1, addrspace(5)
  %lds.ptr = getelementptr inbounds [12800 x i8], ptr addrspace(3) @lds_12800, i32 0, i32 0
  store volatile i8 1, ptr addrspace(3) %lds.ptr, align 1

  %arrayidx = getelementptr inbounds [10 x i8], ptr addrspace(5) %stack, i32 0, i32 %idx
  store i8 7, ptr addrspace(5) %arrayidx, align 1
  %load = load i8, ptr addrspace(5) %arrayidx, align 1
  store i8 %load, ptr addrspace(1) %out, align 1
  ret void
}

; RUN: opt -mtriple=amdgcn--amdhsa -S -passes=inline -inline-threshold=0 -debug-only=inline-cost %s 2>&1 | FileCheck %s

; REQUIRES: asserts

target datalayout = "A5"

; Verify we are properly adding cost of the -amdgpu-inline-arg-alloca-cost to the threshold.

define void @local_access_only(ptr addrspace(5) %p, i32 %idx) {
  %arrayidx = getelementptr inbounds [64 x float], ptr addrspace(5) %p, i32 0, i32 %idx
  %value = load float, ptr addrspace(5) %arrayidx
  store float %value , ptr addrspace(5) %arrayidx, align 4
  ret void
}

; Below the cutoff, the alloca cost is 0, and only the cost of the instructions saved by sroa is counted
; CHECK: Analyzing call of local_access_only... (caller:test_inliner_sroa_single_below_cutoff)
; CHECK: NumAllocaArgs: 1
; CHECK: SROACostSavings: 10
; CHECK: SROACostSavingsLost: 0
; CHECK: Threshold: 66000
define amdgpu_kernel void @test_inliner_sroa_single_below_cutoff(ptr addrspace(1) %a, i32 %n) {
entry:
  %pvt_arr = alloca [64 x float], align 4, addrspace(5)
  call void @local_access_only(ptr addrspace(5) %pvt_arr, i32 4)
  ret void
}

; Above the cutoff, attribute a cost to the alloca
; CHECK: Analyzing call of local_access_only... (caller:test_inliner_sroa_single_above_cutoff)
; CHECK: NumAllocaArgs: 1
; CHECK: SROACostSavings: 66010
; CHECK: SROACostSavingsLost: 0
; CHECK: Threshold: 66000
define amdgpu_kernel void @test_inliner_sroa_single_above_cutoff(ptr addrspace(1) %a, i32 %n) {
entry:
  %pvt_arr = alloca [65 x float], align 4, addrspace(5)
  call void @local_access_only(ptr addrspace(5) %pvt_arr, i32 4)
  ret void
}

define void @use_first_externally(ptr addrspace(5) %p1, ptr addrspace(5) %p2) {
  call void @external(ptr addrspace(5) %p1)
  %arrayidx = getelementptr inbounds [64 x float], ptr addrspace(5) %p2, i32 0, i32 7
  %value = load float, ptr addrspace(5) %arrayidx
  store float %value , ptr addrspace(5) %arrayidx, align 4
  ret void
}

define void @use_both_externally(ptr addrspace(5) %p1, ptr addrspace(5) %p2) {
  call void @external(ptr addrspace(5) %p1)
  call void @external(ptr addrspace(5) %p2)
  ret void
}

; One array cannot get handled by SROA 
; CHECK: Analyzing call of use_first_externally... (caller:test_inliner_sroa_double)
; CHECK: NumAllocaArgs: 2
; CHECK: SROACostSavings: 32502
; CHECK: SROACostSavingsLost: 33507
; CHECK: Threshold: 66000
define amdgpu_kernel void @test_inliner_sroa_double() {
entry:
  %pvt_arr1 = alloca [33 x float], align 4, addrspace(5)
  %pvt_arr2 = alloca [32 x float], align 4, addrspace(5)
  call void @use_first_externally(ptr addrspace(5) %pvt_arr1, ptr addrspace(5) %pvt_arr2)
  ret void
}

; The two arrays cannot get handled by SROA 
; CHECK: Analyzing call of use_both_externally... (caller:test_inliner_no_sroa)
; CHECK: NumAllocaArgs: 2
; CHECK: SROACostSavings: 0
; CHECK: SROACostSavingsLost: 65999
; CHECK: Threshold: 66000
define amdgpu_kernel void @test_inliner_no_sroa() {
entry:
  %pvt_arr1 = alloca [33 x float], align 4, addrspace(5)
  %pvt_arr2 = alloca [32 x float], align 4, addrspace(5)
  call void @use_both_externally(ptr addrspace(5) %pvt_arr1, ptr addrspace(5) %pvt_arr2)
  ret void
}

; No private arrays
; CHECK: Analyzing call of use_both_externally... (caller:test_inliner_no_alloc)
; CHECK: NumAllocaArgs: 0
; CHECK: SROACostSavings: 0
; CHECK: SROACostSavingsLost: 0
; CHECK: Threshold: 0
define amdgpu_kernel void @test_inliner_no_alloc(ptr addrspace(5) %a, ptr addrspace(5) %b) {
entry:
  call void @use_both_externally(ptr addrspace(5) %a, ptr addrspace(5) %b)
  ret void
}

declare void @external(ptr addrspace(5) %p)

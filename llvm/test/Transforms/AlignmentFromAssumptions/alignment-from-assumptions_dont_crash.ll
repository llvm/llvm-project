; Test that we don't crash.
; RUN: opt < %s -passes=alignment-from-assumptions -S | FileCheck %s

target datalayout = "e-p:64:64-p1:64:64-p2:32:32-p3:32:32-p4:64:64-p5:32:32-p6:32:32-p7:160:256:256:32-p8:128:128-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024-v2048:2048-n32:64-S32-A5-G1-ni:7:8"
target triple = "amdgcn-amd-amdhsa"

define amdgpu_kernel void @test_gep(ptr addrspace(1) nocapture readonly %arg, ptr addrspace(3) nocapture %arg1) {
; CHECK-LABEL: @test_gep
; GEPs are supported so the alignment is changed from 2 to 4
; CHECK: load i32, ptr addrspace(1) %tmp2, align 4
bb:
  %tmp2 = getelementptr inbounds i32, ptr addrspace(1) %arg, i64 1
  call void @llvm.assume(i1 true) [ "align"(ptr addrspace(1) %tmp2, i64 4) ]
  %tmp3 = load i32, ptr addrspace(1) %tmp2, align 2
  %tmp4 = getelementptr inbounds i32, ptr addrspace(3) %arg1, i32 1
  store i32 %tmp3, ptr addrspace(3) %tmp4, align 4
  ret void
}

define amdgpu_kernel void @test_phi(ptr addrspace(1) nocapture readonly %arg, i32 %idx, ptr addrspace(3) nocapture %arg1) {
; CHECK-LABEL: @test_phi
; PHI is not supported - align 2 not changed
; CHECK: load i32, ptr addrspace(1) %tmp2, align 2
bb:
  %cond = icmp ugt i32 %idx, 10
  br i1 %cond, label %bb1, label %bb2
  
bb1:
  %gep1 = getelementptr i32, ptr addrspace(1) %arg, i32 6
  br label %bb3
  
bb2:
  %gep2 = getelementptr i32, ptr addrspace(1) %arg, i32 7
  br label %bb3

bb3:
  %gep3 = phi ptr addrspace(1) [%gep1, %bb1], [%gep2, %bb2]
  %tmp2 = getelementptr inbounds i32, ptr addrspace(1) %gep3, i64 4
  call void @llvm.assume(i1 true) [ "align"(ptr addrspace(1) %arg, i64 4) ]
  %tmp3 = load i32, ptr addrspace(1) %tmp2, align 2
  %tmp4 = getelementptr inbounds i32, ptr addrspace(3) %arg1, i32 1
  store i32 %tmp3, ptr addrspace(3) %tmp4, align 4
  ret void
}

define amdgpu_kernel void @test_loop_phi(ptr addrspace(1) nocapture readonly %arg, i32 %idx, ptr addrspace(3) nocapture %arg1) {
; CHECK-LABEL: @test_loop_phi
; PHI is supported - align 2 changed to 4
; CHECK: load i32, ptr addrspace(1) %gep, align 4
bb:
  %ptr = getelementptr i32, ptr addrspace(1) %arg, i32 0
  %end = getelementptr i32, ptr addrspace(1) %arg, i32 10
  %cond = icmp ugt i32 %idx, 10
  br i1 %cond, label %bb1, label %bb2

bb1:
  %ptr1 = phi ptr addrspace(1) [%ptr, %bb], [%ptr2, %bb1]
  %acc1 = phi i32 [0, %bb], [%acc2, %bb1]
  %gep = getelementptr i32, ptr addrspace(1) %ptr1, i32 4
  call void @llvm.assume(i1 true) [ "align"(ptr addrspace(1) %arg, i64 4) ]
  %val = load i32, ptr addrspace(1) %gep, align 2
  %acc2 = add i32 %acc1, %val
  %ptr2 = getelementptr i32, ptr addrspace(1) %ptr1, i32 %idx
  %exit = icmp eq ptr addrspace(1) %ptr2, %end
  br i1 %exit, label %bb1, label %bb2

bb2:
  %sum = phi i32 [0, %bb], [%acc2, %bb1]
  %tmp4 = getelementptr inbounds i32, ptr addrspace(3) %arg1, i32 1
  store i32 %sum, ptr addrspace(3) %tmp4, align 4
  ret void
}

define amdgpu_kernel void @test_select(ptr addrspace(1) nocapture readonly %arg, i32 %idx, ptr addrspace(3) nocapture %arg1) {
; CHECK-LABEL: @test_select
; select is not supported - align 2 not changed
; CHECK: load i32, ptr addrspace(1) %tmp2, align 2
bb:
  %cond = icmp ugt i32 %idx, 10
  %off1_gep = getelementptr i32, ptr addrspace(1) %arg, i32 6
  %off2_gep = getelementptr i32, ptr addrspace(1) %arg, i32 7
  %tmp2 = select i1 %cond, ptr addrspace(1) %off1_gep, ptr addrspace(1) %off2_gep
  call void @llvm.assume(i1 true) [ "align"(ptr addrspace(1) %arg, i64 4) ]
  %tmp3 = load i32, ptr addrspace(1) %tmp2, align 2
  %tmp4 = getelementptr inbounds i32, ptr addrspace(3) %arg1, i32 1
  store i32 %tmp3, ptr addrspace(3) %tmp4, align 4
  ret void
}

define amdgpu_kernel void @test_cast(ptr addrspace(1) nocapture readonly %arg, i32 %idx, ptr addrspace(3) nocapture %arg1) {
bb:
; CHECK-LABEL: @test_cast
; addrspacecast is not supported - align 2 not changed
; CHECK: load i32, ptr addrspace(1) %tmp2, align 2
; store is a user of the GEP so, align 2 is changed to 4
; CHECK: store i32 %tmp3, ptr addrspace(3) %tmp4, align 4
  %cast = addrspacecast ptr addrspace(3) %arg1 to ptr addrspace(1)
  %tmp2 = getelementptr i32, ptr addrspace(1) %cast
  call void @llvm.assume(i1 true) [ "align"(ptr addrspace(3) %arg1, i64 4) ]
  %tmp3 = load i32, ptr addrspace(1) %tmp2, align 2
  %tmp4 = getelementptr inbounds i32, ptr addrspace(3) %arg1, i32 1
  store i32 %tmp3, ptr addrspace(3) %tmp4, align 2
  ret void
}

define amdgpu_kernel void @test_store_ptr(ptr addrspace(1) nocapture readonly %arg, ptr addrspace(3) nocapture %arg1) {
bb:
; CHECK-LABEL: @test_store_ptr
; GEPs are supported so the alignment is changed from 2 to 4
; CHECK: load i32, ptr addrspace(1) %tmp2, align 4
; This store uses a pointer not as adress but as a value to store!
; CHECK: store ptr addrspace(1) %tmp2, ptr addrspace(3) %tmp4, align 2 
  %tmp2 = getelementptr inbounds i32, ptr addrspace(1) %arg, i64 1
  call void @llvm.assume(i1 true) [ "align"(ptr addrspace(1) %arg, i64 4) ]
  %tmp3 = load i32, ptr addrspace(1) %tmp2, align 2
  %tmp4 = getelementptr inbounds i32, ptr addrspace(3) %arg1, i32 1
  store i32 %tmp3, ptr addrspace(3) %tmp4, align 4
  store ptr addrspace(1) %tmp2, ptr addrspace(3) %tmp4, align 2
  ret void
}

declare void @llvm.assume(i1 noundef)

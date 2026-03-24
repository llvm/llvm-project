; RUN: opt -S -verify-each -mtriple=amdgcn-amd-amdhsa -mcpu=gfx900 -passes=amdgpu-promote-alloca < %s | FileCheck %s
; RUN: llc -mtriple=amdgcn-amd-amdhsa -mcpu=gfx900 -mattr=+promote-alloca -filetype=null < %s
;
; Regression test for stale analysis with two allocas using memintrinsics.
; Multiple profitable allocas are analyzed up-front, then rewritten. The pass
; must not invalidate analysis snapshots for later allocas while rewriting
; earlier ones.

declare void @llvm.memcpy.p5.p5.i64(ptr addrspace(5) nocapture, ptr addrspace(5) nocapture, i64, i1 immarg)
declare void @llvm.memmove.p5.p5.i64(ptr addrspace(5) nocapture, ptr addrspace(5) nocapture, i64, i1 immarg)

define amdgpu_kernel void @two_alloca_memintrinsics(i32 %idx, ptr addrspace(1) %out) {
; CHECK-LABEL: @two_alloca_memintrinsics(
; CHECK-NOT: alloca [4 x i32]
; CHECK: @two_alloca_memintrinsics.a
; CHECK: @two_alloca_memintrinsics.b
; CHECK: call void @llvm.memmove.p3.p3.i64
; CHECK: call void @llvm.memcpy.p3.p3.i64
entry:
  %a = alloca [4 x i32], align 16, addrspace(5)
  %b = alloca [4 x i32], align 16, addrspace(5)

  %a.src = getelementptr inbounds [4 x i32], ptr addrspace(5) %a, i32 0, i32 %idx
  %a.dst = getelementptr inbounds [4 x i32], ptr addrspace(5) %a, i32 0, i32 1
  call void @llvm.memmove.p5.p5.i64(ptr addrspace(5) align 4 %a.dst, ptr addrspace(5) align 4 %a.src, i64 8, i1 false)
  %a.val = load i32, ptr addrspace(5) %a.dst, align 4

  %b.row = getelementptr inbounds [4 x i32], ptr addrspace(5) %b, i32 0, i32 %idx
  store i32 7, ptr addrspace(5) %b.row, align 4
  %b.byte = getelementptr inbounds i8, ptr addrspace(5) %b.row, i32 0
  %b.base = getelementptr inbounds [4 x i32], ptr addrspace(5) %b, i32 0, i32 0
  call void @llvm.memcpy.p5.p5.i64(ptr addrspace(5) align 4 %b.base, ptr addrspace(5) align 4 %b.byte, i64 8, i1 false)
  %b.idx1 = getelementptr inbounds [4 x i32], ptr addrspace(5) %b, i32 0, i32 1
  %b.val = load i32, ptr addrspace(5) %b.idx1, align 4

  %sum = add i32 %a.val, %b.val
  store i32 %sum, ptr addrspace(1) %out, align 4
  ret void
}

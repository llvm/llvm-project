; RUN: opt -S -verify-each -mtriple=amdgcn-amd-amdhsa -mcpu=gfx900 -passes=amdgpu-promote-alloca < %s | FileCheck %s
; RUN: llc -mtriple=amdgcn-amd-amdhsa -mcpu=gfx900 -mattr=+promote-alloca -filetype=null < %s
;
; Regression test for stale analysis across mixed promotion modes.
; One alloca is promoted to vector, while two other allocas participate in a
; cross-object memcpy (LDS path). After a successful promotion, remaining
; allocas must be re-analyzed before further rewrites.

declare void @llvm.memcpy.p5.p5.i64(ptr addrspace(5) nocapture, ptr addrspace(5) nocapture, i64, i1 immarg)

define amdgpu_kernel void @mixed_promotion_shared_transfer(i32 %idx, ptr addrspace(1) %out) {
; CHECK-LABEL: @mixed_promotion_shared_transfer(
; CHECK-NOT: alloca [4 x i32]
; CHECK-DAG: @mixed_promotion_shared_transfer.a
; CHECK-DAG: @mixed_promotion_shared_transfer.b
; CHECK: %vec = freeze <4 x i32> poison
; CHECK: call void @llvm.memcpy.p3.p3.i64
entry:
  %vec = alloca [4 x i32], align 16, addrspace(5)
  %a = alloca [4 x i32], align 16, addrspace(5)
  %b = alloca [4 x i32], align 16, addrspace(5)

  ; Vector-friendly alloca.
  %vec.0 = getelementptr inbounds [4 x i32], ptr addrspace(5) %vec, i32 0, i32 0
  %vec.1 = getelementptr inbounds [4 x i32], ptr addrspace(5) %vec, i32 0, i32 1
  store i32 11, ptr addrspace(5) %vec.0, align 4
  store i32 22, ptr addrspace(5) %vec.1, align 4
  %vec.val = load i32, ptr addrspace(5) %vec.1, align 4

  ; Cross-object transfer between allocas a and b.
  %a.base = getelementptr inbounds [4 x i32], ptr addrspace(5) %a, i32 0, i32 0
  store i32 7, ptr addrspace(5) %a.base, align 4
  %b.base = getelementptr inbounds [4 x i32], ptr addrspace(5) %b, i32 0, i32 0
  call void @llvm.memcpy.p5.p5.i64(ptr addrspace(5) align 4 %b.base, ptr addrspace(5) align 4 %a.base, i64 16, i1 false)

  %b.row = getelementptr inbounds [4 x i32], ptr addrspace(5) %b, i32 0, i32 %idx
  %b.val = load i32, ptr addrspace(5) %b.row, align 4

  %sum = add i32 %vec.val, %b.val
  store i32 %sum, ptr addrspace(1) %out, align 4
  ret void
}

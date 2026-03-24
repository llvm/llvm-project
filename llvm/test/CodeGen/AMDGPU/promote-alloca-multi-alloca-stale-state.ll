; RUN: opt -S -verify-each -mtriple=amdgcn-amd-amdhsa -mcpu=gfx900 -passes=amdgpu-promote-alloca < %s | FileCheck %s
; RUN: llc -mtriple=amdgcn-amd-amdhsa -mcpu=gfx900 -mattr=+promote-alloca -filetype=null < %s
;
; Regression test for stale state across multiple profitable allocas.
; This is intended to exercise single-pass analyze + sequential rewrite paths
; with many cached uses/worklists alive at once.

declare void @llvm.memcpy.p5.p5.i64(ptr addrspace(5) nocapture, ptr addrspace(5) nocapture, i64, i1 immarg)
declare void @llvm.memmove.p5.p5.i64(ptr addrspace(5) nocapture, ptr addrspace(5) nocapture, i64, i1 immarg)
declare i32 @llvm.objectsize.i32.p5(ptr addrspace(5), i1, i1, i1)

define amdgpu_kernel void @multi_alloca_stale_state(i32 %idx, ptr addrspace(1) %out) {
; CHECK-LABEL: @multi_alloca_stale_state(
; CHECK-NOT: alloca [4 x i32]
; CHECK-DAG: @multi_alloca_stale_state.a
; CHECK-DAG: @multi_alloca_stale_state.b
; CHECK: %c = freeze <4 x i32> poison
; CHECK: call void @llvm.memmove.p3.p3.i64
; CHECK: call void @llvm.memcpy.p3.p3.i64
entry:
  %a = alloca [4 x i32], align 16, addrspace(5)
  %b = alloca [4 x i32], align 16, addrspace(5)
  %c = alloca [4 x i32], align 16, addrspace(5)

  ; Alloca a: memmove within the same object.
  %a.src = getelementptr inbounds [4 x i32], ptr addrspace(5) %a, i32 0, i32 %idx
  %a.dst = getelementptr inbounds [4 x i32], ptr addrspace(5) %a, i32 0, i32 1
  call void @llvm.memmove.p5.p5.i64(ptr addrspace(5) align 4 %a.dst, ptr addrspace(5) align 4 %a.src, i64 8, i1 false)
  %a.val = load i32, ptr addrspace(5) %a.dst, align 4

  ; Alloca b: GEP-of-GEP plus memcpy within the same object.
  %b.row = getelementptr inbounds [4 x i32], ptr addrspace(5) %b, i32 0, i32 %idx
  store i32 7, ptr addrspace(5) %b.row, align 4
  %b.byte = getelementptr inbounds i8, ptr addrspace(5) %b.row, i32 0
  %b.base = getelementptr inbounds [4 x i32], ptr addrspace(5) %b, i32 0, i32 0
  call void @llvm.memcpy.p5.p5.i64(ptr addrspace(5) align 4 %b.base, ptr addrspace(5) align 4 %b.byte, i64 8, i1 false)
  %b.idx1 = getelementptr inbounds [4 x i32], ptr addrspace(5) %b, i32 0, i32 1
  %b.val = load i32, ptr addrspace(5) %b.idx1, align 4

  ; Alloca c: objectsize plus scalar access.
  %c.sz = call i32 @llvm.objectsize.i32.p5(ptr addrspace(5) %c, i1 false, i1 false, i1 false)
  %c.ptr = getelementptr inbounds [4 x i32], ptr addrspace(5) %c, i32 0, i32 0
  %c.old = load i32, ptr addrspace(5) %c.ptr, align 4
  %c.new = add i32 %c.old, %c.sz
  store i32 %c.new, ptr addrspace(5) %c.ptr, align 4
  %c.val = load i32, ptr addrspace(5) %c.ptr, align 4

  %t0 = add i32 %a.val, %b.val
  %sum = add i32 %t0, %c.val
  store i32 %sum, ptr addrspace(1) %out, align 4
  ret void
}

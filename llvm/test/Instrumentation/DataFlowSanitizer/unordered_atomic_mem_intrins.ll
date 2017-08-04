; RUN: opt < %s -dfsan -dfsan-args-abi -S | FileCheck %s

target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

;; Placeholder tests that will fail once element atomic @llvm.mem[move|set] instrinsics have
;; been added to the MemIntrinsic class hierarchy. These will act as a reminder to
;; verify that dfsan handles these intrinsics properly once they have been
;; added to that class hierarchy.

declare void @llvm.memset.element.unordered.atomic.p0i8.i64(i8* nocapture writeonly, i8, i64, i32) nounwind
declare void @llvm.memmove.element.unordered.atomic.p0i8.p0i8.i64(i8* nocapture writeonly, i8* nocapture readonly, i64, i32) nounwind
declare void @llvm.memcpy.element.unordered.atomic.p0i8.p0i8.i64(i8* nocapture writeonly, i8* nocapture readonly, i64, i32) nounwind

define void @test_memcpy(i8* nocapture, i8* nocapture) {
  ; CHECK-LABEL: dfs$test_memcpy
  ; CHECK: call void @llvm.memcpy.element.unordered.atomic.p0i8.p0i8.i64(i8* align 1 %0, i8* align 1 %1, i64 16, i32 1)
  ; CHECK: ret void
  call void @llvm.memcpy.element.unordered.atomic.p0i8.p0i8.i64(i8* align 1 %0, i8* align 1 %1, i64 16, i32 1)
  ret void
}

define void @test_memmove(i8* nocapture, i8* nocapture) {
  ; CHECK-LABEL: dfs$test_memmove
  ; CHECK: call void @llvm.memmove.element.unordered.atomic.p0i8.p0i8.i64(i8* align 1 %0, i8* align 1 %1, i64 16, i32 1)
  ; CHECK: ret void
  call void @llvm.memmove.element.unordered.atomic.p0i8.p0i8.i64(i8* align 1 %0, i8* align 1 %1, i64 16, i32 1)
  ret void
}

define void @test_memset(i8* nocapture) {
  ; CHECK-LABEL: dfs$test_memset
  ; CHECK: call void @llvm.memset.element.unordered.atomic.p0i8.i64(i8* align 1 %0, i8 88, i64 16, i32 1)
  ; CHECK: ret void
  call void @llvm.memset.element.unordered.atomic.p0i8.i64(i8* align 1 %0, i8 88, i64 16, i32 1)
  ret void
}

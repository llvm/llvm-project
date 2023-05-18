; RUN: opt < %s -passes=dfsan -S | FileCheck %s

target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

;; Placeholder tests that will fail once element atomic @llvm.mem[move|set] intrinsics have
;; been added to the MemIntrinsic class hierarchy. These will act as a reminder to
;; verify that dfsan handles these intrinsics properly once they have been
;; added to that class hierarchy.

declare void @llvm.memset.element.unordered.atomic.p0.i64(ptr nocapture writeonly, i8, i64, i32) nounwind
declare void @llvm.memmove.element.unordered.atomic.p0.p0.i64(ptr nocapture writeonly, ptr nocapture readonly, i64, i32) nounwind
declare void @llvm.memcpy.element.unordered.atomic.p0.p0.i64(ptr nocapture writeonly, ptr nocapture readonly, i64, i32) nounwind

define void @test_memcpy(ptr nocapture, ptr nocapture) {
  ; CHECK-LABEL: test_memcpy.dfsan
  ; CHECK: call void @llvm.memcpy.element.unordered.atomic.p0.p0.i64(ptr align 1 %0, ptr align 1 %1, i64 16, i32 1)
  ; CHECK: ret void
  call void @llvm.memcpy.element.unordered.atomic.p0.p0.i64(ptr align 1 %0, ptr align 1 %1, i64 16, i32 1)
  ret void
}

define void @test_memmove(ptr nocapture, ptr nocapture) {
  ; CHECK-LABEL: test_memmove.dfsan
  ; CHECK: call void @llvm.memmove.element.unordered.atomic.p0.p0.i64(ptr align 1 %0, ptr align 1 %1, i64 16, i32 1)
  ; CHECK: ret void
  call void @llvm.memmove.element.unordered.atomic.p0.p0.i64(ptr align 1 %0, ptr align 1 %1, i64 16, i32 1)
  ret void
}

define void @test_memset(ptr nocapture) {
  ; CHECK-LABEL: test_memset.dfsan
  ; CHECK: call void @llvm.memset.element.unordered.atomic.p0.i64(ptr align 1 %0, i8 88, i64 16, i32 1)
  ; CHECK: ret void
  call void @llvm.memset.element.unordered.atomic.p0.i64(ptr align 1 %0, i8 88, i64 16, i32 1)
  ret void
}

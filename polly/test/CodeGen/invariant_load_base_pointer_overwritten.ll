; RUN: opt %loadNPMPolly '-passes=polly<no-default-opts>' -polly-invariant-load-hoisting=true -polly-process-unprofitable -S < %s | FileCheck %s
;
; Verify that a derived load (MemRef1, BasePtrOriginSAI = MemRef0) is not hoisted
; when the ancestor MemRef0 is written inside the scop.  The ptr load (MemRef0)
; is still hoisted correctly. The i32 load (MemRef1) must not appear in the
; speculative preload path.
;
; CHECK-LABEL: polly.preload.exec:
; CHECK-NEXT:    %.load = load ptr
; CHECK-NOT:     load i32

target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i8:8:32-i16:16:32-i64:64-i128:128-n32:64-S128-Fn32"
target triple = "aarch64-unknown-linux-gnu"

define fastcc void @foo(i32 %arg) {
bb:
  br label %bb1

bb1:
  %getelementptr = getelementptr i8, ptr null, i64 32
  %load          = load ptr, ptr %getelementptr, align 8
  %load2         = load i32, ptr %load,          align 4
  %icmp          = icmp ugt i32 %arg, 0
  br i1 %icmp, label %bb3, label %bb4

bb3:
  store ptr null, ptr null, align 8
  br label %bb4

bb4:
  ret void
}

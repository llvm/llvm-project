; RUN: llc < %s -aarch64-order-frame-objects=0 | FileCheck %s
; Regression test for bug that occured with FP that was not 16-byte aligned.
; We would miscalculate the offset for the st2g.

target datalayout = "e-m:e-i8:8:32-i16:16:32-i64:64-i128:128-n32:64-S128"
target triple = "aarch64-unknown-linux-android10000"

; Function Attrs: sanitize_memtag sspstrong
define void @test(ptr %agg.result, float %call, i32 %size) #1 personality ptr null {
entry:
  %0 = alloca i64, align 8
  %1 = alloca i64, align 8
  %2 = alloca i64, align 8
  %3 = alloca i64, align 8
  %4 = alloca i64, i32 %size, align 8  ; VLA to force use of FP for st2g
  call void @test1(ptr %0)
  call void @test1(ptr %1)
  call void @test1(ptr %2)
  call void @test1(ptr %3)
  store float %call, ptr %agg.result, align 8
  ret void
}

; CHECK-LABEL: test
; CHECK: sub	x8, x29, #88
; CHECK: st2g	sp, [x8, #32]
; CHECK: st2g	sp, [x8]

declare void @test1(ptr)

attributes #1 = { sanitize_memtag sspstrong "frame-pointer"="non-leaf" "target-features"="+mte,+neon" }

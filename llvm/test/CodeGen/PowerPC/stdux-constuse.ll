; RUN: llc -verify-machineinstrs -mcpu=a2 -disable-lsr < %s | FileCheck %s
target datalayout = "E-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v128:128:128-n32:64"
target triple = "powerpc64-unknown-linux-gnu"

define i32 @test1(i64 %add, ptr %ptr) nounwind {
entry:
  %p1 = getelementptr i64, ptr %ptr, i64 144115188075855
  br label %for.cond2.preheader

for.cond2.preheader:
  %nl.018 = phi i32 [ 0, %entry ], [ %inc9, %for.end ]
  br label %for.body4

for.body4:
  %lsr.iv = phi i32 [ %lsr.iv.next, %for.body4 ], [ 16000, %for.cond2.preheader ]
  %i0 = phi ptr [ %p1, %for.cond2.preheader ], [ %i6, %for.body4 ]
  %i6 = getelementptr i64, ptr %i0, i64 400000
  %i7 = getelementptr i64, ptr %i6, i64 300000
  %i8 = getelementptr i64, ptr %i6, i64 200000
  %i9 = getelementptr i64, ptr %i6, i64 100000
  store i64 %add, ptr %i6, align 32
  store i64 %add, ptr %i7, align 32
  store i64 %add, ptr %i8, align 32
  store i64 %add, ptr %i9, align 32
  %lsr.iv.next = add i32 %lsr.iv, -16
  %exitcond.15 = icmp eq i32 %lsr.iv.next, 0
  br i1 %exitcond.15, label %for.end, label %for.body4

; Make sure that we generate the most compact form of this loop with no
; unnecessary moves
; CHECK: @test1
; CHECK: mtctr
; CHECK: stdux
; CHECK-NEXT: stdx
; CHECK-NEXT: stdx
; CHECK-NEXT: stdx
; CHECK-NEXT: bdnz

for.end:
  %inc9 = add nsw i32 %nl.018, 1
  %exitcond = icmp eq i32 %inc9, 400000
  br i1 %exitcond, label %for.end10, label %for.cond2.preheader

for.end10:
  ret i32 0
}


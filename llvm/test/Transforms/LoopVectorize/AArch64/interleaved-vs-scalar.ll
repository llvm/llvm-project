; REQUIRES: asserts
; RUN: opt < %s -force-vector-width=2 -force-vector-interleave=1 -passes=loop-vectorize -S --debug-only=loop-vectorize 2>&1 | FileCheck %s

target datalayout = "e-m:e-i8:8:32-i16:16:32-i64:64-i128:128-n32:64-S128"
target triple = "aarch64--linux-gnu"

%pair = type { i8, i8 }

; CHECK-LABEL: test
; CHECK: Found an estimated cost of 14 for VF 2 For instruction:   {{.*}} load i8
; CHECK: Found an estimated cost of 0 for VF 2 For instruction:   {{.*}} load i8
; CHECK-LABEL: entry:
; CHECK-LABEL: vector.body:
; CHECK: [[LOAD1:%.*]] = load i8
; CHECK: [[LOAD2:%.*]] = load i8
; CHECK: [[INSERT:%.*]] = insertelement <2 x i8> poison, i8 [[LOAD1]], i32 0
; CHECK: insertelement <2 x i8> [[INSERT]], i8 [[LOAD2]], i32 1
; CHECK: br i1 {{.*}}, label %middle.block, label %vector.body

define void @test(ptr %p, ptr %q, i64 %n) {
entry:
  br label %for.body

for.body:
  %i = phi i64 [ 0, %entry ], [ %i.next, %for.body ]
  %tmp0 = getelementptr %pair, ptr %p, i64 %i, i32 0
  %tmp1 = load i8, ptr %tmp0, align 1
  %tmp2 = getelementptr %pair, ptr %p, i64 %i, i32 1
  %tmp3 = load i8, ptr %tmp2, align 1
  %add = add i8 %tmp1, %tmp3
  %qi = getelementptr i8, ptr %q, i64 %i
  store i8 %add, ptr %qi, align 1
  %i.next = add nuw nsw i64 %i, 1
  %cond = icmp eq i64 %i.next, %n
  br i1 %cond, label %for.end, label %for.body

for.end:
  ret void
}

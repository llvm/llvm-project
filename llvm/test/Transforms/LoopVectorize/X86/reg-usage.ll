; RUN: opt < %s -debug-only=loop-vectorize -loop-vectorize -vectorizer-maximize-bandwidth -O2 -S 2>&1 | FileCheck %s
; REQUIRES: asserts

target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

@a = global [1024 x i8] zeroinitializer, align 16
@b = global [1024 x i8] zeroinitializer, align 16

define i32 @foo() {
; This function has a loop of SAD pattern. Here we check when VF = 16 the
; register usage doesn't exceed 16.
;
; CHECK-LABEL: foo
; CHECK:      LV(REG): VF = 4
; CHECK-NEXT: LV(REG): Found max usage: 4
; CHECK:      LV(REG): VF = 8
; CHECK-NEXT: LV(REG): Found max usage: 7
; CHECK:      LV(REG): VF = 16
; CHECK-NEXT: LV(REG): Found max usage: 13

entry:
  br label %for.body

for.cond.cleanup:
  %add.lcssa = phi i32 [ %add, %for.body ]
  ret i32 %add.lcssa

for.body:
  %indvars.iv = phi i64 [ 0, %entry ], [ %indvars.iv.next, %for.body ]
  %s.015 = phi i32 [ 0, %entry ], [ %add, %for.body ]
  %arrayidx = getelementptr inbounds [1024 x i8], [1024 x i8]* @a, i64 0, i64 %indvars.iv
  %0 = load i8, i8* %arrayidx, align 1
  %conv = zext i8 %0 to i32
  %arrayidx2 = getelementptr inbounds [1024 x i8], [1024 x i8]* @b, i64 0, i64 %indvars.iv
  %1 = load i8, i8* %arrayidx2, align 1
  %conv3 = zext i8 %1 to i32
  %sub = sub nsw i32 %conv, %conv3
  %ispos = icmp sgt i32 %sub, -1
  %neg = sub nsw i32 0, %sub
  %2 = select i1 %ispos, i32 %sub, i32 %neg
  %add = add nsw i32 %2, %s.015
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  %exitcond = icmp eq i64 %indvars.iv.next, 1024
  br i1 %exitcond, label %for.cond.cleanup, label %for.body
}

define i64 @bar(i64* nocapture %a) {
; CHECK-LABEL: bar
; CHECK:       LV(REG): VF = 2
; CHECK:       LV(REG): Found max usage: 4
;
entry:
  br label %for.body

for.cond.cleanup:
  %add2.lcssa = phi i64 [ %add2, %for.body ]
  ret i64 %add2.lcssa

for.body:
  %i.012 = phi i64 [ 0, %entry ], [ %inc, %for.body ]
  %s.011 = phi i64 [ 0, %entry ], [ %add2, %for.body ]
  %arrayidx = getelementptr inbounds i64, i64* %a, i64 %i.012
  %0 = load i64, i64* %arrayidx, align 8
  %add = add nsw i64 %0, %i.012
  store i64 %add, i64* %arrayidx, align 8
  %add2 = add nsw i64 %add, %s.011
  %inc = add nuw nsw i64 %i.012, 1
  %exitcond = icmp eq i64 %inc, 1024
  br i1 %exitcond, label %for.cond.cleanup, label %for.body
}

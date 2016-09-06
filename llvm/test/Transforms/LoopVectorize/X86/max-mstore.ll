; RUN: opt -basicaa -loop-vectorize -force-vector-interleave=1 -S -mcpu=core-avx2

target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

@b = common global [256 x i32] zeroinitializer, align 16
@a = common global [256 x i32] zeroinitializer, align 16

; unsigned int a[256], b[256];
; void foo() {
;  for (i = 0; i < 256; i++) {
;    if (b[i] > a[i])
;      a[i] = b[i];
;  }
; }

; CHECK-LABEL: foo
; CHECK: load <8 x i32>
; CHECK: icmp ugt <8 x i32>
; CHECK: masked.store

define void @foo() {
entry:
  br label %for.body

for.body:                                         ; preds = %for.inc, %entry
  %indvars.iv = phi i64 [ 0, %entry ], [ %indvars.iv.next, %for.inc ]
  %arrayidx = getelementptr inbounds [256 x i32], [256 x i32]* @b, i64 0, i64 %indvars.iv
  %0 = load i32, i32* %arrayidx, align 4
  %arrayidx2 = getelementptr inbounds [256 x i32], [256 x i32]* @a, i64 0, i64 %indvars.iv
  %1 = load i32, i32* %arrayidx2, align 4
  %cmp3 = icmp ugt i32 %0, %1
  br i1 %cmp3, label %if.then, label %for.inc

if.then:                                          ; preds = %for.body
  store i32 %0, i32* %arrayidx2, align 4
  br label %for.inc

for.inc:                                          ; preds = %for.body, %if.then
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  %exitcond = icmp eq i64 %indvars.iv.next, 256
  br i1 %exitcond, label %for.end, label %for.body

for.end:                                          ; preds = %for.inc
  ret void
}

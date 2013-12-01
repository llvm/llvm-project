; RUN: opt -S -loop-vectorize -force-vector-width=2 -force-vector-unroll=1 -mcpu=prescott < %s | FileCheck %s

target datalayout = "e-p:32:32:32-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:32:64-f32:32:32-f64:32:64-v64:64:64-v128:128:128-a0:0:64-f80:32:32-n8:16:32-S128"
target triple = "i386-unknown-freebsd11.0"

@big = external global [0 x i32]

; PR18049
; We need to truncate the exit count to i32. This is legal because the
; arithmetic is signed (%inc is nsw).

; CHECK-LABEL: tripcount
; CHECK: trunc i64 %count to i32

define void @tripcount(i64 %count) {
entry:
  %cmp6 = icmp sgt i64 %count, 0
  br i1 %cmp6, label %for.body.preheader, label %for.end

for.body.preheader:
  br label %for.body

for.body:
  %i.07 = phi i32 [ %inc, %for.body ], [ 0, %for.body.preheader ]
  %arrayidx = getelementptr inbounds [0 x i32]* @big, i32 0, i32 %i.07
  %0 = load i32* %arrayidx, align 4
  %neg = xor i32 %0, -1
  store i32 %neg, i32* %arrayidx, align 4
  %inc = add nsw i32 %i.07, 1
  %conv = sext i32 %inc to i64
  %cmp = icmp slt i64 %conv, %count
  br i1 %cmp, label %for.body, label %for.end.loopexit

for.end.loopexit:
  br label %for.end

for.end:
  ret void
}

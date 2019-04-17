; RUN: opt -S -loop-vectorize -dce -instcombine -force-vector-width=2 -force-vector-interleave=2 < %s | FileCheck %s

target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128-n8:16:32:64-S128"

;PR 15830.

;CHECK-LABEL: @foo(
; When scalarizing stores we need to preserve the original order.
; Make sure that we are extracting in the correct order (0101, and not 0011).
;CHECK: extractelement <2 x i64> {{.*}}, i32 0
;CHECK: extractelement <2 x i64> {{.*}}, i32 1
;CHECK: extractelement <2 x i64> {{.*}}, i32 0
;CHECK: extractelement <2 x i64> {{.*}}, i32 1
;CHECK: store
;CHECK: store
;CHECK: store
;CHECK: store
;CHECK: ret

define i32 @foo(i32* nocapture %A) {
entry:
  br label %for.body

for.body:
  %indvars.iv = phi i64 [ 0, %entry ], [ %indvars.iv.next, %for.body ]
  %0 = shl nsw i64 %indvars.iv, 2
  %arrayidx = getelementptr inbounds i32, i32* %A, i64 %0
  store i32 4, i32* %arrayidx, align 4
  %indvars.iv.next = add i64 %indvars.iv, 1
  %lftr.wideiv = trunc i64 %indvars.iv.next to i32
  %exitcond = icmp eq i32 %lftr.wideiv, 10000
  br i1 %exitcond, label %for.end, label %for.body

for.end:
  ret i32 undef
}



; REQUIRES: asserts
; RUN: opt < %s -loop-vectorize -force-vector-width=4 -force-vector-interleave=1 -instcombine -debug-only=loop-vectorize -disable-output -print-after=instcombine 2>&1 | FileCheck %s

target datalayout = "e-m:e-i64:64-i128:128-n32:64-S128"

; CHECK-LABEL: more_than_one_use
;
; PR30627. Check that a compare instruction with more than one use is not
; recognized as uniform and is vectorized.
;
; CHECK-NOT: Found uniform instruction: %cond = icmp slt i64 %i.next, %n
; CHECK:     vector.body
; CHECK:       %[[I:.+]] = add nuw nsw <4 x i64> %vec.ind, <i64 1, i64 1, i64 1, i64 1>
; CHECK:       icmp slt <4 x i64> %[[I]], %broadcast.splat
; CHECK:       br i1 {{.*}}, label %middle.block, label %vector.body
;
define i32 @more_than_one_use(i32* %a, i64 %n) {
entry:
  br label %for.body

for.body:
  %i = phi i64 [ %i.next, %for.body ], [ 0, %entry ]
  %r = phi i32 [ %tmp3, %for.body ], [ 0, %entry ]
  %i.next = add nuw nsw i64 %i, 1
  %cond = icmp slt i64 %i.next, %n
  %tmp0 = select i1 %cond, i64 %i.next, i64 0
  %tmp1 = getelementptr inbounds i32, i32* %a, i64 %tmp0
  %tmp2 = load i32, i32* %tmp1, align 8
  %tmp3 = add i32 %r, %tmp2
  br i1 %cond, label %for.body, label %for.end

for.end:
  %tmp4 = phi i32 [ %tmp3, %for.body ]
  ret i32 %tmp4
}

; REQUIRES: asserts
; RUN: opt < %s -loop-vectorize -force-vector-width=4 -force-vector-interleave=1 -instcombine -debug-only=loop-vectorize -disable-output -print-after=instcombine 2>&1 -enable-new-pm=0 | FileCheck %s
; RUN: opt < %s -passes=loop-vectorize,instcombine -force-vector-width=4 -force-vector-interleave=1 -debug-only=loop-vectorize -disable-output -print-after=instcombine 2>&1 | FileCheck %s

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

; Check for crash exposed by D76992.
; CHECK:       N0 [label =
; CHECK-NEXT:    "loop:\n" +
; CHECK-NEXT:      "WIDEN-INDUCTION %iv = phi 0, %iv.next\l" +
; CHECK-NEXT:      "WIDEN ir<%cond0> = icmp ir<%iv>, ir<13>\l" +
; CHECK-NEXT:      "WIDEN-SELECT ir<%s> = select ir<%cond0>, ir<10>, ir<20>\l"
; CHECK-NEXT:  ]
define void @test() {
entry:
  br label %loop

loop:                       ; preds = %loop, %entry
  %iv = phi i64 [ 0, %entry ], [ %iv.next, %loop ]
  %cond0 = icmp ult i64 %iv, 13
  %s = select i1 %cond0, i32 10, i32 20
  %iv.next = add nuw nsw i64 %iv, 1
  %exitcond = icmp eq i64 %iv.next, 14
  br i1 %exitcond, label %exit, label %loop

exit:           ; preds = %loop
  ret void
}

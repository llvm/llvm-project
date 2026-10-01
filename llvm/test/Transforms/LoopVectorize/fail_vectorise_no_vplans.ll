; REQUIRES: asserts
; RUN: opt -passes=loop-vectorize -force-vector-width=4 -force-vector-interleave=1 -S -disable-output \
; RUN:   -debug-only=loop-vectorize %s 2>&1 | FileCheck %s

; This is testing that we emit more accurate debug messages when we
; fail to vectorize because we didn't create any vector vplans. Essentially,
; we are doing legalisation in vplan more often for things that used to
; happen in LoopVectorizationLegality.
define void @fail_vplan_bad_users_of_for(ptr noalias %A, ptr noalias %B, ptr noalias %C, i64 %n) {
; CHECK: Checking a loop in 'fail_vplan_bad_users_of_for'
; CHECK: LV: Not vectorizing: Failed to sink or hoist user of first-order recurrence.
; CHECK-NEXT: LV: Vectorization is not possible. Failed to create any vector vplans.
entry:
  br label %loop

loop:
  %iv = phi i64 [ 0, %entry ], [ %iv.next, %loop.latch ]
  %for.1 = phi i32 [ 0, %entry ], [ %prev, %loop.latch ]
  %for.2 = phi i32 [ 0, %entry ], [ %or, %loop.latch ]
  %or = or i32 %for.1, 3
  %t = trunc i64 %iv to i32
  %gep.c = getelementptr inbounds i32, ptr %C, i64 %iv
  %c = load i32, ptr %gep.c, align 4
  %cmp = icmp sgt i32 %c, 0
  br i1 %cmp, label %then, label %else

then:
  %gep.a = getelementptr inbounds i32, ptr %A, i64 %iv
  store i32 %for.2, ptr %gep.a, align 4
  br label %merge

else:
  %gep.b = getelementptr inbounds i32, ptr %B, i64 %iv
  store i32 %for.2, ptr %gep.b, align 4
  br label %merge

merge:
  %prev = mul i32 %t, %t
  br label %loop.latch

loop.latch:
  %iv.next = add i64 %iv, 1
  %ec = icmp eq i64 %iv.next, %n
  br i1 %ec, label %exit, label %loop

exit:
  ret void
}

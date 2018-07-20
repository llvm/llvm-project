;RUN: opt  -loop-unswitch -simplifycfg -S < %s | FileCheck %s

define i32 @foo(i32 %a, i32 %b) {
;CHECK-LABEL: foo
entry:
  br label %for.body.lr.ph

for.body.lr.ph:                                   ; preds = %entry
  %cmp0 = icmp sgt i32 %b, 0
  br i1 %cmp0, label %for.body, label %for.cond.cleanup

for.body:                                         ; preds = %for.inc, %for.body.lr.ph
  %inc.i = phi i32 [ 0, %for.body.lr.ph ], [ %inc, %for.inc ]
  %mul.i = phi i32 [ 3, %for.body.lr.ph ], [ %mul.p, %for.inc ]
  %add.i = phi i32 [ %a, %for.body.lr.ph ], [ %add.p, %for.inc ]
  %cmp1 = icmp eq i32 %a, 12345
  br i1 %cmp1, label %if.then, label %if.else, !prof !0
; CHECK: %cmp1 = icmp eq i32 %a, 12345
; CHECK-NEXT: br i1 %cmp1, label %for.body.us, label %for.body, !prof !0
if.then:                                          ; preds = %for.body
; CHECK: for.body.us:
; CHECK: add nsw i32 %{{.*}}, 123
; CHECK: %exitcond.us = icmp eq i32 %inc.us, %b
; CHECK: br i1 %exitcond.us, label %for.cond.cleanup, label %for.body.us
  %add = add nsw i32 %add.i, 123
  br label %for.inc

if.else:                                          ; preds = %for.body
  %mul = mul nsw i32 %mul.i, %b
  br label %for.inc
; CHECK: for.body:
; CHECK: %mul = mul nsw i32 %mul.i, %b
; CHECK: %inc = add nuw nsw i32 %inc.i, 1
; CHECK: %exitcond = icmp eq i32 %inc, %b
; CHECK: br i1 %exitcond, label %for.cond.cleanup, label %for.body
for.inc:                                          ; preds = %if.then, %if.else
  %mul.p = phi i32 [ %b, %if.then ], [ %mul, %if.else ]
  %add.p = phi i32 [ %add, %if.then ], [ %a, %if.else ]
  %inc = add nuw nsw i32 %inc.i, 1
  %exitcond = icmp eq i32 %inc, %b
  br i1 %exitcond, label %for.cond.cleanup, label %for.body

for.cond.cleanup:                                 ; preds = %for.inc, %for.body.lr.ph
  %t2 = phi i32 [ %b, %for.body.lr.ph ], [ %mul.p, %for.inc ]
  %t1 = phi i32 [ %a, %for.body.lr.ph ], [ %add.p, %for.inc ]
  %add3 = add nsw i32 %t2, %t1
  ret i32 %add3
}

define void @foo_swapped(i32 %a, i32 %b) {
;CHECK-LABEL: foo_swapped
entry:
  br label %for.body
;CHECK: entry:
;CHECK-NEXT: %cmp1 = icmp eq i32 1, 2
;CHECK-NEXT: br i1 %cmp1, label %for.body, label %for.cond.cleanup.split, !prof !1
;CHECK: for.body:
for.body:                                         ; preds = %for.inc, %entry
  %inc.i = phi i32 [ 0, %entry ], [ %inc, %if.then ]
  %add.i = phi i32 [ 100, %entry ], [ %add, %if.then ]
  %inc = add nuw nsw i32 %inc.i, 1
  %cmp1 = icmp eq i32 1, 2
  br i1 %cmp1, label %if.then, label  %for.cond.cleanup, !prof !0

if.then:                                          ; preds = %for.body
  %add = add nsw i32 %a, %add.i

  %exitcond = icmp eq i32 %inc, %b
  br i1 %exitcond, label %for.cond.cleanup, label %for.body

for.cond.cleanup:                                 ; preds = %for.inc, %for.body.lr.ph, %for.body
  ret void
}
!0 = !{!"branch_weights", i32 64, i32 4}

;CHECK: !0 = !{!"branch_weights", i32 64, i32 4}
;CHECK: !1 = !{!"branch_weights", i32 4, i32 64}

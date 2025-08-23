; RUN: opt <%s -p "print<scalar-evolution>" -disable-output 2>&1 | FileCheck %s
; RUN: opt <%s -p "loop(loop-idiom),print<scalar-evolution>" -disable-output 2>&1 | FileCheck %s

; CHECK: backedge-taken count is i64 1

; IR corresponds to the following C test:
; extern char a[];
; void foo() {
;   for (long c = 0; c < 6; c += -2078836808675943215)
;     for (long d; d < 6; d++)
;       a[c + d] = 0;
; }

@a = external global [0 x i8]

define void @foo()  {
entry:
  br label %outerL

outerL:                              ; preds = %entry, %outerLatch
  %e = phi i64 [ poison, %entry ], [ %lcssa, %outerLatch ]
  %c = phi i64 [ 0, %entry ], [ %c.next, %outerLatch ]
  %e.cmp = icmp slt i64 %e, 6
  br i1 %e.cmp, label %innerL, label %outerLatch

innerL:                                          ; preds = %outerL, %innerL
  %d = phi i64 [ %d.next, %innerL ], [ %e, %outerL ]
  %add = add nsw i64 %d, %c
  %arrayidx = getelementptr inbounds [0 x i8], ptr @a, i64 0, i64 %add
  store i8 0, ptr %arrayidx
  %d.next = add nsw i64 %d, 1
  %d.cmp = icmp slt i64 %d, 5
  br i1 %d.cmp, label %innerL, label %outerLatch, !llvm.loop !0

outerLatch:                                         ; preds = %innerL, %outerL
  %lcssa = phi i64 [ %e, %outerL ], [ %d.next, %innerL ]
  %c.next = add nsw i64 %c, -2078836808675943215
  %c.cmp = icmp slt i64 %c, 2078836808675943221
  br i1 %c.cmp, label %outerL, label %exit, !llvm.loop !1

exit:                                         ; preds = %outerLatch
  ret void
}

!0 = distinct !{!0, !2}
!1 = distinct !{!1, !2}
!2 = !{!"llvm.loop.mustprogress"}

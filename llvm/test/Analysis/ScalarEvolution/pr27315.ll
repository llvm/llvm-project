; RUN: opt -analyze -scalar-evolution < %s | FileCheck %s

declare i1 @use(i64)

define void @f_0() {
; CHECK-LABEL: Classifying expressions for: @f_0

; CHECK:  %iv = phi i32 [ 0, %entry ], [ %iv.inc.nowrap, %be ]
; CHECK-NEXT: -->  {0,+,1}<nuw><nsw><%loop>
; CHECK: %iv.inc.maywrap = add i32 %iv, 1
; CHECK-NEXT: -->  {1,+,1}<nuw><%loop>
; CHECK:  %iv.inc.maywrap.sext = sext i32 %iv.inc.maywrap to i64
; CHECK-NEXT:  -->  (sext i32 {1,+,1}<nuw><%loop> to i64)
entry:
  br label %loop

loop:
  %iv = phi i32 [ 0, %entry ], [ %iv.inc.nowrap, %be ]
  %iv.inc.maywrap = add i32 %iv, 1
  %iv.inc.maywrap.sext = sext i32 %iv.inc.maywrap to i64
  %cond0 = call i1 @use(i64 %iv.inc.maywrap.sext)
  br i1 %cond0, label %be, label %leave

be:
  %iv.inc.nowrap = add nsw i32 %iv, 1
  %be.cond = call i1 @use(i64 0) ;; Get an unanalyzable value
  br i1 %be.cond, label %loop, label %leave

leave:
  ret void
}

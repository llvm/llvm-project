; RUN: opt -disable-output "-passes=print<scalar-evolution>" < %s 2>&1 | FileCheck %s

define void @f(ptr %condition) {
; CHECK-LABEL: Classifying expressions for: @f
 entry: 
  br label %loop

 loop:
  %idx = phi i32 [ 0, %entry ], [ %idx.inc, %loop ]
  %idx.inc = add nsw i32 %idx, 1

  %idx.inc2 = add i32 %idx.inc, 1
  %idx.inc2.zext = zext i32 %idx.inc2 to i64

; CHECK: %idx.inc2.zext = zext i32 %idx.inc2 to i64
; CHECK-NEXT: -->  {2,+,1}<nuw><%loop>

  %c = load volatile i1, ptr %condition
  br i1 %c, label %loop, label %exit

 exit:
  ret void
}

define void @g(ptr %condition) {
; CHECK-LABEL: Classifying expressions for: @g
 entry:
  br label %loop

 loop:
  %idx = phi i32 [ 0, %entry ], [ %idx.inc, %loop ]
  %idx.inc = add nsw i32 %idx, 3

  %idx.inc2 = add i32 %idx.inc, -1
  %idx.inc2.sext = sext i32 %idx.inc2 to i64
; CHECK: %idx.inc2.sext = sext i32 %idx.inc2 to i64
; CHECK-NEXT: -->  {2,+,3}<nuw><nsw><%loop>

  %cond.gep = getelementptr inbounds i1, ptr %condition, i32 %idx.inc
  %c = load volatile i1, ptr %cond.gep
  br i1 %c, label %loop, label %exit

 exit:
  ret void
}

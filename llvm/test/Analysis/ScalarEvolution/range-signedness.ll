; RUN: opt -disable-output "-passes=print<scalar-evolution>" < %s 2>&1 | FileCheck %s

define void @x(i1* %cond) {
; CHECK-LABEL: Classifying expressions for: @x
 entry:
  br label %loop

 loop:
  %idx = phi i8 [ 0, %entry ], [ %idx.inc, %loop ]
; CHECK: %idx = phi i8 [ 0, %entry ], [ %idx.inc, %loop ]
; CHECK-NEXT:  -->  {0,+,1}<nuw><nsw><%loop> U: [0,-128) S: [0,-128)

  %idx.inc = add nsw i8 %idx, 1

  %c = load volatile i1, i1* %cond
  br i1 %c, label %loop, label %exit

 exit:
  ret void
}

define void @y(i8* %addr) {
; CHECK-LABEL: Classifying expressions for: @y
 entry:
  br label %loop

 loop:
  %idx = phi i8 [-5, %entry ], [ %idx.inc, %loop ]
; CHECK:   %idx = phi i8 [ -5, %entry ], [ %idx.inc, %loop ]
; CHECK-NEXT:  -->  {-5,+,1}<%loop> U: [-5,6) S: [-5,6)

  %idx.inc = add i8 %idx, 1

  %continue = icmp slt i8 %idx.inc, 6
  br i1 %continue, label %loop, label %exit

 exit:
  ret void
}

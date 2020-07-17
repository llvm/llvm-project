; RUN: opt -analyze -enable-new-pm=0 -scalar-evolution -scalar-evolution < %s | FileCheck %s
; RUN: opt -disable-output "-passes=print<scalar-evolution>,print<scalar-evolution>" < %s 2>&1 | FileCheck %s

define void @bad.nsw() {
; CHECK-LABEL: Classifying expressions for: @bad.nsw
; CHECK-LABEL: Classifying expressions for: @bad.nsw
 entry: 
  br label %loop

 loop:
  %i = phi i8 [ -1, %entry ], [ %i.inc, %loop ]
; CHECK:  %i = phi i8 [ -1, %entry ], [ %i.inc, %loop ]
; CHECK-NEXT: -->  {-1,+,-128}<nw><%loop>
; CHECK-NOT: -->  {-1,+,-128}<nsw><%loop>

  %counter = phi i8 [ 0, %entry ], [ %counter.inc, %loop ]

  %i.inc = add i8 %i, -128
  %i.sext = sext i8 %i to i16

  %counter.inc = add i8 %counter, 1
  %continue = icmp eq i8 %counter, 1
  br i1 %continue, label %exit, label %loop

 exit:
  ret void  
}

; RUN: opt -disable-output "-passes=print<scalar-evolution>" < %s 2>&1 | FileCheck %s

define i32 @f0(i32 %x, i32 %y) {
; CHECK-LABEL: Classifying expressions for: @f0
 entry:
  %c = icmp sgt i32 %y, 0
  br i1 %c, label %add, label %merge

 add:
  %sum = add i32 %x, %y
  br label %merge

 merge:
  %v = phi i32 [ %sum, %add ], [ %x, %entry ]
; CHECK:  %v = phi i32 [ %sum, %add ], [ %x, %entry ]
; CHECK-NEXT:  -->  ((0 smax %y) + %x) U: full-set S: full-set
  ret i32 %v
}

define i32 @f1(i32 %x, i32 %y) {
; CHECK-LABEL: Classifying expressions for: @f1
 entry:
  %c = icmp sge i32 %y, 0
  br i1 %c, label %add, label %merge

 add:
  %sum = add i32 %x, %y
  br label %merge

 merge:
  %v = phi i32 [ %sum, %add ], [ %x, %entry ]
; CHECK:  %v = phi i32 [ %sum, %add ], [ %x, %entry ]
; CHECK-NEXT:  -->  ((0 smax %y) + %x) U: full-set S: full-set
  ret i32 %v
}

define i32 @f2(i32 %x, i32 %y, i32* %ptr) {
; CHECK-LABEL: Classifying expressions for: @f2
 entry:
  %c = icmp sge i32 %y, 0
  br i1 %c, label %add, label %merge

 add:
  %lv = load i32, i32* %ptr
  br label %merge

 merge:
  %v = phi i32 [ %lv, %add ], [ %x, %entry ]
; CHECK:  %v = phi i32 [ %lv, %add ], [ %x, %entry ]
; CHECK-NEXT:  -->  %v U: full-set S: full-set
  ret i32 %v
}

define i32 @f3(i32 %x, i32 %init, i32 %lim) {
; CHECK-LABEL: Classifying expressions for: @f3
 entry:
  br label %loop

loop:
  %iv = phi i32 [ %init, %entry ], [ %iv.inc, %merge ]
  %iv.inc = add i32 %iv, 1
  %c = icmp sge i32 %iv, 0
  br i1 %c, label %add, label %merge

 add:
  %sum = add i32 %x, %iv
  br label %merge

 merge:
  %v = phi i32 [ %sum, %add ], [ %x, %loop ]
; CHECK:  %v = phi i32 [ %sum, %add ], [ %x, %loop ]
; CHECK-NEXT:  -->  ((0 smax {%init,+,1}<%loop>) + %x) U: full-set S: full-set
  %be.cond = icmp eq i32 %iv.inc, %lim
  br i1 %be.cond, label %loop, label %leave

 leave:
  ret i32 0
}

define i32 @f4(i32 %x, i32 %init, i32 %lim) {
; CHECK-LABEL: Classifying expressions for: @f4
 entry:
  %c = icmp sge i32 %init, 0
  br i1 %c, label %add, label %merge

 add:
  br label %loop

 loop:
  %iv = phi i32 [ %init, %add ], [ %iv.inc, %loop ]
  %iv.inc = add i32 %iv, 1
  %be.cond = icmp eq i32 %iv.inc, %lim
  br i1 %be.cond, label %loop, label %add.cont

 add.cont:
  %sum = add i32 %x, %iv
  br label %merge

 merge:
  %v = phi i32 [ %sum, %add.cont ], [ %x, %entry ]
; CHECK:  %v = phi i32 [ %sum, %add.cont ], [ %x, %entry ]
; CHECK-NEXT:  -->  %v U: full-set S: full-set
  ret i32 %v
}

define i32 @f5(i32* %val) {
; CHECK-LABEL: Classifying expressions for: @f5
entry:
  br label %for.end

for.condt:
  br i1 true, label %for.cond.0, label %for.end

for.end:
  %inc = load i32, i32* %val
  br i1 false, label %for.condt, label %for.cond.0

for.cond.0:
  %init = phi i32 [ 0, %for.condt ], [ %inc, %for.end ]

; CHECK:  %init = phi i32 [ 0, %for.condt ], [ %inc, %for.end ]
; CHECK-NEXT:  -->  %init U: full-set S: full-set

; Matching "through" %init will break LCSSA at the SCEV expression
; level.

  ret i32 %init
}

define i32 @f6(i32 %x, i32 %y) {
; Do the right thing for unreachable code:

; CHECK-LABEL: Classifying expressions for: @f6
 entry:
  %c0 = icmp sgt i32 %y, 0
  %sum = add i32 %x, %y
  br i1 %c0, label %merge, label %leave_1

 merge:
  %v0 = phi i32 [ %sum, %entry ], [ %v1, %unreachable ]
  %c1 = icmp slt i32 %y, 0
  br i1 %c1, label %leave_0, label %leave_0_cond

leave_0_cond:
  br label %leave_0

leave_0:
  %v1 = phi i32 [ %v0, %merge ], [ 0, %leave_0_cond ]
  ret i32 0

leave_1:
  ret i32 0

unreachable:
  br label %merge
}

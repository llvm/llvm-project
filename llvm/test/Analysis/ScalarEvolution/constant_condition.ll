; RUN: opt -analyze -enable-new-pm=0 -scalar-evolution < %s | FileCheck %s
; RUN: opt -disable-output "-passes=print<scalar-evolution>" < %s 2>&1 | FileCheck %s

define i32 @branch_true(i32 %x, i32 %y) {
; CHECK-LABEL: Classifying expressions for: @branch_true
 entry:
  br i1 true, label %add, label %merge

 add:
  %sum = add i32 %x, %y
  br label %merge

 merge:
  %v = phi i32 [ %sum, %add ], [ %x, %entry ]
; CHECK:  %v = phi i32 [ %sum, %add ], [ %x, %entry ]
; CHECK-NEXT:  -->  (%x + %y) U: full-set S: full-set
  ret i32 %v
}

define i32 @branch_false(i32 %x, i32 %y) {
; CHECK-LABEL: Classifying expressions for: @branch_false
 entry:
  br i1 false, label %add, label %merge

 add:
  %sum = add i32 %x, %y
  br label %merge

 merge:
  %v = phi i32 [ %sum, %add ], [ %x, %entry ]
; CHECK:  %v = phi i32 [ %sum, %add ], [ %x, %entry ]
; CHECK-NEXT:  -->  %x U: full-set S: full-set
  ret i32 %v
}

define i32 @select_true(i32 %x, i32 %y) {
; CHECK-LABEL: Classifying expressions for: @select_true
 entry:
 %v = select i1 true, i32 %x, i32 %y
; CHECK:  %v = select i1 true, i32 %x, i32 %y
; CHECK-NEXT:  -->  %x U: full-set S: full-set
  ret i32 %v
}

define i32 @select_false(i32 %x, i32 %y) {
; CHECK-LABEL: Classifying expressions for: @select_false
 entry:
 %v = select i1 false, i32 %x, i32 %y
; CHECK:  %v = select i1 false, i32 %x, i32 %y
; CHECK-NEXT:  -->  %y U: full-set S: full-set
  ret i32 %v
}

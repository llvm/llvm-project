; REQUIRES: asserts
; RUN: opt -regions -analyze -enable-new-pm=0 < %s | FileCheck %s
; RUN: opt < %s -passes='print<regions>' 2>&1 | FileCheck %s

; While working on improvements to region info analysis, this test
; case caused an incorrect region bb2 => bb3 to be detected.

define internal i8 @main_read() nounwind {
bb:
   br label %bb1

bb1:
   br i1 true, label %bb2, label %bb7

bb2:
  br i1 true, label %bb4, label %bb3

bb3:
  br i1 true, label %bb4, label %bb8

bb4:
   br label %bb5

bb5:
   br label %bb6

bb6:
   br label %bb1

bb7:
   br label %bb5

bb8:
   ret i8 1
}

; CHECK:    [0] bb => <Function Return>
; CHECK-NEXT: [1] bb1 => bb8
; CHECK-NEXT: End region tree

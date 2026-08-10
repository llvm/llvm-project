; REQUIRES: asserts
; RUN: opt < %s -passes='print<regions>' 2>&1 | FileCheck %s

; While working on improvements to the region info analysis, this test
; case caused an incorrect region bb2 => bb3 to be detected. It is incorrect
; because bb2 has an outgoing edge to bb4. This is interesting because
; bb2 dom bb3 and bb3 pdom bb2, which should have been enough to prevent incoming
; forward edges into the region and outgoing forward edges from the region.

define void @meread_() nounwind {
bb:
   br label %bb1

bb1:                                              ; preds = %bb4, %bb
   br label %bb2

bb2:                                              ; preds = %bb1
  br i1 true, label %bb3, label %bb4

bb3:                                              ; preds = %bb2
  br i1 true, label %bb4, label %bb5

bb4:                                              ; preds = %bb3, %bb2
   br label %bb1

bb5:                                              ; preds = %bb3
   ret void
 }

; CHECK:      [0] bb => <Function Return>
; CHECK-NEXT:   [1] bb1 => bb5
; CHECK-NEXT: End region tree

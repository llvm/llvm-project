; RUN: opt -passes=loop-simplify -S < %s | FileCheck %s

; LoopSimplify shouldn't split loop backedges that use indirectbr.

; CHECK: bb1:                                              ; preds = %bb5, %bb
; CHECK-NEXT: indirectbr

; CHECK: bb5:                                              ; preds = %bb1
; CHECK-NEXT: br label %bb1{{$}}

define void @foo(i8* %p) nounwind {
bb:
  br label %bb1

bb1:                                              ; preds = %bb5, %bb1, %bb
  indirectbr i8* %p, [label %bb6, label %bb7, label %bb1, label %bb2, label %bb3, label %bb5, label %bb4]

bb2:                                              ; preds = %bb1
  ret void

bb3:                                              ; preds = %bb1
  ret void

bb4:                                              ; preds = %bb1
  ret void

bb5:                                              ; preds = %bb1
  br label %bb1

bb6:                                              ; preds = %bb1
  ret void

bb7:                                              ; preds = %bb1
  ret void
}

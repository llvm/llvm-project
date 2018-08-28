; RUN: opt -S -o - %s -inline | FileCheck %s

define void @f1() {
bb.0:
  br i1 false, label %bb.2, label %bb.1

bb.1:                                             ; preds = %bb.0
  br label %bb.2

bb.2:                                             ; preds = %bb.0, %bb.1
  %tmp0 = phi i1 [ true, %bb.1 ], [ false, %bb.0 ]
  br i1 %tmp0, label %bb.4, label %bb.3

bb.3:                                             ; preds = %bb.3, %bb.3
  br i1 undef, label %bb.3, label %bb.3

bb.4:                                             ; preds = %bb.2
  ret void
}

define void @f2() {
bb.0:
  call void @f1()
  ret void
}

; f1 should be inlined into f2 and simplified/collapsed to nothing.

; CHECK-LABEL: define void @f2() {
; CHECK-NEXT:  bb.0:
; CHECK-NEXT:    ret void
; CHECK-NEXT:  }

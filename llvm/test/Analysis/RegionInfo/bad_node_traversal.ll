; REQUIRES: asserts
; RUN: opt -passes='print<regions>' -disable-output < %s 2>&1 | FileCheck %s

; While working on improvements to the region info analysis, this test
; case caused an incorrect region 3 => 8 to be detected.

define internal i8 @wibble() {
bb:
  br i1 true, label %bb1, label %bb8

bb1:                                              ; preds = %bb
  switch i32 0, label %bb2 [
    i32 0, label %bb3
    i32 1, label %bb7
  ]

bb2:                                              ; preds = %bb1
  br label %bb4

bb3:                                              ; preds = %bb1
  br label %bb5

bb4:                                              ; preds = %bb2
  br label %bb6

bb5:                                              ; preds = %bb3
  br label %bb6

bb6:                                              ; preds = %bb5, %bb4
  br label %bb7

bb7:                                              ; preds = %bb6, %bb1
  br label %bb8

bb8:                                              ; preds = %bb7, %bb
  ret i8 1
}

; CHECK:      [0] bb => <Function Return>
; CHECK-NEXT:   [1] bb => bb8
; CHECK-NEXT:     [2] bb1 => bb7
; CHECK-NEXT: End region tree


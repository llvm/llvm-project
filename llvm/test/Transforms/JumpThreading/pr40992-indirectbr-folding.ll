; RUN: opt -S < %s -jump-threading | FileCheck %s

; PR40992: Do not incorrectly fold %bb5 into an unconditional br to %bb7.
;          Also verify we correctly thread %bb1 -> %bb7 when %c is false.

define i32 @jtbr(i1 %v1, i1 %v2, i1 %v3) {
; CHECK: bb0:
bb0:
  br label %bb1

; CHECK: bb1:
; CHECK-NEXT: and
; CHECK-NEXT: br i1 %c, label %bb2, label %bb7
bb1:
  %c = and i1 %v1, %v2
  br i1 %c, label %bb2, label %bb5

; CHECK: bb2:
; CHECK-NEXT: select
; CHECK-NEXT: indirectbr i8* %ba, [label %bb3, label %bb5]
bb2:
  %ba = select i1 %v3, i8* blockaddress(@jtbr, %bb3), i8* blockaddress(@jtbr, %bb4)
  indirectbr i8* %ba, [label %bb3, label %bb4]

; CHECK: bb3:
bb3:
  br label %bb1

; CHECK-NOT: bb4:
bb4:
  br label %bb5

; CHECK: bb5:
bb5:
  br i1 %c, label %bb6, label %bb7

; CHECK: bb6:
bb6:
  ret i32 0

; CHECK: bb7:
bb7:
  ret i32 1
}

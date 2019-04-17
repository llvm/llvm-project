; RUN: opt %s -lowerswitch -S | FileCheck %s

define void @foo(i32 %x, i32* %p) {
; Cases 2 and 4 are removed and become the new default case.
; It is now enough to use two icmps to lower the switch.
;
; CHECK-LABEL: @foo
; CHECK:       icmp slt i32 %x, 5
; CHECK:       icmp eq i32 %x, 1
; CHECK-NOT:   icmp
;
entry:
  switch i32 %x, label %default [
    i32 1, label %bb0
    i32 2, label %popular
    i32 4, label %popular
    i32 5, label %bb1
  ]
bb0:
  store i32 0, i32* %p
  br label %exit
bb1:
  store i32 1, i32* %p
  br label %exit
popular:
  store i32 2, i32* %p
  br label %exit
exit:
  ret void
default:
  unreachable
}

define void @unreachable_gap(i64 %x, i32* %p) {
; Cases 6 and INT64_MAX become the new default, but we still exploit the fact
; that 3-4 is unreachable, so four icmps is enough.

; CHECK-LABEL: @unreachable_gap
; CHECK:       icmp slt i64 %x, 2
; CHECK:       icmp slt i64 %x, 5
; CHECK:       icmp eq  i64 %x, 5
; CHECK:       icmp slt i64 %x, 1
; CHECK-NOT:   icmp

entry:
  switch i64 %x, label %default [
    i64 -9223372036854775808, label %bb0
    i64 1, label %bb1
    i64 2, label %bb2
    i64 5, label %bb3
    i64 6, label %bb4
    i64 9223372036854775807, label %bb4
  ]
bb0:
  store i32 0, i32* %p
  br label %exit
bb1:
  store i32 1, i32* %p
  br label %exit
bb2:
  store i32 2, i32* %p
  br label %exit
bb3:
  store i32 3, i32* %p
  br label %exit
bb4:
  store i32 4, i32* %p
  br label %exit
exit:
  ret void
default:
  unreachable
}



define void @nocases(i32 %x, i32* %p) {
; Don't fall over when there are no cases.
;
; CHECK-LABEL: @nocases
; CHECK-LABEL: entry
; CHECK-NEXT:  br label %default
;
entry:
  switch i32 %x, label %default [
  ]
default:
  unreachable
}

define void @nocasesleft(i32 %x, i32* %p) {
; Cases 2 and 4 are removed and we are left with no cases.
;
; CHECK-LABEL: @nocasesleft
; CHECK-LABEL: entry
; CHECK-NEXT:  br label %popular
;
entry:
  switch i32 %x, label %default [
    i32 2, label %popular
    i32 4, label %popular
  ]
popular:
  store i32 2, i32* %p
  br label %exit
exit:
  ret void
default:
  unreachable
}

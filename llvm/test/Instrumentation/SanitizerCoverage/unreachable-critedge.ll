; RUN: opt < %s -S -passes='module(sancov-module)' -sanitizer-coverage-level=3 | FileCheck %s

; The critical edges to unreachable_bb should not be split.
define i32 @foo(i32 %c, i32 %d) {
; CHECK-LABEL: @foo(
; CHECK:         switch i32 [[C:%.*]], label [[UNREACHABLE_BB:%.*]] [
; CHECK-NEXT:    i32 0, label %exit0
; CHECK-NEXT:    i32 1, label %exit1
; CHECK-NEXT:    i32 2, label %cont
; CHECK-NEXT:    ]
; CHECK:       cont:
; CHECK:         switch i32 [[D:%.*]], label [[UNREACHABLE_BB]] [
; CHECK-NEXT:    i32 0, label %exit2
; CHECK-NEXT:    i32 1, label %exit3
; CHECK-NEXT:    i32 2, label %exit4
; CHECK-NEXT:    ]
; CHECK:       unreachable_bb:
; CHECK-NEXT:    unreachable
;
  switch i32 %c, label %unreachable_bb [i32 0, label %exit0
  i32 1, label %exit1
  i32 2, label %cont]

cont:
  switch i32 %d, label %unreachable_bb [i32 0, label %exit2
  i32 1, label %exit3
  i32 2, label %exit4]

exit0:
  ret i32 0

exit1:
  ret i32 1

exit2:
  ret i32 2

exit3:
  ret i32 3

exit4:
  ret i32 4

unreachable_bb:
  unreachable
}

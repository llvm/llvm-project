; RUN: llc -O3 -o - %s | FileCheck %s

; Test that the output in the presence of an unreachable default does not have
; a compare and branch at the top of the switch to handle the default case.

target triple = "aarch64-unknown-linux-gnu"

; Function Attrs: nounwind
define void @fn(i4) {
  switch i4 %0, label %default [
    i4 0, label %case_0
    i4 1, label %case_1
    i4 2, label %case_2
    i4 3, label %case_3
    i4 4, label %case_4
    i4 5, label %case_5
  ]

; CHECK-LABEL: fn:
; CHECK-NOT:    sub
; CHECK-NOT:    cmp
; CHECK-NOT:    b.hi
; CHECK:        ldrb {{w[0-9]+}}, [{{x[0-9]+}}, {{x[0-9]+}}]
; CHECK:        add {{x[0-9]+}}, {{x[0-9]+}}, {{x[0-9]+}}, lsl #2
; CHECK:        br {{x[0-9]+}}

default:
  unreachable

case_0:
  tail call void @handle_case_00(i4 %0) #2
  br label %return_label

case_1:
  tail call void @handle_case_01(i4 %0) #2
  br label %return_label

case_2:
  tail call void @handle_case_02(i4 %0) #2
  br label %return_label

case_3:
  tail call void @handle_case_03(i4 %0) #2
  br label %return_label

case_4:
  tail call void @handle_case_04(i4 %0) #2
  br label %return_label

case_5:
  tail call void @handle_case_05(i4 %0) #2
  br label %return_label

return_label:
  ret void
}

declare  void @handle_case_00(i4)
declare  void @handle_case_01(i4)
declare  void @handle_case_02(i4)
declare  void @handle_case_03(i4)
declare  void @handle_case_04(i4)
declare  void @handle_case_05(i4)



define i32 @reachable_fallthrough(i32 %x) {
entry:
  switch i32 %x, label %def [
    i32 1,  label %bb1
    i32 8,  label %bb2
    i32 16, label %bb3
    i32 32, label %bb4
    i32 64, label %bb5
  ]

; The switch is lowered with a jump table for cases 1--32 and case 64 handled
; separately. Even though the default of the switch is unreachable, the
; fall-through for the jump table *is* reachable so the range check must be
; emitted.
;
; CHECK-LABEL: reachable_fallthrough
; CHECK: sub [[REG:w[0-9]+]], w0, #1
; CHECK: cmp [[REG]], #31
; CHECK: b.hi

def: unreachable
bb1: br label %return
bb2: br label %return
bb3: br label %return
bb4: br label %return
bb5: br label %return

return:
  %p = phi i32 [ 3, %bb1 ], [ 2, %bb2 ], [ 1, %bb3 ], [ 0, %bb4 ], [ 42, %bb5 ]
  ret i32 %p
}

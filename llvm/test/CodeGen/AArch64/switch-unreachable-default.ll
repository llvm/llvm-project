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

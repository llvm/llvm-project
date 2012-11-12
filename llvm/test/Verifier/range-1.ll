; RUN: not llvm-as < %s -o /dev/null 2>&1 | FileCheck %s

define void @f1(i8* %x) {
entry:
  store i8 0, i8* %x, align 1, !range !0
  ret void
}
!0 = metadata !{i8 0, i8 1}
; CHECK: Ranges are only for loads!
; CHECK-NEXT: store i8 0, i8* %x, align 1, !range !0

define i8 @f2(i8* %x) {
entry:
  %y = load i8* %x, align 1, !range !1
  ret i8 %y
}
!1 = metadata !{}
; CHECK: It should have at least one range!
; CHECK-NEXT: metadata

define i8 @f3(i8* %x) {
entry:
  %y = load i8* %x, align 1, !range !2
  ret i8 %y
}
!2 = metadata !{i8 0}
; CHECK: Unfinished range!

define i8 @f4(i8* %x) {
entry:
  %y = load i8* %x, align 1, !range !3
  ret i8 %y
}
!3 = metadata !{double 0.0, i8 0}
; CHECK: The lower limit must be an integer!

define i8 @f5(i8* %x) {
entry:
  %y = load i8* %x, align 1, !range !4
  ret i8 %y
}
!4 = metadata !{i8 0, double 0.0}
; CHECK: The upper limit must be an integer!

define i8 @f6(i8* %x) {
entry:
  %y = load i8* %x, align 1, !range !5
  ret i8 %y
}
!5 = metadata !{i32 0, i8 0}
; CHECK: Range types must match load type!
; CHECK:  %y = load

define i8 @f7(i8* %x) {
entry:
  %y = load i8* %x, align 1, !range !6
  ret i8 %y
}
!6 = metadata !{i8 0, i32 0}
; CHECK: Range types must match load type!
; CHECK:  %y = load

define i8 @f8(i8* %x) {
entry:
  %y = load i8* %x, align 1, !range !7
  ret i8 %y
}
!7 = metadata !{i32 0, i32 0}
; CHECK: Range types must match load type!
; CHECK:  %y = load

define i8 @f9(i8* %x) {
entry:
  %y = load i8* %x, align 1, !range !8
  ret i8 %y
}
!8 = metadata !{i8 0, i8 0}
; CHECK: Range must not be empty!

define i8 @f10(i8* %x) {
entry:
  %y = load i8* %x, align 1, !range !9
  ret i8 %y
}
!9 = metadata !{i8 0, i8 2, i8 1, i8 3}
; CHECK: Intervals are overlapping

define i8 @f11(i8* %x) {
entry:
  %y = load i8* %x, align 1, !range !10
  ret i8 %y
}
!10 = metadata !{i8 0, i8 2, i8 2, i8 3}
; CHECK: Intervals are contiguous

define i8 @f12(i8* %x) {
entry:
  %y = load i8* %x, align 1, !range !11
  ret i8 %y
}
!11 = metadata !{i8 1, i8 2, i8 -1, i8 0}
; CHECK: Intervals are not in order

define i8 @f13(i8* %x) {
entry:
  %y = load i8* %x, align 1, !range !12
  ret i8 %y
}
!12 = metadata !{i8 1, i8 3, i8 5, i8 1}
; CHECK: Intervals are contiguous

define i8 @f14(i8* %x) {
entry:
  %y = load i8* %x, align 1, !range !13
  ret i8 %y
}
!13 = metadata !{i8 1, i8 3, i8 5, i8 2}
; CHECK: Intervals are overlapping

define i8 @f15(i8* %x) {
entry:
  %y = load i8* %x, align 1, !range !14
  ret i8 %y
}
!14 = metadata !{i8 10, i8 1, i8 12, i8 13}
; CHECK: Intervals are overlapping

define i8 @f16(i8* %x) {
entry:
  %y = load i8* %x, align 1, !range !16
  ret i8 %y
}
!16 = metadata !{i8 1, i8 3, i8 4, i8 5, i8 6, i8 2}
; CHECK: Intervals are overlapping

define i8 @f17(i8* %x) {
entry:
  %y = load i8* %x, align 1, !range !17
  ret i8 %y
}
!17 = metadata !{i8 1, i8 3, i8 4, i8 5, i8 6, i8 1}
; CHECK: Intervals are contiguous

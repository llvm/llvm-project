; RUN: not llvm-as < %s -o /dev/null 2>&1 | FileCheck %s

define void @f1(ptr %x) {
entry:
  store i8 0, ptr %x, align 1, !range !0
  ret void
}
!0 = !{i8 0, i8 1}
; CHECK: Ranges are only for loads, calls and invokes!
; CHECK-NEXT: store i8 0, ptr %x, align 1, !range !0

define i8 @f2(ptr %x) {
entry:
  %y = load i8, ptr %x, align 1, !range !1
  ret i8 %y
}
!1 = !{}
; CHECK: It should have at least one range!

define i8 @f3(ptr %x) {
entry:
  %y = load i8, ptr %x, align 1, !range !2
  ret i8 %y
}
!2 = !{i8 0}
; CHECK: Unfinished range!

define i8 @f4(ptr %x) {
entry:
  %y = load i8, ptr %x, align 1, !range !3
  ret i8 %y
}
!3 = !{double 0.0, i8 0}
; CHECK: The lower limit must be an integer!

define i8 @f5(ptr %x) {
entry:
  %y = load i8, ptr %x, align 1, !range !4
  ret i8 %y
}
!4 = !{i8 0, double 0.0}
; CHECK: The upper limit must be an integer!

define i8 @f6(ptr %x) {
entry:
  %y = load i8, ptr %x, align 1, !range !5
  ret i8 %y
}
!5 = !{i32 0, i8 0}
; CHECK: Range pair types must match!
; CHECK:  %y = load

define i8 @f7(ptr %x) {
entry:
  %y = load i8, ptr %x, align 1, !range !6
  ret i8 %y
}
!6 = !{i8 0, i32 0}
; CHECK: Range pair types must match!
; CHECK:  %y = load

define i8 @f8(ptr %x) {
entry:
  %y = load i8, ptr %x, align 1, !range !7
  ret i8 %y
}
!7 = !{i32 0, i32 0}
; CHECK: Range types must match instruction type!
; CHECK:  %y = load

define i8 @f9(ptr %x) {
entry:
  %y = load i8, ptr %x, align 1, !range !8
  ret i8 %y
}
!8 = !{i8 0, i8 0}
; CHECK: Range must not be empty!

define i8 @f10(ptr %x) {
entry:
  %y = load i8, ptr %x, align 1, !range !9
  ret i8 %y
}
!9 = !{i8 0, i8 2, i8 1, i8 3}
; CHECK: Intervals are overlapping

define i8 @f11(ptr %x) {
entry:
  %y = load i8, ptr %x, align 1, !range !10
  ret i8 %y
}
!10 = !{i8 0, i8 2, i8 2, i8 3}
; CHECK: Intervals are contiguous

define i8 @f12(ptr %x) {
entry:
  %y = load i8, ptr %x, align 1, !range !11
  ret i8 %y
}
!11 = !{i8 1, i8 2, i8 -1, i8 0}
; CHECK: Intervals are not in order

define i8 @f13(ptr %x) {
entry:
  %y = load i8, ptr %x, align 1, !range !12
  ret i8 %y
}
!12 = !{i8 1, i8 3, i8 5, i8 1}
; CHECK: Intervals are contiguous

define i8 @f14(ptr %x) {
entry:
  %y = load i8, ptr %x, align 1, !range !13
  ret i8 %y
}
!13 = !{i8 1, i8 3, i8 5, i8 2}
; CHECK: Intervals are overlapping

define i8 @f15(ptr %x) {
entry:
  %y = load i8, ptr %x, align 1, !range !14
  ret i8 %y
}
!14 = !{i8 10, i8 1, i8 12, i8 13}
; CHECK: Intervals are overlapping

define i8 @f16(ptr %x) {
entry:
  %y = load i8, ptr %x, align 1, !range !16
  ret i8 %y
}
!16 = !{i8 1, i8 3, i8 4, i8 5, i8 6, i8 2}
; CHECK: Intervals are overlapping

define i8 @f17(ptr %x) {
entry:
  %y = load i8, ptr %x, align 1, !range !17
  ret i8 %y
}
!17 = !{i8 1, i8 3, i8 4, i8 5, i8 6, i8 1}
; CHECK: Intervals are contiguous

define i8 @f18() {
entry:
  %y = call i8 undef(), !range !18
  ret i8 %y
}
!18 = !{}
; CHECK: It should have at least one range!

define <2 x i8> @vector_range_wrong_type(ptr %x) {
  %y = load <2 x i8>, ptr %x, !range !19
  ret <2 x i8> %y
}
!19 = !{i16 0, i16 10}
; CHECK: Range types must match instruction type!

define i32 @range_assert(ptr %x) {
  %y = load i32, ptr %x, !range !20
  ret i32 %y
}
; CHECK: The upper and lower limits cannot be the same value{{$}}
!20 = !{i32 123, i32 123}

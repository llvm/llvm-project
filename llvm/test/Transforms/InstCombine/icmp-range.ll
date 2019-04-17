; RUN: opt < %s -instcombine -S | FileCheck %s
; These should be InstSimplify checks, but most of the code
; is currently only in InstCombine.  TODO: move supporting code

; Definitely out of range
define i1 @test_nonzero(i32* nocapture readonly %arg) {
; CHECK-LABEL:test_nonzero
; CHECK: ret i1 true
  %val = load i32, i32* %arg, !range !0
  %rval = icmp ne i32 %val, 0
  ret i1 %rval
}
define i1 @test_nonzero2(i32* nocapture readonly %arg) {
; CHECK-LABEL:test_nonzero2
; CHECK: ret i1 false
  %val = load i32, i32* %arg, !range !0
  %rval = icmp eq i32 %val, 0
  ret i1 %rval
}

; Potentially in range
define i1 @test_nonzero3(i32* nocapture readonly %arg) {
; CHECK-LABEL: test_nonzero3
; Check that this does not trigger - it wouldn't be legal
; CHECK: icmp
  %val = load i32, i32* %arg, !range !1
  %rval = icmp ne i32 %val, 0
  ret i1 %rval
}

; Definitely in range
define i1 @test_nonzero4(i8* nocapture readonly %arg) {
; CHECK-LABEL: test_nonzero4
; CHECK: ret i1 false
  %val = load i8, i8* %arg, !range !2
  %rval = icmp ne i8 %val, 0
  ret i1 %rval
}

define i1 @test_nonzero5(i8* nocapture readonly %arg) {
; CHECK-LABEL: test_nonzero5
; CHECK: ret i1 false
  %val = load i8, i8* %arg, !range !2
  %rval = icmp ugt i8 %val, 0
  ret i1 %rval
}

; Cheaper checks (most values in range meet requirements)
define i1 @test_nonzero6(i8* %argw) {
; CHECK-LABEL: test_nonzero6
; CHECK: icmp ne i8 %val, 0
  %val = load i8, i8* %argw, !range !3
  %rval = icmp sgt i8 %val, 0
  ret i1 %rval
}

; Constant not in range, should return true.
define i1 @test_not_in_range(i32* nocapture readonly %arg) {
; CHECK-LABEL: test_not_in_range
; CHECK: ret i1 true
  %val = load i32, i32* %arg, !range !0
  %rval = icmp ne i32 %val, 6
  ret i1 %rval
}

; Constant in range, can not fold.
define i1 @test_in_range(i32* nocapture readonly %arg) {
; CHECK-LABEL: test_in_range
; CHECK: icmp ne i32 %val, 3
  %val = load i32, i32* %arg, !range !0
  %rval = icmp ne i32 %val, 3
  ret i1 %rval
}

; Values in range greater than constant.
define i1 @test_range_sgt_constant(i32* nocapture readonly %arg) {
; CHECK-LABEL: test_range_sgt_constant
; CHECK: ret i1 true
  %val = load i32, i32* %arg, !range !0
  %rval = icmp sgt i32 %val, 0
  ret i1 %rval
}

; Values in range less than constant.
define i1 @test_range_slt_constant(i32* nocapture readonly %arg) {
; CHECK-LABEL: test_range_slt_constant
; CHECK: ret i1 false
  %val = load i32, i32* %arg, !range !0
  %rval = icmp sgt i32 %val, 6
  ret i1 %rval
}

; Values in union of multiple sub ranges not equal to constant.
define i1 @test_multi_range1(i32* nocapture readonly %arg) {
; CHECK-LABEL: test_multi_range1
; CHECK: ret i1 true
  %val = load i32, i32* %arg, !range !4
  %rval = icmp ne i32 %val, 0
  ret i1 %rval
}

; Values in multiple sub ranges not equal to constant, but in
; union of sub ranges could possibly equal to constant. This
; in theory could also be folded and might be implemented in 
; the future if shown profitable in practice.
define i1 @test_multi_range2(i32* nocapture readonly %arg) {
; CHECK-LABEL: test_multi_range2
; CHECK: icmp ne i32 %val, 7
  %val = load i32, i32* %arg, !range !4
  %rval = icmp ne i32 %val, 7
  ret i1 %rval
}

; Values' ranges overlap each other, so it can not be simplified.
define i1 @test_two_ranges(i32* nocapture readonly %arg1, i32* nocapture readonly %arg2) {
; CHECK-LABEL: test_two_ranges
; CHECK: icmp ult i32 %val2, %val1
  %val1 = load i32, i32* %arg1, !range !5
  %val2 = load i32, i32* %arg2, !range !6
  %rval = icmp ult i32 %val2, %val1
  ret i1 %rval
}

; Values' ranges do not overlap each other, so it can simplified to false.
define i1 @test_two_ranges2(i32* nocapture readonly %arg1, i32* nocapture readonly %arg2) {
; CHECK-LABEL: test_two_ranges2
; CHECK: ret i1 false
  %val1 = load i32, i32* %arg1, !range !0
  %val2 = load i32, i32* %arg2, !range !6
  %rval = icmp ult i32 %val2, %val1
  ret i1 %rval
}

; Values' ranges do not overlap each other, so it can simplified to true.
define i1 @test_two_ranges3(i32* nocapture readonly %arg1, i32* nocapture readonly %arg2) {
; CHECK-LABEL: test_two_ranges3
; CHECK: ret i1 true
  %val1 = load i32, i32* %arg1, !range !0
  %val2 = load i32, i32* %arg2, !range !6
  %rval = icmp ugt i32 %val2, %val1
  ret i1 %rval
}

!0 = !{i32 1, i32 6} 
!1 = !{i32 0, i32 6} 
!2 = !{i8 0, i8 1} 
!3 = !{i8 0, i8 6} 
!4 = !{i32 1, i32 6, i32 8, i32 10}
!5 = !{i32 5, i32 10} 
!6 = !{i32 8, i32 16} 

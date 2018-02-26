; RUN: opt < %s -instcombine -S | FileCheck %s

target datalayout = "e-p:32:32:32-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:32:64-f32:32:32-f64:32:64-v64:64:64-v128:128:128-a0:0:64-f80:128:128"

declare i32 @abs(i32)
declare i64 @labs(i64)
declare i64 @llabs(i64)

; Test that the abs library call simplifier works correctly.
; abs(x) -> x >s -1 ? x : -x.

define i32 @test_abs(i32 %x) {
; CHECK-LABEL: @test_abs(
; CHECK-NEXT:    [[ISPOS:%.*]] = icmp sgt i32 %x, -1
; CHECK-NEXT:    [[NEG:%.*]] = sub i32 0, %x
; CHECK-NEXT:    [[TMP1:%.*]] = select i1 [[ISPOS]], i32 %x, i32 [[NEG]]
; CHECK-NEXT:    ret i32 [[TMP1]]
;
  %ret = call i32 @abs(i32 %x)
  ret i32 %ret
}

define i64 @test_labs(i64 %x) {
; CHECK-LABEL: @test_labs(
; CHECK-NEXT:    [[ISPOS:%.*]] = icmp sgt i64 %x, -1
; CHECK-NEXT:    [[NEG:%.*]] = sub i64 0, %x
; CHECK-NEXT:    [[TMP1:%.*]] = select i1 [[ISPOS]], i64 %x, i64 [[NEG]]
; CHECK-NEXT:    ret i64 [[TMP1]]
;
  %ret = call i64 @labs(i64 %x)
  ret i64 %ret
}

define i64 @test_llabs(i64 %x) {
; CHECK-LABEL: @test_llabs(
; CHECK-NEXT:    [[ISPOS:%.*]] = icmp sgt i64 %x, -1
; CHECK-NEXT:    [[NEG:%.*]] = sub i64 0, %x
; CHECK-NEXT:    [[TMP1:%.*]] = select i1 [[ISPOS]], i64 %x, i64 [[NEG]]
; CHECK-NEXT:    ret i64 [[TMP1]]
;
  %ret = call i64 @llabs(i64 %x)
  ret i64 %ret
}

; The following 5 tests use a shift+add+xor to implement abs():
; B = ashr i8 A, 7  -- smear the sign bit.
; xor (add A, B), B -- add -1 and flip bits if negative

define i8 @shifty_abs_commute0(i8 %x) {
; CHECK-LABEL: @shifty_abs_commute0(
; CHECK-NEXT:    [[TMP1:%.*]] = icmp slt i8 %x, 0
; CHECK-NEXT:    [[TMP2:%.*]] = sub i8 0, %x
; CHECK-NEXT:    [[ABS:%.*]] = select i1 [[TMP1]], i8 [[TMP2]], i8 %x
; CHECK-NEXT:    ret i8 [[ABS]]
;
  %signbit = ashr i8 %x, 7
  %add = add i8 %signbit, %x
  %abs = xor i8 %add, %signbit
  ret i8 %abs
}

define <2 x i8> @shifty_abs_commute1(<2 x i8> %x) {
; CHECK-LABEL: @shifty_abs_commute1(
; CHECK-NEXT:    [[TMP1:%.*]] = icmp slt <2 x i8> %x, zeroinitializer
; CHECK-NEXT:    [[TMP2:%.*]] = sub <2 x i8> zeroinitializer, %x
; CHECK-NEXT:    [[ABS:%.*]] = select <2 x i1> [[TMP1]], <2 x i8> [[TMP2]], <2 x i8> %x
; CHECK-NEXT:    ret <2 x i8> [[ABS]]
;
  %signbit = ashr <2 x i8> %x, <i8 7, i8 7>
  %add = add <2 x i8> %signbit, %x
  %abs = xor <2 x i8> %signbit, %add
  ret <2 x i8> %abs
}

define <2 x i8> @shifty_abs_commute2(<2 x i8> %x) {
; CHECK-LABEL: @shifty_abs_commute2(
; CHECK-NEXT:    [[Y:%.*]] = mul <2 x i8> %x, <i8 3, i8 3>
; CHECK-NEXT:    [[TMP1:%.*]] = icmp slt <2 x i8> [[Y]], zeroinitializer
; CHECK-NEXT:    [[TMP2:%.*]] = sub <2 x i8> zeroinitializer, [[Y]]
; CHECK-NEXT:    [[ABS:%.*]] = select <2 x i1> [[TMP1]], <2 x i8> [[TMP2]], <2 x i8> [[Y]]
; CHECK-NEXT:    ret <2 x i8> [[ABS]]
;
  %y = mul <2 x i8> %x, <i8 3, i8 3>   ; extra op to thwart complexity-based canonicalization
  %signbit = ashr <2 x i8> %y, <i8 7, i8 7>
  %add = add <2 x i8> %y, %signbit
  %abs = xor <2 x i8> %signbit, %add
  ret <2 x i8> %abs
}

define i8 @shifty_abs_commute3(i8 %x) {
; CHECK-LABEL: @shifty_abs_commute3(
; CHECK-NEXT:    [[Y:%.*]] = mul i8 %x, 3
; CHECK-NEXT:    [[TMP1:%.*]] = icmp slt i8 [[Y]], 0
; CHECK-NEXT:    [[TMP2:%.*]] = sub i8 0, [[Y]]
; CHECK-NEXT:    [[ABS:%.*]] = select i1 [[TMP1]], i8 [[TMP2]], i8 [[Y]]
; CHECK-NEXT:    ret i8 [[ABS]]
;
  %y = mul i8 %x, 3                    ; extra op to thwart complexity-based canonicalization
  %signbit = ashr i8 %y, 7
  %add = add i8 %y, %signbit
  %abs = xor i8 %add, %signbit
  ret i8 %abs
}

; Negative test - don't transform if it would increase instruction count.

declare void @extra_use(i8)

define i8 @shifty_abs_too_many_uses(i8 %x) {
; CHECK-LABEL: @shifty_abs_too_many_uses(
; CHECK-NEXT:    [[SIGNBIT:%.*]] = ashr i8 %x, 7
; CHECK-NEXT:    [[ADD:%.*]] = add i8 [[SIGNBIT]], %x
; CHECK-NEXT:    [[ABS:%.*]] = xor i8 [[ADD]], [[SIGNBIT]]
; CHECK-NEXT:    call void @extra_use(i8 [[SIGNBIT]])
; CHECK-NEXT:    ret i8 [[ABS]]
;
  %signbit = ashr i8 %x, 7
  %add = add i8 %x, %signbit
  %abs = xor i8 %add, %signbit
  call void @extra_use(i8 %signbit)
  ret i8 %abs
}


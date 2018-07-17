; RUN: opt < %s -instsimplify -S | FileCheck %s

define i32 @negated_operand(i32 %x) {
; CHECK-LABEL: @negated_operand(
; CHECK-NEXT:    [[NEGX:%.*]] = sub nsw i32 0, [[X:%.*]]
; CHECK-NEXT:    [[DIV:%.*]] = sdiv i32 [[NEGX]], [[X]]
; CHECK-NEXT:    ret i32 [[DIV]]
;
  %negx = sub nsw i32 0, %x
  %div = sdiv i32 %negx, %x
  ret i32 %div
}

define <2 x i32> @negated_operand_commute_vec(<2 x i32> %x) {
; CHECK-LABEL: @negated_operand_commute_vec(
; CHECK-NEXT:    [[NEGX:%.*]] = sub nsw <2 x i32> zeroinitializer, [[X:%.*]]
; CHECK-NEXT:    [[DIV:%.*]] = sdiv <2 x i32> [[NEGX]], [[X]]
; CHECK-NEXT:    ret <2 x i32> [[DIV]]
;
  %negx = sub nsw <2 x i32> zeroinitializer, %x
  %div = sdiv <2 x i32> %negx, %x
  ret <2 x i32> %div
}

define i32 @knownnegation(i32 %x, i32 %y) {
; CHECK-LABEL: @knownnegation(
; CHECK-NEXT:    [[XY:%.*]] = sub nsw i32 [[X:%.*]], [[Y:%.*]]
; CHECK-NEXT:    [[YX:%.*]] = sub nsw i32 [[Y]], [[X]]
; CHECK-NEXT:    [[DIV:%.*]] = sdiv i32 [[XY]], [[YX]]
; CHECK-NEXT:    ret i32 [[DIV]]
;
  %xy = sub nsw i32 %x, %y
  %yx = sub nsw i32 %y, %x
  %div = sdiv i32 %xy, %yx
  ret i32 %div
}

define <2 x i32> @knownnegation_commute_vec(<2 x i32> %x, <2 x i32> %y) {
; CHECK-LABEL: @knownnegation_commute_vec(
; CHECK-NEXT:    [[XY:%.*]] = sub nsw <2 x i32> [[X:%.*]], [[Y:%.*]]
; CHECK-NEXT:    [[YX:%.*]] = sub nsw <2 x i32> [[Y]], [[X]]
; CHECK-NEXT:    [[DIV:%.*]] = sdiv <2 x i32> [[XY]], [[YX]]
; CHECK-NEXT:    ret <2 x i32> [[DIV]]
;
  %xy = sub nsw <2 x i32> %x, %y
  %yx = sub nsw <2 x i32> %y, %x
  %div = sdiv <2 x i32> %xy, %yx
  ret <2 x i32> %div
}

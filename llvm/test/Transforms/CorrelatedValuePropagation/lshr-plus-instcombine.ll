; RUN: opt < %s -passes="correlated-propagation,instcombine" -S | FileCheck %s

; The tests below are the same as in lshr.ll
; Here we test whether the CorrelatedValuePropagation pass 
; composed with InstCombinePass produces the expected optimizations.

; CHECK-LABEL: @trunc_test1
; CHECK-NEXT: [[A1:%.*]] = lshr i32 [[A:%.*]], 16
; CHECK-NEXT: [[CARG:%.*]] = trunc nuw i32 [[A1]] to i16
; CHECK-NEXT: [[CSHIFT:%.*]] = trunc i32 [[B:%.*]] to i16
; CHECK-NEXT: [[C1:%.*]] = lshr i16 [[CARG]], [[CSHIFT]]
; CHECK-NEXT: ret i16 [[C1]]

define i16 @trunc_test1(i32 %a, i32 %b) {
  %a.eff.trunc = lshr i32 %a, 16
  %b.eff.trunc = and i32 %b, 65535
  %c = lshr i32 %a.eff.trunc, %b.eff.trunc
  %c.trunc = trunc i32 %c to i16
  ret i16 %c.trunc
}

; CHECK-LABEL: @trunc_test2
; CHECK-NEXT: [[C1:%.*]] = lshr i16 [[A:%.*]], 2
; CHECK-NEXT: ret i16 [[C1]]

define i16 @trunc_test2(i16 %a) {
  %a.ext = zext i16 %a to i32
  %c = lshr i32 %a.ext, 2
  %c.trunc = trunc i32 %c to i16
  ret i16 %c.trunc
}

; CHECK-LABEL: @trunc_test3
; CHECK-NEXT: [[B:%.*]] = lshr i16 [[A:%.*]], 2
; CHECK-NEXT: [[C:%.*]] = add nuw nsw i16 [[B]], 123
; CHECK-NEXT: ret i16 [[C]]

define i16 @trunc_test3(i16 %a) {
  %a.ext = zext i16 %a to i32
  %b = lshr i32 %a.ext, 2
  %c = add i32 %b, 123
  %c.trunc = trunc i32 %c to i16
  ret i16 %c.trunc
}

; CHECK-LABEL: @trunc_test4
; CHECK-NEXT: [[A1:%.*]] = udiv i32 [[A:%.*]], 17000000
; CHECK-NEXT: [[B:%.*]] = trunc nuw nsw i32 [[A1]] to i16
; CHECK-NEXT: [[B1:%.*]] = lshr i16 [[B]], 2
; CHECK-NEXT: ret i16 [[B1]]

define i16 @trunc_test4(i32 %a) {
  %a.eff.trunc = udiv i32 %a, 17000000  ; larger than 2^24
  %b = lshr i32 %a.eff.trunc, 2 
  %b.trunc.1 = trunc i32 %b to i16
  %b.trunc.2 = trunc i32 %b to i8
  ret i16 %b.trunc.1
}

; CHECK-LABEL: @trunc_test5
; CHECK-NEXT: [[A1:%.*]] = udiv i32 [[A:%.*]], 17000000
; CHECK-NEXT: [[B:%.*]] = lshr i32 [[A1]], 2
; CHECK-NEXT: [[C:%.*]] = add nuw nsw i32 [[B]], 123
; CHECK-NEXT: ret i32 [[C]]

define i32 @trunc_test5(i32 %a) {
  %a.eff.trunc = udiv i32 %a, 17000000  ; larger than 2^24
  %b = lshr i32 %a.eff.trunc, 2 
  %b.trunc.1 = trunc i32 %b to i16
  %b.trunc.2 = trunc i32 %b to i8
  %c = add i32 %b, 123
  ret i32 %c
}

; CHECK-LABEL: @zero_test1
; CHECK-NEXT: ret i32 poison
  
define i32 @zero_test1(i32 %a) {
  %b = lshr i32 %a, 32
  %c = add i32 %b, 123
  ret i32 %c
}

; CHECK-LABEL: @zero_test2
; CHECK-NEXT: ret i32 poison

define i32 @zero_test2(i32 %a, i32 %b) {
  %b.large = add nuw nsw i32 %b, 50
  %c = lshr i32 %a, %b.large
  %d = add i32 %c, 123
  ret i32 %d
}

; CHECK-LABEL: @zero_test3
; CHECK-NEXT: ret i32 123

define i32 @zero_test3(i32 %a, i32 %b) {
  %a.small = lshr i32 %a, 16
  %b.large = add nuw nsw i32 %b, 20
  %c = lshr i32 %a.small, %b.large
  %d = add i32 %c, 123
  ret i32 %d
}

; CHECK-LABEL: @zero_test4
; CHECK-NEXT: ret i32 123

define i32 @zero_test4(i32 %a, i32 %b) {
  %a.small = lshr i32 %a, 16
  %b.large = add nuw nsw i32 %b, 20
  %c = lshr exact i32 %a.small, %b.large
  %d = add i32 %c, 123
  ret i32 %d
}

; RUN: opt < %s -instcombine -S | FileCheck %s

target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128"

; PR4548
define i8 @udiv_i8(i8 %a, i8 %b) {
; CHECK-LABEL: @udiv_i8(
; CHECK-NEXT:    [[DIV:%.*]] = udiv i8 %a, %b
; CHECK-NEXT:    ret i8 [[DIV]]
;
  %za = zext i8 %a to i32
  %zb = zext i8 %b to i32
  %udiv = udiv i32 %za, %zb
  %conv3 = trunc i32 %udiv to i8
  ret i8 %conv3
}

define i8 @urem_i8(i8 %a, i8 %b) {
; CHECK-LABEL: @urem_i8(
; CHECK-NEXT:    [[TMP1:%.*]] = urem i8 %a, %b
; CHECK-NEXT:    ret i8 [[TMP1]]
;
  %za = zext i8 %a to i32
  %zb = zext i8 %b to i32
  %udiv = urem i32 %za, %zb
  %conv3 = trunc i32 %udiv to i8
  ret i8 %conv3
}

define i32 @udiv_i32(i8 %a, i8 %b) {
; CHECK-LABEL: @udiv_i32(
; CHECK-NEXT:    [[DIV:%.*]] = udiv i8 %a, %b
; CHECK-NEXT:    [[UDIV:%.*]] = zext i8 [[DIV]] to i32
; CHECK-NEXT:    ret i32 [[UDIV]]
;
  %za = zext i8 %a to i32
  %zb = zext i8 %b to i32
  %udiv = udiv i32 %za, %zb
  ret i32 %udiv
}

define i32 @urem_i32(i8 %a, i8 %b) {
; CHECK-LABEL: @urem_i32(
; CHECK-NEXT:    [[TMP1:%.*]] = urem i8 %a, %b
; CHECK-NEXT:    [[UDIV:%.*]] = zext i8 [[TMP1]] to i32
; CHECK-NEXT:    ret i32 [[UDIV]]
;
  %za = zext i8 %a to i32
  %zb = zext i8 %b to i32
  %udiv = urem i32 %za, %zb
  ret i32 %udiv
}

define i32 @udiv_i32_c(i8 %a) {
; CHECK-LABEL: @udiv_i32_c(
; CHECK-NEXT:    [[DIV:%.*]] = udiv i8 %a, 10
; CHECK-NEXT:    [[UDIV:%.*]] = zext i8 [[DIV]] to i32
; CHECK-NEXT:    ret i32 [[UDIV]]
;
  %za = zext i8 %a to i32
  %udiv = udiv i32 %za, 10
  ret i32 %udiv
}

define i32 @urem_i32_c(i8 %a) {
; CHECK-LABEL: @urem_i32_c(
; CHECK-NEXT:    [[TMP1:%.*]] = urem i8 %a, 10
; CHECK-NEXT:    [[UDIV:%.*]] = zext i8 [[TMP1]] to i32
; CHECK-NEXT:    ret i32 [[UDIV]]
;
  %za = zext i8 %a to i32
  %udiv = urem i32 %za, 10
  ret i32 %udiv
}

define <2 x i8> @udiv_i8_vec(<2 x i8> %a, <2 x i8> %b) {
; CHECK-LABEL: @udiv_i8_vec(
; CHECK-NEXT:    [[DIV:%.*]] = udiv <2 x i8> %a, %b
; CHECK-NEXT:    ret <2 x i8> [[DIV]]
;
  %za = zext <2 x i8> %a to <2 x i32>
  %zb = zext <2 x i8> %b to <2 x i32>
  %udiv = udiv <2 x i32> %za, %zb
  %conv3 = trunc <2 x i32> %udiv to <2 x i8>
  ret <2 x i8> %conv3
}

define <2 x i8> @urem_i8_vec(<2 x i8> %a, <2 x i8> %b) {
; CHECK-LABEL: @urem_i8_vec(
; CHECK-NEXT:    [[TMP1:%.*]] = urem <2 x i8> %a, %b
; CHECK-NEXT:    ret <2 x i8> [[TMP1]]
;
  %za = zext <2 x i8> %a to <2 x i32>
  %zb = zext <2 x i8> %b to <2 x i32>
  %udiv = urem <2 x i32> %za, %zb
  %conv3 = trunc <2 x i32> %udiv to <2 x i8>
  ret <2 x i8> %conv3
}

define <2 x i32> @udiv_i32_vec(<2 x i8> %a, <2 x i8> %b) {
; CHECK-LABEL: @udiv_i32_vec(
; CHECK-NEXT:    [[DIV:%.*]] = udiv <2 x i8> %a, %b
; CHECK-NEXT:    [[UDIV:%.*]] = zext <2 x i8> [[DIV]] to <2 x i32>
; CHECK-NEXT:    ret <2 x i32> [[UDIV]]
;
  %za = zext <2 x i8> %a to <2 x i32>
  %zb = zext <2 x i8> %b to <2 x i32>
  %udiv = udiv <2 x i32> %za, %zb
  ret <2 x i32> %udiv
}

define <2 x i32> @urem_i32_vec(<2 x i8> %a, <2 x i8> %b) {
; CHECK-LABEL: @urem_i32_vec(
; CHECK-NEXT:    [[TMP1:%.*]] = urem <2 x i8> %a, %b
; CHECK-NEXT:    [[UDIV:%.*]] = zext <2 x i8> [[TMP1]] to <2 x i32>
; CHECK-NEXT:    ret <2 x i32> [[UDIV]]
;
  %za = zext <2 x i8> %a to <2 x i32>
  %zb = zext <2 x i8> %b to <2 x i32>
  %udiv = urem <2 x i32> %za, %zb
  ret <2 x i32> %udiv
}

define <2 x i32> @udiv_i32_c_vec(<2 x i8> %a) {
; CHECK-LABEL: @udiv_i32_c_vec(
; CHECK-NEXT:    [[ZA:%.*]] = zext <2 x i8> %a to <2 x i32>
; CHECK-NEXT:    [[UDIV:%.*]] = udiv <2 x i32> [[ZA]], <i32 10, i32 17>
; CHECK-NEXT:    ret <2 x i32> [[UDIV]]
;
  %za = zext <2 x i8> %a to <2 x i32>
  %udiv = udiv <2 x i32> %za, <i32 10, i32 17>
  ret <2 x i32> %udiv
}

define <2 x i32> @urem_i32_c_vec(<2 x i8> %a) {
; CHECK-LABEL: @urem_i32_c_vec(
; CHECK-NEXT:    [[ZA:%.*]] = zext <2 x i8> %a to <2 x i32>
; CHECK-NEXT:    [[UDIV:%.*]] = urem <2 x i32> [[ZA]], <i32 10, i32 17>
; CHECK-NEXT:    ret <2 x i32> [[UDIV]]
;
  %za = zext <2 x i8> %a to <2 x i32>
  %udiv = urem <2 x i32> %za, <i32 10, i32 17>
  ret <2 x i32> %udiv
}


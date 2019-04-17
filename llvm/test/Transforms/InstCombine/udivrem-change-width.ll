; RUN: opt < %s -instcombine -S | FileCheck %s

target datalayout = "n8:32"

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

define i32 @udiv_i32_multiuse(i8 %a, i8 %b) {
; CHECK-LABEL: @udiv_i32_multiuse(
; CHECK-NEXT:    [[ZA:%.*]] = zext i8 %a to i32
; CHECK-NEXT:    [[ZB:%.*]] = zext i8 %b to i32
; CHECK-NEXT:    [[UDIV:%.*]] = udiv i32 [[ZA]], [[ZB]]
; CHECK-NEXT:    [[EXTRA_USES:%.*]] = add nuw nsw i32 [[ZA]], [[ZB]]
; CHECK-NEXT:    [[R:%.*]] = mul nuw nsw i32 [[UDIV]], [[EXTRA_USES]]
; CHECK-NEXT:    ret i32 [[R]]
;
  %za = zext i8 %a to i32
  %zb = zext i8 %b to i32
  %udiv = udiv i32 %za, %zb
  %extra_uses = add i32 %za, %zb
  %r = mul i32 %udiv, %extra_uses
  ret i32 %r
}

define i32 @udiv_illegal_type(i9 %a, i9 %b) {
; CHECK-LABEL: @udiv_illegal_type(
; CHECK-NEXT:    [[DIV:%.*]] = udiv i9 %a, %b
; CHECK-NEXT:    [[UDIV:%.*]] = zext i9 [[DIV]] to i32
; CHECK-NEXT:    ret i32 [[UDIV]]
;
  %za = zext i9 %a to i32
  %zb = zext i9 %b to i32
  %udiv = udiv i32 %za, %zb
  ret i32 %udiv
}

define i32 @urem_i32(i8 %a, i8 %b) {
; CHECK-LABEL: @urem_i32(
; CHECK-NEXT:    [[TMP1:%.*]] = urem i8 %a, %b
; CHECK-NEXT:    [[UREM:%.*]] = zext i8 [[TMP1]] to i32
; CHECK-NEXT:    ret i32 [[UREM]]
;
  %za = zext i8 %a to i32
  %zb = zext i8 %b to i32
  %urem = urem i32 %za, %zb
  ret i32 %urem
}

define <2 x i32> @urem_i32_vec(<2 x i8> %a, <2 x i8> %b) {
; CHECK-LABEL: @urem_i32_vec(
; CHECK-NEXT:    [[TMP1:%.*]] = urem <2 x i8> %a, %b
; CHECK-NEXT:    [[UREM:%.*]] = zext <2 x i8> [[TMP1]] to <2 x i32>
; CHECK-NEXT:    ret <2 x i32> [[UREM]]
;
  %za = zext <2 x i8> %a to <2 x i32>
  %zb = zext <2 x i8> %b to <2 x i32>
  %urem = urem <2 x i32> %za, %zb
  ret <2 x i32> %urem
}

define i32 @urem_i32_multiuse(i8 %a, i8 %b) {
; CHECK-LABEL: @urem_i32_multiuse(
; CHECK-NEXT:    [[ZA:%.*]] = zext i8 %a to i32
; CHECK-NEXT:    [[ZB:%.*]] = zext i8 %b to i32
; CHECK-NEXT:    [[UREM:%.*]] = urem i32 [[ZA]], [[ZB]]
; CHECK-NEXT:    [[EXTRA_USES:%.*]] = add nuw nsw i32 [[ZA]], [[ZB]]
; CHECK-NEXT:    [[R:%.*]] = mul nuw nsw i32 [[UREM]], [[EXTRA_USES]]
; CHECK-NEXT:    ret i32 [[R]]
;
  %za = zext i8 %a to i32
  %zb = zext i8 %b to i32
  %urem = urem i32 %za, %zb
  %extra_uses = add i32 %za, %zb
  %r = mul i32 %urem, %extra_uses
  ret i32 %r
}

define i32 @urem_illegal_type(i9 %a, i9 %b) {
; CHECK-LABEL: @urem_illegal_type(
; CHECK-NEXT:    [[TMP1:%.*]] = urem i9 %a, %b
; CHECK-NEXT:    [[UREM:%.*]] = zext i9 [[TMP1]] to i32
; CHECK-NEXT:    ret i32 [[UREM]]
;
  %za = zext i9 %a to i32
  %zb = zext i9 %b to i32
  %urem = urem i32 %za, %zb
  ret i32 %urem
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

define <2 x i32> @udiv_i32_c_vec(<2 x i8> %a) {
; CHECK-LABEL: @udiv_i32_c_vec(
; CHECK-NEXT:    [[TMP1:%.*]] = udiv <2 x i8> %a, <i8 10, i8 17>
; CHECK-NEXT:    [[UDIV:%.*]] = zext <2 x i8> [[TMP1]] to <2 x i32>
; CHECK-NEXT:    ret <2 x i32> [[UDIV]]
;
  %za = zext <2 x i8> %a to <2 x i32>
  %udiv = udiv <2 x i32> %za, <i32 10, i32 17>
  ret <2 x i32> %udiv
}

define i32 @udiv_i32_c_multiuse(i8 %a) {
; CHECK-LABEL: @udiv_i32_c_multiuse(
; CHECK-NEXT:    [[ZA:%.*]] = zext i8 %a to i32
; CHECK-NEXT:    [[UDIV:%.*]] = udiv i32 [[ZA]], 10
; CHECK-NEXT:    [[EXTRA_USE:%.*]] = add nuw nsw i32 [[UDIV]], [[ZA]]
; CHECK-NEXT:    ret i32 [[EXTRA_USE]]
;
  %za = zext i8 %a to i32
  %udiv = udiv i32 %za, 10
  %extra_use = add i32 %za, %udiv
  ret i32 %extra_use
}

define i32 @udiv_illegal_type_c(i9 %a) {
; CHECK-LABEL: @udiv_illegal_type_c(
; CHECK-NEXT:    [[DIV:%.*]] = udiv i9 %a, 10
; CHECK-NEXT:    [[UDIV:%.*]] = zext i9 [[DIV]] to i32
; CHECK-NEXT:    ret i32 [[UDIV]]
;
  %za = zext i9 %a to i32
  %udiv = udiv i32 %za, 10
  ret i32 %udiv
}

define i32 @urem_i32_c(i8 %a) {
; CHECK-LABEL: @urem_i32_c(
; CHECK-NEXT:    [[TMP1:%.*]] = urem i8 %a, 10
; CHECK-NEXT:    [[UREM:%.*]] = zext i8 [[TMP1]] to i32
; CHECK-NEXT:    ret i32 [[UREM]]
;
  %za = zext i8 %a to i32
  %urem = urem i32 %za, 10
  ret i32 %urem
}

define <2 x i32> @urem_i32_c_vec(<2 x i8> %a) {
; CHECK-LABEL: @urem_i32_c_vec(
; CHECK-NEXT:    [[TMP1:%.*]] = urem <2 x i8> %a, <i8 10, i8 17>
; CHECK-NEXT:    [[UREM:%.*]] = zext <2 x i8> [[TMP1]] to <2 x i32>
; CHECK-NEXT:    ret <2 x i32> [[UREM]]
;
  %za = zext <2 x i8> %a to <2 x i32>
  %urem = urem <2 x i32> %za, <i32 10, i32 17>
  ret <2 x i32> %urem
}

define i32 @urem_i32_c_multiuse(i8 %a) {
; CHECK-LABEL: @urem_i32_c_multiuse(
; CHECK-NEXT:    [[ZA:%.*]] = zext i8 %a to i32
; CHECK-NEXT:    [[UREM:%.*]] = urem i32 [[ZA]], 10
; CHECK-NEXT:    [[EXTRA_USE:%.*]] = add nuw nsw i32 [[UREM]], [[ZA]]
; CHECK-NEXT:    ret i32 [[EXTRA_USE]]
;
  %za = zext i8 %a to i32
  %urem = urem i32 %za, 10
  %extra_use = add i32 %za, %urem
  ret i32 %extra_use
}

define i32 @urem_illegal_type_c(i9 %a) {
; CHECK-LABEL: @urem_illegal_type_c(
; CHECK-NEXT:    [[TMP1:%.*]] = urem i9 %a, 10
; CHECK-NEXT:    [[UREM:%.*]] = zext i9 [[TMP1]] to i32
; CHECK-NEXT:    ret i32 [[UREM]]
;
  %za = zext i9 %a to i32
  %urem = urem i32 %za, 10
  ret i32 %urem
}

define i32 @udiv_c_i32(i8 %a) {
; CHECK-LABEL: @udiv_c_i32(
; CHECK-NEXT:    [[TMP1:%.*]] = udiv i8 10, %a
; CHECK-NEXT:    [[UDIV:%.*]] = zext i8 [[TMP1]] to i32
; CHECK-NEXT:    ret i32 [[UDIV]]
;
  %za = zext i8 %a to i32
  %udiv = udiv i32 10, %za
  ret i32 %udiv
}

define i32 @urem_c_i32(i8 %a) {
; CHECK-LABEL: @urem_c_i32(
; CHECK-NEXT:    [[TMP1:%.*]] = urem i8 10, %a
; CHECK-NEXT:    [[UREM:%.*]] = zext i8 [[TMP1]] to i32
; CHECK-NEXT:    ret i32 [[UREM]]
;
  %za = zext i8 %a to i32
  %urem = urem i32 10, %za
  ret i32 %urem
}

; Make sure constexpr is handled.

@b = external global [1 x i8]

define i32 @udiv_constexpr(i8 %a) {
; CHECK-LABEL: @udiv_constexpr(
; CHECK-NEXT:    [[TMP1:%.*]] = udiv i8 %a, ptrtoint ([1 x i8]* @b to i8)
; CHECK-NEXT:    [[D:%.*]] = zext i8 [[TMP1]] to i32
; CHECK-NEXT:    ret i32 [[D]]
;
  %za = zext i8 %a to i32
  %d = udiv i32 %za, zext (i8 ptrtoint ([1 x i8]* @b to i8) to i32)
  ret i32 %d
}


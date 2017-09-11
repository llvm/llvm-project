; RUN: opt < %s -instsimplify -S | FileCheck %s

define i32 @sdiv_sext_big_divisor(i8 %x) {
; CHECK-LABEL: @sdiv_sext_big_divisor(
; CHECK-NEXT:    [[CONV:%.*]] = sext i8 %x to i32
; CHECK-NEXT:    [[DIV:%.*]] = sdiv i32 [[CONV]], 129
; CHECK-NEXT:    ret i32 [[DIV]]
;
  %conv = sext i8 %x to i32
  %div = sdiv i32 %conv, 129
  ret i32 %div
}

define i32 @not_sdiv_sext_big_divisor(i8 %x) {
; CHECK-LABEL: @not_sdiv_sext_big_divisor(
; CHECK-NEXT:    [[CONV:%.*]] = sext i8 %x to i32
; CHECK-NEXT:    [[DIV:%.*]] = sdiv i32 [[CONV]], 128
; CHECK-NEXT:    ret i32 [[DIV]]
;
  %conv = sext i8 %x to i32
  %div = sdiv i32 %conv, 128
  ret i32 %div
}

define i32 @sdiv_sext_small_divisor(i8 %x) {
; CHECK-LABEL: @sdiv_sext_small_divisor(
; CHECK-NEXT:    [[CONV:%.*]] = sext i8 %x to i32
; CHECK-NEXT:    [[DIV:%.*]] = sdiv i32 [[CONV]], -129
; CHECK-NEXT:    ret i32 [[DIV]]
;
  %conv = sext i8 %x to i32
  %div = sdiv i32 %conv, -129
  ret i32 %div
}

define i32 @not_sdiv_sext_small_divisor(i8 %x) {
; CHECK-LABEL: @not_sdiv_sext_small_divisor(
; CHECK-NEXT:    [[CONV:%.*]] = sext i8 %x to i32
; CHECK-NEXT:    [[DIV:%.*]] = sdiv i32 [[CONV]], -128
; CHECK-NEXT:    ret i32 [[DIV]]
;
  %conv = sext i8 %x to i32
  %div = sdiv i32 %conv, -128
  ret i32 %div
}

define i32 @sdiv_zext_big_divisor(i8 %x) {
; CHECK-LABEL: @sdiv_zext_big_divisor(
; CHECK-NEXT:    [[CONV:%.*]] = zext i8 %x to i32
; CHECK-NEXT:    [[DIV:%.*]] = sdiv i32 [[CONV]], 256
; CHECK-NEXT:    ret i32 [[DIV]]
;
  %conv = zext i8 %x to i32
  %div = sdiv i32 %conv, 256
  ret i32 %div
}

define i32 @not_sdiv_zext_big_divisor(i8 %x) {
; CHECK-LABEL: @not_sdiv_zext_big_divisor(
; CHECK-NEXT:    [[CONV:%.*]] = zext i8 %x to i32
; CHECK-NEXT:    [[DIV:%.*]] = sdiv i32 [[CONV]], 255
; CHECK-NEXT:    ret i32 [[DIV]]
;
  %conv = zext i8 %x to i32
  %div = sdiv i32 %conv, 255
  ret i32 %div
}

define i32 @sdiv_zext_small_divisor(i8 %x) {
; CHECK-LABEL: @sdiv_zext_small_divisor(
; CHECK-NEXT:    [[CONV:%.*]] = zext i8 %x to i32
; CHECK-NEXT:    [[DIV:%.*]] = sdiv i32 [[CONV]], -256
; CHECK-NEXT:    ret i32 [[DIV]]
;
  %conv = zext i8 %x to i32
  %div = sdiv i32 %conv, -256
  ret i32 %div
}

define i32 @not_sdiv_zext_small_divisor(i8 %x) {
; CHECK-LABEL: @not_sdiv_zext_small_divisor(
; CHECK-NEXT:    [[CONV:%.*]] = zext i8 %x to i32
; CHECK-NEXT:    [[DIV:%.*]] = sdiv i32 [[CONV]], -255
; CHECK-NEXT:    ret i32 [[DIV]]
;
  %conv = zext i8 %x to i32
  %div = sdiv i32 %conv, -255
  ret i32 %div
}

define i32 @sdiv_dividend_known_smaller_than_pos_divisor_clear_bits(i32 %x) {
; CHECK-LABEL: @sdiv_dividend_known_smaller_than_pos_divisor_clear_bits(
; CHECK-NEXT:    [[AND:%.*]] = and i32 %x, 253
; CHECK-NEXT:    [[DIV:%.*]] = sdiv i32 [[AND]], 254
; CHECK-NEXT:    ret i32 [[DIV]]
;
  %and = and i32 %x, 253
  %div = sdiv i32 %and, 254
  ret i32 %div
}

define i32 @not_sdiv_dividend_known_smaller_than_pos_divisor_clear_bits(i32 %x) {
; CHECK-LABEL: @not_sdiv_dividend_known_smaller_than_pos_divisor_clear_bits(
; CHECK-NEXT:    [[AND:%.*]] = and i32 %x, 253
; CHECK-NEXT:    [[DIV:%.*]] = sdiv i32 [[AND]], 253
; CHECK-NEXT:    ret i32 [[DIV]]
;
  %and = and i32 %x, 253
  %div = sdiv i32 %and, 253
  ret i32 %div
}

define i32 @sdiv_dividend_known_smaller_than_neg_divisor_clear_bits(i32 %x) {
; CHECK-LABEL: @sdiv_dividend_known_smaller_than_neg_divisor_clear_bits(
; CHECK-NEXT:    [[AND:%.*]] = and i32 %x, 253
; CHECK-NEXT:    [[DIV:%.*]] = sdiv i32 [[AND]], -254
; CHECK-NEXT:    ret i32 [[DIV]]
;
  %and = and i32 %x, 253
  %div = sdiv i32 %and, -254
  ret i32 %div
}

define i32 @not_sdiv_dividend_known_smaller_than_neg_divisor_clear_bits(i32 %x) {
; CHECK-LABEL: @not_sdiv_dividend_known_smaller_than_neg_divisor_clear_bits(
; CHECK-NEXT:    [[AND:%.*]] = and i32 %x, 253
; CHECK-NEXT:    [[DIV:%.*]] = sdiv i32 [[AND]], -253
; CHECK-NEXT:    ret i32 [[DIV]]
;
  %and = and i32 %x, 253
  %div = sdiv i32 %and, -253
  ret i32 %div
}

define i32 @sdiv_dividend_known_smaller_than_pos_divisor_set_bits(i32 %x) {
; CHECK-LABEL: @sdiv_dividend_known_smaller_than_pos_divisor_set_bits(
; CHECK-NEXT:    [[OR:%.*]] = or i32 %x, -253
; CHECK-NEXT:    [[DIV:%.*]] = sdiv i32 [[OR]], 254
; CHECK-NEXT:    ret i32 [[DIV]]
;
  %or = or i32 %x, -253
  %div = sdiv i32 %or, 254
  ret i32 %div
}

define i32 @not_sdiv_dividend_known_smaller_than_pos_divisor_set_bits(i32 %x) {
; CHECK-LABEL: @not_sdiv_dividend_known_smaller_than_pos_divisor_set_bits(
; CHECK-NEXT:    [[OR:%.*]] = or i32 %x, -253
; CHECK-NEXT:    [[DIV:%.*]] = sdiv i32 [[OR]], 253
; CHECK-NEXT:    ret i32 [[DIV]]
;
  %or = or i32 %x, -253
  %div = sdiv i32 %or, 253
  ret i32 %div
}

define i32 @sdiv_dividend_known_smaller_than_neg_divisor_set_bits(i32 %x) {
; CHECK-LABEL: @sdiv_dividend_known_smaller_than_neg_divisor_set_bits(
; CHECK-NEXT:    [[OR:%.*]] = or i32 %x, -253
; CHECK-NEXT:    [[DIV:%.*]] = sdiv i32 [[OR]], -254
; CHECK-NEXT:    ret i32 [[DIV]]
;
  %or = or i32 %x, -253
  %div = sdiv i32 %or, -254
  ret i32 %div
}

define i32 @not_sdiv_dividend_known_smaller_than_neg_divisor_set_bits(i32 %x) {
; CHECK-LABEL: @not_sdiv_dividend_known_smaller_than_neg_divisor_set_bits(
; CHECK-NEXT:    [[OR:%.*]] = or i32 %x, -253
; CHECK-NEXT:    [[DIV:%.*]] = sdiv i32 [[OR]], -253
; CHECK-NEXT:    ret i32 [[DIV]]
;
  %or = or i32 %x, -253
  %div = sdiv i32 %or, -253
  ret i32 %div
}

define i32 @srem_sext_big_divisor(i8 %x) {
; CHECK-LABEL: @srem_sext_big_divisor(
; CHECK-NEXT:    [[CONV:%.*]] = sext i8 %x to i32
; CHECK-NEXT:    [[REM:%.*]] = srem i32 [[CONV]], 129
; CHECK-NEXT:    ret i32 [[REM]]
;
  %conv = sext i8 %x to i32
  %rem = srem i32 %conv, 129
  ret i32 %rem
}

define i32 @not_srem_sext_big_divisor(i8 %x) {
; CHECK-LABEL: @not_srem_sext_big_divisor(
; CHECK-NEXT:    [[CONV:%.*]] = sext i8 %x to i32
; CHECK-NEXT:    [[REM:%.*]] = srem i32 [[CONV]], 128
; CHECK-NEXT:    ret i32 [[REM]]
;
  %conv = sext i8 %x to i32
  %rem = srem i32 %conv, 128
  ret i32 %rem
}

define i32 @srem_sext_small_divisor(i8 %x) {
; CHECK-LABEL: @srem_sext_small_divisor(
; CHECK-NEXT:    [[CONV:%.*]] = sext i8 %x to i32
; CHECK-NEXT:    [[REM:%.*]] = srem i32 [[CONV]], -129
; CHECK-NEXT:    ret i32 [[REM]]
;
  %conv = sext i8 %x to i32
  %rem = srem i32 %conv, -129
  ret i32 %rem
}

define i32 @not_srem_sext_small_divisor(i8 %x) {
; CHECK-LABEL: @not_srem_sext_small_divisor(
; CHECK-NEXT:    [[CONV:%.*]] = sext i8 %x to i32
; CHECK-NEXT:    [[REM:%.*]] = srem i32 [[CONV]], -128
; CHECK-NEXT:    ret i32 [[REM]]
;
  %conv = sext i8 %x to i32
  %rem = srem i32 %conv, -128
  ret i32 %rem
}

define i32 @srem_zext_big_divisor(i8 %x) {
; CHECK-LABEL: @srem_zext_big_divisor(
; CHECK-NEXT:    [[CONV:%.*]] = zext i8 %x to i32
; CHECK-NEXT:    [[REM:%.*]] = srem i32 [[CONV]], 256
; CHECK-NEXT:    ret i32 [[REM]]
;
  %conv = zext i8 %x to i32
  %rem = srem i32 %conv, 256
  ret i32 %rem
}

define i32 @not_srem_zext_big_divisor(i8 %x) {
; CHECK-LABEL: @not_srem_zext_big_divisor(
; CHECK-NEXT:    [[CONV:%.*]] = zext i8 %x to i32
; CHECK-NEXT:    [[REM:%.*]] = srem i32 [[CONV]], 255
; CHECK-NEXT:    ret i32 [[REM]]
;
  %conv = zext i8 %x to i32
  %rem = srem i32 %conv, 255
  ret i32 %rem
}

define i32 @srem_zext_small_divisor(i8 %x) {
; CHECK-LABEL: @srem_zext_small_divisor(
; CHECK-NEXT:    [[CONV:%.*]] = zext i8 %x to i32
; CHECK-NEXT:    [[REM:%.*]] = srem i32 [[CONV]], -256
; CHECK-NEXT:    ret i32 [[REM]]
;
  %conv = zext i8 %x to i32
  %rem = srem i32 %conv, -256
  ret i32 %rem
}

define i32 @not_srem_zext_small_divisor(i8 %x) {
; CHECK-LABEL: @not_srem_zext_small_divisor(
; CHECK-NEXT:    [[CONV:%.*]] = zext i8 %x to i32
; CHECK-NEXT:    [[REM:%.*]] = srem i32 [[CONV]], -255
; CHECK-NEXT:    ret i32 [[REM]]
;
  %conv = zext i8 %x to i32
  %rem = srem i32 %conv, -255
  ret i32 %rem
}

define i32 @srem_dividend_known_smaller_than_pos_divisor_clear_bits(i32 %x) {
; CHECK-LABEL: @srem_dividend_known_smaller_than_pos_divisor_clear_bits(
; CHECK-NEXT:    [[AND:%.*]] = and i32 %x, 253
; CHECK-NEXT:    [[REM:%.*]] = srem i32 [[AND]], 254
; CHECK-NEXT:    ret i32 [[REM]]
;
  %and = and i32 %x, 253
  %rem = srem i32 %and, 254
  ret i32 %rem
}

define i32 @not_srem_dividend_known_smaller_than_pos_divisor_clear_bits(i32 %x) {
; CHECK-LABEL: @not_srem_dividend_known_smaller_than_pos_divisor_clear_bits(
; CHECK-NEXT:    [[AND:%.*]] = and i32 %x, 253
; CHECK-NEXT:    [[REM:%.*]] = srem i32 [[AND]], 253
; CHECK-NEXT:    ret i32 [[REM]]
;
  %and = and i32 %x, 253
  %rem = srem i32 %and, 253
  ret i32 %rem
}

define i32 @srem_dividend_known_smaller_than_neg_divisor_clear_bits(i32 %x) {
; CHECK-LABEL: @srem_dividend_known_smaller_than_neg_divisor_clear_bits(
; CHECK-NEXT:    [[AND:%.*]] = and i32 %x, 253
; CHECK-NEXT:    [[REM:%.*]] = srem i32 [[AND]], -254
; CHECK-NEXT:    ret i32 [[REM]]
;
  %and = and i32 %x, 253
  %rem = srem i32 %and, -254
  ret i32 %rem
}

define i32 @not_srem_dividend_known_smaller_than_neg_divisor_clear_bits(i32 %x) {
; CHECK-LABEL: @not_srem_dividend_known_smaller_than_neg_divisor_clear_bits(
; CHECK-NEXT:    [[AND:%.*]] = and i32 %x, 253
; CHECK-NEXT:    [[REM:%.*]] = srem i32 [[AND]], -253
; CHECK-NEXT:    ret i32 [[REM]]
;
  %and = and i32 %x, 253
  %rem = srem i32 %and, -253
  ret i32 %rem
}

define i32 @srem_dividend_known_smaller_than_pos_divisor_set_bits(i32 %x) {
; CHECK-LABEL: @srem_dividend_known_smaller_than_pos_divisor_set_bits(
; CHECK-NEXT:    [[OR:%.*]] = or i32 %x, -253
; CHECK-NEXT:    [[REM:%.*]] = srem i32 [[OR]], 254
; CHECK-NEXT:    ret i32 [[REM]]
;
  %or = or i32 %x, -253
  %rem = srem i32 %or, 254
  ret i32 %rem
}

define i32 @not_srem_dividend_known_smaller_than_pos_divisor_set_bits(i32 %x) {
; CHECK-LABEL: @not_srem_dividend_known_smaller_than_pos_divisor_set_bits(
; CHECK-NEXT:    [[OR:%.*]] = or i32 %x, -253
; CHECK-NEXT:    [[REM:%.*]] = srem i32 [[OR]], 253
; CHECK-NEXT:    ret i32 [[REM]]
;
  %or = or i32 %x, -253
  %rem = srem i32 %or, 253
  ret i32 %rem
}

define i32 @srem_dividend_known_smaller_than_neg_divisor_set_bits(i32 %x) {
; CHECK-LABEL: @srem_dividend_known_smaller_than_neg_divisor_set_bits(
; CHECK-NEXT:    [[OR:%.*]] = or i32 %x, -253
; CHECK-NEXT:    [[REM:%.*]] = srem i32 [[OR]], -254
; CHECK-NEXT:    ret i32 [[REM]]
;
  %or = or i32 %x, -253
  %rem = srem i32 %or, -254
  ret i32 %rem
}

define i32 @not_srem_dividend_known_smaller_than_neg_divisor_set_bits(i32 %x) {
; CHECK-LABEL: @not_srem_dividend_known_smaller_than_neg_divisor_set_bits(
; CHECK-NEXT:    [[OR:%.*]] = or i32 %x, -253
; CHECK-NEXT:    [[REM:%.*]] = srem i32 [[OR]], -253
; CHECK-NEXT:    ret i32 [[REM]]
;
  %or = or i32 %x, -253
  %rem = srem i32 %or, -253
  ret i32 %rem
}


; RUN: opt -S -mtriple=x86_64-- \
; RUN:   -passes='require<libcall-lowering-info>,expand-ir-insts,simplifycfg,instcombine' \
; RUN:   < %s | FileCheck %s

; The double-rounding witness from issue 234750.
define i32 @ui129_issue_witness() {
; CHECK-LABEL: define i32 @ui129_issue_witness(
; CHECK:       ret i32 20467
  %r = uitofp i129 8136949792 to bfloat
  %bits = bitcast bfloat %r to i16
  %result = zext i16 %bits to i32
  ret i32 %result
}

define i32 @si129_negative_witness() {
; CHECK-LABEL: define i32 @si129_negative_witness(
; CHECK:       ret i32 53235
  %r = sitofp i129 -8136949792 to bfloat
  %bits = bitcast bfloat %r to i16
  %result = zext i16 %bits to i32
  ret i32 %result
}

; INT_MIN for signed i129.
define i32 @si129_min() {
; CHECK-LABEL: define i32 @si129_min(
; CHECK:       ret i32 65408
  %r = sitofp i129 -340282366920938463463374607431768211456 to bfloat
  %bits = bitcast bfloat %r to i16
  %result = zext i16 %bits to i32
  ret i32 %result
}

; 2^128 - 2^119 is the tie that rounds to infinity; one less is finite.
define i32 @ui129_bf16_overflow_tie() {
; CHECK-LABEL: define i32 @ui129_bf16_overflow_tie(
; CHECK:       ret i32 32640
  %r = uitofp i129 339617752923046005526922703901628039168 to bfloat
  %bits = bitcast bfloat %r to i16
  %result = zext i16 %bits to i32
  ret i32 %result
}

define i32 @ui129_bf16_max_finite() {
; CHECK-LABEL: define i32 @ui129_bf16_max_finite(
; CHECK:       ret i32 32639
  %r = uitofp i129 339617752923046005526922703901628039167 to bfloat
  %bits = bitcast bfloat %r to i16
  %result = zext i16 %bits to i32
  ret i32 %result
}

; The f32 intermediate overflows here; the final select saturates to infinity.
define i32 @ui129_f32_overflow() {
; CHECK-LABEL: define i32 @ui129_f32_overflow(
; CHECK:       ret i32 32640
  %r = uitofp i129 340282366920938463463374607431768211456 to bfloat
  %bits = bitcast bfloat %r to i16
  %result = zext i16 %bits to i32
  ret i32 %result
}

; The sticky bit makes the overflowing f32 intermediate a signaling NaN.
define i32 @ui129_f32_overflow_snan() {
; CHECK-LABEL: define i32 @ui129_f32_overflow_snan(
; CHECK:       ret i32 32640
  %r = uitofp i129 340282366920938463463374607431768211457 to bfloat
  %bits = bitcast bfloat %r to i16
  %result = zext i16 %bits to i32
  ret i32 %result
}

; These exercise the sticky bit when discarded bits are clear and set.
define i32 @ui129_sticky_clear() {
; CHECK-LABEL: define i32 @ui129_sticky_clear(
; CHECK:       ret i32 20096
  %r = uitofp i129 1073741824 to bfloat
  %bits = bitcast bfloat %r to i16
  %result = zext i16 %bits to i32
  ret i32 %result
}

define i32 @ui129_sticky_set() {
; CHECK-LABEL: define i32 @ui129_sticky_set(
; CHECK:       ret i32 20096
  %r = uitofp i129 1073741840 to bfloat
  %bits = bitcast bfloat %r to i16
  %result = zext i16 %bits to i32
  ret i32 %result
}

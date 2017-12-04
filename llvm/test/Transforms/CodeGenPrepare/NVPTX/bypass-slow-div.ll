; RUN: opt -S -codegenprepare < %s | FileCheck %s

target datalayout = "e-i64:64-v16:16-v32:32-n16:32:64"
target triple = "nvptx64-nvidia-cuda"

; We only use the div instruction -- the rem should be DCE'ed.
; CHECK-LABEL: @div_only
define void @div_only(i64 %a, i64 %b, i64* %retptr) {
  ; CHECK: udiv i32
  ; CHECK-NOT: urem
  ; CHECK: sdiv i64
  ; CHECK-NOT: rem
  %d = sdiv i64 %a, %b
  store i64 %d, i64* %retptr
  ret void
}

; We only use the rem instruction -- the div should be DCE'ed.
; CHECK-LABEL: @rem_only
define void @rem_only(i64 %a, i64 %b, i64* %retptr) {
  ; CHECK-NOT: div
  ; CHECK: urem i32
  ; CHECK-NOT: div
  ; CHECK: rem i64
  ; CHECK-NOT: div
  %d = srem i64 %a, %b
  store i64 %d, i64* %retptr
  ret void
}

; CHECK-LABEL: @udiv_by_constant(
define i64 @udiv_by_constant(i32 %a) {
; CHECK-NEXT:    [[A_ZEXT:%.*]] = zext i32 [[A:%.*]] to i64
; CHECK-NEXT:    [[TMP1:%.*]] = trunc i64 [[A_ZEXT]] to i32
; CHECK-NEXT:    [[TMP2:%.*]] = udiv i32 [[TMP1]], 50
; CHECK-NEXT:    [[TMP3:%.*]] = zext i32 [[TMP2]] to i64
; CHECK-NEXT:    ret i64 [[TMP3]]

  %a.zext = zext i32 %a to i64
  %wide.div = udiv i64 %a.zext, 50
  ret i64 %wide.div
}

; CHECK-LABEL: @urem_by_constant(
define i64 @urem_by_constant(i32 %a) {
; CHECK-NEXT:    [[A_ZEXT:%.*]] = zext i32 [[A:%.*]] to i64
; CHECK-NEXT:    [[TMP1:%.*]] = trunc i64 [[A_ZEXT]] to i32
; CHECK-NEXT:    [[TMP2:%.*]] = urem i32 [[TMP1]], 50
; CHECK-NEXT:    [[TMP3:%.*]] = zext i32 [[TMP2]] to i64
; CHECK-NEXT:    ret i64 [[TMP3]]

  %a.zext = zext i32 %a to i64
  %wide.div = urem i64 %a.zext, 50
  ret i64 %wide.div
}

; Negative test: instead of emitting a runtime check on %a, we prefer to let the
; DAGCombiner transform this division by constant into a multiplication (with a
; "magic constant").
;
; CHECK-LABEL: @udiv_by_constant_negative_0(
define i64 @udiv_by_constant_negative_0(i64 %a) {
; CHECK-NEXT:    [[WIDE_DIV:%.*]] = udiv i64 [[A:%.*]], 50
; CHECK-NEXT:    ret i64 [[WIDE_DIV]]

  %wide.div = udiv i64 %a, 50
  ret i64 %wide.div
}

; Negative test: while we know the dividend is short, the divisor isn't.  This
; test is here for completeness, but instcombine will optimize this to return 0.
;
; CHECK-LABEL: @udiv_by_constant_negative_1(
define i64 @udiv_by_constant_negative_1(i32 %a) {
; CHECK-NEXT:    [[A_ZEXT:%.*]] = zext i32 [[A:%.*]] to i64
; CHECK-NEXT:    [[WIDE_DIV:%.*]] = udiv i64 [[A_ZEXT]], 8589934592
; CHECK-NEXT:    ret i64 [[WIDE_DIV]]

  %a.zext = zext i32 %a to i64
  %wide.div = udiv i64 %a.zext, 8589934592 ;; == 1 << 33
  ret i64 %wide.div
}

; URem version of udiv_by_constant_negative_0
;
; CHECK-LABEL: @urem_by_constant_negative_0(
define i64 @urem_by_constant_negative_0(i64 %a) {
; CHECK-NEXT:    [[WIDE_DIV:%.*]] = urem i64 [[A:%.*]], 50
; CHECK-NEXT:    ret i64 [[WIDE_DIV]]

  %wide.div = urem i64 %a, 50
  ret i64 %wide.div
}

; URem version of udiv_by_constant_negative_1
;
; CHECK-LABEL: @urem_by_constant_negative_1(
define i64 @urem_by_constant_negative_1(i32 %a) {
; CHECK-NEXT:    [[A_ZEXT:%.*]] = zext i32 [[A:%.*]] to i64
; CHECK-NEXT:    [[WIDE_DIV:%.*]] = urem i64 [[A_ZEXT]], 8589934592
; CHECK-NEXT:    ret i64 [[WIDE_DIV]]

  %a.zext = zext i32 %a to i64
  %wide.div = urem i64 %a.zext, 8589934592 ;; == 1 << 33
  ret i64 %wide.div
}

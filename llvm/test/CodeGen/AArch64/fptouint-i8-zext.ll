; RUN: llc < %s | FileCheck %s

target datalayout = "e-m:e-i8:8:32-i16:16:32-i64:64-i128:128-n32:64-S128"
target triple = "aarch64"

; CHECK-LABEL: float_char_int_func:
; CHECK: fcvtzs [[A:w[0-9]+]], s0
; CHECK-NEXT: and w0, [[A]], #0xff
; CHECK-NEXT: ret
define i32 @float_char_int_func(float %infloatVal) {
entry:
  %conv = fptoui float %infloatVal to i8
  %conv1 = zext i8 %conv to i32
  ret i32 %conv1
}

; RUN: opt -S -codegenprepare < %s | FileCheck %s

target datalayout = "e-i64:64-v16:16-v32:32-n16:32:64"
target triple = "nvptx64-nvidia-cuda"

; When we bypass slow div with a constant numerator which fits into the bypass
; width, we still emit the bypass code, but we don't 'or' the numerator with
; the denominator.
; CHECK-LABEL: @small_constant_numer
define i64 @small_constant_numer(i64 %a) {
  ; CHECK: [[AND:%[0-9]+]] = and i64 %a, -4294967296
  ; CHECK: icmp eq i64 [[AND]], 0

  ; CHECK: [[TRUNC:%[0-9]+]] = trunc i64 %a to i32
  ; CHECK: udiv i32 -1, [[TRUNC]]
  %d = sdiv i64 4294967295, %a  ; 0xffff'ffff
  ret i64 %d
}

; When we try to bypass slow div with a constant numerator which *doesn't* fit
; into the bypass width, leave it as a plain 64-bit div with no bypass.
; CHECK-LABEL: @large_constant_numer
define i64 @large_constant_numer(i64 %a) {
  ; CHECK-NOT: udiv i32
  %d = sdiv i64 4294967296, %a  ; 0x1'0000'0000
  ret i64 %d
}

; For good measure, try a value larger than 2^32.
; CHECK-LABEL: @larger_constant_numer
define i64 @larger_constant_numer(i64 %a) {
  ; CHECK-NOT: udiv i32
  %d = sdiv i64 5000000000, %a
  ret i64 %d
}

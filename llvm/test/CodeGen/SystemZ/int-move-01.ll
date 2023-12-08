; Test moves between GPRs.
;
; RUN: llc < %s -mtriple=s390x-linux-gnu | FileCheck %s

; Test 8-bit moves, which should get promoted to i32.
define i8 @f1(i8 %a, i8 %b) {
; CHECK-LABEL: f1:
; CHECK: lr %r2, %r3
; CHECK: br %r14
  ret i8 %b
}

; Test 16-bit moves, which again should get promoted to i32.
define i16 @f2(i16 %a, i16 %b) {
; CHECK-LABEL: f2:
; CHECK: lr %r2, %r3
; CHECK: br %r14
  ret i16 %b
}

; Test 32-bit moves.
define i32 @f3(i32 %a, i32 %b) {
; CHECK-LABEL: f3:
; CHECK: lr %r2, %r3
; CHECK: br %r14
  ret i32 %b
}

; Test 64-bit moves.
define i64 @f4(i64 %a, i64 %b) {
; CHECK-LABEL: f4:
; CHECK: lgr %r2, %r3
; CHECK: br %r14
  ret i64 %b
}

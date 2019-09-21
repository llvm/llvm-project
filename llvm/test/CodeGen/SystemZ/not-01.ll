; Combined logical operations involving complement on z15
;
; RUN: llc -mcpu=z15 < %s -mtriple=s390x-linux-gnu | FileCheck %s

; And-with-complement 32-bit.
define i32 @f1(i32 %dummy, i32 %a, i32 %b) {
; CHECK-LABEL: f1:
; CHECK: ncrk %r2, %r3, %r4
; CHECK: br %r14
  %neg = xor i32 %b, -1
  %ret = and i32 %neg, %a
  ret i32 %ret
}

; And-with-complement 64-bit.
define i64 @f2(i64 %dummy, i64 %a, i64 %b) {
; CHECK-LABEL: f2:
; CHECK: ncgrk %r2, %r3, %r4
; CHECK: br %r14
  %neg = xor i64 %b, -1
  %ret = and i64 %neg, %a
  ret i64 %ret
}

; Or-with-complement 32-bit.
define i32 @f3(i32 %dummy, i32 %a, i32 %b) {
; CHECK-LABEL: f3:
; CHECK: ocrk %r2, %r3, %r4
; CHECK: br %r14
  %neg = xor i32 %b, -1
  %ret = or i32 %neg, %a
  ret i32 %ret
}

; Or-with-complement 64-bit.
define i64 @f4(i64 %dummy, i64 %a, i64 %b) {
; CHECK-LABEL: f4:
; CHECK: ocgrk %r2, %r3, %r4
; CHECK: br %r14
  %neg = xor i64 %b, -1
  %ret = or i64 %neg, %a
  ret i64 %ret
}

; NAND 32-bit.
define i32 @f5(i32 %dummy, i32 %a, i32 %b) {
; CHECK-LABEL: f5:
; CHECK: nnrk %r2, %r3, %r4
; CHECK: br %r14
  %tmp = and i32 %a, %b
  %ret = xor i32 %tmp, -1
  ret i32 %ret
}

; NAND 64-bit.
define i64 @f6(i64 %dummy, i64 %a, i64 %b) {
; CHECK-LABEL: f6:
; CHECK: nngrk %r2, %r3, %r4
; CHECK: br %r14
  %tmp = and i64 %a, %b
  %ret = xor i64 %tmp, -1
  ret i64 %ret
}

; NOR 32-bit.
define i32 @f7(i32 %dummy, i32 %a, i32 %b) {
; CHECK-LABEL: f7:
; CHECK: nork %r2, %r3, %r4
; CHECK: br %r14
  %tmp = or i32 %a, %b
  %ret = xor i32 %tmp, -1
  ret i32 %ret
}

; NOR 64-bit.
define i64 @f8(i64 %dummy, i64 %a, i64 %b) {
; CHECK-LABEL: f8:
; CHECK: nogrk %r2, %r3, %r4
; CHECK: br %r14
  %tmp = or i64 %a, %b
  %ret = xor i64 %tmp, -1
  ret i64 %ret
}

; NXOR 32-bit.
define i32 @f9(i32 %dummy, i32 %a, i32 %b) {
; CHECK-LABEL: f9:
; CHECK: nxrk %r2, %r3, %r4
; CHECK: br %r14
  %tmp = xor i32 %a, %b
  %ret = xor i32 %tmp, -1
  ret i32 %ret
}

; NXOR 64-bit.
define i64 @f10(i64 %dummy, i64 %a, i64 %b) {
; CHECK-LABEL: f10:
; CHECK: nxgrk %r2, %r3, %r4
; CHECK: br %r14
  %tmp = xor i64 %a, %b
  %ret = xor i64 %tmp, -1
  ret i64 %ret
}

; Or-with-complement 32-bit of a constant.
define i32 @f11(i32 %a) {
; CHECK-LABEL: f11:
; CHECK: lhi [[REG:%r[0-5]]], -256
; CHECK: ocrk %r2, [[REG]], %r2
; CHECK: br %r14
  %neg = xor i32 %a, -1
  %ret = or i32 %neg, -256
  ret i32 %ret
}

; Or-with-complement 64-bit of a constant.
define i64 @f12(i64 %a) {
; CHECK-LABEL: f12:
; CHECK: lghi [[REG:%r[0-5]]], -256
; CHECK: ocgrk %r2, [[REG]], %r2
; CHECK: br %r14
  %neg = xor i64 %a, -1
  %ret = or i64 %neg, -256
  ret i64 %ret
}


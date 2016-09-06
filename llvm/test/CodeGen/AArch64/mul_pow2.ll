; RUN: llc < %s -mtriple=aarch64-eabi | FileCheck %s

; Convert mul x, pow2 to shift.
; Convert mul x, pow2 +/- 1 to shift + add/sub.

define i32 @test2(i32 %x) {
; CHECK-LABEL: test2
; CHECK: lsl w0, w0, #1

  %mul = shl nsw i32 %x, 1
  ret i32 %mul
}

define i32 @test3(i32 %x) {
; CHECK-LABEL: test3
; CHECK: add w0, w0, w0, lsl #1

  %mul = mul nsw i32 %x, 3
  ret i32 %mul
}

define i32 @test4(i32 %x) {
; CHECK-LABEL: test4
; CHECK: lsl w0, w0, #2

  %mul = shl nsw i32 %x, 2
  ret i32 %mul
}

define i32 @test5(i32 %x) {
; CHECK-LABEL: test5
; CHECK: add w0, w0, w0, lsl #2


  %mul = mul nsw i32 %x, 5
  ret i32 %mul
}

define i32 @test7(i32 %x) {
; CHECK-LABEL: test7
; CHECK: lsl {{w[0-9]+}}, w0, #3
; CHECK: sub w0, {{w[0-9]+}}, w0

  %mul = mul nsw i32 %x, 7
  ret i32 %mul
}

define i32 @test8(i32 %x) {
; CHECK-LABEL: test8
; CHECK: lsl w0, w0, #3

  %mul = shl nsw i32 %x, 3
  ret i32 %mul
}

define i32 @test9(i32 %x) {
; CHECK-LABEL: test9
; CHECK: add w0, w0, w0, lsl #3

  %mul = mul nsw i32 %x, 9
  ret i32 %mul
}

; Convert mul x, -pow2 to shift.
; Convert mul x, -(pow2 +/- 1) to shift + add/sub.

define i32 @ntest2(i32 %x) {
; CHECK-LABEL: ntest2
; CHECK: neg w0, w0, lsl #1

  %mul = mul nsw i32 %x, -2
  ret i32 %mul
}

define i32 @ntest3(i32 %x) {
; CHECK-LABEL: ntest3
; CHECK: sub w0, w0, w0, lsl #2

  %mul = mul nsw i32 %x, -3
  ret i32 %mul
}

define i32 @ntest4(i32 %x) {
; CHECK-LABEL: ntest4
; CHECK:neg w0, w0, lsl #2

  %mul = mul nsw i32 %x, -4
  ret i32 %mul
}

define i32 @ntest5(i32 %x) {
; CHECK-LABEL: ntest5
; CHECK: add {{w[0-9]+}}, w0, w0, lsl #2
; CHECK: neg w0, {{w[0-9]+}}
  %mul = mul nsw i32 %x, -5
  ret i32 %mul
}

define i32 @ntest7(i32 %x) {
; CHECK-LABEL: ntest7
; CHECK: sub w0, w0, w0, lsl #3

  %mul = mul nsw i32 %x, -7
  ret i32 %mul
}

define i32 @ntest8(i32 %x) {
; CHECK-LABEL: ntest8
; CHECK: neg w0, w0, lsl #3

  %mul = mul nsw i32 %x, -8
  ret i32 %mul
}

define i32 @ntest9(i32 %x) {
; CHECK-LABEL: ntest9
; CHECK: add {{w[0-9]+}}, w0, w0, lsl #3
; CHECK: neg w0, {{w[0-9]+}}

  %mul = mul nsw i32 %x, -9
  ret i32 %mul
}

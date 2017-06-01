; RUN: llc < %s -mtriple=aarch64-eabi | FileCheck %s

; Convert mul x, pow2 to shift.
; Convert mul x, pow2 +/- 1 to shift + add/sub.
; Convert mul x, (pow2 + 1) * pow2 to shift + add + shift.
; Lowering other positive constants are not supported yet.

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

define i32 @test6_32b(i32 %x) {
; CHECK-LABEL: test6
; CHECK: add {{w[0-9]+}}, w0, w0, lsl #1
; CHECK: lsl w0, {{w[0-9]+}}, #1

  %mul = mul nsw i32 %x, 6 
  ret i32 %mul
}

define i64 @test6_64b(i64 %x) {
; CHECK-LABEL: test6_64b
; CHECK: add {{x[0-9]+}}, x0, x0, lsl #1
; CHECK: lsl x0, {{x[0-9]+}}, #1

  %mul = mul nsw i64 %x, 6 
  ret i64 %mul
}

; mul that appears together with add, sub, s(z)ext is not supported to be 
; converted to the combination of lsl, add/sub yet.
define i64 @test6_umull(i32 %x) {
; CHECK-LABEL: test6_umull
; CHECK: umull x0, w0, {{w[0-9]+}} 

  %ext = zext i32 %x to i64
  %mul = mul nsw i64 %ext, 6 
  ret i64 %mul
}

define i64 @test6_smull(i32 %x) {
; CHECK-LABEL: test6_smull
; CHECK: smull x0, w0, {{w[0-9]+}} 

  %ext = sext i32 %x to i64
  %mul = mul nsw i64 %ext, 6 
  ret i64 %mul
}

define i32 @test6_madd(i32 %x, i32 %y) {
; CHECK-LABEL: test6_madd
; CHECK: madd w0, w0, {{w[0-9]+}}, w1 

  %mul = mul nsw i32 %x, 6 
  %add = add i32 %mul, %y
  ret i32 %add
}

define i32 @test6_msub(i32 %x, i32 %y) {
; CHECK-LABEL: test6_msub
; CHECK: msub w0, w0, {{w[0-9]+}}, w1 

  %mul = mul nsw i32 %x, 6 
  %sub = sub i32 %y, %mul
  ret i32 %sub
}

define i64 @test6_umaddl(i32 %x, i64 %y) {
; CHECK-LABEL: test6_umaddl
; CHECK: umaddl x0, w0, {{w[0-9]+}}, x1 

  %ext = zext i32 %x to i64
  %mul = mul nsw i64 %ext, 6 
  %add = add i64 %mul, %y
  ret i64 %add
}

define i64 @test6_smaddl(i32 %x, i64 %y) {
; CHECK-LABEL: test6_smaddl
; CHECK: smaddl x0, w0, {{w[0-9]+}}, x1

  %ext = sext i32 %x to i64
  %mul = mul nsw i64 %ext, 6 
  %add = add i64 %mul, %y
  ret i64 %add
}

define i64 @test6_umsubl(i32 %x, i64 %y) {
; CHECK-LABEL: test6_umsubl
; CHECK: umsubl x0, w0, {{w[0-9]+}}, x1

  %ext = zext i32 %x to i64
  %mul = mul nsw i64 %ext, 6 
  %sub = sub i64 %y, %mul
  ret i64 %sub
}

define i64 @test6_smsubl(i32 %x, i64 %y) {
; CHECK-LABEL: test6_smsubl
; CHECK: smsubl x0, w0, {{w[0-9]+}}, x1 

  %ext = sext i32 %x to i64
  %mul = mul nsw i64 %ext, 6 
  %sub = sub i64 %y, %mul
  ret i64 %sub
}

define i64 @test6_umnegl(i32 %x) {
; CHECK-LABEL: test6_umnegl
; CHECK: umnegl x0, w0, {{w[0-9]+}} 

  %ext = zext i32 %x to i64
  %mul = mul nsw i64 %ext, 6 
  %sub = sub i64 0, %mul
  ret i64 %sub
}

define i64 @test6_smnegl(i32 %x) {
; CHECK-LABEL: test6_smnegl
; CHECK: smnegl x0, w0, {{w[0-9]+}} 

  %ext = sext i32 %x to i64
  %mul = mul nsw i64 %ext, 6 
  %sub = sub i64 0, %mul
  ret i64 %sub
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

define i32 @test10(i32 %x) {
; CHECK-LABEL: test10
; CHECK: add {{w[0-9]+}}, w0, w0, lsl #2
; CHECK: lsl w0, {{w[0-9]+}}, #1

  %mul = mul nsw i32 %x, 10
  ret i32 %mul
}

define i32 @test11(i32 %x) {
; CHECK-LABEL: test11
; CHECK: mul w0, w0, {{w[0-9]+}}

  %mul = mul nsw i32 %x, 11
  ret i32 %mul
}

define i32 @test12(i32 %x) {
; CHECK-LABEL: test12
; CHECK: add {{w[0-9]+}}, w0, w0, lsl #1
; CHECK: lsl w0, {{w[0-9]+}}, #2

  %mul = mul nsw i32 %x, 12
  ret i32 %mul
}

define i32 @test13(i32 %x) {
; CHECK-LABEL: test13
; CHECK: mul w0, w0, {{w[0-9]+}}

  %mul = mul nsw i32 %x, 13
  ret i32 %mul
}

define i32 @test14(i32 %x) {
; CHECK-LABEL: test14
; CHECK: mul w0, w0, {{w[0-9]+}}

  %mul = mul nsw i32 %x, 14 
  ret i32 %mul
}

define i32 @test15(i32 %x) {
; CHECK-LABEL: test15
; CHECK: lsl {{w[0-9]+}}, w0, #4
; CHECK: sub w0, {{w[0-9]+}}, w0

  %mul = mul nsw i32 %x, 15
  ret i32 %mul
}

define i32 @test16(i32 %x) {
; CHECK-LABEL: test16
; CHECK: lsl w0, w0, #4

  %mul = mul nsw i32 %x, 16
  ret i32 %mul
}

; Convert mul x, -pow2 to shift.
; Convert mul x, -(pow2 +/- 1) to shift + add/sub.
; Lowering other negative constants are not supported yet.

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

define i32 @ntest6(i32 %x) {
; CHECK-LABEL: ntest6
; CHECK: mul w0, w0, {{w[0-9]+}}

  %mul = mul nsw i32 %x, -6
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

define i32 @ntest10(i32 %x) {
; CHECK-LABEL: ntest10
; CHECK: mul w0, w0, {{w[0-9]+}}

  %mul = mul nsw i32 %x, -10
  ret i32 %mul
}

define i32 @ntest11(i32 %x) {
; CHECK-LABEL: ntest11
; CHECK: mul w0, w0, {{w[0-9]+}}

  %mul = mul nsw i32 %x, -11
  ret i32 %mul
}

define i32 @ntest12(i32 %x) {
; CHECK-LABEL: ntest12
; CHECK: mul w0, w0, {{w[0-9]+}}

  %mul = mul nsw i32 %x, -12
  ret i32 %mul
}

define i32 @ntest13(i32 %x) {
; CHECK-LABEL: ntest13
; CHECK: mul w0, w0, {{w[0-9]+}}
  %mul = mul nsw i32 %x, -13
  ret i32 %mul
}

define i32 @ntest14(i32 %x) {
; CHECK-LABEL: ntest14
; CHECK: mul w0, w0, {{w[0-9]+}}

  %mul = mul nsw i32 %x, -14
  ret i32 %mul
}

define i32 @ntest15(i32 %x) {
; CHECK-LABEL: ntest15
; CHECK: sub w0, w0, w0, lsl #4

  %mul = mul nsw i32 %x, -15
  ret i32 %mul
}

define i32 @ntest16(i32 %x) {
; CHECK-LABEL: ntest16
; CHECK: neg w0, w0, lsl #4

  %mul = mul nsw i32 %x, -16
  ret i32 %mul
}

; RUN: llc -verify-machineinstrs -mtriple=aarch64-apple-ios -o - %s | FileCheck %s
; RUN: llc -O0 -fast-isel -verify-machineinstrs -mtriple=aarch64-apple-ios -o - %s | FileCheck %s --check-prefix=CHECK-O0
; RUN: llc -verify-machineinstrs -mtriple=arm64_32-apple-ios -o - %s | FileCheck %s
; RUN: llc -O0 -fast-isel -verify-machineinstrs -mtriple=arm64_32-apple-ios -o - %s | FileCheck %s --check-prefix=CHECK-O0

; CHECK-LABEL: test1
; CHECK: bl      _gen
; CHECK: sxth    [[TMP:w.*]], w0
; CHECK: add     w0, [[TMP]], w1, sxtb
; CHECK-O0-LABEL: test1
; CHECK-O0: bl      _gen
; CHECK-O0: sxth    [[TMP:w.*]], w0
; CHECK-O0: add     {{w[0-9]+}}, [[TMP]], w1, sxtb
define i16 @test1(i32) {
entry:
  %call = call swiftcc { i16, i8 } @gen(i32 %0)
  %v3 = extractvalue { i16, i8 } %call, 0
  %v1 = sext i16 %v3 to i32
  %v5 = extractvalue { i16, i8 } %call, 1
  %v2 = sext i8 %v5 to i32
  %add = add nsw i32 %v1, %v2
  %conv = trunc i32 %add to i16
  ret i16 %conv
}

declare swiftcc { i16, i8 } @gen(i32)

; CHECK-LABEL: test2
; CHECK:  bl      _gen2
; CHECK:  add     [[TMP:x.*]], x0, x1
; CHECK:  add     [[TMP2:x.*]], x2, x3
; CHECK:  add     [[TMP]], [[TMP]], [[TMP2]]
; CHECK:  add     x0, [[TMP]], x4
; CHECK-O0-LABEL: test2
; CHECK-O0:  bl      _gen2
; CHECK-O0:  add     [[TMP:x.*]], x0, x1
; CHECK-O0:  add     [[TMP]], [[TMP]], x2
; CHECK-O0:  add     [[TMP]], [[TMP]], x3
; CHECK-O0:  add     x0, [[TMP]], x4

define i64 @test2(i64 %key) {
entry:
  %key.addr = alloca i64, align 4
  store i64 %key, ptr %key.addr, align 4
  %0 = load i64, ptr %key.addr, align 4
  %call = call swiftcc { i64, i64, i64, i64, i64 } @gen2(i64 %0)

  %v3 = extractvalue { i64, i64, i64, i64, i64 } %call, 0
  %v5 = extractvalue { i64, i64, i64, i64, i64 } %call, 1
  %v6 = extractvalue { i64, i64, i64, i64, i64 } %call, 2
  %v7 = extractvalue { i64, i64, i64, i64, i64 } %call, 3
  %v8 = extractvalue { i64, i64, i64, i64, i64 } %call, 4

  %add = add nsw i64 %v3, %v5
  %add1 = add nsw i64 %add, %v6
  %add2 = add nsw i64 %add1, %v7
  %add3 = add nsw i64 %add2, %v8
  ret i64 %add3
}
; CHECK-LABEL: gen2:
; CHECK:  mov      x1, x0
; CHECK:  mov      x2, x0
; CHECK:  mov      x3, x0
; CHECK:  mov      x4, x0
; CHECK:  ret
define swiftcc { i64, i64, i64, i64, i64 } @gen2(i64 %key) {
  %Y = insertvalue { i64, i64, i64, i64, i64 } undef, i64 %key, 0
  %Z = insertvalue { i64, i64, i64, i64, i64 } %Y, i64 %key, 1
  %Z2 = insertvalue { i64, i64, i64, i64, i64 } %Z, i64 %key, 2
  %Z3 = insertvalue { i64, i64, i64, i64, i64 } %Z2, i64 %key, 3
  %Z4 = insertvalue { i64, i64, i64, i64, i64 } %Z3, i64 %key, 4
  ret { i64, i64, i64, i64, i64 } %Z4
}

; CHECK-LABEL: test3
; CHECK: bl      _gen3
; CHECK: add             [[TMP:w.*]], w0, w1
; CHECK: add             [[TMP2:w.*]], w2, w3
; CHECK: add             w0, [[TMP]], [[TMP2]]
; CHECK-O0-LABEL: test3
; CHECK-O0: bl      _gen3
; CHECK-O0: add             [[TMP:w.*]], w0, w1
; CHECK-O0: add             [[TMP]], [[TMP]], w2
; CHECK-O0: add             w0, [[TMP]], w3
define i32 @test3(i32) {
entry:
  %call = call swiftcc { i32, i32, i32, i32 } @gen3(i32 %0)

  %v3 = extractvalue { i32, i32, i32, i32 } %call, 0
  %v5 = extractvalue { i32, i32, i32, i32 } %call, 1
  %v6 = extractvalue { i32, i32, i32, i32 } %call, 2
  %v7 = extractvalue { i32, i32, i32, i32 } %call, 3

  %add = add nsw i32 %v3, %v5
  %add1 = add nsw i32 %add, %v6
  %add2 = add nsw i32 %add1, %v7
  ret i32 %add2
}

declare swiftcc { i32, i32, i32, i32 } @gen3(i32 %key)

; CHECK-LABEL: test4
; CHECK: bl      _gen4
; CHECK: fadd    s0, s0, s1
; CHECK: fadd    s0, s0, s2
; CHECK: fadd    s0, s0, s3
; CHECK-O0-LABEL: test4
; CHECK-O0: bl      _gen4
; CHECK-O0: fadd    s0, s0, s1
; CHECK-O0: fadd    s0, s0, s2
; CHECK-O0: fadd    s0, s0, s3
define float @test4(float) {
entry:
  %call = call swiftcc { float, float, float, float } @gen4(float %0)

  %v3 = extractvalue { float, float, float, float } %call, 0
  %v5 = extractvalue { float, float, float, float } %call, 1
  %v6 = extractvalue { float, float, float, float } %call, 2
  %v7 = extractvalue { float, float, float, float } %call, 3

  %add = fadd float %v3, %v5
  %add1 = fadd float %add, %v6
  %add2 = fadd float %add1, %v7
  ret float %add2
}

declare swiftcc { float, float, float, float } @gen4(float %key)

; CHECK-LABEL: test5
; CHECK:  bl      _gen5
; CHECK:  fadd    d0, d0, d1
; CHECK:  fadd    d0, d0, d2
; CHECK:  fadd    d0, d0, d3
; CHECK-O0-LABEL: test5
; CHECK-O0:  bl      _gen5
; CHECK-O0:  fadd    d0, d0, d1
; CHECK-O0:  fadd    d0, d0, d2
; CHECK-O0:  fadd    d0, d0, d3
define swiftcc double @test5(){
entry:
  %call = call swiftcc { double, double, double, double } @gen5()

  %v3 = extractvalue { double, double, double, double } %call, 0
  %v5 = extractvalue { double, double, double, double } %call, 1
  %v6 = extractvalue { double, double, double, double } %call, 2
  %v7 = extractvalue { double, double, double, double } %call, 3

  %add = fadd double %v3, %v5
  %add1 = fadd double %add, %v6
  %add2 = fadd double %add1, %v7
  ret double %add2
}

declare swiftcc { double, double, double, double } @gen5()

; CHECK-LABEL: test6
; CHECK:   bl      _gen6
; CHECK-DAG:   fadd    d0, d0, d1
; CHECK-DAG:   fadd    d0, d0, d2
; CHECK-DAG:   fadd    d0, d0, d3
; CHECK-DAG:   add     [[TMP:w.*]], w0, w1
; CHECK-DAG:   add     [[TMP2:w.*]], w2, w3
; CHECK-DAG:   add     w0, [[TMP]], [[TMP2]]
; CHECK-O0-LABEL: test6
; CHECK-O0:   bl      _gen6
; CHECK-O0-DAG:   fadd    d0, d0, d1
; CHECK-O0-DAG:   fadd    d0, d0, d2
; CHECK-O0-DAG:   fadd    d0, d0, d3
; CHECK-O0-DAG:   add     [[TMP:w.*]], w0, w1
; CHECK-O0-DAG:   add     [[TMP]], [[TMP]], w2
; CHECK-O0-DAG:   add     w0, [[TMP]], w3
define swiftcc { double, i32 } @test6() {
entry:
  %call = call swiftcc { double, double, double, double, i32, i32, i32, i32 } @gen6()

  %v3 = extractvalue { double, double, double, double, i32, i32, i32, i32 } %call, 0
  %v5 = extractvalue { double, double, double, double, i32, i32, i32, i32 } %call, 1
  %v6 = extractvalue { double, double, double, double, i32, i32, i32, i32 } %call, 2
  %v7 = extractvalue { double, double, double, double, i32, i32, i32, i32 } %call, 3
  %v3.i = extractvalue { double, double, double, double, i32, i32, i32, i32 } %call, 4
  %v5.i = extractvalue { double, double, double, double, i32, i32, i32, i32 } %call, 5
  %v6.i = extractvalue { double, double, double, double, i32, i32, i32, i32 } %call, 6
  %v7.i = extractvalue { double, double, double, double, i32, i32, i32, i32 } %call, 7

  %add = fadd double %v3, %v5
  %add1 = fadd double %add, %v6
  %add2 = fadd double %add1, %v7

  %add.i = add nsw i32 %v3.i, %v5.i
  %add1.i = add nsw i32 %add.i, %v6.i
  %add2.i = add nsw i32 %add1.i, %v7.i

  %Y = insertvalue { double, i32 } undef, double %add2, 0
  %Z = insertvalue { double, i32 } %Y, i32 %add2.i, 1
  ret { double, i32} %Z
}

declare swiftcc { double, double, double, double, i32, i32, i32, i32 } @gen6()

; CHECK-LABEL: _gen7
; CHECK-DAG:   mov      w1, w0
; CHECK-DAG:   mov      w2, w0
; CHECK-DAG:   mov      w3, w0
; CHECK:   ret
; CHECK-O0-LABEL: _gen7
; CHECK-O0:  mov w3, w0
; CHECK-O0:  mov w0, w3
; CHECK-O0:  mov w1, w3
; CHECK-O0:  mov w2, w3
define swiftcc { i32, i32, i32, i32 } @gen7(i32 %key) {
  %v0 = insertvalue { i32, i32, i32, i32 } undef, i32 %key, 0
  %v1 = insertvalue { i32, i32, i32, i32 } %v0, i32 %key, 1
  %v2 = insertvalue { i32, i32, i32, i32 } %v1, i32 %key, 2
  %v3 = insertvalue { i32, i32, i32, i32 } %v2, i32 %key, 3
  ret { i32, i32, i32, i32 } %v3
}

; CHECK-LABEL: _gen9
; CHECK:  mov      w1, w0
; CHECK:  mov      w2, w0
; CHECK:  mov      w3, w0
; CHECK:  ret
; CHECK-O0-LABEL: _gen9
; CHECK-O0:  mov w3, w0
; CHECK-O0:  mov w0, w3
; CHECK-O0:  mov w1, w3
; CHECK-O0:  mov w2, w3
define swiftcc { i8, i8, i8, i8 } @gen9(i8 %key) {
  %v0 = insertvalue { i8, i8, i8, i8 } undef, i8 %key, 0
  %v1 = insertvalue { i8, i8, i8, i8 } %v0, i8 %key, 1
  %v2 = insertvalue { i8, i8, i8, i8 } %v1, i8 %key, 2
  %v3 = insertvalue { i8, i8, i8, i8 } %v2, i8 %key, 3
  ret { i8, i8, i8, i8 } %v3
}

; CHECK-LABEL: _gen10
; CHECK:  fmov         d1, d0
; CHECK:  fmov         d2, d0
; CHECK:  fmov         d3, d0
; CHECK:  mov      w1, w0
; CHECK:  mov      w2, w0
; CHECK:  mov      w3, w0
; CHECK:  ret
define swiftcc { double, double, double, double, i32, i32, i32, i32 } @gen10(double %keyd, i32 %keyi) {
  %v0 = insertvalue { double, double, double, double, i32, i32, i32, i32 } undef, double %keyd, 0
  %v1 = insertvalue { double, double, double, double, i32, i32, i32, i32 } %v0, double %keyd, 1
  %v2 = insertvalue { double, double, double, double, i32, i32, i32, i32 } %v1, double %keyd, 2
  %v3 = insertvalue { double, double, double, double, i32, i32, i32, i32 } %v2, double %keyd, 3
  %v4 = insertvalue { double, double, double, double, i32, i32, i32, i32 } %v3, i32 %keyi, 4
  %v5 = insertvalue { double, double, double, double, i32, i32, i32, i32 } %v4, i32 %keyi, 5
  %v6 = insertvalue { double, double, double, double, i32, i32, i32, i32 } %v5, i32 %keyi, 6
  %v7 = insertvalue { double, double, double, double, i32, i32, i32, i32 } %v6, i32 %keyi, 7
  ret { double, double, double, double, i32, i32, i32, i32 } %v7
}

; CHECK-LABEL: _test11
; CHECK:  bl      _gen11
; CHECK:  fadd.4s v0, v0, v1
; CHECK:  fadd.4s v0, v0, v2
; CHECK:  fadd.4s v0, v0, v3
define swiftcc <4 x float> @test11() {
entry:
  %call = call swiftcc { <4 x float>, <4 x float>, <4 x float>, <4 x float> } @gen11()

  %v3 = extractvalue { <4 x float>, <4 x float>, <4 x float>, <4 x float> } %call, 0
  %v5 = extractvalue { <4 x float>, <4 x float>, <4 x float>, <4 x float> } %call, 1
  %v6 = extractvalue { <4 x float>, <4 x float>, <4 x float>, <4 x float> } %call, 2
  %v7 = extractvalue { <4 x float>, <4 x float>, <4 x float>, <4 x float> } %call, 3

  %add = fadd <4 x float> %v3, %v5
  %add1 = fadd <4 x float> %add, %v6
  %add2 = fadd <4 x float> %add1, %v7
  ret <4 x float> %add2
}

declare swiftcc { <4 x float>, <4 x float>, <4 x float>, <4 x float> } @gen11()

; CHECK-LABEL: _test12
; CHECK:  fadd.4s v0, v0, v1
; CHECK:  fmov s1, s3
; CHECK:  fadd.4s v0, v0, v2
define swiftcc { <4 x float>, float } @test12() #0 {
entry:
  %call = call swiftcc { <4 x float>, <4 x float>, <4 x float>, float } @gen12()

  %v3 = extractvalue { <4 x float>, <4 x float>, <4 x float>, float } %call, 0
  %v5 = extractvalue { <4 x float>, <4 x float>, <4 x float>, float } %call, 1
  %v6 = extractvalue { <4 x float>, <4 x float>, <4 x float>, float } %call, 2
  %v8 = extractvalue { <4 x float>, <4 x float>, <4 x float>, float } %call, 3

  %add = fadd <4 x float> %v3, %v5
  %add1 = fadd <4 x float> %add, %v6
  %res.0 = insertvalue { <4 x float>, float } undef, <4 x float> %add1, 0
  %res = insertvalue { <4 x float>, float } %res.0, float %v8, 1
  ret { <4 x float>, float } %res
}

declare swiftcc { <4 x float>, <4 x float>, <4 x float>, float } @gen12()

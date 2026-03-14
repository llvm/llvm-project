; RUN: llc -mtriple=armv7k-apple-ios8.0 -mcpu=cortex-a7 -verify-machineinstrs < %s | FileCheck %s
; RUN: llc -mtriple=armv7k-apple-ios8.0 -mcpu=cortex-a7 -verify-machineinstrs < %s -O0 | FileCheck --check-prefix=CHECK-O0 %s

; RUN: llc -mtriple=armv7-apple-ios -verify-machineinstrs < %s | FileCheck %s
; RUN: llc -mtriple=armv7-apple-ios -verify-machineinstrs < %s -O0 | FileCheck --check-prefix=CHECK-O0 %s

; Test how llvm handles return type of {i16, i8}. The return value will be
; passed in %r0 and %r1.
; CHECK-LABEL: test:
; CHECK: bl {{.*}}gen
; CHECK: sxth {{.*}}, r0
; CHECK: sxtab r0, {{.*}}, r1
; CHECK-O0-LABEL: test:
; CHECK-O0: bl {{.*}}gen
; CHECK-O0: sxth r0, r0
; CHECK-O0: sxtb r1, r1
; CHECK-O0: add r0, r0, r1
define i16 @test(i32 %key) {
entry:
  %key.addr = alloca i32, align 4
  store i32 %key, ptr %key.addr, align 4
  %0 = load i32, ptr %key.addr, align 4
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

; We can't pass every return value in register, instead, pass everything in
; memroy.
; The caller provides space for the return value and passes the address in %r0.
; The first input argument will be in %r1.
; CHECK-LABEL: test2:
; CHECK: mov r1, r0
; CHECK: mov r0, sp
; CHECK: bl {{.*}}gen2
; CHECK-DAG: add
; CHECK-DAG: ldr {{.*}}, [sp, #16]
; CHECK-DAG: add
; CHECK-DAG: add
; CHECK-DAG: add
; CHECK-O0-LABEL: test2:
; CHECK-O0: str r0
; CHECK-O0: mov r0, sp
; CHECK-O0: bl {{.*}}gen2
; CHECK-O0-DAG: ldr {{.*}}, [sp]
; CHECK-O0-DAG: ldr {{.*}}, [sp, #4]
; CHECK-O0-DAG: ldr {{.*}}, [sp, #8]
; CHECK-O0-DAG: ldr {{.*}}, [sp, #12]
; CHECK-O0-DAG: ldr {{.*}}, [sp, #16]
; CHECK-O0-DAG: add
; CHECK-O0-DAG: add
; CHECK-O0-DAG: add
; CHECK-O0-DAG: add
define i32 @test2(i32 %key) #0 {
entry:
  %key.addr = alloca i32, align 4
  store i32 %key, ptr %key.addr, align 4
  %0 = load i32, ptr %key.addr, align 4
  %call = call swiftcc { i32, i32, i32, i32, i32 } @gen2(i32 %0)

  %v3 = extractvalue { i32, i32, i32, i32, i32 } %call, 0
  %v5 = extractvalue { i32, i32, i32, i32, i32 } %call, 1
  %v6 = extractvalue { i32, i32, i32, i32, i32 } %call, 2
  %v7 = extractvalue { i32, i32, i32, i32, i32 } %call, 3
  %v8 = extractvalue { i32, i32, i32, i32, i32 } %call, 4

  %add = add nsw i32 %v3, %v5
  %add1 = add nsw i32 %add, %v6
  %add2 = add nsw i32 %add1, %v7
  %add3 = add nsw i32 %add2, %v8
  ret i32 %add3
}

; The address of the return value is passed in %r0.
; CHECK-LABEL: gen2:
; CHECK-DAG: str r1, [r0]
; CHECK-DAG: str r1, [r0, #4]
; CHECK-DAG: str r1, [r0, #8]
; CHECK-DAG: str r1, [r0, #12]
; CHECK-DAG: str r1, [r0, #16]
; CHECK-O0-LABEL: gen2:
; CHECK-O0-DAG: str r1, [r0]
; CHECK-O0-DAG: str r1, [r0, #4]
; CHECK-O0-DAG: str r1, [r0, #8]
; CHECK-O0-DAG: str r1, [r0, #12]
; CHECK-O0-DAG: str r1, [r0, #16]
define swiftcc { i32, i32, i32, i32, i32 } @gen2(i32 %key) {
  %Y = insertvalue { i32, i32, i32, i32, i32 } undef, i32 %key, 0
  %Z = insertvalue { i32, i32, i32, i32, i32 } %Y, i32 %key, 1
  %Z2 = insertvalue { i32, i32, i32, i32, i32 } %Z, i32 %key, 2
  %Z3 = insertvalue { i32, i32, i32, i32, i32 } %Z2, i32 %key, 3
  %Z4 = insertvalue { i32, i32, i32, i32, i32 } %Z3, i32 %key, 4
  ret { i32, i32, i32, i32, i32 } %Z4
}

; The return value {i32, i32, i32, i32} will be returned via registers %r0, %r1,
; %r2, %r3.
; CHECK-LABEL: test3:
; CHECK: bl {{.*}}gen3
; CHECK: add r0, r0, r1
; CHECK: add r0, r0, r2
; CHECK: add r0, r0, r3
; CHECK-O0-LABEL: test3:
; CHECK-O0: bl {{.*}}gen3
; CHECK-O0: add r0, r0, r1
; CHECK-O0: add r0, r0, r2
; CHECK-O0: add r0, r0, r3
define i32 @test3(i32 %key) #0 {
entry:
  %key.addr = alloca i32, align 4
  store i32 %key, ptr %key.addr, align 4
  %0 = load i32, ptr %key.addr, align 4
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

; The return value {float, float, float, float} will be returned via registers
; s0-s3.
; CHECK-LABEL: test4:
; CHECK: bl _gen4
; CHECK: vadd.f32        s0, s0, s1
; CHECK: vadd.f32        s0, s0, s2
; CHECK: vadd.f32        s0, s0, s3
; CHECK-O0-LABEL: test4:
; CHECK-O0: bl _gen4
; CHECK-O0: vadd.f32        s0, s0, s1
; CHECK-O0: vadd.f32        s0, s0, s2
; CHECK-O0: vadd.f32        s0, s0, s3
define float @test4(float %key) #0 {
entry:
  %key.addr = alloca float, align 4
  store float %key, ptr %key.addr, align 4
  %0 = load float, ptr %key.addr, align 4
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
; CHECK:  vadd.f64        [[TMP:d.*]], d0, d1
; CHECK:  vadd.f64        [[TMP]], [[TMP]], d2
; CHECK:  vadd.f64        d0, [[TMP]], d3
define swiftcc double @test5() #0 {
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
; CHECK: bl      _gen6
; CHECK-DAG: vadd.f64        [[TMP:d.*]], d0, d1
; CHECK-DAG: add     r0, r0, r1
; CHECK-DAG: add     r0, r0, r2
; CHECK-DAG: add     r0, r0, r3
; CHECK-DAG: vadd.f64        [[TMP]], [[TMP]], d2
; CHECK-DAG: vadd.f64        d0, [[TMP]], d3
define swiftcc { double, i32 } @test6() #0 {
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

; CHECK-LABEL: gen7
; CHECK:  mov     r1, r0
; CHECK:  mov     r2, r0
; CHECK:  mov     r3, r0
; CHECK:  bx lr
define swiftcc { i32, i32, i32, i32 } @gen7(i32 %key) {
  %v0 = insertvalue { i32, i32, i32, i32 } undef, i32 %key, 0
  %v1 = insertvalue { i32, i32, i32, i32 } %v0, i32 %key, 1
  %v2 = insertvalue { i32, i32, i32, i32 } %v1, i32 %key, 2
  %v3 = insertvalue { i32, i32, i32, i32 } %v2, i32 %key, 3
  ret { i32, i32, i32, i32 } %v3
}

; CHECK-LABEL: gen9
; CHECK:  mov     r1, r0
; CHECK:  mov     r2, r0
; CHECK:  mov     r3, r0
; CHECK:  bx lr
define swiftcc { i8, i8, i8, i8 } @gen9(i8 %key) {
  %v0 = insertvalue { i8, i8, i8, i8 } undef, i8 %key, 0
  %v1 = insertvalue { i8, i8, i8, i8 } %v0, i8 %key, 1
  %v2 = insertvalue { i8, i8, i8, i8 } %v1, i8 %key, 2
  %v3 = insertvalue { i8, i8, i8, i8 } %v2, i8 %key, 3
  ret { i8, i8, i8, i8 } %v3
}
; CHECK-LABEL: gen10
; CHECK-DAG:  vmov.f64        d1, d0
; CHECK-DAG:  mov     r1, r0
; CHECK-DAG:  mov     r2, r0
; CHECK-DAG:  mov     r3, r0
; CHECK-DAG:  vmov.f64        d2, d0
; CHECK-DAG:  vmov.f64        d3, d0
; CHECK-DAG:  bx      lr
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


; CHECK-LABEL: test11
; CHECK:  bl      _gen11
; CHECK:  vadd.f32        [[TMP:q.*]], q0, q1
; CHECK:  vadd.f32        [[TMP]], [[TMP]], q2
; CHECK:  vadd.f32        q0, [[TMP]], q3
define swiftcc <4 x float> @test11() #0 {
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

; CHECK-LABEL: test12
; CHECK-DAG:  vadd.f32        [[TMP:q.*]], q0, q1
; CHECK-DAG:  vmov.f32        s4, s12
; CHECK-DAG:  vadd.f32        q0, [[TMP]], q2
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

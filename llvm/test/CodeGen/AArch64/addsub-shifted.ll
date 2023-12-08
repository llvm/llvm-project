; RUN: llc -debugify-and-strip-all-safe -verify-machineinstrs %s -o - -mtriple=arm64-apple-ios7.0 | FileCheck %s
; RUN: llc -debugify-and-strip-all-safe -verify-machineinstrs %s -o - -mtriple=arm64-apple-ios7.0 -global-isel -pass-remarks-missed=gisel* 2>&1 | FileCheck %s --check-prefixes=CHECK,FALLBACK

; FALLBACK-NOT: remark

@var32 = global i32 0
@var64 = global i64 0

define void @test_lsl_arith(i32 %lhs32, i32 %rhs32, i64 %lhs64, i64 %rhs64) {
; CHECK-LABEL: test_lsl_arith:

  %rhs1 = load volatile i32, ptr @var32
  %shift1 = shl i32 %rhs1, 18
  %val1 = add i32 %lhs32, %shift1
  store volatile i32 %val1, ptr @var32
; CHECK: add {{w[0-9]+}}, {{w[0-9]+}}, {{w[0-9]+}}, lsl #18

  %rhs2 = load volatile i32, ptr @var32
  %shift2 = shl i32 %rhs2, 31
  %val2 = add i32 %shift2, %lhs32
  store volatile i32 %val2, ptr @var32
; CHECK: add {{w[0-9]+}}, {{w[0-9]+}}, {{w[0-9]+}}, lsl #31

  %rhs3 = load volatile i32, ptr @var32
  %shift3 = shl i32 %rhs3, 5
  %val3 = sub i32 %lhs32, %shift3
  store volatile i32 %val3, ptr @var32
; CHECK: sub {{w[0-9]+}}, {{w[0-9]+}}, {{w[0-9]+}}, lsl #5

; Subtraction is not commutative!
  %rhs4 = load volatile i32, ptr @var32
  %shift4 = shl i32 %rhs4, 19
  %val4 = sub i32 %shift4, %lhs32
  store volatile i32 %val4, ptr @var32
; CHECK-NOT: sub {{w[0-9]+}}, {{w[0-9]+}}, {{w[0-9]+}}, lsl #19

  %lhs4a = load volatile i32, ptr @var32
  %shift4a = shl i32 %lhs4a, 15
  %val4a = sub i32 0, %shift4a
  store volatile i32 %val4a, ptr @var32
; CHECK: neg {{w[0-9]+}}, {{w[0-9]+}}, lsl #15

  %rhs5 = load volatile i64, ptr @var64
  %shift5 = shl i64 %rhs5, 18
  %val5 = add i64 %lhs64, %shift5
  store volatile i64 %val5, ptr @var64
; CHECK: add {{x[0-9]+}}, {{x[0-9]+}}, {{x[0-9]+}}, lsl #18

  %rhs6 = load volatile i64, ptr @var64
  %shift6 = shl i64 %rhs6, 31
  %val6 = add i64 %shift6, %lhs64
  store volatile i64 %val6, ptr @var64
; CHECK: add {{x[0-9]+}}, {{x[0-9]+}}, {{x[0-9]+}}, lsl #31

  %rhs7 = load volatile i64, ptr @var64
  %shift7 = shl i64 %rhs7, 5
  %val7 = sub i64 %lhs64, %shift7
  store volatile i64 %val7, ptr @var64
; CHECK: sub {{x[0-9]+}}, {{x[0-9]+}}, {{x[0-9]+}}, lsl #5

; Subtraction is not commutative!
  %rhs8 = load volatile i64, ptr @var64
  %shift8 = shl i64 %rhs8, 19
  %val8 = sub i64 %shift8, %lhs64
  store volatile i64 %val8, ptr @var64
; CHECK-NOT: sub {{x[0-9]+}}, {{x[0-9]+}}, {{x[0-9]+}}, lsl #19

  %lhs8a = load volatile i64, ptr @var64
  %shift8a = shl i64 %lhs8a, 60
  %val8a = sub i64 0, %shift8a
  store volatile i64 %val8a, ptr @var64
; CHECK: neg {{x[0-9]+}}, {{x[0-9]+}}, lsl #60

  ret void
; CHECK: ret
}

define void @test_lsr_arith(i32 %lhs32, i32 %rhs32, i64 %lhs64, i64 %rhs64) {
; CHECK-LABEL: test_lsr_arith:

  %shift1 = lshr i32 %rhs32, 18
  %val1 = add i32 %lhs32, %shift1
  store volatile i32 %val1, ptr @var32
; CHECK: add {{w[0-9]+}}, {{w[0-9]+}}, {{w[0-9]+}}, lsr #18

  %shift2 = lshr i32 %rhs32, 31
  %val2 = add i32 %shift2, %lhs32
  store volatile i32 %val2, ptr @var32
; CHECK: add {{w[0-9]+}}, {{w[0-9]+}}, {{w[0-9]+}}, lsr #31

  %shift3 = lshr i32 %rhs32, 5
  %val3 = sub i32 %lhs32, %shift3
  store volatile i32 %val3, ptr @var32
; CHECK: sub {{w[0-9]+}}, {{w[0-9]+}}, {{w[0-9]+}}, lsr #5

; Subtraction is not commutative!
  %shift4 = lshr i32 %rhs32, 19
  %val4 = sub i32 %shift4, %lhs32
  store volatile i32 %val4, ptr @var32
; CHECK-NOT: sub {{w[0-9]+}}, {{w[0-9]+}}, {{w[0-9]+}}, lsr #19

  %shift4a = lshr i32 %lhs32, 15
  %val4a = sub i32 0, %shift4a
  store volatile i32 %val4a, ptr @var32
; CHECK: neg {{w[0-9]+}}, {{w[0-9]+}}, lsr #15

  %shift5 = lshr i64 %rhs64, 18
  %val5 = add i64 %lhs64, %shift5
  store volatile i64 %val5, ptr @var64
; CHECK: add {{x[0-9]+}}, {{x[0-9]+}}, {{x[0-9]+}}, lsr #18

  %shift6 = lshr i64 %rhs64, 31
  %val6 = add i64 %shift6, %lhs64
  store volatile i64 %val6, ptr @var64
; CHECK: add {{x[0-9]+}}, {{x[0-9]+}}, {{x[0-9]+}}, lsr #31

  %shift7 = lshr i64 %rhs64, 5
  %val7 = sub i64 %lhs64, %shift7
  store volatile i64 %val7, ptr @var64
; CHECK: sub {{x[0-9]+}}, {{x[0-9]+}}, {{x[0-9]+}}, lsr #5

; Subtraction is not commutative!
  %shift8 = lshr i64 %rhs64, 19
  %val8 = sub i64 %shift8, %lhs64
  store volatile i64 %val8, ptr @var64
; CHECK-NOT: sub {{x[0-9]+}}, {{x[0-9]+}}, {{x[0-9]+}}, lsr #19

  %shift8a = lshr i64 %lhs64, 45
  %val8a = sub i64 0, %shift8a
  store volatile i64 %val8a, ptr @var64
; CHECK: neg {{x[0-9]+}}, {{x[0-9]+}}, lsr #45

  ret void
; CHECK: ret
}

define void @test_asr_arith(i32 %lhs32, i32 %rhs32, i64 %lhs64, i64 %rhs64) {
; CHECK-LABEL: test_asr_arith:

  %shift1 = ashr i32 %rhs32, 18
  %val1 = add i32 %lhs32, %shift1
  store volatile i32 %val1, ptr @var32
; CHECK: add {{w[0-9]+}}, {{w[0-9]+}}, {{w[0-9]+}}, asr #18

  %shift2 = ashr i32 %rhs32, 31
  %val2 = add i32 %shift2, %lhs32
  store volatile i32 %val2, ptr @var32
; CHECK: add {{w[0-9]+}}, {{w[0-9]+}}, {{w[0-9]+}}, asr #31

  %shift3 = ashr i32 %rhs32, 5
  %val3 = sub i32 %lhs32, %shift3
  store volatile i32 %val3, ptr @var32
; CHECK: sub {{w[0-9]+}}, {{w[0-9]+}}, {{w[0-9]+}}, asr #5

; Subtraction is not commutative!
  %shift4 = ashr i32 %rhs32, 19
  %val4 = sub i32 %shift4, %lhs32
  store volatile i32 %val4, ptr @var32
; CHECK-NOT: sub {{w[0-9]+}}, {{w[0-9]+}}, {{w[0-9]+}}, asr #19

  %shift4a = ashr i32 %lhs32, 15
  %val4a = sub i32 0, %shift4a
  store volatile i32 %val4a, ptr @var32
; CHECK: neg {{w[0-9]+}}, {{w[0-9]+}}, asr #15

  %shift5 = ashr i64 %rhs64, 18
  %val5 = add i64 %lhs64, %shift5
  store volatile i64 %val5, ptr @var64
; CHECK: add {{x[0-9]+}}, {{x[0-9]+}}, {{x[0-9]+}}, asr #18

  %shift6 = ashr i64 %rhs64, 31
  %val6 = add i64 %shift6, %lhs64
  store volatile i64 %val6, ptr @var64
; CHECK: add {{x[0-9]+}}, {{x[0-9]+}}, {{x[0-9]+}}, asr #31

  %shift7 = ashr i64 %rhs64, 5
  %val7 = sub i64 %lhs64, %shift7
  store volatile i64 %val7, ptr @var64
; CHECK: sub {{x[0-9]+}}, {{x[0-9]+}}, {{x[0-9]+}}, asr #5

; Subtraction is not commutative!
  %shift8 = ashr i64 %rhs64, 19
  %val8 = sub i64 %shift8, %lhs64
  store volatile i64 %val8, ptr @var64
; CHECK-NOT: sub {{x[0-9]+}}, {{x[0-9]+}}, {{x[0-9]+}}, asr #19

  %shift8a = ashr i64 %lhs64, 45
  %val8a = sub i64 0, %shift8a
  store volatile i64 %val8a, ptr @var64
; CHECK: neg {{x[0-9]+}}, {{x[0-9]+}}, asr #45

  ret void
; CHECK: ret
}

define void @test_cmp(i32 %lhs32, i32 %rhs32, i64 %lhs64, i64 %rhs64, i32 %v) {
; CHECK-LABEL: test_cmp:

  %shift1 = shl i32 %rhs32, 13
  %tst1 = icmp uge i32 %lhs32, %shift1
  br i1 %tst1, label %t2, label %end
; CHECK: cmp {{w[0-9]+}}, {{w[0-9]+}}, lsl #13

t2:
  store volatile i32 %v, ptr @var32
  %shift2 = lshr i32 %rhs32, 20
  %tst2 = icmp ne i32 %lhs32, %shift2
  br i1 %tst2, label %t3, label %end
; CHECK: cmp {{w[0-9]+}}, {{w[0-9]+}}, lsr #20

t3:
  store volatile i32 %v, ptr @var32
  %shift3 = ashr i32 %rhs32, 9
  %tst3 = icmp ne i32 %lhs32, %shift3
  br i1 %tst3, label %t4, label %end
; CHECK: cmp {{w[0-9]+}}, {{w[0-9]+}}, asr #9

t4:
  store volatile i32 %v, ptr @var32
  %shift4 = shl i64 %rhs64, 43
  %tst4 = icmp uge i64 %lhs64, %shift4
  br i1 %tst4, label %t5, label %end
; CHECK: cmp {{x[0-9]+}}, {{x[0-9]+}}, lsl #43

t5:
  store volatile i32 %v, ptr @var32
  %shift5 = lshr i64 %rhs64, 20
  %tst5 = icmp ne i64 %lhs64, %shift5
  br i1 %tst5, label %t6, label %end
; CHECK: cmp {{x[0-9]+}}, {{x[0-9]+}}, lsr #20

t6:
  store volatile i32 %v, ptr @var32
  %shift6 = ashr i64 %rhs64, 59
  %tst6 = icmp ne i64 %lhs64, %shift6
  br i1 %tst6, label %t7, label %end
; CHECK: cmp {{x[0-9]+}}, {{x[0-9]+}}, asr #59

t7:
  store volatile i32 %v, ptr @var32
  br label %end

end:
  ret void
; CHECK: ret
}

define i32 @test_cmn(i32 %lhs32, i32 %rhs32, i64 %lhs64, i64 %rhs64) {
; CHECK-LABEL: test_cmn:

  %shift1 = shl i32 %rhs32, 13
  %val1 = sub i32 0, %shift1
  %tst1 = icmp uge i32 %lhs32, %val1
  br i1 %tst1, label %t2, label %end
  ; Important that this isn't lowered to a cmn instruction because if %rhs32 ==
  ; 0 then the results will differ.
; CHECK: neg [[RHS:w[0-9]+]], {{w[0-9]+}}, lsl #13
; CHECK: cmp {{w[0-9]+}}, [[RHS]]

t2:
  %shift2 = lshr i32 %rhs32, 20
  %val2 = sub i32 0, %shift2
  %tst2 = icmp ne i32 %lhs32, %val2
  br i1 %tst2, label %t3, label %end
; CHECK: cmn {{w[0-9]+}}, {{w[0-9]+}}, lsr #20

t3:
  %shift3 = ashr i32 %rhs32, 9
  %val3 = sub i32 0, %shift3
  %tst3 = icmp eq i32 %lhs32, %val3
  br i1 %tst3, label %t4, label %end
; CHECK: cmn {{w[0-9]+}}, {{w[0-9]+}}, asr #9

t4:
  %shift4 = shl i64 %rhs64, 43
  %val4 = sub i64 0, %shift4
  %tst4 = icmp slt i64 %lhs64, %val4
  br i1 %tst4, label %t5, label %end
  ; Again, it's important that cmn isn't used here in case %rhs64 == 0.
; CHECK: neg [[RHS:x[0-9]+]], {{x[0-9]+}}, lsl #43
; CHECK: cmp {{x[0-9]+}}, [[RHS]]

t5:
  %shift5 = lshr i64 %rhs64, 20
  %val5 = sub i64 0, %shift5
  %tst5 = icmp ne i64 %lhs64, %val5
  br i1 %tst5, label %t6, label %end
; CHECK: cmn {{x[0-9]+}}, {{x[0-9]+}}, lsr #20

t6:
  %shift6 = ashr i64 %rhs64, 59
  %val6 = sub i64 0, %shift6
  %tst6 = icmp ne i64 %lhs64, %val6
  br i1 %tst6, label %t7, label %end
; CHECK: cmn {{x[0-9]+}}, {{x[0-9]+}}, asr #59

t7:
  ret i32 1
end:

  ret i32 0
; CHECK: ret
}

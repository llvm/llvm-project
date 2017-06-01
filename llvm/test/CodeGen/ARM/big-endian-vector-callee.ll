; RUN: llc -mtriple armeb-eabi -mattr v7,neon -float-abi soft %s -o - | FileCheck %s -check-prefix CHECK -check-prefix SOFT
; RUN: llc -mtriple armeb-eabi -mattr v7,neon -float-abi hard %s -o - | FileCheck %s -check-prefix CHECK -check-prefix HARD

; CHECK-LABEL: test_i64_f64:
define i64 @test_i64_f64(double %p) {
; SOFT: vmov [[REG:d[0-9]+]], r1, r0
; SOFT: vadd.f64 [[REG]]
; HARD: vadd.f64 d{{[0-9]+}}, d0
    %1 = fadd double %p, %p
    %2 = bitcast double %1 to i64
    %3 = add i64 %2, %2
    ret i64 %3
; CHECK: adds r1
; CHECK: adc r0
}

; CHECK-LABEL: test_i64_v1i64:
define i64 @test_i64_v1i64(<1 x i64> %p) {
; SOFT: vmov [[REG:d[0-9]+]], r1, r0
; SOFT: vadd.i64 [[REG]]
; HARD: vadd.i64 d{{[0-9]+}}, d0
    %1 = add <1 x i64> %p, %p
    %2 = bitcast <1 x i64> %1 to i64
    %3 = add i64 %2, %2
    ret i64 %3
; CHECK: adds r1
; CHECK: adc r0
}

; CHECK-LABEL: test_i64_v2f32:
define i64 @test_i64_v2f32(<2 x float> %p) {
; SOFT: vmov [[REG:d[0-9]+]], r1, r0
; SOFT: vrev64.32 [[REG]]
; HARD: vrev64.32 d{{[0-9]+}}, d0
    %1 = fadd <2 x float> %p, %p
    %2 = bitcast <2 x float> %1 to i64
    %3 = add i64 %2, %2
    ret i64 %3
; CHECK: adds r1
; CHECK: adc r0
}

; CHECK-LABEL: test_i64_v2i32:
define i64 @test_i64_v2i32(<2 x i32> %p) {
; SOFT: vmov [[REG:d[0-9]+]], r1, r0
; SOFT: vrev64.32 [[REG]]
; HARD: vrev64.32 d{{[0-9]+}}, d0
    %1 = add <2 x i32> %p, %p
    %2 = bitcast <2 x i32> %1 to i64
    %3 = add i64 %2, %2
    ret i64 %3
; CHECK: adds r1
; CHECK: adc r0
}

; CHECK-LABEL: test_i64_v4i16:
define i64 @test_i64_v4i16(<4 x i16> %p) {
; SOFT: vmov [[REG:d[0-9]+]], r1, r0
; SOFT: vrev64.16 [[REG]]
; HARD: vrev64.16 d{{[0-9]+}}, d0
    %1 = add <4 x i16> %p, %p
    %2 = bitcast <4 x i16> %1 to i64
    %3 = add i64 %2, %2
    ret i64 %3
; CHECK: adds r1
; CHECK: adc r0
}

; CHECK-LABEL: test_i64_v8i8:
define i64 @test_i64_v8i8(<8 x i8> %p) {
; SOFT: vmov [[REG:d[0-9]+]], r1, r0
; SOFT: vrev64.8 [[REG]]
; HARD: vrev64.8 d{{[0-9]+}}, d0
    %1 = add <8 x i8> %p, %p
    %2 = bitcast <8 x i8> %1 to i64
    %3 = add i64 %2, %2
    ret i64 %3
; CHECK: adds r1
; CHECK: adc r0
}

; CHECK-LABEL: test_f64_i64:
define double @test_f64_i64(i64 %p) {
; CHECK: adds r1
; CHECK: adc r0
    %1 = add i64 %p, %p
    %2 = bitcast i64 %1 to double
    %3 = fadd double %2, %2
    ret double %3
; SOFT: vadd.f64 [[REG:d[0-9]+]]
; SOFT: vmov r1, r0, [[REG]]
; HARD: vadd.f64 d0
}

; CHECK-LABEL: test_f64_v1i64:
define double @test_f64_v1i64(<1 x i64> %p) {
; SOFT: vmov [[REG:d[0-9]+]], r1, r0
; SOFT: vadd.i64 [[REG]]
; HARD: vadd.i64 d{{[0-9]+}}, d0
    %1 = add <1 x i64> %p, %p
    %2 = bitcast <1 x i64> %1 to double
    %3 = fadd double %2, %2
    ret double %3
; SOFT: vadd.f64 [[REG:d[0-9]+]]
; SOFT: vmov r1, r0, [[REG]]
; HARD: vadd.f64 d0
}

; CHECK-LABEL: test_f64_v2f32:
define double @test_f64_v2f32(<2 x float> %p) {
; SOFT: vmov [[REG:d[0-9]+]], r1, r0
; SOFT: vrev64.32 [[REG]]
; HARD: vrev64.32 d{{[0-9]+}}, d0
    %1 = fadd <2 x float> %p, %p
    %2 = bitcast <2 x float> %1 to double
    %3 = fadd double %2, %2
    ret double %3
; SOFT: vadd.f64 [[REG:d[0-9]+]]
; SOFT: vmov r1, r0, [[REG]]
; HARD: vadd.f64 d0
}

; CHECK-LABEL: test_f64_v2i32:
define double @test_f64_v2i32(<2 x i32> %p) {
; SOFT: vmov [[REG:d[0-9]+]], r1, r0
; SOFT: vrev64.32 [[REG]]
; HARD: vrev64.32 d{{[0-9]+}}, d0
    %1 = add <2 x i32> %p, %p
    %2 = bitcast <2 x i32> %1 to double
    %3 = fadd double %2, %2
    ret double %3
; SOFT: vadd.f64 [[REG:d[0-9]+]]
; SOFT: vmov r1, r0, [[REG]]
; HARD: vadd.f64 d0
}

; CHECK-LABEL: test_f64_v4i16:
define double @test_f64_v4i16(<4 x i16> %p) {
; SOFT: vmov [[REG:d[0-9]+]], r1, r0
; SOFT: vrev64.16 [[REG]]
; HARD: vrev64.16 d{{[0-9]+}}, d0
    %1 = add <4 x i16> %p, %p
    %2 = bitcast <4 x i16> %1 to double
    %3 = fadd double %2, %2
    ret double %3
; SOFT: vadd.f64 [[REG:d[0-9]+]]
; SOFT: vmov r1, r0, [[REG]]
; HARD: vadd.f64 d0
}

; CHECK-LABEL: test_f64_v8i8:
define double @test_f64_v8i8(<8 x i8> %p) {
; SOFT: vmov [[REG:d[0-9]+]], r1, r0
; SOFT: vrev64.8 [[REG]]
; HARD: vrev64.8 d{{[0-9]+}}, d0
    %1 = add <8 x i8> %p, %p
    %2 = bitcast <8 x i8> %1 to double
    %3 = fadd double %2, %2
    ret double %3
; SOFT: vadd.f64 [[REG:d[0-9]+]]
; SOFT: vmov r1, r0, [[REG]]
; HARD: vadd.f64 d0
}

; CHECK-LABEL: test_v1i64_i64:
define <1 x i64> @test_v1i64_i64(i64 %p) {
; CHECK: adds r1
; CHECK: adc r0
    %1 = add i64 %p, %p
    %2 = bitcast i64 %1 to <1 x i64>
    %3 = add <1 x i64> %2, %2
    ret <1 x i64> %3
; SOFT: vadd.i64 [[REG:d[0-9]+]]
; SOFT: vmov r1, r0, [[REG]]
; HARD: vadd.i64 d0
}

; CHECK-LABEL: test_v1i64_f64:
define <1 x i64> @test_v1i64_f64(double %p) {
; SOFT: vmov [[REG:d[0-9]+]], r1, r0
; SOFT: vadd.f64 [[REG]]
; HARD: vadd.f64 d{{[0-9]+}}, d0
    %1 = fadd double %p, %p
    %2 = bitcast double %1 to <1 x i64>
    %3 = add <1 x i64> %2, %2
    ret <1 x i64> %3
; SOFT: vadd.i64 [[REG:d[0-9]+]]
; SOFT: vmov r1, r0, [[REG]]
; HARD: vadd.i64 d0
}

; CHECK-LABEL: test_v1i64_v2f32:
define <1 x i64> @test_v1i64_v2f32(<2 x float> %p) {
; SOFT: vmov [[REG:d[0-9]+]], r1, r0
; SOFT: vrev64.32 [[REG]]
; HARD: vrev64.32 d{{[0-9]+}}, d0
    %1 = fadd <2 x float> %p, %p
    %2 = bitcast <2 x float> %1 to <1 x i64>
    %3 = add <1 x i64> %2, %2
    ret <1 x i64> %3
; SOFT: vadd.i64 [[REG:d[0-9]+]]
; SOFT: vmov r1, r0, [[REG]]
; HARD: vadd.i64 d0
}

; CHECK-LABEL: test_v1i64_v2i32:
define <1 x i64> @test_v1i64_v2i32(<2 x i32> %p) {
; SOFT: vmov [[REG:d[0-9]+]], r1, r0
; SOFT: vrev64.32 [[REG]]
; HARD: vrev64.32 d{{[0-9]+}}, d0
    %1 = add <2 x i32> %p, %p
    %2 = bitcast <2 x i32> %1 to <1 x i64>
    %3 = add <1 x i64> %2, %2
    ret <1 x i64> %3
; SOFT: vadd.i64 [[REG:d[0-9]+]]
; SOFT: vmov r1, r0, [[REG]]
; HARD: vadd.i64 d0
}

; CHECK-LABEL: test_v1i64_v4i16:
define <1 x i64> @test_v1i64_v4i16(<4 x i16> %p) {
; SOFT: vmov [[REG:d[0-9]+]], r1, r0
; SOFT: vrev64.16 [[REG]]
; HARD: vrev64.16 d{{[0-9]+}}, d0
    %1 = add <4 x i16> %p, %p
    %2 = bitcast <4 x i16> %1 to <1 x i64>
    %3 = add <1 x i64> %2, %2
    ret <1 x i64> %3
; SOFT: vadd.i64 [[REG:d[0-9]+]]
; SOFT: vmov r1, r0, [[REG]]
; HARD: vadd.i64 d0
}

; CHECK-LABEL: test_v1i64_v8i8:
define <1 x i64> @test_v1i64_v8i8(<8 x i8> %p) {
; SOFT: vmov [[REG:d[0-9]+]], r1, r0
; SOFT: vrev64.8 [[REG]]
; HARD: vrev64.8 d{{[0-9]+}}, d0
    %1 = add <8 x i8> %p, %p
    %2 = bitcast <8 x i8> %1 to <1 x i64>
    %3 = add <1 x i64> %2, %2
    ret <1 x i64> %3
; SOFT: vadd.i64 [[REG:d[0-9]+]]
; SOFT: vmov r1, r0, [[REG]]
; HARD: vadd.i64 d0
}

; CHECK-LABEL: test_v2f32_i64:
define <2 x float> @test_v2f32_i64(i64 %p) {
; CHECK: adds r1
; CHECK: adc r0
    %1 = add i64 %p, %p
    %2 = bitcast i64 %1 to <2 x float>
    %3 = fadd <2 x float> %2, %2
    ret <2 x float> %3
; SOFT: vrev64.32 [[REG:d[0-9]+]]
; SOFT: vmov r1, r0, [[REG]]
; HARD: vrev64.32 d0
}

; CHECK-LABEL: test_v2f32_f64:
define <2 x float> @test_v2f32_f64(double %p) {
; SOFT: vmov [[REG:d[0-9]+]], r1, r0
; SOFT: vadd.f64 [[REG]]
; HARD: vadd.f64 d{{[0-9]+}}, d0
    %1 = fadd double %p, %p
    %2 = bitcast double %1 to <2 x float>
    %3 = fadd <2 x float> %2, %2
    ret <2 x float> %3
; SOFT: vrev64.32 [[REG:d[0-9]+]]
; SOFT: vmov r1, r0, [[REG]]
; HARD: vrev64.32 d0
}

; CHECK-LABEL: test_v2f32_v1i64:
define <2 x float> @test_v2f32_v1i64(<1 x i64> %p) {
; SOFT: vmov [[REG:d[0-9]+]], r1, r0
; SOFT: vadd.i64 [[REG]]
; HARD: vadd.i64 d{{[0-9]+}}, d0
    %1 = add <1 x i64> %p, %p
    %2 = bitcast <1 x i64> %1 to <2 x float>
    %3 = fadd <2 x float> %2, %2
    ret <2 x float> %3
; SOFT: vrev64.32 [[REG:d[0-9]+]]
; SOFT: vmov r1, r0, [[REG]]
; HARD: vrev64.32 d0
}

; CHECK-LABEL: test_v2f32_v2i32:
define <2 x float> @test_v2f32_v2i32(<2 x i32> %p) {
; SOFT: vmov [[REG:d[0-9]+]], r1, r0
; SOFT: vrev64.32 [[REG]]
; HARD: vrev64.32 d{{[0-9]+}}, d0
    %1 = add <2 x i32> %p, %p
    %2 = bitcast <2 x i32> %1 to <2 x float>
    %3 = fadd <2 x float> %2, %2
    ret <2 x float> %3
; SOFT: vrev64.32 [[REG:d[0-9]+]]
; SOFT: vmov r1, r0, [[REG]]
; HARD: vrev64.32 d0
}

; CHECK-LABEL: test_v2f32_v4i16:
define <2 x float> @test_v2f32_v4i16(<4 x i16> %p) {
; SOFT: vmov [[REG:d[0-9]+]], r1, r0
; SOFT: vrev64.16 [[REG]]
; HARD: vrev64.16 d{{[0-9]+}}, d0
    %1 = add <4 x i16> %p, %p
    %2 = bitcast <4 x i16> %1 to <2 x float>
    %3 = fadd <2 x float> %2, %2
    ret <2 x float> %3
; SOFT: vrev64.32 [[REG:d[0-9]+]]
; SOFT: vmov r1, r0, [[REG]]
; HARD: vrev64.32 d0
}

; CHECK-LABEL: test_v2f32_v8i8:
define <2 x float> @test_v2f32_v8i8(<8 x i8> %p) {
; SOFT: vmov [[REG:d[0-9]+]], r1, r0
; SOFT: vrev64.8 [[REG]]
; HARD: vrev64.8 d{{[0-9]+}}, d0
    %1 = add <8 x i8> %p, %p
    %2 = bitcast <8 x i8> %1 to <2 x float>
    %3 = fadd <2 x float> %2, %2
    ret <2 x float> %3
; SOFT: vrev64.32 [[REG:d[0-9]+]]
; SOFT: vmov r1, r0, [[REG]]
; HARD: vrev64.32 d0
}

; CHECK-LABEL: test_v2i32_i64:
define <2 x i32> @test_v2i32_i64(i64 %p) {
; CHECK: adds r1
; CHECK: adc r0
    %1 = add i64 %p, %p
    %2 = bitcast i64 %1 to <2 x i32>
    %3 = add <2 x i32> %2, %2
    ret <2 x i32> %3
; SOFT: vrev64.32 [[REG:d[0-9]+]]
; SOFT: vmov r1, r0, [[REG]]
; HARD: vrev64.32 d0
}

; CHECK-LABEL: test_v2i32_f64:
define <2 x i32> @test_v2i32_f64(double %p) {
; SOFT: vmov [[REG:d[0-9]+]], r1, r0
; SOFT: vadd.f64 [[REG]]
; HARD: vadd.f64 d{{[0-9]+}}, d0
    %1 = fadd double %p, %p
    %2 = bitcast double %1 to <2 x i32>
    %3 = add <2 x i32> %2, %2
    ret <2 x i32> %3
; SOFT: vrev64.32 [[REG:d[0-9]+]]
; SOFT: vmov r1, r0, [[REG]]
; HARD: vrev64.32 d0
}

; CHECK-LABEL: test_v2i32_v1i64:
define <2 x i32> @test_v2i32_v1i64(<1 x i64> %p) {
; SOFT: vmov [[REG:d[0-9]+]], r1, r0
; SOFT: vadd.i64 [[REG]]
; HARD: vadd.i64 d{{[0-9]+}}, d0
    %1 = add <1 x i64> %p, %p
    %2 = bitcast <1 x i64> %1 to <2 x i32>
    %3 = add <2 x i32> %2, %2
    ret <2 x i32> %3
; SOFT: vrev64.32 [[REG:d[0-9]+]]
; SOFT: vmov r1, r0, [[REG]]
; HARD: vrev64.32 d0
}

; CHECK-LABEL: test_v2i32_v2f32:
define <2 x i32> @test_v2i32_v2f32(<2 x float> %p) {
; SOFT: vmov [[REG:d[0-9]+]], r1, r0
; SOFT: vrev64.32 [[REG]]
; HARD: vrev64.32 d{{[0-9]+}}, d0
    %1 = fadd <2 x float> %p, %p
    %2 = bitcast <2 x float> %1 to <2 x i32>
    %3 = add <2 x i32> %2, %2
    ret <2 x i32> %3
; SOFT: vrev64.32 [[REG:d[0-9]+]]
; SOFT: vmov r1, r0, [[REG]]
; HARD: vrev64.32 d0
}

; CHECK-LABEL: test_v2i32_v4i16:
define <2 x i32> @test_v2i32_v4i16(<4 x i16> %p) {
; SOFT: vmov [[REG:d[0-9]+]], r1, r0
; SOFT: vrev64.16 [[REG]]
; HARD: vrev64.16 d{{[0-9]+}}, d0
    %1 = add <4 x i16> %p, %p
    %2 = bitcast <4 x i16> %1 to <2 x i32>
    %3 = add <2 x i32> %2, %2
    ret <2 x i32> %3
; SOFT: vrev64.32 [[REG:d[0-9]+]]
; SOFT: vmov r1, r0, [[REG]]
; HARD: vrev64.32 d0
}

; CHECK-LABEL: test_v2i32_v8i8:
define <2 x i32> @test_v2i32_v8i8(<8 x i8> %p) {
; SOFT: vmov [[REG:d[0-9]+]], r1, r0
; SOFT: vrev64.8 [[REG]]
; HARD: vrev64.8 d{{[0-9]+}}, d0
    %1 = add <8 x i8> %p, %p
    %2 = bitcast <8 x i8> %1 to <2 x i32>
    %3 = add <2 x i32> %2, %2
    ret <2 x i32> %3
; SOFT: vrev64.32 [[REG:d[0-9]+]]
; SOFT: vmov r1, r0, [[REG]]
; HARD: vrev64.32 d0
}

; CHECK-LABEL: test_v4i16_i64:
define <4 x i16> @test_v4i16_i64(i64 %p) {
; CHECK: adds r1
; CHECK: adc r0
    %1 = add i64 %p, %p
    %2 = bitcast i64 %1 to <4 x i16>
    %3 = add <4 x i16> %2, %2
    ret <4 x i16> %3
; SOFT: vrev64.16 [[REG:d[0-9]+]]
; SOFT: vmov r1, r0, [[REG]]
; HARD: vrev64.16 d0
}

; CHECK-LABEL: test_v4i16_f64:
define <4 x i16> @test_v4i16_f64(double %p) {
; SOFT: vmov [[REG:d[0-9]+]], r1, r0
; SOFT: vadd.f64 [[REG]]
; HARD: vadd.f64 d{{[0-9]+}}, d0
    %1 = fadd double %p, %p
    %2 = bitcast double %1 to <4 x i16>
    %3 = add <4 x i16> %2, %2
    ret <4 x i16> %3
; SOFT: vrev64.16 [[REG:d[0-9]+]]
; SOFT: vmov r1, r0, [[REG]]
; HARD: vrev64.16 d0
}

; CHECK-LABEL: test_v4i16_v1i64:
define <4 x i16> @test_v4i16_v1i64(<1 x i64> %p) {
; SOFT: vmov [[REG:d[0-9]+]], r1, r0
; SOFT: vadd.i64 [[REG]]
; HARD: vadd.i64 d{{[0-9]+}}, d0
    %1 = add <1 x i64> %p, %p
    %2 = bitcast <1 x i64> %1 to <4 x i16>
    %3 = add <4 x i16> %2, %2
    ret <4 x i16> %3
; SOFT: vrev64.16 [[REG:d[0-9]+]]
; SOFT: vmov r1, r0, [[REG]]
; HARD: vrev64.16 d0
}

; CHECK-LABEL: test_v4i16_v2f32:
define <4 x i16> @test_v4i16_v2f32(<2 x float> %p) {
; SOFT: vmov [[REG:d[0-9]+]], r1, r0
; SOFT: vrev64.32 [[REG]]
; HARD: vrev64.32 d{{[0-9]+}}, d0
    %1 = fadd <2 x float> %p, %p
    %2 = bitcast <2 x float> %1 to <4 x i16>
    %3 = add <4 x i16> %2, %2
    ret <4 x i16> %3
; SOFT: vrev64.16 [[REG:d[0-9]+]]
; SOFT: vmov r1, r0, [[REG]]
; HARD: vrev64.16 d0
}

; CHECK-LABEL: test_v4i16_v2i32:
define <4 x i16> @test_v4i16_v2i32(<2 x i32> %p) {
; SOFT: vmov [[REG:d[0-9]+]], r1, r0
; SOFT: vrev64.32 [[REG]]
; HARD: vrev64.32 d{{[0-9]+}}, d0
    %1 = add <2 x i32> %p, %p
    %2 = bitcast <2 x i32> %1 to <4 x i16>
    %3 = add <4 x i16> %2, %2
    ret <4 x i16> %3
; SOFT: vrev64.16 [[REG:d[0-9]+]]
; SOFT: vmov r1, r0, [[REG]]
; HARD: vrev64.16 d0
}

; CHECK-LABEL: test_v4i16_v8i8:
define <4 x i16> @test_v4i16_v8i8(<8 x i8> %p) {
; SOFT: vmov [[REG:d[0-9]+]], r1, r0
; SOFT: vrev64.8 [[REG]]
; HARD: vrev64.8 d{{[0-9]+}}, d0
    %1 = add <8 x i8> %p, %p
    %2 = bitcast <8 x i8> %1 to <4 x i16>
    %3 = add <4 x i16> %2, %2
    ret <4 x i16> %3
; SOFT: vrev64.16 [[REG:d[0-9]+]]
; SOFT: vmov r1, r0, [[REG]]
; HARD: vrev64.16 d0
}

; CHECK-LABEL: test_v8i8_i64:
define <8 x i8> @test_v8i8_i64(i64 %p) {
; CHECK: adds r1
; CHECK: adc r0
    %1 = add i64 %p, %p
    %2 = bitcast i64 %1 to <8 x i8>
    %3 = add <8 x i8> %2, %2
    ret <8 x i8> %3
; SOFT: vrev64.8 [[REG:d[0-9]+]]
; SOFT: vmov r1, r0, [[REG]]
; HARD: vrev64.8 d0
}

; CHECK-LABEL: test_v8i8_f64:
define <8 x i8> @test_v8i8_f64(double %p) {
; SOFT: vmov [[REG:d[0-9]+]], r1, r0
; SOFT: vadd.f64 [[REG]]
; HARD: vadd.f64 d{{[0-9]+}}, d0
    %1 = fadd double %p, %p
    %2 = bitcast double %1 to <8 x i8>
    %3 = add <8 x i8> %2, %2
    ret <8 x i8> %3
; SOFT: vrev64.8 [[REG:d[0-9]+]]
; SOFT: vmov r1, r0, [[REG]]
; HARD: vrev64.8 d0
}

; CHECK-LABEL: test_v8i8_v1i64:
define <8 x i8> @test_v8i8_v1i64(<1 x i64> %p) {
; SOFT: vmov [[REG:d[0-9]+]], r1, r0
; SOFT: vadd.i64 [[REG]]
; HARD: vadd.i64 d{{[0-9]+}}, d0
    %1 = add <1 x i64> %p, %p
    %2 = bitcast <1 x i64> %1 to <8 x i8>
    %3 = add <8 x i8> %2, %2
    ret <8 x i8> %3
; SOFT: vrev64.8 [[REG:d[0-9]+]]
; SOFT: vmov r1, r0, [[REG]]
; HARD: vrev64.8 d0
}

; CHECK-LABEL: test_v8i8_v2f32:
define <8 x i8> @test_v8i8_v2f32(<2 x float> %p) {
; SOFT: vmov [[REG:d[0-9]+]], r1, r0
; SOFT: vrev64.32 [[REG]]
; HARD: vrev64.32 d{{[0-9]+}}, d0
    %1 = fadd <2 x float> %p, %p
    %2 = bitcast <2 x float> %1 to <8 x i8>
    %3 = add <8 x i8> %2, %2
    ret <8 x i8> %3
; SOFT: vrev64.8 [[REG:d[0-9]+]]
; SOFT: vmov r1, r0, [[REG]]
; HARD: vrev64.8 d0
}

; CHECK-LABEL: test_v8i8_v2i32:
define <8 x i8> @test_v8i8_v2i32(<2 x i32> %p) {
; SOFT: vmov [[REG:d[0-9]+]], r1, r0
; SOFT: vrev64.32 [[REG]]
; HARD: vrev64.32 d{{[0-9]+}}, d0
    %1 = add <2 x i32> %p, %p
    %2 = bitcast <2 x i32> %1 to <8 x i8>
    %3 = add <8 x i8> %2, %2
    ret <8 x i8> %3
; SOFT: vrev64.8 [[REG:d[0-9]+]]
; SOFT: vmov r1, r0, [[REG]]
; HARD: vrev64.8 d0
}

; CHECK-LABEL: test_v8i8_v4i16:
define <8 x i8> @test_v8i8_v4i16(<4 x i16> %p) {
; SOFT: vmov [[REG:d[0-9]+]], r1, r0
; SOFT: vrev64.16 [[REG]]
; HARD: vrev64.16 d{{[0-9]+}}, d0
    %1 = add <4 x i16> %p, %p
    %2 = bitcast <4 x i16> %1 to <8 x i8>
    %3 = add <8 x i8> %2, %2
    ret <8 x i8> %3
; SOFT: vrev64.8 [[REG:d[0-9]+]]
; SOFT: vmov r1, r0, [[REG]]
; HARD: vrev64.8 d0
}

; CHECK-LABEL: test_f128_v2f64:
define fp128 @test_f128_v2f64(<2 x double> %p) {
; SOFT: vmov [[REG1:d[0-9]+]], r3, r2
; SOFT: vmov [[REG2:d[0-9]+]], r1, r0
; SOFT: vadd.f64 d{{[0-9]+}}, [[REG1]]
; SOFT: vadd.f64 d{{[0-9]+}}, [[REG2]]
; HARD: vadd.f64 d{{[0-9]+}}, d1
; HARD: vadd.f64 d{{[0-9]+}}, d0
    %1 = fadd <2 x double> %p, %p
    %2 = bitcast <2 x double> %1 to fp128
    %3 = fadd fp128 %2, %2
    ret fp128 %3
; CHECK: vst1.32 {d{{[0-9]+}}[1]}, [{{[a-z0-9]+}}:32]
; CHECK: vst1.32 {d{{[0-9]+}}[0]}, [{{[a-z0-9]+}}:32]
}

; CHECK-LABEL: test_f128_v2i64:
define fp128 @test_f128_v2i64(<2 x i64> %p) {
; SOFT: vmov [[REG1:d[0-9]+]], r3, r2
; SOFT: vmov [[REG2:d[0-9]+]], r1, r0
; HARD: vadd.i64 q{{[0-9]+}}, q0
    %1 = add <2 x i64> %p, %p
    %2 = bitcast <2 x i64> %1 to fp128
    %3 = fadd fp128 %2, %2
    ret fp128 %3
; CHECK: vst1.32 {d{{[0-9]+}}[1]}, [{{[a-z0-9]+}}:32]
; CHECK: vst1.32 {d{{[0-9]+}}[0]}, [{{[a-z0-9]+}}:32]
}

; CHECK-LABEL: test_f128_v4f32:
define fp128 @test_f128_v4f32(<4 x float> %p) {
; HARD: vrev64.32 q{{[0-9]+}}, q0
    %1 = fadd <4 x float> %p, %p
    %2 = bitcast <4 x float> %1 to fp128
    %3 = fadd fp128 %2, %2
    ret fp128 %3
; CHECK: vst1.32 {d{{[0-9]+}}[1]}, [{{[a-z0-9]+}}:32]
; CHECK: vst1.32 {d{{[0-9]+}}[0]}, [{{[a-z0-9]+}}:32]
}

; CHECK-LABEL: test_f128_v4i32:
define fp128 @test_f128_v4i32(<4 x i32> %p) {
; HARD: vrev64.32 q{{[0-9]+}}, q0
    %1 = add <4 x i32> %p, %p
    %2 = bitcast <4 x i32> %1 to fp128
    %3 = fadd fp128 %2, %2
    ret fp128 %3
; CHECK: vst1.32 {d{{[0-9]+}}[1]}, [{{[a-z0-9]+}}:32]
; CHECK: vst1.32 {d{{[0-9]+}}[0]}, [{{[a-z0-9]+}}:32]
}

; CHECK-LABEL: test_f128_v8i16:
define fp128 @test_f128_v8i16(<8 x i16> %p) {
; HARD: vrev64.16 q{{[0-9]+}}, q0
    %1 = add <8 x i16> %p, %p
    %2 = bitcast <8 x i16> %1 to fp128
    %3 = fadd fp128 %2, %2
    ret fp128 %3
; CHECK: vst1.32 {d{{[0-9]+}}[1]}, [{{[a-z0-9]+}}:32]
; CHECK: vst1.32 {d{{[0-9]+}}[0]}, [{{[a-z0-9]+}}:32]
}

; CHECK-LABEL: test_f128_v16i8:
define fp128 @test_f128_v16i8(<16 x i8> %p) {
; HARD: vrev64.8 q{{[0-9]+}}, q0
    %1 = add <16 x i8> %p, %p
    %2 = bitcast <16 x i8> %1 to fp128
    %3 = fadd fp128 %2, %2
    ret fp128 %3
; CHECK: vst1.32 {d{{[0-9]+}}[1]}, [{{[a-z0-9]+}}:32]
; CHECK: vst1.32 {d{{[0-9]+}}[0]}, [{{[a-z0-9]+}}:32]
}

; CHECK-LABEL: test_v2f64_f128:
define <2 x double> @test_v2f64_f128(fp128 %p) {
; CHECK: vmov.32 [[REG2:d[0-9]+]][0], r2
; CHECK: vmov.32 [[REG1:d[0-9]+]][0], r0
; CHECK: vmov.32 [[REG2]][1], r3
; CHECK: vmov.32 [[REG1]][1], r1
    %1 = fadd fp128 %p, %p
    %2 = bitcast fp128 %1 to <2 x double>
    %3 = fadd <2 x double> %2, %2
    ret <2 x double> %3
; SOFT: vadd.f64 [[REG1:d[0-9]+]]
; SOFT: vadd.f64 [[REG2:d[0-9]+]]
; SOFT: vmov r1, r0, [[REG2]]
; SOFT: vmov r3, r2, [[REG1]]
; HARD: vadd.f64 d1
; HARD: vadd.f64 d0
}

; CHECK-LABEL: test_v2f64_v2i64:
define <2 x double> @test_v2f64_v2i64(<2 x i64> %p) {
; SOFT: vmov [[REG1:d[0-9]+]], r3, r2
; SOFT: vmov [[REG2:d[0-9]+]], r1, r0
; HARD: vadd.i64 q{{[0-9]+}}, q0
    %1 = add <2 x i64> %p, %p
    %2 = bitcast <2 x i64> %1 to <2 x double>
    %3 = fadd <2 x double> %2, %2
    ret <2 x double> %3
; SOFT: vadd.f64 [[REG1:d[0-9]+]]
; SOFT: vadd.f64 [[REG2:d[0-9]+]]
; SOFT: vmov r1, r0, [[REG2]]
; SOFT: vmov r3, r2, [[REG1]]
; HARD: vadd.f64 d1
; HARD: vadd.f64 d0
}

; CHECK-LABEL: test_v2f64_v4f32:
define <2 x double> @test_v2f64_v4f32(<4 x float> %p) {
; HARD: vrev64.32  q{{[0-9]+}}, q0
    %1 = fadd <4 x float> %p, %p
    %2 = bitcast <4 x float> %1 to <2 x double>
    %3 = fadd <2 x double> %2, %2
    ret <2 x double> %3
; SOFT: vadd.f64 [[REG1:d[0-9]+]]
; SOFT: vadd.f64 [[REG2:d[0-9]+]]
; SOFT: vmov r1, r0, [[REG2]]
; SOFT: vmov r3, r2, [[REG1]]
; HARD: vadd.f64 d1
; HARD: vadd.f64 d0
}

; CHECK-LABEL: test_v2f64_v4i32:
define <2 x double> @test_v2f64_v4i32(<4 x i32> %p) {
; HARD: vrev64.32  q{{[0-9]+}}, q0
    %1 = add <4 x i32> %p, %p
    %2 = bitcast <4 x i32> %1 to <2 x double>
    %3 = fadd <2 x double> %2, %2
    ret <2 x double> %3
; SOFT: vadd.f64 [[REG1:d[0-9]+]]
; SOFT: vadd.f64 [[REG2:d[0-9]+]]
; SOFT: vmov r1, r0, [[REG2]]
; SOFT: vmov r3, r2, [[REG1]]
; HARD: vadd.f64 d1
; HARD: vadd.f64 d0
}

; CHECK-LABEL: test_v2f64_v8i16:
define <2 x double> @test_v2f64_v8i16(<8 x i16> %p) {
; HARD: vrev64.16  q{{[0-9]+}}, q0
    %1 = add <8 x i16> %p, %p
    %2 = bitcast <8 x i16> %1 to <2 x double>
    %3 = fadd <2 x double> %2, %2
    ret <2 x double> %3
; SOFT: vadd.f64 [[REG1:d[0-9]+]]
; SOFT: vadd.f64 [[REG2:d[0-9]+]]
; SOFT: vmov r1, r0, [[REG2]]
; SOFT: vmov r3, r2, [[REG1]]
; HARD: vadd.f64 d1
; HARD: vadd.f64 d0
}

; CHECK-LABEL: test_v2f64_v16i8:
define <2 x double> @test_v2f64_v16i8(<16 x i8> %p) {
; HARD: vrev64.8  q{{[0-9]+}}, q0
    %1 = add <16 x i8> %p, %p
    %2 = bitcast <16 x i8> %1 to <2 x double>
    %3 = fadd <2 x double> %2, %2
    ret <2 x double> %3
; SOFT: vadd.f64 [[REG1:d[0-9]+]]
; SOFT: vadd.f64 [[REG2:d[0-9]+]]
; SOFT: vmov r1, r0, [[REG2]]
; SOFT: vmov r3, r2, [[REG1]]
; HARD: vadd.f64 d1
; HARD: vadd.f64 d0
}

; CHECK-LABEL: test_v2i64_f128:
define <2 x i64> @test_v2i64_f128(fp128 %p) {
; CHECK: vmov.32 [[REG2:d[0-9]+]][0], r2
; CHECK: vmov.32 [[REG1:d[0-9]+]][0], r0
; CHECK: vmov.32 [[REG2]][1], r3
; CHECK: vmov.32 [[REG1]][1], r1
    %1 = fadd fp128 %p, %p
    %2 = bitcast fp128 %1 to <2 x i64>
    %3 = add <2 x i64> %2, %2
    ret <2 x i64> %3
; SOFT: vmov r1, r0
; SOFT: vmov r3, r2
; HARD: vadd.i64 q0
}

; CHECK-LABEL: test_v2i64_v2f64:
define <2 x i64> @test_v2i64_v2f64(<2 x double> %p) {
; SOFT: vmov [[REG1:d[0-9]+]], r3, r2
; SOFT: vmov [[REG2:d[0-9]+]], r1, r0
; SOFT: vadd.f64 d{{[0-9]+}}, [[REG1]]
; SOFT: vadd.f64 d{{[0-9]+}}, [[REG2]]
; HARD: vadd.f64  d{{[0-9]+}}, d1
; HARD: vadd.f64  d{{[0-9]+}}, d0
    %1 = fadd <2 x double> %p, %p
    %2 = bitcast <2 x double> %1 to <2 x i64>
    %3 = add <2 x i64> %2, %2
    ret <2 x i64> %3
; SOFT: vmov r1, r0
; SOFT: vmov r3, r2
; HARD: vadd.i64 q0
}

; CHECK-LABEL: test_v2i64_v4f32:
define <2 x i64> @test_v2i64_v4f32(<4 x float> %p) {
; HARD: vrev64.32  q{{[0-9]+}}, q0
    %1 = fadd <4 x float> %p, %p
    %2 = bitcast <4 x float> %1 to <2 x i64>
    %3 = add <2 x i64> %2, %2
    ret <2 x i64> %3
; SOFT: vmov r1, r0
; SOFT: vmov r3, r2
; HARD: vadd.i64 q0
}

; CHECK-LABEL: test_v2i64_v4i32:
define <2 x i64> @test_v2i64_v4i32(<4 x i32> %p) {
; HARD: vrev64.32  q{{[0-9]+}}, q0
    %1 = add <4 x i32> %p, %p
    %2 = bitcast <4 x i32> %1 to <2 x i64>
    %3 = add <2 x i64> %2, %2
    ret <2 x i64> %3
; SOFT: vmov r1, r0
; SOFT: vmov r3, r2
; HARD: vadd.i64 q0
}

; CHECK-LABEL: test_v2i64_v8i16:
define <2 x i64> @test_v2i64_v8i16(<8 x i16> %p) {
; HARD: vrev64.16  q{{[0-9]+}}, q0
    %1 = add <8 x i16> %p, %p
    %2 = bitcast <8 x i16> %1 to <2 x i64>
    %3 = add <2 x i64> %2, %2
    ret <2 x i64> %3
; SOFT: vmov r1, r0
; SOFT: vmov r3, r2
; HARD: vadd.i64 q0
}

; CHECK-LABEL: test_v2i64_v16i8:
define <2 x i64> @test_v2i64_v16i8(<16 x i8> %p) {
; HARD: vrev64.8  q{{[0-9]+}}, q0
    %1 = add <16 x i8> %p, %p
    %2 = bitcast <16 x i8> %1 to <2 x i64>
    %3 = add <2 x i64> %2, %2
    ret <2 x i64> %3
; SOFT: vmov r1, r0
; SOFT: vmov r3, r2
; HARD: vadd.i64 q0
}

; CHECK-LABEL: test_v4f32_f128:
define <4 x float> @test_v4f32_f128(fp128 %p) {
; CHECK: vmov.32 [[REG2:d[0-9]+]][0], r2
; CHECK: vmov.32 [[REG1:d[0-9]+]][0], r0
; CHECK: vmov.32 [[REG2]][1], r3
; CHECK: vmov.32 [[REG1]][1], r1
    %1 = fadd fp128 %p, %p
    %2 = bitcast fp128 %1 to <4 x float>
    %3 = fadd <4 x float> %2, %2
    ret <4 x float> %3
; SOFT: vmov r1, r0
; SOFT: vmov r3, r2
; HARD: vrev64.32 q0
}

; CHECK-LABEL: test_v4f32_v2f64:
define <4 x float> @test_v4f32_v2f64(<2 x double> %p) {
; SOFT: vmov [[REG1:d[0-9]+]], r3, r2
; SOFT: vmov [[REG2:d[0-9]+]], r1, r0
; SOFT: vadd.f64 d{{[0-9]+}}, [[REG1]]
; SOFT: vadd.f64 d{{[0-9]+}}, [[REG2]]
; HARD: vadd.f64  d{{[0-9]+}}, d1
; HARD: vadd.f64  d{{[0-9]+}}, d0
    %1 = fadd <2 x double> %p, %p
    %2 = bitcast <2 x double> %1 to <4 x float>
    %3 = fadd <4 x float> %2, %2
    ret <4 x float> %3
; SOFT: vmov r1, r0
; SOFT: vmov r3, r2
; HARD: vrev64.32 q0
}

; CHECK-LABEL: test_v4f32_v2i64:
define <4 x float> @test_v4f32_v2i64(<2 x i64> %p) {
; SOFT: vmov [[REG1:d[0-9]+]], r3, r2
; SOFT: vmov [[REG2:d[0-9]+]], r1, r0
; HARD: vadd.i64  q{{[0-9]+}}, q0
    %1 = add <2 x i64> %p, %p
    %2 = bitcast <2 x i64> %1 to <4 x float>
    %3 = fadd <4 x float> %2, %2
    ret <4 x float> %3
; SOFT: vmov r1, r0
; SOFT: vmov r3, r2
; HARD: vrev64.32 q0
}

; CHECK-LABEL: test_v4f32_v4i32:
define <4 x float> @test_v4f32_v4i32(<4 x i32> %p) {
; HARD: vrev64.32  q{{[0-9]+}}, q0
    %1 = add <4 x i32> %p, %p
    %2 = bitcast <4 x i32> %1 to <4 x float>
    %3 = fadd <4 x float> %2, %2
    ret <4 x float> %3
; SOFT: vmov r1, r0
; SOFT: vmov r3, r2
; HARD: vrev64.32 q0
}

; CHECK-LABEL: test_v4f32_v8i16:
define <4 x float> @test_v4f32_v8i16(<8 x i16> %p) {
; HARD: vrev64.16  q{{[0-9]+}}, q0
    %1 = add <8 x i16> %p, %p
    %2 = bitcast <8 x i16> %1 to <4 x float>
    %3 = fadd <4 x float> %2, %2
    ret <4 x float> %3
; SOFT: vmov r1, r0
; SOFT: vmov r3, r2
; HARD: vrev64.32 q0
}

; CHECK-LABEL: test_v4f32_v16i8:
define <4 x float> @test_v4f32_v16i8(<16 x i8> %p) {
; HARD: vrev64.8  q{{[0-9]+}}, q0
    %1 = add <16 x i8> %p, %p
    %2 = bitcast <16 x i8> %1 to <4 x float>
    %3 = fadd <4 x float> %2, %2
    ret <4 x float> %3
; SOFT: vmov r1, r0
; SOFT: vmov r3, r2
; HARD: vrev64.32 q0
}

; CHECK-LABEL: test_v4i32_f128:
define <4 x i32> @test_v4i32_f128(fp128 %p) {
; CHECK: vmov.32 [[REG2:d[0-9]+]][0], r2
; CHECK: vmov.32 [[REG1:d[0-9]+]][0], r0
; CHECK: vmov.32 [[REG2]][1], r3
; CHECK: vmov.32 [[REG1]][1], r1
    %1 = fadd fp128 %p, %p
    %2 = bitcast fp128 %1 to <4 x i32>
    %3 = add <4 x i32> %2, %2
    ret <4 x i32> %3
; SOFT: vmov r1, r0
; SOFT: vmov r3, r2
; HARD: vrev64.32 q0
}

; CHECK-LABEL: test_v4i32_v2f64:
define <4 x i32> @test_v4i32_v2f64(<2 x double> %p) {
; SOFT: vmov [[REG1:d[0-9]+]], r3, r2
; SOFT: vmov [[REG2:d[0-9]+]], r1, r0
; SOFT: vadd.f64 d{{[0-9]+}}, [[REG1]]
; SOFT: vadd.f64 d{{[0-9]+}}, [[REG2]]
; HARD: vadd.f64  d{{[0-9]+}}, d1
; HARD: vadd.f64  d{{[0-9]+}}, d0
    %1 = fadd <2 x double> %p, %p
    %2 = bitcast <2 x double> %1 to <4 x i32>
    %3 = add <4 x i32> %2, %2
    ret <4 x i32> %3
; SOFT: vmov r1, r0
; SOFT: vmov r3, r2
; HARD: vrev64.32 q0
}

; CHECK-LABEL: test_v4i32_v2i64:
define <4 x i32> @test_v4i32_v2i64(<2 x i64> %p) {
; SOFT: vmov [[REG1:d[0-9]+]], r3, r2
; SOFT: vmov [[REG2:d[0-9]+]], r1, r0
; HARD: vadd.i64  q{{[0-9]+}}, q0
    %1 = add <2 x i64> %p, %p
    %2 = bitcast <2 x i64> %1 to <4 x i32>
    %3 = add <4 x i32> %2, %2
    ret <4 x i32> %3
; SOFT: vmov r1, r0
; SOFT: vmov r3, r2
; HARD: vrev64.32 q0
}

; CHECK-LABEL: test_v4i32_v4f32:
define <4 x i32> @test_v4i32_v4f32(<4 x float> %p) {
; SOFT: vmov [[REG1:d[0-9]+]], r3, r2
; SOFT: vmov [[REG2:d[0-9]+]], r1, r0
; HARD: vrev64.32  q{{[0-9]+}}, q0
    %1 = fadd <4 x float> %p, %p
    %2 = bitcast <4 x float> %1 to <4 x i32>
    %3 = add <4 x i32> %2, %2
    ret <4 x i32> %3
; SOFT: vmov r1, r0
; SOFT: vmov r3, r2
; HARD: vrev64.32 q0
}

; CHECK-LABEL: test_v4i32_v8i16:
define <4 x i32> @test_v4i32_v8i16(<8 x i16> %p) {
; SOFT: vmov [[REG1:d[0-9]+]], r3, r2
; SOFT: vmov [[REG2:d[0-9]+]], r1, r0
; HARD: vrev64.16  q{{[0-9]+}}, q0
    %1 = add <8 x i16> %p, %p
    %2 = bitcast <8 x i16> %1 to <4 x i32>
    %3 = add <4 x i32> %2, %2
    ret <4 x i32> %3
; SOFT: vmov r1, r0
; SOFT: vmov r3, r2
; HARD: vrev64.32 q0
}

; CHECK-LABEL: test_v4i32_v16i8:
define <4 x i32> @test_v4i32_v16i8(<16 x i8> %p) {
; SOFT: vmov [[REG1:d[0-9]+]], r3, r2
; SOFT: vmov [[REG2:d[0-9]+]], r1, r0
; HARD: vrev64.8  q{{[0-9]+}}, q0
    %1 = add <16 x i8> %p, %p
    %2 = bitcast <16 x i8> %1 to <4 x i32>
    %3 = add <4 x i32> %2, %2
    ret <4 x i32> %3
; SOFT: vmov r1, r0
; SOFT: vmov r3, r2
; HARD: vrev64.32 q0
}

; CHECK-LABEL: test_v8i16_f128:
define <8 x i16> @test_v8i16_f128(fp128 %p) {
; CHECK: vmov.32 [[REG2:d[0-9]+]][0], r2
; CHECK: vmov.32 [[REG1:d[0-9]+]][0], r0
; CHECK: vmov.32 [[REG2]][1], r3
; CHECK: vmov.32 [[REG1]][1], r1
    %1 = fadd fp128 %p, %p
    %2 = bitcast fp128 %1 to <8 x i16>
    %3 = add <8 x i16> %2, %2
    ret <8 x i16> %3
; SOFT: vmov r1, r0
; SOFT: vmov r3, r2
; HARD: vrev64.16 q0
}

; CHECK-LABEL: test_v8i16_v2f64:
define <8 x i16> @test_v8i16_v2f64(<2 x double> %p) {
; SOFT: vmov [[REG1:d[0-9]+]], r3, r2
; SOFT: vmov [[REG2:d[0-9]+]], r1, r0
; SOFT: vadd.f64 d{{[0-9]+}}, [[REG1]]
; SOFT: vadd.f64 d{{[0-9]+}}, [[REG2]]
; HARD: vadd.f64  d{{[0-9]+}}, d1
; HARD: vadd.f64  d{{[0-9]+}}, d0
    %1 = fadd <2 x double> %p, %p
    %2 = bitcast <2 x double> %1 to <8 x i16>
    %3 = add <8 x i16> %2, %2
    ret <8 x i16> %3
; SOFT: vmov r1, r0
; SOFT: vmov r3, r2
; HARD: vrev64.16 q0
}

; CHECK-LABEL: test_v8i16_v2i64:
define <8 x i16> @test_v8i16_v2i64(<2 x i64> %p) {
; SOFT: vmov [[REG1:d[0-9]+]], r3, r2
; SOFT: vmov [[REG2:d[0-9]+]], r1, r0
; HARD: vadd.i64  q{{[0-9]+}}, q0
    %1 = add <2 x i64> %p, %p
    %2 = bitcast <2 x i64> %1 to <8 x i16>
    %3 = add <8 x i16> %2, %2
    ret <8 x i16> %3
; SOFT: vmov r1, r0
; SOFT: vmov r3, r2
; HARD: vrev64.16 q0
}

; CHECK-LABEL: test_v8i16_v4f32:
define <8 x i16> @test_v8i16_v4f32(<4 x float> %p) {
; SOFT: vmov [[REG1:d[0-9]+]], r3, r2
; SOFT: vmov [[REG2:d[0-9]+]], r1, r0
; HARD: vrev64.32  q{{[0-9]+}}, q0
    %1 = fadd <4 x float> %p, %p
    %2 = bitcast <4 x float> %1 to <8 x i16>
    %3 = add <8 x i16> %2, %2
    ret <8 x i16> %3
; SOFT: vmov r1, r0
; SOFT: vmov r3, r2
; HARD: vrev64.16 q0
}

; CHECK-LABEL: test_v8i16_v4i32:
define <8 x i16> @test_v8i16_v4i32(<4 x i32> %p) {
; SOFT: vmov [[REG1:d[0-9]+]], r3, r2
; SOFT: vmov [[REG2:d[0-9]+]], r1, r0
; HARD: vrev64.32  q{{[0-9]+}}, q0
    %1 = add <4 x i32> %p, %p
    %2 = bitcast <4 x i32> %1 to <8 x i16>
    %3 = add <8 x i16> %2, %2
    ret <8 x i16> %3
; SOFT: vmov r1, r0
; SOFT: vmov r3, r2
; HARD: vrev64.16 q0
}

; CHECK-LABEL: test_v8i16_v16i8:
define <8 x i16> @test_v8i16_v16i8(<16 x i8> %p) {
; SOFT: vmov [[REG1:d[0-9]+]], r3, r2
; SOFT: vmov [[REG2:d[0-9]+]], r1, r0
; HARD: vrev64.8 q{{[0-9]+}}, q0
    %1 = add <16 x i8> %p, %p
    %2 = bitcast <16 x i8> %1 to <8 x i16>
    %3 = add <8 x i16> %2, %2
    ret <8 x i16> %3
; SOFT: vmov r1, r0
; SOFT: vmov r3, r2
; HARD: vrev64.16 q0
}

; CHECK-LABEL: test_v16i8_f128:
define <16 x i8> @test_v16i8_f128(fp128 %p) {
; CHECK: vmov.32 [[REG2:d[0-9]+]][0], r2
; CHECK: vmov.32 [[REG1:d[0-9]+]][0], r0
; CHECK: vmov.32 [[REG2]][1], r3
; CHECK: vmov.32 [[REG1]][1], r1
    %1 = fadd fp128 %p, %p
    %2 = bitcast fp128 %1 to <16 x i8>
    %3 = add <16 x i8> %2, %2
    ret <16 x i8> %3
; SOFT: vmov r1, r0
; SOFT: vmov r3, r2
; HARD: vrev64.8 q0
}

; CHECK-LABEL: test_v16i8_v2f64:
define <16 x i8> @test_v16i8_v2f64(<2 x double> %p) {
; SOFT: vmov [[REG1:d[0-9]+]], r3, r2
; SOFT: vmov [[REG2:d[0-9]+]], r1, r0
; SOFT: vadd.f64 d{{[0-9]+}}, [[REG1]]
; SOFT: vadd.f64 d{{[0-9]+}}, [[REG2]]
; HARD: vadd.f64  d{{[0-9]+}}, d1
; HARD: vadd.f64  d{{[0-9]+}}, d0
    %1 = fadd <2 x double> %p, %p
    %2 = bitcast <2 x double> %1 to <16 x i8>
    %3 = add <16 x i8> %2, %2
    ret <16 x i8> %3
; SOFT: vmov r1, r0
; SOFT: vmov r3, r2
; HARD: vrev64.8 q0
}

; CHECK-LABEL: test_v16i8_v2i64:
define <16 x i8> @test_v16i8_v2i64(<2 x i64> %p) {
; SOFT: vmov [[REG1:d[0-9]+]], r3, r2
; SOFT: vmov [[REG2:d[0-9]+]], r1, r0
; HARD: vadd.i64  q{{[0-9]+}}, q0
    %1 = add <2 x i64> %p, %p
    %2 = bitcast <2 x i64> %1 to <16 x i8>
    %3 = add <16 x i8> %2, %2
    ret <16 x i8> %3
; SOFT: vmov r1, r0
; SOFT: vmov r3, r2
; HARD: vrev64.8 q0
}

; CHECK-LABEL: test_v16i8_v4f32:
define <16 x i8> @test_v16i8_v4f32(<4 x float> %p) {
; SOFT: vmov [[REG1:d[0-9]+]], r3, r2
; SOFT: vmov [[REG2:d[0-9]+]], r1, r0
; HARD: vrev64.32 q{{[0-9]+}}, q0
    %1 = fadd <4 x float> %p, %p
    %2 = bitcast <4 x float> %1 to <16 x i8>
    %3 = add <16 x i8> %2, %2
    ret <16 x i8> %3
; SOFT: vmov r1, r0
; SOFT: vmov r3, r2
; HARD: vrev64.8 q0
}

; CHECK-LABEL: test_v16i8_v4i32:
define <16 x i8> @test_v16i8_v4i32(<4 x i32> %p) {
; SOFT: vmov [[REG1:d[0-9]+]], r3, r2
; SOFT: vmov [[REG2:d[0-9]+]], r1, r0
; HARD: vrev64.32 q{{[0-9]+}}, q0
    %1 = add <4 x i32> %p, %p
    %2 = bitcast <4 x i32> %1 to <16 x i8>
    %3 = add <16 x i8> %2, %2
    ret <16 x i8> %3
; SOFT: vmov r1, r0
; SOFT: vmov r3, r2
; HARD: vrev64.8 q0
}

; CHECK-LABEL: test_v16i8_v8i16:
define <16 x i8> @test_v16i8_v8i16(<8 x i16> %p) {
; SOFT: vmov [[REG1:d[0-9]+]], r3, r2
; SOFT: vmov [[REG2:d[0-9]+]], r1, r0
; HARD: vrev64.16 q{{[0-9]+}}, q0
    %1 = add <8 x i16> %p, %p
    %2 = bitcast <8 x i16> %1 to <16 x i8>
    %3 = add <16 x i8> %2, %2
    ret <16 x i8> %3
; SOFT: vmov r1, r0
; SOFT: vmov r3, r2
; HARD: vrev64.8 q0
}

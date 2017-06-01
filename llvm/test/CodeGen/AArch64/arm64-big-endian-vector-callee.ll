; RUN: llc -mtriple aarch64_be < %s -aarch64-enable-ldst-opt=false -o - | FileCheck %s
; RUN: llc -mtriple aarch64_be < %s -fast-isel=true -aarch64-enable-ldst-opt=false -o - | FileCheck %s

; CHECK-LABEL: test_i64_f64:
define i64 @test_i64_f64(double %p) {
; CHECK-NOT: rev
    %1 = fadd double %p, %p
    %2 = bitcast double %1 to i64
    %3 = add i64 %2, %2
    ret i64 %3
}

; CHECK-LABEL: test_i64_v1i64:
define i64 @test_i64_v1i64(<1 x i64> %p) {
; CHECK-NOT: rev
    %1 = add <1 x i64> %p, %p
    %2 = bitcast <1 x i64> %1 to i64
    %3 = add i64 %2, %2
    ret i64 %3
}

; CHECK-LABEL: test_i64_v2f32:
define i64 @test_i64_v2f32(<2 x float> %p) {
; CHECK: rev64 v{{[0-9]+}}.2s
    %1 = fadd <2 x float> %p, %p
    %2 = bitcast <2 x float> %1 to i64
    %3 = add i64 %2, %2
    ret i64 %3
}

; CHECK-LABEL: test_i64_v2i32:
define i64 @test_i64_v2i32(<2 x i32> %p) {
; CHECK: rev64 v{{[0-9]+}}.2s
    %1 = add <2 x i32> %p, %p
    %2 = bitcast <2 x i32> %1 to i64
    %3 = add i64 %2, %2
    ret i64 %3
}

; CHECK-LABEL: test_i64_v4i16:
define i64 @test_i64_v4i16(<4 x i16> %p) {
; CHECK: rev64 v{{[0-9]+}}.4h
    %1 = add <4 x i16> %p, %p
    %2 = bitcast <4 x i16> %1 to i64
    %3 = add i64 %2, %2
    ret i64 %3
}

; CHECK-LABEL: test_i64_v8i8:
define i64 @test_i64_v8i8(<8 x i8> %p) {
; CHECK: rev64 v{{[0-9]+}}.8b
    %1 = add <8 x i8> %p, %p
    %2 = bitcast <8 x i8> %1 to i64
    %3 = add i64 %2, %2
    ret i64 %3
}

; CHECK-LABEL: test_f64_i64:
define double @test_f64_i64(i64 %p) {
; CHECK-NOT: rev
    %1 = add i64 %p, %p
    %2 = bitcast i64 %1 to double
    %3 = fadd double %2, %2
    ret double %3
}

; CHECK-LABEL: test_f64_v1i64:
define double @test_f64_v1i64(<1 x i64> %p) {
; CHECK-NOT: rev
    %1 = add <1 x i64> %p, %p
    %2 = bitcast <1 x i64> %1 to double
    %3 = fadd double %2, %2
    ret double %3
}

; CHECK-LABEL: test_f64_v2f32:
define double @test_f64_v2f32(<2 x float> %p) {
; CHECK: rev64 v{{[0-9]+}}.2s
    %1 = fadd <2 x float> %p, %p
    %2 = bitcast <2 x float> %1 to double
    %3 = fadd double %2, %2
    ret double %3
}

; CHECK-LABEL: test_f64_v2i32:
define double @test_f64_v2i32(<2 x i32> %p) {
; CHECK: rev64 v{{[0-9]+}}.2s
    %1 = add <2 x i32> %p, %p
    %2 = bitcast <2 x i32> %1 to double
    %3 = fadd double %2, %2
    ret double %3
}

; CHECK-LABEL: test_f64_v4i16:
define double @test_f64_v4i16(<4 x i16> %p) {
; CHECK: rev64 v{{[0-9]+}}.4h
    %1 = add <4 x i16> %p, %p
    %2 = bitcast <4 x i16> %1 to double
    %3 = fadd double %2, %2
    ret double %3
}

; CHECK-LABEL: test_f64_v8i8:
define double @test_f64_v8i8(<8 x i8> %p) {
; CHECK: rev64 v{{[0-9]+}}.8b
    %1 = add <8 x i8> %p, %p
    %2 = bitcast <8 x i8> %1 to double
    %3 = fadd double %2, %2
    ret double %3
}

; CHECK-LABEL: test_v1i64_i64:
define <1 x i64> @test_v1i64_i64(i64 %p) {
; CHECK-NOT: rev
    %1 = add i64 %p, %p
    %2 = bitcast i64 %1 to <1 x i64>
    %3 = add <1 x i64> %2, %2
    ret <1 x i64> %3
}

; CHECK-LABEL: test_v1i64_f64:
define <1 x i64> @test_v1i64_f64(double %p) {
; CHECK-NOT: rev
    %1 = fadd double %p, %p
    %2 = bitcast double %1 to <1 x i64>
    %3 = add <1 x i64> %2, %2
    ret <1 x i64> %3
}

; CHECK-LABEL: test_v1i64_v2f32:
define <1 x i64> @test_v1i64_v2f32(<2 x float> %p) {
; CHECK: rev64 v{{[0-9]+}}.2s
    %1 = fadd <2 x float> %p, %p
    %2 = bitcast <2 x float> %1 to <1 x i64>
    %3 = add <1 x i64> %2, %2
    ret <1 x i64> %3
}

; CHECK-LABEL: test_v1i64_v2i32:
define <1 x i64> @test_v1i64_v2i32(<2 x i32> %p) {
; CHECK: rev64 v{{[0-9]+}}.2s
    %1 = add <2 x i32> %p, %p
    %2 = bitcast <2 x i32> %1 to <1 x i64>
    %3 = add <1 x i64> %2, %2
    ret <1 x i64> %3
}

; CHECK-LABEL: test_v1i64_v4i16:
define <1 x i64> @test_v1i64_v4i16(<4 x i16> %p) {
; CHECK: rev64 v{{[0-9]+}}.4h
    %1 = add <4 x i16> %p, %p
    %2 = bitcast <4 x i16> %1 to <1 x i64>
    %3 = add <1 x i64> %2, %2
    ret <1 x i64> %3
}

; CHECK-LABEL: test_v1i64_v8i8:
define <1 x i64> @test_v1i64_v8i8(<8 x i8> %p) {
; CHECK: rev64 v{{[0-9]+}}.8b
    %1 = add <8 x i8> %p, %p
    %2 = bitcast <8 x i8> %1 to <1 x i64>
    %3 = add <1 x i64> %2, %2
    ret <1 x i64> %3
}

; CHECK-LABEL: test_v2f32_i64:
define <2 x float> @test_v2f32_i64(i64 %p) {
; CHECK: rev64 v{{[0-9]+}}.2s
    %1 = add i64 %p, %p
    %2 = bitcast i64 %1 to <2 x float>
    %3 = fadd <2 x float> %2, %2
    ret <2 x float> %3
}

; CHECK-LABEL: test_v2f32_f64:
define <2 x float> @test_v2f32_f64(double %p) {
; CHECK: rev64 v{{[0-9]+}}.2s
    %1 = fadd double %p, %p
    %2 = bitcast double %1 to <2 x float>
    %3 = fadd <2 x float> %2, %2
    ret <2 x float> %3
}

; CHECK-LABEL: test_v2f32_v1i64:
define <2 x float> @test_v2f32_v1i64(<1 x i64> %p) {
; CHECK: rev64 v{{[0-9]+}}.2s
    %1 = add <1 x i64> %p, %p
    %2 = bitcast <1 x i64> %1 to <2 x float>
    %3 = fadd <2 x float> %2, %2
    ret <2 x float> %3
}

; CHECK-LABEL: test_v2f32_v2i32:
define <2 x float> @test_v2f32_v2i32(<2 x i32> %p) {
; CHECK: rev64 v{{[0-9]+}}.2s
; CHECK: rev64 v{{[0-9]+}}.2s
    %1 = add <2 x i32> %p, %p
    %2 = bitcast <2 x i32> %1 to <2 x float>
    %3 = fadd <2 x float> %2, %2
    ret <2 x float> %3
}

; CHECK-LABEL: test_v2f32_v4i16:
define <2 x float> @test_v2f32_v4i16(<4 x i16> %p) {
; CHECK: rev64 v{{[0-9]+}}.4h
; CHECK: rev64 v{{[0-9]+}}.2s
    %1 = add <4 x i16> %p, %p
    %2 = bitcast <4 x i16> %1 to <2 x float>
    %3 = fadd <2 x float> %2, %2
    ret <2 x float> %3
}

; CHECK-LABEL: test_v2f32_v8i8:
define <2 x float> @test_v2f32_v8i8(<8 x i8> %p) {
; CHECK: rev64 v{{[0-9]+}}.8b
; CHECK: rev64 v{{[0-9]+}}.2s
    %1 = add <8 x i8> %p, %p
    %2 = bitcast <8 x i8> %1 to <2 x float>
    %3 = fadd <2 x float> %2, %2
    ret <2 x float> %3
}

; CHECK-LABEL: test_v2i32_i64:
define <2 x i32> @test_v2i32_i64(i64 %p) {
; CHECK: rev64 v{{[0-9]+}}.2s
    %1 = add i64 %p, %p
    %2 = bitcast i64 %1 to <2 x i32>
    %3 = add <2 x i32> %2, %2
    ret <2 x i32> %3
}

; CHECK-LABEL: test_v2i32_f64:
define <2 x i32> @test_v2i32_f64(double %p) {
; CHECK: rev64 v{{[0-9]+}}.2s
    %1 = fadd double %p, %p
    %2 = bitcast double %1 to <2 x i32>
    %3 = add <2 x i32> %2, %2
    ret <2 x i32> %3
}

; CHECK-LABEL: test_v2i32_v1i64:
define <2 x i32> @test_v2i32_v1i64(<1 x i64> %p) {
; CHECK: rev64 v{{[0-9]+}}.2s
    %1 = add <1 x i64> %p, %p
    %2 = bitcast <1 x i64> %1 to <2 x i32>
    %3 = add <2 x i32> %2, %2
    ret <2 x i32> %3
}

; CHECK-LABEL: test_v2i32_v2f32:
define <2 x i32> @test_v2i32_v2f32(<2 x float> %p) {
; CHECK: rev64 v{{[0-9]+}}.2s
; CHECK: rev64 v{{[0-9]+}}.2s
    %1 = fadd <2 x float> %p, %p
    %2 = bitcast <2 x float> %1 to <2 x i32>
    %3 = add <2 x i32> %2, %2
    ret <2 x i32> %3
}

; CHECK-LABEL: test_v2i32_v4i16:
define <2 x i32> @test_v2i32_v4i16(<4 x i16> %p) {
; CHECK: rev64 v{{[0-9]+}}.4h
; CHECK: rev64 v{{[0-9]+}}.2s
    %1 = add <4 x i16> %p, %p
    %2 = bitcast <4 x i16> %1 to <2 x i32>
    %3 = add <2 x i32> %2, %2
    ret <2 x i32> %3
}

; CHECK-LABEL: test_v2i32_v8i8:
define <2 x i32> @test_v2i32_v8i8(<8 x i8> %p) {
; CHECK: rev64 v{{[0-9]+}}.8b
; CHECK: rev64 v{{[0-9]+}}.2s
    %1 = add <8 x i8> %p, %p
    %2 = bitcast <8 x i8> %1 to <2 x i32>
    %3 = add <2 x i32> %2, %2
    ret <2 x i32> %3
}

; CHECK-LABEL: test_v4i16_i64:
define <4 x i16> @test_v4i16_i64(i64 %p) {
; CHECK: rev64 v{{[0-9]+}}.4h
    %1 = add i64 %p, %p
    %2 = bitcast i64 %1 to <4 x i16>
    %3 = add <4 x i16> %2, %2
    ret <4 x i16> %3
}

; CHECK-LABEL: test_v4i16_f64:
define <4 x i16> @test_v4i16_f64(double %p) {
; CHECK: rev64 v{{[0-9]+}}.4h
    %1 = fadd double %p, %p
    %2 = bitcast double %1 to <4 x i16>
    %3 = add <4 x i16> %2, %2
    ret <4 x i16> %3
}

; CHECK-LABEL: test_v4i16_v1i64:
define <4 x i16> @test_v4i16_v1i64(<1 x i64> %p) {
; CHECK: rev64 v{{[0-9]+}}.4h
    %1 = add <1 x i64> %p, %p
    %2 = bitcast <1 x i64> %1 to <4 x i16>
    %3 = add <4 x i16> %2, %2
    ret <4 x i16> %3
}

; CHECK-LABEL: test_v4i16_v2f32:
define <4 x i16> @test_v4i16_v2f32(<2 x float> %p) {
; CHECK: rev64 v{{[0-9]+}}.2s
; CHECK: rev64 v{{[0-9]+}}.4h
    %1 = fadd <2 x float> %p, %p
    %2 = bitcast <2 x float> %1 to <4 x i16>
    %3 = add <4 x i16> %2, %2
    ret <4 x i16> %3
}

; CHECK-LABEL: test_v4i16_v2i32:
define <4 x i16> @test_v4i16_v2i32(<2 x i32> %p) {
; CHECK: rev64 v{{[0-9]+}}.2s
; CHECK: rev64 v{{[0-9]+}}.4h
    %1 = add <2 x i32> %p, %p
    %2 = bitcast <2 x i32> %1 to <4 x i16>
    %3 = add <4 x i16> %2, %2
    ret <4 x i16> %3
}

; CHECK-LABEL: test_v4i16_v8i8:
define <4 x i16> @test_v4i16_v8i8(<8 x i8> %p) {
; CHECK: rev64 v{{[0-9]+}}.8b
; CHECK: rev64 v{{[0-9]+}}.4h
    %1 = add <8 x i8> %p, %p
    %2 = bitcast <8 x i8> %1 to <4 x i16>
    %3 = add <4 x i16> %2, %2
    ret <4 x i16> %3
}

; CHECK-LABEL: test_v8i8_i64:
define <8 x i8> @test_v8i8_i64(i64 %p) {
; CHECK: rev64 v{{[0-9]+}}.8b
    %1 = add i64 %p, %p
    %2 = bitcast i64 %1 to <8 x i8>
    %3 = add <8 x i8> %2, %2
    ret <8 x i8> %3
}

; CHECK-LABEL: test_v8i8_f64:
define <8 x i8> @test_v8i8_f64(double %p) {
; CHECK: rev64 v{{[0-9]+}}.8b
    %1 = fadd double %p, %p
    %2 = bitcast double %1 to <8 x i8>
    %3 = add <8 x i8> %2, %2
    ret <8 x i8> %3
}

; CHECK-LABEL: test_v8i8_v1i64:
define <8 x i8> @test_v8i8_v1i64(<1 x i64> %p) {
; CHECK: rev64 v{{[0-9]+}}.8b
    %1 = add <1 x i64> %p, %p
    %2 = bitcast <1 x i64> %1 to <8 x i8>
    %3 = add <8 x i8> %2, %2
    ret <8 x i8> %3
}

; CHECK-LABEL: test_v8i8_v2f32:
define <8 x i8> @test_v8i8_v2f32(<2 x float> %p) {
; CHECK: rev64 v{{[0-9]+}}.2s
; CHECK: rev64 v{{[0-9]+}}.8b
    %1 = fadd <2 x float> %p, %p
    %2 = bitcast <2 x float> %1 to <8 x i8>
    %3 = add <8 x i8> %2, %2
    ret <8 x i8> %3
}

; CHECK-LABEL: test_v8i8_v2i32:
define <8 x i8> @test_v8i8_v2i32(<2 x i32> %p) {
; CHECK: rev64 v{{[0-9]+}}.2s
; CHECK: rev64 v{{[0-9]+}}.8b
    %1 = add <2 x i32> %p, %p
    %2 = bitcast <2 x i32> %1 to <8 x i8>
    %3 = add <8 x i8> %2, %2
    ret <8 x i8> %3
}

; CHECK-LABEL: test_v8i8_v4i16:
define <8 x i8> @test_v8i8_v4i16(<4 x i16> %p) {
; CHECK: rev64 v{{[0-9]+}}.4h
; CHECK: rev64 v{{[0-9]+}}.8b
    %1 = add <4 x i16> %p, %p
    %2 = bitcast <4 x i16> %1 to <8 x i8>
    %3 = add <8 x i8> %2, %2
    ret <8 x i8> %3
}

; CHECK-LABEL: test_f128_v2f64:
define fp128 @test_f128_v2f64(<2 x double> %p) {
; CHECK: ext
    %1 = fadd <2 x double> %p, %p
    %2 = bitcast <2 x double> %1 to fp128
    %3 = fadd fp128 %2, %2
    ret fp128 %3
}

; CHECK-LABEL: test_f128_v2i64:
define fp128 @test_f128_v2i64(<2 x i64> %p) {
; CHECK: ext
    %1 = add <2 x i64> %p, %p
    %2 = bitcast <2 x i64> %1 to fp128
    %3 = fadd fp128 %2, %2
    ret fp128 %3
}

; CHECK-LABEL: test_f128_v4f32:
define fp128 @test_f128_v4f32(<4 x float> %p) {
; CHECK: rev64 v{{[0-9]+}}.4s
; CHECK: ext
    %1 = fadd <4 x float> %p, %p
    %2 = bitcast <4 x float> %1 to fp128
    %3 = fadd fp128 %2, %2
    ret fp128 %3
}

; CHECK-LABEL: test_f128_v4i32:
define fp128 @test_f128_v4i32(<4 x i32> %p) {
; CHECK: rev64 v{{[0-9]+}}.4s
; CHECK: ext
    %1 = add <4 x i32> %p, %p
    %2 = bitcast <4 x i32> %1 to fp128
    %3 = fadd fp128 %2, %2
    ret fp128 %3
}

; CHECK-LABEL: test_f128_v8i16:
define fp128 @test_f128_v8i16(<8 x i16> %p) {
; CHECK: rev64 v{{[0-9]+}}.8h
; CHECK: ext
    %1 = add <8 x i16> %p, %p
    %2 = bitcast <8 x i16> %1 to fp128
    %3 = fadd fp128 %2, %2
    ret fp128 %3
}

; CHECK-LABEL: test_f128_v16i8:
define fp128 @test_f128_v16i8(<16 x i8> %p) {
; CHECK: rev64 v{{[0-9]+}}.16b
; CHECK: ext
    %1 = add <16 x i8> %p, %p
    %2 = bitcast <16 x i8> %1 to fp128
    %3 = fadd fp128 %2, %2
    ret fp128 %3
}

; CHECK-LABEL: test_v2f64_f128:
define <2 x double> @test_v2f64_f128(fp128 %p) {
; CHECK: ext
    %1 = fadd fp128 %p, %p
    %2 = bitcast fp128 %1 to <2 x double>
    %3 = fadd <2 x double> %2, %2
    ret <2 x double> %3
}

; CHECK-LABEL: test_v2f64_v2i64:
define <2 x double> @test_v2f64_v2i64(<2 x i64> %p) {
; CHECK: ext
; CHECK: ext
    %1 = add <2 x i64> %p, %p
    %2 = bitcast <2 x i64> %1 to <2 x double>
    %3 = fadd <2 x double> %2, %2
    ret <2 x double> %3
}

; CHECK-LABEL: test_v2f64_v4f32:
define <2 x double> @test_v2f64_v4f32(<4 x float> %p) {
; CHECK: rev64 v{{[0-9]+}}.4s
; CHECK: ext
; CHECK: ext
    %1 = fadd <4 x float> %p, %p
    %2 = bitcast <4 x float> %1 to <2 x double>
    %3 = fadd <2 x double> %2, %2
    ret <2 x double> %3
}

; CHECK-LABEL: test_v2f64_v4i32:
define <2 x double> @test_v2f64_v4i32(<4 x i32> %p) {
; CHECK: rev64 v{{[0-9]+}}.4s
; CHECK: ext
; CHECK: ext
    %1 = add <4 x i32> %p, %p
    %2 = bitcast <4 x i32> %1 to <2 x double>
    %3 = fadd <2 x double> %2, %2
    ret <2 x double> %3
}

; CHECK-LABEL: test_v2f64_v8i16:
define <2 x double> @test_v2f64_v8i16(<8 x i16> %p) {
; CHECK: rev64 v{{[0-9]+}}.8h
; CHECK: ext
; CHECK: ext
    %1 = add <8 x i16> %p, %p
    %2 = bitcast <8 x i16> %1 to <2 x double>
    %3 = fadd <2 x double> %2, %2
    ret <2 x double> %3
}

; CHECK-LABEL: test_v2f64_v16i8:
define <2 x double> @test_v2f64_v16i8(<16 x i8> %p) {
; CHECK: rev64 v{{[0-9]+}}.16b
; CHECK: ext
; CHECK: ext
    %1 = add <16 x i8> %p, %p
    %2 = bitcast <16 x i8> %1 to <2 x double>
    %3 = fadd <2 x double> %2, %2
    ret <2 x double> %3
}

; CHECK-LABEL: test_v2i64_f128:
define <2 x i64> @test_v2i64_f128(fp128 %p) {
; CHECK: ext
    %1 = fadd fp128 %p, %p
    %2 = bitcast fp128 %1 to <2 x i64>
    %3 = add <2 x i64> %2, %2
    ret <2 x i64> %3
}

; CHECK-LABEL: test_v2i64_v2f64:
define <2 x i64> @test_v2i64_v2f64(<2 x double> %p) {
; CHECK: ext
; CHECK: ext
    %1 = fadd <2 x double> %p, %p
    %2 = bitcast <2 x double> %1 to <2 x i64>
    %3 = add <2 x i64> %2, %2
    ret <2 x i64> %3
}

; CHECK-LABEL: test_v2i64_v4f32:
define <2 x i64> @test_v2i64_v4f32(<4 x float> %p) {
; CHECK: rev64 v{{[0-9]+}}.4s
; CHECK: ext
; CHECK: ext
    %1 = fadd <4 x float> %p, %p
    %2 = bitcast <4 x float> %1 to <2 x i64>
    %3 = add <2 x i64> %2, %2
    ret <2 x i64> %3
}

; CHECK-LABEL: test_v2i64_v4i32:
define <2 x i64> @test_v2i64_v4i32(<4 x i32> %p) {
; CHECK: rev64 v{{[0-9]+}}.4s
; CHECK: ext
; CHECK: ext
    %1 = add <4 x i32> %p, %p
    %2 = bitcast <4 x i32> %1 to <2 x i64>
    %3 = add <2 x i64> %2, %2
    ret <2 x i64> %3
}

; CHECK-LABEL: test_v2i64_v8i16:
define <2 x i64> @test_v2i64_v8i16(<8 x i16> %p) {
; CHECK: rev64 v{{[0-9]+}}.8h
; CHECK: ext
; CHECK: ext
    %1 = add <8 x i16> %p, %p
    %2 = bitcast <8 x i16> %1 to <2 x i64>
    %3 = add <2 x i64> %2, %2
    ret <2 x i64> %3
}

; CHECK-LABEL: test_v2i64_v16i8:
define <2 x i64> @test_v2i64_v16i8(<16 x i8> %p) {
; CHECK: rev64 v{{[0-9]+}}.16b
; CHECK: ext
; CHECK: ext
    %1 = add <16 x i8> %p, %p
    %2 = bitcast <16 x i8> %1 to <2 x i64>
    %3 = add <2 x i64> %2, %2
    ret <2 x i64> %3
}

; CHECK-LABEL: test_v4f32_f128:
define <4 x float> @test_v4f32_f128(fp128 %p) {
; CHECK: rev64 v{{[0-9]+}}.4s
; CHECK: ext
    %1 = fadd fp128 %p, %p
    %2 = bitcast fp128 %1 to <4 x float>
    %3 = fadd <4 x float> %2, %2
    ret <4 x float> %3
}

; CHECK-LABEL: test_v4f32_v2f64:
define <4 x float> @test_v4f32_v2f64(<2 x double> %p) {
; CHECK: ext
; CHECK: rev64 v{{[0-9]+}}.4s
; CHECK: ext
    %1 = fadd <2 x double> %p, %p
    %2 = bitcast <2 x double> %1 to <4 x float>
    %3 = fadd <4 x float> %2, %2
    ret <4 x float> %3
}

; CHECK-LABEL: test_v4f32_v2i64:
define <4 x float> @test_v4f32_v2i64(<2 x i64> %p) {
; CHECK: ext
; CHECK: rev64 v{{[0-9]+}}.4s
; CHECK: ext
    %1 = add <2 x i64> %p, %p
    %2 = bitcast <2 x i64> %1 to <4 x float>
    %3 = fadd <4 x float> %2, %2
    ret <4 x float> %3
}

; CHECK-LABEL: test_v4f32_v4i32:
define <4 x float> @test_v4f32_v4i32(<4 x i32> %p) {
; CHECK: rev64 v{{[0-9]+}}.4s
; CHECK: ext
; CHECK: rev64 v{{[0-9]+}}.4s
; CHECK: ext
    %1 = add <4 x i32> %p, %p
    %2 = bitcast <4 x i32> %1 to <4 x float>
    %3 = fadd <4 x float> %2, %2
    ret <4 x float> %3
}

; CHECK-LABEL: test_v4f32_v8i16:
define <4 x float> @test_v4f32_v8i16(<8 x i16> %p) {
; CHECK: rev64 v{{[0-9]+}}.8h
; CHECK: ext
; CHECK: rev64 v{{[0-9]+}}.4s
; CHECK: ext
    %1 = add <8 x i16> %p, %p
    %2 = bitcast <8 x i16> %1 to <4 x float>
    %3 = fadd <4 x float> %2, %2
    ret <4 x float> %3
}

; CHECK-LABEL: test_v4f32_v16i8:
define <4 x float> @test_v4f32_v16i8(<16 x i8> %p) {
; CHECK: rev64 v{{[0-9]+}}.16b
; CHECK: ext
; CHECK: rev64 v{{[0-9]+}}.4s
; CHECK: ext
    %1 = add <16 x i8> %p, %p
    %2 = bitcast <16 x i8> %1 to <4 x float>
    %3 = fadd <4 x float> %2, %2
    ret <4 x float> %3
}

; CHECK-LABEL: test_v4i32_f128:
define <4 x i32> @test_v4i32_f128(fp128 %p) {
; CHECK: rev64 v{{[0-9]+}}.4s
; CHECK: ext
    %1 = fadd fp128 %p, %p
    %2 = bitcast fp128 %1 to <4 x i32>
    %3 = add <4 x i32> %2, %2
    ret <4 x i32> %3
}

; CHECK-LABEL: test_v4i32_v2f64:
define <4 x i32> @test_v4i32_v2f64(<2 x double> %p) {
; CHECK: ext
; CHECK: rev64 v{{[0-9]+}}.4s
; CHECK: ext
    %1 = fadd <2 x double> %p, %p
    %2 = bitcast <2 x double> %1 to <4 x i32>
    %3 = add <4 x i32> %2, %2
    ret <4 x i32> %3
}

; CHECK-LABEL: test_v4i32_v2i64:
define <4 x i32> @test_v4i32_v2i64(<2 x i64> %p) {
; CHECK: ext
; CHECK: rev64 v{{[0-9]+}}.4s
; CHECK: ext
    %1 = add <2 x i64> %p, %p
    %2 = bitcast <2 x i64> %1 to <4 x i32>
    %3 = add <4 x i32> %2, %2
    ret <4 x i32> %3
}

; CHECK-LABEL: test_v4i32_v4f32:
define <4 x i32> @test_v4i32_v4f32(<4 x float> %p) {
; CHECK: rev64 v{{[0-9]+}}.4s
; CHECK: ext
; CHECK: rev64 v{{[0-9]+}}.4s
; CHECK: ext
    %1 = fadd <4 x float> %p, %p
    %2 = bitcast <4 x float> %1 to <4 x i32>
    %3 = add <4 x i32> %2, %2
    ret <4 x i32> %3
}

; CHECK-LABEL: test_v4i32_v8i16:
define <4 x i32> @test_v4i32_v8i16(<8 x i16> %p) {
; CHECK: rev64 v{{[0-9]+}}.8h
; CHECK: ext
; CHECK: rev64 v{{[0-9]+}}.4s
; CHECK: ext
    %1 = add <8 x i16> %p, %p
    %2 = bitcast <8 x i16> %1 to <4 x i32>
    %3 = add <4 x i32> %2, %2
    ret <4 x i32> %3
}

; CHECK-LABEL: test_v4i32_v16i8:
define <4 x i32> @test_v4i32_v16i8(<16 x i8> %p) {
; CHECK: rev64 v{{[0-9]+}}.16b
; CHECK: ext
; CHECK: rev64 v{{[0-9]+}}.4s
; CHECK: ext
    %1 = add <16 x i8> %p, %p
    %2 = bitcast <16 x i8> %1 to <4 x i32>
    %3 = add <4 x i32> %2, %2
    ret <4 x i32> %3
}

; CHECK-LABEL: test_v8i16_f128:
define <8 x i16> @test_v8i16_f128(fp128 %p) {
; CHECK: rev64 v{{[0-9]+}}.8h
; CHECK: ext
    %1 = fadd fp128 %p, %p
    %2 = bitcast fp128 %1 to <8 x i16>
    %3 = add <8 x i16> %2, %2
    ret <8 x i16> %3
}

; CHECK-LABEL: test_v8i16_v2f64:
define <8 x i16> @test_v8i16_v2f64(<2 x double> %p) {
; CHECK: ext
; CHECK: rev64 v{{[0-9]+}}.8h
; CHECK: ext
    %1 = fadd <2 x double> %p, %p
    %2 = bitcast <2 x double> %1 to <8 x i16>
    %3 = add <8 x i16> %2, %2
    ret <8 x i16> %3
}

; CHECK-LABEL: test_v8i16_v2i64:
define <8 x i16> @test_v8i16_v2i64(<2 x i64> %p) {
; CHECK: ext
; CHECK: rev64 v{{[0-9]+}}.8h
; CHECK: ext
    %1 = add <2 x i64> %p, %p
    %2 = bitcast <2 x i64> %1 to <8 x i16>
    %3 = add <8 x i16> %2, %2
    ret <8 x i16> %3
}

; CHECK-LABEL: test_v8i16_v4f32:
define <8 x i16> @test_v8i16_v4f32(<4 x float> %p) {
; CHECK: rev64 v{{[0-9]+}}.4s
; CHECK: ext
; CHECK: rev64 v{{[0-9]+}}.8h
; CHECK: ext
    %1 = fadd <4 x float> %p, %p
    %2 = bitcast <4 x float> %1 to <8 x i16>
    %3 = add <8 x i16> %2, %2
    ret <8 x i16> %3
}

; CHECK-LABEL: test_v8i16_v4i32:
define <8 x i16> @test_v8i16_v4i32(<4 x i32> %p) {
; CHECK: rev64 v{{[0-9]+}}.4s
; CHECK: ext
; CHECK: rev64 v{{[0-9]+}}.8h
; CHECK: ext
    %1 = add <4 x i32> %p, %p
    %2 = bitcast <4 x i32> %1 to <8 x i16>
    %3 = add <8 x i16> %2, %2
    ret <8 x i16> %3
}

; CHECK-LABEL: test_v8i16_v16i8:
define <8 x i16> @test_v8i16_v16i8(<16 x i8> %p) {
; CHECK: rev64 v{{[0-9]+}}.16b
; CHECK: ext
; CHECK: rev64 v{{[0-9]+}}.8h
; CHECK: ext
    %1 = add <16 x i8> %p, %p
    %2 = bitcast <16 x i8> %1 to <8 x i16>
    %3 = add <8 x i16> %2, %2
    ret <8 x i16> %3
}

; CHECK-LABEL: test_v16i8_f128:
define <16 x i8> @test_v16i8_f128(fp128 %p) {
; CHECK: rev64 v{{[0-9]+}}.16b
; CHECK: ext
    %1 = fadd fp128 %p, %p
    %2 = bitcast fp128 %1 to <16 x i8>
    %3 = add <16 x i8> %2, %2
    ret <16 x i8> %3
}

; CHECK-LABEL: test_v16i8_v2f64:
define <16 x i8> @test_v16i8_v2f64(<2 x double> %p) {
; CHECK: ext
; CHECK: rev64 v{{[0-9]+}}.16b
; CHECK: ext
    %1 = fadd <2 x double> %p, %p
    %2 = bitcast <2 x double> %1 to <16 x i8>
    %3 = add <16 x i8> %2, %2
    ret <16 x i8> %3
}

; CHECK-LABEL: test_v16i8_v2i64:
define <16 x i8> @test_v16i8_v2i64(<2 x i64> %p) {
; CHECK: ext
; CHECK: rev64 v{{[0-9]+}}.16b
; CHECK: ext
    %1 = add <2 x i64> %p, %p
    %2 = bitcast <2 x i64> %1 to <16 x i8>
    %3 = add <16 x i8> %2, %2
    ret <16 x i8> %3
}

; CHECK-LABEL: test_v16i8_v4f32:
define <16 x i8> @test_v16i8_v4f32(<4 x float> %p) {
; CHECK: rev64 v{{[0-9]+}}.4s
; CHECK: ext
; CHECK: rev64 v{{[0-9]+}}.16b
; CHECK: ext
    %1 = fadd <4 x float> %p, %p
    %2 = bitcast <4 x float> %1 to <16 x i8>
    %3 = add <16 x i8> %2, %2
    ret <16 x i8> %3
}

; CHECK-LABEL: test_v16i8_v4i32:
define <16 x i8> @test_v16i8_v4i32(<4 x i32> %p) {
; CHECK: rev64 v{{[0-9]+}}.4s
; CHECK: ext
; CHECK: rev64 v{{[0-9]+}}.16b
; CHECK: ext
    %1 = add <4 x i32> %p, %p
    %2 = bitcast <4 x i32> %1 to <16 x i8>
    %3 = add <16 x i8> %2, %2
    ret <16 x i8> %3
}

; CHECK-LABEL: test_v16i8_v8i16:
define <16 x i8> @test_v16i8_v8i16(<8 x i16> %p) {
; CHECK: rev64 v{{[0-9]+}}.8h
; CHECK: ext
; CHECK: rev64 v{{[0-9]+}}.16b
; CHECK: ext
    %1 = add <8 x i16> %p, %p
    %2 = bitcast <8 x i16> %1 to <16 x i8>
    %3 = add <16 x i8> %2, %2
    ret <16 x i8> %3
}

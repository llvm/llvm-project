; RUN: opt -mtriple=x86_64-apple-darwin -mattr=+sse2 -cost-model -analyze < %s | FileCheck --check-prefix=SSE --check-prefix=SSE2 %s
; RUN: opt -mtriple=x86_64-apple-darwin -mattr=+avx  -cost-model -analyze < %s | FileCheck --check-prefix=AVX --check-prefix=AVX1 %s
; RUN: opt -mtriple=x86_64-apple-darwin -mattr=+avx2 -cost-model -analyze < %s | FileCheck --check-prefix=AVX --check-prefix=AVX2 %s
; RUN: opt -mtriple=x86_64-apple-darwin -mattr=+avx512f -cost-model -analyze < %s | FileCheck --check-prefix=AVX512 --check-prefix=AVX512F %s
; RUN: opt -mtriple=x86_64-apple-darwin -mattr=+avx512dq -cost-model -analyze < %s | FileCheck --check-prefix=AVX512 --check-prefix=AVX512DQ %s

; CHECK-LABEL: 'uitofp_i8_double'
define i32 @uitofp_i8_double() {
  ; SSE2: cost of 1 {{.*}} uitofp i8
  ; AVX1: cost of 1 {{.*}} uitofp i8
  ; AVX2: cost of 1 {{.*}} uitofp i8
  ; AVX512: cost of 1 {{.*}} uitofp i8
  %cvt_i8_f64 = uitofp i8 undef to double

  ; SSE2: cost of 20 {{.*}} uitofp <2 x i8>
  ; AVX1: cost of 4 {{.*}} uitofp <2 x i8>
  ; AVX2: cost of 4 {{.*}} uitofp <2 x i8>
  ; AVX512: cost of 2 {{.*}} uitofp <2 x i8>
  %cvt_v2i8_v2f64 = uitofp <2 x i8> undef to <2 x double>

  ; SSE2: cost of 40 {{.*}} uitofp <4 x i8>
  ; AVX1: cost of 2 {{.*}} uitofp <4 x i8>
  ; AVX2: cost of 2 {{.*}} uitofp <4 x i8>
  ; AVX512: cost of 2 {{.*}} uitofp <4 x i8>
  %cvt_v4i8_v4f64 = uitofp <4 x i8> undef to <4 x double>

  ; SSE2: cost of 80 {{.*}} uitofp <8 x i8>
  ; AVX1: cost of 5 {{.*}} uitofp <8 x i8>
  ; AVX2: cost of 5 {{.*}} uitofp <8 x i8>
  ; AVX512: cost of 2 {{.*}} uitofp <8 x i8>
  %cvt_v8i8_v8f64 = uitofp <8 x i8> undef to <8 x double>

  ret i32 undef
}

; CHECK-LABEL: 'uitofp_i16_double'
define i32 @uitofp_i16_double() {
  ; SSE2: cost of 1 {{.*}} uitofp i16
  ; AVX1: cost of 1 {{.*}} uitofp i16
  ; AVX2: cost of 1 {{.*}} uitofp i16
  ; AVX512: cost of 1 {{.*}} uitofp i16
  %cvt_i16_f64 = uitofp i16 undef to double

  ; SSE2: cost of 20 {{.*}} uitofp <2 x i16>
  ; AVX1: cost of 4 {{.*}} uitofp <2 x i16>
  ; AVX2: cost of 4 {{.*}} uitofp <2 x i16>
  ; AVX512: cost of 5 {{.*}} uitofp <2 x i16>
  %cvt_v2i16_v2f64 = uitofp <2 x i16> undef to <2 x double>

  ; SSE2: cost of 40 {{.*}} uitofp <4 x i16>
  ; AVX1: cost of 2 {{.*}} uitofp <4 x i16>
  ; AVX2: cost of 2 {{.*}} uitofp <4 x i16>
  ; AVX512: cost of 2 {{.*}} uitofp <4 x i16>
  %cvt_v4i16_v4f64 = uitofp <4 x i16> undef to <4 x double>

  ; SSE2: cost of 80 {{.*}} uitofp <8 x i16>
  ; AVX1: cost of 5 {{.*}} uitofp <8 x i16>
  ; AVX2: cost of 5 {{.*}} uitofp <8 x i16>
  ; AVX512: cost of 2 {{.*}} uitofp <8 x i16>
  %cvt_v8i16_v8f64 = uitofp <8 x i16> undef to <8 x double>

  ret i32 undef
}

; CHECK-LABEL: 'uitofp_i32_double'
define i32 @uitofp_i32_double() {
  ; SSE2: cost of 1 {{.*}} uitofp i32
  ; AVX1: cost of 1 {{.*}} uitofp i32
  ; AVX2: cost of 1 {{.*}} uitofp i32
  ; AVX512: cost of 1 {{.*}} uitofp i32
  %cvt_i32_f64 = uitofp i32 undef to double

  ; SSE2: cost of 20 {{.*}} uitofp <2 x i32>
  ; AVX1: cost of 6 {{.*}} uitofp <2 x i32>
  ; AVX2: cost of 6 {{.*}} uitofp <2 x i32>
  ; AVX512: cost of 1 {{.*}} uitofp <2 x i32>
  %cvt_v2i32_v2f64 = uitofp <2 x i32> undef to <2 x double>

  ; SSE2: cost of 40 {{.*}} uitofp <4 x i32>
  ; AVX1: cost of 6 {{.*}} uitofp <4 x i32>
  ; AVX2: cost of 6 {{.*}} uitofp <4 x i32>
  ; AVX512: cost of 1 {{.*}} uitofp <4 x i32>
  %cvt_v4i32_v4f64 = uitofp <4 x i32> undef to <4 x double>

  ; SSE2: cost of 80 {{.*}} uitofp <8 x i32>
  ; AVX1: cost of 13 {{.*}} uitofp <8 x i32>
  ; AVX2: cost of 13 {{.*}} uitofp <8 x i32>
  ; AVX512: cost of 1 {{.*}} uitofp <8 x i32>
  %cvt_v8i32_v8f64 = uitofp <8 x i32> undef to <8 x double>

  ret i32 undef
}

; CHECK-LABEL: 'uitofp_i64_double'
define i32 @uitofp_i64_double() {
  ; SSE2: cost of 1 {{.*}} uitofp i64
  ; AVX1: cost of 1 {{.*}} uitofp i64
  ; AVX2: cost of 1 {{.*}} uitofp i64
  ; AVX512: cost of 1 {{.*}} uitofp i64
  %cvt_i64_f64 = uitofp i64 undef to double

  ; SSE2: cost of 20 {{.*}} uitofp <2 x i64>
  ; AVX1: cost of 10 {{.*}} uitofp <2 x i64>
  ; AVX2: cost of 10 {{.*}} uitofp <2 x i64>
  ; AVX512F: cost of 5 {{.*}} uitofp <2 x i64>
  ; AVX512DQ: cost of 1 {{.*}} uitofp <2 x i64>
  %cvt_v2i64_v2f64 = uitofp <2 x i64> undef to <2 x double>

  ; SSE2: cost of 40 {{.*}} uitofp <4 x i64>
  ; AVX1: cost of 20 {{.*}} uitofp <4 x i64>
  ; AVX2: cost of 20 {{.*}} uitofp <4 x i64>
  ; AVX512F: cost of 12 {{.*}} uitofp <4 x i64>
  ; AVX512DQ: cost of 1 {{.*}} uitofp <4 x i64>
  %cvt_v4i64_v4f64 = uitofp <4 x i64> undef to <4 x double>

  ; SSE2: cost of 80 {{.*}} uitofp <8 x i64>
  ; AVX1: cost of 41 {{.*}} uitofp <8 x i64>
  ; AVX2: cost of 41 {{.*}} uitofp <8 x i64>
  ; AVX512F: cost of 26 {{.*}} uitofp <8 x i64>
  ; AVX512DQ: cost of 1 {{.*}} uitofp <8 x i64>
  %cvt_v8i64_v8f64 = uitofp <8 x i64> undef to <8 x double>

  ret i32 undef
}

; CHECK-LABEL: 'uitofp_i8_float'
define i32 @uitofp_i8_float() {
  ; SSE2: cost of 1 {{.*}} uitofp i8
  ; AVX1: cost of 1 {{.*}} uitofp i8
  ; AVX2: cost of 1 {{.*}} uitofp i8
  ; AVX512: cost of 1 {{.*}} uitofp i8
  %cvt_i8_f32 = uitofp i8 undef to float

  ; SSE2: cost of 8 {{.*}} uitofp <4 x i8>
  ; AVX1: cost of 2 {{.*}} uitofp <4 x i8>
  ; AVX2: cost of 2 {{.*}} uitofp <4 x i8>
  ; AVX512: cost of 2 {{.*}} uitofp <4 x i8>
  %cvt_v4i8_v4f32 = uitofp <4 x i8> undef to <4 x float>

  ; SSE2: cost of 15 {{.*}} uitofp <8 x i8>
  ; AVX1: cost of 5 {{.*}} uitofp <8 x i8>
  ; AVX2: cost of 5 {{.*}} uitofp <8 x i8>
  ; AVX512: cost of 2 {{.*}} uitofp <8 x i8>
  %cvt_v8i8_v8f32 = uitofp <8 x i8> undef to <8 x float>

  ; SSE2: cost of 8 {{.*}} uitofp <16 x i8>
  ; AVX1: cost of 11 {{.*}} uitofp <16 x i8>
  ; AVX16: cost of 11 {{.*}} uitofp <16 x i8>
  ; AVX512: cost of 2 {{.*}} uitofp <16 x i8>
  %cvt_v16i8_v16f32 = uitofp <16 x i8> undef to <16 x float>

  ret i32 undef
}

; CHECK-LABEL: 'uitofp_i16_float'
define i32 @uitofp_i16_float() {
  ; SSE2: cost of 1 {{.*}} uitofp i16
  ; AVX1: cost of 1 {{.*}} uitofp i16
  ; AVX2: cost of 1 {{.*}} uitofp i16
  ; AVX512: cost of 1 {{.*}} uitofp i16
  %cvt_i16_f32 = uitofp i16 undef to float

  ; SSE2: cost of 8 {{.*}} uitofp <4 x i16>
  ; AVX1: cost of 2 {{.*}} uitofp <4 x i16>
  ; AVX2: cost of 2 {{.*}} uitofp <4 x i16>
  ; AVX512: cost of 2 {{.*}} uitofp <4 x i16>
  %cvt_v4i16_v4f32 = uitofp <4 x i16> undef to <4 x float>

  ; SSE2: cost of 15 {{.*}} uitofp <8 x i16>
  ; AVX1: cost of 5 {{.*}} uitofp <8 x i16>
  ; AVX2: cost of 5 {{.*}} uitofp <8 x i16>
  ; AVX512: cost of 2 {{.*}} uitofp <8 x i16>
  %cvt_v8i16_v8f32 = uitofp <8 x i16> undef to <8 x float>

  ; SSE2: cost of 30 {{.*}} uitofp <16 x i16>
  ; AVX1: cost of 11 {{.*}} uitofp <16 x i16>
  ; AVX16: cost of 11 {{.*}} uitofp <16 x i16>
  ; AVX512: cost of 2 {{.*}} uitofp <16 x i16>
  %cvt_v16i16_v16f32 = uitofp <16 x i16> undef to <16 x float>

  ret i32 undef
}

; CHECK-LABEL: 'uitofp_i32_float'
define i32 @uitofp_i32_float() {
  ; SSE2: cost of 1 {{.*}} uitofp i32
  ; AVX1: cost of 1 {{.*}} uitofp i32
  ; AVX2: cost of 1 {{.*}} uitofp i32
  ; AVX512: cost of 1 {{.*}} uitofp i32
  %cvt_i32_f32 = uitofp i32 undef to float

  ; SSE2: cost of 8 {{.*}} uitofp <4 x i32>
  ; AVX1: cost of 6 {{.*}} uitofp <4 x i32>
  ; AVX2: cost of 6 {{.*}} uitofp <4 x i32>
  ; AVX512: cost of 1 {{.*}} uitofp <4 x i32>
  %cvt_v4i32_v4f32 = uitofp <4 x i32> undef to <4 x float>

  ; SSE2: cost of 16 {{.*}} uitofp <8 x i32>
  ; AVX1: cost of 9 {{.*}} uitofp <8 x i32>
  ; AVX2: cost of 8 {{.*}} uitofp <8 x i32>
  ; AVX512: cost of 1 {{.*}} uitofp <8 x i32>
  %cvt_v8i32_v8f32 = uitofp <8 x i32> undef to <8 x float>

  ; SSE2: cost of 32 {{.*}} uitofp <16 x i32>
  ; AVX1: cost of 19 {{.*}} uitofp <16 x i32>
  ; AVX2: cost of 17 {{.*}} uitofp <16 x i32>
  ; AVX512: cost of 1 {{.*}} uitofp <16 x i32>
  %cvt_v16i32_v16f32 = uitofp <16 x i32> undef to <16 x float>

  ret i32 undef
}

; CHECK-LABEL: 'uitofp_i64_float'
define i32 @uitofp_i64_float() {
  ; SSE2: cost of 1 {{.*}} uitofp i64
  ; AVX1: cost of 1 {{.*}} uitofp i64
  ; AVX2: cost of 1 {{.*}} uitofp i64
  ; AVX512: cost of 1 {{.*}} uitofp i64
  %cvt_i64_f32 = uitofp i64 undef to float

  ; SSE2: cost of 15 {{.*}} uitofp <2 x i64>
  ; AVX1: cost of 4 {{.*}} uitofp <2 x i64>
  ; AVX2: cost of 4 {{.*}} uitofp <2 x i64>
  ; AVX512F: cost of 5 {{.*}} uitofp <2 x i64>
  ; AVX512DQ: cost of 1 {{.*}} uitofp <2 x i64>
  %cvt_v2i64_v2f32 = uitofp <2 x i64> undef to <2 x float>

  ; SSE2: cost of 30 {{.*}} uitofp <4 x i64>
  ; AVX1: cost of 10 {{.*}} uitofp <4 x i64>
  ; AVX2: cost of 10 {{.*}} uitofp <4 x i64>
  ; AVX512F: cost of 10 {{.*}} uitofp <4 x i64>
  ; AVX512DQ: cost of 1 {{.*}} uitofp <4 x i64>
  %cvt_v4i64_v4f32 = uitofp <4 x i64> undef to <4 x float>

  ; SSE2: cost of 60 {{.*}} uitofp <8 x i64>
  ; AVX1: cost of 21 {{.*}} uitofp <8 x i64>
  ; AVX2: cost of 21 {{.*}} uitofp <8 x i64>
  ; AVX512F: cost of 26 {{.*}} uitofp <8 x i64>
  ; AVX512DQ: cost of 1 {{.*}} uitofp <8 x i64>
  %cvt_v8i64_v8f32 = uitofp <8 x i64> undef to <8 x float>

  ; SSE2: cost of 120 {{.*}} uitofp <16 x i64>
  ; AVX1: cost of 43 {{.*}} uitofp <16 x i64>
  ; AVX2: cost of 43 {{.*}} uitofp <16 x i64>
  ; AVX512F: cost of 53 {{.*}} uitofp <16 x i64>
  ; AVX512DQ: cost of 3 {{.*}} uitofp <16 x i64>
  %cvt_v16i64_v16f32 = uitofp <16 x i64> undef to <16 x float>

  ret i32 undef
}

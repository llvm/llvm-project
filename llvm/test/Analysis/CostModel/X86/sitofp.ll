; RUN: opt -mtriple=x86_64-apple-darwin -mattr=+sse2 -cost-model -analyze < %s | FileCheck --check-prefix=SSE --check-prefix=SSE2 %s
; RUN: opt -mtriple=x86_64-apple-darwin -mattr=+avx  -cost-model -analyze < %s | FileCheck --check-prefix=AVX --check-prefix=AVX1 %s
; RUN: opt -mtriple=x86_64-apple-darwin -mattr=+avx2 -cost-model -analyze < %s | FileCheck --check-prefix=AVX --check-prefix=AVX2 %s
; RUN: opt -mtriple=x86_64-apple-darwin -mattr=+avx512f -cost-model -analyze < %s | FileCheck --check-prefix=AVX512 --check-prefix=AVX512F %s
; RUN: opt -mtriple=x86_64-apple-darwin -mattr=+avx512dq -cost-model -analyze < %s | FileCheck --check-prefix=AVX512 --check-prefix=AVX512DQ %s

; CHECK-LABEL: 'sitofp_i8_double'
define i32 @sitofp_i8_double() {
  ; SSE2: cost of 1 {{.*}} sitofp i8
  ; AVX1: cost of 1 {{.*}} sitofp i8
  ; AVX2: cost of 1 {{.*}} sitofp i8
  ; AVX512: cost of 1 {{.*}} sitofp i8
  %cvt_i8_f64 = sitofp i8 undef to double

  ; SSE2: cost of 20 {{.*}} sitofp <2 x i8>
  ; AVX1: cost of 4 {{.*}} sitofp <2 x i8>
  ; AVX2: cost of 4 {{.*}} sitofp <2 x i8>
  ; AVX512: cost of 4 {{.*}} sitofp <2 x i8>
  %cvt_v2i8_v2f64 = sitofp <2 x i8> undef to <2 x double>

  ; SSE2: cost of 40 {{.*}} sitofp <4 x i8>
  ; AVX1: cost of 3 {{.*}} sitofp <4 x i8>
  ; AVX2: cost of 3 {{.*}} sitofp <4 x i8>
  ; AVX512: cost of 3 {{.*}} sitofp <4 x i8>
  %cvt_v4i8_v4f64 = sitofp <4 x i8> undef to <4 x double>

  ; SSE2: cost of 80 {{.*}} sitofp <8 x i8>
  ; AVX1: cost of 7 {{.*}} sitofp <8 x i8>
  ; AVX2: cost of 7 {{.*}} sitofp <8 x i8>
  ; AVX512: cost of 2 {{.*}} sitofp <8 x i8>
  %cvt_v8i8_v8f64 = sitofp <8 x i8> undef to <8 x double>

  ret i32 undef
}

; CHECK-LABEL: 'sitofp_i16_double'
define i32 @sitofp_i16_double() {
  ; SSE2: cost of 1 {{.*}} sitofp i16
  ; AVX1: cost of 1 {{.*}} sitofp i16
  ; AVX2: cost of 1 {{.*}} sitofp i16
  ; AVX512: cost of 1 {{.*}} sitofp i16
  %cvt_i16_f64 = sitofp i16 undef to double

  ; SSE2: cost of 20 {{.*}} sitofp <2 x i16>
  ; AVX1: cost of 4 {{.*}} sitofp <2 x i16>
  ; AVX2: cost of 4 {{.*}} sitofp <2 x i16>
  ; AVX512: cost of 4 {{.*}} sitofp <2 x i16>
  %cvt_v2i16_v2f64 = sitofp <2 x i16> undef to <2 x double>

  ; SSE2: cost of 40 {{.*}} sitofp <4 x i16>
  ; AVX1: cost of 3 {{.*}} sitofp <4 x i16>
  ; AVX2: cost of 3 {{.*}} sitofp <4 x i16>
  ; AVX512: cost of 3 {{.*}} sitofp <4 x i16>
  %cvt_v4i16_v4f64 = sitofp <4 x i16> undef to <4 x double>

  ; SSE2: cost of 80 {{.*}} sitofp <8 x i16>
  ; AVX1: cost of 7 {{.*}} sitofp <8 x i16>
  ; AVX2: cost of 7 {{.*}} sitofp <8 x i16>
  ; AVX512: cost of 2 {{.*}} sitofp <8 x i16>
  %cvt_v8i16_v8f64 = sitofp <8 x i16> undef to <8 x double>

  ret i32 undef
}

; CHECK-LABEL: 'sitofp_i32_double'
define i32 @sitofp_i32_double() {
  ; SSE2: cost of 1 {{.*}} sitofp i32
  ; AVX1: cost of 1 {{.*}} sitofp i32
  ; AVX2: cost of 1 {{.*}} sitofp i32
  ; AVX512: cost of 1 {{.*}} sitofp i32
  %cvt_i32_f64 = sitofp i32 undef to double

  ; SSE2: cost of 20 {{.*}} sitofp <2 x i32>
  ; AVX1: cost of 4 {{.*}} sitofp <2 x i32>
  ; AVX2: cost of 4 {{.*}} sitofp <2 x i32>
  ; AVX512: cost of 4 {{.*}} sitofp <2 x i32>
  %cvt_v2i32_v2f64 = sitofp <2 x i32> undef to <2 x double>

  ; SSE2: cost of 40 {{.*}} sitofp <4 x i32>
  ; AVX1: cost of 1 {{.*}} sitofp <4 x i32>
  ; AVX2: cost of 1 {{.*}} sitofp <4 x i32>
  ; AVX512: cost of 1 {{.*}} sitofp <4 x i32>
  %cvt_v4i32_v4f64 = sitofp <4 x i32> undef to <4 x double>

  ; SSE2: cost of 80 {{.*}} sitofp <8 x i32>
  ; AVX1: cost of 3 {{.*}} sitofp <8 x i32>
  ; AVX2: cost of 3 {{.*}} sitofp <8 x i32>
  ; AVX512: cost of 1 {{.*}} sitofp <8 x i32>
  %cvt_v8i32_v8f64 = sitofp <8 x i32> undef to <8 x double>

  ret i32 undef
}

; CHECK-LABEL: 'sitofp_i64_double'
define i32 @sitofp_i64_double() {
  ; SSE2: cost of 1 {{.*}} sitofp i64
  ; AVX1: cost of 1 {{.*}} sitofp i64
  ; AVX2: cost of 1 {{.*}} sitofp i64
  ; AVX512: cost of 1 {{.*}} sitofp i64
  %cvt_i64_f64 = sitofp i64 undef to double

  ; SSE2: cost of 20 {{.*}} sitofp <2 x i64>
  ; AVX1: cost of 20 {{.*}} sitofp <2 x i64>
  ; AVX2: cost of 20 {{.*}} sitofp <2 x i64>
  ; AVX512F: cost of 20 {{.*}} sitofp <2 x i64>
  ; AVX512DQ: cost of 1 {{.*}} sitofp <2 x i64>
  %cvt_v2i64_v2f64 = sitofp <2 x i64> undef to <2 x double>

  ; SSE2: cost of 40 {{.*}} sitofp <4 x i64>
  ; AVX1: cost of 13 {{.*}} sitofp <4 x i64>
  ; AVX2: cost of 13 {{.*}} sitofp <4 x i64>
  ; AVX512F: cost of 13 {{.*}} sitofp <4 x i64>
  ; AVX512DQ: cost of 1 {{.*}} sitofp <4 x i64>
  %cvt_v4i64_v4f64 = sitofp <4 x i64> undef to <4 x double>

  ; SSE2: cost of 80 {{.*}} sitofp <8 x i64>
  ; AVX1: cost of 27 {{.*}} sitofp <8 x i64>
  ; AVX2: cost of 27 {{.*}} sitofp <8 x i64>
  ; AVX512F: cost of 22 {{.*}} sitofp <8 x i64>
  ; AVX512DQ: cost of 1 {{.*}} sitofp <8 x i64>
  %cvt_v8i64_v8f64 = sitofp <8 x i64> undef to <8 x double>

  ret i32 undef
}

; CHECK-LABEL: 'sitofp_i8_float'
define i32 @sitofp_i8_float() {
  ; SSE2: cost of 1 {{.*}} sitofp i8
  ; AVX1: cost of 1 {{.*}} sitofp i8
  ; AVX2: cost of 1 {{.*}} sitofp i8
  ; AVX512: cost of 1 {{.*}} sitofp i8
  %cvt_i8_f32 = sitofp i8 undef to float

  ; SSE2: cost of 5 {{.*}} sitofp <4 x i8>
  ; AVX1: cost of 3 {{.*}} sitofp <4 x i8>
  ; AVX2: cost of 3 {{.*}} sitofp <4 x i8>
  ; AVX512: cost of 3 {{.*}} sitofp <4 x i8>
  %cvt_v4i8_v4f32 = sitofp <4 x i8> undef to <4 x float>

  ; SSE2: cost of 15 {{.*}} sitofp <8 x i8>
  ; AVX1: cost of 8 {{.*}} sitofp <8 x i8>
  ; AVX2: cost of 8 {{.*}} sitofp <8 x i8>
  ; AVX512: cost of 8 {{.*}} sitofp <8 x i8>
  %cvt_v8i8_v8f32 = sitofp <8 x i8> undef to <8 x float>

  ; SSE2: cost of 8 {{.*}} sitofp <16 x i8>
  ; AVX1: cost of 17 {{.*}} sitofp <16 x i8>
  ; AVX16: cost of 17 {{.*}} sitofp <16 x i8>
  ; AVX512: cost of 2 {{.*}} sitofp <16 x i8>
  %cvt_v16i8_v16f32 = sitofp <16 x i8> undef to <16 x float>

  ret i32 undef
}

; CHECK-LABEL: 'sitofp_i16_float'
define i32 @sitofp_i16_float() {
  ; SSE2: cost of 1 {{.*}} sitofp i16
  ; AVX1: cost of 1 {{.*}} sitofp i16
  ; AVX2: cost of 1 {{.*}} sitofp i16
  ; AVX512: cost of 1 {{.*}} sitofp i16
  %cvt_i16_f32 = sitofp i16 undef to float

  ; SSE2: cost of 5 {{.*}} sitofp <4 x i16>
  ; AVX1: cost of 3 {{.*}} sitofp <4 x i16>
  ; AVX2: cost of 3 {{.*}} sitofp <4 x i16>
  ; AVX512: cost of 3 {{.*}} sitofp <4 x i16>
  %cvt_v4i16_v4f32 = sitofp <4 x i16> undef to <4 x float>

  ; SSE2: cost of 15 {{.*}} sitofp <8 x i16>
  ; AVX1: cost of 5 {{.*}} sitofp <8 x i16>
  ; AVX2: cost of 5 {{.*}} sitofp <8 x i16>
  ; AVX512: cost of 5 {{.*}} sitofp <8 x i16>
  %cvt_v8i16_v8f32 = sitofp <8 x i16> undef to <8 x float>

  ; SSE2: cost of 30 {{.*}} sitofp <16 x i16>
  ; AVX1: cost of 11 {{.*}} sitofp <16 x i16>
  ; AVX16: cost of 11 {{.*}} sitofp <16 x i16>
  ; AVX512: cost of 2 {{.*}} sitofp <16 x i16>
  %cvt_v16i16_v16f32 = sitofp <16 x i16> undef to <16 x float>

  ret i32 undef
}

; CHECK-LABEL: 'sitofp_i32_float'
define i32 @sitofp_i32_float() {
  ; SSE2: cost of 1 {{.*}} sitofp i32
  ; AVX1: cost of 1 {{.*}} sitofp i32
  ; AVX2: cost of 1 {{.*}} sitofp i32
  ; AVX512: cost of 1 {{.*}} sitofp i32
  %cvt_i32_f32 = sitofp i32 undef to float

  ; SSE2: cost of 5 {{.*}} sitofp <4 x i32>
  ; AVX1: cost of 1 {{.*}} sitofp <4 x i32>
  ; AVX2: cost of 1 {{.*}} sitofp <4 x i32>
  ; AVX512: cost of 1 {{.*}} sitofp <4 x i32>
  %cvt_v4i32_v4f32 = sitofp <4 x i32> undef to <4 x float>

  ; SSE2: cost of 10 {{.*}} sitofp <8 x i32>
  ; AVX1: cost of 1 {{.*}} sitofp <8 x i32>
  ; AVX2: cost of 1 {{.*}} sitofp <8 x i32>
  ; AVX512: cost of 1 {{.*}} sitofp <8 x i32>
  %cvt_v8i32_v8f32 = sitofp <8 x i32> undef to <8 x float>

  ; SSE2: cost of 20 {{.*}} sitofp <16 x i32>
  ; AVX1: cost of 3 {{.*}} sitofp <16 x i32>
  ; AVX2: cost of 3 {{.*}} sitofp <16 x i32>
  ; AVX512: cost of 1 {{.*}} sitofp <16 x i32>
  %cvt_v16i32_v16f32 = sitofp <16 x i32> undef to <16 x float>

  ret i32 undef
}

; CHECK-LABEL: 'sitofp_i64_float'
define i32 @sitofp_i64_float() {
  ; SSE2: cost of 1 {{.*}} sitofp i64
  ; AVX1: cost of 1 {{.*}} sitofp i64
  ; AVX2: cost of 1 {{.*}} sitofp i64
  ; AVX512: cost of 1 {{.*}} sitofp i64
  %cvt_i64_f32 = sitofp i64 undef to float

  ; SSE2: cost of 15 {{.*}} sitofp <2 x i64>
  ; AVX1: cost of 4 {{.*}} sitofp <2 x i64>
  ; AVX2: cost of 4 {{.*}} sitofp <2 x i64>
  ; AVX512F: cost of 4 {{.*}} sitofp <2 x i64>
  ; AVX512DQ: cost of 1 {{.*}} sitofp <2 x i64>
  %cvt_v2i64_v2f32 = sitofp <2 x i64> undef to <2 x float>

  ; SSE2: cost of 30 {{.*}} sitofp <4 x i64>
  ; AVX1: cost of 10 {{.*}} sitofp <4 x i64>
  ; AVX2: cost of 10 {{.*}} sitofp <4 x i64>
  ; AVX512F: cost of 10 {{.*}} sitofp <4 x i64>
  ; AVX512DQ: cost of 1 {{.*}} sitofp <4 x i64>
  %cvt_v4i64_v4f32 = sitofp <4 x i64> undef to <4 x float>

  ; SSE2: cost of 60 {{.*}} sitofp <8 x i64>
  ; AVX1: cost of 21 {{.*}} sitofp <8 x i64>
  ; AVX2: cost of 21 {{.*}} sitofp <8 x i64>
  ; AVX512F: cost of 22 {{.*}} sitofp <8 x i64>
  ; AVX512DQ: cost of 1 {{.*}} sitofp <8 x i64>
  %cvt_v8i64_v8f32 = sitofp <8 x i64> undef to <8 x float>

  ; SSE2: cost of 120 {{.*}} sitofp <16 x i64>
  ; AVX1: cost of 43 {{.*}} sitofp <16 x i64>
  ; AVX2: cost of 43 {{.*}} sitofp <16 x i64>
  ; AVX512F: cost of 45 {{.*}} sitofp <16 x i64>
  ; AVX512DQ: cost of 3 {{.*}} sitofp <16 x i64>
  %cvt_v16i64_v16f32 = sitofp <16 x i64> undef to <16 x float>

  ret i32 undef
}

; RUN: opt -mtriple=x86_64-apple-darwin -mattr=+sse2 -cost-model -analyze < %s | FileCheck --check-prefix=SSE --check-prefix=SSE2 %s
; RUN: opt -mtriple=x86_64-apple-darwin -mattr=+sse4.2 -cost-model -analyze < %s | FileCheck --check-prefix=SSE --check-prefix=SSE42 %s
; RUN: opt -mtriple=x86_64-apple-darwin -mattr=+avx  -cost-model -analyze < %s | FileCheck --check-prefix=AVX --check-prefix=AVX1 %s
; RUN: opt -mtriple=x86_64-apple-darwin -mattr=+avx2 -cost-model -analyze < %s | FileCheck --check-prefix=AVX --check-prefix=AVX2 %s
; RUN: opt -mtriple=x86_64-apple-darwin -mattr=+avx512f -cost-model -analyze < %s | FileCheck --check-prefix=AVX512 --check-prefix=AVX512F %s
; RUN: opt -mtriple=x86_64-apple-darwin -mattr=+avx512f,+avx512dq -cost-model -analyze < %s | FileCheck --check-prefix=AVX512 --check-prefix=AVX512DQ %s

; CHECK-LABEL: 'fptosi_double_i64'
define i32 @fptosi_double_i64(i32 %arg) {
  ; SSE2: cost of 1 {{.*}} %I64 = fptosi
  ; SSE42: cost of 1 {{.*}} %I64 = fptosi
  ; AVX1: cost of 1 {{.*}} %I64 = fptosi
  ; AVX2: cost of 1 {{.*}} %I64 = fptosi
  ; AVX512: cost of 1 {{.*}} %I64 = fptosi
  %I64 = fptosi double undef to i64
  ; SSE2: cost of 6 {{.*}} %V2I64 = fptosi
  ; SSE42: cost of 6 {{.*}} %V2I64 = fptosi
  ; AVX1: cost of 6 {{.*}} %V2I64 = fptosi
  ; AVX2: cost of 6 {{.*}} %V2I64 = fptosi
  ; AVX512F: cost of 6 {{.*}} %V2I64 = fptosi
  ; AVX512DQ: cost of 1 {{.*}} %V2I64 = fptosi
  %V2I64 = fptosi <2 x double> undef to <2 x i64>
  ; SSE2: cost of 13 {{.*}} %V4I64 = fptosi
  ; SSE42: cost of 13 {{.*}} %V4I64 = fptosi
  ; AVX1: cost of 12 {{.*}} %V4I64 = fptosi
  ; AVX2: cost of 12 {{.*}} %V4I64 = fptosi
  ; AVX512F: cost of 12 {{.*}} %V4I64 = fptosi
  ; AVX512DQ: cost of 1 {{.*}} %V4I64 = fptosi
  %V4I64 = fptosi <4 x double> undef to <4 x i64>
  ; SSE2: cost of 27 {{.*}} %V8I64 = fptosi
  ; SSE42: cost of 27 {{.*}} %V8I64 = fptosi
  ; AVX1: cost of 25 {{.*}} %V8I64 = fptosi
  ; AVX2: cost of 25 {{.*}} %V8I64 = fptosi
  ; AVX512F: cost of 24 {{.*}} %V8I64 = fptosi
  ; AVX512DQ: cost of 1 {{.*}} %V8I64 = fptosi
  %V8I64 = fptosi <8 x double> undef to <8 x i64>

  ret i32 undef
}

; CHECK-LABEL: 'fptosi_double_i32'
define i32 @fptosi_double_i32(i32 %arg) {
  ; SSE2: cost of 1 {{.*}} %I32 = fptosi
  ; SSE42: cost of 1 {{.*}} %I32 = fptosi
  ; AVX1: cost of 1 {{.*}} %I32 = fptosi
  ; AVX2: cost of 1 {{.*}} %I32 = fptosi
  ; AVX512: cost of 1 {{.*}} %I32 = fptosi
  %I32 = fptosi double undef to i32
  ; SSE2: cost of 3 {{.*}} %V2I32 = fptosi
  ; SSE42: cost of 3 {{.*}} %V2I32 = fptosi
  ; AVX1: cost of 3 {{.*}} %V2I32 = fptosi
  ; AVX2: cost of 3 {{.*}} %V2I32 = fptosi
  ; AVX512: cost of 3 {{.*}} %V2I32 = fptosi
  %V2I32 = fptosi <2 x double> undef to <2 x i32>
  ; SSE2: cost of 7 {{.*}} %V4I32 = fptosi
  ; SSE42: cost of 7 {{.*}} %V4I32 = fptosi
  ; AVX1: cost of 1 {{.*}} %V4I32 = fptosi
  ; AVX2: cost of 1 {{.*}} %V4I32 = fptosi
  ; AVX512: cost of 1 {{.*}} %V4I32 = fptosi
  %V4I32 = fptosi <4 x double> undef to <4 x i32>
  ; SSE2: cost of 15 {{.*}} %V8I32 = fptosi
  ; SSE42: cost of 15 {{.*}} %V8I32 = fptosi
  ; AVX1: cost of 3 {{.*}} %V8I32 = fptosi
  ; AVX2: cost of 3 {{.*}} %V8I32 = fptosi
  ; AVX512: cost of 1 {{.*}} %V8I32 = fptosi
  %V8I32 = fptosi <8 x double> undef to <8 x i32>

  ret i32 undef
}

; CHECK-LABEL: 'fptosi_double_i16'
define i32 @fptosi_double_i16(i32 %arg) {
  ; SSE2: cost of 1 {{.*}} %I16 = fptosi
  ; SSE42: cost of 1 {{.*}} %I16 = fptosi
  ; AVX1: cost of 1 {{.*}} %I16 = fptosi
  ; AVX2: cost of 1 {{.*}} %I16 = fptosi
  ; AVX512: cost of 1 {{.*}} %I16 = fptosi
  %I16 = fptosi double undef to i16
  ; SSE2: cost of 6 {{.*}} %V2I16 = fptosi
  ; SSE42: cost of 6 {{.*}} %V2I16 = fptosi
  ; AVX1: cost of 6 {{.*}} %V2I16 = fptosi
  ; AVX2: cost of 6 {{.*}} %V2I16 = fptosi
  ; AVX512F: cost of 6 {{.*}} %V2I16 = fptosi
  ; AVX512DQ: cost of 1 {{.*}} %V2I16 = fptosi
  %V2I16 = fptosi <2 x double> undef to <2 x i16>
  ; SSE2: cost of 13 {{.*}} %V4I16 = fptosi
  ; SSE42: cost of 13 {{.*}} %V4I16 = fptosi
  ; AVX1: cost of 1 {{.*}} %V4I16 = fptosi
  ; AVX2: cost of 1 {{.*}} %V4I16 = fptosi
  ; AVX512: cost of 1 {{.*}} %V4I16 = fptosi
  %V4I16 = fptosi <4 x double> undef to <4 x i16>
  ; SSE2: cost of 27 {{.*}} %V8I16 = fptosi
  ; SSE42: cost of 27 {{.*}} %V8I16 = fptosi
  ; AVX1: cost of 3 {{.*}} %V8I16 = fptosi
  ; AVX2: cost of 3 {{.*}} %V8I16 = fptosi
  ; AVX512: cost of 1 {{.*}} %V8I16 = fptosi
  %V8I16 = fptosi <8 x double> undef to <8 x i16>

  ret i32 undef
}

; CHECK-LABEL: 'fptosi_double_i8'
define i32 @fptosi_double_i8(i32 %arg) {
  ; SSE2: cost of 1 {{.*}} %I8 = fptosi
  ; SSE42: cost of 1 {{.*}} %I8 = fptosi
  ; AVX1: cost of 1 {{.*}} %I8 = fptosi
  ; AVX2: cost of 1 {{.*}} %I8 = fptosi
  ; AVX512: cost of 1 {{.*}} %I8 = fptosi
  %I8 = fptosi double undef to i8
  ; SSE2: cost of 6 {{.*}} %V2I8 = fptosi
  ; SSE42: cost of 6 {{.*}} %V2I8 = fptosi
  ; AVX1: cost of 6 {{.*}} %V2I8 = fptosi
  ; AVX2: cost of 6 {{.*}} %V2I8 = fptosi
  ; AVX512F: cost of 6 {{.*}} %V2I8 = fptosi
  ; AVX512DQ: cost of 1 {{.*}} %V2I8 = fptosi
  %V2I8 = fptosi <2 x double> undef to <2 x i8>
  ; SSE2: cost of 13 {{.*}} %V4I8 = fptosi
  ; SSE42: cost of 13 {{.*}} %V4I8 = fptosi
  ; AVX1: cost of 1 {{.*}} %V4I8 = fptosi
  ; AVX2: cost of 1 {{.*}} %V4I8 = fptosi
  ; AVX512: cost of 1 {{.*}} %V4I8 = fptosi
  %V4I8 = fptosi <4 x double> undef to <4 x i8>
  ; SSE2: cost of 27 {{.*}} %V8I8 = fptosi
  ; SSE42: cost of 27 {{.*}} %V8I8 = fptosi
  ; AVX1: cost of 3 {{.*}} %V8I8 = fptosi
  ; AVX2: cost of 3 {{.*}} %V8I8 = fptosi
  ; AVX512: cost of 1 {{.*}} %V8I8 = fptosi
  %V8I8 = fptosi <8 x double> undef to <8 x i8>

  ret i32 undef
}

; CHECK-LABEL: 'fptosi_float_i64'
define i32 @fptosi_float_i64(i32 %arg) {
  ; SSE2: cost of 1 {{.*}} %I64 = fptosi
  ; SSE42: cost of 1 {{.*}} %I64 = fptosi
  ; AVX1: cost of 1 {{.*}} %I64 = fptosi
  ; AVX2: cost of 1 {{.*}} %I64 = fptosi
  ; AVX512: cost of 1 {{.*}} %I64 = fptosi
  %I64 = fptosi float undef to i64
  ; SSE2: cost of 6 {{.*}} %V2I64 = fptosi
  ; SSE42: cost of 6 {{.*}} %V2I64 = fptosi
  ; AVX1: cost of 6 {{.*}} %V2I64 = fptosi
  ; AVX2: cost of 6 {{.*}} %V2I64 = fptosi
  ; AVX512F: cost of 6 {{.*}} %V2I64 = fptosi
  ; AVX512DQ: cost of 1 {{.*}} %V2I64 = fptosi
  %V2I64 = fptosi <2 x float> undef to <2 x i64>
  ; SSE2: cost of 13 {{.*}} %V4I64 = fptosi
  ; SSE42: cost of 13 {{.*}} %V4I64 = fptosi
  ; AVX1: cost of 12 {{.*}} %V4I64 = fptosi
  ; AVX2: cost of 12 {{.*}} %V4I64 = fptosi
  ; AVX512F: cost of 12 {{.*}} %V4I64 = fptosi
  ; AVX512DQ: cost of 1 {{.*}} %V4I64 = fptosi
  %V4I64 = fptosi <4 x float> undef to <4 x i64>
  ; SSE2: cost of 27 {{.*}} %V8I64 = fptosi
  ; SSE42: cost of 27 {{.*}} %V8I64 = fptosi
  ; AVX1: cost of 25 {{.*}} %V8I64 = fptosi
  ; AVX2: cost of 25 {{.*}} %V8I64 = fptosi
  ; AVX512F: cost of 24 {{.*}} %V8I64 = fptosi
  ; AVX512DQ: cost of 1 {{.*}} %V8I64 = fptosi
  %V8I64 = fptosi <8 x float> undef to <8 x i64>
  ; SSE2: cost of 55 {{.*}} %V16I64 = fptosi
  ; SSE42: cost of 55 {{.*}} %V16I64 = fptosi
  ; AVX1: cost of 51 {{.*}} %V16I64 = fptosi
  ; AVX2: cost of 51 {{.*}} %V16I64 = fptosi
  ; AVX512F: cost of 49 {{.*}} %V16I64 = fptosi
  ; AVX512DQ: cost of 3 {{.*}} %V16I64 = fptosi
  %V16I64 = fptosi <16 x float> undef to <16 x i64>

  ret i32 undef
}

; CHECK-LABEL: 'fptosi_float_i32'
define i32 @fptosi_float_i32(i32 %arg) {
  ; SSE2: cost of 1 {{.*}} %I32 = fptosi
  ; SSE42: cost of 1 {{.*}} %I32 = fptosi
  ; AVX1: cost of 1 {{.*}} %I32 = fptosi
  ; AVX2: cost of 1 {{.*}} %I32 = fptosi
  ; AVX512: cost of 1 {{.*}} %I32 = fptosi
  %I32 = fptosi float undef to i32
  ; SSE2: cost of 1 {{.*}} %V4I32 = fptosi
  ; SSE42: cost of 1 {{.*}} %V4I32 = fptosi
  ; AVX1: cost of 1 {{.*}} %V4I32 = fptosi
  ; AVX2: cost of 1 {{.*}} %V4I32 = fptosi
  ; AVX512: cost of 1 {{.*}} %V4I32 = fptosi
  %V4I32 = fptosi <4 x float> undef to <4 x i32>
  ; SSE2: cost of 1 {{.*}} %V8I32 = fptosi
  ; SSE42: cost of 1 {{.*}} %V8I32 = fptosi
  ; AVX1: cost of 1 {{.*}} %V8I32 = fptosi
  ; AVX2: cost of 1 {{.*}} %V8I32 = fptosi
  ; AVX512: cost of 1 {{.*}} %V8I32 = fptosi
  %V8I32 = fptosi <8 x float> undef to <8 x i32>
  ; SSE2: cost of 1 {{.*}} %V16I32 = fptosi
  ; SSE42: cost of 1 {{.*}} %V16I32 = fptosi
  ; AVX1: cost of 1 {{.*}} %V16I32 = fptosi
  ; AVX2: cost of 1 {{.*}} %V16I32 = fptosi
  ; AVX512: cost of 1 {{.*}} %V16I32 = fptosi
  %V16I32 = fptosi <16 x float> undef to <16 x i32>

  ret i32 undef
}

; CHECK-LABEL: 'fptosi_float_i16'
define i32 @fptosi_float_i16(i32 %arg) {
  ; SSE2: cost of 1 {{.*}} %I16 = fptosi
  ; SSE42: cost of 1 {{.*}} %I16 = fptosi
  ; AVX1: cost of 1 {{.*}} %I16 = fptosi
  ; AVX2: cost of 1 {{.*}} %I16 = fptosi
  ; AVX512: cost of 1 {{.*}} %I16 = fptosi
  %I16 = fptosi float undef to i16
  ; SSE2: cost of 1 {{.*}} %V4I16 = fptosi
  ; SSE42: cost of 1 {{.*}} %V4I16 = fptosi
  ; AVX1: cost of 1 {{.*}} %V4I16 = fptosi
  ; AVX2: cost of 1 {{.*}} %V4I16 = fptosi
  ; AVX512: cost of 1 {{.*}} %V4I16 = fptosi
  %V4I16 = fptosi <4 x float> undef to <4 x i16>
  ; SSE2: cost of 3 {{.*}} %V8I16 = fptosi
  ; SSE42: cost of 3 {{.*}} %V8I16 = fptosi
  ; AVX1: cost of 1 {{.*}} %V8I16 = fptosi
  ; AVX2: cost of 1 {{.*}} %V8I16 = fptosi
  ; AVX512: cost of 1 {{.*}} %V8I16 = fptosi
  %V8I16 = fptosi <8 x float> undef to <8 x i16>
  ; SSE2: cost of 7 {{.*}} %V16I16 = fptosi
  ; SSE42: cost of 7 {{.*}} %V16I16 = fptosi
  ; AVX1: cost of 3 {{.*}} %V16I16 = fptosi
  ; AVX2: cost of 3 {{.*}} %V16I16 = fptosi
  ; AVX512: cost of 1 {{.*}} %V16I16 = fptosi
  %V16I16 = fptosi <16 x float> undef to <16 x i16>

  ret i32 undef
}

; CHECK-LABEL: 'fptosi_float_i8'
define i32 @fptosi_float_i8(i32 %arg) {
  ; SSE2: cost of 1 {{.*}} %I8 = fptosi
  ; SSE42: cost of 1 {{.*}} %I8 = fptosi
  ; AVX1: cost of 1 {{.*}} %I8 = fptosi
  ; AVX2: cost of 1 {{.*}} %I8 = fptosi
  ; AVX512: cost of 1 {{.*}} %I8 = fptosi
  %I8 = fptosi float undef to i8
  ; SSE2: cost of 1 {{.*}} %V4I8 = fptosi
  ; SSE42: cost of 1 {{.*}} %V4I8 = fptosi
  ; AVX1: cost of 1 {{.*}} %V4I8 = fptosi
  ; AVX2: cost of 1 {{.*}} %V4I8 = fptosi
  ; AVX512: cost of 1 {{.*}} %V4I8 = fptosi
  %V4I8 = fptosi <4 x float> undef to <4 x i8>
  ; SSE2: cost of 3 {{.*}} %V8I8 = fptosi
  ; SSE42: cost of 3 {{.*}} %V8I8 = fptosi
  ; AVX1: cost of 7 {{.*}} %V8I8 = fptosi
  ; AVX2: cost of 7 {{.*}} %V8I8 = fptosi
  ; AVX512: cost of 7 {{.*}} %V8I8 = fptosi
  %V8I8 = fptosi <8 x float> undef to <8 x i8>
  ; SSE2: cost of 7 {{.*}} %V16I8 = fptosi
  ; SSE42: cost of 7 {{.*}} %V16I8 = fptosi
  ; AVX1: cost of 15 {{.*}} %V16I8 = fptosi
  ; AVX2: cost of 15 {{.*}} %V16I8 = fptosi
  ; AVX512: cost of 1 {{.*}} %V16I8 = fptosi
  %V16I8 = fptosi <16 x float> undef to <16 x i8>

  ret i32 undef
}

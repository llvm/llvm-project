; RUN: opt -mtriple=x86_64-apple-darwin -mattr=+sse2 -cost-model -analyze < %s | FileCheck --check-prefix=SSE --check-prefix=SSE2 %s
; RUN: opt -mtriple=x86_64-apple-darwin -mattr=+sse4.2 -cost-model -analyze < %s | FileCheck --check-prefix=SSE --check-prefix=SSE42 %s
; RUN: opt -mtriple=x86_64-apple-darwin -mattr=+avx  -cost-model -analyze < %s | FileCheck --check-prefix=AVX --check-prefix=AVX1 %s
; RUN: opt -mtriple=x86_64-apple-darwin -mattr=+avx2 -cost-model -analyze < %s | FileCheck --check-prefix=AVX --check-prefix=AVX2 %s
; RUN: opt -mtriple=x86_64-apple-darwin -mattr=+avx512f -cost-model -analyze < %s | FileCheck --check-prefix=AVX512 --check-prefix=AVX512F %s
; RUN: opt -mtriple=x86_64-apple-darwin -mattr=+avx512f,+avx512dq -cost-model -analyze < %s | FileCheck --check-prefix=AVX512 --check-prefix=AVX512DQ %s

; CHECK-LABEL: 'fptoui_double_i64'
define i32 @fptoui_double_i64(i32 %arg) {
  ; SSE2: cost of 4 {{.*}} %I64 = fptoui
  ; SSE42: cost of 4 {{.*}} %I64 = fptoui
  ; AVX1: cost of 4 {{.*}} %I64 = fptoui
  ; AVX2: cost of 4 {{.*}} %I64 = fptoui
  ; AVX512: cost of 1 {{.*}} %I64 = fptoui
  %I64 = fptoui double undef to i64
  ; SSE2: cost of 12 {{.*}} %V2I64 = fptoui
  ; SSE42: cost of 12 {{.*}} %V2I64 = fptoui
  ; AVX1: cost of 12 {{.*}} %V2I64 = fptoui
  ; AVX2: cost of 12 {{.*}} %V2I64 = fptoui
  ; AVX512F: cost of 6 {{.*}} %V2I64 = fptoui
  ; AVX512DQ: cost of 1 {{.*}} %V2I64 = fptoui
  %V2I64 = fptoui <2 x double> undef to <2 x i64>
  ; SSE2: cost of 25 {{.*}} %V4I64 = fptoui
  ; SSE42: cost of 25 {{.*}} %V4I64 = fptoui
  ; AVX1: cost of 24 {{.*}} %V4I64 = fptoui
  ; AVX2: cost of 24 {{.*}} %V4I64 = fptoui
  ; AVX512F: cost of 12 {{.*}} %V4I64 = fptoui
  ; AVX512DQ: cost of 1 {{.*}} %V4I64 = fptoui
  %V4I64 = fptoui <4 x double> undef to <4 x i64>
  ; SSE2: cost of 51 {{.*}} %V8I64 = fptoui
  ; SSE42: cost of 51 {{.*}} %V8I64 = fptoui
  ; AVX1: cost of 49 {{.*}} %V8I64 = fptoui
  ; AVX2: cost of 49 {{.*}} %V8I64 = fptoui
  ; AVX512F: cost of 24 {{.*}} %V8I64 = fptoui
  ; AVX512DQ: cost of 1 {{.*}} %V8I64 = fptoui
  %V8I64 = fptoui <8 x double> undef to <8 x i64>

  ret i32 undef
}

; CHECK-LABEL: 'fptoui_double_i32'
define i32 @fptoui_double_i32(i32 %arg) {
  ; SSE2: cost of 1 {{.*}} %I32 = fptoui
  ; SSE42: cost of 1 {{.*}} %I32 = fptoui
  ; AVX1: cost of 1 {{.*}} %I32 = fptoui
  ; AVX2: cost of 1 {{.*}} %I32 = fptoui
  ; AVX512: cost of 1 {{.*}} %I32 = fptoui
  %I32 = fptoui double undef to i32
  ; SSE2: cost of 6 {{.*}} %V2I32 = fptoui
  ; SSE42: cost of 6 {{.*}} %V2I32 = fptoui
  ; AVX1: cost of 6 {{.*}} %V2I32 = fptoui
  ; AVX2: cost of 6 {{.*}} %V2I32 = fptoui
  ; AVX512F: cost of 6 {{.*}} %V2I32 = fptoui
  ; AVX512DQ: cost of 1 {{.*}} %V2I32 = fptoui
  %V2I32 = fptoui <2 x double> undef to <2 x i32>
  ; SSE2: cost of 13 {{.*}} %V4I32 = fptoui
  ; SSE42: cost of 13 {{.*}} %V4I32 = fptoui
  ; AVX1: cost of 16 {{.*}} %V4I32 = fptoui
  ; AVX2: cost of 16 {{.*}} %V4I32 = fptoui
  ; AVX512: cost of 16 {{.*}} %V4I32 = fptoui
  %V4I32 = fptoui <4 x double> undef to <4 x i32>
  ; SSE2: cost of 27 {{.*}} %V8I32 = fptoui
  ; SSE42: cost of 27 {{.*}} %V8I32 = fptoui
  ; AVX1: cost of 33 {{.*}} %V8I32 = fptoui
  ; AVX2: cost of 33 {{.*}} %V8I32 = fptoui
  ; AVX512: cost of 1 {{.*}} %V8I32 = fptoui
  %V8I32 = fptoui <8 x double> undef to <8 x i32>

  ret i32 undef
}

; CHECK-LABEL: 'fptoui_double_i16'
define i32 @fptoui_double_i16(i32 %arg) {
  ; SSE2: cost of 1 {{.*}} %I16 = fptoui
  ; SSE42: cost of 1 {{.*}} %I16 = fptoui
  ; AVX1: cost of 1 {{.*}} %I16 = fptoui
  ; AVX2: cost of 1 {{.*}} %I16 = fptoui
  ; AVX512: cost of 1 {{.*}} %I16 = fptoui
  %I16 = fptoui double undef to i16
  ; SSE2: cost of 6 {{.*}} %V2I16 = fptoui
  ; SSE42: cost of 6 {{.*}} %V2I16 = fptoui
  ; AVX1: cost of 6 {{.*}} %V2I16 = fptoui
  ; AVX2: cost of 6 {{.*}} %V2I16 = fptoui
  ; AVX512F: cost of 6 {{.*}} %V2I16 = fptoui
  ; AVX512DQ: cost of 1 {{.*}} %V2I16 = fptoui
  %V2I16 = fptoui <2 x double> undef to <2 x i16>
  ; SSE2: cost of 13 {{.*}} %V4I16 = fptoui
  ; SSE42: cost of 13 {{.*}} %V4I16 = fptoui
  ; AVX1: cost of 12 {{.*}} %V4I16 = fptoui
  ; AVX2: cost of 12 {{.*}} %V4I16 = fptoui
  ; AVX512: cost of 1 {{.*}} %V4I16 = fptoui
  %V4I16 = fptoui <4 x double> undef to <4 x i16>
  ; SSE2: cost of 27 {{.*}} %V8I16 = fptoui
  ; SSE42: cost of 27 {{.*}} %V8I16 = fptoui
  ; AVX1: cost of 25 {{.*}} %V8I16 = fptoui
  ; AVX2: cost of 25 {{.*}} %V8I16 = fptoui
  ; AVX512: cost of 2 {{.*}} %V8I16 = fptoui
  %V8I16 = fptoui <8 x double> undef to <8 x i16>

  ret i32 undef
}

; CHECK-LABEL: 'fptoui_double_i8'
define i32 @fptoui_double_i8(i32 %arg) {
  ; SSE2: cost of 1 {{.*}} %I8 = fptoui
  ; SSE42: cost of 1 {{.*}} %I8 = fptoui
  ; AVX1: cost of 1 {{.*}} %I8 = fptoui
  ; AVX2: cost of 1 {{.*}} %I8 = fptoui
  ; AVX512: cost of 1 {{.*}} %I8 = fptoui
  %I8 = fptoui double undef to i8
  ; SSE2: cost of 6 {{.*}} %V2I8 = fptoui
  ; SSE42: cost of 6 {{.*}} %V2I8 = fptoui
  ; AVX1: cost of 6 {{.*}} %V2I8 = fptoui
  ; AVX2: cost of 6 {{.*}} %V2I8 = fptoui
  ; AVX512F: cost of 6 {{.*}} %V2I8 = fptoui
  ; AVX512DQ: cost of 1 {{.*}} %V2I8 = fptoui
  %V2I8 = fptoui <2 x double> undef to <2 x i8>
  ; SSE2: cost of 13 {{.*}} %V4I8 = fptoui
  ; SSE42: cost of 13 {{.*}} %V4I8 = fptoui
  ; AVX1: cost of 12 {{.*}} %V4I8 = fptoui
  ; AVX2: cost of 12 {{.*}} %V4I8 = fptoui
  ; AVX512: cost of 1 {{.*}} %V4I8 = fptoui
  %V4I8 = fptoui <4 x double> undef to <4 x i8>
  ; SSE2: cost of 27 {{.*}} %V8I8 = fptoui
  ; SSE42: cost of 27 {{.*}} %V8I8 = fptoui
  ; AVX1: cost of 25 {{.*}} %V8I8 = fptoui
  ; AVX2: cost of 25 {{.*}} %V8I8 = fptoui
  ; AVX512: cost of 2 {{.*}} %V8I8 = fptoui
  %V8I8 = fptoui <8 x double> undef to <8 x i8>

  ret i32 undef
}

; CHECK-LABEL: 'fptoui_float_i64'
define i32 @fptoui_float_i64(i32 %arg) {
  ; SSE2: cost of 4 {{.*}} %I64 = fptoui
  ; SSE42: cost of 4 {{.*}} %I64 = fptoui
  ; AVX1: cost of 4 {{.*}} %I64 = fptoui
  ; AVX2: cost of 4 {{.*}} %I64 = fptoui
  ; AVX512: cost of 1 {{.*}} %I64 = fptoui
  %I64 = fptoui float undef to i64
  ; SSE2: cost of 12 {{.*}} %V2I64 = fptoui
  ; SSE42: cost of 12 {{.*}} %V2I64 = fptoui
  ; AVX1: cost of 12 {{.*}} %V2I64 = fptoui
  ; AVX2: cost of 12 {{.*}} %V2I64 = fptoui
  ; AVX512F: cost of 6 {{.*}} %V2I64 = fptoui
  ; AVX512DQ: cost of 1 {{.*}} %V2I64 = fptoui
  %V2I64 = fptoui <2 x float> undef to <2 x i64>
  ; SSE2: cost of 25 {{.*}} %V4I64 = fptoui
  ; SSE42: cost of 25 {{.*}} %V4I64 = fptoui
  ; AVX1: cost of 24 {{.*}} %V4I64 = fptoui
  ; AVX2: cost of 24 {{.*}} %V4I64 = fptoui
  ; AVX512F: cost of 12 {{.*}} %V4I64 = fptoui
  ; AVX512DQ: cost of 1 {{.*}} %V4I64 = fptoui
  %V4I64 = fptoui <4 x float> undef to <4 x i64>
  ; SSE2: cost of 51 {{.*}} %V8I64 = fptoui
  ; SSE42: cost of 51 {{.*}} %V8I64 = fptoui
  ; AVX1: cost of 49 {{.*}} %V8I64 = fptoui
  ; AVX2: cost of 49 {{.*}} %V8I64 = fptoui
  ; AVX512F: cost of 24 {{.*}} %V8I64 = fptoui
  ; AVX512DQ: cost of 1 {{.*}} %V8I64 = fptoui
  %V8I64 = fptoui <8 x float> undef to <8 x i64>
  ; SSE2: cost of 103 {{.*}} %V16I64 = fptoui
  ; SSE42: cost of 103 {{.*}} %V16I64 = fptoui
  ; AVX1: cost of 99 {{.*}} %V16I64 = fptoui
  ; AVX2: cost of 99 {{.*}} %V16I64 = fptoui
  ; AVX512F: cost of 49 {{.*}} %V16I64 = fptoui
  ; AVX512DQ: cost of 3 {{.*}} %V16I64 = fptoui
  %V16I64 = fptoui <16 x float> undef to <16 x i64>

  ret i32 undef
}

; CHECK-LABEL: 'fptoui_float_i32'
define i32 @fptoui_float_i32(i32 %arg) {
  ; SSE2: cost of 1 {{.*}} %I32 = fptoui
  ; SSE42: cost of 1 {{.*}} %I32 = fptoui
  ; AVX1: cost of 1 {{.*}} %I32 = fptoui
  ; AVX2: cost of 1 {{.*}} %I32 = fptoui
  ; AVX512: cost of 1 {{.*}} %I32 = fptoui
  %I32 = fptoui float undef to i32
  ; SSE2: cost of 12 {{.*}} %V4I32 = fptoui
  ; SSE42: cost of 12 {{.*}} %V4I32 = fptoui
  ; AVX1: cost of 12 {{.*}} %V4I32 = fptoui
  ; AVX2: cost of 12 {{.*}} %V4I32 = fptoui
  ; AVX512: cost of 1 {{.*}} %V4I32 = fptoui
  %V4I32 = fptoui <4 x float> undef to <4 x i32>
  ; SSE2: cost of 25 {{.*}} %V8I32 = fptoui
  ; SSE42: cost of 25 {{.*}} %V8I32 = fptoui
  ; AVX1: cost of 32 {{.*}} %V8I32 = fptoui
  ; AVX2: cost of 32 {{.*}} %V8I32 = fptoui
  ; AVX512: cost of 1 {{.*}} %V8I32 = fptoui
  %V8I32 = fptoui <8 x float> undef to <8 x i32>
  ; SSE2: cost of 51 {{.*}} %V16I32 = fptoui
  ; SSE42: cost of 51 {{.*}} %V16I32 = fptoui
  ; AVX1: cost of 65 {{.*}} %V16I32 = fptoui
  ; AVX2: cost of 65 {{.*}} %V16I32 = fptoui
  ; AVX512: cost of 1 {{.*}} %V16I32 = fptoui
  %V16I32 = fptoui <16 x float> undef to <16 x i32>

  ret i32 undef
}

; CHECK-LABEL: 'fptoui_float_i16'
define i32 @fptoui_float_i16(i32 %arg) {
  ; SSE2: cost of 1 {{.*}} %I16 = fptoui
  ; SSE42: cost of 1 {{.*}} %I16 = fptoui
  ; AVX1: cost of 1 {{.*}} %I16 = fptoui
  ; AVX2: cost of 1 {{.*}} %I16 = fptoui
  ; AVX512: cost of 1 {{.*}} %I16 = fptoui
  %I16 = fptoui float undef to i16
  ; SSE2: cost of 12 {{.*}} %V4I16 = fptoui
  ; SSE42: cost of 12 {{.*}} %V4I16 = fptoui
  ; AVX1: cost of 12 {{.*}} %V4I16 = fptoui
  ; AVX2: cost of 12 {{.*}} %V4I16 = fptoui
  ; AVX512: cost of 1 {{.*}} %V4I16 = fptoui
  %V4I16 = fptoui <4 x float> undef to <4 x i16>
  ; SSE2: cost of 25 {{.*}} %V8I16 = fptoui
  ; SSE42: cost of 25 {{.*}} %V8I16 = fptoui
  ; AVX1: cost of 1 {{.*}} %V8I16 = fptoui
  ; AVX2: cost of 1 {{.*}} %V8I16 = fptoui
  ; AVX512: cost of 1 {{.*}} %V8I16 = fptoui
  %V8I16 = fptoui <8 x float> undef to <8 x i16>
  ; SSE2: cost of 51 {{.*}} %V16I16 = fptoui
  ; SSE42: cost of 51 {{.*}} %V16I16 = fptoui
  ; AVX1: cost of 3 {{.*}} %V16I16 = fptoui
  ; AVX2: cost of 3 {{.*}} %V16I16 = fptoui
  ; AVX512: cost of 2 {{.*}} %V16I16 = fptoui
  %V16I16 = fptoui <16 x float> undef to <16 x i16>

  ret i32 undef
}

; CHECK-LABEL: 'fptoui_float_i8'
define i32 @fptoui_float_i8(i32 %arg) {
  ; SSE2: cost of 1 {{.*}} %I8 = fptoui
  ; SSE42: cost of 1 {{.*}} %I8 = fptoui
  ; AVX1: cost of 1 {{.*}} %I8 = fptoui
  ; AVX2: cost of 1 {{.*}} %I8 = fptoui
  ; AVX512: cost of 1 {{.*}} %I8 = fptoui
  %I8 = fptoui float undef to i8
  ; SSE2: cost of 12 {{.*}} %V4I8 = fptoui
  ; SSE42: cost of 12 {{.*}} %V4I8 = fptoui
  ; AVX1: cost of 12 {{.*}} %V4I8 = fptoui
  ; AVX2: cost of 12 {{.*}} %V4I8 = fptoui
  ; AVX512: cost of 1 {{.*}} %V4I8 = fptoui
  %V4I8 = fptoui <4 x float> undef to <4 x i8>
  ; SSE2: cost of 25 {{.*}} %V8I8 = fptoui
  ; SSE42: cost of 25 {{.*}} %V8I8 = fptoui
  ; AVX1: cost of 1 {{.*}} %V8I8 = fptoui
  ; AVX2: cost of 1 {{.*}} %V8I8 = fptoui
  ; AVX512: cost of 1 {{.*}} %V8I8 = fptoui
  %V8I8 = fptoui <8 x float> undef to <8 x i8>
  ; SSE2: cost of 51 {{.*}} %V16I8 = fptoui
  ; SSE42: cost of 51 {{.*}} %V16I8 = fptoui
  ; AVX1: cost of 3 {{.*}} %V16I8 = fptoui
  ; AVX2: cost of 3 {{.*}} %V16I8 = fptoui
  ; AVX512: cost of 2 {{.*}} %V16I8 = fptoui
  %V16I8 = fptoui <16 x float> undef to <16 x i8>

  ret i32 undef
}

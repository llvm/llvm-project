; RUN: opt < %s  -cost-model -analyze -mtriple=x86_64-apple-macosx10.8.0 -mattr=+ssse3 | FileCheck %s --check-prefix=CHECK --check-prefix=SSSE3
; RUN: opt < %s  -cost-model -analyze -mtriple=x86_64-apple-macosx10.8.0 -mattr=+sse4.2 | FileCheck %s --check-prefix=CHECK --check-prefix=SSE42
; RUN: opt < %s  -cost-model -analyze -mtriple=x86_64-apple-macosx10.8.0 -mattr=+avx | FileCheck %s --check-prefix=CHECK --check-prefix=AVX
; RUN: opt < %s  -cost-model -analyze -mtriple=x86_64-apple-macosx10.8.0 -mattr=+avx2 | FileCheck %s --check-prefix=CHECK --check-prefix=AVX2
; RUN: opt < %s  -cost-model -analyze -mtriple=x86_64-apple-macosx10.8.0 -mattr=+avx512f | FileCheck %s --check-prefix=CHECK --check-prefix=AVX512 --check-prefix=AVX512F
; RUN: opt < %s  -cost-model -analyze -mtriple=x86_64-apple-macosx10.8.0 -mattr=+avx512f,+avx512bw | FileCheck %s --check-prefix=CHECK --check-prefix=AVX512 --check-prefix=AVX512BW

target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128-n8:16:32:64-S128"
target triple = "x86_64-apple-macosx10.8.0"

; CHECK-LABEL: 'add'
define i32 @add(i32 %arg) {
  ; SSSE3: cost of 1 {{.*}} %A = add
  ; SSE42: cost of 1 {{.*}} %A = add
  ; AVX: cost of 1 {{.*}} %A = add
  ; AVX2: cost of 1 {{.*}} %A = add
  ; AVX512: cost of 1 {{.*}} %A = add
  %A = add <2 x i64> undef, undef
  ; SSSE3: cost of 2 {{.*}} %B = add
  ; SSE42: cost of 2 {{.*}} %B = add
  ; AVX: cost of 4 {{.*}} %B = add
  ; AVX2: cost of 1 {{.*}} %B = add
  ; AVX512: cost of 1 {{.*}} %B = add
  %B = add <4 x i64> undef, undef
  ; SSSE3: cost of 4 {{.*}} %C = add
  ; SSE42: cost of 4 {{.*}} %C = add
  ; AVX: cost of 8 {{.*}} %C = add
  ; AVX2: cost of 2 {{.*}} %C = add
  ; AVX512: cost of 1 {{.*}} %C = add
  %C = add <8 x i64> undef, undef

  ; SSSE3: cost of 1 {{.*}} %D = add
  ; SSE42: cost of 1 {{.*}} %D = add
  ; AVX: cost of 1 {{.*}} %D = add
  ; AVX2: cost of 1 {{.*}} %D = add
  ; AVX512: cost of 1 {{.*}} %D = add
  %D = add <4 x i32> undef, undef
  ; SSSE3: cost of 2 {{.*}} %E = add
  ; SSE42: cost of 2 {{.*}} %E = add
  ; AVX: cost of 4 {{.*}} %E = add
  ; AVX2: cost of 1 {{.*}} %E = add
  ; AVX512: cost of 1 {{.*}} %E = add
  %E = add <8 x i32> undef, undef
  ; SSSE3: cost of 4 {{.*}} %F = add
  ; SSE42: cost of 4 {{.*}} %F = add
  ; AVX: cost of 8 {{.*}} %F = add
  ; AVX2: cost of 2 {{.*}} %F = add
  ; AVX512: cost of 1 {{.*}} %F = add
  %F = add <16 x i32> undef, undef

  ; SSSE3: cost of 1 {{.*}} %G = add
  ; SSE42: cost of 1 {{.*}} %G = add
  ; AVX: cost of 1 {{.*}} %G = add
  ; AVX2: cost of 1 {{.*}} %G = add
  ; AVX512: cost of 1 {{.*}} %G = add
  %G = add <8 x i16> undef, undef
  ; SSSE3: cost of 2 {{.*}} %H = add
  ; SSE42: cost of 2 {{.*}} %H = add
  ; AVX: cost of 2 {{.*}} %H = add
  ; AVX2: cost of 1 {{.*}} %H = add
  ; AVX512: cost of 1 {{.*}} %H = add
  %H = add <16 x i16> undef, undef
  ; SSSE3: cost of 4 {{.*}} %I = add
  ; SSE42: cost of 4 {{.*}} %I = add
  ; AVX: cost of 4 {{.*}} %I = add
  ; AVX2: cost of 2 {{.*}} %I = add
  ; AVX512F: cost of 2 {{.*}} %I = add
  ; AVX512BW: cost of 1 {{.*}} %I = add
  %I = add <32 x i16> undef, undef

  ; SSSE3: cost of 1 {{.*}} %J = add
  ; SSE42: cost of 1 {{.*}} %J = add
  ; AVX: cost of 1 {{.*}} %J = add
  ; AVX2: cost of 1 {{.*}} %J = add
  ; AVX512: cost of 1 {{.*}} %J = add
  %J = add <16 x i8> undef, undef
  ; SSSE3: cost of 2 {{.*}} %K = add
  ; SSE42: cost of 2 {{.*}} %K = add
  ; AVX: cost of 2 {{.*}} %K = add
  ; AVX2: cost of 1 {{.*}} %K = add
  ; AVX512: cost of 1 {{.*}} %K = add
  %K = add <32 x i8> undef, undef
  ; SSSE3: cost of 4 {{.*}} %L = add
  ; SSE42: cost of 4 {{.*}} %L = add
  ; AVX: cost of 4 {{.*}} %L = add
  ; AVX2: cost of 2 {{.*}} %L = add
  ; AVX512F: cost of 2 {{.*}} %L = add
  ; AVX512BW: cost of 1 {{.*}} %L = add
  %L = add <64 x i8> undef, undef

  ret i32 undef
}

; CHECK-LABEL: 'sub'
define i32 @sub(i32 %arg) {
  ; SSSE3: cost of 1 {{.*}} %A = sub
  ; SSE42: cost of 1 {{.*}} %A = sub
  ; AVX: cost of 1 {{.*}} %A = sub
  ; AVX2: cost of 1 {{.*}} %A = sub
  ; AVX512: cost of 1 {{.*}} %A = sub
  %A = sub <2 x i64> undef, undef
  ; SSSE3: cost of 2 {{.*}} %B = sub
  ; SSE42: cost of 2 {{.*}} %B = sub
  ; AVX: cost of 4 {{.*}} %B = sub
  ; AVX2: cost of 1 {{.*}} %B = sub
  ; AVX512: cost of 1 {{.*}} %B = sub
  %B = sub <4 x i64> undef, undef
  ; SSSE3: cost of 4 {{.*}} %C = sub
  ; SSE42: cost of 4 {{.*}} %C = sub
  ; AVX: cost of 8 {{.*}} %C = sub
  ; AVX2: cost of 2 {{.*}} %C = sub
  ; AVX512: cost of 1 {{.*}} %C = sub
  %C = sub <8 x i64> undef, undef

  ; SSSE3: cost of 1 {{.*}} %D = sub
  ; SSE42: cost of 1 {{.*}} %D = sub
  ; AVX: cost of 1 {{.*}} %D = sub
  ; AVX2: cost of 1 {{.*}} %D = sub
  ; AVX512: cost of 1 {{.*}} %D = sub
  %D = sub <4 x i32> undef, undef
  ; SSSE3: cost of 2 {{.*}} %E = sub
  ; SSE42: cost of 2 {{.*}} %E = sub
  ; AVX: cost of 4 {{.*}} %E = sub
  ; AVX2: cost of 1 {{.*}} %E = sub
  ; AVX512: cost of 1 {{.*}} %E = sub
  %E = sub <8 x i32> undef, undef
  ; SSSE3: cost of 4 {{.*}} %F = sub
  ; SSE42: cost of 4 {{.*}} %F = sub
  ; AVX: cost of 8 {{.*}} %F = sub
  ; AVX2: cost of 2 {{.*}} %F = sub
  ; AVX512: cost of 1 {{.*}} %F = sub
  %F = sub <16 x i32> undef, undef

  ; SSSE3: cost of 1 {{.*}} %G = sub
  ; SSE42: cost of 1 {{.*}} %G = sub
  ; AVX: cost of 1 {{.*}} %G = sub
  ; AVX2: cost of 1 {{.*}} %G = sub
  ; AVX512: cost of 1 {{.*}} %G = sub
  %G = sub <8 x i16> undef, undef
  ; SSSE3: cost of 2 {{.*}} %H = sub
  ; SSE42: cost of 2 {{.*}} %H = sub
  ; AVX: cost of 2 {{.*}} %H = sub
  ; AVX2: cost of 1 {{.*}} %H = sub
  ; AVX512: cost of 1 {{.*}} %H = sub
  %H = sub <16 x i16> undef, undef
  ; SSSE3: cost of 4 {{.*}} %I = sub
  ; SSE42: cost of 4 {{.*}} %I = sub
  ; AVX: cost of 4 {{.*}} %I = sub
  ; AVX2: cost of 2 {{.*}} %I = sub
  ; AVX512F: cost of 2 {{.*}} %I = sub
  ; AVX512BW: cost of 1 {{.*}} %I = sub
  %I = sub <32 x i16> undef, undef

  ; SSSE3: cost of 1 {{.*}} %J = sub
  ; SSE42: cost of 1 {{.*}} %J = sub
  ; AVX: cost of 1 {{.*}} %J = sub
  ; AVX2: cost of 1 {{.*}} %J = sub
  ; AVX512: cost of 1 {{.*}} %J = sub
  %J = sub <16 x i8> undef, undef
  ; SSSE3: cost of 2 {{.*}} %K = sub
  ; SSE42: cost of 2 {{.*}} %K = sub
  ; AVX: cost of 2 {{.*}} %K = sub
  ; AVX2: cost of 1 {{.*}} %K = sub
  ; AVX512: cost of 1 {{.*}} %K = sub
  %K = sub <32 x i8> undef, undef
  ; SSSE3: cost of 4 {{.*}} %L = sub
  ; SSE42: cost of 4 {{.*}} %L = sub
  ; AVX: cost of 4 {{.*}} %L = sub
  ; AVX2: cost of 2 {{.*}} %L = sub
  ; AVX512F: cost of 2 {{.*}} %L = sub
  ; AVX512BW: cost of 1 {{.*}} %L = sub
  %L = sub <64 x i8> undef, undef

  ret i32 undef
}

; CHECK-LABEL: 'or'
define i32 @or(i32 %arg) {
  ; SSSE3: cost of 1 {{.*}} %A = or
  ; SSE42: cost of 1 {{.*}} %A = or
  ; AVX: cost of 1 {{.*}} %A = or
  ; AVX2: cost of 1 {{.*}} %A = or
  ; AVX512: cost of 1 {{.*}} %A = or
  %A = or <2 x i64> undef, undef
  ; SSSE3: cost of 2 {{.*}} %B = or
  ; SSE42: cost of 2 {{.*}} %B = or
  ; AVX: cost of 1 {{.*}} %B = or
  ; AVX2: cost of 1 {{.*}} %B = or
  ; AVX512: cost of 1 {{.*}} %B = or
  %B = or <4 x i64> undef, undef
  ; SSSE3: cost of 4 {{.*}} %C = or
  ; SSE42: cost of 4 {{.*}} %C = or
  ; AVX: cost of 2 {{.*}} %C = or
  ; AVX2: cost of 2 {{.*}} %C = or
  ; AVX512: cost of 1 {{.*}} %C = or
  %C = or <8 x i64> undef, undef

  ; SSSE3: cost of 1 {{.*}} %D = or
  ; SSE42: cost of 1 {{.*}} %D = or
  ; AVX: cost of 1 {{.*}} %D = or
  ; AVX2: cost of 1 {{.*}} %D = or
  ; AVX512: cost of 1 {{.*}} %D = or
  %D = or <4 x i32> undef, undef
  ; SSSE3: cost of 2 {{.*}} %E = or
  ; SSE42: cost of 2 {{.*}} %E = or
  ; AVX: cost of 1 {{.*}} %E = or
  ; AVX2: cost of 1 {{.*}} %E = or
  ; AVX512: cost of 1 {{.*}} %E = or
  %E = or <8 x i32> undef, undef
  ; SSSE3: cost of 4 {{.*}} %F = or
  ; SSE42: cost of 4 {{.*}} %F = or
  ; AVX: cost of 2 {{.*}} %F = or
  ; AVX2: cost of 2 {{.*}} %F = or
  ; AVX512: cost of 1 {{.*}} %F = or
  %F = or <16 x i32> undef, undef

  ; SSSE3: cost of 1 {{.*}} %G = or
  ; SSE42: cost of 1 {{.*}} %G = or
  ; AVX: cost of 1 {{.*}} %G = or
  ; AVX2: cost of 1 {{.*}} %G = or
  ; AVX512: cost of 1 {{.*}} %G = or
  %G = or <8 x i16> undef, undef
  ; SSSE3: cost of 2 {{.*}} %H = or
  ; SSE42: cost of 2 {{.*}} %H = or
  ; AVX: cost of 1 {{.*}} %H = or
  ; AVX2: cost of 1 {{.*}} %H = or
  ; AVX512: cost of 1 {{.*}} %H = or
  %H = or <16 x i16> undef, undef
  ; SSSE3: cost of 4 {{.*}} %I = or
  ; SSE42: cost of 4 {{.*}} %I = or
  ; AVX: cost of 2 {{.*}} %I = or
  ; AVX2: cost of 2 {{.*}} %I = or
  ; AVX512F: cost of 2 {{.*}} %I = or
  ; AVX512BW: cost of 1 {{.*}} %I = or
  %I = or <32 x i16> undef, undef

  ; SSSE3: cost of 1 {{.*}} %J = or
  ; SSE42: cost of 1 {{.*}} %J = or
  ; AVX: cost of 1 {{.*}} %J = or
  ; AVX2: cost of 1 {{.*}} %J = or
  ; AVX512: cost of 1 {{.*}} %J = or
  %J = or <16 x i8> undef, undef
  ; SSSE3: cost of 2 {{.*}} %K = or
  ; SSE42: cost of 2 {{.*}} %K = or
  ; AVX: cost of 1 {{.*}} %K = or
  ; AVX2: cost of 1 {{.*}} %K = or
  ; AVX512: cost of 1 {{.*}} %K = or
  %K = or <32 x i8> undef, undef
  ; SSSE3: cost of 4 {{.*}} %L = or
  ; SSE42: cost of 4 {{.*}} %L = or
  ; AVX: cost of 2 {{.*}} %L = or
  ; AVX2: cost of 2 {{.*}} %L = or
  ; AVX512F: cost of 2 {{.*}} %L = or
  ; AVX512BW: cost of 1 {{.*}} %L = or
  %L = or <64 x i8> undef, undef

  ret i32 undef
}

; CHECK-LABEL: 'xor'
define i32 @xor(i32 %arg) {
  ; SSSE3: cost of 1 {{.*}} %A = xor
  ; SSE42: cost of 1 {{.*}} %A = xor
  ; AVX: cost of 1 {{.*}} %A = xor
  ; AVX2: cost of 1 {{.*}} %A = xor
  ; AVX512: cost of 1 {{.*}} %A = xor
  %A = xor <2 x i64> undef, undef
  ; SSSE3: cost of 2 {{.*}} %B = xor
  ; SSE42: cost of 2 {{.*}} %B = xor
  ; AVX: cost of 1 {{.*}} %B = xor
  ; AVX2: cost of 1 {{.*}} %B = xor
  ; AVX512: cost of 1 {{.*}} %B = xor
  %B = xor <4 x i64> undef, undef
  ; SSSE3: cost of 4 {{.*}} %C = xor
  ; SSE42: cost of 4 {{.*}} %C = xor
  ; AVX: cost of 2 {{.*}} %C = xor
  ; AVX2: cost of 2 {{.*}} %C = xor
  ; AVX512: cost of 1 {{.*}} %C = xor
  %C = xor <8 x i64> undef, undef

  ; SSSE3: cost of 1 {{.*}} %D = xor
  ; SSE42: cost of 1 {{.*}} %D = xor
  ; AVX: cost of 1 {{.*}} %D = xor
  ; AVX2: cost of 1 {{.*}} %D = xor
  ; AVX512: cost of 1 {{.*}} %D = xor
  %D = xor <4 x i32> undef, undef
  ; SSSE3: cost of 2 {{.*}} %E = xor
  ; SSE42: cost of 2 {{.*}} %E = xor
  ; AVX: cost of 1 {{.*}} %E = xor
  ; AVX2: cost of 1 {{.*}} %E = xor
  ; AVX512: cost of 1 {{.*}} %E = xor
  %E = xor <8 x i32> undef, undef
  ; SSSE3: cost of 4 {{.*}} %F = xor
  ; SSE42: cost of 4 {{.*}} %F = xor
  ; AVX: cost of 2 {{.*}} %F = xor
  ; AVX2: cost of 2 {{.*}} %F = xor
  ; AVX512: cost of 1 {{.*}} %F = xor
  %F = xor <16 x i32> undef, undef

  ; SSSE3: cost of 1 {{.*}} %G = xor
  ; SSE42: cost of 1 {{.*}} %G = xor
  ; AVX: cost of 1 {{.*}} %G = xor
  ; AVX2: cost of 1 {{.*}} %G = xor
  ; AVX512: cost of 1 {{.*}} %G = xor
  %G = xor <8 x i16> undef, undef
  ; SSSE3: cost of 2 {{.*}} %H = xor
  ; SSE42: cost of 2 {{.*}} %H = xor
  ; AVX: cost of 1 {{.*}} %H = xor
  ; AVX2: cost of 1 {{.*}} %H = xor
  ; AVX512: cost of 1 {{.*}} %H = xor
  %H = xor <16 x i16> undef, undef
  ; SSSE3: cost of 4 {{.*}} %I = xor
  ; SSE42: cost of 4 {{.*}} %I = xor
  ; AVX: cost of 2 {{.*}} %I = xor
  ; AVX2: cost of 2 {{.*}} %I = xor
  ; AVX512F: cost of 2 {{.*}} %I = xor
  ; AVX512BW: cost of 1 {{.*}} %I = xor
  %I = xor <32 x i16> undef, undef

  ; SSSE3: cost of 1 {{.*}} %J = xor
  ; SSE42: cost of 1 {{.*}} %J = xor
  ; AVX: cost of 1 {{.*}} %J = xor
  ; AVX2: cost of 1 {{.*}} %J = xor
  ; AVX512: cost of 1 {{.*}} %J = xor
  %J = xor <16 x i8> undef, undef
  ; SSSE3: cost of 2 {{.*}} %K = xor
  ; SSE42: cost of 2 {{.*}} %K = xor
  ; AVX: cost of 1 {{.*}} %K = xor
  ; AVX2: cost of 1 {{.*}} %K = xor
  ; AVX512: cost of 1 {{.*}} %K = xor
  %K = xor <32 x i8> undef, undef
  ; SSSE3: cost of 4 {{.*}} %L = xor
  ; SSE42: cost of 4 {{.*}} %L = xor
  ; AVX: cost of 2 {{.*}} %L = xor
  ; AVX2: cost of 2 {{.*}} %L = xor
  ; AVX512F: cost of 2 {{.*}} %L = xor
  ; AVX512BW: cost of 1 {{.*}} %L = xor
  %L = xor <64 x i8> undef, undef

  ret i32 undef
}

; CHECK-LABEL: 'and'
define i32 @and(i32 %arg) {
  ; SSSE3: cost of 1 {{.*}} %A = and
  ; SSE42: cost of 1 {{.*}} %A = and
  ; AVX: cost of 1 {{.*}} %A = and
  ; AVX2: cost of 1 {{.*}} %A = and
  ; AVX512: cost of 1 {{.*}} %A = and
  %A = and <2 x i64> undef, undef
  ; SSSE3: cost of 2 {{.*}} %B = and
  ; SSE42: cost of 2 {{.*}} %B = and
  ; AVX: cost of 1 {{.*}} %B = and
  ; AVX2: cost of 1 {{.*}} %B = and
  ; AVX512: cost of 1 {{.*}} %B = and
  %B = and <4 x i64> undef, undef
  ; SSSE3: cost of 4 {{.*}} %C = and
  ; SSE42: cost of 4 {{.*}} %C = and
  ; AVX: cost of 2 {{.*}} %C = and
  ; AVX2: cost of 2 {{.*}} %C = and
  ; AVX512: cost of 1 {{.*}} %C = and
  %C = and <8 x i64> undef, undef

  ; SSSE3: cost of 1 {{.*}} %D = and
  ; SSE42: cost of 1 {{.*}} %D = and
  ; AVX: cost of 1 {{.*}} %D = and
  ; AVX2: cost of 1 {{.*}} %D = and
  ; AVX512: cost of 1 {{.*}} %D = and
  %D = and <4 x i32> undef, undef
  ; SSSE3: cost of 2 {{.*}} %E = and
  ; SSE42: cost of 2 {{.*}} %E = and
  ; AVX: cost of 1 {{.*}} %E = and
  ; AVX2: cost of 1 {{.*}} %E = and
  ; AVX512: cost of 1 {{.*}} %E = and
  %E = and <8 x i32> undef, undef
  ; SSSE3: cost of 4 {{.*}} %F = and
  ; SSE42: cost of 4 {{.*}} %F = and
  ; AVX: cost of 2 {{.*}} %F = and
  ; AVX2: cost of 2 {{.*}} %F = and
  ; AVX512: cost of 1 {{.*}} %F = and
  %F = and <16 x i32> undef, undef

  ; SSSE3: cost of 1 {{.*}} %G = and
  ; SSE42: cost of 1 {{.*}} %G = and
  ; AVX: cost of 1 {{.*}} %G = and
  ; AVX2: cost of 1 {{.*}} %G = and
  ; AVX512: cost of 1 {{.*}} %G = and
  %G = and <8 x i16> undef, undef
  ; SSSE3: cost of 2 {{.*}} %H = and
  ; SSE42: cost of 2 {{.*}} %H = and
  ; AVX: cost of 1 {{.*}} %H = and
  ; AVX2: cost of 1 {{.*}} %H = and
  ; AVX512: cost of 1 {{.*}} %H = and
  %H = and <16 x i16> undef, undef
  ; SSSE3: cost of 4 {{.*}} %I = and
  ; SSE42: cost of 4 {{.*}} %I = and
  ; AVX: cost of 2 {{.*}} %I = and
  ; AVX2: cost of 2 {{.*}} %I = and
  ; AVX512F: cost of 2 {{.*}} %I = and
  ; AVX512BW: cost of 1 {{.*}} %I = and
  %I = and <32 x i16> undef, undef

  ; SSSE3: cost of 1 {{.*}} %J = and
  ; SSE42: cost of 1 {{.*}} %J = and
  ; AVX: cost of 1 {{.*}} %J = and
  ; AVX2: cost of 1 {{.*}} %J = and
  ; AVX512: cost of 1 {{.*}} %J = and
  %J = and <16 x i8> undef, undef
  ; SSSE3: cost of 2 {{.*}} %K = and
  ; SSE42: cost of 2 {{.*}} %K = and
  ; AVX: cost of 1 {{.*}} %K = and
  ; AVX2: cost of 1 {{.*}} %K = and
  ; AVX512: cost of 1 {{.*}} %K = and
  %K = and <32 x i8> undef, undef
  ; SSSE3: cost of 4 {{.*}} %L = and
  ; SSE42: cost of 4 {{.*}} %L = and
  ; AVX: cost of 2 {{.*}} %L = and
  ; AVX2: cost of 2 {{.*}} %L = and
  ; AVX512F: cost of 2 {{.*}} %L = and
  ; AVX512BW: cost of 1 {{.*}} %L = and
  %L = and <64 x i8> undef, undef

  ret i32 undef
}

; CHECK-LABEL: 'mul'
define i32 @mul(i32 %arg) {
  ; SSSE3: cost of 9 {{.*}} %A = mul
  ; SSE42: cost of 9 {{.*}} %A = mul
  ; AVX: cost of 9 {{.*}} %A = mul
  ; AVX2: cost of 9 {{.*}} %A = mul
  ; AVX512: cost of 9 {{.*}} %A = mul
  %A = mul <2 x i64> undef, undef
  ; SSSE3: cost of 18 {{.*}} %B = mul
  ; SSE42: cost of 18 {{.*}} %B = mul
  ; AVX: cost of 18 {{.*}} %B = mul
  ; AVX2: cost of 9 {{.*}} %B = mul
  ; AVX512: cost of 9 {{.*}} %B = mul
  %B = mul <4 x i64> undef, undef
  ; SSSE3: cost of 36 {{.*}} %C = mul
  ; SSE42: cost of 36 {{.*}} %C = mul
  ; AVX: cost of 36 {{.*}} %C = mul
  ; AVX2: cost of 18 {{.*}} %C = mul
  ; AVX512: cost of 2 {{.*}} %C = mul
  %C = mul <8 x i64> undef, undef

  ; SSSE3: cost of 6 {{.*}} %D = mul
  ; SSE42: cost of 1 {{.*}} %D = mul
  ; AVX: cost of 1 {{.*}} %D = mul
  ; AVX2: cost of 1 {{.*}} %D = mul
  ; AVX512: cost of 1 {{.*}} %D = mul
  %D = mul <4 x i32> undef, undef
  ; SSSE3: cost of 12 {{.*}} %E = mul
  ; SSE42: cost of 2 {{.*}} %E = mul
  ; AVX: cost of 4 {{.*}} %E = mul
  ; AVX2: cost of 1 {{.*}} %E = mul
  ; AVX512: cost of 1 {{.*}} %E = mul
  %E = mul <8 x i32> undef, undef
  ; SSSE3: cost of 24 {{.*}} %F = mul
  ; SSE42: cost of 4 {{.*}} %F = mul
  ; AVX: cost of 8 {{.*}} %F = mul
  ; AVX2: cost of 2 {{.*}} %F = mul
  ; AVX512: cost of 1 {{.*}} %F = mul
  %F = mul <16 x i32> undef, undef

  ; SSSE3: cost of 1 {{.*}} %G = mul
  ; SSE42: cost of 1 {{.*}} %G = mul
  ; AVX: cost of 1 {{.*}} %G = mul
  ; AVX2: cost of 1 {{.*}} %G = mul
  ; AVX512: cost of 1 {{.*}} %G = mul
  %G = mul <8 x i16> undef, undef
  ; SSSE3: cost of 2 {{.*}} %H = mul
  ; SSE42: cost of 2 {{.*}} %H = mul
  ; AVX: cost of 4 {{.*}} %H = mul
  ; AVX2: cost of 1 {{.*}} %H = mul
  ; AVX512: cost of 1 {{.*}} %H = mul
  %H = mul <16 x i16> undef, undef
  ; SSSE3: cost of 4 {{.*}} %I = mul
  ; SSE42: cost of 4 {{.*}} %I = mul
  ; AVX: cost of 8 {{.*}} %I = mul
  ; AVX2: cost of 2 {{.*}} %I = mul
  ; AVX512F: cost of 2 {{.*}} %I = mul
  ; AVX512BW: cost of 1 {{.*}} %I = mul
  %I = mul <32 x i16> undef, undef

  ; SSSE3: cost of 2 {{.*}} %J = mul
  ; SSE42: cost of 2 {{.*}} %J = mul
  ; AVX: cost of 2 {{.*}} %J = mul
  ; AVX2: cost of 2 {{.*}} %J = mul
  ; AVX512: cost of 2 {{.*}} %J = mul
  %J = mul <16 x i8> undef, undef
  ; SSSE3: cost of 4 {{.*}} %K = mul
  ; SSE42: cost of 4 {{.*}} %K = mul
  ; AVX: cost of 2 {{.*}} %K = mul
  ; AVX2: cost of 2 {{.*}} %K = mul
  ; AVX512: cost of 2 {{.*}} %K = mul
  %K = mul <32 x i8> undef, undef
  ; SSSE3: cost of 8 {{.*}} %L = mul
  ; SSE42: cost of 8 {{.*}} %L = mul
  ; AVX: cost of 4 {{.*}} %L = mul
  ; AVX2: cost of 4 {{.*}} %L = mul
  ; AVX512F: cost of 4 {{.*}} %L = mul
  ; AVX512BW: cost of 2 {{.*}} %L = mul
  %L = mul <64 x i8> undef, undef

  ret i32 undef
}

; CHECK-LABEL: 'mul_2i32'
define void @mul_2i32() {
  ; A <2 x i32> gets expanded to a <2 x i64> vector.
  ; A <2 x i64> vector multiply is implemented using
  ; 3 PMULUDQ and 2 PADDS and 4 shifts.
  ; SSSE3: cost of 9 {{.*}} %A0 = mul
  ; SSE42: cost of 9 {{.*}} %A0 = mul
  ; AVX: cost of 9 {{.*}} %A0 = mul
  ; AVX2: cost of 9 {{.*}} %A0 = mul
  ; AVX512: cost of 9 {{.*}} %A0 = mul
  %A0 = mul <2 x i32> undef, undef

  ret void
}

; RUN: opt < %s  -cost-model -analyze -mtriple=x86_64-apple-macosx10.8.0 -mattr=+ssse3 | FileCheck %s --check-prefix=CHECK --check-prefix=SSSE3
; RUN: opt < %s  -cost-model -analyze -mtriple=x86_64-apple-macosx10.8.0 -mattr=+sse4.2 | FileCheck %s --check-prefix=CHECK --check-prefix=SSE42
; RUN: opt < %s  -cost-model -analyze -mtriple=x86_64-apple-macosx10.8.0 -mattr=+avx | FileCheck %s --check-prefix=CHECK --check-prefix=AVX
; RUN: opt < %s  -cost-model -analyze -mtriple=x86_64-apple-macosx10.8.0 -mattr=+avx2 | FileCheck %s --check-prefix=CHECK --check-prefix=AVX2

target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128-n8:16:32:64-S128"
target triple = "x86_64-apple-macosx10.8.0"

; CHECK-LABEL: 'add'
define i32 @add(i32 %arg) {
  ; SSSE3: cost of 1 {{.*}} %A = add
  ; SSE42: cost of 1 {{.*}} %A = add
  ; AVX: cost of 1 {{.*}} %A = add
  ; AVX2: cost of 1 {{.*}} %A = add
  %A = add <4 x i32> undef, undef
  ; SSSE3: cost of 2 {{.*}} %B = add
  ; SSE42: cost of 2 {{.*}} %B = add
  ; AVX: cost of 4 {{.*}} %B = add  
  ; AVX2: cost of 1 {{.*}} %B = add
  %B = add <8 x i32> undef, undef
  ; SSSE3: cost of 1 {{.*}} %C = add
  ; SSE42: cost of 1 {{.*}} %C = add
  ; AVX: cost of 1 {{.*}} %C = add
  ; AVX2: cost of 1 {{.*}} %C = add
  %C = add <2 x i64> undef, undef
  ; SSSE3: cost of 2 {{.*}} %D = add
  ; SSE42: cost of 2 {{.*}} %D = add
  ; AVX: cost of 4 {{.*}} %D = add
  ; AVX2: cost of 1 {{.*}} %D = add
  %D = add <4 x i64> undef, undef
  ; SSSE3: cost of 4 {{.*}} %E = add
  ; SSE42: cost of 4 {{.*}} %E = add
  ; AVX: cost of 8 {{.*}} %E = add
  ; AVX2: cost of 2 {{.*}} %E = add
  %E = add <8 x i64> undef, undef
  ret i32 undef
}

; CHECK-LABEL: 'xor'
define i32 @xor(i32 %arg) {
  ; SSSE3: cost of 1 {{.*}} %A = xor
  ; SSE42: cost of 1 {{.*}} %A = xor
  ; AVX: cost of 1 {{.*}} %A = xor
  ; AVX2: cost of 1 {{.*}} %A = xor
  %A = xor <4 x i32> undef, undef
  ; SSSE3: cost of 2 {{.*}} %B = xor
  ; SSE42: cost of 2 {{.*}} %B = xor
  ; AVX: cost of 1 {{.*}} %B = xor
  ; AVX2: cost of 1 {{.*}} %B = xor
  %B = xor <8 x i32> undef, undef
  ; SSSE3: cost of 1 {{.*}} %C = xor
  ; SSE42: cost of 1 {{.*}} %C = xor
  ; AVX: cost of 1 {{.*}} %C = xor
  ; AVX2: cost of 1 {{.*}} %C = xor
  %C = xor <2 x i64> undef, undef
  ; SSSE3: cost of 2 {{.*}} %D = xor
  ; SSE42: cost of 2 {{.*}} %D = xor
  ; AVX: cost of 1 {{.*}} %D = xor
  ; AVX2: cost of 1 {{.*}} %D = xor
  %D = xor <4 x i64> undef, undef
  ret i32 undef
}

; CHECK-LABEL: 'mul'
define void @mul() {
  ; A <2 x i32> gets expanded to a <2 x i64> vector.
  ; A <2 x i64> vector multiply is implemented using
  ; 3 PMULUDQ and 2 PADDS and 4 shifts.
  ; SSSE3: cost of 9 {{.*}} %A0 = mul
  ; SSE42: cost of 9 {{.*}} %A0 = mul
  ; AVX: cost of 9 {{.*}} %A0 = mul
  ; AVX2: cost of 9 {{.*}} %A0 = mul
  %A0 = mul <2 x i32> undef, undef
  ; SSSE3: cost of 6 {{.*}} %A1 = mul
  ; SSE42: cost of 1 {{.*}} %A1 = mul
  ; AVX: cost of 1 {{.*}} %A1 = mul
  ; AVX2: cost of 1 {{.*}} %A1 = mul
  %A1 = mul <4 x i32> undef, undef  
  ; SSSE3: cost of 9 {{.*}} %A2 = mul
  ; SSE42: cost of 9 {{.*}} %A2 = mul
  ; AVX: cost of 9 {{.*}} %A2 = mul
  ; AVX2: cost of 9 {{.*}} %A2 = mul
  %A2 = mul <2 x i64> undef, undef
  ; SSSE3: cost of 18 {{.*}} %A3 = mul
  ; SSE42: cost of 18 {{.*}} %A3 = mul
  ; AVX: cost of 18 {{.*}} %A3 = mul
  ; AVX2: cost of 9 {{.*}} %A3 = mul
  %A3 = mul <4 x i64> undef, undef
  ret void
}

; CHECK-LABEL: 'fmul'
define i32 @fmul(i32 %arg) {
  ; SSSE3: cost of 2 {{.*}} %A = fmul
  ; SSE42: cost of 2 {{.*}} %A = fmul
  ; AVX: cost of 2 {{.*}} %A = fmul
  ; AVX2: cost of 2 {{.*}} %A = fmul
  %A = fmul <4 x float> undef, undef
  ; SSSE3: cost of 4 {{.*}} %B = fmul
  ; SSE42: cost of 4 {{.*}} %B = fmul
  ; AVX: cost of 2 {{.*}} %B = fmul
  ; AVX2: cost of 2 {{.*}} %B = fmul
  %B = fmul <8 x float> undef, undef
  ret i32 undef
}

; CHECK-LABEL: 'shift'
define void @shift() {
  ; SSSE3: cost of 10 {{.*}} %A0 = shl
  ; SSE42: cost of 10 {{.*}} %A0 = shl
  ; AVX: cost of 10 {{.*}} %A0 = shl
  ; AVX2: cost of 1 {{.*}} %A0 = shl
  %A0 = shl <4 x i32> undef, undef
  ; SSSE3: cost of 4 {{.*}} %A1 = shl
  ; SSE42: cost of 4 {{.*}} %A1 = shl
  ; AVX: cost of 4 {{.*}} %A1 = shl
  ; AVX2: cost of 1 {{.*}} %A1 = shl
  %A1 = shl <2 x i64> undef, undef
  ; SSSE3: cost of 20 {{.*}} %A2 = shl
  ; SSE42: cost of 20 {{.*}} %A2 = shl
  ; AVX: cost of 20 {{.*}} %A2 = shl
  ; AVX2: cost of 1 {{.*}} %A2 = shl
  %A2 = shl <8 x i32> undef, undef
  ; SSSE3: cost of 8 {{.*}} %A3 = shl
  ; SSE42: cost of 8 {{.*}} %A3 = shl
  ; AVX: cost of 8 {{.*}} %A3 = shl
  ; AVX2: cost of 1 {{.*}} %A3 = shl
  %A3 = shl <4 x i64> undef, undef

  ; SSSE3: cost of 16 {{.*}} %B0 = lshr
  ; SSE42: cost of 16 {{.*}} %B0 = lshr
  ; AVX: cost of 16 {{.*}} %B0 = lshr
  ; AVX2: cost of 1 {{.*}} %B0 = lshr
  %B0 = lshr <4 x i32> undef, undef
  ; SSSE3: cost of 4 {{.*}} %B1 = lshr
  ; SSE42: cost of 4 {{.*}} %B1 = lshr
  ; AVX: cost of 4 {{.*}} %B1 = lshr
  ; AVX2: cost of 1 {{.*}} %B1 = lshr
  %B1 = lshr <2 x i64> undef, undef
  ; SSSE3: cost of 32 {{.*}} %B2 = lshr
  ; SSE42: cost of 32 {{.*}} %B2 = lshr
  ; AVX: cost of 32 {{.*}} %B2 = lshr
  ; AVX2: cost of 1 {{.*}} %B2 = lshr
  %B2 = lshr <8 x i32> undef, undef
  ; SSSE3: cost of 8 {{.*}} %B3 = lshr
  ; SSE42: cost of 8 {{.*}} %B3 = lshr
  ; AVX: cost of 8 {{.*}} %B3 = lshr
  ; AVX2: cost of 1 {{.*}} %B3 = lshr
  %B3 = lshr <4 x i64> undef, undef

  ; SSSE3: cost of 16 {{.*}} %C0 = ashr
  ; SSE42: cost of 16 {{.*}} %C0 = ashr
  ; AVX: cost of 16 {{.*}} %C0 = ashr
  ; AVX2: cost of 1 {{.*}} %C0 = ashr
  %C0 = ashr <4 x i32> undef, undef
  ; SSSE3: cost of 12 {{.*}} %C1 = ashr
  ; SSE42: cost of 12 {{.*}} %C1 = ashr
  ; AVX: cost of 12 {{.*}} %C1 = ashr
  ; AVX2: cost of 4 {{.*}} %C1 = ashr
  %C1 = ashr <2 x i64> undef, undef
  ; SSSE3: cost of 32 {{.*}} %C2 = ashr
  ; SSE42: cost of 32 {{.*}} %C2 = ashr
  ; AVX: cost of 32 {{.*}} %C2 = ashr
  ; AVX2: cost of 1 {{.*}} %C2 = ashr
  %C2 = ashr <8 x i32> undef, undef
  ; SSSE3: cost of 24 {{.*}} %C3 = ashr
  ; SSE42: cost of 24 {{.*}} %C3 = ashr
  ; AVX: cost of 24 {{.*}} %C3 = ashr
  ; AVX2: cost of 4 {{.*}} %C3 = ashr
  %C3 = ashr <4 x i64> undef, undef

  ret void
}

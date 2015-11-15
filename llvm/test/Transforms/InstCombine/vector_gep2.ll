; RUN: opt < %s -instcombine -S | FileCheck %s

target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

define <2 x i8*> @testa(<2 x i8*> %a) {
; CHECK-LABEL: @testa(
  %g = getelementptr i8, <2 x i8*> %a, <2 x i32> <i32 0, i32 1>
; CHECK: getelementptr i8, <2 x i8*> %a, <2 x i64> <i64 0, i64 1>
  ret <2 x i8*> %g
}

define <8 x double*> @vgep_s_v8i64(double* %a, <8 x i64>%i) {
; CHECK-LABEL: @vgep_s_v8i64
; CHECK: getelementptr double, double* %a, <8 x i64> %i
  %VectorGep = getelementptr double, double* %a, <8 x i64> %i
  ret <8 x double*> %VectorGep
}

define <8 x double*> @vgep_s_v8i32(double* %a, <8 x i32>%i) {
; CHECK-LABEL: @vgep_s_v8i32
; CHECK: %1 = sext <8 x i32> %i to <8 x i64>
; CHECK: getelementptr double, double* %a, <8 x i64> %1
  %VectorGep = getelementptr double, double* %a, <8 x i32> %i
  ret <8 x double*> %VectorGep
}

define <8 x i8*> @vgep_v8iPtr_i32(<8 x i8*> %a, i32 %i) {
; CHECK-LABEL: @vgep_v8iPtr_i32
; CHECK:  %1 = sext i32 %i to i64
; CHECK:  %VectorGep = getelementptr i8, <8 x i8*> %a, i64 %1
  %VectorGep = getelementptr i8, <8 x i8*> %a, i32 %i
  ret <8 x i8*> %VectorGep
}

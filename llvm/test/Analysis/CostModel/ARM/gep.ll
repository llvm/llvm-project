; RUN: opt -cost-model -analyze -mtriple=thumbv7-apple-ios6.0.0 -mcpu=swift < %s | FileCheck %s

target datalayout = "e-p:32:32:32-i1:8:32-i8:8:32-i16:16:32-i32:32:32-i64:32:64-f32:32:32-f64:32:64-v64:32:64-v128:32:128-a0:0:32-n32-S32"
target triple = "thumbv7-apple-ios6.0.0"

define void @test_geps(i32 %i) {
  ; GEPs with index 0 are essentially NOOPs.
;CHECK: cost of 0 for instruction: {{.*}} getelementptr inbounds i8, i8*
  %a0 = getelementptr inbounds i8, i8* undef, i32 0
;CHECK: cost of 0 for instruction: {{.*}} getelementptr inbounds i16, i16*
  %a1 = getelementptr inbounds i16, i16* undef, i32 0
;CHECK: cost of 0 for instruction: {{.*}} getelementptr inbounds i32, i32*
  %a2 = getelementptr inbounds i32, i32* undef, i32 0
;CHECK: cost of 0 for instruction: {{.*}} getelementptr inbounds i64, i64*
  %a3 = getelementptr inbounds i64, i64* undef, i32 0
;CHECK: cost of 0 for instruction: {{.*}} getelementptr inbounds float, float*
  %a4 = getelementptr inbounds float, float* undef, i32 0
;CHECK: cost of 0 for instruction: {{.*}} getelementptr inbounds double, double*
  %a5 = getelementptr inbounds double, double* undef, i32 0
;CHECK: cost of 0 for instruction: {{.*}} getelementptr inbounds <4 x i8>, <4 x i8>*
  %a7 = getelementptr inbounds <4 x i8>, <4 x i8>* undef, i32 0
;CHECK: cost of 0 for instruction: {{.*}} getelementptr inbounds <4 x i16>, <4 x i16>*
  %a8 = getelementptr inbounds <4 x i16>, <4 x i16>* undef, i32 0
;CHECK: cost of 0 for instruction: {{.*}} getelementptr inbounds <4 x i32>, <4 x i32>*
  %a9 = getelementptr inbounds <4 x i32>, <4 x i32>* undef, i32 0
;CHECK: cost of 0 for instruction: {{.*}} getelementptr inbounds <4 x i64>, <4 x i64>*
  %a10 = getelementptr inbounds <4 x i64>, <4 x i64>* undef, i32 0
;CHECK: cost of 0 for instruction: {{.*}} getelementptr inbounds <4 x float>, <4 x float>*
  %a11 = getelementptr inbounds <4 x float>, <4 x float>* undef, i32 0
;CHECK: cost of 0 for instruction: {{.*}} getelementptr inbounds <4 x double>, <4 x double>*
  %a12 = getelementptr inbounds <4 x double>, <4 x double>* undef, i32 0

  ; Cost of GEPs is one if we cannot fold the address computation.
;CHECK: cost of 0 for instruction: {{.*}} getelementptr inbounds i8, i8*
  %b0 = getelementptr inbounds i8, i8* undef, i32 1024
;CHECK: cost of 0 for instruction: {{.*}} getelementptr inbounds i16, i16*
  %b1 = getelementptr inbounds i16, i16* undef, i32 1024
  ; Thumb-2 cannot fold offset >= 2^12 into address computation.
;CHECK: cost of 1 for instruction: {{.*}} getelementptr inbounds i32, i32*
  %b2 = getelementptr inbounds i32, i32* undef, i32 1024
;CHECK: cost of 1 for instruction: {{.*}} getelementptr inbounds i64, i64*
  %b3 = getelementptr inbounds i64, i64* undef, i32 1024
;CHECK: cost of 1 for instruction: {{.*}} getelementptr inbounds float, float*
  %b4 = getelementptr inbounds float, float* undef, i32 1024
;CHECK: cost of 1 for instruction: {{.*}} getelementptr inbounds double, double*
  %b5 = getelementptr inbounds double, double* undef, i32 1024
;CHECK: cost of 1 for instruction: {{.*}} getelementptr inbounds <4 x i8>, <4 x i8>*
  %b7 = getelementptr inbounds <4 x i8>, <4 x i8>* undef, i32 1
;CHECK: cost of 1 for instruction: {{.*}} getelementptr inbounds <4 x i16>, <4 x i16>*
  %b8 = getelementptr inbounds <4 x i16>, <4 x i16>* undef, i32 1
;CHECK: cost of 1 for instruction: {{.*}} getelementptr inbounds <4 x i32>, <4 x i32>*
  %b9 = getelementptr inbounds <4 x i32>, <4 x i32>* undef, i32 1
;CHECK: cost of 1 for instruction: {{.*}} getelementptr inbounds <4 x i64>, <4 x i64>*
  %b10 = getelementptr inbounds <4 x i64>, <4 x i64>* undef, i32 1
;CHECK: cost of 1 for instruction: {{.*}} getelementptr inbounds <4 x float>, <4 x float>*
  %b11 = getelementptr inbounds <4 x float>, <4 x float>* undef, i32 1
;CHECK: cost of 1 for instruction: {{.*}} getelementptr inbounds <4 x double>, <4 x double>*
  %b12 = getelementptr inbounds <4 x double>, <4 x double>* undef, i32 1

;CHECK: cost of 0 for instruction: {{.*}} getelementptr inbounds i8, i8*
  %c0 = getelementptr inbounds i8, i8* undef, i32 %i
;CHECK: cost of 0 for instruction: {{.*}} getelementptr inbounds i16, i16*
  %c1 = getelementptr inbounds i16, i16* undef, i32 %i
;CHECK: cost of 0 for instruction: {{.*}} getelementptr inbounds i32, i32*
  %c2 = getelementptr inbounds i32, i32* undef, i32 %i
;CHECK: cost of 1 for instruction: {{.*}} getelementptr inbounds i64, i64*
  %c3 = getelementptr inbounds i64, i64* undef, i32 %i
;CHECK: cost of 1 for instruction: {{.*}} getelementptr inbounds float, float*
  %c4 = getelementptr inbounds float, float* undef, i32 %i
;CHECK: cost of 1 for instruction: {{.*}} getelementptr inbounds double, double*
  %c5 = getelementptr inbounds double, double* undef, i32 %i
;CHECK: cost of 1 for instruction: {{.*}} getelementptr inbounds <4 x i8>, <4 x i8>*
  %c7 = getelementptr inbounds <4 x i8>, <4 x i8>* undef, i32 %i
;CHECK: cost of 1 for instruction: {{.*}} getelementptr inbounds <4 x i16>, <4 x i16>*
  %c8 = getelementptr inbounds <4 x i16>, <4 x i16>* undef, i32 %i
  ; Thumb-2 cannot fold scales larger than 8 to address computation.
;CHECK: cost of 1 for instruction: {{.*}} getelementptr inbounds <4 x i32>, <4 x i32>*
  %c9 = getelementptr inbounds <4 x i32>, <4 x i32>* undef, i32 %i
;CHECK: cost of 1 for instruction: {{.*}} getelementptr inbounds <4 x i64>, <4 x i64>*
  %c10 = getelementptr inbounds <4 x i64>, <4 x i64>* undef, i32 %i
;CHECK: cost of 1 for instruction: {{.*}} getelementptr inbounds <4 x float>, <4 x float>*
  %c11 = getelementptr inbounds <4 x float>, <4 x float>* undef, i32 %i
;CHECK: cost of 1 for instruction: {{.*}} getelementptr inbounds <4 x double>, <4 x double>*
  %c12 = getelementptr inbounds <4 x double>, <4 x double>* undef, i32 %i

;CHECK: cost of 0 for instruction: {{.*}} getelementptr inbounds i8, i8*
  %d0 = getelementptr inbounds i8, i8* undef, i32 -1

  ret void
}

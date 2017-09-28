; RUN: opt < %s  -cost-model -analyze -mtriple=x86_64-apple-macosx10.8.0 -mcpu=corei7-avx | FileCheck %s

target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128-n8:16:32:64-S128"
target triple = "x86_64-apple-macosx10.8.0"


define void @test_geps() {
  ; Cost of should be zero. We expect it to be folded into
  ; the instruction addressing mode.
;CHECK:  cost of 0 for instruction: {{.*}} getelementptr inbounds i8, i8*
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

 ; Vector geps should also have zero cost.
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

  ; Check that we handle outlandishly large GEPs properly.  This is unlikely to
  ; be a valid pointer, but LLVM still generates GEPs like this sometimes in
  ; dead code.
  ;
  ; This GEP has index INT64_MAX, which is cost 1.
;CHECK: cost of 1 for instruction: {{.*}} getelementptr inbounds i8, i8*
  %giant_gep0 = getelementptr inbounds i8, i8* undef, i64 9223372036854775807

  ; This GEP index wraps around to -1, which is cost 0.
;CHECK: cost of 0 for instruction: {{.*}} getelementptr inbounds i8, i8*
  %giant_gep1 = getelementptr inbounds i8, i8* undef, i128 295147905179352825855

  ret void
}

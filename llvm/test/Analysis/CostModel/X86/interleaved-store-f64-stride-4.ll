; RUN: opt -loop-vectorize -vectorizer-maximize-bandwidth -S -mattr=+sse2 --debug-only=loop-vectorize < %s 2>&1 | FileCheck %s --check-prefixes=CHECK,SSE2
; RUN: opt -loop-vectorize -vectorizer-maximize-bandwidth -S -mattr=+avx  --debug-only=loop-vectorize < %s 2>&1 | FileCheck %s --check-prefixes=CHECK,AVX1
; RUN: opt -loop-vectorize -vectorizer-maximize-bandwidth -S -mattr=+avx2 --debug-only=loop-vectorize < %s 2>&1 | FileCheck %s --check-prefixes=CHECK,AVX2
; RUN: opt -loop-vectorize -vectorizer-maximize-bandwidth -S -mattr=+avx512bw,+avx512vl --debug-only=loop-vectorize < %s 2>&1 | FileCheck %s --check-prefixes=CHECK,AVX512
; REQUIRES: asserts

target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

@A = global [1024 x i8] zeroinitializer, align 128
@B = global [1024 x double] zeroinitializer, align 128

; CHECK: LV: Checking a loop in "test"
;
; SSE2: LV: Found an estimated cost of 1 for VF 1 For instruction:   store double %v3, double* %out3, align 8
; SSE2: LV: Found an estimated cost of 12 for VF 2 For instruction:   store double %v3, double* %out3, align 8
; SSE2: LV: Found an estimated cost of 24 for VF 4 For instruction:   store double %v3, double* %out3, align 8
; SSE2: LV: Found an estimated cost of 48 for VF 8 For instruction:   store double %v3, double* %out3, align 8
;
; AVX1: LV: Found an estimated cost of 1 for VF 1 For instruction:   store double %v3, double* %out3, align 8
; AVX1: LV: Found an estimated cost of 12 for VF 2 For instruction:   store double %v3, double* %out3, align 8
; AVX1: LV: Found an estimated cost of 32 for VF 4 For instruction:   store double %v3, double* %out3, align 8
; AVX1: LV: Found an estimated cost of 64 for VF 8 For instruction:   store double %v3, double* %out3, align 8
; AVX1: LV: Found an estimated cost of 128 for VF 16 For instruction:   store double %v3, double* %out3, align 8
;
; AVX2: LV: Found an estimated cost of 1 for VF 1 For instruction:   store double %v3, double* %out3, align 8
; AVX2: LV: Found an estimated cost of 8 for VF 2 For instruction:   store double %v3, double* %out3, align 8
; AVX2: LV: Found an estimated cost of 12 for VF 4 For instruction:   store double %v3, double* %out3, align 8
; AVX2: LV: Found an estimated cost of 28 for VF 8 For instruction:   store double %v3, double* %out3, align 8
; AVX2: LV: Found an estimated cost of 56 for VF 16 For instruction:   store double %v3, double* %out3, align 8
;
; AVX512: LV: Found an estimated cost of 1 for VF 1 For instruction:   store double %v3, double* %out3, align 8
; AVX512: LV: Found an estimated cost of 5 for VF 2 For instruction:   store double %v3, double* %out3, align 8
; AVX512: LV: Found an estimated cost of 11 for VF 4 For instruction:   store double %v3, double* %out3, align 8
; AVX512: LV: Found an estimated cost of 22 for VF 8 For instruction:   store double %v3, double* %out3, align 8
; AVX512: LV: Found an estimated cost of 44 for VF 16 For instruction:   store double %v3, double* %out3, align 8
; AVX512: LV: Found an estimated cost of 88 for VF 32 For instruction:   store double %v3, double* %out3, align 8
;
; CHECK-NOT: LV: Found an estimated cost of {{[0-9]+}} for VF {{[0-9]+}} For instruction:   store double %v3, double* %out3, align 8

define void @test() {
entry:
  br label %for.body

for.body:
  %iv = phi i64 [ 0, %entry ], [ %iv.next, %for.body ]

  %iv.0 = add nuw nsw i64 %iv, 0
  %iv.1 = add nuw nsw i64 %iv, 1
  %iv.2 = add nuw nsw i64 %iv, 2
  %iv.3 = add nuw nsw i64 %iv, 3

  %in = getelementptr inbounds [1024 x i8], [1024 x i8]* @A, i64 0, i64 %iv.0
  %v.narrow = load i8, i8* %in

  %v = uitofp i8 %v.narrow to double

  %v0 = fadd double %v, 0.0
  %v1 = fadd double %v, 1.0
  %v2 = fadd double %v, 2.0
  %v3 = fadd double %v, 3.0

  %out0 = getelementptr inbounds [1024 x double], [1024 x double]* @B, i64 0, i64 %iv.0
  %out1 = getelementptr inbounds [1024 x double], [1024 x double]* @B, i64 0, i64 %iv.1
  %out2 = getelementptr inbounds [1024 x double], [1024 x double]* @B, i64 0, i64 %iv.2
  %out3 = getelementptr inbounds [1024 x double], [1024 x double]* @B, i64 0, i64 %iv.3

  store double %v0, double* %out0
  store double %v1, double* %out1
  store double %v2, double* %out2
  store double %v3, double* %out3

  %iv.next = add nuw nsw i64 %iv.0, 4
  %cmp = icmp ult i64 %iv.next, 1024
  br i1 %cmp, label %for.body, label %for.cond.cleanup

for.cond.cleanup:
  ret void
}

; RUN: opt -loop-vectorize -vectorizer-maximize-bandwidth -S -mattr=+sse2 --debug-only=loop-vectorize < %s 2>&1 | FileCheck %s --check-prefixes=CHECK,SSE2
; RUN: opt -loop-vectorize -vectorizer-maximize-bandwidth -S -mattr=+avx  --debug-only=loop-vectorize < %s 2>&1 | FileCheck %s --check-prefixes=CHECK,AVX1
; RUN: opt -loop-vectorize -vectorizer-maximize-bandwidth -S -mattr=+avx2 --debug-only=loop-vectorize < %s 2>&1 | FileCheck %s --check-prefixes=CHECK,AVX2
; RUN: opt -loop-vectorize -vectorizer-maximize-bandwidth -S -mattr=+avx512bw,+avx512vl --debug-only=loop-vectorize < %s 2>&1 | FileCheck %s --check-prefixes=CHECK,AVX512
; REQUIRES: asserts

target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

@A = global [1024 x i8] zeroinitializer, align 128
@B = global [1024 x i16] zeroinitializer, align 128

; CHECK: LV: Checking a loop in "test"
;
; SSE2: LV: Found an estimated cost of 1 for VF 1 For instruction:   store i16 %v5, i16* %out5, align 2
; SSE2: LV: Found an estimated cost of 26 for VF 2 For instruction:   store i16 %v5, i16* %out5, align 2
; SSE2: LV: Found an estimated cost of 51 for VF 4 For instruction:   store i16 %v5, i16* %out5, align 2
; SSE2: LV: Found an estimated cost of 102 for VF 8 For instruction:   store i16 %v5, i16* %out5, align 2
; SSE2: LV: Found an estimated cost of 204 for VF 16 For instruction:   store i16 %v5, i16* %out5, align 2
;
; AVX1: LV: Found an estimated cost of 1 for VF 1 For instruction:   store i16 %v5, i16* %out5, align 2
; AVX1: LV: Found an estimated cost of 30 for VF 2 For instruction:   store i16 %v5, i16* %out5, align 2
; AVX1: LV: Found an estimated cost of 53 for VF 4 For instruction:   store i16 %v5, i16* %out5, align 2
; AVX1: LV: Found an estimated cost of 105 for VF 8 For instruction:   store i16 %v5, i16* %out5, align 2
; AVX1: LV: Found an estimated cost of 258 for VF 16 For instruction:   store i16 %v5, i16* %out5, align 2
; AVX1: LV: Found an estimated cost of 516 for VF 32 For instruction:   store i16 %v5, i16* %out5, align 2
;
; AVX2: LV: Found an estimated cost of 1 for VF 1 For instruction:   store i16 %v5, i16* %out5, align 2
; AVX2: LV: Found an estimated cost of 13 for VF 2 For instruction:   store i16 %v5, i16* %out5, align 2
; AVX2: LV: Found an estimated cost of 17 for VF 4 For instruction:   store i16 %v5, i16* %out5, align 2
; AVX2: LV: Found an estimated cost of 24 for VF 8 For instruction:   store i16 %v5, i16* %out5, align 2
; AVX2: LV: Found an estimated cost of 64 for VF 16 For instruction:   store i16 %v5, i16* %out5, align 2
; AVX2: LV: Found an estimated cost of 102 for VF 32 For instruction:   store i16 %v5, i16* %out5, align 2
;
; AVX512: LV: Found an estimated cost of 1 for VF 1 For instruction:   store i16 %v5, i16* %out5, align 2
; AVX512: LV: Found an estimated cost of 13 for VF 2 For instruction:   store i16 %v5, i16* %out5, align 2
; AVX512: LV: Found an estimated cost of 13 for VF 4 For instruction:   store i16 %v5, i16* %out5, align 2
; AVX512: LV: Found an estimated cost of 27 for VF 8 For instruction:   store i16 %v5, i16* %out5, align 2
; AVX512: LV: Found an estimated cost of 40 for VF 16 For instruction:   store i16 %v5, i16* %out5, align 2
; AVX512: LV: Found an estimated cost of 81 for VF 32 For instruction:   store i16 %v5, i16* %out5, align 2
; AVX512: LV: Found an estimated cost of 162 for VF 64 For instruction:   store i16 %v5, i16* %out5, align 2
;
; CHECK-NOT: LV: Found an estimated cost of {{[0-9]+}} for VF {{[0-9]+}} For instruction:   store i16 %v5, i16* %out5, align 2

define void @test() {
entry:
  br label %for.body

for.body:
  %iv = phi i64 [ 0, %entry ], [ %iv.next, %for.body ]

  %iv.0 = add nuw nsw i64 %iv, 0
  %iv.1 = add nuw nsw i64 %iv, 1
  %iv.2 = add nuw nsw i64 %iv, 2
  %iv.3 = add nuw nsw i64 %iv, 3
  %iv.4 = add nuw nsw i64 %iv, 4
  %iv.5 = add nuw nsw i64 %iv, 5

  %in = getelementptr inbounds [1024 x i8], [1024 x i8]* @A, i64 0, i64 %iv.0
  %v.narrow = load i8, i8* %in

  %v = zext i8 %v.narrow to i16

  %v0 = add i16 %v, 0
  %v1 = add i16 %v, 1
  %v2 = add i16 %v, 2
  %v3 = add i16 %v, 3
  %v4 = add i16 %v, 4
  %v5 = add i16 %v, 5

  %out0 = getelementptr inbounds [1024 x i16], [1024 x i16]* @B, i64 0, i64 %iv.0
  %out1 = getelementptr inbounds [1024 x i16], [1024 x i16]* @B, i64 0, i64 %iv.1
  %out2 = getelementptr inbounds [1024 x i16], [1024 x i16]* @B, i64 0, i64 %iv.2
  %out3 = getelementptr inbounds [1024 x i16], [1024 x i16]* @B, i64 0, i64 %iv.3
  %out4 = getelementptr inbounds [1024 x i16], [1024 x i16]* @B, i64 0, i64 %iv.4
  %out5 = getelementptr inbounds [1024 x i16], [1024 x i16]* @B, i64 0, i64 %iv.5

  store i16 %v0, i16* %out0
  store i16 %v1, i16* %out1
  store i16 %v2, i16* %out2
  store i16 %v3, i16* %out3
  store i16 %v4, i16* %out4
  store i16 %v5, i16* %out5

  %iv.next = add nuw nsw i64 %iv.0, 6
  %cmp = icmp ult i64 %iv.next, 1024
  br i1 %cmp, label %for.body, label %for.cond.cleanup

for.cond.cleanup:
  ret void
}

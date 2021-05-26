; RUN: opt -loop-vectorize -vectorizer-maximize-bandwidth -S -mattr=+avx2 --debug-only=loop-vectorize < %s 2>&1 | FileCheck %s
; REQUIRES: asserts

target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

@A = global [1024 x i8] zeroinitializer, align 128
@B = global [1024 x i16] zeroinitializer, align 128

; CHECK: LV: Checking a loop in "test"
; CHECK: LV: Found an estimated cost of 1 for VF 1 For instruction:   store i16 %v1, i16* %out1, align 2
; CHECK: LV: Found an estimated cost of 9 for VF 2 For instruction:   store i16 %v1, i16* %out1, align 2
; CHECK: LV: Found an estimated cost of 17 for VF 4 For instruction:   store i16 %v1, i16* %out1, align 2
; CHECK: LV: Found an estimated cost of 49 for VF 8 For instruction:   store i16 %v1, i16* %out1, align 2
; CHECK: LV: Found an estimated cost of 114 for VF 16 For instruction:   store i16 %v1, i16* %out1, align 2
; CHECK: LV: Found an estimated cost of 228 for VF 32 For instruction:   store i16 %v1, i16* %out1, align 2
; CHECK-NOT: LV: Found an estimated cost of {{[0-9]+}} for VF {{[0-9]+}} For instruction:   store i16 %v1, i16* %out1, align 2

define void @test() {
entry:
  br label %for.body

for.body:
  %iv = phi i64 [ 0, %entry ], [ %iv.next, %for.body ]

  %iv.0 = add nuw nsw i64 %iv, 0
  %iv.1 = add nuw nsw i64 %iv, 1

  %in = getelementptr inbounds [1024 x i8], [1024 x i8]* @A, i64 0, i64 %iv.0
  %v.narrow = load i8, i8* %in

  %v = zext i8 %v.narrow to i16

  %v0 = add i16 %v, 0
  %v1 = add i16 %v, 1

  %out0 = getelementptr inbounds [1024 x i16], [1024 x i16]* @B, i64 0, i64 %iv.0
  %out1 = getelementptr inbounds [1024 x i16], [1024 x i16]* @B, i64 0, i64 %iv.1

  store i16 %v0, i16* %out0
  store i16 %v1, i16* %out1

  %iv.next = add nuw nsw i64 %iv.0, 2
  %cmp = icmp ult i64 %iv.next, 1024
  br i1 %cmp, label %for.body, label %for.cond.cleanup

for.cond.cleanup:
  ret void
}

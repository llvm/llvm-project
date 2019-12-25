; RUN: opt -S %loadPolly -basicaa -polly-opt-isl \
; RUN: -polly-pattern-matching-based-opts=false -polly-vectorizer=polly \
; RUN: -polly-ast -analyze < %s | FileCheck %s 
; RUN: opt -S %loadPolly -basicaa -polly-opt-isl \
; RUN: -polly-pattern-matching-based-opts=false -polly-vectorizer=stripmine \
; RUN: -polly-ast -analyze < %s | FileCheck %s

; RUN: opt -S %loadPolly -basicaa -polly-opt-isl \
; RUN: -polly-vectorizer=polly -polly-pattern-matching-based-opts=false \
; RUN: -polly-ast -analyze -polly-prevect-width=16 < %s | \
; RUN: FileCheck %s -check-prefix=VEC16

target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"

@C = common global [1536 x [1536 x float]] zeroinitializer, align 16
@A = common global [1536 x [1536 x float]] zeroinitializer, align 16
@B = common global [1536 x [1536 x float]] zeroinitializer, align 16

; Function Attrs: nounwind uwtable
define void @foo() #0 {
entry:
  br label %entry.split

entry.split:                                      ; preds = %entry
  br label %for.cond1.preheader

for.cond1.preheader:                              ; preds = %entry.split, %for.inc28
  %indvar4 = phi i64 [ 0, %entry.split ], [ %indvar.next5, %for.inc28 ]
  br label %for.body3

for.body3:                                        ; preds = %for.cond1.preheader, %for.inc25
  %indvar6 = phi i64 [ 0, %for.cond1.preheader ], [ %indvar.next7, %for.inc25 ]
  %arrayidx24 = getelementptr [1536 x [1536 x float]], [1536 x [1536 x float]]* @C, i64 0, i64 %indvar4, i64 %indvar6
  store float 0.000000e+00, float* %arrayidx24, align 4
  br label %for.body8

for.body8:                                        ; preds = %for.body3, %for.body8
  %indvar = phi i64 [ 0, %for.body3 ], [ %indvar.next, %for.body8 ]
  %arrayidx16 = getelementptr [1536 x [1536 x float]], [1536 x [1536 x float]]* @A, i64 0, i64 %indvar4, i64 %indvar
  %arrayidx20 = getelementptr [1536 x [1536 x float]], [1536 x [1536 x float]]* @B, i64 0, i64 %indvar, i64 %indvar6
  %0 = load float, float* %arrayidx24, align 4
  %1 = load float, float* %arrayidx16, align 4
  %2 = load float, float* %arrayidx20, align 4
  %mul = fmul float %1, %2
  %add = fadd float %0, %mul
  store float %add, float* %arrayidx24, align 4
  %indvar.next = add i64 %indvar, 1
  %exitcond = icmp ne i64 %indvar.next, 1536
  br i1 %exitcond, label %for.body8, label %for.inc25

for.inc25:                                        ; preds = %for.body8
  %indvar.next7 = add i64 %indvar6, 1
  %exitcond8 = icmp ne i64 %indvar.next7, 1536
  br i1 %exitcond8, label %for.body3, label %for.inc28

for.inc28:                                        ; preds = %for.inc25
  %indvar.next5 = add i64 %indvar4, 1
  %exitcond9 = icmp ne i64 %indvar.next5, 1536
  br i1 %exitcond9, label %for.cond1.preheader, label %for.end30

for.end30:                                        ; preds = %for.inc28
  ret void
}

attributes #0 = { nounwind uwtable "less-precise-fpmad"="false" "frame-pointer"="all" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "stack-protector-buffer-size"="8" "unsafe-fp-math"="false" "use-soft-float"="false" }

; CHECK: #pragma known-parallel
; CHECK: for (int c0 = 0; c0 <= 47; c0 += 1)
; CHECK:   for (int c1 = 0; c1 <= 47; c1 += 1)
; CHECK:     for (int c2 = 0; c2 <= 31; c2 += 1)
; CHECK:       for (int c3 = 0; c3 <= 7; c3 += 1)
; CHECK:         // SIMD
; CHECK:         for (int c4 = 0; c4 <= 3; c4 += 1)
; CHECK:           Stmt_for_body3(32 * c0 + c2, 32 * c1 + 4 * c3 + c4);
; CHECK: #pragma known-parallel
; CHECK: for (int c0 = 0; c0 <= 47; c0 += 1)
; CHECK:   for (int c1 = 0; c1 <= 47; c1 += 1)
; CHECK:     for (int c2 = 0; c2 <= 47; c2 += 1)
; CHECK:       for (int c3 = 0; c3 <= 31; c3 += 1)
; CHECK:         for (int c4 = 0; c4 <= 7; c4 += 1)
; CHECK:           for (int c5 = 0; c5 <= 31; c5 += 1)
; CHECK:             // SIMD
; CHECK:             for (int c6 = 0; c6 <= 3; c6 += 1)
; CHECK:               Stmt_for_body8(32 * c0 + c3, 32 * c1 + 4 * c4 + c6, 32 * c2 + c5);

; VEC16: {
; VEC16:   #pragma known-parallel
; VEC16:   for (int c0 = 0; c0 <= 47; c0 += 1)
; VEC16:     for (int c1 = 0; c1 <= 47; c1 += 1)
; VEC16:       for (int c2 = 0; c2 <= 31; c2 += 1)
; VEC16:         for (int c3 = 0; c3 <= 1; c3 += 1)
; VEC16:           // SIMD
; VEC16:           for (int c4 = 0; c4 <= 15; c4 += 1)
; VEC16:             Stmt_for_body3(32 * c0 + c2, 32 * c1 + 16 * c3 + c4);
; VEC16:   #pragma known-parallel
; VEC16:   for (int c0 = 0; c0 <= 47; c0 += 1)
; VEC16:     for (int c1 = 0; c1 <= 47; c1 += 1)
; VEC16:       for (int c2 = 0; c2 <= 47; c2 += 1)
; VEC16:         for (int c3 = 0; c3 <= 31; c3 += 1)
; VEC16:           for (int c4 = 0; c4 <= 1; c4 += 1)
; VEC16:             for (int c5 = 0; c5 <= 31; c5 += 1)
; VEC16:               // SIMD
; VEC16:               for (int c6 = 0; c6 <= 15; c6 += 1)
; VEC16:                 Stmt_for_body8(32 * c0 + c3, 32 * c1 + 16 * c4 + c6, 32 * c2 + c5);
; VEC16: }


!llvm.ident = !{!0}

!0 = !{!"clang version 3.5.0 "}

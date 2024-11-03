; RUN: opt -S %loadPolly -basic-aa -polly-tiling=false -polly-pattern-matching-based-opts=false -polly-vectorizer=polly -polly-opt-isl -polly-print-ast -disable-output < %s | FileCheck %s
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
  %arrayidx24 = getelementptr [1536 x [1536 x float]], ptr @C, i64 0, i64 %indvar4, i64 %indvar6
  store float 0.000000e+00, ptr %arrayidx24, align 4
  br label %for.body8

for.body8:                                        ; preds = %for.body3, %for.body8
  %indvar = phi i64 [ 0, %for.body3 ], [ %indvar.next, %for.body8 ]
  %arrayidx16 = getelementptr [1536 x [1536 x float]], ptr @A, i64 0, i64 %indvar4, i64 %indvar
  %arrayidx20 = getelementptr [1536 x [1536 x float]], ptr @B, i64 0, i64 %indvar, i64 %indvar6
  %0 = load float, ptr %arrayidx24, align 4
  %1 = load float, ptr %arrayidx16, align 4
  %2 = load float, ptr %arrayidx20, align 4
  %mul = fmul float %1, %2
  %add = fadd float %0, %mul
  store float %add, ptr %arrayidx24, align 4
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
; CHECK: for (int c0 = 0; c0 <= 1535; c0 += 1)
; CHECK:   for (int c1 = 0; c1 <= 383; c1 += 1)
; CHECK:       // SIMD
; CHECK:     for (int c2 = 0; c2 <= 3; c2 += 1)
; CHECK:       Stmt_for_body3(c0, 4 * c1 + c2);
; CHECK: #pragma known-parallel
; CHECK: for (int c0 = 0; c0 <= 1535; c0 += 1)
; CHECK:   for (int c1 = 0; c1 <= 383; c1 += 1)
; CHECK:     for (int c2 = 0; c2 <= 1535; c2 += 1)
; CHECK:       // SIMD
; CHECK:       for (int c3 = 0; c3 <= 3; c3 += 1)
; CHECK:         Stmt_for_body8(c0, 4 * c1 + c3, c2);

!llvm.ident = !{!0}

!0 = !{!"clang version 3.5.0 "}

; RUN: opt < %s -basicaa -slp-vectorizer -S | FileCheck %s
target datalayout = "e-m:o-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-apple-macosx10.9.0"

@A = common global [2000 x double] zeroinitializer, align 16
@B = common global [2000 x double] zeroinitializer, align 16
@C = common global [2000 x float] zeroinitializer, align 16
@D = common global [2000 x float] zeroinitializer, align 16

; Currently SCEV isn't smart enough to figure out that accesses
; A[3*i], A[3*i+1] and A[3*i+2] are consecutive, but in future
; that would hopefully be fixed. For now, check that this isn't
; vectorized.
; CHECK-LABEL: foo_3double
; CHECK-NOT: x double>
; Function Attrs: nounwind ssp uwtable
define void @foo_3double(i32 %u) #0 {
entry:
  %u.addr = alloca i32, align 4
  store i32 %u, i32* %u.addr, align 4
  %mul = mul nsw i32 %u, 3
  %idxprom = sext i32 %mul to i64
  %arrayidx = getelementptr inbounds [2000 x double], [2000 x double]* @A, i32 0, i64 %idxprom
  %0 = load double, double* %arrayidx, align 8
  %arrayidx4 = getelementptr inbounds [2000 x double], [2000 x double]* @B, i32 0, i64 %idxprom
  %1 = load double, double* %arrayidx4, align 8
  %add5 = fadd double %0, %1
  store double %add5, double* %arrayidx, align 8
  %add11 = add nsw i32 %mul, 1
  %idxprom12 = sext i32 %add11 to i64
  %arrayidx13 = getelementptr inbounds [2000 x double], [2000 x double]* @A, i32 0, i64 %idxprom12
  %2 = load double, double* %arrayidx13, align 8
  %arrayidx17 = getelementptr inbounds [2000 x double], [2000 x double]* @B, i32 0, i64 %idxprom12
  %3 = load double, double* %arrayidx17, align 8
  %add18 = fadd double %2, %3
  store double %add18, double* %arrayidx13, align 8
  %add24 = add nsw i32 %mul, 2
  %idxprom25 = sext i32 %add24 to i64
  %arrayidx26 = getelementptr inbounds [2000 x double], [2000 x double]* @A, i32 0, i64 %idxprom25
  %4 = load double, double* %arrayidx26, align 8
  %arrayidx30 = getelementptr inbounds [2000 x double], [2000 x double]* @B, i32 0, i64 %idxprom25
  %5 = load double, double* %arrayidx30, align 8
  %add31 = fadd double %4, %5
  store double %add31, double* %arrayidx26, align 8
  ret void
}

; SCEV should be able to tell that accesses A[C1 + C2*i], A[C1 + C2*i], ...
; A[C1 + C2*i] are consecutive, if C2 is a power of 2, and C2 > C1 > 0.
; Thus, the following code should be vectorized.
; CHECK-LABEL: foo_2double
; CHECK: x double>
; Function Attrs: nounwind ssp uwtable
define void @foo_2double(i32 %u) #0 {
entry:
  %u.addr = alloca i32, align 4
  store i32 %u, i32* %u.addr, align 4
  %mul = mul nsw i32 %u, 2
  %idxprom = sext i32 %mul to i64
  %arrayidx = getelementptr inbounds [2000 x double], [2000 x double]* @A, i32 0, i64 %idxprom
  %0 = load double, double* %arrayidx, align 8
  %arrayidx4 = getelementptr inbounds [2000 x double], [2000 x double]* @B, i32 0, i64 %idxprom
  %1 = load double, double* %arrayidx4, align 8
  %add5 = fadd double %0, %1
  store double %add5, double* %arrayidx, align 8
  %add11 = add nsw i32 %mul, 1
  %idxprom12 = sext i32 %add11 to i64
  %arrayidx13 = getelementptr inbounds [2000 x double], [2000 x double]* @A, i32 0, i64 %idxprom12
  %2 = load double, double* %arrayidx13, align 8
  %arrayidx17 = getelementptr inbounds [2000 x double], [2000 x double]* @B, i32 0, i64 %idxprom12
  %3 = load double, double* %arrayidx17, align 8
  %add18 = fadd double %2, %3
  store double %add18, double* %arrayidx13, align 8
  ret void
}

; Similar to the previous test, but with different datatype.
; CHECK-LABEL: foo_4float
; CHECK: x float>
; Function Attrs: nounwind ssp uwtable
define void @foo_4float(i32 %u) #0 {
entry:
  %u.addr = alloca i32, align 4
  store i32 %u, i32* %u.addr, align 4
  %mul = mul nsw i32 %u, 4
  %idxprom = sext i32 %mul to i64
  %arrayidx = getelementptr inbounds [2000 x float], [2000 x float]* @C, i32 0, i64 %idxprom
  %0 = load float, float* %arrayidx, align 4
  %arrayidx4 = getelementptr inbounds [2000 x float], [2000 x float]* @D, i32 0, i64 %idxprom
  %1 = load float, float* %arrayidx4, align 4
  %add5 = fadd float %0, %1
  store float %add5, float* %arrayidx, align 4
  %add11 = add nsw i32 %mul, 1
  %idxprom12 = sext i32 %add11 to i64
  %arrayidx13 = getelementptr inbounds [2000 x float], [2000 x float]* @C, i32 0, i64 %idxprom12
  %2 = load float, float* %arrayidx13, align 4
  %arrayidx17 = getelementptr inbounds [2000 x float], [2000 x float]* @D, i32 0, i64 %idxprom12
  %3 = load float, float* %arrayidx17, align 4
  %add18 = fadd float %2, %3
  store float %add18, float* %arrayidx13, align 4
  %add24 = add nsw i32 %mul, 2
  %idxprom25 = sext i32 %add24 to i64
  %arrayidx26 = getelementptr inbounds [2000 x float], [2000 x float]* @C, i32 0, i64 %idxprom25
  %4 = load float, float* %arrayidx26, align 4
  %arrayidx30 = getelementptr inbounds [2000 x float], [2000 x float]* @D, i32 0, i64 %idxprom25
  %5 = load float, float* %arrayidx30, align 4
  %add31 = fadd float %4, %5
  store float %add31, float* %arrayidx26, align 4
  %add37 = add nsw i32 %mul, 3
  %idxprom38 = sext i32 %add37 to i64
  %arrayidx39 = getelementptr inbounds [2000 x float], [2000 x float]* @C, i32 0, i64 %idxprom38
  %6 = load float, float* %arrayidx39, align 4
  %arrayidx43 = getelementptr inbounds [2000 x float], [2000 x float]* @D, i32 0, i64 %idxprom38
  %7 = load float, float* %arrayidx43, align 4
  %add44 = fadd float %6, %7
  store float %add44, float* %arrayidx39, align 4
  ret void
}

; Similar to the previous tests, but now we are dealing with AddRec SCEV.
; CHECK-LABEL: foo_loop
; CHECK: x double>
; Function Attrs: nounwind ssp uwtable
define i32 @foo_loop(double* %A, i32 %n) #0 {
entry:
  %A.addr = alloca double*, align 8
  %n.addr = alloca i32, align 4
  %sum = alloca double, align 8
  %i = alloca i32, align 4
  store double* %A, double** %A.addr, align 8
  store i32 %n, i32* %n.addr, align 4
  store double 0.000000e+00, double* %sum, align 8
  store i32 0, i32* %i, align 4
  %cmp1 = icmp slt i32 0, %n
  br i1 %cmp1, label %for.body.lr.ph, label %for.end

for.body.lr.ph:                                   ; preds = %entry
  br label %for.body

for.body:                                         ; preds = %for.body.lr.ph, %for.body
  %0 = phi i32 [ 0, %for.body.lr.ph ], [ %inc, %for.body ]
  %1 = phi double [ 0.000000e+00, %for.body.lr.ph ], [ %add7, %for.body ]
  %mul = mul nsw i32 %0, 2
  %idxprom = sext i32 %mul to i64
  %arrayidx = getelementptr inbounds double, double* %A, i64 %idxprom
  %2 = load double, double* %arrayidx, align 8
  %mul1 = fmul double 7.000000e+00, %2
  %add = add nsw i32 %mul, 1
  %idxprom3 = sext i32 %add to i64
  %arrayidx4 = getelementptr inbounds double, double* %A, i64 %idxprom3
  %3 = load double, double* %arrayidx4, align 8
  %mul5 = fmul double 7.000000e+00, %3
  %add6 = fadd double %mul1, %mul5
  %add7 = fadd double %1, %add6
  store double %add7, double* %sum, align 8
  %inc = add nsw i32 %0, 1
  store i32 %inc, i32* %i, align 4
  %cmp = icmp slt i32 %inc, %n
  br i1 %cmp, label %for.body, label %for.cond.for.end_crit_edge

for.cond.for.end_crit_edge:                       ; preds = %for.body
  %split = phi double [ %add7, %for.body ]
  br label %for.end

for.end:                                          ; preds = %for.cond.for.end_crit_edge, %entry
  %.lcssa = phi double [ %split, %for.cond.for.end_crit_edge ], [ 0.000000e+00, %entry ]
  %conv = fptosi double %.lcssa to i32
  ret i32 %conv
}

; Similar to foo_2double but with a non-power-of-2 factor and potential
; wrapping (both indices wrap or both don't in the same time)
; CHECK-LABEL: foo_2double_non_power_of_2
; CHECK: load <2 x double>
; CHECK: load <2 x double>
; Function Attrs: nounwind ssp uwtable
define void @foo_2double_non_power_of_2(i32 %u) #0 {
entry:
  %u.addr = alloca i32, align 4
  store i32 %u, i32* %u.addr, align 4
  %mul = mul i32 %u, 6
  %add6 = add i32 %mul, 6
  %idxprom = sext i32 %add6 to i64
  %arrayidx = getelementptr inbounds [2000 x double], [2000 x double]* @A, i32 0, i64 %idxprom
  %0 = load double, double* %arrayidx, align 8
  %arrayidx4 = getelementptr inbounds [2000 x double], [2000 x double]* @B, i32 0, i64 %idxprom
  %1 = load double, double* %arrayidx4, align 8
  %add5 = fadd double %0, %1
  store double %add5, double* %arrayidx, align 8
  %add7 = add i32 %mul, 7
  %idxprom12 = sext i32 %add7 to i64
  %arrayidx13 = getelementptr inbounds [2000 x double], [2000 x double]* @A, i32 0, i64 %idxprom12
  %2 = load double, double* %arrayidx13, align 8
  %arrayidx17 = getelementptr inbounds [2000 x double], [2000 x double]* @B, i32 0, i64 %idxprom12
  %3 = load double, double* %arrayidx17, align 8
  %add18 = fadd double %2, %3
  store double %add18, double* %arrayidx13, align 8
  ret void
}

; Similar to foo_2double_non_power_of_2 but with zext's instead of sext's
; CHECK-LABEL: foo_2double_non_power_of_2_zext
; CHECK: load <2 x double>
; CHECK: load <2 x double>
; Function Attrs: nounwind ssp uwtable
define void @foo_2double_non_power_of_2_zext(i32 %u) #0 {
entry:
  %u.addr = alloca i32, align 4
  store i32 %u, i32* %u.addr, align 4
  %mul = mul i32 %u, 6
  %add6 = add i32 %mul, 6
  %idxprom = zext i32 %add6 to i64
  %arrayidx = getelementptr inbounds [2000 x double], [2000 x double]* @A, i32 0, i64 %idxprom
  %0 = load double, double* %arrayidx, align 8
  %arrayidx4 = getelementptr inbounds [2000 x double], [2000 x double]* @B, i32 0, i64 %idxprom
  %1 = load double, double* %arrayidx4, align 8
  %add5 = fadd double %0, %1
  store double %add5, double* %arrayidx, align 8
  %add7 = add i32 %mul, 7
  %idxprom12 = zext i32 %add7 to i64
  %arrayidx13 = getelementptr inbounds [2000 x double], [2000 x double]* @A, i32 0, i64 %idxprom12
  %2 = load double, double* %arrayidx13, align 8
  %arrayidx17 = getelementptr inbounds [2000 x double], [2000 x double]* @B, i32 0, i64 %idxprom12
  %3 = load double, double* %arrayidx17, align 8
  %add18 = fadd double %2, %3
  store double %add18, double* %arrayidx13, align 8
  ret void
}

; Similar to foo_2double_non_power_of_2, but now we are dealing with AddRec SCEV.
; Alternatively, this is like foo_loop, but with a non-power-of-2 factor and
; potential wrapping (both indices wrap or both don't in the same time)
; CHECK-LABEL: foo_loop_non_power_of_2
; CHECK: <2 x double>
; Function Attrs: nounwind ssp uwtable
define i32 @foo_loop_non_power_of_2(double* %A, i32 %n) #0 {
entry:
  %A.addr = alloca double*, align 8
  %n.addr = alloca i32, align 4
  %sum = alloca double, align 8
  %i = alloca i32, align 4
  store double* %A, double** %A.addr, align 8
  store i32 %n, i32* %n.addr, align 4
  store double 0.000000e+00, double* %sum, align 8
  store i32 0, i32* %i, align 4
  %cmp1 = icmp slt i32 0, %n
  br i1 %cmp1, label %for.body.lr.ph, label %for.end

for.body.lr.ph:                                   ; preds = %entry
  br label %for.body

for.body:                                         ; preds = %for.body.lr.ph, %for.body
  %0 = phi i32 [ 0, %for.body.lr.ph ], [ %inc, %for.body ]
  %1 = phi double [ 0.000000e+00, %for.body.lr.ph ], [ %add7, %for.body ]
  %mul = mul i32 %0, 12
  %add.5 = add i32 %mul, 5
  %idxprom = sext i32 %add.5 to i64
  %arrayidx = getelementptr inbounds double, double* %A, i64 %idxprom
  %2 = load double, double* %arrayidx, align 8
  %mul1 = fmul double 7.000000e+00, %2
  %add.6 = add i32 %mul, 6
  %idxprom3 = sext i32 %add.6 to i64
  %arrayidx4 = getelementptr inbounds double, double* %A, i64 %idxprom3
  %3 = load double, double* %arrayidx4, align 8
  %mul5 = fmul double 7.000000e+00, %3
  %add6 = fadd double %mul1, %mul5
  %add7 = fadd double %1, %add6
  store double %add7, double* %sum, align 8
  %inc = add i32 %0, 1
  store i32 %inc, i32* %i, align 4
  %cmp = icmp slt i32 %inc, %n
  br i1 %cmp, label %for.body, label %for.cond.for.end_crit_edge

for.cond.for.end_crit_edge:                       ; preds = %for.body
  %split = phi double [ %add7, %for.body ]
  br label %for.end

for.end:                                          ; preds = %for.cond.for.end_crit_edge, %entry
  %.lcssa = phi double [ %split, %for.cond.for.end_crit_edge ], [ 0.000000e+00, %entry ]
  %conv = fptosi double %.lcssa to i32
  ret i32 %conv
}

; This is generated by `clang -std=c11 -Wpedantic -Wall -O3 main.c -S -o - -emit-llvm`
; with !{!"clang version 7.0.0 (trunk 337339) (llvm/trunk 337344)"} and stripping off
; the !tbaa metadata nodes to fit the rest of the test file, where `cat main.c` is:
;
;  double bar(double *a, unsigned n) {
;    double x = 0.0;
;    double y = 0.0;
;    for (unsigned i = 0; i < n; i += 2) {
;      x += a[i];
;      y += a[i + 1];
;    }
;    return x * y;
;  }
;
; The resulting IR is similar to @foo_loop, but with zext's instead of sext's.
;
; Make sure we are able to vectorize this from now on:
;
; CHECK-LABEL: @bar
; CHECK: load <2 x double>
define double @bar(double* nocapture readonly %a, i32 %n) local_unnamed_addr #0 {
entry:
  %cmp15 = icmp eq i32 %n, 0
  br i1 %cmp15, label %for.cond.cleanup, label %for.body

for.cond.cleanup:                                 ; preds = %for.body, %entry
  %x.0.lcssa = phi double [ 0.000000e+00, %entry ], [ %add, %for.body ]
  %y.0.lcssa = phi double [ 0.000000e+00, %entry ], [ %add4, %for.body ]
  %mul = fmul double %x.0.lcssa, %y.0.lcssa
  ret double %mul

for.body:                                         ; preds = %entry, %for.body
  %i.018 = phi i32 [ %add5, %for.body ], [ 0, %entry ]
  %y.017 = phi double [ %add4, %for.body ], [ 0.000000e+00, %entry ]
  %x.016 = phi double [ %add, %for.body ], [ 0.000000e+00, %entry ]
  %idxprom = zext i32 %i.018 to i64
  %arrayidx = getelementptr inbounds double, double* %a, i64 %idxprom
  %0 = load double, double* %arrayidx, align 8
  %add = fadd double %x.016, %0
  %add1 = or i32 %i.018, 1
  %idxprom2 = zext i32 %add1 to i64
  %arrayidx3 = getelementptr inbounds double, double* %a, i64 %idxprom2
  %1 = load double, double* %arrayidx3, align 8
  %add4 = fadd double %y.017, %1
  %add5 = add i32 %i.018, 2
  %cmp = icmp ult i32 %add5, %n
  br i1 %cmp, label %for.body, label %for.cond.cleanup
}

attributes #0 = { nounwind ssp uwtable "less-precise-fpmad"="false" "no-frame-pointer-elim"="true" "no-frame-pointer-elim-non-leaf" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "stack-protector-buffer-size"="8" "unsafe-fp-math"="false" "use-soft-float"="false" }

!llvm.ident = !{!0}

!0 = !{!"clang version 3.5.0 "}

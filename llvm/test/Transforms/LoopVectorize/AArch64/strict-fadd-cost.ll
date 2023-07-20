; REQUIRES: asserts
; RUN: opt < %s -passes=loop-vectorize -debug -disable-output -force-ordered-reductions=true -hints-allow-reordering=false \
; RUN:   -force-vector-width=4 -force-vector-interleave=1 -S 2>&1 | FileCheck %s --check-prefix=CHECK-VF4
; RUN: opt < %s -passes=loop-vectorize -debug -disable-output -force-ordered-reductions=true -hints-allow-reordering=false \
; RUN:   -force-vector-width=8 -force-vector-interleave=1 -S 2>&1 | FileCheck %s --check-prefix=CHECK-VF8

target triple="aarch64-unknown-linux-gnu"

; CHECK-VF4: Found an estimated cost of 17 for VF 4 For instruction:   %add = fadd float %0, %sum.07
; CHECK-VF8: Found an estimated cost of 34 for VF 8 For instruction:   %add = fadd float %0, %sum.07

define float @fadd_strict32(ptr noalias nocapture readonly %a, i64 %n) {
entry:
  br label %for.body

for.body:
  %iv = phi i64 [ 0, %entry ], [ %iv.next, %for.body ]
  %sum.07 = phi float [ 0.000000e+00, %entry ], [ %add, %for.body ]
  %arrayidx = getelementptr inbounds float, ptr %a, i64 %iv
  %0 = load float, ptr %arrayidx, align 4
  %add = fadd float %0, %sum.07
  %iv.next = add nuw nsw i64 %iv, 1
  %exitcond.not = icmp eq i64 %iv.next, %n
  br i1 %exitcond.not, label %for.end, label %for.body

for.end:
  ret float %add
}


; CHECK-VF4: Found an estimated cost of 14 for VF 4 For instruction:   %add = fadd double %0, %sum.07
; CHECK-VF8: Found an estimated cost of 28 for VF 8 For instruction:   %add = fadd double %0, %sum.07

define double @fadd_strict64(ptr noalias nocapture readonly %a, i64 %n) {
entry:
  br label %for.body

for.body:
  %iv = phi i64 [ 0, %entry ], [ %iv.next, %for.body ]
  %sum.07 = phi double [ 0.000000e+00, %entry ], [ %add, %for.body ]
  %arrayidx = getelementptr inbounds double, ptr %a, i64 %iv
  %0 = load double, ptr %arrayidx, align 4
  %add = fadd double %0, %sum.07
  %iv.next = add nuw nsw i64 %iv, 1
  %exitcond.not = icmp eq i64 %iv.next, %n
  br i1 %exitcond.not, label %for.end, label %for.body

for.end:
  ret double %add
}

; CHECK-VF4: Found an estimated cost of 19 for VF 4 For instruction:   %muladd = tail call float @llvm.fmuladd.f32(float %0, float %1, float %sum.07)
; CHECK-VF8: Found an estimated cost of 38 for VF 8 For instruction:   %muladd = tail call float @llvm.fmuladd.f32(float %0, float %1, float %sum.07)

define float @fmuladd_strict32(ptr %a, ptr %b, i64 %n) {
entry:
  br label %for.body

for.body:
  %iv = phi i64 [ 0, %entry ], [ %iv.next, %for.body ]
  %sum.07 = phi float [ 0.000000e+00, %entry ], [ %muladd, %for.body ]
  %arrayidx = getelementptr inbounds float, ptr %a, i64 %iv
  %0 = load float, ptr %arrayidx, align 4
  %arrayidx2 = getelementptr inbounds float, ptr %b, i64 %iv
  %1 = load float, ptr %arrayidx2, align 4
  %muladd = tail call float @llvm.fmuladd.f32(float %0, float %1, float %sum.07)
  %iv.next = add nuw nsw i64 %iv, 1
  %exitcond.not = icmp eq i64 %iv.next, %n
  br i1 %exitcond.not, label %for.end, label %for.body

for.end:
  ret float %muladd
}

declare float @llvm.fmuladd.f32(float, float, float)

; CHECK-VF4: Found an estimated cost of 18 for VF 4 For instruction:   %muladd = tail call double @llvm.fmuladd.f64(double %0, double %1, double %sum.07)
; CHECK-VF8: Found an estimated cost of 36 for VF 8 For instruction:   %muladd = tail call double @llvm.fmuladd.f64(double %0, double %1, double %sum.07)

define double @fmuladd_strict64(ptr %a, ptr %b, i64 %n) {
entry:
  br label %for.body

for.body:
  %iv = phi i64 [ 0, %entry ], [ %iv.next, %for.body ]
  %sum.07 = phi double [ 0.000000e+00, %entry ], [ %muladd, %for.body ]
  %arrayidx = getelementptr inbounds double, ptr %a, i64 %iv
  %0 = load double, ptr %arrayidx, align 4
  %arrayidx2 = getelementptr inbounds double, ptr %b, i64 %iv
  %1 = load double, ptr %arrayidx2, align 4
  %muladd = tail call double @llvm.fmuladd.f64(double %0, double %1, double %sum.07)
  %iv.next = add nuw nsw i64 %iv, 1
  %exitcond.not = icmp eq i64 %iv.next, %n
  br i1 %exitcond.not, label %for.end, label %for.body

for.end:
  ret double %muladd
}

declare double @llvm.fmuladd.f64(double, double, double)

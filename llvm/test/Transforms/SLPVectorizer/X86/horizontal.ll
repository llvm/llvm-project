; RUN: opt -slp-vectorizer -S <  %s -mtriple=x86_64-apple-macosx -mcpu=corei7-avx | FileCheck %s --check-prefix=NOSTORE

target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128-n8:16:32:64-S128"

; #include <stdint.h>
;
; int foo(float *A, int n) {
;   float sum = 0;
;   for (intptr_t i=0; i < n; ++i) {
;     sum += 7*A[i*4  ] +
;            7*A[i*4+1] +
;            7*A[i*4+2] +
;            7*A[i*4+3];
;   }
;   return sum;
; }

; NOSTORE-LABEL: add_red
; NOSTORE: fmul <4 x float>
; NOSTORE: shufflevector <4 x float>

define i32 @add_red(float* %A, i32 %n) {
entry:
  %cmp31 = icmp sgt i32 %n, 0
  br i1 %cmp31, label %for.body.lr.ph, label %for.end

for.body.lr.ph:
  %0 = sext i32 %n to i64
  br label %for.body

for.body:
  %i.033 = phi i64 [ 0, %for.body.lr.ph ], [ %inc, %for.body ]
  %sum.032 = phi float [ 0.000000e+00, %for.body.lr.ph ], [ %add17, %for.body ]
  %mul = shl nsw i64 %i.033, 2
  %arrayidx = getelementptr inbounds float, float* %A, i64 %mul
  %1 = load float, float* %arrayidx, align 4
  %mul2 = fmul float %1, 7.000000e+00
  %add28 = or i64 %mul, 1
  %arrayidx4 = getelementptr inbounds float, float* %A, i64 %add28
  %2 = load float, float* %arrayidx4, align 4
  %mul5 = fmul float %2, 7.000000e+00
  %add6 = fadd fast float %mul2, %mul5
  %add829 = or i64 %mul, 2
  %arrayidx9 = getelementptr inbounds float, float* %A, i64 %add829
  %3 = load float, float* %arrayidx9, align 4
  %mul10 = fmul float %3, 7.000000e+00
  %add11 = fadd fast float %add6, %mul10
  %add1330 = or i64 %mul, 3
  %arrayidx14 = getelementptr inbounds float, float* %A, i64 %add1330
  %4 = load float, float* %arrayidx14, align 4
  %mul15 = fmul float %4, 7.000000e+00
  %add16 = fadd fast float %add11, %mul15
  %add17 = fadd fast float %sum.032, %add16
  %inc = add nsw i64 %i.033, 1
  %exitcond = icmp eq i64 %inc, %0
  br i1 %exitcond, label %for.cond.for.end_crit_edge, label %for.body

for.cond.for.end_crit_edge:
  %phitmp = fptosi float %add17 to i32
  br label %for.end

for.end:
  %sum.0.lcssa = phi i32 [ %phitmp, %for.cond.for.end_crit_edge ], [ 0, %entry ]
  ret i32 %sum.0.lcssa
}

; int foo(float * restrict A, float * restrict B, int n) {
;   float sum = 0;
;   for (intptr_t i=0; i < n; ++i) {
;     sum *= B[0]*A[i*4  ] +
;       B[1]*A[i*4+1] +
;       B[2]*A[i*4+2] +
;       B[3]*A[i*4+3];
;   }
;   return sum;
; }

; CHECK-LABEL: mul_red
; CHECK: fmul <4 x float>
; CHECK: shufflevector <4 x float>

define i32 @mul_red(float* noalias %A, float* noalias %B, i32 %n) {
entry:
  %cmp38 = icmp sgt i32 %n, 0
  br i1 %cmp38, label %for.body.lr.ph, label %for.end

for.body.lr.ph:
  %0 = load float, float* %B, align 4
  %arrayidx4 = getelementptr inbounds float, float* %B, i64 1
  %1 = load float, float* %arrayidx4, align 4
  %arrayidx9 = getelementptr inbounds float, float* %B, i64 2
  %2 = load float, float* %arrayidx9, align 4
  %arrayidx15 = getelementptr inbounds float, float* %B, i64 3
  %3 = load float, float* %arrayidx15, align 4
  %4 = sext i32 %n to i64
  br label %for.body

for.body:
  %i.040 = phi i64 [ 0, %for.body.lr.ph ], [ %inc, %for.body ]
  %sum.039 = phi float [ 0.000000e+00, %for.body.lr.ph ], [ %mul21, %for.body ]
  %mul = shl nsw i64 %i.040, 2
  %arrayidx2 = getelementptr inbounds float, float* %A, i64 %mul
  %5 = load float, float* %arrayidx2, align 4
  %mul3 = fmul float %0, %5
  %add35 = or i64 %mul, 1
  %arrayidx6 = getelementptr inbounds float, float* %A, i64 %add35
  %6 = load float, float* %arrayidx6, align 4
  %mul7 = fmul float %1, %6
  %add8 = fadd fast float %mul3, %mul7
  %add1136 = or i64 %mul, 2
  %arrayidx12 = getelementptr inbounds float, float* %A, i64 %add1136
  %7 = load float, float* %arrayidx12, align 4
  %mul13 = fmul float %2, %7
  %add14 = fadd fast float %add8, %mul13
  %add1737 = or i64 %mul, 3
  %arrayidx18 = getelementptr inbounds float, float* %A, i64 %add1737
  %8 = load float, float* %arrayidx18, align 4
  %mul19 = fmul float %3, %8
  %add20 = fadd fast float %add14, %mul19
  %mul21 = fmul float %sum.039, %add20
  %inc = add nsw i64 %i.040, 1
  %exitcond = icmp eq i64 %inc, %4
  br i1 %exitcond, label %for.cond.for.end_crit_edge, label %for.body

for.cond.for.end_crit_edge:
  %phitmp = fptosi float %mul21 to i32
  br label %for.end

for.end:
  %sum.0.lcssa = phi i32 [ %phitmp, %for.cond.for.end_crit_edge ], [ 0, %entry ]
  ret i32 %sum.0.lcssa
}

; int foo(float * restrict A, float * restrict B, int n) {
;   float sum = 0;
;   for (intptr_t i=0; i < n; ++i) {
;     sum += B[0]*A[i*6  ] +
;            B[1]*A[i*6+1] +
;            B[2]*A[i*6+2] +
;            B[3]*A[i*6+3] +
;            B[4]*A[i*6+4] +
;            B[5]*A[i*6+5] +
;            B[6]*A[i*6+6] +
;            B[7]*A[i*6+7] +
;            B[8]*A[i*6+8];
;   }
;   return sum;
; }

; CHECK-LABEL: long_red
; CHECK: fmul fast <4 x float>
; CHECK: shufflevector <4 x float>

define i32 @long_red(float* noalias %A, float* noalias %B, i32 %n) {
entry:
  %cmp81 = icmp sgt i32 %n, 0
  br i1 %cmp81, label %for.body.lr.ph, label %for.end

for.body.lr.ph:
  %0 = load float, float* %B, align 4
  %arrayidx4 = getelementptr inbounds float, float* %B, i64 1
  %1 = load float, float* %arrayidx4, align 4
  %arrayidx9 = getelementptr inbounds float, float* %B, i64 2
  %2 = load float, float* %arrayidx9, align 4
  %arrayidx15 = getelementptr inbounds float, float* %B, i64 3
  %3 = load float, float* %arrayidx15, align 4
  %arrayidx21 = getelementptr inbounds float, float* %B, i64 4
  %4 = load float, float* %arrayidx21, align 4
  %arrayidx27 = getelementptr inbounds float, float* %B, i64 5
  %5 = load float, float* %arrayidx27, align 4
  %arrayidx33 = getelementptr inbounds float, float* %B, i64 6
  %6 = load float, float* %arrayidx33, align 4
  %arrayidx39 = getelementptr inbounds float, float* %B, i64 7
  %7 = load float, float* %arrayidx39, align 4
  %arrayidx45 = getelementptr inbounds float, float* %B, i64 8
  %8 = load float, float* %arrayidx45, align 4
  %9 = sext i32 %n to i64
  br label %for.body

for.body:
  %i.083 = phi i64 [ 0, %for.body.lr.ph ], [ %inc, %for.body ]
  %sum.082 = phi float [ 0.000000e+00, %for.body.lr.ph ], [ %add51, %for.body ]
  %mul = mul nsw i64 %i.083, 6
  %arrayidx2 = getelementptr inbounds float, float* %A, i64 %mul
  %10 = load float, float* %arrayidx2, align 4
  %mul3 = fmul fast float %0, %10
  %add80 = or i64 %mul, 1
  %arrayidx6 = getelementptr inbounds float, float* %A, i64 %add80
  %11 = load float, float* %arrayidx6, align 4
  %mul7 = fmul fast float %1, %11
  %add8 = fadd fast float %mul3, %mul7
  %add11 = add nsw i64 %mul, 2
  %arrayidx12 = getelementptr inbounds float, float* %A, i64 %add11
  %12 = load float, float* %arrayidx12, align 4
  %mul13 = fmul fast float %2, %12
  %add14 = fadd fast float %add8, %mul13
  %add17 = add nsw i64 %mul, 3
  %arrayidx18 = getelementptr inbounds float, float* %A, i64 %add17
  %13 = load float, float* %arrayidx18, align 4
  %mul19 = fmul fast float %3, %13
  %add20 = fadd fast float %add14, %mul19
  %add23 = add nsw i64 %mul, 4
  %arrayidx24 = getelementptr inbounds float, float* %A, i64 %add23
  %14 = load float, float* %arrayidx24, align 4
  %mul25 = fmul fast float %4, %14
  %add26 = fadd fast float %add20, %mul25
  %add29 = add nsw i64 %mul, 5
  %arrayidx30 = getelementptr inbounds float, float* %A, i64 %add29
  %15 = load float, float* %arrayidx30, align 4
  %mul31 = fmul fast float %5, %15
  %add32 = fadd fast float %add26, %mul31
  %add35 = add nsw i64 %mul, 6
  %arrayidx36 = getelementptr inbounds float, float* %A, i64 %add35
  %16 = load float, float* %arrayidx36, align 4
  %mul37 = fmul fast float %6, %16
  %add38 = fadd fast float %add32, %mul37
  %add41 = add nsw i64 %mul, 7
  %arrayidx42 = getelementptr inbounds float, float* %A, i64 %add41
  %17 = load float, float* %arrayidx42, align 4
  %mul43 = fmul fast float %7, %17
  %add44 = fadd fast float %add38, %mul43
  %add47 = add nsw i64 %mul, 8
  %arrayidx48 = getelementptr inbounds float, float* %A, i64 %add47
  %18 = load float, float* %arrayidx48, align 4
  %mul49 = fmul fast float %8, %18
  %add50 = fadd fast float %add44, %mul49
  %add51 = fadd fast float %sum.082, %add50
  %inc = add nsw i64 %i.083, 1
  %exitcond = icmp eq i64 %inc, %9
  br i1 %exitcond, label %for.cond.for.end_crit_edge, label %for.body

for.cond.for.end_crit_edge:
  %phitmp = fptosi float %add51 to i32
  br label %for.end

for.end:
  %sum.0.lcssa = phi i32 [ %phitmp, %for.cond.for.end_crit_edge ], [ 0, %entry ]
  ret i32 %sum.0.lcssa
}

; int foo(float * restrict A, float * restrict B, int n) {
;   float sum = 0;
;   for (intptr_t i=0; i < n; ++i) {
;     sum += B[0]*A[i*4  ];
;     sum += B[1]*A[i*4+1];
;     sum += B[2]*A[i*4+2];
;     sum += B[3]*A[i*4+3];
;   }
;   return sum;
; }

; CHECK-LABEL: chain_red
; CHECK: fmul fast <4 x float>
; CHECK: shufflevector <4 x float>

define i32 @chain_red(float* noalias %A, float* noalias %B, i32 %n) {
entry:
  %cmp41 = icmp sgt i32 %n, 0
  br i1 %cmp41, label %for.body.lr.ph, label %for.end

for.body.lr.ph:
  %0 = load float, float* %B, align 4
  %arrayidx4 = getelementptr inbounds float, float* %B, i64 1
  %1 = load float, float* %arrayidx4, align 4
  %arrayidx10 = getelementptr inbounds float, float* %B, i64 2
  %2 = load float, float* %arrayidx10, align 4
  %arrayidx16 = getelementptr inbounds float, float* %B, i64 3
  %3 = load float, float* %arrayidx16, align 4
  %4 = sext i32 %n to i64
  br label %for.body

for.body:
  %i.043 = phi i64 [ 0, %for.body.lr.ph ], [ %inc, %for.body ]
  %sum.042 = phi float [ 0.000000e+00, %for.body.lr.ph ], [ %add21, %for.body ]
  %mul = shl nsw i64 %i.043, 2
  %arrayidx2 = getelementptr inbounds float, float* %A, i64 %mul
  %5 = load float, float* %arrayidx2, align 4
  %mul3 = fmul fast float %0, %5
  %add = fadd fast float %sum.042, %mul3
  %add638 = or i64 %mul, 1
  %arrayidx7 = getelementptr inbounds float, float* %A, i64 %add638
  %6 = load float, float* %arrayidx7, align 4
  %mul8 = fmul fast float %1, %6
  %add9 = fadd fast float %add, %mul8
  %add1239 = or i64 %mul, 2
  %arrayidx13 = getelementptr inbounds float, float* %A, i64 %add1239
  %7 = load float, float* %arrayidx13, align 4
  %mul14 = fmul fast float %2, %7
  %add15 = fadd fast float %add9, %mul14
  %add1840 = or i64 %mul, 3
  %arrayidx19 = getelementptr inbounds float, float* %A, i64 %add1840
  %8 = load float, float* %arrayidx19, align 4
  %mul20 = fmul fast float %3, %8
  %add21 = fadd fast float %add15, %mul20
  %inc = add nsw i64 %i.043, 1
  %exitcond = icmp eq i64 %inc, %4
  br i1 %exitcond, label %for.cond.for.end_crit_edge, label %for.body

for.cond.for.end_crit_edge:
  %phitmp = fptosi float %add21 to i32
  br label %for.end

for.end:
  %sum.0.lcssa = phi i32 [ %phitmp, %for.cond.for.end_crit_edge ], [ 0, %entry ]
  ret i32 %sum.0.lcssa
}

; int foo(float * restrict A, float * restrict B, float * restrict C, int n) {
;   float sum = 0;
;   for (intptr_t i=0; i < n; ++i) {
;     C[i] = B[0] *A[i*4  ] +
;          B[1] *A[i*4+1] +
;          B[2] *A[i*4+2] +
;          B[3] *A[i*4+3];
;   }
;   return sum;
; }

; CHECK-LABEL: store_red
; CHECK: fmul fast <4 x float>
; CHECK: shufflevector <4 x float>

define i32 @store_red(float* noalias %A, float* noalias %B, float* noalias %C, i32 %n) {
entry:
  %cmp37 = icmp sgt i32 %n, 0
  br i1 %cmp37, label %for.body.lr.ph, label %for.end

for.body.lr.ph:
  %arrayidx4 = getelementptr inbounds float, float* %B, i64 1
  %arrayidx9 = getelementptr inbounds float, float* %B, i64 2
  %arrayidx15 = getelementptr inbounds float, float* %B, i64 3
  %0 = sext i32 %n to i64
  br label %for.body

for.body:
  %i.039 = phi i64 [ 0, %for.body.lr.ph ], [ %inc, %for.body ]
  %C.addr.038 = phi float* [ %C, %for.body.lr.ph ], [ %incdec.ptr, %for.body ]
  %1 = load float, float* %B, align 4
  %mul = shl nsw i64 %i.039, 2
  %arrayidx2 = getelementptr inbounds float, float* %A, i64 %mul
  %2 = load float, float* %arrayidx2, align 4
  %mul3 = fmul fast float %1, %2
  %3 = load float, float* %arrayidx4, align 4
  %add34 = or i64 %mul, 1
  %arrayidx6 = getelementptr inbounds float, float* %A, i64 %add34
  %4 = load float, float* %arrayidx6, align 4
  %mul7 = fmul fast float %3, %4
  %add8 = fadd fast float %mul3, %mul7
  %5 = load float, float* %arrayidx9, align 4
  %add1135 = or i64 %mul, 2
  %arrayidx12 = getelementptr inbounds float, float* %A, i64 %add1135
  %6 = load float, float* %arrayidx12, align 4
  %mul13 = fmul fast float %5, %6
  %add14 = fadd fast float %add8, %mul13
  %7 = load float, float* %arrayidx15, align 4
  %add1736 = or i64 %mul, 3
  %arrayidx18 = getelementptr inbounds float, float* %A, i64 %add1736
  %8 = load float, float* %arrayidx18, align 4
  %mul19 = fmul fast float %7, %8
  %add20 = fadd fast float %add14, %mul19
  store float %add20, float* %C.addr.038, align 4
  %incdec.ptr = getelementptr inbounds float, float* %C.addr.038, i64 1
  %inc = add nsw i64 %i.039, 1
  %exitcond = icmp eq i64 %inc, %0
  br i1 %exitcond, label %for.end, label %for.body

for.end:
  ret i32 0
}


; RUN: opt -slp-vectorizer -slp-vectorize-hor -slp-vectorize-hor-store -S <  %s -mtriple=x86_64-apple-macosx -mcpu=corei7-avx | FileCheck %s --check-prefix=STORE

; void foo(double * restrict A, double * restrict B, double * restrict C,
;          int n) {
;   for (intptr_t i=0; i < n; ++i) {
;     C[i] = B[0] *A[i*4  ] + B[1] *A[i*4+1];
;   }
; }

; STORE-LABEL: store_red_double
; STORE: fmul fast <2 x double>
; STORE: extractelement <2 x double>
; STORE: extractelement <2 x double>

define void @store_red_double(double* noalias %A, double* noalias %B, double* noalias %C, i32 %n) {
entry:
  %cmp17 = icmp sgt i32 %n, 0
  br i1 %cmp17, label %for.body.lr.ph, label %for.end

for.body.lr.ph:
  %0 = load double, double* %B, align 8
  %arrayidx4 = getelementptr inbounds double, double* %B, i64 1
  %1 = load double, double* %arrayidx4, align 8
  %2 = sext i32 %n to i64
  br label %for.body

for.body:
  %i.018 = phi i64 [ 0, %for.body.lr.ph ], [ %inc, %for.body ]
  %mul = shl nsw i64 %i.018, 2
  %arrayidx2 = getelementptr inbounds double, double* %A, i64 %mul
  %3 = load double, double* %arrayidx2, align 8
  %mul3 = fmul fast double %0, %3
  %add16 = or i64 %mul, 1
  %arrayidx6 = getelementptr inbounds double, double* %A, i64 %add16
  %4 = load double, double* %arrayidx6, align 8
  %mul7 = fmul fast double %1, %4
  %add8 = fadd fast double %mul3, %mul7
  %arrayidx9 = getelementptr inbounds double, double* %C, i64 %i.018
  store double %add8, double* %arrayidx9, align 8
  %inc = add nsw i64 %i.018, 1
  %exitcond = icmp eq i64 %inc, %2
  br i1 %exitcond, label %for.end, label %for.body

for.end:
  ret void
}

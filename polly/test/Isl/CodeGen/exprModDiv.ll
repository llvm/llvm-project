; RUN: opt %loadPolly -polly-import-jscop -polly-import-jscop-dir=%S \
; RUN:     -polly-codegen -S < %s | FileCheck %s
; RUN: opt %loadPolly -polly-import-jscop -polly-import-jscop-dir=%S \
; RUN:     -polly-codegen -polly-import-jscop-postfix=pow2 \
; RUN:     -S < %s | FileCheck %s -check-prefix=POW2
;
;    void exprModDiv(float *A, float *B, float *C, long N, long p) {
;      for (long i = 0; i < N; i++)
;        C[i] += A[i] + B[i] + A[i] + B[i + p];
;    }
;
;
; This test case changes the access functions such that the resulting index
; expressions are modulo or division operations. We test that the code we
; generate takes advantage of knowledge about unsigned numerators. This is
; useful as LLVM will translate urem and udiv operations with power-of-two
; denominators to fast bitwise and or shift operations.

; A[i % 127]
; CHECK:  %pexp.pdiv_r = urem i64 %polly.indvar, 127
; CHECK:  %polly.access.A6 = getelementptr float, float* %A, i64 %pexp.pdiv_r

; A[floor(i / 127)]
;
; Note: without the floor, we would create a map i -> i/127, which only contains
;       values of i that are divisible by 127. All other values of i would not
;       be mapped to any value. However, to generate correct code we require
;       each value of i to indeed be mapped to a value.
;
; CHECK:  %pexp.p_div_q = udiv i64 %polly.indvar, 127
; CHECK:  %polly.access.B7 = getelementptr float, float* %B, i64 %pexp.p_div_q

; #define floord(n,d) ((n < 0) ? (n - d + 1) : n) / d
; A[p + 127 * floord(-p - 1, 127) + 127]
; CHECK:  %pexp.fdiv_q.0 = sub i64 %p, 127
; CHECK:  %pexp.fdiv_q.1 = add i64 %pexp.fdiv_q.0, 1
; CHECK:  %pexp.fdiv_q.2 = icmp slt i64 %p, 0
; CHECK:  %pexp.fdiv_q.3 = select i1 %pexp.fdiv_q.2, i64 %pexp.fdiv_q.1, i64 %p
; CHECK:  %pexp.fdiv_q.4 = sdiv i64 %pexp.fdiv_q.3, 127
; CHECK:  %17 = mul nsw i64 127, %pexp.fdiv_q.4
; CHECK:  %18 = sub nsw i64 %p, %17
; CHECK:  %polly.access.A8 = getelementptr float, float* %A, i64 %18

; A[p / 127]
; CHECK:  %pexp.div = sdiv exact i64 %p, 127
; CHECK:  %polly.access.B9 = getelementptr float, float* %B, i64 %pexp.div

; A[i % 128]
; POW2:  %pexp.pdiv_r = urem i64 %polly.indvar, 128
; POW2:  %polly.access.A6 = getelementptr float, float* %A, i64 %pexp.pdiv_r

; A[floor(i / 128)]
; POW2:  %pexp.p_div_q = udiv i64 %polly.indvar, 128
; POW2:  %polly.access.B7 = getelementptr float, float* %B, i64 %pexp.p_div_q

; #define floord(n,d) ((n < 0) ? (n - d + 1) : n) / d
; A[p + 128 * floord(-p - 1, 128) + 128]
; POW2:  %polly.fdiv_q.shr = ashr i64 %p, 7
; POW2:  %17 = mul nsw i64 128, %polly.fdiv_q.shr
; POW2:  %18 = sub nsw i64 %p, %17
; POW2:  %polly.access.A8 = getelementptr float, float* %A, i64 %18

; A[p / 128]
; POW2:  %pexp.div = sdiv exact i64 %p, 128
; POW2:  %polly.access.B9 = getelementptr float, float* %B, i64 %pexp.div

target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"

define void @exprModDiv(float* %A, float* %B, float* %C, i64 %N, i64 %p) {
entry:
  br label %for.cond

for.cond:                                         ; preds = %for.inc, %entry
  %i.0 = phi i64 [ 0, %entry ], [ %inc, %for.inc ]
  %cmp = icmp slt i64 %i.0, %N
  br i1 %cmp, label %for.body, label %for.end

for.body:                                         ; preds = %for.cond
  %arrayidx = getelementptr inbounds float, float* %A, i64 %i.0
  %tmp = load float, float* %arrayidx, align 4
  %arrayidx1 = getelementptr inbounds float, float* %B, i64 %i.0
  %tmp1 = load float, float* %arrayidx1, align 4
  %add = fadd float %tmp, %tmp1
  %arrayidx2 = getelementptr inbounds float, float* %A, i64 %i.0
  %tmp2 = load float, float* %arrayidx2, align 4
  %add3 = fadd float %add, %tmp2
  %padd = add nsw i64 %p, %i.0
  %arrayidx4 = getelementptr inbounds float, float* %B, i64 %padd
  %tmp3 = load float, float* %arrayidx4, align 4
  %add5 = fadd float %add3, %tmp3
  %arrayidx6 = getelementptr inbounds float, float* %C, i64 %i.0
  %tmp4 = load float, float* %arrayidx6, align 4
  %add7 = fadd float %tmp4, %add5
  store float %add7, float* %arrayidx6, align 4
  br label %for.inc

for.inc:                                          ; preds = %for.body
  %inc = add nuw nsw i64 %i.0, 1
  br label %for.cond

for.end:                                          ; preds = %for.cond
  ret void
}

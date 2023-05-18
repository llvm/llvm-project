; RUN: opt %loadPolly -polly-import-jscop \
; RUN:     -polly-codegen -S < %s | FileCheck %s
; RUN: opt %loadPolly -polly-import-jscop \
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
; CHECK:  %polly.access.A9 = getelementptr float, ptr %A, i64 %pexp.pdiv_r

; A[floor(i / 127)]
;
; Note: without the floor, we would create a map i -> i/127, which only contains
;       values of i that are divisible by 127. All other values of i would not
;       be mapped to any value. However, to generate correct code we require
;       each value of i to indeed be mapped to a value.
;
; CHECK:  %pexp.p_div_q = udiv i64 %polly.indvar, 127
; CHECK:  %polly.access.B10 = getelementptr float, ptr %B, i64 %pexp.p_div_q

; A[p % 128]

; A[p / 127]
; CHECK:  %pexp.div = sdiv exact i64 %p, 127
; CHECK:  %polly.access.B12 = getelementptr float, ptr %B, i64 %pexp.div

; A[i % 128]
; POW2:  %pexp.pdiv_r = urem i64 %polly.indvar, 128
; POW2:  %polly.access.A9 = getelementptr float, ptr %A, i64 %pexp.pdiv_r

; A[floor(i / 128)]
; POW2:  %pexp.p_div_q = udiv i64 %polly.indvar, 128
; POW2:  %polly.access.B10 = getelementptr float, ptr %B, i64 %pexp.p_div_q

; A[p % 128]

; A[p / 128]
; POW2:  %pexp.div = sdiv exact i64 %p, 128
; POW2:  %polly.access.B12 = getelementptr float, ptr %B, i64 %pexp.div

target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"

define void @exprModDiv(ptr %A, ptr %B, ptr %C, i64 %N, i64 %p) {
entry:
  br label %for.cond

for.cond:                                         ; preds = %for.inc, %entry
  %i.0 = phi i64 [ 0, %entry ], [ %inc, %for.inc ]
  %cmp = icmp slt i64 %i.0, %N
  br i1 %cmp, label %for.body, label %for.end

for.body:                                         ; preds = %for.cond
  %arrayidx = getelementptr inbounds float, ptr %A, i64 %i.0
  %tmp = load float, ptr %arrayidx, align 4
  %arrayidx1 = getelementptr inbounds float, ptr %B, i64 %i.0
  %tmp1 = load float, ptr %arrayidx1, align 4
  %add = fadd float %tmp, %tmp1
  %arrayidx2 = getelementptr inbounds float, ptr %A, i64 %i.0
  %tmp2 = load float, ptr %arrayidx2, align 4
  %add3 = fadd float %add, %tmp2
  %padd = add nsw i64 %p, %i.0
  %arrayidx4 = getelementptr inbounds float, ptr %B, i64 %padd
  %tmp3 = load float, ptr %arrayidx4, align 4
  %add5 = fadd float %add3, %tmp3
  %arrayidx6 = getelementptr inbounds float, ptr %C, i64 %i.0
  %tmp4 = load float, ptr %arrayidx6, align 4
  %add7 = fadd float %tmp4, %add5
  store float %add7, ptr %arrayidx6, align 4
  br label %for.inc

for.inc:                                          ; preds = %for.body
  %inc = add nuw nsw i64 %i.0, 1
  br label %for.cond

for.end:                                          ; preds = %for.cond
  ret void
}

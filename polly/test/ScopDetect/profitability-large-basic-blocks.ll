; RUN: opt %loadPolly -polly-process-unprofitable=false \
; RUN:                -polly-detect-profitability-min-per-loop-insts=40 \
; RUN: -polly-print-detect -disable-output < %s | FileCheck %s -check-prefix=PROFITABLE

; RUN: opt %loadPolly -polly-process-unprofitable=true \
; RUN: -polly-print-detect -disable-output < %s | FileCheck %s -check-prefix=PROFITABLE

; RUN: opt %loadPolly -polly-process-unprofitable=false \
; RUN: -polly-print-detect -disable-output < %s | FileCheck %s -check-prefix=UNPROFITABLE

; UNPROFITABLE-NOT: Valid Region for Scop:
; PROFITABLE: Valid Region for Scop:

;    void foo(float *A, float *B, long N) {
;      for (long i = 0; i < 100; i++)
;          A[i] += .... / * a  lot of compute */
;    }
;
target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"

define void @foo(float* %A, float* %B, i64 %N) {
entry:
  br label %header

header:
  %i.0 = phi i64 [ 0, %entry ], [ %tmp10, %header ]
  %tmp5 = sitofp i64 %i.0 to float
  %tmp6 = getelementptr inbounds float, float* %A, i64 %i.0
  %tmp7 = load float, float* %tmp6, align 4
  %tmp8 = fadd float %tmp7, %tmp5
  %val0 = fadd float %tmp7, 1.0
  %val1 = fadd float %val0, 1.0
  %val2 = fadd float %val1, 1.0
  %val3 = fadd float %val2, 1.0
  %val4 = fadd float %val3, 1.0
  %val5 = fadd float %val4, 1.0
  %val6 = fadd float %val5, 1.0
  %val7 = fadd float %val6, 1.0
  %val8 = fadd float %val7, 1.0
  %val9 = fadd float %val8, 1.0
  %val10 = fadd float %val9, 1.0
  %val11 = fadd float %val10, 1.0
  %val12 = fadd float %val11, 1.0
  %val13 = fadd float %val12, 1.0
  %val14 = fadd float %val13, 1.0
  %val15 = fadd float %val14, 1.0
  %val16 = fadd float %val15, 1.0
  %val17 = fadd float %val16, 1.0
  %val18 = fadd float %val17, 1.0
  %val19 = fadd float %val18, 1.0
  %val20 = fadd float %val19, 1.0
  %val21 = fadd float %val20, 1.0
  %val22 = fadd float %val21, 1.0
  %val23 = fadd float %val22, 1.0
  %val24 = fadd float %val23, 1.0
  %val25 = fadd float %val24, 1.0
  %val26 = fadd float %val25, 1.0
  %val27 = fadd float %val26, 1.0
  %val28 = fadd float %val27, 1.0
  %val29 = fadd float %val28, 1.0
  %val30 = fadd float %val29, 1.0
  %val31 = fadd float %val30, 1.0
  %val32 = fadd float %val31, 1.0
  %val33 = fadd float %val32, 1.0
  %val34 = fadd float %val33, 1.0
  %val35 = fadd float %val34, 1.0
  %val36 = fadd float %val35, 1.0
  %val37 = fadd float %val36, 1.0
  %val38 = fadd float %val37, 1.0
  %val39 = fadd float %val38, 1.0
  %val40 = fadd float %val39, 1.0
  %val41 = fadd float %val40, 1.0
  %val42 = fadd float %val41, 1.0
  %val43 = fadd float %val42, 1.0
  store float %val34, float* %tmp6, align 4
  %exitcond = icmp ne i64 %i.0, 100
  %tmp10 = add nsw i64 %i.0, 1
  br i1 %exitcond, label %header, label %exit

exit:
  ret void
}

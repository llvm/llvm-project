; RUN: opt %loadPolly -polly-codegen-ppcg -disable-output \
; RUN: -polly-acc-dump-code < %s | FileCheck %s -check-prefix=CODE

; RUN: opt %loadPolly -polly-codegen-ppcg \
; RUN: -S < %s | FileCheck %s -check-prefix=IR
;    void foo(float A[2][100]) {
;      for (long t = 0; t < 100; t++)
;        for (long i = 1; i < 99; i++)
;          A[(t + 1) % 2][i] += A[t % 2][i - 1] + A[t % 2][i] + A[t % 2][i + 1];
;    }

; REQUIRES: pollyacc

; CODE: # host
; CODE-NEXT: {
; CODE-NEXT:   cudaCheckReturn(cudaMemcpy(dev_MemRef_A, MemRef_A, (2) * (100) * sizeof(float), cudaMemcpyHostToDevice));
; CODE-NEXT:   for (int c0 = 0; c0 <= 99; c0 += 1)
; CODE-NEXT:     {
; CODE-NEXT:       dim3 k0_dimBlock(32);
; CODE-NEXT:       dim3 k0_dimGrid(4);
; CODE-NEXT:       kernel0 <<<k0_dimGrid, k0_dimBlock>>> (dev_MemRef_A, c0);
; CODE-NEXT:       cudaCheckKernel();
; CODE-NEXT:     }

; CODE:   cudaCheckReturn(cudaMemcpy(MemRef_A, dev_MemRef_A, (2) * (100) * sizeof(float), cudaMemcpyDeviceToHost));
; CODE-NEXT: }

; IR-LABEL: polly.loop_header:                                ; preds = %polly.loop_header, %polly.loop_preheader
; IR-NEXT:   %polly.indvar = phi i64 [ 0, %polly.loop_preheader ], [ %polly.indvar_next, %polly.loop_header ]
; IR-NEXT:   %polly.indvar_next = add nsw i64 %polly.indvar, 1
; IR-NEXT:   %polly.loop_cond = icmp sle i64 %polly.indvar, 98
; IR-NEXT:   br i1 %polly.loop_cond, label %polly.loop_header, label %polly.loop_exit

target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"

define void @foo([100 x float]* %A) {
entry:
  br label %for.cond

for.cond:                                         ; preds = %for.inc18, %entry
  %t.0 = phi i64 [ 0, %entry ], [ %inc19, %for.inc18 ]
  %exitcond1 = icmp ne i64 %t.0, 100
  br i1 %exitcond1, label %for.body, label %for.end20

for.body:                                         ; preds = %for.cond
  br label %for.cond1

for.cond1:                                        ; preds = %for.inc, %for.body
  %i.0 = phi i64 [ 1, %for.body ], [ %inc, %for.inc ]
  %exitcond = icmp ne i64 %i.0, 99
  br i1 %exitcond, label %for.body3, label %for.end

for.body3:                                        ; preds = %for.cond1
  %sub = add nsw i64 %i.0, -1
  %rem = srem i64 %t.0, 2
  %arrayidx4 = getelementptr inbounds [100 x float], [100 x float]* %A, i64 %rem, i64 %sub
  %tmp = load float, float* %arrayidx4, align 4
  %rem5 = srem i64 %t.0, 2
  %arrayidx7 = getelementptr inbounds [100 x float], [100 x float]* %A, i64 %rem5, i64 %i.0
  %tmp2 = load float, float* %arrayidx7, align 4
  %add = fadd float %tmp, %tmp2
  %add8 = add nuw nsw i64 %i.0, 1
  %rem9 = srem i64 %t.0, 2
  %arrayidx11 = getelementptr inbounds [100 x float], [100 x float]* %A, i64 %rem9, i64 %add8
  %tmp3 = load float, float* %arrayidx11, align 4
  %add12 = fadd float %add, %tmp3
  %add13 = add nuw nsw i64 %t.0, 1
  %rem14 = srem i64 %add13, 2
  %arrayidx16 = getelementptr inbounds [100 x float], [100 x float]* %A, i64 %rem14, i64 %i.0
  %tmp4 = load float, float* %arrayidx16, align 4
  %add17 = fadd float %tmp4, %add12
  store float %add17, float* %arrayidx16, align 4
  br label %for.inc

for.inc:                                          ; preds = %for.body3
  %inc = add nuw nsw i64 %i.0, 1
  br label %for.cond1

for.end:                                          ; preds = %for.cond1
  br label %for.inc18

for.inc18:                                        ; preds = %for.end
  %inc19 = add nuw nsw i64 %t.0, 1
  br label %for.cond

for.end20:                                        ; preds = %for.cond
  ret void
}

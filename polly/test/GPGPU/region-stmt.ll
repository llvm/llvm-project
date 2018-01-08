; RUN: opt %loadPolly -polly-codegen-ppcg -polly-acc-dump-code \
; RUN: -disable-output < %s | \
; RUN: FileCheck -check-prefix=CODE %s

; RUN: opt %loadPolly -polly-codegen-ppcg -S < %s | \
; RUN: FileCheck %s -check-prefix=IR

; CODE:        cudaCheckReturn(cudaMemcpy(dev_MemRef_A, MemRef_A, (128) * sizeof(float), cudaMemcpyHostToDevice));
; CODE-NEXT:   cudaCheckReturn(cudaMemcpy(dev_MemRef_B, MemRef_B, (128) * sizeof(float), cudaMemcpyHostToDevice));
; CODE-NEXT:   {
; CODE-NEXT:     dim3 k0_dimBlock(32);
; CODE-NEXT:     dim3 k0_dimGrid(4);
; CODE-NEXT:     kernel0 <<<k0_dimGrid, k0_dimBlock>>> (dev_MemRef_A, dev_MemRef_B);
; CODE-NEXT:     cudaCheckKernel();
; CODE-NEXT:   }

; CODE:   cudaCheckReturn(cudaMemcpy(MemRef_B, dev_MemRef_B, (128) * sizeof(float), cudaMemcpyDeviceToHost));

; CODE: # kernel0
; CODE-NEXT: Stmt_for_body__TO__if_end(32 * b0 + t0);

; IR: @polly_initContext

; KERNEL-IR: kernel_0

; REQUIRES: pollyacc

;    void foo(float A[], float B[]) {
;      for (long i = 0; i < 128; i++)
;        if (A[i] == 42)
;          B[i] += 2 * i;
;        else
;          B[i] += 4 * i;
;    }
;
source_filename = "/tmp/test.c"
target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"

define void @foo(float* %A, float* %B) {
entry:
  br label %for.cond

for.cond:                                         ; preds = %for.inc, %entry
  %i.0 = phi i64 [ 0, %entry ], [ %inc, %for.inc ]
  %exitcond = icmp ne i64 %i.0, 128
  br i1 %exitcond, label %for.body, label %for.end

for.body:                                         ; preds = %for.cond
  %arrayidx = getelementptr inbounds float, float* %A, i64 %i.0
  %tmp = load float, float* %arrayidx, align 4
  %cmp1 = fcmp oeq float %tmp, 4.200000e+01
  br i1 %cmp1, label %if.then, label %if.else

if.then:                                          ; preds = %for.body
  %mul = shl nsw i64 %i.0, 1
  %conv = sitofp i64 %mul to float
  %arrayidx2 = getelementptr inbounds float, float* %B, i64 %i.0
  %tmp1 = load float, float* %arrayidx2, align 4
  %add = fadd float %tmp1, %conv
  store float %add, float* %arrayidx2, align 4
  br label %if.end

if.else:                                          ; preds = %for.body
  %mul3 = shl nsw i64 %i.0, 2
  %conv4 = sitofp i64 %mul3 to float
  %arrayidx5 = getelementptr inbounds float, float* %B, i64 %i.0
  %tmp2 = load float, float* %arrayidx5, align 4
  %add6 = fadd float %tmp2, %conv4
  store float %add6, float* %arrayidx5, align 4
  br label %if.end

if.end:                                           ; preds = %if.else, %if.then
  br label %for.inc

for.inc:                                          ; preds = %if.end
  %inc = add nuw nsw i64 %i.0, 1
  br label %for.cond

for.end:                                          ; preds = %for.cond
  ret void
}

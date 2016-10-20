; RUN: opt %loadPolly -polly-codegen-ppcg -polly-acc-dump-code \
; RUN: -disable-output < %s | \
; RUN: FileCheck -check-prefix=CODE %s

; REQUIRES: pollyacc

target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

; This test case took at some point forever to schedule, as the isl scheduler
; seems to have problems if domain constraints appear in the dependences
; provided to the scheduler.

;   /* D := alpha*A*B*C + beta*D */
;   for (i = 0; i < _PB_NI; i++)
;     for (j = 0; j < _PB_NJ; j++)
;       {
;   tmp[i][j] = 0;
;   for (k = 0; k < _PB_NK; ++k)
;     tmp[i][j] += alpha * A[i][k] * B[k][j];
;       }
;   for (i = 0; i < _PB_NI; i++)
;     for (j = 0; j < _PB_NL; j++)
;       {
;   D[i][j] *= beta;
;   for (k = 0; k < _PB_NJ; ++k)
;     D[i][j] += tmp[i][k] * C[k][j];
;       }

; CODE:Code
; CODE-NEXT:====
; CODE-NEXT:# host
; CODE-NEXT: {
; CODE-NEXT:   cudaCheckReturn(cudaMemcpy(dev_MemRef_A, MemRef_A, (4096) * (4096) * sizeof(float), cudaMemcpyHostToDevice));
; CODE-NEXT:   cudaCheckReturn(cudaMemcpy(dev_MemRef_B, MemRef_B, (4096) * (4096) * sizeof(float), cudaMemcpyHostToDevice));
; CODE-NEXT:   cudaCheckReturn(cudaMemcpy(dev_MemRef_D, MemRef_D, (4096) * (4096) * sizeof(float), cudaMemcpyHostToDevice));
; CODE-NEXT:   cudaCheckReturn(cudaMemcpy(dev_MemRef_C, MemRef_C, (4096) * (4096) * sizeof(float), cudaMemcpyHostToDevice));
; CODE-NEXT:   {
; CODE-NEXT:     dim3 k0_dimBlock(16, 32);
; CODE-NEXT:     dim3 k0_dimGrid(128, 128);
; CODE-NEXT:     kernel0 <<<k0_dimGrid, k0_dimBlock>>> (dev_MemRef_tmp, dev_MemRef_A, MemRef_alpha, dev_MemRef_B);
; CODE-NEXT:     cudaCheckKernel();
; CODE-NEXT:   }

; CODE:   {
; CODE-NEXT:     dim3 k1_dimBlock(16, 32);
; CODE-NEXT:     dim3 k1_dimGrid(128, 128);
; CODE-NEXT:     kernel1 <<<k1_dimGrid, k1_dimBlock>>> (dev_MemRef_tmp, dev_MemRef_D, MemRef_beta, dev_MemRef_C);
; CODE-NEXT:     cudaCheckKernel();
; CODE-NEXT:   }

; CODE:   cudaCheckReturn(cudaMemcpy(MemRef_tmp, dev_MemRef_tmp, (4096) * (4096) * sizeof(float), cudaMemcpyDeviceToHost));
; CODE-NEXT:   cudaCheckReturn(cudaMemcpy(MemRef_D, dev_MemRef_D, (4096) * (4096) * sizeof(float), cudaMemcpyDeviceToHost));
; CODE-NEXT: }

; CODE: # kernel0
; CODE-NEXT: for (int c2 = 0; c2 <= 127; c2 += 1)
; CODE-NEXT:   for (int c4 = 0; c4 <= 1; c4 += 1) {
; CODE-NEXT:     if (c2 == 0)
; CODE-NEXT:       Stmt_for_body6(32 * b0 + t0, 32 * b1 + t1 + 16 * c4);
; CODE-NEXT:     for (int c5 = 0; c5 <= 31; c5 += 1)
; CODE-NEXT:       Stmt_for_body11(32 * b0 + t0, 32 * b1 + t1 + 16 * c4, 32 * c2 + c5);
; CODE-NEXT:   }

; CODE: # kernel1
; CODE-NEXT: for (int c2 = 0; c2 <= 127; c2 += 1)
; CODE-NEXT:   for (int c4 = 0; c4 <= 1; c4 += 1) {
; CODE-NEXT:     if (c2 == 0)
; CODE-NEXT:       Stmt_for_body36(32 * b0 + t0, 32 * b1 + t1 + 16 * c4);
; CODE-NEXT:     for (int c5 = 0; c5 <= 31; c5 += 1)
; CODE-NEXT:       Stmt_for_body44(32 * b0 + t0, 32 * b1 + t1 + 16 * c4, 32 * c2 + c5);
; CODE-NEXT:   }



; Function Attrs: argmemonly nounwind
declare void @llvm.lifetime.start(i64, i8* nocapture) #0

; Function Attrs: nounwind uwtable
define internal void @kernel_2mm(i32 %ni, i32 %nj, i32 %nk, i32 %nl, float %alpha, float %beta, [4096 x float]* %tmp, [4096 x float]* %A, [4096 x float]* %B, [4096 x float]* %C, [4096 x float]* %D) #1 {
entry:
  br label %entry.split

entry.split:                                      ; preds = %entry
  br label %for.cond4.preheader

for.cond4.preheader:                              ; preds = %entry.split, %for.inc28
  %indvars.iv19 = phi i64 [ 0, %entry.split ], [ %indvars.iv.next20, %for.inc28 ]
  br label %for.body6

for.cond31.preheader:                             ; preds = %for.inc28
  br label %for.cond34.preheader

for.body6:                                        ; preds = %for.cond4.preheader, %for.inc25
  %indvars.iv16 = phi i64 [ 0, %for.cond4.preheader ], [ %indvars.iv.next17, %for.inc25 ]
  %arrayidx8 = getelementptr inbounds [4096 x float], [4096 x float]* %tmp, i64 %indvars.iv19, i64 %indvars.iv16
  store float 0.000000e+00, float* %arrayidx8, align 4, !tbaa !1
  br label %for.body11

for.body11:                                       ; preds = %for.body6, %for.body11
  %indvars.iv13 = phi i64 [ 0, %for.body6 ], [ %indvars.iv.next14, %for.body11 ]
  %arrayidx15 = getelementptr inbounds [4096 x float], [4096 x float]* %A, i64 %indvars.iv19, i64 %indvars.iv13
  %tmp22 = load float, float* %arrayidx15, align 4, !tbaa !1
  %mul = fmul float %tmp22, %alpha
  %arrayidx19 = getelementptr inbounds [4096 x float], [4096 x float]* %B, i64 %indvars.iv13, i64 %indvars.iv16
  %tmp23 = load float, float* %arrayidx19, align 4, !tbaa !1
  %mul20 = fmul float %mul, %tmp23
  %arrayidx24 = getelementptr inbounds [4096 x float], [4096 x float]* %tmp, i64 %indvars.iv19, i64 %indvars.iv16
  %tmp24 = load float, float* %arrayidx24, align 4, !tbaa !1
  %add = fadd float %tmp24, %mul20
  store float %add, float* %arrayidx24, align 4, !tbaa !1
  %indvars.iv.next14 = add nuw nsw i64 %indvars.iv13, 1
  %exitcond15 = icmp ne i64 %indvars.iv.next14, 4096
  br i1 %exitcond15, label %for.body11, label %for.inc25

for.inc25:                                        ; preds = %for.body11
  %indvars.iv.next17 = add nuw nsw i64 %indvars.iv16, 1
  %exitcond18 = icmp ne i64 %indvars.iv.next17, 4096
  br i1 %exitcond18, label %for.body6, label %for.inc28

for.inc28:                                        ; preds = %for.inc25
  %indvars.iv.next20 = add nuw nsw i64 %indvars.iv19, 1
  %exitcond21 = icmp ne i64 %indvars.iv.next20, 4096
  br i1 %exitcond21, label %for.cond4.preheader, label %for.cond31.preheader

for.cond34.preheader:                             ; preds = %for.cond31.preheader, %for.inc65
  %indvars.iv10 = phi i64 [ 0, %for.cond31.preheader ], [ %indvars.iv.next11, %for.inc65 ]
  br label %for.body36

for.body36:                                       ; preds = %for.cond34.preheader, %for.inc62
  %indvars.iv7 = phi i64 [ 0, %for.cond34.preheader ], [ %indvars.iv.next8, %for.inc62 ]
  %arrayidx40 = getelementptr inbounds [4096 x float], [4096 x float]* %D, i64 %indvars.iv10, i64 %indvars.iv7
  %tmp25 = load float, float* %arrayidx40, align 4, !tbaa !1
  %mul41 = fmul float %tmp25, %beta
  store float %mul41, float* %arrayidx40, align 4, !tbaa !1
  br label %for.body44

for.body44:                                       ; preds = %for.body36, %for.body44
  %indvars.iv = phi i64 [ 0, %for.body36 ], [ %indvars.iv.next, %for.body44 ]
  %arrayidx48 = getelementptr inbounds [4096 x float], [4096 x float]* %tmp, i64 %indvars.iv10, i64 %indvars.iv
  %tmp26 = load float, float* %arrayidx48, align 4, !tbaa !1
  %arrayidx52 = getelementptr inbounds [4096 x float], [4096 x float]* %C, i64 %indvars.iv, i64 %indvars.iv7
  %tmp27 = load float, float* %arrayidx52, align 4, !tbaa !1
  %mul53 = fmul float %tmp26, %tmp27
  %arrayidx57 = getelementptr inbounds [4096 x float], [4096 x float]* %D, i64 %indvars.iv10, i64 %indvars.iv7
  %tmp28 = load float, float* %arrayidx57, align 4, !tbaa !1
  %add58 = fadd float %tmp28, %mul53
  store float %add58, float* %arrayidx57, align 4, !tbaa !1
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  %exitcond = icmp ne i64 %indvars.iv.next, 4096
  br i1 %exitcond, label %for.body44, label %for.inc62

for.inc62:                                        ; preds = %for.body44
  %indvars.iv.next8 = add nuw nsw i64 %indvars.iv7, 1
  %exitcond9 = icmp ne i64 %indvars.iv.next8, 4096
  br i1 %exitcond9, label %for.body36, label %for.inc65

for.inc65:                                        ; preds = %for.inc62
  %indvars.iv.next11 = add nuw nsw i64 %indvars.iv10, 1
  %exitcond12 = icmp ne i64 %indvars.iv.next11, 4096
  br i1 %exitcond12, label %for.cond34.preheader, label %for.end67

for.end67:                                        ; preds = %for.inc65
  ret void
}

; Function Attrs: argmemonly nounwind
declare void @llvm.lifetime.end(i64, i8* nocapture) #0

attributes #0 = { argmemonly nounwind }
attributes #1 = { nounwind uwtable "disable-tail-calls"="false" "less-precise-fpmad"="false" "no-frame-pointer-elim"="false" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+fxsr,+mmx,+sse,+sse2,+x87" "unsafe-fp-math"="false" "use-soft-float"="false" }

!llvm.ident = !{!0}

!0 = !{!"clang version 3.9.0 (trunk 275267) (llvm/trunk 275268)"}
!1 = !{!2, !2, i64 0}
!2 = !{!"float", !3, i64 0}
!3 = !{!"omnipotent char", !4, i64 0}
!4 = !{!"Simple C/C++ TBAA"}

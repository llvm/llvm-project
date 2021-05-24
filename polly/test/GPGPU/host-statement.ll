; RUN: opt %loadPolly -polly-codegen-ppcg -polly-acc-dump-code \
; RUN: -polly-invariant-load-hoisting=false \
; RUN: -disable-output < %s | \
; RUN: FileCheck -check-prefix=CODE %s

; RUN: opt %loadPolly -polly-codegen-ppcg -polly-acc-dump-kernel-ir \
; RUN: -polly-invariant-load-hoisting=false \
; RUN: -disable-output < %s | \
; RUN: FileCheck -check-prefix=KERNEL-IR %s

; REQUIRES: pollyacc

target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

declare void @llvm.lifetime.start(i64, i8* nocapture) #0

; This test case tests that we can correctly handle a ScopStmt that is
; scheduled on the host, instead of within a kernel.

; CODE:        cudaCheckReturn(cudaMemcpy(dev_MemRef_A, MemRef_A, (512) * (512) * sizeof(double), cudaMemcpyHostToDevice));
; CODE-NEXT:   cudaCheckReturn(cudaMemcpy(dev_MemRef_R, MemRef_R, (p_0 + 1) * (512) * sizeof(double), cudaMemcpyHostToDevice));
; CODE-NEXT:   cudaCheckReturn(cudaMemcpy(dev_MemRef_Q, MemRef_Q, (512) * (512) * sizeof(double), cudaMemcpyHostToDevice));
; CODE-NEXT:   {
; CODE-NEXT:     dim3 k0_dimBlock(32);
; CODE-NEXT:     dim3 k0_dimGrid(16);
; CODE-NEXT:     kernel0 <<<k0_dimGrid, k0_dimBlock>>> (dev_MemRef_A, dev_MemRef_R, dev_MemRef_Q, p_0, p_1);
; CODE-NEXT:     cudaCheckKernel();
; CODE-NEXT:   }

; CODE:   if (p_0 <= 510 && p_1 <= 510) {
; CODE-NEXT:     {
; CODE-NEXT:       dim3 k1_dimBlock(32);
; CODE-NEXT:       dim3 k1_dimGrid(p_1 <= -1048034 ? 32768 : -p_1 + floord(31 * p_1 + 30, 32) + 16);
; CODE-NEXT:       kernel1 <<<k1_dimGrid, k1_dimBlock>>> (dev_MemRef_A, dev_MemRef_R, dev_MemRef_Q, p_0, p_1);
; CODE-NEXT:       cudaCheckKernel();
; CODE-NEXT:     }

; CODE:     {
; CODE-NEXT:       dim3 k2_dimBlock(16, 32);
; CODE-NEXT:       dim3 k2_dimGrid(16, p_1 <= -7650 ? 256 : -p_1 + floord(31 * p_1 + 30, 32) + 16);
; CODE-NEXT:       kernel2 <<<k2_dimGrid, k2_dimBlock>>> (dev_MemRef_A, dev_MemRef_R, dev_MemRef_Q, p_0, p_1);
; CODE-NEXT:       cudaCheckKernel();
; CODE-NEXT:     }

; CODE:   }
; CODE-NEXT:   cudaCheckReturn(cudaMemcpy(MemRef_A, dev_MemRef_A, (512) * (512) * sizeof(double), cudaMemcpyDeviceToHost));
; CODE-NEXT:   cudaCheckReturn(cudaMemcpy(MemRef_R, dev_MemRef_R, (p_0 + 1) * (512) * sizeof(double), cudaMemcpyDeviceToHost));
; CODE-NEXT:   cudaCheckReturn(cudaMemcpy(MemRef_Q, dev_MemRef_Q, (512) * (512) * sizeof(double), cudaMemcpyDeviceToHost));
; CODE-NEXT:     Stmt_for_cond33_preheader_last();

; CODE: }

; CODE: # kernel0
; CODE-NEXT: Stmt_for_body16(32 * b0 + t0);

; CODE: # kernel1
; CODE-NEXT: for (int c0 = 0; c0 <= (-p_1 - 32 * b0 + 510) / 1048576; c0 += 1)
; CODE-NEXT:   for (int c1 = 0; c1 <= 15; c1 += 1) {
; CODE-NEXT:     if (p_1 + 32 * b0 + t0 + 1048576 * c0 <= 510 && c1 == 0)
; CODE-NEXT:       Stmt_for_body35(32 * b0 + t0 + 1048576 * c0);
; CODE-NEXT:     if (p_1 + 32 * b0 + t0 + 1048576 * c0 <= 510)
; CODE-NEXT:       for (int c3 = 0; c3 <= 31; c3 += 1)
; CODE-NEXT:         Stmt_for_body42(32 * b0 + t0 + 1048576 * c0, 32 * c1 + c3);
; CODE-NEXT:     sync0();
; CODE-NEXT:   }

; CODE: # kernel2
; CODE-NEXT: for (int c0 = 0; c0 <= (-p_1 - 32 * b0 + 510) / 8192; c0 += 1)
; CODE-NEXT:   if (p_1 + 32 * b0 + t0 + 8192 * c0 <= 510)
; CODE-NEXT:     for (int c3 = 0; c3 <= 1; c3 += 1)
; CODE-NEXT:       Stmt_for_body62(32 * b0 + t0 + 8192 * c0, 32 * b1 + t1 + 16 * c3);

; KERNEL-IR: call void @llvm.nvvm.barrier0()

; Function Attrs: nounwind uwtable
define internal void @kernel_gramschmidt(i32 %ni, i32 %nj, [512 x double]* %A, [512 x double]* %R, [512 x double]* %Q) #1 {
entry:
  br label %entry.split

entry.split:                                      ; preds = %entry
  br label %for.cond1.preheader

for.cond1.preheader:                              ; preds = %entry.split, %for.inc86
  %indvars.iv24 = phi i64 [ 0, %entry.split ], [ %indvars.iv.next25, %for.inc86 ]
  %indvars.iv19 = phi i64 [ 1, %entry.split ], [ %indvars.iv.next20, %for.inc86 ]
  br label %for.inc

for.inc:                                          ; preds = %for.cond1.preheader, %for.inc
  %indvars.iv = phi i64 [ 0, %for.cond1.preheader ], [ %indvars.iv.next, %for.inc ]
  %nrm.02 = phi double [ 0.000000e+00, %for.cond1.preheader ], [ %add, %for.inc ]
  %arrayidx5 = getelementptr inbounds [512 x double], [512 x double]* %A, i64 %indvars.iv, i64 %indvars.iv24
  %tmp = load double, double* %arrayidx5, align 8, !tbaa !1
  %arrayidx9 = getelementptr inbounds [512 x double], [512 x double]* %A, i64 %indvars.iv, i64 %indvars.iv24
  %tmp27 = load double, double* %arrayidx9, align 8, !tbaa !1
  %mul = fmul double %tmp, %tmp27
  %add = fadd double %nrm.02, %mul
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  %exitcond = icmp ne i64 %indvars.iv.next, 512
  br i1 %exitcond, label %for.inc, label %for.end

for.end:                                          ; preds = %for.inc
  %add.lcssa = phi double [ %add, %for.inc ]
  %call = tail call double @sqrt(double %add.lcssa) #2
  %arrayidx13 = getelementptr inbounds [512 x double], [512 x double]* %R, i64 %indvars.iv24, i64 %indvars.iv24
  store double %call, double* %arrayidx13, align 8, !tbaa !1
  br label %for.body16

for.cond33.preheader:                             ; preds = %for.body16
  %indvars.iv.next25 = add nuw nsw i64 %indvars.iv24, 1
  %cmp347 = icmp slt i64 %indvars.iv.next25, 512
  br i1 %cmp347, label %for.body35.lr.ph, label %for.inc86

for.body35.lr.ph:                                 ; preds = %for.cond33.preheader
  br label %for.body35

for.body16:                                       ; preds = %for.end, %for.body16
  %indvars.iv10 = phi i64 [ 0, %for.end ], [ %indvars.iv.next11, %for.body16 ]
  %arrayidx20 = getelementptr inbounds [512 x double], [512 x double]* %A, i64 %indvars.iv10, i64 %indvars.iv24
  %tmp28 = load double, double* %arrayidx20, align 8, !tbaa !1
  %arrayidx24 = getelementptr inbounds [512 x double], [512 x double]* %R, i64 %indvars.iv24, i64 %indvars.iv24
  %tmp29 = load double, double* %arrayidx24, align 8, !tbaa !1
  %div = fdiv double %tmp28, %tmp29
  %arrayidx28 = getelementptr inbounds [512 x double], [512 x double]* %Q, i64 %indvars.iv10, i64 %indvars.iv24
  store double %div, double* %arrayidx28, align 8, !tbaa !1
  %indvars.iv.next11 = add nuw nsw i64 %indvars.iv10, 1
  %exitcond12 = icmp ne i64 %indvars.iv.next11, 512
  br i1 %exitcond12, label %for.body16, label %for.cond33.preheader

for.cond33.loopexit:                              ; preds = %for.body62
  %indvars.iv.next22 = add nuw nsw i64 %indvars.iv21, 1
  %lftr.wideiv = trunc i64 %indvars.iv.next22 to i32
  %exitcond23 = icmp ne i32 %lftr.wideiv, 512
  br i1 %exitcond23, label %for.body35, label %for.cond33.for.inc86_crit_edge

for.body35:                                       ; preds = %for.body35.lr.ph, %for.cond33.loopexit
  %indvars.iv21 = phi i64 [ %indvars.iv19, %for.body35.lr.ph ], [ %indvars.iv.next22, %for.cond33.loopexit ]
  %arrayidx39 = getelementptr inbounds [512 x double], [512 x double]* %R, i64 %indvars.iv24, i64 %indvars.iv21
  store double 0.000000e+00, double* %arrayidx39, align 8, !tbaa !1
  br label %for.body42

for.cond60.preheader:                             ; preds = %for.body42
  br label %for.body62

for.body42:                                       ; preds = %for.body35, %for.body42
  %indvars.iv13 = phi i64 [ 0, %for.body35 ], [ %indvars.iv.next14, %for.body42 ]
  %arrayidx46 = getelementptr inbounds [512 x double], [512 x double]* %Q, i64 %indvars.iv13, i64 %indvars.iv24
  %tmp30 = load double, double* %arrayidx46, align 8, !tbaa !1
  %arrayidx50 = getelementptr inbounds [512 x double], [512 x double]* %A, i64 %indvars.iv13, i64 %indvars.iv21
  %tmp31 = load double, double* %arrayidx50, align 8, !tbaa !1
  %mul51 = fmul double %tmp30, %tmp31
  %arrayidx55 = getelementptr inbounds [512 x double], [512 x double]* %R, i64 %indvars.iv24, i64 %indvars.iv21
  %tmp32 = load double, double* %arrayidx55, align 8, !tbaa !1
  %add56 = fadd double %tmp32, %mul51
  store double %add56, double* %arrayidx55, align 8, !tbaa !1
  %indvars.iv.next14 = add nuw nsw i64 %indvars.iv13, 1
  %exitcond15 = icmp ne i64 %indvars.iv.next14, 512
  br i1 %exitcond15, label %for.body42, label %for.cond60.preheader

for.body62:                                       ; preds = %for.cond60.preheader, %for.body62
  %indvars.iv16 = phi i64 [ 0, %for.cond60.preheader ], [ %indvars.iv.next17, %for.body62 ]
  %arrayidx66 = getelementptr inbounds [512 x double], [512 x double]* %A, i64 %indvars.iv16, i64 %indvars.iv21
  %tmp33 = load double, double* %arrayidx66, align 8, !tbaa !1
  %arrayidx70 = getelementptr inbounds [512 x double], [512 x double]* %Q, i64 %indvars.iv16, i64 %indvars.iv24
  %tmp34 = load double, double* %arrayidx70, align 8, !tbaa !1
  %arrayidx74 = getelementptr inbounds [512 x double], [512 x double]* %R, i64 %indvars.iv24, i64 %indvars.iv21
  %tmp35 = load double, double* %arrayidx74, align 8, !tbaa !1
  %mul75 = fmul double %tmp34, %tmp35
  %sub = fsub double %tmp33, %mul75
  %arrayidx79 = getelementptr inbounds [512 x double], [512 x double]* %A, i64 %indvars.iv16, i64 %indvars.iv21
  store double %sub, double* %arrayidx79, align 8, !tbaa !1
  %indvars.iv.next17 = add nuw nsw i64 %indvars.iv16, 1
  %exitcond18 = icmp ne i64 %indvars.iv.next17, 512
  br i1 %exitcond18, label %for.body62, label %for.cond33.loopexit

for.cond33.for.inc86_crit_edge:                   ; preds = %for.cond33.loopexit
  br label %for.inc86

for.inc86:                                        ; preds = %for.cond33.for.inc86_crit_edge, %for.cond33.preheader
  %indvars.iv.next20 = add nuw nsw i64 %indvars.iv19, 1
  %exitcond26 = icmp ne i64 %indvars.iv.next25, 512
  br i1 %exitcond26, label %for.cond1.preheader, label %for.end88

for.end88:                                        ; preds = %for.inc86
  ret void
}

; Function Attrs: argmemonly nounwind
declare void @llvm.lifetime.end(i64, i8* nocapture) #0

; Function Attrs: nounwind
declare double @sqrt(double) #2

attributes #0 = { argmemonly nounwind }
attributes #1 = { nounwind uwtable "disable-tail-calls"="false" "less-precise-fpmad"="false" "frame-pointer"="none" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+fxsr,+mmx,+sse,+sse2,+x87" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #2 = { nounwind }

!llvm.ident = !{!0}

!0 = !{!"clang version 3.9.0 (trunk 275267) (llvm/trunk 275268)"}
!1 = !{!2, !2, i64 0}
!2 = !{!"double", !3, i64 0}
!3 = !{!"omnipotent char", !4, i64 0}
!4 = !{!"Simple C/C++ TBAA"}

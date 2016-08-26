; RUN: opt %loadPolly -polly-codegen-ppcg -polly-acc-dump-code \
; RUN: -disable-output < %s | \
; RUN: FileCheck -check-prefix=CODE %s

; RUN: opt %loadPolly -polly-codegen-ppcg -S < %s | \
; RUN: FileCheck %s -check-prefix=IR

; RUN: opt %loadPolly -polly-codegen-ppcg -polly-acc-dump-kernel-ir \
; RUN: -disable-output < %s | \
; RUN: FileCheck %s -check-prefix=KERNEL-IR

; REQUIRES: pollyacc

target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

; CODE: # host
; CODE-NEXT: {
; CODE-NEXT:   cudaCheckReturn(cudaMemcpy(dev_MemRef_out_l_055__phi, &MemRef_out_l_055__phi, sizeof(i32), cudaMemcpyHostToDevice));
; CODE-NEXT:   {
; CODE-NEXT:     dim3 k0_dimBlock(32);
; CODE-NEXT:     dim3 k0_dimGrid(2);
; CODE-NEXT:     kernel0 <<<k0_dimGrid, k0_dimBlock>>> (dev_MemRef_out_l_055__phi, dev_MemRef_out_l_055, dev_MemRef_c);
; CODE-NEXT:     cudaCheckKernel();
; CODE-NEXT:   }

; CODE:   cudaCheckReturn(cudaMemcpy(&MemRef_out_l_055__phi, dev_MemRef_out_l_055__phi, sizeof(i32), cudaMemcpyDeviceToHost));
; CODE-NEXT:   cudaCheckReturn(cudaMemcpy(&MemRef_out_l_055, dev_MemRef_out_l_055, sizeof(i32), cudaMemcpyDeviceToHost));
; CODE-NEXT:   cudaCheckReturn(cudaMemcpy(MemRef_c, dev_MemRef_c, (50) * sizeof(i32), cudaMemcpyDeviceToHost));
; CODE-NEXT: }

; CODE: # kernel0
; CODE-NEXT: if (32 * b0 + t0 <= 48) {
; CODE-NEXT:   if (b0 == 1 && t0 == 16)
; CODE-NEXT:     Stmt_for_cond1_preheader(0);
; CODE-NEXT:   Stmt_for_body17(0, 32 * b0 + t0);
; CODE-NEXT:   if (b0 == 1 && t0 == 16)
; CODE-NEXT:     Stmt_for_cond15_for_cond12_loopexit_crit_edge(0);
; CODE-NEXT: }

; IR:      %1 = bitcast i32* %out_l.055.phiops to i8*
; IR-NEXT: call void @polly_copyFromHostToDevice(i8* %1, i8* %p_dev_array_MemRef_out_l_055__phi, i64 4)

; IR:      %14 = bitcast i32* %out_l.055.phiops to i8*
; IR-NEXT: call void @polly_copyFromDeviceToHost(i8* %p_dev_array_MemRef_out_l_055__phi, i8* %14, i64 4)
; IR-NEXT: %15 = bitcast i32* %out_l.055.s2a to i8*
; IR-NEXT: call void @polly_copyFromDeviceToHost(i8* %p_dev_array_MemRef_out_l_055, i8* %15, i64 4)

; KERNEL-IR: entry:
; KERNEL-IR-NEXT:   %out_l.055.s2a = alloca i32
; KERNEL-IR-NEXT:   %out_l.055.phiops = alloca i32
; KERNEL-IR-NEXT:   %1 = bitcast i8* %MemRef_out_l_055__phi to i32*
; KERNEL-IR-NEXT:   %2 = load i32, i32* %1
; KERNEL-IR-NEXT:   store i32 %2, i32* %out_l.055.phiops
; KERNEL-IR-NEXT:   %3 = bitcast i8* %MemRef_out_l_055 to i32*
; KERNEL-IR-NEXT:   %4 = load i32, i32* %3
; KERNEL-IR-NEXT:   store i32 %4, i32* %out_l.055.s2a


define void @kernel_dynprog([50 x i32]* %c) {
entry:
  %arrayidx77 = getelementptr inbounds [50 x i32], [50 x i32]* %c, i64 0, i64 49
  br label %for.cond1.preheader

for.cond1.preheader:                              ; preds = %for.cond15.for.cond12.loopexit_crit_edge, %entry
  %out_l.055 = phi i32 [ 0, %entry ], [ %add78, %for.cond15.for.cond12.loopexit_crit_edge ]
  %iter.054 = phi i32 [ 0, %entry ], [ %inc80, %for.cond15.for.cond12.loopexit_crit_edge ]
  br label %for.body17

for.cond15.for.cond12.loopexit_crit_edge:         ; preds = %for.body17
  %tmp = load i32, i32* %arrayidx77, align 4
  %add78 = add nsw i32 %tmp, %out_l.055
  %inc80 = add nuw nsw i32 %iter.054, 1
  br i1 false, label %for.cond1.preheader, label %for.end81

for.body17:                                       ; preds = %for.body17, %for.cond1.preheader
  %indvars.iv71 = phi i64 [ 1, %for.cond1.preheader ], [ %indvars.iv.next72, %for.body17 ]
  %arrayidx69 = getelementptr inbounds [50 x i32], [50 x i32]* %c, i64 0, i64 %indvars.iv71
  store i32 undef, i32* %arrayidx69, align 4
  %indvars.iv.next72 = add nuw nsw i64 %indvars.iv71, 1
  %lftr.wideiv74 = trunc i64 %indvars.iv.next72 to i32
  %exitcond75 = icmp ne i32 %lftr.wideiv74, 50
  br i1 %exitcond75, label %for.body17, label %for.cond15.for.cond12.loopexit_crit_edge

for.end81:                                        ; preds = %for.cond15.for.cond12.loopexit_crit_edge
  ret void
}

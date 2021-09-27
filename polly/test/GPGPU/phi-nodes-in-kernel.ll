; RUN: opt %loadPolly -polly-codegen-ppcg -polly-acc-dump-code \
; RUN: -disable-output < %s | \
; RUN: FileCheck -check-prefix=CODE %s

; RUN: opt %loadPolly -polly-codegen-ppcg -S < %s | \
; RUN: FileCheck %s -check-prefix=IR

; RUN: opt %loadPolly -polly-codegen-ppcg -polly-acc-dump-kernel-ir \
; RUN: -disable-output < %s | \
; RUN: FileCheck %s -check-prefix=KERNEL-IR

; REQUIRES: pollyacc

; Approximate C source:
; void kernel_dynprog(int c[50]) {
;     int iter = 0;
;     int outl = 0;
;
;      while(1) {
;         for(int indvar = 1 ; indvar <= 49; indvar++) {
;             c[indvar] = undef;
;         }
;         add78 = c[49] + outl;
;         inc80 = iter + 1;
;
;         if (true) break;
;
;         outl = add78;
;         iter = inc80;
;      }
;}
target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

; CODE:       cudaCheckReturn(cudaMalloc((void **) &dev_MemRef_c, (50) * sizeof(i32)));

; CODE:       {
; CODE-NEXT:    dim3 k0_dimBlock(32);
; CODE-NEXT:    dim3 k0_dimGrid(2);
; CODE-NEXT:    kernel0 <<<k0_dimGrid, k0_dimBlock>>> (dev_MemRef_c);
; CODE-NEXT:    cudaCheckKernel();
; CODE-NEXT:  }

; CODE:       cudaCheckReturn(cudaMemcpy(MemRef_c, dev_MemRef_c, (50) * sizeof(i32), cudaMemcpyDeviceToHost));
; CODE-NEXT:  cudaCheckReturn(cudaFree(dev_MemRef_c));

; CODE: # kernel0
; CODE-NEXT: if (32 * b0 + t0 <= 48)
; CODE-NEXT:     Stmt_for_body17(0, 32 * b0 + t0);

; IR-LABEL: call void @polly_freeKernel
; IR:       [[REGC:%.+]] =   bitcast i32* %{{[0-9]+}} to i8*
; IR-NEXT:  call void @polly_copyFromDeviceToHost(i8* %p_dev_array_MemRef_c, i8* [[REGC]], i64 196)

; KERNEL-IR: define ptx_kernel void @FUNC_kernel_dynprog_SCOP_0_KERNEL_0(i8 addrspace(1)* %MemRef_c) #0 {
; KERNEL-IR: %polly.access.MemRef_c = getelementptr i32, i32 addrspace(1)* %polly.access.cast.MemRef_c, i64 %9
; KERNEL-IR-NEXT: store i32 422, i32 addrspace(1)* %polly.access.MemRef_c, align 4

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
  store i32 422, i32* %arrayidx69, align 4
  %indvars.iv.next72 = add nuw nsw i64 %indvars.iv71, 1
  %lftr.wideiv74 = trunc i64 %indvars.iv.next72 to i32
  %exitcond75 = icmp ne i32 %lftr.wideiv74, 50
  br i1 %exitcond75, label %for.body17, label %for.cond15.for.cond12.loopexit_crit_edge

for.end81:                                        ; preds = %for.cond15.for.cond12.loopexit_crit_edge
  ret void
}

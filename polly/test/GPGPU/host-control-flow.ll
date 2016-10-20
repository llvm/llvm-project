; RUN: opt %loadPolly -polly-codegen-ppcg -disable-output \
; RUN: -polly-acc-dump-code < %s | FileCheck %s -check-prefix=CODE

; RUN: opt %loadPolly -polly-codegen-ppcg -disable-output \
; RUN: -polly-acc-dump-kernel-ir < %s | FileCheck %s -check-prefix=KERNEL-IR

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
; ...
; IR:  store i64 %polly.indvar, i64* %polly_launch_0_param_1
; IR-NEXT:  [[REGA:%.+]] = getelementptr [2 x i8*], [2 x i8*]* %polly_launch_0_params, i64 0, i64 1
; IR-NEXT:  [[REGB:%.+]] = bitcast i64* %polly_launch_0_param_1 to i8*
; IR-NEXT:  store i8* [[REGB]], i8** [[REGA]]
; IR: call i8* @polly_getKernel
; ...
; IR: call void @polly_freeKernel
; IR-NEXT:   %polly.indvar_next = add nsw i64 %polly.indvar, 1
; IR-NEXT:   %polly.loop_cond = icmp sle i64 %polly.indvar, 98
; IR-NEXT:   br i1 %polly.loop_cond, label %polly.loop_header, label %polly.loop_exit

; KERNEL-IR: define ptx_kernel void @kernel_0(i8* %MemRef_A, i64 %c0)
; KERNEL-IR-LABEL: entry:
; KERNEL-IR-NEXT:   %0 = call i32 @llvm.nvvm.read.ptx.sreg.ctaid.x()
; KERNEL-IR-NEXT:   %b0 = zext i32 %0 to i64
; KERNEL-IR-NEXT:   %1 = call i32 @llvm.nvvm.read.ptx.sreg.tid.x()
; KERNEL-IR-NEXT:   %t0 = zext i32 %1 to i64
; KERNEL-IR-NEXT:   br label %polly.cond

; KERNEL-IR-LABEL: polly.cond:                                       ; preds = %entry
; KERNEL-IR-NEXT:   %2 = mul nsw i64 32, %b0
; KERNEL-IR-NEXT:   %3 = add nsw i64 %2, %t0
; KERNEL-IR-NEXT:   %4 = icmp sle i64 %3, 97
; KERNEL-IR-NEXT:   br i1 %4, label %polly.then, label %polly.else

; KERNEL-IR-LABEL: polly.merge:                                      ; preds = %polly.else, %polly.stmt.for.body3
; KERNEL-IR-NEXT:   ret void

; KERNEL-IR-LABEL: polly.then:                                       ; preds = %polly.cond
; KERNEL-IR-NEXT:   %5 = mul nsw i64 32, %b0
; KERNEL-IR-NEXT:   %6 = add nsw i64 %5, %t0
; KERNEL-IR-NEXT:   br label %polly.stmt.for.body3

; KERNEL-IR-LABEL: polly.stmt.for.body3:                             ; preds = %polly.then
; KERNEL-IR-NEXT:   %polly.access.cast.MemRef_A = bitcast i8* %MemRef_A to float*
; KERNEL-IR-NEXT:   %pexp.pdiv_r = urem i64 %c0, 2
; KERNEL-IR-NEXT:   %polly.access.mul.MemRef_A = mul nsw i64 %pexp.pdiv_r, 100
; KERNEL-IR-NEXT:   %7 = mul nsw i64 32, %b0
; KERNEL-IR-NEXT:   %8 = add nsw i64 %7, %t0
; KERNEL-IR-NEXT:   %polly.access.add.MemRef_A = add nsw i64 %polly.access.mul.MemRef_A, %8
; KERNEL-IR-NEXT:   %polly.access.MemRef_A = getelementptr float, float* %polly.access.cast.MemRef_A, i64 %polly.access.add.MemRef_A
; KERNEL-IR-NEXT:   %tmp_p_scalar_ = load float, float* %polly.access.MemRef_A, align 4
; KERNEL-IR-NEXT:   %polly.access.cast.MemRef_A1 = bitcast i8* %MemRef_A to float*
; KERNEL-IR-NEXT:   %pexp.pdiv_r2 = urem i64 %c0, 2
; KERNEL-IR-NEXT:   %polly.access.mul.MemRef_A3 = mul nsw i64 %pexp.pdiv_r2, 100
; KERNEL-IR-NEXT:   %9 = mul nsw i64 32, %b0
; KERNEL-IR-NEXT:   %10 = add nsw i64 %9, %t0
; KERNEL-IR-NEXT:   %11 = add nsw i64 %10, 1
; KERNEL-IR-NEXT:   %polly.access.add.MemRef_A4 = add nsw i64 %polly.access.mul.MemRef_A3, %11
; KERNEL-IR-NEXT:   %polly.access.MemRef_A5 = getelementptr float, float* %polly.access.cast.MemRef_A1, i64 %polly.access.add.MemRef_A4
; KERNEL-IR-NEXT:   %tmp2_p_scalar_ = load float, float* %polly.access.MemRef_A5, align 4
; KERNEL-IR-NEXT:   %p_add = fadd float %tmp_p_scalar_, %tmp2_p_scalar_
; KERNEL-IR-NEXT:   %polly.access.cast.MemRef_A6 = bitcast i8* %MemRef_A to float*
; KERNEL-IR-NEXT:   %pexp.pdiv_r7 = urem i64 %c0, 2
; KERNEL-IR-NEXT:   %polly.access.mul.MemRef_A8 = mul nsw i64 %pexp.pdiv_r7, 100
; KERNEL-IR-NEXT:   %12 = mul nsw i64 32, %b0
; KERNEL-IR-NEXT:   %13 = add nsw i64 %12, %t0
; KERNEL-IR-NEXT:   %14 = add nsw i64 %13, 2
; KERNEL-IR-NEXT:   %polly.access.add.MemRef_A9 = add nsw i64 %polly.access.mul.MemRef_A8, %14
; KERNEL-IR-NEXT:   %polly.access.MemRef_A10 = getelementptr float, float* %polly.access.cast.MemRef_A6, i64 %polly.access.add.MemRef_A9
; KERNEL-IR-NEXT:   %tmp3_p_scalar_ = load float, float* %polly.access.MemRef_A10, align 4
; KERNEL-IR-NEXT:   %p_add12 = fadd float %p_add, %tmp3_p_scalar_
; KERNEL-IR-NEXT:   %polly.access.cast.MemRef_A11 = bitcast i8* %MemRef_A to float*
; KERNEL-IR-NEXT:   %15 = add nsw i64 %c0, 1
; KERNEL-IR-NEXT:   %pexp.pdiv_r12 = urem i64 %15, 2
; KERNEL-IR-NEXT:   %polly.access.mul.MemRef_A13 = mul nsw i64 %pexp.pdiv_r12, 100
; KERNEL-IR-NEXT:   %16 = mul nsw i64 32, %b0
; KERNEL-IR-NEXT:   %17 = add nsw i64 %16, %t0
; KERNEL-IR-NEXT:   %18 = add nsw i64 %17, 1
; KERNEL-IR-NEXT:   %polly.access.add.MemRef_A14 = add nsw i64 %polly.access.mul.MemRef_A13, %18
; KERNEL-IR-NEXT:   %polly.access.MemRef_A15 = getelementptr float, float* %polly.access.cast.MemRef_A11, i64 %polly.access.add.MemRef_A14
; KERNEL-IR-NEXT:   %tmp4_p_scalar_ = load float, float* %polly.access.MemRef_A15, align 4
; KERNEL-IR-NEXT:   %p_add17 = fadd float %tmp4_p_scalar_, %p_add12
; KERNEL-IR-NEXT:   %polly.access.cast.MemRef_A16 = bitcast i8* %MemRef_A to float*
; KERNEL-IR-NEXT:   %19 = add nsw i64 %c0, 1
; KERNEL-IR-NEXT:   %pexp.pdiv_r17 = urem i64 %19, 2
; KERNEL-IR-NEXT:   %polly.access.mul.MemRef_A18 = mul nsw i64 %pexp.pdiv_r17, 100
; KERNEL-IR-NEXT:   %20 = mul nsw i64 32, %b0
; KERNEL-IR-NEXT:   %21 = add nsw i64 %20, %t0
; KERNEL-IR-NEXT:   %22 = add nsw i64 %21, 1
; KERNEL-IR-NEXT:   %polly.access.add.MemRef_A19 = add nsw i64 %polly.access.mul.MemRef_A18, %22
; KERNEL-IR-NEXT:   %polly.access.MemRef_A20 = getelementptr float, float* %polly.access.cast.MemRef_A16, i64 %polly.access.add.MemRef_A19
; KERNEL-IR-NEXT:   store float %p_add17, float* %polly.access.MemRef_A20, align 4
; KERNEL-IR-NEXT:   br label %polly.merge

; KERNEL-IR-LABEL: polly.else:                                       ; preds = %polly.cond
; KERNEL-IR-NEXT:   br label %polly.merge
; KERNEL-IR-NEXT: }

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

; RUN: opt %loadPolly -polly-codegen-ppcg -polly-acc-dump-code \
; RUN: -polly-invariant-load-hoisting \
; RUN: -S -polly-acc-codegen-managed-memory < %s | FileCheck %s

; RUN: opt %loadPolly -polly-codegen-ppcg -polly-acc-dump-code \
; RUN: -polly-invariant-load-hoisting \
; RUN: -S -polly-acc-codegen-managed-memory -disable-output \
; RUN: -polly-acc-dump-code < %s | FileCheck %s -check-prefix=CODE

; REQUIRES: pollyacc

; CHECK: @polly_launchKernel
; CHECK: @polly_launchKernel
; CHECK: @polly_launchKernel
; CHECK: @polly_launchKernel
; CHECK: @polly_launchKernel
; CHECK-NOT: @polly_launchKernel


; CODE:  if (p_0_loaded_from___data_runcontrol_MOD_lmulti_layer == 0) {
; CODE-NEXT:    {
; CODE-NEXT:      dim3 k0_dimBlock;
; CODE-NEXT:      dim3 k0_dimGrid;
; CODE-NEXT:      kernel0 <<<k0_dimGrid, k0_dimBlock>>> (dev_MemRef__pn__phi, p_0_loaded_from___data_runcontrol_MOD_lmulti_layer);
; CODE-NEXT:      cudaCheckKernel();
; CODE-NEXT:    }

; CODE:  } else {
; CODE-NEXT:    {
; CODE-NEXT:      dim3 k1_dimBlock;
; CODE-NEXT:      dim3 k1_dimGrid;
; CODE-NEXT:      kernel1 <<<k1_dimGrid, k1_dimBlock>>> (dev_MemRef__pn__phi, p_0_loaded_from___data_runcontrol_MOD_lmulti_layer);
; CODE-NEXT:      cudaCheckKernel();
; CODE-NEXT:    }

; CHECK that this program is correctly code generated and does not result in
; 'instruction does not dominate use' errors. At an earlier point, such errors
; have been generated as the preparation of the managed memory pointers was
; performed right before kernel0, which does not dominate all other kernels.
; Now the preparation is performed at the very beginning of the scop.

source_filename = "bugpoint-output-c78f41e.bc"
target datalayout = "e-p:64:64:64-S128-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f16:16:16-f32:32:32-f64:64:64-f128:128:128-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128-n8:16:32:64"
target triple = "x86_64-unknown-linux-gnu"

@__data_radiation_MOD_rad_csalbw = external global [10 x double], align 32
@__data_radiation_MOD_coai = external global [168 x double], align 32
@__data_runcontrol_MOD_lmulti_layer = external global i32

; Function Attrs: nounwind uwtable
define void @__radiation_interface_MOD_radiation_init() #0 {
entry:
  br label %"94"

"94":                                             ; preds = %"97", %entry
  br label %"95"

"95":                                             ; preds = %"95", %"94"
  br i1 undef, label %"97", label %"95"

"97":                                             ; preds = %"95"
  br i1 undef, label %"99", label %"94"

"99":                                             ; preds = %"97"
  br label %"102"

"102":                                            ; preds = %"102", %"99"
  %indvars.iv17 = phi i64 [ %indvars.iv.next18, %"102" ], [ 1, %"99" ]
  %0 = getelementptr [168 x double], [168 x double]* @__data_radiation_MOD_coai, i64 0, i64 0
  store double 1.000000e+00, double* %0, align 8
  %1 = icmp eq i64 %indvars.iv17, 3
  %indvars.iv.next18 = add nuw nsw i64 %indvars.iv17, 1
  br i1 %1, label %"110", label %"102"

"110":                                            ; preds = %"102"
  %2 = load i32, i32* @__data_runcontrol_MOD_lmulti_layer, align 4, !range !0
  %3 = icmp eq i32 %2, 0
  br i1 %3, label %"112", label %"111"

"111":                                            ; preds = %"110"
  br label %"115"

"112":                                            ; preds = %"110"
  br label %"115"

"115":                                            ; preds = %"112", %"111"
  %.pn = phi double [ undef, %"112" ], [ undef, %"111" ]
  %4 = fdiv double 1.000000e+00, %.pn
  br label %"116"

"116":                                            ; preds = %"116", %"115"
  %indvars.iv = phi i64 [ %indvars.iv.next, %"116" ], [ 1, %"115" ]
  %5 = add nsw i64 %indvars.iv, -1
  %6 = fmul double %4, undef
  %7 = getelementptr [10 x double], [10 x double]* @__data_radiation_MOD_rad_csalbw, i64 0, i64 %5
  store double %6, double* %7, align 8
  %8 = icmp eq i64 %indvars.iv, 10
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  br i1 %8, label %return, label %"116"

return:                                           ; preds = %"116"
  ret void
}

attributes #0 = { nounwind uwtable }

!0 = !{i32 0, i32 2}

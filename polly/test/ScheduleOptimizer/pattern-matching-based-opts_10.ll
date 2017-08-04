; RUN: opt %loadPolly -polly-opt-isl -polly-invariant-load-hoisting=true \
; RUN: -polly-pattern-matching-based-opts=true \
; RUN: -polly-target-throughput-vector-fma=1 \
; RUN: -polly-target-latency-vector-fma=1 \
; RUN: -polly-codegen -polly-target-1st-cache-level-associativity=8 \
; RUN: -polly-target-2nd-cache-level-associativity=8 \
; RUN: -polly-target-1st-cache-level-size=32768 \
; RUN: -polly-target-vector-register-bitwidth=256 \
; RUN: -polly-target-2nd-cache-level-size=262144 -S < %s \
; RUN: | FileCheck %s
;
; This test case checks whether Polly generates second level alias metadata
; to distinguish the specific accesses in case of the ublas gemm kernel.
;
; CHECK: !13 = distinct !{!13, !0, !"second level alias metadata"}
; CHECK: !14 = distinct !{!14, !0, !"second level alias metadata"}
; CHECK: !15 = !{!3, !4, !5, !6, !7, !8, !13}
; CHECK: !16 = distinct !{!16, !0, !"second level alias metadata"}
; CHECK: !17 = !{!3, !4, !5, !6, !7, !8, !13, !14}
; CHECK: !18 = distinct !{!18, !0, !"second level alias metadata"}
; CHECK: !19 = !{!3, !4, !5, !6, !7, !8, !13, !14, !16}
;
target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-unknown"

define internal void @kernel_gemm(i32 %arg, i32 %arg1, i32 %arg2, double %arg3, double %arg4, [1056 x double]* %arg5, [1024 x double]* %arg6, [1056 x double]* %arg7) {
bb:
  br label %bb8

bb8:                                              ; preds = %bb29, %bb
  %tmp = phi i64 [ 0, %bb ], [ %tmp30, %bb29 ]
  br label %bb9

bb9:                                              ; preds = %bb26, %bb8
  %tmp10 = phi i64 [ 0, %bb8 ], [ %tmp27, %bb26 ]
  %tmp11 = getelementptr inbounds [1056 x double], [1056 x double]* %arg5, i64 %tmp, i64 %tmp10
  %tmp12 = load double, double* %tmp11, align 8
  %tmp13 = fmul double %tmp12, %arg4
  store double %tmp13, double* %tmp11, align 8
  br label %Copy_0

Copy_0:                                             ; preds = %Copy_0, %bb9
  %tmp15 = phi i64 [ 0, %bb9 ], [ %tmp24, %Copy_0 ]
  %tmp16 = getelementptr inbounds [1024 x double], [1024 x double]* %arg6, i64 %tmp, i64 %tmp15
  %tmp17 = load double, double* %tmp16, align 8
  %tmp18 = fmul double %tmp17, %arg3
  %tmp19 = getelementptr inbounds [1056 x double], [1056 x double]* %arg7, i64 %tmp15, i64 %tmp10
  %tmp20 = load double, double* %tmp19, align 8
  %tmp21 = fmul double %tmp18, %tmp20
  %tmp22 = load double, double* %tmp11, align 8
  %tmp23 = fadd double %tmp22, %tmp21
  store double %tmp23, double* %tmp11, align 8
  %tmp24 = add nuw nsw i64 %tmp15, 1
  %tmp25 = icmp ne i64 %tmp24, 1024
  br i1 %tmp25, label %Copy_0, label %bb26

bb26:                                             ; preds = %Copy_0
  %tmp27 = add nuw nsw i64 %tmp10, 1
  %tmp28 = icmp ne i64 %tmp27, 1056
  br i1 %tmp28, label %bb9, label %bb29

bb29:                                             ; preds = %bb26
  %tmp30 = add nuw nsw i64 %tmp, 1
  %tmp31 = icmp ne i64 %tmp30, 1056
  br i1 %tmp31, label %bb8, label %bb32

bb32:                                             ; preds = %bb29
  ret void
}

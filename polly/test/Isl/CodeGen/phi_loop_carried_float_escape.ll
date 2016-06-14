; RUN: opt %loadPolly -S \
; RUN: -polly-analyze-read-only-scalars=false -polly-codegen < %s | FileCheck %s

; RUN: opt %loadPolly -S \
; RUN: -polly-analyze-read-only-scalars=true -polly-codegen < %s | FileCheck %s
;
;    float f(float *A, int N) {
;      float tmp = 0;
;      for (int i = 0; i < N; i++)
;        tmp += A[i];
;      return tmp;
;    }

; CHECK-LABEL: polly.merge_new_and_old:
; CHECK-NEXT:    %tmp.0.merge = phi float [ %tmp.0.final_reload, %polly.exiting ], [ %tmp.0, %bb8 ]
; CHECK-NEXT:    br label %exit

; CHECK-LABEL: polly.start:
; CHECK-NEXT:    sext
; CHECK-NEXT:    store float 0.000000e+00, float* %tmp.0.phiops

; CHECK-LABEL: polly.exiting:
; CHECK-NEXT:    %tmp.0.final_reload = load float, float* %tmp.0.s2a
; CHECK-NEXT:    br label %polly.merge_new_and_old

; CHECK-LABEL: polly.stmt.bb1{{[0-9]*}}:
; CHECK-NEXT:    %tmp.0.phiops.reload[[R1:[0-9]*]] = load float, float* %tmp.0.phiops
; CHECK-:        store float %tmp.0.phiops.reload[[R1]], float* %tmp.0.s2a

; CHECK-LABEL: polly.stmt.bb4:
; CHECK:         %tmp.0.s2a.reload[[R3:[0-9]*]] = load float, float* %tmp.0.s2a
; CHECK:         %tmp[[R5:[0-9]*]]_p_scalar_ = load float, float* %scevgep, align 4, !alias.scope !0, !noalias !2
; CHECK:         %p_tmp[[R4:[0-9]*]] = fadd float %tmp.0.s2a.reload[[R3]], %tmp[[R5]]_p_scalar_
; CHECK:         store float %p_tmp[[R4]], float* %tmp.0.phiops

; CHECK-LABEL: polly.stmt.bb1{{[0-9]*}}:
; CHECK-NEXT:    %tmp.0.phiops.reload[[R2:[0-9]*]] = load float, float* %tmp.0.phiops
; CHECK:         store float %tmp.0.phiops.reload[[R2]], float* %tmp.0.s2a

target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"

define float @f(float* %A, i32 %N) {
bb:
  %tmp = sext i32 %N to i64
  br label %bb1

bb1:                                              ; preds = %bb4, %bb
  %indvars.iv = phi i64 [ %indvars.iv.next, %bb4 ], [ 0, %bb ]
  %tmp.0 = phi float [ 0.000000e+00, %bb ], [ %tmp7, %bb4 ]
  %tmp2 = icmp slt i64 %indvars.iv, %tmp
  br i1 %tmp2, label %bb3, label %bb8

bb3:                                              ; preds = %bb1
  br label %bb4

bb4:                                              ; preds = %bb3
  %tmp5 = getelementptr inbounds float, float* %A, i64 %indvars.iv
  %tmp6 = load float, float* %tmp5, align 4
  %tmp7 = fadd float %tmp.0, %tmp6
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  br label %bb1

bb8:                                              ; preds = %bb1
  br label %exit

exit:
  ret float %tmp.0
}

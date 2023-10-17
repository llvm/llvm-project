; RUN: opt %loadPolly -polly-codegen \
; RUN:     -polly-allow-nonaffine-branches -S -verify-dom-info \
; RUN:     < %s | FileCheck %s
;
;    void f(float *A) {
;      for (int i = 0; i < 1024; i++)
;        if (A[i] == A[i - 1])
;          A[i]++;
;       A[i]++;
;    }
;
;
; CHECK: polly.stmt.bb2:
; CHECK:   %[[offset:.*]] = shl nuw nsw i64 %polly.indvar, 2
; CHECK:   %scevgep[[R0:[0-9]*]] = getelementptr i8, ptr %A, i64 %[[offset]]
; CHECK:   %tmp3_p_scalar_ = load float, ptr %scevgep[[R0]], align 4, !alias.scope !0, !noalias !3
; CHECK:   %[[offset2:.*]] = shl nuw nsw i64 %polly.indvar, 2
; CHECK:   %scevgep[[R2:[0-9]*]] = getelementptr i8, ptr %scevgep{{[0-9]*}}, i64 %[[offset2]]
; CHECK:   %tmp6_p_scalar_ = load float, ptr %scevgep[[R2]], align 4, !alias.scope !0, !noalias !3
; CHECK:   %p_tmp7 = fcmp oeq float %tmp3_p_scalar_, %tmp6_p_scalar_
; CHECK:   br i1 %p_tmp7, label %polly.stmt.bb8, label %polly.stmt.bb12.[[R:[a-zA-Z_.0-9]*]]

; CHECK: polly.stmt.bb8:
; CHECK:   %[[offset3:.*]] = shl nuw nsw i64 %polly.indvar, 2
; CHECK:   %scevgep[[R3:[0-9]*]] = getelementptr i8, ptr %A, i64 %[[offset3]]
; CHECK:   %tmp10_p_scalar_ = load float, ptr %scevgep[[R3]], align 4, !alias.scope !0, !noalias !3
; CHECK:   %p_tmp11 = fadd float %tmp10_p_scalar_, 1.000000e+00
; CHECK:   store float %p_tmp11, ptr %scevgep[[R3]], align 4, !alias.scope !0, !noalias !3
; CHECK:   br label %polly.stmt.bb12.[[R]]

; CHECK: polly.stmt.bb12.[[R]]:
; CHECK:   br label %polly.stmt.bb12

; CHECK: polly.stmt.bb12:
; CHECK:   %[[offset4:.*]] = shl nuw nsw i64 %polly.indvar, 2
; CHECK:   %scevgep[[R4:[0-9]*]] = getelementptr i8, ptr %A, i64 %[[offset4]]
; CHECK:   %tmp10b_p_scalar_ = load float, ptr %scevgep[[R4]], align 4, !alias.scope !0, !noalias !3
; CHECK:   %p_tmp11b = fadd float %tmp10b_p_scalar_, 1.000000e+00
; CHECK:   store float %p_tmp11b, ptr %scevgep[[R4]], align 4, !alias.scope !0, !noalias !3
; CHECK:   %polly.indvar_next = add nsw i64 %polly.indvar, 1
; CHECK:   %polly.loop_cond = icmp sle i64 %polly.indvar_next, 1023
; CHECK:   br i1 %polly.loop_cond, label %polly.loop_header, label %polly.loop_exit

target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"

define void @f(ptr %A) {
bb:
  br label %bb1

bb1:                                              ; preds = %bb13, %bb
  %indvars.iv = phi i64 [ %indvars.iv.next, %bb13 ], [ 0, %bb ]
  %exitcond = icmp ne i64 %indvars.iv, 1024
  br i1 %exitcond, label %bb2, label %bb14

bb2:                                              ; preds = %bb1
  %tmp = getelementptr inbounds float, ptr %A, i64 %indvars.iv
  %tmp3 = load float, ptr %tmp, align 4
  %tmp4 = add nsw i64 %indvars.iv, -1
  %tmp5 = getelementptr inbounds float, ptr %A, i64 %tmp4
  %tmp6 = load float, ptr %tmp5, align 4
  %tmp7 = fcmp oeq float %tmp3, %tmp6
  br i1 %tmp7, label %bb8, label %bb12

bb8:                                              ; preds = %bb2
  %tmp9 = getelementptr inbounds float, ptr %A, i64 %indvars.iv
  %tmp10 = load float, ptr %tmp9, align 4
  %tmp11 = fadd float %tmp10, 1.000000e+00
  store float %tmp11, ptr %tmp9, align 4
  br label %bb12

bb12:                                             ; preds = %bb8, %bb2
  %tmp9b = getelementptr inbounds float, ptr %A, i64 %indvars.iv
  %tmp10b = load float, ptr %tmp9b, align 4
  %tmp11b = fadd float %tmp10b, 1.000000e+00
  store float %tmp11b, ptr %tmp9b, align 4
  br label %bb13

bb13:                                             ; preds = %bb12
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  br label %bb1

bb14:                                             ; preds = %bb1
  ret void
}

; RUN: opt -opaque-pointers=0 %loadPolly -polly-codegen -polly-codegen-generate-expressions=false \
; RUN:     -S < %s | FileCheck %s -check-prefix=SCEV
; RUN: opt -opaque-pointers=0 %loadPolly -polly-codegen -polly-codegen-generate-expressions=true \
; RUN:     -S < %s | FileCheck %s -check-prefix=ASTEXPR
;
;    void foo(float A[]) {
;      for (long i = 0; i < 100; i++)
;        A[i % 4] += 10;
;    }

; SCEV:      polly.stmt.bb2:                                   ; preds = %polly.loop_header
; SCEV-NEXT:   %0 = trunc i64 %polly.indvar to i2
; SCEV-NEXT:   %1 = zext i2 %0 to i64
; SCEV-NEXT:   %scevgep = getelementptr float, float* %A, i64 %1
; SCEV-NEXT:   %tmp4_p_scalar_ = load float, float* %scevgep, align 4, !alias.scope !0, !noalias !3
; SCEV-NEXT:   %p_tmp5 = fadd float %tmp4_p_scalar_, 1.000000e+01
; SCEV-NEXT:   store float %p_tmp5, float* %scevgep, align 4, !alias.scope !0, !noalias !3
; SCEV-NEXT:   %polly.indvar_next = add nsw i64 %polly.indvar, 1
; SCEV-NEXT:   %polly.loop_cond = icmp sle i64 %polly.indvar_next, 99
; SCEV-NEXT:   br i1 %polly.loop_cond, label %polly.loop_header, label %polly.loop_exit

; ASTEXPR: 	polly.stmt.bb2:                                   ; preds = %polly.loop_header
; ASTEXPR-NEXT:   %pexp.pdiv_r = urem i64 %polly.indvar, 4
; ASTEXPR-NEXT:   %polly.access.A = getelementptr float, float* %A, i64 %pexp.pdiv_r
; ASTEXPR-NEXT:   %tmp4_p_scalar_ = load float, float* %polly.access.A, align 4, !alias.scope !0, !noalias !3
; ASTEXPR-NEXT:   %p_tmp5 = fadd float %tmp4_p_scalar_, 1.000000e+01
; ASTEXPR-NEXT:   %pexp.pdiv_r1 = urem i64 %polly.indvar, 4
; ASTEXPR-NEXT:   %polly.access.A2 = getelementptr float, float* %A, i64 %pexp.pdiv_r1
; ASTEXPR-NEXT:   store float %p_tmp5, float* %polly.access.A2, align 4, !alias.scope !0, !noalias !3
; ASTEXPR-NEXT:   %polly.indvar_next = add nsw i64 %polly.indvar, 1
; ASTEXPR-NEXT:   %polly.loop_cond = icmp sle i64 %polly.indvar_next, 99
; ASTEXPR-NEXT:   br i1 %polly.loop_cond, label %polly.loop_header, label %polly.loop_exit

target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"

define void @foo(float* %A) {
bb:
  br label %bb1

bb1:                                              ; preds = %bb6, %bb
  %i.0 = phi i64 [ 0, %bb ], [ %tmp7, %bb6 ]
  %exitcond = icmp ne i64 %i.0, 100
  br i1 %exitcond, label %bb2, label %bb8

bb2:                                              ; preds = %bb1
  %tmp = srem i64 %i.0, 4
  %tmp3 = getelementptr inbounds float, float* %A, i64 %tmp
  %tmp4 = load float, float* %tmp3, align 4
  %tmp5 = fadd float %tmp4, 1.000000e+01
  store float %tmp5, float* %tmp3, align 4
  br label %bb6

bb6:                                              ; preds = %bb2
  %tmp7 = add nuw nsw i64 %i.0, 1
  br label %bb1

bb8:                                              ; preds = %bb1
  ret void
}

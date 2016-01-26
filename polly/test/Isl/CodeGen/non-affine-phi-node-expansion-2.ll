; RUN: opt %loadPolly -polly-codegen \
; RUN:     -S < %s | FileCheck %s
target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"


; CHECK: polly.stmt.bb3:                                   ; preds = %polly.stmt.bb3.entry
; CHECK:   %tmp6_p_scalar_ = load double, double* %arg1{{[0-9]*}}, !alias.scope !0, !noalias !2
; CHECK:   %p_tmp7 = fadd double 1.000000e+00, %tmp6_p_scalar_
; CHECK:   %p_tmp8 = fcmp olt double 1.400000e+01, %p_tmp7
; CHECK:   br i1 %p_tmp8, label %polly.stmt.bb9, label %polly.stmt.bb10

; CHECK: polly.stmt.bb9:                                   ; preds = %polly.stmt.bb3
; CHECK:   br label %polly.stmt.bb11.exit

; CHECK: polly.stmt.bb10:                                  ; preds = %polly.stmt.bb3
; CHECK:   br label %polly.stmt.bb11.exit

; CHECK: polly.stmt.bb11.exit:                             ; preds = %polly.stmt.bb10, %polly.stmt.bb9
; CHECK:   %polly.tmp12 = phi double [ 1.000000e+00, %polly.stmt.bb9 ], [ 2.000000e+00, %polly.stmt.bb10 ]
; CHECK:   store double %polly.tmp12, double* %tmp12.phiops

define void @hoge(i32 %arg, [1024 x double]* %arg1) {
bb:
  br label %bb2

bb2:                                              ; preds = %bb
  br label %bb3

bb3:                                              ; preds = %bb11, %bb2
  %tmp = phi i64 [ 0, %bb11 ], [ 0, %bb2 ]
  %tmp4 = icmp sgt i32 %arg, 0
  %tmp5 = getelementptr inbounds [1024 x double], [1024 x double]* %arg1, i64 0, i64 0
  %tmp6 = load double, double* %tmp5
  %tmp7 = fadd double 1.0, %tmp6
  %tmp8 = fcmp olt double 14.0, %tmp7
  br i1 %tmp8, label %bb9, label %bb10

bb9:                                              ; preds = %bb3
  br label %bb11

bb10:                                             ; preds = %bb3
  br label %bb11

bb11:                                             ; preds = %bb10, %bb9
  %tmp12 = phi double [ 1.0, %bb9 ], [ 2.0, %bb10 ]
  %tmp13 = getelementptr inbounds [1024 x double], [1024 x double]* %arg1, i64 %tmp, i64 0
  store double %tmp12, double* %tmp13
  %tmp14 = add nuw nsw i64 0, 1
  %tmp15 = trunc i64 %tmp14 to i32
  br i1 false, label %bb3, label %bb16

bb16:                                             ; preds = %bb11
  br label %bb17

bb17:                                             ; preds = %bb16
  ret void
}

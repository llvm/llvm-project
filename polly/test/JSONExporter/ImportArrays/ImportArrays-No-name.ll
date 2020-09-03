; RUN: not --crash opt %loadPolly -polly-scops -analyze -polly-import-jscop -polly-import-jscop-postfix=transformed < %s 2>&1 | FileCheck %s
;
; CHECK: Array has no key 'name'.
;
; Verify if the JSONImporter checks if the arrays have a key name 'name'.
;
;  for (i = 0; i < _PB_NI; i++)
;    for (j = 0; j < _PB_NJ; j++)
;      for (k = 0; k < _PB_NK; ++k)
;        B[i][j] = beta * A[i][k];
;
;

target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-unknown"

; Function Attrs: nounwind uwtable
define internal void @ia3(i32 %arg, i32 %arg1, i32 %arg2, double %arg3, double %beta, [1056 x double]* %A, [1024 x double]* %B, [1056 x double]* %arg7) #0 {
bb:
  br label %bb8

bb8:                                              ; preds = %bb
  br label %bb9

bb9:                                              ; preds = %bb23, %bb8
  %tmp = phi i64 [ 0, %bb8 ], [ %tmp24, %bb23 ]
  br label %bb10

bb10:                                             ; preds = %bb20, %bb9
  %tmp11 = phi i64 [ 0, %bb9 ], [ %tmp21, %bb20 ]
  br label %bb12

bb12:                                             ; preds = %bb12, %bb10
  %tmp13 = phi i64 [ 0, %bb10 ], [ %tmp18, %bb12 ]
  %tmp14 = getelementptr inbounds [1024 x double], [1024 x double]* %B, i64 %tmp, i64 %tmp13
  %tmp15 = load double, double* %tmp14, align 8
  %tmp16 = fmul double %tmp15, %beta
  %tmp17 = getelementptr inbounds [1056 x double], [1056 x double]* %A, i64 %tmp, i64 %tmp11
  store double %tmp16, double* %tmp17, align 8
  %tmp18 = add nuw nsw i64 %tmp13, 1
  %tmp19 = icmp ne i64 %tmp18, 1024
  br i1 %tmp19, label %bb12, label %bb20

bb20:                                             ; preds = %bb12
  %tmp21 = add nuw nsw i64 %tmp11, 1
  %tmp22 = icmp ne i64 %tmp21, 1056
  br i1 %tmp22, label %bb10, label %bb23

bb23:                                             ; preds = %bb20
  %tmp24 = add nuw nsw i64 %tmp, 1
  %tmp25 = icmp ne i64 %tmp24, 1056
  br i1 %tmp25, label %bb9, label %bb26

bb26:                                             ; preds = %bb23
  ret void
}

attributes #0 = { nounwind uwtable "target-cpu"="x86-64" "target-features"="+aes,+avx,+cmov,+cx16,+fxsr,+mmx,+pclmul,+popcnt,+sse,+sse2,+sse3,+sse4.1,+sse4.2,+ssse3,+x87,+xsave,+xsaveopt" }

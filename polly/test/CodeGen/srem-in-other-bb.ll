; RUN: opt %loadPolly -polly-codegen -S \
; RUN:     < %s | FileCheck %s
;
;    void pos(float *A, long n) {
;      for (long i = 0; i < 100; i++)
;        A[n % 42] += 1;
;    }
;
; CHECK:      polly.stmt.bb2:
; CHECK-NEXT:   %p_tmp = srem i64 %n, 42
; CHECK-NEXT:   store i64 %p_tmp, i64* %tmp.s2a
;
; CHECK:      polly.stmt.bb3:
; CHECK:        %tmp.s2a.reload = load i64, i64* %tmp.s2a
; CHECK:        %p_tmp3 = getelementptr inbounds float, float* %A, i64 %tmp.s2a.reload

define void @pos(float* %A, i64 %n) {
bb:
  br label %bb1

bb1:                                              ; preds = %bb6, %bb
  %i.0 = phi i64 [ 0, %bb ], [ %tmp7, %bb6 ]
  %exitcond = icmp ne i64 %i.0, 100
  br i1 %exitcond, label %bb2, label %bb8

bb2:                                              ; preds = %bb1
  %tmp = srem i64 %n, 42
  br label %bb3

bb3:
  %tmp3 = getelementptr inbounds float, float* %A, i64 %tmp
  %tmp4 = load float, float* %tmp3, align 4
  %tmp5 = fadd float %tmp4, 1.000000e+00
  store float %tmp5, float* %tmp3, align 4
  br label %bb6

bb6:                                              ; preds = %bb2
  %tmp7 = add nsw i64 %i.0, 1
  br label %bb1

bb8:                                              ; preds = %bb1
  ret void
}

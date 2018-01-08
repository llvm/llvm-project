; RUN: opt %loadPolly -polly-codegen-ppcg  -polly-acc-fail-on-verify-module-failure \
; RUN: -disable-output < %s

; Make sure that if -polly-acc-fail-on-verify-module-failure is on, we actually
; fail on an illegal module.

; REQUIRES: pollyacc, asserts
; XFAIL: *
;
;    void foo(long A[1024], long B[1024]) {
;      for (long i = 0; i < 1024; i++)
;        A[i] += (B[i] + (long)&B[i]);
;    }


; RUN: opt %loadPolly -polly-codegen-ppcg -S < %s 

target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"

define void @foo(i64* %A, i64* %B) {
bb:
  br label %bb1

bb1:                                              ; preds = %bb10, %bb
  %i.0 = phi i64 [ 0, %bb ], [ %tmp11, %bb10 ]
  %exitcond = icmp ne i64 %i.0, 1024
  br i1 %exitcond, label %bb2, label %bb12

bb2:                                              ; preds = %bb1
  %tmp = getelementptr inbounds i64, i64* %B, i64 %i.0
  %tmp3 = load i64, i64* %tmp, align 8
  %tmp4 = getelementptr inbounds i64, i64* %B, i64 %i.0
  %tmp5 = ptrtoint i64* %tmp4 to i64
  %tmp6 = add nsw i64 %tmp3, %tmp5
  %tmp7 = getelementptr inbounds i64, i64* %A, i64 %i.0
  %tmp8 = load i64, i64* %tmp7, align 8
  %tmp9 = add nsw i64 %tmp8, %tmp6
  store i64 %tmp9, i64* %tmp7, align 8
  br label %bb10

bb10:                                             ; preds = %bb2
  %tmp11 = add nuw nsw i64 %i.0, 1
  br label %bb1

bb12:                                             ; preds = %bb1
  ret void
}

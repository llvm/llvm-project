; RUN: opt %loadPolly -polly-codegen-ppcg -polly-acc-dump-code %s

; XFAIL: *

; REQUIRES: pollyacc

; This fails today with "type mismatch between callee prototype and arguments"

;    void foo(i120 A[], i120 b) {
;      for (long i = 0; i < 1024; i++)
;        A[i] += b;
;    }
;
target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"

define void @i120(i120* %A, i120 %b) {
bb:
  br label %bb1

bb1:                                              ; preds = %bb5, %bb
  %i.0 = phi i120 [ 0, %bb ], [ %tmp6, %bb5 ]
  %exitcond = icmp ne i120 %i.0, 1024
  br i1 %exitcond, label %bb2, label %bb7

bb2:                                              ; preds = %bb1
  %tmp = getelementptr inbounds i120, i120* %A, i120 %i.0
  %tmp3 = load i120, i120* %tmp, align 4
  %tmp4 = add i120 %tmp3, %b
  store i120 %tmp4, i120* %tmp, align 4
  br label %bb5

bb5:                                              ; preds = %bb2
  %tmp6 = add nuw nsw i120 %i.0, 1
  br label %bb1

bb7:                                              ; preds = %bb1
  ret void
}


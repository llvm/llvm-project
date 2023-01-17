; RUN: opt %loadPolly -polly-codegen-ppcg -polly-acc-dump-code -disable-output %s

; XFAIL: *

; REQUIRES: pollyacc, target=nvptx{{.*}}

; This fails today with "Promotion is not suitable for scalars of size larger than 64-bits"

;    void foo(i80 A[], i80 b) {
;      for (long i = 0; i < 1024; i++)
;        A[i] += b;
;    }
;
target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"

define void @i80(ptr %A, i80 %b) {
bb:
  br label %bb1

bb1:                                              ; preds = %bb5, %bb
  %i.0 = phi i80 [ 0, %bb ], [ %tmp6, %bb5 ]
  %exitcond = icmp ne i80 %i.0, 1024
  br i1 %exitcond, label %bb2, label %bb7

bb2:                                              ; preds = %bb1
  %tmp = getelementptr inbounds i80, ptr %A, i80 %i.0
  %tmp3 = load i80, ptr %tmp, align 4
  %tmp4 = add i80 %tmp3, %b
  store i80 %tmp4, ptr %tmp, align 4
  br label %bb5

bb5:                                              ; preds = %bb2
  %tmp6 = add nuw nsw i80 %i.0, 1
  br label %bb1

bb7:                                              ; preds = %bb1
  ret void
}


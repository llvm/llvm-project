; RUN: opt %loadPolly -polly-codegen-ppcg -polly-acc-dump-code -disable-output %s

; XFAIL: *

; REQUIRES: pollyacc, target=nvptx{{.*}}

; This fails today with "Promotion is not suitable for scalars of size larger than 64-bits"

;    void foo(i3000 A[], i3000 b) {
;      for (long i = 0; i < 1024; i++)
;        A[i] += b;
;    }
;
target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"

define void @i3000(ptr %A, i3000 %b) {
bb:
  br label %bb1

bb1:                                              ; preds = %bb5, %bb
  %i.0 = phi i3000 [ 0, %bb ], [ %tmp6, %bb5 ]
  %exitcond = icmp ne i3000 %i.0, 1024
  br i1 %exitcond, label %bb2, label %bb7

bb2:                                              ; preds = %bb1
  %tmp = getelementptr inbounds i3000, ptr %A, i3000 %i.0
  %tmp3 = load i3000, ptr %tmp, align 4
  %tmp4 = add i3000 %tmp3, %b
  store i3000 %tmp4, ptr %tmp, align 4
  br label %bb5

bb5:                                              ; preds = %bb2
  %tmp6 = add nuw nsw i3000 %i.0, 1
  br label %bb1

bb7:                                              ; preds = %bb1
  ret void
}
